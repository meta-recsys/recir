# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass

from torch import fx

# Umbrella node.meta key: holds a state-layout spec describing how a placeholder's
# host tensor maps to device buffers. Read by the XLA lowering (in/out specs) and
# by XLADeviceStateManager (host<->device I/O). Kept open so future strategies
# (batch inputs, replicated, ...) can carry their own spec variants.
STATE_LAYOUT_META_KEY = "recir_state_layout"


@dataclass(frozen=True)
class FSDPWeightShardSpec:
    """Layout of an FSDP-sharded weight placeholder.

    With D devices/process and P processes, N = P*D. The graph is traced with the
    weight sharded over device_mesh(N) (so the baked all-gather has factor N), but
    the canonical host store is the /P per-process shard, re-split into D device
    buffers at runtime.
    """

    shard_dim: int
    device_world_size: int  # N = P*D  (baked all-gather factor)
    process_world_size: int  # P        (canonical host-store divisor; full when P==1)

    def __post_init__(self) -> None:
        """Validate layout invariants at construction time."""
        if self.device_world_size <= 0:
            raise ValueError(
                f"device_world_size (N) must be > 0, got {self.device_world_size}"
            )
        if self.process_world_size <= 0:
            raise ValueError(
                f"process_world_size (P) must be > 0, got {self.process_world_size}"
            )
        if self.device_world_size % self.process_world_size != 0:
            raise ValueError(
                "device_world_size (N) must be divisible by "
                f"process_world_size (P): {self.device_world_size} % "
                f"{self.process_world_size} != 0"
            )
        if self.shard_dim != 0:
            raise ValueError(
                "shard_dim must be 0: only Shard(0) is currently supported "
                f"(got shard_dim={self.shard_dim}); non-zero shard_dim is a "
                "future extension requiring updates to the XLA lowering and "
                "device-state consumers."
            )


def set_state_layout(node: fx.Node, spec: FSDPWeightShardSpec) -> None:
    """Attach a state-layout spec to a placeholder node's meta."""
    node.meta[STATE_LAYOUT_META_KEY] = spec


def get_state_layout(node: fx.Node) -> FSDPWeightShardSpec | None:
    """Return the state-layout spec on a node, or None if absent."""
    return node.meta.get(STATE_LAYOUT_META_KEY)


def copy_state_layout(src: fx.Node, dst: fx.Node) -> None:
    """Propagate the state-layout spec when a pass replaces a node (risk #1 in the
    design: passes that recreate placeholders must carry the spec forward)."""
    spec = src.meta.get(STATE_LAYOUT_META_KEY)
    if spec is not None:
        dst.meta[STATE_LAYOUT_META_KEY] = spec
