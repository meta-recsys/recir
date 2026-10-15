# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""XLA backend implementations (GSPMD, ShardMap, Dynamo bridge)."""

from __future__ import annotations

import functools
from typing import Any, ClassVar, Optional, Sequence, TYPE_CHECKING

from recir.backends.base import Backend, GraphPatch
from recir.backends.types import BackendType
from recir.lowering.device_state import DeviceStateManager
from recir.lowering.xla_device_state import XLADeviceStateManager
from recir.lowering.xla_lower import partial_lower_xla
from recir.passes.fsdp_collectives import (
    bucket_grad_reduce_scatters as _bucket_grad_reduce_scatters,
    merge_backward_gathers as _merge_backward_gathers,
    serialize_backward_gathers as _serialize_backward_gathers,
)
from torch import fx, nn

if TYPE_CHECKING:
    from recir.distributed.parallel_mode import ParallelMode


class _XLABackendBase(Backend):
    """Shared logic for the three XLA-flavored backends.

    Each subclass picks the appropriate :class:`BackendType` tag and is
    expected to override :meth:`validate_with`.
    """

    # pyre-ignore[13]: Initialized in subclasses.
    _backend_type: ClassVar[BackendType]

    def __init__(self, *, dump_hlo: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._dump_hlo = dump_hlo

    @property
    def uses_iter_graph(self) -> bool:
        return False

    def dispatch(self, gm: fx.GraphModule) -> nn.Module:
        return partial_lower_xla(
            gm,
            backend_type=self._backend_type,
            capture_hlo_text=self._dump_hlo,
        )

    def device_state_manager(self) -> DeviceStateManager:
        return XLADeviceStateManager()


class XLAGSPMDBackend(_XLABackendBase):
    """XLA backend using GSPMD auto-parallelism.

    Requires a :class:`GSPMDParallel` parallel mode.  XLA performs all of the
    graph-level optimization passes natively, so :meth:`default_optimize_passes`
    returns an empty tuple.
    """

    _backend_type: ClassVar[BackendType] = BackendType.XLA_GSPMD

    def __init__(self, *, dump_hlo: bool = False, **kwargs: Any) -> None:
        super().__init__(dump_hlo=dump_hlo, **kwargs)
        self._fsdp_axis_size: Optional[int] = None

    def dispatch(self, gm: fx.GraphModule) -> nn.Module:
        if self._fsdp_axis_size is not None:
            gm.meta["gspmd_fsdp_axis_size"] = self._fsdp_axis_size
        return super().dispatch(gm)

    def validate_with(self, parallel_mode: "ParallelMode") -> None:
        from recir.distributed.parallel_mode import GSPMDParallel

        if not isinstance(parallel_mode, GSPMDParallel):
            raise ValueError(
                f"XLAGSPMDBackend requires GSPMDParallel, got "
                f"{type(parallel_mode).__name__}"
            )
        self._fsdp_axis_size = parallel_mode.fsdp_axis_size
        if self.uses_iter_graph:
            raise ValueError(
                "uses_iter_graph=True is incompatible with GSPMD parallel mode. "
                "GSPMD inflates input shapes by num_processes before tracing, "
                "baking global literals into the FX graph that IterGraphModule's "
                "unlowered setup_gm/cleanup_gm cannot handle at per-host runtime. "
                "Set uses_iter_graph=False (now the default) or "
                "use a non-GSPMD parallel mode."
            )


class XLAShardMapBackend(_XLABackendBase):
    """XLA backend that maps ``_c10d_functional`` collectives to JAX via shard_map.

    Requires an explicit-collective parallel mode (DataParallel or
    HybridParallel — anything that retains the collectives in the FX graph
    instead of relying on GSPMD).

    The FSDP shard_map collective optimizations in
    :mod:`recir.passes.fsdp_collectives` are wired into stage-6
    ``optimize_passes`` in a fixed order (merge, then bucket, then serialize).
    The redundant-backward-gather merge is always applied (bit-exact -- it only
    removes duplicate collective volume); reduce-scatter bucketing and all-gather
    in-flight limiting are opt-in.

    Args:
        bucket_reduce_scatters: fuse consecutive per-weight grad reduce_scatters
            into fewer, larger collectives.
        max_inflight_all_gathers: cap how many backward full-weight all-gathers
            may be in flight before forcing a wait (via an optimization barrier),
            bounding co-resident peak. ``0`` (default) = unbounded (no limiting);
            ``1`` = strict serial (lowest peak, least overlap); larger = more
            compute/collective overlap at higher peak.
        all_gather_min_gib: when ``max_inflight_all_gathers`` >= 1, only limit
            gathers whose full weight is >= this many GiB (``0`` = all); the
            co-held base is ~all model weights, so bounding just the largest
            weights drops peak with far fewer barriers.
        optimize_passes: explicit stage-6 pass sequence that fully replaces the
            flag-derived passes above.  ``None`` (default) builds the sequence
            from the flags (always including merge); pass an explicit list (e.g.
            ``[]``) to fully control the passes.

    The ``max_inflight_all_gathers`` / ``all_gather_min_gib`` values 2 / 0.15 are
    the tuned optimum from the Gemma-4-31B @1536/128ch benchmark and reproduce the
    published number.
    """

    _backend_type: ClassVar[BackendType] = BackendType.XLA_SHARD_MAP

    def __init__(
        self,
        *,
        bucket_reduce_scatters: bool = False,
        max_inflight_all_gathers: int = 0,
        all_gather_min_gib: float = 0.15,
        optimize_passes: Optional[Sequence[GraphPatch]] = None,
        **kwargs: Any,
    ) -> None:
        if optimize_passes is None:
            # The redundant-backward-gather merge is bit-exact and only removes
            # duplicate collective volume, so it is always applied.
            passes: list[GraphPatch] = [_merge_backward_gathers]
            if bucket_reduce_scatters:
                passes.append(_bucket_grad_reduce_scatters)
            if max_inflight_all_gathers >= 1:
                passes.append(
                    functools.partial(
                        _serialize_backward_gathers,
                        window=max_inflight_all_gathers,
                        min_gib=all_gather_min_gib,
                    )
                )
            optimize_passes = tuple(passes)
        super().__init__(optimize_passes=optimize_passes, **kwargs)

    def validate_with(self, parallel_mode: "ParallelMode") -> None:
        from recir.distributed.parallel_mode import (
            DataParallel,
            GSPMDParallel,
            HybridParallel,
        )

        if isinstance(parallel_mode, GSPMDParallel):
            raise ValueError(
                "XLAShardMapBackend is incompatible with GSPMDParallel; "
                "use XLAGSPMDBackend instead."
            )
        if not isinstance(parallel_mode, (DataParallel, HybridParallel)):
            raise ValueError(
                f"XLAShardMapBackend requires DataParallel or HybridParallel, "
                f"got {type(parallel_mode).__name__}"
            )


class XLADynamoBridgeBackend(_XLABackendBase):
    """XLA backend using ``torch_xla.core.dynamo_bridge`` for lazy-tensor lowering.

    Accepts any non-GSPMD parallel mode.  The dynamo bridge handles its own
    sharding via the underlying lazy-tensor stack.
    """

    _backend_type: ClassVar[BackendType] = BackendType.XLA_DYNAMO_BRIDGE

    def validate_with(self, parallel_mode: "ParallelMode") -> None:
        from recir.distributed.parallel_mode import GSPMDParallel

        if isinstance(parallel_mode, GSPMDParallel):
            raise ValueError(
                "XLADynamoBridgeBackend is incompatible with GSPMDParallel; "
                "use XLAGSPMDBackend instead."
            )
