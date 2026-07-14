# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from typing import Any, List, Optional, Sequence

import torch
from torch import fx


_CPU_DEVICE = torch.device("cpu")


class DeviceStateManager:
    """Manages device-resident state between train steps."""

    def __init__(self, torch_device: torch.device = _CPU_DEVICE) -> None:
        self.device_state: Optional[List[Any]] = None
        self._used_fast_path: bool = False
        self.torch_device: torch.device = torch_device
        # FQNs (fully-qualified names) aligned 1:1 with flat_state, populated at
        # compile time. Lets sync_to_torch skip specific state entries by name
        # (e.g. large device-resident embedding tables) rather than by a size
        # heuristic. None until the compiler attaches it.
        self.state_fqns: Optional[List[str]] = None
        # When True, process_outputs returns the raw (JAX) step output instead of
        # materializing it to torch every step. This lets a training loop defer the
        # output device->host pull to a cadence so JAX can pipeline (the host runs
        # ahead, overlapping step i+1's H2D/dispatch with step i's compute). The
        # caller is responsible for materializing the output when it actually needs
        # the values (e.g. metric logging). No effect for the base manager (whose
        # outputs are already torch); honored by XLADeviceStateManager.
        self.keep_output_jax: bool = False

    def prepare_inputs(
        self,
        gm: fx.GraphModule,
        flat_state: List[torch.Tensor],
        user_inps: List[Any],
        last_train_step: bool,
    ) -> List[Any]:
        return flat_state + user_inps

    def process_outputs(
        self,
        flat_state: List[torch.Tensor],
        output: Any,
        updated_flat: List[Any],
        last_train_step: bool,
    ) -> torch.Tensor:
        for i, state_tensor in enumerate(flat_state):
            if i < len(updated_flat):
                updated = updated_flat[i]
                assert isinstance(updated, torch.Tensor)
                if updated.data_ptr() != state_tensor.data_ptr():
                    state_tensor.copy_(updated)
        return output

    def materialize_output(self, output: Any) -> Any:
        """Convert a (possibly deferred) step output to torch. No-op for the base
        manager, whose outputs are already torch. See ``keep_output_jax``."""
        return output

    def sync_to_torch(
        self,
        flat_state: List[torch.Tensor],
        skip_fqns: Optional[Sequence[str]] = None,
    ) -> None:
        pass
