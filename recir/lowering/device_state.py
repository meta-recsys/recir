# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from typing import Any, List, Optional

import torch
from torch import fx


_CPU_DEVICE = torch.device("cpu")


class DeviceStateManager:
    """Manages device-resident state between train steps."""

    def __init__(self, torch_device: torch.device = _CPU_DEVICE) -> None:
        self.device_state: Optional[List[Any]] = None
        self._used_fast_path: bool = False
        self.torch_device: torch.device = torch_device

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

    def sync_to_torch(
        self,
        flat_state: List[torch.Tensor],
    ) -> None:
        pass
