# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from typing import Any, List

import torch
from recir.lowering.device_state import DeviceStateManager
from recir.lowering.xla_lower import _jax_to_torch, _XLAModule


class XLADeviceStateManager(DeviceStateManager):
    @staticmethod
    def _set_keep_jax_output(gm: torch.nn.Module, value: bool) -> None:
        for m in gm.modules():
            if isinstance(m, _XLAModule):
                m.keep_jax_output = value

    def prepare_inputs(
        self,
        gm: torch.fx.GraphModule,
        flat_state: List[torch.Tensor],
        user_inps: List[Any],
        last_train_step: bool,
    ) -> List[Any]:
        self._used_fast_path = self.device_state is not None and not last_train_step
        if self._used_fast_path:
            device_state = self.device_state
            assert device_state is not None
            self._set_keep_jax_output(gm, True)
            return device_state + user_inps
        else:
            self._set_keep_jax_output(gm, False)
            return flat_state + user_inps

    @staticmethod
    def _to_local_shard(tensor: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        """Narrow a global tensor back to the local shard shape.

        After GSPMD execution with replicated output shardings,
        _jax_to_torch produces full global tensors. This narrows dim 0
        to match the local shard size from shard_params, so that
        device_state entries have the same shapes as flat_state.
        """
        if tensor.shape == target_shape:
            return tensor
        if (
            tensor.ndim >= 1
            and target_shape[0] < tensor.shape[0]
            and tensor.shape[1:] == target_shape[1:]
        ):
            return tensor.narrow(0, 0, target_shape[0]).contiguous()
        return tensor

    def process_outputs(
        self,
        flat_state: List[torch.Tensor],
        output: Any,
        updated_flat: List[Any],
        last_train_step: bool,
    ) -> torch.Tensor:
        num_state = len(flat_state)
        if not self._used_fast_path:
            self._sync_flat_state(flat_state, updated_flat)
        if last_train_step:
            self.device_state = None
        else:
            # Re-shard outputs to match flat_state shapes (FSDP fix).
            # Step 1 outputs are global (replicated out_shardings), but
            # step 2 needs local shards to match _jit_gspmd expectations.
            device_state = []
            for i in range(min(num_state, len(updated_flat))):
                val = updated_flat[i]
                if not isinstance(val, torch.Tensor):
                    val = _jax_to_torch(val, self.torch_device)
                if i < len(flat_state):
                    val = self._to_local_shard(val, flat_state[i].shape)
                device_state.append(val)
            self.device_state = device_state
        if not isinstance(output, torch.Tensor) and output is not None:
            output = _jax_to_torch(output, self.torch_device)
        return output

    def _sync_flat_state(
        self,
        flat_state: List[torch.Tensor],
        updated_flat: List[Any],
    ) -> None:
        """Copy graph outputs back to flat_state tensors."""
        for i, state_tensor in enumerate(flat_state):
            if i < len(updated_flat):
                updated = updated_flat[i]
                if not isinstance(updated, torch.Tensor):
                    updated = _jax_to_torch(updated, self.torch_device)
                if updated.data_ptr() != state_tensor.data_ptr():
                    # For FSDP, output may be global while flat_state is
                    # a local shard. Narrow to match before copying.
                    if updated.shape != state_tensor.shape:
                        updated = self._to_local_shard(updated, state_tensor.shape)
                    if updated.shape == state_tensor.shape:
                        state_tensor.copy_(updated)

    def sync_to_torch(
        self,
        flat_state: List[torch.Tensor],
    ) -> None:
        device_state = self.device_state
        if device_state is None:
            return
        with torch.no_grad():
            for i, jax_val in enumerate(device_state):
                if i < len(flat_state):
                    flat_state[i].copy_(_jax_to_torch(jax_val, self.torch_device))
