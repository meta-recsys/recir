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
            self.device_state = updated_flat[:num_state]
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
                if isinstance(updated, torch.Tensor):
                    if updated.data_ptr() != state_tensor.data_ptr():
                        state_tensor.copy_(updated)
                else:
                    state_tensor.copy_(_jax_to_torch(updated, self.torch_device))

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
