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

    @staticmethod
    def _convert_flat_state_to_jax(
        flat_state: List[torch.Tensor],
    ) -> List[Any]:
        """Convert flat_state tensors from PyTorch to JAX arrays on TPU.

        For inference, state tensors (model weights) are passed
        through the parent FX graph:

        They enter _XLAModule as inputs but are not returned as outputs.

        This means process_outputs can never capture them as JAX arrays —
        they always remain whatever type was in flat_inps, i.e. torch.Tensor.

        Converting them here, once, and caching in
        device_state ensures _XLAModule._t2j_safe / _to_sharded_jax sees
        jax.Array on subsequent calls and skips the per-tensor dispatch.
        """
        import jax  # @manual
        from recir.torchax.ops.mappings import t2j  # @manual

        jax_device = jax.local_devices()[0]
        jax_state: List[Any] = []
        for t in flat_state:
            if isinstance(t, torch.Tensor):
                jax_state.append(jax.device_put(t2j(t), jax_device))
            else:
                jax_state.append(t)
        return jax_state

    def prepare_inputs(
        self,
        gm: torch.fx.GraphModule,
        flat_state: List[torch.Tensor],
        user_inps: List[Any],
        last_train_step: bool,
    ) -> List[Any]:
        # First call (device_state is None) and not the last training step:
        # convert state tensors to JAX arrays once and cache. On subsequent
        # calls, _XLAModule._t2j_safe / _to_sharded_jax sees jax.Array and
        # returns immediately, skipping per-tensor dispatch.
        # For training, process_outputs overwrites device_state with updated
        # weights from the graph output after each step, so this initial
        # conversion is overwritten and does not affect correctness.
        if self.device_state is None and not last_train_step:
            self.device_state = self._convert_flat_state_to_jax(flat_state)
        device_state = self.device_state
        self._used_fast_path = not last_train_step
        if self._used_fast_path:
            # Only the fast path consumes the cached JAX device_state. The slow
            # path (last_train_step) returns flat_state and may legitimately have
            # device_state is None — process_outputs resets it at each last step.
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
                        assert updated.shape == state_tensor.shape
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
