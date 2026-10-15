# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Sequence

import torch
from torch import fx


_CPU_DEVICE = torch.device("cpu")


class DeviceStateManager:
    """Manages device-resident state between train steps."""

    def __init__(self, torch_device: torch.device = _CPU_DEVICE) -> None:
        self._device_state: Optional[List[Any]] = None
        self._used_fast_path: bool = False
        self.torch_device: torch.device = torch_device
        # FQNs (fully-qualified names) aligned 1:1 with flat_state, populated at
        # compile time. Lets copy_state_to_host skip specific state entries by
        # name (e.g. large device-resident embedding tables) rather than by a
        # size heuristic. None until the compiler attaches it.
        self.state_fqns: Optional[List[str]] = None
        # When True, ``defers_output`` says yes and the runner hands the caller the
        # raw (JAX) step output instead of materializing it every step. This lets a
        # training loop defer the output device->host pull to a cadence so JAX can
        # pipeline (the host runs ahead, overlapping step i+1's H2D/dispatch with
        # step i's compute). The caller is then responsible for calling
        # ``materialize_output`` when it actually needs the values (e.g. metric
        # logging). No effect for the base manager (whose outputs are already
        # torch); honored by XLADeviceStateManager.
        self.keep_output_jax: bool = False

    @property
    def device_state(self) -> Optional[List[Any]]:
        return self._device_state

    @device_state.setter
    def device_state(self, value: Optional[List[Any]]) -> None:
        self._device_state = value

    def share_device_state(
        self,
        store: Dict[str, Any],
        compiled: torch.nn.Module,
        flat_state: List[torch.Tensor],
    ) -> None:
        """Share resident state with the other graphs compiled together with
        this one (``recir.api.compile_together``). No-op here: these graphs
        read and write the module tensors directly."""
        return

    def prepare_inputs(
        self,
        gm: fx.GraphModule,
        flat_state: List[torch.Tensor],
        user_inps: List[Any],
        last_train_step: bool,
    ) -> List[Any]:
        return flat_state + user_inps

    def invalidate_device_state(self) -> None:
        """Discard device-resident state owned by this manager."""
        self.device_state = None

    def commit_step_state(
        self,
        flat_state: List[torch.Tensor],
        updated_flat: List[Any],
        last_train_step: bool,
    ) -> None:
        """Make the step's state outputs the resident state.

        The state half of the runner's per-step epilogue. Says nothing about the
        step's output -- that is ``defers_output`` / ``materialize_output``.
        """
        for i, state_tensor in enumerate(flat_state):
            if i < len(updated_flat):
                updated = updated_flat[i]
                assert isinstance(updated, torch.Tensor)
                if updated.data_ptr() != state_tensor.data_ptr():
                    state_tensor.copy_(updated)

    def defers_output(self, last_train_step: bool) -> bool:
        """Whether the caller asked to pull the step output itself, later.

        The last step always materializes, so a caller that stops training holds
        values rather than a handle it can no longer cash.
        """
        return self.keep_output_jax and not last_train_step

    def materialize_output(self, output: Any) -> Any:
        """Convert a (possibly deferred) step output to torch. No-op for the base
        manager, whose outputs are already torch. See ``keep_output_jax``."""
        return output

    def copy_state_to_host(
        self,
        flat_state: List[torch.Tensor],
        skip_fqns: Optional[Sequence[str]] = None,
    ) -> None:
        pass
