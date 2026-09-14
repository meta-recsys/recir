# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""torch_tpu backend: lower recir FX graphs onto the torch_tpu runtime.

Unlike the XLA backends (which lower via torchax/JAX), this backend lowers each
compatible subgraph through torch_tpu's StableHLO/PJRT stack and carries no JAX
dependency. It is single-controller-agnostic: ``_is_xla_backend`` stays False
for ``BackendType.TORCHTPU`` so multi-chip uses real DeviceMeshes + ``tpu_dist``
process groups (torchrun), not the XLA fake-mesh path.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch
from recir.backends.base import Backend
from recir.lowering.device_state import DeviceStateManager
from recir.lowering.torchtpu_lower import partial_lower_torchtpu
from torch import fx, nn

if TYPE_CHECKING:
    from recir.distributed.parallel_mode import ParallelMode


class TorchTPUBackend(Backend):
    """Lower compatible subgraphs to TPU via torch_tpu (StableHLO/PJRT).

    Accepts non-GSPMD parallel modes only — torch_tpu has no GSPMD
    auto-partitioner; multi-chip is expressed as DataParallel/HybridParallel
    with explicit collectives executed eagerly on ``ProcessGroupTpu``.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @property
    def uses_iter_graph(self) -> bool:
        return False

    def dispatch(self, gm: fx.GraphModule) -> nn.Module:
        return partial_lower_torchtpu(gm)

    def device_state_manager(self) -> DeviceStateManager:
        # State lives on the tpu device (model + lifted params are traced on
        # tpu). The base manager's commit_step_state copies returned (tpu) state
        # back into the resident state tensors by index, matching recir's
        # state-as-output convention; per-subgraph device placement also happens
        # inside _TorchTPUModule.
        return DeviceStateManager(torch.device("tpu"))

    def validate_with(self, parallel_mode: "ParallelMode") -> None:
        from recir.distributed.parallel_mode import (
            DataParallel,
            GSPMDParallel,
            HybridParallel,
        )

        if isinstance(parallel_mode, GSPMDParallel):
            raise ValueError(
                "TorchTPUBackend is incompatible with GSPMDParallel (torch_tpu "
                "has no GSPMD); use DataParallel or HybridParallel."
            )
        if not isinstance(parallel_mode, (DataParallel, HybridParallel)):
            raise ValueError(
                f"TorchTPUBackend requires DataParallel or HybridParallel, got "
                f"{type(parallel_mode).__name__}"
            )
