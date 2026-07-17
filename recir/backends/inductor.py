# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Inductor (Triton/CUDA) backend implementation."""

from __future__ import annotations

from typing import Any, Sequence, TYPE_CHECKING

from recir.backends.base import Backend, GraphPatch
from recir.lowering.partial_lower import partial_lower
from recir.passes import (
    comm_fusion_with_concat,
    remove_copy_from_optimizer,
    schedule_comm_wait,
)
from torch import fx, nn

if TYPE_CHECKING:
    from recir.distributed.parallel_mode import ParallelMode


class InductorBackend(Backend):
    """Lower CUDA-compatible subgraphs to Inductor (Triton kernels).

    Defaults to a stable set of three IterGraph-based optimization passes:
    communication fusion via concat, communication-wait scheduling, and the
    optimizer copy-removal pass.  Pass an explicit ``optimize_passes=`` (e.g.
    ``[]``) to replace this set wholesale.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @classmethod
    def default_optimize_passes(cls) -> Sequence[GraphPatch]:
        return (
            comm_fusion_with_concat,
            schedule_comm_wait,
            remove_copy_from_optimizer,
        )

    @property
    def uses_iter_graph(self) -> bool:
        return True

    def dispatch(self, gm: fx.GraphModule) -> nn.Module:
        return partial_lower(gm)

    def validate_with(self, parallel_mode: "ParallelMode") -> None:
        from recir.distributed.parallel_mode import GSPMDParallel

        if isinstance(parallel_mode, GSPMDParallel):
            raise ValueError(
                "InductorBackend is incompatible with GSPMDParallel (GSPMD is an "
                "XLA-SPMD feature with no CUDA/Inductor equivalent); use "
                "DataParallel/HybridParallel, or XLAGSPMDBackend on XLA."
            )
