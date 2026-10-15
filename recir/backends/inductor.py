# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Inductor (Triton/CUDA) backend implementation."""

from __future__ import annotations

from typing import Any, Sequence, TYPE_CHECKING

from recir.backends.base import Backend, GraphPatch
from recir.lowering.partial_lower import partial_lower
from recir.passes import remove_copy_from_optimizer
from recir.passes.iter_graph_module import IterGraphModule
from recir.pipeline import PipelinedGraphModule, SteadyStateGraphModule
from torch import fx, nn

if TYPE_CHECKING:
    from recir.distributed.parallel_mode import ParallelMode


class InductorBackend(Backend):
    """Lower CUDA-compatible subgraphs to Inductor (Triton kernels).

    The optimizer-copy removal pass operates on ``IterGraphModule``.
    Communication scheduling is left to Inductor because pre-AOT graph edits
    are discarded when ``compile_fx`` retraces the lowered subgraphs.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @classmethod
    def default_optimize_passes(cls) -> Sequence[GraphPatch]:
        # Communication scheduling is configured inside Inductor lowering.
        return (remove_copy_from_optimizer,)

    @property
    def uses_iter_graph(self) -> bool:
        return True

    def dispatch(
        self,
        gm: (
            fx.GraphModule
            | IterGraphModule
            | PipelinedGraphModule
            | SteadyStateGraphModule
        ),
    ) -> nn.Module:
        if isinstance(gm, SteadyStateGraphModule):
            return gm.lower_steady(partial_lower)
        if isinstance(gm, PipelinedGraphModule):
            return gm.lower_stages(partial_lower)
        if isinstance(gm, IterGraphModule):
            for sub_name in ("setup_gm", "main_gm", "cleanup_gm"):
                sub_gm = getattr(gm, sub_name)
                setattr(gm, sub_name, partial_lower(sub_gm))
            return gm
        return partial_lower(gm)

    def validate_with(self, parallel_mode: "ParallelMode") -> None:
        from recir.distributed.parallel_mode import GSPMDParallel

        if isinstance(parallel_mode, GSPMDParallel):
            raise ValueError(
                "InductorBackend is incompatible with GSPMDParallel (GSPMD is an "
                "XLA-SPMD feature with no CUDA/Inductor equivalent); use "
                "DataParallel/HybridParallel, or XLAGSPMDBackend on XLA."
            )
