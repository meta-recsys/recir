# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Shared test helpers for the recir v7.1 public-API smoke tests.

The three ``test_backend_kwarg``, ``test_mark_boundary_survives`` and
``test_trace_compile_split`` modules all need a minimal ``ParallelMode`` +
``Backend`` pair so they can drive ``recir.trace`` / ``recir.compile_from_gm``
on CPU without depending on Inductor / Triton / XLA / NCCL.  Keeping the
implementations here removes the previous triplicate copies and makes future
contract tweaks land in one place.
"""

import operator
from typing import Any, Dict, Set

import torch
import torch.nn as nn
from recir.backends.base import Backend
from recir.distributed.parallel_mode import ParallelMode
from recir.pipeline import mark_pipeline_region
from recir.pipeline_strategies import PipelineRegion
from torch import fx
from torch.fx.experimental.proxy_tensor import make_fx


class _PassthroughParallel(ParallelMode):
    """No-op single-rank parallel mode for CPU tests.

    Returns the gm unchanged from ``partition``; declares 1 rank / 1
    process.  Mirrors the :class:`ParallelMode` contract minimally so
    :func:`recir.compile_from_gm` can drive a CPU-only pipeline without
    requiring ``dist.init_process_group``.
    """

    input_batch_dim: int = 0

    @property
    def num_ranks(self) -> int:
        return 1

    @property
    def num_processes(self) -> int:
        return 1

    def partition(
        self,
        gm: fx.GraphModule,
        *,
        params: Dict[str, nn.Parameter],
        buffers: Dict[str, torch.Tensor],
        named_states: Dict[str, Dict[str, Any]],
    ) -> fx.GraphModule:
        return gm

    def shard_params(
        self,
        model: nn.Module,
        params_and_buffers: Dict[str, Any],
        opt: Any,
    ) -> Set[str]:
        return set()

    def apply_state_layout(
        self,
        gm: fx.GraphModule,
        params_and_buffers: Dict[str, Any],
        named_states: Dict[str, Any],
    ) -> None:
        """No-op: this passthrough mode produces no recir_state_layout meta."""

    def transform_and_compile(self, gm: fx.GraphModule) -> fx.GraphModule:
        return gm


class _NoopBackend(Backend):
    """A no-op backend whose ``dispatch`` returns the gm unchanged.

    Used by the v7.1 smoke tests so they exercise the recir pipeline
    without depending on Inductor / Triton / XLA availability.  No
    optimization passes are applied by default; subclasses can override
    :meth:`default_optimize_passes` per-test for narrower coverage.
    """

    @property
    def uses_iter_graph(self) -> bool:
        return False

    def dispatch(self, gm: fx.GraphModule) -> nn.Module:
        return gm


class _FourStageModule(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        preprocessed = value + 1
        started = preprocessed * 2
        continued = started - 3
        return continued / 2


def _region_graph() -> fx.GraphModule:
    """A four-op graph whose first three ops are marked as SDD pipeline regions."""
    gm = fx.symbolic_trace(_FourStageModule())
    regions = (
        (operator.add, PipelineRegion.PREPROCESS),
        (operator.mul, PipelineRegion.ROUTE_START),
        (operator.sub, PipelineRegion.ROUTE_CONTINUE),
    )
    for target, region in regions:
        node = next(
            node
            for node in gm.graph.nodes
            if node.op == "call_function" and node.target is target
        )
        mark_pipeline_region((node,), region)
    return gm


def _twelve_ops(x: torch.Tensor) -> torch.Tensor:
    for _ in range(6):
        x = torch.relu(x * 2)
    return x


def _graph() -> fx.GraphModule:
    """A traced graph of twelve elementwise ops, which partial lowering carves as one subgraph."""
    return make_fx(_twelve_ops)(torch.randn(4))
