# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""DLRMv3 bindings for the ``recir.embedding`` TPU embedding backends.

The backends are model-agnostic; this module supplies where DLRMv3 keeps its
embeddings (``DlrmHSTU._embedding_collection``) and the static per-feature
sizes that the token-budget layout makes known up front.

  in_graph  (default) tables are parameters inside the GSPMD program, row-wise
            sharded over all chips (ring-shift lookup) and trained by a second
            optimizer compiled into the same graph.
  cpu       torchrec DMP on host CPU over gloo with an eager graph-break lookup
            and FBGEMM-fused sparse optimizer; for debugging without TPUs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
from recir.embedding.backend import (
    EmbeddingBackend,
    TableOptimizer,
    TablePlan,
    TableSharding,
)
from recir.embedding.cpu import CPUEmbeddingBackend, SparseArchDispatcher
from recir.embedding.in_graph import (
    InGraphEmbeddingBackend,
    SparseArchInputBreakDispatcher,
)
from recir.examples.dlrmv3.common.model import (
    DlrmV3TrainModule,
    EMBEDDING_COLLECTION_FQN,
)
from recir.training.setup import build_sparse_lr_buffered_optimizer
from torchrec.modules.embedding_configs import EmbeddingConfig

SPARSE_ARCH_OVERRIDE_ID: int = 1
BACKENDS: Tuple[str, ...] = ("in_graph", "cpu")


@dataclass(frozen=True)
class BatchGeometry:
    """One token-budget block per local chip, ``real_rows`` real rows each."""

    real_rows: int
    local_chips: int
    num_ranks: int


def build_embedding_backend(
    name: str,
    tables: Dict[str, EmbeddingConfig],
    *,
    on_device_init: bool = True,
) -> EmbeddingBackend:
    if name == "in_graph":
        return InGraphEmbeddingBackend(
            embedding_collection_fqn=EMBEDDING_COLLECTION_FQN,
            sparse_optimizer_builder=build_sparse_lr_buffered_optimizer,
            table_plans={
                t: TablePlan(
                    sharding=TableSharding.ROWWISE, optimizer=TableOptimizer.SPARSE
                )
                for t in tables
            },
            on_device_init=on_device_init,
        )
    if name == "cpu":
        return CPUEmbeddingBackend(embedding_collection_fqn=EMBEDDING_COLLECTION_FQN)
    raise ValueError(f"unknown embedding backend {name!r}; use one of {BACKENDS}")


def prepare_backend(
    backend: EmbeddingBackend, module: DlrmV3TrainModule, geom: BatchGeometry
) -> None:
    """Hand the backend the static per-feature sizes it needs before setup()."""
    if isinstance(backend, InGraphEmbeddingBackend):
        backend.set_feature_seqlens(module.layout.feature_totals(geom.real_rows))


def make_dispatchers(
    backend: EmbeddingBackend, module: DlrmV3TrainModule, geom: BatchGeometry
) -> List[Any]:
    """Tracing dispatchers for the sparse arch (the DLRMv3 preprocess is static
    and traced in-graph, so nothing else needs one)."""
    keys, cached = module.sparse_io_shapes(geom.real_rows, geom.local_chips)
    if isinstance(backend, InGraphEmbeddingBackend):
        return [
            SparseArchInputBreakDispatcher(
                input_kjt_keys=keys,
                override_id=SPARSE_ARCH_OVERRIDE_ID,
                num_ranks=geom.num_ranks,
                feature_seqlens=module.layout.feature_totals(geom.real_rows),
            )
        ]
    if isinstance(backend, CPUEmbeddingBackend):
        return [
            SparseArchDispatcher(
                cached_output=cached,
                input_kjt_keys=keys,
                embedding_collection_fqn=EMBEDDING_COLLECTION_FQN,
                override_id=SPARSE_ARCH_OVERRIDE_ID,
            )
        ]
    raise TypeError(f"unsupported backend {type(backend).__name__}")
