# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Build a recir embedding backend by name and bind it to a model by FQN.

Every model binds the same way: it names its torchrec ``EmbeddingCollection``
by FQN and says how each table is placed. What stays per model is the tracing
dispatchers it pairs with the backend.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch
import torch.nn as nn
from recir.embedding.backend import (
    EmbeddingBackend,
    table_plans_from_embedding_tables,
    TablePlan,
)
from recir.embedding.cpu import CPUEmbeddingBackend
from recir.embedding.in_graph import InGraphEmbeddingBackend
from recir.embedding.sparsecore import SparseCoreEmbeddingBackend
from recir.training.optim import OptimConfig
from recir.training.setup import build_sparse_lr_buffered_optimizer

EMBEDDING_BACKENDS: tuple[str, ...] = ("cpu", "tpu", "in_graph")

SparseOptimizerBuilder = Callable[
    [nn.Module, list[nn.Parameter], OptimConfig, torch.device, float],
    tuple[torch.optim.Optimizer, torch.Tensor],
]


def build_embedding_backend(
    name: str,
    *,
    embedding_collection_fqn: str,
    table_plans: Mapping[str, TablePlan] | None = None,
    embedding_tables: dict[str, Any] | None = None,
    sparse_optimizer_builder: SparseOptimizerBuilder = build_sparse_lr_buffered_optimizer,
    on_device_init: bool = True,
) -> EmbeddingBackend:
    """``name`` is one of :data:`EMBEDDING_BACKENDS`.

    ``embedding_tables`` are the model's own table objects; the cpu backend
    reads their placement, and in_graph derives its plans from them when
    ``table_plans`` is not given.
    """
    if name == "cpu":
        return CPUEmbeddingBackend(
            embedding_collection_fqn=embedding_collection_fqn,
            embedding_tables=embedding_tables,
        )
    if name == "tpu":
        return SparseCoreEmbeddingBackend(embedding_collection_fqn)
    if name == "in_graph":
        if table_plans is None and embedding_tables is not None:
            table_plans = table_plans_from_embedding_tables(embedding_tables)
        if table_plans is None:
            raise ValueError(
                "the in_graph backend needs table plans: each table's placement "
                "decides both its layout and which optimizer trains it, and "
                "there is no default to fall back on."
            )
        return InGraphEmbeddingBackend(
            embedding_collection_fqn=embedding_collection_fqn,
            sparse_optimizer_builder=sparse_optimizer_builder,
            table_plans=dict(table_plans),
            on_device_init=on_device_init,
        )
    raise ValueError(
        f"unknown embedding backend {name!r}; use one of {EMBEDDING_BACKENDS}"
    )
