# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""SparseCore sequence embeddings for recir via torchrec.experimental.torch_tpu."""

from __future__ import annotations

import contextlib
import logging
import math
from typing import Any, Dict, Generator, List, Mapping, Optional, Sequence, Tuple, Type

import torch
import torch.distributed as dist
import torch.distributed.tensor as dt

from recir.embedding.backend import check_binding, swap_submodule
from recir.training.optim import OptimConfig
from torch import nn
from torchrec.distributed.embedding_types import OptimType

# pyrefly: ignore[missing-import]
from torchrec.experimental.torch_tpu.datasets.fdo.fdo_client import (
    KeyedSparseCoreInputStats,
    SparseCoreInputStats,
)

# pyrefly: ignore[missing-import]
from torchrec.experimental.torch_tpu.datasets.input_preprocessing import (
    KeyedSparseCorePreprocessedInput,
    SparseCoreInputPreprocessor,
    SparseCorePreprocessedInput,
)

# pyrefly: ignore[missing-import]
from torchrec.experimental.torch_tpu.modules import utils as sc_utils

# pyrefly: ignore[missing-import]
from torchrec.experimental.torch_tpu.modules.embedding_configs import (
    SparseCoreEmbeddingConfig,
)

# pyrefly: ignore[missing-import]
from torchrec.experimental.torch_tpu.modules.fused_embedding_modules import (
    SparseCoreFusedEmbeddingCollection,
)
from torchrec.modules.embedding_configs import BaseEmbeddingConfig, EmbeddingConfig
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor

logger: logging.Logger = logging.getLogger(__name__)


_OPTIMIZER_MAP: Dict[OptimType, Type[torch.optim.Optimizer]] = {
    OptimType.ADAGRAD: torch.optim.Adagrad,
    OptimType.ROWWISE_ADAGRAD: torch.optim.Adagrad,
    OptimType.ADAM: torch.optim.Adam,
    OptimType.PARTIAL_ROWWISE_ADAM: torch.optim.Adam,
    OptimType.SGD: torch.optim.SGD,
}


def _unwrap_dtensors(sc_ec: nn.Module) -> None:
    """Unwrap SparseCore DTensor shards for DataParallel make_fx tracing."""
    for mod in sc_ec.modules():
        for name, param in list(mod._parameters.items()):
            if isinstance(param, dt.DTensor) or hasattr(param, "to_local"):
                local = nn.Parameter(
                    param.to_local(), requires_grad=param.requires_grad
                )
                local.__dict__.update(param.__dict__)
                mod._parameters[name] = local
    if isinstance(sc_ec.learning_rates, nn.Module):
        delattr(sc_ec, "learning_rates")
    sc_ec.learning_rates = {}


class TorchTPUSparseCoreBackedEC(nn.Module):
    """Drop-in torchrec EmbeddingCollection backed by SparseCoreFusedEmbeddingCollection."""

    def __init__(
        self,
        sc_ec: Any,
        preprocessor: Any,
        *,
        rows: int,
        feature_totals: Mapping[str, int],
        embedding_lr: float,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.sc_ec = sc_ec
        self.preprocessor = preprocessor
        self._rows = rows
        self._feature_totals: Dict[str, int] = dict(feature_totals)
        self._bound_tensors: Optional[Sequence[torch.Tensor]] = None
        self.register_buffer(
            "lr_buffer",
            torch.tensor(embedding_lr, dtype=torch.float32, device=device),
            persistent=False,
        )

    @property
    def num_step_tensors(self) -> int:
        return 4 * len(self.sc_ec.stacked_configs)

    def update_preprocessing_parameters(self, features: KeyedJaggedTensor) -> None:
        """Update per-partition SparseCore capacities from a batch via FDO."""
        stats = self.preprocessor(features).stats
        assert stats is not None
        if dist.is_initialized() and dist.get_world_size() > 1:
            names = list(stats.table_stats.keys())
            counts = torch.tensor(
                [
                    [
                        stats.table_stats[n].dropped_count,
                        stats.table_stats[n].observed_max_ids,
                        stats.table_stats[n].observed_max_unique_ids,
                    ]
                    for n in names
                ],
                dtype=torch.int32,
                device=self.lr_buffer.device,
            )
            dist.all_reduce(counts, op=dist.ReduceOp.MAX)
            stats = KeyedSparseCoreInputStats(
                {
                    n: SparseCoreInputStats(*row)
                    for n, row in zip(names, counts.to("cpu").tolist())
                }
            )
        self.preprocessor.update_preprocessing_parameters(
            stats,
            scale_factor=self.preprocessor._fdo_headroom,
            bucket_step_up=True,
        )
        for s in self.sc_ec.stacked_configs:
            s.max_ids_per_partition = max(t.max_ids_per_partition for t in s.tables)
            s.max_unique_ids_per_partition = max(
                t.max_unique_ids_per_partition for t in s.tables
            )
        self.preprocessor._allow_id_dropping = False
        self.preprocessor._rebuild_backend()

    def flatten_preprocessed(self, sc_inputs: Any) -> Tuple[torch.Tensor, ...]:
        out: List[torch.Tensor] = []
        for s in self.sc_ec.stacked_configs:
            t = sc_inputs.table_tensors[s.stack_name]
            out.extend((t.row_pointers, t.embedding_ids, t.sample_ids, t.gains))
        return tuple(out)

    @contextlib.contextmanager
    def bind_step_inputs(
        self, flat_tensors: Sequence[torch.Tensor]
    ) -> Generator[None, None, None]:
        if len(flat_tensors) != self.num_step_tensors:
            raise ValueError(
                f"Expected {self.num_step_tensors} SparseCore step tensors, got "
                f"{len(flat_tensors)}"
            )
        prev = self._bound_tensors
        self._bound_tensors = flat_tensors
        try:
            yield
        finally:
            self._bound_tensors = prev

    def forward(self, features: KeyedJaggedTensor) -> Dict[str, JaggedTensor]:
        if self._bound_tensors is None:
            raise RuntimeError(
                "SparseCore step inputs must be bound via bind_step_inputs()"
            )
        all_lengths = features.lengths()
        key_lengths = {
            k: all_lengths[i * self._rows : (i + 1) * self._rows]
            for i, k in enumerate(features.keys())
        }
        table_tensors = {
            s.stack_name: SparseCorePreprocessedInput(
                row_pointers=self._bound_tensors[4 * i],
                embedding_ids=self._bound_tensors[4 * i + 1],
                sample_ids=self._bound_tensors[4 * i + 2],
                gains=self._bound_tensors[4 * i + 3],
                lengths={f: key_lengths[f] for f in s.feature_names},
                actual_num_ids={f: self._feature_totals[f] for f in s.feature_names},
                enable_minibatching=False,
                max_ids_per_partition=s.max_ids_per_partition,
                max_unique_ids_per_partition=s.max_unique_ids_per_partition,
            )
            for i, s in enumerate(self.sc_ec.stacked_configs)
        }
        sc_features = KeyedSparseCorePreprocessedInput(table_tensors)
        self.sc_ec.learning_rates = {
            s.stack_name: self.lr_buffer for s in self.sc_ec.stacked_configs
        }
        return self.sc_ec(sc_features)


def install_sparsecore_embeddings(
    model: nn.Module,
    embedding_collection_fqn: str,
    table_configs: Sequence[BaseEmbeddingConfig],
    sparse_config: OptimConfig,
    *,
    rows: int,
    feature_totals: Mapping[str, int],
    world_size: int,
    device: torch.device,
) -> TorchTPUSparseCoreBackedEC:
    """Move dense weights to device and swap EmbeddingCollection with SparseCore."""
    check_binding(model, embedding_collection_fqn)
    if not feature_totals or rows <= 0:
        raise ValueError("rows must be positive and feature_totals non-empty")
    num_sc_per_device = sc_utils.get_num_sc_per_device() or 2
    if rows % num_sc_per_device != 0:
        raise ValueError(
            f"Per-device rows ({rows}) must be divisible by "
            f"num_sc_per_device ({num_sc_per_device})"
        )

    sc_tables: List[SparseCoreEmbeddingConfig] = []
    for tc in table_configs:
        if not isinstance(tc, EmbeddingConfig):
            raise ValueError(
                f"SparseCore sequence backend expects EmbeddingConfig, got {tc.name}"
            )
        max_feat_tokens = max(feature_totals[f] for f in tc.feature_names)
        max_seq_len = max(1, math.ceil(max_feat_tokens / rows))
        sc_tables.append(
            SparseCoreEmbeddingConfig(
                config=tc,
                max_seq_len=max_seq_len,
            )
        )

    optimizer_type = _OPTIMIZER_MAP.get(sparse_config.optim_type)
    if optimizer_type is None:
        raise ValueError(
            f"Unsupported SparseCore optimizer {sparse_config.optim_type!r}"
        )
    raw_sc_ec = SparseCoreFusedEmbeddingCollection(
        tables=sc_tables,
        optimizer_type=optimizer_type,
        optimizer_kwargs={
            "lr": sparse_config.lr,
            "eps": sparse_config.eps,
            "beta1": sparse_config.beta1,
            "beta2": sparse_config.beta2,
        },
        batch_size=rows,
        global_device_count=world_size,
        num_sc_per_device=num_sc_per_device,
        auto_stack=True,
    )
    _unwrap_dtensors(raw_sc_ec)
    preprocessor = SparseCoreInputPreprocessor(
        tables=sc_tables,
        batch_size=rows,
        global_device_count=world_size,
        num_sc_per_device=num_sc_per_device,
        auto_stack=True,
        allow_id_dropping=True,
    )
    sc_ec = TorchTPUSparseCoreBackedEC(
        raw_sc_ec,
        preprocessor,
        rows=rows,
        feature_totals=feature_totals,
        embedding_lr=sparse_config.lr,
        device=device,
    )
    swap_submodule(model, embedding_collection_fqn, nn.Identity())
    model.to(device)
    swap_submodule(model, embedding_collection_fqn, sc_ec)
    logger.info(
        "Initialized %d SparseCore stack(s): %s",
        len(raw_sc_ec.stacked_configs),
        [s.stack_name for s in raw_sc_ec.stacked_configs],
    )
    return sc_ec
