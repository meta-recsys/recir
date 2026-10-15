# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""DLRMv3 (generative-recommenders ``DlrmHSTU``) train module for recir.

recir traces the whole train step, so every shape must be known without
reading data. ``DlrmHSTU`` supports that when given static max lengths and
KJTs that carry ``length_per_key``; this module builds those from a
token-budget batch and calls ``DlrmHSTU`` unchanged.

Token-budget layout (compatible with ``recir.region_transforms.padding``):
the batch is ``num_blocks`` blocks (one per chip on TPU, one per rank on GPU),
each holding ``b`` real rows plus ``d`` dummy rows. Real rows keep their real
UIH lengths; every UIH sequence key has exactly ``b * uih_budget`` tokens per
block, the dummy rows absorbing the slack (each at most ``uih_len`` long). So
attention never sees padding, totals are static, and splitting dim 0 evenly
over chips gives each chip its own block. Dummy rows and padded candidates get
supervision weight 0.

Tensors (dim 0 is the sharded dim; blocks are concatenated along it):

* ``uih_seq``: int64 ``[tokens, num_seq_keys]`` -- UIH sequence ids, token-major.
* ``uih_lengths``: int64 ``[rows]`` -- UIH length of every row.
* ``ctx``: int64 ``[rows, sum(contextual lengths)]`` -- contextual ids.
* ``cand``: int64 ``[rows, num_cand_keys * num_candidates]``.
* ``cand_weight``: float32 ``[rows, num_candidates]`` -- 1 for real candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from generative_recommenders.common import HammerKernel
from generative_recommenders.modules.dlrm_hstu import DlrmHSTU, DlrmHSTUConfig
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor

# Where DlrmHSTU keeps its torchrec EmbeddingCollection inside DlrmV3TrainModule.
EMBEDDING_COLLECTION_FQN = "model._embedding_collection"


@dataclass(frozen=True)
class FeatureLayout:
    contextual_keys: Tuple[str, ...]
    contextual_lens: Tuple[int, ...]
    seq_keys: Tuple[str, ...]
    cand_keys: Tuple[str, ...]
    uih_len: int
    num_candidates: int
    uih_budget: int

    @staticmethod
    def from_config(
        cfg: DlrmHSTUConfig, uih_len: int, num_candidates: int, uih_budget: int = 0
    ) -> "FeatureLayout":
        budget = uih_budget or uih_len
        if not 0 < budget <= uih_len:
            raise ValueError(f"uih_budget must be in (0, {uih_len}], got {budget}")
        contextual = dict(cfg.contextual_feature_to_max_length or {})
        return FeatureLayout(
            contextual_keys=tuple(contextual),
            contextual_lens=tuple(contextual.values()),
            seq_keys=tuple(
                k for k in cfg.hstu_uih_feature_names if k not in contextual
            ),
            cand_keys=tuple(cfg.hstu_candidate_feature_names),
            uih_len=uih_len,
            num_candidates=num_candidates,
            uih_budget=budget,
        )

    def dummy_rows(self, real_rows: int) -> int:
        """Rows needed to absorb a block's slack when every real row is empty."""
        tokens = self.tokens(real_rows)
        return (tokens + self.uih_len - 1) // self.uih_len

    def rows(self, real_rows: int) -> int:
        return real_rows + self.dummy_rows(real_rows)

    def tokens(self, real_rows: int) -> int:
        return real_rows * self.uih_budget

    @property
    def kjt_keys(self) -> List[str]:
        return list(self.contextual_keys + self.seq_keys + self.cand_keys)

    def feature_totals(self, real_rows: int) -> Dict[str, int]:
        """Values per KJT key in one block of ``real_rows`` real rows."""
        rows = self.rows(real_rows)
        totals = {
            k: rows * n for k, n in zip(self.contextual_keys, self.contextual_lens)
        }
        totals.update({k: self.tokens(real_rows) for k in self.seq_keys})
        totals.update({k: rows * self.num_candidates for k in self.cand_keys})
        return totals


class DlrmV3TrainModule(nn.Module):
    """``forward(uih_seq, uih_lengths, ctx, cand, cand_weight)``
    ``-> (losses[num_tasks], {"prediction", "label", "weight"})``.

    Embedding lookups run in ``DlrmHSTU``'s own EmbeddingCollection
    (``EMBEDDING_COLLECTION_FQN``): on GPU they trace to ``aten.embedding`` on
    its table weights, which HybridParallel shards; on TPU a ``recir.embedding``
    backend replaces the collection.
    """

    def __init__(
        self,
        hstu_config: DlrmHSTUConfig,
        tables: Dict[str, EmbeddingConfig],
        layout: FeatureLayout,
        *,
        kernel: HammerKernel,
        bf16_training: bool = True,
    ) -> None:
        super().__init__()
        self.model = DlrmHSTU(
            hstu_configs=hstu_config,
            embedding_tables=tables,
            is_inference=False,
            bf16_training=bf16_training,
        )
        self.model.set_hammer_kernel(kernel)
        self.layout = layout
        self._tables = tables
        self._task_names: List[str] = [
            t.task_name for t in hstu_config.multitask_configs
        ]

    @staticmethod
    def _kjt(
        keys: List[str],
        values: torch.Tensor,
        lengths: torch.Tensor,
        length_per_key: List[int],
    ) -> KeyedJaggedTensor:
        # Both per-key lists are given: torchrec otherwise derives offsets from
        # a lengths tensor (.tolist()) when it detects a compile/trace context.
        offset_per_key = [0]
        for n in length_per_key:
            offset_per_key.append(offset_per_key[-1] + n)
        return KeyedJaggedTensor(
            keys=keys,
            values=values,
            lengths=lengths,
            length_per_key=length_per_key,
            offset_per_key=offset_per_key,
        )

    def _kjts(
        self,
        uih_seq: torch.Tensor,
        uih_lengths: torch.Tensor,
        ctx: torch.Tensor,
        cand: torch.Tensor,
    ) -> Tuple[KeyedJaggedTensor, KeyedJaggedTensor]:
        """UIH and candidate KJTs whose per-key sizes come from tensor shapes,
        so keyed access never reads lengths data."""
        L = self.layout
        # Python ints (the batch layout is fixed): the per-key sizes must stay
        # valid when a region transform re-traces the embedding lookup alone.
        rows = int(uih_lengths.shape[0])
        tokens = int(uih_seq.shape[0])
        # TPU Mosaic cannot vectorize sharded int64 count tensors; sequence
        # lengths are bounded by the static batch geometry and fit in int32.
        uih_lengths = uih_lengths.to(torch.int32)

        def full(n: int) -> torch.Tensor:
            return uih_lengths.new_full((rows,), n)

        uih_values, uih_lens, uih_per_key = [], [], []
        start = 0
        for n in L.contextual_lens:
            uih_values.append(ctx[:, start : start + n].reshape(-1))
            uih_lens.append(full(n))
            uih_per_key.append(rows * n)
            start += n
        for i in range(len(L.seq_keys)):
            uih_values.append(uih_seq[:, i])
            uih_lens.append(uih_lengths)
            uih_per_key.append(tokens)
        C = L.num_candidates
        uih = self._kjt(
            list(L.contextual_keys + L.seq_keys),
            torch.cat(uih_values),
            torch.cat(uih_lens),
            uih_per_key,
        )
        candidates = self._kjt(
            list(L.cand_keys),
            torch.cat(
                [
                    cand[:, i * C : (i + 1) * C].reshape(-1)
                    for i in range(len(L.cand_keys))
                ]
            ),
            torch.cat([full(C) for _ in L.cand_keys]),
            [rows * C] * len(L.cand_keys),
        )
        return uih, candidates

    def sparse_io_shapes(
        self, real_rows: int, num_blocks: int = 1
    ) -> Tuple[List[str], Dict[str, JaggedTensor]]:
        """EmbeddingCollection input keys and a zero output of the shapes a
        batch of ``num_blocks`` blocks produces (what tracing stubs cache)."""
        totals = self.layout.feature_totals(real_rows)
        rows = self.layout.rows(real_rows) * num_blocks
        out: Dict[str, JaggedTensor] = {}
        for t in self._tables.values():
            for f in t.feature_names:
                out[f] = JaggedTensor(
                    values=torch.zeros(totals[f] * num_blocks, t.embedding_dim),
                    lengths=torch.zeros(rows, dtype=torch.int64),
                )
        return self.layout.kjt_keys, out

    def forward(
        self,
        uih_seq: torch.Tensor,
        uih_lengths: torch.Tensor,
        ctx: torch.Tensor,
        cand: torch.Tensor,
        cand_weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        uih, candidates = self._kjts(uih_seq, uih_lengths, ctx, cand)
        _, _, aux_losses, preds, labels, weights = self.model(
            uih,
            candidates,
            max_uih_len=self.layout.uih_len,
            max_num_candidates=self.layout.num_candidates,
            candidate_weights=cand_weight.reshape(-1),
        )
        assert preds is not None and labels is not None and weights is not None
        # DlrmHSTU reports losses only in training mode.
        losses = (
            torch.stack([aux_losses[name] for name in self._task_names])
            if aux_losses
            else preds.new_zeros(len(self._task_names))
        )
        return losses, {"prediction": preds, "label": labels, "weight": weights}
