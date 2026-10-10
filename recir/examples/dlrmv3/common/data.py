# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Data pipeline: generative-recommenders DLRMv3 datasets -> token-budget batches.

Samples come from ``generative_recommenders.dlrm_v3.utils.get_dataset`` as
``(uih_kjt, candidates_kjt)`` pairs of variable length. ``TokenBudgetCollate``
packs them into the ``FeatureLayout`` blocks consumed by ``DlrmV3TrainModule``:

* Real rows keep their real UIH length (capped at ``uih_len``, most recent
  events). If a block's real tokens exceed ``b * uih_budget``, the longest
  histories lose their oldest events until they fit (never with the default
  ``uih_budget == uih_len``).
* Dummy rows take the block's remaining UIH tokens (id 0, at most ``uih_len``
  each), have zero contextual ids and zero-weight candidates.
* Candidates keep the first ``num_candidates``; padded ones get weight 0.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import torch
from generative_recommenders.dlrm_v3.utils import get_dataset
from generative_recommenders.modules.dlrm_hstu import DlrmHSTUConfig
from recir.examples.dlrmv3.common.model import FeatureLayout
from torch.utils.data import DataLoader, Dataset as TorchDataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

logger: logging.Logger = logging.getLogger(__name__)

Sample = Tuple[KeyedJaggedTensor, KeyedJaggedTensor]


@dataclass
class TokenBudgetBatch:
    uih_seq: torch.Tensor
    uih_lengths: torch.Tensor
    ctx: torch.Tensor
    cand: torch.Tensor
    cand_weight: torch.Tensor

    def flat(self) -> Tuple[torch.Tensor, ...]:
        return self.uih_seq, self.uih_lengths, self.ctx, self.cand, self.cand_weight

    def pin_memory(self) -> "TokenBudgetBatch":
        return TokenBudgetBatch(*(t.pin_memory() for t in self.flat()))

    def to(
        self, device: torch.device, non_blocking: bool = False
    ) -> "TokenBudgetBatch":
        return TokenBudgetBatch(
            *(t.to(device, non_blocking=non_blocking) for t in self.flat())
        )


class GRDataset(TorchDataset):
    """Adapts a generative-recommenders ``Dataset`` to ``torch.utils.data``."""

    def __init__(self, dataset: Any) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return self.dataset.get_item_count()

    def __getitem__(self, index: int) -> Sample:
        return self.__getitems__([index])[0]

    def __getitems__(self, indices: List[int]) -> List[Sample]:
        self.dataset.load_query_samples(indices)
        samples = [self.dataset.get_sample(i) for i in indices]
        self.dataset.unload_query_samples(indices)
        return samples


def _split_by_key(kjt: KeyedJaggedTensor) -> Dict[str, torch.Tensor]:
    lengths = kjt.lengths().tolist()
    return dict(zip(kjt.keys(), torch.split(kjt.values(), lengths)))


def _right_fit(x: torch.Tensor, n: int, pad: int) -> torch.Tensor:
    if x.numel() >= n:
        return x[:n]
    return torch.cat([x, torch.full((n - x.numel(),), pad, dtype=x.dtype)])


def fit_to_budget(lengths: List[int], budget: int) -> List[int]:
    """Shrink the longest lengths (water-filling) until they sum to ``budget``."""
    if sum(lengths) <= budget:
        return list(lengths)
    lo, hi = 0, max(lengths)
    while lo < hi:  # largest cap with sum(min(l, cap)) <= budget
        mid = (lo + hi + 1) // 2
        if sum(min(n, mid) for n in lengths) <= budget:
            lo = mid
        else:
            hi = mid - 1
    out = [min(n, lo) for n in lengths]
    extra = budget - sum(out)
    for i, n in enumerate(lengths):
        if extra == 0:
            break
        if n > lo:
            out[i] += 1
            extra -= 1
    return out


class TokenBudgetCollate:
    """Pack ``num_blocks * real_rows`` samples into token-budget blocks."""

    def __init__(
        self,
        layout: FeatureLayout,
        hstu_config: DlrmHSTUConfig,
        real_rows: int,
        num_blocks: int = 1,
    ) -> None:
        self.layout = layout
        self.real_rows = real_rows
        self.num_blocks = num_blocks
        self.dummy_rows: int = layout.dummy_rows(real_rows)
        self.cand_time_key: str = hstu_config.candidates_querytime_feature_name

    def _block(self, samples: Sequence[Sample]) -> Tuple[torch.Tensor, ...]:
        L = self.layout
        C = L.num_candidates
        uih = [_split_by_key(s[0]) for s in samples]
        cand = [_split_by_key(s[1]) for s in samples]
        lengths = fit_to_budget(
            [min(int(u[L.seq_keys[0]].numel()), L.uih_len) for u in uih],
            L.tokens(self.real_rows),
        )
        slack = L.tokens(self.real_rows) - sum(lengths)
        for _ in range(self.dummy_rows):
            lengths.append(min(slack, L.uih_len))
            slack -= lengths[-1]
        assert slack == 0
        seq_cols = []
        for key in L.seq_keys:
            # Most recent events of each real row, then zeros for dummy tokens.
            parts = [u[key][u[key].numel() - n :] for u, n in zip(uih, lengths)]
            pad = sum(lengths[len(samples) :])
            parts.append(torch.zeros(pad, dtype=torch.int64))
            seq_cols.append(torch.cat(parts).long())
        ctx_rows, cand_rows, weight_rows = [], [], []
        for u, c in zip(uih, cand):
            ctx_rows.append(
                torch.cat(
                    [
                        _right_fit(u[k], n, 0)
                        for k, n in zip(L.contextual_keys, L.contextual_lens)
                    ]
                    or [torch.zeros(0, dtype=torch.int64)]
                )
            )
            cand_cols = []
            for key in L.cand_keys:
                v = c[key]
                pad = int(v[0]) if key == self.cand_time_key and v.numel() else 0
                cand_cols.append(_right_fit(v, C, pad))
            cand_rows.append(torch.cat(cand_cols))
            weight = torch.zeros(C, dtype=torch.float32)
            weight[: min(int(c[L.cand_keys[0]].numel()), C)] = 1.0
            weight_rows.append(weight)
        ctx = torch.stack(ctx_rows).long()
        cand_t = torch.stack(cand_rows).long()
        weights = torch.stack(weight_rows)
        d = self.dummy_rows
        return (
            torch.stack(seq_cols, dim=1),
            torch.tensor(lengths, dtype=torch.int64),
            torch.cat([ctx, ctx.new_zeros(d, ctx.shape[1])]),
            torch.cat([cand_t, cand_t.new_zeros(d, cand_t.shape[1])]),
            torch.cat([weights, weights.new_zeros(d, C)]),
        )

    def __call__(self, samples: Sequence[Sample]) -> TokenBudgetBatch:
        b = self.real_rows
        if len(samples) != b * self.num_blocks:
            raise ValueError(
                f"expected {b * self.num_blocks} samples, got {len(samples)}"
            )
        blocks = [
            self._block(samples[i * b : (i + 1) * b]) for i in range(self.num_blocks)
        ]
        return TokenBudgetBatch(*(torch.cat(parts) for parts in zip(*blocks)))


def make_dataset(
    dataset: str,
    data_root: str,
    hstu_config: DlrmHSTUConfig,
    tables: Dict[str, EmbeddingConfig],
) -> GRDataset:
    cls, kwargs = get_dataset(name=dataset, new_path_prefix=data_root)
    kwargs["embedding_config"] = tables
    return GRDataset(cls(hstu_config=hstu_config, is_inference=False, **kwargs))


def make_train_eval_loaders(
    dataset: GRDataset,
    collate: TokenBudgetCollate,
    batch_size: int,
    train_split: float,
    num_workers: int,
    num_replicas: int,
    rank: int,
    seed: int,
    pin_memory: bool,
) -> Tuple[DataLoader, DataLoader]:
    n = len(dataset)
    n_train = int(round(train_split * n))
    splits = (Subset(dataset, range(n_train)), Subset(dataset, range(n_train, n)))
    loaders = []
    for i, subset in enumerate(splits):
        sampler = DistributedSampler(
            subset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=(i == 0),
            seed=seed,
            drop_last=True,
        )
        loaders.append(
            DataLoader(
                subset,
                batch_size=batch_size,
                sampler=sampler,
                collate_fn=collate,
                drop_last=True,
                num_workers=num_workers,
                # Citrine C0: callers pin on GPU hosts; TPU hosts have no CUDA.
                pin_memory=pin_memory,  # noqa: CITRINE(unpinned_memcpy)
                persistent_workers=num_workers > 0,
            )
        )
    logger.info("dataset %d samples: train=%d eval=%d", n, n_train, n - n_train)
    return loaders[0], loaders[1]
