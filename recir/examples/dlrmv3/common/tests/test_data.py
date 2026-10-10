# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import List, Tuple

import torch
from generative_recommenders.modules.dlrm_hstu import DlrmHSTUConfig
from recir.examples.dlrmv3.common.data import fit_to_budget, TokenBudgetCollate
from recir.examples.dlrmv3.common.model import FeatureLayout
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

CTX = ["user_id"]
SEQ = ["item_id", "ts"]
CAND = ["cand_item_id", "cand_ts"]


def _layout(uih_len: int, budget: int, num_candidates: int = 2) -> FeatureLayout:
    cfg = DlrmHSTUConfig()
    cfg.contextual_feature_to_max_length = {"user_id": 1}
    cfg.hstu_uih_feature_names = CTX + SEQ
    cfg.hstu_candidate_feature_names = CAND
    cfg.candidates_querytime_feature_name = "cand_ts"
    return FeatureLayout.from_config(cfg, uih_len, num_candidates, budget)


def _sample(
    user: int, n: int, n_cand: int
) -> Tuple[KeyedJaggedTensor, KeyedJaggedTensor]:
    seq = list(range(1, n + 1))
    uih = KeyedJaggedTensor(
        keys=CTX + SEQ,
        values=torch.tensor([user] + seq + [100 + i for i in seq]),
        lengths=torch.tensor([1, n, n]),
    )
    cand = KeyedJaggedTensor(
        keys=CAND,
        values=torch.tensor([7] * n_cand + [999] * n_cand),
        lengths=torch.tensor([n_cand, n_cand]),
    )
    return uih, cand


class FitToBudgetTest(unittest.TestCase):
    def test_under_budget_unchanged(self) -> None:
        self.assertEqual(fit_to_budget([3, 1, 2], 10), [3, 1, 2])

    def test_trims_longest_first(self) -> None:
        out = fit_to_budget([10, 2, 7], 12)
        self.assertEqual(sum(out), 12)
        self.assertEqual(out[1], 2)
        self.assertLessEqual(max(out) - min(out[0], out[2]), 1)


class TokenBudgetCollateTest(unittest.TestCase):
    def _check_block(
        self, layout: FeatureLayout, lengths: List[int], real: int
    ) -> None:
        tokens = layout.tokens(real)
        self.assertEqual(sum(lengths), tokens)
        self.assertEqual(len(lengths), layout.rows(real))
        self.assertTrue(all(0 <= n <= layout.uih_len for n in lengths))

    def test_layout_invariants(self) -> None:
        layout = _layout(uih_len=6, budget=4)
        collate = TokenBudgetCollate(layout, _cfg_stub(), real_rows=3, num_blocks=2)
        samples = [_sample(u, n, c) for u, n, c in [(1, 6, 2), (2, 1, 1), (3, 9, 2)]]
        samples += [_sample(u, n, 2) for u, n in [(4, 2), (5, 2), (6, 3)]]
        batch = collate(samples)
        rows = layout.rows(3)
        self.assertEqual(tuple(batch.uih_seq.shape), (2 * layout.tokens(3), 2))
        self.assertEqual(tuple(batch.uih_lengths.shape), (2 * rows,))
        self.assertEqual(tuple(batch.ctx.shape), (2 * rows, 1))
        self.assertEqual(tuple(batch.cand.shape), (2 * rows, 4))
        for blk in range(2):
            lens = batch.uih_lengths[blk * rows : (blk + 1) * rows].tolist()
            self._check_block(layout, lens, 3)
        # Block 0 is over budget (6 + 1 + 6 > 12): the longest rows are trimmed
        # and keep their most recent events.
        self.assertEqual(batch.uih_lengths[:3].tolist(), [6, 1, 5])
        self.assertEqual(batch.uih_seq[:6, 0].tolist(), [1, 2, 3, 4, 5, 6])
        self.assertEqual(batch.uih_seq[7:12, 0].tolist(), [5, 6, 7, 8, 9])
        # Dummy rows: zero ids, zero candidate weight.
        self.assertEqual(batch.cand_weight[3:rows].sum().item(), 0.0)
        self.assertEqual(batch.ctx[3:rows].sum().item(), 0)
        # Padded candidate of a real row has weight 0.
        self.assertEqual(batch.cand_weight[1].tolist(), [1.0, 0.0])
        # Block 1 is under budget: real lengths kept, dummies take the slack.
        self.assertEqual(batch.uih_lengths[rows : rows + 3].tolist(), [2, 2, 3])


def _cfg_stub() -> DlrmHSTUConfig:
    cfg = DlrmHSTUConfig()
    cfg.candidates_querytime_feature_name = "cand_ts"
    return cfg
