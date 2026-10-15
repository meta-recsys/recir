# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the ``pooling`` mode on ``recir::ringshift_embedding``.

Everything the torch-op layer adds is CPU-testable, which is the reason it is
its own diff: the TPU lowering that follows can only be checked on a slice.

The reference is ``F.embedding_bag``, not a second hand-written reduction.
Reducing flat values by offsets is exactly what it does -- the same layout this
op takes -- so borrowing it means a bug in the fallback cannot be mirrored by
the same bug in the expectation.
"""

from __future__ import annotations

import unittest

import recir.ops.seq_embedding_op  # noqa: F401  (registers recir::ringshift_embedding)
import torch
import torch.nn.functional as F


_VOCAB = 16
_DIM = 4

# Two rows of six ids. The bag cuts are chosen so every awkward case is present
# and forced rather than drawn: an empty bag at the start of a row and one in
# the middle, a singleton, a bag of three identical ids (accumulation), and id 0
# as a real member (a mask bug would fold it in silently).
#
#   row 0:  [3 7 1] [5] [] [2 9]
#   row 1:  [] [4 4 4] [6] [0 8]
_IDS = torch.tensor([[3, 7, 1, 5, 2, 9], [4, 4, 4, 6, 0, 8]])
_BAGS = torch.tensor([[3, 1, 0, 2], [0, 3, 1, 2]])
_ROWS, _COLS = _IDS.shape
_NBAGS = _BAGS.shape[1]

# Sequence mode reads lengths as one valid count per row.
_SEQ_LENGTHS = torch.tensor([_COLS, _COLS])


def _weight(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(_VOCAB, _DIM, generator=g, dtype=torch.float32)


def _embedding_bag_reference(
    ids: torch.Tensor,
    bags: torch.Tensor,
    weight: torch.Tensor,
    mode: str,
    padding_idx: int | None = None,
) -> torch.Tensor:
    """The same reduction via ``F.embedding_bag``, reshaped to [R, K, D].

    The functional form, not ``nn.EmbeddingBag``: the module copies ``weight``
    into a fresh ``Parameter``, so gradients would land on that copy and never
    reach the caller's tensor. Values would still match, which is why only the
    backward tests would have caught it.
    """
    flat_bags = bags.reshape(-1)
    offsets = torch.cat(
        [torch.zeros(1, dtype=torch.long), flat_bags.cumsum(0)[:-1].long()]
    )
    out = F.embedding_bag(
        ids.reshape(-1).long(), weight, offsets, mode=mode, padding_idx=padding_idx
    )
    return out.reshape(bags.shape[0], bags.shape[1], -1)


class PoolingForwardTest(unittest.TestCase):
    def test_none_is_the_unchanged_sequence_path(self) -> None:
        w = _weight()
        out = torch.ops.recir.ringshift_embedding(_IDS, _SEQ_LENGTHS, w)
        self.assertEqual(out.shape, (_ROWS, _COLS, _DIM))
        torch.testing.assert_close(out, F.embedding(_IDS, w))

    def test_sum_matches_embedding_bag(self) -> None:
        w = _weight()
        out = torch.ops.recir.ringshift_embedding(_IDS, _BAGS, w, pooling="sum")
        self.assertEqual(out.shape, (_ROWS, _NBAGS, _DIM))
        torch.testing.assert_close(out, _embedding_bag_reference(_IDS, _BAGS, w, "sum"))

    def test_mean_matches_embedding_bag(self) -> None:
        w = _weight()
        out = torch.ops.recir.ringshift_embedding(_IDS, _BAGS, w, pooling="mean")
        torch.testing.assert_close(
            out, _embedding_bag_reference(_IDS, _BAGS, w, "mean")
        )

    def test_mean_excludes_padding_idx_from_divisor(self) -> None:
        w = _weight()
        padding_idx = 4
        out = torch.ops.recir.ringshift_embedding(
            _IDS, _BAGS, w, padding_idx=padding_idx, pooling="mean"
        )
        torch.testing.assert_close(
            out,
            _embedding_bag_reference(_IDS, _BAGS, w, "mean", padding_idx=padding_idx),
        )

    def test_unknown_pooling_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown pooling mode"):
            torch.ops.recir.ringshift_embedding(_IDS, _BAGS, _weight(), pooling="avg")

    def test_empty_bag_is_zero_not_nan(self) -> None:
        """Bag (0, 2) and bag (1, 0); mean must not divide by their length."""
        w = _weight()
        for mode in ("sum", "mean"):
            out = torch.ops.recir.ringshift_embedding(_IDS, _BAGS, w, pooling=mode)
            self.assertTrue(torch.all(out[0, 2] == 0), mode)
            self.assertTrue(torch.all(out[1, 0] == 0), mode)
            self.assertTrue(torch.isfinite(out).all(), mode)

    def test_the_cut_points_are_where_lengths_say(self) -> None:
        """Swapping ids across a bag boundary must change the answer; within a
        bag it must not.

        This is the sharp test for the segment computation. An off-by-one in
        the boundary still produces plausible sums for most inputs, and it
        would move exactly one id from each bag to its neighbour -- which is
        what these two swaps distinguish.
        """
        w = _weight()
        base = torch.ops.recir.ringshift_embedding(_IDS, _BAGS, w, pooling="sum")

        within = _IDS.clone()
        within[0, 0], within[0, 1] = int(_IDS[0, 1]), int(_IDS[0, 0])  # both bag 0
        torch.testing.assert_close(
            torch.ops.recir.ringshift_embedding(within, _BAGS, w, pooling="sum"),
            base,
        )

        across = _IDS.clone()
        across[0, 2], across[0, 3] = int(_IDS[0, 3]), int(_IDS[0, 2])  # bag 0 <-> 1
        self.assertFalse(
            torch.allclose(
                torch.ops.recir.ringshift_embedding(across, _BAGS, w, pooling="sum"),
                base,
            )
        )

    def test_ids_past_the_row_total_are_ignored(self) -> None:
        """A row whose bags sum to less than C drops its tail rather than
        folding it into the last bag."""
        w = _weight()
        short = torch.tensor([[3, 1, 0, 2], [0, 3, 1, 1]])  # row 1 sums to 5, not 6
        base = torch.ops.recir.ringshift_embedding(_IDS, short, w, pooling="sum")
        poisoned = _IDS.clone()
        poisoned[1, 5] = 11
        torch.testing.assert_close(
            torch.ops.recir.ringshift_embedding(poisoned, short, w, pooling="sum"),
            base,
        )


class PoolingBackwardTest(unittest.TestCase):
    """The table gradient, against autograd through ``F.embedding_bag``."""

    def _grad(self, mode: str, cotangent: torch.Tensor) -> torch.Tensor:
        w = _weight().requires_grad_(True)
        out = torch.ops.recir.ringshift_embedding(_IDS, _BAGS, w, pooling=mode)
        (out * cotangent).sum().backward()
        assert w.grad is not None
        return w.grad

    def _reference_grad(self, mode: str, cotangent: torch.Tensor) -> torch.Tensor:
        w = _weight().requires_grad_(True)
        out = _embedding_bag_reference(_IDS, _BAGS, w, mode)
        (out * cotangent).sum().backward()
        assert w.grad is not None
        return w.grad

    def _cotangent(self, seed: int) -> torch.Tensor:
        # Random, not ones: with a uniform cotangent every bag contributes
        # equally, which hides rows attributed to the wrong bag.
        return torch.randn(
            _ROWS, _NBAGS, _DIM, generator=torch.Generator().manual_seed(seed)
        )

    def test_sum_grad_matches_reference(self) -> None:
        cot = self._cotangent(1)
        torch.testing.assert_close(
            self._grad("sum", cot), self._reference_grad("sum", cot)
        )

    def test_mean_grad_scales_by_bag_length(self) -> None:
        cot = self._cotangent(2)
        torch.testing.assert_close(
            self._grad("mean", cot), self._reference_grad("mean", cot)
        )

    def test_repeated_id_accumulates(self) -> None:
        """Bag (1, 1) is three copies of id 4 -- a hot row must accumulate."""
        ids = torch.tensor([[4, 4, 4, 4]])
        bags = torch.tensor([[4]])
        w = _weight().requires_grad_(True)
        out = torch.ops.recir.ringshift_embedding(ids, bags, w, pooling="sum")
        out.sum().backward()
        assert w.grad is not None
        self.assertTrue(torch.all(w.grad[4] == 4.0))
        # Every other row untouched.
        self.assertEqual(float(w.grad.abs().sum()), 4.0 * _DIM)


class PoolingMetaTest(unittest.TestCase):
    """The fake kernel decides the shape the compiler sees; it must branch too."""

    def test_meta_shape_follows_pooling(self) -> None:
        w = _weight().to("meta")
        seq = torch.ops.recir.ringshift_embedding(
            _IDS.to("meta"), _SEQ_LENGTHS.to("meta"), w
        )
        pooled = torch.ops.recir.ringshift_embedding(
            _IDS.to("meta"), _BAGS.to("meta"), w, pooling="sum"
        )
        self.assertEqual(seq.shape, (_ROWS, _COLS, _DIM))
        # Pooled output follows LENGTHS, not ids: the bag count is knowable
        # from nothing else.
        self.assertEqual(pooled.shape, (_ROWS, _NBAGS, _DIM))

    def test_backward_arity_survives_a_real_grad(self) -> None:
        """Guards the five-entry backward tuple.

        Adding ``pooling`` after ``padding_idx`` made the op take five inputs,
        so the registered backward must return five gradients. A stale tuple traces and
        compiles fine and only raises here, under an actual backward.
        """
        for mode, lengths in (("none", _SEQ_LENGTHS), ("sum", _BAGS), ("mean", _BAGS)):
            w = _weight().requires_grad_(True)
            out = torch.ops.recir.ringshift_embedding(_IDS, lengths, w, pooling=mode)
            out.sum().backward()
            self.assertIsNotNone(w.grad, mode)
