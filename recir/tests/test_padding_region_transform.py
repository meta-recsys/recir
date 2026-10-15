# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""
Tests for PaddingRegionTransform — custom ops, auto-discovery,
bind_signature validation, and the transform_inputs path.
"""

import unittest
from typing import Any, List, Tuple
from unittest.mock import patch

import torch
import torch.nn as nn
from functorch import make_fx
from recir.region_transforms.padding import (
    JaggedPairConfig,
    mark_padding_graph_breaks,
    pad_candidate_axis_to_static,
    pad_count_to_sum,
    pad_jagged_lengths,
    pad_jagged_pair_to_static,
    pad_jagged_values,
    pad_to_static,
    PaddingRegionTransform,
    trim_per_chip_to_size,
    unpad_grad,
    unpad_jagged_grad,
)
from torch._subclasses.fake_tensor import FakeTensorMode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_lengths(n: int, total: int) -> torch.Tensor:
    raw = torch.ones(n, dtype=torch.long)
    remaining = total - n
    for i in range(remaining):
        raw[i % n] += 1
    return raw


def _dummy_train_step(
    mod: nn.Module,
    opt: torch.optim.Optimizer,
    ro_float: torch.Tensor,
    nro_float: torch.Tensor,
    ro_values: torch.Tensor,
    ro_lengths: torch.Tensor,
    nro_values: torch.Tensor,
    num_candidates: torch.Tensor,
    supervision_weight: torch.Tensor,
) -> None:
    pass


# ---------------------------------------------------------------------------
# Tests: custom ops
# ---------------------------------------------------------------------------


class PadToStaticOpTest(unittest.TestCase):
    def test_pads_when_smaller(self) -> None:
        x = torch.randn(6, 16)
        result = pad_to_static(x, target_size=8, dim=0)
        self.assertEqual(result.shape, (8, 16))
        torch.testing.assert_close(result[:6], x)
        self.assertTrue(torch.all(result[6:] == 0))

    def test_trims_when_larger(self) -> None:
        x = torch.randn(10, 16)
        result = pad_to_static(x, target_size=8, dim=0)
        self.assertEqual(result.shape, (8, 16))
        torch.testing.assert_close(result, x[:8])

    @patch("recir.region_transforms.padding.runtime_num_local", return_value=2)
    def test_pads_per_chip(self, _mock_num_local: Any) -> None:
        x = torch.arange(4, dtype=torch.float32)
        result = pad_to_static(x, target_size=8, dim=0)
        self.assertEqual(result.tolist(), [0, 1, 0, 0, 2, 3, 0, 0])

    @patch("recir.region_transforms.padding.runtime_num_local", return_value=2)
    def test_trims_per_chip_and_restores_gradients(self, _mock_num_local: Any) -> None:
        x = torch.arange(8, dtype=torch.float32, requires_grad=True)
        result = pad_to_static(x, target_size=4, dim=0, pad_id=1)
        self.assertEqual(result.tolist(), [0, 1, 4, 5])
        result.backward(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        self.assertEqual(x.grad.tolist(), [1, 2, 0, 0, 3, 4, 0, 0])  # pyre-ignore

    def test_noop_when_exact(self) -> None:
        x = torch.randn(8, 16)
        result = pad_to_static(x, target_size=8, dim=0)
        self.assertEqual(result.shape, (8, 16))

    def test_dim1(self) -> None:
        x = torch.randn(4, 5)
        result = pad_to_static(x, target_size=8, dim=1)
        self.assertEqual(result.shape, (4, 8))

    def test_backward(self) -> None:
        x = torch.randn(6, 16, requires_grad=True)
        y = pad_to_static(x, target_size=8, dim=0, pad_id=0)
        self.assertEqual(y.shape, (8, 16))
        loss = y.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, (6, 16))

    def test_backward_runtime_size(self) -> None:
        """Backward uses runtime original_size from global registry, not trace-time."""
        from recir.region_transforms.padding import _PAD_ORIGINAL_SIZES

        x = torch.randn(6, 16, requires_grad=True)
        y = pad_to_static(x, target_size=8, dim=0, pad_id=42)
        self.assertEqual(_PAD_ORIGINAL_SIZES[42], 6)
        loss = y.sum()
        loss.backward()
        self.assertEqual(x.grad.shape, (6, 16))  # pyre-ignore


class TrimPerChipToSizeTest(unittest.TestCase):
    def test_removes_each_chip_dummy_suffix(self) -> None:
        padded = torch.tensor([1, 2, 0, 3, 4, 0])

        result = trim_per_chip_to_size(padded, target_size=4, num_local=2)

        self.assertEqual(result.tolist(), [1, 2, 3, 4])


class UnpadGradOpTest(unittest.TestCase):
    def test_slices(self) -> None:
        from recir.region_transforms.padding import _PAD_ORIGINAL_SIZES

        _PAD_ORIGINAL_SIZES[99] = 6
        grad = torch.randn(8, 16)
        result = unpad_grad(grad, pad_id=99, dim=0)
        self.assertEqual(result.shape, (6, 16))
        torch.testing.assert_close(result, grad[:6])


class PadJaggedLengthsOpTest(unittest.TestCase):
    def test_simple_kps1(self) -> None:
        lengths = _random_lengths(4, 30)
        result = pad_jagged_lengths(lengths, per_key_budgets=[50], target_batch=8)
        self.assertEqual(result.shape[0], 8)
        self.assertEqual(int(result.sum().item()), 50)

    def test_even_distribution(self) -> None:
        lengths = _random_lengths(4, 30)
        result = pad_jagged_lengths(lengths, per_key_budgets=[50], target_batch=8)
        dummy = result[4:]
        self.assertEqual(int(dummy.sum().item()), 20)
        self.assertTrue(torch.all(dummy >= 4))
        self.assertTrue(torch.all(dummy <= 6))

    def test_multi_key_uniform(self) -> None:
        K = 3
        B = 4
        per_key_total = 10
        key_lengths = [_random_lengths(B, per_key_total) for _ in range(K)]
        lengths = torch.cat(key_lengths)

        result = pad_jagged_lengths(
            lengths, per_key_budgets=[20, 20, 20], target_batch=8
        )
        self.assertEqual(result.shape[0], 8 * K)
        self.assertEqual(int(result.sum().item()), 20 * K)

        padded_keys = result.reshape(K, 8)
        for k in range(K):
            self.assertEqual(int(padded_keys[k].sum().item()), 20)

    def test_multi_key_different_budgets(self) -> None:
        B = 4
        key0_lens = _random_lengths(B, 10)  # sum=10
        key1_lens = _random_lengths(B, 14)  # sum=14
        lengths = torch.cat([key0_lens, key1_lens])

        result = pad_jagged_lengths(lengths, per_key_budgets=[20, 30], target_batch=8)
        self.assertEqual(result.shape[0], 16)  # 2 keys * 8
        padded_keys = result.reshape(2, 8)
        self.assertEqual(int(padded_keys[0].sum().item()), 20)
        self.assertEqual(int(padded_keys[1].sum().item()), 30)

    def test_trims_when_over_budget_kps1(self) -> None:
        lengths = torch.tensor([10, 8, 7, 5], dtype=torch.long)  # sum=30
        result = pad_jagged_lengths(lengths, per_key_budgets=[20], target_batch=4)
        self.assertEqual(result.shape[0], 4)
        self.assertEqual(int(result.sum().item()), 20)
        self.assertEqual(result.tolist(), [10, 8, 2, 0])

    def test_trims_when_over_budget_multi_key(self) -> None:
        # key0: [10, 5] sum=15, key1: [3, 2] sum=5
        lengths = torch.tensor([10, 5, 3, 2], dtype=torch.long)
        result = pad_jagged_lengths(lengths, per_key_budgets=[12, 12], target_batch=4)
        self.assertEqual(result.shape[0], 8)  # 2 keys * 4 samples
        padded_keys = result.reshape(2, 4)
        self.assertEqual(int(padded_keys[0].sum().item()), 12)
        self.assertEqual(padded_keys[0][:2].tolist(), [10, 2])
        self.assertEqual(int(padded_keys[1].sum().item()), 12)

    def test_no_padding_needed(self) -> None:
        lengths = _random_lengths(4, 50)
        result = pad_jagged_lengths(lengths, per_key_budgets=[50], target_batch=4)
        self.assertEqual(result.shape[0], 4)
        self.assertEqual(int(result.sum().item()), 50)


class PadJaggedValuesOpTest(unittest.TestCase):
    def test_per_key_uniform(self) -> None:
        """Each key's values segment is padded independently."""
        key0_lens = torch.tensor([3, 5, 1, 2], dtype=torch.long)  # sum=11
        key1_lens = torch.tensor([4, 1, 6, 3], dtype=torch.long)  # sum=14
        lengths = torch.cat([key0_lens, key1_lens])
        values = torch.randn(25, 4)

        result = pad_jagged_values(values, lengths, per_key_budgets=[20, 20], dim=0)
        self.assertEqual(result.shape, (40, 4))
        torch.testing.assert_close(result[:11], values[:11])
        self.assertTrue(torch.all(result[11:20] == 0))
        torch.testing.assert_close(result[20:34], values[11:25])
        self.assertTrue(torch.all(result[34:40] == 0))

    @patch("recir.region_transforms.padding.runtime_num_local", return_value=2)
    def test_single_key_uses_uneven_per_chip_length_totals(
        self, _mock_num_local: Any
    ) -> None:
        lengths = torch.tensor([1, 2, 3, 2], dtype=torch.long)
        values = torch.arange(8, dtype=torch.float32)

        result = pad_jagged_values(values, lengths, per_key_budgets=[12])

        self.assertEqual(
            result.tolist(),
            [0, 1, 2, 0, 0, 0, 3, 4, 5, 6, 7, 0],
        )

    def test_per_key_different_budgets(self) -> None:
        """Keys with different budgets produce different segment sizes."""
        key0_lens = torch.tensor([3, 2], dtype=torch.long)  # sum=5
        key1_lens = torch.tensor([4, 6], dtype=torch.long)  # sum=10
        lengths = torch.cat([key0_lens, key1_lens])
        values = torch.randn(15, 4)

        result = pad_jagged_values(values, lengths, per_key_budgets=[8, 16], dim=0)
        self.assertEqual(result.shape, (24, 4))  # 8 + 16
        # key0: 5 real + 3 zeros
        torch.testing.assert_close(result[:5], values[:5])
        self.assertTrue(torch.all(result[5:8] == 0))
        # key1: 10 real + 6 zeros
        torch.testing.assert_close(result[8:18], values[5:15])
        self.assertTrue(torch.all(result[18:24] == 0))

    def test_trims_when_over_budget(self) -> None:
        lengths = torch.tensor([10, 15], dtype=torch.long)
        values = torch.randn(25, 4)
        result = pad_jagged_values(values, lengths, per_key_budgets=[8, 8], dim=0)
        self.assertEqual(result.shape, (16, 4))
        torch.testing.assert_close(result[:8], values[:8])
        torch.testing.assert_close(result[8:16], values[10:18])

    def test_noop_when_exact(self) -> None:
        lengths = torch.tensor([10, 10], dtype=torch.long)
        values = torch.randn(20, 4)
        result = pad_jagged_values(values, lengths, per_key_budgets=[10, 10], dim=0)
        self.assertEqual(result.shape, (20, 4))

    def test_backward_per_key(self) -> None:
        """Gradient flows back correctly with per-key unpadding."""
        key0_lens = torch.tensor([3, 2], dtype=torch.long)  # sum=5
        key1_lens = torch.tensor([4, 1], dtype=torch.long)  # sum=5
        lengths = torch.cat([key0_lens, key1_lens])
        values = torch.randn(10, 4, requires_grad=True)

        result = pad_jagged_values(
            values, lengths, per_key_budgets=[8, 8], dim=0, pad_id=200
        )
        self.assertEqual(result.shape, (16, 4))
        loss = result.sum()
        loss.backward()
        self.assertIsNotNone(values.grad)
        self.assertEqual(values.grad.shape, (10, 4))
        torch.testing.assert_close(values.grad, torch.ones(10, 4))

    def test_backward_different_budgets(self) -> None:
        key0_lens = torch.tensor([3], dtype=torch.long)
        key1_lens = torch.tensor([5], dtype=torch.long)
        lengths = torch.cat([key0_lens, key1_lens])
        values = torch.randn(8, 2, requires_grad=True)

        result = pad_jagged_values(
            values, lengths, per_key_budgets=[6, 10], dim=0, pad_id=201
        )
        self.assertEqual(result.shape, (16, 2))  # 6 + 10
        loss = result.sum()
        loss.backward()
        self.assertEqual(values.grad.shape, (8, 2))  # pyre-ignore
        torch.testing.assert_close(values.grad, torch.ones(8, 2))  # pyre-ignore

    def test_tile_pad_fills_with_real_ids(self) -> None:
        """tile_pad fills each key's deficit by tiling its own real ids (no 0s)."""
        key0_lens = torch.tensor([3, 2], dtype=torch.long)  # sum=5
        key1_lens = torch.tensor([4, 6], dtype=torch.long)  # sum=10
        lengths = torch.cat([key0_lens, key1_lens])
        values = torch.arange(15, dtype=torch.long)

        result = pad_jagged_values(
            values, lengths, per_key_budgets=[8, 16], dim=0, tile_pad=True
        )
        self.assertEqual(result.shape, (24,))  # 8 + 16
        # key0: 5 real (0..4) + 3 tiled (arange(3) % 5 -> 0,1,2)
        torch.testing.assert_close(result[:5], values[:5])
        torch.testing.assert_close(result[5:8], values[:3])
        # key1: 10 real (5..14) + 6 tiled (arange(6) % 10 -> 5..10)
        torch.testing.assert_close(result[8:18], values[5:15])
        torch.testing.assert_close(result[18:24], values[5:11])

    def test_tile_pad_real_region_and_backward_match_zerofill(self) -> None:
        """tile_pad changes only the pad region; [0:current] and grad are identical.

        The custom backward slices [0:current], so pad positions are treated as
        non-differentiable in both modes -- correct for id-lists whose pads are
        masked out downstream (grad into pad positions is 0).
        """
        key0_lens = torch.tensor([3, 2], dtype=torch.long)  # sum=5
        key1_lens = torch.tensor([4, 1], dtype=torch.long)  # sum=5
        lengths = torch.cat([key0_lens, key1_lens])
        base = torch.randn(10, 4)

        v0 = base.clone().requires_grad_(True)
        v1 = base.clone().requires_grad_(True)
        r_zero = pad_jagged_values(
            v0, lengths, per_key_budgets=[8, 8], dim=0, pad_id=310, tile_pad=False
        )
        r_tile = pad_jagged_values(
            v1, lengths, per_key_budgets=[8, 8], dim=0, pad_id=311, tile_pad=True
        )
        # Real region identical (key0 [0:5], key1 [8:13]).
        torch.testing.assert_close(r_tile[:5], r_zero[:5])
        torch.testing.assert_close(r_tile[8:13], r_zero[8:13])
        # Backward identical: grad flows only to [0:current] in both modes.
        r_zero.sum().backward()
        r_tile.sum().backward()
        torch.testing.assert_close(v1.grad, v0.grad)  # pyre-ignore
        torch.testing.assert_close(v1.grad, torch.ones(10, 4))  # pyre-ignore

    @patch("recir.region_transforms.padding.runtime_num_local", return_value=2)
    def test_chip_with_zero_rows_yields_zero_block_total(
        self, _mock_num_local: Any
    ) -> None:
        """A chip owning no candidates must produce an empty block, not a negative one.

        `count` puts every candidate on chip 1, so chip 0 owns zero rows. Each
        block total is a difference of prefix sums, and the empty range has to
        read as 0 on both sides; gating the upper term on `end > start` instead
        returns `-csum[start - 1]` for key 1 chip 0 (-10 here), which then blows
        up in `values.narrow`.
        """
        count = torch.tensor([0, 0, 2, 2], dtype=torch.long)  # chip0: 0, chip1: 4
        lengths = torch.tensor([1, 2, 3, 4, 1, 1, 2, 2], dtype=torch.long)
        values = torch.arange(1, 17, dtype=torch.long)  # 1..16, so 0 means pad

        result = pad_jagged_values(values, lengths, [24, 16], 0, -1, 1, count, False)

        self.assertEqual(result.shape, (40,))
        # key0 (budget/chip 12): chip0 owns nothing, chip1 owns all 10 values.
        self.assertEqual(result[:12].tolist(), [0] * 12)
        self.assertEqual(result[12:22].tolist(), list(range(1, 11)))
        self.assertEqual(result[22:24].tolist(), [0, 0])
        # key1 (budget/chip 8): same, 6 real values on chip1.
        self.assertEqual(result[24:32].tolist(), [0] * 8)
        self.assertEqual(result[32:38].tolist(), list(range(11, 17)))
        self.assertEqual(result[38:40].tolist(), [0, 0])


class PadJaggedPairToStaticTest(unittest.TestCase):
    @patch("recir.region_transforms.padding.runtime_num_local", return_value=2)
    def test_candidate_aware_values_and_lengths_share_chip_layout(
        self, _mock_num_local: Any
    ) -> None:
        count = torch.tensor([1, 2, 3, 1], dtype=torch.long)
        lengths = torch.ones(7, dtype=torch.long)
        values = torch.arange(7, dtype=torch.float32)

        padded_values, padded_lengths = pad_jagged_pair_to_static(
            values,
            lengths,
            per_key_budgets=[12],
            target_batch=10,
            count=count,
        )

        self.assertEqual(
            padded_values.tolist(),
            [0, 1, 2, 0, 0, 0, 3, 4, 5, 6, 0, 0],
        )
        self.assertEqual(
            padded_lengths.tolist(),
            [1, 1, 1, 2, 1, 1, 1, 1, 1, 2],
        )
        self.assertEqual(int(padded_lengths.sum().item()), padded_values.shape[0])


class PadCandidateAxisToStaticTest(unittest.TestCase):
    @patch("recir.region_transforms.padding.runtime_num_local", return_value=2)
    def test_builds_per_chip_real_value_mask(self, _mock_num_local: Any) -> None:
        count = torch.tensor([2, 1, 1, 3], dtype=torch.long)
        values = torch.ones(7, dtype=torch.float32)

        result = pad_candidate_axis_to_static(values, count, target_size=12)

        self.assertEqual(
            result.tolist(),
            [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0],
        )


class UnpadJaggedGradOpTest(unittest.TestCase):
    def test_extracts_per_key(self) -> None:
        from recir.region_transforms.padding import _PAD_JAGGED_SIZES

        _PAD_JAGGED_SIZES[300] = (25, [11, 14], 1)
        grad = torch.randn(40, 4)
        result = unpad_jagged_grad(grad, pad_id=300, per_key_budgets=[20, 20], dim=0)
        self.assertEqual(result.shape, (25, 4))
        torch.testing.assert_close(result[:11], grad[:11])
        torch.testing.assert_close(result[11:25], grad[20:34])


class PadCountToSumOpTest(unittest.TestCase):
    def test_shape_and_sum_aligned(self) -> None:
        x = torch.tensor(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 11], dtype=torch.long
        )
        # current_sum = 77 ; target_sum = 80 ; deficit = 3
        result = pad_count_to_sum(x, target_size=20, target_sum=80)
        self.assertEqual(result.shape, (20,))
        self.assertEqual(int(result.sum().item()), 80)

    def test_real_part_unchanged(self) -> None:
        x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.long)  # sum=36
        result = pad_count_to_sum(x, target_size=12, target_sum=44)
        torch.testing.assert_close(result[:8], x)

    def test_dummy_distributed_evenly(self) -> None:
        """Deficit should be split evenly across dummy entries (per_d, per_d+1
        only), not concentrated in the last entry."""
        x = torch.tensor(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 11], dtype=torch.long
        )
        result = pad_count_to_sum(x, target_size=20, target_sum=80)
        dummy = result[16:]  # 4 dummy entries, deficit=3
        self.assertEqual(int(dummy.sum().item()), 3)
        # per_d = 3 // 4 = 0, rem = 3 → first 3 get +1
        self.assertEqual(dummy.tolist(), [1, 1, 1, 0])

    def test_uniform_dummies_when_divisible(self) -> None:
        """deficit divisible by num_dummy → every dummy gets the same value."""
        x = torch.full((16,), 4, dtype=torch.long)  # sum=64
        result = pad_count_to_sum(x, target_size=20, target_sum=84)
        dummy = result[16:]
        # per_d = 20 // 4 = 5, rem = 0 → all dummies = 5
        self.assertEqual(dummy.tolist(), [5, 5, 5, 5])
        self.assertEqual(int(result.sum().item()), 84)

    def test_zero_deficit(self) -> None:
        """target_sum already matches → dummies should all be zero."""
        x = torch.tensor([1, 2, 3, 4], dtype=torch.long)  # sum=10
        result = pad_count_to_sum(x, target_size=8, target_sum=10)
        dummy = result[4:]
        self.assertEqual(dummy.tolist(), [0, 0, 0, 0])
        self.assertEqual(int(result.sum().item()), 10)

    def test_pad_id_records_original_size(self) -> None:
        from recir.region_transforms.padding import _PAD_ORIGINAL_SIZES

        x = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        pad_count_to_sum(x, target_size=8, target_sum=12, pad_id=500)
        self.assertEqual(_PAD_ORIGINAL_SIZES[500], 4)

    def test_num_processes_division(self) -> None:
        """target_size and target_sum are GLOBAL; runtime divides by num_processes."""
        x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.long)  # sum=36
        # global target_size=16, target_sum=80, num_processes=2
        # per_host_target_size=8 (==current → narrow no-op), per_host_target_sum=40
        # current_sum=36, deficit=4, but no dummy slots → bump last entry
        result = pad_count_to_sum(x, target_size=16, target_sum=80, num_processes=2)
        self.assertEqual(result.shape, (8,))
        self.assertEqual(int(result.sum().item()), 40)

    def test_fake_kernel_returns_global_shape(self) -> None:
        """Fake kernel uses GLOBAL target_size, not per_host."""
        with FakeTensorMode():
            x = torch.empty(8, dtype=torch.long)
            result = pad_count_to_sum(x, target_size=20, target_sum=80, num_processes=1)
            self.assertEqual(result.shape, (20,))


class LossDenomSourcePaddingTest(unittest.TestCase):
    """Guards the contract that makes the in-graph loss denominator correct.

    ``flatten_batch``'s trailing ``loss_denom_source`` slot carries the pre-pad
    per-sample candidate counts, and the model reduces it with ``.sum()`` inside
    the traced graph so XLA emits the cross-device all-reduce instead of the host
    paying a blocking gloo collective every step.

    That only works because a plain batch-axis tensor goes through
    ``pad_to_static`` with a ZERO fill, which preserves the sum. The same counts
    at position 8 go through ``pad_count_to_sum`` instead, whose dummies are
    deliberately nonzero. Summing THAT would return the padded budget constant.

    If ``loss_denom_source`` is ever added to PRT's ``skip_args``, or bound as a
    jagged pair's ``count_arg``, the denominator silently becomes wrong with no
    other assertion to catch it -- ``skip_args`` names are not validated.
    """

    def test_zero_fill_preserves_the_sum(self) -> None:
        counts = torch.tensor([3, 5, 2, 4], dtype=torch.int32)  # sum=14
        padded = pad_to_static(counts, target_size=8, dim=0)
        self.assertEqual(padded.shape, (8,))
        self.assertEqual(padded.tolist(), [3, 5, 2, 4, 0, 0, 0, 0])
        self.assertEqual(int(padded.sum()), int(counts.sum()))

    @patch("recir.region_transforms.padding.runtime_num_local", return_value=2)
    def test_zero_fill_preserves_the_sum_per_chip(self, _m: Any) -> None:
        counts = torch.tensor([3, 5, 2, 4], dtype=torch.int32)  # sum=14
        padded = pad_to_static(counts, target_size=8, dim=0)
        # Real rows stay at the front of EACH chip block, dummies are zero.
        self.assertEqual(padded.tolist(), [3, 5, 0, 0, 2, 4, 0, 0])
        self.assertEqual(int(padded.sum()), int(counts.sum()))

    def test_count_arg_padding_does_not_preserve_the_sum(self) -> None:
        """The contrast that motivates the separate slot."""
        counts = torch.tensor([3, 5, 2, 4], dtype=torch.int32)  # sum=14
        as_count_arg = pad_count_to_sum(counts, target_size=8, target_sum=20)
        self.assertEqual(int(as_count_arg.sum()), 20)
        self.assertNotEqual(int(as_count_arg.sum()), int(counts.sum()))


# ---------------------------------------------------------------------------
# Tests: mark_padding_graph_breaks
# ---------------------------------------------------------------------------


class MarkGraphBreaksTest(unittest.TestCase):
    def test_marks_padding_nodes(self) -> None:
        def fn(x: torch.Tensor) -> torch.Tensor:
            return torch.ops.recir_padding.pad_to_static(x, 8, 0)

        gm = make_fx(fn, tracing_mode="fake")(torch.randn(6, 16))
        mark_padding_graph_breaks(gm)

        found = False
        for node in gm.graph.nodes:
            if (
                node.op == "call_function"
                and node.target == torch.ops.recir_padding.pad_to_static.default
            ):
                self.assertTrue(node.meta.get("xla_graph_break", False))
                found = True
        self.assertTrue(found, "pad_to_static node not found in graph")


# ---------------------------------------------------------------------------
# Tests: auto-discovery
# ---------------------------------------------------------------------------


class DiscoveryTest(unittest.TestCase):
    def _make_tensors(
        self,
        batch_size: int = 6,
        uih_total: int = 47,
        nro_total: int = 19,
    ) -> List[torch.Tensor]:
        """User tensors (no model/optimizer):
        [ro_float, nro_float, ro_values, ro_lengths,
         nro_values, num_candidates, supervision_weight]
        """
        return [
            torch.randn(batch_size, 16),  # 0: ro_float (batch)
            torch.randn(nro_total, 8),  # 1: nro_float (derived from nro)
            torch.randn(uih_total, 32),  # 2: ro_values (jagged)
            _random_lengths(batch_size, uih_total),  # 3: ro_lengths
            torch.randn(nro_total, 32),  # 4: nro_values (jagged)
            _random_lengths(batch_size, nro_total),  # 5: num_candidates
            torch.ones(nro_total),  # 6: supervision_weight (derived)
        ]

    def _make_transform(self, **kwargs: Any) -> PaddingRegionTransform:
        defaults: dict[str, Any] = {
            "batch_size_arg": "ro_float",
            "jagged_pairs": {
                "ro_values": JaggedPairConfig("ro_lengths", per_sample_budgets=11),
                "nro_values": JaggedPairConfig("num_candidates", per_sample_budgets=6),
            },
        }
        defaults.update(kwargs)
        t = PaddingRegionTransform(**defaults)
        t.bind_signature(_dummy_train_step)
        return t

    def test_jagged_pairs_from_mapping(self) -> None:
        transform = self._make_transform()
        tensors = self._make_tensors()
        layout = transform._discover(tensors)

        values_indices = {jp.values_idx for jp in layout.jagged_pairs}
        lengths_indices = {jp.lengths_idx for jp in layout.jagged_pairs}
        self.assertIn(2, values_indices)
        self.assertIn(3, lengths_indices)
        self.assertIn(4, values_indices)
        self.assertIn(5, lengths_indices)

    def test_kps_inferred(self) -> None:
        transform = self._make_transform()
        tensors = self._make_tensors()
        layout = transform._discover(tensors)

        for jp in layout.jagged_pairs:
            self.assertEqual(len(jp.per_key_budgets), 1)

    def test_batch_indices(self) -> None:
        transform = self._make_transform()
        tensors = self._make_tensors()
        layout = transform._discover(tensors)
        self.assertIn(0, layout.batch_indices)

    def test_inferred_pad_targets(self) -> None:
        transform = self._make_transform(
            jagged_pairs={
                "nro_values": JaggedPairConfig("num_candidates", per_sample_budgets=6),
            },
        )
        tensors = self._make_tensors()
        layout = transform._discover(tensors)
        self.assertIn(1, layout.inferred_pad_targets)
        self.assertIn(6, layout.inferred_pad_targets)
        # batch_size=6, num_devices=1 (default), _resolved_pairs non-empty
        # → padded_bs = ceil(6/1)*1 = 6, == 6 → +1 → 7
        # nro pair: pair_padded_bs = 7 ; per_sample=6 → per_key = 6*7 = 42
        self.assertEqual(layout.inferred_pad_targets[1], 42)

    def test_batch_size_from_arg(self) -> None:
        transform = self._make_transform(
            jagged_pairs={},
            num_devices=4,
        )
        tensors = self._make_tensors(batch_size=6)
        layout = transform._discover(tensors)
        self.assertEqual(layout.batch_size, 6)
        self.assertEqual(layout.padded_bs, 8)

    def test_padded_batch_size(self) -> None:
        transform = self._make_transform(
            jagged_pairs={},
            num_devices=4,
        )
        tensors = self._make_tensors(batch_size=6)
        layout = transform._discover(tensors)
        self.assertEqual(layout.padded_bs, 8)

    def test_no_item_calls_safe_on_faketensors(self) -> None:
        """_discover uses only shape-based analysis, no .item()."""
        transform = self._make_transform()
        real_tensors = self._make_tensors()
        with FakeTensorMode() as fake_mode:
            fake_tensors = [fake_mode.from_tensor(t) for t in real_tensors]
            layout = transform._discover(fake_tensors)  # pyre-ignore

        self.assertEqual(layout.batch_size, 6)
        self.assertEqual(len(layout.jagged_pairs), 2)

    def test_bind_signature(self) -> None:
        """bind_signature resolves string names to indices."""
        transform = PaddingRegionTransform(
            jagged_pairs={
                "ro_values": JaggedPairConfig("ro_lengths", per_sample_budgets=10),
                "nro_values": JaggedPairConfig("num_candidates", per_sample_budgets=5),
            },
            batch_size_arg="ro_float",
        )
        transform.bind_signature(_dummy_train_step)
        self.assertEqual(transform.batch_size_arg_idx, 0)
        # resolved_pairs: [(v_idx, l_idx, bs_idx, per_sample, count_idx, max_count)]
        resolved = {
            v: (l, bs, ps, c, mc) for v, l, bs, ps, c, mc in transform._resolved_pairs
        }
        self.assertEqual(resolved[2], (3, None, 10, None, None))  # ro_values
        self.assertEqual(resolved[4], (5, None, 5, None, None))  # nro_values

    def test_bind_signature_with_count_arg(self) -> None:
        """count_arg resolves to count_idx in the resolved pair tuple."""
        transform = PaddingRegionTransform(
            jagged_pairs={
                "nro_values": JaggedPairConfig(
                    "ro_lengths",
                    per_sample_budgets=1,
                    batch_size_arg="nro_float",
                    count_arg="num_candidates",
                    max_per_sample_count=8,
                ),
            },
            batch_size_arg="ro_float",
        )
        transform.bind_signature(_dummy_train_step)
        # Single pair: (v_idx=4, l_idx=3, bs_idx=1, ps=1, count_idx=5, max_count=8)
        v_idx, l_idx, bs_idx, ps, count_idx, max_count = transform._resolved_pairs[0]
        self.assertEqual(v_idx, 4)  # nro_values
        self.assertEqual(l_idx, 3)  # ro_lengths
        self.assertEqual(bs_idx, 1)  # nro_float
        self.assertEqual(ps, 1)
        self.assertEqual(count_idx, 5)  # num_candidates
        self.assertEqual(max_count, 8)


class BindSignatureValidationTest(unittest.TestCase):
    """bind_signature must reject configs that can't keep sum(count) aligned."""

    def test_bs_idx_without_count_idx_raises(self) -> None:
        transform = PaddingRegionTransform(
            jagged_pairs={
                "nro_values": JaggedPairConfig(
                    "ro_lengths",
                    per_sample_budgets=1,
                    batch_size_arg="nro_float",  # set bs_idx
                    # count_arg intentionally omitted
                ),
            },
            batch_size_arg="ro_float",
        )
        with self.assertRaisesRegex(ValueError, "count_arg"):
            transform.bind_signature(_dummy_train_step)

    def test_bs_idx_with_count_idx_ok(self) -> None:
        transform = PaddingRegionTransform(
            jagged_pairs={
                "nro_values": JaggedPairConfig(
                    "ro_lengths",
                    per_sample_budgets=1,
                    batch_size_arg="nro_float",
                    count_arg="num_candidates",
                    max_per_sample_count=8,
                ),
            },
            batch_size_arg="ro_float",
        )
        # Should NOT raise.
        transform.bind_signature(_dummy_train_step)

    def test_bs_idx_without_max_count_raises(self) -> None:
        transform = PaddingRegionTransform(
            jagged_pairs={
                "nro_values": JaggedPairConfig(
                    "ro_lengths",
                    per_sample_budgets=1,
                    batch_size_arg="nro_float",
                    count_arg="num_candidates",
                    # max_per_sample_count intentionally omitted
                ),
            },
            batch_size_arg="ro_float",
        )
        with self.assertRaisesRegex(ValueError, "max_per_sample_count"):
            transform.bind_signature(_dummy_train_step)

    def test_no_bs_idx_no_count_idx_ok(self) -> None:
        """Pairs without bs_idx (B_pair == B_main) don't need count_arg."""
        transform = PaddingRegionTransform(
            jagged_pairs={
                "ro_values": JaggedPairConfig("ro_lengths", per_sample_budgets=1),
            },
            batch_size_arg="ro_float",
        )
        # Should NOT raise.
        transform.bind_signature(_dummy_train_step)


# ---------------------------------------------------------------------------
# Tests: transform_inputs
# ---------------------------------------------------------------------------


def _trace_transform_inputs(
    transform: PaddingRegionTransform,
    tensors: Tuple[torch.Tensor, ...],
) -> torch.fx.GraphModule:
    """Trace ``transform.transform_inputs(args, {})`` under make_fx so the
    custom-op calls inside it become FX nodes — same dispatch path that
    fires when ``_local_compile``'s outer ``make_fx`` enters
    ``model(*args, **kwargs)`` via the patched forward."""
    fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
    fake_tensors = tuple(fake_mode.from_tensor(t) for t in tensors)

    def fn(*args: Any) -> Any:
        new_args, _ = transform.transform_inputs(args, {})
        return new_args

    with fake_mode:
        gm = make_fx(fn, tracing_mode="fake", _allow_non_fake_inputs=True)(
            *fake_tensors
        )
    return gm


def _pad_targets(gm: torch.fx.GraphModule, op: Any) -> List[Any]:  # pyre-ignore[24]
    return [n for n in gm.graph.nodes if n.op == "call_function" and n.target == op]


class TransformInputsTest(unittest.TestCase):
    """transform_inputs replaces transform_graph as the single API.

    These tests verify that pad ops are dispatched (and so captured by an
    active make_fx) when transform_inputs runs under tracing.
    """

    def _make_tensors(
        self,
        batch_size: int = 6,
        uih_total: int = 47,
        nro_total: int = 19,
    ) -> Tuple[torch.Tensor, ...]:
        return (
            torch.randn(batch_size, 16),  # 0 ro_float
            torch.randn(nro_total, 8),  # 1 nro_float (derived)
            torch.randn(uih_total, 32),  # 2 ro_values (jagged)
            _random_lengths(batch_size, uih_total),  # 3 ro_lengths
            torch.randn(nro_total, 32),  # 4 nro_values (jagged)
            _random_lengths(batch_size, nro_total),  # 5 num_candidates
            torch.ones(nro_total),  # 6 supervision_weight (derived)
        )

    def _make_transform(self) -> PaddingRegionTransform:
        # batch_size=6, num_devices=4 → padded_bs = ceil(6/4)*4 = 8
        # ro: per_sample=8 → per_key_total = 8 * 8 = 64
        # nro: per_sample=4 → per_key_total = 4 * 8 = 32
        t = PaddingRegionTransform(
            jagged_pairs={
                "ro_values": JaggedPairConfig("ro_lengths", per_sample_budgets=8),
                "nro_values": JaggedPairConfig("num_candidates", per_sample_budgets=4),
            },
            batch_size_arg="ro_float",
            num_devices=4,
        )
        t.bind_signature(_dummy_train_step)
        return t

    def test_pad_to_static_nodes_inserted(self) -> None:
        gm = _trace_transform_inputs(self._make_transform(), self._make_tensors())
        nodes = _pad_targets(gm, torch.ops.recir_padding.pad_to_static.default)
        self.assertGreater(len(nodes), 0)

    def test_pad_jagged_lengths_nodes_inserted(self) -> None:
        gm = _trace_transform_inputs(self._make_transform(), self._make_tensors())
        nodes = _pad_targets(gm, torch.ops.recir_padding.pad_jagged_lengths.default)
        self.assertGreater(len(nodes), 0)

    def test_pad_jagged_values_nodes_inserted(self) -> None:
        gm = _trace_transform_inputs(self._make_transform(), self._make_tensors())
        nodes = _pad_targets(gm, torch.ops.recir_padding.pad_jagged_values.default)
        self.assertGreater(len(nodes), 0)

    def test_batch_pad_target(self) -> None:
        """Batch tensor (ro_float) should be padded to padded_bs=8."""
        gm = _trace_transform_inputs(self._make_transform(), self._make_tensors())
        nodes = _pad_targets(gm, torch.ops.recir_padding.pad_to_static.default)
        targets = {n.args[1] for n in nodes}
        self.assertIn(8, targets)

    def test_jagged_values_target_size(self) -> None:
        """ro per_key=64 (8*8); nro per_key=32 (4*8)."""
        gm = _trace_transform_inputs(self._make_transform(), self._make_tensors())
        jv_nodes = _pad_targets(gm, torch.ops.recir_padding.pad_jagged_values.default)
        budgets_first = {n.args[2][0] for n in jv_nodes}
        self.assertIn(64, budgets_first)
        self.assertIn(32, budgets_first)

    def test_kwargs_passed_through(self) -> None:
        transform = self._make_transform()
        tensors = self._make_tensors()
        sentinel = {"untouched": torch.tensor([1, 2, 3])}
        _, new_kwargs = transform.transform_inputs(tuple(tensors), sentinel)
        self.assertIs(new_kwargs, sentinel)

    def test_graph_lint_passes(self) -> None:
        gm = _trace_transform_inputs(self._make_transform(), self._make_tensors())
        gm.graph.lint()

    def test_no_padding_when_aligned_no_pairs(self) -> None:
        """No jagged pairs + already-aligned batch → no pad ops dispatched."""
        transform = PaddingRegionTransform(
            jagged_pairs={},
            batch_size_arg="ro_float",
            # num_devices=1 (default) → padded_bs == batch_size
        )
        transform.bind_signature(_dummy_train_step)
        gm = _trace_transform_inputs(
            transform, self._make_tensors(batch_size=1, uih_total=47, nro_total=19)
        )
        for op in (
            torch.ops.recir_padding.pad_to_static.default,
            torch.ops.recir_padding.pad_jagged_lengths.default,
            torch.ops.recir_padding.pad_jagged_values.default,
            torch.ops.recir_padding.pad_count_to_sum.default,
        ):
            self.assertEqual(_pad_targets(gm, op), [])


# ---------------------------------------------------------------------------
# Tests: transform_inputs with multi-key jagged
# ---------------------------------------------------------------------------


def _multikey_train_step(
    mod: nn.Module,
    opt: torch.optim.Optimizer,
    dense: torch.Tensor,
    mk_values: torch.Tensor,
    mk_lengths: torch.Tensor,
) -> None:
    pass


class TransformInputsMultiKeyTest(unittest.TestCase):
    def _make_tensors(
        self, B: int = 4, key_totals: List[int] | None = None
    ) -> Tuple[torch.Tensor, ...]:
        if key_totals is None:
            key_totals = [11, 14, 9]
        key_lens = [_random_lengths(B, t) for t in key_totals]
        lengths = torch.cat(key_lens)
        values = torch.randn(sum(key_totals), 8)
        return (torch.randn(B, 16), values, lengths)

    def test_pad_jagged_values_node_inserted(self) -> None:
        """3 keys → single pad_jagged_values node with per-key budgets list."""
        t = PaddingRegionTransform(
            jagged_pairs={
                "mk_values": JaggedPairConfig("mk_lengths", per_sample_budgets=5),
            },
            batch_size_arg="dense",
            num_devices=4,
        )
        t.bind_signature(_multikey_train_step)
        gm = _trace_transform_inputs(t, self._make_tensors(B=4, key_totals=[11, 14, 9]))

        jv_nodes = _pad_targets(gm, torch.ops.recir_padding.pad_jagged_values.default)
        self.assertEqual(len(jv_nodes), 1)
        # B=4, num_devices=4 → padded_bs = 4, == 4 → +4 → 8
        # per_key = 5 * 8 = 40 for each of 3 keys
        self.assertEqual(jv_nodes[0].args[2], [40, 40, 40])

    def test_per_key_different_budgets(self) -> None:
        t = PaddingRegionTransform(
            jagged_pairs={
                "mk_values": JaggedPairConfig("mk_lengths", per_sample_budgets=[5, 8]),
            },
            batch_size_arg="dense",
            num_devices=4,
        )
        t.bind_signature(_multikey_train_step)
        gm = _trace_transform_inputs(t, self._make_tensors(B=4, key_totals=[11, 14]))

        jv_nodes = _pad_targets(gm, torch.ops.recir_padding.pad_jagged_values.default)
        self.assertEqual(len(jv_nodes), 1)
        # padded_bs = 8 → per_key = [5*8, 8*8] = [40, 64]
        self.assertEqual(jv_nodes[0].args[2], [40, 64])

    def test_returned_values_shape(self) -> None:
        t = PaddingRegionTransform(
            jagged_pairs={
                "mk_values": JaggedPairConfig("mk_lengths", per_sample_budgets=[5, 8]),
            },
            batch_size_arg="dense",
            num_devices=4,
        )
        t.bind_signature(_multikey_train_step)
        tensors = self._make_tensors(B=4, key_totals=[11, 14])
        new_args, _ = t.transform_inputs(tensors, {})
        # values: sum(per_key_budgets) = 40 + 64 = 104
        self.assertEqual(new_args[1].shape, (104, 8))
        # lengths: target_batch * num_keys = 8 * 2 = 16
        self.assertEqual(new_args[2].shape, (16,))


# ---------------------------------------------------------------------------
# Tests: integration with bs_idx + count_arg (jagged candidates flow)
# ---------------------------------------------------------------------------


def _roo_train_step(
    mod: nn.Module,
    opt: torch.optim.Optimizer,
    ro_float: torch.Tensor,
    nro_float: torch.Tensor,
    nro_values: torch.Tensor,
    nro_lengths: torch.Tensor,
    num_candidates: torch.Tensor,
) -> None:
    pass


class BsIdxCountArgIntegrationTest(unittest.TestCase):
    """End-to-end: nro pair with its own batch dim + jagged candidates.

    pair_bs is now derived from the upper bound (batch_size * max_per_sample_count),
    not the jagged trace-time tensor sum, so the padded shape is stable across
    uniform / jagged trace batches.
    """

    MAX_C: int = 11

    def _make_transform(self) -> PaddingRegionTransform:
        t = PaddingRegionTransform(
            jagged_pairs={
                "nro_values": JaggedPairConfig(
                    "nro_lengths",
                    per_sample_budgets=1,
                    batch_size_arg="nro_float",
                    count_arg="num_candidates",
                    max_per_sample_count=self.MAX_C,
                ),
            },
            batch_size_arg="ro_float",
            num_devices=4,
        )
        t.bind_signature(_roo_train_step)
        return t

    def _make_tensors(self) -> Tuple[torch.Tensor, ...]:
        # B_RO=16, jagged num_candidates summing to B_NRO=77 (max=11).
        # padded_bs = ceil(16/4)*4 = 16, == B_RO → +4 = 20.
        # pair_padded_bs = padded_bs * max_c = 20 * 11 = 220.
        num_candidates = torch.tensor(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 11], dtype=torch.long
        )
        B_NRO = int(num_candidates.sum().item())  # 77
        num_keys = 6
        return (
            torch.randn(16, 8),  # ro_float
            torch.randn(B_NRO, 4),  # nro_float
            torch.randn(B_NRO * num_keys, 2),  # nro_values
            # nro_lengths: 1 per item per key
            torch.full((B_NRO * num_keys,), 1, dtype=torch.long),
            num_candidates,
        )

    def test_pair_padded_bs_uses_padded_bs_times_max_count(self) -> None:
        """pair_padded_bs = padded_bs * max_count = 20 * 11 = 220.

        Override construction uses ``padded_ro_bs * num_candidates_per_sample``
        for ``padded_nro_bs``; PRT mirrors that so trace-time pair shape and
        runtime pair shape agree.
        """
        t = self._make_transform()
        new_args, _ = t.transform_inputs(self._make_tensors(), {})
        # nro_float: 77 → 220
        self.assertEqual(new_args[1].shape, (220, 4))

    def test_count_arg_uses_pad_count_to_sum(self) -> None:
        """num_candidates should be padded by pad_count_to_sum."""
        gm = _trace_transform_inputs(self._make_transform(), self._make_tensors())
        cnt_nodes = _pad_targets(gm, torch.ops.recir_padding.pad_count_to_sum.default)
        self.assertGreater(len(cnt_nodes), 0)

    def test_padded_num_candidates_sum_matches(self) -> None:
        """sum(padded num_candidates) == pair_padded_bs — the cross-tensor
        invariant that pad_count_to_sum exists to maintain."""
        t = self._make_transform()
        new_args, _ = t.transform_inputs(self._make_tensors(), {})
        # padded num_candidates is at idx 4 (last positional arg)
        padded_count = new_args[4]
        # B_RO=16, padded_bs=20; pair_padded_bs = padded_bs * max_c = 220
        self.assertEqual(padded_count.shape, (20,))
        self.assertEqual(int(padded_count.sum().item()), 220)

    def test_dummy_count_distributed_evenly(self) -> None:
        """Deficit (220-77=143) split across the 4 dummy entries:
        per_d=35, rem=3 → first 3 dummies get +1."""
        t = self._make_transform()
        new_args, _ = t.transform_inputs(self._make_tensors(), {})
        padded_count = new_args[4]
        dummy = padded_count[16:]
        self.assertEqual(dummy.tolist(), [36, 36, 36, 35])

    def test_no_count_arg_then_bs_idx_rejected(self) -> None:
        """Sanity: without count_arg the same config is rejected."""
        t = PaddingRegionTransform(
            jagged_pairs={
                "nro_values": JaggedPairConfig(
                    "nro_lengths",
                    per_sample_budgets=1,
                    batch_size_arg="nro_float",
                ),
            },
            batch_size_arg="ro_float",
            num_devices=4,
        )
        with self.assertRaisesRegex(ValueError, "count_arg"):
            t.bind_signature(_roo_train_step)


if __name__ == "__main__":
    unittest.main()
