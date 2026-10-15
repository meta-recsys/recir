# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""CPU-runnable tests for the pooled (EBC) layer over the ringshift EC kernel.

The production entry point defaults to fused Pallas pooling. A compile-time
environment setting selects the XLA post-reduction reference. This file tests
the shared primitives and everything in both paths that can be checked without
executing the real-TPU-only kernel.

Expected values are hand-computed inline rather than derived from a second
implementation, so a bug in the reference cannot hide a bug in the code.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from recir.ops.pallas.embeddings.ringshift import (
    _bucketize_and_sort,
    _pad_per_peer_2d,
    _POOLING_IMPL_ENV,
    _pooling_impl_from_env,
    _round_up_tile,
    _TILE_INT32,
    apply_pooling_divisor,
    fan_out_pooled_grad,
    lengths_to_segment_ids,
    pool_per_rank,
    ringshift_embedding_lookup,
    ringshift_embedding_lookup_reference,
    sorted_padded_segment_ids,
)

# per_id_emb[i] = [i + 1, -(i + 1)] -- distinct, sign-split, trivial to sum.
_MAX_LOCAL_IDS = 8
_BASE = (np.arange(_MAX_LOCAL_IDS) + 1).astype(np.float32)
_PER_ID_EMB = jnp.array(np.stack([_BASE, -_BASE], axis=1), dtype=jnp.float32)


class LengthsToSegmentIdsTest(unittest.TestCase):
    def test_maps_slots_to_bags_and_padding_to_trash(self) -> None:
        # lengths [3, 2, 1] -> offsets [3, 5, 6]; slots 6,7 are padding.
        seg = lengths_to_segment_ids(jnp.array([3, 2, 1], jnp.int32), _MAX_LOCAL_IDS)
        self.assertEqual(list(np.array(seg)), [0, 0, 0, 1, 1, 2, 3, 3])

    def test_empty_bag_claims_no_slots(self) -> None:
        # Bag 1 is empty, so offsets [3, 3, 5] collide at 3 and the scatter-add
        # bumps the segment id by 2 at once -- slot 3 belongs to bag 2.
        seg = lengths_to_segment_ids(jnp.array([3, 0, 2], jnp.int32), _MAX_LOCAL_IDS)
        self.assertEqual(list(np.array(seg)), [0, 0, 0, 2, 2, 3, 3, 3])

    def test_exact_fit_needs_no_trash_segment(self) -> None:
        # offsets [4, 8]; the trailing 8 lands out of range and is dropped.
        seg = lengths_to_segment_ids(jnp.array([4, 4], jnp.int32), _MAX_LOCAL_IDS)
        self.assertEqual(list(np.array(seg)), [0, 0, 0, 0, 1, 1, 1, 1])

    def test_all_bags_empty_sends_everything_to_trash(self) -> None:
        seg = lengths_to_segment_ids(jnp.array([0, 0, 0], jnp.int32), _MAX_LOCAL_IDS)
        self.assertEqual(list(np.array(seg)), [3] * _MAX_LOCAL_IDS)

    def test_segment_ids_are_non_decreasing(self) -> None:
        # pool_per_rank passes indices_are_sorted=True to segment_sum; that is
        # only sound because this holds.
        seg = np.array(
            lengths_to_segment_ids(jnp.array([2, 0, 1, 3], jnp.int32), _MAX_LOCAL_IDS)
        )
        np.testing.assert_array_equal(np.sort(seg), seg)


class ApplyPoolingDivisorTest(unittest.TestCase):
    """The ``mean`` step, tested apart from either reduction that feeds it.

    Both paths -- ``pool_per_rank`` and the fused Phase 4 accumulator -- produce
    a per-bag sum and then call this. Testing it directly is what makes a path
    that *forgets* to call it a caught bug rather than a silent one: omitting it
    returns the correct shape with un-normalized values, so nothing in a shape
    or trace-level assertion bites.
    """

    _SUMS = jnp.array([[6.0, -6.0], [0.0, 0.0], [9.0, -9.0]])
    _LENGTHS = jnp.array([3, 0, 2], jnp.int32)

    def test_sum_passes_the_value_through(self) -> None:
        np.testing.assert_array_equal(
            np.array(apply_pooling_divisor(self._SUMS, self._LENGTHS, pooling="sum")),
            np.array(self._SUMS),
        )

    def test_mean_divides_each_bag_by_its_own_length(self) -> None:
        np.testing.assert_allclose(
            np.array(apply_pooling_divisor(self._SUMS, self._LENGTHS, pooling="mean")),
            np.array([[2.0, -2.0], [0.0, 0.0], [4.5, -4.5]]),
        )

    def test_mean_actually_changes_the_value(self) -> None:
        """Guards the specific mutation of returning the raw sum for ``mean``."""
        pooled = apply_pooling_divisor(self._SUMS, self._LENGTHS, pooling="mean")
        self.assertFalse(np.allclose(np.array(pooled), np.array(self._SUMS)))

    def test_empty_bag_is_zero_not_nan(self) -> None:
        pooled = np.array(
            apply_pooling_divisor(self._SUMS, self._LENGTHS, pooling="mean")
        )
        self.assertFalse(np.isnan(pooled).any())

    def test_empty_bag_gradient_is_finite(self) -> None:
        """The clamp's second job: an unclamped 1/0 makes the VJP non-finite."""

        def total(sums: jax.Array) -> jax.Array:
            return jnp.sum(apply_pooling_divisor(sums, self._LENGTHS, pooling="mean"))

        grad = np.array(jax.grad(total)(self._SUMS))
        self.assertTrue(np.isfinite(grad).all())


class PoolPerRankForwardTest(unittest.TestCase):
    def test_sum(self) -> None:
        # bag0 = rows 0,1,2 -> 1+2+3 = 6; bag1 empty; bag2 = rows 3,4 -> 4+5 = 9.
        pooled = pool_per_rank(
            _PER_ID_EMB, jnp.array([3, 0, 2], jnp.int32), pooling="sum"
        )
        np.testing.assert_allclose(
            np.array(pooled), np.array([[6.0, -6.0], [0.0, 0.0], [9.0, -9.0]])
        )

    def test_mean(self) -> None:
        pooled = pool_per_rank(
            _PER_ID_EMB, jnp.array([3, 0, 2], jnp.int32), pooling="mean"
        )
        np.testing.assert_allclose(
            np.array(pooled), np.array([[2.0, -2.0], [0.0, 0.0], [4.5, -4.5]])
        )

    def test_empty_bag_is_zero_not_nan(self) -> None:
        """Mean's divisor is clamped to 1, so an empty bag yields 0/1, not 0/0."""
        pooled = np.array(
            pool_per_rank(_PER_ID_EMB, jnp.array([0, 8], jnp.int32), pooling="mean")
        )
        self.assertFalse(np.isnan(pooled).any())
        np.testing.assert_allclose(pooled[0], np.zeros(2))

    def test_padding_rows_are_ignored(self) -> None:
        """Correctness must not depend on the kernel zeroing its padding rows.

        Padding slots route to the trash segment, so poisoning them changes
        nothing.
        """
        poisoned = _PER_ID_EMB.at[5:, :].set(1e6)
        lengths = jnp.array([3, 0, 2], jnp.int32)  # 5 valid ids, slots 5..7 padding
        np.testing.assert_allclose(
            np.array(pool_per_rank(poisoned, lengths, pooling="sum")),
            np.array(pool_per_rank(_PER_ID_EMB, lengths, pooling="sum")),
        )


class PoolingArgumentValidationTest(unittest.TestCase):
    """``pooling`` and ``bag_lengths`` are coupled; the entry points enforce it.

    Checked through the numpy reference because it shares ``_check_pooling``
    with :func:`ringshift_embedding_lookup` but needs no TPU or mesh.
    """

    IDS = np.array([[1, 2, 0, 0]], np.int32)
    COUNT = np.array([2], np.int32)
    TABLE = np.zeros((1, 4, 2), np.float32)
    LENGTHS = np.array([[2]], np.int32)

    def test_rejects_unknown_mode(self) -> None:
        with self.assertRaises(ValueError):
            ringshift_embedding_lookup_reference(
                self.IDS, None, self.TABLE, 1, "max", self.LENGTHS
            )

    def test_rejects_pooling_without_bag_lengths(self) -> None:
        with self.assertRaises(ValueError):
            ringshift_embedding_lookup_reference(
                self.IDS, None, self.TABLE, 1, "sum", None
            )

    def test_rejects_bag_lengths_without_pooling(self) -> None:
        """Bags but no mode would silently return unpooled rows."""
        with self.assertRaises(ValueError):
            ringshift_embedding_lookup_reference(
                self.IDS, self.COUNT, self.TABLE, 1, "none", self.LENGTHS
            )

    def test_rejects_id_count_alongside_pooling(self) -> None:
        """The redundancy itself is rejected, because a mismatch is uncheckable.

        Under ``jit`` both are tracers, so ``sum(bag_lengths) != id_count``
        cannot be asserted at trace time; it would silently drop ids or inflate
        the mean divisor. Refusing the second copy removes the failure mode.
        """
        with self.assertRaises(ValueError):
            ringshift_embedding_lookup_reference(
                self.IDS, self.COUNT, self.TABLE, 1, "sum", self.LENGTHS
            )

    def test_rejects_missing_id_count_without_pooling(self) -> None:
        with self.assertRaises(ValueError):
            ringshift_embedding_lookup_reference(
                self.IDS, None, self.TABLE, 1, "none", None
            )


class PoolPerRankBackwardTest(unittest.TestCase):
    """The pooled backward is what ``jax.grad`` derives; no custom_vjp involved."""

    @staticmethod
    def _grad_of_total(bag_lengths: jax.Array, pooling: str) -> np.ndarray:
        # loss = sum(pooled) makes d_loss/d_pooled all-ones, so the resulting
        # per-id grad is exactly the fan-out factor for each slot.
        def loss_fn(emb: jax.Array) -> jax.Array:
            return jnp.sum(pool_per_rank(emb, bag_lengths, pooling=pooling))

        return np.array(jax.grad(loss_fn)(_PER_ID_EMB))

    def test_sum_grad_fans_out_evenly(self) -> None:
        grad = self._grad_of_total(jnp.array([3, 0, 2], jnp.int32), "sum")
        expected = np.zeros((_MAX_LOCAL_IDS, 2), np.float32)
        expected[:5] = 1.0  # every member of a bag gets the bag's grad verbatim
        np.testing.assert_allclose(grad, expected)

    def test_mean_grad_scales_by_bag_length(self) -> None:
        grad = self._grad_of_total(jnp.array([3, 0, 2], jnp.int32), "mean")
        expected = np.zeros((_MAX_LOCAL_IDS, 2), np.float32)
        expected[0:3] = 1.0 / 3.0  # bag 0 has 3 members
        expected[3:5] = 1.0 / 2.0  # bag 2 has 2 members
        np.testing.assert_allclose(grad, expected, rtol=1e-6)

    def test_padding_rows_get_zero_grad(self) -> None:
        """Trash-segment slots are sliced off, so no gradient flows back to them."""
        grad = self._grad_of_total(jnp.array([3, 0, 2], jnp.int32), "sum")
        np.testing.assert_allclose(grad[5:], np.zeros((3, 2), np.float32))


class ReferenceTest(unittest.TestCase):
    """The numpy reference, checked against hand-computed values.

    Table: global row ``i`` is ``[i, 10*i]``, MOD-sharded over W=2 so
    ``shard[r][j] == full[j * 2 + r]``. Getting this mapping backwards is the
    single most common way to spend a day chasing a phantom kernel bug.
    """

    def setUp(self) -> None:
        world_size, v_local = 2, 4
        full = np.stack(
            [np.arange(8, dtype=np.float32), 10.0 * np.arange(8, dtype=np.float32)],
            axis=1,
        )
        self.full = full
        self.sharded = np.stack(
            [full[np.arange(v_local) * world_size + r] for r in range(world_size)]
        )
        # rank 0: bags [5, 2] and [7];  rank 1: bag 0 empty, bag 1 = [1, 3, 4].
        self.ids = np.array([[5, 2, 7, 0, 0, 0], [1, 3, 4, 0, 0, 0]], np.int32)
        self.bag_lengths = np.array([[2, 1], [0, 3]], np.int32)
        self.id_count = self.bag_lengths.sum(axis=1).astype(np.int32)

    def test_pooling_none_is_the_unchanged_ec_path(self) -> None:
        """Default mode must be byte-for-byte the pre-pooling behavior.

        ``pooling`` and ``bag_lengths`` were added to an entry point the EC path
        already uses, so the no-pooling default is the regression guard for
        every existing caller.
        """
        out = ringshift_embedding_lookup_reference(
            self.ids, self.id_count, self.sharded, 2
        )
        self.assertEqual(out.shape, (2, 6, 2))
        # Row-per-id, caller order, padding zeroed.
        np.testing.assert_array_equal(out[0, 0], self.full[5])
        np.testing.assert_array_equal(out[0, 1], self.full[2])
        np.testing.assert_array_equal(out[0, 2], self.full[7])
        np.testing.assert_array_equal(out[0, 3:], np.zeros((3, 2), np.float32))

    def test_padding_ids_past_the_bag_total_are_never_read(self) -> None:
        """The derived id count gates validity, so junk past it cannot leak in.

        Poisons every slot past ``sum(bag_lengths)`` with an out-of-range id. If
        the reference ever read beyond the bag total this would IndexError
        rather than silently return a wrong-but-plausible answer.
        """
        poisoned = self.ids.copy()
        poisoned[:, 3:] = 99999
        np.testing.assert_allclose(
            ringshift_embedding_lookup_reference(
                poisoned, None, self.sharded, 2, "sum", self.bag_lengths
            ),
            ringshift_embedding_lookup_reference(
                self.ids, None, self.sharded, 2, "sum", self.bag_lengths
            ),
        )

    def test_mod_shard_mapping(self) -> None:
        np.testing.assert_allclose(self.sharded[0], self.full[[0, 2, 4, 6]])
        np.testing.assert_allclose(self.sharded[1], self.full[[1, 3, 5, 7]])

    def test_sum(self) -> None:
        out = ringshift_embedding_lookup_reference(
            self.ids, None, self.sharded, 2, "sum", self.bag_lengths
        )
        expected = np.array(
            [
                [[7.0, 70.0], [7.0, 70.0]],  # 5+2 = 7 ; 7
                [[0.0, 0.0], [8.0, 80.0]],  # empty ; 1+3+4 = 8
            ]
        )
        np.testing.assert_allclose(out, expected)

    def test_mean(self) -> None:
        out = ringshift_embedding_lookup_reference(
            self.ids, None, self.sharded, 2, "mean", self.bag_lengths
        )
        expected = np.array(
            [
                [[3.5, 35.0], [7.0, 70.0]],
                [[0.0, 0.0], [8.0 / 3.0, 80.0 / 3.0]],
            ]
        )
        np.testing.assert_allclose(out, expected, rtol=1e-6)

    def test_matches_jax_pooling_layer(self) -> None:
        """The numpy reference and the JAX layer must agree slot for slot."""
        per_id = np.zeros((2, 6, 2), np.float32)
        for r in range(2):
            for i in range(int(self.bag_lengths[r].sum())):
                per_id[r, i] = self.full[self.ids[r, i]]
        from_jax = np.stack(
            [
                np.array(
                    pool_per_rank(
                        jnp.array(per_id[r]),
                        jnp.array(self.bag_lengths[r]),
                        pooling="sum",
                    )
                )
                for r in range(2)
            ]
        )
        np.testing.assert_allclose(
            from_jax,
            ringshift_embedding_lookup_reference(
                self.ids, None, self.sharded, 2, "sum", self.bag_lengths
            ),
        )


class SortedPaddedSegmentIdsTest(unittest.TestCase):
    """The invariant that makes the fused path (Option A) correct.

    Option A never restores caller order: it scatter-adds rows where they land,
    in the kernel's sorted+padded layout. That is only sound because summation
    is order-independent AND the bag labels are permuted the same way the rows
    were. This class pins both halves down on CPU, which is the part of Option A
    that does not need a TPU.
    """

    W = 4
    MAX_LOCAL_IDS = 8
    MAX_PADDED = MAX_LOCAL_IDS + W * _TILE_INT32

    def test_matches_the_docstring_example(self) -> None:
        """The worked example in the docstring, asserted so it cannot drift."""
        seg = sorted_padded_segment_ids(
            jnp.array([2, 1, 2], jnp.int32),  # bag_lengths
            jnp.array([1, 3, 0, 2, 4, 5], jnp.int32),  # sort_perm
            jnp.array([2, 3], jnp.int32),  # send_counts_real
            jnp.array([2, 4], jnp.int32),  # send_counts_padded
            max_local_ids=6,
            max_padded=10,
            world_size=2,
        )
        self.assertEqual(list(np.array(seg)), [0, 2, 0, 1, 2, 3, 3, 3, 3, 3])

    def setUp(self) -> None:
        # ids chosen so the destination buckets are uneven: 5->1, 2->2, 7->3,
        # 12->0, 3->3. Bags [2, 1, 2] over 5 valid ids, slots 5..7 padding.
        self.ids = jnp.array([5, 2, 7, 12, 3, 0, 0, 0], jnp.int32)
        self.bag_lengths = jnp.array([2, 1, 2], jnp.int32)
        self.id_count = jnp.sum(self.bag_lengths)
        (
            self.sorted_row,
            self.send_counts_real,
            self.sort_perm,
            self.reverse_perm,
        ) = _bucketize_and_sort(self.ids, self.id_count, self.MAX_LOCAL_IDS, self.W)
        self.send_counts_padded = _round_up_tile(self.send_counts_real, _TILE_INT32)

    def _seg_sorted_padded(self, slack: int = 0) -> jax.Array:
        return sorted_padded_segment_ids(
            self.bag_lengths,
            self.sort_perm,
            self.send_counts_real,
            self.send_counts_padded,
            max_local_ids=self.MAX_LOCAL_IDS,
            max_padded=self.MAX_PADDED,
            world_size=self.W,
            slack=slack,
        )

    def test_shape_and_range(self) -> None:
        seg = np.array(self._seg_sorted_padded())
        self.assertEqual(seg.shape, (self.MAX_PADDED,))
        num_bags = int(self.bag_lengths.shape[0])
        self.assertTrue(((seg >= 0) & (seg <= num_bags)).all())

    def test_slack_slots_are_all_trash(self) -> None:
        num_bags = int(self.bag_lengths.shape[0])
        seg = np.array(self._seg_sorted_padded(slack=_TILE_INT32))
        self.assertEqual(seg.shape, (self.MAX_PADDED + _TILE_INT32,))
        np.testing.assert_array_equal(
            seg[self.MAX_PADDED :], np.full(_TILE_INT32, num_bags)
        )

    def test_every_real_slot_keeps_its_bag(self) -> None:
        """A row's label must follow the row through sort + per-peer padding."""
        seg_caller = np.array(
            lengths_to_segment_ids(self.bag_lengths, self.MAX_LOCAL_IDS)
        )
        seg_sorted_padded = np.array(self._seg_sorted_padded())
        sort_perm = np.array(self.sort_perm)
        real = np.array(self.send_counts_real)
        padded = np.array(self.send_counts_padded)

        real_off = np.cumsum(real) - real
        padded_off = np.cumsum(padded) - padded
        for peer in range(self.W):
            for k in range(int(real[peer])):
                sorted_pos = int(real_off[peer]) + k  # position in sorted order
                padded_pos = int(padded_off[peer]) + k  # where the kernel puts it
                caller_pos = int(sort_perm[sorted_pos])
                self.assertEqual(
                    seg_sorted_padded[padded_pos],
                    seg_caller[caller_pos],
                    f"peer {peer} slot {k}",
                )

    def test_pooling_in_sorted_order_equals_pooling_in_caller_order(self) -> None:
        """**The Option A premise**, checked end to end on CPU.

        Pool the kernel-layout rows with kernel-layout labels, and compare
        against pooling caller-order rows with caller-order labels. Equal means
        the fused path may skip ``_unpad_per_peer`` and ``reverse_perm``.
        """
        rng = np.random.default_rng(0)
        dim = 3
        per_id_caller = jnp.array(
            rng.standard_normal((self.MAX_LOCAL_IDS, dim)).astype(np.float32)
        )
        # Zero the padding rows, matching what the kernel returns.
        valid = jnp.arange(self.MAX_LOCAL_IDS) < self.id_count
        per_id_caller = jnp.where(valid[:, None], per_id_caller, 0.0)

        # What the kernel actually holds: sorted, then per-peer tile padded.
        rows_kernel_layout = _pad_per_peer_2d(
            per_id_caller[self.sort_perm],
            self.send_counts_real,
            self.send_counts_padded,
            self.MAX_LOCAL_IDS,
            self.MAX_PADDED,
            self.W,
        )
        num_bags = int(self.bag_lengths.shape[0])
        fused = jax.ops.segment_sum(
            rows_kernel_layout,
            self._seg_sorted_padded(),
            num_segments=num_bags + 1,
        )[:num_bags]

        np.testing.assert_allclose(
            np.array(fused),
            np.array(pool_per_rank(per_id_caller, self.bag_lengths, pooling="sum")),
            atol=1e-6,
            rtol=1e-6,
        )


class FanOutPooledGradTest(unittest.TestCase):
    """The fused path's backward, before it reaches the existing per-id driver."""

    MAX_LOCAL_IDS = 8
    BAG_LENGTHS = jnp.array([3, 0, 2], jnp.int32)

    def test_matches_the_docstring_example(self) -> None:
        """The worked example in the docstring, asserted so it cannot drift."""
        lengths = jnp.array([2, 1, 0], jnp.int32)
        grad_pooled = jnp.array([[10.0], [20.0], [30.0]])
        for pooling, want in (
            ("sum", [[10.0], [10.0], [20.0], [0.0], [0.0]]),
            ("mean", [[5.0], [5.0], [20.0], [0.0], [0.0]]),
        ):
            got = fan_out_pooled_grad(
                grad_pooled, lengths, max_local_ids=5, pooling=pooling
            )
            np.testing.assert_allclose(np.array(got), np.array(want), err_msg=pooling)

    def test_sum_copies_the_bag_gradient_to_each_member(self) -> None:
        grad_pooled = jnp.array(
            [[1.0, 2.0], [9.0, 9.0], [3.0, 4.0]], jnp.float32
        )  # bag1 is empty; its row must reach nobody
        out = np.array(
            fan_out_pooled_grad(
                grad_pooled,
                self.BAG_LENGTHS,
                max_local_ids=self.MAX_LOCAL_IDS,
                pooling="sum",
            )
        )
        expected = np.zeros((self.MAX_LOCAL_IDS, 2), np.float32)
        expected[0:3] = [1.0, 2.0]
        expected[3:5] = [3.0, 4.0]
        np.testing.assert_allclose(out, expected)

    def test_mean_scales_by_bag_length(self) -> None:
        grad_pooled = jnp.array([[3.0, 3.0], [9.0, 9.0], [4.0, 4.0]], jnp.float32)
        out = np.array(
            fan_out_pooled_grad(
                grad_pooled,
                self.BAG_LENGTHS,
                max_local_ids=self.MAX_LOCAL_IDS,
                pooling="mean",
            )
        )
        expected = np.zeros((self.MAX_LOCAL_IDS, 2), np.float32)
        expected[0:3] = 1.0  # 3 / 3
        expected[3:5] = 2.0  # 4 / 2
        np.testing.assert_allclose(out, expected)

    def test_padding_slots_get_zero(self) -> None:
        grad_pooled = jnp.ones((3, 2), jnp.float32)
        for pooling in ("sum", "mean"):
            out = np.array(
                fan_out_pooled_grad(
                    grad_pooled,
                    self.BAG_LENGTHS,
                    max_local_ids=self.MAX_LOCAL_IDS,
                    pooling=pooling,
                )
            )
            np.testing.assert_allclose(out[5:], np.zeros((3, 2), np.float32))

    def test_matches_jax_grad_of_the_reference_pooling(self) -> None:
        """Hand-written VJP vs the one autodiff derives for ``pool_per_rank``."""
        rng = np.random.default_rng(1)
        per_id = jnp.array(
            rng.standard_normal((self.MAX_LOCAL_IDS, 3)).astype(np.float32)
        )
        seed = jnp.array(rng.standard_normal((3, 3)).astype(np.float32))
        for pooling in ("sum", "mean"):

            def loss(e: jax.Array, pooling: str = pooling) -> jax.Array:
                return jnp.sum(
                    pool_per_rank(e, self.BAG_LENGTHS, pooling=pooling) * seed
                )

            np.testing.assert_allclose(
                np.array(
                    fan_out_pooled_grad(
                        seed,
                        self.BAG_LENGTHS,
                        max_local_ids=self.MAX_LOCAL_IDS,
                        pooling=pooling,
                    )
                ),
                np.array(jax.grad(loss)(per_id)),
                atol=1e-6,
                rtol=1e-6,
                err_msg=pooling,
            )


class PooledCustomVjpPlumbingTest(unittest.TestCase):
    """Trace ``jax.grad`` through the pooled op without running the kernel.

    The three edits that move together when ``bag_lengths`` enters the
    ``custom_vjp`` signature -- the ``nondiff_argnums`` shift, the extra
    residual, and the backward tuple arity -- all fail at TRACE time, not at
    run time. ``jax.eval_shape`` therefore catches them on CPU, which matters
    because they are otherwise the silent-failure risk of the fused path: the
    fused kernel itself is real-TPU only.
    """

    W = 1  # single CPU device; the kernel is never executed, only traced
    MAX_LOCAL_IDS = 4
    V_LOCAL = 4
    DIM = 2
    NUM_BAGS = 2

    def _mesh(self) -> jax.sharding.Mesh:
        return jax.sharding.Mesh(np.array(jax.devices()[: self.W]), ("rank",))

    def _inputs(self):
        ids = jnp.zeros((self.W, self.MAX_LOCAL_IDS), jnp.int32)
        table = jnp.zeros((self.W, self.V_LOCAL, self.DIM), jnp.float32)
        bag_lengths = jnp.ones((self.W, self.NUM_BAGS), jnp.int32)
        return ids, table, bag_lengths

    def _grad_shape(self, pooling: str):
        ids, table, bag_lengths = self._inputs()
        mesh = self._mesh()

        def loss(t: jax.Array) -> jax.Array:
            out = ringshift_embedding_lookup(
                ids,
                None if pooling != "none" else jnp.full((self.W,), 2, jnp.int32),
                t,
                mesh=mesh,
                axis_name="rank",
                recv_capacity=_TILE_INT32,
                pooling=pooling,
                bag_lengths=bag_lengths if pooling != "none" else None,
            )
            return jnp.sum(out)

        return jax.eval_shape(jax.grad(loss), table)

    def test_sum_grad_traces_and_keeps_the_table_shape(self) -> None:
        out = self._grad_shape("sum")
        self.assertEqual(out.shape, (self.W, self.V_LOCAL, self.DIM))
        self.assertEqual(out.dtype, jnp.float32)

    def test_mean_grad_traces(self) -> None:
        out = self._grad_shape("mean")
        self.assertEqual(out.shape, (self.W, self.V_LOCAL, self.DIM))

    def test_unpooled_grad_still_traces(self) -> None:
        """The EC path must be unaffected by the signature change."""
        out = self._grad_shape("none")
        self.assertEqual(out.shape, (self.W, self.V_LOCAL, self.DIM))

    def test_pooled_forward_shape_is_per_bag(self) -> None:
        """Pooling now happens inside the custom_vjp op, so its output is [W, B, D]."""
        ids, table, bag_lengths = self._inputs()
        mesh = self._mesh()
        out = jax.eval_shape(
            lambda t: ringshift_embedding_lookup(
                ids,
                None,
                t,
                mesh=mesh,
                axis_name="rank",
                recv_capacity=_TILE_INT32,
                pooling="sum",
                bag_lengths=bag_lengths,
            ),
            table,
        )
        self.assertEqual(out.shape, (self.W, self.NUM_BAGS, self.DIM))


class PoolingImplEnvironmentTest(unittest.TestCase):
    def test_no_pooling_returns_none_mode_without_reading_env(self) -> None:
        with mock.patch.dict(os.environ, {_POOLING_IMPL_ENV: "invalid"}, clear=False):
            self.assertEqual(_pooling_impl_from_env("none"), "none")

    def test_defaults_to_fused_kernel(self) -> None:
        with mock.patch.dict(os.environ):
            os.environ.pop(_POOLING_IMPL_ENV, None)
            self.assertEqual(_pooling_impl_from_env("sum"), "fused_kernel")

    def test_explicit_values(self) -> None:
        for pooling_impl in ("fused_kernel", "xla"):
            with mock.patch.dict(
                os.environ, {_POOLING_IMPL_ENV: pooling_impl}, clear=False
            ):
                self.assertEqual(_pooling_impl_from_env("sum"), pooling_impl)

    def test_unknown_value_is_rejected(self) -> None:
        with (
            mock.patch.dict(os.environ, {_POOLING_IMPL_ENV: "auto"}, clear=False),
            self.assertRaisesRegex(ValueError, _POOLING_IMPL_ENV),
        ):
            _pooling_impl_from_env("sum")


class PoolingImplParityTest(unittest.TestCase):
    """Both environment-selected implementations trace and differentiate.

    Values cannot be compared here -- the fused path only runs on a real TPU --
    so this pins down their output and gradient shapes on CPU.
    """

    W = 1  # single CPU device; the kernel is traced, never executed
    MAX_LOCAL_IDS = 4
    V_LOCAL = 4
    DIM = 2
    NUM_BAGS = 2

    def _call(self, table: jax.Array, pooling_impl: str) -> jax.Array:
        with mock.patch.dict(
            os.environ, {_POOLING_IMPL_ENV: pooling_impl}, clear=False
        ):
            return ringshift_embedding_lookup(
                jnp.zeros((self.W, self.MAX_LOCAL_IDS), jnp.int32),
                None,
                table,
                mesh=jax.sharding.Mesh(np.array(jax.devices()[: self.W]), ("rank",)),
                axis_name="rank",
                recv_capacity=_TILE_INT32,
                pooling="sum",
                bag_lengths=jnp.ones((self.W, self.NUM_BAGS), jnp.int32),
            )

    def _table(self) -> jax.Array:
        return jnp.zeros((self.W, self.V_LOCAL, self.DIM), jnp.float32)

    def test_both_impls_give_the_same_forward_shape(self) -> None:
        shapes = {
            impl: jax.eval_shape(
                lambda t, i=impl: self._call(t, i), self._table()
            ).shape
            for impl in ("xla", "fused_kernel")
        }
        self.assertEqual(shapes["xla"], (self.W, self.NUM_BAGS, self.DIM))
        self.assertEqual(shapes["xla"], shapes["fused_kernel"])

    def test_both_impls_give_the_same_gradient_shape(self) -> None:
        grads = {
            impl: jax.eval_shape(
                jax.grad(lambda t, i=impl: jnp.sum(self._call(t, i))), self._table()
            )
            for impl in ("xla", "fused_kernel")
        }
        self.assertEqual(grads["xla"].shape, (self.W, self.V_LOCAL, self.DIM))
        self.assertEqual(grads["xla"].shape, grads["fused_kernel"].shape)
        self.assertEqual(grads["xla"].dtype, grads["fused_kernel"].dtype)
