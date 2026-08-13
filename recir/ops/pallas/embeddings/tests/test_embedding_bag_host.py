# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""CPU-runnable tests for the per-bag reduction primitives (section F1).

These cover the pieces both pooled-lookup options share, and nothing else: the
reduction itself, the segment-id construction it consumes, the argument
validation, and the numpy oracle both options are checked against. Where the
reduction ultimately runs is not decided at this layer, so nothing here goes
through ``ringshift_embedding_lookup``.

``pool_per_rank`` takes a plain per-id array, so the whole surface -- forward,
backward, jagged/empty bags, padding, argument validation -- is testable
without a TPU. That is what lets both candidate kernels start from a reference
that is already known good.

Expected values are hand-computed inline rather than derived from a second
implementation, so a bug in the reference cannot hide a bug in the code.
"""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from recir.ops.pallas.embeddings.ringshift import (
    apply_pooling_divisor,
    lengths_to_segment_ids,
    pool_per_rank,
    ringshift_embedding_lookup_reference,
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
