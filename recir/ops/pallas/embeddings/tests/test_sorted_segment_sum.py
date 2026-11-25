# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""The sorted segment sum against a scatter-add, in TPU interpret mode on CPU."""

import unittest

import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas import tpu as pltpu
from recir.ops.pallas.embeddings.sorted_segment_sum import sorted_segment_sum


def _sorted_ids(rng: np.random.Generator, n: int, num_segments: int) -> np.ndarray:
    # Runs of repeated ids with empty segments between them, and dropped ids
    # past the last segment at the end, like the exchange's padding rows.
    ids = rng.integers(0, num_segments, n - n // 8)
    dropped = np.full(n // 8, num_segments)
    return np.sort(np.concatenate([ids, dropped])).astype(np.int32)


def _reference(values: np.ndarray, ids: np.ndarray, num_segments: int) -> np.ndarray:
    out = np.zeros((num_segments + 1, values.shape[1]), np.float64)
    np.add.at(out, ids, values.astype(np.float64))
    return out[:num_segments]


class SortedSegmentSumTest(unittest.TestCase):
    def test_matches_a_scatter_add_to_f32_rounding(self) -> None:
        rng = np.random.default_rng(0)
        n, num_segments = 1000, 700
        ids = _sorted_ids(rng, n, num_segments)
        values = rng.normal(size=(n, 128)).astype(np.float32)

        with pltpu.force_tpu_interpret_mode():
            out = sorted_segment_sum(
                jnp.asarray(values),
                jnp.asarray(ids),
                num_segments,
                block_segments=128,
                block_rows=128,
            )

        expected = _reference(values, ids, num_segments)
        self.assertEqual(out.shape, (num_segments, 128))
        np.testing.assert_allclose(np.asarray(out), expected, rtol=0, atol=1e-5)
        empty = np.setdiff1d(np.arange(num_segments), ids)
        self.assertGreater(empty.size, 0)
        self.assertFalse(np.asarray(out)[empty].any())

    def test_scales_and_casts_the_sums(self) -> None:
        rng = np.random.default_rng(1)
        n, num_segments = 640, 384
        ids = _sorted_ids(rng, n, num_segments)
        values = rng.normal(size=(n, 256)).astype(np.float32)

        with pltpu.force_tpu_interpret_mode():
            out = sorted_segment_sum(
                jnp.asarray(values),
                jnp.asarray(ids),
                num_segments,
                scale=0.25,
                out_dtype=jnp.bfloat16,
                block_segments=128,
                block_rows=256,
            )

        self.assertEqual(out.dtype, jnp.bfloat16)
        expected = 0.25 * _reference(values, ids, num_segments)
        np.testing.assert_allclose(
            np.asarray(out.astype(jnp.float32)), expected, rtol=2**-8, atol=1e-6
        )

    def test_only_dropped_ids_give_zeros(self) -> None:
        values = np.ones((300, 128), np.float32)
        ids = np.full(300, 256, np.int32)

        with pltpu.force_tpu_interpret_mode():
            out = sorted_segment_sum(
                jnp.asarray(values),
                jnp.asarray(ids),
                256,
                block_segments=128,
                block_rows=128,
            )

        self.assertFalse(np.asarray(out).any())
