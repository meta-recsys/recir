# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas import tpu as pltpu
from recir.torchax.ops.onehot_segment_sum import (
    _bf16_terms,
    onehot_segment_sum,
    onehot_segment_sum_enabled,
)


def _scatter(values, ids, num_segments):
    """The reduction onehot_segment_sum replaces."""
    return np.asarray(
        jnp.zeros((num_segments, values.shape[1]), jnp.float32).at[ids].add(values)
    )


def _run(values, ids, num_segments):
    with pltpu.force_tpu_interpret_mode():
        return np.asarray(onehot_segment_sum(values, ids, num_segments))


class EnabledGateTest(unittest.TestCase):
    def test_zero_segments_is_not_enabled(self):
        # An empty table divided by zero when sizing the segment blocks.
        with mock.patch.dict("os.environ", {"RECIR_ONEHOT_SEGMENT_SUM": "1"}):
            self.assertFalse(onehot_segment_sum_enabled(0))
            self.assertTrue(onehot_segment_sum_enabled(1))
            self.assertTrue(onehot_segment_sum_enabled(4096))
            self.assertFalse(onehot_segment_sum_enabled(4097))

    def test_flag_off(self):
        with mock.patch.dict("os.environ", {"RECIR_ONEHOT_SEGMENT_SUM": "0"}):
            self.assertFalse(onehot_segment_sum_enabled(512))


class NonFiniteTermsTest(unittest.TestCase):
    def test_terms_reconstruct_every_value(self):
        x = jnp.array([np.inf, -np.inf, np.nan, 1.5, 1e-30, 3.7e20], jnp.float32)

        total = sum(np.asarray(t.astype(jnp.float32)) for t in _bf16_terms(x))

        np.testing.assert_array_equal(total, np.asarray(x))

    def test_signalling_nan_payload_stays_nan(self):
        # Masking the low mantissa off 0x7F800001 leaves the +Inf bit pattern.
        bits = jnp.array([0x7F800001, 0x7F8000FF, 0x7FC00001], jnp.uint32)
        x = jax.lax.bitcast_convert_type(bits, jnp.float32)

        high = _bf16_terms(x)[0].astype(jnp.float32)

        self.assertFalse(bool(jnp.isinf(high).any()))
        self.assertTrue(bool(jnp.isnan(high).all()))

    def test_finite_values_decompose_exactly(self):
        x = jax.random.normal(jax.random.key(4), (4096,), jnp.float32) * 1e6

        terms = [np.asarray(t.astype(jnp.float32)) for t in _bf16_terms(x)]

        np.testing.assert_array_equal(sum(terms), np.asarray(x))

    def test_bfloat16_passes_through(self):
        x = jnp.arange(8, dtype=jnp.bfloat16)

        self.assertEqual(len(_bf16_terms(x)), 1)

    def test_a_non_finite_row_reaches_every_segment(self):
        """Documented divergence from the scatter: the one-hot's zeros multiply
        an Inf row into every other row, so one bad row poisons the table."""
        values = jnp.ones((4, 8), jnp.float32).at[1].set(jnp.inf)
        ids = jnp.array([0, 1, 2, 3], jnp.int32)

        got = _run(values, ids, 128)

        self.assertTrue(bool(np.isinf(got[1]).all()))
        self.assertEqual(int(np.isnan(got).any(axis=1).sum()), 127)


class NegativeIdTest(unittest.TestCase):
    def test_negative_ids_count_from_the_end(self):
        values = jnp.arange(1, 33, dtype=jnp.float32).reshape(4, 8)
        ids = jnp.array([0, -1, 2, -128], jnp.int32)

        got = _run(values, ids, 128)

        np.testing.assert_array_equal(got, _scatter(values, ids, 128))

    def test_ids_outside_the_table_are_dropped(self):
        values = jnp.ones((3, 8), jnp.float32)
        ids = jnp.array([0, 128, -129], jnp.int32)

        got = _run(values, ids, 128)

        np.testing.assert_array_equal(got, _scatter(values, ids, 128))
        self.assertEqual(float(got.sum()), 8.0)

    def test_matches_the_scatter_on_in_range_ids(self):
        values = jax.random.normal(jax.random.key(5), (5000, 32), jnp.float32)
        ids = jax.random.randint(jax.random.key(6), (5000,), 0, 512, jnp.int32)

        got = _run(values, ids, 512)

        np.testing.assert_allclose(got, _scatter(values, ids, 512), atol=1e-5)
