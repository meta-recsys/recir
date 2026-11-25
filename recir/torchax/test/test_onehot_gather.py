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
from recir.torchax.ops.onehot_segment_sum import onehot_gather, onehot_gather_enabled


def _gather(table, ids):
    with pltpu.force_tpu_interpret_mode():
        return np.asarray(onehot_gather(table, ids))


class EnabledGateTest(unittest.TestCase):
    def test_an_empty_table_is_not_enabled(self):
        # Sizing the one-hot for an empty table divides by zero.
        with mock.patch.dict("os.environ", {"RECIR_ONEHOT_GATHER": "1"}):
            self.assertFalse(onehot_gather_enabled(0))
            self.assertTrue(onehot_gather_enabled(1))
            self.assertTrue(onehot_gather_enabled(512))
            self.assertFalse(onehot_gather_enabled(513))

    def test_flag_off(self):
        with mock.patch.dict("os.environ", {"RECIR_ONEHOT_GATHER": "0"}):
            self.assertFalse(onehot_gather_enabled(256))


class OutOfRangeIdTest(unittest.TestCase):
    """``jnp.take`` defaults to mode="fill", whose fill for an inexact dtype is
    NaN. A zero row instead would turn a step-1 crash into silent corruption."""

    def test_matches_take_on_out_of_range_ids(self):
        table = jax.random.normal(jax.random.key(0), (302, 224), jnp.float32)
        ids = jnp.array([0, 5, 302, 301, 312, -1, -302, -303], jnp.int32)

        got = _gather(table, ids)

        np.testing.assert_array_equal(got, np.asarray(jnp.take(table, ids, axis=0)))

    def test_an_out_of_range_id_does_not_vanish_into_the_loss(self):
        table = jax.random.normal(jax.random.key(0), (302, 224), jnp.float32)
        ids = jnp.array([1, 2, 302], jnp.int32)

        self.assertTrue(bool(np.isnan(_gather(table, ids).sum())))

    def test_bfloat16_table(self):
        table = jax.random.normal(jax.random.key(3), (128, 64), jnp.float32).astype(
            jnp.bfloat16
        )
        ids = jnp.array([0, 7, 128, -1], jnp.int32)

        got = _gather(table, ids)
        want = np.asarray(jnp.take(table, ids, axis=0))

        self.assertEqual(got.dtype, jnp.bfloat16)
        # numpy does not treat two bfloat16 NaNs as equal, so widen first.
        np.testing.assert_array_equal(got.astype(np.float32), want.astype(np.float32))


class InRangeIdTest(unittest.TestCase):
    def test_every_row_is_its_table_row_bit_for_bit(self):
        table = jax.random.normal(jax.random.key(1), (302, 224), jnp.float32)
        ids = jax.random.randint(jax.random.key(2), (3000,), 0, 302, jnp.int32)

        got = _gather(table, ids)

        np.testing.assert_array_equal(got, np.asarray(jnp.take(table, ids, axis=0)))

    def test_negative_ids_count_from_the_end(self):
        table = jnp.arange(16 * 8, dtype=jnp.float32).reshape(16, 8)
        ids = jnp.array([0, -1, -16], jnp.int32)

        got = _gather(table, ids)

        np.testing.assert_array_equal(got, np.asarray(jnp.take(table, ids, axis=0)))
