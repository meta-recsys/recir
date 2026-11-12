# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""The SparseCore row-wise Adagrad kernel's FP16 stochastic writeback, on CPU: the
rounding and the random bits it draws (the kernel itself needs a SparseCore)."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from recir.ops.pallas.embeddings.sparsecore_rowwise_adagrad import (
    _hash32,
    _stochastic_round_fp16,
    fp16_rounding_key,
)

_DRAWS = 8192


def _round(values: np.ndarray, seed: int = 7) -> np.ndarray:
    """Each value rounded _DRAWS times with independent random bits."""
    x = jnp.broadcast_to(jnp.asarray(values, jnp.float32), (_DRAWS, len(values)))
    counters = jnp.arange(x.size, dtype=jnp.int32).reshape(x.shape)
    bits = _hash32(_hash32(counters) ^ jnp.int32(seed))
    return np.asarray(jax.jit(_stochastic_round_fp16)(x, bits))


class StochasticRoundFp16Test(unittest.TestCase):
    def test_rounds_to_adjacent_fp16_values_without_bias(self) -> None:
        rng = np.random.default_rng(0)
        values = np.concatenate(
            [
                rng.uniform(-0.05, 0.05, 24),  # embedding-scale normals
                rng.uniform(-(2.0**-14), 2.0**-14, 8),  # FP16 subnormals
                rng.uniform(1000, 60000, 4),
            ]
        ).astype(np.float32)

        rounded = _round(values)

        lower = np.where(
            values.astype(np.float16).astype(np.float32) <= values,
            values.astype(np.float16),
            np.nextafter(values.astype(np.float16), np.float16(-np.inf)),
        ).astype(np.float32)
        upper = np.nextafter(lower.astype(np.float16), np.float16(np.inf)).astype(
            np.float32
        )
        self.assertTrue(
            np.array_equal(rounded.astype(np.float16).astype(np.float32), rounded)
        )
        self.assertTrue(np.all((rounded == lower) | (rounded == upper)))
        gap = upper - lower
        bias = np.abs(rounded.mean(axis=0) - values) / gap
        self.assertLess(float(bias.max()), 5 / np.sqrt(_DRAWS))

    def test_fp16_values_signed_zero_and_specials_pass_through(self) -> None:
        values = np.array(
            [1.0, -0.5, 2.0**-24, 0.0, -0.0, np.inf, -np.inf, np.nan], np.float32
        )

        rounded = _round(values)

        np.testing.assert_array_equal(
            rounded[:, :4], np.broadcast_to(values[:4], (_DRAWS, 4))
        )
        self.assertTrue(np.all(np.signbit(rounded[:, 4]) & (rounded[:, 4] == 0)))
        self.assertTrue(np.all(np.isposinf(rounded[:, 5])))
        self.assertTrue(np.all(np.isneginf(rounded[:, 6])))
        self.assertTrue(np.all(np.isnan(rounded[:, 7])))

    def test_beyond_fp16_range_converts_as_fp16(self) -> None:
        values = np.array([65510.0, -65510.0, 65520.0, -70000.0], np.float32)

        rounded = _round(values)

        np.testing.assert_array_equal(
            rounded,
            np.broadcast_to(
                values.astype(np.float16).astype(np.float32), rounded.shape
            ),
        )


class Fp16RoundingKeyTest(unittest.TestCase):
    def test_keys_differ_across_steps_streams_and_devices(self) -> None:
        keys = {
            int(fp16_rounding_key(jnp.int32(counter), stream, jnp.int32(device)))
            for counter in range(64)
            for stream in range(4)
            for device in range(16)
        }

        self.assertEqual(64 * 4 * 16, len(keys))
