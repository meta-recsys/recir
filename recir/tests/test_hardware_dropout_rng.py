# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
import os
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from recir.torchax.gspmd_context import gspmd_mesh_context
from recir.torchax.ops.jrandom import _jbernoulli_keyed


class HardwareRNGTests(unittest.TestCase):
    def test_context_mesh_controls_device_order_and_batch_sharding(self) -> None:
        devices = np.asarray(jax.devices())[::2][::-1]
        shape = (4 * 1024, 16)
        for grid, names, resource in (
            (devices, ("batch",), "batch"),
            (devices.reshape(2, 2), ("data", "fsdp"), ("data", "fsdp")),
        ):
            with self.subTest(names=names):
                mesh = Mesh(grid, names)
                sharding = NamedSharding(mesh, PartitionSpec(resource, None))
                with (
                    patch.dict(os.environ, {"RECIR_DROPOUT_PRNG_IMPL": "rbg"}),
                    gspmd_mesh_context(
                        mesh, resource, int(mesh.shape[names[-1]]), len(names) == 2
                    ),
                    patch("jax.devices", side_effect=AssertionError("raw devices")),
                    patch("jax.device_count", side_effect=AssertionError("raw count")),
                ):
                    actual = jax.jit(
                        lambda: _jbernoulli_keyed(
                            jnp.zeros(shape), 0.9, jnp.uint32(500), 2, 0
                        ),
                        out_shardings=sharding,
                    )()
                key = jax.random.fold_in(jax.random.key(0, impl="rbg"), 500)
                key = jax.random.fold_in(key, 2)
                expected = np.concatenate(
                    [
                        np.asarray(
                            jax.random.bernoulli(
                                jax.random.fold_in(key, chip), 0.9, (1024, 16)
                            )
                        )
                        for chip in range(4)
                    ]
                )
                np.testing.assert_array_equal(np.asarray(actual), expected)
                self.assertEqual(actual.sharding, sharding)

    def test_context_does_not_partition_a_hidden_dimension(self) -> None:
        mesh = Mesh(np.asarray(jax.devices()), ("batch",))
        shape = (7, 16)
        with (
            patch.dict(os.environ, {"RECIR_DROPOUT_PRNG_IMPL": "rbg"}),
            gspmd_mesh_context(mesh, "batch", 8, False),
            self.assertLogs("recir.torchax.ops.jrandom", level="WARNING"),
        ):
            actual = jax.jit(
                lambda: _jbernoulli_keyed(jnp.zeros(shape), 0.9, jnp.uint32(500), 2, 0)
            )()
        key = jax.random.fold_in(jax.random.key(0, impl="rbg"), 500)
        key = jax.random.fold_in(key, 2)
        expected = jax.random.bernoulli(key, 0.9, shape)
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    def test_nonshardable_draw_is_logged_and_keeps_the_original_key(self) -> None:
        shape = (7, 5)
        counter = jnp.uint32(500)
        with patch.dict(os.environ, {"RECIR_DROPOUT_PRNG_IMPL": "rbg"}):
            with self.assertLogs("recir.torchax.ops.jrandom", level="WARNING") as logs:
                actual = jax.jit(
                    lambda: _jbernoulli_keyed(jnp.zeros(shape), 0.9, counter, 2, 0)
                )()
        key = jax.random.fold_in(jax.random.key(0, impl="rbg"), counter)
        key = jax.random.fold_in(key, 2)
        expected = jax.random.bernoulli(key, jnp.float32(0.9), shape).astype(
            actual.dtype
        )
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
        self.assertIn("no shardable axis", " ".join(logs.output))
        self.assertIn("unsharded RBG draw", " ".join(logs.output))

    def test_sharded_masks_are_repeatable_and_independent(self) -> None:
        self.assertEqual(jax.device_count(), 8)
        shape = (8 * 32768, 32)
        mesh = Mesh(np.asarray(jax.devices()), ("chips",))
        sharding = NamedSharding(mesh, PartitionSpec("chips", None))
        with patch.dict(os.environ, {"RECIR_DROPOUT_PRNG_IMPL": "rbg"}):
            fn = jax.jit(
                lambda c: _jbernoulli_keyed(
                    jnp.zeros(shape, dtype=jnp.bfloat16), 0.9, c, 2, 0
                ),
                out_shardings=sharding,
            )
            first = np.asarray(fn(jnp.uint32(500))).reshape(8, -1)
            repeated = np.asarray(fn(jnp.uint32(500))).reshape(8, -1)
            later = np.asarray(fn(jnp.uint32(501))).reshape(8, -1)
        np.testing.assert_array_equal(first, repeated)
        means = first.astype(np.float32).mean(axis=1)
        np.testing.assert_allclose(means, 0.9, atol=0.005, rtol=0)
        for chip in range(1, 8):
            agreement = np.mean(first[0] == first[chip])
            self.assertLess(abs(float(agreement) - 0.82), 0.005)
        self.assertLess(abs(float(np.mean(first == later)) - 0.82), 0.005)
        print("HARDWARE_RNG_CHIP_MEANS", means.tolist(), flush=True)

    def test_probability_edges_and_output_dtypes(self) -> None:
        with patch.dict(os.environ, {"RECIR_DROPOUT_PRNG_IMPL": "rbg"}):
            for dtype in (jnp.float32, jnp.bfloat16):
                for p in (0.0, 1.0):
                    result = jax.jit(
                        lambda dtype=dtype, p=p: _jbernoulli_keyed(
                            jnp.zeros((32, 16), dtype=dtype), p, jnp.uint32(0), 2, 0
                        )
                    )()
                    self.assertEqual(result.dtype, dtype)
                    np.testing.assert_array_equal(
                        np.asarray(result), np.full((32, 16), p)
                    )

    def test_folded_site_changes_the_mask(self) -> None:
        with patch.dict(os.environ, {"RECIR_DROPOUT_PRNG_IMPL": "rbg"}):
            first = jax.jit(
                lambda: _jbernoulli_keyed(
                    jnp.zeros((8192, 32)), 0.9, jnp.uint32(500), 2, 0
                )
            )()
            second = jax.jit(
                lambda: _jbernoulli_keyed(
                    jnp.zeros((8192, 32)), 0.9, jnp.uint32(500), 3, 0
                )
            )()
            self.assertLess(
                abs(float(np.mean(np.asarray(first) == np.asarray(second))) - 0.82),
                0.005,
            )
