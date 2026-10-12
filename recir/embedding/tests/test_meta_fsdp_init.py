# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""CPU test for on-device (meta) FSDP weight init in _to_sharded_jax.

Verifies that a *meta* FSDP weight is materialized as a correctly-sharded
on-device jax.Array via jax.random -- WITHOUT host allocation (the whole point:
benchmarking huge embedding tables without materializing them on CPU). Forces 8
CPU "devices" so the row-wise shard split is exercised.
"""

import os

# Must be set before jax initializes its CPU backend.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import unittest

import jax
import numpy as np
import torch
from jax.sharding import Mesh
from recir.lowering.xla_lower import _XLAModule
from recir.utils.meta_tensor_init import (
    mark_meta_uniform_init,
    mark_meta_zero_init,
    table_init_stream_id,
)


class MetaFSDPInitTest(unittest.TestCase):
    def test_meta_weight_inits_on_device_sharded(self) -> None:
        devices = jax.devices()
        world = len(devices)
        self.assertGreaterEqual(world, 2, "need >=2 CPU devices for the shard test")
        vocab, dim = 8 * world, 16  # vocab divisible by world
        mesh = Mesh(np.array(devices), ("torch_dist",))

        # Meta param: no host storage.
        meta = torch.empty(vocab, dim, device="meta")
        self.assertTrue(meta.is_meta)

        arr = _XLAModule._to_sharded_jax(meta, mesh, is_fsdp_tensor=True)

        # Correct global shape, on-device, row-wise sharded across all devices.
        self.assertIsInstance(arr, jax.Array)
        self.assertEqual(tuple(arr.shape), (vocab, dim))
        self.assertEqual(str(arr.dtype), "float32")
        self.assertEqual(len(arr.addressable_shards), world)
        for shard in arr.addressable_shards:
            self.assertEqual(tuple(shard.data.shape), (vocab // world, dim))

        # Random init actually ran (not zeros/constant), and shards differ
        # (per-device fold_in -> different rows).
        host = np.array(jax.device_get(arr))
        self.assertGreater(float(host.std()), 0.0)
        s0 = np.array(jax.device_get(arr.addressable_shards[0].data))
        s1 = np.array(jax.device_get(arr.addressable_shards[1].data))
        self.assertFalse(np.allclose(s0, s1), "per-device shards should differ")

    def test_meta_optimizer_state_inits_to_zero(self) -> None:
        devices = jax.devices()
        world = len(devices)
        vocab, dim = 8 * world, 1
        mesh = Mesh(np.array(devices), ("torch_dist",))

        meta_state = torch.empty(vocab, dim, device="meta")
        mark_meta_zero_init(meta_state)

        arr = _XLAModule._to_sharded_jax(meta_state, mesh, is_fsdp_tensor=True)

        host = np.array(jax.device_get(arr))
        self.assertEqual(tuple(host.shape), (vocab, dim))
        self.assertTrue(np.all(host == 0.0))

    def test_meta_weight_uses_explicit_uniform_bound(self) -> None:
        devices = jax.devices()
        world = len(devices)
        vocab, dim = 8 * world, 16
        mesh = Mesh(np.array(devices), ("torch_dist",))

        meta = torch.empty(vocab, dim, device="meta")
        mark_meta_uniform_init(meta, 0.01, table_init_stream_id("explicit_bound_table"))

        arr = _XLAModule._to_sharded_jax(meta, mesh, is_fsdp_tensor=True)

        host = np.array(jax.device_get(arr))
        self.assertGreater(float(host.std()), 0.0)
        self.assertLessEqual(float(np.max(np.abs(host))), 0.01)

    def _materialize(self, name: str, vocab: int, dim: int, bound: float) -> np.ndarray:
        mesh = Mesh(np.array(jax.devices()), ("torch_dist",))
        meta = torch.empty(vocab, dim, device="meta")
        mark_meta_uniform_init(meta, bound, table_init_stream_id(name))
        return np.array(
            jax.device_get(_XLAModule._to_sharded_jax(meta, mesh, is_fsdp_tensor=True))
        )

    def test_two_tables_get_independent_streams(self) -> None:
        """Two tables of the same shape must not materialize identically."""
        world = len(jax.devices())
        vocab, dim = 8 * world, 16
        a = self._materialize("post_id", vocab, dim, 0.01)
        b = self._materialize("viewer_rid", vocab, dim, 0.01)
        self.assertFalse(bool(np.all(a == b)))

    def test_smaller_table_is_not_a_rescaled_prefix(self) -> None:
        """A short table must not be a long one's prefix times a constant."""
        world = len(jax.devices())
        dim = 16
        small = self._materialize("w2v_user", 8 * world, dim, 0.01)
        large = self._materialize("viewer_rid", 32 * world, dim, 0.02)

        ratio = small.ravel() / large.ravel()[: small.size]
        self.assertGreater(float(ratio.max() - ratio.min()), 1e-3)


if __name__ == "__main__":
    unittest.main()
