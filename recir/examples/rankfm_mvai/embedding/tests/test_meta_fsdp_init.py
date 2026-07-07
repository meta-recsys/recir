# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

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

        arr = _XLAModule._to_sharded_jax(meta, mesh, is_fsdp_weight=True)

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


if __name__ == "__main__":
    unittest.main()
