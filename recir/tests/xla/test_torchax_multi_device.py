# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors

"""Tests demonstrating torchax multi-device / multi-process handling.
Tests show how torchax distributes data and model weights across
multiple JAX devices, covering both the single-host (all devices visible
to one process) and the multi-host sharding utilities.
"""

import os
import unittest

import jax  # @manual
import numpy as np
import recir.torchax as torchax
import torch
import torch.nn as nn
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

# IMPORTANT: must run BEFORE any `import jax` so the cuda plugin path patch
# is in effect when jax.devices() / jax.* first triggers plugin discovery.
from recir.tests.xla import _jax_cuda_plugin_fix  # noqa: F401, I001
from recir.tests.xla.xla_test_mixin import jax_distributed
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


DIM = 16


class SimpleMLP(nn.Module):
    def __init__(self, dim: int = DIM):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class TestShardedDevicePut(unittest.TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    def test_sharded_device_put_via_torchax(self):
        with torchax.default_env():
            gpu = jax.devices("gpu")[0]
            mesh = Mesh([gpu], axis_names=("fsdp",))
            batch_sharding = NamedSharding(mesh, P("fsdp"))
            x = torch.randn(4, DIM, device="cpu")
            x_torchax = x.to("jax")
            x_torchax.apply_jax_(jax.device_put, batch_sharding)
            jax_array = x_torchax.jax()
            self.assertEqual(len(jax_array.devices()), 1)
            for shard in jax_array.addressable_shards:
                self.assertEqual(shard.data.shape, (4, DIM))


class TestMakeArrayFromCallback(unittest.TestCase):
    """Tests for jax.make_array_from_callback, the pattern used to
    initialize model weights that are too large to fit on one device.
    Each device gets its shard created independently via a callback,
    so the full weight is never materialized in one place.
    """

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    def test_weight_init_from_callback(self):
        with torchax.default_env() as env:
            gpu = jax.devices("gpu")[0]
            mesh = Mesh([gpu], axis_names=("fsdp",))
            sharding = NamedSharding(mesh, P("fsdp"))
            global_shape = (DIM, DIM)

            def make_shard(slice_index):
                shard_shape = tuple(
                    (s.stop - s.start) if s.start is not None else global_shape[i]
                    for i, s in enumerate(slice_index)
                )
                seed = hash(tuple((s.start, s.stop, s.step) for s in slice_index)) % (
                    2**31 - 1
                )
                key = jax.device_put(jax.random.PRNGKey(seed), gpu)
                return jax.random.normal(key, shard_shape, dtype=jax.numpy.float32)

            weight = jax.make_array_from_callback(global_shape, sharding, make_shard)
            self.assertEqual(weight.shape, global_shape)
            tx_weight = env.j2t_iso(weight)
            self.assertEqual(tx_weight.shape, (DIM, DIM))


class TestMultiProcessShardedDevicePut(DTensorTestBase):
    """Multi-process test for the sharded_device_put pattern.

    The ``jax_distributed()`` context manager initializes JAX distributed,
    bridging PyTorch's env vars so each JAX process owns exactly one GPU,
    and calls ``jax.distributed.shutdown()`` on exit.

    After initialization:
    - ``jax.device_count()`` returns the global device count (2).
    - ``jax.local_device_count()`` returns 1 (one GPU per process).
    - This means ``num_global != num_local``, so the multi-host branch
      of ``sharded_device_put`` is exercised.
    """

    @property
    def world_size(self):
        return 2

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_multi_process_sharded_device_put(self):
        with jax_distributed():
            num_global_devices = jax.device_count()
            num_local_devices = jax.local_device_count()
            self.assertEqual(num_global_devices, 2)
            self.assertEqual(num_local_devices, 1)
            mesh = Mesh(jax.devices(), axis_names=("data",))
            sharding = NamedSharding(mesh, P("data"))
            global_data = np.ones((8, DIM), dtype=np.float32)
            global_shape = global_data.shape
            x_split = [
                jax.device_put(global_data[slices], device)
                for device, slices in sharding.addressable_devices_indices_map(
                    global_shape
                ).items()
            ]
            result = jax.make_array_from_single_device_arrays(
                global_shape, sharding, x_split
            )
            self.assertEqual(result.shape, (8, DIM))
            self.assertEqual(len(result.addressable_shards), 1)
            self.assertEqual(result.addressable_shards[0].data.shape, (4, DIM))

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_multi_process_make_array_from_callback(self):
        with jax_distributed(port_offset=1001):
            local_gpu = jax.local_devices()[0]
            mesh = Mesh(jax.devices(), axis_names=("data",))
            sharding = NamedSharding(mesh, P("data"))
            global_shape = (DIM, DIM)

            def make_shard(slice_index):
                shard_shape = tuple(
                    (s.stop - s.start) if s.start is not None else global_shape[i]
                    for i, s in enumerate(slice_index)
                )
                seed = hash(tuple((s.start, s.stop, s.step) for s in slice_index)) % (
                    2**31 - 1
                )
                key = jax.device_put(jax.random.PRNGKey(seed), local_gpu)
                return jax.random.normal(key, shard_shape, dtype=jax.numpy.float32)

            weight = jax.make_array_from_callback(global_shape, sharding, make_shard)
            self.assertEqual(weight.shape, global_shape)
            self.assertEqual(len(weight.addressable_shards), 1)
            self.assertEqual(weight.addressable_shards[0].data.shape, (DIM // 2, DIM))


if __name__ == "__main__":
    run_tests()
