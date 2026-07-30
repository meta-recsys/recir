# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors

"""CPU unit tests for ``_XLAModule._to_sharded_jax`` input-sharding normalization.

Regression coverage for the multi-host GSPMD bug where a cached ``device_state``
entry -- a single-device ``jax.Array`` produced by
``XLADeviceStateManager._convert_flat_state_to_jax`` -- was fed straight into the
GSPMD executable, whose declared ``in_sharding`` then disagreed with the array's
``SingleDeviceSharding`` ("Computation was compiled for input shardings that
disagree with the shardings of arguments"). ``_to_sharded_jax`` now re-shards
such a ``jax.Array`` to the layout the input expects -- generically across
replicated / batch / FSDP -- and leaves it untouched when already correct.

These spoof multiple JAX CPU devices (no accelerator needed) so the resharding
path can be exercised on any devserver / CI executor. The true multi-*process*
(multi-controller) path that triggered the prod failure can't be reproduced in a
unit test, but the per-input resharding logic that fixes it can.
"""

import os

# Spoof multi-device JAX on CPU before any ``import jax`` (mirrors test_aot.py).
os.environ["JAX_PLATFORMS"] = "cpu"
_existing_xla_flags: str = os.environ.get("XLA_FLAGS", "")
if "xla_force_host_platform_device_count" not in _existing_xla_flags:
    os.environ["XLA_FLAGS"] = (
        _existing_xla_flags + " --xla_force_host_platform_device_count=2"
    ).strip()

import unittest

import jax  # @manual
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from recir.lowering.xla_lower import _XLAModule


def _mesh() -> Mesh:
    return Mesh(np.array(jax.devices()), axis_names=("torch_dist",))


class ToShardedJaxTest(unittest.TestCase):
    def setUp(self) -> None:
        # The resharding path is only meaningful with more than one device.
        self.assertGreater(
            jax.device_count(), 1, "test requires >=2 spoofed CPU devices"
        )
        self.mesh: Mesh = _mesh()
        self.replicated: NamedSharding = NamedSharding(self.mesh, P())

    def test_single_device_array_resharded_to_replicated(self) -> None:
        # Mimics the cached device_state entry: a global-shaped array committed
        # to a single device (SingleDeviceSharding), as
        # _convert_flat_state_to_jax produces on the first step.
        host = np.arange(12, dtype=np.float32).reshape(3, 4)
        arr = jax.device_put(host, jax.devices()[0])
        self.assertIsInstance(arr.sharding, jax.sharding.SingleDeviceSharding)

        out = _XLAModule._to_sharded_jax(
            arr, self.mesh, is_batch_input=False, is_fsdp_weight=False
        )

        # Re-sharded onto the mesh as replicated, values + shape preserved.
        self.assertEqual(out.sharding, self.replicated)
        self.assertEqual(out.shape, host.shape)
        np.testing.assert_array_equal(np.asarray(jax.device_get(out)), host)

    def test_already_replicated_array_is_passthrough(self) -> None:
        # A jax.Array that already carries the target sharding (e.g. device_state
        # captured from the executable's mesh-sharded outputs on later steps) is
        # returned unchanged -- the reshard is a no-op.
        arr = jax.device_put(np.ones((2, 4), dtype=np.float32), self.replicated)
        out = _XLAModule._to_sharded_jax(
            arr, self.mesh, is_batch_input=False, is_fsdp_weight=False
        )
        self.assertIs(out, arr)

    def test_scalar_array_with_fsdp_flag_replicates(self) -> None:
        arr = jax.device_put(np.array(1.0, dtype=np.float32), jax.devices()[0])
        self.assertIsInstance(arr.sharding, jax.sharding.SingleDeviceSharding)

        out = _XLAModule._to_sharded_jax(arr, self.mesh, is_fsdp_weight=True)

        self.assertEqual(out.sharding, self.replicated)
        self.assertEqual(out.shape, ())
        self.assertEqual(float(jax.device_get(out)), 1.0)

    def test_single_device_array_resharded_to_fsdp(self) -> None:
        # An FSDP weight cached as a single-device jax.Array is re-sharded along
        # dim 0 across the mesh (P("torch_dist", ...)), not left single-device.
        # Single host here (host_count=1), so the shape is unchanged; dim 0 is
        # divisible by the device count so np.split is even.
        host = np.arange(16, dtype=np.float32).reshape(4, 4)
        arr = jax.device_put(host, jax.devices()[0])
        self.assertIsInstance(arr.sharding, jax.sharding.SingleDeviceSharding)

        out = _XLAModule._to_sharded_jax(arr, self.mesh, is_fsdp_weight=True)

        fsdp_sharding = NamedSharding(self.mesh, P("torch_dist", None))
        self.assertEqual(out.sharding, fsdp_sharding)
        self.assertEqual(out.shape, host.shape)
        np.testing.assert_array_equal(np.asarray(jax.device_get(out)), host)

    def test_already_fsdp_sharded_array_is_passthrough(self) -> None:
        # An FSDP jax.Array already in the target layout (e.g. captured from the
        # executable's outputs on later steps) is returned unchanged.
        fsdp_sharding = NamedSharding(self.mesh, P("torch_dist", None))
        arr = jax.device_put(
            np.arange(16, dtype=np.float32).reshape(4, 4), fsdp_sharding
        )
        out = _XLAModule._to_sharded_jax(arr, self.mesh, is_fsdp_weight=True)
        self.assertIs(out, arr)


if __name__ == "__main__":
    unittest.main()
