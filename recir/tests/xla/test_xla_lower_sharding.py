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
        _existing_xla_flags + " --xla_force_host_platform_device_count=4"
    ).strip()

import contextlib
import unittest
from unittest import mock

import jax  # @manual
import numpy as np
import torch
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from recir.lowering.xla_lower import _gspmd_mesh, _jax_to_torch, _XLAModule


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

    def test_batch_boundary_output_sharding_uses_batch_dim(self) -> None:
        graph = torch.fx.Graph()
        output = graph.placeholder("prediction")
        output.meta["val"] = torch.empty(81, 12800)
        output.meta["is_batch_input"] = True
        output.meta["batch_dim"] = 1

        sharding = _XLAModule._batch_boundary_output_sharding(output, self.mesh)

        self.assertIsNotNone(sharding)
        self.assertEqual(sharding.spec, P(None, "torch_dist"))

    def test_batch_boundary_skips_fsdp_state(self) -> None:
        graph = torch.fx.Graph()
        state = graph.placeholder("in_proj_weight")
        state.meta["val"] = torch.empty(768, 256)
        state.meta["is_batch_input"] = True
        state.meta["batch_dim"] = 1
        state.meta["is_fsdp_sharded"] = True

        sharding = _XLAModule._batch_boundary_output_sharding(state, self.mesh)

        self.assertIsNone(sharding)

    def test_batch_boundary_skips_explicit_sharded_state_output(self) -> None:
        graph = torch.fx.Graph()
        state = graph.placeholder("in_proj_weight")
        state.meta["val"] = torch.empty(768, 256)
        state.meta["is_batch_input"] = True
        state.meta["batch_dim"] = 1
        state.meta["is_fsdp_sharded_output"] = True

        sharding = _XLAModule._batch_boundary_output_sharding(state, self.mesh)

        self.assertIsNone(sharding)

    def test_batch_boundary_skips_explicit_replicated_state_output(self) -> None:
        graph = torch.fx.Graph()
        state = graph.placeholder("optimizer_step")
        state.meta["val"] = torch.empty(768, 256)
        state.meta["is_batch_input"] = True
        state.meta["batch_dim"] = 1
        state.meta["is_fsdp_sharded_output"] = False

        sharding = _XLAModule._batch_boundary_output_sharding(state, self.mesh)

        self.assertIsNone(sharding)

    def test_hybrid_mesh_shards_batch_over_data_and_fsdp_axes(self) -> None:
        devices = list(jax.devices())
        with mock.patch(
            "recir.lowering.xla_lower._get_local_devices",
            return_value=devices[:1],
        ):
            mesh, batch_resource, hybrid = _gspmd_mesh(devices, fsdp_axis_size=1)

        self.assertTrue(hybrid)
        self.assertEqual(("gspmd_data", "torch_dist"), mesh.axis_names)
        self.assertEqual(("gspmd_data", "torch_dist"), batch_resource)
        self.assertEqual(len(devices), mesh.shape["gspmd_data"])
        self.assertEqual(1, mesh.shape["torch_dist"])

        graph = torch.fx.Graph()
        output = graph.placeholder("prediction")
        output.meta["val"] = torch.empty(81, 12800)
        output.meta["is_batch_input"] = True
        output.meta["batch_dim"] = 0
        sharding = _XLAModule._batch_boundary_output_sharding(
            output,
            mesh,
            batch_resource,
        )
        self.assertIsNotNone(sharding)
        self.assertEqual(sharding.spec, P(("gspmd_data", "torch_dist"), None))

    def test_hybrid_mesh_supports_multi_host_fsdp_groups(self) -> None:
        devices = list(jax.devices())
        with mock.patch(
            "recir.lowering.xla_lower._get_local_devices",
            return_value=devices[:1],
        ):
            mesh, batch_resource, hybrid = _gspmd_mesh(
                devices,
                fsdp_axis_size=2,
            )

        self.assertTrue(hybrid)
        self.assertEqual(("gspmd_data", "torch_dist"), mesh.axis_names)
        self.assertEqual(("gspmd_data", "torch_dist"), batch_resource)
        self.assertEqual(len(devices) // 2, mesh.shape["gspmd_data"])
        self.assertEqual(2, mesh.shape["torch_dist"])

    def test_hybrid_lowering_uses_two_axis_batch_resource_for_examples(self) -> None:
        graph = torch.fx.Graph()
        batch = graph.placeholder("batch")
        batch.meta["val"] = torch.empty(2, 4)
        batch.meta["is_batch_input"] = True
        batch.meta["batch_dim"] = 0
        graph.output(batch)
        gm = torch.fx.GraphModule({}, graph)
        gm.meta["gspmd_fsdp_axis_size"] = 2
        module = _XLAModule(gm)
        fake_jitted = mock.Mock()
        context_events: list[tuple[str, object, object, int, bool] | str] = []

        @contextlib.contextmanager
        def fake_mesh_context(
            mesh: object,
            batch_axis_resource: object,
            fsdp_axis_size: int,
            hybrid_fsdp: bool,
        ):
            context_events.append(
                (
                    "enter",
                    mesh,
                    batch_axis_resource,
                    fsdp_axis_size,
                    hybrid_fsdp,
                )
            )
            try:
                yield
            finally:
                context_events.append(
                    (
                        "exit",
                        mesh,
                        batch_axis_resource,
                        fsdp_axis_size,
                        hybrid_fsdp,
                    )
                )

        def fake_lower(*_args: object) -> object:
            context_events.append("lower")
            return mock.sentinel.lowered

        fake_jitted.lower.side_effect = fake_lower

        with (
            mock.patch(
                "recir.lowering.xla_lower._get_local_devices",
                return_value=list(jax.devices())[:1],
            ),
            mock.patch("recir.lowering.xla_lower.jax.jit", return_value=fake_jitted),
            mock.patch(
                "recir.lowering.xla_lower.gspmd_mesh_context",
                side_effect=fake_mesh_context,
            ),
            mock.patch.object(
                module,
                "_to_sharded_jax",
                return_value=mock.sentinel.example_arg,
            ) as to_sharded,
        ):
            lowered, _ = module._jit_gspmd(
                lambda value: value,
                (torch.ones(2, 4),),
                torch.device("cpu"),
            )

        self.assertIs(lowered, mock.sentinel.lowered)
        to_sharded.assert_called_once()
        self.assertEqual(
            to_sharded.call_args.kwargs["batch_axis_resource"],
            ("gspmd_data", "torch_dist"),
        )
        self.assertEqual(to_sharded.call_args.kwargs["fsdp_axis_name"], "torch_dist")
        self.assertEqual(to_sharded.call_args.kwargs["fsdp_host_shard_count"], 2)
        self.assertEqual("enter", context_events[0][0])
        self.assertEqual(
            ("gspmd_data", "torch_dist"),
            context_events[0][2],
        )
        self.assertEqual(2, context_events[0][3])
        self.assertTrue(context_events[0][4])
        self.assertEqual("lower", context_events[1])
        self.assertEqual("exit", context_events[2][0])
        self.assertIs(context_events[0][1], context_events[2][1])

    def test_dim_one_sharded_output_reconstructs_dim_one(self) -> None:
        host = np.arange(24, dtype=np.float32).reshape(3, 8)
        sharding = NamedSharding(self.mesh, P(None, "torch_dist"))
        sharded = jax.device_put(host, sharding)

        output = _jax_to_torch(sharded, torch.device("cpu"))

        self.assertEqual(output.shape, torch.Size([3, 8]))
        torch.testing.assert_close(output, torch.from_numpy(host))

    def test_fsdp_state_is_not_replicated_as_final_user_output(self) -> None:
        graph = torch.fx.Graph()
        output = graph.placeholder("state")

        self.assertTrue(_XLAModule._is_replicated_final_output(0, output, {0}))

        output.meta["is_fsdp_sharded_output"] = True
        self.assertFalse(_XLAModule._is_replicated_final_output(0, output, {0}))

    def test_single_device_array_resharded_to_replicated(self) -> None:
        # Mimics the cached device_state entry: a global-shaped array committed
        # to a single device (SingleDeviceSharding), as
        # _convert_flat_state_to_jax produces on the first step.
        host = np.arange(12, dtype=np.float32).reshape(3, 4)
        arr = jax.device_put(host, jax.devices()[0])
        self.assertIsInstance(arr.sharding, jax.sharding.SingleDeviceSharding)

        out = _XLAModule._to_sharded_jax(
            arr, self.mesh, is_batch_input=False, is_fsdp_tensor=False
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
            arr, self.mesh, is_batch_input=False, is_fsdp_tensor=False
        )
        self.assertIs(out, arr)

    def test_scalar_array_with_fsdp_flag_replicates(self) -> None:
        arr = jax.device_put(np.array(1.0, dtype=np.float32), jax.devices()[0])
        self.assertIsInstance(arr.sharding, jax.sharding.SingleDeviceSharding)

        out = _XLAModule._to_sharded_jax(arr, self.mesh, is_fsdp_tensor=True)

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

        out = _XLAModule._to_sharded_jax(arr, self.mesh, is_fsdp_tensor=True)

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
        out = _XLAModule._to_sharded_jax(arr, self.mesh, is_fsdp_tensor=True)
        self.assertIs(out, arr)


class HostArrayAssemblyTest(unittest.TestCase):
    """A torch input must reach the mesh in ONE dispatch, not one per device.

    Assembling the global array as ``[jax.device_put(shard, dev) for dev in
    local_devices]`` + ``make_array_from_single_device_arrays`` costs a full JAX
    dispatch per local device. On a 128-chip rankfm_mvai step that was 98 host
    tensors x 8 local devices = 784 dispatches, 157 ms/step, for 15 ms of actual
    H2D. ``make_array_from_process_local_data`` reaches a single
    ``xc.batched_device_put`` covering all of this host's shards instead.

    Counting ``jax.device_put`` catches a reintroduced per-device loop: the loop
    resolves it as a module attribute (so the patch sees it), while the current
    path never calls it by that name. A value-only assertion would pass either
    way, so it would not guard the thing that regressed.
    """

    def setUp(self) -> None:
        self.assertGreater(
            jax.device_count(), 1, "test requires >=2 spoofed CPU devices"
        )
        self.mesh: Mesh = _mesh()
        self.n: int = len(jax.local_devices())

    def _assert_one_dispatch(self, tensor: torch.Tensor, **kwargs: object) -> object:
        real = jax.device_put
        with mock.patch.object(
            jax, "device_put", side_effect=real, autospec=True
        ) as spy:
            out = _XLAModule._to_sharded_jax(tensor, self.mesh, **kwargs)
        self.assertLessEqual(
            spy.call_count,
            1,
            f"{spy.call_count} jax.device_put calls for one input; the "
            "per-local-device assembly loop is back",
        )
        return out

    def test_replicated_input(self) -> None:
        host = np.arange(6, dtype=np.float32)
        out = self._assert_one_dispatch(torch.from_numpy(host))
        self.assertEqual(out.sharding, NamedSharding(self.mesh, P()))
        np.testing.assert_array_equal(np.asarray(jax.device_get(out)), host)

    def test_batch_input(self) -> None:
        host = np.arange(4 * self.n * 3, dtype=np.float32).reshape(4 * self.n, 3)
        out = self._assert_one_dispatch(torch.from_numpy(host), is_batch_input=True)
        self.assertEqual(out.sharding, NamedSharding(self.mesh, P("torch_dist", None)))
        np.testing.assert_array_equal(np.asarray(jax.device_get(out)), host)

    def test_fsdp_input(self) -> None:
        host = np.arange(2 * self.n * 5, dtype=np.float32).reshape(2 * self.n, 5)
        out = self._assert_one_dispatch(torch.from_numpy(host), is_fsdp_tensor=True)
        self.assertEqual(out.sharding, NamedSharding(self.mesh, P("torch_dist", None)))
        np.testing.assert_array_equal(np.asarray(jax.device_get(out)), host)

    def test_bool_input_is_not_recopied(self) -> None:
        # torch.bool -> .numpy() is already np.bool_, so the astype() the old
        # path applied was a pure copy: 5.7 ms/step for one [134676, 28] mask.
        host = (np.arange(4 * self.n * 2).reshape(4 * self.n, 2) % 3) == 0
        out = self._assert_one_dispatch(torch.from_numpy(host), is_batch_input=True)
        self.assertEqual(out.dtype, np.bool_)
        np.testing.assert_array_equal(np.asarray(jax.device_get(out)), host)


if __name__ == "__main__":
    unittest.main()
