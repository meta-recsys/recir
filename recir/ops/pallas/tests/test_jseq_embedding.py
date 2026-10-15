# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from types import SimpleNamespace
from typing import Any
from unittest import TestCase
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P
from recir.torchax.gspmd_context import current_gspmd_mesh_context, gspmd_mesh_context
from recir.torchax.ops import jseq_embedding


class JSeqEmbeddingManualMeshTest(TestCase):
    def test_forward_reuses_manual_mesh_local_driver(self) -> None:
        ids = jnp.array([[0, 1, 2], [3, 4, 5]], dtype=jnp.int32)
        lengths = jnp.array([2, 3], dtype=jnp.int32)
        weight = jnp.zeros((8, 4), dtype=jnp.bfloat16)
        driver_output = jnp.arange(24, dtype=jnp.float32).reshape(6, 4)

        with (
            patch.object(
                jseq_embedding.jax.sharding,
                "get_abstract_mesh",
                return_value=SimpleNamespace(
                    manual_axes={"torch_dist"}, shape={"torch_dist": 4}
                ),
            ),
            patch.object(
                jseq_embedding.jax.lax,
                "axis_index",
                return_value=jnp.asarray(1, dtype=jnp.int32),
            ),
            patch.object(
                jseq_embedding,
                "ringshift_embedding_lookup_block_local",
                return_value=driver_output,
            ) as driver,
            patch.object(jseq_embedding.jax, "shard_map") as shard_map,
        ):
            output = jseq_embedding._jringshift_embedding(ids, lengths, weight)

        self.assertEqual((2, 3, 4), output.shape)
        self.assertEqual(jnp.bfloat16, output.dtype)
        shard_map.assert_not_called()
        local_ids, count, local_weight = driver.call_args.args
        np.testing.assert_array_equal(
            np.asarray(local_ids), np.asarray([0, 1, 8, 3, 4, 5])
        )
        self.assertEqual(6, int(count))
        self.assertEqual((8, 4), local_weight.shape)
        self.assertEqual(jnp.float32, local_weight.dtype)
        self.assertEqual(4, driver.call_args.kwargs["world_size"])

    def test_backward_reuses_manual_mesh_local_driver(self) -> None:
        ids = jnp.array([[0, 1, 2], [3, 4, 5]], dtype=jnp.int32)
        lengths = jnp.array([3, 3], dtype=jnp.int32)
        weight = jnp.zeros((8, 4), dtype=jnp.bfloat16)
        grad_out = jnp.ones((2, 3, 4), dtype=jnp.bfloat16)

        with (
            patch.object(
                jseq_embedding.jax.sharding,
                "get_abstract_mesh",
                return_value=SimpleNamespace(
                    manual_axes={"torch_dist"}, shape={"torch_dist": 4}
                ),
            ),
            patch.object(
                jseq_embedding.jax.lax,
                "axis_index",
                return_value=jnp.asarray(1, dtype=jnp.int32),
            ),
            patch.object(
                jseq_embedding,
                "ringshift_embedding_lookup_block_backward_local",
                return_value=jnp.ones((8, 4), dtype=jnp.float32),
            ) as driver,
            patch.object(jseq_embedding.jax, "shard_map") as shard_map,
            patch.object(jseq_embedding.jax, "vjp") as vjp,
        ):
            grad_weight = jseq_embedding._jringshift_embedding_backward(
                grad_out,
                ids,
                lengths,
                weight,
                padding_idx=4,
            )

        self.assertEqual((8, 4), grad_weight.shape)
        self.assertEqual(jnp.bfloat16, grad_weight.dtype)
        shard_map.assert_not_called()
        vjp.assert_not_called()
        cotangent, local_ids, count = driver.call_args.args
        np.testing.assert_array_equal(np.asarray(local_ids), np.arange(6))
        np.testing.assert_array_equal(np.asarray(cotangent[4]), np.zeros(4))
        np.testing.assert_array_equal(
            np.asarray(cotangent[jnp.array([0, 1, 2, 3, 5])]), np.ones((5, 4))
        )
        self.assertEqual(6, int(count))
        self.assertEqual(4, driver.call_args.kwargs["world_size"])
        self.assertEqual(8, driver.call_args.kwargs["v_local"])

    def test_manual_mesh_rejects_configured_axis_mismatch(self) -> None:
        with (
            gspmd_mesh_context(object(), "torch_dist", 4, False),
            patch.object(
                jseq_embedding.jax.sharding,
                "get_abstract_mesh",
                return_value=SimpleNamespace(
                    manual_axes={"torch_dist"}, shape={"torch_dist": 8}
                ),
            ),
            self.assertRaisesRegex(ValueError, "does not match the active manual mesh"),
        ):
            jseq_embedding._manual_world_size()


class JSeqEmbeddingHybridMeshTest(TestCase):
    def test_config_rejects_nonpositive_fsdp_axis(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be positive"):
            with gspmd_mesh_context(object(), "torch_dist", 0, False):
                pass

    def test_config_is_restored_after_scope(self) -> None:
        self.assertIsNone(current_gspmd_mesh_context())

        with gspmd_mesh_context(object(), "torch_dist", 4, True):
            scoped_context = current_gspmd_mesh_context()
            self.assertIsNotNone(scoped_context)
            self.assertEqual(4, scoped_context.fsdp_axis_size)
            self.assertTrue(scoped_context.hybrid_fsdp)
            with self.assertRaisesRegex(RuntimeError, "stop trace"):
                with gspmd_mesh_context(object(), "torch_dist", 2, True):
                    nested_context = current_gspmd_mesh_context()
                    self.assertIsNotNone(nested_context)
                    self.assertEqual(2, nested_context.fsdp_axis_size)
                    raise RuntimeError("stop trace")
            restored_context = current_gspmd_mesh_context()
            self.assertIsNotNone(restored_context)
            self.assertEqual(4, restored_context.fsdp_axis_size)

        self.assertIsNone(current_gspmd_mesh_context())

    def test_non_ring_mesh_remains_full_world(self) -> None:
        devices = list(range(8))
        with (
            patch.object(jseq_embedding.jax, "devices", return_value=devices),
            patch.object(jseq_embedding, "Mesh", return_value=object()) as mesh,
        ):
            jseq_embedding._device_mesh()

        self.assertEqual((devices, ("torch_dist",)), mesh.call_args.args)

    def test_hybrid_mesh_rejects_cross_process_fsdp_group(self) -> None:
        devices = [
            SimpleNamespace(process_index=0),
            SimpleNamespace(process_index=1),
            SimpleNamespace(process_index=0),
            SimpleNamespace(process_index=1),
        ]
        fake_mesh = SimpleNamespace(
            devices=np.asarray(devices, dtype=object).reshape(2, 2)
        )
        with (
            gspmd_mesh_context(
                fake_mesh,
                ("gspmd_data", "torch_dist"),
                2,
                True,
            ),
            self.assertRaisesRegex(ValueError, "must not cross JAX process boundaries"),
        ):
            jseq_embedding._ringshift_mesh_layout()

    def test_forward_uses_host_local_fsdp_ring(self) -> None:
        ids = jnp.arange(24, dtype=jnp.int32).reshape(8, 3)
        lengths = jnp.full((8,), 3, dtype=jnp.int32)
        weight = jnp.zeros((32, 4), dtype=jnp.bfloat16)
        driver_output = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
        fake_mesh = SimpleNamespace(
            devices=np.asarray(list(range(8)), dtype=object).reshape(2, 4)
        )

        def fake_shard_map(
            fn: Any,
            *,
            mesh: Any,
            in_specs: Any,
            out_specs: Any,
            check_vma: bool,
        ) -> Any:
            self.assertIs(fake_mesh, mesh)
            self.assertEqual(P(("gspmd_data", "torch_dist"), None), in_specs[0])
            self.assertEqual(P("torch_dist", None), in_specs[2])
            self.assertEqual(P(("gspmd_data", "torch_dist"), None, None), out_specs)
            self.assertFalse(check_vma)

            def invoke(global_ids: Any, global_lengths: Any, global_weight: Any) -> Any:
                return fn(global_ids[:1], global_lengths[:1], global_weight[:8])

            return invoke

        with (
            gspmd_mesh_context(
                fake_mesh,
                ("gspmd_data", "torch_dist"),
                4,
                True,
            ),
            patch.object(
                jseq_embedding.jax,
                "shard_map",
                side_effect=fake_shard_map,
            ),
            patch.object(
                jseq_embedding.jax.lax,
                "axis_index",
                return_value=jnp.asarray(1, dtype=jnp.int32),
            ),
            patch.object(
                jseq_embedding,
                "ringshift_embedding_lookup_block_local",
                return_value=driver_output,
            ) as driver,
        ):
            output = jseq_embedding._jringshift_embedding(ids, lengths, weight)

        self.assertEqual((1, 3, 4), output.shape)
        self.assertEqual(4, driver.call_args.kwargs["world_size"])
        self.assertEqual((8, 4), driver.call_args.args[2].shape)

    def test_backward_sums_host_local_ring_gradient_across_data_axis(self) -> None:
        ids = jnp.arange(24, dtype=jnp.int32).reshape(8, 3)
        lengths = jnp.full((8,), 3, dtype=jnp.int32)
        weight = jnp.zeros((32, 4), dtype=jnp.bfloat16)
        grad_out = jnp.ones((8, 3, 4), dtype=jnp.bfloat16)
        fake_mesh = SimpleNamespace(
            devices=np.asarray(list(range(8)), dtype=object).reshape(2, 4)
        )

        def fake_shard_map(fn: Any, **kwargs: Any) -> Any:
            self.assertIs(fake_mesh, kwargs["mesh"])
            self.assertEqual(P("torch_dist", None), kwargs["out_specs"])

            def invoke(
                global_grad: Any,
                global_ids: Any,
                global_lengths: Any,
                global_weight: Any,
            ) -> Any:
                return fn(
                    global_grad[:1],
                    global_ids[:1],
                    global_lengths[:1],
                    global_weight[:8],
                )

            return invoke

        with (
            gspmd_mesh_context(
                fake_mesh,
                ("gspmd_data", "torch_dist"),
                4,
                True,
            ),
            patch.object(
                jseq_embedding.jax,
                "shard_map",
                side_effect=fake_shard_map,
            ),
            patch.object(
                jseq_embedding.jax.lax,
                "axis_index",
                return_value=jnp.asarray(1, dtype=jnp.int32),
            ),
            patch.object(
                jseq_embedding,
                "ringshift_embedding_lookup_block_backward_local",
                return_value=jnp.ones((8, 4), dtype=jnp.float32),
            ) as driver,
            patch.object(
                jseq_embedding.jax.lax,
                "psum",
                side_effect=lambda value, _axis: value * 2,
            ) as psum,
        ):
            grad_weight = jseq_embedding._jringshift_embedding_backward(
                grad_out,
                ids,
                lengths,
                weight,
            )

        self.assertEqual((8, 4), grad_weight.shape)
        self.assertEqual(jnp.bfloat16, grad_weight.dtype)
        self.assertEqual("gspmd_data", psum.call_args.args[1])
        self.assertEqual(4, driver.call_args.kwargs["world_size"])
        self.assertEqual(8, driver.call_args.kwargs["v_local"])
