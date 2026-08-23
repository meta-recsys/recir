# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from types import SimpleNamespace
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
from later.unittest import TestCase
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
