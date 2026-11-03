# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the pooled lowering's contract.

There used to be more here. The lowering carried ``_pack_bag_major``, which
compacted a padded rectangle into the flat bag-major layout the kernel wants,
and that index arithmetic was pure JAX and worth pinning down. The op now takes
that layout directly, so the packing -- and the eight tests covering it -- are
gone rather than merely untested.

What remains CPU-checkable is the contract itself: the shape the lowering
promises, and the one caller mistake it can still catch. Everything else runs
on a TPU slice.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
from recir.torchax.ops import jseq_embedding
from recir.torchax.ops.jseq_embedding import _pooled_lookup


_DIM = 8
_VOCAB = 32
_ROWS = len(jax.devices())
_COLS = 6
_NBAGS = 3


def _inputs():
    ids = jnp.arange(_ROWS * _COLS, dtype=jnp.int32).reshape(_ROWS, _COLS) % _VOCAB
    # 3 + 0 + 3 = 6 = _COLS. The empty bag is deliberate: it is the case a
    # start-of-bag computation is most likely to get wrong.
    bags = jnp.broadcast_to(jnp.asarray([[3, 0, 3]], dtype=jnp.int32), (_ROWS, _NBAGS))
    weight = jnp.zeros((_VOCAB, _DIM), jnp.float32)
    return ids, bags, weight


class PooledLoweringContractTest(unittest.TestCase):
    @unittest.skipUnless(len(jax.devices()) >= 2, "Requires two JAX devices")
    def test_hybrid_fp16_gradient_is_summed_before_narrowing(self) -> None:
        mesh = jax.sharding.Mesh(
            [[device] for device in jax.devices()[:2]],
            ("gspmd_data", "torch_dist"),
        )
        ids = jnp.zeros((2, 1), jnp.int32)
        lengths = jnp.ones((2, 1), jnp.int32)
        weight = jnp.zeros((1, 1), jnp.float16)
        grad_out = jnp.full((2, 1, 1), 2e-8, jnp.float32)
        for pooling in ("sum", "mean"):
            with (
                self.subTest(pooling=pooling),
                patch.object(jseq_embedding, "_manual_world_size", return_value=None),
                patch.object(
                    jseq_embedding,
                    "_ringshift_mesh_layout",
                    return_value=(mesh, ("gspmd_data", "torch_dist"), 1, True),
                ),
                patch.object(
                    jseq_embedding,
                    "ringshift_embedding_lookup_block_backward_local",
                    return_value=jnp.full((1, 1), 2e-8, jnp.float32),
                ),
            ):
                gradient = jax.jit(
                    lambda g,
                    pooling=pooling: jseq_embedding._jringshift_embedding_backward(
                        g, ids, lengths, weight, pooling=pooling
                    )
                )(grad_out)
            self.assertEqual(gradient.dtype, jnp.float16)
            self.assertEqual(float(gradient[0, 0]), 2**-24)

    def test_output_follows_the_bag_lengths_not_the_ids(self) -> None:
        """``[W, C]`` ids and ``[W, K]`` lengths give ``[W, K, D]``.

        Traced, not executed: ``jax.eval_shape`` stops before the Pallas call,
        which has no CPU backend. That is enough to catch the shape drifting
        away from the op's fake kernel, which is what would make the compiler
        and the runtime disagree.
        """
        ids, bags, weight = _inputs()
        out = jax.eval_shape(lambda w: _pooled_lookup(ids, bags, w, "sum"), weight)
        self.assertEqual(out.shape, (_ROWS, _NBAGS, _DIM))
        self.assertEqual(out.dtype, jnp.float32)

    def test_fp16_table_returns_float32_and_an_fp16_gradient(self) -> None:
        """The FP32 pooled sum is not rounded to FP16; only the table gradient is."""
        ids, bags, _weight = _inputs()
        weight = jnp.zeros((_VOCAB, _DIM), jnp.float16)
        out = jax.eval_shape(lambda w: _pooled_lookup(ids, bags, w, "sum"), weight)
        grad = jax.eval_shape(
            jax.grad(lambda w: _pooled_lookup(ids, bags, w, "sum").sum()), weight
        )
        self.assertEqual(out.dtype, jnp.float32)
        self.assertEqual(grad.dtype, jnp.float16)

    def test_mean_has_the_same_shape_as_sum(self) -> None:
        ids, bags, weight = _inputs()
        out = jax.eval_shape(lambda w: _pooled_lookup(ids, bags, w, "mean"), weight)
        self.assertEqual(out.shape, (_ROWS, _NBAGS, _DIM))

    def test_ids_not_laid_out_one_row_per_device_are_rejected(self) -> None:
        """The one caller mistake this layer can still catch.

        Axis 0 of ``ids`` is the mesh axis. Handing it bags instead of chips
        gives shapes that all line up and a result that is silently wrong, so
        the guard is the only thing standing between that and a bad number.
        """
        _ids, _bags, weight = _inputs()
        bag_major = jnp.zeros((_ROWS + 1, _COLS), jnp.int32)
        lengths = jnp.zeros((_ROWS + 1, _NBAGS), jnp.int32)
        with self.assertRaisesRegex(ValueError, "one row per device"):
            _pooled_lookup(bag_major, lengths, weight, "sum")
