# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


import math
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from recir.ops.pallas.hstu_attention import (
    hstu_attention_reference,
    pallas_hstu_attention,
    pallas_hstu_attention_backward,
)


class HstuAttentionTest(unittest.TestCase):
    def _inputs(self, b=2, h=2, seq_len=16, d=8, seed=0):
        key = jax.random.PRNGKey(seed)
        kq, kk, kv = jax.random.split(key, 3)
        q = jax.random.normal(kq, (b, h, seq_len, d), dtype=jnp.float32)
        k = jax.random.normal(kk, (b, h, seq_len, d), dtype=jnp.float32)
        v = jax.random.normal(kv, (b, h, seq_len, d), dtype=jnp.float32)
        return q, k, v

    def _scales(self, q):
        return 1.0 / math.sqrt(q.shape[3]), 1.0 / q.shape[2]  # alpha, inv_n

    def test_matches_reference_causal_full_length(self):
        q, k, v = self._inputs()
        alpha, inv_n = self._scales(q)
        lengths = jnp.full((q.shape[0],), q.shape[2], dtype=jnp.int32)
        num_targets = jnp.zeros((q.shape[0],), dtype=jnp.int32)  # pure causal
        out = pallas_hstu_attention(
            q, k, v, lengths, num_targets, block_k=8, block_q=8, interpret=True
        )
        ref = hstu_attention_reference(q, k, v, lengths, num_targets, alpha, inv_n)
        np.testing.assert_allclose(
            np.asarray(out), np.asarray(ref), rtol=1e-5, atol=1e-5
        )

    def test_target_attention_and_length_mask(self):
        q, k, v = self._inputs(seed=1)
        alpha, inv_n = self._scales(q)
        # row 0: full length, 3 candidates; row 1: padded to 5, 2 candidates.
        lengths = jnp.array([q.shape[2], 5], dtype=jnp.int32)
        num_targets = jnp.array([3, 2], dtype=jnp.int32)
        out = pallas_hstu_attention(
            q, k, v, lengths, num_targets, block_k=8, block_q=8, interpret=True
        )
        ref = hstu_attention_reference(q, k, v, lengths, num_targets, alpha, inv_n)
        np.testing.assert_allclose(
            np.asarray(out), np.asarray(ref), rtol=1e-5, atol=1e-5
        )
        # padded query rows (>=5) of row 1 are exactly zero.
        self.assertTrue(bool(jnp.all(out[1, :, 5:, :] == 0.0)))

    def test_nondivisible_query_and_kv_blocks_match_reference(self):
        # Neither block divides seq_len; common-boundary padding stays masked.
        q, k, v = self._inputs(seed=5, seq_len=16, d=8)
        alpha, inv_n = self._scales(q)
        lengths = jnp.array([16, 11], dtype=jnp.int32)
        num_targets = jnp.array([2, 3], dtype=jnp.int32)
        out = pallas_hstu_attention(
            q, k, v, lengths, num_targets, block_k=6, block_q=10, interpret=True
        )
        ref = hstu_attention_reference(q, k, v, lengths, num_targets, alpha, inv_n)
        np.testing.assert_allclose(
            np.asarray(out), np.asarray(ref), rtol=1e-5, atol=1e-5
        )

    def test_query_and_kv_block_invariance(self):
        # Tiling is exact (no softmax): query and KV block choices preserve results.
        q, k, v = self._inputs(seed=6, seq_len=24, d=8)
        alpha, inv_n = self._scales(q)
        lengths = jnp.array([24, 20], dtype=jnp.int32)
        num_targets = jnp.array([4, 2], dtype=jnp.int32)
        ref = hstu_attention_reference(q, k, v, lengths, num_targets, alpha, inv_n)
        for bk, bq in ((4, 6), (8, 8), (24, 12)):
            out = pallas_hstu_attention(
                q, k, v, lengths, num_targets, block_k=bk, block_q=bq, interpret=True
            )
            np.testing.assert_allclose(
                np.asarray(out), np.asarray(ref), rtol=1e-5, atol=1e-5
            )

    def test_backward_matches_reference_grads(self):
        q, k, v = self._inputs(seed=3)
        alpha, inv_n = self._scales(q)
        lengths = jnp.array([q.shape[2], 7], dtype=jnp.int32)
        num_targets = jnp.array([2, 1], dtype=jnp.int32)

        def kernel_loss(q_, k_, v_):
            o = pallas_hstu_attention(
                q_, k_, v_, lengths, num_targets, block_k=8, block_q=8, interpret=True
            )
            return jnp.sum(o.astype(jnp.float32) ** 2)

        def ref_loss(q_, k_, v_):
            o = hstu_attention_reference(q_, k_, v_, lengths, num_targets, alpha, inv_n)
            return jnp.sum(o.astype(jnp.float32) ** 2)

        gk = jax.grad(kernel_loss, argnums=(0, 1, 2))(q, k, v)
        gr = jax.grad(ref_loss, argnums=(0, 1, 2))(q, k, v)
        for a, b in zip(gk, gr):
            np.testing.assert_allclose(
                np.asarray(a), np.asarray(b), rtol=1e-4, atol=1e-4
            )

    def test_tiled_backward_with_distinct_qk_and_v_dimensions(self):
        q, k, _ = self._inputs(seed=7, seq_len=12, d=8)
        v = jax.random.normal(jax.random.PRNGKey(8), (2, 2, 12, 6), dtype=jnp.float32)
        grad_out = jax.random.normal(jax.random.PRNGKey(9), v.shape, dtype=jnp.float32)
        alpha, inv_n = self._scales(q)
        lengths = jnp.array([12, 7], dtype=jnp.int32)
        num_targets = jnp.array([3, 2], dtype=jnp.int32)

        grads = pallas_hstu_attention_backward(
            grad_out,
            q,
            k,
            v,
            lengths,
            num_targets,
            alpha,
            inv_n,
            block_k=4,
            block_q=6,
            interpret=True,
        )
        _, vjp_fn = jax.vjp(
            lambda q_, k_, v_: hstu_attention_reference(
                q_, k_, v_, lengths, num_targets, alpha, inv_n
            ),
            q,
            k,
            v,
        )
        reference_grads = vjp_fn(grad_out)
        for actual, expected in zip(grads, reference_grads):
            np.testing.assert_allclose(
                np.asarray(actual), np.asarray(expected), rtol=1e-4, atol=1e-4
            )

    def test_output_shape_and_dtype(self):
        q, k, v = self._inputs(b=3, h=4, seq_len=8, d=16, seed=4)
        lengths = jnp.array([8, 3, 0], dtype=jnp.int32)
        out = pallas_hstu_attention(
            q, k, v, lengths, block_k=8, block_q=8, interpret=True
        )
        self.assertEqual(out.shape, (3, 4, 8, 16))
        self.assertEqual(out.dtype, q.dtype)
        self.assertTrue(bool(jnp.all(out[2] == 0.0)))  # zero-length row -> zero
