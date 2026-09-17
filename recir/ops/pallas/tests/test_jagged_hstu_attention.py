# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from __future__ import annotations

import math
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from recir.ops.pallas.jagged_hstu_attention import (
    jagged_hstu_attention_reference,
    pallas_jagged_hstu_attention,
)


class JaggedHSTUAttentionTest(unittest.TestCase):
    def _inputs(
        self,
        lengths: list[int],
        *,
        heads: int = 2,
        dqk: int = 8,
        dv: int = 6,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        total_length = sum(lengths)
        q_key, k_key, v_key = jax.random.split(jax.random.key(0), 3)
        q = jax.random.normal(q_key, (total_length, heads, dqk))
        k = jax.random.normal(k_key, (total_length, heads, dqk))
        v = jax.random.normal(v_key, (total_length, heads, dv))
        offsets = jnp.asarray([0, *np.cumsum(lengths)], dtype=jnp.int32)
        return q, k, v, offsets

    def _assert_parity(
        self,
        lengths: list[int],
        num_targets: list[int],
        *,
        max_seq_len: int,
        inv_n: float = 0.17,
    ) -> None:
        q, k, v, offsets = self._inputs(lengths)
        targets = jnp.asarray(num_targets, dtype=jnp.int32)
        scale = 1.0 / math.sqrt(q.shape[-1])

        actual = pallas_jagged_hstu_attention(
            q,
            k,
            v,
            offsets,
            targets,
            max_seq_len=max_seq_len,
            scale=scale,
            inv_n=inv_n,
            interpret=True,
            block_q=4,
            block_k=4,
        )
        expected = jagged_hstu_attention_reference(
            q,
            k,
            v,
            offsets,
            targets,
            max_seq_len,
            scale,
            inv_n,
        )

        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
        self.assertEqual(actual.shape, (len(lengths), 2, max_seq_len, 6))

    def test_nonuniform_causal_sequences(self) -> None:
        self._assert_parity([7, 3, 9], [0, 0, 0], max_seq_len=9)

    def test_target_attention_and_empty_sequences(self) -> None:
        self._assert_parity([0, 6, 2, 0], [0, 2, 1, 0], max_seq_len=8)

    def test_start_offset_residues(self) -> None:
        for residue in range(8):
            with self.subTest(residue=residue):
                self._assert_parity(
                    [residue, 5],
                    [0, 2],
                    max_seq_len=8,
                )

    def test_block_size_invariance(self) -> None:
        q, k, v, offsets = self._inputs([5, 9])
        targets = jnp.asarray([1, 3], dtype=jnp.int32)
        kwargs = {
            "max_seq_len": 9,
            "scale": 0.31,
            "inv_n": 0.07,
            "interpret": True,
        }

        small = pallas_jagged_hstu_attention(
            q, k, v, offsets, targets, block_q=4, block_k=4, **kwargs
        )
        mixed = pallas_jagged_hstu_attention(
            q, k, v, offsets, targets, block_q=8, block_k=4, **kwargs
        )

        np.testing.assert_allclose(small, mixed, rtol=2e-5, atol=2e-5)

    def test_rejects_invalid_shapes(self) -> None:
        q, k, v, offsets = self._inputs([3, 2])
        with self.assertRaisesRegex(ValueError, "shape \\[B\\]"):
            pallas_jagged_hstu_attention(
                q,
                k,
                v,
                offsets,
                jnp.zeros((3,), dtype=jnp.int32),
                max_seq_len=4,
                interpret=True,
            )
