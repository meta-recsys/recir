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
    pallas_jagged_hstu_attention_backward,
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

    def test_backward_parity(self) -> None:
        for lengths, num_targets, max_seq_len in (
            ([7, 3, 9], [0, 0, 0], 9),
            ([0, 6, 2, 0], [0, 2, 1, 0], 8),
            ([3, 5], [0, 2], 8),
        ):
            with self.subTest(lengths=lengths, num_targets=num_targets):
                q, k, v, offsets = self._inputs(lengths)
                targets = jnp.asarray(num_targets, dtype=jnp.int32)
                alpha = 1.0 / math.sqrt(q.shape[-1])
                inv_n = 0.17
                output = jagged_hstu_attention_reference(
                    q,
                    k,
                    v,
                    offsets,
                    targets,
                    max_seq_len,
                    alpha,
                    inv_n,
                )
                grad_out = jax.random.normal(jax.random.key(7), output.shape)

                _, pullback = jax.vjp(
                    lambda q_,
                    k_,
                    v_,
                    offsets=offsets,
                    targets=targets,
                    max_seq_len=max_seq_len,
                    alpha=alpha,
                    inv_n=inv_n: jagged_hstu_attention_reference(
                        q_,
                        k_,
                        v_,
                        offsets,
                        targets,
                        max_seq_len,
                        alpha,
                        inv_n,
                    ),
                    q,
                    k,
                    v,
                )
                expected = pullback(grad_out)
                actual = pallas_jagged_hstu_attention_backward(
                    grad_out,
                    q,
                    k,
                    v,
                    offsets,
                    targets,
                    jnp.asarray(inv_n, dtype=jnp.float32),
                    max_seq_len=max_seq_len,
                    alpha=alpha,
                    interpret=True,
                    block_q=4,
                    block_k=4,
                )

                for actual_grad, expected_grad in zip(actual, expected):
                    np.testing.assert_allclose(
                        actual_grad,
                        expected_grad,
                        rtol=2e-4,
                        atol=2e-4,
                    )

    def test_native_backward_traces(self) -> None:
        q, k, v, offsets = self._inputs(
            [128],
            heads=1,
            dqk=128,
            dv=128,
        )
        q = q.astype(jnp.bfloat16)
        k = k.astype(jnp.bfloat16)
        v = v.astype(jnp.bfloat16)
        grad_out = jnp.ones((1, 1, 128, 128), dtype=jnp.bfloat16)
        targets = jnp.zeros((1,), dtype=jnp.int32)
        inv_n = jnp.asarray(1.0 / 128, dtype=jnp.float32)

        traced = jax.make_jaxpr(
            lambda do_, q_, k_, v_: pallas_jagged_hstu_attention_backward(
                do_,
                q_,
                k_,
                v_,
                offsets,
                targets,
                inv_n,
                max_seq_len=128,
                alpha=1.0 / math.sqrt(128),
                interpret=False,
                block_q=128,
                block_k=128,
            )
        )(grad_out, q, k, v)

        self.assertIn("pallas_call", str(traced))

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
