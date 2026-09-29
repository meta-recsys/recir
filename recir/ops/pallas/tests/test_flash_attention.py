# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Correctness tests for the optimized Pallas FlashAttention kernel.

All tests use ``interpret=True`` to run on CPU for CI (no TPU required).
Reference attention is implemented in pure JAX/numpy — no PyTorch dependency.
"""

from __future__ import annotations

import math
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from recir.ops.pallas.flash_attention import (
    pallas_flash_attention_batched,
    pallas_flash_attention_varlen,
)


# ---------------------------------------------------------------------------
# Reference implementations (pure numpy/jax, CPU)
# ---------------------------------------------------------------------------


def _build_target_attention_mask_np(
    seq_len: int,
    prefix_len: int,
    tokens_per_item: int,
) -> np.ndarray:
    """Build [seq_len, seq_len] boolean mask for a single sequence."""
    mask = np.zeros((seq_len, seq_len), dtype=bool)
    for qi in range(seq_len):
        for ki in range(seq_len):
            q_in_prefix = qi < prefix_len
            k_in_prefix = ki < prefix_len
            if q_in_prefix and k_in_prefix and ki <= qi:
                mask[qi, ki] = True
            elif (not q_in_prefix) and k_in_prefix:
                mask[qi, ki] = True
            elif (not q_in_prefix) and (not k_in_prefix):
                q_item = (qi - prefix_len) // tokens_per_item
                k_item = (ki - prefix_len) // tokens_per_item
                if q_item == k_item and ki <= qi:
                    mask[qi, ki] = True
    return mask


def _reference_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    cu_seqlens: list[int],
    prefix_lens: list[int],
    tokens_per_item_list: list[int],
    num_kv_heads: int,
) -> np.ndarray:
    """Reference manual attention with target mask.

    Args:
        q: [total_tokens, num_heads, head_dim] float32
        k: [total_tokens, num_kv_heads, head_dim] float32
        v: [total_tokens, num_kv_heads, head_dim] float32
    Returns:
        [total_tokens, num_heads, head_dim] float32
    """
    total_tokens, num_heads, head_dim = q.shape
    groups = num_heads // num_kv_heads
    scale = 1.0 / math.sqrt(head_dim)

    # Build block-diagonal mask
    full_mask = np.full((total_tokens, total_tokens), float("-inf"))
    batch_size = len(cu_seqlens) - 1
    for b in range(batch_size):
        start = cu_seqlens[b]
        end = cu_seqlens[b + 1]
        seq_len = end - start
        seq_mask = _build_target_attention_mask_np(
            seq_len, prefix_lens[b], tokens_per_item_list[b]
        )
        # Convert bool to additive: True -> 0.0, False -> -inf
        additive = np.where(seq_mask, 0.0, float("-inf"))
        full_mask[start:end, start:end] = additive

    # Expand KV heads for GQA
    k_expanded = np.repeat(k, groups, axis=1)  # [T, H, D]
    v_expanded = np.repeat(v, groups, axis=1)

    out = np.zeros_like(q)
    for h in range(num_heads):
        q_h = q[:, h, :]  # [T, D]
        k_h = k_expanded[:, h, :]
        v_h = v_expanded[:, h, :]

        scores = (q_h @ k_h.T) * scale + full_mask  # [T, T]
        # Stable softmax
        scores_max = np.max(scores, axis=-1, keepdims=True)
        scores_max = np.where(
            np.isfinite(scores_max), scores_max, np.zeros_like(scores_max)
        )
        exp_scores = np.exp(scores - scores_max)
        attn = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        out[:, h, :] = attn @ v_h

    return out


# ---------------------------------------------------------------------------
# Helper to create inputs
# ---------------------------------------------------------------------------


def _make_inputs(
    batch_configs: list[tuple[int, int, int]],
    num_heads: int = 16,
    num_kv_heads: int = 8,
    head_dim: int = 64,
):
    """Create test inputs.

    Args:
        batch_configs: list of (prefix_len, num_items, item_len) per sequence.

    Returns:
        dict with JAX arrays and metadata.
    """
    seq_lens = []
    prefix_lens = []
    tpi_list = []

    for prefix_len, num_items, item_len in batch_configs:
        seq_lens.append(prefix_len + num_items * item_len)
        prefix_lens.append(prefix_len)
        tpi_list.append(item_len)

    total_tokens = sum(seq_lens)
    cu_seqlens = [0]
    for sl in seq_lens:
        cu_seqlens.append(cu_seqlens[-1] + sl)

    # pyrefly: ignore [bad-argument-type]
    rng = np.random.RandomState(42)
    q_np = rng.randn(total_tokens, num_heads, head_dim).astype(np.float32)
    k_np = rng.randn(total_tokens, num_kv_heads, head_dim).astype(np.float32)
    v_np = rng.randn(total_tokens, num_kv_heads, head_dim).astype(np.float32)

    q_jax = jnp.array(q_np).astype(jnp.bfloat16)
    k_jax = jnp.array(k_np).astype(jnp.bfloat16)
    v_jax = jnp.array(v_np).astype(jnp.bfloat16)
    cu_seqlens_jax = jnp.array(cu_seqlens, dtype=jnp.int32)
    prefix_lens_jax = jnp.array(prefix_lens, dtype=jnp.int32)
    tpi_jax = jnp.array(tpi_list, dtype=jnp.int32)

    return {
        "q_jax": q_jax,
        "k_jax": k_jax,
        "v_jax": v_jax,
        "cu_seqlens_jax": cu_seqlens_jax,
        "prefix_lens_jax": prefix_lens_jax,
        "tpi_jax": tpi_jax,
        # For reference impl
        "q_np": q_np,
        "k_np": k_np,
        "v_np": v_np,
        "cu_seqlens": cu_seqlens,
        "prefix_lens": prefix_lens,
        "tpi_list": tpi_list,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
    }


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class FlashAttentionTest(unittest.TestCase):
    """Tests for optimized Pallas FlashAttention kernel correctness."""

    def _run_and_compare(
        self,
        batch_configs: list[tuple[int, int, int]],
        num_heads: int = 16,
        num_kv_heads: int = 8,
        head_dim: int = 64,
        atol: float = 1e-2,
        rtol: float = 1e-2,
    ) -> None:
        inputs = _make_inputs(
            batch_configs,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )
        sm_scale = 1.0 / math.sqrt(head_dim)

        # Pallas output (interpret=True for CPU)
        pallas_out = pallas_flash_attention_varlen(
            q=inputs["q_jax"],
            k=inputs["k_jax"],
            v=inputs["v_jax"],
            cu_seqlens=inputs["cu_seqlens_jax"],
            prefix_lens=inputs["prefix_lens_jax"],
            tokens_per_item=inputs["tpi_jax"],
            sm_scale=sm_scale,
            block_q=32,
            block_k=32,
            interpret=True,
        )
        pallas_np = np.array(pallas_out.astype(jnp.float32))

        # Reference output (pure numpy)
        ref_np = _reference_attention(
            # Use bf16 -> f32 round-tripped values for fair comparison
            np.array(inputs["q_jax"].astype(jnp.float32)),
            np.array(inputs["k_jax"].astype(jnp.float32)),
            np.array(inputs["v_jax"].astype(jnp.float32)),
            inputs["cu_seqlens"],
            inputs["prefix_lens"],
            inputs["tpi_list"],
            inputs["num_kv_heads"],
        )

        np.testing.assert_allclose(
            pallas_np,
            ref_np,
            atol=atol,
            rtol=rtol,
        )

    def test_basic_single_sequence(self) -> None:
        """Single sequence: prefix=64, 4 items x 8 tokens."""
        self._run_and_compare([(64, 4, 8)])

    def test_batched_sequences(self) -> None:
        """Batch=2 with different prefix lengths."""
        self._run_and_compare(
            [
                (32, 4, 8),  # prefix=32, 4 items x 8 tokens
                (48, 3, 4),  # prefix=48, 3 items x 4 tokens
            ]
        )

    def test_single_token_items(self) -> None:
        """Edge case: each item is a single token (item_len=1)."""
        self._run_and_compare([(32, 8, 1)])

    def test_gqa(self) -> None:
        """GQA with num_heads=16, num_kv_heads=8."""
        self._run_and_compare(
            [(64, 4, 8)],
            num_heads=16,
            num_kv_heads=8,
        )

    def test_many_items(self) -> None:
        """Stress test: 32 items, prefix=64, item_len=4."""
        self._run_and_compare([(64, 32, 4)])


# ---------------------------------------------------------------------------
# Reference batched attention with target mask (pure JAX, differentiable)
# ---------------------------------------------------------------------------


def _build_target_attention_mask_jax(
    seq_len: int,
    prefix_len: int,
    tokens_per_item: int,
) -> jax.Array:
    """Build [seq_len, seq_len] additive mask (0 or -inf) using pure JAX."""
    qi = jnp.arange(seq_len)[:, None]
    ki = jnp.arange(seq_len)[None, :]
    q_in_prefix = qi < prefix_len
    k_in_prefix = ki < prefix_len
    causal = ki <= qi
    both_prefix = q_in_prefix & k_in_prefix & causal
    q_item_k_prefix = (~q_in_prefix) & k_in_prefix
    q_item = (qi - prefix_len) // tokens_per_item
    k_item = (ki - prefix_len) // tokens_per_item
    same_item = q_item == k_item
    both_items = (~q_in_prefix) & (~k_in_prefix) & same_item & causal
    bool_mask = both_prefix | q_item_k_prefix | both_items
    return jnp.where(bool_mask, 0.0, -1e30)


def _reference_batched_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    prefix_len: int,
    tokens_per_item: int,
    sm_scale: float,
) -> jax.Array:
    """Naive einsum-based batched attention with target mask.

    Used as the differentiable reference for gradient tests.

    Args:
        q: [B, S, H, D]
        k: [B, S, Hkv, D]
        v: [B, S, Hkv, D]
    Returns:
        [B, S, H, D]
    """
    seq_len = q.shape[1]
    num_heads = q.shape[2]
    num_kv_heads = k.shape[2]
    groups = num_heads // num_kv_heads

    # Repeat KV across head groups to match Q.
    k_full = jnp.repeat(k, groups, axis=2)  # [B, S, H, D]
    v_full = jnp.repeat(v, groups, axis=2)

    # Build mask once; same for all batches/heads.
    mask = _build_target_attention_mask_jax(seq_len, prefix_len, tokens_per_item)
    # [B, H, S_q, S_k]
    scores = jnp.einsum("bshd,bthd->bhst", q, k_full) * sm_scale
    scores = scores + mask  # broadcast over (B, H)
    attn = jax.nn.softmax(scores, axis=-1)
    # [B, S_q, H, D]
    out = jnp.einsum("bhst,bthd->bshd", attn, v_full)
    return out


# ---------------------------------------------------------------------------
# Backward pass tests
# ---------------------------------------------------------------------------


class FlashAttentionGradientTest(unittest.TestCase):
    """Tests for the ``custom_vjp`` backward of the batched Pallas kernel."""

    def _run_grad_test(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        prefix_len: int,
        tokens_per_item: int,
        block_q: int = 32,
        block_k: int = 32,
        atol: float = 5e-2,
        rtol: float = 5e-2,
    ) -> None:
        sm_scale = 1.0 / math.sqrt(head_dim)
        # pyrefly: ignore [bad-argument-type]
        rng = np.random.RandomState(123)
        q_np = rng.randn(batch_size, seq_len, num_heads, head_dim).astype(np.float32)
        k_np = rng.randn(batch_size, seq_len, num_kv_heads, head_dim).astype(np.float32)
        v_np = rng.randn(batch_size, seq_len, num_kv_heads, head_dim).astype(np.float32)
        do_np = rng.randn(batch_size, seq_len, num_heads, head_dim).astype(np.float32)
        # Use float32 throughout so gradient comparison is meaningful.
        q = jnp.array(q_np)
        k = jnp.array(k_np)
        v = jnp.array(v_np)
        do = jnp.array(do_np)

        def pallas_loss(q, k, v):
            out = pallas_flash_attention_batched(  # pyre-ignore[29]
                q,
                k,
                v,
                prefix_len=prefix_len,
                tokens_per_item=tokens_per_item,
                sm_scale=sm_scale,
                block_q=block_q,
                block_k=block_k,
                interpret=True,
            )
            return jnp.sum(out * do)

        def ref_loss(q, k, v):
            out = _reference_batched_attention(
                q,
                k,
                v,
                prefix_len=prefix_len,
                tokens_per_item=tokens_per_item,
                sm_scale=sm_scale,
            )
            return jnp.sum(out * do)

        # Compute Pallas output to verify forward correctness too.
        pallas_out = pallas_flash_attention_batched(  # pyre-ignore[29]
            q,
            k,
            v,
            prefix_len=prefix_len,
            tokens_per_item=tokens_per_item,
            sm_scale=sm_scale,
            block_q=block_q,
            block_k=block_k,
            interpret=True,
        )
        ref_out = _reference_batched_attention(
            q,
            k,
            v,
            prefix_len=prefix_len,
            tokens_per_item=tokens_per_item,
            sm_scale=sm_scale,
        )
        np.testing.assert_allclose(
            np.array(pallas_out), np.array(ref_out), atol=atol, rtol=rtol
        )

        dq_p, dk_p, dv_p = jax.grad(pallas_loss, argnums=(0, 1, 2))(q, k, v)
        dq_r, dk_r, dv_r = jax.grad(ref_loss, argnums=(0, 1, 2))(q, k, v)

        np.testing.assert_allclose(np.array(dq_p), np.array(dq_r), atol=atol, rtol=rtol)
        np.testing.assert_allclose(np.array(dk_p), np.array(dk_r), atol=atol, rtol=rtol)
        np.testing.assert_allclose(np.array(dv_p), np.array(dv_r), atol=atol, rtol=rtol)

    def test_grad_basic(self) -> None:
        """Single batch, no GQA, simple target mask."""
        self._run_grad_test(
            batch_size=1,
            seq_len=64,
            num_heads=2,
            num_kv_heads=2,
            head_dim=32,
            prefix_len=16,
            tokens_per_item=4,
            block_q=32,
            block_k=32,
        )

    def test_grad_batched(self) -> None:
        """Batch=2, no GQA."""
        self._run_grad_test(
            batch_size=2,
            seq_len=64,
            num_heads=4,
            num_kv_heads=4,
            head_dim=32,
            prefix_len=16,
            tokens_per_item=4,
            block_q=32,
            block_k=32,
        )

    def test_grad_gqa(self) -> None:
        """GQA: num_heads=4, num_kv_heads=2 (groups=2)."""
        self._run_grad_test(
            batch_size=1,
            seq_len=64,
            num_heads=4,
            num_kv_heads=2,
            head_dim=32,
            prefix_len=16,
            tokens_per_item=4,
            block_q=32,
            block_k=32,
        )

    def test_grad_single_token_items(self) -> None:
        """Edge case: tokens_per_item=1."""
        self._run_grad_test(
            batch_size=1,
            seq_len=64,
            num_heads=2,
            num_kv_heads=2,
            head_dim=32,
            prefix_len=16,
            tokens_per_item=1,
            block_q=32,
            block_k=32,
        )

    def test_grad_padded_seqlen(self) -> None:
        """seq_len that is not a multiple of block_q triggers padding path."""
        self._run_grad_test(
            batch_size=1,
            seq_len=48,
            num_heads=2,
            num_kv_heads=2,
            head_dim=32,
            prefix_len=12,
            tokens_per_item=4,
            block_q=32,
            block_k=32,
        )
