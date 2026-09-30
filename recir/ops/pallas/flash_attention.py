# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Optimized Pallas FlashAttention kernel for TPU with target attention mask.

This module provides a tiled FlashAttention-2 implementation using JAX's Pallas
API targeting TPU.  It supports both varlen (FA3) and 4D (FA4) formats.  Both
forward (inference) and backward (training) passes are implemented as Pallas
kernels and wired through ``jax.custom_vjp``.

Key optimizations over the baseline implementation:
  P0: ``interpret`` defaults to False — runs on TPU hardware, not CPU-emulated.
  P1: Head dimension folded into the Pallas grid — one ``pallas_call`` per
      sequence instead of one per (batch, head) pair.
  P2: Causal KV loop upper bound — skips KV blocks that are entirely masked.
  P3: Batch dimension folded into the Pallas grid — one ``pallas_call`` for
      the entire batch, eliminating per-sequence dispatch overhead (~133ms each).
"""

from __future__ import annotations

import functools
import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl

# Large negative value used to mask out positions (not -inf, to avoid NaN in
# exp2 arithmetic on TPU).  Mirrors the tritonbench Pallas convention.
DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)

# TPU VMEM lane width.  Per-token scalars (LSE, delta) must be stored with this
# trailing dimension to satisfy Pallas BlockSpec alignment ((mult-of-8, 128)).
LANE_DIM = 128


# ---------------------------------------------------------------------------
# A. Target attention mask builder (vectorized, for use inside Pallas kernel)
# ---------------------------------------------------------------------------


def compute_target_attention_mask_block(
    q_indices: jax.Array,
    k_indices: jax.Array,
    seq_start: int | jax.Array,
    prefix_len: int | jax.Array,
    tokens_per_item: int | jax.Array,
) -> jax.Array:
    """Compute boolean mask [BLOCK_Q, BLOCK_K] for target attention pattern.

    Returns True where attention is allowed.
    """
    q_local = q_indices - seq_start  # [BLOCK_Q]
    k_local = k_indices - seq_start  # [BLOCK_K]

    q_local_2d = q_local[:, None]  # [BLOCK_Q, 1]
    k_local_2d = k_local[None, :]  # [1, BLOCK_K]

    q_in_prefix = q_local_2d < prefix_len
    k_in_prefix = k_local_2d < prefix_len
    causal = k_local_2d <= q_local_2d

    # Both in prefix: causal
    both_prefix = q_in_prefix & k_in_prefix & causal
    # Q in items, K in prefix: always attend
    q_item_k_prefix = (~q_in_prefix) & k_in_prefix
    # Both in items: same item + causal
    q_item_idx = (q_local_2d - prefix_len) // tokens_per_item
    k_item_idx = (k_local_2d - prefix_len) // tokens_per_item
    same_item = q_item_idx == k_item_idx
    both_items = (~q_in_prefix) & (~k_in_prefix) & same_item & causal

    return both_prefix | q_item_k_prefix | both_items


# ---------------------------------------------------------------------------
# B. Pallas FlashAttention forward kernel for TPU — multi-head grid
# ---------------------------------------------------------------------------


def _make_flash_attn_kernel(
    *,
    seq_len: int,
    head_dim: int,
    sm_scale: float,
    block_q: int,
    block_k: int,
    seq_start: int | jax.Array,
    prefix_len: int | jax.Array,
    tokens_per_item: int | jax.Array,
):
    """Return a Pallas kernel function closed over the compile-time constants.

    The kernel is invoked on a 2D grid: ``(num_q_blocks, num_heads)``.
    ``pl.program_id(0)`` selects the Q block, ``pl.program_id(1)`` selects
    the head.  GQA head mapping is handled by BlockSpec index functions
    so the kernel receives the correct KV head slice.

    Input refs use a transposed layout ``[1, seq_or_block, head_dim]`` where
    the leading dimension selects the head (via BlockSpec).  This satisfies
    TPU Pallas alignment constraints that require the last two block-shape
    dimensions to be divisible by (8, 128) or equal to the array dimensions.
    """
    qk_scale = sm_scale * math.log2(math.e)

    def flash_attn_kernel(q_ref, k_ref, v_ref, o_ref):
        start_q = pl.program_id(0)

        m_i = jnp.full(block_q, float("-inf"), dtype=jnp.float32)
        l_i = jnp.zeros(block_q, dtype=jnp.float32)
        o = jnp.zeros((block_q, head_dim), dtype=jnp.float32)

        # Q ref is [1, block_q, head_dim]; squeeze head selector dim.
        q = q_ref[0, :, :]  # [block_q, head_dim]

        # K/V refs are [1, seq_len, head_dim] for the selected KV head.

        # --- Hoist Q-side mask indices (invariant across KV loop) ---
        q_indices = start_q * block_q + jnp.arange(block_q)
        q_local = q_indices - seq_start
        q_local_2d = q_local[:, None]  # [block_q, 1]
        q_in_prefix = q_local_2d < prefix_len
        q_item_idx = (q_local_2d - prefix_len) // tokens_per_item

        # --- P2: Causal upper bound on KV loop ---
        q_end = (start_q + 1) * block_q  # exclusive end of Q positions
        causal_upper = pl.cdiv(q_end, block_k)
        upper_bound = jnp.minimum(
            pl.cdiv(seq_len, block_k),
            causal_upper,
        )

        def body(start_k, carry):
            o_prev, m_prev, l_prev = carry

            # K/V refs are [1, seq_len, head_dim]; squeeze head dim, slice seq.
            k = k_ref[0, pl.dslice(start_k * block_k, block_k), :]
            v = v_ref[0, pl.dslice(start_k * block_k, block_k), :]

            qk = pl.dot(q, k.T)  # [block_q, block_k]
            qk = qk * qk_scale

            # --- Apply target attention mask (Q-side hoisted) ---
            k_indices = start_k * block_k + jnp.arange(block_k)
            k_local = k_indices - seq_start
            k_local_2d = k_local[None, :]  # [1, block_k]

            k_in_prefix = k_local_2d < prefix_len
            causal = k_local_2d <= q_local_2d

            both_prefix = q_in_prefix & k_in_prefix & causal
            q_item_k_prefix = (~q_in_prefix) & k_in_prefix
            k_item_idx = (k_local_2d - prefix_len) // tokens_per_item
            same_item = q_item_idx == k_item_idx
            both_items = (~q_in_prefix) & (~k_in_prefix) & same_item & causal

            mask = both_prefix | q_item_k_prefix | both_items
            qk = jnp.where(mask, qk, DEFAULT_MASK_VALUE)

            # --- Online softmax update (base-2) ---
            m_curr = qk.max(axis=-1)
            m_next = jnp.maximum(m_prev, m_curr)
            correction = jnp.exp2(m_prev - m_next)
            l_prev_corr = correction * l_prev
            s_curr = jnp.exp2(qk - m_next[:, None])
            l_curr = s_curr.sum(axis=-1)
            l_next = l_prev_corr + l_curr
            o_prev_corr = correction[:, None] * o_prev

            o_curr = pl.dot(s_curr.astype(v.dtype), v)

            o_next = o_prev_corr + o_curr
            return o_next, m_next, l_next

        o, m_i, l_i = lax.fori_loop(0, upper_bound, body, (o, m_i, l_i))
        o = o / l_i[:, None]
        o_ref[0, :, :] = o.astype(o_ref.dtype)

    return flash_attn_kernel


def pallas_flash_attention_forward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    seq_start: int | jax.Array,
    prefix_len: int | jax.Array,
    tokens_per_item: int | jax.Array,
    sm_scale: float,
    block_q: int = 128,
    block_k: int = 128,
    interpret: bool = False,
) -> jax.Array:
    """Flash attention forward pass using Pallas on TPU.

    Processes all heads of a single sequence in one ``pallas_call``.

    Args:
        q: [seq_len, num_heads, head_dim]
        k: [seq_len, num_kv_heads, head_dim]
        v: [seq_len, num_kv_heads, head_dim]
        seq_start: scalar absolute offset of this sequence in the packed
            tensor (used by the mask builder).
        prefix_len: number of prefix tokens.
        tokens_per_item: tokens per item in the target section.
        sm_scale: softmax scale (typically 1/sqrt(head_dim)).
        block_q: tile size along Q sequence dimension.
        block_k: tile size along KV sequence dimension.
        interpret: if True, run on CPU (for testing). Default False (TPU).

    Returns:
        Output array of shape [seq_len, num_heads, head_dim], same dtype as *q*.
    """
    seq_len, num_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    groups = num_heads // num_kv_heads
    dtype = q.dtype

    # Pad seq_len to multiples of block sizes so that the Pallas grid and
    # BlockSpec dimensions divide evenly.
    pad_q = (block_q - seq_len % block_q) % block_q
    pad_k = (block_k - seq_len % block_k) % block_k
    pad_len = max(pad_q, pad_k)

    if pad_len > 0:
        q = jnp.pad(q, ((0, pad_len), (0, 0), (0, 0)))
        k = jnp.pad(k, ((0, pad_len), (0, 0), (0, 0)))
        v = jnp.pad(v, ((0, pad_len), (0, 0), (0, 0)))
        padded_seq_len = seq_len + pad_len
    else:
        padded_seq_len = seq_len

    # Transpose to [num_heads, seq_len, head_dim].  TPU Pallas requires the
    # last two block-shape dims to be divisible by (8, 128) or equal to array
    # dims.  Layout [H, S, D] with block (1, block_q, D) gives last-two =
    # (block_q, D) = (128, 64) which satisfies the constraint.
    q_t = jnp.transpose(q, (1, 0, 2))  # [num_heads, padded_seq_len, head_dim]
    k_t = jnp.transpose(k, (1, 0, 2))  # [num_kv_heads, padded_seq_len, head_dim]
    v_t = jnp.transpose(v, (1, 0, 2))

    kernel = _make_flash_attn_kernel(
        seq_len=padded_seq_len,
        head_dim=head_dim,
        sm_scale=sm_scale,
        block_q=block_q,
        block_k=block_k,
        seq_start=seq_start,
        prefix_len=prefix_len,
        tokens_per_item=tokens_per_item,
    )

    num_q_blocks = pl.cdiv(padded_seq_len, block_q)

    result = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((num_heads, padded_seq_len, head_dim), dtype),
        grid=(num_q_blocks, num_heads),
        in_specs=[
            # Q: [num_heads, padded_seq_len, head_dim] -> (1, block_q, head_dim)
            pl.BlockSpec((1, block_q, head_dim), lambda q_blk, h: (h, q_blk, 0)),
            # K: [num_kv_heads, padded_seq_len, head_dim] -> full seq for
            # the selected KV head via GQA mapping.
            pl.BlockSpec(
                (1, padded_seq_len, head_dim),
                lambda q_blk, h: (h // groups, 0, 0),
            ),
            # V: same as K.
            pl.BlockSpec(
                (1, padded_seq_len, head_dim),
                lambda q_blk, h: (h // groups, 0, 0),
            ),
        ],
        out_specs=pl.BlockSpec((1, block_q, head_dim), lambda q_blk, h: (h, q_blk, 0)),
        interpret=interpret,
    )(q_t, k_t, v_t)

    # Transpose back to [seq_len, num_heads, head_dim]
    result = jnp.transpose(result, (1, 0, 2))

    # Remove padding
    if pad_len > 0:
        result = result[:seq_len]

    return result


# ---------------------------------------------------------------------------
# B3. Backward kernels — dKV and dQ for the batched path
# ---------------------------------------------------------------------------
#
# FlashAttention-2 backward decomposes into two kernels because the dQ and dKV
# accumulations have *opposite* contracting axes:
#   - dV[j] = sum_i p[i, j] * dO[i]          → loop over Q outer, accumulate
#                                              into dV[j] (one KV block).
#   - dK[j] = sum_i ds[i, j] * Q[i]          → same axis as dV.
#   - dQ[i] = sum_j ds[i, j] * K[j]          → loop over KV outer, accumulate
#                                              into dQ[i] (one Q block).
#
# Both kernels recompute the attention probabilities from saved LSE rather
# than the [seq_len, seq_len] probability matrix, which is the whole point of
# FlashAttention.  The probabilities used here are the *natural* softmax
# probabilities — but recovered from the base-2 forward via
# ``p = exp2(qk_scaled - lse_b2)``, which is mathematically equivalent.
#
# dS (cotangent w.r.t. natural pre-softmax logits):
#     ds_natural = p * (dp - delta), where delta_i = (dO * O).sum(-1)
# Chain rule through s = sm_scale * (Q · K^T):
#     dQ = sm_scale * (dS @ K),     dK = sm_scale * (dS^T @ Q)
#
# GQA handling: the kernels here produce per-Q-head dK / dV
# (shape ``[B, num_heads, S, D]``).  After the call we sum across the head
# group to obtain the final ``[B, num_kv_heads, S, D]``.  This trades a small
# amount of HBM (groups × num_kv_heads × S × D fp32) for kernel simplicity.


def _make_flash_attn_dkv_kernel_batched(
    *,
    seq_len: int,
    head_dim: int,
    sm_scale: float,
    block_q: int,
    block_k: int,
    prefix_len: int,
    tokens_per_item: int,
    groups: int,
):
    """Return the dKV backward kernel with GQA-aware accumulation.

    Grid: ``(num_kv_blocks, num_kv_heads, batch_size)``.  For each KV block
    the kernel iterates over all ``groups`` Q heads in the GQA group and all
    Q blocks, accumulating dK and dV in-kernel at KV-head granularity.

    Refs:
        q, do: ``[1, groups, padded_seq_len, head_dim]``
        k, v: ``[1, 1, block_k, head_dim]``
        lse, delta: ``[1, groups, padded_seq_len, LANE_DIM]``
        dk, dv: ``[1, 1, block_k, head_dim]`` outputs (per KV head)
    """
    qk_scale = sm_scale * math.log2(math.e)

    def dkv_kernel(q_ref, k_ref, v_ref, do_ref, lse_ref, delta_ref, dk_ref, dv_ref):
        start_k = pl.program_id(0)

        k = k_ref[0, 0, :, :]  # [block_k, head_dim]
        v = v_ref[0, 0, :, :]  # [block_k, head_dim]

        # --- Hoist K-side mask indices ---
        k_indices = start_k * block_k + jnp.arange(block_k)
        k_local_2d = k_indices[None, :]  # [1, block_k]
        k_in_prefix = k_local_2d < prefix_len
        k_item_idx = (k_local_2d - prefix_len) // tokens_per_item

        # Causal lower bound: smallest Q block that can attend to this K block.
        q_start_block = (start_k * block_k) // block_q
        num_q_blocks = pl.cdiv(seq_len, block_q)

        dk_acc = jnp.zeros((block_k, head_dim), dtype=jnp.float32)
        dv_acc = jnp.zeros((block_k, head_dim), dtype=jnp.float32)

        # Iterate over all Q heads in the GQA group.
        for g in range(groups):

            def body(start_q, carry, _g=g):
                dk_prev, dv_prev = carry

                q_slice = pl.dslice(start_q * block_q, block_q)
                q = q_ref[0, _g, q_slice, :]  # [block_q, head_dim]
                do = do_ref[0, _g, q_slice, :]  # [block_q, head_dim]
                lse_full = lse_ref[0, _g, q_slice, :]  # [block_q, LANE_DIM]
                delta_full = delta_ref[0, _g, q_slice, :]  # [block_q, LANE_DIM]
                lse = lse_full[:, 0]  # [block_q]
                delta = delta_full[:, 0]  # [block_q]

                # Recompute logits in base-2 domain.
                qk = pl.dot(q, k.T) * qk_scale  # [block_q, block_k] f32

                # --- Apply target attention mask ---
                q_indices = start_q * block_q + jnp.arange(block_q)
                q_local_2d = q_indices[:, None]  # [block_q, 1]
                q_in_prefix = q_local_2d < prefix_len
                q_item_idx = (q_local_2d - prefix_len) // tokens_per_item
                causal = k_local_2d <= q_local_2d

                both_prefix = q_in_prefix & k_in_prefix & causal
                q_item_k_prefix = (~q_in_prefix) & k_in_prefix
                same_item = q_item_idx == k_item_idx
                both_items = (~q_in_prefix) & (~k_in_prefix) & same_item & causal
                mask = both_prefix | q_item_k_prefix | both_items
                qk = jnp.where(mask, qk, DEFAULT_MASK_VALUE)

                # Recompute attention probabilities (natural softmax via base-2).
                p = jnp.exp2(qk - lse[:, None])  # [block_q, block_k]

                # dV += p^T @ dO     (accumulate fp32)
                dv_curr = pl.dot(p.astype(do.dtype).T, do)  # [block_k, head_dim]

                # dP = dO @ V^T,  dS = (dP - delta) * p,  rescale by sm_scale.
                dp = pl.dot(do, v.T)  # [block_q, block_k]
                ds = (dp - delta[:, None]) * p
                ds = ds * sm_scale

                # dK += dS^T @ Q
                dk_curr = pl.dot(ds.astype(q.dtype).T, q)  # [block_k, head_dim]

                return dk_prev + dk_curr, dv_prev + dv_curr

            dk_acc, dv_acc = lax.fori_loop(
                q_start_block, num_q_blocks, body, (dk_acc, dv_acc)
            )

        dk_ref[0, 0, :, :] = dk_acc.astype(dk_ref.dtype)
        dv_ref[0, 0, :, :] = dv_acc.astype(dv_ref.dtype)

    return dkv_kernel


def _make_flash_attn_dq_kernel_batched(
    *,
    seq_len: int,
    head_dim: int,
    sm_scale: float,
    block_q: int,
    block_k: int,
    prefix_len: int,
    tokens_per_item: int,
):
    """Return the dQ backward kernel for the batched path.

    Grid: ``(num_q_blocks, num_heads, batch_size)``.  For each Q block the
    kernel iterates over KV blocks (with causal upper bound, mirroring the
    forward) and accumulates one ``[block_q, head_dim]`` slab of dQ.

    Refs:
        q, do: ``[1, 1, block_q, head_dim]`` (selected Q block)
        k, v: ``[1, 1, padded_seq_len, head_dim]`` (full KV sequence per head)
        lse, delta: ``[1, 1, block_q, LANE_DIM]`` (broadcast lane)
        dq: ``[1, 1, block_q, head_dim]`` output
    """
    qk_scale = sm_scale * math.log2(math.e)

    def dq_kernel(q_ref, k_ref, v_ref, do_ref, lse_ref, delta_ref, dq_ref):
        start_q = pl.program_id(0)

        q = q_ref[0, 0, :, :]  # [block_q, head_dim]
        do = do_ref[0, 0, :, :]  # [block_q, head_dim]
        lse_full = lse_ref[0, 0, :, :]  # [block_q, LANE_DIM]
        delta_full = delta_ref[0, 0, :, :]  # [block_q, LANE_DIM]
        lse = lse_full[:, 0]  # [block_q]
        delta = delta_full[:, 0]  # [block_q]

        # --- Hoist Q-side mask indices ---
        q_indices = start_q * block_q + jnp.arange(block_q)
        q_local_2d = q_indices[:, None]  # [block_q, 1]
        q_in_prefix = q_local_2d < prefix_len
        q_item_idx = (q_local_2d - prefix_len) // tokens_per_item

        # Causal upper bound on KV loop (same as forward).
        q_end = (start_q + 1) * block_q
        causal_upper = pl.cdiv(q_end, block_k)
        upper_bound = jnp.minimum(pl.cdiv(seq_len, block_k), causal_upper)

        dq_acc = jnp.zeros((block_q, head_dim), dtype=jnp.float32)

        def body(start_k, dq_prev):
            k_slice = pl.dslice(start_k * block_k, block_k)
            k = k_ref[0, 0, k_slice, :]
            v = v_ref[0, 0, k_slice, :]

            qk = pl.dot(q, k.T) * qk_scale  # [block_q, block_k]

            k_indices = start_k * block_k + jnp.arange(block_k)
            k_local_2d = k_indices[None, :]
            k_in_prefix = k_local_2d < prefix_len
            k_item_idx = (k_local_2d - prefix_len) // tokens_per_item
            causal = k_local_2d <= q_local_2d

            both_prefix = q_in_prefix & k_in_prefix & causal
            q_item_k_prefix = (~q_in_prefix) & k_in_prefix
            same_item = q_item_idx == k_item_idx
            both_items = (~q_in_prefix) & (~k_in_prefix) & same_item & causal
            mask = both_prefix | q_item_k_prefix | both_items
            qk = jnp.where(mask, qk, DEFAULT_MASK_VALUE)

            p = jnp.exp2(qk - lse[:, None])  # [block_q, block_k]

            dp = pl.dot(do, v.T)  # [block_q, block_k]
            ds = (dp - delta[:, None]) * p
            ds = ds * sm_scale

            # dQ += dS @ K
            dq_curr = pl.dot(ds.astype(k.dtype), k)  # [block_q, head_dim]
            return dq_prev + dq_curr

        dq_acc = lax.fori_loop(0, upper_bound, body, dq_acc)
        dq_ref[0, 0, :, :] = dq_acc.astype(dq_ref.dtype)

    return dq_kernel


def _pallas_flash_attention_dkv_batched(
    q_t: jax.Array,
    k_t: jax.Array,
    v_t: jax.Array,
    do_t: jax.Array,
    lse_b: jax.Array,
    delta_b: jax.Array,
    *,
    prefix_len: int,
    tokens_per_item: int,
    sm_scale: float,
    block_q: int,
    block_k: int,
    interpret: bool,
):
    """Run the GQA-aware dKV Pallas kernel; returns dK, dV at KV-head granularity.

    Args:
        q_t, do_t: [B, H, padded_S, D]
        k_t, v_t: [B, Hkv, padded_S, D]
        lse_b, delta_b: [B, H, padded_S, LANE_DIM] f32
    Returns:
        (dk_t, dv_t): each [B, Hkv, padded_S, D] f32
        (already reduced over GQA groups — no external reduction needed).
    """
    batch_size, num_heads, padded_seq_len, head_dim = q_t.shape
    num_kv_heads = k_t.shape[1]
    groups = num_heads // num_kv_heads

    kernel = _make_flash_attn_dkv_kernel_batched(
        seq_len=padded_seq_len,
        head_dim=head_dim,
        sm_scale=sm_scale,
        block_q=block_q,
        block_k=block_k,
        prefix_len=prefix_len,
        tokens_per_item=tokens_per_item,
        groups=groups,
    )

    num_kv_blocks = pl.cdiv(padded_seq_len, block_k)

    out_shape = [
        # dK per KV head
        jax.ShapeDtypeStruct(
            (batch_size, num_kv_heads, padded_seq_len, head_dim), jnp.float32
        ),
        # dV per KV head
        jax.ShapeDtypeStruct(
            (batch_size, num_kv_heads, padded_seq_len, head_dim), jnp.float32
        ),
    ]

    in_specs = [
        # Q: all groups of Q heads for the selected KV head
        pl.BlockSpec(
            (1, groups, padded_seq_len, head_dim),
            lambda kv_blk, kv_h, b: (b, kv_h * groups, 0, 0),
        ),
        # K: block for selected KV head
        pl.BlockSpec(
            (1, 1, block_k, head_dim),
            lambda kv_blk, kv_h, b: (b, kv_h, kv_blk, 0),
        ),
        # V: same as K
        pl.BlockSpec(
            (1, 1, block_k, head_dim),
            lambda kv_blk, kv_h, b: (b, kv_h, kv_blk, 0),
        ),
        # dO: all groups for the selected KV head
        pl.BlockSpec(
            (1, groups, padded_seq_len, head_dim),
            lambda kv_blk, kv_h, b: (b, kv_h * groups, 0, 0),
        ),
        # LSE: all groups
        pl.BlockSpec(
            (1, groups, padded_seq_len, LANE_DIM),
            lambda kv_blk, kv_h, b: (b, kv_h * groups, 0, 0),
        ),
        # delta: all groups
        pl.BlockSpec(
            (1, groups, padded_seq_len, LANE_DIM),
            lambda kv_blk, kv_h, b: (b, kv_h * groups, 0, 0),
        ),
    ]

    out_specs = [
        # dK per KV head
        pl.BlockSpec(
            (1, 1, block_k, head_dim),
            lambda kv_blk, kv_h, b: (b, kv_h, kv_blk, 0),
        ),
        # dV same
        pl.BlockSpec(
            (1, 1, block_k, head_dim),
            lambda kv_blk, kv_h, b: (b, kv_h, kv_blk, 0),
        ),
    ]

    dk_t, dv_t = pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid=(num_kv_blocks, num_kv_heads, batch_size),
        in_specs=in_specs,
        out_specs=out_specs,
        interpret=interpret,
    )(q_t, k_t, v_t, do_t, lse_b, delta_b)

    return dk_t, dv_t


def _pallas_flash_attention_dq_batched(
    q_t: jax.Array,
    k_t: jax.Array,
    v_t: jax.Array,
    do_t: jax.Array,
    lse_b: jax.Array,
    delta_b: jax.Array,
    *,
    prefix_len: int,
    tokens_per_item: int,
    sm_scale: float,
    block_q: int,
    block_k: int,
    interpret: bool,
):
    """Run the dQ Pallas kernel; returns dQ in transposed layout.

    Returns:
        dq_t: [B, H, padded_S, D] f32
    """
    batch_size, num_heads, padded_seq_len, head_dim = q_t.shape
    num_kv_heads = k_t.shape[1]
    groups = num_heads // num_kv_heads

    kernel = _make_flash_attn_dq_kernel_batched(
        seq_len=padded_seq_len,
        head_dim=head_dim,
        sm_scale=sm_scale,
        block_q=block_q,
        block_k=block_k,
        prefix_len=prefix_len,
        tokens_per_item=tokens_per_item,
    )

    num_q_blocks = pl.cdiv(padded_seq_len, block_q)

    out_shape = jax.ShapeDtypeStruct(
        (batch_size, num_heads, padded_seq_len, head_dim), jnp.float32
    )

    in_specs = [
        # Q block
        pl.BlockSpec(
            (1, 1, block_q, head_dim),
            lambda q_blk, h, b: (b, h, q_blk, 0),
        ),
        # K full seq for selected KV head
        pl.BlockSpec(
            (1, 1, padded_seq_len, head_dim),
            lambda q_blk, h, b: (b, h // groups, 0, 0),
        ),
        # V same
        pl.BlockSpec(
            (1, 1, padded_seq_len, head_dim),
            lambda q_blk, h, b: (b, h // groups, 0, 0),
        ),
        # dO block
        pl.BlockSpec(
            (1, 1, block_q, head_dim),
            lambda q_blk, h, b: (b, h, q_blk, 0),
        ),
        # LSE block
        pl.BlockSpec(
            (1, 1, block_q, LANE_DIM),
            lambda q_blk, h, b: (b, h, q_blk, 0),
        ),
        # delta block
        pl.BlockSpec(
            (1, 1, block_q, LANE_DIM),
            lambda q_blk, h, b: (b, h, q_blk, 0),
        ),
    ]

    out_specs = pl.BlockSpec(
        (1, 1, block_q, head_dim),
        lambda q_blk, h, b: (b, h, q_blk, 0),
    )

    dq_t = pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid=(num_q_blocks, num_heads, batch_size),
        in_specs=in_specs,
        out_specs=out_specs,
        interpret=interpret,
    )(q_t, k_t, v_t, do_t, lse_b, delta_b)

    return dq_t


# ---------------------------------------------------------------------------
# B4. custom_vjp wiring for the batched path
# ---------------------------------------------------------------------------


def _batched_fwd(
    q,
    k,
    v,
    prefix_len,
    tokens_per_item,
    sm_scale,
    block_q,
    block_k,
    interpret,
    block_q_bwd,
    block_k_bwd,
):
    """Forward pass for ``custom_vjp``.  Saves residuals for backward."""
    out, lse, padded_seq_len = _pallas_flash_attention_batched_impl(
        q,
        k,
        v,
        prefix_len=prefix_len,
        tokens_per_item=tokens_per_item,
        sm_scale=sm_scale,
        block_q=block_q,
        block_k=block_k,
        interpret=interpret,
        save_lse=True,
    )
    # Residuals: keep raw (unpadded, untransposed) inputs + lse + padded_seq_len.
    residuals = (q, k, v, out, lse, padded_seq_len)
    return out, residuals


def _batched_bwd(
    prefix_len,
    tokens_per_item,
    sm_scale,
    block_q,
    block_k,
    interpret,
    block_q_bwd,
    block_k_bwd,
    residuals,
    do,
):
    """Backward pass for ``custom_vjp``.

    Args:
        residuals: (q, k, v, out, lse, padded_seq_len) from ``_batched_fwd``.
        do: cotangent of output, shape ``[B, S, H, D]``.
    """
    q, k, v, out, lse, padded_seq_len = residuals
    seq_len = q.shape[1]

    # Resolve backward block sizes.
    bq = block_q_bwd
    bk = block_k_bwd

    # Backward-specific padding (may differ from forward's padded_seq_len).
    bwd_pad_q = (bq - seq_len % bq) % bq
    bwd_pad_k = (bk - seq_len % bk) % bk
    pad_len = max(bwd_pad_q, bwd_pad_k)
    # Ensure forward's lse (at padded_seq_len) fits without trimming.
    pad_len = max(pad_len, padded_seq_len - seq_len)

    # Pad q/k/v/out/do up to padded_seq_len, then transpose to [B, H, S, D].
    if pad_len > 0:
        q_p = jnp.pad(q, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
        k_p = jnp.pad(k, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
        v_p = jnp.pad(v, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
        out_p = jnp.pad(out, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
        do_p = jnp.pad(do, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
    else:
        q_p, k_p, v_p, out_p, do_p = q, k, v, out, do

    q_t = jnp.transpose(q_p, (0, 2, 1, 3))  # [B, H, S, D]
    k_t = jnp.transpose(k_p, (0, 2, 1, 3))  # [B, Hkv, S, D]
    v_t = jnp.transpose(v_p, (0, 2, 1, 3))
    out_t = jnp.transpose(out_p, (0, 2, 1, 3))  # [B, H, S, D]
    do_t = jnp.transpose(do_p, (0, 2, 1, 3))  # [B, H, S, D]

    # Compute delta = sum(O * dO) per Q position in fp32.
    # Shape: [B, H, S]
    delta = jnp.sum(out_t.astype(jnp.float32) * do_t.astype(jnp.float32), axis=-1)

    # lse already padded ([B, H, padded_S]); pad further if backward needs more.
    lse_extra = (seq_len + pad_len) - padded_seq_len
    if lse_extra > 0:
        lse = jnp.pad(lse, ((0, 0), (0, 0), (0, lse_extra)))
    lse_b = jnp.broadcast_to(lse[..., None], (*lse.shape, LANE_DIM))
    delta_b = jnp.broadcast_to(delta[..., None], (*delta.shape, LANE_DIM))

    # dKV kernel: GQA-aware — outputs dK, dV at KV-head granularity.
    dk_t, dv_t = _pallas_flash_attention_dkv_batched(
        q_t,
        k_t,
        v_t,
        do_t,
        lse_b,
        delta_b,
        prefix_len=prefix_len,
        tokens_per_item=tokens_per_item,
        sm_scale=sm_scale,
        block_q=bq,
        block_k=bk,
        interpret=interpret,
    )

    # dQ kernel: unchanged, operates per Q head.
    dq_t = _pallas_flash_attention_dq_batched(
        q_t,
        k_t,
        v_t,
        do_t,
        lse_b,
        delta_b,
        prefix_len=prefix_len,
        tokens_per_item=tokens_per_item,
        sm_scale=sm_scale,
        block_q=bq,
        block_k=bk,
        interpret=interpret,
    )

    # Back to [B, S, H, D] and original (q's) dtype, drop padding.
    dq = jnp.transpose(dq_t, (0, 2, 1, 3)).astype(q.dtype)
    dk = jnp.transpose(dk_t, (0, 2, 1, 3)).astype(k.dtype)
    dv = jnp.transpose(dv_t, (0, 2, 1, 3)).astype(v.dtype)
    if pad_len > 0:
        dq = dq[:, :seq_len, :, :]
        dk = dk[:, :seq_len, :, :]
        dv = dv[:, :seq_len, :, :]

    return dq, dk, dv


# ---------------------------------------------------------------------------
# C. High-level wrapper for FA3 (varlen format)
# ---------------------------------------------------------------------------


def _make_flash_attn_kernel_batched(
    *,
    seq_len: int,
    head_dim: int,
    sm_scale: float,
    block_q: int,
    block_k: int,
    prefix_len: int,
    tokens_per_item: int,
    save_lse: bool,
):
    """Return a batched Pallas kernel closed over compile-time constants.

    The kernel is invoked on a 3D grid: ``(num_q_blocks, num_heads, batch_size)``.
    All sequences in the batch must share the same ``seq_len``, ``prefix_len``,
    and ``tokens_per_item`` (the uniform-batch fast path).

    Input refs use layout ``[1, 1, seq_or_block, head_dim]`` where the first
    two dims select batch element and head via BlockSpec.  Last two dims
    ``(block_q, head_dim)`` satisfy TPU alignment constraints.

    When ``save_lse`` is True, the kernel also writes the per-Q-position
    log-sum-exp (in base-2: ``lse = m_i + log2(l_i)``) to a fourth output ref
    of shape ``[1, 1, block_q, LANE_DIM]`` (broadcast across the lane axis).
    """
    qk_scale = sm_scale * math.log2(math.e)

    def flash_attn_kernel(q_ref, k_ref, v_ref, *out_refs):
        if save_lse:
            o_ref, lse_ref = out_refs
        else:
            (o_ref,) = out_refs
            lse_ref = None
        start_q = pl.program_id(0)

        m_i = jnp.full(block_q, float("-inf"), dtype=jnp.float32)
        l_i = jnp.zeros(block_q, dtype=jnp.float32)
        o = jnp.zeros((block_q, head_dim), dtype=jnp.float32)

        # Q ref is [1, 1, block_q, head_dim]; squeeze batch + head dims.
        q = q_ref[0, 0, :, :]  # [block_q, head_dim]

        # --- Hoist Q-side mask indices (seq_start is 0 for batched) ---
        q_indices = start_q * block_q + jnp.arange(block_q)
        q_local_2d = q_indices[:, None]  # [block_q, 1]
        q_in_prefix = q_local_2d < prefix_len
        q_item_idx = (q_local_2d - prefix_len) // tokens_per_item

        # --- P2: Causal upper bound on KV loop ---
        q_end = (start_q + 1) * block_q
        causal_upper = pl.cdiv(q_end, block_k)
        upper_bound = jnp.minimum(
            pl.cdiv(seq_len, block_k),
            causal_upper,
        )

        def body(start_k, carry):
            o_prev, m_prev, l_prev = carry

            k = k_ref[0, 0, pl.dslice(start_k * block_k, block_k), :]
            v = v_ref[0, 0, pl.dslice(start_k * block_k, block_k), :]

            qk = pl.dot(q, k.T)  # [block_q, block_k]
            qk = qk * qk_scale

            # --- Apply target attention mask ---
            k_indices = start_k * block_k + jnp.arange(block_k)
            k_local_2d = k_indices[None, :]  # [1, block_k]

            k_in_prefix = k_local_2d < prefix_len
            causal = k_local_2d <= q_local_2d

            both_prefix = q_in_prefix & k_in_prefix & causal
            q_item_k_prefix = (~q_in_prefix) & k_in_prefix
            k_item_idx = (k_local_2d - prefix_len) // tokens_per_item
            same_item = q_item_idx == k_item_idx
            both_items = (~q_in_prefix) & (~k_in_prefix) & same_item & causal

            mask = both_prefix | q_item_k_prefix | both_items
            qk = jnp.where(mask, qk, DEFAULT_MASK_VALUE)

            # --- Online softmax update (base-2) ---
            m_curr = qk.max(axis=-1)
            m_next = jnp.maximum(m_prev, m_curr)
            correction = jnp.exp2(m_prev - m_next)
            l_prev_corr = correction * l_prev
            s_curr = jnp.exp2(qk - m_next[:, None])
            l_curr = s_curr.sum(axis=-1)
            l_next = l_prev_corr + l_curr
            o_prev_corr = correction[:, None] * o_prev

            o_curr = pl.dot(s_curr.astype(v.dtype), v)

            o_next = o_prev_corr + o_curr
            return o_next, m_next, l_next

        o, m_i, l_i = lax.fori_loop(0, upper_bound, body, (o, m_i, l_i))
        o = o / l_i[:, None]
        o_ref[0, 0, :, :] = o.astype(o_ref.dtype)

        if save_lse:
            # LSE in base-2 domain: lse = m_i + log2(l_i).  Probabilities are
            # recovered in the backward as ``p = exp2(qk_scaled - lse)``.
            lse_value = m_i + jnp.log2(l_i)  # [block_q] f32
            lse_ref[0, 0, :, :] = jnp.broadcast_to(
                lse_value[:, None], (block_q, LANE_DIM)
            )

    return flash_attn_kernel


def _pallas_flash_attention_batched_impl(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    prefix_len: int,
    tokens_per_item: int,
    sm_scale: float,
    block_q: int,
    block_k: int,
    interpret: bool,
    save_lse: bool,
):
    """Internal: runs the batched Pallas forward kernel.

    Args:
        q: [batch_size, seq_len, num_heads, head_dim]
        k: [batch_size, seq_len, num_kv_heads, head_dim]
        v: [batch_size, seq_len, num_kv_heads, head_dim]
        save_lse: if True, return ``(out, lse, padded_seq_len)`` where ``lse``
            has shape ``[batch_size, num_heads, padded_seq_len]`` (base-2).
            If False, return only ``out``.

    Returns:
        Output of shape [batch_size, seq_len, num_heads, head_dim] (and
        optionally the padded LSE and padded seq len).
    """
    batch_size, seq_len, num_heads, head_dim = q.shape
    num_kv_heads = k.shape[2]
    groups = num_heads // num_kv_heads
    dtype = q.dtype

    # Pad seq_len to multiples of block sizes.
    pad_q = (block_q - seq_len % block_q) % block_q
    pad_k = (block_k - seq_len % block_k) % block_k
    pad_len = max(pad_q, pad_k)

    if pad_len > 0:
        q = jnp.pad(q, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
        k = jnp.pad(k, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
        v = jnp.pad(v, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
        padded_seq_len = seq_len + pad_len
    else:
        padded_seq_len = seq_len

    # Transpose to [B, H, S, D].  Last two block dims (block_q, D) satisfy
    # TPU alignment: block_q % 8 == 0 and D == array dim.
    q_t = jnp.transpose(q, (0, 2, 1, 3))  # [B, H, S, D]
    k_t = jnp.transpose(k, (0, 2, 1, 3))  # [B, Hkv, S, D]
    v_t = jnp.transpose(v, (0, 2, 1, 3))

    kernel = _make_flash_attn_kernel_batched(
        seq_len=padded_seq_len,
        head_dim=head_dim,
        sm_scale=sm_scale,
        block_q=block_q,
        block_k=block_k,
        prefix_len=prefix_len,
        tokens_per_item=tokens_per_item,
        save_lse=save_lse,
    )

    num_q_blocks = pl.cdiv(padded_seq_len, block_q)

    out_shapes: list[Any] = [
        jax.ShapeDtypeStruct((batch_size, num_heads, padded_seq_len, head_dim), dtype)
    ]
    out_specs: list[Any] = [
        pl.BlockSpec((1, 1, block_q, head_dim), lambda q_blk, h, b: (b, h, q_blk, 0))
    ]
    if save_lse:
        out_shapes.append(
            jax.ShapeDtypeStruct(
                (batch_size, num_heads, padded_seq_len, LANE_DIM), jnp.float32
            )
        )
        out_specs.append(
            pl.BlockSpec(
                (1, 1, block_q, LANE_DIM),
                lambda q_blk, h, b: (b, h, q_blk, 0),
            )
        )

    in_specs = [
        # Q: [B, H, S, D] -> (1, 1, block_q, D)
        pl.BlockSpec((1, 1, block_q, head_dim), lambda q_blk, h, b: (b, h, q_blk, 0)),
        # K: [B, Hkv, S, D] -> (1, 1, S, D) for selected KV head
        pl.BlockSpec(
            (1, 1, padded_seq_len, head_dim),
            lambda q_blk, h, b: (b, h // groups, 0, 0),
        ),
        # V: same as K.
        pl.BlockSpec(
            (1, 1, padded_seq_len, head_dim),
            lambda q_blk, h, b: (b, h // groups, 0, 0),
        ),
    ]

    # ``pallas_call`` returns a list iff ``out_shape`` is a list, even with a
    # single element.  Pass scalar/list consistently to control return type.
    if save_lse:
        result_t, lse_4d = pl.pallas_call(
            kernel,
            out_shape=out_shapes,
            grid=(num_q_blocks, num_heads, batch_size),
            in_specs=in_specs,
            out_specs=out_specs,
            interpret=interpret,
        )(q_t, k_t, v_t)
        # lse_4d: [B, H, padded_S, LANE_DIM] — broadcast across lane dim.
        lse = lse_4d[..., 0]  # [B, H, padded_S]
    else:
        result_t = pl.pallas_call(
            kernel,
            out_shape=out_shapes[0],
            grid=(num_q_blocks, num_heads, batch_size),
            in_specs=in_specs,
            out_specs=out_specs[0],
            interpret=interpret,
        )(q_t, k_t, v_t)

    # Transpose back to [B, S, H, D]
    result = jnp.transpose(result_t, (0, 2, 1, 3))

    # Remove padding from output (LSE is kept padded for backward use).
    if pad_len > 0:
        result = result[:, :seq_len, :, :]

    if save_lse:
        return result, lse, padded_seq_len  # pyre-ignore[61]
    return result


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6, 7, 8, 9, 10))
def pallas_flash_attention_batched(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    prefix_len: int,
    tokens_per_item: int,
    sm_scale: float,
    block_q: int = 128,
    block_k: int = 128,
    interpret: bool = False,
    block_q_bwd: int = 256,
    block_k_bwd: int = 256,
) -> jax.Array:
    """Batched flash attention forward pass using a single ``pallas_call``.

    All sequences must have the same length, prefix_len, and tokens_per_item.
    Differentiable w.r.t. ``q``, ``k``, and ``v`` via ``jax.custom_vjp``.

    Args:
        q: [batch_size, seq_len, num_heads, head_dim]
        k: [batch_size, seq_len, num_kv_heads, head_dim]
        v: [batch_size, seq_len, num_kv_heads, head_dim]
        prefix_len: number of prefix tokens (uniform across batch).
        tokens_per_item: tokens per item (uniform across batch).
        sm_scale: softmax scale (typically 1/sqrt(head_dim)).
        block_q: tile size along Q sequence dimension (forward).
        block_k: tile size along KV sequence dimension (forward).
        interpret: if True, run on CPU (for testing). Default False (TPU).
        block_q_bwd: Q tile size for backward kernels (default 256).
        block_k_bwd: KV tile size for backward kernels (default 256).

    Returns:
        Output of shape [batch_size, seq_len, num_heads, head_dim].
    """
    # pyrefly: ignore [bad-return]
    return _pallas_flash_attention_batched_impl(
        q,
        k,
        v,
        prefix_len=prefix_len,
        tokens_per_item=tokens_per_item,
        sm_scale=sm_scale,
        block_q=block_q,
        block_k=block_k,
        interpret=interpret,
        save_lse=False,
    )


pallas_flash_attention_batched.defvjp(_batched_fwd, _batched_bwd)


# ---------------------------------------------------------------------------
# C. High-level wrapper for FA3 (varlen format)
# ---------------------------------------------------------------------------


def pallas_flash_attention_varlen(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    cu_seqlens: jax.Array,
    prefix_lens: jax.Array,
    tokens_per_item: jax.Array | int,
    sm_scale: float,
    block_q: int = 128,
    block_k: int = 128,
    interpret: bool = False,
) -> jax.Array:
    """Flash attention with target attention mask for varlen packed sequences.

    When all sequences have uniform length, prefix_len, and tokens_per_item,
    uses a single batched ``pallas_call`` (P3 optimization).  Otherwise falls
    back to one ``pallas_call`` per sequence.

    Args:
        q: [total_tokens, num_heads, head_dim]
        k: [total_tokens, num_kv_heads, head_dim]
        v: [total_tokens, num_kv_heads, head_dim]
        cu_seqlens: [batch_size + 1] cumulative sequence lengths (int32).
        prefix_lens: [batch_size] prefix length per sequence.
        tokens_per_item: [batch_size] or scalar tokens per item.
        sm_scale: softmax scale.
        block_q: Q block size for the Pallas kernel.
        block_k: K block size for the Pallas kernel.
        interpret: if True, run on CPU (for testing). Default False (TPU).

    Returns:
        Output array of shape [total_tokens, num_heads, head_dim].
    """
    total_tokens, num_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]

    # Convert to numpy for indexing in the host loop
    cu_seqlens_np = np.asarray(cu_seqlens)
    prefix_lens_np = np.asarray(prefix_lens)
    if isinstance(tokens_per_item, (int, float)):
        tokens_per_item_np: Any = int(tokens_per_item)
    else:
        tokens_per_item_np = np.asarray(tokens_per_item)

    batch_size = cu_seqlens_np.shape[0] - 1

    # --- P3: Batched fast path when all sequences are uniform ---
    seq_lens = np.diff(cu_seqlens_np)
    uniform_seqlen = len(set(seq_lens)) == 1
    uniform_prefix = len(set(prefix_lens_np)) == 1
    if isinstance(tokens_per_item_np, int):
        uniform_tpi = True
        tpi_val = tokens_per_item_np
    else:
        uniform_tpi = len(set(tokens_per_item_np)) == 1
        tpi_val = int(tokens_per_item_np[0])

    if uniform_seqlen and uniform_prefix and uniform_tpi:
        seq_len = int(seq_lens[0])
        plen = int(prefix_lens_np[0])

        # Reshape packed [total_tokens, H, D] -> [B, seq_len, H, D]
        q_batched = q.reshape(batch_size, seq_len, num_heads, head_dim)
        k_batched = k.reshape(batch_size, seq_len, num_kv_heads, head_dim)
        v_batched = v.reshape(batch_size, seq_len, num_kv_heads, head_dim)

        result = pallas_flash_attention_batched(  # pyre-ignore[29]
            q_batched,
            k_batched,
            v_batched,
            prefix_len=plen,
            tokens_per_item=tpi_val,
            sm_scale=sm_scale,
            block_q=block_q,
            block_k=block_k,
            interpret=interpret,
        )

        # Reshape back [B, seq_len, H, D] -> [total_tokens, H, D]
        return result.reshape(total_tokens, num_heads, head_dim)

    # --- Fallback: one pallas_call per sequence ---
    seq_outputs = []
    for b in range(batch_size):
        start = int(cu_seqlens_np[b])
        end = int(cu_seqlens_np[b + 1])
        plen = int(prefix_lens_np[b])
        if isinstance(tokens_per_item_np, int):
            tpi = tokens_per_item_np
        else:
            tpi = int(tokens_per_item_np[b])

        q_seq = q[start:end, :, :]
        k_seq = k[start:end, :, :]
        v_seq = v[start:end, :, :]

        o_seq = pallas_flash_attention_forward(
            q_seq,
            k_seq,
            v_seq,
            seq_start=0,
            prefix_len=plen,
            tokens_per_item=tpi,
            sm_scale=sm_scale,
            block_q=block_q,
            block_k=block_k,
            interpret=interpret,
        )
        seq_outputs.append(o_seq)

    return jnp.concatenate(seq_outputs, axis=0)


# ---------------------------------------------------------------------------
# D. High-level wrapper for FA4 (4D format)
# ---------------------------------------------------------------------------


def pallas_flash_attention_4d(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    cu_seqlens: jax.Array,
    prefix_lens: jax.Array,
    tokens_per_item: jax.Array | int,
    sm_scale: float,
    block_q: int = 128,
    block_k: int = 128,
    interpret: bool = False,
) -> jax.Array:
    """Flash attention with target attention mask for 4D tensors.

    Args:
        q: [B, num_heads, seq_len, head_dim]
        k: [B, num_kv_heads, seq_len, head_dim]
        v: [B, num_kv_heads, seq_len, head_dim]
        cu_seqlens: cumulative sequence lengths within the packed dimension.
        prefix_lens: prefix lengths per sub-sequence.
        tokens_per_item: tokens per item per sub-sequence or scalar.
        sm_scale: softmax scale.
        block_q: Q block size.
        block_k: K block size.
        interpret: if True, run on CPU (for testing). Default False (TPU).

    Returns:
        Output of shape [B, num_heads, seq_len, head_dim].
    """
    batch_size, num_heads, seq_len, head_dim = q.shape

    cu_seqlens_np = np.asarray(cu_seqlens)
    prefix_lens_np = np.asarray(prefix_lens)
    if isinstance(tokens_per_item, (int, float)):
        tokens_per_item_np: Any = int(tokens_per_item)
    else:
        tokens_per_item_np = np.asarray(tokens_per_item)

    num_seqs = cu_seqlens_np.shape[0] - 1

    out_batch = []

    for bi in range(batch_size):
        # For each batch element, iterate over sub-sequences.
        # Transpose to [seq_len, num_heads, head_dim] for the kernel.
        q_b = jnp.transpose(q[bi], (1, 0, 2))  # [seq_len, H, D]
        k_b = jnp.transpose(k[bi], (1, 0, 2))  # [seq_len, Hkv, D]
        v_b = jnp.transpose(v[bi], (1, 0, 2))

        seq_outputs = []
        for s in range(num_seqs):
            start = int(cu_seqlens_np[s])
            end = int(cu_seqlens_np[s + 1])
            plen = int(prefix_lens_np[s])
            if isinstance(tokens_per_item_np, int):
                tpi = tokens_per_item_np
            else:
                tpi = int(tokens_per_item_np[s])

            q_slice = q_b[start:end, :, :]  # [sub_seq_len, H, D]
            k_slice = k_b[start:end, :, :]
            v_slice = v_b[start:end, :, :]

            o_slice = pallas_flash_attention_forward(
                q_slice,
                k_slice,
                v_slice,
                seq_start=0,
                prefix_len=plen,
                tokens_per_item=tpi,
                sm_scale=sm_scale,
                block_q=block_q,
                block_k=block_k,
                interpret=interpret,
            )
            seq_outputs.append(o_slice)

        # Concatenate sub-sequences -> [seq_len, H, D]
        batch_out = jnp.concatenate(seq_outputs, axis=0)
        # Transpose back to [H, seq_len, D]
        batch_out = jnp.transpose(batch_out, (1, 0, 2))
        out_batch.append(batch_out)

    # [B, num_heads, seq_len, head_dim]
    return jnp.stack(out_batch, axis=0)
