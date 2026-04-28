# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Optimized Pallas FlashAttention kernel for TPU with target attention mask.

This module provides a tiled FlashAttention-2 implementation using JAX's Pallas
API targeting TPU.  It supports both varlen (FA3) and 4D (FA4) formats.

Key optimizations over the baseline (D96262305):
  P0: ``interpret`` defaults to False — runs on TPU hardware, not CPU-emulated.
  P1: Head dimension folded into the Pallas grid — one ``pallas_call`` per
      sequence instead of one per (batch, head) pair.
  P2: Causal KV loop upper bound — skips KV blocks that are entirely masked.
  P3: Batch dimension folded into the Pallas grid — one ``pallas_call`` for
      the entire batch, eliminating per-sequence dispatch overhead (~133ms each).
"""

from __future__ import annotations

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
# B2. P3 — Batched Pallas FlashAttention kernel (batch in grid)
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
):
    """Return a batched Pallas kernel closed over compile-time constants.

    The kernel is invoked on a 3D grid: ``(num_q_blocks, num_heads, batch_size)``.
    All sequences in the batch must share the same ``seq_len``, ``prefix_len``,
    and ``tokens_per_item`` (the uniform-batch fast path).

    Input refs use layout ``[1, 1, seq_or_block, head_dim]`` where the first
    two dims select batch element and head via BlockSpec.  Last two dims
    ``(block_q, head_dim)`` satisfy TPU alignment constraints.
    """
    qk_scale = sm_scale * math.log2(math.e)

    def flash_attn_kernel(q_ref, k_ref, v_ref, o_ref):
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

    return flash_attn_kernel


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
) -> jax.Array:
    """Batched flash attention forward pass using a single ``pallas_call``.

    All sequences must have the same length, prefix_len, and tokens_per_item.

    Args:
        q: [batch_size, seq_len, num_heads, head_dim]
        k: [batch_size, seq_len, num_kv_heads, head_dim]
        v: [batch_size, seq_len, num_kv_heads, head_dim]
        prefix_len: number of prefix tokens (uniform across batch).
        tokens_per_item: tokens per item (uniform across batch).
        sm_scale: softmax scale (typically 1/sqrt(head_dim)).
        block_q: tile size along Q sequence dimension.
        block_k: tile size along KV sequence dimension.
        interpret: if True, run on CPU (for testing). Default False (TPU).

    Returns:
        Output of shape [batch_size, seq_len, num_heads, head_dim].
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
    )

    num_q_blocks = pl.cdiv(padded_seq_len, block_q)

    result = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(
            (batch_size, num_heads, padded_seq_len, head_dim), dtype
        ),
        grid=(num_q_blocks, num_heads, batch_size),
        in_specs=[
            # Q: [B, H, S, D] -> (1, 1, block_q, D)
            pl.BlockSpec(
                (1, 1, block_q, head_dim), lambda q_blk, h, b: (b, h, q_blk, 0)
            ),
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
        ],
        out_specs=pl.BlockSpec(
            (1, 1, block_q, head_dim), lambda q_blk, h, b: (b, h, q_blk, 0)
        ),
        interpret=interpret,
    )(q_t, k_t, v_t)

    # Transpose back to [B, S, H, D]
    result = jnp.transpose(result, (0, 2, 1, 3))

    # Remove padding
    if pad_len > 0:
        result = result[:, :seq_len, :, :]

    return result


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

        result = pallas_flash_attention_batched(
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
