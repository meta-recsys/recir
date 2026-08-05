# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""Flash-style (KV-tiled) HSTU SiLU attention kernel for TPU via Pallas.

Reproduces the HSTU materialized-attention math (generative_recommenders
``pytorch_hstu_mha``) but WITHOUT ever materializing the full ``[L, L]`` score
matrix: the KV sequence is tiled along ``L`` and the output is accumulated block
by block. Because HSTU uses **SiLU, not softmax**, there is no cross-key
normalization (no online running-max/sum), so flash attention here is just a
masked weighted sum accumulated over KV blocks:

    out[i] = sum over kv-blocks of ( silu(q @ k_blk^T * alpha) * inv_n * mask ) @ v_blk

This keeps VMEM at ``O(L * block_k)`` instead of ``O(L^2)``, so it fits at
S=4096 (the earlier fully-materialized kernel OOMed VMEM at ~65MB for one
``[L,L]`` f32 tile).

Math (matches pt_hstu_attention.py, causal + num_targets, no contextual / local):
    scores  = (q @ k^T) * alpha            # alpha = 1/sqrt(Dqk)
    weights = silu(scores) * inv_n         # inv_n = 1/max_seq_len; NOT softmax
    weights = where(valid_mask, weights, 0)    # causal + target + length
    out     = weights @ v

Only the KV axis is tiled (grid stays ``(B, H)``); q/k/v stay whole in VMEM (they
are ``O(L*D)``, not the bottleneck) and the per-block score tile is
``[L, block_k]``. ``L`` is padded up to a multiple of ``block_k`` so the
fixed-size KV blocks never read past the sequence; padded rows/cols are masked
out (position >= length).

Shapes:
    q, k         : [B, H, L, Dqk]
    v            : [B, H, L, Dv]
    lengths      : [B]  int
    num_targets  : [B]  int   (0 = pure causal)
    output       : [B, H, L, Dv]

Differentiable via ``jax.custom_vjp``; the backward is the pure-JAX reference
(``hstu_attention_reference``) -- XLA handles its ``[L,L]`` in HBM (not Pallas VMEM), so
it is not the VMEM bottleneck the tiled forward addresses.

NOTE (TPU): ``lengths`` / ``num_targets`` are passed in SMEM and indexed by
``program_id`` -- a VMEM scalar used as a mask bound miscompiles and the TPU core
halts (openxla error_0200). The math is validated via ``interpret=True``; the
Mosaic lowering itself needs a TPU run.
"""

from __future__ import annotations

import functools
import math
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _fwd_kernel(
    q_ref,
    k_ref,
    v_ref,
    length_ref,
    ntarget_ref,
    o_ref,
    *,
    alpha: float,
    inv_n: float,
    block_k: int,
) -> None:
    # One (batch, head) tile. seq_len here is the PADDED length (multiple of
    # block_k); padded positions are masked out via ``length``.
    b = pl.program_id(0)
    q = q_ref[0, 0]  # [L, Dqk]
    seq_len = q.shape[0]
    dv = v_ref.shape[-1]
    # Per-row scalar metadata from SMEM (see module docstring).
    length = length_ref[b]
    num_targets = ntarget_ref[b]

    q_pos = jnp.arange(seq_len)
    max_id = length - num_targets  # history boundary; targets collapse to it
    q_id = jnp.minimum(q_pos, max_id)  # [L]

    def body(kv, out):
        start = kv * block_k
        k_blk = k_ref[0, 0, pl.ds(start, block_k), :]  # [block_k, Dqk]
        v_blk = v_ref[0, 0, pl.ds(start, block_k), :]  # [block_k, Dv]
        k_pos = start + jnp.arange(block_k)
        k_id = jnp.minimum(k_pos, max_id)

        scores = jax.lax.dot_general(
            q,
            k_blk,
            dimension_numbers=(((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )  # [L, block_k]
        # HSTU: SiLU (not softmax) then the 1/N scale.
        weights = jax.nn.silu(scores * alpha) * inv_n
        # Causal + num_targets mask for this [L, block_k] tile. Derive the length
        # bounds from 2D int position views (q_pos[:, None] < length): reshaping a
        # 1D BOOLEAN vector to [L, 1] miscompiles in Mosaic ("unsupported shape
        # cast"), whereas the int [:, None] reshape is supported (as used by the
        # TPU flash_attention kernel).
        causal = (q_id[:, None] - k_id[None, :]) > 0
        eye = q_pos[:, None] == k_pos[None, :]
        valid = (causal | eye) & (q_pos[:, None] < length) & (k_pos[None, :] < length)
        weights = jnp.where(valid, weights, 0.0).astype(v_blk.dtype)
        return out + jax.lax.dot_general(
            weights,
            v_blk,
            dimension_numbers=(((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )

    num_kv_blocks = seq_len // block_k
    out: Any = jax.lax.fori_loop(
        0, num_kv_blocks, body, jnp.zeros((seq_len, dv), dtype=jnp.float32)
    )
    # Zero the padded query rows so the output stays clean. Use the 2D int
    # position view (not a 1D boolean reshape) for the same Mosaic reason.
    masked_out: Any = jnp.where(q_pos[:, None] < length, out, 0.0)
    o_ref[0, 0] = masked_out.astype(o_ref.dtype)


def _pallas_fwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    lengths: jax.Array,
    num_targets: jax.Array,
    alpha: float,
    inv_n: float,
    interpret: bool,
    block_k: int,
) -> jax.Array:
    b, h, seq_len, dqk = q.shape
    dv = v.shape[-1]
    # Pad L up to a multiple of block_k so the fixed-size KV blocks never read
    # past the sequence. Padded keys/queries are masked (position >= length).
    pad = (-seq_len) % block_k
    if pad:
        cfg = ((0, 0), (0, 0), (0, pad), (0, 0))
        q = jnp.pad(q, cfg)
        k = jnp.pad(k, cfg)
        v = jnp.pad(v, cfg)
    seq_pad = seq_len + pad

    kernel = functools.partial(_fwd_kernel, alpha=alpha, inv_n=inv_n, block_k=block_k)
    out = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((b, h, seq_pad, dv), q.dtype),
        grid=(b, h),
        in_specs=[
            pl.BlockSpec((1, 1, seq_pad, dqk), lambda bi, hi: (bi, hi, 0, 0)),
            pl.BlockSpec((1, 1, seq_pad, dqk), lambda bi, hi: (bi, hi, 0, 0)),
            pl.BlockSpec((1, 1, seq_pad, dv), lambda bi, hi: (bi, hi, 0, 0)),
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(memory_space=pltpu.SMEM),
        ],
        out_specs=pl.BlockSpec((1, 1, seq_pad, dv), lambda bi, hi: (bi, hi, 0, 0)),
        interpret=interpret,
    )(q, k, v, lengths, num_targets)
    return out[:, :, :seq_len, :]


def hstu_attention_reference(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    lengths: jax.Array,
    num_targets: jax.Array,
    alpha: float,
    inv_n: float,
) -> jax.Array:
    """Pure-JAX materialized HSTU attention (batched). Used as the backward and
    as the parity reference in tests."""
    seq_len = q.shape[2]
    scores = jnp.einsum("bhid,bhjd->bhij", q, k) * alpha
    weights = jax.nn.silu(scores) * inv_n
    pos = jnp.arange(seq_len)
    max_id = (lengths - num_targets)[:, None]  # [B, 1]
    ids = jnp.minimum(pos[None, :], max_id)  # [B, L]
    row = ids[:, :, None]
    col = ids[:, None, :]
    causal = (row - col) > 0
    eye = (pos[:, None] == pos[None, :])[None]  # [1, L, L]
    valid_pos = pos[None, :] < lengths[:, None]  # [B, L]
    in_bounds = valid_pos[:, :, None] & valid_pos[:, None, :]
    valid = ((causal | eye) & in_bounds)[:, None]  # [B, 1, L, L]
    weights = jnp.where(valid, weights, 0.0).astype(v.dtype)
    out = jnp.einsum("bhij,bhjd->bhid", weights, v)
    return jnp.where(valid_pos[:, None, :, None], out, 0.0)


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 8))
def _attention(q, k, v, lengths, num_targets, alpha, inv_n, interpret, block_k):
    return _pallas_fwd(q, k, v, lengths, num_targets, alpha, inv_n, interpret, block_k)


def _attention_fwd(q, k, v, lengths, num_targets, alpha, inv_n, interpret, block_k):
    out = _pallas_fwd(q, k, v, lengths, num_targets, alpha, inv_n, interpret, block_k)
    return out, (q, k, v, lengths, num_targets)


def _attention_bwd(alpha, inv_n, interpret, block_k, residuals, g):
    q, k, v, lengths, num_targets = residuals
    # Reference (materialized JAX) backward; XLA handles its [L,L] in HBM, so it
    # is not VMEM-bound the way the tiled Pallas forward is.
    _, vjp_fn = jax.vjp(
        lambda q_, k_, v_: hstu_attention_reference(
            q_, k_, v_, lengths, num_targets, alpha, inv_n
        ),
        q,
        k,
        v,
    )
    dq, dk, dv = vjp_fn(g)
    return (dq, dk, dv, None, None)  # no grads for int lengths / num_targets


_attention.defvjp(_attention_fwd, _attention_bwd)


def pallas_hstu_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    lengths: jax.Array,
    num_targets: jax.Array | None = None,
    scale: float | None = None,
    inv_n: float | None = None,
    interpret: bool = False,
    block_k: int = 512,
) -> jax.Array:
    """Flash-style (KV-tiled) HSTU SiLU attention over ``[B, H, L, D]``.

    Args:
        q, k: ``[B, H, L, Dqk]`` dense tensors.
        v: ``[B, H, L, Dv]`` dense tensor.
        lengths: ``[B]`` int valid token counts per row.
        num_targets: ``[B]`` int candidate counts (target-attention); ``None``
            -> zeros (pure causal).
        scale: pre-SiLU score multiplier ``alpha``. Default ``1/sqrt(Dqk)``.
        inv_n: post-SiLU multiplier. Default ``1/L`` (HSTU ``1/max_seq_len``).
        interpret: run on CPU via the Pallas interpreter (for testing).
        block_k: KV tile length along the sequence axis. Trades VMEM (score tile
            is ``[L, block_k]``) vs. loop count. Should be a multiple of 128 on
            TPU; ``L`` is padded up to a multiple of it.

    Returns:
        ``[B, H, L, Dv]`` attention output; padded query rows are zero.
    """
    b, _h, seq_len, dqk = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(dqk)
    if inv_n is None:
        inv_n = 1.0 / seq_len
    if num_targets is None:
        num_targets = jnp.zeros((b,), dtype=lengths.dtype)
    return _attention(q, k, v, lengths, num_targets, scale, inv_n, interpret, block_k)
