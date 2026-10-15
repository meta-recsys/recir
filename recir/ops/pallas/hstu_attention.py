# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Flash-style (query- and KV-tiled) HSTU SiLU attention kernel for TPU via Pallas.

Reproduces the HSTU materialized-attention math (generative_recommenders
``pytorch_hstu_mha``) but WITHOUT ever materializing the full ``[L, L]`` score
matrix: the query and KV axes are tiled along ``L`` and output is accumulated block
by block. Because HSTU uses **SiLU, not softmax**, there is no cross-key
normalization (no online running-max/sum), so flash attention here is just a
masked weighted sum accumulated over KV blocks:

    out[i] = sum over kv-blocks of ( silu(q @ k_blk^T * alpha) * inv_n * mask ) @ v_blk

This keeps score VMEM at ``O(block_q * block_k)`` instead of ``O(L^2)``, so it fits at
S=4096 (the earlier fully-materialized kernel OOMed VMEM at ~65MB for one
``[L,L]`` f32 tile).

Math (matches pt_hstu_attention.py, causal + num_targets, no contextual / local):
    scores  = (q @ k^T) * alpha            # alpha = 1/sqrt(Dqk)
    weights = silu(scores) * inv_n         # inv_n = 1/max_seq_len; NOT softmax
    weights = where(valid_mask, weights, 0)    # causal + target + length
    out     = weights @ v

Query and KV axes are tiled. The grid is ``(ceil(L / block_q), H, B)``;
q is loaded per query program while each program streams KV blocks. The score tile is
``[block_q, block_k]``. ``L`` is padded to a common query/KV block
boundary so fixed-size blocks never read past the sequence; padding is masked
out (position >= length).

Shapes:
    q, k         : [B, H, L, Dqk]
    v            : [B, H, L, Dv]
    lengths      : [B]  int
    num_targets  : [B]  int   (0 = pure causal)
    output       : [B, H, L, Dv]

Differentiable via ``jax.custom_vjp``; the backward uses tiled Pallas dQ and dKV kernels

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
    block_q: int,
) -> None:
    # One (batch, head, query-block) tile. seq_len is the padded KV length.
    q_block = pl.program_id(0)
    b = pl.program_id(2)
    q = q_ref[0, 0]  # [block_q, Dqk]
    seq_len = k_ref.shape[2]
    q_start = q_block * block_q
    dv = v_ref.shape[-1]
    # Per-row scalar metadata from SMEM (see module docstring).
    length = length_ref[b]
    num_targets = ntarget_ref[b]

    q_pos = q_start + jnp.arange(block_q)
    max_id = length - num_targets  # history boundary; targets collapse to it
    q_id = jnp.minimum(q_pos, max_id)  # [block_q]

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
        )  # [block_q, block_k]
        # HSTU: SiLU (not softmax) then the 1/N scale.
        weights = jax.nn.silu(scores * alpha) * inv_n
        # Causal + num_targets mask for this [block_q, block_k] tile. Derive the length
        # bounds from 2D int position views (q_pos[:, None] < length): reshaping a
        # 1D BOOLEAN vector to [block_q, 1] miscompiles in Mosaic ("unsupported shape
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

    # No query can attend beyond its block end: history is causal and each
    # target row adds only its own diagonal position. Skip wholly future KV
    # blocks, and skip all work for query blocks beyond the row length.
    q_end = jnp.minimum(q_start + block_q, length)
    causal_upper = pl.cdiv(q_end, block_k)
    causal_upper = jnp.where(q_start < length, causal_upper, 0)
    num_kv_blocks = seq_len // block_k
    out: Any = jax.lax.fori_loop(
        0,
        jnp.minimum(num_kv_blocks, causal_upper),
        body,
        jnp.zeros((block_q, dv), dtype=jnp.float32),
    )
    # Zero the padded query rows so the output stays clean. Use the 2D int
    # position view (not a 1D boolean reshape) for the same Mosaic reason.
    masked_out: Any = jnp.where(q_pos[:, None] < length, out, 0.0)
    o_ref[0, 0] = jnp.asarray(masked_out, dtype=o_ref.dtype)


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
    block_q: int,
) -> jax.Array:
    b, h, seq_len, dqk = q.shape
    dv = v.shape[-1]
    # Pad L to a common query/KV block boundary so every BlockSpec stays in bounds.
    block_multiple = math.lcm(block_q, block_k)
    pad = (-seq_len) % block_multiple
    if pad:
        cfg = ((0, 0), (0, 0), (0, pad), (0, 0))
        q = jnp.pad(q, cfg)
        k = jnp.pad(k, cfg)
        v = jnp.pad(v, cfg)
    seq_pad = seq_len + pad

    kernel = functools.partial(
        _fwd_kernel,
        alpha=alpha,
        inv_n=inv_n,
        block_k=block_k,
        block_q=block_q,
    )
    out = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((b, h, seq_pad, dv), q.dtype),
        grid=(seq_pad // block_q, h, b),
        in_specs=[
            pl.BlockSpec((1, 1, block_q, dqk), lambda qi, hi, bi: (bi, hi, qi, 0)),
            pl.BlockSpec((1, 1, seq_pad, dqk), lambda qi, hi, bi: (bi, hi, 0, 0)),
            pl.BlockSpec((1, 1, seq_pad, dv), lambda qi, hi, bi: (bi, hi, 0, 0)),
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(memory_space=pltpu.SMEM),
        ],
        out_specs=pl.BlockSpec((1, 1, block_q, dv), lambda qi, hi, bi: (bi, hi, qi, 0)),
        interpret=interpret,
    )(q, k, v, lengths, num_targets)
    return out[:, :, :seq_len, :]


def _silu_grad(x: jax.Array) -> jax.Array:
    sigmoid = jax.nn.sigmoid(x)
    return sigmoid * (1.0 + x * (1.0 - sigmoid))


def _bwd_dq_kernel(
    q_ref,
    k_ref,
    v_ref,
    do_ref,
    length_ref,
    ntarget_ref,
    dq_ref,
    *,
    alpha: float,
    inv_n: float,
    block_k: int,
    block_q: int,
) -> None:
    q_block = pl.program_id(0)
    b = pl.program_id(2)
    q = q_ref[0, 0]
    do = do_ref[0, 0]
    seq_len = k_ref.shape[2]
    q_start = q_block * block_q
    length = length_ref[b]
    max_id = length - ntarget_ref[b]
    q_pos = q_start + jnp.arange(block_q)
    q_id = jnp.minimum(q_pos, max_id)

    def body(kv, dq):
        start = kv * block_k
        k = k_ref[0, 0, pl.ds(start, block_k), :]
        v = v_ref[0, 0, pl.ds(start, block_k), :]
        k_pos = start + jnp.arange(block_k)
        k_id = jnp.minimum(k_pos, max_id)
        scores = jax.lax.dot_general(
            q,
            k,
            dimension_numbers=(((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        dp = jax.lax.dot_general(
            do,
            v,
            dimension_numbers=(((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        causal = (q_id[:, None] - k_id[None, :]) > 0
        eye = q_pos[:, None] == k_pos[None, :]
        valid = (causal | eye) & (q_pos[:, None] < length) & (k_pos[None, :] < length)
        ds = jnp.where(valid, dp * _silu_grad(scores * alpha) * inv_n, 0.0)
        return dq + alpha * jax.lax.dot_general(
            ds.astype(k.dtype),
            k,
            dimension_numbers=(((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )

    q_end = jnp.minimum(q_start + block_q, length)
    upper = jnp.where(q_start < length, pl.cdiv(q_end, block_k), 0)
    dq: Any = jax.lax.fori_loop(
        0,
        jnp.minimum(seq_len // block_k, upper),
        body,
        jnp.zeros_like(q, dtype=jnp.float32),
    )
    dq_ref[0, 0] = jnp.asarray(
        jnp.where(q_pos[:, None] < length, dq, 0.0), dtype=dq_ref.dtype
    )


def _bwd_dkv_kernel(
    q_ref,
    k_ref,
    v_ref,
    do_ref,
    length_ref,
    ntarget_ref,
    dk_ref,
    dv_ref,
    *,
    alpha: float,
    inv_n: float,
    block_k: int,
    block_q: int,
) -> None:
    kv_block = pl.program_id(0)
    b = pl.program_id(2)
    k = k_ref[0, 0]
    v = v_ref[0, 0]
    seq_len = q_ref.shape[2]
    k_start = kv_block * block_k
    length = length_ref[b]
    max_id = length - ntarget_ref[b]
    k_pos = k_start + jnp.arange(block_k)
    k_id = jnp.minimum(k_pos, max_id)

    def body(qi, carry):
        dk, dv = carry
        q_start = qi * block_q
        q = q_ref[0, 0, pl.ds(q_start, block_q), :]
        do = do_ref[0, 0, pl.ds(q_start, block_q), :]
        q_pos = q_start + jnp.arange(block_q)
        q_id = jnp.minimum(q_pos, max_id)
        scores = jax.lax.dot_general(
            q,
            k,
            dimension_numbers=(((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        scaled_scores = scores * alpha
        causal = (q_id[:, None] - k_id[None, :]) > 0
        eye = q_pos[:, None] == k_pos[None, :]
        valid = (causal | eye) & (q_pos[:, None] < length) & (k_pos[None, :] < length)
        weights = jnp.where(valid, jax.nn.silu(scaled_scores) * inv_n, 0.0)
        dv_curr = jax.lax.dot_general(
            weights.astype(do.dtype),
            do,
            dimension_numbers=(((0,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        dp = jax.lax.dot_general(
            do,
            v,
            dimension_numbers=(((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        ds = jnp.where(valid, dp * _silu_grad(scaled_scores) * inv_n, 0.0)
        dk_curr = jax.lax.dot_general(
            ds.astype(q.dtype),
            q,
            dimension_numbers=(((0,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        return dk + alpha * dk_curr, dv + dv_curr

    dk, dv = jax.lax.fori_loop(
        k_start // block_q,
        seq_len // block_q,
        body,
        (jnp.zeros_like(k, dtype=jnp.float32), jnp.zeros_like(v, dtype=jnp.float32)),
    )
    valid_k = k_pos[:, None] < length
    dk_ref[0, 0] = jnp.asarray(jnp.where(valid_k, dk, 0.0), dtype=dk_ref.dtype)
    dv_ref[0, 0] = jnp.asarray(jnp.where(valid_k, dv, 0.0), dtype=dv_ref.dtype)


def pallas_hstu_attention_backward(
    grad_out: jax.Array,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    lengths: jax.Array,
    num_targets: jax.Array,
    alpha: float,
    inv_n: float,
    interpret: bool = False,
    block_k: int = 512,
    block_q: int = 512,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Tiled Pallas backward for HSTU SiLU attention."""
    b, h, seq_len, dqk = q.shape
    dv_dim = v.shape[-1]
    pad = (-seq_len) % math.lcm(block_q, block_k)
    if pad:
        cfg = ((0, 0), (0, 0), (0, pad), (0, 0))
        q, k, v, grad_out = (jnp.pad(x, cfg) for x in (q, k, v, grad_out))
    seq_pad = seq_len + pad

    dq = pl.pallas_call(
        functools.partial(
            _bwd_dq_kernel, alpha=alpha, inv_n=inv_n, block_k=block_k, block_q=block_q
        ),
        out_shape=jax.ShapeDtypeStruct((b, h, seq_pad, dqk), q.dtype),
        grid=(seq_pad // block_q, h, b),
        in_specs=[
            pl.BlockSpec((1, 1, block_q, dqk), lambda qi, hi, bi: (bi, hi, qi, 0)),
            pl.BlockSpec((1, 1, seq_pad, dqk), lambda qi, hi, bi: (bi, hi, 0, 0)),
            pl.BlockSpec((1, 1, seq_pad, dv_dim), lambda qi, hi, bi: (bi, hi, 0, 0)),
            pl.BlockSpec((1, 1, block_q, dv_dim), lambda qi, hi, bi: (bi, hi, qi, 0)),
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(memory_space=pltpu.SMEM),
        ],
        out_specs=pl.BlockSpec(
            (1, 1, block_q, dqk), lambda qi, hi, bi: (bi, hi, qi, 0)
        ),
        interpret=interpret,
    )(q, k, v, grad_out, lengths, num_targets)

    dk, dv = pl.pallas_call(
        functools.partial(
            _bwd_dkv_kernel, alpha=alpha, inv_n=inv_n, block_k=block_k, block_q=block_q
        ),
        out_shape=[
            jax.ShapeDtypeStruct((b, h, seq_pad, dqk), k.dtype),
            jax.ShapeDtypeStruct((b, h, seq_pad, dv_dim), v.dtype),
        ],
        grid=(seq_pad // block_k, h, b),
        in_specs=[
            pl.BlockSpec((1, 1, seq_pad, dqk), lambda ki, hi, bi: (bi, hi, 0, 0)),
            pl.BlockSpec((1, 1, block_k, dqk), lambda ki, hi, bi: (bi, hi, ki, 0)),
            pl.BlockSpec((1, 1, block_k, dv_dim), lambda ki, hi, bi: (bi, hi, ki, 0)),
            pl.BlockSpec((1, 1, seq_pad, dv_dim), lambda ki, hi, bi: (bi, hi, 0, 0)),
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(memory_space=pltpu.SMEM),
        ],
        out_specs=[
            pl.BlockSpec((1, 1, block_k, dqk), lambda ki, hi, bi: (bi, hi, ki, 0)),
            pl.BlockSpec((1, 1, block_k, dv_dim), lambda ki, hi, bi: (bi, hi, ki, 0)),
        ],
        interpret=interpret,
    )(q, k, v, grad_out, lengths, num_targets)
    return dq[:, :, :seq_len], dk[:, :, :seq_len], dv[:, :, :seq_len]


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


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 8, 9))
def _attention(
    q, k, v, lengths, num_targets, alpha, inv_n, interpret, block_k, block_q
):
    return _pallas_fwd(
        q, k, v, lengths, num_targets, alpha, inv_n, interpret, block_k, block_q
    )


def _attention_fwd(
    q, k, v, lengths, num_targets, alpha, inv_n, interpret, block_k, block_q
):
    out = _pallas_fwd(
        q, k, v, lengths, num_targets, alpha, inv_n, interpret, block_k, block_q
    )
    return out, (q, k, v, lengths, num_targets)


def _attention_bwd(alpha, inv_n, interpret, block_k, block_q, residuals, g):
    q, k, v, lengths, num_targets = residuals
    dq, dk, dv = pallas_hstu_attention_backward(
        g, q, k, v, lengths, num_targets, alpha, inv_n, interpret, block_k, block_q
    )
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
    block_q: int = 512,
) -> jax.Array:
    """Flash-style (query- and KV-tiled) HSTU SiLU attention over ``[B, H, L, D]``.

    Args:
        q, k: ``[B, H, L, Dqk]`` dense tensors.
        v: ``[B, H, L, Dv]`` dense tensor.
        lengths: ``[B]`` int valid token counts per row.
        num_targets: ``[B]`` int candidate counts (target-attention); ``None``
            -> zeros (pure causal).
        scale: pre-SiLU score multiplier ``alpha``. Default ``1/sqrt(Dqk)``.
        inv_n: post-SiLU multiplier. Default ``1/L`` (HSTU ``1/max_seq_len``).
        interpret: run on CPU via the Pallas interpreter (for testing).
        block_k: KV tile length. Should be a multiple of 128 on TPU.
        block_q: Query tile length. Controls each program's score and output
            working set and the number of query programs. Should be a multiple
            of 128 on TPU. ``L`` is padded to a common block boundary.

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
    return _attention(
        q,
        k,
        v,
        lengths,
        num_targets,
        scale,
        inv_n,
        interpret,
        block_k,
        block_q,
    )
