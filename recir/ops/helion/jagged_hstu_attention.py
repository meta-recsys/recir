# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""recir-owned, config-pinned Helion jagged HSTU attention.

A recir port of the official Helion example ``helion/examples/jagged_hstu_attn_2.py``
(forward + dQ/dK/dV backward + target-aware masking). The causal reference, the
``_hstu_keep_mask``, the tiling, and the transposed dK/dV backward layout follow the
official file; the recir-specific adaptations are scale folding and a fixed
RankFM 256-local/256-full-tail specialization (below).

Official math (per head, per sequence)::

    S   = alpha * (Q @ K^T)          # pre-activation scores
    P   = silu(S) * attn_scale       # SiLU-gated, unnormalized (no softmax)
    O   = (P * keep_mask) @ V
    dP  = dO @ V^T
    dV  = (keep_mask * silu(S) * attn_scale)^T @ dO
    dS  = dP * keep_mask * attn_scale * silu'(S)
    dQ  = alpha * (dS @ K)
    dK  = alpha * (dS^T @ Q)

with ``silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))``.

Scale folding (recir-specific)
------------------------------
``Kernel.jax_fn`` cannot pass scalar float kernel args: the Helion host wrapper
materializes them as real ``torch.tensor`` that are not ``_JaxExportTensor``
adapters, so Pallas raises ``'Tensor' object has no attribute '_jax_arr'``. Since
``alpha``/``attn_scale`` are LINEAR, the ``*_jax_fn`` wrappers fold them into the
tensor inputs and the kernels are scale-free (the official kernels with the
``* alpha`` / ``* attn_scale`` factors removed)::

  forward :  q *= alpha, v *= attn_scale
  backward:  dO_s = attn_scale * dO
             dV : (alpha q, k, dO_s)     ; dv = P^T @ dO_s
             dQ : (q, alpha k, v, dO_s)  ; dq = dS @ (alpha k)
             dK : (alpha q, k, v, dO_s)  ; dk = dS^T @ (alpha q)

Whichever of q/k carries the alpha fold reproduces ``S = alpha q@k^T`` in-kernel, so
silu(S)/silu'(S) are exact and the grads come back w.r.t. the ORIGINAL q/k/v.

Target-aware mask (matches the hammer ``get_invalid_attn_mask`` clamp,
hammer/ops/pytorch/ragged_hstu_attention.py): per sequence, with global token indices
and ``cap = end - num_targets[seq_idx]``::

    keep(m, n) = ((m >= n) & (n < cap)) | (m == n)     # m = query, n = key

History keys (n < cap) are attended causally; the last ``num_targets`` keys are
attended only on the diagonal. The wrappers pass zeros when there are no targets,
giving ``cap == end`` -> plain causal.

``reference_jagged_hstu_attention`` is the pad-then-compute torch reference: the
recir ops' CompositeExplicitAutograd / Meta fallback, and (via its autograd,
``reference_jagged_hstu_attention_bwd``) the numerical oracle for both directions.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Protocol

import helion
import helion.language as hl
import torch
import torch.nn.functional as F

logger: logging.Logger = logging.getLogger(__name__)

# Block sizes for the (tile_q, tile_kv) tile loops. pallas_worklist_grouping=2
# flattens the (sequence, q-tile) pairs into a 1-D worklist grid, so each program
# handles one fixed [block, H, D] window -- VMEM is bounded by block, not the
# per-sequence L. pallas_loop_type="emit_pipeline" lowers the ordered kv-tile
# reduction via pltpu.emit_pipeline, streaming the kv tiles through double-buffered
# VMEM (prefetch overlapped with compute) instead of staging the whole per-sequence
# window resident (the "unroll" resident-cache path). The kernel's packed
# offsets[seq]/offsets[seq+1] bounds satisfy the worklist pattern (owner grid +
# compact q-tile + ordered kv-tile).
#
# The two settings are PINNED TOGETHER and validated as a pair against the 64MB
# physical device VMEM cap: streaming bounds the footprint by the double-buffered
# kv-tile pair rather than the per-sequence window, which is what lets the block be
# 512. Do not change one without re-validating -- the cap is physical (no
# --xla_tpu_scoped_vmem_limit_kib raise absorbs it) and an over-budget config fails
# late and opaquely, surfacing from the AOT warm as a nested-compile
# "SymIntArrayRef ..." rather than as an OOM.
PINNED_CONFIG: helion.Config = helion.Config(
    block_sizes=[512, 512],
    pallas_worklist_grouping=2,
    pallas_loop_type="emit_pipeline",
)

# RankFM's L1 stack uses this fixed semi-local attention shape.  Keep the
# constants static: the Pallas compact-worklist builder requires the dependent
# tile bounds below to be compile-time expressions rather than scalar kernel
# arguments.  L2 still uses the ordinary causal kernels above.
RANKFM_MAX_ATTN_LEN: int = 256
RANKFM_FULL_ATTN_SIZE: int = 256
SEMILOCAL_PINNED_CONFIG: helion.Config = helion.Config(
    block_sizes=[256, 256, 256],
    pallas_worklist_grouping=2,
    pallas_loop_type="emit_pipeline",
)


class _TileLike(Protocol):
    @property
    def index(self) -> torch.Tensor: ...


def use_semilocal_attention(max_attn_len: int, full_attn_size: int) -> bool:
    """Select the causal or fixed RankFM semi-local kernel specialization."""
    if max_attn_len == 0 and full_attn_size == 0:
        return False
    if max_attn_len == RANKFM_MAX_ATTN_LEN and full_attn_size == RANKFM_FULL_ATTN_SIZE:
        return True
    raise ValueError(
        "Helion HSTU attention supports either causal attention "
        "(max_attn_len=0, full_attn_size=0) or RankFM semi-local attention "
        f"(max_attn_len={RANKFM_MAX_ATTN_LEN}, "
        f"full_attn_size={RANKFM_FULL_ATTN_SIZE}); got "
        f"max_attn_len={max_attn_len}, full_attn_size={full_attn_size}"
    )


def _semilocal_keep_mask(
    tile_q: _TileLike,
    tile_kv: _TileLike,
    history_end: torch.Tensor,
) -> torch.Tensor:
    """Hammer's causal 256-local + 256-full-tail mask, including target rules."""
    query = tile_q.index.unsqueeze(1)
    key = tile_kv.index.unsqueeze(0)
    # For short histories this falls before the sequence start, intentionally
    # making every query use full causal attention over the available history.
    full_start = history_end - RANKFM_FULL_ATTN_SIZE
    history_key = key < history_end
    local = (
        (query >= key)
        & (query - key <= RANKFM_MAX_ATTN_LEN)
        & (query < full_start)
        & history_key
    )
    full = (query >= key) & (query >= full_start) & history_key
    return local | full | (query == key)


# ---------------------------------------------------------------------------
# Pad-then-compute torch reference (target-aware, differentiable) -- verbatim
# from helion/examples/jagged_hstu_attn_2.py.
# ---------------------------------------------------------------------------
def _jagged_to_padded(
    values: torch.Tensor,
    offsets: torch.Tensor,
    max_len: int,
) -> torch.Tensor:
    """[L, D] jagged -> [B, max_len, D] padded with zeros."""
    B = offsets.shape[0] - 1
    D = values.shape[1]
    out = values.new_zeros(B, max_len, D)
    for i in range(B):
        s, e = int(offsets[i]), int(offsets[i + 1])
        n = min(e - s, max_len)
        out[i, :n] = values[s : s + n]
    return out


def _padded_to_jagged(
    padded: torch.Tensor,
    offsets: torch.Tensor,
    total_L: int,
) -> torch.Tensor:
    """[B, max_len, D] padded -> [L, D] jagged."""
    D = padded.shape[2]
    out = padded.new_zeros(total_L, D)
    B = offsets.shape[0] - 1
    for i in range(B):
        s, e = int(offsets[i]), int(offsets[i + 1])
        out[s:e] = padded[i, : e - s]
    return out


def _hstu_keep_mask(
    max_seq_len: int,
    seq_lengths: torch.Tensor,
    num_targets: torch.Tensor | None,
) -> torch.Tensor:
    """Build the per-sequence [B, N, N] boolean *keep* mask (True == attend).

    ``keep(m, n) = (m == n) or ((m >= n) and (n < c))`` with ``c = seq_len -
    num_targets`` (``c = N`` when ``num_targets`` is None, giving plain causal).
    """
    device = seq_lengths.device
    N = max_seq_len
    B = seq_lengths.size(0)
    ids = torch.arange(N, device=device)
    if num_targets is None:
        cap = seq_lengths.new_full((B, 1, 1), N)
    else:
        cap = (seq_lengths - num_targets).view(B, 1, 1)
    row = ids.view(1, N, 1)  # query position m
    col = ids.view(1, 1, N)  # key position n
    causal = row >= col
    n_is_history = col < cap
    diag = row == col
    return (causal & n_is_history) | diag


def reference_jagged_hstu_attention(
    max_seq_len: int,
    alpha: float,
    attn_scale: float | torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pad-then-compute reference matching the production HSTU pattern.

    Differentiable w.r.t. ``q``/``k``/``v`` so it can serve as the autograd
    ground truth for the backward kernels. ``attn_scale`` may be a Python float
    or a 0-d tensor (the recir ops pass a 0-d tensor); the multiply broadcasts.
    """
    L, H, D = q.shape

    padded_q = (
        _jagged_to_padded(q.reshape(L, H * D), seq_offsets, max_seq_len)
        .view(-1, max_seq_len, H, D)
        .transpose(1, 2)
    )  # [B, H, N, D]
    padded_k = (
        _jagged_to_padded(k.reshape(L, H * D), seq_offsets, max_seq_len)
        .view(-1, max_seq_len, H, D)
        .transpose(1, 2)
    )  # [B, H, N, D]
    padded_v = (
        _jagged_to_padded(v.reshape(L, H * D), seq_offsets, max_seq_len)
        .view(-1, max_seq_len, H, D)
        .transpose(1, 2)
    )  # [B, H, N, D]

    qk_attn = torch.einsum("bhxa,bhya->bhxy", padded_q, padded_k) * alpha
    qk_attn = F.silu(qk_attn) * attn_scale

    seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
    keep_mask = _hstu_keep_mask(max_seq_len, seq_lengths, num_targets)
    qk_attn = qk_attn * keep_mask.unsqueeze(1).to(qk_attn.dtype)

    attn_out = torch.einsum("bhxd,bhdv->bhxv", qk_attn.to(v.dtype), padded_v)
    return _padded_to_jagged(
        attn_out.transpose(1, 2).flatten(2, 3),
        seq_offsets,
        L,
    ).view(L, H, D)


def reference_jagged_hstu_attention_bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: torch.Tensor | None,
    attn_scale: float | torch.Tensor,
    max_seq_len: int,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Autograd backward through the padded reference: the (dq, dk, dv) oracle."""
    q_ = q.detach().clone().requires_grad_(True)
    k_ = k.detach().clone().requires_grad_(True)
    v_ = v.detach().clone().requires_grad_(True)
    out = reference_jagged_hstu_attention(
        max_seq_len, alpha, attn_scale, q_, k_, v_, seq_offsets, num_targets
    )
    out.backward(dout)
    assert q_.grad is not None and k_.grad is not None and v_.grad is not None
    return q_.grad, k_.grad, v_.grad


# ---------------------------------------------------------------------------
# Helion kernel implementations (scale-free; scales folded by the wrappers).
# Structure mirrors helion/examples/jagged_hstu_attn_2.py: forward + dQ are
# tiled q-outer/kv-inner; dK + dV use the transposed kv-outer/q-inner layout
# (compute K @ Q^T directly instead of transposing the score matrix).
#
# Jagged layout: q/k/v are ``[L, H, D]`` (all sequences packed on the token axis
# L) and ``seq_offsets`` is ``[B+1]``. The grid is over the B sequences; each
# program instance windows a single sequence's ``[start, end)`` token range via
# ``hl.tile(start, end)`` -- no per-sequence pad, so VMEM is bounded by the
# per-sequence width, not the whole L pack. Mask indices are global-frame within
# each window; the causal/cap comparison is frame-invariant, so a nonzero
# ``start`` is fine.
# ---------------------------------------------------------------------------
def _fwd_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: torch.Tensor,
) -> torch.Tensor:
    """Forward: out = (keep * silu(q @ k^T)) @ v, per sequence. Scales folded.

    Jagged ``[L, H, D]`` layout (grid over the B = ``seq_offsets.size(0) - 1``
    sequences). ``num_targets`` is always a tensor here (the wrapper passes zeros
    when there are no targets), so ``cap == end`` => the mask reduces to plain
    causal.
    """
    H = hl.specialize(q.size(1))
    D = hl.specialize(q.size(2))
    num_sequences = seq_offsets.size(0) - 1
    # empty_like, NOT zeros_like: on the ``jax_fn`` export path a zero-filled host
    # allocation is lifted into a plain torch constant, which
    # ``_tensor_arg_to_jax`` materializes via ``.numpy()`` -> "Got unsupported
    # ScalarType BFloat16". The tail past ``seq_offsets[-1]`` (the dropped PRT pad
    # row) is therefore UNINITIALIZED here and is zeroed by
    # ``_zero_unvisited_tail`` in the jax wrappers instead.
    out = torch.empty_like(v)

    for seq_idx in hl.grid(num_sequences):
        start = seq_offsets[seq_idx]
        end = seq_offsets[seq_idx + 1]
        cap = end - num_targets[seq_idx]

        for tile_q in hl.tile(start, end):
            # [tile_q, H, D] -> [H, tile_q, D] for batched matmul
            q_blk = q[tile_q, :, :].transpose(0, 1)
            acc = hl.zeros([H, tile_q, D], dtype=torch.float32)

            # Causal skip: every kv tile starting at/after tile_q.end has n > m
            # for all (m, n), so keep is False there. Clamp the inner loop to the
            # query tile's end to skip those tiles instead of computing then
            # masking them to zero (dependent Tile.end bound).
            for tile_kv in hl.tile(start, min(end, tile_q.end)):
                k_blk = k[tile_kv, :, :].transpose(0, 1)  # [H, tile_kv, D]
                v_blk = v[tile_kv, :, :].transpose(0, 1)  # [H, tile_kv, D]

                # alpha folded into q, attn_scale folded into v.
                scores = torch.bmm(q_blk, k_blk.transpose(-2, -1))  # [H, tq, tkv]
                scores = F.silu(scores)

                q_idx = tile_q.index.unsqueeze(1)
                kv_idx = tile_kv.index.unsqueeze(0)
                keep = ((q_idx >= kv_idx) & (kv_idx < cap)) | (q_idx == kv_idx)
                scores = torch.where(keep[None, :, :], scores, 0.0)

                acc = acc + torch.bmm(scores.to(v.dtype), v_blk)

            out[tile_q, :, :] = acc.transpose(0, 1).to(out.dtype)

    return out


def _bwd_dq_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: torch.Tensor,
) -> torch.Tensor:
    """dQ = (keep * silu'(q @ k^T) * (dO @ v^T)) @ k, per sequence.

    Jagged ``[L, H, D]`` layout (grid over B sequences; see ``_fwd_impl``). q is
    ORIGINAL, k is alpha-folded (so ``q @ (alpha k)^T = alpha q@k^T`` and ``dq = dS @
    (alpha k)`` returns w.r.t. the original q); dO is attn_scale-folded.
    """
    H = hl.specialize(q.size(1))
    D = hl.specialize(q.size(2))
    num_sequences = seq_offsets.size(0) - 1
    dq = torch.empty_like(q)  # tail zeroed by ``_zero_unvisited_tail``; see _fwd_impl.

    for seq_idx in hl.grid(num_sequences):
        start = seq_offsets[seq_idx]
        end = seq_offsets[seq_idx + 1]
        cap = end - num_targets[seq_idx]

        for tile_q in hl.tile(start, end):
            q_blk = q[tile_q, :, :].transpose(0, 1)  # [H, tile_q, D]
            do_blk = dout[tile_q, :, :].transpose(0, 1)  # [H, tile_q, D]
            acc_dq = hl.zeros([H, tile_q, D], dtype=torch.float32)

            # Causal skip: every kv tile starting at/after tile_q.end has n > m
            # for all (m, n), so keep is False there. Clamp the inner loop to the
            # query tile's end to skip those tiles instead of computing then
            # masking them to zero (dependent Tile.end bound).
            for tile_kv in hl.tile(start, min(end, tile_q.end)):
                k_blk = k[tile_kv, :, :].transpose(0, 1)  # [H, tile_kv, D]
                v_blk = v[tile_kv, :, :].transpose(0, 1)  # [H, tile_kv, D]

                # S = alpha * Q @ K^T ; dP = attn_scale * dO @ V^T  [H, tq, tkv]
                s = torch.bmm(q_blk, k_blk.transpose(-2, -1))
                dp = torch.bmm(do_blk, v_blk.transpose(-2, -1))

                # dS = dP * keep * silu'(S)  (attn_scale already folded into dP)
                sig = torch.sigmoid(s)
                silu_grad = sig * (1.0 + s * (1.0 - sig))
                ds = dp * silu_grad

                q_idx = tile_q.index.unsqueeze(1)
                kv_idx = tile_kv.index.unsqueeze(0)
                keep = ((q_idx >= kv_idx) & (kv_idx < cap)) | (q_idx == kv_idx)
                ds = torch.where(keep[None, :, :], ds, 0.0)

                # dQ += dS @ (alpha K)  [H, tq, tkv] @ [H, tkv, D]
                acc_dq = acc_dq + torch.bmm(ds.to(k.dtype), k_blk)

            dq[tile_q, :, :] = acc_dq.transpose(0, 1).to(dq.dtype)

    return dq


def _bwd_dk_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: torch.Tensor,
) -> torch.Tensor:
    """dK = (keep * silu'(q @ k^T) * (dO @ v^T))^T @ q, transposed layout.

    Jagged ``[L, H, D]`` layout (grid over B sequences; see ``_fwd_impl``). q is
    alpha-folded (so ``K @ (alpha Q)^T = alpha K@Q^T`` and ``dk = dS^T @ (alpha Q)``
    returns w.r.t. the original k); k/v ORIGINAL, dO attn_scale-folded.
    """
    H = hl.specialize(q.size(1))
    D = hl.specialize(q.size(2))
    num_sequences = seq_offsets.size(0) - 1
    dk = torch.empty_like(k)  # tail zeroed by ``_zero_unvisited_tail``; see _fwd_impl.

    for seq_idx in hl.grid(num_sequences):
        start = seq_offsets[seq_idx]
        end = seq_offsets[seq_idx + 1]
        cap = end - num_targets[seq_idx]

        for tile_kv in hl.tile(start, end):
            k_blk = k[tile_kv, :, :].transpose(0, 1)  # [H, tile_kv, D]
            v_blk = v[tile_kv, :, :].transpose(0, 1)  # [H, tile_kv, D]
            acc_dk = hl.zeros([H, tile_kv, D], dtype=torch.float32)

            for tile_q in hl.tile(start, end):
                q_blk = q[tile_q, :, :].transpose(0, 1)  # [H, tile_q, D]
                do_blk = dout[tile_q, :, :].transpose(0, 1)  # [H, tile_q, D]

                # S^T = alpha * K @ Q^T ; dP^T = V @ (attn_scale dO)^T  [H, tkv, tq]
                s_t = torch.bmm(k_blk, q_blk.transpose(-2, -1))
                dp_t = torch.bmm(v_blk, do_blk.transpose(-2, -1))

                sig = torch.sigmoid(s_t)
                silu_grad = sig * (1.0 + s_t * (1.0 - sig))
                ds_t = dp_t * silu_grad

                # keep_mask^T: row = kv (n), col = q (m); keep(m, n)
                n_idx = tile_kv.index.unsqueeze(1)
                m_idx = tile_q.index.unsqueeze(0)
                keep_t = ((m_idx >= n_idx) & (n_idx < cap)) | (m_idx == n_idx)
                ds_t = torch.where(keep_t[None, :, :], ds_t, 0.0)

                # dK += dS^T @ (alpha Q)  [H, tkv, tq] @ [H, tq, D]
                acc_dk = acc_dk + torch.bmm(ds_t.to(q.dtype), q_blk)

            dk[tile_kv, :, :] = acc_dk.transpose(0, 1).to(dk.dtype)

    return dk


def _bwd_dv_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    dout: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: torch.Tensor,
) -> torch.Tensor:
    """dV = (keep * silu(q @ k^T))^T @ dO, transposed layout. No v input.

    Jagged ``[L, H, D]`` layout (grid over B sequences; see ``_fwd_impl``). q is
    alpha-folded, dO is attn_scale-folded, so ``P^T = silu(alpha K@Q^T)`` masked and
    ``dv = P^T @ (attn_scale dO)`` returns w.r.t. the original v.
    """
    H = hl.specialize(q.size(1))
    D = hl.specialize(q.size(2))
    num_sequences = seq_offsets.size(0) - 1
    dv = torch.empty_like(k)  # tail zeroed by ``_zero_unvisited_tail``; see _fwd_impl.

    for seq_idx in hl.grid(num_sequences):
        start = seq_offsets[seq_idx]
        end = seq_offsets[seq_idx + 1]
        cap = end - num_targets[seq_idx]

        for tile_kv in hl.tile(start, end):
            k_blk = k[tile_kv, :, :].transpose(0, 1)  # [H, tile_kv, D]
            acc_dv = hl.zeros([H, tile_kv, D], dtype=torch.float32)

            for tile_q in hl.tile(start, end):
                q_blk = q[tile_q, :, :].transpose(0, 1)  # [H, tile_q, D]
                do_blk = dout[tile_q, :, :].transpose(0, 1)  # [H, tile_q, D]

                # P^T = silu(alpha * K @ Q^T)  [H, tkv, tq]  (attn_scale in dO)
                s_t = torch.bmm(k_blk, q_blk.transpose(-2, -1))
                p_t = F.silu(s_t)

                n_idx = tile_kv.index.unsqueeze(1)
                m_idx = tile_q.index.unsqueeze(0)
                keep_t = ((m_idx >= n_idx) & (n_idx < cap)) | (m_idx == n_idx)
                p_t = torch.where(keep_t[None, :, :], p_t, 0.0)

                # dV += P^T @ (attn_scale dO)  [H, tkv, tq] @ [H, tq, D]
                acc_dv = acc_dv + torch.bmm(p_t.to(dout.dtype), do_blk)

            dv[tile_kv, :, :] = acc_dv.transpose(0, 1).to(dv.dtype)

    return dv


def _bwd_dkv_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """dK and dV in one pass over the kv-outer/q-inner tile grid."""
    H = hl.specialize(q.size(1))
    D = hl.specialize(q.size(2))
    num_sequences = seq_offsets.size(0) - 1
    dk = torch.empty_like(k)  # tail zeroed by ``_zero_unvisited_tail``; see _fwd_impl.
    dv = torch.empty_like(k)

    for seq_idx in hl.grid(num_sequences):
        start = seq_offsets[seq_idx]
        end = seq_offsets[seq_idx + 1]
        cap = end - num_targets[seq_idx]

        for tile_kv in hl.tile(start, end):
            k_blk = k[tile_kv, :, :].transpose(0, 1)  # [H, tile_kv, D]
            v_blk = v[tile_kv, :, :].transpose(0, 1)  # [H, tile_kv, D]
            acc_dk = hl.zeros([H, tile_kv, D], dtype=torch.float32)
            acc_dv = hl.zeros([H, tile_kv, D], dtype=torch.float32)

            for tile_q in hl.tile(start, end):
                q_blk = q[tile_q, :, :].transpose(0, 1)  # [H, tile_q, D]
                do_blk = dout[tile_q, :, :].transpose(0, 1)  # [H, tile_q, D]

                # The one shared matmul: S^T = alpha * K @ Q^T  [H, tkv, tq]
                s_t = torch.bmm(k_blk, q_blk.transpose(-2, -1))

                n_idx = tile_kv.index.unsqueeze(1)
                m_idx = tile_q.index.unsqueeze(0)
                keep_t = ((m_idx >= n_idx) & (n_idx < cap)) | (m_idx == n_idx)

                # dV += P^T @ (attn_scale dO), P^T = silu(S^T)
                p_t = torch.where(keep_t[None, :, :], F.silu(s_t), 0.0)
                acc_dv = acc_dv + torch.bmm(p_t.to(dout.dtype), do_blk)

                # dK += dS^T @ (alpha Q), dS^T = (V @ dO_s^T) * silu'(S^T)
                dp_t = torch.bmm(v_blk, do_blk.transpose(-2, -1))
                sig = torch.sigmoid(s_t)
                silu_grad = sig * (1.0 + s_t * (1.0 - sig))
                ds_t = torch.where(keep_t[None, :, :], dp_t * silu_grad, 0.0)
                acc_dk = acc_dk + torch.bmm(ds_t.to(q.dtype), q_blk)

            dk[tile_kv, :, :] = acc_dk.transpose(0, 1).to(dk.dtype)
            dv[tile_kv, :, :] = acc_dv.transpose(0, 1).to(dv.dtype)

    return dk, dv


def _semilocal_fwd_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: torch.Tensor,
) -> torch.Tensor:
    """RankFM 256-local/256-full-tail forward, using two sparse KV ranges."""
    H = hl.specialize(q.size(1))
    D = hl.specialize(q.size(2))
    num_sequences = seq_offsets.size(0) - 1
    max_attn_len = RANKFM_MAX_ATTN_LEN
    full_attn_size = RANKFM_FULL_ATTN_SIZE
    out = torch.empty_like(v)

    for seq_idx in hl.grid(num_sequences):
        start = seq_offsets[seq_idx]
        end = seq_offsets[seq_idx + 1]
        history_end = end - num_targets[seq_idx]
        full_start = history_end - full_attn_size

        for tile_q in hl.tile(start, end):
            q_blk = q[tile_q, :, :].transpose(0, 1)
            acc = hl.zeros([H, tile_q, D], dtype=torch.float32)
            local_begin = max(start, tile_q.begin - max_attn_len)

            # A query tile overlapping the full-attention tail also needs the
            # older prefix excluded by its ordinary sliding-window range.
            # ``tile_q.end`` is exclusive, so equality is still before the tail.
            if tile_q.end > full_start:
                for tile_far in hl.tile(start, local_begin):
                    k_blk = k[tile_far, :, :].transpose(0, 1)
                    v_blk = v[tile_far, :, :].transpose(0, 1)
                    scores = F.silu(torch.bmm(q_blk, k_blk.transpose(-2, -1)))
                    keep = _semilocal_keep_mask(tile_q, tile_far, history_end)
                    scores = torch.where(keep[None, :, :], scores, 0.0)
                    acc = acc + torch.bmm(scores.to(v.dtype), v_blk)

            for tile_kv in hl.tile(local_begin, min(end, tile_q.end)):
                k_blk = k[tile_kv, :, :].transpose(0, 1)
                v_blk = v[tile_kv, :, :].transpose(0, 1)
                scores = F.silu(torch.bmm(q_blk, k_blk.transpose(-2, -1)))
                keep = _semilocal_keep_mask(tile_q, tile_kv, history_end)
                scores = torch.where(keep[None, :, :], scores, 0.0)
                acc = acc + torch.bmm(scores.to(v.dtype), v_blk)

            out[tile_q, :, :] = acc.transpose(0, 1).to(out.dtype)

    return out


def _semilocal_bwd_dq_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: torch.Tensor,
) -> torch.Tensor:
    """RankFM semi-local dQ, with the same two sparse KV ranges as forward."""
    H = hl.specialize(q.size(1))
    D = hl.specialize(q.size(2))
    num_sequences = seq_offsets.size(0) - 1
    max_attn_len = RANKFM_MAX_ATTN_LEN
    full_attn_size = RANKFM_FULL_ATTN_SIZE
    dq = torch.empty_like(q)

    for seq_idx in hl.grid(num_sequences):
        start = seq_offsets[seq_idx]
        end = seq_offsets[seq_idx + 1]
        history_end = end - num_targets[seq_idx]
        full_start = history_end - full_attn_size

        for tile_q in hl.tile(start, end):
            q_blk = q[tile_q, :, :].transpose(0, 1)
            do_blk = dout[tile_q, :, :].transpose(0, 1)
            acc_dq = hl.zeros([H, tile_q, D], dtype=torch.float32)
            local_begin = max(start, tile_q.begin - max_attn_len)

            # ``tile_q.end`` is exclusive, so equality is still before the
            # full-attention tail and the far-prefix mask would keep no entries.
            if tile_q.end > full_start:
                for tile_far in hl.tile(start, local_begin):
                    k_blk = k[tile_far, :, :].transpose(0, 1)
                    v_blk = v[tile_far, :, :].transpose(0, 1)
                    s = torch.bmm(q_blk, k_blk.transpose(-2, -1))
                    dp = torch.bmm(do_blk, v_blk.transpose(-2, -1))
                    sig = torch.sigmoid(s)
                    ds = dp * sig * (1.0 + s * (1.0 - sig))
                    keep = _semilocal_keep_mask(tile_q, tile_far, history_end)
                    ds = torch.where(keep[None, :, :], ds, 0.0)
                    acc_dq = acc_dq + torch.bmm(ds.to(k.dtype), k_blk)

            for tile_kv in hl.tile(local_begin, min(end, tile_q.end)):
                k_blk = k[tile_kv, :, :].transpose(0, 1)
                v_blk = v[tile_kv, :, :].transpose(0, 1)
                s = torch.bmm(q_blk, k_blk.transpose(-2, -1))
                dp = torch.bmm(do_blk, v_blk.transpose(-2, -1))
                sig = torch.sigmoid(s)
                ds = dp * sig * (1.0 + s * (1.0 - sig))
                keep = _semilocal_keep_mask(tile_q, tile_kv, history_end)
                ds = torch.where(keep[None, :, :], ds, 0.0)
                acc_dq = acc_dq + torch.bmm(ds.to(k.dtype), k_blk)

            dq[tile_q, :, :] = acc_dq.transpose(0, 1).to(dq.dtype)

    return dq


def _semilocal_bwd_dk_dv_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RankFM semi-local fused dK/dV with KV tiles as the single writers."""
    H = hl.specialize(q.size(1))
    D = hl.specialize(q.size(2))
    num_sequences = seq_offsets.size(0) - 1
    max_attn_len = RANKFM_MAX_ATTN_LEN
    full_attn_size = RANKFM_FULL_ATTN_SIZE
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    for seq_idx in hl.grid(num_sequences):
        start = seq_offsets[seq_idx]
        end = seq_offsets[seq_idx + 1]
        history_end = end - num_targets[seq_idx]
        full_start = history_end - full_attn_size

        for tile_kv in hl.tile(start, end):
            k_blk = k[tile_kv, :, :].transpose(0, 1)
            v_blk = v[tile_kv, :, :].transpose(0, 1)
            acc_dk = hl.zeros([H, tile_kv, D], dtype=torch.float32)
            acc_dv = hl.zeros([H, tile_kv, D], dtype=torch.float32)

            # Before the full tail, this key is visible only to its causal local
            # query window.  The exact mask handles the two boundary tiles.
            if tile_kv.begin < full_start:
                for tile_local_q in hl.tile(
                    max(start, tile_kv.begin),
                    min(full_start, tile_kv.end + max_attn_len),
                ):
                    q_blk = q[tile_local_q, :, :].transpose(0, 1)
                    do_blk = dout[tile_local_q, :, :].transpose(0, 1)
                    s_t = torch.bmm(k_blk, q_blk.transpose(-2, -1))
                    dp_t = torch.bmm(v_blk, do_blk.transpose(-2, -1))
                    sig = torch.sigmoid(s_t)
                    p_t = s_t * sig
                    ds_t = dp_t * sig * (1.0 + s_t * (1.0 - sig))
                    keep_t = _semilocal_keep_mask(
                        tile_local_q, tile_kv, history_end
                    ).transpose(0, 1)
                    p_t = torch.where(keep_t[None, :, :], p_t, 0.0)
                    ds_t = torch.where(keep_t[None, :, :], ds_t, 0.0)
                    acc_dk = acc_dk + torch.bmm(ds_t.to(q.dtype), q_blk)
                    acc_dv = acc_dv + torch.bmm(p_t.to(dout.dtype), do_blk)

            # Queries in the full-attention tail (including targets) can see all
            # history keys; target keys remain diagonal-only through the mask.
            for tile_full_q in hl.tile(max(full_start, tile_kv.begin), end):
                q_blk = q[tile_full_q, :, :].transpose(0, 1)
                do_blk = dout[tile_full_q, :, :].transpose(0, 1)
                s_t = torch.bmm(k_blk, q_blk.transpose(-2, -1))
                dp_t = torch.bmm(v_blk, do_blk.transpose(-2, -1))
                sig = torch.sigmoid(s_t)
                p_t = s_t * sig
                ds_t = dp_t * sig * (1.0 + s_t * (1.0 - sig))
                keep_t = _semilocal_keep_mask(
                    tile_full_q, tile_kv, history_end
                ).transpose(0, 1)
                p_t = torch.where(keep_t[None, :, :], p_t, 0.0)
                ds_t = torch.where(keep_t[None, :, :], ds_t, 0.0)
                acc_dk = acc_dk + torch.bmm(ds_t.to(q.dtype), q_blk)
                acc_dv = acc_dv + torch.bmm(p_t.to(dout.dtype), do_blk)

            dk[tile_kv, :, :] = acc_dk.transpose(0, 1).to(dk.dtype)
            dv[tile_kv, :, :] = acc_dv.transpose(0, 1).to(dv.dtype)

    return dk, dv


# ---------------------------------------------------------------------------
# Compiled kernels (one module-level object per kind).
# ---------------------------------------------------------------------------
# ``static_shapes=True`` keys a distinct BoundKernel + Pallas module per distinct
# input shape, so a single module-level kernel object compiles one module per
# distinct jagged ``L`` (L1/L2 differ in L => each compiles once, warmed per traced
# sig). This relies on the upstream fix (commit cf8985f871d9) that folds the spec key
# into the PyCodeCache module key for backends caching shape-specific state (Pallas);
# before it, the jax_fn export collapsed distinct widths onto the first-compiled
# module because the jagged kernel emits byte-identical source for every width.
def _make_kernel(
    impl: Callable[..., torch.Tensor | tuple[torch.Tensor, torch.Tensor]],
    *,
    config: helion.Config = PINNED_CONFIG,
) -> Any:
    return helion.kernel(
        backend="pallas",
        static_shapes=True,
        config=config,
    )(impl)


jagged_hstu_attention: Any = _make_kernel(_fwd_impl)
jagged_hstu_attention_bwd_dq: Any = _make_kernel(_bwd_dq_impl)
jagged_hstu_attention_bwd_dk: Any = _make_kernel(_bwd_dk_impl)
jagged_hstu_attention_bwd_dv: Any = _make_kernel(_bwd_dv_impl)
jagged_hstu_attention_bwd_dkv: Any = _make_kernel(_bwd_dkv_impl)
semilocal_hstu_attention: Any = _make_kernel(
    _semilocal_fwd_impl, config=SEMILOCAL_PINNED_CONFIG
)
semilocal_hstu_attention_bwd_dq: Any = _make_kernel(
    _semilocal_bwd_dq_impl, config=SEMILOCAL_PINNED_CONFIG
)
semilocal_hstu_attention_bwd_dk_dv: Any = _make_kernel(
    _semilocal_bwd_dk_dv_impl, config=SEMILOCAL_PINNED_CONFIG
)


# ---------------------------------------------------------------------------
# JAX-export wrappers (mirror the hammer TRITON op arg lists; fold the scales)
# ---------------------------------------------------------------------------
def _num_targets_or_zeros(num_targets: Any, batch: int, ref: Any) -> Any:
    """[B] int num_targets (dtype-matched to seq_offsets); zeros when None."""
    import jax.numpy as jnp

    if num_targets is None:
        return jnp.zeros((batch,), dtype=ref.dtype)
    return num_targets.astype(ref.dtype)


def _zero_unvisited_tail(x: Any, seq_offsets: Any) -> Any:
    """Zero the ``[seq_offsets[-1], L)`` token rows the kernel never wrote.

    The sharded wrapper drops the PRT pad row (see ``_drop_pad_rows``), so that
    row's token range falls outside the grid and the Pallas output buffer keeps
    whatever XLA left there -- arbitrary bits, which in bf16 can decode to NaN/Inf.
    A downstream padding mask cannot scrub that (``0 * NaN == NaN``), and on the
    backward it would scatter into the embedding gradient.

    Done here rather than allocating the kernel outputs with ``zeros_like`` because
    on the ``jax_fn`` export path a zero-filled host allocation is lifted into a
    plain ``torch.Tensor`` constant, which ``_tensor_arg_to_jax`` converts via
    ``.numpy()`` -> "Got unsupported ScalarType BFloat16".

    A no-op in EFFECT but not in COST on the single-device / warm / CPU-test paths,
    where the offsets already cover all of L.
    """
    import jax.numpy as jnp

    valid = jnp.arange(x.shape[0], dtype=seq_offsets.dtype) < seq_offsets[-1]
    return jnp.where(valid[:, None, None], x, 0)


def hstu_ragged_attention_fwd_jax_fn(
    N: int,
    alpha: float,
    q: Any,
    k: Any,
    v: Any,
    seq_offsets: Any,
    num_targets: Any,
    attn_scale: Any,
    max_attn_len: int = 0,
    full_attn_size: int = 0,
) -> tuple[Any, Any]:
    """Pure-JAX forward: fold scales, run the jagged kernel directly.

    Jagged layout: ``q``/``k``/``v`` are ``[L, H, D]`` (all sequences packed on the
    token axis L) and ``seq_offsets`` is ``[B+1]``. Runs the module-level jagged
    kernel (grid over B) at the inputs' own ``L`` -- no densify. Returns
    ``(out, dummy_M)`` mirroring ``hammer::triton_ragged_attention_fwd`` (M unused
    for SiLU). ``N`` is vestigial (kept for op-signature parity); ``alpha`` is a
    Python scalar; ``attn_scale`` a 0-d array or None; tensor args are JAX arrays (or
    tracers inside jax.jit).
    """
    import jax.numpy as jnp

    batch = int(seq_offsets.shape[0]) - 1

    q_scaled = q * alpha
    # attn_scale is often f32; ``v * attn_scale`` upcasts v to f32, changing the kernel
    # arg signature vs the (attn_scale=None) warm compile -> a BoundKernel cache miss
    # -> a nested recompile inside the jax lower tracer ("SymIntArrayRef ..."). Cast
    # back to v's dtype so warm and lower share ONE signature and the warm-compiled
    # kernel is reused.
    v_scaled = v if attn_scale is None else (v * attn_scale).astype(v.dtype)

    nt = _num_targets_or_zeros(num_targets, batch, seq_offsets)

    if use_semilocal_attention(max_attn_len, full_attn_size):
        out = semilocal_hstu_attention.jax_fn(q_scaled, k, v_scaled, seq_offsets, nt)
    else:
        out = jagged_hstu_attention.jax_fn(q_scaled, k, v_scaled, seq_offsets, nt)
    out = _zero_unvisited_tail(out, seq_offsets)
    dummy_M = jnp.zeros((1,), dtype=jnp.float32)
    return out, dummy_M


def hstu_ragged_attention_bwd_jax_fn(
    dout: Any,
    q: Any,
    k: Any,
    v: Any,
    seq_offsets: Any,
    num_targets: Any,
    attn_scale: Any,
    N: int,
    alpha: float,
    max_attn_len: int = 0,
    full_attn_size: int = 0,
) -> tuple[Any, Any, Any]:
    """Pure-JAX backward: fold scales, run the selected jagged kernels directly.

    Jagged ``[L, H, D]`` + ``[B+1]`` ``seq_offsets`` layout (see the forward).
    Returns ``(dq, dk, dv)`` w.r.t. the ORIGINAL q/k/v at the inputs' own ``L`` --
    no densify. ``dO_s = attn_scale * dO``; alpha is folded into k for the dQ kernel
    and into q for the dK/dV kernels so the grads come back unscaled. ``N`` is
    vestigial (kept for op-signature parity).
    """
    batch = int(seq_offsets.shape[0]) - 1

    # See the forward: keep dO in its input dtype so the attn_scale (f32) fold does
    # not change the kernel arg signature vs the (attn_scale=None) warm compile.
    dout_s = dout if attn_scale is None else (dout * attn_scale).astype(dout.dtype)
    q_f = q * alpha
    k_f = k * alpha
    nt = _num_targets_or_zeros(num_targets, batch, seq_offsets)

    # Both modes fuse dK/dV over their shared transposed query ranges.
    if use_semilocal_attention(max_attn_len, full_attn_size):
        dq = semilocal_hstu_attention_bwd_dq.jax_fn(q, k_f, v, dout_s, seq_offsets, nt)
        dk, dv = semilocal_hstu_attention_bwd_dk_dv.jax_fn(
            q_f, k, v, dout_s, seq_offsets, nt
        )
    else:
        dk, dv = jagged_hstu_attention_bwd_dkv.jax_fn(
            q_f, k, v, dout_s, seq_offsets, nt
        )
        dq = jagged_hstu_attention_bwd_dq.jax_fn(q, k_f, v, dout_s, seq_offsets, nt)
    return (
        _zero_unvisited_tail(dq, seq_offsets),
        _zero_unvisited_tail(dk, seq_offsets),
        _zero_unvisited_tail(dv, seq_offsets),
    )


# ---------------------------------------------------------------------------
# Sample-sharded multi-device wrappers: SHARD-LOCAL jagged run on the L axis.
# ---------------------------------------------------------------------------
# q/k/v arrive jagged and TOKEN-sharded (`[L, H, D]`, `in_specs=P(_AXIS, None,
# None)`) -- no input replication, no `psum`. Each device runs the jagged kernel on
# its own `[l_local, H, D]` block INSIDE the `shard_map`, so the op stays
# `Shard(0) -> Shard(0)` on L and GSPMD never sees the data-dependent per-sequence
# indexing. That is what keeps it from inserting an all-gather (fwd) / all-reduce +
# scatter-add (bwd) over the full `[B, S, H, D]` activation -- the ~65 s/step cost
# the earlier global-densify layout paid by densifying OUTSIDE the shard_map.
# `check_rep=False` because Mosaic custom calls cannot be GSPMD auto-partitioned.
#
# PRECONDITION -- BATCH ALIGNMENT (load-bearing, guarded by `_assert_batch_aligned`):
# each device's `l_local = L/world_size` token block must be EXACTLY its
# `b_local = B/world_size` WHOLE sequences. The caller's packing must satisfy this; if
# a future config breaks it the shard-local run SILENTLY mis-attends across device
# boundaries. `_assert_batch_aligned` makes that a hard assert on the CPU/test path
# (under jit the lengths are tracers and cannot be checked).
#
# Per device the wrapper rebuilds `local_offsets = concat([0], cumsum(lengths))` and
# runs the jagged kernel (grid over b_local) on the local pack. The kernel's mask
# uses global-frame indices within each `[start, end)` window, so nonzero local
# starts are fine, and the shard_map concatenates the per-device packs straight back
# into the global jagged `[L, H, D]` with no trailing reshard. Distinct `l_local`
# compile to distinct Pallas modules (see the `_make_kernel` note), which is why
# `warm_traced_hstu_kernels` warms one per traced `(l_local, H, D, b_local, dtype)`.
#
# PRT PAD ROWS -- dropped, not attended (`_PAD_SEQS_PER_DEVICE`):
# recir's PaddingRegionTransform gives each chip dummy rows past its real ones, and
# `pad_jagged_lengths` dumps that chip's ENTIRE token deficit into them -- real rows
# keep their real lengths, the dummies absorb `budget - sum(real)`. In a large
# recommendation model that can be one trailing row per device carrying many tokens against a ~4.1k
# per-sample budget, so the wrapper slices it off before building `local_offsets`.
#
# That drop is a COST optimization, not a correctness requirement: under
# `pallas_loop_type="emit_pipeline"` the kv tiles stream, so an oversized range is
# survivable -- but attention is quadratic, and one 75k pad row is ~5x the FLOPs of
# all 64 real rows combined (~84% of the op). It WOULD be a correctness requirement
# on the resident `unroll` path, whose compile-time-fixed VMEM window such a range
# over-reads.
#
# The pad rows are still load-bearing for batch alignment (they are what make each
# device's lengths sum to exactly `l_local`), so `_assert_batch_aligned` checks the
# FULL `[b_local]` group. It canNOT validate the pad-row COUNT: that group sums to
# `l_local` whatever the count is.
_AXIS = "torch_dist"

# Trailing PRT pad rows per device -- DERIVED at configure time, never assumed.
# `padded_bs//W - real_bs//W`, latched by `set_pad_seqs_per_device`. It is NOT a
# fixed 1: `compute_padded_shape` rounds to the PRT's OWN modulus `nd`, and
# a SparseCore-aware caller may pass `nd = lcm(num_ranks, sc_modulus * num_processes)`; with the
# JAX SparseCore backend's `sc_modulus = local_dev * num_sc` that is `W * num_sc`,
# so an already-`nd`-aligned batch gets `num_sc` pad rows per device, not one.
# Both failure directions are severe and neither is caught by
# `_assert_batch_aligned`: too low leaves a ~70-100k-token pad row in the grid
# (quadratic FLOPs -- it dominates the op), too high silently drops a real sequence.
# So there is no default -- reading it unset raises.
_PAD_SEQS_PER_DEVICE: int | None = None
# Padded global num_sequences the count above was derived for. Cross-checked
# against the traced sigs in `warm_traced_hstu_kernels` to catch a count derived
# for the wrong axis (the NRO candidate axis is the RO sample axis times
# num_candidates, so its pad-row count differs by the same factor).
_PADDED_NUM_SEQS: int | None = None


def set_pad_seqs_per_device(real_bs: int, padded_bs: int, world_size: int) -> int:
    """Derive and latch the per-device PRT pad-row count from the real/padded batch.

    Call POST-trace and PRE-``warm_traced_hstu_kernels``, from the site that knows
    both the real and padded global batch sizes.
    ``padded_bs == real_bs`` (padding disabled) correctly yields 0.

    ``real_bs`` must be ``world_size``-aligned: ``pad_jagged_lengths`` gives chip
    ``c`` ``tpb - rc_c`` dummy rows with ``rc_c = real_bs//W`` a FLOOR division, so
    an unaligned real batch makes the per-chip pad count non-uniform and the plain
    tail slice in ``_drop_pad_rows`` wrong.
    """
    global _PAD_SEQS_PER_DEVICE, _PADDED_NUM_SEQS
    assert world_size >= 1, f"world_size={world_size} must be >= 1"
    assert real_bs % world_size == 0 and padded_bs % world_size == 0, (
        f"jagged HSTU sample-shard: real_bs={real_bs} and padded_bs={padded_bs} "
        f"must both be divisible by world_size={world_size}, or the PRT pad rows "
        f"are not a uniform per-chip tail and cannot be sliced off."
    )
    b_local: int = padded_bs // world_size
    pad_seqs: int = b_local - real_bs // world_size
    assert 0 <= pad_seqs < b_local, (
        f"jagged HSTU sample-shard: derived {pad_seqs} pad row(s) per device from "
        f"real_bs={real_bs}, padded_bs={padded_bs}, world_size={world_size}; "
        f"expected 0 <= pad_seqs < b_local={b_local}."
    )
    _PAD_SEQS_PER_DEVICE = pad_seqs
    _PADDED_NUM_SEQS = padded_bs
    return pad_seqs


def _pad_seqs_per_device() -> int:
    """The latched per-device pad-row count; raises if it was never configured."""
    if _PAD_SEQS_PER_DEVICE is None:
        raise RuntimeError(
            "jagged HSTU sample-shard: the PRT pad-row count is not configured. "
            "Call set_pad_seqs_per_device(real_bs, padded_bs, world_size) after the "
            "trace and before the first lower. It has no default because the count "
            "follows the PRT's padding modulus (1 on the base backend, num_sc on the "
            "SparseCore backend, 0 with padding disabled), and guessing it either "
            "leaves a huge pad row dominating the op (too low) or silently drops a "
            "real sequence (too high)."
        )
    return _PAD_SEQS_PER_DEVICE


def clear_pad_seqs_per_device() -> None:
    """Un-latch the pad-row count (mirrors ``clear_traced_sigs``).

    Call between compiles whose batch/world-size differ, so a stale count cannot
    silently carry over into the next lower.
    """
    global _PAD_SEQS_PER_DEVICE, _PADDED_NUM_SEQS
    _PAD_SEQS_PER_DEVICE = None
    _PADDED_NUM_SEQS = None


# --- traced attention shape capture (for Helion kernel warmup) -----------
# The HSTU ops' register_fake (Meta) kernels are the only attention nodes present
# during recir's make_fx trace, and they run with CONCRETE FakeTensor shapes (GSPMD
# fake mode has no ShapeEnv). Recording each traced signature here lets the model
# path warm the Helion kernels for exactly those shapes AFTER the trace and BEFORE
# the first XLA lower, because Helion compile cannot nest inside the lower's tracer.
# Only (int, dtype) tuples are stored. N is max_seq_len (the per-sequence pad
# width), distinct from L = total ragged tokens; the two attention-window ints
# select the causal or RankFM semi-local kernel family.
_TracedSig = tuple[int, int, int, int, int, int, int, torch.dtype]
_TRACED_FWD_SIGS: set[_TracedSig] = set()
_TRACED_BWD_SIGS: set[_TracedSig] = set()


def record_traced_fwd_sig(
    sig: _TracedSig,
) -> None:
    """Record ``(L,H,D,B,N,max_attn,full_attn,dtype)`` from the trace."""
    _TRACED_FWD_SIGS.add(sig)


def record_traced_bwd_sig(
    sig: _TracedSig,
) -> None:
    """Record ``(L,H,D,B,N,max_attn,full_attn,dtype)`` from the trace."""
    _TRACED_BWD_SIGS.add(sig)


def get_traced_fwd_sigs() -> set[_TracedSig]:
    """Forward signatures recorded during the make_fx trace."""
    return set(_TRACED_FWD_SIGS)


def get_traced_bwd_sigs() -> set[_TracedSig]:
    """Backward signatures recorded during the make_fx trace."""
    return set(_TRACED_BWD_SIGS)


def clear_traced_sigs() -> None:
    """Drop all recorded sigs (call before a compile so only its shapes warm)."""
    _TRACED_FWD_SIGS.clear()
    _TRACED_BWD_SIGS.clear()


def _assert_batch_aligned(seq_lengths: Any, world_size: int, l_local: int) -> None:
    """Guarantee the packing is BATCH-ALIGNED: each device's ``l_local`` token block
    is exactly its ``b_local = B/world_size`` WHOLE sequences.

    The shard-local run splits the jagged token axis into even ``l_local`` blocks and
    ``seq_lengths`` into ``b_local`` groups, then rebuilds each device's offsets from
    its own group. That is only correct if every device boundary is also a SEQUENCE
    boundary -- otherwise the local offsets index past the device's token block and
    the attention SILENTLY mis-attends across it.

    Checked on the FULL ``[b_local]`` group, PRT pad row included: that equality is
    exactly what makes dropping the pad row safe. Concrete/eager (CPU test) path only
    -- under ``jax.jit`` ``seq_lengths`` is a tracer with no readable values.
    """
    import jax

    if isinstance(seq_lengths, jax.core.Tracer):
        return
    b_local: int = seq_lengths.shape[0] // world_size
    group_sums = seq_lengths.reshape(world_size, b_local).sum(axis=1)
    sums = [int(x) for x in group_sums]
    assert all(s == l_local for s in sums), (
        f"jagged HSTU sample-shard: per-device token counts {sums} are not all "
        f"equal to l_local={l_local}; the packing is not batch-aligned, so the "
        f"shard-local run would mis-attend across device boundaries. Ensure "
        f"each device's L//world_size block is exactly its b_local whole sequences."
    )


def _drop_pad_rows(seq_lengths: Any, num_targets: Any) -> tuple[Any, Any]:
    """Slice the trailing PRT pad row(s) off one device's ``[b_local]`` arrays.

    Called INSIDE the ``shard_map``, where ``seq_lengths``/``num_targets`` are the
    device's own ``[b_local]`` block. ``pad_jagged_lengths`` lays each key out as
    ``[chip0: real... | pad][chip1: real... | pad]...`` and the ``P(_AXIS)`` shard
    hands device ``d`` exactly chip ``d``'s block, so the pad rows are always the
    TAIL of what each device sees -- a plain slice is enough. Both arrays are
    sliced so the kernel's ``num_targets[seq_idx]`` stays aligned with
    ``seq_offsets[seq_idx]`` and both shapes match what ``warm_traced_hstu_kernels``
    pre-compiles (``static_shapes=True`` keys on shape; a mismatch would recompile
    nested inside the jax lower tracer).

    The row COUNT is the derived ``_PAD_SEQS_PER_DEVICE`` (see
    ``set_pad_seqs_per_device``), read here rather than assumed -- it must be a
    static Python int because this runs inside the ``shard_map``.
    """
    pad_seqs: int = _pad_seqs_per_device()
    keep: int = int(seq_lengths.shape[0]) - pad_seqs
    assert keep > 0, (
        f"jagged HSTU sample-shard: b_local={int(seq_lengths.shape[0])} leaves no "
        f"real sequence after dropping {pad_seqs} PRT pad row(s)."
    )
    return seq_lengths[:keep], num_targets[:keep]


def sample_sharded_hstu_fwd_jax(
    N: int,
    alpha: float,
    q: Any,
    k: Any,
    v: Any,
    seq_offsets: Any,
    num_targets: Any,
    attn_scale: Any,
    world_size: int,
    mesh: Any,
    max_attn_len: int = 0,
    full_attn_size: int = 0,
    run_kernel: Callable[..., Any] | None = None,
) -> Any:
    """Shard-local sample-sharded forward (see the section comment above).

    ``run_kernel(q, k, v, seq_offsets, num_targets, attn_scale) -> out`` runs the
    jagged per-device forward on ``[l_local, H, D]`` inputs with local ``[b_local+1]``
    offsets. Defaults to ``hstu_ragged_attention_fwd_jax_fn`` (the CPU/unit-test
    path); the torchax lowering passes a variant that runs the kernel inside its
    clean tensor env.

    ``seq_offsets`` (``[B+1]``) and ``num_targets`` (``[B]``) must be concrete arrays
    and ``attn_scale`` a concrete scalar or None -- the caller normalizes. Returns the
    ``[L, H, D]`` output, already L-sharded. ``N`` is vestigial (op-signature parity).
    """
    import jax.numpy as jnp
    from jax.experimental.shard_map import shard_map
    from jax.sharding import PartitionSpec as P

    def _default_runner(q_: Any, k_: Any, v_: Any, so_: Any, nt_: Any, sc_: Any) -> Any:
        out, _m = hstu_ragged_attention_fwd_jax_fn(
            N,
            alpha,
            q_,
            k_,
            v_,
            so_,
            nt_,
            sc_,
            max_attn_len,
            full_attn_size,
        )
        return out

    runner: Callable[..., Any] = _default_runner if run_kernel is None else run_kernel

    total_len: int = int(q.shape[0])
    l_local: int = total_len // world_size  # per-device token block (static).

    # Derive global per-device seq_lengths ([B]); each device's b_local-group must sum
    # to l_local (batch alignment) or the shard-local split mis-attends -- guard it.
    seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
    _assert_batch_aligned(seq_lengths, world_size, l_local)

    def _call(q_: Any, k_: Any, v_: Any, len_: Any, nt_: Any, sc_: Any) -> Any:
        # Per device: jagged [l_local, H, D] pack + [b_local] lengths/targets. Drop
        # the trailing PRT pad row(s), then rebuild local offsets ([b_run+1], max <=
        # l_local) and run the jagged kernel over the REAL sequences only (grid over
        # b_run). The pad row's token range stays unvisited, and the kernels allocate
        # with ``empty_like``, so that tail comes back UNINITIALIZED -- the runner
        # (``hstu_ragged_attention_fwd_jax_fn``) scrubs it via ``_zero_unvisited_tail``.
        # Do not "simplify" the kernels to ``zeros_like`` instead; that breaks the
        # bf16 jax_fn export (see the ``_fwd_impl`` / ``_zero_unvisited_tail`` notes).
        len_, nt_ = _drop_pad_rows(len_, nt_)
        local_offsets = jnp.concatenate(
            [jnp.zeros(1, dtype=len_.dtype), jnp.cumsum(len_)]
        )
        return runner(q_, k_, v_, local_offsets, nt_, sc_)

    out = shard_map(
        _call,
        mesh=mesh,
        in_specs=(
            P(_AXIS, None, None),
            P(_AXIS, None, None),
            P(_AXIS, None, None),
            P(_AXIS),
            P(_AXIS),
            P(),
        ),
        out_specs=P(_AXIS, None, None),
        check_rep=False,
    )(q, k, v, seq_lengths, num_targets, attn_scale)

    # Already jagged + L-sharded (per-device [l_local, H, D] packs concatenated).
    return out


def sample_sharded_hstu_bwd_jax(
    grad_out: Any,
    q: Any,
    k: Any,
    v: Any,
    seq_offsets: Any,
    num_targets: Any,
    attn_scale: Any,
    N: int,
    alpha: float,
    world_size: int,
    mesh: Any,
    max_attn_len: int = 0,
    full_attn_size: int = 0,
    run_kernel: Callable[..., Any] | None = None,
) -> tuple[Any, Any, Any]:
    """Shard-local sample-sharded backward (see the section comment above).

    Same structure as the forward, returning three L-sharded ``[L, H, D]`` grads.
    ``run_kernel(grad_out, q, k, v, seq_offsets, num_targets, attn_scale) ->
    (dq, dk, dv)`` defaults to ``hstu_ragged_attention_bwd_jax_fn`` (the CPU/unit-test
    path); the torchax lowering passes a clean-tensor-env variant. Same
    input-concreteness rules as the forward. ``N`` is vestigial.
    """
    import jax.numpy as jnp
    from jax.experimental.shard_map import shard_map
    from jax.sharding import PartitionSpec as P

    def _default_runner(
        do_: Any, q_: Any, k_: Any, v_: Any, so_: Any, nt_: Any, sc_: Any
    ) -> tuple[Any, Any, Any]:
        return hstu_ragged_attention_bwd_jax_fn(
            do_,
            q_,
            k_,
            v_,
            so_,
            nt_,
            sc_,
            N,
            alpha,
            max_attn_len,
            full_attn_size,
        )

    runner: Callable[..., Any] = _default_runner if run_kernel is None else run_kernel

    total_len: int = int(q.shape[0])
    l_local: int = total_len // world_size  # per-device token block (static).

    # Derive global per-device seq_lengths ([B]); each device's b_local-group must sum
    # to l_local (batch alignment) or the shard-local split mis-attends -- guard it.
    seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
    _assert_batch_aligned(seq_lengths, world_size, l_local)

    def _call(
        do_: Any, q_: Any, k_: Any, v_: Any, len_: Any, nt_: Any, sc_: Any
    ) -> tuple[Any, Any, Any]:
        # Per device: jagged [l_local, H, D] packs + [b_local] lengths/targets. Same
        # PRT pad-row drop as the forward (see ``_drop_pad_rows``) so fwd and bwd run
        # the identical sequence set. The pad row's slice of dq/dk/dv is left
        # UNINITIALIZED by the kernels (``empty_like``) and zeroed by the runner's
        # ``_zero_unvisited_tail`` -- see the forward ``_call`` for why not zeros_like.
        len_, nt_ = _drop_pad_rows(len_, nt_)
        local_offsets = jnp.concatenate(
            [jnp.zeros(1, dtype=len_.dtype), jnp.cumsum(len_)]
        )
        return runner(do_, q_, k_, v_, local_offsets, nt_, sc_)

    dq, dk, dv = shard_map(
        _call,
        mesh=mesh,
        in_specs=(
            P(_AXIS, None, None),
            P(_AXIS, None, None),
            P(_AXIS, None, None),
            P(_AXIS, None, None),
            P(_AXIS),
            P(_AXIS),
            P(),
        ),
        out_specs=(
            P(_AXIS, None, None),
            P(_AXIS, None, None),
            P(_AXIS, None, None),
        ),
        check_rep=False,
    )(grad_out, q, k, v, seq_lengths, num_targets, attn_scale)

    # Already jagged + L-sharded (per-device [l_local, H, D] packs concatenated).
    return dq, dk, dv


# ---------------------------------------------------------------------------
# Warmup helpers (compile the kernels OUTSIDE torchax, before recir traces)
# ---------------------------------------------------------------------------
# Helion kernel COMPILATION cannot run nested inside torchax: the torchax tensor
# modes leak symbolic sizes into Helion's static-shape type propagation
# ("SymIntArrayRef expected to contain only concrete integers"). EXECUTION of an
# already-compiled kernel works (the torchax lowering escapes via
# mode_utils.no_dispatch). Call these once, before recir traces/compiles, with
# tensors whose [L, H, D] / seq_offsets shapes MATCH what the compiled path will
# pass; the torchax-nested call then hits the cache and only runs.
def _torch_dtype_to_jnp(dtype: torch.dtype) -> Any:
    """Map the torch compute dtype to the matching jnp dtype (fp32 fallback).

    Lets the warm path reproduce the compiled kernel's dtype specialization:
    the fp32 default run and the bf16 bench yamls compile DIFFERENT ``jax_fn``
    caches, so warming with the wrong dtype would leave the real dtype to compile
    nested in the lower (the crash this fix avoids)."""
    import jax.numpy as jnp

    if dtype == torch.bfloat16:
        return jnp.bfloat16
    if dtype == torch.float16:
        return jnp.float16
    return jnp.float32


def _to_jax(t: torch.Tensor, dtype: torch.dtype | None = None) -> Any:
    """torch -> jax array. ``dtype`` (torch) selects the output jnp dtype; None
    keeps the legacy fp32 conversion. numpy has no bf16, so the host copy always
    routes through fp32 and then casts to the target jnp dtype."""
    import jax.numpy as jnp

    arr = jnp.asarray(t.detach().to(torch.float32).cpu().numpy())
    return arr if dtype is None else arr.astype(_torch_dtype_to_jnp(dtype))


def _warm_mesh() -> Any:
    """One-local-device mesh named ``_AXIS`` so the warm runs the SHARDED wrapper.

    The warm has to exercise the runtime CALL SITE, not just the kernels. Calling
    ``hstu_ragged_attention_*_jax_fn`` directly skips everything the real step does
    around the Mosaic call: the PRT pad-row drop, the rebuilt ``local_offsets``
    (which end SHORT of ``l_local``), and the ``shard_map`` itself -- inside which
    a scoped-VMEM request is rejected outright rather than clamped down with a
    warning (see the ``vmem_limit_bytes`` note in
    helion/runtime/pallas/launcher.py). ``world_size=1`` over this process's own
    device keeps the warm process-local exactly like the direct call it replaces,
    and the per-device kernel shapes are identical either way.
    """
    import jax
    import numpy as np
    from jax.sharding import Mesh

    return Mesh(np.asarray(jax.local_devices()[:1]), (_AXIS,))


def _warm_num_targets(num_targets: torch.Tensor | None, batch: int) -> Any:
    """``[b_local]`` int32 targets; zeros when None. The sharded wrapper slices
    this alongside the lengths, so unlike the direct call it cannot take None."""
    import jax.numpy as jnp

    if num_targets is None:
        return jnp.zeros((batch,), dtype=jnp.int32)
    return _to_jax(num_targets).astype("int32")


def warmup_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: torch.Tensor | None = None,
    dtype: torch.dtype | None = None,
    max_attn_len: int = 0,
    full_attn_size: int = 0,
) -> None:
    """Pre-compile + run the forward the way the compiled step runs it.

    Jagged q/k/v are ``[l_local, H, D]``. ``seq_offsets`` ``[b_local+1]`` and
    ``num_targets`` ``[b_local]`` are the FULL per-device group INCLUDING the PRT
    pad row(s) -- ``sample_sharded_hstu_fwd_jax`` drops them itself, so the kernel
    sees the same ``[b_run+1]`` local offsets (ending short of ``l_local``) the
    runtime step passes. ``dtype`` (torch) selects the q/k/v JAX dtype so the warm
    compiles the dtype specialization (bf16 vs fp32) the compiled path will run;
    None keeps the legacy fp32 conversion.
    """
    import jax

    jax.block_until_ready(
        sample_sharded_hstu_fwd_jax(
            0,
            1.0,
            _to_jax(q, dtype),
            _to_jax(k, dtype),
            _to_jax(v, dtype),
            jax.numpy.asarray(seq_offsets.detach().cpu().numpy()),
            _warm_num_targets(num_targets, int(seq_offsets.shape[0]) - 1),
            None,
            1,
            _warm_mesh(),
            max_attn_len=max_attn_len,
            full_attn_size=full_attn_size,
        )
    )


def warmup_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: torch.Tensor | None = None,
    dtype: torch.dtype | None = None,
    max_attn_len: int = 0,
    full_attn_size: int = 0,
) -> None:
    """Pre-compile + run the backward kernels the way the step runs them.

    Same FULL-group ``seq_offsets``/``num_targets`` contract as ``warmup_fwd``.
    """
    import jax

    jax.block_until_ready(
        sample_sharded_hstu_bwd_jax(
            _to_jax(v, dtype),
            _to_jax(q, dtype),
            _to_jax(k, dtype),
            _to_jax(v, dtype),
            jax.numpy.asarray(seq_offsets.detach().cpu().numpy()),
            _warm_num_targets(num_targets, int(seq_offsets.shape[0]) - 1),
            None,
            0,
            1.0,
            1,
            _warm_mesh(),
            max_attn_len=max_attn_len,
            full_attn_size=full_attn_size,
        )
    )


def warm_traced_hstu_kernels(world_size: int) -> None:
    """Warm the fwd + bwd kernels for the per-device shapes the sample-sharded path
    will run, from the sigs the op meta kernels recorded during the make_fx trace.

    Call AFTER the eager trace (so the sigs exist) and BEFORE the first
    ``jitted.lower()`` -- Helion compile cannot nest inside the lower's tracer
    ("SymIntArrayRef expected to contain only concrete integers"). Requires
    ``set_pad_seqs_per_device`` to have run, since the warm must synthesize the POST
    pad-row-drop shapes.

    Warms one kernel per distinct
    ``(l_local, H, D, b_local, max_attn_len, full_attn_size, dtype)`` (for udd_fm,
    semi-local L1 + causal L2). ``N`` stays out of the dedup key but its max is used
    as the per-sample cap, so each warm goes through
    ``sample_sharded_hstu_*_jax`` on a one-device mesh and exercises the runtime
    CALL SITE -- pad-row drop, ``local_offsets`` ending short of ``l_local``,
    Mosaic call inside a ``shard_map`` -- not just the shapes.

    Each warm is best-effort: a failure logs and leaves that shape to compile lazily
    rather than hard-failing the build.
    """
    ws: int = max(1, world_size)
    pad_seqs: int = _pad_seqs_per_device()

    def _synth(
        l_local: int, H: int, D: int, b_local: int, n_max: int, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Zero q/k/v [l_local, H, D] (the jagged per-device pack) plus the FULL
        # [b_local] device group the sharded wrapper consumes: b_run real rows at
        # the traced per-sample max `n_max`, and the trailing PRT pad row(s)
        # absorbing the deficit -- exactly how `pad_jagged_lengths` lays a device
        # out. The wrapper drops the pad row(s) itself, so the kernel gets the
        # runtime SHAPE ([b_run+1] offsets, [b_run] targets -- static_shapes keys
        # on shape, and a miss forces a recompile nested inside the jax lower
        # tracer, "SymIntArrayRef ...") AND the runtime LAYOUT: offsets that end
        # SHORT of l_local. The old even partition covered the whole pack, which
        # the real step never does.
        b_run = b_local - pad_seqs
        assert b_run >= 1, (
            f"b_local={b_local} leaves no real sequence after dropping "
            f"{pad_seqs} PRT pad row(s); nothing to warm."
        )
        qkv = torch.zeros(l_local, H, D, dtype=dtype)
        real = min(n_max, l_local // b_run) if n_max > 0 else l_local // b_run
        lengths = torch.zeros(b_local, dtype=torch.int32)
        lengths[:b_run] = real
        # Whole deficit on the LAST pad row (pad_jagged_lengths dumps it there);
        # any earlier pad rows stay empty. Keeps the group summing to exactly
        # l_local, which `_assert_batch_aligned` requires.
        lengths[-1] = l_local - int(lengths[:-1].sum())
        seq_offsets = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32),
                torch.cumsum(lengths, 0).to(torch.int32),
            ]
        )
        num_targets = torch.zeros(b_local, dtype=torch.int32)
        return qkv, seq_offsets, num_targets

    def _distinct_shapes(
        sigs: set[_TracedSig],
    ) -> list[_TracedSig]:
        # One warm per distinct shape, dtype, and attention mode. static_shapes keys
        # on shape, while the mode selects a different module-level kernel. N stays
        # out of the dedup key (the jagged kernel runs at its own L) but its MAX is
        # carried through as the synthesized real-row cap. str(dtype) keeps the sort
        # key orderable.
        shapes: dict[tuple[int, int, int, int, int, int, str], _TracedSig] = {}
        for L, H, D, B, N, max_attn_len, full_attn_size, dtype in sigs:
            assert B == _PADDED_NUM_SEQS, (
                f"jagged HSTU sample-shard: traced num_sequences B={B} != the "
                f"padded_bs={_PADDED_NUM_SEQS} that set_pad_seqs_per_device() "
                f"derived {pad_seqs} pad row(s)/device from. The count belongs to a "
                f"different axis (the NRO candidate axis is the RO sample axis times "
                f"num_candidates), so it would drop the wrong number of rows."
            )
            l_local = L // ws
            b_local = B // ws
            key = (
                l_local,
                H,
                D,
                b_local,
                max_attn_len,
                full_attn_size,
                str(dtype),
            )
            prev = shapes.get(key)
            shapes[key] = (
                l_local,
                H,
                D,
                b_local,
                N if prev is None else max(N, prev[4]),
                max_attn_len,
                full_attn_size,
                dtype,
            )
        return [shapes[key] for key in sorted(shapes)]

    for (
        l_local,
        H,
        D,
        b_local,
        n_max,
        max_attn_len,
        full_attn_size,
        dtype,
    ) in _distinct_shapes(get_traced_fwd_sigs()):
        try:
            qkv, seq_offsets, num_targets = _synth(l_local, H, D, b_local, n_max, dtype)
            warmup_fwd(
                qkv,
                qkv,
                qkv,
                seq_offsets,
                num_targets,
                dtype=dtype,
                max_attn_len=max_attn_len,
                full_attn_size=full_attn_size,
            )
            logger.info(
                "[HELION] warmed fwd kernel for per-device shape "
                f"(l_local={l_local}, H={H}, D={D}, num_sequences={b_local}, "
                f"max_seq_len={n_max}, max_attn_len={max_attn_len}, "
                f"full_attn_size={full_attn_size}, dtype={dtype})"
            )
        except Exception as e:
            logger.error(
                "[HELION] fwd warm FAILED for per-device shape "
                f"(l_local={l_local}, H={H}, D={D}, num_sequences={b_local}, "
                f"max_seq_len={n_max}, max_attn_len={max_attn_len}, "
                f"full_attn_size={full_attn_size}, dtype={dtype}): {e}\n"
                "  On the divide-on-B sharded (TPU) path this is effectively FATAL: "
                "the kernel will instead compile lazily INSIDE the jax lower tracer, "
                "which fails with 'SymIntArrayRef expected to contain only concrete "
                "integers'. If the cause above is CompileTimeScopedVmemOom, raise "
                "--xla_tpu_scoped_vmem_limit_kib (per-sequence kernel VMEM slightly "
                "exceeds the scoped cap)."
            )

    for (
        l_local,
        H,
        D,
        b_local,
        n_max,
        max_attn_len,
        full_attn_size,
        dtype,
    ) in _distinct_shapes(get_traced_bwd_sigs()):
        try:
            qkv, seq_offsets, num_targets = _synth(l_local, H, D, b_local, n_max, dtype)
            warmup_bwd(
                qkv,
                qkv,
                qkv,
                seq_offsets,
                num_targets,
                dtype=dtype,
                max_attn_len=max_attn_len,
                full_attn_size=full_attn_size,
            )
            logger.info(
                "[HELION] warmed bwd kernels for per-device shape "
                f"(l_local={l_local}, H={H}, D={D}, num_sequences={b_local}, "
                f"max_seq_len={n_max}, max_attn_len={max_attn_len}, "
                f"full_attn_size={full_attn_size}, dtype={dtype})"
            )
        except Exception as e:
            logger.error(
                "[HELION] bwd warm FAILED for per-device shape "
                f"(l_local={l_local}, H={H}, D={D}, num_sequences={b_local}, "
                f"max_seq_len={n_max}, max_attn_len={max_attn_len}, "
                f"full_attn_size={full_attn_size}, dtype={dtype}): {e}\n"
                "  On the divide-on-B sharded (TPU) path this is effectively FATAL: "
                "the kernel will instead compile lazily INSIDE the jax lower tracer, "
                "which fails with 'SymIntArrayRef expected to contain only concrete "
                "integers'. If the cause above is CompileTimeScopedVmemOom, raise "
                "--xla_tpu_scoped_vmem_limit_kib (per-sequence kernel VMEM slightly "
                "exceeds the scoped cap)."
            )
