# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Opaque packed-input Pallas HSTU operations and Hammer adapter."""

from __future__ import annotations

import torch
from recir.ops.hstu_attention_op import _hstu_attention_ref

_LIB = torch.library.Library("recir", "FRAGMENT")
_FWD_SCHEMA = (
    "pallas_jagged_hstu_attention_fwd(Tensor q, Tensor k, Tensor v, "
    "Tensor seq_offsets, Tensor num_targets, Tensor attn_scale, float alpha, "
    "int max_seq_len) -> Tensor"
)
_BWD_SCHEMA = (
    "pallas_jagged_hstu_attention_bwd(Tensor grad_out, Tensor q, Tensor k, "
    "Tensor v, Tensor seq_offsets, Tensor num_targets, Tensor attn_scale, "
    "float alpha, int max_seq_len) -> (Tensor, Tensor, Tensor)"
)
_LIB.define(_FWD_SCHEMA)
_LIB.define(_BWD_SCHEMA)


def _pad_packed(
    x: torch.Tensor,
    seq_offsets: torch.Tensor,
    max_seq_len: int,
) -> torch.Tensor:
    total_length, num_heads, dim = x.shape
    return (
        torch.ops.fbgemm.jagged_to_padded_dense(
            x.reshape(total_length, num_heads * dim),
            [seq_offsets],
            [max_seq_len],
            0.0,
        )
        .view(-1, max_seq_len, num_heads, dim)
        .transpose(1, 2)
    )


def _pack_dense(
    x: torch.Tensor,
    seq_offsets: torch.Tensor,
    total_length: int,
) -> torch.Tensor:
    batch, num_heads, _max_seq_len, dim = x.shape
    x_2d = x.transpose(1, 2).reshape(batch, -1, num_heads * dim)
    return torch.ops.fbgemm.dense_to_jagged(
        x_2d,
        [seq_offsets],
        total_length,
    )[0].view(total_length, num_heads, dim)


def _fwd_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: torch.Tensor,
    attn_scale: torch.Tensor,
    alpha: float,
    max_seq_len: int,
) -> torch.Tensor:
    q_dense = _pad_packed(q, seq_offsets, max_seq_len)
    k_dense = _pad_packed(k, seq_offsets, max_seq_len)
    v_dense = _pad_packed(v, seq_offsets, max_seq_len)
    lengths = (seq_offsets[1:] - seq_offsets[:-1]).to(torch.int32)
    dense_out = (
        _hstu_attention_ref(
            q_dense,
            k_dense,
            v_dense,
            lengths,
            num_targets.to(torch.int32),
            alpha,
            1.0,
        )
        * attn_scale
    )
    return _pack_dense(dense_out, seq_offsets, v.shape[0])


def _bwd_fallback(
    grad_out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: torch.Tensor,
    attn_scale: torch.Tensor,
    alpha: float,
    max_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_grad = q.detach().requires_grad_(True)
    k_grad = k.detach().requires_grad_(True)
    v_grad = v.detach().requires_grad_(True)
    with torch.enable_grad():
        output = _fwd_fallback(
            q_grad,
            k_grad,
            v_grad,
            seq_offsets,
            num_targets,
            attn_scale,
            alpha,
            max_seq_len,
        )
        dq, dk, dv = torch.autograd.grad(
            output,
            (q_grad, k_grad, v_grad),
            grad_out,
        )
    return dq, dk, dv


_LIB.impl(
    "pallas_jagged_hstu_attention_fwd",
    _fwd_fallback,
    "CompositeExplicitAutograd",
)
_LIB.impl(
    "pallas_jagged_hstu_attention_bwd",
    _bwd_fallback,
    "CompositeExplicitAutograd",
)


@torch.library.register_fake("recir::pallas_jagged_hstu_attention_fwd")
def _fwd_meta(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: torch.Tensor,
    attn_scale: torch.Tensor,
    alpha: float,
    max_seq_len: int,
) -> torch.Tensor:
    return torch.empty_like(v)


@torch.library.register_fake("recir::pallas_jagged_hstu_attention_bwd")
def _bwd_meta(
    grad_out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: torch.Tensor,
    attn_scale: torch.Tensor,
    alpha: float,
    max_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


# pyre-ignore[2]: variadic context and input contract from torch.library.
def _setup_context(ctx, inputs, output) -> None:  # noqa
    q, k, v, seq_offsets, num_targets, attn_scale, alpha, max_seq_len = inputs
    ctx.save_for_backward(q, k, v, seq_offsets, num_targets, attn_scale)
    ctx.alpha = alpha
    ctx.max_seq_len = max_seq_len


# pyre-ignore[2,3]: variadic context and gradient contract from torch.library.
def _backward(ctx, grad_out):  # noqa
    q, k, v, seq_offsets, num_targets, attn_scale = ctx.saved_tensors
    dq, dk, dv = torch.ops.recir.pallas_jagged_hstu_attention_bwd(
        grad_out,
        q,
        k,
        v,
        seq_offsets,
        num_targets,
        attn_scale,
        ctx.alpha,
        ctx.max_seq_len,
    )
    return dq, dk, dv, None, None, None, None, None


torch.library.register_autograd(
    "recir::pallas_jagged_hstu_attention_fwd",
    _backward,
    setup_context=_setup_context,
)


def _recir_pallas_jagged_hstu_mha(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: float,
    N: int,
    seq_offsets: torch.Tensor,
    invalid_attn_mask_type: str,
    dropout_pr: float,
    training: bool,
    num_targets: torch.Tensor | None = None,
    attn_bias: torch.Tensor | None = None,
    attn_scale: torch.Tensor | None = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
    full_attn_size: int = 0,
    num_softmax_heads: int = 0,
) -> torch.Tensor:
    """Hammer-compatible adapter for direct packed-input Pallas attention."""
    if invalid_attn_mask_type != "lower_triangular":
        raise ValueError(
            "packed Pallas HSTU attention requires a lower-triangular mask"
        )
    if dropout_pr != 0.0:
        raise ValueError("packed Pallas HSTU attention does not support dropout")
    if attn_bias is not None:
        raise ValueError("packed Pallas HSTU attention does not support attention bias")
    if max_attn_len != 0 or contextual_seq_len != 0 or full_attn_size != 0:
        raise ValueError(
            "packed Pallas HSTU attention does not support attention windows"
        )
    if num_softmax_heads != 0:
        raise ValueError("packed Pallas HSTU attention does not support softmax heads")

    targets = (
        seq_offsets.new_zeros((seq_offsets.shape[0] - 1,), dtype=torch.int32)
        if num_targets is None
        else num_targets.to(torch.int32)
    )
    scale = v.new_tensor(1.0 / N) if attn_scale is None else attn_scale
    if scale.ndim != 0:
        raise ValueError("packed Pallas HSTU attention requires a scalar attn_scale")
    if scale.requires_grad:
        raise ValueError(
            "packed Pallas HSTU attention does not differentiate attn_scale"
        )

    return torch.ops.recir.pallas_jagged_hstu_attention_fwd(
        q,
        k,
        v,
        seq_offsets,
        targets,
        scale,
        alpha,
        N,
    )


def install_pallas_jagged_hstu_attention_swap() -> None:
    """Install direct packed-input Pallas attention behind Hammer's seam."""
    import hammer.v2.ops.hstu_preprocess_and_attention as hstu_ops

    hstu_ops.pytorch_hstu_mha = _recir_pallas_jagged_hstu_mha
