# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""Custom op for the recir HSTU attention path.

Defines ``torch.ops.recir.hstu_attention`` (+ ``_backward``) and an installer
that swaps Hammer's PyTorch HSTU attention operation during RecIR setup. Mirrors
``recir.ops.seq_embedding_op`` (ring-shift) structurally:

  * schema + CPU/GPU fallback (a plain-torch materialized HSTU attention, so the
    op is correct off-TPU and opaque to ``make_fx``),
  * ``register_fake`` (shape/meta prop for the fake-tensor trace),
  * ``register_autograd`` -> ``hstu_attention_backward`` (q/k/v grads; lengths /
    num_targets / scalars get ``None``).

The TPU lowering to the Pallas kernel is registered in
``recir.torchax.ops.jhstu_attention``.

Contract:
  hstu_attention(q, k, v, lengths, num_targets, alpha, inv_n) -> out
    q, k        : [B, H, L, Dqk]  (dense, padded to static L)
    v           : [B, H, L, Dv]
    lengths     : [B] int         valid tokens per row
    num_targets : [B] int         candidate count per row (0 = pure causal)
    alpha       : float           pre-SiLU score scale (1/sqrt(Dqk))
    inv_n       : float           post-SiLU scale (1/max_seq_len)
    returns     : [B, H, L, Dv]
Only q/k/v are differentiable.
"""

import torch
import torch.nn.functional as F
from torch.fx._symbolic_trace import is_fx_tracing

_LIB = torch.library.Library("recir", "FRAGMENT")

_SCHEMA = (
    "hstu_attention(Tensor q, Tensor k, Tensor v, Tensor lengths, "
    "Tensor num_targets, float alpha, float inv_n) -> Tensor"
)
_BWD_SCHEMA = (
    "hstu_attention_backward(Tensor grad_out, Tensor q, Tensor k, Tensor v, "
    "Tensor lengths, Tensor num_targets, float alpha, float inv_n) "
    "-> (Tensor, Tensor, Tensor)"
)
_LIB.define(_SCHEMA)
_LIB.define(_BWD_SCHEMA)


def _valid_mask(
    length: torch.Tensor, num_targets: torch.Tensor, seq_len: int, device: torch.device
) -> torch.Tensor:
    """HSTU causal + num_targets mask, ``[B, L, L]`` bool (see _get_valid_attn_mask)."""
    pos = torch.arange(seq_len, device=device)
    max_id = (length - num_targets).unsqueeze(1)  # [B, 1]
    ids = torch.minimum(pos.unsqueeze(0), max_id)  # [B, L]
    row = ids.unsqueeze(2)
    col = ids.unsqueeze(1)
    causal = (row - col) > 0
    eye = (pos.unsqueeze(1) == pos.unsqueeze(0)).unsqueeze(0)
    valid_pos = pos.unsqueeze(0) < length.unsqueeze(1)  # [B, L]
    in_bounds = valid_pos.unsqueeze(2) & valid_pos.unsqueeze(1)
    return (causal | eye) & in_bounds


def _hstu_attention_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lengths: torch.Tensor,
    num_targets: torch.Tensor,
    alpha: float,
    inv_n: float,
) -> torch.Tensor:
    """Plain-torch materialized HSTU attention (matches the JAX reference)."""
    seq_len = q.shape[2]
    scores = torch.einsum("bhid,bhjd->bhij", q.float(), k.float()) * alpha
    weights = F.silu(scores) * inv_n
    valid = _valid_mask(lengths, num_targets, seq_len, q.device)  # [B, L, L]
    weights = torch.where(valid.unsqueeze(1), weights, weights.new_zeros(())).to(
        v.dtype
    )
    out = torch.einsum("bhij,bhjd->bhid", weights, v)
    valid_q = torch.arange(seq_len, device=q.device).unsqueeze(0) < lengths.unsqueeze(1)
    return torch.where(valid_q.unsqueeze(1).unsqueeze(-1), out, out.new_zeros(()))


def _fwd_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lengths: torch.Tensor,
    num_targets: torch.Tensor,
    alpha: float,
    inv_n: float,
) -> torch.Tensor:
    return _hstu_attention_ref(q, k, v, lengths, num_targets, alpha, inv_n)


def _bwd_fallback(
    grad_out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lengths: torch.Tensor,
    num_targets: torch.Tensor,
    alpha: float,
    inv_n: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qd = q.detach().requires_grad_(True)
    kd = k.detach().requires_grad_(True)
    vd = v.detach().requires_grad_(True)
    with torch.enable_grad():
        out = _hstu_attention_ref(qd, kd, vd, lengths, num_targets, alpha, inv_n)
        dq, dk, dv = torch.autograd.grad(out, (qd, kd, vd), grad_out)
    return dq, dk, dv


# CompositeExplicitAutograd: opaque to autograd so make_fx keeps it as one op;
# gradients come from the registered autograd below.
_LIB.impl("hstu_attention", _fwd_fallback, "CompositeExplicitAutograd")
_LIB.impl("hstu_attention_backward", _bwd_fallback, "CompositeExplicitAutograd")


@torch.library.register_fake("recir::hstu_attention")
def _fwd_meta(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lengths: torch.Tensor,
    num_targets: torch.Tensor,
    alpha: float,
    inv_n: float,
) -> torch.Tensor:
    b, h, seq_len, _ = q.shape
    dv = v.shape[-1]
    return q.new_empty((b, h, seq_len, dv), dtype=q.dtype)


@torch.library.register_fake("recir::hstu_attention_backward")
def _bwd_meta(
    grad_out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lengths: torch.Tensor,
    num_targets: torch.Tensor,
    alpha: float,
    inv_n: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


# pyre-ignore[2]: variadic ctx/inputs for torch autograd registration.
def _setup_context(ctx, inputs, output) -> None:  # noqa
    q, k, v, lengths, num_targets, alpha, inv_n = inputs
    ctx.save_for_backward(q, k, v, lengths, num_targets)
    ctx.alpha = alpha
    ctx.inv_n = inv_n


# pyre-ignore[2,3]
def _backward(ctx, grad_out):  # noqa
    q, k, v, lengths, num_targets = ctx.saved_tensors
    dq, dk, dv = torch.ops.recir.hstu_attention_backward(
        grad_out, q, k, v, lengths, num_targets, ctx.alpha, ctx.inv_n
    )
    # Grads for (q, k, v, lengths, num_targets, alpha, inv_n).
    return dq, dk, dv, None, None, None, None


torch.library.register_autograd(
    "recir::hstu_attention", _backward, setup_context=_setup_context
)


# ---------------------------------------------------------------------------
# recir::hstu_attention_varlen -- jagged adapter matching hstu_mha's interface.
#
# The Hammer-compatible swap passes jagged q/k/v ([total_tokens, H, D] +
# seq_offsets) to this op. It pads to dense [B, H, N, D] (exactly like pytorch_hstu_mha._pad_qkv),
# runs the dense kernel op, and unpads -- so the whole path parity-matches
# pytorch_hstu_mha. CompositeImplicitAutograd: autograd + make_fx decompose it
# into fbgemm pad/unpad + recir::hstu_attention, so no separate backward or
# lowering is needed (the dense op carries its own autograd + Pallas lowering).
# ---------------------------------------------------------------------------
_VARLEN_SCHEMA = (
    "hstu_attention_varlen(Tensor q, Tensor k, Tensor v, Tensor seq_offsets, "
    "Tensor num_targets, float alpha, int max_seq_len) -> Tensor"
)
_LIB.define(_VARLEN_SCHEMA)


def _hstu_attention_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: torch.Tensor,
    alpha: float,
    max_seq_len: int,
) -> torch.Tensor:
    total_len, num_heads, dqk = q.shape
    dv = v.shape[2]

    def _pad(x: torch.Tensor, d: int) -> torch.Tensor:
        # [L, H, d] -> [B, H, N, d] (matches pytorch_hstu_mha._pad_qkv).
        return (
            torch.ops.fbgemm.jagged_to_padded_dense(
                x.reshape(total_len, num_heads * d),
                [seq_offsets],
                [max_seq_len],
                0.0,
            )
            .view(-1, max_seq_len, num_heads, d)
            .transpose(1, 2)
        )

    qd = _pad(q, dqk)
    kd = _pad(k, dqk)
    vd = _pad(v, dv)
    lengths = (seq_offsets[1:] - seq_offsets[:-1]).to(torch.int32)
    out = torch.ops.recir.hstu_attention(
        qd, kd, vd, lengths, num_targets.to(torch.int32), alpha, 1.0 / max_seq_len
    )  # [B, H, N, Dv]
    out2d = out.transpose(1, 2).flatten(2, 3)  # [B, N, H*Dv]
    return torch.ops.fbgemm.dense_to_jagged(out2d, [seq_offsets], total_len)[0].view(
        total_len, num_heads, dv
    )


# CompositeImplicitAutograd: decomposes into pad + hstu_attention + unpad, so
# autograd and make_fx flow through the sub-ops (no separate backward).
_LIB.impl("hstu_attention_varlen", _hstu_attention_varlen, "CompositeImplicitAutograd")


def _recir_pytorch_hstu_mha(
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
    """Hammer-compatible adapter for the RankFM HSTU configuration."""
    if invalid_attn_mask_type != "lower_triangular":
        raise ValueError("RecIR HSTU attention requires a lower-triangular mask")
    if dropout_pr != 0.0:
        raise ValueError("RecIR HSTU attention does not support dropout")
    if attn_bias is not None:
        raise ValueError("RecIR HSTU attention does not support attention bias")
    if max_attn_len != 0 or contextual_seq_len != 0 or full_attn_size != 0:
        raise ValueError("RecIR HSTU attention does not support attention windows")
    if num_softmax_heads != 0:
        raise ValueError("RecIR HSTU attention does not support softmax heads")
    if attn_scale is not None and not is_fx_tracing():
        if attn_scale.numel() != 1 or not torch.allclose(
            attn_scale, attn_scale.new_tensor(1.0 / N)
        ):
            raise ValueError("RecIR HSTU attention requires a 1/N attention scale")

    targets = (
        num_targets
        if num_targets is not None
        else torch.zeros(seq_offsets.numel() - 1, dtype=torch.int32, device=q.device)
    )
    return torch.ops.recir.hstu_attention_varlen(
        q, k, v, seq_offsets, targets, alpha, N
    )


def install_hstu_attention_swap() -> None:
    """Install the RecIR operation behind Hammer's PyTorch attention seam."""
    import hammer.v2.ops.hstu_preprocess_and_attention as hstu_ops

    hstu_ops.pytorch_hstu_mha = _recir_pytorch_hstu_mha
