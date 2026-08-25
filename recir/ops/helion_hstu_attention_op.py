# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""The ``recir::helion_ragged_attention_fwd`` / ``_bwd`` HSTU attention ops.

Two opaque torch ops wrapping the recir-owned Helion jagged HSTU attention kernels
(``recir.ops.helion.jagged_hstu_attention``, which owns the kernel math and the
scale-folding contract), lowered to JAX by ``recir/torchax/ops/jhstu_attention.py``.

How the model reaches them
--------------------------
``hammer.ops.pytorch.ragged_hstu_attention.pytorch_hstu_mha`` is only
``@torch.fx.wrap``'d, which affects ``torch.fx.symbolic_trace`` **only**. recir
traces with plain ``make_fx`` (``recir/api.py``), which sees through the wrap and
decomposes attention into ~15 aten ops. To get ONE node under ``make_fx`` the
attention leaf must be a real opaque custom op with a ``register_fake``.

``install_helion_hstu_attention_swap()`` rebinds the hammer module's
``pytorch_hstu_mha`` name to :func:`_recir_pytorch_hstu_mha`, a plain Python
adapter that keeps hammer's exact 16-parameter signature, validates the 8 args the
Helion kernels do not implement, normalizes ``num_targets``/``attn_scale``, and
calls the clean 8-arg op. The traced op is therefore already the lowered op -- no
seam op and no ``op_swap_map`` rewrite are involved.

Autograd
--------
The forward carries ``register_autograd``, so recir's traced ``loss.backward()``
emits a single ``recir::helion_ragged_attention_bwd`` node that the same torchax
module lowers to the three Helion backward kernels.

Eager numerics are byte-identical to production
-----------------------------------------------
The forward CPU/eager fallback calls the **captured original**
``pytorch_hstu_mha`` verbatim, and the backward fallback is the autograd of that
exact same forward. So eager / CPU / non-recir paths are unchanged; only the
XLA-lowered path runs Helion. The captured function object is snapshotted at
module load, BEFORE any rebind, and the origin module attribute is never patched,
so the fallbacks never recurse back into the op.

The fallback is deliberately NOT ``reference_jagged_hstu_attention``: the rebind is
left installed globally (a live production path for non-helion models), and
``parity_forward_loss.py`` compares the compiled cell against the eager cell, so
moving eager numerics would make a Helion regression indistinguishable from an
expected reference delta.
"""

from typing import Optional

import torch

# Snapshot the real ``pytorch_hstu_mha`` at module load, BEFORE any
# ``install_helion_hstu_attention_swap()`` rebind. The rebind patches only the
# ``hstu_preprocess_and_attention`` module attribute (not this origin), so this
# object is always the genuine reference impl and the fallbacks never recurse.
from hammer.ops.pytorch.ragged_hstu_attention import (
    pytorch_hstu_mha as _ORIG_PYTORCH_HSTU_MHA,
)
from recir.ops.helion.jagged_hstu_attention import (
    record_traced_bwd_sig,
    record_traced_fwd_sig,
)

# The only mask regime the target-aware-causal SiLU kernels implement. hammer maps
# this literal to ``causal=True`` in ``get_invalid_attn_mask``; the other
# ``InvalidAttnMask`` members ("upper_triangular", "none") are different math.
_SUPPORTED_ATTN_MASK_TYPE = "lower_triangular"

_LIB = torch.library.Library("recir", "FRAGMENT")

# --- forward -------------------------------------------------------------
# ``attn_scale`` stays a Tensor rather than collapsing to a ``float inv_n``:
# AttnScaleType.MAX_UIH_LEN (what prod sets, see udd_fm_v3's
# train_module_wrapper.py) is 1/max(uih_lengths), which is data-dependent and
# unrelated to N. ``N`` is load-bearing, not vestigial: it is the 5th element of
# the warm-up signature ``(L, H, D, B, N, dtype)`` and ``register_fake`` cannot
# record what is not in the schema.
_FWD_SCHEMA = (
    "helion_ragged_attention_fwd(Tensor q, Tensor k, Tensor v, "
    "Tensor seq_offsets, Tensor num_targets, Tensor attn_scale, "
    "float alpha, int N) -> Tensor"
)
_LIB.define(_FWD_SCHEMA)


def _fwd_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: torch.Tensor,
    attn_scale: torch.Tensor,
    alpha: float,
    N: int,
) -> torch.Tensor:
    # Verbatim production numerics: defer to the captured original impl. The
    # dropped args are pinned to the trivial values the adapter enforces.
    return _ORIG_PYTORCH_HSTU_MHA(
        q,
        k,
        v,
        alpha,
        N,
        seq_offsets,
        _SUPPORTED_ATTN_MASK_TYPE,
        0.0,
        False,
        num_targets,
        None,
        attn_scale,
        0,
        0,
        0,
        0,
    )


# CompositeExplicitAutograd: opaque to autograd so make_fx keeps it as one op;
# the gradient comes from the register_autograd formula below (a single bwd node).
_LIB.impl("helion_ragged_attention_fwd", _fwd_fallback, "CompositeExplicitAutograd")


@torch.library.register_fake("recir::helion_ragged_attention_fwd")
def _fwd_meta(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: torch.Tensor,
    attn_scale: torch.Tensor,
    alpha: float,
    N: int,
) -> torch.Tensor:
    # HSTU self-attention: out is [L, H, V] (q is [L, H, D], v is [L, H, V]).
    L, H, _ = q.shape
    # Record the traced fwd shape so the model path can warm the Helion fwd kernel
    # for it AFTER the trace and BEFORE the first (lazy) XLA lower -- Helion
    # compile cannot nest inside the lower's tracer. This runs with CONCRETE
    # FakeTensor shapes (GSPMD fake mode has no ShapeEnv). B is num_sequences =
    # seq_offsets length - 1; N = max_seq_len (per-device pad width).
    record_traced_fwd_sig(
        (
            int(q.shape[0]),
            int(q.shape[1]),
            int(q.shape[2]),
            int(seq_offsets.shape[0]) - 1,
            int(N),
            q.dtype,
        )
    )
    return q.new_empty((L, H, v.shape[2]))


# --- backward (functional: returns dq, dk, dv) ---------------------------
_BWD_SCHEMA = (
    "helion_ragged_attention_bwd(Tensor grad_out, Tensor q, Tensor k, Tensor v, "
    "Tensor seq_offsets, Tensor num_targets, Tensor attn_scale, float alpha, "
    "int N) -> (Tensor, Tensor, Tensor)"
)
_LIB.define(_BWD_SCHEMA)


def _bwd_fallback(
    grad_out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: torch.Tensor,
    attn_scale: torch.Tensor,
    alpha: float,
    N: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Bulletproof eager match: differentiate the exact captured original forward,
    # so the eager bwd == autograd(eager fwd) for any args.
    q_ = q.detach().clone().requires_grad_(True)
    k_ = k.detach().clone().requires_grad_(True)
    v_ = v.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        out = _ORIG_PYTORCH_HSTU_MHA(
            q_,
            k_,
            v_,
            alpha,
            N,
            seq_offsets,
            _SUPPORTED_ATTN_MASK_TYPE,
            0.0,
            False,
            num_targets,
            None,
            attn_scale,
            0,
            0,
            0,
            0,
        )
        dq, dk, dv = torch.autograd.grad(out, (q_, k_, v_), grad_out)
    return dq, dk, dv


_LIB.impl("helion_ragged_attention_bwd", _bwd_fallback, "CompositeExplicitAutograd")


@torch.library.register_fake("recir::helion_ragged_attention_bwd")
def _bwd_meta(
    grad_out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: torch.Tensor,
    attn_scale: torch.Tensor,
    alpha: float,
    N: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Record the traced bwd shape so the model path can warm the three Helion
    # backward kernels for it before the first lower (see the fwd meta kernel).
    record_traced_bwd_sig(
        (
            int(q.shape[0]),
            int(q.shape[1]),
            int(q.shape[2]),
            int(seq_offsets.shape[0]) - 1,
            int(N),
            q.dtype,
        )
    )
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


# pyre-ignore[2]: variadic ctx/inputs for torch autograd registration.
def _setup_context(ctx, inputs, output) -> None:  # noqa
    q, k, v, seq_offsets, num_targets, attn_scale, alpha, N = inputs
    ctx.save_for_backward(q, k, v, seq_offsets, num_targets, attn_scale)
    ctx.alpha = alpha
    ctx.N = N


# pyre-ignore[2,3]
def _backward(ctx, grad_out):  # noqa
    q, k, v, seq_offsets, num_targets, attn_scale = ctx.saved_tensors
    dq, dk, dv = torch.ops.recir.helion_ragged_attention_bwd(
        grad_out,
        q,
        k,
        v,
        seq_offsets,
        num_targets,
        attn_scale,
        ctx.alpha,
        ctx.N,
    )
    # Grads aligned to the 8 fwd inputs: dq/dk/dv for q/k/v; None for
    # seq_offsets, num_targets, attn_scale, alpha, N.
    return dq, dk, dv, None, None, None, None, None


torch.library.register_autograd(
    "recir::helion_ragged_attention_fwd", _backward, setup_context=_setup_context
)


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
    num_targets: Optional[torch.Tensor] = None,
    attn_bias: Optional[torch.Tensor] = None,
    attn_scale: Optional[torch.Tensor] = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
    full_attn_size: int = 0,
    num_softmax_heads: int = 0,
) -> torch.Tensor:
    """Drop-in ``pytorch_hstu_mha`` replacement that calls the clean recir op.

    Keeps hammer's exact parameter names and order so the STU call site's kwargs
    bind unchanged after the rebind. Validates the args the Helion kernels do not
    implement, then normalizes the two optionals the op requires as real tensors:

    - ``num_targets=None`` -> zeros[B], which makes the target-aware mask reduce to
      plain causal (``cap == seq_len``).
    - ``attn_scale=None`` -> ``v.new_ones(())``. NOTE this deliberately differs
      from hammer, which reads ``None`` as ``1/N``: both Helion sites (the kernel
      wrappers' ``v_scaled`` fold and the reference) read it as ``1.0``, so
      deferring to hammer's ``1/N`` here would be a silent factor-N error. udd_fm_v3
      always passes a real 0-d tensor, so this only pins the previously-divergent
      ``None`` case.

    ``training`` is not validated: it only reaches ``F.dropout(..., p=dropout_pr,
    training=training)``, which is inert once ``dropout_pr == 0.0`` is enforced.
    """
    if invalid_attn_mask_type != _SUPPORTED_ATTN_MASK_TYPE:
        raise ValueError(
            "helion HSTU attention: invalid_attn_mask_type must be "
            f"{_SUPPORTED_ATTN_MASK_TYPE!r}, got {invalid_attn_mask_type!r}"
        )
    if attn_bias is not None:
        raise ValueError("helion HSTU attention: attn_bias must be None")
    if dropout_pr != 0.0:
        raise ValueError(
            f"helion HSTU attention: dropout_pr must be 0, got {dropout_pr}"
        )
    if max_attn_len != 0:
        raise ValueError(
            f"helion HSTU attention: max_attn_len must be 0, got {max_attn_len}"
        )
    if contextual_seq_len != 0:
        raise ValueError(
            "helion HSTU attention: contextual_seq_len must be 0, got "
            f"{contextual_seq_len}"
        )
    if full_attn_size != 0:
        raise ValueError(
            f"helion HSTU attention: full_attn_size must be 0, got {full_attn_size}"
        )
    if num_softmax_heads != 0:
        raise ValueError(
            "helion HSTU attention: num_softmax_heads must be 0 (the kernels are "
            f"pure SiLU), got {num_softmax_heads}"
        )

    nt = (
        seq_offsets.new_zeros((int(seq_offsets.shape[0]) - 1,))
        if num_targets is None
        else num_targets
    )
    sc = v.new_ones(()) if attn_scale is None else attn_scale
    return torch.ops.recir.helion_ragged_attention_fwd(
        q, k, v, seq_offsets, nt, sc, alpha, N
    )


def install_helion_hstu_attention_swap() -> None:
    """Rebind the hammer PYTORCH attention leaf to :func:`_recir_pytorch_hstu_mha`.

    The v2 STU calls ``pytorch_hstu_mha(...)`` resolved as the
    ``hammer.v2.ops.hstu_preprocess_and_attention`` module global (that module did
    ``from ...ragged_hstu_attention import pytorch_hstu_mha``). Patching THAT
    module attribute (not the origin) makes the PYTORCH-kernel STU emit a single
    ``recir::helion_ragged_attention_fwd`` node under make_fx while leaving eager
    numerics identical.

    Idempotent; safe to leave installed for non-helion models (the op's eager
    fallback is the captured original). MUTUALLY EXCLUSIVE with the dense
    pad-then-unpad Pallas path's own install: whichever runs last wins, and the
    loser's kernels are silently never used.
    """
    import hammer.v2.ops.hstu_preprocess_and_attention as _hpa

    _hpa.pytorch_hstu_mha = _recir_pytorch_hstu_mha
