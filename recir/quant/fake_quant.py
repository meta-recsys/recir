# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Fake-quantize with straight-through estimator (STE) + LSQ range learning.

Simulates inference-time quantization during training so weights adapt to the
reduced precision they will be served at. The forward pass rounds and clamps
to a discrete grid; the backward pass propagates gradients using **Learned
Step Size Quantization (LSQ)** [Esser et al., 2020] for the learnable
quantization parameters and straight-through pass-through for the input.

Grad formulas (per LSQ):
    * ``x``   (weight): pass-through inside clip range, zero at saturation.
                        Chained through the multiplicative ``scale`` gain.
    * ``clip`` (LSQ step-size proxy, since Δ = clip / q_max):
        - inside range: ``x_int - z - x/Δ``   (local quant-error signal)
        - at q_min: ``q_min - z``
        - at q_max: ``q_max - z``
      Reduced to the ``clip`` param shape (scalar per_table / [V, 1] per_row /
      [V, G] per_group) and scaled by ``grad_factor``.
    * ``zero_point`` (asymmetric only):
        - inside range: 0
        - at saturation: ``-Δ``
      Reduced + ``grad_factor``.
    * ``scale`` (multiplicative output gain, kept as a separate learnable):
      ``grad_out * x_fq_pre_gain``, reduced + ``grad_factor``.

The ``grad_factor`` follows the LSQ recommendation of ``1 / sqrt(N * q_max)``
(auto-computed when set to ``"auto"``), which balances the scale-parameter
update magnitude against weight-parameter updates.

Rounding modes:
    * ``"nearest"`` (default): ``round(x)`` -- deterministic, standard LSQ.
    * ``"stochastic"``: ``floor(x + U(0, 1))`` -- keeps ``E[SR(x)] == x`` so
      the expected quant grid is unbiased. Helps LSQ convergence at very low
      bit widths (INT4, MXFP4) by decoupling neighbouring weights that would
      otherwise all snap to the same integer.

Because ``recir`` traces a joint fwd+bwd+opt graph via ``functorch.make_fx``,
the STE ``backward`` must be captured *by* ``make_fx``. Wrapping the ops in
``torch.autograd.Function`` and applying it *before* ``make_fx`` runs (via the
``_parametrize_fns`` seam) achieves that.
"""

import math
from typing import Any, Union

import torch

# Numerical epsilon used to guard against division-by-zero on the learnable
# clip / scale parameters early in training.
_EPS: float = 1e-6

GradFactor = Union[float, str]  # "auto" or explicit float.


def _q_bounds(n_bits: int, symmetric: bool) -> tuple[int, int]:
    """Integer quantization grid bounds for ``n_bits`` (signed or unsigned)."""
    if symmetric:
        q_max = (
            1 << (n_bits - 1)
        ) - 1  # reduced symmetric range, e.g. int8 = [-127, 127]
        return -q_max, q_max
    return 0, (1 << n_bits) - 1  # e.g. uint8 = [0, 255]


def _resolve_grad_factor(grad_factor: GradFactor, numel: int, q_max: int) -> float:
    """LSQ-recommended ``grad_factor`` when ``grad_factor == "auto"``.

    Explicit floats pass through unchanged. ``"auto"`` uses
    ``1 / sqrt(numel * q_max)`` so the reduced-then-multiplied grad for
    ``scale`` / ``clip`` / ``zero_point`` sits on the same order of magnitude
    as per-element weight grads.
    """
    if isinstance(grad_factor, str):
        assert grad_factor == "auto", f"unknown grad_factor {grad_factor!r}"
        return 1.0 / math.sqrt(max(numel * q_max, 1))
    return float(grad_factor)


def _reduce_to_target_shape(g: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Sum ``g`` (upstream elementwise grad) down to the ``target`` param shape.

    Handles the three granularities produced by :class:`STEFakeQuant`:

    * ``target`` scalar (shape ``()``): sum over all dims (``per_table``).
    * ``target`` shape ``[V, 1]`` (``per_row``): ``g`` is ``[V, D]``; sum dim=1
      keepdim to get ``[V, 1]``.
    * ``target`` shape ``[V, G]`` (``per_group``): ``g`` is ``[V, G, gsize]``;
      sum dim=-1 to get ``[V, G]``.
    """
    if target.dim() == 0:
        return g.sum()
    # Trailing extra dims (per_group appends a group-element dim).
    while g.dim() > target.dim():
        g = g.sum(dim=-1)
    # Reduce any dims where target has size 1 (per_row keeps a size-1 dim).
    for i in range(target.dim()):
        if target.shape[i] == 1 and g.shape[i] != 1:
            g = g.sum(dim=i, keepdim=True)
    return g


def _stochastic_round(x: torch.Tensor) -> torch.Tensor:
    """Unbiased stochastic rounding: ``floor(x + U(0, 1))``.

    Portable pure-PyTorch implementation. Traceable through ``make_fx``:
    lowers to ``aten.floor`` + ``aten.rand_like`` + ``aten.add`` -- no custom
    op required. ``E[SR(x)] == x`` (verified in tests).
    """
    return torch.floor(x + torch.rand_like(x))


def _maybe_group_reshape(
    x: torch.Tensor,
    granularity: str,
    group_size: int,
) -> torch.Tensor:
    """Reshape ``x`` from ``[V, D]`` to ``[V, G, group_size]`` for per_group,
    otherwise return unchanged.
    """
    if granularity != "per_group":
        return x
    assert x.dim() == 2, f"per_group requires 2-D weight, got shape {tuple(x.shape)}"
    v, d = x.shape
    assert d % group_size == 0, (
        f"per_group group_size={group_size} must divide embedding_dim={d}"
    )
    g = d // group_size
    return x.reshape(v, g, group_size)


def _maybe_group_broadcast(p: torch.Tensor, granularity: str) -> torch.Tensor:
    """For per_group, unsqueeze the trailing dim so ``[V, G]`` broadcasts
    against ``[V, G, group_size]``. Scalars (per_table zero_point sentinel)
    and higher-dim tensors are left unchanged.
    """
    if granularity != "per_group":
        return p
    if p.dim() == 0:
        return p  # scalar broadcasts trivially
    return p.unsqueeze(-1)


class STEFakeQuant(torch.autograd.Function):
    """Fake-quantize with straight-through estimator (STE) + LSQ range learning.

    Forward:
        ``Δ = clip / q_max``  (clip is the LSQ step-size proxy)
        ``z_int = 0`` (symmetric) or ``round(zero_point).clamp(q_min, q_max)``
        ``x_int = round(x/Δ + z_int).clamp(q_min, q_max)``   (round is
        either ``torch.round`` or :func:`_stochastic_round`)
        ``x_fq  = (x_int - z_int) * Δ * scale``

    Backward: LSQ (see module docstring for formulas).

    Broadcast shapes:
        * ``per_table``: scale/clip/zp are scalars.
        * ``per_row``:   scale/clip/zp shape ``[V, 1]`` against weight ``[V, D]``.
        * ``per_group``: scale/clip/zp shape ``[V, G]``; weight is reshaped to
          ``[V, G, group_size]`` inside fwd/bwd, then reshaped back.
    """

    @staticmethod
    # pyre-ignore[14]: torch.autograd.Function.forward has flexible signature.
    def forward(
        ctx: Any,
        x: torch.Tensor,
        scale: torch.Tensor,
        clip: torch.Tensor,
        zero_point: torch.Tensor,
        n_bits: int,
        symmetric: bool,
        granularity: str,
        group_size: int,
        round_mode: str,
        grad_factor: GradFactor,
    ) -> torch.Tensor:
        q_min, q_max = _q_bounds(n_bits, symmetric)
        safe_clip = torch.clamp(clip, min=_EPS)
        safe_scale = torch.clamp(scale, min=_EPS)

        # Reshape weight + broadcast qat params for per_group; no-ops otherwise.
        orig_shape = x.shape
        x_view = _maybe_group_reshape(x, granularity, group_size)
        safe_scale_b = _maybe_group_broadcast(safe_scale, granularity)
        safe_clip_b = _maybe_group_broadcast(safe_clip, granularity)

        delta = safe_clip_b / q_max

        if symmetric:
            z_int = torch.zeros_like(safe_clip_b)
            x_scaled = x_view / delta
        else:
            zp_b = _maybe_group_broadcast(zero_point, granularity)
            z_int = torch.round(zp_b).clamp(q_min, q_max)
            x_scaled = x_view / delta + z_int

        if round_mode == "stochastic":
            x_int_unclamped = _stochastic_round(x_scaled)
        else:
            x_int_unclamped = torch.round(x_scaled)
        x_int = x_int_unclamped.clamp(q_min, q_max)
        x_fq = (x_int - z_int) * delta * safe_scale_b

        # Save the reshaped view (per_group) or original (per_row / per_table)
        # so backward operates on a single canonical shape without re-reshaping.
        ctx.save_for_backward(x_view, scale, clip, zero_point, x_int, z_int, delta)
        ctx.n_bits = n_bits
        ctx.symmetric = symmetric
        ctx.granularity = granularity
        ctx.group_size = group_size
        ctx.round_mode = round_mode
        ctx.grad_factor = grad_factor
        ctx.q_min = q_min
        ctx.q_max = q_max
        ctx.orig_shape = orig_shape

        # Reshape output back to caller-visible shape.
        if granularity == "per_group":
            return x_fq.reshape(orig_shape)
        return x_fq

    @staticmethod
    # pyre-ignore[14]
    def backward(
        ctx: Any, grad_out: torch.Tensor
    ) -> tuple[
        torch.Tensor | None,  # grad_x
        torch.Tensor | None,  # grad_scale
        torch.Tensor | None,  # grad_clip
        torch.Tensor | None,  # grad_zero_point
        None,  # n_bits
        None,  # symmetric
        None,  # granularity
        None,  # group_size
        None,  # round_mode
        None,  # grad_factor
    ]:
        x_view, scale, clip, zero_point, x_int, z_int, delta = ctx.saved_tensors
        symmetric: bool = ctx.symmetric
        granularity: str = ctx.granularity
        q_min: int = ctx.q_min
        q_max: int = ctx.q_max
        orig_shape = ctx.orig_shape

        safe_scale = torch.clamp(scale, min=_EPS)
        safe_scale_b = _maybe_group_broadcast(safe_scale, granularity)

        # Match grad_out's shape to the internal (possibly reshaped) x_view.
        if granularity == "per_group":
            grad_out_view = grad_out.reshape(x_view.shape)
        else:
            grad_out_view = grad_out

        # LSQ grad_factor (auto = 1 / sqrt(N * q_max)); balances update
        # magnitudes so scale / clip / zp don't dwarf weight updates.
        gf = _resolve_grad_factor(ctx.grad_factor, x_view.numel(), q_max)

        # Piecewise masks from the saturation status of the pre-clamp integer.
        at_qmin = x_int <= q_min
        at_qmax = x_int >= q_max
        in_range = ~(at_qmin | at_qmax)
        in_range_f = in_range.to(grad_out_view.dtype)

        # ---- grad_x: STE with clip-mask. Chained through the gain. ----
        grad_x_view = grad_out_view * in_range_f * safe_scale_b
        if granularity == "per_group":
            grad_x = grad_x_view.reshape(orig_shape)
        else:
            grad_x = grad_x_view

        # ---- grad_clip: LSQ formula, mapped from grad_Δ via chain rule. ----
        # Inside range: grad_Δ = (x_int - z_int) - x/Δ = local quant-error signal.
        # At saturation: grad_Δ = q_bound - z_int.
        grad_delta_inside = (x_int - z_int) - x_view / delta
        grad_delta_at_min = q_min - z_int  # broadcasts against x_view
        grad_delta_at_max = q_max - z_int
        grad_delta = torch.where(
            at_qmin,
            grad_delta_at_min.expand_as(x_view),
            torch.where(
                at_qmax, grad_delta_at_max.expand_as(x_view), grad_delta_inside
            ),
        )
        # Chain through gain, then map Δ -> clip via Δ = clip / q_max.
        grad_clip_full = grad_out_view * grad_delta * safe_scale_b / q_max
        grad_clip = _reduce_to_target_shape(grad_clip_full, clip) * gf

        # ---- grad_scale (multiplicative gain): d(x_fq)/d(gain) = x_fq/gain. ----
        x_fq_pre_gain = (x_int - z_int) * delta
        grad_scale_full = grad_out_view * x_fq_pre_gain
        grad_scale = _reduce_to_target_shape(grad_scale_full, scale) * gf

        # ---- grad_zero_point (asymmetric only): -Δ at saturation, 0 inside. ----
        if symmetric:
            grad_zp: torch.Tensor | None = None
        else:
            grad_zp_at_sat = (-delta).expand_as(x_view)
            grad_zp_per_elem = torch.where(
                at_qmin | at_qmax, grad_zp_at_sat, torch.zeros_like(x_view)
            )
            grad_zp_full = grad_out_view * grad_zp_per_elem * safe_scale_b
            grad_zp = _reduce_to_target_shape(grad_zp_full, zero_point) * gf

        return (
            grad_x,
            grad_scale,
            grad_clip,
            grad_zp,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def fake_quantize_ste(
    x: torch.Tensor,
    scale: torch.Tensor,
    clip: torch.Tensor,
    n_bits: int = 8,
    symmetric: bool = True,
    granularity: str = "per_table",
    zero_point: torch.Tensor | None = None,
    group_size: int = 32,
    round_mode: str = "nearest",
    grad_factor: GradFactor = "auto",
) -> torch.Tensor:
    """Public entry point: fake-quantize ``x`` with STE + LSQ range learning.

    Args:
        x: input tensor (embedding weight or activation).
        scale: learnable multiplicative gain (scalar / ``[V, 1]`` / ``[V, G]``).
            Defaults are usually ``1.0``.
        clip: learnable clip range (matching ``scale`` shape). LSQ step-size
            ``Δ = clip / q_max``.
        n_bits: number of quantization bits.
        symmetric: symmetric quantization grid (``True`` = signed centered at 0).
        granularity: ``"per_table"`` / ``"per_row"`` / ``"per_group"``.
        zero_point: learnable real-valued zero point (rounded via STE in fwd).
            Required when ``symmetric=False``; ignored when ``symmetric=True``.
            When None and asymmetric, a zero scalar is used (no learning).
        group_size: group size when ``granularity="per_group"``. Must divide
            the embedding dim.
        round_mode: ``"nearest"`` or ``"stochastic"``.
        grad_factor: LSQ grad multiplier for ``scale`` / ``clip`` /
            ``zero_point`` grads. ``"auto"`` uses ``1 / sqrt(numel * q_max)``.
    """
    if zero_point is None:
        # A zero scalar (non-learnable) works as a well-defined argument for
        # both fwd and bwd paths. Symmetric mode ignores it; asymmetric mode
        # uses it as a fixed zp of 0.
        zero_point = torch.zeros((), device=x.device, dtype=x.dtype)
    return STEFakeQuant.apply(
        x,
        scale,
        clip,
        zero_point,
        n_bits,
        symmetric,
        granularity,
        group_size,
        round_mode,
        grad_factor,
    )
