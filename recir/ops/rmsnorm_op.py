# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""RMSNorm over the last dim with an optional weight and fused SiLU.

Matches hammer's ``triton_rms_norm``: f32 math, ``y`` in ``x``'s dtype, f32
``rstd`` saved for the backward, ``dweight`` in the weight's dtype.
"""

from typing import Any, Optional

import torch
import torch.nn.functional as F

_LIB = torch.library.Library("recir", "FRAGMENT")
_LIB.define(
    "rmsnorm_fwd(Tensor x, Tensor? weight, float eps, bool silu) -> (Tensor, Tensor)"
)
_LIB.define(
    "rmsnorm_backward(Tensor grad, Tensor x, Tensor? weight, Tensor rstd, bool silu) "
    "-> (Tensor, Tensor)"
)


def _rmsnorm_fwd(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    eps: float,
    silu: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    xf = x.float()
    rstd = torch.rsqrt(xf.square().mean(-1) + eps)
    y = xf * rstd.unsqueeze(-1)
    if weight is not None:
        y = y * weight.float()
    if silu:
        y = F.silu(y)
    return y.to(x.dtype), rstd


def _rmsnorm_backward(
    grad: torch.Tensor,
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    rstd: torch.Tensor,
    silu: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    rstd = rstd.unsqueeze(-1)
    xhat = x.float() * rstd
    dy = grad.float()
    w = weight.float() if weight is not None else None
    if silu:
        pre = xhat * w if w is not None else xhat
        sig = torch.sigmoid(pre)
        dy = dy * (sig + pre * sig * (1.0 - sig))
    wdy = dy * w if w is not None else dy
    c1 = (xhat * wdy).mean(-1, keepdim=True)
    dx = ((wdy - xhat * c1) * rstd).to(x.dtype)
    if weight is None:
        return dx, x.new_empty((0,))
    dweight = (dy * xhat).reshape(-1, x.shape[-1]).sum(0).to(weight.dtype)
    return dx, dweight


_LIB.impl("rmsnorm_fwd", _rmsnorm_fwd, "CompositeExplicitAutograd")
_LIB.impl("rmsnorm_backward", _rmsnorm_backward, "CompositeExplicitAutograd")


@torch.library.register_fake("recir::rmsnorm_fwd")
def _rmsnorm_fwd_fake(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    eps: float,
    silu: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(x), x.new_empty(x.shape[:-1], dtype=torch.float32)


@torch.library.register_fake("recir::rmsnorm_backward")
def _rmsnorm_backward_fake(
    grad: torch.Tensor,
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    rstd: torch.Tensor,
    silu: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    dweight = torch.empty_like(weight) if weight is not None else x.new_empty((0,))
    return torch.empty_like(x), dweight


def _setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
    x, weight, _, ctx.silu = inputs
    rstd = output[1]
    ctx.mark_non_differentiable(rstd)
    ctx.has_weight = weight is not None
    ctx.save_for_backward(x, weight, rstd)


def _backward(ctx: Any, grad: torch.Tensor, grad_rstd: Any) -> tuple[Any, ...]:
    x, weight, rstd = ctx.saved_tensors
    dx, dweight = torch.ops.recir.rmsnorm_backward(grad, x, weight, rstd, ctx.silu)
    return dx, dweight if ctx.has_weight else None, None, None


torch.library.register_autograd(
    "recir::rmsnorm_fwd", _backward, setup_context=_setup_context
)


def rmsnorm(
    x: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    silu: bool = False,
) -> torch.Tensor:
    return torch.ops.recir.rmsnorm_fwd(x, weight, eps, silu)[0]
