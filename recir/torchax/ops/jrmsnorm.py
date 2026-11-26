# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""XLA lowering for recir::rmsnorm_fwd and recir::rmsnorm_backward."""

from typing import Any, Optional

import jax
import jax.numpy as jnp
import recir.ops.rmsnorm_op  # noqa: F401
import torch
from recir.torchax.ops.decorators import op


@op(torch.ops.recir.rmsnorm_fwd)
def rmsnorm_fwd(
    x: Any,
    weight: Optional[Any],
    eps: float,
    silu: bool,
) -> tuple[Any, Any]:
    xf = x.astype(jnp.float32)
    rstd = jax.lax.rsqrt(jnp.mean(xf * xf, axis=-1) + eps)
    y = xf * rstd[..., None]
    if weight is not None:
        y = y * weight.astype(jnp.float32)
    if silu:
        y = y * jax.nn.sigmoid(y)
    return y.astype(x.dtype), rstd


@op(torch.ops.recir.rmsnorm_backward)
def rmsnorm_backward(
    grad: Any,
    x: Any,
    weight: Optional[Any],
    rstd: Any,
    silu: bool,
) -> tuple[Any, Any]:
    rstd = rstd[..., None]
    xhat = x.astype(jnp.float32) * rstd
    dy = grad.astype(jnp.float32)
    w = weight.astype(jnp.float32) if weight is not None else None
    if silu:
        pre = xhat * w if w is not None else xhat
        sig = jax.nn.sigmoid(pre)
        dy = dy * (sig + pre * sig * (1.0 - sig))
    wdy = dy * w if w is not None else dy
    c1 = jnp.mean(xhat * wdy, axis=-1, keepdims=True)
    dx = ((wdy - xhat * c1) * rstd).astype(x.dtype)
    if weight is None:
        return dx, jnp.zeros((0,), dtype=x.dtype)
    dweight = jnp.sum(
        (dy * xhat).reshape(-1, x.shape[-1]), axis=0, dtype=jnp.float32
    ).astype(weight.dtype)
    return dx, dweight
