# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""XLA lowering for recir::rmsnorm_fwd and recir::rmsnorm_backward.

``RECIR_RMSNORM_KERNEL=1`` (default off) runs both as the single-pass Pallas
kernels in ``recir/ops/pallas/rmsnorm.py`` where they apply (D a multiple of
128, f32 or bf16, rows divisible by the device count).
"""

import os
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import recir.ops.rmsnorm_op  # noqa: F401
import torch
from jax.sharding import Mesh, PartitionSpec as P
from recir.ops.pallas import rmsnorm as pallas_rmsnorm
from recir.torchax.ops.decorators import op

_AXIS = "torch_dist"


def _kernel_enabled(x: Any) -> bool:
    value = os.environ.get("RECIR_RMSNORM_KERNEL", "0")
    if value not in ("0", "1"):
        raise ValueError(f"Invalid RECIR_RMSNORM_KERNEL: {value!r}")
    if value == "0" or x.ndim < 2 or x.size == 0:
        return False
    rows = x.size // x.shape[-1]
    return (
        pallas_rmsnorm.supported(x.shape[-1], x.dtype)
        and rows % len(jax.devices()) == 0
    )


def _row_sharded(fn: Callable[..., Any], row_args: int, out_specs: Any) -> Any:
    """``fn`` per device on its rows: GSPMD cannot partition a Pallas call."""
    if len(jax.devices()) == 1:
        return fn

    def mapped(*args: Any) -> Any:
        in_specs = tuple(
            P(_AXIS) if i < row_args else (None if a is None else P())
            for i, a in enumerate(args)
        )
        return jax.shard_map(
            fn,
            mesh=Mesh(jax.devices(), (_AXIS,)),
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=False,
        )(*args)

    return mapped


def _interpret() -> bool:
    return jax.default_backend() == "cpu"


def _kernel_fwd(x: Any, weight: Optional[Any], eps: float, silu: bool) -> Any:
    def fwd(x2d: Any, w: Optional[Any]) -> Any:
        return pallas_rmsnorm.rmsnorm_fwd(x2d, w, eps, silu, interpret=_interpret())

    y, rstd = _row_sharded(fwd, 1, (P(_AXIS), P(_AXIS)))(
        x.reshape(-1, x.shape[-1]), weight
    )
    return y.reshape(x.shape), rstd.reshape(x.shape[:-1])


def _kernel_backward(
    grad: Any, x: Any, weight: Optional[Any], rstd: Any, silu: bool
) -> tuple[Any, Any]:
    def bwd(g: Any, x2d: Any, r: Any, w: Optional[Any]) -> Any:
        dx, dweight = pallas_rmsnorm.rmsnorm_backward(
            g, x2d, w, r, silu, interpret=_interpret()
        )
        if dweight is None:
            dweight = jnp.zeros((x2d.shape[1],), jnp.float32)
        return dx, dweight[None]

    d = x.shape[-1]
    dx, dweight = _row_sharded(bwd, 3, (P(_AXIS), P(_AXIS)))(
        grad.reshape(-1, d), x.reshape(-1, d), rstd.reshape(-1), weight
    )
    if weight is None:
        return dx.reshape(x.shape), jnp.zeros((0,), dtype=x.dtype)
    return dx.reshape(x.shape), jnp.sum(dweight, axis=0).astype(weight.dtype)


def xla_rmsnorm_fwd(
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


def xla_rmsnorm_backward(
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


@op(torch.ops.recir.rmsnorm_fwd)
def rmsnorm_fwd(
    x: Any,
    weight: Optional[Any],
    eps: float,
    silu: bool,
) -> tuple[Any, Any]:
    if _kernel_enabled(x):
        return _kernel_fwd(x, weight, eps, silu)
    return xla_rmsnorm_fwd(x, weight, eps, silu)


@op(torch.ops.recir.rmsnorm_backward)
def rmsnorm_backward(
    grad: Any,
    x: Any,
    weight: Optional[Any],
    rstd: Any,
    silu: bool,
) -> tuple[Any, Any]:
    if _kernel_enabled(x):
        return _kernel_backward(grad, x, weight, rstd, silu)
    return xla_rmsnorm_backward(grad, x, weight, rstd, silu)
