# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""RMSNorm forward and backward over [rows, D] as single-pass Pallas kernels.

A grid step owns a block of rows and makes every pass over them in VMEM: the
forward reads x once and writes y and rstd; the backward reads x, dy and rstd
once and writes dx and the step's partial weight gradient. XLA reads x twice
forward and x and dy twice backward.

rstd is stored lane-dense, one f32 row of 128 per 128 rows. Moving a column of
128 values into a row (and back) is a matmul against a diagonal matrix built
from them, which is exact and avoids transposing [128, 128] tiles.
"""

import functools
from typing import Any, Optional

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_LANES = 128
_BLOCK_ELEMS = 1 << 21
_VMEM_LIMIT = 56 << 20
_HIGHEST = jax.lax.Precision.HIGHEST


def supported(d: int, dtype: Any) -> bool:
    return (
        d % _LANES == 0
        and dtype in (jnp.float32, jnp.bfloat16)
        and d <= _BLOCK_ELEMS // _LANES
    )


def block_rows(d: int, dtype: Any) -> int:
    cap = 8192 if jnp.dtype(dtype).itemsize == 2 else 4096
    return min(cap, 1 << ((_BLOCK_ELEMS // d).bit_length() - 1))


def _stat_rows(rows_per_block: int) -> int:
    return max(8, rows_per_block // _LANES)


def _eye() -> Any:
    shape = (_LANES, _LANES)
    return jax.lax.broadcasted_iota(jnp.int32, shape, 0) == jax.lax.broadcasted_iota(
        jnp.int32, shape, 1
    )


def _column_to_tile(column: Any, rows_per_block: int) -> Any:
    t = _stat_rows(rows_per_block)
    sublane = jax.lax.broadcasted_iota(jnp.int32, (t, _LANES), 0)
    eye = _eye()
    ones = jnp.ones((8, _LANES), jnp.float32)
    tile = jnp.zeros((t, _LANES), jnp.float32)
    for g in range(rows_per_block // _LANES):
        part = column[g * _LANES : (g + 1) * _LANES, :]
        diag = jnp.where(eye, jnp.broadcast_to(part, (_LANES, _LANES)), 0.0)
        row = jnp.dot(
            ones, diag, precision=_HIGHEST, preferred_element_type=jnp.float32
        )
        tile = jnp.where(sublane == g, jnp.broadcast_to(row[0:1], (t, _LANES)), tile)
    return tile


def _tile_to_column(tile: Any, rows_per_block: int) -> Any:
    eye = _eye()
    ones = jnp.ones((_LANES, _LANES), jnp.float32)
    parts = []
    for g in range(rows_per_block // _LANES):
        diag = jnp.where(
            eye, jnp.broadcast_to(tile[g : g + 1, :], (_LANES, _LANES)), 0.0
        )
        col = jnp.dot(
            diag, ones, precision=_HIGHEST, preferred_element_type=jnp.float32
        )
        parts.append(col[:, 0:1])
    return jnp.concatenate(parts, axis=0)


def _fwd_kernel(*refs: Any, eps: float, silu: bool, weighted: bool) -> None:
    if weighted:
        x_ref, w_ref, y_ref, rstd_ref = refs
    else:
        x_ref, y_ref, rstd_ref = refs
        w_ref = None
    n, d = x_ref.shape
    x = x_ref[...].astype(jnp.float32)
    rstd = jax.lax.rsqrt(jnp.sum(x * x, axis=1, keepdims=True) * (1.0 / d) + eps)
    y = x * rstd
    if w_ref is not None:
        y = y * w_ref[...].astype(jnp.float32)
    if silu:
        y = y * jax.nn.sigmoid(y)
    y_ref[...] = y.astype(y_ref.dtype)
    rstd_ref[0] = _column_to_tile(rstd, n)


def _specs(rows: int, d: int, dtype: Any) -> tuple[int, int, int, Any, Any]:
    n = min(block_rows(d, dtype), pl.cdiv(rows, _LANES) * _LANES)
    steps = pl.cdiv(rows, n)
    t = _stat_rows(n)
    block = pl.BlockSpec((n, d), lambda i: (i, 0))
    stat = pl.BlockSpec((1, t, _LANES), lambda i: (i, 0, 0))
    return n, steps, t, block, stat


def _params() -> Any:
    return pltpu.CompilerParams(
        dimension_semantics=("parallel",), vmem_limit_bytes=_VMEM_LIMIT
    )


def rmsnorm_fwd(
    x: Any, weight: Optional[Any], eps: float, silu: bool, interpret: bool = False
) -> tuple[Any, Any]:
    """(y, rstd) of RMSNorm over the last dim of x [rows, D], for ``supported``
    D and dtype."""
    rows, d = x.shape
    n, steps, t, block, stat = _specs(rows, d, x.dtype)
    in_specs, args = [block], [x]
    if weight is not None:
        in_specs.append(pl.BlockSpec((1, d), lambda i: (0, 0)))
        args.append(weight.reshape(1, d))
    with jax.enable_x64(False):
        y, rstd = pl.pallas_call(
            functools.partial(
                _fwd_kernel, eps=eps, silu=silu, weighted=weight is not None
            ),
            name="rmsnorm_fwd",
            out_shape=(
                jax.ShapeDtypeStruct((rows, d), x.dtype),
                jax.ShapeDtypeStruct((steps, t, _LANES), jnp.float32),
            ),
            grid=(steps,),
            in_specs=in_specs,
            out_specs=(block, stat),
            compiler_params=_params(),
            interpret=interpret,
        )(*args)
    return y, rstd[:, : n // _LANES, :].reshape(-1)[:rows]


def _bwd_kernel(*refs: Any, silu: bool, weighted: bool, rows: int) -> None:
    if weighted:
        grad_ref, x_ref, rstd_ref, w_ref, dx_ref, dw_ref = refs
        w = w_ref[...].astype(jnp.float32)
    else:
        grad_ref, x_ref, rstd_ref, dx_ref = refs
        w = dw_ref = None
    n, d = x_ref.shape
    rstd = _tile_to_column(rstd_ref[0], n)
    xhat = x_ref[...].astype(jnp.float32) * rstd
    dy = grad_ref[...].astype(jnp.float32)
    if silu:
        pre = xhat * w if w is not None else xhat
        sig = jax.nn.sigmoid(pre)
        dy = dy * (sig + pre * sig * (1.0 - sig))
    wdy = dy * w if w is not None else dy
    c1 = jnp.sum(xhat * wdy, axis=1, keepdims=True) * (1.0 / d)
    dx_ref[...] = ((wdy - xhat * c1) * rstd).astype(dx_ref.dtype)
    if dw_ref is not None:
        part = dy * xhat
        if rows % n:
            row = pl.program_id(0) * n + jax.lax.broadcasted_iota(jnp.int32, (n, d), 0)
            part = jnp.where(row < rows, part, 0.0)
        dw_ref[0] = jnp.sum(part.reshape(n // 8, 8, d), axis=0)


def rmsnorm_backward(
    grad: Any,
    x: Any,
    weight: Optional[Any],
    rstd: Any,
    silu: bool,
    interpret: bool = False,
) -> tuple[Any, Optional[Any]]:
    """(dx, dweight) for ``rmsnorm_fwd``; dweight is the f32 sum over rows,
    None without a weight."""
    rows, d = x.shape
    n, steps, t, block, stat = _specs(rows, d, x.dtype)
    per = n // _LANES
    rstd = jnp.pad(rstd, (0, steps * n - rows)).reshape(steps, per, _LANES)
    if t > per:
        rstd = jnp.pad(rstd, ((0, 0), (0, t - per), (0, 0)))
    weighted = weight is not None
    in_specs, args = [block, block, stat], [grad, x, rstd]
    out_specs = [block]
    out_shape = [jax.ShapeDtypeStruct((rows, d), x.dtype)]
    if weighted:
        in_specs.append(pl.BlockSpec((1, d), lambda i: (0, 0)))
        args.append(weight.reshape(1, d))
        out_specs.append(pl.BlockSpec((1, 8, d), lambda i: (i, 0, 0)))
        out_shape.append(jax.ShapeDtypeStruct((steps, 8, d), jnp.float32))
    with jax.enable_x64(False):
        outs = pl.pallas_call(
            functools.partial(_bwd_kernel, silu=silu, weighted=weighted, rows=rows),
            name="rmsnorm_backward",
            out_shape=tuple(out_shape),
            grid=(steps,),
            in_specs=in_specs,
            out_specs=tuple(out_specs),
            compiler_params=_params(),
            interpret=interpret,
        )(*args)
    if not weighted:
        return outs[0], None
    dx, partial = outs
    return dx, jnp.sum(partial, axis=(0, 1))
