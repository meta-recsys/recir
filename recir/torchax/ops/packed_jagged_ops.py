# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Shard-local grouped matmuls without a [batch, max_len, width] buffer."""

from typing import Any, Callable, cast

import jax
import jax.numpy as jnp
import recir.ops.packed_jagged_ops  # noqa: F401
import torch
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas.ops.tpu.megablox import common
from jax.experimental.pallas.ops.tpu.megablox.gmm import (
    _get_store_mask,
    gmm,
    make_group_metadata,
    tgmm,
)
from jax.sharding import Mesh, PartitionSpec as P
from recir.torchax.ops.decorators import op
from recir.torchax.ops.jagged_ops import (
    _DP_AXIS,
    _segment_broadcast,
    _segment_id_by_cumsum,
)


def _valid_rows(offsets: Any, capacity: int, max_len: int) -> tuple[Any, Any]:
    positions = jnp.arange(capacity, dtype=jnp.int32)
    groups = _segment_id_by_cumsum(offsets[1:-1], capacity)
    groups = jnp.minimum(groups, offsets.shape[0] - 2)
    starts = _segment_broadcast(offsets[:-1], offsets[:-1], capacity)
    valid = (positions < offsets[-1]) & (positions - starts < max_len)
    return groups, valid


def _pad_rows(x: Any) -> Any:
    return jnp.pad(x, ((0, (-x.shape[0]) % 128), (0, 0)))


def _packed_bmm(
    values: Any, weights: Any, offsets: Any, max_len: int, interpret: bool = False
) -> Any:
    if values.shape[0] == 0:
        return jnp.zeros((0, weights.shape[2]), jnp.float32)
    _, valid = _valid_rows(offsets, values.shape[0], max_len)
    lhs = _pad_rows(jnp.where(valid[:, None], values, 0))
    sizes = jnp.diff(offsets).astype(jnp.int32)
    with jax.enable_x64(False):
        out = gmm(
            lhs,
            weights,
            sizes,
            preferred_element_type=jnp.float32,
            tiling=(
                128,
                ((values.shape[1] + 127) // 128) * 128,
                min(512, ((weights.shape[2] + 127) // 128) * 128),
            ),
            existing_out=jnp.zeros((lhs.shape[0], weights.shape[2]), jnp.float32),
            interpret=interpret,
        )[: values.shape[0]]
    return jnp.where(valid[:, None], out, 0)


def _gmm_with_bias(lhs: Any, rhs: Any, bias: Any, sizes: Any, interpret: bool) -> Any:
    m, k = lhs.shape
    n = rhs.shape[2]
    tm, tk, tn = 128, ((k + 127) // 128) * 128, min(512, ((n + 127) // 128) * 128)
    metadata, active_tiles = make_group_metadata(
        group_sizes=sizes,
        m=m,
        tm=tm,
        start_group=jnp.array(0, jnp.int32),
        num_nonzero_groups=rhs.shape[0],
        visit_empty_groups=False,
    )
    input_dtype = common.select_input_dtype(lhs, rhs)

    def kernel(
        md: Any,
        x: Any,
        w: Any,
        b: Any,
        existing: Any,
        out: Any,
        acc: Any,
    ) -> None:
        tile = pl.program_id(1)
        previous = jnp.maximum(tile - 1, 0)

        @pl.when((tile == 0) | (md[2][tile] != md[2][previous]))
        def initialize() -> None:
            out[...] = existing[...]

        xx, ww = x[...], w[...]
        if k % tk:
            xx = jnp.where(jnp.arange(tk)[None, :] < k, xx.astype(jnp.float32), 0)
            ww = jnp.where(jnp.arange(tk)[:, None] < k, ww.astype(jnp.float32), 0)
        acc[...] = jnp.zeros((tm, tn), jnp.float32)
        acc[...] += jax.lax.dot_general(
            xx.astype(input_dtype),
            ww.astype(input_dtype),
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        projected = acc[...] + b[...][0, :][None, :].astype(jnp.float32)
        mask = _get_store_mask(grid_id=tile, group_metadata=md, tm=tm, tn=tn)
        out[...] = jax.lax.select(mask, projected, out[...])

    def lhs_index(n_i: Any, tile: Any, md: Any) -> tuple[Any, int]:
        return md[2][tile], 0

    def rhs_index(n_i: Any, tile: Any, md: Any) -> tuple[Any, int, Any]:
        return md[1][tile], 0, n_i

    def bias_index(n_i: Any, tile: Any, md: Any) -> tuple[Any, int, Any]:
        return md[1][tile], 0, n_i

    def out_index(n_i: Any, tile: Any, md: Any) -> tuple[Any, Any]:
        return md[2][tile], n_i

    out_spec = pl.BlockSpec((tm, tn), out_index)
    call = pl.pallas_call(
        kernel,
        name="packed_bmm_bias",
        out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[
                pl.BlockSpec((tm, tk), lhs_index),
                pl.BlockSpec((None, tk, tn), rhs_index),
                pl.BlockSpec((None, 8, tn), bias_index),
                out_spec,
            ],
            out_specs=out_spec,
            grid=((n + tn - 1) // tn, active_tiles),
            scratch_shapes=[pltpu.VMEM((tm, tn), jnp.float32)],
        ),
        input_output_aliases={6: 0},
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary")
        ),
        interpret=interpret,
    )
    aligned_bias = jnp.pad(bias[:, None, :], ((0, 0), (0, 7), (0, 0)))
    return call(metadata, lhs, rhs, aligned_bias, jnp.zeros((m, n), jnp.float32))


def _packed_bmm_add(
    values: Any,
    weights: Any,
    bias: Any,
    offsets: Any,
    max_len: int,
    interpret: bool = False,
) -> Any:
    if values.shape[0] == 0:
        return jnp.zeros((0, weights.shape[2]), jnp.float32)
    _, valid = _valid_rows(offsets, values.shape[0], max_len)
    lhs = _pad_rows(jnp.where(valid[:, None], values, 0))
    with jax.enable_x64(False):
        out = _gmm_with_bias(
            lhs, weights, bias, jnp.diff(offsets).astype(jnp.int32), interpret
        )[: values.shape[0]]
    return jnp.where(valid[:, None], out, 0)


def _packed_bmm_backward(
    grad: Any,
    values: Any,
    weights: Any,
    offsets: Any,
    max_len: int,
    interpret: bool = False,
) -> tuple[Any, Any]:
    if values.shape[0] == 0:
        return jnp.zeros_like(values), jnp.zeros_like(weights)
    if values.dtype == weights.dtype == jnp.bfloat16:
        grad = grad.astype(jnp.bfloat16)
    _, valid = _valid_rows(offsets, values.shape[0], max_len)
    lhs = _pad_rows(jnp.where(valid[:, None], values, 0))
    grad = _pad_rows(jnp.where(valid[:, None], grad, 0))
    sizes = jnp.diff(offsets).astype(jnp.int32)
    with jax.enable_x64(False):
        dv = gmm(
            grad,
            weights,
            sizes,
            preferred_element_type=values.dtype,
            tiling=(
                128,
                ((grad.shape[1] + 127) // 128) * 128,
                min(512, ((values.shape[1] + 127) // 128) * 128),
            ),
            transpose_rhs=True,
            existing_out=jnp.zeros_like(lhs),
            interpret=interpret,
        )[: values.shape[0]]
        dw = _packed_weight_grad(
            values,
            grad[: values.shape[0]],
            offsets.astype(jnp.int32),
            max_len,
            weights.dtype,
            interpret,
        )
    return jnp.where(valid[:, None], dv, 0), dw


def _packed_weight_grad(
    values: Any,
    grad: Any,
    offsets: Any,
    max_len: int,
    dtype: Any,
    interpret: bool,
) -> Any:
    if values.dtype == jnp.bfloat16:
        grad = grad.astype(jnp.bfloat16)
    groups = offsets.shape[0] - 1
    if max_len == 0 or groups == 0 or values.shape[0] == 0:
        return jnp.zeros((groups, values.shape[1], grad.shape[1]), dtype)
    _, valid = _valid_rows(offsets, values.shape[0], max_len)
    lhs = _pad_rows(jnp.where(valid[:, None], values, 0))
    rhs = _pad_rows(jnp.where(valid[:, None], grad, 0))
    with jax.enable_x64(False):
        result = tgmm(
            lhs.T,
            rhs,
            jnp.diff(offsets).astype(jnp.int32),
            preferred_element_type=dtype,
            tiling=(128, 128, min(512, ((grad.shape[1] + 127) // 128) * 128)),
            interpret=interpret,
        )
    return result.astype(dtype)


def _packed_add(values: Any, bias: Any, offsets: Any, max_len: int) -> Any:
    groups, valid = _valid_rows(offsets, values.shape[0], max_len)
    return jnp.where(valid[:, None], values + bias[groups], 0)


def _packed_add_backward(
    grad: Any, bias: Any, offsets: Any, max_len: int
) -> tuple[Any, Any]:
    groups, valid = _valid_rows(offsets, grad.shape[0], max_len)
    dv = jnp.where(valid[:, None], grad, 0)
    db = jax.ops.segment_sum(
        dv, groups, num_segments=bias.shape[0], indices_are_sorted=True
    )
    return dv, db


def _packed_bmm_add_backward(
    grad: Any,
    values: Any,
    weights: Any,
    bias: Any,
    offsets: Any,
    max_len: int,
    interpret: bool = False,
) -> tuple[Any, Any, Any]:
    dv, dw = _packed_bmm_backward(grad, values, weights, offsets, max_len, interpret)
    _, db = _packed_add_backward(grad, bias, offsets, max_len)
    return dv, dw, db


def _shard_local(
    fn: Callable[..., Any],
    token_inputs: tuple[Any, ...],
    batch_inputs: tuple[Any, ...],
    offsets: Any,
    max_len: int,
    backward: bool = False,
) -> Any:
    world_size = len(jax.devices())
    if world_size == 1:
        return fn(*token_inputs, *batch_inputs, offsets, max_len)
    for x in token_inputs:
        if x.shape[0] % world_size:
            raise ValueError("Packed token capacity must be divisible by world size")
    mesh = Mesh(jax.devices(), ("torch_dist",))
    token_spec = P("torch_dist", None)
    batch_specs = tuple(P("torch_dist", *([None] * (x.ndim - 1))) for x in batch_inputs)
    out_spec = (token_spec, *batch_specs) if backward else token_spec

    def local(*args: Any) -> Any:
        tokens = args[: len(token_inputs)]
        batches = args[len(token_inputs) : -1]
        lengths = args[-1]
        local_offsets = jnp.concatenate(
            (jnp.zeros((1,), lengths.dtype), jnp.cumsum(lengths))
        )
        return fn(*tokens, *batches, local_offsets, max_len)

    mapped = cast(
        Callable[..., Any],
        jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(
                *([token_spec] * len(token_inputs)),
                *batch_specs,
                P("torch_dist"),
            ),
            out_specs=out_spec,
            check_vma=False,
        ),
    )
    return mapped(*token_inputs, *batch_inputs, jnp.diff(offsets))


@op(torch.ops.recir.packed_jagged_bmm.default)
def _recir_packed_bmm(values: Any, weights: Any, offsets: Any, max_len: int) -> Any:
    return _shard_local(_packed_bmm, (values,), (weights,), offsets, max_len)


@op(torch.ops.recir.packed_jagged_bmm_backward.default)
def _recir_packed_bmm_backward(
    grad: Any, values: Any, weights: Any, offsets: Any, max_len: int
) -> Any:
    return _shard_local(
        _packed_bmm_backward,
        (grad, values),
        (weights,),
        offsets,
        max_len,
        backward=True,
    )


@op(torch.ops.recir.packed_jagged_add.default)
def _recir_packed_add(values: Any, bias: Any, offsets: Any, max_len: int) -> Any:
    return _shard_local(_packed_add, (values,), (bias,), offsets, max_len)


@op(torch.ops.recir.packed_jagged_add_backward.default)
def _recir_packed_add_backward(grad: Any, bias: Any, offsets: Any, max_len: int) -> Any:
    return _shard_local(
        _packed_add_backward, (grad,), (bias,), offsets, max_len, backward=True
    )


@op(torch.ops.recir.packed_jagged_bmm_add.default)
def _recir_packed_bmm_add(
    values: Any, weights: Any, bias: Any, offsets: Any, max_len: int
) -> Any:
    return _shard_local(_packed_bmm_add, (values,), (weights, bias), offsets, max_len)


@op(torch.ops.recir.packed_jagged_bmm_add_backward.default)
def _recir_packed_bmm_add_backward(
    grad: Any, values: Any, weights: Any, bias: Any, offsets: Any, max_len: int
) -> Any:
    return _shard_local(
        _packed_bmm_add_backward,
        (grad, values),
        (weights, bias),
        offsets,
        max_len,
        backward=True,
    )
