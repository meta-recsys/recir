# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Shard-local grouped matmuls without a [batch, max_len, width] buffer."""

import os
from typing import Any, Callable, cast

import jax
import jax.numpy as jnp
import recir.ops.packed_jagged_ops  # noqa: F401
import torch
from hammer.v3.modules.mot.loss import packed_supervision  # noqa: F401
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


def _packed_position_indices(
    values: Any, position_indices: Any, time_indices: Any, offsets: Any, max_len: int
) -> tuple[Any, Any, Any]:
    groups, valid = _valid_rows(offsets, values.shape[0], max_len)
    starts = _segment_broadcast(offsets[:-1], offsets[:-1], values.shape[0])
    columns = jnp.arange(values.shape[0], dtype=jnp.int32) - starts
    columns = jnp.clip(columns, 0, max_len - 1)
    return position_indices[groups, columns], time_indices[groups, columns], valid


def _packed_position_add(
    values: Any,
    position: Any,
    time: Any,
    position_indices: Any,
    time_indices: Any,
    offsets: Any,
    max_len: int,
) -> Any:
    if values.shape[0] == 0 or max_len == 0 or position_indices.shape[0] == 0:
        return jnp.zeros_like(values)
    pi, ti, valid = _packed_position_indices(
        values, position_indices, time_indices, offsets, max_len
    )
    embedding = (time[ti] + position[pi]).astype(values.dtype)
    out = jnp.where(valid[:, None], values + embedding, 0)
    return jnp.where(out == 0, jnp.zeros_like(out), out)


def _packed_position_add_backward(
    grad: Any,
    values: Any,
    position: Any,
    time: Any,
    position_indices: Any,
    time_indices: Any,
    offsets: Any,
    max_len: int,
) -> tuple[Any, Any, Any]:
    if values.shape[0] == 0 or max_len == 0 or position_indices.shape[0] == 0:
        return jnp.zeros_like(values), jnp.zeros_like(position), jnp.zeros_like(time)
    pi, ti, valid = _packed_position_indices(
        values, position_indices, time_indices, offsets, max_len
    )
    dv = jnp.where(valid[:, None], grad.astype(values.dtype), 0)
    dv = jnp.where(dv == 0, jnp.zeros_like(dv), dv)
    position_order = jnp.argsort(pi, stable=True)
    time_order = jnp.argsort(ti, stable=True)
    dp = jax.ops.segment_sum(
        dv[position_order].astype(position.dtype),
        pi[position_order],
        num_segments=position.shape[0],
        indices_are_sorted=True,
    )
    dt = jax.ops.segment_sum(
        dv[time_order].astype(time.dtype),
        ti[time_order],
        num_segments=time.shape[0],
        indices_are_sorted=True,
    )
    return dv, dp, dt


def _shard_position(
    values: Any,
    position: Any,
    time: Any,
    position_indices: Any,
    time_indices: Any,
    offsets: Any,
    max_len: int,
    grad: Any = None,
) -> Any:
    if values.shape[0] >= 2**31 or position_indices.size >= 2**31:
        raise ValueError("Packed position indices must fit int32")
    world_size = len(jax.devices())
    if world_size == 1:
        if grad is None:
            return _packed_position_add(
                values, position, time, position_indices, time_indices, offsets, max_len
            )
        return _packed_position_add_backward(
            grad,
            values,
            position,
            time,
            position_indices,
            time_indices,
            offsets,
            max_len,
        )
    if values.shape[0] % world_size or position_indices.shape[0] % world_size:
        raise ValueError("Packed position capacity and batch must divide world size")
    token_spec, index_spec, table_spec = (
        P("torch_dist", None),
        P("torch_dist", None),
        P(),
    )

    def local(v: Any, p: Any, t: Any, pi: Any, ti: Any, sizes: Any, *g: Any) -> Any:
        off = jnp.concatenate((jnp.zeros((1,), jnp.int32), jnp.cumsum(sizes)))
        if not g:
            return _packed_position_add(v, p, t, pi, ti, off, max_len)
        dv, dp, dt = _packed_position_add_backward(g[0], v, p, t, pi, ti, off, max_len)
        return dv, jax.lax.psum(dp, "torch_dist"), jax.lax.psum(dt, "torch_dist")

    backward = grad is not None
    mapped = cast(
        Callable[..., Any],
        jax.shard_map(
            local,
            mesh=Mesh(jax.devices(), ("torch_dist",)),
            in_specs=(
                token_spec,
                table_spec,
                table_spec,
                index_spec,
                index_spec,
                P("torch_dist"),
                *([token_spec] if backward else []),
            ),
            out_specs=(token_spec, table_spec, table_spec) if backward else token_spec,
            check_vma=False,
        ),
    )
    with jax.enable_x64(False):
        return mapped(
            values,
            position,
            time,
            position_indices.astype(jnp.int32),
            time_indices.astype(jnp.int32),
            jnp.diff(offsets.astype(jnp.int32)),
            *([grad] if backward else []),
        )


@op(torch.ops.recir.packed_position_add.default)
def _recir_packed_position_add(
    values: Any,
    position: Any,
    time: Any,
    position_indices: Any,
    time_indices: Any,
    offsets: Any,
    max_len: int,
) -> Any:
    return _shard_position(
        values, position, time, position_indices, time_indices, offsets, max_len
    )


@op(torch.ops.recir.packed_position_add_backward.default)
def _recir_packed_position_add_backward(
    grad: Any,
    values: Any,
    position: Any,
    time: Any,
    position_indices: Any,
    time_indices: Any,
    offsets: Any,
    max_len: int,
) -> Any:
    return _shard_position(
        values, position, time, position_indices, time_indices, offsets, max_len, grad
    )


def _local_supervision_gather(
    values: Any, lengths: Any, indices: Any, max_len: int, owner: Any, batch_size: int
) -> Any:
    two_words = values.ndim == 2
    if values.shape[0] == 0 or max_len == 0:
        return jnp.zeros(
            (*indices.shape, 2) if two_words else indices.shape, jnp.uint32
        )
    global_rows = indices // max_len
    columns = indices % max_len
    rows = jnp.clip(global_rows - owner * batch_size, 0, batch_size - 1)
    offsets = jnp.concatenate((jnp.zeros((1,), lengths.dtype), jnp.cumsum(lengths)))
    source = jnp.clip(offsets[rows] + columns, 0, values.shape[0] - 1)
    valid = (
        (global_rows >= owner * batch_size)
        & (global_rows < (owner + 1) * batch_size)
        & (columns < lengths[rows])
    )
    return jnp.where(
        valid[:, None] if two_words else valid, values[source], jnp.uint32(0)
    )


@op(torch.ops.hammer.packed_supervision_gather.default)
def _recir_packed_supervision_gather(
    values: Any, offsets: Any, indices: Any, max_len: int
) -> Any:
    world_size = len(jax.devices())
    batch_size = (offsets.shape[0] - 1) // world_size
    if max_len * (offsets.shape[0] - 1) >= 2**31 or values.shape[0] >= 2**31:
        raise ValueError("Packed supervision indices must fit int32")
    if (offsets.shape[0] - 1) % world_size or values.shape[0] % world_size:
        raise ValueError("Packed supervision capacity and batch must divide world size")
    if batch_size == 0 or values.shape[0] == 0 or max_len == 0:
        return jnp.zeros(indices.shape, values.dtype)
    bits_dtype = {1: jnp.uint8, 2: jnp.uint16, 4: jnp.uint32, 8: jnp.uint32}[
        values.dtype.itemsize
    ]
    words = jax.lax.bitcast_convert_type(values, bits_dtype).astype(jnp.uint32)

    def local(x: Any, lengths: Any, flat: Any) -> Any:
        owner = jax.lax.axis_index("torch_dist")
        result = _local_supervision_gather(x, lengths, flat, max_len, owner, batch_size)
        return jax.lax.psum(result, "torch_dist")

    mapped = cast(
        Callable[..., Any],
        jax.shard_map(
            local,
            mesh=Mesh(jax.devices(), ("torch_dist",)),
            in_specs=(
                P("torch_dist", None) if words.ndim == 2 else P("torch_dist"),
                P("torch_dist"),
                P(),
            ),
            out_specs=P(),
            check_vma=False,
        ),
    )
    with jax.enable_x64(False):
        result = mapped(
            words, jnp.diff(offsets.astype(jnp.int32)), indices.astype(jnp.int32)
        )
    result = jax.lax.optimization_barrier(result.astype(bits_dtype))
    return jax.lax.bitcast_convert_type(result, values.dtype)
