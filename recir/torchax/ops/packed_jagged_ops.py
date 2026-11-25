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


def select_concat_split_right(
    left: Any,
    ca: Any,
    right: Any,
    cb: Any,
    capacity: int,
    sa: Any,
    sb: Any,
    query_capacity: int,
    left_capacity: int | None = None,
) -> Any:
    left_prefix = left_capacity is not None
    left_capacity = left.shape[0] if left_capacity is None else left_capacity
    if (
        max(capacity, query_capacity, left.shape[0], left_capacity + right.shape[0])
        >= 1 << 31
    ):
        raise ValueError("Jagged query composition requires int32 capacities")
    shape = (query_capacity,) + left.shape[1:]
    if capacity == 0 or left_capacity + right.shape[0] == 0 or ca.shape[0] <= 1:
        return jnp.zeros(shape, jnp.result_type(left, right))
    positions = jnp.arange(query_capacity, dtype=jnp.int32)
    group = jnp.clip(
        jnp.searchsorted(sb, positions, side="right", method="scan_unrolled") - 1,
        0,
        sb.shape[0] - 2,
    )
    selected_row = jnp.clip(positions + sa[group + 1], 0, capacity - 1)
    combined_offsets = ca + cb
    group = jnp.clip(
        jnp.searchsorted(
            combined_offsets, selected_row, side="right", method="scan_unrolled"
        )
        - 1,
        0,
        ca.shape[0] - 2,
    )
    intra = selected_row - combined_offsets[group]
    left_length = ca[group + 1] - ca[group]
    source = jnp.where(
        intra < left_length,
        ca[group] + intra,
        left_capacity + cb[group] + intra - left_length,
    )
    source = jnp.clip(source, 0, left_capacity + right.shape[0] - 1)
    from_left = source < left_capacity
    left_position = jnp.clip(source, 0, max(0, left_capacity - 1))
    left_valid = jnp.ones_like(from_left)
    if left_prefix:
        group = jnp.clip(
            jnp.searchsorted(ca, left_position, side="right", method="scan_unrolled")
            - 1,
            0,
            ca.shape[0] - 2,
        )
        left_source = jnp.clip(left_position + cb[group], 0, max(0, left.shape[0] - 1))
        left_valid = left_position < ca[-1]
    else:
        left_source = jnp.clip(left_position, 0, max(0, left.shape[0] - 1))
    left_values = left[left_source] if left.shape[0] else jnp.zeros(shape, left.dtype)
    right_values = (
        right[jnp.clip(source - left_capacity, 0, right.shape[0] - 1)]
        if right.shape[0]
        else jnp.zeros(shape, right.dtype)
    )
    mask_shape = (query_capacity,) + (1,) * (left_values.ndim - 1)
    value = jnp.where(from_left.reshape(mask_shape), left_values, right_values)
    valid = (
        (positions < sb[-1])
        & (selected_row < combined_offsets[-1])
        & (~from_left | left_valid)
    )
    return jnp.where(valid.reshape(mask_shape), value, 0)


@op(torch.ops.recir.jagged_concat_split_right.default)
def _jagged_concat_split_right(
    left: Any,
    ca: Any,
    right: Any,
    cb: Any,
    capacity: int,
    sa: Any,
    sb: Any,
    query_capacity: int,
    left_capacity: int,
    left_is_prefix: bool,
) -> Any:
    world_size = len(jax.devices())
    if any(offsets.shape != ca.shape for offsets in (cb, sa, sb)):
        raise ValueError("Jagged query offsets must have the same batch size")
    if world_size == 1:
        return select_concat_split_right(
            left,
            ca,
            right,
            cb,
            capacity,
            sa,
            sb,
            query_capacity,
            left_capacity if left_is_prefix else None,
        )
    if any(
        size % world_size
        for size in (
            left.shape[0],
            right.shape[0],
            capacity,
            query_capacity,
            left_capacity,
        )
    ):
        raise ValueError("Jagged query capacities must be divisible by world size")
    mesh = Mesh(jax.devices(), (_DP_AXIS,))
    values_spec = P(*([_DP_AXIS] + [None] * (left.ndim - 1)))

    def per_device(
        values_left: Any, lc: Any, values_right: Any, rc: Any, ls: Any, rs: Any
    ) -> Any:
        offsets = [
            jnp.concatenate((jnp.zeros(1, jnp.int32), jnp.cumsum(lengths))).astype(
                jnp.int32
            )
            for lengths in (lc, rc, ls, rs)
        ]
        return select_concat_split_right(
            values_left,
            offsets[0],
            values_right,
            offsets[1],
            capacity // world_size,
            offsets[2],
            offsets[3],
            query_capacity // world_size,
            left_capacity // world_size if left_is_prefix else None,
        )

    mapped = cast(
        Callable[..., Any],
        jax.shard_map(
            per_device,
            mesh=mesh,
            in_specs=(
                values_spec,
                P(_DP_AXIS),
                values_spec,
                P(_DP_AXIS),
                P(_DP_AXIS),
                P(_DP_AXIS),
            ),
            out_specs=values_spec,
            check_vma=False,
        ),
    )
    return mapped(left, jnp.diff(ca), right, jnp.diff(cb), jnp.diff(sa), jnp.diff(sb))


def _valid_rows(offsets: Any, capacity: int, max_len: int) -> tuple[Any, Any]:
    positions = jnp.arange(capacity, dtype=jnp.int32)
    groups = _segment_id_by_cumsum(offsets[1:-1], capacity)
    groups = jnp.minimum(groups, offsets.shape[0] - 2)
    starts = _segment_broadcast(offsets[:-1], offsets[:-1], capacity)
    valid = (positions < offsets[-1]) & (positions - starts < max_len)
    return groups, valid


def _pad_rows(x: Any) -> Any:
    return jnp.pad(x, ((0, (-x.shape[0]) % 128), (0, 0)))


def _native_pair_forward_enabled() -> bool:
    implementation = os.environ.get("RECIR_PACKED_FORWARD_IMPL", "gmm")
    if implementation not in (
        "gmm",
        "native_pair",
        "native_pair_slice",
        "native_pair_dma",
        "native_pair_prefix_dma",
    ):
        raise ValueError(f"Invalid RECIR_PACKED_FORWARD_IMPL: {implementation!r}")
    return implementation != "gmm"


def _packed_bmm_native_pair(
    values: Any,
    weights: Any,
    bias: Any,
    offsets: Any,
    max_len: int,
    interpret: bool = False,
) -> Any:
    groups, k, n = weights.shape
    capacity = values.shape[0]
    if groups == 0 or capacity == 0 or max_len == 0:
        return jnp.zeros((capacity, n), jnp.float32)
    extra_group = groups % 2
    starts = offsets[:-1].astype(jnp.int32)
    sizes = jnp.diff(offsets).astype(jnp.int32)
    if extra_group:
        starts = jnp.concatenate((starts, offsets[-1:].astype(jnp.int32)))
        sizes = jnp.pad(sizes, (0, 1))
    implementation = os.environ.get("RECIR_PACKED_FORWARD_IMPL")
    use_prefix_read = implementation == "native_pair_prefix_dma"
    use_dma = implementation in ("native_pair_dma", "native_pair_prefix_dma")
    read_rows = ((max_len + 7 + 127) // 128) * 128 if use_dma else max_len
    if use_prefix_read:
        read_rows = max(read_rows, 520)
    x = jnp.pad(values, ((0, read_rows), (0, (-k) % 128 if use_dma else 0)))
    w = jnp.pad(weights, ((0, extra_group), (0, 0), (0, 0)))
    b = jnp.pad(bias, ((0, extra_group), (0, 0)))
    columns = jnp.arange(max_len, dtype=jnp.int32)
    use_slice = implementation == "native_pair_slice"
    store_impl = os.environ.get("RECIR_PACKED_STORE_IMPL", "full")
    if store_impl not in ("full", "prefix_dma"):
        raise ValueError(f"Invalid RECIR_PACKED_STORE_IMPL: {store_impl!r}")
    use_prefix = store_impl == "prefix_dma" and n % 128 == 0

    def project(pair: Any, out: Any) -> Any:
        first = pair * 2
        row_starts = jax.lax.dynamic_slice(starts, (first,), (2,))
        lengths = jax.lax.dynamic_slice(sizes, (first,), (2,))
        source = row_starts[:, None] + columns[None, :]
        valid = (columns[None, :] < lengths[:, None]) & (source < capacity)
        if use_prefix_read:
            pair_values = _packed_native_pair_prefix_dma_read(
                x, row_starts, lengths, max_len, k, capacity, interpret
            ).reshape(2, max_len, k)
        elif use_dma:
            pair_values = _packed_native_pair_dma_read(
                x, row_starts, max_len, k, interpret
            ).reshape(2, max_len, k)
        elif use_slice:
            pair_values = jnp.stack(
                (
                    jax.lax.dynamic_slice(x, (row_starts[0], 0), (max_len, k)),
                    jax.lax.dynamic_slice(x, (row_starts[1], 0), (max_len, k)),
                )
            )
            pair_values = jax.lax.optimization_barrier(
                pair_values.reshape(2 * max_len, k)
            ).reshape(2, max_len, k)
        else:
            pair_values = x[source]
        lhs = jnp.where(valid[:, :, None], pair_values, 0).astype(jnp.float32)
        rhs = jax.lax.dynamic_slice(w, (first, 0, 0), (2, k, n)).astype(jnp.float32)
        pair_bias = jax.lax.dynamic_slice(b, (first, 0), (2, n))
        projected = lhs @ rhs + pair_bias[:, None, :].astype(jnp.float32)
        projected = jnp.where(valid[:, :, None], projected, 0)
        if use_prefix:
            return _packed_native_pair_prefix_store(
                projected, out, row_starts, lengths, capacity, interpret
            )
        out = jax.lax.dynamic_update_slice(out, projected[0], (row_starts[0], 0))
        return jax.lax.dynamic_update_slice(out, projected[1], (row_starts[1], 0))

    with jax.enable_x64(False):
        out = jax.lax.fori_loop(
            0,
            (groups + 1) // 2,
            project,
            jnp.zeros(
                (capacity + max(max_len, 128 if use_prefix else 0), n), jnp.float32
            ),
        )[:capacity]
    return jnp.where(out == 0, jnp.zeros_like(out), out)


def _packed_native_pair_dma_read(
    values: Any, starts: Any, max_len: int, k: int, interpret: bool = False
) -> Any:
    rows = ((max_len + 7 + 127) // 128) * 128
    kt = 128

    def kernel(
        off: Any,
        xx: Any,
        out: Any,
        tile: Any,
        pair: Any,
        flat: Any,
        read_sem: Any,
        write_sem: Any,
    ) -> None:
        column = pl.program_id(0) * kt

        def user(group: Any, unused: Any) -> Any:
            start = off[group]
            aligned = pl.multiple_of(start // 8 * 8, 8)
            shift = start - aligned
            copy = pltpu.async_copy(
                xx.at[pl.ds(aligned, rows), pl.ds(column, kt)], tile, read_sem
            )
            copy.wait()
            rotation = jnp.where(shift == 0, 0, rows - shift)
            if values.dtype == jnp.bfloat16:
                bits = jax.lax.bitcast_convert_type(tile[...], jnp.uint16).astype(
                    jnp.uint32
                )
                rotated = pltpu.roll(bits, rotation, axis=0).astype(jnp.uint16)
                selected = jax.lax.bitcast_convert_type(rotated, jnp.bfloat16)
            else:
                selected = pltpu.roll(tile[...], rotation, axis=0)
            pair.at[group, :, :][...] = selected[:max_len]
            return unused

        jax.lax.fori_loop(0, 2, user, jnp.int32(0))
        flat[...] = pair[...].reshape(2 * max_len, kt)
        write = pltpu.async_copy(flat, out.at[:, pl.ds(column, kt)], write_sem)
        write.wait()

    result = pl.pallas_call(
        kernel,
        name="packed_native_pair_dma_read",
        out_shape=jax.ShapeDtypeStruct((2 * max_len, values.shape[1]), values.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=((k + kt - 1) // kt,),
            in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[
                pltpu.VMEM((rows, kt), values.dtype),
                pltpu.VMEM((2, max_len, kt), values.dtype),
                pltpu.VMEM((2 * max_len, kt), values.dtype),
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
            ],
        ),
        interpret=interpret,
    )(starts.astype(jnp.int32), values)
    return result[:, :k]


def _packed_native_pair_prefix_dma_read(
    values: Any,
    starts: Any,
    sizes: Any,
    max_len: int,
    k: int,
    capacity: int,
    interpret: bool = False,
) -> Any:
    block_rows = 512
    read_rows = 520
    pair_rows = ((max_len + block_rows - 1) // block_rows) * block_rows
    kt = 128
    bitdtype = jnp.uint16 if values.dtype == jnp.bfloat16 else jnp.uint32
    raw_values = jax.lax.bitcast_convert_type(values, bitdtype)

    def kernel(
        off: Any,
        lengths: Any,
        xx: Any,
        out: Any,
        tile: Any,
        pair: Any,
        flat: Any,
        read_sem: Any,
        write_sem: Any,
    ) -> None:
        column = pl.program_id(0) * kt
        pair[...] = jnp.zeros((2, pair_rows, kt), bitdtype)

        def user(group: Any, unused: Any) -> Any:
            start = off[group]
            length = jnp.maximum(
                0, jnp.minimum(lengths[group], jnp.minimum(max_len, capacity - start))
            )
            aligned = pl.multiple_of(start // 8 * 8, 8)
            shift = start - aligned
            count = (length + block_rows - 1) // block_rows

            def block(index: Any, carry: Any) -> Any:
                source = aligned + index * block_rows
                copy = pltpu.async_copy(
                    xx.at[pl.ds(source, read_rows), pl.ds(column, kt)], tile, read_sem
                )
                copy.wait()
                rotation = jnp.where(shift == 0, 0, read_rows - shift)
                bits = tile[...].astype(jnp.uint32)
                rotated = pltpu.roll(bits, rotation, axis=0)[:block_rows]
                position = index * block_rows + jax.lax.broadcasted_iota(
                    jnp.int32, (block_rows, kt), 0
                )
                selected_bits = jnp.where(
                    position < length, rotated, jnp.zeros((block_rows, kt), jnp.uint32)
                )
                selected = selected_bits.astype(bitdtype)
                pair.at[group, pl.ds(index * block_rows, block_rows), :][...] = selected
                return carry

            jax.lax.fori_loop(0, count, block, jnp.int32(0))
            return unused

        jax.lax.fori_loop(0, 2, user, jnp.int32(0))
        flat[...] = pair[:, :max_len, :].reshape(2 * max_len, kt)
        write = pltpu.async_copy(flat, out.at[:, pl.ds(column, kt)], write_sem)
        write.wait()

    result = pl.pallas_call(
        kernel,
        name="packed_native_pair_prefix_read",
        out_shape=jax.ShapeDtypeStruct((2 * max_len, values.shape[1]), bitdtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=((k + kt - 1) // kt,),
            in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[
                pltpu.VMEM((read_rows, kt), bitdtype),
                pltpu.VMEM((2, pair_rows, kt), bitdtype),
                pltpu.VMEM((2 * max_len, kt), bitdtype),
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
            ],
        ),
        interpret=interpret,
    )(starts.astype(jnp.int32), sizes.astype(jnp.int32), raw_values)
    if k % 128 == 0:
        pair_bits = jax.lax.optimization_barrier(result.reshape(2, max_len, k))
        return jax.lax.bitcast_convert_type(pair_bits, values.dtype).reshape(
            2 * max_len, k
        )
    return jax.lax.bitcast_convert_type(result[:, :k], values.dtype)


def _packed_native_pair_prefix_store(
    projected: Any,
    existing: Any,
    starts: Any,
    sizes: Any,
    capacity: int,
    interpret: bool = False,
) -> Any:
    _, max_len, n = projected.shape
    rows = ((max_len + 7 + 127) // 128) * 128 + 8
    projected = jnp.pad(projected, ((0, 0), (0, rows - max_len), (0, 0)))

    def kernel(
        off: Any,
        lengths: Any,
        xx: Any,
        old: Any,
        out: Any,
        tile: Any,
        previous: Any,
        result: Any,
        read_sem: Any,
        old_sem: Any,
        write_sem: Any,
    ) -> None:
        def user(group: Any, unused: Any) -> Any:
            start = off[group]
            length = jnp.maximum(
                0, jnp.minimum(lengths[group], jnp.minimum(max_len, capacity - start))
            )
            aligned = pl.multiple_of(start // 8 * 8, 8)
            shift = start - aligned
            count = jnp.where(length > 0, (length + shift + 127) // 128, 0)

            def block(index: Any, carry: Any) -> Any:
                destination = aligned + index * 128
                source = jnp.maximum(0, index * 128 - jnp.where(shift > 0, 8, 0))
                source = pl.multiple_of(source, 8)
                copy = pltpu.async_copy(
                    xx.at[group, pl.ds(source, 136), :], tile, read_sem
                )
                copy.wait()
                rotation = jnp.where(
                    index == 0, shift, jnp.where(shift > 0, 136 - (8 - shift), 0)
                )
                selected = pltpu.roll(tile[...], rotation, axis=0)[:128]
                position = (
                    index * 128
                    + jax.lax.broadcasted_iota(jnp.int32, (128, n), 0)
                    - shift
                )
                valid = (position >= 0) & (position < length)

                def edge() -> None:
                    read = pltpu.async_copy(
                        out.at[pl.ds(destination, 128), :], previous, old_sem
                    )
                    read.wait()
                    result[...] = jnp.where(valid, selected, previous[...])

                def middle() -> None:
                    result[...] = selected

                full = (index * 128 >= shift) & ((index + 1) * 128 <= shift + length)
                jax.lax.cond(full, middle, edge)
                write = pltpu.async_copy(
                    result, out.at[pl.ds(destination, 128), :], write_sem
                )
                write.wait()
                return carry

            jax.lax.fori_loop(0, count, block, jnp.int32(0))
            return unused

        jax.lax.fori_loop(0, 2, user, jnp.int32(0))

    return pl.pallas_call(
        kernel,
        name="packed_native_pair_prefix_store",
        out_shape=jax.ShapeDtypeStruct(existing.shape, existing.dtype),
        input_output_aliases={3: 0},
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(1,),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[
                pltpu.VMEM((136, n), jnp.float32),
                pltpu.VMEM((128, n), jnp.float32),
                pltpu.VMEM((128, n), jnp.float32),
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
            ],
        ),
        interpret=interpret,
    )(starts.astype(jnp.int32), sizes.astype(jnp.int32), projected, existing)


def _packed_bmm(
    values: Any, weights: Any, offsets: Any, max_len: int, interpret: bool = False
) -> Any:
    if _native_pair_forward_enabled():
        return _packed_bmm_native_pair(
            values,
            weights,
            jnp.zeros((weights.shape[0], weights.shape[2]), jnp.float32),
            offsets,
            max_len,
            interpret,
        )
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
    if _native_pair_forward_enabled():
        return _packed_bmm_native_pair(
            values, weights, bias, offsets, max_len, interpret
        )
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
    reduction = os.environ.get("RECIR_PACKED_WEIGHT_GRAD_REDUCTION", "tgmm")
    if reduction == "aligned_full_k":
        return _packed_weight_grad_aligned(
            values, grad, offsets, max_len, dtype, interpret
        )
    if reduction != "tgmm":
        raise ValueError(f"Invalid RECIR_PACKED_WEIGHT_GRAD_REDUCTION: {reduction!r}")
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


def _packed_weight_grad_aligned(
    values: Any, grad: Any, offsets: Any, max_len: int, dtype: Any, interpret: bool
) -> Any:
    groups = offsets.shape[0] - 1
    k, n = values.shape[1], grad.shape[1]
    if max_len == 0 or groups == 0 or values.shape[0] == 0:
        return jnp.zeros((groups, k, n), dtype)
    if values.dtype == jnp.bfloat16:
        grad = grad.astype(jnp.bfloat16)
    input_dtype = common.select_input_dtype(values, grad)
    rows = ((max_len + 7 + 127) // 128) * 128
    tm, tn = min(256, ((k + 127) // 128) * 128), min(256, ((n + 127) // 128) * 128)
    k_pad, n_pad = (-k) % tm, (-n) % tn
    x = jnp.pad(values, ((0, rows), (0, k_pad)))
    g = jnp.pad(grad, ((0, rows), (0, n_pad)))

    def kernel(
        off: Any,
        xx: Any,
        gg: Any,
        out: Any,
        x_tile: Any,
        g_tile: Any,
        x_sem: Any,
        g_sem: Any,
    ) -> None:
        ki, ni, group = pl.program_id(0), pl.program_id(1), pl.program_id(2)
        start = off[group]
        aligned = pl.multiple_of((start // 8) * 8, 8)
        shift = start - aligned
        rotation = jnp.where(shift == 0, 0, rows - shift)
        length = jnp.minimum(off[group + 1] - start, max_len)
        x_copy = pltpu.async_copy(
            xx.at[pl.ds(aligned, rows), pl.ds(ki * tm, tm)], x_tile, x_sem
        )
        g_copy = pltpu.async_copy(
            gg.at[pl.ds(aligned, rows), pl.ds(ni * tn, tn)], g_tile, g_sem
        )
        x_copy.wait()
        g_copy.wait()
        valid = jnp.arange(rows)[:, None] < length
        lhs = jnp.where(
            valid, pltpu.roll(x_tile[...].astype(jnp.float32), rotation, axis=0), 0
        )
        rhs = jnp.where(
            valid, pltpu.roll(g_tile[...].astype(jnp.float32), rotation, axis=0), 0
        )
        out[...] = jax.lax.dot_general(
            lhs[:max_len].astype(input_dtype),
            rhs[:max_len].astype(input_dtype),
            (((0,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        ).astype(dtype)

    def output_index(ki: Any, ni: Any, group: Any, off: Any) -> Any:
        return group, ki, ni

    with jax.enable_x64(False):
        result = pl.pallas_call(
            kernel,
            name="packed_weight_grad_aligned",
            out_shape=jax.ShapeDtypeStruct((groups, k + k_pad, n + n_pad), dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                grid=((k + k_pad) // tm, (n + n_pad) // tn, groups),
                in_specs=[
                    pl.BlockSpec(memory_space=pltpu.HBM),
                    pl.BlockSpec(memory_space=pltpu.HBM),
                ],
                out_specs=pl.BlockSpec((None, tm, tn), output_index),
                scratch_shapes=[
                    pltpu.VMEM((rows, tm), values.dtype),
                    pltpu.VMEM((rows, tn), grad.dtype),
                    pltpu.SemaphoreType.DMA,
                    pltpu.SemaphoreType.DMA,
                ],
            ),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel", "parallel", "parallel")
            ),
            interpret=interpret,
        )(offsets.astype(jnp.int32), x, g)
    return result[:, :k, :n]


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
    # TPU gathers a two-element row far faster than two scalars (0.4 ms vs 19 ms
    # for 328k tokens), so fetch each token's (position, time) pair at once.
    pairs = jnp.stack(
        (position_indices.astype(jnp.int32), time_indices.astype(jnp.int32)), axis=-1
    )
    picked = pairs[groups, columns]
    return picked[:, 0], picked[:, 1], valid


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
    # Sorting (index, row) pairs yields the sorted indices with the order, which
    # spares the scalar gathers index[order] (2.4 ms each on 328k tokens).
    rows = jnp.arange(pi.shape[0], dtype=jnp.int32)
    sorted_pi, position_order = jax.lax.sort((pi, rows), num_keys=1, is_stable=True)
    sorted_ti, time_order = jax.lax.sort((ti, rows), num_keys=1, is_stable=True)
    dp = jax.ops.segment_sum(
        dv[position_order].astype(position.dtype),
        sorted_pi,
        num_segments=position.shape[0],
        indices_are_sorted=True,
    )
    dt = jax.ops.segment_sum(
        dv[time_order].astype(time.dtype),
        sorted_ti,
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
    starts = jnp.cumsum(lengths) - lengths
    # One gather of (start, length) pairs instead of two scalar gathers.
    row_start, row_length = jnp.moveaxis(
        jnp.stack((starts, lengths), axis=-1)[rows], -1, 0
    )
    source = jnp.clip(row_start + columns, 0, values.shape[0] - 1)
    valid = (
        (global_rows >= owner * batch_size)
        & (global_rows < (owner + 1) * batch_size)
        & (columns < row_length)
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
