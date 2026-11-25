# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Copy contiguous jagged segments with integer DMA tiles."""

from math import prod
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _copy_jagged_segment_bits(
    values: Any,
    source_starts: Any,
    destination_starts: Any,
    sizes: Any,
    output_bits: Any,
    output_capacity: int,
    interpret: bool = False,
) -> Any:
    capacity = values.shape[0]
    if output_capacity == 0 or capacity == 0 or sizes.shape[0] == 0:
        return output_bits
    bits_dtype = jnp.uint16 if values.dtype == jnp.bfloat16 else jnp.uint32
    source_bits = jax.lax.bitcast_convert_type(values, bits_dtype)
    width = source_bits.size // capacity
    padded_width = output_bits.shape[1]
    tile_columns = padded_width if padded_width <= 1024 else 128
    source_bits = jnp.pad(
        source_bits.reshape(capacity, width),
        ((8, 136), (0, padded_width - width)),
    )

    def kernel(
        sources: Any,
        destinations: Any,
        lengths: Any,
        source: Any,
        existing: Any,
        out: Any,
        tile: Any,
        previous: Any,
        selected: Any,
        read_sem: Any,
        previous_sem: Any,
        write_sem: Any,
    ) -> None:
        column = pl.program_id(0) * tile_columns

        def segment(group: Any, carry: Any) -> Any:
            source_start = sources[group]
            destination_start = destinations[group]
            length = jnp.maximum(
                0,
                jnp.minimum(
                    lengths[group],
                    jnp.minimum(
                        capacity - source_start, output_capacity - destination_start
                    ),
                ),
            )
            aligned_destination = pl.multiple_of(destination_start // 8 * 8, 8)
            offset = destination_start - aligned_destination
            count = jnp.where(length > 0, (offset + length + 127) // 128, 0)

            def block(index: Any, unused: Any) -> Any:
                destination = aligned_destination + index * 128
                source_position = source_start + 8 + destination - destination_start
                aligned_source = pl.multiple_of(source_position // 8 * 8, 8)
                shift = source_position - aligned_source
                copy = pltpu.async_copy(
                    source.at[pl.ds(aligned_source, 136), pl.ds(column, tile_columns)],
                    tile,
                    read_sem,
                )
                copy.wait()
                rotation = jnp.where(shift == 0, 0, 136 - shift)
                raw = pltpu.roll(tile[...].astype(jnp.uint32), rotation, axis=0)[:128]
                position = destination + jax.lax.broadcasted_iota(
                    jnp.int32, (128, tile_columns), 0
                )
                valid = (position >= destination_start) & (
                    position < destination_start + length
                )

                def edge() -> None:
                    read = pltpu.async_copy(
                        out.at[pl.ds(destination, 128), pl.ds(column, tile_columns)],
                        previous,
                        previous_sem,
                    )
                    read.wait()
                    selected[...] = jnp.where(
                        valid, raw, previous[...].astype(jnp.uint32)
                    ).astype(bits_dtype)

                def middle() -> None:
                    selected[...] = raw.astype(bits_dtype)

                full = (destination >= destination_start) & (
                    destination + 128 <= destination_start + length
                )
                jax.lax.cond(full, middle, edge)
                write = pltpu.async_copy(
                    selected,
                    out.at[pl.ds(destination, 128), pl.ds(column, tile_columns)],
                    write_sem,
                )
                write.wait()
                return unused

            jax.lax.fori_loop(0, count, block, jnp.int32(0))
            return carry

        jax.lax.fori_loop(0, sizes.shape[0], segment, jnp.int32(0))

    result = pl.pallas_call(
        kernel,
        name="jagged_segment_dma",
        out_shape=jax.ShapeDtypeStruct(output_bits.shape, bits_dtype),
        input_output_aliases={4: 0},
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=(padded_width // tile_columns,),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[
                pltpu.VMEM((136, tile_columns), bits_dtype),
                pltpu.VMEM((128, tile_columns), bits_dtype),
                pltpu.VMEM((128, tile_columns), bits_dtype),
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
            ],
        ),
        interpret=interpret,
    )(
        source_starts.astype(jnp.int32),
        destination_starts.astype(jnp.int32),
        sizes.astype(jnp.int32),
        source_bits,
        output_bits,
    )
    return result


def _segment_output_bits(output: Any) -> Any:
    bits_dtype = jnp.uint16 if output.dtype == jnp.bfloat16 else jnp.uint32
    bits = jax.lax.bitcast_convert_type(output, bits_dtype)
    width = prod(bits.shape[1:])
    capacity = output.shape[0]
    rows = (capacity + 127 + 7) // 8 * 8
    return jnp.pad(
        bits.reshape(capacity, width),
        ((0, rows - capacity), (0, (-width) % 128)),
    )


def _segment_output_from_bits(bits: Any, output: Any) -> Any:
    bits_dtype = jnp.uint16 if output.dtype == jnp.bfloat16 else jnp.uint32
    shape = jax.lax.bitcast_convert_type(output, bits_dtype).shape
    width = prod(shape[1:])
    cropped = bits[: output.shape[0], :width].reshape(shape)
    return jax.lax.bitcast_convert_type(cropped, output.dtype)


def copy_jagged_segments(
    values: Any,
    source_starts: Any,
    destination_starts: Any,
    sizes: Any,
    output: Any,
    interpret: bool = False,
) -> Any:
    bits = _copy_jagged_segment_bits(
        values,
        source_starts,
        destination_starts,
        sizes,
        _segment_output_bits(output),
        output.shape[0],
        interpret,
    )
    return _segment_output_from_bits(bits, output)


def concat_jagged_dma(
    left: Any,
    offsets_left: Any,
    right: Any,
    offsets_right: Any,
    capacity: int,
    interpret: bool = False,
) -> Any:
    output = jnp.zeros((capacity,) + left.shape[1:], left.dtype)
    bits = _segment_output_bits(output)
    starts = offsets_left[:-1] + offsets_right[:-1]
    bits = _copy_jagged_segment_bits(
        left,
        offsets_left[:-1],
        starts,
        jnp.diff(offsets_left),
        bits,
        capacity,
        interpret,
    )
    bits = _copy_jagged_segment_bits(
        right,
        offsets_right[:-1],
        offsets_left[1:] + offsets_right[:-1],
        jnp.diff(offsets_right),
        bits,
        capacity,
        interpret,
    )
    return _segment_output_from_bits(bits, output)


def split_jagged_dma(
    values: Any,
    offsets_left: Any,
    offsets_right: Any,
    capacity_left: int,
    capacity_right: int,
    interpret: bool = False,
) -> tuple[Any, Any]:
    left = copy_jagged_segments(
        values,
        offsets_left[:-1] + offsets_right[:-1],
        offsets_left[:-1],
        jnp.diff(offsets_left),
        jnp.zeros((capacity_left,) + values.shape[1:], values.dtype),
        interpret,
    )
    right = copy_jagged_segments(
        values,
        offsets_left[1:] + offsets_right[:-1],
        offsets_right[:-1],
        jnp.diff(offsets_right),
        jnp.zeros((capacity_right,) + values.shape[1:], values.dtype),
        interpret,
    )
    return left, right


def _copy_jagged_output_tiles(
    left: Any,
    right: Any,
    source_left: Any,
    source_right: Any,
    destination_left: Any,
    destination_right: Any,
    sizes_left: Any,
    sizes_right: Any,
    capacity: int,
    interpret: bool = False,
) -> Any:
    dtype = jnp.uint16 if left.dtype == jnp.bfloat16 else jnp.uint32
    left_bits = jax.lax.bitcast_convert_type(left, dtype)
    right_bits = jax.lax.bitcast_convert_type(right, dtype)
    width = prod(left_bits.shape[1:])
    padded_width = (width + 127) // 128 * 128
    columns = padded_width if padded_width <= 1024 else 128
    block_rows = 2048
    rows = (capacity + block_rows - 1) // block_rows * block_rows
    typed_output = not interpret and left.dtype in (
        jnp.bfloat16,
        jnp.float32,
        jnp.int32,
        jnp.uint32,
    )
    out_dtype = left.dtype if typed_output else dtype
    shape = (capacity,) + left_bits.shape[1:]
    if rows == 0 or sizes_left.shape[0] == 0:
        return jnp.zeros((capacity,) + left.shape[1:], left.dtype)
    cap_left, cap_right = left.shape[0], right.shape[0]
    left_bits = jnp.pad(
        left_bits.reshape(cap_left, width), ((512, 520), (0, padded_width - width))
    )
    right_bits = jnp.pad(
        right_bits.reshape(cap_right, width), ((512, 520), (0, padded_width - width))
    )

    def kernel(
        sa: Any,
        sb: Any,
        da: Any,
        db: Any,
        la: Any,
        lb: Any,
        a: Any,
        b: Any,
        out: Any,
        tile: Any,
        semaphore: Any,
    ) -> None:
        tile_start = pl.program_id(0) * block_rows
        column = pl.program_id(1) * columns
        tile_end = jnp.minimum(tile_start + block_rows, capacity)
        if typed_output:
            out = out.bitcast(dtype)
        out[...] = jnp.zeros((block_rows, columns), dtype)

        def copy_segment(
            source: Any,
            source_capacity: int,
            source_start: Any,
            destination_start: Any,
            requested: Any,
        ) -> None:
            length = jnp.maximum(
                0,
                jnp.minimum(
                    requested,
                    jnp.minimum(
                        source_capacity - source_start, capacity - destination_start
                    ),
                ),
            )
            first = jnp.maximum(destination_start, tile_start)
            end = jnp.minimum(destination_start + length, tile_end)
            first_block = (first - tile_start) // 512
            stop_block = jnp.where(
                end > first, (end - tile_start + 511) // 512, first_block
            )

            def block(index: Any, carry: Any) -> Any:
                destination = tile_start + index * 512
                source_position = source_start + 512 + destination - destination_start
                aligned_source = pl.multiple_of(source_position // 8 * 8, 8)
                shift = source_position - aligned_source
                read = pltpu.async_copy(
                    source.at[pl.ds(aligned_source, 520), pl.ds(column, columns)],
                    tile,
                    semaphore,
                )
                read.wait()
                rotation = jnp.where(shift == 0, 0, 520 - shift)
                raw = pltpu.roll(tile[...].astype(jnp.uint32), rotation, axis=0)[:512]
                position = destination + jax.lax.broadcasted_iota(
                    jnp.int32, (512, columns), 0
                )
                valid = (
                    (position >= destination_start)
                    & (position < destination_start + length)
                    & (position < tile_end)
                )
                window = out.at[pl.ds(index * 512, 512), pl.ds(0, columns)]
                window[...] = jnp.where(
                    valid, raw, window[...].astype(jnp.uint32)
                ).astype(dtype)
                return carry

            jax.lax.fori_loop(first_block, stop_block, block, jnp.int32(0))

        def searching(bounds: tuple[Any, Any]) -> Any:
            return bounds[0] < bounds[1]

        def narrow(bounds: tuple[Any, Any]) -> tuple[Any, Any]:
            lower, upper = bounds
            middle = (lower + upper) // 2
            before = db[middle] + lb[middle] <= tile_start
            return jnp.where(before, middle + 1, lower), jnp.where(
                before, upper, middle
            )

        initial, _ = jax.lax.while_loop(
            searching, narrow, (jnp.int32(0), jnp.int32(la.shape[0]))
        )

        def more(group: Any) -> Any:
            bounded = jnp.minimum(group, la.shape[0] - 1)
            return (group < la.shape[0]) & (da[bounded] < tile_end)

        def group_body(group: Any) -> Any:
            copy_segment(a, cap_left, sa[group], da[group], la[group])
            copy_segment(b, cap_right, sb[group], db[group], lb[group])
            return group + 1

        jax.lax.while_loop(more, group_body, initial)

    bits = pl.pallas_call(
        kernel,
        name="jagged_output_tile_dma",
        out_shape=jax.ShapeDtypeStruct((capacity, width), out_dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=6,
            grid=(rows // block_rows, padded_width // columns),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(
                (block_rows, columns),
                lambda row, col, *unused: (row, col),
                memory_space=pltpu.VMEM,
            ),
            scratch_shapes=[pltpu.VMEM((520, columns), dtype), pltpu.SemaphoreType.DMA],
        ),
        interpret=interpret,
    )(
        source_left.astype(jnp.int32),
        source_right.astype(jnp.int32),
        destination_left.astype(jnp.int32),
        destination_right.astype(jnp.int32),
        sizes_left.astype(jnp.int32),
        sizes_right.astype(jnp.int32),
        left_bits,
        right_bits,
    )
    if typed_output:
        return bits.reshape((capacity,) + left.shape[1:])
    bits = jax.lax.optimization_barrier(bits).reshape(shape)
    return jax.lax.bitcast_convert_type(bits, left.dtype)


def concat_jagged_tiled_dma(
    left: Any,
    offsets_left: Any,
    right: Any,
    offsets_right: Any,
    capacity: int,
    interpret: bool = False,
) -> Any:
    return _copy_jagged_output_tiles(
        left,
        right,
        offsets_left[:-1],
        offsets_right[:-1],
        offsets_left[:-1] + offsets_right[:-1],
        offsets_left[1:] + offsets_right[:-1],
        jnp.diff(offsets_left),
        jnp.diff(offsets_right),
        capacity,
        interpret,
    )


def _copy_jagged_tiled_segments(
    values: Any,
    source_starts: Any,
    destination_starts: Any,
    sizes: Any,
    capacity: int,
    interpret: bool = False,
) -> Any:
    return _copy_jagged_output_tiles(
        values,
        values,
        source_starts,
        source_starts,
        destination_starts,
        destination_starts + sizes,
        sizes,
        jnp.zeros_like(sizes),
        capacity,
        interpret,
    )


def split_jagged_tiled_dma(
    values: Any,
    offsets_left: Any,
    offsets_right: Any,
    capacity_left: int,
    capacity_right: int,
    interpret: bool = False,
) -> tuple[Any, Any]:
    left = _copy_jagged_tiled_segments(
        values,
        offsets_left[:-1] + offsets_right[:-1],
        offsets_left[:-1],
        jnp.diff(offsets_left),
        capacity_left,
        interpret,
    )
    right = _copy_jagged_tiled_segments(
        values,
        offsets_left[1:] + offsets_right[:-1],
        offsets_right[:-1],
        jnp.diff(offsets_right),
        capacity_right,
        interpret,
    )
    return left, right
