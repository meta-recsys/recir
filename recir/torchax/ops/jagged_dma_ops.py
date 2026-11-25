# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Copy contiguous jagged segments with integer DMA tiles."""

from math import prod
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Block reads in flight, plus the one being merged.
_COPY_SLOTS = 4


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


def _copy_jagged_output_tiles(  # noqa: C901
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
        tiles: Any,
        semaphores: Any,
        copies: Any,
    ) -> None:
        tile_start = pl.program_id(0) * block_rows
        column = pl.program_id(1) * columns
        tile_end = jnp.minimum(tile_start + block_rows, capacity)
        groups = la.shape[0]
        if typed_output:
            out = out.bitcast(dtype)
        out[...] = jnp.zeros((block_rows, columns), dtype)

        # The tile is filled by a sequence of 512-row block copies: per group
        # overlapping the tile, its left then its right segment, each in the
        # output blocks it overlaps. A copy is (group, side, block). Up to
        # _COPY_SLOTS - 1 reads are in flight while the oldest one is merged;
        # ``copies`` holds each slot's copy.

        def segment(group: Any, side: Any) -> tuple[Any, Any, Any]:
            group = jnp.minimum(group, groups - 1)
            left = side == 0
            source_start = jnp.where(left, sa[group], sb[group])
            destination_start = jnp.where(left, da[group], db[group])
            source_capacity = jnp.where(left, jnp.int32(cap_left), jnp.int32(cap_right))
            length = jnp.maximum(
                0,
                jnp.minimum(
                    jnp.where(left, la[group], lb[group]),
                    jnp.minimum(
                        source_capacity - source_start, capacity - destination_start
                    ),
                ),
            )
            return source_start, destination_start, length

        def blocks(group: Any, side: Any) -> tuple[Any, Any]:
            _, destination_start, length = segment(group, side)
            first = jnp.maximum(destination_start, tile_start)
            end = jnp.minimum(destination_start + length, tile_end)
            first_block = (first - tile_start) // 512
            return first_block, jnp.where(
                end > first, (end - tile_start + 511) // 512, first_block
            )

        def overlaps(group: Any) -> Any:
            return (group < groups) & (da[jnp.minimum(group, groups - 1)] < tile_end)

        def seek(group: Any, side: Any) -> tuple[Any, Any, Any, Any]:
            """The first (group, side) at or after the given one with a block in
            the tile, with its block range."""

            def empty(cursor: tuple[Any, Any, Any, Any]) -> Any:
                group, _, block, stop = cursor
                return overlaps(group) & (block >= stop)

            def step(cursor: tuple[Any, Any, Any, Any]) -> tuple[Any, Any, Any, Any]:
                group, side, _, _ = cursor
                group, side = group + side, 1 - side
                return (group, side, *blocks(group, side))

            return jax.lax.while_loop(empty, step, (group, side, *blocks(group, side)))

        def read(group: Any, side: Any, block: Any, slot: Any) -> Any:
            source_start, destination_start, _ = segment(group, side)
            destination = tile_start + block * 512
            source_position = source_start + 512 + destination - destination_start
            aligned_source = pl.multiple_of(source_position // 8 * 8, 8)

            def copy(source: Any) -> Any:
                return pltpu.make_async_copy(
                    source.at[pl.ds(aligned_source, 520), pl.ds(column, columns)],
                    tiles.at[slot],
                    semaphores.at[slot],
                )

            return copy, source_position - aligned_source

        def start(group: Any, side: Any, block: Any, slot: Any) -> None:
            copy, _ = read(group, side, block, slot)
            pl.when(side == 0)(lambda: copy(a).start())
            pl.when(side == 1)(lambda: copy(b).start())

        def merge(group: Any, side: Any, block: Any, slot: Any) -> None:
            copy, shift = read(group, side, block, slot)
            copy(a).wait()
            _, destination_start, length = segment(group, side)
            destination = tile_start + block * 512
            rotation = jnp.where(shift == 0, 0, 520 - shift)
            raw = pltpu.roll(tiles[slot].astype(jnp.uint32), rotation, axis=0)[:512]
            position = destination + jax.lax.broadcasted_iota(
                jnp.int32, (512, columns), 0
            )
            valid = (
                (position >= destination_start)
                & (position < destination_start + length)
                & (position < tile_end)
            )
            window = out.at[pl.ds(block * 512, 512), pl.ds(0, columns)]
            window[...] = jnp.where(valid, raw, window[...].astype(jnp.uint32)).astype(
                dtype
            )

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
            searching, narrow, (jnp.int32(0), jnp.int32(groups))
        )

        def following(cursor: tuple[Any, Any, Any, Any]) -> tuple[Any, Any, Any, Any]:
            group, side, block, stop = cursor
            later = seek(group + side, 1 - side)
            within = block + 1 < stop
            return (
                jnp.where(within, group, later[0]),
                jnp.where(within, side, later[1]),
                jnp.where(within, block + 1, later[2]),
                jnp.where(within, stop, later[3]),
            )

        def issue(cursor: tuple[Any, ...], issued: Any) -> tuple[Any, Any]:
            """Start the read of the copy at ``cursor``, if any, into the next slot."""
            group, side, block, _ = cursor
            pending = overlaps(group)
            slot = issued % _COPY_SLOTS

            @pl.when(pending)
            def _issue() -> None:
                copies[slot, 0] = group
                copies[slot, 1] = side
                copies[slot, 2] = block
                start(group, side, block, slot)

            advanced = following(cursor)
            return (
                tuple(jnp.where(pending, x, y) for x, y in zip(advanced, cursor)),
                issued + pending.astype(jnp.int32),
            )

        def filling(state: tuple[Any, ...]) -> Any:
            cursor, issued = state
            return overlaps(cursor[0]) & (issued < _COPY_SLOTS - 1)

        cursor, issued = jax.lax.while_loop(
            filling,
            lambda state: issue(*state),
            (seek(initial, jnp.int32(0)), jnp.int32(0)),
        )

        def unmerged(state: tuple[Any, ...]) -> Any:
            _, issued, merged = state
            return merged < issued

        def merge_next(state: tuple[Any, ...]) -> tuple[Any, ...]:
            cursor, issued, merged = state
            cursor, issued = issue(cursor, issued)
            slot = merged % _COPY_SLOTS
            merge(copies[slot, 0], copies[slot, 1], copies[slot, 2], slot)
            return cursor, issued, merged + 1

        jax.lax.while_loop(unmerged, merge_next, (cursor, issued, jnp.int32(0)))

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
            scratch_shapes=[
                pltpu.VMEM((_COPY_SLOTS, 520, columns), dtype),
                pltpu.SemaphoreType.DMA((_COPY_SLOTS,)),
                pltpu.SMEM((_COPY_SLOTS, 3), jnp.int32),
            ],
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
