# pyre-strict
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def pad_ids_cast_dma(
    gradient: jax.Array,
    rows: jax.Array,
    counts: jax.Array,
    padded_counts: jax.Array,
    pad_value: int,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    size, width = gradient.shape
    capacity = size + counts.shape[0] * 128
    block_rows = 2048
    columns = (width + 127) // 128 * 128
    source = jnp.pad(gradient, ((512, 520), (0, columns - width)))
    row_source = jnp.pad(rows, ((512, 640),))
    real_offsets = jnp.cumsum(counts) - counts
    padded_offsets = jnp.cumsum(padded_counts) - padded_counts

    def kernel(
        starts: Any,
        destinations: Any,
        lengths: Any,
        source_ref: Any,
        row_source_ref: Any,
        out: Any,
        out_rows: Any,
        scratch: Any,
        row_scratch: Any,
        semaphore: Any,
        row_semaphore: Any,
    ) -> None:
        tile_start = pl.program_id(0) * block_rows
        tile_end = jnp.minimum(tile_start + block_rows, capacity)
        if not interpret:
            out = out.bitcast(jnp.uint16)
        out[...] = jnp.zeros((block_rows, columns), jnp.uint16)
        out_rows[...] = jnp.full((block_rows,), pad_value, jnp.int32)

        def searching(bounds: tuple[jax.Array, jax.Array]) -> jax.Array:
            return bounds[0] < bounds[1]

        def narrow(bounds: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
            lower, upper = bounds
            middle = (lower + upper) // 2
            before = destinations[middle] + lengths[middle] <= tile_start
            return (
                jnp.where(before, middle + 1, lower),
                jnp.where(before, upper, middle),
            )

        initial, _ = jax.lax.while_loop(
            searching, narrow, (jnp.int32(0), jnp.int32(counts.shape[0]))
        )

        def more(group: jax.Array) -> jax.Array:
            bounded = jnp.minimum(group, counts.shape[0] - 1)
            return (group < counts.shape[0]) & (destinations[bounded] < tile_end)

        def group_body(group: jax.Array) -> jax.Array:
            length = jnp.maximum(0, jnp.minimum(lengths[group], size - starts[group]))
            first = jnp.maximum(destinations[group], tile_start)
            end = jnp.minimum(destinations[group] + length, tile_end)
            first_block = (first - tile_start) // 512
            stop_block = jnp.where(
                end > first, (end - tile_start + 511) // 512, first_block
            )

            def copy_block(index: jax.Array, carry: jax.Array) -> jax.Array:
                destination = tile_start + index * 512
                source_position = (
                    starts[group] + 512 + destination - destinations[group]
                )
                aligned_source = pl.multiple_of(source_position // 8 * 8, 8)
                shift = source_position - aligned_source
                read = pltpu.async_copy(
                    source_ref.at[pl.ds(aligned_source, 520), pl.ds(0, columns)],
                    scratch,
                    semaphore,
                )
                aligned_row_source = pl.multiple_of(source_position // 128 * 128, 128)
                row_shift = source_position - aligned_row_source
                row_read = pltpu.async_copy(
                    row_source_ref.at[pl.ds(aligned_row_source, 640)],
                    row_scratch,
                    row_semaphore,
                )
                read.wait()
                row_read.wait()
                rotation = jnp.where(shift == 0, 0, 520 - shift)
                floats = pltpu.roll(scratch[...], rotation, axis=0)[:512]
                row_rotation = jnp.where(row_shift == 0, 0, 128 - row_shift)
                row_tiles = pltpu.roll(
                    row_scratch[...].reshape(5, 128), row_rotation, axis=1
                )
                same_row = jnp.arange(128, dtype=jnp.int32) + row_shift < 128
                row_values = jnp.where(
                    same_row[None, :], row_tiles[:4], row_tiles[1:]
                ).reshape(512)
                rounded = floats.astype(jnp.bfloat16)
                bits = jax.lax.bitcast_convert_type(rounded, jnp.uint16).astype(
                    jnp.uint32
                )
                positions = destination + jax.lax.broadcasted_iota(
                    jnp.int32, (512, columns), 0
                )
                valid = (
                    (positions >= destinations[group])
                    & (positions < destinations[group] + length)
                    & (positions < tile_end)
                )
                window = out.at[pl.ds(index * 512, 512), pl.ds(0, columns)]
                window[...] = jnp.where(
                    valid, bits, window[...].astype(jnp.uint32)
                ).astype(jnp.uint16)
                row_positions = destination + jnp.arange(512, dtype=jnp.int32)
                valid_rows = (
                    (row_positions >= destinations[group])
                    & (row_positions < destinations[group] + length)
                    & (row_positions < tile_end)
                )
                row_window = out_rows.at[pl.ds(index * 512, 512)]
                row_window[...] = jnp.where(valid_rows, row_values, row_window[...])
                return carry

            jax.lax.fori_loop(first_block, stop_block, copy_block, jnp.int32(0))
            return group + 1

        jax.lax.while_loop(more, group_body, initial)

    output, output_rows = pl.pallas_call(
        kernel,
        name="ringshift_pad_ids_cast_dma",
        out_shape=(
            jax.ShapeDtypeStruct(
                (capacity, width), jnp.uint16 if interpret else jnp.bfloat16
            ),
            jax.ShapeDtypeStruct((capacity,), jnp.int32),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=((capacity + block_rows - 1) // block_rows,),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=(
                pl.BlockSpec(
                    (block_rows, columns),
                    lambda row, *unused: (row, jnp.int32(0)),
                    memory_space=pltpu.VMEM,
                ),
                pl.BlockSpec(
                    (block_rows,), lambda row, *unused: (row,), memory_space=pltpu.VMEM
                ),
            ),
            scratch_shapes=[
                pltpu.VMEM((520, columns), jnp.float32),
                pltpu.VMEM((640,), jnp.int32),
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
            ],
        ),
        interpret=interpret,
    )(
        real_offsets.astype(jnp.int32),
        padded_offsets.astype(jnp.int32),
        counts.astype(jnp.int32),
        source,
        row_source,
    )
    gradient_output = (
        jax.lax.bitcast_convert_type(output, jnp.bfloat16) if interpret else output
    )
    return gradient_output, output_rows
