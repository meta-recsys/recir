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


def pad_ids_dma(
    rows: jax.Array,
    counts: jax.Array,
    padded_counts: jax.Array,
    pad_value: int = 0,
    interpret: bool = False,
) -> jax.Array:
    size = rows.shape[0]
    capacity = size + counts.shape[0] * 128
    block_rows = 2048
    source = jnp.pad(rows, (512, 640))
    starts = jnp.cumsum(counts) - counts
    destinations = jnp.cumsum(padded_counts) - padded_counts

    def kernel(
        starts: Any,
        destinations: Any,
        lengths: Any,
        source_ref: Any,
        out: Any,
        scratch: Any,
        semaphore: Any,
    ) -> None:
        tile_start = pl.program_id(0) * block_rows
        tile_end = jnp.minimum(tile_start + block_rows, capacity)
        out[...] = jnp.full((block_rows,), pad_value, jnp.int32)

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
                aligned_source = pl.multiple_of(source_position // 128 * 128, 128)
                shift = source_position - aligned_source
                read = pltpu.async_copy(
                    source_ref.at[pl.ds(aligned_source, 640)], scratch, semaphore
                )
                read.wait()
                rotation = jnp.where(shift == 0, 0, 128 - shift)
                tiles = pltpu.roll(scratch[...].reshape(5, 128), rotation, axis=1)
                same_row = jnp.arange(128, dtype=jnp.int32) + shift < 128
                values = jnp.where(same_row[None, :], tiles[:4], tiles[1:]).reshape(512)
                positions = destination + jnp.arange(512, dtype=jnp.int32)
                valid = (
                    (positions >= destinations[group])
                    & (positions < destinations[group] + length)
                    & (positions < tile_end)
                )
                window = out.at[pl.ds(index * 512, 512)]
                window[...] = jnp.where(valid, values, window[...])
                return carry

            jax.lax.fori_loop(first_block, stop_block, copy_block, jnp.int32(0))
            return group + 1

        jax.lax.while_loop(more, group_body, initial)

    return pl.pallas_call(
        kernel,
        name="ringshift_pad_ids_dma",
        out_shape=jax.ShapeDtypeStruct((capacity,), jnp.int32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=((capacity + block_rows - 1) // block_rows,),
            in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
            out_specs=pl.BlockSpec(
                (block_rows,), lambda row, *unused: (row,), memory_space=pltpu.VMEM
            ),
            scratch_shapes=[pltpu.VMEM((640,), jnp.int32), pltpu.SemaphoreType.DMA],
        ),
        interpret=interpret,
    )(
        starts.astype(jnp.int32),
        destinations.astype(jnp.int32),
        counts.astype(jnp.int32),
        source,
    )
