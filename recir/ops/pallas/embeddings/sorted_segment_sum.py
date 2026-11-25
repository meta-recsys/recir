# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Segment sums of rows sorted by segment, as one-hot matmuls on the TPU MXU.

The SparseCore exchange sums each device's position gradients into its send
slots. As a sorted TensorCore scatter-add of [653296, 256] f32 rows that runs at
~85 GB/s, 4.3 ms per stack. Here every block of segments reads only the rows
that land in it (they are contiguous, the ids being sorted) and multiplies a
one-hot [segments, rows] tile into them. Each f32 row is split into three bf16
terms, so every product is exact and the result differs from the scatter-add
only in the f32 summation order.
"""

import functools
import os
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_BLOCK_SEGMENTS = 512
_BLOCK_ROWS = 512
# Work item flags.
_FIRST, _LAST, _ROWS = 1, 2, 4


def sorted_segment_sum_enabled() -> bool:
    """``RECIR_SORTED_SEGMENT_SUM=1``: the SparseCore exchange sums its sorted
    gradient rows with ``sorted_segment_sum``."""
    value = os.environ.get("RECIR_SORTED_SEGMENT_SUM", "0")
    if value not in ("0", "1"):
        raise ValueError(f"Invalid RECIR_SORTED_SEGMENT_SUM: {value!r}")
    return value == "1"


def _bf16_terms(x: Any) -> tuple[Any, ...]:
    """Three bf16 terms that sum to the f32 ``x`` exactly. The low mantissa
    bits are masked off rather than rounded so that no convert pair can be
    folded away."""
    mask = jnp.uint32(0xFFFF0000)
    high = jax.lax.bitcast_convert_type(
        jax.lax.bitcast_convert_type(x, jnp.uint32) & mask, jnp.float32
    )
    rest = x - high
    middle = jax.lax.bitcast_convert_type(
        jax.lax.bitcast_convert_type(rest, jnp.uint32) & mask, jnp.float32
    )
    return tuple(t.astype(jnp.bfloat16) for t in (high, middle, rest - middle))


def _work_items(
    bounds: Any, num_rows: int, block_rows: int, max_items: int
) -> tuple[Any, Any, Any]:
    """(segment block, row chunk, flags) of each grid step.

    Segment block ``b`` reads the row chunks that hold rows ``[bounds[b],
    bounds[b + 1])``, in order; a block without rows gets one step that writes
    zeros. Steps past the last block's repeat it with no flags.
    """
    lo, hi = bounds[:-1], bounds[1:]
    has_rows = hi > lo
    first_chunk = lo // block_rows
    count = jnp.where(has_rows, (hi + block_rows - 1) // block_rows - first_chunk, 1)
    ends = jnp.cumsum(count)
    step = jnp.arange(max_items, dtype=jnp.int32)
    block = jnp.minimum(
        jnp.searchsorted(ends, step, side="right").astype(jnp.int32), lo.shape[0] - 1
    )
    offset = step - (ends - count)[block]
    chunk = jnp.clip(first_chunk[block] + offset, 0, pl.cdiv(num_rows, block_rows) - 1)
    flags = (
        jnp.where(offset == 0, _FIRST, 0)
        | jnp.where(offset == count[block] - 1, _LAST, 0)
        | jnp.where(has_rows[block], _ROWS, 0)
    )
    flags = jnp.where(step < ends[-1], flags, 0)
    return block, chunk.astype(jnp.int32), flags.astype(jnp.int32)


def _kernel(
    block_ref: Any,
    chunk_ref: Any,
    flags_ref: Any,
    ids_ref: Any,
    values_ref: Any,
    out_ref: Any,
    acc_ref: Any,
    *,
    num_rows: int,
    block_segments: int,
    block_rows: int,
    scale: float,
) -> None:
    step = pl.program_id(0)
    flags = flags_ref[step]

    @pl.when((flags & _FIRST) != 0)
    def _clear() -> None:
        acc_ref[...] = jnp.zeros_like(acc_ref)

    @pl.when((flags & _ROWS) != 0)
    def _accumulate() -> None:
        segment = (
            jax.lax.broadcasted_iota(jnp.int32, (block_segments, block_rows), 0)
            + block_ref[step] * block_segments
        )
        onehot = (segment == ids_ref[...]).astype(jnp.bfloat16)
        # Rows past the end of the values are unspecified; zero them, since a
        # zero one-hot weight does not cancel a NaN.
        row = jax.lax.broadcasted_iota(jnp.int32, values_ref.shape, 0)
        values = jnp.where(
            row < num_rows - chunk_ref[step] * block_rows, values_ref[...], 0.0
        )
        total = acc_ref[...]
        for term in _bf16_terms(values):
            total += jnp.dot(onehot, term, preferred_element_type=jnp.float32)
        acc_ref[...] = total

    @pl.when((flags & _LAST) != 0)
    def _emit() -> None:
        total = acc_ref[...]
        if scale != 1.0:
            total = total * scale
        out_ref[...] = total.astype(out_ref.dtype)


def sorted_segment_sum(
    values: Any,
    segment_ids: Any,
    num_segments: int,
    *,
    scale: float = 1.0,
    out_dtype: Any = jnp.float32,
    block_segments: int = _BLOCK_SEGMENTS,
    block_rows: int = _BLOCK_ROWS,
) -> Any:
    """``scale * zeros([num_segments, D]).at[segment_ids].add(values)``, summed
    in f32 and cast to ``out_dtype``, for non-decreasing ``segment_ids``.

    ``values`` is [N, D] f32. Ids at or past ``num_segments`` are dropped, like
    the scatter's default mode; negative ids are not allowed.
    """
    n, d = values.shape
    if d % 128:
        raise ValueError(f"sorted_segment_sum needs a multiple of 128 columns, got {d}")
    rows = pl.cdiv(max(n, 1), block_rows) * block_rows
    blocks = pl.cdiv(num_segments, block_segments)
    ids = segment_ids.astype(jnp.int32)
    bounds = jnp.searchsorted(
        ids,
        jnp.minimum(
            jnp.arange(blocks + 1, dtype=jnp.int32) * block_segments, num_segments
        ),
        side="left",
    ).astype(jnp.int32)
    # Each block's chunks overlap the next block's by at most one.
    block, chunk, flags = _work_items(
        bounds, n, block_rows, rows // block_rows + blocks
    )
    padded_ids = jnp.pad(ids, (0, rows - n), constant_values=-1).reshape(1, rows)
    with jax.enable_x64(False):
        out = pl.pallas_call(
            functools.partial(
                _kernel,
                num_rows=n,
                block_segments=block_segments,
                block_rows=block_rows,
                scale=scale,
            ),
            name="sorted_segment_sum",
            out_shape=jax.ShapeDtypeStruct((blocks * block_segments, d), out_dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=3,
                grid=(block.shape[0],),
                in_specs=[
                    pl.BlockSpec((1, block_rows), lambda k, b, c, f: (0, c[k])),
                    pl.BlockSpec((block_rows, d), lambda k, b, c, f: (c[k], 0)),
                ],
                out_specs=pl.BlockSpec(
                    (block_segments, d), lambda k, b, c, f: (b[k], 0)
                ),
                scratch_shapes=[pltpu.VMEM((block_segments, d), jnp.float32)],
            ),
            compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
        )(block, chunk, flags, padded_ids, values.astype(jnp.float32))
    if out.shape[0] != num_segments:
        out = out[:num_segments]
    return out
