# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Segment sums into small tables as one-hot matmuls on the TPU MXU.

A scatter-add of N rows into a table of a few thousand rows serializes on
conflicting rows: RankFM's 325k-row embedding and position gradients take
1.8-4.7 ms each that way. Multiplying a one-hot [segments, rows] tile into the
rows does the same reduction on the MXU. Each f32 row is split into three bf16
terms (8 + 8 + 8 mantissa bits), so every product is exact and the result
differs from a scatter-add only in the f32 summation order.
"""

import os
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_BLOCK_ROWS = 2048
_BLOCK_SEGMENTS = 512
# Above this the one-hot work (segments x rows) outgrows the scatter.
_MAX_SEGMENTS = 4096


def onehot_segment_sum_enabled(num_segments: int) -> bool:
    """``RECIR_ONEHOT_SEGMENT_SUM=1``: embedding and packed-position gradients
    of tables up to 4096 rows use ``onehot_segment_sum``."""
    value = os.environ.get("RECIR_ONEHOT_SEGMENT_SUM", "0")
    if value not in ("0", "1"):
        raise ValueError(f"Invalid RECIR_ONEHOT_SEGMENT_SUM: {value!r}")
    return value == "1" and 0 < num_segments <= _MAX_SEGMENTS


def _truncate_to_bf16(x: Any) -> Any:
    # Masking the low mantissa bits instead of round-tripping through bf16:
    # XLA may drop an f32 -> bf16 -> f32 convert pair (excess precision).
    bits = jax.lax.bitcast_convert_type(x, jnp.uint32) & jnp.uint32(0xFFFF0000)
    return jax.lax.bitcast_convert_type(bits, jnp.float32)


def _bf16_terms(x: Any) -> tuple[Any, ...]:
    if x.dtype == jnp.bfloat16:
        return (x,)
    x = x.astype(jnp.float32)
    # Inf and NaN ride in the high term alone: masking a signalling NaN's
    # payload can leave +Inf, and Inf - Inf would make every later term NaN.
    finite = jnp.isfinite(x)
    high = jnp.where(finite, _truncate_to_bf16(x), x)
    rest = jnp.where(finite, x - high, 0.0)
    middle = _truncate_to_bf16(rest)
    low = rest - middle
    return tuple(t.astype(jnp.bfloat16) for t in (high, middle, low))


def onehot_segment_sum(values: Any, segment_ids: Any, num_segments: int) -> Any:
    """``zeros([num_segments, D]).at[segment_ids].add(values)`` in f32.

    ``values`` is [N, D]. Negative ids count from the end and ids outside
    ``[-num_segments, num_segments)`` are dropped, like the scatter's default
    mode. A row holding Inf or NaN reaches every segment, not just its own:
    the one-hot's zeros multiply it into every other row.
    """
    n, d = values.shape
    rows = pl.cdiv(max(n, 1), _BLOCK_ROWS) * _BLOCK_ROWS
    block_segments = min(_BLOCK_SEGMENTS, pl.cdiv(num_segments, 128) * 128)
    segments = pl.cdiv(num_segments, block_segments) * block_segments
    width = pl.cdiv(d, 128) * 128
    ids = segment_ids.astype(jnp.int32)
    ids = jnp.where(ids < 0, ids + num_segments, ids)
    ids = jnp.pad(ids, (0, rows - n), constant_values=-1).reshape(1, rows)
    terms = _bf16_terms(jnp.pad(values, ((0, rows - n), (0, width - d))))

    def kernel(ids_ref: Any, *refs: Any) -> None:
        *term_refs, out_ref = refs

        @pl.when(pl.program_id(1) == 0)
        def _init() -> None:
            out_ref[...] = jnp.zeros_like(out_ref)

        first = pl.program_id(0) * block_segments
        segment = jax.lax.broadcasted_iota(jnp.int32, (block_segments, _BLOCK_ROWS), 0)
        onehot = (segment + first == ids_ref[...]).astype(jnp.bfloat16)
        total = out_ref[...]
        for term_ref in term_refs:
            total += jnp.dot(onehot, term_ref[...], preferred_element_type=jnp.float32)
        out_ref[...] = total

    with jax.enable_x64(False):
        out = pl.pallas_call(
            kernel,
            name="onehot_segment_sum",
            out_shape=jax.ShapeDtypeStruct((segments, width), jnp.float32),
            grid=(segments // block_segments, rows // _BLOCK_ROWS),
            in_specs=[pl.BlockSpec((1, _BLOCK_ROWS), lambda s, r: (0, r))]
            + [pl.BlockSpec((_BLOCK_ROWS, width), lambda s, r: (r, 0))] * len(terms),
            out_specs=pl.BlockSpec((block_segments, width), lambda s, r: (s, 0)),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel", "arbitrary")
            ),
        )(ids, *terms)
    return out[:num_segments, :d]
