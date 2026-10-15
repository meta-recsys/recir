# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Torchax JAX lowerings for recir::jagged ops with shard_map.

These lowerings run the jagged<->dense conversions inside jax.shard_map over
the "torch_dist" batch axis, making them shard-local for non-uniform offsets.
This keeps activations batch-sharded and avoids the large replicated buffer OOM due to gather
on TPU/GSPMD.
"""

from typing import Any, List, Optional

import jax
import jax.numpy as jnp

# Import to register the custom ops
import recir.ops.jagged_ops  # noqa: F401
import torch
from jax.sharding import Mesh, PartitionSpec as P
from recir.torchax.ops.decorators import op

_DP_AXIS = "torch_dist"


def _with_sharding_constraint(x: Any, spec: Any) -> Any:
    """Apply sharding constraint to keep output batch-sharded."""
    from jax.sharding import NamedSharding

    mesh = Mesh(jax.devices(), (_DP_AXIS,))
    sharding = NamedSharding(mesh, spec)
    return jax.lax.with_sharding_constraint(x, sharding)


def _jax_jagged_to_padded_dense_impl(
    values: Any,
    offsets: Any,
    max_lengths: Any,
    padding_value: float = 0.0,
) -> Any:
    """Pure-JAX jagged->padded-dense gather (per-device kernel for the shard_map lowerings)."""
    seq_offsets = offsets[0]
    max_len: int = int(max_lengths[0])
    b: int = seq_offsets.shape[0] - 1
    starts = seq_offsets[:-1]
    lengths = seq_offsets[1:] - seq_offsets[:-1]
    col = jnp.arange(max_len)
    mask = col[None, :] < lengths[:, None]
    flat_idx = (starts[:, None] + col[None, :]).clip(0, max(values.shape[0] - 1, 0))
    if values.ndim == 1:
        gathered = values[flat_idx.reshape(-1)].reshape(b, max_len)
        return jnp.where(mask, gathered, jnp.asarray(padding_value, dtype=values.dtype))
    gathered = values[flat_idx.reshape(-1)].reshape(b, max_len, values.shape[-1])
    return jnp.where(
        mask[:, :, None], gathered, jnp.asarray(padding_value, dtype=values.dtype)
    )


def _jax_dense_to_jagged_forward_impl(
    dense: Any,
    offsets: Any,
    total_L: Any = None,
) -> Any:
    """Pure-JAX dense->jagged scatter-free gather (per-device kernel)."""
    seq_offsets = offsets[0]
    b: int = dense.shape[0]
    n: int = dense.shape[1]
    if total_L is None:
        total_L = b * n
    lengths = seq_offsets[1:] - seq_offsets[:-1]
    col = jnp.arange(n)
    mask = col[None, :] < lengths[:, None]
    dest = (seq_offsets[:-1, None] + col[None, :]).clip(0, max(total_L - 1, 0))
    if dense.ndim == 2:
        src = jnp.where(mask, dense, jnp.zeros_like(dense))
        result = jnp.zeros(total_L, dtype=dense.dtype)
        # pyrefly: ignore [missing-attribute]
        return result.at[dest.reshape(-1)].add(src.reshape(-1))
    d: int = dense.shape[2]
    src = jnp.where(mask[:, :, None], dense, jnp.zeros_like(dense))
    result = jnp.zeros((total_L, d), dtype=dense.dtype)
    # pyrefly: ignore [missing-attribute]
    return result.at[dest.reshape(-1), :].add(src.reshape(-1, d))


@op(torch.ops.recir.dense_to_jagged.default)  # pyre-ignore[56]
def _recir_dense_to_jagged(
    dense: Any,
    offsets: List[Any],
    total_L: Optional[int] = None,
) -> Any:
    """Lowering for recir::dense_to_jagged with shard_map.
    Runs the gather inside shard_map over the batch axis so each device
    processes its local batch shard. The output stays batch-sharded.
    Pass lengths (not offsets) sharded by batch, recompute local offsets
    inside the shard_map body. This ensures each device operates on its local
    batch slice with local offsets.
    """
    seq_offsets = offsets[0]
    B = dense.shape[0]
    N = dense.shape[1]
    if total_L is None:
        # Static fallback (fully-padded): B*N.
        total_L = B * N
    world_size = len(jax.devices())
    if world_size <= 1:
        result = _jax_dense_to_jagged_forward_impl(dense, offsets, total_L)
        return result, seq_offsets
    lengths = seq_offsets[1:] - seq_offsets[:-1]  # [B]
    # pyrefly: ignore [unsupported-operation]
    assert total_L % world_size == 0, (
        f"total_L={total_L} must be divisible by world_size={world_size} for batch-aligned sharding"
    )
    # pyrefly: ignore [unsupported-operation]
    local_total_L = total_L // world_size
    mesh = Mesh(jax.devices(), (_DP_AXIS,))

    def _per_device_dense_to_jagged(
        dense_s: Any,
        lengths_s: Any,
    ) -> Any:
        local_offsets = jnp.concatenate(
            [jnp.zeros(1, dtype=lengths_s.dtype), jnp.cumsum(lengths_s)]
        )
        result = _jax_dense_to_jagged_forward_impl(
            dense_s, [local_offsets], local_total_L
        )
        return result

    dense_spec = P(*([_DP_AXIS] + [None] * (dense.ndim - 1)))
    jagged_out_spec = P(*([_DP_AXIS] + [None] * (dense.ndim - 2)))
    result = jax.shard_map(
        _per_device_dense_to_jagged,
        mesh=mesh,
        in_specs=(
            dense_spec,  # dense batch-sharded on dim 0
            P(_DP_AXIS),  # lengths: [B] -> batch-sharded
        ),
        out_specs=jagged_out_spec,  # jagged [local_total_L(, D)] batch-sharded
        check_vma=False,
        # pyrefly: ignore [bad-argument-count]
    )(dense, lengths)
    result = _with_sharding_constraint(result, jagged_out_spec)
    return result, seq_offsets


@op(torch.ops.recir.jagged_to_padded_dense.default)  # pyre-ignore[56]
def _recir_jagged_to_padded_dense(
    values: Any,
    offsets: List[Any],
    max_lengths: List[int],
    padding_value: float = 0.0,
) -> Any:
    """Lowering for recir::jagged_to_padded_dense with shard_map.
    Runs the gather inside shard_map over the batch axis so each device
    processes its local batch shard. The output stays batch-sharded.
    """
    seq_offsets = offsets[0]
    world_size = len(jax.devices())
    if world_size <= 1:
        result = _jax_jagged_to_padded_dense_impl(
            values, offsets, max_lengths, padding_value
        )
        return result
    lengths = seq_offsets[1:] - seq_offsets[:-1]  # [B]
    total_L = values.shape[0]
    assert total_L % world_size == 0, (
        f"total_L={total_L} must be divisible by world_size={world_size}"
    )
    mesh = Mesh(jax.devices(), (_DP_AXIS,))

    def _per_device_jagged_to_padded(
        values_s: Any,
        lengths_s: Any,
    ) -> Any:
        local_offsets = jnp.concatenate(
            [jnp.zeros(1, dtype=lengths_s.dtype), jnp.cumsum(lengths_s)]
        )
        result = _jax_jagged_to_padded_dense_impl(
            values_s, [local_offsets], max_lengths, padding_value
        )
        return result

    values_spec = P(*([_DP_AXIS] + [None] * (values.ndim - 1)))
    dense_out_spec = P(*([_DP_AXIS] + [None] * values.ndim))
    result = jax.shard_map(
        _per_device_jagged_to_padded,
        mesh=mesh,
        in_specs=(
            values_spec,  # values batch-sharded on the jagged dim
            P(_DP_AXIS),  # lengths: [B] -> batch-sharded
        ),
        out_specs=dense_out_spec,  # dense [B_local, max_len(, D)] batch-sharded
        check_vma=False,
        # pyrefly: ignore [bad-argument-count]
    )(values, lengths)
    result = _with_sharding_constraint(result, dense_out_spec)
    return result


def _bcast_mask(mask: Any, ndim: int) -> Any:
    """Reshape a [N] mask to broadcast against an [N, ...] tensor (1D or 2D values)."""
    return mask.reshape(mask.shape + (1,) * (ndim - mask.ndim))


def _segment_id_by_cumsum(boundaries: Any, n: int) -> Any:
    """Vectorized replacement for ``clip(searchsorted(cum, arange(n), "right"), 0, B-1)``.

    ``searchsorted`` over a monotonically increasing offsets array lowers to a
    serialized binary-search ``while`` loop on TPU (memory-bound, ~0 FLOPs). The
    same "which segment does flat position p fall into" mapping is O(n) and
    fully vectorized: scatter-add a 1 at each interior segment start, then take
    an inclusive prefix sum.

    Args:
        boundaries: ``[num_segments - 1]`` int32 -- the interior segment-start
            positions (i.e. the cumulative offsets EXCLUDING the leading 0 and
            the trailing total). Duplicate values (empty segments) accumulate,
            and out-of-range values (>= n) are dropped, so the result saturates
            at ``num_segments - 1`` for trailing positions -- matching the
            ``clip(..., 0, num_segments - 1)`` in the searchsorted form.
        n: static output length.

    Returns:
        ``[n]`` int32 segment id for each position ``0..n-1``.
    """
    delta = (
        jnp.zeros((n,), jnp.int32).at[boundaries.astype(jnp.int32)].add(1, mode="drop")
    )
    return jnp.cumsum(delta)


def _segment_broadcast(per_segment: Any, starts: Any, n: int) -> Any:
    """``_segment_id_by_cumsum`` for an arbitrary per-segment value, not just the id.

    ``per_segment[segment_id]`` gives the same ``[n]`` array but lowers to a
    lookup into a length-B table with a length-n index vector, which on TPU runs
    about two orders of magnitude below a scan. ``starts`` is the position each
    segment begins at; out-of-range and duplicate entries behave as they do
    there.
    """
    values = per_segment.astype(jnp.int32)
    jumps = jnp.diff(values, prepend=jnp.zeros((1,), jnp.int32))
    delta = (
        jnp.zeros((n,), jnp.int32).at[starts.astype(jnp.int32)].add(jumps, mode="drop")
    )
    return jnp.cumsum(delta)


def _jax_concat_2d_jagged(
    values_left: Any,
    offsets_left: Any,
    values_right: Any,
    offsets_right: Any,
    out_total: int,
) -> Any:
    """Per-sample interleave: combined[b] = cat(left[b], right[b]).

    Each output row comes from either the left or the right values, decided by
    which sample it falls in and how far into that sample it sits. Indexing
    dominates this op on TPU, so the per-sample quantities are built by
    ``_segment_broadcast`` instead of looked up per position, and the operands
    share one index space so the left/right choice applies to the source index
    before a single fetch rather than to two fetched rows.
    """
    batch_size = offsets_left.shape[0] - 1
    combined_offsets = offsets_left + offsets_right  # cumulative, [batch_size + 1]
    lengths_left = offsets_left[1:] - offsets_left[:-1]  # [batch_size]
    starts = combined_offsets[:batch_size]
    positions = jnp.arange(out_total)

    # pos_* are per output position; the arrays they broadcast from are per
    # sample, so the two are never a plural apart.
    pos_sample_start = _segment_broadcast(starts, starts, out_total)
    pos_left_len = _segment_broadcast(lengths_left, starts, out_total)
    pos_left_off = _segment_broadcast(offsets_left[:batch_size], starts, out_total)
    pos_right_off = _segment_broadcast(offsets_right[:batch_size], starts, out_total)

    intra_offset = positions - pos_sample_start  # offset within the sample
    is_left = intra_offset < pos_left_len

    # A shard may legitimately receive 0 rows, which contribute no index space.
    n_left = values_left.shape[0]
    if n_left == 0:
        combined = values_right
    elif values_right.shape[0] == 0:
        combined = values_left
    else:
        combined = jnp.concatenate([values_left, values_right], axis=0)

    if combined.shape[0] == 0:
        out = jnp.zeros((out_total,) + values_left.shape[1:], dtype=values_left.dtype)
    else:
        source: Any = jnp.where(
            is_left,
            pos_left_off + intra_offset,
            n_left + pos_right_off + (intra_offset - pos_left_len),
        )
        out = combined[jnp.clip(source, 0, combined.shape[0] - 1)]

    valid = positions < combined_offsets[batch_size]
    # pyrefly: ignore [missing-attribute, no-matching-overload]
    return jnp.where(_bcast_mask(valid, out.ndim), out, 0)


def _jax_split_2d_jagged(
    values: Any,
    offsets_left: Any,
    offsets_right: Any,
    seq_len_a: int,
    seq_len_b: int,
) -> Any:
    """Inverse of _jax_concat_2d_jagged: recover the left and right jagged tensors.

    For each output row we need the sample it belongs to so we can index back
    into the interleaved `values` buffer. left[p] lives at
    values[p + offsets_right[sample]]; right[p] lives at
    values[p + offsets_left[sample + 1]]. Both of those per-sample values are
    built by ``_segment_broadcast`` rather than looked up per position, for the
    reason given there.
    """
    batch_size = offsets_left.shape[0] - 1
    # Guard against an empty input buffer
    if values.shape[0] == 0:
        left = jnp.zeros((seq_len_a,) + values.shape[1:], dtype=values.dtype)
        right = jnp.zeros((seq_len_b,) + values.shape[1:], dtype=values.dtype)
        return left, right
    positions_left = jnp.arange(seq_len_a)
    pos_right_off = _segment_broadcast(
        offsets_right[:batch_size], offsets_left[:batch_size], seq_len_a
    )
    left = values[jnp.clip(positions_left + pos_right_off, 0, values.shape[0] - 1)]
    left = jnp.where(
        _bcast_mask(positions_left < offsets_left[batch_size], left.ndim), left, 0
    )
    positions_right = jnp.arange(seq_len_b)
    pos_next_left_off = _segment_broadcast(
        offsets_left[1 : batch_size + 1], offsets_right[:batch_size], seq_len_b
    )
    right = values[
        jnp.clip(positions_right + pos_next_left_off, 0, values.shape[0] - 1)
    ]
    right = jnp.where(
        _bcast_mask(positions_right < offsets_right[batch_size], right.ndim), right, 0
    )
    return left, right


@op(torch.ops.recir.concat_2d_jagged.default)  # pyre-ignore[56]
def _recir_concat_2d_jagged(
    values_left: Any,
    offsets_left: Any,
    values_right: Any,
    offsets_right: Any,
) -> Any:
    """recir::concat_2d_jagged lowering: per-sample interleave inside shard_map."""
    out_total = values_left.shape[0] + values_right.shape[0]  # static

    world_size = len(jax.devices())
    if world_size <= 1:
        return _jax_concat_2d_jagged(
            values_left, offsets_left, values_right, offsets_right, out_total
        )

    lengths_left = offsets_left[1:] - offsets_left[:-1]
    lengths_right = offsets_right[1:] - offsets_right[:-1]
    total_left = values_left.shape[0]
    total_right = values_right.shape[0]
    assert total_left % world_size == 0 and total_right % world_size == 0, (
        f"concat totals ({total_left},{total_right}) must be divisible by "
        f"world_size={world_size}"
    )
    local_out = (total_left + total_right) // world_size
    mesh = Mesh(jax.devices(), (_DP_AXIS,))

    def _per_device(
        values_left_shard: Any,
        lengths_left_shard: Any,
        values_right_shard: Any,
        lengths_right_shard: Any,
    ) -> Any:
        offsets_left_local = jnp.concatenate(
            [
                jnp.zeros(1, dtype=lengths_left_shard.dtype),
                jnp.cumsum(lengths_left_shard),
            ]
        )
        offsets_right_local = jnp.concatenate(
            [
                jnp.zeros(1, dtype=lengths_right_shard.dtype),
                jnp.cumsum(lengths_right_shard),
            ]
        )
        return _jax_concat_2d_jagged(
            values_left_shard,
            offsets_left_local,
            values_right_shard,
            offsets_right_local,
            local_out,
        )

    values_left_spec = P(*([_DP_AXIS] + [None] * (values_left.ndim - 1)))
    values_right_spec = P(*([_DP_AXIS] + [None] * (values_right.ndim - 1)))
    out_spec = P(*([_DP_AXIS] + [None] * (values_left.ndim - 1)))
    result = jax.shard_map(
        _per_device,
        mesh=mesh,
        in_specs=(values_left_spec, P(_DP_AXIS), values_right_spec, P(_DP_AXIS)),
        out_specs=out_spec,
        check_vma=False,
        # pyrefly: ignore [bad-argument-count]
    )(values_left, lengths_left, values_right, lengths_right)
    return _with_sharding_constraint(result, out_spec)


@op(torch.ops.recir.split_2d_jagged.default)  # pyre-ignore[56]
def _recir_split_2d_jagged(
    values: Any,
    offsets_left: Any,
    offsets_right: Any,
    seq_len_a: int,
    seq_len_b: int,
) -> Any:
    """recir::split_2d_jagged lowering: inverse interleave inside shard_map.

    seq_len_a / seq_len_b are STATIC output sizes supplied by the caller (the mot
    path passes concrete totals), required because the split sizes are otherwise
    data-dependent.
    """
    world_size = len(jax.devices())
    if world_size <= 1:
        return _jax_split_2d_jagged(
            values, offsets_left, offsets_right, seq_len_a, seq_len_b
        )

    lengths_left = offsets_left[1:] - offsets_left[:-1]
    lengths_right = offsets_right[1:] - offsets_right[:-1]
    assert seq_len_a % world_size == 0 and seq_len_b % world_size == 0, (
        f"split sizes ({seq_len_a},{seq_len_b}) must be divisible by world_size={world_size}"
    )
    local_a = seq_len_a // world_size
    local_b = seq_len_b // world_size
    mesh = Mesh(jax.devices(), (_DP_AXIS,))

    def _per_device(
        values_shard: Any, lengths_left_shard: Any, lengths_right_shard: Any
    ) -> Any:
        offsets_left_local = jnp.concatenate(
            [
                jnp.zeros(1, dtype=lengths_left_shard.dtype),
                jnp.cumsum(lengths_left_shard),
            ]
        )
        offsets_right_local = jnp.concatenate(
            [
                jnp.zeros(1, dtype=lengths_right_shard.dtype),
                jnp.cumsum(lengths_right_shard),
            ]
        )
        return _jax_split_2d_jagged(
            values_shard, offsets_left_local, offsets_right_local, local_a, local_b
        )

    values_spec = P(*([_DP_AXIS] + [None] * (values.ndim - 1)))
    out_spec = P(*([_DP_AXIS] + [None] * (values.ndim - 1)))
    # pyrefly: ignore [not-iterable]
    left, right = jax.shard_map(
        _per_device,
        mesh=mesh,
        in_specs=(values_spec, P(_DP_AXIS), P(_DP_AXIS)),
        out_specs=(out_spec, out_spec),
        check_vma=False,
        # pyrefly: ignore [bad-argument-count]
    )(values, lengths_left, lengths_right)
    return (
        _with_sharding_constraint(left, out_spec),
        _with_sharding_constraint(right, out_spec),
    )
