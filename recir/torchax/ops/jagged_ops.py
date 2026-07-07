# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

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
    assert total_L % world_size == 0, (
        f"total_L={total_L} must be divisible by world_size={world_size} for batch-aligned sharding"
    )
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


def _jax_concat_2d_jagged(
    values_left: Any,
    offsets_left: Any,
    values_right: Any,
    offsets_right: Any,
    out_total: int,
) -> Any:
    """Per-sample interleave: combined[b] = cat(left[b], right[b]).

    Each output row must be gathered from either the left or the right values,
    depending on which sample (batch index) and intra-sample offset it maps to.
    The combined offsets (offsets_left + offsets_right) are the monotonically
    increasing prefix sums of the per-sample combined lengths, so a flat output
    position's sample is the bucket it falls into -- which is exactly what
    searchsorted computes (and it works under jit, unlike a data-dependent
    Python loop over samples).
    """
    batch_size = offsets_left.shape[0] - 1
    combined_offsets = offsets_left + offsets_right  # cumulative, [batch_size + 1]
    lengths_left = offsets_left[1:] - offsets_left[:-1]  # [batch_size]
    positions = jnp.arange(out_total)
    # Sample (batch) index each flat output position belongs to.
    batch_idx = jnp.clip(
        jnp.searchsorted(combined_offsets[1:], positions, side="right"),
        0,
        batch_size - 1,
    )
    intra_offset = positions - combined_offsets[batch_idx]  # offset within the sample
    is_left = intra_offset < lengths_left[batch_idx]
    # Guard against an empty side: when one operand has 0 rows (e.g. an empty
    # left jagged tensor, or a per-device shard that received no rows under
    # shard_map), ``values.shape[0] - 1`` is -1 and the gather of a size-1
    # slice from a zero-length dim is out of range。
    if values_left.shape[0] == 0:
        gathered_left = jnp.zeros(
            (out_total,) + values_left.shape[1:], dtype=values_left.dtype
        )
    else:
        source_left = jnp.clip(
            offsets_left[batch_idx] + intra_offset, 0, values_left.shape[0] - 1
        )
        gathered_left = values_left[source_left]
    if values_right.shape[0] == 0:
        gathered_right = jnp.zeros(
            (out_total,) + values_right.shape[1:], dtype=values_right.dtype
        )
    else:
        source_right = jnp.clip(
            offsets_right[batch_idx] + (intra_offset - lengths_left[batch_idx]),
            0,
            values_right.shape[0] - 1,
        )
        gathered_right = values_right[source_right]
    out = jnp.where(
        _bcast_mask(is_left, gathered_left.ndim), gathered_left, gathered_right
    )
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
    into the interleaved `values` buffer. searchsorted on the per-side cumulative
    offsets maps a flat output position to its sample (the same trick as the
    forward op, and jit-safe). left[p] lives at values[p + offsets_right[sample]];
    right[p] lives at values[p + offsets_left[sample + 1]].
    """
    batch_size = offsets_left.shape[0] - 1
    # Guard against an empty input buffer
    if values.shape[0] == 0:
        left = jnp.zeros((seq_len_a,) + values.shape[1:], dtype=values.dtype)
        right = jnp.zeros((seq_len_b,) + values.shape[1:], dtype=values.dtype)
        return left, right
    positions_left = jnp.arange(seq_len_a)
    batch_idx_left = jnp.clip(
        jnp.searchsorted(offsets_left[1:], positions_left, side="right"),
        0,
        batch_size - 1,
    )
    left = values[
        jnp.clip(positions_left + offsets_right[batch_idx_left], 0, values.shape[0] - 1)
    ]
    left = jnp.where(
        _bcast_mask(positions_left < offsets_left[batch_size], left.ndim), left, 0
    )
    positions_right = jnp.arange(seq_len_b)
    batch_idx_right = jnp.clip(
        jnp.searchsorted(offsets_right[1:], positions_right, side="right"),
        0,
        batch_size - 1,
    )
    right = values[
        jnp.clip(
            positions_right + offsets_left[batch_idx_right + 1], 0, values.shape[0] - 1
        )
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
