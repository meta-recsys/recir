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
        return result.at[dest.reshape(-1)].add(src.reshape(-1))
    d: int = dense.shape[2]
    src = jnp.where(mask[:, :, None], dense, jnp.zeros_like(dense))
    result = jnp.zeros((total_L, d), dtype=dense.dtype)
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
    )(values, lengths)
    result = _with_sharding_constraint(result, dense_out_spec)
    return result
