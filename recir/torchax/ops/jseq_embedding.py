# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""torchax JAX lowering of ``recir::ringshift_embedding`` to the ring-shift kernel.

Importing this module registers, on the recir/torchax TPU path:
  - torch.ops.recir.ringshift_embedding          -> ringshift_embedding_lookup_block (fwd)
  - torch.ops.recir.ringshift_embedding_backward -> the ring-shift table grad (bwd)
  - torch.ops.recir.fsdp_embedding               -> shard_map all-gather (FSDP, fwd) +
                                                    reduce-scatter (bwd)

The DP (replicated) path uses plain ``aten.embedding`` (lowered to ``jnp.take`` in
``jaten.py``), not a custom op.

The op (schema + CPU fallback + autograd) is defined in
``recir.ops.seq_embedding_op``. Because the lowering wraps the ring-shift kernel
(which carries its own ``shard_map`` over the ``"torch_dist"`` mesh) it composes
into recir's single compiled GSPMD program -- the cross-host all-to-all is
scheduled together with the outer data-parallel collectives, so it does NOT
deadlock the way the graph-break embedding path did.

Data layout:
  * ``ids`` reshapes to ``[W, max_local_ids]`` (leading axis = the mesh axis), so
    device ``d`` holds its own ids; batch % W == 0.
  * ``weight`` ``[vocab, D]`` is block-sharded -> ``[W, V_local, D]``; vocab % W == 0.
  * Entries past ``lengths`` are padding, position-spread so they stay balanced
    across ranks; masked downstream.
"""

import os
from typing import Any, Tuple

# Define torch.ops.recir.ringshift_embedding (schema + fallbacks) before registering.
import jax
import jax.numpy as jnp
import recir.ops.seq_embedding_op  # noqa: F401
import torch
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from recir.ops.pallas.embeddings import ringshift_embedding_lookup_block
from recir.torchax.ops.decorators import op

_AXIS = "torch_dist"
# Per-wave VMEM recv buffer for the ring-shift kernel (see ringshift docstring).
_RECV_CAPACITY: int = int(os.environ.get("RECIR_RINGSHIFT_RECV_CAPACITY", "8192"))
# Exchange the cross-host embedding (fwd) + gradient (bwd) all-to-all in bf16 to
# halve the ICI bytes (the S=4096 bottleneck). Table + grad accumulation stay f32;
# inputs/outputs stay f32. Default off.
_EXCHANGE_BF16: bool = os.environ.get("RECIR_RINGSHIFT_EXCHANGE_BF16", "0") == "1"


def _device_mesh() -> Any:
    # Matches recir's Mesh(_get_global_devices(), ("torch_dist",)); imported
    # directly from jax to avoid a recir.lowering import cycle.
    return Mesh(jax.devices(), (_AXIS,))


def _prep(ids: Any, lengths: Any, weight: Any) -> Tuple[Any, Any, Any, int, int, int]:
    """Shared fwd/bwd prep: build ring-shift inputs from (ids, lengths, weight).

    Returns (local_ids[W, max_local_ids], count[W], weight3d[W, V_local, D],
    batch B, seq S, dim D).
    """
    world_size = len(jax.devices())
    batch, seq = ids.shape
    vocab, dim = weight.shape
    if vocab % world_size != 0:
        raise ValueError(f"vocab {vocab} not divisible by world_size {world_size}")
    if batch % world_size != 0:
        raise ValueError(f"batch {batch} not divisible by world_size {world_size}")
    v_local = vocab // world_size
    b_local = batch // world_size
    max_local_ids = b_local * seq

    # Block-layout table: rank d owns contiguous rows [d*v_local:(d+1)*v_local].
    weight3d = weight.reshape(world_size, v_local, dim)

    # Distribute padding ids so they don't all route to one rank. Padding slots
    # (position >= length) get a position-spread id; count stays full so the ring
    # traffic is balanced (uniform), and padding embeddings are masked downstream.
    pos = jnp.arange(seq, dtype=jnp.int32)[None, :]
    is_pad = pos >= lengths.astype(jnp.int32)[:, None]
    flat_spread = (jnp.arange(batch * seq, dtype=jnp.int32) % vocab).reshape(batch, seq)
    ids_dist = jnp.where(is_pad, flat_spread, ids.astype(jnp.int32))

    # [rows, cols] (sharded on axis 0) -> [W, max_local_ids] (leading axis = the
    # mesh axis), so device d holds exactly its own ids.
    # pyrefly: ignore [missing-attribute]
    local_ids = ids_dist.reshape(world_size, max_local_ids)
    count = jnp.full((world_size,), max_local_ids, dtype=jnp.int32)
    return local_ids, count, weight3d, batch, seq, dim


@op(torch.ops.recir.ringshift_embedding)  # pyre-ignore[56]
def _jringshift_embedding(ids: Any, lengths: Any, weight: Any) -> Any:
    local_ids, count, weight3d, batch, seq, dim = _prep(ids, lengths, weight)
    out = ringshift_embedding_lookup_block(
        local_ids,
        count,
        weight3d,
        mesh=_device_mesh(),
        axis_name=_AXIS,
        recv_capacity=_RECV_CAPACITY,
        exchange_bf16=_EXCHANGE_BF16,
    )
    out = out.reshape(batch, seq, dim)
    out = jax.lax.with_sharding_constraint(
        out, NamedSharding(_device_mesh(), P(_AXIS, None, None))
    )
    return out


@op(torch.ops.recir.ringshift_embedding_backward)  # pyre-ignore[56]
def _jringshift_embedding_backward(
    grad_out: Any, ids: Any, lengths: Any, weight: Any
) -> Any:
    local_ids, count, weight3d, _batch, _seq, dim = _prep(ids, lengths, weight)

    def _fwd(w3: Any, local_ids: Any = local_ids, count: Any = count) -> Any:
        return ringshift_embedding_lookup_block(
            local_ids,
            count,
            w3,
            mesh=_device_mesh(),
            axis_name=_AXIS,
            recv_capacity=_RECV_CAPACITY,
            exchange_bf16=_EXCHANGE_BF16,
        )

    _, vjp_fn = jax.vjp(_fwd, weight3d)
    cotangent = grad_out.reshape(local_ids.shape[0], local_ids.shape[1], dim)
    (grad_weight3d,) = vjp_fn(cotangent)
    return grad_weight3d.reshape(weight.shape[0], dim)


@op(torch.ops.recir.shard_batch0)  # pyre-ignore[56]
def _jshard_batch0(x: Any) -> Any:
    spec = P(_AXIS, *([None] * (x.ndim - 1)))
    return jax.lax.with_sharding_constraint(x, NamedSharding(_device_mesh(), spec))


@op(torch.ops.recir.fsdp_embedding)  # pyre-ignore[56]
def _jfsdp_embedding(ids: Any, weight: Any) -> Any:
    # True FSDP forward: all-gather the Shard(0) weight inside a shard_map -- the
    # full [vocab, D] table is a transient, persistent storage stays sharded --
    # then gather the local ids. Single device -> plain gather.
    world_size = len(jax.devices())
    if world_size <= 1:
        return jnp.take(weight, ids, axis=0)

    def _per_device(w_shard: Any, ids_local: Any) -> Any:
        full = jax.lax.all_gather(w_shard, _AXIS, axis=0, tiled=True)
        return jnp.take(full, ids_local, axis=0)

    return jax.shard_map(
        _per_device,
        mesh=_device_mesh(),
        in_specs=(P(_AXIS, None), P(_AXIS, *([None] * (ids.ndim - 1)))),
        out_specs=P(_AXIS, *([None] * ids.ndim)),
        check_vma=False,
        # pyrefly: ignore [bad-argument-count]
    )(weight, ids)


@op(torch.ops.recir.fsdp_embedding_backward)  # pyre-ignore[56]
def _jfsdp_embedding_backward(grad_out: Any, ids: Any, weight: Any) -> Any:
    # True FSDP backward: scatter-add the local grad into a transient full
    # [vocab, D], then reduce-scatter so each device keeps its [v_local, D] shard
    # (matches the Shard(0) param). No full replicated grad.
    vocab: int = weight.shape[0]
    dim: int = weight.shape[1]
    weight_dtype: Any = weight.dtype
    world_size = len(jax.devices())
    if world_size <= 1:
        gw = jnp.zeros((vocab, dim), weight_dtype)
        return gw.at[ids.reshape(-1)].add(
            grad_out.reshape(-1, dim).astype(weight_dtype)
        )

    def _per_device(grad_local: Any, ids_local: Any) -> Any:
        gw = jnp.zeros((vocab, dim), weight_dtype)
        gw = gw.at[ids_local.reshape(-1)].add(
            grad_local.reshape(-1, dim).astype(weight_dtype)
        )
        return jax.lax.psum_scatter(gw, _AXIS, scatter_dimension=0, tiled=True)

    return jax.shard_map(
        _per_device,
        mesh=_device_mesh(),
        in_specs=(
            P(_AXIS, *([None] * (grad_out.ndim - 1))),
            P(_AXIS, *([None] * (ids.ndim - 1))),
        ),
        out_specs=P(_AXIS, None),  # [v_local, D] per device -> sharded [vocab, D]
        check_vma=False,
        # pyrefly: ignore [bad-argument-count]
    )(grad_out, ids)
