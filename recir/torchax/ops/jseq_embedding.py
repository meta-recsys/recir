# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""torchax JAX lowering of ``recir::ringshift_embedding`` to the ring-shift kernel.

Importing this module registers, on the recir/torchax TPU path:
  - torch.ops.recir.ringshift_embedding          -> local ring-shift driver (fwd)
  - torch.ops.recir.ringshift_embedding_backward -> the ring-shift table grad (bwd)
  - torch.ops.recir.fsdp_embedding               -> shard_map all-gather (FSDP, fwd) +
                                                    reduce-scatter (bwd)

The DP (replicated) path uses plain ``aten.embedding`` (lowered to ``jnp.take`` in
``jaten.py``), not a custom op.

The op (schema + CPU fallback + autograd) is defined in
``recir.ops.seq_embedding_op``. When the shard-map backend already binds
``"torch_dist"`` as a manual mesh axis, this lowering calls the per-device
driver directly. Otherwise it consumes the active RecIR GSPMD mesh context and
wraps the same driver in ``jax.shard_map``.

Data layout:
  * ``ids`` reshapes to ``[W, max_local_ids]`` (leading axis = the mesh axis), so
    device ``d`` holds its own ids; batch % W == 0.
  * ``weight`` ``[vocab, D]`` is block-sharded -> ``[W, V_local, D]``; vocab % W == 0.
  * Entries past ``lengths`` are padding, position-spread so they stay balanced
    across ranks; masked downstream.
"""

import functools
import os
from typing import Any, cast, Tuple

# Define torch.ops.recir.ringshift_embedding (schema + fallbacks) before registering.
import jax
import jax.numpy as jnp
import recir.ops.seq_embedding_op  # noqa: F401
import torch
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from recir.ops.pallas.embeddings import (
    ringshift_embedding_lookup_block_backward_local,
    ringshift_embedding_lookup_block_local,
)
from recir.ops.pallas.embeddings.ringshift import fan_out_pooled_grad
from recir.torchax.gspmd_context import current_gspmd_mesh_context
from recir.torchax.ops.decorators import op

_AXIS = "torch_dist"
_DATA_AXIS = "gspmd_data"
# Per-wave VMEM recv buffer for the ring-shift kernel (see ringshift docstring).
_RECV_CAPACITY: int = int(os.environ.get("RECIR_RINGSHIFT_RECV_CAPACITY", "8192"))
# Exchange the cross-host embedding (fwd) + gradient (bwd) all-to-all in bf16 to
# halve the ICI bytes (the S=4096 bottleneck). Table + grad accumulation stay f32;
# inputs/outputs stay f32. Default off.
_EXCHANGE_BF16: bool = os.environ.get("RECIR_RINGSHIFT_EXCHANGE_BF16", "0") == "1"


def _validate_host_local_fsdp_groups(devices: Any, fsdp_size: int) -> None:
    flat_devices = list(devices.flat) if hasattr(devices, "flat") else list(devices)
    if fsdp_size <= 0 or len(flat_devices) % fsdp_size != 0:
        raise ValueError(
            "ringshift FSDP groups must cover the mesh exactly; "
            f"devices={len(flat_devices)}, fsdp={fsdp_size}"
        )
    for offset in range(0, len(flat_devices), fsdp_size):
        process_indices = {
            int(process_index)
            for device in flat_devices[offset : offset + fsdp_size]
            if (process_index := getattr(device, "process_index", None)) is not None
        }
        if len(process_indices) > 1:
            raise ValueError(
                "ringshift FSDP groups must not cross JAX process boundaries; "
                f"process_indices={sorted(process_indices)}"
            )


def _ringshift_mesh_layout() -> Tuple[Any, Any, int, bool]:
    """Return the concrete mesh and resources visible to the active JAX trace."""
    gspmd_context = current_gspmd_mesh_context()
    if gspmd_context is not None:
        if gspmd_context.hybrid_fsdp:
            _validate_host_local_fsdp_groups(
                gspmd_context.mesh.devices,
                gspmd_context.fsdp_axis_size,
            )
        return (
            gspmd_context.mesh,
            gspmd_context.batch_axis_resource,
            gspmd_context.fsdp_axis_size,
            gspmd_context.hybrid_fsdp,
        )
    devices = jax.devices()
    current_mesh = jax.sharding.get_abstract_mesh()
    fsdp_size = int(current_mesh.shape.get(_AXIS, len(devices)))
    data_size = len(devices) // fsdp_size
    if fsdp_size <= 0 or data_size <= 0 or fsdp_size * data_size != len(devices):
        raise ValueError(
            "ringshift mesh must cover every device; "
            f"devices={len(devices)}, data={data_size}, fsdp={fsdp_size}"
        )
    if data_size == 1:
        return Mesh(devices, (_AXIS,)), _AXIS, fsdp_size, False

    device_grid = [
        devices[offset : offset + fsdp_size]
        for offset in range(0, len(devices), fsdp_size)
    ]
    _validate_host_local_fsdp_groups(devices, fsdp_size)
    return (
        Mesh(device_grid, (_DATA_AXIS, _AXIS)),
        (_DATA_AXIS, _AXIS),
        fsdp_size,
        True,
    )


def _device_mesh() -> Any:
    # The non-ring embedding lowerings retain their full-world 1-D contract.
    return Mesh(jax.devices(), (_AXIS,))


def _manual_world_size() -> int | None:
    current_mesh = jax.sharding.get_abstract_mesh()
    if _AXIS not in current_mesh.manual_axes:
        return None
    if _AXIS not in current_mesh.shape:
        raise ValueError(f"manual mesh axis {_AXIS!r} has no static size")
    world_size = int(current_mesh.shape[_AXIS])
    if world_size <= 0:
        raise ValueError(f"manual mesh axis {_AXIS!r} has invalid size {world_size}")
    gspmd_context = current_gspmd_mesh_context()
    if gspmd_context is not None and gspmd_context.fsdp_axis_size != world_size:
        raise ValueError(
            "configured ringshift FSDP axis does not match the active manual "
            f"mesh: configured={gspmd_context.fsdp_axis_size}, manual={world_size}"
        )
    return world_size


def _prep_local(
    ids: Any, lengths: Any, weight: Any, world_size: int
) -> Tuple[Any, Any, int, int, int]:
    """Prepare one device's shards under an already-active manual mesh."""
    batch, seq = ids.shape
    v_local, dim = weight.shape
    max_local_ids = batch * seq
    global_vocab = v_local * world_size

    pos = jnp.arange(seq, dtype=jnp.int32)[None, :]
    is_pad = pos >= lengths.astype(jnp.int32)[:, None]
    # int64, not int32: ids span [0, global_vocab) and prod's largest table is
    # 3e9 rows, past int32's 2147483647. An int32 cast here wraps the high ids
    # NEGATIVE without failing, and _block_to_mod_ids downstream builds an
    # intermediate that also reaches vocab. Requires jax_enable_x64; without it
    # JAX silently narrows this back to int32 and only warns.
    rank = jax.lax.axis_index(_AXIS)
    flat_spread = (
        rank * max_local_ids + jnp.arange(max_local_ids, dtype=jnp.int64)
    ) % global_vocab
    ids64 = ids.astype(jnp.int64)
    # Checked on the RESULT of the cast, not on jax.config: JAX narrows int64 to
    # int32 when jax_enable_x64 is off and only warns, which would restore the
    # wraparound silently. Tables that fit in int32 are unaffected either way.
    if global_vocab > 2**31 - 1 and ids64.dtype != jnp.int64:
        raise RuntimeError(
            f"vocab {global_vocab} exceeds int32 but the id path came out as "
            f"{ids64.dtype}: ids above 2147483647 would wrap to negative "
            f"row indices. Set JAX_ENABLE_X64=1."
        )
    ids_dist = cast(
        jax.Array,
        jnp.where(
            is_pad,
            flat_spread.reshape(batch, seq),
            ids64,
        ),
    )
    count = jnp.asarray(max_local_ids, dtype=jnp.int32)
    return ids_dist.reshape(max_local_ids), count, batch, seq, dim


def _pooled_lookup_local_impl(
    ids: Any,
    lengths: Any,
    weight: Any,
    padding_idx: int,
    pooling: str,
    world_size: int,
) -> Any:
    """Run pooled lookup on one device inside an active manual mesh."""
    local_ids = ids.reshape(-1).astype(jnp.int32)
    bag_lengths = lengths.reshape(-1).astype(jnp.int32)
    dim = weight.shape[1]
    out = ringshift_embedding_lookup_block_local(
        local_ids,
        None,
        weight.astype(jnp.float32),
        axis_name=_AXIS,
        world_size=world_size,
        recv_capacity=_RECV_CAPACITY,
        exchange_bf16=_EXCHANGE_BF16,
        pooling=pooling,
        bag_lengths=bag_lengths,
    )
    return out.reshape(*lengths.shape, dim).astype(weight.dtype)


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5))
def _pooled_lookup_local(
    ids: Any,
    lengths: Any,
    weight: Any,
    padding_idx: int,
    pooling: str,
    world_size: int,
) -> Any:
    return _pooled_lookup_local_impl(
        ids, lengths, weight, padding_idx, pooling, world_size
    )


def _pooled_lookup_local_fwd(
    ids: Any,
    lengths: Any,
    weight: Any,
    padding_idx: int,
    pooling: str,
    world_size: int,
) -> Tuple[Any, Tuple[Any, Any, Any]]:
    out = _pooled_lookup_local_impl(
        ids, lengths, weight, padding_idx, pooling, world_size
    )
    return out, (ids, lengths, weight)


def _pooled_lookup_local_bwd(
    padding_idx: int,
    pooling: str,
    world_size: int,
    residuals: Tuple[Any, Any, Any],
    grad_out: Any,
) -> Tuple[None, None, Any]:
    ids, lengths, weight = residuals
    grad_weight = _pooled_backward_local(
        grad_out,
        ids,
        lengths,
        weight,
        padding_idx,
        pooling,
        world_size,
    )
    return None, None, grad_weight


_pooled_lookup_local.defvjp(_pooled_lookup_local_fwd, _pooled_lookup_local_bwd)


def _pooled_lookup(
    ids: Any,
    lengths: Any,
    weight: Any,
    pooling: str,
    padding_idx: int = -1,
) -> Any:
    """Pooled forward: flat per-rank ids plus bag lengths to pooled rows."""
    manual_world_size = _manual_world_size()
    if manual_world_size is not None:
        return _pooled_lookup_local(
            ids, lengths, weight, padding_idx, pooling, manual_world_size
        )

    mesh, batch_resource, world_size, _hybrid = _ringshift_mesh_layout()
    rows = ids.shape[0]
    if rows != mesh.size:
        raise ValueError(
            "pooled ids must have one row per device (axis 0 is the mesh axis): "
            f"got {rows} rows for mesh size {mesh.size}"
        )

    def _per_device(local_ids: Any, local_lengths: Any, local_weight: Any) -> Any:
        return _pooled_lookup_local(
            local_ids,
            local_lengths,
            local_weight,
            padding_idx,
            pooling,
            world_size,
        )

    return jax.shard_map(
        _per_device,
        mesh=mesh,
        in_specs=(
            P(batch_resource, None),
            P(batch_resource, None),
            P(_AXIS, None),
        ),
        out_specs=P(batch_resource, None, None),
        check_vma=False,
        # pyrefly: ignore [bad-argument-count]
    )(ids, lengths, weight)


def _pooled_backward_local(
    grad_out: Any,
    ids: Any,
    lengths: Any,
    weight: Any,
    padding_idx: int,
    pooling: str,
    world_size: int,
) -> Any:
    """Fan a bag gradient back to ids, then use the existing table backward."""
    local_ids = ids.reshape(-1).astype(jnp.int32)
    bag_lengths = lengths.reshape(-1).astype(jnp.int32)
    dim = weight.shape[1]
    cotangent = fan_out_pooled_grad(
        grad_out.reshape(-1, dim).astype(jnp.float32),
        bag_lengths,
        max_local_ids=local_ids.shape[0],
        pooling=pooling,
    )
    cotangent = cast(
        jax.Array,
        jnp.where(local_ids[:, None] == padding_idx, 0, cotangent),
    )
    count = jnp.sum(bag_lengths).astype(jnp.int32)
    grad_weight = ringshift_embedding_lookup_block_backward_local(
        cotangent,
        local_ids,
        count,
        axis_name=_AXIS,
        world_size=world_size,
        recv_capacity=_RECV_CAPACITY,
        v_local=weight.shape[0],
        embedding_dim=dim,
        exchange_bf16=_EXCHANGE_BF16,
    )
    return grad_weight.astype(weight.dtype)


@op(torch.ops.recir.ringshift_embedding)  # pyre-ignore[56]
def _jringshift_embedding(
    ids: Any,
    lengths: Any,
    weight: Any,
    padding_idx: int = -1,
    pooling: str = "none",
) -> Any:
    if pooling != "none":
        return _pooled_lookup(ids, lengths, weight, pooling, padding_idx)

    del padding_idx  # Forward lookup returns the stored padding row, like F.embedding.

    manual_world_size = _manual_world_size()
    if manual_world_size is not None:
        local_ids, count, batch, seq, dim = _prep_local(
            ids, lengths, weight, manual_world_size
        )
        out = ringshift_embedding_lookup_block_local(
            local_ids,
            count,
            weight.astype(jnp.float32),
            axis_name=_AXIS,
            world_size=manual_world_size,
            recv_capacity=_RECV_CAPACITY,
            exchange_bf16=_EXCHANGE_BF16,
        )
        return out.reshape(batch, seq, dim).astype(weight.dtype)

    mesh, batch_resource, world_size, _hybrid = _ringshift_mesh_layout()

    def _per_device(local_ids_input: Any, local_lengths: Any, local_weight: Any) -> Any:
        local_ids, count, batch, seq, dim = _prep_local(
            local_ids_input,
            local_lengths,
            local_weight,
            world_size,
        )
        # Citrine C3: keep persistent storage compact and cast only inside the op.
        out = ringshift_embedding_lookup_block_local(
            local_ids,
            count,
            local_weight.astype(jnp.float32),
            axis_name=_AXIS,
            world_size=world_size,
            recv_capacity=_RECV_CAPACITY,
            exchange_bf16=_EXCHANGE_BF16,
        )
        return out.reshape(batch, seq, dim).astype(local_weight.dtype)

    return jax.shard_map(
        _per_device,
        mesh=mesh,
        in_specs=(
            P(batch_resource, None),
            P(batch_resource),
            P(_AXIS, None),
        ),
        out_specs=P(batch_resource, None, None),
        check_vma=False,
        # pyrefly: ignore [bad-argument-count]
    )(ids, lengths, weight)


@op(torch.ops.recir.ringshift_embedding_backward)  # pyre-ignore[56]
def _jringshift_embedding_backward(
    grad_out: Any,
    ids: Any,
    lengths: Any,
    weight: Any,
    padding_idx: int = -1,
    pooling: str = "none",
) -> Any:
    if pooling != "none":
        manual_world_size = _manual_world_size()
        if manual_world_size is not None:
            return _pooled_backward_local(
                grad_out,
                ids,
                lengths,
                weight,
                padding_idx,
                pooling,
                manual_world_size,
            )

        mesh, batch_resource, world_size, hybrid = _ringshift_mesh_layout()

        def _pooled_per_device(
            local_grad_out: Any,
            local_ids: Any,
            local_lengths: Any,
            local_weight: Any,
        ) -> Any:
            grad_weight = _pooled_backward_local(
                local_grad_out,
                local_ids,
                local_lengths,
                local_weight,
                padding_idx,
                pooling,
                world_size,
            )
            if hybrid:
                grad_weight = jax.lax.psum(grad_weight, _DATA_AXIS)
            return grad_weight

        return jax.shard_map(
            _pooled_per_device,
            mesh=mesh,
            in_specs=(
                P(batch_resource, None, None),
                P(batch_resource, None),
                P(batch_resource, None),
                P(_AXIS, None),
            ),
            out_specs=P(_AXIS, None),
            check_vma=False,
            # pyrefly: ignore [bad-argument-count]
        )(grad_out, ids, lengths, weight)

    manual_world_size = _manual_world_size()
    if manual_world_size is not None:
        local_ids, count, _batch, _seq, dim = _prep_local(
            ids, lengths, weight, manual_world_size
        )
        cotangent = cast(
            jax.Array,
            jnp.where(
                ids[..., None] == padding_idx,
                0,
                grad_out.astype(jnp.float32),
            ),
        )
        cotangent = cotangent.reshape(
            local_ids.shape[0],
            dim,
        )
        grad_weight = ringshift_embedding_lookup_block_backward_local(
            cotangent,
            local_ids,
            count,
            axis_name=_AXIS,
            world_size=manual_world_size,
            recv_capacity=_RECV_CAPACITY,
            v_local=weight.shape[0],
            embedding_dim=dim,
            exchange_bf16=_EXCHANGE_BF16,
        )
        return grad_weight.astype(weight.dtype)

    mesh, batch_resource, world_size, hybrid = _ringshift_mesh_layout()

    def _per_device(
        local_grad_out: Any,
        local_ids_input: Any,
        local_lengths: Any,
        local_weight: Any,
    ) -> Any:
        local_ids, count, _batch, _seq, dim = _prep_local(
            local_ids_input,
            local_lengths,
            local_weight,
            world_size,
        )
        cotangent = cast(
            jax.Array,
            jnp.where(
                local_ids_input[..., None] == padding_idx,
                0,
                local_grad_out.astype(jnp.float32),
            ),
        ).reshape(local_ids.shape[0], dim)
        grad_weight = ringshift_embedding_lookup_block_backward_local(
            cotangent,
            local_ids,
            count,
            axis_name=_AXIS,
            world_size=world_size,
            recv_capacity=_RECV_CAPACITY,
            v_local=local_weight.shape[0],
            embedding_dim=dim,
            exchange_bf16=_EXCHANGE_BF16,
        )
        if hybrid:
            grad_weight = jax.lax.psum(grad_weight, _DATA_AXIS)
        return grad_weight.astype(local_weight.dtype)

    return jax.shard_map(
        _per_device,
        mesh=mesh,
        in_specs=(
            P(batch_resource, None, None),
            P(batch_resource, None),
            P(batch_resource),
            P(_AXIS, None),
        ),
        out_specs=P(_AXIS, None),
        check_vma=False,
        # pyrefly: ignore [bad-argument-count]
    )(grad_out, ids, lengths, weight)


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
