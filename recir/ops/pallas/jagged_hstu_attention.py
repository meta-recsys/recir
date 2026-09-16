# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Packed-input HSTU SiLU attention for TPU via Pallas.

The native path keeps Q/K/V packed in HBM. One program owns one sequence,
stages an eight-row-aligned superset in VMEM, and writes a disjoint dense
``[H, N, Dv]`` result. Seven safe rows are appended internally so the final
aligned DMA never reads beyond the physical input buffer.

Runtime-sized DMA slices are supported by native Mosaic but not by ordinary
Pallas interpret discharge. ``interpret=True`` therefore exercises the same
packed API through the existing dense Pallas interpreter path; native DMA must
be validated on a real TPU.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from typing import Any, cast

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from recir.ops.pallas.hstu_attention import (
    hstu_attention_reference,
    pallas_hstu_attention,
)

_ALIGNMENT_ROWS = 8
_SAFE_TAIL_ROWS = _ALIGNMENT_ROWS - 1
_DEFAULT_BLOCK = 256
_AXIS = "torch_dist"
_PAD_SEQS_PER_DEVICE: int | None = None
_PADDED_NUM_SEQUENCES: int | None = None


def set_pad_seqs_per_device(
    real_batch_size: int,
    padded_batch_size: int,
    world_size: int,
) -> int:
    """Configure trailing PRT dummy sequences present on each token shard."""
    global _PAD_SEQS_PER_DEVICE, _PADDED_NUM_SEQUENCES
    if world_size < 1:
        raise ValueError("world_size must be positive")
    if real_batch_size % world_size or padded_batch_size % world_size:
        raise ValueError("real and padded batch sizes must be divisible by world_size")
    local_batch = padded_batch_size // world_size
    pad_sequences = local_batch - real_batch_size // world_size
    if not 0 <= pad_sequences < local_batch:
        raise ValueError("configured padding must leave at least one real sequence")
    _PAD_SEQS_PER_DEVICE = pad_sequences
    _PADDED_NUM_SEQUENCES = padded_batch_size
    return pad_sequences


def clear_pad_seqs_per_device() -> None:
    """Clear PRT dummy-sequence configuration between compilations."""
    global _PAD_SEQS_PER_DEVICE, _PADDED_NUM_SEQUENCES
    _PAD_SEQS_PER_DEVICE = None
    _PADDED_NUM_SEQUENCES = None


def _pad_seqs_per_device(padded_batch_size: int) -> int:
    if _PAD_SEQS_PER_DEVICE is None or _PADDED_NUM_SEQUENCES is None:
        raise RuntimeError("packed Pallas HSTU padding rows were not configured")
    if padded_batch_size != _PADDED_NUM_SEQUENCES:
        raise ValueError(
            "packed Pallas HSTU traced batch does not match its padding configuration"
        )
    return _PAD_SEQS_PER_DEVICE


def _assert_batch_aligned(
    seq_lengths: Any,
    world_size: int,
    local_token_count: int,
) -> None:
    if isinstance(seq_lengths, jax.core.Tracer):
        return
    local_batch = seq_lengths.shape[0] // world_size
    group_sums = seq_lengths.reshape(world_size, local_batch).sum(axis=1)
    sums = [int(value) for value in group_sums]
    if any(value != local_token_count for value in sums):
        raise ValueError(
            "packed Pallas HSTU requires each token shard to contain complete "
            f"sequences; per-device counts={sums}, expected={local_token_count}"
        )


def _drop_pad_rows(
    seq_lengths: Any,
    num_targets: Any,
    padded_batch_size: int,
) -> tuple[Any, Any]:
    pad_sequences = _pad_seqs_per_device(padded_batch_size)
    keep = int(seq_lengths.shape[0]) - pad_sequences
    if keep <= 0:
        raise ValueError("packed Pallas HSTU padding removed every local sequence")
    return seq_lengths[:keep], num_targets[:keep]


def _packed_to_dense(
    x: jax.Array,
    seq_offsets: jax.Array,
    max_seq_len: int,
) -> tuple[jax.Array, jax.Array]:
    """Gather packed ``[T, H, D]`` data into dense ``[B, H, N, D]``."""
    lengths = seq_offsets[1:] - seq_offsets[:-1]
    positions = jnp.arange(max_seq_len, dtype=seq_offsets.dtype)
    packed_rows = seq_offsets[:-1, None] + positions[None, :]
    valid = positions[None, :] < lengths[:, None]
    packed_rows = jnp.clip(packed_rows, 0, x.shape[0] - 1)
    dense = jnp.take(x, packed_rows, axis=0).transpose(0, 2, 1, 3)
    return jnp.where(valid[:, None, :, None], dense, 0), lengths


def _dense_to_packed(
    x: jax.Array,
    seq_offsets: jax.Array,
    total_length: int,
) -> jax.Array:
    """Gather dense ``[B, H, N, D]`` rows into packed ``[T, H, D]``."""
    batch, num_heads, max_seq_len, dim = x.shape
    lengths = seq_offsets[1:] - seq_offsets[:-1]
    positions = jnp.arange(max_seq_len, dtype=seq_offsets.dtype)
    valid = positions[None, :] < lengths[:, None]
    destinations = jnp.clip(
        seq_offsets[:-1, None] + positions[None, :],
        0,
        total_length - 1,
    )
    source = jnp.where(
        valid[:, :, None],
        x.transpose(0, 2, 1, 3).reshape(batch, max_seq_len, num_heads * dim),
        0,
    )
    packed = jnp.zeros((total_length, num_heads * dim), dtype=x.dtype)
    packed = packed.at[destinations.reshape(-1)].add(
        source.reshape(-1, num_heads * dim)
    )
    return packed.reshape(total_length, num_heads, dim)


def jagged_hstu_attention_reference(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    seq_offsets: jax.Array,
    num_targets: jax.Array,
    max_seq_len: int,
    scale: float,
    inv_n: float,
) -> jax.Array:
    """Materialized packed-input reference returning ``[B, H, N, Dv]``."""
    q_dense, lengths = _packed_to_dense(q, seq_offsets, max_seq_len)
    k_dense, _ = _packed_to_dense(k, seq_offsets, max_seq_len)
    v_dense, _ = _packed_to_dense(v, seq_offsets, max_seq_len)
    return hstu_attention_reference(
        q_dense,
        k_dense,
        v_dense,
        lengths,
        num_targets,
        scale,
        inv_n,
    )


def _jagged_fwd_kernel(
    seq_offsets_ref,
    num_targets_ref,
    attn_scale_ref,
    q_hbm_ref,
    k_hbm_ref,
    v_hbm_ref,
    out_hbm_ref,
    q_stage_ref,
    k_stage_ref,
    v_stage_ref,
    out_tile_ref,
    dma_sems,
    *,
    alpha: float,
    block_k: int,
    block_q: int,
) -> None:
    sequence = pl.program_id(0)
    start = seq_offsets_ref[sequence]
    end = seq_offsets_ref[sequence + 1]
    length = end - start
    num_targets = num_targets_ref[sequence]
    inv_n = attn_scale_ref[0]

    aligned_start = pl.multiple_of(start - start % _ALIGNMENT_ROWS, _ALIGNMENT_ROWS)
    prefix = start - aligned_start
    copy_rows = pl.multiple_of(
        pl.cdiv(prefix + length, _ALIGNMENT_ROWS) * _ALIGNMENT_ROWS,
        _ALIGNMENT_ROWS,
    )

    q_stage_ref[...] = jnp.zeros_like(q_stage_ref)
    k_stage_ref[...] = jnp.zeros_like(k_stage_ref)
    v_stage_ref[...] = jnp.zeros_like(v_stage_ref)

    @pl.when(length > 0)
    def _stage_inputs() -> None:
        q_copy = pltpu.make_async_copy(
            q_hbm_ref.at[pl.ds(aligned_start, copy_rows), :, :],
            q_stage_ref.at[pl.ds(0, copy_rows), :, :],
            dma_sems.at[0],
        )
        k_copy = pltpu.make_async_copy(
            k_hbm_ref.at[pl.ds(aligned_start, copy_rows), :, :],
            k_stage_ref.at[pl.ds(0, copy_rows), :, :],
            dma_sems.at[1],
        )
        v_copy = pltpu.make_async_copy(
            v_hbm_ref.at[pl.ds(aligned_start, copy_rows), :, :],
            v_stage_ref.at[pl.ds(0, copy_rows), :, :],
            dma_sems.at[2],
        )
        q_copy.start()
        k_copy.start()
        v_copy.start()
        q_copy.wait()
        k_copy.wait()
        v_copy.wait()

    num_query_blocks = out_hbm_ref.shape[2] // block_q
    num_kv_blocks = k_stage_ref.shape[0] // block_k
    dv = v_stage_ref.shape[-1]

    for query_block in range(num_query_blocks):
        query_start = query_block * block_q
        query_positions = query_start + jnp.arange(block_q)
        q_tile = jnp.swapaxes(
            q_stage_ref[pl.ds(prefix + query_start, block_q), :, :], 0, 1
        )
        max_id = length - num_targets
        query_ids = jnp.minimum(query_positions, max_id)

        def _accumulate(
            kv_block,
            output,
            max_id=max_id,
            q_tile=q_tile,
            query_ids=query_ids,
            query_positions=query_positions,
        ):
            kv_start = kv_block * block_k
            key_positions = kv_start + jnp.arange(block_k)
            key_ids = jnp.minimum(key_positions, max_id)
            k_tile = jnp.swapaxes(
                k_stage_ref[pl.ds(prefix + kv_start, block_k), :, :], 0, 1
            )
            v_tile = jnp.swapaxes(
                v_stage_ref[pl.ds(prefix + kv_start, block_k), :, :], 0, 1
            )
            scores = jax.lax.dot(
                q_tile,
                k_tile,
                dimension_numbers=(((2,), (2,)), ((0,), (0,))),
                preferred_element_type=jnp.float32,
            )
            weights = jax.nn.silu(scores * alpha) * inv_n
            causal = (query_ids[:, None] - key_ids[None, :]) > 0
            diagonal = query_positions[:, None] == key_positions[None, :]
            valid = (
                (causal | diagonal)
                & (query_positions[:, None] < length)
                & (key_positions[None, :] < length)
            )
            weights = jnp.where(valid[None, :, :], weights, 0).astype(v_tile.dtype)
            return output + jax.lax.dot(
                weights,
                v_tile,
                dimension_numbers=(((2,), (1,)), ((0,), (0,))),
                preferred_element_type=jnp.float32,
            )

        query_end = jnp.minimum(query_start + block_q, length)
        causal_upper = jnp.where(query_start < length, pl.cdiv(query_end, block_k), 0)
        output: Any = jax.lax.fori_loop(
            0,
            jnp.minimum(num_kv_blocks, causal_upper),
            _accumulate,
            jnp.zeros(
                (q_stage_ref.shape[1], block_q, dv),
                dtype=jnp.float32,
            ),
        )
        output = jnp.where(
            query_positions[None, :, None] < length,
            output,
            0,
        )
        out_tile_ref[...] = cast(Any, output).astype(out_tile_ref.dtype)
        pltpu.sync_copy(
            out_tile_ref,
            out_hbm_ref.at[sequence, :, pl.ds(query_start, block_q), :],
        )


def _native_jagged_hstu_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    seq_offsets: jax.Array,
    num_targets: jax.Array,
    max_seq_len: int,
    scale: float,
    inv_n: float,
    block_k: int,
    block_q: int,
) -> jax.Array:
    num_sequences = seq_offsets.shape[0] - 1
    num_heads = q.shape[1]
    dqk = q.shape[2]
    dv = v.shape[2]
    sequence_pad = (-max_seq_len) % math.lcm(block_q, block_k)
    padded_seq_len = max_seq_len + sequence_pad
    stage_rows = (
        pl.cdiv(padded_seq_len + _SAFE_TAIL_ROWS, _ALIGNMENT_ROWS) * _ALIGNMENT_ROWS
    )

    tail_q = jnp.zeros((_SAFE_TAIL_ROWS, num_heads, dqk), dtype=q.dtype)
    tail_v = jnp.zeros((_SAFE_TAIL_ROWS, num_heads, dv), dtype=v.dtype)
    q_physical = jnp.concatenate((q, tail_q), axis=0)
    k_physical = jnp.concatenate((k, tail_q), axis=0)
    v_physical = jnp.concatenate((v, tail_v), axis=0)
    offsets_i32 = seq_offsets.astype(jnp.int32)
    targets_i32 = num_targets.astype(jnp.int32)

    kernel = functools.partial(
        _jagged_fwd_kernel,
        alpha=scale,
        block_k=block_k,
        block_q=block_q,
    )
    output = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(
            (num_sequences, num_heads, padded_seq_len, dv), q.dtype
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            grid=(num_sequences,),
            scratch_shapes=(
                pltpu.VMEM((stage_rows, num_heads, dqk), q.dtype),
                pltpu.VMEM((stage_rows, num_heads, dqk), k.dtype),
                pltpu.VMEM((stage_rows, num_heads, dv), v.dtype),
                pltpu.VMEM((num_heads, block_q, dv), q.dtype),
                pltpu.SemaphoreType.DMA((3,)),
            ),
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        name="pallas_jagged_hstu_attention_fwd",
    )(
        offsets_i32,
        targets_i32,
        jnp.asarray(inv_n, dtype=jnp.float32).reshape((1,)),
        pltpu.with_memory_space_constraint(q_physical, pltpu.HBM),
        pltpu.with_memory_space_constraint(k_physical, pltpu.HBM),
        pltpu.with_memory_space_constraint(v_physical, pltpu.HBM),
    )
    return output[:, :, :max_seq_len, :]


def sample_sharded_jagged_hstu_attention(
    q: Any,
    k: Any,
    v: Any,
    seq_offsets: Any,
    num_targets: Any,
    attn_scale: Any,
    *,
    max_seq_len: int,
    alpha: float,
    world_size: int,
    mesh: Any,
    interpret: bool,
    run_kernel: Callable[..., Any] | None = None,
) -> Any:
    """Run packed-to-dense attention shard-locally on complete sequences."""
    total_length = int(q.shape[0])
    padded_batch_size = int(seq_offsets.shape[0]) - 1
    if total_length % world_size or padded_batch_size % world_size:
        raise ValueError("packed token and sequence counts must divide world_size")
    local_token_count = total_length // world_size
    seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
    _assert_batch_aligned(seq_lengths, world_size, local_token_count)
    local_padded_batch = padded_batch_size // world_size

    def _default_runner(q_, k_, v_, offsets_, targets_, scale_):
        return pallas_jagged_hstu_attention(
            q_,
            k_,
            v_,
            offsets_,
            targets_,
            max_seq_len=max_seq_len,
            scale=alpha,
            inv_n=scale_,
            interpret=interpret,
        )

    runner = _default_runner if run_kernel is None else run_kernel

    def _call(q_, k_, v_, lengths_, targets_, scale_):
        lengths_, targets_ = _drop_pad_rows(lengths_, targets_, padded_batch_size)
        local_offsets = jnp.concatenate(
            (jnp.zeros((1,), dtype=lengths_.dtype), jnp.cumsum(lengths_))
        )
        real_output = runner(q_, k_, v_, local_offsets, targets_, scale_)
        pad_rows = local_padded_batch - real_output.shape[0]
        return jnp.pad(
            real_output,
            ((0, pad_rows), (0, 0), (0, 0), (0, 0)),
        )

    mapped: Any = jax.shard_map(
        _call,
        mesh=mesh,
        in_specs=(
            jax.sharding.PartitionSpec(_AXIS, None, None),
            jax.sharding.PartitionSpec(_AXIS, None, None),
            jax.sharding.PartitionSpec(_AXIS, None, None),
            jax.sharding.PartitionSpec(_AXIS),
            jax.sharding.PartitionSpec(_AXIS),
            jax.sharding.PartitionSpec(),
        ),
        out_specs=jax.sharding.PartitionSpec(_AXIS, None, None, None),
        check_vma=False,
    )
    return mapped(q, k, v, seq_lengths, num_targets, attn_scale)


def pallas_jagged_hstu_attention_backward(
    grad_out: jax.Array,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    seq_offsets: jax.Array,
    num_targets: jax.Array,
    attn_scale: jax.Array,
    *,
    max_seq_len: int,
    alpha: float,
    interpret: bool,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Temporary packed-input backward implemented through dense Pallas."""
    from recir.ops.pallas.hstu_attention import pallas_hstu_attention_backward

    q_dense, lengths = _packed_to_dense(q, seq_offsets, max_seq_len)
    k_dense, _ = _packed_to_dense(k, seq_offsets, max_seq_len)
    v_dense, _ = _packed_to_dense(v, seq_offsets, max_seq_len)
    dq, dk, dv = pallas_hstu_attention_backward(
        grad_out * attn_scale,
        q_dense,
        k_dense,
        v_dense,
        lengths,
        num_targets,
        alpha,
        1.0,
        interpret=interpret,
    )
    return (
        _dense_to_packed(dq, seq_offsets, q.shape[0]),
        _dense_to_packed(dk, seq_offsets, k.shape[0]),
        _dense_to_packed(dv, seq_offsets, v.shape[0]),
    )


def sample_sharded_jagged_hstu_attention_backward(
    grad_out: Any,
    q: Any,
    k: Any,
    v: Any,
    seq_offsets: Any,
    num_targets: Any,
    attn_scale: Any,
    *,
    max_seq_len: int,
    alpha: float,
    world_size: int,
    mesh: Any,
    interpret: bool,
    run_kernel: Callable[..., tuple[Any, Any, Any]] | None = None,
) -> tuple[Any, Any, Any]:
    """Run the temporary dense Pallas backward inside each token shard."""
    total_length = int(q.shape[0])
    padded_batch_size = int(seq_offsets.shape[0]) - 1
    if total_length % world_size or padded_batch_size % world_size:
        raise ValueError("packed token and sequence counts must divide world_size")
    local_token_count = total_length // world_size
    seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
    _assert_batch_aligned(seq_lengths, world_size, local_token_count)

    def _default_runner(do_, q_, k_, v_, offsets_, targets_, scale_):
        real_batch = offsets_.shape[0] - 1
        return pallas_jagged_hstu_attention_backward(
            do_[:real_batch],
            q_,
            k_,
            v_,
            offsets_,
            targets_,
            scale_,
            max_seq_len=max_seq_len,
            alpha=alpha,
            interpret=interpret,
        )

    runner = _default_runner if run_kernel is None else run_kernel

    def _call(do_, q_, k_, v_, lengths_, targets_, scale_):
        lengths_, targets_ = _drop_pad_rows(lengths_, targets_, padded_batch_size)
        local_offsets = jnp.concatenate(
            (jnp.zeros((1,), dtype=lengths_.dtype), jnp.cumsum(lengths_))
        )
        return runner(do_, q_, k_, v_, local_offsets, targets_, scale_)

    packed_spec = jax.sharding.PartitionSpec(_AXIS, None, None)
    mapped: Any = jax.shard_map(
        _call,
        mesh=mesh,
        in_specs=(
            jax.sharding.PartitionSpec(_AXIS, None, None, None),
            packed_spec,
            packed_spec,
            packed_spec,
            jax.sharding.PartitionSpec(_AXIS),
            jax.sharding.PartitionSpec(_AXIS),
            jax.sharding.PartitionSpec(),
        ),
        out_specs=(packed_spec, packed_spec, packed_spec),
        check_vma=False,
    )
    return mapped(grad_out, q, k, v, seq_lengths, num_targets, attn_scale)


def pallas_jagged_hstu_attention(  # noqa: C901
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    seq_offsets: jax.Array,
    num_targets: jax.Array | None = None,
    *,
    max_seq_len: int,
    scale: float | None = None,
    inv_n: float | jax.Array | None = None,
    interpret: bool = False,
    block_k: int = _DEFAULT_BLOCK,
    block_q: int = _DEFAULT_BLOCK,
) -> jax.Array:
    """Compute HSTU attention from packed Q/K/V and return dense output.

    ``interpret=True`` uses the dense Pallas interpreter path because native
    runtime-sized DMA extents cannot be discharged by the CPU interpreter.
    """
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q, k, and v must have shape [T, H, D]")
    if q.shape[:2] != k.shape[:2] or q.shape[:2] != v.shape[:2]:
        raise ValueError("q, k, and v must share packed-token and head dimensions")
    if q.shape[2] != k.shape[2]:
        raise ValueError("q and k must share their feature dimension")
    if seq_offsets.ndim != 1 or seq_offsets.shape[0] < 2:
        raise ValueError("seq_offsets must have shape [B + 1]")
    num_sequences = seq_offsets.shape[0] - 1
    if num_targets is None:
        num_targets = jnp.zeros((num_sequences,), dtype=seq_offsets.dtype)
    if num_targets.shape != (num_sequences,):
        raise ValueError("num_targets must have shape [B]")
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    if block_q <= 0 or block_k <= 0:
        raise ValueError("block_q and block_k must be positive")
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[2])
    if inv_n is None:
        inv_n = 1.0 / max_seq_len
    resolved_inv_n: Any = inv_n
    if interpret:
        q_dense, lengths = _packed_to_dense(q, seq_offsets, max_seq_len)
        k_dense, _ = _packed_to_dense(k, seq_offsets, max_seq_len)
        v_dense, _ = _packed_to_dense(v, seq_offsets, max_seq_len)
        return pallas_hstu_attention(
            q_dense,
            k_dense,
            v_dense,
            lengths,
            num_targets,
            scale=scale,
            inv_n=resolved_inv_n,
            interpret=True,
            block_q=block_q,
            block_k=block_k,
        )

    if q.dtype != jnp.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("the native packed Pallas path currently requires bf16 Q/K/V")
    if block_q % 128 != 0 or block_k % 128 != 0:
        raise ValueError("native TPU block_q and block_k must be multiples of 128")
    return _native_jagged_hstu_attention(
        q,
        k,
        v,
        seq_offsets,
        num_targets,
        max_seq_len,
        scale,
        resolved_inv_n,
        block_k,
        block_q,
    )
