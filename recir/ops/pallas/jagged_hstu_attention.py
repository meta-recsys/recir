# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Packed-input HSTU SiLU attention for TPU via Pallas.

The native paths keep Q/K/V packed in HBM. Forward and dQ programs own query
groups, keep Q (and dO for dQ) resident in VMEM, and double-buffer aligned K/V
tiles shared by the group. The dK/dV program stages an aligned sequence
superset. Native kernels write disjoint sequence slices into a compact aligned
buffer, which is gathered once into the public packed ``[T, H, D]`` layout.

Runtime-sized DMA slices are supported by native Mosaic but not by ordinary
Pallas interpret discharge. ``interpret=True`` therefore exercises the same
packed API through the existing dense Pallas interpreter path; native DMA must
be validated on a real TPU.
"""

from __future__ import annotations

import functools
import itertools
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
_DEFAULT_BLOCK = 640
_DEFAULT_QUERY_TILES_PER_PROGRAM = 4
_DEFAULT_BACKWARD_QUERY_TILES_PER_PROGRAM = 2
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
    if x.shape[0] == 0:
        return (
            jnp.zeros(
                (seq_offsets.shape[0] - 1, x.shape[1], max_seq_len, x.shape[2]),
                dtype=x.dtype,
            ),
            lengths,
        )
    positions = jnp.arange(max_seq_len, dtype=seq_offsets.dtype)
    packed_rows = seq_offsets[:-1, None] + positions[None, :]
    valid = positions[None, :] < lengths[:, None]
    packed_rows = jnp.clip(packed_rows, 0, x.shape[0] - 1)
    dense = jnp.take(x, packed_rows, axis=0).transpose(0, 2, 1, 3)
    return jnp.where(valid[:, None, :, None], dense, 0), lengths


def _packed_row_sources(
    seq_offsets: jax.Array,
    total_length: int,
) -> tuple[jax.Array, jax.Array]:
    seq_offsets = seq_offsets.astype(jnp.int32)
    packed_rows = jnp.arange(total_length, dtype=seq_offsets.dtype)
    if total_length == 0:
        return packed_rows, packed_rows

    sequence_starts = seq_offsets[:-1]
    valid_starts = sequence_starts < total_length
    start_rows = jnp.minimum(sequence_starts, total_length - 1)
    boundary_counts = jnp.zeros((total_length,), dtype=seq_offsets.dtype)
    boundary_counts = boundary_counts.at[start_rows].add(
        valid_starts.astype(seq_offsets.dtype)
    )
    sequence_ids = jnp.cumsum(boundary_counts) - 1
    positions = packed_rows - seq_offsets[sequence_ids]
    return sequence_ids, positions


def _in_32bit_index_mode(fn: Any) -> Any:
    """Trace the pallas kernels with 32-bit integer defaults.

    Scoped rather than global, so callers that need ``JAX_ENABLE_X64=1``
    elsewhere keep it.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        previous = jax.config.values["jax_enable_x64"]
        if not previous:
            return fn(*args, **kwargs)
        jax.config.update("jax_enable_x64", False)
        try:
            return fn(*args, **kwargs)
        finally:
            jax.config.update("jax_enable_x64", previous)

    return wrapper


def _aligned_sequence_offsets(seq_offsets: jax.Array) -> jax.Array:
    seq_offsets = seq_offsets.astype(jnp.int32)
    lengths = seq_offsets[1:] - seq_offsets[:-1]
    aligned_lengths = (
        (lengths + _ALIGNMENT_ROWS - 1) // _ALIGNMENT_ROWS * _ALIGNMENT_ROWS
    )
    return jnp.concatenate(
        (
            jnp.zeros((1,), dtype=jnp.int32),
            jnp.cumsum(aligned_lengths, dtype=jnp.int32),
        )
    )


def _dense_values_to_packed(
    values: tuple[jax.Array, ...],
    seq_offsets: jax.Array,
    total_length: int,
) -> tuple[jax.Array, ...]:
    sequence_ids, positions = _packed_row_sources(seq_offsets, total_length)
    valid = jnp.arange(total_length, dtype=seq_offsets.dtype) < seq_offsets[-1]
    positions = jnp.minimum(positions, values[0].shape[2] - 1)
    feature_sizes = tuple(value.shape[-1] for value in values)
    combined = jnp.concatenate(
        tuple(value.transpose(0, 2, 1, 3) for value in values),
        axis=-1,
    )
    packed = combined[sequence_ids, positions]
    packed = jnp.where(valid[:, None, None], packed, 0)
    split_points = tuple(itertools.accumulate(feature_sizes[:-1]))
    return tuple(jnp.split(packed, split_points, axis=-1))


def _dense_gradients_to_packed(
    dq: jax.Array,
    dk: jax.Array,
    dv: jax.Array,
    seq_offsets: jax.Array,
    total_length: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Gather dense gradient rows with one shared packed-row map."""
    dq_packed, dk_packed, dv_packed = _dense_values_to_packed(
        (dq, dk, dv), seq_offsets, total_length
    )
    return dq_packed, dk_packed, dv_packed


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


def _cdiv(value: Any, divisor: int) -> Any:
    """``pl.cdiv`` with the divisor matched to ``value``'s dtype."""
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return pl.cdiv(value, divisor)
    return pl.cdiv(value, jnp.asarray(divisor, dtype))


def _aligned_sequence_window(start, length) -> tuple[Any, Any, Any]:
    aligned_start = pl.multiple_of(start - start % _ALIGNMENT_ROWS, _ALIGNMENT_ROWS)
    prefix = start - aligned_start
    copy_rows = pl.multiple_of(
        _cdiv(prefix + length, _ALIGNMENT_ROWS) * _ALIGNMENT_ROWS,
        _ALIGNMENT_ROWS,
    )
    return aligned_start, prefix, copy_rows


def _native_sequence_layout(
    max_seq_len: int,
    block_k: int,
    block_q: int,
    query_tiles_per_program: int = 1,
    *,
    pad_to_block_k: bool = True,
) -> tuple[int, int]:
    query_group_rows = block_q * query_tiles_per_program
    row_multiple = (
        math.lcm(query_group_rows, block_k) if pad_to_block_k else query_group_rows
    )
    sequence_pad = (-max_seq_len) % row_multiple
    padded_seq_len = max_seq_len + sequence_pad
    stage_rows = (
        pl.cdiv(padded_seq_len + _SAFE_TAIL_ROWS, _ALIGNMENT_ROWS) * _ALIGNMENT_ROWS
    )
    return padded_seq_len, stage_rows


_TAIL_ZERO_BLOCK = 2048


def _tail_zero_kernel(seq_off_ref, src, out_hbm_ref, *, rows: int, last: int) -> None:
    index = pl.program_id(0)

    def _emit(dst_ref) -> None:
        dst_ref[...] = jnp.zeros_like(dst_ref)

    def _window(_):
        tail_start = seq_off_ref[last]
        start_raw = jnp.maximum(index * _TAIL_ZERO_BLOCK, tail_start)
        stop = jnp.minimum((index + 1) * _TAIL_ZERO_BLOCK, rows)
        size = jnp.clip(stop - start_raw, 0, _TAIL_ZERO_BLOCK)
        start = jnp.clip(start_raw, 0, max(rows - 1, 0))
        return pl.ds(start, size), 0, 0

    pltpu.emit_pipeline(
        _emit,
        grid=(1,),
        out_specs=(
            pl.BlockSpec(
                (
                    pl.BoundedSlice(_TAIL_ZERO_BLOCK),
                    out_hbm_ref.shape[1],
                    out_hbm_ref.shape[2],
                ),
                _window,
                memory_space=pltpu.VMEM,
            ),
        ),
    )(out_hbm_ref)


def _zero_unvisited_tail(packed: jax.Array, seq_offsets: jax.Array) -> jax.Array:
    """Zero ``[seq_offsets[-1], L)``, the rows no grid program writes.

    Each block clamps its write to that region, so blocks before it transfer
    nothing. Aliasing a zeroed buffer to the output also works, but memsets the
    whole tensor.
    """
    rows = int(packed.shape[0])
    return pl.pallas_call(
        functools.partial(
            _tail_zero_kernel, rows=rows, last=int(seq_offsets.shape[0]) - 1
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(pl.cdiv(rows, _TAIL_ZERO_BLOCK),),
            in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
        ),
        out_shape=jax.ShapeDtypeStruct(packed.shape, packed.dtype),
        input_output_aliases={1: 0},
        name="pallas_jagged_tail_zero",
    )(seq_offsets.astype(jnp.int32), packed)


def _native_qkv_inputs(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    seq_offsets: jax.Array,
    num_targets: jax.Array,
    inv_n: float | jax.Array,
    tail_rows: int = _SAFE_TAIL_ROWS,
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    scalar_args = (
        seq_offsets.astype(jnp.int32),
        num_targets.astype(jnp.int32),
        jnp.asarray(inv_n, dtype=jnp.float32).reshape((1,)),
    )
    deficit = max(tail_rows - q.shape[0], 0)
    values = (
        tuple(
            jnp.concatenate(
                (x, jnp.zeros((deficit,) + x.shape[1:], dtype=x.dtype)), axis=0
            )
            for x in (q, k, v)
        )
        if deficit
        else (q, k, v)
    )
    hbm_args = tuple(
        pltpu.with_memory_space_constraint(value, pltpu.HBM) for value in values
    )
    return scalar_args, hbm_args


def _jagged_fwd_kernel(
    seq_offsets_ref,
    aligned_offsets_ref,
    num_targets_ref,
    attn_scale_ref,
    q_hbm_ref,
    k_hbm_ref,
    v_hbm_ref,
    out_hbm_ref,
    q_stage_ref,
    output_acc_ref,
    out_tile_ref,
    q_dma_sem,
    *,
    alpha: float,
    block_k: int,
    block_q: int,
    query_tiles_per_program: int,
) -> None:
    sequence = pl.program_id(0)
    query_group = pl.program_id(1)
    start = seq_offsets_ref[sequence]
    end = seq_offsets_ref[sequence + 1]
    length = end - start
    num_targets = num_targets_ref[sequence]
    inv_n = attn_scale_ref[0]

    aligned_start, prefix, _ = _aligned_sequence_window(start, length)
    query_group_rows = block_q * query_tiles_per_program
    query_group_start = query_group * query_group_rows
    q_stage_rows = query_group_rows + _ALIGNMENT_ROWS
    kv_stage_rows = block_k + _ALIGNMENT_ROWS

    @pl.when(query_group_start >= length)
    def _clear_query() -> None:
        q_stage_ref[...] = jnp.zeros_like(q_stage_ref)

    output_acc_ref[...] = jnp.zeros_like(output_acc_ref)

    q_read_cap = max(q_hbm_ref.shape[0] - q_stage_rows, 0)
    q_raw_start = aligned_start + query_group_start
    q_read_start = jnp.clip(q_raw_start, 0, q_read_cap)
    q_prefix = prefix + (q_raw_start - q_read_start)

    @pl.when(query_group_start < length)
    def _stage_query() -> None:
        copy = pltpu.make_async_copy(
            q_hbm_ref.at[pl.ds(q_read_start, q_stage_rows), :, :],
            q_stage_ref.at[pl.ds(0, q_stage_rows), :, :],
            q_dma_sem,
        )
        copy.start()
        copy.wait()

    max_id = length - num_targets

    def _accumulate(k_stage_ref, v_stage_ref) -> None:
        kv_block = pl.program_id(0)
        kv_start = kv_block * block_k
        key_positions = kv_start + jnp.arange(block_k, dtype=jnp.int32)
        key_ids = jnp.minimum(key_positions, max_id)
        k_tile = jnp.swapaxes(
            k_stage_ref[pl.ds(prefix, block_k), :, :],
            0,
            1,
        )
        v_tile = jnp.swapaxes(
            v_stage_ref[pl.ds(prefix, block_k), :, :],
            0,
            1,
        )

        for query_tile in range(query_tiles_per_program):
            query_offset = query_tile * block_q
            query_start = query_group_start + query_offset
            query_end = jnp.minimum(query_start + block_q, length)

            @pl.when((query_start < length) & (kv_start < query_end))
            def _accumulate_query_tile(
                query_offset=query_offset,
                query_start=query_start,
            ) -> None:
                query_positions = query_start + jnp.arange(block_q, dtype=jnp.int32)
                query_ids = jnp.minimum(query_positions, max_id)
                q_tile = jnp.swapaxes(
                    q_stage_ref[pl.ds(q_prefix + query_offset, block_q), :, :],
                    0,
                    1,
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
                weights = cast(
                    Any,
                    jnp.where(valid[None, :, :], weights, 0),
                ).astype(v_tile.dtype)
                output_update = jax.lax.dot(
                    weights,
                    v_tile,
                    dimension_numbers=(((2,), (1,)), ((0,), (0,))),
                    preferred_element_type=jnp.float32,
                )
                output_acc_ref[:, pl.ds(query_offset, block_q), :] = (
                    output_acc_ref[:, pl.ds(query_offset, block_q), :] + output_update
                )

    def _kv_window(kv_block):
        kv_aligned_start = pl.multiple_of(
            aligned_start + kv_block * block_k,
            _ALIGNMENT_ROWS,
        )
        return pl.ds(kv_aligned_start, kv_stage_rows), 0, 0

    query_group_end = jnp.minimum(query_group_start + query_group_rows, length)
    causal_upper = jnp.where(
        query_group_start < length,
        _cdiv(query_group_end, block_k),
        0,
    )
    kv_input_specs = (
        pl.BlockSpec(
            (
                pl.BoundedSlice(kv_stage_rows),
                q_hbm_ref.shape[1],
                q_hbm_ref.shape[2],
            ),
            _kv_window,
            memory_space=pltpu.VMEM,
            pipeline_mode=pl.Buffered(buffer_count=2),
        ),
        pl.BlockSpec(
            (
                pl.BoundedSlice(kv_stage_rows),
                v_hbm_ref.shape[1],
                v_hbm_ref.shape[2],
            ),
            _kv_window,
            memory_space=pltpu.VMEM,
            pipeline_mode=pl.Buffered(buffer_count=2),
        ),
    )
    pltpu.emit_pipeline(
        _accumulate,
        grid=(causal_upper,),
        in_specs=kv_input_specs,
        dimension_semantics=(pltpu.ARBITRARY,),
    )(
        k_hbm_ref,
        v_hbm_ref,
    )

    query_positions = query_group_start + jnp.arange(query_group_rows, dtype=jnp.int32)
    output = jnp.where(
        query_positions[None, :, None] < length,
        output_acc_ref[...],
        0,
    )
    out_tile_ref[...] = jnp.swapaxes(cast(Any, output).astype(out_tile_ref.dtype), 0, 1)

    def _emit_output(dst_ref) -> None:
        dst_ref[...] = out_tile_ref[...]

    def _output_window(_):
        rows = out_hbm_ref.shape[0]
        start = jnp.clip(
            seq_offsets_ref[sequence] + query_group_start, 0, max(rows - 1, 0)
        )
        size = jnp.clip(
            jnp.minimum(length - query_group_start, rows - start),
            0,
            query_group_rows,
        )
        return pl.ds(start, size), 0, 0

    pltpu.emit_pipeline(
        _emit_output,
        grid=(1,),
        out_specs=(
            pl.BlockSpec(
                (
                    pl.BoundedSlice(query_group_rows),
                    out_hbm_ref.shape[1],
                    out_hbm_ref.shape[2],
                ),
                _output_window,
                memory_space=pltpu.VMEM,
            ),
        ),
    )(out_hbm_ref)


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
    query_tiles_per_program: int,
) -> jax.Array:
    num_sequences = seq_offsets.shape[0] - 1
    num_heads = q.shape[1]
    dqk = q.shape[2]
    dv = v.shape[2]
    aligned_offsets = _aligned_sequence_offsets(seq_offsets)
    num_query_blocks = (max_seq_len + block_q - 1) // block_q
    query_tiles_per_program = min(query_tiles_per_program, num_query_blocks)
    query_group_rows = block_q * query_tiles_per_program
    padded_seq_len, _ = _native_sequence_layout(
        max_seq_len,
        block_k,
        block_q,
        query_tiles_per_program,
        pad_to_block_k=False,
    )
    tail_rows = max(query_group_rows, block_k) + _ALIGNMENT_ROWS
    scalar_args, hbm_args = _native_qkv_inputs(
        q,
        k,
        v,
        seq_offsets,
        num_targets,
        inv_n,
        tail_rows=tail_rows,
    )
    scalar_args = (scalar_args[0], aligned_offsets, *scalar_args[1:])
    num_query_groups = padded_seq_len // query_group_rows
    q_stage_rows = query_group_rows + _ALIGNMENT_ROWS

    kernel = functools.partial(
        _jagged_fwd_kernel,
        alpha=scale,
        block_k=block_k,
        block_q=block_q,
        query_tiles_per_program=query_tiles_per_program,
    )
    output = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((q.shape[0], num_heads, dv), q.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=4,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            grid=(num_sequences, num_query_groups),
            scratch_shapes=(
                pltpu.VMEM((2 * q_stage_rows, num_heads, dqk), q.dtype),
                pltpu.VMEM((num_heads, query_group_rows, dv), jnp.float32),
                pltpu.VMEM((query_group_rows, num_heads, dv), q.dtype),
                pltpu.SemaphoreType.DMA,
            ),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
        name="pallas_jagged_hstu_attention_fwd_pipelined",
    )(*scalar_args, *hbm_args)
    return _zero_unvisited_tail(output, seq_offsets)


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
    """Run packed attention shard-locally on complete sequences."""
    total_length = int(q.shape[0])
    padded_batch_size = int(seq_offsets.shape[0]) - 1
    if total_length % world_size or padded_batch_size % world_size:
        raise ValueError("packed token and sequence counts must divide world_size")
    local_token_count = total_length // world_size
    seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
    _assert_batch_aligned(seq_lengths, world_size, local_token_count)

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
        return runner(q_, k_, v_, local_offsets, targets_, scale_)

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
        out_specs=jax.sharding.PartitionSpec(_AXIS, None, None),
        check_vma=False,
    )
    return mapped(q, k, v, seq_lengths, num_targets, attn_scale)


def _silu_grad(x: jax.Array) -> jax.Array:
    sigmoid = jax.nn.sigmoid(x)
    return sigmoid * (1.0 + x * (1.0 - sigmoid))


def _jagged_bwd_dq_kernel(
    seq_offsets_ref,
    aligned_offsets_ref,
    num_targets_ref,
    attn_scale_ref,
    q_hbm_ref,
    k_hbm_ref,
    v_hbm_ref,
    do_hbm_ref,
    dq_init_hbm_ref,
    dq_hbm_ref,
    q_stage_ref,
    do_stage_ref,
    dq_acc_ref,
    dq_tile_ref,
    q_t_ref,
    do_t_ref,
    dma_sems,
    *,
    alpha: float,
    block_k: int,
    block_q: int,
    query_tiles_per_program: int,
) -> None:
    sequence = pl.program_id(0)
    query_group = pl.program_id(1)
    start = seq_offsets_ref[sequence]
    end = seq_offsets_ref[sequence + 1]
    length = end - start
    num_targets = num_targets_ref[sequence]
    inv_n = attn_scale_ref[0]

    aligned_start, prefix, _ = _aligned_sequence_window(start, length)
    query_group_rows = block_q * query_tiles_per_program
    query_group_start = query_group * query_group_rows
    q_stage_rows = query_group_rows + _ALIGNMENT_ROWS
    kv_stage_rows = block_k + _ALIGNMENT_ROWS

    @pl.when(query_group_start >= length)
    def _clear_inputs() -> None:
        q_stage_ref[...] = jnp.zeros_like(q_stage_ref)
        do_stage_ref[...] = jnp.zeros_like(do_stage_ref)

    dq_acc_ref[...] = jnp.zeros_like(dq_acc_ref)

    q_read_cap = max(q_hbm_ref.shape[0] - q_stage_rows, 0)
    q_raw_start = aligned_start + query_group_start
    q_read_start = jnp.clip(q_raw_start, 0, q_read_cap)
    q_prefix = prefix + (q_raw_start - q_read_start)

    @pl.when(query_group_start < length)
    def _stage_query() -> None:
        copies = (
            pltpu.make_async_copy(
                q_hbm_ref.at[pl.ds(q_read_start, q_stage_rows), :, :],
                q_stage_ref.at[pl.ds(0, q_stage_rows), :, :],
                dma_sems.at[0],
            ),
            pltpu.make_async_copy(
                do_hbm_ref.at[pl.ds(q_read_start, q_stage_rows), :, :],
                do_stage_ref.at[pl.ds(0, q_stage_rows), :, :],
                dma_sems.at[1],
            ),
        )
        for copy in copies:
            copy.start()
        for copy in copies:
            copy.wait()

    @pl.when(query_group_start < length)
    def _transpose_query() -> None:
        q_t_ref[...] = jnp.swapaxes(
            q_stage_ref[pl.ds(q_prefix, query_group_rows), :, :], 0, 1
        )
        do_t_ref[...] = (
            jnp.swapaxes(do_stage_ref[pl.ds(q_prefix, query_group_rows), :, :], 0, 1)
            * inv_n
        ).astype(do_t_ref.dtype)

    max_id = length - num_targets

    def _accumulate(k_stage_ref, v_stage_ref) -> None:
        kv_block = pl.program_id(0)
        kv_start = kv_block * block_k
        key_positions = kv_start + jnp.arange(block_k, dtype=jnp.int32)
        key_ids = jnp.minimum(key_positions, max_id)
        k_tile = (
            jnp.swapaxes(k_stage_ref[pl.ds(prefix, block_k), :, :], 0, 1) * alpha
        ).astype(k_stage_ref.dtype)
        v_tile = jnp.swapaxes(
            v_stage_ref[pl.ds(prefix, block_k), :, :],
            0,
            1,
        )

        for query_tile in range(query_tiles_per_program):
            query_offset = query_tile * block_q
            query_start = query_group_start + query_offset
            query_end = jnp.minimum(query_start + block_q, length)

            @pl.when((query_start < length) & (kv_start < query_end))
            def _accumulate_query_tile(
                query_offset=query_offset,
                query_start=query_start,
            ) -> None:
                query_positions = query_start + jnp.arange(block_q, dtype=jnp.int32)
                query_ids = jnp.minimum(query_positions, max_id)
                q_tile = q_t_ref[:, pl.ds(query_offset, block_q), :]
                do_tile = do_t_ref[:, pl.ds(query_offset, block_q), :]
                scores = jax.lax.dot(
                    q_tile,
                    k_tile,
                    dimension_numbers=(((2,), (2,)), ((0,), (0,))),
                    preferred_element_type=jnp.float32,
                )
                dp = jax.lax.dot(
                    do_tile,
                    v_tile,
                    dimension_numbers=(((2,), (2,)), ((0,), (0,))),
                    preferred_element_type=jnp.float32,
                )
                causal = (query_ids[:, None] - key_ids[None, :]) > 0
                diagonal = query_positions[:, None] == key_positions[None, :]
                valid = (
                    (causal | diagonal)
                    & (query_positions[:, None] < length)
                    & (key_positions[None, :] < length)
                )
                ds = cast(
                    Any,
                    jnp.where(valid[None, :, :], dp * _silu_grad(scores), 0),
                )
                dq_update = jax.lax.dot(
                    ds.astype(k_tile.dtype),
                    k_tile,
                    dimension_numbers=(((2,), (1,)), ((0,), (0,))),
                    preferred_element_type=jnp.float32,
                )
                dq_acc_ref[:, pl.ds(query_offset, block_q), :] = (
                    dq_acc_ref[:, pl.ds(query_offset, block_q), :] + dq_update
                )

    def _kv_window(kv_block):
        kv_aligned_start = pl.multiple_of(
            aligned_start + kv_block * block_k,
            _ALIGNMENT_ROWS,
        )
        return pl.ds(kv_aligned_start, kv_stage_rows), 0, 0

    query_group_end = jnp.minimum(query_group_start + query_group_rows, length)
    causal_upper = jnp.where(
        query_group_start < length,
        _cdiv(query_group_end, block_k),
        0,
    )
    kv_input_specs = (
        pl.BlockSpec(
            (
                pl.BoundedSlice(kv_stage_rows),
                q_hbm_ref.shape[1],
                q_hbm_ref.shape[2],
            ),
            _kv_window,
            memory_space=pltpu.VMEM,
            pipeline_mode=pl.Buffered(buffer_count=2),
        ),
        pl.BlockSpec(
            (
                pl.BoundedSlice(kv_stage_rows),
                v_hbm_ref.shape[1],
                v_hbm_ref.shape[2],
            ),
            _kv_window,
            memory_space=pltpu.VMEM,
            pipeline_mode=pl.Buffered(buffer_count=2),
        ),
    )
    pltpu.emit_pipeline(
        _accumulate,
        grid=(causal_upper,),
        in_specs=kv_input_specs,
        dimension_semantics=(pltpu.ARBITRARY,),
    )(
        k_hbm_ref,
        v_hbm_ref,
    )

    query_positions = query_group_start + jnp.arange(query_group_rows, dtype=jnp.int32)
    dq = jnp.where(
        query_positions[None, :, None] < length,
        dq_acc_ref[...],
        0,
    )
    dq_tile_ref[...] = jnp.swapaxes(cast(Any, dq).astype(dq_tile_ref.dtype), 0, 1)

    def _emit_dq(dst_ref) -> None:
        dst_ref[...] = dq_tile_ref[...]

    def _dq_window(_):
        rows = dq_hbm_ref.shape[0]
        start = jnp.clip(
            seq_offsets_ref[sequence] + query_group_start, 0, max(rows - 1, 0)
        )
        size = jnp.clip(
            jnp.minimum(length - query_group_start, rows - start),
            0,
            query_group_rows,
        )
        return pl.ds(start, size), 0, 0

    pltpu.emit_pipeline(
        _emit_dq,
        grid=(1,),
        out_specs=(
            pl.BlockSpec(
                (
                    pl.BoundedSlice(query_group_rows),
                    dq_hbm_ref.shape[1],
                    dq_hbm_ref.shape[2],
                ),
                _dq_window,
                memory_space=pltpu.VMEM,
            ),
        ),
    )(dq_hbm_ref)


def _jagged_bwd_dkv_kernel(
    seq_offsets_ref,
    aligned_offsets_ref,
    num_targets_ref,
    attn_scale_ref,
    q_hbm_ref,
    k_hbm_ref,
    v_hbm_ref,
    do_hbm_ref,
    dk_init_hbm_ref,
    dv_init_hbm_ref,
    dk_hbm_ref,
    dv_hbm_ref,
    k_stage_ref,
    v_stage_ref,
    dk_acc_ref,
    dv_acc_ref,
    dk_tile_ref,
    dv_tile_ref,
    dma_sems,
    *,
    alpha: float,
    block_k: int,
    block_q: int,
) -> None:
    sequence = pl.program_id(0)
    kv_block = pl.program_id(1)
    start = seq_offsets_ref[sequence]
    end = seq_offsets_ref[sequence + 1]
    length = end - start
    num_targets = num_targets_ref[sequence]
    inv_n = attn_scale_ref[0]

    aligned_start, prefix, _ = _aligned_sequence_window(start, length)
    kv_start = kv_block * block_k
    kv_stage_rows = block_k + _ALIGNMENT_ROWS
    q_stage_rows = block_q + _ALIGNMENT_ROWS
    dk_acc_ref[...] = jnp.zeros_like(dk_acc_ref)
    dv_acc_ref[...] = jnp.zeros_like(dv_acc_ref)

    @pl.when(kv_start >= length)
    def _clear_kv() -> None:
        k_stage_ref[...] = jnp.zeros_like(k_stage_ref)
        v_stage_ref[...] = jnp.zeros_like(v_stage_ref)

    kv_read_cap = max(k_hbm_ref.shape[0] - kv_stage_rows, 0)
    kv_raw_start = aligned_start + kv_start
    kv_read_start = jnp.clip(kv_raw_start, 0, kv_read_cap)
    kv_prefix = prefix + (kv_raw_start - kv_read_start)

    @pl.when(kv_start < length)
    def _stage_kv() -> None:
        copies = (
            pltpu.make_async_copy(
                k_hbm_ref.at[pl.ds(kv_read_start, kv_stage_rows), :, :],
                k_stage_ref.at[pl.ds(0, kv_stage_rows), :, :],
                dma_sems.at[0],
            ),
            pltpu.make_async_copy(
                v_hbm_ref.at[pl.ds(kv_read_start, kv_stage_rows), :, :],
                v_stage_ref.at[pl.ds(0, kv_stage_rows), :, :],
                dma_sems.at[1],
            ),
        )
        for copy in copies:
            copy.start()
        for copy in copies:
            copy.wait()

    max_id = length - num_targets
    key_positions = kv_start + jnp.arange(block_k, dtype=jnp.int32)
    key_ids = jnp.minimum(key_positions, max_id)
    k_tile = jnp.swapaxes(
        k_stage_ref[pl.ds(kv_prefix, block_k), :, :],
        0,
        1,
    )
    v_tile = jnp.swapaxes(
        v_stage_ref[pl.ds(kv_prefix, block_k), :, :],
        0,
        1,
    )
    first_query_block = kv_start // block_q

    def _accumulate(q_stage_ref, do_stage_ref) -> None:
        query_block = first_query_block + pl.program_id(0)
        query_start = query_block * block_q
        query_positions = query_start + jnp.arange(block_q, dtype=jnp.int32)
        query_ids = jnp.minimum(query_positions, max_id)
        q_tile = jnp.swapaxes(
            q_stage_ref[pl.ds(prefix, block_q), :, :],
            0,
            1,
        )
        do_tile = jnp.swapaxes(
            do_stage_ref[pl.ds(prefix, block_q), :, :],
            0,
            1,
        )
        scores = jax.lax.dot(
            q_tile,
            k_tile,
            dimension_numbers=(((2,), (2,)), ((0,), (0,))),
            preferred_element_type=jnp.float32,
        )
        scaled_scores = scores * alpha
        causal = (query_ids[:, None] - key_ids[None, :]) > 0
        diagonal = query_positions[:, None] == key_positions[None, :]
        valid = (
            (causal | diagonal)
            & (query_positions[:, None] < length)
            & (key_positions[None, :] < length)
        )
        weights = cast(
            Any,
            jnp.where(
                valid[None, :, :],
                jax.nn.silu(scaled_scores) * inv_n,
                0,
            ),
        )
        dv_update = jax.lax.dot(
            weights.astype(do_tile.dtype),
            do_tile,
            dimension_numbers=(((1,), (1,)), ((0,), (0,))),
            preferred_element_type=jnp.float32,
        )
        dp = jax.lax.dot(
            do_tile,
            v_tile,
            dimension_numbers=(((2,), (2,)), ((0,), (0,))),
            preferred_element_type=jnp.float32,
        )
        ds = cast(
            Any,
            jnp.where(
                valid[None, :, :],
                dp * _silu_grad(scaled_scores) * inv_n,
                0,
            ),
        )
        dk_update = jax.lax.dot(
            ds.astype(q_tile.dtype),
            q_tile,
            dimension_numbers=(((1,), (1,)), ((0,), (0,))),
            preferred_element_type=jnp.float32,
        )
        dk_acc_ref[...] = dk_acc_ref[...] + alpha * dk_update
        dv_acc_ref[...] = dv_acc_ref[...] + dv_update

    def _query_window(query_block):
        query_start = (first_query_block + query_block) * block_q
        query_aligned_start = pl.multiple_of(
            aligned_start + query_start,
            _ALIGNMENT_ROWS,
        )
        return pl.ds(query_aligned_start, q_stage_rows), 0, 0

    num_query_blocks = cast(
        Any,
        jnp.where(
            kv_start < length,
            _cdiv(length, block_q) - first_query_block,
            0,
        ),
    )
    query_input_specs = (
        pl.BlockSpec(
            (
                pl.BoundedSlice(q_stage_rows),
                q_hbm_ref.shape[1],
                q_hbm_ref.shape[2],
            ),
            _query_window,
            memory_space=pltpu.VMEM,
            pipeline_mode=pl.Buffered(buffer_count=2),
        ),
        pl.BlockSpec(
            (
                pl.BoundedSlice(q_stage_rows),
                do_hbm_ref.shape[1],
                do_hbm_ref.shape[2],
            ),
            _query_window,
            memory_space=pltpu.VMEM,
            pipeline_mode=pl.Buffered(buffer_count=2),
        ),
    )
    pltpu.emit_pipeline(
        _accumulate,
        grid=(num_query_blocks,),
        in_specs=query_input_specs,
        dimension_semantics=(pltpu.ARBITRARY,),
    )(
        q_hbm_ref,
        do_hbm_ref,
    )

    valid_key = key_positions[None, :, None] < length
    dk = jnp.where(valid_key, dk_acc_ref[...], 0)
    dv = jnp.where(valid_key, dv_acc_ref[...], 0)
    dk_tile_ref[...] = jnp.swapaxes(cast(Any, dk).astype(dk_tile_ref.dtype), 0, 1)
    dv_tile_ref[...] = jnp.swapaxes(cast(Any, dv).astype(dv_tile_ref.dtype), 0, 1)

    def _emit_dkv(dk_dst_ref, dv_dst_ref) -> None:
        dk_dst_ref[...] = dk_tile_ref[...]
        dv_dst_ref[...] = dv_tile_ref[...]

    def _kv_out_window(_):
        rows = dk_hbm_ref.shape[0]
        start = jnp.clip(seq_offsets_ref[sequence] + kv_start, 0, max(rows - 1, 0))
        size = jnp.clip(jnp.minimum(length - kv_start, rows - start), 0, block_k)
        return pl.ds(start, size), 0, 0

    pltpu.emit_pipeline(
        _emit_dkv,
        grid=(1,),
        out_specs=(
            pl.BlockSpec(
                (pl.BoundedSlice(block_k), dk_hbm_ref.shape[1], dk_hbm_ref.shape[2]),
                _kv_out_window,
                memory_space=pltpu.VMEM,
            ),
            pl.BlockSpec(
                (pl.BoundedSlice(block_k), dv_hbm_ref.shape[1], dv_hbm_ref.shape[2]),
                _kv_out_window,
                memory_space=pltpu.VMEM,
            ),
        ),
    )(dk_hbm_ref, dv_hbm_ref)


def _native_jagged_hstu_attention_backward(
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
    block_k: int,
    block_q: int,
    query_tiles_per_program: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    num_sequences = seq_offsets.shape[0] - 1
    num_heads = q.shape[1]
    dqk = q.shape[2]
    dv_dim = v.shape[2]
    aligned_offsets = _aligned_sequence_offsets(seq_offsets)

    num_query_blocks = (max_seq_len + block_q - 1) // block_q
    query_tiles_per_program = min(query_tiles_per_program, num_query_blocks)
    query_group_rows = block_q * query_tiles_per_program
    dq_padded_seq_len, _ = _native_sequence_layout(
        max_seq_len,
        block_k,
        block_q,
        query_tiles_per_program,
        pad_to_block_k=False,
    )
    dkv_padded_seq_len, _ = _native_sequence_layout(
        max_seq_len,
        block_k,
        block_q,
    )
    num_kv_blocks = dkv_padded_seq_len // block_k
    tail_rows = max(query_group_rows, block_k) + _ALIGNMENT_ROWS
    scalar_args, qkv_hbm_args = _native_qkv_inputs(
        q,
        k,
        v,
        seq_offsets,
        num_targets,
        attn_scale,
        tail_rows=tail_rows,
    )
    scalar_args = (scalar_args[0], aligned_offsets, *scalar_args[1:])
    grad_deficit = max(tail_rows - grad_out.shape[0], 0)
    packed_grad_out = (
        jnp.concatenate(
            (
                grad_out,
                jnp.zeros((grad_deficit,) + grad_out.shape[1:], dtype=grad_out.dtype),
            ),
            axis=0,
        )
        if grad_deficit
        else grad_out
    )
    backward_hbm_args = (
        *qkv_hbm_args,
        pltpu.with_memory_space_constraint(
            packed_grad_out,
            pltpu.HBM,
        ),
    )
    num_query_groups = dq_padded_seq_len // query_group_rows
    q_stage_rows = query_group_rows + _ALIGNMENT_ROWS
    dq_grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=4,
        in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)] * 5,
        out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
        grid=(num_sequences, num_query_groups),
        scratch_shapes=(
            pltpu.VMEM((2 * q_stage_rows, num_heads, dqk), q.dtype),
            pltpu.VMEM((2 * q_stage_rows, num_heads, dv_dim), grad_out.dtype),
            pltpu.VMEM((num_heads, query_group_rows, dqk), jnp.float32),
            pltpu.VMEM((query_group_rows, num_heads, dqk), q.dtype),
            pltpu.VMEM((num_heads, query_group_rows, dqk), q.dtype),
            pltpu.VMEM((num_heads, query_group_rows, dv_dim), grad_out.dtype),
            pltpu.SemaphoreType.DMA((2,)),
        ),
    )
    dq = pl.pallas_call(
        functools.partial(
            _jagged_bwd_dq_kernel,
            alpha=alpha,
            block_k=block_k,
            block_q=block_q,
            query_tiles_per_program=query_tiles_per_program,
        ),
        out_shape=jax.ShapeDtypeStruct((q.shape[0], num_heads, dqk), q.dtype),
        grid_spec=dq_grid_spec,
        input_output_aliases={8: 0},
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
        name="pallas_jagged_hstu_attention_bwd_dq",
    )(
        *scalar_args,
        *backward_hbm_args,
        pltpu.with_memory_space_constraint(
            jnp.zeros((q.shape[0], num_heads, dqk), q.dtype),
            pltpu.HBM,
        ),
    )

    kv_stage_rows = block_k + _ALIGNMENT_ROWS
    dkv_grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=4,
        in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)] * 6,
        out_specs=[
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ],
        grid=(num_sequences, num_kv_blocks),
        scratch_shapes=(
            pltpu.VMEM((2 * kv_stage_rows, num_heads, dqk), k.dtype),
            pltpu.VMEM((2 * kv_stage_rows, num_heads, dv_dim), v.dtype),
            pltpu.VMEM((num_heads, block_k, dqk), jnp.float32),
            pltpu.VMEM((num_heads, block_k, dv_dim), jnp.float32),
            pltpu.VMEM((block_k, num_heads, dqk), k.dtype),
            pltpu.VMEM((block_k, num_heads, dv_dim), v.dtype),
            pltpu.SemaphoreType.DMA((2,)),
        ),
    )
    dk, dv = pl.pallas_call(
        functools.partial(
            _jagged_bwd_dkv_kernel,
            alpha=alpha,
            block_k=block_k,
            block_q=block_q,
        ),
        out_shape=(
            jax.ShapeDtypeStruct((q.shape[0], num_heads, dqk), k.dtype),
            jax.ShapeDtypeStruct((q.shape[0], num_heads, dv_dim), v.dtype),
        ),
        grid_spec=dkv_grid_spec,
        input_output_aliases={8: 0, 9: 1},
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
        name="pallas_jagged_hstu_attention_bwd_dkv",
    )(
        *scalar_args,
        *backward_hbm_args,
        pltpu.with_memory_space_constraint(
            jnp.zeros((q.shape[0], num_heads, dqk), k.dtype),
            pltpu.HBM,
        ),
        pltpu.with_memory_space_constraint(
            jnp.zeros((q.shape[0], num_heads, dv_dim), v.dtype),
            pltpu.HBM,
        ),
    )
    return dq, dk, dv


@_in_32bit_index_mode
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
    block_k: int = _DEFAULT_BLOCK,
    block_q: int = _DEFAULT_BLOCK,
    query_tiles_per_program: int = _DEFAULT_BACKWARD_QUERY_TILES_PER_PROGRAM,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute HSTU gradients while reading Q/K/V from packed storage."""
    from recir.ops.pallas.hstu_attention import pallas_hstu_attention_backward

    _validate_jagged_hstu_attention_inputs(
        q,
        k,
        v,
        seq_offsets,
        num_targets,
        max_seq_len=max_seq_len,
        block_k=block_k,
        block_q=block_q,
        query_tiles_per_program=query_tiles_per_program,
    )
    if grad_out.shape != v.shape:
        raise ValueError("grad_out must have packed shape [T, H, Dv]")
    if grad_out.dtype != v.dtype:
        raise ValueError("grad_out and v must have the same dtype")

    if interpret:
        q_dense, lengths = _packed_to_dense(q, seq_offsets, max_seq_len)
        k_dense, _ = _packed_to_dense(k, seq_offsets, max_seq_len)
        v_dense, _ = _packed_to_dense(v, seq_offsets, max_seq_len)
        dq, dk, dv = pallas_hstu_attention_backward(
            _packed_to_dense(grad_out, seq_offsets, max_seq_len)[0] * attn_scale,
            q_dense,
            k_dense,
            v_dense,
            lengths,
            num_targets,
            alpha,
            1.0,
            interpret=True,
            block_k=block_k,
            block_q=block_q,
        )
        return _dense_gradients_to_packed(dq, dk, dv, seq_offsets, q.shape[0])
    else:
        if q.dtype != jnp.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
            raise ValueError(
                "the native packed Pallas backward currently requires bf16 Q/K/V"
            )
        if block_q % 128 != 0 or block_k % 128 != 0:
            raise ValueError("native TPU block_q and block_k must be multiples of 128")
        return _native_jagged_hstu_attention_backward(
            grad_out,
            q,
            k,
            v,
            seq_offsets,
            num_targets,
            attn_scale,
            max_seq_len=max_seq_len,
            alpha=alpha,
            block_k=block_k,
            block_q=block_q,
            query_tiles_per_program=query_tiles_per_program,
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
    """Run packed-input Pallas backward inside each complete-sequence shard."""
    total_length = int(q.shape[0])
    padded_batch_size = int(seq_offsets.shape[0]) - 1
    if total_length % world_size or padded_batch_size % world_size:
        raise ValueError("packed token and sequence counts must divide world_size")
    local_token_count = total_length // world_size
    seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
    _assert_batch_aligned(seq_lengths, world_size, local_token_count)

    def _default_runner(do_, q_, k_, v_, offsets_, targets_, scale_):
        return pallas_jagged_hstu_attention_backward(
            do_,
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
            packed_spec,
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


def _validate_jagged_hstu_attention_inputs(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    seq_offsets: jax.Array,
    num_targets: jax.Array | None,
    *,
    max_seq_len: int,
    block_k: int,
    block_q: int,
    query_tiles_per_program: int,
) -> int:
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q, k, and v must have shape [T, H, D]")
    if q.shape[:2] != k.shape[:2] or q.shape[:2] != v.shape[:2]:
        raise ValueError("q, k, and v must share packed-token and head dimensions")
    if q.shape[2] != k.shape[2]:
        raise ValueError("q and k must share their feature dimension")
    if seq_offsets.ndim != 1 or seq_offsets.shape[0] < 2:
        raise ValueError("seq_offsets must have shape [B + 1]")
    num_sequences = seq_offsets.shape[0] - 1
    if num_targets is not None and num_targets.shape != (num_sequences,):
        raise ValueError("num_targets must have shape [B]")
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    if min(block_q, block_k, query_tiles_per_program) <= 0:
        raise ValueError(
            "block_q, block_k, and query_tiles_per_program must be positive"
        )
    return num_sequences


@_in_32bit_index_mode
def pallas_jagged_hstu_attention(
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
    query_tiles_per_program: int = _DEFAULT_QUERY_TILES_PER_PROGRAM,
) -> jax.Array:
    """Compute HSTU attention from packed Q/K/V and return packed output.

    ``interpret=True`` uses the dense Pallas interpreter path because native
    runtime-sized DMA extents cannot be discharged by the CPU interpreter.
    """
    num_sequences = _validate_jagged_hstu_attention_inputs(
        q,
        k,
        v,
        seq_offsets,
        num_targets,
        max_seq_len=max_seq_len,
        block_k=block_k,
        block_q=block_q,
        query_tiles_per_program=query_tiles_per_program,
    )
    if num_targets is None:
        num_targets = jnp.zeros((num_sequences,), dtype=seq_offsets.dtype)
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[2])
    if inv_n is None:
        inv_n = 1.0 / max_seq_len
    resolved_inv_n: Any = inv_n
    if interpret:
        q_dense, lengths = _packed_to_dense(q, seq_offsets, max_seq_len)
        k_dense, _ = _packed_to_dense(k, seq_offsets, max_seq_len)
        v_dense, _ = _packed_to_dense(v, seq_offsets, max_seq_len)
        dense_output = pallas_hstu_attention(
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
        return _dense_values_to_packed((dense_output,), seq_offsets, q.shape[0])[0]

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
        query_tiles_per_program,
    )
