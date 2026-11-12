# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Row-wise Adagrad on SparseCore, applied in place to a table's rows.

For gradient rows that may repeat, the kernel sums each row's gradients and
applies FBGEMM ``EXACT_ROWWISE_ADAGRAD`` (weight decay 0):

    momentum[r] += mean(g[r] ** 2)
    w[r] -= lr * g[r] / (sqrt(momentum[r]) + eps)

It reads and writes the table in whatever layout it arrives in, so it can run
directly on JAX TPU Embedding's SparseCore-layout tables.

For tables trained as FP16, the updated weights are stochastically rounded to
FP16 values (kept in FP32 storage), as `TensorLRRowWiseAdagrad` does for FP16
parameters and TBE does on GPU.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu, tpu_sc as plsc

_LANES = 16


def _rsqrt(a: jax.Array) -> jax.Array:
    """1/sqrt(a) for a >= 0. The SparseCore vector subcore has no sqrt: bit-trick
    seed plus three Newton steps (about f32 precision). a == 0 gives a large finite
    value, so callers form sqrt(a) as a * _rsqrt(a), which is 0 there."""
    i = jax.lax.bitcast_convert_type(a, jnp.int32)
    i = jnp.int32(0x5F3759DF) - jax.lax.shift_right_logical(i, jnp.int32(1))
    y = jax.lax.bitcast_convert_type(i, jnp.float32)
    half_a = a * jnp.float32(0.5)
    for _ in range(3):
        y = y * (jnp.float32(1.5) - half_a * y * y)
    return y


def _hash32(x: jax.Array) -> jax.Array:
    """murmur3's 32-bit finalizer on int32 lanes (wrapping arithmetic)."""
    x = x ^ jax.lax.shift_right_logical(x, jnp.int32(16))
    x = x * jnp.int32(-2048144789)  # 0x85EBCA6B
    x = x ^ jax.lax.shift_right_logical(x, jnp.int32(13))
    x = x * jnp.int32(-1028477387)  # 0xC2B2AE35
    return x ^ jax.lax.shift_right_logical(x, jnp.int32(16))


def _stochastic_round_fp16(x: jax.Array, bits: jax.Array) -> jax.Array:
    """FP32 lanes rounded to one of the two adjacent FP16 values, up with probability
    equal to the distance from the lower one over their gap; the result stays FP32.

    ``bits`` are random int32 lanes. Beyond FP16's largest finite value the result is
    the ordinary FP16 conversion; NaN and infinities pass through.
    """
    i = jax.lax.bitcast_convert_type(x, jnp.int32)
    a = i & jnp.int32(0x7FFFFFFF)
    # FP16 normals keep 10 of FP32's 23 mantissa bits: add 13 random bits, truncate.
    normal = jax.lax.bitcast_convert_type(
        (i + jax.lax.shift_right_logical(bits, jnp.int32(19))) & jnp.int32(-0x2000),
        jnp.float32,
    )
    # FP16 subnormals (|x| < 2**-14) are multiples of 2**-24.
    t = jax.lax.bitcast_convert_type(a, jnp.float32) * jnp.float32(2**24)
    whole = t.astype(jnp.int32)
    u = jax.lax.shift_right_logical(bits, jnp.int32(8)).astype(
        jnp.float32
    ) * jnp.float32(2**-24)
    up = (u < t - whole.astype(jnp.float32)).astype(jnp.int32)
    sub = (whole + up).astype(jnp.float32) * jnp.float32(2**-24)
    y = jnp.where(a >= jnp.int32(0x38800000), normal, jnp.where(i < 0, -sub, sub))
    # Above 65504: 65520 and beyond convert to infinity, the rest to 65504.
    big = jnp.where(
        a >= jnp.int32(0x477FF000), jnp.float32(jnp.inf), jnp.float32(65504)
    )
    y = jnp.where(a > jnp.int32(0x477FE000), jnp.where(i < 0, -big, big), y)
    return jnp.where(a >= jnp.int32(0x7F800000), x, y)


def fp16_rounding_key(counter: jax.Array, stream: int, device: jax.Array) -> jax.Array:
    """The ``rounding_key`` of one table, step and device. Within a step, every
    (``stream`` < 128, ``device`` < 2**24) pair gets a distinct key."""
    return _hash32(
        _hash32(counter) ^ jax.lax.shift_left(jnp.int32(stream), jnp.int32(24)) ^ device
    )


def rowwise_adagrad_update(
    rows: jax.Array,
    grads: jax.Array,
    emb_tables: jax.Array,
    momentum: jax.Array,
    lr: jax.Array,
    eps: float,
    rounding_key: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Row-wise Adagrad for gradient rows that may repeat; returns (emb, momentum).

    Args:
        rows: [N] int32 local table row of each gradient row; -1 entries are skipped.
        grads: [N, D] f32; each row's gradients are summed before the update.
        emb_tables: [V, D] f32; momentum: [V] f32; lr: f32 scalar.
        rounding_key: int32 scalar, unique per step and device; when given, updated
            weights are stochastically rounded to FP16 values (FP16-trained tables).
    """
    # Traced without x64: under JAX_ENABLE_X64 Python ints become int64
    # constants, which Mosaic will not compare or combine with the kernel's int32s.
    with jax.enable_x64(False):
        unique_rows, positions, slots, run_offsets = _row_runs(rows)
        return _batched_rowwise_adagrad_update(
            unique_rows,
            positions,
            slots,
            run_offsets,
            grads,
            emb_tables,
            momentum,
            jnp.asarray(lr, jnp.float32),
            jnp.zeros((), jnp.int32)
            if rounding_key is None
            else rounding_key.astype(jnp.int32),
            eps,
            rounding_key is not None,
        )


def _row_runs(rows: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Sorted runs of equal rows: (row of each run or -1, position of each entry in run
    order, run index % 16 per entry, first entry of each run). -1 rows form no run."""
    n = rows.shape[0]
    key = jnp.where(rows >= 0, rows, jnp.iinfo(jnp.int32).max)
    sorted_rows, positions = jax.lax.sort_key_val(key, jnp.arange(n, dtype=jnp.int32))
    num_valid = jnp.sum(rows >= 0, dtype=jnp.int32)
    starts = (jnp.arange(n, dtype=jnp.int32) < num_valid) & jnp.concatenate(
        (jnp.ones((1,), jnp.bool_), sorted_rows[1:] != sorted_rows[:-1])
    )
    run = jnp.cumsum(starts, dtype=jnp.int32) - 1
    run_offsets = jnp.nonzero(starts, size=n, fill_value=n)[0].astype(jnp.int32)
    unique_rows = jnp.where(
        run_offsets < n, sorted_rows[jnp.minimum(run_offsets, n - 1)], -1
    )
    run_offsets = jnp.minimum(
        jnp.concatenate((run_offsets, jnp.array([n], jnp.int32))), num_valid
    )
    return unique_rows, positions, run % 16, run_offsets


@functools.partial(jax.jit, static_argnames=["eps", "round_fp16"])
def _batched_rowwise_adagrad_update(  # noqa: C901
    unique_rows: jax.Array,
    positions: jax.Array,
    slots: jax.Array,
    run_offsets: jax.Array,
    grad_out: jax.Array,
    emb_tables: jax.Array,
    momentum: jax.Array,
    lr: jax.Array,
    rounding_key: jax.Array,
    eps: float,
    round_fp16: bool,
) -> tuple[jax.Array, jax.Array]:
    """Row-wise Adagrad on SparseCore over runs of equal rows, 16 runs per subcore iteration.

    Args:
        unique_rows: [R] table row of each run (runs sorted by row), -1 past the last run.
        positions: [N] grad_out row of each position, in run order.
        slots: [N] run index % 16 of each position, i.e. its row within a 16-run block.
        run_offsets: [R + 1] first position of each run; entries past the last run equal N.

    Rows of a block are gathered and scattered with one indexed copy each; SparseCore has
    no atomic HBM writes, so each row is owned by exactly one block.
    """
    L = _LANES
    emb_dim = emb_tables.shape[1]
    assert emb_dim % L == 0
    sc = pltpu.get_tpu_info().sparse_core  # pyrefly: ignore [missing-attribute]
    mesh = plsc.VectorSubcoreMesh(
        core_axis_name="c",
        subcore_axis_name="s",
        num_cores=sc.num_cores,
        num_subcores=sc.num_subcores,  # pyrefly: ignore [unexpected-keyword]
    )
    num_subcores = sc.num_cores * sc.num_subcores
    n = positions.shape[0]
    blocks_per_subcore = max(1, -(-unique_rows.shape[0] // (L * num_subcores)))
    grid = (blocks_per_subcore, num_subcores)
    launched = blocks_per_subcore * num_subcores * L
    unique_rows = jnp.pad(
        unique_rows, (0, launched - unique_rows.shape[0]), constant_values=-1
    )
    run_offsets = jnp.pad(
        run_offsets, (0, launched + 2 * L - run_offsets.shape[0]), constant_values=n
    )
    positions = jnp.pad(positions, (0, 2 * L))
    slots = jnp.pad(slots, (0, 2 * L))
    lr_lanes = jnp.full((L,), lr, jnp.float32)
    key_lanes = jnp.full((L,), rounding_key, jnp.int32)

    @pl.kernel(  # pyrefly: ignore [missing-attribute]
        out_type=(),
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
            use_tc_tiling_on_sc=False,  # pyrefly: ignore [unexpected-keyword]
            needs_layout_passes=False,  # pyrefly: ignore [unexpected-keyword]
        ),
        scratch_types=[
            pltpu.VMEM((L,), jnp.float32),  # learning rate
            pltpu.VMEM((L,), jnp.int32),  # rounding key
            pltpu.VMEM((2 * L,), jnp.int32),  # run offsets of the block
            pltpu.VMEM((L,), jnp.int32),  # table rows of the block
            pltpu.VMEM((L,), jnp.int32),  # positions of a chunk
            pltpu.VMEM((L,), jnp.int32),  # slots of a chunk
            pltpu.VMEM((L, emb_dim), grad_out.dtype),  # gradients of a chunk
            pltpu.VMEM((L, emb_dim), jnp.float32),  # summed gradient per row
            pltpu.VMEM((L, emb_dim), emb_tables.dtype),  # table rows
            pltpu.VMEM((L,), momentum.dtype),  # momentum per row
        ],
        name="batched_rowwise_adagrad_sc",
        mesh=mesh,
    )
    def kernel(
        rows_hbm,
        positions_hbm,
        slots_hbm,
        run_offsets_hbm,
        lr_hbm,
        key_hbm,
        grad_out_hbm,
        momentum_hbm,
        emb_tables_hbm,
        lr_vmem,
        key_vmem,
        offsets_vmem,
        rows_vmem,
        positions_vmem,
        slots_vmem,
        grad_vmem,
        acc_vmem,
        emb_vmem,
        momentum_vmem,
    ):
        @functools.partial(
            pltpu.emit_pipeline,
            grid=grid,
            core_axis_name=("c", "s"),
            dimension_semantics=(pltpu.ARBITRARY, pltpu.PARALLEL),
        )
        def _inner():
            lanes = jnp.arange(L, dtype=jnp.int32)
            r0 = pl.multiple_of(
                (pl.program_id(0) * num_subcores + pl.program_id(1)) * L, L
            )
            pltpu.sync_copy(run_offsets_hbm.at[pl.ds(r0, 2 * L)], offsets_vmem)
            start = plsc.load_gather(offsets_vmem, (jnp.zeros((L,), jnp.int32),))[0]
            end = plsc.load_gather(offsets_vmem, (jnp.full((L,), L, jnp.int32),))[0]

            @pl.when(start < end)
            def _block():
                pltpu.sync_copy(lr_hbm, lr_vmem)
                pltpu.sync_copy(key_hbm, key_vmem)
                pltpu.sync_copy(rows_hbm.at[pl.ds(r0, L)], rows_vmem)
                rows = plsc.Indices(rows_vmem, ignored_value=-1)
                pltpu.sync_copy(emb_tables_hbm.at[rows], emb_vmem)
                pltpu.sync_copy(momentum_hbm.at[rows], momentum_vmem)

                @pl.loop(0, L)
                def _zero(j):
                    @pl.loop(0, emb_dim, step=L)
                    def _zero_cols(c):
                        acc_vmem[j, pl.ds(c, L)] = jnp.zeros((L,), jnp.float32)

                aligned = pl.multiple_of((start // 8) * 8, 8)

                @pl.loop(0, (end - aligned + L - 1) // L)
                def _chunk(i):
                    cs = pl.multiple_of(aligned + i * L, 8)
                    pltpu.sync_copy(positions_hbm.at[pl.ds(cs, L)], positions_vmem)
                    pltpu.sync_copy(slots_hbm.at[pl.ds(cs, L)], slots_vmem)
                    p = cs + lanes
                    positions_vmem[...] = jnp.where(
                        (p >= start) & (p < end), positions_vmem[...], -1
                    )
                    pltpu.sync_copy(
                        grad_out_hbm.at[plsc.Indices(positions_vmem, ignored_value=-1)],
                        grad_vmem,
                    )

                    @pl.loop(0, L)
                    def _add(k):
                        @pl.when((cs + k >= start) & (cs + k < end))
                        def _():
                            j = plsc.load_gather(
                                slots_vmem, (jnp.full((L,), k, jnp.int32),)
                            )[0]

                            @pl.loop(0, emb_dim, step=L)
                            def _cols(c):
                                acc_vmem[j, pl.ds(c, L)] += grad_vmem[
                                    k, pl.ds(c, L)
                                ].astype(jnp.float32)

                sums = jnp.zeros((L,), jnp.float32)
                for j in range(L):
                    s = jnp.float32(0)
                    for c in range(0, emb_dim, L):
                        g = acc_vmem[j, pl.ds(c, L)]
                        s += jnp.sum(g * g)
                    sums = jnp.where(lanes == j, s, sums)
                new_m = momentum_vmem[...] + sums * (1.0 / emb_dim)
                step = lr_vmem[...] / (new_m * _rsqrt(new_m) + jnp.float32(eps))
                for j in range(L):
                    step_j = jnp.full(
                        (L,), jnp.sum(jnp.where(lanes == j, step, 0.0)), jnp.float32
                    )
                    # Random bits per (step and device, row, column).
                    row_key = key_vmem[...] ^ _hash32(
                        plsc.load_gather(rows_vmem, (jnp.full((L,), j, jnp.int32),))
                        * jnp.int32(-1640531535)  # 0x9E3779B1
                    )

                    # pl.loop traces its body right away, so j, step_j and row_key
                    # are current.
                    @pl.loop(0, emb_dim, step=L)
                    def _update(c):
                        cols = pl.ds(c, L)
                        new = emb_vmem[j, cols] - step_j * acc_vmem[j, cols]  # noqa: B023
                        if round_fp16:
                            new = _stochastic_round_fp16(
                                new,
                                _hash32(row_key ^ (c + lanes)),  # noqa: B023
                            )
                        emb_vmem[j, cols] = new  # noqa: B023

                momentum_vmem[...] = new_m
                pltpu.sync_copy(emb_vmem, emb_tables_hbm.at[rows])
                pltpu.sync_copy(momentum_vmem, momentum_hbm.at[rows])

        _inner()

    weights_ref = jax.new_ref(emb_tables)
    momentum_ref = jax.new_ref(momentum)
    kernel(
        unique_rows,
        positions,
        slots,
        run_offsets,
        lr_lanes,
        key_lanes,
        grad_out,
        momentum_ref,
        weights_ref,
    )
    return weights_ref[...], momentum_ref[...]
