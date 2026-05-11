# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Litmus tests: does the per-wave REGULAR-sem post-barrier in the v50 multi-
wave Pallas kernel actually provide receiver-side completion sync after
``pltpu.make_async_remote_copy(...).wait()``?

Two minimal 2-rank tests, isolated from the production kernel. Each rank's
per-device kernel issues exactly one cross-device DMA (rank 0 sends a
known constant to rank 1 and vice-versa), guarded fore and aft by REGULAR
semaphore barriers using ``pl.semaphore_signal(..., device_id=peer)`` and
``pl.semaphore_wait``. The host-side test then inspects the receive buffer
on each rank.

  * Test A (single-iter): straight-line kernel body, no ``lax.fori_loop``.
    If A fails, the post-barrier semantics are unsound for inbound flush
    and the v50 design is structurally wrong.
  * Test B (2-iter ``lax.fori_loop`` variant): same body wrapped in a
    fori_loop with per-iteration sem array slots ``barrier_sems.at[w, ...]``
    and per-iteration distinct send values. If A passes but B fails, the
    bug is in ``fori_loop`` lowering of REGULAR sem ops.

Run via the existing JobSet machinery on the v7x cluster:

    python -u -m unittest -v \\
        recir.ops.pallas.embeddings.tests.test_inbound_flush_litmus
"""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


_N = 128  # int32 elements; one TILE
_SENTINEL = -1


def _has_tpu() -> bool:
    """Return True if real TPU devices are available."""
    try:
        return any(d.platform == "tpu" for d in jax.devices())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Test A: single-iter litmus, no fori_loop
# ---------------------------------------------------------------------------


def _make_single_iter_kernel(world_size: int):
    """Per-device kernel: pre-barrier, peer DMA, post-barrier. No reads of
    recv_buf inside the kernel — host inspects after the kernel returns.
    """

    def kernel(
        src_ref,        # [N] int32 HBM (this rank's send payload)
        recv_ref,       # [N] int32 HBM (this rank's recv buffer; overwritten
                        # by the peer's DMA)
        pre_sem,        # REGULAR scratch
        post_sem,       # REGULAR scratch
        send_sem,       # DMA scratch
        recv_sem,       # DMA scratch
        zero_vmem,      # [N] int32 VMEM scratch (used to seed recv with -1)
    ):
        my_rank = lax.axis_index("x")
        peer = (my_rank + 1) % world_size

        # Seed recv with sentinel -1 so a missing inbound flush would leave
        # the slot at -1 (or 0 if the post-barrier did not flush either).
        zero_vmem[...] = jnp.full_like(zero_vmem, _SENTINEL)
        pltpu.sync_copy(zero_vmem, recv_ref)

        # Pre-barrier: every rank must finish seeding recv before any peer
        # DMAs into it.
        pl.semaphore_signal(
            pre_sem, 1, device_id=(peer,),
            device_id_type=pltpu.DeviceIdType.MESH,
        )
        pl.semaphore_wait(pre_sem, 1)

        # Peer DMA: I write src_ref into peer's recv_ref.
        d = pltpu.make_async_remote_copy(
            src_ref=src_ref,
            dst_ref=recv_ref,
            send_sem=send_sem,
            recv_sem=recv_sem,
            device_id=(peer,),
            device_id_type=pltpu.DeviceIdType.MESH,
        )
        d.start()
        d.wait()

        # Post-barrier: every rank must observe its recv_ref fully populated
        # before the kernel returns. Without this, the host read after the
        # call could race with in-flight inbound DMA writes.
        pl.semaphore_signal(
            post_sem, 1, device_id=(peer,),
            device_id_type=pltpu.DeviceIdType.MESH,
        )
        pl.semaphore_wait(post_sem, 1)
        # NO reads of recv_ref inside kernel; the kernel just exits.

    return kernel


def _run_single_iter(world_size: int) -> np.ndarray:
    """Returns ``[world_size, N]`` int32 of each rank's recv_buf after the
    kernel returns.
    """
    devices = jax.devices()[:world_size]
    mesh = jax.sharding.Mesh(np.array(devices), ("x",))

    # Per-rank distinct send payload: rank 0 sends 0xCAFE, rank 1 sends 0xBEEF.
    send_values = jnp.array(
        [[0xCAFE] * _N, [0xBEEF] * _N], dtype=jnp.int32
    )  # [W, N]

    HBM = pltpu.MemorySpace.HBM
    kernel = _make_single_iter_kernel(world_size=world_size)

    def per_device(src):
        # src: [1, N] (sharded along axis 0)
        recv = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((_N,), jnp.int32),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[pl.BlockSpec(memory_space=HBM)],
                out_specs=pl.BlockSpec(memory_space=HBM),
                scratch_shapes=(
                    pltpu.SemaphoreType.REGULAR,  # pre_sem
                    pltpu.SemaphoreType.REGULAR,  # post_sem
                    pltpu.SemaphoreType.DMA,      # send_sem
                    pltpu.SemaphoreType.DMA,      # recv_sem
                    pltpu.VMEM((_N,), jnp.int32),
                ),
            ),
            interpret=False,
        )(src[0])
        return recv[None, ...]

    P = jax.sharding.PartitionSpec
    sharded = jax.shard_map(
        per_device, mesh=mesh,
        in_specs=(P("x"),),
        out_specs=P("x"),
        check_vma=False,
    )
    sharding = jax.sharding.NamedSharding(mesh, P("x"))
    src = jax.device_put(send_values, sharding)
    out = sharded(src)
    return np.array(out)  # [W, N]


# ---------------------------------------------------------------------------
# Test B: lax.fori_loop-wrapped 2-iteration variant
# ---------------------------------------------------------------------------


_NUM_ITERS = 2


def _make_fori_kernel(world_size: int, num_iters: int):
    """Per-device kernel: ``lax.fori_loop`` over ``num_iters`` iterations,
    each iteration sending a different value. Uses per-iter sem array slots.
    Final recv_buf should hold the iter-(num_iters-1) value from the peer.
    """

    def kernel(
        src_ref,        # [num_iters, N] int32 HBM (per-iter payload)
        recv_ref,       # [N] int32 HBM
        barrier_sems,   # REGULAR scratch (num_iters, 2)
        send_sem,       # DMA scratch
        recv_sem,       # DMA scratch
        zero_vmem,      # [N] int32 VMEM scratch
    ):
        my_rank = lax.axis_index("x")
        peer = (my_rank + 1) % world_size

        # Seed once at the start (matches the v50 kernel's per-wave zero-fill
        # pattern in spirit, but simplified — we want to see the iter-1 value
        # land regardless of seed handling).
        zero_vmem[...] = jnp.full_like(zero_vmem, _SENTINEL)
        pltpu.sync_copy(zero_vmem, recv_ref)

        def body(w, _carry):
            pre_sem = barrier_sems.at[w, 0]
            post_sem = barrier_sems.at[w, 1]

            # Pre-barrier.
            pl.semaphore_signal(
                pre_sem, 1, device_id=(peer,),
                device_id_type=pltpu.DeviceIdType.MESH,
            )
            pl.semaphore_wait(pre_sem, 1)

            # Per-iter peer DMA.
            d = pltpu.make_async_remote_copy(
                src_ref=src_ref.at[w],
                dst_ref=recv_ref,
                send_sem=send_sem,
                recv_sem=recv_sem,
                device_id=(peer,),
                device_id_type=pltpu.DeviceIdType.MESH,
            )
            d.start()
            d.wait()

            # Post-barrier.
            pl.semaphore_signal(
                post_sem, 1, device_id=(peer,),
                device_id_type=pltpu.DeviceIdType.MESH,
            )
            pl.semaphore_wait(post_sem, 1)
            return None

        lax.fori_loop(0, num_iters, body, None)

    return kernel


def _run_fori_variant(world_size: int, num_iters: int) -> np.ndarray:
    """Returns ``[world_size, N]`` int32 of each rank's recv_buf after the
    fori_loop kernel returns.
    """
    devices = jax.devices()[:world_size]
    mesh = jax.sharding.Mesh(np.array(devices), ("x",))

    # Per-rank, per-iter payloads.
    #   rank 0: [0xCAFE0, 0xCAFE1, ...]
    #   rank 1: [0xBEEF0, 0xBEEF1, ...]
    rank0_payloads = np.stack([
        np.full((_N,), 0xCAFE0 + i, dtype=np.int32) for i in range(num_iters)
    ])
    rank1_payloads = np.stack([
        np.full((_N,), 0xBEEF0 + i, dtype=np.int32) for i in range(num_iters)
    ])
    send_values = jnp.array(
        np.stack([rank0_payloads, rank1_payloads])
    )  # [W, num_iters, N]

    HBM = pltpu.MemorySpace.HBM
    kernel = _make_fori_kernel(world_size=world_size, num_iters=num_iters)

    def per_device(src):
        # src: [1, num_iters, N]
        recv = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((_N,), jnp.int32),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[pl.BlockSpec(memory_space=HBM)],
                out_specs=pl.BlockSpec(memory_space=HBM),
                scratch_shapes=(
                    # Per-iter sem array: 2 slots (pre, post) per iter.
                    pltpu.SemaphoreType.REGULAR((num_iters, 2)),
                    pltpu.SemaphoreType.DMA,  # send_sem
                    pltpu.SemaphoreType.DMA,  # recv_sem
                    pltpu.VMEM((_N,), jnp.int32),
                ),
            ),
            interpret=False,
        )(src[0])
        return recv[None, ...]

    P = jax.sharding.PartitionSpec
    sharded = jax.shard_map(
        per_device, mesh=mesh,
        in_specs=(P("x"),),
        out_specs=P("x"),
        check_vma=False,
    )
    sharding = jax.sharding.NamedSharding(mesh, P("x"))
    src = jax.device_put(send_values, sharding)
    out = sharded(src)
    return np.array(out)  # [W, N]


# ---------------------------------------------------------------------------
# unittest classes
# ---------------------------------------------------------------------------


@unittest.skipUnless(_has_tpu(), "Litmus test requires real TPU")
class InboundFlushLitmusTest(unittest.TestCase):
    """2-rank inbound-flush litmus for REGULAR-sem post-barrier semantics."""

    def setUp(self) -> None:
        if jax.device_count() < 2:
            self.skipTest(
                f"Need 2 devices, have {jax.device_count()}"
            )

    def test_a_single_iter(self) -> None:
        """Single-iter (no fori_loop): rank 0 should see 0xBEEF, rank 1 0xCAFE."""
        recv = _run_single_iter(world_size=2)
        print(f"[LITMUS_A] rank 0 recv[:8]={recv[0, :8].tolist()} "
              f"rank 1 recv[:8]={recv[1, :8].tolist()}", flush=True)
        # Rank 0 should see peer (rank 1)'s send = 0xBEEF.
        np.testing.assert_array_equal(recv[0], np.full((_N,), 0xBEEF, dtype=np.int32),
            err_msg=(
                f"rank 0 expected all 0xBEEF (={0xBEEF}); got first 8 = "
                f"{recv[0, :8].tolist()} (sentinel -1 means inbound flush "
                f"NOT provided by post-barrier; 0 means the seed was overwritten "
                f"AFTER the DMA landed and the post-barrier did not flush)"
            ),
        )
        # Rank 1 should see peer (rank 0)'s send = 0xCAFE.
        np.testing.assert_array_equal(recv[1], np.full((_N,), 0xCAFE, dtype=np.int32),
            err_msg=(
                f"rank 1 expected all 0xCAFE (={0xCAFE}); got first 8 = "
                f"{recv[1, :8].tolist()}"
            ),
        )

    def test_b_fori_loop(self) -> None:
        """fori_loop variant: each rank should see the peer's iter-(N-1) value.

        Final expected:
          rank 0 recv = peer rank 1's iter-1 send = 0xBEEF1
          rank 1 recv = peer rank 0's iter-1 send = 0xCAFE1
        """
        recv = _run_fori_variant(world_size=2, num_iters=_NUM_ITERS)
        last = _NUM_ITERS - 1
        rank0_expected = 0xBEEF0 + last
        rank1_expected = 0xCAFE0 + last
        print(f"[LITMUS_B] rank 0 recv[:8]={recv[0, :8].tolist()} "
              f"rank 1 recv[:8]={recv[1, :8].tolist()}", flush=True)
        np.testing.assert_array_equal(
            recv[0], np.full((_N,), rank0_expected, dtype=np.int32),
            err_msg=(
                f"rank 0 expected all {rank0_expected} (=0xBEEF{last}); got "
                f"first 8 = {recv[0, :8].tolist()}. If iter-0 value "
                f"({0xBEEF0}) appears, fori_loop is dropping iter-1 work. "
                f"If sentinel -1 appears, the kernel barely ran."
            ),
        )
        np.testing.assert_array_equal(
            recv[1], np.full((_N,), rank1_expected, dtype=np.int32),
            err_msg=(
                f"rank 1 expected all {rank1_expected} (=0xCAFE{last}); got "
                f"first 8 = {recv[1, :8].tolist()}"
            ),
        )


# ---------------------------------------------------------------------------
# W=4 phase-1-only litmus (Codex seq-0030 option I)
#
# Reproduces the per-wave traffic pattern of the failing
# ``test_alltc_f32_world_size_4_skewed_overflow_waves`` configuration but with
# ONLY the phase-1 ID a2a body (no lookup, no phase-3 emb a2a). Isolates the
# W=4 fan-in + ``@pl.when(send_count > 0)`` conditional DMA + per-wave REGULAR
# pre/post barrier path. If this passes, phase 1 is sound and the v50 deadlock
# lives in phase 2/3 interaction.
# ---------------------------------------------------------------------------


# Match the failing-case constants from
# ``test_alltc_f32_world_size_4_skewed_overflow_waves``.
_W4_TILE = 128
_W4_VLOCAL = 256
_W4_MAX_LOCAL_IDS = 4 * _W4_TILE  # 512
_W4_RECV_CAPACITY = 512
_W4_SEED = 2


def _w4_compute_per_wave_traffic():
    """Compute ``(per_wave_traffic, per_wave_send_offsets, per_wave_recv_offsets,
    sorted_row_padded, max_waves)`` matching the failing case, on host.

    Uses the SAME helpers (`_make_skewed_inputs`, `_bucketize_and_sort`,
    `_round_up_tile`, `_pad_per_peer`, `_split_traffic_into_waves`,
    `_max_waves_for_capacity`) as ``jagged_a2a_embedding_lookup_f32_alltc`` so
    the per-wave matrices are byte-for-byte identical to the failing v50 run.
    """
    from recir.ops.pallas.embeddings.jagged_a2a_lookup import (
        _bucketize_and_sort,
        _max_waves_for_capacity,
        _pad_per_peer,
        _round_up_tile,
        _split_traffic_into_waves,
    )
    from recir.ops.pallas.embeddings.tests.test_jagged_a2a_lookup import (
        _make_skewed_inputs,
    )

    world_size = 4
    max_padded = _W4_MAX_LOCAL_IDS + world_size * _W4_TILE
    max_waves = _max_waves_for_capacity(
        _W4_MAX_LOCAL_IDS, world_size, _W4_RECV_CAPACITY
    )

    local_ids_np, local_id_count_np = _make_skewed_inputs(
        world_size, _W4_VLOCAL, _W4_MAX_LOCAL_IDS, seed=_W4_SEED,
    )

    # Per-rank: bucketize → pad → derive padded send_counts → traffic matrix
    # gathered across ranks via NUMPY (we don't need lax.all_gather here since
    # the host computes the same per-rank traffic_matrix as the kernel sees).
    sorted_row_padded_per_rank = []
    send_counts_padded_per_rank = []
    for r in range(world_size):
        sorted_row_real, send_counts_real, _, _ = _bucketize_and_sort(
            jnp.array(local_ids_np[r]),
            jnp.array(local_id_count_np[r]),
            _W4_MAX_LOCAL_IDS, world_size,
        )
        send_counts_padded = _round_up_tile(send_counts_real, _W4_TILE)
        sorted_row_padded = _pad_per_peer(
            sorted_row_real, send_counts_real, send_counts_padded,
            _W4_MAX_LOCAL_IDS, max_padded, world_size,
        )
        sorted_row_padded_per_rank.append(np.array(sorted_row_padded))
        send_counts_padded_per_rank.append(np.array(send_counts_padded))

    traffic_matrix = jnp.array(np.stack(send_counts_padded_per_rank))  # [W, W]
    per_wave_traffic, per_wave_send_offsets, per_wave_recv_offsets = (
        _split_traffic_into_waves(
            traffic_matrix,
            recv_capacity=_W4_RECV_CAPACITY,
            max_waves=max_waves,
            world_size=world_size,
        )
    )

    return (
        np.array(per_wave_traffic),         # [max_waves, W, W]
        np.array(per_wave_send_offsets),    # [max_waves, W, W]
        np.array(per_wave_recv_offsets),    # [max_waves, W, W]
        np.stack(sorted_row_padded_per_rank),  # [W, max_padded]
        int(max_waves),
        max_padded,
    )


def _make_w4_phase1_kernel(world_size: int, max_waves: int, recv_capacity: int):
    """Phase-1-only multi-wave kernel — Codex pairwise refinement (v54).

    Per wave: zero-fill recv_row_buf, PRE-handshake via pairwise REGULAR
    ``ready_sems`` (each rank signals its own slot on every peer; waits on
    each non-self peer's slot), per-peer ``@pl.when(send_count > 0)`` guarded
    DMA, POST-handshake via pairwise REGULAR ``capacity_sems`` (each rank
    signals its own slot on every peer UNCONDITIONALLY — inside the @pl.when
    block, after d.wait(); outside, immediately when count == 0).

    Key change vs v53: replaced single fan-in W barrier_sems[(max_waves, 2)]
    with TWO pairwise sem arrays — ``ready_sems[(max_waves, W)]`` and
    ``capacity_sems[(max_waves, W)]``. Each cross-rank channel becomes a
    pairwise (count=1) handshake, no W-way fan-in. Pre-handshake fences the
    zero-fill against fast peers' inbound DMAs (Codex flagged the missing
    fence as a real race in earlier proposals).
    """
    del recv_capacity  # only used for buffer sizing on the host side
    del max_waves  # consumed by the grid axis at pallas_call site

    def kernel(
        sorted_row_ref,    # [max_padded] int32 HBM
        traffic_ref,       # [max_waves, W, W] int32 SMEM
        send_off_ref,      # [max_waves, W, W] int32 SMEM
        recv_off_ref,      # [max_waves, W, W] int32 SMEM
        recv_row_buf,      # [recv_capacity] int32 HBM (output, host inspects)
        zero_int_vmem,     # VMEM scratch
        ready_sems,        # REGULAR (max_waves, W) — pre-handshake
        capacity_sems,     # REGULAR (max_waves, W) — post-handshake
        send_sem_id,       # DMA send sem
        recv_sem_id,       # DMA recv sem
    ):
        my_rank = lax.axis_index("x")
        w = pl.program_id(0)

        zero_int_vmem[...] = jnp.zeros_like(zero_int_vmem)

        # Phase 0: zero-fill recv_row_buf.
        pltpu.sync_copy(zero_int_vmem, recv_row_buf)

        # PRE-handshake: tell peers "my zero-fill is done" via pairwise
        # ready_sems. Signal my own slot on every peer (incl. self for
        # symmetry — self-signal is a scalar bump, no ICI traffic).
        for peer in range(world_size):
            pl.semaphore_signal(
                ready_sems.at[w, my_rank], 1,
                device_id=(peer,),
                device_id_type=pltpu.DeviceIdType.MESH,
            )
        # Wait on EVERY peer's ready slot including self. ``my_rank`` is a
        # traced scalar so we can't ``if peer != my_rank``-skip in Python; we
        # rely on the symmetry that each rank also signaled itself, so
        # ``ready_sems.at[w, peer]`` has count=1 from peer→self for every peer
        # including self. Wait(1) drains each.
        for peer in range(world_size):
            pl.semaphore_wait(ready_sems.at[w, peer], 1)

        # Phase 1 ID a2a: per-peer DMAs. Capacity signal fires
        # UNCONDITIONALLY for each peer.
        for peer in range(world_size):
            send_count = pl.multiple_of(
                traffic_ref[w, my_rank, peer], _W4_TILE
            )
            send_start = pl.multiple_of(
                send_off_ref[w, my_rank, peer], _W4_TILE
            )
            remote_recv_start = pl.multiple_of(
                recv_off_ref[w, peer, my_rank], _W4_TILE
            )

            @pl.when(send_count > 0)
            def _do_id_send():
                src = sorted_row_ref.at[pl.ds(send_start, send_count)]
                dst = recv_row_buf.at[pl.ds(remote_recv_start, send_count)]
                d = pltpu.make_async_remote_copy(
                    src_ref=src, dst_ref=dst,
                    send_sem=send_sem_id, recv_sem=recv_sem_id,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
                d.start()
                d.wait()

            # CRITICAL: signal capacity_sems UNCONDITIONALLY (outside @pl.when)
            # — when send_count > 0, signal AFTER d.wait() returns
            # — when send_count == 0, signal IMMEDIATELY (no DMA was issued,
            #   but the handshake is needed so peer's pairwise wait doesn't
            #   block forever).
            pl.semaphore_signal(
                capacity_sems.at[w, my_rank], 1,
                device_id=(peer,),
                device_id_type=pltpu.DeviceIdType.MESH,
            )

        # POST-handshake: wait on every peer's capacity slot including self.
        # Same rationale as ready_sems: self-signal symmetry leaves each slot
        # with count=1 from its peer→self path; wait(1) drains each.
        for src in range(world_size):
            pl.semaphore_wait(capacity_sems.at[w, src], 1)

    return kernel


def _run_w4_phase1_litmus():
    """Run the W=4 phase-1-only kernel; return ``(recv_per_rank, expected_first,
    per_wave_traffic, per_wave_send_offsets, per_wave_recv_offsets,
    sorted_row_padded, max_waves)``.

    ``recv_per_rank``: ``[W, recv_capacity]`` int32 — final recv_row_buf on each
    rank after the kernel returns.

    ``expected_first[t, s]``: int32 — the value the LAST wave should have
    deposited at offset ``per_wave_recv_offsets[max_waves-1, t, s]`` in rank
    ``t``'s recv_row_buf, IF the per-(s, t) DMA in the last wave had count > 0.
    Equals ``sorted_row_padded[s, per_wave_send_offsets[max_waves-1, s, t]]``
    when ``per_wave_traffic[max_waves-1, s, t] > 0``; sentinel ``-1`` otherwise
    (meaning "no DMA happened in the last wave for this (s, t)").
    """
    world_size = 4
    devices = jax.devices()[:world_size]
    mesh = jax.sharding.Mesh(np.array(devices), ("x",))

    (per_wave_traffic_np, per_wave_send_offsets_np, per_wave_recv_offsets_np,
     sorted_row_padded_np, max_waves, max_padded) = _w4_compute_per_wave_traffic()

    # Build expected_first[t, s]: if last wave sent anything from s→t, the
    # FIRST element landed at recv_off[last, t, s] is sorted_row_padded[s,
    # send_off[last, s, t]]. Else -1 (sentinel = "no DMA, slot stays zero from
    # the per-wave zero-fill").
    last = max_waves - 1
    expected_first = np.full((world_size, world_size), _SENTINEL, dtype=np.int32)
    for t in range(world_size):
        for s in range(world_size):
            cnt = int(per_wave_traffic_np[last, s, t])
            if cnt > 0:
                so = int(per_wave_send_offsets_np[last, s, t])
                expected_first[t, s] = int(sorted_row_padded_np[s, so])

    # Per-rank inputs.
    HBM = pltpu.MemorySpace.HBM
    SMEM = pltpu.MemorySpace.SMEM

    kernel = _make_w4_phase1_kernel(
        world_size=world_size, max_waves=max_waves,
        recv_capacity=_W4_RECV_CAPACITY,
    )

    # SMEM refs are SHARED inputs across ranks (each rank reads its own row).
    # Per the v50 pattern, traffic_ref is the GLOBAL [max_waves, W, W] matrix,
    # broadcast as a replicated scalar input — but in shard_map we shard along
    # ``x`` only on the per-rank inputs (sorted_row_padded_np). The per-wave
    # matrices are identical on every rank, so we put them in P() (replicated).
    P = jax.sharding.PartitionSpec

    def per_device(src, traffic, send_off, recv_off):
        # src: [1, max_padded] (sharded); traffic/send_off/recv_off: replicated.
        recv = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((_W4_RECV_CAPACITY,), jnp.int32),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(max_waves,),
                in_specs=[
                    pl.BlockSpec(memory_space=HBM),   # sorted_row_padded
                    pl.BlockSpec(memory_space=SMEM),  # per_wave_traffic
                    pl.BlockSpec(memory_space=SMEM),  # per_wave_send_offsets
                    pl.BlockSpec(memory_space=SMEM),  # per_wave_recv_offsets
                ],
                out_specs=pl.BlockSpec(memory_space=HBM),
                scratch_shapes=(
                    pltpu.VMEM((_W4_RECV_CAPACITY,), jnp.int32),
                    pltpu.SemaphoreType.REGULAR((max_waves, world_size)),  # ready_sems
                    pltpu.SemaphoreType.REGULAR((max_waves, world_size)),  # capacity_sems
                    pltpu.SemaphoreType.DMA,
                    pltpu.SemaphoreType.DMA,
                ),
            ),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("arbitrary",),
            ),
            interpret=False,
        )(src[0], traffic, send_off, recv_off)
        return recv[None, ...]

    sharded = jax.shard_map(
        per_device, mesh=mesh,
        in_specs=(P("x"), P(), P(), P()),
        out_specs=P("x"),
        check_vma=False,
    )

    sharding_x = jax.sharding.NamedSharding(mesh, P("x"))
    sharding_rep = jax.sharding.NamedSharding(mesh, P())
    src = jax.device_put(jnp.array(sorted_row_padded_np), sharding_x)
    traffic = jax.device_put(jnp.array(per_wave_traffic_np), sharding_rep)
    send_off = jax.device_put(jnp.array(per_wave_send_offsets_np), sharding_rep)
    recv_off = jax.device_put(jnp.array(per_wave_recv_offsets_np), sharding_rep)

    out = sharded(src, traffic, send_off, recv_off)
    return (
        np.array(out),  # [W, recv_capacity]
        expected_first,
        per_wave_traffic_np,
        per_wave_send_offsets_np,
        per_wave_recv_offsets_np,
        sorted_row_padded_np,
        max_waves,
    )


@unittest.skipUnless(_has_tpu(), "Litmus test requires real TPU")
class W4Phase1Litmus(unittest.TestCase):
    """W=4 phase-1-only litmus matching the failing v50 traffic pattern."""

    def setUp(self) -> None:
        if jax.device_count() < 4:
            self.skipTest(
                f"Need 4 devices, have {jax.device_count()}"
            )

    @unittest.expectedFailure  # known TPU/Mosaic deadlock; even pairwise pre/post handshakes hang on overflow_waves traffic — structural blocker is deeper than W-way fan-in × multi-iter (validated v55, 2026-05-06)
    def test_w4_phase1_overflow_traffic_pattern(self) -> None:
        """W=4 phase-1-only: per-rank recv_row_buf at last-wave recv offsets
        must hold the first element each peer sent in the last wave."""
        (recv, expected_first, traffic, send_off, recv_off,
         sorted_row_padded, max_waves) = _run_w4_phase1_litmus()
        last = max_waves - 1

        # Print top-line state for the report.
        print(f"[LITMUS_W4] max_waves={max_waves}", flush=True)
        print(
            f"[LITMUS_W4] per_wave_traffic[last]=\n{traffic[last]}",
            flush=True,
        )
        print(
            f"[LITMUS_W4] per_wave_recv_offsets[last]=\n{recv_off[last]}",
            flush=True,
        )
        for r in range(4):
            print(
                f"[LITMUS_W4] rank {r} recv_row_buf[:8]={recv[r, :8].tolist()}",
                flush=True,
            )

        # Per-rank check: at offset recv_off[last, r, s], we expect
        # expected_first[r, s]. If sent count == 0 for (s, r) in the last wave,
        # we expect the slot to be 0 (the zero-fill sentinel from this wave).
        mismatches = []
        for r in range(4):
            for s in range(4):
                cnt_last = int(traffic[last, s, r])
                offset = int(recv_off[last, r, s])
                actual = int(recv[r, offset])
                if cnt_last > 0:
                    exp = int(expected_first[r, s])
                    if actual != exp:
                        mismatches.append(
                            f"rank={r} src={s} wave={last} off={offset} "
                            f"actual={actual} expected={exp} (cnt={cnt_last})"
                        )
                # If cnt_last == 0, the recv slot for this (r, s) was not
                # touched by the last wave. We don't assert on it (could be
                # leftover from earlier wave, or zero from the per-wave fill).
        if mismatches:
            self.fail(
                "W=4 phase-1-only DATA MISMATCH:\n  " + "\n  ".join(mismatches)
            )


# ---------------------------------------------------------------------------
# W=4 BARRIER-ONLY multi-iter litmus (v56)
#
# Absolute-minimum diagnostic: W=4 mesh, multi-wave (lax.fori_loop), REGULAR-sem
# cross-device barriers, NO DMAs, NO lookup, NO conditional logic. Just the
# dumbest possible barrier-only loop. This isolates whether the v52/v53/v55
# deadlock is caused by barriers alone at W=4 multi-iter, or requires the
# barrier x DMA interaction.
# ---------------------------------------------------------------------------


_BARRIER_ONLY_MAX_WAVES = 2


def _make_barrier_only_kernel(world_size: int, max_waves: int):
    """Per-device kernel: lax.fori_loop over max_waves; per-iter, signal every
    peer (incl. self) on barrier_sems.at[w], then wait world_size signals.
    Write a sentinel literal via VMEM->HBM sync_copy to prove we got past the
    loop. (HBM refs cannot be directly stored to from kernel; must go via VMEM.)
    """

    def kernel(out_ref, sentinel_vmem, barrier_sems):
        def wave_body(w, _carry):
            sem = barrier_sems.at[w]
            # Signal ALL peers (including self -- symmetric simplifies the loop;
            # self-signal is a scalar bump with no ICI traffic).
            for peer in range(world_size):
                pl.semaphore_signal(
                    sem, 1,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
            # Wait for W signals (one from each peer, including self).
            pl.semaphore_wait(sem, world_size)
            return None

        lax.fori_loop(0, max_waves, wave_body, None)

        # Sentinel: literal proving we got past the loop. Write into VMEM
        # scratch then sync_copy into HBM out.
        sentinel_vmem[...] = jnp.full_like(sentinel_vmem, max_waves)
        pltpu.sync_copy(sentinel_vmem, out_ref)

    return kernel


def _run_barrier_only(world_size: int, max_waves: int) -> np.ndarray:
    """Returns ``[world_size]`` int32 of each rank's out_ref[0] after the
    barrier-only kernel returns. Should be ``max_waves`` on every rank.
    """
    devices = jax.devices()[:world_size]
    mesh = jax.sharding.Mesh(np.array(devices), ("x",))

    HBM = pltpu.MemorySpace.HBM
    kernel = _make_barrier_only_kernel(
        world_size=world_size, max_waves=max_waves,
    )

    def per_device():
        out = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((1,), jnp.int32),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[],
                out_specs=pl.BlockSpec(memory_space=HBM),
                scratch_shapes=(
                    pltpu.VMEM((1,), jnp.int32),  # sentinel_vmem
                    pltpu.SemaphoreType.REGULAR((max_waves,)),
                ),
            ),
            interpret=False,
        )()
        return out[None, ...]

    P = jax.sharding.PartitionSpec
    sharded = jax.shard_map(
        per_device, mesh=mesh,
        in_specs=(),
        out_specs=P("x"),
        check_vma=False,
    )
    out = sharded()
    return np.array(out).reshape(world_size)  # [W]


@unittest.skipUnless(_has_tpu(), "Litmus test requires real TPU")
class W4BarrierOnlyLitmus(unittest.TestCase):
    """W=4 multi-iter REGULAR-sem barrier-only litmus (NO DMAs, NO lookup)."""

    def setUp(self) -> None:
        if jax.device_count() < 4:
            self.skipTest(
                f"Need 4 devices, have {jax.device_count()}"
            )

    def test_barrier_only_no_dma(self) -> None:
        """W=4 barrier-only multi-iter: every rank's out[0] must be max_waves.

        300s host-fetch timeout via threading; if the kernel hangs, the
        executor thread will not return and the test will error out cleanly.
        """
        import threading

        result_box: list = []
        error_box: list = []

        def runner():
            try:
                result_box.append(
                    _run_barrier_only(
                        world_size=4, max_waves=_BARRIER_ONLY_MAX_WAVES,
                    )
                )
            except BaseException as e:  # noqa: BLE001
                error_box.append(e)

        t = threading.Thread(target=runner, daemon=True)
        t.start()
        t.join(timeout=300.0)
        if t.is_alive():
            self.fail(
                "W=4 BARRIER-ONLY litmus HUNG after 300s; barriers alone "
                "deadlock at W=4 multi-iter (no DMAs involved)."
            )
        if error_box:
            raise error_box[0]

        out = result_box[0]
        print(
            f"[LITMUS_W4_BARRIER_ONLY] max_waves={_BARRIER_ONLY_MAX_WAVES} "
            f"out per rank={out.tolist()}",
            flush=True,
        )
        np.testing.assert_array_equal(
            out, np.full((4,), _BARRIER_ONLY_MAX_WAVES, dtype=np.int32),
            err_msg=(
                f"Expected all ranks to write max_waves="
                f"{_BARRIER_ONLY_MAX_WAVES}; got {out.tolist()}"
            ),
        )


# ---------------------------------------------------------------------------
# W=4 UNCONDITIONAL-DMA multi-iter litmus (v58)
#
# Decisive litmus: W=4 mesh, max_waves=2, REGULAR-sem barriers + UNCONDITIONAL
# DMAs (NO @pl.when guard). Uses balanced round-robin traffic so every (src,
# dst) pair has the same NON-ZERO count, allowing us to drop @pl.when entirely.
# Isolates whether @pl.when(send_count > 0) is the trigger for the v52/v53/v55
# deadlock.
#
# v57 proved barriers alone at W=4 multi-iter PASS.
# v23 proved DMAs alone at W=4 single-wave PASS.
# Test B proved barriers+DMAs at W=2 multi-iter PASS (with unconditional DMAs).
# This v58 tests barriers+DMAs at W=4 multi-iter (with unconditional DMAs).
# ---------------------------------------------------------------------------


_W4_UNCOND_TILE = 128
_W4_UNCOND_MAX_WAVES = 2
_W4_UNCOND_PER_PAIR_COUNT = 128  # one TILE per (src, dst) per wave
_W4_UNCOND_WORLD_SIZE = 4
_W4_UNCOND_RECV_CAPACITY = (
    _W4_UNCOND_WORLD_SIZE * _W4_UNCOND_PER_PAIR_COUNT
)  # 512 per rank per wave
_W4_UNCOND_MAX_PADDED = (
    _W4_UNCOND_WORLD_SIZE * _W4_UNCOND_PER_PAIR_COUNT
)  # 512: each rank sends 128 to each of 4 peers


def _make_w4_uncond_kernel(world_size: int, max_waves: int):
    """Per-device kernel: lax.fori_loop over max_waves; per-iter, pre-barrier,
    UNCONDITIONAL per-peer DMAs (no @pl.when), post-barrier. Balanced traffic
    guarantees send_count > 0 for every (src, dst), so @pl.when is unneeded.
    """

    def kernel(
        sorted_row_ref,    # [max_padded] int32 HBM (per-rank sentinel-filled)
        traffic_ref,       # [max_waves, W, W] int32 SMEM
        send_off_ref,      # [max_waves, W, W] int32 SMEM
        recv_off_ref,      # [max_waves, W, W] int32 SMEM
        out_recv_ref,      # [recv_capacity] int32 HBM (final recv buffer)
        barrier_sems,      # REGULAR (max_waves, 2) — pre/post per wave
        send_sem,          # DMA send sem
        recv_sem,          # DMA recv sem
    ):
        my_rank = lax.axis_index("x")

        def wave_body(w, _carry):
            # Pre-barrier: signal every peer (incl. self), wait W signals.
            pre_sem = barrier_sems.at[w, 0]
            for peer in range(world_size):
                pl.semaphore_signal(
                    pre_sem, 1,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
            pl.semaphore_wait(pre_sem, world_size)

            # Per-peer UNCONDITIONAL DMA — no @pl.when guard. send_count is
            # guaranteed > 0 by balanced-traffic construction.
            for peer in range(world_size):
                send_count = pl.multiple_of(
                    traffic_ref[w, my_rank, peer], _W4_UNCOND_TILE
                )
                send_start = pl.multiple_of(
                    send_off_ref[w, my_rank, peer], _W4_UNCOND_TILE
                )
                remote_recv_start = pl.multiple_of(
                    recv_off_ref[w, peer, my_rank], _W4_UNCOND_TILE
                )

                d = pltpu.make_async_remote_copy(
                    src_ref=sorted_row_ref.at[pl.ds(send_start, send_count)],
                    dst_ref=out_recv_ref.at[
                        pl.ds(remote_recv_start, send_count)
                    ],
                    send_sem=send_sem, recv_sem=recv_sem,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
                d.start()
                d.wait()

            # Post-barrier: signal every peer (incl. self), wait W signals.
            post_sem = barrier_sems.at[w, 1]
            for peer in range(world_size):
                pl.semaphore_signal(
                    post_sem, 1,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
            pl.semaphore_wait(post_sem, world_size)
            return None

        lax.fori_loop(0, max_waves, wave_body, None)

    return kernel


def _run_w4_uncond_litmus():
    """Run W=4 unconditional-DMA multi-iter litmus.

    Returns ``(recv_per_rank, sentinels, per_wave_recv_offsets, max_waves)``.
    ``recv_per_rank``: ``[W, recv_capacity]`` int32 — final out_recv on each
    rank. ``sentinels[r]``: the int32 sentinel rank r filled its sorted_row
    with (e.g. 0xCAFE0000 + r). Each rank's recv slot at recv_off[last, r, s]
    should hold sentinels[s] (the source's sentinel).
    """
    world_size = _W4_UNCOND_WORLD_SIZE
    max_waves = _W4_UNCOND_MAX_WAVES
    per_pair_count = _W4_UNCOND_PER_PAIR_COUNT
    devices = jax.devices()[:world_size]
    mesh = jax.sharding.Mesh(np.array(devices), ("x",))

    # Balanced traffic: every (s, t) sends exactly per_pair_count every wave.
    per_wave_traffic_np = np.full(
        (max_waves, world_size, world_size),
        per_pair_count, dtype=np.int32,
    )
    # Send offsets [w, s, t] = t * per_pair_count (prefix sum across t).
    # Recv offsets [w, t, s] = s * per_pair_count (prefix sum across s).
    per_wave_send_offsets_np = np.zeros(
        (max_waves, world_size, world_size), dtype=np.int32,
    )
    per_wave_recv_offsets_np = np.zeros(
        (max_waves, world_size, world_size), dtype=np.int32,
    )
    for w in range(max_waves):
        for s in range(world_size):
            for t in range(world_size):
                per_wave_send_offsets_np[w, s, t] = t * per_pair_count
                per_wave_recv_offsets_np[w, t, s] = s * per_pair_count

    # Per-rank sentinels: rank r fills sorted_row with 0x0CAF0000 + r.
    # (Use 0x0CAF... not 0xCAFE... to stay within signed int32 range.)
    sentinels = np.array(
        [0x0CAF0000 + r for r in range(world_size)], dtype=np.int32,
    )
    sorted_row_padded_np = np.stack([
        np.full(
            (_W4_UNCOND_MAX_PADDED,), sentinels[r], dtype=np.int32,
        )
        for r in range(world_size)
    ])  # [W, max_padded]

    HBM = pltpu.MemorySpace.HBM
    SMEM = pltpu.MemorySpace.SMEM

    kernel = _make_w4_uncond_kernel(
        world_size=world_size, max_waves=max_waves,
    )

    P = jax.sharding.PartitionSpec

    def per_device(src, traffic, send_off, recv_off):
        recv = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(
                (_W4_UNCOND_RECV_CAPACITY,), jnp.int32,
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[
                    pl.BlockSpec(memory_space=HBM),   # sorted_row
                    pl.BlockSpec(memory_space=SMEM),  # traffic
                    pl.BlockSpec(memory_space=SMEM),  # send_off
                    pl.BlockSpec(memory_space=SMEM),  # recv_off
                ],
                out_specs=pl.BlockSpec(memory_space=HBM),
                scratch_shapes=(
                    pltpu.SemaphoreType.REGULAR((max_waves, 2)),
                    pltpu.SemaphoreType.DMA,
                    pltpu.SemaphoreType.DMA,
                ),
            ),
            interpret=False,
        )(src[0], traffic, send_off, recv_off)
        return recv[None, ...]

    sharded = jax.shard_map(
        per_device, mesh=mesh,
        in_specs=(P("x"), P(), P(), P()),
        out_specs=P("x"),
        check_vma=False,
    )

    sharding_x = jax.sharding.NamedSharding(mesh, P("x"))
    sharding_rep = jax.sharding.NamedSharding(mesh, P())
    src = jax.device_put(jnp.array(sorted_row_padded_np), sharding_x)
    traffic = jax.device_put(jnp.array(per_wave_traffic_np), sharding_rep)
    send_off = jax.device_put(
        jnp.array(per_wave_send_offsets_np), sharding_rep,
    )
    recv_off = jax.device_put(
        jnp.array(per_wave_recv_offsets_np), sharding_rep,
    )

    out = sharded(src, traffic, send_off, recv_off)
    return (
        np.array(out),  # [W, recv_capacity]
        sentinels,
        per_wave_recv_offsets_np,
        max_waves,
    )


@unittest.skipUnless(_has_tpu(), "Litmus test requires real TPU")
class W4UnconditionalDMALitmus(unittest.TestCase):
    """W=4 multi-iter REGULAR-sem barriers + UNCONDITIONAL DMAs (no @pl.when).
    """

    def setUp(self) -> None:
        if jax.device_count() < 4:
            self.skipTest(
                f"Need 4 devices, have {jax.device_count()}"
            )

    def test_w4_unconditional_dma_no_pl_when(self) -> None:
        """W=4 unconditional-DMA multi-iter: every rank's recv slot at
        recv_off[last, r, s] should hold the source rank s's sentinel.

        300s host-fetch timeout via threading; if the kernel hangs, the
        executor thread will not return and the test will fail cleanly.
        """
        import threading

        result_box: list = []
        error_box: list = []

        def runner():
            try:
                result_box.append(_run_w4_uncond_litmus())
            except BaseException as e:  # noqa: BLE001
                error_box.append(e)

        t = threading.Thread(target=runner, daemon=True)
        t.start()
        t.join(timeout=300.0)
        if t.is_alive():
            self.fail(
                "W=4 UNCONDITIONAL-DMA litmus HUNG after 300s; barrier x DMA "
                "x W=4 x multi-iter deadlocks even WITHOUT @pl.when "
                "conditionality. Pure upstream blocker."
            )
        if error_box:
            raise error_box[0]

        recv, sentinels, recv_off, max_waves = result_box[0]
        last = max_waves - 1
        per_pair = _W4_UNCOND_PER_PAIR_COUNT

        print(
            f"[LITMUS_W4_UNCOND] max_waves={max_waves} "
            f"sentinels={[hex(int(s)) for s in sentinels]}",
            flush=True,
        )
        for r in range(_W4_UNCOND_WORLD_SIZE):
            print(
                f"[LITMUS_W4_UNCOND] rank {r} recv first-of-each-source: "
                + ", ".join(
                    f"src{s}@off{int(recv_off[last, r, s])}="
                    f"{hex(int(recv[r, int(recv_off[last, r, s])]))}"
                    for s in range(_W4_UNCOND_WORLD_SIZE)
                ),
                flush=True,
            )

        # Per-rank check: at recv_off[last, r, s], we expect sentinels[s]
        # repeated per_pair times.
        mismatches = []
        for r in range(_W4_UNCOND_WORLD_SIZE):
            for s in range(_W4_UNCOND_WORLD_SIZE):
                offset = int(recv_off[last, r, s])
                expected = int(sentinels[s])
                slot = recv[r, offset:offset + per_pair]
                if not np.all(slot == expected):
                    actual_first8 = slot[:8].tolist()
                    mismatches.append(
                        f"rank={r} src={s} off={offset} "
                        f"expected={hex(expected)} actual_first8="
                        f"{[hex(int(x)) for x in actual_first8]}"
                    )
        if mismatches:
            self.fail(
                "W=4 unconditional-DMA DATA MISMATCH:\n  "
                + "\n  ".join(mismatches)
            )


# ---------------------------------------------------------------------------
# W=4 UNCONDITIONAL-DMA TIGHTENED multi-iter litmus (v60)
#
# Codex seq-0084 critique: v59 weakly passed but used identical offsets and
# payloads across waves, so a kernel that only ran wave-0 would also pass.
# Tightenings:
#   1) WAVE-DISTINCT payloads. Each (wave, src, dst) triple maps to a unique
#      sentinel ``0x0A000000 + w*0x10000 + src*0x100 + dst``.
#   2) WAVE-DISTINCT recv offsets. Wave 0 writes the first half of recv_buf;
#      wave 1 writes the second half. Skipping wave 1 leaves the second half
#      at zero (from the per-wave zero-fill).
#   3) PER-WAVE zero-fill of the wave's slot of recv_buf via ``pltpu.sync_copy``
#      (production-shape pre-fill). The W-way REGULAR-sem PRE-barrier fences
#      this against fast-peer inbound DMAs.
# Verifying the second half holds wave-1 sentinels on ALL 4 ranks proves both
# waves executed cleanly without ``@pl.when``.
# ---------------------------------------------------------------------------


_W4_TIGHT_TILE = 128
_W4_TIGHT_MAX_WAVES = 2
_W4_TIGHT_PER_PAIR_COUNT = 128  # one TILE per (src, dst) per wave
_W4_TIGHT_WORLD_SIZE = 4
# recv_buf is now full size: max_waves slots, each = W * per_pair_count.
_W4_TIGHT_PER_WAVE_RECV_SIZE = (
    _W4_TIGHT_WORLD_SIZE * _W4_TIGHT_PER_PAIR_COUNT
)  # 512 ints
_W4_TIGHT_RECV_CAPACITY = (
    _W4_TIGHT_MAX_WAVES * _W4_TIGHT_PER_WAVE_RECV_SIZE
)  # 1024 ints
# sorted_row buffer is also full size: max_waves slots of W*per_pair_count each.
_W4_TIGHT_MAX_PADDED = (
    _W4_TIGHT_MAX_WAVES * _W4_TIGHT_WORLD_SIZE * _W4_TIGHT_PER_PAIR_COUNT
)  # 1024 ints


def _w4_tight_sentinel(w: int, src: int, dst: int) -> int:
    """Wave/source/dest-distinct sentinel: 0x0A000000 + w*0x10000 + src*0x100 + dst."""
    return 0x0A000000 + (w << 16) + (src << 8) + dst


def _make_w4_tight_kernel(world_size: int, max_waves: int, per_pair_count: int):
    """Per-device kernel: lax.fori_loop over max_waves; per-iter:
      0. PER-WAVE zero-fill of the wave's slot of recv_buf via sync_copy.
      1. PRE-barrier (W-way REGULAR-sem fan-in) — fences zero-fill against
         peer inbound DMAs.
      2. UNCONDITIONAL per-peer DMA — wave-distinct sentinels into wave-distinct
         recv slots.
      3. POST-barrier (W-way REGULAR-sem fan-in).
    """
    per_wave_recv_size = world_size * per_pair_count

    def kernel(
        sorted_row_ref,    # [max_waves * W * per_pair_count] int32 HBM
        traffic_ref,       # [max_waves, W, W] int32 SMEM
        send_off_ref,      # [max_waves, W, W] int32 SMEM
        recv_off_ref,      # [max_waves, W, W] int32 SMEM
        out_recv_ref,      # [max_waves * W * per_pair_count] int32 HBM
        zero_int_vmem,     # [W * per_pair_count] int32 VMEM zero buffer
        barrier_sems,      # REGULAR (max_waves, 2) — pre/post per wave
        send_sem,          # DMA send sem
        recv_sem,          # DMA recv sem
    ):
        my_rank = lax.axis_index("x")

        # Initialize zero VMEM scratch once outside the loop.
        zero_int_vmem[...] = jnp.zeros_like(zero_int_vmem)

        def wave_body(w, _carry):
            # Phase 0: zero-fill THIS WAVE'S slot of recv_buf
            # (production-shape pre-fill).
            wave_recv_slot = out_recv_ref.at[
                pl.ds(w * per_wave_recv_size, per_wave_recv_size)
            ]
            pltpu.sync_copy(zero_int_vmem, wave_recv_slot)

            # Pre-barrier: signal every peer (incl. self), wait W signals.
            # Fences the zero-fill above against fast-peer inbound DMAs.
            pre_sem = barrier_sems.at[w, 0]
            for peer in range(world_size):
                pl.semaphore_signal(
                    pre_sem, 1,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
            pl.semaphore_wait(pre_sem, world_size)

            # Per-peer UNCONDITIONAL DMA — no @pl.when guard.
            for peer in range(world_size):
                send_count = pl.multiple_of(
                    traffic_ref[w, my_rank, peer], _W4_TIGHT_TILE
                )
                send_start = pl.multiple_of(
                    send_off_ref[w, my_rank, peer], _W4_TIGHT_TILE
                )
                remote_recv_start = pl.multiple_of(
                    recv_off_ref[w, peer, my_rank], _W4_TIGHT_TILE
                )

                d = pltpu.make_async_remote_copy(
                    src_ref=sorted_row_ref.at[pl.ds(send_start, send_count)],
                    dst_ref=out_recv_ref.at[
                        pl.ds(remote_recv_start, send_count)
                    ],
                    send_sem=send_sem, recv_sem=recv_sem,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
                d.start()
                d.wait()

            # Post-barrier: signal every peer (incl. self), wait W signals.
            post_sem = barrier_sems.at[w, 1]
            for peer in range(world_size):
                pl.semaphore_signal(
                    post_sem, 1,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
            pl.semaphore_wait(post_sem, world_size)
            return None

        lax.fori_loop(0, max_waves, wave_body, None)

    return kernel


def _run_w4_tight_litmus():
    """Run W=4 unconditional-DMA TIGHTENED multi-iter litmus.

    Returns ``(recv_per_rank, max_waves, per_pair_count, per_wave_recv_size)``.
    ``recv_per_rank``: ``[W, recv_capacity]`` int32 — full recv_buf on each
    rank. After both waves complete, on rank ``my_rank`` the contents should be:
      - First half (wave-0 slot): for each src ``s`` at offset
        ``s * per_pair_count``, value ``_w4_tight_sentinel(0, s, my_rank)``
        repeated per_pair_count times.
      - Second half (wave-1 slot): for each src ``s`` at offset
        ``per_wave_recv_size + s * per_pair_count``, value
        ``_w4_tight_sentinel(1, s, my_rank)`` repeated per_pair_count times.
    """
    world_size = _W4_TIGHT_WORLD_SIZE
    max_waves = _W4_TIGHT_MAX_WAVES
    per_pair_count = _W4_TIGHT_PER_PAIR_COUNT
    per_wave_recv_size = _W4_TIGHT_PER_WAVE_RECV_SIZE
    devices = jax.devices()[:world_size]
    mesh = jax.sharding.Mesh(np.array(devices), ("x",))

    # Balanced traffic: every (s, t) sends exactly per_pair_count every wave.
    per_wave_traffic_np = np.full(
        (max_waves, world_size, world_size),
        per_pair_count, dtype=np.int32,
    )
    # Send offsets [w, s, t] = w * W * per_pair_count + t * per_pair_count.
    # Recv offsets [w, t, s] = w * W * per_pair_count + s * per_pair_count.
    # Wave 0 uses [0, per_wave_recv_size); wave 1 uses [per_wave_recv_size, 2*per_wave_recv_size).
    per_wave_send_offsets_np = np.zeros(
        (max_waves, world_size, world_size), dtype=np.int32,
    )
    per_wave_recv_offsets_np = np.zeros(
        (max_waves, world_size, world_size), dtype=np.int32,
    )
    for w in range(max_waves):
        for s in range(world_size):
            for t in range(world_size):
                per_wave_send_offsets_np[w, s, t] = (
                    w * per_wave_recv_size + t * per_pair_count
                )
                per_wave_recv_offsets_np[w, t, s] = (
                    w * per_wave_recv_size + s * per_pair_count
                )

    # Per-rank sorted_row: for each (wave w, dest peer p), fill the per-pair
    # slot with the wave/src/dest-distinct sentinel.
    sorted_row_padded_np = np.zeros(
        (world_size, _W4_TIGHT_MAX_PADDED), dtype=np.int32,
    )
    for r in range(world_size):
        for w in range(max_waves):
            for p in range(world_size):
                start = w * per_wave_recv_size + p * per_pair_count
                end = start + per_pair_count
                sorted_row_padded_np[r, start:end] = _w4_tight_sentinel(w, r, p)

    HBM = pltpu.MemorySpace.HBM
    SMEM = pltpu.MemorySpace.SMEM

    kernel = _make_w4_tight_kernel(
        world_size=world_size, max_waves=max_waves,
        per_pair_count=per_pair_count,
    )

    P = jax.sharding.PartitionSpec

    def per_device(src, traffic, send_off, recv_off):
        recv = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(
                (_W4_TIGHT_RECV_CAPACITY,), jnp.int32,
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[
                    pl.BlockSpec(memory_space=HBM),   # sorted_row
                    pl.BlockSpec(memory_space=SMEM),  # traffic
                    pl.BlockSpec(memory_space=SMEM),  # send_off
                    pl.BlockSpec(memory_space=SMEM),  # recv_off
                ],
                out_specs=pl.BlockSpec(memory_space=HBM),
                scratch_shapes=(
                    pltpu.VMEM((per_wave_recv_size,), jnp.int32),
                    pltpu.SemaphoreType.REGULAR((max_waves, 2)),
                    pltpu.SemaphoreType.DMA,
                    pltpu.SemaphoreType.DMA,
                ),
            ),
            interpret=False,
        )(src[0], traffic, send_off, recv_off)
        return recv[None, ...]

    sharded = jax.shard_map(
        per_device, mesh=mesh,
        in_specs=(P("x"), P(), P(), P()),
        out_specs=P("x"),
        check_vma=False,
    )

    sharding_x = jax.sharding.NamedSharding(mesh, P("x"))
    sharding_rep = jax.sharding.NamedSharding(mesh, P())
    src = jax.device_put(jnp.array(sorted_row_padded_np), sharding_x)
    traffic = jax.device_put(jnp.array(per_wave_traffic_np), sharding_rep)
    send_off = jax.device_put(
        jnp.array(per_wave_send_offsets_np), sharding_rep,
    )
    recv_off = jax.device_put(
        jnp.array(per_wave_recv_offsets_np), sharding_rep,
    )

    out = sharded(src, traffic, send_off, recv_off)
    return (
        np.array(out),  # [W, recv_capacity]
        max_waves,
        per_pair_count,
        per_wave_recv_size,
    )


@unittest.skipUnless(_has_tpu(), "Litmus test requires real TPU")
class W4UnconditionalDMATightenedLitmus(unittest.TestCase):
    """W=4 multi-iter REGULAR-sem barriers + UNCONDITIONAL DMAs (no @pl.when)
    with WAVE-DISTINCT payloads, WAVE-DISTINCT recv offsets, and PER-WAVE
    zero-fill. Codex seq-0084 tightening of v59.
    """

    def setUp(self) -> None:
        if jax.device_count() < 4:
            self.skipTest(
                f"Need 4 devices, have {jax.device_count()}"
            )

    def test_w4_uncond_dma_wave_distinct(self) -> None:
        """W=4 unconditional-DMA TIGHTENED: per-rank recv_buf must hold
        wave-0 sentinels in the first half AND wave-1 sentinels in the second
        half. If the second half is all zero, wave-1 didn't execute.

        300s host-fetch timeout via threading; if the kernel hangs, the
        executor thread will not return and the test will fail cleanly.
        """
        import threading

        result_box: list = []
        error_box: list = []

        def runner():
            try:
                result_box.append(_run_w4_tight_litmus())
            except BaseException as e:  # noqa: BLE001
                error_box.append(e)

        t = threading.Thread(target=runner, daemon=True)
        t.start()
        t.join(timeout=300.0)
        if t.is_alive():
            self.fail(
                "W=4 TIGHTENED unconditional-DMA litmus HUNG after 300s; "
                "barrier x DMA x W=4 x multi-iter deadlocks even WITHOUT "
                "@pl.when. Padding workaround does NOT save us — pure "
                "upstream blocker."
            )
        if error_box:
            raise error_box[0]

        recv, max_waves, per_pair, per_wave_recv_size = result_box[0]

        # Print first 4 ints of each half on each rank.
        print(
            f"[LITMUS_W4_TIGHT] max_waves={max_waves} "
            f"per_pair={per_pair} per_wave_recv_size={per_wave_recv_size}",
            flush=True,
        )
        for r in range(_W4_TIGHT_WORLD_SIZE):
            first_half_first4 = recv[r, :4].tolist()
            second_half_first4 = recv[
                r, per_wave_recv_size:per_wave_recv_size + 4
            ].tolist()
            print(
                f"[LITMUS_W4_TIGHT] rank {r} "
                f"wave0_slot[:4]={[hex(int(x)) for x in first_half_first4]} "
                f"wave1_slot[:4]={[hex(int(x)) for x in second_half_first4]}",
                flush=True,
            )

        # Per-rank check: for each wave w and source s, at offset
        # w * per_wave_recv_size + s * per_pair, we expect
        # _w4_tight_sentinel(w, s, r) repeated per_pair times.
        mismatches = []
        for r in range(_W4_TIGHT_WORLD_SIZE):
            for w in range(max_waves):
                for s in range(_W4_TIGHT_WORLD_SIZE):
                    offset = w * per_wave_recv_size + s * per_pair
                    expected = _w4_tight_sentinel(w, s, r)
                    slot = recv[r, offset:offset + per_pair]
                    if not np.all(slot == expected):
                        actual_first8 = slot[:8].tolist()
                        mismatches.append(
                            f"rank={r} wave={w} src={s} off={offset} "
                            f"expected={hex(expected)} actual_first8="
                            f"{[hex(int(x)) for x in actual_first8]}"
                        )
        if mismatches:
            self.fail(
                "W=4 tightened unconditional-DMA DATA MISMATCH "
                "(if all wave=1 slots are 0x0, kernel skipped wave 1; "
                "v59 PASS was weak):\n  "
                + "\n  ".join(mismatches)
            )


# ---------------------------------------------------------------------------
# v63: W=4 phase-1 + phase-2 ISOLATING litmus.
#
# Same as v60 (W=4, REGULAR pre/post barriers, UNCONDITIONAL DMAs, balanced
# 128-per-pair traffic, wave-distinct payloads, per-wave zero-fill) BUT also
# adds phase 2: an in-VMEM scalar gather over the recv_row_buf, writing to
# looked_up_buf. NO phase 3, NO skewed traffic, NO dummy-offset rewrite. The
# goal is to isolate whether phase 2's nested `lax.fori_loop` inside the
# outer wave loop is the culprit for v62's hang.
#
# Stop conditions:
#   PASS  -> phase 2 isn't the culprit; spawn v64 (phase 3 only) next.
#   HANG  -> phase 2 (nested fori_loop with cross-device DMAs around it) IS
#            the culprit; isolated minimal repro found.
#   WRONG -> unlikely; one debug iter only.
# ---------------------------------------------------------------------------


_W4_P12_TILE = 128
_W4_P12_MAX_WAVES = 2
_W4_P12_PER_PAIR_COUNT = 128
_W4_P12_WORLD_SIZE = 4
_W4_P12_PER_WAVE_RECV_SIZE = (
    _W4_P12_WORLD_SIZE * _W4_P12_PER_PAIR_COUNT
)  # 512 ints
_W4_P12_RECV_CAPACITY = (
    _W4_P12_MAX_WAVES * _W4_P12_PER_WAVE_RECV_SIZE
)  # 1024 ints
_W4_P12_MAX_PADDED = (
    _W4_P12_MAX_WAVES * _W4_P12_WORLD_SIZE * _W4_P12_PER_PAIR_COUNT
)  # 1024 ints
_W4_P12_V_LOCAL = 128
_W4_P12_EMB_DIM = 8


def _w4_p12_sentinel(w: int, src: int, dst: int) -> int:
    """Wave/source/dest-distinct sentinel: same form as v60."""
    return 0x0A000000 + (w << 16) + (src << 8) + dst


def _make_w4_p12_kernel(world_size: int, max_waves: int, per_pair_count: int,
                        v_local: int, embedding_dim: int):
    """Per-device kernel: v60 body + phase-2 in-VMEM gather.

    Per wave:
      0. Per-wave zero-fill of THIS wave's slot of recv_buf via sync_copy.
      1. PRE-barrier (W-way REGULAR-sem fan-in).
      2. UNCONDITIONAL per-peer DMA (phase 1).
      3. POST-barrier (W-way REGULAR-sem fan-in).
      4. Phase 2 (NEW): bring recv_row_buf into SMEM, then nested fori_loop
         row-at-a-time gather from table_vmem into zero_emb_vmem, then
         sync_copy gathered tile to out_lookup_ref.
    NO phase 3.
    """
    per_wave_recv_size = world_size * per_pair_count

    def kernel(
        sorted_row_ref,    # [max_waves * W * per_pair_count] int32 HBM
        table_ref,         # [v_local, embedding_dim] f32 HBM
        traffic_ref,       # [max_waves, W, W] int32 SMEM
        send_off_ref,      # [max_waves, W, W] int32 SMEM
        recv_off_ref,      # [max_waves, W, W] int32 SMEM
        out_recv_ref,      # [max_waves * W * per_pair_count] int32 HBM
        out_lookup_ref,    # [max_waves * W * per_pair_count, D] f32 HBM
        zero_int_vmem,     # [W * per_pair_count] int32 VMEM
        zero_emb_vmem,     # [W * per_pair_count, D] f32 VMEM
        table_vmem,        # [v_local, D] f32 VMEM
        idx_smem,          # [W * per_pair_count] int32 SMEM
        barrier_sems,      # REGULAR (max_waves, 2)
        send_sem,          # DMA send sem
        recv_sem,          # DMA recv sem
    ):
        my_rank = lax.axis_index("x")

        # One-time table copy HBM -> VMEM (for the gather).
        pltpu.sync_copy(table_ref, table_vmem)

        # Initialize zero VMEM scratches once outside the loop.
        zero_int_vmem[...] = jnp.zeros_like(zero_int_vmem)
        zero_emb_vmem[...] = jnp.zeros_like(zero_emb_vmem)

        def wave_body(w, _carry):
            # Phase 0: zero-fill THIS WAVE'S slot of recv_buf.
            wave_recv_slot = out_recv_ref.at[
                pl.ds(w * per_wave_recv_size, per_wave_recv_size)
            ]
            pltpu.sync_copy(zero_int_vmem, wave_recv_slot)

            # Pre-barrier.
            pre_sem = barrier_sems.at[w, 0]
            for peer in range(world_size):
                pl.semaphore_signal(
                    pre_sem, 1,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
            pl.semaphore_wait(pre_sem, world_size)

            # Phase 1: UNCONDITIONAL per-peer DMA.
            for peer in range(world_size):
                send_count = pl.multiple_of(
                    traffic_ref[w, my_rank, peer], _W4_P12_TILE
                )
                send_start = pl.multiple_of(
                    send_off_ref[w, my_rank, peer], _W4_P12_TILE
                )
                remote_recv_start = pl.multiple_of(
                    recv_off_ref[w, peer, my_rank], _W4_P12_TILE
                )

                d = pltpu.make_async_remote_copy(
                    src_ref=sorted_row_ref.at[pl.ds(send_start, send_count)],
                    dst_ref=out_recv_ref.at[
                        pl.ds(remote_recv_start, send_count)
                    ],
                    send_sem=send_sem, recv_sem=recv_sem,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
                d.start()
                d.wait()

            # Post-barrier.
            post_sem = barrier_sems.at[w, 1]
            for peer in range(world_size):
                pl.semaphore_signal(
                    post_sem, 1,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
            pl.semaphore_wait(post_sem, world_size)

            # === NEW v63 addition: phase 2 (in-VMEM gather) ===
            # Bring this wave's recv slot into SMEM.
            pltpu.sync_copy(
                out_recv_ref.at[
                    pl.ds(w * per_wave_recv_size, per_wave_recv_size)
                ],
                idx_smem,
            )

            # Nested fori_loop: row-at-a-time gather from table_vmem.
            # Clamp index to V_local-1 because sentinel ids may exceed v_local.
            def gather_body(i, __):
                row_idx = idx_smem[i]
                clamped = jnp.minimum(row_idx, v_local - 1)
                zero_emb_vmem[pl.ds(i, 1), :] = table_vmem[
                    pl.ds(clamped, 1), :
                ]
                return None

            lax.fori_loop(0, per_wave_recv_size, gather_body, None)

            # Write gathered tile to HBM so host can verify.
            pltpu.sync_copy(
                zero_emb_vmem,
                out_lookup_ref.at[
                    pl.ds(w * per_wave_recv_size, per_wave_recv_size), :
                ],
            )

            # NO phase 3.
            return None

        lax.fori_loop(0, max_waves, wave_body, None)

    return kernel


def _run_w4_p12_litmus():
    """Run W=4 phase-1+phase-2 litmus.

    Returns ``(recv_per_rank, lookup_per_rank, max_waves, per_pair_count,
    per_wave_recv_size)``.
      ``recv_per_rank``:    [W, recv_capacity]              int32
      ``lookup_per_rank``:  [W, recv_capacity, embedding_dim] f32
    """
    world_size = _W4_P12_WORLD_SIZE
    max_waves = _W4_P12_MAX_WAVES
    per_pair_count = _W4_P12_PER_PAIR_COUNT
    per_wave_recv_size = _W4_P12_PER_WAVE_RECV_SIZE
    v_local = _W4_P12_V_LOCAL
    embedding_dim = _W4_P12_EMB_DIM
    devices = jax.devices()[:world_size]
    mesh = jax.sharding.Mesh(np.array(devices), ("x",))

    # Identical to v60: balanced 128-per-pair traffic.
    per_wave_traffic_np = np.full(
        (max_waves, world_size, world_size),
        per_pair_count, dtype=np.int32,
    )
    per_wave_send_offsets_np = np.zeros(
        (max_waves, world_size, world_size), dtype=np.int32,
    )
    per_wave_recv_offsets_np = np.zeros(
        (max_waves, world_size, world_size), dtype=np.int32,
    )
    for w in range(max_waves):
        for s in range(world_size):
            for t in range(world_size):
                per_wave_send_offsets_np[w, s, t] = (
                    w * per_wave_recv_size + t * per_pair_count
                )
                per_wave_recv_offsets_np[w, t, s] = (
                    w * per_wave_recv_size + s * per_pair_count
                )

    # Identical to v60: wave/src/dest-distinct sentinels.
    sorted_row_padded_np = np.zeros(
        (world_size, _W4_P12_MAX_PADDED), dtype=np.int32,
    )
    for r in range(world_size):
        for w in range(max_waves):
            for p in range(world_size):
                start = w * per_wave_recv_size + p * per_pair_count
                end = start + per_pair_count
                sorted_row_padded_np[r, start:end] = _w4_p12_sentinel(w, r, p)

    # Per-rank table: distinct nonzero values so the gather output is non-zero
    # regardless of which rows are picked. Row j on rank r holds value
    # 1.0 + r + j * 0.001 across all D columns.
    table_np = np.zeros(
        (world_size, v_local, embedding_dim), dtype=np.float32,
    )
    for r in range(world_size):
        for j in range(v_local):
            table_np[r, j, :] = 1.0 + r + j * 0.001

    HBM = pltpu.MemorySpace.HBM
    SMEM = pltpu.MemorySpace.SMEM

    kernel = _make_w4_p12_kernel(
        world_size=world_size, max_waves=max_waves,
        per_pair_count=per_pair_count,
        v_local=v_local, embedding_dim=embedding_dim,
    )

    P = jax.sharding.PartitionSpec

    def per_device(src, table, traffic, send_off, recv_off):
        recv, lookup = pl.pallas_call(
            kernel,
            out_shape=[
                jax.ShapeDtypeStruct(
                    (_W4_P12_RECV_CAPACITY,), jnp.int32,
                ),
                jax.ShapeDtypeStruct(
                    (_W4_P12_RECV_CAPACITY, embedding_dim), jnp.float32,
                ),
            ],
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[
                    pl.BlockSpec(memory_space=HBM),   # sorted_row
                    pl.BlockSpec(memory_space=HBM),   # table
                    pl.BlockSpec(memory_space=SMEM),  # traffic
                    pl.BlockSpec(memory_space=SMEM),  # send_off
                    pl.BlockSpec(memory_space=SMEM),  # recv_off
                ],
                out_specs=[
                    pl.BlockSpec(memory_space=HBM),   # out_recv
                    pl.BlockSpec(memory_space=HBM),   # out_lookup
                ],
                scratch_shapes=(
                    pltpu.VMEM((per_wave_recv_size,), jnp.int32),
                    pltpu.VMEM(
                        (per_wave_recv_size, embedding_dim), jnp.float32,
                    ),
                    pltpu.VMEM((v_local, embedding_dim), jnp.float32),
                    SMEM((per_wave_recv_size,), jnp.int32),
                    pltpu.SemaphoreType.REGULAR((max_waves, 2)),
                    pltpu.SemaphoreType.DMA,
                    pltpu.SemaphoreType.DMA,
                ),
            ),
            interpret=False,
        )(src[0], table[0], traffic, send_off, recv_off)
        return recv[None, ...], lookup[None, ...]

    sharded = jax.shard_map(
        per_device, mesh=mesh,
        in_specs=(P("x"), P("x"), P(), P(), P()),
        out_specs=(P("x"), P("x")),
        check_vma=False,
    )

    sharding_x = jax.sharding.NamedSharding(mesh, P("x"))
    sharding_rep = jax.sharding.NamedSharding(mesh, P())
    src = jax.device_put(jnp.array(sorted_row_padded_np), sharding_x)
    table = jax.device_put(jnp.array(table_np), sharding_x)
    traffic = jax.device_put(jnp.array(per_wave_traffic_np), sharding_rep)
    send_off = jax.device_put(
        jnp.array(per_wave_send_offsets_np), sharding_rep,
    )
    recv_off = jax.device_put(
        jnp.array(per_wave_recv_offsets_np), sharding_rep,
    )

    recv_arr, lookup_arr = sharded(src, table, traffic, send_off, recv_off)
    return (
        np.array(recv_arr),    # [W, recv_capacity]
        np.array(lookup_arr),  # [W, recv_capacity, embedding_dim]
        max_waves,
        per_pair_count,
        per_wave_recv_size,
    )


@unittest.skipUnless(_has_tpu(), "Litmus test requires real TPU")
class W4Phase12Litmus(unittest.TestCase):
    """v63: W=4 multi-iter REGULAR-sem barriers + UNCONDITIONAL DMAs (phase 1)
    + nested-fori_loop in-VMEM gather (phase 2). NO phase 3, NO skewed traffic,
    NO dummy-offset rewrite. Isolates whether phase 2's nested ``lax.fori_loop``
    inside the outer wave loop is the culprit for v62's hang.
    """

    def setUp(self) -> None:
        if jax.device_count() < 4:
            self.skipTest(
                f"Need 4 devices, have {jax.device_count()}"
            )

    def test_w4_p1_p2_no_p3(self) -> None:
        """v63 isolating litmus.

        300s host-fetch timeout via threading; if the kernel hangs the executor
        thread will not return and the test fails cleanly.
        """
        import threading

        result_box: list = []
        error_box: list = []

        def runner():
            try:
                result_box.append(_run_w4_p12_litmus())
            except BaseException as e:  # noqa: BLE001
                error_box.append(e)

        t = threading.Thread(target=runner, daemon=True)
        t.start()
        t.join(timeout=300.0)
        if t.is_alive():
            self.fail(
                "v63 W=4 phase-1+phase-2 litmus HUNG after 300s; "
                "phase 2's nested `lax.fori_loop` inside the outer wave loop "
                "with cross-device DMAs IS the culprit for v62's hang."
            )
        if error_box:
            raise error_box[0]

        recv, lookup, max_waves, per_pair, per_wave_recv_size = result_box[0]

        print(
            f"[LITMUS_W4_P12] max_waves={max_waves} "
            f"per_pair={per_pair} per_wave_recv_size={per_wave_recv_size}",
            flush=True,
        )
        for r in range(_W4_P12_WORLD_SIZE):
            first_half_first4 = recv[r, :4].tolist()
            second_half_first4 = recv[
                r, per_wave_recv_size:per_wave_recv_size + 4
            ].tolist()
            print(
                f"[LITMUS_W4_P12] rank {r} "
                f"wave0_recv[:4]={[hex(int(x)) for x in first_half_first4]} "
                f"wave1_recv[:4]={[hex(int(x)) for x in second_half_first4]}",
                flush=True,
            )
            print(
                f"[LITMUS_W4_P12] rank {r} "
                f"wave0_lookup[0,:4]={lookup[r, 0, :4].tolist()} "
                f"wave1_lookup[0,:4]="
                f"{lookup[r, per_wave_recv_size, :4].tolist()}",
                flush=True,
            )

        # Check 1: phase-1 sentinels (same as v60).
        mismatches = []
        for r in range(_W4_P12_WORLD_SIZE):
            for w in range(max_waves):
                for s in range(_W4_P12_WORLD_SIZE):
                    offset = w * per_wave_recv_size + s * per_pair
                    expected = _w4_p12_sentinel(w, s, r)
                    slot = recv[r, offset:offset + per_pair]
                    if not np.all(slot == expected):
                        actual_first8 = slot[:8].tolist()
                        mismatches.append(
                            f"rank={r} wave={w} src={s} off={offset} "
                            f"expected={hex(expected)} actual_first8="
                            f"{[hex(int(x)) for x in actual_first8]}"
                        )
        if mismatches:
            self.fail(
                "v63 phase-1 DATA MISMATCH (proves phase 2 broke phase 1):"
                "\n  " + "\n  ".join(mismatches)
            )

        # Check 2: phase-2 lookup output is non-zero on every rank
        # (proves phase 2 actually executed).
        zero_ranks = []
        for r in range(_W4_P12_WORLD_SIZE):
            if not np.any(lookup[r] != 0.0):
                zero_ranks.append(r)
        if zero_ranks:
            self.fail(
                f"v63 phase-2 NEVER EXECUTED on ranks {zero_ranks}: "
                f"lookup output is all-zero (phase-2 in-VMEM gather did not "
                f"run). This means the fori_loop body skipped or its sync_copy "
                f"never landed."
            )


# ---------------------------------------------------------------------------
# W=4 FULL-SHAPE no-@pl.when litmus (v62, Codex seq-0100 gate)
#
# Mirrors the production `_make_alltc_fused_f32_kernel`'s full 3-phase
# structure (phase 1 ID a2a, phase 2 in-VMEM gather, phase 3 emb a2a back),
# at W=4 and 2 waves, with the REAL skewed-overflow traffic matrices that
# include zero-count entries. The C-v2 dummy-offset rewrite redirects every
# zero-count edge to a wave-distinct dummy region so DMAs are issued
# UNCONDITIONALLY (no `@pl.when`). Expected: real region of out_ref matches
# the numpy reference within float32 tolerance on all 4 ranks.
#
# Codex seq-0100 stop conditions:
#   PASS  -> approve C-v2; promote to production fused kernel.
#   HANG  -> phase 2 + phase 3 add a deeper structural blocker not visible
#            in v60; reject C-v2; keep fused all-TC experimental.
#   WRONG -> effective-offset math has a bug; one debug iter only.
# ---------------------------------------------------------------------------


_FULL_TILE = 128
_FULL_VLOCAL = 256
_FULL_EMB_DIM = 128
_FULL_MAX_LOCAL_IDS = 4 * _FULL_TILE  # 512
_FULL_RECV_CAPACITY = 512
_FULL_SEED = 2  # MATCHES the failing-case skewed-overflow inputs.


def _full_compute_effective_arrays(
    per_wave_traffic: np.ndarray,
    per_wave_send_offsets: np.ndarray,
    per_wave_recv_offsets: np.ndarray,
    *,
    max_padded_real: int,
    recv_capacity: int,
    world_size: int,
    tile: int,
):
    """Build the C-v2 effective offset / count matrices.

    All four matrices are indexed [w, A, B] consistently. The kernel reads
    them via [w, my_rank, peer] for src and [w, peer, my_rank] for dst, in
    BOTH phase 1 and phase 3 (mirroring the production access pattern).

    For each (w, A, B):
      - eff_count[w, A, B] = max(traffic[w, A, B], TILE). When indexed
        [w, my_rank, peer] (phase 1) gives count for FWD A->B; when indexed
        [w, peer, my_rank] (phase 3) gives count for REV B->A's phase-1
        traffic (which is the count being sent BACK in phase 3).

      - eff_p1_src_off[w, A, B] (kernel: [w, my_rank, peer]):
          if traffic[A, B] > 0: send_offsets[A, B]   (real)
          else:                  max_padded_real + w*W*TILE + B*TILE  (dummy)

      - eff_p1_dst_off[w, A, B] (kernel: [w, peer, my_rank]):
          When indexed [w, peer=A, my_rank=B], the transfer is from B to A.
          if traffic[B, A] > 0: recv_offsets[A, B]   (real, recv at A from B)
          else:                  recv_capacity + B*TILE  (dummy slot for src B
                                                          on receiver A's recv)

      - eff_p3_src_off[w, A, B] (kernel: [w, my_rank, peer]):
          When indexed [w, my_rank=A, peer=B], the transfer in P3 is A->B
          with count = traffic[B, A]. Source is in MY (rank A's)
          looked_up_buf at the offset where peer B's data landed in P1.
          if traffic[B, A] > 0: recv_offsets[A, B]   (real)
          else:                  recv_capacity + B*TILE  (dummy slot for peer B
                                                          in MY looked_up_buf)

      - eff_p3_dst_off[w, A, B] (kernel: [w, peer, my_rank]):
          When indexed [w, peer=A, my_rank=B], the transfer in P3 is B->A
          with count = traffic[A, B]. Destination is on peer A's out_ref at
          the offset where A originally extracted data destined for B in P1.
          if traffic[A, B] > 0: send_offsets[A, B]   (real, on peer A's
                                                       out_ref at A's
                                                       send_offsets[A,B])
          else:                  max_padded_real + w*W*TILE + B*TILE  (dummy
                                                       on peer A's out_ref)
    """
    max_waves = per_wave_traffic.shape[0]
    eff_count = np.zeros((max_waves, world_size, world_size), dtype=np.int32)
    eff_p1_src_off = np.zeros_like(eff_count)
    eff_p1_dst_off = np.zeros_like(eff_count)
    eff_p3_src_off = np.zeros_like(eff_count)
    eff_p3_dst_off = np.zeros_like(eff_count)
    for w in range(max_waves):
        wave_dummy_base = max_padded_real + w * world_size * tile
        for A in range(world_size):
            for B in range(world_size):
                fwd = int(per_wave_traffic[w, A, B])  # A -> B in phase 1
                rev = int(per_wave_traffic[w, B, A])  # B -> A in phase 1

                # eff_count: max with TILE so DMAs always >= TILE bytes.
                eff_count[w, A, B] = max(fwd, tile)

                # phase-1 src (A is sender, B is receiver):
                if fwd > 0:
                    eff_p1_src_off[w, A, B] = int(per_wave_send_offsets[w, A, B])
                else:
                    eff_p1_src_off[w, A, B] = wave_dummy_base + B * tile

                # phase-1 dst (kernel indexes [w, peer=A, my_rank=B] => transfer
                # is from B to A, so the "real" condition is on traffic[B, A]):
                if rev > 0:
                    eff_p1_dst_off[w, A, B] = int(per_wave_recv_offsets[w, A, B])
                else:
                    eff_p1_dst_off[w, A, B] = recv_capacity + B * tile

                # phase-3 src (A is sender, B is peer/receiver in P3, count=
                # traffic[B, A] = REV — source is in A's looked_up at the
                # offset where peer B's data landed in P1):
                if rev > 0:
                    eff_p3_src_off[w, A, B] = int(per_wave_recv_offsets[w, A, B])
                else:
                    eff_p3_src_off[w, A, B] = recv_capacity + B * tile

                # phase-3 dst (kernel indexes [w, peer=A, my_rank=B] => transfer
                # is from B to A, count = traffic[A, B] = FWD; dest on A's
                # out_ref at the offset A originally extracted data for B):
                if fwd > 0:
                    eff_p3_dst_off[w, A, B] = int(per_wave_send_offsets[w, A, B])
                else:
                    eff_p3_dst_off[w, A, B] = wave_dummy_base + B * tile
    return eff_count, eff_p1_src_off, eff_p1_dst_off, eff_p3_src_off, eff_p3_dst_off


def _make_w4_full_kernel(*, world_size: int, max_waves: int, recv_capacity: int,
                         max_padded_real: int, embedding_dim: int, v_local: int,
                         dummy_region_size: int, tile: int):
    """Per-device kernel: full 3-phase fused with UNCONDITIONAL DMAs via the
    C-v2 dummy-offset rewrite. Mirrors `_make_alltc_fused_f32_kernel`'s
    structure but reads effective offsets/counts in place of raw traffic.
    """
    sorted_row_combined_size = max_padded_real + dummy_region_size
    looked_up_combined_rows = recv_capacity + world_size * tile
    out_combined_rows = max_padded_real + dummy_region_size

    def kernel(
        sorted_row_combined_ref,  # [sorted_row_combined_size] int32 HBM
        table_ref,                # [v_local, D] f32 HBM
        eff_count_ref,            # [max_waves, W, W] int32 SMEM
        eff_p1_src_off_ref,       # [max_waves, W, W] int32 SMEM
        eff_p1_dst_off_ref,       # [max_waves, W, W] int32 SMEM
        eff_p3_src_off_ref,       # [max_waves, W, W] int32 SMEM
        eff_p3_dst_off_ref,       # [max_waves, W, W] int32 SMEM
        # Outputs
        out_ref_combined,         # [out_combined_rows, D] f32 HBM
        recv_row_buf_combined,    # [looked_up_combined_rows] int32 HBM (scratch via output)
        looked_up_buf_combined,   # [looked_up_combined_rows, D] f32 HBM (scratch via output)
        # Scratch (VMEM/SMEM)
        zero_int_vmem,            # [looked_up_combined_rows] int32 VMEM
        zero_emb_vmem,            # [looked_up_combined_rows, D] f32 VMEM
        table_vmem,               # [v_local, D] f32 VMEM
        idx_smem,                 # [looked_up_combined_rows] int32 SMEM
        barrier_sems,             # REGULAR (max_waves, 2)
        send_sem_id,
        recv_sem_id,
        send_sem_emb,
        recv_sem_emb,
    ):
        my_rank = lax.axis_index("x")

        # One-time table copy HBM -> VMEM.
        pltpu.sync_copy(table_ref, table_vmem)

        # Initialize zero VMEM scratches once.
        zero_int_vmem[...] = jnp.zeros_like(zero_int_vmem)
        zero_emb_vmem[...] = jnp.zeros_like(zero_emb_vmem)

        def wave_body(w, _carry):
            # Phase 0: zero-fill the FULL combined recv buffer (real +
            # dummy slots both). The dummy slots see writes from peers'
            # zero-count redirects too, so they need to be zeroed every
            # wave to keep behavior deterministic.
            pltpu.sync_copy(zero_int_vmem, recv_row_buf_combined)

            # Pre-barrier: fences zero-fill against fast-peer inbound DMAs.
            pre_sem = barrier_sems.at[w, 0]
            for peer in range(world_size):
                pl.semaphore_signal(
                    pre_sem, 1,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
            pl.semaphore_wait(pre_sem, world_size)

            # Phase 1: ID a2a — UNCONDITIONAL using effective offsets.
            for peer in range(world_size):
                cnt = pl.multiple_of(eff_count_ref[w, my_rank, peer], tile)
                src_off = pl.multiple_of(
                    eff_p1_src_off_ref[w, my_rank, peer], tile,
                )
                dst_off = pl.multiple_of(
                    eff_p1_dst_off_ref[w, peer, my_rank], tile,
                )
                d = pltpu.make_async_remote_copy(
                    src_ref=sorted_row_combined_ref.at[pl.ds(src_off, cnt)],
                    dst_ref=recv_row_buf_combined.at[pl.ds(dst_off, cnt)],
                    send_sem=send_sem_id, recv_sem=recv_sem_id,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
                d.start(); d.wait()

            # Post-barrier: receiver-side flush so phase-2 reads are valid.
            post_sem = barrier_sems.at[w, 1]
            for peer in range(world_size):
                pl.semaphore_signal(
                    post_sem, 1,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
            pl.semaphore_wait(post_sem, world_size)

            # Phase 2: in-VMEM gather. Pull recv_row_buf into SMEM, then
            # row-at-a-time gather from table_vmem. Defensive clamp because
            # dummy slots may contain 0 (zero-fill) — table[0] is always
            # defined so no OOB risk.
            pltpu.sync_copy(recv_row_buf_combined, idx_smem)

            def gather_body(i, __):
                row_idx = idx_smem[i]
                zero_emb_vmem[pl.ds(i, 1), :] = table_vmem[pl.ds(row_idx, 1), :]
                return None

            lax.fori_loop(0, looked_up_combined_rows, gather_body, None)
            pltpu.sync_copy(zero_emb_vmem, looked_up_buf_combined)

            # Phase 3: emb a2a back — UNCONDITIONAL using effective offsets.
            # NO post-emb barrier (same as production: out_ref slices are
            # wave-disjoint, so a slow peer in wave w-1 can safely accept
            # my wave-w emb DMA into a different out_ref slice).
            for peer in range(world_size):
                cnt = pl.multiple_of(eff_count_ref[w, peer, my_rank], tile)
                src_off = pl.multiple_of(
                    eff_p3_src_off_ref[w, my_rank, peer], tile,
                )
                dst_off = pl.multiple_of(
                    eff_p3_dst_off_ref[w, peer, my_rank], tile,
                )
                d = pltpu.make_async_remote_copy(
                    src_ref=looked_up_buf_combined.at[pl.ds(src_off, cnt), :],
                    dst_ref=out_ref_combined.at[pl.ds(dst_off, cnt), :],
                    send_sem=send_sem_emb, recv_sem=recv_sem_emb,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
                d.start(); d.wait()

            return None

        lax.fori_loop(0, max_waves, wave_body, None)

    return kernel


def _run_w4_full_litmus():
    """Run the W=4 full-shape no-@pl.when litmus.

    Returns ``(out_combined_per_rank, ref_out_np, sorted_row_padded_np,
    send_counts_real_per_rank, send_counts_padded_per_rank, max_waves,
    max_padded_real, dummy_region_size, per_wave_traffic_np,
    per_wave_send_offsets_np, per_wave_recv_offsets_np, local_id_count_np)``.
    """
    from recir.ops.pallas.embeddings.jagged_a2a_lookup import (
        _bucketize_and_sort,
        _max_waves_for_capacity,
        _pad_per_peer,
        _round_up_tile,
        _split_traffic_into_waves,
        jagged_a2a_embedding_lookup_reference,
    )
    from recir.ops.pallas.embeddings.tests.test_jagged_a2a_lookup import (
        _build_local_table,
        _make_skewed_inputs,
    )

    world_size = 4
    tile = _FULL_TILE
    max_local_ids = _FULL_MAX_LOCAL_IDS
    recv_capacity = _FULL_RECV_CAPACITY
    embedding_dim = _FULL_EMB_DIM
    v_local = _FULL_VLOCAL
    dtype = np.float32

    devices = jax.devices()[:world_size]
    mesh = jax.sharding.Mesh(np.array(devices), ("x",))

    # 1. Build the same skewed-overflow inputs as the failing test.
    local_table_np = _build_local_table(
        world_size, v_local, embedding_dim, dtype, seed=0,
    )
    local_ids_np, local_id_count_np = _make_skewed_inputs(
        world_size, v_local, max_local_ids, seed=_FULL_SEED,
    )

    # 2. Numpy reference for verification.
    ref_out_np = jagged_a2a_embedding_lookup_reference(
        local_ids_np, local_id_count_np,
        local_table_np.astype(np.float32), world_size,
    ).astype(dtype)

    # 3. Per-rank: bucketize + pad. Reuse the host-side helpers.
    max_padded_real = max_local_ids + world_size * tile
    sorted_row_padded_per_rank = []
    send_counts_real_per_rank = []
    send_counts_padded_per_rank = []
    reverse_perm_per_rank = []
    for r in range(world_size):
        sorted_row_real, send_counts_real, _sort_perm, reverse_perm = (
            _bucketize_and_sort(
                jnp.array(local_ids_np[r]),
                jnp.array(local_id_count_np[r]),
                max_local_ids, world_size,
            )
        )
        send_counts_padded = _round_up_tile(send_counts_real, tile)
        sorted_row_padded = _pad_per_peer(
            sorted_row_real, send_counts_real, send_counts_padded,
            max_local_ids, max_padded_real, world_size,
        )
        sorted_row_padded_per_rank.append(np.array(sorted_row_padded))
        send_counts_real_per_rank.append(np.array(send_counts_real))
        send_counts_padded_per_rank.append(np.array(send_counts_padded))
        reverse_perm_per_rank.append(np.array(reverse_perm))

    # 4. Traffic matrix + per-wave split (same as production).
    traffic_matrix = jnp.array(
        np.stack(send_counts_padded_per_rank), dtype=jnp.int32,
    )  # [W, W]
    max_waves = _max_waves_for_capacity(
        max_local_ids, world_size, recv_capacity,
    )
    per_wave_traffic, per_wave_send_offsets, per_wave_recv_offsets = (
        _split_traffic_into_waves(
            traffic_matrix,
            recv_capacity=recv_capacity,
            max_waves=max_waves,
            world_size=world_size,
        )
    )
    per_wave_traffic_np = np.array(per_wave_traffic)
    per_wave_send_offsets_np = np.array(per_wave_send_offsets)
    per_wave_recv_offsets_np = np.array(per_wave_recv_offsets)

    # 5. C-v2 effective offsets + counts (zero-count edges -> dummy region).
    dummy_region_size = max_waves * world_size * tile
    (eff_count_np, eff_p1_src_off_np, eff_p1_dst_off_np,
     eff_p3_src_off_np, eff_p3_dst_off_np) = _full_compute_effective_arrays(
        per_wave_traffic_np, per_wave_send_offsets_np, per_wave_recv_offsets_np,
        max_padded_real=max_padded_real,
        recv_capacity=recv_capacity,
        world_size=world_size,
        tile=tile,
    )

    # 6. Build per-rank combined buffers.
    sorted_row_combined_size = max_padded_real + dummy_region_size
    sorted_row_combined_np = np.zeros(
        (world_size, sorted_row_combined_size), dtype=np.int32,
    )
    for r in range(world_size):
        sorted_row_combined_np[r, :max_padded_real] = (
            sorted_row_padded_per_rank[r]
        )
        # Dummy region stays zero (looks up table[0] -- harmless).

    HBM = pltpu.MemorySpace.HBM
    SMEM = pltpu.MemorySpace.SMEM

    out_combined_rows = max_padded_real + dummy_region_size
    looked_up_combined_rows = recv_capacity + world_size * tile

    kernel = _make_w4_full_kernel(
        world_size=world_size, max_waves=max_waves,
        recv_capacity=recv_capacity, max_padded_real=max_padded_real,
        embedding_dim=embedding_dim, v_local=v_local,
        dummy_region_size=dummy_region_size, tile=tile,
    )

    P = jax.sharding.PartitionSpec

    def per_device(src_combined, table, eff_count, eff_p1_src, eff_p1_dst,
                   eff_p3_src, eff_p3_dst):
        out_combined, _recv_dummy, _looked_dummy = pl.pallas_call(
            kernel,
            out_shape=[
                jax.ShapeDtypeStruct(
                    (out_combined_rows, embedding_dim), jnp.float32,
                ),
                jax.ShapeDtypeStruct((looked_up_combined_rows,), jnp.int32),
                jax.ShapeDtypeStruct(
                    (looked_up_combined_rows, embedding_dim), jnp.float32,
                ),
            ],
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[
                    pl.BlockSpec(memory_space=HBM),   # sorted_row_combined
                    pl.BlockSpec(memory_space=HBM),   # table
                    pl.BlockSpec(memory_space=SMEM),  # eff_count
                    pl.BlockSpec(memory_space=SMEM),  # eff_p1_src_off
                    pl.BlockSpec(memory_space=SMEM),  # eff_p1_dst_off
                    pl.BlockSpec(memory_space=SMEM),  # eff_p3_src_off
                    pl.BlockSpec(memory_space=SMEM),  # eff_p3_dst_off
                ],
                out_specs=[
                    pl.BlockSpec(memory_space=HBM),   # out_combined
                    pl.BlockSpec(memory_space=HBM),   # recv_row_buf_combined
                    pl.BlockSpec(memory_space=HBM),   # looked_up_buf_combined
                ],
                scratch_shapes=(
                    pltpu.VMEM(
                        (looked_up_combined_rows,), jnp.int32,
                    ),
                    pltpu.VMEM(
                        (looked_up_combined_rows, embedding_dim), jnp.float32,
                    ),
                    pltpu.VMEM((v_local, embedding_dim), jnp.float32),
                    SMEM((looked_up_combined_rows,), jnp.int32),
                    pltpu.SemaphoreType.REGULAR((max_waves, 2)),
                    pltpu.SemaphoreType.DMA,
                    pltpu.SemaphoreType.DMA,
                    pltpu.SemaphoreType.DMA,
                    pltpu.SemaphoreType.DMA,
                ),
            ),
            interpret=False,
        )(src_combined[0], table[0], eff_count, eff_p1_src, eff_p1_dst,
          eff_p3_src, eff_p3_dst)
        del _recv_dummy, _looked_dummy
        return out_combined[None, ...]

    sharded = jax.shard_map(
        per_device, mesh=mesh,
        in_specs=(P("x"), P("x"), P(), P(), P(), P(), P()),
        out_specs=P("x"),
        check_vma=False,
    )

    sharding_x = jax.sharding.NamedSharding(mesh, P("x"))
    sharding_rep = jax.sharding.NamedSharding(mesh, P())
    src = jax.device_put(jnp.array(sorted_row_combined_np), sharding_x)
    table = jax.device_put(jnp.array(local_table_np), sharding_x)
    eff_count = jax.device_put(jnp.array(eff_count_np), sharding_rep)
    eff_p1_src = jax.device_put(jnp.array(eff_p1_src_off_np), sharding_rep)
    eff_p1_dst = jax.device_put(jnp.array(eff_p1_dst_off_np), sharding_rep)
    eff_p3_src = jax.device_put(jnp.array(eff_p3_src_off_np), sharding_rep)
    eff_p3_dst = jax.device_put(jnp.array(eff_p3_dst_off_np), sharding_rep)

    out_combined_arr = sharded(
        src, table, eff_count, eff_p1_src, eff_p1_dst, eff_p3_src, eff_p3_dst,
    )
    out_combined_per_rank = np.array(out_combined_arr)  # [W, out_combined_rows, D]

    return (
        out_combined_per_rank,
        ref_out_np,
        np.stack(sorted_row_padded_per_rank),
        np.stack(send_counts_real_per_rank),
        np.stack(send_counts_padded_per_rank),
        [np.array(rp) for rp in reverse_perm_per_rank],
        max_waves,
        max_padded_real,
        dummy_region_size,
        per_wave_traffic_np,
        per_wave_send_offsets_np,
        per_wave_recv_offsets_np,
        local_id_count_np,
    )


@unittest.skipUnless(_has_tpu(), "Litmus test requires real TPU")
class W4FullShapeUnconditionalLitmus(unittest.TestCase):
    """W=4 full-shape (phase 1 + phase 2 + phase 3) no-@pl.when litmus —
    Codex seq-0100 gating test for the C-v2 dummy-offset rewrite. Uses the
    REAL skewed-overflow traffic matrices (with zero-count entries) and the
    C-v2 effective-offset redirect to issue all DMAs UNCONDITIONALLY.
    """

    def setUp(self) -> None:
        if jax.device_count() < 4:
            self.skipTest(
                f"Need 4 devices, have {jax.device_count()}"
            )

    def test_w4_full_shape_no_pl_when(self) -> None:
        """W=4 full-shape no-`@pl.when`: real region of out_combined must
        match the numpy reference within float32 tolerance on all 4 ranks.

        300s host-fetch timeout via threading; if the kernel hangs the
        executor thread will not return and the test fails cleanly.
        """
        from recir.ops.pallas.embeddings.jagged_a2a_lookup import (
            _unpad_per_peer,
        )

        import threading

        result_box: list = []
        error_box: list = []

        def runner():
            try:
                result_box.append(_run_w4_full_litmus())
            except BaseException as e:  # noqa: BLE001
                error_box.append(e)

        t = threading.Thread(target=runner, daemon=True)
        t.start()
        t.join(timeout=300.0)
        if t.is_alive():
            self.fail(
                "W=4 FULL-SHAPE no-`@pl.when` litmus HUNG after 300s; "
                "phase 2 + phase 3 add a deeper structural blocker not "
                "visible in v60. C-v2 REJECTED; keep fused all-TC "
                "experimental."
            )
        if error_box:
            raise error_box[0]

        (out_combined_per_rank, ref_out_np, sorted_row_padded_per_rank,
         send_counts_real_per_rank, send_counts_padded_per_rank,
         reverse_perm_per_rank, max_waves, max_padded_real,
         dummy_region_size, per_wave_traffic_np,
         per_wave_send_offsets_np, per_wave_recv_offsets_np,
         local_id_count_np) = result_box[0]

        world_size = 4
        max_local_ids = _FULL_MAX_LOCAL_IDS

        print(
            f"[LITMUS_W4_FULL] max_waves={max_waves} "
            f"max_padded_real={max_padded_real} "
            f"dummy_region_size={dummy_region_size}",
            flush=True,
        )
        print(
            f"[LITMUS_W4_FULL] per_wave_traffic[0]=\n{per_wave_traffic_np[0]}",
            flush=True,
        )
        if max_waves >= 2:
            print(
                f"[LITMUS_W4_FULL] per_wave_traffic[1]=\n"
                f"{per_wave_traffic_np[1]}",
                flush=True,
            )

        # Strip dummy region: real out is the first max_padded_real rows.
        # Then unpad per peer + un-permute via reverse_perm to get the per-
        # rank, per-input-position output (matches the reference layout).
        out_unpermuted_per_rank = np.zeros(
            (world_size, max_local_ids, _FULL_EMB_DIM), dtype=np.float32,
        )
        for r in range(world_size):
            out_padded_r = out_combined_per_rank[r, :max_padded_real, :]
            out_real_r_jax = _unpad_per_peer(
                jnp.array(out_padded_r),
                jnp.array(send_counts_real_per_rank[r]),
                jnp.array(send_counts_padded_per_rank[r]),
                max_local_ids, max_padded_real, world_size,
            )
            out_real_r = np.array(out_real_r_jax)  # [max_local_ids, D]
            # Un-permute via reverse_perm to caller order.
            out_unpermuted_per_rank[r] = out_real_r[reverse_perm_per_rank[r]]

        # Per-rank verification.
        atol, rtol = 1e-6, 1e-6
        mismatches = []
        for r in range(world_size):
            n = int(local_id_count_np[r])
            try:
                np.testing.assert_allclose(
                    out_unpermuted_per_rank[r, :n],
                    ref_out_np[r, :n],
                    atol=atol, rtol=rtol,
                    err_msg=(
                        f"FULL-SHAPE rank {r}: kernel output does not match "
                        f"numpy reference on the first {n} valid rows"
                    ),
                )
            except AssertionError as e:
                # Compact diagnostic: count mismatched rows + first few.
                kernel_arr = out_unpermuted_per_rank[r, :n]
                ref_arr = ref_out_np[r, :n]
                row_match = np.all(
                    np.isclose(kernel_arr, ref_arr, atol=atol, rtol=rtol),
                    axis=1,
                )
                num_bad = int(np.sum(~row_match))
                bad_idx = np.where(~row_match)[0][:5].tolist()
                mismatches.append(
                    f"rank {r}: {num_bad}/{n} rows mismatch; first bad "
                    f"indices: {bad_idx}; first bad row diff (kernel vs "
                    f"ref @ idx={bad_idx[0] if bad_idx else 0}): "
                    f"kernel[:4]={kernel_arr[bad_idx[0] if bad_idx else 0, :4].tolist()}, "
                    f"ref[:4]={ref_arr[bad_idx[0] if bad_idx else 0, :4].tolist()}"
                )

        if mismatches:
            self.fail(
                "W=4 FULL-SHAPE no-`@pl.when` DATA MISMATCH (effective-"
                "offset math has a bug; one debug iter only):\n  "
                + "\n  ".join(mismatches)
            )

        # Wave-distinctness verification: confirm wave 1 actually contributed
        # by finding a (sender, receiver) pair where wave 1 has count > 0
        # AND wave 0 also has count > 0 (so the per-(s,t) data is split
        # across BOTH waves) — verify the wave-1 portion is present in the
        # final output. If wave 1 silently skipped, the corresponding rows
        # would be missing/zero and the assertion above would have caught it,
        # but we also print a confirmation.
        if max_waves >= 2:
            wave1_active_pairs = []
            for s in range(world_size):
                for t in range(world_size):
                    w0 = int(per_wave_traffic_np[0, s, t])
                    w1 = int(per_wave_traffic_np[1, s, t])
                    if w1 > 0:
                        wave1_active_pairs.append((s, t, w0, w1))
            print(
                f"[LITMUS_W4_FULL] wave-1 active (s,t,w0_count,w1_count) "
                f"pairs: {wave1_active_pairs}",
                flush=True,
            )
            self.assertGreater(
                len(wave1_active_pairs), 0,
                msg=(
                    f"per_wave_traffic split has NO wave-1 entries with "
                    f"count > 0 — the litmus shape does not actually "
                    f"exercise multi-wave (per_wave_traffic={per_wave_traffic_np}). "
                    f"Adjust max_local_ids/recv_capacity to force overflow."
                ),
            )


# ---------------------------------------------------------------------------
# v64: W=4 phase-1 + phase-3 ISOLATING litmus.
#
# v60's exact kernel (W=4 multi-iter, REGULAR pre/post barriers, UNCONDITIONAL
# phase-1 DMAs, balanced 128-per-pair traffic, wave-distinct payloads, per-
# wave zero-fill of recv_buf) BUT also adds phase 3: an UNCONDITIONAL emb-
# shaped a2a back to peers' wave-distinct out_emb_ref slots, with NO post-
# emb barrier (mirroring v62's structure: wave (k+1)'s phase 1 starts
# immediately after wave k's phase 3 completes locally; the wave-distinct
# out_emb_ref slots are what's supposed to prevent races).
#
# NO phase 2, NO skewed traffic, NO dummy-offset rewrite. v63 (phase-2-only)
# PASSed, so phase 2 isn't the bug. v64 tests the next factor.
#
# Stop conditions:
#   PASS  -> phase 3 (unconditional emb DMA, no post-barrier) NOT the
#            culprit; spawn v66 next (skewed-overflow + dummy-offset rewrite
#            isolation).
#   HANG  -> phase 3's unconditional emb DMA without a post-emb barrier IS
#            the culprit; minimal repro for upstream.
#   WRONG -> unlikely; one debug iter only.
# ---------------------------------------------------------------------------


_W4_P13_TILE = 128
_W4_P13_MAX_WAVES = 2
_W4_P13_PER_PAIR_COUNT = 128
_W4_P13_WORLD_SIZE = 4
_W4_P13_PER_WAVE_RECV_SIZE = (
    _W4_P13_WORLD_SIZE * _W4_P13_PER_PAIR_COUNT
)  # 512 ints
_W4_P13_RECV_CAPACITY = (
    _W4_P13_MAX_WAVES * _W4_P13_PER_WAVE_RECV_SIZE
)  # 1024 ints
_W4_P13_MAX_PADDED = (
    _W4_P13_MAX_WAVES * _W4_P13_WORLD_SIZE * _W4_P13_PER_PAIR_COUNT
)  # 1024 ints
_W4_P13_EMB_DIM = 8
# Per-rank src_emb_ref: shape [W * TILE, D]. Slot peer p (rows [p*TILE,
# (p+1)*TILE)) is the source for the phase-3 DMA targeting peer p.
_W4_P13_SRC_EMB_ROWS = _W4_P13_WORLD_SIZE * _W4_P13_TILE  # 512
# Per-rank out_emb_ref: shape [max_waves * W * TILE, D]. Wave w + my_rank
# slot (rows [w*W*TILE + my_rank*TILE, +TILE)) is the destination.
_W4_P13_OUT_EMB_ROWS = (
    _W4_P13_MAX_WAVES * _W4_P13_WORLD_SIZE * _W4_P13_TILE
)  # 1024


def _w4_p13_sentinel(w: int, src: int, dst: int) -> int:
    """Wave/source/dest-distinct sentinel: same form as v60."""
    return 0x0A000000 + (w << 16) + (src << 8) + dst


def _make_w4_p13_kernel(world_size: int, max_waves: int, per_pair_count: int,
                        embedding_dim: int):
    """Per-device kernel: v60 body + phase-3 unconditional emb a2a (no
    post-emb barrier).

    Per wave:
      0. Per-wave zero-fill of THIS wave's slot of recv_buf via sync_copy.
      1. PRE-barrier (W-way REGULAR-sem fan-in).
      2. UNCONDITIONAL per-peer DMA (phase 1, ID a2a).
      3. POST-barrier (W-way REGULAR-sem fan-in).
      4. Phase 3 (NEW): UNCONDITIONAL per-peer emb a2a — balanced TILE-per-
         pair, src = src_emb_ref[peer*TILE:(peer+1)*TILE, :], dst on PEER =
         out_emb_ref[w*W*TILE + my_rank*TILE:+TILE, :]. NO post-emb barrier.
    """
    per_wave_recv_size = world_size * per_pair_count
    tile = _W4_P13_TILE

    def kernel(
        sorted_row_ref,    # [max_waves * W * per_pair_count] int32 HBM
        src_emb_ref,       # [W * TILE, D] f32 HBM (per-rank constant payload)
        traffic_ref,       # [max_waves, W, W] int32 SMEM
        send_off_ref,      # [max_waves, W, W] int32 SMEM
        recv_off_ref,      # [max_waves, W, W] int32 SMEM
        out_recv_ref,      # [max_waves * W * per_pair_count] int32 HBM
        out_emb_ref,       # [max_waves * W * TILE, D] f32 HBM (NEW)
        zero_int_vmem,     # [W * per_pair_count] int32 VMEM
        barrier_sems,      # REGULAR (max_waves, 2)
        send_sem_id,       # phase-1 DMA send sem
        recv_sem_id,       # phase-1 DMA recv sem
        send_sem_emb,      # phase-3 DMA send sem (separate to avoid sem
                           # reuse confusion)
        recv_sem_emb,      # phase-3 DMA recv sem
    ):
        my_rank = lax.axis_index("x")

        # Initialize zero VMEM scratch once outside the loop.
        zero_int_vmem[...] = jnp.zeros_like(zero_int_vmem)

        def wave_body(w, _carry):
            # Phase 0: zero-fill THIS WAVE'S slot of recv_buf.
            wave_recv_slot = out_recv_ref.at[
                pl.ds(w * per_wave_recv_size, per_wave_recv_size)
            ]
            pltpu.sync_copy(zero_int_vmem, wave_recv_slot)

            # Pre-barrier.
            pre_sem = barrier_sems.at[w, 0]
            for peer in range(world_size):
                pl.semaphore_signal(
                    pre_sem, 1,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
            pl.semaphore_wait(pre_sem, world_size)

            # Phase 1: ID a2a — UNCONDITIONAL (v60-style).
            for peer in range(world_size):
                send_count = pl.multiple_of(
                    traffic_ref[w, my_rank, peer], _W4_P13_TILE
                )
                send_start = pl.multiple_of(
                    send_off_ref[w, my_rank, peer], _W4_P13_TILE
                )
                remote_recv_start = pl.multiple_of(
                    recv_off_ref[w, peer, my_rank], _W4_P13_TILE
                )

                d = pltpu.make_async_remote_copy(
                    src_ref=sorted_row_ref.at[pl.ds(send_start, send_count)],
                    dst_ref=out_recv_ref.at[
                        pl.ds(remote_recv_start, send_count)
                    ],
                    send_sem=send_sem_id, recv_sem=recv_sem_id,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
                d.start()
                d.wait()

            # Post-barrier (phase 1 -> phase 3 boundary, no phase 2).
            post_sem = barrier_sems.at[w, 1]
            for peer in range(world_size):
                pl.semaphore_signal(
                    post_sem, 1,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
            pl.semaphore_wait(post_sem, world_size)

            # === NEW v64 addition: phase 3 (unconditional emb a2a) ===
            # Balanced TILE-per-pair, wave-distinct dst slots, NO post-emb
            # barrier (matching v62's structure).
            for peer in range(world_size):
                # Source: src_emb_ref[peer*TILE:(peer+1)*TILE, :] — fixed
                # per-peer slot in this rank's pre-filled emb buffer.
                src_off = peer * tile
                # Destination on PEER: their out_emb_ref at wave-distinct +
                # my_rank slot.
                dst_off = w * world_size * tile + my_rank * tile
                d = pltpu.make_async_remote_copy(
                    src_ref=src_emb_ref.at[pl.ds(src_off, tile), :],
                    dst_ref=out_emb_ref.at[pl.ds(dst_off, tile), :],
                    send_sem=send_sem_emb, recv_sem=recv_sem_emb,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
                d.start()
                d.wait()

            # NO post-emb barrier — wave-distinct out_emb_ref slots prevent
            # races (matching v62's structure).
            return None

        lax.fori_loop(0, max_waves, wave_body, None)

    return kernel


def _run_w4_p13_litmus():
    """Run W=4 phase-1+phase-3 litmus.

    Returns ``(recv_per_rank, out_emb_per_rank, max_waves, per_pair_count,
    per_wave_recv_size)``.
      ``recv_per_rank``:    [W, recv_capacity]              int32
      ``out_emb_per_rank``: [W, max_waves*W*TILE, emb_dim]  f32
    """
    world_size = _W4_P13_WORLD_SIZE
    max_waves = _W4_P13_MAX_WAVES
    per_pair_count = _W4_P13_PER_PAIR_COUNT
    per_wave_recv_size = _W4_P13_PER_WAVE_RECV_SIZE
    embedding_dim = _W4_P13_EMB_DIM
    devices = jax.devices()[:world_size]
    mesh = jax.sharding.Mesh(np.array(devices), ("x",))

    # Identical to v60: balanced 128-per-pair traffic.
    per_wave_traffic_np = np.full(
        (max_waves, world_size, world_size),
        per_pair_count, dtype=np.int32,
    )
    per_wave_send_offsets_np = np.zeros(
        (max_waves, world_size, world_size), dtype=np.int32,
    )
    per_wave_recv_offsets_np = np.zeros(
        (max_waves, world_size, world_size), dtype=np.int32,
    )
    for w in range(max_waves):
        for s in range(world_size):
            for t in range(world_size):
                per_wave_send_offsets_np[w, s, t] = (
                    w * per_wave_recv_size + t * per_pair_count
                )
                per_wave_recv_offsets_np[w, t, s] = (
                    w * per_wave_recv_size + s * per_pair_count
                )

    # Identical to v60: wave/src/dest-distinct sentinels for phase 1.
    sorted_row_padded_np = np.zeros(
        (world_size, _W4_P13_MAX_PADDED), dtype=np.int32,
    )
    for r in range(world_size):
        for w in range(max_waves):
            for p in range(world_size):
                start = w * per_wave_recv_size + p * per_pair_count
                end = start + per_pair_count
                sorted_row_padded_np[r, start:end] = _w4_p13_sentinel(w, r, p)

    # Per-rank src_emb_ref: pre-filled with per-rank constant
    # ``(rank + 1) * 1.0`` so we can verify phase 3 actually moved data.
    src_emb_np = np.zeros(
        (world_size, _W4_P13_SRC_EMB_ROWS, embedding_dim), dtype=np.float32,
    )
    for r in range(world_size):
        src_emb_np[r, :, :] = float(r + 1)

    HBM = pltpu.MemorySpace.HBM
    SMEM = pltpu.MemorySpace.SMEM

    kernel = _make_w4_p13_kernel(
        world_size=world_size, max_waves=max_waves,
        per_pair_count=per_pair_count, embedding_dim=embedding_dim,
    )

    P = jax.sharding.PartitionSpec

    def per_device(src, src_emb, traffic, send_off, recv_off):
        recv, out_emb = pl.pallas_call(
            kernel,
            out_shape=[
                jax.ShapeDtypeStruct(
                    (_W4_P13_RECV_CAPACITY,), jnp.int32,
                ),
                jax.ShapeDtypeStruct(
                    (_W4_P13_OUT_EMB_ROWS, embedding_dim), jnp.float32,
                ),
            ],
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[
                    pl.BlockSpec(memory_space=HBM),   # sorted_row
                    pl.BlockSpec(memory_space=HBM),   # src_emb
                    pl.BlockSpec(memory_space=SMEM),  # traffic
                    pl.BlockSpec(memory_space=SMEM),  # send_off
                    pl.BlockSpec(memory_space=SMEM),  # recv_off
                ],
                out_specs=[
                    pl.BlockSpec(memory_space=HBM),   # out_recv
                    pl.BlockSpec(memory_space=HBM),   # out_emb
                ],
                scratch_shapes=(
                    pltpu.VMEM((per_wave_recv_size,), jnp.int32),
                    pltpu.SemaphoreType.REGULAR((max_waves, 2)),
                    pltpu.SemaphoreType.DMA,
                    pltpu.SemaphoreType.DMA,
                    pltpu.SemaphoreType.DMA,
                    pltpu.SemaphoreType.DMA,
                ),
            ),
            interpret=False,
        )(src[0], src_emb[0], traffic, send_off, recv_off)
        return recv[None, ...], out_emb[None, ...]

    sharded = jax.shard_map(
        per_device, mesh=mesh,
        in_specs=(P("x"), P("x"), P(), P(), P()),
        out_specs=(P("x"), P("x")),
        check_vma=False,
    )

    sharding_x = jax.sharding.NamedSharding(mesh, P("x"))
    sharding_rep = jax.sharding.NamedSharding(mesh, P())
    src = jax.device_put(jnp.array(sorted_row_padded_np), sharding_x)
    src_emb = jax.device_put(jnp.array(src_emb_np), sharding_x)
    traffic = jax.device_put(jnp.array(per_wave_traffic_np), sharding_rep)
    send_off = jax.device_put(
        jnp.array(per_wave_send_offsets_np), sharding_rep,
    )
    recv_off = jax.device_put(
        jnp.array(per_wave_recv_offsets_np), sharding_rep,
    )

    recv_arr, out_emb_arr = sharded(
        src, src_emb, traffic, send_off, recv_off,
    )
    return (
        np.array(recv_arr),     # [W, recv_capacity]
        np.array(out_emb_arr),  # [W, out_emb_rows, embedding_dim]
        max_waves,
        per_pair_count,
        per_wave_recv_size,
    )


@unittest.skipUnless(_has_tpu(), "Litmus test requires real TPU")
class W4Phase13Litmus(unittest.TestCase):
    """v64: W=4 multi-iter REGULAR-sem barriers + UNCONDITIONAL DMAs (phase 1)
    + UNCONDITIONAL emb a2a (phase 3) with NO post-emb barrier. NO phase 2,
    NO skewed traffic, NO dummy-offset rewrite. Isolates whether phase 3's
    unconditional emb DMA without a post-emb barrier is the culprit for
    v62's hang.
    """

    def setUp(self) -> None:
        if jax.device_count() < 4:
            self.skipTest(
                f"Need 4 devices, have {jax.device_count()}"
            )

    def test_w4_p1_p3_no_p2(self) -> None:
        """v64 isolating litmus.

        300s host-fetch timeout via threading; if the kernel hangs the
        executor thread will not return and the test fails cleanly.
        """
        import threading

        result_box: list = []
        error_box: list = []

        def runner():
            try:
                result_box.append(_run_w4_p13_litmus())
            except BaseException as e:  # noqa: BLE001
                error_box.append(e)

        t = threading.Thread(target=runner, daemon=True)
        t.start()
        t.join(timeout=300.0)
        if t.is_alive():
            self.fail(
                "v64 W=4 phase-1+phase-3 litmus HUNG after 300s; "
                "phase 3's unconditional emb DMA without a post-emb barrier "
                "IS the culprit for v62's hang."
            )
        if error_box:
            raise error_box[0]

        recv, out_emb, max_waves, per_pair, per_wave_recv_size = (
            result_box[0]
        )

        print(
            f"[LITMUS_W4_P13] max_waves={max_waves} "
            f"per_pair={per_pair} per_wave_recv_size={per_wave_recv_size}",
            flush=True,
        )
        for r in range(_W4_P13_WORLD_SIZE):
            first_half_first4 = recv[r, :4].tolist()
            second_half_first4 = recv[
                r, per_wave_recv_size:per_wave_recv_size + 4
            ].tolist()
            print(
                f"[LITMUS_W4_P13] rank {r} "
                f"wave0_recv[:4]={[hex(int(x)) for x in first_half_first4]} "
                f"wave1_recv[:4]={[hex(int(x)) for x in second_half_first4]}",
                flush=True,
            )
            # Phase-3: print the wave/src slots of out_emb on this rank.
            # Layout: out_emb[r, w * W * TILE + s * TILE : +TILE, :] holds
            # what src rank s wrote into rank r in wave w. Expected value:
            # float(s + 1) (since src_emb_ref on rank s is filled with
            # (s+1).0).
            for w in range(_W4_P13_MAX_WAVES):
                for s in range(_W4_P13_WORLD_SIZE):
                    off = (
                        w * _W4_P13_WORLD_SIZE * _W4_P13_TILE
                        + s * _W4_P13_TILE
                    )
                    val0 = float(out_emb[r, off, 0])
                    print(
                        f"[LITMUS_W4_P13] rank {r} wave{w} src{s} "
                        f"out_emb[{off},0]={val0} (expected={float(s + 1)})",
                        flush=True,
                    )

        # Check 1: phase-1 sentinels (same as v60).
        mismatches = []
        for r in range(_W4_P13_WORLD_SIZE):
            for w in range(max_waves):
                for s in range(_W4_P13_WORLD_SIZE):
                    offset = w * per_wave_recv_size + s * per_pair
                    expected = _w4_p13_sentinel(w, s, r)
                    slot = recv[r, offset:offset + per_pair]
                    if not np.all(slot == expected):
                        actual_first8 = slot[:8].tolist()
                        mismatches.append(
                            f"rank={r} wave={w} src={s} off={offset} "
                            f"expected={hex(expected)} actual_first8="
                            f"{[hex(int(x)) for x in actual_first8]}"
                        )
        if mismatches:
            self.fail(
                "v64 phase-1 DATA MISMATCH (proves phase 3 broke phase 1):"
                "\n  " + "\n  ".join(mismatches)
            )

        # Check 2: phase-3 emb a2a actually moved data on every rank at
        # every wave-distinct slot. Expected: out_emb[r, w*W*TILE + s*TILE
        # : +TILE, :] == float(s + 1) (sender-rank's payload).
        emb_mismatches = []
        for r in range(_W4_P13_WORLD_SIZE):
            for w in range(max_waves):
                for s in range(_W4_P13_WORLD_SIZE):
                    off = (
                        w * _W4_P13_WORLD_SIZE * _W4_P13_TILE
                        + s * _W4_P13_TILE
                    )
                    expected_val = float(s + 1)
                    slot = out_emb[r, off:off + _W4_P13_TILE, :]
                    if not np.allclose(slot, expected_val, atol=1e-6):
                        emb_mismatches.append(
                            f"rank={r} wave={w} src={s} off={off} "
                            f"expected={expected_val} "
                            f"actual_first4={slot[0, :4].tolist()}"
                        )
        if emb_mismatches:
            self.fail(
                "v64 phase-3 emb a2a DATA MISMATCH (phase 3 didn't land "
                "expected per-sender values; if all-zero on a wave, that "
                "wave's phase 3 was skipped):\n  "
                + "\n  ".join(emb_mismatches)
            )


# ---------------------------------------------------------------------------
# v66a: W=4 phase-1 + skewed-overflow traffic + C-v2 dummy-offset rewrite
#       ISOLATING litmus.
#
# Same as v60 (W=4 multi-iter, REGULAR pre/post barriers, UNCONDITIONAL phase-1
# DMAs, wave-distinct payloads, per-wave zero-fill) BUT:
#   - REAL skewed-overflow traffic matrix (zero-count entries present)
#   - Combined buffers (real + dummy region)
#   - Per-(wave, my_rank, peer): eff_count = max(real, TILE); zero-real-count
#     edges redirected to wave-distinct dummy slots
#
# NO phase 2, NO phase 3. Isolates whether (skewed + dummy-offset rewrite)
# alone deadlocks at W=4 multi-iter, separately from phase 2 / phase 3
# interactions.
#
# Stop conditions:
#   PASS  -> skewed + dummy-offset on phase-1-only NOT the culprit; v62's
#            bug must lie in phase 2 x phase 3 interaction. Spawn v66b next
#            (phase 1+2+3, balanced traffic, no dummy).
#   HANG  -> skewed + dummy-offset rewrite IS the culprit (or specifically
#            the dummy-offset rewrite at W=4 multi-iter). Possibly Mosaic's
#            lowering of dummy-region accesses inside fori_loop.
#   WRONG -> effective-offset math has a bug; one debug iter only.
# ---------------------------------------------------------------------------


_W4_P1SD_TILE = 128
_W4_P1SD_VLOCAL = 256
_W4_P1SD_MAX_LOCAL_IDS = 4 * _W4_P1SD_TILE  # 512
_W4_P1SD_RECV_CAPACITY = 512
_W4_P1SD_WORLD_SIZE = 4
_W4_P1SD_SEED = 2  # matches the failing-case skewed-overflow inputs.


def _w4_p1sd_compute_eff_arrays_p1_only(
    per_wave_traffic: np.ndarray,
    per_wave_send_offsets: np.ndarray,
    per_wave_recv_offsets: np.ndarray,
    *,
    max_padded_real: int,
    recv_capacity: int,
    world_size: int,
    tile: int,
    single_trash_slot: bool = False,
):
    """Phase-1-only effective offsets/counts (subset of v62's
    `_full_compute_effective_arrays`).

    For each (w, A, B):
      eff_count[w, A, B]      = max(traffic[w, A, B], TILE)
      eff_p1_src_off[w, A, B] = if traffic[w, A, B] > 0:
                                    send_offsets[w, A, B]   (real)
                                else:
                                    max_padded_real + w*W*TILE + B*TILE
      eff_p1_dst_off[w, A, B] = (kernel reads as [w, peer=A, my_rank=B] => B->A;
                                 condition is on traffic[w, B, A] = REV)
                                if traffic[w, B, A] > 0:
                                    recv_offsets[w, A, B]   (real, recv at A
                                                              from B)
                                else:
                                    recv_capacity + B*TILE   (dummy slot for
                                                              src B on recv A)
    """
    max_waves = per_wave_traffic.shape[0]
    eff_count = np.zeros((max_waves, world_size, world_size), dtype=np.int32)
    eff_p1_src_off = np.zeros_like(eff_count)
    eff_p1_dst_off = np.zeros_like(eff_count)
    for w in range(max_waves):
        wave_dummy_base = max_padded_real + w * world_size * tile
        for A in range(world_size):
            for B in range(world_size):
                fwd = int(per_wave_traffic[w, A, B])
                rev = int(per_wave_traffic[w, B, A])
                eff_count[w, A, B] = max(fwd, tile)
                if fwd > 0:
                    eff_p1_src_off[w, A, B] = int(
                        per_wave_send_offsets[w, A, B]
                    )
                else:
                    eff_p1_src_off[w, A, B] = wave_dummy_base + B * tile
                if rev > 0:
                    eff_p1_dst_off[w, A, B] = int(
                        per_wave_recv_offsets[w, A, B]
                    )
                else:
                    # v66h: collapse all per-peer dummy slots to ONE shared
                    # trash slot at the start of the dummy region (no B*tile
                    # offset). This aliases all zero-traffic dst writes to a
                    # single location.
                    if single_trash_slot:
                        eff_p1_dst_off[w, A, B] = recv_capacity
                    else:
                        eff_p1_dst_off[w, A, B] = recv_capacity + B * tile
    return eff_count, eff_p1_src_off, eff_p1_dst_off


def _make_w4_p1sd_kernel(*, world_size: int, max_waves: int,
                         recv_capacity: int, max_padded_real: int,
                         dummy_region_size: int, tile: int,
                         recv_dummy_slots: int | None = None,
                         pipelined: bool = False):
    """v60 body, but with combined buffers + effective offsets/counts.

    Per wave:
      0. PER-WAVE zero-fill of the WHOLE combined recv buffer
         ([recv_capacity + W*TILE]) via sync_copy.
      1. PRE-barrier (W-way REGULAR-sem fan-in).
      2. UNCONDITIONAL per-peer DMA using effective offsets/counts.
         If ``pipelined``, all per-peer ``start()`` calls fire BEFORE any
         ``wait()`` (v66i source-pipelined variant); otherwise the loop
         issues ``start(); wait()`` per peer (v66a/v66g/v66h).
      3. POST-barrier (W-way REGULAR-sem fan-in).
    NO phase 2. NO phase 3.
    """
    rds = world_size if recv_dummy_slots is None else recv_dummy_slots
    out_recv_combined_size = recv_capacity + rds * tile

    def kernel(
        sorted_row_combined_ref,  # [max_padded_real + dummy_region_size] int32 HBM
        eff_count_ref,            # [max_waves, W, W] int32 SMEM
        eff_p1_src_off_ref,       # [max_waves, W, W] int32 SMEM
        eff_p1_dst_off_ref,       # [max_waves, W, W] int32 SMEM
        out_recv_combined_ref,    # [recv_capacity + rds*TILE] int32 HBM
        zero_int_vmem,            # [recv_capacity + rds*TILE] int32 VMEM
        barrier_sems,             # REGULAR (max_waves, 2)
        send_sem,                 # DMA send sem
        recv_sem,                 # DMA recv sem
    ):
        my_rank = lax.axis_index("x")

        # Initialize zero VMEM scratch once outside the loop.
        zero_int_vmem[...] = jnp.zeros_like(zero_int_vmem)

        def wave_body(w, _carry):
            # Phase 0: zero-fill the WHOLE combined recv buffer
            # (production-shape; real region + dummy slots both).
            pltpu.sync_copy(zero_int_vmem, out_recv_combined_ref)

            # Pre-barrier: fences zero-fill against fast-peer inbound DMAs.
            pre_sem = barrier_sems.at[w, 0]
            for peer in range(world_size):
                pl.semaphore_signal(
                    pre_sem, 1,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
            pl.semaphore_wait(pre_sem, world_size)

            # Phase 1: ID a2a — UNCONDITIONAL using effective offsets.
            if pipelined:
                # v66i: issue all per-peer start()s BEFORE any wait().
                # `descriptors` is a Python list — fine inside a Pallas
                # kernel because the `for peer` loop is unrolled at trace
                # time (world_size is a Python int).
                descriptors = []
                for peer in range(world_size):
                    cnt = pl.multiple_of(
                        eff_count_ref[w, my_rank, peer], tile,
                    )
                    src_off = pl.multiple_of(
                        eff_p1_src_off_ref[w, my_rank, peer], tile,
                    )
                    dst_off = pl.multiple_of(
                        eff_p1_dst_off_ref[w, peer, my_rank], tile,
                    )
                    d = pltpu.make_async_remote_copy(
                        src_ref=sorted_row_combined_ref.at[
                            pl.ds(src_off, cnt)
                        ],
                        dst_ref=out_recv_combined_ref.at[
                            pl.ds(dst_off, cnt)
                        ],
                        send_sem=send_sem, recv_sem=recv_sem,
                        device_id=(peer,),
                        device_id_type=pltpu.DeviceIdType.MESH,
                    )
                    d.start()
                    descriptors.append(d)
                for d in descriptors:
                    d.wait()
            else:
                for peer in range(world_size):
                    cnt = pl.multiple_of(
                        eff_count_ref[w, my_rank, peer], tile,
                    )
                    src_off = pl.multiple_of(
                        eff_p1_src_off_ref[w, my_rank, peer], tile,
                    )
                    dst_off = pl.multiple_of(
                        eff_p1_dst_off_ref[w, peer, my_rank], tile,
                    )
                    d = pltpu.make_async_remote_copy(
                        src_ref=sorted_row_combined_ref.at[
                            pl.ds(src_off, cnt)
                        ],
                        dst_ref=out_recv_combined_ref.at[
                            pl.ds(dst_off, cnt)
                        ],
                        send_sem=send_sem, recv_sem=recv_sem,
                        device_id=(peer,),
                        device_id_type=pltpu.DeviceIdType.MESH,
                    )
                    d.start(); d.wait()

            # Post-barrier.
            post_sem = barrier_sems.at[w, 1]
            for peer in range(world_size):
                pl.semaphore_signal(
                    post_sem, 1,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
            pl.semaphore_wait(post_sem, world_size)

            # NO phase 2, NO phase 3.
            return None

        lax.fori_loop(0, max_waves, wave_body, None)

    return kernel


def _w4_p1sd_sentinel(w: int, src: int, dst: int) -> int:
    """Wave/source/dest-distinct sentinel (same form as v60)."""
    return 0x0A000000 + (w << 16) + (src << 8) + dst


def _run_w4_p1sd_litmus(
    *,
    single_trash_slot: bool = False,
    compiler_params: object | None = None,
    pipelined: bool = False,
):
    """Run W=4 phase-1 skewed + dummy litmus.

    Args:
      single_trash_slot: if True, dummy region is one shared TILE slot
        (v66h). If False, W per-peer-distinct slots (v66a default).
      compiler_params: optional ``pltpu.CompilerParams`` passed to the
        ``pl.pallas_call`` (v66g uses this to flip libtpu flags).
      pipelined: if True, kernel issues all per-peer ``start()``s before
        any ``wait()`` (v66i source-pipelined variant).

    Returns ``(out_recv_combined_per_rank, per_wave_traffic_np,
    per_wave_recv_offsets_np, max_waves, recv_capacity, world_size, tile)``.
    """
    from recir.ops.pallas.embeddings.jagged_a2a_lookup import (
        _bucketize_and_sort,
        _max_waves_for_capacity,
        _pad_per_peer,
        _round_up_tile,
        _split_traffic_into_waves,
    )
    from recir.ops.pallas.embeddings.tests.test_jagged_a2a_lookup import (
        _make_skewed_inputs,
    )

    world_size = _W4_P1SD_WORLD_SIZE
    tile = _W4_P1SD_TILE
    max_local_ids = _W4_P1SD_MAX_LOCAL_IDS
    recv_capacity = _W4_P1SD_RECV_CAPACITY
    v_local = _W4_P1SD_VLOCAL

    devices = jax.devices()[:world_size]
    mesh = jax.sharding.Mesh(np.array(devices), ("x",))

    # 1. Build the same skewed-overflow inputs as the failing test.
    local_ids_np, local_id_count_np = _make_skewed_inputs(
        world_size, v_local, max_local_ids, seed=_W4_P1SD_SEED,
    )

    # 2. Per-rank: bucketize + pad. Compute send_counts_padded for traffic.
    max_padded_real = max_local_ids + world_size * tile
    send_counts_padded_per_rank = []
    for r in range(world_size):
        _sorted_row_real, send_counts_real, _sp, _rp = _bucketize_and_sort(
            jnp.array(local_ids_np[r]),
            jnp.array(local_id_count_np[r]),
            max_local_ids, world_size,
        )
        send_counts_padded = _round_up_tile(send_counts_real, tile)
        send_counts_padded_per_rank.append(np.array(send_counts_padded))

    # 3. Traffic matrix + per-wave split (same as production).
    traffic_matrix = jnp.array(
        np.stack(send_counts_padded_per_rank), dtype=jnp.int32,
    )
    max_waves = _max_waves_for_capacity(
        max_local_ids, world_size, recv_capacity,
    )
    per_wave_traffic, per_wave_send_offsets, per_wave_recv_offsets = (
        _split_traffic_into_waves(
            traffic_matrix,
            recv_capacity=recv_capacity,
            max_waves=max_waves,
            world_size=world_size,
        )
    )
    per_wave_traffic_np = np.array(per_wave_traffic)
    per_wave_send_offsets_np = np.array(per_wave_send_offsets)
    per_wave_recv_offsets_np = np.array(per_wave_recv_offsets)

    # 4. Phase-1-only effective offsets / counts.
    dummy_region_size = max_waves * world_size * tile
    eff_count_np, eff_p1_src_off_np, eff_p1_dst_off_np = (
        _w4_p1sd_compute_eff_arrays_p1_only(
            per_wave_traffic_np, per_wave_send_offsets_np,
            per_wave_recv_offsets_np,
            max_padded_real=max_padded_real,
            recv_capacity=recv_capacity,
            world_size=world_size,
            tile=tile,
            single_trash_slot=single_trash_slot,
        )
    )

    # 5. Build per-rank combined sorted_row buffer.
    #    Real region holds wave-distinct sentinels for non-zero edges
    #    (placed at the per_wave_send_offsets so the kernel's real DMAs
    #    pick them up). Dummy region stays zero.
    sorted_row_combined_size = max_padded_real + dummy_region_size
    sorted_row_combined_np = np.zeros(
        (world_size, sorted_row_combined_size), dtype=np.int32,
    )
    for r in range(world_size):
        for w in range(max_waves):
            for p in range(world_size):
                cnt = int(per_wave_traffic_np[w, r, p])
                if cnt <= 0:
                    continue
                start = int(per_wave_send_offsets_np[w, r, p])
                end = start + cnt
                sorted_row_combined_np[r, start:end] = _w4_p1sd_sentinel(
                    w, r, p,
                )

    HBM = pltpu.MemorySpace.HBM
    SMEM = pltpu.MemorySpace.SMEM

    recv_dummy_slots = 1 if single_trash_slot else world_size
    out_recv_combined_size = recv_capacity + recv_dummy_slots * tile

    kernel = _make_w4_p1sd_kernel(
        world_size=world_size, max_waves=max_waves,
        recv_capacity=recv_capacity, max_padded_real=max_padded_real,
        dummy_region_size=dummy_region_size, tile=tile,
        recv_dummy_slots=recv_dummy_slots,
        pipelined=pipelined,
    )

    P = jax.sharding.PartitionSpec

    def per_device(src_combined, eff_count, eff_p1_src, eff_p1_dst):
        out_recv = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(
                (out_recv_combined_size,), jnp.int32,
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[
                    pl.BlockSpec(memory_space=HBM),   # sorted_row_combined
                    pl.BlockSpec(memory_space=SMEM),  # eff_count
                    pl.BlockSpec(memory_space=SMEM),  # eff_p1_src_off
                    pl.BlockSpec(memory_space=SMEM),  # eff_p1_dst_off
                ],
                out_specs=pl.BlockSpec(memory_space=HBM),
                scratch_shapes=(
                    pltpu.VMEM(
                        (out_recv_combined_size,), jnp.int32,
                    ),
                    pltpu.SemaphoreType.REGULAR((max_waves, 2)),
                    pltpu.SemaphoreType.DMA,
                    pltpu.SemaphoreType.DMA,
                ),
            ),
            compiler_params=compiler_params,
            interpret=False,
        )(src_combined[0], eff_count, eff_p1_src, eff_p1_dst)
        return out_recv[None, ...]

    sharded = jax.shard_map(
        per_device, mesh=mesh,
        in_specs=(P("x"), P(), P(), P()),
        out_specs=P("x"),
        check_vma=False,
    )

    sharding_x = jax.sharding.NamedSharding(mesh, P("x"))
    sharding_rep = jax.sharding.NamedSharding(mesh, P())
    src = jax.device_put(jnp.array(sorted_row_combined_np), sharding_x)
    eff_count = jax.device_put(jnp.array(eff_count_np), sharding_rep)
    eff_p1_src = jax.device_put(jnp.array(eff_p1_src_off_np), sharding_rep)
    eff_p1_dst = jax.device_put(jnp.array(eff_p1_dst_off_np), sharding_rep)

    out_recv_arr = sharded(src, eff_count, eff_p1_src, eff_p1_dst)
    return (
        np.array(out_recv_arr),  # [W, recv_capacity + W*TILE]
        per_wave_traffic_np,
        per_wave_recv_offsets_np,
        max_waves,
        recv_capacity,
        world_size,
        tile,
    )


@unittest.skipUnless(_has_tpu(), "Litmus test requires real TPU")
class W4Phase1SkewedDummyLitmus(unittest.TestCase):
    """W=4 multi-iter REGULAR-sem barriers + UNCONDITIONAL phase-1 DMAs +
    REAL skewed-overflow traffic matrix + C-v2 dummy-offset rewrite.

    No phase 2, no phase 3. Isolates whether (skewed + dummy-offset rewrite)
    alone causes the v62 hang, separately from phase 2 / phase 3.
    """

    def setUp(self) -> None:
        if jax.device_count() < 4:
            self.skipTest(
                f"Need 4 devices, have {jax.device_count()}"
            )

    def test_w4_p1_skewed_dummy_no_p2_p3(self) -> None:
        """v66a: phase-1-only skewed + dummy litmus.

        300s host-fetch timeout via threading; if the kernel hangs, the
        executor thread will not return and the test will fail cleanly.
        """
        import threading

        result_box: list = []
        error_box: list = []

        def runner():
            try:
                result_box.append(_run_w4_p1sd_litmus())
            except BaseException as e:  # noqa: BLE001
                error_box.append(e)

        t = threading.Thread(target=runner, daemon=True)
        t.start()
        t.join(timeout=300.0)
        if t.is_alive():
            self.fail(
                "v66a (W=4 phase-1 skewed + dummy-offset) HUNG after 300s; "
                "skewed + dummy-offset rewrite IS the culprit (or specifically "
                "the dummy-offset rewrite at W=4 multi-iter)."
            )
        if error_box:
            raise error_box[0]

        (out_recv, per_wave_traffic_np, per_wave_recv_offsets_np,
         max_waves, recv_capacity, world_size, tile) = result_box[0]

        # Diagnostic header.
        print(
            f"[LITMUS_W4_P1SD] max_waves={max_waves} "
            f"recv_capacity={recv_capacity} world_size={world_size} "
            f"tile={tile}",
            flush=True,
        )
        # Per-rank, per-wave nonzero traffic counts.
        for r in range(world_size):
            for w in range(max_waves):
                for s in range(world_size):
                    cnt = int(per_wave_traffic_np[w, s, r])
                    if cnt > 0:
                        off = int(per_wave_recv_offsets_np[w, r, s])
                        first4 = out_recv[r, off:off + 4].tolist()
                        print(
                            f"[LITMUS_W4_P1SD] rank={r} wave={w} src={s} "
                            f"cnt={cnt} off={off} first4="
                            f"{[hex(int(x)) for x in first4]}",
                            flush=True,
                        )

        # Verification: real region only (out_recv[:, :recv_capacity]).
        # For each non-zero traffic edge, expect the wave-distinct sentinel
        # at the corresponding recv offset.
        mismatches = []
        for r in range(world_size):
            for w in range(max_waves):
                for s in range(world_size):
                    cnt = int(per_wave_traffic_np[w, s, r])
                    if cnt <= 0:
                        continue
                    off = int(per_wave_recv_offsets_np[w, r, s])
                    expected = _w4_p1sd_sentinel(w, s, r)
                    slot = out_recv[r, off:off + cnt]
                    if not np.all(slot == expected):
                        actual_first8 = slot[:8].tolist()
                        mismatches.append(
                            f"rank={r} wave={w} src={s} cnt={cnt} off={off} "
                            f"expected={hex(expected)} actual_first8="
                            f"{[hex(int(x)) for x in actual_first8]}"
                        )
        if mismatches:
            self.fail(
                "v66a phase-1 skewed + dummy DATA MISMATCH:\n  "
                + "\n  ".join(mismatches)
            )


# ---------------------------------------------------------------------------
# v66g: v66a verbatim + libtpu flag flip to discourage `dma_done` sinking.
# Real-TPU lowering has NO direct `dma_execution_mode='eager'` knob (that
# attribute is interpret-mode-only on `pltpu.InterpretParams`). The closest
# real-TPU equivalent exposed via `pltpu.CompilerParams.flags` is the libtpu
# boolean `XLA_TPU_RESCHEDULE_DMA_DONES` (the only DMA-rescheduling knob in
# the libtpu flag-config proto enum). Setting it False prevents the compiler
# from rescheduling `dma_done` insertion past the explicitly-coded wait site.
#
# Stop conditions:
#   PASS  -> firmware's deferred (sunk) DMA completion matcher is responsible
#            for the v66a hang.
#   HANG  -> sinking is not the trigger; bug is elsewhere.
# ---------------------------------------------------------------------------


@unittest.skipUnless(_has_tpu(), "Litmus test requires real TPU")
class W4Phase1SkewedDummyEagerDmaLitmus(unittest.TestCase):
    """v66g: v66a + ``flags={'xla_tpu_sink_dma_done': False}``."""

    def setUp(self) -> None:
        if jax.device_count() < 4:
            self.skipTest(f"Need 4 devices, have {jax.device_count()}")

    def test_w4_p1_skewed_dummy_eager_dma(self) -> None:
        import threading

        cp = pltpu.CompilerParams(
            flags={"XLA_TPU_RESCHEDULE_DMA_DONES": False},
        )
        result_box: list = []
        error_box: list = []

        def runner():
            try:
                result_box.append(
                    _run_w4_p1sd_litmus(compiler_params=cp)
                )
            except BaseException as e:  # noqa: BLE001
                error_box.append(e)

        t = threading.Thread(target=runner, daemon=True)
        t.start()
        t.join(timeout=300.0)
        if t.is_alive():
            self.fail(
                "v66g (v66a + xla_tpu_sink_dma_done=False) HUNG after 300s; "
                "non-sunk dma_done does NOT lift the deadlock."
            )
        if error_box:
            raise error_box[0]

        (out_recv, per_wave_traffic_np, per_wave_recv_offsets_np,
         max_waves, recv_capacity, world_size, tile) = result_box[0]
        print(
            f"[LITMUS_W4_P1SD_v66g] max_waves={max_waves} "
            f"recv_capacity={recv_capacity} world_size={world_size} "
            f"tile={tile}",
            flush=True,
        )
        mismatches = []
        for r in range(world_size):
            for w in range(max_waves):
                for s in range(world_size):
                    cnt = int(per_wave_traffic_np[w, s, r])
                    if cnt <= 0:
                        continue
                    off = int(per_wave_recv_offsets_np[w, r, s])
                    expected = _w4_p1sd_sentinel(w, s, r)
                    slot = out_recv[r, off:off + cnt]
                    if not np.all(slot == expected):
                        actual_first8 = slot[:8].tolist()
                        mismatches.append(
                            f"rank={r} wave={w} src={s} cnt={cnt} off={off} "
                            f"expected={hex(expected)} actual_first8="
                            f"{[hex(int(x)) for x in actual_first8]}"
                        )
        if mismatches:
            self.fail(
                "v66g eager-DMA DATA MISMATCH:\n  " + "\n  ".join(mismatches)
            )


# ---------------------------------------------------------------------------
# v66h: v66a verbatim, but the recv-side dummy region collapses to ONE shared
# trash slot. All zero-traffic peers write to ``recv_capacity + 0`` instead of
# ``recv_capacity + B*tile``. Total recv buffer = ``[recv_capacity + tile]``.
# Source-side dummy slots stay W per-peer-distinct (unchanged).
#
# Stop conditions:
#   PASS  -> bug needs DISTINCT dst offsets per peer in BOTH real and dummy
#            halves; aliasing dummies is a near-free workaround.
#   HANG  -> bug is purely about the data-dependent real/dummy boolean choice,
#            not about offset distinctness in the dummy half.
# ---------------------------------------------------------------------------


@unittest.skipUnless(_has_tpu(), "Litmus test requires real TPU")
class W4Phase1SkewedSingleTrashLitmus(unittest.TestCase):
    """v66h: v66a + single shared dummy/trash tile on the recv side."""

    def setUp(self) -> None:
        if jax.device_count() < 4:
            self.skipTest(f"Need 4 devices, have {jax.device_count()}")

    def test_w4_p1_skewed_single_trash(self) -> None:
        import threading

        result_box: list = []
        error_box: list = []

        def runner():
            try:
                result_box.append(
                    _run_w4_p1sd_litmus(single_trash_slot=True)
                )
            except BaseException as e:  # noqa: BLE001
                error_box.append(e)

        t = threading.Thread(target=runner, daemon=True)
        t.start()
        t.join(timeout=300.0)
        if t.is_alive():
            self.fail(
                "v66h (v66a + single-shared-trash recv slot) HUNG after 300s; "
                "data-dependent real/dummy choice ALONE is the trigger; "
                "offset distinctness in the dummy half is irrelevant."
            )
        if error_box:
            raise error_box[0]

        (out_recv, per_wave_traffic_np, per_wave_recv_offsets_np,
         max_waves, recv_capacity, world_size, tile) = result_box[0]
        print(
            f"[LITMUS_W4_P1SD_v66h] max_waves={max_waves} "
            f"recv_capacity={recv_capacity} world_size={world_size} "
            f"tile={tile}",
            flush=True,
        )
        mismatches = []
        for r in range(world_size):
            for w in range(max_waves):
                for s in range(world_size):
                    cnt = int(per_wave_traffic_np[w, s, r])
                    if cnt <= 0:
                        continue
                    off = int(per_wave_recv_offsets_np[w, r, s])
                    expected = _w4_p1sd_sentinel(w, s, r)
                    slot = out_recv[r, off:off + cnt]
                    if not np.all(slot == expected):
                        actual_first8 = slot[:8].tolist()
                        mismatches.append(
                            f"rank={r} wave={w} src={s} cnt={cnt} off={off} "
                            f"expected={hex(expected)} actual_first8="
                            f"{[hex(int(x)) for x in actual_first8]}"
                        )
        if mismatches:
            self.fail(
                "v66h single-trash DATA MISMATCH:\n  " + "\n  ".join(mismatches)
            )


# ---------------------------------------------------------------------------
# v66i: v66a verbatim, but the per-peer DMA loop in phase 1 issues all
# ``start()`` calls BEFORE any ``wait()`` (source-pipelined). This tests
# Gemini's eager-DMA-descriptor-issue hypothesis at the source-ordering level
# (rather than via libtpu flag flips, which gate completion-ack scheduling
# and not descriptor-issue timing — that was v66g, which still hung).
#
# Same buffers, same SMEM inputs, same traffic matrix, same dummy region as
# v66a. Only the inner phase-1 per-peer DMA loop changes:
#   v66a: for peer: d = make_async_remote_copy(...); d.start(); d.wait()
#   v66i: descriptors = []
#         for peer: d = make_async_remote_copy(...); d.start(); descriptors.append(d)
#         for d in descriptors: d.wait()
#
# ``descriptors`` is a Python list, fine inside a Pallas kernel because the
# ``for peer in range(world_size)`` loop is unrolled at trace time.
#
# Stop conditions:
#   PASS  -> source-level pipelined start/wait alone lifts the hang. Strong
#            evidence Gemini's "deferred descriptor submission" theory is
#            correct; we have a near-zero-cost workaround.
#   HANG  -> source ordering of start/wait is NOT the trigger axis; the bug
#            is purely about the data-dependent dst-region branch, with no
#            fix at the start/wait scheduling level.
# ---------------------------------------------------------------------------


@unittest.skipUnless(_has_tpu(), "Litmus test requires real TPU")
class W4Phase1SkewedDummyPipelinedLitmus(unittest.TestCase):
    """v66i: v66a + source-pipelined per-peer start/wait."""

    def setUp(self) -> None:
        if jax.device_count() < 4:
            self.skipTest(f"Need 4 devices, have {jax.device_count()}")

    def test_w4_p1_skewed_dummy_pipelined(self) -> None:
        import threading

        result_box: list = []
        error_box: list = []

        def runner():
            try:
                result_box.append(
                    _run_w4_p1sd_litmus(pipelined=True)
                )
            except BaseException as e:  # noqa: BLE001
                error_box.append(e)

        t = threading.Thread(target=runner, daemon=True)
        t.start()
        t.join(timeout=300.0)
        if t.is_alive():
            self.fail(
                "v66i (v66a + source-pipelined per-peer start/wait) HUNG "
                "after 300s; source-level start/wait ordering is NOT the "
                "trigger axis; bug is purely about the data-dependent "
                "dst-region branch."
            )
        if error_box:
            raise error_box[0]

        (out_recv, per_wave_traffic_np, per_wave_recv_offsets_np,
         max_waves, recv_capacity, world_size, tile) = result_box[0]
        print(
            f"[LITMUS_W4_P1SDP] max_waves={max_waves} "
            f"recv_capacity={recv_capacity} world_size={world_size} "
            f"tile={tile}",
            flush=True,
        )
        for r in range(world_size):
            for w in range(max_waves):
                for s in range(world_size):
                    cnt = int(per_wave_traffic_np[w, s, r])
                    if cnt > 0:
                        off = int(per_wave_recv_offsets_np[w, r, s])
                        first4 = out_recv[r, off:off + 4].tolist()
                        print(
                            f"[LITMUS_W4_P1SDP] rank={r} wave={w} src={s} "
                            f"cnt={cnt} off={off} first4="
                            f"{[hex(int(x)) for x in first4]}",
                            flush=True,
                        )

        mismatches = []
        for r in range(world_size):
            for w in range(max_waves):
                for s in range(world_size):
                    cnt = int(per_wave_traffic_np[w, s, r])
                    if cnt <= 0:
                        continue
                    off = int(per_wave_recv_offsets_np[w, r, s])
                    expected = _w4_p1sd_sentinel(w, s, r)
                    slot = out_recv[r, off:off + cnt]
                    if not np.all(slot == expected):
                        actual_first8 = slot[:8].tolist()
                        mismatches.append(
                            f"rank={r} wave={w} src={s} cnt={cnt} off={off} "
                            f"expected={hex(expected)} actual_first8="
                            f"{[hex(int(x)) for x in actual_first8]}"
                        )
        if mismatches:
            self.fail(
                "v66i pipelined start/wait DATA MISMATCH:\n  "
                + "\n  ".join(mismatches)
            )


# ---------------------------------------------------------------------------
# v66j: W=4 phase-1 ONLY — combine v66d's VARIABLE per-peer counts with v66e's
# COMPILE-TIME-FIXED real/dummy region split. Distinguishes the running
# fingerprint "the bug needs a RUNTIME-data-dependent boolean for the
# dst-region choice" from the alternative "the bug just needs variable size +
# mixed regions regardless of branch source".
#
# Design:
#   - Counts: alternating literal pattern by destination peer:
#       counts[peer] = 128 if peer % 2 == 0 else 256
#     baked as a [max_waves, W, W] SMEM array (eff_count[w, my_rank, peer] =
#     counts[peer]); NOT derived from any runtime/skewed input.
#   - Region choice: Python ``if peer % 2 == 0`` in the unrolled per-peer
#     loop (compile-time, identical to v66e).
#       even peers (0, 2) -> REAL region
#       odd  peers (1, 3) -> DUMMY region
#   - All counts > 0, all TILE-aligned (128 and 256 = 2*TILE).
#   - Real region per (wave, receiver): 2 senders write 128 each = 256 ints
#     plus zero-fill slack. Use uniform per-receiver real slot of size 256.
#     2 waves => 512 ints total real region; recv_capacity = 1024 (slack).
#   - Dummy region per (wave, receiver): 2 senders write 256 each at distinct
#     slots indexed by sender rank. Use slot stride = 256 (= max odd count).
#     world_size * 256 = 1024 ints dummy region per receiver.
#
# Stop conditions:
#   PASS  -> confirms the runtime-data-dependent qualifier in the fingerprint
#            is load-bearing. Compile-time peer-indexed region choice is safe
#            even with variable counts. Fingerprint stands.
#   HANG  -> invalidates the runtime-data-dependent qualifier. The bug is just
#            "variable size + mixed regions" regardless of where the branch
#            comes from. Annotations need rewrite.
# ---------------------------------------------------------------------------


_W4_P1VCFR_TILE = 128
_W4_P1VCFR_MAX_WAVES = 2
_W4_P1VCFR_WORLD_SIZE = 4
# Alternating per-peer count pattern (indexed by destination peer):
# even peer -> 128 (one TILE), odd peer -> 256 (two TILEs). All > 0,
# TILE-aligned. Even peers go to REAL region, odd to DUMMY region.
_W4_P1VCFR_COUNTS = (128, 256, 128, 256)
_W4_P1VCFR_REAL_CNT = 128   # = COUNTS[0] = COUNTS[2]
_W4_P1VCFR_DUMMY_CNT = 256  # = COUNTS[1] = COUNTS[3]
# Per-receiver REAL slot: 2 even senders write _REAL_CNT each at offsets
# 0 and 1*_REAL_CNT (per wave). per_wave_real = world_size * _REAL_CNT = 512
# (2 senders use it; 2 odd senders use the dummy region instead). Pad with
# slack so recv_capacity is TILE-aligned and easy to reason about.
_W4_P1VCFR_PER_WAVE_REAL = _W4_P1VCFR_WORLD_SIZE * _W4_P1VCFR_REAL_CNT  # 512
_W4_P1VCFR_RECV_CAPACITY = (
    _W4_P1VCFR_MAX_WAVES * _W4_P1VCFR_PER_WAVE_REAL
)  # 1024 ints
# Dummy region: per-receiver, slot stride = _DUMMY_CNT, indexed by sender
# rank. World_size senders -> world_size slots, but only odd peers (=odd dst)
# write to the dummy region; from receiver r's perspective, every sender s
# writes to r's dummy region iff r % 2 == 1, at slot recv_capacity + s*_DUMMY_CNT.
_W4_P1VCFR_DUMMY_REGION_SIZE = (
    _W4_P1VCFR_WORLD_SIZE * _W4_P1VCFR_DUMMY_CNT
)  # 1024 ints
_W4_P1VCFR_OUT_RECV_COMBINED_SIZE = (
    _W4_P1VCFR_RECV_CAPACITY + _W4_P1VCFR_DUMMY_REGION_SIZE
)  # 2048 ints
# Sender-side sorted_row buffer (real region only): per wave, sender writes
# to even peers at offsets w*_PER_WAVE_REAL + peer_idx_among_even * _REAL_CNT.
# Two even peers per wave -> 2 * _REAL_CNT = 256 ints used per wave per
# sender. Pad to _MAX_PADDED = max_waves * world_size * _REAL_CNT = 1024.
_W4_P1VCFR_MAX_PADDED = (
    _W4_P1VCFR_MAX_WAVES * _W4_P1VCFR_WORLD_SIZE * _W4_P1VCFR_REAL_CNT
)  # 1024 ints
# Sender-side dummy_src buffer: filled with constant; size = world_size *
# _DUMMY_CNT so each odd peer reads its own slot (peer * _DUMMY_CNT, _DUMMY_CNT).
_W4_P1VCFR_DUMMY_SRC_SIZE = (
    _W4_P1VCFR_WORLD_SIZE * _W4_P1VCFR_DUMMY_CNT
)  # 1024 ints
_W4_P1VCFR_DUMMY_SENTINEL = 0xDEADBEEF


def _w4_p1vcfr_sentinel(w: int, src: int, dst: int) -> int:
    """Wave/source/dest-distinct sentinel: same form as v66a/v66e."""
    return 0x0A000000 + (w << 16) + (src << 8) + dst


def _make_w4_p1vcfr_kernel(world_size: int, max_waves: int,
                           recv_capacity: int, tile: int,
                           real_cnt: int, dummy_cnt: int):
    """v66e structure + v66d-style variable counts (literal alternating pattern).

    Per wave:
      0. Per-wave slot-only zero-fill of THIS wave's REAL slot (size
         per_wave_real = world_size * real_cnt) via sync_copy.
      1. PRE-barrier (W-way REGULAR-sem fan-in).
      2. UNCONDITIONAL per-peer DMA with VARIABLE count (read from SMEM but
         literally [real_cnt, dummy_cnt, real_cnt, dummy_cnt] across peers),
         and COMPILE-TIME fixed region choice via ``if peer % 2 == 0``:
           even peer -> REAL region, count = real_cnt, src = sorted_row
           odd  peer -> DUMMY region, count = dummy_cnt, src = dummy_src
      3. POST-barrier (W-way REGULAR-sem fan-in).
    """
    per_wave_real = world_size * real_cnt

    def kernel(
        sorted_row_ref,        # [_MAX_PADDED] int32 HBM
        dummy_src_ref,         # [_DUMMY_SRC_SIZE] int32 HBM (constant sentinel)
        eff_count_ref,         # [max_waves, W, W] int32 SMEM (literal pattern)
        out_recv_combined_ref, # [recv_capacity + _DUMMY_REGION_SIZE] int32 HBM
        zero_int_vmem,         # [per_wave_real] int32 VMEM zero buffer
        barrier_sems,          # REGULAR (max_waves, 2)
        send_sem,              # DMA send sem
        recv_sem,              # DMA recv sem
    ):
        my_rank = lax.axis_index("x")

        zero_int_vmem[...] = jnp.zeros_like(zero_int_vmem)

        def wave_body(w, _carry):
            # Phase 0: per-wave slot-only zero-fill of REAL region.
            wave_real_slot = out_recv_combined_ref.at[
                pl.ds(w * per_wave_real, per_wave_real)
            ]
            pltpu.sync_copy(zero_int_vmem, wave_real_slot)

            # Pre-barrier.
            pre_sem = barrier_sems.at[w, 0]
            for peer in range(world_size):
                pl.semaphore_signal(
                    pre_sem, 1,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
            pl.semaphore_wait(pre_sem, world_size)

            # Phase 1: per-peer DMA with VARIABLE count + COMPILE-TIME region
            # split. Both axes are statically determined per call site:
            # `if peer % 2 == 0` resolves at trace time (loop is unrolled),
            # and the count is also the static per-peer literal — but we read
            # it from SMEM via eff_count_ref to mirror v66d's data-flow shape
            # (so Mosaic sees the same SMEM-read scalar for the DMA size).
            for peer in range(world_size):
                cnt = pl.multiple_of(
                    eff_count_ref[w, my_rank, peer], tile,
                )
                if peer % 2 == 0:
                    # Even peer -> REAL region (v60-style).
                    # Sender's wave slot has 2 even peers each at
                    # _REAL_CNT-stride; this peer's slot index among evens
                    # is peer // 2.
                    send_start = w * per_wave_real + (peer // 2) * real_cnt
                    # Receiver's per-wave real slot: same layout (sender
                    # rank's slot indexed by my_rank // 2 among evens). But
                    # the receiver doesn't know its own evenness here; we
                    # use the symmetric layout: dst slot indexed by sender
                    # rank's even-position. Since this DMA is from my_rank
                    # to peer, on peer's receive buffer the sender (us) is
                    # my_rank, and the slot is `my_rank // 2`-th among evens
                    # only if my_rank is even. To keep it simple, use a
                    # full sender-indexed layout: slot at sender rank * real_cnt
                    # within the per-wave real region (4 slots, one per sender,
                    # but odd senders won't write here). Per-wave real region
                    # is `world_size * real_cnt = 512` so each sender slot
                    # (real_cnt = 128) fits and is uniquely addressed.
                    remote_recv_start = (
                        w * per_wave_real + my_rank * real_cnt
                    )
                    src = sorted_row_ref.at[pl.ds(send_start, cnt)]
                    dst = out_recv_combined_ref.at[
                        pl.ds(remote_recv_start, cnt)
                    ]
                else:
                    # Odd peer -> DUMMY region (fixed offsets, sender-rank
                    # indexed slot of size _DUMMY_CNT each).
                    src = dummy_src_ref.at[pl.ds(peer * dummy_cnt, dummy_cnt)]
                    dst = out_recv_combined_ref.at[
                        pl.ds(
                            recv_capacity + my_rank * dummy_cnt, dummy_cnt,
                        )
                    ]

                d = pltpu.make_async_remote_copy(
                    src_ref=src, dst_ref=dst,
                    send_sem=send_sem, recv_sem=recv_sem,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
                d.start()
                d.wait()

            # Post-barrier.
            post_sem = barrier_sems.at[w, 1]
            for peer in range(world_size):
                pl.semaphore_signal(
                    post_sem, 1,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
            pl.semaphore_wait(post_sem, world_size)
            return None

        lax.fori_loop(0, max_waves, wave_body, None)

    return kernel


def _run_w4_p1vcfr_litmus():
    """Run W=4 phase-1 variable-count + compile-time-fixed real/dummy region.

    Returns ``(out_recv_combined_per_rank, max_waves, real_cnt, dummy_cnt,
    per_wave_real, recv_capacity, world_size, tile, dummy_sentinel)``.
    """
    world_size = _W4_P1VCFR_WORLD_SIZE
    max_waves = _W4_P1VCFR_MAX_WAVES
    real_cnt = _W4_P1VCFR_REAL_CNT
    dummy_cnt = _W4_P1VCFR_DUMMY_CNT
    per_wave_real = _W4_P1VCFR_PER_WAVE_REAL
    recv_capacity = _W4_P1VCFR_RECV_CAPACITY
    tile = _W4_P1VCFR_TILE
    devices = jax.devices()[:world_size]
    mesh = jax.sharding.Mesh(np.array(devices), ("x",))

    # Literal alternating count pattern, baked into [max_waves, W, W] SMEM.
    # eff_count[w, my_rank, peer] = _W4_P1VCFR_COUNTS[peer] for all w/my_rank.
    eff_count_np = np.zeros(
        (max_waves, world_size, world_size), dtype=np.int32,
    )
    for w in range(max_waves):
        for s in range(world_size):
            for p in range(world_size):
                eff_count_np[w, s, p] = _W4_P1VCFR_COUNTS[p]

    # Per-rank sorted_row: only EVEN peers read it. For each (wave w, even
    # peer p), fill slot [w*per_wave_real + (p//2)*real_cnt, +real_cnt) with
    # the wave/src/dst-distinct sentinel.
    sorted_row_np = np.zeros(
        (world_size, _W4_P1VCFR_MAX_PADDED), dtype=np.int32,
    )
    for r in range(world_size):
        for w in range(max_waves):
            for p in range(world_size):
                if p % 2 == 0:
                    start = w * per_wave_real + (p // 2) * real_cnt
                    end = start + real_cnt
                    sorted_row_np[r, start:end] = _w4_p1vcfr_sentinel(
                        w, r, p,
                    )

    # Per-rank dummy_src: pre-filled with the constant DUMMY sentinel.
    dummy_src_np = np.full(
        (world_size, _W4_P1VCFR_DUMMY_SRC_SIZE),
        _W4_P1VCFR_DUMMY_SENTINEL, dtype=np.uint32,
    ).view(np.int32)

    HBM = pltpu.MemorySpace.HBM
    SMEM = pltpu.MemorySpace.SMEM

    kernel = _make_w4_p1vcfr_kernel(
        world_size=world_size, max_waves=max_waves,
        recv_capacity=recv_capacity, tile=tile,
        real_cnt=real_cnt, dummy_cnt=dummy_cnt,
    )

    P = jax.sharding.PartitionSpec

    def per_device(src, dummy_src, eff_count):
        out_recv = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(
                (_W4_P1VCFR_OUT_RECV_COMBINED_SIZE,), jnp.int32,
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[
                    pl.BlockSpec(memory_space=HBM),   # sorted_row
                    pl.BlockSpec(memory_space=HBM),   # dummy_src
                    pl.BlockSpec(memory_space=SMEM),  # eff_count
                ],
                out_specs=pl.BlockSpec(memory_space=HBM),
                scratch_shapes=(
                    pltpu.VMEM((per_wave_real,), jnp.int32),
                    pltpu.SemaphoreType.REGULAR((max_waves, 2)),
                    pltpu.SemaphoreType.DMA,
                    pltpu.SemaphoreType.DMA,
                ),
            ),
            interpret=False,
        )(src[0], dummy_src[0], eff_count)
        return out_recv[None, ...]

    sharded = jax.shard_map(
        per_device, mesh=mesh,
        in_specs=(P("x"), P("x"), P()),
        out_specs=P("x"),
        check_vma=False,
    )

    sharding_x = jax.sharding.NamedSharding(mesh, P("x"))
    sharding_rep = jax.sharding.NamedSharding(mesh, P())
    src = jax.device_put(jnp.array(sorted_row_np), sharding_x)
    dummy_src = jax.device_put(jnp.array(dummy_src_np), sharding_x)
    eff_count = jax.device_put(jnp.array(eff_count_np), sharding_rep)

    out = sharded(src, dummy_src, eff_count)
    return (
        np.array(out),  # [W, recv_capacity + _DUMMY_REGION_SIZE]
        max_waves,
        real_cnt,
        dummy_cnt,
        per_wave_real,
        recv_capacity,
        world_size,
        tile,
        _W4_P1VCFR_DUMMY_SENTINEL,
    )


@unittest.skipUnless(_has_tpu(), "Litmus test requires real TPU")
class W4Phase1VarCountFixedRegionLitmus(unittest.TestCase):
    """v66j: W=4 multi-iter REGULAR-sem barriers + UNCONDITIONAL phase-1
    DMAs combining VARIABLE per-peer counts (literal alternating pattern,
    [128, 256, 128, 256]) with COMPILE-TIME-FIXED real/dummy region split
    (Python ``if peer % 2 == 0`` in the unrolled per-peer loop, even -> real,
    odd -> dummy). Distinguishes whether v66a's hang requires a
    RUNTIME-data-dependent dst-region branch or just variable size + mixed
    regions regardless of branch source.
    """

    def setUp(self) -> None:
        if jax.device_count() < 4:
            self.skipTest(
                f"Need 4 devices, have {jax.device_count()}"
            )

    def test_w4_p1_var_count_fixed_region(self) -> None:
        """v66j: variable count + compile-time-fixed real/dummy region split.

        300s host-fetch timeout via threading; if the kernel hangs, the
        executor thread will not return and the test will fail cleanly.
        """
        import threading

        result_box: list = []
        error_box: list = []

        def runner():
            try:
                result_box.append(_run_w4_p1vcfr_litmus())
            except BaseException as e:  # noqa: BLE001
                error_box.append(e)

        t = threading.Thread(target=runner, daemon=True)
        t.start()
        t.join(timeout=300.0)
        if t.is_alive():
            self.fail(
                "v66j (W=4 phase-1 variable-count + compile-time-fixed "
                "real/dummy region split) HUNG after 300s; the bug does NOT "
                "require a runtime-data-dependent dst-region branch — "
                "variable size + mixed regions alone is sufficient. "
                "Fingerprint annotation needs a rewrite."
            )
        if error_box:
            raise error_box[0]

        (recv, max_waves, real_cnt, dummy_cnt, per_wave_real, recv_capacity,
         world_size, tile, dummy_sent) = result_box[0]

        print(
            f"[LITMUS_W4_P1VCFR] max_waves={max_waves} real_cnt={real_cnt} "
            f"dummy_cnt={dummy_cnt} per_wave_real={per_wave_real} "
            f"recv_capacity={recv_capacity} world_size={world_size} "
            f"tile={tile} dummy_sentinel={hex(dummy_sent)} "
            f"counts={_W4_P1VCFR_COUNTS}",
            flush=True,
        )
        # Per-rank diagnostics: print first 4 ints of each wave's real slot
        # (per-sender) and the dummy slot for each sender.
        for r in range(world_size):
            for w in range(max_waves):
                for s in range(world_size):
                    real_off = w * per_wave_real + s * real_cnt
                    real_first4 = recv[r, real_off:real_off + 4].tolist()
                    print(
                        f"[LITMUS_W4_P1VCFR] rank={r} wave={w} src={s} "
                        f"REAL off={real_off} first4="
                        f"{[hex(int(x)) for x in real_first4]}",
                        flush=True,
                    )
            for s in range(world_size):
                dummy_off = recv_capacity + s * dummy_cnt
                dummy_first4 = recv[r, dummy_off:dummy_off + 4].tolist()
                print(
                    f"[LITMUS_W4_P1VCFR] rank={r} DUMMY src={s} "
                    f"off={dummy_off} first4="
                    f"{[hex(int(x) & 0xFFFFFFFF) for x in dummy_first4]}",
                    flush=True,
                )

        # Verification semantics:
        # In the kernel, the "real vs dummy" choice is by ``peer % 2`` where
        # ``peer`` is the sender's destination index. So sender s writes to
        # receiver r's REAL region iff r % 2 == 0 (with size real_cnt at the
        # sender-rank indexed slot), and to receiver r's DUMMY region iff
        # r % 2 == 1 (with size dummy_cnt at sender-rank indexed slot).
        #   - Even-rank receiver: every sender wrote real region with
        #                          (w, s, r) sentinel; dummy region untouched.
        #   - Odd-rank receiver: every sender wrote dummy region with the
        #                         constant dummy sentinel; real region all
        #                         zero (zero-filled, no sender wrote there).
        expected_dummy = np.array(
            [_W4_P1VCFR_DUMMY_SENTINEL], dtype=np.uint32,
        ).view(np.int32)[0]
        mismatches = []
        for r in range(world_size):
            for w in range(max_waves):
                for s in range(world_size):
                    real_off = w * per_wave_real + s * real_cnt
                    slot = recv[r, real_off:real_off + real_cnt]
                    if r % 2 == 0:
                        expected = _w4_p1vcfr_sentinel(w, s, r)
                        if not np.all(slot == expected):
                            actual_first8 = slot[:8].tolist()
                            mismatches.append(
                                f"rank={r} wave={w} src={s} off={real_off} "
                                f"REAL(even-recv) expected={hex(expected)} "
                                f"actual_first8="
                                f"{[hex(int(x)) for x in actual_first8]}"
                            )
                    else:
                        if not np.all(slot == 0):
                            actual_first8 = slot[:8].tolist()
                            mismatches.append(
                                f"rank={r} wave={w} src={s} off={real_off} "
                                f"REAL(odd-recv) expected=0x0 (zero-fill) "
                                f"actual_first8="
                                f"{[hex(int(x)) for x in actual_first8]}"
                            )
            # Dummy region: only odd receivers receive there.
            if r % 2 == 1:
                for s in range(world_size):
                    dummy_off = recv_capacity + s * dummy_cnt
                    slot = recv[r, dummy_off:dummy_off + dummy_cnt]
                    if not np.all(slot == expected_dummy):
                        actual_first8 = slot[:8].tolist()
                        mismatches.append(
                            f"rank={r} src={s} off={dummy_off} "
                            f"DUMMY(odd-recv) "
                            f"expected={hex(_W4_P1VCFR_DUMMY_SENTINEL)} "
                            f"actual_first8="
                            f"{[hex(int(x) & 0xFFFFFFFF) for x in actual_first8]}"
                        )
        if mismatches:
            self.fail(
                "v66j var-count + fixed-region DATA MISMATCH:\n  "
                + "\n  ".join(mismatches)
            )


# ---------------------------------------------------------------------------
# v66k: Workaround #3 — minimum-pad to TILE. Skewed-input-derived counts (same
# pipeline as v66a: bucketize + sort + wave-split), then ROUND UP every per-pair
# count to at least TILE (no zero edges). All DMAs target the REAL region only;
# NO dummy region at all. Kernel structurally identical to v66d (variable counts
# + uniform region) but with TILE-padded skewed-input-derived counts. Predicted:
# PASS (degenerates to v66d).
# ---------------------------------------------------------------------------


def _w4_p1minpad_compute_padded_arrays(
    per_wave_traffic: np.ndarray,
    *,
    world_size: int,
    tile: int,
):
    """Round-up to TILE per-pair: zero-edges become exactly one TILE.

    Returns:
      eff_count[w, A, B] = max(traffic[w, A, B], TILE)
      send_off[w, A, B] = sum_{B' < B} eff_count[w, A, B']    (per-sender)
      recv_off[w, A, B] = sum_{A' < A} eff_count[w, A', B]    (per-receiver)
      per_wave_real     = max over (w, B) of sum_{A} eff_count[w, A, B]
    """
    max_waves = per_wave_traffic.shape[0]
    eff_count = np.maximum(per_wave_traffic, tile).astype(np.int32)
    send_off = np.zeros_like(eff_count)
    recv_off = np.zeros_like(eff_count)
    for w in range(max_waves):
        for A in range(world_size):
            cum = 0
            for B in range(world_size):
                send_off[w, A, B] = cum
                cum += int(eff_count[w, A, B])
        for B in range(world_size):
            cum = 0
            for A in range(world_size):
                recv_off[w, A, B] = cum
                cum += int(eff_count[w, A, B])
    per_wave_real = int(np.max(np.sum(eff_count, axis=1)))
    return eff_count, send_off, recv_off, per_wave_real


def _make_w4_p1minpad_kernel(world_size: int, max_waves: int,
                             per_wave_real: int, tile: int):
    """v66d-shaped kernel: variable counts, REAL region only, no dummy."""

    def kernel(
        sorted_row_ref,    # [max_waves * per_wave_real] int32 HBM
        eff_count_ref,     # [max_waves, W, W] int32 SMEM
        send_off_ref,      # [max_waves, W, W] int32 SMEM
        recv_off_ref,      # [max_waves, W, W] int32 SMEM
        out_recv_ref,      # [max_waves * per_wave_real] int32 HBM
        zero_int_vmem,     # [per_wave_real] int32 VMEM zero buffer
        barrier_sems,      # REGULAR (max_waves, 2)
        send_sem,          # DMA send sem
        recv_sem,          # DMA recv sem
    ):
        my_rank = lax.axis_index("x")
        zero_int_vmem[...] = jnp.zeros_like(zero_int_vmem)

        def wave_body(w, _carry):
            wave_recv_slot = out_recv_ref.at[
                pl.ds(w * per_wave_real, per_wave_real)
            ]
            pltpu.sync_copy(zero_int_vmem, wave_recv_slot)

            pre_sem = barrier_sems.at[w, 0]
            for peer in range(world_size):
                pl.semaphore_signal(
                    pre_sem, 1,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
            pl.semaphore_wait(pre_sem, world_size)

            for peer in range(world_size):
                cnt = pl.multiple_of(
                    eff_count_ref[w, my_rank, peer], tile,
                )
                send_start = pl.multiple_of(
                    w * per_wave_real + send_off_ref[w, my_rank, peer], tile,
                )
                remote_recv_start = pl.multiple_of(
                    w * per_wave_real + recv_off_ref[w, peer, my_rank], tile,
                )
                d = pltpu.make_async_remote_copy(
                    src_ref=sorted_row_ref.at[pl.ds(send_start, cnt)],
                    dst_ref=out_recv_ref.at[pl.ds(remote_recv_start, cnt)],
                    send_sem=send_sem, recv_sem=recv_sem,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
                d.start()
                d.wait()

            post_sem = barrier_sems.at[w, 1]
            for peer in range(world_size):
                pl.semaphore_signal(
                    post_sem, 1,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
            pl.semaphore_wait(post_sem, world_size)
            return None

        lax.fori_loop(0, max_waves, wave_body, None)

    return kernel


def _w4_p1minpad_sentinel(w: int, src: int, dst: int) -> int:
    return 0x0C000000 + (w << 16) + (src << 8) + dst


def _run_w4_p1minpad_litmus():
    """v66k: min-pad-to-TILE workaround on skewed inputs."""
    from recir.ops.pallas.embeddings.jagged_a2a_lookup import (
        _bucketize_and_sort,
        _max_waves_for_capacity,
        _round_up_tile,
        _split_traffic_into_waves,
    )
    from recir.ops.pallas.embeddings.tests.test_jagged_a2a_lookup import (
        _make_skewed_inputs,
    )

    world_size = _W4_P1SD_WORLD_SIZE
    tile = _W4_P1SD_TILE
    max_local_ids = _W4_P1SD_MAX_LOCAL_IDS
    recv_capacity = _W4_P1SD_RECV_CAPACITY
    v_local = _W4_P1SD_VLOCAL

    devices = jax.devices()[:world_size]
    mesh = jax.sharding.Mesh(np.array(devices), ("x",))

    local_ids_np, local_id_count_np = _make_skewed_inputs(
        world_size, v_local, max_local_ids, seed=_W4_P1SD_SEED,
    )
    send_counts_padded_per_rank = []
    for r in range(world_size):
        _sr, send_counts_real, _sp, _rp = _bucketize_and_sort(
            jnp.array(local_ids_np[r]),
            jnp.array(local_id_count_np[r]),
            max_local_ids, world_size,
        )
        send_counts_padded = _round_up_tile(send_counts_real, tile)
        send_counts_padded_per_rank.append(np.array(send_counts_padded))

    traffic_matrix = jnp.array(
        np.stack(send_counts_padded_per_rank), dtype=jnp.int32,
    )
    max_waves = _max_waves_for_capacity(
        max_local_ids, world_size, recv_capacity,
    )
    per_wave_traffic, _send_offs, _recv_offs = _split_traffic_into_waves(
        traffic_matrix,
        recv_capacity=recv_capacity,
        max_waves=max_waves,
        world_size=world_size,
    )
    per_wave_traffic_np = np.array(per_wave_traffic)

    eff_count_np, eff_send_off_np, eff_recv_off_np, per_wave_real = (
        _w4_p1minpad_compute_padded_arrays(
            per_wave_traffic_np, world_size=world_size, tile=tile,
        )
    )
    out_recv_size = max_waves * per_wave_real
    sender_per_wave = int(np.max(np.sum(eff_count_np, axis=2)))
    sender_total = max_waves * sender_per_wave

    # Sender buffer: each sender writes wave-distinct sentinels into per-wave
    # slot at offset `w * per_wave_real + send_off[w, src, dst]`. Allocate
    # `max_waves * per_wave_real` to share the layout with recv buffer (so
    # send_off and recv_off can both index into the same wave slot).
    sorted_row_np = np.zeros(
        (world_size, out_recv_size), dtype=np.int32,
    )
    for r in range(world_size):
        for w in range(max_waves):
            for p in range(world_size):
                start = w * per_wave_real + int(eff_send_off_np[w, r, p])
                cnt = int(eff_count_np[w, r, p])
                sorted_row_np[r, start:start + cnt] = (
                    _w4_p1minpad_sentinel(w, r, p)
                )

    HBM = pltpu.MemorySpace.HBM
    SMEM = pltpu.MemorySpace.SMEM

    kernel = _make_w4_p1minpad_kernel(
        world_size=world_size, max_waves=max_waves,
        per_wave_real=per_wave_real, tile=tile,
    )

    P = jax.sharding.PartitionSpec

    def per_device(src, eff_count, send_off, recv_off):
        recv = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((out_recv_size,), jnp.int32),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[
                    pl.BlockSpec(memory_space=HBM),
                    pl.BlockSpec(memory_space=SMEM),
                    pl.BlockSpec(memory_space=SMEM),
                    pl.BlockSpec(memory_space=SMEM),
                ],
                out_specs=pl.BlockSpec(memory_space=HBM),
                scratch_shapes=(
                    pltpu.VMEM((per_wave_real,), jnp.int32),
                    pltpu.SemaphoreType.REGULAR((max_waves, 2)),
                    pltpu.SemaphoreType.DMA,
                    pltpu.SemaphoreType.DMA,
                ),
            ),
            interpret=False,
        )(src[0], eff_count, send_off, recv_off)
        return recv[None, ...]

    sharded = jax.shard_map(
        per_device, mesh=mesh,
        in_specs=(P("x"), P(), P(), P()),
        out_specs=P("x"),
        check_vma=False,
    )
    sharding_x = jax.sharding.NamedSharding(mesh, P("x"))
    sharding_rep = jax.sharding.NamedSharding(mesh, P())
    src = jax.device_put(jnp.array(sorted_row_np), sharding_x)
    eff_count = jax.device_put(jnp.array(eff_count_np), sharding_rep)
    send_off = jax.device_put(jnp.array(eff_send_off_np), sharding_rep)
    recv_off = jax.device_put(jnp.array(eff_recv_off_np), sharding_rep)
    out = sharded(src, eff_count, send_off, recv_off)
    return (
        np.array(out),
        per_wave_traffic_np,
        eff_count_np,
        eff_recv_off_np,
        max_waves,
        per_wave_real,
        world_size,
        tile,
    )


@unittest.skipUnless(_has_tpu(), "Litmus test requires real TPU")
class W4Phase1MinPadLitmus(unittest.TestCase):
    """v66k: skewed-input-derived counts, MIN-PAD to TILE, real region only."""

    def setUp(self) -> None:
        if jax.device_count() < 4:
            self.skipTest(f"Need 4 devices, have {jax.device_count()}")

    def test_w4_p1_min_pad(self) -> None:
        import threading
        result_box: list = []
        error_box: list = []

        def runner():
            try:
                result_box.append(_run_w4_p1minpad_litmus())
            except BaseException as e:  # noqa: BLE001
                error_box.append(e)

        t = threading.Thread(target=runner, daemon=True)
        t.start()
        t.join(timeout=300.0)
        if t.is_alive():
            self.fail(
                "v66k (W=4 phase-1 min-pad-to-TILE workaround) HUNG after "
                "300s; min-pad workaround does NOT lift the deadlock."
            )
        if error_box:
            raise error_box[0]

        (recv, traffic_np, eff_count_np, recv_off_np, max_waves, per_wave_real,
         world_size, tile) = result_box[0]
        print(
            f"[LITMUS_W4_P1MINPAD] max_waves={max_waves} "
            f"per_wave_real={per_wave_real} world_size={world_size} "
            f"tile={tile}",
            flush=True,
        )
        for r in range(world_size):
            for w in range(max_waves):
                for s in range(world_size):
                    cnt = int(eff_count_np[w, s, r])
                    off = w * per_wave_real + int(recv_off_np[w, s, r])
                    first4 = recv[r, off:off + 4].tolist()
                    print(
                        f"[LITMUS_W4_P1MINPAD] rank={r} wave={w} src={s} "
                        f"cnt={cnt} off={off} first4="
                        f"{[hex(int(x)) for x in first4]}",
                        flush=True,
                    )
        mismatches = []
        for r in range(world_size):
            for w in range(max_waves):
                for s in range(world_size):
                    cnt = int(eff_count_np[w, s, r])
                    off = w * per_wave_real + int(recv_off_np[w, s, r])
                    expected = _w4_p1minpad_sentinel(w, s, r)
                    slot = recv[r, off:off + cnt]
                    if not np.all(slot == expected):
                        actual_first8 = slot[:8].tolist()
                        mismatches.append(
                            f"rank={r} wave={w} src={s} cnt={cnt} off={off} "
                            f"expected={hex(expected)} actual_first8="
                            f"{[hex(int(x)) for x in actual_first8]}"
                        )
        if mismatches:
            self.fail(
                "v66k min-pad DATA MISMATCH:\n  " + "\n  ".join(mismatches)
            )


# ---------------------------------------------------------------------------
# v66l: Workaround #2 — split into two homogeneous Python loops. Synthetic
# fixed traffic where every rank sees same real/dummy peer split: peers [0, 2]
# REAL with variable counts (256, 384), peers [1, 3] DUMMY with uniform TILE
# count each. Two separate `for peer in [...]` loops in the same wave_body —
# each loop is uniform-region. Predicted: PASS (uncertain — v66j hung with one
# loop containing compile-time per-peer region split; v66l tests whether
# physically separate loops are different).
# ---------------------------------------------------------------------------


_W4_P1SPLIT_TILE = 128
_W4_P1SPLIT_MAX_WAVES = 2
_W4_P1SPLIT_WORLD_SIZE = 4
# Variable real counts per real peer (in iteration order [0, 2]).
_W4_P1SPLIT_REAL_PEERS = (0, 2)
_W4_P1SPLIT_DUMMY_PEERS = (1, 3)
_W4_P1SPLIT_REAL_CNT = (256, 384)  # peer 0 -> 256, peer 2 -> 384
_W4_P1SPLIT_DUMMY_CNT = 128        # uniform TILE per dummy peer
# Real region per receiver (per wave): senders write at slot indexed by sender
# rank. Slot stride = max real count = 384. Per-wave real region per receiver
# = 4 * 384 = 1536 ints. Two waves -> 3072 ints per receiver real region.
_W4_P1SPLIT_REAL_SLOT = 384
_W4_P1SPLIT_PER_WAVE_REAL = (
    _W4_P1SPLIT_WORLD_SIZE * _W4_P1SPLIT_REAL_SLOT
)  # 1536
_W4_P1SPLIT_RECV_CAPACITY = (
    _W4_P1SPLIT_MAX_WAVES * _W4_P1SPLIT_PER_WAVE_REAL
)  # 3072
# Dummy region: per-receiver, slot stride = TILE, indexed by sender rank.
_W4_P1SPLIT_DUMMY_REGION_SIZE = (
    _W4_P1SPLIT_WORLD_SIZE * _W4_P1SPLIT_DUMMY_CNT
)  # 512
_W4_P1SPLIT_OUT_RECV_COMBINED_SIZE = (
    _W4_P1SPLIT_RECV_CAPACITY + _W4_P1SPLIT_DUMMY_REGION_SIZE
)
# Sender-side sorted_row buffer (real region only): per wave per real peer
# write at offset w*per_wave_real + (real_peer_idx)*real_slot. Use full real
# slot stride so each peer's slot is independent.
_W4_P1SPLIT_MAX_PADDED = _W4_P1SPLIT_RECV_CAPACITY  # mirror layout
_W4_P1SPLIT_DUMMY_SRC_SIZE = (
    _W4_P1SPLIT_WORLD_SIZE * _W4_P1SPLIT_DUMMY_CNT
)
_W4_P1SPLIT_DUMMY_SENTINEL = 0xDEADBEEF


def _w4_p1split_sentinel(w: int, src: int, dst: int) -> int:
    return 0x0D000000 + (w << 16) + (src << 8) + dst


def _make_w4_p1split_kernel():
    world_size = _W4_P1SPLIT_WORLD_SIZE
    max_waves = _W4_P1SPLIT_MAX_WAVES
    per_wave_real = _W4_P1SPLIT_PER_WAVE_REAL
    real_slot = _W4_P1SPLIT_REAL_SLOT
    dummy_cnt = _W4_P1SPLIT_DUMMY_CNT
    recv_capacity = _W4_P1SPLIT_RECV_CAPACITY
    tile = _W4_P1SPLIT_TILE
    real_peers = _W4_P1SPLIT_REAL_PEERS
    dummy_peers = _W4_P1SPLIT_DUMMY_PEERS
    real_cnts = _W4_P1SPLIT_REAL_CNT

    def kernel(
        sorted_row_ref,         # [_W4_P1SPLIT_MAX_PADDED] int32 HBM
        dummy_src_ref,          # [_W4_P1SPLIT_DUMMY_SRC_SIZE] int32 HBM
        eff_real_count_ref,     # [max_waves, W, len(real_peers)] int32 SMEM
        out_recv_combined_ref,  # [_W4_P1SPLIT_OUT_RECV_COMBINED_SIZE] int32 HBM
        zero_int_vmem,          # [per_wave_real] int32 VMEM zero buffer
        barrier_sems,           # REGULAR (max_waves, 2)
        send_sem,               # DMA send sem
        recv_sem,               # DMA recv sem
    ):
        my_rank = lax.axis_index("x")
        zero_int_vmem[...] = jnp.zeros_like(zero_int_vmem)

        def wave_body(w, _carry):
            # Per-wave slot-only zero-fill of REAL region.
            wave_real_slot = out_recv_combined_ref.at[
                pl.ds(w * per_wave_real, per_wave_real)
            ]
            pltpu.sync_copy(zero_int_vmem, wave_real_slot)

            pre_sem = barrier_sems.at[w, 0]
            for peer in range(world_size):
                pl.semaphore_signal(
                    pre_sem, 1,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
            pl.semaphore_wait(pre_sem, world_size)

            # FIRST homogeneous loop: REAL peers only. Variable counts. All
            # DMAs target REAL region.
            for peer_idx, peer in enumerate(real_peers):
                cnt = pl.multiple_of(
                    eff_real_count_ref[w, my_rank, peer_idx], tile,
                )
                # Sender slot for this real peer is contiguous: each real peer
                # gets its own per-wave slot at index peer_idx within the
                # sender's per-wave_real chunk.
                send_start = pl.multiple_of(
                    w * per_wave_real + peer_idx * real_slot, tile,
                )
                # Receiver slot indexed by sender rank within per-wave real
                # region (stride = real_slot = 384).
                remote_recv_start = pl.multiple_of(
                    w * per_wave_real + my_rank * real_slot, tile,
                )
                src = sorted_row_ref.at[pl.ds(send_start, cnt)]
                dst = out_recv_combined_ref.at[
                    pl.ds(remote_recv_start, cnt)
                ]
                d = pltpu.make_async_remote_copy(
                    src_ref=src, dst_ref=dst,
                    send_sem=send_sem, recv_sem=recv_sem,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
                d.start()
                d.wait()

            # SECOND homogeneous loop: DUMMY peers only. Uniform TILE count.
            # All DMAs target DUMMY region (sender-rank indexed slot).
            for peer in dummy_peers:
                src = dummy_src_ref.at[pl.ds(peer * dummy_cnt, dummy_cnt)]
                dst = out_recv_combined_ref.at[
                    pl.ds(recv_capacity + my_rank * dummy_cnt, dummy_cnt)
                ]
                d = pltpu.make_async_remote_copy(
                    src_ref=src, dst_ref=dst,
                    send_sem=send_sem, recv_sem=recv_sem,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
                d.start()
                d.wait()

            post_sem = barrier_sems.at[w, 1]
            for peer in range(world_size):
                pl.semaphore_signal(
                    post_sem, 1,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
            pl.semaphore_wait(post_sem, world_size)
            return None

        lax.fori_loop(0, max_waves, wave_body, None)

    return kernel


def _run_w4_p1split_litmus():
    world_size = _W4_P1SPLIT_WORLD_SIZE
    max_waves = _W4_P1SPLIT_MAX_WAVES
    real_slot = _W4_P1SPLIT_REAL_SLOT
    per_wave_real = _W4_P1SPLIT_PER_WAVE_REAL
    recv_capacity = _W4_P1SPLIT_RECV_CAPACITY
    dummy_cnt = _W4_P1SPLIT_DUMMY_CNT
    tile = _W4_P1SPLIT_TILE
    real_peers = _W4_P1SPLIT_REAL_PEERS
    real_cnts = _W4_P1SPLIT_REAL_CNT

    devices = jax.devices()[:world_size]
    mesh = jax.sharding.Mesh(np.array(devices), ("x",))

    # eff_real_count[w, my_rank, peer_idx] = real_cnts[peer_idx] for all
    # waves and senders.
    eff_real_count_np = np.zeros(
        (max_waves, world_size, len(real_peers)), dtype=np.int32,
    )
    for w in range(max_waves):
        for s in range(world_size):
            for i, _peer in enumerate(real_peers):
                eff_real_count_np[w, s, i] = real_cnts[i]

    # Sender sorted_row: per (wave, real_peer_idx) fill slot at
    # [w*per_wave_real + peer_idx*real_slot, +real_cnts[peer_idx]) with
    # wave/src/dst sentinel.
    sorted_row_np = np.zeros(
        (world_size, _W4_P1SPLIT_MAX_PADDED), dtype=np.int32,
    )
    for r in range(world_size):
        for w in range(max_waves):
            for i, peer in enumerate(real_peers):
                start = w * per_wave_real + i * real_slot
                end = start + real_cnts[i]
                sorted_row_np[r, start:end] = (
                    _w4_p1split_sentinel(w, r, peer)
                )

    dummy_src_np = np.full(
        (world_size, _W4_P1SPLIT_DUMMY_SRC_SIZE),
        _W4_P1SPLIT_DUMMY_SENTINEL, dtype=np.uint32,
    ).view(np.int32)

    HBM = pltpu.MemorySpace.HBM
    SMEM = pltpu.MemorySpace.SMEM

    kernel = _make_w4_p1split_kernel()

    P = jax.sharding.PartitionSpec

    def per_device(src, dummy_src, eff_real_count):
        recv = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(
                (_W4_P1SPLIT_OUT_RECV_COMBINED_SIZE,), jnp.int32,
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[
                    pl.BlockSpec(memory_space=HBM),
                    pl.BlockSpec(memory_space=HBM),
                    pl.BlockSpec(memory_space=SMEM),
                ],
                out_specs=pl.BlockSpec(memory_space=HBM),
                scratch_shapes=(
                    pltpu.VMEM((per_wave_real,), jnp.int32),
                    pltpu.SemaphoreType.REGULAR((max_waves, 2)),
                    pltpu.SemaphoreType.DMA,
                    pltpu.SemaphoreType.DMA,
                ),
            ),
            interpret=False,
        )(src[0], dummy_src[0], eff_real_count)
        return recv[None, ...]

    sharded = jax.shard_map(
        per_device, mesh=mesh,
        in_specs=(P("x"), P("x"), P()),
        out_specs=P("x"),
        check_vma=False,
    )
    sharding_x = jax.sharding.NamedSharding(mesh, P("x"))
    sharding_rep = jax.sharding.NamedSharding(mesh, P())
    src = jax.device_put(jnp.array(sorted_row_np), sharding_x)
    dummy_src = jax.device_put(jnp.array(dummy_src_np), sharding_x)
    eff_real_count = jax.device_put(jnp.array(eff_real_count_np), sharding_rep)
    out = sharded(src, dummy_src, eff_real_count)
    return (
        np.array(out),
        max_waves,
        per_wave_real,
        real_slot,
        recv_capacity,
        dummy_cnt,
        world_size,
        tile,
    )


@unittest.skipUnless(_has_tpu(), "Litmus test requires real TPU")
class W4Phase1SplitLoopsLitmus(unittest.TestCase):
    """v66l: split into two homogeneous Python loops (real + dummy)."""

    def setUp(self) -> None:
        if jax.device_count() < 4:
            self.skipTest(f"Need 4 devices, have {jax.device_count()}")

    def test_w4_p1_split_loops(self) -> None:
        import threading
        result_box: list = []
        error_box: list = []

        def runner():
            try:
                result_box.append(_run_w4_p1split_litmus())
            except BaseException as e:  # noqa: BLE001
                error_box.append(e)

        t = threading.Thread(target=runner, daemon=True)
        t.start()
        t.join(timeout=300.0)
        if t.is_alive():
            self.fail(
                "v66l (W=4 phase-1 split-loops workaround) HUNG after 300s; "
                "splitting real and dummy peers into two physically separate "
                "Python loops does NOT lift the deadlock."
            )
        if error_box:
            raise error_box[0]

        (recv, max_waves, per_wave_real, real_slot, recv_capacity, dummy_cnt,
         world_size, tile) = result_box[0]
        print(
            f"[LITMUS_W4_P1SPLIT] max_waves={max_waves} "
            f"per_wave_real={per_wave_real} real_slot={real_slot} "
            f"recv_capacity={recv_capacity} dummy_cnt={dummy_cnt} "
            f"world_size={world_size} tile={tile}",
            flush=True,
        )
        # Diagnostics: real region per (rank, wave, src) and dummy region
        # per (rank, src).
        for r in range(world_size):
            for w in range(max_waves):
                for s in range(world_size):
                    off = w * per_wave_real + s * real_slot
                    first4 = recv[r, off:off + 4].tolist()
                    print(
                        f"[LITMUS_W4_P1SPLIT] rank={r} wave={w} src={s} "
                        f"REAL off={off} first4="
                        f"{[hex(int(x)) for x in first4]}",
                        flush=True,
                    )
            for s in range(world_size):
                off = recv_capacity + s * dummy_cnt
                first4 = recv[r, off:off + 4].tolist()
                print(
                    f"[LITMUS_W4_P1SPLIT] rank={r} DUMMY src={s} off={off} "
                    f"first4={[hex(int(x) & 0xFFFFFFFF) for x in first4]}",
                    flush=True,
                )

        # Verification: every receiver gets real writes from EVERY sender
        # for each wave (real_peers = [0, 2] are the destination peers, but
        # in the kernel `peer` is the destination, so receiver r gets writes
        # from sender s iff r is one of real_peers — i.e., r in {0, 2}).
        # Dummy: receiver r gets dummy from every sender iff r in {1, 3}.
        expected_dummy = np.array(
            [_W4_P1SPLIT_DUMMY_SENTINEL], dtype=np.uint32,
        ).view(np.int32)[0]
        mismatches = []
        for r in range(world_size):
            for w in range(max_waves):
                for s in range(world_size):
                    off = w * per_wave_real + s * real_slot
                    if r in _W4_P1SPLIT_REAL_PEERS:
                        # r is real -> sender s writes (s, r=peer) sentinel
                        # at slot indexed by sender s. Count = real_cnts of
                        # the entry where r matches the real peer index.
                        i = _W4_P1SPLIT_REAL_PEERS.index(r)
                        cnt = _W4_P1SPLIT_REAL_CNT[i]
                        expected = _w4_p1split_sentinel(w, s, r)
                        slot = recv[r, off:off + cnt]
                        if not np.all(slot == expected):
                            actual_first8 = slot[:8].tolist()
                            mismatches.append(
                                f"rank={r} wave={w} src={s} cnt={cnt} "
                                f"off={off} REAL expected={hex(expected)} "
                                f"actual_first8="
                                f"{[hex(int(x)) for x in actual_first8]}"
                            )
            if r in _W4_P1SPLIT_DUMMY_PEERS:
                for s in range(world_size):
                    off = recv_capacity + s * dummy_cnt
                    slot = recv[r, off:off + dummy_cnt]
                    if not np.all(slot == expected_dummy):
                        actual_first8 = slot[:8].tolist()
                        mismatches.append(
                            f"rank={r} src={s} off={off} DUMMY "
                            f"expected={hex(_W4_P1SPLIT_DUMMY_SENTINEL)} "
                            f"actual_first8="
                            f"{[hex(int(x) & 0xFFFFFFFF) for x in actual_first8]}"
                        )
        if mismatches:
            self.fail(
                "v66l split-loops DATA MISMATCH:\n  " + "\n  ".join(mismatches)
            )


# ---------------------------------------------------------------------------
# v66m: Workaround #1 — uniform-pad to MAX_PER_PAIR. Skewed-input-derived
# counts, then pad EVERY per-pair count to MAX_PER_PAIR = max(per_wave_traffic).
# All counts uniform. All DMAs target the REAL region only. Kernel structurally
# identical to v60 (uniform counts + uniform region) but with bigger arrays.
# Predicted: PASS (essentially guaranteed by v60).
# ---------------------------------------------------------------------------


def _w4_p1unipad_compute_arrays(
    per_wave_traffic: np.ndarray,
    *,
    world_size: int,
    tile: int,
):
    """Pad every per-pair count to MAX_PER_PAIR (max over all entries).

    Returns:
      max_per_pair: int (TILE-aligned, since traffic is TILE-aligned)
      eff_count[w, A, B] = max_per_pair (uniform)
      send_off[w, A, B] = B * max_per_pair       (per-sender)
      recv_off[w, A, B] = A * max_per_pair       (per-receiver)
      per_wave_real     = world_size * max_per_pair
    """
    max_per_pair = int(np.max(per_wave_traffic))
    if max_per_pair == 0:
        max_per_pair = tile
    # Round up to TILE just in case.
    max_per_pair = ((max_per_pair + tile - 1) // tile) * tile
    max_waves = per_wave_traffic.shape[0]
    eff_count = np.full(
        (max_waves, world_size, world_size), max_per_pair, dtype=np.int32,
    )
    send_off = np.zeros_like(eff_count)
    recv_off = np.zeros_like(eff_count)
    for w in range(max_waves):
        for A in range(world_size):
            for B in range(world_size):
                send_off[w, A, B] = B * max_per_pair
                recv_off[w, A, B] = A * max_per_pair
    per_wave_real = world_size * max_per_pair
    return max_per_pair, eff_count, send_off, recv_off, per_wave_real


def _make_w4_p1unipad_kernel(world_size: int, max_waves: int,
                             per_wave_real: int, max_per_pair: int,
                             tile: int):
    """v60-shape kernel: uniform per-pair counts, REAL region only."""

    def kernel(
        sorted_row_ref,    # [max_waves * per_wave_real] int32 HBM
        traffic_ref,       # [max_waves, W, W] int32 SMEM (uniform max_per_pair)
        send_off_ref,      # [max_waves, W, W] int32 SMEM
        recv_off_ref,      # [max_waves, W, W] int32 SMEM
        out_recv_ref,      # [max_waves * per_wave_real] int32 HBM
        zero_int_vmem,     # [per_wave_real] int32 VMEM zero buffer
        barrier_sems,
        send_sem,
        recv_sem,
    ):
        my_rank = lax.axis_index("x")
        zero_int_vmem[...] = jnp.zeros_like(zero_int_vmem)

        def wave_body(w, _carry):
            wave_recv_slot = out_recv_ref.at[
                pl.ds(w * per_wave_real, per_wave_real)
            ]
            pltpu.sync_copy(zero_int_vmem, wave_recv_slot)

            pre_sem = barrier_sems.at[w, 0]
            for peer in range(world_size):
                pl.semaphore_signal(
                    pre_sem, 1,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
            pl.semaphore_wait(pre_sem, world_size)

            for peer in range(world_size):
                cnt = pl.multiple_of(
                    traffic_ref[w, my_rank, peer], tile,
                )
                send_start = pl.multiple_of(
                    w * per_wave_real + send_off_ref[w, my_rank, peer], tile,
                )
                remote_recv_start = pl.multiple_of(
                    w * per_wave_real + recv_off_ref[w, peer, my_rank], tile,
                )
                d = pltpu.make_async_remote_copy(
                    src_ref=sorted_row_ref.at[pl.ds(send_start, cnt)],
                    dst_ref=out_recv_ref.at[pl.ds(remote_recv_start, cnt)],
                    send_sem=send_sem, recv_sem=recv_sem,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
                d.start()
                d.wait()

            post_sem = barrier_sems.at[w, 1]
            for peer in range(world_size):
                pl.semaphore_signal(
                    post_sem, 1,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
            pl.semaphore_wait(post_sem, world_size)
            return None

        lax.fori_loop(0, max_waves, wave_body, None)

    return kernel


def _w4_p1unipad_sentinel(w: int, src: int, dst: int) -> int:
    return 0x0E000000 + (w << 16) + (src << 8) + dst


def _run_w4_p1unipad_litmus():
    """v66m: uniform-pad-to-MAX-PER-PAIR workaround on skewed inputs."""
    from recir.ops.pallas.embeddings.jagged_a2a_lookup import (
        _bucketize_and_sort,
        _max_waves_for_capacity,
        _round_up_tile,
        _split_traffic_into_waves,
    )
    from recir.ops.pallas.embeddings.tests.test_jagged_a2a_lookup import (
        _make_skewed_inputs,
    )

    world_size = _W4_P1SD_WORLD_SIZE
    tile = _W4_P1SD_TILE
    max_local_ids = _W4_P1SD_MAX_LOCAL_IDS
    recv_capacity = _W4_P1SD_RECV_CAPACITY
    v_local = _W4_P1SD_VLOCAL

    devices = jax.devices()[:world_size]
    mesh = jax.sharding.Mesh(np.array(devices), ("x",))

    local_ids_np, local_id_count_np = _make_skewed_inputs(
        world_size, v_local, max_local_ids, seed=_W4_P1SD_SEED,
    )
    send_counts_padded_per_rank = []
    for r in range(world_size):
        _sr, send_counts_real, _sp, _rp = _bucketize_and_sort(
            jnp.array(local_ids_np[r]),
            jnp.array(local_id_count_np[r]),
            max_local_ids, world_size,
        )
        send_counts_padded = _round_up_tile(send_counts_real, tile)
        send_counts_padded_per_rank.append(np.array(send_counts_padded))

    traffic_matrix = jnp.array(
        np.stack(send_counts_padded_per_rank), dtype=jnp.int32,
    )
    max_waves = _max_waves_for_capacity(
        max_local_ids, world_size, recv_capacity,
    )
    per_wave_traffic, _send_offs, _recv_offs = _split_traffic_into_waves(
        traffic_matrix,
        recv_capacity=recv_capacity,
        max_waves=max_waves,
        world_size=world_size,
    )
    per_wave_traffic_np = np.array(per_wave_traffic)

    (max_per_pair, eff_count_np, eff_send_off_np, eff_recv_off_np,
     per_wave_real) = _w4_p1unipad_compute_arrays(
        per_wave_traffic_np, world_size=world_size, tile=tile,
    )
    out_recv_size = max_waves * per_wave_real

    sorted_row_np = np.zeros(
        (world_size, out_recv_size), dtype=np.int32,
    )
    for r in range(world_size):
        for w in range(max_waves):
            for p in range(world_size):
                start = w * per_wave_real + int(eff_send_off_np[w, r, p])
                cnt = max_per_pair
                sorted_row_np[r, start:start + cnt] = (
                    _w4_p1unipad_sentinel(w, r, p)
                )

    HBM = pltpu.MemorySpace.HBM
    SMEM = pltpu.MemorySpace.SMEM

    kernel = _make_w4_p1unipad_kernel(
        world_size=world_size, max_waves=max_waves,
        per_wave_real=per_wave_real, max_per_pair=max_per_pair, tile=tile,
    )

    P = jax.sharding.PartitionSpec

    def per_device(src, traffic, send_off, recv_off):
        recv = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((out_recv_size,), jnp.int32),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[
                    pl.BlockSpec(memory_space=HBM),
                    pl.BlockSpec(memory_space=SMEM),
                    pl.BlockSpec(memory_space=SMEM),
                    pl.BlockSpec(memory_space=SMEM),
                ],
                out_specs=pl.BlockSpec(memory_space=HBM),
                scratch_shapes=(
                    pltpu.VMEM((per_wave_real,), jnp.int32),
                    pltpu.SemaphoreType.REGULAR((max_waves, 2)),
                    pltpu.SemaphoreType.DMA,
                    pltpu.SemaphoreType.DMA,
                ),
            ),
            interpret=False,
        )(src[0], traffic, send_off, recv_off)
        return recv[None, ...]

    sharded = jax.shard_map(
        per_device, mesh=mesh,
        in_specs=(P("x"), P(), P(), P()),
        out_specs=P("x"),
        check_vma=False,
    )
    sharding_x = jax.sharding.NamedSharding(mesh, P("x"))
    sharding_rep = jax.sharding.NamedSharding(mesh, P())
    src = jax.device_put(jnp.array(sorted_row_np), sharding_x)
    traffic = jax.device_put(jnp.array(eff_count_np), sharding_rep)
    send_off = jax.device_put(jnp.array(eff_send_off_np), sharding_rep)
    recv_off = jax.device_put(jnp.array(eff_recv_off_np), sharding_rep)
    out = sharded(src, traffic, send_off, recv_off)
    return (
        np.array(out),
        per_wave_traffic_np,
        max_per_pair,
        max_waves,
        per_wave_real,
        world_size,
        tile,
    )


@unittest.skipUnless(_has_tpu(), "Litmus test requires real TPU")
class W4Phase1UniformPadLitmus(unittest.TestCase):
    """v66m: skewed-input-derived counts, UNIFORM-PAD to MAX_PER_PAIR."""

    def setUp(self) -> None:
        if jax.device_count() < 4:
            self.skipTest(f"Need 4 devices, have {jax.device_count()}")

    def test_w4_p1_uniform_pad(self) -> None:
        import threading
        result_box: list = []
        error_box: list = []

        def runner():
            try:
                result_box.append(_run_w4_p1unipad_litmus())
            except BaseException as e:  # noqa: BLE001
                error_box.append(e)

        t = threading.Thread(target=runner, daemon=True)
        t.start()
        t.join(timeout=300.0)
        if t.is_alive():
            self.fail(
                "v66m (W=4 phase-1 uniform-pad-to-MAX_PER_PAIR workaround) "
                "HUNG after 300s; uniform-pad workaround does NOT lift the "
                "deadlock — invalidates v60 generalization."
            )
        if error_box:
            raise error_box[0]

        (recv, traffic_np, max_per_pair, max_waves, per_wave_real, world_size,
         tile) = result_box[0]
        print(
            f"[LITMUS_W4_P1UNIPAD] max_per_pair={max_per_pair} "
            f"max_waves={max_waves} per_wave_real={per_wave_real} "
            f"world_size={world_size} tile={tile}",
            flush=True,
        )
        for r in range(world_size):
            for w in range(max_waves):
                for s in range(world_size):
                    off = w * per_wave_real + s * max_per_pair
                    first4 = recv[r, off:off + 4].tolist()
                    print(
                        f"[LITMUS_W4_P1UNIPAD] rank={r} wave={w} src={s} "
                        f"off={off} first4="
                        f"{[hex(int(x)) for x in first4]}",
                        flush=True,
                    )
        # Verification: every (w, s, r) wrote MAX_PER_PAIR copies of the
        # uniform sentinel at offset w*per_wave_real + s*max_per_pair.
        mismatches = []
        for r in range(world_size):
            for w in range(max_waves):
                for s in range(world_size):
                    off = w * per_wave_real + s * max_per_pair
                    expected = _w4_p1unipad_sentinel(w, s, r)
                    slot = recv[r, off:off + max_per_pair]
                    if not np.all(slot == expected):
                        actual_first8 = slot[:8].tolist()
                        mismatches.append(
                            f"rank={r} wave={w} src={s} cnt={max_per_pair} "
                            f"off={off} expected={hex(expected)} "
                            f"actual_first8="
                            f"{[hex(int(x)) for x in actual_first8]}"
                        )
        if mismatches:
            self.fail(
                "v66m uniform-pad DATA MISMATCH:\n  "
                + "\n  ".join(mismatches)
            )


# ---------------------------------------------------------------------------
# v66c: W=4 phase-1 ONLY litmus — v60's exact balanced kernel + UNCONDITIONAL
# dummy-region DMAs. Isolation goal: does adding writes to the dummy region
# alone (everything else identical to v60) trigger the multi-iter hang?
#
# v60 (W4UnconditionalDMATightenedLitmus) PASSes: balanced 128-per-pair, no
#   dummy region, 4 real DMAs per wave per rank.
# v66a (W4Phase1SkewedDummyLitmus) HANGs: skewed traffic + dummy-offset
#   rewrite + extended buffer.
#
# v66c keeps v60's traffic (balanced 128-per-pair, wave-distinct sentinels,
# wave-distinct recv offsets, per-wave zero-fill) and adds, after the real
# DMAs, ONE extra DMA per peer that writes a constant sentinel into the
# extended recv buffer's dummy region (slot for our rank on the peer's recv
# buffer). 8 DMAs per wave per rank (4 real + 4 dummy), all using the same
# shared send_sem/recv_sem.
#
# Stop conditions:
#   PASS  -> dummy-region DMA writes alone are NOT the culprit; v66a's hang
#            is from variable per-peer DMA sizes. Spawn v66d (balanced
#            VARIABLE counts, no dummy) next.
#   HANG  -> dummy-region DMA writes ARE the culprit; bug is something
#            specific to the extended buffer / 8-DMA-per-wave / dummy-region
#            access pattern.
# ---------------------------------------------------------------------------


_W4_P1D_TILE = 128
_W4_P1D_MAX_WAVES = 2
_W4_P1D_PER_PAIR_COUNT = 128  # one TILE per (src, dst) per wave
_W4_P1D_WORLD_SIZE = 4
_W4_P1D_PER_WAVE_RECV_SIZE = (
    _W4_P1D_WORLD_SIZE * _W4_P1D_PER_PAIR_COUNT
)  # 512 ints
_W4_P1D_RECV_CAPACITY = (
    _W4_P1D_MAX_WAVES * _W4_P1D_PER_WAVE_RECV_SIZE
)  # 1024 ints
_W4_P1D_MAX_PADDED = (
    _W4_P1D_MAX_WAVES * _W4_P1D_WORLD_SIZE * _W4_P1D_PER_PAIR_COUNT
)  # 1024 ints
_W4_P1D_DUMMY_SENTINEL = 0xDEADBEEF
# Dummy region: one TILE per source rank, appended to recv_capacity.
_W4_P1D_DUMMY_REGION_SIZE = _W4_P1D_WORLD_SIZE * _W4_P1D_TILE  # 512 ints
_W4_P1D_OUT_RECV_COMBINED_SIZE = (
    _W4_P1D_RECV_CAPACITY + _W4_P1D_DUMMY_REGION_SIZE
)  # 1536 ints
# Dummy src buffer: one TILE per peer (we always send the same constant).
_W4_P1D_DUMMY_SRC_SIZE = _W4_P1D_WORLD_SIZE * _W4_P1D_TILE  # 512 ints


def _w4_p1d_sentinel(w: int, src: int, dst: int) -> int:
    """Wave/source/dest-distinct sentinel: same form as v60."""
    return 0x0A000000 + (w << 16) + (src << 8) + dst


def _make_w4_p1d_kernel(world_size: int, max_waves: int, per_pair_count: int,
                        recv_capacity: int, tile: int):
    """v60's exact phase-1-only kernel + UNCONDITIONAL dummy-region DMAs.

    Per wave:
      0. PER-WAVE zero-fill of the WHOLE combined recv buffer
         ([recv_capacity + W*TILE]) via sync_copy.
      1. PRE-barrier (W-way REGULAR-sem fan-in).
      2a. UNCONDITIONAL per-peer REAL DMA — wave-distinct sentinels into
          wave-distinct recv slots. Identical to v60.
      2b. UNCONDITIONAL per-peer DUMMY DMA — constant sentinel into peer's
          dummy region slot for OUR rank.
      3. POST-barrier (W-way REGULAR-sem fan-in).
    Same shared send_sem/recv_sem for all 8 DMAs per wave.
    """
    per_wave_recv_size = world_size * per_pair_count

    def kernel(
        sorted_row_ref,        # [max_waves * W * per_pair_count] int32 HBM
        dummy_src_ref,         # [W * TILE] int32 HBM (constant sentinel)
        traffic_ref,           # [max_waves, W, W] int32 SMEM
        send_off_ref,          # [max_waves, W, W] int32 SMEM
        recv_off_ref,          # [max_waves, W, W] int32 SMEM
        out_recv_combined_ref, # [recv_capacity + W*TILE] int32 HBM
        zero_int_vmem,         # [recv_capacity + W*TILE] int32 VMEM zero
        barrier_sems,          # REGULAR (max_waves, 2)
        send_sem,              # DMA send sem
        recv_sem,              # DMA recv sem
    ):
        my_rank = lax.axis_index("x")

        # Initialize zero VMEM scratch once outside the loop.
        zero_int_vmem[...] = jnp.zeros_like(zero_int_vmem)

        def wave_body(w, _carry):
            # Phase 0: zero-fill the WHOLE combined recv buffer
            # (real region + dummy region both).
            pltpu.sync_copy(zero_int_vmem, out_recv_combined_ref)

            # Pre-barrier: signal every peer (incl. self), wait W signals.
            pre_sem = barrier_sems.at[w, 0]
            for peer in range(world_size):
                pl.semaphore_signal(
                    pre_sem, 1,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
            pl.semaphore_wait(pre_sem, world_size)

            # Phase 1a: REAL DMAs identical to v60 (balanced 128, all
            # targeting real region [0, recv_capacity)).
            for peer in range(world_size):
                send_count = pl.multiple_of(
                    traffic_ref[w, my_rank, peer], tile,
                )
                send_start = pl.multiple_of(
                    send_off_ref[w, my_rank, peer], tile,
                )
                remote_recv_start = pl.multiple_of(
                    recv_off_ref[w, peer, my_rank], tile,
                )
                d = pltpu.make_async_remote_copy(
                    src_ref=sorted_row_ref.at[
                        pl.ds(send_start, send_count)
                    ],
                    dst_ref=out_recv_combined_ref.at[
                        pl.ds(remote_recv_start, send_count)
                    ],
                    send_sem=send_sem, recv_sem=recv_sem,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
                d.start()
                d.wait()

            # Phase 1b: ADDITIONAL dummy DMAs — one per peer, unconditional,
            # into the peer's dummy region slot for our rank. Constant size
            # = TILE. This is the ONLY new variable vs v60.
            for peer in range(world_size):
                d = pltpu.make_async_remote_copy(
                    src_ref=dummy_src_ref.at[
                        pl.ds(peer * tile, tile)
                    ],
                    dst_ref=out_recv_combined_ref.at[
                        pl.ds(recv_capacity + my_rank * tile, tile)
                    ],
                    send_sem=send_sem, recv_sem=recv_sem,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
                d.start()
                d.wait()

            # Post-barrier: signal every peer (incl. self), wait W signals.
            post_sem = barrier_sems.at[w, 1]
            for peer in range(world_size):
                pl.semaphore_signal(
                    post_sem, 1,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
            pl.semaphore_wait(post_sem, world_size)
            return None

        lax.fori_loop(0, max_waves, wave_body, None)

    return kernel


def _run_w4_p1d_litmus():
    """Run W=4 phase-1 dummy-only multi-iter litmus.

    Returns ``(out_recv_combined_per_rank, max_waves, per_pair_count,
    per_wave_recv_size, recv_capacity, world_size, tile, dummy_sentinel)``.
    """
    world_size = _W4_P1D_WORLD_SIZE
    max_waves = _W4_P1D_MAX_WAVES
    per_pair_count = _W4_P1D_PER_PAIR_COUNT
    per_wave_recv_size = _W4_P1D_PER_WAVE_RECV_SIZE
    recv_capacity = _W4_P1D_RECV_CAPACITY
    tile = _W4_P1D_TILE
    devices = jax.devices()[:world_size]
    mesh = jax.sharding.Mesh(np.array(devices), ("x",))

    # Balanced traffic: every (s, t) sends exactly per_pair_count every wave.
    per_wave_traffic_np = np.full(
        (max_waves, world_size, world_size),
        per_pair_count, dtype=np.int32,
    )
    # Send/recv offsets identical to v60 (wave-distinct halves of recv_buf).
    per_wave_send_offsets_np = np.zeros(
        (max_waves, world_size, world_size), dtype=np.int32,
    )
    per_wave_recv_offsets_np = np.zeros(
        (max_waves, world_size, world_size), dtype=np.int32,
    )
    for w in range(max_waves):
        for s in range(world_size):
            for t in range(world_size):
                per_wave_send_offsets_np[w, s, t] = (
                    w * per_wave_recv_size + t * per_pair_count
                )
                per_wave_recv_offsets_np[w, t, s] = (
                    w * per_wave_recv_size + s * per_pair_count
                )

    # Per-rank sorted_row: identical to v60.
    sorted_row_padded_np = np.zeros(
        (world_size, _W4_P1D_MAX_PADDED), dtype=np.int32,
    )
    for r in range(world_size):
        for w in range(max_waves):
            for p in range(world_size):
                start = w * per_wave_recv_size + p * per_pair_count
                end = start + per_pair_count
                sorted_row_padded_np[r, start:end] = _w4_p1d_sentinel(w, r, p)

    # Per-rank dummy_src: pre-filled with the constant sentinel.
    # 0xDEADBEEF (3735928559) is out-of-range for int32; reinterpret via
    # uint32 -> int32 view so the bit pattern is preserved.
    dummy_src_np = np.full(
        (world_size, _W4_P1D_DUMMY_SRC_SIZE),
        _W4_P1D_DUMMY_SENTINEL, dtype=np.uint32,
    ).view(np.int32)

    HBM = pltpu.MemorySpace.HBM
    SMEM = pltpu.MemorySpace.SMEM

    kernel = _make_w4_p1d_kernel(
        world_size=world_size, max_waves=max_waves,
        per_pair_count=per_pair_count, recv_capacity=recv_capacity, tile=tile,
    )

    P = jax.sharding.PartitionSpec

    def per_device(src, dummy_src, traffic, send_off, recv_off):
        out_recv = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(
                (_W4_P1D_OUT_RECV_COMBINED_SIZE,), jnp.int32,
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[
                    pl.BlockSpec(memory_space=HBM),   # sorted_row
                    pl.BlockSpec(memory_space=HBM),   # dummy_src
                    pl.BlockSpec(memory_space=SMEM),  # traffic
                    pl.BlockSpec(memory_space=SMEM),  # send_off
                    pl.BlockSpec(memory_space=SMEM),  # recv_off
                ],
                out_specs=pl.BlockSpec(memory_space=HBM),
                scratch_shapes=(
                    pltpu.VMEM(
                        (_W4_P1D_OUT_RECV_COMBINED_SIZE,), jnp.int32,
                    ),
                    pltpu.SemaphoreType.REGULAR((max_waves, 2)),
                    pltpu.SemaphoreType.DMA,
                    pltpu.SemaphoreType.DMA,
                ),
            ),
            interpret=False,
        )(src[0], dummy_src[0], traffic, send_off, recv_off)
        return out_recv[None, ...]

    sharded = jax.shard_map(
        per_device, mesh=mesh,
        in_specs=(P("x"), P("x"), P(), P(), P()),
        out_specs=P("x"),
        check_vma=False,
    )

    sharding_x = jax.sharding.NamedSharding(mesh, P("x"))
    sharding_rep = jax.sharding.NamedSharding(mesh, P())
    src = jax.device_put(jnp.array(sorted_row_padded_np), sharding_x)
    dummy_src = jax.device_put(jnp.array(dummy_src_np), sharding_x)
    traffic = jax.device_put(jnp.array(per_wave_traffic_np), sharding_rep)
    send_off = jax.device_put(
        jnp.array(per_wave_send_offsets_np), sharding_rep,
    )
    recv_off = jax.device_put(
        jnp.array(per_wave_recv_offsets_np), sharding_rep,
    )

    out = sharded(src, dummy_src, traffic, send_off, recv_off)
    return (
        np.array(out),  # [W, recv_capacity + W*TILE]
        max_waves,
        per_pair_count,
        per_wave_recv_size,
        recv_capacity,
        world_size,
        tile,
        _W4_P1D_DUMMY_SENTINEL,
    )


@unittest.skipUnless(_has_tpu(), "Litmus test requires real TPU")
class W4Phase1DummyOnlyLitmus(unittest.TestCase):
    """v66c: W=4 multi-iter REGULAR-sem barriers + v60's exact balanced
    UNCONDITIONAL real DMAs + ONE additional UNCONDITIONAL dummy-region DMA
    per peer per wave. Isolates whether dummy-region writes alone (no skew,
    no variable counts) trigger the multi-iter hang.
    """

    def setUp(self) -> None:
        if jax.device_count() < 4:
            self.skipTest(
                f"Need 4 devices, have {jax.device_count()}"
            )

    def test_w4_p1_dummy_writes_balanced_counts(self) -> None:
        """v66c: balanced 128-per-pair + extended recv + dummy DMAs.

        300s host-fetch timeout via threading; if the kernel hangs, the
        executor thread will not return and the test will fail cleanly.
        """
        import threading

        result_box: list = []
        error_box: list = []

        def runner():
            try:
                result_box.append(_run_w4_p1d_litmus())
            except BaseException as e:  # noqa: BLE001
                error_box.append(e)

        t = threading.Thread(target=runner, daemon=True)
        t.start()
        t.join(timeout=300.0)
        if t.is_alive():
            self.fail(
                "v66c (W=4 phase-1 balanced + dummy-region DMAs) HUNG after "
                "300s; adding dummy-region DMA writes alone is sufficient to "
                "deadlock at W=4 multi-iter (v60-balanced traffic, v60 "
                "barriers, v60 sems, only the extended buffer + 8-DMA-per-"
                "wave + dummy-region access pattern is new)."
            )
        if error_box:
            raise error_box[0]

        (recv, max_waves, per_pair, per_wave_recv_size, recv_capacity,
         world_size, tile, dummy_sent) = result_box[0]

        # Diagnostic header.
        print(
            f"[LITMUS_W4_P1D] max_waves={max_waves} per_pair={per_pair} "
            f"per_wave_recv_size={per_wave_recv_size} "
            f"recv_capacity={recv_capacity} world_size={world_size} "
            f"tile={tile} dummy_sentinel={hex(dummy_sent)}",
            flush=True,
        )
        # Per-rank: print first 4 ints of each wave's slot (real region) and
        # the dummy region for each src.
        for r in range(world_size):
            first_half_first4 = recv[r, :4].tolist()
            second_half_first4 = recv[
                r, per_wave_recv_size:per_wave_recv_size + 4
            ].tolist()
            print(
                f"[LITMUS_W4_P1D] rank {r} "
                f"wave0_slot[:4]={[hex(int(x)) for x in first_half_first4]} "
                f"wave1_slot[:4]={[hex(int(x)) for x in second_half_first4]}",
                flush=True,
            )
            for s in range(world_size):
                dummy_off = recv_capacity + s * tile
                dummy_first4 = recv[r, dummy_off:dummy_off + 4].tolist()
                print(
                    f"[LITMUS_W4_P1D] rank {r} dummy_slot[src={s}][:4]="
                    f"{[hex(int(x)) for x in dummy_first4]}",
                    flush=True,
                )

        # Real-region check: identical to v60. For each (wave, src),
        # at offset w*per_wave_recv_size + s*per_pair, expect
        # _w4_p1d_sentinel(w, s, r) repeated per_pair times.
        mismatches = []
        for r in range(world_size):
            for w in range(max_waves):
                for s in range(world_size):
                    offset = w * per_wave_recv_size + s * per_pair
                    expected = _w4_p1d_sentinel(w, s, r)
                    slot = recv[r, offset:offset + per_pair]
                    if not np.all(slot == expected):
                        actual_first8 = slot[:8].tolist()
                        mismatches.append(
                            f"rank={r} wave={w} src={s} off={offset} "
                            f"REAL expected={hex(expected)} actual_first8="
                            f"{[hex(int(x)) for x in actual_first8]}"
                        )

        # Dummy-region check: for each (rank r, src s), the dummy slot
        # at recv_capacity + s*tile must hold the dummy sentinel
        # (verifying the dummy DMAs landed). If it's zero, the dummy
        # DMA didn't fire (or the post-barrier didn't fence).
        # 0xDEADBEEF reinterpreted as int32 via uint32 view.
        expected_dummy = np.array(
            [_W4_P1D_DUMMY_SENTINEL], dtype=np.uint32,
        ).view(np.int32)[0]
        for r in range(world_size):
            for s in range(world_size):
                dummy_off = recv_capacity + s * tile
                slot = recv[r, dummy_off:dummy_off + tile]
                if not np.all(slot == expected_dummy):
                    actual_first8 = slot[:8].tolist()
                    mismatches.append(
                        f"rank={r} src={s} off={dummy_off} DUMMY "
                        f"expected={hex(_W4_P1D_DUMMY_SENTINEL)} "
                        f"actual_first8="
                        f"{[hex(int(x) & 0xFFFFFFFF) for x in actual_first8]}"
                    )

        if mismatches:
            self.fail(
                "v66c phase-1 balanced + dummy-region DATA MISMATCH:\n  "
                + "\n  ".join(mismatches)
            )


# ---------------------------------------------------------------------------
# v66d: W=4 phase-1 ONLY — v60's exact balanced kernel with VARIABLE per-peer
# DMA sizes (all > 0, all TILE-aligned, no skipped peers) and ALL DMAs
# targeting the REAL recv region (no dummy region writes). Isolation goal:
# does variable-per-peer DMA sizes alone trigger the W=4 multi-iter hang,
# separate from dummy-region access?
#
# v60 (W4UnconditionalDMATightenedLitmus) PASSes: balanced 128-per-pair, no
#   dummy region.
# v66c (W4Phase1DummyOnlyLitmus) PASSes: balanced 128-per-pair + dummy DMAs.
# v66a (W4Phase1SkewedDummyLitmus) HANGs: variable counts + dummy region.
#
# v66d keeps v60's structure EXACTLY (REGULAR pre/post barriers, unconditional
# phase-1 DMAs, wave-distinct payloads, per-wave slot-only zero-fill, REAL
# region only, recv_capacity sized to fit) and only changes the per-peer
# counts to vary in a cyclic pattern: traffic[w, s, t] = base[(s + t) % 4]
# with base = (128, 256, 128, 256). All counts > 0, all TILE-aligned. Row
# sums and column sums are both 768 ints by cyclic symmetry, so the recv
# buffer can be uniformly sized.
#
# Stop conditions:
#   PASS  -> variable per-peer counts alone are FINE; v66a's hang must be
#            from the MIX of real + dummy region targeting in the same wave
#            loop iteration (writing to two disjoint HBM regions per wave
#            with different sizes — likely a Mosaic DMA-batching pessim or
#            aliasing analysis bug).
#   HANG  -> variable per-peer counts at W=4 multi-iter IS the bug. Final
#            minimal repro: v60 + change uniform counts to cyclic
#            (128, 256, 128, 256) and it deadlocks.
# ---------------------------------------------------------------------------


_W4_P1V_TILE = 128
_W4_P1V_MAX_WAVES = 2
_W4_P1V_WORLD_SIZE = 4
# Cyclic per-peer count pattern: traffic[w, s, t] = _W4_P1V_BASE[(s + t) % 4].
_W4_P1V_BASE = (128, 256, 128, 256)
# Row/col sum: each row and each column of the base cyclic matrix sums to 768.
_W4_P1V_PER_WAVE_RECV_SIZE = sum(_W4_P1V_BASE)  # 768 ints per wave per rank
# 2-wave capacity = 1536; round up to 2048 for slack (TILE-aligned).
_W4_P1V_RECV_CAPACITY = 2048
# Sender-side sorted_row buffer per rank: same size as recv (per-wave row sum).
_W4_P1V_MAX_PADDED = 2048


def _w4_p1v_sentinel(w: int, src: int, dst: int) -> int:
    """Wave/source/dest-distinct sentinel: same form as v60.
    0x0B000000 + w*0x10000 + src*0x100 + dst.
    """
    return 0x0B000000 + (w << 16) + (src << 8) + dst


def _make_w4_p1v_kernel(world_size: int, max_waves: int,
                        per_wave_recv_size: int):
    """Per-device kernel: same as v60 but with VARIABLE per-peer counts.
      0. PER-WAVE zero-fill of THIS wave's slot of recv_buf via sync_copy.
      1. PRE-barrier (W-way REGULAR-sem fan-in).
      2. UNCONDITIONAL per-peer DMA with VARIABLE counts (all > 0, TILE-
         aligned, all targeting the REAL recv buffer region).
      3. POST-barrier (W-way REGULAR-sem fan-in).
    """

    def kernel(
        sorted_row_ref,    # [_W4_P1V_MAX_PADDED] int32 HBM
        traffic_ref,       # [max_waves, W, W] int32 SMEM (variable, all > 0)
        send_off_ref,      # [max_waves, W, W] int32 SMEM
        recv_off_ref,      # [max_waves, W, W] int32 SMEM
        out_recv_ref,      # [_W4_P1V_RECV_CAPACITY] int32 HBM
        zero_int_vmem,     # [per_wave_recv_size] int32 VMEM zero buffer
        barrier_sems,      # REGULAR (max_waves, 2)
        send_sem,          # DMA send sem
        recv_sem,          # DMA recv sem
    ):
        my_rank = lax.axis_index("x")

        zero_int_vmem[...] = jnp.zeros_like(zero_int_vmem)

        def wave_body(w, _carry):
            # Phase 0: per-wave slot-only zero-fill (matches v60, NOT v66c
            # full-buffer fill which was a test bug).
            wave_recv_slot = out_recv_ref.at[
                pl.ds(w * per_wave_recv_size, per_wave_recv_size)
            ]
            pltpu.sync_copy(zero_int_vmem, wave_recv_slot)

            # Pre-barrier.
            pre_sem = barrier_sems.at[w, 0]
            for peer in range(world_size):
                pl.semaphore_signal(
                    pre_sem, 1,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
            pl.semaphore_wait(pre_sem, world_size)

            # Phase 1: UNCONDITIONAL per-peer DMA with VARIABLE counts.
            for peer in range(world_size):
                send_count = pl.multiple_of(
                    traffic_ref[w, my_rank, peer], _W4_P1V_TILE
                )
                send_start = pl.multiple_of(
                    send_off_ref[w, my_rank, peer], _W4_P1V_TILE
                )
                remote_recv_start = pl.multiple_of(
                    recv_off_ref[w, peer, my_rank], _W4_P1V_TILE
                )

                d = pltpu.make_async_remote_copy(
                    src_ref=sorted_row_ref.at[pl.ds(send_start, send_count)],
                    dst_ref=out_recv_ref.at[
                        pl.ds(remote_recv_start, send_count)
                    ],
                    send_sem=send_sem, recv_sem=recv_sem,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
                d.start()
                d.wait()

            # Post-barrier.
            post_sem = barrier_sems.at[w, 1]
            for peer in range(world_size):
                pl.semaphore_signal(
                    post_sem, 1,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
            pl.semaphore_wait(post_sem, world_size)
            return None

        lax.fori_loop(0, max_waves, wave_body, None)

    return kernel


def _run_w4_p1v_litmus():
    """Run W=4 phase-1-only litmus with VARIABLE per-peer counts.

    Returns ``(recv_per_rank, traffic_np, recv_off_np, max_waves,
    per_wave_recv_size, world_size)``.
      ``recv_per_rank``: ``[W, _W4_P1V_RECV_CAPACITY]`` int32.
    """
    world_size = _W4_P1V_WORLD_SIZE
    max_waves = _W4_P1V_MAX_WAVES
    per_wave_recv_size = _W4_P1V_PER_WAVE_RECV_SIZE
    devices = jax.devices()[:world_size]
    mesh = jax.sharding.Mesh(np.array(devices), ("x",))

    # Cyclic variable traffic: traffic[w, s, t] = base[(s + t) % W].
    # All counts > 0, all TILE-aligned, row sum == col sum == 768 per wave.
    per_wave_traffic_np = np.zeros(
        (max_waves, world_size, world_size), dtype=np.int32,
    )
    for w in range(max_waves):
        for s in range(world_size):
            for t in range(world_size):
                per_wave_traffic_np[w, s, t] = _W4_P1V_BASE[(s + t) % world_size]

    # Send offsets: cumulative per source within a wave.
    # send_off[w, s, t] = w * per_wave_recv_size + sum_{t' < t} traffic[w,s,t']
    per_wave_send_offsets_np = np.zeros(
        (max_waves, world_size, world_size), dtype=np.int32,
    )
    # Recv offsets on receiver rank t from src s: cumulative per dest.
    # recv_off[w, t, s] = w * per_wave_recv_size + sum_{s' < s} traffic[w,s',t]
    per_wave_recv_offsets_np = np.zeros(
        (max_waves, world_size, world_size), dtype=np.int32,
    )
    for w in range(max_waves):
        for s in range(world_size):
            cum = 0
            for t in range(world_size):
                per_wave_send_offsets_np[w, s, t] = (
                    w * per_wave_recv_size + cum
                )
                cum += int(per_wave_traffic_np[w, s, t])
        for t in range(world_size):
            cum = 0
            for s in range(world_size):
                per_wave_recv_offsets_np[w, t, s] = (
                    w * per_wave_recv_size + cum
                )
                cum += int(per_wave_traffic_np[w, s, t])

    # Per-rank sorted_row: for each (wave w, dest peer p), fill the
    # per-pair slot with the wave/src/dest-distinct sentinel.
    sorted_row_padded_np = np.zeros(
        (world_size, _W4_P1V_MAX_PADDED), dtype=np.int32,
    )
    for r in range(world_size):
        for w in range(max_waves):
            for p in range(world_size):
                start = int(per_wave_send_offsets_np[w, r, p])
                cnt = int(per_wave_traffic_np[w, r, p])
                sorted_row_padded_np[r, start:start + cnt] = (
                    _w4_p1v_sentinel(w, r, p)
                )

    HBM = pltpu.MemorySpace.HBM
    SMEM = pltpu.MemorySpace.SMEM

    kernel = _make_w4_p1v_kernel(
        world_size=world_size, max_waves=max_waves,
        per_wave_recv_size=per_wave_recv_size,
    )

    P = jax.sharding.PartitionSpec

    def per_device(src, traffic, send_off, recv_off):
        recv = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(
                (_W4_P1V_RECV_CAPACITY,), jnp.int32,
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[
                    pl.BlockSpec(memory_space=HBM),   # sorted_row
                    pl.BlockSpec(memory_space=SMEM),  # traffic
                    pl.BlockSpec(memory_space=SMEM),  # send_off
                    pl.BlockSpec(memory_space=SMEM),  # recv_off
                ],
                out_specs=pl.BlockSpec(memory_space=HBM),
                scratch_shapes=(
                    pltpu.VMEM((per_wave_recv_size,), jnp.int32),
                    pltpu.SemaphoreType.REGULAR((max_waves, 2)),
                    pltpu.SemaphoreType.DMA,
                    pltpu.SemaphoreType.DMA,
                ),
            ),
            interpret=False,
        )(src[0], traffic, send_off, recv_off)
        return recv[None, ...]

    sharded = jax.shard_map(
        per_device, mesh=mesh,
        in_specs=(P("x"), P(), P(), P()),
        out_specs=P("x"),
        check_vma=False,
    )

    sharding_x = jax.sharding.NamedSharding(mesh, P("x"))
    sharding_rep = jax.sharding.NamedSharding(mesh, P())
    src = jax.device_put(jnp.array(sorted_row_padded_np), sharding_x)
    traffic = jax.device_put(jnp.array(per_wave_traffic_np), sharding_rep)
    send_off = jax.device_put(
        jnp.array(per_wave_send_offsets_np), sharding_rep,
    )
    recv_off = jax.device_put(
        jnp.array(per_wave_recv_offsets_np), sharding_rep,
    )

    out = sharded(src, traffic, send_off, recv_off)
    return (
        np.array(out),  # [W, _W4_P1V_RECV_CAPACITY]
        per_wave_traffic_np,
        per_wave_recv_offsets_np,
        max_waves,
        per_wave_recv_size,
        world_size,
    )


@unittest.skipUnless(_has_tpu(), "Litmus test requires real TPU")
class W4Phase1VarCountLitmus(unittest.TestCase):
    """v66d: W=4 multi-iter REGULAR-sem barriers + UNCONDITIONAL phase-1
    DMAs with VARIABLE per-peer counts (all > 0, TILE-aligned). All DMAs
    target the REAL recv region (no dummy region writes). Isolates whether
    variable per-peer counts alone (separate from dummy-region writes)
    trigger the W=4 multi-iter hang.
    """

    def setUp(self) -> None:
        if jax.device_count() < 4:
            self.skipTest(
                f"Need 4 devices, have {jax.device_count()}"
            )

    def test_w4_p1_var_counts_no_dummy(self) -> None:
        """v66d: phase-1 only, variable counts, no dummy region.

        300s host-fetch timeout via threading; if the kernel hangs, the
        executor thread will not return and the test will fail cleanly.
        """
        import threading

        result_box: list = []
        error_box: list = []

        def runner():
            try:
                result_box.append(_run_w4_p1v_litmus())
            except BaseException as e:  # noqa: BLE001
                error_box.append(e)

        t = threading.Thread(target=runner, daemon=True)
        t.start()
        t.join(timeout=300.0)
        if t.is_alive():
            self.fail(
                "v66d (W=4 phase-1 variable-counts, no dummy) HUNG after "
                "300s; variable per-peer DMA sizes alone (no dummy region) "
                "are sufficient to deadlock at W=4 multi-iter."
            )
        if error_box:
            raise error_box[0]

        (recv, traffic_np, recv_off_np, max_waves, per_wave_recv_size,
         world_size) = result_box[0]

        # Diagnostic header.
        print(
            f"[LITMUS_W4_P1V] max_waves={max_waves} "
            f"per_wave_recv_size={per_wave_recv_size} "
            f"world_size={world_size} base={_W4_P1V_BASE} "
            f"recv_capacity={_W4_P1V_RECV_CAPACITY}",
            flush=True,
        )
        # Per-rank, per-wave, per-src nonzero traffic counts and recv first4.
        for r in range(world_size):
            for w in range(max_waves):
                for s in range(world_size):
                    cnt = int(traffic_np[w, s, r])
                    off = int(recv_off_np[w, r, s])
                    first4 = recv[r, off:off + 4].tolist()
                    print(
                        f"[LITMUS_W4_P1V] rank={r} wave={w} src={s} "
                        f"cnt={cnt} off={off} first4="
                        f"{[hex(int(x)) for x in first4]}",
                        flush=True,
                    )

        # Per-rank check: for each wave w and source s, at offset
        # recv_off_np[w, r, s], expect _w4_p1v_sentinel(w, s, r) repeated
        # traffic_np[w, s, r] times.
        mismatches = []
        for r in range(world_size):
            for w in range(max_waves):
                for s in range(world_size):
                    cnt = int(traffic_np[w, s, r])
                    off = int(recv_off_np[w, r, s])
                    expected = _w4_p1v_sentinel(w, s, r)
                    slot = recv[r, off:off + cnt]
                    if not np.all(slot == expected):
                        actual_first8 = slot[:8].tolist()
                        mismatches.append(
                            f"rank={r} wave={w} src={s} cnt={cnt} off={off} "
                            f"expected={hex(expected)} actual_first8="
                            f"{[hex(int(x)) for x in actual_first8]}"
                        )
        if mismatches:
            self.fail(
                "v66d phase-1 variable-counts DATA MISMATCH:\n  "
                + "\n  ".join(mismatches)
            )


# ---------------------------------------------------------------------------
# v66e: W=4 phase-1 ONLY — v60's exact uniform 128-per-pair traffic, with
# per-peer FIXED REGION assignment (compile-time, not data-dependent).
# Isolation goal: split whether v66a's hang (variable counts + dummy region)
# is from (A) region switching alone or (B) the COMBINATION of size variability
# AND region switching.
#
# Design:
#   - Identical to v60: REGULAR pre/post barriers, unconditional phase-1 DMAs,
#     wave-distinct payloads, per-wave slot-only zero-fill, ALL counts = 128.
#   - Extend recv buffer to [recv_capacity + W * TILE] (same as v66c).
#   - For each peer, the destination region is fixed at COMPILE TIME via
#     ``if peer % 2 == 0`` in the unrolled Python loop:
#       even peers (0, 2) -> REAL region (v60 offsets)
#       odd  peers (1, 3) -> DUMMY region (slot indexed by my_rank)
#   - ALL DMAs are TILE = 128 in size (uniform, NOT variable).
#   - All sizes/destinations are statically determined per call site.
#
# Stop conditions:
#   PASS  -> region switching ALONE (with uniform sizes) is FINE; v66a's hang
#            requires the COMBINATION of size variability AND region switching.
#   HANG  -> region switching alone (independent of size) is the killer; final
#            minimal repro: per-peer DMAs in the same wave loop targeting
#            different HBM regions, at W=4 multi-iter.
# ---------------------------------------------------------------------------


_W4_P1MR_TILE = 128
_W4_P1MR_MAX_WAVES = 2
_W4_P1MR_PER_PAIR_COUNT = 128  # one TILE per (src, dst) per wave (uniform)
_W4_P1MR_WORLD_SIZE = 4
_W4_P1MR_PER_WAVE_RECV_SIZE = (
    _W4_P1MR_WORLD_SIZE * _W4_P1MR_PER_PAIR_COUNT
)  # 512 ints
_W4_P1MR_RECV_CAPACITY = (
    _W4_P1MR_MAX_WAVES * _W4_P1MR_PER_WAVE_RECV_SIZE
)  # 1024 ints
_W4_P1MR_MAX_PADDED = (
    _W4_P1MR_MAX_WAVES * _W4_P1MR_WORLD_SIZE * _W4_P1MR_PER_PAIR_COUNT
)  # 1024 ints
_W4_P1MR_DUMMY_SENTINEL = 0xDEADBEEF
# Dummy region: one TILE per rank slot, appended to recv_capacity.
_W4_P1MR_DUMMY_REGION_SIZE = _W4_P1MR_WORLD_SIZE * _W4_P1MR_TILE  # 512 ints
_W4_P1MR_OUT_RECV_COMBINED_SIZE = (
    _W4_P1MR_RECV_CAPACITY + _W4_P1MR_DUMMY_REGION_SIZE
)  # 1536 ints
# Dummy src buffer: one TILE per peer (we always send the same constant).
_W4_P1MR_DUMMY_SRC_SIZE = _W4_P1MR_WORLD_SIZE * _W4_P1MR_TILE  # 512 ints


def _w4_p1mr_sentinel(w: int, src: int, dst: int) -> int:
    """Wave/source/dest-distinct sentinel: same form as v60."""
    return 0x0A000000 + (w << 16) + (src << 8) + dst


def _make_w4_p1mr_kernel(world_size: int, max_waves: int,
                         per_pair_count: int, recv_capacity: int, tile: int):
    """v60's exact phase-1-only kernel + per-peer FIXED REGION assignment.

    Per wave:
      0. PER-WAVE zero-fill of THIS wave's slot of the REAL recv region via
         sync_copy (matches v60 — slot-only, NOT full buffer).
      1. PRE-barrier (W-way REGULAR-sem fan-in).
      2. UNCONDITIONAL per-peer DMA (TILE=128 uniform), with COMPILE-TIME
         fixed region choice via ``if peer % 2 == 0``:
           even peers (0, 2) -> REAL region (v60 offsets, src = sorted_row)
           odd  peers (1, 3) -> DUMMY region (slot indexed by my_rank,
                                src = dummy_src)
      3. POST-barrier (W-way REGULAR-sem fan-in).
    """
    per_wave_recv_size = world_size * per_pair_count

    def kernel(
        sorted_row_ref,        # [max_waves * W * per_pair_count] int32 HBM
        dummy_src_ref,         # [W * TILE] int32 HBM (constant sentinel)
        traffic_ref,           # [max_waves, W, W] int32 SMEM (uniform 128)
        send_off_ref,          # [max_waves, W, W] int32 SMEM
        recv_off_ref,          # [max_waves, W, W] int32 SMEM
        out_recv_combined_ref, # [recv_capacity + W*TILE] int32 HBM
        zero_int_vmem,         # [per_wave_recv_size] int32 VMEM zero buffer
        barrier_sems,          # REGULAR (max_waves, 2)
        send_sem,              # DMA send sem
        recv_sem,              # DMA recv sem
    ):
        my_rank = lax.axis_index("x")

        # Initialize zero VMEM scratch once outside the loop.
        zero_int_vmem[...] = jnp.zeros_like(zero_int_vmem)

        def wave_body(w, _carry):
            # Phase 0: per-wave slot-only zero-fill of the REAL region
            # (matches v60 exactly — does NOT touch dummy region).
            wave_recv_slot = out_recv_combined_ref.at[
                pl.ds(w * per_wave_recv_size, per_wave_recv_size)
            ]
            pltpu.sync_copy(zero_int_vmem, wave_recv_slot)

            # Pre-barrier: signal every peer (incl. self), wait W signals.
            pre_sem = barrier_sems.at[w, 0]
            for peer in range(world_size):
                pl.semaphore_signal(
                    pre_sem, 1,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
            pl.semaphore_wait(pre_sem, world_size)

            # Phase 1: per-peer UNCONDITIONAL DMA (TILE=128 uniform), with
            # COMPILE-TIME fixed region choice. Each of the 4 unrolled call
            # sites has statically determined source and destination region.
            for peer in range(world_size):
                # Uniform TILE count for ALL peers (read from SMEM but always
                # 128, with multiple_of TILE annotation).
                cnt = pl.multiple_of(
                    traffic_ref[w, my_rank, peer], tile,
                )

                if peer % 2 == 0:
                    # Even peer -> REAL region (v60-style).
                    send_start = pl.multiple_of(
                        send_off_ref[w, my_rank, peer], tile,
                    )
                    remote_recv_start = pl.multiple_of(
                        recv_off_ref[w, peer, my_rank], tile,
                    )
                    src = sorted_row_ref.at[pl.ds(send_start, cnt)]
                    dst = out_recv_combined_ref.at[
                        pl.ds(remote_recv_start, cnt)
                    ]
                else:
                    # Odd peer -> DUMMY region (fixed offsets, no SMEM read
                    # needed for offsets; my_rank-indexed slot).
                    src = dummy_src_ref.at[pl.ds(peer * tile, tile)]
                    dst = out_recv_combined_ref.at[
                        pl.ds(recv_capacity + my_rank * tile, tile)
                    ]

                d = pltpu.make_async_remote_copy(
                    src_ref=src, dst_ref=dst,
                    send_sem=send_sem, recv_sem=recv_sem,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
                d.start()
                d.wait()

            # Post-barrier: signal every peer (incl. self), wait W signals.
            post_sem = barrier_sems.at[w, 1]
            for peer in range(world_size):
                pl.semaphore_signal(
                    post_sem, 1,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
            pl.semaphore_wait(post_sem, world_size)
            return None

        lax.fori_loop(0, max_waves, wave_body, None)

    return kernel


def _run_w4_p1mr_litmus():
    """Run W=4 phase-1 mixed-region (uniform-size) litmus.

    Returns ``(out_recv_combined_per_rank, max_waves, per_pair_count,
    per_wave_recv_size, recv_capacity, world_size, tile, dummy_sentinel)``.
    """
    world_size = _W4_P1MR_WORLD_SIZE
    max_waves = _W4_P1MR_MAX_WAVES
    per_pair_count = _W4_P1MR_PER_PAIR_COUNT
    per_wave_recv_size = _W4_P1MR_PER_WAVE_RECV_SIZE
    recv_capacity = _W4_P1MR_RECV_CAPACITY
    tile = _W4_P1MR_TILE
    devices = jax.devices()[:world_size]
    mesh = jax.sharding.Mesh(np.array(devices), ("x",))

    # Uniform balanced traffic (every (s, t) sends per_pair_count = 128).
    per_wave_traffic_np = np.full(
        (max_waves, world_size, world_size),
        per_pair_count, dtype=np.int32,
    )
    # Send/recv offsets identical to v60.
    per_wave_send_offsets_np = np.zeros(
        (max_waves, world_size, world_size), dtype=np.int32,
    )
    per_wave_recv_offsets_np = np.zeros(
        (max_waves, world_size, world_size), dtype=np.int32,
    )
    for w in range(max_waves):
        for s in range(world_size):
            for t in range(world_size):
                per_wave_send_offsets_np[w, s, t] = (
                    w * per_wave_recv_size + t * per_pair_count
                )
                per_wave_recv_offsets_np[w, t, s] = (
                    w * per_wave_recv_size + s * per_pair_count
                )

    # Per-rank sorted_row: identical to v60.
    sorted_row_padded_np = np.zeros(
        (world_size, _W4_P1MR_MAX_PADDED), dtype=np.int32,
    )
    for r in range(world_size):
        for w in range(max_waves):
            for p in range(world_size):
                start = w * per_wave_recv_size + p * per_pair_count
                end = start + per_pair_count
                sorted_row_padded_np[r, start:end] = (
                    _w4_p1mr_sentinel(w, r, p)
                )

    # Per-rank dummy_src: pre-filled with the constant sentinel.
    dummy_src_np = np.full(
        (world_size, _W4_P1MR_DUMMY_SRC_SIZE),
        _W4_P1MR_DUMMY_SENTINEL, dtype=np.uint32,
    ).view(np.int32)

    HBM = pltpu.MemorySpace.HBM
    SMEM = pltpu.MemorySpace.SMEM

    kernel = _make_w4_p1mr_kernel(
        world_size=world_size, max_waves=max_waves,
        per_pair_count=per_pair_count, recv_capacity=recv_capacity, tile=tile,
    )

    P = jax.sharding.PartitionSpec

    def per_device(src, dummy_src, traffic, send_off, recv_off):
        out_recv = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(
                (_W4_P1MR_OUT_RECV_COMBINED_SIZE,), jnp.int32,
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[
                    pl.BlockSpec(memory_space=HBM),   # sorted_row
                    pl.BlockSpec(memory_space=HBM),   # dummy_src
                    pl.BlockSpec(memory_space=SMEM),  # traffic
                    pl.BlockSpec(memory_space=SMEM),  # send_off
                    pl.BlockSpec(memory_space=SMEM),  # recv_off
                ],
                out_specs=pl.BlockSpec(memory_space=HBM),
                scratch_shapes=(
                    pltpu.VMEM(
                        (per_wave_recv_size,), jnp.int32,
                    ),
                    pltpu.SemaphoreType.REGULAR((max_waves, 2)),
                    pltpu.SemaphoreType.DMA,
                    pltpu.SemaphoreType.DMA,
                ),
            ),
            interpret=False,
        )(src[0], dummy_src[0], traffic, send_off, recv_off)
        return out_recv[None, ...]

    sharded = jax.shard_map(
        per_device, mesh=mesh,
        in_specs=(P("x"), P("x"), P(), P(), P()),
        out_specs=P("x"),
        check_vma=False,
    )

    sharding_x = jax.sharding.NamedSharding(mesh, P("x"))
    sharding_rep = jax.sharding.NamedSharding(mesh, P())
    src = jax.device_put(jnp.array(sorted_row_padded_np), sharding_x)
    dummy_src = jax.device_put(jnp.array(dummy_src_np), sharding_x)
    traffic = jax.device_put(jnp.array(per_wave_traffic_np), sharding_rep)
    send_off = jax.device_put(
        jnp.array(per_wave_send_offsets_np), sharding_rep,
    )
    recv_off = jax.device_put(
        jnp.array(per_wave_recv_offsets_np), sharding_rep,
    )

    out = sharded(src, dummy_src, traffic, send_off, recv_off)
    return (
        np.array(out),  # [W, recv_capacity + W*TILE]
        max_waves,
        per_pair_count,
        per_wave_recv_size,
        recv_capacity,
        world_size,
        tile,
        _W4_P1MR_DUMMY_SENTINEL,
    )


@unittest.skipUnless(_has_tpu(), "Litmus test requires real TPU")
class W4Phase1MixedRegionLitmus(unittest.TestCase):
    """v66e: W=4 multi-iter REGULAR-sem barriers + UNCONDITIONAL phase-1
    DMAs with UNIFORM TILE counts (all 128) and per-peer FIXED REGION
    assignment by compile-time pattern (even peers -> real region, odd peers
    -> dummy region). Splits whether v66a's hang is from region switching
    alone (this test should HANG) or from the combination of size variability
    + region switching (this test should PASS).
    """

    def setUp(self) -> None:
        if jax.device_count() < 4:
            self.skipTest(
                f"Need 4 devices, have {jax.device_count()}"
            )

    def test_w4_p1_uniform_tile_mixed_region(self) -> None:
        """v66e: phase-1 only, uniform 128-per-peer, mixed REAL/DUMMY region.

        300s host-fetch timeout via threading; if the kernel hangs, the
        executor thread will not return and the test will fail cleanly.
        """
        import threading

        result_box: list = []
        error_box: list = []

        def runner():
            try:
                result_box.append(_run_w4_p1mr_litmus())
            except BaseException as e:  # noqa: BLE001
                error_box.append(e)

        t = threading.Thread(target=runner, daemon=True)
        t.start()
        t.join(timeout=300.0)
        if t.is_alive():
            self.fail(
                "v66e (W=4 phase-1 uniform TILE, mixed real/dummy region) "
                "HUNG after 300s; region switching ALONE (independent of "
                "size variability) at W=4 multi-iter is the killer. Final "
                "minimal repro: per-peer DMAs in the same wave loop "
                "targeting different HBM regions."
            )
        if error_box:
            raise error_box[0]

        (recv, max_waves, per_pair, per_wave_recv_size, recv_capacity,
         world_size, tile, dummy_sent) = result_box[0]

        # Diagnostic header.
        print(
            f"[LITMUS_W4_P1MR] max_waves={max_waves} per_pair={per_pair} "
            f"per_wave_recv_size={per_wave_recv_size} "
            f"recv_capacity={recv_capacity} world_size={world_size} "
            f"tile={tile} dummy_sentinel={hex(dummy_sent)}",
            flush=True,
        )
        # Per-rank: print first 4 ints of each wave's slot (real region) and
        # the dummy region for our rank's slot.
        for r in range(world_size):
            for w in range(max_waves):
                wave_off = w * per_wave_recv_size
                first4 = recv[r, wave_off:wave_off + 4].tolist()
                print(
                    f"[LITMUS_W4_P1MR] rank {r} wave{w}_slot[:4]="
                    f"{[hex(int(x)) for x in first4]}",
                    flush=True,
                )
            dummy_off = recv_capacity + r * tile
            dummy_first4 = recv[r, dummy_off:dummy_off + 4].tolist()
            print(
                f"[LITMUS_W4_P1MR] rank {r} dummy_slot[my_rank][:4]="
                f"{[hex(int(x)) for x in dummy_first4]}",
                flush=True,
            )

        # Verification (correct semantics):
        # In the kernel, the "real vs dummy" choice is by ``peer % 2``, where
        # ``peer`` is the sender's destination index. So a sender s writes
        # to receiver r's REAL region iff r % 2 == 0, and to receiver r's
        # DUMMY region iff r % 2 == 1.
        #   - Even-rank receiver: ALL 4 sources wrote real region with
        #                          sentinel; dummy region untouched (zero).
        #   - Odd-rank receiver: real region all zero (no source wrote
        #                         there); dummy slot for my_rank holds
        #                         the dummy sentinel (last writer wins, all
        #                         4 sources wrote same constant).
        expected_dummy = np.array(
            [_W4_P1MR_DUMMY_SENTINEL], dtype=np.uint32,
        ).view(np.int32)[0]
        mismatches = []
        for r in range(world_size):
            for w in range(max_waves):
                for s in range(world_size):
                    offset = w * per_wave_recv_size + s * per_pair
                    slot = recv[r, offset:offset + per_pair]
                    if r % 2 == 0:
                        # Even receiver: every source wrote real region.
                        expected = _w4_p1mr_sentinel(w, s, r)
                        if not np.all(slot == expected):
                            actual_first8 = slot[:8].tolist()
                            mismatches.append(
                                f"rank={r} wave={w} src={s} off={offset} "
                                f"REAL(even-recv) expected={hex(expected)} "
                                f"actual_first8="
                                f"{[hex(int(x)) for x in actual_first8]}"
                            )
                    else:
                        # Odd receiver: nobody wrote real region.
                        if not np.all(slot == 0):
                            actual_first8 = slot[:8].tolist()
                            mismatches.append(
                                f"rank={r} wave={w} src={s} off={offset} "
                                f"REAL(odd-recv) expected=0x0 (zero-fill) "
                                f"actual_first8="
                                f"{[hex(int(x)) for x in actual_first8]}"
                            )

            # Dummy region slot for this rank.
            dummy_off = recv_capacity + r * tile
            dummy_slot = recv[r, dummy_off:dummy_off + tile]
            if r % 2 == 0:
                # Even receiver: dummy slot untouched, should be zero
                # (zero-fill of real region only does NOT touch dummy region;
                # initial out-buffer state may be uninitialized — accept
                # either zero or any value with note).
                pass  # No strict check; even-recv dummy is uninitialized.
            else:
                # Odd receiver: every source wrote the dummy sentinel here.
                if not np.all(dummy_slot == expected_dummy):
                    actual_first8 = dummy_slot[:8].tolist()
                    mismatches.append(
                        f"rank={r} dummy_off={dummy_off} DUMMY(odd-recv) "
                        f"expected={hex(_W4_P1MR_DUMMY_SENTINEL)} "
                        f"actual_first8="
                        f"{[hex(int(x) & 0xFFFFFFFF) for x in actual_first8]}"
                    )

        if mismatches:
            self.fail(
                "v66e phase-1 mixed-region (uniform 128) DATA MISMATCH:\n  "
                + "\n  ".join(mismatches)
            )


# ---------------------------------------------------------------------------
# v66n: v66d's kernel VERBATIM, but per-rank-DIFFERENT count patterns via
# cyclic rotation. Where v66d uses traffic[w, s, t] = BASE[(s + t) % W] (which
# yields the SAME outgoing per-peer pattern across all srcs after relabeling),
# v66n uses traffic[w, r, p] = BASE[(p - r) % W] with BASE = (128, 256, 384,
# 512). With this BASE each rank has a DISTINCT outgoing pattern (no period-2
# symmetry). Row sum and column sum of the matrix are both 128+256+384+512 =
# 1280 ints per wave per rank by cyclic invariance.
#
# Same kernel, sentinels, and verifier as v66d.
#
# Stop conditions:
#   PASS  -> "per-rank-different counts" is NOT the load-bearing trigger;
#            v66k's hang must come from something else (specific count
#            distribution, multi-wave-split interaction with skewed input).
#   HANG  -> confirms "per-rank-different counts" IS the load-bearing trigger
#            for the W=4 multi-iter hang; all workarounds need to maintain
#            per-rank-uniform counts.
# ---------------------------------------------------------------------------


_W4_P1VR_TILE = 128
_W4_P1VR_MAX_WAVES = 2
_W4_P1VR_WORLD_SIZE = 4
# Per-rank rotated cyclic pattern: traffic[w, r, p] = BASE[(p - r) % W].
# BASE has no period-2 symmetry, so all 4 ranks see distinct outgoing patterns.
_W4_P1VR_BASE = (128, 256, 384, 512)
# Row sum == column sum == 1280 (cyclic invariance).
_W4_P1VR_PER_WAVE_RECV_SIZE = sum(_W4_P1VR_BASE)  # 1280 ints per wave per rank
# 2-wave capacity = 2560 (TILE-aligned: 128 * 20).
_W4_P1VR_RECV_CAPACITY = 2560
_W4_P1VR_MAX_PADDED = 2560


def _w4_p1vr_sentinel(w: int, src: int, dst: int) -> int:
    """Wave/source/dest-distinct sentinel (v66n-distinct prefix 0x0D)."""
    return 0x0D000000 + (w << 16) + (src << 8) + dst


def _make_w4_p1vr_kernel(world_size: int, max_waves: int,
                         per_wave_recv_size: int):
    """Kernel: identical to v66d's _make_w4_p1v_kernel, only constants change."""

    def kernel(
        sorted_row_ref,    # [_W4_P1VR_MAX_PADDED] int32 HBM
        traffic_ref,       # [max_waves, W, W] int32 SMEM (variable, all > 0)
        send_off_ref,      # [max_waves, W, W] int32 SMEM
        recv_off_ref,      # [max_waves, W, W] int32 SMEM
        out_recv_ref,      # [_W4_P1VR_RECV_CAPACITY] int32 HBM
        zero_int_vmem,     # [per_wave_recv_size] int32 VMEM zero buffer
        barrier_sems,      # REGULAR (max_waves, 2)
        send_sem,          # DMA send sem
        recv_sem,          # DMA recv sem
    ):
        my_rank = lax.axis_index("x")

        zero_int_vmem[...] = jnp.zeros_like(zero_int_vmem)

        def wave_body(w, _carry):
            wave_recv_slot = out_recv_ref.at[
                pl.ds(w * per_wave_recv_size, per_wave_recv_size)
            ]
            pltpu.sync_copy(zero_int_vmem, wave_recv_slot)

            pre_sem = barrier_sems.at[w, 0]
            for peer in range(world_size):
                pl.semaphore_signal(
                    pre_sem, 1,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
            pl.semaphore_wait(pre_sem, world_size)

            for peer in range(world_size):
                send_count = pl.multiple_of(
                    traffic_ref[w, my_rank, peer], _W4_P1VR_TILE
                )
                send_start = pl.multiple_of(
                    send_off_ref[w, my_rank, peer], _W4_P1VR_TILE
                )
                remote_recv_start = pl.multiple_of(
                    recv_off_ref[w, peer, my_rank], _W4_P1VR_TILE
                )

                d = pltpu.make_async_remote_copy(
                    src_ref=sorted_row_ref.at[pl.ds(send_start, send_count)],
                    dst_ref=out_recv_ref.at[
                        pl.ds(remote_recv_start, send_count)
                    ],
                    send_sem=send_sem, recv_sem=recv_sem,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
                d.start()
                d.wait()

            post_sem = barrier_sems.at[w, 1]
            for peer in range(world_size):
                pl.semaphore_signal(
                    post_sem, 1,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
            pl.semaphore_wait(post_sem, world_size)
            return None

        lax.fori_loop(0, max_waves, wave_body, None)

    return kernel


def _run_w4_p1vr_litmus():
    """Run W=4 phase-1 litmus with per-rank rotated cyclic counts."""
    world_size = _W4_P1VR_WORLD_SIZE
    max_waves = _W4_P1VR_MAX_WAVES
    per_wave_recv_size = _W4_P1VR_PER_WAVE_RECV_SIZE
    devices = jax.devices()[:world_size]
    mesh = jax.sharding.Mesh(np.array(devices), ("x",))

    # Per-rank ROTATED cyclic pattern: traffic[w, r, p] = BASE[(p - r) % W].
    # Each rank has a DISTINCT outgoing pattern.
    per_wave_traffic_np = np.zeros(
        (max_waves, world_size, world_size), dtype=np.int32,
    )
    for w in range(max_waves):
        for r in range(world_size):
            for p in range(world_size):
                per_wave_traffic_np[w, r, p] = (
                    _W4_P1VR_BASE[(p - r) % world_size]
                )

    # Send offsets: cumulative per source within a wave.
    per_wave_send_offsets_np = np.zeros(
        (max_waves, world_size, world_size), dtype=np.int32,
    )
    # Recv offsets on receiver rank t from src s: cumulative per dest.
    per_wave_recv_offsets_np = np.zeros(
        (max_waves, world_size, world_size), dtype=np.int32,
    )
    for w in range(max_waves):
        for s in range(world_size):
            cum = 0
            for t in range(world_size):
                per_wave_send_offsets_np[w, s, t] = (
                    w * per_wave_recv_size + cum
                )
                cum += int(per_wave_traffic_np[w, s, t])
        for t in range(world_size):
            cum = 0
            for s in range(world_size):
                per_wave_recv_offsets_np[w, t, s] = (
                    w * per_wave_recv_size + cum
                )
                cum += int(per_wave_traffic_np[w, s, t])

    # Per-rank sorted_row.
    sorted_row_padded_np = np.zeros(
        (world_size, _W4_P1VR_MAX_PADDED), dtype=np.int32,
    )
    for r in range(world_size):
        for w in range(max_waves):
            for p in range(world_size):
                start = int(per_wave_send_offsets_np[w, r, p])
                cnt = int(per_wave_traffic_np[w, r, p])
                sorted_row_padded_np[r, start:start + cnt] = (
                    _w4_p1vr_sentinel(w, r, p)
                )

    HBM = pltpu.MemorySpace.HBM
    SMEM = pltpu.MemorySpace.SMEM

    kernel = _make_w4_p1vr_kernel(
        world_size=world_size, max_waves=max_waves,
        per_wave_recv_size=per_wave_recv_size,
    )

    P = jax.sharding.PartitionSpec

    def per_device(src, traffic, send_off, recv_off):
        recv = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(
                (_W4_P1VR_RECV_CAPACITY,), jnp.int32,
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[
                    pl.BlockSpec(memory_space=HBM),   # sorted_row
                    pl.BlockSpec(memory_space=SMEM),  # traffic
                    pl.BlockSpec(memory_space=SMEM),  # send_off
                    pl.BlockSpec(memory_space=SMEM),  # recv_off
                ],
                out_specs=pl.BlockSpec(memory_space=HBM),
                scratch_shapes=(
                    pltpu.VMEM((per_wave_recv_size,), jnp.int32),
                    pltpu.SemaphoreType.REGULAR((max_waves, 2)),
                    pltpu.SemaphoreType.DMA,
                    pltpu.SemaphoreType.DMA,
                ),
            ),
            interpret=False,
        )(src[0], traffic, send_off, recv_off)
        return recv[None, ...]

    sharded = jax.shard_map(
        per_device, mesh=mesh,
        in_specs=(P("x"), P(), P(), P()),
        out_specs=P("x"),
        check_vma=False,
    )

    sharding_x = jax.sharding.NamedSharding(mesh, P("x"))
    sharding_rep = jax.sharding.NamedSharding(mesh, P())
    src = jax.device_put(jnp.array(sorted_row_padded_np), sharding_x)
    traffic = jax.device_put(jnp.array(per_wave_traffic_np), sharding_rep)
    send_off = jax.device_put(
        jnp.array(per_wave_send_offsets_np), sharding_rep,
    )
    recv_off = jax.device_put(
        jnp.array(per_wave_recv_offsets_np), sharding_rep,
    )

    out = sharded(src, traffic, send_off, recv_off)
    return (
        np.array(out),  # [W, _W4_P1VR_RECV_CAPACITY]
        per_wave_traffic_np,
        per_wave_recv_offsets_np,
        max_waves,
        per_wave_recv_size,
        world_size,
    )


@unittest.skipUnless(_has_tpu(), "Litmus test requires real TPU")
class W4Phase1VarCountPerRankLitmus(unittest.TestCase):
    """v66n: v66d's kernel + per-rank rotated cyclic count pattern. Each rank
    has a DISTINCT outgoing per-peer pattern. Tests whether
    "per-rank-different counts" alone is the load-bearing trigger for the
    W=4 multi-iter hang seen in v66k.
    """

    def setUp(self) -> None:
        if jax.device_count() < 4:
            self.skipTest(
                f"Need 4 devices, have {jax.device_count()}"
            )

    def test_w4_p1_var_counts_per_rank_rotated(self) -> None:
        """v66n: phase-1 only, per-rank-rotated counts, no dummy region."""
        import threading

        result_box: list = []
        error_box: list = []

        def runner():
            try:
                result_box.append(_run_w4_p1vr_litmus())
            except BaseException as e:  # noqa: BLE001
                error_box.append(e)

        t = threading.Thread(target=runner, daemon=True)
        t.start()
        t.join(timeout=300.0)
        if t.is_alive():
            self.fail(
                "v66n (W=4 phase-1 per-rank-rotated counts, no dummy) HUNG "
                "after 300s; per-rank-different per-peer DMA counts ARE the "
                "load-bearing trigger for the W=4 multi-iter deadlock."
            )
        if error_box:
            raise error_box[0]

        (recv, traffic_np, recv_off_np, max_waves, per_wave_recv_size,
         world_size) = result_box[0]

        # Diagnostic header.
        print(
            f"[LITMUS_W4_P1VR] max_waves={max_waves} "
            f"per_wave_recv_size={per_wave_recv_size} "
            f"world_size={world_size} base={_W4_P1VR_BASE} "
            f"recv_capacity={_W4_P1VR_RECV_CAPACITY}",
            flush=True,
        )
        for r in range(world_size):
            for w in range(max_waves):
                for s in range(world_size):
                    cnt = int(traffic_np[w, s, r])
                    off = int(recv_off_np[w, r, s])
                    first4 = recv[r, off:off + 4].tolist()
                    print(
                        f"[LITMUS_W4_P1VR] rank={r} wave={w} src={s} "
                        f"cnt={cnt} off={off} first4="
                        f"{[hex(int(x)) for x in first4]}",
                        flush=True,
                    )

        mismatches = []
        for r in range(world_size):
            for w in range(max_waves):
                for s in range(world_size):
                    cnt = int(traffic_np[w, s, r])
                    off = int(recv_off_np[w, r, s])
                    expected = _w4_p1vr_sentinel(w, s, r)
                    slot = recv[r, off:off + cnt]
                    if not np.all(slot == expected):
                        actual_first8 = slot[:8].tolist()
                        mismatches.append(
                            f"rank={r} wave={w} src={s} cnt={cnt} off={off} "
                            f"expected={hex(expected)} actual_first8="
                            f"{[hex(int(x)) for x in actual_first8]}"
                        )
        if mismatches:
            self.fail(
                "v66n phase-1 per-rank-rotated counts DATA MISMATCH:\n  "
                + "\n  ".join(mismatches)
            )


# ---------------------------------------------------------------------------
# v66o: v66n-shape kernel + helper-driven wave split on a SYMMETRIC/rotated
# total traffic matrix. Tests whether `_split_traffic_into_waves` itself
# (producing per-wave-different chunked traffic with no skewed-input
# pipeline) is the load-bearing trigger for the W=4 multi-iter hang.
#
# Setup:
#   1. total_traffic[r, p] = BASE[(p - r) % W], BASE = (128, 256, 384, 512).
#      Per-receiver column sum = 1280 by cyclic invariance.
#   2. Pass through `_split_traffic_into_waves(total, recv_capacity=640,
#      max_waves=2, world_size=4)`. The greedy bin-packer produces per-wave
#      chunked traffic that DIFFERS across waves and across (s, t) pairs.
#   3. Greedy with W=4 + 2 waves *structurally* yields per-pair zero entries
#      (proof: take_partial at source 0 forces budget < TILE, so subsequent
#      sources in wave 0 take 0 — see analysis comment block below). To use
#      v66n's UNCONDITIONAL-DMA kernel (which crashes libtpu firmware on
#      0-length DMAs), apply a TILE-pad pass: eff_count = max(per_wave, TILE).
#      This is the SAME postprocess as v66k. The DELTA from v66k is that the
#      input matrix here is synthetic/cyclic (no skewed-input bucketize/sort
#      pipeline) — so v66o vs. v66k isolates the contribution of the
#      skewed-input pipeline.
#   4. Same v66n-style wave-body kernel (pre-barrier, unconditional per-peer
#      DMAs, post-barrier) but with eff_count and a per_wave_real-sized
#      recv slot per wave.
#   5. Sentinels and verifier mirror v66n's, with a v66o-distinct prefix
#      0x0E.
#
# Stop conditions (interpretation is conditional on TILE-pad workaround being
# applied identically as in v66k):
#   PASS  -> helper-driven split + TILE-pad on a synthetic SYMMETRIC matrix
#            is fine. v66k's hang must come from the skewed-input pipeline
#            (bucketize + sort + per-rank-different totals derived from the
#            skewed distribution).
#   HANG  -> helper-driven split + TILE-pad alone (regardless of input
#            distribution) IS the load-bearing trigger. The fingerprint
#            reduces to "per-wave-different chunked traffic at W=4
#            multi-iter".
# ---------------------------------------------------------------------------


_W4_P1HS_TILE = 128
_W4_P1HS_MAX_WAVES = 2
_W4_P1HS_WORLD_SIZE = 4
# Symmetric/rotated TOTAL pattern: total[r, p] = BASE[(p - r) % W]. Per-rank
# DISTINCT outgoing pattern; per-receiver column sum = 1280 by cyclic
# invariance.
_W4_P1HS_BASE = (128, 256, 384, 512)
# recv_capacity=640 forces a 2-wave split: per-receiver total 1280 must split
# into ~640 per wave. Greedy bin-packing produces per-wave-different chunked
# traffic with structurally unavoidable per-pair zero entries (see analysis
# above); these are TILE-padded post-helper to satisfy v66n's unconditional
# DMA kernel.
_W4_P1HS_RECV_CAPACITY = 640


def _w4_p1hs_sentinel(w: int, src: int, dst: int) -> int:
    """Wave/source/dest-distinct sentinel (v66o-distinct prefix 0x0E)."""
    return 0x0E000000 + (w << 16) + (src << 8) + dst


def _w4_p1hs_compute_eff_count_and_offsets(
    per_wave_traffic: np.ndarray,
    *,
    world_size: int,
    tile: int,
):
    """Min-pad-to-TILE postprocess on helper-derived per-wave traffic.

    Mirrors v66k's `_w4_p1minpad_compute_padded_arrays`: for each wave and
    (src, dst) pair, replace 0-count entries with a single TILE so the
    resulting eff_count is everywhere >= TILE. Then derive per-wave send/
    recv offsets as cumulative prefix sums of eff_count, and per_wave_real
    as the worst-case per-receiver column sum across waves.
    """
    max_waves = per_wave_traffic.shape[0]
    eff_count = np.maximum(per_wave_traffic, tile).astype(np.int32)
    send_off = np.zeros_like(eff_count)
    recv_off = np.zeros_like(eff_count)
    for w in range(max_waves):
        for A in range(world_size):
            cum = 0
            for B in range(world_size):
                send_off[w, A, B] = cum
                cum += int(eff_count[w, A, B])
        for B in range(world_size):
            cum = 0
            for A in range(world_size):
                recv_off[w, A, B] = cum
                cum += int(eff_count[w, A, B])
    per_wave_real = int(np.max(np.sum(eff_count, axis=1)))
    return eff_count, send_off, recv_off, per_wave_real


def _make_w4_p1hs_kernel(world_size: int, max_waves: int,
                         per_wave_real: int, tile: int):
    """v66n-shape kernel adapted for helper-derived eff_count + per-wave slots.

    Identical wave_body structure to v66n: per-wave zero-fill of recv slot,
    pre-barrier, unconditional per-peer DMA, post-barrier. Only differences
    vs. v66n:
      * Recv buffer is `max_waves * per_wave_real` (vs. v66n's RECV_CAPACITY
        which doubled as the recv_size). Per-wave slot is at offset
        `w * per_wave_real`.
      * Per-pair count comes from helper-derived `eff_count` (TILE-padded
        per-wave-chunked) rather than v66n's hardcoded BASE rotation.
      * Send/recv offsets include the `w * per_wave_real` wave-slot base.
    """

    def kernel(
        sorted_row_ref,    # [max_waves * per_wave_real] int32 HBM
        eff_count_ref,     # [max_waves, W, W] int32 SMEM
        send_off_ref,      # [max_waves, W, W] int32 SMEM
        recv_off_ref,      # [max_waves, W, W] int32 SMEM
        out_recv_ref,      # [max_waves * per_wave_real] int32 HBM
        zero_int_vmem,     # [per_wave_real] int32 VMEM zero buffer
        barrier_sems,      # REGULAR (max_waves, 2)
        send_sem,          # DMA send sem
        recv_sem,          # DMA recv sem
    ):
        my_rank = lax.axis_index("x")

        zero_int_vmem[...] = jnp.zeros_like(zero_int_vmem)

        def wave_body(w, _carry):
            wave_recv_slot = out_recv_ref.at[
                pl.ds(w * per_wave_real, per_wave_real)
            ]
            pltpu.sync_copy(zero_int_vmem, wave_recv_slot)

            pre_sem = barrier_sems.at[w, 0]
            for peer in range(world_size):
                pl.semaphore_signal(
                    pre_sem, 1,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
            pl.semaphore_wait(pre_sem, world_size)

            for peer in range(world_size):
                cnt = pl.multiple_of(
                    eff_count_ref[w, my_rank, peer], tile,
                )
                send_start = pl.multiple_of(
                    w * per_wave_real + send_off_ref[w, my_rank, peer], tile,
                )
                remote_recv_start = pl.multiple_of(
                    w * per_wave_real + recv_off_ref[w, peer, my_rank], tile,
                )
                d = pltpu.make_async_remote_copy(
                    src_ref=sorted_row_ref.at[pl.ds(send_start, cnt)],
                    dst_ref=out_recv_ref.at[pl.ds(remote_recv_start, cnt)],
                    send_sem=send_sem, recv_sem=recv_sem,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
                d.start()
                d.wait()

            post_sem = barrier_sems.at[w, 1]
            for peer in range(world_size):
                pl.semaphore_signal(
                    post_sem, 1,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
            pl.semaphore_wait(post_sem, world_size)
            return None

        lax.fori_loop(0, max_waves, wave_body, None)

    return kernel


def _run_w4_p1hs_litmus():
    """v66o: helper-driven split on a SYMMETRIC/rotated total traffic matrix,
    plus min-pad-to-TILE postprocess to handle structural zero edges from the
    greedy bin-packer.
    """
    from recir.ops.pallas.embeddings.jagged_a2a_lookup import (
        _split_traffic_into_waves,
    )

    world_size = _W4_P1HS_WORLD_SIZE
    tile = _W4_P1HS_TILE
    max_waves = _W4_P1HS_MAX_WAVES
    recv_capacity = _W4_P1HS_RECV_CAPACITY

    devices = jax.devices()[:world_size]
    mesh = jax.sharding.Mesh(np.array(devices), ("x",))

    # Step 1: SYMMETRIC/rotated total traffic matrix.
    total_traffic_np = np.zeros((world_size, world_size), dtype=np.int32)
    for r in range(world_size):
        for p in range(world_size):
            total_traffic_np[r, p] = _W4_P1HS_BASE[(p - r) % world_size]

    # Step 2: pass through helper. Greedy bin-packing yields per-wave-different
    # chunked traffic.
    per_wave_traffic_jax, _send_offs_helper, _recv_offs_helper = (
        _split_traffic_into_waves(
            jnp.array(total_traffic_np),
            recv_capacity=recv_capacity,
            max_waves=max_waves,
            world_size=world_size,
        )
    )
    per_wave_traffic_np = np.array(per_wave_traffic_jax)

    # Sanity: per-wave matrices DIFFER and total per (s, t) sums to
    # total_traffic.
    assert per_wave_traffic_np.sum(axis=0).tolist() == total_traffic_np.tolist(), (
        "Helper output must sum to input total per (s, t)."
    )
    assert not np.array_equal(per_wave_traffic_np[0], per_wave_traffic_np[1]), (
        "Per-wave matrices MUST differ — otherwise the test is moot."
    )

    # Step 3: TILE-pad zero edges (structural greedy artifact at W=4 + 2 waves;
    # see analysis comment above the test).
    eff_count_np, eff_send_off_np, eff_recv_off_np, per_wave_real = (
        _w4_p1hs_compute_eff_count_and_offsets(
            per_wave_traffic_np, world_size=world_size, tile=tile,
        )
    )
    out_recv_size = max_waves * per_wave_real

    # Sender buffer: each sender writes wave-distinct sentinels into per-wave
    # slot at offset `w * per_wave_real + send_off[w, src, dst]`. Same layout
    # as the recv buffer so send/recv offsets share the wave-slot base.
    sorted_row_np = np.zeros((world_size, out_recv_size), dtype=np.int32)
    for r in range(world_size):
        for w in range(max_waves):
            for p in range(world_size):
                start = w * per_wave_real + int(eff_send_off_np[w, r, p])
                cnt = int(eff_count_np[w, r, p])
                sorted_row_np[r, start:start + cnt] = (
                    _w4_p1hs_sentinel(w, r, p)
                )

    HBM = pltpu.MemorySpace.HBM
    SMEM = pltpu.MemorySpace.SMEM

    kernel = _make_w4_p1hs_kernel(
        world_size=world_size, max_waves=max_waves,
        per_wave_real=per_wave_real, tile=tile,
    )

    P = jax.sharding.PartitionSpec

    def per_device(src, eff_count, send_off, recv_off):
        recv = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((out_recv_size,), jnp.int32),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[
                    pl.BlockSpec(memory_space=HBM),
                    pl.BlockSpec(memory_space=SMEM),
                    pl.BlockSpec(memory_space=SMEM),
                    pl.BlockSpec(memory_space=SMEM),
                ],
                out_specs=pl.BlockSpec(memory_space=HBM),
                scratch_shapes=(
                    pltpu.VMEM((per_wave_real,), jnp.int32),
                    pltpu.SemaphoreType.REGULAR((max_waves, 2)),
                    pltpu.SemaphoreType.DMA,
                    pltpu.SemaphoreType.DMA,
                ),
            ),
            interpret=False,
        )(src[0], eff_count, send_off, recv_off)
        return recv[None, ...]

    sharded = jax.shard_map(
        per_device, mesh=mesh,
        in_specs=(P("x"), P(), P(), P()),
        out_specs=P("x"),
        check_vma=False,
    )
    sharding_x = jax.sharding.NamedSharding(mesh, P("x"))
    sharding_rep = jax.sharding.NamedSharding(mesh, P())
    src = jax.device_put(jnp.array(sorted_row_np), sharding_x)
    eff_count = jax.device_put(jnp.array(eff_count_np), sharding_rep)
    send_off = jax.device_put(jnp.array(eff_send_off_np), sharding_rep)
    recv_off = jax.device_put(jnp.array(eff_recv_off_np), sharding_rep)
    out = sharded(src, eff_count, send_off, recv_off)
    return (
        np.array(out),
        total_traffic_np,
        per_wave_traffic_np,
        eff_count_np,
        eff_recv_off_np,
        max_waves,
        per_wave_real,
        world_size,
        tile,
    )


@unittest.skipUnless(_has_tpu(), "Litmus test requires real TPU")
class W4Phase1HelperSplitLitmus(unittest.TestCase):
    """v66o: v66n-shape kernel + helper-driven wave split on a SYMMETRIC/
    rotated total traffic matrix. Tests whether `_split_traffic_into_waves`
    itself (producing per-wave-different chunked traffic with no skewed-input
    pipeline) is the load-bearing trigger for the W=4 multi-iter hang.
    """

    def setUp(self) -> None:
        if jax.device_count() < 4:
            self.skipTest(f"Need 4 devices, have {jax.device_count()}")

    def test_w4_p1_helper_split_symmetric(self) -> None:
        """v66o: phase-1 only, helper-driven split, symmetric total matrix."""
        import threading

        result_box: list = []
        error_box: list = []

        def runner():
            try:
                result_box.append(_run_w4_p1hs_litmus())
            except BaseException as e:  # noqa: BLE001
                error_box.append(e)

        t = threading.Thread(target=runner, daemon=True)
        t.start()
        t.join(timeout=300.0)
        if t.is_alive():
            self.fail(
                "v66o (W=4 phase-1 helper-driven split + TILE-pad on "
                "symmetric matrix) HUNG after 300s; helper-driven "
                "per-wave-different chunked traffic IS the load-bearing "
                "trigger for the W=4 multi-iter deadlock."
            )
        if error_box:
            raise error_box[0]

        (recv, total_np, per_wave_traffic_np, eff_count_np, recv_off_np,
         max_waves, per_wave_real, world_size, tile) = result_box[0]

        print(
            f"[LITMUS_W4_P1HS] max_waves={max_waves} "
            f"per_wave_real={per_wave_real} world_size={world_size} "
            f"tile={tile} base={_W4_P1HS_BASE} "
            f"recv_capacity={_W4_P1HS_RECV_CAPACITY}",
            flush=True,
        )
        print(f"[LITMUS_W4_P1HS] total_traffic[r, p]:", flush=True)
        for r in range(world_size):
            print(f"  row r={r}: {total_np[r].tolist()}", flush=True)
        for w in range(max_waves):
            print(
                f"[LITMUS_W4_P1HS] per_wave_traffic[w={w}, s, t]:",
                flush=True,
            )
            for s in range(world_size):
                print(
                    f"  row s={s}: {per_wave_traffic_np[w, s].tolist()}",
                    flush=True,
                )
            print(
                f"[LITMUS_W4_P1HS] eff_count[w={w}, s, t] "
                f"(TILE-padded):", flush=True,
            )
            for s in range(world_size):
                print(
                    f"  row s={s}: {eff_count_np[w, s].tolist()}",
                    flush=True,
                )
        print(
            f"[LITMUS_W4_P1HS] per_wave_matrices_differ="
            f"{not np.array_equal(per_wave_traffic_np[0], per_wave_traffic_np[1])}",
            flush=True,
        )
        for r in range(world_size):
            for w in range(max_waves):
                for s in range(world_size):
                    cnt = int(eff_count_np[w, s, r])
                    off = w * per_wave_real + int(recv_off_np[w, s, r])
                    first4 = recv[r, off:off + 4].tolist()
                    print(
                        f"[LITMUS_W4_P1HS] rank={r} wave={w} src={s} "
                        f"cnt={cnt} off={off} first4="
                        f"{[hex(int(x)) for x in first4]}",
                        flush=True,
                    )

        mismatches = []
        for r in range(world_size):
            for w in range(max_waves):
                for s in range(world_size):
                    cnt = int(eff_count_np[w, s, r])
                    off = w * per_wave_real + int(recv_off_np[w, s, r])
                    expected = _w4_p1hs_sentinel(w, s, r)
                    slot = recv[r, off:off + cnt]
                    if not np.all(slot == expected):
                        actual_first8 = slot[:8].tolist()
                        mismatches.append(
                            f"rank={r} wave={w} src={s} cnt={cnt} off={off} "
                            f"expected={hex(expected)} actual_first8="
                            f"{[hex(int(x)) for x in actual_first8]}"
                        )
        if mismatches:
            self.fail(
                "v66o helper-driven split DATA MISMATCH:\n  "
                + "\n  ".join(mismatches)
            )


# ---------------------------------------------------------------------------
# v66p: MANUAL per-wave-different traffic with NO TILE-pad zero edges. The
# goal is to surgically isolate "per-wave-different per-pair counts" from
# "TILE-pad zero edges" — both of which were present in v66o (HANG) but only
# the latter was unavoidable from `_split_traffic_into_waves`.
#
# Setup:
#   1. Two manually-constructed per-wave traffic matrices, both per-rank-
#      uniform (every rank has identical row in any given wave).
#      Wave 0 row = BASE_W0 = (128, 256, 128, 256)
#      Wave 1 row = BASE_W1 = (256, 128, 256, 128)  -- different from W0
#   2. ALL entries > 0 (no zero edges, so no TILE-pad needed at all).
#   3. Per-receiver per-wave incoming = W * BASE_W[r] (per-rank-uniform).
#      Wave 0 receiver r=0: 4*128=512, r=1: 4*256=1024, r=2: 4*128=512,
#      r=3: 4*256=1024. Wave 1: r=0: 4*256=1024, r=1: 4*128=512, etc.
#      per_wave_recv_size = 1024 fits both.
#   4. Send/recv offsets built directly from the per-wave matrices via
#      cumsum (no helper). All offsets in real region, no dummy region.
#   5. Same v66n-shape kernel (pre-barrier, unconditional per-peer DMA,
#      post-barrier). Sentinel prefix 0x0F.
#
# Stop conditions:
#   PASS  -> per-wave-different alone is NOT the trigger. TILE-pad of zero
#            edges (or its specific interaction with multi-wave) IS the
#            trigger for v66o/v66k.
#   HANG  -> per-wave-different per-pair counts at W=4 multi-iter IS the
#            trigger, regardless of whether TILE-pad zero edges are present.
# ---------------------------------------------------------------------------


_W4_P1MPWD_TILE = 128
_W4_P1MPWD_MAX_WAVES = 2
_W4_P1MPWD_WORLD_SIZE = 4
# Wave-0 and wave-1 per-rank-uniform send patterns. All entries > 0 and
# TILE-aligned. Wave-1 differs from wave-0 (per-peer order swapped).
_W4_P1MPWD_BASE_W0 = (128, 256, 128, 256)
_W4_P1MPWD_BASE_W1 = (256, 128, 256, 128)
# Per-receiver per-wave incoming = W * BASE_W[r]; max across both waves and
# receivers = 4 * 256 = 1024.
_W4_P1MPWD_PER_WAVE_RECV_SIZE = 1024
# 2-wave recv capacity = 2048 (TILE-aligned).
_W4_P1MPWD_RECV_CAPACITY = 2048
# Per-rank send total per wave = sum(BASE_W) = 768; 2 waves = 1536.
_W4_P1MPWD_MAX_PADDED = 2048


def _w4_p1mpwd_sentinel(w: int, src: int, dst: int) -> int:
    """Wave/source/dest-distinct sentinel (v66p-distinct prefix 0x0F)."""
    return 0x0F000000 + (w << 16) + (src << 8) + dst


def _make_w4_p1mpwd_kernel(world_size: int, max_waves: int,
                           per_wave_recv_size: int):
    """Kernel: identical structure to v66n's `_make_w4_p1vr_kernel`."""

    def kernel(
        sorted_row_ref,    # [_W4_P1MPWD_MAX_PADDED] int32 HBM
        traffic_ref,       # [max_waves, W, W] int32 SMEM
        send_off_ref,      # [max_waves, W, W] int32 SMEM
        recv_off_ref,      # [max_waves, W, W] int32 SMEM
        out_recv_ref,      # [_W4_P1MPWD_RECV_CAPACITY] int32 HBM
        zero_int_vmem,     # [per_wave_recv_size] int32 VMEM zero buffer
        barrier_sems,      # REGULAR (max_waves, 2)
        send_sem,          # DMA send sem
        recv_sem,          # DMA recv sem
    ):
        my_rank = lax.axis_index("x")

        zero_int_vmem[...] = jnp.zeros_like(zero_int_vmem)

        def wave_body(w, _carry):
            wave_recv_slot = out_recv_ref.at[
                pl.ds(w * per_wave_recv_size, per_wave_recv_size)
            ]
            pltpu.sync_copy(zero_int_vmem, wave_recv_slot)

            pre_sem = barrier_sems.at[w, 0]
            for peer in range(world_size):
                pl.semaphore_signal(
                    pre_sem, 1,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
            pl.semaphore_wait(pre_sem, world_size)

            for peer in range(world_size):
                send_count = pl.multiple_of(
                    traffic_ref[w, my_rank, peer], _W4_P1MPWD_TILE
                )
                send_start = pl.multiple_of(
                    send_off_ref[w, my_rank, peer], _W4_P1MPWD_TILE
                )
                remote_recv_start = pl.multiple_of(
                    recv_off_ref[w, peer, my_rank], _W4_P1MPWD_TILE
                )

                d = pltpu.make_async_remote_copy(
                    src_ref=sorted_row_ref.at[pl.ds(send_start, send_count)],
                    dst_ref=out_recv_ref.at[
                        pl.ds(remote_recv_start, send_count)
                    ],
                    send_sem=send_sem, recv_sem=recv_sem,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
                d.start()
                d.wait()

            post_sem = barrier_sems.at[w, 1]
            for peer in range(world_size):
                pl.semaphore_signal(
                    post_sem, 1,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
            pl.semaphore_wait(post_sem, world_size)
            return None

        lax.fori_loop(0, max_waves, wave_body, None)

    return kernel


def _run_w4_p1mpwd_litmus():
    """Run W=4 phase-1 litmus with MANUAL per-wave-different per-rank-uniform
    traffic, NO zero edges, NO TILE-pad."""
    world_size = _W4_P1MPWD_WORLD_SIZE
    max_waves = _W4_P1MPWD_MAX_WAVES
    per_wave_recv_size = _W4_P1MPWD_PER_WAVE_RECV_SIZE
    devices = jax.devices()[:world_size]
    mesh = jax.sharding.Mesh(np.array(devices), ("x",))

    # Per-wave per-rank-uniform traffic: every rank shares the same row.
    bases = (_W4_P1MPWD_BASE_W0, _W4_P1MPWD_BASE_W1)
    per_wave_traffic_np = np.zeros(
        (max_waves, world_size, world_size), dtype=np.int32,
    )
    for w in range(max_waves):
        for r in range(world_size):
            for p in range(world_size):
                per_wave_traffic_np[w, r, p] = bases[w][p]

    # Sanity: all entries > 0, TILE-aligned, matrices differ.
    assert (per_wave_traffic_np > 0).all(), (
        "v66p must have NO zero edges."
    )
    assert (per_wave_traffic_np % _W4_P1MPWD_TILE == 0).all(), (
        "v66p entries must be TILE-aligned."
    )
    assert not np.array_equal(
        per_wave_traffic_np[0], per_wave_traffic_np[1]
    ), "v66p per-wave matrices MUST differ."
    # Each per-wave row must repeat across senders (per-rank-uniform).
    for w in range(max_waves):
        for r in range(1, world_size):
            assert (
                per_wave_traffic_np[w, r] == per_wave_traffic_np[w, 0]
            ).all(), "v66p must be per-rank-uniform within a wave."

    # Per-receiver per-wave sum check: must fit in per_wave_recv_size.
    for w in range(max_waves):
        for t in range(world_size):
            col = int(per_wave_traffic_np[w, :, t].sum())
            assert col <= per_wave_recv_size, (
                f"per_wave_recv_size={per_wave_recv_size} too small for "
                f"wave={w} receiver={t} col_sum={col}"
            )

    # Send offsets: cumulative per source within a wave (real region only).
    per_wave_send_offsets_np = np.zeros(
        (max_waves, world_size, world_size), dtype=np.int32,
    )
    # Recv offsets on receiver t from src s: cumulative per dest within a wave.
    per_wave_recv_offsets_np = np.zeros(
        (max_waves, world_size, world_size), dtype=np.int32,
    )
    for w in range(max_waves):
        for s in range(world_size):
            cum = 0
            for t in range(world_size):
                per_wave_send_offsets_np[w, s, t] = (
                    w * per_wave_recv_size + cum
                )
                cum += int(per_wave_traffic_np[w, s, t])
        for t in range(world_size):
            cum = 0
            for s in range(world_size):
                per_wave_recv_offsets_np[w, t, s] = (
                    w * per_wave_recv_size + cum
                )
                cum += int(per_wave_traffic_np[w, s, t])

    # Per-rank sorted_row, written with wave/src/dst sentinels.
    sorted_row_padded_np = np.zeros(
        (world_size, _W4_P1MPWD_MAX_PADDED), dtype=np.int32,
    )
    for r in range(world_size):
        for w in range(max_waves):
            for p in range(world_size):
                start = int(per_wave_send_offsets_np[w, r, p])
                cnt = int(per_wave_traffic_np[w, r, p])
                sorted_row_padded_np[r, start:start + cnt] = (
                    _w4_p1mpwd_sentinel(w, r, p)
                )

    HBM = pltpu.MemorySpace.HBM
    SMEM = pltpu.MemorySpace.SMEM

    kernel = _make_w4_p1mpwd_kernel(
        world_size=world_size, max_waves=max_waves,
        per_wave_recv_size=per_wave_recv_size,
    )

    P = jax.sharding.PartitionSpec

    def per_device(src, traffic, send_off, recv_off):
        recv = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(
                (_W4_P1MPWD_RECV_CAPACITY,), jnp.int32,
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[
                    pl.BlockSpec(memory_space=HBM),   # sorted_row
                    pl.BlockSpec(memory_space=SMEM),  # traffic
                    pl.BlockSpec(memory_space=SMEM),  # send_off
                    pl.BlockSpec(memory_space=SMEM),  # recv_off
                ],
                out_specs=pl.BlockSpec(memory_space=HBM),
                scratch_shapes=(
                    pltpu.VMEM((per_wave_recv_size,), jnp.int32),
                    pltpu.SemaphoreType.REGULAR((max_waves, 2)),
                    pltpu.SemaphoreType.DMA,
                    pltpu.SemaphoreType.DMA,
                ),
            ),
            interpret=False,
        )(src[0], traffic, send_off, recv_off)
        return recv[None, ...]

    sharded = jax.shard_map(
        per_device, mesh=mesh,
        in_specs=(P("x"), P(), P(), P()),
        out_specs=P("x"),
        check_vma=False,
    )

    sharding_x = jax.sharding.NamedSharding(mesh, P("x"))
    sharding_rep = jax.sharding.NamedSharding(mesh, P())
    src = jax.device_put(jnp.array(sorted_row_padded_np), sharding_x)
    traffic = jax.device_put(jnp.array(per_wave_traffic_np), sharding_rep)
    send_off = jax.device_put(
        jnp.array(per_wave_send_offsets_np), sharding_rep,
    )
    recv_off = jax.device_put(
        jnp.array(per_wave_recv_offsets_np), sharding_rep,
    )

    out = sharded(src, traffic, send_off, recv_off)
    return (
        np.array(out),  # [W, _W4_P1MPWD_RECV_CAPACITY]
        per_wave_traffic_np,
        per_wave_recv_offsets_np,
        max_waves,
        per_wave_recv_size,
        world_size,
    )


@unittest.skipUnless(_has_tpu(), "Litmus test requires real TPU")
class W4Phase1ManualPerWaveDiffLitmus(unittest.TestCase):
    """v66p: MANUAL per-wave-different per-rank-uniform traffic, NO zero
    edges, NO TILE-pad. Surgically isolates "per-wave-different per-pair
    counts" from "TILE-pad zero edges" (both present in v66o which HUNG).
    """

    def setUp(self) -> None:
        if jax.device_count() < 4:
            self.skipTest(
                f"Need 4 devices, have {jax.device_count()}"
            )

    def test_w4_p1_manual_per_wave_different(self) -> None:
        """v66p: phase-1 only, manual per-wave-different traffic, no zeros."""
        import threading

        result_box: list = []
        error_box: list = []

        def runner():
            try:
                result_box.append(_run_w4_p1mpwd_litmus())
            except BaseException as e:  # noqa: BLE001
                error_box.append(e)

        t = threading.Thread(target=runner, daemon=True)
        t.start()
        t.join(timeout=300.0)
        if t.is_alive():
            self.fail(
                "v66p (W=4 phase-1 manual per-wave-different traffic, no "
                "zero edges) HUNG after 300s; per-wave-different per-pair "
                "counts at W=4 multi-iter ARE the load-bearing trigger, "
                "regardless of whether TILE-pad zero edges are present."
            )
        if error_box:
            raise error_box[0]

        (recv, traffic_np, recv_off_np, max_waves, per_wave_recv_size,
         world_size) = result_box[0]

        # Diagnostic header showing both per-wave matrices.
        print(
            f"[LITMUS_W4_P1MPWD] max_waves={max_waves} "
            f"per_wave_recv_size={per_wave_recv_size} "
            f"world_size={world_size} base_w0={_W4_P1MPWD_BASE_W0} "
            f"base_w1={_W4_P1MPWD_BASE_W1} "
            f"recv_capacity={_W4_P1MPWD_RECV_CAPACITY}",
            flush=True,
        )
        for w in range(max_waves):
            print(
                f"[LITMUS_W4_P1MPWD] per_wave_traffic[w={w}, s, t]:",
                flush=True,
            )
            for s in range(world_size):
                print(
                    f"  row s={s}: {traffic_np[w, s].tolist()}",
                    flush=True,
                )
        print(
            f"[LITMUS_W4_P1MPWD] per_wave_matrices_differ="
            f"{not np.array_equal(traffic_np[0], traffic_np[1])}",
            flush=True,
        )
        print(
            f"[LITMUS_W4_P1MPWD] all_entries_positive="
            f"{(traffic_np > 0).all()}",
            flush=True,
        )
        for r in range(world_size):
            for w in range(max_waves):
                for s in range(world_size):
                    cnt = int(traffic_np[w, s, r])
                    off = int(recv_off_np[w, r, s])
                    first4 = recv[r, off:off + 4].tolist()
                    print(
                        f"[LITMUS_W4_P1MPWD] rank={r} wave={w} src={s} "
                        f"cnt={cnt} off={off} first4="
                        f"{[hex(int(x)) for x in first4]}",
                        flush=True,
                    )

        mismatches = []
        for r in range(world_size):
            for w in range(max_waves):
                for s in range(world_size):
                    cnt = int(traffic_np[w, s, r])
                    off = int(recv_off_np[w, r, s])
                    expected = _w4_p1mpwd_sentinel(w, s, r)
                    slot = recv[r, off:off + cnt]
                    if not np.all(slot == expected):
                        actual_first8 = slot[:8].tolist()
                        mismatches.append(
                            f"rank={r} wave={w} src={s} cnt={cnt} off={off} "
                            f"expected={hex(expected)} actual_first8="
                            f"{[hex(int(x)) for x in actual_first8]}"
                        )
        if mismatches:
            self.fail(
                "v66p phase-1 manual per-wave-different DATA MISMATCH:\n  "
                + "\n  ".join(mismatches)
            )


# ---------------------------------------------------------------------------
# v66q: per-wave EVEN-SPLIT workaround (WO#4)
#
# Hypothesis (per v66p HANG converse): if per-wave per-pair DMA traffic is
# IDENTICAL across waves, the W=4 multi-iter Pallas kernel does NOT deadlock,
# even when the underlying TOTAL traffic is per-rank-different (skewed).
#
# Construction:
#   1. Total traffic uses the v66o/v66n rotation pattern:
#        total[r, p] = BASE[(p - r) % W], BASE = (128, 256, 384, 512)
#      Per-rank-different totals (matches realistic skewed-input case).
#   2. Pad each total[r, p] up to the next multiple of MAX_WAVES * TILE = 256.
#   3. Even-split: per_wave_traffic[w, r, p] = total_padded[r, p] // MAX_WAVES
#      ⇒ per_wave_traffic[0] == per_wave_traffic[1] element-wise.
#   4. send/recv offsets: cumulative sums on per-wave matrices (which are
#      identical, so offsets are identical across waves).
#   5. All DMAs target REAL region. No dummy region.
#   6. Same kernel body as v66p / v66n.
#
# Stop conditions:
#   PASS  -> WO#4 (per-wave even-split) is a viable production workaround for
#            the W=4 multi-iter hang. Cost: up to (MAX_WAVES * TILE - 1) extra
#            ints of bandwidth per (src, dst) pair worst-case.
#   HANG  -> per-wave-identical traffic with realistic per-rank-different
#            totals can still hang. Fingerprint needs another refinement.
# ---------------------------------------------------------------------------


_W4_P1ES_TILE = 128
_W4_P1ES_MAX_WAVES = 2
_W4_P1ES_WORLD_SIZE = 4
# Skewed-like total traffic uses v66n's rotation BASE (no period-2 symmetry).
_W4_P1ES_BASE = (128, 256, 384, 512)
# Rotation row sums all equal sum(BASE) = 1280 ints per rank. After padding to
# multiples of MAX_WAVES * TILE = 256: BASE[0]=128 → 256; BASE[1]=256 → 256;
# BASE[2]=384 → 512; BASE[3]=512 → 512. Per-rank padded row sum = 256+256+
# 512+512 = 1536 ints; per-wave per-rank = 1536/2 = 768 ints.
# Per-wave per-receiver col sum = 768 ints (cyclic invariance).
_W4_P1ES_PER_WAVE_RECV_SIZE = 768
# 2-wave recv capacity = 1536 (TILE-aligned).
_W4_P1ES_RECV_CAPACITY = 1536
# Per-rank send total across all waves = 1536; pad to 2048 for headroom.
_W4_P1ES_MAX_PADDED = 2048


def _w4_p1es_sentinel(w: int, src: int, dst: int) -> int:
    """Wave/source/dest-distinct sentinel (v66q-distinct prefix 0x10)."""
    return 0x10000000 + (w << 16) + (src << 8) + dst


def _w4_p1es_compute_traffic_and_offsets(world_size: int, max_waves: int):
    """Compute total, per_wave_traffic, send/recv offsets for v66q.

    Returns:
        total_traffic_np    [W, W] int32     -- pre-pad rotation totals
        total_padded_np     [W, W] int32     -- padded to multiple of
                                                max_waves * TILE
        per_wave_traffic_np [max_waves, W, W] int32 -- IDENTICAL across waves
        per_wave_send_offsets_np [max_waves, W, W] int32
        per_wave_recv_offsets_np [max_waves, W, W] int32
    """
    base = _W4_P1ES_BASE
    pad_mult = max_waves * _W4_P1ES_TILE  # 256

    # 1. total_traffic[r, p] = BASE[(p - r) % W]
    total_traffic_np = np.zeros((world_size, world_size), dtype=np.int32)
    for r in range(world_size):
        for p in range(world_size):
            total_traffic_np[r, p] = base[(p - r) % world_size]

    # 2. Pad each total[r, p] up to next multiple of pad_mult.
    total_padded_np = np.zeros((world_size, world_size), dtype=np.int32)
    for r in range(world_size):
        for p in range(world_size):
            v = int(total_traffic_np[r, p])
            rem = v % pad_mult
            total_padded_np[r, p] = v if rem == 0 else v + (pad_mult - rem)

    # 3. Even-split: per_wave_traffic[w, r, p] = total_padded[r, p] // max_waves
    per_wave_traffic_np = np.zeros(
        (max_waves, world_size, world_size), dtype=np.int32,
    )
    for w in range(max_waves):
        per_wave_traffic_np[w] = total_padded_np // max_waves

    # 4. Cumulative offsets within each wave (identical across waves since
    #    per-wave traffic is identical).
    per_wave_send_offsets_np = np.zeros(
        (max_waves, world_size, world_size), dtype=np.int32,
    )
    per_wave_recv_offsets_np = np.zeros(
        (max_waves, world_size, world_size), dtype=np.int32,
    )
    for w in range(max_waves):
        for s in range(world_size):
            cum = 0
            for t in range(world_size):
                per_wave_send_offsets_np[w, s, t] = (
                    w * _W4_P1ES_PER_WAVE_RECV_SIZE + cum
                )
                cum += int(per_wave_traffic_np[w, s, t])
        for t in range(world_size):
            cum = 0
            for s in range(world_size):
                per_wave_recv_offsets_np[w, t, s] = (
                    w * _W4_P1ES_PER_WAVE_RECV_SIZE + cum
                )
                cum += int(per_wave_traffic_np[w, s, t])

    return (
        total_traffic_np, total_padded_np, per_wave_traffic_np,
        per_wave_send_offsets_np, per_wave_recv_offsets_np,
    )


def _make_w4_p1es_kernel(world_size: int, max_waves: int,
                         per_wave_recv_size: int):
    """Kernel: identical structure to v66p / v66n's kernel."""

    def kernel(
        sorted_row_ref,    # [_W4_P1ES_MAX_PADDED] int32 HBM
        traffic_ref,       # [max_waves, W, W] int32 SMEM
        send_off_ref,      # [max_waves, W, W] int32 SMEM
        recv_off_ref,      # [max_waves, W, W] int32 SMEM
        out_recv_ref,      # [_W4_P1ES_RECV_CAPACITY] int32 HBM
        zero_int_vmem,     # [per_wave_recv_size] int32 VMEM zero buffer
        barrier_sems,      # REGULAR (max_waves, 2)
        send_sem,          # DMA send sem
        recv_sem,          # DMA recv sem
    ):
        my_rank = lax.axis_index("x")

        zero_int_vmem[...] = jnp.zeros_like(zero_int_vmem)

        def wave_body(w, _carry):
            wave_recv_slot = out_recv_ref.at[
                pl.ds(w * per_wave_recv_size, per_wave_recv_size)
            ]
            pltpu.sync_copy(zero_int_vmem, wave_recv_slot)

            pre_sem = barrier_sems.at[w, 0]
            for peer in range(world_size):
                pl.semaphore_signal(
                    pre_sem, 1,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
            pl.semaphore_wait(pre_sem, world_size)

            for peer in range(world_size):
                send_count = pl.multiple_of(
                    traffic_ref[w, my_rank, peer], _W4_P1ES_TILE
                )
                send_start = pl.multiple_of(
                    send_off_ref[w, my_rank, peer], _W4_P1ES_TILE
                )
                remote_recv_start = pl.multiple_of(
                    recv_off_ref[w, peer, my_rank], _W4_P1ES_TILE
                )

                d = pltpu.make_async_remote_copy(
                    src_ref=sorted_row_ref.at[pl.ds(send_start, send_count)],
                    dst_ref=out_recv_ref.at[
                        pl.ds(remote_recv_start, send_count)
                    ],
                    send_sem=send_sem, recv_sem=recv_sem,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
                d.start()
                d.wait()

            post_sem = barrier_sems.at[w, 1]
            for peer in range(world_size):
                pl.semaphore_signal(
                    post_sem, 1,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
            pl.semaphore_wait(post_sem, world_size)
            return None

        lax.fori_loop(0, max_waves, wave_body, None)

    return kernel


def _run_w4_p1es_litmus():
    """Run W=4 phase-1 litmus with per-wave EVEN-SPLIT workaround."""
    world_size = _W4_P1ES_WORLD_SIZE
    max_waves = _W4_P1ES_MAX_WAVES
    per_wave_recv_size = _W4_P1ES_PER_WAVE_RECV_SIZE
    devices = jax.devices()[:world_size]
    mesh = jax.sharding.Mesh(np.array(devices), ("x",))

    (total_traffic_np, total_padded_np, per_wave_traffic_np,
     per_wave_send_offsets_np, per_wave_recv_offsets_np) = (
        _w4_p1es_compute_traffic_and_offsets(world_size, max_waves)
    )

    # Sanity checks.
    assert (per_wave_traffic_np > 0).all(), (
        "v66q must have NO zero per-wave entries (rotation BASE > 0 and pad "
        "preserves that)."
    )
    assert (per_wave_traffic_np % _W4_P1ES_TILE == 0).all(), (
        "v66q per-wave entries must be TILE-aligned."
    )
    assert np.array_equal(
        per_wave_traffic_np[0], per_wave_traffic_np[1]
    ), "v66q per-wave matrices MUST be identical (the workaround invariant)."
    # Per-wave per-receiver sum must fit in per_wave_recv_size.
    for w in range(max_waves):
        for t in range(world_size):
            col = int(per_wave_traffic_np[w, :, t].sum())
            assert col <= per_wave_recv_size, (
                f"per_wave_recv_size={per_wave_recv_size} too small for "
                f"wave={w} receiver={t} col_sum={col}"
            )
    # Per-rank send total across waves must fit in MAX_PADDED.
    for r in range(world_size):
        send_total = int(
            sum(per_wave_traffic_np[w, r, :].sum() for w in range(max_waves))
        )
        assert send_total <= _W4_P1ES_MAX_PADDED, (
            f"_W4_P1ES_MAX_PADDED={_W4_P1ES_MAX_PADDED} too small for "
            f"rank={r} send_total={send_total}"
        )

    # Per-rank sorted_row, written with wave/src/dst sentinels.
    sorted_row_padded_np = np.zeros(
        (world_size, _W4_P1ES_MAX_PADDED), dtype=np.int32,
    )
    for r in range(world_size):
        for w in range(max_waves):
            for p in range(world_size):
                start = int(per_wave_send_offsets_np[w, r, p])
                cnt = int(per_wave_traffic_np[w, r, p])
                sorted_row_padded_np[r, start:start + cnt] = (
                    _w4_p1es_sentinel(w, r, p)
                )

    HBM = pltpu.MemorySpace.HBM
    SMEM = pltpu.MemorySpace.SMEM

    kernel = _make_w4_p1es_kernel(
        world_size=world_size, max_waves=max_waves,
        per_wave_recv_size=per_wave_recv_size,
    )

    P = jax.sharding.PartitionSpec

    def per_device(src, traffic, send_off, recv_off):
        recv = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(
                (_W4_P1ES_RECV_CAPACITY,), jnp.int32,
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[
                    pl.BlockSpec(memory_space=HBM),   # sorted_row
                    pl.BlockSpec(memory_space=SMEM),  # traffic
                    pl.BlockSpec(memory_space=SMEM),  # send_off
                    pl.BlockSpec(memory_space=SMEM),  # recv_off
                ],
                out_specs=pl.BlockSpec(memory_space=HBM),
                scratch_shapes=(
                    pltpu.VMEM((per_wave_recv_size,), jnp.int32),
                    pltpu.SemaphoreType.REGULAR((max_waves, 2)),
                    pltpu.SemaphoreType.DMA,
                    pltpu.SemaphoreType.DMA,
                ),
            ),
            interpret=False,
        )(src[0], traffic, send_off, recv_off)
        return recv[None, ...]

    sharded = jax.shard_map(
        per_device, mesh=mesh,
        in_specs=(P("x"), P(), P(), P()),
        out_specs=P("x"),
        check_vma=False,
    )

    sharding_x = jax.sharding.NamedSharding(mesh, P("x"))
    sharding_rep = jax.sharding.NamedSharding(mesh, P())
    src = jax.device_put(jnp.array(sorted_row_padded_np), sharding_x)
    traffic = jax.device_put(jnp.array(per_wave_traffic_np), sharding_rep)
    send_off = jax.device_put(
        jnp.array(per_wave_send_offsets_np), sharding_rep,
    )
    recv_off = jax.device_put(
        jnp.array(per_wave_recv_offsets_np), sharding_rep,
    )

    out = sharded(src, traffic, send_off, recv_off)
    return (
        np.array(out),
        total_traffic_np,
        total_padded_np,
        per_wave_traffic_np,
        per_wave_recv_offsets_np,
        max_waves,
        per_wave_recv_size,
        world_size,
    )


@unittest.skipUnless(_has_tpu(), "Litmus test requires real TPU")
class W4Phase1EvenSplitLitmus(unittest.TestCase):
    """v66q: per-wave EVEN-SPLIT workaround (WO#4).

    Total traffic uses v66n's per-rank-different rotation pattern (skewed-like).
    Pad each total to multiple of max_waves * TILE = 256, then split evenly so
    every wave sees identical per-pair counts. Predicted PASS per v66p's
    converse evidence.
    """

    def setUp(self) -> None:
        if jax.device_count() < 4:
            self.skipTest(
                f"Need 4 devices, have {jax.device_count()}"
            )

    def test_w4_p1_even_split_workaround(self) -> None:
        """v66q: per-wave-identical traffic via even split of padded totals."""
        import threading

        result_box: list = []
        error_box: list = []

        def runner():
            try:
                result_box.append(_run_w4_p1es_litmus())
            except BaseException as e:  # noqa: BLE001
                error_box.append(e)

        t = threading.Thread(target=runner, daemon=True)
        t.start()
        t.join(timeout=300.0)
        if t.is_alive():
            self.fail(
                "v66q (W=4 phase-1 per-wave EVEN-SPLIT workaround, identical "
                "per-wave traffic with skewed rotation totals) HUNG after "
                "300s; even per-wave-identical traffic with realistic "
                "per-rank-different totals can still hang. Fingerprint needs "
                "another refinement."
            )
        if error_box:
            raise error_box[0]

        (recv, total_np, total_padded_np, traffic_np, recv_off_np,
         max_waves, per_wave_recv_size, world_size) = result_box[0]

        # Diagnostic header.
        print(
            f"[LITMUS_W4_P1ES] max_waves={max_waves} "
            f"per_wave_recv_size={per_wave_recv_size} "
            f"world_size={world_size} base={_W4_P1ES_BASE} "
            f"pad_mult={max_waves * _W4_P1ES_TILE} "
            f"recv_capacity={_W4_P1ES_RECV_CAPACITY}",
            flush=True,
        )
        print("[LITMUS_W4_P1ES] total_traffic[r, p]:", flush=True)
        for r in range(world_size):
            print(f"  row r={r}: {total_np[r].tolist()}", flush=True)
        print("[LITMUS_W4_P1ES] total_padded[r, p]:", flush=True)
        for r in range(world_size):
            print(f"  row r={r}: {total_padded_np[r].tolist()}", flush=True)
        for w in range(max_waves):
            print(
                f"[LITMUS_W4_P1ES] per_wave_traffic[w={w}, s, t]:",
                flush=True,
            )
            for s in range(world_size):
                print(
                    f"  row s={s}: {traffic_np[w, s].tolist()}",
                    flush=True,
                )
        per_wave_identical = bool(
            np.array_equal(traffic_np[0], traffic_np[1])
        )
        print(
            f"[LITMUS_W4_P1ES] per_wave_matrices_identical="
            f"{per_wave_identical}",
            flush=True,
        )
        print(
            f"[LITMUS_W4_P1ES] all_entries_positive="
            f"{(traffic_np > 0).all()}",
            flush=True,
        )
        for r in range(world_size):
            for w in range(max_waves):
                for s in range(world_size):
                    cnt = int(traffic_np[w, s, r])
                    off = int(recv_off_np[w, r, s])
                    first4 = recv[r, off:off + 4].tolist()
                    print(
                        f"[LITMUS_W4_P1ES] rank={r} wave={w} src={s} "
                        f"cnt={cnt} off={off} first4="
                        f"{[hex(int(x)) for x in first4]}",
                        flush=True,
                    )

        mismatches = []
        for r in range(world_size):
            for w in range(max_waves):
                for s in range(world_size):
                    cnt = int(traffic_np[w, s, r])
                    off = int(recv_off_np[w, r, s])
                    expected = _w4_p1es_sentinel(w, s, r)
                    slot = recv[r, off:off + cnt]
                    if not np.all(slot == expected):
                        actual_first8 = slot[:8].tolist()
                        mismatches.append(
                            f"rank={r} wave={w} src={s} cnt={cnt} off={off} "
                            f"expected={hex(expected)} actual_first8="
                            f"{[hex(int(x)) for x in actual_first8]}"
                        )
        if mismatches:
            self.fail(
                "v66q phase-1 even-split DATA MISMATCH:\n  "
                + "\n  ".join(mismatches)
            )


# ---------------------------------------------------------------------------
# v66r: per-wave EVEN-SPLIT WITH PRODUCTION OFFSET DRIFT (WO#4 prod-shape gate)
#
# Hypothesis (per v66q PASS): if per-wave per-pair DMA traffic is identical
# across waves AND offsets drift across waves the way production does
# (cumsum drift inside each (s, t) chunk), the W=4 multi-iter Pallas kernel
# does NOT deadlock. v66q over-equalized by computing offsets via cumsum on
# per-wave-identical matrices (offsets identical across waves). Production
# `_split_traffic_into_waves` retains drift:
#     per_wave_send_offsets[w, s, t] = base_send[s, t]
#                                    + sum_{w' < w} per_wave[w', s, t]
#     per_wave_recv_offsets[w, t, s] = base_recv[t, s]
#                                    + sum_{w' < w} per_wave[w', s, t]
#
# Construction:
#   1. Total traffic: total[r, p] = BASE_padded[(p - r) % W] where
#      BASE_padded = (256, 256, 512, 512). This matches v66q's post-pad
#      rotation (BASE = (128, 256, 384, 512) padded to multiples of
#      max_waves * TILE = 256: 128->256, 256->256, 384->512, 512->512).
#   2. Per-wave: per_wave[w, s, t] = total[s, t] // 2 for all w. Identical
#      across waves => per_wave_traffic[0] == per_wave_traffic[1].
#   3. base_send[s, t] = sum_{t' < t} total[s, t']     (per-row cumsum, axis=1)
#      base_recv[t, s] = sum_{s' < s} total[s', t]     (per-col cumsum, axis=0)
#   4. per_wave_send_offsets[w, s, t] = base_send[s, t] + w * per_wave[s, t]
#      per_wave_recv_offsets[w, t, s] = base_recv[t, s] + w * per_wave[s, t]
#      These DRIFT across waves while the counts stay identical.
#   5. Source buffer per rank: max_padded = sum_t total[my_rank, t] = 1536.
#      Recv buffer per rank: recv_capacity_total = sum_s total[s, my_rank] =
#      1536 (cyclic invariance => per-rank totals match).
#   6. Sentinel: per (wave, src, dst). On receiver, within (s, t)'s slot of
#      length total[s, t], the first per_wave[s, t] ints carry wave-0
#      sentinel and the next per_wave[s, t] ints carry wave-1 sentinel
#      (proves the cumsum drift was actually exercised by the kernel).
#
# Difference vs v66q (the only change):
#   v66q layout: send buffer packs per-(w, s, t) chunks of size per_wave,
#     wave-major. wave-0 of (s,t) and wave-1 of (s,t) are at the SAME
#     per-pair offset within their wave block.
#   v66r layout: send buffer packs per-(s, t) chunks of size total[s,t].
#     wave-0 reads first per_wave ints of each chunk; wave-1 reads next
#     per_wave ints of the same chunk. Per-pair offset DRIFTS by per_wave.
#
# Stop conditions:
#   PASS  -> WO#4 with production-shape offset drift works. Counts-identical
#            is sufficient. WO#4 is genuinely production-viable.
#   HANG  -> Counts-identical is NOT sufficient when offsets drift across
#            waves. Fingerprint reduces to "per-pair offset drift across
#            waves" (or some interaction). New direction needed.
# ---------------------------------------------------------------------------


_W4_P1ESPG_TILE = 128
_W4_P1ESPG_MAX_WAVES = 2
_W4_P1ESPG_WORLD_SIZE = 4
# Post-pad rotation BASE: v66q's underlying BASE = (128, 256, 384, 512) padded
# to multiples of MAX_WAVES * TILE = 256, so 128->256, 256->256, 384->512,
# 512->512. Same totals as v66q ensures per-wave (= total // 2) is TILE
# multiple — required because the kernel uses pl.multiple_of(_, TILE).
_W4_P1ESPG_BASE_PADDED = (256, 256, 512, 512)
# Per-rank send total = 256+256+512+512 = 1536. Per-rank recv total identical
# by cyclic invariance (each receiver receives BASE_padded[(t-s) % W] from s).
_W4_P1ESPG_MAX_PADDED = sum(_W4_P1ESPG_BASE_PADDED)
_W4_P1ESPG_RECV_CAPACITY_TOTAL = sum(_W4_P1ESPG_BASE_PADDED)


def _w4_p1espg_sentinel(w: int, src: int, dst: int) -> int:
    """Wave/source/dest-distinct sentinel (v66r prefix 0x11)."""
    return 0x11000000 + (w << 16) + (src << 8) + dst


def _w4_p1espg_compute_traffic_and_offsets(world_size: int, max_waves: int):
    """Compute total, per_wave_traffic, send/recv offsets for v66r.

    Offsets follow the production cumsum-drift formula:
        per_wave_send_offsets[w, s, t] = base_send[s, t]
                                       + sum_{w' < w} per_wave[w', s, t]
        per_wave_recv_offsets[w, t, s] = base_recv[t, s]
                                       + sum_{w' < w} per_wave[w', s, t]

    Returns:
        total_traffic_np    [W, W] int32     -- post-pad rotation totals
        per_wave_traffic_np [max_waves, W, W] int32 -- IDENTICAL across waves
        per_wave_send_offsets_np [max_waves, W, W] int32 -- DRIFTS across waves
        per_wave_recv_offsets_np [max_waves, W, W] int32 -- DRIFTS across waves
        base_send_np        [W, W] int32
        base_recv_np        [W, W] int32     -- indexed [t, s]
    """
    base = _W4_P1ESPG_BASE_PADDED
    # 1. total_traffic[r, p] = BASE_padded[(p - r) % W]
    total_traffic_np = np.zeros((world_size, world_size), dtype=np.int32)
    for r in range(world_size):
        for p in range(world_size):
            total_traffic_np[r, p] = base[(p - r) % world_size]

    # 2. per_wave[w, s, t] = total[s, t] // max_waves (identical across waves).
    per_wave_traffic_np = np.zeros(
        (max_waves, world_size, world_size), dtype=np.int32,
    )
    for w in range(max_waves):
        per_wave_traffic_np[w] = total_traffic_np // max_waves

    # 3. base_send[s, t] = sum_{t' < t} total[s, t']  (per-row cumsum, axis=1).
    base_send_np = np.zeros((world_size, world_size), dtype=np.int32)
    for s in range(world_size):
        cum = 0
        for t in range(world_size):
            base_send_np[s, t] = cum
            cum += int(total_traffic_np[s, t])

    # 4. base_recv[t, s] = sum_{s' < s} total[s', t]  (per-col cumsum, axis=0).
    base_recv_np = np.zeros((world_size, world_size), dtype=np.int32)
    for t in range(world_size):
        cum = 0
        for s in range(world_size):
            base_recv_np[t, s] = cum
            cum += int(total_traffic_np[s, t])

    # 5. per-wave offsets with cumsum drift across waves.
    per_wave_send_offsets_np = np.zeros(
        (max_waves, world_size, world_size), dtype=np.int32,
    )
    per_wave_recv_offsets_np = np.zeros(
        (max_waves, world_size, world_size), dtype=np.int32,
    )
    for w in range(max_waves):
        for s in range(world_size):
            for t in range(world_size):
                # sum_{w' < w} per_wave[w', s, t]; since identical across
                # waves, this is w * per_wave_traffic_np[0, s, t].
                drift = w * int(per_wave_traffic_np[0, s, t])
                per_wave_send_offsets_np[w, s, t] = (
                    int(base_send_np[s, t]) + drift
                )
                per_wave_recv_offsets_np[w, t, s] = (
                    int(base_recv_np[t, s]) + drift
                )

    return (
        total_traffic_np, per_wave_traffic_np,
        per_wave_send_offsets_np, per_wave_recv_offsets_np,
        base_send_np, base_recv_np,
    )


def _make_w4_p1espg_kernel(world_size: int, max_waves: int,
                           recv_capacity_total: int):
    """Kernel: identical structure to v66q's kernel, single contiguous recv
    buffer of size recv_capacity_total (no per-wave slot striping)."""

    def kernel(
        sorted_row_ref,    # [_W4_P1ESPG_MAX_PADDED] int32 HBM
        traffic_ref,       # [max_waves, W, W] int32 SMEM
        send_off_ref,      # [max_waves, W, W] int32 SMEM
        recv_off_ref,      # [max_waves, W, W] int32 SMEM
        out_recv_ref,      # [recv_capacity_total] int32 HBM
        zero_int_vmem,     # [recv_capacity_total] int32 VMEM zero buffer
        barrier_sems,      # REGULAR (max_waves, 2)
        send_sem,          # DMA send sem
        recv_sem,          # DMA recv sem
    ):
        my_rank = lax.axis_index("x")

        zero_int_vmem[...] = jnp.zeros_like(zero_int_vmem)
        # Single contiguous recv buffer; zeroed once before any wave.
        pltpu.sync_copy(zero_int_vmem, out_recv_ref)

        def wave_body(w, _carry):
            pre_sem = barrier_sems.at[w, 0]
            for peer in range(world_size):
                pl.semaphore_signal(
                    pre_sem, 1,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
            pl.semaphore_wait(pre_sem, world_size)

            for peer in range(world_size):
                send_count = pl.multiple_of(
                    traffic_ref[w, my_rank, peer], _W4_P1ESPG_TILE
                )
                send_start = pl.multiple_of(
                    send_off_ref[w, my_rank, peer], _W4_P1ESPG_TILE
                )
                remote_recv_start = pl.multiple_of(
                    recv_off_ref[w, peer, my_rank], _W4_P1ESPG_TILE
                )

                d = pltpu.make_async_remote_copy(
                    src_ref=sorted_row_ref.at[pl.ds(send_start, send_count)],
                    dst_ref=out_recv_ref.at[
                        pl.ds(remote_recv_start, send_count)
                    ],
                    send_sem=send_sem, recv_sem=recv_sem,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
                d.start()
                d.wait()

            post_sem = barrier_sems.at[w, 1]
            for peer in range(world_size):
                pl.semaphore_signal(
                    post_sem, 1,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
            pl.semaphore_wait(post_sem, world_size)
            return None

        lax.fori_loop(0, max_waves, wave_body, None)

    return kernel


def _run_w4_p1espg_litmus():
    """Run W=4 phase-1 litmus with even-split + production offset drift."""
    world_size = _W4_P1ESPG_WORLD_SIZE
    max_waves = _W4_P1ESPG_MAX_WAVES
    recv_capacity_total = _W4_P1ESPG_RECV_CAPACITY_TOTAL
    devices = jax.devices()[:world_size]
    mesh = jax.sharding.Mesh(np.array(devices), ("x",))

    (total_traffic_np, per_wave_traffic_np,
     per_wave_send_offsets_np, per_wave_recv_offsets_np,
     base_send_np, base_recv_np) = (
        _w4_p1espg_compute_traffic_and_offsets(world_size, max_waves)
    )

    # Sanity checks.
    assert (per_wave_traffic_np > 0).all(), (
        "v66r per-wave entries must all be positive (rotation BASE > 0)."
    )
    assert (per_wave_traffic_np % _W4_P1ESPG_TILE == 0).all(), (
        "v66r per-wave entries must be TILE-aligned."
    )
    assert np.array_equal(
        per_wave_traffic_np[0], per_wave_traffic_np[1]
    ), "v66r per-wave matrices MUST be identical (the workaround invariant)."
    # Per-wave offsets MUST drift across waves for at least one (s, t).
    assert not np.array_equal(
        per_wave_send_offsets_np[0], per_wave_send_offsets_np[1]
    ), "v66r send offsets MUST drift across waves (production-shape gate)."
    assert not np.array_equal(
        per_wave_recv_offsets_np[0], per_wave_recv_offsets_np[1]
    ), "v66r recv offsets MUST drift across waves (production-shape gate)."
    # Per-rank totals must fit in their respective buffers.
    for r in range(world_size):
        send_total = int(total_traffic_np[r, :].sum())
        recv_total = int(total_traffic_np[:, r].sum())
        assert send_total <= _W4_P1ESPG_MAX_PADDED, (
            f"_W4_P1ESPG_MAX_PADDED={_W4_P1ESPG_MAX_PADDED} too small for "
            f"rank={r} send_total={send_total}"
        )
        assert recv_total <= recv_capacity_total, (
            f"recv_capacity_total={recv_capacity_total} too small for "
            f"rank={r} recv_total={recv_total}"
        )

    # Per-rank sorted_row, written with wave/src/dst sentinels at the
    # per-wave send offsets (which drift across waves inside each (s, t)
    # chunk).
    sorted_row_padded_np = np.zeros(
        (world_size, _W4_P1ESPG_MAX_PADDED), dtype=np.int32,
    )
    for r in range(world_size):
        for w in range(max_waves):
            for p in range(world_size):
                start = int(per_wave_send_offsets_np[w, r, p])
                cnt = int(per_wave_traffic_np[w, r, p])
                sorted_row_padded_np[r, start:start + cnt] = (
                    _w4_p1espg_sentinel(w, r, p)
                )

    HBM = pltpu.MemorySpace.HBM
    SMEM = pltpu.MemorySpace.SMEM

    kernel = _make_w4_p1espg_kernel(
        world_size=world_size, max_waves=max_waves,
        recv_capacity_total=recv_capacity_total,
    )

    P = jax.sharding.PartitionSpec

    def per_device(src, traffic, send_off, recv_off):
        recv = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(
                (recv_capacity_total,), jnp.int32,
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[
                    pl.BlockSpec(memory_space=HBM),   # sorted_row
                    pl.BlockSpec(memory_space=SMEM),  # traffic
                    pl.BlockSpec(memory_space=SMEM),  # send_off
                    pl.BlockSpec(memory_space=SMEM),  # recv_off
                ],
                out_specs=pl.BlockSpec(memory_space=HBM),
                scratch_shapes=(
                    pltpu.VMEM((recv_capacity_total,), jnp.int32),
                    pltpu.SemaphoreType.REGULAR((max_waves, 2)),
                    pltpu.SemaphoreType.DMA,
                    pltpu.SemaphoreType.DMA,
                ),
            ),
            interpret=False,
        )(src[0], traffic, send_off, recv_off)
        return recv[None, ...]

    sharded = jax.shard_map(
        per_device, mesh=mesh,
        in_specs=(P("x"), P(), P(), P()),
        out_specs=P("x"),
        check_vma=False,
    )

    sharding_x = jax.sharding.NamedSharding(mesh, P("x"))
    sharding_rep = jax.sharding.NamedSharding(mesh, P())
    src = jax.device_put(jnp.array(sorted_row_padded_np), sharding_x)
    traffic = jax.device_put(jnp.array(per_wave_traffic_np), sharding_rep)
    send_off = jax.device_put(
        jnp.array(per_wave_send_offsets_np), sharding_rep,
    )
    recv_off = jax.device_put(
        jnp.array(per_wave_recv_offsets_np), sharding_rep,
    )

    out = sharded(src, traffic, send_off, recv_off)
    return (
        np.array(out),
        total_traffic_np,
        per_wave_traffic_np,
        per_wave_send_offsets_np,
        per_wave_recv_offsets_np,
        base_send_np,
        base_recv_np,
        max_waves,
        recv_capacity_total,
        world_size,
    )


@unittest.skipUnless(_has_tpu(), "Litmus test requires real TPU")
class W4Phase1EvenSplitProdGeomLitmus(unittest.TestCase):
    """v66r: WO#4 with production-shape offset drift.

    Per-wave per-pair counts are identical (the WO#4 invariant), but per-wave
    offsets drift across waves the way production `_split_traffic_into_waves`
    computes them (cumsum across waves added to base offsets). Predicted PASS
    if WO#4 is genuinely production-viable; HANG if counts-identical is not
    sufficient when offsets drift.
    """

    def setUp(self) -> None:
        if jax.device_count() < 4:
            self.skipTest(
                f"Need 4 devices, have {jax.device_count()}"
            )

    def test_w4_p1_even_split_prod_geom(self) -> None:
        """v66r: identical per-wave counts + production cumsum offset drift."""
        import threading

        result_box: list = []
        error_box: list = []

        def runner():
            try:
                result_box.append(_run_w4_p1espg_litmus())
            except BaseException as e:  # noqa: BLE001
                error_box.append(e)

        t = threading.Thread(target=runner, daemon=True)
        t.start()
        t.join(timeout=300.0)
        if t.is_alive():
            self.fail(
                "v66r (W=4 phase-1 per-wave EVEN-SPLIT with PRODUCTION "
                "OFFSET DRIFT) HUNG after 300s; identical per-wave counts "
                "are NOT sufficient when offsets drift across waves the way "
                "production `_split_traffic_into_waves` computes them. "
                "Fingerprint reduces to per-pair offset drift across waves."
            )
        if error_box:
            raise error_box[0]

        (recv, total_np, traffic_np, send_off_np, recv_off_np,
         base_send_np, base_recv_np,
         max_waves, recv_capacity_total, world_size) = result_box[0]

        # Diagnostic header.
        print(
            f"[LITMUS_W4_P1ESPG] max_waves={max_waves} "
            f"recv_capacity_total={recv_capacity_total} "
            f"world_size={world_size} "
            f"base_padded={_W4_P1ESPG_BASE_PADDED} "
            f"max_padded={_W4_P1ESPG_MAX_PADDED}",
            flush=True,
        )
        print("[LITMUS_W4_P1ESPG] total_traffic[r, p]:", flush=True)
        for r in range(world_size):
            print(f"  row r={r}: {total_np[r].tolist()}", flush=True)
        for w in range(max_waves):
            print(
                f"[LITMUS_W4_P1ESPG] per_wave_traffic[w={w}, s, t]:",
                flush=True,
            )
            for s in range(world_size):
                print(
                    f"  row s={s}: {traffic_np[w, s].tolist()}",
                    flush=True,
                )
        for w in range(max_waves):
            print(
                f"[LITMUS_W4_P1ESPG] per_wave_send_offsets[w={w}, s, t]:",
                flush=True,
            )
            for s in range(world_size):
                print(
                    f"  row s={s}: {send_off_np[w, s].tolist()}",
                    flush=True,
                )
        for w in range(max_waves):
            print(
                f"[LITMUS_W4_P1ESPG] per_wave_recv_offsets[w={w}, t, s]:",
                flush=True,
            )
            for t in range(world_size):
                print(
                    f"  row t={t}: {recv_off_np[w, t].tolist()}",
                    flush=True,
                )

        per_wave_identical = bool(
            np.array_equal(traffic_np[0], traffic_np[1])
        )
        send_offsets_drift = bool(
            not np.array_equal(send_off_np[0], send_off_np[1])
        )
        recv_offsets_drift = bool(
            not np.array_equal(recv_off_np[0], recv_off_np[1])
        )
        print(
            f"[LITMUS_W4_P1ESPG] per_wave_traffic_identical="
            f"{per_wave_identical}",
            flush=True,
        )
        print(
            f"[LITMUS_W4_P1ESPG] per_wave_send_offsets_drift_across_waves="
            f"{send_offsets_drift}",
            flush=True,
        )
        print(
            f"[LITMUS_W4_P1ESPG] per_wave_recv_offsets_drift_across_waves="
            f"{recv_offsets_drift}",
            flush=True,
        )
        # Inspect first 4 ints of each (wave, src, dst) recv slot.
        for r in range(world_size):
            for w in range(max_waves):
                for s in range(world_size):
                    cnt = int(traffic_np[w, s, r])
                    off = int(recv_off_np[w, r, s])
                    first4 = recv[r, off:off + 4].tolist()
                    print(
                        f"[LITMUS_W4_P1ESPG] rank={r} wave={w} src={s} "
                        f"cnt={cnt} off={off} first4="
                        f"{[hex(int(x)) for x in first4]}",
                        flush=True,
                    )

        mismatches = []
        for r in range(world_size):
            for w in range(max_waves):
                for s in range(world_size):
                    cnt = int(traffic_np[w, s, r])
                    off = int(recv_off_np[w, r, s])
                    expected = _w4_p1espg_sentinel(w, s, r)
                    slot = recv[r, off:off + cnt]
                    if not np.all(slot == expected):
                        actual_first8 = slot[:8].tolist()
                        mismatches.append(
                            f"rank={r} wave={w} src={s} cnt={cnt} off={off} "
                            f"expected={hex(expected)} actual_first8="
                            f"{[hex(int(x)) for x in actual_first8]}"
                        )
        if mismatches:
            self.fail(
                "v66r phase-1 even-split prod-geom DATA MISMATCH:\n  "
                + "\n  ".join(mismatches)
            )


# ---------------------------------------------------------------------------
# v67a: RING-SHIFT NAIVE (per-wave-different cnt)
#
# Topology change vs WO#4: instead of every wave doing a full a2a (W^2 DMAs
# across the mesh), each wave has every rank exchange with EXACTLY ONE peer.
# Wave w: rank r exchanges with peer (r + w) % W. After W waves, every
# (src, dst) pair has been exchanged exactly once.
#
# Per-wave per-rank: ONE DMA. Per-wave count = max over the wave's diagonal
# of traffic[r, (r+w)%W]. With v66r-style skewed BASE_PADDED rotation
# (256, 256, 512, 512), per-wave naive cnt = [256, 256, 512, 512] -- DIFFERENT
# across waves. Total in-kernel waves = W = 4.
#
# Hypothesis test: v66p HANG fingerprint says per-wave-different counts
# trigger the deadlock when the in-kernel fori_loop has multi-iter inner DMA
# loop. v67a's inner DMA loop is LENGTH-1 (one peer per wave). If v67a
# ALSO hangs, the fingerprint generalizes to any multi-iter outer loop with
# per-wave-different counts. If v67a PASSES, the firmware matcher confusion
# specifically requires the per-wave inner DMA loop to be multi-iter.
#
# Stop conditions:
#   PASS  -> Ring-shift naive sidesteps the hang (length-1 inner loop is the
#            structural escape). Per-wave-different counts alone insufficient
#            -- multi-DMA inner loop is also load-bearing.
#   HANG  -> Per-wave-different counts trigger deadlock even with length-1
#            inner DMA loop. Fingerprint generalizes; ring-shift not viable
#            without uniform counts (must use v67b).
# ---------------------------------------------------------------------------


_W4_P1RSN_TILE = 128
_W4_P1RSN_WORLD_SIZE = 4
# Same skewed rotation as v66r so naive per-wave cnt is non-uniform.
_W4_P1RSN_BASE_PADDED = (256, 256, 512, 512)
_W4_P1RSN_MAX_WAVES = 4  # ring-shift uses W waves


def _w4_p1rsn_sentinel(w: int, src: int, dst: int) -> int:
    """Wave/source/dest-distinct sentinel (v67a prefix 0x21)."""
    return 0x21000000 + (w << 16) + (src << 8) + dst


def _w4_p1rsn_compute():
    """Compute ring-shift naive geometry.

    Returns:
        total_traffic_np    [W, W] int32 -- BASE_padded[(p-r) % W]
        cnt_per_wave_np     [W] int32 -- naive max over wave's diagonal
        send_off_np         [W] int32 -- per-wave send offset in sender's
                                          send buffer (cumulative cnt_per_wave)
        recv_off_np         [W] int32 -- per-wave recv offset in receiver's
                                          recv buffer (also cumulative)
        send_buf_size       int -- W * max(cnt_per_wave)
        recv_buf_size       int -- same as send (single buffer per rank)
    """
    W = _W4_P1RSN_WORLD_SIZE
    base = _W4_P1RSN_BASE_PADDED
    total_traffic_np = np.zeros((W, W), dtype=np.int32)
    for r in range(W):
        for p in range(W):
            total_traffic_np[r, p] = base[(p - r) % W]
    # Wave w's diagonal: traffic[r, (r+w)%W] for r in range(W).
    # Naive cnt[w] = max over r. With cyclic rotation, all rows of the
    # diagonal are equal, so cnt[w] = base[w].
    cnt_per_wave_np = np.array(
        [int(np.max([total_traffic_np[r, (r + w) % W] for r in range(W)]))
         for w in range(W)],
        dtype=np.int32,
    )
    # Per-wave send/recv offsets: cumulative cnt_per_wave.
    send_off_np = np.zeros(W, dtype=np.int32)
    cum = 0
    for w in range(W):
        send_off_np[w] = cum
        cum += int(cnt_per_wave_np[w])
    recv_off_np = send_off_np.copy()  # same layout: per-wave slot in recv buf
    send_buf_size = int(cum)
    recv_buf_size = send_buf_size
    return (
        total_traffic_np, cnt_per_wave_np, send_off_np, recv_off_np,
        send_buf_size, recv_buf_size,
    )


def _make_w4_p1rsn_kernel(world_size: int, max_waves: int, buf_size: int):
    """Ring-shift naive kernel.

    Per wave w (Python-unrolled): peer = (my_rank + w) % W (traced).
    cnt and offsets read from SMEM at compile-time index w.
    """

    def kernel(
        sorted_row_ref,    # [buf_size] int32 HBM
        cnt_ref,           # [W] int32 SMEM (per-wave count)
        send_off_ref,      # [W] int32 SMEM
        recv_off_ref,      # [W] int32 SMEM
        out_recv_ref,      # [buf_size] int32 HBM
        zero_int_vmem,     # [buf_size] int32 VMEM
        barrier_sems,      # REGULAR (max_waves, 2)
        send_sem,          # DMA send sem
        recv_sem,          # DMA recv sem
    ):
        my_rank = lax.axis_index("x")
        zero_int_vmem[...] = jnp.zeros_like(zero_int_vmem)
        pltpu.sync_copy(zero_int_vmem, out_recv_ref)

        for w in range(max_waves):
            pre_sem = barrier_sems.at[w, 0]
            for peer in range(world_size):
                pl.semaphore_signal(
                    pre_sem, 1,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
            pl.semaphore_wait(pre_sem, world_size)

            send_count = pl.multiple_of(cnt_ref[w], _W4_P1RSN_TILE)
            send_start = pl.multiple_of(send_off_ref[w], _W4_P1RSN_TILE)
            remote_recv_start = pl.multiple_of(
                recv_off_ref[w], _W4_P1RSN_TILE
            )
            peer = (my_rank + w) % world_size

            d = pltpu.make_async_remote_copy(
                src_ref=sorted_row_ref.at[pl.ds(send_start, send_count)],
                dst_ref=out_recv_ref.at[
                    pl.ds(remote_recv_start, send_count)
                ],
                send_sem=send_sem, recv_sem=recv_sem,
                device_id=(peer,),
                device_id_type=pltpu.DeviceIdType.MESH,
            )
            d.start()
            d.wait()

            post_sem = barrier_sems.at[w, 1]
            for peer in range(world_size):
                pl.semaphore_signal(
                    post_sem, 1,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
            pl.semaphore_wait(post_sem, world_size)

    return kernel


def _run_w4_p1rsn_litmus():
    W = _W4_P1RSN_WORLD_SIZE
    max_waves = _W4_P1RSN_MAX_WAVES
    devices = jax.devices()[:W]
    mesh = jax.sharding.Mesh(np.array(devices), ("x",))

    (total_traffic_np, cnt_per_wave_np, send_off_np, recv_off_np,
     send_buf_size, recv_buf_size) = _w4_p1rsn_compute()

    # Sanity: counts must be TILE-aligned and DIFFERENT across waves
    # (otherwise this isn't testing the naive vs global-pad distinction).
    assert (cnt_per_wave_np % _W4_P1RSN_TILE == 0).all(), (
        "v67a per-wave counts must be TILE-aligned."
    )
    assert len(set(cnt_per_wave_np.tolist())) > 1, (
        "v67a per-wave counts MUST differ across waves "
        f"(got {cnt_per_wave_np.tolist()})"
    )

    # Build per-rank send buffer with sentinels at per-wave offsets.
    sorted_row_padded_np = np.zeros((W, send_buf_size), dtype=np.int32)
    for r in range(W):
        for w in range(max_waves):
            start = int(send_off_np[w])
            cnt = int(cnt_per_wave_np[w])
            peer = (r + w) % W
            sorted_row_padded_np[r, start:start + cnt] = (
                _w4_p1rsn_sentinel(w, r, peer)
            )

    HBM = pltpu.MemorySpace.HBM
    SMEM = pltpu.MemorySpace.SMEM

    kernel = _make_w4_p1rsn_kernel(
        world_size=W, max_waves=max_waves, buf_size=recv_buf_size,
    )

    P = jax.sharding.PartitionSpec

    def per_device(src, cnt, send_off, recv_off):
        recv = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((recv_buf_size,), jnp.int32),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[
                    pl.BlockSpec(memory_space=HBM),   # sorted_row
                    pl.BlockSpec(memory_space=SMEM),  # cnt
                    pl.BlockSpec(memory_space=SMEM),  # send_off
                    pl.BlockSpec(memory_space=SMEM),  # recv_off
                ],
                out_specs=pl.BlockSpec(memory_space=HBM),
                scratch_shapes=(
                    pltpu.VMEM((recv_buf_size,), jnp.int32),
                    pltpu.SemaphoreType.REGULAR((max_waves, 2)),
                    pltpu.SemaphoreType.DMA,
                    pltpu.SemaphoreType.DMA,
                ),
            ),
            interpret=False,
        )(src[0], cnt, send_off, recv_off)
        return recv[None, ...]

    sharded = jax.shard_map(
        per_device, mesh=mesh,
        in_specs=(P("x"), P(), P(), P()),
        out_specs=P("x"),
        check_vma=False,
    )
    sharding_x = jax.sharding.NamedSharding(mesh, P("x"))
    sharding_rep = jax.sharding.NamedSharding(mesh, P())
    src = jax.device_put(jnp.array(sorted_row_padded_np), sharding_x)
    cnt = jax.device_put(jnp.array(cnt_per_wave_np), sharding_rep)
    send_off = jax.device_put(jnp.array(send_off_np), sharding_rep)
    recv_off = jax.device_put(jnp.array(recv_off_np), sharding_rep)
    out = sharded(src, cnt, send_off, recv_off)
    return (
        np.array(out), total_traffic_np, cnt_per_wave_np,
        send_off_np, recv_off_np, max_waves, W,
    )


@unittest.skipUnless(_has_tpu(), "Litmus test requires real TPU")
class W4Phase1RingShiftNaiveLitmus(unittest.TestCase):
    """v67a: ring-shift with per-wave-different counts.

    Tests whether the per-wave-different-cnt fingerprint generalizes to
    LENGTH-1 inner DMA loops. PASS -> firmware matcher confusion needs
    multi-iter inner loops. HANG -> any multi-iter outer with cnt drift hangs.
    """

    def setUp(self) -> None:
        if jax.device_count() < 4:
            self.skipTest(f"Need 4 devices, have {jax.device_count()}")

    def test_w4_p1_ringshift_naive(self) -> None:
        import threading
        result_box: list = []
        error_box: list = []

        def runner():
            try:
                result_box.append(_run_w4_p1rsn_litmus())
            except BaseException as e:  # noqa: BLE001
                error_box.append(e)

        t = threading.Thread(target=runner, daemon=True)
        t.start()
        t.join(timeout=300.0)
        if t.is_alive():
            self.fail(
                "v67a (W=4 phase-1 ring-shift NAIVE per-wave-different cnt) "
                "HUNG after 300s; per-wave-different counts trigger deadlock "
                "even with LENGTH-1 inner DMA loop. Fingerprint generalizes."
            )
        if error_box:
            raise error_box[0]

        (recv, total_np, cnt_np, send_off_np, recv_off_np,
         max_waves, world_size) = result_box[0]

        print(
            f"[LITMUS_W4_P1RSN] max_waves={max_waves} world_size={world_size} "
            f"base_padded={_W4_P1RSN_BASE_PADDED} "
            f"cnt_per_wave={cnt_np.tolist()} "
            f"send_off={send_off_np.tolist()} "
            f"recv_off={recv_off_np.tolist()}",
            flush=True,
        )
        print("[LITMUS_W4_P1RSN] total_traffic[r, p]:", flush=True)
        for r in range(world_size):
            print(f"  row r={r}: {total_np[r].tolist()}", flush=True)

        # Verify: rank r at wave w received from src=(r-w+W)%W at slot w.
        mismatches = []
        for r in range(world_size):
            for w in range(max_waves):
                src = (r - w + world_size) % world_size
                cnt = int(cnt_np[w])
                off = int(recv_off_np[w])
                expected = _w4_p1rsn_sentinel(w, src, r)
                slot = recv[r, off:off + cnt]
                first4 = recv[r, off:off + 4].tolist()
                print(
                    f"[LITMUS_W4_P1RSN] rank={r} wave={w} src={src} "
                    f"cnt={cnt} off={off} expected={hex(expected)} "
                    f"first4={[hex(int(x)) for x in first4]}",
                    flush=True,
                )
                if not np.all(slot == expected):
                    mismatches.append(
                        f"rank={r} wave={w} src={src} cnt={cnt} off={off} "
                        f"expected={hex(expected)} actual_first8="
                        f"{[hex(int(x)) for x in slot[:8].tolist()]}"
                    )
        if mismatches:
            self.fail(
                "v67a phase-1 ring-shift naive DATA MISMATCH:\n  "
                + "\n  ".join(mismatches)
            )


# ---------------------------------------------------------------------------
# v67b: RING-SHIFT GLOBAL-PAD (per-wave-uniform cnt = global max)
#
# Same topology as v67a (one DMA per rank per wave, peer = (r+w) % W),
# but cnt is CONSTANT across waves = max over ALL (src, dst) of traffic.
# Per-wave SMEM identical -> should PASS trivially per v66p fingerprint.
#
# Validates the safe baseline + provides perf comparison vs v67a (if v67a
# passes) and vs WO#4 (in benchmark).
#
# Stop conditions:
#   PASS  -> Expected. Confirms ring-shift topology works when uniform.
#   HANG  -> Surprising; would mean topology-level issue independent of cnt.
# ---------------------------------------------------------------------------


_W4_P1RSG_TILE = 128
_W4_P1RSG_WORLD_SIZE = 4
_W4_P1RSG_BASE_PADDED = (256, 256, 512, 512)
_W4_P1RSG_MAX_WAVES = 4


def _w4_p1rsg_sentinel(w: int, src: int, dst: int) -> int:
    return 0x22000000 + (w << 16) + (src << 8) + dst


def _w4_p1rsg_compute():
    """Ring-shift global-pad: cnt = constant global max."""
    W = _W4_P1RSG_WORLD_SIZE
    base = _W4_P1RSG_BASE_PADDED
    total_traffic_np = np.zeros((W, W), dtype=np.int32)
    for r in range(W):
        for p in range(W):
            total_traffic_np[r, p] = base[(p - r) % W]
    cnt_const = int(total_traffic_np.max())
    cnt_per_wave_np = np.full(W, cnt_const, dtype=np.int32)
    send_off_np = np.array([w * cnt_const for w in range(W)], dtype=np.int32)
    recv_off_np = send_off_np.copy()
    buf_size = W * cnt_const
    return (
        total_traffic_np, cnt_per_wave_np, send_off_np, recv_off_np,
        buf_size, buf_size,
    )


# Reuse v67a's kernel maker (identical structure, only inputs differ).
_make_w4_p1rsg_kernel = _make_w4_p1rsn_kernel


def _run_w4_p1rsg_litmus():
    W = _W4_P1RSG_WORLD_SIZE
    max_waves = _W4_P1RSG_MAX_WAVES
    devices = jax.devices()[:W]
    mesh = jax.sharding.Mesh(np.array(devices), ("x",))

    (total_traffic_np, cnt_per_wave_np, send_off_np, recv_off_np,
     send_buf_size, recv_buf_size) = _w4_p1rsg_compute()

    assert (cnt_per_wave_np % _W4_P1RSG_TILE == 0).all()
    assert len(set(cnt_per_wave_np.tolist())) == 1, (
        "v67b counts MUST be uniform across waves "
        f"(got {cnt_per_wave_np.tolist()})"
    )

    sorted_row_padded_np = np.zeros((W, send_buf_size), dtype=np.int32)
    for r in range(W):
        for w in range(max_waves):
            start = int(send_off_np[w])
            cnt = int(cnt_per_wave_np[w])
            peer = (r + w) % W
            sorted_row_padded_np[r, start:start + cnt] = (
                _w4_p1rsg_sentinel(w, r, peer)
            )

    HBM = pltpu.MemorySpace.HBM
    SMEM = pltpu.MemorySpace.SMEM

    kernel = _make_w4_p1rsg_kernel(
        world_size=W, max_waves=max_waves, buf_size=recv_buf_size,
    )

    P = jax.sharding.PartitionSpec

    def per_device(src, cnt, send_off, recv_off):
        recv = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((recv_buf_size,), jnp.int32),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[
                    pl.BlockSpec(memory_space=HBM),
                    pl.BlockSpec(memory_space=SMEM),
                    pl.BlockSpec(memory_space=SMEM),
                    pl.BlockSpec(memory_space=SMEM),
                ],
                out_specs=pl.BlockSpec(memory_space=HBM),
                scratch_shapes=(
                    pltpu.VMEM((recv_buf_size,), jnp.int32),
                    pltpu.SemaphoreType.REGULAR((max_waves, 2)),
                    pltpu.SemaphoreType.DMA,
                    pltpu.SemaphoreType.DMA,
                ),
            ),
            interpret=False,
        )(src[0], cnt, send_off, recv_off)
        return recv[None, ...]

    sharded = jax.shard_map(
        per_device, mesh=mesh,
        in_specs=(P("x"), P(), P(), P()),
        out_specs=P("x"),
        check_vma=False,
    )
    sharding_x = jax.sharding.NamedSharding(mesh, P("x"))
    sharding_rep = jax.sharding.NamedSharding(mesh, P())
    src = jax.device_put(jnp.array(sorted_row_padded_np), sharding_x)
    cnt = jax.device_put(jnp.array(cnt_per_wave_np), sharding_rep)
    send_off = jax.device_put(jnp.array(send_off_np), sharding_rep)
    recv_off = jax.device_put(jnp.array(recv_off_np), sharding_rep)
    out = sharded(src, cnt, send_off, recv_off)
    return (
        np.array(out), total_traffic_np, cnt_per_wave_np,
        send_off_np, recv_off_np, max_waves, W,
    )


@unittest.skipUnless(_has_tpu(), "Litmus test requires real TPU")
class W4Phase1RingShiftGlobalPadLitmus(unittest.TestCase):
    """v67b: ring-shift with per-wave-uniform counts (global max)."""

    def setUp(self) -> None:
        if jax.device_count() < 4:
            self.skipTest(f"Need 4 devices, have {jax.device_count()}")

    def test_w4_p1_ringshift_globalpad(self) -> None:
        import threading
        result_box: list = []
        error_box: list = []

        def runner():
            try:
                result_box.append(_run_w4_p1rsg_litmus())
            except BaseException as e:  # noqa: BLE001
                error_box.append(e)

        t = threading.Thread(target=runner, daemon=True)
        t.start()
        t.join(timeout=300.0)
        if t.is_alive():
            self.fail(
                "v67b (W=4 phase-1 ring-shift GLOBAL-PAD per-wave-uniform "
                "cnt) HUNG after 300s; SURPRISING -- ring-shift topology "
                "itself may have an issue independent of count drift."
            )
        if error_box:
            raise error_box[0]

        (recv, total_np, cnt_np, send_off_np, recv_off_np,
         max_waves, world_size) = result_box[0]

        print(
            f"[LITMUS_W4_P1RSG] max_waves={max_waves} world_size={world_size} "
            f"base_padded={_W4_P1RSG_BASE_PADDED} "
            f"cnt_per_wave={cnt_np.tolist()} "
            f"send_off={send_off_np.tolist()} "
            f"recv_off={recv_off_np.tolist()}",
            flush=True,
        )
        print("[LITMUS_W4_P1RSG] total_traffic[r, p]:", flush=True)
        for r in range(world_size):
            print(f"  row r={r}: {total_np[r].tolist()}", flush=True)

        mismatches = []
        for r in range(world_size):
            for w in range(max_waves):
                src = (r - w + world_size) % world_size
                cnt = int(cnt_np[w])
                off = int(recv_off_np[w])
                expected = _w4_p1rsg_sentinel(w, src, r)
                slot = recv[r, off:off + cnt]
                first4 = recv[r, off:off + 4].tolist()
                print(
                    f"[LITMUS_W4_P1RSG] rank={r} wave={w} src={src} "
                    f"cnt={cnt} off={off} expected={hex(expected)} "
                    f"first4={[hex(int(x)) for x in first4]}",
                    flush=True,
                )
                if not np.all(slot == expected):
                    mismatches.append(
                        f"rank={r} wave={w} src={src} cnt={cnt} off={off} "
                        f"expected={hex(expected)} actual_first8="
                        f"{[hex(int(x)) for x in slot[:8].tolist()]}"
                    )
        if mismatches:
            self.fail(
                "v67b phase-1 ring-shift global-pad DATA MISMATCH:\n  "
                + "\n  ".join(mismatches)
            )


if __name__ == "__main__":
    unittest.main()
