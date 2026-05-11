# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Fused jagged all-to-all + sharded embedding lookup as Pallas TC + SC kernels.

This module implements a TPU-friendly version of the sharded embedding lookup
pattern used in torchrec. The challenge that motivates this kernel:

  * Each rank holds a row-wise shard of an embedding table (rank ``r`` owns
    rows ``i`` such that ``i % world_size == r``).
  * Each rank has a *padded* batch of ``[max_local_ids]`` query IDs whose
    destinations are data-dependent.
  * The natural "torchrec-on-TPU" path — express the all-to-all as
    ``jax.lax.ragged_all_to_all`` plus ``jnp.take`` plus another
    ``ragged_all_to_all`` — forces the receive buffer between the two
    collectives to be statically sized to the global worst-case fan-in
    (``world_size * max_local_ids``). For sparse features that ratio is
    typically 1:100 to 1:1000, so the wasted HBM and bandwidth are
    prohibitive.

Design summary (matches the architecture discussed in the design doc):

  Step 1 (host JAX):  bucketize local_ids by destination rank, sort, and
                      ``lax.all_gather`` the per-rank send-counts into a
                      static-shape ``[world_size, world_size]`` traffic matrix.
                      Then split the traffic matrix into ``[max_waves, W, W]``
                      per-wave chunks so each wave's per-receiver fan-in fits
                      in ``recv_capacity`` (chunked DMA overflow waves).
  Step 2 (Pallas):    *three* kernels chained inside one ``shard_map``,
                      wrapped in a ``lax.fori_loop`` over waves:
                        (a) TC kernel: per-pair ``pltpu.make_async_remote_copy``
                            of int32 row IDs.
                        (b) SC kernel: SparseCore indirect-DMA gather from the
                            local table shard into a padded HBM buffer using
                            ``pltpu.sync_copy(table.at[idx_vmem], scratch)``.
                        (c) TC kernel: per-pair ``pltpu.make_async_remote_copy``
                            of looked-up float embeddings back to the requester.
                      Per-pair extents are runtime SMEM reads — XLA never sees
                      them. The wave loop reuses the ``recv_capacity``-sized
                      recv buffer and the SC scratch across waves; the final
                      embedding output buffer is alias-accumulated across waves.
  Step 3 (host JAX):  un-permute the kernel output via the saved
                      sort permutation.

The only XLA-visible buffers are ``[max_local_ids]`` (input) and
``[max_local_ids, D]`` (output) — both bounded by *local* batch size, never
by the global fan-in. The ``recv_capacity`` argument is now a *wave size*
(P99 / safety bound) rather than a worst-case bound: smaller capacity means
more waves, larger means fewer waves, single-wave is still attainable when
``recv_capacity >= max_local_ids + W * TILE``.

The previous v0 had a single TC kernel that gathered rows from VMEM via
``table_vmem[pl.ds(idx, 1), :]``. That worked for f32 (VMEM dim-0 tile = 1)
but not bf16 (tile = 16). The SC indirect DMA bypasses the VMEM tile
constraint entirely, since the gather is HBM→VMEM with lane-granular offsets.

Caveat on testing: ``interpret=True`` lowers Pallas through HLO discharge,
which materializes ``pl.ds(start, length)`` as ``lax.dynamic_slice`` and
requires ``length`` to be statically known. Runtime-extent slices like
``ref.at[pl.ds(start, runtime_count)]`` only work via the Mosaic backend on
real TPU. So the end-to-end kernel test must run on real TPU (the host-side
prep helpers ``_bucketize_and_sort`` / ``_derive_offsets`` are testable on
CPU separately).
"""

from __future__ import annotations

import functools
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# ``tpu_sc`` is only available in JAX builds with SparseCore support
# (≥ 0.10 on the v7x runner image). Make it optional so this module still
# imports under the older host-side JAX (used for CPU-only host-prep tests).
try:
    from jax.experimental.pallas import tpu_sc as plsc  # type: ignore[import-not-found]
except ImportError:
    plsc = None  # type: ignore[assignment]


# Number of distinct ``collective_id`` values cycled across per-wave
# pallas_call sites in ``_jagged_a2a_embedding_lookup_local_f32_alltc_ringshift``.
# At W=128, using a unique id per wave triggers 128 distinct compiles
# (cold-compile > 40 min). Cycling through K ids reduces to K compiles
# (~minutes); cross-rank sync between waves is provided by the per-wave
# ``psum + optimization_barrier`` carry, so channel reuse between waves
# separated by K is safe.
_COLLECTIVE_ID_CYCLE = 8


# ---------------------------------------------------------------------------
# A. Host-side input preparation (pure JAX, all static shapes)
# ---------------------------------------------------------------------------


def _bucketize_and_sort(
    local_ids: jax.Array,
    local_id_count: jax.Array,
    max_local_ids: int,
    world_size: int,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Bucket local IDs by destination rank (= ``id % world_size``) and sort.

    Padding past ``local_id_count`` is routed to a virtual "drop" bucket so it
    never appears in any per-rank send count. We rely on stable argsort to keep
    the relative order of valid IDs deterministic.

    Returns:
        sorted_local_row: ``[max_local_ids]`` int32 — the per-shard row index
            (``id // world_size``) sorted by destination rank, padded with 0s
            for invalid slots.
        send_counts: ``[world_size]`` int32 — count of valid IDs per
            destination rank.
        sort_perm: ``[max_local_ids]`` int32 — permutation that turns the input
            order into the sorted order.
        reverse_perm: ``[max_local_ids]`` int32 — inverse of ``sort_perm``,
            used to scatter outputs back to caller order.
    """
    position = jnp.arange(max_local_ids, dtype=jnp.int32)
    is_valid = position < local_id_count

    raw_dest = local_ids % world_size
    # Use ``world_size`` as a sentinel "drop" bucket for padding so that
    # bincount with length=world_size+1 keeps invalid IDs out of send_counts.
    dest_rank = jnp.where(is_valid, raw_dest, world_size).astype(jnp.int32)

    # Stable sort so equal-destination IDs preserve their original order.
    sort_perm = jnp.argsort(dest_rank, stable=True).astype(jnp.int32)
    reverse_perm = jnp.argsort(sort_perm, stable=True).astype(jnp.int32)

    sorted_ids = local_ids[sort_perm]
    sorted_local_row = (sorted_ids // world_size).astype(jnp.int32)

    bins = jnp.bincount(dest_rank, length=world_size + 1)
    send_counts = bins[:world_size].astype(jnp.int32)

    return sorted_local_row, send_counts, sort_perm, reverse_perm


def _round_up_tile(x: jax.Array, tile: int) -> jax.Array:
    """Round each element of ``x`` UP to the next multiple of ``tile``."""
    return ((x + tile - 1) // tile * tile).astype(jnp.int32)


def _pad_per_peer(
    sorted_row: jax.Array,
    send_counts_real: jax.Array,
    send_counts_padded: jax.Array,
    max_local_ids: int,
    max_padded: int,
    world_size: int,
) -> jax.Array:
    """Insert sentinel rows (value 0) so each per-peer slice is tile-aligned.

    The kernel does DMA on ``[padded_offset[p], padded_count[p])`` slices; the
    receiver discards the trailing ``padded - real`` rows by indexing only the
    first ``real_count[p]`` entries on the un-permute path. Sentinel value 0
    is harmless because looking up local row 0 is always defined.
    """
    real_offsets = jnp.cumsum(send_counts_real) - send_counts_real
    padded_offsets = jnp.cumsum(send_counts_padded) - send_counts_padded
    pos = jnp.arange(max_padded, dtype=jnp.int32)
    peer_id = jnp.clip(
        jnp.searchsorted(padded_offsets, pos, side="right") - 1,
        0,
        world_size - 1,
    )
    local_pad_pos = pos - padded_offsets[peer_id]
    is_real = local_pad_pos < send_counts_real[peer_id]
    src_pos = jnp.clip(real_offsets[peer_id] + local_pad_pos, 0, max_local_ids - 1)
    return jnp.where(is_real, sorted_row[src_pos], 0).astype(jnp.int32)


def _unpad_per_peer(
    out_padded: jax.Array,
    send_counts_real: jax.Array,
    send_counts_padded: jax.Array,
    max_local_ids: int,
    max_padded: int,
    world_size: int,
) -> jax.Array:
    """Inverse of ``_pad_per_peer`` for the embedding output buffer.

    Returns ``[max_local_ids, D]`` by gathering the first ``real_count[p]`` rows
    of each peer's padded slice. Trailing slots (positions past the total real
    count) are filled with zeros.
    """
    real_offsets = jnp.cumsum(send_counts_real) - send_counts_real
    padded_offsets = jnp.cumsum(send_counts_padded) - send_counts_padded
    total_real = jnp.sum(send_counts_real)
    pos = jnp.arange(max_local_ids, dtype=jnp.int32)
    in_real = pos < total_real
    peer_id = jnp.clip(
        jnp.searchsorted(real_offsets, pos, side="right") - 1,
        0,
        world_size - 1,
    )
    local_real_pos = pos - real_offsets[peer_id]
    padded_src = jnp.clip(
        padded_offsets[peer_id] + local_real_pos, 0, max_padded - 1
    )
    gathered = out_padded[padded_src]
    return jnp.where(in_real[:, None], gathered, 0).astype(out_padded.dtype)


def _max_waves_for_capacity(
    max_local_ids: int, world_size: int, recv_capacity: int
) -> int:
    """Static upper bound on the wave count for a given ``recv_capacity``.

    The per-receiver padded fan-in is bounded by ``max_local_ids + W * TILE``
    (local sender batch plus at most one TILE of sentinel padding per source).
    Greedy wave packing fully drains ``recv_capacity`` per wave (modulo the
    last wave) because remaining per-pair counts are TILE multiples. Hence:

        max_waves = ceil((max_local_ids + W * TILE) / recv_capacity)

    When ``recv_capacity >= max_local_ids + W * TILE`` (the v0 worst-case
    bound) this returns 1 and the kernel runs single-wave with no overhead.
    """
    worst_case = max_local_ids + world_size * _TILE_INT32
    return max(1, (worst_case + recv_capacity - 1) // recv_capacity)


def _max_waves_for_capacity_wo4(
    max_local_ids: int, world_size: int, recv_capacity: int
) -> int:
    """Static wave count for the WO#4 even-split fused path.

    Each per-pair count is rounded up to a multiple of ``max_waves * TILE``
    (the WO#4 invariant), so the worst-case per-pair-per-wave value is
    ``ceil(max_local_ids / (max_waves * TILE)) * TILE`` and the per-receiver
    fan-in per wave is bounded by ``W * that_value``. We solve numerically
    for the smallest ``max_waves`` such that this fan-in fits in
    ``recv_capacity``.
    """
    tile = _TILE_INT32
    for max_waves in range(1, max_local_ids // tile + world_size + 1):
        wo4_pad = max_waves * tile
        max_per_pair_padded = ((max_local_ids + wo4_pad - 1) // wo4_pad) * wo4_pad
        per_wave_per_pair = max_per_pair_padded // max_waves
        if world_size * per_wave_per_pair <= recv_capacity:
            return max_waves
    raise ValueError(
        f"WO#4: cannot fit max_local_ids={max_local_ids}, "
        f"world_size={world_size} into recv_capacity={recv_capacity}; "
        f"increase recv_capacity to at least {world_size * tile + tile}."
    )


def _split_traffic_into_waves(
    padded_traffic: jax.Array,
    *,
    recv_capacity: int,
    max_waves: int,
    world_size: int,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Split padded traffic into per-wave chunks, with per-wave offsets.

    Greedy per-receiver bin-packing: for each wave and receiver ``t``, walk
    sources ``s`` in order and take ``min(remaining[s, t], floor(budget /
    TILE) * TILE)`` rows (so per-pair chunks stay TILE-aligned). Budget
    starts at ``recv_capacity`` and decrements by what each source takes.

    Returns:
        per_wave_traffic: ``[max_waves, W, W]`` int32. Entry ``[w, s, t]`` is
            the rows from ``s`` to ``t`` shipped in wave ``w``.
        per_wave_send_offsets: ``[max_waves, W, W]`` int32. Entry ``[w, s, t]``
            is the start offset in ``s``'s sorted_row_padded buffer for the
            chunk shipped in wave ``w``.
        per_wave_recv_offsets: ``[max_waves, W, W]`` int32. Entry ``[w, t, s]``
            is the start offset in ``t``'s wave-local recv buffer (size
            ``recv_capacity``) for the chunk arriving from ``s`` in wave ``w``.
    """
    per_wave = jnp.zeros((max_waves, world_size, world_size), dtype=jnp.int32)
    remaining = padded_traffic
    for w in range(max_waves):
        for t in range(world_size):
            budget = jnp.int32(recv_capacity)
            for s in range(world_size):
                cap = (budget // _TILE_INT32) * _TILE_INT32
                take = jnp.minimum(remaining[s, t], cap)
                per_wave = per_wave.at[w, s, t].set(take)
                remaining = remaining.at[s, t].add(-take)
                budget = budget - take

    # per_wave_send_offsets[w, s, t] = base_send_offset[s, t]
    #                                + sum_{w' < w} per_wave[w', s, t]
    base_send_offsets = jnp.concatenate(
        [
            jnp.zeros((world_size, 1), dtype=jnp.int32),
            jnp.cumsum(padded_traffic, axis=1)[:, :-1],
        ],
        axis=1,
    ).astype(jnp.int32)
    cum_prior_waves = jnp.cumsum(per_wave, axis=0) - per_wave  # [W, W, W]
    per_wave_send_offsets = (base_send_offsets[None, :, :] + cum_prior_waves).astype(
        jnp.int32
    )

    # per_wave_recv_offsets[w, t, s] = sum_{s' < s} per_wave[w, s', t]
    cum_within_wave = jnp.cumsum(per_wave, axis=1) - per_wave  # along sources
    # Boundary safety: when per_wave[w, s, t] == 0, the natural cumulative
    # offset can land EXACTLY at recv_capacity (e.g., earlier sources fully
    # filled the wave's budget). Even though the corresponding DMA slice has
    # length 0 — which would be a no-op in normal HLO — TPU firmware aborts
    # on a zero-length DMA whose start offset == buffer size. Substitute the
    # safe value 0 for the offset at zero-count entries; it never affects
    # correctness because the DMA slice is empty.
    cum_within_wave = jnp.where(per_wave > 0, cum_within_wave, 0)
    per_wave_recv_offsets = jnp.transpose(cum_within_wave, (0, 2, 1)).astype(jnp.int32)

    # Same fix for send offsets: although the BASE prefix sum is bounded by
    # max_padded (well below the boundary in practice), this mirrors the recv
    # treatment so any 0-count pair has a guaranteed-safe send_start as well.
    per_wave_send_offsets = jnp.where(per_wave > 0, per_wave_send_offsets, 0)

    return per_wave, per_wave_send_offsets, per_wave_recv_offsets


def _split_traffic_into_waves_even(
    padded_traffic: jax.Array,
    *,
    max_waves: int,
    world_size: int,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """WO#4 per-wave EVEN-SPLIT splitter (validated on TPU as v66q + v66r).

    Splits each per-pair total evenly across all ``max_waves`` waves. Requires
    ``padded_traffic`` to be PRE-PADDED such that every entry is a multiple of
    ``max_waves * TILE`` (caller's responsibility — typically via
    ``_round_up_tile(send_counts_real, max_waves * _TILE_INT32)`` followed by
    ``all_gather``). Used only by the fused all-TC f32 path to sidestep the
    W>=4 multi-iter Pallas hang triggered by per-pair DMA counts that differ
    across waves. Same return shapes as ``_split_traffic_into_waves``.
    """
    per_pair_per_wave = padded_traffic // max_waves  # [W, W]
    per_wave = jnp.broadcast_to(
        per_pair_per_wave[None, :, :], (max_waves, world_size, world_size)
    ).astype(jnp.int32)

    base_send_offsets = jnp.concatenate(
        [
            jnp.zeros((world_size, 1), dtype=jnp.int32),
            jnp.cumsum(padded_traffic, axis=1)[:, :-1],
        ],
        axis=1,
    ).astype(jnp.int32)
    wave_idx = jnp.arange(max_waves, dtype=jnp.int32)
    drift = wave_idx[:, None, None] * per_pair_per_wave[None, :, :]  # [W_w, W, W]
    per_wave_send_offsets = (base_send_offsets[None, :, :] + drift).astype(jnp.int32)

    cum_within_wave = jnp.cumsum(per_wave, axis=1) - per_wave  # along sources
    per_wave_recv_offsets = jnp.transpose(cum_within_wave, (0, 2, 1)).astype(jnp.int32)

    per_wave_recv_offsets = jnp.where(per_wave > 0, per_wave_recv_offsets, 0)
    per_wave_send_offsets = jnp.where(per_wave > 0, per_wave_send_offsets, 0)

    return per_wave, per_wave_send_offsets, per_wave_recv_offsets


def _derive_offsets(traffic_matrix: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Derive send / receive offset matrices from the traffic matrix.

    Args:
        traffic_matrix: ``[world_size, world_size]`` int32. Entry ``[s, t]`` is
            the number of IDs sent from rank ``s`` to rank ``t``.

    Returns:
        send_offsets: ``[world_size, world_size]`` — entry ``[s, t]`` is the
            start offset in rank ``s``'s sorted_ids array of the slice
            destined for rank ``t``.
        recv_offsets: ``[world_size, world_size]`` — entry ``[t, s]`` is the
            start offset in rank ``t``'s receive buffer where the slice from
            rank ``s`` lands.
    """
    # Prefix sums along columns (within each source row).
    send_offsets = jnp.concatenate(
        [
            jnp.zeros((traffic_matrix.shape[0], 1), dtype=jnp.int32),
            jnp.cumsum(traffic_matrix, axis=1)[:, :-1],
        ],
        axis=1,
    ).astype(jnp.int32)
    # Prefix sums along rows (within each destination column).
    recv_offsets_T = jnp.concatenate(
        [
            jnp.zeros((1, traffic_matrix.shape[1]), dtype=jnp.int32),
            jnp.cumsum(traffic_matrix, axis=0)[:-1, :],
        ],
        axis=0,
    ).astype(jnp.int32)
    # Transpose so that recv_offsets[t, s] = offset on rank t for slice from s.
    recv_offsets = recv_offsets_T.T
    return send_offsets, recv_offsets


# ---------------------------------------------------------------------------
# B. Pallas kernels: TC a2a (in), SC indirect-DMA gather, TC a2a (out)
# ---------------------------------------------------------------------------


# Tile size for HBM int32 dim 0 — Mosaic requires DMA slice offsets along
# tiled dimensions to be multiples of the tile. Per-pair counts and offsets
# in this kernel are asserted aligned via ``pl.multiple_of`` on the runtime
# SMEM reads. The host-side ``_pad_per_peer`` inserts sentinel rows so that
# arbitrary jagged distributions still satisfy this constraint.
_TILE_INT32 = 128


def _make_id_a2a_kernel(*, world_size: int, recv_capacity: int):
    """TC kernel: jagged ID all-to-all (per-pair async remote copies).

    The kernel zero-fills ``recv_row_ref`` from a VMEM scratch BEFORE issuing
    DMAs, so slots past the per-wave received total are guaranteed 0 — a
    valid sentinel row the downstream SC gather can safely dereference. We do
    NOT rely on ``input_output_aliases`` of a JAX-managed zeros buffer here
    because that pattern is fragile across multiple ``pallas_call`` invocations
    (the constant may be deduped, hoisted, or have its alias hint silently
    dropped, leaving stale/garbage HBM in the slots that the per-wave DMAs
    don't write — which the SC indirect-DMA gather would then dereference,
    OOBing into HBM and crashing libtpu firmware).

    Per-pair DMAs are SKIPPED when ``traffic[my_rank, peer] == 0`` via
    ``pl.when``. Issuing a 0-length ``pltpu.make_async_remote_copy`` (even at
    safe offsets) triggers a libtpu firmware abort on v7x — possibly because
    the firmware can't handle a remote-DMA descriptor with size 0 / no
    payload, even though the slice is a no-op semantically.
    """

    def kernel(
        sorted_row_ref,        # [max_padded]      int32  HBM
        traffic_ref,           # [W, W]            int32  SMEM
        send_offsets_ref,      # [W, W]            int32  SMEM
        recv_offsets_ref,      # [W, W]            int32  SMEM
        recv_row_ref,          # [recv_capacity]   int32  HBM (output)
        send_sem,              # cross-device send semaphore
        recv_sem,              # cross-device recv semaphore
        zero_scratch,          # [recv_capacity]   int32  VMEM (scratch)
    ):
        # Initialize the VMEM scratch to all zeros and copy it into the recv
        # HBM buffer. After this, every slot of recv_row_ref is 0; per-pair
        # DMAs below overwrite the touched slots, the rest stay 0 (sentinel).
        zero_scratch[...] = jnp.zeros_like(zero_scratch)
        pltpu.sync_copy(zero_scratch, recv_row_ref)

        my_rank = lax.axis_index("x")

        # Custom cross-device barrier BEFORE issuing DMAs. Every rank must
        # have completed the local zero-fill of its recv buffer before any
        # peer writes data into it; otherwise a fast sender's DMA can land
        # at a slot that the receiver subsequently overwrites with zeros.
        # The custom barrier (in conjunction with a unique ``collective_id``
        # per pallas_call) also gives JAX/libtpu a fresh barrier semaphore
        # for each kernel invocation, which is required when running >1
        # cross-device pallas_call within a single jit (the implicit/default
        # barrier is shared across calls and silently wedges firmware).
        sem = pltpu.get_barrier_semaphore()
        for peer in range(world_size):
            pl.semaphore_signal(
                sem,
                1,
                device_id=(peer,),
                device_id_type=pltpu.DeviceIdType.MESH,
            )
        pl.semaphore_wait(sem, world_size)

        for peer in range(world_size):
            send_count = pl.multiple_of(traffic_ref[my_rank, peer], _TILE_INT32)
            send_start = pl.multiple_of(send_offsets_ref[my_rank, peer], _TILE_INT32)
            remote_recv_start = pl.multiple_of(
                recv_offsets_ref[peer, my_rank], _TILE_INT32
            )

            @pl.when(send_count > 0)
            def _do_send():
                src = sorted_row_ref.at[pl.ds(send_start, send_count)]
                dst = recv_row_ref.at[pl.ds(remote_recv_start, send_count)]
                d = pltpu.make_async_remote_copy(
                    src_ref=src,
                    dst_ref=dst,
                    send_sem=send_sem,
                    recv_sem=recv_sem,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
                d.start()
                d.wait()

    return kernel


def _make_emb_a2a_kernel(*, world_size: int):
    """TC kernel: jagged embedding all-to-all (per-pair async remote copies).

    Send/recv roles swap relative to the ID a2a: this kernel sends the slice
    that was *received* in phase 1 back to the requester, and the local
    receive buffer ends up at the requester's send_offset. The output buffer
    is alias-accumulated across waves: each wave only writes its per-wave
    dest slices, leaving prior-wave writes intact.
    """

    def kernel(
        looked_up_ref,         # [recv_capacity, D] float HBM
        out_init_ref,          # [max_padded, D]    float HBM (aliased -> out_ref)
        traffic_ref,           # [W, W]             int32 SMEM
        send_offsets_ref,      # [W, W]             int32 SMEM
        recv_offsets_ref,      # [W, W]             int32 SMEM
        out_ref,               # [max_padded, D]    float HBM (output)
        send_sem,              # cross-device send semaphore
        recv_sem,              # cross-device recv semaphore
    ):
        del out_init_ref  # aliased to out_ref; access via that name
        my_rank = lax.axis_index("x")

        # See ID a2a kernel: custom barrier + unique collective_id needed to
        # avoid the implicit-barrier-shared-across-pallas_calls firmware abort.
        sem = pltpu.get_barrier_semaphore()
        for peer in range(world_size):
            pl.semaphore_signal(
                sem,
                1,
                device_id=(peer,),
                device_id_type=pltpu.DeviceIdType.MESH,
            )
        pl.semaphore_wait(sem, world_size)

        for peer in range(world_size):
            send_count = pl.multiple_of(traffic_ref[peer, my_rank], _TILE_INT32)
            send_start = pl.multiple_of(recv_offsets_ref[my_rank, peer], _TILE_INT32)
            remote_recv_start = pl.multiple_of(
                send_offsets_ref[peer, my_rank], _TILE_INT32
            )

            # Skip DMA when count is 0 — see note in id_a2a kernel: a 0-length
            # remote-DMA descriptor crashes libtpu firmware on v7x.
            @pl.when(send_count > 0)
            def _do_send():
                src = looked_up_ref.at[pl.ds(send_start, send_count), :]
                dst = out_ref.at[pl.ds(remote_recv_start, send_count), :]
                d = pltpu.make_async_remote_copy(
                    src_ref=src,
                    dst_ref=dst,
                    send_sem=send_sem,
                    recv_sem=recv_sem,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
                d.start()
                d.wait()

    return kernel


def _make_sc_lookup_kernel(*, chunks_per_subcore: int, chunk_rows: int):
    """SC vector-subcore kernel: indirect-DMA gather of D-wide rows.

    Uses the ``plsc.VectorSubcoreMesh`` + ``pl.kernel`` style so each
    subcore manually indexes its slice of the HBM index/output buffers via
    ``axis_index("subcore")``. Per subcore: a Python-level (unrolled) loop
    over chunks of ``chunk_rows`` (= num_lanes) indices, with three
    ``pltpu.sync_copy`` calls per chunk: HBM→VMEM copy of the index slice,
    HBM→VMEM indirect-DMA gather of the D-wide rows
    (``pltpu.sync_copy(table.at[idx_vmem], scratch)``), and a VMEM→HBM
    copy of the gathered tile to the per-subcore output slice. The inner
    chunks share the same per-subcore VMEM scratch, so the loop has to be
    sequential (a ``plsc.parallel_loop`` would race on the scratch). bf16
    works natively because the indirect DMA is byte-granular.

    Larger per-call gather widths than ``chunk_rows`` (= num_lanes) trigger
    a firmware abort on v7x — keep this at the SC SIMD width.
    """

    def kernel(
        table_ref,           # [V_local, D]              float HBM
        indices_ref,         # [num_subcores, rows_per_subcore] int32 HBM
        out_ref,             # [num_subcores, rows_per_subcore, D] float HBM
        idx_vmem,            # [chunk_rows]              int32 VMEM (per-subcore scratch)
        scratch,             # [chunk_rows, D]           float VMEM (per-subcore scratch)
    ):
        subcore_id = lax.axis_index("subcore")
        for c in range(chunks_per_subcore):
            pltpu.sync_copy(
                indices_ref.at[subcore_id, pl.ds(c * chunk_rows, chunk_rows)],
                idx_vmem,
            )
            pltpu.sync_copy(table_ref.at[idx_vmem], scratch)
            pltpu.sync_copy(
                scratch,
                out_ref.at[subcore_id, pl.ds(c * chunk_rows, chunk_rows), :],
            )

    return kernel


# ---------------------------------------------------------------------------
# C. Public API: high-level wrapper combining prep + kernels + un-permute
# ---------------------------------------------------------------------------


def _jagged_a2a_embedding_lookup_local(
    local_ids: jax.Array,
    local_id_count: jax.Array,
    local_embedding_table: jax.Array,
    *,
    axis_name: str,
    world_size: int,
    recv_capacity: int,
    interpret: bool,
) -> jax.Array:
    """Per-device kernel — intended to be wrapped in ``shard_map``.

    Args:
        local_ids: ``[max_local_ids]`` int32 — this rank's padded queries.
        local_id_count: scalar int32 — actual valid count.
        local_embedding_table: ``[V_local, D]`` float — this rank's shard.

    Returns:
        ``[max_local_ids, D]`` float embeddings in caller order.
    """
    max_local_ids = local_ids.shape[0]
    v_local, embedding_dim = local_embedding_table.shape
    dtype = local_embedding_table.dtype

    # Worst-case padded send buffer: real total + at most TILE-1 sentinels
    # per peer. ``recv_capacity`` is now a *wave size* (P99 / safety bound):
    # the kernel runs ``max_waves`` chunked DMA waves, each wave processing
    # at most ``recv_capacity`` received rows. Wave count is derived from
    # the padded traffic matrix at runtime; the static upper bound depends
    # only on Python-known ``max_local_ids`` / ``world_size`` / ``recv_capacity``.
    max_padded = max_local_ids + world_size * _TILE_INT32
    max_waves = _max_waves_for_capacity(max_local_ids, world_size, recv_capacity)

    # --- Step A: bucketize and sort by destination rank (host-side JAX).
    sorted_row_real, send_counts_real, _sort_perm, reverse_perm = _bucketize_and_sort(
        local_ids, local_id_count, max_local_ids, world_size
    )

    # --- Step A': round per-peer counts UP to TILE; insert sentinel rows.
    send_counts_padded = _round_up_tile(send_counts_real, _TILE_INT32)
    sorted_row_padded = _pad_per_peer(
        sorted_row_real,
        send_counts_real,
        send_counts_padded,
        max_local_ids,
        max_padded,
        world_size,
    )

    # --- Step B: gather the full traffic matrix (PADDED) and split into
    # per-wave chunks. Each wave's per-receiver fan-in is bounded by
    # ``recv_capacity``; per-pair offsets within each wave are TILE-aligned.
    traffic_matrix = lax.all_gather(send_counts_padded, axis_name)  # [W, W]
    per_wave_traffic, per_wave_send_offsets, per_wave_recv_offsets = (
        _split_traffic_into_waves(
            traffic_matrix,
            recv_capacity=recv_capacity,
            max_waves=max_waves,
            world_size=world_size,
        )
    )

    HBM = pltpu.MemorySpace.HBM
    SMEM = pltpu.MemorySpace.SMEM
    VMEM = pltpu.MemorySpace.VMEM

    # --- Step C: build the per-wave kernels (compiled once, called per wave).
    id_a2a_kernel = _make_id_a2a_kernel(
        world_size=world_size, recv_capacity=recv_capacity
    )
    emb_a2a_kernel = _make_emb_a2a_kernel(world_size=world_size)

    sc_info = pltpu.get_tpu_info().sparse_core
    num_subcores = sc_info.num_subcores
    num_lanes = sc_info.num_lanes
    assert recv_capacity % (num_subcores * num_lanes) == 0, (
        f"recv_capacity={recv_capacity} must be a multiple of "
        f"num_subcores * num_lanes = {num_subcores * num_lanes}"
    )
    assert embedding_dim % num_lanes == 0, (
        f"embedding_dim={embedding_dim} must be a multiple of num_lanes={num_lanes}"
    )
    rows_per_subcore = recv_capacity // num_subcores
    chunks_per_subcore = rows_per_subcore // num_lanes
    sc_kernel = _make_sc_lookup_kernel(
        chunks_per_subcore=chunks_per_subcore, chunk_rows=num_lanes,
    )

    def _run_wave(traffic_w, send_off_w, recv_off_w, sorted_emb_padded_in, w_int, w_py):
        # Diagnostic: prove we even entered this wave's Python frame. If
        # `entering` prints but `pre-id` doesn't, the failure is at
        # pl.pallas_call setup time of C1 (compilation / barrier alloc).
        jax.debug.print("[KT] w={w} entering", w=w_int, ordered=False)
        # C1: TC kernel — jagged ID all-to-all into per-wave HBM recv buffer.
        # The kernel itself zero-fills recv_row_buf via a VMEM scratch before
        # issuing DMAs, so untouched slots reliably contain row 0 (sentinel).
        # No JAX-level init or input_output_aliases — those silently break in
        # multi-call settings and cause the SC gather to hit garbage indices.
        jax.debug.print("[KT] w={w} pre-id", w=w_int, ordered=False)
        (recv_row_buf,) = pl.pallas_call(
            id_a2a_kernel,
            out_shape=[jax.ShapeDtypeStruct((recv_capacity,), jnp.int32)],
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[
                    pl.BlockSpec(memory_space=HBM),   # sorted_row_padded
                    pl.BlockSpec(memory_space=SMEM),  # traffic_w
                    pl.BlockSpec(memory_space=SMEM),  # send_offsets_w
                    pl.BlockSpec(memory_space=SMEM),  # recv_offsets_w
                ],
                out_specs=[pl.BlockSpec(memory_space=HBM)],
                scratch_shapes=(
                    pltpu.SemaphoreType.DMA,
                    pltpu.SemaphoreType.DMA,
                    pltpu.VMEM((recv_capacity,), jnp.int32),
                ),
            ),
            # CRITICAL: each cross-device pallas_call needs a UNIQUE
            # collective_id so its implicit device barrier semaphore doesn't
            # collide with the previous (or next) kernel's barrier. Without
            # unique IDs, multi-wave runs deadlock or firmware-abort because
            # wave 1's id_a2a barrier signals are received as wave 0's
            # post-DMA acks (or vice-versa).
            compiler_params=pltpu.CompilerParams(
                collective_id=w_py * 2,
            ),
            interpret=interpret,
        )(sorted_row_padded, traffic_w, send_off_w, recv_off_w)
        jax.debug.print("[KT] w={w} post-id pre-sc", w=w_int, ordered=False)

        # C2: SC kernel — indirect-DMA gather of D-wide rows.
        # Intra-device only (sync_copy from local HBM/VMEM), no cross-device
        # barrier to assign — `compiler_params` with collective_id is rejected
        # by the lowerer ("collective_id has to be unspecified or None when
        # not using a custom barrier").
        indices_2d = recv_row_buf.reshape((num_subcores, rows_per_subcore))
        looked_up_buf_3d = pl.kernel(
            sc_kernel,
            out_shape=jax.ShapeDtypeStruct(
                (num_subcores, rows_per_subcore, embedding_dim), dtype
            ),
            mesh=plsc.VectorSubcoreMesh(
                core_axis_name="core",
                subcore_axis_name="subcore",
                num_cores=1,
                num_subcores=num_subcores,
            ),
            scratch_shapes=(
                pltpu.VMEM((num_lanes,), jnp.int32),
                pltpu.VMEM((num_lanes, embedding_dim), dtype),
            ),
            interpret=interpret,
        )(local_embedding_table, indices_2d)
        looked_up_buf = looked_up_buf_3d.reshape((recv_capacity, embedding_dim))
        jax.debug.print("[KT] w={w} post-sc pre-emb", w=w_int, ordered=False)

        # C3: TC kernel — jagged embedding all-to-all back into the (aliased)
        # accumulating ``sorted_emb_padded`` buffer. Per-wave dest offsets
        # land in disjoint slices, so wave writes don't clobber each other.
        (sorted_emb_padded_out,) = pl.pallas_call(
            emb_a2a_kernel,
            out_shape=[jax.ShapeDtypeStruct((max_padded, embedding_dim), dtype)],
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[
                    pl.BlockSpec(memory_space=HBM),   # looked_up_buf
                    pl.BlockSpec(memory_space=HBM),   # out_init (alias)
                    pl.BlockSpec(memory_space=SMEM),  # traffic_w
                    pl.BlockSpec(memory_space=SMEM),  # send_offsets_w
                    pl.BlockSpec(memory_space=SMEM),  # recv_offsets_w
                ],
                out_specs=[pl.BlockSpec(memory_space=HBM)],
                scratch_shapes=(
                    pltpu.SemaphoreType.DMA,
                    pltpu.SemaphoreType.DMA,
                ),
            ),
            input_output_aliases={1: 0},
            # See note in C1: unique collective_id per cross-device kernel.
            compiler_params=pltpu.CompilerParams(
                collective_id=w_py * 2 + 1,
            ),
            interpret=interpret,
        )(
            looked_up_buf,
            sorted_emb_padded_in,
            traffic_w,
            send_off_w,
            recv_off_w,
        )
        jax.debug.print("[KT] w={w} post-emb", w=w_int, ordered=False)
        return sorted_emb_padded_out

    # --- Step D: wave loop. ``max_waves`` is a Python compile-time constant
    # (derived from static shapes), so unroll as a Python for-loop. This makes
    # each wave's ``recv_row_init = jnp.zeros(...)`` constant a *separate*
    # buffer in HLO, avoiding the ``input_output_aliases`` + constant-input
    # mismatch that occurs inside ``lax.fori_loop`` (where a single hoisted
    # constant would be donated/mutated by iter 0 and read as garbage by
    # iter 1, causing the SC gather to dereference stale row indices and the
    # TPU firmware to abort).
    sorted_emb_padded = jnp.zeros((max_padded, embedding_dim), dtype=dtype)
    for w in range(max_waves):
        sorted_emb_padded = _run_wave(
            per_wave_traffic[w],
            per_wave_send_offsets[w],
            per_wave_recv_offsets[w],
            sorted_emb_padded,
            jnp.int32(w),
            w,
        )
    jax.debug.print("[KT] all waves done", ordered=False)

    # --- Step E: strip sentinel rows back to ``[max_local_ids, D]``, then
    # un-permute to caller order.
    sorted_emb = _unpad_per_peer(
        sorted_emb_padded,
        send_counts_real,
        send_counts_padded,
        max_local_ids,
        max_padded,
        world_size,
    )
    jax.debug.print("[KT] post unpad", ordered=False)
    out = sorted_emb[reverse_perm]
    jax.debug.print("[KT] post reverse perm", ordered=False)
    return out


def jagged_a2a_embedding_lookup(
    local_ids: jax.Array,
    local_id_count: jax.Array,
    local_embedding_table: jax.Array,
    *,
    mesh: jax.sharding.Mesh,
    axis_name: str,
    recv_capacity: int,
    interpret: bool = False,
) -> jax.Array:
    """Fused jagged all-to-all + sharded embedding lookup on TPU.

    The embedding table is row-wise sharded: the row for global id ``i`` lives
    on rank ``i % world_size`` at local offset ``i // world_size``.

    Each rank queries a *padded* batch of ``[max_local_ids]`` IDs; only the
    first ``local_id_count`` entries are valid and contribute to the all-to-all
    traffic. The returned embeddings are in the original input order.

    Inputs are passed as per-rank-stacked tensors with a leading ``world_size``
    axis sharded along ``axis_name`` (the canonical ``shard_map`` convention).
    The kernel sees per-device shapes inside.

    Args:
        local_ids: ``[world_size, max_local_ids]`` int32.
        local_id_count: ``[world_size]`` int32 — valid count per rank.
        local_embedding_table: ``[world_size, V_local, D]`` float — table
            shards stacked along the leading rank axis.
        mesh: 1D JAX mesh whose only axis is ``axis_name``.
        axis_name: name of the mesh axis the leading dim is sharded on.
        recv_capacity: per-wave receive buffer size (P99 / safety bound, not
            a worst-case bound). The kernel runs ``ceil((max_local_ids +
            world_size * TILE) / recv_capacity)`` chunked DMA waves; each
            wave processes at most ``recv_capacity`` received rows. Smaller
            capacity = more waves; larger capacity = fewer waves. Setting it
            to ``>= max_local_ids + world_size * TILE`` reverts to the v0
            single-wave behavior with no overhead. Minimum: ``world_size *
            TILE`` (so each peer fits at least one TILE per wave). Must also
            be a multiple of ``num_subcores * num_lanes`` (16 * 16 = 256 on
            v7x) so the SC kernel grid divides evenly.
        interpret: if True, run the Pallas kernel under the JAX interpreter
            (CPU-emulated; for unit tests).

    Returns:
        ``[world_size, max_local_ids, D]`` float embeddings in caller order.
    """
    world_size = mesh.shape[axis_name]
    P = jax.sharding.PartitionSpec

    def per_device(ids, count, table):
        # ``shard_map`` with ``P(axis_name)`` keeps the leading axis but
        # reduces it to size 1 per device — squeeze so the kernel sees the
        # natural per-device shape. Re-add the leading 1 on output so
        # ``out_specs=P(axis_name)`` concatenates correctly across devices.
        out = _jagged_a2a_embedding_lookup_local(
            ids[0],
            count[0],
            table[0],
            axis_name=axis_name,
            world_size=world_size,
            recv_capacity=recv_capacity,
            interpret=interpret,
        )
        return out[None, ...]

    sharded = jax.shard_map(
        per_device,
        mesh=mesh,
        in_specs=(P(axis_name), P(axis_name), P(axis_name)),
        out_specs=P(axis_name),
        check_vma=False,
    )
    return sharded(local_ids, local_id_count, local_embedding_table)


# ---------------------------------------------------------------------------
# D. Reference implementation (non-fused, for tests / numerical baseline)
# ---------------------------------------------------------------------------


def jagged_a2a_embedding_lookup_reference_jax(
    local_ids: jax.Array,
    local_id_count: jax.Array,
    local_embedding_table: jax.Array,
    *,
    mesh: jax.sharding.Mesh,
    axis_name: str,
) -> jax.Array:
    """Canonical TPU baseline using ``lax.ragged_all_to_all`` + ``jnp.take``.

    Same runtime as the fused kernel: per-rank IDs get bucketized + sorted on
    the host, the IDs travel to their owning ranks via ``ragged_all_to_all``,
    each rank does a local table gather, then the embeddings travel back via
    a symmetric ``ragged_all_to_all``. Any output divergence vs. the kernel is
    purely the kernel's implementation (DMA, lookup, alignment).

    Inputs and outputs match :func:`jagged_a2a_embedding_lookup`.
    """
    world_size = mesh.shape[axis_name]
    max_local_ids = local_ids.shape[1]
    embedding_dim = local_embedding_table.shape[2]
    dtype = local_embedding_table.dtype
    P = jax.sharding.PartitionSpec
    # Static recv capacity: each rank can receive up to ``world_size *
    # max_local_ids`` rows (absolute worst case).
    recv_capacity = world_size * max_local_ids

    def per_device(ids, count, table):
        ids = ids[0]
        count = count[0]
        table = table[0]

        sorted_row, send_counts, _, reverse_perm = _bucketize_and_sort(
            ids, count, max_local_ids, world_size
        )
        traffic = lax.all_gather(send_counts, axis_name)  # [W, W]
        send_offsets, recv_offsets = _derive_offsets(traffic)

        my_rank = lax.axis_index(axis_name)
        # Each device sends one slice per peer (so K = W). For ``output_offsets``,
        # entry t = "where my slice lands on peer t" = ``recv_offsets[t, my_rank]``;
        # vectorised over t this is ``recv_offsets[:, my_rank]``.
        input_offsets = send_offsets[my_rank]              # [W]
        send_sizes = traffic[my_rank]                      # [W]
        output_offsets = recv_offsets[:, my_rank]          # [W]
        recv_sizes = traffic[:, my_rank]                   # [W]

        # Phase 1: jagged ID a2a.
        recv_row = jnp.zeros((recv_capacity,), dtype=jnp.int32)
        recv_row = lax.ragged_all_to_all(
            sorted_row, recv_row, input_offsets, send_sizes,
            output_offsets, recv_sizes, axis_name=axis_name,
        )

        # Phase 2: local lookup.
        looked_up = jnp.take(table, recv_row, axis=0)  # [recv_capacity, D]

        # Phase 3: jagged embedding a2a back. Sender/receiver roles swap; we
        # send to peer p starting at recv_offsets[my_rank, p] (which is where
        # peer p's IDs landed), with size traffic[p, my_rank]. The receiver
        # gets it at send_offsets[p, my_rank].
        emb_input_offsets = recv_offsets[my_rank]
        emb_send_sizes = traffic[:, my_rank]
        emb_output_offsets = send_offsets[:, my_rank]
        emb_recv_sizes = traffic[my_rank]
        sorted_emb = jnp.zeros((max_local_ids, embedding_dim), dtype=dtype)
        sorted_emb = lax.ragged_all_to_all(
            looked_up, sorted_emb, emb_input_offsets, emb_send_sizes,
            emb_output_offsets, emb_recv_sizes, axis_name=axis_name,
        )

        return sorted_emb[reverse_perm][None, ...]

    sharded = jax.shard_map(
        per_device,
        mesh=mesh,
        in_specs=(P(axis_name), P(axis_name), P(axis_name)),
        out_specs=P(axis_name),
        check_vma=False,
    )
    return sharded(local_ids, local_id_count, local_embedding_table)


def jagged_a2a_embedding_lookup_reference(
    local_ids: np.ndarray,
    local_id_count: np.ndarray,
    local_embedding_table: np.ndarray,
    world_size: int,
) -> np.ndarray:
    """Reference numpy implementation that simulates the full distributed flow.

    Takes per-rank inputs as a leading-dim "rank" axis and returns the per-rank
    output embeddings in caller order (matching ``local_ids`` layout).

    Args:
        local_ids: ``[world_size, max_local_ids]`` int32.
        local_id_count: ``[world_size]`` int32.
        local_embedding_table: ``[world_size, V_local, D]`` float — table
            shards stacked along the rank axis.

    Returns:
        ``[world_size, max_local_ids, D]`` float embeddings in input order;
        invalid slots zeroed.
    """
    assert local_ids.shape[0] == world_size
    max_local_ids = local_ids.shape[1]
    embedding_dim = local_embedding_table.shape[2]
    dtype = local_embedding_table.dtype

    out = np.zeros((world_size, max_local_ids, embedding_dim), dtype=dtype)

    # For every (rank, valid id) pair, look up the embedding from the rank
    # that owns the row.
    for r in range(world_size):
        n = int(local_id_count[r])
        for i in range(n):
            global_id = int(local_ids[r, i])
            owner_rank = global_id % world_size
            local_row = global_id // world_size
            out[r, i, :] = local_embedding_table[owner_rank, local_row, :]
    return out


# ---------------------------------------------------------------------------
# E. f32-only ALL-TC fused single-call kernel with in-kernel wave loop
# ---------------------------------------------------------------------------
#
# EXPERIMENTAL — see top-of-function warning on
# ``jagged_a2a_embedding_lookup_f32_alltc`` below. Without the WO#4 even-split
# splitter, this variant is known-blocked on real TPU at world_size >= 4 with
# multi-wave overflow.
#
# Structural hang fingerprint (final, narrowed via the W4* litmus ladder in
# ``test_inbound_flush_litmus.py``):
#
#     At W=4 with an in-kernel ``lax.fori_loop`` over waves, the per-peer
#     DMA loop deadlocks IFF the per-pair DMA COUNTS read from SMEM differ
#     across waves (i.e., ``eff_count_ref[w=0, s, t] != eff_count_ref[w=1, s, t]``
#     for any (s, t)). Per-pair OFFSETS differing across waves (cumsum drift,
#     as production splitting naturally produces) is NOT a trigger when counts
#     are equal.
#
# This single axis explains every PASS/HANG result in the litmus ladder:
#   - PASS (per-wave-IDENTICAL counts): v60, v66c, v66d, v66e, v66n, v66q, v66r.
#   - HANG (per-wave-DIFFERENT counts): v60-v66 series prior to v66p,
#     v66a/g/h/i/j/k/l/o/p.
# Earlier candidate triggers (variable per-peer counts, mixed real/dummy
# regions, runtime data-dependent dst selection, DMA-done rescheduling,
# descriptor issue ordering) were red herrings induced by the helper-driven
# wave splitter producing per-wave-different chunks on those inputs.
#
# Validated production workaround (WO#4, "per-wave even-split"): pad each
# total[s, t] to a multiple of ``max_waves * TILE``, then split evenly so
# every wave gets ``total[s, t] // max_waves`` per pair. Per-wave matrices
# are element-wise identical; per-wave offsets still drift via cumsum exactly
# as production already does, but that is harmless once counts are equal.
# Cost: at most ``(max_waves * TILE - 1)`` extra ints per (s, t) pair
# worst-case (~1 KB at int32 for W=4, max_waves=2, TILE=128). The fused
# kernel itself needs no changes — only the host-side splitter
# (``_split_traffic_into_waves`` → ``_split_traffic_into_waves_even``).
# Validated end-to-end on real TPU as ``W4Phase1EvenSplitLitmus`` (v66q,
# everything-identical) and ``W4Phase1EvenSplitProdGeomLitmus`` (v66r,
# production-shape with cumsum offset drift).
#
# Tracked broadly upstream under jax-ml/jax#34411 (distributed Pallas tests
# disabled on v7x). The fingerprint above is much sharper than that issue
# documents and is suitable for a focused upstream bug report.
#
# Structurally distinct from the 3-kernel pipeline above: instead of splitting
# ID a2a / lookup / emb a2a across three ``pl.pallas_call``s and looping over
# waves at the JAX level, this single fused kernel does ALL three phases inside
# one Mosaic-compiled module, with a ``lax.fori_loop`` over waves INSIDE the
# kernel body. Restricted to f32 because the row-at-a-time gather
# (``table_vmem[pl.ds(idx, 1), :]``) only compiles for dtypes whose VMEM dim-0
# tile size is 1 (i.e., not bf16).


def _make_alltc_fused_f32_kernel(
    *,
    world_size: int,
    max_waves: int,
    recv_capacity: int,
    embedding_dim: int,
    v_local: int,
):
    """Single-call all-TC fused f32 kernel with in-kernel wave loop.

    Layout (one per-device call):
      Inputs:
        sorted_row_ref       [max_padded]                       int32   HBM
        table_ref            [V_local, D]                       f32     HBM
        traffic_ref          [max_waves, W, W]                  int32   SMEM
        send_off_ref         [max_waves, W, W]                  int32   SMEM
        recv_off_ref         [max_waves, W, W]                  int32   SMEM
      Output:
        out_ref              [max_padded, D]                    f32     HBM
      Scratch (allocated once, reused across waves):
        recv_row_buf         [recv_capacity]                    int32   HBM
        looked_up_buf        [recv_capacity, D]                 f32     HBM
        zero_int_vmem        [recv_capacity]                    int32   VMEM
        zero_emb_vmem        [recv_capacity, D]                 f32     VMEM
        table_vmem           [V_local, D]                       f32     VMEM
        send_sem_id          DMA sem
        recv_sem_id          DMA sem
        send_sem_emb         DMA sem
        recv_sem_emb         DMA sem
    """

    def kernel(
        sorted_row_ref,
        table_ref,
        traffic_ref,
        send_off_ref,
        recv_off_ref,
        # Outputs: out_ref is the real result. recv_row_buf and looked_up_buf
        # are declared as outputs because Mosaic only allows HBM allocation
        # via outputs (HBM scratches are unsupported); they're discarded by
        # the wrapper.
        out_ref,
        recv_row_buf,
        looked_up_buf,
        zero_int_vmem,
        zero_emb_vmem,
        table_vmem,
        idx_smem,
        barrier_sems,  # [max_waves, 2] REGULAR sem array — pre/post barrier per wave
        send_sem_id,
        recv_sem_id,
        send_sem_emb,
        recv_sem_emb,
    ):
        my_rank = lax.axis_index("x")

        # One-time table copy HBM -> VMEM. The wave loop reads ``table_vmem``
        # for every per-row gather, never re-reading HBM.
        pltpu.sync_copy(table_ref, table_vmem)

        # Zero the emb-init VMEM tile once. It's used to clear out_ref's
        # per-wave dest slices (we don't need that here — each wave writes
        # disjoint slices via DMA — but we DO use it indirectly via the looked
        # _up zero-fill.) Actually only ``zero_int_vmem`` and ``zero_emb_vmem``
        # are needed for zero-filling the per-wave HBM scratches before the
        # cross-device DMAs start.
        zero_int_vmem[...] = jnp.zeros_like(zero_int_vmem)
        zero_emb_vmem[...] = jnp.zeros_like(zero_emb_vmem)

        def wave_body(w, _carry):
            # ------------------------------------------------------------------
            # Phase 0: zero-fill recv_row_buf (sentinel rows) so any untouched
            # slot is a safe "row 0" for the gather. Same idea as the 3-kernel
            # version's id_a2a kernel, but inline here.
            # ------------------------------------------------------------------
            pltpu.sync_copy(zero_int_vmem, recv_row_buf)

            # ------------------------------------------------------------------
            # Cross-device barrier #1 (pre-id-a2a, slot 0): every rank must
            # finish the recv-buf zero-fill before any peer DMAs into it.
            #
            # Per-wave REGULAR sem array slot — same pattern as JAX upstream's
            # ``all_gather.py``. Indexing into the array at runtime ``w`` means
            # each wave gets its own slot, eliminating cross-wave signal
            # collision.
            # ------------------------------------------------------------------
            pre_sem = barrier_sems.at[w, 0]
            for peer in range(world_size):
                pl.semaphore_signal(
                    pre_sem,
                    1,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
            pl.semaphore_wait(pre_sem, world_size)

            # ------------------------------------------------------------------
            # Phase 1: jagged ID all-to-all.
            # ------------------------------------------------------------------
            for peer in range(world_size):
                send_count = pl.multiple_of(traffic_ref[w, my_rank, peer], _TILE_INT32)
                send_start = pl.multiple_of(
                    send_off_ref[w, my_rank, peer], _TILE_INT32
                )
                remote_recv_start = pl.multiple_of(
                    recv_off_ref[w, peer, my_rank], _TILE_INT32
                )

                @pl.when(send_count > 0)
                def _do_id_send():
                    src = sorted_row_ref.at[pl.ds(send_start, send_count)]
                    dst = recv_row_buf.at[pl.ds(remote_recv_start, send_count)]
                    d = pltpu.make_async_remote_copy(
                        src_ref=src,
                        dst_ref=dst,
                        send_sem=send_sem_id,
                        recv_sem=recv_sem_id,
                        device_id=(peer,),
                        device_id_type=pltpu.DeviceIdType.MESH,
                    )
                    d.start()
                    d.wait()

            # ------------------------------------------------------------------
            # Cross-device barrier #2 (post-id-a2a, slot 1): every rank must
            # finish writing into peers' recv_row_buf before any rank starts
            # its local lookup (which reads recv_row_buf). The sender's own
            # ``d.wait()`` only confirms send completion, not that the bytes
            # have landed in the receiver's HBM. This barrier provides the
            # missing receiver-side synchronization that's normally enforced
            # by the JAX runtime between separate ``pl.pallas_call``s in the
            # 3-kernel pipeline (call boundary == implicit DMA flush).
            # ------------------------------------------------------------------
            post_sem = barrier_sems.at[w, 1]
            for peer in range(world_size):
                pl.semaphore_signal(
                    post_sem,
                    1,
                    device_id=(peer,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
            pl.semaphore_wait(post_sem, world_size)

            # ------------------------------------------------------------------
            # Phase 2: local lookup. Bring recv_row_buf into SMEM (per-element
            # scalar reads from VMEM are NOT allowed because int32 VMEM is
            # tiled as ``(128)`` and a runtime-index scalar load can't be
            # statically proven aligned). SMEM scalar reads have no tiling
            # constraint, and using an SMEM scalar as the row index for the
            # ``table_vmem[pl.ds(row_idx, 1), :]`` gather is the v0 pattern.
            # ------------------------------------------------------------------
            pltpu.sync_copy(recv_row_buf, idx_smem)

            # Row-at-a-time gather into the VMEM emb tile. The v0 monolithic
            # kernel used this exact pattern: SMEM scalar index → 1-row VMEM
            # tile slice (f32 VMEM dim-0 tile is 1). Use a JAX-level fori_loop
            # over recv_capacity rows; Mosaic should lower it to a SC-style
            # scalar loop.
            def gather_body(i, __):
                row_idx = idx_smem[i]
                zero_emb_vmem[pl.ds(i, 1), :] = table_vmem[pl.ds(row_idx, 1), :]
                return None

            lax.fori_loop(0, recv_capacity, gather_body, None)

            # One big VMEM->HBM copy of the gathered tile.
            pltpu.sync_copy(zero_emb_vmem, looked_up_buf)

            # ------------------------------------------------------------------
            # NOTE: no second cross-device barrier here.
            # Phase 3 reads from MY own ``looked_up_buf`` (already written by
            # the local sync_copy gather above), and writes to peers' out_ref
            # slices via DMA. Peers don't read MY looked_up_buf during their
            # own phase 3 (each rank writes its own gather). The receive side
            # of the DMA targets ``out_ref`` slices that are DISJOINT across
            # waves (different ``send_off_ref[w, ...]`` per wave), so a slow
            # peer still in wave 0 can safely accept my wave-1 emb DMA into a
            # different out_ref slice without any race.
            # ------------------------------------------------------------------

            # ------------------------------------------------------------------
            # Phase 3: jagged embedding all-to-all back to requester.
            # Send/recv roles swap: use recv_offsets[my_rank, peer] as local
            # send_start, send_offsets[peer, my_rank] as remote_recv_start.
            # ------------------------------------------------------------------
            for peer in range(world_size):
                send_count = pl.multiple_of(traffic_ref[w, peer, my_rank], _TILE_INT32)
                send_start = pl.multiple_of(
                    recv_off_ref[w, my_rank, peer], _TILE_INT32
                )
                remote_recv_start = pl.multiple_of(
                    send_off_ref[w, peer, my_rank], _TILE_INT32
                )

                @pl.when(send_count > 0)
                def _do_emb_send():
                    src = looked_up_buf.at[pl.ds(send_start, send_count), :]
                    dst = out_ref.at[pl.ds(remote_recv_start, send_count), :]
                    d = pltpu.make_async_remote_copy(
                        src_ref=src,
                        dst_ref=dst,
                        send_sem=send_sem_emb,
                        recv_sem=recv_sem_emb,
                        device_id=(peer,),
                        device_id_type=pltpu.DeviceIdType.MESH,
                    )
                    d.start()
                    d.wait()

            return None

        lax.fori_loop(0, max_waves, wave_body, None)

    return kernel


def _jagged_a2a_embedding_lookup_local_f32_alltc(
    local_ids: jax.Array,
    local_id_count: jax.Array,
    local_embedding_table: jax.Array,
    *,
    axis_name: str,
    world_size: int,
    recv_capacity: int,
    interpret: bool,
) -> jax.Array:
    """Per-device kernel — f32 all-TC fused single-call variant.

    Public API parity with :func:`_jagged_a2a_embedding_lookup_local`, but
    the entire pipeline (ID a2a + lookup + emb a2a + wave loop) lives inside
    a single ``pl.pallas_call``.
    """
    assert local_embedding_table.dtype == jnp.float32, (
        f"all-TC fused kernel is f32-only; got {local_embedding_table.dtype}"
    )
    max_local_ids = local_ids.shape[0]
    v_local, embedding_dim = local_embedding_table.shape
    dtype = local_embedding_table.dtype

    # WO#4 (per-wave even-split): pad each per-pair count to a multiple of
    # ``max_waves * TILE`` so every wave gets identical per-pair counts.
    # Sidesteps the W>=4 multi-iter Pallas hang triggered by per-pair DMA
    # counts that differ across waves. Validated end-to-end as v66q + v66r.
    max_waves = _max_waves_for_capacity_wo4(max_local_ids, world_size, recv_capacity)
    wo4_pad = max_waves * _TILE_INT32
    max_padded = max_local_ids + world_size * wo4_pad

    # --- Step A: bucketize and sort by destination rank.
    sorted_row_real, send_counts_real, _sort_perm, reverse_perm = _bucketize_and_sort(
        local_ids, local_id_count, max_local_ids, world_size
    )

    # --- Step A': sentinel padding to ``max_waves * TILE`` (WO#4).
    send_counts_padded = _round_up_tile(send_counts_real, wo4_pad)
    sorted_row_padded = _pad_per_peer(
        sorted_row_real,
        send_counts_real,
        send_counts_padded,
        max_local_ids,
        max_padded,
        world_size,
    )

    # --- Step B: traffic matrix + per-wave EVEN-SPLIT (WO#4).
    traffic_matrix = lax.all_gather(send_counts_padded, axis_name)  # [W, W]
    per_wave_traffic, per_wave_send_offsets, per_wave_recv_offsets = (
        _split_traffic_into_waves_even(
            traffic_matrix,
            max_waves=max_waves,
            world_size=world_size,
        )
    )

    HBM = pltpu.MemorySpace.HBM
    SMEM = pltpu.MemorySpace.SMEM

    kernel = _make_alltc_fused_f32_kernel(
        world_size=world_size,
        max_waves=max_waves,
        recv_capacity=recv_capacity,
        embedding_dim=embedding_dim,
        v_local=v_local,
    )

    sorted_emb_padded, _recv_row_dummy, _looked_up_dummy = pl.pallas_call(
        kernel,
        out_shape=[
            jax.ShapeDtypeStruct((max_padded, embedding_dim), dtype),
            # HBM allocation only via outputs; these are scratch in spirit and
            # are discarded after the call.
            jax.ShapeDtypeStruct((recv_capacity,), jnp.int32),
            jax.ShapeDtypeStruct((recv_capacity, embedding_dim), dtype),
        ],
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec(memory_space=HBM),   # sorted_row_padded
                pl.BlockSpec(memory_space=HBM),   # local_embedding_table
                pl.BlockSpec(memory_space=SMEM),  # per_wave_traffic
                pl.BlockSpec(memory_space=SMEM),  # per_wave_send_offsets
                pl.BlockSpec(memory_space=SMEM),  # per_wave_recv_offsets
            ],
            out_specs=[
                pl.BlockSpec(memory_space=HBM),  # sorted_emb_padded
                pl.BlockSpec(memory_space=HBM),  # recv_row_buf (HBM scratch via output)
                pl.BlockSpec(memory_space=HBM),  # looked_up_buf (HBM scratch via output)
            ],
            scratch_shapes=(
                pltpu.VMEM((recv_capacity,), jnp.int32),              # zero_int_vmem
                pltpu.VMEM((recv_capacity, embedding_dim), dtype),    # zero_emb_vmem / gather output tile
                pltpu.VMEM((v_local, embedding_dim), dtype),          # table_vmem
                SMEM((recv_capacity,), jnp.int32),                    # idx_smem (per-row scalar reads)
                # Per-wave REGULAR sem array, two slots per wave:
                #   slot [w, 0] — pre-id-a2a barrier (zero-fill done)
                #   slot [w, 1] — post-id-a2a barrier (DMAs landed at receiver)
                # See JAX upstream ``all_gather.py`` and vLLM TPU
                # ``all_gather_matmul.py``: per-step sem arrays prevent
                # cross-wave signal collisions that would otherwise wedge a
                # single shared ``get_barrier_semaphore()``.
                pltpu.SemaphoreType.REGULAR((max_waves, 2)),          # barrier_sems
                pltpu.SemaphoreType.DMA,                              # send_sem_id
                pltpu.SemaphoreType.DMA,                              # recv_sem_id
                pltpu.SemaphoreType.DMA,                              # send_sem_emb
                pltpu.SemaphoreType.DMA,                              # recv_sem_emb
            ),
        ),
        # No ``collective_id`` needed: ``pltpu.get_barrier_semaphore()`` is no
        # longer used; we have a custom REGULAR sem array instead.
        interpret=interpret,
    )(
        sorted_row_padded,
        local_embedding_table,
        per_wave_traffic,
        per_wave_send_offsets,
        per_wave_recv_offsets,
    )
    del _recv_row_dummy, _looked_up_dummy

    sorted_emb = _unpad_per_peer(
        sorted_emb_padded,
        send_counts_real,
        send_counts_padded,
        max_local_ids,
        max_padded,
        world_size,
    )
    out = sorted_emb[reverse_perm]
    return out


def jagged_a2a_embedding_lookup_f32_alltc(
    local_ids: jax.Array,
    local_id_count: jax.Array,
    local_embedding_table: jax.Array,
    *,
    mesh: jax.sharding.Mesh,
    axis_name: str,
    recv_capacity: int,
    interpret: bool = False,
) -> jax.Array:
    """f32-only all-TC fused single-call variant.

    EXPERIMENTAL — without the WO#4 even-split splitter, this variant is
    known-blocked on real TPU at world_size >= 4 with multi-wave overflow.

    Structural hang fingerprint (final, narrowed via the W4* litmus ladder
    in ``test_inbound_flush_litmus.py``): at W=4 with an in-kernel
    ``lax.fori_loop`` over waves, the per-peer DMA loop deadlocks IFF the
    per-pair DMA COUNTS read from SMEM differ across waves
    (``eff_count_ref[w=0, s, t] != eff_count_ref[w=1, s, t]`` for any
    (s, t)). Per-pair OFFSETS differing across waves (cumsum drift, as
    production splitting naturally produces) is NOT a trigger when counts
    are equal. Validated by v66p HANG (manual per-wave-different counts) +
    v66r PASS (per-wave-identical counts with cumsum offset drift).

    Earlier candidate triggers — variable per-peer counts, mixed real/dummy
    regions, runtime data-dependent dst selection, DMA-done rescheduling,
    descriptor issue ordering — were all red herrings induced by the
    helper-driven wave splitter producing per-wave-different chunks on
    those inputs. Tracked broadly under jax-ml/jax#34411.

    Validated production workaround (WO#4, "per-wave even-split"): pad each
    total[s, t] to a multiple of ``max_waves * TILE``, then split evenly so
    every wave gets ``total[s, t] // max_waves`` per pair. Per-wave count
    matrices are element-wise identical; cumsum offset drift is harmless
    once counts are equal. Cost: at most ``(max_waves * TILE - 1)`` extra
    ints per (s, t) pair worst-case (~1 KB at int32 for W=4, max_waves=2,
    TILE=128). The fused kernel itself needs no changes — only the
    host-side splitter
    (``_split_traffic_into_waves`` → ``_split_traffic_into_waves_even``).

    Public API parity with :func:`jagged_a2a_embedding_lookup`. Internally
    everything (ID a2a + table gather + emb a2a + wave loop) lives inside
    one ``pl.pallas_call`` whose body uses ``lax.fori_loop`` for the wave
    dimension. f32 only.
    """
    assert local_embedding_table.dtype == jnp.float32, (
        f"all-TC fused kernel is f32-only; got {local_embedding_table.dtype}"
    )
    world_size = mesh.shape[axis_name]
    P = jax.sharding.PartitionSpec

    def per_device(ids, count, table):
        out = _jagged_a2a_embedding_lookup_local_f32_alltc(
            ids[0],
            count[0],
            table[0],
            axis_name=axis_name,
            world_size=world_size,
            recv_capacity=recv_capacity,
            interpret=interpret,
        )
        return out[None, ...]

    sharded = jax.shard_map(
        per_device,
        mesh=mesh,
        in_specs=(P(axis_name), P(axis_name), P(axis_name)),
        out_specs=P(axis_name),
        check_vma=False,
    )
    return sharded(local_ids, local_id_count, local_embedding_table)


# ---------------------------------------------------------------------------
# Ring-shift variants. Per-wave per-rank ONE DMA; W waves total. Sidesteps the
# W=4 fingerprint without WO#4 padding overhead.
#
# Two splitters differ only in the recv-buffer LAYOUT cnt per wave:
#   * naive     -- layout cnt[w] = max over wave-w diagonal pairs (tight)
#   * globalpad -- layout cnt    = traffic_matrix.max() constant (data-indep.)
# ACTUAL Phase 1 / Phase 3 DMA counts are per-rank per-wave (= the real
# per-pair traffic), so the kernel never overshoots its peer's chunk in
# sorted_row_padded or src's out_ref. Slack bytes inside the layout slot stay
# as zero from the wave-init zero copy.
#
# Validated end-to-end on v7x (W=4, f32, skewed input):
#   v67a (naive, Phase-1-only litmus)        PASS  2026-05-08
#   v67b (globalpad, Phase-1-only litmus)    PASS  2026-05-08
#   v68b (full bench: numpy / ra2a oracle)   PASS  2026-05-08  match=True for
#                                            both naive AND globalpad after
#                                            splitting layout cnt vs DMA cnt
#                                            (v67c/v68a globalpad had max_diff
#                                            ~6.35 because layout-cnt was
#                                            mis-used as Phase 1 / Phase 3 DMA
#                                            cnt, overshooting peer chunks).
# ---------------------------------------------------------------------------


def _ringshift_phase3_remote_offset(
    traffic_matrix: jax.Array,
    *,
    world_size: int,
    my_rank: jax.Array,
) -> jax.Array:
    """Remote dst offset for Phase 3 emb-back DMA per wave.

    At wave w, my Phase 3 dst is src=(my_rank-w+W)%W, and the remote slot is
    src's padded_offset[my_rank] = sum_{i<my_rank} traffic_matrix[src, i].
    Returns [W] int32 (per-wave remote offset into src's out_ref).
    """
    W = world_size
    # padded_offsets_all[r, p] = sum_{i<p} traffic_matrix[r, i]
    padded_offsets_all = (
        jnp.cumsum(traffic_matrix, axis=1) - traffic_matrix
    ).astype(jnp.int32)  # [W, W]
    src_per_wave = (my_rank - jnp.arange(W, dtype=jnp.int32) + W) % W  # [W]
    # For each w, lookup padded_offsets_all[src_per_wave[w], my_rank].
    return padded_offsets_all[src_per_wave, my_rank]


def _split_traffic_ringshift_naive(
    traffic_matrix: jax.Array,
    send_counts_padded: jax.Array,
    *,
    world_size: int,
    my_rank: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Ring-shift NAIVE splitter: layout cnt[w] = max over wave-w diagonal pairs.

    Returns:
        p1_cnt_per_wave [W] -- per-rank, ACTUAL Phase 1 DMA count for wave w
                                (= traffic_matrix[my_rank, (my_rank+w)%W]).
        p3_cnt_per_wave [W] -- per-rank, ACTUAL Phase 3 DMA count for wave w
                                (= traffic_matrix[(my_rank-w)%W, my_rank]).
        send_off_local [W] -- per-rank, offset in MY sorted_row_padded for
                              wave w's outbound (Phase 1).
        recv_off_global [W] -- global, layout offset where wave w's inbound
                                lands in recv_row_buf (cumsum of layout cnt).
        phase3_remote_off [W] -- per-rank, dst offset in remote out_ref for
                                  Phase 3 emb-back DMA.
    """
    W = world_size
    # Single-slot recv layout: every wave writes to recv_row_buf[0 : p1_cnt[w]]
    # and Phase 3 reads from looked_up_buf[0 : p3_cnt[w]]. recv_row_buf is zeroed
    # at the top of each wave, so waves don't accumulate. Caller must pass
    # recv_capacity >= max over (s, t) of traffic_matrix[s, t].
    recv_off_global = jnp.zeros((W,), dtype=jnp.int32)
    padded_offsets = (
        jnp.cumsum(send_counts_padded) - send_counts_padded
    ).astype(jnp.int32)
    peer_per_wave = (my_rank + jnp.arange(W, dtype=jnp.int32)) % W
    send_off_local = padded_offsets[peer_per_wave]
    p1_cnt_per_wave = send_counts_padded[peer_per_wave].astype(jnp.int32)
    src_per_wave = (my_rank - jnp.arange(W, dtype=jnp.int32) + W) % W
    p3_cnt_per_wave = traffic_matrix[src_per_wave, my_rank].astype(jnp.int32)
    phase3_remote_off = _ringshift_phase3_remote_offset(
        traffic_matrix, world_size=W, my_rank=my_rank
    )
    return (
        p1_cnt_per_wave, p3_cnt_per_wave,
        send_off_local, recv_off_global, phase3_remote_off,
    )


def _split_traffic_ringshift_globalpad(
    traffic_matrix: jax.Array,
    send_counts_padded: jax.Array,
    *,
    world_size: int,
    my_rank: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Ring-shift GLOBAL-PAD splitter.

    With single-slot recv layout, "globalpad" and "naive" produce IDENTICAL
    splitter output for the kernel — both use recv_off_global = 0 for every
    wave, and both share the same per-rank per-wave actual DMA counts (p1_cnt,
    p3_cnt). The variant survives only as an API / labeling convenience: the
    caller's recv_capacity is sized for the data-independent worst case
    (= traffic_matrix.max() upper bound) rather than tracked per-input.
    """
    W = world_size
    # Single-slot recv layout (see naive splitter docstring). Caller must pass
    # recv_capacity >= max_local_ids + TILE (NO W factor -- single-slot recv
    # layout means only one peer's payload per wave; per-wave incoming for any
    # rank is bounded by max_local_ids + TILE - 1, the worst-case all-to-one-
    # peer skew).
    recv_off_global = jnp.zeros((W,), dtype=jnp.int32)
    padded_offsets = (
        jnp.cumsum(send_counts_padded) - send_counts_padded
    ).astype(jnp.int32)
    peer_per_wave = (my_rank + jnp.arange(W, dtype=jnp.int32)) % W
    send_off_local = padded_offsets[peer_per_wave]
    p1_cnt_per_wave = send_counts_padded[peer_per_wave].astype(jnp.int32)
    src_per_wave = (my_rank - jnp.arange(W, dtype=jnp.int32) + W) % W
    p3_cnt_per_wave = traffic_matrix[src_per_wave, my_rank].astype(jnp.int32)
    phase3_remote_off = _ringshift_phase3_remote_offset(
        traffic_matrix, world_size=W, my_rank=my_rank
    )
    return (
        p1_cnt_per_wave, p3_cnt_per_wave,
        send_off_local, recv_off_global, phase3_remote_off,
    )


def _make_ringshift_single_wave_kernel(
    *,
    world_size: int,
    recv_capacity: int,
    embedding_dim: int,
    v_local: int,
):
    """Single-wave all-TC fused f32 ring-shift kernel (v69d-style structure).

    The previous design folded the W-wave loop INSIDE one Pallas call as
    ``lax.fori_loop`` (v69w) but at W=32 multi-iter that hung. Codex's
    pivot: revert to the v69d-style multi-call structure where the wave
    loop is hoisted to a JAX-level Python ``for`` loop over individual
    ``pl.pallas_call`` invocations, each with a UNIQUE
    ``compiler_params=CompilerParams(collective_id=w_py)``. Per-call
    unique collective_id ensures the implicit device-barrier semaphore
    for one wave can't collide with the previous (or next) wave's
    semaphore.

    This kernel is the single-wave body: it expects the wave index plus
    the per-rank scalars for that one wave (p1_cnt, p3_cnt, send_off,
    recv_off, phase3_remote_off) as scalar (shape ``(1,)``) SMEM inputs.
    The accumulating ``out`` buffer is aliased through
    ``input_output_aliases`` so each wave writes a disjoint slice and the
    accumulator survives across calls.

    Pair-handshake pre-barrier: 1 signal + 1 wait per wave, with
    my_src/my_dst computed from the scalar wave_idx — ringshift has
    exactly ONE sender->me + me->ONE receiver per wave (same peer when
    wave_idx == 0). Uses ``pltpu.get_barrier_semaphore()`` (custom
    barrier), required by Mosaic when ``collective_id`` is set on
    ``CompilerParams``.
    """

    def kernel(
        sorted_row_ref,
        table_ref,
        wave_idx_ref,           # SMEM (1,) int32 — this wave's index w
        p1_cnt_ref,             # SMEM (1,) int32 — this wave's p1_cnt
        p3_cnt_ref,             # SMEM (1,) int32 — this wave's p3_cnt
        send_off_ref,           # SMEM (1,) int32 — this wave's send_off
        recv_off_ref,           # SMEM (1,) int32 — this wave's recv_off
        phase3_remote_off_ref,  # SMEM (1,) int32 — this wave's phase3_remote_off
        out_init_ref,           # input HBM, aliased to out_ref (accumulator)
        out_ref,                # output HBM, alias of out_init_ref
        recv_row_buf,
        looked_up_buf,
        zero_int_vmem,
        zero_emb_vmem,
        table_vmem,
        idx_smem,
        send_sem_id,
        recv_sem_id,
        send_sem_emb,
        recv_sem_emb,
    ):
        del out_init_ref  # aliased to out_ref via input_output_aliases
        my_rank = lax.axis_index("x")
        wave_idx = wave_idx_ref[0]  # traced scalar int32

        pl.debug_print("[K] START r={} w={}", my_rank, wave_idx)

        # All-peer broadcast pre-barrier FIRST (v70s pattern). Moved before
        # the in-kernel init operations because issuing init ops (sync_copy
        # of table, zeroing scratches, sync_copy of zero into recv_row_buf)
        # BEFORE the cross-rank barrier appears to interfere with the
        # barrier at large W (W=32 hangs in production order; v70s order
        # works). Keeping the barrier as the first cross-rank op matches
        # the working v70s litmus structure.
        # Reverted from pair-handshake (which used a TRACED device_id and
        # may have a multi-host routing bug) back to the original static
        # Python-loop unrolled signals to all peers. The custom barrier
        # semaphore (``pltpu.get_barrier_semaphore()``) is required by
        # Mosaic when ``collective_id`` is set on ``CompilerParams`` (set
        # by the wrapper, unique per wave). Each call's barrier sem is a
        # fresh slot so signals can't leak across waves.
        sem = pltpu.get_barrier_semaphore()
        my_src = (my_rank - wave_idx + world_size) % world_size  # who sends to me
        my_dst = (my_rank + wave_idx) % world_size               # who I send to
        for peer in range(world_size):
            pl.semaphore_signal(
                sem, 1,
                device_id=(peer,),
                device_id_type=pltpu.DeviceIdType.MESH,
            )
        pl.semaphore_wait(sem, world_size)
        pl.debug_print("[K] after_barrier r={} w={}", my_rank, wave_idx)

        # THEN initialize scratches (table prefetch, zero scratches, zero
        # the recv buffer). After the barrier so init can't race with it.
        pltpu.sync_copy(table_ref, table_vmem)
        pl.debug_print("[K] after_table_copy r={} w={}", my_rank, wave_idx)
        zero_int_vmem[...] = jnp.zeros_like(zero_int_vmem)
        zero_emb_vmem[...] = jnp.zeros_like(zero_emb_vmem)
        pltpu.sync_copy(zero_int_vmem, recv_row_buf)
        pl.debug_print("[K] after_init r={} w={}", my_rank, wave_idx)

        # Phase 1: ID DMA — send my chunk for the (my_rank+wave_idx)-th
        # peer, receive into recv_row_buf at this wave's recv offset.
        # Hoist BOTH p1_cnt and p3_cnt early: codex's fix needs p3_cnt
        # available for the Phase 1 recv-side drain (recv extent = OPPOSITE
        # side's send count = p3_cnt). Also pre-compute phase3_remote_start
        # for symmetry though it's only used in Phase 3.
        p1_cnt = pl.multiple_of(p1_cnt_ref[0], _TILE_INT32)
        p3_cnt = pl.multiple_of(p3_cnt_ref[0], _TILE_INT32)
        send_start = pl.multiple_of(send_off_ref[0], _TILE_INT32)
        remote_recv_start = pl.multiple_of(recv_off_ref[0], _TILE_INT32)
        phase3_remote_start = pl.multiple_of(
            phase3_remote_off_ref[0], _TILE_INT32
        )

        # === v71d: Codex Option B — separate inbound DMA descriptor ===
        # The wait_recv() extent is determined by the descriptor's slice, not
        # by the recv_sem signals it drains. So we build TWO descriptors:
        #   d_send: my outbound (extent p1_cnt) — start() + wait_send()
        #   d_recv: peer's send to me (extent p3_cnt = OPPOSITE side's
        #           send count) — wait_recv() ONLY (no start, peer issues it).
        # The DeviceID on d_recv is my_src (who is sending TO me); src/dst
        # refs describe the slice on MY side that gets drained when the peer's
        # signals arrive.
        d_send = pltpu.make_async_remote_copy(
            src_ref=sorted_row_ref.at[pl.ds(send_start, p1_cnt)],
            dst_ref=recv_row_buf.at[pl.ds(remote_recv_start, p1_cnt)],
            send_sem=send_sem_id, recv_sem=recv_sem_id,
            device_id=(my_dst,),
            device_id_type=pltpu.DeviceIdType.MESH,
        )
        d_recv = pltpu.make_async_remote_copy(
            src_ref=sorted_row_ref.at[pl.ds(0, p3_cnt)],
            dst_ref=recv_row_buf.at[pl.ds(0, p3_cnt)],
            send_sem=send_sem_id, recv_sem=recv_sem_id,
            device_id=(my_src,),
            device_id_type=pltpu.DeviceIdType.MESH,
        )

        @pl.when(p1_cnt > 0)
        def _phase1_send():
            d_send.start()
            d_send.wait_send()  # only wait for OUR SEND completion (extent p1_cnt)

        @pl.when(p3_cnt > 0)
        def _phase1_recv():
            d_recv.wait_recv()  # drains recv_sem by p3_cnt (peer's send extent)
        pl.debug_print(
            "[K] after_phase1 r={} w={} cnt={}", my_rank, wave_idx, p1_cnt,
        )

        # Phase 2: local lookup.
        pltpu.sync_copy(recv_row_buf, idx_smem)

        def gather_body(i, __):
            row_idx = idx_smem[i]
            zero_emb_vmem[pl.ds(i, 1), :] = table_vmem[pl.ds(row_idx, 1), :]
            return None

        lax.fori_loop(0, recv_capacity, gather_body, None)
        pltpu.sync_copy(zero_emb_vmem, looked_up_buf)
        pl.debug_print("[K] after_gather r={} w={}", my_rank, wave_idx)

        # Phase 3: emb a2a back to source. p3_cnt is the ACTUAL count of
        # embeddings I'm responsible for in this wave (= what src DMA'd to
        # me in Phase 1). Each wave writes to a DISJOINT slice of the
        # global accumulator (different phase3_remote_start per wave), so
        # waves don't collide on the aliased ``out_ref`` accumulator.
        # Phase 3 destination peer = my_src (send back to the rank that
        # sent us its IDs in Phase 1). p3_cnt and phase3_remote_start were
        # hoisted earlier so the codex Phase 1 recv-drain can reference
        # p3_cnt; reuse them here directly.

        # === v71d: Codex Option B — separate inbound Phase 3 descriptor ===
        # Phase 3 send: my outbound back to my_src (extent p3_cnt rows × D).
        # Phase 3 recv: my_dst sends embeddings for the p1_cnt IDs I sent
        # (extent p1_cnt rows × D = OPPOSITE side's send count).
        d3_send = pltpu.make_async_remote_copy(
            src_ref=looked_up_buf.at[pl.ds(remote_recv_start, p3_cnt), :],
            dst_ref=out_ref.at[pl.ds(phase3_remote_start, p3_cnt), :],
            send_sem=send_sem_emb, recv_sem=recv_sem_emb,
            device_id=(my_src,),  # send back to my source
            device_id_type=pltpu.DeviceIdType.MESH,
        )
        d3_recv = pltpu.make_async_remote_copy(
            src_ref=looked_up_buf.at[pl.ds(0, p1_cnt), :],
            dst_ref=out_ref.at[pl.ds(0, p1_cnt), :],
            send_sem=send_sem_emb, recv_sem=recv_sem_emb,
            device_id=(my_dst,),  # peer who will send TO me
            device_id_type=pltpu.DeviceIdType.MESH,
        )

        @pl.when(p3_cnt > 0)
        def _phase3_send():
            d3_send.start()
            d3_send.wait_send()  # only wait for OUR SEND completion (p3_cnt * D)

        @pl.when(p1_cnt > 0)
        def _phase3_recv():
            d3_recv.wait_recv()  # drains recv_sem_emb by p1_cnt rows × D
        pl.debug_print(
            "[K] after_phase3 r={} w={} cnt={}", my_rank, wave_idx, p3_cnt,
        )

    return kernel


def _jagged_a2a_embedding_lookup_local_f32_alltc_ringshift(
    local_ids: jax.Array,
    local_id_count: jax.Array,
    local_embedding_table: jax.Array,
    *,
    axis_name: str,
    world_size: int,
    recv_capacity: int,
    interpret: bool,
    splitter: str,
) -> jax.Array:
    """Per-device ring-shift fused kernel."""
    assert local_embedding_table.dtype == jnp.float32
    max_local_ids = local_ids.shape[0]
    v_local, embedding_dim = local_embedding_table.shape
    dtype = local_embedding_table.dtype
    max_waves = world_size

    sorted_row_real, send_counts_real, _sort_perm, reverse_perm = _bucketize_and_sort(
        local_ids, local_id_count, max_local_ids, world_size
    )
    send_counts_padded = _round_up_tile(send_counts_real, _TILE_INT32)
    traffic_matrix = lax.all_gather(send_counts_padded, axis_name)  # [W, W]
    my_rank = lax.axis_index(axis_name)

    if splitter == "naive":
        p1_cnt, p3_cnt, send_off, recv_off, phase3_remote_off = (
            _split_traffic_ringshift_naive(
                traffic_matrix, send_counts_padded,
                world_size=world_size, my_rank=my_rank,
            )
        )
    elif splitter == "globalpad":
        p1_cnt, p3_cnt, send_off, recv_off, phase3_remote_off = (
            _split_traffic_ringshift_globalpad(
                traffic_matrix, send_counts_padded,
                world_size=world_size, my_rank=my_rank,
            )
        )
    else:
        raise ValueError(f"unknown splitter {splitter}")

    # Tight upper bound for send buffer (sum of TILE-padded per-peer chunks):
    # sum_p send_counts_padded[p] <= sum_p send_counts_real[p] + W*(TILE-1)
    # <= max_local_ids + W*TILE. The earlier `W * (max_local_ids + W*TILE)`
    # was W× over-allocated — harmless at W=4 (~7 MB at D=128), but at W=128
    # it ballooned `sorted_emb_padded` to 4M rows × D and blew VMEM during
    # Pallas compile.
    max_padded = max_local_ids + world_size * _TILE_INT32
    sorted_row_padded = _pad_per_peer(
        sorted_row_real,
        send_counts_real,
        send_counts_padded,
        max_local_ids,
        max_padded,
        world_size,
    )

    HBM = pltpu.MemorySpace.HBM
    SMEM = pltpu.MemorySpace.SMEM

    single_wave_kernel = _make_ringshift_single_wave_kernel(
        world_size=world_size,
        recv_capacity=recv_capacity,
        embedding_dim=embedding_dim,
        v_local=v_local,
    )

    # JAX-level Python wave loop (v69d-style). Each pallas_call gets a
    # UNIQUE ``collective_id`` so its implicit device-barrier semaphore
    # doesn't collide with the previous (or next) wave's barrier (same
    # fix as v23 lines around 720/776). Each call's ``out_init`` aliases
    # to ``out`` so the accumulator survives across waves; per-wave dst
    # offsets land on disjoint slices.
    #
    # CROSS-RANK EPOCH BARRIER (v69c): even with unique collective_ids,
    # successive semaphore-using epochs can skew across ranks across
    # repeated invocations of the same jit — wave w on rank A may launch
    # before wave w-1 has fully drained on rank B, so rank A's barrier
    # semaphore in epoch w consumes a signal still in flight from epoch
    # w-1. Stitch the waves together with a psum on a real value derived
    # from the previous wave's output, then bind that scalar into the
    # next wave's accumulator via ``optimization_barrier`` (XLA can't
    # statically simplify it away). This forces every rank to finish
    # wave w-1 before any rank launches wave w.
    sorted_emb_padded = jnp.zeros((max_padded, embedding_dim), dtype=dtype)
    sync_dep = jnp.float32(0.0)  # initial — no dep yet

    for w_py in range(max_waves):
        # Slice this wave's (1,)-shaped scalars from the (W,)-shaped
        # splitter outputs. SMEM specs expect rank-1 arrays.
        p1_cnt_w = lax.dynamic_slice(p1_cnt, (w_py,), (1,))
        p3_cnt_w = lax.dynamic_slice(p3_cnt, (w_py,), (1,))
        send_off_w = lax.dynamic_slice(send_off, (w_py,), (1,))
        recv_off_w = lax.dynamic_slice(recv_off, (w_py,), (1,))
        phase3_remote_off_w = lax.dynamic_slice(
            phase3_remote_off, (w_py,), (1,)
        )
        wave_idx_w = jnp.array([w_py], dtype=jnp.int32)

        # Bind sync_dep into the accumulator input via optimization_barrier.
        # ``sorted_emb_padded_dep`` is numerically identical to
        # ``sorted_emb_padded`` but XLA treats it as opaquely depending on
        # ``sync_dep``. For w_py=0 this is a no-op (sync_dep is constant
        # 0). For w_py>=1, sync_dep carries the cross-rank psum from the
        # previous wave — the kernel for wave w cannot launch on any rank
        # until every rank has finished wave w-1.
        sorted_emb_padded_dep, _ = jax.lax.optimization_barrier(
            (sorted_emb_padded, sync_dep)
        )

        sorted_emb_padded, _recv_row_dummy, _looked_up_dummy = pl.pallas_call(
            single_wave_kernel,
            out_shape=[
                jax.ShapeDtypeStruct((max_padded, embedding_dim), dtype),
                jax.ShapeDtypeStruct((recv_capacity,), jnp.int32),
                jax.ShapeDtypeStruct((recv_capacity, embedding_dim), dtype),
            ],
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[
                    pl.BlockSpec(memory_space=HBM),   # sorted_row
                    pl.BlockSpec(memory_space=HBM),   # table
                    pl.BlockSpec(memory_space=SMEM),  # wave_idx
                    pl.BlockSpec(memory_space=SMEM),  # p1_cnt
                    pl.BlockSpec(memory_space=SMEM),  # p3_cnt
                    pl.BlockSpec(memory_space=SMEM),  # send_off
                    pl.BlockSpec(memory_space=SMEM),  # recv_off
                    pl.BlockSpec(memory_space=SMEM),  # phase3_remote_off
                    pl.BlockSpec(memory_space=HBM),   # out_init (alias)
                ],
                out_specs=[
                    pl.BlockSpec(memory_space=HBM),  # out (= aliased out_init)
                    pl.BlockSpec(memory_space=HBM),  # recv_row_buf (HBM scratch)
                    pl.BlockSpec(memory_space=HBM),  # looked_up_buf (HBM scratch)
                ],
                scratch_shapes=(
                    pltpu.VMEM((recv_capacity,), jnp.int32),
                    pltpu.VMEM((recv_capacity, embedding_dim), dtype),
                    pltpu.VMEM((v_local, embedding_dim), dtype),
                    SMEM((recv_capacity,), jnp.int32),
                    # NO REGULAR barrier sem — kernel uses
                    # ``pltpu.get_barrier_semaphore()`` (custom barrier),
                    # required by Mosaic when ``collective_id`` is set
                    # on ``CompilerParams``.
                    pltpu.SemaphoreType.DMA,
                    pltpu.SemaphoreType.DMA,
                    pltpu.SemaphoreType.DMA,
                    pltpu.SemaphoreType.DMA,
                ),
            ),
            input_output_aliases={8: 0},  # out_init (input idx 8) -> out (output idx 0)
            compiler_params=pltpu.CompilerParams(collective_id=w_py),
            interpret=interpret,
        )(
            sorted_row_padded,
            local_embedding_table,
            wave_idx_w,
            p1_cnt_w,
            p3_cnt_w,
            send_off_w,
            recv_off_w,
            phase3_remote_off_w,
            sorted_emb_padded_dep,
        )
        del _recv_row_dummy, _looked_up_dummy

        # Cross-rank epoch barrier: psum a real value derived from this
        # wave's output (so XLA can't statically simplify). One scalar
        # per rank; ``axis_name`` is bound here because we're inside
        # shard_map.
        sync_dep = jax.lax.psum(sorted_emb_padded[0, 0], axis_name)

    sorted_emb = _unpad_per_peer(
        sorted_emb_padded,
        send_counts_real,
        send_counts_padded,
        max_local_ids,
        max_padded,
        world_size,
    )
    out = sorted_emb[reverse_perm]
    return out


def jagged_a2a_embedding_lookup_f32_alltc_ringshift(
    local_ids: jax.Array,
    local_id_count: jax.Array,
    local_embedding_table: jax.Array,
    *,
    mesh: jax.sharding.Mesh,
    axis_name: str,
    recv_capacity: int,
    interpret: bool = False,
    splitter: str = "globalpad",
) -> jax.Array:
    """Ring-shift fused all-TC f32 kernel.

    Per-wave per-rank ONE DMA; W waves total. Validated as v67a (naive) and
    v67b (globalpad) litmus PASS on v7x.
    """
    assert local_embedding_table.dtype == jnp.float32
    world_size = mesh.shape[axis_name]
    P = jax.sharding.PartitionSpec

    def per_device(ids, count, table):
        out = _jagged_a2a_embedding_lookup_local_f32_alltc_ringshift(
            ids[0], count[0], table[0],
            axis_name=axis_name,
            world_size=world_size,
            recv_capacity=recv_capacity,
            interpret=interpret,
            splitter=splitter,
        )
        return out[None, ...]

    sharded = jax.shard_map(
        per_device, mesh=mesh,
        in_specs=(P(axis_name), P(axis_name), P(axis_name)),
        out_specs=P(axis_name),
        check_vma=False,
    )
    return sharded(local_ids, local_id_count, local_embedding_table)


# ---------------------------------------------------------------------------
# v_local-chunking wrapper around the fused ringshift kernel.
#
# Motivation: the fused kernel keeps the FULL local table shard in VMEM
# (`table_vmem` of shape `(v_local, embedding_dim) * f32`). For large
# `v_local * D` (e.g. 1M * 512 * 4B = 2 GB) this blows the ~64 MB VMEM cap.
#
# Approach A (sort + per-chunk dispatch):
#   1. Per rank, sort IDs by which "table chunk" they target — chunk c covers
#      the per-rank rows [c*v_local_chunk_size, (c+1)*v_local_chunk_size). All
#      IDs in chunk c have `(id // W) // v_local_chunk_size == c`. Padding
#      positions are sent to a sentinel chunk (sorts last).
#   2. For each chunk c (Python loop), we dynamic-slice
#      `chunk_max_local_ids` IDs out of the sorted buffer at offset
#      `chunk_offsets[c]`. The kernel sees `chunk_count_c` valid IDs only
#      (the rest are over-slice padding the kernel zeros via _unpad_per_peer).
#   3. We slice the table along v_local: `chunk_table = local_embedding_table
#      [:, c*v_local_chunk_size:(c+1)*v_local_chunk_size, :]`. The IDs are
#      remapped via `chunk_ids = chunk_ids - c * v_local_chunk_size * W` so
#      that `(remapped // W)` indexes the chunk-local slice.
#   4. Each chunk call returns `(W, chunk_max_local_ids, D)` — first
#      `chunk_count_c` rows are valid; rest are zero. Scatter-add back into
#      a `(W, max_local_ids, D)` accumulator at original positions
#      `perm[chunk_offsets[c]:chunk_offsets[c]+chunk_count_c]`.
#
# Each ID falls into exactly one chunk so every output row is written by
# exactly one chunk; .at[].add() is equivalent to .at[].set() under that
# invariant.
# ---------------------------------------------------------------------------


def _chunked_per_rank_sort(
    local_ids: jax.Array,
    local_id_count: jax.Array,
    *,
    world_size: int,
    v_local_chunk_size: int,
    num_chunks: int,
    max_local_ids: int,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Per-rank sort by chunk index.

    Returns:
        sorted_ids: ``[max_local_ids]`` int32 — IDs sorted by chunk_idx.
        chunk_counts: ``[num_chunks]`` int32 — count of IDs per chunk
            (excluding sentinel-padded slots).
        chunk_offsets: ``[num_chunks]`` int32 — cumsum-prefix offsets into
            ``sorted_ids``.
        perm: ``[max_local_ids]`` int32 — sort permutation (so original
            position is recoverable).
    """
    pos = jnp.arange(max_local_ids, dtype=jnp.int32)
    is_valid = pos < local_id_count
    # Chunk index = (id // W) // v_local_chunk_size. For invalid (padding)
    # positions, route to the sentinel chunk (= num_chunks) so they sort to
    # the very end of sorted_ids and get dropped from chunk_counts.
    raw_chunk_idx = (local_ids // world_size) // v_local_chunk_size
    chunk_idx = jnp.where(is_valid, raw_chunk_idx, num_chunks).astype(jnp.int32)
    perm = jnp.argsort(chunk_idx, stable=True).astype(jnp.int32)
    sorted_ids = local_ids[perm]
    bins = jnp.bincount(chunk_idx, length=num_chunks + 1)
    chunk_counts = bins[:num_chunks].astype(jnp.int32)
    chunk_offsets = (jnp.cumsum(chunk_counts) - chunk_counts).astype(jnp.int32)
    return sorted_ids, chunk_counts, chunk_offsets, perm


def _make_cross_rank_sync(mesh: jax.sharding.Mesh, axis_name: str):
    """Build a per-rank -> psum scalar function under shard_map.

    The returned callable takes a ``(W,)`` array sharded along ``axis_name``
    (so each device sees a ``(1,)`` slice), reduces with ``psum`` across the
    axis, and returns a replicated scalar. Used in the chunked wrapper to
    create a cross-rank epoch barrier between per-chunk pallas_call
    dispatches: chunk c+1's ``chunk_table_c`` is forced (via
    optimization_barrier) to depend on chunk c's psum scalar, which itself
    depends on every rank reaching the end of chunk c. Without this, XLA
    can reorder per-chunk kernels across ranks and the cross-rank
    semaphore protocol of the inner kernel deadlocks.
    """
    P = jax.sharding.PartitionSpec

    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(P(axis_name),),
        out_specs=P(),
        check_vma=False,
    )
    def _sync(per_rank_scalar):
        # per_rank_scalar is shape (1,) on each device after sharding split.
        return jax.lax.psum(per_rank_scalar[0], axis_name)

    return _sync


def jagged_a2a_embedding_lookup_f32_alltc_ringshift_chunked(
    local_ids: jax.Array,
    local_id_count: jax.Array,
    local_embedding_table: jax.Array,
    *,
    mesh: jax.sharding.Mesh,
    axis_name: str,
    v_local_chunk_size: int,
    chunk_max_local_ids: int,
    recv_capacity: int,
    interpret: bool = False,
    splitter: str = "globalpad",
    _sync_payload: str = "min",
) -> jax.Array:
    """v_local-chunked wrapper around the fused ringshift kernel.

    Splits the per-rank embedding table along the row axis into chunks of
    ``v_local_chunk_size`` rows each, then dispatches one fused kernel call
    per chunk. Each call sees only the IDs whose destination row falls in
    that chunk, and a slice of the table covering that chunk.

    Args:
        local_ids: ``[W, max_local_ids]`` int32, sharded along ``axis_name``.
        local_id_count: ``[W]`` int32, sharded along ``axis_name``.
        local_embedding_table: ``[W, v_local, D]`` f32, sharded along
            ``axis_name``. ``v_local`` need not divide ``v_local_chunk_size``;
            the last chunk may be partial.
        mesh: JAX mesh.
        axis_name: name of the world-size axis on the mesh.
        v_local_chunk_size: number of per-rank table rows per chunk.
        chunk_max_local_ids: static upper bound on per-chunk valid ID count.
            Should include 2-3x safety factor over (max_local_ids /
            num_chunks) to absorb skew. Will be rounded up to a TILE multiple.
        recv_capacity: per-call receive buffer capacity (passed through to
            the inner kernel).
        interpret: passed through to ``pallas_call``.
        splitter: passed through to inner kernel.
        _sync_payload: shape of the per-chunk psum payload used for the
            cross-rank epoch barrier between chunks. ``"min"`` reduces a
            one-element-per-rank scalar (steady-state default; recommended
            for bench). ``"full"`` reduces the entire chunk_out per rank
            (max-debuggable dependency; use when investigating multi-chunk
            hangs).

    Returns:
        ``[W, max_local_ids, D]`` f32 — same shape and semantics as the
        un-chunked kernel.
    """
    if _sync_payload not in ("min", "full"):
        raise ValueError(
            f"_sync_payload must be 'min' or 'full', got {_sync_payload!r}"
        )
    assert local_embedding_table.dtype == jnp.float32
    world_size = mesh.shape[axis_name]
    max_local_ids = local_ids.shape[1]
    v_local = local_embedding_table.shape[1]
    embedding_dim = local_embedding_table.shape[2]
    num_chunks = (v_local + v_local_chunk_size - 1) // v_local_chunk_size
    # Round chunk_max_local_ids up to TILE so each kernel call's input fits
    # the tile-aligned inner API. The kernel only looks at the first
    # chunk_count_c rows of the chunk_max_local_ids buffer.
    if chunk_max_local_ids % _TILE_INT32 != 0:
        chunk_max_local_ids = (
            (chunk_max_local_ids + _TILE_INT32 - 1) // _TILE_INT32
        ) * _TILE_INT32

    P = jax.sharding.PartitionSpec

    # Step 1: per-rank sort by chunk index, under shard_map.
    def _sort_per_device(ids, count):
        sorted_ids, chunk_counts, chunk_offsets, perm = _chunked_per_rank_sort(
            ids[0], count[0],
            world_size=world_size,
            v_local_chunk_size=v_local_chunk_size,
            num_chunks=num_chunks,
            max_local_ids=max_local_ids,
        )
        return (
            sorted_ids[None, :],
            chunk_counts[None, :],
            chunk_offsets[None, :],
            perm[None, :],
        )

    sort_sharded = jax.shard_map(
        _sort_per_device, mesh=mesh,
        in_specs=(P(axis_name), P(axis_name)),
        out_specs=(P(axis_name), P(axis_name), P(axis_name), P(axis_name)),
        check_vma=False,
    )
    sorted_ids, chunk_counts, chunk_offsets, perm = sort_sharded(
        local_ids, local_id_count
    )

    # Allocate the output accumulator (sharded along axis_name).
    output_total = jnp.zeros(
        (world_size, max_local_ids, embedding_dim),
        dtype=local_embedding_table.dtype,
    )
    # Make sure the accumulator is sharded the same way as inputs so the
    # per-chunk shard_map scatter works on local data.
    output_total = jax.lax.with_sharding_constraint(
        output_total,
        jax.sharding.NamedSharding(mesh, P(axis_name)),
    )

    # Step 2: for each chunk, slice IDs + table, run inner kernel, scatter back.
    #
    # CRITICAL: We force a CROSS-RANK epoch barrier between per-chunk
    # pallas_calls via psum + optimization_barrier on the chunk_table input.
    #
    # An optimization_barrier on output_total (this rank's local accumulator)
    # only orders this rank's kernel sequence — XLA can still emit chunk c+1
    # on rank A while rank B is still in chunk c, and the inner kernel's
    # cross-rank semaphores deadlock when ranks are out of phase.
    #
    # The fix: between chunk c and chunk c+1, every rank reduces a payload
    # derived from chunk c's output via psum across `axis_name`. The
    # resulting replicated scalar `sync_dep` is bound into chunk c+1's
    # `chunk_table_c` via optimization_barrier, so XLA must emit the psum
    # (which itself depends on every rank reaching end-of-chunk-c) before
    # any rank can launch chunk c+1's kernel.
    cross_rank_sync = _make_cross_rank_sync(mesh, axis_name)
    sync_dep = jnp.float32(0.0)

    for c in range(num_chunks):
        # Slice the table along v_local. The last chunk may be partial.
        v_start = c * v_local_chunk_size
        v_end = min(v_start + v_local_chunk_size, v_local)
        actual_chunk_v = v_end - v_start
        # If the chunk would be smaller than v_local_chunk_size, pad it with
        # zero rows so all kernel calls share the same v_local. Using a
        # ones-tail sentinel would shift the row indexing — zeros are safer
        # because the remap math already places valid IDs in [0, actual_chunk_v).
        raw_chunk_table = local_embedding_table[:, v_start:v_end, :]
        if actual_chunk_v < v_local_chunk_size:
            pad_rows = v_local_chunk_size - actual_chunk_v
            raw_chunk_table = jnp.concatenate(
                [
                    raw_chunk_table,
                    jnp.zeros(
                        (world_size, pad_rows, embedding_dim),
                        dtype=local_embedding_table.dtype,
                    ),
                ],
                axis=1,
            )
            raw_chunk_table = jax.lax.with_sharding_constraint(
                raw_chunk_table,
                jax.sharding.NamedSharding(mesh, P(axis_name)),
            )

        # Bind sync_dep into chunk_table via optimization_barrier.
        # chunk_table_c is numerically identical to raw_chunk_table but XLA
        # treats it as opaquely depending on sync_dep. For c=0 this is a
        # no-op (sync_dep is just a constant 0). For c>=1 sync_dep carries
        # the cross-rank psum from chunk c-1 — the kernel for chunk c
        # cannot launch on any rank until every rank has finished chunk c-1.
        chunk_table, _ = lax.optimization_barrier((raw_chunk_table, sync_dep))

        # Per-rank chunk slicing + remap, returning chunk_ids and chunk_count.
        # Close over Python int `c` so it's a static constant inside shard_map.
        c_const = c
        offset_const = c_const * v_local_chunk_size * world_size

        def _slice_per_device(s_ids, c_counts, c_offsets, _c=c_const, _off=offset_const):
            # s_ids: [1, max_local_ids]; c_counts/c_offsets: [1, num_chunks].
            offset = c_offsets[0, _c]
            count = c_counts[0, _c]
            chunk_ids = lax.dynamic_slice(
                s_ids[0], (offset,), (chunk_max_local_ids,)
            )
            # Remap: subtract the global row offset that this chunk represents.
            # chunk_idx = (id // W) // v_local_chunk_size, so
            # (id - c * v_local_chunk_size * W) // W is in [0, v_local_chunk_size).
            chunk_ids = chunk_ids - jnp.int32(_off)
            # Clip to a safe non-negative range so the kernel's `(id // W)`
            # over the over-slice padding does not produce undefined behaviour.
            # Real IDs in this chunk are guaranteed in [0, v_local_chunk_size * W).
            chunk_ids = jnp.maximum(chunk_ids, 0).astype(jnp.int32)
            return chunk_ids[None, :], count[None]

        slice_sharded = jax.shard_map(
            _slice_per_device,
            mesh=mesh,
            in_specs=(P(axis_name), P(axis_name), P(axis_name)),
            out_specs=(P(axis_name), P(axis_name)),
            check_vma=False,
        )
        chunk_ids_arr, chunk_count_arr = slice_sharded(
            sorted_ids, chunk_counts, chunk_offsets
        )

        # Run the inner fused kernel on this chunk.
        chunk_out = jagged_a2a_embedding_lookup_f32_alltc_ringshift(
            chunk_ids_arr,
            chunk_count_arr,
            chunk_table,
            mesh=mesh,
            axis_name=axis_name,
            recv_capacity=recv_capacity,
            interpret=interpret,
            splitter=splitter,
        )
        # chunk_out is [W, chunk_max_local_ids, D] — first chunk_count_c rows
        # are valid; rest are zero (kernel's _unpad_per_peer guarantees this).

        # Step 3: scatter chunk_out back to the global output at original
        # positions perm[chunk_offsets[c]:chunk_offsets[c]+chunk_count_c].
        def _scatter_per_device(out_acc, c_out, c_offsets, p, c_counts, _c=c_const):
            # out_acc: [1, max_local_ids, D]
            # c_out:   [1, chunk_max_local_ids, D]
            # c_offsets: [1, num_chunks]; p: [1, max_local_ids]; c_counts: [1, num_chunks]
            offset = c_offsets[0, _c]
            count = c_counts[0, _c]
            # Positions in the original (un-sorted) array.
            chunk_perm = lax.dynamic_slice(
                p[0], (offset,), (chunk_max_local_ids,)
            )
            # Mask: only the first `count` rows of c_out are valid; the rest
            # are over-slice and should not be added.
            row_mask = (
                jnp.arange(chunk_max_local_ids, dtype=jnp.int32) < count
            )[:, None]
            c_out_masked = jnp.where(row_mask, c_out[0], 0)
            # Mask out positions: clip out-of-range to position 0; we will
            # also add 0 there so it is harmless. (count_<= max_local_ids
            # always holds because all valid IDs are in [0, max_local_ids).)
            safe_perm = jnp.where(
                row_mask[:, 0],
                chunk_perm,
                jnp.int32(0),
            )
            new_acc = out_acc[0].at[safe_perm].add(c_out_masked)
            return new_acc[None, ...]

        scatter_sharded = jax.shard_map(
            _scatter_per_device, mesh=mesh,
            in_specs=(
                P(axis_name), P(axis_name), P(axis_name),
                P(axis_name), P(axis_name),
            ),
            out_specs=P(axis_name),
            check_vma=False,
        )
        output_total = scatter_sharded(
            output_total, chunk_out, chunk_offsets, perm, chunk_counts
        )

        # Cross-rank epoch barrier — psum a payload derived from THIS chunk's
        # kernel output. The resulting replicated scalar is bound into the
        # next chunk's chunk_table via optimization_barrier (above), so XLA
        # cannot launch chunk c+1 on any rank until every rank has finished
        # chunk c (the psum is a true cross-rank join).
        if _sync_payload == "full":
            # Max-debuggable dependency: full reduction over chunk_out.
            local_per_rank = chunk_out.sum(axis=(1, 2))   # shape (W,)
        else:
            # Steady-state default: one-element-per-rank payload — creates
            # the cross-rank join without an extra full-buffer pass.
            local_per_rank = chunk_out[:, :1, :1].sum(axis=(1, 2))   # shape (W,)
        sync_dep = cross_rank_sync(local_per_rank)

    return output_total
