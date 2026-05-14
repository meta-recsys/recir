# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Fused jagged all-to-all + sharded embedding lookup as Pallas TC + SC kernels.

This module implements a TPU-friendly version of the sharded embedding lookup
pattern used in torchrec. The challenge that motivates this kernel:

  * Each rank holds a row-wise shard of an embedding table (rank ``r`` owns
    rows ``i`` such that ``i % world_size == r``).
  * Each rank has a *padded* batch of ``[max_local_ids]`` query IDs whose
    destinations are data-dependent.
  * The natural "torchrec-on-TPU" path -- express the all-to-all as
    ``jax.lax.ragged_all_to_all`` plus ``jnp.take`` plus another
    ``ragged_all_to_all`` -- forces the receive buffer between the two
    collectives to be statically sized to the global worst-case fan-in
    (``world_size * max_local_ids``). For sparse features that ratio is
    typically 1:100 to 1:1000, so the wasted HBM and bandwidth are
    prohibitive.

Design summary (ring-shift fused all-TC f32 variant):

  Step 1 (host JAX):  bucketize local_ids by destination rank, sort, and
                      ``lax.all_gather`` the per-rank send-counts into a
                      static-shape ``[world_size, world_size]`` traffic matrix.
                      Then derive per-wave per-rank scalars (one DMA per rank
                      per wave; W waves total) via the ringshift splitter.
  Step 2 (Pallas):    Per-wave fused kernel inside one ``shard_map``. Each
                      wave does:
                        (a) cross-rank pre-barrier;
                        (b) split-DMA Phase 1 (ID DMA) -- separate send/recv
                            descriptors so each side's DMA extent matches its
                            actual count (the v71d split-DMA fix);
                        (c) local table gather (HBM -> VMEM table prefetch +
                            row-at-a-time scalar gather into VMEM emb tile);
                        (d) split-DMA Phase 3 (embedding DMA back to source).
                      The accumulating output buffer is aliased through
                      ``input_output_aliases`` across waves.
  Step 3 (host JAX):  un-permute the kernel output via the saved
                      sort permutation.

The only XLA-visible buffers are ``[max_local_ids]`` (input) and
``[max_local_ids, D]`` (output) -- both bounded by *local* batch size, never
by the global fan-in.

Caveat on testing: ``interpret=True`` lowers Pallas through HLO discharge,
which materializes ``pl.ds(start, length)`` as ``lax.dynamic_slice`` and
requires ``length`` to be statically known. Runtime-extent slices like
``ref.at[pl.ds(start, runtime_count)]`` only work via the Mosaic backend on
real TPU. So the end-to-end kernel test must run on real TPU (the host-side
prep helpers ``_bucketize_and_sort`` / ``_derive_offsets`` are testable on
CPU separately).
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


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
        sorted_local_row: ``[max_local_ids]`` int32 -- the per-shard row index
            (``id // world_size``) sorted by destination rank, padded with 0s
            for invalid slots.
        send_counts: ``[world_size]`` int32 -- count of valid IDs per
            destination rank.
        sort_perm: ``[max_local_ids]`` int32 -- permutation that turns the input
            order into the sorted order.
        reverse_perm: ``[max_local_ids]`` int32 -- inverse of ``sort_perm``,
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
    padded_src = jnp.clip(padded_offsets[peer_id] + local_real_pos, 0, max_padded - 1)
    gathered = out_padded[padded_src]
    return jnp.where(in_real[:, None], gathered, 0).astype(out_padded.dtype)


def _derive_offsets(traffic_matrix: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Derive send / receive offset matrices from the traffic matrix.

    Args:
        traffic_matrix: ``[world_size, world_size]`` int32. Entry ``[s, t]`` is
            the number of IDs sent from rank ``s`` to rank ``t``.

    Returns:
        send_offsets: ``[world_size, world_size]`` -- entry ``[s, t]`` is the
            start offset in rank ``s``'s sorted_ids array of the slice
            destined for rank ``t``.
        recv_offsets: ``[world_size, world_size]`` -- entry ``[t, s]`` is the
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
# B. Constants used by the kernel and host helpers
# ---------------------------------------------------------------------------


# Tile size for HBM int32 dim 0 -- Mosaic requires DMA slice offsets along
# tiled dimensions to be multiples of the tile. Per-pair counts and offsets
# in this kernel are asserted aligned via ``pl.multiple_of`` on the runtime
# SMEM reads. The host-side ``_pad_per_peer`` inserts sentinel rows so that
# arbitrary jagged distributions still satisfy this constraint.
_TILE_INT32 = 128


# ---------------------------------------------------------------------------
# C. Reference implementations (non-fused, for tests / numerical baseline)
# ---------------------------------------------------------------------------


def ringshift_embedding_lookup_reference_jax(
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

    Inputs and outputs match :func:`ringshift_embedding_lookup`.
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
        input_offsets = send_offsets[my_rank]  # [W]
        send_sizes = traffic[my_rank]  # [W]
        output_offsets = recv_offsets[:, my_rank]  # [W]
        recv_sizes = traffic[:, my_rank]  # [W]

        # Phase 1: jagged ID a2a.
        recv_row = jnp.zeros((recv_capacity,), dtype=jnp.int32)
        recv_row = lax.ragged_all_to_all(
            sorted_row,
            recv_row,
            input_offsets,
            send_sizes,
            output_offsets,
            recv_sizes,
            axis_name=axis_name,
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
            looked_up,
            sorted_emb,
            emb_input_offsets,
            emb_send_sizes,
            emb_output_offsets,
            emb_recv_sizes,
            axis_name=axis_name,
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


def ringshift_embedding_lookup_reference(
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
        local_embedding_table: ``[world_size, V_local, D]`` float -- table
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
# D. Ring-shift splitter helpers (one DMA per rank per wave; W waves total)
#
# Two splitters differ only in the recv-buffer LAYOUT cnt per wave:
#   * naive     -- layout cnt[w] = max over wave-w diagonal pairs (tight)
#   * globalpad -- layout cnt    = traffic_matrix.max() constant (data-indep.)
# ACTUAL Phase 1 / Phase 3 DMA counts are per-rank per-wave (= the real
# per-pair traffic), so the kernel never overshoots its peer's chunk in
# sorted_row_padded or src's out_ref. Slack bytes inside the layout slot stay
# as zero from the wave-init zero copy.
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
    padded_offsets_all = (jnp.cumsum(traffic_matrix, axis=1) - traffic_matrix).astype(
        jnp.int32
    )  # [W, W]
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
    padded_offsets = (jnp.cumsum(send_counts_padded) - send_counts_padded).astype(
        jnp.int32
    )
    peer_per_wave = (my_rank + jnp.arange(W, dtype=jnp.int32)) % W
    send_off_local = padded_offsets[peer_per_wave]
    p1_cnt_per_wave = send_counts_padded[peer_per_wave].astype(jnp.int32)
    src_per_wave = (my_rank - jnp.arange(W, dtype=jnp.int32) + W) % W
    p3_cnt_per_wave = traffic_matrix[src_per_wave, my_rank].astype(jnp.int32)
    phase3_remote_off = _ringshift_phase3_remote_offset(
        traffic_matrix, world_size=W, my_rank=my_rank
    )
    return (
        p1_cnt_per_wave,
        p3_cnt_per_wave,
        send_off_local,
        recv_off_global,
        phase3_remote_off,
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
    splitter output for the kernel -- both use recv_off_global = 0 for every
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
    padded_offsets = (jnp.cumsum(send_counts_padded) - send_counts_padded).astype(
        jnp.int32
    )
    peer_per_wave = (my_rank + jnp.arange(W, dtype=jnp.int32)) % W
    send_off_local = padded_offsets[peer_per_wave]
    p1_cnt_per_wave = send_counts_padded[peer_per_wave].astype(jnp.int32)
    src_per_wave = (my_rank - jnp.arange(W, dtype=jnp.int32) + W) % W
    p3_cnt_per_wave = traffic_matrix[src_per_wave, my_rank].astype(jnp.int32)
    phase3_remote_off = _ringshift_phase3_remote_offset(
        traffic_matrix, world_size=W, my_rank=my_rank
    )
    return (
        p1_cnt_per_wave,
        p3_cnt_per_wave,
        send_off_local,
        recv_off_global,
        phase3_remote_off,
    )


# ---------------------------------------------------------------------------
# E. Pallas kernel: per-wave ring-shift body.
# ---------------------------------------------------------------------------


def _make_ringshift_single_wave_kernel(
    *,
    world_size: int,
    recv_capacity: int,
    embedding_dim: int,
    v_local: int,
):
    """Single-wave all-TC fused f32 ring-shift kernel.

    The wave loop is hoisted to a JAX-level Python ``for`` loop over individual
    ``pl.pallas_call`` invocations, each with a UNIQUE
    ``compiler_params=CompilerParams(collective_id=w_py)``. Per-call unique
    collective_id ensures the implicit device-barrier semaphore for one wave
    can't collide with the previous (or next) wave's semaphore.

    This kernel is the single-wave body: it expects the wave index plus the
    per-rank scalars for that one wave (p1_cnt, p3_cnt, send_off, recv_off,
    phase3_remote_off) as scalar (shape ``(1,)``) SMEM inputs. The
    accumulating ``out`` buffer is aliased through ``input_output_aliases`` so
    each wave writes a disjoint slice and the accumulator survives across
    calls.

    Pre-barrier: 1 signal + 1 wait per wave to all peers. Uses
    ``pltpu.get_barrier_semaphore()`` (custom barrier), required by Mosaic
    when ``collective_id`` is set on ``CompilerParams``.

    Phase 1 / Phase 3 each use the v71d split-DMA pattern: separate send /
    recv descriptors so each side's wait extent matches its own count
    (sender's outbound extent ``p1_cnt`` vs. receiver's inbound extent
    ``p3_cnt`` = OPPOSITE side's send count). Necessary because
    ``wait_recv()`` extent is determined by the descriptor's slice, not by
    the recv_sem signals it drains.
    """

    def kernel(
        sorted_row_ref,
        table_ref,
        wave_idx_ref,  # SMEM (1,) int32 -- this wave's index w
        p1_cnt_ref,  # SMEM (1,) int32 -- this wave's p1_cnt
        p3_cnt_ref,  # SMEM (1,) int32 -- this wave's p3_cnt
        send_off_ref,  # SMEM (1,) int32 -- this wave's send_off
        recv_off_ref,  # SMEM (1,) int32 -- this wave's recv_off
        phase3_remote_off_ref,  # SMEM (1,) int32 -- this wave's phase3_remote_off
        out_init_ref,  # input HBM, aliased to out_ref (accumulator)
        out_ref,  # output HBM, alias of out_init_ref
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

        # All-peer broadcast pre-barrier FIRST. Issuing init ops (sync_copy of
        # table, zeroing scratches, sync_copy of zero into recv_row_buf)
        # BEFORE the cross-rank barrier appears to interfere with the barrier
        # at large W (W=32 hangs in production order; barrier-first works).
        # Keeping the barrier as the first cross-rank op preserves the
        # validated structure. The custom barrier semaphore
        # (``pltpu.get_barrier_semaphore()``) is required by Mosaic when
        # ``collective_id`` is set on ``CompilerParams`` (set by the wrapper,
        # unique per wave).
        sem = pltpu.get_barrier_semaphore()
        my_src = (my_rank - wave_idx + world_size) % world_size  # who sends to me
        my_dst = (my_rank + wave_idx) % world_size  # who I send to
        for peer in range(world_size):
            pl.semaphore_signal(
                sem,
                1,
                device_id=(peer,),
                device_id_type=pltpu.DeviceIdType.MESH,
            )
        pl.semaphore_wait(sem, world_size)

        # THEN initialize scratches (table prefetch, zero scratches, zero
        # the recv buffer). After the barrier so init can't race with it.
        pltpu.sync_copy(table_ref, table_vmem)
        zero_int_vmem[...] = jnp.zeros_like(zero_int_vmem)
        zero_emb_vmem[...] = jnp.zeros_like(zero_emb_vmem)
        pltpu.sync_copy(zero_int_vmem, recv_row_buf)

        # Phase 1: ID DMA -- send my chunk for the (my_rank+wave_idx)-th
        # peer, receive into recv_row_buf at this wave's recv offset.
        # Hoist BOTH p1_cnt and p3_cnt early: the v71d Phase 1 recv-side
        # drain needs p3_cnt available (recv extent = OPPOSITE side's send
        # count = p3_cnt). Also pre-compute phase3_remote_start for symmetry
        # though it's only used in Phase 3.
        p1_cnt = pl.multiple_of(p1_cnt_ref[0], _TILE_INT32)
        p3_cnt = pl.multiple_of(p3_cnt_ref[0], _TILE_INT32)
        send_start = pl.multiple_of(send_off_ref[0], _TILE_INT32)
        remote_recv_start = pl.multiple_of(recv_off_ref[0], _TILE_INT32)
        phase3_remote_start = pl.multiple_of(phase3_remote_off_ref[0], _TILE_INT32)

        # === v71d split-DMA: separate inbound DMA descriptor ===
        # The wait_recv() extent is determined by the descriptor's slice, not
        # by the recv_sem signals it drains. So we build TWO descriptors:
        #   d_send: my outbound (extent p1_cnt) -- start() + wait_send()
        #   d_recv: peer's send to me (extent p3_cnt = OPPOSITE side's
        #           send count) -- wait_recv() ONLY (no start, peer issues it).
        # The DeviceID on d_recv is my_src (who is sending TO me); src/dst
        # refs describe the slice on MY side that gets drained when the peer's
        # signals arrive.
        d_send = pltpu.make_async_remote_copy(
            src_ref=sorted_row_ref.at[pl.ds(send_start, p1_cnt)],
            dst_ref=recv_row_buf.at[pl.ds(remote_recv_start, p1_cnt)],
            send_sem=send_sem_id,
            recv_sem=recv_sem_id,
            device_id=(my_dst,),
            device_id_type=pltpu.DeviceIdType.MESH,
        )
        d_recv = pltpu.make_async_remote_copy(
            src_ref=sorted_row_ref.at[pl.ds(0, p3_cnt)],
            dst_ref=recv_row_buf.at[pl.ds(0, p3_cnt)],
            send_sem=send_sem_id,
            recv_sem=recv_sem_id,
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

        # Phase 2: local lookup.
        pltpu.sync_copy(recv_row_buf, idx_smem)

        def gather_body(i, __):
            row_idx = idx_smem[i]
            zero_emb_vmem[pl.ds(i, 1), :] = table_vmem[pl.ds(row_idx, 1), :]
            return None

        lax.fori_loop(0, recv_capacity, gather_body, None)
        pltpu.sync_copy(zero_emb_vmem, looked_up_buf)

        # Phase 3: emb a2a back to source. p3_cnt is the ACTUAL count of
        # embeddings I'm responsible for in this wave (= what src DMA'd to
        # me in Phase 1). Each wave writes to a DISJOINT slice of the
        # global accumulator (different phase3_remote_start per wave), so
        # waves don't collide on the aliased ``out_ref`` accumulator.
        # Phase 3 destination peer = my_src (send back to the rank that
        # sent us its IDs in Phase 1). p3_cnt and phase3_remote_start were
        # hoisted earlier so the v71d Phase 1 recv-drain can reference
        # p3_cnt; reuse them here directly.

        # === v71d split-DMA: separate inbound Phase 3 descriptor ===
        # Phase 3 send: my outbound back to my_src (extent p3_cnt rows x D).
        # Phase 3 recv: my_dst sends embeddings for the p1_cnt IDs I sent
        # (extent p1_cnt rows x D = OPPOSITE side's send count).
        d3_send = pltpu.make_async_remote_copy(
            src_ref=looked_up_buf.at[pl.ds(remote_recv_start, p3_cnt), :],
            dst_ref=out_ref.at[pl.ds(phase3_remote_start, p3_cnt), :],
            send_sem=send_sem_emb,
            recv_sem=recv_sem_emb,
            device_id=(my_src,),  # send back to my source
            device_id_type=pltpu.DeviceIdType.MESH,
        )
        d3_recv = pltpu.make_async_remote_copy(
            src_ref=looked_up_buf.at[pl.ds(0, p1_cnt), :],
            dst_ref=out_ref.at[pl.ds(0, p1_cnt), :],
            send_sem=send_sem_emb,
            recv_sem=recv_sem_emb,
            device_id=(my_dst,),  # peer who will send TO me
            device_id_type=pltpu.DeviceIdType.MESH,
        )

        @pl.when(p3_cnt > 0)
        def _phase3_send():
            d3_send.start()
            d3_send.wait_send()  # only wait for OUR SEND completion (p3_cnt * D)

        @pl.when(p1_cnt > 0)
        def _phase3_recv():
            d3_recv.wait_recv()  # drains recv_sem_emb by p1_cnt rows x D

    return kernel


# ---------------------------------------------------------------------------
# F. Per-device driver + public API
# ---------------------------------------------------------------------------


def _ringshift_embedding_lookup_local(
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
    """Per-device ring-shift fused kernel.

    Args:
        local_ids: ``[max_local_ids]`` int32 -- this rank's padded queries.
        local_id_count: scalar int32 -- actual valid count.
        local_embedding_table: ``[V_local, D]`` f32 -- this rank's shard.

    Returns:
        ``[max_local_ids, D]`` f32 embeddings in caller order.
    """
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
                traffic_matrix,
                send_counts_padded,
                world_size=world_size,
                my_rank=my_rank,
            )
        )
    elif splitter == "globalpad":
        p1_cnt, p3_cnt, send_off, recv_off, phase3_remote_off = (
            _split_traffic_ringshift_globalpad(
                traffic_matrix,
                send_counts_padded,
                world_size=world_size,
                my_rank=my_rank,
            )
        )
    else:
        raise ValueError(f"unknown splitter {splitter}")

    # Tight upper bound for send buffer (sum of TILE-padded per-peer chunks):
    # sum_p send_counts_padded[p] <= sum_p send_counts_real[p] + W*(TILE-1)
    # <= max_local_ids + W*TILE.
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

    # JAX-level Python wave loop. Each pallas_call gets a UNIQUE
    # ``collective_id`` so its implicit device-barrier semaphore doesn't
    # collide with the previous (or next) wave's barrier. Each call's
    # ``out_init`` aliases to ``out`` so the accumulator survives across
    # waves; per-wave dst offsets land on disjoint slices.
    #
    # CROSS-RANK EPOCH BARRIER: even with unique collective_ids, successive
    # semaphore-using epochs can skew across ranks across repeated invocations
    # of the same jit -- wave w on rank A may launch before wave w-1 has
    # fully drained on rank B, so rank A's barrier semaphore in epoch w
    # consumes a signal still in flight from epoch w-1. Stitch the waves
    # together with a psum on a real value derived from the previous wave's
    # output, then bind that scalar into the next wave's accumulator via
    # ``optimization_barrier`` (XLA can't statically simplify it away). This
    # forces every rank to finish wave w-1 before any rank launches wave w.
    sorted_emb_padded = jnp.zeros((max_padded, embedding_dim), dtype=dtype)
    sync_dep = jnp.float32(0.0)  # initial -- no dep yet

    for w_py in range(max_waves):
        # Slice this wave's (1,)-shaped scalars from the (W,)-shaped
        # splitter outputs. SMEM specs expect rank-1 arrays.
        p1_cnt_w = lax.dynamic_slice(p1_cnt, (w_py,), (1,))
        p3_cnt_w = lax.dynamic_slice(p3_cnt, (w_py,), (1,))
        send_off_w = lax.dynamic_slice(send_off, (w_py,), (1,))
        recv_off_w = lax.dynamic_slice(recv_off, (w_py,), (1,))
        phase3_remote_off_w = lax.dynamic_slice(phase3_remote_off, (w_py,), (1,))
        wave_idx_w = jnp.array([w_py], dtype=jnp.int32)

        # Bind sync_dep into the accumulator input via optimization_barrier.
        # ``sorted_emb_padded_dep`` is numerically identical to
        # ``sorted_emb_padded`` but XLA treats it as opaquely depending on
        # ``sync_dep``. For w_py=0 this is a no-op (sync_dep is constant 0).
        # For w_py>=1, sync_dep carries the cross-rank psum from the previous
        # wave -- the kernel for wave w cannot launch on any rank until every
        # rank has finished wave w-1.
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
                    pl.BlockSpec(memory_space=HBM),  # sorted_row
                    pl.BlockSpec(memory_space=HBM),  # table
                    pl.BlockSpec(memory_space=SMEM),  # wave_idx
                    pl.BlockSpec(memory_space=SMEM),  # p1_cnt
                    pl.BlockSpec(memory_space=SMEM),  # p3_cnt
                    pl.BlockSpec(memory_space=SMEM),  # send_off
                    pl.BlockSpec(memory_space=SMEM),  # recv_off
                    pl.BlockSpec(memory_space=SMEM),  # phase3_remote_off
                    pl.BlockSpec(memory_space=HBM),  # out_init (alias)
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
                    # NO REGULAR barrier sem -- kernel uses
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


def ringshift_embedding_lookup(
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
    """Ring-shift fused all-TC f32 jagged a2a + sharded embedding lookup.

    The embedding table is row-wise sharded: the row for global id ``i`` lives
    on rank ``i % world_size`` at local offset ``i // world_size``.

    Each rank queries a *padded* batch of ``[max_local_ids]`` IDs; only the
    first ``local_id_count`` entries are valid and contribute to the all-to-all
    traffic. The returned embeddings are in the original input order.

    Inputs are passed as per-rank-stacked tensors with a leading ``world_size``
    axis sharded along ``axis_name`` (the canonical ``shard_map`` convention).
    The kernel sees per-device shapes inside.

    Per-wave per-rank ONE DMA; W waves total. The wave loop is hoisted to a
    JAX-level Python ``for`` over individual ``pl.pallas_call`` invocations,
    each with a unique ``collective_id`` so per-wave barrier semaphores
    don't collide. A cross-rank ``psum + optimization_barrier`` epoch
    barrier between waves prevents skew across ranks.

    Args:
        local_ids: ``[world_size, max_local_ids]`` int32.
        local_id_count: ``[world_size]`` int32 -- valid count per rank.
        local_embedding_table: ``[world_size, V_local, D]`` f32 -- table
            shards stacked along the leading rank axis.
        mesh: 1D JAX mesh whose only axis is ``axis_name``.
        axis_name: name of the mesh axis the leading dim is sharded on.
        recv_capacity: per-wave receive buffer size. With single-slot recv
            layout, only one peer's payload lands per wave; per-wave incoming
            for any rank is bounded by ``max_local_ids + TILE - 1``. Caller
            should pass a TILE-aligned value at least that large. Must also
            be divisible by ``num_subcores * num_lanes`` if the SC path is
            used (not used in this forward kernel).
        interpret: if True, run the Pallas kernel under the JAX interpreter
            (CPU-emulated; for unit tests). NOTE: the kernel uses
            runtime-extent slices ``ref.at[pl.ds(start, runtime_count)]`` that
            ``interpret=True`` cannot lower. Real-TPU only.
        splitter: one of ``"naive"`` or ``"globalpad"``. With single-slot
            recv layout these are functionally equivalent; ``"globalpad"`` is
            the steady-state default.

    Returns:
        ``[world_size, max_local_ids, D]`` f32 embeddings in caller order.
    """
    assert local_embedding_table.dtype == jnp.float32
    world_size = mesh.shape[axis_name]
    P = jax.sharding.PartitionSpec

    def per_device(ids, count, table):
        out = _ringshift_embedding_lookup_local(
            ids[0],
            count[0],
            table[0],
            axis_name=axis_name,
            world_size=world_size,
            recv_capacity=recv_capacity,
            interpret=interpret,
            splitter=splitter,
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
