# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


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

  Step 1 (host JAX):  bucketize looked_up_ids by destination rank, sort, and
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
                        (c) local table gather. Small shards: bulk HBM -> VMEM
                            table prefetch + row-at-a-time scalar gather into a
                            VMEM emb tile. Large shards (exceeding the VMEM
                            staging budget, see ``_use_dma_gather``): per-row
                            DMA gather straight from the HBM table shard, so
                            on-chip VMEM stays bounded by ``recv_capacity``
                            independent of table size;
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

import functools
import logging
import os
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

logger: logging.Logger = logging.getLogger(__name__)


# Ablation knobs (perf-only; correctness is garbage when set). Read once at
# import; set in the manifest env. Used to attribute the ring-shift per-wave
# cost in an e2e bench:
#   RECIR_RINGSHIFT_SKIP_GATHER=1 -> skip the row-by-row gather (fwd) /
#       scatter-add (bwd) loops, but KEEP the cross-host DMA. Isolates the
#       TensorCore gather/scatter cost.
#   RECIR_RINGSHIFT_SKIP_DMA=1    -> skip the cross-host all-to-all DMAs
#       (phase-1/phase-3 remote copies), but KEEP the gather/scatter. Isolates
#       the data-movement cost. (All ranks skip symmetrically, so no deadlock.)
# Setting both -> measures the barrier + fixed-overhead floor.
#   RECIR_RINGSHIFT_SKIP_PREFETCH=1 -> skip the per-wave 8MB table shard
#       sync_copy(table_ref -> table_vmem) (fwd) / grad-acc prefetch (bwd) in
#       the VMEM-prefetch path, or the per-row HBM->VMEM gather DMA in the
#       DMA-gather path. Isolates the HBM->VMEM table-prefetch/gather cost
#       (gather then reads garbage VMEM -> output invalid, perf-only). KEEP
#       gather/DMA.
_SKIP_GATHER: bool = os.environ.get("RECIR_RINGSHIFT_SKIP_GATHER") == "1"
_SKIP_DMA: bool = os.environ.get("RECIR_RINGSHIFT_SKIP_DMA") == "1"
_SKIP_PREFETCH: bool = os.environ.get("RECIR_RINGSHIFT_SKIP_PREFETCH") == "1"
# RECIR_RINGSHIFT_LOG_LOAD=1 -> rank-0 prints the per-rank ring-shift LOAD
# (valid id count, per-peer send counts, per-wave DMA rows p1, per-wave gather
# rows p3, total DMA bytes) so the E2E vs microbench load can be compared
# directly before attributing per-phase time.
_LOG_LOAD: bool = os.environ.get("RECIR_RINGSHIFT_LOG_LOAD") == "1"


# Pooled-reduction implementation. Read when the public function is traced so
# separate jitted callables in one process can deliberately compile different
# arms. The selected value is then fixed in the compiled program.
_POOLING_IMPL_ENV = "RECIR_RINGSHIFT_POOLING_IMPL"
_POOLING_IMPLS: tuple[str, ...] = ("fused_kernel", "xla")


def _pooling_impl_from_env(pooling: str) -> str:
    """Return ``none``, ``fused_kernel``, or ``xla`` at compile time."""
    if pooling == "none":
        return "none"
    pooling_impl = os.environ.get(_POOLING_IMPL_ENV, "fused_kernel")
    if pooling_impl not in _POOLING_IMPLS:
        raise ValueError(
            f"{_POOLING_IMPL_ENV}={pooling_impl!r} is not one of {_POOLING_IMPLS}"
        )
    return pooling_impl


# --- Table-shard VMEM staging budget (large-hash-size support) -------------
# The fused kernel gathers embedding rows from this rank's table shard
# (``[v_local, D]``). Historically it staged the WHOLE shard into a VMEM
# scratch via one ``sync_copy`` and then random-indexed it row-by-row. That
# bounds VMEM by ``v_local * D * dtype_bytes`` -- i.e. it caps the usable hash
# size (per-shard rows) at ~VMEM/(D*4). For large tables (e.g. 100M rows /
# 128 shards -> 781k rows/shard -> ~800MB f32) this exceeds the ~64MB VMEM and
# the kernel fails to compile (RESOURCE_EXHAUSTED on the ``table_vmem`` scratch).
#
# When the per-shard table exceeds this budget, the kernel instead gathers each
# row directly from the HBM shard via a per-row DMA (correctness-first: one
# ``make_async_copy`` start/wait per row), so on-chip VMEM stays bounded by the
# ``recv_capacity`` tile independent of table size. Small shards keep the faster
# bulk-prefetch + VMEM random-gather path unchanged.
_TABLE_VMEM_BUDGET_BYTES: int = 16 * 1024 * 1024


def _use_dma_gather(v_local: int, embedding_dim: int, dtype) -> bool:
    """Trace-time choice: DMA-gather from HBM (large shard) vs VMEM prefetch.

    ``v_local`` is static at trace time, so this is a pure compile-time Python
    branch (no runtime control flow). Returns True when the per-shard table does
    not fit the VMEM staging budget.
    """
    return v_local * embedding_dim * np.dtype(dtype).itemsize > _TABLE_VMEM_BUDGET_BYTES


# Number of per-row gather DMAs kept in flight at once in the DMA-gather path.
# The gather issues this many async row-copies from the HBM table shard and only
# then waits on them, so up to this many 1KB transfers overlap -- hiding per-row
# DMA round-trip latency instead of paying it serially (the old start()/wait()
# per row). Bounded by the DMA queue depth / semaphore budget; tune for perf.
_GATHER_DMA_BATCH: int = 128


# Pooling (EBC) modes accepted by the public lookup entry points. ``"none"`` is
# the original sequence / EC behavior: one output row per looked-up id.
_POOLING_MODES: tuple[str, ...] = ("none", "sum", "mean")


def _check_pooling(
    pooling: str,
    bag_lengths: jax.Array | np.ndarray | None,
    looked_up_id_count: jax.Array | np.ndarray | None,
) -> None:
    """Validate the pooling mode and the two arguments coupled to it.

    Both invariants are enforced *structurally* -- by which arguments are
    present -- rather than by comparing values, because values are not
    available to check:

      * ``bag_lengths`` is required exactly when pooling is on. Bags without a
        mode would silently return unpooled rows and surface much later as a
        shape mismatch.
      * ``looked_up_id_count`` must be OMITTED when pooling is on, because
        ``bag_lengths.sum(-1)`` already determines it. Accepting both would
        admit a ``sum(bag_lengths) != looked_up_id_count`` disagreement that no
        assertion can catch: under ``jit`` both are tracers, so their values are
        invisible at trace time. Such a disagreement fails silently -- ids past
        the bag total are dropped into the trash segment when the sum is too
        small, and the mean divisor counts padding rows when it is too large.
    """
    if pooling not in _POOLING_MODES:
        raise ValueError(f"pooling must be one of {_POOLING_MODES}, got {pooling!r}")
    if (pooling == "none") != (bag_lengths is None):
        raise ValueError(
            "bag_lengths must be supplied iff pooling != 'none'; got "
            f"pooling={pooling!r}, bag_lengths="
            f"{'None' if bag_lengths is None else 'set'}"
        )
    if pooling != "none" and looked_up_id_count is not None:
        raise ValueError(
            "looked_up_id_count must be None when pooling is enabled; it is "
            "derived from bag_lengths.sum(-1) so the two cannot disagree"
        )
    if pooling == "none" and looked_up_id_count is None:
        raise ValueError("looked_up_id_count is required when pooling == 'none'")


def _ring_device_id(axis_name: str, rank: Any) -> dict[str, Any]:
    """Address a ring peer while retaining coordinates on every other mesh axis."""
    return {axis_name: rank}


# ---------------------------------------------------------------------------
# A. Host-side input preparation (pure JAX, all static shapes)
# ---------------------------------------------------------------------------


def _bucketize_and_sort(
    looked_up_ids: jax.Array,
    looked_up_id_count: jax.Array,
    max_local_ids: int,
    world_size: int,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Bucket local IDs by destination rank (= ``id % world_size``) and sort.

    Padding past ``looked_up_id_count`` is routed to a virtual "drop" bucket so it
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
    is_valid = position < looked_up_id_count

    raw_dest = looked_up_ids % world_size
    # Use ``world_size`` as a sentinel "drop" bucket for padding so that
    # bincount with length=world_size+1 keeps invalid IDs out of send_counts.
    dest_rank = jnp.where(is_valid, raw_dest, world_size).astype(jnp.int32)

    # Stable sort so equal-destination IDs preserve their original order.
    sort_perm = jnp.argsort(dest_rank, stable=True).astype(jnp.int32)
    reverse_perm = jnp.argsort(sort_perm, stable=True).astype(jnp.int32)

    sorted_ids = looked_up_ids[sort_perm]
    sorted_local_row = (sorted_ids // world_size).astype(jnp.int32)

    bins = jnp.bincount(dest_rank, length=world_size + 1)
    send_counts = bins[:world_size].astype(jnp.int32)

    return sorted_local_row, send_counts, sort_perm, reverse_perm


def _dedup_grads_by_id(
    looked_up_ids: jax.Array,
    looked_up_id_count: jax.Array,
    grad_out: jax.Array,
    max_local_ids: int,
    world_size: int,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Sum this device's gradients per unique looked-up id (backward scatter prep).

    Backward wants, for every embedding row, the SUM of the gradients of all the
    lookups that hit it. This device may have looked up the same id many times
    (hot rows), so ``looked_up_ids`` can contain duplicates. This function collapses
    those duplicates: it groups the lookups by id and adds up their gradient
    vectors, emitting each id once. That makes the rows each destination receives
    unique, which lets the receiver's scatter-add run as batched (parallel) DMAs
    without a read-modify-write hazard.

    Recall a global id splits as ``id = row * world_size + dest`` where
    ``dest = id % world_size`` is the owner device and ``row = id // world_size``
    is the row within that device's shard.

    Args:
      looked_up_ids:      ``[max_local_ids]`` int32 -- global ids this device looked
                      up (padded to a fixed length; can be owned by any device).
      looked_up_id_count: scalar int32 -- number of real entries (rest is padding).
      grad_out:       ``[max_local_ids, D]`` -- ``grad_out[i]`` is the D-dim
                      gradient for lookup ``looked_up_ids[i]`` (aligned 1:1).

    Returns (each destination's entries packed contiguously, in that order):
      compact_row:  ``[max_local_ids]`` int32 -- the unique rows, one per id.
      compact_grad: ``[max_local_ids, D]`` -- the summed gradient for each row.
      send_counts:  ``[world_size]`` int32 -- how many unique ids go to each dest.
                    Its prefix sums say which dest each compact entry belongs to,
                    so ``dest`` need not be stored per entry.

    Worked example (world_size=4, max_local_ids=6, D=1, count=5; last slot padding):
      looked_up_ids  = [ 5,  5,  7, 12,  5,  0]  # id 5 looked up 3x (slots 0,1,4)
      grad_out (D=1) = [ 1,  2,  3,  4,  5,  9]  # slot 5 padding; its grad ignored
        id 5  -> dest 1, row 1,  grad = 1+2+5 = 8   (slots 0,1,4)
        id 7  -> dest 3, row 1,  grad = 3
        id 12 -> dest 0, row 3,  grad = 4
      =>  send_counts  = [1, 1, 0, 1]              (dest0, dest1, dest2, dest3)
          compact_row  = [3, 1, 1, 0, 0, 0]        (dest0 | dest1 | dest3 | unused)
          compact_grad = [4, 8, 3, 0, 0, 0]        (aligned 1:1 with compact_row)
    """
    # --- Step 1: classify each lookup: valid?, owner device, row-in-shard ---
    position = jnp.arange(max_local_ids, dtype=jnp.int32)
    is_valid = position < looked_up_id_count
    # Padding is routed to a sentinel bucket (world_size) so it sorts last and is
    # dropped from send_counts.
    dest = jnp.where(is_valid, looked_up_ids % world_size, world_size).astype(jnp.int32)

    # --- Step 2: sort by (dest, id) so identical ids sit next to each other ---
    # (primary key = dest, secondary key = id; lexsort's LAST arg is primary).
    order = jnp.lexsort((looked_up_ids, dest)).astype(jnp.int32)
    sorted_ids = looked_up_ids[order]
    sorted_valid = is_valid[order]
    # dest and row are elementwise in looked_up_ids, so they commute with the
    # permutation and cost an op here instead of a gather each.
    sorted_dest = jnp.where(sorted_valid, sorted_ids % world_size, world_size).astype(
        jnp.int32
    )
    sorted_row = (sorted_ids // world_size).astype(jnp.int32)
    # Zero out padding grads so they add nothing when we sum groups.
    sorted_grad = jnp.where(sorted_valid[:, None], grad_out[order], 0)

    # --- Step 3: mark the first lookup of each id-group (its "representative") ---
    prev_ids = jnp.concatenate(
        [jnp.full((1,), -1, dtype=looked_up_ids.dtype), sorted_ids[:-1]]
    )
    is_group_start = sorted_valid & (sorted_ids != prev_ids)

    # --- Step 4: sum each id-group's gradient vectors ---
    # group_id counts up by 1 at each group start, so all lookups of one id share
    # a group_id; scatter-add their grads by group_id. Group g's sum lands at slot
    # g, and g is also the slot step 5 packs that group's representative into (the
    # two are the same cumsum), so this buffer IS the packed output -- reading the
    # sums back out and re-scattering the representatives would spend two full
    # [max_local_ids, D] passes rebuilding it.
    group_id = jnp.clip(
        jnp.cumsum(is_group_start.astype(jnp.int32)) - 1, 0, max_local_ids - 1
    )
    compact_grad = (
        jnp.zeros((max_local_ids, grad_out.shape[1]), dtype=grad_out.dtype)
        .at[group_id]
        .add(sorted_grad)
    )

    # --- Step 5: keep only the representatives, packed to the front per dest ---
    # Each representative moves to slot = number of representatives before it;
    # non-representatives are scattered out of bounds (dropped).
    dst = jnp.where(is_group_start, group_id, max_local_ids)  # OOB -> dropped
    compact_row = (
        jnp.zeros((max_local_ids,), jnp.int32).at[dst].set(sorted_row, mode="drop")
    )
    rep_dest = jnp.where(is_group_start, sorted_dest, world_size)
    send_counts = jnp.bincount(rep_dest, length=world_size + 1)[:world_size].astype(
        jnp.int32
    )
    return compact_row, compact_grad, send_counts


def _round_up_tile_static(x: int, tile: int) -> int:
    """Compile-time tile rounding, for sizing static kernel windows."""
    return ((x + tile - 1) // tile) * tile


def _round_up_tile(x: jax.Array, tile: int) -> jax.Array:
    """Round each element of ``x`` UP to the next multiple of ``tile``."""
    return ((x + tile - 1) // tile * tile).astype(jnp.int32)


def _peer_id_by_cumsum(exclusive_offsets: jax.Array, n: int) -> jax.Array:
    """Vectorized replacement for ``clip(searchsorted(off, arange(n), "right")-1, 0, W-1)``.

    ``searchsorted`` over the per-peer offsets lowers to a serialized
    binary-search ``while`` loop on TPU (memory-bound, ~0 FLOPs). Mapping each
    flat position to its owning peer is O(n) and vectorized: scatter-add a 1 at
    each interior peer start, then take an inclusive prefix sum.

    Args:
        exclusive_offsets: ``[world_size]`` int32 -- the EXCLUSIVE per-peer
            prefix sums (``exclusive_offsets[0] == 0``). The interior starts
            ``exclusive_offsets[1:]`` are the boundary markers; empty peers
            (duplicate offsets) accumulate and out-of-range starts are dropped,
            so the result saturates at ``world_size - 1`` -- matching the
            ``clip(..., 0, world_size - 1)`` in the searchsorted form.
        n: static output length.

    Returns:
        ``[n]`` int32 peer id for each position ``0..n-1``.
    """
    delta = (
        jnp.zeros((n,), jnp.int32)
        .at[exclusive_offsets[1:].astype(jnp.int32)]
        .add(1, mode="drop")
    )
    return jnp.cumsum(delta)


def _peer_broadcast(per_peer: jax.Array, starts: jax.Array, n: int) -> jax.Array:
    """:func:`_peer_id_by_cumsum` for an arbitrary per-peer value, not just the id.

    ``per_peer[peer_id]`` gives the same ``[n]`` array but lowers to a lookup
    into a length-``world_size`` table with a length-n index vector, which on
    TPU costs an order of magnitude more than the scan. ``starts`` is the
    position each peer's slice begins at; empty peers and out-of-range starts
    behave as they do there.
    """
    jumps = jnp.diff(per_peer.astype(jnp.int32), prepend=jnp.zeros((1,), jnp.int32))
    delta = (
        jnp.zeros((n,), jnp.int32).at[starts.astype(jnp.int32)].add(jumps, mode="drop")
    )
    return jnp.cumsum(delta)


def _pad_per_peer(
    sorted_row: jax.Array,
    send_counts_real: jax.Array,
    send_counts_padded: jax.Array,
    max_local_ids: int,
    max_padded: int,
    world_size: int,
    pad_value: int = 0,
) -> jax.Array:
    """Insert sentinel rows so each per-peer slice is tile-aligned.

    The kernel does DMA on ``[padded_offset[p], padded_count[p])`` slices; the
    receiver discards the trailing ``padded - real`` rows by indexing only the
    first ``real_count[p]`` entries on the un-permute path. Forward uses sentinel
    ``pad_value=0`` (looking up local row 0 is always defined). Backward uses
    ``pad_value=v_local`` (a reserved "drop" row) so the batched scatter can write
    sentinels to a harmless scratch row without clobbering a real row-0 update.
    """
    real_offsets = jnp.cumsum(send_counts_real) - send_counts_real
    padded_offsets = jnp.cumsum(send_counts_padded) - send_counts_padded
    pos = jnp.arange(max_padded, dtype=jnp.int32)
    local_pad_pos = pos - _peer_broadcast(padded_offsets, padded_offsets, max_padded)
    is_real = local_pad_pos < _peer_broadcast(
        send_counts_real, padded_offsets, max_padded
    )
    src_pos = jnp.clip(
        _peer_broadcast(real_offsets, padded_offsets, max_padded) + local_pad_pos,
        0,
        max_local_ids - 1,
    )
    return jnp.where(is_real, sorted_row[src_pos], pad_value).astype(jnp.int32)


def _pad_per_peer_2d(
    sorted_rows_2d: jax.Array,
    send_counts_real: jax.Array,
    send_counts_padded: jax.Array,
    max_local_ids: int,
    max_padded: int,
    world_size: int,
) -> jax.Array:
    """2D variant of :func:`_pad_per_peer` for f32 row payloads (e.g., grad_out).

    Operates per-row instead of per-int32 scalar: each "row" in
    ``sorted_rows_2d`` is a D-wide embedding gradient. Per-peer padding rows
    are zero-filled (NOT replicated from a sentinel index, because for
    embedding gradients there's no "row 0 is harmless" safety -- a zero
    gradient is the right semantic for slack rows since the receiver scatters
    them but they contribute nothing to the gradient sum).

    Args:
        sorted_rows_2d: ``[max_local_ids, D]`` float -- gradient rows already
            permuted by ``sort_perm`` (so per-peer slices are contiguous).
        send_counts_real: ``[world_size]`` int32 -- real per-peer count.
        send_counts_padded: ``[world_size]`` int32 -- TILE-aligned per-peer
            count.
        max_local_ids: static -- original (un-padded) total count.
        max_padded: static -- total padded length (= sum of
            ``send_counts_padded`` worst case = ``max_local_ids + W * TILE``).
        world_size: static.

    Returns:
        ``[max_padded, D]`` float -- per-peer slices with zero-filled tail
        rows, exactly matching the sentinel layout produced by
        :func:`_pad_per_peer`.
    """
    real_offsets = jnp.cumsum(send_counts_real) - send_counts_real
    padded_offsets = jnp.cumsum(send_counts_padded) - send_counts_padded
    pos = jnp.arange(max_padded, dtype=jnp.int32)
    local_pad_pos = pos - _peer_broadcast(padded_offsets, padded_offsets, max_padded)
    is_real = local_pad_pos < _peer_broadcast(
        send_counts_real, padded_offsets, max_padded
    )
    src_pos = jnp.clip(
        _peer_broadcast(real_offsets, padded_offsets, max_padded) + local_pad_pos,
        0,
        max_local_ids - 1,
    )
    gathered = sorted_rows_2d[src_pos]  # [max_padded, D]
    return jnp.where(is_real[:, None], gathered, 0).astype(sorted_rows_2d.dtype)


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
    local_real_pos = pos - _peer_broadcast(real_offsets, real_offsets, max_local_ids)
    padded_src = jnp.clip(
        _peer_broadcast(padded_offsets, real_offsets, max_local_ids) + local_real_pos,
        0,
        max_padded - 1,
    )
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
    looked_up_ids: jax.Array,
    looked_up_id_count: jax.Array,
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
    max_local_ids = looked_up_ids.shape[1]
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

        # int32: Pallas semaphore_signal rejects int64 device_ids, and
        # lax.axis_index widens to int64 when jax_enable_x64 is on (which
        # prod needs for the 3e9-row table ids). Narrowed here because
        # every device_id below is derived from it.
        my_rank = lax.axis_index(axis_name).astype(jnp.int32)
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
    # pyrefly: ignore [bad-argument-count, bad-return]
    return sharded(looked_up_ids, looked_up_id_count, local_embedding_table)


def ringshift_embedding_lookup_reference(
    looked_up_ids: np.ndarray,
    looked_up_id_count: np.ndarray | None,
    local_embedding_table: np.ndarray,
    world_size: int,
    pooling: str = "none",
    bag_lengths: np.ndarray | None = None,
) -> np.ndarray:
    """Reference numpy implementation that simulates the full distributed flow.

    Takes per-rank inputs as a leading-dim "rank" axis and returns the per-rank
    output embeddings in caller order (matching ``looked_up_ids`` layout).

    Args:
        looked_up_ids: ``[world_size, max_local_ids]`` int32.
        looked_up_id_count: ``[world_size]`` int32 -- required iff
            ``pooling == "none"``; derived from ``bag_lengths`` otherwise.
        local_embedding_table: ``[world_size, V_local, D]`` float -- table
            shards stacked along the rank axis.
        world_size: number of ranks.
        pooling: ``"none"``, ``"sum"`` or ``"mean"`` -- mirrors
            :func:`ringshift_embedding_lookup`.
        bag_lengths: ``[world_size, num_bags]`` int32 -- required iff
            ``pooling != "none"``.

    Returns:
        ``[world_size, max_local_ids, D]`` float embeddings in input order with
        invalid slots zeroed when ``pooling == "none"``; otherwise
        ``[world_size, num_bags, D]`` pooled vectors.
    """
    assert looked_up_ids.shape[0] == world_size
    _check_pooling(pooling, bag_lengths, looked_up_id_count)
    # _check_pooling has already established which of the two is present; the
    # asserts restate that for the type checker, which cannot follow it.
    if pooling == "none":
        assert looked_up_id_count is not None
    else:
        assert bag_lengths is not None
        looked_up_id_count = bag_lengths.sum(axis=1).astype(np.int32)
    max_local_ids = looked_up_ids.shape[1]
    embedding_dim = local_embedding_table.shape[2]
    dtype = local_embedding_table.dtype

    out_per_id = np.zeros((world_size, max_local_ids, embedding_dim), dtype=dtype)

    # For every (rank, valid id) pair, look up the embedding from the rank
    # that owns the row.
    for r in range(world_size):
        # pyrefly: ignore [unsupported-operation]
        n = int(looked_up_id_count[r])
        for i in range(n):
            global_id = int(looked_up_ids[r, i])
            owner_rank = global_id % world_size
            local_row = global_id // world_size
            out_per_id[r, i, :] = local_embedding_table[owner_rank, local_row, :]

    if pooling == "none":
        return out_per_id

    assert bag_lengths is not None  # narrowed by _check_pooling
    num_bags = bag_lengths.shape[1]
    out = np.zeros((world_size, num_bags, embedding_dim), dtype=dtype)
    for r in range(world_size):
        pos = 0
        for b in range(num_bags):
            n = int(bag_lengths[r, b])
            if n > 0:
                total = out_per_id[r, pos : pos + n, :].sum(axis=0)
                out[r, b, :] = total / n if pooling == "mean" else total
            pos += n
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
    # Single-slot recv layout: every wave receives one peer's payload at offset 0
    # of the HBM staging buffer; the kernel consumes it in recv_capacity-sized
    # VMEM chunks. recv_capacity is the chunk-tile size, NOT a per-pair bound; the
    # staging buffer is sized to the worst-case single-pair payload.
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
    # Single-slot recv layout: every wave receives one peer's payload at HBM
    # offset 0 of the staging buffer. The full per-pair payload (bounded by
    # max_local_ids + TILE, the worst-case all-to-one-peer skew) is staged in
    # HBM and consumed by the kernel in recv_capacity-sized VMEM chunks, so the
    # caller's recv_capacity is only the VMEM chunk-tile size, NOT a per-pair
    # payload bound.
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


def _ringshift_phase4_pool(
    *,
    out_ref,  # pyre-ignore[2]
    seg_ref,  # pyre-ignore[2]
    pooled_ref,  # pyre-ignore[2]
    pool_acc_vmem,  # pyre-ignore[2]
    idx_smem,  # pyre-ignore[2]
    emb_xfer_vmem,  # pyre-ignore[2]
    zero_emb_vmem,  # pyre-ignore[2]
    send_start: jax.Array,
    p1_cnt: jax.Array,
    pool_num_chunks: int,
    pool_tile: int,
) -> None:
    """Reduce this wave's freshly-landed rows into their bags, in place.

    Runs at the tail of a wave, right after Phase 3 has drained.
    :func:`_ringshift_phase3_remote_offset` guarantees the peer writes the
    answers back to the slot the question came from, so they sit at
    ``out_ref[send_start : send_start + p1_cnt]`` -- and their bag ids sit at
    the matching offsets of ``seg_ref``. That correspondence is the whole
    reason this can run here instead of after the wave loop.

    Rows are consumed where they landed, in sorted order, and never restored
    to caller order: summation is order-independent, so the labels were
    permuted instead of the rows (:func:`sorted_padded_segment_ids`). That is
    what lets the pooled path skip ``_unpad_per_peer`` and the ``reverse_perm``
    gather entirely.

    Scratch is reused, not added: Phase 2 is done with ``idx_smem``,
    ``emb_xfer_vmem`` and ``zero_emb_vmem`` by the time Phase 3 has drained,
    so the pooled tiles cost no extra VMEM.

    ``pool_tile`` is deliberately NOT ``recv_capacity``. Pallas needs a static
    slice length, so each chunk moves and casts a whole tile whatever the real
    row count is -- and ``recv_capacity`` is sized for a whole rank's payload
    while this only ever handles one wave's share of it, i.e. 1/W of it. At
    W=32 that was a 40x over-copy and cost ~137 us, measured: sweeping the tile
    from 8320 to 2048 moved this arm from 1.66 to 1.53 ms while the arm without
    Phase 4 did not move. The scratch stays ``recv_capacity``-sized and shared
    with Phase 2; only the touched prefix shrinks.
    """
    pltpu.sync_copy(pooled_ref, pool_acc_vmem)

    for c_pool in range(pool_num_chunks):
        pool_base = c_pool * pool_tile  # static
        pool_len = jnp.clip(p1_cnt - pool_base, 0, pool_tile)

        @pl.when(pool_len > 0)
        def _pool_chunk(pool_base=pool_base, pool_len=pool_len):
            # Runtime start, static extent -- the window can run past
            # ``max_padded`` on the last chunk, which is what the caller's
            # ``pool_slack`` rows are reserved for.
            src = send_start + pool_base
            pltpu.sync_copy(
                seg_ref.at[pl.ds(src, pool_tile)], idx_smem.at[pl.ds(0, pool_tile)]
            )
            pltpu.sync_copy(
                out_ref.at[pl.ds(src, pool_tile), :],
                emb_xfer_vmem.at[pl.ds(0, pool_tile), :],
            )
            # Tile-aligned cast up to f32; identity when the exchange dtype is
            # already f32. Accumulating in bf16 would lose small contributions
            # on hot bags. Only the touched prefix, not the whole scratch.
            zero_emb_vmem[pl.ds(0, pool_tile), :] = emb_xfer_vmem[
                pl.ds(0, pool_tile), :
            ].astype(zero_emb_vmem.dtype)

            def _accumulate(i, _):
                # Serial, NOT a batched parallel scatter: unlike the backward
                # path there is no send-side dedup here, and several ids of one
                # bag routinely land in one chunk.
                bag = idx_smem[i]
                pool_acc_vmem[pl.ds(bag, 1), :] = (
                    pool_acc_vmem[pl.ds(bag, 1), :] + zero_emb_vmem[pl.ds(i, 1), :]
                )
                return None

            lax.fori_loop(0, pool_len, _accumulate, None)

    pltpu.sync_copy(pool_acc_vmem, pooled_ref)


def _make_ringshift_single_wave_kernel(
    *,
    world_size: int,
    recv_capacity: int,
    embedding_dim: int,
    v_local: int,
    axis_name: str,
    num_chunks: int,
    use_dma_gather: bool = False,
    pooling: str = "none",
    num_bags: int = 0,
    pool_tile: int = 0,
    pool_num_chunks: int = 0,
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

    Phase 2 (local gather) is CHUNKED: the full per-pair payload is staged in
    the HBM recv buffers (sized to ``num_chunks * recv_capacity``), and the
    table lookup is done ``recv_capacity`` rows at a time through the
    ``recv_capacity``-sized VMEM tile. This bounds the on-chip working set
    independently of the per-pair payload, so skewed traffic where a single
    peer's count exceeds ``recv_capacity`` is handled correctly (the cross-rank
    DMAs and semaphores are unchanged -- they target the full-size HBM buffers).
    """

    pooled = pooling != "none"

    def _body(
        sorted_row_ref,
        table_ref,
        wave_idx_ref,  # SMEM (1,) int32 -- this wave's index w
        p1_cnt_ref,  # SMEM (1,) int32 -- this wave's p1_cnt
        p3_cnt_ref,  # SMEM (1,) int32 -- this wave's p3_cnt
        send_off_ref,  # SMEM (1,) int32 -- this wave's send_off
        recv_off_ref,  # SMEM (1,) int32 -- this wave's recv_off
        phase3_remote_off_ref,  # SMEM (1,) int32 -- this wave's phase3_remote_off
        out_ref,  # output HBM, alias of out_init_ref
        recv_row_buf,
        looked_up_buf,
        zero_int_vmem,
        zero_emb_vmem,
        emb_xfer_vmem,
        table_vmem,
        idx_smem,
        send_sem_id,
        recv_sem_id,
        send_sem_emb,
        recv_sem_emb,
        gather_sems,
        seg_ref=None,  # input HBM [max_padded + slack] int32 -- bag id per slot
        pooled_ref=None,  # output HBM [num_bags+1, D], alias of pooled_init_ref
        pool_acc_vmem=None,  # VMEM staging for the pooled accumulator
    ):
        my_rank = lax.axis_index(axis_name).astype(jnp.int32)
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
                device_id=_ring_device_id(axis_name, jnp.int32(peer)),
                device_id_type=pl.DeviceIdType.MESH,
            )
        pl.semaphore_wait(sem, world_size)

        # THEN initialize scratches (table prefetch, zero the VMEM gather tile).
        # After the barrier so init can't race with it. The HBM recv_row_buf is
        # NOT pre-zeroed: it is a staging buffer sized for the full per-pair
        # payload, and the chunked gather below only reads the valid
        # [0:p3_cnt] region that the Phase-1 DMA populates.
        # THEN initialize scratches. In VMEM-prefetch mode, bulk-copy the whole
        # table shard HBM -> VMEM once (fast random gather below). In DMA-gather
        # mode (large shard that does not fit VMEM) there is no prefetch: each
        # row is DMA'd from the HBM shard on demand in the gather loop.
        if not use_dma_gather and not _SKIP_PREFETCH:
            pltpu.sync_copy(table_ref, table_vmem)
        zero_int_vmem[...] = jnp.zeros_like(zero_int_vmem)
        zero_emb_vmem[...] = jnp.zeros_like(zero_emb_vmem)

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
            device_id=_ring_device_id(axis_name, my_dst),
            device_id_type=pl.DeviceIdType.MESH,
        )
        d_recv = pltpu.make_async_remote_copy(
            src_ref=sorted_row_ref.at[pl.ds(0, p3_cnt)],
            dst_ref=recv_row_buf.at[pl.ds(0, p3_cnt)],
            send_sem=send_sem_id,
            recv_sem=recv_sem_id,
            device_id=_ring_device_id(axis_name, my_src),
            device_id_type=pl.DeviceIdType.MESH,
        )

        @pl.when(p1_cnt > 0)
        def _phase1_send():
            if not _SKIP_DMA:
                d_send.start()
                d_send.wait_send()  # only wait for OUR SEND completion (extent p1_cnt)

        @pl.when(p3_cnt > 0)
        def _phase1_recv():
            if not _SKIP_DMA:
                d_recv.wait_recv()  # drains recv_sem by p3_cnt (peer's send extent)

        # Phase 2: local lookup, CHUNKED through the recv_capacity-sized VMEM
        # tile so the on-chip working set is bounded regardless of p3_cnt.
        # recv_row_buf / looked_up_buf are HBM staging buffers holding the full
        # per-pair payload (staging_capacity = num_chunks * recv_capacity rows);
        # we gather recv_capacity rows at a time at HBM offset c * recv_capacity.
        # The row-at-a-time gather targets an f32 VMEM tile (zero_emb_vmem); a
        # bulk full-tile cast then converts it to emb_xfer_vmem's dtype (bf16
        # when exchanging in bf16 -- bf16 VMEM disallows dynamic single-row
        # stores, so the cast must be a full, aligned tile op) before the
        # sync_copy to the (emb_dtype) HBM staging buffer.

        def _vmem_gather_body(i, __):
            # Prefetch mode: whole shard already staged in table_vmem; random
            # VMEM gather into the f32 tile (on-chip, ~free).
            row_idx = idx_smem[i]
            if _SKIP_DMA:
                row_idx = jnp.clip(row_idx, 0, v_local - 1)
            zero_emb_vmem[pl.ds(i, 1), :] = table_vmem[pl.ds(row_idx, 1), :]
            return None

        def _dma_gather_group(g, chunk_len):
            # DMA-gather mode: issue up to B row-copies from the HBM table shard
            # (async, no wait), then drain them, so up to B 1KB transfers are in
            # flight at once (hides per-row round-trip latency). Uses dynamic
            # fori_loops (NOT a Python unroll) so the Pallas kernel body stays tiny
            # -- unrolling B copies here exploded Mosaic lowering time across the
            # W-wave loop (~12s -> ~4.5min). Each row has a distinct destination
            # slot and its own semaphore, so issuing then waiting is correct.
            B = _GATHER_DMA_BATCH
            base_i = g * B
            n = jnp.minimum(B, chunk_len - base_i)

            def _issue(b, __):
                i = base_i + b
                row_idx = idx_smem[i]
                if _SKIP_DMA:
                    row_idx = jnp.clip(row_idx, 0, v_local - 1)
                pltpu.make_async_copy(
                    src_ref=table_ref.at[pl.ds(row_idx, 1), :],
                    dst_ref=zero_emb_vmem.at[pl.ds(i, 1), :],
                    sem=gather_sems.at[b],
                ).start()
                return None

            lax.fori_loop(0, n, _issue, None)

            def _wait(b, __):
                i = base_i + b
                row_idx = idx_smem[i]
                if _SKIP_DMA:
                    row_idx = jnp.clip(row_idx, 0, v_local - 1)
                pltpu.make_async_copy(
                    src_ref=table_ref.at[pl.ds(row_idx, 1), :],
                    dst_ref=zero_emb_vmem.at[pl.ds(i, 1), :],
                    sem=gather_sems.at[b],
                ).wait()
                return None

            lax.fori_loop(0, n, _wait, None)
            return None

        for c in range(num_chunks):
            base = c * recv_capacity  # static (compile-time) HBM offset, mult of TILE
            chunk_len = jnp.clip(p3_cnt - base, 0, recv_capacity)

            @pl.when(chunk_len > 0)
            def _gather_chunk(base=base, chunk_len=chunk_len):
                pltpu.sync_copy(recv_row_buf.at[pl.ds(base, recv_capacity)], idx_smem)
                if not _SKIP_GATHER:
                    if use_dma_gather:
                        # SKIP_PREFETCH skips the DMA gather entirely (perf-attr:
                        # reads stale VMEM -> invalid output).
                        if not _SKIP_PREFETCH:
                            n_groups = (chunk_len + _GATHER_DMA_BATCH - 1) // (
                                _GATHER_DMA_BATCH
                            )
                            lax.fori_loop(
                                0,
                                n_groups,
                                lambda g, _, cl=chunk_len: _dma_gather_group(g, cl),
                                None,
                            )
                    else:
                        lax.fori_loop(0, chunk_len, _vmem_gather_body, None)
                # Bulk (whole-tile, aligned) cast f32 gather tile -> exchange
                # dtype, then DMA-stage to HBM. Identity when emb_dtype == f32.
                emb_xfer_vmem[...] = zero_emb_vmem[...].astype(emb_xfer_vmem.dtype)
                pltpu.sync_copy(
                    emb_xfer_vmem, looked_up_buf.at[pl.ds(base, recv_capacity)]
                )

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
            device_id=_ring_device_id(axis_name, my_src),  # send back to my source
            device_id_type=pl.DeviceIdType.MESH,
        )
        d3_recv = pltpu.make_async_remote_copy(
            src_ref=looked_up_buf.at[pl.ds(0, p1_cnt), :],
            dst_ref=out_ref.at[pl.ds(0, p1_cnt), :],
            send_sem=send_sem_emb,
            recv_sem=recv_sem_emb,
            device_id=_ring_device_id(axis_name, my_dst),  # peer who sends TO me
            device_id_type=pl.DeviceIdType.MESH,
        )

        @pl.when(p3_cnt > 0)
        def _phase3_send():
            if not _SKIP_DMA:
                d3_send.start()
                d3_send.wait_send()  # only wait for OUR SEND completion (p3_cnt * D)

        @pl.when(p1_cnt > 0)
        def _phase3_recv():
            if not _SKIP_DMA:
                d3_recv.wait_recv()  # drains recv_sem_emb by p1_cnt rows x D

        # ----- Phase 4: per-bag reduction (pooled path) -----
        if pooled:
            _ringshift_phase4_pool(
                out_ref=out_ref,
                seg_ref=seg_ref,
                pooled_ref=pooled_ref,
                pool_acc_vmem=pool_acc_vmem,
                idx_smem=idx_smem,
                emb_xfer_vmem=emb_xfer_vmem,
                zero_emb_vmem=zero_emb_vmem,
                send_start=send_start,
                p1_cnt=p1_cnt,
                pool_num_chunks=pool_num_chunks,
                pool_tile=pool_tile,
            )

    # Two signatures rather than one with placeholder refs: Pallas binds kernel
    # parameters positionally from in_specs -> out_specs -> scratch, so the
    # pooled refs cannot simply be appended. Branching keeps the EC path's
    # ``pallas_call`` byte-for-byte what it was.
    if not pooled:

        def kernel(
            sorted_row_ref,
            table_ref,
            wave_idx_ref,
            p1_cnt_ref,
            p3_cnt_ref,
            send_off_ref,
            recv_off_ref,
            phase3_remote_off_ref,
            out_init_ref,
            out_ref,
            recv_row_buf,
            looked_up_buf,
            *scratch,
        ):
            del out_init_ref  # aliased to out_ref via input_output_aliases
            _body(
                sorted_row_ref,
                table_ref,
                wave_idx_ref,
                p1_cnt_ref,
                p3_cnt_ref,
                send_off_ref,
                recv_off_ref,
                phase3_remote_off_ref,
                out_ref,
                recv_row_buf,
                looked_up_buf,
                *scratch,
            )

    else:

        def kernel(
            sorted_row_ref,
            table_ref,
            wave_idx_ref,
            p1_cnt_ref,
            p3_cnt_ref,
            send_off_ref,
            recv_off_ref,
            phase3_remote_off_ref,
            out_init_ref,
            seg_ref,  # input HBM [max_padded + slack] int32 -- unread until Phase 4
            pooled_init_ref,  # input HBM, aliased to pooled_ref
            out_ref,
            recv_row_buf,
            looked_up_buf,
            pooled_ref,
            zero_int_vmem,
            zero_emb_vmem,
            emb_xfer_vmem,
            table_vmem,
            idx_smem,
            pool_acc_vmem,
            *sems,
        ):
            del out_init_ref, pooled_init_ref  # aliased via input_output_aliases
            _body(
                sorted_row_ref,
                table_ref,
                wave_idx_ref,
                p1_cnt_ref,
                p3_cnt_ref,
                send_off_ref,
                recv_off_ref,
                phase3_remote_off_ref,
                out_ref,
                recv_row_buf,
                looked_up_buf,
                zero_int_vmem,
                zero_emb_vmem,
                emb_xfer_vmem,
                table_vmem,
                idx_smem,
                *sems,
                seg_ref=seg_ref,
                pooled_ref=pooled_ref,
                pool_acc_vmem=pool_acc_vmem,
            )

    return kernel


# ---------------------------------------------------------------------------
# F. Per-device driver + public API
# ---------------------------------------------------------------------------


def _ringshift_embedding_lookup_local(
    looked_up_ids: jax.Array,
    looked_up_id_count: jax.Array,
    local_embedding_table: jax.Array,
    *,
    axis_name: str,
    world_size: int,
    recv_capacity: int,
    interpret: bool,
    splitter: str,
    exchange_bf16: bool = False,
    pooling: str = "none",
    bag_lengths: jax.Array | None = None,
) -> jax.Array:
    """Per-device ring-shift fused kernel.

    Args:
        looked_up_ids: ``[max_local_ids]`` int32 -- this rank's padded queries.
        looked_up_id_count: scalar int32 -- actual valid count.
        local_embedding_table: ``[V_local, D]`` f32 -- this rank's shard.
        exchange_bf16: if True, the embedding all-to-all (Phase-3 DMA) + its HBM
            staging + the output accumulator are bf16 (halves the cross-host ICI
            bytes, the S=4096 bottleneck). The table stays f32; the gather casts
            f32->bf16; the final result is cast back to f32 so the public dtype
            contract is unchanged.

    Returns:
        ``[max_local_ids, D]`` f32 embeddings in caller order.
    """
    assert local_embedding_table.dtype == jnp.float32
    max_local_ids = looked_up_ids.shape[0]
    v_local, embedding_dim = local_embedding_table.shape
    dtype = local_embedding_table.dtype
    # Dtype of the embedding all-to-all + its staging/output buffers. bf16 halves
    # the cross-host ICI traffic; the table (gather source) stays f32.
    emb_dtype = jnp.bfloat16 if exchange_bf16 else dtype
    max_waves = world_size

    sorted_row_real, send_counts_real, sort_perm, reverse_perm = _bucketize_and_sort(
        looked_up_ids, looked_up_id_count, max_local_ids, world_size
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

    if _LOG_LOAD:
        # Print on ALL ranks (include rank id): on GKE TPU, JAX process 0 lands
        # on a random worker pod, so a rank-0-only print may not be on the pod the
        # log-capture dumps. Per-rank lines also expose load imbalance.
        jax.debug.print(
            "[RS_LOAD] rank={r} max_local_ids={mli} valid={v} sendreal_sum={sr} "
            "sendreal_max={srm} p1_sum={p1s} p1_max={p1m} p3_sum={p3s} "
            "p3_max={p3m} dim={d} exch_bytes={eb} gather_rows={gr}",
            r=my_rank,
            mli=max_local_ids,
            v=looked_up_id_count,
            sr=jnp.sum(send_counts_real),
            srm=jnp.max(send_counts_real),
            p1s=jnp.sum(p1_cnt),
            p1m=jnp.max(p1_cnt),
            p3s=jnp.sum(p3_cnt),
            p3m=jnp.max(p3_cnt),
            d=embedding_dim,
            eb=jnp.sum(p1_cnt) * embedding_dim * (2 if exchange_bf16 else 4),
            gr=jnp.sum(p3_cnt),
        )

        # Per-wave p1/p3 vectors (ALL ranks, with rank id, so worker-0-0's log
        # captures them regardless of which pod is process 0). Correlate hot waves
        # (period-8 offsets) with payload: large => data concentration (H2/H3),
        # small/uniform => ICI-topology latency (H1).
        jax.debug.print("[RS_P1CNT] rank={r} {x}", r=my_rank, x=p1_cnt)
        jax.debug.print("[RS_P3CNT] rank={r} {x}", r=my_rank, x=p3_cnt)

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

    # The HBM staging buffers (recv_row_buf / looked_up_buf) must hold a full
    # single-pair payload (<= max_local_ids + TILE <= max_padded). Round up to a
    # multiple of recv_capacity so the chunked gather's recv_capacity-tile reads stay in
    # bounds. recv_capacity is now ONLY the VMEM chunk-tile size (a VMEM/perf
    # knob), no longer a correctness lower bound on the per-pair payload.
    staging_capacity = (
        (max_padded + recv_capacity - 1) // recv_capacity
    ) * recv_capacity
    num_chunks = staging_capacity // recv_capacity

    # Large per-shard tables that would overflow the VMEM ``table_vmem`` scratch
    # are gathered row-by-row from HBM instead (see ``_use_dma_gather``). In that
    # mode ``table_vmem`` is a tiny unused placeholder and the gather uses a
    # dedicated local DMA semaphore (``gather_sem``).
    use_dma_gather = _use_dma_gather(v_local, embedding_dim, dtype)
    table_vmem_rows = _TILE_INT32 if use_dma_gather else v_local
    logger.info(
        "[ringshift.fwd] path=%s v_local=%d D=%d table_shard_bytes=%d budget=%d",
        "dma_gather" if use_dma_gather else "vmem_prefetch",
        v_local,
        embedding_dim,
        v_local * embedding_dim * np.dtype(dtype).itemsize,
        _TABLE_VMEM_BUDGET_BYTES,
    )

    # Pooled path: per-slot bag ids in the kernel's own sorted/padded layout,
    # plus the accumulator Phase 4 will scatter into. The slack mirrors the
    # fixed-width window Phase 4 reads from a runtime offset, which can overrun
    # ``max_padded`` on the last chunk.
    pooled = pooling != "none"
    num_bags = 0 if bag_lengths is None else bag_lengths.shape[0]
    # Phase 4 handles ONE wave's rows, i.e. ~1/W of a rank's payload, so it
    # must not inherit ``recv_capacity`` (sized for the whole rank). A static
    # slice length means every chunk moves and casts a full tile regardless of
    # the real row count, so an oversized tile is paid in full every wave.
    #
    # The floor is empirical: sweeping the tile at W=32 put the optimum near
    # 1-2K, with smaller tiles giving the saving back as chunk-loop overhead.
    # Skew is handled by the chunk loop, not by the tile -- a hot peer simply
    # uses more chunks, and chunks past the data are skipped by ``pl.when``.
    pool_tile = (
        min(
            recv_capacity,
            max(1024, _round_up_tile_static(max_padded // world_size, _TILE_INT32)),
        )
        if pooled
        else 0
    )
    pool_num_chunks = -(-max_padded // pool_tile) if pooled else 0
    pool_slack = pool_tile if pooled else 0
    # Bound unconditionally so the EC path's control flow stays analysable; they
    # are only ever read under ``if pooled``.
    seg_sorted_padded = None
    pooled_acc = None
    if pooled:
        assert bag_lengths is not None  # narrowed by _check_pooling
        seg_sorted_padded = sorted_padded_segment_ids(
            bag_lengths,
            sort_perm,
            send_counts_real,
            send_counts_padded,
            max_local_ids=max_local_ids,
            max_padded=max_padded,
            world_size=world_size,
            slack=pool_slack,
        )
        pooled_acc = jnp.zeros((num_bags + 1, embedding_dim), dtype)

    single_wave_kernel = _make_ringshift_single_wave_kernel(
        world_size=world_size,
        recv_capacity=recv_capacity,
        embedding_dim=embedding_dim,
        v_local=v_local,
        axis_name=axis_name,
        num_chunks=num_chunks,
        use_dma_gather=use_dma_gather,
        pooling=pooling,
        num_bags=num_bags,
        pool_tile=pool_tile,
        pool_num_chunks=pool_num_chunks,
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
    sorted_emb_padded = jnp.zeros(
        (max_padded + pool_slack, embedding_dim), dtype=emb_dtype
    )
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
        # The pooled accumulator joins the SAME optimization_barrier tuple, not
        # a separate one: that is what ties its cross-wave carry to the epoch
        # barrier. Left out, XLA is free to reorder its updates relative to the
        # waves, and the resulting race only shows at large W.
        pooled_acc_dep = None
        if pooled:
            sorted_emb_padded_dep, pooled_acc_dep, _ = jax.lax.optimization_barrier(
                (sorted_emb_padded, pooled_acc, sync_dep)
            )
        else:
            sorted_emb_padded_dep, _ = jax.lax.optimization_barrier(
                (sorted_emb_padded, sync_dep)
            )

        pool_out_shape = (
            [jax.ShapeDtypeStruct((num_bags + 1, embedding_dim), dtype)]
            if pooled
            else []
        )
        pool_in_specs = (
            [
                pl.BlockSpec(memory_space=HBM),  # seg
                pl.BlockSpec(memory_space=HBM),  # pooled_init (alias)
            ]
            if pooled
            else []
        )
        pool_out_specs = (
            [pl.BlockSpec(memory_space=HBM)] if pooled else []  # pooled
        )
        pool_scratch = (
            (pltpu.VMEM((num_bags + 1, embedding_dim), dtype),) if pooled else ()
        )
        pool_operands = (seg_sorted_padded, pooled_acc_dep) if pooled else ()
        # out_init (idx 8) -> out (idx 0); pooled_init (idx 10) -> pooled (idx 3)
        aliases = {8: 0, 10: 3} if pooled else {8: 0}

        results = pl.pallas_call(
            single_wave_kernel,
            out_shape=[
                jax.ShapeDtypeStruct(
                    (max_padded + pool_slack, embedding_dim), emb_dtype
                ),
                jax.ShapeDtypeStruct((staging_capacity,), jnp.int32),
                jax.ShapeDtypeStruct((staging_capacity, embedding_dim), emb_dtype),
            ]
            + pool_out_shape,
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
                ]
                + pool_in_specs,
                out_specs=[
                    pl.BlockSpec(memory_space=HBM),  # out (= aliased out_init)
                    pl.BlockSpec(memory_space=HBM),  # recv_row_buf (HBM scratch)
                    pl.BlockSpec(memory_space=HBM),  # looked_up_buf (HBM scratch)
                ]
                + pool_out_specs,
                scratch_shapes=(
                    pltpu.VMEM((recv_capacity,), jnp.int32),
                    # zero_emb_vmem: f32 gather tile (row-at-a-time dynamic store
                    # needs the f32 dim-0 tile of 1; bf16 would require 16-aligned).
                    pltpu.VMEM((recv_capacity, embedding_dim), dtype),
                    # emb_xfer_vmem: emb_dtype staging tile for the (bf16) DMA out.
                    pltpu.VMEM((recv_capacity, embedding_dim), emb_dtype),
                    # table_vmem: stays f32 (gather source). Full [v_local, D] shard
                    # in prefetch mode; a tiny unused placeholder in DMA-gather mode
                    # (rows gathered directly from the HBM table shard instead).
                    pltpu.VMEM((table_vmem_rows, embedding_dim), dtype),
                    SMEM((recv_capacity,), jnp.int32),
                )
                + pool_scratch
                + (
                    # NO REGULAR barrier sem -- kernel uses
                    # ``pltpu.get_barrier_semaphore()`` (custom barrier),
                    # required by Mosaic when ``collective_id`` is set
                    # on ``CompilerParams``.
                    pltpu.SemaphoreType.DMA,
                    pltpu.SemaphoreType.DMA,
                    pltpu.SemaphoreType.DMA,
                    pltpu.SemaphoreType.DMA,
                    # gather_sems: pool of local HBM->VMEM per-row gather DMAs kept
                    # in flight (DMA-gather mode only; unused in prefetch mode).
                    pltpu.SemaphoreType.DMA((_GATHER_DMA_BATCH,)),
                ),
            ),
            input_output_aliases=aliases,
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
            *pool_operands,
        )
        sorted_emb_padded = results[0]
        if pooled:
            pooled_acc = results[3]

        # Cross-rank epoch barrier: psum a real value derived from this
        # wave's output (so XLA can't statically simplify). One scalar
        # per rank; ``axis_name`` is bound here because we're inside
        # shard_map. Cast to f32 so the sync token dtype is stable across
        # waves regardless of the (possibly bf16) exchange dtype.
        sync_dep = jax.lax.psum(sorted_emb_padded[0, 0].astype(jnp.float32), axis_name)

    if pooled:
        # The pooled path is already DONE: Phase 4 reduced every row as it
        # landed, in the kernel's own sorted order. Neither the un-pad nor the
        # reverse permutation below is reachable from here -- dropping those two
        # full-width [max_local_ids, D] gathers is the payoff of fusing.
        #
        # Phase 4 accumulates SUMS: the divisor cannot be applied per wave,
        # since a bag's ids arrive across several of them. Row ``num_bags`` is
        # the trash bag (per-peer padding) and is sliced off here.
        assert pooled_acc is not None
        assert bag_lengths is not None  # narrowed by _check_pooling
        return apply_pooling_divisor(
            pooled_acc[:num_bags], bag_lengths, pooling=pooling
        )

    sorted_emb = _unpad_per_peer(
        sorted_emb_padded,
        send_counts_real,
        send_counts_padded,
        max_local_ids,
        max_padded,
        world_size,
    )
    # Cast back to the table dtype (f32) so the public output contract is
    # unchanged; only the over-ICI exchange ran in bf16.
    return sorted_emb[reverse_perm].astype(dtype)


# ---------------------------------------------------------------------------
# F1. Pooling (EBC) -- segment-id construction and the pure-JAX reduction
# ---------------------------------------------------------------------------
#
# torchrec's ``EmbeddingBagCollection`` groups ids into *bags* (one per
# sample x feature) and reduces each bag to a single vector.
#
# Where that reduction actually happens has moved. It began as a host-side
# post-pass over the kernel's per-id output, on the reasoning that a bag's ids
# are scattered across all ranks and the owner gathers a row without ever
# learning which bag it came from. ``sorted_padded_segment_ids`` removed that
# obstacle by permuting the bag *labels* into the kernel's own sorted/padded
# layout, so Phase 4 now reduces each row where it lands and the pooled path
# never materializes the per-id array at all.
#
# What lives here after that move:
#   * ``lengths_to_segment_ids`` -- the KJT-lengths to per-slot-label primitive,
#     used by ``sorted_padded_segment_ids`` and ``fan_out_pooled_grad`` as well.
#   * ``apply_pooling_divisor`` -- the ``mean`` step, shared by both reduction
#     paths so they cannot disagree about it.
#   * ``pool_per_rank`` -- the pure-JAX reduction. No longer on the production
#     path; it is the CPU oracle the interpret-mode and torchrec-parity tests
#     compare the fused result against, and it stays because the fused kernel is
#     real-TPU only and therefore cannot serve as its own reference.
#
# Note that pooling is *inside* the ``jax.custom_vjp`` boundary (the op's
# cotangent is per-bag), so ``segment_sum``'s own VJP is no longer what makes
# the backward work -- ``fan_out_pooled_grad`` is, in the backward section
# below.


def lengths_to_segment_ids(bag_lengths: jax.Array, max_local_ids: int) -> jax.Array:
    """Map each id slot to the bag that owns it.

    Padding slots (positions at or past ``sum(bag_lengths)``) map to
    ``num_bags``, a "trash" segment the caller drops after the reduction. That
    keeps the reduction independent of whether the kernel happens to zero its
    padding rows.

    Built as scatter-add + prefix sum rather than ``jnp.searchsorted`` for the
    same reason as :func:`_peer_id_by_cumsum`: searchsorted lowers to a
    serialized binary-search ``while`` loop on TPU. Duplicate offsets (empty
    bags) accumulate, so an empty bag correctly claims zero slots; an offset
    landing exactly at ``max_local_ids`` (no padding) is dropped as out of range.

    Args:
        bag_lengths: ``[num_bags]`` int32 -- per-bag id count.
        max_local_ids: static padded id-buffer length.

    Returns:
        ``[max_local_ids]`` int32 in ``[0, num_bags]``, non-decreasing.

    Example:
        ``bag_lengths=[3, 0, 2]``, ``max_local_ids=8`` gives
        ``[0, 0, 0, 2, 2, 3, 3, 3]`` -- bag 1 is empty, slots 5..7 are padding.
    """
    offsets = jnp.cumsum(bag_lengths).astype(jnp.int32)
    delta = jnp.zeros((max_local_ids,), jnp.int32).at[offsets].add(1, mode="drop")
    return jnp.cumsum(delta).astype(jnp.int32)


def sorted_padded_segment_ids(
    bag_lengths: jax.Array,
    sort_perm: jax.Array,
    send_counts_real: jax.Array,
    send_counts_padded: jax.Array,
    *,
    max_local_ids: int,
    max_padded: int,
    world_size: int,
    slack: int = 0,
) -> jax.Array:
    """Return bag ids aligned with the kernel's sorted, per-peer padded rows.

    The kernel pools rows after they have been grouped by destination rank and
    padded to a tile boundary. This function applies the same layout changes to
    their bag ids::

        caller-order bag ids -> sort_perm -> per-peer padding -> optional slack

    Real rows keep their original bag id. Padding and slack slots use
    ``num_bags``, a trash segment that the caller drops. This lets the fused
    path pool in kernel order without unpadding or restoring the ``[N, D]``
    embedding rows to caller order.

    Args:
        bag_lengths: ``[num_bags]`` lengths of this source rank's bags in
            caller order. Their sum is the number of real local ids; they are
            not sorted by destination rank.
        sort_perm: ``[max_local_ids]`` permutation from
            :func:`_bucketize_and_sort`, where ``sort_perm[sorted_pos]`` is the
            corresponding caller position. It groups id slots by destination
            rank.
        send_counts_real: ``[world_size]`` number of real ids this source rank
            sends to each destination rank.
        send_counts_padded: ``[world_size]`` corresponding per-destination
            counts after rounding each bucket to a tile boundary.
        max_local_ids: Static size of the caller-order id buffer, including id
            padding.
        max_padded: Static size after per-peer padding, before ``slack``.
        world_size: Number of destination-rank buckets.
        slack: Static number of trailing trash slots added for the kernel's
            final fixed-width read. This is a slot count; their value is
            ``num_bags``.

    Returns:
        ``[max_padded + slack]`` int32 in ``[0, num_bags]``.

    Example with ``num_bags=3`` (so trash is ``3``)::

        caller bag ids:      [0, 0, 1, 2, 2, 3]
        sort_perm:           [1, 3, 0, 2, 4, 5]
        sorted bag ids:      [0, 2 | 0, 1, 2, 3]
        real/padded counts:  [2, 3] / [2, 4]
        result:              [0, 2 | 0, 1, 2, 3 | 3, 3, 3, 3]
    """
    num_bags = bag_lengths.shape[0]
    seg = lengths_to_segment_ids(bag_lengths, max_local_ids)
    seg_sorted = seg[sort_perm]
    seg_padded = _pad_per_peer(
        seg_sorted,
        send_counts_real,
        send_counts_padded,
        max_local_ids,
        max_padded,
        world_size,
        pad_value=num_bags,
    )
    if slack:
        seg_padded = jnp.pad(seg_padded, (0, slack), constant_values=num_bags)
    return seg_padded


def fan_out_pooled_grad(
    grad_pooled: jax.Array,
    bag_lengths: jax.Array,
    *,
    max_local_ids: int,
    pooling: str,
) -> jax.Array:
    """Expand a ``[num_bags, D]`` gradient back to one row per id (caller order).

    The VJP of the pooled forward. ``sum`` copies a bag's gradient to every
    member; ``mean`` additionally scales by ``1 / len``. Padding slots point at
    the trash bag, whose gradient row is zero, so they receive nothing.

    Used at the top of the fused path's backward so the existing per-id backward
    driver is reached unchanged (staged plan V1a). Pushing this fan-out further
    down -- folding it into ``_dedup_grads_by_id`` / the ``sort_perm`` permute so
    the ``[max_local_ids, D]`` array is never materialised -- is only worth doing
    if measurement shows the backward is the bottleneck; XLA may already fuse the
    two consecutive gathers.

    Worked example (num_bags=3, bag_lengths=[2, 1, 0], max_local_ids=5, D=1)::

        segment_ids  = [0, 0, 1, 3, 3]     bag 2 is empty so claims no slot;
                                            slots 3,4 are padding -> trash (3)
        grad_pooled  = [[10], [20], [30]]

        grad_padded  = [[10], [20], [30], [0]]      trash row appended
        grad_per_id  = grad_padded[segment_ids]
                     = [[10], [10], [20], [0], [0]]     <- "sum"

        "mean" additionally divides by the owning bag's length:
            divisor              = [2, 1, 1, 1]     clamped, trash appended
            divisor[segment_ids] = [2, 2, 1, 1, 1]
            grad_per_id          = [[5], [5], [20], [0], [0]]

    Three things this shows: a bag's gradient is copied to *every* member, an
    empty bag hands its gradient to nobody (bag 2's 30 appears nowhere), and
    padding slots receive exactly zero rather than bag 0's gradient.
    """
    num_bags, embedding_dim = grad_pooled.shape
    segment_ids = lengths_to_segment_ids(bag_lengths, max_local_ids)
    grad_padded = jnp.concatenate(
        [grad_pooled, jnp.zeros((1, embedding_dim), grad_pooled.dtype)], axis=0
    )
    grad_per_id = grad_padded[segment_ids]
    if pooling == "mean":
        divisor = jnp.maximum(bag_lengths, 1).astype(grad_pooled.dtype)
        divisor = jnp.concatenate([divisor, jnp.ones((1,), grad_pooled.dtype)])
        grad_per_id = grad_per_id / divisor[segment_ids][:, None]
    return grad_per_id


def apply_pooling_divisor(
    pooled_sum: jax.Array,
    bag_lengths: jax.Array,
    *,
    pooling: str,
) -> jax.Array:
    """Turn a per-bag SUM into the requested pooling mode.

    Both reduction paths -- the host-side :func:`pool_per_rank` and the fused
    Phase 4 accumulator -- produce a per-bag sum and must then apply exactly
    this divisor. Keeping it in one place is not tidiness: a path that forgets
    it returns the right *shape* with the wrong *values*, which no shape or
    trace-level assertion can catch.

    The divisor is clamped to 1 so an empty bag yields ``0 / 1`` rather than a
    NaN. That clamp matters twice over -- the unclamped form would also make
    this division's own gradient non-finite at length 0.

    Args:
        pooled_sum: ``[num_bags, D]`` per-bag sums.
        bag_lengths: ``[num_bags]`` int32.
        pooling: ``"sum"`` or ``"mean"``.
    """
    if pooling == "mean":
        return (
            pooled_sum / jnp.maximum(bag_lengths, 1).astype(pooled_sum.dtype)[:, None]
        )
    return pooled_sum


def pool_per_rank(
    per_id_emb: jax.Array,
    bag_lengths: jax.Array,
    *,
    pooling: str,
) -> jax.Array:
    """Reduce one rank's per-id embeddings into per-bag vectors.

    Args:
        per_id_emb: ``[max_local_ids, D]`` -- one row per looked-up id, in
            caller order, padding rows included.
        bag_lengths: ``[num_bags]`` int32.
        pooling: ``"sum"`` or ``"mean"``.

    Returns:
        ``[num_bags, D]``. Empty bags come back as zero rows; for ``"mean"``
        the divisor is clamped to 1 so an empty bag yields ``0 / 1`` rather
        than a NaN.
    """
    num_bags = bag_lengths.shape[0]
    segment_ids = lengths_to_segment_ids(bag_lengths, per_id_emb.shape[0])
    # num_bags + 1 segments: the extra one absorbs padding, then is sliced off.
    # segment_ids is non-decreasing by construction, which lets XLA emit the
    # cheaper sorted-scatter form.
    pooled = jax.ops.segment_sum(
        per_id_emb,
        segment_ids,
        num_segments=num_bags + 1,
        indices_are_sorted=True,
    )[:num_bags]

    return apply_pooling_divisor(pooled, bag_lengths, pooling=pooling)


def ringshift_embedding_lookup(
    looked_up_ids: jax.Array,
    looked_up_id_count: jax.Array | None,
    local_embedding_table: jax.Array,
    *,
    mesh: jax.sharding.Mesh,
    axis_name: str,
    recv_capacity: int,
    interpret: bool = False,
    splitter: str = "globalpad",
    exchange_bf16: bool = False,
    pooling: str = "none",
    bag_lengths: jax.Array | None = None,
) -> jax.Array:
    """Ring-shift fused all-TC f32 jagged a2a + sharded embedding lookup.

    The embedding table is row-wise sharded: the row for global id ``i`` lives
    on rank ``i % world_size`` at local offset ``i // world_size``.

    Each rank queries a *padded* batch of ``[max_local_ids]`` IDs; only the
    first ``looked_up_id_count`` entries are valid and contribute to the all-to-all
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
        looked_up_ids: ``[world_size, max_local_ids]`` int32.
        looked_up_id_count: ``[world_size]`` int32 -- valid count per rank.
            Required when ``pooling == "none"``; must be ``None`` otherwise,
            since ``bag_lengths.sum(-1)`` already determines it. See
            :func:`_check_pooling` for why the redundancy is rejected rather
            than checked.
        local_embedding_table: ``[world_size, V_local, D]`` f32 -- table
            shards stacked along the leading rank axis.
        mesh: 1D JAX mesh whose only axis is ``axis_name``.
        axis_name: name of the mesh axis the leading dim is sharded on.
        recv_capacity: VMEM chunk-tile size for the local gather/scatter, NOT a
            bound on the per-pair payload. The full per-pair payload is staged in
            HBM (sized to ``max_local_ids + W*TILE`` rounded up to a multiple of
            ``recv_capacity``) and the on-chip lookup/scatter-add is processed in
            ``ceil(staging / recv_capacity)`` chunks of ``recv_capacity`` rows, so
            arbitrarily skewed traffic is handled with bounded VMEM. Pass a
            TILE-aligned value; smaller = less VMEM but more chunks (any positive
            multiple of TILE is correct). Must satisfy
            ``recv_capacity * embedding_dim * 4`` (x a few VMEM tiles) <= VMEM.
        interpret: if True, run the Pallas kernel under the JAX interpreter
            (CPU-emulated; for unit tests). NOTE: the kernel uses
            runtime-extent slices ``ref.at[pl.ds(start, runtime_count)]`` that
            ``interpret=True`` cannot lower. Real-TPU only.
        splitter: one of ``"naive"`` or ``"globalpad"``. With single-slot
            recv layout these are functionally equivalent; ``"globalpad"`` is
            the steady-state default.
        exchange_bf16: if True, the cross-host embedding all-to-all (forward) and
            gradient all-to-all (backward) run in bf16, halving the ICI bytes (the
            S=4096 bottleneck). The table stays f32, the table-grad accumulation
            stays f32, and the public inputs/outputs stay f32 -- only the over-ICI
            transfer + staging are bf16. Default False (exact f32 behavior).
        pooling: ``"none"`` (default -- sequence / EC behavior, one row per id),
            ``"sum"`` or ``"mean"`` (EBC behavior, one row per bag). The
            reduction is fused into the wave kernel (Phase 4), so the pooled
            path never materializes the ``[max_local_ids, D]`` per-id array by
            default. Set ``RECIR_RINGSHIFT_POOLING_IMPL=xla`` before tracing to
            select the validation path that reduces the per-id output in XLA.
        bag_lengths: ``[world_size, num_bags]`` int32 -- required iff
            ``pooling != "none"``, rejected otherwise. Ids must be laid out
            bag-major within each rank (the standard KJT jagged layout: bag 0's
            ids first, then bag 1's, ...), and ``bag_lengths.sum(axis=1)`` must
            equal ``looked_up_id_count``.
    Returns:
        ``[world_size, max_local_ids, D]`` f32 embeddings in caller order when
        ``pooling == "none"``; ``[world_size, num_bags, D]`` f32 pooled vectors
        otherwise.

    Differentiability: this function is a thin wrapper around an internal
    :func:`jax.custom_vjp`-registered helper (:func:`_ringshift_with_grad`)
    so ``jax.grad`` / ``jax.value_and_grad`` / ``optax`` compose correctly.
    The backward pass uses the dedicated
    :func:`_ringshift_embedding_lookup_backward_local` driver. The XLA pooling
    path composes ``segment_sum``'s built-in VJP with the per-id VJP; the fused
    path pools inside the custom-VJP boundary and explicitly fans each bag's
    cotangent back out to its member ids.
    """
    _check_pooling(pooling, bag_lengths, looked_up_id_count)
    pooling_impl = _pooling_impl_from_env(pooling)
    if pooling_impl == "none":
        assert looked_up_id_count is not None  # narrowed by _check_pooling
    else:
        # Single source of truth: bag_lengths fully determines the valid id
        # count, so deriving it here makes a mismatch structurally impossible.
        assert bag_lengths is not None  # narrowed by _check_pooling
        looked_up_id_count = jnp.sum(bag_lengths, axis=-1).astype(jnp.int32)

    if pooling_impl in ("none", "xla"):
        # Both paths ask the kernel for per-id rows. The XLA path reduces those
        # rows below; the no-pooling path returns them directly.
        out = _ringshift_with_grad(
            looked_up_ids,
            looked_up_id_count,
            local_embedding_table,
            None,
            mesh,
            axis_name,
            recv_capacity,
            interpret,
            splitter,
            exchange_bf16,
            "none",
        )
        if pooling_impl == "none":
            return out
        assert bag_lengths is not None  # narrowed by _check_pooling
        return jax.vmap(functools.partial(pool_per_rank, pooling=pooling))(
            out, bag_lengths
        )
    elif pooling_impl == "fused_kernel":
        # Pooling happens inside the custom_vjp boundary, so `_bwd` owns the
        # fan-out from each bag gradient to its per-id gradients.
        return _ringshift_with_grad(
            looked_up_ids,
            looked_up_id_count,
            local_embedding_table,
            bag_lengths,
            mesh,
            axis_name,
            recv_capacity,
            interpret,
            splitter,
            exchange_bf16,
            pooling,
        )
    else:
        raise AssertionError(f"unreachable pooling implementation: {pooling_impl!r}")


def _block_to_mod_ids(
    looked_up_ids: jax.Array, world_size: int, v_local: int
) -> jax.Array:
    """Remap block-sharded ids to the MOD-kernel's id space.

    Block layout: id ``i`` lives on rank ``i // v_local`` at row ``i % v_local``.
    The MOD kernel reads ``table[j % W, j // W]`` for query ``j``. To hit the
    block location we send ``j = (i % v_local) * W + (i // v_local)``, giving
    ``j % W == i // v_local`` and ``j // W == i % v_local`` (exact for
    ``i < v_local * W``). Pure elementwise int math; safe on padding/invalid ids
    (they are dropped by ``looked_up_id_count`` in the kernel).
    """
    return (looked_up_ids % v_local) * world_size + (looked_up_ids // v_local)


def ringshift_embedding_lookup_block(
    looked_up_ids: jax.Array,
    looked_up_id_count: jax.Array | None,
    local_embedding_table: jax.Array,
    *,
    mesh: jax.sharding.Mesh,
    axis_name: str,
    recv_capacity: int,
    interpret: bool = False,
    splitter: str = "globalpad",
    exchange_bf16: bool = False,
    pooling: str = "none",
    bag_lengths: jax.Array | None = None,
) -> jax.Array:
    """Block-sharded variant of :func:`ringshift_embedding_lookup`.

    The base kernel is MOD-sharded: global id ``i`` lives on rank ``i % W`` at
    local offset ``i // W``. This wrapper supports the layout recir's GSPMD param
    sharding actually produces -- **contiguous block / FSDP**: a parameter
    ``[vocab, D]`` split by ``np.split(axis=0)`` so rank ``d`` owns the contiguous
    rows ``[d*V_local : (d+1)*V_local)``, i.e. id ``i`` lives on rank
    ``i // V_local`` at local offset ``i % V_local`` (table viewed as
    ``[W, V_local, D]``).

    It achieves this with a pure **id remap** (:func:`_block_to_mod_ids`) and the
    UNCHANGED MOD kernel, so the MOD de-risk stays valid (no kernel/custom_vjp
    edits). The remap is on the non-differentiable ids, so gradients flow to the
    table unchanged and the table gradient lands in the same block
    ``[W, V_local, D]`` layout as the parameter.

    ``pooling`` / ``bag_lengths`` need no special handling here: the remap is
    per-id and order-preserving, so bag membership is unaffected and the pooled
    post-pass runs identically on the MOD path.

    Args/Returns: identical to :func:`ringshift_embedding_lookup`, except
    ``local_embedding_table`` is interpreted as block-sharded (FSDP), the layout
    recir produces for a row-wise-sharded ``nn.Parameter``.
    """
    world_size = local_embedding_table.shape[0]
    v_local = local_embedding_table.shape[1]
    mod_ids = _block_to_mod_ids(looked_up_ids, world_size, v_local)
    return ringshift_embedding_lookup(
        mod_ids,
        looked_up_id_count,
        local_embedding_table,
        mesh=mesh,
        axis_name=axis_name,
        recv_capacity=recv_capacity,
        interpret=interpret,
        splitter=splitter,
        exchange_bf16=exchange_bf16,
        pooling=pooling,
        bag_lengths=bag_lengths,
    )


def ringshift_embedding_lookup_block_local(
    looked_up_ids: jax.Array,
    looked_up_id_count: jax.Array | None,
    local_embedding_table: jax.Array,
    *,
    axis_name: str,
    world_size: int,
    recv_capacity: int,
    interpret: bool = False,
    splitter: str = "globalpad",
    exchange_bf16: bool = False,
    pooling: str = "none",
    bag_lengths: jax.Array | None = None,
) -> jax.Array:
    """Block-sharded lookup inside an already-bound manual ``shard_map`` axis."""
    _check_pooling(pooling, bag_lengths, looked_up_id_count)
    if pooling == "none":
        assert looked_up_id_count is not None  # narrowed by _check_pooling
    else:
        assert bag_lengths is not None  # narrowed by _check_pooling
        looked_up_id_count = jnp.sum(bag_lengths, axis=-1).astype(jnp.int32)
    v_local = local_embedding_table.shape[0]
    mod_ids = _block_to_mod_ids(looked_up_ids, world_size, v_local)
    return _ringshift_embedding_lookup_local(
        mod_ids,
        looked_up_id_count,
        local_embedding_table,
        axis_name=axis_name,
        world_size=world_size,
        recv_capacity=recv_capacity,
        interpret=interpret,
        splitter=splitter,
        exchange_bf16=exchange_bf16,
        pooling=pooling,
        bag_lengths=bag_lengths,
    )


def ringshift_embedding_lookup_block_backward_local(
    grad_out: jax.Array,
    looked_up_ids: jax.Array,
    looked_up_id_count: jax.Array,
    *,
    axis_name: str,
    world_size: int,
    recv_capacity: int,
    v_local: int,
    embedding_dim: int,
    interpret: bool = False,
    splitter: str = "globalpad",
    exchange_bf16: bool = False,
) -> jax.Array:
    """Block-sharded table gradient inside an active manual mesh axis."""
    mod_ids = _block_to_mod_ids(looked_up_ids, world_size, v_local)
    return _ringshift_embedding_lookup_backward_local(
        grad_out,
        mod_ids,
        looked_up_id_count,
        axis_name=axis_name,
        world_size=world_size,
        recv_capacity=recv_capacity,
        splitter=splitter,
        interpret=interpret,
        v_local=v_local,
        embedding_dim=embedding_dim,
        exchange_bf16=exchange_bf16,
    )


# ---------------------------------------------------------------------------
# F. Backward kernel (TC, single-wave) + driver + custom_vjp wiring
# ---------------------------------------------------------------------------
#
# Backward pass for the ringshift kernel. Structurally mirrors the forward
# kernel (single-wave Pallas kernel + Python wave loop in a per-device
# driver, wrapped by shard_map). Differences from forward:
#
#   * Phase 1 is split in TWO sub-DMAs (1a + 1b): per-peer ID payload AND
#     per-peer gradient row payload travel together so the OWNER of each
#     row receives both the destination index and the gradient value to
#     scatter-add into its local table-grad accumulator.
#   * Phase 2 is a local scatter-add (no Phase 3): the OWNER consumes the
#     received (id, grad) pairs and accumulates into ``grad_table_acc`` —
#     a per-rank ``[v_local, D]`` buffer aliased across waves via
#     ``input_output_aliases``.
#   * The cross-rank epoch barrier between waves uses the same
#     ``psum + optimization_barrier`` pattern as forward.
#
# The backward driver is registered as the VJP of the forward via
# :func:`jax.custom_vjp` on a thin internal wrapper
# (:func:`_ringshift_with_grad`). The public API
# :func:`ringshift_embedding_lookup` retains its original keyword-only
# signature; it dispatches to ``_ringshift_with_grad`` so that ``jax.grad``
# / ``jax.value_and_grad`` / ``optax`` compose correctly.


def _make_ringshift_backward_single_wave_kernel(
    *,
    world_size: int,
    recv_capacity: int,
    embedding_dim: int,
    v_local: int,
    axis_name: str,
    num_chunks: int,
    use_dma_gather: bool = False,
):
    """Single-wave backward kernel: ID DMA + grad DMA + local scatter-add.

    Mirror of :func:`_make_ringshift_single_wave_kernel` (forward) but for the
    backward pass. Each wave does THREE steps:

    1. **Phase 1a -- ID DMA** (split-DMA pattern, INT32 row-payload):
       same peers as forward Phase 1 (my outbound to my_dst, my inbound from
       my_src), so the OWNER receives the per-peer row indices it needs to
       scatter into.
    2. **Phase 1b -- gradient DMA** (split-DMA pattern, F32 row-payload):
       SAME peers as Phase 1a, just larger payload (D-wide rows). Distinct
       sems from Phase 1a so the wait_send/wait_recv extents are unambiguous.
    3. **Phase 2 -- local scatter-add**: load received IDs (HBM -> SMEM) and
       received grads (HBM -> VMEM); for each received row i, add
       ``grad_emb_vmem[i, :]`` into ``grad_table_acc_vmem[idx_smem[i], :]``;
       write the accumulator back to HBM (aliased to the input).

    Accumulator carry across waves: ``grad_table_acc`` is aliased via
    ``input_output_aliases`` so each wave reads the running accumulator from
    HBM, performs the local scatter-add into VMEM, and writes the updated
    accumulator back. Cost is O(v_local * D) HBM<->VMEM bytes per wave --
    symmetric with forward's per-wave ``table_vmem`` prefetch.

    Sem plumbing: 4 DMA sems per wave (id-send, id-recv, grad-send,
    grad-recv); ``pltpu.get_barrier_semaphore()`` for the cross-rank
    pre-barrier (required by Mosaic when ``collective_id`` is set on
    ``CompilerParams``).
    """

    def kernel(
        sorted_row_padded_ref,  # input HBM: [max_padded] int32
        grad_emb_padded_ref,  # input HBM: [max_padded, D] f32
        wave_idx_ref,  # SMEM (1,) int32
        p1_cnt_ref,  # SMEM (1,) int32
        p3_cnt_ref,  # SMEM (1,) int32
        send_off_ref,  # SMEM (1,) int32
        recv_off_ref,  # SMEM (1,) int32
        grad_table_acc_init_ref,  # input HBM, ALIASED to output
        grad_table_acc_ref,  # output HBM, alias of init
        row_recv_buf,  # HBM scratch: [recv_capacity] int32
        grad_recv_buf,  # HBM scratch: [recv_capacity, D] f32
        grad_emb_vmem,  # VMEM scratch: [recv_capacity, D] f32 (scatter tile)
        grad_table_acc_vmem,  # VMEM scratch: [v_local, D] f32
        idx_smem,  # SMEM scratch: [recv_capacity] int32
        zero_int_vmem,  # VMEM scratch: [recv_capacity] int32
        grad_xfer_vmem,  # VMEM scratch: [recv_capacity, D] exch_dtype (bf16 stage)
        send_sem_id,  # DMA sem for Phase 1a send
        recv_sem_id,  # DMA sem for Phase 1a recv
        send_sem_emb,  # DMA sem for Phase 1b send
        recv_sem_emb,  # DMA sem for Phase 1b recv
        gather_sems,  # (B,) DMA sem pool for batched HBM<->VMEM RMW (DMA mode)
    ):
        del (
            grad_table_acc_init_ref
        )  # aliased to grad_table_acc_ref via input_output_aliases
        my_rank = lax.axis_index(axis_name).astype(jnp.int32)
        wave_idx = wave_idx_ref[0]

        # All-peer broadcast pre-barrier FIRST. Same ordering as the forward
        # ringshift kernel: barrier-first preserves the cross-rank sync
        # robustness validated through W=128 in the forward path.
        sem = pltpu.get_barrier_semaphore()
        my_src = (my_rank - wave_idx + world_size) % world_size  # who sends to me
        my_dst = (my_rank + wave_idx) % world_size  # who I send to
        for peer in range(world_size):
            pl.semaphore_signal(
                sem,
                1,
                device_id=_ring_device_id(axis_name, jnp.int32(peer)),
                device_id_type=pl.DeviceIdType.MESH,
            )
        pl.semaphore_wait(sem, world_size)

        # THEN initialize scratches. In VMEM-prefetch mode, load the running
        # grad accumulator HBM -> VMEM (per-wave O(v_local * D) carry) for the
        # in-VMEM scatter-add. In DMA mode (large shard) the accumulator stays in
        # HBM and is read-modify-written one row at a time in the scatter loop.
        # The HBM staging buffers row_recv_buf / grad_recv_buf are NOT pre-zeroed:
        # they are sized for the full per-pair payload and the chunked scatter
        # below only reads the valid [0:p3_cnt] region the Phase-1 DMAs populate.
        if not use_dma_gather:
            pltpu.sync_copy(grad_table_acc_ref, grad_table_acc_vmem)

        # Hoist runtime extents (TILE-aligned because both _round_up_tile and
        # the splitter ensure all per-wave per-peer counts/offsets are
        # multiples of _TILE_INT32).
        p1_cnt = pl.multiple_of(p1_cnt_ref[0], _TILE_INT32)
        p3_cnt = pl.multiple_of(p3_cnt_ref[0], _TILE_INT32)
        send_start = pl.multiple_of(send_off_ref[0], _TILE_INT32)
        remote_recv_start = pl.multiple_of(recv_off_ref[0], _TILE_INT32)

        # ----- Phase 1a: ID DMA (split descriptors per direction) -----
        # Same split-descriptor pattern as the v71d forward fix: separate send
        # / recv descriptors with extents matching each side's actual count.
        d1a_send = pltpu.make_async_remote_copy(
            src_ref=sorted_row_padded_ref.at[pl.ds(send_start, p1_cnt)],
            dst_ref=row_recv_buf.at[pl.ds(remote_recv_start, p1_cnt)],
            send_sem=send_sem_id,
            recv_sem=recv_sem_id,
            device_id=_ring_device_id(axis_name, my_dst),
            device_id_type=pl.DeviceIdType.MESH,
        )
        d1a_recv = pltpu.make_async_remote_copy(
            src_ref=sorted_row_padded_ref.at[pl.ds(0, p3_cnt)],
            dst_ref=row_recv_buf.at[pl.ds(0, p3_cnt)],
            send_sem=send_sem_id,
            recv_sem=recv_sem_id,
            device_id=_ring_device_id(axis_name, my_src),
            device_id_type=pl.DeviceIdType.MESH,
        )

        @pl.when(p1_cnt > 0)
        def _phase1a_send():
            if not _SKIP_DMA:
                d1a_send.start()
                d1a_send.wait_send()  # extent p1_cnt

        @pl.when(p3_cnt > 0)
        def _phase1a_recv():
            if not _SKIP_DMA:
                d1a_recv.wait_recv()  # extent p3_cnt = OPPOSITE side's send count

        # ----- Phase 1b: gradient DMA (split descriptors per direction) -----
        # Same peers as Phase 1a, but D-wide row payload and distinct sems.
        d1b_send = pltpu.make_async_remote_copy(
            src_ref=grad_emb_padded_ref.at[pl.ds(send_start, p1_cnt), :],
            dst_ref=grad_recv_buf.at[pl.ds(remote_recv_start, p1_cnt), :],
            send_sem=send_sem_emb,
            recv_sem=recv_sem_emb,
            device_id=_ring_device_id(axis_name, my_dst),
            device_id_type=pl.DeviceIdType.MESH,
        )
        d1b_recv = pltpu.make_async_remote_copy(
            src_ref=grad_emb_padded_ref.at[pl.ds(0, p3_cnt), :],
            dst_ref=grad_recv_buf.at[pl.ds(0, p3_cnt), :],
            send_sem=send_sem_emb,
            recv_sem=recv_sem_emb,
            device_id=_ring_device_id(axis_name, my_src),
            device_id_type=pl.DeviceIdType.MESH,
        )

        @pl.when(p1_cnt > 0)
        def _phase1b_send():
            if not _SKIP_DMA:
                d1b_send.start()
                d1b_send.wait_send()  # extent p1_cnt rows x D

        @pl.when(p3_cnt > 0)
        def _phase1b_recv():
            if not _SKIP_DMA:
                d1b_recv.wait_recv()  # extent p3_cnt rows x D

        # ----- Phase 2: local scatter-add into grad_table_acc_vmem -----
        # CHUNKED through the recv_capacity-sized SMEM/VMEM tiles so on-chip
        # accesses are bounded regardless of p3_cnt. row_recv_buf / grad_recv_buf
        # are HBM staging buffers holding the full p3_cnt payload
        # (staging_capacity = num_chunks * recv_capacity rows); process
        # recv_capacity rows per chunk at HBM offset c * recv_capacity.

        def _vmem_scatter_body(i, _):
            # Prefetch mode: whole grad accumulator in VMEM; serial RMW (correct
            # for duplicates -- each reads the prior write).
            row_idx = idx_smem[i]
            if _SKIP_DMA:
                row_idx = jnp.clip(row_idx, 0, v_local - 1)
            cur = grad_table_acc_vmem[pl.ds(row_idx, 1), :]
            delta = grad_emb_vmem[pl.ds(i, 1), :]
            grad_table_acc_vmem[pl.ds(row_idx, 1), :] = cur + delta
            return None

        def _dma_scatter_group(g, chunk_len):
            # DMA mode: batched read-modify-write of B accumulator rows in HBM.
            # Send-side dedup guarantees the real rows in a wave are DISTINCT, so
            # batching is safe; padding sentinels all target the reserved "drop"
            # row (index v_local, grad 0) whose concurrent +0 RMWs are harmless.
            # Uses dynamic fori_loops (not a Python unroll) to keep Mosaic lowering
            # small, and grad_table_acc_vmem[b] as the per-slot RMW tile.
            B = _GATHER_DMA_BATCH
            base_i = g * B
            n = jnp.minimum(B, chunk_len - base_i)

            def _row(b):
                r = idx_smem[base_i + b]
                if _SKIP_DMA:
                    r = jnp.clip(r, 0, v_local - 1)
                return r

            def _issue_read(b, __):
                pltpu.make_async_copy(
                    src_ref=grad_table_acc_ref.at[pl.ds(_row(b), 1), :],
                    dst_ref=grad_table_acc_vmem.at[pl.ds(b, 1), :],
                    sem=gather_sems.at[b],
                ).start()
                return None

            lax.fori_loop(0, n, _issue_read, None)

            def _wait_read(b, __):
                pltpu.make_async_copy(
                    src_ref=grad_table_acc_ref.at[pl.ds(_row(b), 1), :],
                    dst_ref=grad_table_acc_vmem.at[pl.ds(b, 1), :],
                    sem=gather_sems.at[b],
                ).wait()
                return None

            lax.fori_loop(0, n, _wait_read, None)

            def _add(b, __):
                i = base_i + b
                grad_table_acc_vmem[pl.ds(b, 1), :] = (
                    grad_table_acc_vmem[pl.ds(b, 1), :] + grad_emb_vmem[pl.ds(i, 1), :]
                )
                return None

            lax.fori_loop(0, n, _add, None)

            def _issue_write(b, __):
                pltpu.make_async_copy(
                    src_ref=grad_table_acc_vmem.at[pl.ds(b, 1), :],
                    dst_ref=grad_table_acc_ref.at[pl.ds(_row(b), 1), :],
                    sem=gather_sems.at[b],
                ).start()
                return None

            lax.fori_loop(0, n, _issue_write, None)

            def _wait_write(b, __):
                pltpu.make_async_copy(
                    src_ref=grad_table_acc_vmem.at[pl.ds(b, 1), :],
                    dst_ref=grad_table_acc_ref.at[pl.ds(_row(b), 1), :],
                    sem=gather_sems.at[b],
                ).wait()
                return None

            lax.fori_loop(0, n, _wait_write, None)
            return None

        for c in range(num_chunks):
            base = c * recv_capacity  # static (compile-time) HBM offset, mult of TILE
            chunk_len = jnp.clip(p3_cnt - base, 0, recv_capacity)

            @pl.when(chunk_len > 0)
            def _scatter_chunk(base=base, chunk_len=chunk_len):
                pltpu.sync_copy(row_recv_buf.at[pl.ds(base, recv_capacity)], idx_smem)
                # DMA-stage the (bf16) grad chunk into the emb_dtype staging tile,
                # then bulk (whole-tile, aligned) cast up to the f32 scatter tile.
                # Identity copy when exch_dtype == f32.
                pltpu.sync_copy(
                    grad_recv_buf.at[pl.ds(base, recv_capacity), :], grad_xfer_vmem
                )
                grad_emb_vmem[...] = grad_xfer_vmem[...].astype(grad_emb_vmem.dtype)
                if not _SKIP_GATHER:
                    if use_dma_gather:
                        n_groups = (chunk_len + _GATHER_DMA_BATCH - 1) // (
                            _GATHER_DMA_BATCH
                        )
                        lax.fori_loop(
                            0,
                            n_groups,
                            lambda g, _, cl=chunk_len: _dma_scatter_group(g, cl),
                            None,
                        )
                    else:
                        lax.fori_loop(0, chunk_len, _vmem_scatter_body, None)

        # Write the accumulator back to HBM (aliased to grad_table_acc_ref). In
        # DMA mode the accumulator was RMW'd directly in HBM, so no bulk store.
        if not use_dma_gather:
            pltpu.sync_copy(grad_table_acc_vmem, grad_table_acc_ref)

    return kernel


def _ringshift_embedding_lookup_backward_local(
    grad_out: jax.Array,
    looked_up_ids: jax.Array,
    looked_up_id_count: jax.Array,
    *,
    axis_name: str,
    world_size: int,
    recv_capacity: int,
    splitter: str,
    interpret: bool,
    v_local: int,
    embedding_dim: int,
    exchange_bf16: bool = False,
) -> jax.Array:
    """Per-device backward driver -- wrapped in shard_map by the caller.

    Mirrors :func:`_ringshift_embedding_lookup_local` structurally:
      * Re-derive bucketize+sort + splitter outputs from ``looked_up_ids`` /
        ``looked_up_id_count`` (cheap; saves residual storage).
      * Permute and pad ``grad_out`` per-peer (mirroring forward's row pad).
      * Python wave loop, per-wave pallas_call with unique
        ``collective_id = world_size + w_py`` to avoid clashing with the
        forward kernel's IDs (0..W-1).
      * Cross-wave epoch barrier via ``psum + optimization_barrier`` on
        ``grad_table_acc[0, 0]``.

    Returns:
        ``[v_local, embedding_dim]`` f32 -- gradient w.r.t. the local table
        shard.
    """
    assert grad_out.dtype == jnp.float32
    max_local_ids = looked_up_ids.shape[0]
    max_waves = world_size
    dtype = grad_out.dtype
    # Accumulate the table grad in f32 (acc_dtype); exchange the per-row grads
    # over ICI in exch_dtype (bf16 halves the cross-host bytes). grad_out comes
    # in f32 and the returned table grad stays f32 (master-weight contract).
    acc_dtype = dtype
    exch_dtype = jnp.bfloat16 if exchange_bf16 else dtype

    # Large tables use the DMA scatter path; dedup grads on the send side so each
    # wave's received rows are unique (safe batched scatter). Small tables keep the
    # serial VMEM-prefetch scatter and the original per-id send order.
    use_dma_gather = _use_dma_gather(v_local, embedding_dim, acc_dtype)

    # --- Re-derive splitter inputs (send layout) ---
    if use_dma_gather:
        # Send-side dedup: sum grads per unique local id so the receiver sees each
        # shard row at most once per wave -> no read-modify-write hazard in the
        # batched scatter. Row padding uses v_local (a reserved "drop" row) so
        # sentinel writes never clobber a real row.
        sorted_row_real, grad_emb_sorted_f32, send_counts_real = _dedup_grads_by_id(
            looked_up_ids, looked_up_id_count, grad_out, max_local_ids, world_size
        )
        grad_emb_sorted = grad_emb_sorted_f32.astype(exch_dtype)
        row_pad_value = v_local
    else:
        sorted_row_real, send_counts_real, sort_perm, _reverse_perm = (
            _bucketize_and_sort(
                looked_up_ids, looked_up_id_count, max_local_ids, world_size
            )
        )
        # Permute grad_out by sort_perm so per-peer slices are contiguous; cast to
        # exch_dtype (bf16) to halve the ICI grad all-to-all.
        grad_emb_sorted = grad_out[sort_perm].astype(exch_dtype)
        row_pad_value = 0

    send_counts_padded = _round_up_tile(send_counts_real, _TILE_INT32)
    traffic_matrix = lax.all_gather(send_counts_padded, axis_name)  # [W, W]
    my_rank = lax.axis_index(axis_name)

    if splitter == "naive":
        p1_cnt, p3_cnt, send_off, recv_off, _phase3_remote_off = (
            _split_traffic_ringshift_naive(
                traffic_matrix,
                send_counts_padded,
                world_size=world_size,
                my_rank=my_rank,
            )
        )
    elif splitter == "globalpad":
        p1_cnt, p3_cnt, send_off, recv_off, _phase3_remote_off = (
            _split_traffic_ringshift_globalpad(
                traffic_matrix,
                send_counts_padded,
                world_size=world_size,
                my_rank=my_rank,
            )
        )
    else:
        raise ValueError(f"unknown splitter {splitter}")

    # --- Pad sorted_row + grad per-peer (sentinel padding to TILE) ---
    max_padded = max_local_ids + world_size * _TILE_INT32
    sorted_row_padded = _pad_per_peer(
        sorted_row_real,
        send_counts_real,
        send_counts_padded,
        max_local_ids,
        max_padded,
        world_size,
        pad_value=row_pad_value,
    )
    # Grad padding is always zero rows (zero gradient = no contribution).
    grad_emb_padded = _pad_per_peer_2d(
        grad_emb_sorted,
        send_counts_real,
        send_counts_padded,
        max_local_ids,
        max_padded,
        world_size,
    )

    HBM = pltpu.MemorySpace.HBM
    SMEM = pltpu.MemorySpace.SMEM

    # HBM staging buffers (row_recv_buf / grad_recv_buf) hold a full single-pair
    # payload; round up to a multiple of recv_capacity so the chunked scatter's
    # recv_capacity-tile reads stay in bounds. recv_capacity is the VMEM chunk-tile size.
    staging_capacity = (
        (max_padded + recv_capacity - 1) // recv_capacity
    ) * recv_capacity
    num_chunks = staging_capacity // recv_capacity

    # Large per-shard tables use the DMA scatter path (grad accumulator stays in
    # HBM, RMW'd in batched row DMAs). Reserve one extra "drop" row (index
    # v_local) that padding sentinels write to harmlessly. In prefetch mode the
    # VMEM scratch is the full shard.
    acc_vmem_rows = _GATHER_DMA_BATCH if use_dma_gather else v_local
    acc_rows = (v_local + _TILE_INT32) if use_dma_gather else v_local
    logger.info(
        "[ringshift.bwd] path=%s v_local=%d D=%d table_shard_bytes=%d budget=%d",
        "dma_gather" if use_dma_gather else "vmem_prefetch",
        v_local,
        embedding_dim,
        v_local * embedding_dim * np.dtype(acc_dtype).itemsize,
        _TABLE_VMEM_BUDGET_BYTES,
    )

    backward_kernel = _make_ringshift_backward_single_wave_kernel(
        world_size=world_size,
        recv_capacity=recv_capacity,
        embedding_dim=embedding_dim,
        v_local=v_local,
        axis_name=axis_name,
        num_chunks=num_chunks,
        use_dma_gather=use_dma_gather,
    )

    # Per the design: backward collective_ids start at world_size to avoid
    # collision with the forward kernel's IDs (0..W-1) when forward and
    # backward live in the same jit.
    COLLECTIVE_ID_OFFSET_BWD = world_size

    # acc_rows == v_local (prefetch) or v_local + TILE (DMA: extra rows are the
    # sentinel "drop" region written by padding). The real table grad is [:v_local].
    grad_table_acc = jnp.zeros((acc_rows, embedding_dim), dtype=dtype)
    sync_dep = jnp.float32(0.0)

    for w_py in range(max_waves):
        p1_cnt_w = lax.dynamic_slice(p1_cnt, (w_py,), (1,))
        p3_cnt_w = lax.dynamic_slice(p3_cnt, (w_py,), (1,))
        send_off_w = lax.dynamic_slice(send_off, (w_py,), (1,))
        recv_off_w = lax.dynamic_slice(recv_off, (w_py,), (1,))
        wave_idx_w = jnp.array([w_py], dtype=jnp.int32)

        # Cross-rank epoch barrier between waves (same pattern as forward).
        grad_table_acc_dep, _ = jax.lax.optimization_barrier((grad_table_acc, sync_dep))

        grad_table_acc, _scratch_row, _scratch_grad = pl.pallas_call(
            backward_kernel,
            out_shape=[
                jax.ShapeDtypeStruct((acc_rows, embedding_dim), acc_dtype),
                jax.ShapeDtypeStruct((staging_capacity,), jnp.int32),
                jax.ShapeDtypeStruct((staging_capacity, embedding_dim), exch_dtype),
            ],
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[
                    pl.BlockSpec(memory_space=HBM),  # sorted_row_padded
                    pl.BlockSpec(memory_space=HBM),  # grad_emb_padded
                    pl.BlockSpec(memory_space=SMEM),  # wave_idx
                    pl.BlockSpec(memory_space=SMEM),  # p1_cnt
                    pl.BlockSpec(memory_space=SMEM),  # p3_cnt
                    pl.BlockSpec(memory_space=SMEM),  # send_off
                    pl.BlockSpec(memory_space=SMEM),  # recv_off
                    pl.BlockSpec(memory_space=HBM),  # grad_table_acc (alias)
                ],
                out_specs=[
                    pl.BlockSpec(memory_space=HBM),  # grad_table_acc out
                    pl.BlockSpec(memory_space=HBM),  # row_recv_buf scratch
                    pl.BlockSpec(memory_space=HBM),  # grad_recv_buf scratch
                ],
                scratch_shapes=(
                    # grad_emb_vmem: f32 scatter tile (row-at-a-time dynamic
                    # access needs the f32 dim-0 tile of 1).
                    pltpu.VMEM(
                        (recv_capacity, embedding_dim), acc_dtype
                    ),  # grad_emb_vmem
                    # grad_table_acc_vmem: ALWAYS f32 (accurate scatter-add accum).
                    # Full [v_local, D] shard in prefetch mode; a tiny per-row RMW
                    # tile in DMA mode (accumulator stays in HBM).
                    pltpu.VMEM(
                        (acc_vmem_rows, embedding_dim), acc_dtype
                    ),  # grad_table_acc_vmem
                    SMEM((recv_capacity,), jnp.int32),  # idx_smem
                    pltpu.VMEM((recv_capacity,), jnp.int32),  # zero_int_vmem
                    # grad_xfer_vmem: exch_dtype (bf16) grad DMA staging tile.
                    pltpu.VMEM(
                        (recv_capacity, embedding_dim), exch_dtype
                    ),  # grad_xfer_vmem
                    pltpu.SemaphoreType.DMA,  # ID send sem
                    pltpu.SemaphoreType.DMA,  # ID recv sem
                    pltpu.SemaphoreType.DMA,  # grad send sem
                    pltpu.SemaphoreType.DMA,  # grad recv sem
                    # gather_sems: pool of local HBM<->VMEM RMW DMAs kept in flight
                    # (DMA scatter mode only; unused in prefetch mode).
                    pltpu.SemaphoreType.DMA((_GATHER_DMA_BATCH,)),
                ),
            ),
            input_output_aliases={7: 0},  # grad_table_acc input -> output
            compiler_params=pltpu.CompilerParams(
                collective_id=COLLECTIVE_ID_OFFSET_BWD + w_py,
            ),
            interpret=interpret,
        )(
            sorted_row_padded,
            grad_emb_padded,
            wave_idx_w,
            p1_cnt_w,
            p3_cnt_w,
            send_off_w,
            recv_off_w,
            grad_table_acc_dep,
        )
        del _scratch_row, _scratch_grad

        # Cross-rank epoch barrier -- psum a real value derived from this
        # wave's accumulator so XLA can't statically reorder. Same pattern as
        # forward's `sync_dep = jax.lax.psum(sorted_emb_padded[0, 0], ...)`.
        sync_dep = jax.lax.psum(grad_table_acc[0, 0], axis_name)

    # Slice off the sentinel drop region (DMA mode) so the returned table grad is
    # exactly [v_local, D].
    return grad_table_acc[:v_local]


def _ringshift_forward_impl(
    looked_up_ids: jax.Array,
    looked_up_id_count: jax.Array,
    local_embedding_table: jax.Array,
    *,
    mesh: jax.sharding.Mesh,
    axis_name: str,
    recv_capacity: int,
    interpret: bool,
    splitter: str,
    exchange_bf16: bool,
    pooling_in_fused_kernel: str,
    bag_lengths_in_fused_kernel: jax.Array | None,
) -> jax.Array:
    """Internal: forward-only impl of the ringshift kernel (no custom_vjp).

    Both the public API and the custom_vjp `_fwd` call this so we don't
    recurse into the custom_vjp wrapper from inside `_fwd`. The body is the
    original public-API body (shard_map of the per-device fused kernel).
    """
    assert local_embedding_table.dtype == jnp.float32
    world_size = mesh.shape[axis_name]
    P = jax.sharding.PartitionSpec

    def per_device(ids, count, table, bags):
        out = _ringshift_embedding_lookup_local(
            ids[0],
            count[0],
            table[0],
            axis_name=axis_name,
            world_size=world_size,
            recv_capacity=recv_capacity,
            interpret=interpret,
            splitter=splitter,
            exchange_bf16=exchange_bf16,
            pooling=pooling_in_fused_kernel,
            bag_lengths=None if bags is None else bags[0],
        )
        return out[None, ...]

    sharded = jax.shard_map(
        per_device,
        mesh=mesh,
        in_specs=(P(axis_name),) * 4,
        out_specs=P(axis_name),
        check_vma=False,
    )
    # Bound to a tuple so the whole call fits on the suppressed line: the
    # pyrefly directive only covers the line immediately after it.
    args = (
        looked_up_ids,
        looked_up_id_count,
        local_embedding_table,
        bag_lengths_in_fused_kernel,
    )
    # pyrefly: ignore [bad-argument-count, bad-return]
    return sharded(*args)


def _ringshift_backward_impl(
    grad_out: jax.Array,
    looked_up_ids: jax.Array,
    looked_up_id_count: jax.Array,
    table_shape: Tuple[int, int],
    *,
    mesh: jax.sharding.Mesh,
    axis_name: str,
    recv_capacity: int,
    interpret: bool,
    splitter: str,
    exchange_bf16: bool,
) -> jax.Array:
    """Internal: backward impl that wraps `_ringshift_embedding_lookup_backward_local`
    in a shard_map. Returns ``[world_size, v_local, D]`` (caller-shape).

    Inputs:
      - grad_out:        [W, max_local_ids, D] f32 sharded along axis_name
      - looked_up_ids:       [W, max_local_ids]    int32 sharded along axis_name
      - looked_up_id_count:  [W]                   int32 sharded along axis_name
      - table_shape:     (v_local, embedding_dim) -- Python tuple from the
                         residual.
    """
    world_size = mesh.shape[axis_name]
    v_local, embedding_dim = table_shape
    P = jax.sharding.PartitionSpec

    def per_device(g, ids, count):
        gt = _ringshift_embedding_lookup_backward_local(
            g[0],
            ids[0],
            count[0],
            axis_name=axis_name,
            world_size=world_size,
            recv_capacity=recv_capacity,
            splitter=splitter,
            interpret=interpret,
            v_local=v_local,
            embedding_dim=embedding_dim,
            exchange_bf16=exchange_bf16,
        )
        return gt[None, ...]

    sharded = jax.shard_map(
        per_device,
        mesh=mesh,
        in_specs=(P(axis_name), P(axis_name), P(axis_name)),
        out_specs=P(axis_name),
        check_vma=False,
    )
    # pyrefly: ignore [bad-argument-count, bad-return]
    return sharded(grad_out, looked_up_ids, looked_up_id_count)


# Inner positional-args function on which `jax.custom_vjp` is registered.
# `mesh`, `axis_name`, `recv_capacity`, `interpret`, `splitter` are static
# (non-tracer) and must be positional for ``nondiff_argnums``. The public
# API :func:`ringshift_embedding_lookup` preserves the existing keyword-only
# signature; this inner function exists only because custom_vjp can't take
# static kwargs directly.
# ``bag_lengths_in_fused_kernel`` sits in a DIFFERENTIABLE position (index 3)
# even though it is int32 and receives a ``None`` gradient: it is a traced
# array, so it cannot be a ``nondiff_argnum``. Adding it therefore shifts every
# static argument right by one, and ``_bwd`` must return a 4-tuple. Those two
# edits plus ``pooling_in_fused_kernel`` in ``_bwd``'s parameter list have to
# move together -- getting any one of them wrong fails silently rather than
# raising.
@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6, 7, 8, 9, 10))
def _ringshift_with_grad(
    looked_up_ids: jax.Array,
    looked_up_id_count: jax.Array,
    local_embedding_table: jax.Array,
    bag_lengths_in_fused_kernel: jax.Array | None,
    mesh: jax.sharding.Mesh,
    axis_name: str,
    recv_capacity: int,
    interpret: bool,
    splitter: str,
    exchange_bf16: bool,
    pooling_in_fused_kernel: str,
) -> jax.Array:
    return _ringshift_forward_impl(
        looked_up_ids,
        looked_up_id_count,
        local_embedding_table,
        mesh=mesh,
        axis_name=axis_name,
        recv_capacity=recv_capacity,
        interpret=interpret,
        splitter=splitter,
        exchange_bf16=exchange_bf16,
        pooling_in_fused_kernel=pooling_in_fused_kernel,
        bag_lengths_in_fused_kernel=bag_lengths_in_fused_kernel,
    )


def _ringshift_with_grad_fwd(
    looked_up_ids,
    looked_up_id_count,
    local_embedding_table,
    bag_lengths_in_fused_kernel,
    mesh,
    axis_name,
    recv_capacity,
    interpret,
    splitter,
    exchange_bf16,
    pooling_in_fused_kernel,
):
    """Forward + save residuals for backward.

    Per design v2 section 3 + the implementation-task note: save ONLY
    (looked_up_ids, looked_up_id_count, per_device_table_shape) -- `_bwd`
    re-derives everything else from these (sort_perm, splitter outputs,
    padding) by calling the same host-side helpers. The redundant work
    is cheap and avoids tracking extra residual tensors.

    `local_embedding_table` here has shape ``[world_size, v_local, D]``
    (the public-API-level shape, before ``shard_map`` shards out the
    leading rank axis). We save the per-device shape ``(v_local, D)`` so
    the backward driver can size its per-rank accumulator without
    re-deriving from the mesh.
    """
    out = _ringshift_forward_impl(
        looked_up_ids,
        looked_up_id_count,
        local_embedding_table,
        mesh=mesh,
        axis_name=axis_name,
        recv_capacity=recv_capacity,
        interpret=interpret,
        splitter=splitter,
        exchange_bf16=exchange_bf16,
        pooling_in_fused_kernel=pooling_in_fused_kernel,
        bag_lengths_in_fused_kernel=bag_lengths_in_fused_kernel,
    )
    # local_embedding_table.shape is the user-facing 3-tuple
    # (world_size, v_local, D); strip the rank axis for the per-device
    # backward accumulator.
    full_shape = local_embedding_table.shape
    if len(full_shape) != 3:
        raise ValueError(
            f"_ringshift_with_grad expects local_embedding_table of shape "
            f"[world_size, v_local, D], got shape {full_shape}"
        )
    per_device_shape = (int(full_shape[1]), int(full_shape[2]))
    # ``bag_lengths_in_fused_kernel`` joins the residuals because the fused
    # pooled backward has to fan the per-bag gradient back out to one row per
    # id before the existing per-id driver can consume it.
    residuals = (
        looked_up_ids,
        looked_up_id_count,
        bag_lengths_in_fused_kernel,
        per_device_shape,
    )
    return out, residuals


def _ringshift_with_grad_bwd(
    mesh,
    axis_name,
    recv_capacity,
    interpret,
    splitter,
    exchange_bf16,
    pooling_in_fused_kernel,
    residuals,
    grad_out,
):
    """Backward: compute grad_local_embedding_table.

    Returns gradients matching the differentiable input signature
    (looked_up_ids, looked_up_id_count, local_embedding_table,
    bag_lengths_in_fused_kernel). The three int32 inputs get None gradients;
    only the table is differentiated.

    Note the parameter order: ``jax.custom_vjp`` passes the ``nondiff_argnums``
    first, in their declared order, then the residuals, then the cotangent. So
    ``pooling_in_fused_kernel`` -- the last static argument -- lands just before
    ``residuals``.
    """
    (
        looked_up_ids,
        looked_up_id_count,
        bag_lengths_in_fused_kernel,
        table_shape,
    ) = residuals

    if pooling_in_fused_kernel != "none":
        # The op now returns [W, num_bags, D], so the cotangent arrives per bag.
        # Expand it to one row per id, in caller order, which is what the
        # existing per-id backward driver expects -- it is then reached
        # completely unchanged (staged plan V1a).
        assert bag_lengths_in_fused_kernel is not None
        max_local_ids = looked_up_ids.shape[-1]
        grad_out = jax.vmap(
            functools.partial(
                fan_out_pooled_grad,
                max_local_ids=max_local_ids,
                pooling=pooling_in_fused_kernel,
            )
        )(grad_out, bag_lengths_in_fused_kernel)
        looked_up_id_count = jnp.sum(bag_lengths_in_fused_kernel, axis=-1).astype(
            jnp.int32
        )

    grad_table = _ringshift_backward_impl(
        grad_out,
        looked_up_ids,
        looked_up_id_count,
        table_shape,
        mesh=mesh,
        axis_name=axis_name,
        recv_capacity=recv_capacity,
        interpret=interpret,
        splitter=splitter,
        exchange_bf16=exchange_bf16,
    )
    # One entry per differentiable input; the arity must match exactly.
    return (None, None, grad_table, None)


_ringshift_with_grad.defvjp(_ringshift_with_grad_fwd, _ringshift_with_grad_bwd)
