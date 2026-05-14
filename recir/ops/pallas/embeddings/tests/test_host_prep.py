# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""CPU-runnable host-prep tests for the ringshift embedding lookup kernel.

These tests exercise the pure-JAX host-side prep helpers used by the fused
ringshift kernel:

  * ``_bucketize_and_sort``       -- per-rank ID -> destination bucketing
  * ``_round_up_tile``            -- TILE alignment helper
  * ``_pad_per_peer``             -- sentinel-padded per-peer slice layout
  * ``_unpad_per_peer``           -- inverse of ``_pad_per_peer``
  * ``_derive_offsets``           -- send/recv offset matrices from traffic
  * ``_split_traffic_ringshift_globalpad`` -- ring-shift splitter
  * ``_ringshift_phase3_remote_offset``    -- per-wave remote dst offset

All assertions use small inputs hand-checked against numpy or pure-JAX
references. No ``interpret=True`` is used because the Pallas kernel relies on
runtime-extent slices that ``interpret=True`` cannot lower; the kernel itself
is exercised by the real-TPU correctness tests under
``examples/embedding/``.
"""

from __future__ import annotations

import unittest

import jax.numpy as jnp
import numpy as np
from recir.ops.pallas.embeddings.ringshift import (
    _bucketize_and_sort,
    _derive_offsets,
    _pad_per_peer,
    _ringshift_phase3_remote_offset,
    _round_up_tile,
    _split_traffic_ringshift_globalpad,
    _TILE_INT32,
    _unpad_per_peer,
    ringshift_embedding_lookup_reference,
)


class BucketizeAndSortTest(unittest.TestCase):
    def test_bucketize_routes_by_modulo(self) -> None:
        """IDs route to ``id % world_size``; padding goes to the drop bucket."""
        world_size = 4
        max_local_ids = 8
        local_ids = jnp.array([10, 5, 7, 12, 3, 0, 0, 0], dtype=jnp.int32)
        local_id_count = jnp.array(5, dtype=jnp.int32)

        sorted_row, send_counts, sort_perm, reverse_perm = _bucketize_and_sort(
            local_ids, local_id_count, max_local_ids, world_size
        )

        self.assertEqual(list(np.array(send_counts)), [1, 1, 1, 2])
        # 10 % 4 = 2, 5 % 4 = 1, 7 % 4 = 3, 12 % 4 = 0, 3 % 4 = 3 -- after a
        # stable sort by (dest_rank), the IDs in destination-order are:
        # dest=0: 12 -> local_row 12//4 = 3
        # dest=1: 5  -> local_row  5//4 = 1
        # dest=2: 10 -> local_row 10//4 = 2
        # dest=3: 7  -> local_row  7//4 = 1
        # dest=3: 3  -> local_row  3//4 = 0
        self.assertEqual(list(np.array(sorted_row[:5])), [3, 1, 2, 1, 0])
        np.testing.assert_array_equal(
            np.array(sort_perm)[np.array(reverse_perm)],
            np.arange(max_local_ids, dtype=np.int32),
        )

    def test_bucketize_all_padding_yields_zero_send_counts(self) -> None:
        """When local_id_count == 0, no IDs contribute to send_counts."""
        world_size = 4
        max_local_ids = 8
        local_ids = jnp.zeros(max_local_ids, dtype=jnp.int32)
        local_id_count = jnp.array(0, dtype=jnp.int32)

        _, send_counts, sort_perm, reverse_perm = _bucketize_and_sort(
            local_ids, local_id_count, max_local_ids, world_size
        )

        self.assertEqual(list(np.array(send_counts)), [0, 0, 0, 0])
        # The permutation is still a bijection.
        np.testing.assert_array_equal(
            np.sort(np.array(sort_perm)),
            np.arange(max_local_ids, dtype=np.int32),
        )
        np.testing.assert_array_equal(
            np.array(sort_perm)[np.array(reverse_perm)],
            np.arange(max_local_ids, dtype=np.int32),
        )


class RoundUpTileTest(unittest.TestCase):
    def test_round_up_tile_basic(self) -> None:
        x = jnp.array([0, 1, 127, 128, 129, 256, 257], dtype=jnp.int32)
        out = np.array(_round_up_tile(x, 128))
        self.assertEqual(list(out), [0, 128, 128, 128, 256, 256, 384])

    def test_round_up_tile_with_kernel_tile(self) -> None:
        """Same as above but using the kernel's _TILE_INT32 (= 128)."""
        x = jnp.array([0, _TILE_INT32, _TILE_INT32 + 1], dtype=jnp.int32)
        out = np.array(_round_up_tile(x, _TILE_INT32))
        self.assertEqual(list(out), [0, _TILE_INT32, 2 * _TILE_INT32])


class DeriveOffsetsTest(unittest.TestCase):
    def test_derive_offsets_consistency(self) -> None:
        """``send_offsets[s, t] + traffic[s, t] == send_offsets[s, t+1]``."""
        rng = np.random.RandomState(0)
        world_size = 6
        traffic = jnp.array(
            rng.randint(0, 12, size=(world_size, world_size)), dtype=jnp.int32
        )
        send_offsets, recv_offsets = _derive_offsets(traffic)

        send_np = np.array(send_offsets)
        traffic_np = np.array(traffic)
        for s in range(world_size):
            self.assertEqual(send_np[s, 0], 0)
            for t in range(1, world_size):
                self.assertEqual(
                    send_np[s, t], send_np[s, t - 1] + traffic_np[s, t - 1]
                )

        recv_np = np.array(recv_offsets)
        for t in range(world_size):
            self.assertEqual(recv_np[t, 0], 0)
            for s in range(1, world_size):
                self.assertEqual(
                    recv_np[t, s], recv_np[t, s - 1] + traffic_np[s - 1, t]
                )

    def test_derive_offsets_round_trip(self) -> None:
        """recv_offsets[t, s] should equal send_offsets accumulated by source."""
        traffic = jnp.array(
            [
                [3, 5, 0, 2],
                [1, 4, 6, 0],
                [0, 2, 3, 4],
                [5, 0, 1, 2],
            ],
            dtype=jnp.int32,
        )
        _, recv_offsets = _derive_offsets(traffic)
        recv_np = np.array(recv_offsets)
        # recv_offsets[t, s] = sum_{s' < s} traffic[s', t].
        for t in range(4):
            running = 0
            for s in range(4):
                self.assertEqual(recv_np[t, s], running)
                running += int(traffic[s, t])


class PadUnpadTest(unittest.TestCase):
    def test_pad_per_peer_layout(self) -> None:
        """_pad_per_peer should produce TILE-aligned per-peer slices with zero sentinels."""
        world_size = 4
        max_local_ids = 16
        # Real per-peer counts: skewed.
        send_counts_real = jnp.array([5, 0, 7, 4], dtype=jnp.int32)
        send_counts_padded = _round_up_tile(send_counts_real, 4)  # tile=4 here
        sorted_row = jnp.arange(max_local_ids, dtype=jnp.int32)
        max_padded = max_local_ids + world_size * 4
        padded = _pad_per_peer(
            sorted_row,
            send_counts_real,
            send_counts_padded,
            max_local_ids,
            max_padded,
            world_size,
        )
        # Real counts [5,0,7,4] -> padded counts [8,0,8,4]. Slot layout:
        #   peer 0: padded[0:8)  = [0,1,2,3,4,sent,sent,sent]
        #   peer 1: padded[8:8)  = (empty)
        #   peer 2: padded[8:16) = [5,6,7,8,9,10,11,sent]
        #   peer 3: padded[16:20)= [12,13,14,15]
        padded_np = np.array(padded)
        self.assertEqual(list(padded_np[0:5]), [0, 1, 2, 3, 4])
        self.assertEqual(list(padded_np[5:8]), [0, 0, 0])
        self.assertEqual(list(padded_np[8:15]), [5, 6, 7, 8, 9, 10, 11])
        self.assertEqual(padded_np[15], 0)
        self.assertEqual(list(padded_np[16:20]), [12, 13, 14, 15])

    def test_pad_unpad_round_trip(self) -> None:
        """Pad then un-pad on a synthetic embedding output should round-trip."""
        world_size = 4
        max_local_ids = 16
        send_counts_real = jnp.array([5, 0, 7, 4], dtype=jnp.int32)
        send_counts_padded = _round_up_tile(send_counts_real, 4)
        sorted_row = jnp.arange(max_local_ids, dtype=jnp.int32)
        max_padded = max_local_ids + world_size * 4
        padded = _pad_per_peer(
            sorted_row,
            send_counts_real,
            send_counts_padded,
            max_local_ids,
            max_padded,
            world_size,
        )

        # Treat ``padded`` itself as an embedding column and round-trip.
        d = 3
        emb_padded = jnp.broadcast_to(padded[:, None], (max_padded, d))
        emb_real = _unpad_per_peer(
            emb_padded,
            send_counts_real,
            send_counts_padded,
            max_local_ids,
            max_padded,
            world_size,
        )
        emb_real_np = np.array(emb_real)
        # First 16 valid positions = total real = 5+0+7+4 = 16.
        # Order: peer 0 [0..4], peer 1 [], peer 2 [5..11], peer 3 [12..15].
        expected = np.array(
            [0, 1, 2, 3, 4] + [5, 6, 7, 8, 9, 10, 11] + [12, 13, 14, 15],
            dtype=np.int32,
        )
        np.testing.assert_array_equal(emb_real_np[:, 0], expected)

    def test_pad_per_peer_uses_tile_int32_aligned(self) -> None:
        """With production TILE=128, real counts round up to TILE multiples."""
        send_counts_real = jnp.array([100, 50, 200, 30], dtype=jnp.int32)
        send_counts_padded = _round_up_tile(send_counts_real, _TILE_INT32)
        # Each real count rounds up to next multiple of 128.
        np.testing.assert_array_equal(
            np.array(send_counts_padded),
            np.array([128, 128, 256, 128], dtype=np.int32),
        )


class RingshiftSplitterTest(unittest.TestCase):
    def _hand_traffic(self) -> tuple[int, jnp.ndarray, jnp.ndarray]:
        """Small W=4 traffic matrix shared by several splitter tests."""
        world_size = 4
        # All entries TILE-aligned for kernel-realistic input.
        send_counts_padded = jnp.array(
            [_TILE_INT32 * 2, _TILE_INT32, _TILE_INT32 * 3, _TILE_INT32],
            dtype=jnp.int32,
        )
        # Symmetric-ish traffic matrix: rank r sends to peer (r+w)%W with
        # send_counts_padded as its per-row.
        # Construct as: traffic[s, t] = send_counts_padded[t] (each rank's
        # outgoing to peer t is just send_counts_padded[t] -- arbitrary but
        # internally consistent for splitter testing).
        traffic_matrix = jnp.broadcast_to(
            send_counts_padded[None, :], (world_size, world_size)
        ).astype(jnp.int32)
        return world_size, traffic_matrix, send_counts_padded

    def test_ringshift_globalpad_splits(self) -> None:
        """globalpad splitter outputs per-wave per-rank scalars consistently."""
        world_size, traffic_matrix, send_counts_padded = self._hand_traffic()
        my_rank = jnp.int32(0)  # rank 0's perspective

        p1_cnt, p3_cnt, send_off, recv_off, phase3_remote_off = (
            _split_traffic_ringshift_globalpad(
                traffic_matrix,
                send_counts_padded,
                world_size=world_size,
                my_rank=my_rank,
            )
        )

        # All outputs are per-wave (length W).
        self.assertEqual(p1_cnt.shape, (world_size,))
        self.assertEqual(p3_cnt.shape, (world_size,))
        self.assertEqual(send_off.shape, (world_size,))
        self.assertEqual(recv_off.shape, (world_size,))
        self.assertEqual(phase3_remote_off.shape, (world_size,))

        # recv_off is constant 0 (single-slot recv layout).
        np.testing.assert_array_equal(
            np.array(recv_off), np.zeros(world_size, dtype=np.int32)
        )

        # p1_cnt at wave w on rank 0 = traffic_matrix[0, (0+w)%W] =
        # send_counts_padded[(0+w)%W].
        for w in range(world_size):
            peer = (0 + w) % world_size
            self.assertEqual(
                int(p1_cnt[w]),
                int(send_counts_padded[peer]),
                f"p1_cnt[{w}] mismatch at peer={peer}",
            )

        # p3_cnt at wave w on rank 0 = traffic_matrix[(0-w)%W, 0] =
        # send_counts_padded[0] = TILE*2 (constant by construction).
        for w in range(world_size):
            self.assertEqual(
                int(p3_cnt[w]),
                int(send_counts_padded[0]),
            )

        # send_off at wave w on rank 0 = padded_offsets[(0+w)%W] =
        # cumsum-exclusive of send_counts_padded.
        padded_offsets = np.cumsum(np.array(send_counts_padded)) - np.array(
            send_counts_padded
        )
        for w in range(world_size):
            peer = (0 + w) % world_size
            self.assertEqual(int(send_off[w]), int(padded_offsets[peer]))

    def test_ringshift_globalpad_all_ranks_send_off_partition(self) -> None:
        """Across all ranks, send_off + p1_cnt covers each peer's chunk exactly."""
        world_size, traffic_matrix, send_counts_padded = self._hand_traffic()
        # For each rank, accumulate the (peer, send_start, send_count) tuples.
        # Each (rank, wave) emits one tuple. Across all (rank, wave) on the
        # same rank, every peer appears exactly once and the slice covers
        # [padded_offsets[peer], padded_offsets[peer] + send_counts_padded[peer]).
        padded_offsets = np.cumsum(np.array(send_counts_padded)) - np.array(
            send_counts_padded
        )
        for rank in range(world_size):
            my_rank = jnp.int32(rank)
            p1_cnt, _, send_off, _, _ = _split_traffic_ringshift_globalpad(
                traffic_matrix,
                send_counts_padded,
                world_size=world_size,
                my_rank=my_rank,
            )
            seen_peers = set()
            for w in range(world_size):
                peer = (rank + w) % world_size
                self.assertNotIn(peer, seen_peers)
                seen_peers.add(peer)
                self.assertEqual(int(p1_cnt[w]), int(send_counts_padded[peer]))
                self.assertEqual(int(send_off[w]), int(padded_offsets[peer]))
            self.assertCountEqual(seen_peers, range(world_size))


class RingshiftPhase3RemoteOffsetTest(unittest.TestCase):
    def test_phase3_remote_offset_matches_definition(self) -> None:
        """At wave w on rank r, dst_offset = padded_offsets[src][r],
        where src = (r - w + W) % W.
        """
        world_size = 4
        # Hand traffic matrix; entry [s, t] is what s sends to t.
        traffic_matrix = jnp.array(
            [
                [10, 20, 30, 40],
                [50, 60, 70, 80],
                [11, 22, 33, 44],
                [55, 66, 77, 88],
            ],
            dtype=jnp.int32,
        )
        # padded_offsets_all[r, p] = sum_{i<p} traffic_matrix[r, i]
        traffic_np = np.array(traffic_matrix)
        padded_offsets_all = np.cumsum(traffic_np, axis=1) - traffic_np

        for rank in range(world_size):
            my_rank = jnp.int32(rank)
            phase3_remote_off = _ringshift_phase3_remote_offset(
                traffic_matrix,
                world_size=world_size,
                my_rank=my_rank,
            )
            for w in range(world_size):
                src = (rank - w + world_size) % world_size
                expected = int(padded_offsets_all[src, rank])
                self.assertEqual(
                    int(phase3_remote_off[w]),
                    expected,
                    f"rank={rank} w={w} src={src}",
                )

    def test_phase3_remote_offset_wave0_is_self(self) -> None:
        """Wave 0 corresponds to src == my_rank, so dst = padded_offsets[my_rank, my_rank]."""
        world_size = 5
        traffic_matrix = jnp.arange(world_size * world_size, dtype=jnp.int32).reshape(
            world_size, world_size
        )
        traffic_np = np.array(traffic_matrix)
        padded_offsets_all = np.cumsum(traffic_np, axis=1) - traffic_np

        for rank in range(world_size):
            my_rank = jnp.int32(rank)
            phase3_remote_off = _ringshift_phase3_remote_offset(
                traffic_matrix,
                world_size=world_size,
                my_rank=my_rank,
            )
            self.assertEqual(
                int(phase3_remote_off[0]),
                int(padded_offsets_all[rank, rank]),
            )


class ReferenceTest(unittest.TestCase):
    """Sanity check the numpy reference simulator used by the real-TPU tests."""

    def test_reference_lookup_matches_manual(self) -> None:
        """Hand-computed expected output vs. reference simulator."""
        world_size = 2
        v_local = 4
        d = 3

        rng = np.random.RandomState(7)
        global_table = rng.randn(2 * v_local, d).astype(np.float32)
        local_table = np.zeros((world_size, v_local, d), dtype=np.float32)
        for gi in range(2 * v_local):
            local_table[gi % world_size, gi // world_size, :] = global_table[gi]

        local_ids = np.array(
            [
                [3, 6, 0],  # rank 0 queries 3,6 (and a padding slot)
                [5, 2, 7],  # rank 1 queries all three
            ],
            dtype=np.int32,
        )
        local_id_count = np.array([2, 3], dtype=np.int32)

        out = ringshift_embedding_lookup_reference(
            local_ids, local_id_count, local_table, world_size
        )

        np.testing.assert_array_equal(out[0, 0], global_table[3])
        np.testing.assert_array_equal(out[0, 1], global_table[6])
        np.testing.assert_array_equal(out[0, 2], np.zeros(d, dtype=np.float32))
        np.testing.assert_array_equal(out[1, 0], global_table[5])
        np.testing.assert_array_equal(out[1, 1], global_table[2])
        np.testing.assert_array_equal(out[1, 2], global_table[7])


if __name__ == "__main__":
    unittest.main()
