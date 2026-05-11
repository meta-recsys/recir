# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Tests for the fused jagged a2a + embedding lookup Pallas kernel.

Three test layers:

  1. Pure-JAX host-side prep tests (``_bucketize_and_sort``,
     ``_derive_offsets``, ``_pad_per_peer``, ``_unpad_per_peer``). Run on CPU,
     no Pallas. Catches arithmetic bugs in the offset / padding logic.

  2. Reference implementation tests (numpy oracle, hand-computed).

  3. End-to-end Pallas kernel tests on real TPU. Cross-checked against:
       * the numpy reference simulator, and
       * a non-fused JAX reference using ``lax.ragged_all_to_all``.
     Parameterized over per-pair distribution (round-robin / skewed) and
     dtype (float32 / bfloat16). To run on a single-host TPU::

         python -m unittest \\
             recir.ops.pallas.embeddings.tests.test_jagged_a2a_lookup

     For multi-host runs on the v7x cluster (recir-v7-8-1), submit via the
     JobSet template under ``/tmp/jagged_a2a_test_jobset.yaml``.
"""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from recir.ops.pallas.embeddings.jagged_a2a_lookup import (
    _TILE_INT32,
    _bucketize_and_sort,
    _derive_offsets,
    _max_waves_for_capacity,
    _pad_per_peer,
    _round_up_tile,
    _split_traffic_into_waves,
    _unpad_per_peer,
    jagged_a2a_embedding_lookup,
    jagged_a2a_embedding_lookup_f32_alltc,
    jagged_a2a_embedding_lookup_reference,
    jagged_a2a_embedding_lookup_reference_jax,
)


def _has_tpu() -> bool:
    """Return True if real TPU devices are available."""
    try:
        return any(d.platform == "tpu" for d in jax.devices())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 1. Host-side prep: bucketize, sort, derive offsets, sentinel padding
# ---------------------------------------------------------------------------


class HostPrepTest(unittest.TestCase):
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
        self.assertEqual(list(np.array(sorted_row[:5])), [3, 1, 2, 1, 0])
        np.testing.assert_array_equal(
            np.array(sort_perm)[np.array(reverse_perm)],
            np.arange(max_local_ids, dtype=np.int32),
        )

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

    def test_round_up_tile(self) -> None:
        x = jnp.array([0, 1, 127, 128, 129, 256, 257], dtype=jnp.int32)
        out = np.array(_round_up_tile(x, 128))
        self.assertEqual(list(out), [0, 128, 128, 128, 256, 256, 384])

    def test_pad_unpad_round_trip(self) -> None:
        """Pad then un-pad on a synthetic embedding output should be identity."""
        world_size = 4
        max_local_ids = 16
        # Real per-peer counts: skewed.
        send_counts_real = jnp.array([5, 0, 7, 4], dtype=jnp.int32)
        send_counts_padded = _round_up_tile(send_counts_real, 4)  # tile=4 here
        sorted_row = jnp.arange(max_local_ids, dtype=jnp.int32)
        max_padded = max_local_ids + world_size * 4
        padded = _pad_per_peer(
            sorted_row, send_counts_real, send_counts_padded,
            max_local_ids, max_padded, world_size,
        )
        # Real counts [5,0,7,4] → padded counts [8,0,8,4]. Slot layout:
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

        # Now treat ``padded`` itself as an embedding column and round-trip.
        d = 3
        emb_padded = jnp.broadcast_to(padded[:, None], (max_padded, d))
        emb_real = _unpad_per_peer(
            emb_padded, send_counts_real, send_counts_padded,
            max_local_ids, max_padded, world_size,
        )
        emb_real_np = np.array(emb_real)
        # First 16 valid positions = total real = 5+0+7+4 = 16.
        # Order: peer 0 [0..4], peer 1 [], peer 2 [5..11], peer 3 [12..15].
        expected = np.array(
            [0, 1, 2, 3, 4] + [5, 6, 7, 8, 9, 10, 11] + [12, 13, 14, 15],
            dtype=np.int32,
        )
        np.testing.assert_array_equal(emb_real_np[:, 0], expected)

    def test_max_waves_single_when_capacity_at_worst_case(self) -> None:
        """``recv_capacity >= max_local_ids + W*TILE`` should give wave_count=1."""
        max_local_ids = 512
        world_size = 4
        worst_case = max_local_ids + world_size * _TILE_INT32
        self.assertEqual(
            _max_waves_for_capacity(max_local_ids, world_size, worst_case), 1
        )
        self.assertEqual(
            _max_waves_for_capacity(max_local_ids, world_size, worst_case * 2), 1
        )

    def test_max_waves_scales_with_smaller_capacity(self) -> None:
        max_local_ids = 1024
        world_size = 4
        worst_case = max_local_ids + world_size * _TILE_INT32  # = 1536
        # Half capacity → 2 waves; quarter capacity → 4 waves.
        self.assertEqual(
            _max_waves_for_capacity(max_local_ids, world_size, worst_case // 2), 2
        )
        self.assertEqual(
            _max_waves_for_capacity(max_local_ids, world_size, worst_case // 4), 4
        )

    def test_split_traffic_into_waves_contract(self) -> None:
        """Per-wave split satisfies sum-conservation, capacity, TILE alignment."""
        world_size = 4
        max_local_ids = 1024
        recv_capacity = 768  # < worst_case = 1536; expect 2 waves.
        max_waves = _max_waves_for_capacity(
            max_local_ids, world_size, recv_capacity
        )
        self.assertEqual(max_waves, 2)

        # Skewed padded traffic, all entries TILE-aligned.
        padded_traffic = jnp.array(
            [
                [256, 128, 0, 256],
                [128, 384, 128, 128],
                [0, 256, 256, 256],
                [128, 128, 128, 384],
            ],
            dtype=jnp.int32,
        )
        per_wave, send_off, recv_off = _split_traffic_into_waves(
            padded_traffic,
            recv_capacity=recv_capacity,
            max_waves=max_waves,
            world_size=world_size,
        )

        per_wave_np = np.array(per_wave)
        # Conservation: sum across waves equals padded traffic.
        np.testing.assert_array_equal(per_wave_np.sum(axis=0), np.array(padded_traffic))
        # Capacity: per-wave per-receiver fan-in <= recv_capacity.
        per_recv_per_wave = per_wave_np.sum(axis=1)  # [max_waves, W]
        self.assertTrue(np.all(per_recv_per_wave <= recv_capacity))
        # TILE alignment for counts and offsets.
        self.assertTrue(np.all(per_wave_np % _TILE_INT32 == 0))
        self.assertTrue(np.all(np.array(send_off) % _TILE_INT32 == 0))
        self.assertTrue(np.all(np.array(recv_off) % _TILE_INT32 == 0))

        # Source offsets: per_wave_send_offsets[w, s, t] = base[s, t]
        # + sum_{w' < w} per_wave[w', s, t]
        # ONLY at non-zero entries. Zero entries get a safe 0 (boundary-OOB
        # avoidance — see _split_traffic_into_waves).
        base_send = np.concatenate(
            [
                np.zeros((world_size, 1), dtype=np.int32),
                np.cumsum(np.array(padded_traffic), axis=1)[:, :-1],
            ],
            axis=1,
        )
        for w in range(max_waves):
            prior_cum = per_wave_np[:w].sum(axis=0)
            expected = base_send + prior_cum
            mask = per_wave_np[w] > 0
            actual = np.array(send_off[w])
            np.testing.assert_array_equal(actual[mask], expected[mask])
            np.testing.assert_array_equal(actual[~mask], 0)

        # Recv offsets: per_wave_recv_offsets[w, t, s] = sum_{s' < s}
        # per_wave[w, s', t]. Equivalent to col-cumsum-exclusive transposed.
        # Zero-count entries get safe 0.
        for w in range(max_waves):
            cum = np.cumsum(per_wave_np[w], axis=0) - per_wave_np[w]
            expected = cum.T
            mask = (per_wave_np[w] > 0).T  # transpose to match recv_off layout
            actual = np.array(recv_off[w])
            np.testing.assert_array_equal(actual[mask], expected[mask])
            np.testing.assert_array_equal(actual[~mask], 0)

    def test_split_traffic_single_wave_when_capacity_sufficient(self) -> None:
        """v0-equivalent capacity gives wave_count=1 with all traffic in wave 0."""
        world_size = 4
        max_local_ids = 256
        recv_capacity = max_local_ids + world_size * _TILE_INT32  # worst case
        max_waves = _max_waves_for_capacity(
            max_local_ids, world_size, recv_capacity
        )
        self.assertEqual(max_waves, 1)

        rng = np.random.RandomState(0)
        padded_traffic = jnp.array(
            (rng.randint(0, 3, size=(world_size, world_size)) * _TILE_INT32).astype(
                np.int32
            )
        )
        per_wave, _, _ = _split_traffic_into_waves(
            padded_traffic,
            recv_capacity=recv_capacity,
            max_waves=max_waves,
            world_size=world_size,
        )
        np.testing.assert_array_equal(np.array(per_wave[0]), np.array(padded_traffic))


# ---------------------------------------------------------------------------
# 2. Reference implementation
# ---------------------------------------------------------------------------


class ReferenceTest(unittest.TestCase):
    def test_reference_lookup_matches_manual(self) -> None:
        """Hand-computed expected output vs. reference simulator."""
        world_size = 2
        v_local = 4
        d = 3
        max_local_ids = 3

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

        out = jagged_a2a_embedding_lookup_reference(
            local_ids, local_id_count, local_table, world_size
        )

        np.testing.assert_array_equal(out[0, 0], global_table[3])
        np.testing.assert_array_equal(out[0, 1], global_table[6])
        np.testing.assert_array_equal(out[0, 2], np.zeros(d, dtype=np.float32))
        np.testing.assert_array_equal(out[1, 0], global_table[5])
        np.testing.assert_array_equal(out[1, 1], global_table[2])
        np.testing.assert_array_equal(out[1, 2], global_table[7])


# ---------------------------------------------------------------------------
# 3. End-to-end Pallas kernel tests (TPU only)
# ---------------------------------------------------------------------------


def _build_local_table(
    world_size: int, v_local: int, embedding_dim: int, dtype: np.dtype, seed: int,
) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randn(world_size, v_local, embedding_dim).astype(dtype)


def _make_round_robin_inputs(
    world_size: int,
    v_local: int,
    max_local_ids: int,
    seed: int,
):
    """Each rank sends ``max_local_ids // world_size`` IDs to every destination
    (deterministic round-robin, naturally tile-aligned).
    """
    assert max_local_ids % world_size == 0
    per_pair = max_local_ids // world_size
    assert per_pair % _TILE_INT32 == 0

    rng = np.random.RandomState(seed)
    global_v = world_size * v_local
    local_ids_np = np.zeros((world_size, max_local_ids), dtype=np.int32)
    local_id_count_np = np.full((world_size,), max_local_ids, dtype=np.int32)
    for r in range(world_size):
        for d in range(world_size):
            valid_for_d = np.arange(d, global_v, world_size, dtype=np.int32)
            chosen = rng.choice(valid_for_d, size=per_pair, replace=True)
            local_ids_np[r, d * per_pair : (d + 1) * per_pair] = chosen
    return local_ids_np, local_id_count_np


def _make_skewed_inputs(
    world_size: int,
    v_local: int,
    max_local_ids: int,
    seed: int,
):
    """Skewed traffic: each rank routes ~75% of valid IDs to one "hot" peer
    (rank r picks hot peer ((r + 1) % W)). Per-pair counts are arbitrary
    integers (NOT tile-aligned) — exercises host-side sentinel padding.
    """
    rng = np.random.RandomState(seed)
    global_v = world_size * v_local
    local_ids_np = np.zeros((world_size, max_local_ids), dtype=np.int32)
    # Use ~80% of max_local_ids as the valid count to leave headroom for the
    # padded send buffer (max_padded = max_local_ids + W * TILE upper bound).
    valid = (max_local_ids * 4) // 5
    local_id_count_np = np.full((world_size,), valid, dtype=np.int32)
    for r in range(world_size):
        hot_peer = (r + 1) % world_size
        ids = []
        for i in range(valid):
            # 75% chance the ID's owner is hot_peer; otherwise uniform.
            if rng.rand() < 0.75:
                owner = hot_peer
            else:
                owner = rng.randint(0, world_size)
            row_in_owner = rng.randint(0, v_local)
            ids.append(owner + world_size * row_in_owner)
        local_ids_np[r, :valid] = np.array(ids, dtype=np.int32)
    return local_ids_np, local_id_count_np


@unittest.skipUnless(_has_tpu(), "Pallas kernel test requires real TPU")
class PallasKernelTest(unittest.TestCase):
    """End-to-end correctness on real TPU.

    Cross-checks the kernel against both:
      * the numpy reference simulator (``jagged_a2a_embedding_lookup_reference``)
      * the JAX ``ragged_all_to_all`` reference
        (``jagged_a2a_embedding_lookup_reference_jax``)

    Parameterized over per-pair distribution (round-robin vs. skewed) and
    output dtype (float32 / bfloat16).
    """

    def _run(
        self,
        *,
        world_size: int,
        distribution: str,
        dtype: np.dtype,
        v_local: int = 256,
        embedding_dim: int = 128,
        max_local_ids: int | None = None,
        recv_capacity: int | None = None,
    ) -> None:
        import sys
        def log(msg):
            print(f"[TEST_TRACE] {msg}", flush=True)
            sys.stderr.write(f"[TEST_TRACE_ERR] {msg}\n")
            sys.stderr.flush()

        log(f"_run START world_size={world_size} dist={distribution} dtype={dtype} max_local_ids={max_local_ids} recv_capacity={recv_capacity}")

        devices = jax.devices()[:world_size]
        if len(devices) < world_size:
            self.skipTest(f"Need {world_size} devices, have {len(devices)}")
        mesh = jax.sharding.Mesh(np.array(devices), ("x",))

        if max_local_ids is None:
            max_local_ids = world_size * _TILE_INT32

        local_table_np = _build_local_table(
            world_size, v_local, embedding_dim, dtype, seed=0
        )
        if distribution == "round_robin":
            local_ids_np, local_id_count_np = _make_round_robin_inputs(
                world_size, v_local, max_local_ids, seed=1
            )
        elif distribution == "skewed":
            local_ids_np, local_id_count_np = _make_skewed_inputs(
                world_size, v_local, max_local_ids, seed=2
            )
        else:
            raise ValueError(distribution)
        log(f"inputs built; local_id_count={local_id_count_np.tolist()}")

        # Numpy reference (always float32 for stability, then cast for compare).
        ref_out_np = jagged_a2a_embedding_lookup_reference(
            local_ids_np, local_id_count_np,
            local_table_np.astype(np.float32), world_size,
        ).astype(dtype)
        log("numpy reference computed")

        # Distribute inputs.
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("x"))
        local_ids = jax.device_put(jnp.array(local_ids_np), sharding)
        local_id_count = jax.device_put(jnp.array(local_id_count_np), sharding)
        local_table = jax.device_put(jnp.array(local_table_np), sharding)
        log("inputs sharded")

        if recv_capacity is None:
            recv_capacity = max_local_ids + world_size * _TILE_INT32

        # JAX ragged_all_to_all reference (oracle on the same hardware).
        jax_ref_out = jagged_a2a_embedding_lookup_reference_jax(
            local_ids, local_id_count, local_table,
            mesh=mesh, axis_name="x",
        )
        jax_ref_np = np.array(jax_ref_out)
        log("jax_ref computed")

        # The kernel under test.
        log(f"calling kernel under test: recv_capacity={recv_capacity}")
        out = jagged_a2a_embedding_lookup(
            local_ids, local_id_count, local_table,
            mesh=mesh, axis_name="x",
            recv_capacity=recv_capacity, interpret=False,
        )
        log("kernel call returned, fetching output")
        out_np = np.array(out)
        log("output fetched, comparing")

        atol, rtol = (5e-3, 5e-3) if dtype == jnp.bfloat16 else (1e-6, 1e-6)
        for r in range(world_size):
            n = int(local_id_count_np[r])
            np.testing.assert_allclose(
                out_np[r, :n], ref_out_np[r, :n], atol=atol, rtol=rtol,
                err_msg=f"kernel vs numpy mismatch on rank {r} ({distribution}, {dtype})",
            )
            np.testing.assert_allclose(
                out_np[r, :n], jax_ref_np[r, :n], atol=atol, rtol=rtol,
                err_msg=f"kernel vs jax-ref mismatch on rank {r} ({distribution}, {dtype})",
            )

    # --- Round-robin distribution (the v0 baseline coverage). ---

    def test_world_size_2_round_robin_f32(self) -> None:
        self._run(world_size=2, distribution="round_robin", dtype=np.float32)

    def test_world_size_4_round_robin_f32(self) -> None:
        self._run(world_size=4, distribution="round_robin", dtype=np.float32)

    def test_world_size_4_round_robin_bf16(self) -> None:
        self._run(world_size=4, distribution="round_robin", dtype=jnp.bfloat16)

    # --- Skewed distribution (exercises host-side sentinel padding). ---

    def test_world_size_4_skewed_f32(self) -> None:
        self._run(world_size=4, distribution="skewed", dtype=np.float32)

    def test_world_size_4_skewed_bf16(self) -> None:
        self._run(world_size=4, distribution="skewed", dtype=jnp.bfloat16)

    # --- Overflow waves: ``recv_capacity`` < worst-case forces multiple waves. ---

    def test_world_size_4_skewed_f32_overflow_waves(self) -> None:
        """Pass ``recv_capacity`` ~3x smaller than worst case → multi-wave loop.

        ``max_local_ids = 4 * TILE = 512``, worst-case ``recv_capacity =
        512 + 4*128 = 1024``. We pass ``recv_capacity = 512`` (the minimum
        SC-divisible size given ``num_subcores * num_lanes = 256``), forcing
        ``ceil(1024/512) = 2`` waves at minimum and exercising the wave loop
        end-to-end against the ``ragged_all_to_all`` oracle.
        """
        world_size = 4
        max_local_ids = world_size * _TILE_INT32  # 512
        # Worst case = 1024; pick 512 → 2 waves. Must be a multiple of
        # num_subcores * num_lanes = 256 (v7x SC SIMD width).
        recv_capacity = 512
        self._run(
            world_size=world_size,
            distribution="skewed",
            dtype=np.float32,
            max_local_ids=max_local_ids,
            recv_capacity=recv_capacity,
        )

    # --- All-TC fused f32 single-call variant: in-kernel wave loop. ---

    def test_alltc_f32_world_size_4_skewed_overflow_waves(self) -> None:
        """All-TC fused f32 kernel: multi-wave via in-kernel ``lax.fori_loop``.

        WO#4 (per-wave EVEN-SPLIT) is now in effect: each per-pair count is
        padded to a multiple of ``max_waves * TILE`` so every wave gets
        identical per-pair counts, sidestepping the W>=4 multi-iter Pallas
        hang (previously triggered by per-pair DMA counts that differed
        across waves). ``recv_capacity = 1024`` (bumped from 512) is the WO#4
        cost trade-off — needed to fit ``W * (max_local_ids / max_waves)``
        per-wave per-receiver fan-in with ``max_waves = 2``. Cross-checks
        against the same numpy + ``ragged_all_to_all`` oracles.
        """
        import sys
        def log(msg):
            print(f"[TEST_TRACE] {msg}", flush=True)
            sys.stderr.write(f"[TEST_TRACE_ERR] {msg}\n")
            sys.stderr.flush()

        world_size = 4
        distribution = "skewed"
        dtype = np.float32
        v_local = 256
        embedding_dim = 128
        max_local_ids = world_size * _TILE_INT32  # 512
        recv_capacity = 1024  # WO#4 cost trade-off (was 512); → 2 waves.

        log(
            f"alltc _run START world_size={world_size} dist={distribution} "
            f"dtype={dtype} max_local_ids={max_local_ids} recv_capacity={recv_capacity}"
        )

        devices = jax.devices()[:world_size]
        if len(devices) < world_size:
            self.skipTest(f"Need {world_size} devices, have {len(devices)}")
        mesh = jax.sharding.Mesh(np.array(devices), ("x",))

        local_table_np = _build_local_table(
            world_size, v_local, embedding_dim, dtype, seed=0
        )
        local_ids_np, local_id_count_np = _make_skewed_inputs(
            world_size, v_local, max_local_ids, seed=2
        )
        log(f"inputs built; local_id_count={local_id_count_np.tolist()}")

        ref_out_np = jagged_a2a_embedding_lookup_reference(
            local_ids_np, local_id_count_np,
            local_table_np.astype(np.float32), world_size,
        ).astype(dtype)
        log("numpy reference computed")

        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("x"))
        local_ids = jax.device_put(jnp.array(local_ids_np), sharding)
        local_id_count = jax.device_put(jnp.array(local_id_count_np), sharding)
        local_table = jax.device_put(jnp.array(local_table_np), sharding)
        log("inputs sharded")

        jax_ref_out = jagged_a2a_embedding_lookup_reference_jax(
            local_ids, local_id_count, local_table,
            mesh=mesh, axis_name="x",
        )
        jax_ref_np = np.array(jax_ref_out)
        log("jax_ref computed")

        log(f"calling all-TC fused kernel: recv_capacity={recv_capacity}")
        out = jagged_a2a_embedding_lookup_f32_alltc(
            local_ids, local_id_count, local_table,
            mesh=mesh, axis_name="x",
            recv_capacity=recv_capacity, interpret=False,
        )
        log("kernel call returned, fetching output")
        out_np = np.array(out)
        log("output fetched, comparing")

        atol, rtol = 1e-6, 1e-6
        for r in range(world_size):
            n = int(local_id_count_np[r])
            np.testing.assert_allclose(
                out_np[r, :n], ref_out_np[r, :n], atol=atol, rtol=rtol,
                err_msg=f"alltc kernel vs numpy mismatch on rank {r}",
            )
            np.testing.assert_allclose(
                out_np[r, :n], jax_ref_np[r, :n], atol=atol, rtol=rtol,
                err_msg=f"alltc kernel vs jax-ref mismatch on rank {r}",
            )


if __name__ == "__main__":
    # To reproduce ALL tests above on real TPU (recir-v7-8-1, 4 chips x 2
    # chiplets = 8 JAX devices), inside the v14+ container:
    #
    #   python -m unittest -v recir.ops.pallas.embeddings.tests.test_jagged_a2a_lookup
    #
    # Or submit via JobSet from the shenli-client GCP VM:
    #
    #   kubectl apply -f ~/jagged_a2a_test_jobset.yaml
    unittest.main()
