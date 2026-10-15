# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Real-TPU forward correctness tests for ``ringshift_embedding_lookup``.

``RingshiftFwdCorrectnessTest`` -- bitwise-identical correctness vs the
``lax.ragged_all_to_all`` reference at W=4/32/64/128. Asserts ``max_diff == 0.0``.

Each ``test_w{N}_*`` method targets a different mesh size:
  W=4   -- single host (4 chips)
  W=32  -- 32 hosts x 1 chip per host (devices[::8])
  W=64  -- 32 hosts x 2 chips per host (devices[::4])
  W=128 -- 32 hosts x 4 chips per host (devices[::2])

These run under the JobSet manifests in ``manifests/`` -- each manifest invokes
one method on a TPU-v7x topology sized for that W. Perf/scale benchmarking lives
in the standalone ``ringshift_bench`` argparse binary, not here.
"""

from __future__ import annotations

import functools
import os
import sys
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.multihost_utils import sync_global_devices
from recir.ops.pallas.embeddings.ringshift import (
    _POOLING_IMPL_ENV,
    _use_dma_gather,
    pool_per_rank,
    ringshift_embedding_lookup,
    ringshift_embedding_lookup_reference,
    ringshift_embedding_lookup_reference_jax,
)


def _has_tpu() -> bool:
    """Return True if real TPU devices are available."""
    try:
        return any(d.platform == "tpu" for d in jax.devices())
    except Exception:
        return False


# Multi-host init: jax.devices() only sees the full TPU slice after
# jax.distributed.initialize(). Auto-detected from the TPU runtime --
# TPU_WORKER_HOSTNAMES (GKE) / MEGASCALE_COORDINATOR_ADDRESS (GCE/multislice)
# are injected by the runtime (the same signals jax's GkeTpuCluster uses), so a
# no-arg initialize() then picks up coordinator/num_processes/process_id
# automatically. Must run before jax.devices(); skipped on CPU/dev (where
# initialize() would raise) and idempotent across re-imports.
if not jax.distributed.is_initialized() and (
    os.environ.get("TPU_WORKER_HOSTNAMES")
    or os.environ.get("MEGASCALE_COORDINATOR_ADDRESS")
):
    jax.distributed.initialize()


def _make_skewed_inputs_on_device(mesh, world_size, max_local_ids, v_local, seed):
    """Skewed inputs (75% to hot peer) generated per-rank via shard_map."""
    P = jax.sharding.PartitionSpec

    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(),
        out_specs=(P("x"), P("x")),
        check_vma=False,
    )
    def make_inputs():
        rank = jax.lax.axis_index("x")
        key = jax.random.fold_in(jax.random.PRNGKey(seed), rank)
        valid = (max_local_ids * 4) // 5
        hot_peer = (rank + 1) % world_size
        u_hot, u_owner, u_row = jax.random.split(key, 3)
        is_hot = jax.random.uniform(u_hot, (valid,)) < 0.75
        random_owner = jax.random.randint(u_owner, (valid,), 0, world_size)
        owner = jnp.where(is_hot, hot_peer, random_owner)
        row_in_owner = jax.random.randint(u_row, (valid,), 0, v_local)
        ids = (owner + world_size * row_in_owner).astype(jnp.int32)
        local_ids_padded = jnp.zeros(max_local_ids, dtype=jnp.int32).at[:valid].set(ids)
        local_id_count = jnp.array(valid, dtype=jnp.int32)
        return local_ids_padded[None, :], local_id_count[None]

    return make_inputs()


@unittest.skipUnless(_has_tpu(), "Requires real TPU")
class RingshiftFwdCorrectnessTest(unittest.TestCase):
    """Bitwise-identical correctness vs ra2a reference at W=4/32/64/128."""

    def _run_correctness_with_devices(
        self,
        world_size,
        devices,
        recv_capacity: int | None = None,
        v_local: int = 8 * 1024,
        expect_dma_gather: bool | None = None,
    ) -> None:
        """Shared helper: run fused vs ra2a correctness check at given W.

        D=128, ``v_local`` rows/shard (default 8K), max_local_ids=8K.
        ``recv_capacity`` defaults to max_local_ids+TILE (single-chunk path); pass
        a SMALLER TILE-aligned value to force the chunked-recv path
        (num_chunks > 1) -- the numerical gate for the chunked gather/scatter
        (skewed traffic makes a single peer's payload exceed recv_capacity, so
        multiple chunks are actively used).
        ``v_local`` large enough that ``v_local * D * 4`` exceeds the VMEM budget
        (``_TABLE_VMEM_BUDGET_BYTES``) exercises the HBM per-row DMA-gather path
        instead of the whole-shard VMEM prefetch; ``expect_dma_gather`` (when set)
        asserts which path this config takes so the test can't silently drift onto
        the wrong one.
        Per-rank random fp32 table (different rng key per rank).
        Computes max |out - ref| in-graph; asserts exactly 0.0 (bitwise-identical).
        Prints `[CORRECTNESS_W{N}] max_diff = ...` and PASS/FAIL on rank 0.
        """
        TILE = 128
        embedding_dim = 128
        max_local_ids = 8 * 1024

        if expect_dma_gather is not None:
            self.assertEqual(
                _use_dma_gather(v_local, embedding_dim, jnp.float32),
                expect_dma_gather,
                f"v_local={v_local} D={embedding_dim} does not take the expected "
                f"path (expect_dma_gather={expect_dma_gather})",
            )

        self.assertEqual(
            len(devices),
            world_size,
            f"expected {world_size} devices, got {len(devices)}",
        )
        mesh = jax.sharding.Mesh(np.array(devices), ("x",))

        explicit_recv_capacity = recv_capacity is not None
        if recv_capacity is None:
            recv_capacity = max_local_ids + TILE
        recv_capacity = ((recv_capacity + TILE - 1) // TILE) * TILE

        # When a caller forces a small recv_capacity (the *_chunked tests), make
        # sure it actually exercises the chunked-recv path. num_chunks mirrors the
        # kernel (ceil((max_local_ids + world_size*TILE) / recv_capacity)); without
        # this guard a too-large recv_capacity would make the test pass trivially
        # on the single-chunk path.
        if explicit_recv_capacity:
            max_padded = max_local_ids + world_size * TILE
            num_chunks = -(-max_padded // recv_capacity)  # ceil division
            self.assertGreater(
                num_chunks,
                1,
                f"recv_capacity={recv_capacity} too large to exercise the chunked "
                f"path (num_chunks={num_chunks}); lower it below max_padded="
                f"{max_padded}",
            )

        tag = f"CORRECTNESS_W{world_size}"

        def log(msg):
            if jax.process_index() == 0:
                print(f"[{tag}] {msg}", file=sys.stderr, flush=True)

        log(
            f"=== D={embedding_dim} v_local={v_local} max_local_ids={max_local_ids} ==="
        )

        log("creating inputs ...")
        local_ids, local_id_count = _make_skewed_inputs_on_device(
            mesh, world_size, max_local_ids, v_local, seed=2
        )

        # Per-rank RANDOM fp32 table (NOT zero) so out vs ref comparison is
        # meaningful. Different rng key per rank so neighboring shards differ.
        P = jax.sharding.PartitionSpec

        @functools.partial(
            jax.shard_map,
            mesh=mesh,
            in_specs=(),
            out_specs=P("x"),
            check_vma=False,
        )
        def make_table_random():
            rank = jax.lax.axis_index("x")
            key = jax.random.fold_in(jax.random.PRNGKey(7), rank)
            return jax.random.normal(
                key, (1, v_local, embedding_dim), dtype=jnp.float32
            )

        # pyrefly: ignore [missing-argument]
        local_table = make_table_random()
        jax.block_until_ready((local_ids, local_id_count, local_table))
        args = (local_ids, local_id_count, local_table)
        log("inputs ready (random fp32 table)")

        rs_gp_jit = jax.jit(
            functools.partial(
                ringshift_embedding_lookup,
                mesh=mesh,
                axis_name="x",
                recv_capacity=recv_capacity,
                splitter="globalpad",
            )
        )
        ra2a_jit = jax.jit(
            functools.partial(
                ringshift_embedding_lookup_reference_jax,
                mesh=mesh,
                axis_name="x",
            )
        )

        log("-- computing reference (ra2a) --")
        sync_global_devices("ref_start")
        ref = ra2a_jit(*args)
        jax.block_until_ready(ref)
        log("ref ready")

        log("-- computing fused (ringshift + globalpad) --")
        sync_global_devices("fused_start")
        out = rs_gp_jit(*args)
        jax.block_until_ready(out)
        log("fused ready")

        # Compute max diff in-graph; result is fully replicated -> safe to
        # device_get on every host.
        log("-- computing max_diff in-graph --")
        diff_max = jnp.max(jnp.abs(out - ref))
        diff_max_value = float(jax.device_get(diff_max))
        sync_global_devices("diff_done")

        log(f"max_diff = {diff_max_value:.6e}")
        if diff_max_value == 0.0:
            log(f"PASS: max_diff {diff_max_value:.6e} == 0 (bitwise-identical)")
        else:
            log(f"FAIL: max_diff {diff_max_value:.6e} != 0")
        log(f"=== RESULT: max_diff={diff_max_value:.6e} ===")

        self.assertEqual(diff_max_value, 0.0)

    def test_w4_correctness(self) -> None:
        """W=4 single-host correctness check (fused vs ra2a reference)."""
        all_devices = jax.devices()
        if len(all_devices) < 4:
            self.skipTest(f"Need 4 devices, have {len(all_devices)}")
        devices = all_devices[:4]
        self._run_correctness_with_devices(world_size=4, devices=devices)

    def test_w4_correctness_chunked(self) -> None:
        """W=4 correctness with a SMALL recv_capacity to force num_chunks > 1.

        Numerical gate for the chunked-recv fix: recv_capacity=2048 < the skewed
        per-pair payload (~0.6 * max_local_ids ~= 4.9k at max_local_ids=8K), so the
        gather runs in multiple chunks; result must still be bitwise-identical to
        the ra2a reference. Single-host (4 chips) is enough -- the chunk loop is
        local (no cross-host dependency).
        """
        all_devices = jax.devices()
        if len(all_devices) < 4:
            self.skipTest(f"Need 4 devices, have {len(all_devices)}")
        devices = all_devices[:4]
        self._run_correctness_with_devices(
            world_size=4, devices=devices, recv_capacity=2048
        )

    def test_w4_dma_gather(self) -> None:
        """W=4 correctness with a LARGE table shard to force the HBM DMA-gather.

        v_local=256K rows/shard at D=128 f32 -> 128MB, which is BOTH over the 16MB
        VMEM staging budget (so the kernel picks the DMA path) AND over the ~64MB
        VMEM itself (so the whole-shard prefetch path physically cannot compile).
        A PASS therefore proves the per-row HBM DMA-gather path actually ran --
        not just that the threshold selected it. Result must stay bitwise-identical
        to the ra2a reference. Single-host (4 chips) is enough -- the gather is
        local (no cross-host dependency).
        """
        all_devices = jax.devices()
        if len(all_devices) < 4:
            self.skipTest(f"Need 4 devices, have {len(all_devices)}")
        devices = all_devices[:4]
        self._run_correctness_with_devices(
            world_size=4,
            devices=devices,
            v_local=256 * 1024,
            expect_dma_gather=True,
        )

    def test_w4_dma_gather_chunked(self) -> None:
        """W=4 DMA-gather AND chunked-recv together (num_chunks > 1).

        Combines the large-shard HBM per-row DMA path (v_local=256K, prefetch
        would OOM VMEM) with a small recv_capacity=2048 so the per-row DMA gather
        runs inside the multi-chunk loop. Guards the interaction of the two
        independent tilings.
        """
        all_devices = jax.devices()
        if len(all_devices) < 4:
            self.skipTest(f"Need 4 devices, have {len(all_devices)}")
        devices = all_devices[:4]
        self._run_correctness_with_devices(
            world_size=4,
            devices=devices,
            recv_capacity=2048,
            v_local=256 * 1024,
            expect_dma_gather=True,
        )

    def test_w4_prefetch_path(self) -> None:
        """W=4 sanity that the default small shard stays on the VMEM prefetch path.

        Explicit guard that v_local=8K (D=128 f32 -> 4MB < 16MB budget) does NOT
        take the DMA-gather path, so the hybrid threshold keeps the fast path for
        small tables.
        """
        all_devices = jax.devices()
        if len(all_devices) < 4:
            self.skipTest(f"Need 4 devices, have {len(all_devices)}")
        devices = all_devices[:4]
        self._run_correctness_with_devices(
            world_size=4,
            devices=devices,
            expect_dma_gather=False,
        )

    def test_w32_correctness_chunked(self) -> None:
        """W=32 multi-host correctness with a SMALL recv_capacity (num_chunks > 1).

        Same chunked-path gate as test_w4_correctness_chunked, but multi-host so
        the chunked gather runs alongside the cross-host ring DMAs.
        """
        all_devices = jax.devices()
        if len(all_devices) < 256:
            self.skipTest(f"Need 256 logical devices, have {len(all_devices)}")
        devices = all_devices[::8]
        self._run_correctness_with_devices(
            world_size=32, devices=devices, recv_capacity=2048
        )

    def test_w32_correctness(self) -> None:
        """W=32 numerical correctness check: fused vs ra2a reference.

        Uses devices[::8] (1 device per host across 32 hosts on a 4x4x8
        v7x slice). D=128, v_local=8K, max_local_ids=8K, recv_capacity=8K+TILE.
        """
        all_devices = jax.devices()
        if len(all_devices) < 256:
            self.skipTest(f"Need 256 logical devices, have {len(all_devices)}")
        devices = all_devices[::8]
        self._run_correctness_with_devices(world_size=32, devices=devices)

    def test_w64_correctness(self) -> None:
        """W=64 numerical correctness check: fused vs ra2a reference.

        Uses devices[::4] (2 devices per host across 32 hosts).
        """
        all_devices = jax.devices()
        if len(all_devices) < 256:
            self.skipTest(f"Need 256 logical devices, have {len(all_devices)}")
        devices = all_devices[::4]
        self._run_correctness_with_devices(world_size=64, devices=devices)

    def test_w128_correctness(self) -> None:
        """W=128 numerical correctness check: fused vs ra2a reference.

        Uses devices[::2] (4 devices per host across 32 hosts).
        """
        all_devices = jax.devices()
        if len(all_devices) < 256:
            self.skipTest(f"Need 256 logical devices, have {len(all_devices)}")
        devices = all_devices[::2]
        self._run_correctness_with_devices(world_size=128, devices=devices)


class RingshiftEbcCorrectnessTest(unittest.TestCase):
    """Pooled (EBC) correctness on real TPU, against INDEPENDENT references.

    The benchmark already checks that the two pooled implementations agree with
    each other. That is a relative check: it cannot catch a mistake both make,
    and more importantly it cannot catch one they inherit from the shared
    host-side prep (segment ids, sort permutation, per-peer padding), which is
    the part with the most index arithmetic.

    So both arms are checked here against references that share no code with
    them:

      * forward -- the numpy oracle (``ringshift_embedding_lookup_reference``),
        a plain double loop over (rank, id). No segment_sum, no kernel, no
        permutation.
      * gradient -- ``ringshift_embedding_lookup_reference_jax`` composed with
        ``pool_per_rank``, differentiated by autodiff. That path never enters
        the Pallas kernel, so it is an independent check of Phase 4's
        hand-written ``custom_vjp`` -- which nothing else exercises, on any
        machine.

    Single host only. The oracle needs the whole global table on the host, and
    a W=32 shard set is both large and spread across processes.
    """

    WORLD_SIZE = 4
    DIM = 16
    V_LOCAL = 64
    MAX_LOCAL_IDS = 256
    NUM_BAGS = 32
    # Bags hold ~8 f32 rows, so the arms differ from the oracle only by
    # summation order. Far above that rounding, far below any index bug.
    RTOL = 1e-5

    def _log(self, msg: str) -> None:
        if jax.process_index() == 0:
            print(f"[EBC_CORRECTNESS] {msg}", file=sys.stderr, flush=True)

    def _bag_lengths(self, counts: np.ndarray) -> np.ndarray:
        """Hand-built so the interesting cases are guaranteed, not hoped for.

        A random split would *probably* produce an empty bag; this makes the
        empty, singleton and hot cases certain, and keeps the layout identical
        across runs so a failure is reproducible.
        """
        out = np.zeros((self.WORLD_SIZE, self.NUM_BAGS), dtype=np.int32)
        for r in range(self.WORLD_SIZE):
            total = int(counts[r])
            lengths = np.zeros(self.NUM_BAGS, dtype=np.int32)
            lengths[0] = 0  # empty bag: divisor clamp + trash-segment path
            lengths[1] = 1  # singleton: pooling must be a no-op on it
            spread = self.NUM_BAGS - 3
            base = (total - 1) // (spread + 4)  # leave a surplus for the hot bag
            lengths[2 : 2 + spread] = base
            lengths[-1] = total - int(lengths.sum())  # hot bag takes the rest
            assert lengths[-1] >= 0, (total, base)
            assert int(lengths.sum()) == total
            out[r] = lengths
        return out

    def _inputs(self, mesh):
        local_ids, local_id_count = _make_skewed_inputs_on_device(
            mesh, self.WORLD_SIZE, self.MAX_LOCAL_IDS, self.V_LOCAL, seed=5
        )
        P = jax.sharding.PartitionSpec

        @functools.partial(
            jax.shard_map, mesh=mesh, in_specs=(), out_specs=P("x"), check_vma=False
        )
        def make_table():
            rank = jax.lax.axis_index("x")
            key = jax.random.fold_in(jax.random.PRNGKey(11), rank)
            return jax.random.normal(
                key, (1, self.V_LOCAL, self.DIM), dtype=jnp.float32
            )

        # pyrefly: ignore [missing-argument]
        table = make_table()
        jax.block_until_ready((local_ids, local_id_count, table))
        bags = jnp.asarray(
            self._bag_lengths(np.asarray(jax.device_get(local_id_count)))
        )
        return local_ids, local_id_count, table, bags

    def _rel(self, got, want) -> float:
        want = np.asarray(want, dtype=np.float64)
        got = np.asarray(got, dtype=np.float64)
        # A floor of 1 makes this an atol+rtol-style check: zero-valued bags
        # cannot borrow tolerance from a large value elsewhere in the output.
        scale = np.maximum(np.abs(want), 1.0)
        return float(np.max(np.abs(got - want) / scale))

    def test_w4_ebc_correctness(self) -> None:
        """Both pooled arms vs the numpy oracle (forward) and a non-kernel VJP."""
        all_devices = jax.devices()
        if len(all_devices) < self.WORLD_SIZE:
            self.skipTest(f"Need {self.WORLD_SIZE} devices, have {len(all_devices)}")
        if jax.process_count() > 1:
            self.skipTest("single-host only: the oracle needs the global table")

        mesh = jax.sharding.Mesh(np.array(all_devices[: self.WORLD_SIZE]), ("x",))
        ids, count, table, bags = self._inputs(mesh)
        recv_capacity = self.MAX_LOCAL_IDS + 128
        self._log(
            f"=== W={self.WORLD_SIZE} D={self.DIM} v_local={self.V_LOCAL} "
            f"num_bags={self.NUM_BAGS} ==="
        )

        ids_np = np.asarray(jax.device_get(ids))
        table_np = np.asarray(jax.device_get(table))
        bags_np = np.asarray(jax.device_get(bags))

        def arm(pooling: str, impl: str, t):
            previous = os.environ.get(_POOLING_IMPL_ENV)
            os.environ[_POOLING_IMPL_ENV] = impl
            try:
                return ringshift_embedding_lookup(
                    ids,
                    None,
                    t,
                    mesh=mesh,
                    axis_name="x",
                    recv_capacity=recv_capacity,
                    splitter="globalpad",
                    pooling=pooling,
                    bag_lengths=bags,
                )
            finally:
                if previous is None:
                    os.environ.pop(_POOLING_IMPL_ENV, None)
                else:
                    os.environ[_POOLING_IMPL_ENV] = previous

        failures = []
        for pooling in ("sum", "mean"):
            # ---- forward vs the numpy oracle -------------------------------
            want = ringshift_embedding_lookup_reference(
                ids_np, None, table_np, self.WORLD_SIZE, pooling, bags_np
            )
            for impl in ("xla", "fused_kernel"):
                got = jax.device_get(
                    jax.jit(functools.partial(arm, pooling, impl))(table)
                )
                rel = self._rel(got, want)
                ok = rel <= self.RTOL
                self._log(
                    f"fwd  {pooling:4s} {impl:6s}: max_rel_diff={rel:.3e} "
                    f"{'PASS' if ok else 'FAIL'}"
                )
                if not ok:
                    failures.append(f"fwd/{pooling}/{impl}={rel:.3e}")

            # ---- gradient vs a reference path that never enters the kernel --
            # Random cotangent, not ones: an all-ones cotangent makes many index
            # mistakes invisible, because every bag then contributes equally.
            cot = jax.random.normal(
                jax.random.PRNGKey(13),
                (self.WORLD_SIZE, self.NUM_BAGS, self.DIM),
                dtype=jnp.float32,
            )

            def ref_loss(t, pooling=pooling, cot=cot):
                per_id = ringshift_embedding_lookup_reference_jax(
                    ids, count, t, mesh=mesh, axis_name="x"
                )
                pooled = jax.vmap(functools.partial(pool_per_rank, pooling=pooling))(
                    per_id, bags
                )
                return jnp.sum(pooled * cot)

            want_grad = jax.device_get(jax.jit(jax.grad(ref_loss))(table))
            for impl in ("xla", "fused_kernel"):

                def arm_loss(t, pooling=pooling, impl=impl, cot=cot):
                    return jnp.sum(arm(pooling, impl, t) * cot)

                got_grad = jax.device_get(jax.jit(jax.grad(arm_loss))(table))
                rel = self._rel(got_grad, want_grad)
                ok = rel <= self.RTOL
                self._log(
                    f"grad {pooling:4s} {impl:6s}: max_rel_diff={rel:.3e} "
                    f"{'PASS' if ok else 'FAIL'}"
                )
                if not ok:
                    failures.append(f"grad/{pooling}/{impl}={rel:.3e}")

        self._log(
            f"=== RESULT ebc_correctness: {'PASS' if not failures else 'FAIL'} "
            f"rtol={self.RTOL:.0e} {' '.join(failures)} ==="
        )
        self.assertEqual(
            failures, [], f"pooled arms disagree with reference: {failures}"
        )


if __name__ == "__main__":
    unittest.main()
