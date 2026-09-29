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
    _use_dma_gather,
    ringshift_embedding_lookup,
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


if __name__ == "__main__":
    unittest.main()
