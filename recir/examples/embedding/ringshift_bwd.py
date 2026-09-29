# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Real-TPU backward correctness tests for ``ringshift_embedding_lookup``.

Two ``unittest.TestCase`` classes:

  * ``RingshiftBwdW4Test`` -- W=4 backward correctness suite on a single
    4-chip TPU pod. Compares ``jax.grad(forward_loss)`` (which goes through
    the ``custom_vjp`` ``_bwd``) against ``jax.grad(reference_loss)`` (which
    uses the auto-differentiable ``ringshift_embedding_lookup_reference_jax``).
    Tolerances: ``atol=1e-4, rtol=1e-4``. Coverage:
      - ``test_w4_base_correctness``      -- skewed inputs (~75% to hot peer).
      - ``test_w4_dma_gather_base``       -- large shard -> HBM DMA scatter-add.
      - ``test_w4_dma_gather_repeated_hot_row`` -- HBM RMW accumulation on one row.
      - ``test_w4_repeated_hot_row``      -- worst-case scatter-add collision.
      - ``test_w4_wave0_self_peer``       -- wave 0 my_src == my_dst == my_rank.
      - ``test_w4_zero_count``            -- ``local_id_count = 0`` everywhere.
      - ``test_w4_non_tile_aligned_count``-- per-peer counts not multiples of TILE.
      - ``test_w4_jit_value_and_grad``    -- jit(value_and_grad(loss)) composition.

  * ``RingshiftBwdW32Test`` -- W=32 backward correctness on a 32-host TPU-v7x
    slice (``devices[::8]`` -- 1 device per host across all 32 hosts).

These run under the JobSet manifests in ``manifests/`` -- each manifest
invokes one method (or the whole W=4 suite) on a TPU-v7x topology sized for
that W. Perf/scale benchmarking lives in the standalone ``ringshift_bench``
argparse binary, not here.
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


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------
#
# We compute the gradient w.r.t. the embedding table only (the int32 IDs and
# count are non-differentiable). The "loss" is a deterministic dot product
# with a fixed target tensor -- it has the same value-and-gradient
# structure as a real downstream consumer (e.g. dense layer weights) but
# produces deterministic gradients independent of any RNG.


def _fused_loss_for_table(mesh, axis_name, recv_capacity, splitter):
    """Returns a loss(local_ids, local_id_count, local_table, target).

    Loss = sum(forward(table) * target). Gradient w.r.t. table = grad_out
    backprop = target -> custom_vjp.

    Multi-host safety: all jax.Array inputs are passed as function args
    (not closed over) so jit can serialize them across host processes
    instead of trying to inline the per-host shards as MLIR constants.
    """

    def loss(local_ids, local_id_count, local_table, target):
        out = ringshift_embedding_lookup(
            local_ids,
            local_id_count,
            local_table,
            mesh=mesh,
            axis_name=axis_name,
            recv_capacity=recv_capacity,
            splitter=splitter,
        )
        return jnp.sum(out * target)

    return loss


def _ref_loss_for_table(mesh, axis_name):
    def loss(local_ids, local_id_count, local_table, target):
        out = ringshift_embedding_lookup_reference_jax(
            local_ids,
            local_id_count,
            local_table,
            mesh=mesh,
            axis_name=axis_name,
        )
        return jnp.sum(out * target)

    return loss


def _make_skewed_inputs(mesh, world_size, max_local_ids, v_local, seed):
    """Skewed inputs: ~75% to one hot peer per rank."""
    P = jax.sharding.PartitionSpec

    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(),
        out_specs=(P("x"), P("x")),
        check_vma=False,
    )
    def make():
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

    return make()


def _make_random_table(mesh, world_size, v_local, embedding_dim, seed):
    """Per-rank random fp32 table (different rng key per rank)."""
    P = jax.sharding.PartitionSpec

    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(),
        out_specs=P("x"),
        check_vma=False,
    )
    def make():
        rank = jax.lax.axis_index("x")
        key = jax.random.fold_in(jax.random.PRNGKey(seed), rank)
        return jax.random.normal(key, (1, v_local, embedding_dim), dtype=jnp.float32)

    return make()


def _make_random_target(mesh, world_size, max_local_ids, embedding_dim, seed):
    """Per-rank random fp32 target tensor."""
    P = jax.sharding.PartitionSpec

    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(),
        out_specs=P("x"),
        check_vma=False,
    )
    def make():
        rank = jax.lax.axis_index("x")
        key = jax.random.fold_in(jax.random.PRNGKey(seed), rank)
        return jax.random.normal(
            key, (1, max_local_ids, embedding_dim), dtype=jnp.float32
        )

    return make()


# ---------------------------------------------------------------------------
# Single-host (W=4) tests on the local TPU pod
# ---------------------------------------------------------------------------


@unittest.skipUnless(_has_tpu(), "Backward TPU tests require real TPU")
class RingshiftBwdW4Test(unittest.TestCase):
    """W=4 backward correctness suite on a single 4-chip TPU pod.

    Uses small shapes so each test compiles + runs in seconds. Tolerances:
    ``atol=1e-4, rtol=1e-4`` (atol=1e-4 has ~10x headroom over the
    ``eps * sqrt(N) * ||grad||`` worst-case for hot-row scatter; rtol=1e-4
    catches any large-magnitude misses).
    """

    WORLD_SIZE = 4
    EMBEDDING_DIM = 128
    V_LOCAL = 1024
    # Large per-shard table that forces the HBM per-row DMA scatter-add path AND
    # makes the whole-shard VMEM prefetch physically impossible:
    # V_LOCAL_DMA * EMBEDDING_DIM * 4 = 128MB -- over the 16MB staging budget (so
    # the kernel selects the DMA path) and over the ~64MB VMEM (so a prefetch
    # fallback would fail to compile). A PASS therefore proves the DMA scatter-add
    # path actually executed.
    V_LOCAL_DMA = 256 * 1024
    MAX_LOCAL_IDS = 1024
    TILE = 128
    ATOL = 1e-4
    RTOL = 1e-4

    def setUp(self) -> None:
        devices = jax.devices()
        if len(devices) < self.WORLD_SIZE:
            self.skipTest(f"Need {self.WORLD_SIZE} devices, have {len(devices)}")
        self.devices = devices[: self.WORLD_SIZE]
        self.mesh = jax.sharding.Mesh(np.array(self.devices), ("x",))
        # ringshift: per-wave incoming = ONE peer's payload only (<=
        # max_local_ids + TILE - 1).
        self.recv_capacity = (
            (self.MAX_LOCAL_IDS + self.TILE + self.TILE - 1) // self.TILE
        ) * self.TILE

    def _log(self, tag, msg):
        print(f"[{tag}] {msg}", file=sys.stderr, flush=True)

    def _check_grad_close(
        self,
        tag,
        local_ids,
        local_id_count,
        local_table,
        target,
    ):
        """Run jit(grad(fused_loss)) and jit(grad(ref_loss)) and compare.

        Loss is `sum(forward(table) * target)`. Gradient is taken w.r.t.
        the table arg only (argnums=2). All other args (ids/count/target)
        are passed AS arguments to the jit, not closed over -- required
        for multi-host correctness.
        """
        fused_loss = _fused_loss_for_table(
            mesh=self.mesh,
            axis_name="x",
            recv_capacity=self.recv_capacity,
            splitter="globalpad",
        )
        ref_loss = _ref_loss_for_table(
            mesh=self.mesh,
            axis_name="x",
        )

        fused_grad_jit = jax.jit(jax.grad(fused_loss, argnums=2))
        ref_grad_jit = jax.jit(jax.grad(ref_loss, argnums=2))

        self._log(tag, "computing fused grad ...")
        fused_grad = fused_grad_jit(local_ids, local_id_count, local_table, target)
        jax.block_until_ready(fused_grad)

        self._log(tag, "computing ref grad ...")
        ref_grad = ref_grad_jit(local_ids, local_id_count, local_table, target)
        jax.block_until_ready(ref_grad)

        # Compare in-graph: replicated scalar.
        diff_max = jnp.max(jnp.abs(fused_grad - ref_grad))
        ref_max = jnp.max(jnp.abs(ref_grad))
        diff_max_v = float(jax.device_get(diff_max))
        ref_max_v = float(jax.device_get(ref_max))
        self._log(
            tag,
            f"max_abs_diff = {diff_max_v:.6e}, ref_max = {ref_max_v:.6e}",
        )
        # Relative tolerance check (only meaningful if ref has non-trivial
        # magnitude).
        if ref_max_v > 0:
            rel = diff_max_v / max(ref_max_v, 1e-12)
            self._log(tag, f"relative max_diff = {rel:.6e}")

        np.testing.assert_allclose(
            np.asarray(fused_grad),
            np.asarray(ref_grad),
            atol=self.ATOL,
            rtol=self.RTOL,
            err_msg=f"[{tag}] grad mismatch: max_abs_diff={diff_max_v:.6e}",
        )
        return diff_max_v

    # ---- 1. Base correctness ----

    def test_w4_base_correctness(self) -> None:
        """W=4 base case: skewed inputs (~75% to hot peer)."""
        local_ids, local_id_count = _make_skewed_inputs(
            self.mesh,
            self.WORLD_SIZE,
            self.MAX_LOCAL_IDS,
            self.V_LOCAL,
            seed=2,
        )
        local_table = _make_random_table(
            self.mesh,
            self.WORLD_SIZE,
            self.V_LOCAL,
            self.EMBEDDING_DIM,
            seed=7,
        )
        target = _make_random_target(
            self.mesh,
            self.WORLD_SIZE,
            self.MAX_LOCAL_IDS,
            self.EMBEDDING_DIM,
            seed=13,
        )
        jax.block_until_ready((local_ids, local_id_count, local_table, target))
        d = self._check_grad_close(
            "W4_BASE",
            local_ids,
            local_id_count,
            local_table,
            target,
        )
        self._log("W4_BASE", f"PASS max_diff={d:.6e}")
        self.assertLessEqual(d, self.ATOL + self.RTOL)

    def test_w4_chunked_recv_correctness(self) -> None:
        """W=4 skewed grad with a SMALL recv_capacity to force num_chunks > 1.

        Numerical gate for the chunked-recv fix in the BACKWARD kernel: setting
        recv_capacity below the skewed per-pair payload (~0.6 * MAX_LOCAL_IDS)
        makes the Phase-2 scatter run in multiple chunks; the table grad must
        still match the ra2a reference. This is the path whose old single-shot
        scatter (loop to p3_cnt over recv_capacity-sized tiles) crashed with an
        OOB read on real-scale skew.
        """
        # MAX_LOCAL_IDS=1024 -> per-pair ~0.6*1024 ~= 614 > 256 -> several chunks.
        self.recv_capacity = 256
        local_ids, local_id_count = _make_skewed_inputs(
            self.mesh,
            self.WORLD_SIZE,
            self.MAX_LOCAL_IDS,
            self.V_LOCAL,
            seed=2,
        )
        local_table = _make_random_table(
            self.mesh,
            self.WORLD_SIZE,
            self.V_LOCAL,
            self.EMBEDDING_DIM,
            seed=7,
        )
        target = _make_random_target(
            self.mesh,
            self.WORLD_SIZE,
            self.MAX_LOCAL_IDS,
            self.EMBEDDING_DIM,
            seed=13,
        )
        jax.block_until_ready((local_ids, local_id_count, local_table, target))
        d = self._check_grad_close(
            "W4_CHUNKED",
            local_ids,
            local_id_count,
            local_table,
            target,
        )
        self._log("W4_CHUNKED", f"PASS max_diff={d:.6e}")
        self.assertLessEqual(d, self.ATOL + self.RTOL)

    # ---- 1b. DMA scatter-add path (large table shard) ----

    def test_w4_dma_gather_base(self) -> None:
        """W=4 backward with a LARGE table shard to force the HBM DMA scatter-add.

        v_local=64K rows/shard at D=128 f32 -> 32MB > the 16MB VMEM staging
        budget, so the backward kernel read-modify-writes the grad accumulator one
        row at a time directly in HBM (no whole-shard VMEM staging). Skewed inputs;
        table grad must still match the ra2a reference within tolerance.
        """
        self.assertTrue(
            _use_dma_gather(self.V_LOCAL_DMA, self.EMBEDDING_DIM, jnp.float32),
            "V_LOCAL_DMA does not exceed the VMEM budget; test would not exercise "
            "the DMA scatter-add path",
        )
        local_ids, local_id_count = _make_skewed_inputs(
            self.mesh,
            self.WORLD_SIZE,
            self.MAX_LOCAL_IDS,
            self.V_LOCAL_DMA,
            seed=2,
        )
        local_table = _make_random_table(
            self.mesh,
            self.WORLD_SIZE,
            self.V_LOCAL_DMA,
            self.EMBEDDING_DIM,
            seed=7,
        )
        target = _make_random_target(
            self.mesh,
            self.WORLD_SIZE,
            self.MAX_LOCAL_IDS,
            self.EMBEDDING_DIM,
            seed=13,
        )
        jax.block_until_ready((local_ids, local_id_count, local_table, target))
        d = self._check_grad_close(
            "W4_DMA_BASE",
            local_ids,
            local_id_count,
            local_table,
            target,
        )
        self._log("W4_DMA_BASE", f"PASS max_diff={d:.6e}")
        self.assertLessEqual(d, self.ATOL + self.RTOL)

    def test_w4_dma_gather_repeated_hot_row(self) -> None:
        """W=4 DMA scatter-add with all queries on one row (worst-case collision).

        The key correctness gate for the HBM per-row read-modify-write: every
        query points to global id 0, so rank 0 accumulates world_size *
        max_local_ids deltas into row 0. Sequential per-row HBM RMW (each read
        must observe the previous write) must accumulate ALL contributions -- a
        racy/overwriting implementation would drop them. Large v_local forces the
        DMA path.
        """
        self.assertTrue(
            _use_dma_gather(self.V_LOCAL_DMA, self.EMBEDDING_DIM, jnp.float32),
            "V_LOCAL_DMA does not exceed the VMEM budget; test would not exercise "
            "the DMA scatter-add path",
        )
        P = jax.sharding.PartitionSpec
        max_local_ids = self.MAX_LOCAL_IDS
        ed = self.EMBEDDING_DIM

        @functools.partial(
            jax.shard_map,
            mesh=self.mesh,
            in_specs=(),
            out_specs=(P("x"), P("x")),
            check_vma=False,
        )
        def make_inputs():
            ids = jnp.zeros((max_local_ids,), dtype=jnp.int32)
            count = jnp.array(max_local_ids, dtype=jnp.int32)
            return ids[None, :], count[None]

        # pyrefly: ignore [missing-argument, not-iterable]
        local_ids, local_id_count = make_inputs()
        local_table = _make_random_table(
            self.mesh,
            self.WORLD_SIZE,
            self.V_LOCAL_DMA,
            ed,
            seed=7,
        )
        target = _make_random_target(
            self.mesh,
            self.WORLD_SIZE,
            max_local_ids,
            ed,
            seed=13,
        )
        jax.block_until_ready((local_ids, local_id_count, local_table, target))
        d = self._check_grad_close(
            "W4_DMA_HOTROW",
            local_ids,
            local_id_count,
            local_table,
            target,
        )
        self._log("W4_DMA_HOTROW", f"PASS max_diff={d:.6e}")
        self.assertGreaterEqual(d, 0.0)

    # ---- 2. Repeated hot row (worst-case duplicate accumulation) ----

    def test_w4_repeated_hot_row(self) -> None:
        """All queries point to global id 0 (worst-case scatter-add collision).

        Every rank's max_local_ids slots = id 0. Owner rank for id 0 is
        rank 0 -> its scatter loop accumulates ``world_size *
        max_local_ids`` deltas into row 0 (~4K accumulations on rank 0 at
        W=4, max_local_ids=1024). Verifies the scatter-add doesn't silently
        drop or double-count any contribution.
        """
        P = jax.sharding.PartitionSpec
        max_local_ids = self.MAX_LOCAL_IDS
        v_local = self.V_LOCAL
        ed = self.EMBEDDING_DIM

        @functools.partial(
            jax.shard_map,
            mesh=self.mesh,
            in_specs=(),
            out_specs=(P("x"), P("x")),
            check_vma=False,
        )
        def make_inputs():
            ids = jnp.zeros((max_local_ids,), dtype=jnp.int32)
            count = jnp.array(max_local_ids, dtype=jnp.int32)
            return ids[None, :], count[None]

        # pyrefly: ignore [missing-argument, not-iterable]
        local_ids, local_id_count = make_inputs()
        local_table = _make_random_table(
            self.mesh,
            self.WORLD_SIZE,
            v_local,
            ed,
            seed=7,
        )
        target = _make_random_target(
            self.mesh,
            self.WORLD_SIZE,
            max_local_ids,
            ed,
            seed=13,
        )
        jax.block_until_ready((local_ids, local_id_count, local_table, target))
        d = self._check_grad_close(
            "W4_HOTROW",
            local_ids,
            local_id_count,
            local_table,
            target,
        )
        self._log("W4_HOTROW", f"PASS max_diff={d:.6e}")
        self.assertGreaterEqual(d, 0.0)

    # ---- 3. wave=0 self-peer ----

    def test_w4_wave0_self_peer(self) -> None:
        """All ranks query their own ids only.

        At wave 0 of the ringshift loop, my_src = my_dst = my_rank. The
        DMA targets self. We construct inputs where each rank queries
        only ids owned by itself (id % W == rank), so the wave 0 traffic
        is the dominant load.
        """
        P = jax.sharding.PartitionSpec
        max_local_ids = self.MAX_LOCAL_IDS
        v_local = self.V_LOCAL
        ed = self.EMBEDDING_DIM
        world_size = self.WORLD_SIZE

        @functools.partial(
            jax.shard_map,
            mesh=self.mesh,
            in_specs=(),
            out_specs=(P("x"), P("x")),
            check_vma=False,
        )
        def make_inputs():
            rank = jax.lax.axis_index("x")
            key = jax.random.fold_in(jax.random.PRNGKey(101), rank)
            row_in_owner = jax.random.randint(
                key, (max_local_ids,), 0, v_local, dtype=jnp.int32
            )
            # ids = rank + W * row, so id % W = rank, owner = self.
            ids = (rank + world_size * row_in_owner).astype(jnp.int32)
            count = jnp.array(max_local_ids, dtype=jnp.int32)
            return ids[None, :], count[None]

        # pyrefly: ignore [missing-argument, not-iterable]
        local_ids, local_id_count = make_inputs()
        local_table = _make_random_table(
            self.mesh,
            world_size,
            v_local,
            ed,
            seed=7,
        )
        target = _make_random_target(
            self.mesh,
            world_size,
            max_local_ids,
            ed,
            seed=13,
        )
        jax.block_until_ready((local_ids, local_id_count, local_table, target))
        d = self._check_grad_close(
            "W4_SELFPEER",
            local_ids,
            local_id_count,
            local_table,
            target,
        )
        self._log("W4_SELFPEER", f"PASS max_diff={d:.6e}")
        self.assertGreaterEqual(d, 0.0)

    # ---- 4. zero count (degenerate) ----

    def test_w4_zero_count(self) -> None:
        """All ranks have local_id_count = 0.

        Every wave's p1_cnt and p3_cnt should be 0; all `pl.when` guards
        skip; the kernel completes with zero grad_table.
        """
        P = jax.sharding.PartitionSpec
        max_local_ids = self.MAX_LOCAL_IDS
        v_local = self.V_LOCAL
        ed = self.EMBEDDING_DIM
        world_size = self.WORLD_SIZE

        @functools.partial(
            jax.shard_map,
            mesh=self.mesh,
            in_specs=(),
            out_specs=(P("x"), P("x")),
            check_vma=False,
        )
        def make_inputs():
            ids = jnp.zeros((max_local_ids,), dtype=jnp.int32)
            count = jnp.array(0, dtype=jnp.int32)
            return ids[None, :], count[None]

        # pyrefly: ignore [missing-argument, not-iterable]
        local_ids, local_id_count = make_inputs()
        local_table = _make_random_table(
            self.mesh,
            world_size,
            v_local,
            ed,
            seed=7,
        )
        target = _make_random_target(
            self.mesh,
            world_size,
            max_local_ids,
            ed,
            seed=13,
        )
        jax.block_until_ready((local_ids, local_id_count, local_table, target))
        d = self._check_grad_close(
            "W4_ZEROCNT",
            local_ids,
            local_id_count,
            local_table,
            target,
        )
        self._log("W4_ZEROCNT", f"PASS max_diff={d:.6e}")
        # zero-count -> zero gradient on both sides; max_diff must be exactly 0.
        self.assertEqual(d, 0.0)

    # ---- 5. non-tile-aligned count ----

    def test_w4_non_tile_aligned_count(self) -> None:
        """Per-peer counts not multiples of TILE.

        Construct deterministic inputs where ``send_counts_real[peer]`` is
        not a multiple of TILE for at least one peer. The 28+ slack rows
        added by ``_pad_per_peer_2d`` MUST be zero-filled so they don't
        contribute to the gradient sum on the receiver side. We use 100
        ids per rank (not divisible by TILE=128); after bucketize +
        round-up, slack rows must be zero.
        """
        P = jax.sharding.PartitionSpec
        max_local_ids = self.MAX_LOCAL_IDS  # static buffer is still 1024
        v_local = self.V_LOCAL
        ed = self.EMBEDDING_DIM
        world_size = self.WORLD_SIZE

        @functools.partial(
            jax.shard_map,
            mesh=self.mesh,
            in_specs=(),
            out_specs=(P("x"), P("x")),
            check_vma=False,
        )
        def make_inputs():
            rank = jax.lax.axis_index("x")
            key = jax.random.fold_in(jax.random.PRNGKey(202), rank)
            valid = 100  # not divisible by 128, so per-peer counts won't be
            u_owner, u_row = jax.random.split(key, 2)
            owner = jax.random.randint(u_owner, (valid,), 0, world_size)
            row_in_owner = jax.random.randint(u_row, (valid,), 0, v_local)
            ids_real = (owner + world_size * row_in_owner).astype(jnp.int32)
            ids = jnp.zeros((max_local_ids,), dtype=jnp.int32).at[:valid].set(ids_real)
            count = jnp.array(valid, dtype=jnp.int32)
            return ids[None, :], count[None]

        # pyrefly: ignore [missing-argument, not-iterable]
        local_ids, local_id_count = make_inputs()
        local_table = _make_random_table(
            self.mesh,
            world_size,
            v_local,
            ed,
            seed=7,
        )
        target = _make_random_target(
            self.mesh,
            world_size,
            max_local_ids,
            ed,
            seed=13,
        )
        jax.block_until_ready((local_ids, local_id_count, local_table, target))
        d = self._check_grad_close(
            "W4_NONTILE",
            local_ids,
            local_id_count,
            local_table,
            target,
        )
        self._log("W4_NONTILE", f"PASS max_diff={d:.6e}")
        self.assertGreaterEqual(d, 0.0)

    # ---- 6. jit(value_and_grad(loss)) composition ----

    def test_w4_jit_value_and_grad(self) -> None:
        """End-to-end ``jit(value_and_grad(loss))`` works through custom_vjp.

        Verifies the custom_vjp registration composes correctly with
        ``jit + value_and_grad``, including return of (loss_value, grad).
        """
        local_ids, local_id_count = _make_skewed_inputs(
            self.mesh,
            self.WORLD_SIZE,
            self.MAX_LOCAL_IDS,
            self.V_LOCAL,
            seed=3,
        )
        local_table = _make_random_table(
            self.mesh,
            self.WORLD_SIZE,
            self.V_LOCAL,
            self.EMBEDDING_DIM,
            seed=8,
        )
        target = _make_random_target(
            self.mesh,
            self.WORLD_SIZE,
            self.MAX_LOCAL_IDS,
            self.EMBEDDING_DIM,
            seed=14,
        )
        jax.block_until_ready((local_ids, local_id_count, local_table, target))

        fused_loss = _fused_loss_for_table(
            mesh=self.mesh,
            axis_name="x",
            recv_capacity=self.recv_capacity,
            splitter="globalpad",
        )
        ref_loss = _ref_loss_for_table(
            mesh=self.mesh,
            axis_name="x",
        )

        # Gradient w.r.t. table only (argnums=2). All other args are
        # passed via the call signature so multi-host jit works.
        fused_vag_jit = jax.jit(jax.value_and_grad(fused_loss, argnums=2))
        ref_vag_jit = jax.jit(jax.value_and_grad(ref_loss, argnums=2))

        self._log("W4_VAG", "computing fused value_and_grad ...")
        f_val, f_grad = fused_vag_jit(local_ids, local_id_count, local_table, target)
        jax.block_until_ready((f_val, f_grad))

        self._log("W4_VAG", "computing ref value_and_grad ...")
        r_val, r_grad = ref_vag_jit(local_ids, local_id_count, local_table, target)
        jax.block_until_ready((r_val, r_grad))

        f_val_v = float(jax.device_get(f_val))
        r_val_v = float(jax.device_get(r_val))
        diff_v = abs(f_val_v - r_val_v)
        self._log(
            "W4_VAG",
            f"loss: fused={f_val_v:.6e} ref={r_val_v:.6e} diff={diff_v:.6e}",
        )
        # Loss tolerance: it's a single sum-reduction -> bigger than
        # per-element grad tolerance. Use 1e-2 absolute (8K * 128 * O(1)).
        np.testing.assert_allclose(f_val_v, r_val_v, atol=1e-2, rtol=1e-4)

        # Gradient tolerance: same as element-wise.
        diff_max = float(jax.device_get(jnp.max(jnp.abs(f_grad - r_grad))))
        self._log("W4_VAG", f"grad max_abs_diff = {diff_max:.6e}")
        np.testing.assert_allclose(
            np.asarray(f_grad),
            np.asarray(r_grad),
            atol=self.ATOL,
            rtol=self.RTOL,
            err_msg=f"[W4_VAG] grad mismatch: max_abs_diff={diff_max:.6e}",
        )
        self._log("W4_VAG", f"PASS loss_diff={diff_v:.6e} grad_max={diff_max:.6e}")
        self.assertLessEqual(diff_max, self.ATOL + self.RTOL)


# ---------------------------------------------------------------------------
# Multi-host (W=32) test: shapes match production W=32 correctness benchmark.
# ---------------------------------------------------------------------------


@unittest.skipUnless(_has_tpu(), "Backward TPU tests require real TPU")
class RingshiftBwdW32Test(unittest.TestCase):
    """W=32 backward correctness on a 32-host TPU-v7x slice.

    Uses ``devices[::8]`` (1 device per host across all 32 hosts) -- same
    layout as the existing production W=32 forward correctness test.
    Shapes match (D=128, v_local=8K, max_local_ids=8K).
    """

    def test_w32_correctness(self) -> None:
        world_size = 32
        TILE = 128
        embedding_dim = 128
        v_local = 8 * 1024
        max_local_ids = 8 * 1024
        atol = 1e-4
        rtol = 1e-4

        all_devices = jax.devices()
        if len(all_devices) < 256:
            self.skipTest(f"Need 256 logical devices, have {len(all_devices)}")
        devices = all_devices[::8]  # 1 device per host
        self.assertEqual(
            len(devices),
            world_size,
            f"expected {world_size}, got {len(devices)}",
        )
        mesh = jax.sharding.Mesh(np.array(devices), ("x",))

        recv_capacity = max_local_ids + TILE
        recv_capacity = ((recv_capacity + TILE - 1) // TILE) * TILE

        def log(msg):
            if jax.process_index() == 0:
                print(
                    f"[BWD_W32] {msg}",
                    file=sys.stderr,
                    flush=True,
                )

        log(
            f"=== W={world_size} D={embedding_dim} v_local={v_local} "
            f"max_local_ids={max_local_ids} ==="
        )

        log("creating inputs ...")
        local_ids, local_id_count = _make_skewed_inputs(
            mesh,
            world_size,
            max_local_ids,
            v_local,
            seed=2,
        )
        local_table = _make_random_table(
            mesh,
            world_size,
            v_local,
            embedding_dim,
            seed=7,
        )
        target = _make_random_target(
            mesh,
            world_size,
            max_local_ids,
            embedding_dim,
            seed=13,
        )
        jax.block_until_ready((local_ids, local_id_count, local_table, target))
        log("inputs ready")

        fused_loss = _fused_loss_for_table(
            mesh=mesh,
            axis_name="x",
            recv_capacity=recv_capacity,
            splitter="globalpad",
        )
        ref_loss = _ref_loss_for_table(
            mesh=mesh,
            axis_name="x",
        )

        # Gradient w.r.t. table only (argnums=2). All other args passed as
        # arguments -- required to avoid "Closing over jax.Array that
        # spans non-addressable devices" error in multi-host jit.
        fused_grad_jit = jax.jit(jax.grad(fused_loss, argnums=2))
        ref_grad_jit = jax.jit(jax.grad(ref_loss, argnums=2))

        log("-- computing reference (ra2a) grad --")
        sync_global_devices("bwd_ref_start")
        ref_grad = ref_grad_jit(local_ids, local_id_count, local_table, target)
        jax.block_until_ready(ref_grad)
        log("ref grad ready")

        log("-- computing fused (ringshift custom_vjp) grad --")
        sync_global_devices("bwd_fused_start")
        fused_grad = fused_grad_jit(local_ids, local_id_count, local_table, target)
        jax.block_until_ready(fused_grad)
        log("fused grad ready")

        # All reductions in-graph: jnp.max produces a replicated scalar,
        # safe to fetch on every host. Avoid np.asarray() on the full
        # sharded grad arrays -- that would trigger the
        # "non-addressable devices" runtime error in multi-host.
        diff_max = jnp.max(jnp.abs(fused_grad - ref_grad))
        ref_max = jnp.max(jnp.abs(ref_grad))
        # Combined atol+rtol per-element bound:  atol + rtol * |ref|.
        # max(|fused - ref| - (atol + rtol * |ref|)) > 0 means a violation.
        violation = jnp.max(
            jnp.abs(fused_grad - ref_grad) - (atol + rtol * jnp.abs(ref_grad))
        )
        diff_max_v = float(jax.device_get(diff_max))
        ref_max_v = float(jax.device_get(ref_max))
        violation_v = float(jax.device_get(violation))
        sync_global_devices("bwd_diff_done")

        log(f"max_abs_diff = {diff_max_v:.6e}, ref_max = {ref_max_v:.6e}")
        if ref_max_v > 0:
            rel = diff_max_v / max(ref_max_v, 1e-12)
            log(f"relative max_diff = {rel:.6e}")
        log(f"violation_max (>0 means atol/rtol failure) = {violation_v:.6e}")

        if violation_v <= 0:
            log(f"PASS: all elements within atol={atol:.0e} + rtol={rtol:.0e}*|ref|")
        else:
            log(f"FAIL: max violation {violation_v:.6e} > 0")
        log(f"=== RESULT: max_diff={diff_max_v:.6e} violation={violation_v:.6e} ===")

        # In-graph assertion: the worst per-element violation must be <=0.
        self.assertLessEqual(
            violation_v,
            0.0,
            f"[BWD_W32] tolerance violation: max_abs_diff={diff_max_v:.6e}, "
            f"ref_max={ref_max_v:.6e}, violation={violation_v:.6e}",
        )


# ---------------------------------------------------------------------------
# Multi-iter forward + forward+backward perf bench at W=32/64/128.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    unittest.main()
