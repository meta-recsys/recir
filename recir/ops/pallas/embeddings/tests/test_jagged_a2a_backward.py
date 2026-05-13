# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Tests for the TC backward kernel (custom_vjp) of the ringshift
``jagged_a2a_embedding_lookup_f32_alltc_ringshift`` kernel.

All tests compare the gradient produced by ``jax.grad(forward_loss)`` (which
goes through the custom_vjp `_bwd`) against
``jax.grad(reference_loss)`` (which uses the auto-differentiable
``jagged_a2a_embedding_lookup_reference_jax``). Tolerances:
``atol=1e-4, rtol=1e-4`` per design v2 §10 + Risk 4.

Test coverage:

  * ``test_w4_correctness`` — base case at W=4 (skewed inputs).
  * ``test_w4_repeated_hot_row`` — all queries point to a single row
    (worst-case scatter-add collision).
  * ``test_w4_wave0_self_peer`` — ranks all query themselves; wave 0
    must handle ``my_src == my_dst == my_rank``.
  * ``test_w4_zero_count`` — ``local_id_count = 0`` on every rank.
  * ``test_w4_non_tile_aligned_count`` — per-peer counts not multiples
    of 128 to verify ``_pad_per_peer_2d`` zero-fills slack rows.
  * ``test_w4_jit_value_and_grad`` — jit(value_and_grad(loss)) explicit
    composition.
  * ``test_w32_correctness`` — multi-host scale (32 hosts).
"""

from __future__ import annotations

import functools
import sys
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from recir.ops.pallas.embeddings.jagged_a2a_lookup import (
    jagged_a2a_embedding_lookup_f32_alltc_ringshift,
    jagged_a2a_embedding_lookup_reference_jax,
)
from recir.ops.pallas.embeddings.tests.test_jagged_a2a_lookup import _has_tpu


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------
#
# We compute the gradient w.r.t. the embedding table only (the int32 IDs and
# count are non-differentiable). The "loss" is a deterministic dot product
# with a fixed target tensor — it has the same value-and-gradient
# structure as a real downstream consumer (e.g. dense layer weights) but
# produces deterministic gradients independent of any RNG.


def _fused_loss_for_table(local_ids, local_id_count, target,
                          mesh, axis_name, recv_capacity, splitter):
    """Loss = sum(forward(table) * target). Gradient w.r.t. table = grad_out
    backprop = target → custom_vjp."""
    def loss(local_table):
        out = jagged_a2a_embedding_lookup_f32_alltc_ringshift(
            local_ids, local_id_count, local_table,
            mesh=mesh, axis_name=axis_name,
            recv_capacity=recv_capacity, splitter=splitter,
        )
        return jnp.sum(out * target)
    return loss


def _ref_loss_for_table(local_ids, local_id_count, target, mesh, axis_name):
    def loss(local_table):
        out = jagged_a2a_embedding_lookup_reference_jax(
            local_ids, local_id_count, local_table,
            mesh=mesh, axis_name=axis_name,
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


def _max_diff(out, ref):
    return float(jnp.max(jnp.abs(out - ref)))


# ---------------------------------------------------------------------------
# Single-host (W=4) tests on the local TPU pod
# ---------------------------------------------------------------------------


@unittest.skipUnless(_has_tpu(), "Backward TPU tests require real TPU")
class JaggedA2ABackwardW4(unittest.TestCase):
    """W=4 backward correctness suite on a single 4-chip TPU pod.

    Uses small shapes so each test compiles + runs in seconds. Tolerances:
    ``atol=1e-4, rtol=1e-4`` (atol=1e-4 has ~10x headroom over the
    `eps * sqrt(N) * ||grad||` worst-case for hot-row scatter; rtol=1e-4
    catches any large-magnitude misses).
    """

    WORLD_SIZE = 4
    EMBEDDING_DIM = 128
    V_LOCAL = 1024
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
        self, tag, local_ids, local_id_count, local_table, target,
    ):
        """Run jit(grad(fused_loss)) and jit(grad(ref_loss)) and compare."""
        fused_loss = _fused_loss_for_table(
            local_ids, local_id_count, target,
            mesh=self.mesh, axis_name="x",
            recv_capacity=self.recv_capacity, splitter="globalpad",
        )
        ref_loss = _ref_loss_for_table(
            local_ids, local_id_count, target,
            mesh=self.mesh, axis_name="x",
        )

        fused_grad_jit = jax.jit(jax.grad(fused_loss))
        ref_grad_jit = jax.jit(jax.grad(ref_loss))

        self._log(tag, "computing fused grad ...")
        fused_grad = fused_grad_jit(local_table)
        jax.block_until_ready(fused_grad)

        self._log(tag, "computing ref grad ...")
        ref_grad = ref_grad_jit(local_table)
        jax.block_until_ready(ref_grad)

        # Compare in-graph: replicated scalar.
        diff_max = jnp.max(jnp.abs(fused_grad - ref_grad))
        ref_max = jnp.max(jnp.abs(ref_grad))
        diff_max_v = float(jax.device_get(diff_max))
        ref_max_v = float(jax.device_get(ref_max))
        self._log(
            tag, f"max_abs_diff = {diff_max_v:.6e}, ref_max = {ref_max_v:.6e}",
        )
        # Relative tolerance check (only meaningful if ref has non-trivial
        # magnitude).
        if ref_max_v > 0:
            rel = diff_max_v / max(ref_max_v, 1e-12)
            self._log(tag, f"relative max_diff = {rel:.6e}")

        np.testing.assert_allclose(
            np.asarray(fused_grad), np.asarray(ref_grad),
            atol=self.ATOL, rtol=self.RTOL,
            err_msg=f"[{tag}] grad mismatch: max_abs_diff={diff_max_v:.6e}",
        )
        return diff_max_v

    # ---- 1. Base correctness ----

    def test_w4_correctness(self) -> None:
        """W=4 base case: skewed inputs (~75% to hot peer)."""
        local_ids, local_id_count = _make_skewed_inputs(
            self.mesh, self.WORLD_SIZE, self.MAX_LOCAL_IDS, self.V_LOCAL,
            seed=2,
        )
        local_table = _make_random_table(
            self.mesh, self.WORLD_SIZE, self.V_LOCAL, self.EMBEDDING_DIM,
            seed=7,
        )
        target = _make_random_target(
            self.mesh, self.WORLD_SIZE, self.MAX_LOCAL_IDS, self.EMBEDDING_DIM,
            seed=13,
        )
        jax.block_until_ready((local_ids, local_id_count, local_table, target))
        d = self._check_grad_close(
            "W4_BASE", local_ids, local_id_count, local_table, target,
        )
        self._log("W4_BASE", f"PASS max_diff={d:.6e}")

    # ---- 2. Repeated hot row (worst-case duplicate accumulation) ----

    def test_w4_repeated_hot_row(self) -> None:
        """All queries point to global id 0 (worst-case scatter-add collision).

        Every rank's max_local_ids slots = id 0. Owner rank for id 0 is
        rank 0 → its scatter loop accumulates `world_size *
        max_local_ids` deltas into row 0 (~4K accumulations on rank 0 at
        W=4, max_local_ids=1024). Verifies the scatter-add doesn't silently
        drop or double-count any contribution.
        """
        P = jax.sharding.PartitionSpec
        max_local_ids = self.MAX_LOCAL_IDS
        v_local = self.V_LOCAL
        ed = self.EMBEDDING_DIM

        @functools.partial(
            jax.shard_map, mesh=self.mesh,
            in_specs=(), out_specs=(P("x"), P("x")),
            check_vma=False,
        )
        def make_inputs():
            ids = jnp.zeros((max_local_ids,), dtype=jnp.int32)
            count = jnp.array(max_local_ids, dtype=jnp.int32)
            return ids[None, :], count[None]

        local_ids, local_id_count = make_inputs()
        local_table = _make_random_table(
            self.mesh, self.WORLD_SIZE, v_local, ed, seed=7,
        )
        target = _make_random_target(
            self.mesh, self.WORLD_SIZE, max_local_ids, ed, seed=13,
        )
        jax.block_until_ready((local_ids, local_id_count, local_table, target))
        d = self._check_grad_close(
            "W4_HOTROW", local_ids, local_id_count, local_table, target,
        )
        self._log("W4_HOTROW", f"PASS max_diff={d:.6e}")

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
            jax.shard_map, mesh=self.mesh,
            in_specs=(), out_specs=(P("x"), P("x")),
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

        local_ids, local_id_count = make_inputs()
        local_table = _make_random_table(
            self.mesh, world_size, v_local, ed, seed=7,
        )
        target = _make_random_target(
            self.mesh, world_size, max_local_ids, ed, seed=13,
        )
        jax.block_until_ready((local_ids, local_id_count, local_table, target))
        d = self._check_grad_close(
            "W4_SELFPEER", local_ids, local_id_count, local_table, target,
        )
        self._log("W4_SELFPEER", f"PASS max_diff={d:.6e}")

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
            jax.shard_map, mesh=self.mesh,
            in_specs=(), out_specs=(P("x"), P("x")),
            check_vma=False,
        )
        def make_inputs():
            ids = jnp.zeros((max_local_ids,), dtype=jnp.int32)
            count = jnp.array(0, dtype=jnp.int32)
            return ids[None, :], count[None]

        local_ids, local_id_count = make_inputs()
        local_table = _make_random_table(
            self.mesh, world_size, v_local, ed, seed=7,
        )
        target = _make_random_target(
            self.mesh, world_size, max_local_ids, ed, seed=13,
        )
        jax.block_until_ready((local_ids, local_id_count, local_table, target))
        d = self._check_grad_close(
            "W4_ZEROCNT", local_ids, local_id_count, local_table, target,
        )
        self._log("W4_ZEROCNT", f"PASS max_diff={d:.6e}")

    # ---- 5. non-tile-aligned count ----

    def test_w4_non_tile_aligned_count(self) -> None:
        """Per-peer counts not multiples of TILE.

        Construct deterministic inputs where ``send_counts_real[peer]`` is
        not a multiple of TILE for at least one peer. The 28+ slack rows
        added by `_pad_per_peer_2d` MUST be zero-filled so they don't
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
            jax.shard_map, mesh=self.mesh,
            in_specs=(), out_specs=(P("x"), P("x")),
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

        local_ids, local_id_count = make_inputs()
        local_table = _make_random_table(
            self.mesh, world_size, v_local, ed, seed=7,
        )
        target = _make_random_target(
            self.mesh, world_size, max_local_ids, ed, seed=13,
        )
        jax.block_until_ready((local_ids, local_id_count, local_table, target))
        d = self._check_grad_close(
            "W4_NONTILE", local_ids, local_id_count, local_table, target,
        )
        self._log("W4_NONTILE", f"PASS max_diff={d:.6e}")

    # ---- 6. jit(value_and_grad(loss)) composition ----

    def test_w4_jit_value_and_grad(self) -> None:
        """End-to-end ``jit(value_and_grad(loss))`` works through custom_vjp.

        Verifies the custom_vjp registration composes correctly with
        `jit + value_and_grad`, including return of (loss_value, grad).
        """
        local_ids, local_id_count = _make_skewed_inputs(
            self.mesh, self.WORLD_SIZE, self.MAX_LOCAL_IDS, self.V_LOCAL,
            seed=3,
        )
        local_table = _make_random_table(
            self.mesh, self.WORLD_SIZE, self.V_LOCAL, self.EMBEDDING_DIM,
            seed=8,
        )
        target = _make_random_target(
            self.mesh, self.WORLD_SIZE, self.MAX_LOCAL_IDS, self.EMBEDDING_DIM,
            seed=14,
        )
        jax.block_until_ready((local_ids, local_id_count, local_table, target))

        fused_loss = _fused_loss_for_table(
            local_ids, local_id_count, target,
            mesh=self.mesh, axis_name="x",
            recv_capacity=self.recv_capacity, splitter="globalpad",
        )
        ref_loss = _ref_loss_for_table(
            local_ids, local_id_count, target,
            mesh=self.mesh, axis_name="x",
        )

        fused_vag_jit = jax.jit(jax.value_and_grad(fused_loss))
        ref_vag_jit = jax.jit(jax.value_and_grad(ref_loss))

        self._log("W4_VAG", "computing fused value_and_grad ...")
        f_val, f_grad = fused_vag_jit(local_table)
        jax.block_until_ready((f_val, f_grad))

        self._log("W4_VAG", "computing ref value_and_grad ...")
        r_val, r_grad = ref_vag_jit(local_table)
        jax.block_until_ready((r_val, r_grad))

        f_val_v = float(jax.device_get(f_val))
        r_val_v = float(jax.device_get(r_val))
        diff_v = abs(f_val_v - r_val_v)
        self._log(
            "W4_VAG",
            f"loss: fused={f_val_v:.6e} ref={r_val_v:.6e} diff={diff_v:.6e}",
        )
        # Loss tolerance: it's a single sum-reduction → bigger than
        # per-element grad tolerance. Use 1e-2 absolute (8K * 128 * O(1)).
        np.testing.assert_allclose(f_val_v, r_val_v, atol=1e-2, rtol=1e-4)

        # Gradient tolerance: same as element-wise.
        diff_max = float(jax.device_get(jnp.max(jnp.abs(f_grad - r_grad))))
        self._log("W4_VAG", f"grad max_abs_diff = {diff_max:.6e}")
        np.testing.assert_allclose(
            np.asarray(f_grad), np.asarray(r_grad),
            atol=self.ATOL, rtol=self.RTOL,
            err_msg=f"[W4_VAG] grad mismatch: max_abs_diff={diff_max:.6e}",
        )
        self._log("W4_VAG", f"PASS loss_diff={diff_v:.6e} grad_max={diff_max:.6e}")


# ---------------------------------------------------------------------------
# Multi-host (W=32) test: shapes match production W=32 correctness benchmark.
# ---------------------------------------------------------------------------


@unittest.skipUnless(_has_tpu(), "Backward TPU tests require real TPU")
class JaggedA2ABackwardW32(unittest.TestCase):
    """W=32 backward correctness on the recir-v7-128 cluster.

    Uses ``devices[::8]`` (1 device per host across all 32 hosts) — same
    layout as the existing production W=32 forward correctness test.
    Shapes match (D=128, v_local=8K, max_local_ids=8K).
    """

    def test_w32_correctness(self) -> None:
        from jax.experimental.multihost_utils import sync_global_devices

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
        assert len(devices) == world_size, (
            f"expected {world_size}, got {len(devices)}"
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

        log(f"=== W={world_size} D={embedding_dim} v_local={v_local} "
            f"max_local_ids={max_local_ids} ===")

        log("creating inputs ...")
        local_ids, local_id_count = _make_skewed_inputs(
            mesh, world_size, max_local_ids, v_local, seed=2,
        )
        local_table = _make_random_table(
            mesh, world_size, v_local, embedding_dim, seed=7,
        )
        target = _make_random_target(
            mesh, world_size, max_local_ids, embedding_dim, seed=13,
        )
        jax.block_until_ready((local_ids, local_id_count, local_table, target))
        log("inputs ready")

        fused_loss = _fused_loss_for_table(
            local_ids, local_id_count, target,
            mesh=mesh, axis_name="x",
            recv_capacity=recv_capacity, splitter="globalpad",
        )
        ref_loss = _ref_loss_for_table(
            local_ids, local_id_count, target,
            mesh=mesh, axis_name="x",
        )

        fused_grad_jit = jax.jit(jax.grad(fused_loss))
        ref_grad_jit = jax.jit(jax.grad(ref_loss))

        log("-- computing reference (ra2a) grad --")
        sync_global_devices("bwd_ref_start")
        ref_grad = ref_grad_jit(local_table)
        jax.block_until_ready(ref_grad)
        log("ref grad ready")

        log("-- computing fused (ringshift custom_vjp) grad --")
        sync_global_devices("bwd_fused_start")
        fused_grad = fused_grad_jit(local_table)
        jax.block_until_ready(fused_grad)
        log("fused grad ready")

        diff_max = jnp.max(jnp.abs(fused_grad - ref_grad))
        ref_max = jnp.max(jnp.abs(ref_grad))
        diff_max_v = float(jax.device_get(diff_max))
        ref_max_v = float(jax.device_get(ref_max))
        sync_global_devices("bwd_diff_done")

        log(f"max_abs_diff = {diff_max_v:.6e}, ref_max = {ref_max_v:.6e}")
        if ref_max_v > 0:
            rel = diff_max_v / max(ref_max_v, 1e-12)
            log(f"relative max_diff = {rel:.6e}")

        if diff_max_v < atol:
            log(f"PASS: max_diff {diff_max_v:.6e} < atol {atol:.0e}")
        else:
            log(f"FAIL: max_diff {diff_max_v:.6e} >= atol {atol:.0e}")
        log(f"=== RESULT: max_diff={diff_max_v:.6e} ===")

        # numpy-compatible per-element tolerance check (also enforces rtol).
        np.testing.assert_allclose(
            np.asarray(fused_grad), np.asarray(ref_grad),
            atol=atol, rtol=rtol,
            err_msg=(
                f"[BWD_W32] grad mismatch: max_abs_diff={diff_max_v:.6e}, "
                f"ref_max={ref_max_v:.6e}"
            ),
        )


if __name__ == "__main__":
    unittest.main()
