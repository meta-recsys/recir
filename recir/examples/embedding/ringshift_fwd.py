# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Real-TPU forward correctness + perf bench for ``ringshift_embedding_lookup``.

Two ``unittest.TestCase`` classes:

  * ``RingshiftFwdCorrectnessTest`` -- bitwise-identical correctness vs ra2a
    reference at W=4/32/64/128. Asserts ``max_diff == 0.0``.
  * ``RingshiftFwdPerfTest`` -- multi-iter wall-clock measurement of fused
    ringshift (``splitter='globalpad'``) vs the
    ``ringshift_embedding_lookup_reference_jax`` (``lax.ragged_all_to_all``)
    baseline at W=32/64/128. Reports per-iter elapsed milliseconds.

Each ``test_w{N}_*`` method targets a different mesh size:
  W=4   -- single host (4 chips)
  W=32  -- 32 hosts x 1 chip per host (devices[::8])
  W=64  -- 32 hosts x 2 chips per host (devices[::4])
  W=128 -- 32 hosts x 4 chips per host (devices[::2])

These are intended to run under the JobSet manifests in ``manifests/`` --
each manifest invokes one method on a TPU-v7x topology sized for that W.
"""

from __future__ import annotations

import functools
import sys
import time
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.multihost_utils import sync_global_devices
from recir.ops.pallas.embeddings.ringshift import (
    ringshift_embedding_lookup,
    ringshift_embedding_lookup_reference_jax,
)


def _has_tpu() -> bool:
    """Return True if real TPU devices are available."""
    try:
        return any(d.platform == "tpu" for d in jax.devices())
    except Exception:
        return False


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


def _make_table_zeros_on_device(mesh, v_local, embedding_dim):
    """Per-rank table of zeros (content irrelevant for perf measurement)."""
    P = jax.sharding.PartitionSpec

    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(),
        out_specs=P("x"),
        check_vma=False,
    )
    def make_table():
        return jnp.zeros((1, v_local, embedding_dim), dtype=jnp.float32)

    return make_table()


@unittest.skipUnless(_has_tpu(), "Requires real TPU")
class RingshiftFwdCorrectnessTest(unittest.TestCase):
    """Bitwise-identical correctness vs ra2a reference at W=4/32/64/128."""

    def _run_correctness_with_devices(self, world_size, devices) -> None:
        """Shared helper: run fused vs ra2a correctness check at given W.

        D=128, v_local=8K, max_local_ids=8K, recv_capacity=8K+TILE.
        Per-rank random fp32 table (different rng key per rank).
        Computes max |out - ref| in-graph; asserts exactly 0.0 (bitwise-identical).
        Prints `[CORRECTNESS_W{N}] max_diff = ...` and PASS/FAIL on rank 0.
        """
        TILE = 128
        embedding_dim = 128
        v_local = 8 * 1024
        max_local_ids = 8 * 1024

        self.assertEqual(
            len(devices),
            world_size,
            f"expected {world_size} devices, got {len(devices)}",
        )
        mesh = jax.sharding.Mesh(np.array(devices), ("x",))

        recv_capacity = max_local_ids + TILE
        recv_capacity = ((recv_capacity + TILE - 1) // TILE) * TILE

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


@unittest.skipUnless(_has_tpu(), "Requires real TPU")
class RingshiftFwdPerfTest(unittest.TestCase):
    """Multi-iter perf measurement for fused ringshift vs ra2a baseline."""

    def _run_perf_with_devices(self, world_size, devices, *, tag) -> None:
        """Shared helper: run fused vs ra2a perf bench at given W.

        D=128, v_local=8K, max_local_ids=8K, recv_capacity=8K+TILE.
        Per-rank skewed inputs (75% to hot peer); zero-filled table.
        """
        TILE = 128
        embedding_dim = 128
        v_local = 8 * 1024
        max_local_ids = 8 * 1024

        self.assertEqual(
            len(devices),
            world_size,
            f"expected {world_size} devices, got {len(devices)}",
        )
        mesh = jax.sharding.Mesh(np.array(devices), ("x",))

        recv_capacity = max_local_ids + TILE
        recv_capacity = ((recv_capacity + TILE - 1) // TILE) * TILE

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
        local_table = _make_table_zeros_on_device(mesh, v_local, embedding_dim)
        jax.block_until_ready((local_ids, local_id_count, local_table))
        args = (local_ids, local_id_count, local_table)
        log("inputs ready")

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

        # ra2a baseline
        log("-- ra2a --")
        ra2a_times = []
        for i in range(3):
            sync_global_devices(f"ra2a_iter_{i}_start")
            log(f"ra2a iter {i} pre-call")
            t0 = time.perf_counter()
            out = ra2a_jit(*args)
            log(
                f"ra2a iter {i} dispatched (after {(time.perf_counter() - t0) * 1e3:.1f}ms), blocking..."
            )
            jax.block_until_ready(out)
            elapsed_ms = (time.perf_counter() - t0) * 1e3
            ra2a_times.append(elapsed_ms)
            log(f"ra2a iter {i}: {elapsed_ms:.1f} ms total")
        sync_global_devices("ra2a_done")
        log(f"ra2a all iters: {ra2a_times}")

        # fused
        log("-- fused_rs_globalpad --")
        fused_times = []
        for i in range(3):
            sync_global_devices(f"fused_iter_{i}_start")
            log(f"fused iter {i} pre-call (before jit_fn dispatch)")
            t0 = time.perf_counter()
            out = rs_gp_jit(*args)
            dispatch_ms = (time.perf_counter() - t0) * 1e3
            log(
                f"fused iter {i} dispatched (after {dispatch_ms:.1f}ms), blocking on result..."
            )
            jax.block_until_ready(out)
            elapsed_ms = (time.perf_counter() - t0) * 1e3
            block_ms = elapsed_ms - dispatch_ms
            fused_times.append(elapsed_ms)
            log(
                f"fused iter {i}: {elapsed_ms:.1f} ms total (dispatch={dispatch_ms:.1f}, block={block_ms:.1f})"
            )
        sync_global_devices("fused_done")
        log(f"fused all iters: {fused_times}")
        log(f"=== RESULT: ra2a={ra2a_times} fused={fused_times}")

        # Bench should not silently fail -- assert all iterations completed.
        self.assertEqual(len(ra2a_times), 3)
        self.assertEqual(len(fused_times), 3)

    def test_w32_multi_iter(self) -> None:
        """W=32 multi-iter perf test (1 device per host across 32 hosts)."""
        all_devices = jax.devices()
        if len(all_devices) < 256:
            self.skipTest(f"Need 256 logical devices, have {len(all_devices)}")
        devices = all_devices[::8]
        self._run_perf_with_devices(world_size=32, devices=devices, tag="W=32")

    def test_w64_multi_iter(self) -> None:
        """W=64 multi-iter perf test (2 devices per host across 32 hosts)."""
        all_devices = jax.devices()
        if len(all_devices) < 256:
            self.skipTest(f"Need 256 logical devices, have {len(all_devices)}")
        devices = all_devices[::4]
        self._run_perf_with_devices(world_size=64, devices=devices, tag="W=64")

    def test_w128_multi_iter(self) -> None:
        """W=128 multi-iter perf test (4 devices per host across 32 hosts)."""
        all_devices = jax.devices()
        if len(all_devices) < 256:
            self.skipTest(f"Need 256 logical devices, have {len(all_devices)}")
        devices = all_devices[::2]
        self._run_perf_with_devices(world_size=128, devices=devices, tag="W=128")


if __name__ == "__main__":
    unittest.main()
