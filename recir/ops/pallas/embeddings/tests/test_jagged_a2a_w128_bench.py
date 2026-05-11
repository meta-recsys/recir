# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""W=128 bench for the jagged-A2A + sharded embedding lookup variants.

Targets the largest scale where the FUSED kernels (alltc + ringshift) are
viable — they require the local table AND the per-wave gathered embeddings to
fit in VMEM (~64 MB cap per allocation on v7x). The user-requested config
(D=512, v_local=1M, max_local_ids=256k) blows VMEM by ~10× for those
variants. This bench picks three fused-feasible (D, v_local, max_local_ids)
points and runs all four scalable variants apples-to-apples per config so we
get scaling curves toward W=512 without invalidating fused vs. v23/ra2a
comparisons by varying anything else.

Variants benched (all viable at W=128 for these configs):
  * v23_split          (3-call SC-gather pipeline; production default)
  * ra2a_fallback      (pure JAX ragged_all_to_all baseline)
  * fused_wo4          (single fused alltc f32 + WO#4 even-split splitter)
  * fused_rs_globalpad (single fused ringshift f32 + globalpad splitter)

Skipped at W=128:
  * fused_rs_naive: Python-unrolled W² ``for r,w`` array construction
    intractable above W~64.

Inputs and per-rank table are constructed on-device via shard_map closures —
no host process ever materializes the full (W, v_local, D) tensor.
"""

from __future__ import annotations

import functools
import statistics
import time
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.multihost_utils import process_allgather, sync_global_devices

from recir.ops.pallas.embeddings.jagged_a2a_lookup import (
    jagged_a2a_embedding_lookup,
    jagged_a2a_embedding_lookup_f32_alltc,
    jagged_a2a_embedding_lookup_f32_alltc_ringshift,
    jagged_a2a_embedding_lookup_f32_alltc_ringshift_chunked,
    jagged_a2a_embedding_lookup_reference_jax,
)
from recir.ops.pallas.embeddings.tests.test_jagged_a2a_lookup import _has_tpu


def _peak_hbm_bytes() -> int | None:
    try:
        stats = jax.devices()[0].memory_stats()
    except Exception:
        return None
    if stats is None:
        return None
    for k in ("peak_bytes_in_use", "peak_pool_bytes", "bytes_in_use"):
        if k in stats:
            return int(stats[k])
    return None


def _make_inputs_on_device(mesh, world_size, max_local_ids, v_local, seed):
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


def _make_table_on_device(mesh, v_local, embedding_dim):
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
class JaggedA2AW128Benchmark(unittest.TestCase):
    """W=128 apples-to-apples bench across viable variants + sizes."""

    def test_w128_sweep(self) -> None:
        world_size = 128
        TILE = 128
        embedding_dim = 128
        # v_local capped at 8k so the kernel's `table_vmem` (v_local * D * 4
        # = 4 MB) plus `zero_emb_vmem` (recv_capacity * D * 4) leave VMEM
        # headroom under the v7x ~64 MB cap. Larger v_local needs a redesign
        # to gather from HBM (like the v23 SC pipeline) instead of staging
        # the table in VMEM.
        v_local = 8 * 1024

        # Three fused-feasible sizes. recv_capacity is set to the worst-case
        # bound: max_local_ids + W*TILE rounded up to TILE. zero_emb_vmem at
        # D=128 caps at ~120K rows (~62 MB) under the v7x VMEM 64 MB limit;
        # max max_local_ids ≈ 100K at W=128 keeps headroom.
        configs = [
            dict(name="S", max_local_ids=16 * 1024),
            dict(name="M", max_local_ids=32 * 1024),
            dict(name="L", max_local_ids=64 * 1024),
        ]

        devices = jax.devices()
        if len(devices) < world_size:
            self.skipTest(f"Need {world_size} devices, have {len(devices)}")
        devices = devices[:world_size]
        mesh = jax.sharding.Mesh(np.array(devices), ("x",))

        all_rows = []
        for cfg in configs:
            name = cfg["name"]
            max_local_ids = cfg["max_local_ids"]
            # Ring-shift single-slot recv layout: per-wave incoming = ONE
            # peer's payload only (<= max_local_ids + TILE - 1). The W*TILE
            # factor was leftover from v23/wo4 paths whose all-W-peers-per-
            # wave inner loops legitimately need it.
            recv_capacity = max_local_ids + TILE
            recv_capacity = ((recv_capacity + TILE - 1) // TILE) * TILE

            print(
                f"\n=== Config {name}: W={world_size} max_local_ids={max_local_ids} "
                f"D={embedding_dim} v_local={v_local} recv_capacity={recv_capacity}",
                flush=True,
            )

            t0 = time.perf_counter()
            local_ids, local_id_count = _make_inputs_on_device(
                mesh, world_size, max_local_ids, v_local, seed=2,
            )
            local_table = _make_table_on_device(mesh, v_local, embedding_dim)
            jax.block_until_ready((local_ids, local_id_count, local_table))
            print(f"  inputs ready ({(time.perf_counter() - t0):.1f}s)", flush=True)

            args = (local_ids, local_id_count, local_table)

            ra2a_jit = jax.jit(functools.partial(
                jagged_a2a_embedding_lookup_reference_jax,
                mesh=mesh, axis_name="x",
            ))
            rs_gp_jit = jax.jit(functools.partial(
                jagged_a2a_embedding_lookup_f32_alltc_ringshift,
                mesh=mesh, axis_name="x",
                recv_capacity=recv_capacity, interpret=False,
                splitter="globalpad",
            ))

            # v23_split skipped at W>=64: its 3 Pallas kernels each Python-
            # unroll a `for peer in range(W)` inner loop (~256 unrolled DMA
            # ops total at W=128) — compile time exceeds 15 min with no clear
            # ceiling. We have v23 numbers from W=4 and can extrapolate.
            #
            # fused_wo4 also skipped at W=128: its WO#4 splitter stores per-
            # wave [max_waves, W, W] traffic / offset matrices in SMEM, plus
            # other per-wave SMEM bookkeeping. At W=128 the SMEM ask exceeds
            # the 1 MB cap (4 MB allocation observed at config S).
            for impl_name, jit_fn in [
                ("ra2a_fallback", ra2a_jit),
                ("fused_rs_globalpad", rs_gp_jit),
            ]:
                print(f"  -> {impl_name}", flush=True)
                warmup = 3
                num_iters = 10
                try:
                    for i in range(warmup):
                        t0 = time.perf_counter()
                        out = jit_fn(*args)
                        jax.block_until_ready(out)
                        dt = (time.perf_counter() - t0) * 1e3
                        print(f"     [{impl_name}] warmup iter {i}: {dt:.1f}ms", flush=True)

                    latencies_ms = []
                    for _ in range(num_iters):
                        t0 = time.perf_counter()
                        out = jit_fn(*args)
                        jax.block_until_ready(out)
                        latencies_ms.append((time.perf_counter() - t0) * 1e3)

                    median = statistics.median(latencies_ms)
                    p99 = float(np.percentile(latencies_ms, 99))
                    mean = statistics.mean(latencies_ms)
                    peak_hbm = _peak_hbm_bytes()
                    hbm_str = f"{peak_hbm/1e6:.1f}MB" if peak_hbm else "n/a"
                    print(
                        f"     {impl_name}: median={median:.3f}ms p99={p99:.3f}ms "
                        f"mean={mean:.3f}ms peak_hbm={hbm_str}",
                        flush=True,
                    )
                    all_rows.append(
                        (name, impl_name, median, p99, mean, peak_hbm, "ok")
                    )
                except Exception as e:
                    print(f"     ERROR: {type(e).__name__}: {str(e)[:200]}", flush=True)
                    all_rows.append((name, impl_name, None, None, None, None, str(e)[:80]))

        print(
            f"\n=== W={world_size} BENCHMARK RESULTS (f32, skewed, D={embedding_dim}, v_local={v_local}) ===",
            flush=True,
        )
        print(
            f"{'config':<8}{'impl':<22}{'median_ms':>12}{'p99_ms':>12}"
            f"{'mean_ms':>12}{'peak_MB':>12}  status",
            flush=True,
        )
        for cfg_name, impl_name, med, p99, mean, hbm, status in all_rows:
            if med is None:
                print(
                    f"{cfg_name:<8}{impl_name:<22}{'-':>12}{'-':>12}{'-':>12}{'-':>12}  {status}",
                    flush=True,
                )
            else:
                hbm_mb = f"{hbm/1e6:.1f}" if hbm else "n/a"
                print(
                    f"{cfg_name:<8}{impl_name:<22}{med:>12.3f}{p99:>12.3f}"
                    f"{mean:>12.3f}{hbm_mb:>12}  {status}",
                    flush=True,
                )


    def test_w128_fits_config_d512(self) -> None:
        """W=128 fused ringshift bench at D=128, v_local=8K, max_local_ids=8K.

        v68s run: drops D from 512 -> 128 to give VMEM more headroom so we
        isolate the lax.fori_loop wave-loop conversion (this build) from the
        VMEM allocation issue. v68p hit VMEM OOM at D=512 (each
        [recv_capacity, D] buffer is ~17 MB; kernel needs 3 such buffers +
        table_vmem -- exceeds v7x VMEM budget). v68q debug iter at D=256
        confirmed VMEM fits but Pallas compile of the W=128 *unrolled* fused
        ringshift exceeded the 30-min job-level timeout (warmup iter 0 never
        completed) -- that compile blow-up is what the loop conversion
        targets. Bench BOTH ra2a_fallback and fused_rs_globalpad with
        cross-host sync between timed iterations so per-iter wallclock
        reflects real critical-path execution (the v68h run had half the
        pods reporting 0.004 ms -- multi-host async dispatch artifact).
        """
        world_size = 128
        TILE = 128
        embedding_dim = 128
        v_local = 8 * 1024
        max_local_ids = 8 * 1024
        # Single-slot recv layout (NO W factor): per-wave incoming = ONE
        # peer's payload only (<= max_local_ids + TILE - 1). Round up to TILE.
        recv_capacity = max_local_ids + TILE
        recv_capacity = ((recv_capacity + TILE - 1) // TILE) * TILE

        devices = jax.devices()
        if len(devices) < world_size:
            self.skipTest(f"Need {world_size} devices, have {len(devices)}")
        devices = devices[:world_size]
        mesh = jax.sharding.Mesh(np.array(devices), ("x",))

        print(
            f"\n=== test_w128_fits_config_d512: W={world_size} D={embedding_dim} "
            f"v_local={v_local} max_local_ids={max_local_ids} "
            f"recv_capacity={recv_capacity}",
            flush=True,
        )

        t0 = time.perf_counter()
        local_ids, local_id_count = _make_inputs_on_device(
            mesh, world_size, max_local_ids, v_local, seed=2,
        )
        local_table = _make_table_on_device(mesh, v_local, embedding_dim)
        jax.block_until_ready((local_ids, local_id_count, local_table))
        print(f"  inputs ready ({(time.perf_counter() - t0):.1f}s)", flush=True)

        args = (local_ids, local_id_count, local_table)

        ra2a_jit = jax.jit(functools.partial(
            jagged_a2a_embedding_lookup_reference_jax,
            mesh=mesh, axis_name="x",
        ))
        rs_gp_jit = jax.jit(functools.partial(
            jagged_a2a_embedding_lookup_f32_alltc_ringshift,
            mesh=mesh, axis_name="x",
            recv_capacity=recv_capacity, interpret=False,
            splitter="globalpad",
        ))

        all_rows = []
        for impl_name, jit_fn in [
            ("ra2a_fallback", ra2a_jit),
            ("fused_rs_globalpad", rs_gp_jit),
        ]:
            print(f"  -> {impl_name}", flush=True)
            warmup = 3
            num_iters = 10
            try:
                for i in range(warmup):
                    t0 = time.perf_counter()
                    out = jit_fn(*args)
                    jax.block_until_ready(out)
                    dt = (time.perf_counter() - t0) * 1e3
                    print(
                        f"     [{impl_name}] warmup iter {i}: {dt:.1f}ms",
                        flush=True,
                    )

                # Cross-host barrier before timed iters so every host enters
                # the loop in lockstep.
                sync_global_devices(f"bench_warmup_done_{impl_name}")

                latencies_ms = []
                for it in range(num_iters):
                    sync_global_devices(f"bench_iter_start_{impl_name}_{it}")
                    t0 = time.perf_counter()
                    out = jit_fn(*args)
                    jax.block_until_ready(out)
                    sync_global_devices(f"bench_iter_end_{impl_name}_{it}")
                    latencies_ms.append((time.perf_counter() - t0) * 1e3)

                median = statistics.median(latencies_ms)
                p99 = float(np.percentile(latencies_ms, 99))
                mean = statistics.mean(latencies_ms)
                peak_hbm = _peak_hbm_bytes()
                hbm_str = f"{peak_hbm/1e6:.1f}MB" if peak_hbm else "n/a"
                print(
                    f"     {impl_name}: median={median:.3f}ms p99={p99:.3f}ms "
                    f"mean={mean:.3f}ms peak_hbm={hbm_str}",
                    flush=True,
                )
                print(
                    f"     {impl_name} per-iter: {[f'{x:.3f}' for x in latencies_ms]}",
                    flush=True,
                )
                all_rows.append(
                    (impl_name, median, p99, mean, peak_hbm, "ok")
                )
            except Exception as e:
                print(
                    f"     ERROR: {type(e).__name__}: {str(e)[:300]}",
                    flush=True,
                )
                all_rows.append(
                    (impl_name, None, None, None, None, str(e)[:80])
                )

        print(
            f"\n=== W={world_size} FITS_D512 BENCHMARK RESULTS "
            f"(D={embedding_dim}, v_local={v_local}, max_local_ids={max_local_ids}, "
            f"recv_capacity={recv_capacity}) ===",
            flush=True,
        )
        print(
            f"{'impl':<22}{'median_ms':>12}{'p99_ms':>12}{'mean_ms':>12}"
            f"{'peak_MB':>12}  status",
            flush=True,
        )
        for impl_name, med, p99, mean, hbm, status in all_rows:
            if med is None:
                print(
                    f"{impl_name:<22}{'-':>12}{'-':>12}{'-':>12}{'-':>12}  {status}",
                    flush=True,
                )
            else:
                hbm_mb = f"{hbm/1e6:.1f}" if hbm else "n/a"
                print(
                    f"{impl_name:<22}{med:>12.3f}{p99:>12.3f}{mean:>12.3f}"
                    f"{hbm_mb:>12}  {status}",
                    flush=True,
                )


    def _run_single_invocation_at_full_world_size(self, *, embedding_dim, max_local_ids):
        """Single-invocation correctness at W = jax.device_count() (full mesh).

        Using ALL logical TC devices on the pod (256 on recir-v7-128 = 32
        hosts x 8 logical/host) makes every JAX host participate in the
        kernel mesh. This eliminates the prior asymmetric-mesh issue at
        W=128 where 16 of 32 hosts owned 0 mesh devices and broke any
        cross-host primitive (process_allgather, sync_global_devices).

        With a fully-replicated reduction (jnp.max(jnp.abs(out - ref)))
        every host can device_get the resulting scalar locally -- no
        process_allgather required.
        """
        world_size = jax.device_count()  # = 256 on recir-v7-128
        TILE = 128
        v_local = 8 * 1024

        devices = jax.devices()
        if len(devices) < world_size:
            self.skipTest(f"Need {world_size} devices, have {len(devices)}")
        mesh = jax.sharding.Mesh(np.array(devices), ("x",))

        recv_capacity = max_local_ids + TILE
        recv_capacity = ((recv_capacity + TILE - 1) // TILE) * TILE

        if jax.process_index() == 0:
            print(f"=== W={world_size} D={embedding_dim} v_local={v_local} max_local_ids={max_local_ids} ===", flush=True)

        # On-device input + table generation
        t0 = time.perf_counter()
        local_ids, local_id_count = _make_inputs_on_device(mesh, world_size, max_local_ids, v_local, seed=2)
        local_table = _make_table_on_device(mesh, v_local, embedding_dim)
        jax.block_until_ready((local_ids, local_id_count, local_table))
        if jax.process_index() == 0:
            print(f"  inputs ready ({time.perf_counter() - t0:.1f}s)", flush=True)

        args = (local_ids, local_id_count, local_table)

        # Reference: ra2a
        if jax.process_index() == 0:
            print("  computing ra2a reference...", flush=True)
        t0 = time.perf_counter()
        ref = jax.jit(functools.partial(
            jagged_a2a_embedding_lookup_reference_jax,
            mesh=mesh, axis_name="x",
        ))(*args)
        jax.block_until_ready(ref)
        if jax.process_index() == 0:
            print(f"  ra2a done ({time.perf_counter() - t0:.1f}s)", flush=True)

        # Fused ringshift -- ONE invocation
        if jax.process_index() == 0:
            print("  invoking fused_rs_globalpad ONCE...", flush=True)
        t0 = time.perf_counter()
        out = jax.jit(functools.partial(
            jagged_a2a_embedding_lookup_f32_alltc_ringshift,
            mesh=mesh, axis_name="x",
            recv_capacity=recv_capacity, splitter="globalpad",
        ))(*args)
        jax.block_until_ready(out)
        elapsed = time.perf_counter() - t0
        if jax.process_index() == 0:
            print(f"  fused done ({elapsed:.1f}s)", flush=True)

        # Compute max_diff using JAX-side reduction (handles sharded arrays).
        # The result is a fully-replicated scalar that every host can device_get.
        diff_max = jnp.max(jnp.abs(out - ref))
        diff_max_value = float(jax.device_get(diff_max))

        if jax.process_index() == 0:
            print(f"=== RESULT: D={embedding_dim} fused vs ra2a max_diff = {diff_max_value:.6e}", flush=True)
            print(f"=== fused wallclock = {elapsed*1000:.1f}ms (includes JIT compile)", flush=True)
            if diff_max_value < 1e-4:
                print(f"=== CORRECTNESS_OK at W={world_size} D={embedding_dim}", flush=True)
            else:
                print(f"=== CORRECTNESS_FAIL: max_diff = {diff_max_value}", flush=True)


    def test_w_full_d128(self):
        self._run_single_invocation_at_full_world_size(embedding_dim=128, max_local_ids=8 * 1024)

    def test_w_full_d256(self):
        self._run_single_invocation_at_full_world_size(embedding_dim=256, max_local_ids=4 * 1024)

    def test_w_full_d512(self):
        self._run_single_invocation_at_full_world_size(embedding_dim=512, max_local_ids=2 * 1024)


    def _run_single_invocation_w128_symmetric(self, *, embedding_dim, max_local_ids):
        """W=128 single-invocation correctness test with symmetric mesh.

        recir-v7-128 has 256 logical TC devices (32 hosts x 8 devices/host).
        Picking ``devices[::2]`` gives 128 devices spread across all 32 hosts
        (4 per host) -- symmetric, no host has zero shards. This is the right
        approach for W=128: matches the user's spec, kernel compiles fast,
        and avoids the asymmetric-mesh issue that breaks cross-host primitives.
        """
        world_size = 128
        TILE = 128
        v_local = 8 * 1024

        all_devices = jax.devices()
        if len(all_devices) < 256:
            self.skipTest(f"Need 256 logical devices for symmetric W=128, have {len(all_devices)}")
        devices = all_devices[::2]  # every other device = 4 per host on tpu7x-standard-4t
        assert len(devices) == world_size, f"Expected {world_size}, got {len(devices)}"
        mesh = jax.sharding.Mesh(np.array(devices), ("x",))

        recv_capacity = max_local_ids + TILE
        recv_capacity = ((recv_capacity + TILE - 1) // TILE) * TILE

        if jax.process_index() == 0:
            print(f"=== W={world_size} (symmetric, devices[::2] of {len(all_devices)}) D={embedding_dim} v_local={v_local} max_local_ids={max_local_ids} ===", flush=True)

        # On-device input + table generation
        t0 = time.perf_counter()
        local_ids, local_id_count = _make_inputs_on_device(mesh, world_size, max_local_ids, v_local, seed=2)
        local_table = _make_table_on_device(mesh, v_local, embedding_dim)
        jax.block_until_ready((local_ids, local_id_count, local_table))
        if jax.process_index() == 0:
            print(f"  inputs ready ({time.perf_counter() - t0:.1f}s)", flush=True)

        args = (local_ids, local_id_count, local_table)

        # Reference: ra2a
        if jax.process_index() == 0:
            print("  computing ra2a reference...", flush=True)
        t0 = time.perf_counter()
        ref = jax.jit(functools.partial(
            jagged_a2a_embedding_lookup_reference_jax,
            mesh=mesh, axis_name="x",
        ))(*args)
        jax.block_until_ready(ref)
        if jax.process_index() == 0:
            print(f"  ra2a done ({time.perf_counter() - t0:.1f}s)", flush=True)

        # Fused ringshift -- ONE invocation
        if jax.process_index() == 0:
            print("  invoking fused_rs_globalpad ONCE...", flush=True)
        t0 = time.perf_counter()
        out = jax.jit(functools.partial(
            jagged_a2a_embedding_lookup_f32_alltc_ringshift,
            mesh=mesh, axis_name="x",
            recv_capacity=recv_capacity, splitter="globalpad",
        ))(*args)
        jax.block_until_ready(out)
        elapsed = time.perf_counter() - t0
        if jax.process_index() == 0:
            print(f"  fused done ({elapsed:.1f}s)", flush=True)

        # Compute max_diff using JAX-side reduction (output is fully replicated scalar).
        diff_max = jnp.max(jnp.abs(out - ref))
        diff_max_value = float(jax.device_get(diff_max))

        if jax.process_index() == 0:
            print(f"=== RESULT: D={embedding_dim} fused vs ra2a max_diff = {diff_max_value:.6e}", flush=True)
            print(f"=== fused wallclock = {elapsed*1000:.1f}ms (includes JIT compile)", flush=True)
            if diff_max_value < 1e-4:
                print(f"=== CORRECTNESS_OK at W={world_size} D={embedding_dim}", flush=True)
            else:
                print(f"=== CORRECTNESS_FAIL: max_diff = {diff_max_value}", flush=True)


    def test_w128_sym_d128(self):
        self._run_single_invocation_w128_symmetric(embedding_dim=128, max_local_ids=8 * 1024)

    def test_w128_sym_d256(self):
        self._run_single_invocation_w128_symmetric(embedding_dim=256, max_local_ids=4 * 1024)

    def test_w128_sym_d512(self):
        self._run_single_invocation_w128_symmetric(embedding_dim=512, max_local_ids=2 * 1024)


    def test_w128_multi_iter_with_allgather_barrier(self) -> None:
        """W=128 multi-iter test with process_allgather between iters.

        Hypothesis: previous multi-iter hangs were because some rank's iter 0
        work hadn't fully completed when iter 1 was dispatched. process_allgather
        of a sentinel scalar forces every host to participate, providing a true
        cross-host barrier.
        """
        from jax.experimental.multihost_utils import process_allgather
        import time

        world_size = 128
        TILE = 128
        embedding_dim = 128
        v_local = 8 * 1024
        max_local_ids = 8 * 1024

        all_devices = jax.devices()
        if len(all_devices) < 256:
            self.skipTest(f"Need 256 logical devices, have {len(all_devices)}")
        devices = all_devices[::2]  # symmetric mesh
        mesh = jax.sharding.Mesh(np.array(devices), ("x",))

        recv_capacity = max_local_ids + TILE
        recv_capacity = ((recv_capacity + TILE - 1) // TILE) * TILE

        if jax.process_index() == 0:
            print(f"=== Multi-iter test W={world_size} D={embedding_dim} ===", flush=True)

        local_ids, local_id_count = _make_inputs_on_device(mesh, world_size, max_local_ids, v_local, seed=2)
        local_table = _make_table_on_device(mesh, v_local, embedding_dim)
        jax.block_until_ready((local_ids, local_id_count, local_table))
        args = (local_ids, local_id_count, local_table)

        rs_gp_jit = jax.jit(functools.partial(
            jagged_a2a_embedding_lookup_f32_alltc_ringshift,
            mesh=mesh, axis_name="x",
            recv_capacity=recv_capacity, splitter="globalpad",
        ))

        # Iter 0: cold (compile + first run).
        if jax.process_index() == 0:
            print("  iter 0 (cold)...", flush=True)
        t0 = time.perf_counter()
        out = rs_gp_jit(*args)
        jax.block_until_ready(out)
        iter0_ms = (time.perf_counter() - t0) * 1e3
        if jax.process_index() == 0:
            print(f"  iter 0: {iter0_ms:.1f} ms", flush=True)

        # Cross-host barrier — every host must reach here before any continues.
        sentinel = jnp.array([float(jax.process_index())], dtype=jnp.float32)
        _ = process_allgather(sentinel, tiled=False)
        if jax.process_index() == 0:
            print("  iter 0 barrier done", flush=True)

        # Iters 1-4: should hit the JIT cache and run fast (no compile).
        iter_times_ms = []
        for i in range(1, 5):
            if jax.process_index() == 0:
                print(f"  iter {i}...", flush=True)
            t0 = time.perf_counter()
            out = rs_gp_jit(*args)
            jax.block_until_ready(out)
            elapsed_ms = (time.perf_counter() - t0) * 1e3
            iter_times_ms.append(elapsed_ms)
            if jax.process_index() == 0:
                print(f"  iter {i}: {elapsed_ms:.1f} ms", flush=True)

            # Cross-host barrier between iters
            sentinel = jnp.array([float(jax.process_index())], dtype=jnp.float32)
            _ = process_allgather(sentinel, tiled=False)

        if jax.process_index() == 0:
            print(f"=== RESULT: iter 0 (cold) = {iter0_ms:.1f} ms; "
                  f"iters 1-4 (warm) = {iter_times_ms}", flush=True)
            print(f"=== If iters 1-4 succeeded, multi-iter at W=128 WORKS", flush=True)
            if iter_times_ms:
                print(f"=== steady-state median = {sorted(iter_times_ms)[len(iter_times_ms)//2]:.1f} ms", flush=True)


    def test_w32_multi_iter(self) -> None:
        """W=32 multi-iter test with SYMMETRIC mesh (devices[::8] = 1 per host).

        Detailed checkpoint prints between async dispatch and block_until_ready
        so we can pinpoint whether hangs are in compile/trace or in execution.
        """
        from jax.experimental.multihost_utils import sync_global_devices
        import sys

        world_size = 32
        TILE = 128
        embedding_dim = 128
        v_local = 8 * 1024
        max_local_ids = 8 * 1024

        all_devices = jax.devices()
        if len(all_devices) < 256:
            self.skipTest(f"Need 256 logical devices, have {len(all_devices)}")
        devices = all_devices[::8]  # 1 device per host
        assert len(devices) == world_size
        mesh = jax.sharding.Mesh(np.array(devices), ("x",))

        recv_capacity = max_local_ids + TILE
        recv_capacity = ((recv_capacity + TILE - 1) // TILE) * TILE

        def log(msg):
            if jax.process_index() == 0:
                print(f"[W=32] {msg}", file=sys.stderr, flush=True)

        log(f"=== D={embedding_dim} v_local={v_local} max_local_ids={max_local_ids} ===")

        log("creating inputs ...")
        local_ids, local_id_count = _make_inputs_on_device(mesh, world_size, max_local_ids, v_local, seed=2)
        local_table = _make_table_on_device(mesh, v_local, embedding_dim)
        jax.block_until_ready((local_ids, local_id_count, local_table))
        args = (local_ids, local_id_count, local_table)
        log("inputs ready")

        rs_gp_jit = jax.jit(functools.partial(
            jagged_a2a_embedding_lookup_f32_alltc_ringshift,
            mesh=mesh, axis_name="x",
            recv_capacity=recv_capacity, splitter="globalpad",
        ))
        ra2a_jit = jax.jit(functools.partial(
            jagged_a2a_embedding_lookup_reference_jax,
            mesh=mesh, axis_name="x",
        ))

        # ra2a baseline
        log("-- ra2a --")
        ra2a_times = []
        for i in range(3):
            sync_global_devices(f"ra2a_iter_{i}_start")
            log(f"ra2a iter {i} pre-call")
            t0 = time.perf_counter()
            out = ra2a_jit(*args)
            log(f"ra2a iter {i} dispatched (after {(time.perf_counter()-t0)*1e3:.1f}ms), blocking...")
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
            log(f"fused iter {i} dispatched (after {dispatch_ms:.1f}ms), blocking on result...")
            jax.block_until_ready(out)
            elapsed_ms = (time.perf_counter() - t0) * 1e3
            block_ms = elapsed_ms - dispatch_ms
            fused_times.append(elapsed_ms)
            log(f"fused iter {i}: {elapsed_ms:.1f} ms total (dispatch={dispatch_ms:.1f}, block={block_ms:.1f})")
        sync_global_devices("fused_done")
        log(f"fused all iters: {fused_times}")
        log(f"=== RESULT: ra2a={ra2a_times} fused={fused_times}")


    def _run_multi_iter_with_devices(self, world_size, devices):
        """Shared helper: 3-iter test of ra2a + fused with checkpoint prints
        between async dispatch and block_until_ready.

        Isolates whether a hang is in compile/trace (jit_fn(*args) returns
        slowly because it triggers JIT) vs. in execution (jit_fn(*args)
        returns fast as the cached future, then block_until_ready hangs).
        All progress prints go to stderr to bypass unittest's per-test
        stdout buffering.
        """
        import sys
        from jax.experimental.multihost_utils import sync_global_devices

        TILE = 128
        embedding_dim = 128
        v_local = 8 * 1024
        max_local_ids = 8 * 1024

        assert len(devices) == world_size, f"Expected {world_size}, got {len(devices)}"
        mesh = jax.sharding.Mesh(np.array(devices), ("x",))

        recv_capacity = max_local_ids + TILE
        recv_capacity = ((recv_capacity + TILE - 1) // TILE) * TILE

        def log(msg):
            if jax.process_index() == 0:
                print(f"[W={world_size}] {msg}", file=sys.stderr, flush=True)

        log(f"=== D={embedding_dim} v_local={v_local} max_local_ids={max_local_ids} ===")

        log("creating inputs ...")
        local_ids, local_id_count = _make_inputs_on_device(mesh, world_size, max_local_ids, v_local, seed=2)
        local_table = _make_table_on_device(mesh, v_local, embedding_dim)
        jax.block_until_ready((local_ids, local_id_count, local_table))
        args = (local_ids, local_id_count, local_table)
        log("inputs ready")

        rs_gp_jit = jax.jit(functools.partial(
            jagged_a2a_embedding_lookup_f32_alltc_ringshift,
            mesh=mesh, axis_name="x",
            recv_capacity=recv_capacity, splitter="globalpad",
        ))
        ra2a_jit = jax.jit(functools.partial(
            jagged_a2a_embedding_lookup_reference_jax,
            mesh=mesh, axis_name="x",
        ))

        log("-- ra2a --")
        ra2a_times = []
        for i in range(3):
            sync_global_devices(f"ra2a_iter_{i}_start")
            log(f"ra2a iter {i} pre-call")
            t0 = time.perf_counter()
            out = ra2a_jit(*args)
            log(f"ra2a iter {i} dispatched (after {(time.perf_counter()-t0)*1e3:.1f}ms), blocking...")
            jax.block_until_ready(out)
            elapsed_ms = (time.perf_counter() - t0) * 1e3
            ra2a_times.append(elapsed_ms)
            log(f"ra2a iter {i}: {elapsed_ms:.1f} ms")
        sync_global_devices("ra2a_done")
        log(f"ra2a all iters: {ra2a_times}")

        log("-- fused_rs_globalpad --")
        fused_times = []
        for i in range(3):
            sync_global_devices(f"fused_iter_{i}_start")
            log(f"fused iter {i} pre-call")
            t0 = time.perf_counter()
            out = rs_gp_jit(*args)
            log(f"fused iter {i} dispatched (after {(time.perf_counter()-t0)*1e3:.1f}ms), blocking...")
            jax.block_until_ready(out)
            elapsed_ms = (time.perf_counter() - t0) * 1e3
            fused_times.append(elapsed_ms)
            log(f"fused iter {i}: {elapsed_ms:.1f} ms")
        sync_global_devices("fused_done")
        log(f"fused all iters: {fused_times}")
        log(f"=== RESULT W={world_size}: ra2a={ra2a_times} fused={fused_times}")


    def test_w8_multi_iter(self):
        """W=8 using first 8 logical devices (asymmetric: 1 host's 8 cores)."""
        all_devices = jax.devices()
        if len(all_devices) < 8:
            self.skipTest(f"Need 8 devices, have {len(all_devices)}")
        devices = all_devices[:8]
        self._run_multi_iter_with_devices(world_size=8, devices=devices)


    def test_w16_multi_iter(self):
        """W=16 using first 16 logical devices (asymmetric: 2 hosts)."""
        all_devices = jax.devices()
        if len(all_devices) < 16:
            self.skipTest(f"Need 16 devices, have {len(all_devices)}")
        devices = all_devices[:16]
        self._run_multi_iter_with_devices(world_size=16, devices=devices)


    def test_w32_multi_iter_v2(self):
        """W=32 using first 32 logical devices (asymmetric: 4 hosts)."""
        all_devices = jax.devices()
        if len(all_devices) < 32:
            self.skipTest(f"Need 32 devices, have {len(all_devices)}")
        devices = all_devices[:32]
        self._run_multi_iter_with_devices(world_size=32, devices=devices)


    def test_w128_chunked_v_local(self) -> None:
        """v_local-chunked ringshift bench at the user-requested scale.

        Config: W=128, D=512, v_local=1M, max_local_ids=256K. Per-chunk
        v_local_chunk_size=8K -> 128 chunks, each kernel call sees a
        4096-row over-slice (2x safety on uniform per-chunk count).
        """
        world_size = 128
        TILE = 128
        embedding_dim = 512
        v_local = 1024 * 1024
        max_local_ids = 256 * 1024
        v_local_chunk_size = 8 * 1024
        num_chunks = (v_local + v_local_chunk_size - 1) // v_local_chunk_size
        # 2x safety factor over uniform per-chunk count (max_local_ids /
        # num_chunks = 2048), rounded up to TILE.
        chunk_max_local_ids = 4096
        # Ring-shift single-slot recv layout: per-wave incoming = ONE peer's
        # payload only (<= chunk_max_local_ids + TILE - 1).
        recv_capacity = chunk_max_local_ids + TILE
        recv_capacity = ((recv_capacity + TILE - 1) // TILE) * TILE

        devices = jax.devices()
        if len(devices) < world_size:
            self.skipTest(f"Need {world_size} devices, have {len(devices)}")
        devices = devices[:world_size]
        mesh = jax.sharding.Mesh(np.array(devices), ("x",))

        print(
            f"\n=== test_w128_chunked_v_local: W={world_size} D={embedding_dim} "
            f"v_local={v_local} max_local_ids={max_local_ids} "
            f"v_local_chunk_size={v_local_chunk_size} num_chunks={num_chunks} "
            f"chunk_max_local_ids={chunk_max_local_ids} recv_capacity={recv_capacity}",
            flush=True,
        )

        t0 = time.perf_counter()
        local_ids, local_id_count = _make_inputs_on_device(
            mesh, world_size, max_local_ids, v_local, seed=2,
        )
        local_table = _make_table_on_device(mesh, v_local, embedding_dim)
        jax.block_until_ready((local_ids, local_id_count, local_table))
        print(f"  inputs ready ({(time.perf_counter() - t0):.1f}s)", flush=True)

        args = (local_ids, local_id_count, local_table)

        chunked_jit = jax.jit(functools.partial(
            jagged_a2a_embedding_lookup_f32_alltc_ringshift_chunked,
            mesh=mesh, axis_name="x",
            v_local_chunk_size=v_local_chunk_size,
            chunk_max_local_ids=chunk_max_local_ids,
            recv_capacity=recv_capacity,
            interpret=False,
            splitter="globalpad",
        ))

        warmup = 3
        num_iters = 5
        try:
            for i in range(warmup):
                t0 = time.perf_counter()
                out = chunked_jit(*args)
                jax.block_until_ready(out)
                dt = (time.perf_counter() - t0) * 1e3
                print(
                    f"  [chunked] warmup iter {i}: {dt:.1f}ms",
                    flush=True,
                )

            latencies_ms = []
            t_total = time.perf_counter()
            for i in range(num_iters):
                t0 = time.perf_counter()
                out = chunked_jit(*args)
                jax.block_until_ready(out)
                dt = (time.perf_counter() - t0) * 1e3
                latencies_ms.append(dt)
                print(
                    f"  [chunked] measure iter {i}: {dt:.1f}ms",
                    flush=True,
                )
            total_ms = (time.perf_counter() - t_total) * 1e3

            median = statistics.median(latencies_ms)
            p99 = float(np.percentile(latencies_ms, 99))
            mean = statistics.mean(latencies_ms)
            peak_hbm = _peak_hbm_bytes()
            hbm_str = f"{peak_hbm/1e6:.1f}MB" if peak_hbm else "n/a"
            print(
                f"\n=== W={world_size} CHUNKED V_LOCAL BENCH RESULTS "
                f"(D={embedding_dim}, v_local={v_local}, max_local_ids={max_local_ids}, "
                f"v_local_chunk_size={v_local_chunk_size}) ===",
                flush=True,
            )
            print(
                f"  per-iter: {[f'{x:.1f}' for x in latencies_ms]} ms",
                flush=True,
            )
            print(
                f"  median={median:.3f}ms p99={p99:.3f}ms mean={mean:.3f}ms "
                f"peak_hbm={hbm_str} total_measure={total_ms:.1f}ms",
                flush=True,
            )
        except Exception as e:
            print(
                f"  ERROR: {type(e).__name__}: {str(e)[:500]}",
                flush=True,
            )
            raise


    def test_w32_barrier_only_litmus(self) -> None:
        """W=32 barrier-only litmus: strip everything except all-peer pre-barrier.

        If THIS hangs, the bug is in get_barrier_semaphore + collective_id at
        32-host scale. If it works, the bug is in DMA/gather/phase3.
        """
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        from jax import lax
        from jax.experimental.multihost_utils import sync_global_devices
        import sys

        world_size = 32
        max_waves = world_size
        P = jax.sharding.PartitionSpec

        all_devices = jax.devices()
        if len(all_devices) < 256:
            self.skipTest(f"Need 256 logical devices, have {len(all_devices)}")
        devices = all_devices[::8]  # symmetric: 1 per host
        assert len(devices) == world_size
        mesh = jax.sharding.Mesh(np.array(devices), ("x",))

        def log(msg):
            if jax.process_index() == 0:
                print(f"[BARRIER_LITMUS_W32] {msg}", file=sys.stderr, flush=True)

        # VMEM dim-0 must be at least one lane row (128). We size the output
        # to (128,) and use a (128,) VMEM scratch; the per-device run extracts
        # element 0 for the psum sentinel.
        OUT_LEN = 128

        def make_barrier_kernel(world_size: int):
            def kernel(dummy_in_ref, out_init_ref, out_ref, scratch):
                del out_init_ref
                my_rank = lax.axis_index("x")
                sem = pltpu.get_barrier_semaphore()
                for peer in range(world_size):
                    pl.semaphore_signal(sem, 1, device_id=(peer,),
                                        device_id_type=pltpu.DeviceIdType.MESH)
                pl.semaphore_wait(sem, world_size)
                # Fill scratch with rank, then DMA copy to HBM out_ref.
                scratch[...] = jnp.full((OUT_LEN,), my_rank, dtype=jnp.int32)
                pltpu.sync_copy(scratch, out_ref)
            return kernel

        HBM = pltpu.MemorySpace.HBM

        def per_device_run(local_dummy):
            out_acc = jnp.zeros((OUT_LEN,), dtype=jnp.int32)
            sync_dep = jnp.float32(0.0)
            for w_py in range(max_waves):
                out_acc_dep, _ = jax.lax.optimization_barrier((out_acc, sync_dep))
                kernel = make_barrier_kernel(world_size)
                (out_acc,) = pl.pallas_call(
                    kernel,
                    out_shape=[jax.ShapeDtypeStruct((OUT_LEN,), jnp.int32)],
                    grid_spec=pltpu.PrefetchScalarGridSpec(
                        num_scalar_prefetch=0,
                        in_specs=[
                            pl.BlockSpec(memory_space=HBM),
                            pl.BlockSpec(memory_space=HBM),
                        ],
                        out_specs=[pl.BlockSpec(memory_space=HBM)],
                        scratch_shapes=[
                            pltpu.VMEM((OUT_LEN,), jnp.int32),
                        ],
                    ),
                    input_output_aliases={1: 0},
                    compiler_params=pltpu.CompilerParams(collective_id=w_py),
                )(local_dummy, out_acc_dep)
                sync_dep = jax.lax.psum(out_acc[0].astype(jnp.float32), "x")
            return out_acc

        sharded = jax.shard_map(
            per_device_run, mesh=mesh,
            in_specs=(P("x"),),
            out_specs=P("x"),
            check_vma=False,
        )

        # Build a small dummy input
        dummy_per_rank = jnp.zeros((world_size, 1), dtype=jnp.int32)
        dummy_per_rank = jax.device_put(dummy_per_rank,
                                         jax.sharding.NamedSharding(mesh, P("x")))
        jax.block_until_ready(dummy_per_rank)
        log("inputs ready")

        barrier_jit = jax.jit(sharded)
        log("calling barrier kernel...")
        sync_global_devices("litmus_start")
        t0 = time.perf_counter()
        out = barrier_jit(dummy_per_rank)
        dispatch_ms = (time.perf_counter() - t0) * 1e3
        log(f"barrier dispatched (after {dispatch_ms:.1f}ms), blocking...")
        jax.block_until_ready(out)
        elapsed_ms = (time.perf_counter() - t0) * 1e3
        log(f"barrier ran in {elapsed_ms:.1f} ms total (dispatch={dispatch_ms:.1f}, block={elapsed_ms-dispatch_ms:.1f})")
        log("BARRIER_LITMUS_W32 RESULT: WORKS")

    def test_w32_barrier_plus_phase1_dma_litmus(self) -> None:
        """W=32 barrier + ONE Phase-1 ID DMA litmus.

        Builds on the v70i barrier-only litmus (which WORKS at W=32). Adds ONE
        component back: a single ring-shift Phase-1-style cross-rank ID DMA
        (each rank sends a fixed 128-int32 chunk to its (rank+1)%W neighbor)
        AFTER the all-peer pre-barrier. Wrapper is identical to the v70i
        litmus (Python `for w_py` outer loop over 32 waves + per-call
        collective_id + per-wave psum sync_dep + optimization_barrier carry).
        Same scratch_shapes pattern (pre-allocated VMEM scratch + 2 DMA sems).

        If THIS works at W=32, the barrier + single cross-rank DMA + per-call
        collective_id is fine; bug is in gather or Phase 3.

        If THIS hangs, the bug is in the Phase 1 cross-rank DMA path at
        32-host scale.
        """
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        from jax import lax
        from jax.experimental.multihost_utils import sync_global_devices
        import sys

        world_size = 32
        max_waves = world_size
        P = jax.sharding.PartitionSpec

        all_devices = jax.devices()
        if len(all_devices) < 256:
            self.skipTest(f"Need 256 logical devices, have {len(all_devices)}")
        devices = all_devices[::8]  # symmetric: 1 per host
        assert len(devices) == world_size
        mesh = jax.sharding.Mesh(np.array(devices), ("x",))

        def log(msg):
            if jax.process_index() == 0:
                print(f"[LITMUS_W32_BARRIER_DMA1] {msg}", file=sys.stderr, flush=True)

        OUT_LEN = 128
        # Per-rank input row buffer; shape (128*world_size,) int32 sharded along axis 0.
        # This puts (128,) on each rank in its local HBM, matching the chunk size used
        # by the per-wave fixed DMA below.
        FIXED_COUNT = 128

        def make_barrier_dma_kernel(world_size: int):
            def kernel(sorted_row_ref, dummy_in_ref, out_init_ref,
                       out_ref, recv_row_buf,
                       scratch_vmem, send_sem, recv_sem):
                del dummy_in_ref, out_init_ref
                my_rank = lax.axis_index("x")

                # Pre-barrier (proven to work in v70i barrier-only litmus).
                sem = pltpu.get_barrier_semaphore()
                for peer in range(world_size):
                    pl.semaphore_signal(sem, 1, device_id=(peer,),
                                        device_id_type=pltpu.DeviceIdType.MESH)
                pl.semaphore_wait(sem, world_size)

                # Phase 1 ID DMA: ring-shift to my_dst = (my_rank + 1) % W.
                # Fixed offset 0, fixed count 128 (TILE-aligned, no @pl.when guard).
                my_dst = (my_rank + 1) % world_size
                d = pltpu.make_async_remote_copy(
                    src_ref=sorted_row_ref.at[pl.ds(0, FIXED_COUNT)],
                    dst_ref=recv_row_buf.at[pl.ds(0, FIXED_COUNT)],
                    send_sem=send_sem,
                    recv_sem=recv_sem,
                    device_id=(my_dst,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
                d.start()
                d.wait()

                # Write rank as sentinel (via VMEM + sync_copy as in v70i).
                scratch_vmem[...] = jnp.full(scratch_vmem.shape, my_rank,
                                             dtype=jnp.int32)
                pltpu.sync_copy(scratch_vmem, out_ref)
            return kernel

        HBM = pltpu.MemorySpace.HBM

        def per_device_run(local_sorted, local_dummy):
            # local_sorted shape: (FIXED_COUNT,) int32 (per-rank shard).
            # local_dummy shape: (1,) int32 (dummy carry).
            out_acc = jnp.zeros((OUT_LEN,), dtype=jnp.int32)
            sync_dep = jnp.float32(0.0)
            for w_py in range(max_waves):
                out_acc_dep, _ = jax.lax.optimization_barrier(
                    (out_acc, sync_dep)
                )
                kernel = make_barrier_dma_kernel(world_size)
                # recv_row_buf is exposed as an HBM "output" (discarded), matching
                # the production fused kernel's pattern of HBM scratch via outputs.
                (out_acc, _recv_dummy) = pl.pallas_call(
                    kernel,
                    out_shape=[
                        jax.ShapeDtypeStruct((OUT_LEN,), jnp.int32),
                        jax.ShapeDtypeStruct((FIXED_COUNT,), jnp.int32),
                    ],
                    grid_spec=pltpu.PrefetchScalarGridSpec(
                        num_scalar_prefetch=0,
                        in_specs=[
                            pl.BlockSpec(memory_space=HBM),  # sorted_row_ref
                            pl.BlockSpec(memory_space=HBM),  # dummy_in_ref
                            pl.BlockSpec(memory_space=HBM),  # out_init_ref
                        ],
                        out_specs=[
                            pl.BlockSpec(memory_space=HBM),  # out_ref
                            pl.BlockSpec(memory_space=HBM),  # recv_row_buf
                        ],
                        scratch_shapes=[
                            pltpu.VMEM((OUT_LEN,), jnp.int32),
                            pltpu.SemaphoreType.DMA,
                            pltpu.SemaphoreType.DMA,
                        ],
                    ),
                    input_output_aliases={2: 0},
                    compiler_params=pltpu.CompilerParams(collective_id=w_py),
                )(local_sorted, local_dummy, out_acc_dep)
                del _recv_dummy
                sync_dep = jax.lax.psum(out_acc[0].astype(jnp.float32), "x")
            return out_acc

        sharded = jax.shard_map(
            per_device_run, mesh=mesh,
            in_specs=(P("x"), P("x")),
            out_specs=P("x"),
            check_vma=False,
        )

        # Per-rank sorted_row input: shape (world_size * FIXED_COUNT,) 1D
        # sharded along axis 0 → each rank gets (FIXED_COUNT,) 1D, matching
        # the 1D recv_row_buf so the DMA src/dst shapes agree. Fill each
        # rank's chunk with that rank's id as a sentinel.
        sorted_per_rank = jnp.repeat(
            jnp.arange(world_size, dtype=jnp.int32), FIXED_COUNT
        )
        assert sorted_per_rank.shape == (world_size * FIXED_COUNT,)
        sorted_per_rank = jax.device_put(
            sorted_per_rank,
            jax.sharding.NamedSharding(mesh, P("x")),
        )
        dummy_per_rank = jnp.zeros((world_size, 1), dtype=jnp.int32)
        dummy_per_rank = jax.device_put(
            dummy_per_rank,
            jax.sharding.NamedSharding(mesh, P("x")),
        )
        jax.block_until_ready(sorted_per_rank)
        jax.block_until_ready(dummy_per_rank)
        log("inputs ready")

        bjit = jax.jit(sharded)
        log("calling kernel...")
        sync_global_devices("litmus_dma1_start")
        t0 = time.perf_counter()
        out = bjit(sorted_per_rank, dummy_per_rank)
        dispatch_ms = (time.perf_counter() - t0) * 1e3
        log(f"dispatched (after {dispatch_ms:.1f}ms), blocking...")
        jax.block_until_ready(out)
        elapsed_ms = (time.perf_counter() - t0) * 1e3
        log(f"ran in {elapsed_ms:.1f} ms total (dispatch={dispatch_ms:.1f}, block={elapsed_ms-dispatch_ms:.1f})")
        log("RESULT: WORKS")

    def test_w32_barrier_plus_phase1_plus_gather_litmus(self) -> None:
        """W=32 barrier + Phase-1 ID DMA + Phase-2 gather litmus.

        Builds on test_w32_barrier_plus_phase1_dma_litmus (which WORKS at W=32).
        Adds the next component: a Phase-2 gather sequence (table HBM->VMEM via
        sync_copy, indices HBM->SMEM via sync_copy, fori_loop gather row-by-row,
        gathered VMEM->HBM via sync_copy).

        If THIS works at W=32, gather is fine; bug is in Phase 3.
        If THIS hangs, the bug is in the Phase-2 gather path at 32-host scale.
        """
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        from jax import lax
        from jax.experimental.multihost_utils import sync_global_devices
        import sys

        world_size = 32
        max_waves = world_size
        P = jax.sharding.PartitionSpec

        all_devices = jax.devices()
        if len(all_devices) < 256:
            self.skipTest(f"Need 256 logical devices, have {len(all_devices)}")
        devices = all_devices[::8]  # symmetric: 1 per host
        assert len(devices) == world_size
        mesh = jax.sharding.Mesh(np.array(devices), ("x",))

        def log(msg):
            if jax.process_index() == 0:
                print(f"[LITMUS_W32_GATHER] {msg}", file=sys.stderr, flush=True)

        OUT_LEN = 128
        FIXED_COUNT = 128
        # Per-rank gather/table sizing.
        v_local = 128
        recv_capacity = 128
        D = 64

        def make_barrier_dma_gather_kernel(world_size: int):
            def kernel(sorted_row_ref, table_ref, dummy_in_ref, out_init_ref,
                       out_ref, recv_row_buf, looked_up_buf,
                       scratch_vmem, send_sem, recv_sem,
                       table_vmem, zero_emb_vmem, idx_smem):
                del dummy_in_ref, out_init_ref
                my_rank = lax.axis_index("x")

                # Pre-barrier (proven to work in v70i barrier-only litmus).
                sem = pltpu.get_barrier_semaphore()
                for peer in range(world_size):
                    pl.semaphore_signal(sem, 1, device_id=(peer,),
                                        device_id_type=pltpu.DeviceIdType.MESH)
                pl.semaphore_wait(sem, world_size)

                # Phase 1 ID DMA: ring-shift to my_dst = (my_rank + 1) % W.
                my_dst = (my_rank + 1) % world_size
                d = pltpu.make_async_remote_copy(
                    src_ref=sorted_row_ref.at[pl.ds(0, FIXED_COUNT)],
                    dst_ref=recv_row_buf.at[pl.ds(0, FIXED_COUNT)],
                    send_sem=send_sem,
                    recv_sem=recv_sem,
                    device_id=(my_dst,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
                d.start()
                d.wait()

                # Phase 2 gather: stage table HBM -> VMEM (matches full kernel).
                pltpu.sync_copy(table_ref, table_vmem)

                # Initialize gather output buffer to zeros.
                zero_emb_vmem[...] = jnp.zeros_like(zero_emb_vmem)

                # Stage recv_row_buf HBM -> SMEM (the indices).
                pltpu.sync_copy(recv_row_buf, idx_smem)

                # Gather loop.
                def gather_body(i, __):
                    row_idx = idx_smem[i]
                    zero_emb_vmem[pl.ds(i, 1), :] = table_vmem[pl.ds(row_idx, 1), :]
                    return None
                lax.fori_loop(0, recv_capacity, gather_body, None)

                # Copy gathered output to HBM (matches full kernel).
                pltpu.sync_copy(zero_emb_vmem, looked_up_buf)

                # Write rank as sentinel (via VMEM + sync_copy as in v70i).
                scratch_vmem[...] = jnp.full(scratch_vmem.shape, my_rank,
                                             dtype=jnp.int32)
                pltpu.sync_copy(scratch_vmem, out_ref)
            return kernel

        HBM = pltpu.MemorySpace.HBM

        def per_device_run(local_sorted, local_table, local_dummy):
            # local_sorted shape: (FIXED_COUNT,) int32 (per-rank shard).
            # local_table shape: (v_local, D) f32 (per-rank shard).
            # local_dummy shape: (1,) int32 (dummy carry).
            out_acc = jnp.zeros((OUT_LEN,), dtype=jnp.int32)
            sync_dep = jnp.float32(0.0)
            for w_py in range(max_waves):
                out_acc_dep, _ = jax.lax.optimization_barrier(
                    (out_acc, sync_dep)
                )
                kernel = make_barrier_dma_gather_kernel(world_size)
                (out_acc, _recv_dummy, _looked_up_dummy) = pl.pallas_call(
                    kernel,
                    out_shape=[
                        jax.ShapeDtypeStruct((OUT_LEN,), jnp.int32),
                        jax.ShapeDtypeStruct((FIXED_COUNT,), jnp.int32),
                        jax.ShapeDtypeStruct((recv_capacity, D), jnp.float32),
                    ],
                    grid_spec=pltpu.PrefetchScalarGridSpec(
                        num_scalar_prefetch=0,
                        in_specs=[
                            pl.BlockSpec(memory_space=HBM),  # sorted_row_ref
                            pl.BlockSpec(memory_space=HBM),  # table_ref
                            pl.BlockSpec(memory_space=HBM),  # dummy_in_ref
                            pl.BlockSpec(memory_space=HBM),  # out_init_ref
                        ],
                        out_specs=[
                            pl.BlockSpec(memory_space=HBM),  # out_ref
                            pl.BlockSpec(memory_space=HBM),  # recv_row_buf
                            pl.BlockSpec(memory_space=HBM),  # looked_up_buf
                        ],
                        scratch_shapes=[
                            pltpu.VMEM((OUT_LEN,), jnp.int32),
                            pltpu.SemaphoreType.DMA,
                            pltpu.SemaphoreType.DMA,
                            pltpu.VMEM((v_local, D), jnp.float32),
                            pltpu.VMEM((recv_capacity, D), jnp.float32),
                            pltpu.SMEM((recv_capacity,), jnp.int32),
                        ],
                    ),
                    input_output_aliases={3: 0},
                    compiler_params=pltpu.CompilerParams(collective_id=w_py),
                )(local_sorted, local_table, local_dummy, out_acc_dep)
                del _recv_dummy, _looked_up_dummy
                sync_dep = jax.lax.psum(out_acc[0].astype(jnp.float32), "x")
            return out_acc

        sharded = jax.shard_map(
            per_device_run, mesh=mesh,
            in_specs=(P("x"), P("x"), P("x")),
            out_specs=P("x"),
            check_vma=False,
        )

        # Per-rank sorted_row input clamped to [0, v_local-1] for gather safety.
        sorted_per_rank = (
            jnp.arange(world_size * FIXED_COUNT, dtype=jnp.int32) % v_local
        )
        assert sorted_per_rank.shape == (world_size * FIXED_COUNT,)
        sorted_per_rank = jax.device_put(
            sorted_per_rank,
            jax.sharding.NamedSharding(mesh, P("x")),
        )
        # Per-rank table input: build flat 2D (W*v_local, D) f32 so that
        # sharding along axis 0 gives each rank a clean 2D (v_local, D)
        # ref (no degenerate leading axis as happens with 3D + shard_map).
        table_per_rank = jnp.zeros((world_size * v_local, D), dtype=jnp.float32)
        table_per_rank = jax.device_put(
            table_per_rank,
            jax.sharding.NamedSharding(mesh, P("x")),
        )
        dummy_per_rank = jnp.zeros((world_size, 1), dtype=jnp.int32)
        dummy_per_rank = jax.device_put(
            dummy_per_rank,
            jax.sharding.NamedSharding(mesh, P("x")),
        )
        jax.block_until_ready(sorted_per_rank)
        jax.block_until_ready(table_per_rank)
        jax.block_until_ready(dummy_per_rank)
        log("inputs ready")

        bjit = jax.jit(sharded)
        log("calling kernel...")
        sync_global_devices("litmus_gather_start")
        t0 = time.perf_counter()
        out = bjit(sorted_per_rank, table_per_rank, dummy_per_rank)
        dispatch_ms = (time.perf_counter() - t0) * 1e3
        log(f"dispatched (after {dispatch_ms:.1f}ms), blocking...")
        jax.block_until_ready(out)
        elapsed_ms = (time.perf_counter() - t0) * 1e3
        log(f"ran in {elapsed_ms:.1f} ms total (dispatch={dispatch_ms:.1f}, block={elapsed_ms-dispatch_ms:.1f})")
        log("RESULT: WORKS")

    def test_w32_full_chain_litmus(self) -> None:
        """W=32 barrier + Phase-1 ID DMA + Phase-2 gather + Phase-3 DMA litmus.

        Builds on test_w32_barrier_plus_phase1_plus_gather_litmus (which WORKS
        at W=32). Adds Phase 3: a final ring-shift DMA that sends the gathered
        embeddings (looked_up_buf, shape (recv_capacity, D)) from each rank
        back to a DIFFERENT peer (my_src = (my_rank + 1) % W) into a 2D
        accumulator HBM out_ref (recv_capacity, D) f32. The accumulator uses
        input_output_aliases like the production wrapper.

        Wrapper structure mirrors v70l: Python `for w_py in range(W)` outer
        loop, per-call collective_id=w_py, per-wave psum + optimization_barrier
        carry on (out_acc, sync_dep).

        If THIS works at W=32, Phase 3 alone is fine; bug is in some
        interaction at production scale.
        If THIS hangs, Phase 3 is the smoking gun (consistent with prior
        litmus chain: barrier WORKS, Phase 1 DMA WORKS, gather WORKS).
        """
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        from jax import lax
        from jax.experimental.multihost_utils import sync_global_devices
        import sys

        world_size = 32
        max_waves = world_size
        P = jax.sharding.PartitionSpec

        all_devices = jax.devices()
        if len(all_devices) < 256:
            self.skipTest(f"Need 256 logical devices, have {len(all_devices)}")
        devices = all_devices[::8]  # symmetric: 1 per host
        assert len(devices) == world_size
        mesh = jax.sharding.Mesh(np.array(devices), ("x",))

        def log(msg):
            if jax.process_index() == 0:
                print(f"[LITMUS_W32_PHASE3] {msg}", file=sys.stderr, flush=True)

        FIXED_COUNT = 128
        # Per-rank gather/table sizing.
        v_local = 128
        recv_capacity = 128
        D = 64

        def make_full_chain_kernel(world_size: int):
            def kernel(sorted_row_ref, table_ref, dummy_in_ref, out_init_ref,
                       out_ref, recv_row_buf, looked_up_buf,
                       send_sem, recv_sem, send_sem_emb, recv_sem_emb,
                       table_vmem, zero_emb_vmem, idx_smem):
                del dummy_in_ref, out_init_ref
                my_rank = lax.axis_index("x")

                # Pre-barrier (proven to work in v70i barrier-only litmus).
                sem = pltpu.get_barrier_semaphore()
                for peer in range(world_size):
                    pl.semaphore_signal(sem, 1, device_id=(peer,),
                                        device_id_type=pltpu.DeviceIdType.MESH)
                pl.semaphore_wait(sem, world_size)

                # Phase 1 ID DMA: ring-shift to my_dst = (my_rank + 1) % W.
                my_dst = (my_rank + 1) % world_size
                d = pltpu.make_async_remote_copy(
                    src_ref=sorted_row_ref.at[pl.ds(0, FIXED_COUNT)],
                    dst_ref=recv_row_buf.at[pl.ds(0, FIXED_COUNT)],
                    send_sem=send_sem,
                    recv_sem=recv_sem,
                    device_id=(my_dst,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
                d.start()
                d.wait()

                # Phase 2 gather: stage table HBM -> VMEM (matches full kernel).
                pltpu.sync_copy(table_ref, table_vmem)

                # Initialize gather output buffer to zeros.
                zero_emb_vmem[...] = jnp.zeros_like(zero_emb_vmem)

                # Stage recv_row_buf HBM -> SMEM (the indices).
                pltpu.sync_copy(recv_row_buf, idx_smem)

                # Gather loop.
                def gather_body(i, __):
                    row_idx = idx_smem[i]
                    zero_emb_vmem[pl.ds(i, 1), :] = table_vmem[pl.ds(row_idx, 1), :]
                    return None
                lax.fori_loop(0, recv_capacity, gather_body, None)

                # Copy gathered output to HBM (matches full kernel).
                pltpu.sync_copy(zero_emb_vmem, looked_up_buf)

                # Phase 3 DMA: send gathered embeddings (looked_up_buf, HBM)
                # back to a DIFFERENT peer (my_src = (my_rank + 1) % W) into
                # the accumulator out_ref on that peer. This is the unknown
                # phase in the litmus chain.
                my_src = (my_rank + 1) % world_size
                phase3_count = recv_capacity
                d3 = pltpu.make_async_remote_copy(
                    src_ref=looked_up_buf.at[pl.ds(0, phase3_count), :],
                    dst_ref=out_ref.at[pl.ds(0, phase3_count), :],
                    send_sem=send_sem_emb,
                    recv_sem=recv_sem_emb,
                    device_id=(my_src,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
                d3.start()
                d3.wait()
            return kernel

        HBM = pltpu.MemorySpace.HBM

        def per_device_run(local_sorted, local_table, local_dummy):
            # local_sorted shape: (FIXED_COUNT,) int32 (per-rank shard).
            # local_table shape: (v_local, D) f32 (per-rank shard).
            # local_dummy shape: (1,) int32 (dummy carry).
            # out_acc is now the 2D (recv_capacity, D) f32 accumulator,
            # written by Phase 3 DMA via input_output_aliases.
            out_acc = jnp.zeros((recv_capacity, D), dtype=jnp.float32)
            sync_dep = jnp.float32(0.0)
            for w_py in range(max_waves):
                out_acc_dep, _ = jax.lax.optimization_barrier(
                    (out_acc, sync_dep)
                )
                kernel = make_full_chain_kernel(world_size)
                # out_ref is the 2D HBM accumulator (recv_capacity, D) f32.
                # recv_row_buf and looked_up_buf are HBM "outputs" (discarded).
                (out_acc, _recv_dummy, _looked_up_dummy) = pl.pallas_call(
                    kernel,
                    out_shape=[
                        jax.ShapeDtypeStruct((recv_capacity, D), jnp.float32),
                        jax.ShapeDtypeStruct((FIXED_COUNT,), jnp.int32),
                        jax.ShapeDtypeStruct((recv_capacity, D), jnp.float32),
                    ],
                    grid_spec=pltpu.PrefetchScalarGridSpec(
                        num_scalar_prefetch=0,
                        in_specs=[
                            pl.BlockSpec(memory_space=HBM),  # sorted_row_ref
                            pl.BlockSpec(memory_space=HBM),  # table_ref
                            pl.BlockSpec(memory_space=HBM),  # dummy_in_ref
                            pl.BlockSpec(memory_space=HBM),  # out_init_ref
                        ],
                        out_specs=[
                            pl.BlockSpec(memory_space=HBM),  # out_ref
                            pl.BlockSpec(memory_space=HBM),  # recv_row_buf
                            pl.BlockSpec(memory_space=HBM),  # looked_up_buf
                        ],
                        scratch_shapes=[
                            pltpu.SemaphoreType.DMA,  # send_sem (Phase 1)
                            pltpu.SemaphoreType.DMA,  # recv_sem (Phase 1)
                            pltpu.SemaphoreType.DMA,  # send_sem_emb (Phase 3)
                            pltpu.SemaphoreType.DMA,  # recv_sem_emb (Phase 3)
                            pltpu.VMEM((v_local, D), jnp.float32),
                            pltpu.VMEM((recv_capacity, D), jnp.float32),
                            pltpu.SMEM((recv_capacity,), jnp.int32),
                        ],
                    ),
                    input_output_aliases={3: 0},
                    compiler_params=pltpu.CompilerParams(collective_id=w_py),
                )(local_sorted, local_table, local_dummy, out_acc_dep)
                del _recv_dummy, _looked_up_dummy
                # Sync_dep uses a scalar reduction over out_acc to keep waves ordered.
                sync_dep = jax.lax.psum(out_acc[0, 0].astype(jnp.float32), "x")
            return out_acc

        sharded = jax.shard_map(
            per_device_run, mesh=mesh,
            in_specs=(P("x"), P("x"), P("x")),
            out_specs=P("x"),
            check_vma=False,
        )

        # Per-rank sorted_row input clamped to [0, v_local-1] for gather safety.
        sorted_per_rank = (
            jnp.arange(world_size * FIXED_COUNT, dtype=jnp.int32) % v_local
        )
        assert sorted_per_rank.shape == (world_size * FIXED_COUNT,)
        sorted_per_rank = jax.device_put(
            sorted_per_rank,
            jax.sharding.NamedSharding(mesh, P("x")),
        )
        # Per-rank table input: build flat 2D (W*v_local, D) f32 so that
        # sharding along axis 0 gives each rank a clean 2D (v_local, D)
        # ref (no degenerate leading axis as happens with 3D + shard_map).
        table_per_rank = jnp.zeros((world_size * v_local, D), dtype=jnp.float32)
        table_per_rank = jax.device_put(
            table_per_rank,
            jax.sharding.NamedSharding(mesh, P("x")),
        )
        dummy_per_rank = jnp.zeros((world_size, 1), dtype=jnp.int32)
        dummy_per_rank = jax.device_put(
            dummy_per_rank,
            jax.sharding.NamedSharding(mesh, P("x")),
        )
        jax.block_until_ready(sorted_per_rank)
        jax.block_until_ready(table_per_rank)
        jax.block_until_ready(dummy_per_rank)
        log("inputs ready")

        bjit = jax.jit(sharded)
        log("calling kernel...")
        sync_global_devices("litmus_phase3_start")
        t0 = time.perf_counter()
        out = bjit(sorted_per_rank, table_per_rank, dummy_per_rank)
        dispatch_ms = (time.perf_counter() - t0) * 1e3
        log(f"dispatched (after {dispatch_ms:.1f}ms), blocking...")
        jax.block_until_ready(out)
        elapsed_ms = (time.perf_counter() - t0) * 1e3
        log(f"ran in {elapsed_ms:.1f} ms total (dispatch={dispatch_ms:.1f}, block={elapsed_ms-dispatch_ms:.1f})")
        log("RESULT: WORKS")

    def test_w32_full_chain_litmus_prodsize(self) -> None:
        """W=32 full-chain litmus at PRODUCTION sizes.

        Identical to test_w32_full_chain_litmus (which WORKS at small sizes:
        v_local=128, recv_capacity=128, D=64) except for the size knobs:
          v_local        = 8K
          recv_capacity  = 8K + 128 (8K + TILE)
          embedding_dim  = 128

        Everything else is identical:
        - Wrapper: Python `for w_py in range(W=32)` outer loop, per-call
          collective_id=w_py, per-wave psum + optimization_barrier carry on
          (out_acc, sync_dep).
        - Fixed peers: my_dst = (my_rank+1)%W (Phase 1) and my_src =
          (my_rank+1)%W (Phase 3) — yes same destination/source, simpler.
        - Fixed offsets (0).
        - Fixed counts (recv_capacity for everything).

        If THIS works at W=32 PRODSIZE: bug is in wave-dependent dynamic
        offsets/counts/peers, NOT in raw size.
        If THIS hangs at W=32 PRODSIZE: bug is in production-size interaction
        (likely large gather loop x cross-rank DMA).
        """
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        from jax import lax
        from jax.experimental.multihost_utils import sync_global_devices
        import sys

        world_size = 32
        max_waves = world_size
        P = jax.sharding.PartitionSpec

        all_devices = jax.devices()
        if len(all_devices) < 256:
            self.skipTest(f"Need 256 logical devices, have {len(all_devices)}")
        devices = all_devices[::8]  # symmetric: 1 per host
        assert len(devices) == world_size
        mesh = jax.sharding.Mesh(np.array(devices), ("x",))

        def log(msg):
            if jax.process_index() == 0:
                print(f"[LITMUS_W32_PRODSIZE] {msg}", file=sys.stderr, flush=True)

        FIXED_COUNT = 128
        # PRODUCTION sizing.
        v_local = 8 * 1024            # was 128 — production scale
        recv_capacity = 8 * 1024 + 128  # was 128 — production scale (8K + TILE)
        D = 128                       # was 64 — production scale

        def make_full_chain_kernel(world_size: int):
            def kernel(sorted_row_ref, table_ref, dummy_in_ref, out_init_ref,
                       out_ref, recv_row_buf, looked_up_buf,
                       send_sem, recv_sem, send_sem_emb, recv_sem_emb,
                       table_vmem, zero_emb_vmem, idx_smem):
                del dummy_in_ref, out_init_ref
                my_rank = lax.axis_index("x")

                # Pre-barrier (proven to work in v70i barrier-only litmus).
                sem = pltpu.get_barrier_semaphore()
                for peer in range(world_size):
                    pl.semaphore_signal(sem, 1, device_id=(peer,),
                                        device_id_type=pltpu.DeviceIdType.MESH)
                pl.semaphore_wait(sem, world_size)

                # Phase 1 ID DMA: ring-shift to my_dst = (my_rank + 1) % W.
                my_dst = (my_rank + 1) % world_size
                d = pltpu.make_async_remote_copy(
                    src_ref=sorted_row_ref.at[pl.ds(0, FIXED_COUNT)],
                    dst_ref=recv_row_buf.at[pl.ds(0, FIXED_COUNT)],
                    send_sem=send_sem,
                    recv_sem=recv_sem,
                    device_id=(my_dst,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
                d.start()
                d.wait()

                # Phase 2 gather: stage table HBM -> VMEM (matches full kernel).
                pltpu.sync_copy(table_ref, table_vmem)

                # Initialize gather output buffer to zeros.
                zero_emb_vmem[...] = jnp.zeros_like(zero_emb_vmem)

                # Stage recv_row_buf HBM -> SMEM (the indices).
                # Note: at PRODSIZE recv_capacity (8K+TILE) > FIXED_COUNT (128),
                # so recv_row_buf is shape (FIXED_COUNT,) and idx_smem is sized
                # to match. The gather loop iterates recv_capacity times but
                # wraps the index using i % FIXED_COUNT (dense, in-bounds).
                pltpu.sync_copy(recv_row_buf, idx_smem)

                # Gather loop.
                def gather_body(i, __):
                    row_idx = idx_smem[i % FIXED_COUNT]
                    zero_emb_vmem[pl.ds(i, 1), :] = table_vmem[pl.ds(row_idx, 1), :]
                    return None
                lax.fori_loop(0, recv_capacity, gather_body, None)

                # Copy gathered output to HBM (matches full kernel).
                pltpu.sync_copy(zero_emb_vmem, looked_up_buf)

                # Phase 3 DMA: send gathered embeddings (looked_up_buf, HBM)
                # back to a DIFFERENT peer (my_src = (my_rank + 1) % W) into
                # the accumulator out_ref on that peer.
                my_src = (my_rank + 1) % world_size
                phase3_count = recv_capacity
                d3 = pltpu.make_async_remote_copy(
                    src_ref=looked_up_buf.at[pl.ds(0, phase3_count), :],
                    dst_ref=out_ref.at[pl.ds(0, phase3_count), :],
                    send_sem=send_sem_emb,
                    recv_sem=recv_sem_emb,
                    device_id=(my_src,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
                d3.start()
                d3.wait()
            return kernel

        HBM = pltpu.MemorySpace.HBM

        def per_device_run(local_sorted, local_table, local_dummy):
            # local_sorted shape: (FIXED_COUNT,) int32 (per-rank shard).
            # local_table shape: (v_local, D) f32 (per-rank shard).
            # local_dummy shape: (1,) int32 (dummy carry).
            # out_acc is now the 2D (recv_capacity, D) f32 accumulator,
            # written by Phase 3 DMA via input_output_aliases.
            out_acc = jnp.zeros((recv_capacity, D), dtype=jnp.float32)
            sync_dep = jnp.float32(0.0)
            for w_py in range(max_waves):
                out_acc_dep, _ = jax.lax.optimization_barrier(
                    (out_acc, sync_dep)
                )
                kernel = make_full_chain_kernel(world_size)
                # out_ref is the 2D HBM accumulator (recv_capacity, D) f32.
                # recv_row_buf and looked_up_buf are HBM "outputs" (discarded).
                (out_acc, _recv_dummy, _looked_up_dummy) = pl.pallas_call(
                    kernel,
                    out_shape=[
                        jax.ShapeDtypeStruct((recv_capacity, D), jnp.float32),
                        jax.ShapeDtypeStruct((FIXED_COUNT,), jnp.int32),
                        jax.ShapeDtypeStruct((recv_capacity, D), jnp.float32),
                    ],
                    grid_spec=pltpu.PrefetchScalarGridSpec(
                        num_scalar_prefetch=0,
                        in_specs=[
                            pl.BlockSpec(memory_space=HBM),  # sorted_row_ref
                            pl.BlockSpec(memory_space=HBM),  # table_ref
                            pl.BlockSpec(memory_space=HBM),  # dummy_in_ref
                            pl.BlockSpec(memory_space=HBM),  # out_init_ref
                        ],
                        out_specs=[
                            pl.BlockSpec(memory_space=HBM),  # out_ref
                            pl.BlockSpec(memory_space=HBM),  # recv_row_buf
                            pl.BlockSpec(memory_space=HBM),  # looked_up_buf
                        ],
                        scratch_shapes=[
                            pltpu.SemaphoreType.DMA,  # send_sem (Phase 1)
                            pltpu.SemaphoreType.DMA,  # recv_sem (Phase 1)
                            pltpu.SemaphoreType.DMA,  # send_sem_emb (Phase 3)
                            pltpu.SemaphoreType.DMA,  # recv_sem_emb (Phase 3)
                            pltpu.VMEM((v_local, D), jnp.float32),
                            pltpu.VMEM((recv_capacity, D), jnp.float32),
                            pltpu.SMEM((FIXED_COUNT,), jnp.int32),
                        ],
                    ),
                    input_output_aliases={3: 0},
                    compiler_params=pltpu.CompilerParams(collective_id=w_py),
                )(local_sorted, local_table, local_dummy, out_acc_dep)
                del _recv_dummy, _looked_up_dummy
                # Sync_dep uses a scalar reduction over out_acc to keep waves ordered.
                sync_dep = jax.lax.psum(out_acc[0, 0].astype(jnp.float32), "x")
            return out_acc

        sharded = jax.shard_map(
            per_device_run, mesh=mesh,
            in_specs=(P("x"), P("x"), P("x")),
            out_specs=P("x"),
            check_vma=False,
        )

        # Per-rank sorted_row input clamped to [0, v_local-1] for gather safety.
        sorted_per_rank = (
            jnp.arange(world_size * FIXED_COUNT, dtype=jnp.int32) % v_local
        )
        assert sorted_per_rank.shape == (world_size * FIXED_COUNT,)
        sorted_per_rank = jax.device_put(
            sorted_per_rank,
            jax.sharding.NamedSharding(mesh, P("x")),
        )
        # Per-rank table input: build flat 2D (W*v_local, D) f32 so that
        # sharding along axis 0 gives each rank a clean 2D (v_local, D)
        # ref (no degenerate leading axis as happens with 3D + shard_map).
        table_per_rank = jnp.zeros((world_size * v_local, D), dtype=jnp.float32)
        table_per_rank = jax.device_put(
            table_per_rank,
            jax.sharding.NamedSharding(mesh, P("x")),
        )
        dummy_per_rank = jnp.zeros((world_size, 1), dtype=jnp.int32)
        dummy_per_rank = jax.device_put(
            dummy_per_rank,
            jax.sharding.NamedSharding(mesh, P("x")),
        )
        jax.block_until_ready(sorted_per_rank)
        jax.block_until_ready(table_per_rank)
        jax.block_until_ready(dummy_per_rank)
        log("inputs ready")

        bjit = jax.jit(sharded)
        log("calling kernel...")
        sync_global_devices("litmus_prodsize_start")
        t0 = time.perf_counter()
        out = bjit(sorted_per_rank, table_per_rank, dummy_per_rank)
        dispatch_ms = (time.perf_counter() - t0) * 1e3
        log(f"dispatched (after {dispatch_ms:.1f}ms), blocking...")
        jax.block_until_ready(out)
        elapsed_ms = (time.perf_counter() - t0) * 1e3
        log(f"ran in {elapsed_ms:.1f} ms total (dispatch={dispatch_ms:.1f}, block={elapsed_ms-dispatch_ms:.1f})")
        log("RESULT: WORKS")

    def test_w32_wave_dep_peer_litmus(self) -> None:
        """W=32 full-chain litmus at PROD sizes with WAVE-DEPENDENT peers.

        Identical to test_w32_full_chain_litmus_prodsize (which WORKS) except
        that Phase 1 sends to my_dst = (my_rank + wave_idx) % W and Phase 3
        sends to my_src = (my_rank - wave_idx + W) % W — matching production's
        ring-shift pattern. ``wave_idx`` is passed as a SMEM (1,) int32 input
        scalar (matches production layout); the kernel reads ``wave_idx_ref[0]``.

        If THIS works at W=32 PRODSIZE: the bug is in dynamic per-wave SMEM
        scalar offsets/counts (next litmus). If THIS hangs: production's
        wave-dependent peer pattern at multi-host scale is the smoking gun.
        """
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        from jax import lax
        from jax.experimental.multihost_utils import sync_global_devices
        import sys

        world_size = 32
        max_waves = world_size
        P = jax.sharding.PartitionSpec

        all_devices = jax.devices()
        if len(all_devices) < 256:
            self.skipTest(f"Need 256 logical devices, have {len(all_devices)}")
        devices = all_devices[::8]  # symmetric: 1 per host
        assert len(devices) == world_size
        mesh = jax.sharding.Mesh(np.array(devices), ("x",))

        def log(msg):
            if jax.process_index() == 0:
                print(f"[LITMUS_W32_WAVEDEP] {msg}", file=sys.stderr, flush=True)

        FIXED_COUNT = 128
        # PRODUCTION sizing.
        v_local = 8 * 1024            # production scale
        recv_capacity = 8 * 1024 + 128  # production scale (8K + TILE)
        D = 128                       # production scale

        def make_full_chain_kernel(world_size: int):
            def kernel(sorted_row_ref, table_ref, dummy_in_ref, wave_idx_ref,
                       out_init_ref,
                       out_ref, recv_row_buf, looked_up_buf,
                       send_sem, recv_sem, send_sem_emb, recv_sem_emb,
                       table_vmem, zero_emb_vmem, idx_smem):
                del dummy_in_ref, out_init_ref
                my_rank = lax.axis_index("x")
                wave_idx = wave_idx_ref[0]  # traced scalar int32

                # Pre-barrier (proven to work in v70i barrier-only litmus).
                sem = pltpu.get_barrier_semaphore()
                for peer in range(world_size):
                    pl.semaphore_signal(sem, 1, device_id=(peer,),
                                        device_id_type=pltpu.DeviceIdType.MESH)
                pl.semaphore_wait(sem, world_size)

                # Phase 1 ID DMA: ring-shift to wave-dependent peer
                # my_dst = (my_rank + wave_idx) % W (matches production).
                my_dst = (my_rank + wave_idx) % world_size
                d = pltpu.make_async_remote_copy(
                    src_ref=sorted_row_ref.at[pl.ds(0, FIXED_COUNT)],
                    dst_ref=recv_row_buf.at[pl.ds(0, FIXED_COUNT)],
                    send_sem=send_sem,
                    recv_sem=recv_sem,
                    device_id=(my_dst,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
                d.start()
                d.wait()

                # Phase 2 gather: stage table HBM -> VMEM (matches full kernel).
                pltpu.sync_copy(table_ref, table_vmem)

                # Initialize gather output buffer to zeros.
                zero_emb_vmem[...] = jnp.zeros_like(zero_emb_vmem)

                # Stage recv_row_buf HBM -> SMEM (the indices).
                pltpu.sync_copy(recv_row_buf, idx_smem)

                # Gather loop.
                def gather_body(i, __):
                    row_idx = idx_smem[i % FIXED_COUNT]
                    zero_emb_vmem[pl.ds(i, 1), :] = table_vmem[pl.ds(row_idx, 1), :]
                    return None
                lax.fori_loop(0, recv_capacity, gather_body, None)

                # Copy gathered output to HBM.
                pltpu.sync_copy(zero_emb_vmem, looked_up_buf)

                # Phase 3 DMA: send gathered embeddings back to wave-dependent
                # source peer my_src = (my_rank - wave_idx + W) % W (matches
                # production).
                my_src = (my_rank - wave_idx + world_size) % world_size
                phase3_count = recv_capacity
                d3 = pltpu.make_async_remote_copy(
                    src_ref=looked_up_buf.at[pl.ds(0, phase3_count), :],
                    dst_ref=out_ref.at[pl.ds(0, phase3_count), :],
                    send_sem=send_sem_emb,
                    recv_sem=recv_sem_emb,
                    device_id=(my_src,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
                d3.start()
                d3.wait()
            return kernel

        HBM = pltpu.MemorySpace.HBM
        SMEM = pltpu.MemorySpace.SMEM

        def per_device_run(local_sorted, local_table, local_dummy):
            out_acc = jnp.zeros((recv_capacity, D), dtype=jnp.float32)
            sync_dep = jnp.float32(0.0)
            for w_py in range(max_waves):
                out_acc_dep, _ = jax.lax.optimization_barrier(
                    (out_acc, sync_dep)
                )
                # SMEM scalar carrying this wave's index w (matches production).
                wave_idx_w = jnp.array([w_py], dtype=jnp.int32)
                kernel = make_full_chain_kernel(world_size)
                (out_acc, _recv_dummy, _looked_up_dummy) = pl.pallas_call(
                    kernel,
                    out_shape=[
                        jax.ShapeDtypeStruct((recv_capacity, D), jnp.float32),
                        jax.ShapeDtypeStruct((FIXED_COUNT,), jnp.int32),
                        jax.ShapeDtypeStruct((recv_capacity, D), jnp.float32),
                    ],
                    grid_spec=pltpu.PrefetchScalarGridSpec(
                        num_scalar_prefetch=0,
                        in_specs=[
                            pl.BlockSpec(memory_space=HBM),   # sorted_row_ref
                            pl.BlockSpec(memory_space=HBM),   # table_ref
                            pl.BlockSpec(memory_space=HBM),   # dummy_in_ref
                            pl.BlockSpec(memory_space=SMEM),  # wave_idx_ref
                            pl.BlockSpec(memory_space=HBM),   # out_init_ref
                        ],
                        out_specs=[
                            pl.BlockSpec(memory_space=HBM),   # out_ref
                            pl.BlockSpec(memory_space=HBM),   # recv_row_buf
                            pl.BlockSpec(memory_space=HBM),   # looked_up_buf
                        ],
                        scratch_shapes=[
                            pltpu.SemaphoreType.DMA,  # send_sem (Phase 1)
                            pltpu.SemaphoreType.DMA,  # recv_sem (Phase 1)
                            pltpu.SemaphoreType.DMA,  # send_sem_emb (Phase 3)
                            pltpu.SemaphoreType.DMA,  # recv_sem_emb (Phase 3)
                            pltpu.VMEM((v_local, D), jnp.float32),
                            pltpu.VMEM((recv_capacity, D), jnp.float32),
                            pltpu.SMEM((FIXED_COUNT,), jnp.int32),
                        ],
                    ),
                    input_output_aliases={4: 0},
                    compiler_params=pltpu.CompilerParams(collective_id=w_py),
                )(local_sorted, local_table, local_dummy, wave_idx_w, out_acc_dep)
                del _recv_dummy, _looked_up_dummy
                # Sync_dep keeps waves ordered cross-rank.
                sync_dep = jax.lax.psum(out_acc[0, 0].astype(jnp.float32), "x")
            return out_acc

        sharded = jax.shard_map(
            per_device_run, mesh=mesh,
            in_specs=(P("x"), P("x"), P("x")),
            out_specs=P("x"),
            check_vma=False,
        )

        sorted_per_rank = (
            jnp.arange(world_size * FIXED_COUNT, dtype=jnp.int32) % v_local
        )
        assert sorted_per_rank.shape == (world_size * FIXED_COUNT,)
        sorted_per_rank = jax.device_put(
            sorted_per_rank,
            jax.sharding.NamedSharding(mesh, P("x")),
        )
        table_per_rank = jnp.zeros((world_size * v_local, D), dtype=jnp.float32)
        table_per_rank = jax.device_put(
            table_per_rank,
            jax.sharding.NamedSharding(mesh, P("x")),
        )
        dummy_per_rank = jnp.zeros((world_size, 1), dtype=jnp.int32)
        dummy_per_rank = jax.device_put(
            dummy_per_rank,
            jax.sharding.NamedSharding(mesh, P("x")),
        )
        jax.block_until_ready(sorted_per_rank)
        jax.block_until_ready(table_per_rank)
        jax.block_until_ready(dummy_per_rank)
        log("inputs ready")

        bjit = jax.jit(sharded)
        log("calling kernel...")
        sync_global_devices("litmus_wavedep_start")
        t0 = time.perf_counter()
        out = bjit(sorted_per_rank, table_per_rank, dummy_per_rank)
        dispatch_ms = (time.perf_counter() - t0) * 1e3
        log(f"dispatched (after {dispatch_ms:.1f}ms), blocking...")
        jax.block_until_ready(out)
        elapsed_ms = (time.perf_counter() - t0) * 1e3
        log(f"ran in {elapsed_ms:.1f} ms total (dispatch={dispatch_ms:.1f}, block={elapsed_ms-dispatch_ms:.1f})")
        log("RESULT: WORKS")

    def test_w32_smem_scalars_litmus(self) -> None:
        """W=32 wave-dep litmus + per-wave SMEM scalar inputs (constant values).

        Identical to test_w32_wave_dep_peer_litmus (which WORKS at PRODSIZE)
        except that the kernel takes five additional per-wave SMEM scalar
        inputs sliced from ``(W,)``-shaped arrays:

          - p1_cnt              : Phase 1 DMA count (set to FIXED_COUNT)
          - p3_cnt              : Phase 3 DMA count (set to recv_capacity)
          - send_off            : Phase 1 source offset in sorted_row (=0)
          - recv_off            : Phase 1 dest offset in recv_row_buf  (=0)
          - phase3_remote_off   : Phase 3 dest offset in remote out_ref (=0)

        Values are CONSTANT across waves. Only the SMEM-scalar plumbing is new.
        Compile-time constants in v70p continue to be used for slice extents
        (Pallas requires static extents); the SMEM-scalar values are routed
        into ``pl.multiple_of(...)`` calls and asserted-zero (or asserted-equal
        to FIXED_COUNT) checks via psum-style reads but do NOT change extents.
        This isolates the SMEM scalar input plumbing from any value-dependent
        change in DMA size/offset.

        If this works at W=32 PRODSIZE: the bug is in per-wave VARYING SMEM
        scalar values (next/last litmus). If it hangs: per-wave SMEM scalar
        inputs themselves are the smoking gun.
        """
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        from jax import lax
        from jax.experimental.multihost_utils import sync_global_devices
        import sys

        world_size = 32
        max_waves = world_size
        P = jax.sharding.PartitionSpec

        all_devices = jax.devices()
        if len(all_devices) < 256:
            self.skipTest(f"Need 256 logical devices, have {len(all_devices)}")
        devices = all_devices[::8]  # symmetric: 1 per host
        assert len(devices) == world_size
        mesh = jax.sharding.Mesh(np.array(devices), ("x",))

        def log(msg):
            if jax.process_index() == 0:
                print(f"[LITMUS_W32_SMEM] {msg}", file=sys.stderr, flush=True)

        FIXED_COUNT = 128
        # PRODUCTION sizing.
        v_local = 8 * 1024            # production scale
        recv_capacity = 8 * 1024 + 128  # production scale (8K + TILE)
        D = 128                       # production scale
        TILE = 128

        def make_full_chain_kernel(world_size: int):
            def kernel(sorted_row_ref, table_ref, dummy_in_ref,
                       wave_idx_ref, p1_cnt_ref, p3_cnt_ref,
                       send_off_ref, recv_off_ref, phase3_remote_off_ref,
                       out_init_ref,
                       out_ref, recv_row_buf, looked_up_buf,
                       send_sem, recv_sem, send_sem_emb, recv_sem_emb,
                       table_vmem, zero_emb_vmem, idx_smem):
                del dummy_in_ref, out_init_ref
                my_rank = lax.axis_index("x")
                wave_idx = wave_idx_ref[0]  # traced scalar int32

                # Per-wave SMEM scalars (read but unused for slice extents:
                # Pallas needs static extents; we still route values through
                # the kernel to exercise the SMEM scalar input wiring).
                p1_cnt = pl.multiple_of(p1_cnt_ref[0], TILE)
                p3_cnt = pl.multiple_of(p3_cnt_ref[0], TILE)
                send_start = pl.multiple_of(send_off_ref[0], TILE)
                remote_recv_start = pl.multiple_of(recv_off_ref[0], TILE)
                phase3_remote_start = pl.multiple_of(
                    phase3_remote_off_ref[0], TILE
                )
                # Use values to keep them live (XLA/MLIR can't DCE these
                # because they feed into DMA src/dst offsets via lax.add 0s).
                _ = p1_cnt + p3_cnt  # keep counts referenced

                # Pre-barrier (proven to work in v70i barrier-only litmus).
                sem = pltpu.get_barrier_semaphore()
                for peer in range(world_size):
                    pl.semaphore_signal(sem, 1, device_id=(peer,),
                                        device_id_type=pltpu.DeviceIdType.MESH)
                pl.semaphore_wait(sem, world_size)

                # Phase 1 ID DMA: ring-shift to wave-dependent peer
                # my_dst = (my_rank + wave_idx) % W (matches production).
                my_dst = (my_rank + wave_idx) % world_size
                # Use SMEM-scalar offsets in pl.ds starts (extent stays static).
                d = pltpu.make_async_remote_copy(
                    src_ref=sorted_row_ref.at[pl.ds(send_start, FIXED_COUNT)],
                    dst_ref=recv_row_buf.at[pl.ds(remote_recv_start, FIXED_COUNT)],
                    send_sem=send_sem,
                    recv_sem=recv_sem,
                    device_id=(my_dst,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
                d.start()
                d.wait()

                # Phase 2 gather: stage table HBM -> VMEM (matches full kernel).
                pltpu.sync_copy(table_ref, table_vmem)

                # Initialize gather output buffer to zeros.
                zero_emb_vmem[...] = jnp.zeros_like(zero_emb_vmem)

                # Stage recv_row_buf HBM -> SMEM (the indices).
                pltpu.sync_copy(recv_row_buf, idx_smem)

                # Gather loop.
                def gather_body(i, __):
                    row_idx = idx_smem[i % FIXED_COUNT]
                    zero_emb_vmem[pl.ds(i, 1), :] = table_vmem[pl.ds(row_idx, 1), :]
                    return None
                lax.fori_loop(0, recv_capacity, gather_body, None)

                # Copy gathered output to HBM.
                pltpu.sync_copy(zero_emb_vmem, looked_up_buf)

                # Phase 3 DMA: send gathered embeddings back to wave-dependent
                # source peer my_src = (my_rank - wave_idx + W) % W (matches
                # production). Use SMEM-scalar phase3_remote_start as dst start.
                my_src = (my_rank - wave_idx + world_size) % world_size
                phase3_count = recv_capacity
                d3 = pltpu.make_async_remote_copy(
                    src_ref=looked_up_buf.at[pl.ds(0, phase3_count), :],
                    dst_ref=out_ref.at[pl.ds(phase3_remote_start, phase3_count), :],
                    send_sem=send_sem_emb,
                    recv_sem=recv_sem_emb,
                    device_id=(my_src,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
                d3.start()
                d3.wait()
            return kernel

        HBM = pltpu.MemorySpace.HBM
        SMEM = pltpu.MemorySpace.SMEM

        def per_device_run(local_sorted, local_table, local_dummy):
            out_acc = jnp.zeros((recv_capacity, D), dtype=jnp.float32)
            sync_dep = jnp.float32(0.0)
            # Per-wave SMEM scalar arrays of shape (W,). Constant across waves
            # so behavior matches v70p; only the SMEM plumbing is new.
            p1_cnt_arr = jnp.full((max_waves,), FIXED_COUNT, dtype=jnp.int32)
            p3_cnt_arr = jnp.full((max_waves,), recv_capacity, dtype=jnp.int32)
            send_off_arr = jnp.zeros((max_waves,), dtype=jnp.int32)
            recv_off_arr = jnp.zeros((max_waves,), dtype=jnp.int32)
            phase3_remote_off_arr = jnp.zeros((max_waves,), dtype=jnp.int32)
            for w_py in range(max_waves):
                out_acc_dep, _ = jax.lax.optimization_barrier(
                    (out_acc, sync_dep)
                )
                # SMEM scalar carrying this wave's index w (matches production).
                wave_idx_w = jnp.array([w_py], dtype=jnp.int32)
                p1_cnt_w = lax.dynamic_slice(p1_cnt_arr, (w_py,), (1,))
                p3_cnt_w = lax.dynamic_slice(p3_cnt_arr, (w_py,), (1,))
                send_off_w = lax.dynamic_slice(send_off_arr, (w_py,), (1,))
                recv_off_w = lax.dynamic_slice(recv_off_arr, (w_py,), (1,))
                phase3_remote_off_w = lax.dynamic_slice(
                    phase3_remote_off_arr, (w_py,), (1,)
                )
                kernel = make_full_chain_kernel(world_size)
                (out_acc, _recv_dummy, _looked_up_dummy) = pl.pallas_call(
                    kernel,
                    out_shape=[
                        jax.ShapeDtypeStruct((recv_capacity, D), jnp.float32),
                        jax.ShapeDtypeStruct((FIXED_COUNT,), jnp.int32),
                        jax.ShapeDtypeStruct((recv_capacity, D), jnp.float32),
                    ],
                    grid_spec=pltpu.PrefetchScalarGridSpec(
                        num_scalar_prefetch=0,
                        in_specs=[
                            pl.BlockSpec(memory_space=HBM),   # sorted_row_ref
                            pl.BlockSpec(memory_space=HBM),   # table_ref
                            pl.BlockSpec(memory_space=HBM),   # dummy_in_ref
                            pl.BlockSpec(memory_space=SMEM),  # wave_idx_ref
                            pl.BlockSpec(memory_space=SMEM),  # p1_cnt_ref
                            pl.BlockSpec(memory_space=SMEM),  # p3_cnt_ref
                            pl.BlockSpec(memory_space=SMEM),  # send_off_ref
                            pl.BlockSpec(memory_space=SMEM),  # recv_off_ref
                            pl.BlockSpec(memory_space=SMEM),  # phase3_remote_off_ref
                            pl.BlockSpec(memory_space=HBM),   # out_init_ref
                        ],
                        out_specs=[
                            pl.BlockSpec(memory_space=HBM),   # out_ref
                            pl.BlockSpec(memory_space=HBM),   # recv_row_buf
                            pl.BlockSpec(memory_space=HBM),   # looked_up_buf
                        ],
                        scratch_shapes=[
                            pltpu.SemaphoreType.DMA,  # send_sem (Phase 1)
                            pltpu.SemaphoreType.DMA,  # recv_sem (Phase 1)
                            pltpu.SemaphoreType.DMA,  # send_sem_emb (Phase 3)
                            pltpu.SemaphoreType.DMA,  # recv_sem_emb (Phase 3)
                            pltpu.VMEM((v_local, D), jnp.float32),
                            pltpu.VMEM((recv_capacity, D), jnp.float32),
                            pltpu.SMEM((FIXED_COUNT,), jnp.int32),
                        ],
                    ),
                    input_output_aliases={9: 0},
                    compiler_params=pltpu.CompilerParams(collective_id=w_py),
                )(
                    local_sorted, local_table, local_dummy,
                    wave_idx_w, p1_cnt_w, p3_cnt_w,
                    send_off_w, recv_off_w, phase3_remote_off_w,
                    out_acc_dep,
                )
                del _recv_dummy, _looked_up_dummy
                # Sync_dep keeps waves ordered cross-rank.
                sync_dep = jax.lax.psum(out_acc[0, 0].astype(jnp.float32), "x")
            return out_acc

        sharded = jax.shard_map(
            per_device_run, mesh=mesh,
            in_specs=(P("x"), P("x"), P("x")),
            out_specs=P("x"),
            check_vma=False,
        )

        sorted_per_rank = (
            jnp.arange(world_size * FIXED_COUNT, dtype=jnp.int32) % v_local
        )
        assert sorted_per_rank.shape == (world_size * FIXED_COUNT,)
        sorted_per_rank = jax.device_put(
            sorted_per_rank,
            jax.sharding.NamedSharding(mesh, P("x")),
        )
        table_per_rank = jnp.zeros((world_size * v_local, D), dtype=jnp.float32)
        table_per_rank = jax.device_put(
            table_per_rank,
            jax.sharding.NamedSharding(mesh, P("x")),
        )
        dummy_per_rank = jnp.zeros((world_size, 1), dtype=jnp.int32)
        dummy_per_rank = jax.device_put(
            dummy_per_rank,
            jax.sharding.NamedSharding(mesh, P("x")),
        )
        jax.block_until_ready(sorted_per_rank)
        jax.block_until_ready(table_per_rank)
        jax.block_until_ready(dummy_per_rank)
        log("inputs ready")

        bjit = jax.jit(sharded)
        log("calling kernel...")
        sync_global_devices("litmus_smem_scalars_start")
        t0 = time.perf_counter()
        out = bjit(sorted_per_rank, table_per_rank, dummy_per_rank)
        dispatch_ms = (time.perf_counter() - t0) * 1e3
        log(f"dispatched (after {dispatch_ms:.1f}ms), blocking...")
        jax.block_until_ready(out)
        elapsed_ms = (time.perf_counter() - t0) * 1e3
        log(f"ran in {elapsed_ms:.1f} ms total (dispatch={dispatch_ms:.1f}, block={elapsed_ms-dispatch_ms:.1f})")
        log("RESULT: WORKS")

    def test_w32_smem_varying_litmus(self) -> None:
        """W=32 wave-dep litmus + per-wave SMEM scalar inputs (VARYING values).

        Identical to test_w32_smem_scalars_litmus (v70q) except the per-wave
        SMEM scalar arrays carry VALUES THAT VARY ACROSS WAVES (instead of
        being constant). All values remain TILE-aligned and within in-bounds
        ranges (input/output buffers enlarged to keep the offsets safe).

        Variation pattern:
          - p1_cnt[w]            : alternates between FIXED_COUNT and half_count
          - p3_cnt[w]            : alternates between half_count and FIXED_COUNT
          - send_off[w]          : cycles 0, TILE, 2*TILE, 3*TILE
          - recv_off[w]          : 0 (matches v68c single-slot)
          - phase3_remote_off[w] : cycles 0, TILE, 2*TILE, 3*TILE

        Slice EXTENTS still use compile-time constants (Pallas requires static
        extents); only the offsets and the multiple_of/markers consume the
        SMEM-scalar values. This narrowly isolates "does varying SMEM-scalar
        contents (vs. constant) trigger the W=32 hang?".

        If this works at W=32 PRODSIZE: the hang requires a more specific
        per-wave SMEM-scalar VALUE pattern (likely the splitter's actual
        output). If it hangs: smoking gun — varying SMEM scalar values cause
        the W=32 hang.
        """
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        from jax import lax
        from jax.experimental.multihost_utils import sync_global_devices
        import sys

        world_size = 32
        max_waves = world_size
        P = jax.sharding.PartitionSpec

        all_devices = jax.devices()
        if len(all_devices) < 256:
            self.skipTest(f"Need 256 logical devices, have {len(all_devices)}")
        devices = all_devices[::8]  # symmetric: 1 per host
        assert len(devices) == world_size
        mesh = jax.sharding.Mesh(np.array(devices), ("x",))

        def log(msg):
            if jax.process_index() == 0:
                print(f"[LITMUS_W32_VARYING] {msg}", file=sys.stderr, flush=True)

        FIXED_COUNT = 128
        # PRODUCTION sizing.
        v_local = 8 * 1024            # production scale
        recv_capacity = 8 * 1024 + 128  # production scale (8K + TILE)
        D = 128                       # production scale
        TILE = 128

        # Generous extra room for varying offsets (cycle 0..3*TILE).
        # send_off in [0, 3*TILE] and Phase 1 slice extent = FIXED_COUNT, so
        # per-rank sorted_row buffer needs >= 3*TILE + FIXED_COUNT == 4*TILE.
        # Keep symmetric, sized 2*FIXED_COUNT*MAX_OFF_FACTOR for safety.
        SORTED_PER_RANK_LEN = 4 * FIXED_COUNT  # 512 (room for 3*TILE offset)
        # Out buffer: phase3_remote_off in [0, 3*TILE], extent recv_capacity.
        # So per-rank out_acc needs >= recv_capacity + 3*TILE.
        OUT_PER_RANK_LEN = recv_capacity + 3 * TILE  # 8K + 4*TILE

        def make_full_chain_kernel(world_size: int):
            def kernel(sorted_row_ref, table_ref, dummy_in_ref,
                       wave_idx_ref, p1_cnt_ref, p3_cnt_ref,
                       send_off_ref, recv_off_ref, phase3_remote_off_ref,
                       out_init_ref,
                       out_ref, recv_row_buf, looked_up_buf,
                       send_sem, recv_sem, send_sem_emb, recv_sem_emb,
                       table_vmem, zero_emb_vmem, idx_smem):
                del dummy_in_ref, out_init_ref
                my_rank = lax.axis_index("x")
                wave_idx = wave_idx_ref[0]  # traced scalar int32

                # Per-wave SMEM scalars (read but unused for slice extents:
                # Pallas needs static extents; we still route values through
                # the kernel to exercise the SMEM scalar input wiring).
                p1_cnt = pl.multiple_of(p1_cnt_ref[0], TILE)
                p3_cnt = pl.multiple_of(p3_cnt_ref[0], TILE)
                send_start = pl.multiple_of(send_off_ref[0], TILE)
                remote_recv_start = pl.multiple_of(recv_off_ref[0], TILE)
                phase3_remote_start = pl.multiple_of(
                    phase3_remote_off_ref[0], TILE
                )
                # Use values to keep them live (XLA/MLIR can't DCE these
                # because they feed into DMA src/dst offsets via lax.add 0s).
                _ = p1_cnt + p3_cnt  # keep counts referenced

                # Pre-barrier (proven to work in v70i barrier-only litmus).
                sem = pltpu.get_barrier_semaphore()
                for peer in range(world_size):
                    pl.semaphore_signal(sem, 1, device_id=(peer,),
                                        device_id_type=pltpu.DeviceIdType.MESH)
                pl.semaphore_wait(sem, world_size)

                # Phase 1 ID DMA: ring-shift to wave-dependent peer
                # my_dst = (my_rank + wave_idx) % W (matches production).
                my_dst = (my_rank + wave_idx) % world_size
                # Use SMEM-scalar offsets in pl.ds starts (extent stays static).
                d = pltpu.make_async_remote_copy(
                    src_ref=sorted_row_ref.at[pl.ds(send_start, FIXED_COUNT)],
                    dst_ref=recv_row_buf.at[pl.ds(remote_recv_start, FIXED_COUNT)],
                    send_sem=send_sem,
                    recv_sem=recv_sem,
                    device_id=(my_dst,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
                d.start()
                d.wait()

                # Phase 2 gather: stage table HBM -> VMEM (matches full kernel).
                pltpu.sync_copy(table_ref, table_vmem)

                # Initialize gather output buffer to zeros.
                zero_emb_vmem[...] = jnp.zeros_like(zero_emb_vmem)

                # Stage recv_row_buf HBM -> SMEM (the indices).
                pltpu.sync_copy(recv_row_buf, idx_smem)

                # Gather loop.
                def gather_body(i, __):
                    row_idx = idx_smem[i % FIXED_COUNT]
                    zero_emb_vmem[pl.ds(i, 1), :] = table_vmem[pl.ds(row_idx, 1), :]
                    return None
                lax.fori_loop(0, recv_capacity, gather_body, None)

                # Copy gathered output to HBM.
                pltpu.sync_copy(zero_emb_vmem, looked_up_buf)

                # Phase 3 DMA: send gathered embeddings back to wave-dependent
                # source peer my_src = (my_rank - wave_idx + W) % W (matches
                # production). Use SMEM-scalar phase3_remote_start as dst start.
                my_src = (my_rank - wave_idx + world_size) % world_size
                phase3_count = recv_capacity
                d3 = pltpu.make_async_remote_copy(
                    src_ref=looked_up_buf.at[pl.ds(0, phase3_count), :],
                    dst_ref=out_ref.at[pl.ds(phase3_remote_start, phase3_count), :],
                    send_sem=send_sem_emb,
                    recv_sem=recv_sem_emb,
                    device_id=(my_src,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
                d3.start()
                d3.wait()
            return kernel

        HBM = pltpu.MemorySpace.HBM
        SMEM = pltpu.MemorySpace.SMEM

        def per_device_run(local_sorted, local_table, local_dummy):
            out_acc = jnp.zeros((OUT_PER_RANK_LEN, D), dtype=jnp.float32)
            sync_dep = jnp.float32(0.0)
            # Per-wave SMEM scalar arrays of shape (W,). VARY across waves
            # (still TILE-aligned, still in-bounds for the enlarged buffers).
            half_count = (FIXED_COUNT // 2 // TILE) * TILE
            half_count = max(half_count, TILE)
            p1_cnt_arr = jnp.array(
                [FIXED_COUNT if w % 2 == 0 else half_count for w in range(max_waves)],
                dtype=jnp.int32,
            )
            p3_cnt_arr = jnp.array(
                [half_count if w % 2 == 0 else FIXED_COUNT for w in range(max_waves)],
                dtype=jnp.int32,
            )
            # send_off cycles 0, TILE, 2*TILE, 3*TILE — TILE-aligned, in-bounds.
            send_off_arr = jnp.array(
                [(w % 4) * TILE for w in range(max_waves)],
                dtype=jnp.int32,
            )
            # recv_off stays 0 (matches v68c single-slot).
            recv_off_arr = jnp.zeros((max_waves,), dtype=jnp.int32)
            # phase3_remote_off cycles 0, TILE, 2*TILE, 3*TILE.
            phase3_remote_off_arr = jnp.array(
                [(w % 4) * TILE for w in range(max_waves)],
                dtype=jnp.int32,
            )
            for w_py in range(max_waves):
                out_acc_dep, _ = jax.lax.optimization_barrier(
                    (out_acc, sync_dep)
                )
                # SMEM scalar carrying this wave's index w (matches production).
                wave_idx_w = jnp.array([w_py], dtype=jnp.int32)
                p1_cnt_w = lax.dynamic_slice(p1_cnt_arr, (w_py,), (1,))
                p3_cnt_w = lax.dynamic_slice(p3_cnt_arr, (w_py,), (1,))
                send_off_w = lax.dynamic_slice(send_off_arr, (w_py,), (1,))
                recv_off_w = lax.dynamic_slice(recv_off_arr, (w_py,), (1,))
                phase3_remote_off_w = lax.dynamic_slice(
                    phase3_remote_off_arr, (w_py,), (1,)
                )
                kernel = make_full_chain_kernel(world_size)
                (out_acc, _recv_dummy, _looked_up_dummy) = pl.pallas_call(
                    kernel,
                    out_shape=[
                        jax.ShapeDtypeStruct((OUT_PER_RANK_LEN, D), jnp.float32),
                        jax.ShapeDtypeStruct((FIXED_COUNT,), jnp.int32),
                        jax.ShapeDtypeStruct((recv_capacity, D), jnp.float32),
                    ],
                    grid_spec=pltpu.PrefetchScalarGridSpec(
                        num_scalar_prefetch=0,
                        in_specs=[
                            pl.BlockSpec(memory_space=HBM),   # sorted_row_ref
                            pl.BlockSpec(memory_space=HBM),   # table_ref
                            pl.BlockSpec(memory_space=HBM),   # dummy_in_ref
                            pl.BlockSpec(memory_space=SMEM),  # wave_idx_ref
                            pl.BlockSpec(memory_space=SMEM),  # p1_cnt_ref
                            pl.BlockSpec(memory_space=SMEM),  # p3_cnt_ref
                            pl.BlockSpec(memory_space=SMEM),  # send_off_ref
                            pl.BlockSpec(memory_space=SMEM),  # recv_off_ref
                            pl.BlockSpec(memory_space=SMEM),  # phase3_remote_off_ref
                            pl.BlockSpec(memory_space=HBM),   # out_init_ref
                        ],
                        out_specs=[
                            pl.BlockSpec(memory_space=HBM),   # out_ref
                            pl.BlockSpec(memory_space=HBM),   # recv_row_buf
                            pl.BlockSpec(memory_space=HBM),   # looked_up_buf
                        ],
                        scratch_shapes=[
                            pltpu.SemaphoreType.DMA,  # send_sem (Phase 1)
                            pltpu.SemaphoreType.DMA,  # recv_sem (Phase 1)
                            pltpu.SemaphoreType.DMA,  # send_sem_emb (Phase 3)
                            pltpu.SemaphoreType.DMA,  # recv_sem_emb (Phase 3)
                            pltpu.VMEM((v_local, D), jnp.float32),
                            pltpu.VMEM((recv_capacity, D), jnp.float32),
                            pltpu.SMEM((FIXED_COUNT,), jnp.int32),
                        ],
                    ),
                    input_output_aliases={9: 0},
                    compiler_params=pltpu.CompilerParams(collective_id=w_py),
                )(
                    local_sorted, local_table, local_dummy,
                    wave_idx_w, p1_cnt_w, p3_cnt_w,
                    send_off_w, recv_off_w, phase3_remote_off_w,
                    out_acc_dep,
                )
                del _recv_dummy, _looked_up_dummy
                # Sync_dep keeps waves ordered cross-rank.
                sync_dep = jax.lax.psum(out_acc[0, 0].astype(jnp.float32), "x")
            return out_acc

        sharded = jax.shard_map(
            per_device_run, mesh=mesh,
            in_specs=(P("x"), P("x"), P("x")),
            out_specs=P("x"),
            check_vma=False,
        )

        # Per-rank sorted_row enlarged to SORTED_PER_RANK_LEN to fit varying
        # send_off in [0, 3*TILE] + extent FIXED_COUNT.
        sorted_per_rank = (
            jnp.arange(world_size * SORTED_PER_RANK_LEN, dtype=jnp.int32) % v_local
        )
        assert sorted_per_rank.shape == (world_size * SORTED_PER_RANK_LEN,)
        sorted_per_rank = jax.device_put(
            sorted_per_rank,
            jax.sharding.NamedSharding(mesh, P("x")),
        )
        table_per_rank = jnp.zeros((world_size * v_local, D), dtype=jnp.float32)
        table_per_rank = jax.device_put(
            table_per_rank,
            jax.sharding.NamedSharding(mesh, P("x")),
        )
        dummy_per_rank = jnp.zeros((world_size, 1), dtype=jnp.int32)
        dummy_per_rank = jax.device_put(
            dummy_per_rank,
            jax.sharding.NamedSharding(mesh, P("x")),
        )
        jax.block_until_ready(sorted_per_rank)
        jax.block_until_ready(table_per_rank)
        jax.block_until_ready(dummy_per_rank)
        log("inputs ready")

        bjit = jax.jit(sharded)
        log("calling kernel...")
        sync_global_devices("litmus_smem_varying_start")
        t0 = time.perf_counter()
        out = bjit(sorted_per_rank, table_per_rank, dummy_per_rank)
        dispatch_ms = (time.perf_counter() - t0) * 1e3
        log(f"dispatched (after {dispatch_ms:.1f}ms), blocking...")
        jax.block_until_ready(out)
        elapsed_ms = (time.perf_counter() - t0) * 1e3
        log(f"ran in {elapsed_ms:.1f} ms total (dispatch={dispatch_ms:.1f}, block={elapsed_ms-dispatch_ms:.1f})")
        log("RESULT: WORKS")

    def test_w32_smem_real_splitter_litmus(self) -> None:
        """W=32 litmus + REAL per-wave SMEM scalars from production splitter.

        Identical structure to test_w32_smem_varying_litmus (v70r) except the
        five per-wave SMEM scalar arrays (p1_cnt, p3_cnt, send_off, recv_off,
        phase3_remote_off) are computed from REAL inputs via
        ``_bucketize_and_sort`` -> ``_round_up_tile`` ->
        ``_split_traffic_ringshift_globalpad`` (the production splitter chain
        from jagged_a2a_lookup.py). Each splitter output array is shape ``(W,)``
        per rank, sharded along axis 0 — values vary BOTH across waves AND
        across ranks (the missing axis vs v70r).

        Per-rank-varying values arise because each rank's traffic distribution
        differs (skewed inputs, hot peer = (rank+1)%W). The per-wave slices
        feed the same kernel structure as v70r — so the only delta is the
        VALUE PATTERN of the SMEM scalars. If this hangs at W=32, the smoking
        gun is per-rank-varying SMEM scalar values from the real splitter.

        Note: real ``send_off_local`` can reach up to ``max_padded =
        max_local_ids + W*TILE`` (12288 at W=32, max_local_ids=8K). Real
        ``phase3_remote_off`` can reach up to ``max_local_ids``. Buffers are
        sized accordingly.

        IMPORTANT: NO ``pl.when(p1_cnt > 0)`` guards. If a rank sends 0 to a
        peer in some wave, the resulting 0-length DMA may crash libtpu firmware
        on v7x — that is part of the smoking-gun signal we want to detect.
        """
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        from jax import lax
        from jax.experimental.multihost_utils import sync_global_devices
        from recir.ops.pallas.embeddings.jagged_a2a_lookup import (
            _bucketize_and_sort,
            _round_up_tile,
            _split_traffic_ringshift_globalpad,
            _TILE_INT32,
        )
        import sys

        world_size = 32
        max_waves = world_size
        P = jax.sharding.PartitionSpec

        all_devices = jax.devices()
        if len(all_devices) < 256:
            self.skipTest(f"Need 256 logical devices, have {len(all_devices)}")
        devices = all_devices[::8]  # symmetric: 1 per host
        assert len(devices) == world_size
        mesh = jax.sharding.Mesh(np.array(devices), ("x",))

        def log(msg):
            if jax.process_index() == 0:
                print(f"[LITMUS_W32_REALSPLIT] {msg}", file=sys.stderr, flush=True)

        # Production sizing (matches v70n2/v70p/v70q/v70r).
        max_local_ids = 8 * 1024
        v_local = 8 * 1024
        recv_capacity = 8 * 1024 + 128  # 8K + TILE
        D = 128
        TILE = 128

        # Real splitter `send_off_local` values can reach max_padded =
        # max_local_ids + W*TILE = 12288 at W=32. The Phase 1 DMA copies
        # FIXED_COUNT rows starting at send_off, but here Phase 1 extent is
        # the static FIXED_COUNT (we keep extents static). The send buffer
        # must hold up to send_off_max + FIXED_COUNT rows.
        FIXED_COUNT = 128
        max_padded = max_local_ids + world_size * TILE  # 12288
        SORTED_PER_RANK_LEN = max_padded + FIXED_COUNT  # 12416

        # Real `phase3_remote_off` values can reach up to max_local_ids
        # (sum_{i<my_rank} traffic_matrix[src, i] is bounded by total fan-in
        # = sum over a row of traffic_matrix <= max_local_ids + W*TILE,
        # but in practice for skewed input it stays under max_padded). The
        # Phase 3 DMA dst extent is recv_capacity, so out buffer per rank
        # must hold phase3_remote_off_max + recv_capacity. Worst case:
        # phase3_remote_off can reach max_padded - 1 (one peer's offset).
        OUT_PER_RANK_LEN = max_padded + recv_capacity  # 12288 + 8320 = 20608

        log(f"max_padded={max_padded} SORTED={SORTED_PER_RANK_LEN} OUT={OUT_PER_RANK_LEN}")

        def make_full_chain_kernel(world_size: int):
            def kernel(sorted_row_ref, table_ref, dummy_in_ref,
                       wave_idx_ref, p1_cnt_ref, p3_cnt_ref,
                       send_off_ref, recv_off_ref, phase3_remote_off_ref,
                       out_init_ref,
                       out_ref, recv_row_buf, looked_up_buf,
                       send_sem, recv_sem, send_sem_emb, recv_sem_emb,
                       table_vmem, zero_emb_vmem, idx_smem):
                del dummy_in_ref, out_init_ref
                my_rank = lax.axis_index("x")
                wave_idx = wave_idx_ref[0]  # traced scalar int32

                # Per-wave SMEM scalars from REAL splitter (this rank's
                # values for this wave). Pallas requires static slice
                # extents, so we still use FIXED_COUNT / recv_capacity for
                # extents — the SMEM-scalar values feed offsets and the
                # multiple_of/markers. This narrowly tests "do REAL
                # splitter values trigger the W=32 hang?".
                p1_cnt = pl.multiple_of(p1_cnt_ref[0], TILE)
                p3_cnt = pl.multiple_of(p3_cnt_ref[0], TILE)
                send_start = pl.multiple_of(send_off_ref[0], TILE)
                remote_recv_start = pl.multiple_of(recv_off_ref[0], TILE)
                phase3_remote_start = pl.multiple_of(
                    phase3_remote_off_ref[0], TILE
                )
                # Use values to keep them live (XLA/MLIR can't DCE these
                # because they feed into DMA src/dst offsets).
                _ = p1_cnt + p3_cnt  # keep counts referenced

                # Pre-barrier (proven to work in v70i barrier-only litmus).
                sem = pltpu.get_barrier_semaphore()
                for peer in range(world_size):
                    pl.semaphore_signal(sem, 1, device_id=(peer,),
                                        device_id_type=pltpu.DeviceIdType.MESH)
                pl.semaphore_wait(sem, world_size)

                # Phase 1 ID DMA: ring-shift to wave-dependent peer
                # my_dst = (my_rank + wave_idx) % W (matches production).
                my_dst = (my_rank + wave_idx) % world_size
                # Use SMEM-scalar offsets in pl.ds starts (extent stays static).
                # NO p1_cnt > 0 guard — let 0-length DMAs crash naturally if
                # that's the issue.
                d = pltpu.make_async_remote_copy(
                    src_ref=sorted_row_ref.at[pl.ds(send_start, FIXED_COUNT)],
                    dst_ref=recv_row_buf.at[pl.ds(remote_recv_start, FIXED_COUNT)],
                    send_sem=send_sem,
                    recv_sem=recv_sem,
                    device_id=(my_dst,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
                d.start()
                d.wait()

                # Phase 2 gather: stage table HBM -> VMEM (matches full kernel).
                pltpu.sync_copy(table_ref, table_vmem)

                # Initialize gather output buffer to zeros.
                zero_emb_vmem[...] = jnp.zeros_like(zero_emb_vmem)

                # Stage recv_row_buf HBM -> SMEM (the indices).
                pltpu.sync_copy(recv_row_buf, idx_smem)

                # Gather loop.
                def gather_body(i, __):
                    row_idx = idx_smem[i % FIXED_COUNT]
                    zero_emb_vmem[pl.ds(i, 1), :] = table_vmem[pl.ds(row_idx, 1), :]
                    return None
                lax.fori_loop(0, recv_capacity, gather_body, None)

                # Copy gathered output to HBM.
                pltpu.sync_copy(zero_emb_vmem, looked_up_buf)

                # Phase 3 DMA: send gathered embeddings back to wave-dependent
                # source peer my_src = (my_rank - wave_idx + W) % W (matches
                # production). Use SMEM-scalar phase3_remote_start as dst start.
                my_src = (my_rank - wave_idx + world_size) % world_size
                phase3_count = recv_capacity
                d3 = pltpu.make_async_remote_copy(
                    src_ref=looked_up_buf.at[pl.ds(0, phase3_count), :],
                    dst_ref=out_ref.at[pl.ds(phase3_remote_start, phase3_count), :],
                    send_sem=send_sem_emb,
                    recv_sem=recv_sem_emb,
                    device_id=(my_src,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
                d3.start()
                d3.wait()
            return kernel

        HBM = pltpu.MemorySpace.HBM
        SMEM = pltpu.MemorySpace.SMEM

        def per_device_run(local_sorted, local_table, local_dummy,
                           p1_cnt_arr, p3_cnt_arr, send_off_arr,
                           recv_off_arr, phase3_remote_off_arr):
            # local_sorted/_table/_dummy are sharded; per-rank shapes have
            # leading axis 1 from shard_map (matches v70r). Same for the
            # five per-wave splitter arrays — each (1, W) per device.
            out_acc = jnp.zeros((OUT_PER_RANK_LEN, D), dtype=jnp.float32)
            sync_dep = jnp.float32(0.0)
            # Strip the leading axis 0 from sharded splitter arrays (each is
            # shape (1, W) per device).
            p1_cnt_local = p1_cnt_arr[0]
            p3_cnt_local = p3_cnt_arr[0]
            send_off_local = send_off_arr[0]
            recv_off_local = recv_off_arr[0]
            phase3_remote_off_local = phase3_remote_off_arr[0]
            for w_py in range(max_waves):
                out_acc_dep, _ = jax.lax.optimization_barrier(
                    (out_acc, sync_dep)
                )
                # SMEM scalar carrying this wave's index w (matches production).
                wave_idx_w = jnp.array([w_py], dtype=jnp.int32)
                p1_cnt_w = lax.dynamic_slice(p1_cnt_local, (w_py,), (1,))
                p3_cnt_w = lax.dynamic_slice(p3_cnt_local, (w_py,), (1,))
                send_off_w = lax.dynamic_slice(send_off_local, (w_py,), (1,))
                recv_off_w = lax.dynamic_slice(recv_off_local, (w_py,), (1,))
                phase3_remote_off_w = lax.dynamic_slice(
                    phase3_remote_off_local, (w_py,), (1,)
                )
                kernel = make_full_chain_kernel(world_size)
                (out_acc, _recv_dummy, _looked_up_dummy) = pl.pallas_call(
                    kernel,
                    out_shape=[
                        jax.ShapeDtypeStruct((OUT_PER_RANK_LEN, D), jnp.float32),
                        jax.ShapeDtypeStruct((FIXED_COUNT,), jnp.int32),
                        jax.ShapeDtypeStruct((recv_capacity, D), jnp.float32),
                    ],
                    grid_spec=pltpu.PrefetchScalarGridSpec(
                        num_scalar_prefetch=0,
                        in_specs=[
                            pl.BlockSpec(memory_space=HBM),   # sorted_row_ref
                            pl.BlockSpec(memory_space=HBM),   # table_ref
                            pl.BlockSpec(memory_space=HBM),   # dummy_in_ref
                            pl.BlockSpec(memory_space=SMEM),  # wave_idx_ref
                            pl.BlockSpec(memory_space=SMEM),  # p1_cnt_ref
                            pl.BlockSpec(memory_space=SMEM),  # p3_cnt_ref
                            pl.BlockSpec(memory_space=SMEM),  # send_off_ref
                            pl.BlockSpec(memory_space=SMEM),  # recv_off_ref
                            pl.BlockSpec(memory_space=SMEM),  # phase3_remote_off_ref
                            pl.BlockSpec(memory_space=HBM),   # out_init_ref
                        ],
                        out_specs=[
                            pl.BlockSpec(memory_space=HBM),   # out_ref
                            pl.BlockSpec(memory_space=HBM),   # recv_row_buf
                            pl.BlockSpec(memory_space=HBM),   # looked_up_buf
                        ],
                        scratch_shapes=[
                            pltpu.SemaphoreType.DMA,  # send_sem (Phase 1)
                            pltpu.SemaphoreType.DMA,  # recv_sem (Phase 1)
                            pltpu.SemaphoreType.DMA,  # send_sem_emb (Phase 3)
                            pltpu.SemaphoreType.DMA,  # recv_sem_emb (Phase 3)
                            pltpu.VMEM((v_local, D), jnp.float32),
                            pltpu.VMEM((recv_capacity, D), jnp.float32),
                            pltpu.SMEM((FIXED_COUNT,), jnp.int32),
                        ],
                    ),
                    input_output_aliases={9: 0},
                    compiler_params=pltpu.CompilerParams(collective_id=w_py),
                )(
                    local_sorted, local_table, local_dummy,
                    wave_idx_w, p1_cnt_w, p3_cnt_w,
                    send_off_w, recv_off_w, phase3_remote_off_w,
                    out_acc_dep,
                )
                del _recv_dummy, _looked_up_dummy
                # Sync_dep keeps waves ordered cross-rank.
                sync_dep = jax.lax.psum(out_acc[0, 0].astype(jnp.float32), "x")
            return out_acc

        # Run the production splitter chain on real skewed inputs to get
        # per-rank-varying SMEM scalar arrays.
        local_ids, local_id_count = _make_inputs_on_device(
            mesh, world_size, max_local_ids, v_local, seed=2,
        )
        log(f"local_ids shape={local_ids.shape}")

        def per_rank_splitter(ids, count):
            # ids is (1, max_local_ids); count is (1,).
            sorted_row_real, send_counts_real, _sort_perm, _reverse_perm = (
                _bucketize_and_sort(
                    ids[0], count[0], max_local_ids, world_size,
                )
            )
            del sorted_row_real, _sort_perm, _reverse_perm
            send_counts_padded = _round_up_tile(send_counts_real, _TILE_INT32)
            traffic_matrix = jax.lax.all_gather(send_counts_padded, "x")
            my_rank = jax.lax.axis_index("x")
            p1_cnt, p3_cnt, send_off, recv_off, phase3_remote_off = (
                _split_traffic_ringshift_globalpad(
                    traffic_matrix, send_counts_padded,
                    world_size=world_size, my_rank=my_rank,
                )
            )
            return (
                p1_cnt[None], p3_cnt[None], send_off[None],
                recv_off[None], phase3_remote_off[None],
            )

        splitter_fn = jax.shard_map(
            per_rank_splitter, mesh=mesh,
            in_specs=(P("x"), P("x")),
            out_specs=(P("x"), P("x"), P("x"), P("x"), P("x")),
            check_vma=False,
        )
        (p1_cnt_arr, p3_cnt_arr, send_off_arr,
         recv_off_arr, phase3_remote_off_arr) = splitter_fn(
            local_ids, local_id_count,
        )
        jax.block_until_ready((p1_cnt_arr, p3_cnt_arr, send_off_arr,
                               recv_off_arr, phase3_remote_off_arr))
        log(f"splitter outputs shape={p1_cnt_arr.shape}")

        sharded = jax.shard_map(
            per_device_run, mesh=mesh,
            in_specs=(P("x"), P("x"), P("x"),
                      P("x"), P("x"), P("x"), P("x"), P("x")),
            out_specs=P("x"),
            check_vma=False,
        )

        # Per-rank sorted_row enlarged to SORTED_PER_RANK_LEN to fit real
        # send_off_local in [0, max_padded] + extent FIXED_COUNT.
        sorted_per_rank = (
            jnp.arange(world_size * SORTED_PER_RANK_LEN, dtype=jnp.int32) % v_local
        )
        assert sorted_per_rank.shape == (world_size * SORTED_PER_RANK_LEN,)
        sorted_per_rank = jax.device_put(
            sorted_per_rank,
            jax.sharding.NamedSharding(mesh, P("x")),
        )
        table_per_rank = jnp.zeros((world_size * v_local, D), dtype=jnp.float32)
        table_per_rank = jax.device_put(
            table_per_rank,
            jax.sharding.NamedSharding(mesh, P("x")),
        )
        dummy_per_rank = jnp.zeros((world_size, 1), dtype=jnp.int32)
        dummy_per_rank = jax.device_put(
            dummy_per_rank,
            jax.sharding.NamedSharding(mesh, P("x")),
        )
        jax.block_until_ready(sorted_per_rank)
        jax.block_until_ready(table_per_rank)
        jax.block_until_ready(dummy_per_rank)
        log("inputs ready")

        bjit = jax.jit(sharded)
        log("calling kernel...")
        sync_global_devices("litmus_realsplit_start")
        t0 = time.perf_counter()
        out = bjit(
            sorted_per_rank, table_per_rank, dummy_per_rank,
            p1_cnt_arr, p3_cnt_arr, send_off_arr,
            recv_off_arr, phase3_remote_off_arr,
        )
        dispatch_ms = (time.perf_counter() - t0) * 1e3
        log(f"dispatched (after {dispatch_ms:.1f}ms), blocking...")
        jax.block_until_ready(out)
        elapsed_ms = (time.perf_counter() - t0) * 1e3
        log(f"ran in {elapsed_ms:.1f} ms total (dispatch={dispatch_ms:.1f}, block={elapsed_ms-dispatch_ms:.1f})")
        log("RESULT: WORKS")


    def test_w32_production_kernel_litmus(self) -> None:
        """W=32 litmus that calls the PRODUCTION ringshift kernel directly.

        Companion to ``test_w32_smem_real_splitter_litmus`` (v70s, which works).
        v70s reimplements the kernel inline; THIS test calls the production
        ``jagged_a2a_embedding_lookup_f32_alltc_ringshift`` (which uses
        ``_make_ringshift_single_wave_kernel`` from jagged_a2a_lookup.py).

        Same wrapper pattern: jit + sync_global_devices + checkpoint prints +
        single dispatch + block_until_ready. Same sizing as v70s
        (max_local_ids=8K, v_local=8K, D=128, recv_capacity=8K+TILE).

        If THIS hangs (as test_w32_multi_iter does), the bug is INSIDE the
        production single-wave kernel body. The diff vs the v70s litmus inline
        kernel narrows where to look:
          - ordering of pre-barrier vs sync_copy(table) / zero scratch init
          - extra recv_row_buf zero-init (sync_copy of zero_int_vmem)
          - Phase 3 src offset uses ``remote_recv_start`` (= recv_off_ref[0]),
            v70s uses ``0``
          - extent of Phase 1 DMA = real ``p1_cnt`` (variable), v70s uses
            FIXED_COUNT=128
          - extent of Phase 3 DMA = real ``p3_cnt`` (variable), v70s uses
            recv_capacity (constant)
          - Phase 3 DMA src window (looked_up_buf[remote_recv_start:p3_cnt])
            differs from v70s (looked_up_buf[0:recv_capacity])
        """
        from jax.experimental.multihost_utils import sync_global_devices
        import sys

        world_size = 32
        TILE = 128
        embedding_dim = 128
        v_local = 8 * 1024
        max_local_ids = 8 * 1024

        all_devices = jax.devices()
        if len(all_devices) < 256:
            self.skipTest(f"Need 256 logical devices, have {len(all_devices)}")
        devices = all_devices[::8]  # symmetric: 1 per host (matches v70s)
        assert len(devices) == world_size
        mesh = jax.sharding.Mesh(np.array(devices), ("x",))

        recv_capacity = max_local_ids + TILE
        recv_capacity = ((recv_capacity + TILE - 1) // TILE) * TILE

        def log(msg):
            if jax.process_index() == 0:
                print(f"[LITMUS_W32_PRODKERNEL] {msg}", file=sys.stderr, flush=True)

        log(f"=== D={embedding_dim} v_local={v_local} max_local_ids={max_local_ids} W={world_size} ===")

        log("creating inputs ...")
        local_ids, local_id_count = _make_inputs_on_device(
            mesh, world_size, max_local_ids, v_local, seed=2,
        )
        local_table = _make_table_on_device(mesh, v_local, embedding_dim)
        jax.block_until_ready((local_ids, local_id_count, local_table))
        log("inputs ready")

        rs_jit = jax.jit(functools.partial(
            jagged_a2a_embedding_lookup_f32_alltc_ringshift,
            mesh=mesh, axis_name="x",
            recv_capacity=recv_capacity, splitter="globalpad",
        ))

        log("calling production kernel...")
        sync_global_devices("litmus_prodkernel_start")
        t0 = time.perf_counter()
        out = rs_jit(local_ids, local_id_count, local_table)
        dispatch_ms = (time.perf_counter() - t0) * 1e3
        log(f"dispatched (after {dispatch_ms:.1f}ms), blocking...")
        jax.block_until_ready(out)
        elapsed_ms = (time.perf_counter() - t0) * 1e3
        log(f"ran in {elapsed_ms:.1f} ms total (dispatch={dispatch_ms:.1f}, block={elapsed_ms-dispatch_ms:.1f})")
        log("RESULT: WORKS")


    # =====================================================================
    # REVERSE-BISECT W=32 LITMUS METHODS
    #
    # Start from v70s (test_w32_smem_real_splitter_litmus, which PASSES at
    # W=32 in ~27s) and add production-specific elements one at a time,
    # building toward _make_ringshift_single_wave_kernel. Each addition is
    # tested at W=32 with the same wrapper/inputs as v70s. The first
    # addition that breaks the litmus is the smoking gun.
    #
    # Production diffs (vs v70s):
    #   A: ADD pltpu.sync_copy(zero_int_vmem, recv_row_buf) zero-init
    #   B: ADD extra (recv_capacity,) int VMEM scratch (zero_int_vmem) —
    #      paired with A (sync_copy needs the scratch to source from)
    #   C: ORDER: init FIRST, barrier AFTER (v70s: barrier first, init after)
    #   D: Phase 1 DMA extent = pl.multiple_of(p1_cnt_ref[0], TILE) (runtime)
    #      [v70s: extent = FIXED_COUNT (compile-time constant)]
    #   E: Phase 3 DMA extent = pl.multiple_of(p3_cnt_ref[0], TILE) (runtime)
    #      [v70s: extent = recv_capacity (compile-time constant)]
    #
    # Layered methods:
    #   _plus_A_litmus     : v70s + A (and B, since A requires B's scratch)
    #   _plus_AB_litmus    : same as _plus_A (A and B are inseparable;
    #                        kept for naming consistency with the plan)
    #   _plus_ABC_litmus   : + C (init first, barrier after)
    #   _plus_ABCDE_litmus : + D, E (runtime Phase 1/3 extents from SMEM)
    # =====================================================================

    def _run_w32_reverse_bisect(self, *, label: str, kernel_factory) -> None:
        """Shared driver for the W=32 reverse-bisect litmus suite.

        Uses the same wrapper/inputs as test_w32_smem_real_splitter_litmus
        (v70s). The kernel body differs per step via ``kernel_factory``.
        """
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        from jax import lax
        from jax.experimental.multihost_utils import sync_global_devices
        from recir.ops.pallas.embeddings.jagged_a2a_lookup import (
            _bucketize_and_sort,
            _round_up_tile,
            _split_traffic_ringshift_globalpad,
            _TILE_INT32,
        )
        import sys

        world_size = 32
        max_waves = world_size
        P = jax.sharding.PartitionSpec

        all_devices = jax.devices()
        if len(all_devices) < 256:
            self.skipTest(f"Need 256 logical devices, have {len(all_devices)}")
        devices = all_devices[::8]
        assert len(devices) == world_size
        mesh = jax.sharding.Mesh(np.array(devices), ("x",))

        def log(msg):
            if jax.process_index() == 0:
                print(f"[LITMUS_W32_REV_{label}] {msg}", file=sys.stderr, flush=True)

        max_local_ids = 8 * 1024
        v_local = 8 * 1024
        recv_capacity = 8 * 1024 + 128
        D = 128
        TILE = 128
        FIXED_COUNT = 128
        max_padded = max_local_ids + world_size * TILE
        SORTED_PER_RANK_LEN = max_padded + FIXED_COUNT
        OUT_PER_RANK_LEN = max_padded + recv_capacity

        log(f"max_padded={max_padded} SORTED={SORTED_PER_RANK_LEN} OUT={OUT_PER_RANK_LEN}")

        HBM = pltpu.MemorySpace.HBM
        SMEM = pltpu.MemorySpace.SMEM

        kernel = kernel_factory(
            world_size=world_size,
            FIXED_COUNT=FIXED_COUNT,
            recv_capacity=recv_capacity,
            TILE=TILE,
        )

        # Each step uses one of two scratch layouts:
        #   layout "v70s"  — original v70s scratches (no zero_int_vmem)
        #   layout "withZ" — adds zero_int_vmem just before idx_smem (matches
        #                    production ordering: ints VMEM, then emb VMEM,
        #                    then table VMEM, then idx SMEM). To minimize
        #                    structural diffs vs v70s, we INSERT zero_int_vmem
        #                    at the start of the VMEM block.
        #
        # The kernel_factory tells us which layout it expects via its return
        # tuple: (kernel_fn, scratch_layout) — but to keep the API simple we
        # let the kernel_factory return just the fn and assume layout based on
        # label.
        if label.startswith("A") or label.startswith("AB") or label.startswith("ABC"):
            # When diff B is in play, zero_int_vmem is sized (recv_capacity,)
            # and the sync_copy requires recv_row_buf to also be sized
            # (recv_capacity,). idx_smem grows accordingly so the
            # sync_copy(recv_row_buf, idx_smem) shapes match. These size
            # changes are inseparable from diff B and are NOT a new diff.
            recv_row_buf_len = recv_capacity
            idx_smem_len = recv_capacity
            scratch_shapes = [
                pltpu.SemaphoreType.DMA,  # send_sem (Phase 1)
                pltpu.SemaphoreType.DMA,  # recv_sem (Phase 1)
                pltpu.SemaphoreType.DMA,  # send_sem_emb (Phase 3)
                pltpu.SemaphoreType.DMA,  # recv_sem_emb (Phase 3)
                pltpu.VMEM((recv_capacity,), jnp.int32),  # zero_int_vmem (NEW: diff B)
                pltpu.VMEM((v_local, D), jnp.float32),     # table_vmem
                pltpu.VMEM((recv_capacity, D), jnp.float32),  # zero_emb_vmem
                pltpu.SMEM((idx_smem_len,), jnp.int32),     # idx_smem
            ]
            has_zero_int = True
        else:
            # Should not happen given our 4 step labels, but safe fallback.
            recv_row_buf_len = FIXED_COUNT
            idx_smem_len = FIXED_COUNT
            scratch_shapes = [
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.VMEM((v_local, D), jnp.float32),
                pltpu.VMEM((recv_capacity, D), jnp.float32),
                pltpu.SMEM((FIXED_COUNT,), jnp.int32),
            ]
            has_zero_int = False

        def per_device_run(local_sorted, local_table, local_dummy,
                           p1_cnt_arr, p3_cnt_arr, send_off_arr,
                           recv_off_arr, phase3_remote_off_arr):
            out_acc = jnp.zeros((OUT_PER_RANK_LEN, D), dtype=jnp.float32)
            sync_dep = jnp.float32(0.0)
            p1_cnt_local = p1_cnt_arr[0]
            p3_cnt_local = p3_cnt_arr[0]
            send_off_local = send_off_arr[0]
            recv_off_local = recv_off_arr[0]
            phase3_remote_off_local = phase3_remote_off_arr[0]
            for w_py in range(max_waves):
                out_acc_dep, _ = jax.lax.optimization_barrier(
                    (out_acc, sync_dep)
                )
                wave_idx_w = jnp.array([w_py], dtype=jnp.int32)
                p1_cnt_w = lax.dynamic_slice(p1_cnt_local, (w_py,), (1,))
                p3_cnt_w = lax.dynamic_slice(p3_cnt_local, (w_py,), (1,))
                send_off_w = lax.dynamic_slice(send_off_local, (w_py,), (1,))
                recv_off_w = lax.dynamic_slice(recv_off_local, (w_py,), (1,))
                phase3_remote_off_w = lax.dynamic_slice(
                    phase3_remote_off_local, (w_py,), (1,)
                )
                (out_acc, _recv_dummy, _looked_up_dummy) = pl.pallas_call(
                    kernel,
                    out_shape=[
                        jax.ShapeDtypeStruct((OUT_PER_RANK_LEN, D), jnp.float32),
                        jax.ShapeDtypeStruct((recv_row_buf_len,), jnp.int32),
                        jax.ShapeDtypeStruct((recv_capacity, D), jnp.float32),
                    ],
                    grid_spec=pltpu.PrefetchScalarGridSpec(
                        num_scalar_prefetch=0,
                        in_specs=[
                            pl.BlockSpec(memory_space=HBM),
                            pl.BlockSpec(memory_space=HBM),
                            pl.BlockSpec(memory_space=HBM),
                            pl.BlockSpec(memory_space=SMEM),
                            pl.BlockSpec(memory_space=SMEM),
                            pl.BlockSpec(memory_space=SMEM),
                            pl.BlockSpec(memory_space=SMEM),
                            pl.BlockSpec(memory_space=SMEM),
                            pl.BlockSpec(memory_space=SMEM),
                            pl.BlockSpec(memory_space=HBM),
                        ],
                        out_specs=[
                            pl.BlockSpec(memory_space=HBM),
                            pl.BlockSpec(memory_space=HBM),
                            pl.BlockSpec(memory_space=HBM),
                        ],
                        scratch_shapes=scratch_shapes,
                    ),
                    input_output_aliases={9: 0},
                    compiler_params=pltpu.CompilerParams(collective_id=w_py),
                )(
                    local_sorted, local_table, local_dummy,
                    wave_idx_w, p1_cnt_w, p3_cnt_w,
                    send_off_w, recv_off_w, phase3_remote_off_w,
                    out_acc_dep,
                )
                del _recv_dummy, _looked_up_dummy
                sync_dep = jax.lax.psum(out_acc[0, 0].astype(jnp.float32), "x")
            return out_acc

        local_ids, local_id_count = _make_inputs_on_device(
            mesh, world_size, max_local_ids, v_local, seed=2,
        )
        log(f"local_ids shape={local_ids.shape}")

        def per_rank_splitter(ids, count):
            sorted_row_real, send_counts_real, _sort_perm, _reverse_perm = (
                _bucketize_and_sort(
                    ids[0], count[0], max_local_ids, world_size,
                )
            )
            del sorted_row_real, _sort_perm, _reverse_perm
            send_counts_padded = _round_up_tile(send_counts_real, _TILE_INT32)
            traffic_matrix = jax.lax.all_gather(send_counts_padded, "x")
            my_rank = jax.lax.axis_index("x")
            p1_cnt, p3_cnt, send_off, recv_off, phase3_remote_off = (
                _split_traffic_ringshift_globalpad(
                    traffic_matrix, send_counts_padded,
                    world_size=world_size, my_rank=my_rank,
                )
            )
            return (
                p1_cnt[None], p3_cnt[None], send_off[None],
                recv_off[None], phase3_remote_off[None],
            )

        splitter_fn = jax.shard_map(
            per_rank_splitter, mesh=mesh,
            in_specs=(P("x"), P("x")),
            out_specs=(P("x"), P("x"), P("x"), P("x"), P("x")),
            check_vma=False,
        )
        (p1_cnt_arr, p3_cnt_arr, send_off_arr,
         recv_off_arr, phase3_remote_off_arr) = splitter_fn(
            local_ids, local_id_count,
        )
        jax.block_until_ready((p1_cnt_arr, p3_cnt_arr, send_off_arr,
                               recv_off_arr, phase3_remote_off_arr))
        log(f"splitter outputs shape={p1_cnt_arr.shape}")

        sharded = jax.shard_map(
            per_device_run, mesh=mesh,
            in_specs=(P("x"), P("x"), P("x"),
                      P("x"), P("x"), P("x"), P("x"), P("x")),
            out_specs=P("x"),
            check_vma=False,
        )

        sorted_per_rank = (
            jnp.arange(world_size * SORTED_PER_RANK_LEN, dtype=jnp.int32) % v_local
        )
        sorted_per_rank = jax.device_put(
            sorted_per_rank,
            jax.sharding.NamedSharding(mesh, P("x")),
        )
        table_per_rank = jnp.zeros((world_size * v_local, D), dtype=jnp.float32)
        table_per_rank = jax.device_put(
            table_per_rank,
            jax.sharding.NamedSharding(mesh, P("x")),
        )
        dummy_per_rank = jnp.zeros((world_size, 1), dtype=jnp.int32)
        dummy_per_rank = jax.device_put(
            dummy_per_rank,
            jax.sharding.NamedSharding(mesh, P("x")),
        )
        jax.block_until_ready(sorted_per_rank)
        jax.block_until_ready(table_per_rank)
        jax.block_until_ready(dummy_per_rank)
        log(f"inputs ready (has_zero_int_scratch={has_zero_int})")

        bjit = jax.jit(sharded)
        log("calling kernel...")
        sync_global_devices(f"litmus_rev_{label}_start")
        t0 = time.perf_counter()
        out = bjit(
            sorted_per_rank, table_per_rank, dummy_per_rank,
            p1_cnt_arr, p3_cnt_arr, send_off_arr,
            recv_off_arr, phase3_remote_off_arr,
        )
        dispatch_ms = (time.perf_counter() - t0) * 1e3
        log(f"dispatched (after {dispatch_ms:.1f}ms), blocking...")
        jax.block_until_ready(out)
        elapsed_ms = (time.perf_counter() - t0) * 1e3
        log(f"ran in {elapsed_ms:.1f} ms total (dispatch={dispatch_ms:.1f}, block={elapsed_ms-dispatch_ms:.1f})")
        log("RESULT: WORKS")


    @staticmethod
    def _make_kernel_plus_A(*, world_size, FIXED_COUNT, recv_capacity, TILE):
        """v70s kernel + diff A (sync_copy zero into recv_row_buf) and the
        required diff B (zero_int_vmem scratch)."""
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        from jax import lax

        def kernel(sorted_row_ref, table_ref, dummy_in_ref,
                   wave_idx_ref, p1_cnt_ref, p3_cnt_ref,
                   send_off_ref, recv_off_ref, phase3_remote_off_ref,
                   out_init_ref,
                   out_ref, recv_row_buf, looked_up_buf,
                   send_sem, recv_sem, send_sem_emb, recv_sem_emb,
                   zero_int_vmem,  # NEW (diff B)
                   table_vmem, zero_emb_vmem, idx_smem):
            del dummy_in_ref, out_init_ref
            my_rank = lax.axis_index("x")
            wave_idx = wave_idx_ref[0]

            p1_cnt = pl.multiple_of(p1_cnt_ref[0], TILE)
            p3_cnt = pl.multiple_of(p3_cnt_ref[0], TILE)
            send_start = pl.multiple_of(send_off_ref[0], TILE)
            remote_recv_start = pl.multiple_of(recv_off_ref[0], TILE)
            phase3_remote_start = pl.multiple_of(
                phase3_remote_off_ref[0], TILE
            )
            _ = p1_cnt + p3_cnt

            # v70s order: barrier FIRST, init AFTER (NOT swapped here — that's diff C)
            sem = pltpu.get_barrier_semaphore()
            for peer in range(world_size):
                pl.semaphore_signal(sem, 1, device_id=(peer,),
                                    device_id_type=pltpu.DeviceIdType.MESH)
            pl.semaphore_wait(sem, world_size)

            my_dst = (my_rank + wave_idx) % world_size
            d = pltpu.make_async_remote_copy(
                src_ref=sorted_row_ref.at[pl.ds(send_start, FIXED_COUNT)],
                dst_ref=recv_row_buf.at[pl.ds(remote_recv_start, FIXED_COUNT)],
                send_sem=send_sem, recv_sem=recv_sem,
                device_id=(my_dst,),
                device_id_type=pltpu.DeviceIdType.MESH,
            )
            d.start()
            d.wait()

            pltpu.sync_copy(table_ref, table_vmem)
            zero_emb_vmem[...] = jnp.zeros_like(zero_emb_vmem)

            # === DIFF A: zero recv_row_buf via sync_copy from zero_int_vmem ===
            zero_int_vmem[...] = jnp.zeros_like(zero_int_vmem)
            pltpu.sync_copy(zero_int_vmem, recv_row_buf)

            pltpu.sync_copy(recv_row_buf, idx_smem)

            def gather_body(i, __):
                row_idx = idx_smem[i % FIXED_COUNT]
                zero_emb_vmem[pl.ds(i, 1), :] = table_vmem[pl.ds(row_idx, 1), :]
                return None
            lax.fori_loop(0, recv_capacity, gather_body, None)

            pltpu.sync_copy(zero_emb_vmem, looked_up_buf)

            my_src = (my_rank - wave_idx + world_size) % world_size
            phase3_count = recv_capacity
            d3 = pltpu.make_async_remote_copy(
                src_ref=looked_up_buf.at[pl.ds(0, phase3_count), :],
                dst_ref=out_ref.at[pl.ds(phase3_remote_start, phase3_count), :],
                send_sem=send_sem_emb, recv_sem=recv_sem_emb,
                device_id=(my_src,),
                device_id_type=pltpu.DeviceIdType.MESH,
            )
            d3.start()
            d3.wait()
        return kernel


    @staticmethod
    def _make_kernel_plus_ABC(*, world_size, FIXED_COUNT, recv_capacity, TILE):
        """v70s kernel + diff A (sync_copy zero), B (zero_int_vmem scratch),
        C (init FIRST, barrier AFTER — production order)."""
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        from jax import lax

        def kernel(sorted_row_ref, table_ref, dummy_in_ref,
                   wave_idx_ref, p1_cnt_ref, p3_cnt_ref,
                   send_off_ref, recv_off_ref, phase3_remote_off_ref,
                   out_init_ref,
                   out_ref, recv_row_buf, looked_up_buf,
                   send_sem, recv_sem, send_sem_emb, recv_sem_emb,
                   zero_int_vmem,
                   table_vmem, zero_emb_vmem, idx_smem):
            del dummy_in_ref, out_init_ref
            my_rank = lax.axis_index("x")
            wave_idx = wave_idx_ref[0]

            p1_cnt = pl.multiple_of(p1_cnt_ref[0], TILE)
            p3_cnt = pl.multiple_of(p3_cnt_ref[0], TILE)
            send_start = pl.multiple_of(send_off_ref[0], TILE)
            remote_recv_start = pl.multiple_of(recv_off_ref[0], TILE)
            phase3_remote_start = pl.multiple_of(
                phase3_remote_off_ref[0], TILE
            )
            _ = p1_cnt + p3_cnt

            # === DIFF C: init FIRST (production order) ===
            pltpu.sync_copy(table_ref, table_vmem)
            zero_int_vmem[...] = jnp.zeros_like(zero_int_vmem)
            zero_emb_vmem[...] = jnp.zeros_like(zero_emb_vmem)
            pltpu.sync_copy(zero_int_vmem, recv_row_buf)

            # Barrier AFTER init (production order).
            sem = pltpu.get_barrier_semaphore()
            for peer in range(world_size):
                pl.semaphore_signal(sem, 1, device_id=(peer,),
                                    device_id_type=pltpu.DeviceIdType.MESH)
            pl.semaphore_wait(sem, world_size)

            my_dst = (my_rank + wave_idx) % world_size
            d = pltpu.make_async_remote_copy(
                src_ref=sorted_row_ref.at[pl.ds(send_start, FIXED_COUNT)],
                dst_ref=recv_row_buf.at[pl.ds(remote_recv_start, FIXED_COUNT)],
                send_sem=send_sem, recv_sem=recv_sem,
                device_id=(my_dst,),
                device_id_type=pltpu.DeviceIdType.MESH,
            )
            d.start()
            d.wait()

            pltpu.sync_copy(recv_row_buf, idx_smem)

            def gather_body(i, __):
                row_idx = idx_smem[i % FIXED_COUNT]
                zero_emb_vmem[pl.ds(i, 1), :] = table_vmem[pl.ds(row_idx, 1), :]
                return None
            lax.fori_loop(0, recv_capacity, gather_body, None)

            pltpu.sync_copy(zero_emb_vmem, looked_up_buf)

            my_src = (my_rank - wave_idx + world_size) % world_size
            phase3_count = recv_capacity
            d3 = pltpu.make_async_remote_copy(
                src_ref=looked_up_buf.at[pl.ds(0, phase3_count), :],
                dst_ref=out_ref.at[pl.ds(phase3_remote_start, phase3_count), :],
                send_sem=send_sem_emb, recv_sem=recv_sem_emb,
                device_id=(my_src,),
                device_id_type=pltpu.DeviceIdType.MESH,
            )
            d3.start()
            d3.wait()
        return kernel


    @staticmethod
    def _make_kernel_plus_ABCDE(*, world_size, FIXED_COUNT, recv_capacity, TILE):
        """v70s kernel + diffs A, B, C, D, E:
          D: Phase 1 DMA extent = pl.multiple_of(p1_cnt_ref[0], TILE) (runtime)
          E: Phase 3 DMA extent = pl.multiple_of(p3_cnt_ref[0], TILE) (runtime)
        """
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        from jax import lax

        def kernel(sorted_row_ref, table_ref, dummy_in_ref,
                   wave_idx_ref, p1_cnt_ref, p3_cnt_ref,
                   send_off_ref, recv_off_ref, phase3_remote_off_ref,
                   out_init_ref,
                   out_ref, recv_row_buf, looked_up_buf,
                   send_sem, recv_sem, send_sem_emb, recv_sem_emb,
                   zero_int_vmem,
                   table_vmem, zero_emb_vmem, idx_smem):
            del dummy_in_ref, out_init_ref
            my_rank = lax.axis_index("x")
            wave_idx = wave_idx_ref[0]

            p1_cnt = pl.multiple_of(p1_cnt_ref[0], TILE)
            p3_cnt = pl.multiple_of(p3_cnt_ref[0], TILE)
            send_start = pl.multiple_of(send_off_ref[0], TILE)
            remote_recv_start = pl.multiple_of(recv_off_ref[0], TILE)
            phase3_remote_start = pl.multiple_of(
                phase3_remote_off_ref[0], TILE
            )

            # Init first (diff C).
            pltpu.sync_copy(table_ref, table_vmem)
            zero_int_vmem[...] = jnp.zeros_like(zero_int_vmem)
            zero_emb_vmem[...] = jnp.zeros_like(zero_emb_vmem)
            pltpu.sync_copy(zero_int_vmem, recv_row_buf)

            sem = pltpu.get_barrier_semaphore()
            for peer in range(world_size):
                pl.semaphore_signal(sem, 1, device_id=(peer,),
                                    device_id_type=pltpu.DeviceIdType.MESH)
            pl.semaphore_wait(sem, world_size)

            # === DIFF D: Phase 1 extent = runtime p1_cnt ===
            # NO pl.when guard (matches v70s; production has @pl.when(p1_cnt>0)
            # but we keep the v70s no-guard pattern to isolate just the extent
            # change).
            my_dst = (my_rank + wave_idx) % world_size
            d = pltpu.make_async_remote_copy(
                src_ref=sorted_row_ref.at[pl.ds(send_start, p1_cnt)],
                dst_ref=recv_row_buf.at[pl.ds(remote_recv_start, p1_cnt)],
                send_sem=send_sem, recv_sem=recv_sem,
                device_id=(my_dst,),
                device_id_type=pltpu.DeviceIdType.MESH,
            )
            d.start()
            d.wait()

            pltpu.sync_copy(recv_row_buf, idx_smem)

            def gather_body(i, __):
                row_idx = idx_smem[i % FIXED_COUNT]
                zero_emb_vmem[pl.ds(i, 1), :] = table_vmem[pl.ds(row_idx, 1), :]
                return None
            lax.fori_loop(0, recv_capacity, gather_body, None)

            pltpu.sync_copy(zero_emb_vmem, looked_up_buf)

            # === DIFF E: Phase 3 extent = runtime p3_cnt ===
            # Still using src offset 0 (NOT diff F) and dst offset
            # phase3_remote_start (already runtime in v70s).
            my_src = (my_rank - wave_idx + world_size) % world_size
            d3 = pltpu.make_async_remote_copy(
                src_ref=looked_up_buf.at[pl.ds(0, p3_cnt), :],
                dst_ref=out_ref.at[pl.ds(phase3_remote_start, p3_cnt), :],
                send_sem=send_sem_emb, recv_sem=recv_sem_emb,
                device_id=(my_src,),
                device_id_type=pltpu.DeviceIdType.MESH,
            )
            d3.start()
            d3.wait()
        return kernel


    def test_w32_v70s_plus_A_litmus(self) -> None:
        """v70s + diff A (sync_copy zero into recv_row_buf).

        Note: diff A requires diff B (the zero_int_vmem scratch) to source
        from. So this method effectively tests A+B together; the next method
        test_w32_v70s_plus_AB_litmus is identical.
        """
        self._run_w32_reverse_bisect(
            label="A",
            kernel_factory=self._make_kernel_plus_A,
        )


    def test_w32_v70s_plus_AB_litmus(self) -> None:
        """v70s + diffs A and B (zero recv_row_buf + extra int VMEM scratch).

        Identical to _plus_A_litmus since A and B are inseparable; this
        method is kept for naming consistency with the bisect plan.
        """
        self._run_w32_reverse_bisect(
            label="AB",
            kernel_factory=self._make_kernel_plus_A,
        )


    def test_w32_v70s_plus_ABC_litmus(self) -> None:
        """v70s + diffs A, B, C (init FIRST, barrier AFTER — production order)."""
        self._run_w32_reverse_bisect(
            label="ABC",
            kernel_factory=self._make_kernel_plus_ABC,
        )


    def test_w32_v70s_plus_ABCDE_litmus(self) -> None:
        """v70s + diffs A, B, C, D, E (also runtime Phase 1 & 3 DMA extents)."""
        self._run_w32_reverse_bisect(
            label="ABCDE",
            kernel_factory=self._make_kernel_plus_ABCDE,
        )


    @staticmethod
    def _make_kernel_plus_ABCD(*, world_size, FIXED_COUNT, recv_capacity, TILE):
        """v70s kernel + diffs A, B, C, D ONLY (Phase 1 runtime extent).
          D: Phase 1 DMA extent = pl.multiple_of(p1_cnt_ref[0], TILE) (runtime)
          Phase 3 extent stays at recv_capacity (compile-time const) — no E.
        Used to isolate whether D alone is sufficient to trigger the W=32 hang.
        """
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        from jax import lax

        def kernel(sorted_row_ref, table_ref, dummy_in_ref,
                   wave_idx_ref, p1_cnt_ref, p3_cnt_ref,
                   send_off_ref, recv_off_ref, phase3_remote_off_ref,
                   out_init_ref,
                   out_ref, recv_row_buf, looked_up_buf,
                   send_sem, recv_sem, send_sem_emb, recv_sem_emb,
                   zero_int_vmem,
                   table_vmem, zero_emb_vmem, idx_smem):
            del dummy_in_ref, out_init_ref
            my_rank = lax.axis_index("x")
            wave_idx = wave_idx_ref[0]

            p1_cnt = pl.multiple_of(p1_cnt_ref[0], TILE)
            p3_cnt = pl.multiple_of(p3_cnt_ref[0], TILE)
            send_start = pl.multiple_of(send_off_ref[0], TILE)
            remote_recv_start = pl.multiple_of(recv_off_ref[0], TILE)
            phase3_remote_start = pl.multiple_of(
                phase3_remote_off_ref[0], TILE
            )
            _ = p3_cnt  # keep symbol live; not used as DMA extent here.

            # Init first (diff C).
            pltpu.sync_copy(table_ref, table_vmem)
            zero_int_vmem[...] = jnp.zeros_like(zero_int_vmem)
            zero_emb_vmem[...] = jnp.zeros_like(zero_emb_vmem)
            pltpu.sync_copy(zero_int_vmem, recv_row_buf)

            sem = pltpu.get_barrier_semaphore()
            for peer in range(world_size):
                pl.semaphore_signal(sem, 1, device_id=(peer,),
                                    device_id_type=pltpu.DeviceIdType.MESH)
            pl.semaphore_wait(sem, world_size)

            # === DIFF D: Phase 1 extent = runtime p1_cnt (no E) ===
            my_dst = (my_rank + wave_idx) % world_size
            d = pltpu.make_async_remote_copy(
                src_ref=sorted_row_ref.at[pl.ds(send_start, p1_cnt)],
                dst_ref=recv_row_buf.at[pl.ds(remote_recv_start, p1_cnt)],
                send_sem=send_sem, recv_sem=recv_sem,
                device_id=(my_dst,),
                device_id_type=pltpu.DeviceIdType.MESH,
            )
            d.start()
            d.wait()

            pltpu.sync_copy(recv_row_buf, idx_smem)

            def gather_body(i, __):
                row_idx = idx_smem[i % FIXED_COUNT]
                zero_emb_vmem[pl.ds(i, 1), :] = table_vmem[pl.ds(row_idx, 1), :]
                return None
            lax.fori_loop(0, recv_capacity, gather_body, None)

            pltpu.sync_copy(zero_emb_vmem, looked_up_buf)

            # NO E: Phase 3 extent stays compile-time const (recv_capacity).
            my_src = (my_rank - wave_idx + world_size) % world_size
            phase3_count = recv_capacity
            d3 = pltpu.make_async_remote_copy(
                src_ref=looked_up_buf.at[pl.ds(0, phase3_count), :],
                dst_ref=out_ref.at[pl.ds(phase3_remote_start, phase3_count), :],
                send_sem=send_sem_emb, recv_sem=recv_sem_emb,
                device_id=(my_src,),
                device_id_type=pltpu.DeviceIdType.MESH,
            )
            d3.start()
            d3.wait()
        return kernel


    @staticmethod
    def _make_kernel_plus_ABCE(*, world_size, FIXED_COUNT, recv_capacity, TILE):
        """v70s kernel + diffs A, B, C, E ONLY (Phase 3 runtime extent).
          E: Phase 3 DMA extent = pl.multiple_of(p3_cnt_ref[0], TILE) (runtime)
          Phase 1 extent stays at FIXED_COUNT (compile-time const) — no D.
        Used to isolate whether E alone is sufficient to trigger the W=32 hang.
        """
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        from jax import lax

        def kernel(sorted_row_ref, table_ref, dummy_in_ref,
                   wave_idx_ref, p1_cnt_ref, p3_cnt_ref,
                   send_off_ref, recv_off_ref, phase3_remote_off_ref,
                   out_init_ref,
                   out_ref, recv_row_buf, looked_up_buf,
                   send_sem, recv_sem, send_sem_emb, recv_sem_emb,
                   zero_int_vmem,
                   table_vmem, zero_emb_vmem, idx_smem):
            del dummy_in_ref, out_init_ref
            my_rank = lax.axis_index("x")
            wave_idx = wave_idx_ref[0]

            p1_cnt = pl.multiple_of(p1_cnt_ref[0], TILE)
            p3_cnt = pl.multiple_of(p3_cnt_ref[0], TILE)
            send_start = pl.multiple_of(send_off_ref[0], TILE)
            remote_recv_start = pl.multiple_of(recv_off_ref[0], TILE)
            phase3_remote_start = pl.multiple_of(
                phase3_remote_off_ref[0], TILE
            )
            _ = p1_cnt  # keep symbol live; not used as DMA extent here.

            # Init first (diff C).
            pltpu.sync_copy(table_ref, table_vmem)
            zero_int_vmem[...] = jnp.zeros_like(zero_int_vmem)
            zero_emb_vmem[...] = jnp.zeros_like(zero_emb_vmem)
            pltpu.sync_copy(zero_int_vmem, recv_row_buf)

            sem = pltpu.get_barrier_semaphore()
            for peer in range(world_size):
                pl.semaphore_signal(sem, 1, device_id=(peer,),
                                    device_id_type=pltpu.DeviceIdType.MESH)
            pl.semaphore_wait(sem, world_size)

            # NO D: Phase 1 extent stays compile-time const (FIXED_COUNT).
            my_dst = (my_rank + wave_idx) % world_size
            d = pltpu.make_async_remote_copy(
                src_ref=sorted_row_ref.at[pl.ds(send_start, FIXED_COUNT)],
                dst_ref=recv_row_buf.at[pl.ds(remote_recv_start, FIXED_COUNT)],
                send_sem=send_sem, recv_sem=recv_sem,
                device_id=(my_dst,),
                device_id_type=pltpu.DeviceIdType.MESH,
            )
            d.start()
            d.wait()

            pltpu.sync_copy(recv_row_buf, idx_smem)

            def gather_body(i, __):
                row_idx = idx_smem[i % FIXED_COUNT]
                zero_emb_vmem[pl.ds(i, 1), :] = table_vmem[pl.ds(row_idx, 1), :]
                return None
            lax.fori_loop(0, recv_capacity, gather_body, None)

            pltpu.sync_copy(zero_emb_vmem, looked_up_buf)

            # === DIFF E: Phase 3 extent = runtime p3_cnt (no D) ===
            my_src = (my_rank - wave_idx + world_size) % world_size
            d3 = pltpu.make_async_remote_copy(
                src_ref=looked_up_buf.at[pl.ds(0, p3_cnt), :],
                dst_ref=out_ref.at[pl.ds(phase3_remote_start, p3_cnt), :],
                send_sem=send_sem_emb, recv_sem=recv_sem_emb,
                device_id=(my_src,),
                device_id_type=pltpu.DeviceIdType.MESH,
            )
            d3.start()
            d3.wait()
        return kernel


    def test_w32_v70s_plus_ABCD_litmus(self) -> None:
        """v70s + diffs A, B, C, D ONLY (runtime Phase 1 extent; Phase 3
        stays compile-time const). Bisects D vs E."""
        self._run_w32_reverse_bisect(
            label="ABCD",
            kernel_factory=self._make_kernel_plus_ABCD,
        )


    def test_w32_v70s_plus_ABCE_litmus(self) -> None:
        """v70s + diffs A, B, C, E ONLY (runtime Phase 3 extent; Phase 1
        stays compile-time const). Bisects D vs E."""
        self._run_w32_reverse_bisect(
            label="ABCE",
            kernel_factory=self._make_kernel_plus_ABCE,
        )


    @staticmethod
    def _make_kernel_plus_ABCDE_with_guards(
        *, world_size, FIXED_COUNT, recv_capacity, TILE,
    ):
        """Identical to _make_kernel_plus_ABCDE but wraps Phase 1 and Phase 3
        DMAs in pl.when(cnt > 0) guards so 0-length DMAs are skipped.

        Hypothesis being tested: the ABCDE hang at W=32 is caused by 0-length
        DMAs (runtime extent == 0) deadlocking the TPU async copy engine. If
        this kernel WORKS, the smoking gun is conclusively 0-length DMAs.
        """
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        from jax import lax

        def kernel(sorted_row_ref, table_ref, dummy_in_ref,
                   wave_idx_ref, p1_cnt_ref, p3_cnt_ref,
                   send_off_ref, recv_off_ref, phase3_remote_off_ref,
                   out_init_ref,
                   out_ref, recv_row_buf, looked_up_buf,
                   send_sem, recv_sem, send_sem_emb, recv_sem_emb,
                   zero_int_vmem,
                   table_vmem, zero_emb_vmem, idx_smem):
            del dummy_in_ref, out_init_ref
            my_rank = lax.axis_index("x")
            wave_idx = wave_idx_ref[0]

            p1_cnt = pl.multiple_of(p1_cnt_ref[0], TILE)
            p3_cnt = pl.multiple_of(p3_cnt_ref[0], TILE)
            send_start = pl.multiple_of(send_off_ref[0], TILE)
            remote_recv_start = pl.multiple_of(recv_off_ref[0], TILE)
            phase3_remote_start = pl.multiple_of(
                phase3_remote_off_ref[0], TILE
            )

            # Init first (diff C).
            pltpu.sync_copy(table_ref, table_vmem)
            zero_int_vmem[...] = jnp.zeros_like(zero_int_vmem)
            zero_emb_vmem[...] = jnp.zeros_like(zero_emb_vmem)
            pltpu.sync_copy(zero_int_vmem, recv_row_buf)

            sem = pltpu.get_barrier_semaphore()
            for peer in range(world_size):
                pl.semaphore_signal(sem, 1, device_id=(peer,),
                                    device_id_type=pltpu.DeviceIdType.MESH)
            pl.semaphore_wait(sem, world_size)

            # === DIFF D + GUARD: Phase 1 DMA gated by p1_cnt > 0 ===
            my_dst = (my_rank + wave_idx) % world_size

            @pl.when(p1_cnt > 0)
            def _do_phase1_dma():
                d = pltpu.make_async_remote_copy(
                    src_ref=sorted_row_ref.at[pl.ds(send_start, p1_cnt)],
                    dst_ref=recv_row_buf.at[pl.ds(remote_recv_start, p1_cnt)],
                    send_sem=send_sem, recv_sem=recv_sem,
                    device_id=(my_dst,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
                d.start()
                d.wait()

            pltpu.sync_copy(recv_row_buf, idx_smem)

            def gather_body(i, __):
                row_idx = idx_smem[i % FIXED_COUNT]
                zero_emb_vmem[pl.ds(i, 1), :] = table_vmem[pl.ds(row_idx, 1), :]
                return None
            lax.fori_loop(0, recv_capacity, gather_body, None)

            pltpu.sync_copy(zero_emb_vmem, looked_up_buf)

            # === DIFF E + GUARD: Phase 3 DMA gated by p3_cnt > 0 ===
            my_src = (my_rank - wave_idx + world_size) % world_size

            @pl.when(p3_cnt > 0)
            def _do_phase3_dma():
                d3 = pltpu.make_async_remote_copy(
                    src_ref=looked_up_buf.at[pl.ds(0, p3_cnt), :],
                    dst_ref=out_ref.at[pl.ds(phase3_remote_start, p3_cnt), :],
                    send_sem=send_sem_emb, recv_sem=recv_sem_emb,
                    device_id=(my_src,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
                d3.start()
                d3.wait()
        return kernel


    def test_w32_v70s_plus_ABCDE_with_guards_litmus(self) -> None:
        """v70s + diffs A, B, C, D, E with pl.when(cnt > 0) guards on Phase 1
        and Phase 3 DMAs.

        Diagnostic: if this WORKS but ABCDE HANGS, the smoking gun is
        conclusively 0-length DMAs at W=32 (runtime extent == 0 deadlocks the
        async copy engine).
        """
        self._run_w32_reverse_bisect(
            label="ABCDE_GUARDED",
            kernel_factory=self._make_kernel_plus_ABCDE_with_guards,
        )


if __name__ == "__main__":
    unittest.main()
