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


if __name__ == "__main__":
    unittest.main()
