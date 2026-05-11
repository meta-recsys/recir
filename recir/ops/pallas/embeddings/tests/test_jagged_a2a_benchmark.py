# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""3-way perf benchmark on real TPU v7x: v23_split vs ra2a_fallback vs fused_wo4.

Per-impl recv_capacity rationale (W=4, TILE=128):
  * v23_split: pick recv_capacity >= max_local_ids + W*TILE so max_waves=1
    (multi-wave v23 is unproven on real TPU; same WO#4 deadlock fingerprint
    may apply to its cross-kernel DMA pairing). Round up to multiple of 256.
  * ra2a_fallback: no recv_capacity (pre-sizes worst case internally).
  * fused_wo4: minimum WO#4-compatible recv_capacity that fits the per-wave
    fan-in W * ceil(max_local_ids/(max_waves*TILE)) * TILE; we just use the
    same value as v23 (single-wave is also valid for WO#4).

5 warmup + 20 measurement iters per cell. Each impl is `jax.jit`'d once so
the warmup absorbs the JIT cost and steady-state iters are real kernel time.
"""

from __future__ import annotations

import functools
import statistics
import time
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from recir.ops.pallas.embeddings.jagged_a2a_lookup import (
    jagged_a2a_embedding_lookup,
    jagged_a2a_embedding_lookup_f32_alltc,
    jagged_a2a_embedding_lookup_f32_alltc_ringshift,
    jagged_a2a_embedding_lookup_reference_jax,
)
from recir.ops.pallas.embeddings.tests.test_jagged_a2a_lookup import (
    _build_local_table,
    _has_tpu,
    _make_skewed_inputs,
)


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


@unittest.skipUnless(_has_tpu(), "Benchmark requires real TPU")
class JaggedA2ABenchmark(unittest.TestCase):
    def _benchmark(self, impl_name, jit_fn, args, num_iters=20, warmup=5):
        print(f"     [{impl_name}] warmup x{warmup} (compile + steady)...", flush=True)
        for i in range(warmup):
            t0 = time.perf_counter()
            out = jit_fn(*args)
            jax.block_until_ready(out)
            dt = (time.perf_counter() - t0) * 1e3
            print(f"     [{impl_name}] warmup iter {i}: {dt:.1f}ms", flush=True)

        print(f"     [{impl_name}] measure x{num_iters}...", flush=True)
        latencies_ms = []
        for _ in range(num_iters):
            t0 = time.perf_counter()
            out = jit_fn(*args)
            jax.block_until_ready(out)
            latencies_ms.append((time.perf_counter() - t0) * 1e3)

        peak_hbm = _peak_hbm_bytes()
        return {
            "median_ms": statistics.median(latencies_ms),
            "p99_ms": float(np.percentile(latencies_ms, 99)),
            "mean_ms": statistics.mean(latencies_ms),
            "peak_hbm": peak_hbm,
        }

    def test_three_way_skewed(self) -> None:
        world_size = 4
        embedding_dim = 128
        dtype = np.float32

        # recv_capacity sized for v23 single-wave (>= max_local_ids + W*TILE,
        # multiple of 256). Same value used for WO#4 for fairness.
        configs = [
            dict(name="A_small", max_local_ids=256, recv_capacity=1024, v_local=256),
            dict(name="B_medium", max_local_ids=512, recv_capacity=1024, v_local=256),
            dict(name="C_large", max_local_ids=1024, recv_capacity=2048, v_local=512),
        ]

        devices = jax.devices()[:world_size]
        if len(devices) < world_size:
            self.skipTest(f"Need {world_size} devices, have {len(devices)}")
        mesh = jax.sharding.Mesh(np.array(devices), ("x",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("x"))

        all_rows = []
        for cfg in configs:
            name = cfg["name"]
            max_local_ids = cfg["max_local_ids"]
            recv_capacity = cfg["recv_capacity"]
            v_local = cfg["v_local"]

            print(
                f"\n=== Config {name}: W={world_size} max_local_ids={max_local_ids} "
                f"D={embedding_dim} recv_capacity={recv_capacity} v_local={v_local} ===",
                flush=True,
            )

            local_table_np = _build_local_table(
                world_size, v_local, embedding_dim, dtype, seed=0
            )
            local_ids_np, local_id_count_np = _make_skewed_inputs(
                world_size, v_local, max_local_ids, seed=2
            )
            local_ids = jax.device_put(jnp.array(local_ids_np), sharding)
            local_id_count = jax.device_put(jnp.array(local_id_count_np), sharding)
            local_table = jax.device_put(jnp.array(local_table_np), sharding)
            jax.block_until_ready((local_ids, local_id_count, local_table))
            args = (local_ids, local_id_count, local_table)
            print(f"  inputs ready; local_id_count={local_id_count_np.tolist()}", flush=True)

            v23_jit = jax.jit(functools.partial(
                jagged_a2a_embedding_lookup,
                mesh=mesh, axis_name="x",
                recv_capacity=recv_capacity, interpret=False,
            ))
            ra2a_jit = jax.jit(functools.partial(
                jagged_a2a_embedding_lookup_reference_jax,
                mesh=mesh, axis_name="x",
            ))
            wo4_jit = jax.jit(functools.partial(
                jagged_a2a_embedding_lookup_f32_alltc,
                mesh=mesh, axis_name="x",
                recv_capacity=recv_capacity, interpret=False,
            ))
            # Ring-shift uses single-slot recv layout: recv_capacity must
            # hold ONE wave (worst-case = traffic_matrix.max() <= max_local_ids
            # + TILE-1 in the all-to-one-peer skew). Round up to TILE.
            TILE = 128
            rs_recv_capacity = max_local_ids + world_size * TILE
            rs_recv_capacity = ((rs_recv_capacity + TILE - 1) // TILE) * TILE
            print(f"  ring-shift recv_capacity={rs_recv_capacity}", flush=True)
            rs_naive_jit = jax.jit(functools.partial(
                jagged_a2a_embedding_lookup_f32_alltc_ringshift,
                mesh=mesh, axis_name="x",
                recv_capacity=rs_recv_capacity, interpret=False,
                splitter="naive",
            ))
            rs_gp_jit = jax.jit(functools.partial(
                jagged_a2a_embedding_lookup_f32_alltc_ringshift,
                mesh=mesh, axis_name="x",
                recv_capacity=rs_recv_capacity, interpret=False,
                splitter="globalpad",
            ))

            # Compute reference output once (ra2a_fallback is the JAX
            # all-to-all reference). For correctness sanity check.
            try:
                ref_out = ra2a_jit(*args)
                jax.block_until_ready(ref_out)
                ref_np = np.asarray(jax.device_get(ref_out))
                print(f"  reference computed; shape={ref_np.shape}", flush=True)
            except Exception as e:
                print(f"  WARN: reference failed: {e}", flush=True)
                ref_np = None

            for impl_name, jit_fn in [
                ("v23_split", v23_jit),
                ("ra2a_fallback", ra2a_jit),
                ("fused_wo4", wo4_jit),
                ("fused_rs_naive", rs_naive_jit),
                ("fused_rs_globalpad", rs_gp_jit),
            ]:
                print(f"  -> {impl_name}", flush=True)
                # Single-shot correctness check (only for fused variants;
                # v23/ra2a known-good).
                if impl_name in ("fused_rs_naive", "fused_rs_globalpad") and ref_np is not None:
                    try:
                        my_out = jit_fn(*args)
                        jax.block_until_ready(my_out)
                        my_np = np.asarray(jax.device_get(my_out))
                        max_diff = float(np.max(np.abs(my_np - ref_np)))
                        match = bool(np.allclose(my_np, ref_np, atol=1e-4))
                        print(
                            f"     [{impl_name}] correctness: max_diff={max_diff:.6f} "
                            f"match={match}",
                            flush=True,
                        )
                        if not match:
                            print(
                                f"     [{impl_name}] WARNING: output mismatch! "
                                f"my[0,0,:4]={my_np.flat[:4].tolist()} "
                                f"ref[0,0,:4]={ref_np.flat[:4].tolist()}",
                                flush=True,
                            )
                    except Exception as e:
                        print(f"     [{impl_name}] correctness ERROR: {type(e).__name__}: {e}", flush=True)
                try:
                    r = self._benchmark(impl_name, jit_fn, args)
                except Exception as e:
                    print(f"     ERROR: {type(e).__name__}: {e}", flush=True)
                    all_rows.append((name, impl_name, None, None, None, None, str(e)[:80]))
                    continue
                hbm_str = f"{r['peak_hbm']/1e6:.1f}MB" if r["peak_hbm"] else "n/a"
                print(
                    f"     {impl_name}: median={r['median_ms']:.3f}ms "
                    f"p99={r['p99_ms']:.3f}ms mean={r['mean_ms']:.3f}ms "
                    f"peak_hbm={hbm_str}",
                    flush=True,
                )
                all_rows.append((
                    name, impl_name, r["median_ms"], r["p99_ms"],
                    r["mean_ms"], r["peak_hbm"], "ok",
                ))

        print("\n=== BENCHMARK RESULTS (W=4, f32, skewed) ===", flush=True)
        print(
            f"{'config':<10}{'impl':<18}{'median_ms':>12}{'p99_ms':>12}"
            f"{'mean_ms':>12}{'peak_MB':>12}  status",
            flush=True,
        )
        for cfg_name, impl_name, med, p99, mean, hbm, status in all_rows:
            if med is None:
                print(f"{cfg_name:<10}{impl_name:<18}{'-':>12}{'-':>12}{'-':>12}{'-':>12}  {status}", flush=True)
            else:
                hbm_mb = f"{hbm/1e6:.1f}" if hbm else "n/a"
                print(
                    f"{cfg_name:<10}{impl_name:<18}{med:>12.3f}{p99:>12.3f}"
                    f"{mean:>12.3f}{hbm_mb:>12}  {status}",
                    flush=True,
                )


if __name__ == "__main__":
    unittest.main()
