# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Feasibility spike: HBM-direct row gather inside a TC Pallas kernel.

Question we're trying to answer: does Mosaic compile and correctly execute
``chunk_vmem[pl.ds(i, 1), :] = table_ref[pl.ds(row_idx, 1), :]`` inside a
TC kernel, where ``table_ref`` lives in HBM and ``chunk_vmem`` lives in VMEM?

Three configurations are tested per spike, all single-rank, single-device:

  * BASELINE (control): the existing pattern — ``pltpu.sync_copy(table_ref,
    table_vmem)`` upfront, then ``chunk_vmem[i] = table_vmem[idx[i]]``. Known
    to compile + work; gives us a numerical reference for correctness.

  * HBM_GATHER_PL_DS: Mosaic try #1 — direct VMEM <- HBM via ``pl.ds`` indices.
    This is the pattern proposed in the kernel redesign plan.

  * HBM_GATHER_INDEX: Mosaic try #2 — fallback if pl.ds-on-HBM doesn't work;
    use the simpler ``table_ref[idx, :]`` integer-index form.

Each variant is run + compared against the baseline. We report:
  - compile/run success or the exact Mosaic error
  - max abs diff vs baseline
  - wallclock per call (rough, single-device)
"""

from __future__ import annotations

import time
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from recir.ops.pallas.embeddings.tests.test_jagged_a2a_lookup import _has_tpu


def _make_baseline_kernel(num_rows_to_gather: int, embedding_dim: int):
    """Control: copy full table HBM -> VMEM, then VMEM -> VMEM gather."""

    def kernel(table_ref, idx_ref, out_ref, table_vmem, scratch_vmem):
        pltpu.sync_copy(table_ref, table_vmem)

        def body(i, _):
            row = idx_ref[i]
            scratch_vmem[pl.ds(i, 1), :] = table_vmem[pl.ds(row, 1), :]
            return None

        lax.fori_loop(0, num_rows_to_gather, body, None)
        pltpu.sync_copy(scratch_vmem, out_ref)

    return kernel


def _make_hbm_gather_pl_ds_kernel(num_rows_to_gather: int, embedding_dim: int):
    """Mosaic try #1: pl.ds indexed HBM read straight into VMEM."""

    def kernel(table_ref, idx_ref, out_ref, scratch_vmem):
        def body(i, _):
            row = idx_ref[i]
            scratch_vmem[pl.ds(i, 1), :] = table_ref[pl.ds(row, 1), :]
            return None

        lax.fori_loop(0, num_rows_to_gather, body, None)
        pltpu.sync_copy(scratch_vmem, out_ref)

    return kernel


def _make_hbm_gather_index_kernel(num_rows_to_gather: int, embedding_dim: int):
    """Mosaic try #2: integer-index HBM read."""

    def kernel(table_ref, idx_ref, out_ref, scratch_vmem):
        def body(i, _):
            row = idx_ref[i]
            scratch_vmem[i, :] = table_ref[row, :]
            return None

        lax.fori_loop(0, num_rows_to_gather, body, None)
        pltpu.sync_copy(scratch_vmem, out_ref)

    return kernel


@unittest.skipUnless(_has_tpu(), "Requires real TPU")
class HBMGatherSpikeTest(unittest.TestCase):
    """Single-device feasibility spike for HBM-direct row gather."""

    def _run_one(
        self,
        *,
        kernel_factory,
        kernel_name: str,
        v: int,
        d: int,
        k: int,
        baseline_out: np.ndarray | None,
    ) -> dict:
        """Compile + run + check + return dict of results."""
        rng = np.random.RandomState(0)
        table_np = rng.randn(v, d).astype(np.float32)
        idx_np = rng.randint(0, v, size=(k,), dtype=np.int32)

        table = jnp.array(table_np)
        idx = jnp.array(idx_np)

        kernel = kernel_factory(num_rows_to_gather=k, embedding_dim=d)
        HBM = pltpu.MemorySpace.HBM
        SMEM = pltpu.MemorySpace.SMEM

        if kernel_name == "baseline":
            scratches = (
                pltpu.VMEM((v, d), jnp.float32),
                pltpu.VMEM((k, d), jnp.float32),
            )
            in_specs = [
                pl.BlockSpec(memory_space=HBM),
                pl.BlockSpec(memory_space=SMEM),
            ]
        else:
            scratches = (pltpu.VMEM((k, d), jnp.float32),)
            in_specs = [
                pl.BlockSpec(memory_space=HBM),
                pl.BlockSpec(memory_space=SMEM),
            ]

        try:
            jit_fn = jax.jit(
                lambda t, i: pl.pallas_call(
                    kernel,
                    out_shape=jax.ShapeDtypeStruct((k, d), jnp.float32),
                    grid_spec=pltpu.PrefetchScalarGridSpec(
                        num_scalar_prefetch=0,
                        in_specs=in_specs,
                        out_specs=pl.BlockSpec(memory_space=HBM),
                        scratch_shapes=scratches,
                    ),
                )(t, i)
            )

            t0 = time.perf_counter()
            out = jit_fn(table, idx)
            jax.block_until_ready(out)
            wall_compile_ms = (time.perf_counter() - t0) * 1e3

            t0 = time.perf_counter()
            for _ in range(5):
                out = jit_fn(table, idx)
                jax.block_until_ready(out)
            wall_steady_ms = (time.perf_counter() - t0) * 1e3 / 5

            out_np = np.asarray(jax.device_get(out))
            expected = table_np[idx_np]

            if baseline_out is not None:
                max_diff_vs_baseline = float(np.max(np.abs(out_np - baseline_out)))
            else:
                max_diff_vs_baseline = float(np.max(np.abs(out_np - expected)))

            ok_vs_expected = bool(np.allclose(out_np, expected, atol=1e-6))

            print(
                f"  [{kernel_name}] OK: compile+1run={wall_compile_ms:.1f}ms "
                f"steady={wall_steady_ms:.3f}ms  max_diff={max_diff_vs_baseline:.2e} "
                f"matches_numpy={ok_vs_expected}",
                flush=True,
            )
            return {
                "status": "ok",
                "compile_ms": wall_compile_ms,
                "steady_ms": wall_steady_ms,
                "max_diff": max_diff_vs_baseline,
                "matches_numpy": ok_vs_expected,
                "out_np": out_np,
            }

        except Exception as e:
            err_str = f"{type(e).__name__}: {str(e)[:400]}"
            print(f"  [{kernel_name}] FAILED: {err_str}", flush=True)
            return {"status": "error", "error": err_str}

    def test_hbm_gather_spike(self) -> None:
        # Small enough to compile fast on a single chip.
        v = 1024            # table rows
        d = 64              # embedding dim
        k = 32              # indices to gather

        print(
            f"\n=== HBM-direct row gather spike: v={v} d={d} k={k} ===",
            flush=True,
        )

        baseline = self._run_one(
            kernel_factory=_make_baseline_kernel,
            kernel_name="baseline",
            v=v, d=d, k=k,
            baseline_out=None,
        )
        baseline_out = baseline.get("out_np")

        hbm_pl_ds = self._run_one(
            kernel_factory=_make_hbm_gather_pl_ds_kernel,
            kernel_name="hbm_pl_ds",
            v=v, d=d, k=k,
            baseline_out=baseline_out,
        )

        hbm_index = self._run_one(
            kernel_factory=_make_hbm_gather_index_kernel,
            kernel_name="hbm_index",
            v=v, d=d, k=k,
            baseline_out=baseline_out,
        )

        print("\n=== SPIKE SUMMARY ===", flush=True)
        for name, r in [("baseline", baseline), ("hbm_pl_ds", hbm_pl_ds),
                         ("hbm_index", hbm_index)]:
            if r["status"] == "ok":
                print(
                    f"  {name:<14} OK  steady={r['steady_ms']:.3f}ms  "
                    f"max_diff={r['max_diff']:.2e}  numpy_match={r['matches_numpy']}",
                    flush=True,
                )
            else:
                print(f"  {name:<14} FAIL: {r['error']}", flush=True)


if __name__ == "__main__":
    unittest.main()
