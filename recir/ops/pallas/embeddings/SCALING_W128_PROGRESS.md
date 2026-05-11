# Scaling the fused all-TC ringshift kernel to W=128 — progress notes

## Goal

Make `jagged_a2a_embedding_lookup_f32_alltc_ringshift` work at W=128 on
`recir-v7-128` (the largest TPU v7x cluster available) toward the user-target
spec (W=512, max_local_ids=256K, D=512, v_local=1M). Single fused kernel
(per the user's "must fuse" constraint).

## Current state — TL;DR

**Single-call works at W=128.** v69h confirmed the kernel runs to completion
in ~12.5s wallclock (compile + first run) for D=128, v_local=8K,
max_local_ids=8K on 128 of 256 logical TC devices. No hang, `block_until_ready`
returns cleanly.

**Multi-iter does NOT work.** Subsequent invocations of the same `jit_fn`
hang in `BlockUntilReady` after iter 0. Hypothesized cause: cross-rank
semaphore / collective_id state not draining cleanly between same-program
invocations. Not yet resolved.

**Numerical correctness at W=128 not yet verified.** `v69p` test code is
ready (uses `devices[::2]` symmetric mesh + JAX-side `jnp.max(jnp.abs(...))`
reduction); cluster contention prevented the actual run. Image
`gcr.io/mrs-tpu-y317h/shenli-runner:v69p` and JobSet YAMLs (`d128`, `d256`,
`d512`) are staged on `linjian-client:~/`.

## What landed in this commit

### Kernel changes (jagged_a2a_lookup.py)

1. **`_make_ringshift_single_wave_kernel`** (new). Single-wave variant of
   the alltc ringshift kernel. Replaces the old `_make_alltc_fused_f32_ringshift_kernel`
   which had the wave loop INSIDE the kernel body. New version:
   - Wave index becomes a scalar SMEM input
   - SMEM refs are per-wave scalars (was `[W]`-shaped arrays)
   - Single barrier semaphore (was `[max_waves, 2]`)
   - Uses `pltpu.get_barrier_semaphore()` (required when `collective_id` is set)
   - Inner peer-signal loop kept Python-unrolled (HLO body is single-wave so it's small)

2. **`_jagged_a2a_embedding_lookup_local_f32_alltc_ringshift`** (rewritten).
   - Wave loop is now Python `for w_py in range(max_waves)` at JAX level
   - Each wave's pallas_call gets `compiler_params=pltpu.CompilerParams(collective_id=w_py)` —
     unique per call, matching v23's pattern
   - `input_output_aliases={8: 0}` makes `sorted_emb_padded` an in-place
     accumulator across waves
   - Added per-wave `psum + optimization_barrier` cross-rank epoch carry:
     ```python
     sorted_emb_padded_dep, _ = jax.lax.optimization_barrier(
         (sorted_emb_padded, sync_dep)
     )
     # ... pallas_call uses sorted_emb_padded_dep ...
     sync_dep = jax.lax.psum(sorted_emb_padded[0, 0], axis_name)
     ```
     Forces every rank to finish wave w-1 before any rank launches wave w.

3. **`_split_traffic_ringshift_globalpad`** docstring updated to clarify
   `recv_capacity >= max_local_ids + TILE` (no W factor — single-slot recv
   layout).

4. **Tightened `recv_capacity` formula** in benchmark files (was
   `max_local_ids + W*TILE`, now `max_local_ids + TILE`).

### New test files

- `tests/test_jagged_a2a_w128_bench.py` — bench infrastructure for W=128:
  `_make_inputs_on_device` / `_make_table_on_device` (avoid host-side
  materialization of W*v_local*D tensors), `test_w128_sweep`, several
  per-D `test_w128_*` variants. Uses `jax.experimental.multihost_utils.sync_global_devices`
  for cross-host timing barriers.

- `tests/test_chunked_v_local_correctness.py` — W=4 correctness test for
  the v_local-chunking wrapper (separate scaling axis, currently blocked
  by the same multi-pallas_call hang issue at >1 chunk).

- `tests/test_hbm_gather_spike.py` — minimal Pallas spike that confirmed
  Mosaic rejects HBM-direct row gather inside TC kernels (`Loads are only
  allowed on VMEM and SMEM references`). The result drove the design
  pivot to v23's split-pipeline pattern instead of trying to do all-TC
  HBM indirect gather.

## What works and what doesn't (W=128, recir-v7-128)

| variant | status |
|---|---|
| `ra2a_fallback` | ✓ 36 ms median (D=512, v_local=8K, max_local_ids=8K) — robust baseline |
| `v23_split` | ✗ compile time exceeds 15 min at W=128 (Python-unrolled `for peer in range(W)` × 3 kernels) |
| `fused_wo4` | ✗ SMEM 4 MB > 1 MB cap (per-wave [max_waves, W, W] SMEM matrices don't fit at W=128) |
| `fused_rs_globalpad` (single-call) | ✓ runs in ~12.5s (D=128, v_local=8K, max_local_ids=8K) |
| `fused_rs_globalpad` (multi-iter) | ✗ iter 1+ hang in BlockUntilReady |

## Iteration log (v68p → v69p)

| tag | change | result |
|---|---|---|
| v68p | Original Python loops + W=128 | compile >30 min |
| v68s | both `for w` and `for peer` → `lax.fori_loop` | compile 883 ms ✓; runtime hang on `sync_global_devices` after warmup |
| v68v | revert inner peer loop to Python | compile >23 min |
| v68y | drop post barrier | iter 0 ran (841 ms); iter 1+ hang |
| v69b2 | single-wave kernel + Python `for w_py` + `collective_id=w_py` (v23 pattern) | iter 0 ran (12 s); iter 1+ hang |
| v69d | + per-wave `psum + optimization_barrier` carry | iter 0 ran (12 s); iter 1+ hang |
| v69g | + final-output `psum` | iter 0 itself hung (regression) |
| v69h | revert v69g; single-invocation test (no multi-iter) | **kernel ran 12.5 s, `block_until_ready` returned cleanly ✓** |
| v69i/m/n/o | verification scaffolding for correctness check | failed: asymmetric mesh (devices[:128] of 256 logical) blocks `process_allgather` |
| v69p | `devices[::2]` symmetric mesh + `jnp.max(jnp.abs(out-ref))` JAX-side reduction | code+image ready; cluster contention blocked the run |

## Open questions / next steps

1. **Multi-iter hang root cause**: psum + optimization_barrier between waves
   doesn't fix it. Codex hypothesized inter-invocation epoch skew. Final-psum
   on output made things worse (iter 0 itself hung). Need a different angle.

2. **Compile-time of single-call (~12.5s)**: dominated by 128 distinct
   pallas_call compiles (one per wave with unique collective_id). JAX compile
   cache should help across sessions. Could potentially reduce by reusing
   collective_ids cyclically (e.g., `w_py % 4`) — but that risks the same
   channel-reuse issues we pivoted away from.

3. **Numerical correctness at W=128**: ready to verify with v69p code; needs
   a free cluster window.

4. **D=512 at W=128**: should fit VMEM (table_vmem 16 MB + zero_emb_vmem ~4 MB
   for max_local_ids=2K), not yet tested.

## Reference paste index

- P2312633161 — WO#4 (per-wave even-split) workaround
- P2312675956 — full project resume context as of 2026-05-07
- P2315119554 — what's blocking the fused kernel from scaling (TC HBM-load
  restriction + VMEM caps)
- Manifold notes: `manifold://k2b/tree/entries/tpu/` entries 00-11
  (08 = v68 series, 09 = Mosaic constraints, 10 = multi-host gotchas,
  11 = v_local chunking attempt)
