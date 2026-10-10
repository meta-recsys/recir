# Ringshift Sharded Embedding Lookup -- v7x bench

Real-TPU correctness and perf bench for `ringshift_embedding_lookup`, the
fused all-TC f32 ring-shift kernel that implements jagged all-to-all + sharded
embedding lookup as a single Pallas pipeline.

The kernel itself ships under
[`recir/ops/pallas/embeddings/ringshift.py`](../../ops/pallas/embeddings/ringshift.py)
with the public API:

```python
from recir.ops.pallas.embeddings import (
    ringshift_embedding_lookup,
    ringshift_embedding_lookup_reference_jax,  # ra2a baseline
)
```

## Layout

| Path                                        | Purpose                                                       |
| ------------------------------------------- | ------------------------------------------------------------- |
| `Dockerfile`                                | TPU image (JAX[tpu] + recir editable install).                |
| `build_image.sh`                            | OSS: build + push image to GCR. Run from the OSS recir checkout (`setup.py` at top). |
| `build_image_podman.sh`                     | fbcode: stage `recir/github/setup.py` + recir tree, build with podman + Meta corp creds, push to GCR. |
| `ringshift_fwd.py`                          | unittest test runner: forward correctness + perf (`RingshiftFwdCorrectnessTest`, `RingshiftFwdPerfTest`). |
| `ringshift_bwd.py`                          | unittest test runner: backward correctness (W=4 + W=32) + fwd+bwd perf (`RingshiftBwdW4Test`, `RingshiftBwdW32Test`, `RingshiftBwdPerfTest`). |
| `manifests/w{4,32,64,128}-correctness.yaml` | JobSet manifests for forward correctness at each W.           |
| `manifests/w{32,64,128}-perf.yaml`          | JobSet manifests for forward perf at each W.                  |
| `manifests/w4-bwd-suite.yaml`               | JobSet manifest for the full W=4 backward unittest suite (6 tests, single host). |
| `manifests/w32-bwd-correctness.yaml`        | JobSet manifest for backward correctness at W=32 (multi-host). |
| `manifests/w{32,64,128}-bwd-perf.yaml`      | JobSet manifests for fwd+bwd multi-iter perf at each W (multi-host). |
| `ringshift_ebc_parity_dump.py`              | TPU half of the torchrec EBC parity check: runs the fused pooled op and emits a self-describing artifact. See below. |
| `manifests/w4-ebc-parity-dump.yaml`         | JobSet manifest for that dump.                                |
| `test_embedding_bag_tpu_vs_torchrec_with_tpu_input.sh` | Runs the complete TPU artifact + CPU TorchRec comparison workflow. |
| `run_correctness.sh`                        | Builds/pushes the current TPU image, then runs every correctness suite or one named suite. |
| `run_perf.sh`                               | Builds/pushes the current TPU image, then runs every performance suite or one named suite. |

Run the complete TPU correctness matrix, or select one suite:

```sh
./run_correctness.sh
./run_correctness.sh ebc-w4
./run_correctness.sh --skip-image-build ebc-w4
```

The image build/push is enabled by default so a correctness run cannot silently
exercise stale code from the registry. Use `--skip-image-build` only for an
intentional rerun against the image that is already published.

Run the complete EBC performance matrix, or select one suite:

```sh
./run_perf.sh
./run_perf.sh ebc-w16
./run_perf.sh --skip-image-build ebc-w4
```

The performance runner follows the same rule: it builds and pushes the current
image once by default. Use `--skip-image-build` only for an intentional rerun
against the image that is already published. Each of `ebc-w4`, `ebc-w16`,
`ebc-w32`, `ebc-w64`, and `ebc-w128` runs
`torchrec.EmbeddingBagCollection` on the host at the shape parsed from the TPU
log and reports TorchRec/fused speedup. The XLA pooled arm is retained as an
additional same-TPU reference, not as the baseline.

The full flow is also available as copy-pasteable individual steps. This W=4
example can be changed to W16 or W32 by substituting the manifest and log name.
Run these commands from this directory:

```sh
# 1. Build and publish the TPU image.
./build_image_podman.sh

# 2. Submit the manifest, locate JAX process 0, tail it to RESULT, and archive
#    the rank-0 log as bench_logs/w4-ebc-perf-latest.log.
./run_microbench.sh manifests/w4-ebc-perf.yaml

# 3. Time the TorchRec EBC baseline at the exact shape recorded in that log.
cd ../../..
buck2 run fbcode//recir/examples/embedding:ringshift_ebc_torchrec_bench -- \
  --tpu-log recir/examples/embedding/bench_logs/w4-ebc-perf-latest.log 2>&1 | \
  tee -a recir/examples/embedding/bench_logs/w4-ebc-perf-latest.log
```

The final line has the existing benchmark result shape and names the baseline
explicitly:

```
[BENCH_EBC_TORCHREC] === RESULT ebc_vs_torchrec: baseline=torchrec.EmbeddingBagCollection torchrec=...ms(CPU) fused=...ms(TPU) speedup=...x xla=...ms(TPU,reference) shape=(world_size=4,dim=128,num_bags=1024,bag_len=6.399,distribution=skewed,pooling=sum) ===
```

`bag_len` is the average valid IDs per bag. The generated bags remain jagged;
`distribution=skewed` controls both the heavy-tailed bag lengths and the hot-peer
ID ownership pattern. Because TorchRec runs on the host CPU and the fused arm
runs on TPU, this is the requested end-to-end implementation baseline, not a
same-device kernel-only comparison.

## torchrec EBC parity (two stages)

The pooled kernel's headline reference is torchrec's `EmbeddingBagCollection`,
but the TPU image does not install torchrec, so the two cannot meet in one
process. Stage 1 runs on TPU and prints an artifact carrying both the inputs and
the fused results; stage 2 replays it against torchrec on CPU:

```sh
./test_embedding_bag_tpu_vs_torchrec_with_tpu_input.sh
```

The artifact travels as framed base64 in the job log (the pod FS is ephemeral
and the image has no `gsutil`), and it carries its own inputs, so the two halves
share no fixture and cannot drift apart. Stage 2 skips when the env var is
unset, and its log-parsing half is covered by CPU tests that always run.

This differs from `w4-ebc-correctness`, which diffs against the in-repo numpy
oracle: that one is faster to iterate on, this one is what deliverable 1.2 asks
for.

## Build the image

### OSS recir checkout

From the OSS recir checkout (where `setup.py` lives at the top):

```sh
cd ~/recir
./recir/examples/embedding/build_image.sh
```

### fbsource/fbcode

In fbcode there is no `setup.py` at the recir root (the only one lives at
`recir/github/setup.py`), so `build_image.sh` will not work. Use the
podman variant which stages a tmp build context with the right layout
and authenticates via Meta corp credentials:

```sh
~/fbsource/fbcode/recir/examples/embedding/build_image_podman.sh
```

Both scripts push
`us-central1-docker.pkg.dev/mrs-tpu-y317h/recml/ringshift_tpu:latest` to GCR.

## Submit a job

Each manifest invokes one `unittest` test method on a TPU-v7x topology sized
for that W.

### Forward

| Manifest                  | Topology  | Method                                                        |
| ------------------------- | --------- | ------------------------------------------------------------- |
| `w4-correctness.yaml`     | `2x2x1`   | `RingshiftFwdCorrectnessTest.test_w4_correctness`             |
| `w32-correctness.yaml`    | `4x4x8`   | `RingshiftFwdCorrectnessTest.test_w32_correctness`            |
| `w64-correctness.yaml`    | `4x4x8`   | `RingshiftFwdCorrectnessTest.test_w64_correctness`            |
| `w128-correctness.yaml`   | `4x4x8`   | `RingshiftFwdCorrectnessTest.test_w128_correctness`           |
| `w32-perf.yaml`           | `4x4x8`   | `RingshiftFwdPerfTest.test_w32_multi_iter`                    |
| `w64-perf.yaml`           | `4x4x8`   | `RingshiftFwdPerfTest.test_w64_multi_iter`                    |
| `w128-perf.yaml`          | `4x4x8`   | `RingshiftFwdPerfTest.test_w128_multi_iter`                   |

### Backward

The backward kernel ships as a `jax.custom_vjp` for `ringshift_embedding_lookup`,
so `jax.grad` / `jax.value_and_grad` on the fused op routes through the fused
backward Pallas kernel (mirrors the forward ringshift: barrier-first, split
DMA descriptors per direction, scatter-add into a per-wave HBM accumulator).

Correctness manifests compare `jax.grad(fused)` against `jax.grad(reference_jax)`
at `atol=rtol=1e-4`. Perf manifests time `value_and_grad` (fwd+bwd) and report
median + p99 + the estimated backward-only delta vs the forward-only baseline.

| Manifest                  | Topology  | Method                                                                                  |
| ------------------------- | --------- | --------------------------------------------------------------------------------------- |
| `w4-bwd-suite.yaml`       | `2x2x1`   | `RingshiftBwdW4Test` (full suite: 6 tests -- base, repeated hot row, wave0 self-peer, zero count, non-tile-aligned, jit value_and_grad). |
| `w32-bwd-correctness.yaml`| `4x4x8`   | `RingshiftBwdW32Test.test_w32_correctness`                                              |
| `w32-bwd-perf.yaml`       | `4x4x8`   | `RingshiftBwdPerfTest.test_w32_multi_iter_backward`                                     |
| `w64-bwd-perf.yaml`       | `4x4x8`   | `RingshiftBwdPerfTest.test_w64_multi_iter_backward`                                     |
| `w128-bwd-perf.yaml`      | `4x4x8`   | `RingshiftBwdPerfTest.test_w128_multi_iter_backward`                                    |

To submit (forward or backward, same command):

```sh
kubectl create -f recir/examples/embedding/manifests/w32-correctness.yaml
kubectl create -f recir/examples/embedding/manifests/w32-bwd-correctness.yaml
```

## View results

Tail logs from rank 0 (which is where all the `[CORRECTNESS_W{N}] ...` and
`[W={N}] ...` lines are printed):

```sh
kubectl logs -f \
  -l 'jobset.sigs.k8s.io/jobset-name=<jobset-name>,jobset.sigs.k8s.io/job-index=0' \
  --tail=-1
```

A passing correctness run prints:

```
[CORRECTNESS_W32] max_diff = 0.000000e+00
[CORRECTNESS_W32] PASS: max_diff 0.000000e+00 == 0 (bitwise-identical)
[CORRECTNESS_W32] === RESULT: max_diff=0.000000e+00 ===
```

A W=4 TPU-stage perf run prints per-iter timings followed by:

```
[BENCH_EBC_W4_D128_Q8192_skewed] === RESULT ebc: ... xla=...ms kernel=...ms speedup=...x ===
```

The local TorchRec stage then prints the `ebc_vs_torchrec` headline shown above.
Use its TorchRec/fused ratio as the headline speedup; XLA is an additional TPU
reference point.

## SparseCore vs TensorCore (forward)

`ringshift_bench.py` can also benchmark the **existing SparseCore solution**
(DLRMv3): the raw `@SparseDenseMatmulOp` custom call emitted by
`recir.embedding.sparsecore_custom_call.sc_matmul_fwd` (NOT jte's
`tpu_sparse_dense_matmul` primitive), driven in the non-pooling "sequence token
trick" mode (valency=1, one id per output row) and distributed across all cores
via `shard_map` so it triggers the SparseCore all-to-all. `jax_tpu_embedding`
(jte) is used only for host-side CSR preprocessing + feature/table specs. This is
measured against the two TensorCore arms at a matched
`(world_size, dim, max_local_ids, distribution)` shape:

```python
from recir.examples.embedding.ringshift_bench import main  # --backend {tc,sc,all}
```

| `--backend` | Arms measured                                                     |
| ----------- | ----------------------------------------------------------------- |
| `tc`        | `ra2a` (TC `lax.ragged_all_to_all`) + `fused` (TC ring-shift)     |
| `sc`        | `sc` (SparseCore `tpu_sparse_dense_matmul`)                       |
| `all`       | all three (default for `--mode fwd`)                              |

A single headline line carries the medians:

```
[BENCH_FWD_W128_D256_Q550816_uniform] === RESULT fwd: fused=…ms sc=…ms ===
```

Notes:
- The SC arm needs `jax-tpu-embedding` (jte). It is installed in the embedding
  image (see `Dockerfile`); rebuild via `build_image_podman.sh` before running
  the SC arm. jte is imported lazily, so `--backend tc` still runs without it.
- The SC arm meshes its table over **all** `jax.devices()`, so run the
  comparison with `--device-stride 1` (the SC manifest does) so both backends
  use the same device set.
- Host-side CSR preprocessing (`preprocess_sparse_dense_matmul_input`, CPU) is
  logged separately as `sc CSR preprocess: …ms (host)`; the `sc=…` median is the
  **device forward** only, for an apples-to-apples comparison with the ring-shift
  device forward.
- `bwd` mode is TensorCore-only here (SparseCore backward is out of scope).

Manifest:

| Manifest             | Topology | What it runs                                         |
| -------------------- | -------- | ---------------------------------------------------- |
| `rankfm-550k.yaml`   | `4x4x4`  | `--backend all` at rankfm E2E scale (fused/sc, bf16, traces) |

```sh
./recir/examples/embedding/run_microbench.sh manifests/rankfm-550k.yaml
```

To add the SC arm to any existing perf manifest, append `--backend all
--device-stride 1` to its `ringshift_bench` command (after rebuilding the image
with jte).
