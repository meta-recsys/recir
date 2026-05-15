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

A perf run prints per-iter timings followed by:

```
[W=32] === RESULT: ra2a=[..., ..., ...] fused=[..., ..., ...]
```

Compare medians of `ra2a` vs `fused` for the headline speedup.
