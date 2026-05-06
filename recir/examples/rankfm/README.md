# RankFM (xsurface V3) on GKE TPU v7x

End-to-end RankFM (xsurface V3 / `BlueReelsVDDTrainV3`) training with:
- **torchrec DMP** ROW_WISE embedding sharding on CPU (gloo PG)
- **recir @compile** XLA GSPMD dense path on TPU (JAX PJRT)
- **Dual process group**: gloo (TCP) for embeddings + PJRT (ICI) for dense
- **PreprocessorOverride + SparseArchOverride** from `recir.tests.xla.xsurface_test_utils`
  - The heavy preprocessor and `ECSparseArch` are wrapped with dummy modules during tracing,
    then dispatched to the real modules at runtime via `replace_override_anchors`.

## Quick Start (fresh shell -> training running)

```bash
# --- One-time per devserver ---
feature install gcloud k8s_tools
# kubectl needs the GKE exec credential plugin (split out of kubectl in
# 2022). The Meta-managed gcloud disables `components install`, so the
# only sanctioned install path is dnf. Disable all non-google repos so
# flaky NVIDIA/fbpkg mirrors don't take dnf down with them:
sudo dnf install -y --disablerepo='*' --enablerepo='*google*' \
    google-cloud-cli-gke-gcloud-auth-plugin
# If you get 502/403 from fbpkg-http-proxy.fbinfra.net, the fbpkg proxy
# itself is down on this devserver — retry in a few minutes, or report
# at https://fb.workplace.com/groups/registryusers with #pingoncall fbpkg.
HTTPS_PROXY=http://fwdproxy:8080 gcloud auth login
HTTPS_PROXY=http://fwdproxy:8080 \
    gcloud container clusters get-credentials recir-v7-128 \
    --region us-central1 --project mrs-tpu-y317h

# --- Every session ---
cd ~/local/fbsource/fbcode
source recir/examples/rankfm/helpers.sh    # defines rankfm_*, exports CTX

# --- Kick off training ---
rankfm_iterate    # build_image_podman.sh -> kubectl create -> tail rank 0
```

`rankfm_iterate` defaults to the 128-chip yaml; pass any other yaml as `$1`
(e.g. `rankfm_iterate recir/examples/rankfm/rankfm-4chip.yaml`).

## Build the image manually

### fbcode devserver, podman

```bash
cd ~/local/fbsource/fbcode
./recir/examples/rankfm/build_image_podman.sh
```

What the script does end-to-end:
1. Exports `HTTPS_PROXY=http://fwdproxy:8080` (override with `PROXY_URL=...`) so
   `gcloud`, `podman build`, and `podman push` all reach the public internet.
2. Stages `recir/`, `hammer/v3/` (manifest-traced), `mvai_cloud/minimal_viable_ai/`,
   and `generative_recommenders/` into a temp context.
3. Builds the image (no `apt`/`git`-clone steps — everything's pre-staged).
4. Runs an in-container import smoke check.
5. Mints a fresh OAuth2 access token from `gcloud`, logs `podman` into Artifact
   Registry, and pushes `${IMAGE_PATH}`.

Override the proxy URL if your devserver uses a different one:

```bash
PROXY_URL=http://fwdproxy.facebook.com:8080 ./recir/examples/rankfm/build_image_podman.sh
```

### Build Docker image (any environment, plain docker)

```bash
RECIR_DIR=/path/to/recir HAMMER_DIR=/path/to/hammer \
    MVAI_DIR=/path/to/minimal_viable_ai \
    GENREC_DIR=/path/to/generative_recommenders \
    ./recir/examples/rankfm/build_image.sh
```

### Test the image locally before pushing to TPU

The Dockerfile's `CMD` runs `--preset small --max_steps 5` by default — perfect
for a fast smoke on the devserver (JAX falls back to CPU; catches import,
model-build, and KJT-shape errors before the TPU round-trip):

```bash
podman run --rm -it rankfm_tpu:latest
```

For a live-edit loop without rebuilding, mount fbsource over the COPYed dirs:

```bash
podman run --rm -it \
    -v ~/local/fbsource/fbcode/recir:/app/recir:Z \
    -v ~/local/fbsource/fbcode/hammer:/app/hammer:Z \
    -v ~/local/fbsource/fbcode/mvai_cloud/minimal_viable_ai:/app/minimal_viable_ai:Z \
    -v ~/local/fbsource/fbcode/generative_recommenders:/app/generative_recommenders:Z \
    rankfm_tpu:latest bash
```

### Run on a 4-chip TPU slice (smoke)

```bash
kubectl create -f recir/examples/rankfm/rankfm-4chip.yaml \
  --context=gke_mrs-tpu-y317h_us-central1_recir-v7-128
```

### Run on a 128-chip TPU slice

```bash
kubectl create -f recir/examples/rankfm/rankfm-128chip.yaml \
  --context=gke_mrs-tpu-y317h_us-central1_recir-v7-128
```

Use `kubectl create` (not `apply`) — the JobSet YAML uses `generateName` so each
invocation spawns a new run with an auto-suffixed name (e.g.
`rankfm-4chip-abc12`). `apply` requires a fixed `metadata.name` and will reject
`generateName`.

### Presets

| Preset | Hash cap | L1 layers | L2 layers | UIH len | Cands | Local batch |
|--------|----------|-----------|-----------|---------|-------|-------------|
| `--preset small` | 1K | 1 | 1 | 32 | 4 | 4 |
| `--preset medium` | 100K | 2 | 4 | 64 | 8 | 16 |
| (default) | 1M | 3 | 11 | 32 | 8 | 4 |

### Monitor

```bash
kubectl get pods -o wide -w --context=gke_mrs-tpu-y317h_us-central1_recir-v7-128
kubectl logs <pod-name> --context=gke_mrs-tpu-y317h_us-central1_recir-v7-128
```

### Bash helpers (submit / monitor / iterate loop)

`helpers.sh` bundles the submit / status / logs / kill / iterate workflow into
a few short functions. Source it once per shell:

```bash
source recir/examples/rankfm/helpers.sh
```

Workflow:

```bash
rankfm_submit                # submit default 128-chip yaml; sets $JS
rankfm_status                # pods for current $JS, with node placement
rankfm_logs                  # tail rank 0
rankfm_dump_logs             # dump all 32 pod logs to /tmp/ranklogs/, summarize
rankfm_traceback 27          # full traceback from a specific rank
rankfm_kill                  # delete current $JS

rankfm_iterate               # build_image_podman.sh -> submit -> tail rank 0
```

Override the cluster by exporting `CTX=gke_<project>_<region>_<cluster>` before
sourcing. The script also exports `HTTPS_PROXY=http://fwdproxy:8080` so
`kubectl` works on Meta devservers; unset it if you're elsewhere.

Note: `rankfm_kill` only deletes the current `$JS`. If a previous JobSet is
still holding the 128-chip slice, your next submit will sit in `Pending` with
`FailedScheduling: Insufficient google.com/tpu` — clean it up explicitly:

```bash
kubectl get jobsets --context="$CTX"
kubectl delete jobset/<old-name> --context="$CTX"
```

## Architecture

```
Pod (1 of N)
+- CPU: torchrec DMP EmbeddingCollection (ROW_WISE sharded via gloo)
|       (lives inside model.recblock.tokenizer_stack.mot_sparse_arch._sparse_arch._embedding_collection;
|        FUSED in-backward optimizer, lr=embedding_lr)
+- TPU: recir @compile(XLA GSPMD)
   +- BlueReelsVDDTrainV3
      +- recblock (PreprocessorStack -> TokenizerStack -> ComputeBlock -> Loss)
      |     PreprocessorStack    -> overridden by DummyPreprocessorStack at trace time
      |     ECSparseArch (token) -> overridden by DummySparseArch at trace time
      |     (HSTU layers traced normally through gen_rec kernels)
      +- Over-arch: DenseArch -> MoE -> SwishLayerNorm -> MMoE -> per-task MLPs -> TPM
            +- GraphModuleTransformation(enable_device_state=True)
```

## Dual Process Group

- **JAX PJRT**: auto-initialized by GKE env vars (`TPU_HOST_BOUNDS`, `TPU_PROCESS_ADDRESSES`)
- **torch.distributed gloo**: manually initialized using `JOB_COMPLETION_INDEX` as rank, first `TPU_PROCESS_ADDRESSES` entry as coordinator on port 29500
- Ranks differ between the two PGs (JAX = topology order, gloo = pod index order)
- The `_ECSparseArch` wrapper instance is left intact after DMP, so `SparseArchOverride.replacement` (which checks `isinstance(orig_submodule, ECSparseArch)`) still matches; only the inner `TorchRecEmbeddingCollection` becomes the sharded `ShardedEmbeddingCollection`.

## Why the overrides?

- `PreprocessorStack` and `ECSparseArch` perform jagged tensor manipulations
  and embedding lookups that are hard to trace through `make_fx` and lower
  to GSPMD: dynamic shapes, data-dependent ops, KJT pytree quirks.
- `xsurface_test_utils.py` defines `PreprocessorOverride` and
  `SparseArchOverride` (subclassing `recir.api.Override`) which:
  1. Swap the real submodule with a dummy returning fixed-shape buffers at
     `replacement()` time (so the trace sees stable shapes).
  2. Mark inputs with `anchor_input` and outputs with `anchor_output` so
     `replace_override_anchors` can wire up runtime dispatch back to the
     real modules in `transform()`.
  3. Inject a `_grad_anchor` parameter and `_batch_dep` term to keep
     autograd connectivity for `make_fx` backward tracing.

## How `train.py` works

1. `_patch_test_config_provider(cfg)` injects CLI knobs (hash cap, STU
   depth, sequence sampling toggle) into `xsurface_test_utils.TestConfigProvider`
   so `_build_full_model` picks them up.
2. `_build_full_model(device)` builds a `BlueReelsVDDTrainV3` with the
   real `RecBlock` (preprocessor + tokenizer + sparse arch + HSTU + post).
3. `_make_sample_input(recblock, device, batch_size)` produces a sample
   `FeatureBundle` with the right KJT keys.
4. `_create_random_train_batch(feature_config, ...)` produces a synthetic
   `MTMLROOTrainBatch` aligned to those KJT keys.
5. `_build_overrides(recblock, batch_sample, device, batch_size)` runs the
   preprocessor + tokenizer eagerly once to capture sparse-arch I/O via
   forward hooks, then returns the two `Override` instances.
6. `@compile(module_override=overrides, parallel_mode=GSPMDParallel(...),
   gm_transformation=GraphModuleTransformation(backend=XLA_GSPMD,
   enable_device_state=True))` compiles `train_step`.
7. The compiled step decomposes the batch into plain tensors at the
   boundary (KJT values + lengths + keys captured in closure) and
   reconstructs the `MTMLROOTrainBatch` inside via `dataclasses.replace`.

## Comparison with the dlrmv3 example

| | `recir/examples/dlrmv3` | `recir/examples/rankfm` |
|---|---|---|
| Model | `DlrmHSTU` (gen_rec OSS) | `BlueReelsVDDTrainV3` (hammer/v3) |
| Override | `DlrmV3SparseOverride` (one) | `PreprocessorOverride` + `SparseArchOverride` (two) |
| Sharding | torchrec DMP ROW_WISE on CPU + GSPMD on TPU | torchrec DMP ROW_WISE on CPU + GSPMD on TPU |
| Batch type | `(uih_values, uih_lengths, cand_values, cand_lengths)` | `MTMLROOTrainBatch` (rebuilt via `dataclasses.replace`) |
| Image base | OSS only | OSS + bundled hammer/v3 + minimal_viable_ai |

## Caveats

- `_create_random_train_batch` produces uniform-shape jagged tensors, not
  the production `RandomDataPipe` jagged distribution. To use the real
  data pipe, replace this call with a stream of `MTMLROOTrainBatch`
  fetched from your dataloader.
- Sequence-length sampling is disabled by default
  (`--keep_seq_len_sampling` to re-enable). With sampling on, the
  post-preprocessor jagged totals are non-deterministic, breaking the
  GSPMD shard-divisibility check on override outputs.
- DMP is wired up but only sharding the
  `_ECSparseArch._embedding_collection`. If hammer/v3 adds additional
  embedding tables in other paths (e.g., `additional_embeddings`,
  `unary_embedding_module`), update `setup_dmp` to include them in the
  planner walk.
- The `embedding_lr` flag drives the FUSED in-backward optimizer for
  sharded EC params; `dense_lr` drives the rest via SGD.
