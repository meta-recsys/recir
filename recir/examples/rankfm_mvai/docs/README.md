# RankFM-MVAI (xsurface V3) on GKE TPU v7x — via the mvai_cloud RecTrainer

End-to-end RankFM (xsurface V3 / `BlueReelsVDDTrainV3`) training **driven by
the `mvai_cloud` training framework** (`RecTrainer` + `RecModelFamily` +
`RecTrainPipeline` and its config dataclasses). This example is self-contained:
it owns the model/trainer config, the recir embedding backends (cpu / tpu /
in_graph), the data pipe, and the benchmark driver.

- **Model**: `BlueReelsVDDTrainV3` (hammer/v3).
- **mvai_cloud orchestration**: `RecTrainer` owns the train loop; the
  model/optimizer build and the train step are plugged in via a custom
  `RecModelFamily` and `RecTrainPipeline`.
- **recir @compile** XLA GSPMD dense path on TPU (JAX PJRT).
- **torchrec / recir embedding backend** ROW_WISE on CPU (gloo PG), selectable
  CPU (default) or TPU SparseCore.

## How it maps onto mvai_cloud

The GPU path of `mvai_cloud` is reused unchanged; only three seams are swapped
to reach TPU, all inside this package (no `mvai_cloud` edits):

| Seam | mvai_cloud (GPU) | rankfm_mvai (TPU) |
|---|---|---|
| Device init | `RecTrainer.__init__` cuda/NCCL | `SystemConfig(use_gpu=False)` CPU/gloo path + `jax.distributed.initialize()` in `main()` (recir lowers the dense graph to TPU via `jax.devices()` regardless of the torch device) |
| Model + optimizer build | `RecModelFamily.create_train_module_and_optimizer` (TorchRec DMP + FBGEMM fused opt) | `RankFMMvaiModelFamily` overrides it to set up the recir embedding backend + optimizer (incl. the in_graph dedicated embedding optimizer) |
| Train step | `SimpleSGDTrainPipeline.progress` (eager fwd/bwd/step) | `RecirXLATrainPipeline` runs the recir-compiled `train_step` (`recir.trace` → `compile_from_gm(XLAGSPMDBackend)` → `make_compiled_runner`), then `sync_device_state_to_torch` |

`RankFMMvaiModelFamily` subclasses hammer/v3's `BlueReelsVDDModelFamily` (which
already builds `BlueReelsVDDTrainV3` in `create_vanilla_train_module` and is a
`RecModelFamily`), so the hammer family and `RecTrainer` share one consistent
`RecModelFamily` base. The embedding backends, data pipe, and bespoke compile
helpers all live in this package.

## Entry point

The fixed training entrypoint is the python module:

```
python3 -u -m recir.examples.rankfm_mvai.train <flags>
```

The Dockerfile `CMD`, both YAMLs, and the BUCK `python_binary` all use exactly
this module.

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
HTTPS_PROXY=http://fwdproxy:8080 gcloud auth login
HTTPS_PROXY=http://fwdproxy:8080 \
    gcloud container clusters get-credentials tpu-prod-training-cluster \
    --region us-central1 --project mrs-tpu-y317h

# --- Every session ---
cd ~/local/fbsource/fbcode
source recir/examples/rankfm_mvai/scripts/helpers.sh   # defines rankfm_mvai_*, exports CTX

# --- Kick off training ---
rankfm_mvai_iterate    # build_image_podman.sh -> kubectl create -> tail rank 0
```

`rankfm_mvai_iterate` defaults to the 128-chip yaml; pass any other yaml as `$1`
(e.g. `rankfm_mvai_iterate recir/examples/rankfm_mvai/manifests/rankfm_mvai-4chip.yaml`).

## Build the image manually

### fbcode devserver, podman

```bash
cd ~/local/fbsource/fbcode
./recir/examples/rankfm_mvai/scripts/build_image_podman.sh
```

What the script does end-to-end:
1. Exports `HTTPS_PROXY=http://fwdproxy:8080` (override with `PROXY_URL=...`) so
   `gcloud`, `podman build`, and `podman push` all reach the public internet.
2. Stages `recir/`, `hammer/v3/` (manifest-traced), `mvai_cloud/minimal_viable_ai/`,
   and `generative_recommenders/` into a temp context.
3. Builds the image (no `apt`/`git`-clone steps — everything's pre-staged).
4. Runs an in-container import smoke check (includes
   `minimal_viable_ai.core.trainer.trainer.RecTrainer`).
5. Mints a fresh OAuth2 access token from `gcloud`, logs `podman` into Artifact
   Registry, and pushes `${IMAGE_PATH}` (`rankfm_mvai_tpu:latest`).

### Test the image locally before pushing to TPU

The Dockerfile's `CMD` runs a tiny model for a few steps by default
(`--max_steps 5 --local_batch_size 4 --num_candidates 4 --hash_size_cap 1000
--l1_num_layers 1 --l2_num_layers 1 --embedding_backend cpu`) — perfect for a
fast smoke on the devserver (JAX falls back to CPU; catches import, model-build,
and KJT-shape errors before the TPU round-trip):

```bash
podman run --rm -it rankfm_mvai_tpu:latest
```

### Run on a 4-chip TPU slice (smoke)

```bash
# Context name format varies by gcloud version. Check yours with
# `kubectl config get-contexts`; newer gcloud uses the slash form below.
kubectl create -f recir/examples/rankfm_mvai/manifests/rankfm_mvai-4chip.yaml \
  --context=mrs-tpu-y317h/us-central1/tpu-prod-training-cluster
```

### Run on a 128-chip TPU slice

```bash
kubectl create -f recir/examples/rankfm_mvai/manifests/rankfm_mvai-128chip.yaml \
  --context=mrs-tpu-y317h/us-central1/tpu-prod-training-cluster
```

Use `kubectl create` (not `apply`) — the JobSet YAML uses `generateName` so each
invocation spawns a new run with an auto-suffixed name (e.g.
`rankfm-mvai-4chip-abc12`). The 128-chip yaml pulls training batches from a
barnfs dataporter dump mounted via the `barnfs.csi.meta.com` CSI driver at
`/mnt/barnfs`, and writes profiles to
`gs://mrs-tpu-y317h/tpu-profiles/${JOBSET_NAME}`.

## Bash helpers (submit / monitor / iterate loop)

`helpers.sh` bundles the submit / status / logs / kill / iterate workflow into
a few short functions. Source it once per shell:

```bash
source recir/examples/rankfm_mvai/scripts/helpers.sh
```

Workflow:

```bash
rankfm_mvai_submit                # submit default 128-chip yaml; sets $JS
rankfm_mvai_status                # pods for current $JS, with node placement
rankfm_mvai_logs                  # tail rank 0
rankfm_mvai_dump_logs             # dump all 32 pod logs to /tmp/ranklogs/, summarize
rankfm_mvai_traceback 27          # full traceback from a specific rank
rankfm_mvai_kill                  # delete current $JS

rankfm_mvai_iterate               # build_image_podman.sh -> submit -> tail rank 0
```

`rankfm_mvai_submit` autodetects the target namespace from the yaml's
`namespace:` field (falls back to `$NS`, default `ns-mrs-core-shared`).
By default `CTX` is your active kubeconfig context (and a stale exported `CTX`
that isn't in the kubeconfig is auto-corrected to the active one). Override it
by exporting `CTX=<project>/<region>/<cluster>` (run `kubectl config
get-contexts` for the exact name) before sourcing. The script also exports `HTTPS_PROXY=http://fwdproxy:8080` so
`kubectl` works on Meta devservers; unset it if you're elsewhere.

Note: `rankfm_mvai_kill` only deletes the current `$JS`. If a previous JobSet is
still holding the 128-chip slice, your next submit will sit in `Pending` with
`FailedScheduling: Insufficient google.com/tpu` — clean it up explicitly:

```bash
kubectl get jobsets --context="$CTX"
kubectl delete jobset/<old-name> --context="$CTX"
```

## Benchmark driver (`run_tpu_bench.sh`)

`run_tpu_bench.sh` is a one-shot driver that builds the image, submits a JobSet,
tails rank 0, re-fetches the complete rank-0 log, then retrieves the xprof TPU
trace from GCS and pushes the rank-0 worker trace to Kernelhub (perfetto). It
uses the `rankfm_mvai_*` helpers under the hood, so the same `GKE_SKIP_BUILD`
knob applies (exposed as `RANKFM_SKIP_BUILD`).

```bash
cd ~/local/fbsource/fbcode
# defaults to rankfm_mvai-4chip-bench.yaml; pass any bench yaml as $1
./recir/examples/rankfm_mvai/scripts/run_tpu_bench.sh rankfm_mvai-64chip-bench.yaml
# reuse the current :latest image (recir source unchanged):
RANKFM_SKIP_BUILD=1 ./recir/examples/rankfm_mvai/scripts/run_tpu_bench.sh rankfm_mvai-64chip-bench.yaml
```

The trace is emitted by `RecirXLATrainPipeline.progress` over
`[--profile_start_step, --profile_end_step]` to `--profile_dir`
(`gs://.../tpu-profiles/${JOBSET_NAME}`). Logs land in
`~/rankfm_mvai_bench_logs/` (override with `RANKFM_BENCH_LOG_DIR`).

The `*-bench` yamls exercise the `in_graph` embedding backend (ring-shift
seq-embedding lookup) with its dedicated embedding optimizer; `cpu` / `tpu`
backends use the unified dense optimizer only.

## Relationship to the other recir examples

| | `recir/examples/rankfm_mvai` |
|---|---|
| Model | `BlueReelsVDDTrainV3` (hammer/v3) |
| Train loop | `mvai_cloud` `RecTrainer` |
| Model/opt build | `RankFMMvaiModelFamily.create_train_module_and_optimizer` |
| Train step | `RecirXLATrainPipeline.progress` (recir-compiled) |
| Embedding backend | `recir.examples.rankfm_mvai.embedding` (cpu / tpu / in_graph) |
| Data | `recir.examples.rankfm_mvai.data_pipe` (synthetic + barnfs replay) |
| Image | `rankfm_mvai_tpu` |
