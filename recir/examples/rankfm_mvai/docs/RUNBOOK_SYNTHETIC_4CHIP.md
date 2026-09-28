# Runbook — 4 chips on synthetic data

From a fresh shell to TensorBoard curves and DCP checkpoints in GCS. No dump,
no ConfigMap, no prod access — the only input is the image. ~10 min build,
~4 min run.

For the prod-data and 128-chip runs, see `RUNBOOK_RUNNING_JOBS.md`.

## What it proves

That the pipeline runs: the graph traces, compiles and steps, metrics reach
TensorBoard, checkpoints are written and completed.

Not that the model is any good. Labels are random, so loss/NE/calibration are
noise. No part of the data path runs. Tables are capped at 10000.

## 1. One-time per devserver

```bash
feature install gcloud k8s_tools
sudo dnf install -y --disablerepo='*' --enablerepo='*google*' \
    google-cloud-cli-gke-gcloud-auth-plugin
HTTPS_PROXY=http://fwdproxy:8080 gcloud auth login
HTTPS_PROXY=http://fwdproxy:8080 \
    gcloud container clusters get-credentials tpu-prod-training-cluster \
    --region us-central1 --project mrs-tpu-y317h
kubectl config current-context   # mrs-tpu-y317h/us-central1/tpu-prod-training-cluster
```

The auth plugin was split out of kubectl in 2022 and Meta's gcloud disables
`components install`, so dnf is the only path.

## 2. Build and push

```bash
cd ~/local/fbsource/fbcode
SKIP_BARNFS=1 IMAGE_TAG=prod PODMAN_PULL_POLICY=never \
    bash recir/examples/rankfm_mvai/scripts/rebuild_push_image.sh
```

`SKIP_BARNFS=1` is correct here, not a shortcut: barnfs only encrypts DCP
writes when a job sets `TORCH_ENABLE_BARNFS=ON`, which no manifest here does.
It also skips the build's only `buck2` call, whose target has broken twice
under the `opt` config. See `README.md` for the alternative.

`PODMAN_PULL_POLICY=never` reuses the local base image; if it has been evicted
the build dies at STEP 1 with `python:3.12-slim: image not known`. Re-run once
with `missing`.

Expect `BarnFS: absent, as built (SKIP_BARNFS=1)` and a final
`pushed ...rankfm_mvai_tpu:prod`. A `DIRTY tree` warning listing uncommitted
files is expected while iterating — those files are in the image.

If buck2 itself fails with `daemon constraint mismatch`, run `buck2 kill`.

## 3. Submit

```bash
export HTTPS_PROXY=http://fwdproxy:8080
kubectl create -f recir/examples/rankfm_mvai/manifests/rankfm_mvai-synthetic-4chip.yaml \
    -n ns-mrs-core-shared
JS=<the name it prints>
```

`create`, not `apply` — the JobSet uses `generateName`.

## 4. Watch

```bash
POD=$(kubectl get pods -n ns-mrs-core-shared \
        -l jobset.sigs.k8s.io/jobset-name=$JS --no-headers | head -1 | awk '{print $1}')
kubectl logs -n ns-mrs-core-shared -f "$POD" -c rankfm
```

The container is `rankfm`. `2/2 Running` is trainer + gcsfuse sidecar, not two
ranks. The first few hundred lines are CUDA/triton/fbgemm load failures — noise
on TPU. Milestones:

```
=== jax devices ===                            8 devices, not 4
TB sync loop -> gs://.../tensorboard/<jobset> every 60s
[build] tag=prod revision=...
Trainer config: RankFMTrainerConfig(...)       gcs_data_path='' = synthetic
XLA HLO compiled (backend=xla_gspmd)           ~2 min in; the slow part
Checkpoint saved: .../step_0000010 (step=10)
Training complete. Process group destroyed.
```

## 5. Save the log, then verify

```bash
kubectl logs -n ns-mrs-core-shared "$POD" -c rankfm > /tmp/run-$JS.log
gcloud storage ls -r "gs://cq-mrs-core-shared_mrs_tpu_y317h/checkpoints/$JS/"
gcloud storage ls -l "gs://cq-mrs-core-shared_mrs_tpu_y317h/tensorboard/$JS/"
```

Save the log first — deleting the jobset takes the pod's logs with it.

Three checkpoints leave two `step_` directories; `--keep_last_n 2` prunes the
oldest. `latest.txt` names the newest *complete* one — a `step_` directory
absent from it is a partial write.

## 6. TensorBoard in a browser

```bash
setsid nohup bash recir/examples/rankfm_mvai/scripts/sync_run_artifacts.sh $JS \
    --interval 180 > /tmp/sync-$JS.log 2>&1 &
```

Then `https://www.internalfb.com/intern/tensorboard/?dir=manifold://mrs_recsys_gcp/tree/tensorboard/<jobset>`.
Safe to run mid-job. gcsfuse can lag, so "not ready yet" early on is accurate.

## 7. Clean up

```bash
kubectl delete jobset $JS -n ns-mrs-core-shared
```

## Failure modes

None of these names its own cause.

**`ValueError: ... disagree about STUNormPostProcConfig. Carried but not
declared here: ['use_rms_norm']`** — only reachable if you add
`--prod_faithful`. The exported artifact is pinned to the hammer revision it
was exported against. Re-export it and publish under a *new* ConfigMap name;
overwriting `rankfm-prod-cfg` can change the config under a running job -- six
manifests mount it. A `mvai-cloud run` is not exposed to this: it bakes the
artifact into its own image.

**`GeneratedBackwardFor_recir_split_2d_jagged_default returned an invalid
gradient`** — `--seq_ro_per_sample_budget` exceeded `--max_uih_seq_len`. Config
accepts it, backward does not. The shapes differ by the L1/L2 split ratio.

**`LLVM ERROR: Not enough space for TILESPMEM spill`** — model too large; seen
at 8 L1 / 10 L2 layers with batch 2048. TILESPMEM is per-core scalar memory
(128 KiB), not HBM, and this is a hard compiler abort. It happens *after*
`XLA HLO compiled`, so that line is not the all-clear.

**`RuntimeError: zeros: Dimension size must be non-negative` in
`pad_jagged_lengths`** — batch too small; seen at 32. Per-chip rows exceeded
the padding target. 2048 is known good.

The last three come from hand-picking shapes on the non-`--prod_faithful` path,
where nothing cross-checks them. The manifest's values satisfy all three at
once — big batch, small model, budget not above sequence length. Change one and
expect to re-derive the others.

**Pods `Pending`** — slice contention. `kubectl describe pod <pod> | grep -A3
'^Events:'`.
