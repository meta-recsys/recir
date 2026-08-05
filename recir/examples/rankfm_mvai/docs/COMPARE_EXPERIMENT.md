# rankfm_mvai TPU-vs-eager comparison experiment

Run a full A/B in one shot: launch two JobSets (the compiled **TPU** run and the
matched **CPU-eager** reference), **wait for both to finish**, then (1) print a
side-by-side per-task NE diff, and (2) copy each run's **real TensorBoard events
from GCS** into ONE Manifold logdir (both runs overlaid) and print a shareable
link.

> Why copy from GCS (not parse stdout): the in-pod TB export **does** write real
> scalar events to `gs://<bucket>/tensorboard/<jobset>/` — but they only flush to
> GCS when the run **completes** (gcsfuse doesn't reliably flush the open events
> file mid-run). So this harness waits for terminal state, then mirrors the real
> events GCS → Manifold (the proven `rankfm_mvai_tb_to_manifold` flow). stdout is
> used only for the text table.

**Consequence:** the runs must **complete** to get TB data, so the harness waits
for completion (bounded by `--max-steps` so it finishes in reasonable time) — it
does **not** time-box-and-kill mid-run.

## One-prompt usage (for Claude)

Paste this to Claude Code from anywhere in the repo:

> Launch the rankfm_mvai TPU-vs-eager comparison (single shard, 2000 steps):
> `python3 fbcode/recir/examples/rankfm_mvai/compare_experiment.py --max-steps 2000 --wait-minutes 90`
> then give me the TensorBoard link and summarize the biggest per-task deltas.

To validate a fix vs eager: `--yaml-a <fix>.yaml --label-a fixed --yaml-b
rankfm_mvai-eager-1shard.yaml --label-b eager`.

**Build + compare in one shot** (rebuild the image from your current checkout,
then run the comparison against it) — add `--build`:

> `python3 fbcode/recir/examples/rankfm_mvai/compare_experiment.py --build --max-steps 500 --wait-minutes 40`

`--build` runs `build_image_podman.sh` (build + push) first and **aborts the
whole comparison if the build fails**, so the runs always exercise freshly built
code. It needs `podman` + an authenticated `gcloud` (`gcloud auth login`). Pass
`--image-tag <tag>` to build/launch a unique tag instead of the shared `:latest`
(isolates concurrent experiments; the yamls' image ref is patched to match).

## What it does

0. (With `--build`) build + push the image via `build_image_podman.sh` before
   anything else; a nonzero exit aborts before any JobSet is launched.
1. (Optionally) patch both yamls' `--max_steps` so the runs **complete** in a
   bounded time (default 2000; `--max-steps 0` keeps each yaml's own value), and
   (with `--image-tag`) patch the `rankfm_mvai_tpu` image tag to match the build.
2. Launch both JobSets, **poll until both reach terminal state** (or
   `--wait-minutes` cap).
3. Pull each run's stdout → print a per-task NE table (sorted by largest gap) +
   summary (counts under 1.0 / 0.5 / 0.2, which side wins). Saved to
   `comparison.txt`.
4. For each run, `gcloud storage cp` its `tensorboard/<jobset>/` events from GCS
   into `<out-dir>/<tb-name>/<label>/`, then `manifold putr` the combined dir and
   print the link. **Real events** → full curves, both overlapping from the start.

## TensorBoard output

```
<out-dir>/<tb-name>/tpu/events.out.tfevents...     (copied from gs://<bucket>/tensorboard/<tpu-jobset>/)
<out-dir>/<tb-name>/eager/events.out.tfevents...   (copied from gs://<bucket>/tensorboard/<eager-jobset>/)
```

Uploaded to `manifold://<tb-manifold-path>/<tb-name>`; the tool prints:

```
TENSORBOARD (both runs overlaid, real events):
  https://www.internalfb.com/intern/tensorboard/?dir=manifold://mrs_recsys_gcp/tree/tensorboard/<tb-name>
```

View locally with `tensorboard --logdir <out-dir>/<tb-name>`.

## Prerequisites

- **Image built + pushed** for the configs under test — or pass `--build` to
  build + push it first (needs `podman` + `gcloud auth login`).
- **kubectl** on the TPU cluster, or pass `--context` / `--namespace` (defaults:
  current context, `ns-mrs-core-shared`).
- The run's **data source is reachable** and the TB export writes to
  `gs://<--tb-gcs-bucket>/tensorboard/<jobset>/` (default bucket
  `cq-mrs-core-shared_mrs_tpu_y317h`).

## Flags

| flag | default | meaning |
|---|---|---|
| `--build` | off | build + push the image (`build_image_podman.sh`) before launching; aborts on build failure |
| `--build-script` | `scripts/build_image_podman.sh` | path to the build+push script for `--build` |
| `--image-tag` | `latest` | tag to build/launch; unique tag isolates from shared `:latest` (yaml image ref patched to match) |
| `--yaml-a` / `--label-a` | `rankfm_mvai-1shard.yaml` / `tpu` | run A config + label |
| `--yaml-b` / `--label-b` | `rankfm_mvai-eager-1shard.yaml` / `eager` | run B config + label |
| `--max-steps` | `2000` | patch both yamls so runs complete (`0` = keep yaml value) |
| `--wait-minutes` | `90` | max minutes to wait for BOTH to reach terminal state |
| `--context` / `--namespace` | current ctx / `ns-mrs-core-shared` | kube target |
| `--out-dir` | `/tmp/recir_compare` | logs + `comparison.txt` + copied tfevents |
| `--no-cleanup` | off | keep JobSets (default deletes them after copying) |
| `--no-tensorboard` | off | skip the GCS→Manifold TB copy |
| `--tb-name` | `cmp-<A-jobset>` | Manifold subdir for the combined TB |
| `--tb-gcs-bucket` | `cq-mrs-core-shared_mrs_tpu_y317h` | GCS bucket pods write TB into |
| `--tb-manifold-path` | `mrs_recsys_gcp/tree/tensorboard` | Manifold base path |
| `--tb-ttl` | `2592000` (30d) | Manifold user-data TTL (seconds) |

## Reading the output

- **delta(a-b) < 0** = run A lower (better) NE; `> 0` = B better.
- SUMMARY counts how many tasks are `<1.0 / <0.5 / <0.2` per run (mean omitted —
  rare tasks can blow up to huge NE and swamp it).

## Caveats / gotchas

- **Runs must complete to get TB.** If a run is killed or the wait window elapses
  first, its events never flush to GCS and that side's TB is empty — the tool
  warns and still uploads whatever did flush. Use a smaller `--max-steps` or
  larger `--wait-minutes`.
- **Per-run step scales differ.** The TB x-axis is each run's own cumulative
  example counter, which advances faster for the larger-batch (TPU) run — so the
  two curves overlap but are **not** point-for-point aligned on the x-axis. They
  *do* both start near 0 and span the full run (no more "TPU starts after eager
  ends" — that was a stdout-truncation artifact of the old version).
- **Eager TB flush is worth confirming** on first use: all observed non-empty TB
  in GCS so far is from TPU runs; the eager path uses the same trainer/export, so
  it should flush on completion, but verify the eager side copied >0 bytes.
- For a fair compiled-vs-eager test the two yamls must match on everything except
  the compile axis (optimizer/LR/clip/embeddings/data/metrics).
