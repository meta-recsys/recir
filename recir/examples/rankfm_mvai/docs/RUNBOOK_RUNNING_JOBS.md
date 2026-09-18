# Runbook — running jobs on 4 chips and on 128 chips

Three ways to put this model on TPU. Picking the wrong one wastes either an
hour or a slice, so start here.

`RUNBOOK_128CHIP_PROD_CONFIG.md` is the deep reference for the 128-chip path:
exporting prod's config, publishing the ConfigMap, what the dump is, what to
expect line by line in the log. This document does not repeat it. What is here
is the two 4-chip paths, the choice between them, and the failure modes that
cost real time.

## Which one

| | 4 chips, synthetic | 4 chips, prod data | 128 chips |
|---|---|---|---|
| manifest | `rankfm_mvai-synthetic-4chip.yaml` | `rankfm_mvai-verify-4chip.yaml` | `rankfm_mvai-128chip.yaml` |
| pods x chips | 1 x 4 | 1 x 4 | 32 x 4 |
| jax devices | 8 | 8 | **256** (v7x is 2 cores per chip) |
| data | random jagged batches | barnfs replay dump | barnfs replay dump |
| tables | capped, `--hash_size_cap 100000` | capped, `--hash_size_cap 100000` | **3e9, uncapped** |
| slice | usually free | usually free | contended; has waited 16+ min |
| turnaround | ~5 min | ~5 min | ~20 min plus queue |
| container name | **`rankfm`** | **`rankfm`** | **`rankfm-mvai`** |
| image tag | `prod` | `prod` | `linjianma-pipeline` |
| checkpoints | yes, every 10 steps | yes, every 10 steps | no |

Use **synthetic** when the dump is not the point: does the graph trace, does it
compile, does a code change break the step, are TensorBoard and checkpoints
produced. Also when the dump has expired and you do not want to rotate it yet.

Use **4 chips on prod data** for anything that depends on the data path: file
discovery, ID remapping, read order, real sequence shapes. Also for HBM scaling
measurements, since activations dominate there once the tables are capped.

Use **128 chips** for anything that depends on the real vocabulary or real
scale: the 3e9 table, HBM at production shard size, loss and NE, the LR curve.

**Neither 4-chip run can tell you about**: int32 id overflow, the ring-shift at
production vocabulary, HBM with a real table shard, or any metric you intend to
compare against prod. **Synthetic additionally cannot tell you** anything about
loss or NE — its labels are random.

## 4 chips on prod data

```bash
export HTTPS_PROXY=http://fwdproxy:8080
kubectl create -f recir/examples/rankfm_mvai/manifests/rankfm_mvai-verify-4chip.yaml \
    -n ns-mrs-core-shared
```

Logs — note the container is `rankfm` here, not `rankfm-mvai`:

```bash
JS=<jobset-name>
POD=$(kubectl get pods -n ns-mrs-core-shared \
        -l jobset.sigs.k8s.io/jobset-name=$JS --no-headers | head -1 | awk '{print $1}')
kubectl logs -n ns-mrs-core-shared -f "$POD" -c rankfm
```

The manifest carries `--prod_faithful` with prod's config artifact, prod's
seq-len sampling and prod's dropout. Only the table size differs from the
128-chip run.

It runs the shared `prod` tag, not a personal one. Rebuild with
`IMAGE_TAG=prod bash .../scripts/rebuild_push_image.sh`, and do not push an
experimental build over it while a run is live — `imagePullPolicy: Always`
means the next pod restart picks it up.

`RECIR_DUMP_GRAPHS` is `0` by default. Set it to `full` to dump HLO, which
costs roughly 270k log lines and is the only way to answer questions about what
is actually in the compiled program.

### What it trains on

A 3-day Dataporter dump of the udd_fm training stream, the same one the
128-chip run uses, mounted by the `barnfs.csi.meta.com` CSI driver at
`/mnt/barnfs`. The path is a literal `sourcePath` in the manifest.
`--data_replicate_across_ranks` makes the single pod read the `rank_0` shard.

Shards are BarnFS-encrypted and decrypted at mount, so the logical files are
not visible under plain `gcloud storage ls`; the Manifold original is the
inspectable copy.

**Dumps expire, and that does not fail the pod** — it waits on the mount-ready
probe forever and looks hung. The path is a literal in six manifests, so
rotating it by hand is a dozen edits. Use the synthetic manifest below instead,
and rotate separately.

### TensorBoard and checkpoints

Both are produced by the job itself, under the jobset name:

```
gs://cq-mrs-core-shared_mrs_tpu_y317h/tensorboard/$JS/
gs://cq-mrs-core-shared_mrs_tpu_y317h/checkpoints/$JS/step_<NNNNNNN>/
gs://cq-mrs-core-shared_mrs_tpu_y317h/checkpoints/$JS/latest.txt
```

TB goes through a local emptyDir and is copied out every `TB_SYNC_SECS` plus
once at exit; written straight to gcsfuse the growing events file is
re-uploaded whole on each flush and hits the per-object rate limit. Checkpoints
have no such problem — each save writes fresh files — so they go direct.

`latest.txt` names the newest *complete* checkpoint; a `step_` directory absent
from it is a partial write. Resume by setting `MVAI_CHECKPOINT_LOAD_DIR`
(optionally `MVAI_CHECKPOINT_VERSION=step_0000020`; unset means latest).

Both paths key off `MVAI_JOB_NAME`, bound to the jobset name. Change one and
not the other and the TB copy finds nothing.

## 4 chips on synthetic data

`rankfm_mvai-synthetic-4chip.yaml`. No dump, no ConfigMap, no prod access —
same TB and checkpoint wiring as above, just no data source.

**Step by step in `RUNBOOK_SYNTHETIC_4CHIP.md`.** Short form:

```bash
export HTTPS_PROXY=http://fwdproxy:8080
kubectl create -f recir/examples/rankfm_mvai/manifests/rankfm_mvai-synthetic-4chip.yaml \
    -n ns-mrs-core-shared
```

Reach for it when the dump is not the point, or when it has expired — that does
not present as a data problem, the pod just sits in the mount-ready probe and
looks hung. Its labels are random, so the curves show that metrics flow, not
that they are good.

## 128 chips

Full procedure in `RUNBOOK_128CHIP_PROD_CONFIG.md`. The short form, once the
ConfigMap is published and the image is pushed:

```bash
export HTTPS_PROXY=http://fwdproxy:8080
kubectl create -f recir/examples/rankfm_mvai/manifests/rankfm_mvai-128chip.yaml \
    -n ns-mrs-core-shared
```

If the pods sit `Pending`, the 4x4x8 slice is taken. Confirm before waiting:

```bash
kubectl describe pod <pod> -n ns-mrs-core-shared | grep -A3 '^Events:'
```

`NotTriggerScaleUp` or `FailedScheduling ... didn't match Pod's node affinity`
means the pool is full, not that anything is wrong with the job. Check who
holds it with `kubectl get jobsets -n ns-mrs-core-shared`.

## Artifacts in a browser

Both 4-chip runs and the 128-chip run write TensorBoard and (where enabled) the
xprof trace to GCS themselves — see the paths above. GCS is awkward to read
from a browser, so
`scripts/sync_run_artifacts.sh` mirrors them into Manifold. TB is re-synced on
an interval so the curves fill in while the job runs; the trace is picked up
once, after the profile window. Checkpoints are not mirrored — they stay in
GCS.

```bash
nohup bash recir/examples/rankfm_mvai/scripts/sync_run_artifacts.sh $JS \
    --interval 180 > /tmp/sync-$JS.log 2>&1 &
```

TensorBoard lands at:

```
https://www.internalfb.com/intern/tensorboard/?dir=manifold://mrs_recsys_gcp/tree/tensorboard/$JS
```

The trace URL is printed into the sync log when it uploads.

Two things about the path it travels. The job writes TB to a local emptyDir and
copies to GCS on its own loop, and gcsfuse takes a while to make that visible
in the bucket — an hour is normal — so "not ready yet" early on is accurate
rather than broken. And the Manifold bucket requires `--userData` on upload;
without it the upload fails with a StorageException that is easy to mistake for
"no data yet" if stderr is discarded.

## The profile window

```
--profile_start_step 200 --profile_end_step 205
```

Steady state, deliberately. At step 3 the compile has just finished and nothing
is warm, so a window there measures a phase the run spends almost none of its
time in.

**The trace is capped at 1,000,000 events.** A 5-step window at this model's
size blows through that: one measured trace covered only the first 1.1 s of a
3.0 s span, with the remaining ops simply absent. It looks like the trace is
truncated mid-run, because it is. Narrow to a single step for a complete
timeline, or read `xplane.pb` through the TensorBoard Profile plugin, which is
aggregated and not subject to the cap.

## Failure modes worth recognising

**Pods `Pending` forever.** Slice contention, not your job. See above.

**Everything succeeds but nothing changed.** Check the image tag actually used.
`rankfm_mvai-verify-4chip.yaml` pulls `prod`; the parity manifests pull
`linjianma-parity`. These are different images. Four consecutive rounds of
diagnostics once went into an image that the running job never pulled, and
every one of them presented as "the code did not execute". Verify with:

```bash
podman run --rm --entrypoint python3 localhost/rankfm_mvai_tpu:prod \
    -c "import inspect, <module>; print('<marker>' in inspect.getsource(<module>))"
```

**A clean run that proves nothing.** Reaching the last step is not evidence of
correctness. A dropout key baked in as a constant and a learning rate frozen at
its initial value both produce runs that complete with no error, no warning,
and plausible loss. Both were found only by reading the HLO. If a fix concerns
what the compiled program computes, the check is the HLO, not the exit code.

**Two readings that look like a trend.** A single metric value cannot show that
something is changing. `lr/dense` reading slightly above its base looked like a
climbing warmup; two steps later it was bit-for-bit identical, which is what
actually exposed the bug. Always compare two points.

**`RESOURCE_EXHAUSTED` on HBM.** Expected above a certain batch size: batch
2048 does not fit, and the 128-chip ceiling measured out near 690.

**`SIGSEGV` with no traceback.** Both manifests now set
`PYTHONFAULTHANDLER=1`, which makes Python print its own stack on a fatal
signal. Without it the log carries only stripped native frames reading
`(unknown)` and pages of the signal handler faulting inside itself.

## Cleaning up

```bash
kubectl delete jobset $JS -n ns-mrs-core-shared
```

Worth doing promptly on 128 chips — the slice is contended, and a finished
jobset still holds it.
