# Runbook — running jobs on 4, 64 and 128 chips

Three ways to put this model on TPU. Picking the wrong one wastes either an
hour or a slice, so start here.

The production runs are no longer manifests. They are the `rankfm-prod` launch config
in `launch_configs.py`, launched with `mvai-cloud run --launch-config`, which also builds and
pushes the image, exports prod's config into it, and submits — the five manual
steps this document used to open with are one command. `mvai-cloud` is the
same launcher used for the B200 runs; set it up once with
`cd ~/fbsource/fbcode/mvai_cloud && bash scripts/dev-setup.sh`.

`RUNBOOK_128CHIP_PROD_CONFIG.md` is the deep reference for the production path:
what the dump is, what every flag in the launch config is for, what to expect line by
line in the log. This document does not repeat it. What is here is the two
4-chip paths, the choice between them, and the failure modes that cost real
time.

## Which one

| | 4 chips, synthetic | 4 chips, prod data | 64 or 128 chips |
|---|---|---|---|
| how | `kubectl create -f` | `kubectl create -f` | `mvai-cloud run --launch-config rankfm-prod` |
| config | `rankfm_mvai-synthetic-4chip.yaml` | `rankfm_mvai-verify-4chip.yaml` | `launch_configs.py` |
| pods x chips | 1 x 4 | 1 x 4 | 16 x 4 or 32 x 4 |
| jax devices | 8 | 8 | **128** or **256** (v7x is 2 cores per chip) |
| data | random jagged batches | barnfs replay dump | barnfs replay dump |
| tables | capped, `--hash_size_cap 100000` | capped, `--hash_size_cap 100000` | **3e9, uncapped** |
| slice | usually free | usually free | contended; 4x4x8 more so than 4x4x4 |
| turnaround | ~5 min | ~5 min | ~20 min plus queue |
| container name | **`rankfm`** | **`rankfm`** | **`main`** |
| image tag | `prod` | `prod` | per-job, built by the launcher |
| checkpoints | yes, every 10 steps | yes, every 10 steps | yes, every 5000 steps |

Use **synthetic** when the dump is not the point: does the graph trace, does it
compile, does a code change break the step, are TensorBoard and checkpoints
produced. Also when the dump has expired and you do not want to rotate it yet.

Use **4 chips on prod data** for anything that depends on the data path: file
discovery, ID remapping, read order, real sequence shapes. Also for HBM scaling
measurements, since activations dominate there once the tables are capped.

Use **64 or 128 chips** for anything that depends on the real vocabulary or
real scale: the 3e9 table, HBM at production shard size, loss and NE, the LR
curve.

**Neither 4-chip run can tell you about**: int32 id overflow, the ring-shift at
production vocabulary, HBM with a real table shard, or any metric you intend to
compare against prod. **Synthetic additionally cannot tell you** anything about
loss or NE — its labels are random.

## 64 or 128 chips

```bash
~/fbsource/fbcode/mvai_cloud/.venv/bin/mvai-cloud run \
    --hardware tpu-v7 -j 16x8 --launch-config rankfm-prod
```

That exports prod's config, builds and pushes a per-job image, renders the
JobSet, validates it with gcpsubmitter and submits. `--dry-run` prints what it
would submit without doing any of it.

`-j 16x8` is 64 chips (a 4x4x4 slice), `-j 32x8` is 128 (4x4x8). Nothing else
changes between them: the launch config derives `--local_batch_size` from the
host count so the global batch -- the thing that has to match production --
stays 32768 either way, and `--max_steps` comes from `dump.py` as one pass over
the dump.

**Reach for 64 first.** It is the same run on a slice that is less often held
by someone else.

Then:

```bash
mvai-cloud logs <job-name> --hardware tpu-v7     # all ranks
mvai-cloud ls --hardware tpu-v7                  # your jobs
```

To stop one, delete the JobSet:

```bash
export HTTPS_PROXY=http://fwdproxy:8080
kubectl --context=mrs-tpu-y317h/us-central1/tpu-prod-training-cluster \
    -n ns-mrs-core-shared delete jobset <job-name>
```

`mvai-cloud cancel` does NOT work here: it sets `spec.suspend=true`, but these
JobSets are admitted by Kueue, which owns that field and sets it back --
observed recreating all 16 pods seconds after the "cancel" reported success.
Worth stopping a finished run promptly either way; the slice is contended.

### Overriding a flag

Anything after `--` replaces what the launch config sets:

```bash
mvai-cloud run --hardware tpu-v7 -j 16x8 --launch-config rankfm-prod \
    -- --max_steps 2000
```

Names are checked against the trainer's own fields before anything is built, so
a typo fails here rather than from inside the pod.

### The manifests, if you need them

`manifests/rankfm_mvai-64chip.yaml` and `-128chip.yaml` still run:

```bash
export HTTPS_PROXY=http://fwdproxy:8080
kubectl create -f recir/examples/rankfm_mvai/manifests/rankfm_mvai-64chip.yaml \
    -n ns-mrs-core-shared
```

**Prefer the launcher.** The manifests pull the shared `:prod` image tag and
mount the cluster-wide `rankfm-prod-cfg` ConfigMap -- both are last-writer-wins
across everyone using this cluster, and neither records which revision it came
from. A launcher run pins its own image and stamps `source_revision` into the
config artifact.

They stay because they are the reference the launcher is audited against. Two
guards, for anyone changing either side:

- `//mvai_cloud/tests:test_tpu_template_capabilities` -- the template still
  renders every capability the 64-chip manifest has.
- `scripts/audit_launch_config.py` -- diffs a rendered JobSet against the
  manifest leaf by leaf, plus a shell capability inventory. Run it before
  changing either side; it is currently at `0 capabilities LOST`.

### What a dead pod leaves behind

A diagnostics sidecar publishes these for **every** rank, continuously, under
`gs://cq-mrs-core-shared_mrs_tpu_y317h/logs/<job-name>/`:

| file | what it answers |
|---|---|
| `rank<N>.<host>.log` | the trainer's output, appended as it appears — survives SIGKILL |
| `rank<N>.<host>.err.log` | just the error-shaped lines, so triage does not start by scrolling 18k lines |
| `rank<N>.<host>.mem.log` | host memory, cgroup pressure, trainer RSS. An OOM kill leaves no Python traceback; this is what says it was one |
| `rank<N>.<host>.fs.log` | timed read/write against the durable mount. A wedged gcsfuse and a wedged trainer look identical from outside otherwise |
| `rank<N>.<host>.signal` | which signal killed it — the difference between "it died" and "it was preempted" |
| `rank<N>.<host>.exit` | the exit code, with `killed by signal N` decoded |

The trainer's own end-of-run copy is rank 0 only and runs on the normal exit
path, so it is absent from exactly the runs that need explaining.

### Changing something

Flags after `--` override the launch config's, and `--env NAME=VALUE` overrides its
environment. Both replace rather than stack, so the rendered command shows one
value:

```bash
mvai-cloud run --hardware tpu-v7 -j 16x8 --launch-config rankfm-prod \
    --env RECIR_DUMP_GRAPHS=1 -- --max_steps 200
```

If the change is permanent, put it in `launch_configs.py` instead. `tests/test_launch_configs.py`
pins the launch config against what the retired manifests carried, so it will tell you
what you changed.

**Reusing an image.** `--image <tag>` skips the build. That freezes the baked
prod config too, which is usually what you want when iterating on flags rather
than code.

### If the pods sit `Pending`

The slice is taken. Confirm before waiting:

```bash
kubectl describe pod <pod> -n ns-mrs-core-shared | grep -A3 '^Events:'
```

`NotTriggerScaleUp` or `FailedScheduling ... didn't match Pod's node affinity`
means the pool is full, not that anything is wrong with the job. Check who
holds it with `kubectl get jobsets -n ns-mrs-core-shared`, and consider
`-j 16x8` if you were asking for 4x4x8.

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

TB goes through a local emptyDir and is copied out periodically plus once at
exit; written straight to gcsfuse the growing events file is re-uploaded whole
on each flush and hits the per-object rate limit. Checkpoints have no such
problem — each save writes fresh files — so they go direct.

What does the copying differs by path: a launcher run has a
`tensorboard-uploader` sidecar on a 60s interval, and the manifests that set
`TB_SYNC_SECS` (64chip, 128chip, synthetic-4chip) run an in-pod loop.

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

## Artifacts in a browser

Both 4-chip runs and the 128-chip run write TensorBoard and (where enabled) the
xprof trace to GCS themselves — see the paths above. GCS is awkward to read
from a browser, so
`scripts/sync_run_artifacts.sh` mirrors them into Manifold. TB is re-synced on
an interval so the curves fill in while the job runs; the trace is picked up
once, after the profile window. Checkpoints are not mirrored — they stay in
GCS.

```bash
setsid nohup bash recir/examples/rankfm_mvai/scripts/sync_run_artifacts.sh $JS \
    --interval 180 > /tmp/sync-$JS.log 2>&1 < /dev/null &
```

`setsid`, not `nohup` alone: `nohup` blocks SIGHUP but leaves the process in
the caller's process group, so it dies with the shell that started it.

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
every one of them presented as "the code did not execute". This is the one
class of failure `mvai-cloud run` removes outright — it builds a per-job image
and submits that exact tag — but it comes straight back with `--image`, and
still applies to every manifest. Verify with:

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

**`SIGSEGV` with no traceback.** The 4-chip manifests and the `rankfm-prod`
launch config all set `PYTHONFAULTHANDLER=1`, which makes Python print its own stack
on a fatal signal. Without it the log carries only stripped native frames
reading `(unknown)` and pages of the signal handler faulting inside itself.

**A warning you should not scroll past.** `[config] DIAGNOSTIC: ...` means a
flag combination is not what it looks like — most importantly
`--hash_size_cap` under `--prod_faithful`, which clamps prod's 3e9-row table
and then rebuilds `feature_hash_sizes` from the capped sizes, so the run stays
internally consistent and nothing downstream can detect it. That is legitimate
for the 4-chip and parity runs and fatal for anything you intend to compare
against prod. `--l1_num_layers 3 --l2_num_layers 11` warns too: those are
prod's real depths, they look right, and under `--prod_faithful` they reach
nothing.

## Cleaning up

```bash
kubectl --context=mrs-tpu-y317h/us-central1/tpu-prod-training-cluster \
    -n ns-mrs-core-shared delete jobset <job-name>
```

Delete, not `mvai-cloud cancel` -- see "64 or 128 chips" above for why the
latter does not stop a Kueue-admitted JobSet.

Worth doing promptly at 64 and 128 chips — the slice is contended, and a
finished jobset still holds it.
