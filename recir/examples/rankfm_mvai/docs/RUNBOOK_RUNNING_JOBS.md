# Runbook — running jobs on 4 chips and on 128 chips

Two ways to put this model on TPU. Picking the wrong one wastes either an hour
or a slice, so start here.

`RUNBOOK_128CHIP_PROD_CONFIG.md` is the deep reference for the 128-chip path:
exporting prod's config, publishing the ConfigMap, what the dump is, what to
expect line by line in the log. This document does not repeat it. What is here
is the 4-chip path, the choice between them, and the failure modes that cost
real time.

## Which one

| | 4 chips | 128 chips |
|---|---|---|
| manifest | `rankfm_mvai-verify-4chip.yaml` | `rankfm_mvai-128chip.yaml` |
| pods x chips | 1 x 4 | 32 x 4 |
| jax devices | 8 | **256** (v7x is 2 cores per chip) |
| tables | capped, `--hash_size_cap 100000` | **3e9, uncapped** |
| slice | usually free | contended; has waited 16+ min |
| turnaround | ~5 min | ~20 min plus queue |
| container name | **`rankfm`** | **`rankfm-mvai`** |

Use **4 chips** for anything structural: does the graph trace, does it compile,
does a code change break the step. Also for HBM scaling measurements, since
activations dominate there once the tables are capped.

Use **128 chips** for anything that depends on the real vocabulary or real
scale: the 3e9 table, HBM at production shard size, loss and NE, the LR curve.

**4 chips cannot tell you about**: int32 id overflow, the ring-shift at
production vocabulary, HBM with a real table shard, or any metric you intend to
compare against prod.

## 4 chips

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

`RECIR_DUMP_GRAPHS` is `0` by default. Set it to `full` to dump HLO, which
costs roughly 270k log lines and is the only way to answer questions about what
is actually in the compiled program.

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

## Artifacts

`scripts/sync_run_artifacts.sh` mirrors TensorBoard and the xprof trace into
Manifold so they can be opened in a browser. TB is re-synced on an interval so
the curves fill in while the job runs; the trace is picked up once, after the
profile window.

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
`rankfm_mvai-verify-4chip.yaml` pulls `linjianma-compiled`; the parity
manifests pull `linjianma-parity`. These are different images. Four consecutive
rounds of diagnostics once went into an image that the running job never
pulled, and every one of them presented as "the code did not execute". Verify
with:

```bash
podman run --rm --entrypoint python3 localhost/rankfm_mvai_tpu:linjianma-compiled \
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
