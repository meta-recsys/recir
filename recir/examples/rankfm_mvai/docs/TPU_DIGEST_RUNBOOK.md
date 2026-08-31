# Runbook: the compiled (`in_graph`, TPU) leg in the parity comparison

One TPU job. The light leg and the checkpoint already exist and are reused, so
this is: arm, build, submit, compare.

## What already exists -- do not regenerate any of it

| | |
|---|---|
| checkpoint | `gs://cq-mrs-core-shared_mrs_tpu_y317h/parity/pi_256_100k.pth`, sha256 `3f30039f1f01daa0` |
| light leg log | `~/parity_evidence/20260815-1123-current-stack/7-light.log` |
| its settings | seq **256**, batch/rank **512**, hash cap **100000**, 4 ranks |
| CPU-eager log (third leg, optional) | same directory, `8-recir.log` |

The TPU leg must run with the SAME checkpoint, seq, batch and cap. Anything
else and there is nothing to compare against without re-running light.

In the pod the checkpoint path is **`/mnt/parity/parity/pi_256_100k.pth`** --
the gcsfuse CSI driver ignores `onlyDir` and mounts the whole bucket, so the
path carries the bucket-relative `parity/` prefix. `/mnt/parity` alone is a
directory and fails with `IsADirectoryError`.

## Step 1 -- arm the digest

Nothing to instrument. `model_family.py` already calls
`step_digest.arm(train_step_fn, padding_spec)` on both compile paths and
`watch_params(self)` on the compiled module, and both are inert until the env
var is set. Arming is one line in the manifest:

```
PARITY_DIGEST_KINDS=loss,grad,winit,weight
```

| kind | what it digests | emitted from |
|---|---|---|
| `loss` | the per-task loss vector, as `per_task_vector` | forward hook on the root module, inside the graph |
| `grad` | every parameter gradient, before the optimizer sees it | tensor hooks, inside the graph |
| `act` / `gradout` | module-boundary activations and their gradients | module hooks + the STU probe, inside the graph |
| `winit` / `weight` | parameters either side of the step | pre/post hooks on the compiled module, **outside** the graph |
| `update` | `weight - winit`, per parameter | same, outside the graph |

Two properties worth knowing before reading a log:

* **The parameter kinds are emitted outside the graph on purpose.** A host
  callback on a weight is a side-effecting HLO carrying a replicated sharding,
  which XLA's SPMD partitioner rejects (`spmd_partitioner.cc:6464`). Outside, it
  is also the eager legs' own code path on real torch tensors, so the reductions
  are bit-identical rather than merely equivalent.
* **`sharded=` is per tensor.** Dense params are replicated across the mesh, so
  a mesh reduction over them reports N times the true value; the `in_graph`
  tables are genuinely row-sharded. Covered by
  `test_replicated_is_not_summed_across_the_mesh`.

An unknown entry in `PARITY_DIGEST_KINDS` raises rather than being ignored -- a
typo is otherwise a silently missing digest.

## Step 2 -- the manifest

**There is no in_graph training manifest at the right size.** `64chip.yaml` is
in_graph but 4x4x4; `4chip.yaml` is 2x2x1 but `embedding_backend cpu`. Copy
`4chip.yaml` and change:

* `--embedding_backend in_graph`
* `--hash_size_cap 100000`
* `--local_batch_size 512`
* `--max_steps 1`
* `--init_state_dict_path /mnt/parity/parity/pi_256_100k.pth` and `--prod_faithful`
* env `HSTU_MAX_SEQ_LEN=256`, `PARITY_DIGEST_KINDS=loss,grad,winit,weight`
* the image tag (step 3)
* the parity checkpoint volume. Copy **three** blocks verbatim from
  `rankfm_mvai-parity-cpu-eager.yaml`, all of them:
  * the gcsfuse CSI annotation (`:53-57`) -- without it the volume never
    mounts and the pod hangs on the missing file rather than reporting
    anything useful;
  * the `parity-ckpt` volumeMount (`:132`);
  * the `parity-ckpt` volume definition (`:164`).

**Device count.** `4chip.yaml` is 2x2x1 = 4 chips = **8 jax devices**, and
light is 4 ranks. Run the TPU leg on **4 devices** or the row shards will not
line up and the embedding tables drop out of the comparison. Check what
`jax.device_count()` reports in the pod log and adjust the topology if it is 8.

## Step 3 -- build, push, submit

```bash
cd ~/fbsource/fbcode

IMAGE_TAG=$USER-compiled PODMAN_PULL_POLICY=never \
    bash recir/examples/rankfm_mvai/scripts/build_image_podman.sh
gcloud auth print-access-token | podman login \
    -u oauth2accesstoken --password-stdin https://us-central1-docker.pkg.dev
podman tag  localhost/rankfm_mvai_tpu:$USER-compiled \
    us-central1-docker.pkg.dev/mrs-tpu-y317h/recml/rankfm_mvai_tpu:$USER-compiled
podman push us-central1-docker.pkg.dev/mrs-tpu-y317h/recml/rankfm_mvai_tpu:$USER-compiled

kubectl -n ns-mrs-core-shared create -f <your-manifest>.yaml

# find the pod (newest last)
kubectl get pods -n ns-mrs-core-shared --sort-by=.metadata.creationTimestamp | tail -3

kubectl logs -n ns-mrs-core-shared <pod> -c rankfm > /tmp/tpu_compiled.log

# clean up, or the next create collides
kubectl delete jobset -n ns-mrs-core-shared <jobset-name>
```

`PODMAN_PULL_POLICY=never` reuses the base image in local storage; without it
the build tries to reach Docker Hub.

## Step 4 -- compare

```bash
buck2 run //hammer/v3/models/udd_fm_v3/parity:compare -- --only compare \
    --light-log ~/parity_evidence/20260815-1123-current-stack/7-light.log \
    --recir-log /tmp/tpu_compiled.log \
    --log-dir /tmp/compiled_compare
```

The stage refuses to run if the two logs report different `init ckpt sha256`.
If it does, the pod loaded the wrong checkpoint -- check the mount path.

## What to check, in order

1. **Are there any `[PARITY-DIGEST]` lines at all?** The digest prints from
   inside the compiled program via `jax.debug.callback` in a `jax.shard_map` body.
   That it survives recir's op-splice has only been verified on CPU --
   `jdebug.py:12-19` documents that effects at the top level of an `@op`
   lowering get dropped. Zero lines means the mechanism did not survive real
   XLA; stop and report that, nothing else matters.
2. **`init ckpt sha256` matches the light leg's `3f30039f1f01daa0`.**
3. **Read the ratio distribution, not the worst list.** Sorting by relative
   delta floats near-zero sums to the top, where 1e-5 against 3e-2 reads as
   "440x" and means nothing. Two readings of this comparison have been wrong
   for exactly that reason. Reference from the eager pair:

   ```
   weight  n=247  median 1.0
   grad    n=315  median 0.985-0.992
   act     n=226  median 1.0
   ```

4. **Do the embedding tables pair?** With 4 devices and batch 512 matched, a
   table's `shape` and `count` should agree between the legs. Verify rather
   than assume -- light shards row-wise through torchrec DMP, `in_graph`
   through FSDP, and they only pair if both put the same rows at the same
   index. If they differ, the tables are reported but not asserted, as they are
   in today's light-vs-CPU-eager comparison.
5. **Weight delta per embedding table.** A sparse-trained table moves ~50x
   further per step than a dense one. Independent check on the placement work:
   if this disagrees with
   `test_in_graph_places_tables_the_way_the_config_asks`, one of them is wrong.

Copy the pod log and the report out of `/tmp` when done -- it gets cleaned, and
that is how an earlier run's evidence was lost.

## Things that waste time if you do not know them

* **`image` and `push` cannot run from an agent session** -- the build pulls
  `python:3.12-slim` from Docker Hub and the registry hostname does not
  resolve. A human runs those two.
* **Only pod-side code needs an image rebuild.** Config reaches the pod through
  the artifact and ConfigMap.
* **`PARITY_DIGEST_KINDS` is read at TRACE time**, so it changes the graph. A
  gated and an ungated run are not bit-identical; do not compare across that
  boundary.
* **Do not raise seq to 512.** Parity mode forces the PYTORCH attention kernel
  so both legs match, and it materializes a dense `[B, H, N, N]` score matrix.
  At batch 512 the light leg OOMs mid-backward -- after the weight digests
  print but before any gradient is populated, so the run looks like it produced
  output and every gradient reads zero.
* **Delete the jobset when done.** A leftover one keeps restarting and the next
  `create` fails or, worse, you read the old pod's log.
