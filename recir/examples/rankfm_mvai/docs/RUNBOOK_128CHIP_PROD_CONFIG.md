# Runbook — the production run, with production config

The `rankfm-prod` launch config in `launch_configs.py`, at 64 or 128 chips, on the czheo_test
dataporter dump, with the model built from prod's config instead of CLI layer
flags.

```bash
~/fbsource/fbcode/mvai_cloud/.venv/bin/mvai-cloud run \
    --hardware tpu-v7 -j 16x8 --launch-config rankfm-prod
```

That is the whole procedure. It builds a per-job image, exports prod's config
into it, renders the JobSet, validates it and submits. `-j 32x8` for 128 chips.

`manifests/rankfm_mvai-64chip.yaml` and `-128chip.yaml` still exist and still
run. They are the reference the launcher is checked against -- see "64 or 128
chips" in `RUNBOOK_RUNNING_JOBS.md` for the two guards and how to run the
equivalence audit. Prefer the launcher; the manifests are not going to be
edited further.

`docs/PROD_CONFIG_OVERRIDES.md` audits everything this run changes on top of
prod's config; read it once before trusting a number off this run.

Two things to know before starting:

- **The image cannot build prod's config itself.** `//minimal_viable_ai` is not
  staged into it. The config is exported on the devserver under buck and baked
  into the image by `scripts/build_image_podman.sh`;
  `prod_config_artifact.py` is the shim and `prod_config_spec.py` holds the
  path. This used to be a hand-published ConfigMap shared cluster-wide with
  the parity manifests, where whoever submitted last won; an artifact inside
  the image cannot be overwritten by someone else's submit.
- **The sequence length is not yours to choose.** Prod's xsurface config
  hardcodes `HSTU_MAX_SEQ_LEN = 3072 + 512 - 10 - 20 = 3554`
  (`model_v5_hstu_standalone_cint_datafm_fm_xsurface_config.py:111`). There is
  no environment variable on this path — `prod_config_shim` inherits the prod
  provider unchanged. Exporting with `HSTU_MAX_SEQ_LEN=...` set changes nothing.
  The build asserts the exported artifact reads 3554, so an export that picked
  up a parity or replay shim fails the image build rather than the comparison.
  The parity legs get a shorter length through `replay_config_shim`, which pins
  it as a class attribute — a different path.

---

## 0. What the dump actually is

**Which dump, and its shape, live in `dump.py`.** The numbers are deliberately
not repeated here: they were, and a rotation left this page describing the
previous dump. Read them there, and verify them against the dump itself with

```bash
python3 recir/examples/rankfm_mvai/scripts/set_dump.py --check
```

Manifold holds the original the light leg replays; the GCS path the manifests
mount is the BarnFS/Dataporter-packed copy of it. What matters when changing a
data flag:

| | |
|---|---|
| layout | **v0 bucketed**: `rank_R/batch_S/F.pt.zst`, 1000 files per `batch_S` |
| shards | `NUM_RANKS` in `dump.py` — must exceed the pod count, or pods repeat each other's data |
| stride | `BATCH_SIZE` in `dump.py`, baked into every dumped KJT |
| ids | already hashed at prod's table sizes, so no cap is needed and none is applied |

The GCS side shows only `BFS2_SHARD_MAP` and numbered content-addressed shard
dirs under `gcloud storage ls` — that is expected, the `barnfs.csi.meta.com`
driver reconstitutes the logical tree at `/mnt/barnfs`.

### Rotating to a new dump

One command does the whole rotation. Do not hand-edit `dump.py` — it is
generated, and the paths used to be copy-pasted across six manifests and two
modules, which is how stale copies got found by grep instead of by something
that fails.

```bash
python3 recir/examples/rankfm_mvai/scripts/set_dump.py \
    --manifold manifold://mrs_recsys_gcp/tree/czheo_test/<dump-name> \
    --gcs 'gs://mrs-tpu-y317h/czheo_test/<dump-name>/dataporter_2:...:v0/'
```

It reads the dump's own `metadata.json` for the shape and rewrites:

| | |
|---|---|
| `dump.py` | `DUMP_NAME`, the Manifold/GCS path parts, `NUM_RANKS`, `NUM_BATCHES`, `BATCH_SIZE` |
| every `manifests/*.yaml` | the BarnFS `sourcePath` and any `manifold://` reference |
| the epoch manifests | `--max_steps`, recomputed from `num_ranks × num_batches × batch_size` |
| the light leg's replay provider | `BATCH_SIZE` and `REPLAY_PATH`, so both legs replay the same dump |

Which manifests count as "epoch" ones — the runs whose `--max_steps` means one
pass over the dump rather than a deliberate short cap — is a hardcoded list in
`set_dump.py`. A bench or parity manifest is left alone on purpose; rewriting
its step count would destroy what it is for.

**Get the transfer id from the packing job, not from the old path.** It is
assigned per packing job, so it changes on every rotation even when the dump
name does not. The script refuses a `--manifold`/`--gcs` pair that names two
different dump directories, which is the one mistake that would otherwise have
the two legs quietly reading different data — the failure this whole comparison
already lost time to once.

Then `--check` and commit the three things it touched.

---

## 1. What the launch config passes, and what it deliberately does not

`launch_configs.py` carries the flag set. Each absent flag is load-bearing, and
`config._warn_on_incoherent_flags` only *warns* about them — nothing refuses.

| | |
|---|---|
| prod config | baked at `/app/prod_config/prod_cfg.pt`, `PROD_CONFIG_ARTIFACT` set by the launcher |
| config flags | `--prod_faithful`, and **no** `--l1_num_layers` / `--l2_num_layers` / `--hash_size_cap` |
| data flags | `--gcs_data_path /mnt/barnfs`, and **no** `--data_file_template` / `--data_replicate_across_ranks` / `--num_candidates` |
| sizes | `--local_batch_size` and `--max_steps` derived from the host count and `dump.py` |

**The layer flags.** `--prod_faithful` decides the stack depths, and a leftover
`--l1_num_layers` does nothing at all — it only reaches the config through
`_apply_bench_shape_overrides`, which prod-faithful skips. Prod is L1=3, L2=11
(`..._xsurface_config.py:1088-1089`) — so the old `--l1_num_layers 3
--l2_num_layers 11` *looked* like it matched prod, and did not: the standalone
udd_fm config only defines 10 L2 layers, and `_trim_stu_stack` warns and
silently runs 10 rather than raising. **L2 = 11 vs 10 is the tell that the prod
path took**, not "3 and 11". Passing either now produces a `[config]` warning
saying it was ignored.

**`--hash_size_cap`.** Absent, so the tables keep prod's real `num_embeddings`.
That is the point of the run, and also the main thing that can sink it — prod's
largest table is ~3B rows, and the row-wise shards land at tens of GB per host.
If a pod dies during model construction, before any step, suspect this first.

With no cap, `remap_batch_ids` is effectively the identity: the dump was hashed
at prod's sizes, so every id is already in range and the modulo does nothing
(`data_pipe.py:410`). Putting a cap back would change the modulus, i.e. change
which row each id lands on — a different model on the same data, not just a
smaller one. It also rewrites `feature_hash_sizes` from the post-cap sizes, so
the whole pipeline agrees with the capped model and nothing can detect the
divergence at runtime. The remap applies `remainder()` unconditionally to every
key that has a table, so an out-of-range id gets folded, never raised: no
`IndexError` will warn you. This is why it warns loudly now.

**`--data_file_template`.** Absent, and it must stay absent.
`model_family.py:564` resolves it to exactly ONE file:

```python
explicit_files = [trainer_cfg.data_file_template.format(i=1)]
```

which `replay_batch_iter` then recycles through its rotating row window. A run
with that flag is a throughput benchmark, not training, and its NE/CTR curve is
not comparable to anything. Its stated justification — avoiding a 58-minute `ls
/mnt/barnfs` — does not apply: `data_pipe._discover_files` reads `metadata.json`
and computes the v0 filenames arithmetically, never listing the directory.

**`--data_replicate_across_ranks`.** Absent, so the pods shard instead of all
reading the same file. Pod K takes `global_idx = K + step*hosts` across the
dump's 400 shards, giving one distinct file per pod per step.
`PaddingRegionTransform` pads every loaded batch to fixed post-pad shapes, so
the divergent per-rank jagged lengths that sharding introduces do not break the
ROW_WISE all-to-all.

**The batch sizes.** `--local_batch_size` is derived as
`global_batch // hosts`, so 2048 at 64 chips and 1024 at 128, and the global
batch is 32768 either way. It is always a whole multiple of the dump's own
`batch_size` (512), so each step consumes whole files and `select_rows_slice`
is a no-op; `launch_configs.py` refuses a host count where that would not hold. An
earlier value of 32 trained on a rotating 1/16 subsample of each file.

**`--num_candidates`.** Absent because under `--prod_faithful` it is inert: the
NRO budget comes from prod's `hstu_max_output_len` (=20) via
`_resolve_nro_per_sample` (`xsurface_helpers.py:94-99`). Passing `32` would be
read by a future reader as if it applied, and now warns.

### Running the HSTU attention on Helion

**On** in the `rankfm-prod` launch config, which sets
`RANKFM_MVAI_ATTENTION_BACKEND=helion`.
The retired `-128chip.yaml` did not, so the 128-chip path ran the stock
PyTorch attention and the 64-chip path did not; unifying them onto one launch config
picked Helion for both. To go back to the stock path for a comparison:

```bash
mvai-cloud run --hardware tpu-v7 -j 32x8 --launch-config rankfm-prod \
    --env RANKFM_MVAI_ATTENTION_BACKEND=baseline
```

`rankfm_model_utils.resolve_attention_backend()` reads it and
`install_attention_backend` rebinds the attention leaf to the recir Helion
adapter. The STU then traces attention as a single
`recir::helion_ragged_attention_fwd` node that torchax lowers onto the Helion
Pallas → Mosaic kernels, instead of the dense `[B, H, N, N]` score matrix the
stock path materializes. That difference is the point: it is what makes larger
per-chip batches fit at all.

**The kernel is plain causal attention and refuses anything else.**
`helion_hstu_attention_op.py` raises `ValueError` if `max_attn_len`,
`full_attn_size` (i.e. `global_attn_window`), `contextual_seq_len`,
`num_softmax_heads` or `dropout_pr` is non-zero. It fails loudly at
construction rather than silently computing something else, so a config
mismatch cannot reach a number you might believe.

Nothing extra is needed to satisfy that on this path: `prod_config_shim` clears
semi-local attention and dropout at the config layer for both legs, so the
artifact this run bakes in already carries zeros. If the assertion does fire,
the artifact is older than that provider change — rebuild the image, which
re-exports it, rather than reaching for a flag to suppress it.

`manifests/rankfm_mvai-64chip-bench.yaml` also runs this way.

## 2. Build

`mvai-cloud run` does this for you. What it runs underneath is
`scripts/build_image_podman.sh`, which has two knobs worth knowing:

- `BARNFS_WHEEL=<path>` — building the wheel from source fails on master (a
  `_Nonnull` clang extension compiled by GCC under
  `opt-linux-x86_64-clang21-no-san`). Pass a prebuilt one in the environment.
  `SKIP_BARNFS=1` omits it, which is right for `rankfm-prod`: it runs
  unencrypted, so nothing imports barnfs. Only a launch config that sets
  `barnfs_encryption=True` needs the wheel, and an image built without it fails
  to import rather than silently writing plaintext.
- `PROD_CONFIG_ARTIFACT_SRC=<path>` — reuse an earlier export instead of paying
  the `buck2 run` again. `SKIP_PROD_CONFIG=1` omits it entirely, which is right
  for synthetic and bench images and wrong for this one.

The build asserts the baked artifact loads and reads `hstu_max_seq_len=3554`,
so a bad export fails here rather than 20 minutes into a run.

## 3. TensorBoard while it runs

A `tensorboard-uploader` sidecar copies the events file to GCS every 60
seconds, not just at exit. That matters here: at `--max_steps 320725` an
end-of-run copy would mean nothing appeared until the run terminated, so a job
still going — or one that got preempted — would show an empty dashboard. It is
a separate container from the trainer, so a slow or rate-limited gcsfuse write
stalls the copy and never the training step. (The retired manifests did the
same thing with an in-pod loop on `TB_SYNC_SECS`.)

The job writes to:

```
gs://cq-mrs-core-shared_mrs_tpu_y317h/tensorboard/$JOBSET_NAME
```

To read it in a browser, mirror it into Manifold and leave the mirror running:

```bash
setsid nohup bash recir/examples/rankfm_mvai/scripts/sync_run_artifacts.sh $JS \
    --interval 180 > /tmp/sync-$JS.log 2>&1 < /dev/null &
```

`setsid`, not `nohup` alone: `nohup` blocks SIGHUP but leaves the process in
the caller's process group, so it dies with the shell that started it.

```
https://www.internalfb.com/intern/tensorboard/?dir=manifold://mrs_recsys_gcp/tree/tensorboard/$JS
```

Empty for the first while is normal rather than broken — gcsfuse takes its time
making the pod's write visible in the bucket. `RUNBOOK_RUNNING_JOBS.md` has the
rest, including the `--userData` failure that looks exactly like "no data yet".

What to read first is section 6: CTR before NE, always.

## 4. The trace

The manifest profiles steps 900–906 and writes the capture to

```
gs://cq-mrs-core-shared_mrs_tpu_y317h/tpu-profiles/$JOBSET_NAME
```

Steady state on purpose — a window near step 3 measures compile-warm behaviour
the run spends almost none of its time in. The same `sync_run_artifacts.sh`
picks the trace up once it appears (only at `profile_end_step`, so not for the
first several hundred steps) and uploads it to kernelhub; the Perfetto link is
printed into the sync log.

**A 6-step window will not give you 6 steps in Perfetto.** The
`.trace.json.gz` is written under a hard event cap and this model emits ~17k
device ops per step, so the cap is spent inside the first step — and the
eviction is indiscriminate, dropping the structural `Steps` / `XLA Modules`
lines along with it. What opens looks like a one-step capture rather than a
truncated one.

The full data is in the `.xplane.pb` sitting next to it. Convert that instead:

```bash
python3 recir/examples/rankfm_mvai/scripts/xplane_to_chrome.py <worker>.xplane.pb
```

Or read `xplane.pb` through the TensorBoard Profile plugin, which is aggregated
and not subject to the cap.

## 5. Checkpoint and resume

The launch config saves every 5000 steps and keeps the last two:

```
--save_every_n_steps 5000 --keep_last_n 2 --async_checkpoint
```

**`rankfm-prod` writes plaintext checkpoints to `checkpoints/`**, the same
prefix the manifests use, because `barnfs_encryption` and `checkpoint_save_dir`
are launch config fields and it sets them that way. A resume reads the
configured prefix, so a run that turned encryption on would not find the
existing checkpoints. `sync_run_artifacts.sh` probes both prefixes. For an
encrypted run, decrypting to read locally is the "Transmit Checkpoint back to
Meta" flow in the TPU RecSys Training Runbook.

**The save directory has to be on `/mnt/gcs-out`.** That is the durable GCS CSI
bucket. `/mnt/gcs/mvai_cloud` — where TensorBoard writes — is a pod-local
emptyDir, so a checkpoint written there is gone on exactly the restart it
exists for.

The two directory variables are not symmetric, and it is not a typo:
`CheckpointManager` appends `$MVAI_JOB_NAME` to the **save** dir itself
(`checkpoint.py:37-43`) but uses the **load** dir verbatim. So `LOAD_DIR`
carries the jobset name and `SAVE_DIR` does not. Use `mvai-cloud run
--resume-from <job-name>` rather than setting either by hand; it also replaces
the `SEED_CHECKPOINT_JOBSET` variable the retired manifests carried.

`--async_checkpoint` overlaps the upload with training; the blocking part is
only the staging.

### What is in it

Everything needed to make step N+1 identical to an uninterrupted run, which
took more than the model weights:

| | |
|---|---|
| model params and buffers | the `_sharded_tables` as `Shard(0)` DTensors, one row block per host; `_dp_tables` and dense params replicated |
| `__recir_optimizer__` | the dense optimizer |
| `__recir_embedding_optimizer__` | the embedding optimizer — the tables' accumulators, row-sharded the same way |
| `__recir_warmup_iter__` | the LR warmup cursor, which lives in no `state_dict` |
| `__recir_shard_order__` | the save-time `mesh slot -> torch rank` permutation |
| `__recir_sl_sampler__` | the sequence-length sampler's load-balance counters |
| `data_pipe` | the replay cursor, counted in batches whose step *completed* |

The row blocks are numbered by FSDP ordinal, not by torch rank. That matters
because the rank-to-slice assignment can change between the save and the
resume, and numbering by rank would reassemble the tables permuted — with
every shape matching and no error anywhere. The restore logs both orders:

```
[checkpoint] shard order at save [...], at restore [...]
```

They do not have to be equal. If they are equal every time you look, you have
not yet exercised the case the ordinal numbering exists for.

### What is NOT in it

**The rolling metric window.** torchrec registers window states with
`persistent=False`, so `window_ne` and friends restart their window empty and
converge over the next `--metric_window_size` rows. Expect a visible step in
the windowed metrics at the join; the lifetime metrics are continuous. Read
CTR and the lifetime NE across a resume, not the windowed ones.

### Resuming

Point `MVAI_CHECKPOINT_LOAD_DIR` at a specific run's checkpoint directory. A
resume that finds nothing there starts at step 0 rather than failing, which
combined with a restart policy is how you get two curves from zero on one
dashboard — check the log says `Resuming training from step N` before
believing a restarted job resumed.

Wait for `.metadata` before treating a checkpoint as complete. DCP writes it
last, so it is a valid completion marker even under `--async_checkpoint`.

The data cursor counts batches whose training step **finished**, not batches
fetched. With `--data_prefetch_buffer 32` and `--pipeline_preprocessor` those
differ by the whole lookahead, and the older encoding saved the fetch position
— so a checkpoint from before that change is read as legacy and corrected by
the lookahead depth. A resume replays the in-flight batches rather than
skipping them.

### Reading what a run has saved

`sync_run_artifacts.sh` pushes an index — not the checkpoints, which are
hundreds of GB — to Manifold alongside the TensorBoard mirror:

```bash
python3 recir/examples/rankfm_mvai/scripts/checkpoint_index.py <gcs-prefix>
```

It emits `index.json` and a rendered `index.md`: step, wall-clock time, size,
and which one `latest.txt` points at.

### If the checkpoint phase looks slow

Two log lines attribute it. `[stage] state_dict ...` splits the blocking phase
into materialize / flush / sanitize / wrap / opt, and `[flush] full
device->host sync: ...` breaks the flush down further, listing the eight
slowest entries by FQN. A uniform slowdown across all entries is a host or bus
effect; one FQN standing out is a shape, dtype or sharding change.

## 6. Does the data behave like production

The reference is the production MAST run
[fire-linjianma-train-hammerv3-20260804-524295](https://www.internalfb.com/mlhub/pipelines/runs/mast/fire-linjianma-train-hammerv3-20260804-524295?version=0&env=PRODUCTION).

Compare in this order — the first two are cheap and catch the failures that
matter:

| Signal | Why it comes first |
|---|---|
| **CTR** | A data-identity check. It is a property of the labels alone, so it should match production closely regardless of model differences. If CTR is off, the two runs are not seeing the same distribution and nothing downstream is worth reading. |
| **Example counts / batch shapes** | Confirms global batch (32 × 512 = 16384) and the NRO budget (prod's `hstu_max_output_len` = 20) line up with what production consumed. |
| **NE** | Only meaningful once CTR agrees. Expect a gap: this is 20 steps against a full production run, and the embedding tables start from scratch. |

Deliberately NOT the same: this run trains for 20 steps from random init on 128
chips. NE will not match production and is not supposed to. CTR is the one that
should.

One residual unknown: the preproc truncation the dump was written at. Prod's
config runs at 3554 and `mot_preprocessor.truncate_ro_features` only ever
shortens, so a dump written shorter cannot fail — it just means the cap never
binds and the real sequences are shorter than prod's. That would show up as a
low `sum(num_candidates)` fill and an optimistic step time, not as an error.
If the numbers look too good, check the per-step `[Replay]` line before
believing them.

## 7. What the log should say

Beyond the generic startup, this run has four checkpoints:

1. `Pod N waiting for /mnt/barnfs/metadata.json and /mnt/barnfs/rank_N/batch_0/0.pt.zst as mount-ready probe...`
   — each pod waits for its own shard. The CSI driver publishes before the
   container starts, so this normally clears immediately. If it hangs, the
   mount failed or the dump's layout is not what section 0 says; exec in and
   look rather than waiting.
2. `[Replay] Dump (v0 bucketed): N dump ranks x M batches.` — N and M must match `dump.py`. Proof that `_discover_files` read the manifest and that
   the pods are sharding rather than replicating.
3. Config load from the artifact. **Confirm the L2 stack has 11 layers** and
   that no `_trim_stu_stack` warning appears — that is the evidence the prod
   path actually took.
4. Per step,
   `[Replay] step K (pos K): sum(num_candidates)=..., NRO budget=10240 (fill=...%, max nc[i]=.../20)`.
   `max nc[i]` above 20 means the dump carries more candidates per row than
   prod's budget and the padding is clipping.

## Iterating

`mvai-cloud run --launch-config rankfm-prod` again. The config export is part of the
image build, so it picks up a changed prod config automatically — there is no
longer a separate publish step to forget, and no shared ConfigMap for a parity
run to overwrite underneath you.

Iterating on flags rather than code? `--image <tag>` reuses a pushed image and
skips the build entirely, and flags after `--` override the launch config's.

## Verifying the prod path really took

Worth doing once. `--prod_faithful` failing open would leave you with the
default config and a run that looks fine:

- The log's L2 layer count should be 11, not 10.
- No `[config] DIAGNOSTIC` warning in the log. One means a flag is quietly not
  doing what it reads as; see `RUNBOOK_RUNNING_JOBS.md`.
- `tests/test_prod_faithful_config.py` checks the built model's tables against
  the artifact — count, names, dims, placement. Run it locally with
  `PROD_CONFIG_ARTIFACT` pointed at your export to confirm the artifact is sound
  before spending cluster time.
- `tests/test_launch_configs.py` pins the launch config's flag set against what the retired
  manifests carried.

```bash
PROD_CONFIG_ARTIFACT=/tmp/prod_cfg.pt \
  buck2 test //recir/examples/rankfm_mvai/tests:test_prod_faithful_config
```

## Note on uploads

Do not `gcloud storage cp` anything into the bucket from your devserver — the
MRS GCP audit flags it (T284199285). Nothing in this runbook needs it: the
config is baked into the image and the data is already in the dump. The trainer
writing profiles and TensorBoard out from inside the cluster is a different
path and is not affected.
