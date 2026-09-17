# Runbook — 128-chip real-data run with production config

Runs `manifests/rankfm_mvai-128chip.yaml` (32 pods × 4 TPU) on the czheo_test
dataporter dump, with the model built from prod's config instead of CLI layer
flags.

Goal of this configuration: make the CTR/NE this run reports comparable against
the production MAST run, so the numbers mean something. Everything below that
looks like fussiness — the batch size, the absent flags, the seq len — is in
service of that.

`docs/RUNBOOK_RUNNING_JOBS.md` covers the mechanics shared with every other job
(submitting, tailing logs, artifacts, cleanup, the common failure modes). This
document is only what is specific to the prod-config run.
`docs/PROD_CONFIG_OVERRIDES.md` audits everything this run changes on top of
prod's config; read it once before trusting a number off this run.

Two things to know before starting:

- **The image cannot build prod's config itself.** `//minimal_viable_ai` is not
  staged into it. The config is exported on the devserver into a small artifact
  and mounted as a ConfigMap. `prod_config_artifact.py` is the shim.
- **The sequence length is not yours to choose.** Prod's xsurface config
  hardcodes `HSTU_MAX_SEQ_LEN = 3072 + 512 - 10 - 20 = 3554`
  (`model_v5_hstu_standalone_cint_datafm_fm_xsurface_config.py:111`). There is
  no environment variable on this path — `prod_config_shim` inherits the prod
  provider unchanged. Exporting with `HSTU_MAX_SEQ_LEN=...` set changes nothing.
  The parity legs get a shorter length through `replay_config_shim`, which pins
  it as a class attribute — a different path.

---

## 0. What the dump actually is

Verified from the Manifold original that the light leg replays,
`manifold://mrs_recsys_gcp/tree/czheo_test/udd_fm_3d_20260722`:

```json
{"version": 0, "num_ranks": 480, "num_batches": 41942, "batch_size": 512}
```

The GCS path the manifest mounts is the BarnFS/Dataporter-packed copy of that
dump. Read this before changing any data flag:

| | |
|---|---|
| layout | **v0 bucketed**: `rank_R/batch_S/F.pt.zst`, 1000 files per `batch_S` |
| shards | 480, so all 32 pods get a distinct one with room to spare |
| stride | 512 rows per file — this is baked into every dumped KJT |
| ids | already hashed at prod's table sizes, so no cap is needed and none is applied |

Re-checkable from a devserver at any time; no pod required:

```bash
manifold get mrs_recsys_gcp/tree/czheo_test/udd_fm_3d_20260722/metadata.json /tmp/md.json && cat /tmp/md.json
manifold ls mrs_recsys_gcp/tree/czheo_test/udd_fm_3d_20260722/rank_0/batch_0/ | head
```

The GCS side shows only `BFS2_SHARD_MAP` and numbered content-addressed shard
dirs under `gcloud storage ls` — that is expected, the `barnfs.csi.meta.com`
driver reconstitutes the logical tree at `/mnt/barnfs`.

---

## 1. Export prod's config

```bash
cd ~/fbsource/fbcode
buck2 run //recir/examples/rankfm_mvai:export_prod_config -- --out /tmp/prod_cfg.pt
```

Needs the full prod config chain, so devserver only. The exporter prints what
it built:

```
wrote cloud config artifact to /tmp/prod_cfg.pt
    (N embedding tables, hstu_max_seq_len=3554)
```

**`hstu_max_seq_len` must read 3554.** Anything else means you picked up
`--parity` or a replay shim, and the artifact describes a different model than
production. Nothing downstream checks this.

## 2. Publish it to the cluster

ConfigMap, not the bucket: it is tens of KB, far under the 1MB limit, and this
avoids a GCS round trip on every iteration.

```bash
kubectl create configmap rankfm-prod-cfg -n ns-mrs-core-shared \
    --from-file=prod_cfg.pt=/tmp/prod_cfg.pt \
    --dry-run=client -o yaml | kubectl apply -f -
```

`create --dry-run | apply` rather than plain `create` so a rerun replaces the
old one instead of failing on "already exists". Verify the byte count moved to
match your local file:

```bash
ls -la /tmp/prod_cfg.pt
kubectl get configmap rankfm-prod-cfg -n ns-mrs-core-shared \
    -o jsonpath='{.binaryData.prod_cfg\.pt}' | wc -c
```

**This ConfigMap is shared with the parity manifests, which want a *different*
artifact.** Whoever submits last wins, so re-export and republish before each
run rather than trusting what is there.

## 3. Check the manifest

Nothing to edit except the image tag — `rankfm_mvai-128chip.yaml` carries the
prod-config and data wiring:

| | |
|---|---|
| `PROD_CONFIG_ARTIFACT` | `/mnt/prodcfg/prod_cfg.pt` |
| volume `prod-cfg` | `configMap: rankfm-prod-cfg` |
| mount | `/mnt/prodcfg`, read-only |
| config flags | `--prod_faithful`, and **no** `--l1_num_layers` / `--l2_num_layers` / `--hash_size_cap` |
| data flags | `--gcs_data_path /mnt/barnfs`, `--local_batch_size 512`, and **no** `--data_file_template` / `--data_replicate_across_ranks` / `--num_candidates` |

Each absent flag is load-bearing.

**The layer flags.** `--prod_faithful` decides the stack depths, and a leftover
`--l1_num_layers` either fights it or quietly wins. Prod is L1=3, L2=11
(`..._xsurface_config.py:1088-1089`) — so the old `--l1_num_layers 3
--l2_num_layers 11` *looked* like it matched prod, and did not: the standalone
udd_fm config only defines 10 L2 layers, and `_trim_stu_stack` warns and
silently runs 10 rather than raising. **L2 = 11 vs 10 is the tell that the prod
path took**, not "3 and 11".

**`--hash_size_cap`.** Absent, so the tables keep prod's real `num_embeddings`.
That is the point of the run, and also the main thing that can sink it — prod's
largest table is ~3B rows, and the row-wise shards land at tens of GB per host
across the 32 pods. If a pod dies during model construction, before any step,
suspect this first.

With no cap, `remap_batch_ids` is effectively the identity: the dump was hashed
at prod's sizes, so every id is already in range and the modulo does nothing
(`data_pipe.py:410`). Putting a cap back would change the modulus, i.e. change
which row each id lands on — a different model on the same data, not just a
smaller one. The remap applies `remainder()` unconditionally to every key that
has a table, so an out-of-range id gets folded, never raised: no `IndexError`
will warn you.

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
reading the same file. Pod K takes `global_idx = K + step*32` across the dump's
480 shards, giving 32 distinct files per step. `PaddingRegionTransform` pads
every loaded batch to fixed post-pad shapes, so the divergent per-rank jagged
lengths that sharding introduces do not break the ROW_WISE all-to-all.

**`--local_batch_size 512`.** The dump's own `batch_size`, so each step consumes
a whole dumped file and `select_rows_slice` is a no-op. The previous value of 32
trained on a rotating 1/16 subsample of each file. Global batch is therefore
32 × 512 = 16384, which GSPMD shards to 128 rows per chip — comfortably under
prod's own 512 rows per GPU rank.

**`--num_candidates`.** Absent because under `--prod_faithful` it is inert: the
NRO budget comes from prod's `hstu_max_output_len` (=20) via
`_resolve_nro_per_sample` (`xsurface_helpers.py:94-99`). Passing `32` would be
read by a future reader as if it applied.

## 4. Build, push, submit

Image build and job submission are the standard flow in
`docs/RUNBOOK_RUNNING_JOBS.md`. Two things specific to this manifest:

- Point `image:` at the tag you are about to push; it ships as `:latest`.
- The build needs a prebuilt `BARNFS_WHEEL`: building that wheel from source
  fails on master (a `_Nonnull` clang extension compiled by GCC under
  `opt-linux-x86_64-clang21-no-san`). Pass the artifact path in the environment.

## 5. Does the data behave like production

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

## 6. What the log should say

Beyond the generic startup, this run has four checkpoints:

1. `Pod N waiting for /mnt/barnfs/metadata.json and /mnt/barnfs/rank_N/batch_0/0.pt.zst as mount-ready probe...`
   — each pod waits for its own shard. The CSI driver publishes before the
   container starts, so this normally clears immediately. If it hangs, the
   mount failed or the dump's layout is not what section 0 says; exec in and
   look rather than waiting.
2. `[Replay] Dump (v0 bucketed): 480 dump ranks x 41942 batches. Rank N assigned
   629130 batches.` — proof that `_discover_files` read the manifest and that
   the pods are sharding rather than replicating.
3. Config load from the artifact. **Confirm the L2 stack has 11 layers** and
   that no `_trim_stu_stack` warning appears — that is the evidence the prod
   path actually took.
4. Per step,
   `[Replay] step K (pos K): sum(num_candidates)=..., NRO budget=10240 (fill=...%, max nc[i]=.../20)`.
   `max nc[i]` above 20 means the dump carries more candidates per row than
   prod's budget and the padding is clipping.

## Iterating

Only the image build and the submit repeat for a code change. The export and
publish (sections 1–2) repeat when prod's config changes, and whenever a parity
run has overwritten the shared `rankfm-prod-cfg` ConfigMap in between.

## Verifying the prod path really took

Worth doing once. `--prod_faithful` failing open would leave you with the
default config and a run that looks fine:

- The log's L2 layer count should be 11, not 10.
- `tests/test_prod_faithful_config.py` checks the built model's tables against
  the artifact — count, names, dims, placement. Run it locally with
  `PROD_CONFIG_ARTIFACT` pointed at your export to confirm the artifact is sound
  before spending cluster time.

```bash
PROD_CONFIG_ARTIFACT=/tmp/prod_cfg.pt \
  buck2 test //recir/examples/rankfm_mvai/tests:test_prod_faithful_config
```

## Note on uploads

Do not `gcloud storage cp` anything into the bucket from your devserver — the
MRS GCP audit flags it (T284199285). Nothing in this runbook needs it: the
config goes via ConfigMap and the data is already in the dump. The trainer
writing profiles and TensorBoard out from inside the cluster is a different
path and is not affected.
