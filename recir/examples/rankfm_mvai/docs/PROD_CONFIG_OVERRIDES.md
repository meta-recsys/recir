# What we override on top of prod's config

Baseline: `hammer.v3.models.udd_fm_v3.launch.prod_config_shim.ConfigProvider`,
which is prod's xsurface config
(`minimal_viable_ai/models/blue_reels_vdd/standalone/rt_mc/model_v5_hstu_standalone_cint_datafm_fm_xsurface_config.py`)
with one change of its own: `train_module_class_fqn` points at the hammer v3
train-module wrapper. Everything else below is ours.

This is a decision list for the 128-chip prod-config run
(`docs/RUNBOOK_128CHIP_PROD_CONFIG.md`). Each row says what changed, why, and
whether it can go. Tick them off as you decide.

Prod values worth having in front of you while reading:

| | |
|---|---|
| `HSTU_MAX_SEQ_LEN` | `3072 + 512 - 10 - 20 = 3554` (`..._xsurface_config.py:111`), hardcoded, no env var |
| HSTU depth | L1 = 3, L2 = 11 (`..._xsurface_config.py:1088-1089`) |
| `hstu_l2_max_seq_len` | 512 |
| `hstu_max_output_len` | 20 (`..._xsurface_config.py:1397`) |
| sparse optimizer | `ROWWISE_ADAGRAD, lr=0.02, eps=0.001`, warmup LINEAR 37500 / 0.1 / 1.0 |
| dense optimizer | `ADAMW, lr=4e-4, eps=1e-5, beta=(0.9, 0.998), wd=1e-4, clip=VALUE@1.0`, warmup LINEAR 37500 / 0.1 / 1.0 |
| DDP | `static_graph=True`, `allreduce_comm_precision="bf16"` (`..._xsurface_config.py:3793`) |

---

## A. Losses in transport (`prod_config_artifact.py`)

Not choices — the cost of carrying the config into an image that has no
`//minimal_viable_ai`.

- [ ] **A1. Per-table `init_fn` is dropped.** `_table_to_dict:402-404` skips any
  callable field, so a table that prod initialises with a custom function falls
  back to the default init in-image. This is a from-scratch run, so table init
  is not incidental.
  **RESOLVED — prod sets no `init_fn`.** All 10 tables are constructed inline in
  `_embedding_tables()` (`..._xsurface_config.py:2472`) and none passes the
  argument; the internal `EmbeddingTable.init_fn` defaults to `None`
  (`minimal_viable_ai/core/model_family_api/configs.py:1355`), and `callable(None)`
  is False, so the skip branch never fires and `init_fn=None` is carried like any
  other field. Nothing is lost today. Re-check if prod ever starts setting it —
  the drop would still be silent.

- [ ] **A2. Fields the mvai_cloud fork does not declare are dropped.**
  `_to_cloud:145-178` and `_decode_dataclass:283-316`. Guarded: dropping is only
  allowed when prod left the field at its default; a non-default value in a
  dropped field is a hard error. Safe as written.

- [ ] **A3. The feature config carries only primitives.**
  `_feature_config_primitives:434` keeps ints/floats/bools/strs and flat
  lists/dicts of them; prod's nested objects (`ro_float_features`,
  `nro_float_features`, ...) are lost, and the image starts from a fresh
  `BlueReelsVDDFeatureConfig()` with those primitives copied on.
  Mitigating: the nested objects are consumed at EXPORT time by
  `_translate_module_config` (`train_module_wrapper.py:250`), which reduces them
  to `float_feature_num` / `nro_float_feature_num` / task partitions and puts
  those in `module_config`, which does survive. On the replay path little else
  reads the feature config.

---

## B. Config assembly (`config.py`)

- [ ] **B1. The base `RecModelFamilyConfig` is the standalone one.**
  `build_model_family_config:540` starts from
  `udd_fm_config.ConfigProvider().create_config()` and overwrites only
  `module_config`, `feature_config`, and `optimizer_config` with prod's. Every
  other field on that object — `qcomms_config`, `embedding_tables={}`,
  `use_distributed_sharding=False` — is the standalone default, not prod's.
  Probably inert (recir owns embedding sharding through its own backend), but
  `qcomms_config` is worth a look if you care about comm precision.

- [ ] **B2. Static-shape mode is forced on.** `_apply_static_shape_mode:376-395`
  sets `static_shape_mode=True`, `static_nro_per_sample_budget =
  hstu_max_output_len` (20), and `static_ro_per_sample_budgets` = 3554 for each
  sequential RO key, each contextual key's own max length, 1 for timestamp.
  Required by the TPU/GSPMD path. Note the budgets are DERIVED from prod's own
  sequence parameters under `--prod_faithful` — no new numbers are introduced.

- [ ] **B3. `feature_config.feature_hash_sizes` is unconditionally rewritten**
  (`build_module_config:326`) to `{feature: table.num_embeddings}`. With no hash
  cap these are prod's real sizes, so this is a no-op in value.

- [x] **C8. RESOLVED -- the cap is gone and prod's 3e9 table runs uncapped.**
  The ring-shift path now carries ids wider than int32, so nothing forces a
  cap; `--hash_size_cap` is absent from the 128-chip manifest. Kept as the
  record of why the capped runs' numbers could not be compared to prod.

  It was forced by an int32 limit in the ring-shift kernel, not chosen.
  `recir/torchax/ops/jseq_embedding.py:86` computes `% vocab` against a Python
  int and `:87` builds the ids as int32. Prod's `nonpooled_post_id` is 3e9 rows,
  above int32's 2147483647, so XLA lowering fails:

  ```
  OverflowError: An overflow was encountered while parsing an argument to a
  jitted computation, whose argument path is x2. Got <class 'int'> with value
  3000000000
  ```

  The `%` is the loud half and is a no-op anyway whenever `vocab > batch*seq`.
  The dangerous half is the `astype(jnp.int32)` on the next line: ids reach this
  kernel folded into `[0, vocab)`, so anything above 2147483647 wraps NEGATIVE
  and silently indexes the wrong row. Observed on
  `rankfm-mvai-128chip-4lqmz`.

  2e9 fits int32 and divides evenly by the 256-device world size, so `_round_up`
  does not push it back over. It touches exactly one table -- `config.py` caps
  with `min(num_embeddings, cap)` and the next largest is `viewer_rid` at 1e9.

  **Why this invalidates the numbers:** `remap_batch_ids` folds ids by the
  table's size, so changing the size changes the MODULUS. Ids land on different
  rows than production put them on -- a different model on the same data, not a
  smaller one. Run it to see how much further the pipeline gets; do not compare
  its CTR or NE against the MAST baseline.

  The fix was to carry ids as int64 through the ring-shift and drop the no-op
  modulo, which needs `JAX_ENABLE_X64=1` on the manifest. A guard now raises if
  the vocabulary exceeds int32 while the id path came out narrower, so the
  silent wraparound cannot come back.

- [x] **C9. RESOLVED -- prod's 3554 is back.** The cap was never about the
  sequence: capping it to 512 moved the peak by 1G (144.43 -> 143.42). The
  actual consumer was a single `[T, B, D_out, E]` tensor in `FusedMMoE`'s
  group-balance branch (`hammer/v3/models/udd_fm_v3/modules.py`), 82.5GiB at
  T=65/B=332800/D_out=128/E=8, materialized only to apply a per-task scalar to
  the backward gradient. Scaling after the reduction instead of fanning out
  before it removes it exactly -- same forward, same scaled expert gradient,
  same unscaled gating gradient. The `seq_len_cap` knob stays in `config.py` as
  a diagnostic; this run no longer passes it.

  Superseded, kept for the reasoning:

  Forced by HBM, not chosen. With prod's length the run compiled and then died
  executing the step:

  ```
  RESOURCE_EXHAUSTED: Ran out of memory on HBM, the total memory required for
  HLO temporaries (144.43G) exceeds available HBM (94.75G)
  ```

  Note that is TEMPORARIES, not tables. The tables were fine -- 8GB
  (post_id at the 2e9 cap) + 4GB (viewer_rid) + 0.16GB (w2v) per device against
  94.75GB of HBM. What does not fit is the activation working set, and the
  reason is **C2**: the PyTorch reference attention kernel materializes a dense
  `[B, H, N, N]` score matrix and masks afterwards, so `max_attn_len` buys
  nothing and the cost is quadratic in N. The same shape was measured on the
  parity legs at 77GB of scores per layer per leg (B=512, H=4, N=3067).
  Observed on `rankfm-mvai-128chip-czznv`.

  `--prod_faithful` ignores `--max_uih_seq_len` (it arrives through
  `_apply_bench_shape_overrides`, which prod-faithful skips) and the prod
  provider hardcodes 3554 with no env override, so `_apply_seq_len_cap` in
  `config.py` is the only way in. It caps BOTH stacks, L2 to half the cap:
  at `l2 >= history` the split `minus_k = clamp(ro_lengths - l2_max_seq_len,
  min=0)` collapses to 0 (`mot_preprocessor.py:1069`) and the L1 stack gets
  zero-length sequences while the run keeps going. hammer's own
  `replay_config_shim` takes `min()` on L2 for the same reason.

  **Why this invalidates the numbers:** `hstu_max_seq_len` is a real data-side
  truncation -- `truncate_ro_features` keeps only the most recent N events per
  row. At 512 the model reads a seventh of the user history production reads.
  That is a different model, not a cheaper one.

  The fix that would let prod's length run is the Pallas HSTU attention kernel
  (`RECIR_HSTU_PALLAS=1`, `rankfm_model_utils.py:118`), which does not
  materialize the N x N scores. It is wired into the bench manifests but has
  never completed a run, so it is not a lever to pull yet.

- [ ] **B4. Deliberately NOT overridden**, listed so the absence is on the
  record: `inter_arch_type` untouched, no bench shape overrides
  (`_apply_bench_shape_overrides` is skipped under `--prod_faithful`). The hash
  cap and the sequence-length sampler used to be on this list and are not any
  more -- see C8 and C7.

---

## C. Model construction (`rankfm_model_utils.build_model_from_config`)

The group to read most carefully — these mutate the built model directly.

- [ ] **C1. `use_cuda_attention = False` on every STU layer** (`:71-75`).
  Forced; there is no CUDA on TPU.

- [ ] **C2. `_hammer_kernel = HammerKernel.PYTORCH`** (`:103`). Forced; the
  Triton kernels have no TPU support. Cost: the PyTorch reference kernel
  materializes a dense `[B, H, N, N]` score matrix and masks afterwards, so it
  does not exploit `max_attn_len`. At N ~ 3554 this dominates the step. Not
  removable, but it explains the step time.

- [x] **C3. `_bf16_training`** (`:122`, from `RankFMTrainerConfig`'s default).
  **Prod trains in bf16** and additionally does its DDP allreduce in bf16
  (`..._xsurface_config.py:3793`), while the dataclass default is fp32.
  **FIXED: the 128-chip manifest now passes `--bf16_training`.** fp32 master
  weights are still kept and layernorm / softmax / loss reductions stay fp32.
  This changes activation dtypes, so it forces an XLA recompile.

- [x] **C4. Dropout was zeroed; it is now live.** Prod runs
  `hstu_linear_dropout_rate=0.1` (`..._xsurface_config.py:1405`), which reaches
  the STU layers as `dropout_ratio` via `train_module_wrapper.py:139,208`.
  (`hstu_input_dropout_ratio` is 0.0 in prod, so only the linear-path dropout was
  ever lost.) This was the one override that changed training dynamics with no
  basis in prod's config.

  **RESOLVED.** Two independent problems had to be solved, and it took four
  attempts because the first three fixed only the visible one:

  1. *The tracer leak.* `RuntimeProperty.get_and_rotate_prng_key`
     (`recir/torchax/tensor.py:431-435`) does `self.prng = new_prng_key`, so
     during a `jit` trace it stores a `DynamicJaxprTracer` in Python state. Note
     the "second jit call" happens *within the first training step*:
     `xla_lower.py` traces at both `:1191` and `:1169`.
  2. *The constant fold, which was the real blocker.* The step is traced and
     compiled ONCE and the PRNG key was not part of `flat_state`, so a key that
     merely avoided the tracer error was staged in as a constvar and every step
     used the identical mask. That is worse than no dropout, and it is
     indistinguishable from a working fix by any check short of reading the HLO.

  The fix threads the key as a tensor ARGUMENT instead of environment state.
  `recir::bernoulli_keyed` (`recir/ops/random_op.py`, lowered in
  `recir/torchax/ops/jrandom.py`) takes the LR warmup counter, which is already a
  registered buffer and therefore already in `flat_state`, and folds the key from
  it; `DropoutPRNGTransform` (`recir/region_transforms/dropout_prng.py`) rewrites
  the traced `aten.bernoulli_` nodes to call it. The data dependency is what pulls
  the counter into the same lowered subgraph as the dropout. Three earlier
  attempts looked verified and were not; what separated them was reading the
  HLO, not the exit code.

  **Still no test runs dropout > 0 through the recir compiled path**, and the old
  workaround remains hardcoded in recir's own test utils
  (`recir/tests/xla/dlrmv3_test_utils.py:359-362`). The test to write is a model
  with dropout > 0 through `compile_gspmd_step` for >= 3 steps, asserting the
  losses on a fixed batch differ across steps. That one test catches both the
  tracer leak and the constant fold.

- [x] **C5. Every floating-point parameter is set `requires_grad_(True)`**
  (`model_family.py:668-670`).
  **RESOLVED — prod freezes nothing, so this is a no-op today.** No
  `freeze_layers_schedule` is set anywhere under
  `minimal_viable_ai/models/blue_reels_vdd/` (the knob exists, declared at
  `minimal_viable_ai/core/model_family_api/configs.py:2940`), and nothing in the
  hammer v3 model path calls `requires_grad_(False)`. The only "frozen" strings
  in the prod config are feature names
  (`watch_vs_reels_segment_v1_frozen_20240301:0`).
  Related gap worth knowing: `freeze_layers_schedule` lives on
  `RecModelFamilyConfig`, which the artifact does not carry at all, and it is not
  declared on the mvai_cloud fork either — so if prod ever sets one it will
  vanish in transport rather than error. See B1.

- [ ] **C6. `torch.manual_seed(0)`** before the module build and before the
  trace batch (`model_family.py:666,685`). Reproducibility; harmless.

- [x] **C7. RESOLVED -- sampling is on, as prod has it.** It used to be forced
  off in `rankfm_model_utils.build_model_from_config`, not as a choice but
  because the trace did not survive it. Kept as the record of what the
  override cost and what actually blocked it.

  Prod enables the L1 sampler and derives the output segment from it as a
  matched pair (`..._xsurface_config.py:1664-1665`):

  ```python
  output_segments=["unsampled_inputs"] if SEQ_LEN_SAMPLING_CONFIGS else None,
  hstu_seq_len_sampling_configs=SEQ_LEN_SAMPLING_CONFIGS,
  ```

  with `SEQ_LEN_SAMPLING_CONFIGS` set (`:143`) and `L2_SEQ_LEN_SAMPLING_CONFIGS`
  None (`:153`).

  Why it does not survive here: `mot_preprocessor` derives its two flags PER
  HSTU STACK (`:800-807`) while `output_segments` is a single global list. For
  the L2 stack that gives `_needs_unsampled_inputs=True` with
  `_has_seq_len_sampling=False`, so the export at `:987-997` falls through to
  its `elif supervision_weight is not None` branch -- which does not fire
  either, so nothing sets the `"unsampled_inputs"` export and
  `udd_fm_train_v3._post_processing:1693` raises
  `KeyError: 'unsampled_inputs'` during the FX trace. Observed on
  `rankfm-mvai-128chip-n8qqg`.

  `--disable_seq_len_sampling` still clears both fields for a run that wants
  them off, and the light leg's shim does the same. Leaving them off is a real
  behavioural difference, and a trap that hides well: prod randomly truncates
  user history every step, so two legs that do not both sample see different
  sequences from the same batch -- and it changes SHAPES rather than values, so
  no reduction or l2 comparison can see it.

  **How it was fixed:** the export was being dropped by the trace-time stack
  swap, not by the sampler. `DummyPreprocessorStack` rebuilds its output bundle
  from tensor/KJT/non-tensor descriptors, and exports ride a separate str-keyed
  channel on `FeatureBundle` that the rebuild never reached, so the segment the
  real preprocessor produced never made it into the trace. Exports now travel
  the same path as every other output.

---

## D. Optimizers (`model_family.py`)

- [ ] **D1. Sparse side is faithful.** `optimizer_config.sparse.default`
  (`:303`) comes from the artifact, i.e. prod's ROWWISE_ADAGRAD.

- [x] **D2. Dense side did NOT come from prod's config.** `:320-323` passed
  `trainer_cfg.dense_optimizer_config` — a hardcoded default on
  `RankFMTrainerConfig` — and the artifact's `optimizer_config.dense` was never
  read.
  **Verified: the hardcoded values were field-for-field identical to prod's**
  (ADAMW, 4e-4, 1e-5, 0.9/0.998, wd 1e-4, VALUE clip at 1.0, LINEAR warmup
  37500/0.1/1.0), so this was never wrong — only a copy that would not follow
  prod.
  **FIXED: `config.resolve_dense_optim_config` now returns
  `optimizer_config.dense.default` whenever `PROD_CONFIG_ARTIFACT` is set.**
  Gated on the artifact rather than switched unconditionally: with no artifact,
  `build_optimizer_config` falls back to the STANDALONE udd_fm config, whose
  dense side is `ADAM/1e-3/eps 1e-8` against prod's `ADAMW/4e-4/eps 1e-5`, and
  every bench and parity manifest takes that path expecting the dataclass
  default. The chosen source is logged as `[optimizer] dense: ...`.

- [ ] **D3. Only `sparse.default` is used**; `sparse.embedding_optim_groups` is
  ignored. Prod's config sets no per-group optimizers, so no effect today.

- [x] **D4. `RankFMTrainerConfig.embedding_lr = 0.1` was dead code.** Nothing in
  `rankfm_mvai` read it (only `recir/examples/dlrmv3` has its own). It had
  already cost one wrong diagnosis: an NE investigation chased an `lr=0.1`
  that turned out to be this field showing up in a config dump while both legs
  actually ran `ROWWISE_ADAGRAD lr=0.02`.
  **FIXED: field deleted**, along with the now-unused `skip` parameter on
  `_add_dataclass_args`.

- [ ] **D5. Sharding type is our heuristic, not prod's.**
  `recir/embedding/cpu.py:290` picks ROW_WISE vs TABLE_WISE from the table's row
  count against a per-rank minimum. Prod's `sharding` field is only consulted to
  split DATA_PARALLEL tables out of the fused/sparse path
  (`_split_out_replicated_tables:246`). Sharding layout does not change
  numerics, but the DATA_PARALLEL split does decide which optimizer a table
  gets, so it matters that `embedding_tables` is passed (it is).

---

## E. Data path

- [ ] **E1. `remap_batch_ids`** (`data_pipe.py:410`) applies a per-key modulo
  into each table's `num_embeddings`. With no hash cap the sizes are prod's own
  and the dump is already hashed at them, so this is the identity. Note it
  applies `remainder()` unconditionally, so an out-of-range id would be folded
  silently rather than raising.

- [ ] **E2. `--local_batch_size 512`** matches the dump's stride and prod's own
  per-rank batch size, so `select_rows_slice` is a no-op and each step consumes
  exactly one dumped batch.

- [x] **E3. Global batch is 32 x 512 = 16384.** Accepted as-is. Production's
  global batch depends on the MAST run's rank count and is not matched here.
  This does not affect CTR (a label statistic) but does affect NE.

- [x] **E4. Sequence-length sampling is ON, as prod has it.** See **C7**.
  Worth knowing it randomly truncates user history every step, so two legs that
  do not both run it see different sequences from the same batch.

---

## F. Trainer layer (`config.py:build_trainer_config`)

Orchestration only, nothing that touches the model:
`SystemConfig(use_gpu=False, dist_backend="gloo")` (recir still places dense
compute on TPU via `jax.devices()`); `RandomDataSourceConfig` purely so the QPS
metric has a positive batch size — real data comes from
`create_train_datapipe`; `max_steps` enforced by the data pipe raising
`StopIteration`; checkpointing off.

---

## Status

Resolved, no action needed: **A1** (prod sets no `init_fn`), **C5** (prod
freezes nothing), **D3**, **E1**, **E2**, **E3**, **E4**, **B3**, **B4**.

Fixed: **C3** (manifest now passes `--bf16_training`), **D2**
(`build_optimizer_config` is now the single source for BOTH sides;
`RankFMTrainerConfig.dense_optimizer_config` deleted), **D4** (`embedding_lr`
deleted).

**Open deltas from prod, in priority order:**

1. **C9 -- sequence length capped at 512 against prod's 3554.** Forced by HBM,
   because the PyTorch reference attention kernel is quadratic in N (see C2).
   Truncates the user history the model reads to a seventh of production's.
2. **C8 -- the largest table is capped at 2e9 against prod's 3e9.** Forced by an
   int32 limit in the ring-shift kernel. Changes the id->row modulus.
3. **C7 -- sequence-length sampling is off, prod has it on.** Carried only
   because the FX trace raises `KeyError('unsampled_inputs')` without it. The
   root cause is understood down to the branch; the remaining unknown is why
   `supervision_weight` is None there.
4. **C4 -- dropout is zeroed, prod runs `hstu_linear_dropout_rate=0.1`.** Needs
   the PRNG key threaded through recir's compiled state; the cheap workaround
   produces a fixed mask and is worse than the current state.

C9 and C8 both change what the model sees, not just how fast it runs. **While
either is in place this configuration is a pipeline probe: do not compare its
CTR or NE against the MAST baseline.** C7 and C4 are smaller but point the same
way.

The single change that would retire C9 -- and probably make C8 worth revisiting
-- is the Pallas HSTU attention kernel (`RECIR_HSTU_PALLAS=1`), which avoids
materializing the N x N score matrix. It is wired into the bench manifests and
has never completed a run.

Worth a second look if the numbers disagree: **A3** (feature config carries only
primitives), **B1** (base config is the standalone one, so anything on
`RecModelFamilyConfig` that prod sets — including a future
`freeze_layers_schedule` — is silently dropped), **C2** (the PyTorch attention
kernel is O(N^2) and dominates the step at N ~ 3554), **D5**.
