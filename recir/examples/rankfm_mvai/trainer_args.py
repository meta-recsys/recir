# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""The trainer's CLI surface: two dataclasses and the argv parsing around them.

Split out of ``config.py`` so it can be read without the prod xsurface chain.
``config.py`` imports hammer, torchrec and the udd_fm config providers, which
takes minutes to load and reaches Configerator; these are plain int/bool/str
dataclasses and an argparse pass, and three things want them cheaply:

* ``launch_configs.py``, to check that every flag a launch config names is a real field --
  otherwise a typo reaches argparse as "unrecognized arguments" at pod start,
  twenty minutes in.
* the tests for the flag warnings below.
* ``config.py`` itself, which re-exports everything here, so no caller had to
  change.

Stdlib-only, and it needs to stay that way for the first of those.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from dataclasses import dataclass
from typing import Any, Tuple

logger: logging.Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RankFMTrainerConfig:
    """Pure trainer knobs: batching, optimizer, runtime mode, profiling.

    Nothing here changes the model identity or FX graph topology.
    """

    # Data. Must be a whole multiple of the dump's own batch_size (512) on the
    # replay path -- data_pipe.BatchReplayPipe.batches_per_step raises
    # otherwise, because a step reads whole files and anything else silently
    # trains on a subsample. A multiple above 1 merges that many dumped files
    # per step, so the global batch is local_batch_size * hosts, and every
    # doubling halves the steps per epoch, so max_steps has to follow.
    local_batch_size: int = 32

    # Training
    max_steps: int = 20

    # Checkpointing (DCP via mvai_cloud CheckpointManager). Periodic save every
    # N steps (0 = disabled); keep_last_n bounds retained checkpoints. Paths come
    # from the MVAI_JOB_NAME / MVAI_CHECKPOINT_SAVE_DIR / MVAI_CHECKPOINT_LOAD_DIR
    # env vars (see CheckpointManager).
    save_every_n_steps: int = 0
    keep_last_n: int = 2
    # Stage a consistent host snapshot at the checkpoint boundary, then overlap
    # the distributed storage write with subsequent train steps.
    async_checkpoint: bool = False

    # Mean over the padded candidate width instead of the local candidate sum,
    # which is what light does (its forward omits loss_norm_denom, so the module
    # falls back to torch.mean). Set this to make the eager loss directly
    # comparable to a light run. Default False keeps the TPU leg's global-sum
    # scope. Only meaningful with enable_padding=False.
    light_loss_norm: bool = False

    # Micro-batches accumulated per optimizer step, so the CPU leg matches a
    # multi-rank GPU leg's GLOBAL batch without materializing it.
    grad_accum_steps: int = 1

    # Precision. When True, run the dense tower (HSTU + composer + loss matmuls)
    # in bf16 on device via recir.torchax AMP autocast + the HSTU's
    # ``_bf16_training`` activation casts. fp32 master weights are kept (SGD over
    # fp32 dense params), and layernorm / softmax / loss reductions stay fp32 ---
    # the same mixed-precision recipe the GPU baseline uses. NOTE: unlike the
    # other knobs in this dataclass, this DOES change activation dtypes / the FX
    # graph, so toggling it forces an XLA recompile.
    bf16_training: bool = False

    # Real data replay (synthetic data is used when gcs_data_path is empty)
    gcs_data_path: str = ""
    data_file_template: str = ""
    data_replicate_across_ranks: bool = False
    # Background-thread batch prefetch depth: how many batches the loader may
    # prepare ahead so file load, decompression, slicing, and ID remapping overlap
    # the device train_step. 0 disables prefetch (synchronous load).
    data_prefetch_buffer: int = 32
    data_load_workers: int = 2

    # Initial weights from //hammer/v3/models/udd_fm_v3/parity:export_parity_checkpoint.
    # Empty (default) = normal initialisation.
    init_state_dict_path: str = ""

    # Embedding backend: "cpu" (torchrec DMP) | "tpu" (JAX/SparseCore).
    # Validated by ``build_embedding_backend`` in ``embedding/__init__.py``.
    embedding_backend: str = "cpu"

    # Embedding table init for the "in_graph" backend. True = meta param +
    # on-device HBM random init (no host alloc/H2D; needed for huge/1T tables).
    # False = real host param (materialize + H2D; fine at ~1M rows).
    on_device_init: bool = True

    # Runtime mode: enables PaddingRegionTransform AND flips
    # static_shape_mode on the model config (since the two must agree).
    enable_padding: bool = True

    # Input-pipeline overlap (host->device). prefetch_batches>0 builds/loads the
    # next batch(es) on a background thread; pin_memory pins host batch buffers.
    # NOTE: defaulted OFF. For the synthetic bench these regressed slightly — the
    # batch build is cheap and isn't the bottleneck (the H2D is, and it overlaps
    # via JAX async dispatch once per-step output materialization is deferred; see
    # Phase 3 / `_pull_metrics_every`). The background producer thread only adds
    # GIL contention with the main thread's critical-path H2D here. Re-enable for
    # a real (slow) GCS data pipeline, where overlapping the load does help.
    prefetch_batches: int = 0
    pin_memory: bool = False

    # Phase-3 pipelining: pull the step outputs (loss/metrics) host-ward only every
    # N steps instead of every step. Deferring the per-step output materialization
    # lets JAX run the host ahead, overlapping step i+1's H2D/dispatch with step
    # i's compute (the input gap collapses toward the device floor). EVERY step's
    # output handle is retained and flushed in one D2H burst at the pull, so no
    # batch's metrics are dropped — only the host<->device sync cadence changes.
    # 1 = pull every step (no pipelining).
    pull_metrics_every: int = 10

    # How often the metrics reporter emits NE/loss, in steps. Set to 1 to get a
    # point per step, which a short converging run needs -- at the default 10 a
    # 30-step run yields 3 points, which is not a curve. Must match the light
    # leg's METRICS_REPORT_INTERVAL for the two curves to line up.
    metrics_report_interval: int = 10
    metric_window_size: int = 1_000_000

    # Reduce the metric inputs on device instead of pulling them host-ward.
    in_graph_metrics: bool = False

    # Profiling
    profile_start_step: int = -1
    profile_end_step: int = -1
    profile_dir: str = "/tmp/tpu_profile"
    enable_memory_log: bool = False

    # Precompute the preprocessor graph break one step ahead on a background
    # thread, so it overlaps the previous step's device compute instead of
    # stalling it.
    pipeline_preprocessor: bool = False

    # Make the sequence-length sampler enforce a derived token budget
    # rather than letting the padder truncate whatever overshoots. A trainer
    # knob, not a model one: the sampler is swapped on the built modules at
    # setup, and only the static-shape path has a budget to enforce. Off by
    # default -- it changes which rows keep their history, so it is not
    # prod-faithful.
    budgeted_seq_len_sampling: bool = False

    # TensorBoard scalar logging. Empty tb_dir disables TB entirely.
    # Local paths write directly; gs:// paths buffer to a local tempdir
    # during training and upload on flush/close (gcsfs has no append).
    tb_dir: str = ""
    tb_log_every: int = 10


@dataclass
class RankFMModelArgs:
    """CLI overrides for ``BlueReelsVDDModuleConfig``.

    Anything that changes the model architecture, sequence shapes,
    or static-shape padding budgets.  Changing these requires
    rebuilding the model.
    """

    # Embedding sizing: cap large hash tables for the TPU memory budget.
    # 0 (default) disables the cap -> tables keep their natural num_embeddings;
    # set a positive value to clamp every table's num_embeddings down to it.
    hash_size_cap: int = 0

    # HSTU stack sizing
    l1_num_layers: int = 3
    l2_num_layers: int = 11

    # This is the only way to shorten sequences under --prod_faithful:
    # ``max_uih_seq_len`` reaches the config through
    # ``_apply_bench_shape_overrides``, which prod-faithful skips, and the prod
    # provider hardcodes its length with no env override.
    seq_len_cap: int = 0

    # Sequence shapes (architectural maxes)
    max_uih_seq_len: int = 32
    num_candidates: int = 8

    # L2 (second HSTU stack) max sequence length. When > 0 this pins
    # hstu_l2_max_seq_len directly (used to align with the GPU bench at 512);
    # when 0 (default) it falls back to the historical max_uih_seq_len // 4
    # derivation, preserving behavior for existing users.
    l2_max_seq_len: int = 0

    # Sequence-length sampling (data-side dropout during training)
    disable_seq_len_sampling: bool = False

    # Per-sample RO history token budget for static_shape_mode.
    # Applies to every key in ``merge_item_uih_feature_mapping``;
    # contextual keys take their per-feature max length; the timestamp
    # key takes 1.  Lower than ``max_uih_seq_len`` shrinks the HSTU dense buffer
    seq_ro_per_sample_budget: int = 24

    # Build the model from the PROD xsurface provider rather than the standalone
    # udd_fm defaults.
    prod_faithful: bool = False

    # DIAGNOSTIC: drop prod's feature-interaction DCN even under
    # --prod_faithful.
    disable_inter_arch: bool = False


# ---------------------------------------------------------------------------
# CLI parsing -- launchers (YAML / shell) pass --field value to override the
# dataclass defaults above.  Primitive (int/float/str/bool) fields are
# auto-registered; non-primitive fields (e.g. Dict budgets) are not
# CLI-exposed and must be set in code.
# ---------------------------------------------------------------------------


def _add_dataclass_args(
    parser: argparse.ArgumentParser,
    cls: Any,
    *,
    suppress_defaults: bool = False,
) -> None:
    """Register `--<field>` CLI args for each primitive field of *cls*.

    Booleans get a paired ``--<field>`` / ``--no_<field>`` toggle so launchers
    can flip either direction regardless of the dataclass default.

    With ``suppress_defaults`` every default becomes ``argparse.SUPPRESS``, so
    the resulting namespace holds only the flags actually present on the
    command line. ``_explicitly_passed_flags`` uses that to tell "passed the
    default value" apart from "did not pass it", which comparing against the
    dataclass default cannot do -- and the one mistake this has to catch,
    ``--l1_num_layers 3 --l2_num_layers 11`` under --prod_faithful, passes
    exactly the defaults.
    """
    for f in dataclasses.fields(cls):
        default = argparse.SUPPRESS if suppress_defaults else f.default
        flag = f"--{f.name}"
        if f.type == "bool":
            grp = parser.add_mutually_exclusive_group()
            grp.add_argument(flag, dest=f.name, action="store_true", default=default)
            grp.add_argument(
                f"--no_{f.name}", dest=f.name, action="store_false", default=default
            )
        elif f.type == "int":
            parser.add_argument(flag, type=int, default=default)
        elif f.type == "float":
            parser.add_argument(flag, type=float, default=default)
        elif f.type == "str":
            parser.add_argument(flag, type=str, default=default)


def _explicitly_passed_flags() -> set[str]:
    """Field names present on the command line, whatever value they carry."""
    probe = argparse.ArgumentParser(add_help=False)
    _add_dataclass_args(probe, RankFMTrainerConfig, suppress_defaults=True)
    _add_dataclass_args(probe, RankFMModelArgs, suppress_defaults=True)
    # parse_known_args, not parse_args: the real parser above has already
    # rejected anything unrecognised, and --help has already exited.
    namespace, _unknown = probe.parse_known_args()
    return set(vars(namespace))


# Flags that only reach the config through ``_apply_bench_shape_overrides``,
# which ``build_module_config`` skips under --prod_faithful. Passing one there
# changes nothing, but reads like it does.
_BENCH_SHAPE_FLAGS: tuple[str, ...] = (
    "l1_num_layers",
    "l2_num_layers",
    "max_uih_seq_len",
    "l2_max_seq_len",
    "num_candidates",
    "seq_ro_per_sample_budget",
)


def _warn_on_incoherent_flags(
    trainer_cfg: RankFMTrainerConfig,
    model_args: RankFMModelArgs,
    passed: set[str],
) -> None:
    """Warn about flag combinations that are silently not what they look like.

    *passed* is the set of field names actually present on the command line.
    It matters for the inert-flag check: the documented mistake is
    ``--l1_num_layers 3 --l2_num_layers 11`` under --prod_faithful, which
    *looks* like it matches prod (prod is L1=3, L2=11) and does nothing --
    and those are the dataclass defaults, so a default comparison is blind to
    it.

    Warnings, never errors: every combination below has a legitimate caller
    (the 4-chip verify run and both parity legs cap the tables on purpose), so
    raising would break them. What is not legitimate is reaching one of these
    by accident on a run whose numbers are meant to be compared against prod,
    and today nothing says so -- ``hash_size_cap`` rewrites feature_hash_sizes
    from the post-cap sizes, leaving the whole pipeline self-consistent and the
    divergence undetectable at runtime.
    """
    if model_args.prod_faithful and model_args.hash_size_cap > 0:
        logger.warning(
            "[config] DIAGNOSTIC: --hash_size_cap %d clamps prod's tables "
            "(largest is ~3e9 rows). feature_hash_sizes is rebuilt from the "
            "capped sizes, so nothing downstream can detect this. The model is "
            "no longer prod-faithful and its NE is not comparable with prod.",
            model_args.hash_size_cap,
        )

    if trainer_cfg.data_file_template:
        logger.warning(
            "[config] DIAGNOSTIC: --data_file_template %r resolves to exactly "
            "ONE file, which replay_batch_iter then recycles through its "
            "rotating row window. This run is a throughput benchmark, not "
            "training; its NE/CTR curve is not comparable to anything.",
            trainer_cfg.data_file_template,
        )

    if model_args.prod_faithful:
        inert = [name for name in _BENCH_SHAPE_FLAGS if name in passed]
        if inert:
            logger.warning(
                "[config] --prod_faithful takes the shapes from prod's config, "
                "so these flags are IGNORED: %s. Prod's own values are used "
                "instead, whatever you passed. --seq_len_cap is the only way "
                "to shorten sequences on this path.",
                ", ".join(f"--{name}={getattr(model_args, name)}" for name in inert),
            )


def parse_cli() -> Tuple[RankFMTrainerConfig, RankFMModelArgs]:
    """Parse argv, returning dataclass instances with any --flag overrides applied."""
    parser = argparse.ArgumentParser()
    _add_dataclass_args(parser, RankFMTrainerConfig)
    _add_dataclass_args(parser, RankFMModelArgs)
    args = parser.parse_args()
    trainer_cfg = RankFMTrainerConfig(
        **{
            f.name: getattr(args, f.name)
            for f in dataclasses.fields(RankFMTrainerConfig)
            if hasattr(args, f.name)
        }
    )
    model_args = RankFMModelArgs(
        **{
            f.name: getattr(args, f.name)
            for f in dataclasses.fields(RankFMModelArgs)
            if hasattr(args, f.name)
        }
    )
    _warn_on_incoherent_flags(trainer_cfg, model_args, _explicitly_passed_flags())
    return trainer_cfg, model_args
