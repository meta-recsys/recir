#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Configurable RankFM (xsurface V3) model + training parameters, plus the
mvai_cloud config builders that RecTrainer consumes.

The xsurface model has a heavy production-default config in
``hammer.v3.models.udd_fm_v3.udd_fm_config.BlueReelsVDDModuleConfig``,
hardcoded hash tables and feature lists). For TPU experiments we only
expose a small set of configs split into two clean buckets:

* :class:`RankFMTrainerConfig` -- pure trainer knobs (batching, optimizer,
  runtime mode toggles, profiling). Nothing here affects the model identity
  or FX graph topology;
* :class:`RankFMModelArgs` -- overrides that mutate ``BlueReelsVDDModuleConfig``
  (architecture depth, sequence shapes, hash caps).

On top of the CLI surface this module also produces the two mvai_cloud config
objects RecTrainer consumes:

* :func:`build_model_family_config` -- a ``RecModelFamilyConfig`` built from the
  hammer udd_fm ``ConfigProvider().create_config()`` with the RankFMModelArgs
  mutations applied (via :func:`build_module_config`).
* :func:`build_trainer_config` -- a ``RecTrainerConfig`` with
  ``SystemConfig(use_gpu=False)`` so ``RecTrainer.__init__`` takes its CPU/gloo
  device path; recir lowers the dense graph to TPU via ``jax.devices()``
  regardless of the torch device.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Dict, Tuple

from hammer.v2.modules.configs import STUStackConfig
from hammer.v2.modules.uih_sampling_modules import SequenceLengthSamplingStrategy
from hammer.v3.models.udd_fm_v3.batch_loader import BlueReelsVDDFeatureConfig
from hammer.v3.models.udd_fm_v3.udd_fm_config import (
    BlueReelsVDDModuleConfig,
    ConfigProvider,
)
from minimal_viable_ai.core.dataloader.dataloader_configs import RandomDataSourceConfig
from minimal_viable_ai.core.model_family_api.configs import RecModelFamilyConfig
from minimal_viable_ai.core.model_family_api.optimizer_configs import RecOptimizerConfig
from minimal_viable_ai.core.trainer.trainer_config_def import (
    CheckpointConfig,
    MetricsConfig,
    RecTrainerConfig,
    SystemConfig,
)

__all__ = [
    "RankFMTrainerConfig",
    "RankFMModelArgs",
    "parse_cli",
    "build_module_config",
    "build_optimizer_config",
    "build_model_family_config",
    "build_trainer_config",
]

logger: logging.Logger = logging.getLogger(__name__)


from recir.examples.rankfm_mvai.trainer_args import (  # noqa: F401
    parse_cli,
    RankFMModelArgs,
    RankFMTrainerConfig,
)


# ---------------------------------------------------------------------------
# Optimizer-config builder
# ---------------------------------------------------------------------------


def build_optimizer_config() -> RecOptimizerConfig:
    from recir.examples.rankfm_mvai.prod_config_artifact import (
        artifact_path,
        load_prod_optimizer_config,
    )

    path = artifact_path()
    config = (
        load_prod_optimizer_config(path)
        if path is not None
        else ConfigProvider().create_config().optimizer_config
    )
    logger.info(
        "[optimizer] source=%s dense=%s lr=%g sparse=%s lr=%g",
        "prod config artifact" if path is not None else "standalone udd_fm",
        config.dense.default.optim_type,
        config.dense.default.lr,
        config.sparse.default.optim_type,
        config.sparse.default.lr,
    )
    return config


# ---------------------------------------------------------------------------
# Module-config builder (single source of truth for CLI -> module_config)
# ---------------------------------------------------------------------------


def _load_prod_module_config() -> Tuple[
    BlueReelsVDDModuleConfig, BlueReelsVDDFeatureConfig
]:
    from recir.examples.rankfm_mvai.prod_config_artifact import (
        artifact_path,
        build_prod_config_live,
        load_prod_config,
    )

    path = artifact_path()
    return load_prod_config(path) if path is not None else build_prod_config_live()


def build_module_config(
    model_args: RankFMModelArgs,
    *,
    enable_padding: bool,
    budgeted_seq_len_sampling: bool = False,
) -> Tuple[BlueReelsVDDModuleConfig, BlueReelsVDDFeatureConfig]:
    """CLI args -> fully configured (module_config, feature_config)."""
    # Declared up front so both branches narrow to the same types; without this
    # pyre unions the prod-faithful branch with RecModelFamilyConfig's loosely
    # typed attributes and widens everything downstream to `object`.
    module_config: BlueReelsVDDModuleConfig
    feature_config: BlueReelsVDDFeatureConfig
    if model_args.prod_faithful:
        module_config, feature_config = _load_prod_module_config()
    else:
        config: RecModelFamilyConfig = ConfigProvider().create_config()
        module_config = config.module_config  # pyre-ignore[9]
        feature_config = config.feature_config  # pyre-ignore[9]

    if not model_args.prod_faithful or model_args.disable_inter_arch:
        if model_args.disable_inter_arch and model_args.prod_faithful:
            logger.warning(
                "[config] DIAGNOSTIC: dropping prod's inter_arch DCN "
                "(inter_arch_type=%r -> None). The model is no longer "
                "prod-faithful.",
                module_config.inter_arch_type,
            )
        module_config.inter_arch_type = None

    # Cap embedding-table hash sizes (hash_size_cap <= 0 disables capping).
    if model_args.hash_size_cap > 0:
        for table in module_config.embedding_tables.values():
            table.num_embeddings = min(table.num_embeddings, model_args.hash_size_cap)

    # Per-feature hash size (post-cap), passed to the synthetic data generator so
    # it samples each id over its table's full num_embeddings. This makes
    # row-wise/block-sharding spread ids across all ranks; without it the toy
    # generator drew tiny ids that all land on shard 0, degenerating the
    # ring-shift all-to-all to a single hot peer (~100x slower lookup).
    feature_hash_sizes: Dict[str, int] = {}
    for table in module_config.embedding_tables.values():
        for feat in table.features:
            feature_hash_sizes[feat] = table.num_embeddings
    for emb in module_config.additional_embeddings.values():
        emb_feats = emb.features
        if emb_feats:
            for feat in emb_feats:
                feature_hash_sizes[feat] = emb.num_embeddings
    feature_config.feature_hash_sizes = feature_hash_sizes

    if not model_args.prod_faithful:
        _apply_bench_shape_overrides(module_config, feature_config, model_args)
    _apply_seq_len_cap(module_config, model_args.seq_len_cap)
    if model_args.disable_seq_len_sampling:
        module_config.hstu_seq_len_sampling_configs = None

    # Static-shape mode (TPU/GSPMD path).
    if enable_padding:
        _apply_static_shape_mode(module_config, model_args, budgeted_seq_len_sampling)

    return module_config, feature_config


def _apply_seq_len_cap(
    module_config: BlueReelsVDDModuleConfig,
    cap: int,
) -> None:
    """Cap both HSTU stacks' sequence lengths. ``cap <= 0`` is a no-op."""
    if cap <= 0:
        return
    if module_config.hstu_l2_max_seq_len >= cap:
        raise ValueError(
            f"seq_len_cap={cap} is not above hstu_l2_max_seq_len="
            f"{module_config.hstu_l2_max_seq_len}: the L2 window would cover "
            f"every row's history, minus_k would be 0 everywhere and the L1 "
            f"stack would see empty sequences. Raise the cap above the L2 "
            f"window, or lower hstu_l2_max_seq_len deliberately."
        )
    logger.info(
        "[config] sequence-length cap %d: hstu_max_seq_len %d -> %d "
        "(hstu_l2_max_seq_len %d unchanged)",
        cap,
        module_config.hstu_max_seq_len,
        min(module_config.hstu_max_seq_len, cap),
        module_config.hstu_l2_max_seq_len,
    )
    module_config.hstu_max_seq_len = min(module_config.hstu_max_seq_len, cap)
    _sync_l2_max_seq_len(module_config)


def _apply_bench_shape_overrides(
    module_config: BlueReelsVDDModuleConfig,
    feature_config: BlueReelsVDDFeatureConfig,
    model_args: RankFMModelArgs,
) -> None:
    """Shrink the standalone config down to the CLI's bench shapes."""
    if len(module_config.stu_stack_configs) >= 1:
        _trim_stu_stack(module_config.stu_stack_configs[0], model_args.l1_num_layers)
    if len(module_config.stu_stack_configs) >= 2:
        _trim_stu_stack(module_config.stu_stack_configs[1], model_args.l2_num_layers)

    module_config.hstu_max_seq_len = model_args.max_uih_seq_len
    module_config.hstu_l2_max_seq_len = (
        model_args.l2_max_seq_len
        if model_args.l2_max_seq_len > 0
        else max(1, model_args.max_uih_seq_len // 4)
    )
    module_config.hstu_max_output_len = model_args.num_candidates
    _sync_l2_max_seq_len(module_config)

    # Sync feature config with the test constants so randgen produces
    # shapes that match the padding budgets and override expectations.
    feature_config.history_len = model_args.max_uih_seq_len
    feature_config.num_candidates_per_request = model_args.num_candidates


def _apply_static_shape_mode(
    module_config: BlueReelsVDDModuleConfig,
    model_args: RankFMModelArgs,
    budgeted_seq_len_sampling: bool = False,
) -> None:
    """Pin the padding budgets the TPU/GSPMD path needs.

    Two paths for the RO budget. By default it stays ``hstu_max_seq_len``:
    nothing bounds the per-chip token total, so the padding has to cover the
    UNSAMPLED worst case -- every row at the longest history the config
    admits, even though the sequence-length sampler truncates most rows far
    below it.

    With ``budgeted_seq_len_sampling`` the sampler enforces a total, so the
    budget can be what the sampler actually emits (see
    ``_sampler_derived_ro_budget``) rather than what it might. The gap is pure
    padding: the budget sizes a packed tensor and is a per-chip total, not a
    per-row cap.

    Outside prod-faithful mode both budgets come from the CLI.
    """
    module_config.static_shape_mode = True
    if model_args.prod_faithful:
        nro_budget = module_config.hstu_max_output_len
        ro_budget = (
            _sampler_derived_ro_budget(module_config)
            if budgeted_seq_len_sampling
            else module_config.hstu_max_seq_len
        )
    else:
        nro_budget = model_args.num_candidates
        ro_budget = model_args.seq_ro_per_sample_budget
    module_config.static_nro_per_sample_budget = nro_budget
    module_config.static_ro_per_sample_budgets = _build_static_ro_budgets(
        module_config, ro_budget
    )


def _sampler_derived_ro_budget(module_config: BlueReelsVDDModuleConfig) -> int:
    configs = module_config.hstu_seq_len_sampling_configs
    if configs is None:
        raise ValueError(
            "budgeted_seq_len_sampling needs a sequence-length sampler to "
            "derive the budget from, but this config ships none (is "
            "--disable_seq_len_sampling set?)."
        )
    _distribution_type, strategy = configs
    if not isinstance(strategy, SequenceLengthSamplingStrategy):
        raise TypeError(
            f"the budget is derived from alpha/beta, which only "
            f"SequenceLengthSamplingStrategy carries; this config ships "
            f"{type(strategy).__name__}."
        )
    alpha = float(strategy.alpha)
    beta = float(strategy.beta)
    max_seq_len = float(module_config.hstu_max_seq_len)

    sampled_len = beta * max_seq_len ** (alpha / 2.0)
    # alpha >= 2 makes the keep-whole probability >= 1: nothing is ever
    # sampled and the budget is just the full length.
    keep_whole_prob = min(1.0, max_seq_len ** (alpha - 2.0))
    budget = int(
        math.ceil(keep_whole_prob * max_seq_len + (1.0 - keep_whole_prob) * sampled_len)
    )
    logger.info(
        "[config] RO budget %d derived from the sampler at L=N: alpha=%.3g, "
        "beta=%.3g, N=%d -> P(keep whole)=%.3g, sampled length=%.0f. "
        "hstu_max_seq_len stays %d -- the budget sizes the packed sequence "
        "tensor as a per-chip total, so rows longer than it are fine and only "
        "an over-budget chip total is not.",
        budget,
        alpha,
        beta,
        int(max_seq_len),
        keep_whole_prob,
        sampled_len,
        module_config.hstu_max_seq_len,
    )
    return budget


def _trim_stu_stack(stack: STUStackConfig, num_layers: int) -> None:
    """Truncate a STU stack to ``num_layers``, complaining if it is shorter.

    Asking for more layers than the base config defines used to silently yield
    the shorter list, so a run requesting prod's 11 L2 layers actually trained
    10 (``_build_default_stu_stack_configs`` builds 8 L1 / 10 L2). Warn instead
    of raising: several manifests already pass 11 and have always run 10, so
    raising would break them while changing no numerics.
    """
    available = len(stack.stu_configs)
    if num_layers > available:
        logger.warning(
            "[config] STU stack %r has only %d layers but %d were requested; "
            "running with %d. The standalone udd_fm config caps L1 at 8 and L2 "
            "at 10 -- use --prod_faithful to get prod's real stack depths.",
            stack.name,
            available,
            num_layers,
            available,
        )
        return
    stack.stu_configs = stack.stu_configs[:num_layers]


def _sync_l2_max_seq_len(module_config: BlueReelsVDDModuleConfig) -> None:
    """Mirror ``hstu_l2_max_seq_len`` into the L2 stack's constructor kwargs.

    ``SingleSeqL2HSTU`` reads the value from ``additional_kwargs`` while
    ``MoTPreprocessor`` reads it off the module config, so leaving the kwarg at
    its hardcoded 512 makes the two disagree whenever the config field moves.
    """
    for stack in module_config.stu_stack_configs:
        if "l2_max_seq_len" in stack.additional_kwargs:
            stack.additional_kwargs["l2_max_seq_len"] = (
                module_config.hstu_l2_max_seq_len
            )


def _build_static_ro_budgets(
    module_config: BlueReelsVDDModuleConfig, seq_budget: int
) -> Dict[str, int]:
    """Build per-key per-sample RO token budgets.

    Sequential (history) RO keys all share ``seq_budget`` (the
    preprocessor's per-key split assumes uniform per-key lengths within
    the seq group). Contextual RO keys take their per-feature max
    length. The timestamp key takes 1 (matches
    ``jagged_to_padded_dense(max_lengths=[1])`` in
    ``FeaturePreprocessorWrapper``).
    """
    seq_ro_keys = {ro for ro, _nro in module_config.merge_item_uih_feature_mapping}
    budgets: Dict[str, int] = {k: seq_budget for k in seq_ro_keys}
    for k, max_len in module_config.contextual_feature_to_max_length:
        budgets[k] = int(max_len)
    budgets[module_config.hstu_sparse_timestamp_feature_name] = 1
    return budgets


# ---------------------------------------------------------------------------
# mvai_cloud RecModelFamilyConfig builder
# ---------------------------------------------------------------------------


def build_model_family_config(
    trainer_cfg: RankFMTrainerConfig,
    model_args: RankFMModelArgs,
) -> RecModelFamilyConfig:
    """CLI args -> a RecModelFamilyConfig carrying the mutated module/feature config.

    Starts from the hammer udd_fm ``ConfigProvider().create_config()`` (so the
    optimizer_config, qcomms_config, and embedding_tables defaults match the
    canonical udd_fm RecTrainer entrypoint), then overwrites module_config and
    feature_config with the RankFMModelArgs-mutated pair produced by the shared
    ``build_module_config``. The mutated module_config is what
    BlueReelsVDDModelFamily.create_vanilla_train_module reads to build the
    byte-identical BlueReelsVDDTrainV3, and the mutated feature_config is what
    the (reused) rankfm data path consumes.
    """
    base_config: RecModelFamilyConfig = ConfigProvider().create_config()
    # Needs the CPU embedding backend's sparse-warmup support (cpu.py
    # _build_sparse_warmup); without it SparseOptimizerRuntime rejects a
    # schedule outright.
    base_config.optimizer_config = build_optimizer_config()

    module_config: BlueReelsVDDModuleConfig
    feature_config: BlueReelsVDDFeatureConfig
    module_config, feature_config = build_module_config(
        model_args,
        enable_padding=trainer_cfg.enable_padding,
        budgeted_seq_len_sampling=trainer_cfg.budgeted_seq_len_sampling,
    )

    base_config.module_config = module_config
    base_config.feature_config = feature_config
    # HSTU owns its own EmbeddingCollection (recir backend handles sharding),
    # so leave the legacy single-sharder path (use_distributed_sharding=False,
    # embedding_tables={}) exactly as ConfigProvider().create_config() sets it.
    return base_config


# ---------------------------------------------------------------------------
# mvai_cloud RecTrainerConfig builder
# ---------------------------------------------------------------------------


def build_trainer_config(
    trainer_cfg: RankFMTrainerConfig,
) -> RecTrainerConfig:
    """CLI trainer knobs -> a RecTrainerConfig for the TPU/recir path.

    Three choices worth knowing:
      * ``use_gpu=False`` puts RecTrainer on its CPU/gloo branch; recir still
        places dense compute on TPU through jax.devices().
      * ``data_source_config`` is Random purely so the QPS metric has a positive
        batch_size. Real data comes from ``create_train_datapipe``.
      * ``max_steps`` is enforced by the data pipe, which stops the trainer by
        raising StopIteration.
    """
    data_source_config = RandomDataSourceConfig(
        batch_size=trainer_cfg.local_batch_size,
        num_batches=trainer_cfg.max_steps,
    )

    # A load path enables restore even when periodic saves are disabled. This
    # supports a load-only continuation without uploading another checkpoint at
    # the end of the run.
    # At runtime CheckpointConfig is the OSS mvai_cloud replica (has
    # save_every_n_steps / keep_last_n); pyre resolves the canonical
    # //minimal_viable_ai variant which lacks those fields, hence the ignores.
    checkpoint_config = (
        CheckpointConfig(
            save_every_n_steps=trainer_cfg.save_every_n_steps,  # pyre-ignore[28]
            keep_last_n=trainer_cfg.keep_last_n,  # pyre-ignore[28]
            async_save=trainer_cfg.async_checkpoint,  # pyre-ignore[28]
        )
        if trainer_cfg.save_every_n_steps > 0
        or bool(os.environ.get("MVAI_CHECKPOINT_LOAD_DIR"))
        else None
    )

    return RecTrainerConfig(
        data_source_config=data_source_config,
        system_config=SystemConfig(use_gpu=False, dist_backend="gloo"),
        metrics_config=MetricsConfig(
            report_interval=trainer_cfg.metrics_report_interval,
            tensorboard_logger=True,
        ),
        model_config_module=("hammer.v3.models.udd_fm_v3.udd_fm_config"),
        checkpoint_config=checkpoint_config,
    )
