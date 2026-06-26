#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Configurable RankFM (xsurface V3) model and training parameters for TPU experiments.

The xsurface model has a heavy production-default config in
``hammer.v3.models.udd_fm_v3.udd_fm_config.BlueReelsVDDModuleConfig``,
hardcoded hash tables and feature lists). For TPU experiments we only
expose a small set of configs split into two clean buckets:

* :class:`RankFMTrainerConfig` -- pure trainer knobs (batching, optimizer,
  runtime mode toggles, profiling). Nothing here affects the model identity
  or FX graph topology;
* :class:`RankFMModelArgs` -- overrides that mutate ``BlueReelsVDDModuleConfig``
  (architecture depth, sequence shapes, hash caps).
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from hammer.v3.models.udd_fm_v3.batch_loader import BlueReelsVDDFeatureConfig
from hammer.v3.models.udd_fm_v3.udd_fm_config import (
    BlueReelsVDDModuleConfig,
    ConfigProvider,
)
from minimal_viable_ai.core.model_family_api.configs import RecModelFamilyConfig
from minimal_viable_ai.core.model_family_api.optimizer_configs import OptimConfig
from torchrec.distributed.embedding_types import OptimType
from torchrec.optim.clipping import GradientClipping
from torchrec.optim.warmup import WarmupPolicy, WarmupStage


# ---------------------------------------------------------------------------
# CLI config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RankFMTrainerConfig:
    """Pure trainer knobs: batching, optimizer, runtime mode, profiling.

    Nothing here changes the model identity or FX graph topology.
    """

    # Data
    local_batch_size: int = 32

    # Training
    max_steps: int = 20
    embedding_lr: float = 0.1
    dense_optimizer_config: OptimConfig = field(
        default_factory=lambda: OptimConfig(
            optim_type=OptimType.ADAMW,
            lr=0.0004,
            eps=0.00001,
            beta1=0.9,
            beta2=0.998,
            weight_decay=0.0001,
            max_gradient=1.0,
            gradient_clipping=GradientClipping.VALUE,
            lr_warmup_schedule=[
                WarmupStage(
                    policy=WarmupPolicy.LINEAR,
                    max_iters=37500,
                    value=0.1,
                    lr_scale=1.0,
                )
            ],
        )
    )

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
    # prepare ahead of the training loop so file load / zstd decompress / pad
    # overlap the device train_step. 0 disables prefetch (synchronous load).
    data_prefetch_buffer: int = 2

    # Embedding backend: "cpu" (torchrec DMP) | "tpu" (JAX/SparseCore).
    # Validated by ``build_embedding_backend`` in ``embedding/__init__.py``.
    embedding_backend: str = "cpu"

    # In-graph embedding sharding thresholds (per-table row counts), used only by
    # the "in_graph" backend. Each table's mode:
    #   rows >= ringshift_min_rows                 -> ROWWISE (ring-shift)
    #   fsdp_min_rows <= rows < ringshift_min_rows -> FSDP    (F.embedding)
    #   rows < fsdp_min_rows                       -> DP       (replicated)
    # fsdp_min_rows < 0 means "= ringshift_min_rows" (empty FSDP band -> below
    # ring-shift is DP).
    ringshift_min_rows: int = 4096
    fsdp_min_rows: int = -1

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

    # Profiling
    profile_start_step: int = -1
    profile_end_step: int = -1
    profile_dir: str = "/tmp/tpu_profile"
    enable_memory_log: bool = False

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

    # Embedding sizing (cap large hash tables for memory budget on TPU)
    hash_size_cap: int = 1_000_000

    # HSTU stack sizing
    l1_num_layers: int = 3
    l2_num_layers: int = 11

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


# ---------------------------------------------------------------------------
# Module-config builder (single source of truth for CLI -> module_config)
# ---------------------------------------------------------------------------


def build_module_config(
    model_args: RankFMModelArgs,
    *,
    enable_padding: bool,
) -> Tuple[BlueReelsVDDModuleConfig, BlueReelsVDDFeatureConfig]:
    """CLI args -> fully configured (module_config, feature_config)."""
    config: RecModelFamilyConfig = ConfigProvider().create_config()
    module_config: BlueReelsVDDModuleConfig = config.module_config  # pyre-ignore[9]
    feature_config: BlueReelsVDDFeatureConfig = config.feature_config  # pyre-ignore[9]

    # Cap embedding-table hash sizes.
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

    # STU stack trim.
    if len(module_config.stu_stack_configs) >= 1:
        module_config.stu_stack_configs[
            0
        ].stu_configs = module_config.stu_stack_configs[0].stu_configs[
            : model_args.l1_num_layers
        ]
    if len(module_config.stu_stack_configs) >= 2:
        module_config.stu_stack_configs[
            1
        ].stu_configs = module_config.stu_stack_configs[1].stu_configs[
            : model_args.l2_num_layers
        ]

    # Sequence shapes.
    module_config.hstu_max_seq_len = model_args.max_uih_seq_len
    module_config.hstu_l2_max_seq_len = (
        model_args.l2_max_seq_len
        if model_args.l2_max_seq_len > 0
        else max(1, model_args.max_uih_seq_len // 4)
    )
    module_config.hstu_max_output_len = model_args.num_candidates

    if model_args.disable_seq_len_sampling:
        module_config.hstu_seq_len_sampling_configs = None

    # Static-shape mode (TPU/GSPMD path).
    if enable_padding:
        module_config.static_shape_mode = True
        module_config.static_nro_per_sample_budget = model_args.num_candidates
        module_config.static_ro_per_sample_budgets = _build_static_ro_budgets(
            module_config, model_args.seq_ro_per_sample_budget
        )

    # Sync feature config with the test constants so randgen produces
    # shapes that match the padding budgets and override expectations.
    feature_config.history_len = model_args.max_uih_seq_len
    feature_config.num_candidates_per_request = model_args.num_candidates

    return module_config, feature_config


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
# CLI parsing -- launchers (YAML / shell) pass --field value to override the
# dataclass defaults above.  Primitive (int/float/str/bool) fields are
# auto-registered; non-primitive fields (e.g. Dict budgets) are not
# CLI-exposed and must be set in code.
# ---------------------------------------------------------------------------


def _add_dataclass_args(
    parser: argparse.ArgumentParser,
    cls: Any,
    skip: frozenset[str] = frozenset(),
) -> None:
    """Register `--<field>` CLI args for each primitive field of *cls*.

    Booleans get a paired ``--<field>`` / ``--no_<field>`` toggle so launchers
    can flip either direction regardless of the dataclass default. Fields named
    in ``skip`` are not CLI-exposed (e.g. optimizer LR knobs that live only in
    the dataclass / config objects).
    """
    for f in dataclasses.fields(cls):
        if f.name in skip:
            continue
        default = f.default
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


def parse_cli() -> Tuple[RankFMTrainerConfig, RankFMModelArgs]:
    """Parse argv, returning dataclass instances with any --flag overrides applied."""
    parser = argparse.ArgumentParser()
    # Optimizer LR knobs are config-only (dense_optimizer_config / embedding_lr),
    # not CLI-exposed.
    _add_dataclass_args(parser, RankFMTrainerConfig, skip=frozenset({"embedding_lr"}))
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
    return trainer_cfg, model_args
