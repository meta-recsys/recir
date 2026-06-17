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
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from hammer.v3.models.udd_fm_v3.batch_loader import BlueReelsVDDFeatureConfig
from hammer.v3.models.udd_fm_v3.udd_fm_config import (
    BlueReelsVDDModuleConfig,
    ConfigProvider,
)
from minimal_viable_ai.core.model_family_api.configs import RecModelFamilyConfig


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
    dense_lr: float = 0.01
    embedding_lr: float = 0.1

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

    # Runtime mode: enables PaddingRegionTransform AND flips
    # static_shape_mode on the model config (since the two must agree).
    enable_padding: bool = True

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
    module_config.hstu_l2_max_seq_len = max(1, model_args.max_uih_seq_len // 4)
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


def _add_dataclass_args(parser: argparse.ArgumentParser, cls: Any) -> None:
    """Register `--<field>` CLI args for each primitive field of *cls*.

    Booleans get a paired ``--<field>`` / ``--no_<field>`` toggle so launchers
    can flip either direction regardless of the dataclass default.
    """
    for f in dataclasses.fields(cls):
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
    return trainer_cfg, model_args
