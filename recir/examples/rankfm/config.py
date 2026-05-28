# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Configurable RankFM (xsurface V3) model and training parameters for TPU experiments.

The xsurface model has a heavy production-default config in
``hammer.v3.models.udd_fm_v3.udd_fm_config.BlueReelsVDDModuleConfig`` (~700
lines, 68 tasks, hardcoded hash tables and feature lists). For TPU experiments
we only expose a small set of CLI knobs and let an auto-helper apply them on
top of the production defaults.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hammer.v3.models.udd_fm_v3.batch_loader import BlueReelsVDDFeatureConfig
    from hammer.v3.models.udd_fm_v3.udd_fm_config import BlueReelsVDDModuleConfig
    from minimal_viable_ai.core.model_family_api.configs import RecModelFamilyConfig


@dataclass
class RankFMConfig:
    """CLI-tunable knobs. Anything not listed here uses the production default."""

    # Embedding sizing (cap large hash tables for memory budget on TPU)
    hash_size_cap: int = 1_000_000

    # HSTU stack sizing (trim the L1 / L2 stacks to keep tracing fast)
    l1_num_layers: int = 3
    l2_num_layers: int = 11

    # Sequence shapes
    max_uih_seq_len: int = 32
    num_candidates: int = 8

    # Data
    local_batch_size: int = 4

    # Training
    max_steps: int = 20
    dense_lr: float = 0.01
    embedding_lr: float = 0.1

    # Real data replay (synthetic data is used when gcs_data_path is empty)
    gcs_data_path: str = ""
    data_file_template: str = ""
    data_replicate_across_ranks: bool = False

    # Embedding backend: "cpu" (torchrec DMP) | "tpu" (JAX/SparseCore)
    embedding_backend: str = "cpu"

    disable_seq_len_sampling: bool = False

    # Enable PaddingRegionTransform for dynamic-shape batches. When True,
    # jagged inputs are padded to static budgets at each step, allowing
    # different batches with varying jagged lengths.
    enable_padding: bool = True

    # Profiling
    profile_start_step: int = -1
    profile_end_step: int = -1
    profile_dir: str = "/tmp/tpu_profile"
    enable_memory_log: bool = False

    @staticmethod
    def from_args() -> RankFMConfig:
        p = argparse.ArgumentParser(description="RankFM (xsurface V3) on TPU")
        cfg = RankFMConfig()

        p.add_argument("--hash_size_cap", type=int, default=cfg.hash_size_cap)
        p.add_argument("--l1_num_layers", type=int, default=cfg.l1_num_layers)
        p.add_argument("--l2_num_layers", type=int, default=cfg.l2_num_layers)
        p.add_argument("--max_uih_seq_len", type=int, default=cfg.max_uih_seq_len)
        p.add_argument("--num_candidates", type=int, default=cfg.num_candidates)
        p.add_argument("--local_batch_size", type=int, default=cfg.local_batch_size)
        p.add_argument("--max_steps", type=int, default=cfg.max_steps)
        p.add_argument("--dense_lr", type=float, default=cfg.dense_lr)
        p.add_argument("--embedding_lr", type=float, default=cfg.embedding_lr)
        p.add_argument("--gcs_data_path", type=str, default=cfg.gcs_data_path)
        p.add_argument("--data_file_template", type=str, default=cfg.data_file_template)
        p.add_argument(
            "--data_replicate_across_ranks",
            action="store_true",
            default=cfg.data_replicate_across_ranks,
        )
        p.add_argument(
            "--keep_seq_len_sampling",
            dest="disable_seq_len_sampling",
            action="store_false",
            default=cfg.disable_seq_len_sampling,
        )
        p.add_argument(
            "--enable_padding", action="store_true", default=cfg.enable_padding
        )
        p.add_argument("--disable_padding", dest="enable_padding", action="store_false")
        p.add_argument("--profile_start_step", type=int, default=cfg.profile_start_step)
        p.add_argument("--profile_end_step", type=int, default=cfg.profile_end_step)
        p.add_argument("--profile_dir", type=str, default=cfg.profile_dir)
        p.add_argument(
            "--enable_memory_log", action="store_true", default=cfg.enable_memory_log
        )
        p.add_argument(
            "--embedding_backend",
            type=str,
            default=cfg.embedding_backend,
            choices=["cpu", "tpu"],
            help="'cpu': torchrec DMP on CPU; 'tpu': JAX/SparseCore",
        )

        p.add_argument(
            "--preset",
            type=str,
            default=None,
            choices=["small", "medium", "large"],
        )

        args = p.parse_args()

        presets = {
            "small": {
                "hash_size_cap": 1_000,
                "l1_num_layers": 1,
                "l2_num_layers": 1,
                "max_uih_seq_len": 32,
                "num_candidates": 4,
                "local_batch_size": 4,
            },
            "medium": {
                "hash_size_cap": 100_000,
                "l1_num_layers": 2,
                "l2_num_layers": 4,
                "max_uih_seq_len": 64,
                "num_candidates": 8,
                "local_batch_size": 16,
            },
        }
        if args.preset in presets:
            for key, value in presets[args.preset].items():
                if p.get_default(key) == getattr(args, key):
                    setattr(args, key, value)

        return RankFMConfig(
            hash_size_cap=args.hash_size_cap,
            l1_num_layers=args.l1_num_layers,
            l2_num_layers=args.l2_num_layers,
            max_uih_seq_len=args.max_uih_seq_len,
            num_candidates=args.num_candidates,
            local_batch_size=args.local_batch_size,
            max_steps=args.max_steps,
            dense_lr=args.dense_lr,
            embedding_lr=args.embedding_lr,
            gcs_data_path=args.gcs_data_path,
            data_file_template=args.data_file_template,
            data_replicate_across_ranks=args.data_replicate_across_ranks,
            embedding_backend=args.embedding_backend,
            disable_seq_len_sampling=args.disable_seq_len_sampling,
            enable_padding=args.enable_padding,
            profile_start_step=args.profile_start_step,
            profile_end_step=args.profile_end_step,
            profile_dir=args.profile_dir,
            enable_memory_log=args.enable_memory_log,
        )


def create_model_family_config(cfg: RankFMConfig) -> RecModelFamilyConfig:
    """Auto helper: produce a RecModelFamilyConfig adapted from production defaults.

    Mirrors the pattern in ``recir.tests.xla.xsurface_test_utils.TestConfigProvider``:
    construct via ``ConfigProvider().create_config()`` (which carries all 68
    tasks, feature mappings, and embedding tables from production), then mutate
    only the fields the CLI exposes. This keeps the model identical to the
    fbcode-side training while letting us shrink hash sizes and STU depth for
    TPU experiments.
    """
    from hammer.v3.models.udd_fm_v3.udd_fm_config import (
        BlueReelsVDDModuleConfig as _ModuleConfig,
        ConfigProvider,
    )

    config = ConfigProvider().create_config()
    module_config: _ModuleConfig = config.module_config  # pyre-ignore[9]

    # Cap embedding-table hash sizes.
    module_config.embedding_table_configs = {  # pyre-ignore[16]
        name: (min(num_rows, cfg.hash_size_cap), dim)
        for name, (num_rows, dim) in module_config.embedding_table_configs.items()
    }

    # Trim STU stacks. The default helper builds two stacks with
    # (l1_num_layers=3, l2_num_layers=11) - clip to user-requested depths.
    if len(module_config.stu_stack_configs) >= 1:
        module_config.stu_stack_configs[
            0
        ].stu_configs = module_config.stu_stack_configs[0].stu_configs[
            : cfg.l1_num_layers
        ]
    if len(module_config.stu_stack_configs) >= 2:
        module_config.stu_stack_configs[
            1
        ].stu_configs = module_config.stu_stack_configs[1].stu_configs[
            : cfg.l2_num_layers
        ]

    if cfg.disable_seq_len_sampling:
        # The field is hstu_seq_len_sampling_configs (with the hstu_ prefix);
        # see TestConfigProvider for why this matters for GSPMD.
        module_config.hstu_seq_len_sampling_configs = None

    return config


def get_feature_config(
    config: RecModelFamilyConfig,
) -> BlueReelsVDDFeatureConfig:
    return config.feature_config  # pyre-ignore[7]


def get_module_config(
    config: RecModelFamilyConfig,
) -> BlueReelsVDDModuleConfig:
    return config.module_config  # pyre-ignore[7]
