# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Configurable DLRMv3 model and training parameters for TPU experiments."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from generative_recommenders.modules.dlrm_hstu import DlrmHSTUConfig
    from generative_recommenders.modules.multitask_module import TaskConfig


@dataclass
class DLRMv3Config:
    # Embedding
    hash_size: int = 10_000_000
    embedding_dim: int = 256
    num_tables: int = 3

    # Dense / HSTU
    hidden_dim: int = 256
    transducer_dim: int = 512
    attn_dim: int = 256
    num_layers: int = 12
    num_heads: int = 8

    # Data
    local_batch_size: int = 128
    uih_len: int = 50
    num_candidates: int = 4
    max_seq_len: int = 256

    # Training
    max_steps: int = 20
    dense_lr: float = 0.01
    embedding_lr: float = 0.1

    # Embedding backend
    embedding_backend: str = "tpu"  # "cpu" (torchrec DMP) | "tpu" (JAX/SparseCore)

    # Profiling
    profile_start_step: int = -1
    profile_end_step: int = -1
    profile_dir: str = "/tmp/tpu_profile"
    enable_memory_log: bool = False

    @staticmethod
    def from_args() -> DLRMv3Config:
        p = argparse.ArgumentParser(description="DLRMv3 on TPU")
        cfg = DLRMv3Config()

        # Embedding
        p.add_argument("--hash_size", type=int, default=cfg.hash_size)
        p.add_argument("--embedding_dim", type=int, default=cfg.embedding_dim)
        p.add_argument("--num_tables", type=int, default=cfg.num_tables)

        # Dense
        p.add_argument("--hidden_dim", type=int, default=cfg.hidden_dim)
        p.add_argument("--transducer_dim", type=int, default=cfg.transducer_dim)
        p.add_argument("--attn_dim", type=int, default=cfg.attn_dim)
        p.add_argument("--num_layers", type=int, default=cfg.num_layers)
        p.add_argument("--num_heads", type=int, default=cfg.num_heads)

        # Data
        p.add_argument("--local_batch_size", type=int, default=cfg.local_batch_size)
        p.add_argument("--uih_len", type=int, default=cfg.uih_len)
        p.add_argument("--num_candidates", type=int, default=cfg.num_candidates)

        # Training
        p.add_argument("--max_steps", type=int, default=cfg.max_steps)
        p.add_argument("--dense_lr", type=float, default=cfg.dense_lr)
        p.add_argument("--embedding_lr", type=float, default=cfg.embedding_lr)

        # Profiling
        p.add_argument("--profile_start_step", type=int, default=cfg.profile_start_step)
        p.add_argument("--profile_end_step", type=int, default=cfg.profile_end_step)
        p.add_argument("--profile_dir", type=str, default=cfg.profile_dir)
        p.add_argument("--enable_memory_log", action="store_true")
        p.add_argument(
            "--embedding_backend",
            type=str,
            default=cfg.embedding_backend,
            choices=["cpu", "tpu"],
            help="'cpu': torchrec DMP on CPU; 'tpu': JAX/SparseCore (default)",
        )

        # Preset: small (for smoke testing)
        p.add_argument(
            "--preset",
            type=str,
            default=None,
            choices=["small", "medium", "large"],
        )

        args = p.parse_args()

        if args.preset == "small":
            args.hash_size = 1000
            args.embedding_dim = 64
            args.hidden_dim = 32
            args.transducer_dim = 64
            args.attn_dim = 32
            args.num_layers = 2
            args.num_heads = 2
            args.local_batch_size = 4
            args.uih_len = 50
            args.num_candidates = 4
        elif args.preset == "medium":
            args.hash_size = 100_000
            args.embedding_dim = 128
            args.hidden_dim = 128
            args.transducer_dim = 256
            args.attn_dim = 128
            args.num_layers = 6
            args.num_heads = 4
            args.local_batch_size = 32
            args.uih_len = 100
            args.num_candidates = 8

        return DLRMv3Config(
            hash_size=args.hash_size,
            embedding_dim=args.embedding_dim,
            num_tables=args.num_tables,
            hidden_dim=args.hidden_dim,
            transducer_dim=args.transducer_dim,
            attn_dim=args.attn_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            local_batch_size=args.local_batch_size,
            uih_len=args.uih_len,
            num_candidates=args.num_candidates,
            max_seq_len=args.uih_len + args.num_candidates + 16,
            max_steps=args.max_steps,
            dense_lr=args.dense_lr,
            embedding_lr=args.embedding_lr,
            embedding_backend=args.embedding_backend,
            profile_start_step=args.profile_start_step,
            profile_end_step=args.profile_end_step,
            profile_dir=args.profile_dir,
            enable_memory_log=args.enable_memory_log,
        )


def create_hstu_config(cfg: DLRMv3Config) -> DlrmHSTUConfig:
    """Create DlrmHSTUConfig without depending on internal get_hstu_configs.

    Replicates the "debug" gin config from generative_recommenders with
    all necessary fields for DLRMv3 + HSTU.
    """
    from generative_recommenders.modules.dlrm_hstu import (
        DlrmHSTUConfig as _DlrmHSTUConfig,
    )

    return _DlrmHSTUConfig(
        # Embedding
        hstu_embedding_table_dim=cfg.embedding_dim,
        # Preprocessor
        hstu_preprocessor_hidden_dim=cfg.hidden_dim,
        # Transducer (HSTU attention)
        hstu_transducer_embedding_dim=cfg.transducer_dim,
        hstu_attn_linear_dim=cfg.attn_dim,
        hstu_attn_qk_dim=cfg.attn_dim,
        hstu_attn_num_layers=cfg.num_layers,
        hstu_num_heads=cfg.num_heads,
        # Sequence lengths
        max_seq_len=cfg.max_seq_len,
        max_num_candidates=cfg.num_candidates,
        # Feature names (matching the debug gin config)
        hstu_uih_feature_names=[
            "viewer_id",
            "dummy_contexual",
            "uih_post_id",
            "uih_owner_id",
        ],
        hstu_candidate_feature_names=["item_post_id", "item_owner_id"],
        user_embedding_feature_names=[
            "uih_post_id",
            "uih_owner_id",
            "viewer_id",
            "dummy_contexual",
        ],
        item_embedding_feature_names=["item_post_id", "item_owner_id"],
        uih_post_id_feature_name="uih_post_id",
        uih_action_time_feature_name="uih_action_time",
        candidates_querytime_feature_name="candidates_querytime",
        candidates_weight_feature_name="candidates_weight",
        candidates_watchtime_feature_name="candidates_watchtime",
        merge_uih_candidate_feature_mapping=[
            ("uih_post_id", "item_post_id"),
            ("uih_owner_id", "item_owner_id"),
        ],
        contextual_feature_to_max_length={
            "viewer_id": 1,
            "dummy_contexual": 1,
        },
        contextual_feature_to_min_uih_length={
            "viewer_id": 10,
            "dummy_contexual": 10,
        },
        # Dropout disabled for XLA tracing compatibility
        hstu_input_dropout_ratio=0.0,
        hstu_linear_dropout_rate=0.0,
        # Default multitask config (required by DefaultMultitaskModule)
        multitask_configs=[
            _default_task_config(),
        ],
    )


def _default_task_config() -> TaskConfig:
    from generative_recommenders.modules.multitask_module import (
        MultitaskTaskType,
        TaskConfig as _TaskConfig,
    )

    return _TaskConfig(
        task_name="click",
        task_weight=1,
        task_type=MultitaskTaskType.BINARY_CLASSIFICATION,
    )
