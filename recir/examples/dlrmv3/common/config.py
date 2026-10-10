# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Command-line configuration shared by the DLRMv3 GPU and TPU trainers.

Replaces the generative-recommenders gin files: the model architecture comes
from ``generative_recommenders.dlrm_v3.configs`` keyed by ``--dataset``, and
every training knob is a flag.
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from generative_recommenders.dlrm_v3.configs import (
    get_embedding_table_config,
    get_hstu_configs,
)
from generative_recommenders.modules.dlrm_hstu import DlrmHSTUConfig
from recir.examples.dlrmv3.common.model import FeatureLayout
from torchrec.distributed.embedding_types import OptimType
from torchrec.modules.embedding_configs import DataType, EmbeddingConfig
from torchrec.optim.clipping import GradientClipping
from torchrec.optim.warmup import WarmupPolicy, WarmupStage


@dataclass
class OptimizerConfig:
    """One optimizer's settings (a ``recir.training.optim.OptimConfig``)."""

    optim_type: OptimType
    lr: float
    eps: float = 1e-8
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.0
    gradient_clipping: GradientClipping = GradientClipping.NONE
    max_gradient: float = 1.0
    lr_warmup_schedule: List[WarmupStage] = field(default_factory=list)


@dataclass
class OptimizerGroup:
    default: OptimizerConfig


@dataclass
class OptimizerConfigs:
    """Dense + sparse settings (a ``recir.training.setup.RecOptimizerConfig``)."""

    dense: OptimizerGroup
    sparse: OptimizerGroup


@dataclass
class DLRMv3TrainConfig:
    dataset: str = "debug"
    data_root: str = ""
    # Max UIH events per row (most recent kept) and candidates per row; 0 means
    # "derive from the dataset's HSTU config". ``uih_budget`` is the average UIH
    # tokens per real row each block reserves (default ``uih_len``: nothing is
    # ever trimmed); smaller budgets need fewer dummy rows. See FeatureLayout.
    uih_len: int = 0
    num_candidates: int = 0
    uih_budget: int = 0
    batch_size: int = 16
    num_workers: int = 2
    train_split: float = 0.75
    num_epochs: int = 1
    num_train_batches: Optional[int] = None
    num_eval_batches: Optional[int] = None
    eval_every: int = 0
    log_every: int = 10
    dense_optim: str = "adam"
    dense_lr: float = 1e-3
    dense_betas: List[float] = field(default_factory=lambda: [0.95, 0.999])
    sparse_optim: str = "rowwise_adagrad"
    sparse_lr: float = 1e-3
    eps: float = 1e-8
    warmup_iters: int = 0
    ckpt_dir: str = ""
    ckpt_every: int = 0
    resume_from: str = ""
    tb_dir: str = ""
    seed: int = 0
    # Small-scale overrides for smoke tests (0 keeps the dataset config).
    embedding_dim: int = 0
    num_layers: int = 0
    hash_size: int = 0

    def hstu_config(self) -> DlrmHSTUConfig:
        cfg = get_hstu_configs(self.dataset)
        if self.embedding_dim:
            cfg.hstu_embedding_table_dim = self.embedding_dim
        if self.num_layers:
            cfg.hstu_attn_num_layers = self.num_layers
        return cfg

    def embedding_tables(self) -> Dict[str, EmbeddingConfig]:
        # Tables train in fp32, like the dense model consuming their lookups
        # (the dataset configs declare fp16, an inference storage choice).
        return {
            name: dataclasses.replace(
                t,
                embedding_dim=self.embedding_dim or t.embedding_dim,
                num_embeddings=self.hash_size or t.num_embeddings,
                data_type=DataType.FP32,
            )
            for name, t in get_embedding_table_config(self.dataset).items()
        }

    def resolved_uih_len(self, hstu: DlrmHSTUConfig) -> int:
        if self.uih_len:
            return self.uih_len
        contextual = sum(dict(hstu.contextual_feature_to_max_length or {}).values())
        return hstu.max_seq_len - hstu.max_num_candidates - contextual

    def resolved_num_candidates(self, hstu: DlrmHSTUConfig) -> int:
        return self.num_candidates or hstu.max_num_candidates

    def layout(self, hstu: DlrmHSTUConfig) -> FeatureLayout:
        return FeatureLayout.from_config(
            hstu,
            self.resolved_uih_len(hstu),
            self.resolved_num_candidates(hstu),
            self.uih_budget,
        )

    def _warmup(self) -> List[WarmupStage]:
        if not self.warmup_iters:
            return []
        return [
            # Start at 1/warmup_iters of the base LR, not 0: the tensor-LR Adam
            # (capturable=True) divides by the LR, so a zero LR gives NaN.
            WarmupStage(
                policy=WarmupPolicy.LINEAR,
                max_iters=self.warmup_iters,
                value=1.0 / self.warmup_iters,
            )
        ]

    def dense_optimizer(self) -> OptimizerConfig:
        return OptimizerConfig(
            optim_type=OptimType[self.dense_optim.upper()],
            lr=self.dense_lr,
            eps=self.eps,
            beta1=self.dense_betas[0],
            beta2=self.dense_betas[1],
            lr_warmup_schedule=self._warmup(),
        )

    def sparse_optimizer(self) -> OptimizerConfig:
        return OptimizerConfig(
            optim_type=OptimType[self.sparse_optim.upper()],
            lr=self.sparse_lr,
            eps=self.eps,
            lr_warmup_schedule=self._warmup(),
        )

    def optimizer_configs(self) -> OptimizerConfigs:
        return OptimizerConfigs(
            dense=OptimizerGroup(self.dense_optimizer()),
            sparse=OptimizerGroup(self.sparse_optimizer()),
        )


def add_config_args(parser: argparse.ArgumentParser) -> None:
    for f in dataclasses.fields(DLRMv3TrainConfig):
        default = (
            f.default_factory()
            if f.default_factory is not dataclasses.MISSING
            else f.default
        )
        flag = f"--{f.name}"
        if isinstance(default, list):
            parser.add_argument(flag, type=float, nargs="+", default=default)
        elif default is None:
            parser.add_argument(flag, type=int, default=None)
        else:
            parser.add_argument(flag, type=type(default), default=default)


def config_from_args(args: argparse.Namespace) -> DLRMv3TrainConfig:
    names = {f.name for f in dataclasses.fields(DLRMv3TrainConfig)}
    return DLRMv3TrainConfig(**{k: v for k, v in vars(args).items() if k in names})


def parse_config(
    argv: Optional[Sequence[str]] = None,
    extra: Optional[argparse.ArgumentParser] = None,
) -> tuple[DLRMv3TrainConfig, argparse.Namespace]:
    parser = extra or argparse.ArgumentParser()
    add_config_args(parser)
    args = parser.parse_args(argv)
    return config_from_args(args), args
