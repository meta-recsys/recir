#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Train the TPU RankFM example configuration with CUDA RecTrainer."""

from __future__ import annotations

import argparse
import logging
import os
from typing import cast

import torch
from hammer.v3.models.udd_fm_v3.udd_fm_config import BlueReelsVDDModuleConfig
from minimal_viable_ai.core.dataloader.dataloader_configs import RandomDataSourceConfig
from minimal_viable_ai.core.ranking_common.metrics.ctr import FixedStepCTR
from minimal_viable_ai.core.ranking_common.metrics.mse import FixedStepMSE
from minimal_viable_ai.core.ranking_common.metrics.ne import FixedStepNE
from minimal_viable_ai.core.ranking_common.train_pipeline import RankingMetricDef
from minimal_viable_ai.core.trainer.job_resolver import JobResolver
from minimal_viable_ai.core.trainer.trainer import RecTrainer
from minimal_viable_ai.core.trainer.trainer_config_def import (
    MetricsConfig,
    RecTrainerConfig,
    SystemConfig,
)
from minimal_viable_ai.models.udd_fm.rankfm_parity import (
    build_cls_reg_partition,
    build_gpu_model_config,
    RankFMGPUModelFamily,
    RankFMGPUTrainPipelineProvider,
)
from minimal_viable_ai.models.udd_fm.rankfm_parity_utils import (
    rank_data_file,
    validate_batch_partition,
    validate_world_size,
)

from .config import (
    configs_from_namespace,
    create_cli_parser,
    RankFMModelArgs,
    RankFMTrainerConfig,
)

logger = logging.getLogger(__name__)


def _create_parser() -> argparse.ArgumentParser:
    parser = create_cli_parser(description=__doc__)
    parser.set_defaults(
        local_batch_size=8,
        num_candidates=32,
        gcs_data_path="/mnt/barnfs",
        data_file_template="rank_{rank}/batch_0/{i}.pt.zst",
        data_replicate_across_ranks=True,
        enable_padding=False,
    )
    parser.add_argument(
        "--data-path",
        dest="gcs_data_path",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--data-file-template",
        dest="data_file_template",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--local-batch-size",
        dest="local_batch_size",
        type=int,
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--num-candidates",
        dest="num_candidates",
        type=int,
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--batch-index", type=int, default=1)
    parser.add_argument("--expected-world-size", type=int, default=128)
    parser.add_argument("--tpu-data-rank-count", type=int, default=32)
    parser.add_argument("--tpu-local-batch-size", type=int, default=32)
    parser.add_argument("--allow-world-size-mismatch", action="store_true")
    return parser


def _parse_args() -> tuple[
    argparse.Namespace,
    RankFMTrainerConfig,
    RankFMModelArgs,
]:
    args = _create_parser().parse_args()
    trainer_cfg, model_args = configs_from_namespace(args)
    return args, trainer_cfg, model_args


def _build_gpu_trainer_config(
    trainer_cfg: RankFMTrainerConfig,
) -> RecTrainerConfig:
    return RecTrainerConfig(
        data_source_config=RandomDataSourceConfig(
            batch_size=trainer_cfg.local_batch_size,
            num_batches=trainer_cfg.max_steps,
        ),
        system_config=SystemConfig(
            use_gpu=True,
            dist_backend="nccl",
            use_tf32=False,
            seed_value=0,
            set_seed=True,
        ),
        metrics_config=MetricsConfig(
            report_interval=trainer_cfg.report_interval,
            tensorboard_logger=True,
        ),
        model_config_module="hammer.v3.models.udd_fm_v3.udd_fm_config",
        checkpoint_config=None,
    )


def _build_pipeline_provider(
    module_config: BlueReelsVDDModuleConfig,
    trainer_cfg: RankFMTrainerConfig,
    report_interval: int,
) -> RankFMGPUTrainPipelineProvider:
    cls_names, cls_indices, reg_names, reg_indices = build_cls_reg_partition(
        module_config
    )
    return RankFMGPUTrainPipelineProvider(
        dense_optimizer_config=trainer_cfg.dense_optimizer_config,
        cls_metric_defs=[
            RankingMetricDef(
                metric_cls=FixedStepNE,
                tasks=cls_names,
                batch_size=trainer_cfg.local_batch_size,
                report_interval=report_interval,
            ),
            RankingMetricDef(
                metric_cls=FixedStepCTR,
                tasks=cls_names,
                batch_size=trainer_cfg.local_batch_size,
                report_interval=report_interval,
            ),
        ],
        reg_metric_defs=[
            RankingMetricDef(
                metric_cls=FixedStepMSE,
                tasks=reg_names,
                batch_size=trainer_cfg.local_batch_size,
                report_interval=report_interval,
            )
        ],
        cls_task_indices=cls_indices,
        reg_task_indices=reg_indices,
        report_interval=report_interval,
    )


def _log_manifest(
    args: argparse.Namespace,
    trainer_cfg: RankFMTrainerConfig,
    model_args: RankFMModelArgs,
    world_size: int,
) -> None:
    logger.info(
        "RankFM GPU manifest: world_size=%d local_batch_size=%d "
        "global_batch_size=%d tpu_data_ranks=%d steps=%d data=%s/%s "
        "model=(hash=%d, L1=%d, L2=%d, max_seq=%d, candidates=%d) "
        "dense_optimizer=%r embedding_optimizer=EXACT_ROWWISE_ADAGRAD@%g "
        "embedding_eps=%g "
        "kernel=PYTORCH custom_kernel=False tf32=False bf16=%s dropout=0",
        world_size,
        trainer_cfg.local_batch_size,
        world_size * trainer_cfg.local_batch_size,
        args.tpu_data_rank_count,
        trainer_cfg.max_steps,
        trainer_cfg.gcs_data_path,
        rank_data_file(trainer_cfg.data_file_template, 0, args.batch_index),
        model_args.hash_size_cap,
        model_args.l1_num_layers,
        model_args.l2_num_layers,
        model_args.max_uih_seq_len,
        model_args.num_candidates,
        trainer_cfg.dense_optimizer_config,
        trainer_cfg.embedding_lr,
        trainer_cfg.embedding_eps,
        trainer_cfg.bf16_training,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    # Match the TPU parity path's deterministic timestamp-embedding backward.
    os.environ["RANKFM_TS_ONEHOT_BWD"] = "1"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    args, trainer_cfg, model_args = _parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    validate_world_size(
        world_size,
        args.expected_world_size,
        args.allow_world_size_mismatch,
    )
    validate_batch_partition(
        world_size,
        args.tpu_data_rank_count,
        trainer_cfg.local_batch_size,
        args.tpu_local_batch_size,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("train_on_gpu requires CUDA")
    if trainer_cfg.enable_padding:
        raise ValueError("GPU eager training requires --no_enable_padding")

    model_config = build_gpu_model_config(trainer_cfg, model_args, world_size)
    module_config = cast(BlueReelsVDDModuleConfig, model_config.module_config)
    RankFMGPUModelFamily.configure(
        trainer_cfg,
        model_args,
        args.batch_index,
        args.tpu_data_rank_count,
        args.tpu_local_batch_size,
    )
    trainer_config = _build_gpu_trainer_config(trainer_cfg)
    if int(os.environ.get("RANK", "0")) == 0:
        _log_manifest(args, trainer_cfg, model_args, world_size)

    RecTrainer(
        model_family_clazz=RankFMGPUModelFamily,
        trainer_config=trainer_config,
        model_config=model_config,
        train_pipeline_provider=_build_pipeline_provider(
            module_config,
            trainer_cfg,
            trainer_config.metrics_config.report_interval,
        ),
        job_resolver=JobResolver(),
    ).train()


if __name__ == "__main__":
    main()
