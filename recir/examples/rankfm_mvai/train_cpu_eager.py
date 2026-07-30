#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""RankFM (xsurface V3) CPU-EAGER training -- matched-config reference for the
TPU run, driven by the SAME mvai_cloud RecTrainer.

This is the CPU half of the eager-vs-TPU parity comparison. It is byte-identical
to ``train.py`` EXCEPT:
  * it plugs ``EagerCPUTrainPipelineProvider`` (runs the raw model eagerly, no
    recir/XLA compile) into RecTrainer instead of ``RecirXLATrainPipelineProvider``;
  * it forces ``embedding_backend='cpu'`` (the only eager-runnable backend) and
    ``enable_padding=False`` (PaddingRegionTransform is compile-only);
  * it never calls ``jax.distributed.initialize()`` (CPU, single process).

Driving the eager run through RecTrainer (rather than a hand-rolled loop) means
it reports the EXACT same metric set under the SAME tag names as the TPU run
(``variable_step_metrics/NE|CTR|MSE/global/{lifetime,window}/<task>/train`` via
RecTrainer's TensorboardLogger), so the two runs overlay directly in TensorBoard.

Usage:
    JAX_PLATFORMS=cpu WORLD_SIZE=1 RANK=0 \\
        python3 -u -m recir.examples.rankfm_mvai.train_cpu_eager --max_steps 50
"""

from __future__ import annotations

import logging
import os
from typing import Any, cast, TYPE_CHECKING

import torch.distributed as dist
from hammer.v3.models.udd_fm_v3.udd_fm_config import BlueReelsVDDModuleConfig
from minimal_viable_ai.core.ranking_common.metrics.ctr import FixedStepCTR
from minimal_viable_ai.core.ranking_common.metrics.mse import FixedStepMSE
from minimal_viable_ai.core.ranking_common.metrics.ne import FixedStepNE

# minimal_viable_ai is provided at RUNTIME via the Docker image (Dockerfile
# rsyncs fbcode/mvai_cloud/minimal_viable_ai). pyre type-checks against the
# canonical buck package (a different API with no RankingMetricDef), so provide
# the runtime class's constructor shape for type checking.
if TYPE_CHECKING:

    class RankingMetricDef:
        def __init__(
            self,
            *,
            metric_cls: Any,
            tasks: list[str],
            batch_size: int,
            report_interval: int,
        ) -> None: ...

else:
    from minimal_viable_ai.core.ranking_common.train_pipeline import RankingMetricDef
from minimal_viable_ai.core.trainer.job_resolver import JobResolver
from minimal_viable_ai.core.trainer.trainer import RecTrainer

from .config import build_model_family_config, build_trainer_config, parse_cli
from .model_family import RankFMMvaiModelFamily

# Shared bootstrap (gr::batched_complete_cumsum, runs on import) + dist/metric
# helpers, kept identical with the TPU entry (train.py).
from .train_common import (
    build_cls_reg_partition,
    force_cpu_barrier,
    resolve_dist_env,
    setup_logging,
)
from .train_pipeline import EagerCPUTrainPipelineProvider

logger: logging.Logger = logging.getLogger(__name__)


def main() -> None:
    """CPU-eager entrypoint: RecTrainer + EagerCPUTrainPipelineProvider.

    No ``jax.distributed.initialize()`` (CPU, single process; JAX_PLATFORMS=cpu
    keeps the incidental jax.device_count() in the model build on CPU). Forces
    ``embedding_backend='cpu'`` and ``enable_padding=False``.
    """
    setup_logging()

    trainer_cfg, model_args = parse_cli()
    if trainer_cfg.embedding_backend != "cpu":
        logger.warning(
            "[EAGER-CPU] overriding embedding_backend '%s' -> 'cpu' "
            "(only the cpu backend runs eagerly)",
            trainer_cfg.embedding_backend,
        )
        trainer_cfg.embedding_backend = "cpu"
    if trainer_cfg.enable_padding:
        logger.info(
            "[EAGER-CPU] overriding enable_padding=True -> False "
            "(PaddingRegionTransform is compile-only)"
        )
        trainer_cfg.enable_padding = False
    logger.info(f"[EAGER-CPU] trainer config: {trainer_cfg}")
    logger.info(f"[EAGER-CPU] model args: {model_args}")

    resolve_dist_env()
    # Route torch.distributed.barrier onto CPU before any gloo collective runs
    # (the metric all-gather at report time). See train_common.force_cpu_barrier.
    force_cpu_barrier()

    # Inject the rankfm CLI knobs onto the family class.
    RankFMMvaiModelFamily.configure(trainer_cfg, model_args)

    model_config = build_model_family_config(trainer_cfg, model_args)
    trainer_config = build_trainer_config(trainer_cfg)

    # cls/reg task partition from the (mutated) udd_fm_v3 module config -- the
    # SAME structure the TPU run reports, so the metric tags match exactly.
    module_config: BlueReelsVDDModuleConfig = cast(
        BlueReelsVDDModuleConfig, model_config.module_config
    )
    cls_task_names, cls_task_indices, reg_task_names, reg_task_indices = (
        build_cls_reg_partition(module_config)
    )
    report_interval = trainer_config.metrics_config.report_interval
    batch_size = trainer_config.data_source_config.batch_size

    RecTrainer(
        model_family_clazz=RankFMMvaiModelFamily,
        trainer_config=trainer_config,
        model_config=model_config,
        train_pipeline_provider=EagerCPUTrainPipelineProvider(
            cls_metric_defs=[
                RankingMetricDef(
                    metric_cls=FixedStepNE,
                    tasks=cls_task_names,
                    batch_size=batch_size,
                    report_interval=report_interval,
                ),
                RankingMetricDef(
                    metric_cls=FixedStepCTR,
                    tasks=cls_task_names,
                    batch_size=batch_size,
                    report_interval=report_interval,
                ),
            ],
            reg_metric_defs=[
                RankingMetricDef(
                    metric_cls=FixedStepMSE,
                    tasks=reg_task_names,
                    batch_size=batch_size,
                    report_interval=report_interval,
                ),
            ],
            cls_task_indices=cls_task_indices,
            reg_task_indices=reg_task_indices,
            is_variable_step=True,
            report_interval=report_interval,
        ),
        job_resolver=JobResolver(),
    ).train()

    if dist.is_initialized():
        dist.destroy_process_group()
    logger.info("[EAGER-CPU] cleanup complete; exiting.")
    os._exit(0)


if __name__ == "__main__":
    main()
