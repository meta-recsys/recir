#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""RankFM (xsurface V3) TPU training, driven by the mvai_cloud RecTrainer.

Runs the SAME model as recir/examples/rankfm (the hammer/v3 udd_fm
``BlueReelsVDDTrainV3``) on TPU via recir XLA-GSPMD, but orchestrated by the
mvai_cloud training framework (RecTrainer + RecModelFamily + RecTrainPipeline)
instead of rankfm's bespoke train.py loop.

The TPU path is achieved WITHOUT editing mvai_cloud:
  * ``jax.distributed.initialize()`` is called here, before RecTrainer is
    constructed (mirrors rankfm/train.py:682).
  * ``SystemConfig(use_gpu=False)`` makes RecTrainer take its CPU/gloo device
    branch (trainer.py:81-83); recir lowers the dense graph to TPU via
    ``jax.devices()`` regardless of the torch device being cpu.
  * All TPU/recir logic lives in RankFMMvaiModelFamily (model + recir embedding
    backend + optimizer) and RecirXLATrainPipeline (recir compile + step).

Usage:
    python3 -u -m recir.examples.rankfm_mvai.train --max_steps 10
    python3 -u -m recir.examples.rankfm_mvai.train --max_steps 5 \\
        --embedding_backend tpu
"""

from __future__ import annotations

import logging
import os
from typing import Any, cast, TYPE_CHECKING

import jax
import torch.distributed as dist
from hammer.v3.models.udd_fm_v3.udd_fm_config import BlueReelsVDDModuleConfig
from minimal_viable_ai.core.ranking_common.metrics.ctr import FixedStepCTR
from minimal_viable_ai.core.ranking_common.metrics.mse import FixedStepMSE
from minimal_viable_ai.core.ranking_common.metrics.ne import FixedStepNE

# minimal_viable_ai is provided at RUNTIME via the Docker image (Dockerfile
# rsyncs fbcode/mvai_cloud/minimal_viable_ai -- the lightweight RankingSGD* API
# this example targets). pyre type-checks against the canonical buck package
# //minimal_viable_ai instead (a different, RankingMTML* API with no
# RankingMetricDef), so under the type checker we stub it as Any.
if TYPE_CHECKING:
    RankingMetricDef = Any
else:
    from minimal_viable_ai.core.ranking_common.train_pipeline import RankingMetricDef
from minimal_viable_ai.core.trainer.job_resolver import JobResolver
from minimal_viable_ai.core.trainer.trainer import RecTrainer

from .config import build_model_family_config, build_trainer_config, parse_cli
from .model_family import RankFMMvaiModelFamily

# Shared bootstrap (gr::batched_complete_cumsum, runs on import) + dist/metric
# helpers, kept identical with the CPU-eager entry (train_cpu_eager.py).
from .train_common import (
    build_cls_reg_partition,
    force_cpu_barrier,
    resolve_dist_env,
    setup_logging,
)
from .train_pipeline import RecirXLATrainPipelineProvider

logger: logging.Logger = logging.getLogger(__name__)


def main() -> None:
    # Must be called before any JAX op. On GKE TPU the PJRT plugin reads the
    # coordinator address + process index from TPU runtime metadata; without
    # this each pod is jax.process_count()==1 and the SparseCore collective
    # (which spans all chips) deadlocks. Mirrors rankfm/train.py:682.
    jax.distributed.initialize()
    setup_logging()
    logger.info(
        f"[DIST] jax.distributed.initialize() done: "
        f"process_count={jax.process_count()}, "
        f"device_count={jax.device_count()}, "
        f"local_device_count={jax.local_device_count()}, "
        f"process_index={jax.process_index()}"
    )

    trainer_cfg, model_args = parse_cli()
    logger.info(f"Trainer config: {trainer_cfg}")
    logger.info(f"Model args: {model_args}")

    resolve_dist_env()
    # Route torch.distributed.barrier onto CPU before any gloo collective runs
    # (the metric all-gather at report time). See train_common.force_cpu_barrier.
    force_cpu_barrier()

    # Inject the rankfm CLI knobs onto the family class (the base RecModelFamily
    # ctor signature -- config/pg/device -- can't carry them).
    RankFMMvaiModelFamily.configure(trainer_cfg, model_args)

    model_config = build_model_family_config(trainer_cfg, model_args)
    trainer_config = build_trainer_config(trainer_cfg)

    # Build the cls/reg task partition from the (mutated) udd_fm_v3 module config
    # -- the SAME model udd_fm runs, so the same task structure applies. This is
    # passed into the provider so it reports the udd_fm metric set (NE + CTR for
    # cls tasks, MSE + RMSE for reg tasks, example_count, variable_step_qps).
    # cast: the canonical mvai RecModelFamilyConfig types .module_config as
    # `object`; this example only runs the udd_fm BlueReelsVDDModuleConfig.
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
        train_pipeline_provider=RecirXLATrainPipelineProvider(
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

    # Cleanup. JAX's PJRT runtime + libtpu workers + any gloo collective thread
    # keep the process alive after train() returns, so the JobSet pod never
    # reaches Completed. Shut both down, then force-exit (mirrors
    # rankfm/train.py:787-798).
    if dist.is_initialized():
        dist.destroy_process_group()
    try:
        jax.distributed.shutdown()
    except Exception as e:  # noqa: BLE001 -- best-effort cleanup
        logger.warning(f"jax.distributed.shutdown() raised (non-fatal): {e}")
    logger.info("Cleanup complete; exiting.")
    os._exit(0)


if __name__ == "__main__":
    main()
