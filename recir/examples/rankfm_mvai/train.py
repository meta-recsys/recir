#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""RankFM (xsurface V3) TPU training, driven by the mvai_cloud RecTrainer.

Runs the hammer/v3 udd_fm ``BlueReelsVDDTrainV3`` on TPU via recir XLA-GSPMD,
orchestrated by the mvai_cloud training framework (RecTrainer + RecModelFamily +
RecTrainPipeline).

The TPU path is achieved WITHOUT editing mvai_cloud:
  * For multi-host jobs, ``jax.distributed.initialize()`` is called here before
    RecTrainer is constructed. Single-host jobs use JAX's local autodetection.
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
from .xsurface_helpers import _is_jax_profiling_enabled

logger: logging.Logger = logging.getLogger(__name__)


def _init_gloo_pg() -> None:
    """Create the process-wide gloo process group at the entrypoint.

    Multi-host checkpointing (DCP) and the host-side metric collectives need a
    torch gloo PG. Owning it here -- before the RecTrainer and the embedding
    backend -- keeps the process-wide PG a cross-cutting entrypoint concern
    rather than a side effect of a specific embedding backend, and makes it
    independent of which backend (cpu/tpu/in_graph) is selected. Uses the env
    populated by resolve_dist_env(). Single-host (world<=1) is a no-op.
    Idempotent: RecTrainer's init_process_group and the backends' init_pg() both
    defer when the PG already exists, so this stays the single creation site.
    """
    from datetime import timedelta

    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world <= 1 or dist.is_initialized():
        return
    rank = int(os.environ.get("RANK", "0"))
    coordinator = os.environ.get("MASTER_ADDR", "localhost")
    port = int(os.environ.get("MASTER_PORT", "29500"))
    logger.info(
        f"[DIST] gloo PG init (entrypoint): rank={rank}, world={world}, "
        f"coordinator={coordinator}:{port}"
    )
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://{coordinator}:{port}",
        world_size=world,
        rank=rank,
        timeout=timedelta(seconds=300),
    )


def main() -> None:
    setup_logging()
    resolve_dist_env()

    # Multi-host (e.g. 128-chip): must call jax.distributed.initialize() before
    # any JAX op so the PJRT plugin reads the coordinator address + process index
    # from TPU runtime metadata; without it each pod is jax.process_count()==1 and
    # the cross-pod GSPMD collective deadlocks.
    #
    # Single-host (1 pod, e.g. the 2x2x1 one-shard run): do NOT call it.
    # jax.distributed.initialize() is for MULTI-process coordination; on a lone
    # pod it mis-initializes the runtime in a way that makes the subsequent eager
    # get_override CPU forward raise (a caught error -> "Sparse arch forward hook
    # did not fire"). jax auto-detects the local chips without it (the CPU-eager
    # entry, which never calls it, builds get_override cleanly -- the proof).
    # WORLD_SIZE is preserved when provided, otherwise derived from the
    # launcher-provided TPU process addresses.
    single_host = int(os.environ.get("WORLD_SIZE", "1")) == 1
    if not single_host:
        jax.distributed.initialize()
    logger.info(
        f"[DIST] JAX state: "
        f"process_count={jax.process_count()}, "
        f"device_count={jax.device_count()}, "
        f"local_device_count={jax.local_device_count()}, "
        f"process_index={jax.process_index()}"
    )

    trainer_cfg, model_args = parse_cli()
    logger.info(f"Trainer config: {trainer_cfg}")
    logger.info(f"Model args: {model_args}")

    # Create the process-wide gloo PG here (entrypoint-owned) before the trainer
    # / embedding backend need it -- see _init_gloo_pg.
    _init_gloo_pg()
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

    # JAX profiling is gated on a real TPU + a positive profile_start_step; on
    # CPU / when unset it is a no-op. The trace (when enabled) lands in
    # profile_dir (gs://.../tpu-profiles/<JOBSET_NAME>) and is fetched by
    # run_tpu_bench.sh.
    jax_profiling = _is_jax_profiling_enabled(trainer_cfg)

    RecTrainer(
        model_family_clazz=RankFMMvaiModelFamily,
        trainer_config=trainer_config,
        model_config=model_config,
        train_pipeline_provider=RecirXLATrainPipelineProvider(
            cls_metric_defs=[
                # pyrefly: ignore [bad-instantiation]
                RankingMetricDef(
                    metric_cls=FixedStepNE,
                    tasks=cls_task_names,
                    batch_size=batch_size,
                    report_interval=report_interval,
                ),
                # pyrefly: ignore [bad-instantiation]
                RankingMetricDef(
                    metric_cls=FixedStepCTR,
                    tasks=cls_task_names,
                    batch_size=batch_size,
                    report_interval=report_interval,
                ),
            ],
            reg_metric_defs=[
                # pyrefly: ignore [bad-instantiation]
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
            jax_profiling=jax_profiling,
            profile_start_step=trainer_cfg.profile_start_step,
            profile_end_step=trainer_cfg.profile_end_step,
            profile_dir=trainer_cfg.profile_dir,
            enable_memory_log=trainer_cfg.enable_memory_log,
        ),
        job_resolver=JobResolver(),
    ).train()

    # Cleanup. JAX's PJRT runtime + libtpu workers + any gloo collective thread
    # keep the process alive after train() returns, so the JobSet pod never
    # reaches Completed. Shut both down, then force-exit.
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
