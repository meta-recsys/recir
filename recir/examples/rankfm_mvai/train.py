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

import json
import logging
import os
from pathlib import Path
from typing import Any, cast, TYPE_CHECKING

import jax
import torch.distributed as dist
from hammer.v3.models.udd_fm_v3.parity import leak_probe
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

# Shared bootstrap (gr::batched_complete_cumsum, runs on import) + dist/metric
# helpers, kept identical with the CPU-eager entry (train_cpu_eager.py).
from recir.training.tpu_runtime import (
    force_cpu_barrier,
    init_jax_distributed,
    resolve_dist_env,
    setup_logging,
    shutdown,
)

from .config import build_model_family_config, build_trainer_config, parse_cli
from .model_family import RankFMMvaiModelFamily
from .task_partition import build_cls_reg_partition
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


def _log_build_info() -> None:
    """Report what the image was built from.

    ``build_image_podman.sh`` stages the working tree, so the code in an image
    can match no commit; this is the only mapping from a running pod back to
    source, and it is worth a line in every log.
    """
    path = Path(__file__).parents[2] / "BUILD_INFO.json"
    try:
        info = json.loads(path.read_text())
    except (OSError, ValueError):
        logger.warning(
            "[build] no readable %s -- image predates build stamping, so its "
            "contents cannot be mapped to a revision",
            path,
        )
        return
    logger.info(
        "[build] tag=%s revision=%s built_at=%s by=%s",
        info.get("tag"),
        info.get("revision"),
        info.get("built_at"),
        info.get("built_by"),
    )
    # Absent in images built before this key existed, hence the None check --
    # only an explicit False means the wheel was deliberately left out.
    if info.get("barnfs") is False:
        logger.warning(
            "[build] this image has NO barnfs. DCP checkpoints are written "
            "unencrypted, and TORCH_ENABLE_BARNFS=ON would fail to import "
            "rather than encrypt them."
        )
    dirty = info.get("dirty_files") or []
    if dirty:
        logger.warning(
            "[build] built from a DIRTY tree: %d uncommitted file(s) are in "
            "this image and may exist in no commit: %s",
            len(dirty),
            ", ".join(dirty),
        )


def main() -> None:
    setup_logging()
    _log_build_info()
    resolve_dist_env()

    # Arm the leak probe here and nowhere later. Everything downstream of the
    # preprocessor is traced once into a compiled graph, so a probe installed
    # after the model is built never runs -- and its export key is absent when
    # the dummy preprocessor caches the graph's output shapes, which means its
    # reading comes back as zeros rather than as missing. Inert unless
    # RANKFM_DUP_TARGETS is set.
    leak_probe.install(leak_probe.from_env())

    init_jax_distributed()

    trainer_cfg, model_args = parse_cli()
    logger.info(f"Trainer config: {trainer_cfg}")
    logger.info(f"Model args: {model_args}")

    # Create the process-wide gloo PG here (entrypoint-owned) before the trainer
    # / embedding backend need it -- see _init_gloo_pg.
    _init_gloo_pg()
    # Route torch.distributed.barrier onto CPU before any gloo collective runs
    # (the metric all-gather at report time). See tpu_runtime.force_cpu_barrier.
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
    partition = build_cls_reg_partition(module_config)
    report_interval = trainer_config.metrics_config.report_interval
    batch_size = trainer_config.data_source_config.batch_size
    window_size = trainer_cfg.metric_window_size

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
                    tasks=partition.cls_task_names,
                    batch_size=batch_size,
                    report_interval=report_interval,
                    window_size=window_size,
                ),
                # pyrefly: ignore [bad-instantiation]
                RankingMetricDef(
                    metric_cls=FixedStepCTR,
                    tasks=partition.cls_task_names,
                    batch_size=batch_size,
                    report_interval=report_interval,
                    window_size=window_size,
                ),
            ],
            reg_metric_defs=[
                # pyrefly: ignore [bad-instantiation]
                RankingMetricDef(
                    metric_cls=FixedStepMSE,
                    tasks=partition.reg_task_names,
                    batch_size=batch_size,
                    report_interval=report_interval,
                    window_size=window_size,
                ),
            ],
            cls_task_indices=partition.cls_task_indices,
            reg_task_indices=partition.reg_task_indices,
            all_task_names=partition.all_task_names,
            is_variable_step=True,
            report_interval=report_interval,
            jax_profiling=jax_profiling,
            profile_start_step=trainer_cfg.profile_start_step,
            profile_end_step=trainer_cfg.profile_end_step,
            profile_dir=trainer_cfg.profile_dir,
            enable_memory_log=trainer_cfg.enable_memory_log,
            pipeline_preprocessor=trainer_cfg.pipeline_preprocessor,
            window_size=window_size,
        ),
        job_resolver=JobResolver(),
    ).train()

    shutdown()


if __name__ == "__main__":
    main()
