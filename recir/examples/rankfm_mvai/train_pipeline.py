#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""recir XLA-GSPMD train pipeline for the mvai_cloud-driven RankFM TPU example.

The self-compiling model wrapper (RecirXLACompiledModule in model_family.py)
owns compile + fwd + backward + dense opt.step + device-state sync; calling it
returns the detached ``(losses, named_tensors)`` that
``recir_core.run_step_and_sync`` produces. This pipeline consumes that return
to report the SAME ranking metric set as the mvai_cloud udd_fm reference
(per-task NE + CTR for classification, MSE + RMSE for regression,
example_count, variable_step_qps).

  * ``RecirXLATrainPipeline`` subclasses mvai_cloud's
    ``RankingSGDTrainPipeline`` to reuse its metric construction
    (``__init__``), its ``_update_ranking_metrics(ret)`` cls/reg slicing, and
    its ``.metrics`` property unchanged. ``progress`` is overridden to run the
    self-compiling model wrapper (instead of the eager fwd/bwd/opt.step), rebuild
    the ``{prediction, label, weight}`` ``ret`` dict from the wrapper's return,
    and drive the JAX profiler trace over the configured step window (ported from
    the bespoke rankfm loop so run_tpu_bench.sh's xprof trace fetch keeps working).
  * ``RecirXLATrainPipelineProvider`` reuses mvai_cloud's
    ``RankingSGDTrainPipelineProvider`` to construct the cls/reg/step metric
    lists (including the ``is_variable_step`` swap) exactly as udd_fm/train.py
    does, then hands them to ``RecirXLATrainPipeline``.

Metrics are constructed on CPU because recir returns CPU torch tensors.
"""

from __future__ import annotations

import logging
import os
from typing import Any, cast, Dict, Iterator, List, Optional, Tuple, TYPE_CHECKING

import jax
import torch
import torch.nn as nn
from minimal_viable_ai.core.trainer.train_pipeline import RecTrainPipelineProvider

from .xsurface_helpers import _log_hbm_memory

# minimal_viable_ai is provided at RUNTIME via the Docker image (Dockerfile
# rsyncs fbcode/mvai_cloud/minimal_viable_ai -- the lightweight RankingSGD* API
# this example targets). pyre type-checks against the canonical buck package
# //minimal_viable_ai instead (a different, RankingMTML* API where these names
# don't exist), so under the type checker we stub them as Any (runtime-only dep).
if TYPE_CHECKING:
    RankingMetricDef = Any
    RankingSGDTrainPipeline = Any
    RankingSGDTrainPipelineProvider = Any

    from .model_family import RecirXLACompiledModule
else:
    from minimal_viable_ai.core.ranking_common.train_pipeline import (
        RankingMetricDef,
        RankingSGDTrainPipeline,
        RankingSGDTrainPipelineProvider,
    )

logger: logging.Logger = logging.getLogger(__name__)


class RecirXLATrainPipeline(RankingSGDTrainPipeline):
    """Self-compiling-model pipeline + udd_fm-equivalent ranking metrics.

    Reuses RankingSGDTrainPipeline's metric construction (``__init__``), its
    ``_update_ranking_metrics(ret)`` cls/reg slicing, and its ``.metrics``
    property. ``progress`` is overridden: it runs the self-compiling model
    wrapper (which does fwd -> backward -> dense opt.step -> sync internally and
    returns the detached ``(losses, named_tensors)``), rebuilds the
    ``{prediction, label, weight}`` ``ret`` dict the base metric update consumes,
    and drives the JAX profiler trace over [profile_start_step, profile_end_step]
    (1-indexed step count). RecTrainer owns the step loop, so the profiler hook
    lives here -- the per-step driver -- rather than around train().
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        cls_metrics: Optional[List[Any]] = None,
        reg_metrics: Optional[List[Any]] = None,
        step_metrics: Optional[List[Any]] = None,
        cls_task_indices: Optional[List[int]] = None,
        reg_task_indices: Optional[List[int]] = None,
        host_sync_every: int = 1,
        jax_profiling: bool = False,
        profile_start_step: int = -1,
        profile_end_step: int = -1,
        profile_dir: str = "/tmp/tpu_profile",
        enable_memory_log: bool = False,
    ) -> None:
        # pyre-ignore[19]: base RankingSGDTrainPipeline is a runtime-only import
        # stubbed as Any (see module-level TYPE_CHECKING block), so pyre resolves
        # super().__init__ to object.__init__; the real base takes these args.
        super().__init__(
            model,
            optimizer,
            device,
            cls_metrics=cls_metrics,
            reg_metrics=reg_metrics,
            step_metrics=step_metrics,
            cls_task_indices=cls_task_indices,
            reg_task_indices=reg_task_indices,
        )
        self._model.train(True)
        self._jax_profiling = jax_profiling
        self._profile_start_step = profile_start_step
        self._profile_end_step = profile_end_step
        self._profile_dir = profile_dir
        self._enable_memory_log = enable_memory_log
        self._step = 0
        self._host_sync_every: int = max(1, host_sync_every)
        self._profiling_active = False
        if jax_profiling:
            logger.info(
                f"JAX profiling enabled: steps {profile_start_step}-"
                f"{profile_end_step}, dir={profile_dir}"
            )

    def progress(
        self, dataloader_iter: Iterator[Any]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        batch = next(dataloader_iter)
        self._step += 1

        # Start the xprof trace at the configured step (before the step runs so
        # the step lands fully captured). Local dirs are created; gs:// paths are
        # written directly by the profiler.
        if self._jax_profiling and self._step == self._profile_start_step:
            if not self._profile_dir.startswith("gs://"):
                os.makedirs(self._profile_dir, exist_ok=True)
            jax.profiler.start_trace(self._profile_dir)
            self._profiling_active = True
            logger.info(f"[PROFILE] Started trace at step {self._step}")

        # The self-compiling wrapper lazily compiles on the first call, then
        # runs fwd -> backward (fires the SparseCore sparse opt off-GM, or the
        # in-graph emb opt in-GM) -> dense opt.step on TPU. It returns the
        # (losses, named_tensors); metric update runs every step so the
        # example-count / variable_step_qps stay exact.
        losses, named = self._model(batch)

        if self._profiling_active and self._enable_memory_log:
            _log_hbm_memory()

        # Stop the trace at the end step (after the step ran).
        if (
            self._jax_profiling
            and self._profiling_active
            and self._step == self._profile_end_step
        ):
            jax.profiler.stop_trace()
            self._profiling_active = False
            logger.info(
                f"[PROFILE] Stopped trace at step {self._step} -- saved to "
                f"{self._profile_dir}"
            )

        # Mirror device-resident state (dense params / LR / metric buffers) back to
        # torch only every N steps. The next step's math runs entirely from the
        # cached JAX device_state, so this is pure host-visibility work; doing it
        # every step serializes ~0.3s/step of device->host copy with no device
        # overlap. Host params/metric buffers are then fresh every N steps (and at
        # train end via on_train_end); report_interval=N keeps the metric report
        # compute on the same cadence.
        if self._step % self._host_sync_every == 0:
            cast("RecirXLACompiledModule", self._model).flush_host_state()

        # Rebuild the ret dict RankingSGDTrainPipeline._update_ranking_metrics
        # expects; the metrics were constructed on torch.device("cpu") to match
        # these host CPU tensors. weight=0 marks padded rows so NE/CTR/MSE stay
        # correct (variable_step_qps may overcount padded rows -- acceptable).
        ret: Dict[str, Any] = {
            "prediction": named["prediction"],
            "label": named["label"],
            "weight": named["weight"],
        }
        self._update_ranking_metrics(ret)
        return losses, ret

    def on_train_end(self) -> None:
        # Stop any still-active trace (e.g. training ended before profile_end_step).
        if self._profiling_active:
            jax.profiler.stop_trace()
            self._profiling_active = False
            logger.info("[PROFILE] Stopped trace (training ended).")
        # Final host-state flush so params/buffers are fresh for any end-of-run
        # inspection/checkpoint even if the last step was not a flush step.
        cast("RecirXLACompiledModule", self._model).flush_host_state()
        logger.info("RecirXLATrainPipeline: training complete.")


class RecirXLATrainPipelineProvider(
    RecTrainPipelineProvider[Any, Tuple[torch.Tensor, Dict[str, Any]]]
):
    """Returns the self-compiling-model pipeline + udd_fm-equivalent metrics.

    Reuses mvai_cloud's ``RankingSGDTrainPipelineProvider`` to construct the
    cls/reg/step metric lists (including the ``is_variable_step`` swap) exactly
    as udd_fm/train.py does; all compile work lives in the model wrapper. The
    profiling config is threaded into the pipeline so the JAX profiler trace is
    emitted over the configured step window (consumed by run_tpu_bench.sh).
    """

    def __init__(
        self,
        cls_metric_defs: Optional[List[RankingMetricDef]] = None,
        reg_metric_defs: Optional[List[RankingMetricDef]] = None,
        cls_task_indices: Optional[List[int]] = None,
        reg_task_indices: Optional[List[int]] = None,
        is_variable_step: bool = True,
        report_interval: int = 1,
        jax_profiling: bool = False,
        profile_start_step: int = -1,
        profile_end_step: int = -1,
        profile_dir: str = "/tmp/tpu_profile",
        enable_memory_log: bool = False,
    ) -> None:
        self._cls_task_indices = cls_task_indices
        self._reg_task_indices = reg_task_indices
        self._host_sync_every = report_interval
        self._jax_profiling = jax_profiling
        self._profile_start_step = profile_start_step
        self._profile_end_step = profile_end_step
        self._profile_dir = profile_dir
        self._enable_memory_log = enable_memory_log
        # Delegate metric-list construction (and the variable-step swap) to the
        # mvai_cloud provider verbatim -- same args udd_fm/train.py passes.
        # Annotated Any: RankingSGDTrainPipelineProvider is a runtime-only import
        # (stubbed Any for pyre -- see the module-level TYPE_CHECKING block).
        # pyrefly: ignore [bad-instantiation]
        self._metrics_provider: Any = RankingSGDTrainPipelineProvider(
            cls_metric_defs=cls_metric_defs,
            reg_metric_defs=reg_metric_defs,
            cls_task_indices=cls_task_indices,
            reg_task_indices=reg_task_indices,
            is_variable_step=is_variable_step,
            report_interval=report_interval,
        )

    # pyre-ignore
    def create_train_pipeline(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: Optional[torch.device] = None,
        pg: Optional[Any] = None,
    ) -> RecirXLATrainPipeline:
        # recir returns CPU torch tensors (named_tensors), so the metric objects
        # (and the cls/reg index tensors) must live on CPU. Force a CPU device
        # for metric construction regardless of the `device` RecTrainer passes
        # (which is cpu on the use_gpu=False branch anyway).
        cpu_device = torch.device("cpu")
        base_pipeline = self._metrics_provider.create_train_pipeline(
            model, optimizer, cpu_device, pg
        )
        return RecirXLATrainPipeline(
            model,
            optimizer,
            cpu_device,
            cls_metrics=base_pipeline._cls_metrics,
            reg_metrics=base_pipeline._reg_metrics,
            step_metrics=base_pipeline._step_metrics,
            cls_task_indices=self._cls_task_indices,
            reg_task_indices=self._reg_task_indices,
            host_sync_every=self._host_sync_every,
            jax_profiling=self._jax_profiling,
            profile_start_step=self._profile_start_step,
            profile_end_step=self._profile_end_step,
            profile_dir=self._profile_dir,
            enable_memory_log=self._enable_memory_log,
        )
