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
    its ``.metrics`` property unchanged. Only ``progress`` is overridden to run
    the self-compiling model wrapper (instead of the eager fwd/bwd/opt.step) and
    rebuild the ``{prediction, label, weight}`` ``ret`` dict from the wrapper's
    return value. ``named_tensors`` already carries DETACHED prediction / label
    / weight (shape [T, B]) from ``BlueReelsVDDTrainV3.forward``.
  * ``RecirXLATrainPipelineProvider`` reuses mvai_cloud's
    ``RankingSGDTrainPipelineProvider`` to construct the cls/reg/step metric
    lists (including the ``is_variable_step`` swap) exactly as udd_fm/train.py
    does, then hands them to ``RecirXLATrainPipeline``.

Metrics are constructed on CPU because recir returns CPU torch tensors.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, List, Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn
from minimal_viable_ai.core.trainer.train_pipeline import RecTrainPipelineProvider

# minimal_viable_ai is provided at RUNTIME via the Docker image (Dockerfile
# rsyncs fbcode/mvai_cloud/minimal_viable_ai -- the lightweight RankingSGD* API
# this example targets). pyre type-checks against the canonical buck package
# //minimal_viable_ai instead (a different, RankingMTML* API where these names
# don't exist), so under the type checker we stub them as Any (runtime-only dep).
if TYPE_CHECKING:
    RankingMetricDef = Any
    RankingSGDTrainPipeline = Any
    RankingSGDTrainPipelineProvider = Any
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
    property. Only ``progress`` is overridden: it runs the self-compiling model
    wrapper (which does fwd -> backward -> dense opt.step -> sync internally and
    returns the detached ``(losses, named_tensors)``) and rebuilds the
    ``{prediction, label, weight}`` ``ret`` dict the base metric update consumes.
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

    def progress(
        self, dataloader_iter: Iterator[Any]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        batch = next(dataloader_iter)
        # The self-compiling wrapper lazily compiles on the first call, then
        # runs fwd -> backward (fires the SparseCore sparse opt off-GM) -> dense
        # opt.step on TPU and syncs device state. It returns the detached
        # (losses, named_tensors) from recir_core.run_step_and_sync.
        losses, named = self._model(batch)

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
        logger.info("RecirXLATrainPipeline: training complete.")


class RecirXLATrainPipelineProvider(
    RecTrainPipelineProvider[Any, Tuple[torch.Tensor, Dict[str, Any]]]
):
    """Returns the self-compiling-model pipeline + udd_fm-equivalent metrics.

    Reuses mvai_cloud's ``RankingSGDTrainPipelineProvider`` to construct the
    cls/reg/step metric lists (including the ``is_variable_step`` swap) exactly
    as udd_fm/train.py does; all compile work lives in the model wrapper.
    """

    def __init__(
        self,
        cls_metric_defs: Optional[List[RankingMetricDef]] = None,
        reg_metric_defs: Optional[List[RankingMetricDef]] = None,
        cls_task_indices: Optional[List[int]] = None,
        reg_task_indices: Optional[List[int]] = None,
        is_variable_step: bool = True,
        report_interval: int = 1,
    ) -> None:
        self._cls_task_indices = cls_task_indices
        self._reg_task_indices = reg_task_indices
        # Delegate metric-list construction (and the variable-step swap) to the
        # mvai_cloud provider verbatim -- same args udd_fm/train.py passes.
        # Annotated Any: RankingSGDTrainPipelineProvider is a runtime-only import
        # (stubbed Any for pyre -- see the module-level TYPE_CHECKING block).
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
        )
