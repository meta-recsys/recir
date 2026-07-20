#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Per-step training-loss metric for the RankFM mvai pipelines.

The mvai_cloud ``RankingSGDTrainPipeline`` logs only prediction-derived metrics
(NE / CTR / MSE); the scalar training loss returned by ``progress()`` is dropped
(the trainer discards it, and no metric consumes it). Unlike the prod FIRE stack
-- which routes ``tb:loss/aux/...`` named tensors through a ``CombinedMetrics``
aggregator -- the lightweight fork has no loss-logging channel at all.

``WindowLoss`` is a minimal ``PeriodicStepMetrics`` (same contract as the fork's
``VariableStepQPS``) that accumulates the loss and reports its window + lifetime
mean, so it shows up in TensorBoard alongside NE/CTR as
``loss/global/{window,lifetime}/train`` (the reporter appends the ``/train``
phase suffix). It is updated explicitly from the pipeline's ``progress()`` (the
loss never flows through ``_update_ranking_metrics``, which only sees
prediction/label/weight), and included via the pipeline's ``metrics`` property.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, TYPE_CHECKING

import torch

# minimal_viable_ai is a RUNTIME-only dep (rsynced mvai_cloud fork); pyre checks
# against the canonical package. Stub the base under the type checker, mirroring
# recir/examples/rankfm_mvai/train_pipeline.py.
if TYPE_CHECKING:
    MetricValueType = Any
    PeriodicStepMetrics = Any
else:
    from minimal_viable_ai.core.trainer.metrics import (
        MetricValueType,
        PeriodicStepMetrics,
    )


class WindowLoss(PeriodicStepMetrics):
    """Window + lifetime mean of one or more named loss series.

    Fed a dict of ``{series_key: scalar-tensor}`` per step via ``update(values=)``;
    for each key it reports ``{key}/global/window`` and ``{key}/global/lifetime``
    (the MetricsReporter appends the ``/train`` phase suffix). The pipeline feeds:
      * ``"loss"``                 -> ``loss/global/window/train`` (aggregate)
      * ``"loss/<task>"`` per task -> ``loss/<task>/global/window/train``
    so per-task loss overlays each task's NE curve. It rides the pipeline's
    ``metrics`` property (not the step-metric list, whose auto-update passes
    prediction/label/weight, not loss).

    Sums are accumulated on device as 0-dim tensors and converted to Python
    scalars only in ``report_metrics`` (once per report interval), so the training
    loop never triggers a per-step device-to-host sync (Citrine C6).
    """

    def __init__(
        self,
        *_: Any,
        rank: int,
        report_interval: int,
        report_on_all_rank: bool = False,
    ) -> None:
        super().__init__(
            rank=rank,
            report_interval=report_interval,
            report_on_all_rank=report_on_all_rank,
        )
        self._window_sum: Dict[str, torch.Tensor] = {}
        self._window_count: Dict[str, int] = {}
        self._lifetime_sum: Dict[str, torch.Tensor] = {}
        self._lifetime_count: Dict[str, int] = {}
        self._step_stride: int = 0

    # pyre-ignore[14]: base update() has a generic (*args, **kwargs) signature.
    def update(self, *_: Any, values: Any = None, **__: Any) -> None:
        # Bump step bookkeeping unconditionally so the iteration/stride counters
        # stay aligned with the actual step count even if `values` is ever empty.
        self._itrn += 1
        self._step_stride += 1
        if not values:
            return
        for key, v in values.items():
            # Keep the running sum a detached 0-dim tensor -- no .item() here.
            t = v.detach() if isinstance(v, torch.Tensor) else torch.as_tensor(v)
            self._window_sum[key] = (
                self._window_sum[key] + t if key in self._window_sum else t.clone()
            )
            self._window_count[key] = self._window_count.get(key, 0) + 1
            self._lifetime_sum[key] = (
                self._lifetime_sum[key] + t if key in self._lifetime_sum else t.clone()
            )
            self._lifetime_count[key] = self._lifetime_count.get(key, 0) + 1

    def report_metrics(self) -> Dict[str, Tuple[int, MetricValueType]]:
        if not self.should_compute():
            return {}

        # Snapshot and reset the window every compute tick -- even on ranks that
        # don't report -- so the window accumulators never grow unbounded.
        stride = self._step_stride
        self._step_stride = 0
        window_sum = self._window_sum
        window_count = self._window_count
        self._window_sum = {}
        self._window_count = {}

        if not self.should_report():
            return {}

        # Single device-to-host sync per report interval (.item()), not per step.
        reported: Dict[str, Tuple[int, MetricValueType]] = {}
        for key, count in self._lifetime_count.items():
            if count > 0:
                reported[f"{key}/global/lifetime"] = (
                    stride,
                    self._lifetime_sum[key].item() / count,
                )
        for key, count in window_count.items():
            if count > 0:
                reported[f"{key}/global/window"] = (
                    stride,
                    window_sum[key].item() / count,
                )
        return reported

    def reset(self) -> None:
        self._window_sum = {}
        self._window_count = {}
        self._lifetime_sum = {}
        self._lifetime_count = {}
        self._step_stride = 0
        super().reset()
