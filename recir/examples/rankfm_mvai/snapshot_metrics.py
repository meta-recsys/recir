#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""The per-task snapshot channel, matching what the MAST/mvai leg emits.

MAST routes four quantities through one mechanism -- a named-tensor snapshot fed
into ``FixedStepLocalTensor`` -- producing::

    loss/local/lifetime/{task}/train
    llw_weights/local/lifetime/{task}/train
    loss_scale/local/lifetime/{task}/train
    learning_rate/optimizer___dense/train
    learning_rate/optimizer___sparse_fused/train

All five come off the same channel at the same cadence and on the same x-axis
(cumulative examples), which is why the LR pair is expressed here as just
another two-row "task" tensor rather than as a special case.

The value semantics matter as much as the names: these are point-in-time
snapshots, not window means. Averaging an LLW weight or an LR over a report
interval smears the trajectory being inspected, so a mean-based channel is not
a substitute even if the tags line up.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    FixedStepLocalTensor = Any
    FixedStepEffectiveSampleCnt = Any
else:
    from minimal_viable_ai.core.ranking_common.metrics.effective_sample_cnt import (
        FixedStepEffectiveSampleCnt,
    )
    from minimal_viable_ai.core.ranking_common.metrics.local_tensor import (
        FixedStepLocalTensor,
    )

logger: logging.Logger = logging.getLogger(__name__)

# MAST derives these from the optimizer's param-group names
# (ranking_common/train_pipeline.py:2024-2031 builds f"learning_rate/optimizer_{name}"
# with names "__dense" / "__sparse_fused"). The TPU leg has no equivalent param
# groups -- it reads the LR straight off the warmup buffers -- so the two labels
# are pinned here to keep the tags identical.
LR_TASKS: List[str] = ["optimizer___dense", "optimizer___sparse_fused"]


def global_state_scale() -> float:
    if not (dist.is_available() and dist.is_initialized()):
        return 1.0
    return 1.0 / float(dist.get_world_size())


class GlobalStateEffectiveSampleCnt(FixedStepEffectiveSampleCnt):
    """``FixedStepEffectiveSampleCnt`` for a leg whose inputs are already global."""

    def __init__(self, *, state_scale: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.state_scale: float = state_scale

    # pyre-fixme[14]: matches the base's `weights`-keyword signature.
    def update(
        self,
        *_: Any,
        weights: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> None:
        # Mirrors the base body (`effective_sample_cnt.py:83-92`); reproduced
        # rather than wrapped because the scale has to land on `counts` before
        # it is folded into two separate buffers.
        self._itrn += 1
        if not self.should_update(**kwargs):
            return
        if weights is None:
            return
        counts = (weights.detach() > 0).sum(dim=1).to(torch.float64)
        counts = counts * self.state_scale
        n = min(counts.numel(), self._lifetime.numel())
        self._lifetime[:n] += counts[:n]
        self._window[:n] += counts[:n]
        self._step_stride += self._global_batch_size


class SnapshotMetrics:
    """Owns the four ``FixedStepLocalTensor`` instances and knows how to feed them.

    Shared by the TPU and CPU-eager pipelines so both emit byte-identical tags.
    """

    def __init__(
        self,
        rank: int,
        report_interval: int,
        global_batch_size: int,
        all_task_names: List[str],
    ) -> None:
        def make(key_name: str, tasks: List[str]) -> FixedStepLocalTensor:
            # pyrefly: ignore [bad-instantiation]
            return FixedStepLocalTensor(
                rank=rank,
                tasks=tasks,
                key_name=key_name,
                global_batch_size=global_batch_size,
                report_interval=report_interval,
                # Explicit because the two trees disagree on the default: the
                # internal class defaults to True (safe there -- its TB writer
                # exists only on rank 0), while this fork builds a writer on
                # every rank, so reporting from all of them would drop 32 event
                # files into one gcsfuse dir. Pinning it here keeps the buck and
                # in-image code paths behaving identically.
                report_on_all_rank=False,
            )

        self._loss: FixedStepLocalTensor = make("loss/local/lifetime", all_task_names)
        self._llw_weights: FixedStepLocalTensor = make(
            "llw_weights/local/lifetime", all_task_names
        )
        self._loss_scale: FixedStepLocalTensor = make(
            "loss_scale/local/lifetime", all_task_names
        )
        self._learning_rate: FixedStepLocalTensor = make("learning_rate", LR_TASKS)
        # Not a snapshot -- it accumulates -- but it belongs to the same
        # all-task channel: it needs the full per-task weight tensor, which only
        # the unsliced step-metric path carries.
        self._effective_sample_cnt: GlobalStateEffectiveSampleCnt = (
            GlobalStateEffectiveSampleCnt(
                rank=rank,
                tasks=all_task_names,
                global_batch_size=global_batch_size,
                report_interval=report_interval,
                state_scale=global_state_scale(),
            )
        )
        self._enabled: bool = bool(all_task_names)
        if not self._enabled:
            logger.warning(
                "no task names supplied; per-task snapshot metrics are disabled"
            )

    @property
    def metrics(self) -> List[Any]:
        """Metrics that report on their own (registered on the pipeline)."""
        if not self._enabled:
            return []
        return [self._loss, self._llw_weights, self._loss_scale, self._learning_rate]

    @property
    def step_metrics(self) -> List[Any]:
        """Metrics that must be fed the UNSLICED per-task tensors.

        Registered on the pipeline's ``_step_metrics`` list, which is the only
        one ``_update_ranking_metrics`` hands the full tensor to -- cls and reg
        metrics see an ``index_select``-ed subset.
        """
        if not self._enabled:
            return []
        return [self._effective_sample_cnt]

    def update(
        self,
        losses: torch.Tensor,
        named: Dict[str, Any],
        model: Any,
    ) -> None:
        """Snapshot this step's per-task tensors.

        ``losses`` is the post-LLW per-task loss vector the model returns; the
        LLW tensors come off ``named``. Every metric's ``update`` tolerates a
        missing value, so a model that does not run LLW simply reports nothing
        under those keys rather than crashing.
        """
        if not self._enabled:
            return
        self._loss.update(value=losses)
        self._llw_weights.update(value=named.get("llw_weights"))
        self._loss_scale.update(value=named.get("loss_scale"))
        self._learning_rate.update(value=_lr_tensor(model))


def _lr_tensor(model: Any) -> Optional[torch.Tensor]:
    """Stack the live dense/sparse LRs into one 2-row tensor, LR_TASKS order."""
    warmup = getattr(model, "_warmup", None)
    if warmup is None:
        return None
    rows: List[torch.Tensor] = []
    for attr in ("_dense_lr_buffer", "_sparse_lr_buffer"):
        buf = getattr(warmup, attr, None)
        if not isinstance(buf, torch.Tensor) or buf.numel() == 0:
            return None
        rows.append(buf.detach().float().reshape(-1)[:1])
    return torch.stack(rows)
