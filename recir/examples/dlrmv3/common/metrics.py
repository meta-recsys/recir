# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Metric helpers around generative-recommenders' ``MetricsLogger``.

NE / accuracy / GAUC for classification tasks, MSE / MAE for regression, all
torchrec metric computations; TensorBoard logging when ``tb_dir`` is set.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from generative_recommenders.dlrm_v3.utils import MetricsLogger
from generative_recommenders.modules.dlrm_hstu import DlrmHSTUConfig


def make_metrics(
    hstu_config: DlrmHSTUConfig,
    batch_size: int,
    device: torch.device,
    rank: int,
    tb_dir: str,
    window_size: int = 1000,
) -> MetricsLogger:
    return MetricsLogger(
        multitask_configs=hstu_config.multitask_configs,
        batch_size=batch_size,
        window_size=window_size,
        device=device,
        rank=rank,
        tensorboard_log_path=tb_dir,
    )


def update_metrics(
    metrics: MetricsLogger,
    named: Dict[str, torch.Tensor],
    num_candidates: int,
    mode: str,
) -> None:
    # Dummy rows and padded candidates carry weight 0 (no NE/accuracy
    # contribution) and all-negative labels (single-class sessions, which GAUC
    # skips), so no host-side filtering (and no device sync) is needed.
    preds = named["prediction"]
    num_rows = preds.shape[-1] // num_candidates
    metrics.update(
        predictions=preds,
        labels=named["label"],
        weights=named["weight"],
        num_candidates=torch.full(
            (num_rows,), num_candidates, dtype=torch.int64, device=preds.device
        ),
        mode=mode,
    )


def log_metrics(
    metrics: MetricsLogger,
    mode: str,
    losses: Optional[torch.Tensor] = None,
    task_names: Optional[list[str]] = None,
) -> Dict[str, float]:
    if metrics.tb_logger is None:
        return metrics.compute(mode)
    extra = None
    if losses is not None and task_names is not None:
        extra = {"losses": {name: losses[i] for i, name in enumerate(task_names)}}
    return metrics.compute_and_log(mode, additional_logs=extra)
