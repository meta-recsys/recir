# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Step helpers shared by the DLRMv3 GPU and TPU trainers."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from recir.examples.dlrmv3.common.data import TokenBudgetBatch


def flatten_batch(
    batch: TokenBudgetBatch, *, global_loss_denom: bool
) -> Tuple[torch.Tensor, ...]:
    """``flatten_batch`` for ``recir.training.compile``. DlrmV3TrainModule
    normalizes its losses per row, so the loss-denominator mode is unused."""
    return batch.flat()


def eval_step(
    mod: nn.Module, *batch: Any
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Forward only, in eval mode: no dropout, autograd, optimizer or warmup
    updates. The mode flips around the call so a train graph traced or run
    with the same module is unaffected."""
    was_training = mod.training
    mod.eval()
    try:
        with torch.no_grad():
            losses, named = mod(*batch)
    finally:
        mod.train(was_training)
    return losses.detach(), {k: v.detach() for k, v in named.items()}
