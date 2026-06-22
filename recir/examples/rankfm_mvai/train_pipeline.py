#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""recir XLA-GSPMD train pipeline for the mvai_cloud-driven RankFM TPU example.

Replaces mvai_cloud's eager ``SimpleSGDTrainPipeline`` (fwd/bwd/step) with a
recir-compiled train_step. The provider:

  1. Reads the ``RecirCompileContext`` that RankFMMvaiModelFamily stashed on
     the model (it already ran backend.get_override + backend.setup).
  2. Builds a ``GSPMDParallel`` and compiles the rankfm train_step via
     ``recir.examples.rankfm.train._compile_with_xsurface_dispatchers`` (which
     drives ``recir.trace`` -> ``recir.compile_from_gm(XLAGSPMDBackend())`` ->
     ``make_compiled_runner`` with the PreprocessorDispatcher +
     SparseArchDispatcher pair and the PaddingRegionTransform).

``RecirXLATrainPipeline.progress(dataloader_iter)`` then: ``next(dataloader_iter)`` ->
run the compiled step on the extracted tensor args -> ``sync_device_state_to_torch``
-> return ``(loss, ret)``. The compiled rankfm step does fwd/bwd/opt.step
internally (and returns None, matching rankfm), so progress returns a
placeholder loss tensor that RecTrainer only uses for ``loss.cpu()`` logging.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, Optional, Tuple

import torch
import torch.nn as nn
from minimal_viable_ai.core.trainer.train_pipeline import (
    RecTrainPipeline,
    RecTrainPipelineProvider,
)

logger: logging.Logger = logging.getLogger(__name__)


class RecirXLATrainPipeline(RecTrainPipeline[Any, Tuple[torch.Tensor, Dict[str, Any]]]):
    """Thin pipeline: the self-compiling model owns compile + run + sync."""

    def __init__(self, model: nn.Module) -> None:
        self._model = model

    def progress(
        self, dataloader_iter: Iterator[Any]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        batch = next(dataloader_iter)
        # The wrapper lazily compiles on the first call, then runs fwd -> backward
        # (fires the SparseCore sparse Adagrad off-GM) -> dense opt.step on TPU,
        # and syncs device state. Its (device-resident) return is ignored here;
        # placeholder loss for RecTrainer's host-side loss.cpu() logging.
        self._model(batch)
        return torch.zeros(1), {}

    def on_train_end(self) -> None:
        logger.info("RecirXLATrainPipeline: training complete.")


class RecirXLATrainPipelineProvider(
    RecTrainPipelineProvider[Any, Tuple[torch.Tensor, Dict[str, Any]]]
):
    """Returns the thin pipeline; all compile work lives in the model wrapper."""

    # pyre-ignore
    def create_train_pipeline(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        # `device` is part of the RecTrainPipelineProvider interface (the
        # mvai_cloud RecTrainer passes it by keyword); recir derives device
        # placement from the RecirCompileContext / GSPMDParallel, so it is
        # accepted here but unused.
        device: Optional[torch.device] = None,
        pg: Optional[Any] = None,
    ) -> RecirXLATrainPipeline:
        return RecirXLATrainPipeline(model)
