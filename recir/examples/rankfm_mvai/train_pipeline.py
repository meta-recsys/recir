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

``RecirXLATrainPipeline.progress(dataiter)`` then: ``next(dataiter)`` ->
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
from recir.api import sync_device_state_to_torch
from recir.backends import XLAGSPMDBackend
from recir.distributed.parallel_mode import GSPMDParallel

# Reuse rankfm's compile driver + arg extractor verbatim (do NOT copy). The
# nested train_step_fn inside _build_compiled_train_step is not importable, so
# it is reproduced below byte-for-byte.
from recir.examples.rankfm.train import (
    _compile_with_xsurface_dispatchers,
    _extract_train_step_args,
)

from .model_family import RecirCompileContext

logger: logging.Logger = logging.getLogger(__name__)


# pyre-ignore -- mirror of rankfm/train.py:336 train_step_fn (nested there).
def _train_step_fn(
    mod: nn.Module,
    opt: torch.optim.Optimizer,
    ro_float_features: torch.Tensor,
    nro_float_features: torch.Tensor,
    ro_id_list_values: torch.Tensor,
    ro_id_list_lengths: torch.Tensor,
    nro_id_list_values: torch.Tensor,
    nro_id_list_lengths: torch.Tensor,
    unary_values: torch.Tensor,
    unary_lengths: torch.Tensor,
    num_candidates: torch.Tensor,
    supervision_label_t: torch.Tensor,
    supervision_weight_t: torch.Tensor,
    all_labels: Dict[str, torch.Tensor],
) -> None:
    all_losses, _ = mod(
        ro_float_features,
        nro_float_features,
        ro_id_list_values,
        ro_id_list_lengths,
        nro_id_list_values,
        nro_id_list_lengths,
        unary_values,
        unary_lengths,
        num_candidates,
        supervision_label_t,
        supervision_weight_t,
        all_labels,
    )
    loss: torch.Tensor = all_losses.sum()
    loss.backward()
    opt.step()


class RecirXLATrainPipeline(RecTrainPipeline):
    """RecTrainPipeline whose step is a recir XLA-GSPMD compiled train_step."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_step: Any,
    ) -> None:
        self._model = model
        self._optimizer = optimizer
        self._train_step = train_step
        self._model.train(True)

    def progress(self, dataiter: Iterator[Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        batch = next(dataiter)
        # Compiled rankfm step: fwd -> backward -> opt.step on TPU (recir).
        self._train_step(self._model, self._optimizer, *_extract_train_step_args(batch))
        sync_device_state_to_torch(self._train_step)
        # rankfm's train_step_fn returns None; RecTrainer only needs a tensor
        # to call loss.cpu() on for logging and a ret dict for metrics.
        loss = torch.zeros(1)
        ret: Dict[str, Any] = {}
        return loss, ret

    def on_train_end(self) -> None:
        logger.info("RecirXLATrainPipeline: training complete.")


class RecirXLATrainPipelineProvider(RecTrainPipelineProvider):
    """Builds the recir-compiled train_step from the model's RecirCompileContext."""

    def create_train_pipeline(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        pg: Optional[Any] = None,
    ) -> RecTrainPipeline:
        ctx: RecirCompileContext = getattr(model, "_recir_compile_ctx", None)
        assert ctx is not None, (
            "model._recir_compile_ctx missing -- RankFMMvaiModelFamily."
            "create_train_module_and_optimizer must run before "
            "RecirXLATrainPipelineProvider.create_train_pipeline."
        )

        parallel_mode = GSPMDParallel(
            "replicate",
            loss_reduction="sum",
            num_ranks=ctx.num_ranks,
            num_processes=ctx.num_processes,
        )

        train_step = _compile_with_xsurface_dispatchers(
            model,
            optimizer,
            ctx.trace_batch,
            ctx.overrides,
            _train_step_fn,
            parallel_mode=parallel_mode,
            backend=XLAGSPMDBackend(),
            region_transforms=ctx.region_transforms,
        )

        return RecirXLATrainPipeline(model, optimizer, train_step)
