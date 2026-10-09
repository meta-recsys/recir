# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Base for self-compiling recir train modules (shared by MVAI and MVAI cloud).

A `RecirCompiledModuleBase` is the artifact a recir model family RETURNS as its
"model": it holds the (already-built) model + optimizer(s) and, on the first
forward, lazily compiles the train step; every subsequent forward just runs it.

Subclasses implement cell-specific hooks:
  * `_build_compiled_step(first_batch)` - build + compile the train step
    (GPU: patch EC/MoT + HybridParallel + InductorBackend via compile_hdp_step;
    TPU: GSPMDParallel + XLAGSPMDBackend via compile_gspmd_step).
  * `_run_step(compiled_step, batch)` - invoke it on a batch
    (GPU: compiled_step(impl, dense, sparse, *flat);
    TPU: run_step_and_sync(runner, model, dense, batch)).
  * `_drain_step(compiled_step)` - advance without admitting a new batch.
And optionally `_post_step()` (e.g. GPU warmup advance; default no-op).

The base owns lazy compilation, pipeline draining, and the once-only compiled
graph diagnostic dump so both cells share the same orchestration.
"""

import logging
from typing import Any, Callable, Optional

import torch.nn as nn
from pyre_extensions import none_throws
from recir.api import COMPILED_OBJECT_KEY
from recir.pipeline import is_pipeline_pending, PipelineConfig
from recir.training.diagnostics import dump_compiled_gm

logger: logging.Logger = logging.getLogger(__name__)


class RecirCompiledModuleBase(nn.Module):
    """nn.Module that lazily compiles its train step on the first forward."""

    _compiled_step: Optional[Callable[..., Any]]
    _did_dump_gm: bool

    def __init__(self, pipeline_config: Optional[PipelineConfig] = None) -> None:
        super().__init__()
        self._compiled_step = None
        self._did_dump_gm = False
        self._pipeline_config = pipeline_config or PipelineConfig.none()

    @property
    def pipeline_config(self) -> PipelineConfig:
        return self._pipeline_config

    @property
    def has_pending_work(self) -> bool:
        compiled_step = self._compiled_step
        if compiled_step is None:
            return False
        compiled_obj = getattr(compiled_step, "__dict__", {}).get(COMPILED_OBJECT_KEY)
        return bool(
            compiled_obj is not None and getattr(compiled_obj.gm, "has_pending", False)
        )

    def _build_compiled_step(self, first_batch: Any) -> Callable[..., Any]:
        """Build and compile the train step. Called once on first forward."""
        raise NotImplementedError

    def _run_step(
        self,
        compiled_step: Callable[..., Any],
        batch: Any,
    ) -> Any:
        """Invoke the compiled step on a batch. Cell-specific call convention."""
        raise NotImplementedError

    def _drain_step(self, compiled_step: Callable[..., Any]) -> Any:
        """Advance a pipeline without admitting another input batch."""
        raise NotImplementedError

    def _post_step(self) -> None:
        """Optional hook after one logical train step retires."""
        return None

    def _finish_step(self, compiled_step: Callable[..., Any], result: Any) -> Any:
        self._post_step()
        if not self._did_dump_gm:
            self._did_dump_gm = True
            try:
                dump_compiled_gm(compiled_step)
            except Exception as e:
                logger.warning(
                    "[recir gm dump] failed: %s (%s); continuing.",
                    e,
                    type(e).__name__,
                )
        return result

    def _get_or_build_compiled_step(self, batch: Any) -> Callable[..., Any]:
        if self._compiled_step is None:
            self._compiled_step = self._build_compiled_step(batch)
        return none_throws(self._compiled_step)

    def _return_or_finish_step(
        self,
        compiled_step: Callable[..., Any],
        result: Any,
    ) -> Any:
        if is_pipeline_pending(result):
            return result
        return self._finish_step(compiled_step, result)

    def forward(self, batch: Any) -> Any:
        compiled_step = self._get_or_build_compiled_step(batch)
        result = self._run_step(compiled_step, batch)
        return self._return_or_finish_step(compiled_step, result)

    def drain(self) -> Any:
        if not self._pipeline_config.enabled:
            raise RuntimeError("Cannot drain a model without an enabled pipeline")
        compiled_step = none_throws(self._compiled_step)
        result = self._drain_step(compiled_step)
        return self._return_or_finish_step(compiled_step, result)
