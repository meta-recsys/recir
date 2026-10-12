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
from typing import Any, Callable, cast, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.utils._pytree as pytree
from pyre_extensions import none_throws
from recir.compiler.runtime import (
    COMPILED_OBJECT_KEY,
    sync_device_state_to_torch,
    sync_named_state_to_torch,
)
from recir.pipeline import is_pipeline_pending, PipelineConfig
from recir.training.compile import in_graph_metrics_enabled
from recir.training.diagnostics import dump_compiled_gm
from recir.training.optim import WarmupController
from recir.training.tpu_checkpoint import HostStateBuffers

logger: logging.Logger = logging.getLogger(__name__)

# Per-step device->host state sync (sync_device_state_to_torch) skips state
# entries whose FQN contains any of these substrings. The on-device embedding
# tables (InGraphSparseArch's "_sharded_tables" / "_dp_tables" module dicts) are
# GBs each and are kept resident in the cached JAX device_state across steps, so
# mirroring them to host every step is O(table-size) waste (D2H copy + layout
# transpose) that dominates the step at large hash sizes. Dense params / LR /
# metric buffers don't match and still sync. Full sync (skip_fqns=None) is used
# before checkpointing/inspecting the tables (flush_host_state(full=True)).
_EMB_SYNC_SKIP_FQNS: Tuple[str, ...] = ("_sharded_tables", "_dp_tables")

# The only state the training loop needs mirrored host-ward between
# checkpoints: the warmup controller's LR buffers, which
# ``snapshot_metrics._lr_tensor`` reads for the learning_rate/* curves. Named
# after the module buffers (recir/training/compile.py LR_*_BUFFER); the
# WarmupController holds the same tensors under its own attribute names.
_LR_SYNC_ONLY_FQNS: Tuple[str, ...] = (
    "_lr_dense_buffer",
    "_lr_sparse_buffer",
    "_lr_iter_buffer",
)


def has_pending_work(compiled_step: Optional[Callable[..., Any]]) -> bool:
    """True while a cross-batch pipelined compiled step still holds in-flight
    steps."""
    if compiled_step is None:
        return False
    compiled_obj = getattr(compiled_step, "__dict__", {}).get(COMPILED_OBJECT_KEY)
    return bool(
        compiled_obj is not None and getattr(compiled_obj.gm, "has_pending", False)
    )


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
        return has_pending_work(self._compiled_step)

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


class RecirGSPMDCompiledModuleBase(RecirCompiledModuleBase):
    """GSPMD/TPU counterpart of the compiled module: per-step device->host sync
    policy, host-state flushes, checkpoint restore hooks and the DCP
    ``Stateful`` view (``make_checkpointable``).

    Subclasses build the step (``_build_compiled_step``) and its arguments
    (``_step_args``), and set ``_model``, ``_dense_opt``, ``_emb_opt`` (the
    in_graph embedding optimizer, or None), ``_warmup``, ``_trace_batch`` and
    ``_num_processes``.
    """

    _model: nn.Module
    _dense_opt: torch.optim.Optimizer
    _emb_opt: Optional[torch.optim.Optimizer]
    _warmup: WarmupController
    _trace_batch: Any
    _num_processes: int
    _host_state_buffers: Any = None

    def _step_args(self, batch: Any) -> Tuple[Any, ...]:
        """``(model, *optimizers, *flat_batch)`` for the compiled step."""
        raise NotImplementedError

    @property
    def dense_opt(self) -> torch.optim.Optimizer:
        return self._dense_opt

    def flat_step_args(self, batch: Any) -> List[Any]:
        return pytree.arg_tree_leaves(*self._step_args(batch))

    def _run_step(
        self, compiled_step: Any, batch: Any
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        result: Tuple[torch.Tensor, Dict[str, torch.Tensor]] = compiled_step(
            *self._step_args(batch)
        )
        if self._emb_opt is None:
            sync_device_state_to_torch(compiled_step)
        return result

    @property
    def in_graph_metrics(self) -> bool:
        return in_graph_metrics_enabled(self._model)

    def flush_host_state(self, full: bool = False) -> None:
        """Mirror device-resident state back to torch for host consumers (e.g.
        before checkpointing).

        By default the GB-scale embedding tables are skipped (kept resident on
        device -- see _EMB_SYNC_SKIP_FQNS); pass full=True before checkpointing
        to also mirror them (skip_fqns=None). The table host params must already
        have real storage (see tpu_checkpoint.HostStateBuffers) or the
        copy_ is a silent no-op.
        """
        compiled_step = self._compiled_step
        if compiled_step is not None:
            skip_fqns = None if full else _EMB_SYNC_SKIP_FQNS
            sync_device_state_to_torch(compiled_step, skip_fqns=skip_fqns)

    def sync_lr_to_host(self) -> None:
        """Refresh only the LR buffers ``snapshot_metrics._lr_tensor`` reads."""
        compiled_step = self._compiled_step
        if compiled_step is not None:
            sync_named_state_to_torch(compiled_step, _LR_SYNC_ONLY_FQNS)

    def enable_output_pipelining(self) -> bool:
        manager = self._state_manager()
        if manager is None:
            return False
        manager.keep_output_jax = True
        return True

    def materialize_output(self, output: Any) -> Any:
        manager = self._state_manager()
        if manager is None:
            return output
        return manager.materialize_output(output)

    def _state_manager(self) -> Any:
        compiled_step = self._compiled_step
        if compiled_step is None:
            return None
        compiled_obj = cast(Any, compiled_step).__dict__.get(COMPILED_OBJECT_KEY)
        return None if compiled_obj is None else compiled_obj.state_manager

    def prepare_for_restore(self) -> None:
        """Compile, then build the run's host buffers so dcp.load can write into them."""
        if self._compiled_step is None:
            self._compiled_step = self._build_compiled_step(self._trace_batch)
        self._host_buffers()

    def _host_buffers(self) -> Any:
        buffers = self._host_state_buffers
        if buffers is None:
            compiled_obj = cast(Any, self._compiled_step).__dict__.get(
                COMPILED_OBJECT_KEY
            )
            if compiled_obj is None:
                raise RuntimeError(
                    "host state buffers need a compiled step; call "
                    "prepare_for_restore after the force-compile"
                )
            buffers = HostStateBuffers(
                compiled_obj.flat_state,
                compiled_obj.state_manager.state_fqns or [],
                self._model,
                [self._dense_opt, self._emb_opt],
                _EMB_SYNC_SKIP_FQNS,
            )
            buffers.build()
            self._host_state_buffers = buffers
        return buffers

    def finalize_restore(self) -> None:
        """Seed device state from the freshly loaded host buffers."""
        compiled_step = self._compiled_step
        if compiled_step is None:
            return
        compiled_obj = cast(Any, compiled_step).__dict__.get(COMPILED_OBJECT_KEY)
        if compiled_obj is None:
            return
        compiled_obj.state_manager.prefill_device_state_from_host(
            compiled_obj.gm, compiled_obj.flat_state
        )

    def make_checkpointable(self) -> Any:
        """Return a DCP Stateful view of this module (see tpu_checkpoint.py).

        Duck-typed hook: mvai_cloud's CheckpointManager uses this if present, so
        the recir/JAX-specific checkpoint logic stays out of the shared framework.
        """
        from recir.embedding.in_graph import DP_TABLE_PREFIX, SHARDED_TABLE_PREFIX
        from recir.training.tpu_checkpoint import (
            gather_shard_order,
            RecirCheckpointable,
        )

        return RecirCheckpointable(
            self,
            num_processes=self._num_processes,
            sharded_prefixes=(SHARDED_TABLE_PREFIX,),
            table_prefixes=(SHARDED_TABLE_PREFIX, DP_TABLE_PREFIX),
            warmup=self._warmup,
            host_buffers=self._host_buffers,
            dense_optimizer=self._dense_opt,
            embedding_optimizer=self._emb_opt,
            shard_order=(
                gather_shard_order(self._num_processes)
                if self._num_processes > 1
                else None
            ),
        )
