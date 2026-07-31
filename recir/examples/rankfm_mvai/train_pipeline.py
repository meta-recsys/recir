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
    ``RankingTrainPipeline`` to reuse its metric construction
    (``__init__``), its ``_update_ranking_metrics(ret)`` cls/reg slicing, and
    its ``.metrics`` property unchanged. ``progress`` is overridden to run the
    self-compiling model wrapper (instead of the eager fwd/bwd/opt.step), rebuild
    the ``{prediction, label, weight}`` ``ret`` dict from the wrapper's return,
    and drive the JAX profiler trace over the configured step window (ported from
    the bespoke rankfm loop so run_tpu_bench.sh's xprof trace fetch keeps working).
  * ``RecirXLATrainPipelineProvider`` reuses mvai_cloud's
    ``RankingTrainPipelineProvider`` to construct the cls/reg/step metric
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
import torch.distributed as dist
import torch.nn as nn
from hammer.v3.models.udd_fm_v3.recir_core.trace_sample import flatten_batch
from minimal_viable_ai.core.trainer.train_pipeline import RecTrainPipelineProvider

from .loss_metric import WindowLoss
from .xsurface_helpers import _log_hbm_memory

# minimal_viable_ai is provided at RUNTIME via the Docker image (Dockerfile
# rsyncs fbcode/mvai_cloud/minimal_viable_ai -- the lightweight RankingSGD* API
# this example targets). pyre type-checks against the canonical buck package
# //minimal_viable_ai instead (a different, RankingMTML* API where these names
# don't exist), so under the type checker we stub them as Any (runtime-only dep).
if TYPE_CHECKING:
    RankingMetricDef = Any
    RankingTrainPipeline = Any
    RankingTrainPipelineProvider = Any

    from .model_family import RecirXLACompiledModule
else:
    from minimal_viable_ai.core.ranking_common.train_pipeline import (
        RankingMetricDef,
        RankingSGDTrainPipeline as RankingTrainPipeline,
        RankingSGDTrainPipelineProvider as RankingTrainPipelineProvider,
    )

logger: logging.Logger = logging.getLogger(__name__)


def _build_loss_metric(*metric_lists: Any) -> WindowLoss:
    """Build a WindowLoss whose rank + report cadence match the sibling metrics
    (so the loss scalar is written at the same interval as NE/CTR/MSE)."""
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    # First sibling metric (across lists, in order) with a report interval; the
    # `is not None` check keeps interval==1 a valid hit so selection is
    # deterministic regardless of which list is scanned first.
    report_interval = 1
    for lst in metric_lists:
        for m in lst or []:
            ri = getattr(m, "_report_interval", None)
            if ri is not None:
                report_interval = int(ri)
                break
        else:
            continue
        break
    return WindowLoss(rank=rank, report_interval=report_interval)


def _cls_task_names(cls_metrics: Optional[List[Any]]) -> List[str]:
    """Task names for the classification (NE) metrics, in cls_task_indices order
    (each NE metric exposes its per-task names as `_tasks`)."""
    names: List[str] = []
    for m in cls_metrics or []:
        names.extend(getattr(m, "_tasks", None) or [])
    return names


def _loss_series(
    losses: torch.Tensor, task_names: List[str], cls_indices: Any
) -> Dict[str, torch.Tensor]:
    """Build the {series_key: scalar-tensor} dict for WindowLoss: the aggregate
    loss plus one entry per NE task. `losses` is the model's per-task loss vector
    (same task order as `prediction`), so index_select(cls_indices) aligns with
    the NE tasks. Values stay 0-dim tensors -- WindowLoss accumulates them on
    device and only syncs to Python scalars at report cadence (Citrine C6: no
    per-step .item()/.tolist() device-to-host sync in the training loop)."""
    series: Dict[str, torch.Tensor] = {"loss": losses.detach().sum()}
    if task_names and cls_indices is not None:
        try:
            per_task = losses.detach().index_select(0, cls_indices)
            if per_task.numel() == len(task_names):
                for i, name in enumerate(task_names):
                    series[f"loss/{name}"] = per_task[i]
        except Exception as e:  # never break training for a metric
            logger.warning("[loss-metric] per-task loss skipped: %r", e)
    return series


def _llw_series(
    named: Dict[str, Any], task_names: List[str]
) -> Dict[str, torch.Tensor]:
    """LLW `s` (loss_scale) + derived weights as TB scalars, one series per task
    (plus s_mean/min/max aggregates over all tasks). Per-task series are keyed by
    task name when the per-task count matches `task_names` (mirrors `_loss_series`,
    so the LLW curves cross-reference the loss curves and survive task reordering)
    and fall back to positional index otherwise. Values stay 0-dim tensors so
    WindowLoss accumulates them on device and only syncs to Python scalars at
    report cadence (Citrine C6: no per-step .item()/.tolist() device-to-host sync
    in the training loop)."""
    series: Dict[str, torch.Tensor] = {}

    def _label(i: int, n: int) -> str:
        return task_names[i] if len(task_names) == n else str(i)

    s = named.get("loss_scale")
    if isinstance(s, torch.Tensor) and s.numel() > 0:
        sf = s.detach().float().reshape(-1)
        series["llw/s_mean"] = sf.mean()
        series["llw/s_min"] = sf.min()
        series["llw/s_max"] = sf.max()
        n = sf.numel()
        for i in range(n):
            series[f"llw/s_{_label(i, n)}"] = sf[i]
    w = named.get("llw_weights")
    if isinstance(w, torch.Tensor) and w.numel() > 0:
        wf = w.detach().float().reshape(-1)
        n = wf.numel()
        for i in range(n):
            series[f"llw/w_{_label(i, n)}"] = wf[i]
    return series


def _hstu_series(named: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """MoT mt_aux (ue.ie) total loss as a TB scalar. Value stays a 0-dim tensor so
    WindowLoss syncs it to a Python scalar only at report cadence (Citrine C6)."""
    series: Dict[str, torch.Tensor] = {}
    m = named.get("mt_loss_total")
    if isinstance(m, torch.Tensor) and m.numel() > 0:
        series["mt_loss/total"] = m.detach().float().reshape(-1)[0]
    return series


class RecirXLATrainPipeline(RankingTrainPipeline):
    """Self-compiling-model pipeline + udd_fm-equivalent ranking metrics.

    Reuses RankingTrainPipeline's metric construction (``__init__``), its
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
        # pyre-ignore[19]: base RankingTrainPipeline is a runtime-only import
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
        self._loss_metric: WindowLoss = _build_loss_metric(
            step_metrics, cls_metrics, reg_metrics
        )
        self._loss_task_names: List[str] = _cls_task_names(cls_metrics)
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

        # Rebuild the ret dict RankingTrainPipeline._update_ranking_metrics
        # expects; the metrics were constructed on torch.device("cpu") to match
        # these host CPU tensors. weight=0 keeps padded rows out of NE/CTR/MSE;
        # num_real_candidates keeps variable-step x-axes and QPS padding-invariant.
        ret: Dict[str, Any] = {
            "prediction": named["prediction"],
            "label": named["label"],
            "weight": named["weight"],
            # Real (pre-pad) per-host candidate count -> variable-step metrics
            # x-axis stride (else it falls back to the PADDED labels.size(1)).
            # Computed from the raw batch so the model needn't emit it.
            "num_real_candidates": int(batch.num_candidates.sum()),
        }
        self._update_ranking_metrics(ret)
        # Log the training loss to TB (aggregate + per NE task + LLW s/weights + MT
        # mt_loss); the base pipeline only reports prediction-derived NE/CTR/MSE.
        _series: Dict[str, Any] = _loss_series(
            losses, self._loss_task_names, self._cls_indices
        )
        _series.update(_llw_series(named, self._loss_task_names))
        _series.update(_hstu_series(named))
        self._loss_metric.update(values=_series)
        return losses, ret

    @property
    def metrics(self) -> List[Any]:
        return list(super().metrics) + [self._loss_metric]

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

    Reuses mvai_cloud's ``RankingTrainPipelineProvider`` to construct the
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
        # Annotated Any: RankingTrainPipelineProvider is a runtime-only import
        # (stubbed Any for pyre -- see the module-level TYPE_CHECKING block).
        # pyrefly: ignore [bad-instantiation]
        self._metrics_provider: Any = RankingTrainPipelineProvider(
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


class EagerCPUTrainPipeline(RankingTrainPipeline):
    """CPU-eager counterpart of RecirXLATrainPipeline (same model, no compile).

    This is the matched-config reference for the TPU run: it drives the EXACT
    same self-compiling model wrapper that RecirXLATrainPipeline does, but never
    calls ``wrapper(batch)`` (which would lazily compile the GSPMD GM). Instead it
    runs the wrapper's underlying raw model eagerly on CPU, so the numerics line
    up axis-for-axis with the TPU run:

      * forward/backward: ``raw_model(*flatten_batch(batch))`` then
        ``loss.backward()`` -- the same forward the compiled GM traces.
      * embeddings: the DMP ``EmbeddingComputeKernel.FUSED`` Adagrad fires inside
        ``loss.backward()`` (CPURankFMBackend.setup takes the DMP+FUSED path
        because RecTrainer initializes a gloo PG). The TPU run uses the SAME
        host DMP-FUSED Adagrad (--embedding_backend cpu), so embeddings match.
      * dense params: a PLAIN ``torch.optim.AdamW`` (capturable=False) with the
        SAME params/betas/eps/weight_decay as the wrapper's ``TensorLRAdamW``.
        TensorLRAdamW needs ``capturable=True`` for its 0-d *tensor* LR, which
        CPU rejects ("params must be on cuda/xla/..."); a tensor LR also can't run
        under ``capturable=False`` (``addcdiv_`` needs a scalar). The Adam UPDATE
        MATH is identical, so we snapshot the warmup LR buffer to a Python scalar
        each step -> byte-equivalent dense update on CPU.
      * LR warmup + grad clip: the warmup controller mutates the SAME 0-d LR
        buffer in place each step; we read it for the scalar LR, and replicate
        TensorLRAdamW.step's ``clip_grad_value_`` on the dense params.

    The metric path (``_update_ranking_metrics``) is reused verbatim, so the
    reported per-task NE/CTR/MSE are the SAME torchrec metrics the TPU run emits.
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
        # pyre-ignore[19]: see RecirXLATrainPipeline.__init__ -- runtime-only base.
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
        # The wrapper compiles lazily on its first __call__; we deliberately never
        # call it. Reach through to the raw model + dense opt + warmup it built
        # (backend.setup/DMP already applied, dense AdamW + warmup already wired).
        wrapper: Any = model
        self._raw_model: nn.Module = wrapper._model
        self._warmup: Any = wrapper._warmup

        # Rebuild the dense optimizer as a plain CPU AdamW (capturable=False,
        # scalar LR). See the class docstring for why TensorLRAdamW's tensor-LR
        # capturable=True path can't run on CPU. Same params + hyperparameters ->
        # identical Adam update.
        tensor_lr_opt: Any = wrapper._dense_opt
        self._dense_params: List[nn.Parameter] = [
            p for group in tensor_lr_opt.param_groups for p in group["params"]
        ]
        defaults = tensor_lr_opt.defaults
        self._dense_opt: torch.optim.Optimizer = torch.optim.AdamW(
            self._dense_params,
            lr=defaults["lr"],
            betas=defaults["betas"],
            eps=defaults["eps"],
            weight_decay=defaults["weight_decay"],
            amsgrad=defaults.get("amsgrad", False),
            capturable=False,
            # Citrine C2: multi-tensor AdamW. Same update math (parity preserved);
            # the dense params are non-sparse (embeddings run DMP-FUSED Adagrad
            # in-backward, handled separately).
            foreach=True,
        )
        # Reuse the pre-trace state already allocated for TensorLRAdamW. The
        # wrapper is never compiled on this path, so transfer ownership to the
        # CPU optimizer instead of retaining one copy and lazily creating another.
        self._dense_opt.state.update(tensor_lr_opt.state)
        tensor_lr_opt.state.clear()
        # The warmup controller mutates this 0-d buffer in place (fill_) every
        # step; it is the SAME tensor TensorLRAdamW bound as its tensor LR. Read
        # it each step for the current scalar LR.
        self._lr_buffer: torch.Tensor = tensor_lr_opt.param_groups[0]["lr"]
        # Match TensorLRAdamW.step's optional grad-value clip on the dense params.
        self._max_grad: Optional[float] = getattr(tensor_lr_opt, "_max_grad", None)
        self._loss_metric: WindowLoss = _build_loss_metric(
            step_metrics, cls_metrics, reg_metrics
        )
        self._loss_task_names: List[str] = _cls_task_names(cls_metrics)
        self._raw_model.train(True)

    def progress(
        self, dataloader_iter: Iterator[Any]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        batch = next(dataloader_iter)
        self._dense_opt.zero_grad(set_to_none=True)
        # Forward + backward eagerly. loss.backward() fires the DMP-FUSED
        # embedding Adagrad in-backward (matching the TPU run's host embeddings).

        # This eager path has no cross-rank dense-gradient reduction, so its loss
        # numerator is local. Use the local candidate count to match its scope.
        flat_args = list(flatten_batch(batch))
        flat_args[-1] = batch.num_candidates.sum()
        all_losses, named = self._raw_model(*flat_args)
        loss = all_losses.sum()
        loss.backward()
        # Dense step: clip (matching TensorLRAdamW.step) then AdamW with the
        # CURRENT warmup LR as a scalar; then advance the warmup buffer for the
        # next step (the wrapper's _post_step ordering: step uses current LR, then
        # advance).
        if self._max_grad is not None:
            torch.nn.utils.clip_grad_value_(self._dense_params, self._max_grad)
        lr_val = float(self._lr_buffer.item())
        for group in self._dense_opt.param_groups:
            group["lr"] = lr_val
        self._dense_opt.step()
        self._warmup.step()

        ret: Dict[str, Any] = {
            "prediction": named["prediction"].detach(),
            "label": named["label"].detach(),
            "weight": named["weight"].detach(),
            # Real (pre-pad) per-host candidate count -> variable-step metrics
            # x-axis stride (else PADDED labels.size(1)); from the raw batch.
            "num_real_candidates": int(batch.num_candidates.sum()),
        }
        self._update_ranking_metrics(ret)
        # Log the training loss to TB (aggregate + per NE task + LLW s/weights + MT
        # mt_loss); parity with RecirXLATrainPipeline. all_losses is the per-task vector.
        _series: Dict[str, Any] = _loss_series(
            all_losses, self._loss_task_names, self._cls_indices
        )
        _series.update(_llw_series(named, self._loss_task_names))
        _series.update(_hstu_series(named))
        self._loss_metric.update(values=_series)
        return all_losses.detach(), ret

    @property
    def metrics(self) -> List[Any]:
        return list(super().metrics) + [self._loss_metric]

    def on_train_end(self) -> None:
        logger.info("EagerCPUTrainPipeline: training complete.")


class EagerCPUTrainPipelineProvider(
    RecTrainPipelineProvider[Any, Tuple[torch.Tensor, Dict[str, Any]]]
):
    """Returns the CPU-eager pipeline + the udd_fm-equivalent metric set.

    Identical wiring to RecirXLATrainPipelineProvider (same metric construction
    via mvai_cloud's RankingTrainPipelineProvider); only the pipeline class
    differs (EagerCPUTrainPipeline runs the raw model eagerly, no recir compile).
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
        # Annotated Any: RankingTrainPipelineProvider is a runtime-only import
        # (stubbed Any for pyre -- see the module-level TYPE_CHECKING block).
        # pyrefly: ignore [bad-instantiation]
        self._metrics_provider: Any = RankingTrainPipelineProvider(
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
    ) -> EagerCPUTrainPipeline:
        cpu_device = torch.device("cpu")
        base_pipeline = self._metrics_provider.create_train_pipeline(
            model, optimizer, cpu_device, pg
        )
        return EagerCPUTrainPipeline(
            model,
            optimizer,
            cpu_device,
            cls_metrics=base_pipeline._cls_metrics,
            reg_metrics=base_pipeline._reg_metrics,
            step_metrics=base_pipeline._step_metrics,
            cls_task_indices=self._cls_task_indices,
            reg_task_indices=self._reg_task_indices,
        )
