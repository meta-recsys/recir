#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""recir XLA-GSPMD train pipeline for the mvai_cloud-driven RankFM TPU example.

The self-compiling model wrapper (``RecirXLACompiledModule``) owns compile,
forward, backward, dense step and device sync; calling it returns detached
``(losses, named_tensors)``. This pipeline consumes that and reports the SAME
ranking metric set as the mvai_cloud udd_fm reference.

``RecirXLATrainPipeline`` subclasses mvai_cloud's ``RankingTrainPipeline`` to
reuse its metric construction and slicing unchanged, overriding only
``progress`` to drive the wrapper instead of an eager fwd/bwd/step, rebuild the
``ret`` dict from its return, and run the JAX profiler over the configured step
window. Metrics are built on CPU because recir returns CPU tensors.
"""

from __future__ import annotations

import logging
import os
from typing import Any, cast, Dict, Iterator, List, Optional, Tuple, TYPE_CHECKING

import jax
import torch
import torch.distributed as dist
import torch.nn as nn
from hammer.v3.models.udd_fm_v3.parity.step_probe import StepProbe
from minimal_viable_ai.core.trainer.train_pipeline import RecTrainPipelineProvider

from .eager_step import EagerCPUStep
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


def _lr_series(model: Any) -> Dict[str, torch.Tensor]:
    """The live learning rates as TB scalars."""
    series: Dict[str, torch.Tensor] = {}
    warmup = getattr(model, "_warmup", None)
    if warmup is None:
        return series
    for attr, key in (
        ("_dense_lr_buffer", "lr/dense"),
        ("_sparse_lr_buffer", "lr/sparse"),
    ):
        buf = getattr(warmup, attr, None)
        if isinstance(buf, torch.Tensor) and buf.numel() > 0:
            series[key] = buf.detach().float().reshape(-1)[0]
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
        _series.update(_lr_series(self._model))
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

    The matched-config reference for the TPU run: it drives the EXACT same model
    wrapper, but never calls ``wrapper(batch)`` (which would lazily compile the
    GSPMD GM) and instead runs the underlying raw model eagerly, so the numerics
    line up axis for axis.

    Embeddings step inside ``loss.backward()`` under the same host DMP-FUSED
    Adagrad the TPU run uses. Dense params go through a plain ``AdamW`` rather than
    the wrapper's ``TensorLRAdamW``, whose 0-d tensor LR needs ``capturable=True``
    and so cannot run on CPU -- the update MATH is identical, so the warmup LR
    buffer is snapshotted to a scalar each step. Grad clipping replicates what
    ``TensorLRAdamW.step`` does.

    The metric path is reused verbatim, so the reported NE/CTR/MSE are the same
    torchrec metrics the TPU run emits.
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
        light_loss_norm: bool = False,
        grad_accum_steps: int = 1,
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
        #
        # The step itself lives in eager_step.py, which imports neither
        # mvai_cloud nor jax, so the parity test can drive this exact code on a
        # devserver instead of reproducing it.
        self._step: EagerCPUStep = EagerCPUStep.from_wrapper(
            model,
            light_loss_norm=light_loss_norm,
            grad_accum_steps=grad_accum_steps,
        )
        self._grad_accum_steps: int = self._step.grad_accum_steps
        self._raw_model: nn.Module = self._step.raw_model
        # Inert unless PARITY_EMIT_DIGEST is set, and latched after one step.
        self._probe: StepProbe = StepProbe(self._raw_model)
        self._loss_metric: WindowLoss = _build_loss_metric(
            step_metrics, cls_metrics, reg_metrics
        )
        self._loss_task_names: List[str] = _cls_task_names(cls_metrics)
        self._raw_model.train(True)

    def progress(
        self, dataloader_iter: Iterator[Any]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """One optimizer step, over ``self._grad_accum_steps`` micro-batches.

        Getting the loss scale right is the subtle part, and the two denominator
        modes need different treatment.

        * ``light_loss_norm``: the denominator stays None, so
          ``_candidate_axis_mean`` falls back to ``torch.mean`` and each
          micro-batch arrives already normalized by its OWN candidate count.
          The light leg runs those same per-rank means on K ranks and lets DDP
          ALL-REDUCE-MEAN the gradients, giving ``(1/K) * sum_r grad(mean_r)``
          -- every rank weighted equally regardless of how many candidates it
          drew. So scale by ``1/K``, nothing else.

          It is tempting to weight by ``n_i / n_total`` instead, on the reasoning
          that the terms then telescope to ``sum_total / n_total``. They do --
          but that is the gradient of ONE GLOBAL MEAN over the concatenated
          batch, which is not what DDP computes. The two agree only when every
          ``n_i`` is equal; on this dump adjacent batches differ by ~19%
          (2254 vs 2681 candidates). The discrepancy compounds every step and
          surfaces as slow monotonic NE divergence, invisible to any
          single-forward check. This code shipped with that bug once.

        * otherwise: pass ``n_total`` as the denominator, which already
          normalizes every micro-batch by the group, so no extra weighting.

        Both reduce to today's behaviour when ``grad_accum_steps == 1``.
        Pinned by ``tests/test_grad_accum.py`` against an explicit DDP
        all-reduce-mean reference.

        The numerics of the step live in ``eager_step.EagerCPUStep``; what is
        left here is the metric / digest instrumentation that needs the
        mvai_cloud base class.
        """
        self._step.zero_grad()
        self._probe.arm()
        micro_batches: List[Any] = [
            next(dataloader_iter) for _ in range(self._grad_accum_steps)
        ]
        ret: Dict[str, Any] = {}
        self._probe.emit_weights_once()

        def _before_forward(batch: Any) -> None:
            self._probe.on_micro_batch(batch)

        def _after_forward(all_losses: torch.Tensor) -> None:
            self._probe.on_loss(all_losses)

        def _after_backward(named: Dict[str, Any], n_i: int) -> None:
            nonlocal ret
            ret = {
                "prediction": named["prediction"].detach(),
                "label": named["label"].detach(),
                "weight": named["weight"].detach(),
                "num_real_candidates": n_i,
            }
            self._update_ranking_metrics(ret)

        all_losses, named = self._step.run_step(
            micro_batches,
            before_forward=_before_forward,
            after_forward=_after_forward,
            after_backward=_after_backward,
        )
        self._probe.finish()
        # Log the training loss to TB (aggregate + per NE task + LLW s/weights + MT
        # mt_loss); parity with RecirXLATrainPipeline. all_losses is the per-task vector.
        _series: Dict[str, Any] = _loss_series(
            all_losses, self._loss_task_names, self._cls_indices
        )
        _series.update(_llw_series(named, self._loss_task_names))
        _series.update(_hstu_series(named))
        _series.update(_lr_series(self._model))
        self._loss_metric.update(values=_series)
        return all_losses, ret

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
        light_loss_norm: bool = False,
        grad_accum_steps: int = 1,
    ) -> None:
        self._cls_task_indices = cls_task_indices
        self._reg_task_indices = reg_task_indices
        self._light_loss_norm = light_loss_norm
        self._grad_accum_steps = grad_accum_steps
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
            light_loss_norm=self._light_loss_norm,
            grad_accum_steps=self._grad_accum_steps,
        )
