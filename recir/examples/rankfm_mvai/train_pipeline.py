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
from .example_lengths import num_sequence_groups, real_example_lengths
from .metric_states import extract_states, update_ranking_metrics_from_states, W_SUM
from .rankfm_dispatchers import PreprocessorDispatcher, SEQ_LENGTHS_BASIS_KEY
from .snapshot_metrics import SnapshotMetrics
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


def _sibling_env(*metric_lists: Any) -> Tuple[int, int, int]:
    """Rank, report cadence and global batch size, read off the sibling metrics."""
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    # First sibling (across lists, in order) that carries each attribute; the
    # `is not None` check keeps interval==1 a valid hit so selection is
    # deterministic regardless of which list is scanned first.
    report_interval = 1
    global_batch_size = 0
    for lst in metric_lists:
        for m in lst or []:
            ri = getattr(m, "_report_interval", None)
            if ri is not None and report_interval == 1:
                report_interval = int(ri)
            gbs = getattr(m, "_global_batch_size", None)
            if gbs is not None and not global_batch_size:
                global_batch_size = int(gbs)
    if not global_batch_size:
        raise ValueError(
            "no sibling metric exposes _global_batch_size; the snapshot "
            "channel would report every scalar at cumulative-examples 0"
        )
    return rank, report_interval, global_batch_size


def _build_snapshot_metrics(
    all_task_names: List[str], model: Any, *metric_lists: Any
) -> SnapshotMetrics:
    """Build the MAST-equivalent per-task snapshot channel."""
    rank, report_interval, global_batch_size = _sibling_env(*metric_lists)
    return SnapshotMetrics(
        rank=rank,
        report_interval=report_interval,
        global_batch_size=global_batch_size,
        all_task_names=all_task_names,
        num_sequence_groups=num_sequence_groups(model),
    )


def _uih_lengths(batch: Any) -> Optional[torch.Tensor]:
    """Per-sample UIH length off the RAW batch, before any pad row exists."""
    kjt = getattr(batch, "ro_id_list_features", None)
    if kjt is None:
        return None
    keys = list(kjt.keys())
    if SEQ_LENGTHS_BASIS_KEY not in keys:
        return None
    stride = int(kjt.stride())
    return kjt.lengths().view(len(keys), stride)[keys.index(SEQ_LENGTHS_BASIS_KEY)]


def _assert_task_rows(prediction: torch.Tensor, all_task_names: List[str]) -> None:
    """Fail fast when the model's row count disagrees with the task partition."""
    if not all_task_names:
        return
    n_rows = prediction.size(0)
    if n_rows != len(all_task_names):
        raise ValueError(
            f"model emitted {n_rows} task rows but the partition names "
            f"{len(all_task_names)} tasks; per-task metrics would be misaligned. "
            "Check module_config.num_main_tasks and mt_tasks_configs."
        )


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
        all_task_names: Optional[List[str]] = None,
        host_sync_every: int = 1,
        jax_profiling: bool = False,
        profile_start_step: int = -1,
        profile_end_step: int = -1,
        profile_dir: str = "/tmp/tpu_profile",
        enable_memory_log: bool = False,
        pipeline_preprocessor: bool = False,
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
        self._pipeline_preprocessor: bool = pipeline_preprocessor
        self._pending_batch: Optional[Any] = None
        self._all_task_names: List[str] = list(all_task_names or [])
        self._snapshot_metrics: SnapshotMetrics = _build_snapshot_metrics(
            self._all_task_names, self._model, step_metrics, cls_metrics, reg_metrics
        )
        # Appended to the base's step-metric list rather than returned from
        # `metrics`: only step metrics are handed the unsliced per-task tensors
        # by `_update_ranking_metrics`, and that list also feeds `super().metrics`.
        self._step_metrics.extend(self._snapshot_metrics.step_metrics)
        self._step = 0
        self._host_sync_every: int = max(1, host_sync_every)
        self._in_graph_metrics: bool = cast(
            "RecirXLACompiledModule", model
        ).in_graph_metrics
        self._profiling_active = False
        self._warned_missing_loss_norm_denom = False
        # The consumer (materialize + metric update) runs one step late, on the
        # worker. These two are its input and its output: the input is the step
        # output parked as JAX handles, the output is what `progress` returns.
        self._deferred_consumer_input: Optional[
            Tuple[Any, Any, int, Optional[torch.Tensor], int]
        ] = None
        self._deferred_consumer_output: Optional[
            Tuple[torch.Tensor, Dict[str, Any]]
        ] = None
        self._dispatcher: Optional[PreprocessorDispatcher] = next(
            (
                d
                for d in getattr(model, "_dispatchers", ()) or ()
                if isinstance(d, PreprocessorDispatcher)
            ),
            None,
        )
        if pipeline_preprocessor:
            dispatcher = self._dispatcher
            if dispatcher is None:
                raise RuntimeError(
                    "pipeline_preprocessor is set but the compiled module has "
                    "no PreprocessorDispatcher; there is no graph break to run "
                    "ahead of"
                )
            dispatcher.enable_pipelining(
                cast("RecirXLACompiledModule", model).flat_step_args
            )
            logger.info("Preprocessor pipelining enabled (1-step lookahead).")
        if jax_profiling:
            logger.info(
                f"JAX profiling enabled: steps {profile_start_step}-"
                f"{profile_end_step}, dir={profile_dir}"
            )

    def _next_batch(self, dataloader_iter: Iterator[Any]) -> Tuple[Any, Optional[Any]]:
        if not self._pipeline_preprocessor:
            return next(dataloader_iter), None

        if self._pending_batch is None:
            self._pending_batch = next(dataloader_iter)
        batch = self._pending_batch
        try:
            self._pending_batch = next(dataloader_iter)
        except StopIteration:
            self._pending_batch = None
        return batch, self._pending_batch

    def progress(
        self, dataloader_iter: Iterator[Any]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        batch, next_batch = self._next_batch(dataloader_iter)
        self._step += 1
        pending = self._deferred_consumer_input
        self._deferred_consumer_input = None
        self._deferred_consumer_output = None
        dispatcher = self._dispatcher
        if dispatcher is not None:
            dispatcher.begin_step(self._step, batch, next_batch)
            if pending is not None:
                dispatcher.set_consumer(lambda: self._stash_consume(*pending))

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
        # compute on the same cadence. Only the LR buffers the metrics reporter
        # reads go back.
        if self._step % self._host_sync_every == 0:
            self._drain_deferred()
            cast("RecirXLACompiledModule", self._model).sync_lr_to_host()

        # Rebuild the ret dict RankingTrainPipeline._update_ranking_metrics
        # expects; the metrics were constructed on torch.device("cpu") to match
        # these host CPU tensors. weight=0 keeps padded rows out of NE/CTR/MSE;
        # num_real_candidates keeps variable-step x-axes and QPS padding-invariant.
        num_real_candidates = int(batch.num_candidates.sum())
        real_lengths = real_example_lengths(
            _uih_lengths(batch),
            batch.num_candidates,
            self._snapshot_metrics.num_groups,
        )

        if self._pipeline_preprocessor:
            if (
                self._step == 1
                and not cast(
                    "RecirXLACompiledModule", self._model
                ).enable_output_pipelining()
            ):
                raise RuntimeError(
                    "output pipelining could not be armed after the first step; "
                    "the model has no compiled step"
                )
            self._deferred_consumer_input = (
                losses,
                named,
                num_real_candidates,
                real_lengths,
                self._step,
            )
            if dispatcher is not None:
                dispatcher.drain_producer()
            deferred_result = self._deferred_consumer_output
            if deferred_result is None:
                if pending is not None:
                    raise RuntimeError(
                        f"step {self._step - 1}'s deferred consume never "
                        "produced a result"
                    )
                return torch.zeros(()), {}
            return deferred_result
        else:
            return self._consume_step(
                losses, named, num_real_candidates, real_lengths, self._step
            )

    def _stash_consume(
        self,
        losses: Any,
        named: Any,
        nrc: int,
        real_lengths: Optional[torch.Tensor],
        step: int,
    ) -> None:
        self._deferred_consumer_output = self._consume_step(
            losses, named, nrc, real_lengths, step
        )

    def _consume_step(
        self,
        losses: Any,
        named: Any,
        num_real_candidates: int,
        real_lengths: Optional[torch.Tensor],
        step: int,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Materialize a step's output and fold it into the metrics.

        ``losses``/``named`` are JAX handles when output deferral is armed, torch
        tensors otherwise; ``materialize_output`` is a no-op on the latter.
        """
        model: RecirXLACompiledModule = cast("RecirXLACompiledModule", self._model)
        losses = model.materialize_output(losses)
        named = model.materialize_output(named)
        if self._in_graph_metrics:
            # The graph reduced prediction/label/weight to per-task accumulators,
            # so those three were never emitted and never crossed the bus.
            # `extract_states` raises if they are missing, which would mean the
            # traced program and this flag disagree.
            return losses, self._update_metrics_from_states(
                named, losses, num_real_candidates, real_lengths, step
            )
        else:
            ret: Dict[str, Any] = {
                "prediction": named["prediction"],
                "label": named["label"],
                "weight": named["weight"],
                # Real (pre-pad) per-host candidate count -> variable-step metrics
                # x-axis stride (else it falls back to the PADDED labels.size(1)).
                # Computed from the raw batch so the model needn't emit it.
                "num_real_candidates": num_real_candidates,
                "step": step,
            }
            _assert_task_rows(ret["prediction"], self._all_task_names)
            self._update_ranking_metrics(ret)
            self._snapshot_metrics.update(losses, named, self._model, real_lengths)
            return losses, ret

    def _update_metrics_from_states(
        self,
        named: Dict[str, Any],
        losses: torch.Tensor,
        num_real_candidates: int,
        real_lengths: Optional[torch.Tensor],
        step: int,
    ) -> Dict[str, Any]:
        states = extract_states(named)
        _assert_task_rows(states[W_SUM], self._all_task_names)
        real_rows = named.get("loss_norm_denom")
        if real_rows is None:
            # The producer emits None when its forward computed no denominator.
            # Falling back to the padded candidate count reinstates the very
            # accounting this path exists to avoid, so say so rather than drift.
            if not self._warned_missing_loss_norm_denom:
                self._warned_missing_loss_norm_denom = True
                logger.warning(
                    "loss_norm_denom absent; crediting the metric window the "
                    "PADDED candidate count, so window length varies with the "
                    "pad fraction"
                )
            num_samples = num_real_candidates
        else:
            # An all-padded / all-zero-weight microbatch has no real examples;
            # a 0 would park a never-evicted entry in the row-sized window.
            num_samples = max(1, int(real_rows.item()))
        unfed = update_ranking_metrics_from_states(
            states,
            num_samples=num_samples,
            num_real_candidates=num_real_candidates,
            cls_metrics=self._cls_metrics,
            reg_metrics=self._reg_metrics,
            step_metrics=self._step_metrics,
            cls_indices=self._cls_indices,
            reg_indices=self._reg_indices,
        )
        if unfed:
            # Loud rather than a frozen curve: a metric with no state mapper
            # would otherwise report its seed value forever.
            raise RuntimeError(
                "in-graph metrics cannot feed "
                f"{[type(getattr(m, 'rec_metric', m)).__name__ for m in unfed]}; "
                "add a mapper in metric_states._mapper_for or keep this metric "
                "off the in-graph path"
            )
        self._snapshot_metrics.update(losses, named, self._model, real_lengths)
        return {
            "num_real_candidates": num_real_candidates,
            "step": step,
            **states,
        }

    def _drain_deferred(self) -> None:
        """Consume the outstanding deferred step, if any.

        Must run before anything READS metric state -- the host-state flush, the
        metrics property, train end -- or the last step's contribution is
        missing from the report.
        """
        prev = self._deferred_consumer_input
        if prev is not None:
            self._deferred_consumer_input = None
            self._consume_step(*prev)
        dispatcher = self._dispatcher
        if dispatcher is not None:
            dispatcher.drain_producer()

    @property
    def metrics(self) -> List[Any]:
        self._drain_deferred()
        return list(super().metrics) + self._snapshot_metrics.metrics

    def on_train_end(self) -> None:
        # Stop any still-active trace (e.g. training ended before profile_end_step).
        if self._profiling_active:
            jax.profiler.stop_trace()
            self._profiling_active = False
            logger.info("[PROFILE] Stopped trace (training ended).")
        # Final host-state flush so params/buffers are fresh for any end-of-run
        # inspection/checkpoint even if the last step was not a flush step.
        self._drain_deferred()
        cast("RecirXLACompiledModule", self._model).flush_host_state()
        dispatcher = self._dispatcher
        if dispatcher is not None:
            dispatcher.close_pipelining()
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
        all_task_names: Optional[List[str]] = None,
        is_variable_step: bool = True,
        report_interval: int = 1,
        jax_profiling: bool = False,
        profile_start_step: int = -1,
        profile_end_step: int = -1,
        profile_dir: str = "/tmp/tpu_profile",
        enable_memory_log: bool = False,
        pipeline_preprocessor: bool = False,
        window_size: int = 1_000_000,
    ) -> None:
        self._cls_task_indices = cls_task_indices
        self._reg_task_indices = reg_task_indices
        self._all_task_names = all_task_names
        self._host_sync_every = report_interval
        self._jax_profiling = jax_profiling
        self._profile_start_step = profile_start_step
        self._profile_end_step = profile_end_step
        self._profile_dir = profile_dir
        self._enable_memory_log = enable_memory_log
        self._pipeline_preprocessor = pipeline_preprocessor
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
            window_size=window_size,
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
            all_task_names=self._all_task_names,
            host_sync_every=self._host_sync_every,
            jax_profiling=self._jax_profiling,
            profile_start_step=self._profile_start_step,
            profile_end_step=self._profile_end_step,
            profile_dir=self._profile_dir,
            enable_memory_log=self._enable_memory_log,
            pipeline_preprocessor=self._pipeline_preprocessor,
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
        all_task_names: Optional[List[str]] = None,
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
        self._all_task_names: List[str] = list(all_task_names or [])
        self._snapshot_metrics: SnapshotMetrics = _build_snapshot_metrics(
            self._all_task_names,
            self._raw_model,
            step_metrics,
            cls_metrics,
            reg_metrics,
        )
        self._step_metrics.extend(self._snapshot_metrics.step_metrics)
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
            _assert_task_rows(ret["prediction"], self._all_task_names)
            self._update_ranking_metrics(ret)

        all_losses, named = self._step.run_step(
            micro_batches,
            before_forward=_before_forward,
            after_forward=_after_forward,
            after_backward=_after_backward,
        )
        self._probe.finish()
        # `named` is the last micro-batch's forward, so measure that batch.
        last = micro_batches[-1]
        self._snapshot_metrics.update(
            all_losses,
            named,
            self._model,
            real_example_lengths(
                _uih_lengths(last),
                last.num_candidates,
                self._snapshot_metrics.num_groups,
            ),
        )
        return all_losses, ret

    @property
    def metrics(self) -> List[Any]:
        return list(super().metrics) + self._snapshot_metrics.metrics

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
        all_task_names: Optional[List[str]] = None,
        is_variable_step: bool = True,
        report_interval: int = 1,
        window_size: int = 1_000_000,
        light_loss_norm: bool = False,
        grad_accum_steps: int = 1,
    ) -> None:
        self._cls_task_indices = cls_task_indices
        self._reg_task_indices = reg_task_indices
        self._all_task_names = all_task_names
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
            window_size=window_size,
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
            all_task_names=self._all_task_names,
            light_loss_norm=self._light_loss_norm,
            grad_accum_steps=self._grad_accum_steps,
        )
