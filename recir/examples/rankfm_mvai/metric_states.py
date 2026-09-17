#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""The seven per-task reductions that NE, CTR, MSE and calibration are built from.

The formulas mirror, and must stay in sync with:

- ``torchrec/metrics/ne.py:22`` ``compute_cross_entropy`` / ``:79`` ``get_ne_states``
- ``torchrec/metrics/ctr.py:30`` ``get_ctr_states``
- ``torchrec/metrics/mse.py:57`` ``compute_error_sum`` / ``:70`` ``get_mse_states``
- ``torchrec/metrics/calibration.py:34`` ``get_calibration_states``
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch

# torchrec's NE eta (torchrec/metrics/ne.py:147). Used for the prediction clamp
# that keeps log2 finite; the same constant has to be applied here because the
# clamp happens before the sum and therefore cannot be reapplied on the host.
NE_ETA: float = 1e-12

W_SUM: str = "w_sum"
WL_SUM: str = "wl_sum"
W1ML_SUM: str = "w1ml_sum"
WL2_SUM: str = "wl2_sum"
WP_SUM: str = "wp_sum"
WERR_SUM: str = "werr_sum"
WCE_SUM: str = "wce_sum"
NNZ_SUM: str = "nnz_sum"
PADDED_ROWS: str = "padded_rows"

STATE_KEYS: tuple[str, ...] = (
    W_SUM,
    WL_SUM,
    W1ML_SUM,
    WL2_SUM,
    WP_SUM,
    WERR_SUM,
    WCE_SUM,
    NNZ_SUM,
    PADDED_ROWS,
)

# Prefix under which the accumulators travel in ``named_tensors``. The producer
# (inside the traced graph) and the consumer (the train pipeline) must agree, so
# both go through ``emit_keys`` / ``extract_states`` rather than spelling the
# strings out twice.
STATE_PREFIX: str = "metric_state/"


def emit_keys(states: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Namespace the accumulators for insertion into ``named_tensors``."""
    return {STATE_PREFIX + k: v for k, v in states.items()}


def extract_states(named: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Inverse of ``emit_keys``. Raises if the graph emitted a partial set."""
    missing = [k for k in STATE_KEYS if STATE_PREFIX + k not in named]
    if missing:
        raise KeyError(
            f"named_tensors is missing metric states {missing}; the graph and "
            "metric_states.STATE_KEYS have drifted apart"
        )
    return {k: named[STATE_PREFIX + k] for k in STATE_KEYS}


def compute_metric_states(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    eta: float = NE_ETA,
) -> Dict[str, torch.Tensor]:
    """Reduce ``[num_tasks, n]`` predictions/labels/weights to eight ``[num_tasks]``."""
    wl = weights * labels
    w1ml = weights * (1.0 - labels)
    # Out-of-place, unlike torchrec's `predictions.clamp_`, which mutates its
    # own fp64 copy. Here the input is the live graph tensor.
    p_hi = predictions.clamp(min=eta, max=1.0 - eta)
    p_lo = (1.0 - predictions).clamp(min=eta, max=1.0 - eta)
    cross_entropy = -wl * torch.log2(p_hi) - w1ml * torch.log2(p_lo)
    return {
        W_SUM: torch.sum(weights, dim=-1),
        WL_SUM: torch.sum(wl, dim=-1),
        W1ML_SUM: torch.sum(w1ml, dim=-1),
        WL2_SUM: torch.sum(weights * torch.square(labels), dim=-1),
        WP_SUM: torch.sum(predictions * weights, dim=-1),
        WERR_SUM: torch.sum(weights * torch.square(labels - predictions), dim=-1),
        WCE_SUM: torch.sum(cross_entropy, dim=-1),
        NNZ_SUM: torch.sum((weights > 0).to(weights.dtype), dim=-1),
        PADDED_ROWS: torch.full(
            (), float(weights.shape[-1]), dtype=weights.dtype, device=weights.device
        ),
    }


def ne_states(states: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Same keys and order as ``torchrec.metrics.ne.get_ne_states``."""
    return {
        "cross_entropy_sum": states[WCE_SUM],
        "weighted_num_samples": states[W_SUM],
        "pos_labels": states[WL_SUM],
        "neg_labels": states[W1ML_SUM],
    }


def ctr_states(states: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Same keys as ``torchrec.metrics.ctr.get_ctr_states``.

    Imported lazily so this module stays importable inside the traced graph,
    where torchrec's metrics package is not a dependency.
    """
    from torchrec.metrics.ctr import CTR_DENOM, CTR_NUM

    return {CTR_NUM: states[WL_SUM], CTR_DENOM: states[W_SUM]}


def mse_states(states: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Same keys as ``torchrec.metrics.mse.get_mse_states``."""
    return {
        "error_sum": states[WERR_SUM],
        "weighted_num_samples": states[W_SUM],
        "label_sum": states[WL_SUM],
        "label_squared_sum": states[WL2_SUM],
    }


def calibration_states(states: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Same keys as ``torchrec.metrics.calibration.get_calibration_states``."""
    from torchrec.metrics.calibration import CALIBRATION_DENOM, CALIBRATION_NUM

    return {CALIBRATION_NUM: states[WP_SUM], CALIBRATION_DENOM: states[WL_SUM]}


def _mapper_for(
    computation: object,
) -> Optional[Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]]:
    """Pick the state mapper for a ``RecMetricComputation``, or None."""
    from torchrec.metrics.calibration import CalibrationMetricComputation
    from torchrec.metrics.ctr import CTRMetricComputation
    from torchrec.metrics.mse import MSEMetricComputation
    from torchrec.metrics.ne import NEMetricComputation

    mappers: Dict[
        type, Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]
    ] = {
        NEMetricComputation: ne_states,
        CTRMetricComputation: ctr_states,
        MSEMetricComputation: mse_states,
        CalibrationMetricComputation: calibration_states,
    }
    try:
        # mvai_cloud's NE. Registered explicitly after checking it adds no
        # `_add_state` -- it overrides only `_ne_report` and `_compute`, the
        # reporting side -- so its state set is the base class's.
        #
        # Resolved dynamically rather than imported: two modules answer to this
        # name. The buck-visible one (//minimal_viable_ai/core/ranking_common/
        # metrics:ne) does not define this class; the mvai_cloud copy that does
        # is staged by the Docker image, not by buck. A static import is
        # therefore unresolvable at type-check time even though it succeeds in
        # the image.
        ne_mod = importlib.import_module(
            "minimal_viable_ai.core.ranking_common.metrics.ne"
        )
        mappers[ne_mod._PerTaskNEMetricComputation] = ne_states
    except (ImportError, RuntimeError, AttributeError):
        pass
    return mappers.get(type(computation))


def _accumulate(
    computation: Any, mapped: Dict[str, torch.Tensor], num_samples: int
) -> None:
    """Add one step's states into a computation's buffers."""
    for name, value in mapped.items():
        state = getattr(computation, name)
        # Buffers are float64 (`_add_state(..., dtype=torch.double)`) while the
        # values arrive float32 from the device. Cast up front so the window
        # buffer stores exactly what the lifetime state stored.
        value = value.to(state.dtype)
        state += value
        computation._aggregate_window_state(name, value, num_samples)
    computation._computed = None
    # hasattr, not try/except: these live on an nn.Module, whose __getattr__
    # raises AttributeError for a missing name rather than returning None.
    if hasattr(computation, "_update_count"):
        computation._update_count += 1
    elif hasattr(computation, "_update_called"):
        computation._update_called = True


def _feed_effective_sample_cnt(metric: Any, nnz: torch.Tensor) -> bool:
    """Feed ``FixedStepEffectiveSampleCnt`` from the row count. False if not one."""
    lifetime = getattr(metric, "_lifetime", None)
    window = getattr(metric, "_window", None)
    if not (isinstance(lifetime, torch.Tensor) and isinstance(window, torch.Tensor)):
        return False
    metric._itrn += 1
    if not metric.should_update():
        return True
    counts = nnz.to(lifetime.dtype)
    n = min(counts.numel(), lifetime.numel())
    lifetime[:n] += counts[:n]
    window[:n] += counts[:n]
    metric._step_stride += metric._global_batch_size
    return True


def _feed_row_count_metric(metric: Any, num_samples: int) -> bool:
    """Feed ``VariableStepQPS``, which needs a row count, not tensor values."""
    if not hasattr(metric, "_local_total_num_examples"):
        return False
    placeholder = torch.zeros(1, 1).expand(1, num_samples)
    metric.update(labels=placeholder, num_real_candidates=num_samples)
    return True


def _rec_metric_children(metric: Any) -> List[Any]:
    """Sub-metrics of a fan-out container, or empty if this is not one."""
    return [
        v for v in vars(metric).values() if getattr(v, "rec_metric", None) is not None
    ]


def update_ranking_metrics_from_states(
    states: Dict[str, torch.Tensor],
    *,
    num_samples: int,
    num_real_candidates: int,
    cls_metrics: Sequence[Any] = (),
    reg_metrics: Sequence[Any] = (),
    step_metrics: Sequence[Any] = (),
    cls_indices: Optional[torch.Tensor] = None,
    reg_indices: Optional[torch.Tensor] = None,
) -> List[Any]:
    """The compiled-path counterpart to ``_update_ranking_metrics``."""
    unfed: List[Any] = []

    def feed(metrics: Sequence[Any], indices: Optional[torch.Tensor]) -> None:
        for metric in metrics:
            computation = getattr(metric, "rec_metric", None)
            mapper = _mapper_for(computation) if computation is not None else None
            if mapper is None:
                # The step-metric list also carries two non-RecMetric entries,
                # which take a row count rather than per-task states.
                children = _rec_metric_children(metric)
                if children:
                    # Fan-out container. Mirrors its own update bookkeeping
                    # (variable_step_metrics_collection.py:87-96) before
                    # delegating, because report_metrics gates on `_updated`
                    # and reports against `_local_report_interval_num_examples`.
                    metric._itrn += 1
                    metric._updated = True
                    if not metric.should_update():
                        continue
                    metric._local_report_interval_num_examples += num_real_candidates
                    feed(children, indices)
                    continue
                if indices is None and (
                    _feed_effective_sample_cnt(metric, states[NNZ_SUM])
                    or _feed_row_count_metric(metric, num_real_candidates)
                ):
                    continue
                unfed.append(metric)
                continue
            # Mirrors FixedStepRecMetric.update's bookkeeping
            # (mvai_cloud/.../metrics/torchrec_mtml_metrics.py:92).
            metric._itrn += 1
            if not metric.should_update():
                continue
            mapped = mapper(states)
            if indices is not None:
                mapped = {
                    k: torch.index_select(v, 0, indices) for k, v in mapped.items()
                }
            _accumulate(computation, mapped, num_samples)
            metric._step_stride += metric._global_batch_size

    feed(step_metrics, None)
    feed(cls_metrics, cls_indices)
    feed(reg_metrics, reg_indices)
    return unfed
