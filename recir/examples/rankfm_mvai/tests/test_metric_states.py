#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from __future__ import annotations

import unittest
from typing import Any, cast

import torch
from recir.examples.rankfm_mvai.metric_states import (
    _accumulate,
    calibration_states,
    compute_metric_states,
    ctr_states,
    mse_states,
    NE_ETA,
    ne_states,
    NNZ_SUM,
    PADDED_ROWS,
    update_ranking_metrics_from_states,
    W1ML_SUM,
    W_SUM,
    WCE_SUM,
    WL_SUM,
)
from torch.fx.experimental.proxy_tensor import make_fx
from torchrec.metrics.calibration import get_calibration_states
from torchrec.metrics.ctr import get_ctr_states
from torchrec.metrics.mse import get_mse_states
from torchrec.metrics.ne import get_ne_states

_TASKS = 9
_N = 4096


def _inputs(
    dtype: torch.dtype, n: int = _N, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    predictions = torch.rand((_TASKS, n), generator=g, dtype=dtype)
    labels = torch.randint(0, 2, (_TASKS, n), generator=g).to(dtype)
    weights = torch.rand((_TASKS, n), generator=g, dtype=dtype)
    return predictions, labels, weights


def _saturating_inputs(
    dtype: torch.dtype, n: int = _N, seed: int = 1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Like ``_inputs`` but with 5% of predictions pinned to exactly 0.0 and 1.0.

    ``torch.rand`` never lands on the eta clamp, so a test built on it cannot
    see the one place fp32 and fp64 diverge. A trained sigmoid saturates all the
    time, so this is the realistic case, not a corner case.
    """
    g = torch.Generator().manual_seed(seed)
    predictions = torch.rand((_TASKS, n), generator=g, dtype=dtype)
    k = max(1, n // 40)
    predictions[:, :k] = 1.0
    predictions[:, k : 2 * k] = 0.0
    labels = torch.randint(0, 2, (_TASKS, n), generator=g).to(dtype)
    weights = torch.rand((_TASKS, n), generator=g, dtype=dtype)
    return predictions, labels, weights


# Only `cross_entropy_sum` takes a log, and only it depends on the eta clamp;
# everything else is a plain weighted sum.
_CE_STATE = "cross_entropy_sum"


def _torchrec_reference(
    predictions: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor
) -> dict[str, dict[str, torch.Tensor]]:
    """Call each torchrec state fn on its own copy of ``predictions``.

    ``compute_cross_entropy`` does ``predictions = predictions.double()`` then
    ``predictions.clamp_(...)``. When the input is already fp64 the ``.double()``
    is a no-op returning the *same* tensor, so the in-place clamp escapes into
    the caller's tensor and the next state fn would see clamped predictions.
    Harmless in production (predictions arrive fp32, so ``.double()`` copies),
    but it would silently corrupt this comparison.
    """
    return {
        "ne": get_ne_states(labels, predictions.clone(), weights, NE_ETA),
        "ctr": get_ctr_states(labels, predictions.clone(), weights),
        "mse": get_mse_states(labels, predictions.clone(), weights),
        "calibration": get_calibration_states(labels, predictions.clone(), weights),
    }


class MetricStatesTest(unittest.TestCase):
    def test_matches_torchrec_bit_exactly_in_fp64(self) -> None:
        predictions, labels, weights = _inputs(torch.float64)
        expected = _torchrec_reference(predictions, labels, weights)
        states = compute_metric_states(predictions, labels, weights)
        got = {
            "ne": ne_states(states),
            "ctr": ctr_states(states),
            "mse": mse_states(states),
            "calibration": calibration_states(states),
        }
        for metric, ref in expected.items():
            self.assertEqual(set(ref), set(got[metric]), f"{metric} key set")
            for name, ref_value in ref.items():
                self.assertTrue(
                    torch.equal(got[metric][name], ref_value),
                    f"{metric}/{name} not bit-exact: "
                    f"max|d|={(got[metric][name] - ref_value).abs().max()}",
                )

    def test_fp32_matches_torchrec_with_saturated_predictions(self) -> None:
        """fp32 vs the fp64 host path, on inputs that actually hit the clamp.

        torchrec upcasts to fp64 before doing anything, so bit-exactness is out
        of reach on TPU regardless of how the clamp is written -- even ``1 - p``
        rounds differently. The split below records where that lands: the plain
        sums stay near-exact, and only the cross entropy carries fp32 epsilon.
        """
        p32, l32, w32 = _saturating_inputs(torch.float32)
        expected = _torchrec_reference(p32.double(), l32.double(), w32.double())
        states = compute_metric_states(p32, l32, w32)
        got = {
            "ne": ne_states(states),
            "ctr": ctr_states(states),
            "mse": mse_states(states),
            "calibration": calibration_states(states),
        }
        worst_sum, worst_ce = 0.0, 0.0
        for metric, ref in expected.items():
            for name, ref_value in ref.items():
                value = got[metric][name]
                self.assertFalse(
                    bool(torch.isnan(value).any() or torch.isinf(value).any()),
                    f"{metric}/{name} is not finite: {value}",
                )
                rel = float(
                    (
                        (value.double() - ref_value).abs()
                        / ref_value.abs().clamp(min=1e-30)
                    ).max()
                )
                if name == _CE_STATE:
                    worst_ce = max(worst_ce, rel)
                    self.assertLess(rel, 1e-6, f"{metric}/{name} rel {rel}")
                else:
                    worst_sum = max(worst_sum, rel)
                    self.assertLess(rel, 1e-6, f"{metric}/{name} rel {rel}")
        # fp32 eps is ~6e-8, so the cross entropy is already at the floor; a
        # different clamp formulation cannot beat it. Anything materially above
        # this means the reduction itself regressed.
        self.assertLess(worst_ce, 1e-6)
        self.assertLess(worst_sum, 1e-6)

    def test_zero_weight_rows_do_not_contribute(self) -> None:
        predictions, labels, weights = _inputs(torch.float64)
        base = compute_metric_states(predictions, labels, weights)

        pad = 512
        g = torch.Generator().manual_seed(7)
        padded = compute_metric_states(
            torch.cat(
                [predictions, torch.rand((_TASKS, pad), generator=g).double()], dim=-1
            ),
            torch.cat(
                [labels, torch.rand((_TASKS, pad), generator=g).double()], dim=-1
            ),
            torch.cat([weights, torch.zeros((_TASKS, pad), dtype=torch.float64)], -1),
        )
        for name, value in base.items():
            if name == PADDED_ROWS:
                # The one state that SHOULD move: it is the padded width, which
                # is exactly the quantity the unpad fix stops crediting to the
                # window. Everything else must be blind to the padding.
                self.assertEqual(
                    int(padded[name].item()), int(value.item()) + pad, PADDED_ROWS
                )
                continue
            self.assertTrue(
                torch.equal(padded[name], value),
                f"{name} moved when zero-weight rows were appended",
            )

    def test_does_not_mutate_its_inputs(self) -> None:
        predictions, labels, weights = _inputs(torch.float64)
        before = [t.clone() for t in (predictions, labels, weights)]
        compute_metric_states(predictions, labels, weights)
        for name, now, then in zip(
            ("predictions", "labels", "weights"),
            (predictions, labels, weights),
            before,
        ):
            self.assertTrue(torch.equal(now, then), f"{name} was mutated")

    def test_neg_labels_is_emitted_not_derived(self) -> None:
        """``w_sum - wl_sum`` cancels when labels are mostly 1, which is the
        normal regime for a high-CTR task. Emitting the sum directly avoids it."""
        n = 8192
        g = torch.Generator().manual_seed(3)
        weights = torch.rand((1, n), generator=g, dtype=torch.float32)
        labels = torch.ones((1, n), dtype=torch.float32)
        # A single genuine negative in a sea of positives.
        labels[0, 0] = 0.0
        predictions = torch.rand((1, n), generator=g, dtype=torch.float32)

        states = compute_metric_states(predictions, labels, weights)
        exact = weights[0, 0].double()
        emitted = states[W1ML_SUM].double()
        derived = (states[W_SUM] - states[WL_SUM]).double()
        self.assertLess(
            float((emitted - exact).abs()),
            float((derived - exact).abs()),
            "emitting w1ml_sum should beat deriving it",
        )

    def test_is_traceable(self) -> None:
        """It has to survive make_fx to become XLA ops."""
        predictions, labels, weights = _inputs(torch.float32, n=64)
        # eta is passed explicitly: make_fx matches the full signature, so
        # leaving it defaulted raises "expected 4 arguments but got 3".
        gm = make_fx(compute_metric_states)(predictions, labels, weights, NE_ETA)
        out = gm(predictions, labels, weights, NE_ETA)
        for name, value in compute_metric_states(predictions, labels, weights).items():
            self.assertTrue(torch.equal(out[name], value), name)


class _FakeFixedStepRecMetric:
    """The slice of ``FixedStepRecMetric`` the helper touches."""

    def __init__(self, computation: Any, global_batch_size: int = 1024) -> None:
        self.rec_metric = computation
        self._itrn = 0
        self._step_stride = 0
        self._global_batch_size = global_batch_size

    def should_update(self) -> bool:
        return True


class UpdateFromStatesTest(unittest.TestCase):
    """Does feeding pre-reduced states leave the same buffers as feeding tensors"""

    def _computations(self, n_tasks: int = _TASKS) -> dict[str, Any]:
        from torchrec.metrics.calibration import CalibrationMetricComputation
        from torchrec.metrics.ctr import CTRMetricComputation
        from torchrec.metrics.mse import MSEMetricComputation
        from torchrec.metrics.ne import NEMetricComputation

        # window_size large enough that one update never evicts, so both paths
        # leave identical window buffers.
        kwargs = {
            "my_rank": 0,
            "batch_size": _N,
            "n_tasks": n_tasks,
            "window_size": 1_000_000,
        }
        return {
            "ne": NEMetricComputation(**kwargs),
            "ctr": CTRMetricComputation(**kwargs),
            "mse": MSEMetricComputation(**kwargs),
            "calibration": CalibrationMetricComputation(**kwargs),
        }

    def _state_names(self, computation: Any) -> list[str]:
        return [
            n
            for n in getattr(computation, "_defaults", {})
            if not n.startswith("window_")
        ]

    def test_accumulates_the_same_buffers_as_torchrec_update(self) -> None:
        predictions, labels, weights = _inputs(torch.float64)
        reference = self._computations()
        under_test = self._computations()

        for computation in reference.values():
            # clone(): torchrec's in-place clamp escapes on fp64 input, see
            # _torchrec_reference above.
            computation.update(
                predictions=predictions.clone(), labels=labels, weights=weights
            )

        states = compute_metric_states(predictions, labels, weights)
        mappers = {
            "ne": ne_states,
            "ctr": ctr_states,
            "mse": mse_states,
            "calibration": calibration_states,
        }
        for name, computation in under_test.items():
            _accumulate(computation, mappers[name](states), num_samples=_N)

        for name, computation in reference.items():
            checked = 0
            for state_name in self._state_names(computation):
                expected = getattr(computation, state_name)
                got = getattr(under_test[name], state_name)
                self.assertTrue(
                    torch.equal(got, expected),
                    f"{name}/{state_name} differs: max|d|={(got - expected).abs().max()}",
                )
                window_name = f"window_{state_name}"
                self.assertTrue(
                    torch.equal(
                        getattr(under_test[name], window_name),
                        getattr(computation, window_name),
                    ),
                    f"{name}/{window_name} differs",
                )
                checked += 1
            self.assertGreater(checked, 0, f"{name} exposed no states to compare")

    def test_fans_out_and_slices_like_update_ranking_metrics(self) -> None:
        """cls/reg metrics see an index_select-ed subset, step metrics do not."""
        cls_indices = torch.tensor([0, 2, 4])
        computations = self._computations()
        cls_metric = _FakeFixedStepRecMetric(
            self._computations(n_tasks=len(cls_indices))["ne"]
        )
        step_metric = _FakeFixedStepRecMetric(computations["ctr"])

        predictions, labels, weights = _inputs(torch.float64)
        states = compute_metric_states(predictions, labels, weights)
        unfed = update_ranking_metrics_from_states(
            states,
            num_samples=_N,
            num_real_candidates=_N,
            cls_metrics=[cls_metric],
            step_metrics=[step_metric],
            cls_indices=cls_indices,
        )

        self.assertEqual(unfed, [])
        self.assertEqual(cls_metric._itrn, 1)
        self.assertEqual(cls_metric._step_stride, 1024)
        self.assertTrue(
            torch.equal(
                cls_metric.rec_metric.cross_entropy_sum,
                torch.index_select(states[WCE_SUM], 0, cls_indices),
            )
        )
        # Step metrics get every task.
        self.assertTrue(torch.equal(step_metric.rec_metric.ctr_denom, states[W_SUM]))

    def test_unsupported_metric_is_reported_not_skipped(self) -> None:
        """A metric type with no mapper must surface, not silently freeze."""
        from torchrec.metrics.ne import NEMetricComputation

        class _Subclass(NEMetricComputation):
            pass

        opaque = _FakeFixedStepRecMetric(object())
        subclass = _FakeFixedStepRecMetric(
            _Subclass(my_rank=0, batch_size=_N, n_tasks=_TASKS, window_size=1_000_000)
        )
        predictions, labels, weights = _inputs(torch.float64)
        unfed = update_ranking_metrics_from_states(
            compute_metric_states(predictions, labels, weights),
            num_samples=_N,
            num_real_candidates=_N,
            cls_metrics=[opaque, subclass],
        )
        # Exact-type dispatch: the subclass is refused too, because torchrec's
        # real NE subclasses carry different state sets.
        self.assertEqual(unfed, [opaque, subclass])


class ComputeCacheTest(unittest.TestCase):
    """A second step has to change what ``compute()`` returns.

    ``RecMetricComputation`` is a torchmetrics ``Metric``, so ``compute`` is
    wrapped by ``Metric._wrap_compute`` (``torchmetrics/metric.py:688``), which
    short-circuits to the cached ``self._computed`` whenever it is not None. The
    only thing that clears that cache is ``Metric._wrap_update``
    (``metric.py:542``) -- the wrapper around ``update()``, which ``_accumulate``
    deliberately bypasses so it never has to materialise ``[tasks, n]`` tensors.

    Writing the buffers without invalidating the cache freezes every report at
    the first computed value. On the 4-chip A/B that showed up as NE, CTR,
    calibration and MSE bit-identical across all 60 steps while the buffers
    underneath were correct, which read as "19 tasks report NE = eta forever"
    because those 19 were the tasks that happened to be degenerate at step 1.

    One update is not enough to see this: with a single step the cache holds the
    right answer. Both tests below take two steps and assert against a torchrec
    reference driven the same way.
    """

    def _pair(self, n_tasks: int = _TASKS) -> tuple[Any, Any]:
        from torchrec.metrics.ctr import CTRMetricComputation

        kwargs = {
            "my_rank": 0,
            "batch_size": _N,
            "n_tasks": n_tasks,
            "window_size": 1_000_000,
        }
        return CTRMetricComputation(**kwargs), CTRMetricComputation(**kwargs)

    @staticmethod
    def _lifetime_ctr(computation: Any) -> torch.Tensor:
        # compute() -> [lifetime report, window report]; take the lifetime value.
        return computation.compute()[0].value.clone()

    def test_second_step_is_not_served_from_the_first_steps_cache(self) -> None:
        reference, under_test = self._pair()
        # Different seeds so step 2's CTR genuinely differs from step 1's; a
        # repeat of the same batch would leave the ratio unchanged and the
        # stale cache would pass.
        step1 = _inputs(torch.float64, seed=0)
        step2 = _inputs(torch.float64, seed=7)

        first: torch.Tensor = torch.zeros(0)
        for step, (predictions, labels, weights) in enumerate((step1, step2)):
            reference.update(
                predictions=predictions.clone(), labels=labels, weights=weights
            )
            _accumulate(
                under_test,
                ctr_states(compute_metric_states(predictions, labels, weights)),
                num_samples=_N,
            )
            expected = self._lifetime_ctr(reference)
            if step == 0:
                first = expected
            else:
                # Guard: if the reference itself did not move, this test cannot
                # detect a stale cache and the seeds need changing.
                self.assertFalse(
                    torch.allclose(expected, first),
                    "reference CTR did not change between steps; the inputs "
                    "cannot expose a stale compute() cache",
                )
            got = self._lifetime_ctr(under_test)
            self.assertTrue(
                torch.allclose(got, expected),
                f"step {step + 1}: compute() returned {got.tolist()} but the "
                f"buffers say {expected.tolist()}",
            )

    def test_buffers_were_right_all_along(self) -> None:
        """The states are correct even when ``compute()`` is stale.

        Pins the shape of the bug: a failure of the test above with this one
        passing means the reduction is fine and only the report is frozen.
        """
        reference, under_test = self._pair()
        for predictions, labels, weights in (
            _inputs(torch.float64, seed=0),
            _inputs(torch.float64, seed=7),
        ):
            reference.update(
                predictions=predictions.clone(), labels=labels, weights=weights
            )
            _accumulate(
                under_test,
                ctr_states(compute_metric_states(predictions, labels, weights)),
                num_samples=_N,
            )
        for state_name in ("ctr_num", "ctr_denom"):
            self.assertTrue(
                torch.equal(
                    getattr(under_test, state_name), getattr(reference, state_name)
                ),
                f"{state_name} differs",
            )


class _FakeEffectiveSampleCnt:
    """The buffers ``_feed_effective_sample_cnt`` duck-types on."""

    def __init__(self, n_tasks: int, global_batch_size: int = 1024) -> None:
        self._lifetime = torch.zeros(n_tasks, dtype=torch.float64)
        self._window = torch.zeros(n_tasks, dtype=torch.float64)
        self._itrn = 0
        self._step_stride = 0
        self._global_batch_size = global_batch_size

    def should_update(self) -> bool:
        return True

    def update(self, *_: Any, weights: torch.Tensor, **__: Any) -> None:
        counts = (weights.detach() > 0).sum(dim=1).to(torch.float64)
        n = min(counts.numel(), self._lifetime.numel())
        self._lifetime[:n] += counts[:n]
        self._window[:n] += counts[:n]
        self._step_stride += self._global_batch_size


class _FakeVariableStepQPS:
    def __init__(self) -> None:
        self._local_total_num_examples = 0
        self.seen: list[int] = []

    def update(self, *, labels: torch.Tensor, **kwargs: Any) -> None:
        n = kwargs.get("num_real_candidates")
        rows = int(n) if n is not None else labels.size(dim=1)
        self._local_total_num_examples += rows
        self.seen.append(rows)


class StepMetricsTest(unittest.TestCase):
    """The two `_step_metrics` entries that are not RecMetric-backed."""

    def test_effective_sample_cnt_matches_the_weight_tensor_path(self) -> None:
        predictions, labels, weights = _inputs(torch.float64)
        # Zero out a chunk so nnz is genuinely below the row count.
        weights[:, :1000] = 0.0

        reference = _FakeEffectiveSampleCnt(_TASKS)
        reference.update(weights=weights)

        under_test = _FakeEffectiveSampleCnt(_TASKS)
        unfed = update_ranking_metrics_from_states(
            compute_metric_states(predictions, labels, weights),
            num_samples=_N,
            num_real_candidates=_N,
            step_metrics=[under_test],
        )
        self.assertEqual(unfed, [])
        self.assertTrue(torch.equal(under_test._lifetime, reference._lifetime))
        self.assertTrue(torch.equal(under_test._window, reference._window))
        self.assertEqual(under_test._step_stride, reference._step_stride)

    def test_nnz_is_a_count_not_a_weight_sum(self) -> None:
        """`sum(w > 0)` and `sum(w)` are different quantities; the plan
        originally assumed w_sum would serve, and it does not."""
        weights = torch.tensor([[0.0, 0.25, 0.25, 0.25]])
        states = compute_metric_states(
            torch.full((1, 4), 0.5), torch.ones(1, 4), weights
        )
        self.assertEqual(states[NNZ_SUM].tolist(), [3.0])
        self.assertEqual(states[W_SUM].tolist(), [0.75])

    def test_qps_gets_the_real_count_not_the_window_count(self) -> None:
        qps = _FakeVariableStepQPS()
        predictions, labels, weights = _inputs(torch.float64)
        unfed = update_ranking_metrics_from_states(
            compute_metric_states(predictions, labels, weights),
            num_samples=4096,
            num_real_candidates=1234,
            step_metrics=[qps],
        )
        self.assertEqual(unfed, [])
        self.assertEqual(qps.seen, [1234])

    def test_padded_rows_reproduces_the_host_paths_window_size(self) -> None:
        predictions, labels, weights = _inputs(torch.float64, n=777)
        states = compute_metric_states(predictions, labels, weights)
        self.assertEqual(int(states[PADDED_ROWS].item()), predictions.shape[-1])
        self.assertEqual(states[PADDED_ROWS].ndim, 0)


class _FakeCollection:
    """A fan-out container shaped like VariableStepMetricsCollection.

    The real one holds `_ne`/`_ctr`/`_calibration` as plain attributes and has
    no `rec_metric` of its own; `report_metrics` gates on `_updated` and reports
    against `_local_report_interval_num_examples`.
    """

    def __init__(self, children: dict[str, Any]) -> None:
        for name, child in children.items():
            setattr(self, name, child)
        self._itrn = 0
        self._updated = False
        self._local_report_interval_num_examples = 0

    def should_update(self) -> bool:
        return True


class ContainerFanOutTest(unittest.TestCase):
    """The metric list holds containers, not RecMetrics.

    A 4-chip run failed on step 1 with
    `cannot feed ['VariableStepMetricsCollection', 'VariableStepMSEs']`: the
    mvai provider swaps the FixedStep* classes named in train.py for
    VariableStep* ones, and those are fan-out containers.
    """

    def test_feeds_children_and_keeps_container_bookkeeping(self) -> None:
        from torchrec.metrics.ctr import CTRMetricComputation

        kwargs = {
            "my_rank": 0,
            "batch_size": _N,
            "n_tasks": _TASKS,
            "window_size": 1_000_000,
        }
        child = _FakeFixedStepRecMetric(CTRMetricComputation(**kwargs))
        collection = _FakeCollection({"_ctr": child})

        predictions, labels, weights = _inputs(torch.float64)
        states = compute_metric_states(predictions, labels, weights)
        unfed = update_ranking_metrics_from_states(
            states,
            num_samples=_N,
            num_real_candidates=1234,
            step_metrics=[collection],
        )

        self.assertEqual(unfed, [])
        self.assertTrue(torch.equal(child.rec_metric.ctr_denom, states[W_SUM]))
        self.assertTrue(collection._updated)
        self.assertEqual(collection._itrn, 1)
        # Reported against real examples, like the container's own update.
        self.assertEqual(collection._local_report_interval_num_examples, 1234)

    def test_container_with_no_rec_metric_children_is_still_reported(self) -> None:
        empty = _FakeCollection({})
        predictions, labels, weights = _inputs(torch.float64)
        unfed = update_ranking_metrics_from_states(
            compute_metric_states(predictions, labels, weights),
            num_samples=_N,
            num_real_candidates=_N,
            step_metrics=[empty],
        )
        self.assertEqual(unfed, [empty])


class TaskSliceEquivalenceTest(unittest.TestCase):
    """Reduce-then-slice must equal slice-then-reduce.

    The host path index_selects `[tasks, n]` prediction/label/weight and then
    reduces; this change reduces to `[tasks]` and index_selects that. A 4-chip
    A/B produced NE = eta and CTR = 0 from the in-graph path while
    effective_sample_cnt (fed the UNSLICED nnz vector) matched exactly, which
    points at the slicing rather than at the reduction.
    """

    def test_matches_the_host_path_under_a_task_subset(self) -> None:
        from torchrec.metrics.ne import NEMetricComputation

        predictions, labels, weights = _inputs(torch.float64)
        indices = torch.tensor([0, 2, 4])

        kwargs = {"my_rank": 0, "batch_size": _N, "window_size": 1_000_000}
        reference = NEMetricComputation(n_tasks=len(indices), **kwargs)
        reference.update(
            predictions=torch.index_select(predictions, 0, indices).clone(),
            labels=torch.index_select(labels, 0, indices),
            weights=torch.index_select(weights, 0, indices),
        )

        under_test = NEMetricComputation(n_tasks=len(indices), **kwargs)
        mapped = ne_states(compute_metric_states(predictions, labels, weights))
        _accumulate(
            under_test,
            {k: torch.index_select(v, 0, indices) for k, v in mapped.items()},
            num_samples=_N,
        )

        for name in ("cross_entropy_sum", "weighted_num_samples", "pos_labels"):
            got = getattr(under_test, name)
            expected = getattr(reference, name)
            self.assertTrue(
                torch.equal(got, expected),
                f"{name}: got {got.tolist()} expected {expected.tolist()}",
            )
        # nn.Module.__getattr__ types as `Module | Tensor`, so the chained
        # `.all()` lands on the Module arm; the loop above dodges it only
        # because getattr() is Any.
        weighted = cast(torch.Tensor, under_test.weighted_num_samples)
        self.assertTrue(bool(weighted.all()))


class ClampBoundaryTest(unittest.TestCase):
    """Predictions at the eta clamp, which is the only place fp32 and fp64 part.

    A 4-chip run returned `wce_sum` as all-NaN while the other seven
    accumulators were fine. torchrec clamps AFTER `predictions.double()`, so
    `1 - (1 - 1e-12)` is ~1e-12; in fp32 `1 - 1e-12` rounds to exactly 1.0, the
    same expression is 0.0, `log2(0)` is -inf and `0 * -inf` is NaN. Clamping
    the complement directly avoids ever forming that subtraction.
    """

    def _boundary(
        self, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Saturated predictions with the OPPOSING weight term non-zero.

        The obvious data -- p=1 with label 1, p=0 with label 0 -- hides the
        whole problem, because it zeroes `w1ml` exactly where p saturates high
        and `wl` where it saturates low. Two earlier versions of this test
        passed for that reason. Labels are chosen against the saturation here.
        """
        predictions = torch.tensor([[1.0, 0.0, 0.5, 1.0]], dtype=dtype)
        labels = torch.tensor([[0.0, 1.0, 1.0, 0.0]], dtype=dtype)
        weights = torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=dtype)
        return predictions, labels, weights

    def test_fp32_cross_entropy_is_finite(self) -> None:
        """The NaN regression guard. This is the one that matters."""
        states = compute_metric_states(*self._boundary(torch.float32))
        wce = states[WCE_SUM]
        self.assertTrue(
            bool(torch.isfinite(wce).all()), f"non-finite wce_sum: {wce.tolist()}"
        )

    def test_the_other_states_are_unaffected_by_saturation(self) -> None:
        """Only the cross entropy takes a log; the rest are plain sums and stay
        bit-exact against torchrec even at the boundary."""
        predictions, labels, weights = self._boundary(torch.float64)
        expected = _torchrec_reference(predictions, labels, weights)
        states = compute_metric_states(predictions, labels, weights)
        got = {
            "ne": ne_states(states),
            "ctr": ctr_states(states),
            "mse": mse_states(states),
            "calibration": calibration_states(states),
        }
        for metric, ref in expected.items():
            for name, ref_value in ref.items():
                if name == _CE_STATE:
                    continue
                self.assertTrue(
                    torch.equal(got[metric][name], ref_value),
                    f"{metric}/{name} not bit-exact at the clamp boundary",
                )

    def test_cross_entropy_deviates_from_torchrec_by_a_bounded_amount(self) -> None:
        """Documents a real, permanent deviation rather than asserting it away.

        `clamp(1 - p)` and `1 - clamp(p)` disagree once p passes the ceiling:
        torchrec's fp64 subtraction yields ~9.9998e-13 where clamping the
        complement yields exactly 1e-12. Reproducing torchrec's value would mean
        reproducing its rounding artifact, which fp32 cannot do anyway -- even
        `1 - p` rounds differently there. So the deviation is bounded and
        recorded, not eliminated.
        """
        predictions, labels, weights = self._boundary(torch.float64)
        expected = get_ne_states(labels, predictions.clone(), weights, NE_ETA)
        got = ne_states(compute_metric_states(predictions, labels, weights))

        ce_got, ce_ref = got[_CE_STATE], expected[_CE_STATE]
        self.assertFalse(
            torch.equal(ce_got, ce_ref),
            "cross_entropy_sum is now bit-exact at the boundary -- if the clamp "
            "was reworked to match torchrec, drop this test and tighten "
            "ClampBoundaryTest and the fp32 tolerance accordingly",
        )
        rel = float(((ce_got - ce_ref).abs() / ce_ref.abs()).max())
        self.assertLess(rel, 1e-5, f"deviation grew to {rel}")


class UnpadWindowTest(unittest.TestCase):
    """The metric window is sized in ROWS, so crediting it padded rows makes its
    true length depend on the pad fraction.
    """

    def _window_after(self, pad: int, *, unpad: bool) -> torch.Tensor:
        from torchrec.metrics.ne import NEMetricComputation

        predictions, labels, weights = _inputs(torch.float64, n=_N)
        real_rows = _N
        if pad:
            g = torch.Generator().manual_seed(11)
            predictions = torch.cat(
                [predictions, torch.rand((_TASKS, pad), generator=g).double()], dim=-1
            )
            labels = torch.cat(
                [labels, torch.rand((_TASKS, pad), generator=g).double()], dim=-1
            )
            weights = torch.cat(
                [weights, torch.zeros((_TASKS, pad), dtype=torch.float64)], dim=-1
            )

        states = compute_metric_states(predictions, labels, weights)
        # window_size chosen so eviction actually happens; that is the whole
        # mechanism the padded accounting corrupts.
        computation = NEMetricComputation(
            my_rank=0, batch_size=_N, n_tasks=_TASKS, window_size=_N * 2
        )
        metric = _FakeFixedStepRecMetric(computation)
        for _ in range(4):
            update_ranking_metrics_from_states(
                states,
                num_samples=(real_rows if unpad else int(states[PADDED_ROWS].item())),
                num_real_candidates=real_rows,
                cls_metrics=[metric],
                cls_indices=None,
            )
        return computation.get_window_state("weighted_num_samples").clone()

    def test_window_is_padding_invariant_when_unpadded(self) -> None:
        """What the pipeline now always does; there is no flag."""
        self.assertTrue(
            torch.equal(
                self._window_after(0, unpad=True),
                self._window_after(2048, unpad=True),
            ),
            "window state moved with the pad fraction",
        )

    def test_padded_row_accounting_would_still_be_wrong(self) -> None:
        """The bug this closed, kept as a guard on the test above."""
        self.assertFalse(
            torch.equal(
                self._window_after(0, unpad=False),
                self._window_after(2048, unpad=False),
            )
        )
