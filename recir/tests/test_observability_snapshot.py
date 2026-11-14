# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the step compile ``compile_from_gm`` writes, and the graphs its stages write."""

from __future__ import annotations

import logging
import os
import pickle
import unittest
from typing import Any
from unittest import mock

import recir
import torch
import torch.nn as nn
from recir.compiler import compilation
from recir.compiler.tracing import _GraphSpec, _trace_graphs, _trace_record
from recir.observability import compiles
from recir.observability.compiles import StepCompile, TlparseCompile, TraceRecord
from recir.observability.graph_dump import GraphDump
from recir.observability.settings import Settings, Tier
from recir.tests.test_utils import _NoopBackend, _PassthroughParallel
from recir.training.compile import compile_step
from torch import fx
from torch._dynamo import convert_frame
from torch._logging._internal import trace_log

_LOGGER = "recir.observability"
_RESERVED = {"rank", "frame_id", "frame_compile_id", "attempt", "has_payload"}


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).relu()


def _train_step(model: nn.Module, opt: torch.optim.Optimizer, x: torch.Tensor) -> None:
    opt.zero_grad()
    model(x).sum().backward()
    opt.step()


def _step_inputs() -> tuple[nn.Module, torch.optim.Optimizer, torch.Tensor]:
    model = _ToyModel()
    opt = torch.optim.Adam(model.parameters(), lr=0.01, foreach=True)
    x = torch.randn(2, 4)
    # One eager step materializes Adam's state, as on a resumed job.
    _train_step(model, opt, x)
    return model, opt, x


def _trace() -> tuple[fx.GraphModule, Any]:
    return recir.trace(_train_step, *_step_inputs())


def _compile_traced(gm: fx.GraphModule, flat_state: Any) -> Any:
    return recir.compile_from_gm(
        gm, flat_state, parallel_mode=_PassthroughParallel(), backend=_NoopBackend()
    )


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[tuple[str, dict[str, Any], str | None]] = []

    def emit(self, record: logging.LogRecord) -> None:
        metadata = record.__dict__["metadata"]
        # Nothing in these tests runs Dynamo, so every event with a frame id is RecIR's.
        if "frame_id" in metadata:
            name = next(k for k in metadata if k not in _RESERVED)
            self.events.append((name, metadata, record.__dict__["payload"]))


class StepCompileTest(unittest.TestCase):
    def setUp(self) -> None:
        self.capture = _Capture()
        trace_log.addHandler(self.capture)
        self.addCleanup(trace_log.removeHandler, self.capture)
        self.gm: fx.GraphModule = fx.symbolic_trace(_ToyModel())
        self.step = StepCompile(TraceRecord.of(_ToyModel.forward, self.gm))

    def names(self) -> list[str]:
        return [
            m["artifact"]["name"] if n == "artifact" else n
            for n, m, _ in self.capture.events
        ]

    def metrics(self) -> dict[str, Any]:
        name, metadata, _ = self.capture.events[-1]
        self.assertEqual(name, "compilation_metrics")
        return metadata["compilation_metrics"]

    def test_compiling_writes_one_compile(self) -> None:
        with self.step.compiling() as started:
            self.assertIs(started, self.step)
        self.assertEqual(
            self.names(), ["dynamo_start", "recir_trace_graph", "compilation_metrics"]
        )
        for _, metadata, _ in self.capture.events:
            self.assertEqual(
                (metadata["frame_id"], metadata["frame_compile_id"]),
                (self.step.frame_id, 0),
            )
        self.assertEqual(self.capture.events[1][2], GraphDump(self.gm).text())

    def test_metrics_describe_the_traced_graph(self) -> None:
        with self.step.compiling():
            pass
        metrics = self.metrics()
        code = _ToyModel.forward.__code__
        self.assertEqual(metrics["co_name"], "recir step compile")
        self.assertEqual(
            (metrics["co_filename"], metrics["co_firstlineno"]),
            (code.co_filename, code.co_firstlineno),
        )
        counts = GraphDump(self.gm).counts
        self.assertEqual(metrics["graph_node_count"], counts["nodes"])
        self.assertEqual(metrics["graph_input_count"], counts["placeholder"])
        self.assertIsNone(metrics["fail_type"])
        self.assertEqual(metrics["restart_reasons"], [])

    def test_raising_work_reports_the_failure(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "boom"):
            with self.step.compiling():
                raise RuntimeError("boom")
        metrics = self.metrics()
        self.assertEqual(metrics["fail_type"], "RuntimeError")
        self.assertEqual(metrics["fail_reason"], "raised after RecIR stage trace")
        self.assertNotIn("boom", str(metrics))

    def test_each_step_compile_gets_its_own_id(self) -> None:
        second = StepCompile(TraceRecord.of(_ToyModel.forward, self.gm))
        self.assertEqual(second.frame_id, self.step.frame_id + 1)

    def test_frame_id_is_reserved_under_dynamos_compile_lock(self) -> None:
        from torch._dynamo import convert_frame

        lock = mock.MagicMock()
        with mock.patch.object(convert_frame, "compile_lock", lock):
            TlparseCompile._reserve_frame_id()
        lock.__enter__.assert_called_once()

    def test_no_sink_writes_no_payload(self) -> None:
        self.assertTrue(TlparseCompile.has_sink())
        body = mock.Mock(return_value="text")
        with mock.patch.object(trace_log, "handlers", []):
            self.assertFalse(TlparseCompile.has_sink())
            self.step.write("recir_x_graph", body)
        body.assert_not_called()

    def test_write_fault_is_logged(self) -> None:
        with (
            mock.patch.object(compiles, "trace_structured", side_effect=RuntimeError),
            self.assertLogs(_LOGGER, level="WARNING") as logs,
        ):
            self.step.write("recir_x_graph", lambda: "text")
        self.assertIn("recir_x_graph not written: RuntimeError", logs.output[0])

    def test_start_fault_runs_the_work_as_no_compile(self) -> None:
        with (
            mock.patch.object(TlparseCompile, "_stack", side_effect=RuntimeError),
            self.assertLogs(_LOGGER, level="WARNING") as logs,
        ):
            with self.step.compiling() as started:
                self.assertIsNone(started)
        self.assertIn("recir step compile not started: RuntimeError", logs.output[0])
        self.assertEqual(self.capture.events, [])

    def test_start_fault_lets_the_works_exception_through(self) -> None:
        with (
            mock.patch.object(TlparseCompile, "_stack", side_effect=RuntimeError),
            self.assertLogs(_LOGGER, level="WARNING"),
            self.assertRaisesRegex(ValueError, "work"),
        ):
            with self.step.compiling():
                raise ValueError("work")

    def test_finish_fault_never_fails_the_work(self) -> None:
        with (
            mock.patch.object(StepCompile, "_fail_reason", side_effect=RuntimeError),
            self.assertLogs(_LOGGER, level="WARNING") as logs,
            self.assertRaisesRegex(ValueError, "work"),
        ):
            with self.step.compiling():
                raise ValueError("work")
        self.assertIn("recir step compile not finished: RuntimeError", logs.output[0])

    def test_trace_record_pickles(self) -> None:
        record = TraceRecord.of(_ToyModel.forward, self.gm)
        self.assertEqual(pickle.loads(pickle.dumps(record)), record)


def _new_step() -> StepCompile:
    return StepCompile(
        TraceRecord.of(_ToyModel.forward, fx.symbolic_trace(_ToyModel()))
    )


@compiles.entry_point
def _compile_now() -> None:
    with _new_step().compiling():
        pass


@compiles.entry_point
def _reserve() -> StepCompile:
    return _new_step()


@compiles.entry_point
def _outer_entry() -> None:
    _compile_now()


class StackTest(unittest.TestCase):
    def setUp(self) -> None:
        self.capture = _Capture()
        trace_log.addHandler(self.capture)
        self.addCleanup(trace_log.removeHandler, self.capture)

    def caller(self) -> str:
        stack = self.capture.events[0][1]["dynamo_start"]["stack"]
        self.assertEqual(stack[-1]["name"], "recir step compile")
        return stack[-2]["name"]

    def test_stack_ends_at_the_call_into_the_entry_point(self) -> None:
        _compile_now()
        self.assertEqual(
            self.caller(), "test_stack_ends_at_the_call_into_the_entry_point"
        )

    def test_nested_entry_points_keep_the_outermost_caller(self) -> None:
        _outer_entry()
        self.assertEqual(
            self.caller(), "test_nested_entry_points_keep_the_outermost_caller"
        )

    def test_stack_is_taken_on_construction(self) -> None:
        step = _reserve()
        with step.compiling():
            pass
        self.assertEqual(self.caller(), "test_stack_is_taken_on_construction")

    def test_outside_an_entry_point_the_stack_is_empty(self) -> None:
        with _new_step().compiling():
            pass
        stack = self.capture.events[0][1]["dynamo_start"]["stack"]
        self.assertEqual([f["name"] for f in stack], ["recir step compile"])


def _set_tier(case: unittest.TestCase, tier: Tier) -> None:
    patcher = mock.patch.object(Settings, "_loaded", Settings(tier=tier))
    patcher.start()
    case.addCleanup(patcher.stop)


class CompileFromGmTest(unittest.TestCase):
    def setUp(self) -> None:
        self.capture = _Capture()
        trace_log.addHandler(self.capture)
        self.addCleanup(trace_log.removeHandler, self.capture)
        _set_tier(self, Tier.INDEX)

    def names(self) -> list[str]:
        return [
            m["artifact"]["name"] if n == "artifact" else n
            for n, m, _ in self.capture.events
        ]

    def metrics(self) -> dict[str, Any]:
        name, metadata, _ = self.capture.events[-1]
        self.assertEqual(name, "compilation_metrics")
        return metadata["compilation_metrics"]

    def caller(self) -> str:
        stack = self.capture.events[0][1]["dynamo_start"]["stack"]
        self.assertEqual(stack[-1]["name"], "recir step compile")
        return stack[-2]["name"]

    def test_trace_alone_writes_nothing(self) -> None:
        gm, _ = _trace()
        self.assertEqual(self.capture.events, [])
        self.assertIsNotNone(_trace_record(gm))

    def test_compile_writes_the_traced_graph_and_closes(self) -> None:
        gm, flat_state = _trace()
        text = GraphDump(gm).text()
        _compile_traced(gm, flat_state)
        self.assertEqual(
            self.names(), ["dynamo_start", "recir_trace_graph", "compilation_metrics"]
        )
        self.assertEqual(self.capture.events[1][2], text)
        metrics = self.metrics()
        code = _train_step.__code__
        self.assertEqual(
            (metrics["co_filename"], metrics["co_firstlineno"]),
            (code.co_filename, code.co_firstlineno),
        )
        self.assertEqual(metrics["graph_node_count"], GraphDump(gm).counts["nodes"])
        self.assertIsNone(metrics["fail_type"])

    def test_step_compile_sits_under_the_call_into_compile_from_gm(self) -> None:
        gm, flat_state = _trace()
        recir.compile_from_gm(
            gm, flat_state, parallel_mode=_PassthroughParallel(), backend=_NoopBackend()
        )
        self.assertEqual(
            self.caller(), "test_step_compile_sits_under_the_call_into_compile_from_gm"
        )

    def test_step_compile_sits_under_the_call_into_recir_compile(self) -> None:
        @recir.compile(parallel_mode=_PassthroughParallel(), backend=_NoopBackend())
        def train_step(
            model: nn.Module, opt: torch.optim.Optimizer, x: torch.Tensor
        ) -> None:
            _train_step(model, opt, x)

        # recir.compile traces and compiles on the first call.
        train_step(*_step_inputs())
        self.assertEqual(
            self.caller(), "test_step_compile_sits_under_the_call_into_recir_compile"
        )

    def test_step_compile_sits_under_the_call_into_compile_step(self) -> None:
        model, opt, x = _step_inputs()
        compile_step(
            _train_step,
            model,
            [opt],
            x,
            parallel_mode=_PassthroughParallel(),
            backend=_NoopBackend(),
            flatten_batch=lambda batch, global_loss_denom: (batch,),
        )
        self.assertEqual(
            self.caller(), "test_step_compile_sits_under_the_call_into_compile_step"
        )

    def test_each_compile_writes_its_own_entry(self) -> None:
        gm, flat_state = _trace()
        _compile_traced(gm, flat_state)
        _compile_traced(gm, flat_state)
        starts = [m for n, m, _ in self.capture.events if n == "dynamo_start"]
        self.assertEqual(len(starts), 2)
        self.assertNotEqual(starts[0]["frame_id"], starts[1]["frame_id"])

    def test_traced_graph_and_counts_exclude_post_trace_edits(self) -> None:
        gm, flat_state = _trace()
        counts = GraphDump(gm).counts
        output = next(n for n in gm.graph.nodes if n.op == "output")
        with gm.graph.inserting_before(output):
            gm.graph.create_node(
                "call_function",
                torch.ops.aten.zeros.default,
                ([1],),
                name="post_trace_edit",
            )
        gm.recompile()
        _compile_traced(gm, flat_state)
        text = self.capture.events[1][2]
        assert text is not None
        self.assertNotIn("post_trace_edit", text)
        self.assertEqual(self.metrics()["graph_node_count"], counts["nodes"])

    def test_raising_compile_reports_the_failure(self) -> None:
        gm, flat_state = _trace()
        with (
            mock.patch.object(
                compilation, "_compile_from_gm", side_effect=RuntimeError("boom")
            ),
            self.assertRaisesRegex(RuntimeError, "boom"),
        ):
            _compile_traced(gm, flat_state)
        metrics = self.metrics()
        self.assertEqual(metrics["fail_type"], "RuntimeError")
        self.assertEqual(metrics["fail_reason"], "raised after RecIR stage trace")

    def test_start_fault_never_fails_the_compile(self) -> None:
        gm, flat_state = _trace()
        with (
            mock.patch.object(TlparseCompile, "_stack", side_effect=RuntimeError),
            self.assertLogs(_LOGGER, level="WARNING") as logs,
        ):
            compiled = _compile_traced(gm, flat_state)
        self.assertIsNotNone(compiled.gm)
        self.assertIn("recir step compile not started: RuntimeError", logs.output[0])
        self.assertEqual(self.capture.events, [])

    def test_tier_off_spends_no_compile_id(self) -> None:
        _set_tier(self, Tier.OFF)
        frames_before = convert_frame.FRAME_COUNTER
        _compile_traced(*_trace())
        self.assertEqual(self.capture.events, [])
        self.assertEqual(convert_frame.FRAME_COUNTER, frames_before)

    def test_no_handler_spends_no_compile_id(self) -> None:
        frames_before = convert_frame.FRAME_COUNTER
        with mock.patch.object(trace_log, "handlers", []):
            _compile_traced(*_trace())
        self.assertEqual(convert_frame.FRAME_COUNTER, frames_before)

    def test_handler_gone_by_compile_opens_nothing(self) -> None:
        gm, flat_state = _trace()
        frames_before = convert_frame.FRAME_COUNTER
        with mock.patch.object(trace_log, "handlers", []):
            _compile_traced(gm, flat_state)
        self.assertEqual(convert_frame.FRAME_COUNTER, frames_before)

    def test_trace_fault_never_fails_trace(self) -> None:
        with (
            mock.patch.object(GraphDump, "text", side_effect=RuntimeError),
            self.assertLogs(_LOGGER, level="WARNING") as logs,
        ):
            gm, _ = _trace()
        self.assertIsNone(_trace_record(gm))
        self.assertIn("trace not kept: RuntimeError", logs.output[0])

    def test_traced_gm_pickles_with_its_trace_record(self) -> None:
        gm, _ = _trace()
        restored = pickle.loads(pickle.dumps(gm))
        self.assertEqual(_trace_record(restored), _trace_record(gm))

    def test_graphs_traced_together_each_keep_a_record(self) -> None:
        spec = _GraphSpec(_train_step, _step_inputs(), None)
        traced = _trace_graphs([spec, spec])
        self.assertTrue(all(_trace_record(gm) is not None for gm, _ in traced))
        self.assertEqual(len(traced), 2)


class SettingsTest(unittest.TestCase):
    def test_tier_ignores_case_and_whitespace(self) -> None:
        with mock.patch.dict(os.environ, {"RECIR_SNAPSHOT_TIER": " Index\n"}):
            self.assertIs(Settings.from_env().tier, Tier.INDEX)

    def test_empty_tier_means_off(self) -> None:
        with mock.patch.dict(os.environ, {"RECIR_SNAPSHOT_TIER": " "}):
            self.assertIs(Settings.from_env().tier, Tier.OFF)

    def test_malformed_tier_turns_snapshots_off(self) -> None:
        with (
            mock.patch.dict(os.environ, {"RECIR_SNAPSHOT_TIER": "bogus"}),
            self.assertLogs(_LOGGER, level="WARNING"),
        ):
            settings = Settings.from_env()
        self.assertFalse(settings.enabled)
