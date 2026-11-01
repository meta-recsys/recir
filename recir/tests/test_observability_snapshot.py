# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the step compile and the graphs its stages write."""

from __future__ import annotations

import contextlib
import logging
import pickle
import unittest
from typing import Any
from unittest import mock

import torch
import torch.nn as nn
from recir.observability import compiles
from recir.observability.compiles import StepCompile, TlparseCompile, TraceRecord
from recir.observability.graph_dump import GraphDump
from torch import fx
from torch._logging._internal import trace_log

_LOGGER = "recir.observability"
_RESERVED = {"rank", "frame_id", "frame_compile_id", "attempt", "has_payload"}


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).relu()


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
        stack = self.capture.events[0][1]["dynamo_start"]["stack"]
        self.assertEqual(
            [f["name"] for f in stack[-2:]],
            ["test_compiling_writes_one_compile", "recir step compile"],
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


def _compile_from(module_name: str, step: StepCompile) -> None:
    # Enters step.compiling() from a function defined in module_name.
    namespace: dict[str, Any] = {"__name__": module_name, "step": step}
    exec("def entry():\n    with step.compiling():\n        pass\n", namespace)
    namespace["entry"]()


class StackTest(unittest.TestCase):
    def setUp(self) -> None:
        self.capture = _Capture()
        trace_log.addHandler(self.capture)
        self.addCleanup(trace_log.removeHandler, self.capture)
        gm = fx.symbolic_trace(_ToyModel())
        self.step = StepCompile(TraceRecord.of(_ToyModel.forward, gm))

    def caller(self) -> str:
        stack = self.capture.events[0][1]["dynamo_start"]["stack"]
        self.assertEqual(stack[-1]["name"], "recir step compile")
        return stack[-2]["name"]

    def test_stack_ends_at_the_last_call_into_recir(self) -> None:
        _compile_from("recir.compiler.entry", self.step)
        self.assertEqual(self.caller(), "_compile_from")

    def test_recir_examples_are_callers(self) -> None:
        _compile_from("recir.examples.demo", self.step)
        self.assertEqual(self.caller(), "entry")

    def test_recir_fb_is_a_caller(self) -> None:
        _compile_from("recir.fb.mvai.launcher", self.step)
        self.assertEqual(self.caller(), "entry")

    def test_tests_inside_recir_packages_are_callers(self) -> None:
        _compile_from("recir.training.tests.test_compile", self.step)
        self.assertEqual(self.caller(), "entry")

    def test_a_library_module_named_like_a_test_is_internal(self) -> None:
        _compile_from("recir.testing_utils.entry", self.step)
        self.assertEqual(self.caller(), "_compile_from")

    def test_context_manager_plumbing_is_skipped(self) -> None:
        @contextlib.contextmanager
        def wrapped() -> Any:
            with self.step.compiling():
                yield

        with wrapped():
            pass
        self.assertEqual(self.caller(), "wrapped")
