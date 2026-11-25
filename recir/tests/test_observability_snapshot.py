# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the step compile ``compile_from_gm`` writes, and the graphs its stages write."""

from __future__ import annotations

import contextlib
import functools
import logging
import os
import pickle
import unittest
from contextlib import nullcontext
from typing import Any, Callable, Iterator
from unittest import mock

import recir
import torch
import torch.nn as nn
from recir.backends.inductor import InductorBackend
from recir.compiler import compilation
from recir.compiler.tracing import _GraphSpec, _trace_graphs, _trace_record
from recir.composite import Part
from recir.lowering import lowering_utils
from recir.lowering.lowering_inputs import WHOLE
from recir.lowering.lowering_utils import ExtractedSubgraph, LazilyBuiltModule
from recir.lowering.partial_lower import partial_lower
from recir.observability import builds, compiles, snapshot
from recir.observability.builds import BuildTicket, SubgraphBuild
from recir.observability.compiles import StepCompile, TlparseCompile, TraceRecord
from recir.observability.graph_dump import GraphDump
from recir.observability.lowering_observer import SnapshotLoweringObserver
from recir.observability.settings import Settings, Tier
from recir.pipeline import PipelinedGraphModule, SteadyStateGraphModule
from recir.region_transforms import RegionTransform
from recir.tests.test_utils import _graph, _NoopBackend, _PassthroughParallel
from recir.training.compile import compile_step
from torch import fx
from torch._dynamo import convert_frame
from torch._dynamo.utils import counters
from torch._guards import CompileContext
from torch._inductor.utils import fresh_cache
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
            self.names(),
            [
                "dynamo_start",
                "recir_trace_graph",
                "recir_assembly_graph",
                "compilation_metrics",
            ],
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

    def test_failure_after_assembly_names_it(self) -> None:
        class _RaisingBackend(_NoopBackend):
            def dispatch(self, gm: fx.GraphModule) -> nn.Module:
                raise RuntimeError("dispatch")

        gm, flat_state = _trace()
        with self.assertRaisesRegex(RuntimeError, "dispatch"):
            recir.compile_from_gm(
                gm,
                flat_state,
                parallel_mode=_PassthroughParallel(),
                backend=_RaisingBackend(),
            )
        self.assertEqual(
            self.metrics()["fail_reason"], "raised after RecIR stage assembly"
        )

    def test_compile_work_runs_outside_any_compile_id(self) -> None:
        seen = []

        class _RecordingBackend(_NoopBackend):
            def dispatch(self, gm: fx.GraphModule) -> nn.Module:
                # Runs after the trace and assembly writes pushed their id.
                seen.append(CompileContext.current_compile_id())
                return gm

        recir.compile_from_gm(
            *_trace(), parallel_mode=_PassthroughParallel(), backend=_RecordingBackend()
        )
        self.assertIn("recir_assembly_graph", self.names())
        self.assertEqual(seen, [None])

    def test_failure_before_assembly_names_the_trace(self) -> None:
        class _FailingFinalize(RegionTransform):
            def finalize_compiled_graph(self, gm: fx.GraphModule) -> None:
                raise RuntimeError("finalize")

        gm, flat_state = _trace()
        with self.assertRaisesRegex(RuntimeError, "finalize"):
            recir.compile_from_gm(
                gm,
                flat_state,
                parallel_mode=_PassthroughParallel(),
                backend=_NoopBackend(),
                region_transforms=[_FailingFinalize()],
            )
        self.assertNotIn("recir_assembly_graph", self.names())
        self.assertEqual(
            self.metrics()["fail_reason"], "raised after RecIR stage trace"
        )

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

    def test_create_fault_never_fails_the_compile(self) -> None:
        gm, flat_state = _trace()
        with (
            mock.patch.object(StepCompile, "__init__", side_effect=RuntimeError),
            self.assertLogs("recir.observability.guard", level="WARNING") as logs,
        ):
            compiled = _compile_traced(gm, flat_state)
        self.assertIsNotNone(compiled.gm)
        self.assertIn("step compile not created: RuntimeError", logs.output[0])
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


@contextlib.contextmanager
def _dispatching(backend: str = "InductorBackend") -> Iterator[StepCompile]:
    """A running step compile whose backend dispatch is observed, as in compile_from_gm."""
    gm = fx.symbolic_trace(_ToyModel())
    with StepCompile(TraceRecord.of(_ToyModel.forward, gm)).compiling() as step:
        assert step is not None
        with snapshot.dispatching(step, backend):
            yield step


def _observer() -> SnapshotLoweringObserver:
    observer = snapshot.lowering_observer()
    assert isinstance(observer, SnapshotLoweringObserver)
    return observer


def _lower(
    observer: SnapshotLoweringObserver, part: Part, gm: fx.GraphModule
) -> nn.Module:
    """Lower ``gm`` as the dispatch input ``part``, reporting to ``observer`` as ``lower_inputs`` does."""
    return partial_lower(
        gm, on_subgraph_extracted=functools.partial(observer.subgraph_extracted, part)
    )


@compiles.entry_point
def _lower_from_entry() -> nn.Module:
    with _dispatching():
        return _lower(_observer(), WHOLE, _graph())


def _report_extracted(
    observer: SnapshotLoweringObserver, part: Part, gm: fx.GraphModule, count: int
) -> list[LazilyBuiltModule]:
    """Report ``count`` subgraphs extracted from ``part`` without lowering anything."""
    modules = [LazilyBuiltModule() for _ in range(count)]
    for index, module in enumerate(modules):
        observer.subgraph_extracted(
            part, ExtractedSubgraph(f"subgraph_{index}", gm, module)
        )
    return modules


def _lazy_modules(*lowered: nn.Module) -> list[LazilyBuiltModule]:
    return [
        m
        for root in lowered
        for m in root.modules()
        if isinstance(m, LazilyBuiltModule)
    ]


class _SnapshotTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.capture = _Capture()
        trace_log.addHandler(self.capture)
        self.addCleanup(trace_log.removeHandler, self.capture)
        _set_tier(self, Tier.INDEX)

    def metrics(self) -> dict[str, Any]:
        name, metadata, _ = self.capture.events[-1]
        self.assertEqual(name, "compilation_metrics")
        return metadata["compilation_metrics"]

    def build_names(self) -> list[str]:
        """The builds that started, in start order."""
        return [
            m["dynamo_start"]["stack"][-1]["name"]
            for n, m, _ in self.capture.events
            if n == "dynamo_start"
            and m["dynamo_start"]["stack"][-1]["name"].startswith("recir build")
        ]

    def build_frame(self, name: str) -> int:
        [frame] = [
            m["frame_id"]
            for n, m, _ in self.capture.events
            if n == "dynamo_start" and m["dynamo_start"]["stack"][-1]["name"] == name
        ]
        return frame

    def events_at(self, frame: int) -> list[str]:
        return [
            m["artifact"]["name"] if name == "artifact" else name
            for name, m, _ in self.capture.events
            if m["frame_id"] == frame
        ]

    def build(self, lowered: nn.Module, compile_fx: Any) -> None:
        with mock.patch(
            "torch._inductor.compile_fx.compile_fx", side_effect=compile_fx
        ):
            lowered(torch.randn(4))


def _passthrough(sub_gm: fx.GraphModule, inputs: Any, **kwargs: Any) -> Any:
    return sub_gm


class SubgraphBuildTest(_SnapshotTestCase):
    def test_extraction_takes_no_compile_id(self) -> None:
        with _dispatching():
            frames = convert_frame.FRAME_COUNTER
            _report_extracted(_observer(), WHOLE, _graph(), 3)
            self.assertEqual(convert_frame.FRAME_COUNTER, frames)

    def test_builds_are_named_after_their_part_and_numbered_as_they_run(
        self,
    ) -> None:
        gm = _graph()
        with _dispatching() as step:
            observer = _observer()
            first = _report_extracted(observer, Part("route_start", 0), gm, 1)
            later = _report_extracted(observer, Part("compute", 2), gm, 2)
        for module in [later[1], first[0]]:
            with module.compile_scope():
                pass
        self.assertEqual(
            self.build_names(),
            [
                "recir build stage2_compute subgraph_1",
                "recir build stage0_route_start subgraph_0",
            ],
        )
        self.assertGreater(
            self.build_frame("recir build stage0_route_start subgraph_0"),
            self.build_frame("recir build stage2_compute subgraph_1"),
        )
        self.assertGreater(
            self.build_frame("recir build stage2_compute subgraph_1"), step.frame_id
        )

    def test_real_build_writes_pt2_artifacts_under_its_compile_id(self) -> None:
        with _dispatching():
            lowered = _lower(_observer(), WHOLE, _graph())
        # A real, uncached Inductor build, so PT2 writes its own artifacts.
        with mock.patch.object(torch.compiler.config, "force_disable_caches", True):
            lowered(torch.randn(4))
        names = self.events_at(self.build_frame("recir build whole subgraph_0"))
        for artifact in (
            "aot_inference_graph",
            "inductor_post_grad_graph",
            "inductor_output_code",
        ):
            self.assertIn(artifact, names)
        self.assertIsNone(self.metrics()["fail_type"])

    def test_a_build_served_from_the_cache_still_closes(self) -> None:
        counters.clear()
        with (
            fresh_cache(),
            torch._inductor.config.patch(
                fx_graph_cache=True,
                fx_graph_remote_cache=False,
                autotune_remote_cache=False,
            ),
        ):
            for _ in range(2):
                with _dispatching():
                    lowered = _lower(_observer(), WHOLE, _graph())
                lowered(torch.randn(4))
        closes = [
            m["compilation_metrics"]
            for n, m, _ in self.capture.events
            if n == "compilation_metrics"
            and m["compilation_metrics"]["co_name"] == "recir build whole subgraph_0"
        ]
        self.assertEqual(len(closes), 2)
        self.assertEqual([c["fail_type"] for c in closes], [None, None])
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)

    def test_build_sits_under_the_call_that_extracted_it(self) -> None:
        lowered = _lower_from_entry()
        self.build(lowered, _passthrough)
        [start] = [
            m
            for name, m, _ in self.capture.events
            if name == "dynamo_start"
            and m["dynamo_start"]["stack"][-1]["name"] == "recir build whole subgraph_0"
        ]
        names = [frame["name"] for frame in start["dynamo_start"]["stack"]]
        self.assertEqual(
            names[-2:],
            [
                "test_build_sits_under_the_call_that_extracted_it",
                "recir build whole subgraph_0",
            ],
        )

    def test_no_build_outside_a_dispatch(self) -> None:
        self.assertIsNone(snapshot.lowering_observer())
        lowered = InductorBackend().dispatch(_graph())
        self.assertIs(lowered.get_submodule("subgraph_0").compile_scope, nullcontext)

    def test_no_build_when_the_compile_is_not_snapshotted(self) -> None:
        with snapshot.dispatching(None, "InductorBackend"):
            self.assertIsNone(snapshot.lowering_observer())

    def test_no_build_once_snapshots_are_off(self) -> None:
        with _dispatching():
            [module] = _lazy_modules(_lower(_observer(), WHOLE, _graph()))
        restored = pickle.loads(pickle.dumps(module.compile_scope))
        frames = convert_frame.FRAME_COUNTER
        seen = []
        with mock.patch.object(Settings, "_loaded", Settings(tier=Tier.OFF)):
            with restored():
                seen.append(CompileContext.current_compile_id())
        with mock.patch.object(trace_log, "handlers", []):
            with module.compile_scope():
                seen.append(CompileContext.current_compile_id())
        self.assertEqual(seen, [None, None])
        self.assertEqual(convert_frame.FRAME_COUNTER, frames)

    def test_observer_fault_never_fails_lowering(self) -> None:
        with (
            _dispatching(),
            mock.patch.object(BuildTicket, "__init__", side_effect=RuntimeError),
            self.assertLogs("recir.observability", level="WARNING") as logs,
        ):
            lowered = _lower(_observer(), WHOLE, _graph())
        self.assertIsInstance(lowered.get_submodule("subgraph_0"), nn.Module)
        self.assertIn("whole subgraph_0 build not reserved", logs.output[0])

    def test_partial_lower_builds_under_their_compile_id(self) -> None:
        def fake_compile_fx(
            sub_gm: fx.GraphModule, inputs: Any, **kwargs: Any
        ) -> fx.GraphModule:
            torch._logging.trace_structured(
                "artifact",
                metadata_fn=lambda: {"name": "fake_output_code", "encoding": "string"},
                payload_fn=lambda: "code",
            )
            return sub_gm

        with _dispatching():
            lowered = _lower(_observer(), Part("compute", 2), _graph())
        self.build(lowered, fake_compile_fx)
        name = "recir build stage2_compute subgraph_0"
        events = self.events_at(self.build_frame(name))
        self.assertEqual(
            (events[0], events[-1]), ("dynamo_start", "compilation_metrics")
        )
        # The backend's own artifact lands under the build's id.
        self.assertIn("fake_output_code", events)
        metrics = self.metrics()
        self.assertEqual(metrics["co_name"], name)
        self.assertIsNone(metrics["fail_type"])
        self.assertGreater(metrics["graph_op_count"], 0)

    def test_failed_build_is_reported(self) -> None:
        with _dispatching():
            lowered = InductorBackend().dispatch(_graph())
        with self.assertRaises(RuntimeError):
            self.build(lowered, mock.Mock(side_effect=RuntimeError("boom")))
        metrics = self.metrics()
        self.assertEqual(metrics["fail_type"], "RuntimeError")
        self.assertEqual(metrics["fail_reason"], "the InductorBackend build raised")
        self.assertEqual(metrics["co_name"], "recir build whole subgraph_0")

    def test_a_build_that_never_started_is_never_closed(self) -> None:
        with _dispatching():
            lowered = _lower(_observer(), WHOLE, _graph())
        events = len(self.capture.events)
        with (
            mock.patch.object(SubgraphBuild, "_start", side_effect=RuntimeError),
            self.assertLogs(_LOGGER, level="WARNING") as logs,
        ):
            self.build(lowered, _passthrough)
        self.assertNotIn(
            "compilation_metrics", [n for n, _, _ in self.capture.events[events:]]
        )
        self.assertIn("recir build whole subgraph_0 not started", logs.output[0])
        self.assertIsNone(CompileContext.current_compile_id())

    def test_create_fault_builds_with_no_compile_id(self) -> None:
        with _dispatching():
            lowered = _lower(_observer(), WHOLE, _graph())
        events = len(self.capture.events)
        seen = []

        def record(sub_gm: fx.GraphModule, inputs: Any, **kwargs: Any) -> Any:
            seen.append(CompileContext.current_compile_id())
            return sub_gm

        with (
            mock.patch.object(SubgraphBuild, "__init__", side_effect=RuntimeError),
            self.assertLogs("recir.observability.guard", level="WARNING") as logs,
        ):
            self.build(lowered, record)
        self.assertEqual(seen, [None])
        self.assertIn(
            "recir build whole subgraph_0 not created: RuntimeError", logs.output[0]
        )
        self.assertEqual(self.capture.events[events:], [])

    def test_setup_fault_builds_with_no_compile_id(self) -> None:
        with _dispatching():
            lowered = _lower(_observer(), WHOLE, _graph())
        events = len(self.capture.events)
        seen = []

        def record(sub_gm: fx.GraphModule, inputs: Any, **kwargs: Any) -> Any:
            seen.append(CompileContext.current_compile_id())
            return sub_gm

        with (
            mock.patch.object(builds, "compile_context", side_effect=RuntimeError),
            self.assertLogs(_LOGGER, level="WARNING"),
        ):
            self.build(lowered, record)
        self.assertEqual(seen, [None])
        self.assertEqual(self.capture.events[events:], [])

    def test_undo_fault_never_replaces_the_builds_exception(self) -> None:
        @contextlib.contextmanager
        def undo_raises(context: CompileContext) -> Iterator[None]:
            yield
            raise RuntimeError("undo")

        with _dispatching():
            lowered = _lower(_observer(), WHOLE, _graph())
        with (
            mock.patch.object(builds, "compile_context", undo_raises),
            self.assertLogs(_LOGGER, level="WARNING") as logs,
            self.assertRaisesRegex(ValueError, "build"),
        ):
            self.build(lowered, mock.Mock(side_effect=ValueError("build")))
        self.assertIn("setup not undone", logs.output[0])
        self.assertEqual(self.metrics()["fail_type"], "ValueError")

    def test_dispatch_names_builds_after_pipeline_stages(self) -> None:
        pipeline = mock.MagicMock(spec=PipelinedGraphModule)
        graphs = {"route_start": _graph(), "compute": _graph()}
        lowered: list[nn.Module] = []

        def lower_parts(lower: Callable[[Part, fx.GraphModule], nn.Module]) -> Any:
            for index, (name, graph) in enumerate(graphs.items()):
                lowered.append(lower(Part(name, index), graph))
            return pipeline

        pipeline.lower_parts = lower_parts
        with _dispatching():
            self.assertIs(InductorBackend().dispatch(pipeline), pipeline)
        for module in _lazy_modules(*lowered):
            with module.compile_scope():
                pass
        self.assertEqual(
            self.build_names(),
            [
                "recir build stage0_route_start subgraph_0",
                "recir build stage1_compute subgraph_0",
            ],
        )

    def test_dispatch_names_the_steady_state_build(self) -> None:
        steady = mock.MagicMock(spec=SteadyStateGraphModule)
        steady.steady_gm = _graph()
        steady.lower_parts = lambda lower: SteadyStateGraphModule.lower_parts(
            steady, lower
        )
        with _dispatching():
            InductorBackend().dispatch(steady)
        for module in _lazy_modules(steady.steady_gm):
            with module.compile_scope():
                pass
        self.assertEqual(self.build_names(), ["recir build steady subgraph_0"])

    def test_unbuilt_subgraph_pickles(self) -> None:
        with _dispatching():
            [module] = _lazy_modules(_lower(_observer(), WHOLE, _graph()))
        restored = pickle.loads(pickle.dumps(module.compile_scope))
        with restored():
            pass
        self.assertEqual(self.build_names(), ["recir build whole subgraph_0"])


class _FakeModule(LazilyBuiltModule):
    """An extracted module whose "compile" only records what it saw."""

    raises = False

    def __init__(self, gm: fx.GraphModule) -> None:
        super().__init__()
        self.gm = gm
        self.compiled = False

    def forward(self, *args: torch.Tensor, tag: str = "") -> Any:
        if not self.compiled:
            with self.compile_scope():
                if self.raises:
                    raise RuntimeError("fake build failed")
                self.compiled = True
        return self.gm(*args)


def _fake_lower(
    gm: fx.GraphModule,
    *,
    on_subgraph_extracted: Any = None,
    module_factory: Callable[[fx.GraphModule], nn.Module] = _FakeModule,
) -> nn.Module:
    """Extract every run of nodes, keeping profiler ops as TorchTPU does."""
    return lowering_utils.segment_and_lower(
        gm,
        compatibility_fn=lambda n: (True, ""),
        module_factory=module_factory,
        on_subgraph_extracted=on_subgraph_extracted,
    )


class _FakeLoweringBackend(_NoopBackend):
    def __init__(self, lower: Callable[..., nn.Module] = _fake_lower) -> None:
        super().__init__()
        self._lower = lower

    def dispatch(self, gm: fx.GraphModule) -> nn.Module:
        return self._lower_inputs(gm, self._lower)


class FakeBackendTest(_SnapshotTestCase):
    """A backend other than Inductor gets the same neutral builds."""

    def lower(self, gm: fx.GraphModule, **kwargs: Any) -> nn.Module:
        """Dispatch ``gm`` through the fake backend in a step compile; ``kwargs`` go to its lowerer."""
        backend = _FakeLoweringBackend(functools.partial(_fake_lower, **kwargs))
        with _dispatching("_FakeLoweringBackend"):
            return backend.dispatch(gm)

    def test_failed_neutral_build_names_its_backend(self) -> None:
        lowered = self.lower(_graph())
        with (
            mock.patch.object(_FakeModule, "raises", True),
            self.assertRaises(RuntimeError),
        ):
            lowered(torch.randn(4))
        self.assertEqual(
            self.metrics()["fail_reason"], "the _FakeLoweringBackend build raised"
        )

    def test_a_retried_build_gets_its_own_entry(self) -> None:
        lowered = self.lower(_graph())
        with (
            mock.patch.object(_FakeModule, "raises", True),
            self.assertRaises(RuntimeError),
        ):
            lowered(torch.randn(4))
        lowered(torch.randn(4))
        self.assertEqual(
            self.build_names(),
            ["recir build whole subgraph_0", "recir build whole subgraph_0"],
        )
        self.assertIsNone(self.metrics()["fail_type"])

    def test_module_without_the_build_hook_gets_no_build(self) -> None:
        lowered = self.lower(_graph(), module_factory=lambda gm: nn.Module())
        self.assertEqual(_lazy_modules(lowered), [])
