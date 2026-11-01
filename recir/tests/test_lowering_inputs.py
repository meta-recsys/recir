# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for ``lower_inputs`` and the carve reports ``segment_and_lower`` makes to its hooks."""

import dataclasses
import sys
import types
import unittest
from contextlib import contextmanager
from typing import Any, cast, Iterator
from unittest import mock

import torch
import torch.nn as nn
from recir.composite import Part
from recir.lowering.lowering_inputs import lower_inputs, LoweringObserver, WHOLE
from recir.lowering.lowering_utils import CarvedRegion, segment_and_lower
from recir.lowering.torchtpu_lower import _TorchTPUModule
from recir.passes.iter_graph_module import IterGraphModule
from recir.pipeline import build_pipeline, PipelinedGraphModule, SteadyStateGraphModule
from recir.pipeline_strategies import sdd_pipeline_config
from recir.tests.test_utils import _graph, _NoopBackend, _region_graph, _twelve_ops
from torch import fx
from torch.fx.experimental.proxy_tensor import make_fx


def _two_runs(x: torch.Tensor) -> torch.Tensor:
    return _twelve_ops(_twelve_ops(x))


def _pipeline() -> PipelinedGraphModule:
    gm = build_pipeline(_region_graph(), sdd_pipeline_config())
    assert isinstance(gm, PipelinedGraphModule)
    return gm


def _steady_state() -> SteadyStateGraphModule:
    config = dataclasses.replace(sdd_pipeline_config(), materialize_steady_state=True)
    gm = build_pipeline(_region_graph(), config)
    assert isinstance(gm, SteadyStateGraphModule)
    return gm


def _stages(gm: PipelinedGraphModule) -> list[nn.Module]:
    return [gm.split_gm.get_submodule(cast(str, c.target)) for c in gm._stage_calls]


def _iter_graph() -> IterGraphModule:
    gm = mock.MagicMock(spec=IterGraphModule)
    gm.setup_gm, gm.main_gm, gm.cleanup_gm = _graph(), _graph(), _graph()
    gm.lower_parts = lambda lower: IterGraphModule.lower_parts(gm, lower)
    return gm


def _lower(
    gm: fx.GraphModule, *, hooks: Any = None, reject: Any = None
) -> fx.GraphModule:
    return segment_and_lower(
        gm,
        compatibility_fn=lambda n: (n is not reject, "rejected"),
        module_factory=lambda sub: sub,
        hooks=hooks,
    )


def _carve_once(gm: fx.GraphModule, *, hooks: Any = None) -> fx.GraphModule:
    hooks.on_carved(CarvedRegion("subgraph_0", gm, nn.Identity()))
    return gm


class _Recorder(LoweringObserver):
    def __init__(self) -> None:
        self.carves: list[tuple[Part, CarvedRegion]] = []

    def _on_carved(self, part: Part, region: CarvedRegion) -> None:
        self.carves.append((part, region))


class CarveReportTest(unittest.TestCase):
    def test_each_region_is_reported_in_carve_order(self) -> None:
        gm = make_fx(_two_runs)(torch.randn(4))
        middle = [n for n in gm.graph.nodes if n.op == "call_function"][12]
        regions: list[CarvedRegion] = []
        lowered = _lower(gm, hooks=mock.Mock(on_carved=regions.append), reject=middle)
        self.assertEqual([r.name for r in regions], ["subgraph_0", "subgraph_1"])
        for region in regions:
            self.assertIs(region.module, lowered.get_submodule(region.name))


class TraversalTest(unittest.TestCase):
    def lowered(self, gm: Any) -> list[fx.GraphModule]:
        seen: list[fx.GraphModule] = []

        def lower(graph: fx.GraphModule, *, hooks: Any = None) -> fx.GraphModule:
            seen.append(graph)
            return graph

        lower_inputs(gm, lower)
        return seen

    def test_whole_graph(self) -> None:
        gm = _graph()
        self.assertEqual(self.lowered(gm), [gm])

    def test_iter_graph_phases(self) -> None:
        gm = _iter_graph()
        phases = [gm.setup_gm, gm.main_gm, gm.cleanup_gm]
        self.assertEqual(self.lowered(gm), phases)

    def test_pipeline_stages(self) -> None:
        gm = _pipeline()
        self.assertEqual(self.lowered(gm), list(gm.stage_graph_modules()))

    def test_pipeline_stages_are_replaced_by_the_lowered_modules(self) -> None:
        gm = _pipeline()
        lowered: list[nn.Module] = []

        def lower(graph: fx.GraphModule, *, hooks: Any = None) -> nn.Module:
            lowered.append(nn.Identity())
            return lowered[-1]

        lower_inputs(gm, lower)
        self.assertEqual(_stages(gm), lowered)

    def test_steady_state_graph(self) -> None:
        gm = _steady_state()
        steady = gm.steady_gm
        self.assertEqual(self.lowered(gm), [steady])

    def test_without_an_observer_lower_gets_no_hooks(self) -> None:
        lower = mock.Mock(side_effect=lambda gm, hooks: gm)
        gm = _graph()
        self.assertIs(lower_inputs(gm, lower), gm)
        lower.assert_called_once_with(gm, hooks=None)


class ObserverTest(unittest.TestCase):
    def parts(self, gm: Any) -> list[Part]:
        observer = _Recorder()
        lower_inputs(gm, _carve_once, observer)
        return [part for part, _ in observer.carves]

    def test_whole_graph(self) -> None:
        self.assertEqual(self.parts(_graph()), [WHOLE])

    def test_iter_graph_phases(self) -> None:
        self.assertEqual(
            self.parts(_iter_graph()), [Part("setup"), Part("main"), Part("cleanup")]
        )

    def test_pipeline_stages_are_numbered(self) -> None:
        gm = _pipeline()
        self.assertEqual(
            self.parts(gm),
            [Part(name, index) for index, name in enumerate(gm._stage_names)],
        )

    def test_steady_state_graph(self) -> None:
        self.assertEqual(self.parts(_steady_state()), [Part("steady")])

    def test_observer_fault_never_fails_lowering(self) -> None:
        class _Raising(LoweringObserver):
            def _on_carved(self, part: Part, region: CarvedRegion) -> None:
                raise RuntimeError("observer")

        with self.assertLogs("recir.lowering.lowering_inputs", level="WARNING") as logs:
            lowered = lower_inputs(_graph(), _lower, _Raising())
        self.assertIn("subgraph_0", dict(lowered.named_children()))
        self.assertIn("carved failed", logs.output[0])


class _Events(LoweringObserver):
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    def _on_input(self, part: Part, gm: fx.GraphModule) -> None:
        self.events.append(("input", part.name))

    def _on_carved(self, part: Part, region: CarvedRegion) -> None:
        self.events.append(("carved", part.name))


def _raising(gm: fx.GraphModule, *, hooks: Any = None) -> fx.GraphModule:
    raise RuntimeError("lowering")


class LifecycleTest(unittest.TestCase):
    def test_each_input_is_reported_before_its_carves(self) -> None:
        observer = _Events()
        lower_inputs(_graph(), _carve_once, observer)
        self.assertEqual(observer.events, [("input", "whole"), ("carved", "whole")])

    def test_every_phase_is_reported_in_order(self) -> None:
        observer = _Events()
        lower_inputs(_iter_graph(), _carve_once, observer)
        self.assertEqual(
            [e for e in observer.events if e[0] == "input"],
            [("input", "setup"), ("input", "main"), ("input", "cleanup")],
        )

    def test_a_raising_lowering_raises_after_its_input_is_reported(self) -> None:
        observer = _Events()
        with self.assertRaisesRegex(RuntimeError, "lowering"):
            lower_inputs(_graph(), _raising, observer)
        self.assertEqual(observer.events, [("input", "whole")])


class _LoweringBackend(_NoopBackend):
    def dispatch(self, gm: fx.GraphModule) -> nn.Module:
        return self._lower_inputs(gm, _lower)


class BackendLowerInputsTest(unittest.TestCase):
    def test_plain_graph_is_lowered(self) -> None:
        lowered = _LoweringBackend().dispatch(_graph())
        self.assertIn("subgraph_0", dict(lowered.named_children()))

    def test_a_backend_without_composites_gets_the_module_whole(self) -> None:
        lower = mock.Mock(side_effect=lambda gm, hooks: gm)
        pipeline = _pipeline()
        backend = _LoweringBackend()
        self.assertFalse(backend.lowers_composites)
        self.assertIs(backend._lower_inputs(pipeline, lower), pipeline)
        lower.assert_called_once_with(pipeline, hooks=None)


class PartTest(unittest.TestCase):
    def test_labels_are_identifiers_and_unique_within_a_module(self) -> None:
        self.assertEqual(Part("main").label, "main")
        self.assertEqual(
            [Part("route start", 0).label, Part("a", 1).label, Part("a", 2).label],
            ["stage0_route_start", "stage1_a", "stage2_a"],
        )


class TorchTPUCompileScopeTest(unittest.TestCase):
    def test_tpu_backend_compiles_inside_the_compile_scope(self) -> None:
        inside, seen = [False], []

        @contextmanager
        def scope() -> Iterator[None]:
            inside[0] = True
            try:
                yield
            finally:
                inside[0] = False

        class _FakeTpuBackend:
            def __call__(self, gm: fx.GraphModule, args: Any) -> fx.GraphModule:
                seen.append(inside[0])
                return gm

        backend = types.ModuleType("torch_tpu._internal.compile._backend")
        vars(backend)["TpuBackend"] = _FakeTpuBackend
        fake_torch_tpu = {
            name: types.ModuleType(name)
            for name in (
                "torch_tpu",
                "torch_tpu._internal",
                "torch_tpu._internal.compile",
            )
        }
        module = _TorchTPUModule(_graph())
        module.compile_scope = scope
        with (
            mock.patch.dict(sys.modules, {**fake_torch_tpu, backend.__name__: backend}),
            mock.patch.object(_TorchTPUModule, "_to_tpu", side_effect=list),
        ):
            module(torch.randn(4))
            module(torch.randn(4))
        self.assertEqual(seen, [True])
