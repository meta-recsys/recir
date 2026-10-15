# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import operator
import unittest
from types import SimpleNamespace
from typing import Sequence
from unittest import mock

import torch
import torch.fx as fx
from recir.api import make_compiled_runner
from recir.lowering.device_state import DeviceStateManager
from recir.pipeline import (
    build_pipeline,
    is_pipeline_pending,
    mark_execution_stream,
    mark_pipeline_region,
    PipelineAction,
    PipelineConfig,
    PipelinedGraphModule,
    PipelinePlan,
    PipelineSchedule,
    PipelineStageSpec,
    PipelineStreamSpec,
    SteadyStateGraphModule,
)
from recir.pipeline_strategies import (
    embedding_steady_state_pipeline_config,
    PipelineRegion,
    sdd_pipeline_config,
    semi_sync_pipeline_config,
)
from torch.fx.experimental.proxy_tensor import make_fx


class _TwoStageModule(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        distributed = value + 1
        return distributed * 2


class _RunnerModule(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor]:
        distributed = value + 1
        return (distributed * 2,)


class _InputReadyRunnerModule(torch.nn.Module):
    def __init__(self, calls: list[str]) -> None:
        super().__init__()
        self.calls = calls
        self.input_ready_event: object | None = None

    def forward(
        self,
        value: torch.Tensor,
        *,
        input_ready_event: object | None = None,
    ) -> tuple[torch.Tensor]:
        self.calls.append("gm")
        self.input_ready_event = input_ready_event
        return (value,)


class _FanoutModule(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        preprocessed = value + 1
        first = preprocessed * 2
        second = preprocessed * 3
        return first + second


class _InterleavedModule(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        compute_prefix = value * 3
        distributed = value + 1
        return compute_prefix + distributed


class _ContainerBoundaryModule(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        maximum, index = torch.max(value, dim=0)
        distributed = maximum + 1
        return distributed + index


class _FourStageModule(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        preprocessed = value + 1
        started = preprocessed * 2
        continued = started - 3
        return continued / 2


class _SemiSyncModule(torch.nn.Module):
    def forward(
        self,
        state: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        preprocessed = value + 1
        routed = preprocessed - 1
        embedding = state * routed
        updated = state.add_(1).clone()
        return embedding, updated


class _FunctionalStateModule(torch.nn.Module):
    def forward(
        self,
        state: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        preprocessed = value + 1
        routed = preprocessed - 1
        embedding = state * routed
        return embedding, state + 1


def _structured_output(
    state: torch.Tensor, value: torch.Tensor
) -> tuple[tuple[torch.Tensor], torch.Tensor]:
    distributed = value + 1
    return ((distributed * 2,), state + 1)


def _mark_add(gm: fx.GraphModule) -> fx.GraphModule:
    add_node = next(
        node
        for node in gm.graph.nodes
        if node.op == "call_function" and node.target is operator.add
    )
    mark_pipeline_region((add_node,), PipelineRegion.ROUTE_START)
    return gm


def _two_stage_graph() -> fx.GraphModule:
    return _mark_add(fx.symbolic_trace(_TwoStageModule()))


def _conditional_graph() -> fx.GraphModule:
    def true_branch(value: torch.Tensor) -> torch.Tensor:
        return value + 1

    def false_branch(value: torch.Tensor) -> torch.Tensor:
        return value - 1

    def forward(predicate: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        routed = value * 2
        selected = torch.cond(predicate, true_branch, false_branch, (routed,))
        return selected * 3

    gm = make_fx(forward)(torch.tensor(True), torch.tensor(1))
    cond = next(
        node
        for node in gm.graph.nodes
        if node.op == "call_function" and node.target is torch.ops.higher_order.cond
    )
    mark_pipeline_region((cond,), PipelineRegion.ROUTE_START)
    return gm


def _two_stage_config(*, materialize_steady_state: bool = False) -> PipelineConfig:
    return PipelineConfig.from_plan(
        PipelinePlan(
            stages=(
                PipelineStageSpec(
                    name="route_start",
                    regions=(PipelineRegion.ROUTE_START,),
                    required_regions=(PipelineRegion.ROUTE_START,),
                ),
                PipelineStageSpec(
                    name="compute",
                    include_unassigned_nodes=True,
                ),
            ),
            schedule=PipelineSchedule(),
        ),
        materialize_steady_state=materialize_steady_state,
    )


def _semi_sync_graph() -> fx.GraphModule:
    gm = fx.symbolic_trace(_SemiSyncModule())
    state = next(node for node in gm.graph.nodes if node.name == "state")
    state.meta["is_weight"] = True
    add = next(
        node
        for node in gm.graph.nodes
        if node.op == "call_function" and node.target is operator.add
    )
    sub = next(
        node
        for node in gm.graph.nodes
        if node.op == "call_function" and node.target is operator.sub
    )
    lookup = next(
        node
        for node in gm.graph.nodes
        if node.op == "call_function" and node.target is operator.mul
    )
    mark_pipeline_region((add,), PipelineRegion.ROUTE_START)
    mark_pipeline_region((sub,), PipelineRegion.ROUTE_CONTINUE)
    mark_pipeline_region((lookup,), PipelineRegion.EMBEDDING_LOOKUP)
    return gm


def _functional_state_graph() -> fx.GraphModule:
    gm = fx.symbolic_trace(_FunctionalStateModule())
    state = next(node for node in gm.graph.nodes if node.name == "state")
    state.meta["is_state_input"] = True
    state.meta["is_weight"] = True
    value = next(node for node in gm.graph.nodes if node.name == "value")
    value.meta["is_state_input"] = False
    value.meta["is_weight"] = True
    add, _ = (
        node
        for node in gm.graph.nodes
        if node.op == "call_function" and node.target is operator.add
    )
    sub = next(
        node
        for node in gm.graph.nodes
        if node.op == "call_function" and node.target is operator.sub
    )
    lookup = next(
        node
        for node in gm.graph.nodes
        if node.op == "call_function" and node.target is operator.mul
    )
    mark_pipeline_region((add,), PipelineRegion.ROUTE_START)
    mark_pipeline_region((sub,), PipelineRegion.ROUTE_CONTINUE)
    mark_pipeline_region((lookup,), PipelineRegion.EMBEDDING_LOOKUP)
    return gm


def _region_graph() -> fx.GraphModule:
    gm = fx.symbolic_trace(_FourStageModule())
    regions = (
        (operator.add, PipelineRegion.PREPROCESS),
        (operator.mul, PipelineRegion.ROUTE_START),
        (operator.sub, PipelineRegion.ROUTE_CONTINUE),
    )
    for target, region in regions:
        node = next(
            node
            for node in gm.graph.nodes
            if node.op == "call_function" and node.target is target
        )
        mark_pipeline_region((node,), region)
    return gm


def _four_stage_plan(schedule: PipelineSchedule) -> PipelinePlan:
    return PipelinePlan(
        stages=(
            PipelineStageSpec(
                name="preprocess",
                regions=(PipelineRegion.PREPROCESS,),
            ),
            PipelineStageSpec(
                name="route_start",
                regions=(PipelineRegion.ROUTE_START,),
                required_regions=(PipelineRegion.ROUTE_START,),
            ),
            PipelineStageSpec(
                name="route_continue",
                regions=(PipelineRegion.ROUTE_CONTINUE,),
            ),
            PipelineStageSpec(
                name="compute",
                include_unassigned_nodes=True,
            ),
        ),
        schedule=schedule,
    )


class _AscendingPipelineSchedule(PipelineSchedule):
    def order_actions(
        self,
        ready_actions: Sequence[PipelineAction],
    ) -> Sequence[PipelineAction]:
        return tuple(sorted(ready_actions, key=lambda action: action.stage_index))


class PipelineTest(unittest.TestCase):
    def test_pipeline_preserves_cond_branches(self) -> None:
        gm = _conditional_graph()
        partition_meta = "recir_pipeline_partition"
        branch_get_attrs = [node for node in gm.graph.nodes if node.op == "get_attr"]
        self.assertTrue(branch_get_attrs)
        for node in branch_get_attrs:
            node.meta[partition_meta] = 99
        pipeline = build_pipeline(gm, _two_stage_config())
        self.assertIsInstance(pipeline, PipelinedGraphModule)
        for node in branch_get_attrs:
            self.assertNotIn(partition_meta, node.meta)
        self.assertFalse(
            any(
                node.op == "get_attr" and not node.users
                for node in pipeline.graph.nodes
            )
        )
        for stage in pipeline.stage_graph_modules():
            for node in stage.graph.nodes:
                if node.op == "get_attr":
                    self.assertNotIn(partition_meta, node.meta)

        self.assertTrue(
            is_pipeline_pending(pipeline(torch.tensor(True), torch.tensor(2)))
        )
        self.assertEqual(
            pipeline(torch.tensor(False), torch.tensor(2)),
            torch.tensor(15),
        )
        self.assertEqual(pipeline(last_iter=True), torch.tensor(9))

    def test_mark_execution_stream_preserves_metadata(self) -> None:
        gm = _two_stage_graph()
        node = next(node for node in gm.graph.nodes if node.op == "call_function")
        node.meta["custom"] = {"existing": "metadata"}

        mark_execution_stream((node,), 7)
        mark_execution_stream((node,), 7)

        self.assertEqual(
            node.meta["custom"],
            {"existing": "metadata", "stream": 7},
        )
        with self.assertRaisesRegex(ValueError, "conflicting stream indices"):
            mark_execution_stream((node,), 8)

    def test_plan_rejects_conflicting_stream_assignments(self) -> None:
        base = _four_stage_plan(PipelineSchedule())
        with self.assertRaisesRegex(ValueError, "only one stream"):
            PipelinePlan(
                stages=base.stages,
                schedule=base.schedule,
                side_streams=(
                    PipelineStreamSpec(
                        name="first", regions=(PipelineRegion.PREPROCESS,)
                    ),
                    PipelineStreamSpec(
                        name="second", regions=(PipelineRegion.PREPROCESS,)
                    ),
                ),
            )

    def test_embedding_steady_state_places_preprocess_on_side_stream(self) -> None:
        config = embedding_steady_state_pipeline_config()

        plan = config.plan
        self.assertIsNotNone(plan)
        if plan is None:
            raise AssertionError("Expected an enabled pipeline plan")
        self.assertEqual(len(plan.stages), 2)
        self.assertEqual(
            plan.stages[0].regions,
            (
                PipelineRegion.PREPROCESS,
                PipelineRegion.ROUTE_START,
                PipelineRegion.ROUTE_CONTINUE,
            ),
        )
        self.assertTrue(plan.stages[1].include_unassigned_nodes)
        self.assertEqual(
            plan.side_streams,
            (
                PipelineStreamSpec(
                    name="embedding_input",
                    regions=(
                        PipelineRegion.PREPROCESS,
                        PipelineRegion.ROUTE_START,
                    ),
                ),
            ),
        )

    def test_steady_state_lowering_materializes_stream_and_events(self) -> None:
        plan = PipelinePlan(
            stages=(
                PipelineStageSpec(
                    name="embedding",
                    regions=(
                        PipelineRegion.PREPROCESS,
                        PipelineRegion.ROUTE_START,
                        PipelineRegion.ROUTE_CONTINUE,
                    ),
                    required_regions=(
                        PipelineRegion.ROUTE_START,
                        PipelineRegion.ROUTE_CONTINUE,
                    ),
                ),
                PipelineStageSpec(name="compute", include_unassigned_nodes=True),
            ),
            schedule=PipelineSchedule(),
            side_streams=(
                PipelineStreamSpec(
                    name="preprocess",
                    regions=(PipelineRegion.PREPROCESS,),
                ),
            ),
        )
        pipeline = build_pipeline(
            _region_graph(),
            PipelineConfig.from_plan(plan, materialize_steady_state=True),
        )
        self.assertIsInstance(pipeline, SteadyStateGraphModule)
        lowered: list[fx.GraphModule] = []
        stream = mock.MagicMock(spec=torch.Stream)
        current_stream = SimpleNamespace(device=torch.device("cuda"))

        def get_external_object(index: int) -> object:
            return stream if index == 7 else object()

        with (
            mock.patch.object(torch.accelerator, "is_available", return_value=True),
            mock.patch.object(
                torch.accelerator,
                "current_stream",
                return_value=current_stream,
            ),
            mock.patch(
                "torch._dynamo.variables.streams.get_current_stream",
                return_value=5,
            ),
            mock.patch("torch._dynamo.variables.streams.new_stream", return_value=7),
            mock.patch("torch._dynamo.variables.streams.new_event", return_value=8),
            mock.patch(
                "torch._dynamo.graph_bytecode_inputs.get_external_object_by_index",
                side_effect=get_external_object,
            ),
        ):
            pipeline.lower_steady(lambda gm: lowered.append(gm) or gm)

        self.assertEqual(len(lowered), 1)
        nodes = list(lowered[0].graph.nodes)
        preprocessed = next(node for node in nodes if node.target is operator.add)
        routed = next(node for node in nodes if node.target is operator.mul)
        self.assertEqual(preprocessed.meta["custom"]["stream"], 7)
        self.assertNotIn("stream", routed.meta.get("custom", {}))
        targets = [node.target for node in nodes]
        self.assertIn(torch.ops.streams.record_event.default, targets)
        self.assertIn(torch.ops.streams.wait_event.default, targets)

    def test_steady_state_lowering_materializes_multiple_streams(self) -> None:
        plan = PipelinePlan(
            stages=(
                PipelineStageSpec(
                    name="embedding",
                    regions=(
                        PipelineRegion.PREPROCESS,
                        PipelineRegion.ROUTE_START,
                        PipelineRegion.ROUTE_CONTINUE,
                    ),
                    required_regions=(
                        PipelineRegion.ROUTE_START,
                        PipelineRegion.ROUTE_CONTINUE,
                    ),
                ),
                PipelineStageSpec(name="compute", include_unassigned_nodes=True),
            ),
            schedule=PipelineSchedule(),
            side_streams=(
                PipelineStreamSpec(
                    name="preprocess",
                    regions=(PipelineRegion.PREPROCESS,),
                    priority=-1,
                ),
                PipelineStreamSpec(
                    name="route",
                    regions=(PipelineRegion.ROUTE_START,),
                    priority=0,
                ),
            ),
        )
        pipeline = build_pipeline(
            _region_graph(),
            PipelineConfig.from_plan(plan, materialize_steady_state=True),
        )
        self.assertIsInstance(pipeline, SteadyStateGraphModule)
        lowered: list[fx.GraphModule] = []
        preprocess_stream = mock.MagicMock(spec=torch.Stream)
        route_stream = mock.MagicMock(spec=torch.Stream)
        current_stream = SimpleNamespace(device=torch.device("cuda"))

        def get_external_object(index: int) -> object:
            if index == 7:
                return preprocess_stream
            if index == 9:
                return route_stream
            return object()

        with (
            mock.patch.object(torch.accelerator, "is_available", return_value=True),
            mock.patch.object(
                torch.accelerator,
                "current_stream",
                return_value=current_stream,
            ),
            mock.patch(
                "torch._dynamo.variables.streams.get_current_stream",
                return_value=5,
            ),
            mock.patch(
                "torch._dynamo.variables.streams.new_stream",
                side_effect=(7, 9),
            ) as new_stream,
            mock.patch(
                "torch._dynamo.variables.streams.new_event",
                side_effect=(8, 10),
            ),
            mock.patch(
                "torch._dynamo.graph_bytecode_inputs.get_external_object_by_index",
                side_effect=get_external_object,
            ),
        ):
            pipeline.lower_steady(lambda gm: lowered.append(gm) or gm)

        nodes = list(lowered[0].graph.nodes)
        preprocessed = next(node for node in nodes if node.target is operator.add)
        routed = next(node for node in nodes if node.target is operator.mul)
        continued = next(node for node in nodes if node.target is operator.sub)
        self.assertEqual(preprocessed.meta["custom"]["stream"], 7)
        self.assertEqual(routed.meta["custom"]["stream"], 9)
        self.assertNotIn("stream", continued.meta.get("custom", {}))
        new_stream.assert_has_calls([mock.call(priority=-1), mock.call(priority=0)])
        targets = [node.target for node in nodes]
        self.assertEqual(targets.count(torch.ops.streams.record_event.default), 2)
        self.assertEqual(targets.count(torch.ops.streams.wait_event.default), 2)

    def test_cross_stream_fanout_reuses_producer_event(self) -> None:
        gm = fx.symbolic_trace(_FanoutModule())
        producer = next(
            node
            for node in gm.graph.nodes
            if node.op == "call_function" and node.target is operator.add
        )
        mark_pipeline_region((producer,), PipelineRegion.PREPROCESS)
        pipeline = SteadyStateGraphModule(
            gm,
            lambda output: output,
            node_stages={
                node: 0 for node in gm.graph.nodes if node.op == "call_function"
            },
            stage_names=("compute",),
            schedule=PipelineSchedule(),
            side_streams=(
                PipelineStreamSpec(
                    name="preprocess",
                    regions=(PipelineRegion.PREPROCESS,),
                ),
            ),
        )
        lowered: list[fx.GraphModule] = []
        stream = mock.MagicMock(spec=torch.Stream)
        current_stream = SimpleNamespace(device=torch.device("cuda"))

        with (
            mock.patch.object(torch.accelerator, "is_available", return_value=True),
            mock.patch.object(
                torch.accelerator, "current_stream", return_value=current_stream
            ),
            mock.patch(
                "torch._dynamo.variables.streams.get_current_stream", return_value=5
            ),
            mock.patch("torch._dynamo.variables.streams.new_stream", return_value=7),
            mock.patch("torch._dynamo.variables.streams.new_event", return_value=8),
            mock.patch(
                "torch._dynamo.graph_bytecode_inputs.get_external_object_by_index",
                return_value=stream,
            ),
        ):
            pipeline.lower_steady(
                lambda lowered_gm: lowered.append(lowered_gm) or lowered_gm
            )

        targets = [node.target for node in lowered[0].graph.nodes]
        self.assertEqual(targets.count(torch.ops.streams.record_event.default), 1)
        self.assertEqual(targets.count(torch.ops.streams.wait_event.default), 1)

    def test_two_stage_pipeline_launches_compute_before_next_prefix(self) -> None:
        pipeline = build_pipeline(_two_stage_graph(), _two_stage_config())
        self.assertIsInstance(pipeline, PipelinedGraphModule)
        self.assertTrue(is_pipeline_pending(pipeline(torch.tensor(1))))

        with mock.patch.object(
            pipeline,
            "_run_stage",
            wraps=pipeline._run_stage,
        ) as run_stage:
            self.assertEqual(pipeline(torch.tensor(2)), torch.tensor(4))

        self.assertEqual(
            [call.args[0] for call in run_stage.call_args_list],
            [1, 0],
        )

    def test_steady_state_pipeline_keeps_stages_in_one_graph(self) -> None:
        pipeline = build_pipeline(
            _two_stage_graph(),
            _two_stage_config(materialize_steady_state=True),
        )
        self.assertIsInstance(pipeline, SteadyStateGraphModule)
        steady = pipeline.steady_gm
        self.assertIsInstance(steady, fx.GraphModule)
        targets = [
            node.target for node in steady.graph.nodes if node.op == "call_function"
        ]
        self.assertIn(operator.add, targets)
        self.assertIn(operator.mul, targets)
        self.assertFalse(any(node.op == "call_module" for node in steady.graph.nodes))

        self.assertTrue(is_pipeline_pending(pipeline(torch.tensor(1))))
        self.assertEqual(pipeline(torch.tensor(2)), torch.tensor(4))
        self.assertEqual(pipeline(torch.tensor(3)), torch.tensor(6))
        self.assertEqual(pipeline(last_iter=True), torch.tensor(8))

    def test_steady_state_records_inputs_on_default_and_joins_streams(self) -> None:
        pipeline = build_pipeline(
            _two_stage_graph(),
            _two_stage_config(materialize_steady_state=True),
        )
        self.assertIsInstance(pipeline, SteadyStateGraphModule)
        self.assertTrue(is_pipeline_pending(pipeline(torch.tensor(1))))
        main_stream = mock.Mock()
        first_stream = mock.Mock()
        second_stream = mock.Mock()
        first_event = mock.Mock()
        second_event = mock.Mock()
        completion_event = mock.Mock()
        pipeline._side_streams = {
            "first": first_stream,
            "second": second_stream,
        }

        with (
            mock.patch.object(
                pipeline, "_wait_for_steady_inputs", return_value=main_stream
            ),
            mock.patch.object(pipeline, "_register_execution_streams"),
            mock.patch.object(pipeline, "_record_values_on_stream") as record_values,
            mock.patch.object(
                torch.cuda,
                "Event",
                side_effect=(first_event, second_event, completion_event),
            ),
        ):
            self.assertEqual(pipeline(torch.tensor(2)), torch.tensor(4))

        record_values.assert_called_once_with(mock.ANY, main_stream)
        first_event.record.assert_called_once_with(first_stream)
        second_event.record.assert_called_once_with(second_stream)
        main_stream.wait_event.assert_has_calls(
            [mock.call(first_event), mock.call(second_event)]
        )
        completion_event.record.assert_called_once_with(main_stream)

    def test_steady_state_waits_all_streams_for_ready_inputs(self) -> None:
        pipeline = build_pipeline(
            _two_stage_graph(),
            _two_stage_config(materialize_steady_state=True),
        )
        self.assertIsInstance(pipeline, SteadyStateGraphModule)
        self.assertTrue(is_pipeline_pending(pipeline(torch.tensor(1))))
        main_stream = mock.Mock()
        first_stream = mock.Mock()
        second_stream = mock.Mock()
        input_event = mock.Mock()
        slot_event = mock.Mock()
        pipeline._side_streams = {
            "first": first_stream,
            "second": second_stream,
        }
        slot = pipeline._slots[0]
        self.assertIsNotNone(slot)
        if slot is None:
            raise AssertionError("Expected a filled pipeline slot")
        slot.event = slot_event

        with (
            mock.patch.object(
                pipeline,
                "_first_cuda_device",
                return_value=torch.device("cuda"),
            ),
            mock.patch.object(torch.cuda, "current_stream", return_value=main_stream),
        ):
            stream = pipeline._wait_for_steady_inputs((torch.tensor(2),), input_event)

        self.assertIs(stream, main_stream)
        main_stream.wait_event.assert_called_once_with(slot_event)
        for execution_stream in (first_stream, second_stream):
            execution_stream.wait_event.assert_has_calls(
                [mock.call(input_event), mock.call(slot_event)]
            )

    def test_steady_state_skips_redundant_default_stream_wait(self) -> None:
        pipeline = build_pipeline(
            _two_stage_graph(),
            _two_stage_config(materialize_steady_state=True),
        )
        self.assertIsInstance(pipeline, SteadyStateGraphModule)
        self.assertTrue(is_pipeline_pending(pipeline(torch.tensor(1))))
        main_stream = mock.Mock()
        side_stream = mock.Mock()
        completion_event = mock.Mock()
        pipeline._side_streams = {"side": side_stream}
        slot = pipeline._slots[0]
        self.assertIsNotNone(slot)
        if slot is None:
            raise AssertionError("Expected a filled pipeline slot")
        slot.event = completion_event
        pipeline._slots = [slot, slot]
        pipeline._last_completion_event = completion_event
        pipeline._last_completion_stream = main_stream

        with (
            mock.patch.object(
                pipeline, "_first_cuda_device", return_value=torch.device("cuda")
            ),
            mock.patch.object(torch.cuda, "current_stream", return_value=main_stream),
        ):
            pipeline._wait_for_steady_inputs((torch.tensor(2),), None)

        main_stream.wait_event.assert_not_called()
        side_stream.wait_event.assert_called_once_with(completion_event)

    def test_steady_state_fill_and_drain_use_default_stream(self) -> None:
        pipeline = build_pipeline(
            _two_stage_graph(),
            _two_stage_config(materialize_steady_state=True),
        )
        self.assertIsInstance(pipeline, SteadyStateGraphModule)
        side_stream = mock.Mock()
        main_stream = mock.Mock()
        pipeline._side_streams = {"side": side_stream}

        with (
            mock.patch.object(
                pipeline,
                "_first_cuda_device",
                return_value=torch.device("cuda"),
            ),
            mock.patch.object(torch.cuda, "current_stream", return_value=main_stream),
        ):
            stream = pipeline._stream_for_stage(0, {})

        self.assertIs(stream, main_stream)

    def test_steady_state_fill_keeps_only_live_state(self) -> None:
        pipeline = build_pipeline(
            _two_stage_graph(),
            _two_stage_config(materialize_steady_state=True),
        )
        self.assertIsInstance(pipeline, SteadyStateGraphModule)

        self.assertTrue(is_pipeline_pending(pipeline(torch.tensor(1))))

        slot = pipeline._slots[0]
        self.assertIsNotNone(slot)
        self.assertEqual(set(slot.env), set(pipeline._live_after[0]))

    def test_steady_state_pipeline_lowers_one_graph(self) -> None:
        pipeline = build_pipeline(
            _two_stage_graph(),
            _two_stage_config(materialize_steady_state=True),
        )
        self.assertIsInstance(pipeline, SteadyStateGraphModule)
        lowered: list[fx.GraphModule] = []

        def lower(gm: fx.GraphModule) -> torch.nn.Module:
            lowered.append(gm)
            return gm

        pipeline.lower_steady(lower)
        self.assertEqual(lowered, [pipeline.steady_gm])

    def test_steady_state_pipeline_reads_updated_state_for_next_batch(self) -> None:
        state = torch.tensor(2.0)
        pipeline = build_pipeline(
            _functional_state_graph(),
            embedding_steady_state_pipeline_config(),
        )

        self.assertTrue(is_pipeline_pending(pipeline(state, torch.tensor(3.0))))
        first = pipeline(state, torch.tensor(4.0))
        self.assertEqual(first, (torch.tensor(6.0), torch.tensor(3.0)))
        second = pipeline(first[1], torch.tensor(5.0))
        self.assertEqual(second, (torch.tensor(12.0), torch.tensor(4.0)))

    def test_four_stage_pipeline_advances_each_stage_independently(self) -> None:
        pipeline = build_pipeline(
            _region_graph(),
            PipelineConfig.from_plan(_four_stage_plan(PipelineSchedule())),
        )
        self.assertIsInstance(pipeline, PipelinedGraphModule)
        self.assertEqual(len(pipeline.stage_graph_modules()), 4)

        self.assertTrue(is_pipeline_pending(pipeline(torch.tensor(1))))
        self.assertTrue(is_pipeline_pending(pipeline(torch.tensor(2))))
        self.assertTrue(is_pipeline_pending(pipeline(torch.tensor(3))))
        with mock.patch.object(
            pipeline,
            "_run_stage",
            wraps=pipeline._run_stage,
        ) as run_stage:
            self.assertEqual(pipeline(torch.tensor(4)), torch.tensor(0.5))

        self.assertEqual(
            [call.args[0] for call in run_stage.call_args_list],
            [3, 2, 1, 0],
        )
        self.assertEqual(pipeline(last_iter=True), torch.tensor(1.5))
        self.assertEqual(pipeline(last_iter=True), torch.tensor(2.5))
        self.assertEqual(pipeline(last_iter=True), torch.tensor(3.5))

    def test_custom_schedule_controls_ready_stage_order(self) -> None:
        pipeline = build_pipeline(
            _region_graph(),
            PipelineConfig.from_plan(_four_stage_plan(_AscendingPipelineSchedule())),
        )
        self.assertTrue(is_pipeline_pending(pipeline(torch.tensor(1))))
        self.assertTrue(is_pipeline_pending(pipeline(torch.tensor(2))))
        self.assertTrue(is_pipeline_pending(pipeline(torch.tensor(3))))

        with mock.patch.object(
            pipeline,
            "_run_stage",
            wraps=pipeline._run_stage,
        ) as run_stage:
            self.assertEqual(pipeline(torch.tensor(4)), torch.tensor(0.5))

        self.assertEqual(
            [call.args[0] for call in run_stage.call_args_list],
            [1, 2, 3, 0],
        )

    def test_sdd_plan_merges_preprocess_and_route_start(self) -> None:
        pipeline = build_pipeline(_region_graph(), sdd_pipeline_config())
        self.assertIsInstance(pipeline, PipelinedGraphModule)
        stages = pipeline.stage_graph_modules()

        self.assertEqual(len(stages), 3)
        first_stage_targets = {
            node.target for node in stages[0].graph.nodes if node.op == "call_function"
        }
        self.assertIn(operator.add, first_stage_targets)
        self.assertIn(operator.mul, first_stage_targets)

    def test_custom_plan_can_split_preprocess_from_route_start(self) -> None:
        pipeline = build_pipeline(
            _region_graph(),
            PipelineConfig.from_plan(_four_stage_plan(PipelineSchedule())),
        )
        self.assertIsInstance(pipeline, PipelinedGraphModule)
        self.assertEqual(len(pipeline.stage_graph_modules()), 4)

    def test_plan_requires_exactly_one_unassigned_nodes_stage(self) -> None:
        region_stage = PipelineStageSpec(
            name="route_start",
            regions=(PipelineRegion.ROUTE_START,),
        )
        remainder_stage = PipelineStageSpec(
            name="compute",
            include_unassigned_nodes=True,
        )
        for stages in (
            (region_stage,),
            (
                region_stage,
                remainder_stage,
                PipelineStageSpec(name="output", include_unassigned_nodes=True),
            ),
        ):
            with self.subTest(stage_count=len(stages)):
                with self.assertRaisesRegex(ValueError, "exactly one"):
                    PipelinePlan(stages=stages, schedule=PipelineSchedule())

    def test_plan_requires_unassigned_nodes_stage_last(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be last"):
            PipelinePlan(
                stages=(
                    PipelineStageSpec(
                        name="compute",
                        include_unassigned_nodes=True,
                    ),
                    PipelineStageSpec(
                        name="route_start",
                        regions=(PipelineRegion.ROUTE_START,),
                    ),
                ),
                schedule=PipelineSchedule(),
            )

    def test_stage_spec_separates_regions_from_unassigned_nodes(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot declare pipeline regions"):
            PipelineStageSpec(
                name="compute",
                regions=(PipelineRegion.ROUTE_START,),
                include_unassigned_nodes=True,
            )
        with self.assertRaisesRegex(ValueError, "must declare regions"):
            PipelineStageSpec(name="empty")

    def test_pipeline_preserves_pytree_output_adapter(self) -> None:
        gm = make_fx(_structured_output)(torch.tensor(10), torch.tensor(1))
        distributed = next(
            node
            for node in gm.graph.nodes
            if node.op == "call_function" and node.target is torch.ops.aten.add.Tensor
        )
        mark_pipeline_region((distributed,), PipelineRegion.ROUTE_START)
        pipeline = build_pipeline(gm, _two_stage_config())

        self.assertTrue(
            is_pipeline_pending(pipeline(torch.tensor(10), torch.tensor(1)))
        )
        self.assertEqual(
            pipeline(torch.tensor(10), torch.tensor(2)),
            ((torch.tensor(4),), torch.tensor(11)),
        )
        self.assertEqual(
            pipeline(last_iter=True),
            ((torch.tensor(6),), torch.tensor(11)),
        )

    def test_container_outputs_are_unpacked_before_stage_boundary(self) -> None:
        gm = fx.symbolic_trace(_ContainerBoundaryModule())
        distributed = next(
            node
            for node in gm.graph.nodes
            if node.op == "call_function" and node.target is operator.add
        )
        mark_pipeline_region((distributed,), PipelineRegion.ROUTE_START)
        pipeline = build_pipeline(gm, _two_stage_config())
        self.assertIsInstance(pipeline, PipelinedGraphModule)

        first_stage = pipeline.stage_graph_modules()[0]
        maximum = next(
            node
            for node in first_stage.graph.nodes
            if node.op == "call_function" and node.target is torch.max
        )
        self.assertTrue(
            all(
                user.op == "call_function" and user.target is operator.getitem
                for user in maximum.users
            )
        )
        self.assertTrue(is_pipeline_pending(pipeline(torch.tensor([1, 3]))))
        self.assertEqual(pipeline(torch.tensor([2, 5])), torch.tensor(5))
        self.assertEqual(pipeline(last_iter=True), torch.tensor(7))

    def test_split_orders_stages_by_dependency(self) -> None:
        gm = fx.symbolic_trace(_InterleavedModule())
        distributed = next(
            node
            for node in gm.graph.nodes
            if node.op == "call_function" and node.target is operator.add
        )
        mark_pipeline_region((distributed,), PipelineRegion.ROUTE_START)
        pipeline = build_pipeline(gm, _two_stage_config())

        self.assertTrue(is_pipeline_pending(pipeline(torch.tensor(1))))
        self.assertEqual(pipeline(torch.tensor(2)), torch.tensor(5))
        self.assertEqual(pipeline(last_iter=True), torch.tensor(9))

    def test_disabled_pipeline_preserves_graph(self) -> None:
        gm = _two_stage_graph()
        self.assertIs(build_pipeline(gm, PipelineConfig.none()), gm)

    def test_pipeline_primes_runs_and_drains(self) -> None:
        pipeline = build_pipeline(_two_stage_graph(), _two_stage_config())

        first = pipeline(torch.tensor(1))
        second = pipeline(torch.tensor(2))
        final = pipeline(last_iter=True)

        self.assertTrue(is_pipeline_pending(first))
        self.assertEqual(second, torch.tensor(4))
        self.assertEqual(final, torch.tensor(6))

    def test_compiled_runner_handles_pipeline_pending_and_drain(self) -> None:
        pipeline = build_pipeline(
            _mark_add(fx.symbolic_trace(_RunnerModule())),
            _two_stage_config(),
        )
        compiled_obj = SimpleNamespace(
            gm=pipeline,
            flat_state=[],
            state_manager=DeviceStateManager(),
        )
        runner = make_compiled_runner(compiled_obj)

        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            first = runner(torch.tensor(1), input_ready_event=object())
        second = runner(torch.tensor(2))
        final = runner(last_train_step=True)

        self.assertTrue(is_pipeline_pending(first))
        self.assertEqual(second, torch.tensor(4))
        self.assertEqual(final, torch.tensor(6))

    def test_compiled_runner_forwards_input_ready_event(self) -> None:
        calls: list[str] = []
        input_ready_event = object()
        state_manager = mock.Mock()
        state_manager.prepare_inputs.side_effect = (
            lambda gm, flat_state, user_inps, last_train_step: (
                calls.append("prepare") or user_inps
            )
        )
        state_manager.defers_output.return_value = False
        state_manager.materialize_output.side_effect = lambda output: output
        gm = _InputReadyRunnerModule(calls)
        compiled_obj = SimpleNamespace(
            gm=gm,
            flat_state=[],
            state_manager=state_manager,
        )
        runner = make_compiled_runner(compiled_obj)

        output = runner(torch.tensor(1), input_ready_event=input_ready_event)

        self.assertEqual(calls, ["prepare", "gm"])
        self.assertEqual(output, torch.tensor(1))
        self.assertIs(gm.input_ready_event, input_ready_event)

    def test_sdd_pipeline_rejects_state_read_in_prefix(self) -> None:
        gm = _region_graph()
        placeholder = next(node for node in gm.graph.nodes if node.op == "placeholder")
        placeholder.meta["is_weight"] = True

        with self.assertRaisesRegex(ValueError, "cannot read model or optimizer state"):
            build_pipeline(gm, sdd_pipeline_config())

    def test_semi_sync_uses_bounded_stale_embedding_state(self) -> None:
        pipeline = build_pipeline(
            _semi_sync_graph(),
            semi_sync_pipeline_config(start_batch=0),
        )
        self.assertIsInstance(pipeline, PipelinedGraphModule)
        self.assertEqual(len(pipeline.stage_graph_modules()), 4)
        state = torch.tensor(10)

        self.assertTrue(is_pipeline_pending(pipeline(state, torch.tensor(1))))
        self.assertTrue(is_pipeline_pending(pipeline(state, torch.tensor(2))))
        self.assertTrue(is_pipeline_pending(pipeline(state, torch.tensor(3))))
        self.assertEqual(
            pipeline(last_iter=True),
            (torch.tensor(10), torch.tensor(11)),
        )
        self.assertEqual(
            pipeline(last_iter=True),
            (torch.tensor(20), torch.tensor(12)),
        )
        self.assertEqual(
            pipeline(last_iter=True),
            (torch.tensor(33), torch.tensor(13)),
        )

    def test_semi_sync_warmup_keeps_updates_synchronous(self) -> None:
        pipeline = build_pipeline(
            _semi_sync_graph(),
            semi_sync_pipeline_config(start_batch=2),
        )
        state = torch.tensor(10)

        outputs = [pipeline(state, torch.tensor(value)) for value in range(1, 5)]
        outputs.extend(pipeline(last_iter=True) for _ in range(3))
        completed = [output for output in outputs if not is_pipeline_pending(output)]

        self.assertEqual(
            completed,
            [
                (torch.tensor(10), torch.tensor(11)),
                (torch.tensor(22), torch.tensor(12)),
                (torch.tensor(36), torch.tensor(13)),
                (torch.tensor(48), torch.tensor(14)),
            ],
        )

    def test_semi_sync_requires_embedding_lookup_marker(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "requires marked regions: embedding_lookup"
        ):
            build_pipeline(
                _region_graph(),
                semi_sync_pipeline_config(start_batch=0),
            )

    def test_gradient_accumulation_is_rejected_until_scheduled(self) -> None:
        with self.assertRaisesRegex(
            NotImplementedError,
            "Gradient accumulation scheduling is not implemented yet",
        ):
            build_pipeline(
                _two_stage_graph(),
                sdd_pipeline_config(gradient_accumulation_steps=2),
            )
