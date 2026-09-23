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
    mark_pipeline_region,
    PipelineAction,
    PipelineConfig,
    PipelinedGraphModule,
    PipelinePlan,
    PipelineSchedule,
    PipelineStageSpec,
)
from recir.pipeline_strategies import (
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


def _two_stage_config() -> PipelineConfig:
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
        )
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

        first = runner(torch.tensor(1), input_ready_event=object())
        second = runner(torch.tensor(2))
        final = runner(last_train_step=True)

        self.assertTrue(is_pipeline_pending(first))
        self.assertEqual(second, torch.tensor(4))
        self.assertEqual(final, torch.tensor(6))

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
