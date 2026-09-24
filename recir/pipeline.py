# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
import operator
from collections.abc import Hashable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, cast, Iterable, Optional, Sequence

import torch
import torch.fx as fx
import torch.nn as nn
from torch.fx.graph import map_arg
from torch.fx.passes.split_module import split_module
from torch.utils import _pytree as pytree


logger: logging.Logger = logging.getLogger(__name__)
_PIPELINE_REGION_META = "recir_pipeline_region"
_PIPELINE_PARTITION_META = "recir_pipeline_partition"


def _is_pipeline_partition_node(node: fx.Node) -> bool:
    return node.op not in ("placeholder", "get_attr", "output")


@dataclass(frozen=True)
class PipelineAction:
    stage_index: int
    batch_index: int
    stage_name: str


class PipelineSchedule:
    """Order ready stage actions and declare same-tick dependencies.

    Slot movement and stage execution stay in :class:`PipelinedGraphModule`.
    Subclasses only arrange independent stage actions, so new linear pipelines
    do not require branches in the executor.
    """

    def order_actions(
        self,
        ready_actions: Sequence[PipelineAction],
    ) -> Sequence[PipelineAction]:
        return tuple(
            sorted(
                ready_actions,
                key=lambda action: action.stage_index,
                reverse=True,
            )
        )

    def dependencies(
        self,
        action: PipelineAction,
        ready_actions: Sequence[PipelineAction],
    ) -> Sequence[PipelineAction]:
        del action, ready_actions
        return ()


@dataclass(frozen=True)
class PipelineStageSpec:
    """One named stage containing regions or all otherwise unassigned nodes."""

    name: str
    regions: tuple[Hashable, ...] = ()
    required_regions: tuple[Hashable, ...] = ()
    include_unassigned_nodes: bool = False
    allow_state_reads: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Pipeline stage name must not be empty")
        if self.include_unassigned_nodes:
            if self.regions or self.required_regions:
                raise ValueError(
                    "The unassigned-nodes stage cannot declare pipeline regions"
                )
        elif not self.regions:
            raise ValueError(
                "A pipeline stage must declare regions or include unassigned nodes"
            )
        if not set(self.required_regions).issubset(self.regions):
            raise ValueError("Required pipeline regions must belong to the stage")


@dataclass(frozen=True)
class PipelinePlan:
    """Declarative graph partition and runtime schedule for a pipeline."""

    stages: tuple[PipelineStageSpec, ...]
    schedule: PipelineSchedule

    def __post_init__(self) -> None:
        names = [stage.name for stage in self.stages]
        if len(names) != len(set(names)):
            raise ValueError("Pipeline stage names must be unique")
        unassigned_stages = [
            stage for stage in self.stages if stage.include_unassigned_nodes
        ]
        if len(unassigned_stages) != 1:
            raise ValueError(
                "A pipeline plan must contain exactly one unassigned-nodes stage"
            )
        if not self.stages[-1].include_unassigned_nodes:
            raise ValueError("The unassigned-nodes stage must be last")
        regions = [region for stage in self.stages for region in stage.regions]
        if len(regions) != len(set(regions)):
            raise ValueError("A pipeline region may belong to only one stage")


@dataclass(frozen=True)
class PipelineConfig:
    plan: Optional[PipelinePlan] = None
    gradient_accumulation_steps: int = 1

    def __post_init__(self) -> None:
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be at least 1")

    @classmethod
    def none(cls) -> PipelineConfig:
        return cls()

    @classmethod
    def from_plan(
        cls,
        plan: PipelinePlan,
        *,
        gradient_accumulation_steps: int = 1,
    ) -> PipelineConfig:
        return cls(
            plan=plan,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

    @property
    def enabled(self) -> bool:
        return self.plan is not None


class _PipelinePending:
    pass


PIPELINE_PENDING: Any = _PipelinePending()


def is_pipeline_pending(value: Any) -> bool:
    return value is PIPELINE_PENDING


def mark_pipeline_region(nodes: Iterable[fx.Node], region: Hashable) -> None:
    for node in nodes:
        existing = node.meta.get(_PIPELINE_REGION_META)
        if existing is not None and existing != region:
            raise ValueError(
                f"Pipeline node {node.name!r} belongs to conflicting region "
                f"{existing!r}"
            )
        node.meta[_PIPELINE_REGION_META] = region


@dataclass
class _PipelineSlot:
    env: dict[fx.Node, Any]
    index: int
    event: Optional[torch.cuda.Event] = None


class PipelinedGraphModule(nn.Module):
    """Run a linearly split FX graph as a cross-batch device pipeline."""

    def __init__(
        self,
        split_gm: fx.GraphModule,
        process_outputs: Callable[[Any], Any],
        *,
        stage_names: Sequence[str],
        schedule: PipelineSchedule,
    ) -> None:
        super().__init__()
        self.split_gm = split_gm
        self._process_outputs = process_outputs
        self._placeholders = tuple(
            node for node in split_gm.graph.nodes if node.op == "placeholder"
        )
        stage_calls = tuple(
            node for node in split_gm.graph.nodes if node.op == "call_module"
        )
        indexed_calls: list[tuple[int, fx.Node]] = []
        for stage_call in stage_calls:
            target = cast(str, stage_call.target)
            stage = split_gm.get_submodule(target)
            partition_indices = {
                index
                for node in cast(fx.GraphModule, stage).graph.nodes
                if _is_pipeline_partition_node(node)
                and isinstance(
                    (index := node.meta.get(_PIPELINE_PARTITION_META)),
                    int,
                )
            }
            if len(partition_indices) != 1:
                raise ValueError(
                    f"Pipeline stage {target!r} has ambiguous partition metadata: "
                    f"{sorted(partition_indices)}"
                )
            indexed_calls.append((partition_indices.pop(), stage_call))
        indexed_calls.sort(key=operator.itemgetter(0))
        self._stage_calls = tuple(stage_call for _, stage_call in indexed_calls)
        if [index for index, _ in indexed_calls] != list(range(len(stage_calls))):
            raise ValueError("Pipeline stage indices must be contiguous")
        if len(self._stage_calls) < 2:
            raise ValueError("A device pipeline requires at least two stages")
        self._segments = self._build_segments()
        if len(stage_names) != len(self._stage_calls):
            raise ValueError("Pipeline stage names must match the split graph stages")
        self._stage_names = tuple(stage_names)
        self._slots: list[Optional[_PipelineSlot]] = [
            None for _ in range(len(self._stage_calls) - 1)
        ]
        self._data_dist_stream: Optional[torch.cuda.Stream] = None
        self._schedule = schedule
        self._next_index = 0

    @property
    def graph(self) -> fx.Graph:
        return self.split_gm.graph

    @property
    def has_pending(self) -> bool:
        return any(slot is not None for slot in self._slots)

    def print_readable(self, print_output: bool = True) -> str:
        return self.split_gm.print_readable(print_output)

    def stage_graph_modules(self) -> tuple[fx.GraphModule, ...]:
        stages: list[fx.GraphModule] = []
        for stage_call in self._stage_calls:
            target = cast(str, stage_call.target)
            stage = self.split_gm.get_submodule(target)
            if not isinstance(stage, fx.GraphModule):
                raise TypeError(
                    f"Pipeline stage {target!r} must remain an FX GraphModule, "
                    f"got {type(stage).__name__}"
                )
            stages.append(stage)
        return tuple(stages)

    def lower_stages(
        self, lower: Callable[[fx.GraphModule], nn.Module]
    ) -> PipelinedGraphModule:
        for stage_call in self._stage_calls:
            target = cast(str, stage_call.target)
            stage = self.split_gm.get_submodule(target)
            if not isinstance(stage, fx.GraphModule):
                raise TypeError(
                    f"Pipeline stage {target!r} must be an FX GraphModule, "
                    f"got {type(stage).__name__}"
                )
            self.split_gm.set_submodule(target, lower(stage))
        return self

    def _build_segments(self) -> tuple[tuple[fx.Node, ...], ...]:
        stage_indices = {node: index for index, node in enumerate(self._stage_calls)}
        node_stages = dict(stage_indices)
        pending = [
            node
            for node in self.split_gm.graph.nodes
            if node.op not in ("placeholder", "output") and node not in node_stages
        ]
        while pending:
            remaining: list[fx.Node] = []
            for node in pending:
                parent_stages = {
                    node_stages[parent]
                    for parent in node.all_input_nodes
                    if parent in node_stages
                }
                if parent_stages:
                    node_stages[node] = max(parent_stages)
                    continue
                user_stages = {
                    node_stages[user] for user in node.users if user in node_stages
                }
                if node.op == "get_attr" and user_stages:
                    node_stages[node] = min(user_stages)
                    continue
                remaining.append(node)
            if len(remaining) == len(pending):
                names = ", ".join(node.name for node in remaining)
                raise ValueError(f"Cannot assign pipeline wrapper nodes: {names}")
            pending = remaining

        segments: list[list[fx.Node]] = [[] for _ in self._stage_calls]
        for node in self.split_gm.graph.nodes:
            if node.op == "placeholder":
                continue
            stage_index = (
                len(self._stage_calls) - 1 if node.op == "output" else node_stages[node]
            )
            segments[stage_index].append(node)
        return tuple(tuple(segment) for segment in segments)

    def _bind_inputs(self, args: Sequence[Any]) -> dict[fx.Node, Any]:
        if len(args) != len(self._placeholders):
            raise ValueError(
                f"Expected {len(self._placeholders)} pipeline inputs, got {len(args)}"
            )
        return dict(zip(self._placeholders, args))

    def _fetch_attr(self, target: str) -> Any:
        value: Any = self.split_gm
        for atom in target.split("."):
            value = getattr(value, atom)
        return value

    def _run_node(self, node: fx.Node, env: dict[fx.Node, Any]) -> Any:
        args = map_arg(node.args, lambda input_node: env[input_node])
        kwargs = map_arg(node.kwargs, lambda input_node: env[input_node])
        if node.op == "get_attr":
            return self._fetch_attr(cast(str, node.target))
        if node.op == "call_function":
            return cast(Callable[..., Any], node.target)(*args, **kwargs)
        if node.op == "call_method":
            self_arg, *other_args = args
            return getattr(self_arg, cast(str, node.target))(*other_args, **kwargs)
        if node.op == "call_module":
            return self.split_gm.get_submodule(cast(str, node.target))(*args, **kwargs)
        if node.op == "output":
            return self._process_outputs(args[0])
        raise RuntimeError(f"Unsupported pipeline wrapper node: {node.format_node()}")

    @staticmethod
    def _first_cuda_device(values: Iterable[Any]) -> Optional[torch.device]:
        for value in pytree.tree_leaves(tuple(values)):
            if isinstance(value, torch.Tensor) and value.device.type == "cuda":
                return value.device
        return None

    def _stream_for_stage(
        self, stage_index: int, env: dict[fx.Node, Any]
    ) -> Optional[torch.cuda.Stream]:
        device = self._first_cuda_device(env.values())
        if device is None:
            return None
        if stage_index < len(self._stage_calls) - 1:
            if self._data_dist_stream is None:
                with torch.cuda.device(device):
                    self._data_dist_stream = torch.cuda.Stream(priority=-1)
            return self._data_dist_stream
        return torch.cuda.current_stream(device)

    def _record_compute_inputs(
        self,
        slot: _PipelineSlot,
        stream: torch.cuda.Stream,
    ) -> None:
        seen_tensors: set[int] = set()
        for input_node in self._stage_calls[-1].all_input_nodes:
            if input_node.op == "get_attr" or (
                input_node.op == "placeholder"
                and input_node.meta.get("is_weight", False)
            ):
                continue
            for value in pytree.tree_leaves(slot.env[input_node]):
                if not isinstance(value, torch.Tensor) or value.device.type != "cuda":
                    continue
                tensor_id = id(value)
                if tensor_id in seen_tensors:
                    continue
                seen_tensors.add(tensor_id)
                value.record_stream(stream)

    def _run_stage(
        self,
        stage_index: int,
        slot: _PipelineSlot,
        extra_wait_events: Sequence[torch.cuda.Event] = (),
    ) -> Any:
        stream = self._stream_for_stage(stage_index, slot.env)
        if stream is not None:
            if stage_index == 0 and slot.event is None:
                stream.wait_stream(torch.cuda.current_stream(stream.device))
            if slot.event is not None:
                stream.wait_event(slot.event)
            for event in extra_wait_events:
                stream.wait_event(event)
            if stage_index == len(self._stage_calls) - 1:
                self._record_compute_inputs(slot, stream)

        result: Any = None
        stream_context = (
            torch.cuda.stream(stream) if stream is not None else nullcontext()
        )
        with stream_context:
            for node in self._segments[stage_index]:
                result = self._run_node(node, slot.env)
                if node.op != "output":
                    slot.env[node] = result

        if stream is not None:
            event = torch.cuda.Event()
            event.record(stream)
            slot.event = event
        return result

    def _new_slot(
        self,
        args: Sequence[Any],
        input_ready_event: Optional[torch.cuda.Event],
    ) -> _PipelineSlot:
        slot = _PipelineSlot(
            self._bind_inputs(args),
            self._next_index,
            event=input_ready_event,
        )
        self._next_index += 1
        return slot

    def forward(
        self,
        *args: Any,
        last_iter: bool = False,
        input_ready_event: Optional[torch.cuda.Event] = None,
    ) -> Any:
        """Advance every ready batch by one stage, then optionally admit one."""
        next_slots: list[Optional[_PipelineSlot]] = [
            None for _ in range(len(self._stage_calls) - 1)
        ]
        ready_slots = {
            PipelineAction(
                stage_index, slot.index, self._stage_names[stage_index]
            ): slot
            for stage_index, slot in enumerate(self._slots, start=1)
            if slot is not None
        }
        ready_actions = tuple(ready_slots)
        ordered_actions = tuple(self._schedule.order_actions(ready_actions))
        if len(ordered_actions) != len(ready_actions) or set(ordered_actions) != set(
            ready_actions
        ):
            raise ValueError(
                "Pipeline schedule must return every ready action exactly once; "
                f"ready={ready_actions}, ordered={ordered_actions}"
            )

        retired_output: Any = PIPELINE_PENDING
        completed_events: dict[PipelineAction, Optional[torch.cuda.Event]] = {}
        for action in ordered_actions:
            dependencies = self._schedule.dependencies(action, ready_actions)
            missing_dependencies = [
                dependency
                for dependency in dependencies
                if dependency not in completed_events
            ]
            if missing_dependencies:
                raise ValueError(
                    f"Pipeline action {action} ran before its dependencies "
                    f"{missing_dependencies}"
                )
            extra_wait_events = tuple(
                event
                for dependency in dependencies
                if (event := completed_events[dependency]) is not None
            )
            slot = ready_slots[action]
            result = self._run_stage(action.stage_index, slot, extra_wait_events)
            completed_events[action] = slot.event
            if action.stage_index == len(self._stage_calls) - 1:
                retired_output = result
            else:
                next_slots[action.stage_index] = slot

        if not last_iter:
            new_slot = self._new_slot(args, input_ready_event)
            self._run_stage(0, new_slot)
            next_slots[0] = new_slot

        self._slots = next_slots
        return retired_output


def _assign_stage_dependencies(
    seeds: Iterable[fx.Node],
    stage_index: int,
    explicit_stages: dict[fx.Node, int],
    node_stages: dict[fx.Node, int],
) -> None:
    pending = list(seeds)
    while pending:
        node = pending.pop()
        if node.op in ("placeholder", "get_attr"):
            continue
        explicit_stage = explicit_stages.get(node)
        if explicit_stage is not None and explicit_stage > stage_index:
            raise ValueError(
                f"Pipeline stage {stage_index} depends on later stage "
                f"{explicit_stage} through node {node.name!r}"
            )
        if node in node_stages:
            continue
        node_stages[node] = stage_index
        pending.extend(node.all_input_nodes)


def _assign_stage_getitems(
    stage_index: int,
    explicit_stages: dict[fx.Node, int],
    node_stages: dict[fx.Node, int],
) -> None:
    pending = [node for node, stage in node_stages.items() if stage == stage_index]
    while pending:
        node = pending.pop()
        for user in node.users:
            if (
                user.op != "call_function"
                or user.target is not operator.getitem
                or user in node_stages
                or explicit_stages.get(user, stage_index) != stage_index
            ):
                continue
            node_stages[user] = stage_index
            pending.append(user)


def _validate_pipeline_stages(
    node_stages: dict[fx.Node, int],
    state_read_stages: set[int],
) -> None:
    for node, stage_index in node_stages.items():
        for parent in node.all_input_nodes:
            if (
                parent.op == "placeholder"
                and parent.meta.get("is_weight", False)
                and stage_index not in state_read_stages
            ):
                raise ValueError(
                    "Pipeline stage cannot read model or optimizer state without allow_state_reads: "
                    f"{parent.name}"
                )
            parent_stage = node_stages.get(parent)
            if parent_stage is not None and parent_stage > stage_index:
                raise ValueError(
                    f"Pipeline stage {stage_index} depends on later stage "
                    f"{parent_stage} through node {parent.name!r}"
                )


def _collect_pipeline_stages(
    gm: fx.GraphModule,
    plan: PipelinePlan,
) -> tuple[dict[fx.Node, int], int, tuple[str, ...]]:
    region_nodes: dict[fx.Node, Hashable] = {
        node: cast(Hashable, node.meta[_PIPELINE_REGION_META])
        for node in gm.graph.nodes
        if _PIPELINE_REGION_META in node.meta
    }
    explicit_stages: dict[fx.Node, int] = {}
    mutable_stage_names: list[str] = []
    present_regions = set(region_nodes.values())
    state_read_stages: set[int] = set()
    for stage in plan.stages[:-1]:
        missing_regions = set(stage.required_regions) - present_regions
        if missing_regions:
            missing = ", ".join(sorted(map(str, missing_regions)))
            raise ValueError(
                f"Pipeline stage {stage.name!r} requires marked regions: {missing}"
            )
        stage_nodes = {
            node for node, region in region_nodes.items() if region in stage.regions
        }
        if not stage_nodes:
            continue
        stage_index = len(mutable_stage_names)
        explicit_stages.update({node: stage_index for node in stage_nodes})
        mutable_stage_names.append(stage.name)
        if stage.allow_state_reads:
            state_read_stages.add(stage_index)
    if not explicit_stages:
        raise ValueError(
            "Device pipeline requested, but the graph contains no planned regions"
        )

    compute_stage = len(mutable_stage_names)
    stage_names = (*mutable_stage_names, plan.stages[-1].name)
    node_stages: dict[fx.Node, int] = {}
    for stage_index in range(compute_stage):
        seeds = (
            node for node, stage in explicit_stages.items() if stage == stage_index
        )
        _assign_stage_dependencies(
            seeds,
            stage_index,
            explicit_stages,
            node_stages,
        )
        _assign_stage_getitems(stage_index, explicit_stages, node_stages)

    _validate_pipeline_stages(node_stages, state_read_stages)
    return node_stages, compute_stage, stage_names


def build_pipeline(
    gm: fx.GraphModule,
    config: PipelineConfig,
) -> nn.Module:
    if not config.enabled:
        return gm
    if config.plan is None:
        raise ValueError("Enabled pipeline config requires a pipeline plan")
    if config.gradient_accumulation_steps != 1:
        raise NotImplementedError(
            "Gradient accumulation scheduling is not implemented yet"
        )

    process_outputs = gm.graph._codegen.process_outputs
    node_stages, compute_stage, stage_names = _collect_pipeline_stages(gm, config.plan)
    for node in gm.graph.nodes:
        if _is_pipeline_partition_node(node):
            node.meta[_PIPELINE_PARTITION_META] = node_stages.get(node, compute_stage)
        else:
            node.meta.pop(_PIPELINE_PARTITION_META, None)
    split_gm = split_module(
        gm,
        gm,
        lambda node: node_stages.get(node, compute_stage),
        keep_original_order=False,
        tuple_return=True,
        partition_affix="pipeline",
    )
    for node in tuple(split_gm.graph.nodes):
        if node.op == "get_attr" and not node.users:
            split_gm.graph.erase_node(node)
    split_gm.graph.lint()
    split_gm.recompile()
    pipeline = PipelinedGraphModule(
        split_gm,
        process_outputs,
        stage_names=stage_names,
        schedule=config.plan.schedule,
    )
    logger.info(
        "Built Recir pipeline with %d stages",
        len(pipeline.stage_graph_modules()),
    )
    return pipeline
