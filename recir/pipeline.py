# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import copy
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
_INDUCTOR_STREAM_META = "stream"
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
class PipelineStreamSpec:
    """Assign pipeline regions to one accelerator stream."""

    name: str
    regions: tuple[Hashable, ...]
    priority: int = -1

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Pipeline stream name must not be empty")
        if not self.regions:
            raise ValueError("Pipeline stream must contain at least one region")
        if len(self.regions) != len(set(self.regions)):
            raise ValueError("A pipeline stream may contain each region only once")


@dataclass(frozen=True)
class PipelinePlan:
    """Declarative graph partition and runtime schedule for a pipeline."""

    stages: tuple[PipelineStageSpec, ...]
    schedule: PipelineSchedule
    side_streams: tuple[PipelineStreamSpec, ...] = ()

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
        stream_names = [stream.name for stream in self.side_streams]
        if len(stream_names) != len(set(stream_names)):
            raise ValueError("Pipeline stream names must be unique")
        stream_regions = [
            region for stream in self.side_streams for region in stream.regions
        ]
        if len(stream_regions) != len(set(stream_regions)):
            raise ValueError("A pipeline region may belong to only one stream")
        if not set(stream_regions).issubset(regions):
            raise ValueError("Side-stream regions must belong to a pipeline stage")


@dataclass(frozen=True)
class PipelineConfig:
    plan: Optional[PipelinePlan] = None
    gradient_accumulation_steps: int = 1
    materialize_steady_state: bool = False

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
        materialize_steady_state: bool = False,
    ) -> PipelineConfig:
        return cls(
            plan=plan,
            gradient_accumulation_steps=gradient_accumulation_steps,
            materialize_steady_state=materialize_steady_state,
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


def mark_execution_stream(nodes: Iterable[fx.Node], stream_index: int) -> None:
    """Assign FX nodes to a Dynamo-registered accelerator stream."""
    for node in nodes:
        custom = dict(node.meta.get("custom", {}))
        existing = custom.get(_INDUCTOR_STREAM_META)
        if existing is not None and existing != stream_index:
            raise ValueError(
                f"Pipeline node {node.name!r} has conflicting stream indices "
                f"{existing} and {stream_index}"
            )
        custom[_INDUCTOR_STREAM_META] = stream_index
        node.meta["custom"] = custom


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


class SteadyStateGraphModule(nn.Module):
    """Execute logical stages as one flattened steady-state GraphModule."""

    def __init__(
        self,
        gm: fx.GraphModule,
        process_outputs: Callable[[Any], Any],
        *,
        node_stages: dict[fx.Node, int],
        stage_names: Sequence[str],
        schedule: PipelineSchedule,
        side_streams: Sequence[PipelineStreamSpec],
    ) -> None:
        """Analyze the source graph and materialize steady-state execution state."""
        super().__init__()
        self.source_gm = gm
        self._process_outputs = process_outputs

        # Classify source inputs so weights and loop-carried state can follow
        # different handoff rules from ordinary per-batch inputs.
        self._placeholders = tuple(
            node for node in gm.graph.nodes if node.op == "placeholder"
        )
        self._weight_placeholders = {
            node for node in self._placeholders if node.meta.get("is_weight", False)
        }
        self._state_placeholders = {
            node
            for node in self._placeholders
            if node.meta.get("is_state_input", False)
        }
        self._get_attrs = tuple(
            node for node in gm.graph.nodes if node.op == "get_attr"
        )
        self._output = next(node for node in gm.graph.nodes if node.op == "output")
        self._updated_weight_nodes = self._find_updated_weight_nodes()

        # Cache each stage's ordered nodes and the values crossing its boundary.
        self._stage_names = tuple(stage_names)
        self._node_stages = {
            node: node_stages.get(node, len(stage_names) - 1)
            for node in gm.graph.nodes
            if node.op not in ("placeholder", "get_attr", "output")
        }
        self._stage_nodes = tuple(
            tuple(
                node
                for node in gm.graph.nodes
                if self._node_stages.get(node) == stage_index
            )
            for stage_index in range(len(stage_names))
        )
        self._live_after = self._compute_live_after()
        self._schedule = schedule
        self._stage_order = self._compute_stage_order(schedule)

        # Stream metadata is populated lazily when the graph is lowered.
        self._side_stream_specs = tuple(side_streams)
        self._default_stream_index: Optional[int] = None
        self._side_stream_indices: dict[str, int] = {}
        self._registered_stream_objects: dict[int, object] = {}
        self.steady_gm: nn.Module = self._build_steady_graph()

        # Slots retain one batch's live boundary values between forward calls.
        self._slots: list[Optional[_PipelineSlot]] = [
            None for _ in range(len(stage_names) - 1)
        ]
        self._side_streams: dict[str, torch.cuda.Stream] = {}
        self._last_completion_event: Optional[torch.cuda.Event] = None
        self._last_completion_stream: Optional[torch.cuda.Stream] = None
        self._next_index = 0

    @property
    def graph(self) -> fx.Graph:
        """Expose the original graph for callers that inspect pipeline modules."""
        return self.source_gm.graph

    @property
    def has_pending(self) -> bool:
        """Return whether any admitted batch still occupies a pipeline slot."""
        return any(slot is not None for slot in self._slots)

    def print_readable(self, print_output: bool = True) -> str:
        """Render the materialized graph when available, else the source graph."""
        if isinstance(self.steady_gm, fx.GraphModule):
            return self.steady_gm.print_readable(print_output)
        return str(self.graph)

    def lower_steady(
        self, lower: Callable[[fx.GraphModule], nn.Module]
    ) -> SteadyStateGraphModule:
        """Materialize stream semantics, then lower the flattened steady graph."""
        if not isinstance(self.steady_gm, fx.GraphModule):
            raise TypeError("The steady-state graph has already been lowered")
        self._materialize_execution_streams(self.steady_gm)
        lowered = lower(self.steady_gm)
        self.steady_gm = lowered
        return self

    @staticmethod
    def _runs_on_cuda(node: fx.Node) -> bool:
        """Conservatively determine whether a graph node may execute on CUDA."""
        value = node.meta.get("val")
        if value is None:
            return True
        return any(
            isinstance(leaf, torch.Tensor) and leaf.device.type == "cuda"
            for leaf in pytree.tree_leaves(value)
        )

    def _node_stream(self, node: fx.Node) -> int:
        """Resolve a node's assigned stream, falling back to the default stream."""
        from torch._dynamo.graph_bytecode_inputs import CURRENT_STREAM_INDEX

        default_stream_index = (
            self._default_stream_index
            if self._default_stream_index is not None
            else CURRENT_STREAM_INDEX
        )
        return cast(
            int,
            node.meta.get("custom", {}).get(
                _INDUCTOR_STREAM_META, default_stream_index
            ),
        )

    def _insert_cross_stream_syncs(self, gm: fx.GraphModule) -> None:
        """Insert synchronization for data dependencies that cross streams."""
        from torch._dynamo.graph_bytecode_inputs import get_external_object_by_index
        from torch._dynamo.variables.streams import new_event

        graph = gm.graph
        # Find CUDA producer-consumer edges assigned to different streams.
        edges = [
            (producer, consumer)
            for consumer in graph.nodes
            if consumer.op == "call_function" and self._runs_on_cuda(consumer)
            for producer in consumer.all_input_nodes
            if producer.op == "call_function"
            and self._runs_on_cuda(producer)
            and self._node_stream(producer) != self._node_stream(consumer)
        ]
        event_indices: dict[fx.Node, int] = {}
        waited_streams: set[tuple[fx.Node, int]] = set()
        for producer, consumer in edges:
            # Reuse one completion event for every consumer of a producer.
            event_index = event_indices.get(producer)
            if event_index is None:
                event_index = new_event()
                event_indices[producer] = event_index
                self._registered_stream_objects[event_index] = (
                    get_external_object_by_index(event_index)
                )
                with graph.inserting_after(producer):
                    graph.call_function(
                        torch.ops.streams.record_event.default,
                        (event_index, self._node_stream(producer)),
                    )
            consumer_stream = self._node_stream(consumer)
            wait_key = (producer, consumer_stream)
            # One wait covers every consumer on the same destination stream.
            if wait_key in waited_streams:
                continue
            waited_streams.add(wait_key)
            with graph.inserting_before(consumer):
                graph.call_function(
                    torch.ops.streams.wait_event.default,
                    (event_index, consumer_stream),
                )

    def _materialize_execution_streams(self, gm: fx.GraphModule) -> None:
        """Allocate configured side streams and annotate their graph nodes."""
        if not self._side_stream_specs or self._side_stream_indices:
            return

        # Group eligible CUDA nodes by the region-to-stream plan.
        stream_nodes = {
            spec.name: [
                node
                for node in gm.graph.nodes
                if node.meta.get(_PIPELINE_REGION_META) in spec.regions
                and node.op == "call_function"
                and self._runs_on_cuda(node)
            ]
            for spec in self._side_stream_specs
        }
        if not any(stream_nodes.values()) or not torch.accelerator.is_available():
            return

        from torch._dynamo.graph_bytecode_inputs import get_external_object_by_index
        from torch._dynamo.variables.streams import get_current_stream, new_stream

        current_stream = torch.accelerator.current_stream()
        default_stream_index = get_current_stream(current_stream.device)
        self._default_stream_index = default_stream_index
        self._registered_stream_objects[default_stream_index] = current_stream

        # Register each side stream as a graph input and mark its assigned nodes.
        for spec in self._side_stream_specs:
            nodes = stream_nodes[spec.name]
            if not nodes:
                continue
            stream_index = new_stream(priority=spec.priority)
            stream = get_external_object_by_index(stream_index)
            if not isinstance(stream, torch.Stream):
                raise TypeError(f"Expected accelerator stream, got {type(stream)}")
            self._side_stream_indices[spec.name] = stream_index
            self._side_streams[spec.name] = cast(torch.cuda.Stream, stream)
            self._registered_stream_objects[stream_index] = stream
            mark_execution_stream(nodes, stream_index)

        # Add ordering after all producer and consumer streams are known.
        self._insert_cross_stream_syncs(gm)
        gm.graph.lint()
        gm.recompile()

    def _find_updated_weight_nodes(self) -> dict[fx.Node, fx.Node]:
        # Preserve placeholder order because updated states occupy matching
        # trailing positions in the source graph's flattened output.
        """Match loop-carried state inputs to their updated graph outputs."""
        state_inputs = tuple(
            node for node in self._placeholders if node in self._state_placeholders
        )
        if not state_inputs:
            return {}
        output_leaves = pytree.tree_leaves(self._output.args[0])
        state_outputs = output_leaves[-len(state_inputs) :]
        return {
            state_input: output
            for state_input, output in zip(state_inputs, state_outputs)
            if isinstance(output, fx.Node)
        }

    def _compute_live_after(self) -> tuple[tuple[fx.Node, ...], ...]:
        """Find values that must survive each pipeline-stage boundary."""
        boundaries: list[tuple[fx.Node, ...]] = []
        for boundary in range(len(self._stage_names) - 1):
            # Collect values produced by this boundary that have later consumers.
            live: list[fx.Node] = []
            for node in self.source_gm.graph.nodes:
                # Attributes are fetched on demand and weights are refreshed
                # separately, so neither belongs in per-batch boundary state.
                if (
                    node.op in ("get_attr", "output")
                    or node in self._weight_placeholders
                ):
                    continue
                producer_stage = self._node_stages.get(node, -1)
                if node.op != "placeholder" and producer_stage > boundary:
                    continue
                if any(
                    user.op == "output"
                    or self._node_stages.get(user, len(self._stage_names) - 1)
                    > boundary
                    for user in node.users
                ):
                    live.append(node)
            # Async collectives also need their source storage kept alive
            # until a later invocation consumes the collective result.
            live_set = set(live)
            for node in live:
                if (
                    node.op == "call_function"
                    and node.target
                    == torch.ops._c10d_functional.all_to_all_single.default
                    and isinstance(node.args[0], fx.Node)
                ):
                    # Keep the source storage live while its asynchronous work is
                    # carried to a later pipeline invocation.
                    live_set.add(node.args[0])
            boundaries.append(
                tuple(node for node in self.source_gm.graph.nodes if node in live_set)
            )
        return tuple(boundaries)

    def _compute_stage_order(self, schedule: PipelineSchedule) -> tuple[int, ...]:
        # Model a full pipeline with one synthetic batch at every stage.
        """Validate and return the stage order for one steady-state tick."""
        actions = tuple(
            PipelineAction(
                stage_index,
                len(self._stage_names) - stage_index - 1,
                stage_name,
            )
            for stage_index, stage_name in enumerate(self._stage_names)
        )
        ordered = tuple(schedule.order_actions(actions))
        # The flattened graph must contain every stage exactly once.
        if len(ordered) != len(actions) or set(ordered) != set(actions):
            raise ValueError(
                "Steady-state schedule must return every stage exactly once"
            )
        # Explicit same-tick dependencies need graph edges that are not yet
        # represented by the steady-state materializer.
        for action in ordered:
            if schedule.dependencies(action, actions):
                raise NotImplementedError(
                    "Steady-state materialization does not yet support explicit "
                    "same-tick schedule dependencies"
                )
        return tuple(action.stage_index for action in ordered)

    @staticmethod
    def _copy_node(
        graph: fx.Graph,
        node: fx.Node,
        values: dict[fx.Node, fx.Node],
    ) -> fx.Node:
        """Copy a source node while remapping its input values."""
        copied = graph.node_copy(node, lambda source: values[source])
        copied.meta = copy.copy(node.meta)
        return copied

    def _build_steady_graph(self) -> fx.GraphModule:
        """Flatten all in-flight stages into one state-in/state-out graph."""
        graph = fx.Graph()

        # Recreate the public inputs shared by the in-flight stages.
        argument_nodes: dict[fx.Node, fx.Node] = {}
        for placeholder in self._placeholders:
            copied = graph.placeholder(
                cast(str, placeholder.target), type_expr=placeholder.type
            )
            copied.meta = copy.copy(placeholder.meta)
            argument_nodes[placeholder] = copied

        # Module attributes are copied once and reused by every copied stage.
        attribute_nodes: dict[fx.Node, fx.Node] = {}
        for get_attr in self._get_attrs:
            attribute_nodes[get_attr] = self._copy_node(graph, get_attr, {})

        # Turn each boundary's live values into explicit steady-graph inputs.
        state_inputs: list[dict[fx.Node, fx.Node]] = [{}]
        for stage_index, live_values in enumerate(self._live_after, start=1):
            inputs: dict[fx.Node, fx.Node] = {}
            for value in live_values:
                state_input = graph.placeholder(
                    f"pipeline_stage_{stage_index}_state_{value.name}",
                    type_expr=value.type,
                )
                state_input.meta = copy.copy(value.meta)
                inputs[value] = state_input
            state_inputs.append(inputs)

        # Copy stages in schedule order while tracking each source-to-copy map.
        stage_values: list[dict[fx.Node, fx.Node]] = [{} for _ in self._stage_names]
        for stage_index in self._stage_order:
            values = dict(attribute_nodes)
            values.update(state_inputs[stage_index])
            for placeholder in self._placeholders:
                if stage_index != 0 and placeholder not in self._weight_placeholders:
                    continue
                # Feed state updated by the retiring batch directly into the
                # newly admitted batch when the final stage is already copied.
                # TODO: Make fresh-vs-stale state selection explicit and validate
                # the required stage order instead of inferring it from copy order.
                updated = self._updated_weight_nodes.get(placeholder)
                final_values = stage_values[-1]
                values[placeholder] = (
                    final_values[updated]
                    if stage_index == 0
                    and updated is not None
                    and updated in final_values
                    else argument_nodes[placeholder]
                )
            for node in self._stage_nodes[stage_index]:
                values[node] = self._copy_node(graph, node, values)
            stage_values[stage_index] = values

        # Return both the user-visible result and every next-boundary state.
        final_values = stage_values[-1]
        raw_output = map_arg(self._output.args[0], lambda node: final_values[node])
        next_states = tuple(
            tuple(stage_values[index][node] for node in live_values)
            for index, live_values in enumerate(self._live_after)
        )
        graph.output((raw_output, next_states))
        steady_gm = fx.GraphModule(self.source_gm, graph)
        steady_gm.meta = copy.copy(getattr(self.source_gm, "meta", {}))
        return steady_gm

    def _bind_inputs(self, args: Sequence[Any]) -> dict[fx.Node, Any]:
        """Bind runtime positional arguments to source placeholders."""
        if len(args) != len(self._placeholders):
            raise ValueError(
                f"Expected {len(self._placeholders)} pipeline inputs, got {len(args)}"
            )
        return dict(zip(self._placeholders, args))

    def _fetch_attr(self, target: str) -> Any:
        """Resolve a dotted get_attr target from the source module."""
        value: Any = self.source_gm
        for atom in target.split("."):
            value = getattr(value, atom)
        return value

    def _run_node(self, node: fx.Node, env: dict[fx.Node, Any]) -> Any:
        """Interpret one supported FX node against a slot environment."""
        args = map_arg(node.args, lambda input_node: env[input_node])
        kwargs = map_arg(node.kwargs, lambda input_node: env[input_node])
        if node.op == "call_function":
            return cast(Callable[..., Any], node.target)(*args, **kwargs)
        if node.op == "call_method":
            self_arg, *other_args = args
            return getattr(self_arg, cast(str, node.target))(*other_args, **kwargs)
        if node.op == "call_module":
            return self.source_gm.get_submodule(cast(str, node.target))(*args, **kwargs)
        raise RuntimeError(f"Unsupported pipeline node: {node.format_node()}")

    @staticmethod
    def _first_cuda_device(values: Iterable[Any]) -> Optional[torch.device]:
        """Find the first CUDA device represented in nested runtime values."""
        for value in pytree.tree_leaves(tuple(values)):
            if isinstance(value, torch.Tensor) and value.device.type == "cuda":
                return value.device
        return None

    def _stream_for_stage(
        self, stage_index: int, env: dict[fx.Node, Any]
    ) -> Optional[torch.cuda.Stream]:
        """Select the current CUDA stream for a stage's runtime values."""
        del stage_index
        device = self._first_cuda_device(env.values())
        if device is None:
            return None
        return torch.cuda.current_stream(device)

    @staticmethod
    def _record_values_on_stream(
        values: Iterable[Any],
        stream: torch.cuda.Stream,
    ) -> None:
        """Keep CUDA tensors alive until their consuming stream completes."""
        seen: set[int] = set()
        for value in pytree.tree_leaves(tuple(values)):
            if not isinstance(value, torch.Tensor) or not value.is_cuda:
                continue
            tensor_id = id(value)
            if tensor_id in seen:
                continue
            seen.add(tensor_id)
            value.record_stream(stream)

    def _refresh_weight_inputs(self, slot: _PipelineSlot, args: Sequence[Any]) -> None:
        """Refresh a pending slot with the current weight and state inputs."""
        for placeholder, value in zip(self._placeholders, args):
            if placeholder in self._weight_placeholders:
                slot.env[placeholder] = value

    def _run_stage_nodes(
        self,
        stage_index: int,
        slot: _PipelineSlot,
    ) -> None:
        """Execute one stage and release values after their final use."""
        stage_nodes = self._stage_nodes[stage_index]

        # Count uses within this stage so dead intermediates can be released early.
        remaining_uses: dict[fx.Node, int] = {}
        for stage_node in stage_nodes:
            for input_node in stage_node.all_input_nodes:
                remaining_uses[input_node] = remaining_uses.get(input_node, 0) + 1
        # Preserve graph outputs at the last stage and boundary state otherwise.
        if stage_index == len(self._stage_names) - 1:
            keep_values = {
                leaf
                for leaf in pytree.tree_leaves(self._output.args[0])
                if isinstance(leaf, fx.Node)
            }
        else:
            keep_values = set(self._live_after[stage_index])

        # Execute in graph order and drop values immediately after their last use.
        for node in stage_nodes:
            slot.env[node] = self._run_node(node, slot.env)
            for input_node in node.all_input_nodes:
                uses = remaining_uses[input_node] - 1
                remaining_uses[input_node] = uses
                if uses == 0 and input_node not in keep_values:
                    slot.env.pop(input_node, None)
            if remaining_uses.get(node, 0) == 0 and node not in keep_values:
                slot.env.pop(node, None)

    def _run_stage(
        self,
        stage_index: int,
        slot: _PipelineSlot,
        extra_wait_events: Sequence[torch.cuda.Event] = (),
    ) -> Any:
        # Wait for prior work, then transfer tensor lifetime to this stage's stream.
        """Synchronize, execute, and retain the live outputs of one stage."""
        stream = self._stream_for_stage(stage_index, slot.env)
        if stream is not None:
            if slot.event is not None:
                stream.wait_event(slot.event)
            for event in extra_wait_events:
                stream.wait_event(event)
            self._record_values_on_stream(slot.env.values(), stream)

        # Resolve attributes at execution time so mutations remain visible.
        for get_attr in self._get_attrs:
            slot.env[get_attr] = self._fetch_attr(cast(str, get_attr.target))
        is_last_stage = stage_index == len(self._stage_names) - 1

        stream_context = (
            torch.cuda.stream(stream) if stream is not None else nullcontext()
        )
        # Only the final stage materializes the source graph's user-facing output.
        with stream_context:
            self._run_stage_nodes(stage_index, slot)
            result = None
            if is_last_stage:
                raw_output = map_arg(
                    self._output.args[0], lambda input_node: slot.env[input_node]
                )
                result = self._process_outputs(raw_output)

        # Publish completion and retain only values needed across the next boundary.
        if stream is not None:
            event = torch.cuda.Event()
            event.record(stream)
            slot.event = event
        if stage_index < len(self._stage_names) - 1:
            slot.env = {node: slot.env[node] for node in self._live_after[stage_index]}
        return result

    def _advance_eager(
        self,
        args: Sequence[Any],
        last_iter: bool,
        input_ready_event: Optional[torch.cuda.Event],
    ) -> Any:
        # Associate each occupied slot with the stage it is ready to enter.
        """Advance existing slots and optionally admit one batch eagerly."""
        next_slots: list[Optional[_PipelineSlot]] = [None for _ in self._slots]
        ready_slots = {
            PipelineAction(
                stage_index, slot.index, self._stage_names[stage_index]
            ): slot
            for stage_index, slot in enumerate(self._slots, start=1)
            if slot is not None
        }
        # Validate that scheduling neither drops nor duplicates ready work.
        actions = tuple(ready_slots)
        ordered = tuple(self._schedule.order_actions(actions))
        if len(ordered) != len(actions) or set(ordered) != set(actions):
            raise ValueError("Pipeline schedule must return every ready action once")
        # Advance existing batches, honoring dependencies between this tick's work.
        retired_output: Any = PIPELINE_PENDING
        completed_events: dict[PipelineAction, Optional[torch.cuda.Event]] = {}
        for action in ordered:
            dependencies = self._schedule.dependencies(action, actions)
            missing = [
                dependency
                for dependency in dependencies
                if dependency not in completed_events
            ]
            if missing:
                raise ValueError(
                    f"Pipeline action {action} ran before its dependencies {missing}"
                )
            slot = ready_slots[action]
            self._refresh_weight_inputs(slot, args)
            waits = tuple(
                event
                for dependency in dependencies
                if (event := completed_events[dependency]) is not None
            )
            result = self._run_stage(action.stage_index, slot, waits)
            completed_events[action] = slot.event
            if action.stage_index == len(self._stage_names) - 1:
                retired_output = result
            else:
                next_slots[action.stage_index] = slot

        # During fill or normal operation, admit one new batch into stage zero.
        if not last_iter:
            slot = _PipelineSlot(
                self._bind_inputs(args), self._next_index, input_ready_event
            )
            self._next_index += 1
            self._run_stage(0, slot)
            next_slots[0] = slot
        self._slots = next_slots
        return retired_output

    def _steady_inputs(self, args: Sequence[Any]) -> list[Any]:
        """Pack current arguments and pending boundary state for steady execution."""
        inputs = list(args)

        # Append boundary state in the same order as the generated placeholders.
        for stage_index, live_values in enumerate(self._live_after, start=1):
            slot = self._slots[stage_index - 1]
            if slot is None:
                raise RuntimeError("Steady-state pipeline slot is empty")
            inputs.extend(slot.env[node] for node in live_values)
        return inputs

    def _wait_for_steady_inputs(
        self,
        inputs: Sequence[Any],
        input_ready_event: Optional[torch.cuda.Event],
    ) -> Optional[torch.cuda.Stream]:
        # CPU execution needs no stream-level synchronization.
        """Make execution streams wait for all incoming state and input events."""
        device = self._first_cuda_device(inputs)
        if device is None:
            return None
        stream = torch.cuda.current_stream(device)
        streams = (stream, *self._side_streams.values())

        # New external inputs are consumed only by side-stream graph work.
        if input_ready_event is not None:
            for execution_stream in self._side_streams.values():
                execution_stream.wait_event(input_ready_event)
        # Make every execution stream wait once for each pending slot.
        seen_events: set[int] = set()
        for slot in self._slots:
            if slot is not None and slot.event is not None:
                event_id = id(slot.event)
                if event_id in seen_events:
                    continue
                seen_events.add(event_id)
                execution_streams = streams
                # The default stream produced this event, so only side streams
                # need an explicit wait when both invocations use that stream.
                if (
                    slot.event is self._last_completion_event
                    and stream == self._last_completion_stream
                ):
                    execution_streams = tuple(self._side_streams.values())
                for execution_stream in execution_streams:
                    execution_stream.wait_event(slot.event)
        return stream

    def _register_execution_streams(self, stream: torch.cuda.Stream) -> None:
        """Bind captured stream indices to this invocation's stream objects."""
        from torch._dynamo.graph_bytecode_inputs import set_external_object_by_index

        for index, value in self._registered_stream_objects.items():
            set_external_object_by_index(
                index,
                stream if index == self._default_stream_index else value,
            )

    def _run_steady(
        self,
        args: Sequence[Any],
        input_ready_event: Optional[torch.cuda.Event],
    ) -> Any:
        # Pack explicit state inputs and establish their stream ownership.
        """Run one flattened tick and rebuild slots from its state outputs."""
        inputs = self._steady_inputs(args)
        stream = self._wait_for_steady_inputs(inputs, input_ready_event)
        if stream is not None:
            self._register_execution_streams(stream)
            self._record_values_on_stream(inputs, stream)

        # The generated graph's ABI is (user output, state for each boundary).
        result = self.steady_gm(*inputs)
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise RuntimeError(
                "Steady-state graph must return output and pipeline state"
            )
        raw_output, state_outputs = result
        if len(state_outputs) != len(self._live_after):
            raise RuntimeError("Steady-state graph returned an invalid state layout")

        # Join side streams onto the default stream before publishing completion.
        event: Optional[torch.cuda.Event] = None
        if stream is not None:
            for side_stream in self._side_streams.values():
                side_stream_event = torch.cuda.Event()
                side_stream_event.record(side_stream)
                stream.wait_event(side_stream_event)
            event = torch.cuda.Event()
            event.record(stream)
        self._last_completion_event = event
        self._last_completion_stream = stream

        # Rebuild one slot per boundary from the graph's explicit state outputs.
        next_slots: list[Optional[_PipelineSlot]] = []
        # Retain current weights as well so a later eager drain can read them.
        weight_values = {
            placeholder: value
            for placeholder, value in zip(self._placeholders, args)
            if placeholder in self._weight_placeholders
        }
        for stage_index, (live_values, values) in enumerate(
            zip(self._live_after, state_outputs)
        ):
            env = dict(weight_values)
            env.update(zip(live_values, values))
            # Boundary zero belongs to the newly admitted batch; later
            # boundaries preserve the batch identity from the preceding slot.
            batch_index = (
                self._next_index
                if stage_index == 0
                else cast(_PipelineSlot, self._slots[stage_index - 1]).index
            )
            next_slots.append(_PipelineSlot(env, batch_index, event))
        self._next_index += 1
        self._slots = next_slots
        return self._process_outputs(raw_output)

    def forward(
        self,
        *args: Any,
        last_iter: bool = False,
        input_ready_event: Optional[torch.cuda.Event] = None,
    ) -> Any:
        """Use flattened execution only when all pipeline slots are populated."""
        if not last_iter and all(slot is not None for slot in self._slots):
            return self._run_steady(
                args,
                input_ready_event,
            )
        return self._advance_eager(args, last_iter, input_ready_event)


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
    if config.materialize_steady_state:
        return SteadyStateGraphModule(
            gm,
            process_outputs,
            node_stages=node_stages,
            stage_names=stage_names,
            schedule=config.plan.schedule,
            side_streams=config.plan.side_streams,
        )
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
