# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# This file is a fork and is not kept in sync with its upstream version.

import operator
from contextlib import ExitStack
from typing import Any, Callable, cast, List, Optional, Tuple

import torch
from recir.lowering.lowering_utils import (
    has_meta_kernel,
    segment_and_lower,
    strip_profiler_nodes,
)
from torch._inductor.custom_graph_pass import CustomGraphPass


def _detach_tensor_args(
    args: tuple,  # type: ignore[type-arg]
) -> tuple:  # type: ignore[type-arg]
    """Detach inputs so the pre-baked forward/backward/optimizer graph uses the
    AOTAutograd inference path while preserving storage for parameter updates."""
    return tuple(a.detach() if isinstance(a, torch.Tensor) else a for a in args)


def _shape_envs_for_value(value: Any) -> list[Any]:
    shape_envs = []
    seen: set[int] = set()
    for leaf in torch.utils._pytree.tree_leaves(value):
        graph_mode = torch._guards.detect_fake_mode((leaf,))
        graph_shape_env = (
            graph_mode.shape_env
            if graph_mode is not None and graph_mode.shape_env is not None
            else None
        )
        if graph_shape_env is None and isinstance(
            leaf, (torch.SymInt, torch.SymFloat, torch.SymBool)
        ):
            graph_shape_env = leaf.node.shape_env
        if graph_shape_env is not None and id(graph_shape_env) not in seen:
            shape_envs.append(graph_shape_env)
            seen.add(id(graph_shape_env))
    return shape_envs


def _shape_env_for_subgraph(gm: torch.fx.GraphModule) -> Any:
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    shape_env = ShapeEnv()
    next_unbacked = 0
    next_unbacked_float = 0
    for node in gm.graph.nodes:
        value = node.meta.get("val")
        if value is None:
            continue
        for graph_shape_env in _shape_envs_for_value(value):
            next_unbacked = max(
                next_unbacked,
                graph_shape_env.unbacked_symint_counter,
            )
            next_unbacked_float = max(
                next_unbacked_float,
                graph_shape_env.unbacked_symfloat_counter,
            )
            for symbol in node.meta.get("unbacked_bindings", {}):
                value_range = graph_shape_env.var_to_range.get(symbol)
                if value_range is not None:
                    shape_env._update_var_to_range(symbol, value_range)
                symbol_stack = graph_shape_env.var_to_stack.get(symbol)
                if symbol_stack is not None:
                    shape_env.var_to_stack[symbol] = symbol_stack
    shape_env.unbacked_symint_counter = next_unbacked
    shape_env.unbacked_symfloat_counter = next_unbacked_float
    return shape_env


def _compile_fx_with_fresh_shape_env(
    gm: torch.fx.GraphModule,
    example_inputs: tuple,  # type: ignore[type-arg]
) -> Callable[..., Any]:
    """Compile with a ShapeEnv scoped to this lowered subgraph."""
    from torch._dynamo.source import LocalSource
    from torch._inductor.compile_fx import compile_fx
    from torch._subclasses import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import (
        DimDynamic,
        StatelessSymbolicContext,
    )

    fake_mode = FakeTensorMode(shape_env=_shape_env_for_subgraph(gm))
    fake_inputs = []
    placeholders = (node for node in gm.graph.nodes if node.op == "placeholder")
    for index, (placeholder, example_input) in enumerate(
        zip(placeholders, example_inputs)
    ):
        graph_input = placeholder.meta.get("val")
        if not (
            isinstance(graph_input, torch.Tensor)
            and isinstance(example_input, torch.Tensor)
        ):
            fake_inputs.append(example_input)
            continue
        batch_dim_value = placeholder.meta.get("batch_dim")
        batch_dim: Optional[int] = (
            batch_dim_value
            if placeholder.meta.get("is_batch_input")
            and isinstance(batch_dim_value, int)
            else None
        )
        symbolic_context = StatelessSymbolicContext(
            dynamic_sizes=[
                DimDynamic.DYNAMIC
                if dimension == batch_dim
                or isinstance(size, torch.SymInt)
                and size.node.maybe_as_int() is None
                else DimDynamic.STATIC
                for dimension, size in enumerate(graph_input.shape)
            ]
        )
        fake_inputs.append(
            fake_mode.from_tensor(
                example_input,
                static_shapes=False,
                source=LocalSource(f"partial_lower_input_{index}"),
                symbolic_context=symbolic_context,
            )
        )

    with torch._guards.tracing(torch._guards.TracingContext(fake_mode)):
        return cast(
            Callable[..., Any],
            compile_fx(gm, tuple(fake_inputs), ignore_shape_env=True),
        )


_ALL_REDUCE_BUCKET_CAP_MB = 600.0  # TODO: make this configurable
_ORDER_AFTER_STREAM_SYNC = "recir_order_after_stream_sync"


def _fixed_bucket_cap_mb(_bucket_index: int) -> float:
    return _ALL_REDUCE_BUCKET_CAP_MB


def _prune_control_deps_to_frontier(graph: torch.fx.Graph) -> None:
    """Remove ordering deps transitively covered by another ordering dep."""
    from torch._inductor.fx_passes.control_dependencies import control_deps

    for node in graph.nodes:
        if node.op != "call_function" or node.target is not control_deps:
            continue
        deps = cast(tuple[torch.fx.Node, ...], node.args[0])
        dep_set = set(deps)
        redundant_deps: set[torch.fx.Node] = set()
        visited: set[torch.fx.Node] = set()
        worklist = [input_node for dep in deps for input_node in dep.all_input_nodes]
        while worklist:
            candidate = worklist.pop()
            if candidate in visited:
                continue
            visited.add(candidate)
            if candidate in dep_set:
                redundant_deps.add(candidate)
            worklist.extend(candidate.all_input_nodes)
        frontier = tuple(dep for dep in deps if dep not in redundant_deps)
        if frontier != deps:
            node.args = (frontier, *node.args[1:])


def _mark_order_after_stream_sync(
    node: torch.fx.Node, sync_node: torch.fx.Node
) -> None:
    custom = node.meta.get("custom")
    custom = custom.copy() if isinstance(custom, dict) else {}
    ordering = list(custom.get(_ORDER_AFTER_STREAM_SYNC, ()))
    if sync_node.name not in ordering:
        ordering.append(sync_node.name)
    custom[_ORDER_AFTER_STREAM_SYNC] = tuple(ordering)
    node.meta["custom"] = custom


def _remove_stream_sync_passthroughs(graph: torch.fx.Graph) -> None:  # noqa: C901
    from torch._inductor.fx_passes.control_dependencies import control_deps

    owning_module = graph.owning_module
    if owning_module is None:
        raise AssertionError("expected graph to have an owning module")

    sync_targets = (
        torch.ops.streams.record_event.default,
        torch.ops.streams.wait_event.default,
    )
    for node in list(graph.nodes):
        if (
            node.op != "call_function"
            or node.target is not control_deps
            or len(node.args) <= 2
        ):
            continue
        subgraph_attr = node.args[1]
        if not isinstance(subgraph_attr, torch.fx.Node) or not isinstance(
            subgraph_attr.target, str
        ):
            continue
        subgraph = getattr(owning_module, subgraph_attr.target)
        sync_nodes = [
            subgraph_node
            for subgraph_node in subgraph.graph.nodes
            if subgraph_node.op == "call_function"
            and subgraph_node.target in sync_targets
        ]
        if len(sync_nodes) != 1:
            continue

        for index, passthrough in enumerate(node.args[2:], start=1):
            if not isinstance(passthrough, torch.fx.Node):
                raise AssertionError("expected tensor passthrough to be an FX node")
            projections = [
                user
                for user in list(node.users)
                if user.op == "call_function"
                and user.target is operator.getitem
                and user.args[1] == index
            ]
            for projection in projections:
                for user in list(projection.users):
                    _mark_order_after_stream_sync(user, node)
                projection.replace_all_uses_with(passthrough)
                graph.erase_node(projection)

        node.args = node.args[:2]
        output = subgraph.graph.output_node()
        output_value = output.args[0]
        if not isinstance(output_value, tuple) or not output_value:
            raise AssertionError("expected stream sync subgraph tuple output")
        sync_output = output_value[0]
        output.args = (sync_output,)
        output.meta["val"] = (
            sync_output.meta.get("val")
            if isinstance(sync_output, torch.fx.Node)
            else sync_output
        )
        node.meta["val"] = output.meta["val"]
        for subgraph_node in reversed(list(subgraph.graph.nodes)):
            if subgraph_node.op == "placeholder" and not subgraph_node.users:
                subgraph.graph.erase_node(subgraph_node)
        subgraph.recompile()


class _StreamSyncOrderingGraphPass(CustomGraphPass):
    def __call__(self, graph: torch.fx.Graph) -> None:
        _remove_stream_sync_passthroughs(graph)
        _prune_control_deps_to_frontier(graph)

        from torch._inductor.fx_passes.control_dependencies import (
            control_deps,
            preserve_node_ordering,
        )
        from torch.utils._ordered_set import OrderedSet

        sync_nodes = {
            node.name: node
            for node in graph.nodes
            if node.op == "call_function" and node.target is control_deps
        }
        deps_map = {}
        for node in graph.nodes:
            custom = node.meta.get("custom")
            if not isinstance(custom, dict):
                continue
            sync_names = custom.pop(_ORDER_AFTER_STREAM_SYNC, ())
            if (
                sync_names
                and node.op == "call_function"
                and node.target is not control_deps
            ):
                deps_map[node] = OrderedSet(
                    sync_nodes[sync_name]
                    for sync_name in sync_names
                    if sync_name in sync_nodes
                )
        preserve_node_ordering(graph, deps_map)

    def uuid(self) -> str:
        return "recir-stream-sync-ordering-v2"


class _InductorModule(torch.nn.Module):
    def __init__(self, gm: torch.fx.GraphModule) -> None:
        super().__init__()
        self.gm = gm
        self.compiled: Optional[Callable[..., Any]] = None

    def forward(self, *args: torch.Tensor, tag: str) -> List[torch.Tensor]:
        detached_args = _detach_tensor_args(args)
        if self.compiled is None:
            has_stream_sync = any(
                node.op == "call_function"
                and node.target
                in (
                    torch.ops.streams.record_event.default,
                    torch.ops.streams.wait_event.default,
                )
                for node in self.gm.graph.nodes
            )
            if has_stream_sync:
                from torch._functorch._aot_autograd.streams import (
                    wrap_all_sync_nodes_with_control_deps,
                )

                wrap_all_sync_nodes_with_control_deps(self.gm)
                _prune_control_deps_to_frontier(self.gm.graph)
            has_collectives = any(
                n.op == "call_function"
                and hasattr(n.target, "__module__")
                and "_c10d_functional" in str(n.target)
                for n in self.gm.graph.nodes
            )
            # Replace aten.empty with aten.zeros to avoid uninitialized
            # memory reads if Inductor reorders ops.
            for n in self.gm.graph.nodes:
                if n.op == "call_function" and n.target in (
                    torch.ops.aten.empty.memory_format,
                ):
                    n.target = torch.ops.aten.zeros.default
                    if "memory_format" in n.kwargs:
                        n.kwargs = {
                            k: v for k, v in n.kwargs.items() if k != "memory_format"
                        }
            self.gm.recompile()

            from torch._inductor import config as _inductor_config

            inductor_config: dict[str, Any] = {}
            comms_config_patch: Any = None
            if has_collectives:
                from torch._inductor import config_comms as _inductor_config_comms

                inductor_config.update(
                    {
                        "bucket_all_reduces_fx": "all",
                        "bucket_all_reduces_fx_bucket_size_determinator": _fixed_bucket_cap_mb,
                        "reorder_for_peak_memory": False,
                        "reorder_for_compute_comm_overlap": True,
                        "reorder_for_compute_comm_overlap_passes": [
                            "reorder_communication_preserving_peak_memory",
                            "sink_waits_iterative",
                            "reorder_communication_preserving_peak_memory",
                        ],
                        "aten_distributed_optimizations.enable_simple_overlap": False,
                        "aten_distributed_optimizations.enable_overlap_scheduling": False,
                    }
                )
                fixed_order_config = {
                    "reorder_iterative_unsafe_collectives_reorder": False,
                    "sink_waits_iterative_unsafe_collectives_reorder": False,
                    "reorder_iterative_peak_memory_budget": 0.0,
                    "sink_iterative_peak_memory_budget": 0.0,
                }
                comms_config_patch = cast(Any, _inductor_config_comms).patch(
                    fixed_order_config
                )

            if has_stream_sync:
                from torch._inductor.custom_graph_pass import get_custom_graph_passes

                inductor_config["post_grad_custom_pre_pass"] = [
                    *get_custom_graph_passes(
                        _inductor_config.post_grad_custom_pre_pass
                    ),
                    _StreamSyncOrderingGraphPass(),
                ]

            with ExitStack() as stack:
                stack.enter_context(_inductor_config.patch(inductor_config))
                if comms_config_patch is not None:
                    stack.enter_context(comms_config_patch)
                self.compiled = _compile_fx_with_fresh_shape_env(self.gm, detached_args)
        with torch.profiler.record_function(tag):
            compiled = self.compiled
            assert compiled is not None
            result = compiled(*detached_args)
            if isinstance(result, (tuple, list)):
                result = list(result)
            else:
                result = [result]
            return result


_INDUCTOR_INCOMPATIBLE_OPS: set[torch._ops.OpOverload] = {
    torch.ops.aten._fused_adam_.default,
    torch.ops.aten._fused_adam.default,
    torch.ops.aten._foreach_add_.Scalar,
    torch.ops.aten._foreach_add.Scalar,
}

_INDUCTOR_COMPATIBLE_OPS: set[torch._ops.OpOverload] = {
    torch.ops.aten._local_scalar_dense.default,
    torch.ops.aten.native_dropout.default,
    torch.ops.aten.flatten.using_ints,
    torch.ops.aten.full_like.default,
    torch.ops.aten.rand_like.default,
    torch.ops.aten.sym_size.int,
    torch.ops.aten.sym_stride.int,
    torch.ops.aten.select.int,
    torch.ops.aten.view.default,
    torch.ops.aten.lift_fresh_copy.default,
}


def _is_inductor_compatible(node: torch.fx.Node) -> Tuple[bool, str]:
    if node.target in _INDUCTOR_INCOMPATIBLE_OPS:
        return False, "fused adam is not supported yet"

    if node.target in _INDUCTOR_COMPATIBLE_OPS:
        return True, ""

    if isinstance(node.target, torch._ops.OpOverload):
        target = node.target
        if "_c10d_functional" in target.name():
            return True, ""
        if not has_meta_kernel(target):
            return False, f"{node.target} doesn't have a meta kernel registered"
    return True, ""


def partial_lower(
    gm: torch.fx.GraphModule,
    node_predicate: Callable[[torch.fx.Node], bool] = lambda x: True,
    subgraph_predicate: Callable[[List[torch.fx.Node]], bool] = lambda x: True,
    dumper: Callable[[str], str] = lambda x: "subgraph",
) -> torch.fx.GraphModule:
    """
    Lower Inductor compatible portions of the graph module to Inductor.

    Args:
        node_predicate: user predicate for determining whether to consider a node for
            lowering.
        subgraph_predicate: user predicate for determining whether to consider a list of
            candidate nodes for lowering.
        dumper: a callback for dumping subgraphs for human digestion. For exmaple, it
            can be a function that writes to disk/blob storage and returns the
            path/handle. The returned path/handle for each subgraph will be made
            available in the subgraph call node in the parent graph, as well as the
            label of the profiler block for the subgraph.
    """
    gm = strip_profiler_nodes(gm)
    return segment_and_lower(
        gm,
        compatibility_fn=_is_inductor_compatible,
        module_factory=_InductorModule,
        subgraph_prefix="subgraph",
        node_predicate=node_predicate,
        subgraph_predicate=subgraph_predicate,
        dumper=dumper,
    )
