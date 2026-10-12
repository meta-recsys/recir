# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Smoke tests for the v7.1 promoted graph utilities.

Verifies that ``find_markers``, ``extract_subgraph``, ``replace_region``, and
``add_state`` work end-to-end on a small traced graph and are reachable from
the top-level ``recir`` package.
"""

import unittest
from types import SimpleNamespace
from typing import Any, Callable, List
from unittest.mock import MagicMock, patch

import recir
import recir.compiler.compilation as recir_compilation
import torch
import torch.nn as nn
from recir.api import (
    add_state,
    compile as recir_compile,
    extract_subgraph,
    find_markers,
    replace_region,
)
from recir.compiler.process_state import _build_compiled_process_state
from recir.compiler.runtime import (
    COMPILED_OBJECT_KEY,
    get_compiled_process_state,
    invalidate_device_state,
    make_compiled_runner,
    materialize_compiled_output,
    prepare_compiled,
    set_compiled_output_deferred,
)
from recir.distributed.sharding_spec import FSDPWeightShardSpec
from recir.region_transforms import RegionTransform
from torch import fx
from torch.fx.experimental.proxy_tensor import make_fx


def _make_traced_gm_with_markers(num_markers: int = 2) -> fx.GraphModule:
    """Trace a tiny model with ``num_markers`` ``mark_boundary`` calls."""

    class _Model(nn.Module):
        def __init__(self, n: int) -> None:
            super().__init__()
            self._n = n
            self.linear = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for i in range(self._n):
                x = torch.ops.recir.mark_boundary(x, f"marker_{i}")
            return self.linear(x)

    model = _Model(num_markers)
    x = torch.randn(2, 4)
    return make_fx(model, tracing_mode="fake", _allow_non_fake_inputs=True)(x)


class FindMarkersTest(unittest.TestCase):
    """recir.find_markers locates every ``mark_boundary`` node."""

    def test_returns_all_markers_in_order(self) -> None:
        """Every mark_boundary call appears in the returned list, in graph order."""
        gm = _make_traced_gm_with_markers(num_markers=3)
        markers = find_markers(gm)
        self.assertEqual(len(markers), 3)
        names = [m.args[1] for m in markers]
        self.assertEqual(names, ["marker_0", "marker_1", "marker_2"])

    def test_returns_empty_list_when_no_markers(self) -> None:
        """An ungraph-marked graph returns an empty list (not an error)."""

        class _Plain(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(4, 4)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x)

        gm = make_fx(_Plain(), tracing_mode="fake", _allow_non_fake_inputs=True)(
            torch.randn(2, 4)
        )
        self.assertEqual(find_markers(gm), [])

    def test_top_level_import_matches_api(self) -> None:
        """``recir.find_markers`` is the same function as ``recir.api.find_markers``."""
        self.assertIs(recir.find_markers, find_markers)


class ExtractSubgraphTest(unittest.TestCase):
    """recir.extract_subgraph lifts a set of nodes into a sub-GraphModule."""

    def test_extracts_single_node_into_sub_gm(self) -> None:
        """Lifting a single marker node yields a valid sub-GraphModule."""
        gm = _make_traced_gm_with_markers(num_markers=1)
        markers = find_markers(gm)
        self.assertEqual(len(markers), 1)

        sub_gm = extract_subgraph(gm, [markers[0]])
        self.assertIsInstance(sub_gm, fx.GraphModule)
        # The sub-GraphModule contains the lifted call_function and an output.
        ops = [n.op for n in sub_gm.graph.nodes]
        self.assertIn("call_function", ops)
        self.assertIn("output", ops)
        self.assertIn("placeholder", ops)


class ReplaceRegionTest(unittest.TestCase):
    """recir.replace_region swaps a node-set for a single call."""

    def test_replaces_marker_with_custom_op(self) -> None:
        """A marker node is replaced by a single call_function to ``replacement``."""
        gm = _make_traced_gm_with_markers(num_markers=1)
        markers = find_markers(gm)
        self.assertEqual(len(markers), 1)

        def _replacement(x: torch.Tensor) -> torch.Tensor:
            return x

        replace_region(gm, [markers[0]], _replacement)

        # The original mark_boundary node should be gone.
        remaining_markers = find_markers(gm)
        self.assertEqual(remaining_markers, [])
        # And a new call_function to _replacement should exist.
        replacement_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is _replacement
        ]
        self.assertEqual(len(replacement_nodes), 1)


class AddStateTest(unittest.TestCase):
    """recir.add_state atomically adds a placeholder + flat_state entry."""

    def test_adds_placeholder_and_appends_to_flat_state(self) -> None:
        """The new placeholder index equals len(flat_state) - 1 after append."""
        gm = _make_traced_gm_with_markers(num_markers=1)
        flat_state: List[torch.Tensor] = []
        new_tensor = torch.randn(3, 4)

        before_placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        idx = add_state(gm, flat_state, "my_new_state", new_tensor)

        after_placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        self.assertEqual(len(after_placeholders), len(before_placeholders) + 1)
        # New placeholder is at the end.
        self.assertEqual(
            after_placeholders[-1].op, "placeholder", "new node must be a placeholder"
        )
        # Returned index points to the new flat_state slot.
        self.assertEqual(idx, len(flat_state) - 1)
        self.assertEqual(idx, 0)
        # flat_state now contains the new tensor.
        self.assertIs(flat_state[-1], new_tensor)

    def test_new_placeholder_has_fake_val_meta(self) -> None:
        """The new placeholder has a fake-tensor ``val`` in its meta dict."""
        gm = _make_traced_gm_with_markers(num_markers=1)
        flat_state: List[torch.Tensor] = []
        new_tensor = torch.randn(5, 6)

        add_state(gm, flat_state, "new_state", new_tensor)

        new_placeholder = [n for n in gm.graph.nodes if n.op == "placeholder"][-1]
        self.assertIn("val", new_placeholder.meta)
        val = new_placeholder.meta["val"]
        self.assertEqual(tuple(val.shape), (5, 6))

    def test_returned_index_grows_monotonically(self) -> None:
        """Successive calls return increasing indices: 0, 1, 2, ..."""
        gm = _make_traced_gm_with_markers(num_markers=1)
        flat_state: List[torch.Tensor] = []
        indices = []
        for i in range(3):
            indices.append(add_state(gm, flat_state, f"state_{i}", torch.zeros(2, 2)))
        self.assertEqual(indices, [0, 1, 2])
        self.assertEqual(len(flat_state), 3)


class MakeCompiledRunnerTest(unittest.TestCase):
    def test_prepares_commits_and_materializes_runtime_state(self) -> None:
        class _RuntimeGraph(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.last_iter = False

            def forward(
                self,
                state: torch.Tensor,
                value: torch.Tensor,
                last_iter: bool = False,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                self.last_iter = last_iter
                return value + 1, state + 1

        graph = _RuntimeGraph()
        state = torch.tensor(2)
        value = torch.tensor(5)
        flat_state = [state]
        materialized = object()
        state_manager = MagicMock()
        state_manager.prepare_inputs.return_value = [state, value]
        state_manager.defers_output.return_value = False
        state_manager.materialize_output.return_value = materialized
        compiled_obj = SimpleNamespace(
            gm=graph,
            flat_state=flat_state,
            state_manager=state_manager,
        )

        runner = make_compiled_runner(compiled_obj)
        result = runner(value, last_train_step=True)

        self.assertIs(result, materialized)
        self.assertIs(runner.__dict__[COMPILED_OBJECT_KEY], compiled_obj)
        self.assertTrue(graph.last_iter)
        prepare_args = state_manager.prepare_inputs.call_args.args
        self.assertIs(prepare_args[0], graph)
        self.assertIs(prepare_args[1], flat_state)
        self.assertEqual(len(prepare_args[2]), 1)
        self.assertIs(prepare_args[2][0], value)
        self.assertTrue(prepare_args[3])
        commit_args = state_manager.commit_step_state.call_args.args
        self.assertIs(commit_args[0], flat_state)
        torch.testing.assert_close(commit_args[1][0], state + 1)
        self.assertTrue(commit_args[2])
        state_manager.materialize_output.assert_called_once()
        torch.testing.assert_close(
            state_manager.materialize_output.call_args.args[0], value + 1
        )


class InvalidateDeviceStateTest(unittest.TestCase):
    def test_delegates_to_device_state_manager(self) -> None:
        state_manager = MagicMock()

        def compiled_function() -> None:
            pass

        compiled_function.__dict__[COMPILED_OBJECT_KEY] = SimpleNamespace(
            gm=torch.fx.symbolic_trace(torch.nn.Identity()),
            state_manager=state_manager,
        )

        invalidate_device_state(compiled_function)

        state_manager.invalidate_device_state.assert_called_once_with()


class PrepareCompiledTest(unittest.TestCase):
    def test_compiles_once_without_executing_graph(self) -> None:
        compiled_obj = SimpleNamespace()
        backend = MagicMock()
        parallel_mode = MagicMock()

        with patch.object(
            recir_compilation, "_compile_via_backend", return_value=[compiled_obj]
        ) as compile_backend:
            compiled_function = recir_compile(
                parallel_mode=parallel_mode,
                backend=backend,
            )(lambda value: value)

            self.assertTrue(prepare_compiled(compiled_function, torch.ones(1)))
            self.assertTrue(prepare_compiled(compiled_function, torch.zeros(1)))

        compile_backend.assert_called_once()
        self.assertIs(
            compiled_obj,
            compiled_function.__dict__[COMPILED_OBJECT_KEY],
        )

    def test_returns_false_for_plain_callable(self) -> None:
        self.assertFalse(prepare_compiled(lambda: None))


class MultiFunctionCompileTest(unittest.TestCase):
    def test_clones_stateful_region_transforms_per_function(self) -> None:
        class _StatefulTransform(RegionTransform):
            def __init__(self) -> None:
                self.bound_function: Callable[..., Any] | None = None

            def bind_signature(self, function: Callable[..., Any]) -> None:
                self.bound_function = function

        def train_function(model, optimizer, value):
            return model(value)

        def eval_function(model, value):
            return model(value)

        transform = _StatefulTransform()
        model = nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, foreach=False)
        traced_specs = []

        def trace_graphs(specs):
            traced_specs.extend(specs)
            for spec in specs:
                for graph_transform in spec.region_transforms:
                    graph_transform.bind_signature(spec.fn)
            return [object() for _ in specs]

        compiled = [SimpleNamespace(), SimpleNamespace()]
        with (
            patch.object(recir_compilation, "_trace_graphs", side_effect=trace_graphs),
            patch.object(
                recir_compilation, "_compile_graphs", return_value=compiled
            ) as compile_graphs,
        ):
            train_step, eval_step = recir_compile(
                parallel_mode=MagicMock(),
                backend=MagicMock(),
                region_transforms=[transform],
            )(
                train_function,
                eval_function,
                example_args=[
                    (model, optimizer, torch.ones(1)),
                    (model, torch.ones(1)),
                ],
            )

        self.assertIsNone(transform.bound_function)
        self.assertEqual(len(traced_specs), 2)
        train_transform = traced_specs[0].region_transforms[0]
        eval_transform = traced_specs[1].region_transforms[0]
        self.assertIsNot(train_transform, eval_transform)
        self.assertIs(train_transform.bound_function, train_function)
        self.assertIs(eval_transform.bound_function, eval_function)
        compiled_transforms = compile_graphs.call_args.kwargs["region_transforms"]
        self.assertIs(compiled_transforms[0][0], train_transform)
        self.assertIs(compiled_transforms[1][0], eval_transform)
        self.assertIs(
            train_step.__dict__[COMPILED_OBJECT_KEY],
            compiled[0],
        )
        self.assertIs(
            eval_step.__dict__[COMPILED_OBJECT_KEY],
            compiled[1],
        )


class GetCompiledProcessStateTest(unittest.TestCase):
    def test_returns_canonical_state_after_compilation(self) -> None:
        model_tensor = torch.arange(8).reshape(4, 2)
        optimizer_tensor = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        graph = fx.Graph()
        model_placeholder = graph.placeholder("model_weight")
        optimizer_placeholder = graph.placeholder("optimizer_exp_avg")
        graph.output((model_placeholder, optimizer_placeholder))
        spec = FSDPWeightShardSpec(
            shard_dim=0,
            device_world_size=4,
            process_world_size=2,
        )
        for placeholder in (model_placeholder, optimizer_placeholder):
            placeholder.meta["val"] = torch.empty(2, 2)
            placeholder.meta["is_state_input"] = True
            placeholder.meta["recir_state_layout"] = spec
        model_placeholder.meta["fqn"] = "proj.weight"
        optimizer_placeholder.meta["recir_state_kind"] = "optimizer"
        optimizer_placeholder.meta["recir_state_fqn"] = "proj.weight"
        optimizer_placeholder.meta["recir_state_name"] = "exp_avg"
        process_state = _build_compiled_process_state(
            fx.GraphModule(nn.Module(), graph),
            {"proj.weight": model_tensor},
            {"proj.weight": {"exp_avg": optimizer_tensor}},
        )
        assert process_state is not None

        def compiled_function() -> None:
            pass

        self.assertIsNone(get_compiled_process_state(compiled_function))
        compiled_function.__dict__[COMPILED_OBJECT_KEY] = SimpleNamespace(
            process_state=process_state
        )

        result = get_compiled_process_state(compiled_function)

        self.assertIs(result, process_state)
        assert result is not None
        self.assertIs(result.model["proj.weight"], model_tensor)
        self.assertIs(result.optimizer["proj.weight"]["exp_avg"], optimizer_tensor)
        self.assertEqual((8, 2), result.model_global_shapes["proj.weight"])
        self.assertEqual(
            (8, 2),
            result.optimizer_global_shapes["proj.weight"]["exp_avg"],
        )

    def test_rejects_placeholder_owner_order_mismatch(self) -> None:
        tensor = torch.arange(8).reshape(4, 2)
        graph = fx.Graph()
        placeholder = graph.placeholder("model_weight")
        graph.output(placeholder)
        placeholder.meta["val"] = torch.empty(2, 2)
        placeholder.meta["is_state_input"] = True
        placeholder.meta["fqn"] = "other.weight"
        placeholder.meta["recir_state_layout"] = FSDPWeightShardSpec(
            shard_dim=0,
            device_world_size=4,
            process_world_size=2,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "placeholder metadata does not match",
        ):
            _build_compiled_process_state(
                fx.GraphModule(nn.Module(), graph),
                {"proj.weight": tensor},
                {},
            )


class SetCompiledOutputDeferredTest(unittest.TestCase):
    def test_returns_false_before_compilation(self) -> None:
        def compiled_function() -> None:
            pass

        self.assertFalse(set_compiled_output_deferred(compiled_function, True))

    def test_updates_compiled_device_state_manager(self) -> None:
        state_manager = SimpleNamespace(keep_output_jax=False)

        def compiled_function() -> None:
            pass

        compiled_function.__dict__[COMPILED_OBJECT_KEY] = SimpleNamespace(
            state_manager=state_manager
        )

        self.assertTrue(set_compiled_output_deferred(compiled_function, True))
        self.assertTrue(state_manager.keep_output_jax)
        self.assertFalse(set_compiled_output_deferred(compiled_function, False))
        self.assertFalse(state_manager.keep_output_jax)


class MaterializeCompiledOutputTest(unittest.TestCase):
    def test_passes_through_before_compilation(self) -> None:
        output = object()

        def compiled_function() -> None:
            pass

        self.assertIs(output, materialize_compiled_output(compiled_function, output))

    def test_delegates_to_device_state_manager(self) -> None:
        output = object()
        materialized = object()
        state_manager = SimpleNamespace(
            materialize_output=MagicMock(return_value=materialized)
        )

        def compiled_function() -> None:
            pass

        compiled_function.__dict__[COMPILED_OBJECT_KEY] = SimpleNamespace(
            state_manager=state_manager
        )

        self.assertIs(
            materialized,
            materialize_compiled_output(compiled_function, output),
        )
        state_manager.materialize_output.assert_called_once_with(output)


class TopLevelExportTest(unittest.TestCase):
    """All promoted utilities are reachable from the top-level ``recir`` package."""

    def test_extract_subgraph_exported(self) -> None:
        self.assertIs(recir.extract_subgraph, extract_subgraph)

    def test_replace_region_exported(self) -> None:
        self.assertIs(recir.replace_region, replace_region)

    def test_find_markers_exported(self) -> None:
        self.assertIs(recir.find_markers, find_markers)

    def test_add_state_exported(self) -> None:
        self.assertIs(recir.add_state, add_state)

    def test_set_compiled_output_deferred_exported(self) -> None:
        self.assertIs(
            recir.set_compiled_output_deferred,
            set_compiled_output_deferred,
        )

    def test_materialize_compiled_output_exported(self) -> None:
        self.assertIs(
            recir.materialize_compiled_output,
            materialize_compiled_output,
        )
