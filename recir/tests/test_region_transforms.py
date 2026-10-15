# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import annotations

import unittest
from collections import OrderedDict
from contextlib import ExitStack
from typing import Any, cast

import torch
import torch.fx as fx
import torch.nn as nn
from functorch import make_fx
from recir.region_transforms import (
    _apply_region_transforms,
    _discover_called_modules,
    _extract_subgraph,
    _find_forward_roots,
    _preserve_all_autograd_function_leaves,
    _replace_region,
    RecomputeTransform,
    RegatherTransform,
    RegionTransform,
)
from recir.utils.graph import OP
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.testing._internal.common_utils import run_tests


def _make_fake_mode() -> FakeTensorMode:
    """Create a FakeTensorMode that accepts real (non-fake) parameter tensors."""
    return FakeTensorMode(allow_non_fake_inputs=True)


# ---------------------------------------------------------------------------
# Test models
# ---------------------------------------------------------------------------


class SimpleMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net1 = nn.Linear(50, 32)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(32, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net2(self.relu(self.net1(x))))


class TwoPathModel(nn.Module):
    """Model with two submodules called separately (not via parent forward)."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(16, 16)
        self.decoder = nn.Linear(16, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.encoder(x))
        return self.decoder(h)


class ModelWithUnused(nn.Module):
    """Model with an unused submodule branch."""

    def __init__(self) -> None:
        super().__init__()
        self.used = nn.Linear(10, 10)
        self.unused = nn.Linear(10, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.used(x)


class SharedWeightMLP(nn.Module):
    """Single linear layer reused 3 times in forward."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(32, 32, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.linear(x))
        x = torch.relu(self.linear(x))
        return self.linear(x)


class ModelWithMetaLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.meta_layer = nn.Linear(8, 8, device="meta")
        self.output = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.meta_layer(x))


# ---------------------------------------------------------------------------
# Unit tests: _discover_called_modules
# ---------------------------------------------------------------------------


class TestDiscoverCalledModules(unittest.TestCase):
    def test_simple_model(self) -> None:
        model = SimpleMLP()
        fake_mode = _make_fake_mode()
        x = torch.randn(4, 50)
        called = _discover_called_modules(
            model, fake_mode, lambda m, inp: m(inp), model, x
        )

        fqns = set(called.keys())
        self.assertIn("", fqns)
        self.assertIn("net1", fqns)
        self.assertIn("relu", fqns)
        self.assertIn("net2", fqns)

    def test_unused_submodule_not_discovered(self) -> None:
        model = ModelWithUnused()
        fake_mode = _make_fake_mode()
        x = torch.randn(4, 10)
        called = _discover_called_modules(
            model, fake_mode, lambda m, inp: m(inp), model, x
        )

        fqns = set(called.keys())
        self.assertIn("used", fqns)
        self.assertNotIn("unused", fqns)

    def test_captures_input_shapes(self) -> None:
        model = SimpleMLP()
        fake_mode = _make_fake_mode()
        x = torch.randn(4, 50)
        called = _discover_called_modules(
            model, fake_mode, lambda m, inp: m(inp), model, x
        )

        _, root_args, _root_kwargs = called[""]
        self.assertEqual(len(root_args), 1)
        self.assertEqual(root_args[0].shape, torch.Size([4, 50]))

    def test_ordered_dict_preserves_call_order(self) -> None:
        model = SimpleMLP()
        fake_mode = _make_fake_mode()
        x = torch.randn(4, 50)
        called = _discover_called_modules(
            model, fake_mode, lambda m, inp: m(inp), model, x
        )

        fqns = list(called.keys())
        self.assertEqual(fqns[0], "")

    def test_meta_parameters_use_input_device_during_discovery(self) -> None:
        model = ModelWithMetaLayer()
        fake_mode = _make_fake_mode()
        called = _discover_called_modules(
            model, fake_mode, lambda m, inp: m(inp), model, torch.randn(2, 8)
        )

        self.assertIn("meta_layer", called)
        self.assertIn("output", called)
        self.assertEqual(model.meta_layer.weight.device.type, "meta")


# ---------------------------------------------------------------------------
# Unit tests: _find_forward_roots
# ---------------------------------------------------------------------------


class TestFindForwardRoots(unittest.TestCase):
    def test_standard_usage_single_root(self) -> None:
        """When model.forward() is called, model is the root."""
        model = SimpleMLP()
        fake_mode = _make_fake_mode()
        x = torch.randn(4, 50)
        called = _discover_called_modules(
            model, fake_mode, lambda m, inp: m(inp), model, x
        )
        roots = _find_forward_roots(called)

        self.assertEqual(len(roots), 1)
        fqn, mod, _args, _kwargs = roots[0]
        self.assertEqual(fqn, "")
        self.assertIs(mod, model)

    def test_submodules_are_not_roots(self) -> None:
        """net1, relu, net2 are NOT roots because their parent (model) IS called."""
        model = SimpleMLP()
        fake_mode = _make_fake_mode()
        x = torch.randn(4, 50)
        called = _discover_called_modules(
            model, fake_mode, lambda m, inp: m(inp), model, x
        )
        roots = _find_forward_roots(called)

        root_fqns = {fqn for fqn, _, _, _ in roots}
        self.assertNotIn("net1", root_fqns)
        self.assertNotIn("relu", root_fqns)
        self.assertNotIn("net2", root_fqns)

    def test_manual_sub_model_calls(self) -> None:
        """When encoder and decoder are called directly (not via parent forward),
        both should be forward roots."""
        model = TwoPathModel()
        called: OrderedDict[str, tuple[nn.Module, tuple[Any, ...], dict[str, Any]]] = (
            OrderedDict()
        )
        called["encoder"] = (model.encoder, (torch.randn(4, 16),), {})
        called["decoder"] = (model.decoder, (torch.randn(4, 16),), {})

        roots = _find_forward_roots(called)
        root_fqns = {fqn for fqn, _, _, _ in roots}
        self.assertEqual(root_fqns, {"encoder", "decoder"})

    def test_nested_modules_not_roots_when_parent_called(self) -> None:
        """If parent IS called, children are NOT roots."""
        called: OrderedDict[str, tuple[nn.Module, tuple[Any, ...], dict[str, Any]]] = (
            OrderedDict()
        )
        dummy_mod = nn.Linear(10, 10)
        dummy_args: tuple[Any, ...] = (torch.randn(4, 10),)
        called[""] = (dummy_mod, dummy_args, {})
        called["encoder"] = (dummy_mod, dummy_args, {})
        called["encoder.attn"] = (dummy_mod, dummy_args, {})

        roots = _find_forward_roots(called)
        root_fqns = {fqn for fqn, _, _, _ in roots}
        self.assertEqual(root_fqns, {""})

    def test_container_module_skipped_in_ancestor_walk(self) -> None:
        """Children of container modules (e.g. nn.ModuleList) that don't
        call forward() should NOT become roots if a grandparent IS called.

        Reproduces the STU pattern:
          Model ("") → STUStack ("stu") → ModuleList ("stu.layers")
                                           → Layer ("stu.layers.0")

        ModuleList.forward() is never called (it's just iterated), so
        "stu.layers" is absent from called_modules.  "stu.layers.0"
        must NOT be a root because its grandparent "stu" IS called.
        """
        called: OrderedDict[str, tuple[nn.Module, tuple[Any, ...], dict[str, Any]]] = (
            OrderedDict()
        )
        dummy_mod = nn.Linear(10, 10)
        dummy_args: tuple[Any, ...] = (torch.randn(4, 10),)
        called[""] = (dummy_mod, dummy_args, {})
        called["stu"] = (dummy_mod, dummy_args, {})
        # "stu.layers" is NOT in called — ModuleList never calls forward().
        called["stu.layers.0"] = (dummy_mod, dummy_args, {})

        roots = _find_forward_roots(called)
        root_fqns = {fqn for fqn, _, _, _ in roots}
        self.assertEqual(root_fqns, {""})
        # "stu.layers.0" should NOT be a root because "stu" is called.


# ---------------------------------------------------------------------------
# Unit tests: _extract_subgraph and _replace_region
# ---------------------------------------------------------------------------


class TestExtractSubgraph(unittest.TestCase):
    def _make_simple_graph(self) -> fx.GraphModule:
        """Create a simple graph: x → relu(x) → neg(relu_x) → output."""

        class M(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.neg(torch.relu(x))

        return make_fx(M())(torch.randn(4, 8))

    def test_single_node_extraction(self) -> None:
        gm = self._make_simple_graph()
        relu_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == OP.CALL_FUNCTION and "relu" in str(n.target)
        ]
        self.assertEqual(len(relu_nodes), 1)
        relu_node = relu_nodes[0]

        sub_gm = _extract_subgraph(gm, {relu_node})

        ops = [n.op for n in sub_gm.graph.nodes]
        self.assertEqual(ops.count(OP.PLACEHOLDER), 1)
        self.assertEqual(ops.count(OP.OUTPUT), 1)
        call_fns = [n for n in sub_gm.graph.nodes if n.op == OP.CALL_FUNCTION]
        self.assertEqual(len(call_fns), 1)

    def test_multi_node_extraction(self) -> None:
        """Extract a chain: relu + neg as a region."""
        gm = self._make_simple_graph()
        region = {n for n in gm.graph.nodes if n.op == OP.CALL_FUNCTION}
        self.assertEqual(len(region), 2)

        sub_gm = _extract_subgraph(gm, region)

        ops = [n.op for n in sub_gm.graph.nodes]
        self.assertEqual(ops.count(OP.PLACEHOLDER), 1)
        self.assertEqual(ops.count(OP.OUTPUT), 1)
        call_fns = [n for n in sub_gm.graph.nodes if n.op == OP.CALL_FUNCTION]
        self.assertEqual(len(call_fns), 2)

    def test_extracted_subgraph_runs_correctly(self) -> None:
        gm = self._make_simple_graph()
        relu_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == OP.CALL_FUNCTION and "relu" in str(n.target)
        ]
        sub_gm = _extract_subgraph(gm, set(relu_nodes))

        x = torch.randn(4, 8)
        expected = torch.relu(x)
        result = sub_gm(x)
        torch.testing.assert_close(result, expected)


class TestReplaceRegion(unittest.TestCase):
    def test_single_node_replacement(self) -> None:
        """Replace relu with a custom function, verify graph still works."""

        class M(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.neg(torch.relu(x))

        gm = make_fx(M())(torch.randn(4, 8))

        relu_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == OP.CALL_FUNCTION and "relu" in str(n.target)
        ]
        self.assertEqual(len(relu_nodes), 1)

        def custom_fn(x: torch.Tensor) -> torch.Tensor:
            return torch.abs(x)

        _replace_region(gm, set(relu_nodes), custom_fn)

        x = torch.randn(4, 8)
        expected = torch.neg(torch.abs(x))
        result = gm(x)
        torch.testing.assert_close(result, expected)

    def test_graph_lint_passes_after_replacement(self) -> None:
        class M(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.neg(torch.relu(x))

        gm = make_fx(M())(torch.randn(4, 8))
        relu_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == OP.CALL_FUNCTION and "relu" in str(n.target)
        ]

        def identity_fn(x: torch.Tensor) -> torch.Tensor:
            return x

        _replace_region(gm, set(relu_nodes), identity_fn)

        # Verify graph is valid after replacement.
        gm.graph.lint()
        self.assertTrue(True, "graph.lint() passed after replacement")


# ---------------------------------------------------------------------------
# Unit tests: RegatherTransform
# ---------------------------------------------------------------------------


class TestRegatherTransformDetection(unittest.TestCase):
    def test_select_configured_forward_root(self) -> None:
        model = SimpleMLP()
        fake_mode = _make_fake_mode()
        called = _discover_called_modules(
            model, fake_mode, lambda m, inp: m(inp), model, torch.randn(4, 50)
        )
        transform = RegatherTransform(forward_root_fqns={"net2"})

        self.assertEqual(transform.select_forward_root_fqns(called), {"net2"})

    def test_uncalled_configured_forward_root_raises(self) -> None:
        model = SimpleMLP()
        fake_mode = _make_fake_mode()
        called = _discover_called_modules(
            model, fake_mode, lambda m, inp: m(inp), model, torch.randn(4, 50)
        )
        transform = RegatherTransform(forward_root_fqns={"different.alias"})

        with self.assertRaisesRegex(RuntimeError, "different.alias"):
            transform.select_forward_root_fqns(called)

    def test_transform_graph_wraps_chains(self) -> None:
        """transform_graph wraps each parameter chain with checkpoint."""
        model = SimpleMLP()
        fake_mode = _make_fake_mode()
        x = torch.randn(4, 50)

        from torch.nn.utils import stateless

        params = dict(model.named_parameters(remove_duplicate=False))
        buffers = dict(model.named_buffers(remove_duplicate=False))

        def _stateless_forward(  # pyre-ignore[53]
            params_and_buffers: dict[str, Any], *args: Any
        ) -> Any:
            with stateless._reparametrize_module(model, params_and_buffers):
                return model(*args)

        with fake_mode:
            gm = make_fx(
                _stateless_forward,
                tracing_mode="fake",
                _allow_non_fake_inputs=True,
            )({**params, **buffers}, fake_mode.from_tensor(x))

        called = _discover_called_modules(
            model, fake_mode, lambda m, inp: m(inp), model, x
        )

        num_nodes_before = len(list(gm.graph.nodes))
        transform = RegatherTransform()
        transform.transform_graph(gm, called)
        num_nodes_after = len(list(gm.graph.nodes))

        self.assertNotEqual(num_nodes_before, num_nodes_after)
        # SimpleMLP has two Linear layers. Each layer's weight chain
        # ({t, addmm}) absorbs its bias chain ({addmm}) via the
        # seen_nodes dedup in RegatherTransform.transform_graph, so
        # we expect exactly one wrapper per Linear (target is the
        # `_apply` closure returned by `RegatherTransform._wrap_checkpoint`).
        wrapper_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == OP.CALL_FUNCTION
            and getattr(n.target, "__name__", None) == "_apply"
        ]
        self.assertEqual(len(wrapper_nodes), 2)

    def test_skips_only_ringshift_owned_region(self) -> None:
        class _Schema:
            name: str = "recir::ringshift_embedding"
            is_mutable: bool = False

        def fake_ringshift_embedding(
            ids: torch.Tensor,
            lengths: torch.Tensor,
            weight: torch.Tensor,
        ) -> torch.Tensor:
            del lengths
            return weight.index_select(0, ids.reshape(-1)).reshape(*ids.shape, -1)

        cast(Any, fake_ringshift_embedding)._schema = _Schema()

        class RingAndDenseModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.randn(8, 4))

        model = RingAndDenseModel()
        graph = fx.Graph()
        weight = graph.placeholder("weight")
        weight.meta["val"] = torch.randn(8, 4)
        ids = graph.placeholder("ids")
        ids.meta["val"] = torch.tensor([[0, 1]])
        lengths = graph.placeholder("lengths")
        lengths.meta["val"] = torch.tensor([2], dtype=torch.int32)
        activation = graph.placeholder("activation")
        activation.meta["val"] = torch.randn(2, 8)
        ring = graph.call_function(
            fake_ringshift_embedding,
            args=(ids, lengths, weight),
        )
        ring.meta["val"] = torch.randn(1, 2, 4)
        dense = graph.call_function(
            torch.ops.aten.mm.default, args=(activation, weight)
        )
        dense.meta["val"] = torch.randn(2, 4)
        graph.output((ring, dense))
        gm = fx.GraphModule(model, graph)
        called: OrderedDict[str, tuple[nn.Module, tuple[Any, ...], dict[str, Any]]] = (
            OrderedDict({"": (model, (), {})})
        )

        RegatherTransform().transform_graph(gm, called)

        wrapper_nodes = [
            node
            for node in gm.graph.nodes
            if node.op == OP.CALL_FUNCTION
            and getattr(node.target, "__name__", None) == "_apply"
        ]
        ring_nodes = [
            node
            for node in gm.graph.nodes
            if node.op == OP.CALL_FUNCTION
            and getattr(getattr(node.target, "_schema", None), "name", None)
            == "recir::ringshift_embedding"
        ]
        self.assertEqual(len(wrapper_nodes), 1)
        self.assertEqual(len(ring_nodes), 1)


# ---------------------------------------------------------------------------
# Unit tests: RecomputeTransform
# ---------------------------------------------------------------------------


class TestRecomputeTransformDetection(unittest.TestCase):
    def test_transform_graph_wraps_submodule(self) -> None:
        model = SimpleMLP()
        fake_mode = _make_fake_mode()

        # nn_module_stack is only populated when make_fx runs with
        # ``record_module_stack=True`` AND the traced callable exposes
        # ``_orig_mod`` for ``_ModuleStackTracer`` to scope from.  This
        # matches what ``_apply_region_transforms`` does internally.
        traced_fn = lambda *args: model(*args)  # noqa: E731
        traced_fn._orig_mod = model  # type: ignore[attr-defined]
        with fake_mode:
            gm = make_fx(
                traced_fn,
                tracing_mode="fake",
                _allow_non_fake_inputs=True,
                record_module_stack=True,
            )(fake_mode.from_tensor(torch.randn(4, 50)))

        transform = RecomputeTransform(submodule_fqns={"net1"})
        called: OrderedDict[str, tuple[nn.Module, tuple[Any, ...], dict[str, Any]]] = (
            OrderedDict()
        )
        num_nodes_before = len(list(gm.graph.nodes))
        transform.transform_graph(gm, called)
        num_nodes_after = len(list(gm.graph.nodes))

        # net1's call_function nodes should collapse into a single
        # checkpoint wrapper call (target is the `_apply` closure
        # returned by `RecomputeTransform._wrap_checkpoint`).
        self.assertLess(num_nodes_after, num_nodes_before)
        wrapper_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == OP.CALL_FUNCTION
            and getattr(n.target, "__name__", None) == "_apply"
        ]
        self.assertEqual(len(wrapper_nodes), 1)

    def test_nonexistent_submodule_is_noop(self) -> None:
        model = SimpleMLP()
        fake_mode = _make_fake_mode()

        with fake_mode:
            gm = make_fx(
                lambda *args: model(*args),
                tracing_mode="fake",
                _allow_non_fake_inputs=True,
            )(fake_mode.from_tensor(torch.randn(4, 50)))

        transform = RecomputeTransform(submodule_fqns={"nonexistent"})
        called: OrderedDict[str, tuple[nn.Module, tuple[Any, ...], dict[str, Any]]] = (
            OrderedDict()
        )
        num_nodes_before = len(list(gm.graph.nodes))
        transform.transform_graph(gm, called)
        num_nodes_after = len(list(gm.graph.nodes))
        self.assertEqual(num_nodes_before, num_nodes_after)


# ---------------------------------------------------------------------------
# Unit tests: _apply_region_transforms end-to-end (no distributed)
# ---------------------------------------------------------------------------


class TestApplyRegionTransforms(unittest.TestCase):
    def test_identity_transform_preserves_output(self) -> None:
        """A transform that wraps regions but doesn't change behavior should
        produce the same output."""

        class IdentityTransform(RegionTransform):
            def transform_graph(
                self,
                gm: fx.GraphModule,
                called_modules: OrderedDict[
                    str, tuple[nn.Module, tuple[Any, ...], dict[str, Any]]
                ],
            ) -> None:
                for n in list(gm.graph.nodes):
                    if n.op == OP.CALL_FUNCTION:
                        region_gm = _extract_subgraph(gm, {n})

                        def wrapped(*args: Any, _gm: Any = region_gm) -> Any:
                            return _gm(*args)

                        _replace_region(gm, {n}, wrapped)

        model = SimpleMLP()
        x = torch.randn(4, 50)
        expected = model(x)

        fake_mode = _make_fake_mode()
        called = _discover_called_modules(
            model, fake_mode, lambda m, inp: m(inp), model, x
        )
        roots = _find_forward_roots(called)

        with ExitStack() as stack:
            for _fqn, root_mod, fake_args, fake_kwargs in roots:
                stack.enter_context(
                    _apply_region_transforms(
                        root_mod,
                        fake_args,
                        called,
                        [IdentityTransform()],
                        fake_mode,
                        fake_kwargs=fake_kwargs,
                    )
                )
            result = model(x)
        torch.testing.assert_close(result, expected)

    def test_forward_restored_after_context_exit(self) -> None:
        """After the context manager exits, the original forward is restored."""
        model = SimpleMLP()
        fake_mode = _make_fake_mode()
        x = torch.randn(4, 50)
        called = _discover_called_modules(
            model, fake_mode, lambda m, inp: m(inp), model, x
        )
        roots = _find_forward_roots(called)

        class DummyTransform(RegionTransform):
            def transform_graph(
                self,
                gm: fx.GraphModule,
                called_modules: OrderedDict[
                    str, tuple[nn.Module, tuple[Any, ...], dict[str, Any]]
                ],
            ) -> None:
                pass

        original_forward_fn = type(model).forward
        original_forward = model.forward

        with ExitStack() as stack:
            for _fqn, root_mod, fake_args, fake_kwargs in roots:
                stack.enter_context(
                    _apply_region_transforms(
                        root_mod,
                        fake_args,
                        called,
                        [DummyTransform()],
                        fake_mode,
                        fake_kwargs=fake_kwargs,
                    )
                )
            self.assertNotEqual(model.forward, original_forward)

        # After exit, the patched forward must be removed and the original
        # class method must be reachable again.
        self.assertIs(
            model.forward.__func__,  # pyre-fixme[16]
            original_forward_fn,
        )
        result = model(x)
        self.assertEqual(result.shape, torch.Size([4, 8]))

    def test_empty_transforms_is_noop(self) -> None:
        """Passing empty transforms list should not modify the module."""
        model = SimpleMLP()
        original_forward_fn = type(model).forward
        fake_mode = _make_fake_mode()
        called = _discover_called_modules(
            model, fake_mode, lambda m, inp: m(inp), model, torch.randn(4, 50)
        )
        roots = _find_forward_roots(called)

        with ExitStack() as stack:
            for _fqn, root_mod, fake_args, fake_kwargs in roots:
                stack.enter_context(
                    _apply_region_transforms(
                        root_mod,
                        fake_args,
                        called,
                        [],
                        fake_mode,
                        fake_kwargs=fake_kwargs,
                    )
                )
            self.assertIs(
                model.forward.__func__,  # pyre-fixme[16]
                original_forward_fn,
            )


# ---------------------------------------------------------------------------
# Unit tests: RegatherTransform._wrap_checkpoint
# ---------------------------------------------------------------------------


class TestRegatherTransformCheckpoint(unittest.TestCase):
    def test_checkpoint_produces_correct_forward(self) -> None:
        """Checkpoint-wrapped region should produce correct output."""
        g = fx.Graph()
        ph = g.placeholder("x")
        g.output(ph)
        identity_gm = fx.GraphModule(nn.Module(), g)

        wrapped_fn = RegatherTransform._wrap_checkpoint(identity_gm)

        x = torch.randn(4, 8, requires_grad=True)
        result = wrapped_fn(x)
        self.assertEqual(result.shape, x.shape)

    def test_checkpoint_backward_produces_gradient(self) -> None:
        """Checkpoint-wrapped region should produce valid gradients in backward."""
        g = fx.Graph()
        ph = g.placeholder("x")
        mul_node = g.call_function(torch.mul, (ph, 2.0))
        g.output(mul_node)
        scale_gm = fx.GraphModule(nn.Module(), g)

        wrapped_fn = RegatherTransform._wrap_checkpoint(scale_gm)

        x = torch.randn(4, 8, requires_grad=True)
        result = wrapped_fn(x)
        loss = result.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        torch.testing.assert_close(x.grad, torch.full_like(x, 2.0))


# ---------------------------------------------------------------------------
# Unit tests: RecomputeTransform._wrap_checkpoint
# ---------------------------------------------------------------------------


class TestRecomputeTransformCheckpoint(unittest.TestCase):
    def test_recompute_forward_correct(self) -> None:
        g = fx.Graph()
        ph = g.placeholder("x")
        relu_node = g.call_function(torch.relu, (ph,))
        g.output(relu_node)
        relu_gm = fx.GraphModule(nn.Module(), g)

        wrapped_fn = RecomputeTransform._wrap_checkpoint(relu_gm)

        x = torch.randn(4, 8)
        expected = torch.relu(x)
        result = wrapped_fn(x)
        torch.testing.assert_close(result, expected)

    def test_recompute_backward_correct(self) -> None:
        g = fx.Graph()
        ph = g.placeholder("x")
        relu_node = g.call_function(torch.relu, (ph,))
        g.output(relu_node)
        relu_gm = fx.GraphModule(nn.Module(), g)

        wrapped_fn = RecomputeTransform._wrap_checkpoint(relu_gm)

        x = torch.randn(4, 8, requires_grad=True)
        result = wrapped_fn(x)
        loss = result.sum()
        loss.backward()

        x2 = x.detach().clone().requires_grad_(True)
        torch.relu(x2).sum().backward()

        self.assertIsNotNone(x.grad)
        torch.testing.assert_close(x.grad, x2.grad)


# ---------------------------------------------------------------------------
# Unit tests: Transform ordering / composability
# ---------------------------------------------------------------------------


class TestTransformOrdering(unittest.TestCase):
    def test_sequential_application_wraps_correctly(self) -> None:
        """Applying two transforms sequentially: the second transform should
        see already-wrapped nodes from the first."""

        call_order: list[str] = []

        class TrackingTransformA(RegionTransform):
            def transform_graph(
                self,
                gm: fx.GraphModule,
                called_modules: OrderedDict[
                    str, tuple[nn.Module, tuple[Any, ...], dict[str, Any]]
                ],
            ) -> None:
                call_order.append("transform_A")

        class TrackingTransformB(RegionTransform):
            def transform_graph(
                self,
                gm: fx.GraphModule,
                called_modules: OrderedDict[
                    str, tuple[nn.Module, tuple[Any, ...], dict[str, Any]]
                ],
            ) -> None:
                call_order.append("transform_B")

        model = SimpleMLP()
        fake_mode = _make_fake_mode()
        x = torch.randn(4, 50)
        called = _discover_called_modules(
            model, fake_mode, lambda m, inp: m(inp), model, x
        )
        roots = _find_forward_roots(called)

        with ExitStack() as stack:
            for _fqn, root_mod, fake_args, fake_kwargs in roots:
                stack.enter_context(
                    _apply_region_transforms(
                        root_mod,
                        fake_args,
                        called,
                        [TrackingTransformA(), TrackingTransformB()],
                        fake_mode,
                        fake_kwargs=fake_kwargs,
                    )
                )

        self.assertEqual(call_order, ["transform_A", "transform_B"])


# ---------------------------------------------------------------------------
# Custom autograd.Function for testing leaf preservation
# ---------------------------------------------------------------------------


class ScaleAndShiftFunction(torch.autograd.Function):
    """Custom autograd.Function with asymmetric forward/backward.

    Forward: out = input * scale  (trivial multiply)
    Backward: grad_input = grad_output * (scale + shift)

    The backward multiplies by (scale + shift), NOT scale.  This asymmetry
    is the key test: if make_fx decomposes forward() into aten.mul, the
    backward becomes aten.mul's standard backward (multiply by scale), and
    the +shift term is lost.
    """

    @staticmethod
    def forward(  # pyre-ignore[14]
        ctx: Any,
        input: torch.Tensor,
        scale: float,
        shift: float,
    ) -> torch.Tensor:
        ctx.scale = scale
        ctx.shift = shift
        return input * scale

    @staticmethod
    def backward(  # pyre-fixme[14]
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None]:
        return grad_output * (ctx.scale + ctx.shift), None, None


class ModelWithCustomAF(nn.Module):
    """Model that uses ScaleAndShiftFunction in its forward."""

    def __init__(self, scale: float = 2.0, shift: float = 0.5) -> None:
        super().__init__()
        self.linear = nn.Linear(8, 4)
        self.scale = scale
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)
        return ScaleAndShiftFunction.apply(h, self.scale, self.shift)


@torch.library.custom_op("recir_test::region_transform_scale", mutates_args=())
def _region_transform_scale(x: torch.Tensor) -> torch.Tensor:
    return x * 2


@_region_transform_scale.register_fake
def _region_transform_scale_fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


def _region_transform_scale_backward(
    ctx: Any,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    return grad_output * 2


_region_transform_scale.register_autograd(_region_transform_scale_backward)


# ---------------------------------------------------------------------------
# Unit tests: _preserve_all_autograd_function_leaves
# ---------------------------------------------------------------------------


class TestPreserveAutogradFunctionLeaves(unittest.TestCase):
    def test_torch_library_op_uses_registered_fake_and_backward(self) -> None:
        fake_mode = _make_fake_mode()
        x = torch.randn(4, 8, requires_grad=True)

        with fake_mode, _preserve_all_autograd_function_leaves():
            gm = make_fx(
                _region_transform_scale,
                tracing_mode="fake",
                _allow_non_fake_inputs=True,
            )(fake_mode.from_tensor(x))

        self.assertIn(
            torch.ops.recir_test.region_transform_scale.default,
            [node.target for node in gm.graph.nodes],
        )
        traced_input = torch.randn(4, 8, requires_grad=True)
        output = gm(traced_input)
        output.sum().backward()
        torch.testing.assert_close(output, traced_input.detach() * 2)
        torch.testing.assert_close(
            traced_input.grad,
            torch.full_like(traced_input, 2),
        )

    def test_leaf_produces_single_node(self) -> None:
        """With leaf preservation, the AF appears as a single call_function
        node in the make_fx graph, not decomposed into individual aten ops."""
        model = ModelWithCustomAF(scale=2.0, shift=0.5)
        fake_mode = _make_fake_mode()
        x = torch.randn(4, 8)

        # Trace WITHOUT leaf preservation — AF is decomposed.
        with fake_mode:
            gm_decomposed = make_fx(
                model,
                tracing_mode="fake",
                _allow_non_fake_inputs=True,
            )(fake_mode.from_tensor(x))

        # Count mul nodes — the ScaleAndShiftFunction.forward() does input*scale
        # which decomposes to aten.mul. There may be other muls from nn.Linear.
        mul_count_decomposed = sum(
            1
            for n in gm_decomposed.graph.nodes
            if n.op == OP.CALL_FUNCTION and "mul" in str(n.target)
        )

        # Trace WITH leaf preservation.
        fake_mode2 = _make_fake_mode()
        with fake_mode2, _preserve_all_autograd_function_leaves():
            gm_preserved = make_fx(
                model,
                tracing_mode="fake",
                _allow_non_fake_inputs=True,
            )(fake_mode2.from_tensor(x))

        # The preserved graph should have fewer mul nodes (the one from
        # ScaleAndShiftFunction is now an opaque node, not a mul).
        mul_count_preserved = sum(
            1
            for n in gm_preserved.graph.nodes
            if n.op == OP.CALL_FUNCTION and "mul" in str(n.target)
        )

        self.assertLess(
            mul_count_preserved,
            mul_count_decomposed,
            "Leaf preservation should reduce the number of decomposed mul ops",
        )

        # Check there's a call_function node whose target is the bound
        # wrapper for ScaleAndShiftFunction.apply (not a standard aten op).
        af_nodes = [
            n
            for n in gm_preserved.graph.nodes
            if n.op == OP.CALL_FUNCTION
            and hasattr(n.target, "__name__")
            and n.target.__name__ == "_bound_apply"
        ]
        self.assertGreaterEqual(
            len(af_nodes),
            1,
            "Should have at least one opaque ScaleAndShiftFunction node",
        )

    def test_noop_outside_tracing(self) -> None:
        """Outside make_fx, the patched apply behaves identically to original."""
        model = ModelWithCustomAF(scale=3.0, shift=1.0)
        x = torch.randn(4, 8)
        expected = model(x)

        with _preserve_all_autograd_function_leaves():
            result = model(x)

        torch.testing.assert_close(result, expected)

    def test_backward_preserved_through_make_fx(self) -> None:
        """The custom backward (scale+shift) is preserved when the AF is
        treated as a leaf, whereas decomposed tracing loses the shift term."""
        scale, shift = 2.0, 0.5

        # Direct execution — ground truth backward.
        x_ref = torch.randn(4, 8, requires_grad=True)
        out_ref = ScaleAndShiftFunction.apply(x_ref, scale, shift)
        out_ref.sum().backward()
        grad_ref = x_ref.grad
        assert grad_ref is not None

        # Every element should be (scale + shift) = 2.5
        torch.testing.assert_close(grad_ref, torch.full_like(x_ref, scale + shift))

        # Trace WITH leaf preservation, execute, check backward.
        fake_mode = _make_fake_mode()

        def fn(x: torch.Tensor) -> torch.Tensor:  # pyre-fixme[53]
            return ScaleAndShiftFunction.apply(x, scale, shift)

        with fake_mode, _preserve_all_autograd_function_leaves():
            gm = make_fx(
                fn,
                tracing_mode="fake",
                _allow_non_fake_inputs=True,
            )(fake_mode.from_tensor(torch.randn(4, 8)))

        # Execute the traced graph with real tensors.
        x_traced = torch.randn(4, 8, requires_grad=True)
        out_traced = gm(x_traced)
        out_traced.sum().backward()
        grad_traced = x_traced.grad
        assert grad_traced is not None

        # The preserved backward should give (scale + shift) = 2.5,
        # NOT just scale = 2.0 (which is what decomposed tracing gives).
        torch.testing.assert_close(
            grad_traced, torch.full_like(x_traced, scale + shift)
        )

    def test_context_manager_restores_apply(self) -> None:
        """After exiting the context, the base Function.apply is restored."""
        original = torch.autograd.Function.__dict__["apply"]

        with _preserve_all_autograd_function_leaves():
            pass

        self.assertIs(torch.autograd.Function.__dict__["apply"], original)


# ---------------------------------------------------------------------------
# Reproduce: AF whose forward() calls a submodule
# ---------------------------------------------------------------------------


_INNER_LINEAR = nn.Linear(8, 4)


class SubmoduleCallingFunction(torch.autograd.Function):
    """AF whose forward() calls an nn.Module — reproduces the STU failure."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:  # pyre-fixme[14]
        return _INNER_LINEAR(x)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor]:  # pyre-fixme[14]
        return (grad,)


class TestLeafApplyWithSubmoduleCall(unittest.TestCase):
    def test_leaf_apply_submodule_call(self) -> None:
        """Reproduces the STU failure: AF forward calls a submodule."""
        fake_mode = _make_fake_mode()

        def fn(x: torch.Tensor) -> torch.Tensor:
            return SubmoduleCallingFunction.apply(x)

        with fake_mode, _preserve_all_autograd_function_leaves():
            gm = make_fx(
                fn,
                tracing_mode="fake",
                _allow_non_fake_inputs=True,
            )(fake_mode.from_tensor(torch.randn(4, 8)))

        x = torch.randn(4, 8)
        result = gm(x)
        self.assertEqual(result.shape, torch.Size([4, 4]))


# ---------------------------------------------------------------------------
# Integration test: _apply_region_transforms with autograd function leaves
# ---------------------------------------------------------------------------


class TestApplyRegionTransformsWithLeaves(unittest.TestCase):
    def test_leaf_af_backward_through_region_transforms(self) -> None:
        """End-to-end: region transforms + autograd function leaves produce
        correct backward gradients."""
        model = ModelWithCustomAF(scale=2.0, shift=0.5)
        x = torch.randn(4, 8)
        fake_mode = _make_fake_mode()

        called = _discover_called_modules(
            model, fake_mode, lambda m, inp: m(inp), model, x
        )
        roots = _find_forward_roots(called)

        class IdentityTransform(RegionTransform):
            def transform_graph(
                self,
                gm: fx.GraphModule,
                called_modules: OrderedDict[
                    str, tuple[nn.Module, tuple[Any, ...], dict[str, Any]]
                ],
            ) -> None:
                pass

        with ExitStack() as stack:
            for _fqn, root_mod, fake_args, fake_kwargs in roots:
                stack.enter_context(
                    _apply_region_transforms(
                        root_mod,
                        fake_args,
                        called,
                        [IdentityTransform()],
                        fake_mode,
                        fake_kwargs=fake_kwargs,
                    )
                )
            # Inside the context, run forward+backward.
            x_test = torch.randn(4, 8, requires_grad=True)
            out = model(x_test)
            out.sum().backward()

        grad = x_test.grad
        assert grad is not None
        # The custom backward multiplies by (scale + shift) = 2.5 for the
        # ScaleAndShiftFunction part. The full gradient also flows through
        # nn.Linear, so we verify the gradient is non-zero and has correct
        # shape rather than checking exact values.
        self.assertEqual(grad.shape, x_test.shape)
        self.assertFalse(torch.all(grad == 0))


if __name__ == "__main__":
    run_tests()
