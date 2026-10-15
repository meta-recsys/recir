# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Tests for ``segment_and_lower`` submodule-attribute registration.

Background
==========

``recir.lowering.lowering_utils.lower_subgraph_nodes`` splices a
``call_module(target="<subgraph_prefix>_<idx>")`` node into the parent
``fx.Graph`` and is expected to install the lowered subgraph as an
attribute on the parent ``nn.Module`` so the call_module target
resolves.

For a plain ``fx.GraphModule`` parent a single ``setattr(gm, name,
module)`` does both jobs: the attribute is installed on the same
``nn.Module`` whose ``fx.Graph`` holds the call_module node, and
``gm.graph.lint`` resolves the target via
``getattr(gm.owning_module, name)``.

For an ``IterGraphModule`` parent, however, ``setattr(iter_gm, name,
module)`` lands on the outer ``nn.Module`` wrapper *only*. The
``call_module`` node lives in ``iter_gm.main_gm.graph`` (an
``IterGraph`` whose ``owning_module`` is the inner ``main_gm``).
``IterGraph.lint`` then walks ``main_gm`` for the attribute, doesn't
find it, and raises::

    RuntimeError: Node <name> target <name> references nonexistent
    attribute <name>

(at ``recir/passes/iter_graph_module.py:493`` -- the ``super().lint()`` call).

Commit 17 adds ``IterGraphModule.add_subgraph_module`` which
propagates the attribute to all three sub-GMs (setup / main /
cleanup); ``lower_subgraph_nodes`` invokes it when the parent is an
``IterGraphModule`` and falls back to the bare ``setattr`` for plain
``fx.GraphModule`` parents. The tests below assert both arms:
``gm.graph.lint`` passes post-lower and ``getattr(gm, name)`` returns
the lowered submodule.
"""

import operator
import unittest
from typing import List
from unittest import mock

import torch
from recir.lowering.lowering_utils import create_subgraph_module, segment_and_lower
from recir.lowering.partial_lower import (
    _prune_control_deps_to_frontier,
    _shape_env_for_subgraph,
    _StreamSyncOrderingGraphPass,
    partial_lower,
)
from recir.passes.iter_graph_module import IterGraphModule
from torch._inductor.fx_passes.control_dependencies import control_deps
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv


def _always_compatible(node: torch.fx.Node) -> tuple[bool, str]:
    """Compatibility fn used in tests: every node is compatible."""
    return True, ""


def _make_relu_chain_gm(num_ops: int) -> torch.fx.GraphModule:
    """Build a ``fx.GraphModule`` with a single chain of ``num_ops`` reluops.

    The chain is long enough (default 12 > ``MIN_ATEN_OPS_TO_LOWER``) for
    ``segment_and_lower`` to coalesce the whole body into one subgraph
    without needing a custom subgraph_predicate.
    """
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    last = x
    for _ in range(num_ops):
        last = graph.call_function(torch.ops.aten.relu.default, args=(last,))
    graph.output(last)
    return torch.fx.GraphModule(root={}, graph=graph)


class _StubSubgraphModule(torch.nn.Module):
    """Minimal ``nn.Module`` wrapper used as the ``module_factory`` output.

    The lowered subgraph itself is not executed by these tests; we only
    care about the attribute-registration plumbing, so a stub that holds
    a reference to the extracted ``fx.GraphModule`` is sufficient.
    """

    def __init__(self, sub_gm: torch.fx.GraphModule) -> None:
        super().__init__()
        self.sub_gm = sub_gm


class SegmentAndLowerAttributesTest(unittest.TestCase):
    def test_stream_sync_passthroughs_become_ordering_edges(self) -> None:
        from torch._functorch._aot_autograd.streams import (
            wrap_all_sync_nodes_with_control_deps,
        )

        graph = torch.fx.Graph()
        value = graph.placeholder("value")
        value.meta["val"] = torch.ones(2)
        side = graph.call_function(torch.ops.aten.relu.default, args=(value,))
        side.meta.update({"val": torch.ones(2), "custom": {"stream": 5}})
        graph.call_function(
            torch.ops.streams.record_event.default,
            args=(11, 5),
        )
        other = graph.call_function(torch.ops.aten.neg.default, args=(value,))
        other.meta.update({"val": torch.ones(2), "custom": {"stream": 7}})
        graph.call_function(
            torch.ops.streams.wait_event.default,
            args=(11, 7),
        )
        consumer = graph.call_function(
            torch.ops.aten.add.Tensor,
            args=(side, other),
        )
        consumer.meta.update({"val": torch.ones(2), "custom": {"stream": 7}})
        graph.output(consumer)
        gm = torch.fx.GraphModule(root={}, graph=graph)

        wrap_all_sync_nodes_with_control_deps(gm)

        control_nodes = [
            node
            for node in gm.graph.nodes
            if node.op == "call_function" and node.target is control_deps
        ]
        self.assertEqual(len(control_nodes), 2)
        self.assertTrue(all(len(node.args) > 2 for node in control_nodes))
        wait_node = control_nodes[1]
        subgraph_attr = wait_node.args[1]
        if not isinstance(subgraph_attr, torch.fx.Node) or not isinstance(
            subgraph_attr.target, str
        ):
            raise AssertionError("expected control-deps subgraph attribute")
        wait_subgraph = getattr(gm, subgraph_attr.target)
        self.assertTrue(
            any(
                subgraph_node.op == "call_function"
                and subgraph_node.target is torch.ops.streams.wait_event.default
                for subgraph_node in wait_subgraph.graph.nodes
            )
        )
        _StreamSyncOrderingGraphPass()(gm.graph)

        self.assertEqual([len(node.args) for node in control_nodes], [2, 2])
        self.assertFalse(
            any(
                node.op == "call_function" and node.target is operator.getitem
                for node in gm.graph.nodes
            )
        )
        ordered_consumer = gm.graph.output_node().args[0]
        if not isinstance(ordered_consumer, torch.fx.Node):
            raise AssertionError("expected FX node output")
        self.assertIs(ordered_consumer.target, control_deps)
        self.assertEqual(ordered_consumer.args[0], (wait_node,))
        gm.graph.lint()

    def test_control_deps_are_pruned_to_dependency_frontier(self) -> None:
        graph = torch.fx.Graph()
        value = graph.placeholder("value")
        first = graph.call_function(torch.ops.aten.relu.default, args=(value,))
        middle = graph.call_function(torch.ops.aten.neg.default, args=(first,))
        second = graph.call_function(torch.ops.aten.relu.default, args=(middle,))
        independent = graph.call_function(torch.ops.aten.neg.default, args=(value,))
        barrier = graph.call_function(
            control_deps,
            args=((value, first, second, independent), None),
        )
        graph.output(barrier)

        _prune_control_deps_to_frontier(graph)

        self.assertEqual(barrier.args[0], (second, independent))

    def test_partial_lower_keeps_stream_syncs_inside_one_subgraph(self) -> None:
        graph = torch.fx.Graph()
        value = graph.placeholder("value")
        value.meta["val"] = torch.ones(2)
        current = value
        for _ in range(12):
            current = graph.call_function(
                torch.ops.aten.relu.default,
                args=(current,),
            )
            current.meta["val"] = torch.ones(2)
        collective = graph.call_function(
            torch.ops._c10d_functional.all_reduce.default,
            args=(current, "sum", "0"),
        )
        collective.meta["val"] = torch.ones(2)
        current = graph.call_function(
            torch.ops._c10d_functional.wait_tensor.default,
            args=(collective,),
        )
        current.meta["val"] = torch.ones(2)
        graph.call_function(
            torch.ops.streams.record_event.default,
            args=(11, 5),
        )
        graph.call_function(
            torch.ops.streams.wait_event.default,
            args=(11, 7),
        )
        for _ in range(12):
            current = graph.call_function(
                torch.ops.aten.relu.default,
                args=(current,),
            )
            current.meta["val"] = torch.ones(2)
        graph.output(current)
        gm = torch.fx.GraphModule(root={}, graph=graph)

        lowered = partial_lower(gm)
        calls = [node for node in lowered.graph.nodes if node.op == "call_module"]
        self.assertEqual(len(calls), 1)

        compiled_graphs: list[torch.fx.GraphModule] = []

        def compile_fx(
            compiled_gm: torch.fx.GraphModule,
            _inputs: tuple[object, ...],
            **_kwargs: object,
        ) -> object:
            compiled_graphs.append(compiled_gm)
            return lambda *args: (args[0],)

        with mock.patch("torch._inductor.compile_fx.compile_fx", compile_fx):
            result = lowered(torch.ones(2))

        self.assertTrue(torch.equal(result, torch.ones(2)))
        self.assertEqual(len(compiled_graphs), 1)
        control_nodes = [
            node
            for node in compiled_graphs[0].graph.nodes
            if node.op == "call_function" and node.target is control_deps
        ]
        self.assertEqual([len(node.args[0]) for node in control_nodes], [1, 1])
        collective_node = next(
            node
            for node in compiled_graphs[0].graph.nodes
            if node.target is torch.ops._c10d_functional.all_reduce.default
        )
        self.assertEqual(
            [user.target for user in collective_node.users],
            [torch.ops._c10d_functional.wait_tensor.default],
        )

    def test_shape_env_accepts_metadata_from_multiple_fake_modes(self) -> None:
        graph = torch.fx.Graph()
        value = graph.placeholder("value")
        output = graph.output(value)
        first_mode = FakeTensorMode(shape_env=ShapeEnv())
        second_mode = FakeTensorMode(shape_env=ShapeEnv())
        output.meta["val"] = (
            first_mode.from_tensor(torch.ones(2)),
            second_mode.from_tensor(torch.ones(2)),
        )
        gm = torch.fx.GraphModule(root={}, graph=graph)

        self.assertIsInstance(_shape_env_for_subgraph(gm), ShapeEnv)

    def test_subgraph_output_shares_boundary_metadata(self) -> None:
        graph = torch.fx.Graph()
        source = graph.placeholder("source")
        output = graph.call_function(torch.ops.aten.relu.default, args=(source,))
        output.meta["placement"] = "unresolved"
        graph.output(output)

        subgraph = create_subgraph_module([source], [output], [output])
        copied_output = next(
            node for node in subgraph.graph.nodes if node.op == "call_function"
        )

        self.assertIs(copied_output.meta, output.meta)
        copied_output.meta["placement"] = "fsdp_shard_0"
        self.assertEqual(output.meta["placement"], "fsdp_shard_0")

    def test_plain_graph_module_lint_passes(self) -> None:
        """Baseline: bare ``fx.GraphModule`` parent still works after the fix.

        Confirms the non-IterGraphModule path (``setattr`` directly on
        ``gm``) is preserved -- the fallback arm of
        ``lower_subgraph_nodes`` should not regress.
        """
        gm = _make_relu_chain_gm(num_ops=12)
        lowered = segment_and_lower(
            gm,
            compatibility_fn=_always_compatible,
            module_factory=_StubSubgraphModule,
            subgraph_prefix="xla_subgraph",
        )

        # The call_module node was added; the corresponding submodule
        # attribute must exist on the parent gm.
        call_nodes: List[torch.fx.Node] = [
            n for n in lowered.graph.nodes if n.op == "call_module"
        ]
        self.assertEqual(
            len(call_nodes), 1, "expected exactly one call_module after lowering"
        )
        self.assertEqual(call_nodes[0].target, "xla_subgraph_0")

        # Attribute is gettable on the parent.
        sub = getattr(lowered, "xla_subgraph_0", None)
        self.assertIsNotNone(sub, "xla_subgraph_0 attribute missing on parent gm")
        self.assertIsInstance(sub, _StubSubgraphModule)
        self.assertEqual(sub.final_output_indices, {0})

        # And lint passes -- the regression we're guarding against.
        lowered.graph.lint()

    def test_iter_graph_module_lint_passes_post_lower(self) -> None:
        """Regression: ``IterGraphModule`` parent passes lint after lowering.

        Before commit 17, ``segment_and_lower`` only invoked
        ``setattr(iter_gm, "xla_subgraph_0", module)``, which installed
        the submodule on the outer wrapper but NOT on
        ``iter_gm.main_gm`` (the ``owning_module`` of the IterGraph
        whose call_module node we just inserted). ``IterGraph.lint`` ->
        ``super().lint()`` -> ``fx.Graph.lint`` then raised::

            RuntimeError: Node xla_subgraph_0 target xla_subgraph_0
            references nonexistent attribute xla_subgraph_0

        After commit 17, the attribute is propagated to all three
        sub-GMs via ``IterGraphModule.add_subgraph_module``, so lint
        passes.
        """
        gm = _make_relu_chain_gm(num_ops=12)
        iter_gm = IterGraphModule(gm)

        # Before the fix this raised inside segment_and_lower's final
        # ``gm.graph.lint()`` call (IterGraph.lint walks all three).
        lowered = segment_and_lower(
            iter_gm,  # pyre-ignore[6] -- IterGraphModule shadows fx.GraphModule shape
            compatibility_fn=_always_compatible,
            module_factory=_StubSubgraphModule,
            subgraph_prefix="xla_subgraph",
        )
        # Sanity: the returned object is the same IterGraphModule we passed in.
        self.assertIs(lowered, iter_gm)

        # The call_module was inserted into the *main* IterGraph only
        # (IterGraph.create_node is not overridden to mirror).
        main_call_nodes: List[torch.fx.Node] = [
            n for n in iter_gm.main_gm.graph.nodes if n.op == "call_module"
        ]
        self.assertEqual(
            len(main_call_nodes),
            1,
            "expected exactly one call_module in main_gm after lowering",
        )
        self.assertEqual(main_call_nodes[0].target, "xla_subgraph_0")

        # Re-lint to defend against future regressions in IterGraph.lint.
        iter_gm.main_gm.graph.lint()

    def test_iter_graph_module_attribute_propagates_to_all_sub_gms(self) -> None:
        """``add_subgraph_module`` installs the attribute on every sub-GM.

        The call_module node currently lives only in ``main_gm``, but
        installing on all three keeps the sub-GM submodule registries
        consistent and future-proofs any pass that mirrors the
        call_module into ``setup_gm`` / ``cleanup_gm`` later.
        """
        gm = _make_relu_chain_gm(num_ops=12)
        iter_gm = IterGraphModule(gm)

        segment_and_lower(
            iter_gm,  # pyre-ignore[6]
            compatibility_fn=_always_compatible,
            module_factory=_StubSubgraphModule,
            subgraph_prefix="xla_subgraph",
        )

        # All three sub-GMs must expose the lowered submodule attribute.
        for sub_gm_name in ("setup_gm", "main_gm", "cleanup_gm"):
            sub_gm = getattr(iter_gm, sub_gm_name)
            sub_attr = getattr(sub_gm, "xla_subgraph_0", None)
            self.assertIsNotNone(
                sub_attr,
                f"xla_subgraph_0 missing on {sub_gm_name}",
            )
            self.assertIsInstance(sub_attr, _StubSubgraphModule)

    def test_iter_graph_module_add_subgraph_module_direct(self) -> None:
        """Unit test for the ``add_subgraph_module`` helper in isolation.

        Bypasses ``segment_and_lower`` to exercise the helper directly:
        it should install the same ``nn.Module`` instance on all three
        sub-GMs and ``getattr`` should round-trip identically on each.
        """
        gm = _make_relu_chain_gm(num_ops=2)
        iter_gm = IterGraphModule(gm)
        stub = _StubSubgraphModule(_make_relu_chain_gm(num_ops=1))

        iter_gm.add_subgraph_module("xla_subgraph_42", stub)

        for sub_gm_name in ("setup_gm", "main_gm", "cleanup_gm"):
            sub_gm = getattr(iter_gm, sub_gm_name)
            self.assertIs(
                sub_gm.xla_subgraph_42,
                stub,
                f"xla_subgraph_42 on {sub_gm_name} should be the same stub instance",
            )

    def test_multiple_subgraphs_each_register_their_attribute(self) -> None:
        """Two segmentation breaks -> two call_modules -> two attributes.

        Constructs a chain where a graph break in the middle yields
        two separate subgraphs (``xla_subgraph_0`` and
        ``xla_subgraph_1``). Each must get its own attribute installed
        on every sub-GM, and lint must pass for the parent.

        Uses a custom ``compatibility_fn`` that rejects a single
        ``aten.tanh`` op to force the break (``relu``s on both sides
        remain compatible, each side has enough ops to satisfy the
        ``MIN_ATEN_OPS_TO_LOWER`` gate).
        """
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        last = x
        for _ in range(12):
            last = graph.call_function(torch.ops.aten.relu.default, args=(last,))
        # Graph break: tanh is rejected by the compatibility_fn.
        last = graph.call_function(torch.ops.aten.tanh.default, args=(last,))
        for _ in range(12):
            last = graph.call_function(torch.ops.aten.relu.default, args=(last,))
        graph.output(last)
        gm = torch.fx.GraphModule(root={}, graph=graph)
        iter_gm = IterGraphModule(gm)

        def compat(node: torch.fx.Node) -> tuple[bool, str]:
            if (
                node.op == "call_function"
                and node.target == torch.ops.aten.tanh.default
            ):
                return False, "tanh forced break"
            return True, ""

        segment_and_lower(
            iter_gm,  # pyre-ignore[6]
            compatibility_fn=compat,
            module_factory=_StubSubgraphModule,
            subgraph_prefix="xla_subgraph",
        )

        # Two call_module nodes in main; both targets resolve.
        call_nodes = [n for n in iter_gm.main_gm.graph.nodes if n.op == "call_module"]
        self.assertEqual(len(call_nodes), 2)
        targets = sorted(n.target for n in call_nodes)  # type: ignore[arg-type]
        self.assertEqual(targets, ["xla_subgraph_0", "xla_subgraph_1"])

        for name in targets:
            for sub_gm_name in ("setup_gm", "main_gm", "cleanup_gm"):
                sub_gm = getattr(iter_gm, sub_gm_name)
                self.assertIsNotNone(
                    getattr(sub_gm, name, None),
                    f"{name} missing on {sub_gm_name}",
                )

        # And the lint at the end of segment_and_lower would have raised
        # if any call_module attribute were missing in main_gm; re-lint
        # here for an explicit guard.
        iter_gm.main_gm.graph.lint()
