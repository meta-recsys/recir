# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

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

(at ``recir/iter_graph_module.py:493`` -- the ``super().lint()`` call).

Commit 17 adds ``IterGraphModule.add_subgraph_module`` which
propagates the attribute to all three sub-GMs (setup / main /
cleanup); ``lower_subgraph_nodes`` invokes it when the parent is an
``IterGraphModule`` and falls back to the bare ``setattr`` for plain
``fx.GraphModule`` parents. The tests below assert both arms:
``gm.graph.lint`` passes post-lower and ``getattr(gm, name)`` returns
the lowered submodule.
"""

import unittest
from typing import List

import torch
from recir.iter_graph_module import IterGraphModule
from recir.lowering.lowering_utils import segment_and_lower


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
