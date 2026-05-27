# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Smoke tests for the v7.1 promoted graph utilities.

Verifies that ``find_markers``, ``extract_subgraph``, ``replace_region``, and
``add_state`` work end-to-end on a small traced graph and are reachable from
the top-level ``recir`` package.
"""

import unittest
from typing import List

import recir
import torch
import torch.nn as nn
from recir.api import add_state, extract_subgraph, find_markers, replace_region
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
