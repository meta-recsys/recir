# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors

"""Tests for recir.ops.marker — the mark_boundary custom op and hook helpers."""

import unittest

import torch
import torch.nn as nn
from recir.ops.marker import (
    attach_markers,
    AuxSpec,
    DeriveSpec,
    InjectSpec,
    mark_boundary,
    remove_markers,
)
from torch.fx.experimental.proxy_tensor import make_fx


class TestMarkBoundary(unittest.TestCase):
    """Unit tests for the recir::mark_boundary custom op."""

    def test_mark_boundary_identity(self) -> None:
        """mark_boundary must return a tensor with the same values as the input."""
        x = torch.randn(3, 4)
        y = mark_boundary(x, "foo")
        self.assertTrue(torch.equal(x, y))

    def test_mark_boundary_survives_make_fx(self) -> None:
        """mark_boundary should appear as a call_function node after make_fx tracing."""

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 4)

            def forward(self, x):
                x = torch.ops.recir.mark_boundary(x, "probe")
                return self.linear(x)

        model = TinyModel()
        x = torch.randn(2, 4)
        gm = make_fx(model, tracing_mode="fake", _allow_non_fake_inputs=True)(x)

        # Look for the mark_boundary node in the traced graph
        found = False
        for node in gm.graph.nodes:
            if (
                node.op == "call_function"
                and node.target is torch.ops.recir.mark_boundary.default
            ):
                found = True
                break
        self.assertTrue(found, "mark_boundary node not found in traced graph")

    def test_attach_markers_inject(self) -> None:
        """attach_markers with an inject spec should fire a pre-hook that inserts mark_boundary."""

        model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        spec = InjectSpec(at="0")  # target the Linear layer

        handles = attach_markers(model, inject_specs=[spec])
        self.assertEqual(len(handles), 1)

        # Run the model and trace to verify the marker shows up
        x = torch.randn(2, 4)
        gm = make_fx(model, tracing_mode="fake", _allow_non_fake_inputs=True)(x)

        found = False
        for node in gm.graph.nodes:
            if (
                node.op == "call_function"
                and node.target is torch.ops.recir.mark_boundary.default
            ):
                found = True
                break
        self.assertTrue(found, "mark_boundary node not found after attach_markers")

        remove_markers(handles)

    def test_attach_markers_aux(self) -> None:
        """attach_markers with an aux spec should fire a hook that wraps output."""

        model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        spec = AuxSpec(at="1")  # target the ReLU layer

        handles = attach_markers(model, inject_specs=[], aux_specs=[spec])
        self.assertEqual(len(handles), 1)

        x = torch.randn(2, 4)
        gm = make_fx(model, tracing_mode="fake", _allow_non_fake_inputs=True)(x)

        found = False
        for node in gm.graph.nodes:
            if (
                node.op == "call_function"
                and node.target is torch.ops.recir.mark_boundary.default
            ):
                found = True
                break
        self.assertTrue(found, "mark_boundary node not found after aux attach_markers")

        remove_markers(handles)

    def test_attach_remove(self) -> None:
        """After remove_markers, the hooks must no longer fire."""

        model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        spec = InjectSpec(at="0")

        handles = attach_markers(model, inject_specs=[spec])
        remove_markers(handles)

        # Trace again — no mark_boundary should appear
        x = torch.randn(2, 4)
        gm = make_fx(model, tracing_mode="fake", _allow_non_fake_inputs=True)(x)

        found = False
        for node in gm.graph.nodes:
            if (
                node.op == "call_function"
                and node.target is torch.ops.recir.mark_boundary.default
            ):
                found = True
                break
        self.assertFalse(found, "mark_boundary node should NOT be present after remove")

    def test_attach_markers_derive(self) -> None:
        """attach_markers with a derive spec should fire a hook on the source module."""

        class TwoLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer0 = nn.Linear(4, 4)
                self.layer1 = nn.Linear(4, 4)

            def forward(self, x):
                x = self.layer0(x)
                x = self.layer1(x)
                return x

        model = TwoLayer()
        spec = DeriveSpec(at="layer0", input_from="layer0::output")

        handles = attach_markers(model, inject_specs=[], derive_specs=[spec])
        self.assertEqual(len(handles), 1)

        x = torch.randn(2, 4)
        gm = make_fx(model, tracing_mode="fake", _allow_non_fake_inputs=True)(x)

        found = any(
            n.op == "call_function"
            and n.target is torch.ops.recir.mark_boundary.default
            for n in gm.graph.nodes
        )
        self.assertTrue(found, "mark_boundary node not found for derive spec")
        remove_markers(handles)

    def test_empty_specs(self) -> None:
        """attach_markers with empty spec lists should return no handles."""
        model = nn.Linear(4, 4)
        handles = attach_markers(model, inject_specs=[], derive_specs=[], aux_specs=[])
        self.assertEqual(len(handles), 0)

    def test_mark_boundary_autograd(self) -> None:
        """Gradient passes through mark_boundary unchanged."""
        x = torch.randn(3, 4, requires_grad=True)
        y = mark_boundary(x, "test")
        loss = y.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.equal(x.grad, torch.ones_like(x)))

    def test_mark_boundary_backward_in_model(self) -> None:
        """mark_boundary inside a model's forward does not block backward."""

        class ModelWithMarker(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 4)

            def forward(self, x):
                x = torch.ops.recir.mark_boundary(x, "input")
                return self.linear(x)

        model = ModelWithMarker()
        x = torch.randn(2, 4)
        out = model(x)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(model.linear.weight.grad)


if __name__ == "__main__":
    unittest.main()
