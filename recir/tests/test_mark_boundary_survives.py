# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Smoke test: recir.mark_boundary survives the full v7.1 pipeline.

Plants a marker via attach_markers, runs the model through trace -> finalize
-> partition -> optimize -> dispatch using the new public API, and asserts
that recir.find_markers can still locate it on the final gm.

This guards against DCE / pass regressions that would silently strip
graph-composition markers used by external surgery integrations.
"""

import unittest
from typing import Any, Sequence

import recir
import torch
import torch.nn as nn
from recir.tests.test_utils import _NoopBackend, _PassthroughParallel
from torch import fx


class _MarkedModel(nn.Module):
    """Tiny model that calls mark_boundary in its forward."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.ops.recir.mark_boundary(x, "my_marker")
        return self.linear(x).sum()


class MarkBoundarySurvivesTest(unittest.TestCase):
    """recir.mark_boundary must survive the full v7.1 pipeline."""

    def test_marker_survives_trace_only(self) -> None:
        """After recir.trace, the marker is still findable in the gm."""
        model = _MarkedModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=True)

        def train_step(m: nn.Module, o: torch.optim.Optimizer, x: torch.Tensor) -> None:
            o.zero_grad()
            loss = m(x)
            loss.backward()
            o.step()

        x = torch.randn(2, 4)
        gm, _flat_state = recir.trace(train_step, model, opt, x)
        markers = recir.find_markers(gm)
        self.assertEqual(len(markers), 1, "marker disappeared during trace")
        self.assertEqual(markers[0].args[1], "my_marker")

    def test_marker_survives_full_pipeline(self) -> None:
        """trace -> partition -> finalize -> optimize -> dispatch keeps the marker."""
        model = _MarkedModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=True)

        def train_step(m: nn.Module, o: torch.optim.Optimizer, x: torch.Tensor) -> None:
            o.zero_grad()
            loss = m(x)
            loss.backward()
            o.step()

        x = torch.randn(2, 4)
        gm, flat_state = recir.trace(train_step, model, opt, x)
        compiled = recir.compile_from_gm(
            gm,
            flat_state,
            parallel_mode=_PassthroughParallel(),
            backend=_NoopBackend(),
        )
        markers = recir.find_markers(compiled.gm)
        self.assertGreaterEqual(
            len(markers), 1, "mark_boundary was DCE'd through the pipeline"
        )
        marker_names = [m.args[1] for m in markers]
        self.assertIn("my_marker", marker_names)

    def test_marker_survives_attach_markers_path(self) -> None:
        """Markers installed via attach_markers also survive the pipeline."""

        class _PlainModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer = nn.Linear(4, 4)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.layer(x).sum()

        model = _PlainModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=True)
        # AuxSpec at "layer" -> wraps the layer's output with mark_boundary.
        handles = recir.attach_markers(
            model,
            inject_specs=[],
            aux_specs=[recir.ops.marker.AuxSpec(at="layer")],
        )
        self.assertEqual(len(handles), 1)

        try:

            def train_step(
                m: nn.Module, o: torch.optim.Optimizer, x: torch.Tensor
            ) -> None:
                o.zero_grad()
                loss = m(x)
                loss.backward()
                o.step()

            x = torch.randn(2, 4)
            gm, flat_state = recir.trace(train_step, model, opt, x)
            compiled = recir.compile_from_gm(
                gm,
                flat_state,
                parallel_mode=_PassthroughParallel(),
                backend=_NoopBackend(),
            )
            markers = recir.find_markers(compiled.gm)
            self.assertGreaterEqual(
                len(markers),
                1,
                "attach_markers-installed marker DCE'd through the pipeline",
            )
            self.assertTrue(
                any("layer" in str(m.args[1]) for m in markers),
                "expected 'layer' marker name to appear after pipeline",
            )
        finally:
            recir.ops.marker.remove_markers(handles)

    def test_marker_survives_optimize_passes(self) -> None:
        """A no-op optimize pass shouldn't remove markers."""

        called: list[bool] = []

        def _identity_pass(gm: fx.GraphModule) -> fx.GraphModule:
            called.append(True)
            return gm

        class _PassBackend(_NoopBackend):
            @classmethod
            def default_optimize_passes(cls) -> Sequence[Any]:
                return (_identity_pass,)

        model = _MarkedModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=True)

        def train_step(m: nn.Module, o: torch.optim.Optimizer, x: torch.Tensor) -> None:
            o.zero_grad()
            loss = m(x)
            loss.backward()
            o.step()

        x = torch.randn(2, 4)
        gm, flat_state = recir.trace(train_step, model, opt, x)
        compiled = recir.compile_from_gm(
            gm,
            flat_state,
            parallel_mode=_PassthroughParallel(),
            backend=_PassBackend(),
        )
        self.assertTrue(called, "optimize pass was not invoked")
        markers = recir.find_markers(compiled.gm)
        self.assertGreaterEqual(len(markers), 1)
