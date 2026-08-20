# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Export-time gate firing tests.

The two gates under test:

* :func:`recir.bundle_export._gates._validate_kwargs_meta_consistency`
  rejects FX nodes whose ``device`` kwarg disagrees with
  ``meta['val'].device``.
* :func:`recir.bundle_export._gates._validate_no_cpu_scalar_in_cuda_bundle`
  rejects ``aten.scalar_tensor.default`` nodes pinned to ``device='cpu'``
  inside a CUDA-targeted bundle.

Each test below builds a hand-assembled FX graph that violates exactly
ONE invariant (so the gate's rejection is unambiguous), then asserts
:meth:`BundleExporter.export` raises before any artifact is written.

The integration tests use :class:`BundleExporter` directly (rather than
the public :func:`export_bundle` ``nn.Module`` -> bundle entry point) so
we can feed a hand-assembled bad ``GraphModule`` straight to the gates
-- there is no ``nn.Module`` whose ``make_fx`` trace produces these
exact wrong patterns.

Note: :class:`GateUnitTests` deliberately tests the underscore-prefixed
gate helpers directly so we can cover edge cases -- absent ``device``
kwargs, hand-assembled drift patterns, ``_devices_compatible`` truth-table
corners -- that the integration path would otherwise mask behind a
bundle of side effects. :class:`ExportLevelGateTests` then covers the
integration story.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import torch.fx as fx
import torch.nn as nn
from recir.bundle_export._gates import (
    _devices_compatible,
    _validate_kwargs_meta_consistency,
    _validate_no_cpu_scalar_in_cuda_bundle,
)
from recir.bundle_export.exporter import BundleExporter, CompiledBundleInputs
from recir.serde import SchemaViolationError
from torch._subclasses.fake_tensor import FakeTensorMode


def _empty_bundle_dir() -> Path:
    return Path(tempfile.mkdtemp())


def _build_scalar_tensor_gm(scalar_device: str) -> fx.GraphModule:
    """Build a FX graph that calls ``aten.scalar_tensor.default(0.5, device=...)``.

    The graph has one placeholder ``x`` and adds the scalar tensor to it.
    The scalar's ``device`` kwarg is exactly the value passed by the
    caller, and the node's ``meta['val']`` agrees with that device so the
    *kwargs/meta consistency* gate passes -- only the *no-CPU-in-CUDA*
    gate is being exercised here.
    """
    fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
    target_device = torch.device(scalar_device)
    graph = fx.Graph()
    x_node = graph.placeholder("x")
    with fake_mode:
        x_fake = torch.empty((2, 2), device=target_device)
    x_node.meta["val"] = x_fake

    scalar_node = graph.create_node(
        op="call_function",
        target=torch.ops.aten.scalar_tensor.default,
        args=(0.5,),
        kwargs={"device": target_device, "dtype": torch.float32},
        name="scalar_0",
    )
    with fake_mode:
        scalar_fake = torch.empty((), device=target_device, dtype=torch.float32)
    scalar_node.meta["val"] = scalar_fake

    add_node = graph.create_node(
        op="call_function",
        target=torch.ops.aten.add.Tensor,
        args=(x_node, scalar_node),
        kwargs={},
        name="add_0",
    )
    with fake_mode:
        add_fake = torch.empty((2, 2), device=target_device)
    add_node.meta["val"] = add_fake

    graph.output(add_node)
    return fx.GraphModule(nn.Module(), graph)


def _build_drift_gm() -> fx.GraphModule:
    """Build a graph whose ``device`` kwarg disagrees with ``meta['val'].device``.

    Targets the kwargs/meta consistency gate.  The kwargs claim ``cuda``
    while the meta carries a CPU FakeTensor -- exactly the silent drift
    the gate is meant to catch.
    """
    fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
    graph = fx.Graph()
    x_node = graph.placeholder("x")
    with fake_mode:
        x_fake = torch.empty((2, 2), device="cpu")
    x_node.meta["val"] = x_fake

    scalar_node = graph.create_node(
        op="call_function",
        target=torch.ops.aten.scalar_tensor.default,
        args=(0.5,),
        # Drift: kwarg device disagrees with meta device.
        kwargs={"device": torch.device("cuda"), "dtype": torch.float32},
        name="scalar_drift",
    )
    with fake_mode:
        scalar_fake = torch.empty((), device="cpu", dtype=torch.float32)
    scalar_node.meta["val"] = scalar_fake

    graph.output(x_node)
    return fx.GraphModule(nn.Module(), graph)


def _build_clean_cpu_gm() -> fx.GraphModule:
    """Build a small CPU graph that should pass both gates cleanly.

    Used to assert the gates aren't trivially failing on every input.
    """
    fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
    graph = fx.Graph()
    x_node = graph.placeholder("x")
    with fake_mode:
        x_fake = torch.empty((2, 2), device="cpu")
    x_node.meta["val"] = x_fake

    add_node = graph.create_node(
        op="call_function",
        target=torch.ops.aten.add.Tensor,
        args=(x_node, x_node),
        kwargs={},
        name="add_0",
    )
    with fake_mode:
        add_node.meta["val"] = torch.empty((2, 2), device="cpu")
    graph.output(add_node)
    return fx.GraphModule(nn.Module(), graph)


class GateUnitTests(unittest.TestCase):
    """Direct unit tests against the gate functions (no full export)."""

    def test_devices_compatible_basics(self) -> None:
        self.assertTrue(_devices_compatible(torch.device("cpu"), torch.device("cpu")))
        self.assertTrue(
            _devices_compatible(torch.device("cuda"), torch.device("cuda:0"))
        )
        self.assertFalse(_devices_compatible(torch.device("cuda"), torch.device("cpu")))
        self.assertFalse(
            _devices_compatible(torch.device("cuda:0"), torch.device("cuda:1"))
        )

    def test_kwargs_meta_consistency_passes_on_clean_graph(self) -> None:
        """Clean CPU graph must NOT trigger the kwargs/meta gate."""
        gm = _build_clean_cpu_gm()
        # Should not raise.
        _validate_kwargs_meta_consistency(gm)

    def test_kwargs_meta_consistency_fires_on_drift(self) -> None:
        """Drift case (cuda kwarg, cpu meta) must raise SchemaViolationError."""
        gm = _build_drift_gm()
        with self.assertRaisesRegex(SchemaViolationError, r"device kwarg"):
            _validate_kwargs_meta_consistency(gm)

    def test_no_cpu_scalar_in_cuda_bundle_passes_on_cpu_target(self) -> None:
        """A CPU-targeted bundle is allowed to carry CPU scalars."""
        gm = _build_scalar_tensor_gm("cpu")
        # Should not raise.
        _validate_no_cpu_scalar_in_cuda_bundle(gm, "cpu")

    def test_no_cpu_scalar_in_cuda_bundle_passes_when_scalar_omits_device(self) -> None:
        """Scalar without a device kwarg passes regardless of bundle target."""
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
        graph = fx.Graph()
        scalar_node = graph.create_node(
            op="call_function",
            target=torch.ops.aten.scalar_tensor.default,
            args=(0.5,),
            kwargs={"dtype": torch.float32},  # no `device` kwarg
            name="scalar_no_dev",
        )
        with fake_mode:
            scalar_node.meta["val"] = torch.empty((), dtype=torch.float32)
        graph.output(scalar_node)
        gm = fx.GraphModule(nn.Module(), graph)
        # Should not raise -- runtime fills in the device.
        _validate_no_cpu_scalar_in_cuda_bundle(gm, "cuda")

    def test_no_cpu_scalar_in_cuda_bundle_fires_on_cpu_in_cuda(self) -> None:
        """The exact case the gate exists to catch."""
        gm = _build_scalar_tensor_gm("cpu")
        with self.assertRaisesRegex(
            SchemaViolationError, r"CPU scalars baked into a CUDA bundle"
        ):
            _validate_no_cpu_scalar_in_cuda_bundle(gm, "cuda")


class ExportLevelGateTests(unittest.TestCase):
    """Tests that run the full :meth:`BundleExporter.export` pipeline.

    These exercise the gates in their natural calling context AND assert
    the output dir is left in a clean state (no half-written artifacts)
    when a gate fires.
    """

    def test_export_rejects_cpu_scalar_in_cuda_bundle(self) -> None:
        """End-to-end: export of a CPU-scalar / CUDA-target combo must raise."""
        gm = _build_scalar_tensor_gm("cpu")
        compiled = CompiledBundleInputs(
            gm=gm,
            params={},
            buffers={},
            user_input_count=1,
        )
        out = _empty_bundle_dir()
        with self.assertRaises(SchemaViolationError):
            BundleExporter().export(compiled, output_dir=out, target_device="cuda")
        # No bundle artifacts should have been written.
        for fname in (
            "manifest.json",
            "train_step.fx.json",
            "train.binding.json",
            "model_state.safetensors",
            "constants.safetensors",
        ):
            self.assertFalse(
                (out / fname).exists(),
                f"{fname!r} should not exist after a gate failure",
            )

    def test_export_rejects_kwargs_meta_drift(self) -> None:
        """End-to-end: export of a graph with kwargs/meta drift must raise."""
        gm = _build_drift_gm()
        compiled = CompiledBundleInputs(
            gm=gm,
            params={},
            buffers={},
            user_input_count=1,
        )
        out = _empty_bundle_dir()
        with self.assertRaises(SchemaViolationError):
            # target_device='cpu' so the no-CPU-in-CUDA gate doesn't also
            # catch this; we want to isolate the drift gate.
            BundleExporter().export(compiled, output_dir=out, target_device="cpu")
        self.assertFalse((out / "manifest.json").exists())

    def test_export_accepts_clean_cpu_graph(self) -> None:
        """Sanity: a clean CPU graph passes both gates and writes the bundle."""
        gm = _build_clean_cpu_gm()
        compiled = CompiledBundleInputs(
            gm=gm,
            params={},
            buffers={},
            user_input_count=1,
        )
        out = _empty_bundle_dir()
        manifest = BundleExporter().export(
            compiled, output_dir=out, target_device="cpu"
        )
        self.assertEqual(manifest.target_device, "cpu")


if __name__ == "__main__":
    unittest.main()
