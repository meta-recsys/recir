# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Round-trip tests for the recir bundle producer.

The happy-path test below builds a tiny module with TWO distinct
``user_input`` arguments, exports it via the public
:func:`recir.bundle_export.export_bundle` API (raw ``nn.Module`` +
sample inputs), and asserts:

* every artifact is on disk;
* tensor constants reload via the safetensors sidecar AND match the
  ``required_constants`` list in the FX payload;
* the :class:`recir.serde.BindingTable` parsed from the binding sidecar
  preserves the user-input ordering ``[<input_0>, <input_1>]``;
* constants live in the binding table's ``constants()`` list (metadata)
  and are absent from the call vector;
* the producer-populated ``bundle_kind`` key in the manifest's
  ``extras`` wins over a caller-supplied ``manifest_extras`` collision.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from recir.bundle_export import BUNDLE_FORMAT_VERSION, export_bundle
from recir.bundle_export.exporter import (
    BINDING_FILENAME,
    CONSTANTS_FILENAME,
    GRAPH_FILENAME,
    MANIFEST_FILENAME,
    MODEL_STATE_FILENAME,
)
from recir.serde import parse_bindings, required_constants
from safetensors.torch import load_file


class _TwoInputModule(nn.Module):
    """Module whose forward takes TWO distinct user-input tensors.

    The inline ``torch.tensor([0.5, 0.5])`` literal lifts to a
    ``_tensor_constantN`` get_attr after tracing, exercising the
    constants path; the linear layer contributes a parameter to
    ``model_state.safetensors``; and having two args proves the
    ``user_input`` binding ordering survives end-to-end.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc: nn.Linear = nn.Linear(4, 2, bias=False)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        scale = torch.tensor([0.5, 0.5])
        return self.fc(a) * scale + b


def _make_fixture() -> tuple[_TwoInputModule, tuple[torch.Tensor, torch.Tensor]]:
    """Construct the fixture module + 2 sample tensors used across tests."""
    torch.manual_seed(0)
    return _TwoInputModule(), (torch.randn(2, 4), torch.randn(2, 2))


class BundleExportRoundTripTest(unittest.TestCase):
    def test_export_writes_all_five_artifacts(self) -> None:
        """End-to-end: trace a 2-input module, export, assert 5 files exist."""
        mod, inputs = _make_fixture()
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            export_bundle(mod, inputs, output_dir=out, target_device="cpu")
            for fname in (
                MANIFEST_FILENAME,
                GRAPH_FILENAME,
                BINDING_FILENAME,
                MODEL_STATE_FILENAME,
                CONSTANTS_FILENAME,
            ):
                self.assertTrue(
                    (out / fname).is_file(),
                    f"Expected {fname!r} to exist in bundle dir",
                )

    def test_export_preserves_user_input_binding_order(self) -> None:
        """Build a GM with 2 distinct user_input args; assert binding order.

        Minimum two ``user_input`` bindings to prove order is preserved
        end-to-end, not just the *exclusion* of constants from the call
        vector.
        """
        mod, inputs = _make_fixture()
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            export_bundle(mod, inputs, output_dir=out, target_device="cpu")
            with (out / BINDING_FILENAME).open() as f:
                binding_payload = json.load(f)

        parsed_table = parse_bindings(binding_payload)
        self.assertIsNotNone(parsed_table)
        assert parsed_table is not None  # for pyre

        # Ordering assertion: the user-input slice of the call vector must
        # carry the synthesised <input_0>, <input_1> FQNs in that exact
        # order.
        user_inputs = [b for b in parsed_table.call_vector() if b.kind == "user_input"]
        self.assertEqual(
            [b.fqn for b in user_inputs],
            ["<input_0>", "<input_1>"],
            "user_input binding order must be preserved end-to-end",
        )
        self.assertGreaterEqual(
            len(user_inputs),
            2,
            "fixture must exercise at least two user_input bindings",
        )

    def test_export_constants_excluded_from_call_vector(self) -> None:
        """Constants are metadata-only -- they must NOT be in call_vector()."""
        mod, inputs = _make_fixture()
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            export_bundle(mod, inputs, output_dir=out, target_device="cpu")
            with (out / BINDING_FILENAME).open() as f:
                binding_payload = json.load(f)
            with (out / GRAPH_FILENAME).open() as f:
                graph_payload = json.load(f)

        parsed_table = parse_bindings(binding_payload)
        assert parsed_table is not None  # for pyre
        constant_fqns = {b.fqn for b in parsed_table.constants()}
        self.assertTrue(
            constant_fqns,
            "fixture is degenerate: no constants in BindingTable",
        )

        # Every constant FQN must be present in the FX payload's
        # required_constants list (the producer is sourcing constants from
        # the GM via the serializer-emitted list, NOT from the call vector).
        required = required_constants(graph_payload)
        for fqn in constant_fqns:
            self.assertIn(fqn, required, f"constant {fqn!r} not in required_constants")

        # And NONE of those FQNs may show up in the call vector.
        call_vector_fqns = {b.fqn for b in parsed_table.call_vector()}
        for fqn in constant_fqns:
            self.assertNotIn(
                fqn,
                call_vector_fqns,
                f"constant {fqn!r} must not appear in call_vector()",
            )

    def test_export_constants_round_trip_through_safetensors(self) -> None:
        """Constants written to constants.safetensors match the FX-payload list."""
        mod, inputs = _make_fixture()
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            export_bundle(mod, inputs, output_dir=out, target_device="cpu")
            with (out / GRAPH_FILENAME).open() as f:
                graph_payload = json.load(f)
            reloaded = load_file(str(out / CONSTANTS_FILENAME))

        # The serializer-emitted required_constants list is the source of
        # truth for what should land in the safetensors sidecar.
        required = required_constants(graph_payload)
        self.assertTrue(
            required,
            "fixture is degenerate: traced GM has no get_attr constants",
        )
        for fqn in required:
            self.assertIn(fqn, reloaded, f"constant {fqn!r} missing from sidecar")

    def test_export_bundle_format_version_two(self) -> None:
        """The exported manifest carries BUNDLE_FORMAT_VERSION = '2.0'."""
        self.assertEqual(BUNDLE_FORMAT_VERSION, "2.0")
        mod, inputs = _make_fixture()
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            manifest = export_bundle(mod, inputs, output_dir=out, target_device="cpu")
        self.assertEqual(manifest.bundle_format_version, "2.0")

    def test_bundle_kind_wins_over_manifest_extras_collision(self) -> None:
        """Producer-populated ``bundle_kind`` overrides caller's ``manifest_extras``.

        Documents the merge order: ``manifest_extras`` is folded in FIRST
        and then the producer-populated ``bundle_kind`` key overwrites --
        the producer is the single source of truth for that field and an
        unrelated extras bag must not silently override it.
        """
        mod, inputs = _make_fixture()
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            manifest = export_bundle(
                mod,
                inputs,
                output_dir=out,
                target_device="cpu",
                manifest_extras={
                    "experiment": "round_trip_test",
                    # This collision must lose to the producer-populated key.
                    "bundle_kind": "WRONG",
                },
            )
        self.assertEqual(manifest.extras["experiment"], "round_trip_test")
        self.assertEqual(manifest.extras["bundle_kind"], "forward_only")


if __name__ == "__main__":
    unittest.main()
