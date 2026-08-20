# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Manifest schema + checksum tests for the recir bundle producer.

Covers:

* :data:`BUNDLE_FORMAT_VERSION` is exactly ``"2.0"`` (the fresh-stack
  starting version).
* :data:`SCHEMA_VERSION` is exactly ``"1.0"`` (the manifest schema
  version, distinct from the bundle wire-format version).
* :class:`BundleManifest` JSON round-trips via
  :func:`manifest_to_json` / :func:`manifest_from_json`.
* The manifest emitted by :func:`export_bundle` carries:

  - the declared ``target_device`` field,
  - sha256 hex digests (64 chars each) for every sibling artifact,
  - matching digests when recomputed over the file bytes,
  - no topology fields (parallelism config, world_size, rank).
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from recir.bundle_export import (
    BUNDLE_FORMAT_VERSION,
    BundleManifest,
    export_bundle,
    SCHEMA_VERSION,
)
from recir.bundle_export.exporter import (
    BINDING_FILENAME,
    CONSTANTS_FILENAME,
    GRAPH_FILENAME,
    MANIFEST_FILENAME,
    MODEL_STATE_FILENAME,
)
from recir.bundle_export.manifest import manifest_from_json, manifest_to_json


class _TwoLayerMLP(nn.Module):
    """Tiny two-layer MLP for the manifest tests' real-trace fixture.

    Its forward emits a parameter-bearing graph (so model_state.safetensors
    is non-empty) without lifting any tensor literals (so the constants
    sidecar tests in test_export_round_trip.py stay separately exercised).
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1: nn.Linear = nn.Linear(4, 4, bias=False)
        self.fc2: nn.Linear = nn.Linear(4, 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def _make_fixture() -> tuple[_TwoLayerMLP, tuple[torch.Tensor]]:
    """Construct the tiny MLP + a single sample tensor used across tests."""
    torch.manual_seed(0)
    return _TwoLayerMLP(), (torch.randn(2, 4),)


class BundleVersionTests(unittest.TestCase):
    def test_bundle_format_version_is_two_oh(self) -> None:
        """Fresh-stack consumers gate on this; never silently migrate."""
        self.assertEqual(BUNDLE_FORMAT_VERSION, "2.0")

    def test_schema_version_is_one_oh(self) -> None:
        """Manifest schema starts at 1.0 in this stack."""
        self.assertEqual(SCHEMA_VERSION, "1.0")


class ManifestRoundTripTests(unittest.TestCase):
    def test_manifest_dataclass_round_trips_via_json(self) -> None:
        """`BundleManifest` -> JSON -> `BundleManifest` is lossless."""
        manifest = BundleManifest(
            bundle_format_version=BUNDLE_FORMAT_VERSION,
            producer_recir_sha="0123456789abcdef",
            producer_torch_version=torch.__version__,
            producer_python_version="3.10.0 (test)",
            target_device="cuda:0",
            artifact_index={"foo.safetensors": "a" * 64},
            extras={"experiment": "unit_test"},
        )
        encoded = manifest_to_json(manifest)
        rebuilt = manifest_from_json(encoded)
        self.assertEqual(rebuilt, manifest)

    def test_manifest_from_json_rejects_missing_required_fields(self) -> None:
        """Missing required producer-id fields raise a clear ValueError."""
        with self.assertRaisesRegex(ValueError, r"missing required fields"):
            manifest_from_json({"bundle_format_version": "2.0"})


class ExportedManifestTests(unittest.TestCase):
    def test_exported_manifest_carries_target_device(self) -> None:
        """The manifest records the declared target_device verbatim."""
        mod, inputs = _make_fixture()
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            export_bundle(mod, inputs, output_dir=out, target_device="cuda:0")
            payload = json.loads((out / MANIFEST_FILENAME).read_text())
        self.assertEqual(payload["target_device"], "cuda:0")
        self.assertEqual(payload["bundle_format_version"], BUNDLE_FORMAT_VERSION)
        self.assertEqual(payload["schema_version"], SCHEMA_VERSION)

    def test_exported_manifest_artifact_index_covers_every_sibling(self) -> None:
        """Every sibling artifact must appear in the index with a 64-char digest."""
        mod, inputs = _make_fixture()
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            export_bundle(mod, inputs, output_dir=out, target_device="cpu")
            payload = json.loads((out / MANIFEST_FILENAME).read_text())
        artifact_index = payload["artifact_index"]
        for fname in (
            GRAPH_FILENAME,
            BINDING_FILENAME,
            MODEL_STATE_FILENAME,
            CONSTANTS_FILENAME,
        ):
            self.assertIn(fname, artifact_index)
            self.assertIsInstance(artifact_index[fname], str)
            self.assertEqual(len(artifact_index[fname]), 64)

    def test_exported_manifest_checksums_match_file_bytes(self) -> None:
        """Recomputing sha256 over the file bytes yields the manifest digest."""
        mod, inputs = _make_fixture()
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            export_bundle(mod, inputs, output_dir=out, target_device="cpu")
            payload = json.loads((out / MANIFEST_FILENAME).read_text())
            artifact_index = payload["artifact_index"]
            for fname, expected_digest in artifact_index.items():
                actual = hashlib.sha256((out / fname).read_bytes()).hexdigest()
                self.assertEqual(
                    actual,
                    expected_digest,
                    f"{fname!r} digest in manifest does not match file bytes",
                )

    def test_exported_manifest_omits_topology_fields(self) -> None:
        """Bundle is topology-free: no parallelism/world_size/rank in manifest."""
        mod, inputs = _make_fixture()
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            export_bundle(mod, inputs, output_dir=out, target_device="cpu")
            payload = json.loads((out / MANIFEST_FILENAME).read_text())
        for forbidden in (
            "world_size",
            "rank",
            "parallelism",
            "parallel_mode",
            "device_mesh",
        ):
            self.assertNotIn(
                forbidden,
                payload,
                f"Manifest must not carry topology field {forbidden!r}",
            )

    def test_exported_manifest_omits_self_digest(self) -> None:
        """The manifest does NOT digest itself (chicken-and-egg avoidance)."""
        mod, inputs = _make_fixture()
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            export_bundle(mod, inputs, output_dir=out, target_device="cpu")
            payload = json.loads((out / MANIFEST_FILENAME).read_text())
        self.assertNotIn(MANIFEST_FILENAME, payload["artifact_index"])


if __name__ == "__main__":
    unittest.main()
