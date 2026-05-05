# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Unit tests for :func:`recir.bundle_runtime.import_bundle`.

The hard-required test in this module is the **format-version gate**:
the loader must reject ANY value that does not match the supported
:data:`~recir.bundle_export.manifest.BUNDLE_FORMAT_VERSION`, not just a
specific legacy version.  The test exercises three distinct unsupported
values (legacy ``"1.0"``, future ``"3.0"``, junk ``"abc"``) so the
symmetric semantics are explicit in the test surface.

Other tests cover focused failure modes (missing file, checksum
mismatch) and the happy-path shape of :class:`LoadedBundle`.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from recir.bundle_export import BUNDLE_FORMAT_VERSION, export_bundle
from recir.bundle_export.exporter import (
    CONSTANTS_FILENAME,
    MANIFEST_FILENAME,
    MODEL_STATE_FILENAME,
)
from recir.bundle_export.manifest import manifest_from_json, write_manifest
from recir.bundle_runtime import import_bundle, LoadedBundle
from torch.fx.graph_module import _get_attr


class _ConstAdderModule(nn.Module):
    """Tiny module whose forward adds an inline tensor constant.

    The ``torch.tensor([1.0, 2.0])`` literal lifts to a
    ``_tensor_constantN`` get_attr after tracing, exercising the
    constants pipeline.  The :class:`nn.Linear` contributes a parameter
    to ``model_state.safetensors``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc: nn.Linear = nn.Linear(2, 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x) + torch.tensor([1.0, 2.0])


def _export_const_adder(out_dir: Path) -> None:
    """Export the const-adder fixture into ``out_dir`` for loader tests."""
    torch.manual_seed(0)
    mod = _ConstAdderModule()
    x = torch.randn(2, 2)
    export_bundle(mod, (x,), output_dir=out_dir, target_device="cpu")


class ImportBundleHappyPathTest(unittest.TestCase):
    """Loader returns a fully populated :class:`LoadedBundle`."""

    def test_import_bundle_returns_loaded_bundle_with_populated_fields(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as d:
            _export_const_adder(Path(d))
            loaded = import_bundle(d)

        self.assertIsInstance(loaded, LoadedBundle)
        # The traced module has exactly one nn.Linear with a single
        # weight parameter.
        self.assertIn("fc.weight", loaded.params)
        self.assertIsInstance(loaded.params["fc.weight"], nn.Parameter)
        # No buffers in this fixture.
        self.assertEqual(loaded.buffers, {})
        # No optimizer state in this fixture.
        self.assertEqual(loaded.named_states, {})
        # At least one tensor constant from the inline ``torch.tensor``
        # literal.
        self.assertGreater(len(loaded.constants), 0)
        # Manifest carries the supported version (sanity check that it
        # was actually parsed and gated).
        self.assertIsNotNone(loaded.manifest)
        assert loaded.manifest is not None  # for pyre
        self.assertEqual(loaded.manifest.bundle_format_version, BUNDLE_FORMAT_VERSION)
        # Bindings: constants are present in the table but excluded
        # from the call vector by design.
        constant_fqns = {b.fqn for b in loaded.bindings.constants()}
        call_vector_fqns = {b.fqn for b in loaded.bindings.call_vector()}
        for fqn in constant_fqns:
            self.assertNotIn(fqn, call_vector_fqns)

    def test_import_bundle_seeds_constants_as_attr_kind_constant_on_gm(
        self,
    ) -> None:
        """The deserialized GM has ``get_attr`` constants as CONSTANT-kind attrs.

        Per :func:`recir.serde.deserialize_fx_graph`, supplying the
        ``constants`` dict triggers the seed-before-construct + reflip
        dance so the values end up as ``_AttrKind.CONSTANT`` rather than
        silent buffers.  This test asserts the loader actually invoked
        the deserializer with the constants dict (otherwise the constants
        would be missing or registered as buffers).
        """
        with tempfile.TemporaryDirectory() as d:
            _export_const_adder(Path(d))
            loaded = import_bundle(d)

        # Every get_attr FQN in the graph must resolve to a tensor on
        # the GM, AND must NOT appear in named_buffers (proving the
        # CONSTANT-kind survived).
        buffer_fqns = set(dict(loaded.gm.named_buffers()).keys())
        get_attr_fqns = {
            n.target
            for n in loaded.gm.graph.nodes
            if n.op == "get_attr" and isinstance(n.target, str)
        }
        self.assertGreater(
            len(get_attr_fqns),
            0,
            "fixture is degenerate: no get_attr nodes",
        )
        for fqn in get_attr_fqns:
            value = _get_attr(loaded.gm, fqn)
            self.assertIsInstance(value, torch.Tensor)
            self.assertNotIn(
                fqn,
                buffer_fqns,
                f"constant {fqn!r} leaked into named_buffers (CONSTANT-kind lost)",
            )


class ImportBundleVersionGateTest(unittest.TestCase):
    """The format-version gate rejects ANY unsupported value."""

    def _rewrite_manifest_version(self, bundle_dir: Path, new_version: str) -> None:
        """Mutate the manifest's ``bundle_format_version`` AND its sha256.

        After mutation the manifest's own digest in ``artifact_index``
        would be wrong if it were listed there.  The producer does NOT
        list ``manifest.json`` in ``artifact_index`` (the manifest IS
        the index; checksumming itself would be circular), so a simple
        rewrite suffices to exercise the version gate without tripping
        the checksum verifier first.
        """
        manifest_path = bundle_dir / MANIFEST_FILENAME
        manifest = manifest_from_json(manifest_path.read_text())
        manifest.bundle_format_version = new_version
        write_manifest(manifest, manifest_path)

    def test_loader_rejects_unsupported_bundle_format_version(self) -> None:
        """Loader rejects every unsupported value: legacy, future, and junk.

        Three distinct values exercise the symmetric gate semantics --
        the loader should reject anything that isn't exactly the
        currently-supported version, not just a specific legacy
        version.  Each error message must cite both the rejected value
        and the supported value so an operator can diagnose at a glance.
        """
        for unsupported in ("1.0", "3.0", "abc"):
            with self.subTest(version=unsupported):
                with tempfile.TemporaryDirectory() as d:
                    out = Path(d)
                    _export_const_adder(out)
                    self._rewrite_manifest_version(out, unsupported)
                    with self.assertRaises(RuntimeError) as cm:
                        import_bundle(out)
                    msg = str(cm.exception)
                    self.assertIn(
                        unsupported,
                        msg,
                        f"error must cite the rejected value {unsupported!r}",
                    )
                    self.assertIn(
                        BUNDLE_FORMAT_VERSION,
                        msg,
                        f"error must cite the supported value "
                        f"{BUNDLE_FORMAT_VERSION!r}",
                    )

    def test_loader_accepts_supported_bundle_format_version(self) -> None:
        """Sanity check: an unmodified bundle does not trigger the gate.

        Pairs with :meth:`test_loader_rejects_unsupported_bundle_format_version`
        to confirm the gate is not over-eager (every test bundle in this
        module relies on this baseline).
        """
        with tempfile.TemporaryDirectory() as d:
            _export_const_adder(Path(d))
            # Should not raise.
            import_bundle(d)


class ImportBundleIntegrityTest(unittest.TestCase):
    """Loader rejects bundles with missing files or bad checksums."""

    def test_loader_rejects_missing_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            _export_const_adder(out)
            (out / MODEL_STATE_FILENAME).unlink()
            with self.assertRaises(RuntimeError) as cm:
                import_bundle(out)
            self.assertIn(MODEL_STATE_FILENAME, str(cm.exception))

    def test_loader_rejects_checksum_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            _export_const_adder(out)
            # Flip a byte in the constants safetensors -- the manifest's
            # recorded sha will no longer match.
            constants_path = out / CONSTANTS_FILENAME
            data = bytearray(constants_path.read_bytes())
            # Mutate the LAST byte (bundle is small enough that any
            # byte mutation is detectable; the last byte avoids touching
            # safetensors header magic).
            data[-1] ^= 0xFF
            constants_path.write_bytes(bytes(data))
            with self.assertRaises(RuntimeError) as cm:
                import_bundle(out)
            self.assertIn("checksum mismatch", str(cm.exception))
            self.assertIn(CONSTANTS_FILENAME, str(cm.exception))

    def test_loader_rejects_non_directory_path(self) -> None:
        with tempfile.NamedTemporaryFile() as f:
            with self.assertRaises(RuntimeError) as cm:
                import_bundle(f.name)
            self.assertIn("not a directory", str(cm.exception))

    def test_loader_rejects_missing_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            # Empty directory, no manifest.
            with self.assertRaises(RuntimeError) as cm:
                import_bundle(d)
            self.assertIn(MANIFEST_FILENAME, str(cm.exception))


class ImportBundleBindingTableTest(unittest.TestCase):
    """The BindingTable round-trips through the loader."""

    def test_loader_call_vector_excludes_constants_by_design(self) -> None:
        """The producer recorded constants in the table but
        :meth:`BindingTable.call_vector` filters them; the loader does
        not need to strip them manually."""
        with tempfile.TemporaryDirectory() as d:
            _export_const_adder(Path(d))
            loaded = import_bundle(d)
        constant_fqns = {b.fqn for b in loaded.bindings.constants()}
        call_vector_fqns = {b.fqn for b in loaded.bindings.call_vector()}
        # The fixture has at least one constant.
        self.assertGreater(len(constant_fqns), 0)
        # Every constant FQN is absent from the call vector.
        for fqn in constant_fqns:
            self.assertNotIn(fqn, call_vector_fqns)
        # And the call vector includes the param + user_input bindings
        # the producer emitted (sanity that the table was loaded).
        kinds = {b.kind for b in loaded.bindings.call_vector()}
        self.assertIn("param", kinds)
        self.assertIn("user_input", kinds)


if __name__ == "__main__":
    unittest.main()
