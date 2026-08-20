# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Manifest schema + JSON helpers for the recir training-step bundle.

The manifest is the entry-point document inside a bundle directory.  It
identifies the producer (recir SHA + torch + python versions), enumerates
the sibling artifact files together with their sha256 digests, and records
the declared target device class.

The bundle is forward-only on the producer side; optimizer reconstruction
happens consumer-side, driven by the consumer's own launch config.  The
manifest carries no optimizer hints.

The manifest is intentionally **topology-free**: no parallelism config, no
world_size, no rank.  The ``target_device`` field carries only the device
class (``"cpu"`` / ``"cuda"`` / ``"cuda:N"``) the producer validated
against; it is NOT a runtime device assignment.  Topology is a runtime
concern; the same bundle should be loadable on any topology.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# Manifest schema version -- bumped on additive changes (forward-compat);
# breaking changes bump the major.  Distinct from
# :data:`BUNDLE_FORMAT_VERSION`: schema_version is for migration tooling,
# bundle_format_version is for compatibility gating.
SCHEMA_VERSION: str = "1.0"

# Bundle wire-format version.  Consumers gate compatibility on this value.
# Version 2.0 is the first stateful-aware bundle: it ships explicit
# ``user_input`` bindings, declares its target device class for export-time
# validation, and assumes the FX wire format speaks the stateful (major=2)
# schema documented in :mod:`recir.serde.schema`.
BUNDLE_FORMAT_VERSION: str = "2.0"


@dataclass
class BundleManifest:
    """Top-level metadata document written to ``manifest.json`` in a bundle.

    Attributes:
        bundle_format_version: Wire-format version of the bundle layout
            (currently :data:`BUNDLE_FORMAT_VERSION`).  Consumers gate
            compatibility on this value.
        producer_recir_sha: Source SHA of the recir tree at export time.
            ``"unknown"`` when the producer cannot determine its own SHA.
        producer_torch_version: ``torch.__version__`` at export time.
        producer_python_version: ``sys.version`` at export time.
        target_device: The device class the producer validated the bundle
            against (e.g. ``"cpu"``, ``"cuda"``, ``"cuda:0"``).  This is a
            *declared* class -- not a runtime assignment.  The export-time
            gates use this to reject bundles whose graph kwargs disagree
            with the declared class.
        artifact_index: Map from sibling filename to sha256 hex digest of
            that file's bytes.  The consumer recomputes the digest on read
            and raises if a mismatch is detected.  Pure ``filename ->
            64-char sha256 hex``; non-digest metadata lives in
            :attr:`extras`.
        schema_version: Manifest schema version
            (currently :data:`SCHEMA_VERSION`).  Distinct from
            :attr:`bundle_format_version`.
        extras: Optional dict of caller-supplied JSON-friendly metadata
            (experiment ID, dataset name, etc.) that does not belong in
            :attr:`artifact_index`.  Defaults to an empty dict.
    """

    bundle_format_version: str
    producer_recir_sha: str
    producer_torch_version: str
    producer_python_version: str
    target_device: str
    artifact_index: dict[str, str] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION
    extras: dict[str, Any] = field(default_factory=dict)


def manifest_to_json(manifest: BundleManifest, *, indent: int | None = 2) -> str:
    """Encode a :class:`BundleManifest` as a JSON string.

    The manifest dataclass uses only JSON-friendly field types
    (``str``/``dict``/``list`` of primitives), so :func:`dataclasses.asdict`
    + :func:`json.dumps` is sufficient.

    Args:
        manifest: The manifest to encode.
        indent: ``json.dumps`` indent parameter.  ``None`` produces a
            compact single-line payload; ``2`` (default) is human-readable.

    Returns:
        The encoded JSON string.
    """
    return json.dumps(dataclasses.asdict(manifest), indent=indent, sort_keys=True)


def manifest_from_json(payload: str | dict[str, Any]) -> BundleManifest:
    """Decode a JSON string or dict back into a :class:`BundleManifest`.

    Missing optional fields fall back to their dataclass defaults; all
    required producer-identification fields must be present.

    Args:
        payload: Either a raw JSON string (must start with ``{``) or an
            already-parsed dict.

    Returns:
        The reconstructed :class:`BundleManifest`.

    Raises:
        ValueError: If the payload is missing required fields.
    """
    data: dict[str, Any] = (
        json.loads(payload) if isinstance(payload, str) else dict(payload)
    )
    required: tuple[str, ...] = (
        "bundle_format_version",
        "producer_recir_sha",
        "producer_torch_version",
        "producer_python_version",
        "target_device",
    )
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"BundleManifest payload missing required fields: {missing}")
    return BundleManifest(
        bundle_format_version=data["bundle_format_version"],
        producer_recir_sha=data["producer_recir_sha"],
        producer_torch_version=data["producer_torch_version"],
        producer_python_version=data["producer_python_version"],
        target_device=data["target_device"],
        artifact_index=dict(data.get("artifact_index", {})),
        schema_version=data.get("schema_version", SCHEMA_VERSION),
        extras=dict(data.get("extras", {})),
    )


def write_manifest(manifest: BundleManifest, path: str | Path) -> None:
    """Write a :class:`BundleManifest` to disk as JSON.

    The output is human-readable (indent=2) so producers and consumers can
    hand-inspect the bundle metadata.

    Args:
        manifest: The manifest to write.
        path: Destination path.  Will be overwritten if it already exists.
    """
    Path(path).write_text(manifest_to_json(manifest))
