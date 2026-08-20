# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Bundle producer for the recir training-step wire format.

This package writes a topology-free, source-free directory of artifacts
that a downstream consumer loads to reconstruct an executable training
step.  See :mod:`recir.bundle_export.exporter` for the artifact layout.

Public API:

* :func:`export_bundle` -- raw ``nn.Module`` + sample inputs -> bundle on
  disk. Single producer-side entry point: traces, validates, writes.
* :class:`BundleManifest` -- the manifest dataclass written to
  ``manifest.json``.
* :data:`BUNDLE_FORMAT_VERSION` -- the wire-format version consumers gate
  compatibility on.
"""

from __future__ import annotations

from .exporter import export_bundle
from .manifest import BUNDLE_FORMAT_VERSION, BundleManifest, SCHEMA_VERSION


__all__ = [
    "BUNDLE_FORMAT_VERSION",
    "BundleManifest",
    "SCHEMA_VERSION",
    "export_bundle",
]
