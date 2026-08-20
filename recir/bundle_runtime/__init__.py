# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Bundle consumer for loading + lowering a recir bundle on the runtime side.

A bundle (produced by :mod:`recir.bundle_export`) is a topology-free,
source-free directory of artifacts that this package loads to
reconstruct an executable
:class:`recir.iter_graph_module.IterGraphModule` ready for lazy Inductor
first-execution compilation.

Public API:

* :func:`import_bundle` -- read the manifest, validate the format
  version, verify checksums, deserialize the FX graph (seeding ``get_attr``
  constants as :data:`~torch.export.unflatten._AttrKind.CONSTANT`
  attributes), and return a :class:`LoadedBundle`.
* :func:`wrap_and_lower` -- thread a :class:`LoadedBundle` through the
  parallel-mode partitioner + ``GraphModuleTransformation(INDUCTOR)`` and
  restore CONSTANT-kind on every :class:`IterGraphModule` sub-GM.
* :class:`LoadedBundle` -- the in-memory view of a bundle on disk.
"""

from __future__ import annotations

from .loader import import_bundle, LoadedBundle
from .runtime import wrap_and_lower


__all__ = [
    "LoadedBundle",
    "import_bundle",
    "wrap_and_lower",
]
