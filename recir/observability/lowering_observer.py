# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""The snapshot's observer of a backend's lowering."""

from __future__ import annotations

from recir.composite import Part
from recir.lowering.lowering_utils import ExtractedSubgraph, LazilyBuiltModule
from recir.observability.builds import BuildTicket
from recir.observability.guard import contained


class SnapshotLoweringObserver:
    """
    The snapshot's observer of one backend dispatch in a step compile: told
    what lowering does, input by input.

    This class is responsible for giving each extracted subgraph that builds
    lazily a ticket for its build, and installing the ticket as the module's
    ``compile_scope``. A module that is not a ``LazilyBuiltModule``
    gets none, since nothing would ever enter it. Every event runs under
    ``contained``, so a fault is logged and never fails lowering; an event
    edits nothing lowering returns beyond that ``compile_scope``.
    """

    def __init__(self, backend: str) -> None:
        self._backend = backend

    def subgraph_extracted(self, part: Part, subgraph: ExtractedSubgraph) -> None:
        """``subgraph`` was extracted from the dispatch input ``part``."""
        with contained(f"{part.label} {subgraph.name} build not reserved"):
            module = subgraph.module
            if isinstance(module, LazilyBuiltModule):
                ticket = BuildTicket(part, subgraph, self._backend)
                module.compile_scope = ticket.building
