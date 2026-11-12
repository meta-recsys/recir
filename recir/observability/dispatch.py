# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""The snapshot's observer of a backend dispatch."""

from __future__ import annotations

from recir.composite import Part
from recir.lowering.lowering_inputs import LoweringObserver
from recir.lowering.lowering_utils import BuildKind, CarvedRegion, LazilyBuiltModule
from recir.observability.builds import BuildTicket, SubgraphBuild
from recir.observability.inductor_build import InductorBuild

# The build class of each LazilyBuiltModule.build_kind handled beyond a compile id.
_BUILD_CLASSES: dict[BuildKind | None, type[SubgraphBuild]] = {
    BuildKind.INDUCTOR: InductorBuild
}


class ObservedLowering(LoweringObserver):
    """
    The snapshot's observer of one backend dispatch in a step compile.

    This class is responsible for giving each carved subgraph that builds
    lazily a ticket for its build, of the class its module's ``build_kind``
    names or a plain ``SubgraphBuild``, and installing the ticket as the
    carved module's ``compile_scope``. A carved module that is not a
    ``LazilyBuiltModule`` gets none, since nothing would ever enter it.
    """

    def __init__(self, backend: str) -> None:
        self._backend = backend

    def _on_carved(self, part: Part, region: CarvedRegion) -> None:
        module = region.module
        if isinstance(module, LazilyBuiltModule):
            build_cls = _BUILD_CLASSES.get(module.build_kind, SubgraphBuild)
            ticket = BuildTicket(build_cls, part, region, self._backend)
            module.compile_scope = ticket.building
