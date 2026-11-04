# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Builds of carved subgraphs, each under its own compile id."""

from __future__ import annotations

from contextlib import contextmanager, ExitStack
from typing import Iterator

from recir.composite import Part
from recir.lowering.lowering_utils import CarvedRegion
from recir.observability.compiles import code_location, snapshotted, TlparseCompile
from recir.observability.graph_dump import GraphDump
from torch._guards import compile_context


class SubgraphBuild(TlparseCompile):
    """
    One carved subgraph's build: its own compile id in tlparse.

    This class is responsible for the build's entry in tlparse. A build is
    created each time its subgraph's forward compiles it, after the step
    compile has closed, and the whole backend compile runs under its id, so a
    backend's own files for the build land in it. It is named ``name``, sits
    in tlparse's stack trie under the last call from outside RecIR, like every
    compile, and reports the carved subgraph's node counts. ``backend`` names
    the backend that builds it.

    This can be extended to record more of a backend's build by subclassing
    and extending ``_setup``.
    """

    def __init__(
        self,
        name: str,
        counts: dict[str, int],
        code_location: tuple[str, int] | None,
        backend: str,
    ) -> None:
        super().__init__(counts, code_location)
        self._name = name
        self._backend = backend

    @property
    def name(self) -> str:
        """The name shown in tlparse's stack trie, e.g. "recir build stage2_compute subgraph_0"."""
        return self._name

    def _fail_reason(self, error: BaseException) -> str:
        return f"the {self._backend} build raised"

    def _setup(self, stack: ExitStack) -> None:
        stack.enter_context(compile_context(self._compile_context()))


class BuildTicket:
    """
    What a carved subgraph keeps for its build: plain data, taken when the
    subgraph is carved.

    This class is responsible for creating a ``build_cls`` build each time the
    subgraph's forward compiles it, and running that compile as the build.
    The dispatch observer installs ``building`` as the module's
    ``compile_scope``. The build is named after the dispatch input ``part``
    it was carved from and the subgraph's name in that input's lowered graph.
    """

    def __init__(
        self,
        build_cls: type[SubgraphBuild],
        part: Part,
        region: CarvedRegion,
        backend: str,
    ) -> None:
        self._build_cls = build_cls
        self._name = f"recir build {part.label} {region.name}"
        self._counts: dict[str, int] = GraphDump(region.gm).counts
        self._code_location: tuple[str, int] | None = code_location(
            type(region.module).forward
        )
        self._backend = backend

    @contextmanager
    def building(self) -> Iterator[None]:
        """Run the subgraph's backend compile inside this context, as a new build while snapshots are on."""
        with snapshotted(
            lambda: self._build_cls(
                self._name, self._counts, self._code_location, self._backend
            ),
            self._name,
        ):
            yield
