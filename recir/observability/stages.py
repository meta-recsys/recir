# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""The stages a step compile snapshots, and what each one writes."""

from __future__ import annotations

from typing import Any, Callable, ClassVar, TYPE_CHECKING

from recir.composite import Part
from recir.lowering.lowering_utils import CarvedRegion
from recir.observability.graph_dump import GraphDump
from recir.observability.status import ArtifactStatus, Outcome
from torch import fx, nn

if TYPE_CHECKING:
    from recir.observability.compiles import TlparseCompile


class Stage:
    """
    Base class for all snapshot stages: one snapshot point in a compile.

    This class is responsible for naming the stage's artifacts, keeping each
    one's status, and for its lifecycle: its compile begins it, its RecIR code
    writes to it as it runs, and ends it with what that code raised, if
    anything, which writes the stage's manifest, ``recir_{name}_manifest``:
    its outcome, its artifacts' statuses, and its own fields. It writes only
    through its owning compile, whose writes never raise, and does not decide
    stage order.

    This can be extended to a new snapshot point by subclassing, setting
    ``name``, adding the methods its RecIR code writes through, and
    overriding ``_fields`` with what its manifest adds.
    """

    name: ClassVar[str]

    def __init__(self, owner: TlparseCompile) -> None:
        self.owner = owner
        # How the stage's RecIR code ended; None until it ends.
        self.outcome: Outcome | None = None
        self.artifacts: dict[str, ArtifactStatus] = {}

    def end(self, error: BaseException | None) -> None:
        """
        The stage's RecIR code is over; ``error`` is what it raised, if
        anything. Writes the stage's manifest.
        """
        self.outcome = Outcome.of(error)
        self.owner.write_manifest(
            self.name, error=error, artifacts=self.artifacts, fields=self._fields
        )

    def _fields(self) -> dict[str, Any]:
        # What this stage's manifest adds; called only when the manifest is written.
        return {}

    def _write_text(self, body: Callable[[], str], suffix: str = "") -> None:
        name = f"recir_{self.name}{suffix}_graph"
        status = self.owner.write(name, body)
        if status is not None:
            self.artifacts[name] = status

    def _write_graph(self, gm: fx.GraphModule, suffix: str = "") -> GraphDump:
        dump = GraphDump(gm)
        self._write_text(dump.text, suffix)
        return dump


class GraphStage(Stage):
    """
    Base class for all stages that write one graph, as ``recir_{name}_graph``.

    Its manifest adds the graph's block: counts, row hashes and collective
    census. This can be extended to a new snapshot point by subclassing and
    setting ``name``; the stage's RecIR code records the graph it produced.
    """

    def __init__(self, owner: TlparseCompile) -> None:
        super().__init__(owner)
        self._dump: GraphDump | None = None

    def record(self, gm: fx.GraphModule) -> None:
        """Write ``gm``, the graph the stage's RecIR code produced, and end the stage."""
        self._dump = self._write_graph(gm)
        self.end(None)

    def _fields(self) -> dict[str, Any]:
        return {} if self._dump is None else self._dump.fields(collectives=True)


class TraceStage(Stage):
    """
    The trace stage: the graph ``recir.trace`` returned.

    This class is responsible for writing that graph's text, taken when the
    trace ended, as ``recir_trace_graph``, so no edit a caller makes ahead of
    compiling shows in it; its manifest adds the fields taken with it.
    """

    name = "trace"

    def __init__(self, owner: TlparseCompile) -> None:
        super().__init__(owner)
        self._trace_fields: dict[str, Any] = {}

    def record(self, text: str, fields: dict[str, Any]) -> None:
        """Write ``text``, the traced graph's text, and end the stage with ``fields`` in its manifest."""
        self._trace_fields = fields
        self._write_text(lambda: text)
        self.end(None)

    def _fields(self) -> dict[str, Any]:
        return self._trace_fields


class AssemblyStage(GraphStage):
    """
    The assembly stage: the flattened graph ``compile_from_gm`` assembles.

    This class is responsible for writing that graph as ``recir_assembly_graph``:
    the graph ``compile_from_gm`` hands to the backend's stages, flattened, after
    RecIR's internal transforms and the caller's region-transform finalize, and
    before the backend's optimize passes and any pipeline split.
    """

    name = "assembly"


class PipelineStage(GraphStage):
    """
    The pipeline stage: the split graph ``build_pipeline`` returns.

    This class is responsible for writing that graph as ``recir_pipeline_graph``;
    the text holds each stage body as a nested module. It is recorded only when
    the compile's pipeline plan split the graph: a compile with no plan, or a
    steady-state plan, which flattens the graph instead, records none.
    """

    name = "pipeline"


class DispatchStage(Stage):
    """
    The dispatch stage: the graphs of one backend dispatch, written as lowering goes.

    This class is responsible for writing the dispatch's graphs, named after
    the input part's ``label``: each input's as
    ``recir_dispatch_input_{label}_graph`` before it is lowered, since partial
    lowering edits it in place; each subgraph carved from it as
    ``recir_dispatch_{label}_{subgraph}_graph``, named like its build; and the
    parent left after carving as ``recir_dispatch_lowered_{label}_graph``. It
    is completed when the dispatch ends without raising.
    """

    name = "dispatch"

    def add_input(self, part: Part, gm: fx.GraphModule) -> None:
        """Write ``gm``, the input ``part``, before it is lowered."""
        self._write_graph(gm, f"_input_{part.label}")

    def add_subgraph(self, part: Part, region: CarvedRegion) -> None:
        """Write the graph of ``region``, just carved from the input ``part``."""
        self._write_graph(region.gm, f"_{part.label}_{region.name}")

    def add_lowered(self, part: Part, lowered: nn.Module) -> None:
        """Write the parent left after lowering the input ``part``, if lowering returned a graph."""
        if isinstance(lowered, fx.GraphModule):
            self._write_graph(lowered, f"_lowered_{part.label}")
