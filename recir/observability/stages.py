# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""The stages a step compile snapshots, and what each one writes."""

from __future__ import annotations

from typing import Callable, ClassVar, TYPE_CHECKING

from recir.observability.graph_dump import GraphDump
from torch import fx

if TYPE_CHECKING:
    from recir.observability.compiles import TlparseCompile


class Stage:
    """
    Base class for all snapshot stages: one snapshot point in a compile.

    This class is responsible for naming the stage's artifacts. It writes
    only through its owning compile, whose write never raises, and does not
    decide stage order.

    This can be extended to a new snapshot point by subclassing, setting
    ``name``, and adding the method its RecIR code records through.
    """

    name: ClassVar[str]

    def __init__(self, owner: TlparseCompile) -> None:
        self.owner = owner

    def _write_text(self, body: Callable[[], str]) -> None:
        self.owner.write(f"recir_{self.name}_graph", body)


class GraphStage(Stage):
    """
    Base class for all stages that write one graph, as ``recir_{name}_graph``.

    This can be extended to a new snapshot point by subclassing and setting
    ``name``; the stage's RecIR code records the graph it produced.
    """

    def record(self, gm: fx.GraphModule) -> None:
        """Write ``gm``, the graph the stage's RecIR code produced."""
        self._write_text(GraphDump(gm).text)


class TraceStage(Stage):
    """
    The trace stage: the graph ``recir.trace`` returned.

    This class is responsible for writing that graph's text, taken when the
    trace ended, as ``recir_trace_graph``, so no edit a caller makes ahead of
    compiling shows in it.
    """

    name = "trace"

    def record(self, text: str) -> None:
        """Write ``text``, the traced graph's text."""
        self._write_text(lambda: text)


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
