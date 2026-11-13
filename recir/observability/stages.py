# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""The stages a step compile snapshots, and what each one writes."""

from __future__ import annotations

from typing import Callable, ClassVar, TYPE_CHECKING

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
