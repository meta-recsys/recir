# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""The snapshot entry points RecIR's code calls; none raises, so a snapshot
never fails RecIR. Snapshots are on when ``RECIR_SNAPSHOT_TIER`` is not ``off``
(its default) and a structured-trace sink (``TORCH_TRACE``, or MAST's) is
attached; the classes in ``compiles`` and ``stages`` do the work.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Iterator

from recir.observability.compiles import snapshots_on, StepCompile, TraceRecord
from recir.observability.guard import contained
from recir.observability.stages import GraphStage
from torch import fx


def keep_trace(traced: Callable[..., Any], gm: fx.GraphModule) -> TraceRecord | None:
    """
    What the step compiles of ``gm``, traced from ``traced``, need from the
    trace, taken now so no later edit to ``gm`` shows in it; None when
    snapshots are off. Never raises.
    """
    with contained("trace not kept"):
        if snapshots_on():
            return TraceRecord.of(traced, gm)
    return None


@contextmanager
def compiling_step(trace: TraceRecord | None) -> Iterator[StepCompile | None]:
    """
    Run ``compile_from_gm`` inside this context as one step compile, which it
    gives, starting with the graph ``trace`` kept and closed with
    ``compilation_metrics`` on exit. Without a trace, or when snapshots are
    off, no step compile is opened, ``compile_from_gm`` runs as without
    snapshots, and it gives None. Exceptions from ``compile_from_gm`` pass
    through unchanged.
    """
    step: StepCompile | None = None
    with contained("step compile not created"):
        if trace is not None and snapshots_on():
            step = StepCompile(trace)
    if step is None:
        yield None
        return
    with step.compiling() as started:
        yield started


def record_stage(
    step: StepCompile | None, stage_cls: type[GraphStage], gm: fx.GraphModule
) -> None:
    """Snapshot a ``stage_cls`` stage from ``gm`` in ``step``, if any. Never raises."""
    with contained(
        f"{getattr(stage_cls, 'name', stage_cls.__name__)} stage not recorded"
    ):
        if step is not None:
            step.record_stage(stage_cls, gm)
