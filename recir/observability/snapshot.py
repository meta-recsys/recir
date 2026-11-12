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
from contextvars import ContextVar
from typing import Any, Callable, Iterator

import torch.utils._pytree as pytree
from recir.lowering.lowering_inputs import LoweringObserver
from recir.observability.compiles import (
    snapshots_on,
    snapshotted,
    StepCompile,
    TraceRecord,
)
from recir.observability.dispatch import ObservedLowering
from recir.observability.guard import contained
from recir.observability.stages import DispatchStage, GraphStage
from torch import fx


# The observer of the backend dispatch this thread is running: a backend's own
# lowering cannot be handed it through the public Backend API.
_observer: ContextVar[LoweringObserver | None] = ContextVar(
    "recir_dispatch_observer", default=None
)


def keep_trace(
    traced: Callable[..., Any],
    gm: fx.GraphModule,
    *,
    params: dict[str, Any],
    buffers: dict[str, Any],
    named_states: dict[str, Any],
    trace_s: float,
) -> TraceRecord | None:
    """
    What the step compiles of ``gm``, traced from ``traced`` in ``trace_s``
    seconds, need from the trace, taken now so no later edit to ``gm`` shows
    in it; None when snapshots are off. ``params``, ``buffers`` and
    ``named_states`` (optimizer states) are the state the trace lifted into
    ``gm``'s placeholders. Never raises.
    """
    with contained("trace not kept"):
        if snapshots_on():
            state = {
                "params": len(pytree.tree_leaves(params)),
                "buffers": len(pytree.tree_leaves(buffers)),
                "optimizer_states": len(pytree.tree_leaves(named_states)),
            }
            return TraceRecord.of(traced, gm, state=state, trace_s=trace_s)
    return None


@contextmanager
def compiling_step(trace: TraceRecord | None) -> Iterator[StepCompile | None]:
    """
    Run ``compile_from_gm`` inside this context as one step compile, which it
    gives, starting with the graph ``trace`` kept and closed with
    ``compilation_metrics`` on exit. Without a trace, or when snapshots are
    off, or after a snapshot fault, no step compile is opened,
    ``compile_from_gm`` runs as without snapshots, and it gives None.
    Exceptions from ``compile_from_gm`` pass through unchanged.
    """
    if trace is None:
        yield None
        return
    with snapshotted(lambda: StepCompile(trace), "step compile") as step:
        yield step


def record_stage(
    step: StepCompile | None, stage_cls: type[GraphStage], gm: fx.GraphModule
) -> None:
    """Snapshot a ``stage_cls`` stage from ``gm`` in ``step``, if any. Never raises."""
    with contained(
        f"{getattr(stage_cls, 'name', stage_cls.__name__)} stage not recorded"
    ):
        if step is not None:
            step.record_stage(stage_cls, gm)


@contextmanager
def dispatching(step: StepCompile | None, backend: str) -> Iterator[None]:
    """
    Run ``step``'s backend dispatch inside this context as its dispatch stage,
    observed through ``lowering_observer``, and end the stage with how the
    dispatch ended; ``backend`` names the backend that builds the subgraphs
    the dispatch carves. Without a step, nothing is observed. Exceptions from
    the dispatch pass through unchanged.
    """
    stage = None
    observer = None
    with contained("dispatch stage not begun"):
        if step is not None:
            stage = step.begin(DispatchStage)
            observer = ObservedLowering(stage, backend)
    token = _observer.set(observer)
    error: BaseException | None = None
    try:
        yield
    except BaseException as e:
        error = e
        raise
    finally:
        _observer.reset(token)
        if stage is not None:
            with contained("dispatch stage not ended"):
                stage.end(error)


def lowering_observer() -> LoweringObserver | None:
    """The observer of the backend dispatch this thread is running, if any."""
    return _observer.get()
