# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Compile ids in tlparse, the stages created for them, and the graphs written under them."""

from __future__ import annotations

import inspect
import sys
import time
import traceback
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
from types import FrameType
from typing import Any, Callable, Iterator, Self, TypeVar

from recir.observability.graph_dump import GraphDump
from recir.observability.guard import contained
from recir.observability.settings import Settings
from recir.observability.stages import GraphStage, Stage, TraceStage
from torch import fx
from torch._guards import compile_context, CompileContext, CompileId
from torch._logging import trace_structured
from torch._logging._internal import trace_log
from torch._logging.structured import from_traceback

_S = TypeVar("_S", bound=Stage)
_C = TypeVar("_C", bound="TlparseCompile")

# recir packages that call into RecIR like any user, as test modules anywhere
# in recir do; every other recir module is RecIR's own.
_CALLERS: frozenset[str] = frozenset({"examples", "fb"})


def _is_internal(frame: FrameType) -> bool:
    # RecIR's own frames and the call plumbing between them: context managers,
    # torch (nn.Module and FX calls, wrappers like DDP or checkpoint) and FX's
    # generated forwards, which run with no module name.
    if "__name__" not in frame.f_globals:
        return True
    module = frame.f_globals["__name__"]
    if module in ("contextlib", "torch") or module.startswith("torch."):
        return True
    path = module.split(".")
    if path[0] != "recir":
        return False
    if len(path) > 1 and path[1] in _CALLERS:
        return False
    return not any(
        name in ("tests", "test") or name.startswith("test_") for name in path[1:]
    )


def snapshots_on() -> bool:
    """Whether to snapshot now: the tier is on and a structured-trace handler is attached."""
    return Settings.load().enabled and TlparseCompile.has_sink()


def code_location(fn: Callable[..., Any]) -> tuple[str, int] | None:
    """Where ``fn`` is defined, as ``(filename, first line)``; None for a callable without Python code."""
    code = getattr(inspect.unwrap(fn), "__code__", None)
    return None if code is None else (code.co_filename, code.co_firstlineno)


@dataclass(frozen=True)
class TraceRecord:
    """
    What a step compile keeps of its trace, taken when the trace ends: the
    traced graph's text and node counts, and where the traced function is
    defined. Plain data, so it pickles with the traced graph.
    """

    text: str
    counts: dict[str, int]
    code_location: tuple[str, int] | None

    @classmethod
    def of(cls, traced: Callable[..., Any], gm: fx.GraphModule) -> TraceRecord:
        """The record of ``gm``, traced from ``traced``."""
        dump = GraphDump(gm)
        return cls(dump.text(), dump.counts, code_location(traced))


class TlparseCompile:
    """
    Base class for all tlparse compiles: one compile id in tlparse.

    This class is responsible for reserving the compile id, placing it in
    tlparse's stack trie with ``dynamo_start``, writing payloads under the id,
    and closing it with ``compilation_metrics``, which color its stack-trie
    link green or red. A write is skipped when no trace handler is attached,
    and a write fault is logged rather than raised. It holds all of RecIR's
    contact with torch's structured-trace internals.

    The id is reserved on construction; the compile appears in tlparse when
    its work starts inside ``compiling()``. ``counts`` are the node counts its
    ``compilation_metrics`` report, and ``code_location`` is where the compiled
    function is defined.

    This can be extended to another kind of compile by subclassing and
    providing ``name`` and ``_fail_reason()``, and by overriding ``_setup``.
    """

    def __init__(
        self, counts: dict[str, int], code_location: tuple[str, int] | None
    ) -> None:
        self.frame_id: int = self._reserve_frame_id()
        self._counts = counts
        self._code_location = code_location
        self._start_s: float = 0.0

    @staticmethod
    def has_sink() -> bool:
        """
        Whether a structured-trace handler is attached. torch's lazy handler
        counts until its first emit finds no sink to write to.
        """
        return bool(trace_log.handlers)

    @property
    def name(self) -> str:
        """
        The name shown in tlparse's stack trie.
        Subclasses must override this property.
        """
        raise NotImplementedError

    def _fail_reason(self, error: BaseException) -> str:
        raise NotImplementedError

    @contextmanager
    def compiling(self) -> Iterator[Self | None]:
        """
        Run the compile's work inside this context, which gives the compile, or
        None when it could not start. A started compile is closed on exit with
        ``compilation_metrics``, raised or not; after a fault while it sets up
        or starts, the work runs as no compile and nothing is closed. An
        exception from the work passes through unchanged.
        """
        setup = ExitStack()
        started = False
        with contained(f"{self.name} not started"):
            self._setup(setup)
            self._start()
            started = True
        if not started:
            with contained(f"{self.name} setup not undone"):
                setup.close()
            yield None
            return
        self._start_s = time.time()
        error: BaseException | None = None
        try:
            yield self
        except BaseException as e:
            error = e
            raise
        finally:
            with contained(f"{self.name} setup not undone"):
                setup.close()
            with contained(f"{self.name} not finished"):
                self._finish(error)

    def _setup(self, stack: ExitStack) -> None:
        # What must be in force while the work runs, entered before dynamo_start
        # with its undo on stack; the undo runs before compilation_metrics.
        pass

    def _start(self) -> None:
        # Emit dynamo_start, placing this compile in tlparse's stack trie under
        # the last call from outside RecIR into it.
        summaries = self._stack()
        self._emit("dynamo_start", lambda: {"stack": from_traceback(summaries)})

    def _finish(self, error: BaseException | None) -> None:
        counts = self._counts
        filename, firstlineno = self._code_location or (None, None)
        metrics = {
            "co_name": self.name,
            "co_filename": filename,
            "co_firstlineno": firstlineno,
            "graph_node_count": counts.get("nodes"),
            "graph_op_count": counts.get("call_function", 0)
            + counts.get("call_module", 0),
            "graph_input_count": counts.get("placeholder"),
            "start_time": self._start_s,
            "entire_frame_compile_time_s": time.time() - self._start_s,
            "fail_type": type(error).__name__ if error is not None else None,
            "fail_reason": self._fail_reason(error) if error is not None else None,
            # tlparse reads a missing list as a restart.
            "restart_reasons": [],
        }
        self._emit("compilation_metrics", lambda: metrics)

    def _emit(
        self,
        event: str,
        metadata: Callable[[], dict[str, Any]],
        payload: Callable[[], str | None] = lambda: None,
    ) -> None:
        """
        Emit one structured-trace event under this compile id.

        The id is pushed around the emit only, so PT2 books none of RecIR's own
        work under it; a build holds it for its whole compile in ``_setup``.
        """
        with compile_context(self._compile_context()):
            trace_structured(
                event, metadata, payload_fn=payload, record_logging_overhead=False
            )

    def _compile_context(self) -> CompileContext:
        return CompileContext(CompileId(frame_id=self.frame_id, frame_compile_id=0))

    def write(self, name: str, body: Callable[[], str]) -> None:
        """
        Write the text payload ``name`` under this compile id, unless no trace
        handler is attached. ``body`` is only called when the payload is
        written, and a fault, formatting included, is logged and never raised.
        """
        if not self.has_sink():
            return
        with contained(f"{name} not written"):
            text = body()
            self._emit(
                "artifact",
                lambda: {"name": name, "encoding": "string"},
                lambda: text,
            )

    @staticmethod
    def _reserve_frame_id() -> int:
        # Dynamo's frame counter, so RecIR's compile ids never collide with a
        # torch.compile frame's (until torch._dynamo.reset() restarts it).
        # TODO: give RecIR its own compile-id namespace once tlparse can group events by it.
        from torch._dynamo import convert_frame

        counter: Any = convert_frame
        # Dynamo's frame-compile path bumps the counter under this lock too.
        with convert_frame.compile_lock:
            frame_id = counter.FRAME_COUNTER
            counter.FRAME_COUNTER = frame_id + 1
        return frame_id

    def _stack(self) -> list[traceback.FrameSummary]:
        # The stack up to the last call from outside RecIR into it, then a frame
        # named after this compile, as Dynamo appends the compiled function's frame.
        frame: FrameType | None = sys._getframe(1)
        while frame is not None and _is_internal(frame):
            frame = frame.f_back
        summaries = traceback.extract_stack(frame) if frame is not None else []
        summaries.append(traceback.FrameSummary("recir", 0, self.name, line=""))
        return summaries


@contextmanager
def snapshotted(create: Callable[[], _C], what: str) -> Iterator[_C | None]:
    """
    Run the work inside this context as the compile ``create`` makes, which it
    gives, while snapshots are on; otherwise, or after a snapshot fault, the
    work runs as no compile and it gives None.
    """
    made = None
    with contained(f"{what} not created"):
        if snapshots_on():
            made = create()
    if made is None:
        yield None
        return
    with made.compiling() as started:
        yield started


class StepCompile(TlparseCompile):
    """
    The step compile: one traced graph's ``compile_from_gm``, with the graph
    ``recir.trace`` returned and the RecIR stages that follow it.

    This class is responsible for snapshotting the step's stages as RecIR's
    code reaches them, starting with the trace stage from ``trace`` when the
    compile starts. It appears in tlparse as "recir step compile", at the
    traced function's definition, with the traced graph's node counts.
    """

    def __init__(self, trace: TraceRecord) -> None:
        super().__init__(trace.counts, trace.code_location)
        self._trace_text = trace.text
        self.stages: list[Stage] = []

    @property
    def name(self) -> str:
        """The name shown in tlparse's stack trie: "recir step compile"."""
        return "recir step compile"

    def record_stage(self, stage_cls: type[GraphStage], gm: fx.GraphModule) -> None:
        """Snapshot a new ``stage_cls`` stage from ``gm``, the graph its RecIR code produced."""
        self.begin(stage_cls).record(gm)

    def begin(self, stage_cls: type[_S]) -> _S:
        """
        Create a ``stage_cls`` stage of this compile, for its RecIR code to
        write to as it runs and end; stages are kept in the order begun.
        """
        stage = stage_cls(self)
        self.stages.append(stage)
        return stage

    def _start(self) -> None:
        super()._start()
        self.begin(TraceStage).record(self._trace_text)

    def _fail_reason(self, error: BaseException) -> str:
        completed = [stage.name for stage in self.stages if stage.completed]
        if not completed:
            return "raised before any RecIR stage completed"
        return f"raised after RecIR stage {completed[-1]}"
