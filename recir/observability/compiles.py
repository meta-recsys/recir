# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Compile ids in tlparse, the stages created for them, and the graphs written under them."""

from __future__ import annotations

import functools
import inspect
import sys
import time
import traceback
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from types import FrameType
from typing import Any, Callable, Iterator, ParamSpec, Self, TypeVar

from recir.observability.graph_dump import GraphDump
from recir.observability.guard import contained
from recir.observability.stages import Stage, TraceStage
from torch import fx
from torch._guards import compile_context, CompileContext, CompileId
from torch._logging import trace_structured
from torch._logging._internal import trace_log
from torch._logging.structured import from_traceback

_Args = ParamSpec("_Args")
_R = TypeVar("_R")

# The frame that called the outermost entry point now running.
_entry_caller: ContextVar[FrameType | None] = ContextVar(
    "recir_entry_caller", default=None
)


def entry_point(fn: Callable[_Args, _R]) -> Callable[_Args, _R]:
    """
    Mark ``fn`` as a call from user code into RecIR: a compile constructed
    while it runs sits in tlparse's stack trie under its caller. Nested entry
    points keep the outermost caller.
    """

    @functools.wraps(fn)
    def entered(*args: _Args.args, **kwargs: _Args.kwargs) -> _R:
        if _entry_caller.get() is not None:
            return fn(*args, **kwargs)
        token = _entry_caller.set(sys._getframe(1))
        try:
            return fn(*args, **kwargs)
        finally:
            _entry_caller.reset(token)

    return entered


def entry_caller_stack() -> list[traceback.FrameSummary]:
    """The stack up to the call into the running ``entry_point``; empty outside one."""
    caller = _entry_caller.get()
    return traceback.extract_stack(caller) if caller is not None else []


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
    its work starts inside ``compiling()``, under ``caller_stack``, normally
    the ``entry_caller_stack()`` of the call that asked for the work.
    ``counts`` are the node counts its ``compilation_metrics`` report, and
    ``code_location`` is where the compiled function is defined.

    This can be extended to another kind of compile by subclassing and
    providing ``name`` and ``_fail_reason()``.
    """

    def __init__(
        self,
        counts: dict[str, int],
        code_location: tuple[str, int] | None,
        caller_stack: list[traceback.FrameSummary],
    ) -> None:
        self.frame_id: int = self._reserve_frame_id()
        self._caller_stack = caller_stack
        self._counts = counts
        self._code_location = code_location
        self._start_s: float = 0.0

    @staticmethod
    def has_sink() -> bool:
        """
        Whether a structured-trace handler is attached.

        torch's lazy handler stays attached until its first emit finds no sink.
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
        ``compilation_metrics``, raised or not; after a fault while it starts,
        the work runs as no compile and nothing is closed. An exception from
        the work passes through unchanged.
        """
        started = False
        with contained(f"{self.name} not started"):
            self._start()
            started = True
        if not started:
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
            with contained(f"{self.name} not finished"):
                self._finish(error)

    def _start(self) -> None:
        # Emit dynamo_start, placing this compile in tlparse's stack trie under
        # its caller's stack.
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

        The id is pushed around the emit only, so PT2 never books RecIR's own
        work as compile time.
        """
        compile_id = CompileId(frame_id=self.frame_id, frame_compile_id=0)
        with compile_context(CompileContext(compile_id)):
            trace_structured(
                event, metadata, payload_fn=payload, record_logging_overhead=False
            )

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
        # The caller's stack, then a frame named after this compile, as Dynamo
        # appends the compiled function's frame.
        frame = traceback.FrameSummary("recir", 0, self.name, line="")
        return [*self._caller_stack, frame]


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
        super().__init__(trace.counts, trace.code_location, entry_caller_stack())
        self._trace_text = trace.text
        self.stages: list[Stage] = []

    @property
    def name(self) -> str:
        """The name shown in tlparse's stack trie: "recir step compile"."""
        return "recir step compile"

    def _start(self) -> None:
        super()._start()
        stage = TraceStage(self)
        self.stages.append(stage)
        stage.record(self._trace_text)

    def _fail_reason(self, error: BaseException) -> str:
        if not self.stages:
            return "raised before any RecIR stage"
        return f"raised after RecIR stage {self.stages[-1].name}"
