# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Inductor's subgraph builds: the IR each one records."""

from __future__ import annotations

import functools
import logging
import sys
from contextlib import AbstractContextManager, ExitStack
from typing import Any, ClassVar

from recir.observability.builds import SubgraphBuild
from recir.observability.guard import contained
from recir.observability.settings import Settings

# Each dump's logger, by its public name in torch._inductor.debug.
_IR_LOGGERS: dict[str, str] = {
    "pre": "ir_pre_fusion_log",
    "post": "ir_post_fusion_log",
}


class _Collect(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.chunks: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.chunks.append(record.getMessage())


def _restore_logger(
    target: logging.Logger, handler: logging.Handler, saved: tuple[int, bool]
) -> None:
    target.removeHandler(handler)
    target.setLevel(saved[0])
    target.propagate = saved[1]


class IrCapture:
    """
    Inductor's IR before and after fusion, captured from inside one build.

    This class is responsible for turning on Inductor's two IR fusion loggers
    for the build, keeping what they log, and registering how to put them
    back on the build's undo stack. A user who already enabled a logger for
    INFO still sees its output. When the IR formatter fails on a dump whose logger only
    the capture enabled, the dump keeps an error marker in place of that IR
    instead of failing the build. The loggers and the formatter are
    process-global, so only one capture runs at a time: ``active()`` says
    whether one is running.
    """

    _active: ClassVar[bool] = False

    def __init__(self) -> None:
        self._collect: dict[str, _Collect] = {k: _Collect() for k in _IR_LOGGERS}
        # The dumps whose loggers only this capture enabled.
        self._owned: set[str] = set()

    @classmethod
    def active(cls) -> bool:
        """Whether a capture is running in this process."""
        return cls._active

    def begin(self, undo: ExitStack) -> None:
        """Start capturing, registering on ``undo`` how to put back each thing before changing it."""
        from torch._inductor import debug
        from torch._inductor.debug import DebugFormatter

        undo.callback(setattr, IrCapture, "_active", False)
        IrCapture._active = True

        saved_write_ir = DebugFormatter.__dict__["_write_ir"]
        write_ir = DebugFormatter._write_ir
        # Only the logger's call in each log_ir_*_fusion is guarded; another
        # caller, such as the debug dir's own writer, raises as it would anyway.
        callers = {
            debug.log_ir_pre_fusion.__code__: "pre",
            debug.log_ir_post_fusion.__code__: "post",
        }

        def guarded_write_ir(nodes: Any) -> str:
            dump = callers.get(sys._getframe(1).f_code)
            if dump is None or dump not in self._owned:
                return write_ir(nodes)
            try:
                return write_ir(nodes)
            except Exception as e:
                return f"<recir snapshot: IR not formatted: {e!r}>"

        undo.callback(setattr, DebugFormatter, "_write_ir", saved_write_ir)
        DebugFormatter._write_ir = staticmethod(guarded_write_ir)
        for key, name in _IR_LOGGERS.items():
            ir_logger = getattr(debug, name)
            saved = (ir_logger.level, ir_logger.propagate)
            undo.callback(_restore_logger, ir_logger, self._collect[key], saved)
            if not ir_logger.isEnabledFor(logging.INFO):
                self._owned.add(key)
                ir_logger.setLevel(logging.INFO)
                ir_logger.propagate = False
            ir_logger.addHandler(self._collect[key])

    def dumps(self) -> list[str]:
        """The dumps (``pre``, ``post``) that logged IR."""
        return [k for k, c in self._collect.items() if c.chunks]

    def text(self, dump: str) -> str:
        """The IR one dump logged."""
        return "\n".join(self._collect[dump].chunks)


def _without_bundled_autotune_cache() -> AbstractContextManager[object]:
    # Inductor reads and writes its bundled autotune cache only inside a compile
    # context, which a build has only because it is snapshotted. The patch is
    # process-global, so this assumes one build runs at a time.
    import torch._inductor.config as inductor_config

    return inductor_config.patch(bundled_autotune_remote_cache=False)


class InductorBuild(SubgraphBuild):
    """
    One carved subgraph's Inductor build.

    This class is responsible for what an Inductor build adds to the neutral
    build: keeping Inductor's bundled autotune cache off while the build's
    compile id is pushed, so the build autotunes as it would without
    snapshots; and at tier full, with a trace handler to write it, capturing
    Inductor's IR from inside the compile and writing each dump as
    ``recir_build_ir_pre_fusion`` and ``recir_build_ir_post_fusion`` into the
    build's compile id. A build that raised part way keeps the IR it logged
    before raising; a build served from Inductor's FX graph cache runs no
    scheduler, so it logs none, and a build that starts while another
    build's capture runs captures none.
    """

    def _setup(self, stack: ExitStack) -> None:
        super()._setup(stack)
        stack.enter_context(_without_bundled_autotune_cache())
        if Settings.load().full and not IrCapture.active():
            with contained(f"{self.name} IR not captured"):
                self._capture_ir(stack)

    def _capture_ir(self, stack: ExitStack) -> None:
        # The record runs after the capture's undo and before the build closes;
        # a fault part way through starting undoes what started, captures nothing.
        capture = IrCapture()
        with ExitStack() as undo:
            capture.begin(undo)
            stack.callback(self._record, capture)
            stack.enter_context(undo.pop_all())

    def _record(self, capture: IrCapture) -> None:
        for dump in capture.dumps():
            self.write(
                f"recir_build_ir_{dump}_fusion", functools.partial(capture.text, dump)
            )
