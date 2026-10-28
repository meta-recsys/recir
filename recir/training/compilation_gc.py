# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Keep the compiled training heap out of runtime cyclic-GC scans."""

from __future__ import annotations

import contextlib
import contextvars
import gc
import logging
import time
from collections.abc import Iterator

logger: logging.Logger = logging.getLogger(__name__)


class _CompilationHeap:
    def __init__(self) -> None:
        self.frozen = False
        self._thaw_on_close = False

    def freeze(self) -> None:
        if self.frozen or not gc.isenabled():
            return
        self._thaw_on_close = gc.get_freeze_count() == 0
        started = time.monotonic()
        collected = gc.collect()
        gc.freeze()
        self.frozen = True
        logger.info(
            "Frozen compiled Python heap: objects=%d collected=%d seconds=%.3f",
            gc.get_freeze_count(),
            collected,
            time.monotonic() - started,
        )

    def close(self) -> None:
        if self.frozen:
            if self._thaw_on_close:
                gc.unfreeze()
            self.frozen = False


_HEAP: contextvars.ContextVar[_CompilationHeap | None] = contextvars.ContextVar(
    "recir_compilation_heap", default=None
)


def freeze_compilation_heap() -> None:
    heap = _HEAP.get()
    if heap is not None:
        heap.freeze()


@contextlib.contextmanager
def compilation_gc_scope(enabled: bool) -> Iterator[None]:
    heap = _CompilationHeap() if enabled else None
    token = _HEAP.set(heap)
    try:
        yield
    finally:
        _HEAP.reset(token)
        if heap is not None:
            heap.close()
