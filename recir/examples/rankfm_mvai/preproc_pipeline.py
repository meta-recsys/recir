# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""One-step-ahead worker for the preprocessor graph break."""

from __future__ import annotations

import logging
import queue
import threading
from collections import OrderedDict
from typing import Any, Callable, Generic, Optional, Tuple, TypeVar

logger: logging.Logger = logging.getLogger(__name__)

_T = TypeVar("_T")


class PreprocPipeline(Generic[_T]):
    """Runs ``compute`` one step ahead on a background thread.

    ``submit`` hands the worker the work for a future step; ``take`` retrieves
    it when the graph asks, or returns ``None`` if it is not ready or is not for
    that step. The caller is responsible for the inline fallback on ``None``.

    Two slots, not one. The submit for step N+1 happens *before* the graph runs
    step N, so a fast worker would finish N+1 and clobber the not-yet-consumed
    result for N. Keeping the last two results makes the handoff independent of
    who wins that race; beyond two is pointless because the consumer can never
    be more than one step behind.
    """

    _CAPACITY = 2

    def __init__(self, name: str = "preproc-producer") -> None:
        # Items are (step, work). `step is None` means the result is discarded;
        # otherwise it is kept for `take(step)`. That is the only difference
        # between the two kinds of work this thread runs -- and it is one
        # thread, not two, because a second heavy Python thread starves the data
        # loader (see the module docstring).
        self._inbox: "queue.Queue[Optional[Tuple[Optional[int], Callable[[], Any]]]]" = queue.Queue(
            maxsize=4
        )
        self._idle = threading.Event()
        self._idle.set()
        self._lock = threading.Lock()
        # Outstanding items, guarded by _lock. `drain` waits on the count
        # reaching zero rather than on the worker looking busy: an item that has
        # been queued but not yet picked up still has to run, and a `drain` that
        # returned early there would let the caller read a result the worker has
        # not written -- which reads as a silently dropped metric update.
        self._pending: int = 0
        self._ready: "OrderedDict[int, _T]" = OrderedDict()
        self._error: Optional[BaseException] = None
        self._closed = False
        self.hits: int = 0
        self.misses: int = 0
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            item = self._inbox.get()
            if item is None:
                return
            step, work = item
            try:
                result = work()
            except Exception as e:
                # Stashed, not swallowed: `take`/`drain` re-raise it on the
                # consumer thread so the failure lands on the step that would
                # have used it instead of disappearing into this thread.
                with self._lock:
                    self._error = e
                self._finish_one()
                continue
            if step is not None:
                with self._lock:
                    self._ready[step] = result
                    while len(self._ready) > self._CAPACITY:
                        self._ready.popitem(last=False)
            self._finish_one()

    def _finish_one(self) -> None:
        with self._lock:
            self._pending -= 1
            if self._pending == 0:
                self._idle.set()

    def _enqueue(self, item: Tuple[Optional[int], Callable[[], Any]]) -> bool:
        with self._lock:
            try:
                self._inbox.put_nowait(item)
            except queue.Full:
                return False
            self._pending += 1
            self._idle.clear()
        return True

    def queue(self, work: Callable[[], Any], step: Optional[int] = None) -> bool:
        """Queue ``work`` on the worker. Returns False if the queue is full."""
        if self._closed:
            return False
        return self._enqueue((step, work))

    def take(self, step: int) -> Optional[_T]:
        """Return the precomputed result for ``step``, else ``None``."""
        with self._lock:
            if self._error is not None:
                err = self._error
                self._error = None
                raise RuntimeError("preprocessor producer thread failed") from err
            slot = self._ready.pop(step, None)
            if slot is None:
                self.misses += 1
                return None
            self.hits += 1
            return slot

    def drain(self) -> None:
        """Block until queued work has finished, re-raising worker errors."""
        if not self._idle.wait(timeout=60.0):
            raise RuntimeError(
                "preprocessor producer did not finish within 60s; it runs ~0.4s "
                "of work per step, so this is a hang, not a slow step"
            )
        with self._lock:
            if self._error is not None:
                err = self._error
                self._error = None
                raise RuntimeError("preprocessor producer thread failed") from err

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._inbox.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=5.0)

    def __enter__(self) -> "PreprocPipeline[_T]":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()
