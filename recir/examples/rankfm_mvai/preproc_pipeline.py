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
import sys
import threading
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Generic, List, Optional, Tuple, TypeVar

logger: logging.Logger = logging.getLogger(__name__)

_T = TypeVar("_T")


@dataclass
class _WorkRecord:
    """One queued unit of work, for attributing a `drain` timeout."""

    label: str
    queued_at: float
    started: bool = False
    done: bool = False


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
    _WARN_SECS = 60.0
    _FATAL_SECS = 240.0

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
        # Count of items queued but not yet finished, guarded by _lock.
        self._unfinished_count: int = 0
        # Per-item diagnostic records for the items `_unfinished_count` counts,
        # guarded by the same lock.
        self._work_log: List[_WorkRecord] = []
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
            self._mark_started()
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

    def _mark_started(self) -> None:
        """Flag the item the worker just picked up as running."""
        with self._lock:
            for entry in self._work_log:
                if not entry.started:
                    entry.started = True
                    return

    def _finish_one(self) -> None:
        with self._lock:
            self._unfinished_count -= 1
            for entry in self._work_log:
                if entry.started and not entry.done:
                    entry.done = True
                    break
            if self._unfinished_count == 0:
                self._idle.set()
                self._work_log.clear()

    def _enqueue(
        self, item: Tuple[Optional[int], Callable[[], Any]], label: str
    ) -> bool:
        with self._lock:
            try:
                self._inbox.put_nowait(item)
            except queue.Full:
                return False
            self._unfinished_count += 1
            self._work_log.append(_WorkRecord(label, time.monotonic()))
            self._idle.clear()
        return True

    def queue(
        self,
        work: Callable[[], Any],
        step: Optional[int] = None,
        label: str = "work",
    ) -> bool:
        """Queue ``work`` on the worker. Returns False if the queue is full."""
        if self._closed:
            return False
        return self._enqueue((step, work), label)

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
        recovered = False
        if not self._idle.wait(timeout=self._WARN_SECS):
            logger.warning(
                "preprocessor worker still busy after %.0fs -- %s",
                self._WARN_SECS,
                self._busy_state(),
            )
            remaining = self._FATAL_SECS - self._WARN_SECS
            if not self._idle.wait(timeout=remaining):
                raise RuntimeError(
                    f"preprocessor worker did not finish within "
                    f"{self._FATAL_SECS:.0f}s -- {self._busy_state()}"
                )
            recovered = True
        with self._lock:
            if self._error is not None:
                err = self._error
                self._error = None
                raise RuntimeError("preprocessor producer thread failed") from err
        if recovered:
            logger.warning("preprocessor worker recovered; continuing")

    def _busy_state(self) -> str:
        """Where the worker actually is, captured while it is still stuck."""
        with self._lock:
            unfinished, ready = self._unfinished_count, sorted(self._ready)
            now = time.monotonic()
            items = [
                (entry.label, now - entry.queued_at, entry.started, entry.done)
                for entry in self._work_log
            ]
        rendered = []
        for label, age, started, done in reversed(items):
            if done:
                rendered.append(f"{label} done")
            elif started:
                rendered.append(f"{label} queued {age:.1f}s ago, still running")
            else:
                rendered.append(f"{label} queued {age:.1f}s ago, waiting")
        work_log = "; ".join(rendered) if rendered else "(none)"
        try:
            frame = sys._current_frames().get(self._thread.ident or -1)
            where = (
                "".join(traceback.format_stack(frame)[-6:]) if frame else "(no frame)"
            )
        except Exception as exc:
            where = f"(stack unavailable: {exc!r})"
        return (
            f"unfinished={unfinished} inbox={self._inbox.qsize()} ready={ready} "
            f"alive={self._thread.is_alive()}\nwork log: {work_log}\n"
            f"worker stack:\n{where}"
        )

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
