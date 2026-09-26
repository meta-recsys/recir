# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Background-thread prefetching iterator.

Used by the rankfm trainer to overlap per-step batch build / data load with the
previous step's on-device compute, so the host input pipeline stops stalling the
device between steps. Stdlib-only (no torch / jax) so it is cheap to unit-test.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Any, Iterator, Optional

logger: logging.Logger = logging.getLogger(__name__)


class PrefetchingIterator:
    """Produce items from ``source`` on a background thread, up to ``depth``
    ahead.

    The producer thread runs the wrapped iterator and enqueues results; the
    consumer pulls them via the iterator protocol. Because torch CPU kernels and
    JAX device-waits both release the GIL, the producer makes real progress while
    the main thread is busy with a training step.

    Semantics:
      * Order-preserving: items come out in the source's order.
      * Sentinel-terminated: ``StopIteration`` once the source is exhausted.
      * Exception-propagating: an exception raised by the source is re-raised on
        the consumer thread (after any already-queued items are drained).
      * ``depth`` bounds how far ahead the producer may run (back-pressure via a
        bounded queue), so memory stays O(depth) batches.
    """

    _DONE: Any = object()
    _WAIT_WARN_SECS = 30.0

    def __init__(self, source: Iterator[Any], depth: int = 1) -> None:
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        self._source = source
        self._queue: "queue.Queue[Any]" = queue.Queue(maxsize=depth)
        self._exc: Optional[BaseException] = None
        self._closed: bool = False
        self._thread = threading.Thread(
            target=self._produce, name="batch-prefetch", daemon=True
        )
        self._thread.start()

    def _produce(self) -> None:
        try:
            for item in self._source:
                if self._closed:
                    break
                self._queue.put(item)
        except Exception as e:
            # Surfaced to (re-raised on) the consumer thread in __next__.
            self._exc = e
        finally:
            self._queue.put(self._DONE)

    def __iter__(self) -> "PrefetchingIterator":
        return self

    def __next__(self) -> Any:
        item = self._get_logging_long_waits()
        if item is self._DONE:
            if self._exc is not None:
                raise self._exc
            raise StopIteration
        return item

    def _get_logging_long_waits(self) -> Any:
        waited = 0.0
        while True:
            try:
                return self._queue.get(timeout=_WAIT_WARN_SECS)
            except queue.Empty:
                waited += _WAIT_WARN_SECS
                logger.warning(
                    "batch prefetch has had nothing to hand out for %.0fs "
                    "(producer alive=%s); this rank is input-starved",
                    waited,
                    self._thread.is_alive(),
                )

    def close(self) -> None:
        """Signal the producer to stop and drain the queue so its thread exits.

        Idempotent. Safe to call from a ``finally`` after partial consumption
        (e.g. when the training loop breaks early on pipe exhaustion).
        """
        self._closed = True
        # Drain until the producer thread observes _closed and exits. A single
        # drain can race: the producer may put one more item (unblocking on the
        # freed slot) and then block again on a full queue before it checks
        # _closed, so loop draining + joining until the thread terminates.
        while self._thread.is_alive():
            try:
                while True:
                    self._queue.get_nowait()
            except queue.Empty:
                pass
            self._thread.join(timeout=0.05)

    def __enter__(self) -> "PrefetchingIterator":
        return self

    def __exit__(self, *_exc_info: Any) -> None:
        self.close()
