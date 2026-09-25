# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Tests for the background-thread batch prefetcher used by the rankfm trainer."""

import itertools
import time
import unittest
from typing import Iterator, List

from recir.examples.rankfm_mvai.prefetch import PrefetchingIterator


class PrefetchingIteratorTest(unittest.TestCase):
    def test_preserves_order_and_values(self) -> None:
        out = list(PrefetchingIterator(iter(range(20)), depth=3))
        self.assertEqual(out, list(range(20)))

    def test_empty_source(self) -> None:
        out = list(PrefetchingIterator(iter([]), depth=1))
        self.assertEqual(out, [])

    def test_depth_one(self) -> None:
        out = list(PrefetchingIterator(iter([1, 2, 3]), depth=1))
        self.assertEqual(out, [1, 2, 3])

    def test_invalid_depth_raises(self) -> None:
        with self.assertRaises(ValueError):
            PrefetchingIterator(iter([1]), depth=0)

    def test_propagates_source_exception_after_draining(self) -> None:
        def boom() -> Iterator[int]:
            yield 1
            yield 2
            raise RuntimeError("kaboom")

        it = PrefetchingIterator(boom(), depth=1)
        got = []
        with self.assertRaises(RuntimeError):
            for x in it:
                got.append(x)
        # items produced before the exception are still delivered
        self.assertEqual(got, [1, 2])

    def test_close_is_idempotent(self) -> None:
        def infinite() -> Iterator[int]:
            i = 0
            while True:
                yield i
                i += 1

        it = PrefetchingIterator(infinite(), depth=2)
        self.assertEqual(next(it), 0)
        it.close()
        it.close()  # idempotent, must not raise

    def test_context_manager_closes(self) -> None:
        def infinite() -> Iterator[int]:
            i = 0
            while True:
                yield i
                i += 1

        with PrefetchingIterator(infinite(), depth=1) as it:
            self.assertEqual(next(it), 0)
        # exiting the context closes without error; closed flag is set
        self.assertTrue(it._closed)

    def test_consumer_cursor_resumes_at_the_next_unread_batch(self) -> None:
        cursor = 0
        with PrefetchingIterator(iter(range(20)), depth=4) as prefetched:
            consumed = []
            for _ in range(3):
                consumed.append(next(prefetched))
                cursor += 1

        resumed = iter(range(cursor, 20))
        self.assertEqual(consumed, [0, 1, 2])
        self.assertEqual(next(resumed), 3)

    def test_resume_replays_what_was_fetched_but_never_trained(self) -> None:
        """Why the resume index is the trained count, not the fetched one.

        The prefetch buffer and the pipeline's 1-step lookahead both pull ahead,
        so when a run dies several batches have left the iterator without their
        optimizer step finishing. Resuming at the trained count redoes them;
        resuming at the fetched count would drop them from the epoch.
        """
        source = PrefetchingIterator(iter(range(20)), depth=3)
        fetched = [next(source) for _ in range(5)]
        source.close()
        trained_steps = 2  # the other three were in flight when the run died

        resumed = list(itertools.islice(iter(range(trained_steps, 20)), 3))

        self.assertEqual(fetched, [0, 1, 2, 3, 4])
        self.assertEqual(resumed, [2, 3, 4])

    def test_the_producer_runs_ahead_of_the_consumer(self) -> None:
        """Why a checkpointed data cursor must be advanced on the CONSUMER side.

        Running ahead is the whole point of the class, but it means "batches
        built" and "batches trained on" are different numbers, differing by up
        to ``depth``. ``_RankFMIterableDataset.__iter__`` therefore advances
        ``_batch_index`` around the ``__next__`` that returns a batch, not
        inside the wrapped generator that produces it: a cursor written by the
        producer would checkpoint a position past what the model has actually
        seen, and the resumed run would silently skip those batches -- no error,
        just a hole in the data.
        """
        produced: List[int] = []

        def source() -> Iterator[int]:
            for i in range(100):
                produced.append(i)
                yield i

        depth = 4
        with PrefetchingIterator(source(), depth=depth) as it:
            self.assertEqual(next(it), 0)
            # Let the producer fill the queue; it then blocks on a full put.
            deadline = time.monotonic() + 30
            while len(produced) < depth + 1 and time.monotonic() < deadline:
                time.sleep(0.01)
            consumed = 1
            self.assertGreater(
                len(produced),
                consumed,
                "producer never ran ahead, so this invariant is untested",
            )
            # Bounded by the queue: depth queued + one handed out + one in the
            # producer's hand, blocked on put.
            self.assertLessEqual(len(produced), consumed + depth + 1)
