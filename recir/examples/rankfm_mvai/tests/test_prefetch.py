# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Tests for the background-thread batch prefetcher used by the rankfm trainer."""

import unittest
from typing import Iterator

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
