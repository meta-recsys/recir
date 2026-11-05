# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Host-side input preparation of the SparseCore tables: the lookup's COO input and
per-position slots, and the backward's exchange layout."""

import unittest
from typing import List, Tuple

import numpy as np
from recir.embedding.sparsecore_rowwise import (
    _device_inputs,
    _exchange_layout,
    _SubTable,
)

_NUM_SC = 2
_WORLD = 4
# (shard_size, rotation, row offset): 8 shards of 125 rows, so 250 rows per device.
_LAYOUT = (125, 3, 0)


def _phys(ids: np.ndarray) -> np.ndarray:
    shard_size, rotation, offset = _LAYOUT
    shards = _WORLD * _NUM_SC
    return ((ids % shards + rotation) % shards) * shard_size + ids // shards + offset


class DeviceInputsTest(unittest.TestCase):
    def test_coo_and_slots_rebuild_every_position(self) -> None:
        rng = np.random.default_rng(0)
        ids = rng.integers(0, 600, size=(2, 37))
        ids[:, ::3] = 499  # a hot id
        padded = 38
        subs = [_SubTable("t@0", "t", 0, 300), _SubTable("t@1", "t", 300, 300)]
        slots = np.empty(ids.shape, np.int32)

        for d in range(2):
            coo, sorted_uniq, inv = _device_inputs(
                ids[d], d, padded, _NUM_SC, subs, slots
            )

            rebuilt = np.full(2 * padded, -1)
            for (samples, values), sub in zip(coo, subs):
                self.assertTrue(np.all(np.diff(samples) > 0))
                self.assertTrue(np.all(rebuilt[samples] == -1))
                rebuilt[samples] = values + sub.start
            self.assertTrue(np.array_equal(rebuilt[d * padded + slots[d]], ids[d]))
            self.assertTrue(np.array_equal(sorted_uniq[inv], ids[d]))
            per_sc = np.bincount(
                (np.flatnonzero(rebuilt >= 0) - d * padded) // (padded // _NUM_SC),
                minlength=_NUM_SC,
            )
            self.assertLessEqual(per_sc.max() - per_sc.min(), 1)
            self.assertEqual(np.unique(ids[d]).size, per_sc.sum())


class ExchangeLayoutTest(unittest.TestCase):
    def _parts(
        self, sub: _SubTable
    ) -> Tuple[List[Tuple[np.ndarray, np.ndarray, int, _SubTable]], List[np.ndarray]]:
        # Two features of one stack; id 7 appears in both, 999 is outside the sub-table.
        a = np.array([7, 3, 7, 650, 12])
        b = np.array([999, 7, 640])
        parts = [
            (*np.unique(a, return_inverse=True), padded, sub)
            for a, padded in ((a, 6), (b, 4))
        ]
        return parts, [a, b]

    def test_positions_map_to_their_rows(self) -> None:
        sub = _SubTable("t@0", "t", 0, 700)
        parts, features = self._parts(sub)
        cap = 8

        dest, send_rows, fill = _exchange_layout(parts, _LAYOUT, _WORLD, _NUM_SC, cap)

        per_device = _LAYOUT[0] * _NUM_SC
        flat_rows = send_rows.reshape(-1)
        pos, seen = 0, {}
        for ids, (_, _, padded, _) in zip(features, parts):
            inside = (ids >= sub.start) & (ids < sub.start + sub.rows)
            d = dest[pos : pos + padded]
            self.assertTrue(np.all(d[len(ids) :] == _WORLD * cap))
            self.assertTrue(np.all(d[: len(ids)][~inside] == _WORLD * cap))
            got = (d[: len(ids)][inside] // cap) * per_device + flat_rows[
                d[: len(ids)][inside]
            ]
            self.assertTrue(np.array_equal(got, _phys(ids[inside] - sub.start)))
            for i, slot in zip(ids[inside], d[: len(ids)][inside]):
                self.assertEqual(seen.setdefault(int(i), int(slot)), int(slot))
            pos += padded
        self.assertEqual(len(seen), int((send_rows >= 0).sum()))
        self.assertAlmostEqual(
            np.bincount(_phys(np.array(sorted(seen)) - sub.start) // per_device).max()
            / cap,
            fill,
        )

    def test_overflow_is_refused(self) -> None:
        sub = _SubTable("t@0", "t", 0, 1000)
        ids = np.arange(0, 640, 16)  # every id lands on one owner under _LAYOUT
        parts = [(*np.unique(ids, return_inverse=True), ids.size, sub)]

        with self.assertRaisesRegex(RuntimeError, "more than the exchange buffer"):
            _exchange_layout(parts, _LAYOUT, _WORLD, _NUM_SC, cap=ids.size - 1)
