# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Host-side input preparation of the SparseCore tables: the lookup's COO input and
per-position slots, and the backward's exchange layout."""

import os
import threading
from concurrent.futures import Future

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import unittest
from types import SimpleNamespace
from typing import Any, List, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from recir.embedding.sparsecore_rowwise import (
    _device_inputs,
    _exchange_layout,
    _ExchangeInputs,
    _HostInputs,
    _split_table,
    _SUB_TABLE_MAX_ROWS,
    _SubTable,
    _take_rows,
    _unique_inverse,
    SparseCoreRowwiseTables,
    SparseCoreTable,
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


class UniqueInverseTest(unittest.TestCase):
    def test_matches_np_unique(self) -> None:
        rng = np.random.default_rng(0)
        cases = [
            rng.integers(0, 3_000_000_000, size=5000),
            rng.integers(0, 50, size=5000).astype(np.int32),
            np.array([7], np.int64),
            np.zeros(0, np.int64),
            # Outside what the packed sort handles: falls back to np.unique.
            np.array([-3, 5, -3]),
            np.array([2**62, 1, 2**62]),
        ]
        for values in cases:
            want_values, want_inverse = np.unique(values, return_inverse=True)
            got_values, got_inverse = _unique_inverse(values)
            self.assertEqual(got_values.dtype, want_values.dtype)
            self.assertTrue(np.array_equal(got_values, want_values))
            self.assertTrue(np.array_equal(got_inverse, want_inverse.reshape(-1)))


class TakeRowsTest(unittest.TestCase):
    def test_matches_indexing_for_any_count(self) -> None:
        rng = np.random.default_rng(0)
        table = jnp.asarray(rng.standard_normal((50, 3)).astype(np.float32))
        for n in (1, 7, 8, 13, 16):
            index = jnp.asarray(rng.integers(0, 50, size=n).astype(np.int32))
            got = jax.jit(_take_rows)(table, index)
            self.assertEqual(got.shape, (n, 3))
            self.assertTrue(np.array_equal(got, np.asarray(table)[np.asarray(index)]))


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


class SplitTableTest(unittest.TestCase):
    def test_sub_tables_cover_every_id_within_both_limits(self) -> None:
        # (rows, dim, devices, sub-tables): the int32 bound decides the first,
        # the 16 GiB per-device tensor limit the second.
        for rows, dim, world, expected in [
            (3_000_000_000, 256, 128, 2),
            (3_000_000_000, 256, 64, 3),
            (1_000, 256, 4, 1),
        ]:
            t = SparseCoreTable("t", rows, dim, ("f",), 0.1)
            subs = _split_table(t, _SUB_TABLE_MAX_ROWS, world, _NUM_SC)

            self.assertEqual(len(subs), expected, (rows, world))
            self.assertEqual(subs[0].start, 0)
            for a, b in zip(subs, subs[1:]):
                self.assertEqual(b.start, a.start + a.rows)
            self.assertEqual(subs[-1].start + subs[-1].rows, rows)
            for sub in subs:
                self.assertLessEqual(sub.rows, _SUB_TABLE_MAX_ROWS)
                device_rows = -(-sub.rows // world) + 8 * _NUM_SC
                self.assertLessEqual(device_rows * dim * 4, 16 * 2**30)


class _Variables(NamedTuple):
    table: Any


class IdOrderBlocksTest(unittest.TestCase):
    """export_table / import_table on 8 CPU devices with a layout JTE does not use
    for single-table stacks (rotation 3, offset 2), so every term of it counts."""

    _N, _D, _SC = 1003, 4, 2
    _SUBS = [_SubTable("t@0", "t", 0, 600), _SubTable("t@1", "t", 600, 403)]

    def setUp(self) -> None:
        os.environ["RECIR_SPARSECORE_CKPT_CHUNK_ROWS"] = "50"
        devices = jax.devices()[:8]
        world, shards = len(devices), len(devices) * self._SC
        self.shards = shards
        self.layout = {
            sub.stack: (-(-sub.rows // shards) + 2, 3, 2) for sub in self._SUBS
        }
        sc = SparseCoreRowwiseTables.__new__(SparseCoreRowwiseTables)
        sc._mesh, sc._axis = Mesh(np.array(devices), ("x",)), "x"
        sc._world, sc._num_sc, sc._programs = world, self._SC, {}
        sc._tables = {"t": SparseCoreTable("t", self._N, self._D, ("f",), 0.1)}
        sc._subs, sc._layout = {"t": self._SUBS}, self.layout
        sc._host_blocks = {}
        sharding = NamedSharding(sc._mesh, P("x"))
        sc.ev_w, sc.momentum = {}, {}
        for sub in self._SUBS:
            rows = shards * self.layout[sub.stack][0]
            w = np.full((rows, self._D), -1.0, np.float32)
            m = np.full((rows,), -1.0, np.float32)
            phys = self._phys(sub, np.arange(sub.rows))
            ids = sub.start + np.arange(sub.rows)
            w[phys] = ids[:, None] + np.arange(self._D) / 10
            m[phys] = ids
            sc.ev_w[sub.stack] = _Variables(jax.device_put(jnp.asarray(w), sharding))
            sc.momentum[sub.stack] = jax.device_put(jnp.asarray(m), sharding)
        self.sc = sc

    def tearDown(self) -> None:
        del os.environ["RECIR_SPARSECORE_CKPT_CHUNK_ROWS"]

    def _phys(self, sub: _SubTable, u: np.ndarray) -> np.ndarray:
        size, rotation, offset = self.layout[sub.stack]
        shards = self.shards
        return ((u % shards + rotation) % shards) * size + u // shards + offset

    def test_export_gives_rows_in_id_order(self) -> None:
        weights = self.sc.export_table("t", "weights")
        momentum = self.sc.export_table("t", "momentum")

        ids = np.arange(self._N)
        padded = 8 * -(-self._N // 8)
        self.assertEqual((padded, self._D), weights.shape)
        np.testing.assert_array_equal(
            weights[: self._N],
            (ids[:, None] + np.arange(self._D) / 10).astype(np.float32),
        )
        np.testing.assert_array_equal(momentum[: self._N, 0], ids)
        self.assertFalse(weights[self._N :].any())

    def test_import_puts_each_id_at_its_sparsecore_row(self) -> None:
        rng = np.random.default_rng(0)
        padded = 8 * -(-self._N // 8)
        weights = rng.standard_normal((padded, self._D)).astype(np.float32)
        momentum = rng.uniform(0, 1, (padded, 1)).astype(np.float32)

        self.sc.import_table("t", "weights", weights)
        self.sc.import_table("t", "momentum", momentum)

        for sub in self._SUBS:
            table = np.asarray(self.sc.ev_w[sub.stack].table)
            mom = np.asarray(self.sc.momentum[sub.stack])
            phys = self._phys(sub, np.arange(sub.rows))
            np.testing.assert_array_equal(
                table[phys], weights[sub.start : sub.start + sub.rows]
            )
            np.testing.assert_array_equal(
                mom[phys], momentum[sub.start : sub.start + sub.rows, 0]
            )
            untouched = np.setdiff1d(np.arange(table.shape[0]), phys)
            self.assertTrue((table[untouched] == -1).all())


class PrefetchTest(unittest.TestCase):
    """``prefetch`` builds a lookup's host inputs on a worker; the lookup of the
    same id arrays takes them, any other lookup builds its own."""

    _SIZES = {"f": (4, 4)}

    def setUp(self) -> None:
        sc = SparseCoreRowwiseTables.__new__(SparseCoreRowwiseTables)
        sc._local_devices, sc._num_sc, sc._limits = 2, _NUM_SC, (8, 8)
        sc._exchange_fill, sc._lookups = 0.0, 0
        sc._prefetcher, sc._prefetched = None, []
        sc._prefetch_lock = threading.Lock()
        sc.ev_w = {"t@0": "weights"}
        self.built: List[Tuple[str, Any]] = []

        def host_inputs(host_ids: Any, sizes: Any, specs: Any, exchange: Any) -> Any:
            self.built.append((threading.current_thread().name, host_ids))
            per_partition = {"t@0": np.array([1])}
            done: Future[_ExchangeInputs] = Future()
            done.set_result(_ExchangeInputs(dest={}, send_rows={}, fill=0.5))
            return _HostInputs(
                csr=f"csr{len(self.built)}",
                slots={},
                exchange=done,
                stats=SimpleNamespace(
                    max_ids_per_partition=per_partition,
                    max_unique_ids_per_partition=per_partition,
                ),
            )

        sc._host_inputs = host_inputs
        sc._programs = {
            tuple(sorted(self._SIZES.items())): (
                "specs",
                lambda csr, ev_w, slots: {"f": csr},
                None,
                "exchange",
            )
        }
        self.sc = sc
        self.ids = np.arange(16)

    def tearDown(self) -> None:
        if self.sc._prefetcher is not None:
            self.sc._prefetcher.shutdown()

    def test_lookup_takes_the_inputs_prefetched_from_the_same_memory(self) -> None:
        self.sc.prefetch({"f": self.ids[:8]})
        # A separate view of the same memory, as each slicing of the step's ids is.
        acts = self.sc.lookup({"f": self.ids[:8]})

        self.assertEqual(["csr1"], acts)
        self.assertEqual(1, len(self.built))
        self.assertTrue(self.built[0][0].startswith("sparsecore-prefetch"))
        self.assertEqual([], self.sc._prefetched)
        self.assertEqual(0.5, self.sc._step_inputs[1].result().fill)

    def test_lookup_of_other_ids_builds_its_own_inputs(self) -> None:
        self.sc.prefetch({"f": self.ids[:8]})
        acts = self.sc.lookup({"f": self.ids[:8].copy()})

        self.assertEqual(["csr2"], acts)
        self.assertEqual(threading.main_thread().name, self.built[1][0])
        self.assertEqual(1, len(self.sc._prefetched))

    def test_only_the_last_two_prefetches_are_kept(self) -> None:
        for start in (0, 4, 8):
            self.sc.prefetch({"f": self.ids[start : start + 8]})

        self.assertEqual([4, 8], [ids["f"][0] for ids, _, _ in self.sc._prefetched])

    def test_prefetch_waits_for_a_lookup_to_build_the_program(self) -> None:
        self.sc._programs = {}

        self.sc.prefetch({"f": self.ids[:8]})

        self.assertIsNone(self.sc._prefetcher)
        self.assertEqual([], self.sc._prefetched)

    def test_a_rewritten_id_buffer_is_not_taken_for_its_prefetch(self) -> None:
        """_same_arrays only says the memory matches. A producer that wrote the
        next step's ids into the buffer it handed over would otherwise be served
        the previous step's inputs, with nothing to notice it."""
        ids = self.ids[:8].copy()
        self.sc.prefetch({"f": ids})
        ids[0] = 999

        acts = self.sc.lookup({"f": ids})

        self.assertEqual(["csr2"], acts)
        self.assertEqual(threading.main_thread().name, self.built[1][0])

    def test_an_untouched_buffer_is_still_taken(self) -> None:
        ids = self.ids[:8].copy()
        self.sc.prefetch({"f": ids})

        acts = self.sc.lookup({"f": ids})

        self.assertEqual(["csr1"], acts)
        self.assertEqual(1, len(self.built))

    def test_exchange_inputs_join_each_stacks_devices(self) -> None:
        def on_device(local: np.ndarray) -> jax.Array:
            return jnp.asarray(local)

        self.sc._global = on_device

        def done(d: int) -> Any:
            future: Future[Any] = Future()
            future.set_result((np.array([d]), np.array([[d, -1]]), 0.1 * d))
            return future

        # Stack-major, two local devices per stack.
        out = self.sc._exchange_inputs(["s0", "s1"], [done(d) for d in range(4)])

        np.testing.assert_array_equal([0, 1], out.dest["s0"])
        np.testing.assert_array_equal([2, 3], out.dest["s1"])
        np.testing.assert_array_equal([2, -1, 3, -1], out.send_rows["s1"])
        self.assertAlmostEqual(0.3, out.fill)


class BackwardFailureTest(unittest.TestCase):
    """The exchange layout is built on a pool thread, so an overflowing batch
    raises inside ``backward`` rather than before the step runs."""

    def _tables(self) -> SparseCoreRowwiseTables:
        sc = SparseCoreRowwiseTables.__new__(SparseCoreRowwiseTables)
        sc._expected, sc._pending = None, {}
        overflowed: Future[Any] = Future()
        overflowed.set_exception(
            RuntimeError("a device sends 9 unique rows to one owner")
        )
        sc._step_inputs = ({"f": (4, 4)}, overflowed)
        return sc

    def test_an_overflow_does_not_leave_the_step_behind(self) -> None:
        sc = self._tables()

        with self.assertRaisesRegex(RuntimeError, "unique rows to one owner"):
            sc.backward(jnp.zeros((4, 4)), 0)

        self.assertEqual({}, sc._pending)
        self.assertIsNone(sc._step_inputs)

    def test_the_next_backward_is_refused_rather_than_mixed_in(self) -> None:
        sc = self._tables()
        with self.assertRaises(RuntimeError):
            sc.backward(jnp.zeros((4, 4)), 0)

        with self.assertRaisesRegex(RuntimeError, "without a preceding lookup"):
            sc.backward(jnp.zeros((4, 4)), 0)
