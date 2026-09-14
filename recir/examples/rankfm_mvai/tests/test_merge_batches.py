#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""Merging dumped batches must be indistinguishable from a bigger dumped batch."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
from later.unittest import TestCase
from pyre_extensions import none_throws
from recir.examples.rankfm_mvai.data_pipe import (
    BatchReplayPipe,
    merge_batches,
    replay_batch_iter,
    select_rows_slice,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

_RO_KEYS: List[str] = ["ro_a", "ro_b"]
_NRO_KEYS: List[str] = ["nro_a", "nro_b", "nro_c"]
_UNARY_KEYS: List[str] = ["POSITION_ID"]
_NUM_TASKS = 3
_RO_FLOAT_DIM = 4
_NRO_FLOAT_DIM = 2
_LABEL_NAMES: List[str] = ["click", "like"]


@dataclass
class _FakeBatch:
    """Stand-in for ``MTMLROOTrainBatch``."""

    supervision_label: torch.Tensor  # (num_tasks, B_NRO)
    supervision_weight: torch.Tensor  # (num_tasks, B_NRO)
    ro_float_features: torch.Tensor  # (B_RO, F_ro)
    nro_float_features: torch.Tensor  # (B_NRO, F_nro)
    ro_id_list_features: KeyedJaggedTensor  # stride B_RO
    nro_id_list_features: KeyedJaggedTensor  # stride B_NRO
    num_candidates: torch.Tensor  # (B_RO,)
    unary_features: Optional[KeyedJaggedTensor] = None  # stride B_NRO
    all_labels: Optional[Dict[str, torch.Tensor]] = None  # each (B_NRO,)
    B_NRO: Optional[int] = None
    request_id: Optional[torch.Tensor] = None


def _kjt(
    keys: Sequence[str], rows: Sequence[Sequence[Sequence[int]]]
) -> KeyedJaggedTensor:
    """Build a KJT from ``rows[key_index][row_index] -> list of ids``."""
    values: List[int] = []
    lengths: List[int] = []
    for per_row in rows:
        for ids in per_row:
            values.extend(ids)
            lengths.append(len(ids))
    return KeyedJaggedTensor(
        keys=list(keys),
        values=torch.tensor(values, dtype=torch.int64),
        lengths=torch.tensor(lengths, dtype=torch.int32),
    )


def _ragged_ids(base: int, key_index: int, row: int) -> List[int]:
    n = row % 3 + 1
    return [base + key_index * 1000 + row * 10 + j for j in range(n)]


def _build(base: int, num_candidates: List[int]) -> _FakeBatch:
    """A batch whose every element is a distinct function of its row index."""
    b_ro = len(num_candidates)
    b_nro = sum(num_candidates)
    ro_rows = float(base)
    return _FakeBatch(
        supervision_label=torch.arange(_NUM_TASKS * b_nro, dtype=torch.float32).view(
            _NUM_TASKS, b_nro
        )
        + ro_rows,
        supervision_weight=torch.arange(_NUM_TASKS * b_nro, dtype=torch.float32).view(
            _NUM_TASKS, b_nro
        )
        + ro_rows
        + 0.5,
        ro_float_features=torch.arange(b_ro * _RO_FLOAT_DIM, dtype=torch.float32).view(
            b_ro, _RO_FLOAT_DIM
        )
        + ro_rows,
        nro_float_features=torch.arange(
            b_nro * _NRO_FLOAT_DIM, dtype=torch.float32
        ).view(b_nro, _NRO_FLOAT_DIM)
        + ro_rows,
        ro_id_list_features=_kjt(
            _RO_KEYS,
            [
                [_ragged_ids(base, k, r) for r in range(b_ro)]
                for k in range(len(_RO_KEYS))
            ],
        ),
        nro_id_list_features=_kjt(
            _NRO_KEYS,
            [
                [_ragged_ids(base, k, r) for r in range(b_nro)]
                for k in range(len(_NRO_KEYS))
            ],
        ),
        num_candidates=torch.tensor(num_candidates, dtype=torch.int32),
        unary_features=_kjt(
            _UNARY_KEYS, [[[base + r] for r in range(b_nro)] for _ in _UNARY_KEYS]
        ),
        all_labels={
            name: torch.arange(b_nro, dtype=torch.float32) + ro_rows + i
            for i, name in enumerate(_LABEL_NAMES)
        },
        B_NRO=b_nro,
        request_id=torch.arange(b_ro, dtype=torch.int64) + base,
    )


def _concat_kjt_rows(
    keys: Sequence[str], parts: Sequence[tuple[int, int]]
) -> KeyedJaggedTensor:
    """Reference KJT over ``(base, num_rows)`` segments laid out end to end."""
    return _kjt(
        keys,
        [
            [
                _ragged_ids(base, k, r)
                for base, num_rows in parts
                for r in range(num_rows)
            ]
            for k in range(len(keys))
        ],
    )


class MergeBatchesTest(TestCase):
    def test_merge_equals_a_batch_built_from_the_combined_rows(self) -> None:
        """The whole contract, in one comparison.

        A naive ``torch.cat`` on KJT values/lengths, or a dim-0 cat of
        ``supervision_*``, still produces the right row counts and the right
        totals -- only the placement is wrong. Comparing against the
        directly-built batch is what makes that visible.
        """
        nc_a = [2, 1, 3]
        nc_b = [1, 2]
        a = _build(base=0, num_candidates=nc_a)
        b = _build(base=100, num_candidates=nc_b)
        got = merge_batches([a, b])

        b_nro_a, b_nro_b = sum(nc_a), sum(nc_b)
        torch.testing.assert_close(
            got.supervision_label,
            torch.cat([a.supervision_label, b.supervision_label], dim=1),
        )
        torch.testing.assert_close(
            got.supervision_weight,
            torch.cat([a.supervision_weight, b.supervision_weight], dim=1),
        )
        self.assertEqual(
            list(got.supervision_label.shape), [_NUM_TASKS, b_nro_a + b_nro_b]
        )
        torch.testing.assert_close(
            got.ro_float_features,
            torch.cat([a.ro_float_features, b.ro_float_features]),
        )
        torch.testing.assert_close(
            got.nro_float_features,
            torch.cat([a.nro_float_features, b.nro_float_features]),
        )
        self.assertEqual(got.num_candidates.tolist(), nc_a + nc_b)

        expected_ro = _concat_kjt_rows(_RO_KEYS, [(0, len(nc_a)), (100, len(nc_b))])
        self.assertEqual(got.ro_id_list_features.keys(), _RO_KEYS)
        self.assertEqual(
            got.ro_id_list_features.lengths().tolist(), expected_ro.lengths().tolist()
        )
        self.assertEqual(
            got.ro_id_list_features.values().tolist(), expected_ro.values().tolist()
        )

        expected_nro = _concat_kjt_rows(_NRO_KEYS, [(0, b_nro_a), (100, b_nro_b)])
        self.assertEqual(
            got.nro_id_list_features.lengths().tolist(), expected_nro.lengths().tolist()
        )
        self.assertEqual(
            got.nro_id_list_features.values().tolist(), expected_nro.values().tolist()
        )

        self.assertEqual(
            got.unary_features.values().tolist(),
            list(range(b_nro_a)) + [100 + r for r in range(b_nro_b)],
        )
        a_labels, b_labels = none_throws(a.all_labels), none_throws(b.all_labels)
        for name in _LABEL_NAMES:
            torch.testing.assert_close(
                none_throws(got.all_labels)[name],
                torch.cat([a_labels[name], b_labels[name]]),
            )
        self.assertEqual(got.B_NRO, b_nro_a + b_nro_b)

    def test_a_flat_cat_of_kjt_lengths_would_be_wrong(self) -> None:
        """Pins the key-major trap the merge exists to avoid.

        If this ever becomes an equality, ``_merge_kjts`` has been rewritten as
        a flat cat and every multi-key feature is silently transposed.

        The two batches need DIFFERENT row counts. At equal counts the flat cat
        and the key-major one coincide for symmetric fixtures, and the test
        passes while proving nothing -- which is exactly how the real bug would
        hide behind a dump of uniformly sized files.
        """
        a = _build(base=0, num_candidates=[2, 1, 3])
        b = _build(base=100, num_candidates=[1, 2])
        got = merge_batches([a, b]).ro_id_list_features
        flat = torch.cat(
            [a.ro_id_list_features.lengths(), b.ro_id_list_features.lengths()]
        )
        self.assertEqual(sorted(got.lengths().tolist()), sorted(flat.tolist()))
        self.assertNotEqual(got.lengths().tolist(), flat.tolist())

    def test_single_batch_is_the_identity(self) -> None:
        """A one-file step must not perturb the existing data path."""
        a = _build(base=0, num_candidates=[2, 1, 3])
        self.assertIs(merge_batches([a]), a)

    def test_merge_then_slice_recovers_the_first_batch(self) -> None:
        """``select_rows_slice`` is the left inverse of the merge.

        This is what ``replay_batch_iter`` actually does when
        ``local_batch_size`` is smaller than ``512 * n``, and it only holds if
        the merge put batch 0's rows first AND kept the RO->NRO segmentation
        (``num_candidates``) aligned with them.
        """
        nc_a = [2, 1, 3]
        a = _build(base=0, num_candidates=nc_a)
        b = _build(base=100, num_candidates=[1, 2])
        sliced = select_rows_slice(merge_batches([a, b]), 0, len(nc_a))

        torch.testing.assert_close(sliced.ro_float_features, a.ro_float_features)
        torch.testing.assert_close(sliced.nro_float_features, a.nro_float_features)
        torch.testing.assert_close(sliced.supervision_label, a.supervision_label)
        self.assertEqual(sliced.num_candidates.tolist(), nc_a)
        self.assertEqual(
            sliced.ro_id_list_features.values().tolist(),
            a.ro_id_list_features.values().tolist(),
        )
        self.assertEqual(
            sliced.nro_id_list_features.values().tolist(),
            a.nro_id_list_features.values().tolist(),
        )

    def test_mismatched_kjt_keys_raise(self) -> None:
        """Silently taking the first batch's keys would mis-address every table."""
        a = _build(base=0, num_candidates=[2, 1])
        b = _build(base=100, num_candidates=[1, 2])
        renamed: List[str] = ["ro_a", "ro_ZZZ"]
        b = dataclasses.replace(
            b,
            ro_id_list_features=KeyedJaggedTensor(
                keys=renamed,
                values=b.ro_id_list_features.values(),
                lengths=b.ro_id_list_features.lengths(),
            ),
        )
        with self.assertRaisesRegex(ValueError, "different keys"):
            merge_batches([a, b])

    def test_mismatched_all_labels_raise(self) -> None:
        """A missing label key would otherwise be filled from the wrong batch."""
        a = _build(base=0, num_candidates=[2, 1])
        b = _build(base=100, num_candidates=[1, 2])
        labels: Dict[str, Any] = dict(b.all_labels or {})
        labels.pop(_LABEL_NAMES[-1])
        with self.assertRaisesRegex(ValueError, "all_labels"):
            merge_batches([a, dataclasses.replace(b, all_labels=labels)])


_DUMP_BATCH_SIZE = 512


def _dump_root(stack: contextlib.ExitStack, **metadata: Any) -> str:
    """A dump root holding only ``metadata.json``.

    ``batches_per_step`` reads the manifest and nothing else, so no batch files
    are needed -- which is the point of deciding the file count from metadata
    rather than from a loaded batch.
    """
    root = stack.enter_context(tempfile.TemporaryDirectory())
    with open(os.path.join(root, "metadata.json"), "w") as fh:
        json.dump({"version": 0, "num_ranks": 4, "num_batches": 8, **metadata}, fh)
    return root


class BatchesPerStepTest(TestCase):
    """How many files a step reads is DERIVED from ``--local_batch_size``.

    There is deliberately no second knob. Two independent numbers can disagree
    -- ``--local_batch_size 512`` with two files per step would load 1024 rows
    and silently discard 512 through the rotating window -- and the arithmetic
    that relates them is not a judgement call.

    The batch size must be a whole multiple of the dump's own. Both ways of
    breaking that rule read part of a file and never come back to the rest, so
    both raise rather than quietly training on a subsample.
    """

    def _pipe(self, stack: contextlib.ExitStack, **metadata: Any) -> BatchReplayPipe:
        return BatchReplayPipe(
            path=_dump_root(stack, batch_size=_DUMP_BATCH_SIZE, **metadata),
            replicate_across_ranks=True,
        )

    def test_exact_multiples_give_that_many_files(self) -> None:
        with contextlib.ExitStack() as stack:
            pipe = self._pipe(stack)
            for multiple in (1, 2, 4):
                self.assertEqual(
                    pipe.batches_per_step(_DUMP_BATCH_SIZE * multiple), multiple
                )

    def test_batches_smaller_than_a_file_raise(self) -> None:
        """A sub-file batch reads part of every file and never returns.

        This was the pre-existing behaviour -- ``--local_batch_size 32``
        against a 512-row dump took a rotating 1/16 window and the other 15/16
        of each file was never read. It is the same silent-subsample failure as
        a non-multiple, just in the other direction, so it raises the same way.
        """
        with contextlib.ExitStack() as stack:
            pipe = self._pipe(stack)
            for smaller in (1, 32, _DUMP_BATCH_SIZE - 1):
                with self.assertRaisesRegex(ValueError, "whole multiple"):
                    pipe.batches_per_step(smaller)

    def test_a_non_multiple_raises_rather_than_rounding(self) -> None:
        """Rounding up would train on a subsample while reading full price.

        A step reads whole files. At 768 rows it would pull two (1024 rows) and
        drop 256 of them, and the dropped rows are never revisited -- so the
        run silently sees 3/4 of the dump. The message names the two batch
        sizes that do work.
        """
        with contextlib.ExitStack() as stack:
            pipe = self._pipe(stack)
            with self.assertRaisesRegex(ValueError, "whole multiple") as cm:
                pipe.batches_per_step(768)
            self.assertIn("512", str(cm.exception))
            self.assertIn("1024", str(cm.exception))

    def test_the_suggestion_is_usable_at_sub_file_sizes(self) -> None:
        """``floor`` is 0 below one file; suggesting it would be useless advice.

        Pinned because the arithmetic that builds the message is easy to get
        right for 768 and wrong for 32, and nothing else would notice.
        """
        with contextlib.ExitStack() as stack:
            pipe = self._pipe(stack)
            with self.assertRaises(ValueError) as cm:
                pipe.batches_per_step(32)
            self.assertIn("Use 512 or 1024.", str(cm.exception))

    def test_a_dump_without_batch_size_falls_back_to_one_file(self) -> None:
        """Older dumps carry no ``batch_size``; there is nothing to divide by.

        Guessing from a loaded batch would work only after the first read, i.e.
        after the model is built and traced -- exactly the late failure this
        check exists to avoid. One file per step is what those dumps did before.
        """
        with contextlib.ExitStack() as stack:
            pipe = BatchReplayPipe(path=_dump_root(stack), replicate_across_ranks=True)
            self.assertEqual(pipe.batches_per_step(4096), 1)


# ---------------------------------------------------------------------------
# Concurrent per-step file loading
# ---------------------------------------------------------------------------

_FILES_PER_STEP = 2
_NC_PER_FILE: List[int] = [2, 1, 3]
_ROWS_PER_FILE: int = len(_NC_PER_FILE)


class _ProbePipe:
    """``BatchReplayPipe`` stand-in exposing only what ``replay_batch_iter`` uses.

    Faked rather than dumped to disk because what is under test is the loader's
    fan-out, not the file format: a real dump would make the load fast and
    uniform, which is precisely the condition under which an ordering bug is
    invisible. ``on_load`` lets a test control completion order or force the
    loads to be simultaneous, and the recorded thread ids are how "did it
    actually run in parallel" is answered without timing anything.
    """

    def __init__(
        self,
        files_per_step: int = _FILES_PER_STEP,
        num_files: int = 64,
        on_load: Optional[Callable[[int], None]] = None,
    ) -> None:
        self._files_per_step = files_per_step
        self._num_files = num_files
        self._on_load = on_load
        self._lock = threading.Lock()
        self.threads: List[int] = []
        self.positions: List[int] = []

    def batches_per_step(self, local_batch_size: int) -> int:
        return self._files_per_step

    def __len__(self) -> int:
        return self._num_files

    def load_at(self, pos: int) -> _FakeBatch:
        if self._on_load is not None:
            self._on_load(pos)
        with self._lock:
            self.threads.append(threading.get_ident())
            self.positions.append(pos)
        # base is a function of pos, so every field of a wrongly ordered merge
        # differs -- not just the row count, which a bad merge gets right.
        return _build(base=100 * (pos + 1), num_candidates=list(_NC_PER_FILE))


def _replay(
    pipe: _ProbePipe,
    *,
    max_steps: int = 2,
    files_per_step: int = _FILES_PER_STEP,
    **kwargs: Any,
) -> List[_FakeBatch]:
    """Drive ``replay_batch_iter`` over ``pipe`` to exhaustion.

    ``local_batch_size`` is set to exactly the merged row count so the rotating
    window is a no-op (``n_windows == 1``) and each yielded batch is the plain
    in-order merge of that step's files -- the thing being asserted on.
    """
    return list(
        replay_batch_iter(
            pipe,  # pyre-ignore[6]: structural stand-in for BatchReplayPipe
            {},
            max_steps=max_steps,
            local_batch_size=_ROWS_PER_FILE * files_per_step,
            num_candidates=max(_NC_PER_FILE),
            **kwargs,
        )
    )


class ReplayLoadConcurrencyTest(TestCase):
    """A step's files are read concurrently, and that must change nothing.

    The reads are independent and dominated by blocking FUSE I/O, so loading
    them one after another put 1.19 s/step -- 37% of the step -- in front of
    the device on the 128-chip profile. Overlapping them is free throughput
    only if the merge still sees the files in position order, which is the
    invariant these tests defend.
    """

    def test_out_of_order_completion_still_merges_in_position_order(self) -> None:
        """Delay the EARLY positions so completion order is the reverse.

        ``merge_batches`` is position-sensitive: file 0's rows must land first
        or ``select_rows_slice``'s window stops lining up with the RO->NRO
        segmentation (see test_merge_then_slice_recovers_the_first_batch), and
        nothing downstream raises -- it just trains on scrambled rows. An
        implementation that consumed futures as they completed would produce
        the reverse here and fail; ``Executor.map`` yields in argument order.
        """

        def slow_early(pos: int) -> None:
            time.sleep(0.05 * (_FILES_PER_STEP - 1 - pos % _FILES_PER_STEP))

        serial = _replay(_ProbePipe())
        threaded = _replay(_ProbePipe(on_load=slow_early), load_workers=_FILES_PER_STEP)

        self.assertEqual(len(threaded), len(serial))
        for i, (want, got) in enumerate(zip(serial, threaded)):
            self.assertEqual(
                got.num_candidates.tolist(), want.num_candidates.tolist(), f"step {i}"
            )
            torch.testing.assert_close(got.ro_float_features, want.ro_float_features)
            torch.testing.assert_close(got.nro_float_features, want.nro_float_features)
            torch.testing.assert_close(got.supervision_label, want.supervision_label)
            self.assertEqual(
                got.ro_id_list_features.values().tolist(),
                want.ro_id_list_features.values().tolist(),
            )
            self.assertEqual(
                got.nro_id_list_features.values().tolist(),
                want.nro_id_list_features.values().tolist(),
            )

    def test_the_files_of_one_step_are_in_flight_together(self) -> None:
        """The barrier IS the assertion.

        It only clears when every file of the step is inside ``load_at`` at the
        same moment, so a serial loader (or a pool that was silently bypassed)
        hits the timeout and raises ``BrokenBarrierError`` instead of passing
        slowly. Asserting on elapsed time would be flaky; this is not.
        """
        barrier = threading.Barrier(_FILES_PER_STEP, timeout=60)

        def wait(pos: int) -> None:
            # Barrier.wait returns the arrival index; swallow it so the callback
            # matches on_load's None return.
            barrier.wait()

        pipe = _ProbePipe(on_load=wait)
        batches = _replay(pipe, max_steps=1, load_workers=_FILES_PER_STEP)

        self.assertEqual(len(batches), 1)
        self.assertEqual(len(set(pipe.threads)), _FILES_PER_STEP)

    def test_workers_are_capped_at_the_files_a_step_needs(self) -> None:
        """One file per step has nothing to overlap, so no pool is created.

        Worth pinning: that is the shape of every unmerged run (local batch ==
        the dump's own), and handing each load to a worker thread there would
        add a queue round-trip to the critical path for no I/O overlap.
        """
        pipe = _ProbePipe(files_per_step=1)
        _replay(pipe, max_steps=3, files_per_step=1, load_workers=8)

        self.assertEqual(set(pipe.threads), {threading.get_ident()})

    def test_the_default_is_the_previous_serial_behaviour(self) -> None:
        """Callers that never opt in are untouched.

        ``replay_batch_iter`` is also used for the one-off trace-batch pull and
        by the parity harness, where a background thread would only add
        nondeterminism to a comparison that exists to be bit-exact.
        """
        pipe = _ProbePipe()
        _replay(pipe, max_steps=2)

        self.assertEqual(set(pipe.threads), {threading.get_ident()})
        self.assertEqual(pipe.positions, [0, 1, 2, 3])
