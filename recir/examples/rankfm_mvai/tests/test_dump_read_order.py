#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""Pin the dump read order to MVAI's.

Both replay readers walk the same 2D grid of dump files -- ``num_batches`` rows
(how far a writer got) by ``num_ranks`` columns (which of the dump job's writer
processes produced it). Only the traversal order is a choice, and the two sides
have to make the same one.

MVAI's ``PreprocOutputReplayDataLoader`` is rank-major
(``preproc_output_dataloaders.py``):

    global_idx = replay_rank + i * world_size
    dump_rank  = global_idx % num_ranks
    batch_idx  = global_idx // num_ranks

i.e. file ``(b, r)`` sits at flat position ``b * num_ranks + r``.

``BatchReplayPipe._discover_files`` must agree. Its replicated branch was once
nested the other way, putting ``(b, r)`` at ``r * num_batches + b`` -- the
transpose. Nothing failed: both orders replay the whole dump. But a bounded run
read entirely different data on the two legs. With 480 dump ranks, the first 400
files were 400 writers at batch 0 for light versus writer 0 across 400 batches
for recir, overlapping in exactly one file. An NE comparison across that is
measuring the data difference rather than the model, which is the one thing the
comparison exists to rule out.
"""

import json
import os
import re
import tempfile
from typing import List

from later.unittest import TestCase
from pyre_extensions import none_throws
from recir.examples.rankfm_mvai.data_pipe import (
    _BATCHES_PER_SUBDIR,
    _SUPPORTED_FORMAT_VERSION,
    BatchReplayPipe,
)

# Small stand-ins for the real dump's 480 x 41942; the ordering property does
# not depend on the magnitudes, and a wide-but-shallow grid is exactly the shape
# that made the transpose invisible.
_NUM_RANKS: int = 8
_NUM_BATCHES: int = 5


def _mvai_expected_order(num_ranks: int, num_batches: int) -> List[str]:
    """The file sequence MVAI's loader produces, at world_size 1."""
    names: List[str] = []
    for global_idx in range(num_ranks * num_batches):
        dump_rank = global_idx % num_ranks
        batch_idx = global_idx // num_ranks
        names.append(f"rank_{dump_rank}/batch_0/{batch_idx}.pt.zst")
    return names


def _write_metadata(root: str) -> None:
    with open(os.path.join(root, "metadata.json"), "w") as fh:
        json.dump(
            {
                "version": 0,
                "num_ranks": _NUM_RANKS,
                "num_batches": _NUM_BATCHES,
                "batch_size": 512,
            },
            fh,
        )


class DumpReadOrderTest(TestCase):
    def test_layout_constants_still_match_the_writer(self) -> None:
        """The dump layout is copied from the writer; catch it drifting.

        ``_batch_file_name`` SYNTHESIZES paths from the metadata rather than
        listing the directory, so it has to know the writer's bucketing. The
        writer lives in ``minimal_viable_ai``, which is not staged into the TPU
        image (only the mvai_cloud fork is) and which ``data_pipe`` must run
        under, so the constants are duplicated rather than imported. The
        duplication is forced; the drift is not.

        ``_BATCHES_PER_SUBDIR`` is the dangerous one. It is arithmetic, not a
        check: raise it to 5000 upstream and every synthesized path silently
        addresses the wrong file, surfacing as a FileNotFoundError with nothing
        pointing back here. ``_SUPPORTED_FORMAT_VERSION`` is safer -- a new
        writer version makes ``_discover_files`` refuse the dump outright --
        but it is the same copy, so it is pinned the same way.

        Reads the writer's SOURCE rather than importing it. Importing pulls
        //minimal_viable_ai/core/dataloader:preproc_output_dataloaders into the
        test binary, whose closure is large enough to overflow the linker
        (``relocation R_X86_64_PC32 out of range``) -- the same closure size
        that made the config artifact necessary in the first place. Same
        approach as test_artifact_is_loadable_against_the_fork.
        """
        writer_src = os.path.join(
            os.environ.get("FBCODE_DIR", os.getcwd()),
            "minimal_viable_ai/core/dataloader/preproc_output_dataloaders.py",
        )
        if not os.path.exists(writer_src):
            self.skipTest(f"writer source not found at {writer_src}")
        with open(writer_src) as fh:
            text = fh.read()

        for ours, their_name in (
            (_BATCHES_PER_SUBDIR, "_BATCHES_PER_SUBDIR"),
            (_SUPPORTED_FORMAT_VERSION, "_FORMAT_VERSION"),
        ):
            m = re.search(rf"^{their_name}: int = (\d+)$", text, re.M)
            self.assertIsNotNone(
                m, f"{their_name} is gone from the writer; the layout moved"
            )
            self.assertEqual(
                ours,
                int(none_throws(m).group(1)),
                f"{their_name} drifted from the writer. _BATCHES_PER_SUBDIR is "
                "arithmetic, so a change there makes every synthesized path "
                "address the wrong file, surfacing as a FileNotFoundError with "
                "nothing pointing back here.",
            )

    def test_replicated_order_matches_mvai(self) -> None:
        """The replicated branch must be rank-major, not batch-major."""
        expected = _mvai_expected_order(_NUM_RANKS, _NUM_BATCHES)
        with tempfile.TemporaryDirectory() as root:
            _write_metadata(root)
            pipe = BatchReplayPipe(path=root, replicate_across_ranks=True)

            got = ["/".join(f.rsplit("/", 3)[-3:]) for f in pipe._file_list]
            self.assertEqual(
                got,
                expected,
                "replicated read order diverged from MVAI's, so the two legs "
                "would train on different slices of the dump.\n"
                f"  first four here:  {got[:4]}\n"
                f"  first four MVAI:  {expected[:4]}\n"
                "A run shorter than num_ranks never leaves the first row, so "
                "the two orders differ in exactly the way that hides best: "
                "MVAI sweeps WRITERS at batch 0, the transpose walks writer 0 "
                "through successive batches, and both look equally healthy in "
                "the log.",
            )

    def test_sharded_ranks_take_mvais_stride_and_partition_the_dump(self) -> None:
        """The non-replicated branch was always correct; keep it that way.

        Exercised at world_size=4 rather than 1. At world_size=1 the stride
        collapses to 1 and ``global_idx == i``, so the assertion passes for any
        implementation that walks the grid in the right order -- including a
        block split, which would hand rank 0 the first quarter of the dump
        instead of every fourth file. Four ranks is the smallest case where
        striding and blocking disagree.

        The partition check is the property that actually matters: across the
        ranks every file appears exactly once, so a bounded multi-rank run
        covers the same set the replicated reader does.
        """
        expected = _mvai_expected_order(_NUM_RANKS, _NUM_BATCHES)
        world_size = 4
        seen: List[str] = []
        for rank in range(world_size):
            with tempfile.TemporaryDirectory() as root:
                _write_metadata(root)
                pipe = BatchReplayPipe(path=root, rank=rank, world_size=world_size)

                got = ["/".join(f.rsplit("/", 3)[-3:]) for f in pipe._file_list]
                self.assertEqual(
                    got,
                    expected[rank::world_size],
                    f"rank {rank} of {world_size} did not get MVAI's stride "
                    "through the global file order",
                )
                seen.extend(got)

        self.assertCountEqual(
            seen,
            expected,
            "the ranks do not partition the dump: some file is read twice or "
            "not at all",
        )
