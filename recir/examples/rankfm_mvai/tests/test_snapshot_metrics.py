#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""The two model-shaped reads the snapshot channel does.

``num_sequence_groups`` walks an attribute chain that only exists at runtime,
and ``real_example_lengths`` is the metric that must never be computed off a
padded tensor -- the dispatch's pad rows absorb each chip's whole token deficit,
so averaging them in reports a number that is not a sequence length.
"""

import unittest
from types import SimpleNamespace

import torch
from recir.examples.rankfm_mvai.example_lengths import (
    num_sequence_groups,
    real_example_lengths,
)


def _model(groups: int) -> SimpleNamespace:
    """The attribute chain num_sequence_groups walks, nothing more."""
    return SimpleNamespace(
        recblock=SimpleNamespace(
            preprocessor_stack=SimpleNamespace(
                mot_preprocessor=SimpleNamespace(num_sequence=groups)
            )
        )
    )


class RealExampleLengthsTest(unittest.TestCase):
    def test_mean_of_uih_plus_candidates(self) -> None:
        uih = torch.tensor([100, 200, 300])
        nc = torch.tensor([10, 20, 30])

        out = real_example_lengths(uih, nc, 1)

        assert out is not None
        # (110 + 220 + 330) / 3
        self.assertAlmostEqual(out.item(), 220.0)

    def test_result_is_not_zero_dim(self) -> None:
        """report_metrics asserts ndim != 0, and one row per group."""
        out = real_example_lengths(torch.tensor([5]), torch.tensor([7]), 1)

        assert out is not None
        self.assertEqual(out.shape, (1, 1))

    def test_a_pad_row_would_have_changed_the_answer(self) -> None:
        """Why this is computed off the raw batch.

        The dispatch appends a pad row per chip and PRT dumps that chip's whole
        token deficit into it. Here: three real samples averaging 220, plus one
        pad row holding a 9000-token deficit. Reading the padded tensor reports
        2415; the raw batch still reports 220.
        """
        uih = torch.tensor([100, 200, 300])
        nc = torch.tensor([10, 20, 30])
        padded = torch.cat([uih + nc, torch.tensor([9000])])

        self.assertAlmostEqual(padded.float().mean().item(), 2415.0)
        out = real_example_lengths(uih, nc, 1)
        assert out is not None
        self.assertAlmostEqual(out.item(), 220.0)

    def test_multi_group_reports_nothing(self) -> None:
        """The raw batch carries one RO KJT; the preprocessor is what splits it
        into groups, so a per-group number cannot be derived here. Report
        nothing rather than a wrong row."""
        self.assertIsNone(real_example_lengths(torch.tensor([1]), torch.tensor([2]), 2))

    def test_missing_inputs_report_nothing(self) -> None:
        self.assertIsNone(real_example_lengths(None, torch.tensor([2]), 1))
        self.assertIsNone(real_example_lengths(torch.tensor([1]), None, 1))

    def test_mismatched_shapes_report_nothing(self) -> None:
        """A shape disagreement means one of the two has been padded."""
        self.assertIsNone(
            real_example_lengths(torch.tensor([1, 2]), torch.tensor([3]), 1)
        )


class NumSequenceGroupsTest(unittest.TestCase):
    def test_reads_the_preprocessor(self) -> None:
        self.assertEqual(num_sequence_groups(_model(3)), 3)

    def test_unwraps_the_compiled_wrapper(self) -> None:
        wrapped = SimpleNamespace(_model=_model(2))
        self.assertEqual(num_sequence_groups(wrapped), 2)

    def test_a_broken_chain_falls_back_to_one(self) -> None:
        self.assertEqual(num_sequence_groups(SimpleNamespace()), 1)
        self.assertEqual(
            num_sequence_groups(SimpleNamespace(recblock=SimpleNamespace())), 1
        )
