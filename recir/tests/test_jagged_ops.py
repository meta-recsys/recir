#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest

import recir.ops.jagged_ops  # noqa: F401
import torch
from functorch import make_fx


class JaggedOpsTest(unittest.TestCase):
    def test_split_accepts_dynamic_output_sizes(self) -> None:
        def round_trip(
            left: torch.Tensor,
            offsets_left: torch.Tensor,
            right: torch.Tensor,
            offsets_right: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            combined = torch.ops.recir.concat_2d_jagged(
                left,
                offsets_left,
                right,
                offsets_right,
            )
            return torch.ops.recir.split_2d_jagged(
                combined,
                offsets_left,
                offsets_right,
                left.shape[0],
                right.shape[0],
            )

        trace_inputs = (
            torch.randn(3, 4),
            torch.tensor([0, 2, 3]),
            torch.randn(2, 4),
            torch.tensor([0, 1, 2]),
        )
        traced = make_fx(round_trip, tracing_mode="symbolic")(*trace_inputs)

        left = torch.randn(5, 4, requires_grad=True)
        right = torch.randn(3, 4, requires_grad=True)
        actual_left, actual_right = traced(
            left,
            torch.tensor([0, 3, 5]),
            right,
            torch.tensor([0, 1, 3]),
        )
        torch.testing.assert_close(actual_left, left)
        torch.testing.assert_close(actual_right, right)

        (actual_left.sum() + actual_right.sum()).backward()
        torch.testing.assert_close(left.grad, torch.ones_like(left))
        torch.testing.assert_close(right.grad, torch.ones_like(right))
