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
    def test_fixed_capacity_split_concat_round_trip_and_backward(self) -> None:
        offsets_left = torch.tensor([0, 1, 3])
        offsets_right = torch.tensor([0, 2, 3])
        values = torch.tensor(
            [[1.0], [10.0], [11.0], [2.0], [3.0], [12.0]],
            requires_grad=True,
        )

        left, right = torch.ops.recir.split_2d_jagged(
            values, offsets_left, offsets_right, 5, 4
        )
        torch.testing.assert_close(
            left, torch.tensor([[1.0], [2.0], [3.0], [0.0], [0.0]])
        )
        torch.testing.assert_close(right, torch.tensor([[10.0], [11.0], [12.0], [0.0]]))

        round_trip = torch.ops.recir.concat_2d_jagged(
            left, offsets_left, right, offsets_right, values.shape[0]
        )
        torch.testing.assert_close(round_trip, values)
        round_trip.sum().backward()
        torch.testing.assert_close(values.grad, torch.ones_like(values))

    def test_fixed_capacity_concat_pads_output_and_gradients(self) -> None:
        offsets_left = torch.tensor([0, 1, 2])
        offsets_right = torch.tensor([0, 2, 2])

        for feature_width in (None, 3):
            with self.subTest(feature_width=feature_width):
                left = torch.tensor([1.0, 2.0, 0.0, 0.0, 0.0])
                right = torch.tensor([10.0, 11.0, 0.0, 0.0])
                expected = torch.tensor([1.0, 10.0, 11.0, 2.0, 0.0, 0.0, 0.0])
                if feature_width is not None:
                    left = left.unsqueeze(1).expand(-1, feature_width).clone()
                    right = right.unsqueeze(1).expand(-1, feature_width).clone()
                    expected = expected.unsqueeze(1).expand(-1, feature_width).clone()
                left.requires_grad_()
                right.requires_grad_()

                actual = torch.ops.recir.concat_2d_jagged(
                    left,
                    offsets_left,
                    right,
                    offsets_right,
                    7,
                )

                torch.testing.assert_close(actual, expected)
                actual.sum().backward()
                expected_left_grad = torch.zeros_like(left)
                expected_left_grad[:2] = 1
                expected_right_grad = torch.zeros_like(right)
                expected_right_grad[:2] = 1
                torch.testing.assert_close(left.grad, expected_left_grad)
                torch.testing.assert_close(right.grad, expected_right_grad)

    def test_fixed_capacity_handles_empty_sides(self) -> None:
        cases = (
            (
                "empty_left",
                torch.tensor([10.0, 11.0, 12.0]),
                torch.tensor([0, 0, 0]),
                torch.tensor([0, 2, 3]),
                0,
                5,
                5,
                torch.empty(0),
                torch.tensor([10.0, 11.0, 12.0, 0.0, 0.0]),
                torch.tensor([10.0, 11.0, 12.0, 0.0, 0.0]),
            ),
            (
                "empty_right",
                torch.tensor([1.0, 2.0, 3.0]),
                torch.tensor([0, 1, 3]),
                torch.tensor([0, 0, 0]),
                5,
                0,
                5,
                torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0]),
                torch.empty(0),
                torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0]),
            ),
            (
                "both_empty",
                torch.empty(0),
                torch.tensor([0, 0, 0]),
                torch.tensor([0, 0, 0]),
                2,
                3,
                4,
                torch.tensor([0.0, 0.0]),
                torch.tensor([0.0, 0.0, 0.0]),
                torch.tensor([0.0, 0.0, 0.0, 0.0]),
            ),
        )
        for (
            name,
            values,
            offsets_left,
            offsets_right,
            left_capacity,
            right_capacity,
            output_capacity,
            expected_left,
            expected_right,
            expected_output,
        ) in cases:
            with self.subTest(name=name):
                left, right = torch.ops.recir.split_2d_jagged(
                    values,
                    offsets_left,
                    offsets_right,
                    left_capacity,
                    right_capacity,
                )
                torch.testing.assert_close(left, expected_left)
                torch.testing.assert_close(right, expected_right)

                output = torch.ops.recir.concat_2d_jagged(
                    left,
                    offsets_left,
                    right,
                    offsets_right,
                    output_capacity,
                )
                torch.testing.assert_close(output, expected_output)

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
