# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import jax.numpy as jnp
import numpy as np
import torch
from recir.ops.jagged_ops import (
    _replace_last_n_jagged_backward_cpu,
    _replace_last_n_jagged_cpu,
)
from recir.torchax.ops.jagged_ops import _jax_replace_last_n_jagged


def _reference(values_left, offsets_left, values_right, offsets_right):
    """hammer's pytorch_replace_last_n_with_jagged, row by row."""
    out = values_left.clone()
    for b in range(offsets_left.numel() - 1):
        left_end = int(offsets_left[b + 1])
        right_start, right_end = int(offsets_right[b]), int(offsets_right[b + 1])
        n = right_end - right_start
        if n:
            out[left_end - n : left_end] = values_right[right_start:right_end]
    return out


class ReplaceLastNJaggedTest(unittest.TestCase):
    def test_matches_the_reference_on_a_tight_layout(self):
        offsets_left = torch.tensor([0, 3, 5], dtype=torch.int64)
        offsets_right = torch.tensor([0, 2, 3], dtype=torch.int64)
        values_left = torch.arange(5 * 4, dtype=torch.float32).reshape(5, 4)
        values_right = -torch.arange(3 * 4, dtype=torch.float32).reshape(3, 4)

        got = _replace_last_n_jagged_cpu(
            values_left, offsets_left, values_right, offsets_right
        )

        torch.testing.assert_close(
            got, _reference(values_left, offsets_left, values_right, offsets_right)
        )

    def test_values_padded_past_the_last_sequence_pass_through(self):
        # recir routes this op for the padded packed layout, where values_left
        # has rows beyond offsets_left[-1]; they belong to no sequence.
        offsets_left = torch.tensor([0, 3, 5], dtype=torch.int64)
        offsets_right = torch.tensor([0, 2, 3], dtype=torch.int64)
        values_left = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4)
        values_right = -torch.arange(3 * 4, dtype=torch.float32).reshape(3, 4)

        got = _replace_last_n_jagged_cpu(
            values_left, offsets_left, values_right, offsets_right
        )

        want = _reference(values_left, offsets_left, values_right, offsets_right)
        torch.testing.assert_close(got, want)
        torch.testing.assert_close(got[5:], values_left[5:])

    def test_backward_leaves_the_padded_tail_to_the_left_gradient(self):
        offsets_left = torch.tensor([0, 3, 5], dtype=torch.int64)
        offsets_right = torch.tensor([0, 2, 3], dtype=torch.int64)
        grad = torch.ones(8, 4, dtype=torch.float32)

        grad_left, grad_right = _replace_last_n_jagged_backward_cpu(
            grad, offsets_left, offsets_right, 3
        )

        # Every gradient row goes to exactly one of the two outputs.
        self.assertEqual(float(grad_left.sum() + grad_right.sum()), float(grad.sum()))
        torch.testing.assert_close(grad_left[5:], grad[5:])
        self.assertEqual(float(grad_left[1].sum()), 0.0)

    def test_the_cpu_fallback_and_the_jax_lowering_agree_on_a_padded_layout(self):
        offsets_left = torch.tensor([0, 3, 5], dtype=torch.int64)
        offsets_right = torch.tensor([0, 2, 3], dtype=torch.int64)
        values_left = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4)
        values_right = -torch.arange(3 * 4, dtype=torch.float32).reshape(3, 4)

        cpu = _replace_last_n_jagged_cpu(
            values_left, offsets_left, values_right, offsets_right
        )
        lowered = _jax_replace_last_n_jagged(
            jnp.asarray(values_left.numpy()),
            jnp.asarray(offsets_left.numpy()),
            jnp.asarray(values_right.numpy()),
            jnp.asarray(offsets_right.numpy()),
        )

        np.testing.assert_array_equal(cpu.numpy(), np.asarray(lowered))

    def test_an_empty_right_sequence_replaces_nothing(self):
        offsets_left = torch.tensor([0, 3, 5], dtype=torch.int64)
        offsets_right = torch.tensor([0, 0, 0], dtype=torch.int64)
        values_left = torch.arange(5 * 4, dtype=torch.float32).reshape(5, 4)
        values_right = torch.zeros(0, 4, dtype=torch.float32)

        got = _replace_last_n_jagged_cpu(
            values_left, offsets_left, values_right, offsets_right
        )

        torch.testing.assert_close(got, values_left)
