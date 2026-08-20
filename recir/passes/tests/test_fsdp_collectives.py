# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Unit tests for recir.passes.fsdp_collectives + XLAShardMapBackend wiring.

Covers imports, backend option wiring, and no-op safety on a trivial graph. No
TPU or jax device is required: jax is imported by the passes module, but no
collective is actually executed on a device.
"""

import unittest

import torch
import torch.nn as nn
from recir.backends.xla import XLAShardMapBackend
from recir.passes.fsdp_collectives import (
    _group_consecutive,
    bucket_grad_reduce_scatters,
    merge_backward_gathers,
    serialize_backward_gathers,
)
from torch import fx


def _trivial_gm() -> fx.GraphModule:
    """A tiny valid fx GraphModule with no reduce_scatter / all_gather nodes."""
    graph = fx.Graph()
    x = graph.placeholder("x")
    add = graph.call_function(torch.ops.aten.add.Tensor, (x, x))
    graph.output((add,))
    return fx.GraphModule(nn.Module(), graph)


class FsdpCollectivesTest(unittest.TestCase):
    def test_backend_default_is_merge_only(self) -> None:
        # The redundant-gather merge is bit-exact and always applied, so the
        # flag-derived path with no opt-in flags yields exactly the merge pass.
        passes = XLAShardMapBackend().optimize_passes
        self.assertEqual(passes, (merge_backward_gathers,))

    def test_backend_bucket_reduce_scatters_adds_pass(self) -> None:
        passes = XLAShardMapBackend(bucket_reduce_scatters=True).optimize_passes
        # merge (always) + bucket.
        self.assertEqual(len(passes), 2)
        self.assertTrue(all(callable(p) for p in passes))

    def test_backend_all_passes_enabled(self) -> None:
        passes = XLAShardMapBackend(
            bucket_reduce_scatters=True,
            max_inflight_all_gathers=2,
        ).optimize_passes
        # merge (always) + bucket + serialize; merge is always first.
        self.assertEqual(len(passes), 3)
        self.assertIs(passes[0], merge_backward_gathers)
        self.assertTrue(all(callable(p) for p in passes))

    def test_backend_optimize_passes_override_passthrough(self) -> None:
        # Explicit optimize_passes= must not collide with the base kwarg
        # (was a "multiple values" TypeError) and is used unchanged.
        self.assertEqual(XLAShardMapBackend(optimize_passes=[]).optimize_passes, [])
        override = [merge_backward_gathers]
        self.assertEqual(
            XLAShardMapBackend(optimize_passes=override).optimize_passes, override
        )

    def test_group_consecutive_runs(self) -> None:
        # The reduce_scatter bucketing groups by (rows, reduceOp, group_size,
        # group_name); _group_consecutive splits into maximal equal-key runs.
        self.assertEqual(
            _group_consecutive([1, 1, 2, 3, 3, 3], lambda x: x),
            [[1, 1], [2], [3, 3, 3]],
        )
        self.assertEqual(_group_consecutive([], lambda x: x), [])
        self.assertEqual(_group_consecutive([5], lambda x: x), [[5]])
        # Non-adjacent equal keys are NOT merged (consecutive-only, so an
        # interleaved differing key breaks the run).
        self.assertEqual(_group_consecutive([1, 2, 1], lambda x: x), [[1], [2], [1]])

    def test_merge_backward_gathers_is_noop_without_gathers(self) -> None:
        gm = _trivial_gm()
        n_before = len(gm.graph.nodes)
        result = merge_backward_gathers(gm)
        self.assertIs(result, gm)
        self.assertEqual(len(result.graph.nodes), n_before)

    def test_bucket_grad_reduce_scatters_is_noop_without_scatters(self) -> None:
        gm = _trivial_gm()
        n_before = len(gm.graph.nodes)
        result = bucket_grad_reduce_scatters(gm)
        self.assertIs(result, gm)
        self.assertEqual(len(result.graph.nodes), n_before)

    def test_serialize_backward_gathers_is_noop_without_gathers(self) -> None:
        gm = _trivial_gm()
        n_before = len(gm.graph.nodes)
        result = serialize_backward_gathers(gm, window=2, min_gib=0.15)
        self.assertIs(result, gm)
        self.assertEqual(len(result.graph.nodes), n_before)
