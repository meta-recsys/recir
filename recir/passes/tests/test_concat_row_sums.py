# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from recir.backends.xla import XLAGSPMDBackend
from recir.passes.concat_row_sums import split_concat_row_sums
from recir.passes.jagged_queries import fuse_jagged_queries
from torch.fx.experimental.proxy_tensor import make_fx


def _blocks() -> list[torch.Tensor]:
    # Integer values, so every summation order gives the same result.
    gen = torch.Generator().manual_seed(0)
    return [
        torch.randint(-8, 8, (6, cols), generator=gen).float() for cols in (4, 2, 3)
    ]


def _sums_of_concats(gm: torch.fx.GraphModule) -> int:
    return sum(
        1
        for node in gm.graph.nodes
        if node.target == torch.ops.aten.sum.dim_IntList
        and node.args[0].target == torch.ops.aten.cat.default
    )


class SplitConcatRowSumsTest(unittest.TestCase):
    def test_row_sums_of_a_column_concat_are_summed_per_block(self) -> None:
        def step(a, b, c, x):
            grad = torch.cat([a, b, c], dim=1)
            return grad.sum(0), grad.sum(0, keepdim=True), grad.t() @ x

        a, b, c = _blocks()
        x = torch.ones(6, 5)
        gm = make_fx(step, tracing_mode="fake")(a, b, c, x)
        self.assertEqual(_sums_of_concats(gm), 2)

        split_concat_row_sums(gm)

        self.assertEqual(_sums_of_concats(gm), 0)
        self.assertEqual(gm.meta["concat_row_sum_count"], 2)
        for actual, expected in zip(gm(a, b, c, x), step(a, b, c, x)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_other_sums_and_concats_are_left_alone(self) -> None:
        def step(a, b):
            return (
                torch.cat([a, b], dim=0).sum(0),
                torch.cat([a, b], dim=1).sum(1),
                torch.cat([a, b], dim=1).sum(0, dtype=torch.float64),
            )

        a = _blocks()[0]
        gm = make_fx(step, tracing_mode="fake")(a, a)
        before = gm.code

        split_concat_row_sums(gm)

        self.assertEqual(gm.code, before)
        self.assertEqual(gm.meta["concat_row_sum_count"], 0)

    def test_backend_adds_the_pass_after_the_jagged_query_fusion(self) -> None:
        self.assertEqual(
            XLAGSPMDBackend(split_concat_row_sums=True).optimize_passes,
            (split_concat_row_sums,),
        )
        self.assertEqual(
            XLAGSPMDBackend(
                fuse_jagged_queries=True, split_concat_row_sums=True
            ).optimize_passes,
            (fuse_jagged_queries, split_concat_row_sums),
        )
