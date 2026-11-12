# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from recir.lowering.torchtpu_lower import (
    _copy_cross_partition_views,
    _shape_mismatches,
    _TorchTPUModule,
    partial_lower_torchtpu,
)
from torch.fx.experimental.proxy_tensor import make_fx


def _profiler_targets(gm: torch.fx.GraphModule) -> list[str]:
    return [
        node.target.name()
        for module in gm.modules()
        if isinstance(module, torch.fx.GraphModule)
        for node in module.graph.nodes
        if isinstance(node.target, torch._ops.OpOverload)
        and node.target.name().startswith("profiler::")
    ]


class TorchTPULowerTest(unittest.TestCase):
    def test_profiler_handles_do_not_become_subgraph_inputs(self) -> None:
        def step(x: torch.Tensor) -> torch.Tensor:
            # Enough ops on both sides of the scope to clear
            # MIN_ATEN_OPS_TO_LOWER as one subgraph.
            for _ in range(6):
                x = (x * 2).relu()
            with torch.profiler.record_function("scope"):
                for _ in range(6):
                    x = (x + 1).tanh()
            return x

        gm = make_fx(step)(torch.randn(4))
        self.assertTrue(_profiler_targets(gm))

        lowered = partial_lower_torchtpu(gm)

        self.assertEqual(_profiler_targets(lowered), [])
        subgraphs = [m for m in lowered.modules() if isinstance(m, _TorchTPUModule)]
        self.assertEqual(len(subgraphs), 1)
        for node in subgraphs[0].gm.graph.find_nodes(op="placeholder"):
            self.assertIsInstance(node.meta["val"], torch.Tensor)

    def test_empty_views_reach_consumers_as_fresh_tensors(self) -> None:
        def step(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return x * 2, x[4:] + 1

        x = torch.randn(4)
        module = _TorchTPUModule(make_fx(step)(x))

        targets = [node.target for node in module.gm.graph.nodes]
        self.assertIn(torch.ops.aten.new_empty.default, targets)
        doubled, empty = module.gm(x)
        torch.testing.assert_close(doubled, x * 2)
        self.assertEqual(empty.shape, (0,))
        self.assertIsNone(empty._base)

    def test_inputs_are_checked_against_their_traced_shapes(self) -> None:
        gm = make_fx(lambda x, y: x + y.sum())(torch.zeros(4), torch.zeros(3))
        self.assertEqual(_shape_mismatches(gm, [torch.zeros(4), torch.zeros(3)]), [])
        self.assertEqual(
            _shape_mismatches(gm, [torch.zeros(4), torch.zeros(2)]),
            ["y_1: traced (3,), got (2,)"],
        )

    def test_views_used_after_a_collective_are_copied_before_it(self) -> None:
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        column = graph.call_function(torch.ops.aten.select.int, (x, 1, 3))
        row = graph.call_function(torch.ops.aten.select.int, (x, 0, 0))
        local = graph.call_function(torch.ops.aten.mul.Tensor, (row, 2))
        reduced = graph.call_function(
            torch.ops._c10d_functional.all_reduce.default, (local, "sum", "0")
        )
        waited = graph.call_function(
            torch.ops._c10d_functional.wait_tensor.default, (reduced,)
        )
        out = graph.call_function(torch.ops.aten.add.Tensor, (column, waited))
        graph.output(out)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        self.assertEqual(_copy_cross_partition_views(gm), 1)

        copy = out.args[0]
        assert isinstance(copy, torch.fx.Node)
        self.assertIs(copy.target, torch.ops.aten.clone.default)
        self.assertIs(copy.args[0], column)
        self.assertIs(local.args[0], row)
        nodes = list(gm.graph.nodes)
        self.assertLess(nodes.index(copy), nodes.index(reduced))
