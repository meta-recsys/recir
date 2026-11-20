# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
from recir.lowering.torchtpu_lower import (
    _check_same_graph_on_all_ranks,
    _graph_fingerprint,
    _mark_collectives_spmd_safe,
    _register_tpu_fbgemm_kernels,
    _shape_mismatches,
    _TorchTPUModule,
    partial_lower_torchtpu,
)
from torch.fx.experimental.proxy_tensor import make_fx


def _view_used_across_collective() -> tuple[
    torch.fx.GraphModule, tuple[torch.fx.Node, ...]
]:
    """A column view read after an all_reduce, and the graph's named nodes."""
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
    return gm, (column, row, local, reduced, out)


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

    def test_collectives_are_marked_spmd_safe(self) -> None:
        gm, (_, _, local, reduced, _) = _view_used_across_collective()
        self.assertEqual(_mark_collectives_spmd_safe(gm), 1)
        self.assertEqual(reduced.meta["custom"], {"spmd_safe": True})
        self.assertNotIn("custom", local.meta)

    def test_fingerprint_follows_structure_not_literal_values(self) -> None:
        def scale(factor: float) -> torch.fx.GraphModule:
            return make_fx(lambda x: x * factor)(torch.randn(4))

        same, _ = _graph_fingerprint(scale(2.0))
        self.assertEqual(_graph_fingerprint(scale(3.0))[0], same)
        wider = make_fx(lambda x: x * 2.0)(torch.randn(5))
        self.assertNotEqual(_graph_fingerprint(wider)[0], same)

    def test_fingerprint_covers_collectives(self) -> None:
        gm, (_, _, _, reduced, _) = _view_used_across_collective()
        digest, collectives = _graph_fingerprint(gm)
        self.assertEqual(sum(collectives.values()), 1)
        reduced.args = (reduced.args[0], "avg", reduced.args[2])
        self.assertNotEqual(_graph_fingerprint(gm)[0], digest)

    def test_ranks_compiling_different_graphs_raise(self) -> None:
        gm = make_fx(lambda x: x * 2.0)(torch.randn(4))

        def check(other_rank_offset: int) -> None:
            def all_gather(out: list[torch.Tensor], mine: torch.Tensor) -> None:
                out[0].copy_(mine)
                out[1].copy_(mine + other_rank_offset)

            with (
                patch.object(dist, "is_initialized", return_value=True),
                patch.object(dist, "get_world_size", return_value=2),
                patch.object(dist, "all_gather", side_effect=all_gather),
            ):
                _check_same_graph_on_all_ranks(gm, "subgraph")

        check(0)
        with self.assertRaisesRegex(RuntimeError, r"ranks \[1\]"):
            check(1)

    def test_fbgemm_ops_left_in_a_graph_get_tpu_kernels(self) -> None:
        _register_tpu_fbgemm_kernels()
        for name in (
            "fbgemm::jagged_to_padded_dense_forward",
            "fbgemm::jagged_to_padded_dense_backward",
        ):
            self.assertTrue(
                torch._C._dispatch_has_kernel_for_dispatch_key(name, "PrivateUse1"),
                name,
            )
        self.assertEqual(_register_tpu_fbgemm_kernels(), [])
