# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# Owner(s): ["oncall: gr_dev"]
# pyre-ignore-all-errors

import copy
import unittest

import torch
import torch.nn as nn
from functorch import make_fx
from recir.backend import BackendType
from recir.lowering.xla_lower import _XLAModule, partial_lower_xla


class SimpleModel(nn.Module):
    def __init__(self, layers: int = 6, dim: int = 16) -> None:
        super().__init__()
        modules = []
        for _ in range(layers):
            modules.extend([nn.Linear(dim, dim), nn.ReLU()])
        self.mod = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mod(x)


def _get_test_graph_module(device: str = "cuda") -> torch.fx.GraphModule:
    """Trace a simple model to get an fx.GraphModule for testing."""
    model = SimpleModel().to(device)
    x = torch.randn(4, 16, device=device)

    def fn(params, x):
        return torch.func.functional_call(model, params, (x,))

    params = dict(model.named_parameters())
    gm = make_fx(fn)(params, x)
    return gm


class TestPartialLowerXLA(unittest.TestCase):
    """Test partial_lower_xla graph segmentation and lowering."""

    def test_subgraph_naming(self) -> None:
        """Verify XLA subgraphs are named with xla_ prefix."""
        gm = _get_test_graph_module()
        lowered = partial_lower_xla(
            gm,
            subgraph_predicate=lambda nodes: len(nodes) >= 1,
        )
        for name, mod in lowered.named_modules():
            if isinstance(mod, _XLAModule):
                self.assertTrue(
                    name.startswith("xla_subgraph_"),
                    f"Expected xla_ prefix, got: {name}",
                )

    def test_custom_node_predicate(self) -> None:
        """Verify node_predicate can exclude specific nodes."""
        gm = _get_test_graph_module()
        lowered = partial_lower_xla(
            gm,
            node_predicate=lambda _n: False,
        )
        xla_modules = [mod for mod in lowered.modules() if isinstance(mod, _XLAModule)]
        self.assertEqual(len(xla_modules), 0)

    def test_backend_type_propagated_to_module(self) -> None:
        """Verify the backend_type is correctly passed to _XLAModule."""
        gm = _get_test_graph_module()
        lowered = partial_lower_xla(
            gm,
            backend_type=BackendType.XLA_SHARD_MAP,
            subgraph_predicate=lambda nodes: len(nodes) >= 1,
        )
        for mod in lowered.modules():
            if isinstance(mod, _XLAModule):
                self.assertEqual(mod.backend_type, BackendType.XLA_SHARD_MAP)

    def test_numerical_correctness_after_lowering(self) -> None:
        """Verify that a lowered graph produces numerically correct output.
        Compiles subgraphs through torchax -> JAX -> XLA and verifies
        the outputs match the original un-lowered graph.
        Uses CUDA tensors to exercise the GPU XLA compilation path.
        """
        torch.manual_seed(42)
        device = "cuda"
        model = SimpleModel().to(device)
        x = torch.randn(4, 16, device=device)

        def fn(params, x):
            return torch.func.functional_call(model, params, (x,))

        params = dict(model.named_parameters())
        gm = make_fx(fn)(params, x)
        ref_output = gm(params, x)
        lowered_gm = partial_lower_xla(
            copy.deepcopy(gm),
            backend_type=BackendType.XLA_SHARD_MAP,
        )
        lowered_output = lowered_gm(params, x)
        self.assertEqual(ref_output.shape, lowered_output.shape)
        self.assertEqual(ref_output.device, lowered_output.device)
        torch.testing.assert_close(ref_output, lowered_output, atol=1e-4, rtol=1e-4)

    def test_numerical_correctness_multiple_inputs(self) -> None:
        """Verify numerical parity across several random CUDA inputs."""
        torch.manual_seed(123)
        device = "cuda"
        model = SimpleModel().to(device)

        def fn(params, x):
            return torch.func.functional_call(model, params, (x,))

        params = dict(model.named_parameters())
        gm = make_fx(fn)(params, torch.randn(4, 16, device=device))
        lowered_gm = partial_lower_xla(
            copy.deepcopy(gm),
            backend_type=BackendType.XLA_SHARD_MAP,
        )
        for _ in range(5):
            x = torch.randn(4, 16, device=device)
            ref = gm(params, x)
            low = lowered_gm(params, x)
            self.assertEqual(ref.device, low.device)
            torch.testing.assert_close(ref, low, atol=1e-4, rtol=1e-4)

    def test_xla_shard_map_produces_xla_hlo(self) -> None:
        """Verify that XLA_SHARD_MAP backend produces an XLA HLO graph on GPU.

        After lowering, each _XLAModule should have:
          - xla_hlo_text: the StableHLO/HLO text representation
        """
        torch.manual_seed(42)
        device = "cuda"
        model = SimpleModel().to(device)
        x = torch.randn(4, 16, device=device)

        def fn(params, x):
            return torch.func.functional_call(model, params, (x,))

        params = dict(model.named_parameters())
        gm = make_fx(fn)(params, x)
        lowered_gm = partial_lower_xla(
            copy.deepcopy(gm),
            backend_type=BackendType.XLA_SHARD_MAP,
        )
        lowered_gm(params, x)
        for mod in lowered_gm.modules():
            if isinstance(mod, _XLAModule):
                self.assertIsNotNone(mod.xla_hlo_text)
                self.assertIsInstance(mod.xla_hlo_text, str)
                self.assertGreater(len(mod.xla_hlo_text), 0)

    def test_foreach_ops_are_xla_compatible(self) -> None:
        """Verify that foreach ops are no longer blocked by _is_xla_compatible."""
        from recir.lowering.xla_lower import _is_xla_compatible

        # Build a graph containing foreach ops
        graph = torch.fx.Graph()
        p1 = graph.placeholder("p1")
        p2 = graph.placeholder("p2")
        foreach_add = graph.call_function(
            torch.ops.aten._foreach_add.Scalar, args=([p1, p2], 1.0)
        )
        foreach_mul = graph.call_function(
            torch.ops.aten._foreach_mul.Scalar, args=([p1, p2], 2.0)
        )
        graph.output((foreach_add, foreach_mul))

        for node in graph.nodes:
            if node.op == "call_function":
                compatible, reason = _is_xla_compatible(node)
                self.assertTrue(
                    compatible,
                    f"{node.target} should be XLA-compatible but got: {reason}",
                )

    def test_foreach_add_no_graph_break(self) -> None:
        """Verify that a graph with foreach_add has no graph breaks during XLA lowering."""
        gm = _get_test_graph_module()

        # Inject a foreach_add node into the graph before the output
        output_node = None
        last_call = None
        for node in gm.graph.nodes:
            if node.op == "output":
                output_node = node
            elif node.op == "call_function":
                last_call = node

        assert output_node is not None and last_call is not None

        # Splice in a _foreach_add.Scalar node
        with gm.graph.inserting_before(output_node):
            gm.graph.call_function(
                torch.ops.aten._foreach_add.Scalar,
                args=([last_call], 0.0),
            )

        gm.recompile()

        lowered = partial_lower_xla(
            gm,
            backend_type=BackendType.XLA_SHARD_MAP,
            subgraph_predicate=lambda nodes: len(nodes) >= 1,
        )
        # No graph breaks means the entire graph is lowered into exactly one _XLAModule
        xla_modules = [mod for mod in lowered.modules() if isinstance(mod, _XLAModule)]
        self.assertEqual(
            len(xla_modules), 1, "Expected exactly one _XLAModule (no graph breaks)"
        )


if __name__ == "__main__":
    unittest.main()
