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

    def test_to_sharded_jax_bfloat16(self) -> None:
        """Verify _to_sharded_jax handles bfloat16 tensors.

        NumPy does not support bfloat16, so the conversion must cast to
        float32 for the numpy() call and then back to jnp.bfloat16.
        """
        import jax
        import jax.numpy as jnp
        from jax.sharding import Mesh

        devices = jax.local_devices()
        mesh = Mesh(devices, axis_names=("torch_dist",))
        bf16_tensor = torch.randn(4, 8).to(torch.bfloat16)
        jax_array = _XLAModule._to_sharded_jax(bf16_tensor, mesh)
        self.assertEqual(jax_array.dtype, jnp.bfloat16)
        # Verify values are preserved (compare in float32 to avoid precision issues)
        expected = bf16_tensor.float().cpu().numpy()
        actual = jax_array.astype(jnp.float32).__array__()
        torch.testing.assert_close(
            torch.tensor(actual),
            torch.tensor(expected),
            atol=0,
            rtol=0,
        )

    def test_to_sharded_jax_bool(self) -> None:
        """Verify _to_sharded_jax handles bool tensors (existing behavior)."""
        import jax
        import jax.numpy as jnp
        from jax.sharding import Mesh

        devices = jax.local_devices()
        mesh = Mesh(devices, axis_names=("torch_dist",))
        bool_tensor = torch.tensor([True, False, True, False])
        jax_array = _XLAModule._to_sharded_jax(bool_tensor, mesh)
        self.assertEqual(jax_array.dtype, jnp.bool_)

    def test_to_sharded_jax_float32(self) -> None:
        """Verify _to_sharded_jax passes through float32 without extra casts."""
        import jax
        import jax.numpy as jnp
        from jax.sharding import Mesh

        devices = jax.local_devices()
        mesh = Mesh(devices, axis_names=("torch_dist",))
        f32_tensor = torch.randn(4, 8)
        jax_array = _XLAModule._to_sharded_jax(f32_tensor, mesh)
        self.assertEqual(jax_array.dtype, jnp.float32)

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

    def test_full_like_is_xla_compatible(self) -> None:
        """Verify aten.full_like.default is in the XLA allowlist and does not
        cause a graph break despite lacking a Meta kernel registration."""
        from recir.lowering.xla_lower import _is_xla_compatible

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        full_like = graph.call_function(torch.ops.aten.full_like.default, args=(x, 0.0))
        graph.output(full_like)

        for node in graph.nodes:
            if node.op == "call_function":
                compatible, reason = _is_xla_compatible(node)
                self.assertTrue(
                    compatible,
                    f"{node.target} should be XLA-compatible but got: {reason}",
                )

    def test_full_like_no_graph_break(self) -> None:
        """Verify that a graph with full_like has no graph breaks during XLA lowering."""
        gm = _get_test_graph_module()

        # Inject a full_like node into the graph before the output
        output_node = None
        last_call = None
        for node in gm.graph.nodes:
            if node.op == "output":
                output_node = node
            elif node.op == "call_function":
                last_call = node

        assert output_node is not None and last_call is not None

        with gm.graph.inserting_before(output_node):
            gm.graph.call_function(
                torch.ops.aten.full_like.default,
                args=(last_call, 0.0),
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

    def test_strip_profiler_nodes(self) -> None:
        """Verify _strip_profiler_nodes removes profiler ops from the graph.

        Profiler ops like _record_function_enter_new return ScriptObject
        handles that JAX cannot represent. _strip_profiler_nodes runs on
        the parent graph before segmentation so ScriptObject handles never
        become XLA subgraph inputs.
        """
        from recir.lowering.xla_lower import _strip_profiler_nodes

        # Build a graph with profiler nodes interleaved with aten ops
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        record_fn = graph.call_function(
            torch.ops.profiler._record_function_enter_new.default,
            args=("## test_block ##",),
        )
        relu = graph.call_function(torch.ops.aten.relu.default, args=(x,))
        record_exit = graph.call_function(
            torch.ops.profiler._record_function_exit.default,
            args=(record_fn,),
        )
        graph.output((record_fn, relu))
        gm = torch.fx.GraphModule(root={}, graph=graph)

        # Before stripping: profiler nodes present
        profiler_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and isinstance(n.target, torch._ops.OpOverload)
            and n.target.name().startswith("profiler::")
        ]
        self.assertGreater(len(profiler_nodes), 0)

        # Strip using module-level function (called by partial_lower_xla)
        stripped = _strip_profiler_nodes(gm)

        # After stripping: no profiler nodes remain
        profiler_nodes_after = [
            n
            for n in stripped.graph.nodes
            if n.op == "call_function"
            and isinstance(n.target, torch._ops.OpOverload)
            and n.target.name().startswith("profiler::")
        ]
        self.assertEqual(len(profiler_nodes_after), 0)

        # The relu node should still be present
        relu_nodes = [
            n
            for n in stripped.graph.nodes
            if n.op == "call_function" and n.target == torch.ops.aten.relu.default
        ]
        self.assertEqual(len(relu_nodes), 1)

        # ScriptObject handles (produced by profiler ops) should be cleaned
        # up by eliminate_dead_code since their only consumers were removed.
        for n in stripped.graph.nodes:
            val = n.meta.get("val", None)
            self.assertNotIsInstance(
                val,
                torch.ScriptObject,
                f"ScriptObject handle should be removed but found in node {n.name}",
            )

    def test_strip_profiler_nodes_no_profiler(self) -> None:
        """Verify _strip_profiler_nodes is a no-op on graphs without profiler ops."""
        from recir.lowering.xla_lower import _strip_profiler_nodes

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        relu = graph.call_function(torch.ops.aten.relu.default, args=(x,))
        graph.output(relu)
        gm = torch.fx.GraphModule(root={}, graph=graph)

        node_count_before = len(list(gm.graph.nodes))
        stripped = _strip_profiler_nodes(gm)
        node_count_after = len(list(stripped.graph.nodes))
        self.assertEqual(node_count_before, node_count_after)


if __name__ == "__main__":
    unittest.main()
