# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest
from typing import Any

import jax  # @manual
import recir.torchax
import torch
import torch.nn as nn
import triton  # @manual
import triton.language as tl  # @manual
from functorch import make_fx
from jax.experimental import pallas as pl  # @manual
from recir.backends.types import BackendType
from recir.lowering.op_swap import OpSwapRegistry, swap_ops
from recir.lowering.xla_lower import _XLAModule, partial_lower_xla
from recir.torchax import interop
from recir.torchax.ops import ops_registry
from recir.torchax.ops.mappings import t2j


@triton.jit
def _triton_mul2_kernel(
    x_ptr: Any, output_ptr: Any, n_elements: Any, BLOCK_SIZE: tl.constexpr
) -> None:
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x * 2.0, mask=mask)


@triton.jit
def _triton_add1_kernel(
    x_ptr: Any, output_ptr: Any, n_elements: Any, BLOCK_SIZE: tl.constexpr
) -> None:
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + 1.0, mask=mask)


_test_lib: torch.library.Library = torch.library.Library("recir_test", "DEF")  # noqa: F841
_test_lib.define("triton_mul2(Tensor x) -> Tensor")
_test_lib.define("triton_add1(Tensor x) -> Tensor")
_test_lib.define("pallas_mul2(Tensor x) -> Tensor")
_test_lib.define("pallas_add1(Tensor x) -> Tensor")


@torch.library.impl(_test_lib, "triton_mul2", "CPU")
def _triton_mul2_cpu(x: torch.Tensor) -> torch.Tensor:
    return x * 2.0


@torch.library.impl(_test_lib, "triton_mul2", "CUDA")
def _triton_mul2_cuda(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = x.numel()
    _triton_mul2_kernel[(triton.cdiv(n_elements, 1024),)](
        x,
        output,
        n_elements,
        BLOCK_SIZE=1024,  # pyre-ignore[6]: constexpr
    )
    return output


@torch.library.impl(_test_lib, "triton_mul2", "Meta")
def _triton_mul2_meta(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.impl(_test_lib, "pallas_mul2", "CPU")
def _pallas_mul2_cpu(x: torch.Tensor) -> torch.Tensor:
    return x * 2.0


@torch.library.impl(_test_lib, "pallas_mul2", "Meta")
def _pallas_mul2_meta(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.impl(_test_lib, "triton_add1", "CPU")
def _triton_add1_cpu(x: torch.Tensor) -> torch.Tensor:
    return x + 1.0


@torch.library.impl(_test_lib, "triton_add1", "CUDA")
def _triton_add1_cuda(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = x.numel()
    _triton_add1_kernel[(triton.cdiv(n_elements, 1024),)](
        x,
        output,
        n_elements,
        BLOCK_SIZE=1024,  # pyre-ignore[6]: constexpr
    )
    return output


@torch.library.impl(_test_lib, "triton_add1", "Meta")
def _triton_add1_meta(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.impl(_test_lib, "pallas_add1", "CPU")
def _pallas_add1_cpu(x: torch.Tensor) -> torch.Tensor:
    return x + 1.0


def _pallas_mul2_kernel(x_ref: Any, o_ref: Any) -> None:
    o_ref[...] = x_ref[...] * 2.0


def _pallas_add1_kernel(x_ref: Any, o_ref: Any) -> None:
    o_ref[...] = x_ref[...] + 1.0


def _pallas_mul2_jax(x: jax.Array) -> jax.Array:
    return pl.pallas_call(
        _pallas_mul2_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        interpret=True,
    )(x)


def _pallas_add1_jax(x: jax.Array) -> jax.Array:
    return pl.pallas_call(
        _pallas_add1_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        interpret=True,
    )(x)


# Register Pallas JAX implementations in torchax so that _compile_torchax
# can lower the pallas custom ops through JAX/XLA.
ops_registry.register_torch_dispatch_op(
    torch.ops.recir_test.pallas_mul2.default,
    _pallas_mul2_jax,
    is_jax_function=True,
    is_user_defined=True,
)
ops_registry.register_torch_dispatch_op(
    torch.ops.recir_test.pallas_add1.default,
    _pallas_add1_jax,
    is_jax_function=True,
    is_user_defined=True,
)


class ModelWithTritonOps(nn.Module):
    def __init__(self, layers: int = 6, dim: int = 16) -> None:
        super().__init__()
        modules = []
        for _ in range(layers):
            modules.extend([nn.Linear(dim, dim), nn.ReLU()])
        self.mod = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mod(x)
        x = torch.ops.recir_test.triton_mul2(x)
        return x


class ModelWithMultipleTritonOps(nn.Module):
    def __init__(self, layers: int = 6, dim: int = 16) -> None:
        super().__init__()
        modules = []
        for _ in range(layers):
            modules.extend([nn.Linear(dim, dim), nn.ReLU()])
        self.mod = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mod(x)
        x = torch.ops.recir_test.triton_mul2(x)
        x = torch.ops.recir_test.triton_add1(x)
        return x


def _trace_model(model: nn.Module, device: str = "cpu") -> torch.fx.GraphModule:
    model = model.to(device)
    x = torch.randn(4, 16, device=device)

    def fn(params: dict[str, Any], x: torch.Tensor) -> torch.Tensor:
        return torch.func.functional_call(model, params, (x,))

    params = dict(model.named_parameters())
    gm = make_fx(fn)(params, x)
    return gm


class TestOpSwapRegistry(unittest.TestCase):
    def test_register_and_lookup(self) -> None:
        registry = OpSwapRegistry()
        triton_op = torch.ops.recir_test.triton_mul2.default
        pallas_op = torch.ops.recir_test.pallas_mul2.default
        self.assertFalse(registry.has(triton_op))
        self.assertIsNone(registry.get(triton_op))
        registry.register(triton_op, pallas_op)
        self.assertTrue(registry.has(triton_op))
        self.assertIs(registry.get(triton_op), pallas_op)
        self.assertEqual(len(registry), 1)

    def test_unregister(self) -> None:
        registry = OpSwapRegistry()
        triton_op = torch.ops.recir_test.triton_mul2.default
        pallas_op = torch.ops.recir_test.pallas_mul2.default
        registry.register(triton_op, pallas_op)
        self.assertTrue(registry.has(triton_op))
        registry.unregister(triton_op)
        self.assertFalse(registry.has(triton_op))
        self.assertEqual(len(registry), 0)

    def test_clear(self) -> None:
        registry = OpSwapRegistry()
        registry.register(
            torch.ops.recir_test.triton_mul2.default,
            torch.ops.recir_test.pallas_mul2.default,
        )
        registry.register(
            torch.ops.recir_test.triton_add1.default,
            torch.ops.recir_test.pallas_add1.default,
        )
        self.assertEqual(len(registry), 2)
        registry.clear()
        self.assertEqual(len(registry), 0)

    def test_overwrite_warns(self) -> None:
        registry = OpSwapRegistry()
        triton_op = torch.ops.recir_test.triton_mul2.default
        pallas_op_1 = torch.ops.recir_test.pallas_mul2.default
        pallas_op_2 = torch.ops.recir_test.pallas_add1.default
        registry.register(triton_op, pallas_op_1)
        self.assertIs(registry.get(triton_op), pallas_op_1)
        registry.register(triton_op, pallas_op_2)
        self.assertIs(registry.get(triton_op), pallas_op_2)
        self.assertEqual(len(registry), 1)

    def test_registered_targets(self) -> None:
        registry = OpSwapRegistry()
        triton_mul2 = torch.ops.recir_test.triton_mul2.default
        triton_add1 = torch.ops.recir_test.triton_add1.default
        registry.register(triton_mul2, torch.ops.recir_test.pallas_mul2.default)
        registry.register(triton_add1, torch.ops.recir_test.pallas_add1.default)
        self.assertEqual(registry.registered_targets, {triton_mul2, triton_add1})

    def test_swap_map_returns_copy(self) -> None:
        registry = OpSwapRegistry()
        triton_op = torch.ops.recir_test.triton_mul2.default
        pallas_op = torch.ops.recir_test.pallas_mul2.default
        registry.register(triton_op, pallas_op)
        swap_map = registry.swap_map
        swap_map.clear()
        self.assertEqual(len(registry), 1)


class TestSwapOps(unittest.TestCase):
    def _make_registry(self, mappings: dict[Any, Any]) -> OpSwapRegistry:
        registry = OpSwapRegistry()
        for src, dst in mappings.items():
            registry.register(src, dst)
        return registry

    def test_single_swap(self) -> None:
        gm = _trace_model(ModelWithTritonOps())
        triton_op = torch.ops.recir_test.triton_mul2.default
        pallas_op = torch.ops.recir_test.pallas_mul2.default
        triton_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target == triton_op
        ]
        self.assertEqual(len(triton_nodes), 1)
        registry = self._make_registry({triton_op: pallas_op})
        swap_ops(gm, registry)
        pallas_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target == pallas_op
        ]
        triton_nodes_after = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target == triton_op
        ]
        self.assertEqual(len(pallas_nodes), 1)
        self.assertEqual(len(triton_nodes_after), 0)

    def test_multiple_swaps(self) -> None:
        gm = _trace_model(ModelWithMultipleTritonOps())
        registry = self._make_registry(
            {
                torch.ops.recir_test.triton_mul2.default: torch.ops.recir_test.pallas_mul2.default,
                torch.ops.recir_test.triton_add1.default: torch.ops.recir_test.pallas_add1.default,
            }
        )
        swap_ops(gm, registry)
        for n in gm.graph.nodes:
            if n.op == "call_function":
                self.assertNotIn(
                    n.target,
                    [
                        torch.ops.recir_test.triton_mul2.default,
                        torch.ops.recir_test.triton_add1.default,
                    ],
                )

    def test_numerical_correctness_multi_swap(self) -> None:
        torch.manual_seed(42)
        model = ModelWithMultipleTritonOps()
        x = torch.randn(4, 16)
        gm = _trace_model(model)
        params = dict(model.named_parameters())
        ref_output = gm(params, x)
        registry = self._make_registry(
            {
                torch.ops.recir_test.triton_mul2.default: torch.ops.recir_test.pallas_mul2.default,
                torch.ops.recir_test.triton_add1.default: torch.ops.recir_test.pallas_add1.default,
            }
        )
        swapped_gm = copy.deepcopy(gm)
        swap_ops(swapped_gm, registry)
        swapped_output = swapped_gm(params, x)
        self.assertEqual(ref_output.shape, swapped_output.shape)
        torch.testing.assert_close(ref_output, swapped_output, atol=1e-6, rtol=1e-6)


class TestPartialLowerXLAWithOpSwap(unittest.TestCase):
    def _make_registry(self, mappings: dict[Any, Any]) -> OpSwapRegistry:
        registry = OpSwapRegistry()
        for src, dst in mappings.items():
            registry.register(src, dst)
        return registry

    def test_xla_modules_created_after_swap(self) -> None:
        gm = _trace_model(ModelWithTritonOps())
        registry = self._make_registry(
            {
                torch.ops.recir_test.triton_mul2.default: torch.ops.recir_test.pallas_mul2.default,
            }
        )
        lowered = partial_lower_xla(
            gm,
            op_swap_map=registry,
            subgraph_predicate=lambda nodes: len(nodes) >= 1,
        )
        for node in lowered.graph.nodes:
            if node.op == "call_function":
                self.assertNotEqual(
                    node.target,
                    torch.ops.recir_test.triton_mul2.default,
                )
        xla_modules = [mod for mod in lowered.modules() if isinstance(mod, _XLAModule)]
        self.assertGreater(len(xla_modules), 0)
        for xla_mod in xla_modules:
            for node in xla_mod.gm.graph.nodes:
                if node.op == "call_function":
                    self.assertNotEqual(
                        node.target,
                        torch.ops.recir_test.triton_mul2.default,
                    )


class TestPallasKernelIntegration(unittest.TestCase):
    def test_pallas_mul2_kernel_standalone(self) -> None:
        x_torch = torch.randn(4, 16)
        x_jax = t2j(x_torch)
        out_jax = _pallas_mul2_jax(x_jax)
        out_torch = torch.tensor(out_jax.__array__())
        expected = x_torch * 2.0
        torch.testing.assert_close(out_torch, expected, atol=1e-6, rtol=1e-6)

    def test_pallas_add1_kernel_standalone(self) -> None:
        x_torch = torch.randn(4, 16)
        x_jax = t2j(x_torch)
        out_jax = _pallas_add1_jax(x_jax)
        out_torch = torch.tensor(out_jax.__array__())
        expected = x_torch + 1.0
        torch.testing.assert_close(out_torch, expected, atol=1e-6, rtol=1e-6)

    def test_pallas_kernel_via_call_jax(self) -> None:
        """Verify the Pallas kernel works through torchax.interop.call_jax.

        This exercises the torch → JAX → Pallas → torch round-trip that
        a real Pallas-backed custom op would use at runtime.
        """
        recir.torchax.enable_globally()
        x = torch.randn(4, 16, device="jax")
        result = interop.call_jax(_pallas_mul2_jax, x)
        # pyre-ignore[16]: torchax tensor has `.jax()` attribute
        result_cpu = torch.tensor(result.jax().__array__())
        # pyre-ignore[16]: torchax tensor has `.jax()` attribute
        x_cpu = torch.tensor(x.jax().__array__())
        expected = x_cpu * 2.0
        # pyrefly: ignore [missing-attribute]
        self.assertEqual(result.shape, x.shape)
        torch.testing.assert_close(result_cpu, expected, atol=1e-5, rtol=1e-5)

    def test_swap_to_pallas_then_xla_lower_produces_xla_modules(self) -> None:
        """End-to-end: swap Triton → Pallas, then XLA-lower the graph.

        Verifies that after swapping, the Pallas custom ops end up inside
        _XLAModule subgraphs (not causing graph breaks).
        """
        gm = _trace_model(ModelWithTritonOps())
        registry = OpSwapRegistry()
        registry.register(
            torch.ops.recir_test.triton_mul2.default,
            torch.ops.recir_test.pallas_mul2.default,
        )
        lowered = partial_lower_xla(
            gm,
            op_swap_map=registry,
            subgraph_predicate=lambda nodes: len(nodes) >= 1,
        )
        xla_modules = [mod for mod in lowered.modules() if isinstance(mod, _XLAModule)]
        self.assertGreater(len(xla_modules), 0)
        found_pallas = False
        for xla_mod in xla_modules:
            for node in xla_mod.gm.graph.nodes:
                if node.op == "call_function":
                    self.assertNotEqual(
                        node.target,
                        torch.ops.recir_test.triton_mul2.default,
                    )
                    if node.target == torch.ops.recir_test.pallas_mul2.default:
                        found_pallas = True
        self.assertTrue(
            found_pallas, "Pallas op should be inside an _XLAModule subgraph"
        )

    def test_swap_then_xla_lower_and_execute(self) -> None:
        """End-to-end: swap → XLA-lower → execute the lowered graph.

        This test actually runs the XLA-lowered graph, triggering
        ``_compile_torchax`` which traces through torchax and dispatches
        the Pallas custom ops to their registered JAX/Pallas implementations.
        Without ``_register_pallas_ops_in_torchax()``, this test will fail.
        """
        torch.manual_seed(42)
        device = "cuda"
        model: torch.nn.Module = ModelWithTritonOps().to(device)
        x = torch.randn(4, 16, device=device)

        def fn(params: dict[str, Any], x: torch.Tensor) -> torch.Tensor:
            return torch.func.functional_call(model, params, (x,))

        params = dict(model.named_parameters())
        gm = make_fx(fn)(params, x)
        ref_output = gm(params, x)
        registry = OpSwapRegistry()
        registry.register(
            torch.ops.recir_test.triton_mul2.default,
            torch.ops.recir_test.pallas_mul2.default,
        )
        lowered_gm = partial_lower_xla(
            copy.deepcopy(gm),
            backend_type=BackendType.XLA_SHARD_MAP,
            op_swap_map=registry,
        )
        lowered_output = lowered_gm(params, x)
        self.assertEqual(ref_output.shape, lowered_output.shape)
        self.assertEqual(ref_output.device, lowered_output.device)
        torch.testing.assert_close(ref_output, lowered_output, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
