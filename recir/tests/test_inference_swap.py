# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Tests for InferenceSwapTransform — post-trace inference graph surgery.

Covers the four training/inference discrepancies the transform targets:
dropout removal, stochastic-sampling -> deterministic, module-region swap, and
partial publish (pruning a tower, e.g. "skip overarch").
"""

from __future__ import annotations

import unittest
from collections import OrderedDict
from typing import Any, Callable

import torch
import torch.fx as fx
import torch.nn as nn
from recir.region_transforms.base import RegionTransform
from recir.region_transforms.inference_swap import (
    apply_inference_swaps,
    deterministic_argmax,
    identity_passthrough,
    InferenceSwapTransform,
)
from recir.utils.graph import OP
from torch.fx.experimental.proxy_tensor import make_fx


# -- test models -----------------------------------------------------------


class DropoutModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(16, 16)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.linear(x))


class MultiDropoutModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.Linear(16, 32)
        self.d1 = nn.Dropout(p=0.3)
        self.l2 = nn.Linear(32, 16)
        self.d2 = nn.Dropout(p=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.d2(self.l2(self.d1(self.l1(x))))


class CleanModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(16, 16)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.linear(x))


class StochasticSampler(nn.Module):
    """Gumbel-style sampling head (param-free): draws noise from logits."""

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        noise = torch.rand_like(logits)
        gumbel = -torch.log(-torch.log(noise + 1e-9) + 1e-9)
        return logits + gumbel


class SamplingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(16, 8)
        self.sampler = StochasticSampler()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sampler(self.proj(x))


class AuxHead(nn.Module):
    """An auxiliary tail tower removed at publish time."""

    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.head(x))


class MultiHeadModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(16, 16)
        self.aux = AuxHead()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(x)
        return feats, self.aux(feats)


class Overarch(nn.Module):
    """Final tower; "skip overarch" publishes the upstream features instead."""

    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.mlp(x))


class OverarchModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sparse = nn.Linear(16, 16)
        self.over = Overarch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.over(self.sparse(x))


# -- helpers ---------------------------------------------------------------


def _trace(model: nn.Module, x: torch.Tensor) -> fx.GraphModule:
    """Trace in train() so training-only ops appear in the graph.

    ``pre_dispatch=True`` captures ops before the dispatcher decomposes
    composite ops, so ``aten.dropout`` survives as a single removable node
    while the graph stays executable for numerical-equivalence checks.
    """
    model.train()
    return make_fx(model, pre_dispatch=True)(x)


def _trace_with_stack(model: nn.Module, x: torch.Tensor) -> fx.GraphModule:
    """Trace with nn_module_stack populated (Level-2 module detection)."""
    model.train()
    traced_fn = lambda *args: model(*args)  # noqa: E731
    traced_fn._orig_mod = model  # type: ignore[attr-defined]
    return make_fx(traced_fn, record_module_stack=True)(x)


def _count_target(gm: fx.GraphModule, needle: str) -> int:
    return sum(
        1
        for n in gm.graph.nodes
        if n.op == OP.CALL_FUNCTION and needle in str(n.target).lower()
    )


def _fqn(cls: type[object]) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


# -- dropout (Level 1) -----------------------------------------------------


class TestDropoutRemoval(unittest.TestCase):
    def test_is_region_transform(self) -> None:
        self.assertIsInstance(InferenceSwapTransform(), RegionTransform)

    def test_remove_single_dropout(self) -> None:
        gm = _trace(DropoutModel(), torch.randn(4, 16))
        self.assertGreater(_count_target(gm, "dropout"), 0)
        apply_inference_swaps(gm)
        self.assertEqual(_count_target(gm, "dropout"), 0)

    def test_remove_multiple_dropouts(self) -> None:
        gm = _trace(MultiDropoutModel(), torch.randn(4, 16))
        self.assertGreater(_count_target(gm, "dropout"), 0)
        apply_inference_swaps(gm)
        self.assertEqual(_count_target(gm, "dropout"), 0)

    def test_numerical_equivalence_with_eval(self) -> None:
        model = DropoutModel()
        x = torch.randn(4, 16)
        model.eval()
        with torch.no_grad():
            ref = model(x)
        gm = _trace(model, x)
        apply_inference_swaps(gm)
        with torch.no_grad():
            out = gm(x)
        torch.testing.assert_close(out, ref)

    def test_clean_model_detects_nothing(self) -> None:
        model = CleanModel()
        x = torch.randn(4, 16)
        gm = _trace(model, x)
        self.assertEqual(InferenceSwapTransform()._detect_regions(gm), [])
        with torch.no_grad():
            ref = gm(x)
            apply_inference_swaps(gm)
            out = gm(x)
        torch.testing.assert_close(out, ref)

    def test_disable_dropout_removal(self) -> None:
        gm = _trace(DropoutModel(), torch.randn(4, 16))
        before = _count_target(gm, "dropout")
        apply_inference_swaps(gm, enable_dropout_removal=False)
        self.assertEqual(_count_target(gm, "dropout"), before)

    def test_idempotent(self) -> None:
        gm = _trace(DropoutModel(), torch.randn(4, 16))
        apply_inference_swaps(gm)
        n1 = len(list(gm.graph.nodes))
        apply_inference_swaps(gm)
        self.assertEqual(len(list(gm.graph.nodes)), n1)

    def test_transform_graph_hook(self) -> None:
        gm = _trace(DropoutModel(), torch.randn(4, 16))
        self.assertGreater(_count_target(gm, "dropout"), 0)
        InferenceSwapTransform().transform_graph(gm, OrderedDict())
        self.assertEqual(_count_target(gm, "dropout"), 0)


# -- stochastic sampling (Level 2 swap -> deterministic) -------------------


class TestStochasticSampling(unittest.TestCase):
    def test_sampler_replaced_with_argmax(self) -> None:
        model = SamplingModel()
        x = torch.randn(4, 16)
        gm = _trace_with_stack(model, x)
        # Random draw is present before surgery.
        self.assertGreater(_count_target(gm, "rand"), 0)

        apply_inference_swaps(
            gm,
            module_swaps={_fqn(StochasticSampler): deterministic_argmax(dim=-1)},
            enable_dropout_removal=False,
        )

        # Stochastic op gone; output is the deterministic argmax over logits.
        self.assertEqual(_count_target(gm, "rand"), 0)
        with torch.no_grad():
            out = gm(x)
            ref = torch.argmax(model.proj(x), dim=-1)
        torch.testing.assert_close(out, ref)

    def test_sampling_op_removal_toggle(self) -> None:
        class BernoulliModel(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.bernoulli(torch.sigmoid(x))

        gm = _trace(BernoulliModel(), torch.rand(4, 16))
        self.assertGreater(_count_target(gm, "bernoulli"), 0)
        apply_inference_swaps(
            gm, enable_dropout_removal=False, enable_sampling_removal=True
        )
        self.assertEqual(_count_target(gm, "bernoulli"), 0)


# -- module-region swap ----------------------------------------------------


class TestModuleRegionSwap(unittest.TestCase):
    def test_region_detected_and_replaced(self) -> None:
        class ScaleByTwo(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return (x + x) * 1.0

        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(16, 16)
                self.region = ScaleByTwo()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.region(self.linear(x))

        gm = _trace_with_stack(M(), torch.randn(4, 16))
        cls_fqn = _fqn(ScaleByTwo)

        def _double(region_gm: fx.GraphModule) -> Callable[..., Any]:
            def _apply(*inputs: torch.Tensor) -> torch.Tensor:
                return inputs[0] * 2.0

            return _apply

        before = len(list(gm.graph.nodes))
        apply_inference_swaps(
            gm, module_swaps={cls_fqn: _double}, enable_dropout_removal=False
        )
        self.assertLess(len(list(gm.graph.nodes)), before)


# -- partial publish (prune) -----------------------------------------------


class TestPartialPublish(unittest.TestCase):
    def test_prune_aux_tail_tower(self) -> None:
        model = MultiHeadModel()
        x = torch.randn(4, 16)
        gm = _trace_with_stack(model, x)
        linears_before = _count_target(gm, "addmm") + _count_target(gm, "linear")

        apply_inference_swaps(
            gm,
            prune_module_classes={_fqn(AuxHead)},
            enable_dropout_removal=False,
        )

        # Aux tower's Linear is gone; the model now returns a single output.
        linears_after = _count_target(gm, "addmm") + _count_target(gm, "linear")
        self.assertLess(linears_after, linears_before)
        self.assertEqual(_count_target(gm, "relu"), 0)
        with torch.no_grad():
            out = gm(x)
        self.assertEqual(len(out), 1)
        torch.testing.assert_close(out[0], model.backbone(x))

    def test_skip_overarch_exposes_upstream(self) -> None:
        model = OverarchModel()
        x = torch.randn(4, 16)
        gm = _trace_with_stack(model, x)
        self.assertGreater(_count_target(gm, "sigmoid"), 0)

        apply_inference_swaps(
            gm,
            prune_module_classes={_fqn(Overarch)},
            enable_dropout_removal=False,
        )

        # Overarch removed; published graph outputs the upstream features.
        self.assertEqual(_count_target(gm, "sigmoid"), 0)
        with torch.no_grad():
            out = gm(x)
        torch.testing.assert_close(out, model.sparse(x))


# -- registry bridge / extensibility ---------------------------------------


class TestRegistryBridge(unittest.TestCase):
    def test_from_module_swap_registry(self) -> None:
        class FakeRegistry:
            registered_types: list[type[object]] = [StochasticSampler]

            def get_factory(self, _cls: type[object]) -> Callable[[], nn.Module]:
                return lambda: nn.Identity()

        transform = InferenceSwapTransform.from_module_swap_registry(FakeRegistry())
        self.assertIn(_fqn(StochasticSampler), transform._module_swaps)

    def test_custom_op_matcher(self) -> None:
        def _match_relu(node: fx.Node) -> bool:
            return (
                node.op == OP.CALL_FUNCTION
                and node.target is torch.ops.aten.relu.default
            )

        gm = _trace(CleanModel(), torch.randn(4, 16))
        regions = InferenceSwapTransform(
            enable_dropout_removal=False,
            custom_op_matchers=[(_match_relu, identity_passthrough)],
        )._detect_regions(gm)
        self.assertEqual(len(regions), 1)
