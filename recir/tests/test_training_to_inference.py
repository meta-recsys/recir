# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Tests for the training-step -> inference graph bridge.

Coverage:
- strip_grad_tags: removal + rewire, multi-node, no-op when absent
- strip_collectives: removal + rewire, custom names, no-op, executable identity
- inputs_to_ops: single/multi substring, dedup, no-match
- select_inference_outputs: single, multi (tuple), empty -> raise
- fabricate_inference_graph: loss cut + DCE, dropout/sampling/prune composition,
  collective + loss combined, forwarded kwargs, bad node name -> raise
"""

from __future__ import annotations

import unittest
from collections.abc import Callable

import torch
import torch.fx as fx
import torch.nn as nn
from recir.region_transforms.inference_swap import (
    deterministic_argmax,
    dtype_cast_matcher,
    identity_passthrough,
)
from recir.region_transforms.training_to_inference import (
    fabricate_inference_graph,
    inputs_to_ops,
    mark_inference_output,
    marked_inference_outputs,
    select_inference_outputs,
    strip_collectives,
    strip_grad_tags,
)
from recir.utils.graph import OP
from torch.fx.experimental.proxy_tensor import make_fx


# Local identity ops named like the real training-only ops, so the bridge's
# name-based matching can be exercised without a distributed process group.
def _define_test_ops() -> None:
    for name in ("tag_grad", "all_reduce", "wait_tensor", "all_gather"):
        qual = f"recir_t2i_test::{name}"
        try:
            torch.library.define(qual, "(Tensor x) -> Tensor")
            torch.library.impl(qual, "CompositeExplicitAutograd", lambda x: x)
        except RuntimeError:
            pass  # already registered in this process


_define_test_ops()
_TAG_GRAD: Callable[..., object] = torch.ops.recir_t2i_test.tag_grad.default
_ALL_REDUCE: Callable[..., object] = torch.ops.recir_t2i_test.all_reduce.default
_WAIT_TENSOR: Callable[..., object] = torch.ops.recir_t2i_test.wait_tensor.default
_ALL_GATHER: Callable[..., object] = torch.ops.recir_t2i_test.all_gather.default


def _count(gm: fx.GraphModule, needle: str) -> int:
    return sum(
        1
        for n in gm.graph.nodes
        if n.op == OP.CALL_FUNCTION and needle in str(n.target).lower()
    )


def _fqn(cls: type[object]) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _trace_with_stack(model: nn.Module, x: torch.Tensor) -> fx.GraphModule:
    model.train()
    traced_fn = lambda *args: model(*args)  # noqa: E731
    traced_fn._orig_mod = model  # type: ignore[attr-defined]
    return make_fx(traced_fn, record_module_stack=True)(x)


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(16, 4)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.linear(x))


class AuxHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.head(x))


class TwoHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(16, 4)
        self.aux = AuxHead()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(x)
        return feats, self.aux(feats)


class Sampler(nn.Module):
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits + torch.rand_like(logits)


class SamplingNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(16, 4)
        self.sampler = Sampler()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sampler(self.proj(x))


class TestStripGradTags(unittest.TestCase):
    def test_removes_tag_grad_and_rewires(self) -> None:
        g = fx.Graph()
        x = g.placeholder("x")
        tagged = g.call_function(_TAG_GRAD, (x,))
        relu = g.call_function(torch.ops.aten.relu.default, (tagged,))
        g.output((relu,))
        gm = fx.GraphModule(nn.Module(), g)

        removed = strip_grad_tags(gm)
        self.assertEqual(removed, 1)
        self.assertEqual(_count(gm, "tag_grad"), 0)
        relu_node = next(n for n in gm.graph.nodes if n.op == OP.CALL_FUNCTION)
        self.assertEqual(relu_node.args[0].op, OP.PLACEHOLDER)

    def test_removes_multiple_tag_grads(self) -> None:
        g = fx.Graph()
        x = g.placeholder("x")
        t1 = g.call_function(_TAG_GRAD, (x,))
        r = g.call_function(torch.ops.aten.relu.default, (t1,))
        t2 = g.call_function(_TAG_GRAD, (r,))
        g.output((t2,))
        gm = fx.GraphModule(nn.Module(), g)

        self.assertEqual(strip_grad_tags(gm), 2)
        self.assertEqual(_count(gm, "tag_grad"), 0)

    def test_noop_when_absent(self) -> None:
        g = fx.Graph()
        x = g.placeholder("x")
        r = g.call_function(torch.ops.aten.relu.default, (x,))
        g.output((r,))
        gm = fx.GraphModule(nn.Module(), g)
        self.assertEqual(strip_grad_tags(gm), 0)


class TestStripCollectives(unittest.TestCase):
    def test_removes_all_reduce_and_wait(self) -> None:
        g = fx.Graph()
        x = g.placeholder("x")
        ar = g.call_function(_ALL_REDUCE, (x,))
        wait = g.call_function(_WAIT_TENSOR, (ar,))
        relu = g.call_function(torch.ops.aten.relu.default, (wait,))
        g.output((relu,))
        gm = fx.GraphModule(nn.Module(), g)

        self.assertEqual(strip_collectives(gm), 2)
        self.assertEqual(_count(gm, "all_reduce"), 0)
        self.assertEqual(_count(gm, "wait_tensor"), 0)

    def test_noop_when_absent(self) -> None:
        g = fx.Graph()
        x = g.placeholder("x")
        r = g.call_function(torch.ops.aten.relu.default, (x,))
        g.output((r,))
        gm = fx.GraphModule(nn.Module(), g)
        self.assertEqual(strip_collectives(gm), 0)

    def test_custom_collective_names(self) -> None:
        g = fx.Graph()
        x = g.placeholder("x")
        ag = g.call_function(_ALL_GATHER, (x,))
        relu = g.call_function(torch.ops.aten.relu.default, (ag,))
        g.output((relu,))
        gm = fx.GraphModule(nn.Module(), g)

        # all_gather is not in the default set, so it survives by default ...
        self.assertEqual(strip_collectives(gm), 0)
        self.assertEqual(_count(gm, "all_gather"), 1)
        # ... but is stripped when named explicitly.
        self.assertEqual(strip_collectives(gm, collective_names=("all_gather",)), 1)
        self.assertEqual(_count(gm, "all_gather"), 0)

    def test_executable_identity_strip(self) -> None:
        g = fx.Graph()
        x = g.placeholder("x")
        ar = g.call_function(_ALL_REDUCE, (x,))
        relu = g.call_function(torch.ops.aten.relu.default, (ar,))
        g.output((relu,))
        gm = fx.GraphModule(nn.Module(), g)

        sample = torch.randn(4, 4)
        strip_collectives(gm)
        self.assertEqual(_count(gm, "all_reduce"), 0)
        out = gm(sample)
        torch.testing.assert_close(out[0], torch.relu(sample))


class TestInputsToOps(unittest.TestCase):
    def test_selects_single_loss_input(self) -> None:
        net = Net()
        net.train()
        x = torch.randn(4, 16)
        gm = make_fx(lambda inp: net(inp).sum(), pre_dispatch=True)(x)
        self.assertEqual(len(inputs_to_ops({"sum"})(gm)), 1)

    def test_no_match_returns_empty(self) -> None:
        net = Net()
        net.train()
        x = torch.randn(4, 16)
        gm = make_fx(net, pre_dispatch=True)(x)
        self.assertEqual(inputs_to_ops({"nonexistent_op_xyz"})(gm), [])

    def test_multiple_substrings_dedup(self) -> None:
        g = fx.Graph()
        x = g.placeholder("x")
        a = g.call_function(torch.ops.aten.relu.default, (x,))
        # both ops consume `a`; `a` must appear once across the union.
        s = g.call_function(torch.ops.aten.sum.default, (a,))
        m = g.call_function(torch.ops.aten.mean.default, (a,))
        g.output((s, m))
        gm = fx.GraphModule(nn.Module(), g)
        selected = inputs_to_ops({"sum", "mean"})(gm)
        self.assertEqual(selected, [a])


class TestSelectInferenceOutputs(unittest.TestCase):
    def test_empty_raises(self) -> None:
        net = Net()
        net.train()
        gm = make_fx(net, pre_dispatch=True)(torch.randn(4, 16))
        with self.assertRaises(ValueError):
            select_inference_outputs(gm, [])

    def test_multiple_outputs_become_tuple(self) -> None:
        g = fx.Graph()
        x = g.placeholder("x")
        a = g.call_function(torch.ops.aten.relu.default, (x,))
        b = g.call_function(torch.ops.aten.sigmoid.default, (x,))
        c = g.call_function(torch.ops.aten.tanh.default, (a,))
        g.output((c,))
        gm = fx.GraphModule(nn.Module(), g)

        select_inference_outputs(gm, [a, b])
        sample = torch.randn(4, 4)
        out = gm(sample)
        self.assertEqual(len(out), 2)
        torch.testing.assert_close(out[0], torch.relu(sample))
        torch.testing.assert_close(out[1], torch.sigmoid(sample))


class TestFabricateInferenceGraph(unittest.TestCase):
    def test_strips_loss_and_dropout_end_to_end(self) -> None:
        net = Net()
        net.train()
        x = torch.randn(4, 16)
        gm = make_fx(lambda inp: net(inp).sum(), pre_dispatch=True)(x)
        self.assertGreater(_count(gm, "sum"), 0)
        self.assertGreater(_count(gm, "dropout"), 0)

        fabricate_inference_graph(gm, inference_outputs=inputs_to_ops({"sum"}))

        self.assertEqual(_count(gm, "sum"), 0)
        self.assertEqual(_count(gm, "dropout"), 0)
        net.eval()
        with torch.no_grad():
            ref = net(x)
            out = gm(x)
        # Selected inference outputs are returned as a tuple.
        torch.testing.assert_close(out[0], ref)

    def test_forward_only_graph_keeps_outputs(self) -> None:
        net = Net()
        net.train()
        x = torch.randn(4, 16)
        gm = make_fx(net, pre_dispatch=True)(x)
        fabricate_inference_graph(gm)
        self.assertEqual(_count(gm, "dropout"), 0)
        net.eval()
        with torch.no_grad():
            ref = net(x)
            out = gm(x)
        torch.testing.assert_close(out, ref)

    def test_select_by_node_name(self) -> None:
        net = Net()
        net.train()
        x = torch.randn(4, 16)
        gm = make_fx(lambda inp: net(inp).sum(), pre_dispatch=True)(x)
        pred = inputs_to_ops({"sum"})(gm)[0]
        fabricate_inference_graph(gm, inference_outputs=[pred.name])
        self.assertEqual(_count(gm, "sum"), 0)
        self.assertEqual(_count(gm, "dropout"), 0)

    def test_unknown_node_name_raises(self) -> None:
        net = Net()
        net.train()
        gm = make_fx(net, pre_dispatch=True)(torch.randn(4, 16))
        with self.assertRaises(ValueError):
            fabricate_inference_graph(gm, inference_outputs=["does_not_exist"])

    def test_disable_collective_strip_is_noop(self) -> None:
        net = Net()
        net.train()
        x = torch.randn(4, 16)
        gm = make_fx(net, pre_dispatch=True)(x)
        fabricate_inference_graph(gm, strip_training_collectives=False)
        self.assertEqual(_count(gm, "dropout"), 0)

    def test_forwards_disable_dropout_removal(self) -> None:
        net = Net()
        net.train()
        x = torch.randn(4, 16)
        gm = make_fx(net, pre_dispatch=True)(x)
        before = _count(gm, "dropout")
        fabricate_inference_graph(gm, enable_dropout_removal=False)
        self.assertEqual(_count(gm, "dropout"), before)

    def test_strips_collective_and_loss_combined(self) -> None:
        g = fx.Graph()
        x = g.placeholder("x")
        ar = g.call_function(_ALL_REDUCE, (x,))
        relu = g.call_function(torch.ops.aten.relu.default, (ar,))
        loss = g.call_function(torch.ops.aten.sum.default, (relu,))
        g.output((loss,))
        gm = fx.GraphModule(nn.Module(), g)

        fabricate_inference_graph(gm, inference_outputs=inputs_to_ops({"sum"}))
        self.assertEqual(_count(gm, "all_reduce"), 0)
        self.assertEqual(_count(gm, "sum"), 0)
        sample = torch.randn(4, 4)
        # Selected inference outputs are returned as a tuple.
        out = gm(sample)
        torch.testing.assert_close(out[0], torch.relu(sample))

    def test_composes_with_module_prune(self) -> None:
        model = TwoHead()
        x = torch.randn(4, 16)
        gm = _trace_with_stack(model, x)
        fabricate_inference_graph(gm, prune_module_classes={_fqn(AuxHead)})
        with torch.no_grad():
            out = gm(x)
        self.assertEqual(len(out), 1)
        torch.testing.assert_close(out[0], model.backbone(x))

    def test_composes_with_sampling_swap(self) -> None:
        model = SamplingNet()
        x = torch.randn(4, 16)
        gm = _trace_with_stack(model, x)
        self.assertGreater(_count(gm, "rand"), 0)

        fabricate_inference_graph(
            gm,
            module_swaps={_fqn(Sampler): deterministic_argmax(dim=-1)},
        )

        self.assertEqual(_count(gm, "rand"), 0)
        with torch.no_grad():
            out = gm(x)
            ref = torch.argmax(model.proj(x), dim=-1)
        torch.testing.assert_close(out, ref)


# Models exercising marker-based inference-output selection (the DLRMv3 pattern:
# the prediction is tagged but the *training* step returns only the loss).


class PredAndLossNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pred = mark_inference_output(self.linear(x))
        return pred.sum()  # training returns the loss; pred is otherwise unused


class PredWithDropoutNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(16, 4)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pred = mark_inference_output(self.dropout(self.linear(x)))
        return pred.sum()


class MarkedInferenceOutputTest(unittest.TestCase):
    def test_marker_selects_prediction_drops_loss(self) -> None:
        model = PredAndLossNet()
        x = torch.randn(4, 16)
        gm = make_fx(model, pre_dispatch=True)(x)

        fabricate_inference_graph(gm, inference_outputs=marked_inference_outputs())

        # The loss reduction is dropped; the output is the marked prediction.
        self.assertEqual(_count(gm, "sum"), 0)
        with torch.no_grad():
            out = gm(x)
        torch.testing.assert_close(out[0], model.linear(x))

    def test_dropout_removed_in_prediction_cone(self) -> None:
        model = PredWithDropoutNet()
        model.train()
        x = torch.randn(4, 16)
        gm = make_fx(model, pre_dispatch=True)(x)
        self.assertGreater(_count(gm, "dropout"), 0)

        fabricate_inference_graph(gm, inference_outputs=marked_inference_outputs())

        self.assertEqual(_count(gm, "dropout"), 0)
        with torch.no_grad():
            out1 = gm(x)
            out2 = gm(x)
        # Deterministic (dropout gone) and equal to the eval-mode prediction.
        torch.testing.assert_close(out1[0], out2[0])
        torch.testing.assert_close(out1[0], model.linear(x))

    def test_no_marker_selector_returns_empty(self) -> None:
        model = Net()
        model.train()
        gm = make_fx(model, pre_dispatch=True)(torch.randn(4, 16))
        self.assertEqual(marked_inference_outputs()(gm), [])


class Bf16Net(nn.Module):
    """Mimics DLRMv3's training-only bf16 autocast: cast in, compute, cast back."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear(x).to(torch.bfloat16)
        h = h * 2.0
        pred = mark_inference_output(h.to(torch.float32))
        return pred.sum()


class Bf16CastMatcherTest(unittest.TestCase):
    def _bf16_casts(self, gm: fx.GraphModule) -> int:
        match = dtype_cast_matcher(torch.bfloat16)
        return sum(1 for n in gm.graph.nodes if match(n))

    def test_bf16_casts_stripped_restores_fp32(self) -> None:
        model = Bf16Net()
        x = torch.randn(4, 16)
        gm = make_fx(model, pre_dispatch=True)(x)
        self.assertGreater(self._bf16_casts(gm), 0)

        fabricate_inference_graph(
            gm,
            inference_outputs=marked_inference_outputs(),
            custom_op_matchers=[(dtype_cast_matcher(), identity_passthrough)],
        )

        # No bf16 cast remains; the prediction is the pure-fp32 computation.
        self.assertEqual(self._bf16_casts(gm), 0)
        with torch.no_grad():
            out = gm(x)
        torch.testing.assert_close(out[0], model.linear(x) * 2.0)
