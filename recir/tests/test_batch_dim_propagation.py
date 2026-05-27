# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Unit tests for `propagate_batch_input_meta`.

These tests construct tiny FX graphs by hand, prime the placeholder(s)
with ``is_batch_input`` / ``batch_dim`` (mimicking what ``_tag_batch_inputs``
does at top-level), then run propagation and assert the call_function
output node inherits the right tags.

The four ops covered here previously broke propagation (logged by
``propagate_batch_input_meta`` as "input was batched but op not in
allowlist") and caused JAX broadcast-shape mismatches at lower time
because the surgery-spliced subgraph placeholders defaulted to replicated:

  * ``recir::mark_boundary`` -- pure identity (used as graph-break marker)
  * ``aten.expand`` -- broadcasts size-1 dims; preserves batch_dim
  * ``aten.view`` -- collapse / expand trailing dims; preserves batch_dim
    when sizes[batch_dim] matches the input size
  * ``aten.embedding_dense_backward`` -- backward grad for embedding
"""

import unittest

import torch
from recir.batch_dim_propagation import propagate_batch_input_meta

# Importing recir.ops.marker registers the `recir::mark_boundary` custom op,
# which the tests below construct call_function nodes against. Without this
# import, `torch.ops.recir.mark_boundary` is undefined at test-collection
# time.
from recir.ops import marker as _marker  # noqa: F401
from torch import fx


def _make_placeholder(
    graph: fx.Graph, name: str, val: torch.Tensor, batch_dim: int = 0
) -> fx.Node:
    """Create a placeholder node tagged like a top-level batch input."""
    node = graph.placeholder(name)
    node.meta["val"] = val
    node.meta["is_batch_input"] = True
    node.meta["batch_dim"] = batch_dim
    node.meta["is_weight"] = False
    return node


def _untagged_placeholder(graph: fx.Graph, name: str, val: torch.Tensor) -> fx.Node:
    """Create a placeholder node *without* batch tagging (e.g. a weight)."""
    node = graph.placeholder(name)
    node.meta["val"] = val
    node.meta["is_batch_input"] = False
    node.meta["is_weight"] = True
    return node


def _set_val(node: fx.Node, val: torch.Tensor) -> None:
    node.meta["val"] = val


class PropagateBatchInputMetaTest(unittest.TestCase):
    """Tests covering the 4 ops added in v7.1 fix #2."""

    def _run_propagation_on(self, graph: fx.Graph) -> fx.GraphModule:
        gm = fx.GraphModule(torch.nn.Module(), graph)
        propagate_batch_input_meta(gm)
        return gm

    def test_mark_boundary_passthrough(self) -> None:
        """recir::mark_boundary(x, name) must inherit is_batch_input from x."""
        graph = fx.Graph()
        x = _make_placeholder(graph, "x", torch.empty(128, 32), batch_dim=0)
        marker = graph.call_function(
            torch.ops.recir.mark_boundary.default, args=(x, "engram::out")
        )
        _set_val(marker, torch.empty(128, 32))
        graph.output(marker)

        self._run_propagation_on(graph)

        self.assertTrue(marker.meta.get("is_batch_input"))
        self.assertEqual(marker.meta.get("batch_dim"), 0)

    def test_expand_preserving_batch(self) -> None:
        """expand([B, 1, D] -> [B, 1, D]) must propagate batch_dim=0."""
        graph = fx.Graph()
        x = _make_placeholder(graph, "x", torch.empty(128, 1, 128), batch_dim=0)
        expanded = graph.call_function(
            torch.ops.aten.expand.default, args=(x, [128, 1, 128])
        )
        _set_val(expanded, torch.empty(128, 1, 128))
        graph.output(expanded)

        self._run_propagation_on(graph)

        self.assertTrue(expanded.meta.get("is_batch_input"))
        self.assertEqual(expanded.meta.get("batch_dim"), 0)

    def test_expand_dash_one_preserves_batch(self) -> None:
        """expand([B, 1, D] with size -1 at batch_dim) must propagate."""
        graph = fx.Graph()
        x = _make_placeholder(graph, "x", torch.empty(128, 1, 128), batch_dim=0)
        expanded = graph.call_function(
            torch.ops.aten.expand.default, args=(x, [-1, 16, 128])
        )
        _set_val(expanded, torch.empty(128, 16, 128))
        graph.output(expanded)

        self._run_propagation_on(graph)

        self.assertTrue(expanded.meta.get("is_batch_input"))
        self.assertEqual(expanded.meta.get("batch_dim"), 0)

    def test_expand_changing_batch_propagates(self) -> None:
        """expand that broadcasts batch dim size from 1 still propagates.

        We use the relaxed rank-check rule, so even a size-1 broadcast
        keeps the tensor flagged as batch-input. This is the right
        behavior for engram's [B, 1, D] -> [B, S, D] pattern after the
        size-1 dim has been moved by an earlier permute.
        """
        graph = fx.Graph()
        x = _make_placeholder(graph, "x", torch.empty(1, 1, 128), batch_dim=0)
        expanded = graph.call_function(
            torch.ops.aten.expand.default, args=(x, [4096, 16, 128])
        )
        _set_val(expanded, torch.empty(4096, 16, 128))
        graph.output(expanded)

        self._run_propagation_on(graph)

        self.assertTrue(expanded.meta.get("is_batch_input"))
        self.assertEqual(expanded.meta.get("batch_dim"), 0)

    def test_view_preserving_batch(self) -> None:
        """view([B, 16, D] -> [B, 16*D]) must propagate batch_dim=0."""
        graph = fx.Graph()
        x = _make_placeholder(graph, "x", torch.empty(128, 16, 64), batch_dim=0)
        viewed = graph.call_function(
            torch.ops.aten.view.default, args=(x, [128, 16 * 64])
        )
        _set_val(viewed, torch.empty(128, 1024))
        graph.output(viewed)

        self._run_propagation_on(graph)

        self.assertTrue(viewed.meta.get("is_batch_input"))
        self.assertEqual(viewed.meta.get("batch_dim"), 0)

    def test_view_flatten_batch_seq_propagates(self) -> None:
        """view([B, 16, D] -> [B*16, D]) still propagates batch_dim=0.

        Flattening batch+seq into a leading axis is the common engram
        pattern; the result is still sharded along axis 0 (just with
        per-process size (B/world)*16 instead of B/world).
        """
        graph = fx.Graph()
        x = _make_placeholder(graph, "x", torch.empty(128, 16, 64), batch_dim=0)
        viewed = graph.call_function(
            torch.ops.aten.view.default, args=(x, [128 * 16, 64])
        )
        _set_val(viewed, torch.empty(2048, 64))
        graph.output(viewed)

        self._run_propagation_on(graph)

        self.assertTrue(viewed.meta.get("is_batch_input"))
        self.assertEqual(viewed.meta.get("batch_dim"), 0)

    def test_view_drops_below_batch_dim_does_not_propagate(self) -> None:
        """view that produces a shape with rank <= batch_dim must NOT propagate."""
        graph = fx.Graph()
        x = _make_placeholder(graph, "x", torch.empty(128, 16, 64), batch_dim=2)
        # New shape has rank 2 but batch_dim is 2 -- no batch slot in output.
        viewed = graph.call_function(torch.ops.aten.view.default, args=(x, [2048, 64]))
        _set_val(viewed, torch.empty(2048, 64))
        graph.output(viewed)

        self._run_propagation_on(graph)

        self.assertFalse(viewed.meta.get("is_batch_input"))

    def test_unsafe_view_preserving_batch(self) -> None:
        """_unsafe_view (post-decomp) should be handled like view."""
        graph = fx.Graph()
        x = _make_placeholder(graph, "x", torch.empty(128, 16, 64), batch_dim=0)
        viewed = graph.call_function(
            torch.ops.aten._unsafe_view.default, args=(x, [128, 1024])
        )
        _set_val(viewed, torch.empty(128, 1024))
        graph.output(viewed)

        self._run_propagation_on(graph)

        self.assertTrue(viewed.meta.get("is_batch_input"))
        self.assertEqual(viewed.meta.get("batch_dim"), 0)

    def test_embedding_dense_backward_propagates_from_grad(self) -> None:
        """embedding_dense_backward(grad, indices, V, ..) inherits batch tag
        from grad (arg 0)."""
        graph = fx.Graph()
        grad = _make_placeholder(graph, "grad", torch.empty(128, 16, 64), batch_dim=0)
        indices = _untagged_placeholder(
            graph, "indices", torch.empty(128, 16, dtype=torch.int64)
        )
        bwd = graph.call_function(
            torch.ops.aten.embedding_dense_backward.default,
            args=(grad, indices, 1024, -1, False),
        )
        # Output shape is [num_weights, embed_dim] -- batch axis collapsed by
        # the reduction, but we propagate the tag so downstream surgery can
        # tell this grad came from a batched embedding.
        _set_val(bwd, torch.empty(1024, 64))
        graph.output(bwd)

        self._run_propagation_on(graph)

        self.assertTrue(bwd.meta.get("is_batch_input"))
        self.assertEqual(bwd.meta.get("batch_dim"), 0)

    def test_chained_propagation_end_to_end(self) -> None:
        """mark_boundary -> view -> expand: all three should propagate."""
        graph = fx.Graph()
        x = _make_placeholder(graph, "x", torch.empty(128, 16, 64), batch_dim=0)

        marked = graph.call_function(
            torch.ops.recir.mark_boundary.default, args=(x, "boundary")
        )
        _set_val(marked, torch.empty(128, 16, 64))

        viewed = graph.call_function(
            torch.ops.aten.view.default, args=(marked, [128, 1024])
        )
        _set_val(viewed, torch.empty(128, 1024))

        # expand with size-1 dim inserted: view -> unsqueeze -> expand
        # Use expand_copy with size -1 to skip the unsqueeze step.
        expanded = graph.call_function(
            torch.ops.aten.expand.default, args=(viewed, [-1, 1024])
        )
        _set_val(expanded, torch.empty(128, 1024))
        graph.output(expanded)

        self._run_propagation_on(graph)

        for node in (marked, viewed, expanded):
            self.assertTrue(
                node.meta.get("is_batch_input"),
                f"is_batch_input not propagated through {node.target}",
            )
            self.assertEqual(node.meta.get("batch_dim"), 0)

    def test_unbatched_input_does_not_propagate(self) -> None:
        """Untagged inputs stay untagged after each op."""
        graph = fx.Graph()
        x = _untagged_placeholder(graph, "x", torch.empty(128, 16, 64))

        marked = graph.call_function(
            torch.ops.recir.mark_boundary.default, args=(x, "weight")
        )
        _set_val(marked, torch.empty(128, 16, 64))
        graph.output(marked)

        self._run_propagation_on(graph)

        self.assertFalse(marked.meta.get("is_batch_input", False))

    def test_remainder_tensor_propagates(self) -> None:
        """aten.remainder.Tensor preserves shape; batch_dim propagates."""
        graph = fx.Graph()
        x = _make_placeholder(graph, "x", torch.empty(128, 16, 4), batch_dim=0)
        divisor = _untagged_placeholder(graph, "d", torch.empty(1, 1, 4))
        out = graph.call_function(torch.ops.aten.remainder.Tensor, args=(x, divisor))
        _set_val(out, torch.empty(128, 16, 4))
        graph.output(out)

        self._run_propagation_on(graph)

        self.assertTrue(out.meta.get("is_batch_input"))
        self.assertEqual(out.meta.get("batch_dim"), 0)

    def test_convolution_propagates(self) -> None:
        """aten.convolution preserves the N-axis batch_dim."""
        graph = fx.Graph()
        # [N, C, L] = [128, 32, 64]
        x = _make_placeholder(graph, "x", torch.empty(128, 32, 64), batch_dim=0)
        weight = _untagged_placeholder(graph, "w", torch.empty(32, 1, 4))
        # convolution(input, weight, bias, stride, padding, dilation,
        #             transposed, output_padding, groups)
        out = graph.call_function(
            torch.ops.aten.convolution.default,
            args=(x, weight, None, [1], [0], [1], False, [0], 32),
        )
        _set_val(out, torch.empty(128, 32, 61))
        graph.output(out)

        self._run_propagation_on(graph)

        self.assertTrue(out.meta.get("is_batch_input"))
        self.assertEqual(out.meta.get("batch_dim"), 0)

    def test_sum_dim_intlist_non_batch_propagates_with_keepdim(self) -> None:
        """sum(input, dim=[-1], keepdim=True) on [B, L, D] -> [B, L, 1]."""
        graph = fx.Graph()
        x = _make_placeholder(graph, "x", torch.empty(128, 16, 64), batch_dim=0)
        out = graph.call_function(torch.ops.aten.sum.dim_IntList, args=(x, [-1], True))
        _set_val(out, torch.empty(128, 16, 1))
        graph.output(out)

        self._run_propagation_on(graph)

        self.assertTrue(out.meta.get("is_batch_input"))
        self.assertEqual(out.meta.get("batch_dim"), 0)

    def test_sum_dim_intlist_non_batch_no_keepdim(self) -> None:
        """sum(input, dim=[-1], keepdim=False) on [B, L, D] -> [B, L]."""
        graph = fx.Graph()
        x = _make_placeholder(graph, "x", torch.empty(128, 16, 64), batch_dim=0)
        out = graph.call_function(torch.ops.aten.sum.dim_IntList, args=(x, [-1], False))
        _set_val(out, torch.empty(128, 16))
        graph.output(out)

        self._run_propagation_on(graph)

        self.assertTrue(out.meta.get("is_batch_input"))
        # batch dim unchanged because the reduced dim (-1=2) is to the
        # RIGHT of batch_dim=0.
        self.assertEqual(out.meta.get("batch_dim"), 0)

    def test_sum_dim_intlist_reduces_batch_does_not_propagate(self) -> None:
        """sum(input, dim=[0]) collapses the batch axis -> no propagation."""
        graph = fx.Graph()
        x = _make_placeholder(graph, "x", torch.empty(128, 16, 64), batch_dim=0)
        out = graph.call_function(torch.ops.aten.sum.dim_IntList, args=(x, [0], False))
        _set_val(out, torch.empty(16, 64))
        graph.output(out)

        self._run_propagation_on(graph)

        self.assertFalse(out.meta.get("is_batch_input"))

    def test_mean_dim_non_batch_propagates(self) -> None:
        """mean.dim on a non-batch axis preserves batch_dim (RMSNorm pattern)."""
        graph = fx.Graph()
        x = _make_placeholder(graph, "x", torch.empty(128, 16, 64), batch_dim=0)
        out = graph.call_function(torch.ops.aten.mean.dim, args=(x, [-1], True))
        _set_val(out, torch.empty(128, 16, 1))
        graph.output(out)

        self._run_propagation_on(graph)

        self.assertTrue(out.meta.get("is_batch_input"))
        self.assertEqual(out.meta.get("batch_dim"), 0)

    def test_alias_passthrough(self) -> None:
        """aten.alias is a pure view; batch_dim is preserved."""
        graph = fx.Graph()
        x = _make_placeholder(graph, "x", torch.empty(128, 16, 64), batch_dim=0)
        out = graph.call_function(torch.ops.aten.alias.default, args=(x,))
        _set_val(out, torch.empty(128, 16, 64))
        graph.output(out)

        self._run_propagation_on(graph)

        self.assertTrue(out.meta.get("is_batch_input"))
        self.assertEqual(out.meta.get("batch_dim"), 0)

    def test_sign_cos_sin_propagate(self) -> None:
        """sign / cos / sin are pure elementwise unary; batch_dim preserved."""
        for op in (
            torch.ops.aten.sign.default,
            torch.ops.aten.cos.default,
            torch.ops.aten.sin.default,
        ):
            with self.subTest(op=str(op)):
                graph = fx.Graph()
                x = _make_placeholder(graph, "x", torch.empty(128, 16, 64), batch_dim=0)
                out = graph.call_function(op, args=(x,))
                _set_val(out, torch.empty(128, 16, 64))
                graph.output(out)

                self._run_propagation_on(graph)

                self.assertTrue(
                    out.meta.get("is_batch_input"),
                    f"is_batch_input not propagated through {op}",
                )
                self.assertEqual(out.meta.get("batch_dim"), 0)

    def test_add__scalar_inplace_propagates(self) -> None:
        """aten.add_.Scalar (in-place, self is arg 0) preserves batch_dim.

        After functionalization, the FX call_function for the in-place op
        still has target=aten.add_.Scalar; the output is the modified
        self. We propagate the batch tag from self.
        """
        graph = fx.Graph()
        x = _make_placeholder(graph, "x", torch.empty(128, 16, 64), batch_dim=0)
        out = graph.call_function(torch.ops.aten.add_.Scalar, args=(x, 1.0))
        _set_val(out, torch.empty(128, 16, 64))
        graph.output(out)

        self._run_propagation_on(graph)

        self.assertTrue(out.meta.get("is_batch_input"))
        self.assertEqual(out.meta.get("batch_dim"), 0)

    def test_lerp__scalar_inplace_propagates(self) -> None:
        """aten.lerp_.Scalar(self, end, weight): self is arg 0, batch_dim preserved.

        end is typically an unbatched accumulator; both contributing to
        the consolidator with the same batch_dim is also OK.
        """
        graph = fx.Graph()
        x = _make_placeholder(graph, "x", torch.empty(128, 16, 64), batch_dim=0)
        end = _untagged_placeholder(graph, "end", torch.empty(128, 16, 64))
        out = graph.call_function(torch.ops.aten.lerp_.Scalar, args=(x, end, 0.5))
        _set_val(out, torch.empty(128, 16, 64))
        graph.output(out)

        self._run_propagation_on(graph)

        self.assertTrue(out.meta.get("is_batch_input"))
        self.assertEqual(out.meta.get("batch_dim"), 0)

    def test_addcmul__addcdiv__propagate(self) -> None:
        """Optimizer-step in-place ops addcmul_ / addcdiv_: self is arg 0."""
        for op in (
            torch.ops.aten.addcmul_.default,
            torch.ops.aten.addcdiv_.default,
        ):
            with self.subTest(op=str(op)):
                graph = fx.Graph()
                x = _make_placeholder(graph, "x", torch.empty(128, 16, 64), batch_dim=0)
                t1 = _untagged_placeholder(graph, "t1", torch.empty(128, 16, 64))
                t2 = _untagged_placeholder(graph, "t2", torch.empty(128, 16, 64))
                out = graph.call_function(op, args=(x, t1, t2), kwargs={"value": 1.0})
                _set_val(out, torch.empty(128, 16, 64))
                graph.output(out)

                self._run_propagation_on(graph)

                self.assertTrue(
                    out.meta.get("is_batch_input"),
                    f"is_batch_input not propagated through {op}",
                )
                self.assertEqual(out.meta.get("batch_dim"), 0)

    def test_inplace_binary_propagates(self) -> None:
        """In-place add_/sub_/mul_/div_ (.Scalar and .Tensor) preserve batch_dim."""
        for op in (
            torch.ops.aten.add_.Tensor,
            torch.ops.aten.sub_.Scalar,
            torch.ops.aten.sub_.Tensor,
            torch.ops.aten.mul_.Scalar,
            torch.ops.aten.mul_.Tensor,
            torch.ops.aten.div_.Scalar,
            torch.ops.aten.div_.Tensor,
        ):
            with self.subTest(op=str(op)):
                graph = fx.Graph()
                x = _make_placeholder(graph, "x", torch.empty(128, 16, 64), batch_dim=0)
                # For .Tensor ops, pass a tensor; for .Scalar ops, pass a scalar.
                if "Tensor" in str(op):
                    other = _untagged_placeholder(
                        graph, "other", torch.empty(128, 16, 64)
                    )
                    args = (x, other)
                else:
                    args = (x, 1.0)
                out = graph.call_function(op, args=args)
                _set_val(out, torch.empty(128, 16, 64))
                graph.output(out)

                self._run_propagation_on(graph)

                self.assertTrue(
                    out.meta.get("is_batch_input"),
                    f"is_batch_input not propagated through {op}",
                )
                self.assertEqual(out.meta.get("batch_dim"), 0)

    def test_log_softmax_backward_propagates(self) -> None:
        """aten._log_softmax_backward_data preserves shape and batch_dim."""
        graph = fx.Graph()
        grad = _make_placeholder(graph, "grad", torch.empty(128, 16), batch_dim=0)
        output = _make_placeholder(graph, "out", torch.empty(128, 16), batch_dim=0)
        out = graph.call_function(
            torch.ops.aten._log_softmax_backward_data.default,
            args=(grad, output, 1, torch.float32),
        )
        _set_val(out, torch.empty(128, 16))
        graph.output(out)

        self._run_propagation_on(graph)

        self.assertTrue(out.meta.get("is_batch_input"))
        self.assertEqual(out.meta.get("batch_dim"), 0)

    def test_nll_loss_backward_propagates(self) -> None:
        """nll_loss_backward(grad_out, self, target, weight, ...) -> grad_self.

        Output shape == self.shape, so batch_dim is preserved from self.
        """
        graph = fx.Graph()
        # grad_output is a scalar (reduction='mean') -> untagged.
        grad_output = _untagged_placeholder(graph, "grad", torch.empty(()))
        # nll_loss_backward's `self` arg (predictions) has shape [N, C];
        # batched on axis 0. Use "preds" as the placeholder name to avoid
        # shadowing the implicit `self` in the generated GraphModule
        # `forward(self, ...)` signature.
        preds = _make_placeholder(graph, "preds", torch.empty(128, 1024), batch_dim=0)
        target = _make_placeholder(
            graph, "target", torch.empty(128, dtype=torch.int64), batch_dim=0
        )
        weight = _untagged_placeholder(graph, "weight", torch.empty(1024))
        total_weight = _untagged_placeholder(graph, "tw", torch.empty(()))
        out = graph.call_function(
            torch.ops.aten.nll_loss_backward.default,
            args=(grad_output, preds, target, weight, 1, -100, total_weight),
        )
        _set_val(out, torch.empty(128, 1024))
        graph.output(out)

        self._run_propagation_on(graph)

        self.assertTrue(out.meta.get("is_batch_input"))
        self.assertEqual(out.meta.get("batch_dim"), 0)
