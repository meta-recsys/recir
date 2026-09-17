# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from __future__ import annotations

import math
import unittest

import fbgemm_gpu.sparse_ops  # noqa: F401 -- registers jagged tensor operators
import recir.ops.pallas_jagged_hstu_attention_op as packed_op
import torch
from functorch import make_fx
from generative_recommenders.ops.pytorch.pt_hstu_attention import pytorch_hstu_mha
from torch._subclasses.fake_tensor import FakeTensorMode


class PallasJaggedHSTUAttentionOpTest(unittest.TestCase):
    def _inputs(
        self,
        *,
        requires_grad: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        lengths = torch.tensor([7, 3, 5], dtype=torch.int64)
        offsets = torch.cat((torch.zeros(1, dtype=torch.int64), lengths.cumsum(0)))
        torch.manual_seed(0)
        shape = (int(offsets[-1]), 2, 8)
        q = torch.randn(shape, requires_grad=requires_grad)
        k = torch.randn(shape, requires_grad=requires_grad)
        v = torch.randn(shape, requires_grad=requires_grad)
        return q, k, v, offsets

    def _adapter(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        offsets: torch.Tensor,
        *,
        attn_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return packed_op._recir_pallas_jagged_hstu_mha(
            q=q,
            k=k,
            v=v,
            alpha=1.0 / math.sqrt(q.shape[-1]),
            N=8,
            seq_offsets=offsets,
            invalid_attn_mask_type="lower_triangular",
            dropout_pr=0.0,
            training=True,
            num_targets=torch.tensor([2, 1, 0], dtype=torch.int32),
            attn_bias=None,
            attn_scale=attn_scale,
        )

    def test_forward_preserves_runtime_scalar_scale(self) -> None:
        q, k, v, offsets = self._inputs()
        scale = torch.tensor(0.37)
        targets = torch.tensor([2, 1, 0], dtype=torch.int32)

        actual = self._adapter(q, k, v, offsets, attn_scale=scale)
        expected = pytorch_hstu_mha(
            max_seq_len=8,
            alpha=1.0 / math.sqrt(q.shape[-1]),
            q=q,
            k=k,
            v=v,
            seq_offsets=offsets,
            causal=True,
            num_targets=targets,
            attn_scale=scale,
        )

        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
        self.assertEqual(actual.shape, q.shape)

    def test_autograd_matches_reference(self) -> None:
        q, k, v, offsets = self._inputs()
        scale = torch.tensor(0.23)
        targets = torch.tensor([2, 1, 0], dtype=torch.int32)

        def gradients(fn):
            q_grad, k_grad, v_grad = (
                tensor.detach().clone().requires_grad_(True) for tensor in (q, k, v)
            )
            fn(q_grad, k_grad, v_grad).square().sum().backward()
            return q_grad.grad, k_grad.grad, v_grad.grad

        actual = gradients(
            lambda q_, k_, v_: self._adapter(q_, k_, v_, offsets, attn_scale=scale)
        )
        expected = gradients(
            lambda q_, k_, v_: pytorch_hstu_mha(
                max_seq_len=8,
                alpha=1.0 / math.sqrt(q.shape[-1]),
                q=q_,
                k=k_,
                v=v_,
                seq_offsets=offsets,
                causal=True,
                num_targets=targets,
                attn_scale=scale,
            )
        )

        for actual_grad, expected_grad in zip(actual, expected):
            torch.testing.assert_close(
                actual_grad,
                expected_grad,
                rtol=2e-3,
                atol=2e-3,
            )

    def test_fake_forward_shape(self) -> None:
        with FakeTensorMode():
            q = torch.randn(15, 4, 16)
            k = torch.randn(15, 4, 16)
            v = torch.randn(15, 4, 32)
            offsets = torch.tensor([0, 8, 15], dtype=torch.int64)
            targets = torch.tensor([0, 2], dtype=torch.int32)
            output = torch.ops.recir.pallas_jagged_hstu_attention_fwd(
                q,
                k,
                v,
                offsets,
                targets,
                v.new_tensor(0.1),
                0.25,
                8,
            )
            self.assertEqual(output.shape, (2, 4, 8, 32))
            self.assertEqual(output.dtype, v.dtype)

    def test_trace_keeps_forward_and_backward_opaque(self) -> None:
        q, k, v, offsets = self._inputs(requires_grad=True)

        def fn(
            q_: torch.Tensor,
            k_: torch.Tensor,
            v_: torch.Tensor,
            offsets_: torch.Tensor,
        ) -> tuple[torch.Tensor, ...]:
            output = self._adapter(
                q_,
                k_,
                v_,
                offsets_,
                attn_scale=v_.new_tensor(0.19),
            )
            return torch.autograd.grad(output.sum(), (q_, k_, v_))

        graph = make_fx(fn, tracing_mode="fake")(q, k, v, offsets)
        targets = [
            str(node.target) for node in graph.graph.nodes if node.op == "call_function"
        ]
        forward = [
            target
            for target in targets
            if target.endswith("pallas_jagged_hstu_attention_fwd.default")
        ]
        backward = [
            target
            for target in targets
            if target.endswith("pallas_jagged_hstu_attention_bwd.default")
        ]
        input_padding_indices = [
            index
            for index, target in enumerate(targets)
            if "jagged_to_padded_dense" in target
        ]
        output_packing_indices = [
            index for index, target in enumerate(targets) if "dense_to_jagged" in target
        ]
        self.assertEqual(len(forward), 1, targets)
        self.assertEqual(len(backward), 1, targets)
        self.assertEqual(len(output_packing_indices), 1, targets)
        forward_index = targets.index(forward[0])
        backward_index = targets.index(backward[0])
        self.assertFalse(
            [index for index in input_padding_indices if index < forward_index],
            targets,
        )
        self.assertTrue(
            all(
                output_packing_indices[0] < index < backward_index
                for index in input_padding_indices
            ),
            targets,
        )
        self.assertFalse([target for target in targets if "silu" in target], targets)

    def test_adapter_rejects_unsupported_scales(self) -> None:
        q, k, v, offsets = self._inputs()
        with self.assertRaisesRegex(ValueError, "scalar attn_scale"):
            self._adapter(q, k, v, offsets, attn_scale=torch.ones(2))
        with self.assertRaisesRegex(ValueError, "does not differentiate"):
            self._adapter(
                q,
                k,
                v,
                offsets,
                attn_scale=torch.tensor(0.1, requires_grad=True),
            )
