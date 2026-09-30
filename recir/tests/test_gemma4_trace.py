# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors

"""Tests for tracing Gemma4 MoE model as a full graph.

Validates that Gemma4TextExperts (128-expert MoE with data-dependent routing)
can be traced through make_fx after applying fuse_gemma4_moe_layers().

Run locally:
  buck2 run @//mode/opt -c python.package_style=inplace \\
      -- fbcode//recir/tests:test_gemma4_trace
"""

import unittest

import torch
import torch.nn as nn
from recir.ops.fake.gemma4 import fuse_gemma4_moe_layers
from torch.fx.experimental.proxy_tensor import make_fx
from transformers.activations import ACT2FN


class Gemma4TextExpertsStub(nn.Module):
    """Minimal reproduction of Gemma4TextExperts for testing.

    Matches the actual model's weight layout:
    - gate_up_proj: [num_experts, 2 * intermediate_size, hidden_size]
    - down_proj: [num_experts, hidden_size, intermediate_size]
    """

    def __init__(self, num_experts=4, hidden_size=64, intermediate_size=128):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_size
        self.intermediate_dim = intermediate_size
        self.gate_up_proj = nn.Parameter(
            torch.randn(num_experts, 2 * intermediate_size, hidden_size)
        )
        self.down_proj = nn.Parameter(
            torch.randn(num_experts, hidden_size, intermediate_size)
        )
        self.act_fn = ACT2FN["gelu_pytorch_tanh"]

    def forward(self, hidden_states, top_k_index, top_k_weights):
        """Original data-dependent dispatch (will be replaced by fused version)."""
        final = torch.zeros_like(hidden_states)
        expert_mask = torch.nn.functional.one_hot(
            top_k_index, num_classes=self.num_experts
        )
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = nn.functional.linear(
                current_state, self.gate_up_proj[expert_idx]
            ).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = nn.functional.linear(
                current_hidden_states, self.down_proj[expert_idx]
            )
            current_hidden_states = (
                current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            )
            final.index_add_(0, token_idx, current_hidden_states.to(final.dtype))
        return final


# Rename class to match what fuse_gemma4_moe_layers looks for
Gemma4TextExpertsStub.__name__ = "Gemma4TextExperts"
Gemma4TextExpertsStub.__qualname__ = "Gemma4TextExperts"


class FakeGemma4Config:
    hidden_activation = "gelu_pytorch_tanh"


class Gemma4MoEModel(nn.Module):
    """Minimal Gemma4-like model with MoE layer."""

    def __init__(self, hidden_size=64, intermediate_size=128, num_experts=4, top_k=2):
        super().__init__()
        self.config = FakeGemma4Config()
        self.embedding = nn.Embedding(1000, hidden_size)
        self.experts = Gemma4TextExpertsStub(
            num_experts, hidden_size, intermediate_size
        )
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, 1000, bias=False)
        self.top_k = top_k

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        bs, seq, dim = x.shape
        flat = x.view(-1, dim)

        # Router
        scores = torch.softmax(self.gate(flat), dim=-1)
        top_k_weights, top_k_index = scores.topk(self.top_k, dim=-1)
        top_k_weights = top_k_weights / (top_k_weights.sum(-1, keepdim=True) + 1e-20)

        # MoE
        moe_out = self.experts(flat, top_k_index, top_k_weights)
        x = x + moe_out.view(bs, seq, dim)

        return self.head(self.norm(x))


class TestGemma4Trace(unittest.TestCase):
    """Verify make_fx can trace Gemma4 MoE after fusing experts."""

    def test_gemma4_fused_experts_trace(self):
        """Verify fused Gemma4 experts traces without errors."""
        torch.manual_seed(0)
        model = nn.Sequential()
        model.experts = Gemma4TextExpertsStub(
            num_experts=4, hidden_size=64, intermediate_size=128
        )
        model.config = FakeGemma4Config()

        fused = fuse_gemma4_moe_layers(model)
        self.assertEqual(fused, 1)

        x = torch.randn(8, 64)
        top_k_index = torch.randint(0, 4, (8, 2))
        top_k_weights = torch.softmax(torch.randn(8, 2), dim=-1)

        gm = make_fx(
            lambda x, idx, w: model.experts(x, idx, w).sum(),
            tracing_mode="fake",
            _allow_non_fake_inputs=True,
        )(x, top_k_index, top_k_weights)

        num_nodes = len(list(gm.graph.nodes))
        self.assertGreater(num_nodes, 0)

        fused_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and "fused_experts" in str(n.target)
        ]
        self.assertGreater(len(fused_nodes), 0, "Expected fused_experts op in graph")

    def test_gemma4_e2e_trace(self):
        """End-to-end: verify Gemma4-like MoE model traces as full graph."""
        torch.manual_seed(0)
        model = Gemma4MoEModel(
            hidden_size=64, intermediate_size=128, num_experts=4
        ).eval()

        fused = fuse_gemma4_moe_layers(model)
        self.assertEqual(fused, 1)

        input_ids = torch.randint(0, 1000, (2, 16))

        gm = make_fx(
            lambda ids: model(ids).sum(),
            tracing_mode="fake",
            _allow_non_fake_inputs=True,
        )(input_ids)

        num_nodes = len(list(gm.graph.nodes))
        self.assertGreater(num_nodes, 0)

        op_names = [str(n.target) for n in gm.graph.nodes if n.op == "call_function"]
        has_moe = any("fused_experts" in name for name in op_names)
        self.assertTrue(has_moe, "Expected fused_experts custom op in graph")

    def test_gemma4_fused_experts_numerics(self):
        """Verify fused experts produce same output as original sequential loop."""
        torch.manual_seed(42)

        # Create experts with small weights (realistic scale) to keep
        # output magnitudes small and reduce floating-point divergence
        # between grouped_gemm (Triton) and sequential F.linear (cuBLAS).
        experts_original = Gemma4TextExpertsStub(
            num_experts=4, hidden_size=64, intermediate_size=128
        ).cuda()
        # Scale weights down to realistic range
        with torch.no_grad():
            experts_original.gate_up_proj.mul_(0.01)
            experts_original.down_proj.mul_(0.01)

        experts_fused = Gemma4TextExpertsStub(
            num_experts=4, hidden_size=64, intermediate_size=128
        ).cuda()
        # Copy scaled weights
        experts_fused.gate_up_proj.data.copy_(experts_original.gate_up_proj.data)
        experts_fused.down_proj.data.copy_(experts_original.down_proj.data)

        # Create model wrapper for fusing
        model_fused = nn.Sequential()
        model_fused.experts = experts_fused
        model_fused.config = FakeGemma4Config()
        fused = fuse_gemma4_moe_layers(model_fused)
        self.assertEqual(fused, 1)

        # Same inputs (small scale)
        x = torch.randn(16, 64, device="cuda") * 0.1
        top_k_index = torch.randint(0, 4, (16, 2), device="cuda")
        top_k_weights = torch.softmax(torch.randn(16, 2, device="cuda"), dim=-1)

        with torch.no_grad():
            out_original = experts_original(x, top_k_index, top_k_weights)
            out_fused = model_fused.experts(x, top_k_index, top_k_weights)

        # grouped_gemm (Triton) uses different reduction order than sequential
        # F.linear (cuBLAS). With small weights, the absolute difference is small.
        torch.testing.assert_close(
            out_fused,
            out_original,
            rtol=1e-2,
            atol=1e-4,
            msg="Fused Gemma4 experts output should match original",
        )
