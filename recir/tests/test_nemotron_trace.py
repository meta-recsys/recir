# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-ignore-all-errors

"""Tests for tracing Mamba2-based models (e.g., NemotronH) as a full graph.

Validates that mamba-ssm's custom ops can be traced through make_fx after
applying the FakeTensor registrations from recir.ops.fake.mamba.

Run locally:
  buck2 run @//mode/opt -c python.package_style=inplace \\
      -- fbcode//recir/tests:test_nemotron_trace

Run via RE (A100):
  buck2 test @//mode/opt -c fbcode.nvcc_arch=a100 -c fbcode.platform010_cuda_version=12.4 \\
      -c 'fbcode.enable_gpu_sections=true' -c 'hpc_comms.use_nccl=2.27' \\
      fbcode//recir/tests:test_nemotron_trace
"""

import unittest

import torch
import torch.nn as nn
from recir.ops.fake.mamba import patch_mamba_for_tracing
from torch.fx.experimental.proxy_tensor import make_fx


class SimpleMamba2Model(nn.Module):
    """Minimal model that exercises the Mamba2 fused training path."""

    def __init__(self, d_model=256, n_layer=2):
        super().__init__()
        from mamba_ssm.modules.mamba2 import Mamba2

        self.embedding = nn.Embedding(1000, d_model)
        self.layers = nn.ModuleList([Mamba2(d_model=d_model) for _ in range(n_layer)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1000, bias=False)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x) + x
        x = self.norm(x)
        return self.head(x)


class TestMamba2Trace(unittest.TestCase):
    """Verify make_fx can trace through Mamba2 after patching ops."""

    def test_mamba2_make_fx_forward(self):
        """Verify make_fx traces Mamba2 forward without errors."""
        patch_mamba_for_tracing()

        torch.manual_seed(0)
        model = SimpleMamba2Model(d_model=256, n_layer=2).eval()
        input_ids = torch.randint(0, 1000, (2, 32))

        def forward_fn(input_ids):
            return model(input_ids).sum()

        gm = make_fx(
            forward_fn,
            tracing_mode="fake",
            _allow_non_fake_inputs=True,
        )(input_ids)

        num_nodes = len(list(gm.graph.nodes))
        self.assertGreater(num_nodes, 0, "Traced graph should have nodes")

        # Verify the custom op appears in the graph (not traced through)
        custom_op_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and "recir_mamba" in str(n.target)
        ]
        self.assertGreater(
            len(custom_op_nodes),
            0,
            "Expected recir_mamba custom op nodes in traced graph; "
            "monkey-patching may have failed",
        )

    def test_mamba2_make_fx_numerics(self):
        """Verify traced Mamba2 graph produces same output as eager (real mode).

        Requires CUDA — mamba-ssm Triton kernels only run on GPU.
        """
        patch_mamba_for_tracing()

        torch.manual_seed(0)
        model = SimpleMamba2Model(d_model=256, n_layer=2).eval().cuda()
        input_ids = torch.randint(0, 1000, (2, 32)).cuda()

        def forward_fn(input_ids):
            return model(input_ids)

        # Eager forward
        with torch.no_grad():
            eager_out = forward_fn(input_ids)

        # Traced forward (real mode — executes actual ops on GPU)
        gm = make_fx(forward_fn, tracing_mode="real")(input_ids)

        with torch.no_grad():
            traced_out = gm(input_ids)

        torch.testing.assert_close(
            traced_out,
            eager_out,
            msg="Traced Mamba2 output should match eager",
        )

    def test_fused_moe_trace(self):
        """Verify make_fx traces fused MoE without errors."""
        from recir.ops.fused_moe import fuse_moe_layers

        torch.manual_seed(0)

        class FakeMLP(nn.Module):
            def __init__(self, hidden_size, intermediate_size):
                super().__init__()
                self.hidden_size = hidden_size
                self.intermediate_size = intermediate_size
                self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
                self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
                self.act_fn = lambda x: torch.relu(x) ** 2

            def forward(self, x):
                return self.down_proj(self.act_fn(self.up_proj(x)))

        class FakeConfig:
            n_routed_experts = 4
            num_experts_per_tok = 2
            hidden_size = 64
            moe_intermediate_size = 128
            moe_shared_expert_intermediate_size = 128
            mlp_hidden_act = "relu"
            mlp_bias = False
            routed_scaling_factor = 1.0
            n_group = 1
            topk_group = 1
            norm_topk_prob = True

        class FakeMOE(nn.Module):
            """Minimal NemotronHMOE-like module for testing."""

            def __init__(self):
                super().__init__()
                cfg = FakeConfig()
                self.config = cfg
                self.experts = nn.ModuleList([FakeMLP(64, 128) for _ in range(4)])

            def moe(self, hidden_states, topk_indices, topk_weights):
                final = torch.zeros_like(hidden_states)
                for i in range(len(self.experts)):
                    mask = (topk_indices == i).any(dim=-1)
                    if mask.any():
                        final[mask] += self.experts[i](hidden_states[mask])
                return final

            def forward(self, hidden_states):
                bs, dim = hidden_states.shape
                scores = torch.randn(bs, 4, device=hidden_states.device)
                topk_weights, topk_indices = scores.topk(2, dim=-1)
                topk_weights = torch.softmax(topk_weights, dim=-1)
                return self.moe(hidden_states, topk_indices, topk_weights)

        FakeMOE.__name__ = "NemotronHMOE"
        FakeMOE.__qualname__ = "NemotronHMOE"

        model = nn.Sequential()
        model.moe = FakeMOE()

        fused = fuse_moe_layers(model)
        self.assertEqual(fused, 1)

        x = torch.randn(8, 64)

        gm = make_fx(
            lambda x: model.moe(x).sum(),
            tracing_mode="fake",
            _allow_non_fake_inputs=True,
        )(x)

        num_nodes = len(list(gm.graph.nodes))
        self.assertGreater(num_nodes, 0)

        fused_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and "fused_moe" in str(n.target)
        ]
        self.assertGreater(len(fused_nodes), 0, "Expected fused_moe op in graph")

    def test_nemotronh_full_graph_trace(self):
        """End-to-end: verify a NemotronH-like hybrid model traces as full graph."""
        from recir.ops.fused_moe import fuse_moe_layers

        patch_mamba_for_tracing()

        torch.manual_seed(0)

        class FakeMLP(nn.Module):
            def __init__(self, d):
                super().__init__()
                self.hidden_size = d
                self.intermediate_size = d * 2
                self.up_proj = nn.Linear(d, d * 2, bias=False)
                self.down_proj = nn.Linear(d * 2, d, bias=False)
                self.act_fn = lambda x: torch.relu(x) ** 2

            def forward(self, x):
                return self.down_proj(self.act_fn(self.up_proj(x)))

        class FakeConfig:
            n_routed_experts = 4
            num_experts_per_tok = 2
            hidden_size = 256
            moe_intermediate_size = 512
            moe_shared_expert_intermediate_size = 512
            mlp_hidden_act = "relu"
            mlp_bias = False
            routed_scaling_factor = 1.0
            n_group = 1
            topk_group = 1
            norm_topk_prob = True

        class FakeMOE(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = FakeConfig()
                self.experts = nn.ModuleList([FakeMLP(256) for _ in range(4)])
                self.shared_experts = FakeMLP(256)
                self.gate_weight = nn.Parameter(torch.randn(4, 256))

            def moe(self, hidden_states, topk_indices, topk_weights):
                raise RuntimeError("Should be replaced by fuse_moe_layers")

            def forward(self, hidden_states):
                residuals = hidden_states
                orig_shape = hidden_states.shape
                hs = hidden_states.view(-1, 256)
                scores = (hs.float() @ self.gate_weight.float().t()).sigmoid()
                topk_weights, topk_indices = scores.topk(2, dim=-1)
                topk_weights = topk_weights / (
                    topk_weights.sum(-1, keepdim=True) + 1e-20
                )
                hs = self.moe(hs, topk_indices, topk_weights).view(*orig_shape)
                return hs + self.shared_experts(residuals)

        FakeMOE.__name__ = "NemotronHMOE"
        FakeMOE.__qualname__ = "NemotronHMOE"

        class HybridNemotronH(nn.Module):
            """Minimal NemotronH: Mamba2 + MoE + MLP layers."""

            def __init__(self):
                super().__init__()
                from mamba_ssm.modules.mamba2 import Mamba2

                self.embedding = nn.Embedding(1000, 256)
                self.mamba = Mamba2(d_model=256)
                self.moe = FakeMOE()
                self.mlp = FakeMLP(256)
                self.norm = nn.LayerNorm(256)
                self.head = nn.Linear(256, 1000, bias=False)

            def forward(self, input_ids):
                x = self.embedding(input_ids)
                x = self.mamba(x) + x
                x = self.moe(x) + x
                x = self.mlp(x) + x
                return self.head(self.norm(x))

        model = HybridNemotronH().eval()
        fused = fuse_moe_layers(model)
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
        has_mamba = any("mamba_split_conv1d" in name for name in op_names)
        has_moe = any("fused_moe" in name for name in op_names)
        self.assertTrue(has_mamba, "Expected mamba custom op in graph")
        self.assertTrue(has_moe, "Expected fused_moe custom op in graph")

    def test_mamba2_rmsnorm_trace(self):
        """Verify make_fx traces gated RMSNorm (rmsnorm_fn) without errors."""
        patch_mamba_for_tracing()

        from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated

        torch.manual_seed(0)

        class RMSNormModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(256, 256)
                self.norm = RMSNormGated(256, group_size=256)

            def forward(self, x):
                h = self.fc(x)
                gate = torch.sigmoid(h)
                return self.norm(h, z=gate).sum()

        model = RMSNormModel().eval()
        x = torch.randn(2, 32, 256)

        gm = make_fx(
            lambda x: model(x),
            tracing_mode="fake",
            _allow_non_fake_inputs=True,
        )(x)

        num_nodes = len(list(gm.graph.nodes))
        self.assertGreater(num_nodes, 0)


if __name__ == "__main__":
    unittest.main()
