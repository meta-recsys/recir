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
