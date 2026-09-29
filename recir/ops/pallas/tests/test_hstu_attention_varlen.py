# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Parity: recir::hstu_attention_varlen vs the production pytorch_hstu_mha.

Validates the full jagged path installed behind Hammer's PyTorch operation:
jagged pad -> dense HSTU kernel op -> unpad, matching generative_recommenders
pytorch_hstu_mha (causal + num_targets, SiLU, 1/max_seq_len scale) on CPU.
"""

import math
import unittest

import fbgemm_gpu.sparse_ops  # noqa: F401 -- registers the jagged<->dense ops
import recir.ops.hstu_attention_op  # noqa: F401 -- registers the recir ops
import torch
from generative_recommenders.ops.pytorch.pt_hstu_attention import pytorch_hstu_mha


class HstuAttentionVarlenParityTest(unittest.TestCase):
    def _jagged(self, lengths, h, d, seed=0):
        torch.manual_seed(seed)
        seq_offsets = torch.zeros(len(lengths) + 1, dtype=torch.int64)
        seq_offsets[1:] = torch.tensor(lengths, dtype=torch.int64).cumsum(0)
        total = int(seq_offsets[-1].item())
        q = torch.randn(total, h, d)
        k = torch.randn(total, h, d)
        v = torch.randn(total, h, d)
        return q, k, v, seq_offsets

    def _check(self, lengths, num_targets, h=4, d=8):
        max_seq_len = max(lengths)
        alpha = 1.0 / math.sqrt(d)
        q, k, v, seq_offsets = self._jagged(lengths, h, d)
        nt = torch.tensor(num_targets, dtype=torch.int32)

        ref = pytorch_hstu_mha(
            max_seq_len=max_seq_len,
            alpha=alpha,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            causal=True,
            num_targets=nt,
            attn_scale=None,  # -> SiLU / max_seq_len, matching our inv_n
        )
        out = torch.ops.recir.hstu_attention_varlen(
            q, k, v, seq_offsets, nt, alpha, max_seq_len
        )
        torch.testing.assert_close(out, ref, rtol=2e-4, atol=2e-4)

    def test_parity_causal_no_targets(self):
        self._check(lengths=[16, 11, 8], num_targets=[0, 0, 0])

    def test_parity_target_attention(self):
        self._check(lengths=[16, 12, 9], num_targets=[3, 2, 4])

    def test_parity_backward(self):
        # grads of the varlen op match grads of pytorch_hstu_mha.
        lengths, h, d = [14, 10], 4, 8
        max_seq_len = max(lengths)
        alpha = 1.0 / math.sqrt(d)
        q, k, v, seq_offsets = self._jagged(lengths, h, d, seed=1)
        nt = torch.tensor([2, 1], dtype=torch.int32)

        def grads(fn):
            qi, ki, vi = (t.clone().requires_grad_(True) for t in (q, k, v))
            fn(qi, ki, vi).pow(2).sum().backward()
            return qi.grad, ki.grad, vi.grad

        g_op = grads(
            lambda qi, ki, vi: torch.ops.recir.hstu_attention_varlen(
                qi, ki, vi, seq_offsets, nt, alpha, max_seq_len
            )
        )
        g_ref = grads(
            lambda qi, ki, vi: pytorch_hstu_mha(
                max_seq_len=max_seq_len,
                alpha=alpha,
                q=qi,
                k=ki,
                v=vi,
                seq_offsets=seq_offsets,
                causal=True,
                num_targets=nt,
                attn_scale=None,
            )
        )
        for a, b in zip(g_op, g_ref):
            torch.testing.assert_close(a, b, rtol=2e-3, atol=2e-3)

    def test_preprocess_and_attention_swap_routes_to_our_op(self):
        # Guard the actual RankFM training seam after installing the RecIR swap.
        from hammer.utils import HammerKernel
        from hammer.v2.modules.configs import (
            InvalidAttnMask,
            STUSelfAttnConfig,
            STUSelfAttnPreprocConfig,
        )
        from hammer.v2.ops.hstu_preprocess_and_attention import (
            hstu_preprocess_and_attention_without_bias,
        )
        from recir.ops.hstu_attention_op import install_hstu_attention_swap
        from torch.fx.experimental.proxy_tensor import make_fx

        install_hstu_attention_swap()
        heads, dqk, dv, e = 4, 128, 128, 128
        lengths = [16, 12]
        max_seq_len = max(lengths)
        batch = len(lengths)

        seq_offsets = torch.zeros(batch + 1, dtype=torch.int64)
        seq_offsets[1:] = torch.tensor(lengths, dtype=torch.int64).cumsum(0)
        total = int(seq_offsets[-1].item())

        torch.manual_seed(0)
        x = torch.randn(total, e)
        norm_weight = torch.randn(e)
        norm_bias = torch.randn(e)
        uvqk_cols = (2 * dv + 2 * dqk) * heads
        uvqk_weight = torch.randn(e, uvqk_cols)
        uvqk_bias = torch.randn(uvqk_cols)
        nt = torch.tensor([3, 2], dtype=torch.int32)

        attn_config = STUSelfAttnConfig(
            num_heads=heads,
            attention_dim=dqk,
            linear_hidden_dim=dv,
            alpha=1.0 / math.sqrt(dqk),
            invalid_attn_mask_type=InvalidAttnMask.LOWER_TRIANGULAR,
            contextual_seq_len=0,
            max_attn_len=0,
            global_attn_window=0,
            num_softmax_heads=0,
            use_cuda_attention=False,
        )
        preproc_config = STUSelfAttnPreprocConfig()

        def fn(x, uvqk_weight, uvqk_bias):
            return hstu_preprocess_and_attention_without_bias(
                x=x,
                norm_weight=norm_weight,
                norm_bias=norm_bias,
                norm_eps=1e-6,
                uvqk_weight=uvqk_weight,
                uvqk_bias=uvqk_bias,
                max_seq_len=max_seq_len,
                seq_offsets=seq_offsets,
                num_targets=nt,
                attn_config=attn_config,
                preproc_config=preproc_config,
                rotary_weights=None,
                attn_scale=None,
                silu_u=True,
                kernel=HammerKernel.PYTORCH,
                is_inference=False,
                skip_u=False,
            )

        gm = make_fx(fn)(x, uvqk_weight, uvqk_bias)
        targets = {str(n.target) for n in gm.graph.nodes if n.op == "call_function"}
        self.assertTrue(
            any("hstu_attention" in t for t in targets),
            f"Operation swap did not route to recir.hstu_attention; targets={sorted(targets)}",
        )
        self.assertFalse(
            any("bmm" in t for t in targets),
            f"Operation swap emitted a materialized attention bmm; targets={sorted(targets)}",
        )
