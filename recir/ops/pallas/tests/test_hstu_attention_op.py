# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.


import math
import unittest

import recir.ops.hstu_attention_op  # noqa: F401 -- registers torch.ops.recir.hstu_attention
import torch
from recir.ops.hstu_attention_op import _hstu_attention_ref


class HstuAttentionOpTest(unittest.TestCase):
    def _inputs(self, b=2, h=2, seq_len=12, d=8, seed=0):
        torch.manual_seed(seed)
        q = torch.randn(b, h, seq_len, d)
        k = torch.randn(b, h, seq_len, d)
        v = torch.randn(b, h, seq_len, d)
        return q, k, v

    def test_op_registered(self):
        self.assertTrue(hasattr(torch.ops.recir, "hstu_attention"))
        self.assertTrue(hasattr(torch.ops.recir, "hstu_attention_backward"))

    def test_forward_matches_reference(self):
        q, k, v = self._inputs()
        seq_len, d = q.shape[2], q.shape[3]
        lengths = torch.tensor([seq_len, 7], dtype=torch.int32)
        num_targets = torch.tensor([3, 2], dtype=torch.int32)
        alpha, inv_n = 1.0 / math.sqrt(d), 1.0 / seq_len
        out = torch.ops.recir.hstu_attention(
            q, k, v, lengths, num_targets, alpha, inv_n
        )
        ref = _hstu_attention_ref(q, k, v, lengths, num_targets, alpha, inv_n)
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
        # padded query rows (>=7) of row 1 are exactly zero.
        self.assertTrue(bool(torch.all(out[1, :, 7:, :] == 0.0)))

    def test_autograd_matches_reference(self):
        q, k, v = self._inputs(seed=1)
        seq_len, d = q.shape[2], q.shape[3]
        lengths = torch.tensor([seq_len, 9], dtype=torch.int32)
        num_targets = torch.tensor([2, 1], dtype=torch.int32)
        alpha, inv_n = 1.0 / math.sqrt(d), 1.0 / seq_len

        def run(fn, qi, ki, vi):
            qi, ki, vi = (t.clone().requires_grad_(True) for t in (qi, ki, vi))
            fn(qi, ki, vi, lengths, num_targets, alpha, inv_n).pow(2).sum().backward()
            return qi.grad, ki.grad, vi.grad

        g_op = run(torch.ops.recir.hstu_attention, q, k, v)
        g_ref = run(_hstu_attention_ref, q, k, v)
        for a, b in zip(g_op, g_ref):
            torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)

    def test_fake_shape(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            q = torch.randn(2, 4, 8, 16)
            k = torch.randn(2, 4, 8, 16)
            v = torch.randn(2, 4, 8, 32)
            lengths = torch.tensor([8, 8], dtype=torch.int32)
            num_targets = torch.tensor([0, 0], dtype=torch.int32)
            out = torch.ops.recir.hstu_attention(
                q, k, v, lengths, num_targets, 0.1, 0.1
            )
            self.assertEqual(tuple(out.shape), (2, 4, 8, 32))
