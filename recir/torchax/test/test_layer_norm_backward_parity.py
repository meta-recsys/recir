#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Does torchax's layer-norm BACKWARD match eager's, given identical inputs?

The parity comparison localised its last standing disagreement to one op: in
hstu_compute_output's backward, the gradient w.r.t. ``attn`` -- the input that
crosses a layer norm -- came out 1.8% apart, while the gradient w.r.t. ``u``
from the same call, which only crosses an elementwise multiply, agreed to
0.11%. The inputs to that backward differ by 0.12%.

Two readings fit that, and they need different fixes:
  (a) the two legs compute layer_norm_backward differently, or
  (b) they compute it identically and the op is ill-conditioned, so it
      amplifies the 0.12% already present in its inputs by ~15x.

Feeding BOTH implementations byte-identical fp32 inputs separates them: any
disagreement here is (a), because there is nothing left for conditioning to
amplify. Runs on the JAX CPU backend, so it needs no TPU.

The forward is already covered (test_core_aten_ops: native_layer_norm); the
backward was not, which is why a difference in it could sit unnoticed.
"""

from typing import cast
from unittest import TestCase

import torch
from pyre_extensions import none_throws
from recir.torchax import tensor
from torch._decomp.decompositions import native_layer_norm_backward as ref_bwd


def _to_cpu(t: torch.Tensor) -> torch.Tensor:
    """Back to a plain torch.Tensor; inside the environment these are torchax's."""
    return cast(tensor.Tensor, t).torch()


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative L2 difference, the same measure the parity harness reports."""
    denom = a.to(torch.float64).norm().item()
    if denom == 0.0:
        return 0.0
    return ((a - b).to(torch.float64).norm() / denom).item()


class LayerNormBackwardParityTest(TestCase):
    # Shaped and scaled like the tensor in question: [tokens, 512] with an rms
    # near 1e-2. A unit-scale input would hide a conditioning effect that only
    # shows when the norm's denominator is small.
    ROWS = 4096
    DIM = 512
    SCALE = 1e-2

    def setUp(self) -> None:
        super().setUp()
        self.env: tensor.Environment = tensor.Environment()

    def _inputs(
        self, seed: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(self.ROWS, self.DIM, generator=g) * self.SCALE
        grad_out = torch.randn(self.ROWS, self.DIM, generator=g) * self.SCALE
        weight = torch.randn(self.DIM, generator=g)
        bias = torch.randn(self.DIM, generator=g)
        return x, grad_out, weight, bias

    def _backward(
        self,
        x: torch.Tensor,
        grad_out: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Through AUTOGRAD, not by calling the aten backward directly.

        native_layer_norm_backward is in torchax's decomposition list, so the
        compiled path never reaches its lowering -- it runs a decomposition
        into primitives instead. Calling the aten op straight would compare a
        code path the model does not use (and would hit that lowering's wrong
        signature: it declares 6 parameters against aten's 8, reading mean and
        rstd as weight and bias, and calls a jax.lax entry point that does not
        exist).
        """
        x = x.detach().requires_grad_(True)
        weight = weight.detach().requires_grad_(True)
        bias = bias.detach().requires_grad_(True)
        out = torch.nn.functional.layer_norm(
            x, (self.DIM,), weight=weight, bias=bias, eps=1e-6
        )
        (out * grad_out).sum().backward()
        return none_throws(x.grad), none_throws(weight.grad), none_throws(bias.grad)

    def test_the_input_gradient_matches(self) -> None:
        """The one the parity run flagged: d(loss)/d(x) through the norm."""
        args = self._inputs()
        eager = self._backward(*args)[0]
        moved = self.env.to_xla(args)
        with self.env:
            lowered = _to_cpu(self._backward(*moved)[0])
        rel = _rel_l2(eager, lowered)
        # fp32 accumulation over 512 elements leaves room around 1e-6; the
        # disagreement being explained is 1.8e-2, four orders up.
        self.assertLess(rel, 1e-5, f"relative L2 difference {rel:.3e}")

    def test_the_weight_and_bias_gradients_match(self) -> None:
        """These reduce over rows rather than within them, so they fail
        differently -- a reduction-order difference shows here first."""
        args = self._inputs()
        _, gw_e, gb_e = self._backward(*args)
        moved = self.env.to_xla(args)
        with self.env:
            _, gw_l, gb_l = self._backward(*moved)
        self.assertLess(_rel_l2(gw_e, _to_cpu(gw_l)), 1e-5)
        self.assertLess(_rel_l2(gb_e, _to_cpu(gb_l)), 1e-5)

    def test_a_unit_scale_input_also_matches(self) -> None:
        """Control. If only the small-scale case disagrees, the finding is
        about conditioning rather than about the implementation."""
        x, grad_out, weight, bias = self._inputs(seed=1)
        args = (x / self.SCALE, grad_out, weight, bias)
        eager = self._backward(*args)[0]
        moved = self.env.to_xla(args)
        with self.env:
            lowered = _to_cpu(self._backward(*moved)[0])
        rel = _rel_l2(eager, lowered)
        self.assertLess(rel, 1e-5, f"relative L2 difference {rel:.3e}")


class FusedKernelVsReferenceDecompositionTest(TestCase):
    """The two implementations the parity legs actually run.

    recir/ops/decompositions.py maps aten.native_layer_norm_backward.default to
    torch._decomp.decompositions.native_layer_norm_backward, so the compiled
    leg runs the REFERENCE DECOMPOSITION (mean/sub/mul/rsqrt, lowered op by op)
    while the eager leg runs the fused kernel. Same mathematics, two
    implementations -- which makes their disagreement a property of PyTorch,
    measurable here without JAX, torchax or a TPU.

    Scale matters: the tensor in question has an rms near 1e-2, and layer
    norm's backward subtracts two means, so its conditioning depends on how
    much of the input the normalisation removes.
    """

    ROWS = 8192
    DIM = 512

    def _compare(self, scale: float) -> dict[str, float]:
        g = torch.Generator().manual_seed(0)
        x = torch.randn(self.ROWS, self.DIM, generator=g) * scale
        grad_out = torch.randn(self.ROWS, self.DIM, generator=g) * scale
        w = torch.randn(self.DIM, generator=g)
        b = torch.randn(self.DIM, generator=g)
        _, mean, rstd = torch.ops.aten.native_layer_norm(x, [self.DIM], w, b, 1e-6)
        mask = [True, True, True]
        fused = torch.ops.aten.native_layer_norm_backward(
            grad_out, x, [self.DIM], mean, rstd, w, b, mask
        )
        decomp = ref_bwd(grad_out, x, [self.DIM], mean, rstd, w, b, mask)
        names = ("grad_input", "grad_weight", "grad_bias")
        return {
            n: _rel_l2(fused[i], none_throws(decomp[i])) for i, n in enumerate(names)
        }

    def test_they_agree_at_the_measured_scale(self) -> None:
        """rms 1e-2, matching the attn tensor. The disagreement being chased is
        1.8e-2; anything near that here would explain it outright."""
        got = self._compare(1e-2)
        self.assertLess(got["grad_input"], 1e-4, f"fused vs decomposition: {got}")

    def test_they_agree_at_unit_scale(self) -> None:
        """Control: if only the small-scale case disagrees, the effect is
        conditioning rather than a formula difference."""
        got = self._compare(1.0)
        self.assertLess(got["grad_input"], 1e-4, f"fused vs decomposition: {got}")
