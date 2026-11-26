# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import unittest
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import recir.torchax
import torch
from generative_recommenders.ops.pytorch.pt_layer_norm import pytorch_rms_norm
from parameterized import parameterized
from recir.ops.rmsnorm_op import rmsnorm
from recir.torchax.ops.jrmsnorm import rmsnorm_backward, rmsnorm_fwd
from torch.fx.experimental.proxy_tensor import make_fx

_EPS = 1e-6
_CASES = list(
    itertools.product(
        (False, True),
        (torch.float32, torch.bfloat16),
        (False, True),
    )
)


def _name(func, _, param) -> str:  # noqa
    weighted, dtype, silu = param.args
    return (
        f"{func.__name__}_{'weighted' if weighted else 'plain'}_"
        f"{str(dtype).split('.')[-1]}_{'silu' if silu else 'nosilu'}"
    )


def _inputs(
    rows: int, dim: int, weighted: bool, dtype: torch.dtype, seed: int = 0
) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    x = (torch.randn(rows, dim, generator=gen) * 3.0).to(dtype)
    weight = (
        (1.0 + 0.5 * torch.randn(dim, generator=gen)).to(dtype) if weighted else None
    )
    grad = torch.randn(rows, dim, generator=gen).to(dtype)
    return x, weight, grad


def _oracle(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    grad: torch.Tensor,
    silu: bool,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    x = x.clone().requires_grad_(True)
    if weight is not None:
        weight = weight.clone().requires_grad_(True)
    y = pytorch_rms_norm(x, [x.shape[-1]], weight, _EPS, silu)
    y.backward(grad)
    return y.detach(), x.grad, weight.grad if weight is not None else None


def _assert_close(
    actual: Optional[torch.Tensor], expected: Optional[torch.Tensor], dtype: torch.dtype
) -> None:
    if dtype == torch.bfloat16:
        torch.testing.assert_close(actual, expected, rtol=1.6e-2, atol=1e-2)
    else:
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def _jax(t: Optional[torch.Tensor]) -> Optional[jax.Array]:
    if t is None:
        return None
    if t.dtype == torch.bfloat16:
        return jnp.asarray(t.float().numpy(), dtype=jnp.bfloat16)
    return jnp.asarray(t.numpy())


def _torch(a: jax.Array) -> torch.Tensor:
    if a.dtype == jnp.bfloat16:
        return torch.from_numpy(np.asarray(a, dtype=np.float32)).bfloat16()
    return torch.from_numpy(np.asarray(a))


def _jax_fwd_bwd(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    grad: torch.Tensor,
    silu: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    fwd = jax.jit(rmsnorm_fwd, static_argnums=(2, 3))
    bwd = jax.jit(rmsnorm_backward, static_argnums=(4,))
    y, rstd = fwd(_jax(x), _jax(weight), _EPS, silu)
    dx, dweight = bwd(_jax(grad), _jax(x), _jax(weight), rstd, silu)
    return _torch(y), _torch(rstd), _torch(dx), _torch(dweight)


class RMSNormOpTest(unittest.TestCase):
    @parameterized.expand(_CASES, name_func=_name)
    def test_fallback_matches_oracle(
        self, weighted: bool, dtype: torch.dtype, silu: bool
    ) -> None:
        x, weight, grad = _inputs(37, 96, weighted, dtype)
        y_ref, dx_ref, dw_ref = _oracle(x, weight, grad, silu)

        x_op = x.clone().requires_grad_(True)
        w_op = weight.clone().requires_grad_(True) if weight is not None else None
        y = rmsnorm(x_op, w_op, _EPS, silu)
        y.backward(grad)

        self.assertEqual(y.dtype, dtype)
        _assert_close(y, y_ref, dtype)
        _assert_close(x_op.grad, dx_ref, dtype)
        if w_op is not None:
            _assert_close(w_op.grad, dw_ref, dtype)

    @parameterized.expand(_CASES, name_func=_name)
    def test_jax_lowering_matches_oracle(
        self, weighted: bool, dtype: torch.dtype, silu: bool
    ) -> None:
        x, weight, grad = _inputs(37, 256, weighted, dtype, seed=1)
        y_ref, dx_ref, dw_ref = _oracle(x, weight, grad, silu)

        y, rstd, dx, dweight = _jax_fwd_bwd(x, weight, grad, silu)

        self.assertEqual(y.dtype, dtype)
        self.assertEqual(rstd.dtype, torch.float32)
        self.assertEqual(dx.dtype, dtype)
        _assert_close(y, y_ref, dtype)
        _assert_close(dx, dx_ref, dtype)
        if weighted:
            self.assertEqual(dweight.dtype, dtype)
            _assert_close(dweight, dw_ref, dtype)
        else:
            self.assertEqual(tuple(dweight.shape), (0,))

    def test_jax_lowering_matches_fallback_with_leading_dims(self) -> None:
        x, weight, grad = _inputs(2 * 3 * 5, 200, True, torch.float32, seed=2)
        x, grad = x.view(2, 3, 5, 200), grad.view(2, 3, 5, 200)
        y_ref, rstd_ref = torch.ops.recir.rmsnorm_fwd(x, weight, _EPS, True)
        dx_ref, dw_ref = torch.ops.recir.rmsnorm_backward(
            grad, x, weight, rstd_ref, True
        )

        y, rstd, dx, dweight = _jax_fwd_bwd(x, weight, grad, True)

        self.assertEqual(tuple(rstd.shape), (2, 3, 5))
        torch.testing.assert_close(y, y_ref)
        torch.testing.assert_close(rstd, rstd_ref)
        torch.testing.assert_close(dx, dx_ref)
        torch.testing.assert_close(dweight, dw_ref)

    @parameterized.expand([(False,), (True,)])
    def test_empty_rows(self, weighted: bool) -> None:
        x, weight, grad = _inputs(0, 128, weighted, torch.bfloat16)

        y_ref, rstd_ref = torch.ops.recir.rmsnorm_fwd(x, weight, _EPS, True)
        dx_ref, dw_ref = torch.ops.recir.rmsnorm_backward(
            grad, x, weight, rstd_ref, True
        )
        y, rstd, dx, dweight = _jax_fwd_bwd(x, weight, grad, True)

        for out in (y_ref, dx_ref, y, dx):
            self.assertEqual(tuple(out.shape), (0, 128))
        for out in (rstd_ref, rstd):
            self.assertEqual(tuple(out.shape), (0,))
        expected_dw = torch.zeros(128 if weighted else 0, dtype=torch.bfloat16)
        torch.testing.assert_close(dw_ref, expected_dw)
        torch.testing.assert_close(dweight, expected_dw)

    @parameterized.expand(_CASES, name_func=_name)
    def test_padded_rows(self, weighted: bool, dtype: torch.dtype, silu: bool) -> None:
        x, weight, grad = _inputs(24, 128, weighted, dtype, seed=3)
        x[5:9] = 0
        x[20:] = 0
        y_ref, dx_ref, dw_ref = _oracle(x, weight, grad, silu)

        y, _, dx, dweight = _jax_fwd_bwd(x, weight, grad, silu)

        for out in (y, dx):
            self.assertTrue(torch.isfinite(out.float()).all())
        torch.testing.assert_close(y[20:], torch.zeros_like(y[20:]))
        _assert_close(y, y_ref, dtype)
        _assert_close(dx, dx_ref, dtype)
        if weighted:
            _assert_close(dweight, dw_ref, dtype)

    def test_fake_shapes_and_dtypes(self) -> None:
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            x = torch.randn(4, 6, 256, dtype=torch.bfloat16)
            weight = torch.randn(256, dtype=torch.float32)
            y, rstd = torch.ops.recir.rmsnorm_fwd(x, weight, _EPS, False)
            dx, dweight = torch.ops.recir.rmsnorm_backward(y, x, weight, rstd, False)
            _, plain_dweight = torch.ops.recir.rmsnorm_backward(y, x, None, rstd, True)

        self.assertEqual((y.shape, y.dtype), (x.shape, torch.bfloat16))
        self.assertEqual((rstd.shape, rstd.dtype), ((4, 6), torch.float32))
        self.assertEqual((dx.shape, dx.dtype), (x.shape, torch.bfloat16))
        self.assertEqual((dweight.shape, dweight.dtype), ((256,), torch.float32))
        self.assertEqual(tuple(plain_dweight.shape), (0,))

    def test_forward_and_backward_stay_opaque_under_make_fx(self) -> None:
        x, weight, grad = _inputs(8, 64, True, torch.float32)

        def fwd_bwd(x, weight, grad):  # noqa
            x = x.requires_grad_(True)
            weight = weight.requires_grad_(True)
            y = rmsnorm(x, weight, _EPS, True)
            return torch.autograd.grad(y, (x, weight), grad)

        graph = make_fx(fwd_bwd, tracing_mode="fake")(x, weight, grad)
        targets = [node.target for node in graph.graph.nodes]

        self.assertIn(torch.ops.recir.rmsnorm_fwd.default, targets)
        self.assertIn(torch.ops.recir.rmsnorm_backward.default, targets)

    @parameterized.expand([(False,), (True,)])
    def test_torchax_dispatch_uses_lowering(self, weighted: bool) -> None:
        x, weight, grad = _inputs(16, 128, weighted, torch.float32, seed=4)
        y_ref, rstd_ref = torch.ops.recir.rmsnorm_fwd(x, weight, _EPS, True)
        dx_ref, dw_ref = torch.ops.recir.rmsnorm_backward(
            grad, x, weight, rstd_ref, True
        )
        env = recir.torchax.default_env()
        xla_x, xla_w, xla_grad = env.to_xla((x, weight, grad))

        with env:
            y, rstd = torch.ops.recir.rmsnorm_fwd(xla_x, xla_w, _EPS, True)
            dx, dweight = torch.ops.recir.rmsnorm_backward(
                xla_grad, xla_x, xla_w, rstd, True
            )

        torch.testing.assert_close(y.torch(), y_ref)
        torch.testing.assert_close(rstd.torch(), rstd_ref)
        torch.testing.assert_close(dx.torch(), dx_ref)
        torch.testing.assert_close(dweight.torch(), dw_ref)


if __name__ == "__main__":
    unittest.main()
