# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import os
import unittest
from typing import Any, Optional
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from parameterized import parameterized
from recir.ops.pallas import rmsnorm as kernel
from recir.torchax.ops import jrmsnorm

_EPS = 1e-6


def _inputs(
    rows: int, dim: int, dtype: Any, weighted: bool, seed: int = 0
) -> tuple[Any, Optional[Any], Any]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((rows, dim)) * 3.0
    x[5:9] = 0.0
    weight = 1.0 + 0.5 * rng.standard_normal(dim) if weighted else None
    grad = rng.standard_normal((rows, dim))
    return (
        jnp.asarray(x, dtype),
        jnp.asarray(weight, dtype) if weight is not None else None,
        jnp.asarray(grad, dtype),
    )


def _assert_close(actual: Any, expected: Any) -> None:
    tol = 2e-2 if expected.dtype == jnp.bfloat16 else 2e-5
    np.testing.assert_allclose(
        np.asarray(actual, np.float32),
        np.asarray(expected, np.float32),
        rtol=tol,
        atol=tol,
    )


def _has_pallas_call(fn: Any, *args: Any) -> bool:
    return "pallas_call" in str(jax.make_jaxpr(fn)(*args))


class RMSNormKernelTest(unittest.TestCase):
    @parameterized.expand(
        itertools.product(
            (1024, 3000, 100),
            (128, 256, 512),
            (jnp.float32, jnp.bfloat16),
            (False, True),
            (False, True),
        )
    )
    def test_kernel_matches_xla(
        self, rows: int, dim: int, dtype: Any, weighted: bool, silu: bool
    ) -> None:
        x, weight, grad = _inputs(rows, dim, dtype, weighted)

        y_ref, rstd_ref = jrmsnorm.xla_rmsnorm_fwd(x, weight, _EPS, silu)
        dx_ref, dw_ref = jrmsnorm.xla_rmsnorm_backward(grad, x, weight, rstd_ref, silu)
        y, rstd = kernel.rmsnorm_fwd(x, weight, _EPS, silu, interpret=True)
        dx, dweight = kernel.rmsnorm_backward(
            grad, x, weight, rstd, silu, interpret=True
        )

        self.assertEqual(y.dtype, dtype)
        self.assertEqual(dx.dtype, dtype)
        self.assertEqual(rstd.shape, (rows,))
        _assert_close(y, y_ref)
        _assert_close(rstd, rstd_ref)
        _assert_close(dx, dx_ref)
        self.assertEqual(dweight is not None, weighted)
        if dweight is not None:
            self.assertEqual(dweight.dtype, jnp.float32)
            _assert_close(dweight.astype(dtype), dw_ref)

    def test_flag_off_does_not_use_the_kernel(self) -> None:
        x, weight, grad = _inputs(2048, 256, jnp.bfloat16, True)
        with mock.patch.dict(os.environ, {"RECIR_RMSNORM_KERNEL": "0"}):
            self.assertFalse(
                _has_pallas_call(
                    lambda x, w: jrmsnorm.rmsnorm_fwd(x, w, _EPS, True), x, weight
                )
            )

    @parameterized.expand([(False,), (True,)])
    def test_flag_on_matches_xla_across_devices(self, weighted: bool) -> None:
        self.assertEqual(len(jax.devices()), 2)
        x, weight, grad = _inputs(2 * 3 * 1000, 256, jnp.float32, weighted, seed=1)
        x, grad = x.reshape(2, 3000, 256), grad.reshape(2, 3000, 256)

        def fwd_bwd(x: Any, w: Optional[Any], g: Any) -> Any:
            y, rstd = jrmsnorm.rmsnorm_fwd(x, w, _EPS, True)
            return (y, rstd) + jrmsnorm.rmsnorm_backward(g, x, w, rstd, True)

        expected = fwd_bwd(x, weight, grad)
        with mock.patch.dict(os.environ, {"RECIR_RMSNORM_KERNEL": "1"}):
            self.assertTrue(_has_pallas_call(fwd_bwd, x, weight, grad))
            actual = jax.jit(fwd_bwd)(x, weight, grad)

        for a, e in zip(actual, expected):
            self.assertEqual((a.shape, a.dtype), (e.shape, e.dtype))
            _assert_close(a, e)

    def test_flag_on_falls_back_where_the_kernel_does_not_apply(self) -> None:
        cases = [
            _inputs(2048, 200, jnp.float32, True)[0],
            _inputs(2047, 256, jnp.float32, True)[0],
            jnp.zeros((0, 256), jnp.float32),
        ]
        with mock.patch.dict(os.environ, {"RECIR_RMSNORM_KERNEL": "1"}):
            for x in cases:
                self.assertFalse(
                    _has_pallas_call(
                        lambda x: jrmsnorm.rmsnorm_fwd(x, None, _EPS, False), x
                    )
                )

    def test_invalid_flag_raises(self) -> None:
        x = _inputs(1024, 128, jnp.float32, False)[0]
        with mock.patch.dict(os.environ, {"RECIR_RMSNORM_KERNEL": "yes"}):
            with self.assertRaises(ValueError):
                jrmsnorm.rmsnorm_fwd(x, None, _EPS, False)


if __name__ == "__main__":
    unittest.main()
