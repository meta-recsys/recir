# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Packed projection values, VJPs, truncation and static-capacity padding."""

import operator
import unittest
from functools import partial
from itertools import product
from typing import Any, Callable
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import torch
from recir.ops.packed_jagged_ops import (
    jagged_dense_bmm,
    jagged_dense_bmm_add,
    jagged_dense_broadcast_add,
)
from recir.torchax.ops.jagged_ops import (
    _jax_dense_to_jagged_gather_impl,
    _jax_dense_to_jagged_scatter_impl,
)
from recir.torchax.ops.packed_jagged_ops import (
    _packed_bmm,
    _packed_bmm_add,
    _packed_bmm_add_backward,
    _packed_bmm_backward,
)
from torch.fx.experimental.proxy_tensor import make_fx


def _reference(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor,
    offsets: torch.Tensor,
    max_len: int,
) -> torch.Tensor:
    padded = x.new_zeros((w.shape[0], max_len, x.shape[1]), dtype=torch.float32)
    for b in range(w.shape[0]):
        start = int(offsets[b].item())
        length = min(int((offsets[b + 1] - offsets[b]).item()), max_len)
        padded[b, :length] = x[start : start + length].float()
    projected = torch.bmm(padded, w.float()) + bias.float().unsqueeze(1)
    out = x.new_zeros((x.shape[0], w.shape[2]), dtype=torch.float32)
    for b in range(w.shape[0]):
        start = int(offsets[b].item())
        length = min(int((offsets[b + 1] - offsets[b]).item()), max_len)
        out[start : start + length] = projected[b, :length]
    return out.to(x.dtype)


class PackedJaggedTest(unittest.TestCase):
    def test_bf16_matrix_gradients_with_unrounded_f32_grad(self) -> None:
        offsets = torch.tensor([0, 0, 3, 132])
        values = torch.ones(141, 128, dtype=torch.bfloat16)
        weights = torch.ones(3, 128, 128, dtype=torch.bfloat16)
        bias = torch.zeros(3, 128, dtype=torch.float32)
        grad = torch.full((141, 128), 1 + 2**-8, dtype=torch.float32)
        padded = torch.zeros(3, 180, 128, dtype=torch.float32)
        for row in range(3):
            start, stop = offsets[row : row + 2]
            padded[row, : stop - start] = grad[start:stop]
        rounded = padded.to(torch.bfloat16).float()
        expected_dx = torch.zeros_like(values)
        dense_dx = torch.bmm(rounded, weights.float().transpose(1, 2))
        dense_values = torch.zeros_like(padded)
        for row in range(3):
            start, stop = offsets[row : row + 2]
            expected_dx[start:stop] = dense_dx[row, : stop - start]
            dense_values[row, : stop - start] = values[start:stop].float()
        expected_dw = torch.bmm(dense_values.transpose(1, 2), rounded).bfloat16()
        unrounded_dw = torch.bmm(dense_values.transpose(1, 2), padded).bfloat16()
        self.assertFalse(torch.equal(expected_dw, unrounded_dw))
        expected_db = padded.sum(1)
        cpu = torch.ops.recir.packed_jagged_bmm_add_backward(
            grad, values, weights, bias, offsets, 180
        )
        native = _packed_bmm_add_backward(
            jnp.asarray(grad.numpy()),
            jnp.asarray(values.float().numpy()).astype(jnp.bfloat16),
            jnp.asarray(weights.float().numpy()).astype(jnp.bfloat16),
            jnp.asarray(bias.numpy()),
            jnp.asarray(offsets.numpy(), jnp.int32),
            180,
            interpret=True,
        )
        for actual, expected in zip(cpu, (expected_dx, expected_dw, expected_db)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for actual, expected in zip(native, (expected_dx, expected_dw, expected_db)):
            np.testing.assert_allclose(
                np.asarray(actual.astype(jnp.float32)),
                expected.float().numpy(),
                rtol=1e-6,
                atol=1e-6,
            )

    def test_pallas_group_boundaries_and_padding(self) -> None:
        rng = np.random.default_rng(19)
        # More than one tile, groups crossing tile boundaries, duplicate offsets,
        # truncated sequences and unused physical capacity.
        offsets = np.array([0, 0, 3, 3, 132, 141, 141], np.int32)
        for k, n, dtype in ((8, 6, torch.float32), (224, 512, torch.bfloat16)):
            with self.subTest(k=k, n=n):
                x = rng.normal(size=(173, k)).astype(np.float32)
                w = rng.normal(size=(6, k, n)).astype(np.float32)
                grad = rng.normal(size=(173, n)).astype(np.float32)
                tx = torch.tensor(x, dtype=dtype, requires_grad=True)
                tw = torch.tensor(w, dtype=dtype, requires_grad=True)
                tg = torch.tensor(grad, dtype=dtype)
                want = jagged_dense_bmm(128, torch.tensor(offsets), tx, tw)
                dx, dw = torch.autograd.grad(want, (tx, tw), tg)
                input_dtype = jnp.bfloat16 if dtype == torch.bfloat16 else jnp.float32
                jx = jnp.asarray(tx.detach().float().numpy()).astype(input_dtype)
                jw = jnp.asarray(tw.detach().float().numpy()).astype(input_dtype)
                jg = jnp.asarray(tg.float().numpy())
                got = _packed_bmm(jx, jw, jnp.asarray(offsets), 128, interpret=True)
                bias = rng.normal(size=(6, n)).astype(np.float32)
                jb = jnp.asarray(bias)
                fused = _packed_bmm_add(
                    jx, jw, jb, jnp.asarray(offsets), 128, interpret=True
                )
                expected_fused = _reference(
                    tx, tw, torch.tensor(bias), torch.tensor(offsets), 128
                )
                gx, gw = _packed_bmm_backward(
                    jg,
                    jx,
                    jw,
                    jnp.asarray(offsets),
                    128,
                    interpret=True,
                )
                fx, fw, fb = _packed_bmm_add_backward(
                    jg, jx, jw, jb, jnp.asarray(offsets), 128, interpret=True
                )
                np.testing.assert_array_equal(fx, gx)
                np.testing.assert_array_equal(fw, gw)
                expected_db = np.zeros((6, n), np.float32)
                for group in range(6):
                    start, stop = offsets[group : group + 2]
                    expected_db[group] = np.asarray(jg)[
                        start : min(stop, start + 128)
                    ].sum(0)
                np.testing.assert_allclose(fb, expected_db, rtol=1e-5, atol=5e-6)
                if dtype == torch.bfloat16:
                    for actual, expected in (
                        (got, want),
                        (gx, dx),
                        (gw, dw),
                        (fused, expected_fused),
                    ):
                        materialized = torch.tensor(
                            np.asarray(actual.astype(jnp.bfloat16), dtype=np.float32)
                        ).to(dtype)
                        torch.testing.assert_close(
                            materialized, expected, rtol=2**-6, atol=1e-5
                        )
                else:
                    np.testing.assert_allclose(
                        np.asarray(fused),
                        expected_fused.detach().numpy(),
                        rtol=1e-5,
                        atol=2e-6,
                    )
                    np.testing.assert_allclose(
                        np.asarray(got), want.detach().numpy(), rtol=1e-5, atol=2e-6
                    )
                    np.testing.assert_allclose(
                        np.asarray(gx), dx.numpy(), rtol=1e-5, atol=2e-6
                    )
                    np.testing.assert_allclose(
                        np.asarray(gw), dw.numpy(), rtol=1e-5, atol=5e-6
                    )

    def test_values_and_vjp(self) -> None:
        offsets = torch.tensor([0, 0, 3, 3, 12, 12], dtype=torch.int64)
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                torch.manual_seed(17)
                x = torch.randn(17, 8, dtype=dtype, requires_grad=True)
                w = torch.randn(5, 8, 6, dtype=dtype, requires_grad=True)
                bias = torch.randn(5, 6, dtype=dtype, requires_grad=True)
                grad = torch.randn(17, 6, dtype=dtype)
                want = _reference(x, w, bias, offsets, 6)
                got = jagged_dense_bmm_add(6, offsets, x, w, bias)
                torch.testing.assert_close(got, want)
                expected_grads = torch.autograd.grad(want, (x, w, bias), grad)
                actual_grads = torch.autograd.grad(got, (x, w, bias), grad)
                for actual, expected in zip(actual_grads, expected_grads):
                    torch.testing.assert_close(actual, expected)
                self.assertEqual(torch.count_nonzero(got[9:]).item(), 0)
                self.assertEqual(torch.count_nonzero(actual_grads[0][9:]).item(), 0)

    def test_separate_add_and_elementwise_bias(self) -> None:
        offsets = torch.tensor([0, 2, 5])
        x = torch.randn(9, 8, requires_grad=True)
        w = torch.randn(2, 8, 6, requires_grad=True)
        bias = torch.randn(2, 6, requires_grad=True)
        got = jagged_dense_broadcast_add(
            4, offsets, jagged_dense_bmm(4, offsets, x, w), bias
        )
        torch.testing.assert_close(got, _reference(x, w, bias, offsets, 4))
        eb = torch.randn(9, 6)
        torch.testing.assert_close(
            jagged_dense_bmm_add(4, offsets, x, w, eb, elementwise=True),
            jagged_dense_bmm(4, offsets, x, w) + eb,
        )

    def test_empty_capacity(self) -> None:
        x = torch.empty(0, 8, requires_grad=True)
        w = torch.randn(3, 8, 6, requires_grad=True)
        offsets = torch.zeros(4, dtype=torch.int32)
        y = jagged_dense_bmm(4, offsets, x, w)
        dx, dw = torch.autograd.grad(y.sum(), (x, w))
        self.assertEqual(tuple(dx.shape), (0, 8))
        self.assertEqual(torch.count_nonzero(dw).item(), 0)

    def test_trace_keeps_packed_forward_and_backward(self) -> None:
        def train(
            x: torch.Tensor, w: torch.Tensor, offsets: torch.Tensor
        ) -> torch.Tensor:
            y = jagged_dense_bmm(4, offsets, x, w)
            y.sum().backward()
            return y

        gm = make_fx(train, tracing_mode="fake")(
            torch.randn(9, 8, requires_grad=True),
            torch.randn(2, 8, 6, requires_grad=True),
            torch.tensor([0, 2, 5]),
        )
        targets = {str(n.target) for n in gm.graph.nodes if n.op == "call_function"}
        self.assertIn("recir.packed_jagged_bmm.default", targets)
        self.assertIn("recir.packed_jagged_bmm_backward.default", targets)
        self.assertFalse(
            any("dense_to_jagged" in t or "jagged_to_padded" in t for t in targets)
        )

        def train_with_bias(
            x: torch.Tensor,
            w: torch.Tensor,
            bias: torch.Tensor,
            offsets: torch.Tensor,
        ) -> torch.Tensor:
            y = jagged_dense_bmm_add(4, offsets, x, w, bias)
            y.sum().backward()
            return y

        fused = make_fx(train_with_bias, tracing_mode="fake")(
            torch.randn(9, 8, requires_grad=True),
            torch.randn(2, 8, 6, requires_grad=True),
            torch.randn(2, 6, requires_grad=True),
            torch.tensor([0, 2, 5]),
        )
        targets = {str(n.target) for n in fused.graph.nodes if n.op == "call_function"}
        self.assertIn("recir.packed_jagged_bmm_add.default", targets)
        self.assertIn("recir.packed_jagged_bmm_add_backward.default", targets)
        self.assertFalse(
            any("dense_to_jagged" in t or "jagged_to_padded" in t for t in targets)
        )

    def test_inverse_gather_values_and_adjoint(self) -> None:
        offsets = jnp.asarray([0, 0, 3, 3, 132, 141, 141], jnp.int32)
        rng = np.random.default_rng(23)
        with (
            jax.enable_x64(),
            patch.dict("os.environ", {"RECIR_PACKED_JAGGED_INT64": "1"}),
        ):
            for dtype in (jnp.float32, jnp.bfloat16, jnp.int64):
                for width in (None, 3):
                    shape = (6, 129) if width is None else (6, 129, width)
                    values = rng.normal(size=shape).astype(np.float32)
                    if dtype == jnp.int64:
                        data = jnp.asarray(values.astype(np.int64)) * (1 << 40)
                    else:
                        values[1, 0] = -0.0
                        values[3, 0] = np.nan
                        values[3, 1] = np.inf
                        values[3, 2] = -np.inf
                        data = jnp.asarray(values).astype(dtype)
                    for capacity in (173, 3, 0):
                        with self.subTest(dtype=dtype, width=width, capacity=capacity):
                            expected = _jax_dense_to_jagged_scatter_impl(
                                data, [offsets], capacity
                            )
                            actual = _jax_dense_to_jagged_gather_impl(
                                data, [offsets], capacity
                            )
                            host_type = np.int64 if dtype == jnp.int64 else np.float32
                            want = np.asarray(expected, dtype=host_type)
                            got = np.asarray(actual, dtype=host_type)
                            np.testing.assert_array_equal(got, want)
                            np.testing.assert_array_equal(
                                np.signbit(got[got == 0]), np.signbit(want[want == 0])
                            )
                            if dtype == jnp.float32:

                                def loss(
                                    x: jax.Array, gather: bool, length: int = capacity
                                ) -> jax.Array:
                                    fn = (
                                        _jax_dense_to_jagged_gather_impl
                                        if gather
                                        else _jax_dense_to_jagged_scatter_impl
                                    )
                                    return fn(x, [offsets], length).sum()

                                expected_grad = (
                                    jnp.zeros_like(data)
                                    if capacity == 0
                                    else jax.grad(loss)(data, False)
                                )
                                np.testing.assert_array_equal(
                                    jax.grad(loss)(data, True),
                                    expected_grad,
                                )


if __name__ == "__main__":
    unittest.main()
