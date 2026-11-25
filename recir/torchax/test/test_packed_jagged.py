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
from hammer.v3.modules.mot.loss.packed_supervision import gather_padded_sequence
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
    _local_supervision_gather,
    _packed_bmm,
    _packed_bmm_add,
    _packed_bmm_add_backward,
    _packed_bmm_backward,
    _packed_position_add,
    _packed_position_add_backward,
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












    def test_packed_position_values_and_gradients(self) -> None:
        offsets = torch.tensor([0, 0, 3, 9])
        pi = torch.arange(12).reshape(3, 4) % 5
        ti = torch.arange(12).reshape(3, 4) % 7
        for dtype in (torch.float32, torch.bfloat16):
            values = torch.randn(13, 128, dtype=dtype, requires_grad=True)
            position = (
                torch.arange(5).float()[:, None].expand(5, 128) / 8 + 1 + 2**-8
            ).requires_grad_()
            time = (
                -torch.arange(7).float()[:, None].expand(7, 128) / 16 - 1
            ).requires_grad_()
            dense = torch.zeros(3, 4, 128, dtype=dtype)
            for row in range(3):
                start = int(offsets[row])
                length = min(int(offsets[row + 1]) - start, 4)
                dense[row, :length] = values[start : start + length]
            dense = dense + (time[ti] + position[pi]).to(dtype)
            want = torch.zeros_like(values)
            for row in range(3):
                start = int(offsets[row])
                length = min(int(offsets[row + 1]) - start, 4)
                want[start : start + length] = dense[row, :length]
            got = torch.ops.recir.packed_position_add(
                values, position, time, pi, ti, offsets, 4
            )
            grad = torch.randn_like(values)
            expected_grads = torch.autograd.grad(want, (values, position, time), grad)
            actual_grads = torch.autograd.grad(got, (values, position, time), grad)
            torch.testing.assert_close(got, want, rtol=0, atol=0)
            for actual, expected in zip(actual_grads, expected_grads):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            jdtype = jnp.bfloat16 if dtype == torch.bfloat16 else jnp.float32
            args = (
                jnp.asarray(values.detach().float().numpy()).astype(jdtype),
                jnp.asarray(position.detach().numpy()),
                jnp.asarray(time.detach().numpy()),
                jnp.asarray(pi.numpy(), jnp.int32),
                jnp.asarray(ti.numpy(), jnp.int32),
                jnp.asarray(offsets.numpy(), jnp.int32),
                4,
            )
            native = _packed_position_add(*args)
            native_grads = _packed_position_add_backward(
                jnp.asarray(grad.float().numpy()).astype(jdtype), *args
            )
            np.testing.assert_array_equal(
                np.asarray(native.astype(jnp.float32)), want.detach().float().numpy()
            )
            for actual, expected in zip(native_grads, expected_grads):
                np.testing.assert_allclose(
                    np.asarray(actual.astype(jnp.float32)),
                    expected.float().numpy(),
                    rtol=1e-6,
                    atol=1e-6,
                )
        zeros = (
            jnp.full((2, 128), -0.0, jnp.bfloat16),
            jnp.full((2, 128), -0.0, jnp.float32),
            jnp.full((2, 128), -0.0, jnp.float32),
            jnp.zeros((1, 2), jnp.int32),
            jnp.zeros((1, 2), jnp.int32),
            jnp.asarray([0, 1], jnp.int32),
            2,
        )
        zero_results = (
            _packed_position_add(*zeros),
            *_packed_position_add_backward(zeros[0], *zeros),
        )
        for actual in zero_results:
            self.assertFalse(np.any(np.signbit(np.asarray(actual.astype(jnp.float32)))))
        empty = (
            *zeros[:3],
            jnp.empty((0, 2), jnp.int32),
            jnp.empty((0, 2), jnp.int32),
            jnp.asarray([0], jnp.int32),
            2,
        )
        np.testing.assert_array_equal(
            _packed_position_add(*empty), jnp.zeros_like(zeros[0])
        )
        for actual in _packed_position_add_backward(zeros[0], *empty):
            self.assertEqual(int(jnp.count_nonzero(actual)), 0)


    def test_packed_supervision_gather(self) -> None:
        offsets = torch.tensor([0, 0, 3, 3, 9], dtype=torch.int64)
        indices = torch.tensor([0, 6, 7, 8, 9, 12, 18, 19, 23, 24])
        for dtype in (torch.int64, torch.float32, torch.bfloat16):
            source = torch.arange(13).to(dtype)
            if dtype == torch.int64:
                source = source * (1 << 40)
            else:
                source[0] = -0.0
                source[1] = float("nan")
                source[2] = float("inf")
            padded = torch.zeros(4, 7, dtype=dtype)
            for row in range(4):
                start, stop = offsets[row : row + 2]
                padded[row, : stop - start] = source[start:stop]
            expected = padded.flatten()[indices]
            actual = gather_padded_sequence(source, offsets, indices, 7)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
            torch.testing.assert_close(
                torch.signbit(actual[actual == 0]),
                torch.signbit(expected[expected == 0]),
            )
        torch.testing.assert_close(
            gather_padded_sequence(torch.empty(0), torch.zeros(5).long(), indices, 7),
            torch.zeros_like(indices, dtype=torch.float32),
        )
        gm = make_fx(gather_padded_sequence, tracing_mode="fake")(
            torch.arange(13), offsets, indices, 7
        )
        targets = {str(n.target) for n in gm.graph.nodes if n.op == "call_function"}
        self.assertIn("hammer.packed_supervision_gather.default", targets)

    def test_supervision_chip_padding(self) -> None:
        indices = jnp.asarray([0, 1, 2, 4, 5], jnp.int32)
        words = jnp.zeros(indices.shape, jnp.uint32)
        for owner, values, lengths in (
            (0, [10, 11, 0, 0], [2]),
            (1, [20, 0, 0, 0], [1]),
        ):
            words += _local_supervision_gather(
                jax.lax.bitcast_convert_type(
                    jnp.asarray(values, jnp.int32), jnp.uint32
                ),
                jnp.asarray(lengths, jnp.int32),
                indices,
                4,
                owner,
                1,
            )
        np.testing.assert_array_equal(
            jax.lax.bitcast_convert_type(words, jnp.int32), [10, 11, 0, 20, 0]
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
