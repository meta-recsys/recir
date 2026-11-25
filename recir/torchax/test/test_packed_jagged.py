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
    _packed_native_pair_dma_read,
    _packed_native_pair_prefix_dma_read,
    _packed_native_pair_prefix_store,
    _packed_position_add,
    _packed_position_add_backward,
    _packed_weight_grad_aligned,
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








    def test_native_pair_projection_boundaries(self) -> None:
        for raw_offsets in ([0, 0, 3, 3, 12, 14], [0, 0, 3, 3, 12], [0, 3]):
            offsets = torch.tensor(raw_offsets)
            groups = len(raw_offsets) - 1
            for dtype in (torch.float32, torch.bfloat16):
                values = (torch.arange(23 * 128).reshape(23, 128) / 256).to(dtype)
                values[raw_offsets[-1] :] = float("nan")
                if groups > 1:
                    values[7:12] = float("nan")
                weights = (
                    (
                        torch.arange(groups * 128 * 128).reshape(groups, 128, 128) % 17
                        - 8
                    )
                    / 8
                ).to(dtype)
                bias = torch.arange(groups * 128).reshape(groups, 128) / 128 + 2**-8
                jdtype = jnp.bfloat16 if dtype == torch.bfloat16 else jnp.float32
                args = (
                    jnp.asarray(values.float().numpy()).astype(jdtype),
                    jnp.asarray(weights.float().numpy()).astype(jdtype),
                    jnp.asarray(offsets.numpy(), jnp.int32),
                )
                for max_len in (0, 4):
                    expected = _reference(
                        values.float(), weights.float(), bias, offsets, max_len
                    )
                    expected_plain = _reference(
                        values.float(),
                        weights.float(),
                        torch.zeros_like(bias),
                        offsets,
                        max_len,
                    )
                    for implementation, store_impl in product(
                        (
                            "native_pair",
                            "native_pair_slice",
                            "native_pair_dma",
                            "native_pair_prefix_dma",
                        ),
                        ("full", "prefix_dma"),
                    ):
                        interpret = store_impl == "prefix_dma" or implementation in (
                            "native_pair_dma",
                            "native_pair_prefix_dma",
                        )
                        with patch.dict(
                            "os.environ",
                            {
                                "RECIR_PACKED_FORWARD_IMPL": implementation,
                                "RECIR_PACKED_STORE_IMPL": store_impl,
                            },
                        ):
                            actual = jax.jit(
                                partial(
                                    _packed_bmm_add,
                                    interpret=interpret,
                                ),
                                static_argnums=(4,),
                            )(
                                args[0],
                                args[1],
                                jnp.asarray(bias.numpy()),
                                args[2],
                                max_len,
                            )
                            plain = _packed_bmm(*args, max_len, interpret=interpret)
                        np.testing.assert_array_equal(
                            np.asarray(actual), expected.numpy()
                        )
                        np.testing.assert_array_equal(
                            np.asarray(plain), expected_plain.numpy()
                        )
        for implementation in ("native_pair", "native_pair_slice"):
            with patch.dict(
                "os.environ", {"RECIR_PACKED_FORWARD_IMPL": implementation}
            ):
                zero = _packed_bmm_add(
                    jnp.full((3, 128), -0.0, jnp.bfloat16),
                    jnp.ones((1, 128, 128), jnp.bfloat16),
                    jnp.full((1, 128), -0.0, jnp.float32),
                    jnp.asarray([0, 1], jnp.int32),
                    4,
                )
            self.assertFalse(np.any(np.signbit(np.asarray(zero))))

    def test_prefix_dma_read_preserves_valid_bits(self) -> None:
        rng = np.random.default_rng(20261028)
        for dtype in (jnp.float32, jnp.bfloat16):
            for max_len, k in (
                (4, 128),
                (129, 224),
                (512, 128),
                (513, 128),
                (3574, 224),
                (3574, 896),
            ):
                capacity = 2 * max_len + 32
                source = rng.normal(size=(capacity, k)).astype(np.float32)
                source[8, 1] = -0.0
                source[9, 1] = np.nan
                values = jnp.asarray(source, dtype=dtype)
                padded = jnp.pad(values, ((0, max(max_len + 7, 520)), (0, (-k) % 128)))
                if dtype == jnp.bfloat16:
                    padded = jax.lax.bitcast_convert_type(
                        jax.lax.bitcast_convert_type(padded, jnp.uint16)
                        .at[9, 1]
                        .set(jnp.uint16(0x7FC1)),
                        jnp.bfloat16,
                    )
                raw = np.asarray(padded)[:capacity, :k]
                for shift in (0, 1, 7):
                    for raw_starts, lengths in (
                        ((shift, shift), (0, 0)),
                        (
                            (shift, shift + min(129, max_len)),
                            (min(129, max_len), min(7, max_len)),
                        ),
                        ((shift, shift + max_len), (max_len, max_len)),
                        ((capacity - shift - 1, capacity), (max_len + 11, 0)),
                    ):
                        starts = jnp.array(raw_starts, jnp.int32)
                        sizes = jnp.array(lengths, jnp.int32)
                        actual = np.asarray(
                            _packed_native_pair_prefix_dma_read(
                                padded,
                                starts,
                                sizes,
                                max_len,
                                k,
                                capacity,
                                interpret=True,
                            )
                        ).reshape(2, max_len, k)
                        expected = np.zeros((2, max_len, k), raw.dtype)
                        for group in range(2):
                            start = int(starts[group])
                            length = min(lengths[group], max_len, capacity - start)
                            expected[group, :length] = raw[start : start + length]
                        bitdtype = np.uint16 if dtype == jnp.bfloat16 else np.uint32
                        np.testing.assert_array_equal(
                            actual.view(bitdtype), expected.view(bitdtype)
                        )

    def test_dma_read_preserves_input_bits(self) -> None:
        rng = np.random.default_rng(719)
        for dtype, fixture in product(
            (jnp.float32, jnp.bfloat16),
            ((4, 128), (129, 224), (3574, 224), (3574, 896)),
        ):
            max_len, width = fixture
            capacity = 2 * max_len + 23
            source = rng.normal(size=(capacity, width)).astype(np.float32)
            source[::13, 0] = -0.0
            source[::17, 1] = np.nan
            values = jnp.asarray(source).astype(dtype)
            if dtype == jnp.bfloat16:
                payloads = np.asarray(values).view(np.uint16).copy()
                payloads[8, 1] = 0x7FC1
                payloads[9, 1] = 0xFFC5
                values = jax.lax.bitcast_convert_type(jnp.asarray(payloads), dtype)
            rows = ((max_len + 7 + 127) // 128) * 128
            padded = jnp.pad(values, ((0, rows), (0, (-width) % 128)))
            for shift in (0, 1, 7):
                starts = jnp.asarray([shift, capacity], jnp.int32)
                actual = _packed_native_pair_dma_read(
                    padded, starts, max_len, width, interpret=True
                )
                expected = padded[
                    starts[:, None] + jnp.arange(max_len)[None, :], :width
                ].reshape(2 * max_len, width)
                bits = np.uint16 if dtype == jnp.bfloat16 else np.uint32
                np.testing.assert_array_equal(
                    np.asarray(actual).view(bits), np.asarray(expected).view(bits)
                )

    def test_prefix_store_preserves_boundary_rows(self) -> None:
        rng = np.random.default_rng(719)
        fixtures = []
        for shift in range(8):
            fixtures.extend(
                (
                    (129, 301, [shift, shift + 129], [129, 137], 128),
                    (129, 301, [shift, shift], [0, 129], 128),
                    (129, 150, [shift, 140], [129, 129], 128),
                )
            )
        fixtures.append((3574, 7199, [7, 1266], [1259, 3574], 512))
        for max_len, capacity, starts, lengths, width in fixtures:
            values = rng.normal(size=(2, max_len, width)).astype(np.float32)
            values[:, ::13, 0] = -0.0
            values[:, ::17, 1] = np.nan
            initial = rng.normal(size=(capacity + max(max_len, 128), width)).astype(
                np.float32
            )
            expected = initial.copy()
            for group in range(2):
                length = min(lengths[group], max_len, capacity - starts[group])
                expected[starts[group] : starts[group] + length] = values[
                    group, :length
                ]
            actual = _packed_native_pair_prefix_store(
                jnp.asarray(values),
                jnp.asarray(initial),
                jnp.asarray(starts, jnp.int32),
                jnp.asarray(lengths, jnp.int32),
                capacity,
                interpret=True,
            )
            np.testing.assert_array_equal(
                np.asarray(actual).view(np.uint32), expected.view(np.uint32)
            )

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
        aligned_dw = _packed_weight_grad_aligned(
            jnp.asarray(values.float().numpy()).astype(jnp.bfloat16),
            jnp.asarray(grad.numpy()),
            jnp.asarray(offsets.numpy(), jnp.int32),
            180,
            jnp.bfloat16,
            interpret=True,
        )
        np.testing.assert_array_equal(
            np.asarray(aligned_dw.astype(jnp.float32)), expected_dw.float().numpy()
        )

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
                if dtype == torch.bfloat16:
                    aligned_dw = _packed_weight_grad_aligned(
                        jx, jg, jnp.asarray(offsets), 128, jw.dtype, interpret=True
                    )
                    torch.testing.assert_close(
                        torch.tensor(np.asarray(aligned_dw.astype(jnp.float32))).to(
                            dtype
                        ),
                        dw,
                        rtol=2**-6,
                        atol=1e-5,
                    )
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
