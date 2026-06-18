# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Numerical tests for the fbgemm jagged op lowering (torch fallback).

Focus: the scatter-free (gather) ``dense_to_jagged`` lowering in
``recir.lowering.fbgemm_fallback`` -- the ``recir @compile`` /
``SPMD_DECOMP_TABLE`` path used by the rankfm TPU training bench. It replaced an
XLA scatter-add that dominated TPU step time. We verify it against an independent
oracle and via a jagged->padded->jagged round trip (the forward + backward
pattern exercised in training), plus the opt-in uniform fast path.

Also tests the new recir::dense_to_jagged and recir::jagged_to_padded_dense
custom ops with non-uniform offsets (Phase A shard-local implementation).
"""

import os

# Force multiple CPU devices so the multi-device shard_map branch of the
# jagged lowerings is actually exercised by tests below (must be set before
# jax initializes its backend).
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import random
import unittest
from typing import Any

# fbgemm_fallback registers the torch.ops.fbgemm op schemas (fbgemm_gpu is not
# loaded in this CPU test env).
import recir.lowering.fbgemm_fallback  # noqa: F401
import recir.ops.jagged_ops  # noqa: F401  # Register custom ops
import torch


def _offsets(lengths: list[int]) -> list[int]:
    o = [0]
    for length in lengths:
        o.append(o[-1] + length)
    return o


def _ref_dense_to_jagged_2d(
    dense: list[list[float]], lengths: list[int], total_l: int
) -> list[float]:
    """Oracle: concatenate the valid prefix of each row, pad to total_l."""
    out: list[float] = []
    for b, length in enumerate(lengths):
        for i in range(length):
            out.append(dense[b][i])
    out = out[:total_l] + [0.0] * max(0, total_l - len(out))
    return out


class RecirJaggedOpsTest(unittest.TestCase):
    """Tests for recir::dense_to_jagged and recir::jagged_to_padded_dense custom ops.

    These ops are the Phase A shard-local implementation that routes fbgemm ops
    through custom ops with shard_map lowerings. Tests verify numerics for
    non-uniform offsets and the single-device short-circuit path.
    """

    def _run_recir_d2j(self, dense: list[list[float]], lengths: list[int]) -> None:
        """Test recir::dense_to_jagged with non-uniform offsets."""
        offsets = _offsets(lengths)
        total_l = offsets[-1]
        dense_t = torch.tensor(dense, dtype=torch.float32)
        off_t = [torch.tensor(offsets, dtype=torch.int32)]

        # Call the custom op
        result, result_offsets = torch.ops.recir.dense_to_jagged(
            dense_t, off_t, total_l
        )

        # Verify against oracle
        want = _ref_dense_to_jagged_2d(dense, lengths, total_l)
        self.assertEqual(
            [round(x, 5) for x in result.tolist()],
            [round(x, 5) for x in want],
        )
        self.assertEqual(result.shape[0], total_l)
        # Offsets should be passed through
        self.assertTrue(torch.equal(result_offsets, off_t[0]))

    def _run_recir_j2pd(
        self,
        jagged: list[list[float]],
        lengths: list[int],
        max_len: int,
    ) -> None:
        """Test recir::jagged_to_padded_dense with non-uniform offsets."""
        offsets = _offsets(lengths)
        jagged_t = torch.tensor(jagged, dtype=torch.float32)
        off_t = [torch.tensor(offsets, dtype=torch.int32)]

        result = torch.ops.recir.jagged_to_padded_dense(jagged_t, off_t, [max_len], 0.0)

        # Verify shape
        B = len(lengths)
        D = len(jagged[0]) if jagged else 0
        self.assertEqual(tuple(result.shape), (B, max_len, D))

        # Verify values match expected padded layout
        for b, length in enumerate(lengths):
            for i in range(length):
                for d in range(D):
                    self.assertAlmostEqual(
                        result[b, i, d].item(),
                        jagged[offsets[b] + i][d],
                        places=5,
                    )
            # Padding should be zero
            for i in range(length, max_len):
                for d in range(D):
                    self.assertAlmostEqual(result[b, i, d].item(), 0.0, places=5)

    def test_non_uniform_d2j(self) -> None:
        """Non-uniform lengths: recir::dense_to_jagged matches oracle."""
        self._run_recir_d2j(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
            [2, 4, 1],
        )

    def test_non_uniform_j2pd(self) -> None:
        """Non-uniform lengths: recir::jagged_to_padded_dense pads correctly."""
        jagged = [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0],
            [13.0, 14.0],
        ]
        # lengths: [2, 3, 2] -> offsets [0, 2, 5, 7]
        self._run_recir_j2pd(jagged, [2, 3, 2], max_len=4)

    def test_zero_length_segments(self) -> None:
        """Zero-length segments handled correctly."""
        self._run_recir_d2j(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [2, 0, 1],
        )

    def test_single_device_short_circuit(self) -> None:
        """Single-device path produces correct results (no shard_map)."""
        # This tests the CPU fallback which is the single-device path
        dense = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        lengths = [2, 3]
        offsets = _offsets(lengths)
        total_l = offsets[-1]
        dense_t = torch.tensor(dense, dtype=torch.float32)
        off_t = [torch.tensor(offsets, dtype=torch.int32)]

        result, _ = torch.ops.recir.dense_to_jagged(dense_t, off_t, total_l)
        want = _ref_dense_to_jagged_2d(dense, lengths, total_l)
        self.assertEqual(
            [round(x, 5) for x in result.tolist()],
            [round(x, 5) for x in want],
        )

    def test_roundtrip_via_custom_ops(self) -> None:
        """Round-trip through custom ops recovers original values."""
        random.seed(23)
        B = 3
        max_len = 5
        D = 4
        lengths = [2, 4, 1]
        offsets = _offsets(lengths)
        total_l = offsets[-1]

        jagged = [[round(random.random(), 3) for _ in range(D)] for _ in range(total_l)]
        jagged_t = torch.tensor(jagged, dtype=torch.float32)
        off_t = [torch.tensor(offsets, dtype=torch.int32)]

        # jagged -> padded
        padded = torch.ops.recir.jagged_to_padded_dense(jagged_t, off_t, [max_len], 0.0)
        self.assertEqual(tuple(padded.shape), (B, max_len, D))

        # padded -> jagged (back)
        back, _ = torch.ops.recir.dense_to_jagged(padded, off_t, total_l)
        self.assertEqual(tuple(back.shape), (total_l, D))
        self.assertTrue(
            bool(torch.allclose(back, jagged_t, atol=1e-5)),
            "Custom op round-trip failed",
        )


class RecirJaggedOpsMakeFxTest(unittest.TestCase):
    """Regression for 'Mixing fake modes NYI': the register_fake kernels must
    allocate outputs in the ambient fake mode (no new FakeTensorMode). Tracing
    the custom ops under make_fx(tracing_mode='fake') exercises the fake kernels
    the same way recir.trace / make_fx does on the TPU path (the plain CPU tests
    call the impls directly and do NOT catch this)."""

    def test_dense_to_jagged_make_fx_fake(self) -> None:
        from torch.fx.experimental.proxy_tensor import make_fx

        def f(dense: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
            out, _ = torch.ops.recir.dense_to_jagged(
                dense, [offsets], dense.shape[0] * dense.shape[1]
            )
            return out

        dense = torch.randn(3, 4, 8)
        offsets = torch.tensor([0, 4, 8, 12], dtype=torch.int64)
        gm = make_fx(f, tracing_mode="fake")(dense, offsets)
        self.assertTrue(any(n.op == "call_function" for n in gm.graph.nodes))

    def test_jagged_to_padded_dense_make_fx_fake(self) -> None:
        from torch.fx.experimental.proxy_tensor import make_fx

        def f(values: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
            return torch.ops.recir.jagged_to_padded_dense(values, [offsets], [4], 0.0)

        values = torch.randn(12, 8)
        offsets = torch.tensor([0, 4, 8, 12], dtype=torch.int64)
        gm = make_fx(f, tracing_mode="fake")(values, offsets)
        self.assertTrue(any(n.op == "call_function" for n in gm.graph.nodes))

    def test_load_ops_imports_jagged_ops(self) -> None:
        # Regression guard for "torchax Tensors can only do math within the
        # torchax environment": Environment.load_ops() MUST import jagged_ops so
        # the custom-op JAX lowerings register in all_aten_ops before the env
        # snapshots it. Dropped twice already; if it regresses the bench dies
        # ~15 min into a TPU run. Checking the source catches the drop on CPU in
        # ms (constructing the env is unreliable here due to an unrelated
        # aten.prod overload).
        import inspect

        from recir.torchax.tensor import Environment

        src = inspect.getsource(Environment.load_ops)
        self.assertIn(
            "jagged_ops",
            src,
            "Environment.load_ops() must import recir.torchax.ops.jagged_ops so "
            "the recir::dense_to_jagged / jagged_to_padded_dense JAX lowerings "
            "register before the torchax env snapshots all_aten_ops.",
        )


class RecirJaggedOpsMultiDeviceJitTest(unittest.TestCase):
    """Exercises the MULTI-device shard_map branch UNDER jax.jit -- the path the
    TPU bench actually uses. The other tests only hit the single-device
    short-circuit, so they missed every multi-device bug (shard_map shape
    mismatch, int(seq_offsets[-1]) ConcretizationTypeError). XLA_FLAGS at the top
    of this module forces >=2 CPU devices so these run on CI."""

    def _uniform_offsets(self, b: int, length: int) -> "Any":
        import jax.numpy as jnp

        return jnp.arange(0, (b + 1) * length, length, dtype=jnp.int32)

    def test_dense_to_jagged_multidevice_jit(self) -> None:
        import jax
        import jax.numpy as jnp
        from recir.torchax.ops.jagged_ops import _recir_dense_to_jagged

        w = len(jax.devices())
        if w < 2:
            self.skipTest("need >= 2 jax devices")
        b, n, d = 4 * w, 4, 8  # uniform L=n so total_L == b*n, divisible by w
        dense = jnp.arange(b * n * d, dtype=jnp.float32).reshape(b, n, d)
        offs = self._uniform_offsets(b, n)
        total_l: int = b * n

        def f(dense_in: "Any", offs_in: "Any") -> "Any":
            out, _ = _recir_dense_to_jagged(dense_in, [offs_in], total_l)
            return out

        # jit-tracing raises ConcretizationTypeError if the lowering int()'d a
        # traced offset; the >=2-device mesh runs the shard_map branch.
        out = jax.jit(f)(dense, offs)
        self.assertEqual(tuple(out.shape), (total_l, d))

    def test_jagged_to_padded_dense_multidevice_jit(self) -> None:
        import jax
        import jax.numpy as jnp
        from recir.torchax.ops.jagged_ops import _recir_jagged_to_padded_dense

        w = len(jax.devices())
        if w < 2:
            self.skipTest("need >= 2 jax devices")
        b, d = 4 * w, 8
        length: int = 4
        total_l = b * length
        values = jnp.arange(total_l * d, dtype=jnp.float32).reshape(total_l, d)
        offs = self._uniform_offsets(b, length)

        def f(values_in: "Any", offs_in: "Any") -> "Any":
            return _recir_jagged_to_padded_dense(values_in, [offs_in], [length], 0.0)

        out = jax.jit(f)(values, offs)
        self.assertEqual(tuple(out.shape), (b, length, d))

    def test_jagged_to_padded_dense_1d_multidevice_jit(self) -> None:
        # 1D jagged values (mask/timestamp tensors) -> dense [B, max_len].
        # Regression for shard_map "in_specs entry too long" on rank-1 values.
        import jax
        import jax.numpy as jnp
        from recir.torchax.ops.jagged_ops import _recir_jagged_to_padded_dense

        w = len(jax.devices())
        if w < 2:
            self.skipTest("need >= 2 jax devices")
        b = 4 * w
        length: int = 4
        total_l = b * length
        values = jnp.arange(total_l, dtype=jnp.float32)  # rank-1
        offs = self._uniform_offsets(b, length)

        def f(values_in: "Any", offs_in: "Any") -> "Any":
            return _recir_jagged_to_padded_dense(values_in, [offs_in], [length], 0.0)

        out = jax.jit(f)(values, offs)
        self.assertEqual(tuple(out.shape), (b, length))  # dense [B, max_len]

    def test_dense_to_jagged_2d_multidevice_jit(self) -> None:
        # 2D dense [B, N] -> 1D jagged [total_L].
        import jax
        import jax.numpy as jnp
        from recir.torchax.ops.jagged_ops import _recir_dense_to_jagged

        w = len(jax.devices())
        if w < 2:
            self.skipTest("need >= 2 jax devices")
        b, n = 4 * w, 4
        dense = jnp.arange(b * n, dtype=jnp.float32).reshape(b, n)  # rank-2
        offs = self._uniform_offsets(b, n)
        total_l: int = b * n

        def f(dense_in: "Any", offs_in: "Any") -> "Any":
            out, _ = _recir_dense_to_jagged(dense_in, [offs_in], total_l)
            return out

        out = jax.jit(f)(dense, offs)
        self.assertEqual(tuple(out.shape), (total_l,))  # jagged 1D


if __name__ == "__main__":
    unittest.main()
