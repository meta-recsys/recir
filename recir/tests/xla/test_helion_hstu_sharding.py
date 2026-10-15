# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""CPU tests for the sample-sharded jagged HSTU wrappers' PRT pad-row handling.

The Helion kernels themselves need a TPU, so these tests inject a stand-in
``run_kernel`` and assert only on the sharding/offset math that
``sample_sharded_hstu_{fwd,bwd}_jax`` does around it -- specifically that each
device drops its trailing PaddingRegionTransform pad row before building
``local_offsets``. That row carries the chip's whole token deficit, so leaving it
in the grid would dominate the (quadratic) attention cost; see the PRT PAD ROWS
section comment in recir/ops/helion/jagged_hstu_attention.py.

The row COUNT is derived from the real vs padded batch (``set_pad_seqs_per_device``)
rather than assumed, so these tests configure it the way ``model_family`` does and
also pin the derivation itself.
"""

from typing import Any, Dict, List, Tuple
from unittest import mock, TestCase

import jax  # @manual
import jax.numpy as jnp
import recir.ops.helion.jagged_hstu_attention as jhstu
import torch
from jax.sharding import Mesh

# 2 spoofed CPU devices (see the BUCK env). b_local = 4 rows per device: 3 real
# + 1 PRT pad row. Per-sample budget 5 => l_local = 4 * 5 = 20, L = 40.
_WORLD_SIZE: int = 2
_B_LOCAL: int = 4
_B_RUN: int = 3  # real sequences per device, after the pad-row drop
_L_LOCAL: int = 20
_H: int = 1
_D: int = 2

# Real lengths per device, then the pad row absorbing that device's deficit so
# the full group sums to exactly l_local (the batch-alignment invariant).
_REAL_LENGTHS: List[List[int]] = [[4, 3, 5], [2, 5, 4]]
_PAD_LENGTHS: List[int] = [20 - 12, 20 - 11]  # 8, 9
# Distinct per-row num_targets; 99/98 sit on the pad rows and must never be read.
_NUM_TARGETS: List[int] = [1, 2, 3, 99, 4, 5, 6, 98]


def _seq_offsets() -> Any:
    lengths: List[int] = []
    for dev in range(_WORLD_SIZE):
        lengths.extend(_REAL_LENGTHS[dev])
        lengths.append(_PAD_LENGTHS[dev])
    return jnp.concatenate(
        [jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(jnp.asarray(lengths, jnp.int32))]
    )


def _encode(q_: Any, so_: Any, nt_: Any) -> Any:
    """Smuggle a device's local offsets/targets out through the output buffer.

    Under ``shard_map`` the runner's args are tracers, so their VALUES cannot be
    captured by closure; writing them into the returned ``[l_local, H, D]`` block
    lets the concatenated global output carry each device's view.
    """
    out = jnp.zeros_like(q_)
    out = out.at[: so_.shape[0], 0, 0].set(so_.astype(q_.dtype))
    return out.at[: nt_.shape[0], 0, 1].set(nt_.astype(q_.dtype))


class HelionHstuShardingTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.assertEqual(
            len(jax.devices()),
            _WORLD_SIZE,
            "test requires --xla_force_host_platform_device_count=2",
        )
        self.mesh: Mesh = Mesh(jax.devices(), (jhstu._AXIS,))
        self.q: Any = jnp.zeros((_L_LOCAL * _WORLD_SIZE, _H, _D), jnp.float32)
        self.seq_offsets: Any = _seq_offsets()
        self.num_targets: Any = jnp.asarray(_NUM_TARGETS, jnp.int32)
        self.attn_scale: Any = jnp.asarray(1.0, jnp.float32)
        self.captured: Dict[str, Tuple[int, ...]] = {}
        # b_local=4 padded rows/device vs b_run=3 real ones => 1 pad row. Both the
        # warm and the shard_map read this; configure it as model_family does.
        jhstu.set_pad_seqs_per_device(
            real_bs=_B_RUN * _WORLD_SIZE,
            padded_bs=_B_LOCAL * _WORLD_SIZE,
            world_size=_WORLD_SIZE,
        )

    def tearDown(self) -> None:
        jhstu.clear_pad_seqs_per_device()
        super().tearDown()

    def test_pad_seqs_is_derived_from_the_padded_and_real_batch(self) -> None:
        # NOT a fixed 1: compute_padded_shape rounds to the PRT's own modulus, and
        # a SparseCore-aware caller passes lcm(num_ranks, sc_modulus * num_processes), which the JAX
        # SparseCore backend makes world_size * num_sparsecores_per_device, so an
        # already-aligned batch gets num_sc pad rows per device.
        self.assertEqual(
            jhstu.set_pad_seqs_per_device(real_bs=6, padded_bs=8, world_size=2), 1
        )
        self.assertEqual(
            jhstu.set_pad_seqs_per_device(real_bs=256, padded_bs=512, world_size=64),
            4,
        )
        # Padding disabled -> padded == real -> nothing to drop.
        self.assertEqual(
            jhstu.set_pad_seqs_per_device(real_bs=8, padded_bs=8, world_size=2), 0
        )

    def _record_shapes(self, so_: Any, nt_: Any) -> None:
        self.captured["seq_offsets"] = tuple(so_.shape)
        self.captured["num_targets"] = tuple(nt_.shape)

    def test_drop_pad_rows_slices_lengths_and_targets_together(self) -> None:
        lengths = jnp.asarray([4, 3, 5, 8], jnp.int32)
        targets = jnp.asarray([1, 2, 3, 99], jnp.int32)

        kept_lengths, kept_targets = jhstu._drop_pad_rows(lengths, targets)

        # Both sliced by the same amount, so num_targets[i] still describes
        # sequence i -- a one-sided slice would silently shift the target cap.
        self.assertEqual(kept_lengths.tolist(), [4, 3, 5])
        self.assertEqual(kept_targets.tolist(), [1, 2, 3])

    def test_fwd_runs_real_sequences_only_with_local_offsets(self) -> None:
        def runner(q_: Any, k_: Any, v_: Any, so_: Any, nt_: Any, sc_: Any) -> Any:
            self._record_shapes(so_, nt_)
            return _encode(q_, so_, nt_)

        out = jhstu.sample_sharded_hstu_fwd_jax(
            0,
            1.0,
            self.q,
            self.q,
            self.q,
            self.seq_offsets,
            self.num_targets,
            self.attn_scale,
            _WORLD_SIZE,
            self.mesh,
            run_kernel=runner,
        )

        # The pad row is gone: b_run sequences, so offsets are [b_run+1].
        self.assertEqual(self.captured["seq_offsets"], (_B_RUN + 1,))
        self.assertEqual(self.captured["num_targets"], (_B_RUN,))
        # Offsets are rebuilt from the REAL lengths only and restart at 0 on each
        # device, and the pad rows' sentinel targets (99/98) are never passed in.
        for dev, real in enumerate(_REAL_LENGTHS):
            base = dev * _L_LOCAL
            expected_offsets = [0, real[0], real[0] + real[1], sum(real)]
            self.assertEqual(
                out[base : base + _B_RUN + 1, 0, 0].tolist(), expected_offsets
            )
            self.assertEqual(
                out[base : base + _B_RUN, 0, 1].tolist(),
                [float(t) for t in _NUM_TARGETS[dev * _B_LOCAL :][:_B_RUN]],
            )

    def test_bwd_drops_the_same_pad_row_as_fwd(self) -> None:
        def runner(
            do_: Any, q_: Any, k_: Any, v_: Any, so_: Any, nt_: Any, sc_: Any
        ) -> Tuple[Any, Any, Any]:
            self._record_shapes(so_, nt_)
            enc = _encode(q_, so_, nt_)
            return enc, enc, enc

        dq, dk, dv = jhstu.sample_sharded_hstu_bwd_jax(
            self.q,
            self.q,
            self.q,
            self.q,
            self.seq_offsets,
            self.num_targets,
            self.attn_scale,
            0,
            1.0,
            _WORLD_SIZE,
            self.mesh,
            run_kernel=runner,
        )

        self.assertEqual(self.captured["seq_offsets"], (_B_RUN + 1,))
        self.assertEqual(self.captured["num_targets"], (_B_RUN,))
        for grad in (dq, dk, dv):
            self.assertEqual(grad.shape, self.q.shape)
            self.assertEqual(
                grad[: _B_RUN + 1, 0, 0].tolist(),
                [0, 4, 7, 12],  # device 0
            )

    def test_zero_unvisited_tail_clears_only_the_dropped_pad_range(self) -> None:
        # The kernels allocate with empty_like (zeros_like breaks the bf16 jax_fn
        # export), so the dropped pad row's token range comes back as whatever the
        # buffer held. Anything non-finite there survives a downstream padding mask
        # (0 * NaN == NaN) and, on the backward, reaches the embedding gradient.
        offsets = jnp.asarray([0, 4, 7, 12], jnp.int32)
        x = jnp.full((_L_LOCAL, _H, _D), jnp.nan, jnp.float32).at[:12].set(3.0)

        out = jhstu._zero_unvisited_tail(x, offsets)

        self.assertTrue(bool(jnp.all(out[:12] == 3.0)))
        self.assertTrue(bool(jnp.all(out[12:] == 0.0)))

    def test_zero_unvisited_tail_preserves_bf16(self) -> None:
        # bf16 is the bench dtype; the mask must not upcast (which would change the
        # kernel arg signature downstream) and must not route through numpy.
        offsets = jnp.asarray([0, 2], jnp.int32)
        x = jnp.ones((4, _H, _D), jnp.bfloat16)

        out = jhstu._zero_unvisited_tail(x, offsets)

        self.assertEqual(out.dtype, jnp.bfloat16)
        self.assertEqual(out[:2].sum().astype(jnp.float32), 2 * _H * _D)
        self.assertEqual(out[2:].sum().astype(jnp.float32), 0)

    def test_warm_precompiles_the_shapes_and_attention_modes(self) -> None:
        # static_shapes=True keys the compiled kernel on shape, so the warm must hand
        # the kernel the POST-drop seq_offsets/num_targets shapes. Drift here is a
        # cache miss at runtime -> Helion recompiles nested inside the jax lower
        # tracer -> "SymIntArrayRef expected to contain only concrete integers".
        # The warm drives the SAME sharded wrapper the step does, so this asserts at
        # the kernel boundary in both paths: patch the wrapper's default runner and
        # compare what it receives from the warm vs from a runtime call.
        warmed: List[Tuple[int, int, int, int]] = []

        def fwd_fn(
            N: Any,
            alpha: Any,
            q_: Any,
            k_: Any,
            v_: Any,
            so_: Any,
            nt_: Any,
            sc_: Any,
            max_attn_len: Any,
            full_attn_size: Any,
        ) -> Any:
            warmed.append(
                (
                    so_.shape[0],
                    nt_.shape[0],
                    int(max_attn_len),
                    int(full_attn_size),
                )
            )
            return jnp.zeros_like(q_), jnp.zeros((1,), jnp.float32)

        def bwd_fn(
            do_: Any,
            q_: Any,
            k_: Any,
            v_: Any,
            so_: Any,
            nt_: Any,
            sc_: Any,
            N: Any,
            alpha: Any,
            max_attn_len: Any,
            full_attn_size: Any,
        ) -> Any:
            warmed.append(
                (
                    so_.shape[0],
                    nt_.shape[0],
                    int(max_attn_len),
                    int(full_attn_size),
                )
            )
            return jnp.zeros_like(q_), jnp.zeros_like(q_), jnp.zeros_like(q_)

        sig = {
            (
                _L_LOCAL * _WORLD_SIZE,
                _H,
                _D,
                _B_LOCAL * _WORLD_SIZE,
                5,
                0,
                0,
                torch.float32,
            ),
            (
                _L_LOCAL * _WORLD_SIZE,
                _H,
                _D,
                _B_LOCAL * _WORLD_SIZE,
                5,
                256,
                256,
                torch.float32,
            ),
        }
        with (
            mock.patch.object(jhstu, "hstu_ragged_attention_fwd_jax_fn", fwd_fn),
            mock.patch.object(jhstu, "hstu_ragged_attention_bwd_jax_fn", bwd_fn),
            mock.patch.object(jhstu, "get_traced_fwd_sigs", lambda: sig),
            mock.patch.object(jhstu, "get_traced_bwd_sigs", lambda: sig),
        ):
            jhstu.warm_traced_hstu_kernels(_WORLD_SIZE)

        def runner(q_: Any, k_: Any, v_: Any, so_: Any, nt_: Any, sc_: Any) -> Any:
            self._record_shapes(so_, nt_)
            return jnp.zeros_like(q_)

        jhstu.sample_sharded_hstu_fwd_jax(
            0,
            1.0,
            self.q,
            self.q,
            self.q,
            self.seq_offsets,
            self.num_targets,
            self.attn_scale,
            _WORLD_SIZE,
            self.mesh,
            run_kernel=runner,
        )

        runtime_shape = (
            self.captured["seq_offsets"][0],
            self.captured["num_targets"][0],
        )
        self.assertEqual(
            warmed,
            [
                (*runtime_shape, 0, 0),
                (*runtime_shape, 256, 256),
                (*runtime_shape, 0, 0),
                (*runtime_shape, 256, 256),
            ],
        )
