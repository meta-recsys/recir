# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest
from typing import Any

import jax
import jax.numpy as jnp
import recir.ops.pallas.jagged_hstu_attention as jagged_hstu
from jax.sharding import Mesh

_WORLD_SIZE = 2
_LOCAL_PADDED_BATCH = 4
_LOCAL_REAL_BATCH = 3
_LOCAL_TOKENS = 20
_HEADS = 1
_DIM = 2
_MAX_SEQ_LEN = 5
_REAL_LENGTHS = ((4, 3, 5), (2, 5, 4))
_PAD_LENGTHS = (8, 9)
_TARGETS = (1, 2, 3, 99, 4, 5, 6, 98)


def _offsets() -> jax.Array:
    lengths: list[int] = []
    for device in range(_WORLD_SIZE):
        lengths.extend(_REAL_LENGTHS[device])
        lengths.append(_PAD_LENGTHS[device])
    return jnp.concatenate(
        (
            jnp.zeros((1,), dtype=jnp.int32),
            jnp.cumsum(jnp.asarray(lengths, dtype=jnp.int32)),
        )
    )


class PallasJaggedHSTUShardingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.assertEqual(len(jax.devices()), _WORLD_SIZE)
        self.mesh = Mesh(jax.devices(), (jagged_hstu._AXIS,))
        self.q = jnp.zeros(
            (_LOCAL_TOKENS * _WORLD_SIZE, _HEADS, _DIM),
            dtype=jnp.float32,
        )
        self.offsets = _offsets()
        self.targets = jnp.asarray(_TARGETS, dtype=jnp.int32)
        self.scale = jnp.asarray(0.25, dtype=jnp.float32)
        jagged_hstu.set_pad_seqs_per_device(
            real_batch_size=_LOCAL_REAL_BATCH * _WORLD_SIZE,
            padded_batch_size=_LOCAL_PADDED_BATCH * _WORLD_SIZE,
            world_size=_WORLD_SIZE,
        )

    def tearDown(self) -> None:
        jagged_hstu.clear_pad_seqs_per_device()

    def test_forward_uses_local_offsets_and_drops_pad_sequences(self) -> None:
        def runner(
            q_: Any,
            k_: Any,
            v_: Any,
            offsets_: Any,
            targets_: Any,
            scale_: Any,
        ) -> Any:
            output = jnp.zeros(
                (_LOCAL_REAL_BATCH, _HEADS, _MAX_SEQ_LEN, _DIM),
                dtype=q_.dtype,
            )
            output = output.at[:, 0, 0, 0].set(offsets_[1:].astype(q_.dtype))
            return output.at[:, 0, 0, 1].set(targets_.astype(q_.dtype))

        output = jagged_hstu.sample_sharded_jagged_hstu_attention(
            self.q,
            self.q,
            self.q,
            self.offsets,
            self.targets,
            self.scale,
            max_seq_len=_MAX_SEQ_LEN,
            alpha=1.0,
            world_size=_WORLD_SIZE,
            mesh=self.mesh,
            interpret=True,
            run_kernel=runner,
        )

        self.assertEqual(
            output[:, 0, 0, 0].tolist(),
            [4, 7, 12, 2, 7, 11],
        )
        self.assertEqual(
            output[:, 0, 0, 1].tolist(),
            [1, 2, 3, 4, 5, 6],
        )

    def test_backward_uses_same_sequences_and_zeros_dropped_token_tail(self) -> None:
        grad_out = jnp.zeros(
            (
                _LOCAL_PADDED_BATCH * _WORLD_SIZE,
                _HEADS,
                _MAX_SEQ_LEN,
                _DIM,
            ),
            dtype=jnp.float32,
        )

        def runner(
            grad_out_: Any,
            q_: Any,
            k_: Any,
            v_: Any,
            offsets_: Any,
            targets_: Any,
            scale_: Any,
        ) -> tuple[Any, Any, Any]:
            output = jnp.zeros_like(q_)
            output = output.at[: offsets_.shape[0], 0, 0].set(offsets_.astype(q_.dtype))
            output = output.at[: targets_.shape[0], 0, 1].set(targets_.astype(q_.dtype))
            valid_tokens = offsets_[-1]
            output = jnp.where(
                jnp.arange(q_.shape[0])[:, None, None] < valid_tokens,
                output,
                0,
            )
            return output, output, output

        gradients = jagged_hstu.sample_sharded_jagged_hstu_attention_backward(
            grad_out,
            self.q,
            self.q,
            self.q,
            self.offsets,
            self.targets,
            self.scale,
            max_seq_len=_MAX_SEQ_LEN,
            alpha=1.0,
            world_size=_WORLD_SIZE,
            mesh=self.mesh,
            interpret=True,
            run_kernel=runner,
        )

        for gradient in gradients:
            self.assertEqual(gradient.shape, self.q.shape)
            self.assertEqual(gradient[:4, 0, 0].tolist(), [0, 4, 7, 12])
            self.assertEqual(
                gradient[_LOCAL_TOKENS : _LOCAL_TOKENS + 4, 0, 0].tolist(),
                [0, 2, 7, 11],
            )
            self.assertTrue(bool(jnp.all(gradient[12:_LOCAL_TOKENS] == 0)))
            self.assertTrue(bool(jnp.all(gradient[31:] == 0)))

    def test_rejects_crossed_sequence_boundaries(self) -> None:
        bad_offsets = self.offsets.at[4].set(self.offsets[4] - 1)
        with self.assertRaisesRegex(ValueError, "complete sequences"):
            jagged_hstu.sample_sharded_jagged_hstu_attention(
                self.q,
                self.q,
                self.q,
                bad_offsets,
                self.targets,
                self.scale,
                max_seq_len=_MAX_SEQ_LEN,
                alpha=1.0,
                world_size=_WORLD_SIZE,
                mesh=self.mesh,
                interpret=True,
                run_kernel=lambda *_: jnp.zeros(
                    (_LOCAL_REAL_BATCH, _HEADS, _MAX_SEQ_LEN, _DIM),
                    dtype=jnp.float32,
                ),
            )
