# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for reusing device copies of unchanged torch state inputs."""

import unittest

import jax
import jax.numpy as jnp
import torch
from recir.lowering.xla_lower import _reuse_unchanged_state_input


class _CountingConvert:
    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor
        self.calls = 0

    def __call__(self) -> jax.Array:
        self.calls += 1
        return jnp.asarray(self.tensor.numpy())


class StateInputReuseTest(unittest.TestCase):
    def test_unchanged_frozen_tensor_is_converted_once(self) -> None:
        cache = {}
        frozen = torch.arange(6, dtype=torch.float32)
        convert = _CountingConvert(frozen)

        first = _reuse_unchanged_state_input(cache, 3, frozen, False, True, convert)
        second = _reuse_unchanged_state_input(cache, 3, frozen, False, True, convert)

        self.assertIs(first, second)
        self.assertEqual(1, convert.calls)

    def test_in_place_update_forces_reconversion(self) -> None:
        cache = {}
        state = torch.zeros(4)
        convert = _CountingConvert(state)
        _reuse_unchanged_state_input(cache, 0, state, False, True, convert)

        state.add_(1.0)
        updated = _reuse_unchanged_state_input(cache, 0, state, False, True, convert)

        self.assertEqual(2, convert.calls)
        self.assertEqual([1.0] * 4, updated.tolist())

    def test_new_tensor_object_is_never_a_hit(self) -> None:
        cache = {}
        first = torch.ones(2)
        _reuse_unchanged_state_input(
            cache, 0, first, False, True, _CountingConvert(first)
        )
        replacement = torch.ones(2)
        convert = _CountingConvert(replacement)

        _reuse_unchanged_state_input(cache, 0, replacement, False, True, convert)

        self.assertEqual(1, convert.calls)

    def test_batch_inputs_are_not_cached(self) -> None:
        cache = {}
        batch = torch.ones(2, 3)
        convert = _CountingConvert(batch)

        _reuse_unchanged_state_input(cache, 0, batch, True, True, convert)
        _reuse_unchanged_state_input(cache, 0, batch, True, True, convert)

        self.assertEqual(2, convert.calls)
        self.assertEqual({}, cache)

    def test_deleted_device_copy_is_reconverted(self) -> None:
        cache = {}
        donated = torch.ones(3)
        convert = _CountingConvert(donated)
        _reuse_unchanged_state_input(cache, 1, donated, False, True, convert).delete()

        _reuse_unchanged_state_input(cache, 1, donated, False, True, convert)

        self.assertEqual(2, convert.calls)

    def test_mutable_state_is_not_cached(self) -> None:
        cache = {}
        state = torch.zeros(2)
        convert = _CountingConvert(state)

        _reuse_unchanged_state_input(cache, 0, state, False, False, convert)
        state.data.add_(1.0)
        updated = _reuse_unchanged_state_input(cache, 0, state, False, False, convert)

        self.assertEqual(2, convert.calls)
        self.assertEqual([1.0, 1.0], updated.tolist())
        self.assertEqual({}, cache)

    def test_inference_tensor_is_not_cached(self) -> None:
        cache = {}
        with torch.inference_mode():
            state = torch.ones(2)
        convert = _CountingConvert(state)

        _reuse_unchanged_state_input(cache, 0, state, False, True, convert)
        _reuse_unchanged_state_input(cache, 0, state, False, True, convert)

        self.assertEqual(2, convert.calls)
        self.assertEqual({}, cache)

    def test_jax_input_evicts_prior_host_copy(self) -> None:
        cache = {}
        frozen = torch.ones(2)
        _reuse_unchanged_state_input(
            cache, 0, frozen, False, True, _CountingConvert(frozen)
        )
        device_state = jnp.ones(2)

        actual = _reuse_unchanged_state_input(
            cache,
            0,
            device_state,
            False,
            True,
            lambda: device_state,
        )

        self.assertIs(device_state, actual)
        self.assertEqual({}, cache)
