# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for reusing device copies of unchanged torch state inputs."""

import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import torch
from recir.lowering.xla_device_state import XLADeviceStateManager
from recir.lowering.xla_lower import _reuse_unchanged_state_input, _XLAModule


class _CountingConvert:
    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor
        self.calls = 0

    def __call__(self) -> jax.Array:
        self.calls += 1
        return jnp.asarray(self.tensor.numpy())


def _parent_with_xla_module() -> tuple[torch.fx.GraphModule, _XLAModule]:
    lowered = _XLAModule(torch.fx.symbolic_trace(torch.nn.Identity()))
    root = torch.nn.Module()
    root.add_module("lowered", lowered)
    graph = torch.fx.Graph()
    arg = graph.placeholder("arg")
    result = graph.call_module("lowered", args=(arg,))
    graph.output(result)
    return torch.fx.GraphModule(root, graph), lowered


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

    def test_shared_tensor_and_layout_reuse_one_device_copy(self) -> None:
        manager = XLADeviceStateManager()
        gm, first_lowered = _parent_with_xla_module()
        second_lowered = _XLAModule(torch.fx.symbolic_trace(torch.nn.Identity()))
        gm.add_module("second_lowered", second_lowered)
        frozen = torch.ones(2)
        manager.device_state = [object()]
        manager.prepare_inputs(gm, [frozen], [], last_train_step=False)
        first_convert = _CountingConvert(frozen)
        second_convert = _CountingConvert(frozen)

        first = _reuse_unchanged_state_input(
            first_lowered._state_input_cache,
            (first_lowered._state_input_cache_namespace, 0),
            frozen,
            False,
            True,
            first_convert,
            "replicated",
        )
        second = _reuse_unchanged_state_input(
            second_lowered._state_input_cache,
            (second_lowered._state_input_cache_namespace, 0),
            frozen,
            False,
            True,
            second_convert,
            "replicated",
        )

        self.assertIs(first, second)
        self.assertIs(
            first_lowered._state_input_cache,
            second_lowered._state_input_cache,
        )
        self.assertEqual(1, first_convert.calls)
        self.assertEqual(0, second_convert.calls)

    def test_shared_tensor_with_different_layout_is_reconverted(self) -> None:
        cache = {}
        frozen = torch.ones(2)
        _reuse_unchanged_state_input(
            cache,
            (1, 0),
            frozen,
            False,
            True,
            _CountingConvert(frozen),
            "replicated",
        )
        convert = _CountingConvert(frozen)

        _reuse_unchanged_state_input(
            cache,
            (2, 0),
            frozen,
            False,
            True,
            convert,
            "fsdp",
        )

        self.assertEqual(1, convert.calls)

    def test_same_module_aliases_keep_distinct_device_copies(self) -> None:
        cache = {}
        frozen = torch.ones(2)
        namespace = object()
        first_convert = _CountingConvert(frozen)
        second_convert = _CountingConvert(frozen)

        first = _reuse_unchanged_state_input(
            cache,
            (namespace, 0),
            frozen,
            False,
            True,
            first_convert,
            "replicated",
        )
        second = _reuse_unchanged_state_input(
            cache,
            (namespace, 1),
            frozen,
            False,
            True,
            second_convert,
            "replicated",
        )

        self.assertIsNot(first, second)
        self.assertEqual(1, first_convert.calls)
        self.assertEqual(1, second_convert.calls)

    def test_cross_module_aliases_remain_distinct_within_each_module(self) -> None:
        cache = {}
        frozen = torch.ones(2)
        first_namespace = object()
        second_namespace = object()
        first_a = _reuse_unchanged_state_input(
            cache,
            (first_namespace, 0),
            frozen,
            False,
            True,
            _CountingConvert(frozen),
            "replicated",
        )
        first_b = _reuse_unchanged_state_input(
            cache,
            (first_namespace, 1),
            frozen,
            False,
            True,
            _CountingConvert(frozen),
            "replicated",
        )
        second_a = _reuse_unchanged_state_input(
            cache,
            (second_namespace, 0),
            frozen,
            False,
            True,
            _CountingConvert(frozen),
            "replicated",
        )
        second_convert = _CountingConvert(frozen)
        second_b = _reuse_unchanged_state_input(
            cache,
            (second_namespace, 1),
            frozen,
            False,
            True,
            second_convert,
            "replicated",
        )

        self.assertIsNot(first_a, first_b)
        self.assertIsNot(second_a, second_b)
        self.assertTrue(
            (second_a is first_a and second_b is first_b)
            or (second_a is first_b and second_b is first_a)
        )
        self.assertEqual(0, second_convert.calls)

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

    def test_invalidation_reloads_storage_alias_mutation(self) -> None:
        manager = XLADeviceStateManager()
        gm, lowered = _parent_with_xla_module()
        frozen = torch.zeros(2)
        manager.device_state = [object()]
        manager.prepare_inputs(gm, [frozen], [], last_train_step=False)
        convert = _CountingConvert(frozen)
        _reuse_unchanged_state_input(
            lowered._state_input_cache, 0, frozen, False, True, convert
        )
        version = frozen._version
        data_ptr = frozen.data_ptr()

        frozen.data.fill_(3.0)
        self.assertEqual(version, frozen._version)
        self.assertEqual(data_ptr, frozen.data_ptr())
        manager.invalidate_device_state()
        updated = _reuse_unchanged_state_input(
            lowered._state_input_cache, 0, frozen, False, True, convert
        )

        self.assertEqual(2, convert.calls)
        self.assertEqual([3.0, 3.0], updated.tolist())

    def test_successful_terminal_commit_releases_input_cache(self) -> None:
        manager = XLADeviceStateManager()
        gm, lowered = _parent_with_xla_module()
        frozen = torch.ones(2)
        manager.device_state = [object()]
        manager.prepare_inputs(gm, [frozen], [], last_train_step=False)
        convert = _CountingConvert(frozen)
        first = _reuse_unchanged_state_input(
            lowered._state_input_cache, 0, frozen, False, True, convert
        )
        manager.device_state = None
        manager.prepare_inputs(gm, [frozen], [], last_train_step=True)

        manager.commit_step_state([frozen], [frozen], last_train_step=True)
        second = _reuse_unchanged_state_input(
            lowered._state_input_cache, 0, frozen, False, True, convert
        )

        self.assertIsNone(manager.device_state)
        self.assertIsNot(first, second)
        self.assertEqual(2, convert.calls)

    def test_successful_prefill_releases_prior_input_cache(self) -> None:
        manager = XLADeviceStateManager()
        gm, lowered = _parent_with_xla_module()
        frozen = torch.ones(2)
        manager.device_state = [object()]
        manager.prepare_inputs(gm, [frozen], [], last_train_step=False)
        convert = _CountingConvert(frozen)
        _reuse_unchanged_state_input(
            lowered._state_input_cache, 0, frozen, False, True, convert
        )
        replacement = jnp.full((2,), 4.0)

        with mock.patch.object(_XLAModule, "_to_sharded_jax", return_value=replacement):
            manager.prefill_device_state_from_host(gm, [frozen])
        updated = _reuse_unchanged_state_input(
            lowered._state_input_cache, 0, frozen, False, True, convert
        )

        self.assertIsNotNone(manager.device_state)
        assert manager.device_state is not None
        self.assertIs(replacement, manager.device_state[0])
        self.assertEqual(2, convert.calls)
        self.assertEqual([1.0, 1.0], updated.tolist())

    def test_failed_terminal_commit_preserves_input_cache_for_retry(self) -> None:
        manager = XLADeviceStateManager()
        gm, lowered = _parent_with_xla_module()
        frozen = torch.ones(2)
        manager.device_state = [frozen]
        manager.prepare_inputs(gm, [frozen], [], last_train_step=False)
        convert = _CountingConvert(frozen)
        first = _reuse_unchanged_state_input(
            lowered._state_input_cache, 0, frozen, False, True, convert
        )
        resident_state = manager.device_state
        manager.prepare_inputs(gm, [frozen], [], last_train_step=True)

        with self.assertRaises(AssertionError):
            manager.commit_step_state([frozen], [torch.ones(3)], last_train_step=True)
        retry = _reuse_unchanged_state_input(
            lowered._state_input_cache, 0, frozen, False, True, convert
        )

        self.assertIs(manager.device_state, resident_state)
        self.assertIs(first, retry)
        self.assertEqual(1, convert.calls)
