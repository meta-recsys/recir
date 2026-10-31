# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import functools
import itertools
import unittest
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F
from parameterized import parameterized
from recir.torchax.ops.jaten import _aten__embedding_bag


class EmbeddingBagTest(unittest.TestCase):
    def setUp(self) -> None:
        self.enterContext(jax.enable_x64(True))

    def _assert_close(self, actual: Any, expected: torch.Tensor) -> None:
        tolerance = 2 * torch.finfo(expected.dtype).eps
        np.testing.assert_allclose(
            np.asarray(actual, dtype=np.float64),
            expected.detach().double().numpy(),
            rtol=tolerance,
            atol=tolerance,
        )

    @parameterized.expand(
        [
            (torch_dtype, jax_dtype, mode, sentinel)
            for (torch_dtype, jax_dtype), mode, sentinel in itertools.product(
                (
                    (torch.float16, jnp.float16),
                    (torch.bfloat16, jnp.bfloat16),
                    (torch.float32, jnp.float32),
                    (torch.float64, jnp.float64),
                ),
                (0, 1),
                (False, True),
            )
        ]
    )
    def test_sum_and_mean_with_empty_bags(
        self, torch_dtype: torch.dtype, jax_dtype: Any, mode: int, sentinel: bool
    ) -> None:
        indices = torch.tensor([4, 1, 4, 0, 3, 1], dtype=torch.int64)
        sentinel_offsets = torch.tensor([0, 0, 3, 3, 4, 6, 6], dtype=torch.int64)
        offsets = sentinel_offsets if sentinel else sentinel_offsets[:-1]
        weight = (torch.arange(18).reshape(6, 3).to(torch_dtype) / 4).requires_grad_()
        expected = F.embedding_bag(
            indices,
            weight,
            offsets,
            mode="sum" if mode == 0 else "mean",
            include_last_offset=sentinel,
        )
        coeff = torch.arange(18).reshape(6, 3).to(torch_dtype) / 8
        (expected * coeff).sum().backward()
        expected_grad = weight.grad
        assert expected_grad is not None
        bag = functools.partial(
            _aten__embedding_bag,
            indices=jnp.asarray(indices.numpy()),
            offsets=jnp.asarray(offsets.numpy()),
            mode=mode,
            include_last_offset=sentinel,
        )
        jax_weight = jnp.asarray(weight.detach().double().numpy(), jax_dtype)
        actual = jax.jit(lambda w: bag(w)[0])(jax_weight)
        jax_coeff = jnp.asarray(coeff.double().numpy(), jax_dtype)
        grad = jax.jit(jax.grad(lambda w: jnp.sum(bag(w)[0] * jax_coeff)))(jax_weight)
        self._assert_close(actual, expected)
        self._assert_close(grad, expected_grad)

    def test_all_bags_empty(self) -> None:
        bag = jax.jit(
            _aten__embedding_bag, static_argnames=("mode", "include_last_offset")
        )
        for mode in (0, 1):
            for include_last_offset in (False, True):
                with self.subTest(mode=mode, sentinel=include_last_offset):
                    offsets = jnp.zeros(4, dtype=jnp.int64)
                    result = bag(
                        jnp.ones((6, 3)),
                        jnp.array([], dtype=jnp.int64),
                        offsets,
                        mode=mode,
                        include_last_offset=include_last_offset,
                    )
                    num_bags = 3 if include_last_offset else 4
                    np.testing.assert_array_equal(result[0], np.zeros((num_bags, 3)))
                    np.testing.assert_array_equal(result[2], np.zeros(num_bags))
                    self.assertEqual(result[1].shape, (0,))

    def test_generated_bag_ids_are_sorted(self) -> None:
        for mode in (0, 1):
            bag = functools.partial(
                _aten__embedding_bag, mode=mode, include_last_offset=True
            )
            args = (
                jnp.arange(18, dtype=jnp.float32).reshape(6, 3),
                jnp.array([4, 1, 4, 0, 3, 1]),
                jnp.array([0, 0, 3, 3, 4, 6, 6]),
            )
            result = bag(*args)
            np.testing.assert_array_equal(result[1], np.array([1, 1, 1, 3, 4, 4]))
            np.testing.assert_array_equal(result[2], np.array([0, 3, 0, 1, 2, 0]))
            scatters = [
                eqn
                for eqn in jax.make_jaxpr(bag)(*args).jaxpr.eqns
                if eqn.primitive.name == "scatter-add"
            ]
            self.assertEqual(len(scatters), 1)
            self.assertTrue(scatters[0].params["indices_are_sorted"])
