# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import unittest
from collections.abc import Callable
from typing import Any

import recir.torchax
import recir.torchax.tensor
import torch


class SeqModel(torch.nn.Module):
    """Architecture is LLM generated"""

    def __init__(self) -> None:
        super().__init__()
        self.gru = torch.nn.GRU(20, 30, batch_first=True)
        self.linear = torch.nn.Linear(30, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(x)  # output, hidden state
        output = self.linear(output)
        return output


class TestTorchFunctions(unittest.TestCase):
    def setUp(self) -> None:
        recir.torchax.enable_globally()
        recir.torchax.enable_accuracy_mode()
        self.env = recir.torchax.default_env()

    def _test_tensor_constructor(
        self, arg: Any, kwargs: dict[str, Any] | None = None
    ) -> None:
        kwargs = kwargs or {}
        expected = torch.tensor(arg, **kwargs)

        actual = torch.tensor(arg, device="jax", **kwargs)
        self.assertIsInstance(actual, recir.torchax.tensor.Tensor)

        torch.testing.assert_close(actual.to("cpu"), expected)

    def test_tensor_2d(self) -> None:
        self._test_tensor_constructor([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]])

    def test_tensor_1d(self) -> None:
        self._test_tensor_constructor([0, 1])

    def test_tensor_scalar(self) -> None:
        self._test_tensor_constructor(3.14159)

    def test_tensor_empty(self) -> None:
        self._test_tensor_constructor([])

    def test_tensor_dtype(self) -> None:
        self._test_tensor_constructor(
            [[0.11111, 0.222222, 0.3333333]], {"dtype": torch.float64}
        )

    def test_full_int(self) -> None:
        a = torch.full((2, 2), 1, device="jax")
        self.assertEqual(a.dtype, torch.int64)

    def test_dont_capture_conversion(self) -> None:
        t = torch.tensor([1, 2, 3])
        with self.env:
            self.env.to_xla(t)
            # assert no exceptions

    def test_brackets(self) -> None:
        with self.env:
            a = torch.randn((2, 3))
            a[1] = 9
            self.assertEqual(a[1, 0].item(), 9)

    def test_bernoulli_inplace(self) -> None:
        with self.env:
            a = torch.randn((2, 3))
            a.bernoulli_(0.4)

    def test_flatten(self) -> None:
        with self.env:
            a = torch.randn((2, 3, 4))
            a = a.flatten(0, 1)
            self.assertEqual(tuple(a.shape), (6, 4))

    def test_rnn(self) -> None:
        model = SeqModel()
        x = torch.randn((2, 100, 20))
        res = model(x)
        with self.env:
            model.to("jax")
            x = x.to("jax")
            res2 = model(x)
            print(res.shape, res2.shape)

            self.assertEqual(res.shape, res2.shape)

    def test_rms_norm(self) -> None:
        model = torch.nn.RMSNorm([100, 20])
        x = torch.randn((2, 100, 20))
        res = model(x)

        with self.env:
            model.to("jax")
            x = x.to("jax")
            res2 = model(x)
            self.assertTrue(torch.allclose(res, res2.to("cpu")))

    def _test_requires_grad(
        self, func: Callable[..., Any], args: tuple[Any, ...]
    ) -> None:
        x = func(*args, requires_grad=True, device="jax")
        self.assertEqual(x.requires_grad, True)

    def test_requires_grad_ones(self) -> None:
        self._test_requires_grad(torch.ones, ((2, 2),))

    def test_requires_grad_zeros(self) -> None:
        self._test_requires_grad(torch.zeros, ((2, 2),))

    def test_requires_grad_empty(self) -> None:
        self._test_requires_grad(torch.empty, ((2, 2),))

    def test_requires_grad_empty_strided(self) -> None:
        self._test_requires_grad(torch.empty_strided, ((2, 2), (2, 1)))

    def test_requires_grad_tensor(self) -> None:
        self._test_requires_grad(torch.tensor, ([2.0, 2.0],))

    def test_requires_grad_eye(self) -> None:
        self._test_requires_grad(torch.eye, (2,))

    def test_requires_grad_randn(self) -> None:
        self._test_requires_grad(torch.randn, ((2, 2),))

    def test_requires_grad_rand(self) -> None:
        self._test_requires_grad(torch.rand, ((2, 2),))

    def test_requires_grad_full(self) -> None:
        self._test_requires_grad(torch.full, ((2, 2), 0.0))


if __name__ == "__main__":
    unittest.main()
