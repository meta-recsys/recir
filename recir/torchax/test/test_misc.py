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


"""If you don't know which file a test should go, and don't want to make a new file
for a small test. PUt it here
"""

import unittest
from typing import Any

import jax
import jax.numpy as jnp
import recir.torchax
import torch


class MiscTest(unittest.TestCase):
    def test_extract_jax_kwargs(self) -> None:
        class M(torch.nn.Module):
            def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                return torch.sin(a) + torch.cos(b)

        weights, func = recir.torchax.extract_jax(M())
        res = func(
            weights,
            args=(),
            kwargs={"a": jnp.array([1, 2, 3]), "b": jnp.array([3, 4, 5])},
        )
        self.assertTrue(
            jnp.allclose(
                res, jnp.sin(jnp.array([1, 2, 3])) + jnp.cos(jnp.array([3, 4, 5]))
            )
        )

    def test_to_device(self) -> None:
        env = recir.torchax.default_env()
        with env:
            step1 = torch.ones(
                100,
                100,
            )
            step2 = torch.triu(step1, diagonal=1)
            step3 = step2.to(dtype=torch.bool, device="jax")
            self.assertEqual(step3.device.type, "jax")

    def test_to_device_twice(self) -> None:
        env = recir.torchax.default_env()
        with env:
            step1 = torch.ones(
                100,
                100,
            )
            step2 = torch.triu(step1, diagonal=1)
            step3 = step2.to(dtype=torch.bool, device="jax")
            step3.to("jax")
            self.assertEqual(step3.device.type, "jax")

    def test_random_with_tensor_input(self) -> None:
        env = recir.torchax.default_env()
        with env:
            env.manual_seed(torch.tensor(2))
            x = torch.randn((2, 2), device="jax")

        with env:
            env.manual_seed(torch.tensor(2))
            y = torch.randn((2, 2), device="jax")

            self.assertTrue(torch.allclose(x, y))

    def test_random_with_tensor_input(self) -> None:
        env: Any = recir.torchax.default_env()

        with env:

            @recir.torchax.interop.jax_jit
            def rand_plus_one(rng: int) -> torch.Tensor:
                env.manual_seed(torch.tensor(2))
                x = torch.randn((2, 2), device="jax") + 1
                return x

            x = rand_plus_one(0)
            y = rand_plus_one(0)
            self.assertTrue(torch.allclose(x, y))

    def test_zeros_with_explicit_size(self) -> None:
        with recir.torchax.default_env():
            a = torch.zeros(size=(4,), device="jax")
            self.assertEqual(a.shape[0], 4)


if __name__ == "__main__":
    unittest.main()
