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
from typing import Any

import jax
import jax.numpy as jnp
import recir.torchax
import torch
from flax import linen as nn
from recir.torchax.flax import FlaxNNModule
from recir.torchax.interop import jax_jit


class CNN(nn.Module):
    """A simple CNN model."""

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x


class FlaxTest(unittest.TestCase):
    def test_flax_simple(self) -> None:
        flax_model = CNN()

        inputs = jnp.ones((1, 28, 28, 1))
        env = recir.torchax.default_env()
        state = flax_model.init(env.prng_key, inputs)
        expected = flax_model.apply(state, inputs)

        env = recir.torchax.default_env()
        nn_module: FlaxNNModule = FlaxNNModule(env, flax_model, (inputs,), {})
        res = nn_module.forward(inputs)

        # pyrefly: ignore [bad-argument-type]
        self.assertTrue(jnp.allclose(res.jax(), expected))

    def test_flax_functional_call(self) -> None:
        flax_model = CNN()

        inputs = jnp.ones((1, 28, 28, 1))
        env = recir.torchax.default_env()
        state = flax_model.init(env.prng_key, inputs)
        expected = flax_model.apply(state, inputs)

        env = recir.torchax.default_env()
        nn_module: FlaxNNModule = FlaxNNModule(env, flax_model, (inputs,), {})

        @jax_jit
        def jitted(weights: Any, args: Any) -> Any:
            return torch.func.functional_call(nn_module, weights, args)

        with env:
            inputs_torch = torch.ones((1, 28, 28, 1), device="jax")
            state_dict = nn_module.state_dict()
            res = jitted(state_dict, inputs_torch)
            # pyrefly: ignore [bad-argument-type]
            self.assertTrue(jnp.allclose(res.jax(), expected))

    def test_flax_module_nested(self) -> None:
        env = recir.torchax.default_env()

        class Parent(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = torch.nn.Linear(28, 28)
                sample_cnn_inputs = torch.ones((1, 28, 28, 1), device="jax")
                self.cnn = FlaxNNModule(env, CNN(), (sample_cnn_inputs,), {})

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                y = self.a(x)
                y = y.reshape((-1, 28, 28, 1))
                res = self.cnn(y)
                return res

        with env:
            nn_module: torch.nn.Module = Parent().to("jax")

            @jax_jit
            def jitted(weights: Any, args: Any) -> Any:
                return torch.func.functional_call(nn_module, weights, args)

            inputs_torch = torch.ones((1, 28, 28), device="jax")
            state_dict = nn_module.state_dict()
            res = jitted(state_dict, inputs_torch)
            print(res)


if __name__ == "__main__":
    unittest.main()
