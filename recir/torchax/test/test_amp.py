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

import jax
import jax.numpy as jnp
import recir.torchax
import torch
from recir.torchax import interop


class AutocastTest(unittest.TestCase):
    def setUp(self) -> None:
        self.env = recir.torchax.default_env()

    def test_auto_cast_ir(self) -> None:
        with self.env:
            with recir.torchax.amp.autocast("jax", dtype=torch.bfloat16, env=self.env):
                a = jax.ShapeDtypeStruct((2, 2), jnp.float32)
                b = jax.ShapeDtypeStruct((2, 2), jnp.float32)
                ir_text = jax.jit(interop.jax_view(torch.matmul)).lower(a, b).as_text()
                self.assertIn("tensor<2x2xbf16>", ir_text)

    def test_auto_cast_matmul(self) -> None:
        with self.env:
            a = torch.randn(2, 2, device="jax")
            b = torch.randn(2, 2, device="jax")
            with recir.torchax.amp.autocast("jax", dtype=torch.bfloat16, env=self.env):
                c = a @ b

            self.assertEqual(c.dtype, torch.bfloat16)

            with torch.autocast("cpu", dtype=torch.bfloat16):
                c_cpu = a.cpu() @ b.cpu()

            self.assertTrue(torch.allclose(c.cpu(), c_cpu))

    def test_auto_cast_cat_stack_promote(self) -> None:
        # cat/stack use the PROMOTE policy and take a LIST of tensors as their
        # first arg. Regression for "'list' object has no attribute 'dtype'":
        # execute_policy must flatten args instead of assuming each is a tensor.
        with self.env:
            a = torch.randn(2, 2, device="jax")
            b = torch.randn(2, 2, device="jax")
            with recir.torchax.amp.autocast("jax", dtype=torch.bfloat16, env=self.env):
                cat = torch.cat([a, b], dim=0)
                stk = torch.stack([a, b], dim=0)
            # PROMOTE keeps the widest float dtype among inputs (both fp32) -> fp32.
            self.assertEqual(cat.dtype, torch.float32)
            self.assertEqual(stk.dtype, torch.float32)

    def test_auto_cast_cat_mixed_dtype_promotes_to_widest(self) -> None:
        with self.env:
            a = torch.randn(2, 2, device="jax")
            b = torch.randn(2, 2, device="jax").to(torch.bfloat16)
            with recir.torchax.amp.autocast("jax", dtype=torch.bfloat16, env=self.env):
                cat = torch.cat([a, b], dim=0)
            # Widest of {fp32, bf16} is fp32, so the bf16 input is up-cast.
            self.assertEqual(cat.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
