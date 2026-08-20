# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import recir.torchax
import torch


class OutVariantTest(unittest.TestCase):
    def setUp(self) -> None:
        self.env = recir.torchax.default_env()
        self.env.config.debug_print_each_op = True

    def test_add_out(self) -> None:
        with self.env:
            a = torch.ones((2, 2), device="jax")
            b = torch.ones((2, 2), device="jax")
            c = torch.ones((2, 2), device="jax")

            torch.add(a, b, out=c)
            torch.testing.assert_close(c, a + b, check_device=False)

    def test_sub_out(self) -> None:
        with self.env:
            a = torch.ones((2, 2), device="jax")
            b = torch.ones((2, 2), device="jax")
            c = torch.ones((2, 2), device="jax")

            torch.sub(a, b, out=c)
            torch.testing.assert_close(c, a - b, check_device=False)

    def test_mul_out(self) -> None:
        with self.env:
            a = torch.ones((2, 2), device="jax")
            b = torch.ones((2, 2), device="jax")
            c = torch.ones((2, 2), device="jax")

            torch.mul(a, b, out=c)
            torch.testing.assert_close(c, a * b, check_device=False)

    def test_div_out(self) -> None:
        with self.env:
            a = torch.ones((2, 2), device="jax")
            b = torch.ones((2, 2), device="jax")
            c = torch.ones((2, 2), device="jax")

            torch.div(a, b, out=c)
            torch.testing.assert_close(c, a / b, check_device=False)
