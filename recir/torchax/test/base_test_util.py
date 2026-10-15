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

import torch
from recir.torchax import tensor
from torch.utils import _pytree as pytree

TestCase: type[unittest.TestCase] = unittest.TestCase
main: Any = unittest.main


def diff_output(
    testcase: Any,
    output1: Any,
    output2: Any,
    rtol: float,
    atol: float,
    equal_nan: bool = True,
) -> None:
    if isinstance(output1, torch.Tensor):
        testcase.assertIsInstance(output2, torch.Tensor)
        output2_cpu = output2.detach().cpu()
        if output2_cpu.dtype != output1.dtype:
            output2_cpu = output2_cpu.to(output1.dtype)
        testcase.assertTrue(
            torch.allclose(
                output1, output2_cpu, atol=atol, rtol=rtol, equal_nan=equal_nan
            )
        )
    elif isinstance(output1, (tuple, list)):
        testcase.assertIsInstance(output2, (tuple, list))
        testcase.assertEqual(len(output1), len(output2))
        for o1, o2 in zip(output1, output2, strict=False):
            diff_output(testcase, o1, o2, rtol, atol)
    else:
        testcase.assertEqual(output1, output2)


def run_function_and_compare(
    testcase: Any,
    func: Any,
    args: Any,
    kwargs: Any,
    atol: float = 1e-3,
    rtol: float = 1e-5,
    equal_nan: bool = True,
    ignore_indices: bool = False,
) -> None:
    with testcase.subTest("torch_eval"):
        res = func(*args, **kwargs)
        with testcase.subTest("torchax_eval"):
            args2, kwargs2 = pytree.tree_map_only(
                torch.Tensor,
                tensor.move_to_device,  # pyre-ignore[16]
                (args, kwargs),
            )
            res2 = func(*args2, **kwargs2)
            res2 = pytree.tree_map_only(tensor.Tensor, lambda t: t.torch(), res2)
            with testcase.subTest("torchax_diff:" + str(atol)):
                if ignore_indices and isinstance(res, tuple) and len(res) == 2:
                    diff_output(
                        testcase,
                        res[0],
                        res2[0],
                        atol=atol,
                        rtol=rtol,
                        equal_nan=equal_nan,
                    )
                else:
                    diff_output(
                        testcase, res, res2, atol=atol, rtol=rtol, equal_nan=equal_nan
                    )
