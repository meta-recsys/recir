# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import torch
from recir.lowering import fbgemm_fallback
from recir.lowering.fbgemm_fallback import (
    _asynchronous_complete_cumsum_impl,
    _batched_complete_cumsum_impl,
)


class CompleteCumsumTest(unittest.TestCase):
    def test_int64_lengths_give_int64_offsets(self) -> None:
        lengths = torch.randint(0, 1000, (32896,), dtype=torch.int64)
        out = _asynchronous_complete_cumsum_impl(lengths)
        self.assertEqual(out.dtype, torch.int64)
        self.assertTrue(
            torch.equal(
                out, torch.cat([torch.zeros(1, dtype=torch.int64), lengths.cumsum(0)])
            )
        )

    def test_batched_int64_lengths_give_int64_offsets(self) -> None:
        lengths = torch.randint(0, 1000, (9, 32896), dtype=torch.int64)
        out = _batched_complete_cumsum_impl(lengths)
        self.assertEqual(out.dtype, torch.int64)
        self.assertTrue(
            torch.equal(
                out,
                torch.cat([torch.zeros(9, 1, dtype=torch.int64), lengths.cumsum(1)], 1),
            )
        )

    def test_float_values_keep_their_dtype(self) -> None:
        values = torch.rand(3, 5)
        out = _batched_complete_cumsum_impl(values)
        self.assertEqual(out.dtype, torch.float32)
        torch.testing.assert_close(out[:, 1:], values.cumsum(1))

    def test_a_total_past_int32_wraps_rather_than_raising(self) -> None:
        """The int32 accumulator is what makes this fast, and it is only safe
        while the total stays under 2**31. Past it torch wraps silently, so
        the caller reads rows from the wrong end of the values tensor -- there
        is no error to catch. This pins that the bound is real, so the comment
        on `_cumsum_lengths` is not just an assertion.
        """
        lengths = torch.tensor([2**30, 2**30, 2**30], dtype=torch.int64)
        out = fbgemm_fallback._cumsum_lengths(lengths, dim=0)

        self.assertEqual(torch.int64, out.dtype)
        self.assertTrue(
            (out < 0).any(),
            "a total past 2**31 no longer wraps; the int32 accumulator may "
            "have been widened, and the bound in the comment is stale",
        )
