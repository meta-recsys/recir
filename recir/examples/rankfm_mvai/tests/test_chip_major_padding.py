# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from recir.examples.rankfm_mvai.row_padding import fit_rows_chip_major


def _row_indexed(rows: int, width: int) -> torch.Tensor:
    """Row r is filled with the value r, so a misplaced row names itself."""
    return torch.arange(rows, dtype=torch.float32).unsqueeze(1).repeat(1, width)


class FitRowsChipMajorTest(unittest.TestCase):
    def test_pads_each_chip_separately(self) -> None:
        num_local = 8
        t = _row_indexed(2048, 4)

        out = fit_rows_chip_major(t, 2056, num_local)

        self.assertEqual(tuple(out.shape), (2056, 4))
        # 2048/8 = 256 real rows per chip, 2056/8 = 257 slots per chip.
        for chip in range(num_local):
            base = chip * 257
            torch.testing.assert_close(
                out[base : base + 256],
                t[chip * 256 : (chip + 1) * 256],
            )
            self.assertTrue(
                torch.equal(out[base + 256], torch.zeros(4)),
                f"chip {chip} dummy row is not zero",
            )

    def test_dummy_rows_land_where_the_offsets_expect_them(self) -> None:
        out = fit_rows_chip_major(_row_indexed(2048, 4), 2056, 8)

        dummies = (out == 0).all(dim=1).nonzero().flatten().tolist()
        # Row 0 is legitimately all-zero (it holds the value 0), so it shows up
        # here too; every other zero row is a dummy.
        self.assertEqual(
            dummies,
            [0, 256, 513, 770, 1027, 1284, 1541, 1798, 2055],
        )

    def test_single_chip_is_a_plain_tail_pad(self) -> None:
        out = fit_rows_chip_major(_row_indexed(2048, 4), 2056, 1)

        torch.testing.assert_close(out[:2048], _row_indexed(2048, 4))
        self.assertTrue(torch.equal(out[2048:], torch.zeros(8, 4)))

    def test_exact_fit_is_returned_unchanged(self) -> None:
        t = _row_indexed(2048, 4)

        self.assertIs(fit_rows_chip_major(t, 2048, 8), t)

    def test_indivisible_shapes_fall_back_to_tail_pad(self) -> None:
        # 2050 is not a multiple of 8, so there is no chip-major layout to
        # honour; the generic path must still produce the requested shape.
        out = fit_rows_chip_major(_row_indexed(2050, 4), 2056, 8)

        self.assertEqual(tuple(out.shape), (2056, 4))
        torch.testing.assert_close(out[:2050], _row_indexed(2050, 4))
