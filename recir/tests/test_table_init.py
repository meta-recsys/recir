# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import List, Sequence, Tuple
from unittest.mock import patch

import torch
from recir.utils import meta_tensor_init
from recir.utils.meta_tensor_init import fill_table_shard

_ROWS, _DIM = 10, 6
_BOUND = 0.5


def _assemble(
    row_splits: Sequence[Tuple[int, int]], name: str = "t", dim: int = _DIM
) -> torch.Tensor:
    """The table built from shards covering ``row_splits``, as separate ranks would."""
    shards: List[torch.Tensor] = []
    for start, end in row_splits:
        shard = torch.zeros(end - start, dim)
        fill_table_shard(shard, name, (_ROWS, dim), (start, 0), -_BOUND, _BOUND)
        shards.append(shard)
    return torch.cat(shards)


# Blocks of 4 rows put block boundaries at rows 4 and 8, inside the shards below.
@patch.object(meta_tensor_init, "_INIT_BLOCK_ROWS", 4)
class FillTableShardTest(unittest.TestCase):
    def test_any_row_split_assembles_the_same_table(self) -> None:
        whole = _assemble([(0, _ROWS)])
        for splits in (
            [(0, 5), (5, 10)],
            [(0, 3), (3, 7), (7, 10)],
            [(0, 4), (4, 8), (8, 10)],
            [(row, row + 1) for row in range(_ROWS)],
        ):
            self.assertTrue(torch.equal(_assemble(splits), whole), splits)

    def test_shards_are_distinct(self) -> None:
        halves = _assemble([(0, 5), (5, 10)])
        self.assertFalse(torch.equal(halves[:5], halves[5:]))

    def test_column_shards_assemble_the_same_table(self) -> None:
        whole = _assemble([(0, _ROWS)])
        left, right = torch.zeros(_ROWS, 2), torch.zeros(_ROWS, _DIM - 2)
        fill_table_shard(left, "t", (_ROWS, _DIM), (0, 0), -_BOUND, _BOUND)
        fill_table_shard(right, "t", (_ROWS, _DIM), (0, 2), -_BOUND, _BOUND)
        self.assertTrue(torch.equal(torch.cat([left, right], dim=1), whole))

    def test_tables_differ_by_name(self) -> None:
        self.assertFalse(
            torch.equal(_assemble([(0, _ROWS)], "a"), _assemble([(0, _ROWS)], "b"))
        )

    def test_values_stay_within_the_bounds(self) -> None:
        whole = _assemble([(0, _ROWS)])
        self.assertLessEqual(float(whole.abs().max()), _BOUND)

    def test_fills_lower_precision_shards(self) -> None:
        shard = torch.zeros(3, _DIM, dtype=torch.float16)
        fill_table_shard(shard, "t", (_ROWS, _DIM), (2, 0), -_BOUND, _BOUND)
        expected = _assemble([(0, _ROWS)])[2:5].to(torch.float16)
        self.assertTrue(torch.equal(shard, expected))

    def test_empty_shard_is_left_alone(self) -> None:
        shard = torch.zeros(0, _DIM)
        self.assertIs(
            fill_table_shard(shard, "t", (_ROWS, _DIM), (5, 0), -_BOUND, _BOUND),
            shard,
        )
