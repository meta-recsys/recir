# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from unittest.mock import patch

import torch
from recir.tbe.local import LocalTBEGroup, LocalTBETable


class LocalTBEGroupTest(unittest.TestCase):
    def _make_group(self) -> LocalTBEGroup:
        with patch(
            "fbgemm_gpu.config.feature_list.FeatureGate.is_enabled",
            return_value=False,
        ):
            group = LocalTBEGroup(
                tables=(
                    LocalTBETable("table_0", local_rows=8, embedding_dim=4),
                    LocalTBETable("table_1", local_rows=6, embedding_dim=4),
                ),
                feature_table_map=(0, 1),
                device=torch.device("cpu"),
                weight_dtype=torch.float32,
                learning_rate=0.1,
                eps=1.0e-8,
                stochastic_rounding=False,
            )
        group.load_table_weights(
            (
                torch.arange(32, dtype=torch.float32).reshape(8, 4),
                torch.arange(24, dtype=torch.float32).reshape(6, 4) + 100,
            )
        )
        return group

    def test_unpooled_forward_matches_local_embedding(self) -> None:
        group = self._make_group()
        indices = torch.tensor([1, 3, 0, 2, 5], dtype=torch.int64)
        offsets = torch.tensor([0, 2, 5], dtype=torch.int64)

        actual = group.forward_flat(indices, offsets)
        table_0, table_1 = group.split_embedding_weights()
        expected = torch.cat((table_0[indices[:2]], table_1[indices[2:]]), dim=0)

        torch.testing.assert_close(actual, expected)

    def test_backward_updates_only_touched_rows(self) -> None:
        group = self._make_group()
        before = [weight.clone() for weight in group.split_embedding_weights()]
        indices = torch.tensor([1, 1, 3, 2], dtype=torch.int64)
        offsets = torch.tensor([0, 3, 4], dtype=torch.int64)

        group.forward_flat(indices, offsets).sum().backward()

        after = group.split_embedding_weights()
        self.assertFalse(torch.equal(before[0][1], after[0][1]))
        self.assertFalse(torch.equal(before[0][3], after[0][3]))
        self.assertFalse(torch.equal(before[1][2], after[1][2]))
        torch.testing.assert_close(before[0][0], after[0][0])
        torch.testing.assert_close(before[1][0], after[1][0])

    def test_features_share_one_grouped_tbe_call(self) -> None:
        group = self._make_group()
        table_0_indices = torch.tensor([1, 3], dtype=torch.int64)
        table_1_indices = torch.tensor([0, 2, 5], dtype=torch.int64)

        actual_0, actual_1 = group((table_0_indices, table_1_indices))
        table_0, table_1 = group.split_embedding_weights()

        torch.testing.assert_close(actual_0, table_0[table_0_indices])
        torch.testing.assert_close(actual_1, table_1[table_1_indices])
