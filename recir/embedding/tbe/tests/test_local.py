# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from unittest.mock import patch

import torch
from recir.embedding.tbe.local import LocalTBEGroup, LocalTBETable


class LocalTBEGroupTest(unittest.TestCase):
    def _make_group(
        self,
        learning_rate_tensor: torch.Tensor | None = None,
        segments_per_feature: int = 1,
        pooling_mode: str = "none",
    ) -> LocalTBEGroup:
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
                learning_rate_tensor=learning_rate_tensor,
                eps=1.0e-8,
                stochastic_rounding=False,
                segments_per_feature=segments_per_feature,
                pooling_mode=pooling_mode,
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
        group = self._make_group(segments_per_feature=4)
        table_0_indices = torch.tensor([1, 3], dtype=torch.int64)
        table_1_indices = torch.tensor([0, 2, 5], dtype=torch.int64)

        actual_0, actual_1 = group((table_0_indices, table_1_indices))
        table_0, table_1 = group.split_embedding_weights()

        torch.testing.assert_close(actual_0, table_0[table_0_indices])
        torch.testing.assert_close(actual_1, table_1[table_1_indices])

    def test_pooled_sum_preserves_empty_bags_and_updates_touched_rows(self) -> None:
        group = self._make_group(pooling_mode="sum")
        table_0_indices = torch.tensor([1, 3, 2], dtype=torch.int64)
        table_1_indices = torch.tensor([0, 2], dtype=torch.int64)
        table_0_offsets = torch.tensor([0, 2, 2, 3], dtype=torch.int64)
        table_1_offsets = torch.tensor([0, 0, 1, 2], dtype=torch.int64)

        actual_0, actual_1 = group(
            (table_0_indices, table_1_indices),
            (table_0_offsets, table_1_offsets),
        )
        table_0, table_1 = group.split_embedding_weights()
        expected_0 = torch.nn.functional.embedding_bag(
            table_0_indices,
            table_0,
            table_0_offsets,
            mode="sum",
            include_last_offset=True,
        )
        expected_1 = torch.nn.functional.embedding_bag(
            table_1_indices,
            table_1,
            table_1_offsets,
            mode="sum",
            include_last_offset=True,
        )

        torch.testing.assert_close(actual_0, expected_0)
        torch.testing.assert_close(actual_1, expected_1)
        before = [weight.clone() for weight in group.split_embedding_weights()]

        (actual_0.sum() + actual_1.sum()).backward()

        after = group.split_embedding_weights()
        self.assertFalse(torch.equal(before[0][1], after[0][1]))
        self.assertFalse(torch.equal(before[1][2], after[1][2]))
        torch.testing.assert_close(before[0][0], after[0][0])
        torch.testing.assert_close(before[1][1], after[1][1])

    def test_pooled_mean_preserves_empty_bags(self) -> None:
        group = self._make_group(pooling_mode="mean")
        feature_indices = (
            torch.tensor([1, 3, 2], dtype=torch.int64),
            torch.tensor([0, 2], dtype=torch.int64),
        )
        feature_offsets = (
            torch.tensor([0, 2, 2, 3], dtype=torch.int64),
            torch.tensor([0, 0, 1, 2], dtype=torch.int64),
        )

        actual = group(feature_indices, feature_offsets)
        weights = group.split_embedding_weights()
        expected = tuple(
            torch.nn.functional.embedding_bag(
                indices,
                weight,
                offsets,
                mode="mean",
                include_last_offset=True,
            )
            for indices, offsets, weight in zip(
                feature_indices,
                feature_offsets,
                weights,
            )
        )

        torch.testing.assert_close(actual, expected)

    def test_pooled_features_can_share_one_table(self) -> None:
        with patch(
            "fbgemm_gpu.config.feature_list.FeatureGate.is_enabled",
            return_value=False,
        ):
            group = LocalTBEGroup(
                tables=(LocalTBETable("shared", local_rows=8, embedding_dim=4),),
                feature_table_map=(0, 0),
                device=torch.device("cpu"),
                weight_dtype=torch.float32,
                learning_rate=0.1,
                eps=1.0e-8,
                stochastic_rounding=False,
                pooling_mode="sum",
            )
        weight = torch.arange(32, dtype=torch.float32).reshape(8, 4)
        group.load_table_weights((weight,))
        feature_indices = (
            torch.tensor([1, 2], dtype=torch.int64),
            torch.tensor([1, 3], dtype=torch.int64),
        )
        feature_offsets = (
            torch.tensor([0, 1, 2], dtype=torch.int64),
            torch.tensor([0, 1, 2], dtype=torch.int64),
        )

        actual = group(feature_indices, feature_offsets)

        for output, indices in zip(actual, feature_indices):
            torch.testing.assert_close(output, weight[indices])
        before = group.split_embedding_weights()[0].clone()
        torch.stack([output.sum() for output in actual]).sum().backward()
        after = group.split_embedding_weights()[0]
        for touched_row in (1, 2, 3):
            self.assertFalse(torch.equal(after[touched_row], before[touched_row]))
        torch.testing.assert_close(after[0], before[0])

    def test_pooled_group_requires_offsets(self) -> None:
        group = self._make_group(pooling_mode="sum")

        with self.assertRaisesRegex(ValueError, "requires per-feature offsets"):
            group((torch.tensor([1]), torch.tensor([2])))

    def test_pooled_group_requires_offsets_to_span_each_feature(self) -> None:
        group = self._make_group(pooling_mode="sum")
        feature_indices = (
            torch.tensor([1, 2], dtype=torch.int64),
            torch.tensor([2, 3], dtype=torch.int64),
        )
        valid_offsets = torch.tensor([0, 1, 2], dtype=torch.int64)
        cases = (
            (torch.tensor([1, 2, 2]), "start at zero"),
            (torch.tensor([0, 1, 1]), "consume every feature ID"),
        )

        for invalid_offsets, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(RuntimeError, message):
                    group(
                        feature_indices,
                        (invalid_offsets, valid_offsets),
                    )

    def test_pooled_group_rejects_unpooled_segmentation(self) -> None:
        with self.assertRaisesRegex(ValueError, "only applies to unpooled"):
            self._make_group(segments_per_feature=2, pooling_mode="sum")

    def test_pooled_group_requires_equal_bag_counts(self) -> None:
        group = self._make_group(pooling_mode="sum")

        with self.assertRaisesRegex(ValueError, "same number of bags"):
            group(
                (torch.tensor([1]), torch.tensor([2])),
                (
                    torch.tensor([0, 1, 1]),
                    torch.tensor([0, 1, 1, 1]),
                ),
            )

    def test_runtime_learning_rate_controls_fused_update(self) -> None:
        learning_rate = torch.tensor(0.1, dtype=torch.float32)
        dynamic_group = self._make_group(learning_rate)
        constant_group = self._make_group()
        indices = torch.tensor([1, 2], dtype=torch.int64)
        offsets = torch.tensor([0, 1, 2], dtype=torch.int64)

        dynamic_group.forward_flat(indices, offsets).sum().backward()
        constant_group.forward_flat(indices, offsets).sum().backward()
        dynamic_before = [
            weight.clone() for weight in dynamic_group.split_embedding_weights()
        ]
        constant_before = [
            weight.clone() for weight in constant_group.split_embedding_weights()
        ]

        learning_rate.fill_(0.2)
        dynamic_group.forward_flat(indices, offsets).sum().backward()
        constant_group.forward_flat(indices, offsets).sum().backward()

        dynamic_after = dynamic_group.split_embedding_weights()
        constant_after = constant_group.split_embedding_weights()
        torch.testing.assert_close(
            dynamic_before[0][1] - dynamic_after[0][1],
            2 * (constant_before[0][1] - constant_after[0][1]),
        )
        torch.testing.assert_close(
            dynamic_before[1][2] - dynamic_after[1][2],
            2 * (constant_before[1][2] - constant_after[1][2]),
        )
