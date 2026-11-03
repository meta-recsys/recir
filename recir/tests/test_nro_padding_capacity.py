# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Padding below the per-request bound must preserve every real candidate."""

from unittest import TestCase
from unittest.mock import patch

import torch
from recir.region_transforms.padding import JaggedPairConfig, PaddingRegionTransform


def _pad_inputs(
    ro_float: torch.Tensor,
    nro_float: torch.Tensor,
    unary_values: torch.Tensor,
    unary_lengths: torch.Tensor,
    num_candidates: torch.Tensor,
) -> torch.Tensor:
    return ro_float


def _transform() -> PaddingRegionTransform:
    transform = PaddingRegionTransform(
        jagged_pairs={
            "unary_values": JaggedPairConfig(
                "unary_lengths",
                per_sample_budgets=1,
                batch_size_arg="nro_float",
                count_arg="num_candidates",
                max_per_sample_count=6,
            ),
        },
        batch_size_arg="ro_float",
        num_devices=2,
    )
    transform.bind_signature(_pad_inputs)
    return transform


def _inputs(counts: torch.Tensor) -> tuple[torch.Tensor, ...]:
    candidates = int(counts.sum().item())
    return (
        torch.arange(counts.shape[0]).reshape(-1, 1),
        torch.arange(candidates).reshape(-1, 1),
        torch.arange(candidates),
        torch.ones(candidates, dtype=torch.int64),
        counts,
    )


class TestNroCandidatePadding(TestCase):
    @patch("recir.region_transforms.padding.runtime_num_local", return_value=2)
    def test_transform_preserves_all_real_candidate_rows(self, _mock: object) -> None:
        counts = torch.tensor([20, 1, 1, 1, 1, 1, 1, 1])
        original = _inputs(counts)
        padded, _ = _transform().transform_inputs(original, {})
        torch.testing.assert_close(padded[1][:23], original[1][:23], rtol=0, atol=0)
        torch.testing.assert_close(padded[1][30:34], original[1][23:], rtol=0, atol=0)
        torch.testing.assert_close(padded[4][:4], counts[:4], rtol=0, atol=0)
        torch.testing.assert_close(padded[4][5:9], counts[4:], rtol=0, atol=0)
        self.assertEqual(padded[1].shape[0], 60)
