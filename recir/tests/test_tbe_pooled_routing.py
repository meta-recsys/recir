# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
import torch.distributed as dist
import torch.nn.functional as F
from recir.region_transforms.tbe_sparse_embedding import (
    return_row_wise_pooled_embeddings_grouped,
    route_row_wise_pooled_gradients_grouped,
    route_row_wise_pooled_indices_grouped,
)
from torch.distributed import _functional_collectives as funcol
from torchrec.distributed.test_utils.multi_process import MultiProcessTestBase


_VARIABLE_BATCH_SIZES: tuple[int, int] = (2, 3)
_UNIFORM_BATCH_SIZES: tuple[int, int] = (2, 2)
_FEATURE_ROWS: tuple[int, int] = (9, 7)
_EMBEDDING_DIM: int = 2


def _rank_features(
    rank: int,
    variable_batch_per_rank: bool,
) -> tuple[tuple[torch.Tensor, ...], ...]:
    if rank == 0:
        return (
            (
                torch.tensor([0, 5, 8], dtype=torch.int64),
                torch.tensor([1, 2], dtype=torch.int64),
            ),
            (
                torch.tensor([1, 4], dtype=torch.int64),
                torch.tensor([2, 0], dtype=torch.int64),
            ),
        )
    variable_features = (
        (
            torch.tensor([3, 6], dtype=torch.int64),
            torch.tensor([0, 1, 1], dtype=torch.int64),
        ),
        (
            torch.tensor([0, 3, 6], dtype=torch.int64),
            torch.tensor([1, 1, 1], dtype=torch.int64),
        ),
    )
    if variable_batch_per_rank:
        return variable_features
    return (
        (variable_features[0][0][:1], variable_features[0][1][:2]),
        (variable_features[1][0][:2], variable_features[1][1][:2]),
    )


def _offsets(lengths: torch.Tensor) -> torch.Tensor:
    return torch.cat((lengths.new_zeros((1,)), lengths)).cumsum(dim=0)


def _global_weights() -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.arange(
            _FEATURE_ROWS[0] * _EMBEDDING_DIM,
            dtype=torch.float32,
        ).reshape(_FEATURE_ROWS[0], _EMBEDDING_DIM),
        torch.arange(
            _FEATURE_ROWS[1] * _EMBEDDING_DIM,
            dtype=torch.float32,
        ).reshape(_FEATURE_ROWS[1], _EMBEDDING_DIM)
        + 100,
    )


class TBEPooledRoutingTest(MultiProcessTestBase):
    @classmethod
    def _run_pooled_routing(
        cls,
        rank: int,
        world_size: int,
        variable_batch_per_rank: bool,
    ) -> None:
        dist.init_process_group(rank=rank, world_size=world_size, backend="gloo")
        try:
            batch_sizes = (
                _VARIABLE_BATCH_SIZES
                if variable_batch_per_rank
                else _UNIFORM_BATCH_SIZES
            )
            features = _rank_features(rank, variable_batch_per_rank)
            feature_count = len(features)
            packed_lengths = torch.cat(tuple(lengths for _indices, lengths in features))
            packed_indices = torch.cat(tuple(indices for indices, _lengths in features))
            packed_offsets = _offsets(packed_lengths)
            process_group = dist.group.WORLD
            assert process_group is not None
            group_name = funcol._resolve_group_name(process_group)

            (received_indices, local_offsets, cpu_count_splits), _, _ = (
                route_row_wise_pooled_indices_grouped(
                    packed_indices,
                    packed_offsets,
                    num_embeddings=_FEATURE_ROWS,
                    group_name=group_name,
                    world_size=world_size,
                    variable_batch_per_rank=variable_batch_per_rank,
                )
            )

            total_batch_size = sum(batch_sizes)
            feature_boundary = int(local_offsets[total_batch_size].item())
            global_weights = _global_weights()
            local_pooled = []
            for feature_index, rows in enumerate(_FEATURE_ROWS):
                block_size = (rows + world_size - 1) // world_size
                local_weight = global_weights[feature_index][
                    rank * block_size : min((rank + 1) * block_size, rows)
                ]
                value_start = 0 if feature_index == 0 else feature_boundary
                value_end = (
                    feature_boundary
                    if feature_index == 0
                    else received_indices.shape[0]
                )
                feature_offsets = (
                    local_offsets[
                        feature_index * total_batch_size : (feature_index + 1)
                        * total_batch_size
                        + 1
                    ]
                    - value_start
                )
                local_pooled.append(
                    F.embedding_bag(
                        received_indices[value_start:value_end],
                        local_weight,
                        feature_offsets,
                        mode="sum",
                        include_last_offset=True,
                    )
                )
            local_embeddings = torch.cat(local_pooled, dim=1)
            actual = return_row_wise_pooled_embeddings_grouped(
                local_embeddings,
                packed_offsets,
                cpu_count_splits,
                feature_count=feature_count,
                group_name=group_name,
                world_size=world_size,
                variable_batch_per_rank=variable_batch_per_rank,
            )
            expected_sum = torch.cat(
                tuple(
                    F.embedding_bag(
                        indices,
                        weight,
                        _offsets(lengths),
                        mode="sum",
                        include_last_offset=True,
                    )
                    for (indices, lengths), weight in zip(features, global_weights)
                ),
                dim=1,
            )
            torch.testing.assert_close(actual, expected_sum)

            actual_mean = torch.cat(
                tuple(
                    actual[
                        :,
                        feature_index * _EMBEDDING_DIM : (feature_index + 1)
                        * _EMBEDDING_DIM,
                    ]
                    / lengths.clamp_min(1).to(actual.dtype).unsqueeze(1)
                    for feature_index, (_indices, lengths) in enumerate(features)
                ),
                dim=1,
            )
            expected_mean = torch.cat(
                tuple(
                    F.embedding_bag(
                        indices,
                        weight,
                        _offsets(lengths),
                        mode="mean",
                        include_last_offset=True,
                    )
                    for (indices, lengths), weight in zip(features, global_weights)
                ),
                dim=1,
            )
            torch.testing.assert_close(actual_mean, expected_mean)

            source_gradients = (
                torch.arange(
                    batch_sizes[rank] * feature_count * _EMBEDDING_DIM,
                    dtype=torch.float32,
                ).reshape(batch_sizes[rank], -1)
                + rank * 100
            )
            actual_gradients = route_row_wise_pooled_gradients_grouped(
                source_gradients,
                local_embeddings,
                cpu_count_splits,
                feature_count=feature_count,
                group_name=group_name,
                world_size=world_size,
                variable_batch_per_rank=variable_batch_per_rank,
            )
            expected_gradients = torch.cat(
                tuple(
                    torch.arange(
                        batch_size * feature_count * _EMBEDDING_DIM,
                        dtype=torch.float32,
                    ).reshape(batch_size, -1)
                    + source_rank * 100
                    for source_rank, batch_size in enumerate(batch_sizes)
                )
            )
            torch.testing.assert_close(actual_gradients, expected_gradients)
        finally:
            dist.destroy_process_group()

    @classmethod
    def _run_uniform_batch_routing(
        cls,
        rank: int,
        world_size: int,
    ) -> None:
        cls._run_pooled_routing(rank, world_size, False)

    @classmethod
    def _run_variable_batch_routing(
        cls,
        rank: int,
        world_size: int,
    ) -> None:
        cls._run_pooled_routing(rank, world_size, True)

    @unittest.skipIf(not dist.is_available(), "torch.distributed is unavailable")
    def test_sum_and_mean_pooling_route_uniform_rank_batches(self) -> None:
        self._run_multi_process_test(
            callable=self._run_uniform_batch_routing,
            world_size=2,
        )

    @unittest.skipIf(not dist.is_available(), "torch.distributed is unavailable")
    def test_sum_and_mean_pooling_route_uneven_rank_batches(self) -> None:
        self._run_multi_process_test(
            callable=self._run_variable_batch_routing,
            world_size=2,
        )
