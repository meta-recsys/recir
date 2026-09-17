# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.distributed as dist
import torch.nn.functional as F
from recir.region_transforms.sparse_embedding import (
    decomposed_row_wise_embedding,
    decomposed_row_wise_embedding_backward,
)
from torch.distributed import _functional_collectives as funcol
from torchrec.distributed.test_utils.multi_process import MultiProcessTestBase


_NUM_EMBEDDINGS: int = 7
_EMBEDDING_DIM: int = 3


def _verify_unpooled(
    *,
    rank: int,
    world_size: int,
    group_name: str,
    full_weight: torch.Tensor,
    local_weight: torch.Tensor,
) -> None:
    indices = torch.tensor([3, 4, 6, 0], dtype=torch.int64)
    actual = decomposed_row_wise_embedding(
        local_weight,
        indices,
        _NUM_EMBEDDINGS,
        group_name,
        world_size,
    )
    torch.testing.assert_close(actual, F.embedding(indices, full_weight))

    grad_output = torch.arange(
        indices.shape[0] * _EMBEDDING_DIM,
        dtype=torch.float32,
    ).reshape(indices.shape[0], _EMBEDDING_DIM)
    actual_grad = decomposed_row_wise_embedding_backward(
        grad_output,
        indices,
        _NUM_EMBEDDINGS,
        local_weight.shape[0],
        group_name,
        world_size,
        padding_idx=-1,
        scale_grad_by_freq=False,
    )
    expected_grad = (
        torch.ops.aten.embedding_dense_backward.default(
            grad_output,
            indices,
            _NUM_EMBEDDINGS,
            -1,
            False,
        )
        * world_size
    )
    block_rows = (_NUM_EMBEDDINGS + world_size - 1) // world_size
    row_start = rank * block_rows
    row_end = row_start + local_weight.shape[0]
    torch.testing.assert_close(actual_grad, expected_grad[row_start:row_end])


class SparseEmbeddingRoutingTest(MultiProcessTestBase):
    @classmethod
    def _run_uneven_row_shards(cls, rank: int, world_size: int) -> None:
        dist.init_process_group(rank=rank, world_size=world_size, backend="gloo")
        try:
            full_weight = torch.arange(
                _NUM_EMBEDDINGS * _EMBEDDING_DIM,
                dtype=torch.float32,
            ).reshape(_NUM_EMBEDDINGS, _EMBEDDING_DIM)
            block_rows = (_NUM_EMBEDDINGS + world_size - 1) // world_size
            row_start = rank * block_rows
            row_end = min(row_start + block_rows, _NUM_EMBEDDINGS)
            local_weight = full_weight[row_start:row_end]
            process_group = dist.group.WORLD
            assert process_group is not None
            group_name = funcol._resolve_group_name(process_group)
            _verify_unpooled(
                rank=rank,
                world_size=world_size,
                group_name=group_name,
                full_weight=full_weight,
                local_weight=local_weight,
            )
        finally:
            dist.destroy_process_group()

    @unittest.skipIf(not dist.is_available(), "torch.distributed is unavailable")
    def test_ec_routes_uneven_row_shards(self) -> None:
        self._run_multi_process_test(
            callable=self._run_uneven_row_shards,
            world_size=2,
        )
