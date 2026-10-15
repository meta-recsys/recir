# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from collections.abc import Sequence
from typing import Any
from unittest.mock import MagicMock

import torch
import torch.nn as nn
from recir.api import _make_hybrid_sparse_transform, _prepare_transforms_after_shard
from recir.distributed.parallel_mode import HybridParallel
from recir.distributed.sharding_plan import EmbeddingShardingPlan, EmbeddingShardingType
from recir.region_transforms import RegionTransform, SparseEmbeddingTransform
from torch.distributed.tensor import DeviceMesh


class _RecordingTransform(RegionTransform):
    def __init__(self) -> None:
        self.prepared_module: nn.Module | None = None
        self.prepared_optimizers: tuple[torch.optim.Optimizer, ...] = ()

    def prepare_after_shard(
        self,
        module: nn.Module,
        optimizers: Sequence[torch.optim.Optimizer],
    ) -> None:
        self.prepared_module = module
        self.prepared_optimizers = tuple(optimizers)


class SparseTransformLifecycleTest(unittest.TestCase):
    def test_prepare_after_shard_receives_module_and_optimizers(self) -> None:
        transform = _RecordingTransform()
        model = nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, foreach=True)

        _prepare_transforms_after_shard((transform,), model, (optimizer,))

        self.assertIs(transform.prepared_module, model)
        self.assertEqual(transform.prepared_optimizers, (optimizer,))

    def test_hybrid_parallel_uses_default_sparse_transform(self) -> None:
        model = nn.ModuleDict({"embedding": nn.Embedding(4, 2)})
        mesh = MagicMock(spec=DeviceMesh)
        mesh.size.return_value = 2
        mesh._dim_group_names = ["fake_test_group"]
        mode = HybridParallel(
            embedding_plan=EmbeddingShardingPlan(
                sharding_map={"embedding.weight": EmbeddingShardingType.ROW_WISE}
            )
        )
        mode._mesh = mesh

        transform = _make_hybrid_sparse_transform(mode, {"embedding.weight"}, model)

        self.assertIsInstance(transform, SparseEmbeddingTransform)

    def test_hybrid_parallel_uses_sparse_transform_factory(self) -> None:
        model = nn.Linear(4, 2)
        mesh = MagicMock(spec=DeviceMesh)
        mesh.size.return_value = 2
        mesh._dim_group_names = ["fake_test_group"]
        expected = _RecordingTransform()
        calls: list[tuple[set[str], DeviceMesh, nn.Module]] = []

        def factory(fqns: set[str], device_mesh: DeviceMesh, module: nn.Module) -> Any:
            calls.append((fqns, device_mesh, module))
            return expected

        mode = HybridParallel(
            embedding_plan=EmbeddingShardingPlan(
                sharding_map={"embedding.weight": EmbeddingShardingType.ROW_WISE}
            ),
            sparse_transform_factory=factory,
        )
        mode._mesh = mesh

        actual = _make_hybrid_sparse_transform(mode, {"embedding.weight"}, model)

        self.assertIs(actual, expected)
        self.assertEqual(calls, [({"embedding.weight"}, mesh, model)])

    def test_sparse_transform_factory_rejects_wrong_type(self) -> None:
        model = nn.Linear(4, 2)
        mesh = MagicMock(spec=DeviceMesh)
        mesh.size.return_value = 2
        mesh._dim_group_names = ["fake_test_group"]
        mode = HybridParallel(
            embedding_plan=EmbeddingShardingPlan(
                sharding_map={"embedding.weight": EmbeddingShardingType.ROW_WISE}
            ),
            sparse_transform_factory=lambda _fqns, _mesh, _module: object(),
        )
        mode._mesh = mesh

        with self.assertRaisesRegex(TypeError, "must return a RegionTransform"):
            _make_hybrid_sparse_transform(mode, {"embedding.weight"}, model)
