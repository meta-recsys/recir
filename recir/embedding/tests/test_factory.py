#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Building a backend by name."""

from dataclasses import dataclass
from unittest import TestCase

from recir.embedding.backend import TableOptimizer, TablePlan, TableSharding
from recir.embedding.cpu import CPUEmbeddingBackend
from recir.embedding.factory import build_embedding_backend
from recir.embedding.in_graph import InGraphEmbeddingBackend
from recir.embedding.sparsecore import SparseCoreEmbeddingBackend


@dataclass
class _Sharding:
    value: str


@dataclass
class _Table:
    """Stands in for the model family's EmbeddingTable; only sharding is read."""

    sharding: _Sharding


class BuildEmbeddingBackendTest(TestCase):
    def test_each_name_builds_its_backend(self) -> None:
        plans = {
            "t": TablePlan(
                sharding=TableSharding.ROWWISE, optimizer=TableOptimizer.SPARSE
            )
        }
        expected = {
            "cpu": CPUEmbeddingBackend,
            "tpu": SparseCoreEmbeddingBackend,
            "in_graph": InGraphEmbeddingBackend,
        }
        for name, cls in expected.items():
            backend = build_embedding_backend(
                name, embedding_collection_fqn="owner.collection", table_plans=plans
            )
            self.assertIsInstance(backend, cls, name)

    def test_in_graph_derives_its_plans_from_the_tables(self) -> None:
        backend = build_embedding_backend(
            "in_graph",
            embedding_collection_fqn="owner.collection",
            embedding_tables={"t": _Table(_Sharding("row_wise"))},
        )

        self.assertIsInstance(backend, InGraphEmbeddingBackend)
        self.assertEqual(
            backend._table_plans["t"],
            TablePlan(sharding=TableSharding.ROWWISE, optimizer=TableOptimizer.SPARSE),
        )

    def test_in_graph_without_plans_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "table plans"):
            build_embedding_backend(
                "in_graph", embedding_collection_fqn="owner.collection"
            )

    def test_sparsecore_tables_need_the_in_graph_backend(self) -> None:
        with self.assertRaisesRegex(ValueError, "in_graph"):
            build_embedding_backend(
                "cpu",
                embedding_collection_fqn="owner.collection",
                sparsecore_tables="all",
            )

    def test_an_unknown_name_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown embedding backend"):
            build_embedding_backend("gpu", embedding_collection_fqn="owner.collection")
