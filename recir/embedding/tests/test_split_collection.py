#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Tests for splitting an EmbeddingCollection by per-table placement.

Every failure this guards against is silent. A table on the wrong side of the
split trains under a different optimizer at a different rate, with no shape
change, no key change, and no error -- which is how seven tables ran at 50x the
intended rate. So these assert on WHICH side each table lands and on the
invariants the split relies on, not just that it runs.
"""

from dataclasses import dataclass
from unittest import TestCase

import torch
from recir.embedding.split_collection import (
    assert_replicated_half_is_unplanned,
    partition_by_sharding,
    replicate_embedding_collection,
    ReplicatedEmbeddingCollection,
    split_embedding_collection,
)
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.embedding_modules import EmbeddingCollection


@dataclass
class _Plan:
    """Stands in for a torchrec ShardingPlan; only ``.plan`` keys are read."""

    plan: dict[str, object]


@dataclass
class _Table:
    """Stands in for the model family's EmbeddingTable; only sharding is read."""

    sharding: object = None


@dataclass
class _Sharding:
    value: str


def _collection(names: list[str]) -> EmbeddingCollection:
    return EmbeddingCollection(
        tables=[
            EmbeddingConfig(
                name=n, embedding_dim=4, num_embeddings=8, feature_names=[f"f_{n}"]
            )
            for n in names
        ],
        device=torch.device("cpu"),
    )


class PartitionBySharding(TestCase):
    def test_data_parallel_goes_replicated_and_the_rest_shard(self) -> None:
        sharded, replicated = partition_by_sharding(
            {
                "a": _Table(_Sharding("row_wise")),
                "b": _Table(_Sharding("data_parallel")),
                "c": _Table(_Sharding("table_wise")),
            }
        )
        self.assertEqual(sorted(sharded), ["a", "c"])
        self.assertEqual(replicated, ["b"])

    def test_a_table_with_no_placement_is_shardable(self) -> None:
        """The planner's own default, and it keeps this a no-op for models
        that never expressed a placement."""
        sharded, replicated = partition_by_sharding({"a": _Table(None)})
        self.assertEqual((sharded, replicated), (["a"], []))

    def test_a_string_placement_is_rejected_rather_than_ignored(self) -> None:
        """The artifact once round-tripped the enum's NAME instead of the member.

        That is a uniquely nasty third state: ``== ShardingType.DATA_PARALLEL``
        is False while ``is not None`` is True, so every reader sees "no
        placement" and the tables quietly go back to the sparse optimizer.
        """
        with self.assertRaises(TypeError) as caught:
            partition_by_sharding({"a": _Table("DATA_PARALLEL")})
        self.assertIn("plain string", str(caught.exception))


class SplitCollection(TestCase):
    def test_split_keeps_every_table_and_its_config(self) -> None:
        parent = torch.nn.Module()
        parent.ec = _collection(["a", "b", "c"])
        split = split_embedding_collection(parent, "ec", ["b"])
        self.assertIsNotNone(split, "asked for a split and got none")
        assert split is not None  # narrow for pyre

        sharded = {t.name for t in split.sharded.embedding_configs()}
        replicated = {t.name for t in split.replicated.embedding_configs()}
        self.assertEqual(sharded, {"a", "c"})
        self.assertEqual(replicated, {"b"})
        self.assertEqual(
            sharded | replicated,
            {"a", "b", "c"},
            "a table vanished in the split; it would silently stop training",
        )

    def test_the_replicated_half_is_a_type_the_planner_skips(self) -> None:
        """The whole mechanism rests on this being an inexact type match.

        If torchrec's sharder lookup ever accepts a subclass, those tables move
        back to the fused kernel and the sparse optimizer with no shape or
        state-dict change to show for it.
        """
        parent = torch.nn.Module()
        parent.ec = _collection(["a", "b"])
        split = split_embedding_collection(parent, "ec", ["b"])
        self.assertIsNotNone(split, "asked for a split and got none")
        assert split is not None  # narrow for pyre

        self.assertIsInstance(split.replicated, ReplicatedEmbeddingCollection)
        self.assertIsNot(
            type(split.replicated),
            EmbeddingCollection,
            "the replicated half must NOT be an exact EmbeddingCollection",
        )
        assert_replicated_half_is_unplanned(_Plan({}), split, "ec")

    def test_a_planned_replicated_half_is_caught(self) -> None:
        """If the planner ever reaches those tables they are back on the fused
        kernel and the sparse optimizer, with nothing else to show for it."""
        parent = torch.nn.Module()
        parent.ec = _collection(["a", "b"])
        split = split_embedding_collection(parent, "ec", ["b"])
        self.assertIsNotNone(split, "asked for a split and got none")
        assert split is not None  # narrow for pyre
        with self.assertRaises(RuntimeError) as caught:
            assert_replicated_half_is_unplanned(
                _Plan({"ec.replicated": object()}), split, "ec"
            )
        self.assertIn("must stay", str(caught.exception))

    def test_a_plan_that_touches_only_the_sharded_half_is_fine(self) -> None:
        parent = torch.nn.Module()
        parent.ec = _collection(["a", "b"])
        split = split_embedding_collection(parent, "ec", ["b"])
        self.assertIsNotNone(split, "asked for a split and got none")
        assert split is not None  # narrow for pyre
        assert_replicated_half_is_unplanned(
            _Plan({"ec.sharded": object()}), split, "ec"
        )


class ReplicateCollection(TestCase):
    def test_retypes_in_place_and_keeps_the_live_tables(self) -> None:
        parent = torch.nn.Module()
        ec = _collection(["a", "b"])
        parent.ec = ec
        tables = dict(ec.embeddings.items())

        replicated = replicate_embedding_collection(parent, "ec")

        self.assertIs(parent.ec, replicated)
        self.assertIs(type(replicated), ReplicatedEmbeddingCollection)
        for name, table in tables.items():
            self.assertIs(
                replicated.embeddings[name],
                table,
                "the table was copied rather than moved; its weights would fork",
            )

    def test_rejects_a_non_collection(self) -> None:
        parent = torch.nn.Module()
        parent.ec = torch.nn.Linear(2, 2)
        with self.assertRaises(TypeError):
            replicate_embedding_collection(parent, "ec")
