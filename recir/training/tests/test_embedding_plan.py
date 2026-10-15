# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


import unittest
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast, Optional

import torch
from recir.distributed.sharding_plan import EmbeddingShardingType
from recir.training.embedding_plan import (
    build_embedding_plan,
    DEFAULT_ROW_WISE_MIN_ROWS,
    should_row_wise_shard,
)
from torch import nn
from torchrec.distributed.types import ShardingType
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection


class _LocationType(Enum):
    HBM = "HBM"
    UVM = "UVM"


@dataclass(frozen=True)
class _Table:
    num_embeddings: int
    embedding_dim: int
    sharding: Optional[ShardingType]
    location: _LocationType


def _table(
    num_embeddings: int,
    sharding: Optional[ShardingType] = None,
    location: _LocationType = _LocationType.HBM,
) -> _Table:
    return _Table(
        num_embeddings=num_embeddings,
        embedding_dim=4,
        sharding=sharding,
        location=location,
    )


class _Sub(nn.Module):
    def __init__(self, num_embeddings: int) -> None:
        super().__init__()
        # meta avoids allocating multi-million-row tensors; the plan only reads
        # param.shape[0].
        self.weight = nn.Parameter(torch.zeros(num_embeddings, 4, device="meta"))


class _Model(nn.Module):
    """Wraps named sub-tables so FQNs are ``ec.<name>.weight``."""

    def __init__(self, sizes: dict[str, int]) -> None:
        super().__init__()
        ec = nn.Module()
        for name, n in sizes.items():
            ec.add_module(name, _Sub(n))
        self.ec = ec  # pyre-ignore


class _SharedEmbeddingContainer(nn.Module):
    def __init__(self, num_embeddings: int) -> None:
        super().__init__()
        self.primary = _Sub(num_embeddings)
        self.alias = _Sub(num_embeddings)
        self.alias.weight = self.primary.weight


class _SharedModel(nn.Module):
    def __init__(self, num_embeddings: int) -> None:
        super().__init__()
        self.ec = _SharedEmbeddingContainer(num_embeddings)


class _EmbeddingBagModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ebc = EmbeddingBagCollection(
            tables=[
                EmbeddingBagConfig(
                    name="post_id",
                    embedding_dim=4,
                    num_embeddings=8,
                    feature_names=["post_id"],
                )
            ],
            device=torch.device("meta"),
        )


class ShouldRowWiseShardTest(unittest.TestCase):
    def test_data_parallel_never_shards_even_when_large(self) -> None:
        # The 1M-row DATA_PARALLEL "city" table: config wins over size.
        self.assertFalse(
            should_row_wise_shard(1_000_000, 8, ShardingType.DATA_PARALLEL)
        )

    def test_row_wise_large_shards(self) -> None:
        self.assertTrue(should_row_wise_shard(2_000_000, 8, ShardingType.ROW_WISE))

    def test_row_wise_too_small_raises(self) -> None:
        with self.assertRaises(ValueError):
            should_row_wise_shard(4, 8, ShardingType.ROW_WISE)

    def test_row_wise_single_host_small_ok(self) -> None:
        # world_size <= 1 is always safe -> no raise.
        self.assertTrue(should_row_wise_shard(4, 1, ShardingType.ROW_WISE))

    def test_unset_large_shards(self) -> None:
        self.assertTrue(should_row_wise_shard(DEFAULT_ROW_WISE_MIN_ROWS, 8, None))

    def test_unset_small_replicates(self) -> None:
        self.assertFalse(should_row_wise_shard(500_000, 8, None))

    def test_non_row_wise_strategy_is_not_silently_replicated(self) -> None:
        with self.assertRaises(NotImplementedError):
            should_row_wise_shard(2_000_000, 8, ShardingType.TABLE_WISE)


class BuildEmbeddingPlanTest(unittest.TestCase):
    def test_matches_embedding_bag_collection_parameter_fqn(self) -> None:
        model = _EmbeddingBagModel()
        table_config = {"post_id": _table(8, ShardingType.ROW_WISE)}

        plan = build_embedding_plan(
            model,
            set(table_config),
            world_size=1,
            embedding_tables=cast(Any, table_config),
        )

        self.assertEqual(
            set(plan.sharding_map),
            {"ebc.embedding_bags.post_id.weight"},
        )

    def test_rejects_parameter_that_is_not_the_configured_global_shape(self) -> None:
        table_config = {"post_id": _table(8, ShardingType.ROW_WISE)}

        with self.assertRaisesRegex(ValueError, "unsharded table shape"):
            build_embedding_plan(
                _Model({"post_id": 4}),
                set(table_config),
                world_size=2,
                embedding_tables=cast(Any, table_config),
            )

    def test_honors_declared_sharding(self) -> None:
        tables_cfg = {
            "post_id": _table(2_000_000, ShardingType.ROW_WISE, _LocationType.UVM),
            "city": _table(1_000_000, ShardingType.DATA_PARALLEL),
            "tiny": _table(100, ShardingType.DATA_PARALLEL),
            "unset_big": _table(2_000_000, None),
            "unset_small": _table(500_000, None),
        }
        sizes = {n: t.num_embeddings for n, t in tables_cfg.items()}
        model = _Model(sizes)
        plan = build_embedding_plan(
            model, set(tables_cfg), world_size=8, embedding_tables=cast(Any, tables_cfg)
        )

        self.assertEqual(
            set(plan.sharding_map),
            {"ec.post_id.weight", "ec.unset_big.weight"},
        )
        # Regression: a 1M-row table declared DATA_PARALLEL must NOT be sharded.
        self.assertNotIn("ec.city.weight", plan.sharding_map)
        for st in plan.sharding_map.values():
            self.assertEqual(st, EmbeddingShardingType.ROW_WISE)
        # location_map carries location only for the row-wise tables.
        self.assertEqual(
            set(plan.location_map), {"ec.post_id.weight", "ec.unset_big.weight"}
        )
        self.assertEqual(
            plan.location_map["ec.post_id.weight"], _LocationType.UVM.value
        )


class BuildMixedPlanTest(unittest.TestCase):
    def test_shared_table_aliases_receive_one_table_wise_owner(self) -> None:
        model = _SharedModel(300)
        tables_cfg = {
            "primary": _table(300, ShardingType.TABLE_WISE),
            "alias": _table(300, ShardingType.TABLE_WISE),
        }

        plan = build_embedding_plan(
            model,
            set(tables_cfg),
            world_size=2,
            embedding_tables=cast(Any, tables_cfg),
        )

        self.assertEqual(
            plan.table_wise_rank_map,
            {
                "ec.primary.weight": 0,
                "ec.alias.weight": 0,
            },
        )
        self.assertEqual(
            plan.param_fqns,
            {fqn for fqn, _ in model.named_parameters(remove_duplicate=False)},
        )

    def test_shared_table_aliases_require_complete_consistent_configuration(
        self,
    ) -> None:
        model = _SharedModel(300)
        cases = (
            (
                {"primary"},
                {"primary": _table(300, ShardingType.DATA_PARALLEL)},
                "all have an embedding placement",
            ),
            (
                {"primary", "alias"},
                {
                    "primary": _table(300, ShardingType.TABLE_WISE),
                    "alias": _table(300, ShardingType.ROW_WISE),
                },
                "one sharding type",
            ),
            (
                {"primary", "alias"},
                {
                    "primary": _table(300, ShardingType.TABLE_WISE),
                    "alias": _table(
                        300,
                        ShardingType.TABLE_WISE,
                        _LocationType.UVM,
                    ),
                },
                "one storage location",
            ),
        )
        for table_names, tables_cfg, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                build_embedding_plan(
                    model,
                    table_names,
                    world_size=2,
                    embedding_tables=cast(Any, tables_cfg),
                )

    def test_resolves_table_wise_owner_and_preserves_row_wise(self) -> None:
        tables_cfg = {
            "large": _table(2_000_000, ShardingType.ROW_WISE),
            "pooled_a": _table(200, ShardingType.TABLE_WISE, _LocationType.UVM),
            "pooled_b": _table(100, ShardingType.TABLE_WISE),
            "replicated": _table(100, ShardingType.DATA_PARALLEL),
        }
        model = _Model(
            {name: table.num_embeddings for name, table in tables_cfg.items()}
        )

        plan = build_embedding_plan(
            model,
            set(tables_cfg),
            world_size=2,
            embedding_tables=cast(Any, tables_cfg),
        )

        self.assertEqual(
            plan.sharding_map,
            {
                "ec.large.weight": EmbeddingShardingType.ROW_WISE,
                "ec.pooled_a.weight": EmbeddingShardingType.TABLE_WISE,
                "ec.pooled_b.weight": EmbeddingShardingType.TABLE_WISE,
            },
        )
        self.assertEqual(
            plan.table_wise_rank_map,
            {
                "ec.pooled_a.weight": 0,
                "ec.pooled_b.weight": 1,
            },
        )
        self.assertEqual(plan.location_map["ec.pooled_a.weight"], "UVM")
        self.assertNotIn("ec.replicated.weight", plan.param_fqns)

    def test_balances_table_wise_bytes_greedily(self) -> None:
        tables_cfg = {
            "large": _table(300, ShardingType.TABLE_WISE),
            "medium": _table(200, ShardingType.TABLE_WISE),
            "small": _table(100, ShardingType.TABLE_WISE),
        }

        plan = build_embedding_plan(
            _Model({name: table.num_embeddings for name, table in tables_cfg.items()}),
            set(tables_cfg),
            world_size=2,
            embedding_tables=cast(Any, tables_cfg),
        )

        self.assertEqual(
            plan.table_wise_rank_map,
            {
                "ec.large.weight": 0,
                "ec.medium.weight": 1,
                "ec.small.weight": 1,
            },
        )

    def test_table_wise_equal_size_ties_are_deterministic(self) -> None:
        tables_cfg = {
            name: _table(100, ShardingType.TABLE_WISE)
            for name in ("zebra", "alpha", "middle")
        }

        plan = build_embedding_plan(
            _Model({name: table.num_embeddings for name, table in tables_cfg.items()}),
            set(tables_cfg),
            world_size=2,
            embedding_tables=cast(Any, tables_cfg),
        )

        self.assertEqual(
            plan.table_wise_rank_map,
            {
                "ec.alpha.weight": 0,
                "ec.middle.weight": 1,
                "ec.zebra.weight": 0,
            },
        )

    def test_rejects_unsupported_explicit_sharding(self) -> None:
        tables_cfg = {"table": _table(100, ShardingType.COLUMN_WISE)}
        with self.assertRaises(NotImplementedError):
            build_embedding_plan(
                _Model({"table": 100}),
                set(tables_cfg),
                world_size=2,
                embedding_tables=cast(Any, tables_cfg),
            )
