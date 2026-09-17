# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Set


class EmbeddingShardingType(Enum):
    """Sharding strategy for an embedding table's weight parameter.

    ROW_WISE: partition the embedding table along dim 0 (vocabulary).
        Each rank owns V/N rows. At runtime, input indices are bucketized
        and routed via AllToAll to the rank that owns each row.
    TABLE_WISE: place the complete table on one owner rank. At runtime, sparse
        features are routed to the owner and pooled outputs are returned to the
        source rank.
    """

    ROW_WISE = "row_wise"
    TABLE_WISE = "table_wise"


@dataclass
class EmbeddingShardingPlan:
    """Maps embedding parameter FQNs to their sharding strategy.

    Example::

        plan = EmbeddingShardingPlan(
            sharding_map={
                "large.weight": EmbeddingShardingType.ROW_WISE,
                "small.weight": EmbeddingShardingType.TABLE_WISE,
            },
            table_wise_rank_map={"small.weight": 1},
        )

    The FQN format matches ``model.named_parameters()`` output
    (e.g., ``"layers.0.embedding.weight"``).
    """

    sharding_map: Dict[str, EmbeddingShardingType] = field(default_factory=dict)
    location_map: Dict[str, str] = field(default_factory=dict)
    table_wise_rank_map: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unexpected_locations = sorted(set(self.location_map) - set(self.sharding_map))
        if unexpected_locations:
            raise ValueError(
                "Embedding storage locations must reference sharded parameters: "
                f"unexpected={unexpected_locations}."
            )
        table_wise_fqns = {
            fqn
            for fqn, sharding_type in self.sharding_map.items()
            if sharding_type == EmbeddingShardingType.TABLE_WISE
        }
        owner_fqns = set(self.table_wise_rank_map)
        if owner_fqns != table_wise_fqns:
            missing = sorted(table_wise_fqns - owner_fqns)
            unexpected = sorted(owner_fqns - table_wise_fqns)
            raise ValueError(
                "Table-wise placements must have exactly one owner rank: "
                f"missing={missing}, unexpected={unexpected}."
            )
        invalid_owners = {
            fqn: rank for fqn, rank in self.table_wise_rank_map.items() if rank < 0
        }
        if invalid_owners:
            raise ValueError(
                f"Table-wise owner ranks must be non-negative: {invalid_owners}."
            )

    @property
    def param_fqns(self) -> Set[str]:
        """All parameter FQNs covered by this plan."""
        return set(self.sharding_map.keys())

    def is_embedding_param(self, fqn: str) -> bool:
        """Return True if *fqn* is an embedding param in this plan."""
        return fqn in self.sharding_map

    def is_row_wise(self, fqn: str) -> bool:
        """Return True if *fqn* is assigned row-wise sharding."""
        return self.sharding_map.get(fqn) == EmbeddingShardingType.ROW_WISE

    def is_table_wise(self, fqn: str) -> bool:
        """Return True if *fqn* is assigned table-wise sharding."""
        return self.sharding_map.get(fqn) == EmbeddingShardingType.TABLE_WISE

    def table_wise_owner(self, fqn: str) -> int:
        """Return the owner rank for a table-wise embedding parameter."""
        if not self.is_table_wise(fqn):
            raise ValueError(f"Embedding parameter {fqn!r} is not table-wise.")
        return self.table_wise_rank_map[fqn]
