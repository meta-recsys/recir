# pyre-unsafe
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Set


class EmbeddingShardingType(Enum):
    """Sharding strategy for an embedding table's weight parameter.

    ROW_WISE: partition the embedding table along dim 0 (vocabulary).
        Each rank owns V/N rows. At runtime, input indices are bucketized
        and routed via AllToAll to the rank that owns each row.
    """

    ROW_WISE = "row_wise"


@dataclass
class EmbeddingShardingPlan:
    """Maps embedding parameter FQNs to their sharding strategy.

    Example::

        plan = EmbeddingShardingPlan(sharding_map={
            "emb.weight": EmbeddingShardingType.ROW_WISE,
            "pos_emb.weight": EmbeddingShardingType.ROW_WISE,
        })

    The FQN format matches ``model.named_parameters()`` output
    (e.g., ``"layers.0.embedding.weight"``).
    """

    sharding_map: Dict[str, EmbeddingShardingType] = field(default_factory=dict)

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
