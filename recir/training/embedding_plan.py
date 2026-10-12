# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Embedding sharding plan for recir.

Builds an ``EmbeddingShardingPlan`` from the model's parameter FQNs and the
config-declared ``EmbeddingTable`` entries. The per-table
``EmbeddingTable.sharding`` field is authoritative. Row-wise tables are split
across every rank, table-wise tables are assigned whole to one rank, and
data-parallel tables are replicated (excluded from the plan).

The plan's FQN set is what downstream optimizer setup reads to split params into
sparse vs dense (see ``setup._partition_params``).

Framework-agnostic: torch + recir.distributed.sharding_plan + the EmbeddingTable config.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional, Protocol, Set

import torch
from recir.distributed.sharding_plan import EmbeddingShardingPlan, EmbeddingShardingType
from torchrec.distributed.types import ShardingType


class EmbeddingTable(Protocol):
    """The table fields required by the embedding planner."""

    @property
    def num_embeddings(self) -> int: ...

    @property
    def embedding_dim(self) -> int: ...

    @property
    def sharding(self) -> Optional[ShardingType]: ...

    # An enum whose ``.value`` is the storage location, e.g. "HBM" or "UVM".
    @property
    def location(self) -> Any: ...


logger: logging.Logger = logging.getLogger(__name__)

# Default ROW_WISE threshold for tables that do NOT declare a sharding type.
# An explicit EmbeddingTable.sharding always overrides this.
DEFAULT_ROW_WISE_MIN_ROWS = 1_000_000


def _safe_for_row_wise(num_embeddings: int, world_size: int) -> bool:
    """Check that ``torch.chunk(num_embeddings, world_size)`` produces
    exactly ``world_size`` chunks (no rank gets a 0-row shard).

    ``torch.chunk`` uses ``chunk_size = ceil(N / W)`` then
    ``torch.split(tensor, chunk_size)``, which yields
    ``ceil(N / chunk_size)`` chunks - often fewer than W when N is small
    or not divisible by W.
    """
    if world_size <= 1:
        return True
    chunk_size = (num_embeddings + world_size - 1) // world_size
    if chunk_size == 0:
        return False
    actual_chunks = (num_embeddings + chunk_size - 1) // chunk_size
    return actual_chunks >= world_size


def should_row_wise_shard(
    num_embeddings: int,
    world_size: int,
    declared: Optional[ShardingType] = None,
) -> bool:
    """Single source of truth for "row-wise shard vs replicate" per table.

    The per-table ``EmbeddingTable.sharding`` config is authoritative:

      * ``ROW_WISE``      -> row-wise shard. Must pass the 0-row-shard safety
        check, else raise.
      * ``DATA_PARALLEL`` -> replicate (never sharded, regardless of size).
      * any other explicit ShardingType must be handled by the caller or raises;
        it must never silently change into replication.
      * ``None`` (unset)  -> size-based default: row-wise iff the table is large
        (>= ``DEFAULT_ROW_WISE_MIN_ROWS``) AND safe to chunk; otherwise
        replicate.
    """
    if declared == ShardingType.ROW_WISE:
        if not _safe_for_row_wise(num_embeddings, world_size):
            raise ValueError(
                f"table declared ShardingType.ROW_WISE has {num_embeddings} "
                f"rows, which cannot be chunked into {world_size} non-empty "
                "shards (a trailing rank would get 0 rows, crashing "
                "SparseEmbeddingTransform). Declare ShardingType.DATA_PARALLEL "
                "for a table this small, or increase num_embeddings."
            )
        return True
    if declared == ShardingType.DATA_PARALLEL:
        return False
    if declared is not None:
        raise NotImplementedError(
            f"should_row_wise_shard does not resolve {declared}; "
            "use build_embedding_plan for mixed placements."
        )
    # Unset: size-based default, still safety-gated.
    return num_embeddings >= DEFAULT_ROW_WISE_MIN_ROWS and _safe_for_row_wise(
        num_embeddings, world_size
    )


def _resolve_sharding(
    fqn: str,
    param: torch.nn.Parameter,
    table: EmbeddingTable | None,
    world_size: int,
) -> EmbeddingShardingType | None:
    if table is not None:
        expected_shape = (table.num_embeddings, table.embedding_dim)
        if tuple(param.shape) != expected_shape:
            raise ValueError(
                f"Embedding parameter {fqn!r} has shape {tuple(param.shape)}, "
                f"expected unsharded table shape {expected_shape}."
            )
    declared = table.sharding if table is not None else None
    if declared == ShardingType.TABLE_WISE:
        return EmbeddingShardingType.TABLE_WISE
    if should_row_wise_shard(param.shape[0], world_size, declared):
        return EmbeddingShardingType.ROW_WISE
    logger.info(
        "[recir plan] %s (%d rows, %d ranks, sharding=%s): replicated",
        fqn,
        param.shape[0],
        world_size,
        declared.value if declared is not None else "default",
    )
    return None


def _validate_shared_aliases(
    parameter_aliases: Dict[int, list[str]],
    resolved_sharding: Dict[str, EmbeddingShardingType | None],
    resolved_locations: Dict[str, str],
) -> None:
    for aliases in parameter_aliases.values():
        if len(aliases) == 1:
            continue
        configured_aliases = [alias for alias in aliases if alias in resolved_sharding]
        if configured_aliases and len(configured_aliases) != len(aliases):
            raise ValueError(
                "Shared embedding parameter aliases must all have an embedding "
                f"placement: aliases={aliases!r}, configured={configured_aliases!r}."
            )
        if len({resolved_sharding[alias] for alias in configured_aliases}) > 1:
            raise ValueError(
                "Shared embedding parameter aliases must use one sharding type: "
                f"aliases={aliases!r}."
            )
        alias_locations = {
            resolved_locations.get(alias, "HBM") for alias in configured_aliases
        }
        if len(alias_locations) > 1:
            raise ValueError(
                "Shared embedding parameter aliases must use one storage location: "
                f"aliases={aliases!r}, locations={sorted(alias_locations)!r}."
            )


def build_embedding_plan(
    model: torch.nn.Module,
    table_names: Set[str],
    world_size: int = 1,
    embedding_tables: Optional[Mapping[str, EmbeddingTable]] = None,
) -> EmbeddingShardingPlan:
    """Resolve row-wise, table-wise, and replicated table placements.

    Walks ``model.named_parameters()`` and matches FQNs ending in
    ``.{table_name}.weight`` against ``table_names``. Explicit table-wise tables
    are assigned whole to balance table-wise-owned parameter bytes across ranks;
    ties are broken by rank. This is the resolved owner required by table-wise
    routing. Replicated and row-wise memory is outside this placement heuristic.

    When ``embedding_tables`` is provided, the ``location`` field of each
    sharded table is stored in the plan's ``location_map`` so the compile path
    can pick the right allocator (UVM vs HBM) when materializing meta-device
    shards.
    """
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}.")
    sharding_map: Dict[str, EmbeddingShardingType] = {}
    location_map: Dict[str, str] = {}
    resolved_sharding: Dict[str, EmbeddingShardingType | None] = {}
    resolved_locations: Dict[str, str] = {}
    parameter_aliases: Dict[int, list[str]] = {}
    table_wise_tables: Dict[int, tuple[int, list[str]]] = {}
    matched = 0
    for fqn, param in model.named_parameters(remove_duplicate=False):
        parameter_aliases.setdefault(id(param), []).append(fqn)
        for name in table_names:
            if fqn.endswith(f".{name}.weight"):
                matched += 1
                table = embedding_tables.get(name) if embedding_tables else None
                sharding = _resolve_sharding(fqn, param, table, world_size)
                resolved_sharding[fqn] = sharding
                if table is not None:
                    resolved_locations[fqn] = table.location.value
                if sharding is None:
                    break
                sharding_map[fqn] = sharding
                if table is not None:
                    location_map[fqn] = table.location.value
                if sharding == EmbeddingShardingType.TABLE_WISE:
                    _, aliases = table_wise_tables.setdefault(
                        id(param),
                        (int(param.numel()) * param.element_size(), []),
                    )
                    aliases.append(fqn)
                break
    _validate_shared_aliases(
        parameter_aliases,
        resolved_sharding,
        resolved_locations,
    )
    assert matched > 0, (
        "No embedding params matched the configured table names. The FQN "
        "format produced by ECSparseArch may have changed."
    )
    table_wise_bytes_by_rank = [0] * world_size
    table_wise_rank_map: Dict[str, int] = {}
    for table_bytes, aliases in sorted(
        table_wise_tables.values(), key=lambda item: (-item[0], item[1])
    ):
        owner_rank = min(
            range(world_size),
            key=lambda rank: (table_wise_bytes_by_rank[rank], rank),
        )
        table_wise_rank_map.update({fqn: owner_rank for fqn in aliases})
        table_wise_bytes_by_rank[owner_rank] += table_bytes

    return EmbeddingShardingPlan(
        sharding_map=sharding_map,
        location_map=location_map,
        table_wise_rank_map=table_wise_rank_map,
    )
