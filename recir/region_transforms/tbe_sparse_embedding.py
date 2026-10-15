# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""TBE-backed sharded sparse embedding transform.

The inner rewrite replaces compatible ``aten.embedding`` and
``aten._embedding_bag`` sites with grouped, rank-local FBGEMM TBE calls. The
outer rewrite keeps row-wise and table-wise routing as ordinary functional
collective nodes around the opaque native TBE operators.
"""

from __future__ import annotations

import operator
from collections import OrderedDict
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from functools import partial
from itertools import accumulate
from typing import Any, cast, Literal

import torch
import torch.distributed._functional_collectives as funcol
import torch.fx as fx
import torch.nn as nn
from functorch import make_fx
from recir.distributed.sharding_plan import EmbeddingShardingPlan, EmbeddingShardingType
from recir.embedding.tbe.local import LocalTBEGroup, LocalTBETable, nobag_tbe_offsets
from recir.embedding.tbe.ops import get_permute_2d_sparse_data_op
from recir.pipeline import mark_pipeline_region
from recir.pipeline_strategies import PipelineRegion
from recir.region_transforms.base import RegionTransform
from recir.region_transforms.sparse_collectives import (
    declare_split_invariants,
    materialize_splits,
    return_table_wise_rows,
    route_table_wise_jagged,
    route_table_wise_rows,
    split_lists,
)
from recir.region_transforms.sparse_embedding import _embedding_bag_mode
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DeviceMesh
from torch.fx.experimental.dynamic_spec import ShapeVar, TensorSpec
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


_PLACEHOLDER_SIZE: int = 3
_TBE_ROOT_NAME: str = "_recir_tbe_sparse_groups"
_PoolingMode = Literal["none", "sum", "mean"]
_WEIGHT_PASSTHROUGH_OPS: tuple[object, ...] = (
    torch.ops.aten._to_copy.default,
    torch.ops.aten.clone.default,
    torch.ops.aten.detach.default,
    torch.ops.aten.view.default,
    torch.ops.aten.reshape.default,
)


def _node_argument(
    node: fx.Node,
    position: int,
    name: str,
    default: object,
) -> object:
    if len(node.args) > position:
        return node.args[position]
    return node.kwargs.get(name, default)


def _resolve_weight_placeholder(
    weight_arg: object,
    weight_placeholders: Collection[fx.Node],
) -> fx.Node | None:
    node = weight_arg
    while isinstance(node, fx.Node) and node.op == "call_function":
        if node.target not in _WEIGHT_PASSTHROUGH_OPS and not getattr(
            node.target, "_recir_preserved_autograd_leaf", False
        ):
            break
        node = node.args[0]
    if not isinstance(node, fx.Node) or node not in weight_placeholders:
        return None
    if node is not weight_arg:
        arg_val = (
            weight_arg.meta.get("val") if isinstance(weight_arg, fx.Node) else None
        )
        root_val = node.meta.get("val")
        if arg_val is None or root_val is None or arg_val.shape != root_val.shape:
            return None
    return node


@dataclass(frozen=True)
class _OptimizerSettings:
    learning_rate: float
    eps: float
    stochastic_rounding: bool
    learning_rate_tensor: torch.Tensor | None


@dataclass(frozen=True)
class _TableState:
    fqn: str
    parameter: nn.Parameter
    location: str
    optimizer: torch.optim.Optimizer | None
    optimizer_group: dict[str, Any] | None
    optimizer_state: dict[str, Any] | None
    settings: _OptimizerSettings


@dataclass(frozen=True)
class _FeatureSite:
    fqn: str
    global_rows: int
    embedding_dim: int


@dataclass(frozen=True)
class _LookupSite:
    output: fx.Node
    indices: fx.Node
    fqn: str
    offsets: fx.Node | None = None
    pooling: _PoolingMode = "none"


@dataclass(frozen=True)
class _CompiledGroup:
    sites: tuple[_FeatureSite, ...]
    device: torch.device
    hash_size_cumsum: tuple[int, ...]
    hash_size_offset: tuple[int, ...]
    is_pooled: bool
    sharding_type: EmbeddingShardingType
    owner_rank: int | None


def _as_float(value: object, name: str) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"TBE {name} tensor must contain one value.")
        # This runs once during compilation, not in the training loop.
        return float(value.detach().to(device="cpu").item())
    return float(value)  # pyre-ignore[6]


def _row_wise_block_size(num_rows: int, world_size: int) -> int:
    return (num_rows + world_size - 1) // world_size


def _row_wise_local_rows(num_rows: int, world_size: int, rank: int) -> int:
    block_size = _row_wise_block_size(num_rows, world_size)
    return max(0, min(block_size, num_rows - rank * block_size))


def route_row_wise_indices_grouped(
    *route_inputs: torch.Tensor,
    feature_count: int,
    num_embeddings: tuple[int, ...],
    group_name: str,
    world_size: int,
    use_index_dedup: bool,
    segments_per_feature: int,
) -> tuple[
    tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]
]:
    """Route a TBE feature group with one counts and one indices all-to-all."""
    expected_input_count = feature_count + (2 if use_index_dedup else 0)
    if len(route_inputs) != expected_input_count:
        raise ValueError(
            "Sparse routing inputs do not match features and hash metadata."
        )
    feature_indices = route_inputs[:feature_count]
    if feature_count != len(num_embeddings):
        raise ValueError("Feature and table cardinality counts must match.")

    feature_lengths = torch.stack(
        tuple(
            torch.scalar_tensor(
                indices.shape[0],
                dtype=torch.int64,
                device=indices.device,
            )
            for indices in feature_indices
        )
    )
    segment_offsets = nobag_tbe_offsets(feature_lengths, segments_per_feature)
    packed_lengths = segment_offsets[1:] - segment_offsets[:-1]
    packed_indices = torch.cat(
        tuple(indices.reshape(-1) for indices in feature_indices)
    )
    if use_index_dedup:
        hash_size_cumsum = route_inputs[feature_count]
        hash_size_offset = route_inputs[feature_count + 1]
        dedup_lengths, unique_indices, reverse_indices = (
            torch.ops.recir.dedup_sparse_features(
                packed_lengths,
                packed_indices,
                hash_size_cumsum,
                hash_size_offset,
            )
        )
        packed_lengths = dedup_lengths.reshape(packed_lengths.shape)
        reverse_indices = reverse_indices.reshape(packed_indices.shape)
        packed_indices = unique_indices
    else:
        reverse_indices = torch.arange(
            packed_indices.shape[0],
            dtype=torch.int64,
            device=packed_indices.device,
        )
    block_sizes = torch.stack(
        tuple(
            packed_indices.new_full((), _row_wise_block_size(rows, world_size))
            for rows in num_embeddings
        )
    )
    (
        bucketized_lengths,
        bucketized_indices,
        bucketize_permute,
        unbucketize_permute,
    ) = torch.ops.recir.bucketize_sparse_features(
        packed_lengths,
        packed_indices,
        block_sizes,
        world_size,
    )

    segments_per_rank = feature_count * segments_per_feature
    pending_received_lengths = torch.ops._c10d_functional.all_to_all_single(
        bucketized_lengths,
        [segments_per_rank] * world_size,
        [segments_per_rank] * world_size,
        group_name,
    )
    received_lengths = torch.ops._c10d_functional.wait_tensor(
        pending_received_lengths
    ).reshape(world_size * feature_count, segments_per_feature)
    input_splits = bucketized_lengths.reshape(world_size, -1).sum(dim=1)
    output_splits = received_lengths.reshape(world_size, -1).sum(dim=1)
    input_split_list, output_split_list, cpu_splits = materialize_splits(
        input_splits, output_splits
    )
    declare_split_invariants(input_split_list, packed_indices.shape[0])
    declare_split_invariants(output_split_list)

    pending_received_indices = torch.ops._c10d_functional.all_to_all_single(
        bucketized_indices,
        output_split_list,
        input_split_list,
        group_name,
    )
    received_indices = torch.ops._c10d_functional.wait_tensor(pending_received_indices)

    receive_recat = (
        torch.arange(
            world_size * feature_count,
            dtype=torch.int32,
            device=received_lengths.device,
        )
        .reshape(world_size, feature_count)
        .permute(1, 0)
        .reshape(-1)
    )
    feature_major_lengths, feature_major_indices = (
        torch.ops.recir.permute_sparse_data_2d(
            receive_recat,
            received_lengths,
            received_indices,
        )
    )
    offsets = torch.cat(
        (
            feature_major_lengths.new_zeros((1,)),
            feature_major_lengths.reshape(-1),
        )
    ).cumsum(dim=0)
    return (
        (
            feature_major_indices,
            offsets,
            bucketize_permute,
            unbucketize_permute,
            receive_recat,
            received_lengths,
            feature_major_lengths,
            cpu_splits,
            reverse_indices,
        ),
        (
            bucketized_lengths,
            pending_received_lengths,
            bucketized_indices,
            bucketize_permute,
            unbucketize_permute,
            reverse_indices,
        ),
        (
            pending_received_indices,
            received_lengths,
            cpu_splits,
        ),
    )


def _dependency_closure(
    outputs: Sequence[fx.Node], candidates: set[fx.Node]
) -> set[fx.Node]:
    closure: set[fx.Node] = set()
    pending = list(outputs)
    while pending:
        node = pending.pop()
        if node in closure or node not in candidates:
            continue
        closure.add(node)
        pending.extend(node.all_input_nodes)
    return closure


def _exchange_pooled_counts(
    send_counts: torch.Tensor,
    packed_lengths: torch.Tensor,
    feature_count: int,
    group_name: str,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int], torch.Tensor, torch.Tensor]:
    local_batch_size = packed_lengths.shape[0] // feature_count
    local_batch_size_tensor = torch.scalar_tensor(
        local_batch_size,
        dtype=torch.int64,
        device=packed_lengths.device,
    ).reshape(1)
    batch_sizes = torch.ops._c10d_functional.all_gather_into_tensor(
        local_batch_size_tensor,
        world_size,
        group_name,
    )
    batch_sizes = torch.ops._c10d_functional.wait_tensor(batch_sizes)
    uniform_batch_per_rank = (batch_sizes == local_batch_size_tensor).all()
    sent_segment_counts = batch_sizes.new_full(
        (world_size,),
        local_batch_size * feature_count,
    )
    received_segment_counts = batch_sizes * feature_count
    send_splits, receive_splits, cpu_segment_splits = materialize_splits(
        sent_segment_counts,
        received_segment_counts,
    )
    received_counts = torch.ops._c10d_functional.all_to_all_single(
        send_counts,
        receive_splits,
        send_splits,
        group_name,
    )
    return (
        received_counts,
        batch_sizes,
        receive_splits,
        cpu_segment_splits,
        uniform_batch_per_rank,
    )


def _exchange_pooled_indices(
    bucketized_indices: torch.Tensor,
    send_counts: torch.Tensor,
    received_counts: torch.Tensor,
    received_count_splits: list[int],
    group_name: str,
    world_size: int,
) -> torch.Tensor:
    send_splits = send_counts.reshape(world_size, -1).sum(dim=1)
    receive_splits = torch.stack(
        tuple(
            source_counts.sum()
            for source_counts in torch.split(
                received_counts,
                received_count_splits,
            )
        )
    )
    send_split_list, receive_split_list, _cpu_splits = materialize_splits(
        send_splits,
        receive_splits,
    )
    declare_split_invariants(send_split_list, bucketized_indices.shape[0])
    declare_split_invariants(receive_split_list)
    received_indices = torch.ops._c10d_functional.all_to_all_single(
        bucketized_indices,
        receive_split_list,
        send_split_list,
        group_name,
    )
    return received_indices


def _pooled_value_splits(
    cpu_segment_splits: torch.Tensor,
    embedding_dim: int,
    world_size: int,
) -> tuple[list[int], list[int]]:
    """Convert feature-bag counts to flattened value counts.

    Each split counts bags across all features for one rank, so multiplying by
    the per-feature embedding dimension produces the flattened TBE value count.
    """
    local_segment_splits, source_segment_splits = split_lists(
        cpu_segment_splits,
        world_size,
    )
    return (
        [size * embedding_dim for size in local_segment_splits],
        [size * embedding_dim for size in source_segment_splits],
    )


def route_table_wise_pooled_indices_grouped(
    packed_indices: torch.Tensor,
    packed_offsets: torch.Tensor,
    *,
    feature_count: int,
    owner_rank: int,
    group_name: str,
    world_size: int,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, ...],
    tuple[torch.Tensor, ...],
]:
    """Route complete pooled features to one table owner."""
    packed_lengths = packed_offsets[1:] - packed_offsets[:-1]
    received_indices, received_lengths, cpu_bag_splits, received_bag_counts = (
        route_table_wise_jagged(
            packed_indices,
            packed_lengths,
            owner_rank,
            group_name,
            world_size,
            segments_per_bag=feature_count,
        )
    )

    source_to_feature = torch.ops.recir.pooled_source_to_feature_permute(
        received_lengths,
        received_bag_counts,
        feature_count,
    )
    owner_lengths, owner_indices = torch.ops.recir.permute_sparse_data_1d(
        source_to_feature.to(torch.int32),
        received_lengths,
        received_indices,
    )
    owner_offsets = torch.cat((owner_lengths.new_zeros((1,)), owner_lengths)).cumsum(
        dim=0
    )
    return (
        (owner_indices, owner_offsets, cpu_bag_splits),
        (cpu_bag_splits,),
        (owner_indices, owner_offsets),
    )


def return_table_wise_pooled_embeddings_grouped(
    owner_embeddings: torch.Tensor,
    cpu_bag_splits: torch.Tensor,
    *,
    group_name: str,
    world_size: int,
) -> torch.Tensor:
    """Return complete pooled rows from the table owner to source ranks."""
    send_bag_splits, _received_bag_splits = split_lists(cpu_bag_splits, world_size)
    return return_table_wise_rows(
        owner_embeddings,
        cpu_bag_splits,
        sum(send_bag_splits),
        group_name,
        world_size,
    )


def route_table_wise_pooled_gradients_grouped(
    gradients: torch.Tensor,
    cpu_bag_splits: torch.Tensor,
    *,
    group_name: str,
    world_size: int,
) -> torch.Tensor:
    """Route source gradients to the complete table owner."""
    return route_table_wise_rows(
        gradients,
        cpu_bag_splits,
        group_name,
        world_size,
    )


def route_row_wise_pooled_indices_grouped(
    packed_indices: torch.Tensor,
    packed_offsets: torch.Tensor,
    *,
    num_embeddings: tuple[int, ...],
    group_name: str,
    world_size: int,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, ...],
    tuple[torch.Tensor, ...],
]:
    """Route pooled IDs while preserving every feature and bag boundary."""
    feature_count = len(num_embeddings)
    packed_lengths = packed_offsets[1:] - packed_offsets[:-1]
    block_sizes = torch.stack(
        tuple(
            packed_indices.new_full((), _row_wise_block_size(rows, world_size))
            for rows in num_embeddings
        )
    )
    send_counts, bucketized_indices, _bucketize_permute, _unbucketize_permute = (
        torch.ops.recir.bucketize_sparse_features(
            packed_lengths,
            packed_indices,
            block_sizes,
            world_size,
        )
    )
    (
        pending_received_counts,
        batch_sizes,
        received_count_splits,
        cpu_segment_splits,
        uniform_batch_per_rank,
    ) = _exchange_pooled_counts(
        send_counts,
        packed_lengths,
        feature_count,
        group_name,
        world_size,
    )
    received_counts = torch.ops._c10d_functional.wait_tensor(pending_received_counts)
    pending_received_indices = _exchange_pooled_indices(
        bucketized_indices,
        send_counts,
        received_counts,
        received_count_splits,
        group_name,
        world_size,
    )
    received_indices = torch.ops._c10d_functional.wait_tensor(pending_received_indices)
    source_to_feature = torch.ops.recir.pooled_source_to_feature_permute(
        received_counts,
        batch_sizes,
        feature_count,
    )
    feature_major_counts, feature_major_indices = (
        torch.ops.recir.permute_sparse_data_1d(
            source_to_feature.to(torch.int32),
            received_counts,
            received_indices,
        )
    )
    local_offsets = torch.cat(
        (feature_major_counts.new_zeros((1,)), feature_major_counts)
    ).cumsum(dim=0)
    return (
        (
            feature_major_indices,
            local_offsets,
            cpu_segment_splits,
            uniform_batch_per_rank,
        ),
        (pending_received_counts, bucketized_indices, cpu_segment_splits),
        (pending_received_indices, received_counts),
    )


def return_row_wise_pooled_embeddings_grouped(
    local_embeddings: torch.Tensor,
    source_offsets: torch.Tensor,
    cpu_segment_splits: torch.Tensor,
    uniform_batch_per_rank: torch.Tensor,
    *,
    feature_count: int,
    group_name: str,
    world_size: int,
) -> torch.Tensor:
    """Return owner-local partial bag sums and combine them on source ranks."""

    def uniform_batch(
        embeddings: torch.Tensor,
        _offsets: torch.Tensor,
        _splits: torch.Tensor,
    ) -> torch.Tensor:
        reduced = torch.ops._c10d_functional.reduce_scatter_tensor(
            embeddings,
            "sum",
            world_size,
            group_name,
        )
        return torch.ops._c10d_functional.wait_tensor(reduced)

    def variable_batch(
        embeddings: torch.Tensor,
        offsets: torch.Tensor,
        segment_splits: torch.Tensor,
    ) -> torch.Tensor:
        output_width = embeddings.shape[1]
        embedding_dim = output_width // feature_count
        local_splits, source_splits = _pooled_value_splits(
            segment_splits,
            embedding_dim,
            world_size,
        )
        declare_split_invariants(source_splits, embeddings.numel())
        declare_split_invariants(local_splits)
        returned = torch.ops._c10d_functional.all_to_all_single(
            embeddings.reshape(-1),
            local_splits,
            source_splits,
            group_name,
        )
        returned = torch.ops._c10d_functional.wait_tensor(returned)
        return torch.ops.recir.sum_pooled_embeddings(
            returned,
            offsets,
            feature_count,
            world_size,
            output_width,
        )

    return torch.cond(
        uniform_batch_per_rank,
        uniform_batch,
        variable_batch,
        (local_embeddings, source_offsets, cpu_segment_splits),
    )


def route_row_wise_pooled_gradients_grouped(
    gradients: torch.Tensor,
    local_embeddings: torch.Tensor,
    cpu_segment_splits: torch.Tensor,
    uniform_batch_per_rank: torch.Tensor,
    *,
    feature_count: int,
    group_name: str,
    world_size: int,
) -> torch.Tensor:
    """Send each source bag gradient to every row owner."""

    def uniform_batch(
        source_gradients: torch.Tensor,
        _local_embeddings: torch.Tensor,
        _splits: torch.Tensor,
    ) -> torch.Tensor:
        gathered = torch.ops._c10d_functional.all_gather_into_tensor(
            source_gradients,
            world_size,
            group_name,
        )
        return torch.ops._c10d_functional.wait_tensor(gathered)

    def variable_batch(
        source_gradients: torch.Tensor,
        embeddings: torch.Tensor,
        segment_splits: torch.Tensor,
    ) -> torch.Tensor:
        output_width = source_gradients.shape[1]
        embedding_dim = output_width // feature_count
        local_splits, source_splits = _pooled_value_splits(
            segment_splits,
            embedding_dim,
            world_size,
        )
        replicated = (
            source_gradients.unsqueeze(0).expand(world_size, -1, -1).contiguous()
        )
        declare_split_invariants(local_splits, replicated.numel())
        declare_split_invariants(source_splits, embeddings.numel())
        received = torch.ops._c10d_functional.all_to_all_single(
            replicated.reshape(-1),
            source_splits,
            local_splits,
            group_name,
        )
        received = torch.ops._c10d_functional.wait_tensor(received)
        return torch.ops.recir.reshape_pooled_gradients(received, embeddings)

    return torch.cond(
        uniform_batch_per_rank,
        uniform_batch,
        variable_batch,
        (gradients, local_embeddings, cpu_segment_splits),
    )


def _mean_from_pooled_sum(
    pooled: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    lengths = offsets[1:] - offsets[:-1]
    divisor = lengths.clamp_min(1).to(pooled.dtype).unsqueeze(1)
    return pooled / divisor


def _pooled_lookup_site(
    node: fx.Node,
    fqn: str,
) -> _LookupSite:
    indices = _node_argument(node, 1, "indices", None)
    offsets = _node_argument(node, 2, "offsets", None)
    if not isinstance(indices, fx.Node) or len(indices.meta["val"].shape) != 1:
        raise ValueError("TBE sparse transform requires one-dimensional IDs.")
    if not isinstance(offsets, fx.Node) or len(offsets.meta["val"].shape) != 1:
        raise ValueError("TBE pooled transform requires one-dimensional offsets.")
    mode = _embedding_bag_mode(node)
    outputs = [
        user
        for user in node.users
        if user.op == "call_function"
        and user.target == operator.getitem
        and user.args[1] == 0
    ]
    if len(outputs) != 1 or len(node.users) != 1:
        raise NotImplementedError(
            "TBE pooled transform only supports the embedding_bag output tensor."
        )
    return _LookupSite(
        outputs[0],
        indices,
        fqn,
        offsets,
        "sum" if mode == 0 else "mean",
    )


def _lookup_site_from_node(
    node: fx.Node,
    weight_nodes: dict[fx.Node, str],
) -> _LookupSite | None:
    if node.op != "call_function":
        return None
    if node.target == torch.ops.aten.embedding.default:
        root = _resolve_weight_placeholder(
            _node_argument(node, 0, "weight", None), weight_nodes
        )
        indices = _node_argument(node, 1, "indices", None)
        if root is None:
            return None
        if not isinstance(indices, fx.Node) or len(indices.meta["val"].shape) != 1:
            raise ValueError("TBE sparse transform requires one-dimensional IDs.")
        return _LookupSite(node, indices, weight_nodes[root])
    if node.target != torch.ops.aten._embedding_bag.default:
        return None

    root = _resolve_weight_placeholder(
        _node_argument(node, 0, "weight", None), weight_nodes
    )
    if root is None:
        return None
    return _pooled_lookup_site(node, weight_nodes[root])


def return_row_wise_embeddings_grouped(
    local_embeddings: torch.Tensor,
    unbucketize_permute: torch.Tensor,
    feature_major_lengths: torch.Tensor,
    cpu_splits: torch.Tensor,
    group_name: str,
    world_size: int,
) -> torch.Tensor:
    """Return one grouped TBE output and restore original feature order."""
    input_splits, output_splits = split_lists(cpu_splits, world_size)
    declare_split_invariants(input_splits, unbucketize_permute.shape[0])
    declare_split_invariants(output_splits, local_embeddings.shape[0])
    embedding_dim = local_embeddings.shape[1]
    feature_count = feature_major_lengths.shape[0] // world_size
    feature_to_source_recat = (
        torch.arange(
            world_size * feature_count,
            dtype=torch.int32,
            device=local_embeddings.device,
        )
        .reshape(feature_count, world_size)
        .permute(1, 0)
        .reshape(-1)
    )
    _, source_major_embeddings = torch.ops.recir.permute_sparse_data_2d(
        feature_to_source_recat,
        feature_major_lengths * embedding_dim,
        local_embeddings.reshape(-1),
    )
    returned = torch.ops._c10d_functional.all_to_all_single(
        source_major_embeddings,
        [size * embedding_dim for size in input_splits],
        [size * embedding_dim for size in output_splits],
        group_name,
    )
    returned = torch.ops._c10d_functional.wait_tensor(returned)
    return returned.reshape(-1, embedding_dim)[unbucketize_permute]


def route_row_wise_gradients_grouped(
    gradients: torch.Tensor,
    bucketize_permute: torch.Tensor,
    reverse_indices: torch.Tensor,
    receive_recat: torch.Tensor,
    received_lengths: torch.Tensor,
    cpu_splits: torch.Tensor,
    group_name: str,
    world_size: int,
    use_index_dedup: bool,
) -> torch.Tensor:
    """Route packed feature gradients using grouped forward metadata."""
    input_splits, output_splits = split_lists(cpu_splits, world_size)
    embedding_dim = gradients.shape[1]
    if use_index_dedup:
        unique_gradients = gradients.new_zeros(
            (bucketize_permute.shape[0], embedding_dim)
        )
        gradients = torch.index_add(
            unique_gradients,
            0,
            reverse_indices,
            gradients,
        )
    declare_split_invariants(input_splits, gradients.shape[0])
    received = torch.ops._c10d_functional.all_to_all_single(
        gradients[bucketize_permute].reshape(-1),
        [size * embedding_dim for size in output_splits],
        [size * embedding_dim for size in input_splits],
        group_name,
    )
    received = torch.ops._c10d_functional.wait_tensor(received)
    _, feature_major_gradients = torch.ops.recir.permute_sparse_data_2d(
        receive_recat,
        received_lengths * embedding_dim,
        received,
    )
    return feature_major_gradients.reshape(-1, embedding_dim)


def _inline_before(
    gm: fx.GraphModule,
    target: fx.Node,
    function: Any,
    example_inputs: Sequence[torch.Tensor],
    input_nodes: Sequence[fx.Node],
    *,
    tracing_mode: Literal["fake", "real", "symbolic"] = "symbolic",
    dynamic_shapes: dict[str, TensorSpec] | None = None,
) -> tuple[Any, set[fx.Node]]:
    traced = make_fx(
        function,
        tracing_mode=tracing_mode,
        dynamic_shapes=dynamic_shapes,
    )(*example_inputs)
    get_attrs = [node for node in traced.graph.nodes if node.op == "get_attr"]
    if get_attrs:
        inlined_name = "_recir_inlined"
        suffix = 0
        while hasattr(gm, inlined_name):
            suffix += 1
            inlined_name = f"_recir_inlined_{suffix}"
        gm.add_module(inlined_name, traced)
        for node in get_attrs:
            node.target = f"{inlined_name}.{cast(str, node.target)}"
    placeholders = [node for node in traced.graph.nodes if node.op == "placeholder"]
    val_map = dict(zip(placeholders, input_nodes))
    existing = set(gm.graph.nodes)
    with gm.graph.inserting_before(target):
        output = gm.graph.graph_copy(traced.graph, val_map=val_map)
    new_nodes = set(gm.graph.nodes) - existing
    for node in new_nodes:
        node.meta.pop("unbacked_bindings", None)
    return output, new_nodes


def _lookup_dependencies(sites: Sequence[_LookupSite]) -> set[fx.Node]:
    relevant: set[fx.Node] = set()
    pending = [site.indices for site in sites]
    pending.extend(site.offsets for site in sites if site.offsets is not None)
    while pending:
        node = pending.pop()
        if node in relevant:
            continue
        relevant.add(node)
        pending.extend(node.all_input_nodes)
    return relevant


def _rewrite_relevant_kjt_permutations(
    gm: fx.GraphModule,
    sites: Sequence[_LookupSite],
) -> None:
    relevant = _lookup_dependencies(sites)
    permutation_op = get_permute_2d_sparse_data_op()

    for node in tuple(gm.graph.nodes):
        if (
            node not in relevant
            or node.op != "call_function"
            or node.target != torch.ops.fbgemm.permute_2D_sparse_data.default
        ):
            continue
        lengths_2d = _node_argument(node, 1, "lengths", None)
        if not (
            isinstance(lengths_2d, fx.Node)
            and lengths_2d.op == "call_function"
            and lengths_2d.target == torch.ops.aten.view.default
            and isinstance(lengths_2d.args[0], fx.Node)
        ):
            continue
        lengths_value = lengths_2d.meta.get("val")
        if not isinstance(lengths_value, torch.Tensor) or lengths_value.dim() != 2:
            continue
        feature_count = int(lengths_value.shape[0])
        flat_lengths = cast(fx.Node, lengths_2d.args[0])
        permutation = _node_argument(node, 0, "permute", None)
        if not isinstance(permutation, fx.Node):
            continue
        values = _node_argument(node, 2, "values", None)
        if not isinstance(values, fx.Node):
            continue
        with gm.graph.inserting_before(node):
            replacement = gm.graph.call_function(
                permutation_op,
                args=cast(
                    tuple[Any, ...],
                    (
                        permutation,
                        flat_lengths,
                        values,
                        feature_count,
                        _node_argument(node, 3, "weights", None),
                    ),
                ),
            )
        replacement.meta = dict(node.meta)
        node.replace_all_uses_with(replacement)
        gm.graph.erase_node(node)


class TBESparseEmbeddingTransform(RegionTransform):
    """Use grouped local TBE while retaining sparse collectives in the graph.

    Supports unpooled and SUM/MEAN pooled one-dimensional feature ID tensors
    with Exact RowWiseAdagrad and no weight decay. Compatible lookup sites are
    grouped by pooling shape, dtype, device, embedding dimension, storage
    location, and optimizer settings. Row-wise pooled groups with equal bag
    counts across ranks select reduce-scatter automatically; unequal counts use
    variable-split all-to-all.
    """

    def __init__(
        self,
        embedding_param_fqns: set[str],
        mesh: DeviceMesh,
        mod: nn.Module | None = None,
        *,
        process_group: ProcessGroup | None = None,
        embedding_plan: EmbeddingShardingPlan | None = None,
        num_embeddings_map: dict[str, int] | None = None,
        learning_rate: float = 0.1,
        eps: float = 1.0e-8,
        stochastic_rounding: bool = True,
        segments_per_feature: int = 128,
        use_index_dedup: bool = True,
        forward_root_fqn: str | None = None,
    ) -> None:
        """``forward_root_fqn``: the called module whose forward is re-traced
        and rewritten (it must own every TBE table; ``""`` is the root module).
        Defaults to the deepest such module."""
        if segments_per_feature <= 0:
            raise ValueError("TBE segments per feature must be positive.")
        self.embedding_param_fqns = embedding_param_fqns
        self._source_fqns = set(embedding_param_fqns)
        self.mesh = mesh
        self.world_size = mesh.size()
        if process_group is not None and process_group.size() != self.world_size:
            raise ValueError(
                "TBE process group size must match the embedding device mesh size."
            )
        self.group_name = funcol._resolve_group_name(
            process_group if process_group is not None else (mesh, 0)
        )
        self.mod = mod
        self.embedding_plan = embedding_plan
        self.num_embeddings_map = num_embeddings_map or (
            {fqn: int(mod.get_parameter(fqn).shape[0]) for fqn in self._source_fqns}
            if mod is not None
            else {}
        )
        self.default_settings = _OptimizerSettings(
            learning_rate, eps, stochastic_rounding, None
        )
        self._table_states: dict[str, _TableState] = {}
        self._optimizers: tuple[torch.optim.Optimizer, ...] = ()
        self._compiled_groups: list[_CompiledGroup] = []
        self._requested_forward_root_fqn: str | None = forward_root_fqn
        self._forward_root_fqn: str | None = None
        self._segments_per_feature: int = segments_per_feature
        self._use_index_dedup: bool = use_index_dedup
        for fqn in self._source_fqns:
            self._table_wise_owner(fqn)

    def _sharding_type(self, fqn: str) -> EmbeddingShardingType:
        if self.embedding_plan is None:
            return EmbeddingShardingType.ROW_WISE
        sharding_type = self.embedding_plan.sharding_map.get(fqn)
        if sharding_type is None:
            raise ValueError(
                f"Embedding sharding plan does not classify parameter {fqn!r}."
            )
        if sharding_type not in (
            EmbeddingShardingType.ROW_WISE,
            EmbeddingShardingType.TABLE_WISE,
        ):
            raise NotImplementedError(
                f"TBE transform does not support sharding type {sharding_type.value!r}."
            )
        return sharding_type

    def _table_wise_owner(self, fqn: str) -> int | None:
        sharding_type = self._sharding_type(fqn)
        if sharding_type == EmbeddingShardingType.ROW_WISE:
            return None
        embedding_plan = self.embedding_plan
        if embedding_plan is None:
            raise RuntimeError("Table-wise TBE requires an embedding sharding plan.")
        return embedding_plan.table_wise_owner(fqn)

    def _embedding_location(self, fqn: str) -> str:
        if self.embedding_plan is None:
            return "HBM"
        return self.embedding_plan.location_map.get(fqn, "HBM")

    def _rank(self) -> int:
        coordinate = self.mesh.get_coordinate()
        if coordinate is None:
            raise RuntimeError("Current rank is not part of the TBE device mesh.")
        return coordinate[0]

    def select_forward_root_fqns(
        self,
        called_modules: OrderedDict[
            str, tuple[nn.Module, tuple[Any, ...], dict[str, Any]]
        ],
    ) -> set[str]:
        candidates = [
            fqn
            for fqn in called_modules
            if not fqn
            or all(
                source == fqn or source.startswith(f"{fqn}.")
                for source in self._source_fqns
            )
        ]
        if not candidates:
            raise RuntimeError("No called module owns all TBE embedding parameters.")
        requested = self._requested_forward_root_fqn
        if requested is not None:
            if requested not in candidates:
                raise ValueError(
                    f"TBE forward root {requested!r} is not a called module that "
                    f"owns all TBE embedding parameters; candidates: {candidates}"
                )
            self._forward_root_fqn = requested
        else:
            self._forward_root_fqn = max(candidates, key=lambda fqn: fqn.count("."))
        return {self._forward_root_fqn}

    def transform_inputs(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        for value in (*args, *kwargs.values()):
            if isinstance(value, KeyedJaggedTensor):
                value.lengths()
                value.offsets()
        return args, kwargs

    def prepare_after_shard(
        self,
        module: nn.Module,
        optimizers: Sequence[torch.optim.Optimizer],
    ) -> None:
        self.mod = module
        self._optimizers = tuple(optimizers)
        self._table_states = {
            fqn: self._capture_table_state(module, fqn) for fqn in self._source_fqns
        }

    def _expected_local_rows(self, fqn: str, global_rows: int, rank: int) -> int:
        if self._sharding_type(fqn) == EmbeddingShardingType.TABLE_WISE:
            owner_rank = self._table_wise_owner(fqn)
            return global_rows if owner_rank == rank else 0
        expected_rows = _row_wise_local_rows(global_rows, self.world_size, rank)
        if expected_rows == 0:
            raise NotImplementedError(
                "TBE row-wise sharding requires non-empty local table shards."
            )
        return expected_rows

    def _optimizer_entry(
        self,
        fqn: str,
        parameter: nn.Parameter,
    ) -> tuple[torch.optim.Optimizer, dict[str, Any]] | None:
        found = [
            (optimizer, group)
            for optimizer in self._optimizers
            for group in optimizer.param_groups
            if any(candidate is parameter for candidate in group["params"])
        ]
        if len(found) > 1:
            raise ValueError(f"TBE embedding parameter {fqn!r} has multiple owners.")
        return found[0] if found else None

    def _capture_table_state(self, module: nn.Module, fqn: str) -> _TableState:
        parameter = module.get_parameter(fqn)
        if parameter.dim() != 2:
            raise ValueError(f"TBE embedding parameter {fqn!r} must be 2-D.")
        global_rows = self.num_embeddings_map.get(
            fqn, int(parameter.shape[0]) * self.world_size
        )
        rank = self._rank()
        local_rows = int(parameter.shape[0])
        expected_local_rows = self._expected_local_rows(fqn, global_rows, rank)
        if local_rows != expected_local_rows:
            raise NotImplementedError(
                "TBE sparse routing received an unexpected local table shape; "
                f"{fqn!r} has global_rows={global_rows}, "
                f"local_rows={local_rows}, expected_local_rows={expected_local_rows}, "
                f"rank={rank}, world_size={self.world_size}."
            )
        optimizer_entry = self._optimizer_entry(fqn, parameter)
        if optimizer_entry is None:
            return _TableState(
                fqn,
                parameter,
                self._embedding_location(fqn),
                None,
                None,
                None,
                self.default_settings,
            )
        optimizer, group = optimizer_entry
        if not type(optimizer).__name__.endswith("RowWiseAdagrad"):
            raise TypeError(
                f"TBE embedding parameter {fqn!r} must use RowWiseAdagrad; "
                f"got {type(optimizer).__name__}."
            )
        weight_decay = _as_float(group.get("weight_decay", 0.0), "weight_decay")
        if weight_decay != 0.0:
            raise NotImplementedError(
                "TBE transform does not yet migrate weight decay."
            )
        lr_value = group.get("lr", self.default_settings.learning_rate)
        lr_tensor = lr_value if isinstance(lr_value, torch.Tensor) else None
        settings = _OptimizerSettings(
            _as_float(lr_value, "learning_rate"),
            _as_float(group.get("eps", self.default_settings.eps), "eps"),
            bool(group.get("stochastic_rounding", True)),
            lr_tensor,
        )
        state = optimizer.state.get(parameter)
        return _TableState(
            fqn,
            parameter,
            self._embedding_location(fqn),
            optimizer,
            group,
            state if isinstance(state, dict) else None,
            settings,
        )

    def transform_graph(
        self,
        gm: fx.GraphModule,
        called_modules: OrderedDict[
            str, tuple[nn.Module, tuple[Any, ...], dict[str, Any]]
        ],
    ) -> None:
        if not self._source_fqns:
            return
        if self.mod is None or not self._table_states:
            raise RuntimeError("prepare_after_shard must run before transform_graph.")
        if self._forward_root_fqn is None:
            raise RuntimeError("TBE forward root was not selected before tracing.")
        gm.graph.eliminate_dead_code()
        root_module = called_modules[self._forward_root_fqn][0]
        relative_fqns = [
            fqn for fqn, _ in root_module.named_parameters(remove_duplicate=False)
        ]
        placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
        weight_nodes: dict[fx.Node, str] = {}
        for placeholder, relative_fqn in zip(placeholders, relative_fqns):
            full_fqn = (
                f"{self._forward_root_fqn}.{relative_fqn}"
                if self._forward_root_fqn
                else relative_fqn
            )
            if full_fqn in self._source_fqns:
                weight_nodes[placeholder] = full_fqn
        sites: list[_LookupSite] = []
        for node in gm.graph.nodes:
            site = _lookup_site_from_node(node, weight_nodes)
            if site is not None:
                sites.append(site)
        rewritten_fqns = {site.fqn for site in sites}
        missing_fqns = self._source_fqns - rewritten_fqns
        if missing_fqns:
            raise RuntimeError(
                "TBE sparse transform did not find lookup sites for sharded "
                f"embedding parameters: {sorted(missing_fqns)}"
            )
        _rewrite_relevant_kjt_permutations(gm, sites)
        self._rewrite_compatible_groups(gm, sites)
        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()

    def _rewrite_compatible_groups(
        self,
        gm: fx.GraphModule,
        sites: Sequence[_LookupSite],
    ) -> None:
        grouped: dict[tuple[object, ...], list[_LookupSite]] = {}
        for site in sites:
            state = self._table_states[site.fqn]
            parameter = state.parameter
            sharding_type = self._sharding_type(site.fqn)
            owner_rank = self._table_wise_owner(site.fqn)
            key = (
                sharding_type,
                owner_rank,
                site.pooling
                if sharding_type == EmbeddingShardingType.TABLE_WISE
                else site.pooling != "none",
                parameter.device,
                parameter.dtype,
                int(parameter.shape[1]),
                state.location,
                state.settings.learning_rate,
                state.settings.eps,
                state.settings.stochastic_rounding,
                id(state.settings.learning_rate_tensor),
            )
            grouped.setdefault(key, []).append(site)
        module = self.mod
        if module is None:
            raise RuntimeError("prepare_after_shard must set the source module.")
        if hasattr(module, _TBE_ROOT_NAME):
            raise RuntimeError(f"Module already owns {_TBE_ROOT_NAME!r}.")
        owner = nn.ModuleDict()
        module.add_module(_TBE_ROOT_NAME, owner)
        for group_index, group_sites in enumerate(grouped.values()):
            table_order = tuple(dict.fromkeys(site.fqn for site in group_sites))
            ordered_sites = tuple(
                site
                for table_fqn in table_order
                for site in group_sites
                if site.fqn == table_fqn
            )
            self._rewrite_group(gm, owner, group_index, ordered_sites)

    def _rewrite_group(
        self,
        gm: fx.GraphModule,
        owner: nn.ModuleDict,
        group_index: int,
        sites: Sequence[_LookupSite],
    ) -> None:
        is_pooled = sites[0].pooling != "none"
        if any((site.pooling != "none") != is_pooled for site in sites):
            raise RuntimeError("TBE group mixed pooled and unpooled lookup sites.")
        sharding_type = self._sharding_type(sites[0].fqn)
        owner_rank = self._table_wise_owner(sites[0].fqn)
        if sharding_type == EmbeddingShardingType.TABLE_WISE and not is_pooled:
            raise NotImplementedError(
                "Table-wise TBE currently supports pooled EBC lookups only."
            )
        pooling_mode = sites[0].pooling
        if sharding_type == EmbeddingShardingType.ROW_WISE and pooling_mode == "mean":
            pooling_mode = "sum"
        if any(
            self._sharding_type(site.fqn) != sharding_type
            or self._table_wise_owner(site.fqn) != owner_rank
            or (
                sharding_type == EmbeddingShardingType.TABLE_WISE
                and site.pooling != pooling_mode
            )
            for site in sites
        ):
            raise RuntimeError("TBE group mixed embedding placements or pooling modes.")
        table_fqns, local_tbe = self._create_local_tbe_group(sites, pooling_mode)
        owner_name = f"group_{group_index}"
        owner[owner_name] = local_tbe
        graph_name = f"_recir_tbe_group_{group_index}"
        gm.add_submodule(graph_name, local_tbe)
        self._migrate_group_state(table_fqns, local_tbe)
        insertion_target, grouped_output = self._insert_group_call(
            gm,
            graph_name,
            sites,
            is_pooled,
        )
        self._replace_group_outputs(
            gm,
            grouped_output,
            insertion_target,
            sites,
        )
        self._compiled_groups.append(
            self._build_compiled_group(
                sites,
                self._table_states[sites[0].fqn].parameter.device,
                is_pooled,
            )
        )

    def _create_local_tbe_group(
        self,
        sites: Sequence[_LookupSite],
        pooling_mode: _PoolingMode,
    ) -> tuple[tuple[str, ...], LocalTBEGroup]:
        is_pooled = pooling_mode != "none"
        table_fqns = tuple(dict.fromkeys(site.fqn for site in sites))
        table_index = {fqn: index for index, fqn in enumerate(table_fqns)}
        first_state = self._table_states[table_fqns[0]]
        source_weights = tuple(self._tbe_source_weight(fqn) for fqn in table_fqns)
        tables = tuple(
            LocalTBETable(
                name=fqn,
                local_rows=int(source_weight.shape[0]),
                embedding_dim=int(self._table_states[fqn].parameter.shape[1]),
                location=self._table_states[fqn].location,
            )
            for fqn, source_weight in zip(table_fqns, source_weights)
        )
        with unset_fake_temporarily():
            local_tbe = LocalTBEGroup(
                tables,
                tuple(table_index[site.fqn] for site in sites),
                device=first_state.parameter.device,
                weight_dtype=first_state.parameter.dtype,
                learning_rate=first_state.settings.learning_rate,
                learning_rate_tensor=first_state.settings.learning_rate_tensor,
                eps=first_state.settings.eps,
                stochastic_rounding=first_state.settings.stochastic_rounding,
                segments_per_feature=1 if is_pooled else self._segments_per_feature,
                pooling_mode=pooling_mode,
            )
            local_tbe.load_table_weights(source_weights)
        return table_fqns, local_tbe

    def _owns_table(self, fqn: str) -> bool:
        if self._sharding_type(fqn) == EmbeddingShardingType.ROW_WISE:
            return True
        owner_rank = self._table_wise_owner(fqn)
        return self._rank() == owner_rank

    def _tbe_source_weight(self, fqn: str) -> torch.Tensor:
        parameter = self._table_states[fqn].parameter
        if self._owns_table(fqn):
            return parameter
        return parameter.new_zeros((1, parameter.shape[1]))

    @staticmethod
    def _insert_group_call(
        gm: fx.GraphModule,
        graph_name: str,
        sites: Sequence[_LookupSite],
        is_pooled: bool,
    ) -> tuple[fx.Node, fx.Node]:
        node_positions = {node: index for index, node in enumerate(gm.graph.nodes)}
        insertion_target = min(
            sites, key=lambda site: node_positions[site.output]
        ).output
        if any(
            node_positions[site.indices] >= node_positions[insertion_target]
            or (
                site.offsets is not None
                and node_positions[site.offsets] >= node_positions[insertion_target]
            )
            for site in sites
        ):
            raise RuntimeError("TBE feature inputs must be available before the group.")
        if is_pooled and any(site.offsets is None for site in sites):
            raise RuntimeError("Pooled TBE lookup site is missing offsets.")
        with gm.graph.inserting_before(insertion_target):
            if is_pooled:
                grouped_output = gm.graph.call_module(
                    graph_name,
                    args=(
                        tuple(site.indices for site in sites),
                        tuple(cast(fx.Node, site.offsets) for site in sites),
                    ),
                )
            else:
                grouped_output = gm.graph.call_module(
                    graph_name,
                    args=(tuple(site.indices for site in sites),),
                )
        return insertion_target, grouped_output

    def _replace_group_outputs(
        self,
        gm: fx.GraphModule,
        grouped_output: fx.Node,
        insertion_target: fx.Node,
        sites: Sequence[_LookupSite],
    ) -> None:
        replacements: list[tuple[fx.Node, fx.Node]] = []
        for feature_index, site in enumerate(sites):
            output = self._replacement_output(
                gm,
                grouped_output,
                insertion_target,
                feature_index,
                site,
            )
            replacements.append((site.output, output))
        for original, replacement in replacements:
            original.replace_all_uses_with(replacement)
            gm.graph.erase_node(original)

    def _replacement_output(
        self,
        gm: fx.GraphModule,
        grouped_output: fx.Node,
        insertion_target: fx.Node,
        feature_index: int,
        site: _LookupSite,
    ) -> fx.Node:
        with gm.graph.inserting_before(insertion_target):
            output = gm.graph.call_function(
                operator.getitem, args=(grouped_output, feature_index)
            )
            output.meta = dict(site.output.meta)
        if (
            site.pooling == "mean"
            and self._sharding_type(site.fqn) == EmbeddingShardingType.ROW_WISE
        ):
            offsets = cast(fx.Node, site.offsets)
            parameter = self._table_states[site.fqn].parameter
            output, _ = _inline_before(
                gm,
                insertion_target,
                _mean_from_pooled_sum,
                (
                    torch.randn(
                        3,
                        int(parameter.shape[1]),
                        device=parameter.device,
                    ),
                    torch.tensor(
                        [0, 1, 3, 3],
                        dtype=torch.int64,
                        device=parameter.device,
                    ),
                ),
                (output, offsets),
                tracing_mode="fake",
            )
            output.meta = dict(site.output.meta)
        return output

    def _build_compiled_group(
        self,
        sites: Sequence[_LookupSite],
        device: torch.device,
        is_pooled: bool,
    ) -> _CompiledGroup:
        sharding_type = self._sharding_type(sites[0].fqn)
        owner_rank = self._table_wise_owner(sites[0].fqn)
        if any(
            self._sharding_type(site.fqn) != sharding_type
            or self._table_wise_owner(site.fqn) != owner_rank
            for site in sites
        ):
            raise RuntimeError("TBE group mixed embedding placements.")
        return _CompiledGroup(
            sites=tuple(
                _FeatureSite(
                    site.fqn,
                    self.num_embeddings_map.get(
                        site.fqn,
                        int(self._table_states[site.fqn].parameter.shape[0])
                        * self.world_size,
                    ),
                    int(self._table_states[site.fqn].parameter.shape[1]),
                )
                for site in sites
            ),
            device=device,
            hash_size_cumsum=self._dedup_hash_size_cumsum(sites),
            hash_size_offset=self._dedup_hash_size_offset(sites),
            is_pooled=is_pooled,
            sharding_type=sharding_type,
            owner_rank=owner_rank,
        )

    def _dedup_hash_size_cumsum(
        self,
        sites: Sequence[_LookupSite],
    ) -> tuple[int, ...]:
        feature_hash_sizes: list[int] = []
        for table_fqn in dict.fromkeys(site.fqn for site in sites):
            table_feature_count = sum(site.fqn == table_fqn for site in sites)
            feature_hash_sizes.extend([0] * (table_feature_count - 1))
            feature_hash_sizes.append(self.num_embeddings_map[table_fqn])
        return (0, *accumulate(feature_hash_sizes))

    @staticmethod
    def _dedup_hash_size_offset(
        sites: Sequence[_LookupSite],
    ) -> tuple[int, ...]:
        feature_hash_size_lengths: list[int] = []
        for table_fqn in dict.fromkeys(site.fqn for site in sites):
            table_feature_count = sum(site.fqn == table_fqn for site in sites)
            feature_hash_size_lengths.append(table_feature_count)
            feature_hash_size_lengths.extend([0] * (table_feature_count - 1))
        return (0, *accumulate(feature_hash_size_lengths))

    def _checkpoint_optimizer_state(
        self,
        fqn: str,
        state: _TableState,
        tbe_states: Sequence[torch.Tensor],
    ) -> torch.Tensor | None:
        if not self._owns_table(fqn) or state.optimizer is None:
            return None
        if len(tbe_states) != 1:
            raise RuntimeError(
                f"TBE table {fqn!r} must expose one RowWiseAdagrad state."
            )
        checkpoint_state = tbe_states[0]
        source_sum = (
            state.optimizer_state.get("sum")
            if state.optimizer_state is not None
            else None
        )
        if isinstance(source_sum, torch.Tensor):
            reshaped_source = source_sum.reshape(checkpoint_state.shape)
            if reshaped_source.is_meta:
                # Primed meta Adagrad state represents zero initialization.
                checkpoint_state.zero_()
            else:
                checkpoint_state.copy_(reshaped_source)
        return checkpoint_state

    def _install_checkpoint_parameter(
        self,
        fqn: str,
        tbe_weight: torch.Tensor,
    ) -> nn.Parameter | None:
        if not self._owns_table(fqn):
            return None
        parent, attribute = self._parameter_parent(fqn)
        checkpoint_parameter = nn.Parameter(tbe_weight, requires_grad=False)
        setattr(parent, attribute, checkpoint_parameter)
        return checkpoint_parameter

    @staticmethod
    def _replace_optimizer_parameter(
        state: _TableState,
        checkpoint_parameter: nn.Parameter | None,
        checkpoint_state: torch.Tensor | None,
    ) -> None:
        if state.optimizer is None or state.optimizer_group is None:
            return
        replacement_parameter = (
            checkpoint_parameter
            if checkpoint_parameter is not None
            else state.parameter
        )
        state.optimizer_group["params"] = [
            replacement_parameter if parameter is state.parameter else parameter
            for parameter in state.optimizer_group["params"]
            if parameter is not state.parameter or checkpoint_parameter is not None
        ]
        state.optimizer.state.pop(state.parameter, None)
        if checkpoint_parameter is not None and checkpoint_state is not None:
            state.optimizer.state[checkpoint_parameter] = {
                "sum": checkpoint_state.reshape(-1, 1)
            }

    @torch.no_grad()
    def _migrate_group_state(
        self,
        table_fqns: Sequence[str],
        local_tbe: LocalTBEGroup,
    ) -> None:
        tbe_weights = local_tbe.split_embedding_weights()
        tbe_states = local_tbe.split_optimizer_states()
        for table_index, fqn in enumerate(table_fqns):
            state = self._table_states[fqn]
            checkpoint_state = self._checkpoint_optimizer_state(
                fqn,
                state,
                tbe_states[table_index],
            )
            checkpoint_parameter = self._install_checkpoint_parameter(
                fqn,
                tbe_weights[table_index],
            )
            self._replace_optimizer_parameter(
                state,
                checkpoint_parameter,
                checkpoint_state,
            )
        module = self.mod
        if module is None:
            raise RuntimeError("prepare_after_shard must set the source module.")
        prefix = f"{_TBE_ROOT_NAME}."
        self.embedding_param_fqns.update(
            name for name, _ in module.named_parameters() if name.startswith(prefix)
        )

    def _parameter_parent(self, fqn: str) -> tuple[nn.Module, str]:
        assert self.mod is not None
        atoms = fqn.split(".")
        parent = self.mod
        for atom in atoms[:-1]:
            child = getattr(parent, atom)
            if not isinstance(child, nn.Module):
                raise TypeError(
                    f"Embedding parent {'.'.join(atoms[:-1])!r} is not a module."
                )
            parent = child
        return parent, atoms[-1]

    def finalize_compiled_graph(self, gm: fx.GraphModule) -> None:
        if self.world_size == 1 or not self._compiled_groups:
            return
        forwards = [
            node
            for node in gm.graph.nodes
            if node.op == "call_function"
            and node.target == torch.ops.recir.tbe_forward.default
        ]
        if len(forwards) != len(self._compiled_groups):
            raise RuntimeError(
                "TBE forward count changed between inner and outer traces: "
                f"expected {len(self._compiled_groups)}, found {len(forwards)}."
            )
        for group_index, (forward, group) in enumerate(
            zip(forwards, self._compiled_groups)
        ):
            self._route_group(gm, forward, group, group_index)
        gm.graph.lint()
        gm.recompile()

    @staticmethod
    def _tbe_forward_io(forward: fx.Node) -> tuple[fx.Node, fx.Node, fx.Node]:
        packed_indices = forward.args[5]
        packed_offsets = forward.args[6]
        forward_output = next(
            (
                user
                for user in forward.users
                if user.op == "call_function"
                and user.target == operator.getitem
                and user.args[1] == 0
            ),
            None,
        )
        if forward_output is None:
            raise RuntimeError("Expected tensor output from recir::tbe_forward.")
        if not isinstance(packed_indices, fx.Node) or not isinstance(
            packed_offsets, fx.Node
        ):
            raise RuntimeError("Expected TBE indices and offsets nodes.")
        return packed_indices, packed_offsets, forward_output

    @staticmethod
    def _tbe_backward_and_gradient(
        gm: fx.GraphModule,
        forward: fx.Node,
    ) -> tuple[fx.Node, fx.Node]:
        backward = next(
            (
                node
                for node in gm.graph.nodes
                if node.op == "call_function"
                and node.target == torch.ops.recir.tbe_backward_update.default
                and node.args[0] == forward.args[0]
            ),
            None,
        )
        if backward is None:
            raise RuntimeError("Could not find fused TBE backward for forward group.")
        grad_output = backward.args[4]
        if not isinstance(grad_output, fx.Node):
            raise RuntimeError("TBE backward gradient must be an FX node.")
        return backward, grad_output

    def _route_group(
        self,
        gm: fx.GraphModule,
        forward: fx.Node,
        group: _CompiledGroup,
        group_index: int,
    ) -> None:
        if group.is_pooled:
            if group.sharding_type == EmbeddingShardingType.TABLE_WISE:
                self._route_table_wise_pooled_group(gm, forward, group)
                return
            if group.sharding_type != EmbeddingShardingType.ROW_WISE:
                raise NotImplementedError(
                    "TBE pooled routing does not support sharding type "
                    f"{group.sharding_type.value!r}."
                )
            self._route_pooled_group(gm, forward, group)
            return
        if group.sharding_type != EmbeddingShardingType.ROW_WISE:
            raise NotImplementedError(
                "TBE unpooled routing does not support sharding type "
                f"{group.sharding_type.value!r}."
            )
        packed_indices, packed_offsets, forward_output = self._tbe_forward_io(forward)
        if packed_indices.target != torch.ops.aten.cat.default:
            raise RuntimeError("Expected LocalTBEGroup to pack indices with aten.cat.")
        raw_features = packed_indices.args[0]
        if not isinstance(raw_features, (list, tuple)) or len(raw_features) != len(
            group.sites
        ):
            raise RuntimeError(
                "TBE packed feature count does not match transform metadata."
            )
        if any(not isinstance(raw, fx.Node) for raw in raw_features):
            raise RuntimeError("TBE feature indices must be FX nodes.")
        feature_nodes = tuple(cast(fx.Node, raw) for raw in raw_features)
        feature_count = len(group.sites)
        example_features = tuple(
            torch.randint(
                0,
                site.global_rows,
                (_PLACEHOLDER_SIZE,),
                device=group.device,
            )
            for site in group.sites
        )
        route_example_inputs = example_features
        route_input_nodes = feature_nodes
        if self._use_index_dedup:
            # Citrine C3: materialize static metadata directly on its runtime device.
            hash_size_cumsum = torch.tensor(
                group.hash_size_cumsum,
                dtype=torch.int64,
                device=group.device,
            )
            hash_size_offset = torch.tensor(
                group.hash_size_offset,
                dtype=torch.int64,
                device=group.device,
            )
            hash_size_cumsum_name = f"_recir_dedup_hash_size_cumsum_{group_index}"
            hash_size_offset_name = f"_recir_dedup_hash_size_offset_{group_index}"
            gm.register_buffer(
                hash_size_cumsum_name,
                hash_size_cumsum,
                persistent=False,
            )
            gm.register_buffer(
                hash_size_offset_name,
                hash_size_offset,
                persistent=False,
            )
            with gm.graph.inserting_before(forward):
                hash_size_cumsum_node = gm.graph.get_attr(hash_size_cumsum_name)
                hash_size_offset_node = gm.graph.get_attr(hash_size_offset_name)
            hash_size_cumsum_node.meta["val"] = hash_size_cumsum
            hash_size_offset_node.meta["val"] = hash_size_offset
            route_example_inputs = (
                *route_example_inputs,
                hash_size_cumsum,
                hash_size_offset,
            )
            route_input_nodes = (
                *route_input_nodes,
                hash_size_cumsum_node,
                hash_size_offset_node,
            )
        route_output, route_nodes = _inline_before(
            gm,
            forward,
            partial(
                route_row_wise_indices_grouped,
                feature_count=feature_count,
                num_embeddings=tuple(site.global_rows for site in group.sites),
                group_name=self.group_name,
                world_size=self.world_size,
                use_index_dedup=self._use_index_dedup,
                segments_per_feature=self._segments_per_feature,
            ),
            route_example_inputs,
            route_input_nodes,
        )
        output, start_boundary, continue_boundary = route_output
        start_nodes = _dependency_closure(start_boundary, route_nodes)
        continue_nodes = (
            _dependency_closure(continue_boundary, route_nodes) - start_nodes
        )
        graph_nodes = set(gm.graph.nodes)
        preprocess_nodes = {
            node
            for node in _dependency_closure(feature_nodes, graph_nodes - route_nodes)
            if node.op not in ("placeholder", "get_attr")
        }
        mark_pipeline_region(preprocess_nodes, PipelineRegion.PREPROCESS)
        mark_pipeline_region(start_nodes, PipelineRegion.ROUTE_START)
        mark_pipeline_region(continue_nodes, PipelineRegion.ROUTE_CONTINUE)
        (
            received_indices,
            offsets,
            bucketize_permute,
            unbucketize_permute,
            receive_recat,
            received_lengths,
            feature_major_lengths,
            cpu_splits,
            reverse_indices,
        ) = output

        lookup_nodes = self._replace_forward_inputs(
            gm,
            forward,
            packed_indices,
            packed_offsets,
            received_indices,
            offsets,
        )

        split_node = next(
            (
                user
                for user in forward_output.users
                if user.op == "call_function"
                and user.target == torch.ops.aten.split_with_sizes.default
            ),
            None,
        )
        if split_node is None:
            raise RuntimeError("Expected grouped TBE output to use split_with_sizes.")
        original_lengths: list[fx.Node] = []
        with gm.graph.inserting_before(split_node):
            for feature in feature_nodes:
                original_lengths.append(
                    gm.graph.call_function(
                        torch.ops.aten.sym_size.int, args=(feature, 0)
                    )
                )
        returned_output, returned_nodes = _inline_before(
            gm,
            split_node,
            partial(
                return_row_wise_embeddings_grouped,
                group_name=self.group_name,
                world_size=self.world_size,
            ),
            (
                torch.randn(
                    (feature_count * _PLACEHOLDER_SIZE, group.sites[0].embedding_dim),
                    device=group.device,
                ),
                torch.arange(feature_count * _PLACEHOLDER_SIZE, device=group.device),
                torch.ones(
                    (
                        feature_count * self.world_size,
                        self._segments_per_feature,
                    ),
                    dtype=torch.int64,
                    device=group.device,
                ),
                torch.ones((2 * self.world_size,), dtype=torch.int64),
            ),
            (
                forward_output,
                unbucketize_permute,
                feature_major_lengths,
                cpu_splits,
            ),
        )
        lookup_nodes.update(returned_nodes)
        if self._use_index_dedup:
            with gm.graph.inserting_before(split_node):
                returned_output = gm.graph.call_function(
                    torch.ops.aten.index_select.default,
                    args=(returned_output, 0, reverse_indices),
                )
            lookup_nodes.add(returned_output)
        split_node.args = (returned_output, original_lengths)
        lookup_nodes.add(split_node)
        mark_pipeline_region(lookup_nodes, PipelineRegion.EMBEDDING_LOOKUP)

        self._route_group_backward(
            gm,
            forward,
            group,
            bucketize_permute,
            reverse_indices,
            receive_recat,
            received_lengths,
            cpu_splits,
        )

    def _route_pooled_group(
        self,
        gm: fx.GraphModule,
        forward: fx.Node,
        group: _CompiledGroup,
    ) -> None:
        packed_indices, packed_offsets, forward_output = self._tbe_forward_io(forward)

        # The unpooled dedup path reconstructs one output row per original ID.
        # Pooled routing must preserve per-bag membership instead, so it routes
        # every ID until a bag-aware reverse mapping is available.
        feature_count = len(group.sites)
        route_output, route_nodes = _inline_before(
            gm,
            forward,
            partial(
                route_row_wise_pooled_indices_grouped,
                num_embeddings=tuple(site.global_rows for site in group.sites),
                group_name=self.group_name,
                world_size=self.world_size,
            ),
            (
                torch.arange(
                    feature_count * _PLACEHOLDER_SIZE,
                    device=group.device,
                ),
                torch.arange(
                    feature_count * _PLACEHOLDER_SIZE + 1,
                    dtype=torch.int64,
                    device=group.device,
                ),
            ),
            (packed_indices, packed_offsets),
        )
        output, start_boundary, continue_boundary = route_output
        start_nodes = _dependency_closure(start_boundary, route_nodes)
        continue_nodes = (
            _dependency_closure(continue_boundary, route_nodes) - start_nodes
        )
        graph_nodes = set(gm.graph.nodes)
        preprocess_nodes = {
            node
            for node in _dependency_closure(
                (packed_indices, packed_offsets), graph_nodes - route_nodes
            )
            if node.op not in ("placeholder", "get_attr")
        }
        mark_pipeline_region(preprocess_nodes, PipelineRegion.PREPROCESS)
        mark_pipeline_region(start_nodes, PipelineRegion.ROUTE_START)
        mark_pipeline_region(continue_nodes, PipelineRegion.ROUTE_CONTINUE)
        (
            received_indices,
            local_offsets,
            cpu_segment_splits,
            uniform_batch_per_rank,
        ) = output
        lookup_nodes = self._replace_forward_inputs(
            gm,
            forward,
            packed_indices,
            packed_offsets,
            received_indices,
            local_offsets,
        )
        lookup_nodes.update(
            self._return_pooled_group_output(
                gm,
                forward_output,
                packed_offsets,
                cpu_segment_splits,
                uniform_batch_per_rank,
                group,
            )
        )
        mark_pipeline_region(lookup_nodes, PipelineRegion.EMBEDDING_LOOKUP)
        self._route_pooled_group_backward(
            gm,
            forward,
            forward_output,
            cpu_segment_splits,
            uniform_batch_per_rank,
            group,
        )

    def _route_table_wise_pooled_group(
        self,
        gm: fx.GraphModule,
        forward: fx.Node,
        group: _CompiledGroup,
    ) -> None:
        owner_rank = group.owner_rank
        if owner_rank is None:
            raise RuntimeError("Table-wise TBE group is missing its owner rank.")
        packed_indices, packed_offsets, forward_output = self._tbe_forward_io(forward)
        feature_count = len(group.sites)
        route_output, route_nodes = _inline_before(
            gm,
            forward,
            partial(
                route_table_wise_pooled_indices_grouped,
                feature_count=feature_count,
                owner_rank=owner_rank,
                group_name=self.group_name,
                world_size=self.world_size,
            ),
            (
                torch.arange(
                    feature_count * _PLACEHOLDER_SIZE,
                    device=group.device,
                ),
                torch.arange(
                    feature_count * _PLACEHOLDER_SIZE + 1,
                    dtype=torch.int64,
                    device=group.device,
                ),
            ),
            (packed_indices, packed_offsets),
        )
        output, start_boundary, continue_boundary = route_output
        start_nodes = _dependency_closure(start_boundary, route_nodes)
        continue_nodes = (
            _dependency_closure(continue_boundary, route_nodes) - start_nodes
        )
        graph_nodes = set(gm.graph.nodes)
        preprocess_nodes = {
            node
            for node in _dependency_closure(
                (packed_indices, packed_offsets), graph_nodes - route_nodes
            )
            if node.op not in ("placeholder", "get_attr")
        }
        mark_pipeline_region(preprocess_nodes, PipelineRegion.PREPROCESS)
        mark_pipeline_region(start_nodes, PipelineRegion.ROUTE_START)
        mark_pipeline_region(continue_nodes, PipelineRegion.ROUTE_CONTINUE)
        owner_indices, owner_offsets, cpu_bag_splits = output
        lookup_nodes = self._replace_forward_inputs(
            gm,
            forward,
            packed_indices,
            packed_offsets,
            owner_indices,
            owner_offsets,
        )
        lookup_nodes.update(
            self._return_table_wise_pooled_group_output(
                gm,
                forward_output,
                cpu_bag_splits,
                packed_offsets,
                group,
            )
        )
        mark_pipeline_region(lookup_nodes, PipelineRegion.EMBEDDING_LOOKUP)
        self._route_table_wise_pooled_group_backward(
            gm,
            forward,
            cpu_bag_splits,
            group,
        )

    def _return_table_wise_pooled_group_output(
        self,
        gm: fx.GraphModule,
        forward_output: fx.Node,
        cpu_bag_splits: fx.Node,
        source_offsets: fx.Node,
        group: _CompiledGroup,
    ) -> set[fx.Node]:
        output_width = len(group.sites) * group.sites[0].embedding_dim
        node_positions = {node: index for index, node in enumerate(gm.graph.nodes)}
        first_output_user = min(forward_output.users, key=node_positions.__getitem__)
        returned_output, new_nodes = _inline_before(
            gm,
            first_output_user,
            partial(
                return_table_wise_pooled_embeddings_grouped,
                group_name=self.group_name,
                world_size=self.world_size,
            ),
            (
                torch.randn(
                    self.world_size * _PLACEHOLDER_SIZE,
                    output_width,
                    device=group.device,
                ),
                torch.full(
                    (2 * self.world_size,),
                    _PLACEHOLDER_SIZE,
                    dtype=torch.int64,
                ),
            ),
            (forward_output, cpu_bag_splits),
        )
        if not isinstance(returned_output, fx.Node):
            raise RuntimeError("Expected table-wise routing to return an FX node.")
        source_offsets_value = source_offsets.meta.get("val")
        forward_output_value = forward_output.meta.get("val")
        if not isinstance(source_offsets_value, torch.Tensor) or not isinstance(
            forward_output_value, torch.Tensor
        ):
            raise RuntimeError(
                "Expected table-wise routing nodes to carry tensor metadata."
            )
        returned_output.meta = dict(forward_output.meta)
        source_bag_count = (source_offsets_value.shape[0] - 1) // len(group.sites)
        returned_output.meta["val"] = forward_output_value.new_empty(
            (source_bag_count, output_width)
        )
        for user in tuple(forward_output.users):
            if user not in new_nodes:
                user.replace_input_with(forward_output, returned_output)
        return new_nodes

    def _route_table_wise_pooled_group_backward(
        self,
        gm: fx.GraphModule,
        forward: fx.Node,
        cpu_bag_splits: fx.Node,
        group: _CompiledGroup,
    ) -> None:
        backward, grad_output = self._tbe_backward_and_gradient(gm, forward)
        output_width = len(group.sites) * group.sites[0].embedding_dim
        routed_grad, _ = _inline_before(
            gm,
            backward,
            partial(
                route_table_wise_pooled_gradients_grouped,
                group_name=self.group_name,
                world_size=self.world_size,
            ),
            (
                torch.randn(_PLACEHOLDER_SIZE, output_width, device=group.device),
                torch.full(
                    (2 * self.world_size,),
                    _PLACEHOLDER_SIZE,
                    dtype=torch.int64,
                ),
            ),
            (grad_output, cpu_bag_splits),
        )
        backward_args = list(backward.args)
        backward_args[4] = routed_grad
        backward.args = tuple(backward_args)

    def _return_pooled_group_output(
        self,
        gm: fx.GraphModule,
        forward_output: fx.Node,
        source_offsets: fx.Node,
        cpu_segment_splits: fx.Node,
        uniform_batch_per_rank: fx.Node,
        group: _CompiledGroup,
    ) -> set[fx.Node]:
        feature_count = len(group.sites)
        output_width = feature_count * group.sites[0].embedding_dim
        node_positions = {node: index for index, node in enumerate(gm.graph.nodes)}
        first_output_user = min(
            forward_output.users,
            key=node_positions.__getitem__,
        )
        returned_output, new_nodes = _inline_before(
            gm,
            first_output_user,
            partial(
                return_row_wise_pooled_embeddings_grouped,
                feature_count=feature_count,
                group_name=self.group_name,
                world_size=self.world_size,
            ),
            (
                torch.randn(
                    self.world_size * _PLACEHOLDER_SIZE,
                    output_width,
                    device=group.device,
                ),
                torch.arange(
                    feature_count * _PLACEHOLDER_SIZE + 1,
                    dtype=torch.int64,
                    device=group.device,
                ),
                torch.full(
                    (2 * self.world_size,),
                    feature_count * _PLACEHOLDER_SIZE,
                    dtype=torch.int64,
                ),
                torch.tensor(True, device=group.device),
            ),
            (
                forward_output,
                source_offsets,
                cpu_segment_splits,
                uniform_batch_per_rank,
            ),
            tracing_mode="fake",
            dynamic_shapes={
                "local_embeddings": TensorSpec(
                    [ShapeVar("pooled_total_batch"), output_width]
                ),
                "source_offsets": TensorSpec(
                    [ShapeVar("pooled_source_offset_count", min=1)]
                ),
            },
        )
        if not isinstance(returned_output, fx.Node):
            raise RuntimeError("Expected pooled routing to return an FX node.")
        source_offsets_value = source_offsets.meta.get("val")
        forward_output_value = forward_output.meta.get("val")
        if not isinstance(source_offsets_value, torch.Tensor) or not isinstance(
            forward_output_value, torch.Tensor
        ):
            raise RuntimeError(
                "Expected pooled routing nodes to carry tensor metadata."
            )
        returned_output.meta = dict(forward_output.meta)
        source_bag_count = (source_offsets_value.shape[0] - 1) // feature_count
        returned_output.meta["val"] = forward_output_value.new_empty(
            (source_bag_count, output_width)
        )
        for user in tuple(forward_output.users):
            if user not in new_nodes:
                user.replace_input_with(forward_output, returned_output)
        return new_nodes

    @staticmethod
    def _replace_forward_inputs(
        gm: fx.GraphModule,
        forward: fx.Node,
        old_indices: fx.Node,
        old_offsets: fx.Node,
        new_indices: fx.Node,
        new_offsets: fx.Node,
    ) -> set[fx.Node]:
        forward_args = list(forward.args)
        forward_args[5] = new_indices
        forward_args[6] = new_offsets
        forward.args = tuple(forward_args)
        lookup_nodes = {forward}
        for node in tuple(gm.graph.nodes):
            if (
                node.op == "call_function"
                and node.target == torch.ops.fbgemm.bounds_check_indices.default
                and len(node.args) > 2
                and node.args[1] is old_indices
                and node.args[2] is old_offsets
            ):
                bounds_args = list(node.args)
                bounds_args[1] = new_indices
                bounds_args[2] = new_offsets
                node.args = tuple(bounds_args)
                forward.prepend(node)
                lookup_nodes.add(node)
        return lookup_nodes

    def _route_pooled_group_backward(
        self,
        gm: fx.GraphModule,
        forward: fx.Node,
        forward_output: fx.Node,
        cpu_segment_splits: fx.Node,
        uniform_batch_per_rank: fx.Node,
        group: _CompiledGroup,
    ) -> None:
        backward, grad_output = self._tbe_backward_and_gradient(gm, forward)
        feature_count = len(group.sites)
        output_width = feature_count * group.sites[0].embedding_dim
        routed_grad, _ = _inline_before(
            gm,
            backward,
            partial(
                route_row_wise_pooled_gradients_grouped,
                feature_count=feature_count,
                group_name=self.group_name,
                world_size=self.world_size,
            ),
            (
                torch.randn(_PLACEHOLDER_SIZE, output_width, device=group.device),
                torch.randn(
                    self.world_size * _PLACEHOLDER_SIZE,
                    output_width,
                    device=group.device,
                ),
                torch.full(
                    (2 * self.world_size,),
                    feature_count * _PLACEHOLDER_SIZE,
                    dtype=torch.int64,
                ),
                torch.tensor(True, device=group.device),
            ),
            (
                grad_output,
                forward_output,
                cpu_segment_splits,
                uniform_batch_per_rank,
            ),
            tracing_mode="fake",
            dynamic_shapes={
                "gradients": TensorSpec(
                    [ShapeVar("pooled_source_batch"), output_width]
                ),
                "local_embeddings": TensorSpec(
                    [ShapeVar("pooled_gradient_total_batch"), output_width]
                ),
            },
        )
        backward_args = list(backward.args)
        backward_args[4] = routed_grad
        backward.args = tuple(backward_args)

    def _route_group_backward(
        self,
        gm: fx.GraphModule,
        forward: fx.Node,
        group: _CompiledGroup,
        bucketize_permute: fx.Node,
        reverse_indices: fx.Node,
        receive_recat: fx.Node,
        received_lengths: fx.Node,
        cpu_splits: fx.Node,
    ) -> None:
        backward, grad_output = self._tbe_backward_and_gradient(gm, forward)
        embedding_dim = group.sites[0].embedding_dim
        packed_grad, _ = _inline_before(
            gm,
            backward,
            partial(
                route_row_wise_gradients_grouped,
                group_name=self.group_name,
                world_size=self.world_size,
                use_index_dedup=self._use_index_dedup,
            ),
            (
                torch.randn(
                    (len(group.sites) * _PLACEHOLDER_SIZE, embedding_dim),
                    device=group.device,
                ),
                torch.arange(len(group.sites) * _PLACEHOLDER_SIZE, device=group.device),
                torch.arange(len(group.sites) * _PLACEHOLDER_SIZE, device=group.device),
                torch.arange(
                    len(group.sites) * self.world_size,
                    dtype=torch.int32,
                    device=group.device,
                ),
                torch.ones(
                    (
                        len(group.sites) * self.world_size,
                        self._segments_per_feature,
                    ),
                    dtype=torch.int64,
                    device=group.device,
                ),
                torch.ones((2 * self.world_size,), dtype=torch.int64),
            ),
            (
                grad_output,
                bucketize_permute,
                reverse_indices,
                receive_recat,
                received_lengths,
                cpu_splits,
            ),
        )
        backward_args = list(backward.args)
        backward_args[4] = packed_grad
        backward.args = tuple(backward_args)
