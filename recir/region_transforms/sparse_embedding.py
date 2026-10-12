# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""SparseEmbeddingTransform - replace sharded sparse lookup nodes.

Row-wise ``aten.embedding`` and ``aten._embedding_bag`` sites route IDs to row
owners. Table-wise ``aten._embedding_bag`` sites route complete bags to the one
rank that owns the table. Both lower to decomposed functional collectives.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from functools import partial
from typing import Callable, cast, Dict, List, Optional, Set, Tuple

import torch
import torch.distributed._functional_collectives as funcol
import torch.fx as fx
import torch.nn as nn
from functorch import make_fx
from recir.distributed.sharding_plan import EmbeddingShardingPlan, EmbeddingShardingType
from recir.region_transforms.base import RegionTransform
from recir.region_transforms.sparse_collectives import (
    declare_split_invariants,
    return_table_wise_rows,
    route_table_wise_jagged,
    route_table_wise_rows,
)
from torch.distributed.tensor import DeviceMesh


# Placeholder dim used for example inputs to ``make_fx(tracing_mode="symbolic")``
# in the embedding rewrite path.  PyTorch specializes 0 and 1 (treated as
# constants, not symbols), so we need a value >= 2 to ensure the inlined
# subgraph uses real symbolic dims.  Picked small but non-power-of-two to
# avoid any divisibility-based specialization elsewhere.
_SYMBOLIC_TRACE_PLACEHOLDER_DIM: int = 3


# ---------------------------------------------------------------------------
# Decomposed sharded-embedding kernels (pure Python; traced via ``make_fx``).
# ---------------------------------------------------------------------------


def decomposed_row_wise_embedding(
    sharded_weight: torch.Tensor,
    indices: torch.Tensor,
    num_embeddings: int,
    group_name: str,
    world_size: int,
) -> torch.Tensor:
    """Row-wise embedding lookup decomposed into traceable ops.

    Steps:
      1. Bucketize: map each index to the rank that owns it.
      2. Index AllToAll: exchange indices so each rank gets the indices it owns.
      3. Local embedding lookup on the local shard.
      4. Embedding AllToAll: return results to the originating ranks.
      5. Unbucketize: restore the original index order.

    Args:
        sharded_weight: local embedding shard, shape ``(local_V, D)``.
        indices: input indices, shape ``(*batch_shape)``.
        num_embeddings: global vocabulary size ``V``.
        group_name: process-group name for collectives.
        world_size: number of ranks in the group.
    """
    # world_size == 1: skip all_to_all, also avoids a FakeProcessGroup
    # hazard where the size-exchange collective leaves output uninitialized,
    # feeding garbage split sizes into the next all_to_all and OOMing.
    if world_size == 1:
        return torch.ops.aten.embedding.default(sharded_weight, indices)

    # Step 1: Bucketize
    flat_indices = indices.reshape(-1)
    block_rows = (num_embeddings + world_size - 1) // world_size
    ranks = flat_indices // block_rows
    local_indices = flat_indices % block_rows
    sort_order = torch.argsort(ranks, stable=True)
    sorted_local_indices = local_indices[sort_order]
    unsort_order = torch.argsort(sort_order)

    split_sizes = torch.stack([(ranks == r).sum() for r in range(world_size)])
    input_split_list = split_sizes.tolist()
    declare_split_invariants(input_split_list, flat_indices.shape[0])

    # Step 2: Index AllToAll
    # Exchange split sizes (equal-split, each rank sends world_size ints)
    output_splits_tensor = torch.ops._c10d_functional.all_to_all_single(
        split_sizes, [1] * world_size, [1] * world_size, group_name
    )
    output_splits_tensor = torch.ops._c10d_functional.wait_tensor(output_splits_tensor)
    output_split_list = output_splits_tensor.tolist()
    # Peer counts: non-negative, but not relatable to any local size.
    declare_split_invariants(output_split_list, None)

    # Exchange indices
    received_indices = torch.ops._c10d_functional.all_to_all_single(
        sorted_local_indices, output_split_list, input_split_list, group_name
    )
    received_indices = torch.ops._c10d_functional.wait_tensor(received_indices)

    # Step 3: Local embedding lookup
    local_result = torch.ops.aten.embedding.default(sharded_weight, received_indices)

    # Step 4: Embedding AllToAll
    D = sharded_weight.shape[1]
    result_input_splits = [s * D for s in output_split_list]
    result_output_splits = [s * D for s in input_split_list]
    returned_flat = torch.ops._c10d_functional.all_to_all_single(
        local_result.reshape(-1),
        result_output_splits,
        result_input_splits,
        group_name,
    )
    returned_flat = torch.ops._c10d_functional.wait_tensor(returned_flat)
    returned = returned_flat.reshape(-1, D)

    # Step 5: Unbucketize
    result = returned[unsort_order]
    return result.reshape(*indices.shape, D)


def decomposed_row_wise_embedding_backward(
    grad_output: torch.Tensor,
    indices: torch.Tensor,
    num_embeddings: int,
    local_num_embeddings: int,
    group_name: str,
    world_size: int,
    padding_idx: int,
    scale_grad_by_freq: bool,
) -> torch.Tensor:
    """Backward of row-wise embedding lookup decomposed into traceable ops.

    Computes the weight gradient for a row-wise sharded embedding by
    routing gradients to the rank that owns the corresponding weight rows.

    Steps:
      1. Bucketize: map each index to the owning rank (same as forward).
      2. Size AllToAll: exchange per-rank split sizes.
      3. Index AllToAll: exchange sorted local indices.
      4. Gradient AllToAll: exchange sorted gradients to owning ranks.
      5. Local embedding_dense_backward on received data.

    Args:
        grad_output: gradient of the embedding output, shape ``(*batch_shape, D)``.
        indices: input indices (global), shape ``(*batch_shape)``.
        num_embeddings: global vocabulary size ``V``.
        local_num_embeddings: row count of this rank's weight shard.
        group_name: process-group name for collectives.
        world_size: number of ranks in the group.
        padding_idx: padding index (passed through to local backward).
        scale_grad_by_freq: frequency scaling flag (passed through).
    """
    if world_size == 1:
        return torch.ops.aten.embedding_dense_backward.default(
            grad_output,
            indices,
            local_num_embeddings,
            padding_idx,
            scale_grad_by_freq,
        )

    block_rows = (num_embeddings + world_size - 1) // world_size
    D = grad_output.shape[-1]

    # Step 1: Bucketize (identical to forward)
    flat_indices = indices.reshape(-1)
    ranks = flat_indices // block_rows
    local_indices = flat_indices % block_rows
    sort_order = torch.argsort(ranks, stable=True)
    sorted_local_indices = local_indices[sort_order]

    flat_grad = grad_output.reshape(-1, D)
    sorted_grad = flat_grad[sort_order]

    split_sizes = torch.stack([(ranks == r).sum() for r in range(world_size)])
    input_split_list = split_sizes.tolist()
    declare_split_invariants(input_split_list, flat_indices.shape[0])

    # Step 2: Size AllToAll
    output_splits_tensor = torch.ops._c10d_functional.all_to_all_single(
        split_sizes, [1] * world_size, [1] * world_size, group_name
    )
    output_splits_tensor = torch.ops._c10d_functional.wait_tensor(output_splits_tensor)
    output_split_list = output_splits_tensor.tolist()
    # Peer counts: non-negative, but not relatable to any local size.
    declare_split_invariants(output_split_list, None)

    # Step 3: Index AllToAll
    received_indices = torch.ops._c10d_functional.all_to_all_single(
        sorted_local_indices, output_split_list, input_split_list, group_name
    )
    received_indices = torch.ops._c10d_functional.wait_tensor(received_indices)

    # Step 4: Gradient AllToAll
    grad_input_splits = [s * D for s in input_split_list]
    grad_output_splits = [s * D for s in output_split_list]
    received_grad = torch.ops._c10d_functional.all_to_all_single(
        sorted_grad.reshape(-1),
        grad_output_splits,
        grad_input_splits,
        group_name,
    )
    received_grad = torch.ops._c10d_functional.wait_tensor(received_grad)
    received_grad = received_grad.reshape(-1, D)

    # Step 5: Local embedding_dense_backward
    return torch.ops.aten.embedding_dense_backward(
        received_grad,
        received_indices,
        local_num_embeddings,
        padding_idx,
        scale_grad_by_freq,
    )


def _bag_ids(offsets: torch.Tensor, num_indices: int) -> torch.Tensor:
    positions = torch.arange(
        num_indices,
        dtype=offsets.dtype,
        device=offsets.device,
    )
    return torch.searchsorted(offsets[1:], positions, right=True)


def _check_complete_bags(offsets: torch.Tensor, num_indices: int) -> None:
    """Validate the canonical ``include_last_offset=True`` representation."""
    torch._check(
        offsets.shape[0] > 0,
        lambda: "EmbeddingBag offsets must include an initial offset.",
    )
    torch._check_tensor_all(
        offsets[0] == 0,
        lambda: "EmbeddingBag offsets must start at zero.",
    )
    torch._check_tensor_all(
        offsets[1:] >= offsets[:-1],
        lambda: "EmbeddingBag offsets must be nondecreasing.",
    )
    torch._check_tensor_all(
        offsets[-1] == num_indices,
        lambda: "EmbeddingBag offsets must consume every index.",
    )


def _pool_embedding_rows(
    rows: torch.Tensor,
    offsets: torch.Tensor,
    mode: int,
) -> torch.Tensor:
    _check_complete_bags(offsets, rows.shape[0])
    lengths = offsets[1:] - offsets[:-1]
    pooled = rows.new_zeros((lengths.shape[0], rows.shape[1]))
    pooled = pooled.index_add(0, _bag_ids(offsets, rows.shape[0]), rows)
    if mode == 0:
        return pooled
    if mode == 1:
        return pooled / lengths.clamp_min(1).to(pooled.dtype).unsqueeze(1)
    raise ValueError(f"Unsupported embedding_bag pooling mode: {mode}")


def decomposed_row_wise_embedding_bag(
    sharded_weight: torch.Tensor,
    indices: torch.Tensor,
    offsets: torch.Tensor,
    num_embeddings: int,
    group_name: str,
    world_size: int,
    mode: int,
) -> torch.Tensor:
    rows = decomposed_row_wise_embedding(
        sharded_weight,
        indices,
        num_embeddings,
        group_name,
        world_size,
    )
    offsets = torch.ops.aten._to_copy.default(offsets, device=rows.device)
    return _pool_embedding_rows(rows, offsets, mode)


def decomposed_row_wise_embedding_bag_backward(
    grad_output: torch.Tensor,
    indices: torch.Tensor,
    offsets: torch.Tensor,
    num_embeddings: int,
    local_num_embeddings: int,
    group_name: str,
    world_size: int,
    mode: int,
) -> torch.Tensor:
    offsets = torch.ops.aten._to_copy.default(offsets, device=grad_output.device)
    _check_complete_bags(offsets, indices.shape[0])
    torch._check(
        grad_output.shape[0] == offsets.shape[0] - 1,
        lambda: "EmbeddingBag gradient must have one row per bag.",
    )
    bag_ids = _bag_ids(offsets, indices.shape[0])
    per_index_grad = grad_output.index_select(0, bag_ids)
    if mode == 1:
        lengths = offsets[1:] - offsets[:-1]
        divisors = lengths.clamp_min(1).to(grad_output.dtype)
        per_index_grad = per_index_grad / divisors.index_select(0, bag_ids).unsqueeze(1)
    return decomposed_row_wise_embedding_backward(
        per_index_grad,
        indices,
        num_embeddings,
        local_num_embeddings,
        group_name,
        world_size,
        padding_idx=-1,
        scale_grad_by_freq=False,
    )


def _route_table_wise_bags_to_owner(
    indices: torch.Tensor,
    offsets: torch.Tensor,
    owner_rank: int,
    group_name: str,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _check_complete_bags(offsets, indices.shape[0])
    lengths = offsets[1:] - offsets[:-1]
    owner_indices, owner_lengths, cpu_bag_splits, _received_bag_counts = (
        route_table_wise_jagged(
            indices,
            lengths,
            owner_rank,
            group_name,
            world_size,
        )
    )
    owner_offsets = torch.cat((owner_lengths.new_zeros((1,)), owner_lengths)).cumsum(
        dim=0
    )
    return owner_indices, owner_offsets, cpu_bag_splits


def decomposed_table_wise_embedding_bag(
    local_weight: torch.Tensor,
    indices: torch.Tensor,
    offsets: torch.Tensor,
    owner_rank: int,
    rank: int,
    group_name: str,
    world_size: int,
    mode: int,
) -> torch.Tensor:
    """Run a complete pooled table on its owner and return local-batch output."""
    offsets = torch.ops.aten._to_copy.default(offsets, device=indices.device)
    if world_size == 1:
        rows = torch.ops.aten.embedding.default(local_weight, indices)
        return _pool_embedding_rows(rows, offsets, mode)
    owner_indices, owner_offsets, cpu_bag_splits = _route_table_wise_bags_to_owner(
        indices,
        offsets,
        owner_rank,
        group_name,
        world_size,
    )
    if rank == owner_rank:
        owner_rows = torch.ops.aten.embedding.default(local_weight, owner_indices)
        owner_embeddings = _pool_embedding_rows(owner_rows, owner_offsets, mode)
    else:
        owner_embeddings = local_weight.new_empty((0, local_weight.shape[1]))
    return return_table_wise_rows(
        owner_embeddings,
        cpu_bag_splits,
        offsets.shape[0] - 1,
        group_name,
        world_size,
    )


def decomposed_table_wise_embedding_bag_backward(
    grad_output: torch.Tensor,
    local_weight: torch.Tensor,
    indices: torch.Tensor,
    offsets: torch.Tensor,
    num_embeddings: int,
    owner_rank: int,
    rank: int,
    group_name: str,
    world_size: int,
    mode: int,
) -> torch.Tensor:
    """Route pooled gradients to the table owner and form its full gradient."""
    offsets = torch.ops.aten._to_copy.default(offsets, device=indices.device)
    torch._check(
        grad_output.shape[0] == offsets.shape[0] - 1,
        lambda: "EmbeddingBag gradient must have one row per bag.",
    )
    if world_size == 1:
        return decomposed_row_wise_embedding_bag_backward(
            grad_output,
            indices,
            offsets,
            num_embeddings,
            num_embeddings,
            group_name,
            world_size,
            mode,
        )
    owner_indices, owner_offsets, cpu_bag_splits = _route_table_wise_bags_to_owner(
        indices,
        offsets,
        owner_rank,
        group_name,
        world_size,
    )
    owner_grad = route_table_wise_rows(
        grad_output,
        cpu_bag_splits,
        group_name,
        world_size,
    )
    if rank != owner_rank:
        return local_weight.new_zeros(local_weight.shape)
    bag_ids = _bag_ids(owner_offsets, owner_indices.shape[0])
    per_index_grad = owner_grad.index_select(0, bag_ids)
    if mode == 1:
        lengths = owner_offsets[1:] - owner_offsets[:-1]
        divisors = lengths.clamp_min(1).to(owner_grad.dtype)
        per_index_grad = per_index_grad / divisors.index_select(0, bag_ids).unsqueeze(1)
    return torch.ops.aten.embedding_dense_backward.default(
        per_index_grad,
        owner_indices,
        num_embeddings,
        -1,
        False,
    )


def _trace_and_inline_subgraph(
    gm: fx.GraphModule,
    target_node: fx.Node,
    decomp_fn: Callable[..., torch.Tensor],
    example_inputs: List[torch.Tensor],
    input_nodes: List[fx.Node],
) -> None:
    """Trace *decomp_fn* via ``make_fx(symbolic)``, inline the sub-graph
    into *gm* before *target_node*, and replace *target_node* with the
    inlined output."""
    subgraph_gm = make_fx(decomp_fn, tracing_mode="symbolic")(*example_inputs)
    placeholders = [n for n in subgraph_gm.graph.nodes if n.op == "placeholder"]
    val_map: Dict[fx.Node, fx.Node] = dict(zip(placeholders, input_nodes))

    pre_existing = set(gm.graph.nodes)
    with gm.graph.inserting_before(target_node):
        output_value = gm.graph.graph_copy(subgraph_gm.graph, val_map=val_map)

    # Discard bindings from the helper's ShapeEnv so the outer trace creates
    # independent split symbols for each embedding site.
    for node in gm.graph.nodes:
        if node not in pre_existing:
            node.meta.pop("unbacked_bindings", None)

    assert isinstance(output_value, fx.Node)
    target_node.replace_all_uses_with(output_value)
    gm.graph.erase_node(target_node)


# ---------------------------------------------------------------------------
# Region transform
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PooledEmbeddingMeta:
    indices: fx.Node
    offsets: fx.Node
    num_embeddings: int
    local_num_embeddings: int
    embedding_dim: int
    mode: int


def _node_argument(
    node: fx.Node,
    position: int,
    name: str,
    default: object,
) -> object:
    if len(node.args) > position:
        return node.args[position]
    return node.kwargs.get(name, default)


# Same schema; autograd dispatches the forward-only variant when no input
# requires grad (e.g. an eval graph traced under no_grad).
_EMBEDDING_BAG_OPS: Tuple[torch._ops.OpOverload, ...] = (
    torch.ops.aten._embedding_bag.default,
    torch.ops.aten._embedding_bag_forward_only.default,
)


def _embedding_bag_mode(node: fx.Node) -> int:
    """Validate the shared native/TBE pooling contract and return its mode."""
    unsupported = (
        (_node_argument(node, 3, "scale_grad_by_freq", False), "frequency scaling"),
        (_node_argument(node, 5, "sparse", False), "sparse gradients"),
        (_node_argument(node, 6, "per_sample_weights", None) is not None, "weights"),
        (
            not _node_argument(node, 7, "include_last_offset", False),
            "offsets without a final offset",
        ),
        (_node_argument(node, 8, "padding_idx", -1) != -1, "padding_idx"),
    )
    for is_unsupported, behavior in unsupported:
        if is_unsupported:
            raise NotImplementedError(
                f"Sparse embedding_bag transform does not support {behavior}."
            )
    mode = _node_argument(node, 4, "mode", 0)
    if mode not in (0, 1):
        raise NotImplementedError(
            f"Sparse embedding_bag transform does not support mode={mode}."
        )
    return cast(int, mode)


def _matching_embedding_bag_forward(
    node: fx.Node,
    forward_meta: Dict[fx.Node, _PooledEmbeddingMeta],
) -> Optional[Tuple[fx.Node, _PooledEmbeddingMeta]]:
    offset2bag = node.args[3]
    if not (
        isinstance(offset2bag, fx.Node)
        and offset2bag.op == "call_function"
        and offset2bag.target == operator.getitem
    ):
        return None
    if len(offset2bag.args) <= 1 or offset2bag.args[1] != 1:
        raise RuntimeError(
            "SparseEmbeddingTransform requires embedding_bag backward "
            "to use the offset2bag output."
        )
    pooled_node = offset2bag.args[0]
    if not isinstance(pooled_node, fx.Node) or pooled_node not in forward_meta:
        return None
    meta = forward_meta[pooled_node]
    if node.args[1] is not meta.indices or node.args[2] is not meta.offsets:
        raise RuntimeError(
            "EmbeddingBag backward must use the same indices and offsets as its "
            "forward."
        )
    return pooled_node, meta


def _embedding_bag_example_indices(
    indices_node: fx.Node,
    offsets_node: fx.Node,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.zeros(3, dtype=indices_node.meta["val"].dtype, device=device),
        torch.tensor(
            [0, 2, 2, 3],
            dtype=offsets_node.meta["val"].dtype,
            device=device,
        ),
    )


class SparseEmbeddingTransform(RegionTransform):
    """Inline sharded embedding decompositions into the joint
    graph as a post-compile pass.

    The rewrite cannot run during the inner trace — the decomposition's
    ``Tensor.tolist()`` would crash under fake-mode dispatch.  Instead it
    runs in :meth:`finalize_compiled_graph`, which is invoked from
    ``compile_train_step`` after the AOT compile (and ``_finalize_graph``)
    have returned.
    """

    def __init__(
        self,
        embedding_param_fqns: Set[str],
        mesh: DeviceMesh,
        mod: Optional[nn.Module] = None,
        embedding_plan: Optional[EmbeddingShardingPlan] = None,
        forward_only: bool = False,
    ) -> None:
        """
        Args:
            embedding_param_fqns: FQNs (as returned by
                ``nn.Module.named_parameters``) of the sharded embedding
                weights to decompose.
            mesh: ``DeviceMesh`` whose dim-0 process group is used for
                the AllToAll collectives.
            mod: optional reference to the user module.  ``shard_params``
                mutates ``mod`` in place before ``_local_compile`` runs,
                so by ``finalize_compiled_graph`` time
                ``mod.named_parameters`` / ``mod.named_buffers`` return
                the sharded views.  The transform reads these to obtain
                concrete shapes for the inner ``make_fx`` trace —
                required when the outer ``fake_mode`` carries a
                ``ShapeEnv`` (placeholder shapes are then symbolic and
                ``torch.randn`` / ``torch.randint`` reject
                ``SymIntArrayRef``).  When omitted (e.g. CPU unit tests
                without a ``ShapeEnv``), the transform falls back to the
                shapes in ``placeholder.meta["val"]``. Table-wise lowering
                requires the pre-shard module because an empty non-owner
                parameter does not encode the global table size.
            forward_only: the traced graph has no backward (eval /
                inference): only the forward lookups are rewritten.
        """
        self.forward_only: bool = forward_only
        self.embedding_param_fqns: Set[str] = embedding_param_fqns
        self.mesh: DeviceMesh = mesh
        self.world_size: int = mesh.size()
        self.group_name: str = funcol._resolve_group_name((mesh, 0))
        self.mod: Optional[nn.Module] = mod
        self.embedding_plan = embedding_plan
        table_wise_fqns = (
            embedding_param_fqns & set(embedding_plan.table_wise_rank_map)
            if embedding_plan is not None
            else set()
        )
        if mod is None and table_wise_fqns:
            raise ValueError(
                "Table-wise sparse lowering requires the pre-shard module."
            )
        self._global_rows = (
            {fqn: int(mod.get_parameter(fqn).shape[0]) for fqn in embedding_param_fqns}
            if mod is not None
            else {}
        )

    def finalize_compiled_graph(
        self,
        gm: fx.GraphModule,
    ) -> None:
        """Replace supported lookups according to each weight's placement.

        Overrides :meth:`RegionTransform.finalize_compiled_graph` (which
        is a no-op by default).  Identifies sharded weight placeholders
        via ``node.meta["fqn"]`` (set in ``_local_compile``) so the
        lookup is robust to leading-placeholder shifts introduced by
        ``Override.transform`` passes inside ``_finalize_graph``.

        Args:
            gm: the final outer GraphModule returned by the compile
                pipeline (after ``_finalize_graph``).
        """
        if not self.embedding_param_fqns:
            return

        embedding_weight_nodes: Set[fx.Node] = {
            n
            for n in gm.graph.nodes
            if n.op == "placeholder" and n.meta.get("fqn") in self.embedding_param_fqns
        }
        if not embedding_weight_nodes:
            return

        weight_nodes_by_sharding: Dict[EmbeddingShardingType, Set[fx.Node]] = {}
        for node in embedding_weight_nodes:
            fqn = cast(str, node.meta.get("fqn"))
            sharding_type = (
                EmbeddingShardingType.ROW_WISE
                if self.embedding_plan is None
                else self.embedding_plan.sharding_map.get(fqn)
            )
            if sharding_type is None:
                raise ValueError(
                    f"Embedding sharding plan does not classify parameter {fqn!r}."
                )
            weight_nodes_by_sharding.setdefault(sharding_type, set()).add(node)

        supported_sharding_types = {
            EmbeddingShardingType.ROW_WISE,
            EmbeddingShardingType.TABLE_WISE,
        }
        unsupported_sharding_types = (
            set(weight_nodes_by_sharding) - supported_sharding_types
        )
        if unsupported_sharding_types:
            raise NotImplementedError(
                "Sparse embedding transform does not support sharding types "
                f"{sorted(kind.value for kind in unsupported_sharding_types)}."
            )
        row_wise_weight_nodes = weight_nodes_by_sharding.get(
            EmbeddingShardingType.ROW_WISE, set()
        )
        table_wise_weight_nodes = weight_nodes_by_sharding.get(
            EmbeddingShardingType.TABLE_WISE, set()
        )

        concrete_param_tensors = self._concrete_embedding_param_tensors()

        forward_indices_meta = self._rewrite_forward_embeddings(
            gm, row_wise_weight_nodes, concrete_param_tensors
        )
        table_wise_unpooled = [
            node
            for node in gm.graph.nodes
            if node.op == "call_function"
            and node.target == torch.ops.aten.embedding.default
            and node.args[0] in table_wise_weight_nodes
        ]
        if table_wise_unpooled:
            raise NotImplementedError(
                "Table-wise sharding currently supports pooled EBC lookups only."
            )
        pooled_forward_meta = self._rewrite_forward_embedding_bags(
            gm, row_wise_weight_nodes, concrete_param_tensors
        )
        table_wise_forward_meta = self._rewrite_forward_table_wise_embedding_bags(
            gm,
            table_wise_weight_nodes,
            concrete_param_tensors,
        )
        # A forward-only graph (eval / inference) has no embedding backward.
        if not self.forward_only:
            bwd_nodes = self._rewrite_backward_embeddings(gm, forward_indices_meta)
            pooled_bwd_nodes = self._rewrite_backward_embedding_bags(
                gm, pooled_forward_meta
            )
            table_wise_bwd_nodes = self._rewrite_backward_table_wise_embedding_bags(
                gm,
                table_wise_forward_meta,
                concrete_param_tensors,
            )
            if forward_indices_meta:
                assert bwd_nodes, (
                    f"SparseEmbeddingTransform: rewrote "
                    f"{len(forward_indices_meta)} forward embedding(s) but found "
                    f"no corresponding aten.embedding_dense_backward nodes"
                )
            if pooled_forward_meta and pooled_bwd_nodes != set(pooled_forward_meta):
                raise RuntimeError(
                    "SparseEmbeddingTransform requires one backward per forward "
                    "embedding_bag node."
                )
            if table_wise_forward_meta and table_wise_bwd_nodes != set(
                table_wise_forward_meta
            ):
                raise RuntimeError(
                    "SparseEmbeddingTransform requires one backward per table-wise "
                    "embedding_bag node."
                )
        if forward_indices_meta or pooled_forward_meta or table_wise_forward_meta:
            gm.graph.eliminate_dead_code()
            rewritten_pooled = set(pooled_forward_meta) | set(table_wise_forward_meta)
            remaining_pooled = set(gm.graph.nodes).intersection(rewritten_pooled)
            if remaining_pooled:
                raise RuntimeError(
                    "SparseEmbeddingTransform does not support live auxiliary "
                    "embedding_bag outputs."
                )
            gm.graph.lint()
            gm.recompile()

    def _rewrite_forward_table_wise_embedding_bags(
        self,
        gm: fx.GraphModule,
        embedding_weight_nodes: Set[fx.Node],
        concrete_param_tensors: Dict[str, torch.Tensor],
    ) -> Dict[fx.Node, _PooledEmbeddingMeta]:
        result: Dict[fx.Node, _PooledEmbeddingMeta] = {}
        if not embedding_weight_nodes:
            return result
        coordinate = self.mesh.get_coordinate()
        if coordinate is None:
            raise RuntimeError("Current rank is not part of the embedding device mesh.")
        rank = coordinate[0]
        for pooled_node in list(gm.graph.nodes):
            if (
                pooled_node.op != "call_function"
                or pooled_node.target not in _EMBEDDING_BAG_OPS
                or pooled_node.args[0] not in embedding_weight_nodes
            ):
                continue
            weight_node = cast(fx.Node, pooled_node.args[0])
            indices_node = cast(fx.Node, pooled_node.args[1])
            offsets_node = cast(fx.Node, pooled_node.args[2])
            fqn = cast(str, weight_node.meta["fqn"])
            mode = _embedding_bag_mode(pooled_node)
            _local_rows, embedding_dim, weight_dtype = self._weight_concrete_meta(
                weight_node,
                concrete_param_tensors,
            )
            device = self._weight_device(weight_node, concrete_param_tensors)
            num_embeddings = self._global_rows[fqn]
            owner_rank = cast(
                EmbeddingShardingPlan, self.embedding_plan
            ).table_wise_owner(fqn)
            output_nodes = [
                user
                for user in pooled_node.users
                if user.op == "call_function"
                and user.target == operator.getitem
                and user.args[1] == 0
            ]
            if len(output_nodes) != 1:
                raise NotImplementedError(
                    "Sparse embedding transform requires one embedding_bag output."
                )
            local_rows = num_embeddings if rank == owner_rank else 0
            example_rows = min(local_rows, _SYMBOLIC_TRACE_PLACEHOLDER_DIM)
            example_indices, example_offsets = _embedding_bag_example_indices(
                indices_node,
                offsets_node,
                device,
            )
            _trace_and_inline_subgraph(
                gm,
                output_nodes[0],
                decomp_fn=partial(
                    decomposed_table_wise_embedding_bag,
                    owner_rank=owner_rank,
                    rank=rank,
                    group_name=self.group_name,
                    world_size=self.world_size,
                    mode=mode,
                ),
                example_inputs=[
                    torch.randn(
                        (example_rows, embedding_dim),
                        dtype=weight_dtype,
                        device=device,
                    ),
                    example_indices,
                    example_offsets,
                ],
                input_nodes=[weight_node, indices_node, offsets_node],
            )
            result[pooled_node] = _PooledEmbeddingMeta(
                indices=indices_node,
                offsets=offsets_node,
                num_embeddings=num_embeddings,
                local_num_embeddings=local_rows,
                embedding_dim=embedding_dim,
                mode=mode,
            )
        return result

    def _rewrite_backward_table_wise_embedding_bags(
        self,
        gm: fx.GraphModule,
        forward_meta: Dict[fx.Node, _PooledEmbeddingMeta],
        concrete_param_tensors: Dict[str, torch.Tensor],
    ) -> Set[fx.Node]:
        matched_forwards: Set[fx.Node] = set()
        if not forward_meta:
            return matched_forwards
        coordinate = self.mesh.get_coordinate()
        if coordinate is None:
            raise RuntimeError("Current rank is not part of the embedding device mesh.")
        rank = coordinate[0]
        for node in list(gm.graph.nodes):
            if (
                node.op != "call_function"
                or node.target != torch.ops.aten._embedding_bag_backward.default
            ):
                continue
            match = _matching_embedding_bag_forward(node, forward_meta)
            if match is None:
                continue
            pooled_node, meta = match
            if pooled_node in matched_forwards:
                raise RuntimeError(
                    "SparseEmbeddingTransform requires exactly one backward "
                    "per table-wise embedding_bag forward."
                )
            weight_node = cast(fx.Node, pooled_node.args[0])
            grad_output = cast(fx.Node, node.args[0])
            fqn = cast(str, weight_node.meta["fqn"])
            owner_rank = cast(
                EmbeddingShardingPlan, self.embedding_plan
            ).table_wise_owner(fqn)
            _local_rows, _embedding_dim, weight_dtype = self._weight_concrete_meta(
                weight_node,
                concrete_param_tensors,
            )
            device = self._weight_device(weight_node, concrete_param_tensors)
            local_rows = meta.num_embeddings if rank == owner_rank else 0
            example_rows = min(local_rows, _SYMBOLIC_TRACE_PLACEHOLDER_DIM)
            example_indices, example_offsets = _embedding_bag_example_indices(
                meta.indices,
                meta.offsets,
                device,
            )
            _trace_and_inline_subgraph(
                gm,
                node,
                decomp_fn=partial(
                    decomposed_table_wise_embedding_bag_backward,
                    num_embeddings=meta.num_embeddings,
                    owner_rank=owner_rank,
                    rank=rank,
                    group_name=self.group_name,
                    world_size=self.world_size,
                    mode=meta.mode,
                ),
                example_inputs=[
                    torch.randn(
                        (_SYMBOLIC_TRACE_PLACEHOLDER_DIM, meta.embedding_dim),
                        dtype=grad_output.meta["val"].dtype,
                        device=device,
                    ),
                    torch.randn(
                        (example_rows, meta.embedding_dim),
                        dtype=weight_dtype,
                        device=device,
                    ),
                    example_indices,
                    example_offsets,
                ],
                input_nodes=[
                    grad_output,
                    weight_node,
                    meta.indices,
                    meta.offsets,
                ],
            )
            matched_forwards.add(pooled_node)
        return matched_forwards

    def _concrete_embedding_param_tensors(self) -> Dict[str, torch.Tensor]:
        """Resolve FQN→concrete sharded tensor for each embedding weight.

        Reads from ``self.mod`` (mutated in place by ``shard_params``
        before ``_local_compile``).  Returns ``{}`` when ``mod`` was not
        provided — callers without a ``ShapeEnv`` (e.g. CPU unit tests)
        can fall back to symbolic ``meta["val"]`` shapes.
        """
        mod = self.mod
        if mod is None:
            return {}
        params_buffers: Dict[str, torch.Tensor] = {
            **dict(mod.named_parameters(remove_duplicate=False)),
            **dict(mod.named_buffers(remove_duplicate=False)),
        }
        return {
            fqn: tensor
            for fqn, tensor in params_buffers.items()
            if fqn in self.embedding_param_fqns
        }

    def _rewrite_forward_embeddings(
        self,
        gm: fx.GraphModule,
        embedding_weight_nodes: Set[fx.Node],
        concrete_param_tensors: Dict[str, torch.Tensor],
    ) -> Dict[fx.Node, Tuple[int, int, int]]:
        """Inline ``decomposed_row_wise_embedding`` over each
        ``aten.embedding.default`` node whose weight is a sharded
        embedding placeholder.  Returns a dict mapping each rewritten
        node's original ``indices`` FX node to ``(local_V,
        num_embeddings, embedding_dim)`` so the backward rewrite can
        locate the matching ``aten.embedding_dense_backward`` nodes and
        reconstruct concrete grad-output shapes.

        Concrete weight shapes are required because the outer
        ``fake_mode`` under HybridParallel attaches a ``ShapeEnv``,
        which symbolicizes every placeholder's ``meta["val"]``; the
        symbolic shapes cannot be passed to ``torch.randn`` /
        ``torch.randint`` when building example inputs for the inner
        ``make_fx`` trace.  When available, use the concrete shape from
        ``concrete_param_tensors[fqn]``.  For the indices placeholder
        we use a small constant rank-preserving shape — the symbolic
        trace re-symbolicizes the shape on its own.
        """
        embedding_nodes: List[fx.Node] = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target == torch.ops.aten.embedding.default
            and n.args[0] in embedding_weight_nodes
        ]

        forward_indices_meta: Dict[fx.Node, Tuple[int, int, int]] = {}
        for emb_node in embedding_nodes:
            weight_node = emb_node.args[0]
            indices_node = emb_node.args[1]
            assert isinstance(weight_node, fx.Node)
            assert isinstance(indices_node, fx.Node)

            local_V, embedding_dim, weight_dtype = self._weight_concrete_meta(
                weight_node, concrete_param_tensors
            )
            fqn = cast(str, weight_node.meta["fqn"])
            num_embeddings = self._global_rows.get(
                fqn,
                local_V * self.world_size,
            )

            indices_val = indices_node.meta["val"]
            indices_ndim = len(indices_val.shape)

            forward_indices_meta[indices_node] = (
                local_V,
                num_embeddings,
                embedding_dim,
            )

            _trace_and_inline_subgraph(
                gm,
                emb_node,
                decomp_fn=partial(
                    decomposed_row_wise_embedding,
                    num_embeddings=num_embeddings,
                    group_name=self.group_name,
                    world_size=self.world_size,
                ),
                example_inputs=[
                    torch.randn(
                        (local_V, embedding_dim), dtype=weight_dtype, device="cpu"
                    ),
                    torch.randint(
                        0,
                        num_embeddings,
                        (_SYMBOLIC_TRACE_PLACEHOLDER_DIM,) * indices_ndim,
                        device="cpu",
                    ),
                ],
                input_nodes=[weight_node, indices_node],
            )
        return forward_indices_meta

    def _weight_concrete_meta(
        self,
        weight_node: fx.Node,
        concrete_param_tensors: Dict[str, torch.Tensor],
    ) -> Tuple[int, int, torch.dtype]:
        """Resolve ``(local_V, embedding_dim, dtype)`` for *weight_node*.

        Prefer the concrete sharded tensor from
        ``concrete_param_tensors[fqn]``; fall back to ``meta["val"]`` if
        no concrete tensor is available (callers without ShapeEnv only).
        """
        fqn = weight_node.meta.get("fqn")
        concrete = concrete_param_tensors.get(fqn) if fqn is not None else None
        if concrete is not None:
            return int(concrete.shape[0]), int(concrete.shape[1]), concrete.dtype
        weight_val = weight_node.meta["val"]
        return (
            int(weight_val.shape[0]),
            int(weight_val.shape[1]),
            weight_val.dtype,
        )

    def _rewrite_backward_embeddings(
        self,
        gm: fx.GraphModule,
        forward_indices_meta: Dict[fx.Node, Tuple[int, int, int]],
    ) -> List[fx.Node]:
        """Inline ``decomposed_row_wise_embedding_backward`` over each
        ``aten.embedding_dense_backward.default`` node whose ``indices``
        argument matches a previously rewritten forward embedding.
        Returns the list of rewritten backward nodes.

        Uses concrete ``embedding_dim`` from ``forward_indices_meta`` to
        avoid reading symbolic shapes from ``meta["val"]``.
        """
        bwd_nodes: List[fx.Node] = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target == torch.ops.aten.embedding_dense_backward.default
            and n.args[1] in forward_indices_meta
        ]

        for bwd_node in bwd_nodes:
            grad_output_node = bwd_node.args[0]
            indices_node = bwd_node.args[1]
            assert isinstance(grad_output_node, fx.Node)
            assert isinstance(indices_node, fx.Node)
            padding_idx = cast(int, bwd_node.args[3]) if len(bwd_node.args) > 3 else -1
            scale_grad_by_freq = (
                cast(bool, bwd_node.args[4]) if len(bwd_node.args) > 4 else False
            )
            local_V, num_embeddings, embedding_dim = forward_indices_meta[indices_node]
            grad_val = grad_output_node.meta["val"]
            indices_val = indices_node.meta["val"]
            indices_ndim = len(indices_val.shape)

            _trace_and_inline_subgraph(
                gm,
                bwd_node,
                decomp_fn=partial(
                    decomposed_row_wise_embedding_backward,
                    num_embeddings=num_embeddings,
                    local_num_embeddings=local_V,
                    group_name=self.group_name,
                    world_size=self.world_size,
                    padding_idx=padding_idx,
                    scale_grad_by_freq=scale_grad_by_freq,
                ),
                example_inputs=[
                    torch.randn(
                        (_SYMBOLIC_TRACE_PLACEHOLDER_DIM,) * indices_ndim
                        + (embedding_dim,),
                        dtype=grad_val.dtype,
                        device="cpu",
                    ),
                    torch.randint(
                        0,
                        num_embeddings,
                        (_SYMBOLIC_TRACE_PLACEHOLDER_DIM,) * indices_ndim,
                        device="cpu",
                    ),
                ],
                input_nodes=[grad_output_node, indices_node],
            )
        return bwd_nodes

    def _rewrite_forward_embedding_bags(
        self,
        gm: fx.GraphModule,
        embedding_weight_nodes: Set[fx.Node],
        concrete_param_tensors: Dict[str, torch.Tensor],
    ) -> Dict[fx.Node, _PooledEmbeddingMeta]:
        pooled_nodes = [
            node
            for node in gm.graph.nodes
            if node.op == "call_function"
            and node.target in _EMBEDDING_BAG_OPS
            and node.args[0] in embedding_weight_nodes
        ]
        result: Dict[fx.Node, _PooledEmbeddingMeta] = {}
        for pooled_node in pooled_nodes:
            result[pooled_node] = self._rewrite_forward_embedding_bag(
                gm,
                pooled_node,
                concrete_param_tensors,
            )
        return result

    def _rewrite_forward_embedding_bag(
        self,
        gm: fx.GraphModule,
        pooled_node: fx.Node,
        concrete_param_tensors: Dict[str, torch.Tensor],
    ) -> _PooledEmbeddingMeta:
        weight_node = pooled_node.args[0]
        indices_node = pooled_node.args[1]
        offsets_node = pooled_node.args[2]
        assert isinstance(weight_node, fx.Node)
        assert isinstance(indices_node, fx.Node)
        assert isinstance(offsets_node, fx.Node)
        mode = _embedding_bag_mode(pooled_node)
        local_rows, embedding_dim, weight_dtype = self._weight_concrete_meta(
            weight_node, concrete_param_tensors
        )
        fqn = cast(str, weight_node.meta["fqn"])
        num_embeddings = self._global_rows.get(
            fqn,
            local_rows * self.world_size,
        )
        device = self._weight_device(weight_node, concrete_param_tensors)
        output_nodes = [
            user
            for user in pooled_node.users
            if user.op == "call_function"
            and user.target == operator.getitem
            and user.args[1] == 0
        ]
        if len(output_nodes) != 1:
            raise NotImplementedError(
                "Sparse embedding transform requires one embedding_bag output."
            )
        example_indices, example_offsets = _embedding_bag_example_indices(
            indices_node,
            offsets_node,
            device,
        )
        _trace_and_inline_subgraph(
            gm,
            output_nodes[0],
            decomp_fn=partial(
                decomposed_row_wise_embedding_bag,
                num_embeddings=num_embeddings,
                group_name=self.group_name,
                world_size=self.world_size,
                mode=mode,
            ),
            example_inputs=[
                torch.randn(
                    (min(local_rows, _SYMBOLIC_TRACE_PLACEHOLDER_DIM), embedding_dim),
                    dtype=weight_dtype,
                    device=device,
                ),
                example_indices,
                example_offsets,
            ],
            input_nodes=[weight_node, indices_node, offsets_node],
        )
        return _PooledEmbeddingMeta(
            indices=indices_node,
            offsets=offsets_node,
            num_embeddings=num_embeddings,
            local_num_embeddings=local_rows,
            embedding_dim=embedding_dim,
            mode=mode,
        )

    def _weight_device(
        self,
        weight_node: fx.Node,
        concrete_param_tensors: Dict[str, torch.Tensor],
    ) -> torch.device:
        fqn = weight_node.meta.get("fqn")
        concrete = concrete_param_tensors.get(fqn) if fqn is not None else None
        if concrete is not None:
            return concrete.device
        return weight_node.meta["val"].device

    def _rewrite_backward_embedding_bags(
        self,
        gm: fx.GraphModule,
        forward_meta: Dict[fx.Node, _PooledEmbeddingMeta],
    ) -> Set[fx.Node]:
        matched_forwards: Set[fx.Node] = set()
        for node in list(gm.graph.nodes):
            if (
                node.op != "call_function"
                or node.target != torch.ops.aten._embedding_bag_backward.default
            ):
                continue
            match = _matching_embedding_bag_forward(node, forward_meta)
            if match is None:
                continue
            pooled_node, meta = match
            if pooled_node in matched_forwards:
                raise RuntimeError(
                    "SparseEmbeddingTransform requires exactly one backward per "
                    "forward embedding_bag node."
                )
            grad_output = node.args[0]
            assert isinstance(grad_output, fx.Node)
            example_indices, example_offsets = _embedding_bag_example_indices(
                meta.indices,
                meta.offsets,
                grad_output.meta["val"].device,
            )
            _trace_and_inline_subgraph(
                gm,
                node,
                decomp_fn=partial(
                    decomposed_row_wise_embedding_bag_backward,
                    num_embeddings=meta.num_embeddings,
                    local_num_embeddings=meta.local_num_embeddings,
                    group_name=self.group_name,
                    world_size=self.world_size,
                    mode=meta.mode,
                ),
                example_inputs=[
                    torch.randn(
                        (_SYMBOLIC_TRACE_PLACEHOLDER_DIM, meta.embedding_dim),
                        dtype=grad_output.meta["val"].dtype,
                        device=grad_output.meta["val"].device,
                    ),
                    example_indices,
                    example_offsets,
                ],
                input_nodes=[grad_output, meta.indices, meta.offsets],
            )
            matched_forwards.add(pooled_node)
        return matched_forwards
