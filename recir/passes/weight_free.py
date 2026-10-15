# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import operator
from typing import Any, Dict, Set

import torch
from recir.distributed.data_parallel import NodeType
from torch import fx
from torch.fx import GraphModule

aten: torch._ops._OpNamespace = torch.ops.aten

# View ops that share the underlying storage with their input tensor.
# t, view, and detach are always views. reshape is a view when the input is
# contiguous (otherwise it copies). In practice, wait_tensor outputs are
# contiguous because all_gather_into_tensor allocates via torch.empty(), but
# this is an implementation detail of NCCL/functional collectives, not a
# formal API guarantee. If reshape ever copies, we mistakenly treat it as
# storage-sharing and delay free_storage(wait_tensor) until after the copy's
# consumers finish — but this is still safe because free_storage only frees
# wait_tensor's storage, which the independent copy doesn't depend on. We
# just hold wait_tensor's storage slightly longer than necessary.
#
# split/narrow/slice produce views too. They appear in Shard(dim>0) →
# Replicate redistribution, where the allgather output is split along dim 0
# then cat'd along the shard dim. operator.getitem extracts individual views
# from the tuple returned by split/chunk.
_VIEW_OPS: Set[Any] = {
    aten.t.default,
    aten.view.default,
    aten.reshape.default,
    aten.detach.default,
    aten.split.Tensor,
    aten.narrow.default,
    aten.slice.Tensor,
    operator.getitem,
}


def _is_param_allgather_wait(node: fx.Node) -> bool:
    """Return True if *node* is a wait_tensor for a PARAM allgather."""
    if (
        node.op != "call_function"
        or node.target != torch.ops._c10d_functional.wait_tensor.default
    ):
        return False
    if node.meta.get("node_types", []) != [NodeType.PARAM]:
        return False
    ag_node = node.args[0]
    return (
        isinstance(ag_node, fx.Node)
        and ag_node.op == "call_function"
        and "all_gather_into_tensor" in str(ag_node.target)
    )


def _collect_storage_sharing_nodes(node: fx.Node) -> Set[fx.Node]:
    """BFS from *node* through view ops to find all nodes sharing its storage."""
    storage_sharing: Set[fx.Node] = {node}
    queue = [node]
    while queue:
        current = queue.pop()
        for user in current.users:
            if user.target in _VIEW_OPS and user not in storage_sharing:
                storage_sharing.add(user)
                queue.append(user)
    return storage_sharing


def _find_last_consumer(
    wait_node: fx.Node,
    storage_sharing: Set[fx.Node],
    node_order: Dict[fx.Node, int],
) -> fx.Node:
    """Find the topologically last node that must finish before we can free.

    This includes every storage-sharing node (view ops need valid storage to
    execute) AND every compute consumer of those nodes.
    """
    last_node = wait_node
    best_order = node_order[wait_node]
    for ss_node in storage_sharing:
        order = node_order.get(ss_node, -1)
        if order > best_order:
            best_order = order
            last_node = ss_node
        for user in ss_node.users:
            if user not in storage_sharing:
                order = node_order.get(user, -1)
                if order > best_order:
                    best_order = order
                    last_node = user
    return last_node


def insert_weight_free_nodes(gm: GraphModule) -> None:
    """Insert ``recir::free_storage`` calls to release allgathered PARAM weights
    immediately after their last consumer, freeing GPU memory eagerly.

    This function runs after the partitioner has inserted allgather collectives
    and after ``_dedup_collectives`` has removed duplicates.  At this point each
    compute use of a parameter weight has its own ``all_gather_into_tensor`` +
    ``wait_tensor`` pair in the graph (thanks to ``_clone_param_chains_for_backward``).

    Steps:

    1. Find each ``wait_tensor`` node that is a PARAM allgather result.
    2. BFS from that ``wait_tensor`` through view ops (t, view, reshape, detach)
       to find all nodes sharing the same underlying GPU storage.
    3. Find the topologically last node that either is a storage-sharing node
       or is a compute consumer of one — this is the safe insertion point.
    4. Insert ``free_storage(wait_tensor)`` right after that point.

    Example (before)::

        W_shard
          ├→ all_gather_1 → wait_tensor_1 → t → addmm (forward)
          └→ all_gather_2 → wait_tensor_2 → t' → t'' → mm (backward)

    Example (after)::

        W_shard
          ├→ all_gather_1 → wait_tensor_1 → t → addmm → free_storage(wait_tensor_1)
          └→ all_gather_2 → wait_tensor_2 → t' → t'' → mm → free_storage(wait_tensor_2)

    Each full-size weight is freed right after its consumer, so only one full
    weight is in GPU memory at a time.
    """
    graph = gm.graph

    # Build topological ordering for fast last-consumer lookup.
    node_order: Dict[fx.Node, int] = {}
    for idx, node in enumerate(graph.nodes):
        node_order[node] = idx

    for node in list(graph.nodes):
        # Filter to wait_tensor nodes that are allgathered full-size parameter
        # weights. This is the tensor whose storage we want to eagerly free,
        # since it's a temporary full copy reconstructed from a shard and can
        # be large (e.g., 4MB for a 1024×1024 float32 weight).
        if not _is_param_allgather_wait(node):
            continue

        storage_sharing = _collect_storage_sharing_nodes(node)
        last_node = _find_last_consumer(node, storage_sharing, node_order)

        # Insert free_storage after the last relevant node.
        with graph.inserting_after(last_node):
            graph.call_function(torch.ops.recir.free_storage.default, args=(node,))

    gm.recompile()
