# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""SparseEmbeddingTransform — replace ``aten.embedding`` and
``aten.embedding_dense_backward`` nodes whose weight is a row-wise
sharded embedding parameter with the corresponding decomposed AllToAll
sub-graphs.
"""

from __future__ import annotations

from functools import partial
from typing import Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.distributed._functional_collectives as funcol
import torch.fx as fx
import torch.nn as nn
from functorch import make_fx
from recir.region_transforms.base import RegionTransform
from torch.distributed.tensor import DeviceMesh


# Placeholder dim used for example inputs to ``make_fx(tracing_mode="symbolic")``
# in the embedding rewrite path.  PyTorch specializes 0 and 1 (treated as
# constants, not symbols), so we need a value >= 2 to ensure the inlined
# subgraph uses real symbolic dims.  Picked small but non-power-of-two to
# avoid any divisibility-based specialization elsewhere.
_SYMBOLIC_TRACE_PLACEHOLDER_DIM: int = 3


def _declare_split_invariants(
    split_list: List[int], total: Optional[int] = None
) -> None:
    """Constrain the unbacked SymInts produced by ``Tensor.tolist()``."""
    for s in split_list:
        torch._check_is_size(s)
    if total is not None:
        torch._check(sum(split_list) == total)


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
    local_rows = num_embeddings // world_size
    ranks = flat_indices // local_rows
    local_indices = flat_indices % local_rows
    sort_order = torch.argsort(ranks, stable=True)
    sorted_local_indices = local_indices[sort_order]
    unsort_order = torch.argsort(sort_order)

    split_sizes = torch.stack([(ranks == r).sum() for r in range(world_size)])
    input_split_list = split_sizes.tolist()
    _declare_split_invariants(input_split_list, flat_indices.shape[0])

    # Step 2: Index AllToAll
    # Exchange split sizes (equal-split, each rank sends world_size ints)
    output_splits_tensor = torch.ops._c10d_functional.all_to_all_single(
        split_sizes, [1] * world_size, [1] * world_size, group_name
    )
    output_splits_tensor = torch.ops._c10d_functional.wait_tensor(output_splits_tensor)
    output_split_list = output_splits_tensor.tolist()
    # Peer counts: non-negative, but not relatable to any local size.
    _declare_split_invariants(output_split_list, None)

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
        group_name: process-group name for collectives.
        world_size: number of ranks in the group.
        padding_idx: padding index (passed through to local backward).
        scale_grad_by_freq: frequency scaling flag (passed through).
    """
    if world_size == 1:
        return torch.ops.aten.embedding_dense_backward.default(
            grad_output, indices, num_embeddings, padding_idx, scale_grad_by_freq
        )

    local_rows = num_embeddings // world_size
    D = grad_output.shape[-1]

    # Step 1: Bucketize (identical to forward)
    flat_indices = indices.reshape(-1)
    ranks = flat_indices // local_rows
    local_indices = flat_indices % local_rows
    sort_order = torch.argsort(ranks, stable=True)
    sorted_local_indices = local_indices[sort_order]

    flat_grad = grad_output.reshape(-1, D)
    sorted_grad = flat_grad[sort_order]

    split_sizes = torch.stack([(ranks == r).sum() for r in range(world_size)])
    input_split_list = split_sizes.tolist()
    _declare_split_invariants(input_split_list, flat_indices.shape[0])

    # Step 2: Size AllToAll
    output_splits_tensor = torch.ops._c10d_functional.all_to_all_single(
        split_sizes, [1] * world_size, [1] * world_size, group_name
    )
    output_splits_tensor = torch.ops._c10d_functional.wait_tensor(output_splits_tensor)
    output_split_list = output_splits_tensor.tolist()
    # Peer counts: non-negative, but not relatable to any local size.
    _declare_split_invariants(output_split_list, None)

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
        received_grad, received_indices, local_rows, padding_idx, scale_grad_by_freq
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


class SparseEmbeddingTransform(RegionTransform):
    """Inline the row-wise sharded embedding decomposition into the joint
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
    ) -> None:
        """
        Args:
            embedding_param_fqns: FQNs (as returned by
                ``nn.Module.named_parameters``) of the row-wise sharded
                embedding weights to decompose.
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
                shapes in ``placeholder.meta["val"]``.
        """
        self.embedding_param_fqns: Set[str] = embedding_param_fqns
        self.mesh: DeviceMesh = mesh
        self.world_size: int = mesh.size()
        self.group_name: str = funcol._resolve_group_name((mesh, 0))
        self.mod: Optional[nn.Module] = mod

    def finalize_compiled_graph(
        self,
        gm: fx.GraphModule,
    ) -> None:
        """Replace ``aten.embedding`` / ``aten.embedding_dense_backward``
        nodes on sharded embedding weights with decomposed AllToAll
        sub-graphs, in place.

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

        concrete_param_tensors = self._concrete_embedding_param_tensors()

        forward_indices_meta = self._rewrite_forward_embeddings(
            gm, embedding_weight_nodes, concrete_param_tensors
        )
        bwd_nodes = self._rewrite_backward_embeddings(gm, forward_indices_meta)

        if forward_indices_meta:
            assert bwd_nodes, (
                f"SparseEmbeddingTransform: rewrote "
                f"{len(forward_indices_meta)} forward embedding(s) but found "
                f"no corresponding aten.embedding_dense_backward nodes"
            )
            gm.graph.lint()
            gm.recompile()

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
            num_embeddings = local_V * self.world_size

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
            padding_idx = bwd_node.args[3] if len(bwd_node.args) > 3 else -1
            scale_grad_by_freq = bwd_node.args[4] if len(bwd_node.args) > 4 else False
            _local_V, num_embeddings, embedding_dim = forward_indices_meta[indices_node]
            grad_val = grad_output_node.meta["val"]
            indices_val = indices_node.meta["val"]
            indices_ndim = len(indices_val.shape)

            _trace_and_inline_subgraph(
                gm,
                bwd_node,
                decomp_fn=partial(
                    decomposed_row_wise_embedding_backward,
                    num_embeddings=num_embeddings,
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
