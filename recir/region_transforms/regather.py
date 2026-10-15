# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""RegatherTransform — per-op FSDP weight regathering via activation checkpointing."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable, Optional, Sequence, Set

import torch
import torch.fx as fx
import torch.nn as nn
import torch.utils._pytree as pytree
from recir.region_transforms.base import RegionTransform
from recir.region_transforms.core import _extract_subgraph, _replace_region
from recir.region_transforms.recompute import RecomputeTransform
from recir.utils.graph import OP


_RINGSHIFT_EMBEDDING_SCHEMA = "recir::ringshift_embedding"


def _is_ringshift_embedding_region(region_nodes: set[fx.Node]) -> bool:
    """Whether the region's compute is the owner-local ring embedding op."""
    return any(
        getattr(getattr(node.target, "_schema", None), "name", None)
        == _RINGSHIFT_EMBEDDING_SCHEMA
        for node in region_nodes
    )


def _collect_chain_regions(
    param_node: fx.Node,
) -> list[set[fx.Node]]:
    """Walk users of *param_node* and collect per-branch regions.

    A region is the set of nodes on one usage path from the parameter through
    chain ops to (and including) the first compute consumer.  The parameter
    placeholder itself is excluded so that it remains a region *input* (not a
    member), avoiding issues with subgraph extraction and replacement.

    A node is classified as a "chain op" if ALL its ``fx.Node`` inputs come
    from the current chain path (including the parameter itself).  This
    criterion automatically handles ATen ops (t, view, clone, detach) *and*
    DTensor collective ops (all_gather_into_tensor, wait_tensor) without an
    explicit whitelist.

    Example graph::

        param → t() ─┬→ addmm (fwd)
                      └→ mm    (bwd)

    Produces two regions: ``{t, addmm}`` and ``{t, mm}``.
    (The ``t`` node is duplicated across regions so each region is
    self-contained.)

    .. note::

       **TODO:** When regions share chain nodes (e.g. ``t`` appears in
       both ``{t, addmm}`` and ``{t, mm}``), sequential replacement
       causes the shared node to become a checkpoint output saved by
       subsequent regions — the full weight stays in memory, defeating
       the memory saving.  The fix is to **clone shared chain nodes**
       before replacement so each region has its own copy:
       ``{t_clone1, addmm}`` and ``{t_clone2, mm}``.  This keeps
       regions as simple linear single-output chains whose only saved
       input is the small sharded parameter.
    """
    regions: list[set[fx.Node]] = []
    chain_set: set[fx.Node] = {param_node}

    def _walk(
        node: fx.Node,
        path: list[fx.Node],
    ) -> None:
        """DFS through chain-op users, emitting a region at each compute consumer."""
        for user in node.users:
            if not isinstance(user, fx.Node):
                continue
            if user.op == "output":
                continue
            has_outside_input = False
            for arg in pytree.arg_tree_leaves(*user.args, **user.kwargs):
                if (
                    isinstance(arg, fx.Node)
                    and arg not in chain_set
                    # get_attr nodes are graph constants (e.g. DeviceMesh,
                    # process group metadata) stored as GraphModule attributes.
                    # They are always available and don't consume checkpoint
                    # memory, so they should not terminate the chain.  Without
                    # this exclusion, DTensor collective ops (which take
                    # DeviceMesh as a get_attr input) would be misclassified
                    # as compute consumers, stopping the chain prematurely.
                    and arg.op != "get_attr"
                ):
                    has_outside_input = True
                    break
            if not has_outside_input:
                chain_set.add(user)
                _walk(user, path + [user])
            else:
                regions.append({*path, user})

    _walk(param_node, [])
    return regions


class RegatherTransform(RegionTransform):
    """Per-op FSDP: recompute the allgather + reshape chain in backward."""

    def __init__(
        self,
        skip_param_fqns: Optional[Set[str]] = None,
        forward_root_fqns: Optional[Set[str]] = None,
    ) -> None:
        self.skip_param_fqns: Set[str] = skip_param_fqns or set()
        self.forward_root_fqns: Optional[Set[str]] = (
            None if forward_root_fqns is None else set(forward_root_fqns)
        )

    def select_forward_root_fqns(
        self,
        called_modules: OrderedDict[
            str, tuple[nn.Module, tuple[Any, ...], dict[str, Any]]
        ],
    ) -> Optional[set[str]]:
        if self.forward_root_fqns is None:
            return None
        missing = self.forward_root_fqns - called_modules.keys()
        if missing:
            raise RuntimeError(
                f"Requested regather forward roots were not called: {sorted(missing)}"
            )
        return set(self.forward_root_fqns)

    @staticmethod
    def skip_fqns_for_recompute(
        mod: nn.Module,
        region_transforms: Sequence[RegionTransform],
        base_skip: Optional[Set[str]] = None,
    ) -> Set[str]:
        """Params to exclude from per-op regather: the caller base (e.g. embedding
        params owned by the row-wise rewrite) plus any param already covered by a
        RecomputeTransform. That per-layer checkpoint already re-runs the covered
        layer's forward -- including the weight all-gather -- in the backward, so
        nesting a per-op regather inside it would gather the weight a redundant
        3rd time (fwd + layer-recompute + regather-recompute). Skipping keeps 2
        gathers/weight (fwd + one shared backward recompute), matching JAX, at no
        extra memory (the enclosing recompute already frees the forward gather)."""
        skip: Set[str] = set(base_skip or ())
        recompute = [t for t in region_transforms if isinstance(t, RecomputeTransform)]
        if recompute:
            all_fqns = [fqn for fqn, _ in mod.named_parameters(remove_duplicate=False)]
            for rt in recompute:
                skip |= {
                    fqn
                    for fqn in all_fqns
                    if any(
                        fqn == s or fqn.startswith(s + ".") for s in rt.submodule_fqns
                    )
                }
        return skip

    def transform_graph(
        self,
        gm: fx.GraphModule,
        called_modules: OrderedDict[
            str, tuple[nn.Module, tuple[Any, ...], dict[str, Any]]
        ],
    ) -> None:
        if gm.graph.eliminate_dead_code():
            gm.recompile()
        root_mod, _root_args, _root_kwargs = next(iter(called_modules.values()))
        param_fqns = [fqn for fqn, _ in root_mod.named_parameters()]
        skip_indices = {
            i for i, fqn in enumerate(param_fqns) if fqn in self.skip_param_fqns
        }
        placeholders = [n for n in gm.graph.nodes if n.op == OP.PLACEHOLDER]
        param_placeholders = placeholders[: len(param_fqns)]
        all_regions: list[set[fx.Node]] = []
        seen_nodes: set[fx.Node] = set()
        for i, node in enumerate(param_placeholders):
            if i in skip_indices:
                continue
            for region_nodes in _collect_chain_regions(node):
                # The registered backward saves only ids, lengths, and the local
                # weight shard. Recomputing forward adds a second ring exchange.
                if _is_ringshift_embedding_region(region_nodes):
                    continue
                unique = region_nodes - seen_nodes
                if unique:
                    all_regions.append(unique)
                    seen_nodes.update(unique)
        for region_nodes in all_regions:
            region_gm = _extract_subgraph(gm, region_nodes)
            wrapped_fn = self._wrap_checkpoint(region_gm)
            _replace_region(gm, region_nodes, wrapped_fn)

    @staticmethod
    def _wrap_checkpoint(region_gm: fx.GraphModule) -> Callable[..., Any]:
        def _apply(*inputs: Any) -> Any:
            return torch.utils.checkpoint.checkpoint(
                region_gm,
                *inputs,
                use_reentrant=True,
            )

        return _apply
