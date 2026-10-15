# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict, List, Optional, Set

import torch
import torch.fx as fx
import torch.utils._pytree as pytree
from torch import Tensor
from torch.distributed.tensor import DeviceMesh, Replicate, Shard
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._ops._view_ops import dim_maps, DimSpec, InputDim
from torch.distributed.tensor.placement_types import Partial as _Partial


aten: torch._ops._OpNamespace = torch.ops.aten

# Sentinel batch-dim values used by ``BatchDimAnalyzer``:
#   REPLICATED: tensor has no batch dim and is fully replicated.
#   PARTIAL:    tensor is a partial (un-reduced) result across the shard mesh
#               dim; ``compute_act_spec`` converts this into a ``_Partial()``
#               placement so a downstream allreduce can materialise the global
#               value.
# Non-negative values are real (output) batch-dim indices.
REPLICATED: int = -1
PARTIAL: int = -2


# Registry of op-specific batch-dim propagation rules consulted *before* the
# fallback shape-match heuristic in ``BatchDimAnalyzer.compute_batch_dim``.
#
# Why a pre-fallback hook is needed: the trailing fallback in
# ``compute_batch_dim`` would correctly emit ``PARTIAL`` for a reduction whose
# output drops the batch dim -- *if* it ever ran.  But the intervening shape
# loop returns the first output dim whose size matches ``batch_dim_size``,
# preempting the fallback.  Reductions like
# ``aten.sum.dim_IntList(x, [0, 1, 2], keepdim=True)`` on an input shaped
# ``(B, ..., F)`` where ``F == B`` produce an output whose trailing dim
# incidentally equals ``batch_dim_size`` and get mis-classified as
# ``Shard(F_axis)``.  Ops registered here run first and can short-circuit
# that mis-classification.
#
# A rule is ``Callable[[node, operand_batch_dim, operand], Optional[int]]``;
# returning ``None`` means "no opinion, continue with the default flow".
_BatchDimRule = Callable[[fx.Node, int, fx.Node], Optional[int]]
_BATCH_DIM_PROPAGATION_RULES: Dict[torch._ops.OpOverload, _BatchDimRule] = {}


def _register_batch_dim_rule(
    op: torch._ops.OpOverload,
) -> Callable[[_BatchDimRule], _BatchDimRule]:
    def _decorator(fn: _BatchDimRule) -> _BatchDimRule:
        _BATCH_DIM_PROPAGATION_RULES[op] = fn
        return fn

    return _decorator


@_register_batch_dim_rule(aten.sum.dim_IntList)
def _propagate_sum_dim_intlist(
    node: fx.Node, operand_batch_dim: int, operand: fx.Node
) -> Optional[int]:
    """Emit ``PARTIAL`` when ``aten.sum.dim_IntList`` reduces the shard axis.

    Scope is intentionally narrow: only ``aten.sum.dim_IntList``;
    mean/max/prod have differing partial-reduction semantics and are out of
    scope.
    """
    if len(node.args) < 2:
        return None
    reduce_dims_arg = node.args[1]
    if not isinstance(reduce_dims_arg, (list, tuple)):
        return None
    operand_ndim = operand.meta["val"].ndim
    normalised_reduce_dims = {
        (d + operand_ndim) if d < 0 else d
        for d in reduce_dims_arg
        if isinstance(d, int)
    }
    if operand_batch_dim in normalised_reduce_dims:
        return PARTIAL
    return None


class BatchDimAnalyzer:
    """This class is used to analyze the batch dimension of each tensor/node in the graph.

    We need to know the batch dimension of each tensor/node so that we know
    exactly the sharding layout of intermediate tensors.

    We possibly should evaluate using symbolic shapes to track the batch dimension.
    We can experiment it later with dynamo integration (as dynamo have mark_dynamic
    API which allows marking batch dimension only) or try to use FakeTensorMode to
    mark the batch dimension. For now, let's just use the batch dimension of the first
    input tensor as the hint to track the batch dimension of all tensors/nodes in
    the graph.
    """

    def __init__(self, batch_dim: int = 0) -> None:
        self.batch_dim = batch_dim

        self.batch_dim_map: Dict[fx.Node, int] = {}
        # batch dim size is used to track the batch dim size of the input tensor
        self.batch_dim_size = -1

        self.dim_rule_map: Dict[torch._ops.OpOverload, Callable[..., torch.Tensor]] = {
            aten.squeeze.default: torch.squeeze,
            aten.squeeze.dim: torch.squeeze,
            aten.view.default: Tensor.view,
            aten.reshape.default: torch.reshape,
            aten._unsafe_view.default: Tensor.view,
            aten.unsqueeze.default: torch.unsqueeze,
            aten.expand.default: Tensor.expand,
            aten.permute.default: torch.permute,
            aten.repeat.default: Tensor.repeat,
            aten.transpose.int: torch.transpose,
        }

    def init_batch_dim_size(self, batch_dim_size: int) -> None:
        """Initialize batch dim size base on the first input batch size."""
        if self.batch_dim_size != -1 and self.batch_dim_size != batch_dim_size:
            raise RuntimeError(
                f"batch dim size is already initialized! "
                f"Found new batch size: {batch_dim_size} not "
                f"matching existing batch dim size: {self.batch_dim_size}!"
            )
        self.batch_dim_size = batch_dim_size

    def set_batch_dim(self, node: fx.Node, batch_dim: int) -> None:
        self.batch_dim_map[node] = batch_dim

    def get_batch_dim(self, node: fx.Node) -> int:
        if node not in self.batch_dim_map:
            raise RuntimeError(f"batch dim analysis failed on node: {node}!")
        return self.batch_dim_map[node]

    def compute_batch_dim(self, node: fx.Node, full_reduction: bool = False) -> int:
        """Compute the batch dimension for the `node`."""
        assert self.batch_dim_size != -1, "batch dim size is not initialized!"

        if node in self.batch_dim_map:
            # if batch dim already computed, simply return it
            return self.batch_dim_map[node]

        if node.target in self.dim_rule_map:
            dim_map = dim_maps[self.dim_rule_map[node.target]]  # type: ignore[index]
            args_val = pytree.tree_map_only(fx.Node, lambda n: n.meta["val"], node.args)
            kwargs_val = pytree.tree_map_only(
                fx.Node, lambda n: n.meta["val"], node.kwargs
            )
            output_dim_rules = dim_map(*args_val, **kwargs_val)

            def collect_input_dim(cmd: DimSpec, input_dims: Set[int]) -> None:
                if isinstance(cmd, InputDim):
                    input_dims.add(cmd.input_dim)
                for inp in cmd.inputs():
                    collect_input_dim(inp, input_dims)

            output_dim_to_input_dims: List[Set[int]] = []
            for inp in output_dim_rules:
                input_dims: Set[int] = set()
                collect_input_dim(inp, input_dims=input_dims)
                output_dim_to_input_dims.append(input_dims)

            operand = node.all_input_nodes[0]
            operand_batch_dim = self.get_batch_dim(operand)
            for output_dim, input_dims in enumerate(output_dim_to_input_dims):
                if operand_batch_dim in input_dims:
                    self.set_batch_dim(node, output_dim)
                    # update batch dim size before return
                    # this is because batch dim size might change during the middle
                    self.batch_dim_size = node.meta["val"].shape[output_dim]
                    return output_dim

        # if there's no hints from the output_dim_rules, we infer from output
        # shape to see if there's batch dim, and shard correspondingly
        if "val" not in node.meta:
            # Node doesn't have a 'val' in meta, skip shape analysis
            shapes = []
        else:
            node_val = node.meta["val"]
            if isinstance(node_val, (list, tuple)):
                shapes = [val.shape for val in node_val if hasattr(val, "shape")]
            elif hasattr(node_val, "shape"):
                shapes = [node_val.shape]
            else:
                # node_val doesn't have a shape (e.g., _RecordFunction), skip shape analysis
                shapes = []

        # Op-specific propagation rules run *before* the size-match fallback
        # below.  See ``_BATCH_DIM_PROPAGATION_RULES`` for the rationale.
        # pyrefly: ignore [bad-argument-type]
        rule = _BATCH_DIM_PROPAGATION_RULES.get(node.target)
        if rule is not None:
            rule_inputs = node.all_input_nodes
            if rule_inputs and rule_inputs[0] in self.batch_dim_map:
                operand = rule_inputs[0]
                operand_batch_dim = self.batch_dim_map[operand]
                if operand_batch_dim >= 0:
                    rule_result = rule(node, operand_batch_dim, operand)
                    if rule_result is not None:
                        self.set_batch_dim(node, rule_result)
                        return rule_result

        # for reduction op that reduces over the sharded batch dim
        # we don't generate partial, but rather, we generate shard
        # This is because the intention of data parallel is to never
        # do full reduction across batch dimension, it would still
        # keep the reduction activation as sharded.
        full_reduction = False
        # loop through the dim size to find the output batch dim
        for shape in shapes:
            if len(shape) == 0:
                full_reduction = True

            for i, dim_size in enumerate(shape):
                if dim_size == self.batch_dim_size:
                    self.set_batch_dim(node, i)
                    return i

        operands = node.all_input_nodes
        if not operands:
            # if there's no operands, it must be factory ops and it's a tensor
            # generated for computation and should be marked as replicated
            self.set_batch_dim(node, REPLICATED)
            return REPLICATED
        else:
            # if there's operand we see the operand have batch dim, if operand
            # have batch dim but output does not, it's either a full reduction,
            # where we should stay sharded, or it's a reduction on batch dim only
            # where we should produce partial
            operand_batch_dim = -1
            for operand in operands:
                if operand in self.batch_dim_map:
                    operand_batch_dim = self.get_batch_dim(operand)
            # self.get_batch_dim(operands[0])
            if operand_batch_dim < 0:
                # if operand does not have batch dim, we also don't have batch dim
                self.set_batch_dim(node, operand_batch_dim)
                return operand_batch_dim
            elif full_reduction:
                self.set_batch_dim(node, operand_batch_dim)
                return operand_batch_dim
            else:
                # if operand has batch dim but output does not, produce
                # partial.
                self.set_batch_dim(node, PARTIAL)
                return PARTIAL

    def compute_act_spec(self, node: fx.Node, mesh: DeviceMesh) -> DTensorSpec:
        """Compute the batch dimension for the current node, then generate the sharding spec that shards on the batch dimension."""
        node_batch_dim = self.compute_batch_dim(node)
        if node_batch_dim == REPLICATED:
            act_spec = DTensorSpec(mesh=mesh, placements=(Replicate(),))
        elif node_batch_dim == PARTIAL:
            act_spec = DTensorSpec(mesh=mesh, placements=(_Partial(),))
        else:
            act_spec = DTensorSpec(mesh=mesh, placements=(Shard(node_batch_dim),))

        return act_spec
