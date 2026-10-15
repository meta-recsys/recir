# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import math
import operator
from contextlib import contextmanager
from enum import Enum
from functools import cached_property
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

import torch
import torch.distributed._functional_collectives as funcol
import torch.fx as fx
import torch.library
import torch.nn as nn
import torch.utils._pytree as pytree
from recir.distributed.batch_dim_utils import BatchDimAnalyzer
from recir.distributed.sharding_plan import EmbeddingShardingPlan

# v1's ``_rewrite_row_wise_embeddings`` (used by ``partition_data_parallel``
# in the bundle_runtime path) reuses the row-wise embedding decomposition
# that now lives in ``recir/region_transforms/sparse_embedding.py``.
from recir.region_transforms.sparse_embedding import decomposed_row_wise_embedding
from recir.utils.meta_tensor_init import (
    copy_meta_init_attrs,
    default_embedding_init_bound,
    get_meta_init_spec,
    initialize_tensor_from_meta_init,
    mark_meta_zero_init,
)
from torch.distributed.tensor import DeviceMesh, DTensor, Replicate, Shard
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._op_schema import (
    OpSpec,
    OpStrategy,
    StrategyType,
    TupleStrategy,
)
from torch.distributed.tensor._redistribute import redistribute_local_tensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
from torch.distributed.tensor.placement_types import Partial
from torch.fx import GraphModule
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.passes.shape_prop import _extract_tensor_metadata


# Backward-compat for ``shard_params*``: historically a single
# Optional[Optimizer]; multi-opt callers (sparse + dense partition) pass a list.
OptOrOpts = Union[None, torch.optim.Optimizer, Sequence[torch.optim.Optimizer]]


def compute_local_shape(global_shape: Any, mesh: DeviceMesh, placements: Any) -> Any:
    """Helper function to compute local shape (wrapper around compute_local_shape_and_global_offset)."""
    if global_shape in [[], torch.Size([])]:
        return global_shape
    local_shape, _ = compute_local_shape_and_global_offset(
        global_shape, mesh, placements, skip_offset=True
    )
    return local_shape


aten: torch._ops._OpNamespace = torch.ops.aten

# Dummy op used by data parallel to tag gradients.
_spmd_lib_def = torch.library.Library("_spmd", "DEF")
_spmd_lib_def.define("tag_grad(Tensor self) -> Tensor")

_spmd_lib_impl = torch.library.Library("_spmd", "IMPL")
_spmd_lib_impl.impl("tag_grad", lambda x: x, "CompositeExplicitAutograd")

# Op to free the underlying storage of a tensor (used by per-op FSDP to release
# allgathered weights immediately after their last consumer).
_recir_lib_def = torch.library.Library("recir", "DEF")
_recir_lib_def.define("free_storage(Tensor(a!) t) -> ()")


def _free_storage_impl(t: torch.Tensor) -> None:
    if t.untyped_storage().size() > 0:
        t.untyped_storage().resize_(0)


_recir_lib_impl = torch.library.Library("recir", "IMPL")
_recir_lib_impl.impl(
    "free_storage",
    _free_storage_impl,
    "CompositeExplicitAutograd",
)

# Mark as side-effectful so dead-code elimination doesn't remove it.
torch.fx.node._side_effectful_functions.add(torch.ops.recir.free_storage.default)

non_compute_ops: List[Any] = [
    aten._to_copy.default,
    aten.clone.default,
    aten.detach.default,
    aten.ones_like.default,
    aten.reshape.default,
    aten.t.default,
    aten.view.default,
    torch.ops._spmd.tag_grad.default,
    operator.getitem,
]
fused_adam_op: Any = aten._fused_adam.default
no_ops: List[Any] = [
    torch.ops.profiler._record_function_exit._RecordFunction,
    torch.ops.profiler._record_function_enter_new.default,
]


class DataParallelStyle(Enum):
    """This enum represents the style of the data-parallel operation.

    We have three types of Data Parallel style:
    1. DEFAULT: the default data parallel style, which is to represent a mixed
                replicate and fully shard behavior. For each parameter that is able
                to be sharded evenly, we shard it, otherwise we would replicate the
                parameter. This style avoids potential padding if the parameters
                cannot be sharded evenly, but it would generate a mixed of all_reduce
                and reduce_scatter.
    2. REPLICATE: the data parallel style that replicates all model parameters.
                  This is similar to the behavior of DistributedDataParallel.
    3. FULLY_SHARD: the data parallel style that shards all model parameters. This
                    is similar to the behavior of FullyShardedDataParallel, the
                    difference is that FullyShardedDataParallel (ZERO-3), which
                    shards the model using FlatParameter based sharding,
                    while this style shards each parameter into DTensor.
    """

    DEFAULT = 0
    REPLICATE = 1
    FULLY_SHARD = 2


class NodeType(Enum):
    """NodeType is an enum that records the type of the tensors in the graph.

    This is used to determine the data parallel strategy.
    """

    PARAM = 0
    ACT = 1
    GRAD = 2
    STATE = 3
    NON_TENSOR = 4  # NON_TENSOR is to tag non tensor node (i.e. graph output)
    EMBEDDING_PARAM = 5  # Embedding param with model-parallel sharding (e.g., row-wise)


class DataParallelStrategy(OpStrategy):
    """DataParallelStrategy is a special case of OpStrategy that only records the "data parallel style" placement
    strategy for each fx Node.

    It takes a OpSpec, which describes one way to distribute the tensor and computation.
    In the DataParallel case, there're two possible ways to distribute the parameters:
        1. replicate the parameter over a set of devices (DDP like behavior)
        2. shard the parameter on its tensor dimension 0 over a set of devices
           (FSDP like behavior).

    In addition to the strategy, we also need to:
    1. `node_type`: record the type of each node in the graph, so that we can
        determine how to propagate in a data parallel fashion.
    2. `reduce_over_batch` is specifically tied to data parallel as the loss
        calculation usually results in scalar tensor where it comes from a
        reduction over the batch dimension. We need to know this information
        so that we could keep the output as sharded.
    """

    def __init__(
        self,
        node_types: List[NodeType],
        op_spec: Optional[OpSpec],
        reduce_over_batch: bool = False,
    ) -> None:
        strategies = [] if op_spec is None else [op_spec]
        super().__init__(strategies)
        self.node_types = node_types
        self.reduce_over_batch = reduce_over_batch

    def __str__(self) -> str:
        return f"type: {self.node_types}, {super().__str__()}"

    @classmethod
    def from_mesh(
        cls,
        node_types: List[NodeType],
        mesh: DeviceMesh,
        parallel_mode: DataParallelStyle,
        num_outputs: int = 1,
    ) -> "DataParallelStrategy":
        assert parallel_mode in [
            DataParallelStyle.REPLICATE,
            DataParallelStyle.FULLY_SHARD,
        ]
        if parallel_mode == DataParallelStyle.REPLICATE:
            return cls(
                node_types,
                _gen_replicate_spec(mesh, num_outputs=num_outputs),
            )
        else:
            return cls(
                node_types, _gen_shard_spec(mesh, shard_dim=0, num_outputs=num_outputs)
            )

    @cached_property
    def node_type(self) -> NodeType:
        assert len(self.node_types) == 1
        return self.node_types[0]

    @cached_property
    def op_spec(self) -> OpSpec:
        assert len(self.strategies) == 1
        return self.strategies[0]


def unique_spec(
    input_spec: Union[DTensorSpec, Tuple[Optional[DTensorSpec], ...]],
) -> DTensorSpec:
    if isinstance(input_spec, DTensorSpec):
        return input_spec
    elif len(input_spec) == 1:
        spec = input_spec[0]
        assert spec is not None
        return spec
    else:
        spec = input_spec[0]
        assert spec is not None
        for s in input_spec[1:]:
            assert s is not None
            assert s.mesh == spec.mesh
            assert s.placements == spec.placements
        return spec


@contextmanager
def gradients_tagging(params: Dict[str, torch.Tensor]) -> Any:
    """Tag the gradient of the parameters with a special tag, so that we can identify them during SPMD expansion.

    It's safe to trace those hooks and we would remove those nodes later.
    """
    tagging_hooks = []
    try:
        for p in params.values():
            if p.requires_grad:
                h = p.register_hook(torch.ops._spmd.tag_grad)
                tagging_hooks.append(h)
        yield
    finally:
        # remove those hooks after tracing
        for h in tagging_hooks:
            h.remove()


# Cache key is (base_class, frozenset(param_names)) — two ``nn.Linear``
# instances with different parameter sets (e.g. ``bias=True`` vs
# ``bias=False``) need different parametrized subclasses because each
# subclass installs ``property`` getters for exactly the params present.
# Sharing a single subclass across all Linears would make later instances
# inherit a property they have no entry for in ``_parametrizations``.
_PARAMETRIZATION_CLASS_CACHE: Dict[
    Tuple[Type[nn.Module], FrozenSet[str]], Type[nn.Module]
] = {}


def _register_parametrization(
    module: nn.Module,
    parametrize_fn: Callable[..., torch.Tensor],
    recurse: bool = True,
    exclude_param_fqns: Optional[Set[str]] = None,
    include_param_fqns: Optional[Set[str]] = None,
) -> None:
    """Register a parametrization on all parameters of *module*.

    When ``module.weight`` is accessed, the raw value from
    ``self._parameters["weight"]`` is passed through every registered
    ``parametrize_fn`` in order.  Multiple calls **stack** — each new
    function is appended to the chain.

    Parametrize fns are called as ``fn(value)`` by default.  Fns tagged
    with ``fn._recir_parametrize_wants_context = True`` receive the
    owning submodule and parameter name too (``fn(value, submod, pname)``)
    so they can dispatch on FQN (e.g. QAT applies STE only to target
    embedding weights).

    Args:
        module: The module (and optionally its submodules) to parametrize.
        parametrize_fn: ``(Tensor) -> Tensor`` (or
            ``(Tensor, nn.Module, str) -> Tensor`` when the fn is marked
            context-aware) applied to the raw parameter.
        recurse: If True, also parametrize all submodules.
        exclude_param_fqns: Optional set of fully-qualified parameter names
            to exclude from parametrization (e.g., embedding params that
            use AllToAll instead of allgather).
        include_param_fqns: Optional set of fully-qualified parameter names
            to parametrize. Parameters outside this set are skipped.
    """
    modules = module.named_modules() if recurse else [("", module)]
    for _fqn, submod in modules:
        exclude_param_fqns = exclude_param_fqns or set()
        all_param_names = [
            name for name in submod._parameters if submod._parameters[name] is not None
        ]
        param_names = [
            pn
            for pn in all_param_names
            if (
                (f"{_fqn}.{pn}" if _fqn else pn) not in exclude_param_fqns
                and (
                    include_param_fqns is None
                    or (f"{_fqn}.{pn}" if _fqn else pn) in include_param_fqns
                )
            )
        ]
        if not param_names:
            continue

        base_class = submod.__class__
        if not hasattr(base_class, "_recir_parametrized_base"):
            # First parametrization on this module — create class + storage.
            submod._parametrizations = {}  # type: ignore[assignment]
            for pn in all_param_names:
                submod._parametrizations[pn] = []  # type: ignore[index]

            cache_key = (base_class, frozenset(all_param_names))
            if cache_key not in _PARAMETRIZATION_CLASS_CACHE:
                properties: Dict[str, Any] = {}

                def _make_getter(param_name: str) -> property:
                    def getter(self: nn.Module) -> torch.Tensor:
                        value = self._parameters[param_name]
                        for p in self._parametrizations[param_name]:  # type: ignore[index]
                            # Context-aware fns (marked with
                            # ``_recir_parametrize_wants_context``) receive
                            # the owning module + param name so they can
                            # dispatch on FQN (e.g. QAT applies only to
                            # target embedding weights). Ordinary fns keep
                            # the single-arg contract used by
                            # ``collectives_insertion``.
                            if getattr(p, "_recir_parametrize_wants_context", False):
                                value = p(value, self, param_name)
                            else:
                                value = p(value)
                        # pyrefly: ignore [bad-return]
                        return value

                    # pyrefly: ignore [bad-return]
                    return property(getter)

                for pn in all_param_names:
                    properties[pn] = _make_getter(pn)

                new_cls = type(
                    base_class.__name__,
                    (base_class,),
                    properties,
                )
                new_cls._recir_parametrized_base = base_class  # type: ignore[attr-defined]
                _PARAMETRIZATION_CLASS_CACHE[cache_key] = new_cls

            submod.__class__ = _PARAMETRIZATION_CLASS_CACHE[cache_key]

        for pn in param_names:
            submod._parametrizations[pn].append(parametrize_fn)  # type: ignore[index]


def _deregister_parametrization(
    module: nn.Module,
    parametrize_fn: Callable[..., torch.Tensor],
    recurse: bool = True,
) -> None:
    """Remove *parametrize_fn* from *module* (inverse of :func:`_register_parametrization`)."""
    modules = module.named_modules() if recurse else [("", module)]
    for _fqn, submod in modules:
        if not hasattr(submod, "_parametrizations"):
            continue
        for pn in list(submod._parametrizations):  # type: ignore[union-attr]
            plist = submod._parametrizations[pn]  # type: ignore[index]
            if parametrize_fn in plist:
                # pyrefly: ignore [missing-attribute]
                plist.remove(parametrize_fn)

        # Restore original class if no parametrizations remain.
        if all(
            len(plist) == 0
            for plist in submod._parametrizations.values()  # type: ignore[union-attr]
        ):
            orig_cls = getattr(submod.__class__, "_recir_parametrized_base", None)
            if orig_cls is not None:
                submod.__class__ = orig_cls
            del submod._parametrizations  # type: ignore[union-attr]


@contextmanager
def parametrize_params(
    module: nn.Module,
    parametrize_fn: Callable[..., torch.Tensor],
    exclude_param_fqns: Optional[Set[str]] = None,
    include_param_fqns: Optional[Set[str]] = None,
) -> Any:
    """Temporarily register *parametrize_fn* on *module*; remove on exit.

    Supports nesting — each layer appends to the parametrization chain::
        with parametrize_params(module, collectives_fn):
            with parametrize_params(module, mixed_precision_fn):
                # module.weight → mixed_precision_fn(collectives_fn(raw_weight))
                ...

    Args:
        exclude_param_fqns: Optional set of fully-qualified parameter names
            to exclude from parametrization.
        include_param_fqns: Optional set of fully-qualified parameter names
            to parametrize. Parameters outside this set are skipped.
    """
    _register_parametrization(
        module,
        parametrize_fn,
        exclude_param_fqns=exclude_param_fqns,
        include_param_fqns=include_param_fqns,
    )
    try:
        yield
    finally:
        _deregister_parametrization(module, parametrize_fn)


# ---------------------------------------------------------------------------
# Autograd-based collective insertion (replaces gradients_tagging + partition)
# ---------------------------------------------------------------------------


def collectives_insertion(
    parallel_style: DataParallelStyle,
    mesh: DeviceMesh,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a per-parameter transform that inserts DTensor collectives."""

    def _parametrize_fn(param: torch.Tensor) -> torch.Tensor:
        if parallel_style == DataParallelStyle.FULLY_SHARD:
            dtensor = DTensor.from_local(param, mesh, [Shard(0)], run_check=False)
            full = dtensor.redistribute(mesh, [Replicate()])
            return full.to_local(grad_placements=[Partial("avg")])
        else:
            dtensor = DTensor.from_local(param, mesh, [Replicate()], run_check=False)
            return dtensor.to_local(grad_placements=[Partial("avg")])

    return _parametrize_fn


def _gen_shard_spec(
    mesh: DeviceMesh,
    shard_dim: int,
    input_specs: Optional[List[DTensorSpec]] = None,
    num_outputs: int = 1,
) -> OpSpec:
    """Util function to generate a shard strategy on shard_dim."""
    out_specs = tuple(
        [DTensorSpec(mesh=mesh, placements=(Shard(shard_dim),))] * num_outputs
    )
    return OpSpec(output_specs=out_specs, input_specs=input_specs)


def _gen_replicate_spec(
    mesh: DeviceMesh,
    input_specs: Optional[List[DTensorSpec]] = None,
    num_outputs: int = 1,
) -> OpSpec:
    """Util function to generate a replicate strategy."""
    out_specs = tuple([DTensorSpec(mesh=mesh, placements=(Replicate(),))] * num_outputs)
    return OpSpec(output_specs=out_specs, input_specs=input_specs)


def _gen_partial_strategy(mesh: DeviceMesh, grad_reduction: str = "avg") -> OpSpec:
    """Util function to generate a partial strategy."""
    # NOTE: The reduction type should match the loss function:
    #   - "avg" (default): correct for sum-based losses (each rank's sum
    #     is averaged to match DDP semantics).
    #   - "sum": correct for mean-based losses (each rank's mean-backward
    #     already normalises by local numel; a plain sum gives the
    #     equivalent of a global-mean gradient without double-dividing).
    # TODO: Only NCCL supports AVG so using backend like Gloo would
    # crash, we should figure out a way to support avg reduction
    # for non-NCCL backend
    return OpSpec(
        output_specs=DTensorSpec(mesh=mesh, placements=(Partial(grad_reduction),)),
    )


def user_to_index_map(node: fx.Node) -> Dict[fx.Node, int]:
    """
    Maps each user of a multi-output node to the output index it consumes.
    """
    user_to_index: Dict[fx.Node, int] = {}
    for user in node.users:
        if user.op == "call_function" and user.target == operator.getitem:
            index = user.args[1]
            user_to_index[user] = index  # pyre-ignore [6]
    return user_to_index


# Non-compute ops for which we can track how the shard dimension transforms.
# Maps op target -> function(input_shard_dim) -> output_shard_dim.
_SHARD_DIM_TRANSFORM: Dict[Any, Callable[[int], int]] = {
    aten.t.default: lambda dim: 1 - dim,
    aten.clone.default: lambda dim: dim,
    aten.detach.default: lambda dim: dim,
    aten._to_copy.default: lambda dim: dim,
}


def _propagate_param_non_compute_strategy(
    node: fx.Node,
    arg_strategy: DataParallelStrategy,
    mesh: DeviceMesh,
    parallel_mode: DataParallelStyle,
) -> DataParallelStrategy:
    """Propagate strategy for non-compute ops with PARAM or EMBEDDING_PARAM input.

    In FULLY_SHARD mode (or for EMBEDDING_PARAM which is always sharded),
    propagates the shard placement through ops that have a known
    shard-dimension transformation (t, clone, detach, _to_copy). This
    delays the allgather communication to right before the first compute op
    instead of materializing it at the non-compute op boundary.

    Falls back to Replicate for REPLICATE mode or ops where shard dimension
    tracking is not supported (view, reshape, ones_like).
    """
    node_type = arg_strategy.node_type
    # EMBEDDING_PARAM is always sharded (regardless of parallel_mode),
    # so treat it like FULLY_SHARD for shard-dim tracking.
    is_sharded = (
        parallel_mode == DataParallelStyle.FULLY_SHARD
        or node_type == NodeType.EMBEDDING_PARAM
    )
    if is_sharded:
        output_specs = arg_strategy.op_spec.output_specs
        assert output_specs is not None
        input_spec = unique_spec(output_specs)
        input_placement = input_spec.placements[0]
        if isinstance(input_placement, Shard) and node.target in _SHARD_DIM_TRANSFORM:
            input_shard_dim = input_placement.dim
            output_shard_dim = _SHARD_DIM_TRANSFORM[node.target](input_shard_dim)
            output_spec = _gen_shard_spec(
                mesh,
                shard_dim=output_shard_dim,
                input_specs=[
                    DTensorSpec(mesh=mesh, placements=(Shard(input_shard_dim),))
                ],
            )
            return DataParallelStrategy([node_type], output_spec)
    return DataParallelStrategy([node_type], _gen_replicate_spec(mesh))


_RINGSHIFT_EMBEDDING_SCHEMA = "recir::ringshift_embedding"
_RINGSHIFT_EMBEDDING_BACKWARD_SCHEMA = "recir::ringshift_embedding_backward"


def _target_schema_name(target: Any) -> Optional[str]:
    schema = getattr(target, "_schema", None)
    name = getattr(schema, "name", None)
    return name if isinstance(name, str) else None


def _require_shard0_strategy(
    node: fx.Node,
    strategy: DataParallelStrategy,
    *,
    role: str,
) -> None:
    output_specs = strategy.op_spec.output_specs
    if output_specs is None:
        raise ValueError(f"{role} {node.name!r} has no output sharding spec")
    spec = unique_spec(output_specs)
    if spec.placements != (Shard(0),):
        raise ValueError(
            f"{role} {node.name!r} must be Shard(0) for ring-shift embedding; "
            f"got {spec.placements}"
        )


def _propagate_ringshift_embedding_strategy(
    node: fx.Node,
    dp_strategy_map: Dict[fx.Node, StrategyType],
    mesh: DeviceMesh,
    grad_reduction: str,
) -> bool:
    """Keep ring-shift table state and gradients row-sharded end to end."""
    schema_name = _target_schema_name(node.target)
    if schema_name not in {
        _RINGSHIFT_EMBEDDING_SCHEMA,
        _RINGSHIFT_EMBEDDING_BACKWARD_SCHEMA,
    }:
        return False

    input_nodes = node.all_input_nodes
    is_backward = schema_name == _RINGSHIFT_EMBEDDING_BACKWARD_SCHEMA
    expected_inputs = 4 if is_backward else 3
    if len(input_nodes) != expected_inputs:
        raise ValueError(
            f"{schema_name} expected {expected_inputs} tensor inputs, "
            f"got {len(input_nodes)}"
        )

    input_strategies = []
    for input_node in input_nodes:
        strategy = dp_strategy_map[input_node]
        if not isinstance(strategy, DataParallelStrategy):
            raise ValueError(
                f"{schema_name} input {input_node.name!r} has unsupported "
                f"strategy {strategy}"
            )
        input_strategies.append(strategy)

    weight_node = input_nodes[-1]
    weight_strategy = input_strategies[-1]
    if weight_strategy.node_type not in (NodeType.PARAM, NodeType.EMBEDDING_PARAM):
        raise ValueError(
            f"{schema_name} weight must be a parameter, got {weight_strategy.node_type}"
        )
    _require_shard0_strategy(
        weight_node,
        weight_strategy,
        role=f"{schema_name} weight",
    )

    for activation_node, strategy in zip(input_nodes[:-1], input_strategies[:-1]):
        if strategy.node_type != NodeType.ACT:
            raise ValueError(
                f"{schema_name} activation {activation_node.name!r} must have "
                f"ACT type, got {strategy.node_type}"
            )
        _require_shard0_strategy(
            activation_node,
            strategy,
            role=f"{schema_name} activation",
        )

    if is_backward and grad_reduction != "sum":
        raise ValueError(
            "ring-shift embedding backward performs a global sum and requires "
            f"grad_reduction='sum'; got {grad_reduction!r}"
        )

    input_specs = [DTensorSpec(mesh=mesh, placements=(Shard(0),)) for _ in input_nodes]
    output_spec = _gen_shard_spec(mesh, shard_dim=0, input_specs=input_specs)
    output_type = NodeType.GRAD if is_backward else NodeType.ACT
    dp_strategy_map[node] = DataParallelStrategy([output_type], output_spec)
    return True


def _propagate_grad_non_compute_strategy(
    node: fx.Node,
    input_node: fx.Node,
    arg_strategy: DataParallelStrategy,
    mesh: DeviceMesh,
    grad_reduction: str,
) -> DataParallelStrategy:
    """Preserve an already-reduced sharded gradient through layout-safe ops."""
    input_specs = arg_strategy.op_spec.output_specs
    if input_specs is not None:
        input_spec = unique_spec(input_specs)
        input_placement = input_spec.placements[0]
        output_shard_dim: Optional[int] = None
        if isinstance(input_placement, Shard):
            if node.target in _SHARD_DIM_TRANSFORM:
                output_shard_dim = _SHARD_DIM_TRANSFORM[node.target](
                    input_placement.dim
                )
            elif node.target in (aten.reshape.default, aten.view.default):
                input_val = input_node.meta.get("val")
                output_val = node.meta.get("val")
                if (
                    hasattr(input_val, "shape")
                    and hasattr(output_val, "shape")
                    and tuple(input_val.shape) == tuple(output_val.shape)
                ):
                    output_shard_dim = input_placement.dim
        if output_shard_dim is not None:
            return DataParallelStrategy(
                [NodeType.GRAD],
                _gen_shard_spec(
                    mesh,
                    shard_dim=output_shard_dim,
                    input_specs=[input_spec],
                ),
            )
    return DataParallelStrategy(
        [NodeType.GRAD], _gen_partial_strategy(mesh, grad_reduction)
    )


def _propagate_call_function_strategy(  # noqa: C901
    node: fx.Node,
    dp_strategy_map: Dict[fx.Node, StrategyType],
    mesh: DeviceMesh,
    parallel_mode: DataParallelStyle,
    batch_dim_analyzer: BatchDimAnalyzer,
    grad_type_dict: Dict[fx.Node, Set[int]],
    grad_reduction: str = "avg",
) -> None:
    """Propagate sharding strategies through call_function nodes.

    Handles four categories of ops:
      1. Non-compute ops (view/reshape/getitem): inherit input strategy.
      2. Fused optimizer ops (fused_adam): build TupleStrategy.
      3. GRAD/STATE ops (optimizer step): inherit param/state type.
      4. General compute ops (forward/backward): build input/output specs,
         marking gradient outputs as GRAD.
    """
    input_nodes = node.all_input_nodes

    if _propagate_ringshift_embedding_strategy(
        node, dp_strategy_map, mesh, grad_reduction
    ):
        return

    # --- Category 1: non-compute ops (view, reshape, getitem, etc.) ---
    if node.target in non_compute_ops:
        assert node.target != torch.ops._spmd.tag_grad.default and len(input_nodes) == 1
        arg_strategy = dp_strategy_map[input_nodes[0]]
        if node.target == operator.getitem:
            getitem_idx = node.args[1]
            if isinstance(arg_strategy, TupleStrategy):
                # pyre-ignore[6]: getitem_idx is always int at runtime
                dp_strategy_map[node] = arg_strategy.children[getitem_idx]
            else:
                assert isinstance(arg_strategy, DataParallelStrategy)
                out_spec = arg_strategy.op_spec.output_specs[  # pyre-ignore[16]
                    getitem_idx
                ]
                dp_strategy_map[node] = DataParallelStrategy(
                    [arg_strategy.node_types[getitem_idx]],  # pyre-ignore[6]
                    OpSpec(output_specs=out_spec),
                )
        else:
            # Non-getitem, non-compute ops (e.g. t, clone, detach, reshape,
            # view, _to_copy). The handling depends on what kind of data
            # flows through:
            #   PARAM / EMBEDDING_PARAM → track shard dim through the op
            #   GRAD → propagate gradient sharding
            #   ACT  → propagate activation sharding
            assert isinstance(arg_strategy, DataParallelStrategy)
            arg_node_type = arg_strategy.node_type
            assert arg_node_type in [
                NodeType.PARAM,
                NodeType.EMBEDDING_PARAM,
                NodeType.GRAD,
                NodeType.ACT,
                # STATE flows through non-compute ops (view / reshape /
                # _to_copy / clone) on the new ``compile_from_gm`` path
                # whenever an optimizer (e.g. Adam) keeps per-param state
                # tensors and a kernel rewrites their layout before the
                # optimizer step.  STATE is always replicated, so we just
                # inherit the input strategy unchanged.
                NodeType.STATE,
                # NON_TENSOR can reach here when a constant / scalar wrapped
                # as a 0-d tensor flows through a no-op view; propagate it
                # unchanged.
                NodeType.NON_TENSOR,
            ]
            if arg_node_type in (NodeType.PARAM, NodeType.EMBEDDING_PARAM):
                dp_strategy_map[node] = _propagate_param_non_compute_strategy(
                    node, arg_strategy, mesh, parallel_mode
                )
            elif arg_node_type == NodeType.GRAD:
                dp_strategy_map[node] = _propagate_grad_non_compute_strategy(
                    node,
                    input_nodes[0],
                    arg_strategy,
                    mesh,
                    grad_reduction,
                )
            elif arg_node_type in (NodeType.STATE, NodeType.NON_TENSOR):
                # STATE / NON_TENSOR: inherit input's (replicated) strategy.
                # No batch-dim tracking required.
                dp_strategy_map[node] = arg_strategy
            else:  # ACT: propagate batch-dim sharding
                arg_node_spec = batch_dim_analyzer.compute_act_spec(
                    input_nodes[0], mesh
                )
                output_spec = batch_dim_analyzer.compute_act_spec(node, mesh)
                shard_strategy = OpSpec(
                    output_specs=output_spec, input_specs=[arg_node_spec]
                )
                dp_strategy_map[node] = DataParallelStrategy(
                    [NodeType.ACT], shard_strategy
                )
        return

    # --- Category 2: fused optimizer ops (e.g., fused_adam) ---
    if node.target == fused_adam_op:
        # ops that need to build tuple strategy instead of normal strategy. This should happen rarely and only needed
        # when we need to generate different node strategy for multiple outputs (i.e. fused_adam op)
        # TODO: Currently this specializes to fused optimizer ops, but we need to see how to generalize this strategy building logic
        output_strategy_len = len(node.args) - 1
        tuple_strategies = []
        for i in range(output_strategy_len):
            assert isinstance(node.args[i], list)
            # for list/tuple arg, use the first one to find out the node type
            if len(node.args[i]) > 0:  # pyre-ignore[6]
                arg_strategy = dp_strategy_map[node.args[i][0]]  # pyre-ignore[16]
                assert isinstance(arg_strategy, DataParallelStrategy)
                assert arg_strategy.node_type in [
                    NodeType.PARAM,
                    NodeType.GRAD,
                    NodeType.STATE,
                ]
                out_node_strategy: StrategyType = DataParallelStrategy.from_mesh(
                    [arg_strategy.node_type] * len(node.args[i]),  # pyre-ignore[6]
                    mesh,
                    parallel_mode,
                    num_outputs=len(node.args[i]),  # pyre-ignore[6]
                )
            else:
                out_node_strategy = DataParallelStrategy([NodeType.NON_TENSOR], None)
            tuple_strategies.append(out_node_strategy)
        output_tuple_strategy = TupleStrategy(tuple(tuple_strategies))
        dp_strategy_map[node] = output_tuple_strategy
        return

    # --- Category 3 & 4: general compute ops ---
    # The same op (e.g. aten.add_) can appear in different contexts, so we
    # dispatch based on the input node types:
    #   - GRAD/STATE inputs → optimizer step (e.g. add_(weight, grad))
    #   - PARAM/EMBEDDING_PARAM only → param-only ops (e.g. mul_(weight, scalar))
    #   - ACT inputs → forward/backward compute (e.g. mm(activation, weight))
    input_node_types = [
        cast(DataParallelStrategy, dp_strategy_map[arg]).node_type
        for arg in input_nodes
    ]

    # Category 3: ops with GRAD or STATE inputs (optimizer step ops)
    if NodeType.GRAD in input_node_types or NodeType.STATE in input_node_types:
        if NodeType.GRAD in input_node_types:
            # Gradient applied to a param or state: e.g. add_(weight, grad).
            # Output type is the non-grad input type (PARAM, EMBEDDING_PARAM,
            # or STATE), since the result represents the updated
            # param/embedding param/state, not a gradient.
            non_grad_types = [t for t in input_node_types if t != NodeType.GRAD]
            assert len(non_grad_types) != 0, (
                f"Expected at least one non-GRAD input, "
                f"but all input_node_types are GRAD for node {node.target}"
            )
            for non_grad_type in non_grad_types:
                assert non_grad_type == non_grad_types[0]
            assert non_grad_types[0] in [
                NodeType.PARAM,
                NodeType.EMBEDDING_PARAM,
                NodeType.STATE,
            ]
            output_node_type = non_grad_types[0]
        else:
            # State-only update (no gradient): e.g. mul_(momentum, beta).
            # Preserve the input's NodeType on the output so downstream
            # nodes continue to apply the correct propagation logic
            # (e.g. shard-dim tracking for params, replicate for state).
            if NodeType.PARAM in input_node_types:
                output_node_type = NodeType.PARAM
            elif NodeType.EMBEDDING_PARAM in input_node_types:
                output_node_type = NodeType.EMBEDDING_PARAM
            else:
                output_node_type = NodeType.STATE
        num_outputs = (
            len(node.meta["val"]) if isinstance(node.meta["val"], (tuple, list)) else 1
        )
        dp_strategy_map[node] = DataParallelStrategy.from_mesh(
            [output_node_type] * num_outputs,
            mesh,
            parallel_mode,
            num_outputs=num_outputs,
        )
        return

    # Param-only ops: copy_, mul_, lift_fresh_copy, etc.
    # By this point, ops with GRAD/STATE inputs (Category 3) have already
    # returned. This catches ops with no ACT and no GRAD/STATE — i.e. inputs
    # are only PARAM, EMBEDDING_PARAM, or NON_TENSOR. These are
    # param-manipulation ops outside forward/backward and optimizer step
    # (e.g. weight initialization, weight decay).
    # If any input is a param, inherit its strategy; otherwise NON_TENSOR.
    if NodeType.ACT not in input_node_types and len(input_nodes) > 0:
        param_inputs = [
            inp
            for inp, ntype in zip(input_nodes, input_node_types)
            if ntype in (NodeType.PARAM, NodeType.EMBEDDING_PARAM)
        ]
        if len(param_inputs) > 0:
            dp_strategy_map[node] = dp_strategy_map[param_inputs[0]]
        else:
            dp_strategy_map[node] = DataParallelStrategy([NodeType.NON_TENSOR], None)
        return

    # Category 4: general compute ops (forward/backward with ACT inputs)
    if len(input_nodes) > 0:
        assert NodeType.ACT in input_node_types, (
            f"input_node_types is {input_node_types}, node is {node.target}"
        )
    input_specs = []
    for arg in input_nodes:
        arg_strategy = dp_strategy_map[arg]
        assert isinstance(arg_strategy, DataParallelStrategy)
        node_type = arg_strategy.node_type
        assert node_type in [
            NodeType.ACT,
            NodeType.PARAM,
            NodeType.EMBEDDING_PARAM,
            NodeType.NON_TENSOR,
        ], (
            f"Unexpected node_type {node_type} for input {arg.target} "
            f"of node {node.target}"
        )
        if node_type == NodeType.ACT:
            act_spec = batch_dim_analyzer.compute_act_spec(arg, mesh)
            input_specs.append(act_spec)
        else:
            # PARAM, EMBEDDING_PARAM, and NON_TENSOR are all replicated
            # for strategy propagation purposes. For EMBEDDING_PARAM, the
            # aten.embedding node is decomposed into explicit AllToAll ops
            # by _rewrite_row_wise_embeddings before the partitioner runs,
            # so these specs are not consumed.
            input_specs.append(DTensorSpec(mesh=mesh, placements=(Replicate(),)))
    node_type_len = (
        len(node.meta["val"]) if isinstance(node.meta["val"], (tuple, list)) else 1
    )
    act_spec = batch_dim_analyzer.compute_act_spec(node, mesh)
    grad_spec = DTensorSpec(mesh=mesh, placements=(Partial(grad_reduction),))
    if node not in grad_type_dict:
        out_specs = [act_spec for _ in range(node_type_len)]
        node_types = [NodeType.ACT for _ in range(node_type_len)]
    else:
        out_specs = []
        node_types = []
        for i in range(node_type_len):
            if i in grad_type_dict[node]:
                out_specs.append(grad_spec)
                node_types.append(NodeType.GRAD)
            else:
                out_specs.append(act_spec)
                node_types.append(NodeType.ACT)
    strategy = OpSpec(output_specs=tuple(out_specs), input_specs=input_specs)
    dp_strategy_map[node] = DataParallelStrategy(node_types, strategy)


def build_data_parallel_strategies(  # noqa: C901
    train_step_graph: GraphModule,
    num_params: int,
    num_states: int,
    mesh: DeviceMesh,
    batch_dim: int = 0,
    parallel_mode: DataParallelStyle = DataParallelStyle.FULLY_SHARD,
    grad_reduction: str = "avg",
    embedding_param_indices: Optional[Set[int]] = None,
    replicated_param_state_indices: Optional[Set[int]] = None,
) -> Dict[fx.Node, StrategyType]:
    activation_idx = num_params + num_states
    # Map each node to outputs with NodeType.GRAD type
    grad_type_dict: Dict[fx.Node, Set[int]] = {}
    dp_strategy_map: Dict[fx.Node, StrategyType] = {}
    batch_dim_analyzer = BatchDimAnalyzer(batch_dim)
    placeholder_idx = 0
    # Backward propagate to mark the param gradients sharding with tag_grad node helps and then delete the tag_grad nodes
    for node in reversed(list(train_step_graph.graph.nodes)):
        if not node.target == torch.ops._spmd.tag_grad.default:
            continue
        cur_node = node
        while cur_node.target in non_compute_ops:
            parent_node = cur_node.args[0]
            if parent_node not in grad_type_dict:
                grad_type_dict[parent_node] = set()
            index = 0
            if cur_node.target == operator.getitem:
                index = cur_node.args[1]
            grad_type_dict[parent_node].add(index)
            cur_node = parent_node
        node.replace_all_uses_with(node.args[0])
        train_step_graph.graph.erase_node(node)
    # Forward propagate to mark all the sharding
    for node in train_step_graph.graph.nodes:
        assert node.op in ["placeholder", "call_function", "output", "get_attr"]
        if node.target in no_ops:
            continue
        elif node.op == "get_attr":
            # ``get_attr`` nodes reference module attributes which may be
            # tensors (e.g. inline tensor literals lifted by ``make_fx``, the
            # step counter in capturable Adam) or arbitrary Python objects
            # (e.g. ``DeviceMesh``, process-group metadata). Walk the
            # qualified target name on the GraphModule to recover the value
            # so we can pick the right strategy. Tensor constants are
            # identical on every rank and therefore REPLICATE.
            attr_val: Any = train_step_graph
            for attr in node.target.split("."):
                attr_val = getattr(attr_val, attr, None)
                if attr_val is None:
                    break
            if isinstance(attr_val, torch.Tensor):
                dp_strategy_map[node] = DataParallelStrategy.from_mesh(
                    [NodeType.PARAM], mesh, DataParallelStyle.REPLICATE
                )
            else:
                dp_strategy_map[node] = DataParallelStrategy(
                    [NodeType.NON_TENSOR], None
                )
        elif "val" not in node.meta:
            # NOTE: There're certain cases where the nodes do not have real tensor values:
            # 1. optimizer states can be None sometimes, i.e. SGD with no momentum, optimizer states populate `momentum` state
            #    as None, the full graph we get from `compile` would have None as the placeholder value
            # 2. function args might not only contain params or activations, but also contain other non-tensor inputs, i.e. the model
            #    and optimizer instances baked in as a placeholder, there might also be some scalar argument which is not a tensor
            # 3. some inplace ops (such as torch.ops.aten._foreach_lerp_.Scalar) will return None.
            # For the above cases, we create a NON_TENSOR stratgy so that we know it's not a tensor and we don't need to shard it
            dp_strategy_map[node] = DataParallelStrategy([NodeType.NON_TENSOR], None)
        elif node.op == "output":
            dp_strategy_map[node] = DataParallelStrategy([NodeType.NON_TENSOR], None)
        elif node.op == "placeholder":
            if placeholder_idx < activation_idx:
                if (
                    embedding_param_indices
                    and placeholder_idx in embedding_param_indices
                ):
                    # Label embedding params as EMBEDDING_PARAM with Shard(0) on
                    # the vocabulary dim, regardless of dense parallel_mode.
                    # This label is the source of truth that downstream
                    # _propagate_call_function_strategy reads to distinguish
                    # embedding params from dense params during strategy
                    # propagation through the graph.
                    dp_strategy_map[node] = DataParallelStrategy(
                        [NodeType.EMBEDDING_PARAM],
                        _gen_shard_spec(mesh, shard_dim=0),
                    )
                elif (
                    replicated_param_state_indices
                    and placeholder_idx in replicated_param_state_indices
                ):
                    node_type = (
                        NodeType.PARAM
                        if placeholder_idx < num_params
                        else NodeType.STATE
                    )
                    dp_strategy_map[node] = DataParallelStrategy.from_mesh(
                        [node_type], mesh, DataParallelStyle.REPLICATE
                    )
                elif placeholder_idx < num_params:
                    dp_strategy_map[node] = DataParallelStrategy.from_mesh(
                        [NodeType.PARAM], mesh, parallel_mode
                    )
                else:
                    dp_strategy_map[node] = DataParallelStrategy.from_mesh(
                        [NodeType.STATE], mesh, parallel_mode
                    )
            else:
                activation_batch_dim_size = node.meta["val"].shape[batch_dim]
                # find the first activation node and use its batch dim size
                if batch_dim_analyzer.batch_dim_size == -1:
                    batch_dim_analyzer.init_batch_dim_size(activation_batch_dim_size)
                batch_dim_analyzer.set_batch_dim(node, batch_dim)
                shard_strategy = _gen_shard_spec(mesh, batch_dim)
                dp_strategy_map[node] = DataParallelStrategy(
                    [NodeType.ACT], shard_strategy
                )
            placeholder_idx += 1
        else:
            _propagate_call_function_strategy(
                node,
                dp_strategy_map,
                mesh,
                parallel_mode,
                batch_dim_analyzer,
                grad_type_dict,
                grad_reduction,
            )
    return dp_strategy_map


def mark_data_parallel_shardings(
    train_step_graph: GraphModule, dp_strategy_map: Dict[fx.Node, StrategyType]
) -> None:
    """Mark the sharding for the nodes in the train_step_graph."""
    placeholder_idx = 0
    for node in train_step_graph.graph.nodes:
        assert node.op in ["placeholder", "call_function", "output", "get_attr"]
        if node.target in no_ops or node.op in ("output", "get_attr"):
            node.meta["sharding"] = None
            continue
        node_strategy = dp_strategy_map[node]
        if (
            isinstance(node_strategy, DataParallelStrategy)
            and len(node_strategy.node_types) == 1
            and node_strategy.node_type == NodeType.NON_TENSOR
        ):
            node.meta["sharding"] = None
        elif node.op == "placeholder":
            assert isinstance(node_strategy, DataParallelStrategy)
            node.meta["sharding"] = node_strategy.strategies[0]  # type: ignore[possibly-undefined]
            placeholder_idx += 1
        elif isinstance(node_strategy, TupleStrategy):
            # For tuple strategy in the data parallel mode, it should have the same strategy
            # for all tuple elements, assert that then use the first element's strategy as sharding
            first_strategy = cast(DataParallelStrategy, node_strategy.children[0])
            for child_strategy in node_strategy.children:
                assert isinstance(child_strategy, DataParallelStrategy)
                assert (
                    child_strategy.strategies == first_strategy.strategies
                    or len(child_strategy.strategies) == 0
                )
            node.meta["sharding"] = first_strategy.strategies[0]
        else:
            assert isinstance(node_strategy, DataParallelStrategy)
            node.meta["sharding"] = node_strategy.strategies[0]


def partition_value(
    val: Any,
    spec: Union[DTensorSpec, Tuple[DTensorSpec]],
    *,
    own_storage: bool = False,
) -> Any:
    """Return the local value selected by ``spec``.

    ``own_storage`` gives the shard its own memory instead of a view into ``val``.
    """
    if val is None or isinstance(val, (int, float, bool, str)):
        return val
    if isinstance(val, torch.Tensor):
        spec = unique_spec(spec)
        local_shard = val
        if val.ndim == 0:
            return local_shard
        was_split = False
        for idx, placement in enumerate(spec.placements):
            if placement.is_shard():
                placement = cast(Shard, placement)
                num_chunks = spec.mesh.size(mesh_dim=idx)
                my_coord = spec.mesh.get_coordinate()
                assert my_coord is not None, "current rank not in mesh!"
                my_coord_on_mesh_dim = my_coord[idx]
                local_shard = placement._split_tensor(
                    local_shard, num_chunks, with_padding=False, contiguous=False
                )[0][my_coord_on_mesh_dim]
                was_split = was_split or num_chunks > 1
        if not own_storage or not was_split:
            return local_shard
        owned_shard = local_shard.detach().clone()
        owned_shard.requires_grad_(local_shard.requires_grad)
        return owned_shard
    elif isinstance(val, (tuple, list)):
        if not isinstance(val[0], torch.Tensor):
            # fused op (TupleStrategy), skip here
            for i in range(len(val)):
                assert not isinstance(val[i], torch.Tensor)
            return val
        elif isinstance(spec, DTensorSpec):
            return val.__class__(
                partition_value(v, spec, own_storage=own_storage) for v in val
            )
        else:
            assert len(val) == len(spec)
            return val.__class__(
                partition_value(v, s, own_storage=own_storage)
                for v, s in zip(val, spec)
            )
    elif isinstance(val, torch.ScriptObject):
        return val  # ScriptObjects (e.g., profiler._RecordFunction) don't need partitioning
    else:
        raise RuntimeError(f"val type {type(val)} not supported")


def partitioner(graph: GraphModule) -> GraphModule:  # noqa: C901
    shape_adjustment_ops = {
        aten._unsafe_view.default: 1,
        aten.expand.default: 1,
        aten.full.default: 0,
        aten.new_zeros.default: 1,
        aten.ones.default: 0,
        aten.reshape.default: 1,
        aten.slice_backward.default: 1,
        aten.view.default: 1,
        aten.zeros.default: 0,
    }
    # Maps op -> (start_arg_index or None, end_arg_index)
    arange_ops: Dict[Any, Tuple[Optional[int], int]] = {
        # arange(end, ...) - end is arg 0, no start
        aten.arange.default: (None, 0),
        # arange(start, end, ...) - start is arg 0, end is arg 1
        aten.arange.start: (0, 1),
        # arange(start, end, step, ...) - start is arg 0, end is arg 1
        aten.arange.start_step: (0, 1),
    }
    for node in graph.graph.nodes:
        node_sharding = node.meta["sharding"]
        if node_sharding is None or "val" not in node.meta:
            continue  # Skip nodes that don't have 'val' in meta (e.g., profiler nodes)
        assert node.op in ["placeholder", "call_function", "output", "get_attr"]
        if node.op == "output":
            break
        elif node.op == "placeholder":
            out_spec: DTensorSpec = unique_spec(node_sharding.output_specs)
            local_val = partition_value(node.meta["val"], out_spec)
            node.meta["val"] = local_val
        elif (
            node.target == operator.getitem
            and node.all_input_nodes[0].target == fused_adam_op
        ):
            continue
        else:
            for idx, input_node in enumerate(node.all_input_nodes):
                if input_node.meta.get("sharding") is None:
                    # Skip non-tensor inputs (e.g., get_attr constants)
                    continue
                input_spec: DTensorSpec = (
                    input_node.meta["sharding"].output_specs[node.args[1]]
                    if node.target == operator.getitem
                    else unique_spec(input_node.meta["sharding"].output_specs)
                )
                desired_spec: DTensorSpec = (
                    unique_spec(node_sharding.output_specs)
                    if node_sharding.input_specs is None
                    else node_sharding.input_specs[idx]
                )
                if (
                    input_spec == desired_spec
                    or "tensor_meta" not in input_node.meta
                    or "val" not in input_node.meta
                ):
                    continue
                input_spec.tensor_meta = input_node.meta["tensor_meta"]
                desired_spec.tensor_meta = input_node.meta["tensor_meta"]
                input_tensor = input_node.meta["val"]

                def reshard_fn(
                    local_tensor: torch.Tensor,
                    input_spec: DTensorSpec = input_spec,
                    desired_spec: DTensorSpec = desired_spec,
                ) -> torch.Tensor:
                    return redistribute_local_tensor(
                        local_tensor, input_spec, desired_spec
                    )

                reshard_gm = make_fx(reshard_fn)(input_tensor)
                reshard_gm_nodes = list(reshard_gm.graph.nodes)
                reshard_input_node = reshard_gm_nodes[0]
                with graph.graph.inserting_before(node):
                    output_node = graph.graph.graph_copy(
                        reshard_gm.graph, val_map={reshard_input_node: input_node}
                    )
                node.replace_input_with(input_node, output_node)
            output_val = node.meta["val"]
            if node.target == torch.ops.aten.repeat.default:
                out_spec: DTensorSpec = unique_spec(node_sharding.output_specs)
                assert isinstance(output_val, torch.Tensor)
                local_shape = compute_local_shape(
                    output_val.shape, out_spec.mesh, out_spec.placements
                )
                input_shape = node.args[0].meta["val"].shape

                def infer_repeat_sizes(
                    repeated_shape: Any, input_shape: Any
                ) -> List[int]:
                    repeated_size = [1] * len(repeated_shape)
                    padded_length = len(repeated_shape) - len(input_shape)
                    for i in range(len(repeated_shape)):
                        if i < padded_length:
                            repeated_size[i] = repeated_shape[i]
                        else:
                            repeated_size[i] = (
                                repeated_shape[i] // input_shape[i - padded_length]
                            )
                    return repeated_size

                node.update_arg(1, infer_repeat_sizes(local_shape, input_shape))
            elif node.target in shape_adjustment_ops:
                out_spec: DTensorSpec = unique_spec(node_sharding.output_specs)
                assert isinstance(output_val, torch.Tensor)
                local_shape = compute_local_shape(
                    output_val.shape, out_spec.mesh, out_spec.placements
                )
                shape_arg_num = shape_adjustment_ops[node.target]
                node.update_arg(shape_arg_num, local_shape)
            elif node.target in arange_ops:
                out_spec: DTensorSpec = unique_spec(node_sharding.output_specs)
                # Handle arange ops for data parallelism.
                # This handles patterns like: torch.arange(x.numel(), ...).view(x.shape)
                # where x.numel() is computed with global batch size but the view expects
                # local shape after sharding.
                # Unlike other factory ops, arange doesn't have tensor inputs to inherit
                # sharding from. The BatchDimAnalyzer marks it as replicated (-1) because
                # its 1D output shape doesn't match batch_dim_size. However, if arange
                # output flows into a view/reshape that produces a sharded tensor, we
                # need to adjust the arange end argument to produce the correct local size.
                # This follows DTensor's approach where the output placement determines
                # how operations should behave locally.
                assert isinstance(output_val, torch.Tensor)
                start_arg_idx, end_arg_idx = arange_ops[node.target]
                global_start = (
                    node.args[start_arg_idx] if start_arg_idx is not None else 0
                )
                for user in node.users:
                    if user.target in shape_adjustment_ops:
                        user_sharding = user.meta.get("sharding", None)
                        if user_sharding is not None:
                            user_output_val = user.meta["val"]
                            assert isinstance(user_output_val, torch.Tensor)
                            user_out_spec = unique_spec(user_sharding.output_specs)
                            local_shape = compute_local_shape(
                                user_output_val.shape,
                                user_out_spec.mesh,
                                user_out_spec.placements,
                            )
                            local_size = math.prod(local_shape)
                            local_end = global_start + local_size
                            node.update_arg(end_arg_idx, local_end)
                            break
            node.meta["val"] = partition_value(output_val, node_sharding.output_specs)
    for node in graph.graph.nodes:
        if "sharding" in node.meta:
            del node.meta["sharding"]
        if "val" in node.meta and isinstance(node.meta["val"], torch.Tensor):
            local_tensor_meta = _extract_tensor_metadata(node.meta["val"])
            node.meta["tensor_meta"] = local_tensor_meta
    graph.graph.lint()
    graph.recompile()
    return graph


def _group_shared_parameters(
    params_buffers: Dict[str, torch.Tensor],
) -> List[Tuple[torch.Tensor, List[str]]]:
    """Group the FQNs that name the same tensor object (weight tying)."""
    fqns_by_tensor: Dict[int, Tuple[torch.Tensor, List[str]]] = {}
    for fqn, tensor in params_buffers.items():
        tensor_id = id(tensor)
        if tensor_id not in fqns_by_tensor:
            fqns_by_tensor[tensor_id] = (tensor, [])
        fqns_by_tensor[tensor_id][1].append(fqn)
    return list(fqns_by_tensor.values())


def _select_shared_parameter_group(
    tensor: torch.Tensor,
    fqns: Sequence[str],
    shard_filter: Optional[Callable[[str, torch.Tensor], bool]],
) -> bool:
    if shard_filter is None:
        return True
    selections = {shard_filter(fqn, tensor) for fqn in fqns}
    if len(selections) != 1:
        raise RuntimeError(
            "Parameter sharding must select every name of a shared parameter "
            f"consistently: fqns={fqns!r}"
        )
    return selections == {True}


def _install_parameter_aliases(
    model: nn.Module,
    params_buffers: Dict[str, torch.Tensor],
    aliases: Sequence[str],
    full_tensor: torch.Tensor,
    local_tensor: torch.Tensor,
) -> nn.Parameter:
    copy_meta_init_attrs(local_tensor, full_tensor)
    new_param = nn.Parameter(local_tensor, requires_grad=full_tensor.requires_grad)
    copy_meta_init_attrs(new_param, full_tensor)
    for fqn in aliases:
        params_buffers[fqn] = local_tensor
        atoms = fqn.split(".")
        parent = model
        for atom in atoms[:-1]:
            parent = getattr(parent, atom)
        setattr(parent, atoms[-1], new_param)
    return new_param


def _shard_parameter_aliases(
    model: nn.Module,
    params_buffers: Dict[str, torch.Tensor],
    aliases: Sequence[str],
    full_tensor: torch.Tensor,
    shard_spec: DTensorSpec,
    *,
    materialize_meta_on_cuda: bool,
    location: str = "HBM",
    own_storage: bool = False,
) -> nn.Parameter:
    """Shard one tensor and rebind every FQN that names it."""
    local_shard = partition_value(full_tensor, shard_spec, own_storage=own_storage)
    if materialize_meta_on_cuda and local_shard.device.type == "meta":
        local_shard = _materialize_meta_shard(
            local_shard,
            full_tensor,
            torch.device(f"cuda:{torch.cuda.current_device()}"),
            location,
        )
    return _install_parameter_aliases(
        model,
        params_buffers,
        aliases,
        full_tensor,
        local_shard,
    )


def _place_table_wise_parameter_aliases(
    model: nn.Module,
    params_buffers: Dict[str, torch.Tensor],
    aliases: Sequence[str],
    full_tensor: torch.Tensor,
    *,
    owner_rank: int,
    rank: int,
    location: str = "HBM",
) -> nn.Parameter:
    owns_table = rank == owner_rank
    if owns_table:
        local_tensor = full_tensor.detach()
    else:
        local_tensor = full_tensor.new_empty((0, *full_tensor.shape[1:]))
    if local_tensor.device.type == "meta":
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        if owns_table:
            local_tensor = _materialize_meta_shard(
                local_tensor,
                full_tensor,
                device,
                location,
            )
        else:
            local_tensor = torch.empty(
                local_tensor.shape,
                dtype=local_tensor.dtype,
                device=device,
            )
    return _install_parameter_aliases(
        model,
        params_buffers,
        aliases,
        full_tensor,
        local_tensor,
    )


def shard_params_and_states(
    model: nn.Module,
    params_buffers: Dict[str, torch.Tensor],
    opt: OptOrOpts,
    mesh: DeviceMesh,
    parallel_style: DataParallelStyle,
    shard_filter: Optional[Callable[[str, torch.Tensor], bool]] = None,
    *,
    own_storage: bool = False,
) -> Set[str]:
    """Shard model parameters and optimizer states for FULLY_SHARD mode.

    For FULLY_SHARD, the partitioner rewrites the graph to expect locally
    sharded parameter/state tensors as inputs (and inserts allgather
    collectives to reconstruct full tensors before compute). We must
    actually shard the real tensors in params_buffers to match, otherwise
    the allgather will double the already-full tensors.

    ``opt`` accepts None, a single Optimizer, or a Sequence (for multi-opt
    sparse + dense setups). Each opt's state and param_groups are migrated
    from old to new Parameter identity after the swap.
    """
    if parallel_style != DataParallelStyle.FULLY_SHARD:
        return set()

    opts = _normalize_opts(opt)
    sharded_keys: Set[str] = set()
    shard_spec = DTensorSpec(mesh=mesh, placements=(Shard(0),))
    old_to_new: List[Tuple[torch.Tensor, torch.Tensor, DTensorSpec]] = []
    for old_param, fqns in _group_shared_parameters(params_buffers):
        if not _select_shared_parameter_group(old_param, fqns, shard_filter):
            continue
        sharded_keys.update(fqns)
        new_param = _shard_parameter_aliases(
            model,
            params_buffers,
            fqns,
            old_param,
            shard_spec,
            materialize_meta_on_cuda=False,
            own_storage=own_storage,
        )
        old_to_new.append((old_param, new_param, shard_spec))
    _migrate_opt_after_param_swap(
        opts,
        old_to_new,
        own_storage=own_storage,
    )
    return sharded_keys


def shard_optimizer_states_for_replicated_params(
    model: nn.Module,
    opt: OptOrOpts,
    mesh: DeviceMesh,
    *,
    parameter_filter: Callable[[str, torch.Tensor], bool],
    state_keys: FrozenSet[str],
) -> Set[Tuple[str, str]]:
    """Row-shard selected optimizer leaves without replacing parameters."""
    if not state_keys:
        raise ValueError("State-only optimizer sharding requires state keys")

    opts = _normalize_opts(opt)
    shard_spec = DTensorSpec(mesh=mesh, placements=(Shard(0),))
    sharded: Set[Tuple[str, str]] = set()
    matched_parameters = 0
    aliases_by_parameter = _group_parameter_aliases(model)
    pending_shards: List[Tuple[Dict[str, Any], str, torch.Tensor, List[str]]] = []

    for parameter, aliases in aliases_by_parameter.values():
        if not _select_shared_parameter(parameter, aliases, parameter_filter):
            continue
        matched_parameters += 1
        state = _optimizer_state_for_parameter(opts, parameter, aliases)
        for state_key in state_keys:
            value = state.get(state_key)
            if value is None:
                raise RuntimeError(
                    "State-only optimizer sharding is missing required state "
                    f"{aliases!r}/{state_key!r}"
                )
            if not isinstance(value, torch.Tensor):
                raise RuntimeError(
                    f"Optimizer state {aliases!r}/{state_key!r} must be a tensor"
                )
            if value.ndim == 0 or value.shape != parameter.shape:
                raise RuntimeError(
                    "State-only optimizer sharding requires parameter-shaped "
                    f"ranked tensors: {aliases!r}/{state_key!r} has "
                    f"shape {tuple(value.shape)}, parameter has "
                    f"shape {tuple(parameter.shape)}"
                )
            pending_shards.append((state, state_key, value, aliases))
    if matched_parameters == 0:
        raise RuntimeError("State-only optimizer sharding matched no parameters")
    for state, state_key, value, aliases in pending_shards:
        state[state_key] = partition_value(value, shard_spec)
        sharded.update((fqn, state_key) for fqn in aliases)
    return sharded


def _group_parameter_aliases(
    model: nn.Module,
) -> Dict[int, Tuple[nn.Parameter, List[str]]]:
    aliases_by_parameter: Dict[int, Tuple[nn.Parameter, List[str]]] = {}
    for fqn, parameter in model.named_parameters(remove_duplicate=False):
        parameter_id = id(parameter)
        if parameter_id not in aliases_by_parameter:
            aliases_by_parameter[parameter_id] = (parameter, [])
        aliases_by_parameter[parameter_id][1].append(fqn)
    return aliases_by_parameter


def _select_shared_parameter(
    parameter: nn.Parameter,
    aliases: Sequence[str],
    parameter_filter: Callable[[str, torch.Tensor], bool],
) -> bool:
    selections = {parameter_filter(fqn, parameter) for fqn in aliases}
    if len(selections) != 1:
        raise RuntimeError(
            "State-only optimizer sharding must select every alias of a "
            f"shared parameter consistently: aliases={aliases!r}"
        )
    return selections == {True}


def _optimizer_state_for_parameter(
    opts: Sequence[torch.optim.Optimizer],
    parameter: nn.Parameter,
    aliases: Sequence[str],
) -> Dict[str, Any]:
    owners = [
        optimizer
        for optimizer in opts
        if any(
            candidate is parameter
            for group in optimizer.param_groups
            for candidate in group["params"]
        )
    ]
    if len(owners) > 1:
        raise RuntimeError(
            f"Optimizer state for {aliases!r} belongs to multiple optimizers"
        )
    if not owners:
        raise RuntimeError(
            f"State-only optimizer sharding found no state owner for {aliases!r}"
        )
    if parameter not in owners[0].state:
        raise RuntimeError(
            "State-only optimizer sharding found uninitialized optimizer state for "
            f"{aliases!r}; initialize the configured state keys before sharding"
        )
    return owners[0].state[parameter]


def _clone_param_chains_for_backward(  # noqa: C901
    graph: GraphModule, strategy_map: Dict[fx.Node, StrategyType]
) -> None:
    """Clone non-compute-op chains (t, view, reshape, ...) so each compute
    consumer gets its own allgather, enabling per-op weight lifetime.

    Without cloning, a shared chain like ``t(W)`` consumed by both forward
    ``addmm`` and backward ``mm`` produces one allgather whose full weight
    lives until all consumers finish. After cloning, each consumer gets a
    separate chain with ``Shard(0)`` spec, triggering independent allgathers.

    Algorithm:
      1. Find PARAM placeholders from the strategy map.
      2. Group each param's chain-op children into branches. Collect
         ``(chain_node, [compute_consumers])`` pairs per branch.
      3. Per branch, the earliest consumer keeps the original chain.
         Later consumers get cloned chains with ``Shard(0)`` spec.
      4. The partitioner sees ``Shard(0)`` vs ``Replicate()`` mismatches
         and inserts separate allgathers for each clone.

    Per-branch (not global) earliest preserves each branch's forward
    consumer. Branch-level (not per-node) earliest handles linear chains
    like ``W → t → t(t)`` where deeper nodes have only one consumer each.

    Before::

        W ─→ t(W) ─┬→ addmm (fwd)    One allgather, weight alive fwd→bwd.
                    └→ mm    (bwd)

    After::

        W ─┬→ t(W)  ─→ addmm (fwd)   allgather #1, freed after addmm
           └→ t(W)' ─→ mm    (bwd)   allgather #2, freed after mm
    """
    # Build topological ordering for finding earliest consumers.
    node_order: Dict[fx.Node, int] = {}
    for idx, node in enumerate(graph.graph.nodes):
        node_order[node] = idx

    param_chain_ops = {
        aten._to_copy.default,
        aten.clone.default,
        aten.detach.default,
        aten.reshape.default,
        aten.t.default,
        aten.view.default,
    }

    # Find PARAM placeholders using NodeType from the strategy map.
    param_placeholders = []
    for node in graph.graph.nodes:
        if node.op != "placeholder":
            break
        strategy = strategy_map.get(node)
        if not isinstance(strategy, DataParallelStrategy):
            continue
        if strategy.node_type == NodeType.PARAM:
            param_placeholders.append(node)

    for param_node in param_placeholders:
        # Group chain nodes into branches (one per direct chain-op child).
        branches: List[List[Tuple[fx.Node, List[fx.Node]]]] = []
        for user in list(param_node.users):
            if user.target not in param_chain_ops:
                continue
            branch_consumers: List[Tuple[fx.Node, List[fx.Node]]] = []
            compute_consumers = [
                c for c in list(user.users) if c.target not in param_chain_ops
            ]
            if compute_consumers:
                branch_consumers.append((user, compute_consumers))
            _collect_chain_nodes(user, param_chain_ops, branch_consumers)
            if branch_consumers:
                branches.append(branch_consumers)

        # Direct compute consumers of param (no chain ops) form a branch too.
        direct_compute = [
            u
            for u in list(param_node.users)
            if u.target not in param_chain_ops and u.op == "call_function"
        ]
        if direct_compute and branches:
            branches.append([(param_node, direct_compute)])

        for branch_consumers in branches:
            earliest_consumer = min(
                (c for _, consumers in branch_consumers for c in consumers),
                key=lambda c: node_order[c],
            )

            for chain_node, consumers in branch_consumers:
                later_consumers = [c for c in consumers if c is not earliest_consumer]
                if not later_consumers:
                    continue

                path = _build_chain_path(param_node, chain_node, param_chain_ops)
                if path is None:
                    continue

                # Clone the path from param to chain_node.
                clone_map = {param_node: param_node}
                for path_node in path[1:]:
                    with graph.graph.inserting_before(chain_node):
                        new_args = tuple(
                            clone_map.get(a, a) if isinstance(a, fx.Node) else a
                            for a in path_node.args
                        )
                        new_kwargs = {
                            k: clone_map.get(v, v) if isinstance(v, fx.Node) else v
                            for k, v in path_node.kwargs.items()
                        }
                        cloned = graph.graph.call_function(
                            cast(Callable[..., Any], path_node.target),
                            args=new_args,
                            kwargs=new_kwargs,
                        )
                        cloned.meta = {**path_node.meta}
                        cloned.meta["sharding"] = path_node.meta["sharding"]
                        clone_map[path_node] = cloned

                # Redirect later consumers to the cloned chain end.
                cloned_end = clone_map[chain_node]
                for consumer in later_consumers:
                    consumer.replace_input_with(chain_node, cloned_end)


def _collect_chain_nodes(
    node: fx.Node,
    param_chain_ops: Set[Any],
    result: List[Tuple[fx.Node, List[fx.Node]]],
) -> None:
    """Recursively walk non-compute-op users of *node* and collect those with compute consumers."""
    for user in list(node.users):
        if user.target in param_chain_ops:
            compute_consumers = [
                consumer
                for consumer in list(user.users)
                if consumer.target not in param_chain_ops
            ]
            if compute_consumers:
                result.append((user, compute_consumers))
            _collect_chain_nodes(user, param_chain_ops, result)


def _build_chain_path(
    start: fx.Node, end: fx.Node, param_chain_ops: Set[Any]
) -> Optional[List[fx.Node]]:
    """Return the list of nodes on the non-compute-op path from *start* to *end*, inclusive."""
    if start is end:
        return [start]
    for user in start.users:
        if user.target in param_chain_ops or user is end:
            sub = _build_chain_path(user, end, param_chain_ops)
            if sub is not None:
                return [start] + sub
    return None


def materialize_embedding_weight(
    shape: Sequence[int],
    dtype: torch.dtype,
    device: torch.device,
    *,
    location: str = "HBM",
    init_kind: str = "zeros",
    init_bound: float = 0.0,
) -> torch.Tensor:
    """Allocate and initialize an embedding weight on ``device``.

    UVM locations use FBGEMM managed memory when available; otherwise the
    regular PyTorch device allocator is used. Initialization follows the shared
    recir meta-init specification.
    """
    weight: Optional[torch.Tensor] = None
    if location in ("UVM", "UVM_CACHING"):
        try:
            weight = torch.ops.fbgemm.new_managed_tensor(
                torch.empty(0, dtype=dtype, device=device),
                list(shape),
            )
        except (RuntimeError, AttributeError):
            weight = None
    if weight is None:
        weight = torch.empty(tuple(shape), dtype=dtype, device=device)
    return initialize_tensor_from_meta_init(weight, init_kind, init_bound)


def _materialize_meta_shard(
    meta_shard: torch.Tensor,
    source_tensor: torch.Tensor,
    device: torch.device,
    location: str = "HBM",
) -> torch.Tensor:
    init_kind, init_bound = get_meta_init_spec(
        source_tensor,
        default_embedding_init_bound(source_tensor.shape[0]),
    )
    return materialize_embedding_weight(
        meta_shard.shape,
        meta_shard.dtype,
        device,
        location=location,
        init_kind=init_kind,
        init_bound=init_bound,
    )


def _shard_state_dict(
    state: Any,
    shard_spec: DTensorSpec,
    state_device: Optional[torch.device] = None,
    *,
    own_storage: bool = False,
) -> None:
    """Shard optimizer state and materialize meta shards on the parameter device."""
    if isinstance(state, dict):
        for key in state:
            if isinstance(state[key], torch.Tensor):
                local_state = partition_value(
                    state[key], shard_spec, own_storage=own_storage
                )
                if (
                    local_state.device.type == "meta"
                    and state_device is not None
                    and state_device.type != "meta"
                ):
                    local_state = torch.zeros_like(local_state, device=state_device)
                else:
                    mark_meta_zero_init(local_state)
                state[key] = local_state


def _normalize_opts(
    opt: OptOrOpts,
) -> List[torch.optim.Optimizer]:
    """Normalize a single-or-list-or-None opt argument to a list."""
    if opt is None:
        return []
    if isinstance(opt, torch.optim.Optimizer):
        return [opt]
    return list(opt)


def _replace_optimizer_parameters(
    opt: torch.optim.Optimizer,
    replacements: Dict[int, torch.Tensor],
) -> None:
    for group in opt.param_groups:
        group["params"] = [
            replacements.get(id(param), param) for param in group["params"]
        ]


def _migrate_opt_after_param_swap(
    opts: List[torch.optim.Optimizer],
    old_to_new: List[Tuple[torch.Tensor, torch.Tensor, DTensorSpec]],
    *,
    own_storage: bool = False,
) -> None:
    """For each opt, migrate state[old_param] -> state[new_param] (with
    state tensor sharding) AND rewrite every param_group's `params`
    list to swap old_param -> new_param. Each entry in ``old_to_new``
    is ``(old_param, new_param, shard_spec_used_for_old)``.

    Called ONCE per sharding pass with the full set of (old, new) pairs
    that the pass created; the sharding itself only runs once even
    though we loop over multiple optimizers here.
    """
    for opt in opts:
        for old_param, new_param, shard_spec in old_to_new:
            if old_param in opt.state:
                state = opt.state.pop(old_param)
                _shard_state_dict(
                    state,
                    shard_spec,
                    new_param.device,
                    own_storage=own_storage,
                )
                opt.state[new_param] = state
        # param_groups rewrite is unconditional - even if a group's
        # params list doesn't contain any of the swapped params, the
        # list comprehension is a no-op for unaffected entries.
        id_to_new: Dict[int, torch.Tensor] = {
            id(old): new for old, new, _ in old_to_new
        }
        _replace_optimizer_parameters(opt, id_to_new)


def _migrate_table_wise_opt_after_param_swaps(
    opts: Sequence[torch.optim.Optimizer],
    old_to_new: Sequence[Tuple[torch.Tensor, torch.Tensor, bool]],
) -> None:
    for opt in opts:
        replacements: Dict[int, torch.Tensor] = {}
        for old_param, new_param, owns_table in old_to_new:
            state = opt.state.pop(old_param, None)
            if owns_table and state is not None:
                if isinstance(state, dict):
                    for key, value in state.items():
                        if (
                            isinstance(value, torch.Tensor)
                            and value.device.type == "meta"
                        ):
                            state[key] = torch.zeros_like(
                                value, device=new_param.device
                            )
                opt.state[new_param] = state
            replacements[id(old_param)] = new_param
        _replace_optimizer_parameters(opt, replacements)


def _embedding_group_placement(
    aliases: Sequence[str],
    embedding_plan: EmbeddingShardingPlan,
) -> Tuple[str, int | None] | None:
    planned_aliases = [
        alias for alias in aliases if embedding_plan.is_embedding_param(alias)
    ]
    if not planned_aliases:
        return None
    if len(planned_aliases) != len(aliases):
        raise RuntimeError(
            "Shared parameter aliases must all be present in the embedding plan: "
            f"aliases={aliases!r}, planned={planned_aliases!r}."
        )
    sharding_types = {embedding_plan.sharding_map[alias] for alias in planned_aliases}
    if len(sharding_types) != 1:
        raise RuntimeError(
            "Shared parameter aliases must use one embedding sharding type: "
            f"aliases={aliases!r}."
        )
    resolved_locations = {
        embedding_plan.location_map.get(alias, "HBM") for alias in aliases
    }
    if len(resolved_locations) != 1:
        raise RuntimeError(
            "A shared parameter must use one embedding storage location: "
            f"fqns={aliases!r}, locations={sorted(resolved_locations)!r}"
        )
    location = next(iter(resolved_locations))
    if not embedding_plan.is_table_wise(planned_aliases[0]):
        return location, None
    owner_ranks = {embedding_plan.table_wise_owner(alias) for alias in planned_aliases}
    if len(owner_ranks) != 1:
        raise RuntimeError(
            "Shared table-wise parameter aliases must have one owner rank: "
            f"aliases={aliases!r}, owners={sorted(owner_ranks)!r}."
        )
    return location, next(iter(owner_ranks))


def _validate_table_wise_owners(
    embedding_plan: EmbeddingShardingPlan,
    world_size: int,
) -> None:
    for fqn, owner_rank in embedding_plan.table_wise_rank_map.items():
        if owner_rank >= world_size:
            raise ValueError(
                f"Table-wise owner rank {owner_rank} for {fqn!r} is outside "
                f"world size {world_size}."
            )


def _validate_embedding_plan(
    model: nn.Module,
    params_buffers: Dict[str, torch.Tensor],
    mesh: DeviceMesh,
    embedding_plan: EmbeddingShardingPlan,
) -> Tuple[int | None, Dict[int, Tuple[str, int | None]]]:
    embedding_fqns = embedding_plan.param_fqns
    model_parameter_fqns = {
        fqn for fqn, _ in model.named_parameters(remove_duplicate=False)
    }
    non_parameter_fqns = embedding_fqns - model_parameter_fqns
    if non_parameter_fqns:
        raise ValueError(
            "Embedding plan contains FQNs that are not model parameters: "
            f"{sorted(non_parameter_fqns)!r}."
        )
    missing_fqns = embedding_fqns - set(params_buffers)
    if missing_fqns:
        raise ValueError(
            "Embedding plan contains parameters absent from the state snapshot: "
            f"{sorted(missing_fqns)!r}."
        )
    group_placements: Dict[int, Tuple[str, int | None]] = {}
    for parameter, aliases in _group_shared_parameters(params_buffers):
        placement = _embedding_group_placement(aliases, embedding_plan)
        if placement is not None:
            group_placements[id(parameter)] = placement

    rank: int | None = None
    if embedding_plan.table_wise_rank_map:
        if mesh.ndim != 1:
            raise ValueError(
                "Table-wise embedding sharding requires a one-dimensional device mesh; "
                f"got shape {mesh.shape}."
            )
        _validate_table_wise_owners(embedding_plan, mesh.size(0))
        rank_coordinate = mesh.get_coordinate()
        if rank_coordinate is None:
            raise RuntimeError("Current rank is not part of the embedding device mesh.")
        rank = rank_coordinate[0]
    return rank, group_placements


def shard_params_by_plan(
    model: nn.Module,
    params_buffers: Dict[str, torch.Tensor],
    opt: OptOrOpts,
    mesh: DeviceMesh,
    dense_style: DataParallelStyle,
    *,
    embedding_plan: EmbeddingShardingPlan,
    dense_shard_param_fqns: Optional[Set[str]] = None,
) -> Set[str]:
    """Shard parameters and optimizer states with per-param logic.

    Each parameter is sharded exactly once based on its role:
      - Row-wise embedding params: ``Shard(0)`` on the vocabulary dimension.
      - Table-wise embedding params: the owner rank keeps the complete table;
        non-owner ranks keep an empty local placeholder.
      - Dense params: ``Shard(0)`` in ``FULLY_SHARD`` mode when selected by
        ``dense_shard_param_fqns``; otherwise they remain replicated.

    ``embedding_plan`` is the sole source of embedding FQNs, sharding types,
    table owners, and storage locations. When a meta-device embedding is
    materialized, UVM tables use FBGEMM's managed allocator (host RAM + GPU
    direct access), matching DMP/TorchRec's placement.

    ``opt`` accepts None, a single Optimizer, or a Sequence of Optimizers
    (for sparse + dense multi-opt setups). For each opt passed, the
    optimizer's ``state`` dict and every ``param_group['params']`` list
    are migrated from old Parameter identity to new Parameter identity
    after the shard-driven Parameter object swap.
    """
    embedding_fqns = embedding_plan.param_fqns
    rank, group_placements = _validate_embedding_plan(
        model, params_buffers, mesh, embedding_plan
    )

    opts = _normalize_opts(opt)
    sharded_keys: Set[str] = set()
    shard_spec = DTensorSpec(mesh=mesh, placements=(Shard(0),))
    old_to_new: List[Tuple[torch.Tensor, torch.Tensor, DTensorSpec]] = []
    table_wise_old_to_new: List[Tuple[torch.Tensor, torch.Tensor, bool]] = []

    def should_shard(key: str, _tensor: torch.Tensor) -> bool:
        return key in embedding_fqns or (
            dense_style == DataParallelStyle.FULLY_SHARD
            and (dense_shard_param_fqns is None or key in dense_shard_param_fqns)
        )

    for old_param, aliases in _group_shared_parameters(params_buffers):
        if not _select_shared_parameter_group(old_param, aliases, should_shard):
            continue
        location, owner_rank = group_placements.get(id(old_param), ("HBM", None))
        sharded_keys.update(aliases)
        if owner_rank is not None:
            if rank is None:
                raise RuntimeError(
                    "Table-wise placement requires the current mesh rank."
                )
            new_param = _place_table_wise_parameter_aliases(
                model,
                params_buffers,
                aliases,
                old_param,
                owner_rank=owner_rank,
                rank=rank,
                location=location,
            )
            table_wise_old_to_new.append((old_param, new_param, rank == owner_rank))
            continue
        new_param = _shard_parameter_aliases(
            model,
            params_buffers,
            aliases,
            old_param,
            shard_spec,
            materialize_meta_on_cuda=True,
            location=location,
        )
        old_to_new.append((old_param, new_param, shard_spec))
    _migrate_table_wise_opt_after_param_swaps(opts, table_wise_old_to_new)
    _migrate_opt_after_param_swap(opts, old_to_new)

    # DDP synchronizes replicated parameters and buffers from rank 0 during construction
    if mesh.size() > 1 and torch.distributed.is_initialized():
        process_group = mesh.get_group()
        source_rank = int(mesh.mesh.flatten()[0].item())
        with torch.no_grad():
            for key, value in params_buffers.items():
                if key not in sharded_keys:
                    # NCCL requires staging CPU buffers on the mesh device.
                    broadcast_value = (
                        value
                        if value.device.type == mesh.device_type
                        else value.to(mesh.device_type)
                    )
                    torch.distributed.broadcast(
                        broadcast_value, src=source_rank, group=process_group
                    )
                    if broadcast_value is not value:
                        value.copy_(broadcast_value.to(value.device))
    return sharded_keys


def _rewrite_row_wise_embeddings(
    graph: GraphModule,
    mesh: DeviceMesh,
    strategy_map: Dict[fx.Node, StrategyType],
) -> None:
    """Replace ``aten.embedding`` nodes whose weight is EMBEDDING_PARAM with
    decomposed row-wise ops (bucketize → AllToAll → lookup → AllToAll → unbucketize).

    Each replacement is produced by tracing :func:`decomposed_row_wise_embedding`
    with ``make_fx(tracing_mode="symbolic")`` and inlining the resulting sub-graph.
    The new nodes all get ``meta["sharding"] = None`` so the downstream partitioner
    skips them (the AllToAll collectives are already explicit).

    Must be called BEFORE the partitioner so that:
      1. The partitioner never sees aten.embedding with mismatched specs.
      2. meta["val"] still has global shapes (no need to reverse-engineer
         global vocab size from local shard shapes).
    """
    from functools import partial

    world_size = mesh.size()
    group_name = funcol._resolve_group_name((mesh, 0))

    # Identify embedding nodes by checking the weight input's strategy type
    # in the strategy map, rather than relying on output node_types (which
    # are always ACT for compute ops).
    embedding_nodes: List[fx.Node] = []
    for node in graph.graph.nodes:
        if (
            node.op == "call_function"
            and node.target == torch.ops.aten.embedding.default
        ):
            weight_node = node.args[0]
            weight_strategy = strategy_map.get(weight_node)
            if (
                isinstance(weight_strategy, DataParallelStrategy)
                and weight_strategy.node_type == NodeType.EMBEDDING_PARAM
            ):
                embedding_nodes.append(node)

    for emb_node in embedding_nodes:
        weight_node = cast(fx.Node, emb_node.args[0])
        indices_node = cast(fx.Node, emb_node.args[1])

        weight_val = weight_node.meta["val"]
        indices_val = indices_node.meta["val"]
        num_embeddings = weight_val.shape[0]  # global vocab size

        # Trace the decomposed function to get a sub-graph of raw ops.
        # Create fresh example tensors (not FakeTensors) so make_fx can
        # set up its own FakeTensorMode + ShapeEnv for symbolic tracing.
        example_weight = torch.randn(
            weight_val.shape, dtype=weight_val.dtype, device="cpu"
        )
        example_indices = torch.randint(
            0, num_embeddings, indices_val.shape, device="cpu"
        )
        decomp_fn = partial(
            decomposed_row_wise_embedding,
            num_embeddings=num_embeddings,
            group_name=group_name,
            world_size=world_size,
        )
        subgraph_gm = make_fx(decomp_fn, tracing_mode="symbolic")(
            example_weight, example_indices
        )

        # Map sub-graph placeholders to existing graph nodes.
        placeholders = [n for n in subgraph_gm.graph.nodes if n.op == "placeholder"]
        val_map: Dict[fx.Node, fx.Node] = {
            placeholders[0]: weight_node,
            placeholders[1]: indices_node,
        }

        # Inline sub-graph before the embedding node.
        with graph.graph.inserting_before(emb_node):
            output_value = graph.graph.graph_copy(subgraph_gm.graph, val_map=val_map)

        # Mark newly-inlined nodes with sharding=None so the partitioner
        # skips them.
        for node in graph.graph.nodes:
            if "sharding" not in node.meta:
                node.meta["sharding"] = None
                strategy_map[node] = DataParallelStrategy([NodeType.ACT], None)

        # Replace the original embedding node.
        assert isinstance(output_value, fx.Node)
        emb_node.replace_all_uses_with(output_value)
        graph.graph.erase_node(emb_node)
        # Remove the now-dead strategy entry.
        strategy_map.pop(emb_node, None)


def partition_data_parallel(
    graph: GraphModule,
    *,
    params: Dict[str, nn.Parameter],
    buffers: Dict[str, torch.Tensor],
    named_states: Dict[str, Dict[str, Any]],
    mesh: DeviceMesh,
    parallel_style: DataParallelStyle,
    input_batch_dim: int,
    grad_reduction: str = "avg",
    embedding_param_indices: Optional[Set[int]] = None,
    replicated_param_state_indices: Optional[Set[int]] = None,
) -> GraphModule:
    """Partition the graph into a data parallel graph.

    Inputs are FQN-keyed dicts so callers without a live ``nn.Module`` (e.g.
    a bundle-runtime consumer that loads safetensors at job start) can drive
    partitioning directly.  ``named_states`` maps each parameter FQN to its
    optimizer state dict; ``step`` may be an int scalar, a tensor, or ``None``
    depending on the optimizer flavor.
    """
    num_params_buffers = len(params) + len(buffers)
    flattened_states = pytree.tree_leaves(named_states)
    num_states = len(flattened_states)
    changed = graph.graph.eliminate_dead_code()
    if changed:
        graph.recompile()
    strategy_map = build_data_parallel_strategies(
        graph,
        num_params_buffers,
        num_states,
        mesh=mesh,
        batch_dim=input_batch_dim,
        parallel_mode=parallel_style,
        grad_reduction=grad_reduction,
        embedding_param_indices=embedding_param_indices,
        replicated_param_state_indices=replicated_param_state_indices,
    )
    mark_data_parallel_shardings(graph, dp_strategy_map=strategy_map)

    # Rewrite embedding nodes BEFORE the partitioner.  At this point
    # meta["val"] still carries global shapes and meta["sharding"] has been
    # set by mark_data_parallel_shardings.  Rewriting here guarantees the
    # partitioner never sees aten.embedding with an EMBEDDING_PARAM weight
    # (which would have mismatched Shard(0)/Replicate specs).
    if embedding_param_indices is not None:
        _rewrite_row_wise_embeddings(graph, mesh, strategy_map)

    # Clone param chains so backward gets its own allgathers (per-op FSDP).
    if parallel_style == DataParallelStyle.FULLY_SHARD:
        _clone_param_chains_for_backward(graph, strategy_map)
        graph.graph.eliminate_dead_code()
        graph.recompile()

    partitioned_graph = partitioner(graph)
    # preserve node types for the distributed graph
    for node in partitioned_graph.graph.nodes:
        if node in strategy_map:
            node_strategy = strategy_map[node]
            if isinstance(node_strategy, DataParallelStrategy):
                node.meta["node_types"] = node_strategy.node_types
            elif isinstance(node_strategy, TupleStrategy):
                node.meta["node_types"] = [NodeType.NON_TENSOR]
            else:
                raise RuntimeError(f"Unknown node strategy {node_strategy}")
        elif node.target not in no_ops:
            # If the nodes are collectives, we mark them the same type as the input node.
            input_node = node.all_input_nodes[0]
            node.meta["node_types"] = input_node.meta["node_types"]

    return partitioned_graph
