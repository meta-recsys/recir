# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import contextmanager, ExitStack, nullcontext
from copy import copy
from dataclasses import dataclass
from functools import partial, wraps
from typing import Any, Callable, cast, Dict, Generator, List, Optional, Tuple, Union

import torch

# We need to import _functional_collectives to trigger op registration
import torch.distributed._functional_collectives
import torch.nn as nn
import torch.utils._pytree as pytree
from functorch import make_fx
from recir.batch_dim_propagation import propagate_batch_input_meta
from recir.decompositions import DEDUP_TARGETS, SPMD_DECOMP_TABLE
from recir.distributed.data_parallel import (
    collectives_insertion,
    DataParallelStyle,
    gradients_tagging,
    parametrize_params,
    rewrite_sparse_embeddings,
)
from recir.distributed.parallel_mode import (
    DataParallel,
    GSPMDParallel,
    HybridParallel,
    ParallelMode,
)
from recir.lowering.device_state import DeviceStateManager
from recir.region_transforms import (
    _apply_region_transforms,
    _discover_autograd_functions,
    _discover_called_modules,
    _find_forward_roots,
    RegatherTransform,
    RegionTransform,
)
from torch import fx
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.export.unflatten import _assign_attr, _AttrKind
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo, CodeGen
from torch.fx.graph_module import _del_attr, _get_attr
from torch.nn.utils import stateless
from torch.nn.utils._named_member_accessor import NamedMemberAccessor

logger: logging.Logger = logging.getLogger(__name__)


class Override(ABC):
    r"""Override the tracing and transformation behavior of :meth:`~torch.distributed._spmd.compile`.

    This is useful when any part of the model is not traceable or if you prefer
    to not trace it due to any reason. More specifically, users can implement
    :meth:`torch.distributed._spmd.Override.replacement` to replace an original
    submodule with the return new submodule. The new submodule contains
    operations that users preferred to be traced, which simply be a dummy
    placeholder operator. After tracing, users can implement
    :meth:`torch.distributed._spmd.Override.transform` to transform the traced
    graph, where the dummy placeholder operator serves as an anchor to insert
    new sub-graphs.

    For overrides that need integrated backward support (i.e., the override's
    backward should fire within the compiled ``train_step`` automatically),
    use :func:`recir.override_ops.replace_override_anchors` inside
    ``transform()`` to replace anchor ops with runtime dispatch calls.
    """

    @abstractmethod
    def replacement(self, fqn: str, orig_submodule: torch.nn.Module) -> torch.nn.Module:
        r"""Implement this method to return a new :class:`nn.Module` instance to replace the ``orig_submodule``
        argument in the model.

        This helps if ``orig_submodule`` is not traceable or should not be traced.

        Args:
            fqn (str): fully quantified name of the submodule.
            orig_submodule (class:`nn.Module`): original submodule instance to replace.

        Returns:
            A new :class:`nn.Module` instance to replace the original one.

        """
        pass

    @abstractmethod
    def transform(
        self,
        gm: fx.GraphModule,
        flat_state: List[torch.Tensor],
    ) -> fx.GraphModule:
        r"""
        Given a DTensor-expanded graph and sharding schema for every node,
        conduct additional transformation for the sub-graph from the :class:`nn.Module`
        returned by :meth:`torch.distributed._spmd.Override.replacement` if
        necessary.

        Args:
            gm (:class:`fx.Graph`): a DTensor-expanded graph.
            flat_state (List[str, :class:`Tensor`]): a reference to the list of
                flattened state. The elements in ``flat_state`` map to the first
                ``len(flat_state)`` placeholders in the graph. The transformation
                can add state to or remove state from ``flat_state`` as long as
                it keeps ``flat_state`` and the placeholders consistent.

        Returns:
            The :class:`fx.Graph` after transformation.

        """
        pass


class _PyTreeCodeGenOutputsOnly(_PyTreeCodeGen):
    # pyre-ignore[3]
    def process_inputs(self, *args: Any) -> Any:
        return args

    # pyre-ignore[2, 3, 14]
    def gen_fn_def(self, free_vars, maybe_return_annotation, expanded_def=False):
        return CodeGen.gen_fn_def(self, free_vars, maybe_return_annotation)


def _to_caller_flattened_graph_module(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Move the responsibility of flattening the input arguments from the graph module to the caller.

    Example:

        output = gm(my_struct)

        gm = gm(to_caller_flattened_graph_module)

        output = gm(*pytree.flatten(my_struct)[0])

    """
    # pyre-ignore
    gm._graph._codegen = _PyTreeCodeGenOutputsOnly(
        pytree_info=_PyTreeInfo(
            # pyre-ignore[6]
            orig_args=None,  # type: ignore[arg-type]
            # pyre-ignore[6]
            in_spec=None,  # type: ignore[arg-type]
            # pyre-ignore[16]
            out_spec=gm._graph._codegen.pytree_info.out_spec,
        )
    )
    gm.recompile()
    return gm


@contextmanager
def _rematerialize_optimizer(
    opt: torch.optim.Optimizer,
    named_states: Dict[str, Any],
    params: Dict[str, nn.Parameter],
) -> Generator[None, None, None]:
    assert opt is not None

    # update opt.state with proxy tensors
    orig_states = copy(opt.state)
    for n in named_states:
        # opt.state's key type is string, but optimizer uses Parameter as keys
        opt.state[params[n]] = named_states[n]  # type: ignore[index]

    # FIXME: support multiple parameter groups
    param_group = opt.param_groups[0]
    orig_params = param_group["params"]
    param_group["params"] = params.values()

    try:
        yield
    finally:
        param_group["params"] = orig_params
        opt.state = orig_states


aten = torch.ops.aten  # pyre-ignore


@contextmanager
def _enable_compile() -> Generator[None, None, None]:
    # The return value of torch._utils.is_compiling changes optimizer behavior.
    # We need that function to return True to include optimizer in the graph.
    # See: https://github.com/pytorch/pytorch/blob/a524123c91ab399c9dd6882c1189596dd77e7734/torch/optim/optimizer.py#L41
    def f_true() -> bool:
        return True

    orig_is_compiling_code = torch._utils.is_compiling.__code__  # pyre-ignore[16]
    torch._utils.is_compiling.__code__ = f_true.__code__  # pyre-ignore[16]
    # PyTorch also checks torch.compiler.is_compiling() which
    # reads _is_compiling_flag.  Setting this flag lets the optimizer skip
    # device assertions for capturable mode and avoid .item() calls in
    # _get_value() during tracing.
    orig_compiling_flag = torch.compiler._is_compiling_flag  # pyre-ignore[16]
    torch.compiler._is_compiling_flag = True  # pyre-ignore[16]
    try:
        yield
    finally:
        torch._utils.is_compiling.__code__ = orig_is_compiling_code  # pyre-ignore[16]
        torch.compiler._is_compiling_flag = orig_compiling_flag  # pyre-ignore[16]


def _dedup_collectives(gm: fx.GraphModule) -> fx.GraphModule:
    args_to_node: Dict[Tuple[Any, ...], fx.Node] = {}

    for node in gm.graph.nodes:
        # replace all args with the results from the first unique comm op
        args = pytree.arg_tree_leaves(*node.args)

        if node.target in DEDUP_TARGETS:
            args_key = (node.target, *args)
            unique_node = args_to_node.get(args_key, None)
            if unique_node is None:
                # first time seeing this combination, remember it
                args_to_node[args_key] = node
            else:
                # the current node is a duplicate, replace it
                node.replace_all_uses_with(unique_node)
                gm.graph.erase_node(node)

    gm.recompile()

    return gm


@dataclass
class _CompiledResult:
    gm: fx.GraphModule
    mod: nn.Module
    opt: Optional[torch.optim.Optimizer]
    flat_state: List[torch.Tensor]
    state_manager: DeviceStateManager


@dataclass
class _LocalCompiledResult:
    gm: fx.GraphModule
    mod: nn.Module
    opt: Optional[torch.optim.Optimizer]
    params: Dict[str, nn.Parameter]
    buffers: Dict[str, torch.Tensor]
    named_states: Dict[str, Any]
    # Tensor constants referenced via ``get_attr`` nodes in ``gm`` (e.g. inline
    # tensor literals lifted by ``make_fx``). Keyed by fully-qualified attribute
    # name on ``gm`` so downstream serde / inspection can recover them. These
    # are stored as gm attributes (not placeholders) and registered via
    # ``_AttrKind.CONSTANT`` so they survive ``gm.recompile()`` cleanly.
    constants: Dict[str, torch.Tensor]


def _apply_module_overrides(
    mod: nn.Module,
    module_override: List[Override],
) -> None:
    """Swap non-traceable submodules with override replacements in-place."""
    accessor: NamedMemberAccessor = NamedMemberAccessor(mod)

    def swap(fqn_prefix: str, module: torch.nn.Module) -> None:
        for override in module_override:
            for name, child in module.named_children():
                if len(name) == 0:
                    continue
                fqn = fqn_prefix + "." + name if fqn_prefix != "" else name
                new_child = override.replacement(fqn, child)
                if id(new_child) == id(child):
                    swap(fqn, new_child)
                else:
                    accessor.swap_submodule(fqn, new_child)

    swap("", mod)


def _local_compile(
    func: Callable[..., Any],
    *args: Any,
    fake_mode: FakeTensorMode,
    _parametrize_fns: Optional[List[Callable[[torch.Tensor], torch.Tensor]]] = None,
    _exclude_param_fqns: Optional[set[str]] = None,
    _stateless_func_wrapper: Optional[Callable[..., Any]] = None,
    **kwargs: Any,
) -> _LocalCompiledResult:
    """Trace a train_step into a local FX graph.

    This performs the local tracing portion of _compile:
    1. Extract nn.Module and Optimizer from args
    2. Trace with make_fx using FakeTensorMode

    Does NOT require dist.init_process_group by default. Does NOT perform
    DataParallel partitioning, collective dedup, or state flattening.

    Args:
        func: The train_step function to trace.
        *args: Arguments to the train_step function (must include nn.Module and Optimizer).
        fake_mode: The FakeTensorMode to use for tracing. Each compile path creates
            its own mode and passes it here. Callers are responsible for
            converting inputs to FakeTensors before calling this function.
        _parametrize_fns: Optional list of per-parameter transform functions.
            Each ``fn(param) -> param`` is registered via :func:`parametrize_params`
            on the module during tracing, and deregistered after.  Multiple
            functions are stacked in order (first fn applied first).
        _stateless_func_wrapper: Optional context manager factory to wrap the
            traced function execution (e.g., gradients_tagging for the v1/GSPMD
            DataParallel path).  Receives the ``params`` dict and returns a
            context manager.  If None, no additional wrapping is applied.
        **kwargs: Keyword arguments to the train_step function.
    """
    # 1. Extract nn.Module and Optimizer from args and kwargs
    # FIXME(@mrshenli): support multiple nn.Module instances
    # FIXME(@mrshenli): support multiple Optiimzer instances
    # FIXME(@mrshenli): need to broadcast model to sync parameters
    mod: Optional[nn.Module] = None
    opt: Optional[torch.optim.Optimizer] = None
    for arg in pytree.arg_tree_leaves(*args, **kwargs):
        if isinstance(arg, nn.Module):
            assert mod is None, "Only support single nn.Module for now"
            mod = arg
        if isinstance(arg, torch.optim.Optimizer):
            assert opt is None, "Only support single Optimizer for now"
            opt = arg

    assert mod is not None, "Couldn't find nn.Module instances from the arguments."
    _mod: nn.Module = mod
    _opt: Optional[torch.optim.Optimizer] = opt

    # 2. Trace stateless version of the train_step
    params = dict(mod.named_parameters(remove_duplicate=False))
    buffers = dict(mod.named_buffers(remove_duplicate=False))

    named_states: Dict[str, Any]
    named_states = {}
    if opt is not None:
        for n, p in params.items():
            if p in opt.state:
                named_states[n] = opt.state[p]  # type: ignore[index]

    # Lift states and parameters as function arguments so that make_fx
    # can trace operations applied to them.
    def stateless_func(
        func: Callable[..., Any],
        params: Dict[str, nn.Parameter],
        buffers: Dict[str, torch.Tensor],
        named_states: Dict[str, Any],
        args: Any,
        kwargs: Any,
    ) -> Any:
        with (
            stateless._reparametrize_module(_mod, {**params, **buffers}),
            _rematerialize_optimizer(_opt, named_states, params)
            if _opt
            else nullcontext(),
        ):
            with ExitStack() as stack:
                for fn in _parametrize_fns or []:
                    stack.enter_context(
                        parametrize_params(
                            _mod, fn, exclude_param_fqns=_exclude_param_fqns
                        )
                    )
                if _stateless_func_wrapper:
                    stack.enter_context(_stateless_func_wrapper(params))
                ret = func(*args, **kwargs)

            # Return updated state in the same pytree structure used to build
            # flat_state so DeviceStateManager can copy outputs back by index.
            return ret, params, buffers, named_states

    with fake_mode, _enable_compile(), torch.autograd.detect_anomaly(check_nan=False):
        # FIXME(@mrshenli): functionalization does not work for our use
        # case yet. Use explicit decompositions for foreach ops.
        # Remove this when the following issue is addressed.
        # Issue: https://github.com/pytorch/pytorch/issues/97852
        gm = make_fx(
            partial(stateless_func, func),
            tracing_mode="fake",
            decomposition_table=SPMD_DECOMP_TABLE,  # type: ignore[arg-type]
            _allow_non_fake_inputs=False,
        )(params, buffers, named_states, args, kwargs)

    # Re-register tensor ``get_attr`` constants as ``_AttrKind.CONSTANT``.
    #
    # ``make_fx`` lifts inline tensor literals into module attributes referenced
    # by ``get_attr`` nodes. ``fx.GraphModule.__init__`` silently registers any
    # tensor attribute that isn't an ``nn.Parameter`` as a *buffer*. Buffers are
    # rewritten on subsequent ``gm.recompile()`` calls (the codegen treats them
    # like state) and aren't propagated through later passes that move sub-gms
    # around. By deregistering each tensor get_attr and reassigning it via
    # ``_assign_attr(..., _AttrKind.CONSTANT)`` we (a) preserve the value across
    # ``recompile`` and (b) give downstream passes (e.g. data-parallel strategy
    # building) a stable tensor attribute they can introspect via the qualified
    # path.
    #
    # In production today, ``make_fx`` emits flat top-level attribute names
    # (e.g. ``_tensor_constant0``), so qualified paths don't actually appear
    # here. The qualname-aware helpers from ``torch.fx.graph_module`` are used
    # for robustness against any FUTURE qualified targets — for example, custom
    # tracers, downstream passes that move attrs into submodules, or upstream
    # changes to ``make_fx``. Every ``get_attr`` target produced by ``make_fx``
    # references an attribute that exists on ``gm``; if ``_get_attr`` ever
    # raises here it indicates an upstream invariant break, so let the
    # exception propagate rather than silently skip the node.
    constants: Dict[str, torch.Tensor] = {}
    for node in gm.graph.nodes:
        if node.op != "get_attr":
            continue
        fqn = node.target
        if not isinstance(fqn, str):
            continue
        value = _get_attr(gm, fqn)
        if not isinstance(value, torch.Tensor):
            continue
        # Multiple ``get_attr`` nodes can share the same target FQN; only
        # re-register once per unique attribute name.
        if fqn in constants:
            continue
        _del_attr(gm, fqn)
        _assign_attr(value, gm, fqn, attr_kind=_AttrKind.CONSTANT)
        constants[fqn] = value

    return _LocalCompiledResult(gm, mod, opt, params, buffers, named_states, constants)


def _make_batch_transform(
    input_batch_dim: int,
    ratio: float,
    fake_mode: FakeTensorMode,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create an input transform that expands or shrinks mini-batch tensors.

    For ratio > 1 (e.g., v1 path: local → global batch), ``repeat`` expands.
    For ratio < 1 (e.g., v2 XLA path: multi-device local → per-device batch),
    ``narrow`` shrinks.  For ratio == 1 the tensor is unchanged.
    """

    def _get_full_batch_arg(arg: torch.Tensor) -> torch.Tensor:
        fake_arg = fake_mode.from_tensor(arg)
        curr_batch = fake_arg.shape[input_batch_dim]
        target_batch = max(1, int(curr_batch * ratio))
        if target_batch > curr_batch:
            repeat_dims = [1] * arg.ndim
            repeat_dims[input_batch_dim] = target_batch // curr_batch
            return fake_arg.repeat(repeat_dims)
        elif target_batch < curr_batch:
            return fake_arg.narrow(input_batch_dim, 0, target_batch)
        return fake_arg

    return _get_full_batch_arg


def _tag_batch_inputs(
    gm: fx.GraphModule, num_model_tensors: int, input_batch_dim: int
) -> None:
    tensor_idx = 0
    for node in gm.graph.nodes:
        if node.op != "placeholder":
            continue
        val = node.meta.get("val", None)
        if isinstance(val, torch.Tensor):
            is_batch = tensor_idx >= num_model_tensors
            node.meta["is_batch_input"] = is_batch
            node.meta["is_weight"] = not is_batch
            if is_batch:
                node.meta["batch_dim"] = input_batch_dim
            tensor_idx += 1
        else:
            node.meta["is_batch_input"] = False
            node.meta["is_weight"] = False


def _finalize_graph(
    gm: fx.GraphModule,
    mod: nn.Module,
    opt: Optional[torch.optim.Optimizer],
    params_and_buffers: Dict[str, Any],
    named_states: Dict[str, Any],
    module_override: Optional[List[Override]],
    input_batch_dim: int,
    *,
    dedup: bool = True,
) -> _CompiledResult:
    """Flatten state, dedup collectives, apply overrides, and tag batch inputs."""
    flat_state = pytree.tree_leaves([params_and_buffers, named_states])
    gm = _to_caller_flattened_graph_module(gm)
    if dedup:
        gm = _dedup_collectives(gm)
    if module_override:
        for override in module_override:
            gm = override.transform(gm, flat_state)
    _tag_batch_inputs(gm, len(flat_state), input_batch_dim)
    # Propagate batch-input meta through ops so intermediate FX nodes that
    # become subgraph boundaries (after segment_and_lower) inherit
    # is_batch_input / batch_dim. Without this, downstream subgraph
    # placeholders default to replicated and JAX sees per-process shape
    propagate_batch_input_meta(gm)
    return _CompiledResult(gm, mod, opt, flat_state, DeviceStateManager())


@dataclass
class _RegionDiscovery:
    """Result of `_discover_for_region_transforms`.

    Holds the called-module map and the autograd-function leaves discovered
    by running the train step under FakeTensorMode hooks.
    """

    called_modules: OrderedDict[str, Tuple[nn.Module, Tuple[Any, ...], Dict[str, Any]]]
    autograd_function_leaves: set[type]  # pyre-fixme[24]


def _discover_for_region_transforms(
    transforms: List[RegionTransform],
    func: Callable[..., Any],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    *,
    mod: Optional[nn.Module] = None,
) -> Optional[_RegionDiscovery]:
    """Run `_discover_called_modules` and `_discover_autograd_functions`.

    Returns ``None`` (no-op) when ``transforms`` is empty.  When ``mod`` is
    not provided the first :class:`nn.Module` in ``args``/``kwargs`` is used.
    """
    if not transforms:
        return None
    if mod is None:
        for arg in pytree.arg_tree_leaves(*args, **kwargs):
            if isinstance(arg, nn.Module):
                mod = arg
                break
        assert mod is not None, (
            "region_transforms requires an nn.Module in the arguments"
        )
    return _RegionDiscovery(
        called_modules=_discover_called_modules(
            mod,
            FakeTensorMode(allow_non_fake_inputs=True),
            func,
            *args,
            **kwargs,
        ),
        autograd_function_leaves=_discover_autograd_functions(
            mod,
            FakeTensorMode(allow_non_fake_inputs=True),
            func,
            *args,
            **kwargs,
        ),
    )


def _enter_region_transform_contexts(
    region_stack: ExitStack,  # pyre-ignore
    transforms: List[RegionTransform],
    discovery: Optional[_RegionDiscovery],
    fake_mode: FakeTensorMode,
    *,
    parametrize_cm_factory: Optional[Callable[[nn.Module], Callable[[], Any]]] = None,
) -> None:
    """Enter `_apply_region_transforms` once per forward root on `region_stack`.

    No-op when ``discovery`` is None or ``transforms`` is empty.
    ``parametrize_cm_factory(root_mod)`` returns a context-manager factory
    used during the per-root trace inside ``_apply_region_transforms``;
    pass ``None`` when no parametrization is needed (e.g. GSPMD path).
    """
    if discovery is None or not transforms:
        return
    for _fqn, root_mod, root_fake_args, root_fake_kwargs in _find_forward_roots(
        discovery.called_modules
    ):
        cm_factories = (
            [parametrize_cm_factory(root_mod)]
            if parametrize_cm_factory is not None
            else []
        )
        region_stack.enter_context(
            _apply_region_transforms(
                root_mod,
                root_fake_args,
                discovery.called_modules,
                transforms,
                fake_mode,
                _parametrize_cm_factories=cm_factories,
                _autograd_function_leaves=discovery.autograd_function_leaves,
                fake_kwargs=root_fake_kwargs,
            )
        )


def _compile_and_transform(
    func: Callable[..., Any],
    module_override: Optional[List[Override]],
    parallel_mode: ParallelMode,
    gm_transformation: Optional[Callable[[fx.GraphModule], fx.GraphModule]],
    *args: Any,
    region_transforms: Optional[List[RegionTransform]] = None,
    **kwargs: Any,
) -> _CompiledResult:
    if region_transforms:
        for transform in region_transforms:
            bind = getattr(transform, "bind_signature", None)
            if bind is not None:
                bind(func)

    if isinstance(parallel_mode, GSPMDParallel):
        result = _compile_gspmd(
            func,
            module_override,
            parallel_mode,
            *args,
            region_transforms=region_transforms,
            **kwargs,
        )
    elif (
        isinstance(parallel_mode, (DataParallel, HybridParallel))
        and parallel_mode._use_parametrization
    ):
        result = _compile_data_parallel_v2(
            func,
            module_override,
            parallel_mode,
            *args,
            region_transforms=region_transforms,
            **kwargs,
        )
    else:
        result = _compile_data_parallel(
            func, module_override, cast(DataParallel, parallel_mode), *args, **kwargs
        )
    if region_transforms:
        for transform in region_transforms:
            transform.finalize_compiled_graph(result.gm)

    if gm_transformation is not None:
        result.gm = gm_transformation(result.gm)
        result.state_manager = getattr(
            gm_transformation, "device_state_manager", DeviceStateManager()
        )
    return result


def _compile_data_parallel(
    func: Callable[..., Any],
    module_override: Optional[List[Override]],
    parallel_mode: DataParallel,
    *args: Any,
    **kwargs: Any,
) -> _CompiledResult:
    """Compile a train step with DTensor-based data parallelism (DDP/FSDP).

    Steps:
      1-3. Trace with global batch shapes via _local_compile
      4. Shard params/states with DTensor
      5. Expand graph with DTensor collectives (partition)
      6. Flatten args for the caller
      7. Deduplicate collective ops
      8. Apply module overrides
    """
    fake_mode: FakeTensorMode = FakeTensorMode()
    input_transform = _make_batch_transform(
        parallel_mode.input_batch_dim,
        ratio=parallel_mode.num_processes,
        fake_mode=fake_mode,
    )
    args = pytree.tree_map_only(torch.Tensor, input_transform, args)
    kwargs = pytree.tree_map_only(torch.Tensor, input_transform, kwargs)
    if module_override:
        for arg in pytree.arg_tree_leaves(*args, **kwargs):
            if isinstance(arg, nn.Module):
                _apply_module_overrides(arg, module_override)
                break
    local_result = _local_compile(
        func,
        *args,
        fake_mode=fake_mode,
        _stateless_func_wrapper=gradients_tagging,
        **kwargs,
    )
    gm = local_result.gm
    mod = local_result.mod
    opt = local_result.opt
    params = local_result.params
    buffers = local_result.buffers
    params_and_buffers: Dict[str, Union[torch.Tensor, nn.Parameter]] = {
        **params,
        **buffers,
    }
    # 4. Shard parameters and optimizer states before partitioning
    parallel_mode.shard_params(mod, params_and_buffers, opt)
    # Re-extract named_states from opt.state (now has correct sharded
    # states keyed by the new parameter objects).
    named_states: Dict[str, Any] = {}
    if opt is not None:
        for n, p in mod.named_parameters(remove_duplicate=False):
            if p in opt.state:
                named_states[n] = opt.state[p]  # type: ignore[index]
    # 5. parallel mode to expand a single device graph to a distributed graph
    # Re-split params/buffers from the (possibly re-keyed) module after
    # shard_params, since shard_params may have replaced parameter objects.
    sharded_params: Dict[str, nn.Parameter] = {}
    sharded_buffers: Dict[str, torch.Tensor] = {}
    for fqn, val in params_and_buffers.items():
        if isinstance(val, nn.Parameter):
            sharded_params[fqn] = val
        else:
            sharded_buffers[fqn] = val
    gm = parallel_mode.partition(
        gm,
        params=sharded_params,
        buffers=sharded_buffers,
        named_states=named_states,
    )
    return _finalize_graph(
        gm,
        mod,
        opt,
        params_and_buffers,
        named_states,
        module_override,
        parallel_mode.input_batch_dim,
    )


def _compile_data_parallel_v2(
    func: Callable[..., Any],
    module_override: Optional[List[Override]],
    parallel_mode: Union[DataParallel, HybridParallel],
    *args: Any,
    region_transforms: Optional[List[RegionTransform]] = None,
    **kwargs: Any,
) -> _CompiledResult:
    """Compile with autograd-based collective insertion (no DTensor partition).

    Handles both ``DataParallel`` (dense-only) and ``HybridParallel``
    (dense + embedding sharding) v2 paths.  For ``HybridParallel``,
    embedding parameters are excluded from collective insertion and
    instead rewritten post-trace into decomposed AllToAll ops.

    Caller-supplied ``region_transforms`` (e.g. activation checkpointing via
    :class:`RecomputeTransform`) are appended to the path-internal transforms
    (``RegatherTransform`` for FULLY_SHARD).  All transforms share a single
    discovery + ``_apply_region_transforms`` pass.
    """
    # Extract mesh and DP style from the parallel mode.
    if isinstance(parallel_mode, HybridParallel):
        mesh = parallel_mode.mesh
        dp_style = parallel_mode.dense_style
    else:
        mesh = parallel_mode.device_mesh
        dp_style = parallel_mode.parallel_style
    world_size: int = parallel_mode.num_ranks

    # 1. Pre-extract mod/opt (need originals for pre-trace sharding).
    mod: Optional[nn.Module] = None
    opt: Optional[torch.optim.Optimizer] = None
    for arg in pytree.arg_tree_leaves(*args, **kwargs):
        if isinstance(arg, nn.Module):
            mod = arg
        if isinstance(arg, torch.optim.Optimizer):
            opt = arg
    assert mod is not None
    _mod: nn.Module = mod

    # 2. Resolve embedding param FQNs from the plan (HybridParallel only).
    embedding_param_fqns: Optional[set[str]] = None
    if isinstance(parallel_mode, HybridParallel):
        fqns = parallel_mode.embedding_plan.param_fqns
        embedding_param_fqns = fqns if fqns else None

    # 3. Build transforms list (RegatherTransform for FULLY_SHARD + caller
    #    region_transforms).  Embedding params are excluded from the regather
    #    so that step 9's row-wise rewrite owns them.
    transforms: list[RegionTransform] = []
    if dp_style == DataParallelStyle.FULLY_SHARD:
        transforms.append(RegatherTransform(skip_param_fqns=embedding_param_fqns))
    if region_transforms:
        transforms.extend(region_transforms)

    # Apply module overrides so that non-traceable submodules are
    # replaced with dummy placeholders for discovery, region-transform
    # inner trace, and the outer _local_compile trace.
    if module_override:
        _apply_module_overrides(_mod, module_override)

    # 4. Extract params_and_buffers AFTER overrides so the dict matches the
    #    (post-override) module that _local_compile will trace.  Critical
    #    for correctness: shard_params mutates _mod by FQN, and embedding
    #    indices into this dict must map to placeholder positions in the
    #    traced graph (which uses the post-override params ordering).
    params_and_buffers: Dict[str, Any] = {
        **dict(_mod.named_parameters(remove_duplicate=False)),
        **dict(_mod.named_buffers(remove_duplicate=False)),
    }
    embedding_param_indices: Optional[set[int]] = None
    if isinstance(parallel_mode, HybridParallel):
        embedding_param_indices = parallel_mode._compute_embedding_param_indices(
            params_and_buffers
        )

    # 5. Set up external fake mode for tracing.
    #    HybridParallel only: a ShapeEnv is attached so custom ops with
    #    data-dependent output shapes (``recir_torchrec::kjt_split_per_feature``,
    #    ``recir_preproc::mot_preprocessor_forward``) can mint fresh SymInts via
    #    ``torch.library.get_ctx().new_dynamic_size()`` from their fake kernels.
    #    Without a ShapeEnv those calls raise ``DynamicOutputShapeException``.
    fake_mode: FakeTensorMode = FakeTensorMode(
        allow_non_fake_inputs=True,
        shape_env=ShapeEnv() if isinstance(parallel_mode, HybridParallel) else None,
    )

    # Apply the per-process → per-device batch transform up front, so
    # discovery, region-transform inner trace, and the outer trace all
    # see the same (per-device) input shape.  Otherwise the region's
    # ``gm_fwd`` would bake per-process shapes that conflict with the
    # outer trace's narrowed inputs.
    input_transform = _make_batch_transform(
        parallel_mode.input_batch_dim,
        ratio=1 / (world_size // parallel_mode.num_processes),
        fake_mode=fake_mode,
    )
    args = pytree.tree_map_only(torch.Tensor, input_transform, args)
    kwargs = pytree.tree_map_only(torch.Tensor, input_transform, kwargs)
    discovery = _discover_for_region_transforms(
        transforms, func, args, kwargs, mod=_mod
    )

    # 6. Shard params on the module and optimizer states.
    #    shard_params mutates both the dict and the module in-place.
    #    It also updates opt.state so that parameter→state mappings remain valid.
    parallel_mode.shard_params(_mod, params_and_buffers, opt)

    # 7. Apply region transforms AFTER sharding, with parametrization.
    #    The traced forward includes allgather ops from collectives_insertion
    #    so each per-parameter region encompasses allgather + chain + compute.
    #    Embedding params (if any) are excluded from collectives — they use
    #    AllToAll (inserted in step 9), not allgather.
    collectives_fn = collectives_insertion(dp_style, mesh)
    parametrize_kwargs: Dict[str, Any] = {}
    if embedding_param_fqns:
        parametrize_kwargs["exclude_param_fqns"] = embedding_param_fqns
    with ExitStack() as region_stack:
        _enter_region_transform_contexts(
            region_stack,
            transforms,
            discovery,
            fake_mode,
            parametrize_cm_factory=lambda root_mod: partial(
                parametrize_params,
                root_mod,
                collectives_fn,
                **parametrize_kwargs,
            ),
        )

        # 8. Set up parametrize fns and trace via _local_compile.
        #    Region transform context managers scope the rewritten forwards
        #    to this block — originals are restored when the ExitStack exits.
        #    When region transforms are active, gm_fwd already contains the
        #    allgather ops (baked in during the parametrized trace), so we
        #    must NOT also pass collectives_fn to _local_compile — that would
        #    double-allgather and corrupt gradients.
        local_result = _local_compile(
            func,
            *args,
            fake_mode=fake_mode,
            _parametrize_fns=[] if transforms else [collectives_fn],
            _exclude_param_fqns=embedding_param_fqns if not transforms else None,
            **kwargs,
        )

    # 9. Embedding rewrite on the compiled graph (HybridParallel only).
    #    Pattern-match aten.embedding nodes on sharded embedding params
    #    and replace with decomposed AllToAll ops via secondary symbolic trace.
    #
    #    The rewrite needs concrete tensor shapes to build example inputs
    #    for the symbolic sub-trace.  Reading shapes from ``placeholder.meta["val"]``
    #    does not work here: the outer ``fake_mode`` carries a ``ShapeEnv`` which
    #    symbolicizes every placeholder shape.
    if embedding_param_indices:
        flat_params_buffers = list(local_result.params.values()) + list(
            local_result.buffers.values()
        )
        embedding_param_tensors: Dict[int, torch.Tensor] = {
            i: flat_params_buffers[i] for i in embedding_param_indices
        }
        rewrite_sparse_embeddings(
            local_result.gm,
            mesh,
            embedding_param_tensors,
        )

    # 10. Finalize.
    return _finalize_graph(
        local_result.gm,
        local_result.mod,
        local_result.opt,
        {**local_result.params, **local_result.buffers},
        local_result.named_states,
        module_override,
        parallel_mode.input_batch_dim,
    )


def _compile_gspmd(
    func: Callable[..., Any],
    module_override: Optional[List[Override]],
    parallel_mode: GSPMDParallel,
    *args: Any,
    region_transforms: Optional[List[RegionTransform]] = None,
    **kwargs: Any,
) -> _CompiledResult:
    """Compile a train step with GSPMD auto-parallelism.

    Unlike the DataParallel path, this skips DTensor partitioning entirely.
    The graph is traced with global batch shapes (expanded from the local
    mini-batch by world_size) and metadata is attached for the GSPMD
    lowering step.

    When ``region_transforms`` is provided, region-level rewrites (e.g.
    activation checkpointing via :class:`RecomputeTransform`) are applied
    to the user's forward methods during tracing.  They are scoped to the
    trace via an ``ExitStack`` and reverted before this function returns.
    """
    fake_mode: FakeTensorMode = FakeTensorMode()
    transforms = region_transforms or []
    # Apply module overrides so that non-traceable submodules (e.g. MoE)
    # are replaced with dummy placeholders for discovery, region-transform
    # inner trace, and the outer _local_compile trace.
    if module_override:
        for arg in pytree.arg_tree_leaves(*args, **kwargs):
            if isinstance(arg, nn.Module):
                _apply_module_overrides(arg, module_override)
                break
    # Apply the per-process → global batch transform up front, so discovery,
    # region-transform inner trace, and the outer ``_local_compile`` trace
    # all see the same (global) input shape.  Otherwise the region's
    # ``gm_fwd`` would bake per-process shapes (e.g. ``aten.view`` target
    # lists) that conflict with the outer trace's expanded inputs.
    input_transform = _make_batch_transform(
        parallel_mode.input_batch_dim,
        ratio=parallel_mode.num_processes,
        fake_mode=fake_mode,
    )
    args = pytree.tree_map_only(torch.Tensor, input_transform, args)
    kwargs = pytree.tree_map_only(torch.Tensor, input_transform, kwargs)
    discovery = _discover_for_region_transforms(transforms, func, args, kwargs)
    with ExitStack() as region_stack:
        _enter_region_transform_contexts(region_stack, transforms, discovery, fake_mode)
        local_result = _local_compile(
            func,
            *args,
            fake_mode=fake_mode,
            _stateless_func_wrapper=gradients_tagging,
            **kwargs,
        )
    gm = local_result.gm
    # GSPMD auto-inserts all-reduce sum for gradients
    # all-reduce average. Use the tag_grad markers to find gradient tensors
    # and scale them by 1/num_ranks to match DDP's averaging semantics.
    # For mean loss (grad_reduction="sum"), skip the scaling since the
    # traced backward already normalises by the global numel — the XLA
    # all-reduce sum alone produces the correct global-mean gradient.
    # Then remove the tag_grad nodes since they've served their purpose.
    num_ranks = parallel_mode.num_ranks
    if num_ranks > 1 and parallel_mode.grad_reduction == "avg":
        for node in list(gm.graph.nodes):
            if (
                node.op == "call_function"
                and node.target is torch.ops._spmd.tag_grad.default
            ):
                with gm.graph.inserting_after(node):
                    div_node = gm.graph.call_function(
                        torch.ops.aten.div.Scalar, args=(node, num_ranks)
                    )
                    div_node.meta = node.meta.copy() if node.meta else {}
                    node.replace_all_uses_with(div_node)
                    div_node.args = (node, num_ranks)
    for node in list(gm.graph.nodes):
        if (
            node.op == "call_function"
            and node.target is torch.ops._spmd.tag_grad.default
        ):
            node.replace_all_uses_with(node.args[0])
            gm.graph.erase_node(node)
    gm.recompile()

    params_and_buffers = {**local_result.params, **local_result.buffers}
    # FSDP: shard model params AFTER tracing but BEFORE _finalize_graph.
    # The graph retains global shapes; runtime inputs are local shards
    # (_to_sharded_jax bridges the gap). Sharding before _finalize_graph
    # means flat_state is built from already-sharded tensors.
    sharded_keys = parallel_mode.shard_params(
        local_result.mod, params_and_buffers, local_result.opt
    )
    result = _finalize_graph(
        gm,
        local_result.mod,
        local_result.opt,
        params_and_buffers,
        local_result.named_states,
        module_override=module_override,
        input_batch_dim=parallel_mode.input_batch_dim,
        dedup=False,
    )
    # Set is_fsdp_sharded on placeholder nodes so _jit_gspmd knows
    # which inputs are local shards.
    if sharded_keys:
        is_sharded: list[bool] = [key in sharded_keys for key in params_and_buffers]
        for fqn, state in local_result.named_states.items():
            is_sharded.extend([fqn in sharded_keys] * len(state))
        tensor_idx = 0
        for node in result.gm.graph.nodes:
            if node.op != "placeholder":
                continue
            val = node.meta.get("val", None)
            if isinstance(val, torch.Tensor):
                node.meta["is_fsdp_sharded"] = (
                    is_sharded[tensor_idx] if tensor_idx < len(is_sharded) else False
                )
                tensor_idx += 1
        logger.info(
            "FSDP: %d sharded keys, %d sharded placeholders, flat_state=%d entries",
            len(sharded_keys),
            sum(
                1
                for n in result.gm.graph.nodes
                if n.op == "placeholder" and n.meta.get("is_fsdp_sharded", False)
            ),
            len(result.flat_state),
        )
    return result


# Note that the Python convention of __dict__ requires the key to be str.
# TODO: ensure the key is unique.
COMPILED_OBJECT_KEY = "_compiled_obj"


def compile(
    module_override: Optional[List[Override]] = None,
    gm_transformation: Optional[Callable[[fx.GraphModule], fx.GraphModule]] = None,
    parallel_mode: Optional[ParallelMode] = None,
    region_transforms: Optional[List[RegionTransform]] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    r"""Compile and optimize a callable, which can be a train step within a training loop.

    This method will extract :class:`nn.Module` and :class:`torch.optim.Optimizer`
    instances from the input arguments and trace operations applied to their
    parameters and states.

    Args:
        module_override (Optional[List[Override]]): a list of Override instances
            that will be applied to the module in order. The :class:`Override`
            objects provide :class:`nn.Module` replacements during tracing and a
            graph transformation function after tracing. (Default: ``None``)
        gm_transformation (Optional[Callable[fx.GraphModule, fx.GraphModule]]):
            a callback that will be called after the original callable is
            compiled and distributed (usually after the first iteration) to
            transform the compiled GraphModule into a new optimized one.
        parallel_mode (Optional[ParallelMode]): a :class:`ParallelMode` object
            that specifies how to parallelize the callable. Each ParallelMode
            would have its own strategy to partition the model and the captured
            graph (Default: ``None``)
        region_transforms (Optional[List[RegionTransform]]): a list of
            :class:`RegionTransform` instances applied during tracing to rewrite
            sub-regions of the forward graph (e.g. activation checkpointing via
            :class:`RecomputeTransform`). (Default: ``None``)

    """

    def inner(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_train_step = kwargs.pop("last_train_step", False) if kwargs else False
            # Put the COMPILED_OBJECT_KEY in ``wrapper`` instead of ``func`` as
            # ``wrapper`` is the one that users will get.
            compiled_obj = wrapper.__dict__.get(COMPILED_OBJECT_KEY, None)
            if compiled_obj is None:
                mode: ParallelMode = (
                    DataParallel() if parallel_mode is None else parallel_mode
                )
                compiled_obj = _compile_and_transform(
                    func,
                    module_override,
                    mode,
                    gm_transformation,
                    *args,
                    region_transforms=region_transforms,
                    **kwargs,
                )
                wrapper.__dict__[COMPILED_OBJECT_KEY] = compiled_obj

            user_inps = pytree.arg_tree_leaves(*args, **kwargs)
            flat_inps = compiled_obj.state_manager.prepare_inputs(
                compiled_obj.gm,
                compiled_obj.flat_state,
                user_inps,
                last_train_step,
            )
            with torch.no_grad():
                # N.B.: we don't need autograd as backward has already been
                # captured in the graph.
                if not last_train_step:
                    output_all = compiled_obj.gm(*flat_inps)
                else:
                    # This is the last train step. Call IterGraphModule.forward()
                    # with the `last_iter` argument and catch the exception in
                    # case the compiled_obj is not wrapped with IterGraphModule.
                    try:
                        output_all = compiled_obj.gm(
                            *flat_inps, last_iter=last_train_step
                        )
                    except TypeError as e:
                        if "last_iter" not in str(e):
                            raise e
                        output_all = compiled_obj.gm(*flat_inps)

                output = output_all[0]
                updated_flat = pytree.tree_leaves(output_all[1:])
                output = compiled_obj.state_manager.process_outputs(
                    compiled_obj.flat_state,
                    output,
                    updated_flat,
                    last_train_step,
                )

                return output

        return wrapper

    return inner


def sync_device_state_to_torch(compiled_fn: Callable[..., Any]) -> None:
    """Synchronize device-resident state back to PyTorch tensors.
    Call before checkpointing or inspecting model parameters.
    """
    compiled_obj = compiled_fn.__dict__.get(COMPILED_OBJECT_KEY, None)
    if compiled_obj is None:
        return
    if compiled_obj.state_manager.device_state is None:
        return
    compiled_obj.state_manager.sync_to_torch(compiled_obj.flat_state)
