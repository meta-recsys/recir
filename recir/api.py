# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from abc import ABC, abstractmethod
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
from recir.decompositions import DEDUP_TARGETS, SPMD_DECOMP_TABLE
from recir.distributed.data_parallel import (
    collectives_insertion,
    gradients_tagging,
    parametrize_params,
)
from recir.distributed.parallel_mode import DataParallel, GSPMDParallel, ParallelMode
from recir.lowering.device_state import DeviceStateManager
from torch import fx
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo, CodeGen
from torch.nn.utils import stateless
from torch.nn.utils._named_member_accessor import NamedMemberAccessor


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
    try:
        yield
    finally:
        torch._utils.is_compiling.__code__ = orig_is_compiling_code  # pyre-ignore[16]


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


def _local_compile(
    func: Callable[..., Any],
    module_override: Optional[List[Override]],
    *args: Any,
    fake_mode: FakeTensorMode,
    _parametrize_fns: Optional[List[Callable[[torch.Tensor], torch.Tensor]]] = None,
    _stateless_func_wrapper: Optional[Callable[..., Any]] = None,
    _input_transform: Optional[Callable[[Any], Any]] = None,
    **kwargs: Any,
) -> _LocalCompiledResult:
    """Trace a train_step into a local FX graph.

    This performs the local tracing portion of _compile:
    1. Extract nn.Module and Optimizer from args
    2. Apply module overrides
    3. Trace with make_fx using FakeTensorMode

    Does NOT require dist.init_process_group by default. Does NOT perform
    DataParallel partitioning, collective dedup, or state flattening.

    Args:
        func: The train_step function to trace.
        module_override: Optional list of Override instances for submodule replacement.
        *args: Arguments to the train_step function (must include nn.Module and Optimizer).
        fake_mode: The FakeTensorMode to use for tracing. Each compile path creates
            its own mode and passes it here. This mode wraps the make_fx call
            and is used to convert inputs when no _input_transform is provided.
        _parametrize_fns: Optional list of per-parameter transform functions.
            Each ``fn(param) -> param`` is registered via :func:`parametrize_params`
            on the module during tracing, and deregistered after.  Multiple
            functions are stacked in order (first fn applied first).
        _stateless_func_wrapper: Optional context manager factory to wrap the
            traced function execution (e.g., gradients_tagging for the v1/GSPMD
            DataParallel path).  Receives the ``params`` dict and returns a
            context manager.  If None, no additional wrapping is applied.
        _input_transform: Optional function to transform input tensors before tracing
            (e.g., FakeTensor conversion with batch expansion for DataParallel).
            If None, inputs are converted to FakeTensors using fake_mode.
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

    # 2. Override target submodules (e.g., MoE) with dummy replacements
    if module_override:
        accessor: NamedMemberAccessor = NamedMemberAccessor(mod)

        def swap(fqn_prefix: str, module: torch.nn.Module) -> None:
            for override in module_override:  # type: ignore[union-attr]
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

    # 3. Trace stateless version of the train_step
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
                    stack.enter_context(parametrize_params(_mod, fn))
                if _stateless_func_wrapper:
                    stack.enter_context(_stateless_func_wrapper(params))
                ret = func(*args, **kwargs)

            # make sure updated parameters are returned
            return ret, list(_mod.parameters()), list(named_states.values())

    # Convert inputs (default: simple FakeTensor conversion).
    if _input_transform is not None:
        args = pytree.tree_map_only(torch.Tensor, _input_transform, args)
        kwargs = pytree.tree_map_only(torch.Tensor, _input_transform, kwargs)
    else:
        args = pytree.tree_map_only(
            torch.Tensor, lambda t: fake_mode.from_tensor(t), args
        )
        kwargs = pytree.tree_map_only(
            torch.Tensor, lambda t: fake_mode.from_tensor(t), kwargs
        )

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

    return _LocalCompiledResult(gm, mod, opt, params, buffers, named_states)


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
            if is_batch:
                node.meta["batch_dim"] = input_batch_dim
            tensor_idx += 1
        else:
            node.meta["is_batch_input"] = False


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
    return _CompiledResult(gm, mod, opt, flat_state, DeviceStateManager())


def _compile_and_transform(
    func: Callable[..., Any],
    module_override: Optional[List[Override]],
    parallel_mode: ParallelMode,
    gm_transformation: Optional[Callable[[fx.GraphModule], fx.GraphModule]],
    *args: Any,
    **kwargs: Any,
) -> _CompiledResult:
    if isinstance(parallel_mode, GSPMDParallel):
        result = _compile_gspmd(func, module_override, parallel_mode, *args, **kwargs)
    elif isinstance(parallel_mode, DataParallel) and parallel_mode._use_parametrization:
        result = _compile_data_parallel_v2(
            func, module_override, parallel_mode, *args, **kwargs
        )
    else:
        result = _compile_data_parallel(
            func, module_override, cast(DataParallel, parallel_mode), *args, **kwargs
        )
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
    local_result = _local_compile(
        func,
        module_override,
        *args,
        fake_mode=fake_mode,
        _stateless_func_wrapper=gradients_tagging,
        _input_transform=_make_batch_transform(
            parallel_mode.input_batch_dim,
            ratio=parallel_mode.num_processes,
            fake_mode=fake_mode,
        ),
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
    gm = parallel_mode.partition(
        gm,
        mod,
        opt,
        params_and_buffers,
        named_states,
        args,
        kwargs,
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
    parallel_mode: DataParallel,
    *args: Any,
    **kwargs: Any,
) -> _CompiledResult:
    """Compile with autograd-based collective insertion (no DTensor partition).

    Both replicate and fully_shard use external FakeTensorMode + DTensor
    redistribute for collective insertion during tracing.
    """
    mesh = parallel_mode.device_mesh
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

    # 2. Shard params on the module and optimizer states (FSDP only).
    #    Extract params_and_buffers first so shard_params can mutate both
    #    the dict and the module in-place.  shard_params also updates
    #    opt.state in-place so that parameter→state mappings remain valid.
    params_and_buffers: Dict[str, Any] = {
        **dict(_mod.named_parameters(remove_duplicate=False)),
        **dict(_mod.named_buffers(remove_duplicate=False)),
    }
    parallel_mode.shard_params(_mod, params_and_buffers, opt)

    # 3. Set up external fake mode and input transform.
    fake_mode: FakeTensorMode = FakeTensorMode(allow_non_fake_inputs=True)

    # 4. Set up parametrize fns and trace via _local_compile.
    local_result = _local_compile(
        func,
        module_override,
        *args,
        fake_mode=fake_mode,
        _parametrize_fns=[
            collectives_insertion(parallel_mode.parallel_style, mesh),
        ],
        _input_transform=_make_batch_transform(
            parallel_mode.input_batch_dim,
            ratio=1 / (world_size // parallel_mode.num_processes),
            fake_mode=fake_mode,
        ),
        **kwargs,
    )

    # 5. Finalize.
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
    **kwargs: Any,
) -> _CompiledResult:
    """Compile a train step with GSPMD auto-parallelism.

    Unlike the DataParallel path, this skips DTensor partitioning entirely.
    The graph is traced with global batch shapes (expanded from the local
    mini-batch by world_size) and metadata is attached for the GSPMD
    lowering step.
    """
    fake_mode: FakeTensorMode = FakeTensorMode()
    local_result = _local_compile(
        func,
        module_override,
        *args,
        fake_mode=fake_mode,
        _stateless_func_wrapper=gradients_tagging,
        _input_transform=_make_batch_transform(
            parallel_mode.input_batch_dim,
            ratio=parallel_mode.num_processes,
            fake_mode=fake_mode,
        ),
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

    return _finalize_graph(
        gm,
        local_result.mod,
        local_result.opt,
        {**local_result.params, **local_result.buffers},
        local_result.named_states,
        module_override=None,
        input_batch_dim=parallel_mode.input_batch_dim,
        dedup=False,
    )


# Note that the Python convention of __dict__ requires the key to be str.
# TODO: ensure the key is unique.
COMPILED_OBJECT_KEY = "_compiled_obj"


def compile(
    module_override: Optional[List[Override]] = None,
    gm_transformation: Optional[Callable[[fx.GraphModule], fx.GraphModule]] = None,
    parallel_mode: Optional[ParallelMode] = None,
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
                    func, module_override, mode, gm_transformation, *args, **kwargs
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
