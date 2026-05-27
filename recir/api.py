# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import inspect
import logging
import os
from collections import OrderedDict
from contextlib import contextmanager, ExitStack, nullcontext
from copy import copy
from dataclasses import dataclass
from functools import partial, wraps
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Tuple,
    TYPE_CHECKING,
    Union,
)

if TYPE_CHECKING:
    from recir.backends.base import Backend

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

logger: logging.Logger = logging.getLogger(__name__)


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
            traced function execution (e.g., gradients_tagging for the GSPMD
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

    # ``torch.autograd.detect_anomaly`` was historically wrapped around the
    # ``make_fx`` call to catch nan/inf in the trace, but it segfaults under
    # TPU multi-host runtime (the autograd anomaly hook allocates Python
    # wrappers around FakeTensors during the trace). Default to OFF; set
    # ``RECIR_DETECT_ANOMALY=1`` to re-enable for local debugging.
    _detect_anomaly_ctx = (
        torch.autograd.detect_anomaly(check_nan=False)
        if os.environ.get("RECIR_DETECT_ANOMALY", "0") == "1"
        else nullcontext()
    )
    with fake_mode, _enable_compile(), _detect_anomaly_ctx:
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

    For ratio > 1 (e.g., GSPMD: local → global batch), ``repeat`` expands.
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


def _resolve_trace_batch_ratio(
    parallel_mode: Optional[ParallelMode],
) -> Optional[float]:
    """Determine the per-host -> trace batch ratio for ``parallel_mode``.

    Returns ``None`` when ``parallel_mode`` doesn't request a pre-trace
    batch transform (e.g., when ``parallel_mode`` is ``None`` or an
    unrecognized mode).

    For GSPMD-style modes (the partition is a no-op; XLA inflates
    placeholder shardings at compile time), the trace should see global
    batch shapes so HuggingFace's modeling code bakes the correct
    Python-int literals into ``expand``/``view`` calls.  The ratio is
    ``num_processes`` (per-host -> per-process global; for single-host
    runs ``num_processes == 1`` and the transform is a no-op).

    For ``DataParallel`` / ``HybridParallel`` driven through the new
    ``trace`` + ``compile_from_gm`` pipeline, the DTensor partition step
    re-shards user-input placeholders back to per-process shapes, so the
    trace must start from the global batch for the runtime per-process
    inputs to match.  This mirrors the per-host -> global transform that
    the legacy :func:`_compile_data_parallel` applies inline on its own
    path; without it, ``partition_data_parallel`` narrows the placeholder
    to ``per_host_batch // num_processes`` and runtime invocation hits an
    ``aten.mm`` shape mismatch (e.g. ``(N, 10) @ (10/num_processes, K)``).
    """
    if parallel_mode is None:
        return None
    # Use isinstance only when the parallel-mode classes are import-safe.
    # Avoid a hard import at module load to keep recir importable in
    # environments where the XLA/distributed deps are missing.
    if isinstance(parallel_mode, GSPMDParallel):
        # GSPMD: partition is a no-op; XLA lowering inflates placeholders
        # to the global batch via _to_sharded_jax + _jit_gspmd shardings.
        # Tracing with the global batch makes baked Python-int literals
        # (HF Gemma3 / Llama / Qwen modeling code) consistent with the
        # GSPMD runtime shape.
        return float(parallel_mode.num_processes)
    # DataParallel / HybridParallel: the v2 parametrization-based trace path
    # (in :func:`trace`) applies its own per-process -> per-device batch
    # transform via ``1/(world_size // num_processes)``.  Returning ``None``
    # here keeps the GSPMD-only contract of this helper explicit and
    # preserves the original behavior for plain
    # ``recir.compile(parallel_mode=DataParallel(), ...)``.
    return None


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
    input_batch_dim: int,
    *,
    dedup: bool = True,
) -> _CompiledResult:
    """Flatten state, dedup collectives, and tag batch inputs."""
    flat_state = pytree.tree_leaves([params_and_buffers, named_states])
    gm = _to_caller_flattened_graph_module(gm)
    if dedup:
        gm = _dedup_collectives(gm)
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
    parallel_mode: ParallelMode,
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
            parallel_mode,
            *args,
            region_transforms=region_transforms,
            **kwargs,
        )
    elif isinstance(parallel_mode, (DataParallel, HybridParallel)):
        result = _compile_data_parallel_v2(
            func,
            parallel_mode,
            *args,
            region_transforms=region_transforms,
            **kwargs,
        )
    else:
        raise TypeError(
            f"Unsupported parallel_mode type: {type(parallel_mode).__name__}"
        )
    if region_transforms:
        for transform in region_transforms:
            transform.finalize_compiled_graph(result.gm)

    return result


# ---------------------------------------------------------------------------
# v7.1 public lower-level API: trace + compile_from_gm
# ---------------------------------------------------------------------------
#
# These two functions split the legacy ``_compile_and_transform`` pipeline at
# the partition boundary so surgical orchestrators (e.g.
# ``mivilm.surgeon.compile``) can splice graph-level edits between the local
# trace (stage 2) and partition (stage 5).
#
# ``trace()`` runs stages 0 + 2 (eager pre-trace context + ``make_fx``) and
# returns ``(gm, flat_state)``.  The bundle of (mod, opt, params, buffers,
# named_states) is stashed on the gm as ``gm._recir_trace_ctx`` so
# ``compile_from_gm`` can recover it without forcing the user to pass it
# through.
#
# ``compile_from_gm()`` runs stages 4-10 (region_transforms.finalize,
# partition, optimize_passes, IterGraph wrap, dispatch) using the trace
# context recovered from the gm, the explicit ``parallel_mode``, and the
# explicit ``backend``.


# Marker attribute used to stash trace context on a graph module.
_TRACE_CTX_ATTR = "_recir_trace_ctx"


@dataclass
class _TraceContext:
    """Per-trace bundle stashed on ``gm._recir_trace_ctx``.

    Holds the references that :func:`compile_from_gm` needs to drive the
    post-trace stages: the original module, optimizer, parameter / buffer /
    optimizer-state dicts, and any embedding-specific data computed during
    the v2 trace (used by :func:`rewrite_sparse_embeddings`).
    """

    mod: nn.Module
    opt: Optional[torch.optim.Optimizer]
    params: Dict[str, nn.Parameter]
    buffers: Dict[str, torch.Tensor]
    named_states: Dict[str, Any]
    input_batch_dim: int
    embedding_param_indices: Optional[set[int]] = None


def _set_trace_ctx(gm: fx.GraphModule, ctx: _TraceContext) -> None:
    """Attach ``ctx`` to ``gm`` so :func:`compile_from_gm` can recover it.

    Stored as a Python attribute (not a buffer), so it survives
    ``gm.recompile()`` but is not serialized by ``torch.save``.
    """
    setattr(gm, _TRACE_CTX_ATTR, ctx)


def _get_trace_ctx(gm: fx.GraphModule) -> _TraceContext:
    """Recover the trace context stashed by :func:`trace`."""
    ctx = getattr(gm, _TRACE_CTX_ATTR, None)
    if ctx is None:
        raise RuntimeError(
            "recir.compile_from_gm requires a gm produced by recir.trace "
            "(missing trace context attribute). If you mutated the gm via "
            "extract/replace_region, the stash is preserved automatically; "
            "if you built the gm from scratch you must call recir.trace first."
        )
    return cast(_TraceContext, ctx)


def trace(  # noqa: C901 — orchestration: dispatches on parallel_mode flavor (DP/HybridParallel vs GSPMD vs None); splitting the branches into helpers would obscure the shared post-trace tagging.
    callable_: Callable[..., Any],
    *args: Any,
    region_transforms: Sequence[RegionTransform] = (),
    input_batch_dim: int = 0,
    parallel_mode: Optional[ParallelMode] = None,
    **kwargs: Any,
) -> Tuple[fx.GraphModule, List[torch.Tensor]]:
    """Stage 2 standalone: trace ``callable_`` into an FX graph module.

    Wraps :func:`_local_compile` plus the ``region_transforms`` bind +
    discovery + per-root context setup that the legacy paths perform
    pre-trace.  Returns ``(gm, flat_state)`` where ``flat_state`` is 1:1 with
    the first ``len(flat_state)`` placeholders of the gm.

    The returned gm has an internal ``_recir_trace_ctx`` attribute holding the
    references (``mod``, ``opt``, ``params``, ``buffers``, ``named_states``)
    that :func:`compile_from_gm` uses to drive the post-trace stages.

    After tracing this also runs :func:`_tag_batch_inputs` +
    :func:`propagate_batch_input_meta` so every placeholder carries
    ``is_batch_input`` / ``is_weight`` / ``batch_dim`` metadata.  This makes
    the meta-driven branch in mivilm's ``_semantic_input_map`` work without
    falling back to the (broken) ordinal heuristic that misidentifies
    optimizer-state placeholders (e.g. Adam's 0-dim ``step`` scalar) as
    user inputs.

    Args:
        callable_: the function to trace (typically a ``train_step``).
        *args: positional arguments to ``callable_``.  Must include an
            :class:`nn.Module` (and optionally a :class:`torch.optim.Optimizer`).
        region_transforms: optional :class:`RegionTransform` sequence applied
            to scope-rewrite sub-regions of the forward graph during tracing.
            Their ``bind_signature(callable_)`` is called once before the
            outer trace and their ``_apply_region_transforms`` ExitStack
            wrappers are entered around the trace itself.
        input_batch_dim: the batch dimension on user-input tensors.  Stored on
            each user-input placeholder as ``meta["batch_dim"]`` by
            :func:`_tag_batch_inputs`.  Defaults to ``0`` (matches
            :class:`DataParallel` and most :class:`GSPMDParallel` setups).
        parallel_mode: optional :class:`ParallelMode` driving the pre-trace
            setup.  Three flavors are recognized:

              * :class:`DataParallel` / :class:`HybridParallel`: trace via
                autograd-emitted parametrization (``parametrize_params`` +
                ``collectives_insertion``) so the resulting gm has the
                allgather / reduce-scatter ops baked in.  Inputs are shrunk
                from per-process to per-device shape via the
                ``1/(world_size // num_processes)`` batch transform.
                ``shard_params`` runs BEFORE tracing so the gm captures
                sharded parameter shapes.  For :class:`HybridParallel`, the
                embedding parameters are excluded from collectives at trace
                time and rewritten post-trace by
                :func:`compile_from_gm` via :func:`rewrite_sparse_embeddings`.
              * :class:`GSPMDParallel`: expand per-process batch tensors to
                the global batch (HuggingFace-style modeling code bakes
                ``inputs_embeds.shape[0]`` as a Python int; without this
                pre-trace expansion the literal becomes stale once GSPMD
                lowering inflates placeholders to global shape).  Trace
                under :func:`gradients_tagging` so the gm carries
                ``_spmd::tag_grad`` markers consumed by
                :func:`compile_from_gm`'s GSPMD branch.
              * ``None``: trace as-is (legacy behavior for callers that
                don't request a parallelism strategy).
        **kwargs: keyword arguments to ``callable_``.

    Returns:
        ``(gm, flat_state)``.
    """
    region_transforms_list = list(region_transforms)

    # Stage 2 prep: bind signatures so region transforms can resolve
    # signature-dependent inputs (e.g. RecomputeTransform's selector).
    for transform in region_transforms_list:
        bind = getattr(transform, "bind_signature", None)
        if bind is not None:
            bind(callable_)

    # Pre-extract mod (and opt for the DP/HybridParallel branch's pre-trace
    # ``shard_params`` call); ``_local_compile`` re-extracts both itself.
    mod: Optional[nn.Module] = None
    opt: Optional[torch.optim.Optimizer] = None
    for arg in pytree.arg_tree_leaves(*args, **kwargs):
        if isinstance(arg, nn.Module) and mod is None:
            mod = arg
        if isinstance(arg, torch.optim.Optimizer) and opt is None:
            opt = arg
    assert mod is not None, "recir.trace requires an nn.Module in the arguments"
    _mod: nn.Module = mod

    # Dispatch on parallel_mode flavor for the trace setup.
    embedding_param_indices: Optional[set[int]] = None
    if isinstance(parallel_mode, (DataParallel, HybridParallel)):
        # DataParallel / HybridParallel v2: parametrization-based collective
        # insertion (no DTensor partitioner downstream).  Mirrors the body
        # of the legacy-path ``_compile_data_parallel_v2`` helper,
        # restructured so the post-trace tagging / context-stash steps are
        # shared with the GSPMD branch below.

        # Extract mesh and DP style from the parallel mode.
        if isinstance(parallel_mode, HybridParallel):
            mesh = parallel_mode.mesh
            dp_style = parallel_mode.dense_style
        else:
            mesh = parallel_mode.device_mesh
            dp_style = parallel_mode.parallel_style
        world_size: int = parallel_mode.num_ranks

        # Resolve embedding param FQNs from the plan (HybridParallel only).
        embedding_param_fqns: Optional[set[str]] = None
        if isinstance(parallel_mode, HybridParallel):
            fqns = parallel_mode.embedding_plan.param_fqns
            embedding_param_fqns = fqns if fqns else None

        # Build transforms list (RegatherTransform for FULLY_SHARD + caller
        # region_transforms).  Embedding params are excluded from the
        # regather so that the post-trace row-wise embedding rewrite owns
        # them.
        dp_transforms: list[RegionTransform] = []
        if dp_style == DataParallelStyle.FULLY_SHARD:
            dp_transforms.append(
                RegatherTransform(skip_param_fqns=embedding_param_fqns)
            )
        if region_transforms_list:
            dp_transforms.extend(region_transforms_list)

        # Extract params_and_buffers BEFORE shard_params so the FQN order
        # matches the placeholder ordering ``_local_compile`` will emit.
        # Embedding indices computed from this snapshot map to placeholder
        # positions in the traced graph.
        params_and_buffers: Dict[str, Any] = {
            **dict(_mod.named_parameters(remove_duplicate=False)),
            **dict(_mod.named_buffers(remove_duplicate=False)),
        }
        if isinstance(parallel_mode, HybridParallel):
            embedding_param_indices = parallel_mode._compute_embedding_param_indices(
                params_and_buffers
            )

        # HybridParallel-only ShapeEnv: custom ops with data-dependent output
        # shapes (``recir_torchrec::kjt_split_per_feature``,
        # ``recir_preproc::mot_preprocessor_forward``) mint fresh SymInts via
        # ``torch.library.get_ctx().new_dynamic_size()`` from their fake
        # kernels; without a ShapeEnv those calls raise
        # ``DynamicOutputShapeException``.
        fake_mode: FakeTensorMode = FakeTensorMode(
            allow_non_fake_inputs=True,
            shape_env=ShapeEnv() if isinstance(parallel_mode, HybridParallel) else None,
        )

        # Apply the per-process -> per-device batch transform up front, so
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
            dp_transforms, callable_, args, kwargs, mod=_mod
        )

        # Shard params on the module and optimizer states.  Mutates the dict
        # and the module in-place; also updates ``opt.state`` so that
        # parameter -> state mappings remain valid post-shard.
        parallel_mode.shard_params(_mod, params_and_buffers, opt)

        # Apply region transforms AFTER sharding, with parametrization.  The
        # traced forward includes allgather ops from ``collectives_insertion``
        # so each per-parameter region encompasses allgather + chain +
        # compute.  Embedding params (if any) are excluded from collectives
        # — they use AllToAll inserted post-trace, not allgather.
        collectives_fn = collectives_insertion(dp_style, mesh)
        parametrize_kwargs: Dict[str, Any] = {}
        if embedding_param_fqns:
            parametrize_kwargs["exclude_param_fqns"] = embedding_param_fqns
        with ExitStack() as region_stack:
            _enter_region_transform_contexts(
                region_stack,
                dp_transforms,
                discovery,
                fake_mode,
                parametrize_cm_factory=lambda root_mod: partial(
                    parametrize_params,
                    root_mod,
                    collectives_fn,
                    **parametrize_kwargs,
                ),
            )
            # When region transforms are active, gm_fwd already contains the
            # allgather ops (baked in during the parametrized trace), so we
            # must NOT also pass ``collectives_fn`` to ``_local_compile`` —
            # that would double-allgather and corrupt gradients.
            local_result = _local_compile(
                callable_,
                *args,
                fake_mode=fake_mode,
                _parametrize_fns=[] if dp_transforms else [collectives_fn],
                _exclude_param_fqns=embedding_param_fqns if not dp_transforms else None,
                **kwargs,
            )
    else:
        # GSPMDParallel or no parallel_mode: legacy trace path.  Expand
        # per-process batch tensors to the global batch BEFORE tracing so
        # ``inputs_embeds.shape[0]`` resolves to the global batch and
        # HuggingFace's modeling code bakes the correct literal into
        # ``expand`` / ``view`` / ``reshape`` calls.  Otherwise GSPMD lowering
        # inflates placeholders to global shape post-trace but consumer
        # literals still reference the per-host shape -> JAX broadcast
        # mismatch.  Mirrors the pre-trace transform in :func:`_compile_gspmd`.
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
        trace_batch_ratio = _resolve_trace_batch_ratio(parallel_mode)
        if trace_batch_ratio is not None and trace_batch_ratio != 1:
            input_transform = _make_batch_transform(
                input_batch_dim,
                ratio=trace_batch_ratio,
                fake_mode=fake_mode,
            )
            args = pytree.tree_map_only(torch.Tensor, input_transform, args)
            kwargs = pytree.tree_map_only(torch.Tensor, input_transform, kwargs)

        discovery = _discover_for_region_transforms(
            region_transforms_list, callable_, args, kwargs, mod=_mod
        )
        with ExitStack() as region_stack:
            _enter_region_transform_contexts(
                region_stack,
                region_transforms_list,
                discovery,
                fake_mode,
            )
            # Apply ``gradients_tagging`` during the inner trace so the
            # resulting gm carries ``_spmd::tag_grad`` markers on every
            # parameter gradient.  ``_compile_gspmd`` historically always
            # passed this wrapper; without it ``compile_from_gm``'s GSPMD
            # branch would have no grad markers to scale / erase.  Mirrors
            # ``_local_compile(..., _stateless_func_wrapper=gradients_tagging)``
            # in the legacy GSPMD entry point.
            local_result = _local_compile(
                callable_,
                *args,
                fake_mode=fake_mode,
                _stateless_func_wrapper=gradients_tagging,
                **kwargs,
            )

    flat_state = pytree.tree_leaves(
        [
            {**local_result.params, **local_result.buffers},
            local_result.named_states,
        ]
    )
    # Tag placeholders with is_batch_input / is_weight / batch_dim so the
    # meta-driven branch in downstream consumers (e.g. mivilm.surgeon's
    # ``_semantic_input_map``) can identify user-input placeholders without
    # relying on the broken ordinal heuristic that misses opt named_states.
    # The first ``len(flat_state)`` placeholders are params + buffers + opt
    # named_states; anything past that is a user input.
    _tag_batch_inputs(local_result.gm, len(flat_state), input_batch_dim)
    # Propagate batch-input meta through ops so intermediate FX nodes that
    # become subgraph boundaries (after later partition/lower) inherit
    # is_batch_input / batch_dim — mirrors the legacy ``_finalize_graph`` flow.
    propagate_batch_input_meta(local_result.gm)
    ctx = _TraceContext(
        mod=local_result.mod,
        opt=local_result.opt,
        params=local_result.params,
        buffers=local_result.buffers,
        named_states=local_result.named_states,
        input_batch_dim=input_batch_dim,
        embedding_param_indices=embedding_param_indices,
    )
    _set_trace_ctx(local_result.gm, ctx)
    return local_result.gm, flat_state


def compile_from_gm(  # noqa: C901 — orchestration: stages 4-8 of the pipeline are intentionally laid out inline so the # Stage N comments stay readable; refactor would split the pipeline across helpers.
    gm: fx.GraphModule,
    flat_state: List[torch.Tensor],
    *,
    parallel_mode: ParallelMode,
    backend: "Backend",
    region_transforms: Sequence[RegionTransform] = (),
) -> _CompiledResult:
    """Stages 4-10: continue the pipeline from a pre-traced gm.

    The full recir compile pipeline is split into ten stages, of which
    :func:`trace` runs stages 0-3 and this function runs stages 4-10:

    Pipeline stages (consult the inline ``# Stage N:`` comments below for
    the source-level boundaries):

    * **Stage 0** -- ``eager_preproc`` hooks (handled by :func:`trace`).
      User-supplied callbacks mutate the original ``nn.Module`` *before*
      tracing (e.g. attach hooks, register buffers, swap non-traceable
      submodules).  Not run by this function.
    * **Stage 2** -- ``make_fx`` trace (handled by :func:`trace`). Produce
      the initial ``fx.GraphModule`` plus its placeholder ↔ flat_state
      binding via ``_local_compile``.  Not run here.
    * **Stage 3** -- post-trace meta tagging (handled by :func:`trace`).
      Run ``_tag_batch_inputs`` + ``propagate_batch_input_meta`` so every
      placeholder carries ``is_batch_input`` / ``is_weight`` / ``batch_dim``
      metadata.  Not run here.
    * **Stage 4** -- ``region_transforms[*].finalize_compiled_graph(gm)``.
      Each :class:`RegionTransform` gets a chance to finalize after the
      graph is partitioned (note: stage-4 runs *after* stage-5 in the code
      below to match the legacy ``_compile_and_transform`` ordering).
    * **Stage 5** -- ``parallel_mode.shard_params`` + ``partition``.
      ``shard_params`` updates ``opt.state`` and may re-key parameters
      in-place; we then re-extract ``named_states`` by walking the
      now-sharded module and call ``partition_from_module`` to rewrite the
      graph for the chosen parallelism strategy (DP/FSDP/GSPMD/etc.).
      Also re-flattens the graph via ``_to_caller_flattened_graph_module``
      so the caller can invoke ``gm(*flat_state, *user_inputs)`` directly,
      and rebuilds ``flat_state`` from the post-shard params/buffers/states.
    * **Stage 6** -- ``backend.optimize_passes`` applied to
      ``(gm, flat_state)``.  Each pass may rewrite the graph and/or update
      flat_state.  Passes are dispatched by ``inspect.signature`` arity:
      two-argument passes get ``(gm, flat_state)``; single-argument passes
      get ``(gm,)``.
    * **Stage 7** -- ``IterGraphModule(gm)`` wrap if
      ``backend.uses_iter_graph``.  Wraps the graph in an
      :class:`IterGraphModule` so cross-iteration passes (e.g. comm-fusion,
      lazy-lowering hooks) can operate.
    * **Stage 8** -- ``backend.dispatch(gm)``.  Backend-owned final
      rewrite.  For ``XLA*Backend`` this performs ``partial_lower_xla`` to
      splice ``_XLAModule`` call_modules into the graph; for
      ``InductorBackend`` this is a no-op pass-through.
    * **Stage 9** -- legacy ``_finalize_graph`` (handled by the
      :func:`compile` wrapper for the data-parallel/GSPMD paths, not by
      this function).
    * **Stage 10** -- runtime-time wrapping (handled by :func:`compile` /
      :func:`sync_device_state_to_torch`, not by this function).

    Args:
        gm: a graph module produced by :func:`trace` (must carry the
            internal trace context attribute populated by ``trace`` via
            ``_set_trace_ctx``).
        flat_state: the flat state list mirroring the leading placeholders of
            ``gm``.  Reused by the returned ``_CompiledResult``.
        parallel_mode: how to partition the graph across ranks.  Validated
            against ``backend`` via ``backend.validate_with(parallel_mode)``
            before stage 5 runs.
        backend: owns stages 6-8 (optimize passes, IterGraph wrap, dispatch).
        region_transforms: same sequence passed to :func:`trace`.  Each
            transform's ``finalize_compiled_graph(gm)`` is invoked after
            partition (stage 4 in code-order, but conceptually stage 4 of
            the pipeline).

    Returns:
        A :class:`_CompiledResult` ready for runtime use via
        :func:`recir.compile`'s wrapper invocation path.
    """
    # Validate backend ↔ parallel_mode compatibility early.
    backend.validate_with(parallel_mode)

    ctx = _get_trace_ctx(gm)
    region_transforms_list = list(region_transforms)

    # Stage 5: per-mode post-trace assembly.
    #
    # The trace context already contains the (post-shard, for DP/HybridParallel)
    # ``params`` / ``buffers`` / ``named_states`` snapshot captured inside
    # :func:`trace`.  Branch shape:
    #
    #   * :class:`DataParallel` / :class:`HybridParallel`: ``shard_params`` ran
    #     inside :func:`trace` so the autograd-emitted parametrization could
    #     bake the per-parameter collectives into ``gm`` at trace time.  Here
    #     we just consume the ctx snapshot — no further partition pass is
    #     required (the DTensor partitioner is skipped entirely on this
    #     path).  For :class:`HybridParallel`, the embedding parameters were
    #     excluded from collectives at trace time; rewrite them now into
    #     decomposed AllToAll ops via :func:`rewrite_sparse_embeddings`.
    #   * :class:`GSPMDParallel`: ``shard_params`` is deferred to this stage
    #     because the GSPMD lowering inflates placeholders to the global
    #     batch post-trace; sharding before the trace would leave the gm
    #     with per-shard literals that conflict with the inflated
    #     placeholders.  After sharding we still call
    #     ``partition_from_module`` (a no-op for the GSPMD partition path,
    #     kept for symmetry with the abstract :class:`ParallelMode` contract)
    #     and then scale + erase the autograd-emitted ``_spmd::tag_grad``
    #     markers so XLA lowering can proceed.
    sharded_keys: set[str] = set()
    if isinstance(parallel_mode, (DataParallel, HybridParallel)):
        # Trace already sharded; reuse ctx snapshot.
        params_and_buffers: Dict[str, Any] = {**ctx.params, **ctx.buffers}
        named_states: Dict[str, Any] = ctx.named_states

        # HybridParallel: pattern-match aten.embedding nodes on sharded
        # embedding params and replace with decomposed AllToAll ops via
        # secondary symbolic trace.  ``embedding_param_indices`` was
        # populated by :func:`trace` against the pre-shard params dict; the
        # ordering is stable across shard_params (which mutates values in
        # place without re-keying the dict), so the indices still map to
        # placeholder positions in the traced graph.
        embedding_param_indices = ctx.embedding_param_indices
        if isinstance(parallel_mode, HybridParallel) and embedding_param_indices:
            flat_params_buffers = list(ctx.params.values()) + list(ctx.buffers.values())
            embedding_param_tensors: Dict[int, torch.Tensor] = {
                i: flat_params_buffers[i] for i in embedding_param_indices
            }
            rewrite_sparse_embeddings(
                gm,
                parallel_mode.mesh,
                embedding_param_tensors,
            )
    else:
        # GSPMDParallel (or future modes that still require a DTensor partition
        # pass).  Mirror the legacy ``_compile_gspmd`` ordering: shard_params
        # first (updates ``opt.state`` and may re-key parameters), then
        # re-extract named_states by walking the now-sharded module, then
        # ``partition_from_module``.
        params_and_buffers = {**ctx.params, **ctx.buffers}
        sharded_keys = parallel_mode.shard_params(ctx.mod, params_and_buffers, ctx.opt)
        named_states = {}
        _opt = ctx.opt
        if _opt is not None and len(_opt.state) > 0:
            # ``shard_params`` may have re-keyed ``opt.state`` to the new
            # (post-shard) Parameter identities; walk the live module to pick
            # up the current binding.
            for n, p in ctx.mod.named_parameters(remove_duplicate=False):
                if p in _opt.state:
                    named_states[n] = _opt.state[p]  # type: ignore[index]
            # Coverage check: ``shard_params`` may re-key params off the
            # module so the walk above misses entries that ``trace`` captured
            # in ``ctx.named_states``. Warn (rather than silently drop) so
            # downstream optimizer-state restoration does not lose entries.
            dropped = set(ctx.named_states.keys()) - set(named_states.keys())
            if dropped:
                logger.warning(
                    "GSPMD named_states rebuild dropped %d entries that were "
                    "present in the pre-shard ctx snapshot: %s",
                    len(dropped),
                    sorted(dropped),
                )
        else:
            # Either no optimizer was supplied, or ``_opt.state`` is empty
            # (optimizer not stepped yet, or a partial-lower path that
            # never populated state).  In both cases ``ctx.named_states``
            # — captured by ``trace`` — is the authoritative snapshot;
            # fall through so we do not silently discard it.
            named_states = ctx.named_states
        gm = parallel_mode.partition_from_module(
            gm, ctx.mod, ctx.opt, params_and_buffers, named_states
        )

        # GSPMD-specific tag_grad fixup.  ``partition_from_module`` is a no-op
        # for GSPMD, so the autograd-emitted ``_spmd::tag_grad`` markers
        # survive into the lowered graph; they must be scaled (for
        # ``grad_reduction == "avg"``) and erased here.  Mirrors the
        # post-trace block in the legacy ``_compile_gspmd``:
        #   * scale: GSPMD lowers the autograd-emitted grad accumulation
        #     through an all-reduce SUM.  For sum-loss + "avg" grad
        #     reduction (DDP semantics), divide each tag_grad's output by
        #     ``num_ranks``.  For mean-loss (grad_reduction="sum"), the
        #     traced backward already normalises by global numel so the SUM
        #     alone is correct.
        #   * erase: tag_grad has no XLA lowering, so leaving the markers
        #     in the graph triggers ``_is_xla_compatible`` failures at
        #     ``segment_and_lower`` time (the symptom that surfaces as
        #     ``test_no_graph_breaks_gspmd`` failing).
        if isinstance(parallel_mode, GSPMDParallel):
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
                            # Filter out div_node itself so the rewrite skips
                            # the freshly-created (node, num_ranks) edge --
                            # otherwise div_node would self-reference and we
                            # would need to restore div_node.args post-hoc.
                            # Default-arg ``dn=div_node`` pins the current
                            # iteration's div_node (avoids B023 late binding).
                            node.replace_all_uses_with(
                                div_node, lambda u, dn=div_node: u is not dn
                            )
            for node in list(gm.graph.nodes):
                if (
                    node.op == "call_function"
                    and node.target is torch.ops._spmd.tag_grad.default
                ):
                    node.replace_all_uses_with(node.args[0])
                    gm.graph.erase_node(node)
            gm.recompile()

    # Move the pytree-flattening responsibility from the gm to the caller so
    # the recir wrapper can invoke ``gm(*flat_inps)`` directly.  Mirrors the
    # legacy ``_finalize_graph`` step; without this, the gm's codegen still
    # expects the original nested-pytree (model, opt, *args, **kwargs)
    # structure and runtime invocation explodes.
    gm = _to_caller_flattened_graph_module(gm)

    # Rebuild flat_state from the post-shard params/buffers/named_states.
    # ``shard_params`` may have substituted parameter objects, so the
    # original ``flat_state`` references can be stale.  Tail entries beyond
    # the model state (added via ``recir.add_state``) are preserved.
    rebuilt_state = pytree.tree_leaves([params_and_buffers, named_states])
    if len(flat_state) > len(rebuilt_state):
        rebuilt_state.extend(flat_state[len(rebuilt_state) :])
    flat_state = rebuilt_state

    # GSPMD/FSDP: mirror legacy ``_compile_gspmd`` by setting
    # ``is_fsdp_sharded`` on placeholders that correspond to FSDP-sharded
    # params/states.  ``_jit_gspmd`` inspects this meta to decide which
    # inputs are local shards (vs replicated tensors).  Without the tag,
    # the GSPMD lowering would treat sharded inputs as replicated and the
    # cross-process gather/all-reduce shape contract breaks.  Only populated
    # for the GSPMD branch above — the DP/HybridParallel branch consumes
    # the parametrized gm directly and does not need placeholder-level
    # FSDP tagging.
    if sharded_keys:
        is_sharded: list[bool] = [key in sharded_keys for key in params_and_buffers]
        for fqn, state in named_states.items():
            is_sharded.extend([fqn in sharded_keys] * len(state))
        tensor_idx = 0
        for node in gm.graph.nodes:
            if node.op != "placeholder":
                continue
            val = node.meta.get("val", None)
            if isinstance(val, torch.Tensor):
                node.meta["is_fsdp_sharded"] = (
                    is_sharded[tensor_idx] if tensor_idx < len(is_sharded) else False
                )
                tensor_idx += 1

    # Stage 4 (post-partition is the legacy ordering used by
    # ``_compile_and_transform``): region transforms get a chance to finalize
    # after the graph has been partitioned.
    for transform in region_transforms_list:
        transform.finalize_compiled_graph(gm)

    # Stage 6: backend-configured optimize passes.  Each pass may rewrite the
    # graph and/or update flat_state.  Some passes accept (gm,) and others
    # accept (gm, flat_state) — dispatch via signature inspection so a
    # ``TypeError`` raised from inside ``opt_pass`` is not silently swapped
    # for the single-arg form.
    #
    # Only count *positional* params toward the arity check.  Otherwise a
    # signature like ``def my_pass(gm, *, mode="x")`` (1 positional + 1
    # kw-only) would falsely set ``wants_state=True`` and we'd call
    # ``opt_pass(gm, flat_state)`` → ``TypeError`` at runtime.
    _POSITIONAL_KINDS = (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.POSITIONAL_ONLY,
    )
    for opt_pass in backend.optimize_passes:
        try:
            sig_params = inspect.signature(opt_pass).parameters.values()
            positional_params = [p for p in sig_params if p.kind in _POSITIONAL_KINDS]
            wants_state = len(positional_params) >= 2
        except (TypeError, ValueError):
            # ``inspect.signature`` raises ``ValueError`` on builtins /
            # C-extensions and ``TypeError`` on non-callables; in either
            # case fall back to the legacy single-arg form.
            wants_state = False
        if wants_state:
            result = opt_pass(gm, flat_state)
        else:
            result = opt_pass(gm)
        if isinstance(result, tuple) and len(result) == 2:
            gm, flat_state = result  # pyre-ignore[9]
        elif isinstance(result, fx.GraphModule):
            gm = result

    # Stage 7: optionally wrap in IterGraphModule for cross-iteration passes
    # and lazy-lowering hooks.
    if backend.uses_iter_graph:
        # Local import to avoid pulling IterGraphModule into the
        # module-import path for callers that don't use this path.
        from recir.iter_graph_module import IterGraphModule

        gm = cast(fx.GraphModule, IterGraphModule(gm))

    # Stage 8: dispatch (backend.dispatch returns an nn.Module — typically
    # the same gm rewritten as a partial-lowered module, but the contract
    # only requires it be a callable nn.Module compatible with the flat-state
    # invocation convention).
    dispatched = backend.dispatch(gm)
    # Preserve the gm reference (the dispatched module is typically the gm
    # itself, rewritten in place).  ``_CompiledResult.gm`` is callable, so
    # we use the dispatched module here.
    final_gm = cast(fx.GraphModule, dispatched)

    return _CompiledResult(
        gm=final_gm,
        mod=ctx.mod,
        opt=ctx.opt,
        flat_state=flat_state,
        state_manager=backend.device_state_manager(),
    )


def _compile_via_backend(
    func: Callable[..., Any],
    parallel_mode: ParallelMode,
    backend: "Backend",
    *,
    eager_preproc: Sequence[Callable[[nn.Module], None]],
    region_transforms: Optional[List[RegionTransform]],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> _CompiledResult:
    """Internal: drive the new v7.1 trace + compile_from_gm pipeline.

    Centralises the eager_preproc → trace → compile_from_gm call sequence so
    :func:`compile` doesn't have to duplicate it.  Used only when the user
    passes ``backend=`` to :func:`compile`.
    """
    rt = list(region_transforms or ())

    # Stage 0: apply eager pre-trace hooks to every nn.Module argument.
    # Multi-module signatures (e.g. model + loss module, teacher/student)
    # all need preprocessing -- not just the first one encountered.
    if eager_preproc:
        for arg in pytree.arg_tree_leaves(*args, **kwargs):
            if isinstance(arg, nn.Module):
                for hook in eager_preproc:
                    hook(arg)

    # Stage 2: trace.  Propagate the parallel mode's batch dim so placeholder
    # meta tagging matches what stages 4-10 will later assume.  Also pass the
    # parallel mode itself so the trace can expand per-process tensors to the
    # global batch when the mode is GSPMD-like -- avoids the HF Gemma3 /
    # Llama / Qwen ``shape[0]``-baked-as-int problem that surfaces post-GSPMD
    # at JAX broadcast time.  Mirrors the per-host -> global transform that
    # ``_compile_gspmd`` performs on the legacy partition-only path.
    gm, flat_state = trace(
        func,
        *args,
        region_transforms=rt,
        input_batch_dim=parallel_mode.input_batch_dim,
        parallel_mode=parallel_mode,
        **kwargs,
    )

    # Stages 4-10.
    return compile_from_gm(
        gm,
        flat_state,
        parallel_mode=parallel_mode,
        backend=backend,
        region_transforms=rt,
    )


def _compile_data_parallel_v2(
    func: Callable[..., Any],
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

    # 4. Extract params_and_buffers.  Critical for correctness: shard_params
    #    mutates _mod by FQN, and embedding indices into this dict must map
    #    to placeholder positions in the traced graph (which uses the
    #    post-shard params ordering).
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
        parallel_mode.input_batch_dim,
    )


def _compile_gspmd(
    func: Callable[..., Any],
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
                    # Filter out div_node itself so the rewrite skips the
                    # freshly-created (node, num_ranks) edge -- otherwise
                    # div_node would self-reference and we would need to
                    # restore div_node.args post-hoc.
                    # Default-arg ``dn=div_node`` pins the current iteration's
                    # div_node (avoids B023 late binding).
                    node.replace_all_uses_with(
                        div_node, lambda u, dn=div_node: u is not dn
                    )
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


def make_compiled_runner(compiled_obj: Any) -> Callable[..., Any]:
    """Wrap a ``_CompiledResult`` into a user-facing callable.

    Builds the standard "prepare_inputs -> gm(*flat_inps) -> process_outputs"
    runtime adapter that every recir orchestrator (``recir.compile``,
    :func:`recir.examples.dlrmv3.jax_embedding.compile_with_jax_embedding`,
    the rankfm trainer, custom dispatchers in tests) emits after stage 10.
    The wrapper:

    * pops ``last_train_step`` from ``kwargs`` (default ``False``);
    * routes user args through :meth:`compiled_obj.state_manager.prepare_inputs`
      to flatten and dedup against ``compiled_obj.flat_state``;
    * invokes ``compiled_obj.gm(*flat_inps)``, falling back to a plain call
      when the GM doesn't accept the ``last_iter`` kwarg (i.e. when no
      :class:`IterGraphModule` wrap was applied);
    * post-processes via :meth:`compiled_obj.state_manager.process_outputs`
      to map updated state tensors back into the user-visible return value;
    * stashes ``compiled_obj`` on the wrapper's ``__dict__`` under
      :data:`COMPILED_OBJECT_KEY` so :func:`sync_device_state_to_torch` can
      find it.

    Args:
        compiled_obj: a :class:`_CompiledResult` returned by
            :func:`compile_from_gm` (or the legacy
            :func:`_compile_and_transform`).

    Returns:
        A callable with the same signature as the original train step.
        Pass ``last_train_step=True`` on the final call so IterGraph-wrapped
        modules can flush their state.
    """

    # Introspect once per wrapper: determine whether the underlying GM's
    # ``forward`` accepts a ``last_iter`` parameter (i.e. it's an
    # IterGraphModule wrap). Avoids relying on TypeError substring matching,
    # which is fragile across Python/torch versions. Explicit ``bool``
    # annotation is required because the value is captured by the ``runner``
    # closure below and pyre [53] flags unannotated captured locals.
    accepts_last_iter: bool
    try:
        gm_sig = inspect.signature(compiled_obj.gm)
        accepts_last_iter = "last_iter" in gm_sig.parameters
    except (TypeError, ValueError):
        accepts_last_iter = False

    def runner(*call_args: Any, **call_kwargs: Any) -> Any:
        last_train_step = (
            call_kwargs.pop("last_train_step", False) if call_kwargs else False
        )
        user_inps = pytree.arg_tree_leaves(*call_args, **call_kwargs)
        flat_inps = compiled_obj.state_manager.prepare_inputs(
            compiled_obj.gm,
            compiled_obj.flat_state,
            user_inps,
            last_train_step,
        )
        with torch.no_grad():
            if last_train_step and accepts_last_iter:
                # Final iter: pass ``last_iter`` so IterGraphModule can flush.
                output_all = compiled_obj.gm(*flat_inps, last_iter=last_train_step)
            else:
                output_all = compiled_obj.gm(*flat_inps)
            output = output_all[0]
            updated_flat = pytree.tree_leaves(output_all[1:])
            return compiled_obj.state_manager.process_outputs(
                compiled_obj.flat_state,
                output,
                updated_flat,
                last_train_step,
            )

    runner.__dict__[COMPILED_OBJECT_KEY] = compiled_obj
    return runner


def compile(  # noqa: C901 — orchestration: stages 0-9 + legacy DP/GSPMD wiring intentionally kept inline; refactor would obscure the pipeline.
    parallel_mode: Optional[ParallelMode] = None,
    region_transforms: Optional[List[RegionTransform]] = None,
    *,
    backend: Optional["Backend"] = None,
    eager_preproc: Sequence[Callable[[nn.Module], None]] = (),
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    r"""Compile and optimize a callable, which can be a train step within a training loop.

    This is the public top-level entry point for the recir compile pipeline.
    Extracts the :class:`nn.Module` + :class:`torch.optim.Optimizer` from the
    decorated callable's arguments, traces them via :func:`trace`, runs
    stages 3-9 via :func:`compile_from_gm`, then wraps the result in a
    runtime callable that manages device state, flattens user args, threads
    ``flat_state`` through every invocation, and routes ``last_iter=True``
    for IterGraph cleanup on the final step.

    Pipeline stages (the canonical recir vocabulary; consult the inline
    ``# Stage N:`` comments in :func:`compile_from_gm` and the per-mode
    helpers for the source-level boundaries).  **Stages 5-8 (optimize,
    stage, dispatch, build) are owned by the configured**
    :class:`~recir.backends.base.Backend`; the rest are owned by the
    orchestrator.

    * **Stage 0 -- eager_preproc.** User-supplied
      ``Sequence[Callable[[nn.Module], None]]`` hooks that mutate the eager
      module *before* tracing — attach hooks, register buffers, swap
      non-traceable submodules.  Pre-FX; no graph artifact yet.  Driven by
      :func:`_compile_via_backend` on the new path; the legacy path (no
      ``backend=``) also runs ``eager_preproc`` so it remains useful with
      raw :class:`ParallelMode` callers.
    * **Stage 1 -- trace.** :func:`functorch.make_fx` capture (Python -> FX)
      via :func:`trace` and its inner :func:`_local_compile`.  Produces the
      initial :class:`fx.GraphModule` plus a ``flat_state`` list whose
      entries map 1:1 to the leading placeholders.  Multi-process modes
      (DP / HybridParallel / GSPMD) inflate the input batch dim from
      per-host to global before tracing via ``_make_batch_transform``.
    * **Stage 2 -- post-trace meta tagging.** :func:`_tag_batch_inputs` plus
      :func:`propagate_batch_input_meta` walk the gm and tag every
      placeholder with ``is_batch_input`` / ``is_weight`` / ``batch_dim``
      meta keys.  Drives downstream correctness for surgical orchestrators
      that need to identify user-input placeholders semantically.
    * **Stage 3 -- transform.**
      ``region_transforms[*].finalize_compiled_graph(gm)``.  Each
      :class:`RegionTransform` (recompute / regather / padding) finalizes
      its rewrite after partition.  Code-order is *after* stage 4 to match
      the legacy ``_compile_and_transform`` ordering; the verb "transform"
      is reserved for these semantic rewrites.
    * **Stage 4 -- partition.** Multi-device graph split:
      ``parallel_mode.shard_params(...)`` followed by
      ``parallel_mode.partition_from_module(...)``.  Inserts collectives,
      rewrites the graph for DP / FSDP / GSPMD / Hybrid, then re-flattens
      via :func:`_to_caller_flattened_graph_module` and rebuilds
      ``flat_state``.  The verb "partition" is reserved for this distributed
      graph split.
    * **Stage 5 -- optimize** *(Backend)*.  ``backend.optimize_passes``
      rewrites the gm without changing meaning: comm fusion, dead-code
      elimination, scheduling, sparse-embedding decomposition.
      Two-argument passes get ``(gm, flat_state)``; single-argument passes
      get ``(gm,)``.  Owned by :attr:`Backend.optimize_passes`.
    * **Stage 6 -- stage** *(Backend)*.  Wrap the gm in
      :class:`IterGraphModule` when ``backend.uses_iter_graph`` is ``True``.
      Splits one logical step into setup / main / cleanup sub-graphs so
      cross-iteration passes can operate.  Gated by
      :attr:`Backend.uses_iter_graph` and applied by
      :func:`compile_from_gm`.
    * **Stage 7 -- dispatch** *(Backend)*.  Backend-owned final rewrite.
      XLA backends call ``partial_lower_xla`` to splice ``_XLAModule``
      ``call_module`` nodes into the graph; :class:`InductorBackend` is a
      no-op pass-through.  No backend IR yet — just subgraph carving +
      routing.  The verb "lower" is banned here (it collides with JAX's
      ``.lower()``); see :meth:`Backend.dispatch`.
    * **Stage 8 -- build** *(Backend)*.  First point real backend artifacts
      exist.  XLA: ``_XLAModule._compile_xla`` runs
      ``jax.jit(...).lower(...).compile(...)``.  Inductor:
      ``torch._inductor.compile_fx``.  Lazy — runs on the first forward of
      the dispatched ``call_module``, not at decoration time.  The verb
      "compile" is banned here (it's reserved for ``recir.compile``).
    * **Stage 9 -- prebuild.** Stage 8 run ahead of time on the build host,
      serialized to disk, reloaded on a production host.  Used for TPU
      canaries to avoid paying compile cost in the training pod.  See
      ``recir.lowering.aot.aot_compile_and_serialize``.

    **Runtime wrap (post-stage 9).** :func:`compile` wraps the compiled
    gm in a callable that manages device state via
    :class:`~recir.lowering.device_state.DeviceStateManager` (e.g. for XLA
    so state stays on TPU between steps), pytree-flattens user args,
    threads ``flat_state`` through every invocation, and routes
    ``last_iter=True`` for :class:`IterGraphModule` cleanup on the final
    step.  This wrapping is what :func:`compile` adds on top of the lower-
    level :func:`trace` + :func:`compile_from_gm` pair.

    **Pipeline dataflow** (``[B]`` = owned by :class:`Backend`)::

        nn.Module + args
           |
           v   Stage 0    eager_preproc (user callbacks)
           |
           v   Stage 1    trace (make_fx)              -- (gm, flat_state)
           |
           v   Stage 2    post-trace meta tagging
           |
           v   Stage 4    partition (shard + collectives)
           |
           v   Stage 3    transform (region finalize; code-order is AFTER stage 4)
           |
           v   Stage 5    optimize          [B]  Backend.optimize_passes
           |
           v   Stage 6    stage             [B]  IterGraphModule wrap if Backend.uses_iter_graph
           |
           v   Stage 7    dispatch          [B]  Backend.dispatch -- subgraph carving
           |
           v   Stage 8    build             [B]  JAX .lower-then-compile / Inductor -- lazy on first forward
           |
           v   Stage 9    prebuild               (optional AOT path -- serialize + load on target host)
           |
           v   runtime wrap (DeviceStateManager, input flattening, last_iter routing)

    **Naming discipline.**  In recir, name by the artifact you produce.
    If you emit FX, it is ``trace``.  If you make targeted edits to a
    graph, it is ``patch``.  If you intentionally change model semantics
    by rewriting a region, it is ``transform``.  If you split work across
    devices, it is ``partition``.  If you improve scheduling / fusion /
    elimination without semantic change, it is ``optimize``.  If you split
    into setup / main / cleanup, it is ``stage``.  If you carve
    backend-compatible islands and wrap them, it is ``dispatch`` — never
    ``lower``.  If you produce backend IR or an executable, it is
    ``build``.  If you do that ahead of runtime and persist the result,
    it is ``prebuild``.  Only :func:`recir.compile` may use the verb
    ``compile``.

    Args:
        parallel_mode (Optional[ParallelMode]): a :class:`ParallelMode`
            instance specifying how to parallelize the callable.  Each
            ``ParallelMode`` owns stage 4 (its
            :meth:`~ParallelMode.partition_from_module` rewrites the
            graph for the chosen DP / FSDP / GSPMD / Hybrid strategy).
            ``DataParallel()`` is used when ``None`` is passed.
            (Default: ``None``)
        region_transforms (Optional[List[RegionTransform]]): a list of
            :class:`RegionTransform` instances applied at stage 3 to
            rewrite sub-regions of the forward graph (e.g. activation
            checkpointing via :class:`RecomputeTransform`, jagged padding
            via :class:`PaddingRegionTransform`).  Stage 3 is code-order
            *after* stage 4 partition so the region rewrites see the
            post-partition graph.  (Default: ``None``)
        backend (Optional[Backend]): a :class:`recir.backends.Backend`
            instance that owns stages 5-8 (``optimize_passes``,
            ``uses_iter_graph`` IterGraph wrap, ``dispatch``, lazy
            ``build``).  When provided, ``recir.compile`` routes through
            the new :func:`trace` + :func:`compile_from_gm` pipeline.
            When ``None``, the legacy partition-only path is used (no
            stage-5+ rewrites; useful for raw DP / FSDP / GSPMD captures
            that downstream code consumes directly).
            (Default: ``None``)
        eager_preproc (Sequence[Callable[[nn.Module], None]]): a sequence
            of in-place pre-trace hooks that run against the model
            argument (stage 0).  Each hook is invoked once on the first
            :class:`nn.Module` discovered in the decorated callable's
            arguments, before :func:`trace` runs.  (Default: ``()``)

    Returns:
        A decorator that, when applied to a callable, returns a wrapped
        callable.  On the first invocation the wrapped callable runs the
        full pipeline (stages 0-9) and caches the compiled object on
        ``wrapper.__dict__[COMPILED_OBJECT_KEY]``; subsequent invocations
        reuse the cached object and pay only the runtime cost (device
        state sync + flat-state threading + gm forward).

    See Also:
        :func:`trace`: stages 0-2 alone, exposed so surgical orchestrators
            (e.g. ``mivilm.surgeon.compile``) can splice graph-level edits
            between the local trace and partition.
        :func:`compile_from_gm`: stages 3-9 alone, takes a pre-traced gm
            from :func:`trace` and drives partition / optimize / dispatch.
        :mod:`recir.backends`: the :class:`Backend` ABC and concrete
            subclasses (Inductor, XLA Dynamo-bridge, XLA GSPMD, XLA
            ShardMap) that own stages 5-8.
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
                if backend is not None:
                    # New v7.1 path: trace + compile_from_gm.
                    compiled_obj = _compile_via_backend(
                        func,
                        mode,
                        backend,
                        eager_preproc=eager_preproc,
                        region_transforms=region_transforms,
                        args=args,
                        kwargs=kwargs,
                    )
                else:
                    # Legacy partition-only path (no Backend supplied):
                    # eager_preproc still applies pre-trace (against every
                    # nn.Module argument so multi-module signatures all
                    # benefit), then the gm is partitioned but skips backend
                    # stages 6-10.
                    if eager_preproc:
                        for arg in pytree.arg_tree_leaves(*args, **kwargs):
                            if isinstance(arg, nn.Module):
                                for hook in eager_preproc:
                                    hook(arg)
                    compiled_obj = _compile_and_transform(
                        func,
                        mode,
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


# Pull through the ``GraphPatch`` type alias so consumers don't need to import
# from ``recir.backends`` if they only want the type.
from recir.backends.base import GraphPatch  # noqa: E402, F401
# ---------------------------------------------------------------------------
# v7.1 public surface: graph manipulation primitives
# ---------------------------------------------------------------------------


# Re-exports so callers can write ``from recir.api import ...`` or
# ``recir.find_markers`` interchangeably.
from recir.ops.marker import find_markers  # noqa: E402, F401
from recir.region_transforms import extract_subgraph, replace_region  # noqa: E402, F401


def add_state(
    gm: fx.GraphModule,
    flat_state: List[torch.Tensor],
    name: str,
    tensor: torch.Tensor,
) -> int:
    """Atomically add a new state tensor + matching placeholder to ``gm``.

    The new placeholder is inserted immediately after the existing trailing
    placeholder so it stays grouped with the rest of the flat state.  The
    tensor is appended to ``flat_state`` and the index of the appended entry
    in ``flat_state`` is returned (callers use this when wiring downstream
    uses against the flat-state list).

    Args:
        gm: the host graph module.  Its graph is mutated in place and
            recompiled.
        flat_state: the flat state list mirroring the leading placeholders of
            ``gm``.  Mutated in place: ``tensor`` is appended.
        name: the placeholder name (suffixed with a counter if it collides
            with an existing node name).
        tensor: the tensor whose value will be passed at runtime.

    Returns:
        The index of the newly-appended entry in ``flat_state`` (i.e.,
        ``len(flat_state) - 1`` after append).  This equals the new
        placeholder's positional index among ``gm``'s placeholders only when
        no non-flat-state placeholders precede it in ``gm``.
    """
    # Locate the insertion point: after the last existing placeholder.  This
    # keeps the placeholder block contiguous so the caller's existing
    # flat_state-to-placeholder pairing stays positional.
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    if placeholders:
        anchor = placeholders[-1]
        insert_ctx = gm.graph.inserting_after(anchor)
    else:
        # Empty graph (no existing inputs).  Insert at the very front.
        first_node = next(iter(gm.graph.nodes), None)
        insert_ctx = (
            gm.graph.inserting_before(first_node)
            if first_node is not None
            else gm.graph.inserting_after()
        )

    with insert_ctx:
        new_placeholder = gm.graph.placeholder(name)

    # Attach a fake-tensor val to the new placeholder so downstream passes
    # that inspect ``node.meta["val"]`` (e.g. batch-dim propagation, region
    # transforms) see a concrete shape.  Reuse the FakeTensorMode already
    # attached to ``gm`` (via existing placeholders' meta) when possible so
    # the new fake-tensor shares the same mode identity as the rest of the
    # graph — constructing a fresh mode per call would create fake-tensors
    # tied to a different mode object and break downstream invariants that
    # rely on a single mode per graph.
    fake_mode = torch._guards.detect_fake_mode(
        [n.meta["val"] for n in placeholders if "val" in n.meta]
    )
    if fake_mode is None:
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
    with fake_mode:
        new_placeholder.meta["val"] = fake_mode.from_tensor(tensor)
    new_placeholder.meta.setdefault("is_batch_input", False)
    new_placeholder.meta.setdefault("is_weight", False)

    flat_state.append(tensor)

    gm.graph.lint()
    gm.recompile()

    # The new placeholder is now the last placeholder; its positional index
    # among placeholders equals len(flat_state) - 1.
    return len(flat_state) - 1
