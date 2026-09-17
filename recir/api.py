# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import logging
import os
from collections import OrderedDict
from contextlib import contextmanager, ExitStack, nullcontext
from copy import copy
from dataclasses import dataclass, field
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
from recir.compile_utils import _enable_compile
from recir.decompositions import DEDUP_TARGETS, SPMD_DECOMP_TABLE
from recir.distributed.data_parallel import (
    collectives_insertion,
    DataParallelStyle,
    gradients_tagging,
    parametrize_params,
)
from recir.distributed.parallel_mode import (
    DataParallel,
    GSPMDParallel,
    HybridParallel,
    ParallelMode,
)
from recir.distributed.sharding_spec import FSDPWeightShardSpec, get_state_layout
from recir.lowering.device_state import DeviceStateManager
from recir.pipeline import build_pipeline, is_pipeline_pending, PipelineConfig
from recir.region_transforms import (
    _apply_region_transforms,
    _discover_called_modules,
    _find_forward_roots,
    RegatherTransform,
    RegionTransform,
    SparseEmbeddingTransform,
)
from torch import fx
from torch._dynamo.source import LocalSource
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.export.unflatten import _assign_attr, _AttrKind
from torch.fx.experimental.symbolic_shapes import (
    DimDynamic,
    ShapeEnv,
    StatelessSymbolicContext,
)
from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo, CodeGen
from torch.fx.graph_module import _del_attr, _get_attr
from torch.nn.utils import stateless

logger: logging.Logger = logging.getLogger(__name__)


def _maybe_dump_fx_graph(gm: torch.fx.GraphModule) -> None:
    """When RECIR_DUMP_GRAPHS is set, log the make_fx-output graph: readable code
    plus a per-node op/target/out-shape/stack-trace listing. The stack traces are
    still attached at this stage (before the downstream recir transforms strip
    them), so op sources can be inspected without a TPU run. Diagnostics must
    never break compile, so all failures are swallowed.
    """
    if os.environ.get("RECIR_DUMP_GRAPHS", "") in ("", "0", "false", "False"):
        return
    try:
        logger.info(
            "[RECIR_DUMP_GRAPHS] make_fx output (readable):\n%s",
            gm.print_readable(print_output=False),
        )
        lines = []
        for n in gm.graph.nodes:
            val = n.meta.get("val") if hasattr(n, "meta") else None
            shp = tuple(val.shape) if val is not None and hasattr(val, "shape") else ""
            st = (n.meta.get("stack_trace") or "").replace("\n", " || ")
            lines.append(f"{n.op} | {n.target} | out={shp} | {st}")
        logger.info("[RECIR_DUMP_GRAPHS] make_fx per-node:\n%s", "\n".join(lines))
    except Exception as e:  # noqa: BLE001 -- diagnostics must never break compile
        logger.warning("[RECIR_DUMP_GRAPHS] make_fx dump failed: %s", e)


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
    mod: Optional[nn.Module] = None,
    real_id_to_fqn: Optional[Dict[int, str]] = None,
    buffers: Optional[Dict[str, torch.Tensor]] = None,
    real_buf_id_to_fqn: Optional[Dict[int, str]] = None,
) -> Generator[None, None, None]:
    """Temporarily point ``opt``'s state and param_groups at the proxy
    (FakeTensor) params used during make_fx tracing, restoring on exit.
    Both must be rewired or the traced ``step()`` is a silent no-op:
    ``opt.state`` is keyed by Parameter identity, and ``param_group["params"]``
    is the list ``step()`` iterates to know what to update.

    Multi-opt (sparse + dense): each call only touches the FQNs ITS opt owns,
    derived by mapping each Parameter in ``opt.param_groups`` back to its FQN.

    ``real_id_to_fqn`` MUST come from the real, pre-reparametrize params (what
    ``opt.param_groups`` still holds). Inside the reparametrize context
    ``mod.named_parameters()`` returns proxies whose ids miss, so ``owned_fqns``
    ends up empty -> no-op step, zero optimizer ops in the GM. Prefer the
    caller-supplied map; fall back to ``mod.named_parameters()`` only without
    reparametrize.
    """
    assert opt is not None

    # Build id -> fqn from real (pre-reparametrize) Parameters.
    if real_id_to_fqn is None and mod is not None:
        real_id_to_fqn = {
            id(p): fqn for fqn, p in mod.named_parameters(remove_duplicate=False)
        }

    if real_id_to_fqn is not None:
        owned_fqns: set[str] = set()
        for group in opt.param_groups:
            for p in group["params"]:
                fqn = real_id_to_fqn.get(id(p))
                if fqn is not None:
                    owned_fqns.add(fqn)
    else:
        # No identity map available: legacy "this opt owns everything"
        # behavior, correct only for single-opt no-reparametrize setups.
        owned_fqns = set(named_states.keys())

    # update opt.state with proxy tensors (this opt's owned FQNs only)
    orig_states = copy(opt.state)
    for n, state in named_states.items():
        if n in owned_fqns:
            # opt.state's key type is Parameter, but indexed by FQN here
            opt.state[params[n]] = state  # type: ignore[index]

    # Rewrite every param_group's params list to point at the proxy
    # Parameter for each owned FQN, preserving per-group membership.
    orig_params_per_group: List[Any] = [group["params"] for group in opt.param_groups]
    if real_id_to_fqn is not None:
        for group in opt.param_groups:
            new_list: List[Any] = []
            for p in group["params"]:
                fqn = real_id_to_fqn.get(id(p))
                if fqn is not None and fqn in params:
                    new_list.append(params[fqn])
                else:
                    new_list.append(p)
            group["params"] = new_list
    else:
        # Legacy fallback: assign all proxy params to every group.
        # Correct only when one opt owns exactly all model params.
        proxy_param_values = list(params.values())
        for group in opt.param_groups:
            group["params"] = proxy_param_values

    # Hyperparameters can be TENSORS held by reference -- a 0-d LR buffer driven
    # by a schedule is the case that motivated this. Those entries point at the
    # pre-reparametrize object, so without rewiring them the optimizer reads a
    # concrete value (baked into the graph as a constant) while the schedule
    # writes the proxy, and the two silently disconnect. Same id -> proxy
    # mapping as params, applied to every non-"params" tensor entry so no
    # single hyperparameter is special-cased.
    orig_hypers: List[Tuple[Any, str, Any]] = []
    if buffers and real_buf_id_to_fqn:
        for group in opt.param_groups:
            for key, val in list(group.items()):
                if key == "params" or not isinstance(val, torch.Tensor):
                    continue
                fqn = real_buf_id_to_fqn.get(id(val))
                if fqn is not None and fqn in buffers:
                    orig_hypers.append((group, key, val))
                    group[key] = buffers[fqn]

    try:
        yield
    finally:
        for group, key, orig_val in orig_hypers:
            group[key] = orig_val
        for group, orig in zip(opt.param_groups, orig_params_per_group):
            group["params"] = orig
        opt.state = orig_states


@contextmanager
def _rematerialize_tensor_attrs(
    holders: Sequence[Any],
    lifted: Dict[str, Any],
    real_id_to_fqn: Dict[int, str],
) -> Generator[None, None, None]:
    """Point captured tensor ATTRIBUTES at the lifted proxies, and restore.

    The object-attribute counterpart of the ``param_groups`` hyperparameter
    rewiring in :func:`_rematerialize_optimizer`, and it exists for the same
    reason: an optimizer is reachable because it is passed into the traced
    step, but an object merely attached to the module is not, so nothing was
    rewiring it.

    ``holders`` is supplied by the caller -- the objects it declares are
    mutating module state from inside the traced step. Nothing is discovered:
    an object reached only through a list, a nested attribute or ``__slots__``
    is not something this can find, and guessing would make the mechanism fire
    or not fire for reasons the caller cannot see.

    Within a holder, only attributes whose CURRENT value is, by identity, one
    of the module's real parameters or buffers are touched, so a holder that
    also carries unrelated tensors is safe to declare.
    """
    saved: List[Tuple[Any, str, Any]] = []
    for holder in holders:
        for attr, value in list(vars(holder).items()):
            if not isinstance(value, torch.Tensor):
                continue
            fqn = real_id_to_fqn.get(id(value))
            if fqn is None or fqn not in lifted:
                continue
            saved.append((holder, attr, value))
            setattr(holder, attr, lifted[fqn])
    if holders and not saved:
        # Declaring a holder whose tensors are not module state is the silent
        # failure this rewiring exists to prevent: the writes would land on
        # pre-trace tensors, the graph would echo the buffers it was handed,
        # and the state would freeze at its seed value with nothing to see.
        raise AssertionError(
            f"none of the {len(holders)} declared tensor-attribute holder(s) "
            f"({[type(h).__name__ for h in holders]}) hold a tensor that is, "
            f"by identity, one of the module's parameters or buffers, so "
            f"nothing was rewired into the graph. Register the tensors they "
            f"write on the module (register_buffer), or drop the holder."
        )
    logger.debug(
        "rematerialized %d captured tensor attribute(s): %s",
        len(saved),
        [(type(h).__name__, a) for h, a, _ in saved],
    )
    try:
        yield
    finally:
        for holder, attr, orig in saved:
            setattr(holder, attr, orig)


aten = torch.ops.aten  # pyre-ignore


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
class CompiledProcessState:
    """Canonical host-process state for explicit-collective FSDP.

    Model tensors are keyed by module FQN. Optimizer tensors remain nested under
    their owning parameter FQN and optimizer state name. The global-shape maps
    contain only leaves that RecIR lays out as process shards.
    """

    model: Dict[str, torch.Tensor]
    optimizer: Dict[str, Dict[str, Any]]
    model_global_shapes: Dict[str, Tuple[int, ...]]
    optimizer_global_shapes: Dict[str, Dict[str, Tuple[int, ...]]]


@dataclass
class _CompiledResult:
    gm: fx.GraphModule
    mod: nn.Module
    opts: List[torch.optim.Optimizer]
    flat_state: List[torch.Tensor]
    state_manager: DeviceStateManager
    process_state: Optional[CompiledProcessState] = None
    # Transforms added by the sub-path (e.g. ``_compile_data_parallel_v2``
    # adds ``SparseEmbeddingTransform``).  Their ``finalize_compiled_graph``
    # is invoked by ``_compile_and_transform`` alongside the caller-supplied
    # ``region_transforms``, in a single consolidated loop.
    internal_transforms: List["RegionTransform"] = field(default_factory=list)


def _compiled_process_state_entries(
    params_and_buffers: Dict[str, Any],
    named_states: Dict[str, Any],
) -> List[Tuple[str, str, Optional[str], torch.Tensor]]:
    entries: List[Tuple[str, str, Optional[str], torch.Tensor]] = []
    for fqn, value in params_and_buffers.items():
        for leaf in pytree.tree_leaves(value):
            if isinstance(leaf, torch.Tensor):
                entries.append(("model", fqn, None, leaf))
    for fqn, state in named_states.items():
        if not isinstance(state, dict):
            for leaf in pytree.tree_leaves(state):
                if isinstance(leaf, torch.Tensor):
                    entries.append(("optimizer", fqn, None, leaf))
            continue
        for state_name, value in state.items():
            stable_state_name = state_name if isinstance(state_name, str) else None
            for leaf in pytree.tree_leaves(value):
                if isinstance(leaf, torch.Tensor):
                    entries.append(("optimizer", fqn, stable_state_name, leaf))
    return entries


def _build_compiled_process_state(
    gm: fx.GraphModule,
    params_and_buffers: Dict[str, Any],
    named_states: Dict[str, Any],
) -> Optional[CompiledProcessState]:
    entries = _compiled_process_state_entries(params_and_buffers, named_states)
    state_placeholders = [
        node
        for node in gm.graph.nodes
        if node.op == "placeholder"
        and node.meta.get("is_state_input", False)
        and isinstance(node.meta.get("val"), torch.Tensor)
    ]
    if not any(get_state_layout(node) is not None for node in state_placeholders):
        return None
    if len(state_placeholders) != len(entries):
        raise RuntimeError(
            "Compiled process-state FQNs do not align with state placeholders: "
            f"fqns={len(entries)}, placeholders={len(state_placeholders)}"
        )

    model_global_shapes: Dict[str, Tuple[int, ...]] = {}
    optimizer_global_shapes: Dict[str, Dict[str, Tuple[int, ...]]] = {}
    for node, (kind, fqn, state_name, tensor) in zip(state_placeholders, entries):
        node_kind = node.meta.get("recir_state_kind", "model")
        node_fqn = node.meta.get(
            "recir_state_fqn",
            node.meta.get("fqn"),
        )
        node_state_name = node.meta.get("recir_state_name")
        if (node_kind, node_fqn, node_state_name) != (kind, fqn, state_name):
            raise RuntimeError(
                "Compiled process-state placeholder metadata does not match its "
                "canonical state entry: "
                f"placeholder={(node_kind, node_fqn, node_state_name)!r}, "
                f"entry={(kind, fqn, state_name)!r}"
            )
        layout = get_state_layout(node)
        if layout is None:
            continue
        if not isinstance(layout, FSDPWeightShardSpec):
            raise RuntimeError(
                f"Unsupported compiled process-state layout for {fqn!r}: {layout!r}"
            )
        if not 0 <= layout.shard_dim < tensor.ndim:
            raise RuntimeError(
                f"Compiled process-state tensor {fqn!r} cannot use {layout!r}"
            )
        global_shape = [int(dim) for dim in tensor.shape]
        global_shape[layout.shard_dim] *= layout.process_world_size
        shape = tuple(global_shape)
        if kind == "model":
            model_global_shapes[fqn] = shape
            continue
        if state_name is None:
            raise RuntimeError(
                "Compiled process optimizer state requires a stable string key: "
                f"parameter={fqn!r}"
            )
        optimizer_global_shapes.setdefault(fqn, {})[state_name] = shape

    model = {
        fqn: value
        for fqn, value in params_and_buffers.items()
        if isinstance(value, torch.Tensor)
    }
    optimizer = {
        fqn: dict(state)
        for fqn, state in named_states.items()
        if isinstance(state, dict)
    }
    return CompiledProcessState(
        model=model,
        optimizer=optimizer,
        model_global_shapes=model_global_shapes,
        optimizer_global_shapes=optimizer_global_shapes,
    )


@dataclass
class _LocalCompiledResult:
    gm: fx.GraphModule
    mod: nn.Module
    opts: List[torch.optim.Optimizer]
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
    _parametrize_fns: Optional[List[Callable[..., torch.Tensor]]] = None,
    _exclude_param_fqns: Optional[set[str]] = None,
    _stateless_func_wrapper: Optional[Callable[..., Any]] = None,
    _tensor_attr_holders: Sequence[Any] = (),
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
        _tensor_attr_holders: Objects attached to the module that mutate module
            state from inside the traced step (e.g. an LR schedule controller
            holding the 0-d LR buffers).  ``_reparametrize_module`` swaps the
            module's own attributes and cannot reach these, so the caller has
            to name them; see :func:`_rematerialize_tensor_attrs`.
        **kwargs: Keyword arguments to the train_step function.
    """
    # 1. Extract nn.Module and Optimizer(s) from args and kwargs
    # FIXME(@mrshenli): support multiple nn.Module instances
    # FIXME(@mrshenli): need to broadcast model to sync parameters
    mod: Optional[nn.Module] = None
    opts: List[torch.optim.Optimizer] = []
    for arg in pytree.arg_tree_leaves(*args, **kwargs):
        if isinstance(arg, nn.Module):
            assert mod is None, "Only support single nn.Module for now"
            mod = arg
        if isinstance(arg, torch.optim.Optimizer):
            opts.append(arg)

    assert mod is not None, "Couldn't find nn.Module instances from the arguments."
    _mod: nn.Module = mod
    _opts: List[torch.optim.Optimizer] = opts

    # 2. Trace stateless version of the train_step
    params = dict(mod.named_parameters(remove_duplicate=False))
    buffers = dict(mod.named_buffers(remove_duplicate=False))

    # Snapshot the id -> FQN map from the REAL (pre-reparametrize)
    # Parameter objects. `opt.param_groups['params']` holds these real
    # Parameter instances (recir's shard_params_by_plan already
    # migrated them to post-shard identity). Once
    # `stateless._reparametrize_module` enters inside stateless_func,
    # `mod.named_parameters()` returns proxies whose ids no longer
    # match the real Params in `opt.param_groups`, so the lookup MUST
    # be done out here to correctly identify each opt's owned params.
    real_id_to_fqn: Dict[int, str] = {id(p): n for n, p in params.items()}
    real_buf_id_to_fqn: Dict[int, str] = {id(b): n for n, b in buffers.items()}

    # Collect per-param optimizer state across ALL opts into one
    # FQN-keyed dict. Each Parameter must belong to at most one opt
    # (otherwise two opts would both step it, double-updating).
    named_states: Dict[str, Any] = {}
    fqn_to_opt_idx: Dict[str, int] = {}
    for opt_idx, opt in enumerate(opts):
        for n, p in params.items():
            if p in opt.state:
                assert n not in named_states, (
                    f"Parameter {n!r} has optimizer state in multiple "
                    f"Optimizers (opts[{fqn_to_opt_idx[n]}] and "
                    f"opts[{opt_idx}]); each param must be owned by "
                    f"exactly one Optimizer."
                )
                named_states[n] = opt.state[p]  # type: ignore[index]
                fqn_to_opt_idx[n] = opt_idx

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
        with ExitStack() as opt_stack:
            opt_stack.enter_context(
                stateless._reparametrize_module(_mod, {**params, **buffers})
            )
            # Same treatment for tensors captured on objects ATTACHED to the
            # module (schedulers, controllers). _reparametrize_module swaps the
            # module's own attributes and cannot reach these, so without this
            # their in-place writes land on pre-trace tensors and never make it
            # into the graph.
            opt_stack.enter_context(
                _rematerialize_tensor_attrs(
                    _tensor_attr_holders,
                    {**params, **buffers},
                    {**real_id_to_fqn, **real_buf_id_to_fqn},
                )
            )
            # Per-opt rematerialize so each opt's state + param_groups
            # are rewritten to point at the proxy params, scoped to
            # the FQNs that opt actually owns. Pass the pre-built
            # real-id -> fqn map (built outside this closure, before
            # reparametrize); querying mod.named_parameters() INSIDE
            # the reparametrize context returns proxies whose ids do
            # not match the real Params in opt.param_groups.
            for opt in _opts:
                opt_stack.enter_context(
                    _rematerialize_optimizer(
                        opt,
                        named_states,
                        params,
                        real_id_to_fqn=real_id_to_fqn,
                        buffers=buffers,
                        real_buf_id_to_fqn=real_buf_id_to_fqn,
                    )
                )
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

    _maybe_dump_fx_graph(gm)

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

    # Tag placeholders for params and buffers with their FQN so any
    # post-compile pass can identify them by name rather than positional
    # index.  Positional indexing is brittle: Override.transform passes
    # that run inside _finalize_graph may add or remove leading
    # placeholders, shifting the params/buffers prefix.  FQN-tagged meta
    # survives recompile, _to_caller_flattened_graph_module, and
    # _dedup_collectives because those passes do not mutate placeholder
    # nodes.
    params_buffers_fqns = list(params.keys()) + list(buffers.keys())
    placeholder_idx = 0
    for node in gm.graph.nodes:
        if node.op != "placeholder":
            continue
        if placeholder_idx < len(params_buffers_fqns):
            node.meta["fqn"] = params_buffers_fqns[placeholder_idx]
        placeholder_idx += 1

    return _LocalCompiledResult(gm, mod, opts, params, buffers, named_states, constants)


def _make_trace_input_transform(
    input_batch_dim: int,
    ratio: float,
    fake_mode: FakeTensorMode,
    dynamic_batch_dim: bool = False,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create a transform that fakeifies inputs and optionally resizes a dimension.

    Every tensor is converted with ``fake_mode``. When ``dynamic_batch_dim``
    is set, ``input_batch_dim`` remains symbolic even when no resize is needed.
    For ratio > 1 (e.g., GSPMD: local → global batch), ``repeat`` expands.
    For ratio < 1 (e.g., v2 XLA path: multi-device local → per-device batch),
    ``narrow`` shrinks. For ratio == 1 only the FakeTensor conversion occurs.
    """

    input_index = 0

    def _to_fake(arg: torch.Tensor) -> torch.Tensor:
        nonlocal input_index
        if not dynamic_batch_dim or arg.ndim <= input_batch_dim:
            return fake_mode.from_tensor(arg)
        symbolic_context = StatelessSymbolicContext(
            dynamic_sizes=[
                DimDynamic.DYNAMIC
                if dimension == input_batch_dim
                else DimDynamic.STATIC
                for dimension in range(arg.ndim)
            ]
        )
        source = LocalSource(f"recir_batch_input_{input_index}")
        input_index += 1
        return fake_mode.from_tensor(
            arg,
            static_shapes=False,
            source=source,
            symbolic_context=symbolic_context,
        )

    def _get_full_batch_arg(arg: torch.Tensor) -> torch.Tensor:
        # Return the fake input untouched in two no-op cases:
        #   * No mini-batch dim to expand/shrink: 0-d scalar tensors (or any
        #     tensor whose rank is <= input_batch_dim). Indexing the missing
        #     dim below would otherwise IndexError.
        #   * ratio == 1 (documented no-op). Skipping here is also what keeps
        #     dynamic shapes alive: falling through to ``int(curr_batch *
        #     ratio)`` would emit an equality guard that specializes the batch
        #     SymInt to its trace-time value, silently freezing the per-batch
        #     (candidate) dimension on the single-device-per-process path.
        if arg.ndim <= input_batch_dim or ratio == 1:
            return _to_fake(arg)
        # Integer expand (ratio > 1 and integer-valued).
        if isinstance(ratio, int) or (
            isinstance(ratio, float) and ratio > 1.0 and ratio.is_integer()
        ):
            fake_arg = _to_fake(arg)
            n = int(ratio)
            repeat_dims = [1] * arg.ndim
            repeat_dims[input_batch_dim] = n
            return fake_arg.repeat(repeat_dims)
        # Integer fraction shrink (ratio = 1/N, N integer >= 2).
        if isinstance(ratio, float) and 0 < ratio < 1.0:
            inv = 1.0 / ratio
            if inv.is_integer():
                n = int(inv)
                fake_arg = _to_fake(arg)
                target_batch = fake_arg.shape[input_batch_dim] // n
                return fake_arg.narrow(input_batch_dim, 0, target_batch)
        # Any other ratio (non-integer and non-1/N) could only be applied via
        # ``int(curr_batch * ratio)``, which specializes the batch SymInt and
        # silently breaks dynamic shapes. No current caller produces such a
        # ratio (HDP/DP -> 1/N; GSPMD -> integer num_processes), so reject it
        # instead of specializing behind the caller's back.
        raise AssertionError(
            f"_make_trace_input_transform: unsupported batch ratio {ratio!r}; "
            "expected 1, a positive integer, or 1/N for integer N >= 2."
        )

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
    gm: fx.GraphModule, num_state_tensors: int, input_batch_dim: int
) -> None:
    tensor_idx = 0
    for node in gm.graph.nodes:
        if node.op != "placeholder":
            continue
        val = node.meta.get("val", None)
        if isinstance(val, torch.Tensor):
            is_state = tensor_idx < num_state_tensors
            is_batch = not is_state and val.dim() > 0
            node.meta["is_state_input"] = is_state
            node.meta["is_batch_input"] = is_batch
            node.meta["is_weight"] = not is_batch
            if is_batch:
                node.meta["batch_dim"] = input_batch_dim
            tensor_idx += 1
        else:
            node.meta["is_state_input"] = False
            node.meta["is_batch_input"] = False
            node.meta["is_weight"] = False


def _compute_state_fqns(structure: Any) -> List[str]:
    """FQN string per flat_state leaf, aligned with ``pytree.tree_leaves(structure)``.

    ``structure`` must be the SAME object passed to ``tree_leaves`` to build
    flat_state (params/buffers dict + named_states), so the paths line up 1:1
    with flat_state. Used to let ``copy_state_to_host`` skip entries by name.
    """
    try:
        leaves_with_path, _ = pytree.tree_flatten_with_path(structure)
        return [pytree.keystr(kp) for kp, _ in leaves_with_path]
    except Exception:
        return []


def _is_fsdp_sharded_leaf(fqn: str, leaf: Any, sharded_keys: set[str]) -> bool:
    return fqn in sharded_keys and isinstance(leaf, torch.Tensor) and leaf.ndim >= 1


def _build_fsdp_sharded_flags(
    params_and_buffers: Dict[str, Any],
    named_states: Dict[str, Any],
    sharded_keys: set[str],
    optimizer_state_only_sharded_keys: frozenset[tuple[str, str]] = frozenset(),
) -> list[bool]:
    """Return per-tensor FSDP flags in the same pytree order as flat_state."""
    params_buffers_flags = {
        fqn: pytree.tree_map(
            lambda leaf, fqn=fqn: _is_fsdp_sharded_leaf(fqn, leaf, sharded_keys),
            value,
        )
        for fqn, value in params_and_buffers.items()
    }
    named_state_flags: Dict[str, Any] = {}
    for fqn, state in named_states.items():
        if not isinstance(state, dict):
            named_state_flags[fqn] = pytree.tree_map(
                lambda leaf, fqn=fqn: _is_fsdp_sharded_leaf(fqn, leaf, sharded_keys),
                state,
            )
            continue
        state_flags: Dict[str, Any] = {}
        for state_key, state_value in state.items():
            state_flags[state_key] = pytree.tree_map(
                lambda leaf, fqn=fqn, state_key=state_key: (
                    _is_fsdp_sharded_leaf(fqn, leaf, sharded_keys)
                    or (
                        (fqn, state_key) in optimizer_state_only_sharded_keys
                        and isinstance(leaf, torch.Tensor)
                        and leaf.ndim >= 1
                    )
                ),
                state_value,
            )
        named_state_flags[fqn] = state_flags
    flat_values, value_spec = pytree.tree_flatten([params_and_buffers, named_states])
    flat_flags, flag_spec = pytree.tree_flatten(
        [params_buffers_flags, named_state_flags]
    )
    if value_spec != flag_spec:
        raise RuntimeError(
            "FSDP flag tree must mirror flat_state TreeSpec: "
            f"values={value_spec}, flags={flag_spec}"
        )
    return [
        bool(flat_flags[index])
        for index, value in enumerate(flat_values)
        if isinstance(value, torch.Tensor)
    ]


def _align_fsdp_sharded_flags(
    base_flat_state: Sequence[Any],
    base_state_flags: Sequence[bool],
    final_flat_state: Sequence[Any],
) -> list[bool]:
    """Carry placement by identity; new transform state is replicated."""
    base_tensors = [
        state_value
        for state_value in base_flat_state
        if isinstance(state_value, torch.Tensor)
    ]
    if len(base_tensors) != len(base_state_flags):
        raise RuntimeError(
            "FSDP flags must mirror base flat_state tensor leaves: "
            f"flags={len(base_state_flags)}, tensors={len(base_tensors)}"
        )
    flags_by_identity: Dict[int, bool] = {}
    for tensor, is_sharded in zip(base_tensors, base_state_flags):
        tensor_identity = id(tensor)
        if (
            tensor_identity in flags_by_identity
            and flags_by_identity[tensor_identity] != is_sharded
        ):
            raise RuntimeError("Aliased state tensor has conflicting FSDP placements")
        flags_by_identity[tensor_identity] = is_sharded
    final_flags: list[bool] = []
    for state_value in final_flat_state:
        if not isinstance(state_value, torch.Tensor):
            continue
        final_flags.append(flags_by_identity.get(id(state_value), False))
    return final_flags


def _annotate_fsdp_state_input_sharding(
    gm: fx.GraphModule,
    flat_state: Sequence[Any],
    state_flags: Sequence[bool],
) -> None:
    """Attach placement to the state portion of RecIR's runtime input."""
    tensor_placeholders = [
        node
        for node in gm.graph.nodes
        if node.op == "placeholder"
        and isinstance(node.meta.get("val", None), torch.Tensor)
    ]
    untagged_placeholders = [
        node.name for node in tensor_placeholders if "is_state_input" not in node.meta
    ]
    if untagged_placeholders:
        raise RuntimeError(
            "FSDP state input annotation requires is_state_input metadata on every "
            f"tensor placeholder; missing={untagged_placeholders!r}"
        )
    state_placeholders = [
        node for node in tensor_placeholders if node.meta["is_state_input"]
    ]
    state_values = [value for value in flat_state if isinstance(value, torch.Tensor)]
    if len(state_values) != len(state_flags):
        raise RuntimeError(
            "FSDP flags must mirror final flat_state tensor leaves: "
            f"state={len(state_values)}, flags={len(state_flags)}"
        )
    if len(state_placeholders) != len(state_values):
        raise RuntimeError(
            "Compiled graph state inputs must mirror final flat_state: "
            f"inputs={len(state_placeholders)}, state={len(state_values)}"
        )
    for node in tensor_placeholders:
        node.meta["is_fsdp_sharded"] = False
        node.meta.pop("fsdp_shard_dim", None)
    for leaf, state_value, is_sharded in zip(
        state_placeholders,
        state_values,
        state_flags,
    ):
        state_identity = id(state_value)
        prior_identity = leaf.meta.get("recir_state_input_identity")
        if prior_identity is not None and prior_identity != state_identity:
            raise RuntimeError(
                "Compiled graph state input order no longer mirrors flat_state"
            )
        leaf.meta["recir_state_input_identity"] = state_identity
        leaf.meta["is_fsdp_sharded"] = is_sharded
        if is_sharded:
            leaf.meta["fsdp_shard_dim"] = 0


def _annotate_fsdp_state_output_sharding(
    gm: fx.GraphModule,
    flat_state: Sequence[Any],
    state_flags: Sequence[bool],
) -> None:
    """Attach placement to the state portion of RecIR's runtime output."""
    if not state_flags:
        return
    output_node = next(node for node in gm.graph.nodes if node.op == "output")
    graph_outputs = output_node.args[0]
    if not isinstance(graph_outputs, (list, tuple)) or not graph_outputs:
        raise RuntimeError(
            "Compiled graph output must contain a user result followed by state"
        )
    graph_output_leaves = pytree.tree_leaves(graph_outputs)
    if len(graph_output_leaves) < len(flat_state):
        raise RuntimeError(
            "Compiled graph output must contain every final flat_state leaf: "
            f"outputs={len(graph_output_leaves)}, state={len(flat_state)}"
        )
    raw_state_outputs = graph_output_leaves[-len(flat_state) :] if flat_state else []
    state_output_leaves = [
        leaf
        for leaf in raw_state_outputs
        if isinstance(leaf, fx.Node)
        and isinstance(leaf.meta.get("val", None), torch.Tensor)
    ]
    state_values = [value for value in flat_state if isinstance(value, torch.Tensor)]
    if len(state_values) != len(state_flags):
        raise RuntimeError(
            "FSDP flags must mirror final flat_state tensor leaves: "
            f"state={len(state_values)}, flags={len(state_flags)}"
        )
    if len(state_output_leaves) != len(state_flags):
        raise RuntimeError(
            "Compiled graph state outputs must mirror final flat_state: "
            f"outputs={len(state_output_leaves)}, flags={len(state_flags)}"
        )
    for leaf, state_value, is_sharded in zip(
        state_output_leaves,
        state_values,
        state_flags,
    ):
        state_identity = id(state_value)
        prior_identity = leaf.meta.get("recir_state_output_identity")
        if prior_identity is not None and prior_identity != state_identity:
            raise RuntimeError(
                "Compiled graph state output order no longer mirrors flat_state"
            )
        leaf.meta["recir_state_output_identity"] = state_identity
        leaf.meta["is_fsdp_sharded_output"] = is_sharded


def _finalize_graph(
    gm: fx.GraphModule,
    mod: nn.Module,
    opts: List[torch.optim.Optimizer],
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
    _tag_batch_inputs(
        gm,
        sum(isinstance(value, torch.Tensor) for value in flat_state),
        input_batch_dim,
    )
    # Propagate batch-input meta through ops so intermediate FX nodes that
    # become subgraph boundaries (after segment_and_lower) inherit
    # is_batch_input / batch_dim. Without this, downstream subgraph
    # placeholders default to replicated and JAX sees per-process shape
    propagate_batch_input_meta(gm)
    result = _CompiledResult(gm, mod, opts, flat_state, DeviceStateManager())
    result.state_manager.state_fqns = _compute_state_fqns(
        [params_and_buffers, named_states]
    )
    return result


@dataclass
class _RegionDiscovery:
    """Result of `_discover_for_region_transforms`.

    Holds the called-module map (used to find forward roots) built by running
    the train step under FakeTensorMode hooks.
    """

    called_modules: OrderedDict[str, Tuple[nn.Module, Tuple[Any, ...], Dict[str, Any]]]


def _discover_for_region_transforms(
    transforms: List[RegionTransform],
    func: Callable[..., Any],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    *,
    mod: Optional[nn.Module] = None,
    fake_mode: Optional[FakeTensorMode] = None,
) -> Optional[_RegionDiscovery]:
    """Run `_discover_called_modules` to find the forward roots.

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
    # Inputs may already be FakeTensors produced by the trace's batch
    # transform. Reuse that mode when supplied; FakeTensorMode does not permit
    # tensors allocated by a different active mode.
    discovery_mode = fake_mode or FakeTensorMode(
        allow_non_fake_inputs=True, shape_env=ShapeEnv()
    )
    shape_env = discovery_mode.shape_env
    ignore_fresh_symbols = (
        shape_env.ignore_fresh_unbacked_symbols()
        if shape_env is not None
        else nullcontext()
    )
    with _enable_compile(), ignore_fresh_symbols:
        called_modules = _discover_called_modules(
            mod,
            discovery_mode,
            func,
            *args,
            **kwargs,
        )
    return _RegionDiscovery(called_modules=called_modules)


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
    default_roots = {
        fqn: (root_mod, root_fake_args, root_fake_kwargs)
        for fqn, root_mod, root_fake_args, root_fake_kwargs in _find_forward_roots(
            discovery.called_modules
        )
    }
    transforms_by_root: OrderedDict[str, list[RegionTransform]] = OrderedDict()
    root_data = dict(default_roots)
    for transform in transforms:
        selected = transform.select_forward_root_fqns(discovery.called_modules)
        root_fqns = default_roots.keys() if selected is None else selected
        for fqn in root_fqns:
            if fqn not in discovery.called_modules:
                raise ValueError(
                    f"Region transform selected an uncalled module FQN: {fqn!r}."
                )
            root_data[fqn] = discovery.called_modules[fqn]
            transforms_by_root.setdefault(fqn, []).append(transform)

    for fqn in sorted(
        transforms_by_root, key=lambda name: name.count("."), reverse=True
    ):
        root_mod, root_fake_args, root_fake_kwargs = root_data[fqn]
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
                transforms_by_root[fqn],
                fake_mode,
                _parametrize_cm_factories=cm_factories,
                fake_kwargs=root_fake_kwargs,
            )
        )


def _compile_and_transform(
    func: Callable[..., Any],
    parallel_mode: ParallelMode,
    *args: Any,
    region_transforms: Optional[List[RegionTransform]] = None,
    _tensor_attr_holders: Sequence[Any] = (),
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
            _tensor_attr_holders=_tensor_attr_holders,
            **kwargs,
        )
    elif isinstance(parallel_mode, (DataParallel, HybridParallel)):
        result = _compile_data_parallel_v2(
            func,
            parallel_mode,
            *args,
            region_transforms=region_transforms,
            _tensor_attr_holders=_tensor_attr_holders,
            **kwargs,
        )
    else:
        raise TypeError(
            f"Unsupported parallel_mode type: {type(parallel_mode).__name__}"
        )
    for transform in result.internal_transforms:
        transform.finalize_compiled_graph(result.gm)
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
    opts: List[torch.optim.Optimizer]
    params: Dict[str, nn.Parameter]
    buffers: Dict[str, torch.Tensor]
    named_states: Dict[str, Any]
    input_batch_dim: int
    embedding_param_indices: Optional[set[int]] = None
    internal_transforms: List[RegionTransform] = field(default_factory=list)


def _make_hybrid_sparse_transform(
    parallel_mode: HybridParallel,
    embedding_param_fqns: set[str],
    mod: nn.Module,
) -> RegionTransform:
    factory = parallel_mode.sparse_transform_factory
    if factory is not None:
        transform = factory(embedding_param_fqns, parallel_mode.mesh, mod)
        if not isinstance(transform, RegionTransform):
            raise TypeError(
                "HybridParallel sparse_transform_factory must return a "
                f"RegionTransform, got {type(transform).__name__}."
            )
        return transform
    return SparseEmbeddingTransform(
        embedding_param_fqns=embedding_param_fqns,
        mesh=parallel_mode.mesh,
        mod=mod,
        embedding_plan=parallel_mode.embedding_plan,
    )


def _prepare_transforms_after_shard(
    transforms: Sequence[RegionTransform],
    mod: nn.Module,
    opts: Sequence[torch.optim.Optimizer],
) -> None:
    for transform in transforms:
        transform.prepare_after_shard(mod, opts)


def _bind_transform_signatures(
    transforms: Sequence[RegionTransform],
    callable_: Callable[..., Any],
) -> None:
    for transform in transforms:
        bind = getattr(transform, "bind_signature", None)
        if bind is not None:
            bind(callable_)


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
    _parametrize_fns: Optional[Sequence[Callable[..., torch.Tensor]]] = None,
    _tensor_attr_holders: Sequence[Any] = (),
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
        _tensor_attr_holders: objects attached to the module that mutate module
            state from inside the traced step (e.g. an LR schedule controller
            holding the 0-d LR buffers).  See
            :func:`_rematerialize_tensor_attrs`.
        **kwargs: keyword arguments to ``callable_``.

    Returns:
        ``(gm, flat_state)``.
    """
    region_transforms_list = list(region_transforms)

    # Pre-extract mod (and all opts for the DP/HybridParallel branch's
    # pre-trace ``shard_params`` call); ``_local_compile`` re-extracts both
    # itself. Multi-opt setups (sparse + dense partition) pass >1 Optimizer.
    mod: Optional[nn.Module] = None
    opts: List[torch.optim.Optimizer] = []
    for arg in pytree.arg_tree_leaves(*args, **kwargs):
        if isinstance(arg, nn.Module) and mod is None:
            mod = arg
        if isinstance(arg, torch.optim.Optimizer):
            opts.append(arg)
    assert mod is not None, "recir.trace requires an nn.Module in the arguments"
    _mod: nn.Module = mod

    # Dispatch on parallel_mode flavor for the trace setup.
    embedding_param_indices: Optional[set[int]] = None
    internal_transforms: list[RegionTransform] = []
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
        regather_transform: Optional[RegatherTransform] = None
        if dp_style == DataParallelStyle.FULLY_SHARD:
            # Per-op regather skips embedding params (owned by the row-wise
            # rewrite) and any param already covered by a RecomputeTransform;
            # RegatherTransform owns that policy (keeps api.py transform-blind).
            regather_skip = RegatherTransform.skip_fqns_for_recompute(
                _mod, region_transforms_list, base_skip=embedding_param_fqns
            )
            regather_transform = RegatherTransform(
                skip_param_fqns=regather_skip or None
            )
            internal_transforms.append(regather_transform)
        if isinstance(parallel_mode, HybridParallel) and embedding_param_fqns:
            internal_transforms.append(
                _make_hybrid_sparse_transform(parallel_mode, embedding_param_fqns, _mod)
            )
        dp_transforms.extend(internal_transforms)
        if region_transforms_list:
            dp_transforms.extend(region_transforms_list)
        _bind_transform_signatures(dp_transforms, callable_)

        # Extract params_and_buffers BEFORE shard_params so the FQN order
        # matches the placeholder ordering ``_local_compile`` will emit.
        # Embedding indices computed from this snapshot map to placeholder
        # positions in the traced graph.
        params_and_buffers: Dict[str, Any] = {
            **dict(_mod.named_parameters(remove_duplicate=False)),
            **dict(_mod.named_buffers(remove_duplicate=False)),
        }
        parameter_fqns = set(dict(_mod.named_parameters(remove_duplicate=False)))
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

        # Convert every input through the shared FakeTensorMode and resize it
        # when one process owns multiple devices. HybridParallel currently has
        # one process per GPU, so its ratio is 1: no resize occurs, but this
        # call still keeps input_batch_dim symbolic for candidate/jagged inputs.
        input_transform = _make_trace_input_transform(
            parallel_mode.input_batch_dim,
            ratio=1 / (world_size // parallel_mode.num_processes),
            fake_mode=fake_mode,
            dynamic_batch_dim=isinstance(parallel_mode, HybridParallel),
        )
        args = pytree.tree_map_only(torch.Tensor, input_transform, args)
        kwargs = pytree.tree_map_only(torch.Tensor, input_transform, kwargs)
        discovery = _discover_for_region_transforms(
            dp_transforms,
            callable_,
            args,
            kwargs,
            mod=_mod,
            fake_mode=fake_mode,
        )

        # Shard params on the module and optimizer states.  Mutates the dict
        # and the module in-place; also updates each ``opt.state`` so that
        # parameter -> state mappings remain valid post-shard.
        sharded_keys = parallel_mode.shard_params(_mod, params_and_buffers, opts)
        sharded_param_fqns: Optional[set[str]] = None
        replicated_param_fqns: set[str] = set()
        if (
            isinstance(parallel_mode, DataParallel)
            and dp_style == DataParallelStyle.FULLY_SHARD
        ):
            sharded_param_fqns = parameter_fqns & sharded_keys
            replicated_param_fqns = parameter_fqns - sharded_keys
            if regather_transform is not None:
                regather_transform.skip_param_fqns.update(replicated_param_fqns)
        _prepare_transforms_after_shard(dp_transforms, _mod, opts)

        # Apply region transforms AFTER sharding, with parametrization.  The
        # traced forward includes allgather ops from ``collectives_insertion``
        # so each per-parameter region encompasses allgather + chain +
        # compute.  Embedding params (if any) are excluded from collectives
        # — they use AllToAll inserted post-trace, not allgather.
        collectives_fn = collectives_insertion(dp_style, mesh)
        parametrize_kwargs: Dict[str, Any] = {}
        if sharded_param_fqns is not None:
            parametrize_kwargs["include_param_fqns"] = sharded_param_fqns
        elif embedding_param_fqns:
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
            # Whether the OUTER trace must parametrize dense params with
            # ``collectives_fn`` depends on the dense style, NOT merely on
            # whether region transforms exist:
            #   * FULLY_SHARD always adds ``RegatherTransform``, which bakes the
            #     dense allgather collectives into its ``gm_fwd``; re-applying
            #     here would double-allgather, so pass [].
            #   * REPLICATE has NO dense region transform (HybridParallel only
            #     adds the embedding-scoped ``SparseEmbeddingTransform``), so the
            #     dense-grad all_reduce must be inserted by parametrizing dense
            #     params in the outer trace. Embedding params are excluded so
            #     they keep their all_to_all path (``None`` when no embeddings).
            if dp_style == DataParallelStyle.FULLY_SHARD:
                _outer_parametrize_fns: List[Callable[..., torch.Tensor]] = []
                _outer_exclude_fqns: Optional[set[str]] = None
            else:
                _outer_parametrize_fns = [collectives_fn]
                _outer_exclude_fqns = embedding_param_fqns
            # User-supplied parametrize fns (e.g. QAT fake-quant) are
            # appended AFTER collectives so the chain is
            # ``user_fn(collectives_fn(raw))`` -- collectives run first so
            # QAT sees the gathered full-precision weight.
            if _parametrize_fns:
                _outer_parametrize_fns = [
                    *_outer_parametrize_fns,
                    *_parametrize_fns,
                ]
            if replicated_param_fqns:
                region_stack.enter_context(
                    parametrize_params(
                        _mod,
                        collectives_insertion(DataParallelStyle.REPLICATE, mesh),
                        include_param_fqns=replicated_param_fqns,
                    )
                )
            local_result = _local_compile(
                callable_,
                *args,
                fake_mode=fake_mode,
                _parametrize_fns=_outer_parametrize_fns,
                _exclude_param_fqns=_outer_exclude_fqns,
                _tensor_attr_holders=_tensor_attr_holders,
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
        _bind_transform_signatures(region_transforms_list, callable_)
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
        trace_batch_ratio = _resolve_trace_batch_ratio(parallel_mode)
        if trace_batch_ratio is not None and trace_batch_ratio != 1:
            input_transform = _make_trace_input_transform(
                input_batch_dim,
                ratio=trace_batch_ratio,
                fake_mode=fake_mode,
            )
            args = pytree.tree_map_only(torch.Tensor, input_transform, args)
            kwargs = pytree.tree_map_only(torch.Tensor, input_transform, kwargs)

        discovery = _discover_for_region_transforms(
            region_transforms_list,
            callable_,
            args,
            kwargs,
            mod=_mod,
            fake_mode=fake_mode,
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
                _parametrize_fns=list(_parametrize_fns) if _parametrize_fns else None,
                _tensor_attr_holders=_tensor_attr_holders,
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
    # Tensor placeholders for params, buffers, and optimizer state precede user
    # inputs; non-tensor state leaves do not occupy tensor-placeholder slots.
    _tag_batch_inputs(
        local_result.gm,
        sum(isinstance(value, torch.Tensor) for value in flat_state),
        input_batch_dim,
    )
    # Propagate batch-input meta through ops so intermediate FX nodes that
    # become subgraph boundaries (after later partition/lower) inherit
    # is_batch_input / batch_dim — mirrors the legacy ``_finalize_graph`` flow.
    propagate_batch_input_meta(local_result.gm)
    ctx = _TraceContext(
        mod=local_result.mod,
        opts=local_result.opts,
        params=local_result.params,
        buffers=local_result.buffers,
        named_states=local_result.named_states,
        input_batch_dim=input_batch_dim,
        embedding_param_indices=embedding_param_indices,
        internal_transforms=internal_transforms,
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
    pipeline_config: Optional[PipelineConfig] = None,
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
      now-sharded module.  For DP/FSDP/Hybrid the graph is already
      parametrized at trace time; GSPMD needs no DTensor partition pass.
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
    #     decomposed AllToAll ops via :class:`SparseEmbeddingTransform`.
    #   * :class:`GSPMDParallel`: ``shard_params`` is deferred to this stage
    #     because the GSPMD lowering inflates placeholders to the global
    #     batch post-trace; sharding before the trace would leave the gm
    #     with per-shard literals that conflict with the inflated
    #     placeholders.  GSPMD needs no DTensor partition pass, so after
    #     sharding we scale + erase the autograd-emitted ``_spmd::tag_grad``
    #     markers so XLA lowering can proceed.
    sharded_keys: set[str] = set()
    optimizer_state_only_sharded_keys: frozenset[tuple[str, str]] = frozenset()
    fsdp_base_state: Optional[Tuple[Any, ...]] = None
    fsdp_state_flags: Optional[list[bool]] = None
    process_state: Optional[CompiledProcessState] = None
    if isinstance(parallel_mode, (DataParallel, HybridParallel)):
        # Trace already sharded; reuse ctx snapshot.
        params_and_buffers: Dict[str, Any] = {**ctx.params, **ctx.buffers}
        named_states: Dict[str, Any] = ctx.named_states

        # Let the parallel mode stamp any state-layout meta it produced during
        # ``shard_params`` onto the graph and substitute its canonical host store
        # into ``params_and_buffers`` / ``named_states`` (both rebuilt into
        # ``flat_state`` below). No-op for modes that produce no layout; keeps
        # this stage layout-agnostic.
        parallel_mode.apply_state_layout(gm, params_and_buffers, named_states)
        process_state = _build_compiled_process_state(
            gm,
            params_and_buffers,
            named_states,
        )

        # Finalize HybridParallel-internal semantic transforms before state is
        # flattened and before IterGraph scheduling. The default sparse
        # transform preserves the existing ATen embedding path; an alternate
        # transform may materialize a TBE-backed sparse graph.
        for transform in ctx.internal_transforms:
            transform.finalize_compiled_graph(gm)
    else:
        # GSPMDParallel.  Mirror the legacy ``_compile_gspmd`` ordering:
        # shard_params first (updates ``opt.state`` and may re-key
        # parameters), then re-extract named_states by walking the
        # now-sharded module.  GSPMD needs no DTensor partition pass.
        params_and_buffers = {**ctx.params, **ctx.buffers}
        sharded_keys = parallel_mode.shard_params(ctx.mod, params_and_buffers, ctx.opts)
        optimizer_state_only_sharded_keys = getattr(
            parallel_mode,
            "optimizer_state_only_sharded_keys",
            frozenset(),
        )
        rebuilt_named_states: Dict[str, Any] = {}
        _opts = ctx.opts
        if any(len(o.state) > 0 for o in _opts):
            # ``shard_params`` may have re-keyed each ``opt.state`` to the new
            # (post-shard) Parameter identities. Resolve the current FQN bindings,
            # then restore the trace-time dict order: pytree flattening is
            # positional, so module order would misalign state leaves with the
            # graph placeholders when optimizer groups interleave in the module.
            for n, p in ctx.mod.named_parameters(remove_duplicate=False):
                for o in _opts:
                    if p in o.state:
                        rebuilt_named_states[n] = o.state[p]  # type: ignore[index]
                        break
            named_states = {
                n: rebuilt_named_states[n]
                for n in ctx.named_states
                if n in rebuilt_named_states
            }
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
            # Either no optimizer was supplied, or every ``opt.state`` is
            # empty (optimizer not stepped yet, or a partial-lower path that
            # never populated state).  In both cases ``ctx.named_states``
            # — captured by ``trace`` — is the authoritative snapshot;
            # fall through so we do not silently discard it.
            named_states = ctx.named_states

        # GSPMD-specific tag_grad fixup.  GSPMD needs no DTensor partition
        # pass, so the autograd-emitted ``_spmd::tag_grad`` markers survive
        # into the lowered graph; they must be scaled (for
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
    if sharded_keys or optimizer_state_only_sharded_keys:
        fsdp_state_flags = _build_fsdp_sharded_flags(
            params_and_buffers,
            named_states,
            sharded_keys,
            optimizer_state_only_sharded_keys,
        )
        fsdp_base_state = tuple(flat_state)
        placeholder_flags = _align_fsdp_sharded_flags(
            fsdp_base_state,
            fsdp_state_flags,
            flat_state,
        )
        _annotate_fsdp_state_input_sharding(gm, flat_state, placeholder_flags)
        _annotate_fsdp_state_output_sharding(gm, flat_state, placeholder_flags)

    # Stage 4 (post-partition is the legacy ordering used by
    # ``_compile_and_transform``): region transforms get a chance to finalize
    # after the graph has been partitioned.
    for transform in region_transforms_list:
        transform.finalize_compiled_graph(gm)

    pipeline = pipeline_config or PipelineConfig.none()
    use_iter_graph = backend.uses_iter_graph and not pipeline.enabled
    # Stage 5.5: wrap before running passes that use IterGraph-only APIs.
    if use_iter_graph:
        # Local import to avoid pulling IterGraphModule into the
        # module-import path for callers that don't use this path.
        from recir.iter_graph_module import IterGraphModule

        gm = cast(fx.GraphModule, IterGraphModule(gm))

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

    if fsdp_base_state is not None and fsdp_state_flags is not None:
        final_state_flags = _align_fsdp_sharded_flags(
            fsdp_base_state,
            fsdp_state_flags,
            flat_state,
        )
        _annotate_fsdp_state_input_sharding(gm, flat_state, final_state_flags)
        _annotate_fsdp_state_output_sharding(gm, flat_state, final_state_flags)

    # Stage 7: freeze cross-iteration rewrites and initialize extra-output state.
    if use_iter_graph:
        from recir.iter_graph_module import IterGraphModule

        cast(IterGraphModule, gm).finalize_setup()

    scheduled_gm: nn.Module = build_pipeline(gm, pipeline)

    # Stage 8: dispatch (backend.dispatch returns an nn.Module — typically
    # the same gm rewritten as a partial-lowered module, but the contract
    # only requires it be a callable nn.Module compatible with the flat-state
    # invocation convention).
    dispatched = backend.dispatch(cast(fx.GraphModule, scheduled_gm))
    # Preserve the gm reference (the dispatched module is typically the gm
    # itself, rewritten in place).  ``_CompiledResult.gm`` is callable, so
    # we use the dispatched module here.
    final_gm = dispatched

    sm = backend.device_state_manager()
    sm.state_fqns = _compute_state_fqns(
        [{**ctx.params, **ctx.buffers}, ctx.named_states]
    )
    return _CompiledResult(
        gm=cast(fx.GraphModule, final_gm),
        mod=ctx.mod,
        opts=ctx.opts,
        flat_state=flat_state,
        state_manager=sm,
        process_state=process_state,
    )


def _compile_via_backend(
    func: Callable[..., Any],
    parallel_mode: ParallelMode,
    backend: "Backend",
    *,
    eager_preproc: Sequence[Callable[[nn.Module], None]],
    region_transforms: Optional[List[RegionTransform]],
    _parametrize_fns: Sequence[Callable[..., torch.Tensor]] = (),
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
        _parametrize_fns=list(_parametrize_fns) if _parametrize_fns else None,
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
    _tensor_attr_holders: Sequence[Any] = (),
    **kwargs: Any,
) -> _CompiledResult:
    """Compile with autograd-based collective insertion (no DTensor partition).

    Handles both ``DataParallel`` (dense-only) and ``HybridParallel``
    (dense + embedding sharding) v2 paths.

    Caller-supplied ``region_transforms`` (e.g. activation checkpointing via
    :class:`RecomputeTransform`) are appended to the path-internal transforms
    (``RegatherTransform`` for FULLY_SHARD, ``SparseEmbeddingTransform`` for
    HybridParallel).  All transforms share a single discovery +
    ``_apply_region_transforms`` pass.
    """
    # Extract mesh and DP style from the parallel mode.
    if isinstance(parallel_mode, HybridParallel):
        mesh = parallel_mode.mesh
        dp_style = parallel_mode.dense_style
    else:
        mesh = parallel_mode.device_mesh
        dp_style = parallel_mode.parallel_style
    world_size: int = parallel_mode.num_ranks

    # 1. Pre-extract mod and ALL Optimizer instances (need originals
    #    for pre-trace sharding; multi-opt setups partition params into
    #    sparse + dense buckets and pass two opts).
    mod: Optional[nn.Module] = None
    opts: List[torch.optim.Optimizer] = []
    for arg in pytree.arg_tree_leaves(*args, **kwargs):
        if isinstance(arg, nn.Module):
            mod = arg
        if isinstance(arg, torch.optim.Optimizer):
            opts.append(arg)
    assert mod is not None
    _mod: nn.Module = mod

    # 2. Resolve embedding param FQNs from the plan (HybridParallel only).
    embedding_param_fqns: Optional[set[str]] = None
    if isinstance(parallel_mode, HybridParallel):
        fqns = parallel_mode.embedding_plan.param_fqns
        embedding_param_fqns = fqns if fqns else None

    # 3. Build transforms list (RegatherTransform for FULLY_SHARD + caller
    #    region_transforms).  Embedding params are excluded from the regather
    #    so that the post-compile sparse-embedding rewrite owns them.
    transforms: list[RegionTransform] = []
    regather_transform: Optional[RegatherTransform] = None
    if dp_style == DataParallelStyle.FULLY_SHARD:
        regather_transform = RegatherTransform(skip_param_fqns=embedding_param_fqns)
        transforms.append(regather_transform)
    if isinstance(parallel_mode, HybridParallel) and embedding_param_fqns:
        transforms.append(
            _make_hybrid_sparse_transform(parallel_mode, embedding_param_fqns, _mod)
        )
    # Internal transforms whose ``finalize_compiled_graph`` must be invoked
    # by this path (the caller-supplied ``region_transforms`` are finalized
    # by ``_compile_and_transform``).
    internal_transforms: list[RegionTransform] = list(transforms)
    _bind_transform_signatures(internal_transforms, func)
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
    parameter_fqns = set(dict(_mod.named_parameters(remove_duplicate=False)))

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

    # Convert every input through the shared FakeTensorMode and resize it when
    # one process owns multiple devices. HybridParallel currently has one
    # process per GPU, so its ratio is 1: no resize occurs, but this call still
    # keeps input_batch_dim symbolic for candidate/jagged inputs.
    input_transform = _make_trace_input_transform(
        parallel_mode.input_batch_dim,
        ratio=1 / (world_size // parallel_mode.num_processes),
        fake_mode=fake_mode,
        dynamic_batch_dim=isinstance(parallel_mode, HybridParallel),
    )
    args = pytree.tree_map_only(torch.Tensor, input_transform, args)
    kwargs = pytree.tree_map_only(torch.Tensor, input_transform, kwargs)
    discovery = _discover_for_region_transforms(
        transforms, func, args, kwargs, mod=_mod, fake_mode=fake_mode
    )

    # 6. Shard params on the module and optimizer states.
    #    shard_params mutates both the dict and the module in-place.
    #    It also updates each opt's state + param_groups so that
    #    parameter -> state mappings remain valid after sparse Parameter
    #    identity swaps.
    sharded_keys = parallel_mode.shard_params(_mod, params_and_buffers, opts)
    sharded_param_fqns: Optional[set[str]] = None
    replicated_param_fqns: set[str] = set()
    if (
        isinstance(parallel_mode, DataParallel)
        and dp_style == DataParallelStyle.FULLY_SHARD
    ):
        sharded_param_fqns = parameter_fqns & sharded_keys
        replicated_param_fqns = parameter_fqns - sharded_keys
        if regather_transform is not None:
            regather_transform.skip_param_fqns.update(replicated_param_fqns)
    _prepare_transforms_after_shard(transforms, _mod, opts)

    # 7. Apply region transforms AFTER sharding, with parametrization.
    #    The traced forward includes allgather ops from collectives_insertion
    #    so each per-parameter region encompasses allgather + chain + compute.
    #    Embedding params (if any) are excluded from collectives — they use
    #    AllToAll (inserted in step 9), not allgather.
    collectives_fn = collectives_insertion(dp_style, mesh)
    parametrize_kwargs: Dict[str, Any] = {}
    if sharded_param_fqns is not None:
        parametrize_kwargs["include_param_fqns"] = sharded_param_fqns
    elif embedding_param_fqns:
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
        #
        #    Whether the OUTER trace must parametrize the dense params with
        #    ``collectives_fn`` depends on the dense style, NOT merely on
        #    whether region transforms exist:
        #      * FULLY_SHARD always adds ``RegatherTransform``, which bakes the
        #        dense allgather collectives into its ``gm_fwd``; re-applying
        #        ``collectives_fn`` here would double-allgather, so pass [].
        #      * REPLICATE has NO dense region transform (HybridParallel only
        #        adds the embedding-scoped ``SparseEmbeddingTransform``), so the
        #        dense-grad all_reduce must be inserted by parametrizing dense
        #        params in the outer trace. Embedding params are excluded so
        #        they keep their all_to_all path (``None`` when no embeddings).
        if dp_style == DataParallelStyle.FULLY_SHARD:
            _outer_parametrize_fns: List[Callable[[torch.Tensor], torch.Tensor]] = []
            _outer_exclude_fqns: Optional[set[str]] = None
        else:
            _outer_parametrize_fns = [collectives_fn]
            _outer_exclude_fqns = embedding_param_fqns
        if replicated_param_fqns:
            region_stack.enter_context(
                parametrize_params(
                    _mod,
                    collectives_insertion(DataParallelStyle.REPLICATE, mesh),
                    include_param_fqns=replicated_param_fqns,
                )
            )
        local_result = _local_compile(
            func,
            *args,
            fake_mode=fake_mode,
            _parametrize_fns=_outer_parametrize_fns,
            _exclude_param_fqns=_outer_exclude_fqns,
            _tensor_attr_holders=_tensor_attr_holders,
            **kwargs,
        )

    # 9. Finalize.
    result = _finalize_graph(
        local_result.gm,
        local_result.mod,
        local_result.opts,
        {**local_result.params, **local_result.buffers},
        local_result.named_states,
        parallel_mode.input_batch_dim,
    )
    result.internal_transforms = internal_transforms
    return result


def _compile_gspmd(
    func: Callable[..., Any],
    parallel_mode: GSPMDParallel,
    *args: Any,
    region_transforms: Optional[List[RegionTransform]] = None,
    _tensor_attr_holders: Sequence[Any] = (),
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
    input_transform = _make_trace_input_transform(
        parallel_mode.input_batch_dim,
        ratio=parallel_mode.num_processes,
        fake_mode=fake_mode,
    )
    args = pytree.tree_map_only(torch.Tensor, input_transform, args)
    kwargs = pytree.tree_map_only(torch.Tensor, input_transform, kwargs)
    discovery = _discover_for_region_transforms(
        transforms, func, args, kwargs, fake_mode=fake_mode
    )
    with ExitStack() as region_stack:
        _enter_region_transform_contexts(region_stack, transforms, discovery, fake_mode)
        local_result = _local_compile(
            func,
            *args,
            fake_mode=fake_mode,
            _stateless_func_wrapper=gradients_tagging,
            _tensor_attr_holders=_tensor_attr_holders,
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
        local_result.mod, params_and_buffers, local_result.opts
    )
    optimizer_state_only_sharded_keys = parallel_mode.optimizer_state_only_sharded_keys
    result = _finalize_graph(
        gm,
        local_result.mod,
        local_result.opts,
        params_and_buffers,
        local_result.named_states,
        input_batch_dim=parallel_mode.input_batch_dim,
        dedup=False,
    )
    result.gm.meta["gspmd_fsdp_axis_size"] = parallel_mode.fsdp_axis_size
    # Set is_fsdp_sharded on placeholder nodes so _jit_gspmd knows
    # which inputs are local shards.
    if sharded_keys or optimizer_state_only_sharded_keys:
        is_sharded = _build_fsdp_sharded_flags(
            params_and_buffers,
            local_result.named_states,
            sharded_keys,
            optimizer_state_only_sharded_keys,
        )
        is_sharded = _align_fsdp_sharded_flags(
            result.flat_state,
            is_sharded,
            result.flat_state,
        )
        _annotate_fsdp_state_input_sharding(
            result.gm,
            result.flat_state,
            is_sharded,
        )
        _annotate_fsdp_state_output_sharding(
            result.gm,
            result.flat_state,
            is_sharded,
        )
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
PREPARE_COMPILED_ATTR = "_recir_prepare_compiled"


def make_compiled_runner(compiled_obj: Any) -> Callable[..., Any]:
    """Wrap a ``_CompiledResult`` into a user-facing callable.

    Builds the standard "prepare_inputs -> gm(*flat_inps) -> commit_step_state"
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
    * splits the epilogue between
      :meth:`compiled_obj.state_manager.commit_step_state` (make the step's
      state outputs the resident state) and :meth:`~materialize_output` (hand
      the caller the step's output, unless it armed ``keep_output_jax`` to pull
      it itself later);
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
    accepts_input_ready_event: bool
    try:
        gm_forward = compiled_obj.gm.forward
        gm_sig = inspect.signature(gm_forward)
        accepts_last_iter = "last_iter" in gm_sig.parameters
        accepts_input_ready_event = "input_ready_event" in gm_sig.parameters
    except (TypeError, ValueError):
        accepts_last_iter = False
        accepts_input_ready_event = False

    def runner(*call_args: Any, **call_kwargs: Any) -> Any:
        last_train_step = call_kwargs.pop("last_train_step", False)
        input_ready_event: Optional[torch.cuda.Event] = call_kwargs.pop(
            "input_ready_event", None
        )
        user_inps = pytree.arg_tree_leaves(*call_args, **call_kwargs)
        flat_inps = compiled_obj.state_manager.prepare_inputs(
            compiled_obj.gm,
            compiled_obj.flat_state,
            user_inps,
            last_train_step,
        )
        with torch.no_grad():
            gm_kwargs: Dict[str, Any] = {}
            if last_train_step and accepts_last_iter:
                # Final iter: pass ``last_iter`` so IterGraphModule can flush.
                gm_kwargs["last_iter"] = last_train_step
            if input_ready_event is not None:
                if accepts_input_ready_event:
                    gm_kwargs["input_ready_event"] = input_ready_event
                else:
                    torch.cuda.current_stream().wait_event(input_ready_event)
            output_all = compiled_obj.gm(*flat_inps, **gm_kwargs)
            if is_pipeline_pending(output_all):
                return output_all
            output = output_all[0]
            updated_flat = pytree.tree_leaves(output_all[1:])
            state_manager = compiled_obj.state_manager
            state_manager.commit_step_state(
                compiled_obj.flat_state, updated_flat, last_train_step
            )
            if state_manager.defers_output(last_train_step):
                return output
            return state_manager.materialize_output(output)

    runner.__dict__[COMPILED_OBJECT_KEY] = compiled_obj
    return runner


def compile(  # noqa: C901 — orchestration: stages 0-9 + legacy DP/GSPMD wiring intentionally kept inline; refactor would obscure the pipeline.
    parallel_mode: Optional[ParallelMode] = None,
    region_transforms: Optional[List[RegionTransform]] = None,
    *,
    backend: Optional["Backend"] = None,
    eager_preproc: Sequence[Callable[[nn.Module], None]] = (),
    _parametrize_fns: Sequence[Callable[..., torch.Tensor]] = (),
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
      entries map 1:1 to the leading placeholders. Parallel modes normalize
      trace inputs via ``_make_trace_input_transform``; depending on the mode,
      this may resize the batch dimension or only fakeify and symbolize it.
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
      ``parallel_mode.shard_params(...)`` (partitioning for DP / FSDP /
      Hybrid is baked in at trace time; GSPMD needs no DTensor pass), then
      re-flattens via :func:`_to_caller_flattened_graph_module` and rebuilds
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
            :meth:`~ParallelMode.partition` rewrites the
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
        def ensure_compiled(
            args: Tuple[Any, ...], kwargs: Dict[str, Any]
        ) -> _CompiledResult:
            compiled_obj = wrapper.__dict__.get(COMPILED_OBJECT_KEY, None)
            if compiled_obj is not None:
                return compiled_obj
            mode: ParallelMode = (
                DataParallel() if parallel_mode is None else parallel_mode
            )
            if backend is not None:
                compiled_obj = _compile_via_backend(
                    func,
                    mode,
                    backend,
                    eager_preproc=eager_preproc,
                    region_transforms=region_transforms,
                    _parametrize_fns=_parametrize_fns,
                    args=args,
                    kwargs=kwargs,
                )
            else:
                if _parametrize_fns:
                    raise ValueError(
                        "recir.compile(_parametrize_fns=...) requires "
                        "backend= to be set (the legacy partition-only "
                        "path does not forward user parametrize fns)."
                    )
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
            return compiled_obj

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_train_step = kwargs.pop("last_train_step", False) if kwargs else False
            compiled_obj = ensure_compiled(args, kwargs)

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
                state_manager = compiled_obj.state_manager
                state_manager.commit_step_state(
                    compiled_obj.flat_state, updated_flat, last_train_step
                )
                if not state_manager.defers_output(last_train_step):
                    output = state_manager.materialize_output(output)

                return output

        def prepare_compiled(*args: Any, **kwargs: Any) -> None:
            if kwargs:
                kwargs.pop("last_train_step", None)
            ensure_compiled(args, kwargs)

        wrapper.__dict__[PREPARE_COMPILED_ATTR] = prepare_compiled
        return wrapper

    return inner


def prepare_compiled(
    compiled_fn: Callable[..., Any], *args: Any, **kwargs: Any
) -> bool:
    """Compile and cache a RecIR callable without executing its graph."""
    prepare = compiled_fn.__dict__.get(PREPARE_COMPILED_ATTR)
    if not callable(prepare):
        return False
    prepare(*args, **kwargs)
    return True


def sync_device_state_to_torch(
    compiled_fn: Callable[..., Any],
    skip_fqns: Optional[Sequence[str]] = None,
) -> None:
    """Synchronize device-resident state back to PyTorch tensors.
    Call before checkpointing or inspecting model parameters.

    ``skip_fqns``: if set, state entries whose FQN contains any of these
    substrings are NOT copied back to host. Use for per-step syncs to skip large
    device-resident tensors (e.g. on-device embedding-table shards, matched by
    their module prefix) whose host copy is only needed at checkpoint time; pass
    ``None`` (default) for a full sync before checkpointing.
    """
    compiled_obj = compiled_fn.__dict__.get(COMPILED_OBJECT_KEY, None)
    if compiled_obj is None:
        return
    if compiled_obj.state_manager.device_state is None:
        return
    compiled_obj.state_manager.copy_state_to_host(
        compiled_obj.flat_state, skip_fqns=skip_fqns
    )


def set_compiled_output_deferred(
    compiled_fn: Callable[..., Any], deferred: bool
) -> bool:
    """Choose whether an XLA compiled call returns device-resident outputs.

    Returns ``False`` before the callable has compiled. Callers can use that
    signal to process the first synchronous result normally, then defer later
    results once the device-state manager exists.
    """
    compiled_obj = compiled_fn.__dict__.get(COMPILED_OBJECT_KEY, None)
    if compiled_obj is None:
        return False
    compiled_obj.state_manager.keep_output_jax = deferred
    return deferred


def materialize_compiled_output(compiled_fn: Callable[..., Any], output: Any) -> Any:
    """Materialize an output retained by ``set_compiled_output_deferred``.

    Before the first compilation, or for a backend whose outputs are already
    Torch tensors, this is a no-op.
    """
    compiled_obj = compiled_fn.__dict__.get(COMPILED_OBJECT_KEY, None)
    if compiled_obj is None:
        return output
    return compiled_obj.state_manager.materialize_output(output)


def get_compiled_process_state(
    compiled_fn: Callable[..., Any],
) -> Optional[CompiledProcessState]:
    """Return the canonical post-layout host state after compilation."""
    compiled_obj = compiled_fn.__dict__.get(COMPILED_OBJECT_KEY, None)
    if compiled_obj is None:
        return None
    process_state = getattr(compiled_obj, "process_state", None)
    if not isinstance(process_state, CompiledProcessState):
        return None
    return process_state


def invalidate_device_state(compiled_fn: Callable[..., Any]) -> None:
    """Discard cached device-resident state so the next call reads PyTorch state."""
    compiled_obj = compiled_fn.__dict__.get(COMPILED_OBJECT_KEY, None)
    if compiled_obj is None:
        return
    compiled_obj.state_manager.device_state = None


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
