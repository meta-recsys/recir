# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Local FX tracing primitives used by RecIR compilation."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager, ExitStack, nullcontext
from copy import copy
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
import torch.distributed._functional_collectives
import torch.fx as fx
import torch.nn as nn
import torch.utils._pytree as pytree
from functorch import make_fx
from recir.decompositions import SPMD_DECOMP_TABLE
from recir.distributed.data_parallel import parametrize_params
from recir.utils.compile import _enable_compile
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.export.unflatten import _assign_attr, _AttrKind
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
    def process_inputs(self, *args: Any) -> Any:
        return args

    def gen_fn_def(self, free_vars, maybe_return_annotation, expanded_def=False):
        return CodeGen.gen_fn_def(self, free_vars, maybe_return_annotation)


def _to_caller_flattened_graph_module(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Move the responsibility of flattening the input arguments from the graph module to the caller.

    Example:

        output = gm(my_struct)

        gm = gm(to_caller_flattened_graph_module)

        output = gm(*pytree.flatten(my_struct)[0])

    """
    codegen = gm._graph._codegen
    if not isinstance(codegen, _PyTreeCodeGen):
        raise TypeError("caller-flattened graphs require PyTree code generation")
    gm._graph._codegen = _PyTreeCodeGenOutputsOnly(
        pytree_info=_PyTreeInfo(
            orig_args=None,  # type: ignore[arg-type]
            in_spec=None,  # type: ignore[arg-type]
            out_spec=codegen.pytree_info.out_spec,
        )
    )
    gm.recompile()
    return gm


@contextmanager
def _rematerialize_optimizer(
    opt: torch.optim.Optimizer,
    named_states: dict[str, Any],
    params: dict[str, nn.Parameter],
    mod: nn.Module | None = None,
    real_id_to_fqn: dict[int, str] | None = None,
    buffers: dict[str, torch.Tensor] | None = None,
    real_buf_id_to_fqn: dict[int, str] | None = None,
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
    orig_params_per_group: list[Any] = [group["params"] for group in opt.param_groups]
    if real_id_to_fqn is not None:
        for group in opt.param_groups:
            new_list: list[Any] = []
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
    orig_hypers: list[tuple[Any, str, Any]] = []
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
    lifted: dict[str, Any],
    real_id_to_fqn: dict[int, str],
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
    saved: list[tuple[Any, str, Any]] = []
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


@dataclass
class _LocalCompiledResult:
    gm: fx.GraphModule
    mod: nn.Module
    opts: list[torch.optim.Optimizer]
    params: dict[str, nn.Parameter]
    buffers: dict[str, torch.Tensor]
    named_states: dict[str, Any]
    # Tensor constants referenced via ``get_attr`` nodes in ``gm`` (e.g. inline
    # tensor literals lifted by ``make_fx``). Keyed by fully-qualified attribute
    # name on ``gm`` so downstream serde / inspection can recover them. These
    # are stored as gm attributes (not placeholders) and registered via
    # ``_AttrKind.CONSTANT`` so they survive ``gm.recompile()`` cleanly.
    constants: dict[str, torch.Tensor]


def _local_compile(
    func: Callable[..., Any],
    *args: Any,
    fake_mode: FakeTensorMode,
    _parametrize_fns: list[Callable[..., torch.Tensor]] | None = None,
    _exclude_param_fqns: set[str] | None = None,
    _stateless_func_wrapper: Callable[..., Any] | None = None,
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
    mod: nn.Module | None = None
    opts: list[torch.optim.Optimizer] = []
    for arg in pytree.arg_tree_leaves(*args, **kwargs):
        if isinstance(arg, nn.Module):
            assert mod is None, "Only support single nn.Module for now"
            mod = arg
        if isinstance(arg, torch.optim.Optimizer):
            opts.append(arg)

    assert mod is not None, "Couldn't find nn.Module instances from the arguments."
    _mod: nn.Module = mod
    _opts: list[torch.optim.Optimizer] = opts

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
    real_id_to_fqn: dict[int, str] = {id(p): n for n, p in params.items()}
    real_buf_id_to_fqn: dict[int, str] = {id(b): n for n, b in buffers.items()}

    # Collect per-param optimizer state across ALL opts into one
    # FQN-keyed dict. Each Parameter must belong to at most one opt
    # (otherwise two opts would both step it, double-updating).
    named_states: dict[str, Any] = {}
    fqn_to_opt_idx: dict[str, int] = {}
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
        params: dict[str, nn.Parameter],
        buffers: dict[str, torch.Tensor],
        named_states: dict[str, Any],
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
    constants: dict[str, torch.Tensor] = {}
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
