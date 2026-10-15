# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Core infrastructure for region transforms: discovery, subgraph extraction,
replacement, and the top-level orchestrator."""

from __future__ import annotations

import logging
import operator
from collections import OrderedDict
from contextlib import contextmanager, ExitStack, nullcontext
from typing import Any, Callable, Generator, Iterable

import torch
import torch.fx as fx
import torch.nn as nn
import torch.utils._pytree as pytree
from functorch import make_fx
from recir.region_transforms.base import RegionTransform
from recir.utils.compile import _enable_compile
from recir.utils.graph import rebuild_graph
from torch._guards import detect_fake_mode
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx._pytree import tree_flatten_spec
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    fetch_sym_proxy,
    get_proxy_mode,
    get_proxy_slot,
    track_tensor_tree,
)
from torch.nn.utils import stateless
from torch.types import py_sym_types

logger: logging.Logger = logging.getLogger(__name__)


@contextmanager
def _forward_pre_hooks(
    mod: nn.Module,
    called: OrderedDict[str, tuple[nn.Module, tuple[Any, ...], dict[str, Any]]],
) -> Generator[None, None, None]:
    """Register forward_pre_hooks on every submodule to capture call info.

    Hooks record ``{fqn: (submodule, input_args, input_kwargs)}`` into
    *called*.  All hooks are removed when the context exits, even on
    exception.
    """
    hooks: list[Any] = []
    for name, submod in mod.named_modules():

        def _hook(
            m: nn.Module,
            input_args: tuple[Any, ...],
            input_kwargs: dict[str, Any],
            name: str = name,
        ) -> None:
            called.setdefault(name, (m, input_args, input_kwargs))

        hooks.append(submod.register_forward_pre_hook(_hook, with_kwargs=True))
    try:
        yield
    finally:
        for h in hooks:
            h.remove()


def _discover_called_modules(
    mod: nn.Module,
    fake_mode: FakeTensorMode,
    func: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> OrderedDict[str, tuple[nn.Module, tuple[Any, ...], dict[str, Any]]]:
    """Run the train_step function under FakeTensorMode + no_grad with hooks.
    Executes ``func(*args, **kwargs)`` (the full train step) so that the
    model receives exactly the arguments the train step passes to it.
    """
    called: OrderedDict[str, tuple[nn.Module, tuple[Any, ...], dict[str, Any]]] = (
        OrderedDict()
    )
    input_device = next(
        (
            value.device
            for value in pytree.arg_tree_leaves(*args, **kwargs)
            if isinstance(value, torch.Tensor) and value.device.type != "meta"
        ),
        None,
    )
    meta_state: dict[str, torch.Tensor] = {}
    if input_device is not None:
        with fake_mode:
            meta_state = {
                name: torch.empty_like(value, device=input_device)
                for name, value in (
                    *mod.named_parameters(remove_duplicate=False),
                    *mod.named_buffers(remove_duplicate=False),
                )
                if value is not None and value.device.type == "meta"
            }

    with (
        _forward_pre_hooks(mod, called),
        fake_mode,
        torch.no_grad(),
        stateless._reparametrize_module(mod, meta_state),
    ):
        try:
            func(*args, **kwargs)
        except Exception:
            # Errors from backward/optimizer are expected since we only
            # need the forward pass.  Log at debug level for diagnostics.
            logger.debug(
                "_discover_called_modules: suppressed exception during "
                "discovery forward (expected — only forward hooks needed)",
                exc_info=True,
            )

    return called


def _build_leaf_apply(orig_fn: Any) -> Any:
    """Preserve user-defined ``autograd.Function`` calls as ``make_fx`` leaves.

    During ``make_fx``, ``autograd.Function.apply()`` is traced *through*: each
    aten op inside ``forward()`` becomes a separate FX node and the custom
    ``backward()`` is lost.  The returned classmethod instead, while tracing:

    1. Disables proxy tracing (``disable_proxy_modes_tracing``).
    2. Runs the original ``apply()`` (``orig_fn``) to get real/fake outputs.
    3. Creates a **single** ``call_function`` proxy node that re-invokes
       ``.apply()`` (so its custom ``backward()`` survives).
    4. Associates outputs with the proxy via ``track_tensor_tree``.
    """

    @classmethod  # type: ignore[misc]
    def _leaf_apply(
        klass: type,  # pyre-fixme[24]
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        proxy_mode = get_proxy_mode()
        if proxy_mode is None:
            return orig_fn(klass, *args, **kwargs)
        # torch.library autograd kernels pass private dispatcher metadata that
        # FX cannot represent. Trace through these implementation wrappers so
        # ProxyTensor records the registered OpOverload instead.
        if klass.__module__ == "torch._library.autograd":
            return orig_fn(klass, *args, **kwargs)

        tracer: Any = proxy_mode.tracer
        # Trace apply as an opaque leaf. Ignore fresh symbols used only inside
        # its forward; symbols exposed through outputs remain tracked.
        fake_mode = detect_fake_mode()
        shape_env = getattr(fake_mode, "shape_env", None)
        ignore_ctx = (
            shape_env.ignore_fresh_unbacked_symbols()
            if shape_env is not None
            else nullcontext()
        )
        with disable_proxy_modes_tracing(), ignore_ctx:
            real_out = orig_fn(klass, *args, **kwargs)

        def _bound_apply(*a: Any, **kw: Any) -> Any:
            return orig_fn(klass, *a, **kw)

        _bound_apply._recir_preserved_autograd_leaf = True  # type: ignore[attr-defined]

        # Tensor args unwrap to their tracked proxy; SymInt/SymFloat/SymBool
        # args unwrap via fetch_sym_proxy (a raw symbolic SymInt has no constant
        # and cannot be embedded by the tracer's create_arg); else pass through.
        _sym_proxy: Callable[[Any], Any] = fetch_sym_proxy(tracer)

        def _to_proxy(x: Any) -> Any:
            if isinstance(x, torch.Tensor):
                return get_proxy_slot(x, tracer, default=x, transform=lambda p: p.proxy)
            if isinstance(x, py_sym_types):
                return _sym_proxy(x)
            return x

        proxy_out = tracer.create_proxy(
            "call_function",
            _bound_apply,
            pytree.tree_map(_to_proxy, args),
            pytree.tree_map(_to_proxy, kwargs),
        )
        return track_tensor_tree(real_out, proxy_out, constant=None, tracer=tracer)

    return _leaf_apply


@contextmanager
def _preserve_all_autograd_function_leaves() -> Generator[None, None, None]:
    """Make user-defined ``autograd.Function`` calls opaque to ``make_fx``.

    ``torch.library`` generated wrappers continue through the dispatcher so
    their registered fake and autograd implementations remain authoritative.
    """
    original_descriptor = torch.autograd.Function.__dict__["apply"]
    orig_fn: Any = getattr(original_descriptor, "__func__", original_descriptor)
    torch.autograd.Function.apply = _build_leaf_apply(orig_fn)  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.autograd.Function.apply = original_descriptor  # type: ignore[assignment]


def _find_forward_roots(
    called_modules: OrderedDict[str, tuple[nn.Module, tuple[Any, ...], dict[str, Any]]],
) -> list[tuple[str, nn.Module, tuple[Any, ...], dict[str, Any]]]:
    """Find forward roots: modules whose forward IS called, but no
    ancestor's forward is called.

    Iterates FQNs sorted by depth (shallowest first).  A module is a
    root if no prefix of its FQN is already a root.  Container modules
    (``nn.ModuleList``, ``nn.Sequential``, etc.) whose ``forward()`` is
    never called are naturally skipped — they are not in
    ``called_modules`` — so their children correctly find the nearest
    called ancestor as their root.
    """
    called_fqns = sorted(called_modules.keys(), key=lambda fqn: fqn.count("."))
    root_fqns: set[str] = set()
    for fqn in called_fqns:
        if fqn == "":
            mod, args, kwargs = called_modules[""]
            return [("", mod, args, kwargs)]
        if any(fqn.startswith(r + ".") for r in root_fqns):
            continue
        root_fqns.add(fqn)
    return [
        (fqn, mod, args, kwargs)
        for fqn, (mod, args, kwargs) in called_modules.items()
        if fqn in root_fqns
    ]


def _topo_sort_nodes(
    region_nodes: set[fx.Node],
    gm: fx.GraphModule,
) -> list[fx.Node]:
    """Return region nodes in graph topological order."""
    return [n for n in gm.graph.nodes if n in region_nodes]


def _compute_region_io(
    sorted_region: list[fx.Node],
) -> tuple[list[fx.Node], list[fx.Node]]:
    """Compute the inputs and outputs of a region."""
    region_set: set[fx.Node] = set(sorted_region)
    region_inputs: list[fx.Node] = []
    region_input_set: set[fx.Node] = set()
    for node in sorted_region:
        for arg in pytree.arg_tree_leaves(*node.args, **node.kwargs):
            if (
                isinstance(arg, fx.Node)
                and arg not in region_set
                and arg not in region_input_set
            ):
                region_inputs.append(arg)
                region_input_set.add(arg)

    region_outputs: list[fx.Node] = []
    region_output_set: set[fx.Node] = set()
    for node in sorted_region:
        has_external_user = False
        for user in node.users:
            if isinstance(user, fx.Node) and user not in region_set:
                # aten.detach nodes inserted by make_fx for parameter
                # tensors are trivial pass-throughs that should not force
                # intermediate chain nodes to become region outputs.
                # Ignoring them prevents DTensor intermediates from being
                # exposed as region outputs, which would cause mixed
                # DTensor/Tensor errors during checkpoint backward.
                if (
                    user.op == "call_function"
                    and user.target is torch.ops.aten.detach.default
                ):
                    continue
                has_external_user = True
                break
        if has_external_user:
            if node not in region_output_set:
                region_outputs.append(node)
                region_output_set.add(node)

    return region_inputs, region_outputs


def _extract_subgraph(
    gm: fx.GraphModule,
    region_nodes: set[fx.Node],
) -> fx.GraphModule:
    """Extract a set of nodes into a standalone sub-GraphModule."""
    sorted_region = _topo_sort_nodes(region_nodes, gm)
    region_inputs, region_outputs = _compute_region_io(sorted_region)
    sub_graph = fx.Graph()
    env: dict[fx.Node, fx.Node] = {}
    for inp in region_inputs:
        ph = sub_graph.placeholder(inp.name)
        ph.meta = dict(inp.meta)
        env[inp] = ph
    for node in sorted_region:

        def _map_arg(x: Any) -> Any:
            if isinstance(x, fx.Node):
                return env[x]
            return x

        new_args = pytree.tree_map(_map_arg, node.args)
        new_kwargs = pytree.tree_map(_map_arg, node.kwargs)
        new_node = sub_graph.call_function(
            node.target,  # pyre-ignore[6]
            new_args,
            new_kwargs,
        )
        new_node.meta = dict(node.meta)
        env[node] = new_node
    if len(region_outputs) == 0:
        sub_graph.output(None)
    elif len(region_outputs) == 1:
        sub_graph.output(env[region_outputs[0]])
    else:
        sub_graph.output(tuple(env[n] for n in region_outputs))
    sub_gm = fx.GraphModule(gm, sub_graph)
    rebuild_graph(sub_gm)
    return sub_gm


def _replace_region(
    gm: fx.GraphModule,
    region_nodes: set[fx.Node],
    wrapped_fn: Callable[..., Any],
) -> None:
    """Replace a set of nodes in ``gm`` with a single call to ``wrapped_fn``."""
    sorted_region = _topo_sort_nodes(region_nodes, gm)
    region_inputs, region_outputs = _compute_region_io(sorted_region)

    # Find insertion point: after the last region input in graph order.
    # get_attr nodes (graph constants like DeviceMesh) may be lazily
    # emitted by make_fx and interleaved between region nodes, so we
    # must insert after ALL inputs to avoid topological violations.
    region_input_set = set(region_inputs)
    insertion_point: fx.Node | None = None
    for node in gm.graph.nodes:
        if node in region_input_set:
            insertion_point = node

    if insertion_point is not None:
        insert_ctx = gm.graph.inserting_after(insertion_point)
    else:
        insert_ctx = gm.graph.inserting_before(sorted_region[0])

    with insert_ctx:
        wrapped_node = gm.graph.call_function(
            wrapped_fn,
            tuple(region_inputs),
        )
        if region_outputs:
            wrapped_node.meta = dict(region_outputs[0].meta)

    if len(region_outputs) == 1:
        region_outputs[0].replace_all_uses_with(
            wrapped_node,
            delete_user_cb=lambda user: user not in region_nodes,
        )
    else:
        for idx, out_node in enumerate(region_outputs):
            with gm.graph.inserting_after(wrapped_node):
                getitem_node = gm.graph.call_function(
                    operator.getitem,
                    (wrapped_node, idx),
                )
                getitem_node.meta = dict(out_node.meta)
            out_node.replace_all_uses_with(
                getitem_node,
                delete_user_cb=lambda user: user not in region_nodes,
            )

    for node in reversed(sorted_region):
        # Before erasing, remove any orphaned aten.detach users that
        # were excluded from region outputs by _compute_region_io but
        # still reference this node.  These are trivial pass-throughs
        # inserted by make_fx that can be safely erased.
        for user in list(node.users):
            if (
                isinstance(user, fx.Node)
                and user not in region_nodes
                and user.op == "call_function"
                and user.target is torch.ops.aten.detach.default
            ):
                user.replace_all_uses_with(node)
                gm.graph.erase_node(user)
        gm.graph.erase_node(node)

    gm.graph.lint()
    gm.recompile()


def extract_subgraph(
    gm: fx.GraphModule,
    nodes: Iterable[fx.Node],
) -> fx.GraphModule:
    """Extract ``nodes`` from ``gm`` into a standalone sub-GraphModule.

    Public wrapper around the internal :func:`_extract_subgraph`.  Accepts any
    iterable of ``fx.Node`` (list, set, tuple, generator) and converts it to
    the node-set shape that the internal helper expects.

    Args:
        gm: the host graph module.
        nodes: the nodes to lift into a new sub-GraphModule.

    Returns:
        A new ``fx.GraphModule`` whose graph contains exactly ``nodes`` plus
        the placeholders / output node required to make it self-contained.
    """
    return _extract_subgraph(gm, set(nodes))


def replace_region(
    gm: fx.GraphModule,
    nodes: Iterable[fx.Node],
    replacement: Callable[..., Any],
) -> None:
    """Replace ``nodes`` in ``gm`` with a single call to ``replacement``.

    Public wrapper around the internal :func:`_replace_region`.  Accepts any
    iterable of ``fx.Node`` and a callable that will be invoked with the
    region inputs as positional arguments.

    Args:
        gm: the host graph module (mutated in place).
        nodes: the nodes to replace.
        replacement: callable invoked with the region inputs.  Its outputs
            replace any external uses of the region's outputs.
    """
    _replace_region(gm, set(nodes), replacement)


@contextmanager
def _apply_region_transforms(
    root_mod: nn.Module,
    fake_args: tuple[Any, ...],
    called_modules: OrderedDict[str, tuple[nn.Module, tuple[Any, ...], dict[str, Any]]],
    transforms: list[RegionTransform],
    fake_mode: FakeTensorMode,
    _parametrize_cm_factories: list[Callable[[], Any]] | None = None,
    _grad_scale: float | None = None,
    fake_kwargs: dict[str, Any] | None = None,
) -> Generator[None, None, None]:
    """
    Trace root_mod.forward, detect regions for each transform,
    rewrite the graph, and temporarily replace root_mod.forward.

    When *_parametrize_cm_factories* is supplied (e.g. from the v2 FSDP path),
    each factory is entered as a context manager during the ``make_fx`` trace
    so that the traced graph includes parametrization ops (allgather, etc.).

    When *_grad_scale* is supplied, the forward output is multiplied by this
    constant to compensate for DTensor's ``Partial("avg")`` gradient placement
    being lost when ``to_local(grad_placements=...)`` is decomposed by make_fx.

    User-defined ``torch.autograd.Function`` calls are opaque during the inner
    trace, preserving their custom backward. ``torch.library`` ops retain their
    registered fake and autograd behavior.

    When *fake_kwargs* is supplied, keyword arguments captured during
    discovery are flattened into positional args for ``make_fx`` tracing
    (which only supports positional args) and reconstructed when calling
    ``root_mod.forward``.
    """
    if not transforms:
        yield
        return

    _fake_kwargs = fake_kwargs or {}

    def _chain_inputs(
        args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        for t in transforms:
            args, kwargs = t.transform_inputs(args, kwargs)
        return args, kwargs

    # Only spin up the inner ``make_fx`` trace when at least one transform
    # actually overrides it.
    needs_inner_trace = any(
        type(t).transform_graph is not RegionTransform.transform_graph
        for t in transforms
    )

    original_forward = root_mod.forward

    if not needs_inner_trace:
        # pyre-fixme[53]
        def _patched_forward_inputs_only(*args: Any, **kw: Any) -> Any:
            new_args, new_kw = _chain_inputs(args, kw)
            return original_forward(*new_args, **new_kw)

        root_mod.forward = _patched_forward_inputs_only  # type: ignore[method-assign]
        try:
            yield
        finally:
            root_mod.forward = original_forward  # type: ignore[method-assign]
        return

    # Apply input transforms to fake inputs FIRST so the inner trace
    # and the outer trace will see same shapes
    with fake_mode:
        padded_fake_args, padded_fake_kwargs = _chain_inputs(fake_args, _fake_kwargs)

    params = dict(root_mod.named_parameters(remove_duplicate=False))
    buffers = dict(root_mod.named_buffers(remove_duplicate=False))
    traced_state_names = (*params.keys(), *buffers.keys())
    _padded_arg_leaves, _padded_arg_spec = pytree.tree_flatten(padded_fake_args)
    _padded_kwarg_leaves, _padded_kwarg_spec = pytree.tree_flatten(padded_fake_kwargs)
    _n_pos_arg_leaves: int = len(_padded_arg_leaves)

    # pyre-ignore
    def _stateless_forward(
        params_and_buffers: dict[str, Any],
        *all_args: Any,
    ) -> Any:
        pos_args = pytree.tree_unflatten(
            list(all_args[:_n_pos_arg_leaves]), _padded_arg_spec
        )
        kw_dict: dict[str, Any] = (
            pytree.tree_unflatten(
                list(all_args[_n_pos_arg_leaves:]), _padded_kwarg_spec
            )
            if _padded_kwarg_leaves
            else {}
        )
        with stateless._reparametrize_module(root_mod, params_and_buffers):
            with ExitStack() as pstack:
                for factory in _parametrize_cm_factories or []:
                    pstack.enter_context(factory())
                out = root_mod(*pos_args, **kw_dict)
                if _grad_scale is not None:
                    out = out * _grad_scale
                return out

    # ``record_module_stack=True`` makes make_fx use ``_ModuleStackTracer``
    # which populates ``node.meta["nn_module_stack"]`` (used by FQN-based
    # transforms like RecomputeTransform).  ``_ModuleStackTracer`` requires
    # ``_orig_mod`` on the traced callable to identify the scope root.
    _stateless_forward._orig_mod = root_mod  # type: ignore[attr-defined]
    with (
        fake_mode,
        _enable_compile(),
        # Preserve ALL autograd.Functions as opaque leaves so the inner trace
        # keeps their custom backward (decomposing would lose it).
        _preserve_all_autograd_function_leaves(),
    ):
        _kwarg_leaves_refaked = tuple(
            torch.empty(v.shape, dtype=v.dtype, device=v.device)
            if isinstance(v, torch.Tensor)
            else v
            for v in _padded_kwarg_leaves
        )
        gm_fwd = make_fx(
            _stateless_forward,
            tracing_mode="fake",
            _allow_non_fake_inputs=True,
            record_module_stack=True,
        )({**params, **buffers}, *_padded_arg_leaves, *_kwarg_leaves_refaked)

    state_placeholders = [
        node for node in gm_fwd.graph.nodes if node.op == "placeholder"
    ][: len(traced_state_names)]
    if len(state_placeholders) != len(traced_state_names):
        raise RuntimeError(
            "Region trace did not preserve all parameter and buffer placeholders."
        )
    for node, fqn in zip(state_placeholders, traced_state_names):
        node.meta["fqn"] = fqn

    for transform in transforms:
        transform.transform_graph(gm_fwd, called_modules)

    def _patched_forward(*args: Any, **kw: Any) -> Any:  # pyre-fixme[53]
        new_args, new_kw = _chain_inputs(args, kw)
        current_state = {
            **dict(root_mod.named_parameters(remove_duplicate=False)),
            **dict(root_mod.named_buffers(remove_duplicate=False)),
        }
        params_and_buffers = {name: current_state[name] for name in traced_state_names}
        arg_leaves = tree_flatten_spec(new_args, _padded_arg_spec)
        kw_leaves = tree_flatten_spec(new_kw, _padded_kwarg_spec) if new_kw else []
        result = gm_fwd(params_and_buffers, *arg_leaves, *kw_leaves)
        return result

    root_mod.forward = _patched_forward  # type: ignore[method-assign]
    try:
        yield
    finally:
        root_mod.forward = original_forward  # type: ignore[method-assign]
