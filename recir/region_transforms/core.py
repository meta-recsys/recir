# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Core infrastructure for region transforms: discovery, subgraph extraction,
replacement, and the top-level orchestrator."""

from __future__ import annotations

import logging
import operator
from collections import OrderedDict
from contextlib import contextmanager, ExitStack
from typing import Any, Callable, Generator

import torch
import torch.fx as fx
import torch.nn as nn
import torch.utils._pytree as pytree
from functorch import make_fx
from recir.graph_utils import rebuild_graph
from recir.region_transforms.base import RegionTransform
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    get_proxy_mode,
    get_proxy_slot,
    track_tensor_tree,
)
from torch.nn.utils import stateless

logger: logging.Logger = logging.getLogger(__name__)


@contextmanager
def _forward_pre_hooks(
    mod: nn.Module,
    called: OrderedDict[str, tuple[nn.Module, tuple[Any, ...]]],
) -> Generator[None, None, None]:
    """Register forward_pre_hooks on every submodule to capture call info.

    Hooks record ``{fqn: (submodule, input_args)}`` into *called*.
    All hooks are removed when the context exits, even on exception.
    """
    hooks: list[Any] = []
    for name, submod in mod.named_modules():

        def _hook(
            m: nn.Module,
            input_args: tuple[Any, ...],
            name: str = name,
        ) -> None:
            called.setdefault(name, (m, input_args))

        hooks.append(submod.register_forward_pre_hook(_hook))
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
) -> OrderedDict[str, tuple[nn.Module, tuple[Any, ...]]]:
    """Run the train_step function under FakeTensorMode + no_grad with hooks.
    Executes ``func(*args, **kwargs)`` (the full train step) so that the
    model receives exactly the arguments the train step passes to it.
    """
    called: OrderedDict[str, tuple[nn.Module, tuple[Any, ...]]] = OrderedDict()

    with (
        _forward_pre_hooks(mod, called),
        fake_mode,
        torch.no_grad(),
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


def _discover_autograd_functions(
    mod: nn.Module,
    fake_mode: FakeTensorMode,
    func: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> set[type]:  # pyre-fixme[24]  # pyre-ignore[11]
    """Discover which ``torch.autograd.Function`` subclasses are invoked.

    Runs the train step under FakeTensorMode + no_grad with a temporary
    monkey-patch on ``torch.autograd.Function.apply`` that records the
    concrete subclass each time ``.apply()`` is called.

    DTensor's internal AFs (``_ToTorchTensor``, ``_FromTorchTensor``,
    ``Redistribute``) are always included in the returned set so their
    custom backward (reduce_scatter, redistribute) survives the inner
    ``make_fx`` trace even if they are not called during the unsharded
    discovery forward (they only appear after FSDP parametrization).

    Returns a set of ``autograd.Function`` subclass types.
    """
    from torch.distributed.tensor._api import _FromTorchTensor, _ToTorchTensor
    from torch.distributed.tensor._redistribute import Redistribute

    discovered: set[type] = set()  # pyre-fixme[24]  # pyre-ignore[11]
    # Always include DTensor's internal AFs — they are not called during
    # the unsharded discovery forward but must be preserved as opaque
    # leaf ops during the inner make_fx trace (which runs post-sharding
    # with DTensor parametrization).
    discovered.update({_ToTorchTensor, _FromTorchTensor, Redistribute})
    # Save the actual classmethod descriptor from __dict__, not the bound
    # method produced by attribute access.
    original_descriptor = torch.autograd.Function.__dict__["apply"]
    original_fn: Any = torch.autograd.Function.apply.__func__  # type: ignore[attr-defined]

    @classmethod  # type: ignore[misc]
    def _recording_apply(cls: type, *a: Any, **kw: Any) -> Any:  # pyre-fixme[24,53]
        discovered.add(cls)
        return original_fn(cls, *a, **kw)

    torch.autograd.Function.apply = _recording_apply  # type: ignore[assignment]
    try:
        with fake_mode, torch.no_grad():
            try:
                func(*args, **kwargs)
            except Exception:
                # Errors from backward/optimizer are expected since we only
                # need the forward pass.  Log at debug level for diagnostics.
                logger.debug(
                    "_discover_autograd_functions: suppressed exception during "
                    "discovery forward (expected — only forward hooks needed)",
                    exc_info=True,
                )
    finally:
        torch.autograd.Function.apply = original_descriptor  # type: ignore[assignment]

    return discovered


@contextmanager
def _preserve_autograd_function_leaves(
    af_classes: set[type],  # pyre-fixme[24]
) -> Generator[None, None, None]:
    """Make ``autograd.Function`` subclasses opaque to ``make_fx`` tracing.

    During ``make_fx``, ``autograd.Function.apply()`` is traced *through*:
    each aten op inside ``forward()`` becomes a separate FX node, and the
    custom ``backward()`` is lost.  This context manager patches each
    discovered subclass's ``apply`` so that during tracing it:

    1. Disables proxy tracing (``disable_proxy_modes_tracing``).
    2. Runs the original ``apply()`` to get real/fake outputs.
    3. Creates a **single** ``call_function`` proxy node for ``.apply()``.
    4. Associates outputs with the proxy via ``track_tensor_tree``.

    Outside ``make_fx`` (i.e. when ``get_proxy_mode()`` returns ``None``),
    the original ``apply`` is called directly — no overhead.
    """
    if not af_classes:
        yield
        return

    original_applies: dict[type, Any] = {}  # pyre-fixme[24]

    def _make_leaf_apply(
        cls: type,  # pyre-fixme[24]
        orig_apply: Any,
    ) -> Any:
        # Capture the original apply's underlying function so we can call
        # it without going through the descriptor protocol again.
        orig_fn: Any = orig_apply.__func__  # type: ignore[attr-defined]

        # Build a wrapper callable that binds the class.  This avoids
        # passing the class object (FunctionMeta) as an FX graph argument
        # — FX's create_arg cannot serialise metaclass instances.
        def _bound_apply(*args: Any, **kwargs: Any) -> Any:
            return orig_fn(cls, *args, **kwargs)

        @classmethod  # type: ignore[misc]
        def _leaf_apply(
            klass: type,  # pyre-fixme[24]
            *args: Any,
            **kwargs: Any,
        ) -> Any:  # pyre-fixme[53]
            proxy_mode = get_proxy_mode()
            if proxy_mode is None:
                return orig_fn(klass, *args, **kwargs)

            tracer: Any = proxy_mode.tracer
            # Run the real apply with proxy tracing disabled so the
            # individual aten ops inside forward() are NOT recorded.
            with disable_proxy_modes_tracing():
                real_out = orig_fn(klass, *args, **kwargs)

            # Build proxy args: for each tensor arg that the tracer
            # tracks, unwrap its proxy; non-tensor args pass through.
            def _to_proxy(x: Any) -> Any:
                if isinstance(x, torch.Tensor):
                    return get_proxy_slot(
                        x, tracer, default=x, transform=lambda p: p.proxy
                    )
                return x

            proxy_args = pytree.tree_map(_to_proxy, args)
            proxy_kwargs = pytree.tree_map(_to_proxy, kwargs)
            # Create a single opaque node for this .apply() call.
            # Use _bound_apply (which binds cls) as the target so the
            # class object does not appear as a graph argument.
            proxy_out = tracer.create_proxy(
                "call_function",
                _bound_apply,
                proxy_args,
                proxy_kwargs,
            )
            # Associate the real outputs with the proxy node.
            return track_tensor_tree(real_out, proxy_out, constant=None, tracer=tracer)

        return _leaf_apply

    for cls in af_classes:
        original_applies[cls] = cls.__dict__.get("apply", None)
        cls.apply = _make_leaf_apply(cls, cls.apply)  # type: ignore[assignment]

    try:
        yield
    finally:
        for cls, orig in original_applies.items():
            if orig is not None:
                cls.apply = orig  # type: ignore[assignment]
            elif "apply" in cls.__dict__:
                delattr(cls, "apply")


def _find_forward_roots(
    called_modules: OrderedDict[str, tuple[nn.Module, tuple[Any, ...]]],
) -> list[tuple[str, nn.Module, tuple[Any, ...]]]:
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
            mod, args = called_modules[""]
            return [("", mod, args)]
        if any(fqn.startswith(r + ".") for r in root_fqns):
            continue
        root_fqns.add(fqn)
    return [
        (fqn, mod, args)
        for fqn, (mod, args) in called_modules.items()
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


@contextmanager
def _apply_region_transforms(
    root_mod: nn.Module,
    fake_args: tuple[Any, ...],
    called_modules: OrderedDict[str, tuple[nn.Module, tuple[Any, ...]]],
    transforms: list[RegionTransform],
    fake_mode: FakeTensorMode,
    _parametrize_cm_factories: list[Callable[[], Any]] | None = None,
    _grad_scale: float | None = None,
    _autograd_function_leaves: set[type] | None = None,  # pyre-fixme[24]
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

    When *_autograd_function_leaves* is supplied, the listed
    ``torch.autograd.Function`` subclasses are treated as opaque leaf
    operations during the inner ``make_fx`` trace — their ``forward()`` is
    NOT decomposed into individual aten ops, preserving custom ``backward()``
    behavior in the traced graph.  This replaces the ``_grad_scale``
    workaround for specific cases like DTensor's ``to_local`` with
    ``grad_placements``.
    """
    if not transforms:
        yield
        return

    params = dict(root_mod.named_parameters(remove_duplicate=False))
    buffers = dict(root_mod.named_buffers(remove_duplicate=False))

    def _stateless_forward(
        params_and_buffers: dict[str, Any],
        *args: Any,
    ) -> Any:
        with stateless._reparametrize_module(root_mod, params_and_buffers):
            with ExitStack() as pstack:
                for factory in _parametrize_cm_factories or []:
                    pstack.enter_context(factory())
                out = root_mod(*args)
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
        _preserve_autograd_function_leaves(_autograd_function_leaves or set()),
    ):
        gm_fwd = make_fx(
            _stateless_forward,
            tracing_mode="fake",
            _allow_non_fake_inputs=True,
            record_module_stack=True,
        )({**params, **buffers}, *fake_args)

    for transform in transforms:
        regions = transform.detect_regions(gm_fwd, called_modules)
        for region_nodes in regions:
            if not region_nodes:
                continue
            placeholders = [n for n in region_nodes if n.op == "placeholder"]
            if placeholders:
                raise ValueError(
                    f"Region contains placeholder nodes {placeholders}; "
                    f"regions must only contain call_function / call_method "
                    f"nodes. Parameter placeholders should be region inputs, "
                    f"not members."
                )
            region_gm = _extract_subgraph(gm_fwd, region_nodes)
            wrapped_fn = transform.wrap_region(region_gm)
            _replace_region(gm_fwd, region_nodes, wrapped_fn)

    original_forward = root_mod.forward

    def _patched_forward(*args: Any, **kw: Any) -> Any:  # pyre-fixme[53]
        params_and_buffers = {
            **dict(root_mod.named_parameters(remove_duplicate=False)),
            **dict(root_mod.named_buffers(remove_duplicate=False)),
        }
        return gm_fwd(params_and_buffers, *args, **kw)

    root_mod.forward = _patched_forward  # type: ignore[method-assign]
    try:
        yield
    finally:
        root_mod.forward = original_forward  # type: ignore[method-assign]
