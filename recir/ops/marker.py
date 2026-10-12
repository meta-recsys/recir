# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors

"""Boundary-marker custom op for graph composition.

Inserts identity ``recir::mark_boundary`` nodes into the FX graph so that
downstream passes can locate injection / derivation / auxiliary boundaries
without relying on fragile string matching of module names.

Usage:
    from recir.ops.marker import mark_boundary, attach_markers, remove_markers

    handles = attach_markers(model, inject_specs, derive_specs, aux_specs)
    # ... trace / run the model ...
    remove_markers(handles)
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
from torch import fx
from torch.fx.node import has_side_effect


# ---------------------------------------------------------------------------
# Minimal spec stubs (duck-typed; no external imports required)
# ---------------------------------------------------------------------------


@dataclass
class InjectSpec:
    """Lightweight descriptor for an injection point."""

    at: str


@dataclass
class DeriveSpec:
    """Lightweight descriptor for a derivation point."""

    at: str
    input_from: str


@dataclass
class AuxSpec:
    """Lightweight descriptor for an auxiliary output point."""

    at: str


# ---------------------------------------------------------------------------
# Custom op: recir::mark_boundary
# ---------------------------------------------------------------------------


@torch.library.custom_op("recir::mark_boundary", mutates_args=())
def mark_boundary(x: torch.Tensor, name: str) -> torch.Tensor:
    """Identity op that survives ``make_fx`` tracing.

    During real execution the tensor is returned unchanged.  The
    ``register_fake`` implementation ensures ``make_fx(tracing_mode="fake")``
    can propagate shapes through this node.
    """
    return x.clone()


@mark_boundary.register_fake
def _mark_boundary_fake(x: torch.Tensor, name: str) -> torch.Tensor:
    return torch.empty_like(x)


def _mark_boundary_setup_context(ctx, inputs, output):
    # register_autograd requires a setup_context function. This is where
    # you would call ctx.save_for_backward() if the backward needed any
    # saved tensors. Since mark_boundary is identity, the backward needs
    # no saved state — so this is a no-op.
    pass


def _mark_boundary_backward(ctx, grad_output):
    return grad_output, None  # identity backward, None for name


mark_boundary.register_autograd(
    _mark_boundary_backward, setup_context=_mark_boundary_setup_context
)


# Pin mark_boundary against fx eliminate_dead_code. The graph-break dispatch
# (recir/ops/eager_dispatch.py) consumes these boundary markers OFF-graph -- the
# embedding backward fires as a side effect -- so to fx DCE they look dead and
# get pruned before the sparse dispatcher can wire its ``::bwd::`` markers,
# freezing the embedding tables on the compiled path. Registered here, at the
# op's definition site, so importers (eager_dispatch) needn't touch
# ``torch.ops.recir`` at module-import time.
has_side_effect(torch.ops.recir.mark_boundary.default)


# ---------------------------------------------------------------------------
# Hook helpers
# ---------------------------------------------------------------------------


def _make_pre_hook(name: str):
    """Create a forward_pre_hook that marks an input boundary."""

    def _pre_hook(module, args):
        new_arg0 = torch.ops.recir.mark_boundary(args[0], f"{name}::input")
        return (new_arg0,) + args[1:]

    return _pre_hook


def _make_output_hook(name: str):
    """Create a forward_hook that marks an output boundary."""

    def _hook(module, args, output):
        return torch.ops.recir.mark_boundary(output, f"{name}::output")

    return _hook


def _attach_inject_hooks(
    model: torch.nn.Module,
    inject_specs: Sequence,
    handles: List[torch.utils.hooks.RemovableHandle],
) -> None:
    """Register forward_pre_hooks for inject specs."""
    for spec in inject_specs:
        at = spec.at
        fqns = [at] if isinstance(at, str) else list(at)
        for fqn in fqns:
            submodule = model.get_submodule(fqn)
            handle = submodule.register_forward_pre_hook(_make_pre_hook(fqn))
            handles.append(handle)


def _attach_derive_hooks(
    model: torch.nn.Module,
    derive_specs: Sequence,
    handles: List[torch.utils.hooks.RemovableHandle],
) -> None:
    """Register forward_hooks for derive specs whose input_from uses markers."""
    for spec in derive_specs:
        input_from = spec.input_from
        if "::" not in input_from:
            continue
        fqn = input_from.split("::")[0]
        submodule = model.get_submodule(fqn)
        handle = submodule.register_forward_hook(_make_output_hook(fqn))
        handles.append(handle)


def _attach_aux_hooks(
    model: torch.nn.Module,
    aux_specs: Sequence,
    handles: List[torch.utils.hooks.RemovableHandle],
) -> None:
    """Register forward_hooks for aux specs."""
    for spec in aux_specs:
        at = spec.at
        fqns = [at] if isinstance(at, str) else list(at)
        for fqn in fqns:
            submodule = model.get_submodule(fqn)
            handle = submodule.register_forward_hook(_make_output_hook(fqn))
            handles.append(handle)


def attach_markers(
    model: torch.nn.Module,
    inject_specs: Sequence,
    derive_specs: Optional[Sequence] = None,
    aux_specs: Optional[Sequence] = None,
) -> List[torch.utils.hooks.RemovableHandle]:
    """Attach forward hooks that insert ``mark_boundary`` calls.

    Args:
        model: The root module.
        inject_specs: Each spec must have an ``.at`` attribute naming a
            submodule.  A ``forward_pre_hook`` is registered that wraps
            ``args[0]`` with ``mark_boundary(args[0], "{at}::input")``.
        derive_specs: Each spec must have ``.at`` and ``.input_from``
            attributes.  If ``input_from`` contains ``"::"`` (marker-style),
            a ``forward_hook`` is registered on the *source* submodule
            (the part before ``"::"``) that wraps the output with
            ``mark_boundary(output, "{fqn}::output")``.
        aux_specs: Each spec must have an ``.at`` attribute.  A
            ``forward_hook`` wraps the output with
            ``mark_boundary(output, "{at}::output")``.

    Returns:
        A list of hook handles that can be passed to :func:`remove_markers`.
    """
    if derive_specs is None:
        derive_specs = []
    if aux_specs is None:
        aux_specs = []

    handles: List[torch.utils.hooks.RemovableHandle] = []
    _attach_inject_hooks(model, inject_specs, handles)
    _attach_derive_hooks(model, derive_specs, handles)
    _attach_aux_hooks(model, aux_specs, handles)
    return handles


def remove_markers(handles: Sequence) -> None:
    """Remove all hooks previously returned by :func:`attach_markers`."""
    for handle in handles:
        handle.remove()


# ---------------------------------------------------------------------------
# Marker discovery (graph-level)
# ---------------------------------------------------------------------------


def find_markers(gm: fx.GraphModule) -> List[fx.Node]:
    """Locate all ``recir::mark_boundary`` nodes in ``gm``.

    Returns the nodes in graph-iteration order.  Duplicate marker names are
    preserved (each occurrence is returned separately); callers that want a
    name-to-node mapping should build it themselves.

    Args:
        gm: a traced graph module.

    Returns:
        A list of ``fx.Node`` objects whose ``target`` is
        ``torch.ops.recir.mark_boundary.default``.
    """
    return [
        node
        for node in gm.graph.nodes
        if node.op == "call_function"
        and node.target is torch.ops.recir.mark_boundary.default
    ]
