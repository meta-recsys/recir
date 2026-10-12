# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors

"""Graph-break dispatch primitives for orchestrator-style overrides.

Provides a marker-based replacement for the legacy ``override_ops`` module.
Orchestrators plant ``recir.mark_boundary`` nodes during tracing (typically
inside a Dummy nn.Module that stands in for a non-traceable submodule), then
call :func:`replace_marker_with_dispatch` to swap each marker with a graph-
break dispatch call. The dispatch call runs an arbitrary user-supplied fwd
callable at runtime and (optionally) wires a custom backward dispatch that
fires inside the compiled ``train_step`` autograd graph.

Replaces ``recir/override_ops.py`` (deleted in v7.1 C3). The new flow is::

    # Tracing-time: orchestrator plants markers inside the Dummy module.
    x = torch.ops.recir.mark_boundary(x, f"{fqn}::output::0")

    # Post-trace: orchestrator finds the marker and installs the dispatch.
    for node in recir.find_markers(gm):
        if node.args[1] == f"{fqn}::output::0":
            replace_marker_with_dispatch(
                gm, node, fwd_fn=jax_preprocess, num_state_tensors=...,
            )
"""

from __future__ import annotations

import itertools
import operator
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

import recir.ops.marker  # noqa: F401 -- register recir::mark_boundary
import torch
from torch import fx
from torch.fx.node import has_side_effect


# Module-level dispatch registries. Keyed by an opaque (dispatcher-supplied)
# integer id so :func:`_dispatch_fwd` / :func:`_dispatch_bwd` can locate the
# user callables at runtime (FX recompile requires the call targets to be
# importable module-level functions, not closures).
#
# Lifecycle note: these dicts are append-only by design. Each call to
# :func:`install_graph_break_dispatch` allocates a fresh
# ``dispatch_id`` via :func:`_allocate_dispatch_id` and inserts one
# ``(fwd, bwd)`` tuple per id; entries are never removed for the
# lifetime of the process. In practice this is bounded by the number
# of compiled graphs a training/inference process owns (typically
# O(1) — a single train_step per training process or per-checkpoint
# in inference), so the memory growth is negligible. We deliberately
# don't wire a weakref/cleanup hook here because (a) FX graphs
# already hold strong refs to the dispatch_id constants via the
# splice in ``install_graph_break_dispatch``, so a weakref keyed off
# the call_function node would never fire as long as the gm is
# alive; and (b) consumers that recompile in a loop (e.g.
# ``recir.examples``) already share the registry across recompiles
# since the wrapped fwd/bwd closures are stable. If a long-running
# server ever holds >10k compiled graphs and the registries become a
# concern, switch to a WeakValueDictionary keyed on a wrapper class
# that owns the gm reference -- but until then the simplicity is
# worth more than the bounded leak.
_FWD_REGISTRY: Dict[int, Callable[..., Tuple[torch.Tensor, ...]]] = {}
_BWD_REGISTRY: Dict[int, Callable[[torch.Tensor, int], None]] = {}

# ``itertools.count().__next__`` is atomic under the GIL; combined with a
# ``threading.Lock`` around registry writes this lets concurrent compiles
# (e.g. multiple models in parallel threads) safely allocate distinct
# dispatch ids without racing on the counter or stomping each other's
# registry entries.
_dispatch_id_counter: "itertools.count[int]" = itertools.count()
_registry_lock: threading.Lock = threading.Lock()
_gradient_division_lock: threading.Lock = threading.Lock()


def _allocate_dispatch_id() -> int:
    return next(_dispatch_id_counter)


def _dispatch_fwd(inputs: List[torch.Tensor], dispatch_id: int) -> List[torch.Tensor]:
    """Module-level forward dispatch target used by replaced markers."""
    return list(_FWD_REGISTRY[dispatch_id](*inputs))


def _dispatch_bwd(grad: torch.Tensor, dispatch_id: int, grad_idx: int) -> torch.Tensor:
    """Module-level backward dispatch target used by replaced markers."""
    _BWD_REGISTRY[dispatch_id](grad, grad_idx)
    return grad


# Pin the backward dispatch against eliminate_dead_code. Its result is consumed
# OFF-graph -- the embedding update fires as a side effect (_dispatch_bwd ->
# torch.autograd.backward -> FUSED / SparseCore Adagrad), so to fx DCE the node
# looks dead. Without this:
#   (a) an EARLIER dispatcher's install runs eliminate_dead_code and prunes a
#       LATER dispatcher's still-unwired ``::bwd::`` markers (so the sparse-arch
#       backward markers vanish before SparseArchDispatcher.install_graph can
#       wire them -> install sees bwd_markers=0), and
#   (b) the final eliminate_dead_code prunes the _dispatch_bwd node itself
#       (its output feeds only the dead dummy-backward chain).
# Either way the embedding backward never runs on the compiled path and the
# embedding tables stay frozen (the dense arch still learns; embeddings do not).
# Marking _dispatch_bwd impure keeps it through every DCE. The ``mark_boundary``
# op is pinned the same way, but at its definition site (recir/ops/marker.py) so
# this module needn't touch ``torch.ops.recir`` at import time; install still
# explicitly erases each marker it processes, so no stale markers linger.
has_side_effect(_dispatch_bwd)


def _wrap_with_save_for_backward(
    fwd_fn: Callable[..., object],
    disable_torchrec_gradient_division: bool,
) -> Tuple[
    Callable[..., Tuple[torch.Tensor, ...]],
    Callable[[torch.Tensor, int], None],
]:
    """Bridge autograd across a graph break with one combined backward."""
    import torch.utils._pytree as pytree

    state: dict[str, Any] = {"saved": None, "grads": {}}

    def fwd(*inputs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        with torch.enable_grad():
            raw_output = fwd_fn(*inputs)
        flat = [
            t for t in pytree.tree_leaves(raw_output) if isinstance(t, torch.Tensor)
        ]
        state["saved"] = [t for t in flat if t.requires_grad]
        state["grads"] = {}
        return tuple(t.detach() for t in flat)

    def bwd(grad: torch.Tensor, grad_idx: int = 0) -> None:
        saved = state["saved"]
        if saved is None or grad_idx >= len(saved):
            return
        grads = state["grads"]
        grads[grad_idx] = grad.to(saved[grad_idx].device)
        if len(grads) != len(saved):
            return

        ordered_grads = [grads[i] for i in range(len(saved))]
        try:
            if disable_torchrec_gradient_division:
                from torchrec.distributed.comm_ops import (
                    get_gradient_division,
                    set_gradient_division,
                )

                with _gradient_division_lock:
                    previous = get_gradient_division()
                    set_gradient_division(False)
                    try:
                        torch.autograd.backward(saved, ordered_grads)
                    finally:
                        set_gradient_division(previous)
            else:
                torch.autograd.backward(saved, ordered_grads)
        finally:
            state["saved"] = None
            state["grads"] = {}

    return fwd, bwd


def _collect_markers_by_name(gm: fx.GraphModule, marker_name: str) -> List[fx.Node]:
    """Return marker nodes whose second positional arg equals ``marker_name``."""
    out: List[fx.Node] = []
    for node in gm.graph.nodes:
        if (
            node.op == "call_function"
            and node.target is torch.ops.recir.mark_boundary.default
            and len(node.args) >= 2
            and node.args[1] == marker_name
        ):
            out.append(node)
    return out


def _collect_marker_groups(
    gm: fx.GraphModule, marker_prefix: str
) -> Tuple[List[fx.Node], List[fx.Node]]:
    """Return (forward, backward) marker nodes whose names begin with ``marker_prefix``.

    Forward markers carry suffix ``"::output::<idx>"`` or ``"::input::<idx>"``.
    Backward markers (planted via :func:`mark_output_with_bwd`) carry suffix
    ``"::bwd::<idx>"`` and live in the backward sub-section of the FX graph.

    We match against ``marker_prefix + "::"`` rather than ``marker_prefix``
    directly so that two prefixes which share an integer-id prefix relation
    (e.g. ``ns::0`` vs ``ns::01``) cannot silently capture each other's
    markers; every marker name is required to be followed by the ``::``
    sub-key separator.
    """
    boundary = marker_prefix + "::"
    fwd_nodes: List[fx.Node] = []
    bwd_nodes: List[fx.Node] = []
    for node in gm.graph.nodes:
        if (
            node.op != "call_function"
            or node.target is not torch.ops.recir.mark_boundary.default
            or len(node.args) < 2
        ):
            continue
        name = node.args[1]
        if not isinstance(name, str) or not name.startswith(boundary):
            continue
        suffix = name[len(marker_prefix) :]
        if suffix.startswith("::bwd::"):
            bwd_nodes.append(node)
        else:
            fwd_nodes.append(node)
    return fwd_nodes, bwd_nodes


def _replace_input_markers(gm: fx.GraphModule, marker_prefix: str) -> List[fx.Node]:
    """Strip and return ``{prefix}::input::<idx>`` markers, ordered by idx.

    The returned nodes are the *upstream* nodes (i.e. ``marker.args[0]``)
    after the marker itself has been removed from the graph.
    """
    inputs: List[Tuple[int, fx.Node, fx.Node]] = []
    for node in list(gm.graph.nodes):
        if (
            node.op != "call_function"
            or node.target is not torch.ops.recir.mark_boundary.default
            or len(node.args) < 2
        ):
            continue
        name = node.args[1]
        if not isinstance(name, str):
            continue
        if not name.startswith(marker_prefix + "::input::"):
            continue
        idx_str = name[len(marker_prefix + "::input::") :]
        try:
            idx = int(idx_str)
        except ValueError:
            continue
        inputs.append((idx, node, node.args[0]))

    inputs.sort(key=lambda t: t[0])
    upstream: List[fx.Node] = []
    for _idx, marker, src in inputs:
        marker.replace_all_uses_with(src)
        gm.graph.erase_node(marker)
        upstream.append(src)
    return upstream


def install_graph_break_dispatch(
    gm: fx.GraphModule,
    marker_prefix: str,
    fwd_fn: Callable[..., object],
    *,
    num_state_tensors: int,
    num_ranks: int = 1,
    disable_torchrec_gradient_division: bool = False,
    register_backward: bool = True,
) -> Optional[int]:
    """Replace markers planted under ``marker_prefix`` with a graph-break dispatch.

    Walks ``gm`` looking for ``recir::mark_boundary`` nodes whose name begins
    with ``marker_prefix`` and (depending on suffix) installs:

    * ``::input::<idx>`` markers → stripped; their upstream values become the
      dispatch inputs.  If no input markers are present, the trailing
      placeholders (those after the first ``num_state_tensors`` model-state
      placeholders) become the dispatch inputs (mirrors the legacy
      ``replace_override_anchors`` default).
    * ``::output::<idx>`` markers → replaced with ``getitem(_dispatch_fwd(...),
      idx)`` calls so their downstream users see the runtime dispatch result.
    * ``::bwd::<idx>`` markers (if any) → replaced with ``_dispatch_bwd(...)``
      calls so gradient grads flow into the user-supplied backward callable.

    ``disable_torchrec_gradient_division`` is only for torchrec sparse seams,
    where the combined backward must produce a cross-host gradient sum.

    Returns the allocated dispatch id (``None`` if no markers matched).
    """
    fwd_markers, bwd_markers = _collect_marker_groups(gm, marker_prefix)
    if not fwd_markers and not bwd_markers:
        return None

    if bwd_markers and not register_backward:
        for marker in bwd_markers:
            marker.replace_all_uses_with(marker.args[0])
            gm.graph.erase_node(marker)
        bwd_markers = []

    # Detect input-only graphs: ``::input::`` markers are pure metadata that
    # ``_replace_input_markers`` strips without inserting any dispatch call.
    # If there are no ``::output::`` markers and no actionable ``::bwd::``
    # markers, the dispatch_id we would allocate is never spliced into the
    # graph, so the registry entry leaks for the lifetime of the process.
    # Skip registration in that case and just clean up the input markers.
    has_output_marker = any("::output::" in node.args[1] for node in fwd_markers)
    has_actionable_bwd = bool(bwd_markers) and register_backward
    if not has_output_marker and not has_actionable_bwd:
        # Strip input markers (so they don't linger as no-ops in the graph),
        # but do not register a dispatch callable.
        _replace_input_markers(gm, marker_prefix)
        gm.graph.eliminate_dead_code()
        gm.recompile()
        return None

    fwd, bwd = _wrap_with_save_for_backward(fwd_fn, disable_torchrec_gradient_division)
    dispatch_id = _allocate_dispatch_id()
    with _registry_lock:
        _FWD_REGISTRY[dispatch_id] = fwd
        if register_backward:
            _BWD_REGISTRY[dispatch_id] = bwd

    # 1. Determine dispatch inputs.
    input_nodes = _replace_input_markers(gm, marker_prefix)
    if not input_nodes:
        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        tensor_placeholders = [
            (i, n)
            for i, n in enumerate(placeholders)
            if isinstance(n.meta.get("val", None), torch.Tensor)
        ]
        input_nodes = [
            placeholders[i] for i, _ in tensor_placeholders if i >= num_state_tensors
        ]

    # 2. Pick the insertion point: just before the FIRST output marker so the
    #    dispatch call dominates every getitem we insert.
    output_markers = sorted(
        (
            (
                int(node.args[1].split("::output::")[1]),
                node,
            )
            for node in fwd_markers
            if "::output::" in node.args[1]
        ),
        key=lambda t: t[0],
    )

    if output_markers:
        with gm.graph.inserting_before(output_markers[0][1]):
            fwd_call = gm.graph.call_function(
                _dispatch_fwd,
                args=(input_nodes, dispatch_id),
            )
            fwd_call.meta["xla_graph_break"] = True

        for output_idx, marker in output_markers:
            with gm.graph.inserting_before(marker):
                getitem_node = gm.graph.call_function(
                    operator.getitem,
                    args=(fwd_call, output_idx),
                )
                getitem_node.meta = dict(marker.meta)
                output_val = marker.meta.get("val", None)
                is_batch_output = not (
                    isinstance(output_val, torch.Tensor) and output_val.ndim == 0
                )
                getitem_node.meta["is_batch_input"] = is_batch_output
                if is_batch_output:
                    getitem_node.meta["batch_dim"] = 0
                else:
                    getitem_node.meta.pop("batch_dim", None)
            marker.replace_all_uses_with(getitem_node)
            gm.graph.erase_node(marker)

    # 3. Backward markers.
    if bwd_markers and register_backward:
        sorted_bwd = sorted(
            bwd_markers,
            key=lambda n: int(n.args[1].split("::bwd::")[1]),
        )
        for grad_idx, marker in enumerate(sorted_bwd):
            grad_node = marker.args[0]
            with gm.graph.inserting_before(marker):
                bwd_call = gm.graph.call_function(
                    _dispatch_bwd,
                    args=(grad_node, dispatch_id, grad_idx),
                )
                bwd_call.meta = dict(marker.meta)
                bwd_call.meta["xla_graph_break"] = True
            marker.replace_all_uses_with(bwd_call)
            gm.graph.erase_node(marker)

    gm.graph.eliminate_dead_code()
    gm.recompile()
    return dispatch_id


# ---------------------------------------------------------------------------
# Trace-time helpers that plant the marker pairs Dummies need.
# ---------------------------------------------------------------------------


class _OutputBackwardMarker(torch.autograd.Function):
    """Plant ``{prefix}::output::idx`` forward + ``{prefix}::bwd::idx`` backward.

    The forward inserts a ``recir::mark_boundary`` node wrapping ``x`` with the
    output marker name.  The backward inserts a second ``mark_boundary`` node
    on the gradient tensor with the matching bwd marker name so the post-trace
    orchestrator can find and replace both halves uniformly.
    """

    @staticmethod
    def forward(ctx, x, prefix: str, idx: int, *deps):  # pyre-ignore[14]
        ctx.prefix = prefix
        ctx.idx = idx
        ctx._num_deps = len(deps)
        return torch.ops.recir.mark_boundary(x, f"{prefix}::output::{idx}")

    @staticmethod
    def backward(ctx, grad):  # pyre-ignore[14]
        return (
            torch.ops.recir.mark_boundary(grad, f"{ctx.prefix}::bwd::{ctx.idx}"),
            None,
            None,
        ) + (None,) * ctx._num_deps


def mark_output_with_bwd(
    tensor: torch.Tensor,
    prefix: str,
    idx: int,
    deps: Optional[List[torch.Tensor]] = None,
) -> torch.Tensor:
    """Plant a forward/backward marker pair on ``tensor``.

    ``deps`` (optional) is a list of tensors carried as extra positional inputs
    to the autograd Function purely to keep functionalization/DCE from pruning
    them (e.g. marker inputs that would otherwise become orphans after
    constant-folding).  ``deps`` does not influence the value of ``tensor`` and
    receives no gradient.
    """
    if deps is None:
        deps = []
    return _OutputBackwardMarker.apply(tensor, prefix, idx, *deps)


def mark_input(tensor: torch.Tensor, prefix: str, idx: int) -> torch.Tensor:
    """Plant an input marker that the orchestrator will use as a dispatch input."""
    return torch.ops.recir.mark_boundary(tensor, f"{prefix}::input::{idx}")
