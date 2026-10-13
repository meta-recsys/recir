# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Bridge a traced training-step FX graph into a clean inference graph.

RecIR captures a full training step (forward + loss + backward + optimizer) as
one FX graph.  This module turns that graph into a forward-only inference graph
*in place*, without retracing, by composing four steps:

1. **Cut the output to the forward predictions.**  In a train-step graph the
   prediction tensors are usually *intermediate* nodes consumed by the loss, not
   graph outputs (the output is the loss / updated state).  Re-pointing the
   ``output`` node to the prediction nodes makes the loss, backward, optimizer,
   and grad-only collectives unreachable.

2. **Strip ``tag_grad`` markers** (``torch.ops._spmd.tag_grad``).  These identity
   hooks sit on the backward path and block dead-code elimination.

3. **Strip training collectives** (``all_reduce`` + ``wait_tensor`` by default).
   For a single-device inference graph an all-reduce over one rank is the
   identity, so the collective is replaced by its data input.  Inference
   *re-sharding* is left to the downstream serving stack (TGIF).

4. **Dead-code elimination + region surgery.**  DCE drops everything now
   unreachable from the inference outputs (loss/backward/optimizer); then
   :func:`apply_inference_swaps` removes dropout / stochastic sampling, swaps
   training-only module regions, and prunes pruned towers.

The expensive ``make_fx`` trace is shared with training — this bridge is pure
graph surgery (milliseconds), preserving a single source of truth.

Usage::

    from recir.region_transforms.training_to_inference import (
        fabricate_inference_graph,
        inputs_to_ops,
    )

    # Cut at the loss: inference outputs are the loss node's inputs (predictions)
    gm = fabricate_inference_graph(gm, inference_outputs=inputs_to_ops({"sum"}))
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable

import torch
import torch.fx as fx
import torch.utils._pytree as pytree
from recir.ops.marker import mark_boundary
from recir.region_transforms.inference_swap import (
    _graph_output_node,
    apply_inference_swaps,
)
from recir.utils.graph import OP, rebuild_graph

logger: logging.Logger = logging.getLogger(__name__)

# A selector resolves the inference output nodes from the (training) graph.
OutputSelector = Callable[[fx.GraphModule], list[fx.Node]]

# Collective op-name fragments that are the identity on a single rank and so can
# be replaced by their data input when fabricating a single-device inference
# graph.  ``wait_tensor`` is always stripped (it is the identity once its async
# collective is gone).
_DEFAULT_COLLECTIVE_NAMES: tuple[str, ...] = ("all_reduce", "wait_tensor")
_GRAD_TAG_NAME: str = "tag_grad"

# Marker-name prefix used by :func:`mark_inference_output`.  The ``::output::``
# suffix mirrors RecIR's existing forward-marker convention (eager_dispatch).
_INFERENCE_OUTPUT_PREFIX: str = "recir_inference_output"


def mark_inference_output(t: torch.Tensor, idx: int = 0) -> torch.Tensor:
    """Tag ``t`` as an inference output so fabrication can find it post-trace.

    Identity at runtime (reuses ``torch.ops.recir.mark_boundary``, a clone).
    The tag is the robust replacement for op-name substring matching: plant it
    on the real prediction tensor in the model's forward, then select it with
    :func:`marked_inference_outputs`.  It survives ``make_fx`` and stays in the
    graph even when the training step discards the prediction (only the loss is
    used), which is exactly the DLRMv3 case.
    """
    return mark_boundary(t, f"{_INFERENCE_OUTPUT_PREFIX}::output::{idx}")


def marked_inference_outputs(
    prefix: str = _INFERENCE_OUTPUT_PREFIX,
) -> OutputSelector:
    """Selector: every :func:`mark_inference_output` marker node, in graph order.

    Returns the marker nodes themselves (identity ops) as the inference outputs;
    the forward-cone prune then keeps only the prediction subgraph feeding them.
    """

    def _selector(gm: fx.GraphModule) -> list[fx.Node]:
        outputs: list[fx.Node] = []
        for node in gm.graph.nodes:
            if (
                node.op == OP.CALL_FUNCTION
                and node.target is torch.ops.recir.mark_boundary.default
                and len(node.args) >= 2
                and isinstance(node.args[1], str)
                and node.args[1].startswith(prefix)
            ):
                outputs.append(node)
        return outputs

    return _selector


def inputs_to_ops(name_substrings: Iterable[str]) -> OutputSelector:
    """Selector: the inputs of every op whose target name matches a substring.

    Use to cut a train-step graph at the loss — e.g. ``inputs_to_ops({"sum",
    "nll_loss", "binary_cross_entropy"})`` returns the prediction tensors that
    feed the loss, which become the inference outputs.
    """
    substrings: tuple[str, ...] = tuple(name_substrings)

    def _selector(gm: fx.GraphModule) -> list[fx.Node]:
        outputs: list[fx.Node] = []
        seen: set[fx.Node] = set()
        for node in gm.graph.nodes:
            if node.op != OP.CALL_FUNCTION:
                continue
            if not any(s in str(node.target) for s in substrings):
                continue
            for inp in node.all_input_nodes:
                if inp not in seen:
                    seen.add(inp)
                    outputs.append(inp)
        return outputs

    return _selector


def select_inference_outputs(gm: fx.GraphModule, outputs: list[fx.Node]) -> None:
    """Re-point the graph ``output`` node to ``outputs`` (single node or tuple).

    Also rewrites the pytree ``out_spec`` so the recompiled ``forward`` unflattens
    the *new* structure.  A RecIR training graph was traced returning
    ``(ret, params, buffers, named_states)``; without updating the spec the
    generated ``pytree.tree_unflatten(select, self._out_spec)`` tries to rebuild
    that old structure from the (now fewer) inference outputs and fails.
    """
    if not outputs:
        raise ValueError("inference output selection is empty")
    output_node = _graph_output_node(gm)
    # Always return a tuple: the codegen then passes a *list* of output values
    # to ``tree_unflatten``.  A bare single node makes the codegen pass the raw
    # (possibly 0-d) tensor, which ``tree_unflatten`` then tries to iterate.
    output_node.args = (tuple(outputs),)
    _set_out_spec(gm, len(outputs))
    gm.recompile()


def _set_out_spec(gm: fx.GraphModule, num_outputs: int) -> None:
    """Point the gm's pytree out_spec at the new (tuple) output structure."""
    new_spec = pytree.tree_structure(tuple(range(num_outputs)))
    codegen = gm.graph._codegen
    info = getattr(codegen, "pytree_info", None)
    if info is not None:
        # pyre-ignore[16]: _PyTreeInfo is a NamedTuple with _replace.
        gm.graph._codegen.pytree_info = info._replace(out_spec=new_spec)
    # The generated forward reads ``self._out_spec`` directly.
    # pyre-ignore[16]: GraphModule sets ``_out_spec`` dynamically via codegen.
    gm._out_spec = new_spec


def strip_grad_tags(gm: fx.GraphModule) -> int:
    """Remove ``_spmd.tag_grad`` identity markers, rewiring users to the input."""
    removed = 0
    for node in list(gm.graph.nodes):
        if node.op != OP.CALL_FUNCTION or _GRAD_TAG_NAME not in str(node.target):
            continue
        node.replace_all_uses_with(node.args[0])
        gm.graph.erase_node(node)
        removed += 1
    if removed:
        rebuild_graph(gm)
    return removed


def strip_collectives(
    gm: fx.GraphModule,
    collective_names: Iterable[str] = _DEFAULT_COLLECTIVE_NAMES,
) -> int:
    """Replace single-rank-identity collectives with their data input.

    Defaults to ``all_reduce`` + ``wait_tensor`` (always identity on one rank).
    Pass extra fragments (e.g. ``all_gather_into_tensor``) only for graphs known
    to be world-size 1, where those are also the identity.
    """
    names = tuple(collective_names)
    removed = 0
    for node in list(gm.graph.nodes):
        if node.op != OP.CALL_FUNCTION:
            continue
        if not any(name in str(node.target) for name in names):
            continue
        data_inputs = node.all_input_nodes
        if not data_inputs:
            continue
        node.replace_all_uses_with(data_inputs[0])
        gm.graph.erase_node(node)
        removed += 1
    if removed:
        rebuild_graph(gm)
    return removed


def prune_to_forward_cone(gm: fx.GraphModule, outputs: list[fx.Node]) -> int:
    """Keep only nodes backward-reachable from ``outputs``; erase the rest.

    Unlike ``eliminate_dead_code``, this removes **impure** ops too — the
    backward graph, in-place optimizer updates, ``tag_grad`` markers, and the
    autograd ``::bwd::`` marker ops that RecIR plants for graph-break dispatch.
    Those are side-effecting, so DCE keeps them and they hold the whole backward
    subgraph alive; re-pointing the output is not enough on a real training
    graph.  Placeholders are always kept so the graph's input signature (and
    therefore the caller's positional binding) is unchanged.

    Returns the number of nodes erased.
    """
    keep: set[fx.Node] = set()
    stack: list[fx.Node] = list(outputs)
    while stack:
        node = stack.pop()
        if node in keep:
            continue
        keep.add(node)
        stack.extend(node.all_input_nodes)

    removed = 0
    # Reverse topological order: a non-kept node's users are themselves non-kept
    # (else the node would be in ``keep``), so they are erased first.
    for node in reversed(list(gm.graph.nodes)):
        if node.op in (OP.PLACEHOLDER, OP.OUTPUT):
            continue
        if node in keep:
            continue
        gm.graph.erase_node(node)
        removed += 1
    rebuild_graph(gm)
    return removed


def fabricate_inference_graph(
    gm: fx.GraphModule,
    inference_outputs: OutputSelector | list[str] | None = None,
    strip_training_collectives: bool = True,
    # pyre-ignore[2]: forwarded verbatim to apply_inference_swaps.
    **inference_swap_kwargs: object,
) -> fx.GraphModule:
    """Turn a traced training-step graph into a forward-only inference graph.

    Args:
        gm: the traced training-step graph (mutated in place).
        inference_outputs: how to pick the forward outputs to keep — a selector
            callable, a list of node names, or ``None`` to keep the existing
            graph outputs (when the graph is already forward-only).
        strip_training_collectives: replace single-rank collectives with identity.
        **inference_swap_kwargs: forwarded to :func:`apply_inference_swaps`
            (``module_swaps``, ``prune_module_classes``, ``enable_dropout_removal``,
            ``enable_sampling_removal``, ``custom_op_matchers``).
    """
    if inference_outputs is not None:
        resolved = _resolve_outputs(gm, inference_outputs)
        select_inference_outputs(gm, resolved)
        # Forward-cone prune is the robust backward/optimizer strip: it drops
        # impure backward + optimizer + autograd-marker ops that DCE cannot.
        prune_to_forward_cone(gm, resolved)

    # Any collectives / grad-tags still in the forward cone are rewritten to
    # identity for a single-device inference graph.
    strip_grad_tags(gm)
    if strip_training_collectives:
        strip_collectives(gm)
    rebuild_graph(gm)

    # pyre-ignore[6]: kwargs validated by apply_inference_swaps.
    apply_inference_swaps(gm, **inference_swap_kwargs)
    return gm


def _resolve_outputs(
    gm: fx.GraphModule,
    spec: OutputSelector | list[str],
) -> list[fx.Node]:
    if callable(spec):
        return list(spec(gm))
    name_to_node = {node.name: node for node in gm.graph.nodes}
    missing = [name for name in spec if name not in name_to_node]
    if missing:
        raise ValueError(f"unknown output node name(s): {missing}")
    return [name_to_node[name] for name in spec]
