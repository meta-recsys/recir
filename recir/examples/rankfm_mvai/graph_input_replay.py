#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""Reproduce a graph-break dispatch's inputs outside the compiled graph.

A STEP HAS FOUR PARTS

    prologue    host FX nodes at the head of the compiled graph: the
                PaddingRegionTransform pads the variable-size batch to the
                static shapes the device program requires.
    producer    the preprocessor graph-break callback. The compiled program
                stops on device, calls into Python, and cannot resume until it
                returns -- 335 ms/step of device idle at 4 chips.
    device      the compiled program itself: fwd, bwd, optimizer.
    consumer    the step output's device->host pull and the metric update it
                feeds.

PRODUCER AND CONSUMER ARE PIPELINED

Both are pure functions of one step's data and neither needs the device idle, so
they run on a worker inside a device window instead of in the gap between two
device executions: the producer works a step ahead, the consumer a step behind.
That is the whole speedup. This module exists only because the producer cannot
be moved without also solving its arguments.

WHY THE PROLOGUE IS NOT -- AND WHAT THAT COSTS

The producer's arguments are not the batch. They are what the prologue
computed from the batch, and for step N+1 they do not exist yet: producing them
is part of what running step N+1 means. So running the producer early requires
running the prologue early too, and this module is that -- a copy of the
prologue, on the worker.

A copy, not a move. The obvious simplification is to delete the prologue from
the graph once the worker owns it, leave the callback with no arguments, and let
the replay be the single source of truth -- no second producer, so no per-step
check either. Measured, not assumed: it does not cut cleanly. Of the four
derived nodes, two are also read by the device-side program --
``pad_count_to_sum`` by ``repeat_interleave`` / ``cumsum`` (the padded candidate
counts drive every jagged offset downstream; see ``flatten_batch`` in
``recir_core/trace_sample.py``) and the transposed supervision weight by an
``index``. Those have to be computed in the graph whatever the host does, so
lifting the prologue means passing them in as new step arguments -- a change to
the compiled program's input signature.

Two things follow. The duplicated work is small: the in-graph copy was never
removable, 96 of the 100 arguments are bare placeholders, and only four are
derived. And two producers of those values really do exist, which is why
:meth:`DispatchInputReplay.assert_matches` checks one against the other every
step rather than trusting the copy.

STRUCTURE

:func:`build_dispatch_input_replay` reads the recipe off the graph -- the
dispatch call's first argument IS the list of input nodes -- walks their
transitive closure back to placeholders, refuses anything the host cannot
reproduce (model state, a nested graph break, an in-place op), and ``node_copy``
s the rest into a standalone GraphModule whose parameters are the step's user
args. :class:`DispatchInputReplay` then maps a batch's args onto it.

The nodes are copied rather than reimplemented because reimplementing them is
what corrupted training data on the first attempt at this. Three of its
hand-built slots drifted from the graph's:

    LOCAL_NUM_CANDIDATES   graph: trim_per_chip_to_size(pad_count_to_sum(nc))
                           hand:  batch.num_candidates
    ro_float_features      graph: None       hand: batch.ro_float_features
    supervision_weight     graph: present    hand: missing

The first alone shifted every jagged offset downstream -- 89% of the elements of
the largest output KJT differed. Nothing raised; the model just trained worse
for nine hours. A copy has no algorithm to re-derive and no baked-in constant to
fill in wrong, and it follows the graph automatically when the model or the
PaddingRegionTransform changes.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence, Set

import torch
from recir.eager_dispatch import _dispatch_bwd, _dispatch_fwd
from torch import fx

logger: logging.Logger = logging.getLogger(__name__)


class DispatchInputReplay:
    """Recomputes a dispatch's input tensors from a step's flattened arguments."""

    def __init__(
        self,
        replay_gm: fx.GraphModule,
        arg_indices: List[int],
    ) -> None:
        self._replay_gm = replay_gm
        self._arg_indices = arg_indices

    def __call__(self, flat_step_args: Sequence[Any]) -> List[torch.Tensor]:
        for i in self._arg_indices:
            if i >= len(flat_step_args):
                raise RuntimeError(
                    f"replay wants step arg {i} but the step passed only "
                    f"{len(flat_step_args)}; the graph's placeholders and the "
                    "runner's arguments have drifted apart"
                )
        with torch.no_grad():
            return list(
                self._replay_gm(*[flat_step_args[i] for i in self._arg_indices])
            )

    def assert_matches(
        self,
        from_graph: Sequence[torch.Tensor],
        flat_step_args: Sequence[Any],
    ) -> None:
        replayed = self(flat_step_args)
        if len(replayed) != len(from_graph):
            raise RuntimeError(
                f"replay produced {len(replayed)} dispatch inputs, the graph "
                f"passed {len(from_graph)}"
            )
        for i, (want, got) in enumerate(zip(from_graph, replayed)):
            if want.shape != got.shape or want.dtype != got.dtype:
                raise RuntimeError(
                    f"replayed dispatch input {i} is "
                    f"{tuple(got.shape)}/{got.dtype}, the graph passed "
                    f"{tuple(want.shape)}/{want.dtype}"
                )
            if not torch.equal(want, got):
                differing = int((want != got).sum().item())
                raise RuntimeError(
                    f"replayed dispatch input {i} differs from the graph's in "
                    f"{differing}/{want.numel()} elements"
                )


def _find_dispatch_node(gm: fx.GraphModule, dispatch_id: int) -> fx.Node:
    for node in gm.graph.nodes:
        if (
            node.op == "call_function"
            and node.target is _dispatch_fwd
            and len(node.args) >= 2
            and node.args[1] == dispatch_id
        ):
            return node
    raise RuntimeError(
        f"no _dispatch_fwd call for dispatch_id {dispatch_id} in the graph"
    )


def _reject_unreplayable(node: fx.Node) -> None:
    """Raise unless ``node`` can be re-executed on the host, off-step."""
    if node.op == "placeholder":
        return
    if node.op == "get_attr":
        return
    if node.op != "call_function":
        raise RuntimeError(
            f"dispatch input depends on {node.op} node {node.name!r}, which the "
            "replay cannot re-execute"
        )
    if node.target in (_dispatch_fwd, _dispatch_bwd):
        raise RuntimeError(
            f"dispatch input depends on another graph break ({node.name!r}); "
            "the replay would re-enter it"
        )
    schema = getattr(node.target, "_schema", None)
    if schema is not None and schema.is_mutable:
        # Replaying a mutating op would apply it a second time -- to the NEXT
        # batch's tensors, before the graph gets to them.
        raise RuntimeError(
            f"dispatch input depends on the mutating op {node.target}; the "
            "replay cannot run it without corrupting the batch"
        )


def build_dispatch_input_replay(
    gm: fx.GraphModule,
    dispatch_id: int,
    num_state_tensors: int,
) -> DispatchInputReplay:
    dispatch_node = _find_dispatch_node(gm, dispatch_id)
    spliced = dispatch_node.args[0]
    if not isinstance(spliced, (list, tuple)) or not all(
        isinstance(n, fx.Node) for n in spliced
    ):
        raise RuntimeError(
            f"_dispatch_fwd for {dispatch_id} does not carry a list of input "
            f"nodes: {spliced!r}"
        )
    input_nodes: List[fx.Node] = [n for n in spliced if isinstance(n, fx.Node)]
    placeholder_index: Dict[fx.Node, int] = {
        node: i
        for i, node in enumerate(n for n in gm.graph.nodes if n.op == "placeholder")
    }
    needed: Set[fx.Node] = set()
    stack: List[fx.Node] = list(input_nodes)
    while stack:
        node = stack.pop()
        if node in needed:
            continue
        needed.add(node)
        stack.extend(node.all_input_nodes)
    for node in needed:
        _reject_unreplayable(node)
        if node.op == "placeholder" and placeholder_index[node] < num_state_tensors:
            raise RuntimeError(
                f"dispatch input depends on model-state placeholder "
                f"{node.name!r} (index {placeholder_index[node]} < "
                f"{num_state_tensors}); it is not a pure function of the batch"
            )

    replay_graph = fx.Graph()
    env: Dict[fx.Node, fx.Node] = {}
    arg_indices: List[int] = []
    for node in gm.graph.nodes:
        if node not in needed:
            continue
        if node.op == "placeholder":
            arg_indices.append(placeholder_index[node] - num_state_tensors)
        env[node] = replay_graph.node_copy(node, lambda n: env[n])
    replay_graph.output(tuple(env[node] for node in input_nodes))

    replay_gm = fx.GraphModule(gm, replay_graph)
    replay_gm.recompile()

    computed = sum(1 for n in needed if n.op != "placeholder")
    logger.info(
        f"[PIPE] dispatch {dispatch_id} input replay: {len(input_nodes)} inputs "
        f"from {len(arg_indices)} step args and {computed} computed nodes"
    )
    return DispatchInputReplay(replay_gm, arg_indices)
