# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors

"""Standard anchor ops and dispatch helpers for Override backward integration."""

from __future__ import annotations

import operator
from typing import Callable, Dict, Tuple

import torch
import torch.utils._pytree as pytree
from torch import fx


_OVERRIDE_FWD: Dict[int, Callable] = {}
_OVERRIDE_BWD: Dict[int, Callable] = {}

# ---------------------------------------------------------------------------
# Anchor ops — appear in the make_fx-traced graph as identifiable nodes.
# Both have FakeTensor impl (for tracing) but NO Meta kernel → graph break.
# ---------------------------------------------------------------------------

_lib: torch.library.Library = torch.library.Library("recir_override", "DEF")  # noqa: TOR901


def _fwd_anchor_cpu(x: torch.Tensor, override_id: int, output_idx: int) -> torch.Tensor:
    return x


_lib.define("fwd_anchor(Tensor x, int override_id, int output_idx) -> Tensor")
_lib.impl("fwd_anchor", _fwd_anchor_cpu, "CPU")


@torch.library.impl_abstract("recir_override::fwd_anchor")
def _fwd_anchor_fake(
    x: torch.Tensor, override_id: int, output_idx: int
) -> torch.Tensor:
    return torch.empty_like(x)


def _bwd_anchor_cpu(
    grad: torch.Tensor, override_id: int, output_idx: int
) -> torch.Tensor:
    return grad


_lib.define("bwd_anchor(Tensor grad, int override_id, int output_idx) -> Tensor")
_lib.impl("bwd_anchor", _bwd_anchor_cpu, "CPU")


@torch.library.impl_abstract("recir_override::bwd_anchor")
def _bwd_anchor_fake(
    grad: torch.Tensor, override_id: int, output_idx: int
) -> torch.Tensor:
    return torch.empty_like(grad)


# ---------------------------------------------------------------------------
# Input anchor op — marks inputs to an overridden module so that
# replace_override_anchors can use the correct dispatch inputs instead
# of all batch placeholders.  Needed for deeply nested overrides whose
# inputs differ from the train_step batch inputs.
# ---------------------------------------------------------------------------


def _input_anchor_cpu(
    x: torch.Tensor, override_id: int, input_idx: int
) -> torch.Tensor:
    return x


_lib.define("input_anchor(Tensor x, int override_id, int input_idx) -> Tensor")
_lib.impl("input_anchor", _input_anchor_cpu, "CPU")


@torch.library.impl_abstract("recir_override::input_anchor")
def _input_anchor_fake(
    x: torch.Tensor, override_id: int, input_idx: int
) -> torch.Tensor:
    return torch.empty_like(x)


def default_override_transform(
    gm: fx.GraphModule,
    flat_state: list[torch.Tensor],
    orig_module: torch.nn.Module,
    override_id: int = 0,
    num_ranks: int = 1,
) -> fx.GraphModule:
    """Standard transform: wire up *orig_module* as override forward/backward.

    Forward: runs *orig_module* under ``enable_grad()``, flattens outputs
    via ``pytree.tree_leaves`` (filtering to tensors), saves
    autograd-connected tensors, returns detached.

    Backward: scales gradient by ``1/num_ranks`` and calls
    ``torch.autograd.backward`` on saved outputs.

    Args:
        gm: the traced graph module to transform.
        flat_state: flattened model state (params + buffers + optimizer states).
        orig_module: the real module to run at runtime (e.g., the original
            preprocess that was replaced by a dummy during tracing).
    """

    state: dict[str, list[torch.Tensor] | None] = {"saved": None}

    def fwd_fn(*args: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        with torch.enable_grad():
            raw_output = orig_module(*args)
        flat = [
            t for t in pytree.tree_leaves(raw_output) if isinstance(t, torch.Tensor)
        ]
        state["saved"] = [t for t in flat if t.requires_grad]
        return tuple(t.detach() for t in flat)

    def bwd_fn(grad: torch.Tensor, grad_idx: int = 0) -> None:
        saved = state["saved"]
        if saved is not None and grad_idx < len(saved):
            s = saved[grad_idx]
            is_last = grad_idx == len(saved) - 1
            torch.autograd.backward([s], [grad / num_ranks], retain_graph=not is_last)
            if is_last:
                state["saved"] = None

    return replace_override_anchors(
        gm,
        override_id=override_id,
        fwd_fn=fwd_fn,
        bwd_fn=bwd_fn,
        num_state_tensors=len(flat_state),
    )


# ---------------------------------------------------------------------------
# Module-level dispatch functions — inserted into the FX graph by
# replace_override_anchors.  Must be module-level (not closures) so that
# FX recompile() can reference them by qualified name.
# Inputs are always torch tensors: batch inputs come from user_inps (torch),
# and _XLAModule outputs that cross into un-lowered ops are converted to
# torch by _postprocess_jax_output (via torch_output_indices).
# ---------------------------------------------------------------------------


def _dispatch_fwd(inputs: list[torch.Tensor], override_id: int) -> list[torch.Tensor]:
    return list(_OVERRIDE_FWD[override_id](*inputs))


def _dispatch_bwd(grad: torch.Tensor, override_id: int, grad_idx: int) -> torch.Tensor:
    _OVERRIDE_BWD[override_id](grad, grad_idx)
    return grad


def replace_override_anchors(
    gm: fx.GraphModule,
    override_id: int,
    fwd_fn: Callable[..., Tuple[torch.Tensor, ...]],
    bwd_fn: Callable[[torch.Tensor, int], None],
    num_state_tensors: int,
) -> fx.GraphModule:
    """Replace fwd_anchor/bwd_anchor nodes with dispatch calls to *fwd_fn*/*bwd_fn*.

    This is the helper that Override subclasses call from ``transform()``
    to wire up their forward/backward callables into the traced graph.

    *fwd_fn* and *bwd_fn* are registered internally so the module-level
    dispatch functions (``_dispatch_fwd`` / ``_dispatch_bwd``) can look
    them up at runtime.  The inserted dispatch nodes carry
    ``meta["xla_graph_break"] = True`` so the XLA lowering pass treats
    them as graph breaks.
    """

    fwd_anchors = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and n.target is torch.ops.recir_override.fwd_anchor.default
        and n.args[1] == override_id
    ]
    bwd_anchors = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and n.target is torch.ops.recir_override.bwd_anchor.default
        and n.args[1] == override_id
    ]
    if not fwd_anchors and not bwd_anchors:
        return gm

    # Register callbacks for module-level dispatch functions.
    _OVERRIDE_FWD[override_id] = fwd_fn
    _OVERRIDE_BWD[override_id] = bwd_fn

    # Determine dispatch input nodes.
    # If input_anchor nodes exist for this override_id, use them as the
    # dispatch inputs (the actual inputs to the overridden module).
    # Otherwise, fall back to all batch input placeholders
    input_anchors = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and n.target is torch.ops.recir_override.input_anchor.default
        and n.args[1] == override_id
    ]
    if input_anchors:
        sorted_inputs = sorted(input_anchors, key=lambda n: n.args[2])
        input_nodes = [a.args[0] for a in sorted_inputs]
        for a in input_anchors:
            a.replace_all_uses_with(a.args[0])
            gm.graph.erase_node(a)
    else:
        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        tensor_placeholders = [
            (i, n)
            for i, n in enumerate(placeholders)
            if isinstance(n.meta.get("val", None), torch.Tensor)
        ]
        input_nodes = [
            placeholders[i] for i, _ in tensor_placeholders if i >= num_state_tensors
        ]

    # Replace forward anchors.
    if fwd_anchors:
        with gm.graph.inserting_before(fwd_anchors[0]):
            fwd_node = gm.graph.call_function(
                _dispatch_fwd,
                args=(input_nodes, override_id),
            )
            fwd_node.meta["xla_graph_break"] = True
        for anchor in fwd_anchors:
            output_idx = anchor.args[2]
            with gm.graph.inserting_before(anchor):
                getitem_node = gm.graph.call_function(
                    operator.getitem,
                    args=(fwd_node, output_idx),
                )
                getitem_node.meta = anchor.meta.copy()
                getitem_node.meta["is_batch_input"] = True
                getitem_node.meta["batch_dim"] = 0
            anchor.replace_all_uses_with(getitem_node)

    # Replace backward anchors.
    if bwd_anchors:
        sorted_bwd = sorted(bwd_anchors, key=lambda n: n.args[2])
        for grad_idx, anchor in enumerate(sorted_bwd):
            grad_node = anchor.args[0]
            with gm.graph.inserting_before(anchor):
                bwd_call = gm.graph.call_function(
                    _dispatch_bwd,
                    args=(grad_node, override_id, grad_idx),
                )
                bwd_call.meta = anchor.meta.copy()
                bwd_call.meta["xla_graph_break"] = True
            anchor.replace_all_uses_with(bwd_call)

    gm.graph.eliminate_dead_code()
    gm.recompile()
    return gm


# ---------------------------------------------------------------------------
# OverrideAnchorFn — autograd.Function that produces anchor nodes during tracing
# ---------------------------------------------------------------------------


class OverrideAnchorFn(torch.autograd.Function):
    """Wrap a single output tensor with fwd_anchor (forward) and bwd_anchor (backward).

    Both ops have FakeTensor impls (for make_fx tracing) but no Meta kernel,
    so they appear as graph-breaking nodes in the traced graph.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        override_id: int,
        output_idx: int,
    ) -> torch.Tensor:
        ctx.override_id = override_id  # pyre-ignore[16]
        ctx.output_idx = output_idx  # pyre-ignore[16]
        return torch.ops.recir_override.fwd_anchor(x, override_id, output_idx)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad: torch.Tensor,
    ) -> Tuple[torch.Tensor, None, None]:
        override_id = ctx.override_id  # pyre-ignore[16]
        output_idx = ctx.output_idx  # pyre-ignore[16]
        return (
            torch.ops.recir_override.bwd_anchor(grad, override_id, output_idx),
            None,
            None,
        )


def anchor_output(
    tensor: torch.Tensor, override_id: int, output_idx: int
) -> torch.Tensor:
    """Wrap a DummyPreprocess output tensor with OverrideAnchorFn.

    This creates identifiable fwd_anchor (forward) and bwd_anchor (backward)
    nodes in the make_fx-traced FX graph.
    """
    return OverrideAnchorFn.apply(tensor, override_id, output_idx)


class InputAnchorFn(torch.autograd.Function):
    """Wrap a single input tensor with input_anchor.

    The forward inserts a findable ``input_anchor`` node in the FX graph.
    The backward is a trivial pass-through so make_fx can trace through it.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        override_id: int,
        input_idx: int,
    ) -> torch.Tensor:
        return torch.ops.recir_override.input_anchor(x, override_id, input_idx)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad: torch.Tensor,
    ) -> Tuple[torch.Tensor, None, None]:
        return grad, None, None


def anchor_input(
    tensor: torch.Tensor, override_id: int, input_idx: int
) -> torch.Tensor:
    """Wrap an input tensor with InputAnchorFn.

    Creates an identifiable input_anchor node in the make_fx-traced FX graph.
    Used by replace_override_anchors to determine the correct dispatch inputs
    for deeply nested overridden modules.
    """
    return InputAnchorFn.apply(tensor, override_id, input_idx)
