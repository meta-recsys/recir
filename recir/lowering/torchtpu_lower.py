# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Partial-lowering of FX subgraphs onto the torch_tpu runtime.

Mirrors :mod:`recir.lowering.xla_lower` but lowers each compatible subgraph
via torch_tpu's ``TpuBackend`` (the same backend ``torch.compile(backend="tpu")``
uses) instead of torchax/JAX. ``TpuBackend`` runs ``aot_autograd`` ->
``StaticCompiler`` -> ``fx_to_mlir`` -> ``compile_mlir`` -> a PJRT executable.

This module deliberately does NOT import torchax/JAX so the torch_tpu path
carries no JAX dependency.
"""

from __future__ import annotations

import logging
import operator
from typing import Any, Callable, List, Optional, Tuple

import torch
from recir.lowering.lowering_utils import segment_and_lower, strip_profiler_nodes
from recir.ops.decompositions import SPMD_DECOMP_TABLE
from torch._decomp import get_decompositions

logger: logging.Logger = logging.getLogger(__name__)

# torch_tpu lacks eager kernels for some ATen ops the HSTU multitask loss emits
# (binary_cross_entropy_with_logits, whose TPU path hits log_sigmoid_forward).
# Register their decompositions into recir's SPMD_DECOMP_TABLE — which make_fx
# consults at trace time (recir.api) — so they lower to supported ops instead of
# hitting the eager TPU dispatch. Registered at import of this module, i.e.
# whenever the torch_tpu backend is imported, which is before any trace.
SPMD_DECOMP_TABLE.update(
    get_decompositions(
        [
            torch.ops.aten.binary_cross_entropy_with_logits,
            torch.ops.aten.log_sigmoid_forward,
            torch.ops.aten.log_sigmoid_backward,
        ]
    )
)


def _is_torchtpu_compatible(node: torch.fx.Node) -> Tuple[bool, str]:
    """Per-node check: can this node be absorbed into a torch_tpu subgraph?

    torch_tpu lowers a subgraph by re-running it op-by-op on the tpu device
    (``fx_to_mlir``), so we let it attempt to lower EVERY node except the
    embedding graph-break markers (``xla_graph_break``), which must stay eager
    (they call back into the eager torchrec/SparseCore embedding). Any op
    torch_tpu cannot lower then surfaces as a loud ``fx_to_mlir``/compile error
    rather than being silently kept eager.

    This is intentionally more permissive than the XLA path's
    ``_is_xla_compatible``, which conservatively gates on meta-kernel
    registration and op allowlists. If a class of ops needs to stay eager on
    torch_tpu (e.g. ``_c10d_functional`` collectives), add a targeted reject
    here. ``profiler::`` ops never reach this check: ``partial_lower_torchtpu``
    strips them first.
    """
    # Override-dispatch (embedding graph-break) nodes must never be absorbed.
    if node.meta.get("xla_graph_break", False):
        return False, "graph break (override dispatch)"

    return True, ""


def _fresh_empty_tensors(gm: torch.fx.GraphModule) -> int:
    """Route every use of an empty intermediate through a fresh ``new_empty``.

    TorchTPU hands a view across its collective partitions as the whole
    underlying buffer, which an empty view (e.g. a zero-length slice of a
    padded batch tensor) does not match. An empty tensor holds no data, so a
    fresh one of the same shape is equivalent. Returns the number replaced.
    """
    replaced = 0
    for node in list(gm.graph.nodes):
        val = node.meta.get("val")
        if (
            node.op != "call_function"
            or not isinstance(val, torch.Tensor)
            or val.numel() != 0
            or not node.users
        ):
            continue
        with gm.graph.inserting_after(node):
            fresh = gm.graph.call_function(
                torch.ops.aten.new_empty.default, (node, list(val.shape))
            )
        fresh.meta["val"] = val
        node.replace_all_uses_with(
            fresh, delete_user_cb=lambda user, fresh=fresh: user is not fresh
        )
        replaced += 1
    if replaced:
        gm.recompile()
    return replaced


def _partition_ids(gm: torch.fx.GraphModule) -> dict[torch.fx.Node, int]:
    """The partition ``TpuBackend``'s split compiler puts each node in.

    Mirrors ``split_compiler``: every collective gets a partition of its own,
    and its ``wait_tensor`` and ``getitem`` users join it.
    """
    ids: dict[torch.fx.Node, int] = {}
    current = 0
    for node in gm.graph.nodes:
        if node.op in ("placeholder", "output"):
            continue
        name = (
            node.target.name() if isinstance(node.target, torch._ops.OpOverload) else ""
        )
        source = node.args[0] if node.args else None
        if isinstance(source, torch.fx.Node) and (
            name == "_c10d_functional::wait_tensor"
            or (
                node.target is operator.getitem
                and source in ids
                and str(source.target).startswith("_c10d_functional.")
            )
        ):
            ids[node] = ids.get(source, current)
        elif name.startswith("_c10d_functional::"):
            ids[node] = current + 1
            current += 2
        else:
            ids[node] = current
    return ids


def _is_view(node: torch.fx.Node) -> bool:
    target = node.target
    if target is operator.getitem and isinstance(node.args[0], torch.fx.Node):
        target = node.args[0].target
    if not isinstance(target, torch._ops.OpOverload):
        return False
    returns = target._schema.returns
    return bool(returns) and any(
        r.alias_info is not None and not r.alias_info.is_write for r in returns
    )


def _copy_cross_partition_views(gm: torch.fx.GraphModule) -> int:
    """Copy each view used in a later ``TpuBackend`` partition, before it leaves.

    The split compiler sizes a partition's inputs by running the graph on fake
    tensors, which keep a view's base: a column of a ``[20, 65]`` slice of a
    ``[20, 580]`` tensor is compiled as a 580-stride read. The partition that
    produces the view returns it as a dense buffer of its own, so the consumer
    rejects it at run time (``RuntimeProgramInputMismatch``). A contiguous copy
    made in the producer has the same layout both ways. Returns the number
    copied.
    """
    ids = _partition_ids(gm)
    copied = 0
    for node in list(gm.graph.nodes):
        if node not in ids or not _is_view(node):
            continue
        later = [u for u in node.users if u in ids and ids[u] != ids[node]]
        if not later:
            continue
        with gm.graph.inserting_after(node):
            copy = gm.graph.call_function(
                torch.ops.aten.clone.default,
                (node,),
                {"memory_format": torch.contiguous_format},
            )
        if "val" in node.meta:
            copy.meta["val"] = node.meta["val"]
        for user in later:
            user.replace_input_with(node, copy)
        copied += 1
    if copied:
        gm.recompile()
    return copied


def _shape_mismatches(gm: torch.fx.GraphModule, args: List[Any]) -> List[str]:
    """Inputs whose shape differs from the shape traced for their placeholder.

    ``TpuBackend`` compiles from the first call's inputs, but sizes recorded
    in the graph (views, ``select_backward``'s ``input_sizes``) come from the
    trace, so the two must agree.
    """
    mismatches = []
    for node, arg in zip(gm.graph.find_nodes(op="placeholder"), args):
        val = node.meta.get("val")
        if (
            isinstance(arg, torch.Tensor)
            and isinstance(val, torch.Tensor)
            and arg.shape != val.shape
        ):
            mismatches.append(
                f"{node.name}: traced {tuple(val.shape)}, got {tuple(arg.shape)}"
            )
    return mismatches


class _TorchTPUModule(torch.nn.Module):
    """Wraps an ``fx.GraphModule`` subgraph for lazy torch_tpu compilation.

    On the first forward call the subgraph is compiled via torch_tpu's
    ``TpuBackend`` and the resulting callable PJRT executable is cached;
    subsequent calls reuse it. Inputs are moved to the ``tpu`` device at the
    subgraph boundary (the runtime feeds host/state tensors per recir's
    ``DeviceStateManager`` convention).
    """

    def __init__(self, gm: torch.fx.GraphModule) -> None:
        super().__init__()
        self.gm = gm
        self.compiled: Optional[Callable[..., Any]] = None
        logger.info(
            "torch_tpu subgraph: %d empty intermediates made fresh, %d views "
            "copied before leaving their partition",
            _fresh_empty_tensors(gm),
            _copy_cross_partition_views(gm),
        )

    @staticmethod
    def _to_tpu(args: Tuple[Any, ...]) -> List[Any]:
        out: List[Any] = []
        for a in args:
            if isinstance(a, torch.Tensor) and a.numel() == 0:
                # Same buffer mismatch as _fresh_empty_tensors, for an input.
                out.append(torch.empty(a.shape, dtype=a.dtype, device="tpu"))
            elif isinstance(a, torch.Tensor) and a.device.type != "tpu":
                out.append(a.to("tpu"))
            else:
                out.append(a)
        return out

    def forward(self, *args: Any, tag: str = "") -> Any:
        tpu_args = self._to_tpu(args)
        if self.compiled is None:
            # Local import: keep torch_tpu out of import-time deps and only
            # require it when a torch_tpu subgraph is actually compiled.
            # pyre-ignore[21]: torch_tpu is a TPU-runtime-only package (no buck
            # dep), so it is unresolvable at build/type-check time.
            from torch_tpu._internal.compile._backend import TpuBackend

            for a in tpu_args:
                if isinstance(a, torch.Tensor):
                    assert not a.is_meta, (
                        "torch_tpu backend requires real (non-fake) tensors at "
                        "compile time; got a meta/fake tensor."
                    )
            mismatches = _shape_mismatches(self.gm, list(args))
            if mismatches:
                raise ValueError(
                    f"torch_tpu subgraph ({tag}) inputs differ from their traced "
                    f"shapes:\n" + "\n".join(mismatches)
                )
            logger.info(
                "Compiling subgraph (%s) via torch_tpu TpuBackend; %d empty inputs",
                tag,
                sum(isinstance(a, torch.Tensor) and a.numel() == 0 for a in args),
            )
            self.compiled = TpuBackend()(self.gm, list(tpu_args))
            logger.info("Completed torch_tpu compilation for subgraph (%s)", tag)
        with torch.profiler.record_function(tag):
            assert self.compiled is not None
            return self.compiled(*tpu_args)


def partial_lower_torchtpu(
    gm: torch.fx.GraphModule,
    node_predicate: Callable[[torch.fx.Node], bool] = lambda x: True,
    subgraph_predicate: Callable[[List[torch.fx.Node]], bool] = lambda x: True,
    dumper: Callable[[str], str] = lambda x: "subgraph",
) -> torch.fx.GraphModule:
    """Segment ``gm`` and lower each compatible subgraph via ``TpuBackend``.

    Reuses the backend-neutral :func:`segment_and_lower` algorithm (shared with
    the Inductor and XLA partial-lowering paths).
    """
    gm = strip_profiler_nodes(gm)
    return segment_and_lower(
        gm,
        compatibility_fn=_is_torchtpu_compatible,
        module_factory=lambda sub_gm: _TorchTPUModule(sub_gm),
        subgraph_prefix="torchtpu_subgraph",
        node_predicate=node_predicate,
        subgraph_predicate=subgraph_predicate,
        dumper=dumper,
    )
