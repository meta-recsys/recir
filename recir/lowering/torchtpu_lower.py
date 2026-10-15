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
from typing import Any, Callable, List, Optional, Tuple

import torch
from recir.lowering.lowering_utils import segment_and_lower
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
    torch_tpu (e.g. ``_c10d_functional`` collectives or ``profiler::`` ops),
    add a targeted reject here.
    """
    # Override-dispatch (embedding graph-break) nodes must never be absorbed.
    if node.meta.get("xla_graph_break", False):
        return False, "graph break (override dispatch)"

    return True, ""


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

    @staticmethod
    def _to_tpu(args: Tuple[Any, ...]) -> List[Any]:
        out: List[Any] = []
        for a in args:
            if isinstance(a, torch.Tensor) and a.device.type != "tpu":
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
            logger.info("Compiling subgraph (%s) via torch_tpu TpuBackend...", tag)
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
    return segment_and_lower(
        gm,
        compatibility_fn=_is_torchtpu_compatible,
        module_factory=lambda sub_gm: _TorchTPUModule(sub_gm),
        subgraph_prefix="torchtpu_subgraph",
        node_predicate=node_predicate,
        subgraph_predicate=subgraph_predicate,
        dumper=dumper,
    )
