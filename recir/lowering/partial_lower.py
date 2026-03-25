# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# This file is copied from Meta internal repo and is not synced with the
# internal version. Once the internal version is fully mature, we should
# upstream again and retire the internal version. @yifuwang

# pyre-strict

import logging
from typing import Callable, List, Optional, Tuple

import torch
from functorch import make_fx
from recir.lowering.lowering_utils import segment_and_lower
from torch._inductor.compile_fx import compile_fx_inner
from torch._inductor.decomposition import select_decomp_table


logger: logging.Logger = logging.getLogger(__name__)


class _InductorModule(torch.nn.Module):
    def __init__(self, gm: torch.fx.GraphModule) -> None:
        super().__init__()
        self.gm = gm
        self.compiled: Optional[Callable[[List[torch.Tensor]], List[torch.Tensor]]] = (
            None
        )

    def forward(self, *args: torch.Tensor, tag: str) -> List[torch.Tensor]:
        if self.compiled is None:
            inductor_decompositions = select_decomp_table()
            # TODO: figure out why turning on cudagraphs cause exceptions.
            decomp_gm = make_fx(self.gm, decomposition_table=inductor_decompositions)(
                *args
            )
            logger.info("Lowering subgraph (%s) to Inductor...", tag)
            self.compiled = compile_fx_inner(
                decomp_gm,
                list(args),
                cudagraphs=False,  # pyre-ignore
            )
            logger.info("Completed lowering subgraph (%s) to Inductor", tag)
        with torch.profiler.record_function(tag):
            assert self.compiled is not None
            return self.compiled(list(args))


def _is_inductor_compatible(node: torch.fx.Node) -> Tuple[bool, str]:
    # `has_tag` is not supported yet
    # if has_tag(node, "non_lowerable"):

    if node.target in (
        torch.ops.aten._fused_adam_.default,
        torch.ops.aten._fused_adam.default,
        torch.ops.aten._foreach_add_.Scalar,
        torch.ops.aten._foreach_add.Scalar,
    ):
        return False, "fused adam is not supported yet"

    # TODO(yifu): apparently having a meta kernel is not a necessary
    # condition for Inductor compatiblity. We should refine the check.
    # Sneaking this one in for now to support comm_fusion_with_cat.
    if node.target == torch.ops.aten.flatten.using_ints:
        return True, ""

    if isinstance(node.target, torch._ops.OpOverload):
        if not node.target.has_kernel_for_dispatch_key(torch._C.DispatchKey.Meta):
            return False, f"{node.target} doesn't have a meta kernel registered"
    return True, ""


def partial_lower(
    gm: torch.fx.GraphModule,
    node_predicate: Callable[[torch.fx.Node], bool] = lambda x: True,
    subgraph_predicate: Callable[[List[torch.fx.Node]], bool] = lambda x: True,
    dumper: Callable[[str], str] = lambda x: "subgraph",
) -> torch.fx.GraphModule:
    """
    Lower Inductor compatible portions of the graph module to Inductor.

    Args:
        node_predicate: user predicate for determining whether to consider a node for
            lowering.
        subgraph_predicate: user predicate for determining whether to consider a list of
            candidate nodes for lowering.
        dumper: a callback for dumping subgraphs for human digestion. For exmaple, it
            can be a function that writes to disk/blob storage and returns the
            path/handle. The returned path/handle for each subgraph will be made
            available in the subgraph call node in the parent graph, as well as the
            label of the profiler block for the subgraph.
    """
    return segment_and_lower(
        gm,
        compatibility_fn=_is_inductor_compatible,
        module_factory=_InductorModule,
        subgraph_prefix="subgraph",
        node_predicate=node_predicate,
        subgraph_predicate=subgraph_predicate,
        dumper=dumper,
    )
