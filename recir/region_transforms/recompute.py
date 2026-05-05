# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""RecomputeTransform — activation checkpointing via region recomputation."""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Callable

import torch
import torch.fx as fx
import torch.nn as nn
from recir.graph_utils import OP
from recir.region_transforms.base import RegionTransform

logger: logging.Logger = logging.getLogger(__name__)


class RecomputeTransform(RegionTransform):
    """Activation checkpointing: save region inputs, re-execute all ops
    in backward.
    """

    def __init__(self, submodule_fqns: set[str]) -> None:
        self.submodule_fqns = submodule_fqns

    def detect_regions(
        self,
        gm: fx.GraphModule,
        called_modules: OrderedDict[str, tuple[nn.Module, tuple[Any, ...]]],
    ) -> list[set[fx.Node]]:
        fqn_to_nodes: dict[str, set[fx.Node]] = {}
        for node in gm.graph.nodes:
            if node.op not in (OP.CALL_FUNCTION, OP.CALL_METHOD):
                continue
            nn_module_stack = node.meta.get("nn_module_stack", None)
            if nn_module_stack is None:
                continue
            # nn_module_stack is an OrderedDict of
            # {tracer_key: (module_fqn, module_class_fqn)} as populated by
            # torch.fx.experimental.proxy_tensor.PythonKeyTracer.create_node.
            # The dict keys are opaque tracer identifiers; the FQN to match
            # against is the first element of the value tuple.
            for fqn, _ in nn_module_stack.values():
                if fqn in self.submodule_fqns:
                    fqn_to_nodes.setdefault(fqn, set()).add(node)
        regions: list[set[fx.Node]] = []
        for fqn in self.submodule_fqns:
            nodes = fqn_to_nodes.get(fqn)
            if nodes:
                regions.append(nodes)
            else:
                logger.warning(
                    "RecomputeTransform: no nodes found for submodule '%s'",
                    fqn,
                )
        return regions

    def wrap_region(
        self,
        region_gm: fx.GraphModule,
    ) -> Callable[..., Any]:
        def _apply(*inputs: torch.Tensor) -> Any:
            return torch.utils.checkpoint.checkpoint(
                region_gm,
                *inputs,
                use_reentrant=True,
            )

        return _apply
