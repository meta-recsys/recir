# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Sum the column blocks of a concatenation separately."""

import contextlib
import logging
from typing import Any

import torch
from torch import fx

logger: logging.Logger = logging.getLogger(__name__)

_CAT: Any = torch.ops.aten.cat.default
_SUM: Any = torch.ops.aten.sum.dim_IntList


def _summed_blocks(node: fx.Node) -> tuple[list[fx.Node], bool] | None:
    """The column blocks whose concatenation ``node`` sums over rows, and the
    sum's ``keepdim``."""
    if (
        node.op != "call_function"
        or node.target != _SUM
        or node.kwargs.get("dtype") is not None
        or len(node.args) < 2
    ):
        return None
    concat, dims = node.args[0], node.args[1]
    keepdim = node.args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False)
    if not isinstance(concat, fx.Node) or concat.target != _CAT:
        return None
    value = concat.meta.get("val")
    parts = concat.args[0]
    cat_dim = concat.args[1] if len(concat.args) > 1 else concat.kwargs.get("dim", 0)
    if (
        not isinstance(value, torch.Tensor)
        or value.dim() != 2
        or not isinstance(dims, (list, tuple))
        or list(dims) not in ([0], [-2])
        or cat_dim not in (1, -1)
        or not isinstance(parts, (list, tuple))
    ):
        return None
    blocks = [part for part in parts if isinstance(part, fx.Node)]
    if len(blocks) != len(parts) or not all(
        isinstance(block.meta.get("val"), torch.Tensor)
        and block.meta["val"].dtype == value.dtype
        for block in blocks
    ):
        return None
    return blocks, bool(keepdim)


def split_concat_row_sums(gm: fx.GraphModule) -> fx.GraphModule:
    """Rewrite ``sum(cat(blocks, dim=1), dim=0)`` as ``cat([sum(b, dim=0) ...])``.

    The bias gradient of a projection whose output is split into blocks has this
    form. XLA fuses the concatenation into the reduction, and on TPU that fusion
    can read at a third of HBM bandwidth: four bf16 [133640, 512] blocks take
    0.45 ms concatenated and 0.17 ms summed separately. The blocks are
    accumulated in a different order, so the last bit of a sum can change.
    """
    graph = gm.graph
    rewritten = 0
    for node in list(graph.nodes):
        match = _summed_blocks(node)
        if match is None:
            continue
        blocks, keepdim = match
        concat = node.args[0]
        assert isinstance(concat, fx.Node)
        mode = (
            getattr(concat.meta["val"], "fake_mode", None) or contextlib.nullcontext()
        )
        with graph.inserting_before(node):
            sums = []
            for block in blocks:
                block_sum = graph.call_function(_SUM, (block, [0], keepdim))
                with mode:
                    block_sum.meta["val"] = _SUM(block.meta["val"], [0], keepdim)
                sums.append(block_sum)
            summed = graph.call_function(_CAT, (sums, 1 if keepdim else 0))
            summed.meta = node.meta.copy()
        node.replace_all_uses_with(summed)
        graph.erase_node(node)
        if not concat.users:
            graph.erase_node(concat)
        rewritten += 1
    graph.lint()
    gm.recompile()
    gm.meta["concat_row_sum_count"] = rewritten
    logger.info("[CONCAT_ROW_SUMS] rewritten=%d", rewritten)
    return gm
