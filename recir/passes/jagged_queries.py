# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Compose target-only jagged selections after tracing backward."""

import logging
import operator
from dataclasses import dataclass
from typing import Any

import recir.ops.jagged_ops  # noqa: F401
import recir.ops.packed_jagged_ops  # noqa: F401
import torch
from torch import fx

logger: logging.Logger = logging.getLogger(__name__)


def _shape(node: fx.Node) -> tuple[int, ...] | None:
    value = node.meta.get("val")
    if not isinstance(value, torch.Tensor):
        return None
    if not all(isinstance(dim, int) for dim in value.shape):
        return None
    return tuple(value.shape)


def _full_view_source(node: fx.Node) -> tuple[fx.Node, list[fx.Node]]:
    chain: list[fx.Node] = []
    aliases = (
        torch.ops.aten.alias.default,
        torch.ops.aten.clone.default,
        torch.ops.aten.detach.default,
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten.slice.Tensor,
    )
    while node.op == "call_function" and len(node.users) == 1:
        source: Any = None
        extra: list[fx.Node] = []
        if node.target in aliases:
            source = node.args[0]
        elif node.target == operator.getitem and node.args[1] == 0:
            split = node.args[0]
            sizes = (
                split.args[1]
                if isinstance(split, fx.Node) and len(split.args) > 1
                else None
            )
            if (
                isinstance(split, fx.Node)
                and split.target == torch.ops.aten.split_with_sizes.default
                and len(split.users) == 1
                and isinstance(sizes, (tuple, list))
                and len(sizes) == 1
            ):
                source = split.args[0]
                extra = [split]
        if not isinstance(source, fx.Node) or _shape(node) != _shape(source):
            break
        if _shape(node) is None:
            break
        if node.meta["val"].dtype != source.meta["val"].dtype:
            break
        chain.extend([node, *extra])
        node = source
    return node, chain


@dataclass
class _QuerySelection:
    query: fx.Node
    obsolete: list[fx.Node]
    prefix_nodes: list[fx.Node]
    args: tuple[Any, ...]


def _prefix_source(
    left: fx.Node, concat: fx.Node
) -> tuple[fx.Node, int, bool, list[fx.Node]]:
    left_capacity = left.meta["val"].shape[0]
    left_is_prefix = False
    prefix, prefix_views = _full_view_source(left)
    if prefix.target == operator.getitem and prefix.args[1] == 0:
        source_split = prefix.args[0]
        if (
            isinstance(source_split, fx.Node)
            and source_split.target == torch.ops.recir.split_2d_jagged.default
            and len(source_split.args) == 5
            and source_split.args[1] is concat.args[1]
            and source_split.args[2] is concat.args[3]
            and isinstance(source_split.args[3], int)
            and isinstance(source_split.args[0], fx.Node)
        ):
            left = source_split.args[0]
            left_capacity = source_split.args[3]
            left_is_prefix = True
    return left, left_capacity, left_is_prefix, prefix_views + [prefix]


def _match_query(split: fx.Node) -> _QuerySelection | None:
    if (
        split.op != "call_function"
        or split.target != torch.ops.recir.split_2d_jagged.default
        or len(split.args) != 5
        or not isinstance(split.args[0], fx.Node)
    ):
        return None
    queries = [
        user
        for user in split.users
        if user.op == "call_function"
        and user.target == operator.getitem
        and user.args[1] == 1
    ]
    if len(queries) != 1:
        return None
    query = queries[0]
    unused = [user for user in split.users if user is not query]
    if any(
        user.op != "call_function"
        or user.target != operator.getitem
        or user.args[1] != 0
        or user.users
        for user in unused
    ):
        return None
    return _selection_inputs(split, query, unused)


def _selection_inputs(
    split: fx.Node, query: fx.Node, unused: list[fx.Node]
) -> _QuerySelection | None:
    source = split.args[0]
    if not isinstance(source, fx.Node):
        return None
    concat, views = _full_view_source(source)
    if (
        concat.op != "call_function"
        or concat.target != torch.ops.recir.concat_2d_jagged.default
        or len(concat.users) != 1
        or len(concat.args) not in (4, 5)
    ):
        return None
    shape = _shape(concat)
    left = concat.args[0]
    right = concat.args[2]
    if shape is None or not isinstance(left, fx.Node) or not isinstance(right, fx.Node):
        return None
    left_shape, right_shape = _shape(left), _shape(right)
    tensors: list[torch.Tensor] = []
    for node in (left, right, query):
        value = node.meta.get("val")
        if not isinstance(value, torch.Tensor):
            return None
        tensors.append(value)
    if (
        left_shape is None
        or right_shape is None
        or any(value.dtype != tensors[0].dtype for value in tensors)
        or tensors[0].dtype not in (torch.bfloat16, torch.float32)
    ):
        return None
    capacity = concat.args[4] if len(concat.args) == 5 else shape[0]
    capacity = shape[0] if capacity is None else capacity
    if not isinstance(capacity, int) or not isinstance(split.args[4], int):
        return None
    if max(capacity, split.args[4], left_shape[0] + right_shape[0]) >= 1 << 31:
        return None
    left, left_capacity, left_is_prefix, prefix_nodes = _prefix_source(left, concat)
    if not isinstance(left, fx.Node) or _shape(left) is None:
        return None
    if max(capacity, split.args[4], left.meta["val"].shape[0]) >= 1 << 31:
        return None
    return _QuerySelection(
        query,
        [*unused, split, *views, concat],
        prefix_nodes if left_is_prefix else [],
        (
            left,
            concat.args[1],
            right,
            concat.args[3],
            capacity,
            split.args[1],
            split.args[2],
            split.args[4],
            left_capacity,
            left_is_prefix,
        ),
    )


def fuse_jagged_queries(gm: fx.GraphModule) -> fx.GraphModule:
    graph = gm.graph
    fused = 0
    prefix_sources = 0
    for split in list(graph.nodes):
        selection = _match_query(split)
        if selection is None:
            continue
        query = selection.query
        with graph.inserting_before(query):
            selected = graph.call_function(
                torch.ops.recir.jagged_concat_split_right.default, selection.args
            )
            selected.meta = query.meta.copy()
        query.replace_all_uses_with(selected)
        graph.erase_node(query)
        for node in [*selection.obsolete, *selection.prefix_nodes]:
            if not node.users:
                graph.erase_node(node)
        fused += 1
        prefix_sources += int(selection.args[-1])
    graph.lint()
    gm.recompile()
    gm.meta["jagged_query_fusion_count"] = fused
    logger.info(
        "[JAGGED_QUERY_FUSION] matches=%d prefix_sources=%d", fused, prefix_sources
    )
    return gm
