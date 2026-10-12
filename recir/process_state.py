# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Canonical host-process state produced by RecIR compilation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.fx as fx
from recir.distributed.sharding_spec import FSDPWeightShardSpec, get_state_layout
from torch.utils import _pytree as pytree


@dataclass
class CompiledProcessState:
    """Canonical host-process state for explicit-collective FSDP.

    Model tensors are keyed by module FQN. Optimizer tensors remain nested under
    their owning parameter FQN and optimizer state name. The global-shape maps
    contain only leaves that RecIR lays out as process shards.
    """

    model: dict[str, torch.Tensor]
    optimizer: dict[str, dict[str, Any]]
    model_global_shapes: dict[str, tuple[int, ...]]
    optimizer_global_shapes: dict[str, dict[str, tuple[int, ...]]]


def _compiled_process_state_entries(
    params_and_buffers: dict[str, Any],
    named_states: dict[str, Any],
) -> list[tuple[str, str, str | None, torch.Tensor]]:
    entries: list[tuple[str, str, str | None, torch.Tensor]] = []
    for fqn, value in params_and_buffers.items():
        for leaf in pytree.tree_leaves(value):
            if isinstance(leaf, torch.Tensor):
                entries.append(("model", fqn, None, leaf))
    for fqn, state in named_states.items():
        if not isinstance(state, dict):
            for leaf in pytree.tree_leaves(state):
                if isinstance(leaf, torch.Tensor):
                    entries.append(("optimizer", fqn, None, leaf))
            continue
        for state_name, value in state.items():
            stable_state_name = state_name if isinstance(state_name, str) else None
            for leaf in pytree.tree_leaves(value):
                if isinstance(leaf, torch.Tensor):
                    entries.append(("optimizer", fqn, stable_state_name, leaf))
    return entries


def _build_compiled_process_state(
    gm: fx.GraphModule,
    params_and_buffers: dict[str, Any],
    named_states: dict[str, Any],
) -> CompiledProcessState | None:
    entries = _compiled_process_state_entries(params_and_buffers, named_states)
    state_placeholders = [
        node
        for node in gm.graph.nodes
        if node.op == "placeholder"
        and node.meta.get("is_state_input", False)
        and isinstance(node.meta.get("val"), torch.Tensor)
    ]
    if not any(get_state_layout(node) is not None for node in state_placeholders):
        return None
    if len(state_placeholders) != len(entries):
        raise RuntimeError(
            "Compiled process-state FQNs do not align with state placeholders: "
            f"fqns={len(entries)}, placeholders={len(state_placeholders)}"
        )

    model_global_shapes: dict[str, tuple[int, ...]] = {}
    optimizer_global_shapes: dict[str, dict[str, tuple[int, ...]]] = {}
    for node, (kind, fqn, state_name, tensor) in zip(state_placeholders, entries):
        node_kind = node.meta.get("recir_state_kind", "model")
        node_fqn = node.meta.get(
            "recir_state_fqn",
            node.meta.get("fqn"),
        )
        node_state_name = node.meta.get("recir_state_name")
        if (node_kind, node_fqn, node_state_name) != (kind, fqn, state_name):
            raise RuntimeError(
                "Compiled process-state placeholder metadata does not match its "
                "canonical state entry: "
                f"placeholder={(node_kind, node_fqn, node_state_name)!r}, "
                f"entry={(kind, fqn, state_name)!r}"
            )
        layout = get_state_layout(node)
        if layout is None:
            continue
        if not isinstance(layout, FSDPWeightShardSpec):
            raise RuntimeError(
                f"Unsupported compiled process-state layout for {fqn!r}: {layout!r}"
            )
        if not 0 <= layout.shard_dim < tensor.ndim:
            raise RuntimeError(
                f"Compiled process-state tensor {fqn!r} cannot use {layout!r}"
            )
        global_shape = [int(dim) for dim in tensor.shape]
        global_shape[layout.shard_dim] *= layout.process_world_size
        shape = tuple(global_shape)
        if kind == "model":
            model_global_shapes[fqn] = shape
            continue
        if state_name is None:
            raise RuntimeError(
                "Compiled process optimizer state requires a stable string key: "
                f"parameter={fqn!r}"
            )
        optimizer_global_shapes.setdefault(fqn, {})[state_name] = shape

    model = {
        fqn: value
        for fqn, value in params_and_buffers.items()
        if isinstance(value, torch.Tensor)
    }
    optimizer = {
        fqn: dict(state)
        for fqn, state in named_states.items()
        if isinstance(state, dict)
    }
    return CompiledProcessState(
        model=model,
        optimizer=optimizer,
        model_global_shapes=model_global_shapes,
        optimizer_global_shapes=optimizer_global_shapes,
    )
