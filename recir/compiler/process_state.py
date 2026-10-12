# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Canonical host-process state produced by RecIR compilation."""

from __future__ import annotations

from collections.abc import Sequence
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


def _compute_state_fqns(structure: Any) -> list[str]:
    """Return a pytree keypath for each leaf in canonical state order."""
    try:
        leaves_with_path, _ = pytree.tree_flatten_with_path(structure)
        return [pytree.keystr(keypath) for keypath, _ in leaves_with_path]
    except Exception:
        return []


def _is_fsdp_sharded_leaf(
    fqn: str, leaf: Any, sharded_parameter_fqns: set[str]
) -> bool:
    return (
        fqn in sharded_parameter_fqns
        and isinstance(leaf, torch.Tensor)
        and leaf.ndim >= 1
    )


def _build_fsdp_sharded_flags(
    params_and_buffers: dict[str, Any],
    named_states: dict[str, Any],
    sharded_parameter_fqns: set[str],
    optimizer_state_only_sharded_keys: frozenset[tuple[str, str]] = frozenset(),
) -> list[bool]:
    """Return per-tensor FSDP flags in the same pytree order as flat state."""
    params_buffers_flags = {
        fqn: pytree.tree_map(
            lambda leaf, fqn=fqn: _is_fsdp_sharded_leaf(
                fqn, leaf, sharded_parameter_fqns
            ),
            value,
        )
        for fqn, value in params_and_buffers.items()
    }
    named_state_flags: dict[str, Any] = {}
    for fqn, state in named_states.items():
        if not isinstance(state, dict):
            named_state_flags[fqn] = pytree.tree_map(
                lambda leaf, fqn=fqn: _is_fsdp_sharded_leaf(
                    fqn, leaf, sharded_parameter_fqns
                ),
                state,
            )
            continue
        state_flags: dict[str, Any] = {}
        for state_key, state_value in state.items():
            state_flags[state_key] = pytree.tree_map(
                lambda leaf, fqn=fqn, state_key=state_key: (
                    _is_fsdp_sharded_leaf(fqn, leaf, sharded_parameter_fqns)
                    or (
                        (fqn, state_key) in optimizer_state_only_sharded_keys
                        and isinstance(leaf, torch.Tensor)
                        and leaf.ndim >= 1
                    )
                ),
                state_value,
            )
        named_state_flags[fqn] = state_flags
    flat_values, value_spec = pytree.tree_flatten([params_and_buffers, named_states])
    flat_flags, flag_spec = pytree.tree_flatten(
        [params_buffers_flags, named_state_flags]
    )
    if value_spec != flag_spec:
        raise RuntimeError(
            "FSDP flag tree must mirror flat_state TreeSpec: "
            f"values={value_spec}, flags={flag_spec}"
        )
    return [
        bool(flat_flags[index])
        for index, value in enumerate(flat_values)
        if isinstance(value, torch.Tensor)
    ]


def _align_fsdp_sharded_flags(
    base_flat_state: Sequence[Any],
    base_state_flags: Sequence[bool],
    final_flat_state: Sequence[Any],
) -> list[bool]:
    """Carry placement by identity; new transform state is replicated."""
    base_tensors = [
        state_value
        for state_value in base_flat_state
        if isinstance(state_value, torch.Tensor)
    ]
    if len(base_tensors) != len(base_state_flags):
        raise RuntimeError(
            "FSDP flags must mirror base flat_state tensor leaves: "
            f"flags={len(base_state_flags)}, tensors={len(base_tensors)}"
        )
    flags_by_identity: dict[int, bool] = {}
    for tensor, is_sharded in zip(base_tensors, base_state_flags):
        tensor_identity = id(tensor)
        if (
            tensor_identity in flags_by_identity
            and flags_by_identity[tensor_identity] != is_sharded
        ):
            raise RuntimeError("Aliased state tensor has conflicting FSDP placements")
        flags_by_identity[tensor_identity] = is_sharded
    final_flags: list[bool] = []
    for state_value in final_flat_state:
        if not isinstance(state_value, torch.Tensor):
            continue
        final_flags.append(flags_by_identity.get(id(state_value), False))
    return final_flags


def _annotate_fsdp_state_input_sharding(
    gm: fx.GraphModule,
    flat_state: Sequence[Any],
    state_flags: Sequence[bool],
) -> None:
    """Attach placement to the state portion of RecIR's runtime input."""
    tensor_placeholders = [
        node
        for node in gm.graph.nodes
        if node.op == "placeholder"
        and isinstance(node.meta.get("val", None), torch.Tensor)
    ]
    untagged_placeholders = [
        node.name for node in tensor_placeholders if "is_state_input" not in node.meta
    ]
    if untagged_placeholders:
        raise RuntimeError(
            "FSDP state input annotation requires is_state_input metadata on every "
            f"tensor placeholder; missing={untagged_placeholders!r}"
        )
    state_placeholders = [
        node for node in tensor_placeholders if node.meta["is_state_input"]
    ]
    state_values = [value for value in flat_state if isinstance(value, torch.Tensor)]
    if len(state_values) != len(state_flags):
        raise RuntimeError(
            "FSDP flags must mirror final flat_state tensor leaves: "
            f"state={len(state_values)}, flags={len(state_flags)}"
        )
    if len(state_placeholders) != len(state_values):
        raise RuntimeError(
            "Compiled graph state inputs must mirror final flat_state: "
            f"inputs={len(state_placeholders)}, state={len(state_values)}"
        )
    for node in tensor_placeholders:
        node.meta["is_fsdp_sharded"] = False
        node.meta.pop("fsdp_shard_dim", None)
    for leaf, state_value, is_sharded in zip(
        state_placeholders,
        state_values,
        state_flags,
    ):
        state_identity = id(state_value)
        prior_identity = leaf.meta.get("recir_state_input_identity")
        if prior_identity is not None and prior_identity != state_identity:
            raise RuntimeError(
                "Compiled graph state input order no longer mirrors flat_state"
            )
        leaf.meta["recir_state_input_identity"] = state_identity
        leaf.meta["is_fsdp_sharded"] = is_sharded
        if is_sharded:
            leaf.meta["fsdp_shard_dim"] = 0


def _annotate_fsdp_state_output_sharding(
    gm: fx.GraphModule,
    flat_state: Sequence[Any],
    state_flags: Sequence[bool],
) -> None:
    """Attach placement to the state portion of RecIR's runtime output."""
    if not state_flags:
        return
    output_node = next(node for node in gm.graph.nodes if node.op == "output")
    graph_outputs = output_node.args[0]
    if not isinstance(graph_outputs, (list, tuple)) or not graph_outputs:
        raise RuntimeError(
            "Compiled graph output must contain a user result followed by state"
        )
    graph_output_leaves = pytree.tree_leaves(graph_outputs)
    if len(graph_output_leaves) < len(flat_state):
        raise RuntimeError(
            "Compiled graph output must contain every final flat_state leaf: "
            f"outputs={len(graph_output_leaves)}, state={len(flat_state)}"
        )
    raw_state_outputs = graph_output_leaves[-len(flat_state) :] if flat_state else []
    state_output_leaves = [
        leaf
        for leaf in raw_state_outputs
        if isinstance(leaf, fx.Node)
        and isinstance(leaf.meta.get("val", None), torch.Tensor)
    ]
    state_values = [value for value in flat_state if isinstance(value, torch.Tensor)]
    if len(state_values) != len(state_flags):
        raise RuntimeError(
            "FSDP flags must mirror final flat_state tensor leaves: "
            f"state={len(state_values)}, flags={len(state_flags)}"
        )
    if len(state_output_leaves) != len(state_flags):
        raise RuntimeError(
            "Compiled graph state outputs must mirror final flat_state: "
            f"outputs={len(state_output_leaves)}, flags={len(state_flags)}"
        )
    for leaf, state_value, is_sharded in zip(
        state_output_leaves,
        state_values,
        state_flags,
    ):
        state_identity = id(state_value)
        prior_identity = leaf.meta.get("recir_state_output_identity")
        if prior_identity is not None and prior_identity != state_identity:
            raise RuntimeError(
                "Compiled graph state output order no longer mirrors flat_state"
            )
        leaf.meta["recir_state_output_identity"] = state_identity
        leaf.meta["is_fsdp_sharded_output"] = is_sharded


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
