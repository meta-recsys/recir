# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""TBE-backed row-wise sparse embedding transform.

The inner rewrite replaces compatible ``aten.embedding`` sites with grouped,
rank-local FBGEMM TBE calls.  The outer rewrite keeps row-wise routing as
ordinary functional collective nodes around the opaque native TBE operators.
"""

from __future__ import annotations

import operator
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, cast

import torch
import torch.distributed._functional_collectives as funcol
import torch.fx as fx
import torch.nn as nn
from functorch import make_fx
from recir.region_transforms.base import RegionTransform
from recir.region_transforms.sparse_embedding import _declare_split_invariants
from recir.tbe.local import LocalTBEGroup, LocalTBETable
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch.distributed.tensor import DeviceMesh
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


_PLACEHOLDER_SIZE: int = 3
_TBE_ROOT_NAME: str = "_recir_tbe_sparse_groups"
_WEIGHT_PASSTHROUGH_OPS: tuple[object, ...] = (
    torch.ops.aten._to_copy.default,
    torch.ops.aten.clone.default,
    torch.ops.aten.detach.default,
    torch.ops.aten.view.default,
    torch.ops.aten.reshape.default,
)


def _resolve_weight_placeholder(
    weight_arg: object,
    weight_placeholders: set[fx.Node],
) -> fx.Node | None:
    node = weight_arg
    while isinstance(node, fx.Node) and node.op == "call_function":
        if node.target not in _WEIGHT_PASSTHROUGH_OPS and not getattr(
            node.target, "_recir_preserved_autograd_leaf", False
        ):
            break
        node = node.args[0]
    if not isinstance(node, fx.Node) or node not in weight_placeholders:
        return None
    if node is not weight_arg:
        arg_val = (
            weight_arg.meta.get("val") if isinstance(weight_arg, fx.Node) else None
        )
        root_val = node.meta.get("val")
        if arg_val is None or root_val is None or arg_val.shape != root_val.shape:
            return None
    return node


@dataclass(frozen=True)
class _OptimizerSettings:
    learning_rate: float
    eps: float
    stochastic_rounding: bool
    learning_rate_tensor: torch.Tensor | None


@dataclass(frozen=True)
class _TableState:
    fqn: str
    parameter: nn.Parameter
    location: str
    optimizer: torch.optim.Optimizer | None
    optimizer_group: dict[str, Any] | None
    optimizer_state: dict[str, Any] | None
    settings: _OptimizerSettings


@dataclass(frozen=True)
class _FeatureSite:
    fqn: str
    global_rows: int
    embedding_dim: int


@dataclass(frozen=True)
class _CompiledGroup:
    sites: tuple[_FeatureSite, ...]
    device: torch.device


def _as_float(value: object, name: str) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"TBE {name} tensor must contain one value.")
        # This runs once during compilation, not in the training loop.
        return float(value.detach().to(device="cpu").item())
    return float(value)  # pyre-ignore[6]


def _materialize_splits(
    input_splits: torch.Tensor,
    output_splits: torch.Tensor,
) -> tuple[list[int], list[int], torch.Tensor]:
    split_count = input_splits.shape[0]
    # Citrine C6: one batched D2H copy supplies all later sparse routing stages.
    cpu_splits = torch.cat((input_splits, output_splits)).to(device="cpu", copy=True)
    return (
        cpu_splits[:split_count].tolist(),
        cpu_splits[split_count:].tolist(),
        cpu_splits,
    )


def _split_lists(
    cpu_splits: torch.Tensor, world_size: int
) -> tuple[list[int], list[int]]:
    return (
        cpu_splits[:world_size].tolist(),
        cpu_splits[world_size:].tolist(),
    )


def route_row_wise_indices_grouped(
    *feature_indices: torch.Tensor,
    num_embeddings: tuple[int, ...],
    group_name: str,
    world_size: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Route a TBE feature group with one counts and one indices all-to-all."""
    feature_count = len(feature_indices)
    if feature_count != len(num_embeddings):
        raise ValueError("Feature and table cardinality counts must match.")

    local_indices: list[torch.Tensor] = []
    route_keys: list[torch.Tensor] = []
    for feature, (indices, rows) in enumerate(zip(feature_indices, num_embeddings)):
        flat_indices = indices.reshape(-1)
        local_rows = rows // world_size
        ranks = flat_indices // local_rows
        local_indices.append(flat_indices % local_rows)
        route_keys.append(ranks * feature_count + feature)

    packed_local_indices = torch.cat(local_indices)
    packed_route_keys = torch.cat(route_keys)
    send_order = torch.argsort(packed_route_keys, stable=True)
    restore_order = torch.argsort(send_order)
    send_counts = torch.stack(
        [
            (packed_route_keys == segment).sum()
            for segment in range(world_size * feature_count)
        ]
    )

    received_counts = torch.ops._c10d_functional.all_to_all_single(
        send_counts,
        [feature_count] * world_size,
        [feature_count] * world_size,
        group_name,
    )
    received_counts = torch.ops._c10d_functional.wait_tensor(received_counts)
    input_splits = send_counts.reshape(world_size, feature_count).sum(dim=1)
    output_splits = received_counts.reshape(world_size, feature_count).sum(dim=1)
    input_split_list, output_split_list, cpu_splits = _materialize_splits(
        input_splits, output_splits
    )
    _declare_split_invariants(input_split_list, packed_local_indices.shape[0])
    _declare_split_invariants(output_split_list)

    received_indices = torch.ops._c10d_functional.all_to_all_single(
        packed_local_indices[send_order],
        output_split_list,
        input_split_list,
        group_name,
    )
    received_indices = torch.ops._c10d_functional.wait_tensor(received_indices)

    source_feature_segments = torch.repeat_interleave(
        torch.arange(
            world_size * feature_count,
            dtype=torch.int64,
            device=received_counts.device,
        ),
        received_counts,
        output_size=received_indices.shape[0],
    )
    receive_keys = (source_feature_segments % feature_count) * world_size + torch.div(
        source_feature_segments, feature_count, rounding_mode="floor"
    )
    receive_order = torch.argsort(receive_keys, stable=True)
    receive_restore_order = torch.argsort(receive_order)
    feature_lengths = received_counts.reshape(world_size, feature_count).sum(dim=0)
    return (
        received_indices[receive_order],
        feature_lengths,
        send_order,
        restore_order,
        receive_order,
        receive_restore_order,
        cpu_splits,
    )


def return_row_wise_embeddings_grouped(
    local_embeddings: torch.Tensor,
    restore_order: torch.Tensor,
    receive_restore_order: torch.Tensor,
    cpu_splits: torch.Tensor,
    group_name: str,
    world_size: int,
) -> torch.Tensor:
    """Return one grouped TBE output and restore original feature order."""
    input_splits, output_splits = _split_lists(cpu_splits, world_size)
    _declare_split_invariants(input_splits, restore_order.shape[0])
    _declare_split_invariants(output_splits, local_embeddings.shape[0])
    embedding_dim = local_embeddings.shape[1]
    source_major_embeddings = local_embeddings[receive_restore_order]
    returned = torch.ops._c10d_functional.all_to_all_single(
        source_major_embeddings.reshape(-1),
        [size * embedding_dim for size in input_splits],
        [size * embedding_dim for size in output_splits],
        group_name,
    )
    returned = torch.ops._c10d_functional.wait_tensor(returned)
    return returned.reshape(-1, embedding_dim)[restore_order]


def route_row_wise_gradients_grouped(
    gradients: torch.Tensor,
    send_order: torch.Tensor,
    receive_order: torch.Tensor,
    cpu_splits: torch.Tensor,
    group_name: str,
    world_size: int,
) -> torch.Tensor:
    """Route packed feature gradients using grouped forward metadata."""
    input_splits, output_splits = _split_lists(cpu_splits, world_size)
    _declare_split_invariants(input_splits, gradients.shape[0])
    embedding_dim = gradients.shape[1]
    received = torch.ops._c10d_functional.all_to_all_single(
        gradients[send_order].reshape(-1),
        [size * embedding_dim for size in output_splits],
        [size * embedding_dim for size in input_splits],
        group_name,
    )
    received = torch.ops._c10d_functional.wait_tensor(received)
    return received.reshape(-1, embedding_dim)[receive_order]


def _inline_before(
    gm: fx.GraphModule,
    target: fx.Node,
    function: Any,
    example_inputs: Sequence[torch.Tensor],
    input_nodes: Sequence[fx.Node],
) -> tuple[Any, set[fx.Node]]:
    traced = make_fx(function, tracing_mode="symbolic")(*example_inputs)
    placeholders = [node for node in traced.graph.nodes if node.op == "placeholder"]
    val_map = dict(zip(placeholders, input_nodes))
    existing = set(gm.graph.nodes)
    with gm.graph.inserting_before(target):
        output = gm.graph.graph_copy(traced.graph, val_map=val_map)
    new_nodes = set(gm.graph.nodes) - existing
    for node in new_nodes:
        node.meta.pop("unbacked_bindings", None)
    return output, new_nodes


class TBESparseEmbeddingTransform(RegionTransform):
    """Use grouped local TBE while retaining sparse collectives in the graph.

    This first implementation supports unpooled, one-dimensional feature ID
    tensors and Exact RowWiseAdagrad without weight decay.  Compatible lookup
    sites are grouped by dtype, device, embedding dimension, storage location,
    and optimizer settings.
    """

    def __init__(
        self,
        embedding_param_fqns: set[str],
        mesh: DeviceMesh,
        mod: nn.Module | None = None,
        *,
        embedding_location_map: dict[str, str] | None = None,
        num_embeddings_map: dict[str, int] | None = None,
        learning_rate: float = 0.1,
        eps: float = 1.0e-8,
        stochastic_rounding: bool = True,
    ) -> None:
        self.embedding_param_fqns = embedding_param_fqns
        self._source_fqns = set(embedding_param_fqns)
        self.mesh = mesh
        self.world_size = mesh.size()
        self.group_name = funcol._resolve_group_name((mesh, 0))
        self.mod = mod
        self.embedding_location_map = embedding_location_map or {}
        self.num_embeddings_map = num_embeddings_map or (
            {fqn: int(mod.get_parameter(fqn).shape[0]) for fqn in self._source_fqns}
            if mod is not None
            else {}
        )
        self.default_settings = _OptimizerSettings(
            learning_rate, eps, stochastic_rounding, None
        )
        self._table_states: dict[str, _TableState] = {}
        self._optimizers: tuple[torch.optim.Optimizer, ...] = ()
        self._compiled_groups: list[_CompiledGroup] = []
        self._forward_root_fqn: str | None = None

    def select_forward_root_fqns(
        self,
        called_modules: OrderedDict[
            str, tuple[nn.Module, tuple[Any, ...], dict[str, Any]]
        ],
    ) -> set[str]:
        candidates = [
            fqn
            for fqn in called_modules
            if not fqn
            or all(
                source == fqn or source.startswith(f"{fqn}.")
                for source in self._source_fqns
            )
        ]
        if not candidates:
            raise RuntimeError("No called module owns all TBE embedding parameters.")
        self._forward_root_fqn = max(candidates, key=lambda fqn: fqn.count("."))
        return {self._forward_root_fqn}

    def transform_inputs(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        for value in (*args, *kwargs.values()):
            if isinstance(value, KeyedJaggedTensor):
                value.lengths()
                value.offsets()
        return args, kwargs

    def prepare_after_shard(
        self,
        module: nn.Module,
        optimizers: Sequence[torch.optim.Optimizer],
    ) -> None:
        self.mod = module
        self._optimizers = tuple(optimizers)
        self._table_states = {
            fqn: self._capture_table_state(module, fqn) for fqn in self._source_fqns
        }

    def _capture_table_state(self, module: nn.Module, fqn: str) -> _TableState:
        parameter = module.get_parameter(fqn)
        if parameter.dim() != 2:
            raise ValueError(f"TBE embedding parameter {fqn!r} must be 2-D.")
        global_rows = self.num_embeddings_map.get(
            fqn, int(parameter.shape[0]) * self.world_size
        )
        if global_rows != int(parameter.shape[0]) * self.world_size:
            raise NotImplementedError(
                "TBE sparse routing currently requires even row-wise shards; "
                f"{fqn!r} has global_rows={global_rows}, "
                f"local_rows={int(parameter.shape[0])}, world_size={self.world_size}."
            )
        found: list[tuple[torch.optim.Optimizer, dict[str, Any]]] = []
        for optimizer in self._optimizers:
            for group in optimizer.param_groups:
                if any(candidate is parameter for candidate in group["params"]):
                    found.append((optimizer, group))
        if len(found) > 1:
            raise ValueError(f"TBE embedding parameter {fqn!r} has multiple owners.")
        if not found:
            return _TableState(
                fqn,
                parameter,
                self.embedding_location_map.get(fqn, "HBM"),
                None,
                None,
                None,
                self.default_settings,
            )
        optimizer, group = found[0]
        if not type(optimizer).__name__.endswith("RowWiseAdagrad"):
            raise TypeError(
                f"TBE embedding parameter {fqn!r} must use RowWiseAdagrad; "
                f"got {type(optimizer).__name__}."
            )
        weight_decay = _as_float(group.get("weight_decay", 0.0), "weight_decay")
        if weight_decay != 0.0:
            raise NotImplementedError(
                "TBE transform does not yet migrate weight decay."
            )
        lr_value = group.get("lr", self.default_settings.learning_rate)
        lr_tensor = lr_value if isinstance(lr_value, torch.Tensor) else None
        settings = _OptimizerSettings(
            _as_float(lr_value, "learning_rate"),
            _as_float(group.get("eps", self.default_settings.eps), "eps"),
            bool(group.get("stochastic_rounding", True)),
            lr_tensor,
        )
        state = optimizer.state.get(parameter)
        return _TableState(
            fqn,
            parameter,
            self.embedding_location_map.get(fqn, "HBM"),
            optimizer,
            group,
            state if isinstance(state, dict) else None,
            settings,
        )

    def transform_graph(
        self,
        gm: fx.GraphModule,
        called_modules: OrderedDict[
            str, tuple[nn.Module, tuple[Any, ...], dict[str, Any]]
        ],
    ) -> None:
        if not self._source_fqns:
            return
        if self.mod is None or not self._table_states:
            raise RuntimeError("prepare_after_shard must run before transform_graph.")
        if self._forward_root_fqn is None:
            raise RuntimeError("TBE forward root was not selected before tracing.")
        root_module = called_modules[self._forward_root_fqn][0]
        relative_fqns = [
            fqn for fqn, _ in root_module.named_parameters(remove_duplicate=False)
        ]
        placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
        weight_nodes: dict[fx.Node, str] = {}
        for placeholder, relative_fqn in zip(placeholders, relative_fqns):
            full_fqn = (
                f"{self._forward_root_fqn}.{relative_fqn}"
                if self._forward_root_fqn
                else relative_fqn
            )
            if full_fqn in self._source_fqns:
                weight_nodes[placeholder] = full_fqn
        sites: list[tuple[fx.Node, fx.Node, str]] = []
        roots = set(weight_nodes)
        for node in gm.graph.nodes:
            if (
                node.op != "call_function"
                or node.target != torch.ops.aten.embedding.default
            ):
                continue
            root = _resolve_weight_placeholder(node.args[0], roots)
            if root is None:
                continue
            indices = node.args[1]
            if not isinstance(indices, fx.Node) or len(indices.meta["val"].shape) != 1:
                raise ValueError("TBE sparse transform requires one-dimensional IDs.")
            sites.append((node, indices, weight_nodes[root]))
        rewritten_fqns = {fqn for _, _, fqn in sites}
        missing_fqns = self._source_fqns - rewritten_fqns
        if missing_fqns:
            raise RuntimeError(
                "TBE sparse transform did not find lookup sites for row-wise "
                f"embedding parameters: {sorted(missing_fqns)}"
            )
        self._rewrite_compatible_groups(gm, sites)
        gm.graph.lint()
        gm.recompile()

    def _rewrite_compatible_groups(
        self,
        gm: fx.GraphModule,
        sites: Sequence[tuple[fx.Node, fx.Node, str]],
    ) -> None:
        grouped: dict[tuple[object, ...], list[tuple[fx.Node, fx.Node, str]]] = {}
        for site in sites:
            state = self._table_states[site[2]]
            parameter = state.parameter
            key = (
                parameter.device,
                parameter.dtype,
                int(parameter.shape[1]),
                state.location,
                state.settings.learning_rate,
                state.settings.eps,
                state.settings.stochastic_rounding,
                id(state.settings.learning_rate_tensor),
            )
            grouped.setdefault(key, []).append(site)
        module = self.mod
        if module is None:
            raise RuntimeError("prepare_after_shard must set the source module.")
        if hasattr(module, _TBE_ROOT_NAME):
            raise RuntimeError(f"Module already owns {_TBE_ROOT_NAME!r}.")
        owner = nn.ModuleDict()
        module.add_module(_TBE_ROOT_NAME, owner)
        for group_index, group_sites in enumerate(grouped.values()):
            self._rewrite_group(gm, owner, group_index, group_sites)

    def _rewrite_group(
        self,
        gm: fx.GraphModule,
        owner: nn.ModuleDict,
        group_index: int,
        sites: Sequence[tuple[fx.Node, fx.Node, str]],
    ) -> None:
        table_fqns = tuple(dict.fromkeys(fqn for _, _, fqn in sites))
        table_index = {fqn: index for index, fqn in enumerate(table_fqns)}
        first_state = self._table_states[table_fqns[0]]
        tables = tuple(
            LocalTBETable(
                name=fqn,
                local_rows=int(self._table_states[fqn].parameter.shape[0]),
                embedding_dim=int(self._table_states[fqn].parameter.shape[1]),
                location=self._table_states[fqn].location,
            )
            for fqn in table_fqns
        )
        with unset_fake_temporarily():
            local_tbe = LocalTBEGroup(
                tables,
                tuple(table_index[fqn] for _, _, fqn in sites),
                device=first_state.parameter.device,
                weight_dtype=first_state.parameter.dtype,
                learning_rate=first_state.settings.learning_rate,
                learning_rate_tensor=first_state.settings.learning_rate_tensor,
                eps=first_state.settings.eps,
                stochastic_rounding=first_state.settings.stochastic_rounding,
            )
            local_tbe.load_table_weights(
                tuple(self._table_states[fqn].parameter for fqn in table_fqns)
            )
        owner_name = f"group_{group_index}"
        owner[owner_name] = local_tbe
        graph_name = f"_recir_tbe_group_{group_index}"
        gm.add_submodule(graph_name, local_tbe)
        self._migrate_group_state(table_fqns, local_tbe)

        first_node = sites[0][0]
        node_positions = {node: index for index, node in enumerate(gm.graph.nodes)}
        if any(
            node_positions[index] >= node_positions[first_node] for _, index, _ in sites
        ):
            raise RuntimeError("TBE feature inputs must be available before the group.")
        with gm.graph.inserting_before(first_node):
            grouped_output = gm.graph.call_module(
                graph_name,
                args=(tuple(indices for _, indices, _ in sites),),
            )
        for feature_index, (embedding, _indices, _fqn) in enumerate(sites):
            with gm.graph.inserting_before(embedding):
                output = gm.graph.call_function(
                    operator.getitem, args=(grouped_output, feature_index)
                )
                output.meta = dict(embedding.meta)
            embedding.replace_all_uses_with(output)
            gm.graph.erase_node(embedding)

        self._compiled_groups.append(
            _CompiledGroup(
                tuple(
                    _FeatureSite(
                        fqn,
                        self.num_embeddings_map.get(
                            fqn,
                            int(self._table_states[fqn].parameter.shape[0])
                            * self.world_size,
                        ),
                        int(self._table_states[fqn].parameter.shape[1]),
                    )
                    for _, _, fqn in sites
                ),
                first_state.parameter.device,
            )
        )

    @torch.no_grad()
    def _migrate_group_state(
        self,
        table_fqns: Sequence[str],
        local_tbe: LocalTBEGroup,
    ) -> None:
        tbe_weights = local_tbe.split_embedding_weights()
        tbe_states = local_tbe.split_optimizer_states()
        for table_index, fqn in enumerate(table_fqns):
            state = self._table_states[fqn]
            if state.optimizer_state is not None and "sum" in state.optimizer_state:
                source_sum = state.optimizer_state["sum"]
                if isinstance(source_sum, torch.Tensor) and tbe_states[table_index]:
                    destination = tbe_states[table_index][0]
                    destination.copy_(source_sum.reshape(destination.shape))
            if state.optimizer is not None and state.optimizer_group is not None:
                state.optimizer_group["params"] = [
                    parameter
                    for parameter in state.optimizer_group["params"]
                    if parameter is not state.parameter
                ]
                state.optimizer.state.pop(state.parameter, None)
            parent, attribute = self._parameter_parent(fqn)
            setattr(
                parent,
                attribute,
                nn.Parameter(tbe_weights[table_index], requires_grad=False),
            )
        module = self.mod
        if module is None:
            raise RuntimeError("prepare_after_shard must set the source module.")
        prefix = f"{_TBE_ROOT_NAME}."
        self.embedding_param_fqns.update(
            name for name, _ in module.named_parameters() if name.startswith(prefix)
        )

    def _parameter_parent(self, fqn: str) -> tuple[nn.Module, str]:
        assert self.mod is not None
        atoms = fqn.split(".")
        parent = self.mod
        for atom in atoms[:-1]:
            child = getattr(parent, atom)
            if not isinstance(child, nn.Module):
                raise TypeError(
                    f"Embedding parent {'.'.join(atoms[:-1])!r} is not a module."
                )
            parent = child
        return parent, atoms[-1]

    def finalize_compiled_graph(self, gm: fx.GraphModule) -> None:
        if self.world_size == 1 or not self._compiled_groups:
            return
        forwards = [
            node
            for node in gm.graph.nodes
            if node.op == "call_function"
            and node.target == torch.ops.recir.tbe_forward.default
        ]
        if len(forwards) != len(self._compiled_groups):
            raise RuntimeError(
                "TBE forward count changed between inner and outer traces: "
                f"expected {len(self._compiled_groups)}, found {len(forwards)}."
            )
        for forward, group in zip(forwards, self._compiled_groups):
            self._route_group(gm, forward, group)
        gm.graph.lint()
        gm.recompile()

    def _route_group(
        self,
        gm: fx.GraphModule,
        forward: fx.Node,
        group: _CompiledGroup,
    ) -> None:
        packed_indices = forward.args[5]
        packed_offsets = forward.args[6]
        forward_output = next(
            (
                user
                for user in forward.users
                if user.op == "call_function"
                and user.target == operator.getitem
                and user.args[1] == 0
            ),
            None,
        )
        if forward_output is None:
            raise RuntimeError("Expected tensor output from recir::tbe_forward.")
        if (
            not isinstance(packed_indices, fx.Node)
            or packed_indices.target != torch.ops.aten.cat.default
        ):
            raise RuntimeError("Expected LocalTBEGroup to pack indices with aten.cat.")
        raw_features = packed_indices.args[0]
        if not isinstance(raw_features, (list, tuple)) or len(raw_features) != len(
            group.sites
        ):
            raise RuntimeError(
                "TBE packed feature count does not match transform metadata."
            )
        if any(not isinstance(raw, fx.Node) for raw in raw_features):
            raise RuntimeError("TBE feature indices must be FX nodes.")
        feature_nodes = tuple(cast(fx.Node, raw) for raw in raw_features)
        feature_count = len(group.sites)
        example_features = tuple(
            torch.randint(
                0,
                site.global_rows,
                (_PLACEHOLDER_SIZE,),
                device=group.device,
            )
            for site in group.sites
        )
        output, _ = _inline_before(
            gm,
            forward,
            partial(
                route_row_wise_indices_grouped,
                num_embeddings=tuple(site.global_rows for site in group.sites),
                group_name=self.group_name,
                world_size=self.world_size,
            ),
            example_features,
            feature_nodes,
        )
        (
            received_indices,
            received_feature_lengths,
            send_order,
            restore_order,
            receive_order,
            receive_restore_order,
            cpu_splits,
        ) = output

        with gm.graph.inserting_before(forward):
            zero = gm.graph.call_function(
                torch.ops.aten.new_zeros.default, args=(received_feature_lengths, [1])
            )
            offsets = gm.graph.call_function(
                torch.ops.aten.cumsum.default,
                args=(
                    gm.graph.call_function(
                        torch.ops.aten.cat.default,
                        args=([zero, received_feature_lengths],),
                    ),
                    0,
                ),
            )
        forward_args = list(forward.args)
        forward_args[5] = received_indices
        forward_args[6] = offsets
        forward.args = tuple(forward_args)
        for node in gm.graph.nodes:
            if (
                node.op == "call_function"
                and node.target == torch.ops.fbgemm.bounds_check_indices.default
                and len(node.args) > 2
                and node.args[1] is packed_indices
                and node.args[2] is packed_offsets
            ):
                bounds_args = list(node.args)
                bounds_args[1] = received_indices
                bounds_args[2] = offsets
                node.args = tuple(bounds_args)
                forward.prepend(node)

        split_node = next(
            (
                user
                for user in forward_output.users
                if user.op == "call_function"
                and user.target == torch.ops.aten.split_with_sizes.default
            ),
            None,
        )
        if split_node is None:
            raise RuntimeError("Expected grouped TBE output to use split_with_sizes.")
        original_lengths: list[fx.Node] = []
        with gm.graph.inserting_before(split_node):
            for feature in feature_nodes:
                original_lengths.append(
                    gm.graph.call_function(
                        torch.ops.aten.sym_size.int, args=(feature, 0)
                    )
                )
        returned_output, _ = _inline_before(
            gm,
            split_node,
            partial(
                return_row_wise_embeddings_grouped,
                group_name=self.group_name,
                world_size=self.world_size,
            ),
            (
                torch.randn(
                    (feature_count * _PLACEHOLDER_SIZE, group.sites[0].embedding_dim),
                    device=group.device,
                ),
                torch.arange(feature_count * _PLACEHOLDER_SIZE, device=group.device),
                torch.arange(feature_count * _PLACEHOLDER_SIZE, device=group.device),
                torch.ones((2 * self.world_size,), dtype=torch.int64),
            ),
            (forward_output, restore_order, receive_restore_order, cpu_splits),
        )
        split_node.args = (returned_output, original_lengths)

        self._route_group_backward(
            gm,
            forward,
            group,
            send_order,
            receive_order,
            cpu_splits,
        )

    def _route_group_backward(
        self,
        gm: fx.GraphModule,
        forward: fx.Node,
        group: _CompiledGroup,
        send_order: fx.Node,
        receive_order: fx.Node,
        cpu_splits: fx.Node,
    ) -> None:
        backward = next(
            (
                node
                for node in gm.graph.nodes
                if node.op == "call_function"
                and node.target == torch.ops.recir.tbe_backward_update.default
                and node.args[0] == forward.args[0]
            ),
            None,
        )
        if backward is None:
            raise RuntimeError("Could not find fused TBE backward for forward group.")
        grad_output = backward.args[4]
        if not isinstance(grad_output, fx.Node):
            raise RuntimeError("TBE backward gradient must be an FX node.")
        embedding_dim = group.sites[0].embedding_dim
        packed_grad, _ = _inline_before(
            gm,
            backward,
            partial(
                route_row_wise_gradients_grouped,
                group_name=self.group_name,
                world_size=self.world_size,
            ),
            (
                torch.randn(
                    (len(group.sites) * _PLACEHOLDER_SIZE, embedding_dim),
                    device=group.device,
                ),
                torch.arange(len(group.sites) * _PLACEHOLDER_SIZE, device=group.device),
                torch.arange(len(group.sites) * _PLACEHOLDER_SIZE, device=group.device),
                torch.ones((2 * self.world_size,), dtype=torch.int64),
            ),
            (grad_output, send_order, receive_order, cpu_splits),
        )
        backward_args = list(backward.args)
        backward_args[4] = packed_grad
        backward.args = tuple(backward_args)
