# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Rank-local FBGEMM TBE ownership for ReCIR sparse embedding groups.

This module deliberately contains no distributed communication. Callers route
indices to the owning rank before invoking :class:`LocalTBEGroup` and route the
returned embeddings back afterward. This keeps communication in the ReCIR FX
graph while delegating local lookup, backward accumulation, and optimizer
update to FBGEMM TBE.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from fbgemm_gpu.split_embedding_configs import EmbOptimType, SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    SplitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.tbe.config.embedding_config import (
    ComputeDevice,
    EmbeddingLocation,
    PoolingMode,
)
from recir.tbe.ops import register_tbe_group, TBEFunction


@dataclass(frozen=True)
class LocalTBETable:
    """Static description of one rank-local table owned by a TBE group."""

    name: str
    local_rows: int
    embedding_dim: int
    location: str = "HBM"


def _length_tensor(indices: torch.Tensor) -> torch.Tensor:
    return torch.scalar_tensor(
        indices.shape[0],
        dtype=torch.int64,
        device=indices.device,
    )


def _embedding_location(location: str, device: torch.device) -> EmbeddingLocation:
    if device.type == "cpu":
        if location != "HBM":
            raise ValueError(
                f"CPU TBE only supports host storage; got location={location!r}."
            )
        return EmbeddingLocation.HOST
    if device.type != "cuda":
        raise ValueError(f"LocalTBEGroup only supports CPU or CUDA, got {device}.")
    locations = {
        "HBM": EmbeddingLocation.DEVICE,
        "UVM": EmbeddingLocation.MANAGED,
        "UVM_CACHING": EmbeddingLocation.MANAGED_CACHING,
    }
    try:
        return locations[location]
    except KeyError as error:
        raise ValueError(
            f"Unsupported TBE embedding location: {location!r}."
        ) from error


class LocalTBEGroup(torch.nn.Module):
    """A grouped, unpooled TBE containing rank-local table shards.

    The main interface accepts one routed ID tensor per feature and returns one
    embedding tensor per feature. ``forward_flat`` exposes TBE's native packed
    ``indices`` and ``offsets`` interface for lower-level callers.

    The module owns embedding weights and RowWiseAdagrad state. Its backward
    updates touched rows inside FBGEMM; callers must not put these weights in a
    separate ``torch.optim.Optimizer``.
    """

    def __init__(
        self,
        tables: Sequence[LocalTBETable],
        feature_table_map: Sequence[int],
        *,
        device: torch.device,
        weight_dtype: torch.dtype,
        output_dtype: torch.dtype | None = None,
        learning_rate: float,
        eps: float,
        stochastic_rounding: bool = True,
    ) -> None:
        super().__init__()
        if not tables:
            raise ValueError("LocalTBEGroup requires at least one table.")
        if not feature_table_map:
            raise ValueError("LocalTBEGroup requires at least one feature.")
        if any(index < 0 or index >= len(tables) for index in feature_table_map):
            raise ValueError(
                "feature_table_map contains an index outside the table list."
            )

        embedding_dims = {table.embedding_dim for table in tables}
        if len(embedding_dims) != 1:
            raise ValueError(
                "Unpooled tables in one LocalTBEGroup must have the same "
                "embedding dimension."
            )
        for table in tables:
            if table.local_rows <= 0:
                raise ValueError(
                    f"Table {table.name!r} must have at least one local row."
                )
            if table.embedding_dim <= 0 or table.embedding_dim % 4 != 0:
                raise ValueError(
                    f"Table {table.name!r} embedding_dim must be a positive "
                    f"multiple of 4, got {table.embedding_dim}."
                )

        compute_device = (
            ComputeDevice.CPU if device.type == "cpu" else ComputeDevice.CUDA
        )
        resolved_output_dtype = output_dtype or weight_dtype
        self.embedding_dim: int = next(iter(embedding_dims))
        self.output_dtype: torch.dtype = resolved_output_dtype
        self.table_names: tuple[str, ...] = tuple(table.name for table in tables)
        self.feature_table_map: tuple[int, ...] = tuple(feature_table_map)
        self.tbe: SplitTableBatchedEmbeddingBagsCodegen = (
            SplitTableBatchedEmbeddingBagsCodegen(
                embedding_specs=[
                    (
                        table.local_rows,
                        table.embedding_dim,
                        _embedding_location(table.location, device),
                        compute_device,
                    )
                    for table in tables
                ],
                feature_table_map=list(feature_table_map),
                pooling_mode=PoolingMode.NONE,
                weights_precision=SparseType.from_dtype(weight_dtype),
                output_dtype=SparseType.from_dtype(resolved_output_dtype),
                optimizer=EmbOptimType.EXACT_ROWWISE_ADAGRAD,
                learning_rate=learning_rate,
                eps=eps,
                stochastic_rounding=stochastic_rounding,
                device=device,
                table_names=list(self.table_names),
            )
        )
        self.register_buffer(
            "_recir_state_token",
            torch.zeros((), dtype=torch.int64, device=device),
            persistent=False,
        )
        self._recir_tbe_state_names: tuple[str, ...] = tuple(
            name for name, _ in self.tbe.named_parameters()
        ) + tuple(name for name, _ in self.tbe.named_buffers())
        self._recir_handle: int = register_tbe_group(self)

    def forward(
        self,
        feature_indices: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        """Pack per-feature routed IDs into one grouped local lookup."""
        if len(feature_indices) != len(self.feature_table_map):
            raise ValueError(
                f"Expected {len(self.feature_table_map)} feature tensors, "
                f"got {len(feature_indices)}."
            )
        if any(indices.dim() != 1 for indices in feature_indices):
            raise ValueError("Local TBE feature indices must be one-dimensional.")

        feature_lengths = [indices.shape[0] for indices in feature_indices]
        flat_indices = torch.cat(tuple(feature_indices))
        offsets = torch.cat(
            (
                flat_indices.new_zeros((1,)),
                torch.stack(tuple(_length_tensor(x) for x in feature_indices)),
            )
        ).cumsum(dim=0)
        flat_output = self.forward_flat(flat_indices, offsets)
        return tuple(torch.split(flat_output, feature_lengths, dim=0))

    def forward_flat(
        self,
        indices: torch.Tensor,
        offsets: torch.Tensor,
        per_sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run a grouped local lookup with TBE-native flat inputs."""
        if per_sample_weights is not None:
            raise NotImplementedError(
                "ReCIR TBE custom ops do not yet support per-sample weights."
            )
        state: dict[str, torch.Tensor] = {}
        for name, parameter in self.tbe.named_parameters():
            state[name] = parameter
        for name, buffer in self.tbe.named_buffers():
            state[name] = buffer
        tbe_state = [state[name] for name in self._recir_tbe_state_names]
        return TBEFunction.apply(
            self._recir_handle,
            self._recir_state_token,
            self.tbe.placeholder_autograd_tensor,
            tbe_state,
            indices,
            offsets,
        )

    def _forward_impl(
        self,
        indices: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        return self.tbe(
            indices=indices,
            offsets=offsets,
            per_sample_weights=None,
        )

    @torch.no_grad()
    def load_table_weights(self, weights: Sequence[torch.Tensor]) -> None:
        """Copy existing local table shards into TBE-owned storage."""
        destinations = self.tbe.split_embedding_weights()
        if len(weights) != len(destinations):
            raise ValueError(
                f"Expected {len(destinations)} table weights, got {len(weights)}."
            )
        for name, source, destination in zip(self.table_names, weights, destinations):
            if source.shape != destination.shape:
                raise ValueError(
                    f"Table {name!r} shape mismatch: source={tuple(source.shape)}, "
                    f"TBE={tuple(destination.shape)}."
                )
            destination.copy_(
                source.to(device=destination.device, dtype=destination.dtype)
            )

    def split_embedding_weights(self) -> list[torch.Tensor]:
        """Return per-table views into TBE-owned embedding storage."""
        return self.tbe.split_embedding_weights()

    def split_optimizer_states(self) -> list[list[torch.Tensor]]:
        """Return per-table views into TBE-owned optimizer state."""
        return self.tbe.split_optimizer_states()

    def set_learning_rate(self, learning_rate: float) -> None:
        """Update the learning rate used by subsequent fused backward calls."""
        self.tbe.set_learning_rate(learning_rate)
