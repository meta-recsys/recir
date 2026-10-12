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
from recir.embedding.tbe.ops import register_tbe_group, TBEFunction


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


def nobag_tbe_offsets(
    feature_lengths: torch.Tensor,
    segments_per_feature: int,
) -> torch.Tensor:
    """Split each feature's flat IDs into balanced no-bag TBE work items."""
    # Citrine C3: create segment boundaries directly on the feature device.
    segment_boundaries = torch.arange(
        segments_per_feature + 1,
        dtype=torch.int64,
        device=feature_lengths.device,
    )
    feature_boundaries = torch.div(
        feature_lengths[:, None] * segment_boundaries,
        segments_per_feature,
        rounding_mode="floor",
    )
    segment_lengths = feature_boundaries[:, 1:] - feature_boundaries[:, :-1]
    return torch.cat(
        (feature_lengths.new_zeros((1,)), segment_lengths.reshape(-1))
    ).cumsum(dim=0)


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


def _pooling_mode(pooling_mode: str) -> PoolingMode:
    if pooling_mode == "none":
        return PoolingMode.NONE
    if pooling_mode == "sum":
        return PoolingMode.SUM
    if pooling_mode == "mean":
        return PoolingMode.MEAN
    raise ValueError(
        "LocalTBEGroup pooling_mode must be 'none', 'sum', or 'mean', "
        f"got {pooling_mode!r}."
    )


def _validate_group(
    tables: Sequence[LocalTBETable],
    feature_table_map: Sequence[int],
    segments_per_feature: int,
) -> int:
    if segments_per_feature <= 0:
        raise ValueError("TBE segments per feature must be positive.")
    if not tables:
        raise ValueError("LocalTBEGroup requires at least one table.")
    if not feature_table_map:
        raise ValueError("LocalTBEGroup requires at least one feature.")
    if any(index < 0 or index >= len(tables) for index in feature_table_map):
        raise ValueError("feature_table_map contains an index outside the table list.")
    embedding_dims = {table.embedding_dim for table in tables}
    if len(embedding_dims) != 1:
        raise ValueError("Tables in one LocalTBEGroup must have the same dimension.")
    for table in tables:
        if table.local_rows <= 0:
            raise ValueError(f"Table {table.name!r} must have at least one local row.")
        if table.embedding_dim <= 0 or table.embedding_dim % 4 != 0:
            raise ValueError(
                f"Table {table.name!r} embedding_dim must be a positive "
                f"multiple of 4, got {table.embedding_dim}."
            )
    return next(iter(embedding_dims))


def _validate_pooled_offset_spans(
    feature_indices: Sequence[torch.Tensor],
    feature_offsets: Sequence[torch.Tensor],
) -> None:
    for offsets in feature_offsets:
        torch._check(
            offsets.shape[0] > 0,
            lambda: "Pooled local TBE offsets must contain an initial boundary.",
        )
    spans = torch.stack(
        tuple(
            (offsets[0] == 0) & (offsets[-1] == indices.shape[0])
            for indices, offsets in zip(feature_indices, feature_offsets)
        )
    )
    torch._assert_async(
        spans.all(),
        "Pooled local TBE offsets must start at zero and consume every feature ID.",
    )


class LocalTBEGroup(torch.nn.Module):
    """A grouped TBE containing rank-local table shards.

    The main interface accepts one routed ID tensor per feature. Unpooled groups
    return one embedding per ID. SUM groups also accept one offsets tensor per
    feature and return one embedding per bag. ``forward_flat`` exposes TBE's
    native packed ``indices`` and ``offsets`` interface for lower-level callers.
    ``segments_per_feature`` controls only unpooled no-bag work subdivision.

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
        learning_rate_tensor: torch.Tensor | None = None,
        eps: float,
        stochastic_rounding: bool = True,
        segments_per_feature: int = 1,
        pooling_mode: str = "none",
    ) -> None:
        super().__init__()
        resolved_pooling_mode = _pooling_mode(pooling_mode)
        if pooling_mode != "none" and segments_per_feature != 1:
            raise ValueError(
                "TBE segments_per_feature only applies to unpooled groups."
            )
        embedding_dim = _validate_group(
            tables,
            feature_table_map,
            segments_per_feature,
        )

        compute_device = (
            ComputeDevice.CPU if device.type == "cpu" else ComputeDevice.CUDA
        )
        resolved_output_dtype = output_dtype or weight_dtype
        self.embedding_dim: int = embedding_dim
        self.output_dtype: torch.dtype = resolved_output_dtype
        self.table_names: tuple[str, ...] = tuple(table.name for table in tables)
        self.feature_table_map: tuple[int, ...] = tuple(feature_table_map)
        self.segments_per_feature: int = segments_per_feature
        self.pooling_mode: str = pooling_mode
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
                pooling_mode=resolved_pooling_mode,
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
        if learning_rate_tensor is not None:
            if learning_rate_tensor.device.type != "cpu":
                raise ValueError("TBE learning-rate tensor must be on CPU.")
            if learning_rate_tensor.dtype != torch.float32:
                raise ValueError("TBE learning-rate tensor must use torch.float32.")
            if learning_rate_tensor.dim() != 0:
                raise ValueError("TBE learning-rate tensor must be zero-dimensional.")
        runtime_learning_rate = (
            self.tbe.learning_rate_tensor
            if learning_rate_tensor is None
            else learning_rate_tensor
        )
        self.register_buffer(
            "_recir_learning_rate",
            runtime_learning_rate,
            persistent=False,
        )
        self.tbe.learning_rate_tensor = runtime_learning_rate
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
        feature_offsets: Sequence[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Pack per-feature routed IDs into one grouped local lookup.

        Pooled callers provide canonical ``include_last_offset=True`` offsets
        spanning each corresponding ID tensor. RecIR's EBC tracing and routing
        paths construct that representation before calling this module.
        """
        if len(feature_indices) != len(self.feature_table_map):
            raise ValueError(
                f"Expected {len(self.feature_table_map)} feature tensors, "
                f"got {len(feature_indices)}."
            )
        if any(indices.dim() != 1 for indices in feature_indices):
            raise ValueError("Local TBE feature indices must be one-dimensional.")

        if self.pooling_mode in ("sum", "mean"):
            if feature_offsets is None:
                raise ValueError("Pooled local TBE requires per-feature offsets.")
            if len(feature_offsets) != len(feature_indices):
                raise ValueError(
                    f"Expected {len(feature_indices)} feature offsets tensors, "
                    f"got {len(feature_offsets)}."
                )
            if any(offsets.dim() != 1 for offsets in feature_offsets):
                raise ValueError("Local TBE feature offsets must be one-dimensional.")
            bag_count = feature_offsets[0].shape[0]
            if any(offsets.shape[0] != bag_count for offsets in feature_offsets[1:]):
                raise ValueError(
                    "Pooled local TBE features must have the same number of bags."
                )
            _validate_pooled_offset_spans(feature_indices, feature_offsets)
            packed_lengths = torch.cat(
                tuple(offsets[1:] - offsets[:-1] for offsets in feature_offsets)
            )
            flat_indices = torch.cat(tuple(feature_indices))
            packed_offsets = torch.cat(
                (packed_lengths.new_zeros((1,)), packed_lengths)
            ).cumsum(dim=0)
            flat_output = self.forward_flat(flat_indices, packed_offsets)
            return tuple(torch.split(flat_output, self.embedding_dim, dim=1))

        if feature_offsets is not None:
            raise ValueError("Unpooled local TBE does not accept feature offsets.")
        feature_lengths = [indices.shape[0] for indices in feature_indices]
        flat_indices = torch.cat(tuple(feature_indices))
        offsets = nobag_tbe_offsets(
            torch.stack(tuple(_length_tensor(x) for x in feature_indices)),
            self.segments_per_feature,
        )
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
            self.get_buffer("_recir_learning_rate"),
            indices,
            offsets,
            self.pooling_mode != "none",
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
        self.get_buffer("_recir_learning_rate").fill_(learning_rate)
