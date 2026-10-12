# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Graph-visible custom operators around rank-local FBGEMM TBE."""

from __future__ import annotations

import itertools
import weakref
from typing import Any

import fbgemm_gpu.sparse_ops  # noqa: F401 - registers sparse op implementations
import torch
from torch.nn.utils import stateless
from torch.utils._python_dispatch import _disable_current_modes


class _HostIterationCounter:
    """Avoid TBE's first-call GPU sync for plain Exact RowWiseAdagrad."""

    def __init__(self) -> None:
        self._value: int = 1

    def cpu(self) -> _HostIterationCounter:
        return self

    def item(self) -> int:
        return self._value

    def add_(self, increment: int) -> _HostIterationCounter:
        self._value += increment
        return self


_TBE_GROUPS: weakref.WeakValueDictionary[int, Any] = weakref.WeakValueDictionary()
_TBE_CONTEXTS: dict[int, tuple[int, torch.Tensor]] = {}
_TBE_ITER_CPU: dict[int, _HostIterationCounter] = {}
_NEXT_CONTEXT = itertools.count()


@torch.library.custom_op("recir::dedup_sparse_features", mutates_args=())
def dedup_sparse_features(
    feature_lengths: torch.Tensor,
    indices: torch.Tensor,
    hash_size_cumsum: torch.Tensor,
    hash_size_offset: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deduplicate EC indices using TorchRec's FBGEMM primitive."""
    if hash_size_cumsum.numel() != hash_size_offset.numel():
        raise ValueError("Dedup hash metadata must have one entry per feature.")
    offsets = torch.cat((feature_lengths.new_zeros((1,)), feature_lengths)).cumsum(
        dim=0
    )
    dedup_lengths, _dedup_offsets, unique_indices, reverse_indices = (
        torch.ops.fbgemm.jagged_unique_indices(
            hash_size_cumsum,
            hash_size_offset,
            offsets.to(torch.int64),
            indices.to(torch.int64),
        )
    )
    return dedup_lengths, unique_indices, reverse_indices


@dedup_sparse_features.register_fake
def _fake_dedup_sparse_features(
    feature_lengths: torch.Tensor,
    indices: torch.Tensor,
    hash_size_cumsum: torch.Tensor,
    hash_size_offset: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del hash_size_cumsum, hash_size_offset
    unique_count = torch.library.get_ctx().new_dynamic_size()
    torch._check_is_size(unique_count)
    return (
        feature_lengths.new_empty(feature_lengths.shape),
        indices.new_empty((unique_count,), dtype=torch.int64),
        indices.new_empty(indices.shape, dtype=torch.int64),
    )


@torch.library.custom_op("recir::bucketize_sparse_features", mutates_args=())
def bucketize_sparse_features(
    feature_lengths: torch.Tensor,
    indices: torch.Tensor,
    block_sizes: torch.Tensor,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bucketize grouped features by destination rank with FBGEMM."""
    batch_size = feature_lengths.numel() // block_sizes.numel()
    (
        bucketized_lengths,
        bucketized_indices,
        _bucketized_weights,
        _bucketized_positions,
        unbucketize_permute,
    ) = torch.ops.fbgemm.block_bucketize_sparse_features(
        feature_lengths,
        indices,
        bucketize_pos=False,
        sequence=True,
        block_sizes=block_sizes,
        my_size=world_size,
        weights=None,
        batch_size_per_feature=None,
        max_B=batch_size,
        keep_orig_idx=False,
    )
    if unbucketize_permute is None:
        raise RuntimeError("Expected sequence bucketization to return a permutation.")
    bucketize_permute = torch.ops.fbgemm.invert_permute(unbucketize_permute)
    return (
        bucketized_lengths,
        bucketized_indices,
        bucketize_permute,
        unbucketize_permute,
    )


@bucketize_sparse_features.register_fake
def _fake_bucketize_sparse_features(
    feature_lengths: torch.Tensor,
    indices: torch.Tensor,
    block_sizes: torch.Tensor,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bucketized_lengths = feature_lengths.new_empty(
        (feature_lengths.shape[0] * world_size,)
    )
    return (
        bucketized_lengths,
        indices.new_empty(indices.shape),
        indices.new_empty(indices.shape),
        indices.new_empty(indices.shape),
    )


@torch.library.custom_op("recir::permute_sparse_data_2d", mutates_args=())
def permute_sparse_data_2d(
    permutation: torch.Tensor,
    lengths: torch.Tensor,
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Permute fixed-width groups of variable-length segments with FBGEMM."""
    permuted_lengths, permuted_values, _permuted_weights = (
        torch.ops.fbgemm.permute_2D_sparse_data(
            permutation,
            lengths,
            values,
            None,
            values.numel(),
        )
    )
    return permuted_lengths, permuted_values


@permute_sparse_data_2d.register_fake
def _fake_permute_sparse_data_2d(
    permutation: torch.Tensor,
    lengths: torch.Tensor,
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    del permutation
    return lengths.new_empty(lengths.shape), values.new_empty(values.shape)


@torch.library.custom_op("recir::permute_sparse_data_1d", mutates_args=())
def permute_sparse_data_1d(
    permutation: torch.Tensor,
    lengths: torch.Tensor,
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Permute arbitrary variable-length segments with FBGEMM."""
    permuted_lengths, permuted_values, _permuted_weights = (
        torch.ops.fbgemm.permute_1D_sparse_data(
            permutation,
            lengths,
            values,
            None,
            values.numel(),
        )
    )
    return permuted_lengths, permuted_values


@permute_sparse_data_1d.register_fake
def _fake_permute_sparse_data_1d(
    permutation: torch.Tensor,
    lengths: torch.Tensor,
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    del permutation
    return lengths.new_empty(lengths.shape), values.new_empty(values.shape)


def _permute_2d_sparse_data_stride(
    lengths: torch.Tensor,
    feature_count: int,
) -> int:
    if feature_count <= 0:
        raise ValueError("KJT feature count must be positive.")
    if lengths.dim() != 1:
        raise ValueError("KJT lengths must be one-dimensional.")
    length_count = lengths.numel()
    if isinstance(length_count, torch.SymInt):
        torch._check(
            length_count % feature_count == 0,
            lambda: "KJT lengths must contain the same stride for every feature.",
        )
    elif length_count % feature_count != 0:
        raise ValueError("KJT lengths must contain the same stride for every feature.")
    return length_count // feature_count


@torch.library.custom_op(
    "recir::permute_2d_sparse_data",
    mutates_args=(),
    schema=(
        "(Tensor permutation, Tensor lengths, Tensor values, "
        "SymInt feature_count, Tensor? weights=None) -> "
        "(Tensor, Tensor, Tensor?)"
    ),
)
def permute_2d_sparse_data(
    permutation: torch.Tensor,
    lengths: torch.Tensor,
    values: torch.Tensor,
    feature_count: int,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Permute feature-major KJT data with a runtime-derived stride."""
    stride = _permute_2d_sparse_data_stride(lengths, feature_count)
    return torch.ops.fbgemm.permute_2D_sparse_data_input1D(
        permutation,
        lengths,
        values,
        stride,
        weights,
        None,
    )


@permute_2d_sparse_data.register_fake
def _fake_permute_2d_sparse_data(
    permutation: torch.Tensor,
    lengths: torch.Tensor,
    values: torch.Tensor,
    feature_count: int,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    stride = _permute_2d_sparse_data_stride(lengths, feature_count)
    output_lengths = lengths.new_empty((permutation.shape[0] * stride,))
    output_size = torch.library.get_ctx().new_dynamic_size()
    torch._check_is_size(output_size)
    return (
        output_lengths,
        values.new_empty((output_size,)),
        weights.new_empty((output_size,)) if weights is not None else None,
    )


def get_permute_2d_sparse_data_op() -> torch._ops.OpOverload:
    return torch.ops.recir.permute_2d_sparse_data.default


@torch.library.custom_op("recir::pooled_source_to_feature_permute", mutates_args=())
def pooled_source_to_feature_permute(
    source_major_lengths: torch.Tensor,
    batch_sizes: torch.Tensor,
    feature_count: int,
) -> torch.Tensor:
    """Build the segment permutation from source-major to feature-major bags.

    For each source rank, ``source_major_lengths`` contains
    ``batch_sizes[source] * feature_count`` consecutive bag lengths.
    """
    if source_major_lengths.dim() != 1 or batch_sizes.dim() != 1:
        raise ValueError("Pooled lengths and batch sizes must be one-dimensional.")
    if feature_count <= 0:
        raise ValueError("Pooled feature count must be positive.")
    source_spans = batch_sizes * feature_count
    source_ids = torch.repeat_interleave(
        torch.arange(batch_sizes.shape[0], device=batch_sizes.device),
        source_spans,
        output_size=source_major_lengths.shape[0],
    )
    source_starts = source_spans.cumsum(dim=0) - source_spans
    source_indices = torch.arange(
        source_major_lengths.shape[0], device=source_major_lengths.device
    )
    source_positions = source_indices - source_starts[source_ids]
    source_batch_sizes = batch_sizes[source_ids]
    feature_ids = torch.div(
        source_positions,
        source_batch_sizes,
        rounding_mode="floor",
    )
    bag_ids = torch.remainder(source_positions, source_batch_sizes)
    batch_starts = batch_sizes.cumsum(dim=0) - batch_sizes
    target_positions = (
        feature_ids * batch_sizes.sum() + batch_starts[source_ids] + bag_ids
    )
    return torch.empty_like(source_indices).scatter(
        0,
        target_positions,
        source_indices,
    )


@pooled_source_to_feature_permute.register_fake
def _fake_pooled_source_to_feature_permute(
    source_major_lengths: torch.Tensor,
    batch_sizes: torch.Tensor,
    feature_count: int,
) -> torch.Tensor:
    del batch_sizes, feature_count
    return source_major_lengths.new_empty(
        source_major_lengths.shape,
        dtype=torch.int64,
    )


@torch.library.custom_op("recir::sum_pooled_embeddings", mutates_args=())
def sum_pooled_embeddings(
    partials: torch.Tensor,
    source_offsets: torch.Tensor,
    feature_count: int,
    world_size: int,
    output_width: int,
) -> torch.Tensor:
    """Combine one row-wise pooled partial per owner and source bag."""
    if partials.dim() != 1:
        raise ValueError("Pooled embedding partials must be one-dimensional.")
    if source_offsets.dim() != 1:
        raise ValueError("Pooled source offsets must be one-dimensional.")
    if feature_count <= 0:
        raise ValueError("Pooled feature count must be positive.")
    if (source_offsets.shape[0] - 1) % feature_count != 0:
        raise ValueError("Pooled source offsets do not contain equal-sized features.")
    batch_size = (source_offsets.shape[0] - 1) // feature_count
    if partials.shape[0] != world_size * batch_size * output_width:
        raise ValueError("Pooled embedding partial size does not match the mesh.")
    return partials.reshape(world_size, batch_size, output_width).sum(dim=0)


@sum_pooled_embeddings.register_fake
def _fake_sum_pooled_embeddings(
    partials: torch.Tensor,
    source_offsets: torch.Tensor,
    feature_count: int,
    world_size: int,
    output_width: int,
) -> torch.Tensor:
    del world_size
    batch_size = (source_offsets.shape[0] - 1) // feature_count
    return partials.new_empty((batch_size, output_width))


@torch.library.custom_op("recir::reshape_pooled_gradients", mutates_args=())
def reshape_pooled_gradients(
    gradients: torch.Tensor,
    local_embeddings: torch.Tensor,
) -> torch.Tensor:
    """Restore owner-local bag gradients after a flat all-to-all."""
    if gradients.dim() != 1:
        raise ValueError("Pooled gradients must be one-dimensional.")
    if local_embeddings.dim() != 2:
        raise ValueError("Pooled local embeddings must be two-dimensional.")
    if gradients.shape[0] != local_embeddings.numel():
        raise ValueError("Pooled gradient size does not match local embeddings.")
    return gradients.reshape(local_embeddings.shape).clone()


@reshape_pooled_gradients.register_fake
def _fake_reshape_pooled_gradients(
    gradients: torch.Tensor,
    local_embeddings: torch.Tensor,
) -> torch.Tensor:
    return gradients.new_empty(local_embeddings.shape)


def register_tbe_group(group: Any) -> int:
    handle = id(group)
    _TBE_GROUPS[handle] = group
    _TBE_ITER_CPU[handle] = _HostIterationCounter()
    return handle


def _get_tbe_group(handle: int) -> Any:
    group = _TBE_GROUPS.get(handle)
    if group is None:
        raise RuntimeError(f"Unknown ReCIR TBE group handle: {handle}.")
    return group


@torch.library.custom_op("recir::tbe_forward", mutates_args=())
def tbe_forward(
    handle: int,
    state_token: torch.Tensor,
    autograd_placeholder: torch.Tensor,
    tbe_state: list[torch.Tensor],
    learning_rate: torch.Tensor,
    indices: torch.Tensor,
    offsets: torch.Tensor,
    pooled: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    del state_token, autograd_placeholder
    group = _get_tbe_group(handle)
    if pooled != (group.pooling_mode != "none"):
        raise RuntimeError("TBE pooling metadata does not match the registered group.")
    autograd_key = (
        torch._C.DispatchKey.AutogradCUDA
        if indices.device.type == "cuda"
        else torch._C.DispatchKey.AutogradCPU
    )
    autograd_key_set = torch._C.DispatchKeySet(autograd_key)
    include = torch._C._dispatch_tls_local_include_set() | autograd_key_set
    exclude = torch._C._dispatch_tls_local_exclude_set() - autograd_key_set
    if len(tbe_state) != len(group._recir_tbe_state_names):
        raise RuntimeError(
            "TBE state count changed after tracing: "
            f"expected {len(group._recir_tbe_state_names)}, got {len(tbe_state)}."
        )
    state = dict(zip(group._recir_tbe_state_names, tbe_state))
    if learning_rate.device.type != "cpu":
        raise RuntimeError("TBE learning-rate tensor must be on CPU.")
    if learning_rate.dtype != torch.float32 or learning_rate.dim() != 0:
        raise RuntimeError("TBE learning-rate tensor must be a float32 scalar.")
    iter_cpu = _TBE_ITER_CPU.get(handle)
    if iter_cpu is None:
        raise RuntimeError(
            f"No host iteration counter registered for TBE handle {handle}."
        )
    with (
        _disable_current_modes(),
        torch.inference_mode(False),
        torch.enable_grad(),
        torch._C._ForceDispatchKeyGuard(include, exclude),
    ):
        # AOT runtime inputs are detached; native TBE needs a local autograd leaf.
        state["placeholder_autograd_tensor"] = torch.empty(
            0, dtype=torch.float32, device=indices.device, requires_grad=True
        )
        with stateless._reparametrize_module(group.tbe, state):
            group.tbe.iter_cpu = iter_cpu
            group.tbe.learning_rate_tensor = learning_rate
            output = group._forward_impl(indices, offsets)
    context_id = next(_NEXT_CONTEXT)
    _TBE_CONTEXTS[context_id] = (handle, output)
    context = torch.tensor(context_id, dtype=torch.int64, device="cpu")
    return output.detach(), context


@tbe_forward.register_fake
def _fake_tbe_forward(
    handle: int,
    state_token: torch.Tensor,
    autograd_placeholder: torch.Tensor,
    tbe_state: list[torch.Tensor],
    learning_rate: torch.Tensor,
    indices: torch.Tensor,
    offsets: torch.Tensor,
    pooled: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    del state_token, autograd_placeholder, tbe_state, learning_rate
    group = _get_tbe_group(handle)
    if pooled != (group.pooling_mode != "none"):
        raise RuntimeError("TBE pooling metadata does not match the registered group.")
    if not pooled:
        output_shape = (indices.shape[0], group.embedding_dim)
    else:
        feature_count = len(group.feature_table_map)
        output_shape = (
            (offsets.shape[0] - 1) // feature_count,
            feature_count * group.embedding_dim,
        )
    output = indices.new_empty(output_shape, dtype=group.output_dtype)
    context = torch.empty((), dtype=torch.int64, device="cpu")
    return output, context


@torch.library.custom_op("recir::tbe_backward_update", mutates_args=("state_token",))
def tbe_backward_update(
    handle: int,
    state_token: torch.Tensor,
    tbe_state: list[torch.Tensor],
    context: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    del tbe_state
    context_id = int(context.item())
    context_handle, forward_output = _TBE_CONTEXTS.pop(context_id)
    if context_handle != handle:
        raise RuntimeError("TBE backward context belongs to a different group.")
    torch.autograd.backward(forward_output, grad_output)
    state_token.add_(1)
    return grad_output.new_empty((0,))


@tbe_backward_update.register_fake
def _fake_tbe_backward_update(
    handle: int,
    state_token: torch.Tensor,
    tbe_state: list[torch.Tensor],
    context: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    del handle, state_token, tbe_state, context
    return grad_output.new_empty((0,))


class TBEFunction(torch.autograd.Function):
    """Expose separate graph nodes while retaining TBE's native autograd state."""

    @staticmethod
    def forward(
        ctx: Any,
        handle: int,
        state_token: torch.Tensor,
        autograd_placeholder: torch.Tensor,
        tbe_state: list[torch.Tensor],
        learning_rate: torch.Tensor,
        indices: torch.Tensor,
        offsets: torch.Tensor,
        pooled: bool,
    ) -> torch.Tensor:
        output, context = tbe_forward(
            handle,
            state_token,
            autograd_placeholder,
            tbe_state,
            learning_rate,
            indices,
            offsets,
            pooled,
        )
        ctx.handle = handle
        ctx.tbe_state = tbe_state
        ctx.save_for_backward(state_token, context)
        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        if len(grad_outputs) != 1 or not isinstance(grad_outputs[0], torch.Tensor):
            raise RuntimeError("TBEFunction expects one tensor gradient.")
        state_token, context = ctx.saved_tensors
        tbe_backward_update(
            ctx.handle,
            state_token,
            ctx.tbe_state,
            context,
            grad_outputs[0].contiguous(),
        )
        return None, None, None, None, None, None, None, None
