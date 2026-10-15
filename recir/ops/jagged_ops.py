# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""``recir::dense_to_jagged``, ``recir::jagged_to_padded_dense``,
``recir::concat_2d_jagged``, and ``recir::split_2d_jagged`` custom ops.

These ops provide shard-local jagged operations for non-uniform offsets
on TPU/GSPMD. The torchax lowering runs inside jax.shard_map over the
"torch_dist" batch axis, keeping activations batch-sharded and avoiding
the 122GB replicated buffer OOM.

The ops are each other's backward:
- dense_to_jagged <-> jagged_to_padded_dense
- concat_2d_jagged <-> split_2d_jagged
"""

from typing import Any, List, Optional, Sequence

import torch
# Pure-torch jagged<->dense gathers (eager-CPU fallback for the custom ops
# below). Self-contained so this module is a leaf -- fbgemm_fallback routes
# to these custom ops rather than importing a shared impl, so no BUCK cycle.


def _jagged_to_padded_dense_impl(
    values: torch.Tensor,
    offsets: List[torch.Tensor],
    max_lengths: Sequence[int],
    padding_value: float = 0.0,
) -> torch.Tensor:
    seq_offsets = offsets[0]
    max_len = max_lengths[0]
    B = seq_offsets.shape[0] - 1

    # Fully-padded fast path
    if values.shape[0] == B * max_len:
        if values.dim() == 1:
            return values.reshape(B, max_len)
        return values.reshape(B, max_len, *values.shape[1:])

    starts = seq_offsets[:-1]  # (B,)
    lengths = seq_offsets[1:] - seq_offsets[:-1]  # (B,)
    col_range = torch.arange(max_len, device=values.device)  # (N,)
    mask = col_range.unsqueeze(0) < lengths.unsqueeze(1)  # (B, N)

    flat_indices = starts.unsqueeze(1) + col_range.unsqueeze(0)  # (B, N)
    flat_indices = flat_indices.clamp(0, max(values.shape[0] - 1, 0))

    if values.dim() == 1:
        gathered = values[flat_indices.reshape(-1)].reshape(B, max_len)
        result = torch.where(mask, gathered, torch.full_like(gathered, padding_value))
    else:
        flat_idx = flat_indices.reshape(-1)
        gathered = values[flat_idx].reshape(B, max_len, *values.shape[1:])
        mask_exp = mask.reshape(B, max_len, *([1] * (values.dim() - 1)))
        mask_exp = mask_exp.expand_as(gathered)
        result = torch.where(
            mask_exp, gathered, torch.full_like(gathered, padding_value)
        )
    return result


def _dense_to_jagged_forward_impl(
    dense: torch.Tensor,
    offsets: List[torch.Tensor],
    total_L: Optional[int] = None,
) -> torch.Tensor:
    """Forward-only variant: returns just the jagged values tensor."""
    seq_offsets = offsets[0]  # (B+1,)
    B = dense.shape[0]
    N = dense.shape[1]

    if total_L is None:
        total_L = int(seq_offsets[-1].item())

    device = dense.device
    if total_L == 0:
        out_shape = (0,) if dense.dim() == 2 else (0, dense.shape[2])
        return torch.zeros(out_shape, dtype=dense.dtype, device=device)

    # Fully-padded fast path
    if total_L == B * N:
        if dense.dim() == 2:
            return dense.reshape(total_L)
        return dense.reshape(total_L, dense.shape[2])

    lengths = seq_offsets[1:] - seq_offsets[:-1]  # (B,)

    j = torch.arange(total_L, device=device)  # (total_L,)
    markers = torch.zeros(total_L, dtype=torch.int64, device=device)
    boundaries = seq_offsets[1:-1]  # (B-1,) interior segment starts
    in_range = (boundaries < total_L).to(torch.int64)  # (B-1,)
    bidx = boundaries.clamp(0, max(total_L - 1, 0)).to(torch.int64)  # (B-1,)
    markers = markers.scatter_add(0, bidx, in_range)  # tiny scatter: B-1 updates
    seg = torch.cumsum(markers, 0).clamp(0, B - 1)  # (total_L,) segment id
    intra = j - seq_offsets[seg]  # (total_L,) offset within segment
    valid = (j < seq_offsets[B]) & (intra >= 0) & (intra < lengths[seg])
    flat_src = (seg * N + intra).clamp(0, max(B * N - 1, 0))  # (total_L,)

    zero = torch.zeros((), dtype=dense.dtype, device=device)
    if dense.dim() == 2:
        gathered = dense.reshape(B * N)[flat_src]
        return torch.where(valid, gathered, zero)
    else:
        D = dense.shape[2]
        gathered = dense.reshape(B * N, D)[flat_src]
        return torch.where(valid.unsqueeze(1), gathered, zero)


_LIB = torch.library.Library("recir", "FRAGMENT")

# Schema for dense_to_jagged: (dense, offsets, total_L?) -> (jagged, offsets)
# Returns offsets for API compatibility with fbgemm op
_DENSE_TO_JAGGED_SCHEMA = "dense_to_jagged(Tensor dense, Tensor[] offsets, int? total_L=None) -> (Tensor, Tensor)"

# Schema for jagged_to_padded_dense: (values, offsets, max_lengths, padding_value?) -> Tensor
_JAGGED_TO_PADDED_DENSE_SCHEMA = "jagged_to_padded_dense(Tensor values, Tensor[] offsets, int[] max_lengths, float padding_value=0) -> Tensor"

# Schema for concat_2d_jagged: (values_left, offsets_left, values_right, offsets_right) -> Tensor
# Concatenates two jagged tensors along the sequence dimension. The backward is
# split_2d_jagged (registered via register_autograd), so no separate backward op.
_CONCAT_2D_JAGGED_SCHEMA = "concat_2d_jagged(Tensor values_left, Tensor offsets_left, Tensor values_right, Tensor offsets_right) -> Tensor"

# Schema for split_2d_jagged: (values, offsets_left, offsets_right) -> (Tensor, Tensor)
# Splits a jagged tensor into left and right parts. The backward is
# concat_2d_jagged (registered via register_autograd), so no separate backward op.
_SPLIT_2D_JAGGED_SCHEMA = "split_2d_jagged(Tensor values, Tensor offsets_left, Tensor offsets_right, SymInt seq_len_a, SymInt seq_len_b) -> (Tensor, Tensor)"

_LIB.define(_DENSE_TO_JAGGED_SCHEMA)
_LIB.define(_JAGGED_TO_PADDED_DENSE_SCHEMA)
_LIB.define(_CONCAT_2D_JAGGED_SCHEMA)
_LIB.define(_SPLIT_2D_JAGGED_SCHEMA)


def _dense_to_jagged_cpu(
    dense: torch.Tensor,
    offsets: List[torch.Tensor],
    total_L: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU fallback: use the existing fbgemm_fallback implementation."""
    result = _dense_to_jagged_forward_impl(dense, offsets, total_L)
    return result, offsets[0]


def _jagged_to_padded_dense_cpu(
    values: torch.Tensor,
    offsets: List[torch.Tensor],
    max_lengths: List[int],
    padding_value: float = 0.0,
) -> torch.Tensor:
    """CPU fallback: use the existing fbgemm_fallback implementation."""
    return _jagged_to_padded_dense_impl(values, offsets, max_lengths, padding_value)


# Register CPU implementations
_LIB.impl("dense_to_jagged", _dense_to_jagged_cpu, "CompositeExplicitAutograd")
_LIB.impl(
    "jagged_to_padded_dense", _jagged_to_padded_dense_cpu, "CompositeExplicitAutograd"
)


# Fake tensor implementations for torch.compile
@torch.library.register_fake("recir::dense_to_jagged")
def _dense_to_jagged_fake(
    dense: torch.Tensor,
    offsets: List[torch.Tensor],
    total_L: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fake tensor implementation: infer output shape from inputs."""
    seq_offsets = offsets[0]
    if total_L is None:
        total_L = int(seq_offsets[-1].item()) if seq_offsets.numel() > 0 else 0

    # Output is [total_L, D] where D is dense.shape[2] if 3D, else 1D
    if dense.dim() == 2:
        out_shape = (total_L,)
    else:
        out_shape = (total_L, dense.shape[2])

    # Allocate outputs in the AMBIENT fake mode (do NOT open a new
    # FakeTensorMode -- the inputs belong to make_fx's fake mode and mixing
    # modes is unsupported -> "Mixing fake modes NYI"). new_empty/empty_like
    # inherit the input's fake mode + device (output follows the dense/compute
    # device).
    out = dense.new_empty(out_shape)
    out_offsets = torch.empty_like(seq_offsets)
    return out, out_offsets


@torch.library.register_fake("recir::jagged_to_padded_dense")
def _jagged_to_padded_dense_fake(
    values: torch.Tensor,
    offsets: List[torch.Tensor],
    max_lengths: List[int],
    padding_value: float = 0.0,
) -> torch.Tensor:
    """Fake tensor implementation: infer output shape from inputs."""
    seq_offsets = offsets[0]
    B = seq_offsets.shape[0] - 1
    max_len = max_lengths[0]

    # Output is [B, max_len, D] where D is values.shape[1] if 2D
    if values.dim() == 1:
        out_shape = (B, max_len)
    else:
        out_shape = (B, max_len, values.shape[1])

    # Allocate in the ambient fake mode (no new FakeTensorMode -- see above).
    out = values.new_empty(out_shape)
    return out


def _concat_2d_jagged_cpu(
    values_left: torch.Tensor,
    offsets_left: torch.Tensor,
    values_right: torch.Tensor,
    offsets_right: torch.Tensor,
) -> torch.Tensor:
    """CPU fallback: per-sample interleave concat via searchsorted+index_select.

    Self-contained to avoid a dependency cycle with dispatchers that route
    into this op.
    """
    # A globally-empty side has nothing to interleave; the combined jagged is
    # just the other side (per-sample order is preserved by the offset sums).
    # Without this, the empty side's index_select clamps to [0, -1] and throws.
    if values_left.shape[0] == 0:
        return values_right
    if values_right.shape[0] == 0:
        return values_left
    n = offsets_left.shape[0] - 1
    la = offsets_left[1:] - offsets_left[:-1]
    comb = offsets_left + offsets_right  # cumulative combined offsets
    total = values_left.shape[0] + values_right.shape[0]
    p = torch.arange(total, device=values_left.device)
    b = torch.clamp(torch.searchsorted(comb[1:], p, right=True), 0, n - 1)
    intra = p - comb[b]
    is_left = intra < la[b]
    src_left = torch.clamp(offsets_left[b] + intra, 0, values_left.shape[0] - 1)
    src_right = torch.clamp(
        offsets_right[b] + (intra - la[b]), 0, values_right.shape[0] - 1
    )
    gl = values_left.index_select(0, src_left)
    gr = values_right.index_select(0, src_right)
    mask = is_left if values_left.dim() == 1 else is_left.unsqueeze(-1)
    return torch.where(mask, gl, gr)


def _split_2d_jagged_cpu(
    values: torch.Tensor,
    offsets_left: torch.Tensor,
    offsets_right: torch.Tensor,
    seq_len_a: int,
    seq_len_b: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU fallback: inverse interleave split via searchsorted+index_select.

    Self-contained to avoid a dependency cycle with the dispatcher that routes
    into this op.
    """
    n = offsets_left.shape[0] - 1
    pa = torch.arange(seq_len_a, device=values.device)
    ba = torch.clamp(torch.searchsorted(offsets_left[1:], pa, right=True), 0, n - 1)
    left = values.index_select(
        0, torch.clamp(pa + offsets_right[ba], 0, values.shape[0] - 1)
    )
    pb = torch.arange(seq_len_b, device=values.device)
    bb = torch.clamp(torch.searchsorted(offsets_right[1:], pb, right=True), 0, n - 1)
    right = values.index_select(
        0, torch.clamp(pb + offsets_left[bb + 1], 0, values.shape[0] - 1)
    )
    return left, right


# Register CPU implementations
_LIB.impl("concat_2d_jagged", _concat_2d_jagged_cpu, "CompositeExplicitAutograd")
_LIB.impl("split_2d_jagged", _split_2d_jagged_cpu, "CompositeExplicitAutograd")


# Fake tensor implementations
@torch.library.register_fake("recir::concat_2d_jagged")
def _concat_2d_jagged_fake(
    values_left: torch.Tensor,
    offsets_left: torch.Tensor,
    values_right: torch.Tensor,
    offsets_right: torch.Tensor,
) -> torch.Tensor:
    """Fake tensor implementation for concat."""
    # Output size is sum of input sizes
    total_L = values_left.shape[0] + values_right.shape[0]
    if values_left.dim() == 1:
        out_shape = (total_L,)
    else:
        out_shape = (total_L, values_left.shape[1])
    out = values_left.new_empty(out_shape)
    return out


@torch.library.register_fake("recir::split_2d_jagged")
def _split_2d_jagged_fake(
    values: torch.Tensor,
    offsets_left: torch.Tensor,
    offsets_right: torch.Tensor,
    seq_len_a: int,
    seq_len_b: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fake tensor implementation for split with caller-provided output sizes."""
    if values.dim() == 1:
        left_shape = (seq_len_a,)
        right_shape = (seq_len_b,)
    else:
        D = values.shape[1]
        left_shape = (seq_len_a, D)
        right_shape = (seq_len_b, D)

    left = values.new_empty(left_shape)
    right = values.new_empty(right_shape)
    return left, right


# Autograd for concat/split (they are duals)
def _concat_2d_jagged_autograd_setup_context(
    ctx: Any,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    output: torch.Tensor,
) -> None:
    values_left, offsets_left, values_right, offsets_right = inputs
    ctx.save_for_backward(offsets_left, offsets_right)
    ctx.left_shape = values_left.shape
    ctx.right_shape = values_right.shape


def _concat_2d_jagged_autograd_backward(
    ctx: Any,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, None, torch.Tensor, None]:
    # Backward of concat is split (per-sample), NOT a flat slice.
    offsets_left, offsets_right = ctx.saved_tensors
    seq_len_a = ctx.left_shape[0]
    seq_len_b = ctx.right_shape[0]
    grad_left, grad_right = torch.ops.recir.split_2d_jagged(
        grad_output, offsets_left, offsets_right, seq_len_a, seq_len_b
    )
    return grad_left, None, grad_right, None


def _split_2d_jagged_autograd_setup_context(
    ctx: Any,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int],
    output: tuple[torch.Tensor, torch.Tensor],
) -> None:
    values, offsets_left, offsets_right, seq_len_a, seq_len_b = inputs
    ctx.save_for_backward(offsets_left, offsets_right)
    ctx.values_shape = values.shape


def _split_2d_jagged_autograd_backward(
    ctx: Any,
    grad_left: torch.Tensor,
    grad_right: torch.Tensor,
) -> tuple[torch.Tensor, None, None, None, None]:
    # Backward of split is concat (per-sample), NOT a flat cat.
    offsets_left, offsets_right = ctx.saved_tensors
    grad_output = torch.ops.recir.concat_2d_jagged(
        grad_left, offsets_left, grad_right, offsets_right
    )
    return grad_output, None, None, None, None


torch.library.register_autograd(
    "recir::concat_2d_jagged",
    _concat_2d_jagged_autograd_backward,
    setup_context=_concat_2d_jagged_autograd_setup_context,
)
torch.library.register_autograd(
    "recir::split_2d_jagged",
    _split_2d_jagged_autograd_backward,
    setup_context=_split_2d_jagged_autograd_setup_context,
)
