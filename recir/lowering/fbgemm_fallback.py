# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pure PyTorch fallbacks for fbgemm ops used by generative_recommenders.

When fbgemm_gpu is not available (e.g., on TPU), importing this module
registers pure PyTorch implementations of the fbgemm ops that
generative_recommenders uses internally (jagged_to_padded_dense,
dense_to_jagged, etc.).

The ops are registered with CompositeImplicitAutograd dispatch and also
added to recir's SPMD_DECOMP_TABLE so that make_fx(tracing_mode="fake")
decomposes them into standard aten ops that torchax can lower to JAX/XLA.

Usage:
    import recir.lowering.fbgemm_fallback  # noqa: F401 -- registers ops as side effect
"""

from typing import List, Optional, Sequence

import torch
from recir.decompositions import SPMD_DECOMP_TABLE


def _is_fbgemm_available() -> bool:
    try:
        torch.ops.fbgemm.jagged_to_padded_dense
        return True
    except AttributeError:
        return False


# ── Pure PyTorch implementations ────────────────────────────────────────


def _jagged_to_padded_dense_impl(
    values: torch.Tensor,
    offsets: List[torch.Tensor],
    max_lengths: Sequence[int],
    padding_value: float = 0.0,
) -> torch.Tensor:
    seq_offsets = offsets[0]
    max_len = max_lengths[0]
    B = seq_offsets.shape[0] - 1

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
    seq_offsets = offsets[0]
    N = dense.shape[1]

    if total_L is None:
        total_L = int(seq_offsets[-1].item())

    lengths = seq_offsets[1:] - seq_offsets[:-1]  # (B,)
    col_range = torch.arange(N, device=dense.device)  # (N,)
    mask = col_range.unsqueeze(0) < lengths.unsqueeze(1)  # (B, N)

    dest_indices = seq_offsets[:-1].unsqueeze(1) + col_range.unsqueeze(0)  # (B, N)
    dest_indices = torch.where(mask, dest_indices, torch.zeros_like(dest_indices))
    dest_indices = dest_indices.clamp(0, max(total_L - 1, 0))

    if dense.dim() == 2:
        result = torch.zeros(total_L, dtype=dense.dtype, device=dense.device)
        src = torch.where(mask, dense, torch.zeros_like(dense))
        result = result.scatter_add(0, dest_indices.reshape(-1), src.reshape(-1))
    else:
        D = dense.shape[2]
        result = torch.zeros(total_L, D, dtype=dense.dtype, device=dense.device)
        mask_exp = mask.unsqueeze(-1).expand_as(dense)
        src = torch.where(mask_exp, dense, torch.zeros_like(dense))
        dest_exp = dest_indices.unsqueeze(-1).expand_as(dense)
        result = result.scatter_add(0, dest_exp.reshape(-1, D), src.reshape(-1, D))

    return result


def _jagged_to_padded_dense_backward_impl(
    grad_output: torch.Tensor,
    offsets: List[torch.Tensor],
    total_L: int,
) -> torch.Tensor:
    """Backward of jagged_to_padded_dense: converts dense grad to jagged."""
    return _dense_to_jagged_forward_impl(grad_output, offsets, total_L)


def _dense_to_jagged_impl(
    dense: torch.Tensor,
    offsets: List[torch.Tensor],
    total_L: Optional[int] = None,
):
    result = _dense_to_jagged_forward_impl(dense, offsets, total_L)
    return result, offsets[0]


def _asynchronous_complete_cumsum_impl(lengths: torch.Tensor) -> torch.Tensor:
    cumsum = torch.cumsum(lengths, dim=0)
    zero = torch.zeros(1, dtype=lengths.dtype, device=lengths.device)
    return torch.cat([zero, cumsum], dim=0)


def _jagged_dense_elementwise_add_jagged_output_impl(
    jagged: torch.Tensor,
    offsets: List[torch.Tensor],
    dense: torch.Tensor,
):
    seq_offsets = offsets[0]
    max_len = dense.shape[1]

    padded = _jagged_to_padded_dense_impl(jagged, offsets, [max_len], 0.0)
    summed = padded + dense
    result, _ = _dense_to_jagged_impl(summed, offsets, jagged.shape[0])
    return result, seq_offsets


def _jagged_index_select_impl(
    values: torch.Tensor,
    lengths: torch.Tensor,
    indices: torch.Tensor,
):
    offsets = torch.zeros(
        lengths.numel() + 1, dtype=lengths.dtype, device=lengths.device
    )
    offsets[1:] = torch.cumsum(lengths, dim=0)

    selected_lengths = lengths[indices]
    out_offsets = torch.zeros(
        selected_lengths.numel() + 1,
        dtype=selected_lengths.dtype,
        device=selected_lengths.device,
    )
    out_offsets[1:] = torch.cumsum(selected_lengths, dim=0)

    max_len = int(lengths.max().item()) if lengths.numel() > 0 else 0
    N = lengths.numel()

    if max_len == 0 or N == 0:
        result = torch.empty(
            0, *values.shape[1:], dtype=values.dtype, device=values.device
        )
        return result, out_offsets

    padded = _jagged_to_padded_dense_impl(values, [offsets], [max_len], 0.0)
    selected_padded = padded[indices]
    result, _ = _dense_to_jagged_impl(
        selected_padded, [out_offsets], int(out_offsets[-1].item())
    )
    return result, out_offsets


# ── Registration ────────────────────────────────────────────────────────

# Keep the Library object alive at module level. If it is garbage-collected,
# the op schemas it defines are unregistered while the implementations remain
# dangling, causing "schema_.has_value() INTERNAL ASSERT FAILED" on any call.
_fbgemm_lib: Optional[torch.library.Library] = None


def register_fbgemm_fallbacks() -> bool:
    """Register pure PyTorch fallbacks for fbgemm ops.

    When fbgemm is not available, registers pure PyTorch op implementations.
    Always populates SPMD_DECOMP_TABLE so that make_fx decomposes fbgemm ops
    into standard aten ops for XLA tracing — even when fbgemm is installed,
    its CUDA kernels cannot run during fake-tensor tracing on TPU.

    Returns True if op fallbacks were registered, False if fbgemm was already
    available (only SPMD_DECOMP_TABLE entries were added).
    """
    global _fbgemm_lib

    registered_ops = False
    if not _is_fbgemm_available():
        lib = torch.library.Library("fbgemm", "DEF")
        _fbgemm_lib = lib

        # Define op schemas
        lib.define(
            "jagged_to_padded_dense(Tensor values, Tensor[] offsets, "
            "int[] max_lengths, float padding_value=0.0) -> Tensor"
        )
        lib.define(
            "dense_to_jagged(Tensor dense, Tensor[] offsets, "
            "int? total_L=None) -> (Tensor, Tensor)"
        )
        lib.define(
            "dense_to_jagged_forward(Tensor dense, Tensor[] offsets, "
            "int? total_L=None) -> Tensor"
        )
        lib.define(
            "jagged_to_padded_dense_forward(Tensor values, Tensor[] offsets, "
            "int[] max_lengths, float padding_value=0.0) -> Tensor"
        )
        lib.define(
            "jagged_to_padded_dense_backward(Tensor grad_output, "
            "Tensor[] offsets, int total_L) -> Tensor"
        )
        lib.define("asynchronous_complete_cumsum(Tensor lengths) -> Tensor")
        lib.define(
            "jagged_dense_elementwise_add_jagged_output("
            "Tensor jagged, Tensor[] offsets, Tensor dense"
            ") -> (Tensor, Tensor)"
        )
        lib.define(
            "jagged_index_select(Tensor values, Tensor lengths, Tensor indices"
            ") -> (Tensor, Tensor)"
        )

        # Register implementations with CompositeImplicitAutograd
        torch.library.impl(lib, "jagged_to_padded_dense", "CompositeImplicitAutograd")(
            _jagged_to_padded_dense_impl
        )
        torch.library.impl(lib, "dense_to_jagged", "CompositeImplicitAutograd")(
            _dense_to_jagged_impl
        )
        torch.library.impl(lib, "dense_to_jagged_forward", "CompositeImplicitAutograd")(
            _dense_to_jagged_forward_impl
        )
        torch.library.impl(
            lib, "jagged_to_padded_dense_forward", "CompositeImplicitAutograd"
        )(_jagged_to_padded_dense_impl)
        torch.library.impl(
            lib, "jagged_to_padded_dense_backward", "CompositeImplicitAutograd"
        )(_jagged_to_padded_dense_backward_impl)
        torch.library.impl(
            lib, "asynchronous_complete_cumsum", "CompositeImplicitAutograd"
        )(_asynchronous_complete_cumsum_impl)
        torch.library.impl(
            lib,
            "jagged_dense_elementwise_add_jagged_output",
            "CompositeImplicitAutograd",
        )(_jagged_dense_elementwise_add_jagged_output_impl)
        torch.library.impl(lib, "jagged_index_select", "CompositeImplicitAutograd")(
            _jagged_index_select_impl
        )
        registered_ops = True

    # Always add to SPMD_DECOMP_TABLE so make_fx(tracing_mode="fake")
    # decomposes these ops into standard aten ops for XLA/JAX lowering.
    # This is needed even when fbgemm is installed because the real
    # fbgemm CUDA kernels segfault during fake-tensor tracing on TPU.
    SPMD_DECOMP_TABLE[torch.ops.fbgemm.jagged_to_padded_dense.default] = (
        _jagged_to_padded_dense_impl
    )
    SPMD_DECOMP_TABLE[torch.ops.fbgemm.dense_to_jagged.default] = _dense_to_jagged_impl
    SPMD_DECOMP_TABLE[torch.ops.fbgemm.dense_to_jagged_forward.default] = (
        _dense_to_jagged_forward_impl
    )
    SPMD_DECOMP_TABLE[torch.ops.fbgemm.jagged_to_padded_dense_forward.default] = (
        _jagged_to_padded_dense_impl
    )
    SPMD_DECOMP_TABLE[torch.ops.fbgemm.jagged_to_padded_dense_backward.default] = (
        _jagged_to_padded_dense_backward_impl
    )
    SPMD_DECOMP_TABLE[torch.ops.fbgemm.asynchronous_complete_cumsum.default] = (
        _asynchronous_complete_cumsum_impl
    )
    SPMD_DECOMP_TABLE[
        torch.ops.fbgemm.jagged_dense_elementwise_add_jagged_output.default
    ] = _jagged_dense_elementwise_add_jagged_output_impl
    SPMD_DECOMP_TABLE[torch.ops.fbgemm.jagged_index_select.default] = (
        _jagged_index_select_impl
    )

    return registered_ops


# Auto-register on import
register_fbgemm_fallbacks()
