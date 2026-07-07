# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""JAX implementations of fbgemm ops for torchax dispatch.

These registrations allow DlrmV3Dense (and other models using fbgemm sparse ops)
to run inside jax.jit via torchax.extract_jax without D2H copies.

The companion module recir.lowering.fbgemm_fallback must be imported first:
  - It defines the op schemas if fbgemm_gpu is not installed.
  - It monkey-patches concat_2D_jagged → _static_concat_2D_jagged_jagged
    (pure reshape+cat) so the jagged concat never hits fbgemm at runtime.

After those two side effects, the only fbgemm op called directly in
DlrmV3Dense.forward() is asynchronous_complete_cumsum. The rest are registered
here for completeness in case other code paths hit them.

Static-shape constraint for dense_to_jagged:
  tpu_sparse_dense_matmul returns activations in a padded layout
  (batch * max_seq_len, emb_dim), so dense_to_jagged always receives a fully
  padded input in our training loop. total_L is therefore B * max_len and is
  supplied explicitly by the caller (e.g. from the last element of offsets,
  which is static in fixed-shape batches).
"""

from typing import Any

import jax.numpy as jnp
import torch
from recir.torchax.ops.decorators import op


# ---------------------------------------------------------------------------
# asynchronous_complete_cumsum
# ---------------------------------------------------------------------------


@op(torch.ops.fbgemm.asynchronous_complete_cumsum)  # pyre-ignore[56]
def _fbgemm_acc_cumsum(lengths: Any) -> Any:
    """Exclusive prefix sum: [l0, l1, ...] → [0, l0, l0+l1, ...]."""
    cumsum = jnp.cumsum(lengths)
    zero = jnp.zeros(1, dtype=lengths.dtype)
    return jnp.concatenate([zero, cumsum])


# ---------------------------------------------------------------------------
# jagged_to_padded_dense  (and _forward alias)
# ---------------------------------------------------------------------------
#
# values:   [total_len, D] or [total_len]
# offsets:  list containing one [B+1] int tensor
# max_lengths: list containing one int
# → [B, max_len] or [B, max_len, D]


def _jax_jagged_to_padded_dense_impl(
    values: Any,
    offsets: Any,  # list[jax.Array] of length 1
    max_lengths: Any,  # list[int]
    padding_value: float = 0.0,
) -> Any:
    seq_offsets = offsets[0]  # [B+1]
    max_len: int = int(max_lengths[0])
    B: int = seq_offsets.shape[0] - 1

    starts = seq_offsets[:-1]  # [B]
    lengths = seq_offsets[1:] - seq_offsets[:-1]  # [B]
    col = jnp.arange(max_len)  # [max_len]
    mask = col[None, :] < lengths[:, None]  # [B, max_len]

    # Gather source indices; clamp to avoid out-of-bounds during JIT tracing.
    flat_idx = (starts[:, None] + col[None, :]).clip(
        0, max(values.shape[0] - 1, 0)
    )  # [B, max_len]

    if values.ndim == 1:
        gathered = values[flat_idx.reshape(-1)].reshape(B, max_len)
        return jnp.where(mask, gathered, jnp.asarray(padding_value, dtype=values.dtype))
    else:
        # values: [total_len, D]
        gathered = values[flat_idx.reshape(-1)].reshape(B, max_len, values.shape[-1])
        return jnp.where(
            mask[:, :, None],
            gathered,
            jnp.asarray(padding_value, dtype=values.dtype),
        )


@op(torch.ops.fbgemm.jagged_to_padded_dense)  # pyre-ignore[56]
def _fbgemm_jagged_to_padded_dense(
    values: Any,
    offsets: Any,
    max_lengths: Any,
    padding_value: float = 0.0,
) -> Any:
    return _jax_jagged_to_padded_dense_impl(values, offsets, max_lengths, padding_value)


@op(torch.ops.fbgemm.jagged_to_padded_dense_forward)  # pyre-ignore[56]
def _fbgemm_jagged_to_padded_dense_forward(
    values: Any,
    offsets: Any,
    max_lengths: Any,
    padding_value: float = 0.0,
) -> Any:
    return _jax_jagged_to_padded_dense_impl(values, offsets, max_lengths, padding_value)


# ---------------------------------------------------------------------------
# dense_to_jagged  (and _forward alias)
# ---------------------------------------------------------------------------
#
# dense:   [B, max_len] or [B, max_len, D]
# offsets: list containing one [B+1] int tensor
# total_L: optional int — must be provided for static shape in jax.jit.
#           If None, falls back to B * max_len (valid for fully-padded batches).
# → [total_L, D] or [total_L]


def _jax_dense_to_jagged_forward_impl(
    dense: Any,
    offsets: Any,
    total_L: Any = None,
) -> Any:
    seq_offsets = offsets[0]  # [B+1]
    B: int = dense.shape[0]
    N: int = dense.shape[1]

    if total_L is None:
        # Safe for static-padded batches where every row is fully occupied.
        # If sequences are variable-length, pass total_L explicitly.
        total_L = B * N

    lengths = seq_offsets[1:] - seq_offsets[:-1]  # [B]
    col = jnp.arange(N)  # [N]
    mask = col[None, :] < lengths[:, None]  # [B, N]

    dest = (seq_offsets[:-1, None] + col[None, :]).clip(0, max(total_L - 1, 0))

    if dense.ndim == 2:
        src = jnp.where(mask, dense, jnp.zeros_like(dense))
        result = jnp.zeros(total_L, dtype=dense.dtype)
        # pyrefly: ignore [missing-attribute]
        return result.at[dest.reshape(-1)].add(src.reshape(-1))
    else:
        D: int = dense.shape[2]
        src = jnp.where(mask[:, :, None], dense, jnp.zeros_like(dense))
        result = jnp.zeros((total_L, D), dtype=dense.dtype)
        # pyrefly: ignore [missing-attribute]
        return result.at[dest.reshape(-1), :].add(src.reshape(-1, D))


@op(torch.ops.fbgemm.dense_to_jagged_forward)  # pyre-ignore[56]
def _fbgemm_dense_to_jagged_forward(
    dense: Any,
    offsets: Any,
    total_L: Any = None,
) -> Any:
    return _jax_dense_to_jagged_forward_impl(dense, offsets, total_L)


@op(torch.ops.fbgemm.dense_to_jagged)  # pyre-ignore[56]
def _fbgemm_dense_to_jagged(
    dense: Any,
    offsets: Any,
    total_L: Any = None,
) -> Any:
    result = _jax_dense_to_jagged_forward_impl(dense, offsets, total_L)
    return result, offsets[0]


# ---------------------------------------------------------------------------
# jagged_to_padded_dense_backward  (= dense_to_jagged without the offsets ret)
# ---------------------------------------------------------------------------


@op(torch.ops.fbgemm.jagged_to_padded_dense_backward)  # pyre-ignore[56]
def _fbgemm_jagged_to_padded_dense_backward(
    grad_output: Any,
    offsets: Any,
    total_L: int,
) -> Any:
    return _jax_dense_to_jagged_forward_impl(grad_output, offsets, total_L)


# ---------------------------------------------------------------------------
# jagged_dense_elementwise_add_jagged_output
# ---------------------------------------------------------------------------


@op(torch.ops.fbgemm.jagged_dense_elementwise_add_jagged_output)  # pyre-ignore[56]
def _fbgemm_jagged_dense_add(
    jagged: Any,
    offsets: Any,
    dense: Any,
) -> Any:
    seq_offsets = offsets[0]
    max_len: int = dense.shape[1]
    padded = _jax_jagged_to_padded_dense_impl(jagged, offsets, [max_len], 0.0)
    summed = padded + dense
    total_L: int = jagged.shape[0]
    result = _jax_dense_to_jagged_forward_impl(summed, offsets, total_L)
    return result, seq_offsets


# ---------------------------------------------------------------------------
# jagged_index_select
# ---------------------------------------------------------------------------


@op(torch.ops.fbgemm.jagged_index_select)  # pyre-ignore[56]
def _fbgemm_jagged_index_select(
    values: Any,
    lengths: Any,
    indices: Any,
) -> Any:
    offsets = jnp.concatenate([jnp.zeros(1, dtype=lengths.dtype), jnp.cumsum(lengths)])

    selected_lengths = lengths[indices]
    out_offsets = jnp.concatenate(
        [jnp.zeros(1, dtype=selected_lengths.dtype), jnp.cumsum(selected_lengths)]
    )

    # Use values.shape[0] as a static upper bound for max_len and total_out so
    # this op is jit-safe. int(lengths.max()) / int(out_offsets[-1]) would
    # trigger D2H sync in eager mode and ConcretizationTypeError inside jax.jit.
    # Upper bound is correct: max(lengths) <= sum(lengths) == values.shape[0],
    # and any selected subset has sum(selected_lengths) <= values.shape[0].
    max_len: int = int(values.shape[0])
    padded = _jax_jagged_to_padded_dense_impl(values, [offsets], [max_len], 0.0)
    selected_padded = padded[indices]
    total_out: int = int(values.shape[0])
    result = _jax_dense_to_jagged_forward_impl(
        selected_padded, [out_offsets], total_out
    )
    return result, out_offsets
