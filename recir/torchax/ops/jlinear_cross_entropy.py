# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""JAX scan lowering for the opaque RecIR linear cross-entropy op."""

from typing import Any

import jax
import jax.numpy as jnp
import recir.ops.linear_cross_entropy_op  # noqa: F401
import torch
from jax.sharding import PartitionSpec as P
from recir.torchax.ops.decorators import op


def _chunk_inputs(
    hidden: Any,
    labels: Any,
    num_chunks: int,
    ignore_index: int,
) -> tuple[Any, Any, Any, int]:
    batch_size, sequence_length, hidden_size = hidden.shape
    chunk_size = (sequence_length + num_chunks - 1) // num_chunks
    padded_length = chunk_size * num_chunks
    sequence_padding = padded_length - sequence_length
    if sequence_padding:
        hidden = jnp.pad(hidden, ((0, 0), (0, sequence_padding), (0, 0)))
        labels = jnp.pad(
            labels,
            ((0, 0), (0, sequence_padding)),
            constant_values=ignore_index,
        )
    hidden_chunks = hidden.reshape(
        batch_size,
        num_chunks,
        chunk_size,
        hidden_size,
    ).transpose(1, 0, 2, 3)
    label_chunks = labels.reshape(
        batch_size,
        num_chunks,
        chunk_size,
    ).transpose(1, 0, 2)
    return hidden_chunks, label_chunks, hidden, sequence_length


def _scan_sharding_specs() -> tuple[Any | None, Any | None]:
    mesh_shape = getattr(jax.sharding.get_abstract_mesh(), "shape", {})
    if "torch_dist" not in mesh_shape:
        return None, None
    batch_axes: str | tuple[str, str]
    if "gspmd_data" in mesh_shape:
        batch_axes = ("gspmd_data", "torch_dist")
    else:
        batch_axes = "torch_dist"
    return P(batch_axes, None, None), P("torch_dist", None)


def _constrain_scan_carries(
    grad_weight: Any,
    grad_hidden: Any,
) -> tuple[Any, Any]:
    hidden_spec, weight_spec = _scan_sharding_specs()
    if hidden_spec is not None:
        grad_hidden = jax.lax.with_sharding_constraint(grad_hidden, hidden_spec)
    if weight_spec is not None:
        grad_weight = jax.lax.with_sharding_constraint(grad_weight, weight_spec)
    return grad_weight, grad_hidden


def _chunk_loss(
    hidden: Any,
    labels: Any,
    weight: Any,
    valid_vocab_size: int,
    ignore_index: int,
    softcap: float,
) -> Any:
    logits = jnp.einsum(
        "bth,vh->btv",
        hidden,
        weight,
    )[..., :valid_vocab_size].astype(jnp.float32)
    if softcap > 0.0:
        logits = jnp.tanh(logits / softcap) * softcap
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    safe_labels = jnp.clip(labels, 0, valid_vocab_size - 1)
    target_log_probs = jnp.take_along_axis(
        log_probs,
        safe_labels[..., None],
        axis=-1,
    )[..., 0]
    valid = labels != ignore_index
    invalid = valid & ((labels < 0) | (labels >= valid_vocab_size))
    loss = -jnp.sum(
        jnp.where(valid, target_log_probs, 0.0),
        dtype=jnp.float32,
    )
    return jnp.where(jnp.any(invalid), jnp.asarray(jnp.nan, jnp.float32), loss)


@op(torch.ops.recir.xla_linear_cross_entropy)  # pyre-ignore[56]
def linear_cross_entropy_scan(
    hidden: Any,
    labels: Any,
    weight: Any,
    num_chunks: int,
    valid_vocab_size: int,
    ignore_index: int,
    softcap: float,
) -> Any:
    hidden_chunks, label_chunks, _, _ = _chunk_inputs(
        hidden,
        labels,
        num_chunks,
        ignore_index,
    )

    def _scan_body(total_loss: Any, chunk: tuple[Any, Any]) -> tuple[Any, None]:
        hidden_chunk, label_chunk = chunk
        total_loss = total_loss + _chunk_loss(
            hidden_chunk,
            label_chunk,
            weight,
            valid_vocab_size,
            ignore_index,
            softcap,
        )
        return total_loss, None

    total_loss, _ = jax.lax.scan(
        _scan_body,
        jnp.zeros((), dtype=jnp.float32),
        (hidden_chunks, label_chunks),
    )
    return total_loss


@op(torch.ops.recir.xla_linear_cross_entropy_backward)  # pyre-ignore[56]
def linear_cross_entropy_scan_backward(
    grad_out: Any,
    hidden: Any,
    labels: Any,
    weight: Any,
    num_chunks: int,
    valid_vocab_size: int,
    ignore_index: int,
    softcap: float,
) -> tuple[Any, Any]:
    hidden_chunks, label_chunks, padded_hidden, sequence_length = _chunk_inputs(
        hidden,
        labels,
        num_chunks,
        ignore_index,
    )
    chunk_size = hidden_chunks.shape[2]

    def _scan_body(
        carry: tuple[Any, Any],
        chunk: tuple[Any, Any, Any],
    ) -> tuple[tuple[Any, Any], None]:
        grad_weight_accumulator, grad_hidden_buffer = carry
        chunk_index, hidden_chunk, label_chunk = chunk

        def _loss_for_vjp(input_weight: Any, input_hidden: Any) -> Any:
            return _chunk_loss(
                input_hidden,
                label_chunk,
                input_weight,
                valid_vocab_size,
                ignore_index,
                softcap,
            )

        _, vjp_fn = jax.vjp(_loss_for_vjp, weight, hidden_chunk)
        grad_weight_update, grad_hidden_chunk = vjp_fn(grad_out)
        grad_hidden_buffer = jax.lax.dynamic_update_slice(
            grad_hidden_buffer,
            grad_hidden_chunk,
            (0, chunk_index * chunk_size, 0),
        )
        return _constrain_scan_carries(
            grad_weight_accumulator + grad_weight_update.astype(jnp.float32),
            grad_hidden_buffer,
        ), None

    initial_carry = _constrain_scan_carries(
        jnp.zeros_like(weight, dtype=jnp.float32),
        jnp.zeros_like(padded_hidden),
    )
    (grad_weight, grad_hidden), _ = jax.lax.scan(
        _scan_body,
        initial_carry,
        (
            jnp.arange(num_chunks, dtype=jnp.int32),
            hidden_chunks,
            label_chunks,
        ),
    )
    grad_weight, grad_hidden = _constrain_scan_carries(grad_weight, grad_hidden)
    grad_hidden = grad_hidden[:, :sequence_length]
    return grad_hidden.astype(hidden.dtype), grad_weight.astype(weight.dtype)
