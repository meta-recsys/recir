# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Packed per-sequence projections, with explicit first-order gradients."""

from typing import Any

import torch

_LIB = torch.library.Library("recir", "FRAGMENT")
_LIB.define(
    "packed_jagged_bmm(Tensor values, Tensor weights, Tensor offsets, int max_len) -> Tensor"
)
_LIB.define(
    "packed_jagged_bmm_backward(Tensor grad, Tensor values, Tensor weights, Tensor offsets, int max_len) -> (Tensor, Tensor)"
)
_LIB.define(
    "packed_jagged_bmm_add(Tensor values, Tensor weights, Tensor bias, Tensor offsets, int max_len) -> Tensor"
)
_LIB.define(
    "packed_jagged_bmm_add_backward(Tensor grad, Tensor values, Tensor weights, Tensor bias, Tensor offsets, int max_len) -> (Tensor, Tensor, Tensor)"
)
_LIB.define(
    "packed_jagged_add(Tensor values, Tensor bias, Tensor offsets, int max_len) -> Tensor"
)
_LIB.define(
    "packed_jagged_add_backward(Tensor grad, Tensor bias, Tensor offsets, int max_len) -> (Tensor, Tensor)"
)


def _bmm(
    values: torch.Tensor,
    weights: torch.Tensor,
    offsets: torch.Tensor,
    max_len: int,
) -> torch.Tensor:
    out = values.new_zeros((values.shape[0], weights.shape[2]), dtype=torch.float32)
    for b in range(weights.shape[0]):
        start = int(offsets[b].item())
        end = min(int(offsets[b + 1].item()), start + max_len)
        out[start:end] = values[start:end].float() @ weights[b].float()
    return out


def _bmm_backward(
    grad: torch.Tensor,
    values: torch.Tensor,
    weights: torch.Tensor,
    offsets: torch.Tensor,
    max_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    dv = torch.zeros_like(values)
    dw = torch.zeros_like(weights)
    if values.dtype == weights.dtype == torch.bfloat16:
        grad = grad.to(torch.bfloat16)
    for b in range(weights.shape[0]):
        start = int(offsets[b].item())
        end = min(int(offsets[b + 1].item()), start + max_len)
        dv[start:end] = grad[start:end].float() @ weights[b].float().T
        dw[b] = values[start:end].float().T @ grad[start:end].float()
    return dv, dw


def _add(
    values: torch.Tensor,
    bias: torch.Tensor,
    offsets: torch.Tensor,
    max_len: int,
) -> torch.Tensor:
    out = torch.zeros_like(values)
    for b in range(bias.shape[0]):
        start = int(offsets[b].item())
        end = min(int(offsets[b + 1].item()), start + max_len)
        out[start:end] = values[start:end] + bias[b]
    return out


def _add_backward(
    grad: torch.Tensor,
    bias: torch.Tensor,
    offsets: torch.Tensor,
    max_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    dv = torch.zeros_like(grad)
    db = torch.zeros_like(bias)
    for b in range(bias.shape[0]):
        start = int(offsets[b].item())
        end = min(int(offsets[b + 1].item()), start + max_len)
        dv[start:end] = grad[start:end]
        db[b] = grad[start:end].sum(0)
    return dv, db


def _bmm_add(
    values: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor,
    offsets: torch.Tensor,
    max_len: int,
) -> torch.Tensor:
    return _add(_bmm(values, weights, offsets, max_len), bias, offsets, max_len)


def _bmm_add_backward(
    grad: torch.Tensor,
    values: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor,
    offsets: torch.Tensor,
    max_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dv, dw = _bmm_backward(grad, values, weights, offsets, max_len)
    _, db = _add_backward(grad, bias, offsets, max_len)
    return dv, dw, db


_LIB.impl("packed_jagged_bmm", _bmm, "CompositeExplicitAutograd")
_LIB.impl("packed_jagged_bmm_backward", _bmm_backward, "CompositeExplicitAutograd")
_LIB.impl("packed_jagged_bmm_add", _bmm_add, "CompositeExplicitAutograd")
_LIB.impl(
    "packed_jagged_bmm_add_backward", _bmm_add_backward, "CompositeExplicitAutograd"
)
_LIB.impl("packed_jagged_add", _add, "CompositeExplicitAutograd")
_LIB.impl("packed_jagged_add_backward", _add_backward, "CompositeExplicitAutograd")


@torch.library.register_fake("recir::packed_jagged_bmm")
def _bmm_fake(
    values: torch.Tensor,
    weights: torch.Tensor,
    offsets: torch.Tensor,
    max_len: int,
) -> torch.Tensor:
    return values.new_empty((values.shape[0], weights.shape[2]), dtype=torch.float32)


@torch.library.register_fake("recir::packed_jagged_bmm_backward")
def _bmm_backward_fake(
    grad: torch.Tensor,
    values: torch.Tensor,
    weights: torch.Tensor,
    offsets: torch.Tensor,
    max_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(values), torch.empty_like(weights)


@torch.library.register_fake("recir::packed_jagged_bmm_add")
def _bmm_add_fake(
    values: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor,
    offsets: torch.Tensor,
    max_len: int,
) -> torch.Tensor:
    return _bmm_fake(values, weights, offsets, max_len)


@torch.library.register_fake("recir::packed_jagged_bmm_add_backward")
def _bmm_add_backward_fake(
    grad: torch.Tensor,
    values: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor,
    offsets: torch.Tensor,
    max_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.empty_like(values), torch.empty_like(weights), torch.empty_like(bias)


@torch.library.register_fake("recir::packed_jagged_add")
def _add_fake(
    values: torch.Tensor,
    bias: torch.Tensor,
    offsets: torch.Tensor,
    max_len: int,
) -> torch.Tensor:
    return torch.empty_like(values)


@torch.library.register_fake("recir::packed_jagged_add_backward")
def _add_backward_fake(
    grad: torch.Tensor,
    bias: torch.Tensor,
    offsets: torch.Tensor,
    max_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(grad), torch.empty_like(bias)


def _setup_bmm(ctx: Any, inputs: tuple[Any, ...], output: torch.Tensor) -> None:
    values, weights, offsets, ctx.max_len = inputs
    ctx.save_for_backward(values, weights, offsets)


def _backward_bmm(ctx: Any, grad: torch.Tensor) -> tuple[Any, ...]:
    values, weights, offsets = ctx.saved_tensors
    dv, dw = torch.ops.recir.packed_jagged_bmm_backward(
        grad, values, weights, offsets, ctx.max_len
    )
    return dv, dw, None, None


def _setup_bmm_add(ctx: Any, inputs: tuple[Any, ...], output: torch.Tensor) -> None:
    values, weights, bias, offsets, ctx.max_len = inputs
    ctx.save_for_backward(values, weights, bias, offsets)


def _backward_bmm_add(ctx: Any, grad: torch.Tensor) -> tuple[Any, ...]:
    values, weights, bias, offsets = ctx.saved_tensors
    dv, dw, db = torch.ops.recir.packed_jagged_bmm_add_backward(
        grad, values, weights, bias, offsets, ctx.max_len
    )
    return dv, dw, db, None, None


def _setup_add(ctx: Any, inputs: tuple[Any, ...], output: torch.Tensor) -> None:
    _, bias, offsets, ctx.max_len = inputs
    ctx.save_for_backward(bias, offsets)


def _backward_add(ctx: Any, grad: torch.Tensor) -> tuple[Any, ...]:
    bias, offsets = ctx.saved_tensors
    dv, db = torch.ops.recir.packed_jagged_add_backward(
        grad, bias, offsets, ctx.max_len
    )
    return dv, db, None, None


torch.library.register_autograd(
    "recir::packed_jagged_bmm", _backward_bmm, setup_context=_setup_bmm
)
torch.library.register_autograd(
    "recir::packed_jagged_bmm_add", _backward_bmm_add, setup_context=_setup_bmm_add
)
torch.library.register_autograd(
    "recir::packed_jagged_add", _backward_add, setup_context=_setup_add
)


def jagged_dense_bmm(
    max_seq_len: int,
    seq_offsets: torch.Tensor,
    jagged: torch.Tensor,
    dense: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.recir.packed_jagged_bmm(
        _matmul_input(jagged), _matmul_input(dense), seq_offsets, max_seq_len
    ).to(jagged.dtype)


def jagged_dense_broadcast_add(
    max_seq_len: int,
    seq_offsets: torch.Tensor,
    jagged: torch.Tensor,
    dense: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.recir.packed_jagged_add(
        jagged.float(), dense.float(), seq_offsets, max_seq_len
    ).to(jagged.dtype)


def jagged_dense_bmm_add(
    max_seq_len: int,
    seq_offsets: torch.Tensor,
    jagged: torch.Tensor,
    dense: torch.Tensor,
    bias: torch.Tensor,
    elementwise: bool = False,
) -> torch.Tensor:
    if elementwise:
        out = torch.ops.recir.packed_jagged_bmm(
            _matmul_input(jagged), _matmul_input(dense), seq_offsets, max_seq_len
        )
        out = out + bias
    else:
        out = torch.ops.recir.packed_jagged_bmm_add(
            _matmul_input(jagged),
            _matmul_input(dense),
            bias.float(),
            seq_offsets,
            max_seq_len,
        )
    return out.to(jagged.dtype)


def _matmul_input(values: torch.Tensor) -> torch.Tensor:
    return values if values.dtype == torch.bfloat16 else values.float()
