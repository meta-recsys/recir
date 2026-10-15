# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Opaque linear cross-entropy op for memory-bounded XLA lowering."""

import torch
import torch.nn.functional as F


_LIB = torch.library.Library("recir", "FRAGMENT")
_LIB.define(
    "xla_linear_cross_entropy(Tensor hidden, Tensor labels, Tensor weight, "
    "int num_chunks, int valid_vocab_size, int ignore_index, float softcap) "
    "-> Tensor"
)
_LIB.define(
    "xla_linear_cross_entropy_backward(Tensor grad_out, Tensor hidden, "
    "Tensor labels, Tensor weight, int num_chunks, int valid_vocab_size, "
    "int ignore_index, float softcap) -> (Tensor, Tensor)"
)


def _reference_loss(
    hidden: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor,
    valid_vocab_size: int,
    ignore_index: int,
    softcap: float,
) -> torch.Tensor:
    logits = F.linear(hidden, weight)[..., :valid_vocab_size].float()
    if softcap > 0.0:
        logits = torch.tanh(logits / softcap) * softcap
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    log_probs = F.log_softmax(
        flat_logits,
        dim=-1,
        dtype=flat_logits.dtype,
    )
    return F.nll_loss(
        log_probs,
        flat_labels,
        reduction="sum",
        ignore_index=ignore_index,
    ).float()


def _forward_fallback(
    hidden: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor,
    num_chunks: int,
    valid_vocab_size: int,
    ignore_index: int,
    softcap: float,
) -> torch.Tensor:
    del num_chunks
    return _reference_loss(
        hidden,
        labels,
        weight,
        valid_vocab_size,
        ignore_index,
        softcap,
    )


def _backward_fallback(
    grad_out: torch.Tensor,
    hidden: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor,
    num_chunks: int,
    valid_vocab_size: int,
    ignore_index: int,
    softcap: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del num_chunks
    differentiable_hidden = hidden.detach().requires_grad_(True)
    differentiable_weight = weight.detach().requires_grad_(True)
    with torch.enable_grad():
        loss = _reference_loss(
            differentiable_hidden,
            labels,
            differentiable_weight,
            valid_vocab_size,
            ignore_index,
            softcap,
        )
        grad_hidden, grad_weight = torch.autograd.grad(
            loss,
            (differentiable_hidden, differentiable_weight),
            grad_out,
        )
    return grad_hidden, grad_weight


_LIB.impl(
    "xla_linear_cross_entropy",
    _forward_fallback,
    "CompositeExplicitAutograd",
)
_LIB.impl(
    "xla_linear_cross_entropy_backward",
    _backward_fallback,
    "CompositeExplicitAutograd",
)


@torch.library.register_fake("recir::xla_linear_cross_entropy")
def _forward_meta(
    hidden: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor,
    num_chunks: int,
    valid_vocab_size: int,
    ignore_index: int,
    softcap: float,
) -> torch.Tensor:
    del labels, weight, num_chunks, valid_vocab_size, ignore_index, softcap
    return hidden.new_empty((), dtype=torch.float32)


@torch.library.register_fake("recir::xla_linear_cross_entropy_backward")
def _backward_meta(
    grad_out: torch.Tensor,
    hidden: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor,
    num_chunks: int,
    valid_vocab_size: int,
    ignore_index: int,
    softcap: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del grad_out, labels, num_chunks, valid_vocab_size, ignore_index, softcap
    return torch.empty_like(hidden), torch.empty_like(weight)


# pyre-ignore[2]: autograd registration uses a variadic context protocol.
def _setup_context(ctx, inputs, output) -> None:  # noqa
    del output
    hidden, labels, weight, num_chunks, valid_vocab_size, ignore_index, softcap = inputs
    ctx.save_for_backward(hidden, labels, weight)
    ctx.num_chunks = num_chunks
    ctx.valid_vocab_size = valid_vocab_size
    ctx.ignore_index = ignore_index
    ctx.softcap = softcap


# pyre-ignore[2,3]: autograd registration uses a dynamic context protocol.
def _backward(ctx, grad_out):  # noqa
    hidden, labels, weight = ctx.saved_tensors
    grad_hidden, grad_weight = torch.ops.recir.xla_linear_cross_entropy_backward(
        grad_out,
        hidden,
        labels,
        weight,
        ctx.num_chunks,
        ctx.valid_vocab_size,
        ctx.ignore_index,
        ctx.softcap,
    )
    return grad_hidden, None, grad_weight, None, None, None, None


torch.library.register_autograd(
    "recir::xla_linear_cross_entropy",
    _backward,
    setup_context=_setup_context,
)
