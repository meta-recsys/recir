# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Decomposition table for SPMD tracing.

Contains decomposition functions for foreach/fused optimizer ops
that convert in-place operations to their out-of-place equivalents
with explicit copy_ calls, making them traceable by FX.
"""

from functools import partial
from typing import Any, Dict, List, Optional, Set

import torch
from torch._decomp.decompositions import native_layer_norm_backward

aten: torch._ops._OpNamespace = torch.ops.aten


def _stack_decomp(tensors: List[torch.Tensor], dim: int = 0) -> torch.Tensor:
    return aten.cat.default(
        [aten.unsqueeze.default(tensor, dim) for tensor in tensors], dim
    )


def _foreach_add_decomp(
    self: List[torch.Tensor], other: List[torch.Tensor], alpha: float = 1
) -> None:
    self_updated = aten._foreach_add.List(self, other, alpha=alpha)
    for s, s_u in zip(self, self_updated):
        s.copy_(s_u)


def _foreach_unaop_decomp(op: Any, self: List[torch.Tensor]) -> None:
    self_updated = op(self)
    for s, s_u in zip(self, self_updated):
        s.copy_(s_u)


def _foreach_binop_list_decomp(
    op: Any, self: List[torch.Tensor], other: List[torch.Tensor]
) -> None:
    self_updated = op(self, other)
    for s, s_u in zip(self, self_updated):
        s.copy_(s_u)


def _foreach_binop_scalar_decomp(
    op: Any, self: List[torch.Tensor], scalar: float = 1
) -> None:
    self_updated = op(self, scalar)
    for s, s_u in zip(self, self_updated):
        s.copy_(s_u)


def _foreach_addcop_scalar_decomp(
    op: Any,
    self: List[torch.Tensor],
    tensor1: List[torch.Tensor],
    tensor2: List[torch.Tensor],
    scalar: float = 1,
) -> None:
    self_updated = op(self, tensor1, tensor2, scalar)
    for s, s_u in zip(self, self_updated):
        s.copy_(s_u)


def _foreach_lerp_scalar_decomp(
    self: List[torch.Tensor], other: List[torch.Tensor], scalar: float
) -> None:
    self_updated = aten._foreach_lerp.Scalar(self, other, scalar)
    for s, s_u in zip(self, self_updated):
        s.copy_(s_u)


def _fused_adam_decomp(
    self: List[torch.Tensor],
    grads: List[torch.Tensor],
    exp_avgs: List[torch.Tensor],
    exp_avg_sqs: List[torch.Tensor],
    max_exp_avg_sqs: List[torch.Tensor],
    state_steps: List[torch.Tensor],
    *,
    lr: float = 1,
    beta1: float = 1,
    beta2: float = 1,
    weight_decay: float = 1,
    eps: float = 1,
    amsgrad: bool = True,
    maximize: bool = True,
    grad_scale: Optional[torch.Tensor] = None,
    found_inf: Optional[torch.Tensor] = None,
) -> None:
    orig_tuple = (self, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs)
    updated_tuple = aten._fused_adam.default(
        self,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        weight_decay=weight_decay,
        eps=eps,
        amsgrad=amsgrad,
        maximize=maximize,
        grad_scale=grad_scale,
        found_inf=found_inf,
    )

    for idx, (orig, updated) in enumerate(zip(orig_tuple, updated_tuple)):
        if idx == 1:
            # skip gradient copying as we don't need to copy gradients back
            continue
        for o, u in zip(orig, updated):
            o.copy_(u)


SPMD_DECOMP_TABLE: Dict[Any, Any] = {
    aten.stack.default: _stack_decomp,
    aten._foreach_add_.List: _foreach_add_decomp,
    aten._foreach_add_.Scalar: partial(
        _foreach_binop_scalar_decomp, aten._foreach_add.Scalar
    ),
    aten._foreach_addcdiv_.Scalar: partial(
        _foreach_addcop_scalar_decomp, aten._foreach_addcdiv.Scalar
    ),
    aten._foreach_addcmul_.Scalar: partial(
        _foreach_addcop_scalar_decomp, aten._foreach_addcmul.Scalar
    ),
    aten._foreach_div_.List: partial(
        _foreach_binop_list_decomp, aten._foreach_div.List
    ),
    aten._foreach_mul_.Scalar: partial(
        _foreach_binop_scalar_decomp, aten._foreach_mul.Scalar
    ),
    aten._foreach_div_.Scalar: partial(
        _foreach_binop_scalar_decomp, aten._foreach_div.Scalar
    ),
    aten._foreach_lerp_.Scalar: _foreach_lerp_scalar_decomp,
    aten._foreach_neg_.default: partial(
        _foreach_unaop_decomp, aten._foreach_neg.default
    ),
    aten._foreach_reciprocal_.default: partial(
        _foreach_unaop_decomp, aten._foreach_reciprocal.default
    ),
    aten._foreach_sqrt_.default: partial(
        _foreach_unaop_decomp, aten._foreach_sqrt.default
    ),
    aten._foreach_sub_.Scalar: partial(
        _foreach_binop_scalar_decomp, aten._foreach_sub.Scalar
    ),
    aten._fused_adam_.default: _fused_adam_decomp,
    aten.native_layer_norm_backward.default: native_layer_norm_backward,
}


DEDUP_TARGETS: Set[torch._ops.OpOverload] = {
    torch.ops._c10d_functional.all_reduce.default,
    torch.ops._c10d_functional.wait_tensor.default,
}
