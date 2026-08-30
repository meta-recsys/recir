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
    indices: torch.Tensor,
    offsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    del state_token, autograd_placeholder
    group = _get_tbe_group(handle)
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
    indices: torch.Tensor,
    offsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    del state_token, autograd_placeholder, tbe_state, offsets
    group = _get_tbe_group(handle)
    output = indices.new_empty(
        (indices.shape[0], group.embedding_dim), dtype=group.output_dtype
    )
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
        indices: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        output, context = tbe_forward(
            handle,
            state_token,
            autograd_placeholder,
            tbe_state,
            indices,
            offsets,
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
        return None, None, None, None, None, None
