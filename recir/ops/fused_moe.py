# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-ignore-all-errors

"""Fused MoE replacement for tracing compatibility.

Replaces the sequential expert loop in NemotronHMOE (which uses
torch.where for data-dependent shapes) with a sort-and-scatter approach
that has fixed output shapes — compatible with make_fx(tracing_mode="fake").

The original NemotronHMOE.moe() method:
  1. Uses torch.where(mask) which produces data-dependent output sizes
  2. Has if token_indices.numel() > 0 (data-dependent control flow)
  Both are incompatible with FakeTensor tracing.

The fused replacement:
  1. Stacks all expert weights into contiguous [E*N, K] tensors
  2. Sorts tokens by expert assignment (fixed-shape output)
  3. Processes all tokens via grouped_gemm Triton kernel (one kernel call
     for all experts instead of a sequential loop)
  4. Scatters results back using index_add_ (fixed-shape output)

The entire fused moe() method is registered as a custom_op with
register_fake so that make_fx sees it as an opaque node. The real
implementation uses fast_moe's grouped_gemm for ~6x speedup over
the sequential expert loop.
"""

import logging
import types
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)

_op_registered = False


def _register_fused_moe_op():
    """Register the custom op once (idempotent)."""
    global _op_registered
    if _op_registered:
        return
    _op_registered = True

    @torch.library.custom_op("recir_mamba::fused_moe", mutates_args=())
    def fused_moe_op(
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        fused_up_weight: torch.Tensor,
        fused_down_weight: torch.Tensor,
        num_experts: int,
        intermediate_size: int,
        hidden_size: int,
        act_fn_name: str,
    ) -> torch.Tensor:
        """Sort-and-scatter MoE using grouped_gemm Triton kernel."""
        from fast_moe.kernels.triton.grouped_gemm import grouped_gemm
        from transformers.activations import ACT2FN

        act_fn = ACT2FN[act_fn_name]
        flat_indices = topk_indices.view(-1)

        # Sort tokens by expert assignment
        _, sort_order = flat_indices.sort()
        token_indices = sort_order // topk_indices.shape[1]
        weight_indices = sort_order % topk_indices.shape[1]
        expert_weights = topk_weights[token_indices, weight_indices]
        sorted_input = hidden_states[token_indices]

        # Compute tokens per expert
        m_sizes = torch.zeros(
            num_experts, dtype=torch.int32, device=hidden_states.device
        )
        m_sizes.scatter_add_(
            0,
            flat_indices.to(torch.int64),
            torch.ones_like(flat_indices, dtype=torch.int32),
        )

        # grouped_gemm: x @ W.T for all experts in one kernel call
        up_out = grouped_gemm(sorted_input, fused_up_weight, m_sizes)
        act_out = act_fn(up_out)
        # Activation may promote dtype (e.g. bf16→fp32); cast back to match
        # weight dtype so both operands to tl.dot() in grouped_gemm agree.
        act_out = act_out.to(fused_up_weight.dtype)
        down_out = grouped_gemm(act_out, fused_down_weight, m_sizes)

        # Scatter back with weighting
        weighted_out = down_out * expert_weights.unsqueeze(-1)
        final = torch.zeros_like(hidden_states, dtype=weighted_out.dtype)
        final.index_add_(0, token_indices, weighted_out)
        return final.to(hidden_states.dtype)

    @fused_moe_op.register_fake
    def _(
        hidden_states,
        topk_indices,
        topk_weights,
        fused_up_weight,
        fused_down_weight,
        num_experts,
        intermediate_size,
        hidden_size,
        act_fn_name,
    ):
        return hidden_states.new_empty(hidden_states.shape)


def fuse_moe_layers(model: nn.Module) -> int:
    """Replace sequential expert loops with fused matmul in MoE layers.

    Finds all modules with class name "NemotronHMOE" and replaces their
    moe() method with a trace-compatible version.

    Args:
        model: The model to patch (modified in-place).

    Returns:
        Number of MoE layers fused.
    """
    _register_fused_moe_op()

    fused_count = 0
    for name, mod in model.named_modules():
        if type(mod).__name__ != "NemotronHMOE":
            continue

        if not hasattr(mod, "experts") or len(mod.experts) == 0:
            continue

        num_experts = len(mod.experts)
        hidden_size = mod.experts[0].hidden_size
        intermediate_size = mod.experts[0].intermediate_size
        act_fn_name = mod.config.mlp_hidden_act

        # Stack expert weights into contiguous tensors
        mod._fused_up_weight = nn.Parameter(
            torch.cat(
                [mod.experts[i].up_proj.weight.data for i in range(num_experts)],
                dim=0,
            )
        )
        mod._fused_down_weight = nn.Parameter(
            torch.cat(
                [mod.experts[i].down_proj.weight.data for i in range(num_experts)],
                dim=0,
            )
        )
        mod._num_experts = num_experts
        mod._hidden_size = hidden_size
        mod._intermediate_size = intermediate_size
        mod._act_fn_name = act_fn_name

        # Delete individual expert modules to free memory
        del mod.experts
        mod.experts = nn.ModuleList()

        def _fused_moe(
            self: Any,
            hidden_states: torch.Tensor,
            topk_indices: torch.Tensor,
            topk_weights: torch.Tensor,
        ) -> torch.Tensor:
            return torch.ops.recir_mamba.fused_moe(
                hidden_states,
                topk_indices,
                topk_weights,
                self._fused_up_weight,
                self._fused_down_weight,
                self._num_experts,
                self._intermediate_size,
                self._hidden_size,
                self._act_fn_name,
            )

        mod.moe = types.MethodType(_fused_moe, mod)
        fused_count += 1
        logger.info(
            "Fused MoE: %s (%d experts, %d -> %d)",
            name,
            num_experts,
            hidden_size,
            intermediate_size,
        )

    return fused_count
