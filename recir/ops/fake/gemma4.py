# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors

"""FakeTensor registration for Gemma4 MoE expert dispatch.

Gemma4TextExperts.forward() uses nonzero() and torch.where() for
data-dependent expert routing, which is incompatible with
make_fx(tracing_mode="fake"). This module provides a fused replacement
that uses grouped_gemm with fixed output shapes.

Usage:
    from recir.ops.fake.gemma4 import fuse_gemma4_moe_layers
    fuse_gemma4_moe_layers(model)
    # Now make_fx(tracing_mode="fake") can trace through Gemma4 MoE layers.
"""

import logging
import types
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)

_op_registered = False


def _register_gemma4_fused_experts_op():
    """Register the Gemma4 fused experts custom op (idempotent)."""
    global _op_registered
    if _op_registered:
        return
    _op_registered = True

    @torch.library.custom_op("recir_gemma4::fused_experts", mutates_args=())
    def fused_experts_op(
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
        gate_up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        num_experts: int,
        act_fn_name: str,
    ) -> torch.Tensor:
        """Fused Gemma4 expert dispatch using grouped_gemm."""
        from fast_moe.kernels.triton.grouped_gemm import grouped_gemm
        from transformers.activations import ACT2FN

        act_fn = ACT2FN[act_fn_name]
        flat_indices = top_k_index.view(-1)
        intermediate_size = gate_up_proj.shape[1] // 2
        hidden_size = hidden_states.shape[-1]

        # Sort tokens by expert assignment
        _, sort_order = flat_indices.sort()
        token_indices = sort_order // top_k_index.shape[1]
        weight_indices = sort_order % top_k_index.shape[1]
        expert_weights = top_k_weights[token_indices, weight_indices]
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

        # gate_up_proj is [E, 2*I, H] → reshape to [E*2I, H] for grouped_gemm
        # grouped_gemm: w is [G*N, K], output is [M, N]. With G=E, N=2I, K=H.
        gate_up_out = grouped_gemm(
            sorted_input,
            gate_up_proj.reshape(-1, hidden_size),
            m_sizes,
        )  # [total_tokens, 2*I]
        gate, up = gate_up_out.chunk(2, dim=-1)
        act_out = act_fn(gate) * up
        # gelu_pytorch_tanh promotes bf16→fp32; cast back to match weight dtype
        # so both operands to tl.dot() in grouped_gemm have the same dtype.
        act_out = act_out.to(gate_up_proj.dtype)

        # down_proj is [E, H, I] — reshape to [E*H, I] for grouped_gemm
        down_out = grouped_gemm(
            act_out,
            down_proj.reshape(-1, intermediate_size),
            m_sizes,
        )  # [total_tokens, H]

        # Scatter back with weighting
        weighted_out = down_out * expert_weights.unsqueeze(-1)
        final = torch.zeros_like(hidden_states, dtype=weighted_out.dtype)
        final.index_add_(0, token_indices, weighted_out)
        return final.to(hidden_states.dtype)

    @fused_experts_op.register_fake
    def _(
        hidden_states,
        top_k_index,
        top_k_weights,
        gate_up_proj,
        down_proj,
        num_experts,
        act_fn_name,
    ):
        return hidden_states.new_empty(hidden_states.shape)


def fuse_gemma4_moe_layers(model: nn.Module) -> int:
    """Replace Gemma4TextExperts with fused grouped_gemm dispatch.

    Finds all modules with class name "Gemma4TextExperts" and replaces
    their forward() method with a trace-compatible version.

    Args:
        model: The model to patch (modified in-place).

    Returns:
        Number of expert layers fused.
    """
    _register_gemma4_fused_experts_op()

    fused_count = 0
    for name, mod in model.named_modules():
        if type(mod).__name__ != "Gemma4TextExperts":
            continue

        num_experts = mod.num_experts
        act_fn_name = None
        # Find activation function name from config
        for _, parent_mod in model.named_modules():
            if hasattr(parent_mod, "config") and hasattr(
                parent_mod.config, "hidden_activation"
            ):
                act_fn_name = parent_mod.config.hidden_activation
                break
        if act_fn_name is None:
            act_fn_name = "gelu_pytorch_tanh"  # Gemma4 default

        def _make_fused_forward(fn_name: str):
            def _fused_forward(
                self: Any,
                hidden_states: torch.Tensor,
                top_k_index: torch.Tensor,
                top_k_weights: torch.Tensor,
            ) -> torch.Tensor:
                return torch.ops.recir_gemma4.fused_experts(
                    hidden_states,
                    top_k_index,
                    top_k_weights,
                    self.gate_up_proj,
                    self.down_proj,
                    self.num_experts,
                    fn_name,
                )

            return _fused_forward

        mod.forward = types.MethodType(_make_fused_forward(act_fn_name), mod)
        fused_count += 1
        logger.info(
            "Fused Gemma4 MoE: %s (%d experts)",
            name,
            num_experts,
        )

    return fused_count
