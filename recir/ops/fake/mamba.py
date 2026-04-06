# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-ignore-all-errors

"""FakeTensor (Meta dispatch) registrations for mamba-ssm custom ops.

Makes mamba-ssm's Triton/CUDA kernels compatible with make_fx(tracing_mode="fake")
by wrapping them as torch.library.custom_op with register_fake implementations.

Without these registrations, make_fx traces INTO the autograd.Function.forward()
methods and fails when Triton kernels encounter FakeTensors.

Usage:
    from recir.ops.fake.mamba import patch_mamba_for_tracing
    patch_mamba_for_tracing()
    # Now make_fx(tracing_mode="fake") can trace through Mamba2 layers.

See https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html
"""

import logging
from typing import Optional

import causal_conv1d as conv1d_mod
import mamba_ssm.modules.mamba2 as mamba2_mod
import mamba_ssm.ops.triton.ssd_combined as ssd_mod
import torch
from causal_conv1d import (
    causal_conv1d_fn as _original_conv1d,
    causal_conv1d_update as _original_conv1d_update,
)
from mamba_ssm.ops.triton.ssd_combined import (
    mamba_split_conv1d_scan_combined as _original_split,
)

logger = logging.getLogger(__name__)

_patched = False


def register_mamba_split_conv1d_scan_combined_custom_op(original_fn):
    """Wrap mamba_split_conv1d_scan_combined as a custom op.

    This is the primary fused Mamba2 training path: combines in_proj split,
    conv1d, SSD scan, gated RMSNorm, and out_proj into a single kernel.
    """

    @torch.library.custom_op(
        "recir_mamba::mamba_split_conv1d_scan_combined", mutates_args=()
    )
    def op(
        zxbcdt: torch.Tensor,
        conv1d_weight: torch.Tensor,
        conv1d_bias: Optional[torch.Tensor],
        dt_bias: Optional[torch.Tensor],
        A: torch.Tensor,
        D: Optional[torch.Tensor],
        chunk_size: int,
        initial_states: Optional[torch.Tensor],
        seq_idx: Optional[torch.Tensor],
        activation: str,
        rmsnorm_weight: Optional[torch.Tensor],
        rmsnorm_eps: float,
        outproj_weight: torch.Tensor,
        outproj_bias: Optional[torch.Tensor],
        headdim: int,
        ngroups: int,
        norm_before_gate: bool,
        dt_limit_min: float,
        dt_limit_max: float,
    ) -> torch.Tensor:
        return original_fn(
            zxbcdt,
            conv1d_weight,
            conv1d_bias,
            dt_bias,
            A,
            D=D,
            chunk_size=chunk_size,
            initial_states=initial_states,
            seq_idx=seq_idx,
            activation=activation,
            rmsnorm_weight=rmsnorm_weight,
            rmsnorm_eps=rmsnorm_eps,
            outproj_weight=outproj_weight,
            outproj_bias=outproj_bias,
            headdim=headdim,
            ngroups=ngroups,
            norm_before_gate=norm_before_gate,
            dt_limit=(dt_limit_min, dt_limit_max),
        )

    @op.register_fake
    def _(
        zxbcdt,
        conv1d_weight,
        conv1d_bias,
        dt_bias,
        A,
        D,
        chunk_size,
        initial_states,
        seq_idx,
        activation,
        rmsnorm_weight,
        rmsnorm_eps,
        outproj_weight,
        outproj_bias,
        headdim,
        ngroups,
        norm_before_gate,
        dt_limit_min,
        dt_limit_max,
    ):
        batch, seqlen, _ = zxbcdt.shape
        d_model = outproj_weight.shape[0]
        return zxbcdt.new_empty((batch, seqlen, d_model))

    def wrapper(
        zxbcdt,
        conv1d_weight,
        conv1d_bias,
        dt_bias,
        A,
        D=None,
        chunk_size=256,
        initial_states=None,
        seq_idx=None,
        dt_limit=(0.0, float("inf")),
        return_final_states=False,
        activation="silu",
        rmsnorm_weight=None,
        rmsnorm_eps=1e-6,
        outproj_weight=None,
        outproj_bias=None,
        headdim=None,
        ngroups=1,
        norm_before_gate=True,
    ):
        if return_final_states:
            return original_fn(
                zxbcdt,
                conv1d_weight,
                conv1d_bias,
                dt_bias,
                A,
                D=D,
                chunk_size=chunk_size,
                initial_states=initial_states,
                seq_idx=seq_idx,
                activation=activation,
                rmsnorm_weight=rmsnorm_weight,
                rmsnorm_eps=rmsnorm_eps,
                outproj_weight=outproj_weight,
                outproj_bias=outproj_bias,
                headdim=headdim,
                ngroups=ngroups,
                norm_before_gate=norm_before_gate,
                dt_limit=dt_limit,
                return_final_states=True,
            )
        return op(
            zxbcdt,
            conv1d_weight,
            conv1d_bias,
            dt_bias,
            A,
            D,
            chunk_size,
            initial_states,
            seq_idx,
            activation,
            rmsnorm_weight,
            rmsnorm_eps,
            outproj_weight,
            outproj_bias,
            headdim,
            ngroups,
            norm_before_gate,
            dt_limit[0],
            dt_limit[1],
        )

    return wrapper


def register_causal_conv1d_custom_op(original_fn):
    """Wrap causal_conv1d_fn as a custom op."""

    @torch.library.custom_op("recir_mamba::causal_conv1d_fwd", mutates_args=())
    def op(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        activation: str,
    ) -> torch.Tensor:
        return original_fn(x=x, weight=weight, bias=bias, activation=activation)

    @op.register_fake
    def _(x, weight, bias, activation):
        return x.new_empty(x.shape)

    return op


def register_causal_conv1d_update_custom_op(original_fn):
    """Wrap causal_conv1d_update as a custom op."""

    @torch.library.custom_op(
        "recir_mamba::causal_conv1d_update", mutates_args=("conv_state",)
    )
    def op(
        x: torch.Tensor,
        conv_state: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        activation: str,
    ) -> torch.Tensor:
        return original_fn(x, conv_state, weight, bias, activation)

    @op.register_fake
    def _(x, conv_state, weight, bias, activation):
        return x.new_empty(x.shape)

    return op


def patch_mamba_for_tracing():
    """Monkey-patch mamba-ssm ops with custom_op wrappers for make_fx tracing.

    Replaces module-level function references in both source and consumer
    modules so that make_fx(tracing_mode="fake") can trace through Mamba2 layers.

    Must be called BEFORE the model is traced. Safe to call multiple times
    (patches are applied only once).
    """
    global _patched
    if _patched:
        return
    _patched = True

    # --- mamba_split_conv1d_scan_combined (primary Mamba2 training path) ---
    wrapper_split = register_mamba_split_conv1d_scan_combined_custom_op(_original_split)
    ssd_mod.mamba_split_conv1d_scan_combined = wrapper_split
    if hasattr(mamba2_mod, "mamba_split_conv1d_scan_combined"):
        mamba2_mod.mamba_split_conv1d_scan_combined = wrapper_split
    logger.info("Patched mamba_split_conv1d_scan_combined for tracing")

    # --- causal_conv1d_fn ---
    wrapper_conv1d = register_causal_conv1d_custom_op(_original_conv1d)
    conv1d_mod.causal_conv1d_fn = wrapper_conv1d
    if hasattr(mamba2_mod, "causal_conv1d_fn"):
        mamba2_mod.causal_conv1d_fn = wrapper_conv1d
    logger.info("Patched causal_conv1d_fn for tracing")

    # --- causal_conv1d_update ---
    wrapper_conv1d_update = register_causal_conv1d_update_custom_op(
        _original_conv1d_update
    )
    conv1d_mod.causal_conv1d_update = wrapper_conv1d_update
    if hasattr(mamba2_mod, "causal_conv1d_update"):
        mamba2_mod.causal_conv1d_update = wrapper_conv1d_update
    logger.info("Patched causal_conv1d_update for tracing")
