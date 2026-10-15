# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

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
from typing import Optional, Tuple

import causal_conv1d as conv1d_mod
import mamba_ssm.modules.mamba2 as mamba2_mod
import mamba_ssm.ops.selective_scan_interface as sscan_mod
import mamba_ssm.ops.triton.layernorm_gated as ln_mod
import mamba_ssm.ops.triton.selective_state_update as ssu_mod
import mamba_ssm.ops.triton.ssd_combined as ssd_mod
import torch
from causal_conv1d import (
    causal_conv1d_fn as _original_conv1d,
    causal_conv1d_update as _original_conv1d_update,
)
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _original_sscan
from mamba_ssm.ops.triton.layernorm_gated import layernorm_fn as _original_ln
from mamba_ssm.ops.triton.selective_state_update import (
    selective_state_update as _original_ssu,
)
from mamba_ssm.ops.triton.ssd_combined import (
    mamba_chunk_scan_combined as _original_chunk,
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


def register_selective_scan_custom_op(original_fn):
    """Wrap selective_scan_fn as a custom op (Mamba1 SSM)."""

    @torch.library.custom_op("recir_mamba::selective_scan_fwd", mutates_args=())
    def op(
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: Optional[torch.Tensor],
        z: Optional[torch.Tensor],
        delta_bias: Optional[torch.Tensor],
        delta_softplus: bool,
        return_last_state: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        result = original_fn(
            u,
            delta,
            A,
            B,
            C,
            D,
            z=z,
            delta_bias=delta_bias,
            delta_softplus=delta_softplus,
            return_last_state=return_last_state,
        )
        if return_last_state:
            return result
        batch, dim, seqlen = u.shape
        d_state = A.shape[1]
        dummy_state = u.new_zeros(batch, dim, d_state)
        return result, dummy_state

    @op.register_fake
    def _(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state):
        batch, dim, seqlen = u.shape
        d_state = A.shape[1]
        out = u.new_empty((batch, dim, seqlen))
        last_state = u.new_empty((batch, dim, d_state))
        return out, last_state

    def wrapper(
        u,
        delta,
        A,
        B,
        C,
        D=None,
        z=None,
        delta_bias=None,
        delta_softplus=False,
        return_last_state=False,
    ):
        out, last_state = op(
            u,
            delta,
            A,
            B,
            C,
            D,
            z,
            delta_bias,
            delta_softplus,
            return_last_state,
        )
        if return_last_state:
            return out, last_state
        return out

    return wrapper


def register_selective_state_update_custom_op(original_fn):
    """Wrap selective_state_update as a custom op."""

    @torch.library.custom_op(
        "recir_mamba::selective_state_update", mutates_args=("ssm_state",)
    )
    def op(
        ssm_state: torch.Tensor,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: Optional[torch.Tensor],
        z: Optional[torch.Tensor],
        dt_bias: Optional[torch.Tensor],
        dt_softplus: bool,
    ) -> torch.Tensor:
        return original_fn(
            ssm_state,
            x,
            dt,
            A,
            B,
            C,
            D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=dt_softplus,
        )

    @op.register_fake
    def _(ssm_state, x, dt, A, B, C, D, z, dt_bias, dt_softplus):
        return x.new_empty(x.shape)

    def wrapper(
        ssm_state,
        x,
        dt,
        A,
        B,
        C,
        D=None,
        z=None,
        dt_bias=None,
        dt_softplus=False,
    ):
        return op(ssm_state, x, dt, A, B, C, D, z, dt_bias, dt_softplus)

    return wrapper


def register_mamba_chunk_scan_combined_custom_op(original_fn):
    """Wrap mamba_chunk_scan_combined as a custom op.

    Used in the non-fused Mamba2 path (inference or when use_mem_eff_path=False).
    """

    @torch.library.custom_op("recir_mamba::mamba_chunk_scan_combined", mutates_args=())
    def op(
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        chunk_size: int,
        D: Optional[torch.Tensor],
        z: Optional[torch.Tensor],
        dt_bias: Optional[torch.Tensor],
        initial_states: Optional[torch.Tensor],
        seq_idx: Optional[torch.Tensor],
        dt_softplus: bool,
        dt_limit_min: float,
        dt_limit_max: float,
        return_final_states: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        result = original_fn(
            x,
            dt,
            A,
            B,
            C,
            chunk_size,
            D=D,
            z=z,
            dt_bias=dt_bias,
            initial_states=initial_states,
            seq_idx=seq_idx,
            dt_softplus=dt_softplus,
            dt_limit=(dt_limit_min, dt_limit_max),
            return_final_states=return_final_states,
        )
        if return_final_states:
            return result
        batch, seqlen, nheads, headdim = x.shape
        dstate = B.shape[-1]
        dummy_states = x.new_zeros(batch, nheads, headdim, dstate)
        return result, dummy_states

    @op.register_fake
    def _(
        x,
        dt,
        A,
        B,
        C,
        chunk_size,
        D,
        z,
        dt_bias,
        initial_states,
        seq_idx,
        dt_softplus,
        dt_limit_min,
        dt_limit_max,
        return_final_states,
    ):
        batch, seqlen, nheads, headdim = x.shape
        dstate = B.shape[-1]
        out = x.new_empty(x.shape)
        final_states = x.new_empty(batch, nheads, headdim, dstate)
        return out, final_states

    def wrapper(
        x,
        dt,
        A,
        B,
        C,
        chunk_size,
        D=None,
        z=None,
        dt_bias=None,
        initial_states=None,
        seq_idx=None,
        cu_seqlens=None,
        dt_softplus=False,
        dt_limit=(0.0, float("inf")),
        return_final_states=False,
        return_varlen_states=False,
    ):
        if return_varlen_states:
            return original_fn(
                x,
                dt,
                A,
                B,
                C,
                chunk_size,
                D=D,
                z=z,
                dt_bias=dt_bias,
                initial_states=initial_states,
                seq_idx=seq_idx,
                cu_seqlens=cu_seqlens,
                dt_softplus=dt_softplus,
                dt_limit=dt_limit,
                return_final_states=return_final_states,
                return_varlen_states=True,
            )
        out, final_states = op(
            x,
            dt,
            A,
            B,
            C,
            chunk_size,
            D,
            z,
            dt_bias,
            initial_states,
            seq_idx,
            dt_softplus,
            dt_limit[0],
            dt_limit[1],
            return_final_states,
        )
        if return_final_states:
            return out, final_states
        return out

    return wrapper


def register_layernorm_fn_custom_op(original_fn):
    """Wrap layernorm_fn / rmsnorm_fn as a custom op.

    Used by MambaRMSNormGated in NemotronH for gated RMS normalization.
    """

    @torch.library.custom_op("recir_mamba::layernorm_fn", mutates_args=())
    def op(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        z: Optional[torch.Tensor],
        eps: float,
        group_size: int,
        norm_before_gate: bool,
        is_rms_norm: bool,
    ) -> torch.Tensor:
        return original_fn(
            x,
            weight,
            bias,
            z=z,
            eps=eps,
            group_size=group_size,
            norm_before_gate=norm_before_gate,
            is_rms_norm=is_rms_norm,
        )

    @op.register_fake
    def _(x, weight, bias, z, eps, group_size, norm_before_gate, is_rms_norm):
        return x.new_empty(x.shape)

    def wrapper(
        x,
        weight,
        bias,
        z=None,
        eps=1e-6,
        group_size=None,
        norm_before_gate=True,
        is_rms_norm=False,
    ):
        gs = group_size if group_size is not None else x.shape[-1]
        return op(x, weight, bias, z, eps, gs, norm_before_gate, is_rms_norm)

    return wrapper


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

    # --- mamba_chunk_scan_combined (non-fused Mamba2 path) ---
    wrapper_chunk = register_mamba_chunk_scan_combined_custom_op(_original_chunk)
    ssd_mod.mamba_chunk_scan_combined = wrapper_chunk
    if hasattr(mamba2_mod, "mamba_chunk_scan_combined"):
        mamba2_mod.mamba_chunk_scan_combined = wrapper_chunk
    logger.info("Patched mamba_chunk_scan_combined for tracing")

    # --- selective_scan_fn (Mamba1) ---
    wrapper_sscan = register_selective_scan_custom_op(_original_sscan)
    sscan_mod.selective_scan_fn = wrapper_sscan
    logger.info("Patched selective_scan_fn for tracing")

    # --- selective_state_update ---
    wrapper_ssu = register_selective_state_update_custom_op(_original_ssu)
    ssu_mod.selective_state_update = wrapper_ssu
    logger.info("Patched selective_state_update for tracing")

    # --- layernorm_fn / rmsnorm_fn (gated RMSNorm used by Mamba2) ---
    wrapper_ln = register_layernorm_fn_custom_op(_original_ln)
    ln_mod.layernorm_fn = wrapper_ln
    ln_mod.rmsnorm_fn = (
        lambda x,
        weight,
        bias,
        z=None,
        eps=1e-6,
        group_size=None,
        norm_before_gate=True: wrapper_ln(
            x,
            weight,
            bias,
            z=z,
            eps=eps,
            group_size=group_size,
            norm_before_gate=norm_before_gate,
            is_rms_norm=True,
        )
    )
    logger.info("Patched layernorm_fn/rmsnorm_fn for tracing")
