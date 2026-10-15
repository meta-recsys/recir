# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Pre-trace parametrize fn for QAT.

Injected via ``_local_compile(..., _parametrize_fns=[qat_fake_quant(cfg)])``
so ``make_fx`` captures both the fwd fake-quant and its STE-backward into the
joint gm. The fn dispatches on ``module._recir_qat_meta`` -- installed by
``install_qat`` during Stage-0 eager preprocessing -- so only target embedding
weights are quantized; all other params flow through unchanged.

The fn is marked ``_recir_parametrize_wants_context = True`` so the getter in
``recir.distributed.data_parallel._register_parametrization`` passes the
owning submodule + param name at each access.
"""

from typing import Any, Callable

import torch
import torch.nn as nn
from recir.quant.config import QATConfig
from recir.quant.fake_quant import GradFactor, STEFakeQuant

# Attribute name used to stash per-module QAT metadata (populated by
# ``install_qat``; consumed by :func:`qat_fake_quant`).
QAT_META_ATTR: str = "_recir_qat_meta"


def _compute_batch_max(
    weight: torch.Tensor, granularity: str, group_size: int
) -> torch.Tensor:
    """Compute ``|weight|.max()`` reduced to match the ``qat_clip`` shape for
    the given granularity.

    Used by the MA-min-max calibration warmup to update ``qat_running_max``.

    Shapes:
        * ``per_table``: scalar.
        * ``per_row``: ``[V, 1]`` (max across each row).
        * ``per_group``: ``[V, G]`` (max across each contiguous ``group_size``
          slice of columns per row).
    """
    abs_w = weight.abs()
    if granularity == "per_row":
        return abs_w.max(dim=1, keepdim=True).values
    if granularity == "per_group":
        v, d = abs_w.shape
        assert d % group_size == 0, (
            f"per_group group_size={group_size} must divide embedding_dim={d}"
        )
        g = d // group_size
        return abs_w.reshape(v, g, group_size).max(dim=-1).values
    return abs_w.max()


# TODO(dense-qat): extend selection to nn.Linear once the config surface has a
# ``target_linear_names`` field. The parametrize mechanism itself will Just Work
# for aten.linear / aten.mm / aten.addmm because make_fx traces through the
# weight access under the same getter.


class _QatModuleMeta:
    """Per-module QAT metadata attached during Stage-0 eager preprocessing.

    Stored as ``module._recir_qat_meta``. Holds config-derived scalars needed at
    parametrize-time (bit width, granularity, symmetric grid, group size,
    round mode, LSQ grad factor, calibration momentum). The learnable
    ``qat_scale`` / ``qat_clip`` (and, for asymmetric, ``qat_zero_point``)
    tensors live as ``nn.Parameter``s directly on the module so
    ``stateless._reparametrize_module`` swaps them along with the ``weight``
    during ``make_fx``. Calibration buffers (``qat_running_max``,
    ``qat_calibration_active``) are similarly swapped when present.
    """

    __slots__ = (
        "n_bits",
        "symmetric",
        "granularity",
        "group_size",
        "round_mode",
        "grad_factor",
        "calibration_momentum",
        "weight_param_name",
    )

    def __init__(
        self,
        n_bits: int,
        symmetric: bool,
        granularity: str,
        group_size: int,
        round_mode: str,
        grad_factor: GradFactor,
        calibration_momentum: float | None = None,
        weight_param_name: str = "weight",
    ) -> None:
        self.n_bits = n_bits
        self.symmetric = symmetric
        self.granularity = granularity
        self.group_size = group_size
        self.round_mode = round_mode
        self.grad_factor = grad_factor
        self.calibration_momentum = calibration_momentum
        self.weight_param_name = weight_param_name


def qat_fake_quant(
    cfg: QATConfig,
) -> Callable[[torch.Tensor, nn.Module, str], torch.Tensor]:
    """Return a context-aware parametrize fn that applies STE fake-quant to
    target embedding weights.

    Suitable for passing to ``_local_compile(..., _parametrize_fns=[fn])`` or
    to ``recir.compile(..., _parametrize_fns=[fn])``.

    The returned fn is marked ``_recir_parametrize_wants_context = True`` so
    the parametrization getter passes the owning submodule + param name. The
    fn:

      1. Checks the module for ``_recir_qat_meta`` (installed by
         :func:`install_qat`). If absent, or if ``param_name`` does not match
         the recorded weight-param name, returns the raw value unchanged.
      2. Reads ``module._parameters["qat_scale"]``, ``qat_clip``, and
         (asymmetric only) ``qat_zero_point`` -- accessing via ``_parameters``
         (not attribute access) picks up the reparametrized proxies during
         ``make_fx`` tracing, so the STE call is captured as ordinary FX ops.
      3. Calls ``STEFakeQuant.apply(value, scale, clip, zp, ...)``. Because
         this happens *inside* ``make_fx``, both the fwd fake-quant nodes and
         the LSQ-backward nodes end up in the joint gm.
    """
    del cfg  # config is baked into per-module meta by install_qat.

    def _fn(
        value: torch.Tensor,
        module: nn.Module,
        param_name: str,
    ) -> torch.Tensor:
        meta: _QatModuleMeta | None = getattr(module, QAT_META_ATTR, None)
        if meta is None or param_name != meta.weight_param_name:
            return value
        # Access via ``_parameters`` so the stateless-reparametrize proxies
        # are the ones threaded through STEFakeQuant during make_fx trace.
        scale = module._parameters["qat_scale"]
        clip = module._parameters["qat_clip"]
        assert scale is not None and clip is not None, (
            f"install_qat must register qat_scale and qat_clip on "
            f"module for QAT weight {param_name!r}."
        )
        if meta.symmetric:
            # Zero scalar (non-learnable) — STE forward ignores it and STE
            # backward returns None grad for it (short-circuited on symmetric).
            zero_point: torch.Tensor = value.new_zeros(())
        else:
            zp = module._parameters.get("qat_zero_point")
            assert zp is not None, (
                f"install_qat with symmetric=False must register "
                f"qat_zero_point on module for QAT weight {param_name!r}."
            )
            zero_point = zp

        # Optional MA-min-max calibration warmup. When enabled, every fwd
        # updates ``qat_running_max`` in-place (EMA over batch weight
        # magnitude, per granularity), and the effective clip is switched
        # via ``torch.where(active > 0.5, running_max, qat_clip)``. The
        # ``active`` flag is a scalar buffer toggled by the public
        # :func:`set_qat_calibration` helper.
        effective_clip = clip
        if meta.calibration_momentum is not None:
            running_max = module._buffers.get("qat_running_max")
            active = module._buffers.get("qat_calibration_active")
            assert running_max is not None and active is not None, (
                "install_qat with calibration_momentum set must register "
                "qat_running_max and qat_calibration_active buffers."
            )
            batch_stat = _compute_batch_max(value, meta.granularity, meta.group_size)
            momentum = float(meta.calibration_momentum)
            # Detach so backward doesn't propagate stat-collection grads back
            # into the weight -- running_max is a measurement, not a target.
            new_running_max = (
                momentum * running_max + (1.0 - momentum) * batch_stat.detach()
            )
            # Only actually adopt the new value while calibration is on;
            # otherwise leave the buffer frozen at its calibrated value.
            gated = torch.where(active > 0.5, new_running_max, running_max)
            running_max.copy_(gated)
            effective_clip = torch.where(active > 0.5, running_max, clip)

        return STEFakeQuant.apply(
            value,
            scale,
            effective_clip,
            zero_point,
            meta.n_bits,
            meta.symmetric,
            meta.granularity,
            meta.group_size,
            meta.round_mode,
            meta.grad_factor,
        )

    # Marker consumed by ``_register_parametrization``'s getter.
    _fn._recir_parametrize_wants_context = True  # pyre-ignore[16]
    return _fn


def _has_qat_meta(module: nn.Module) -> bool:
    """Return True iff ``install_qat`` has tagged ``module``."""
    return getattr(module, QAT_META_ATTR, None) is not None


def _get_qat_meta(module: nn.Module) -> _QatModuleMeta:
    """Return the ``_QatModuleMeta`` stashed on ``module``; raise if missing."""
    meta: Any = getattr(module, QAT_META_ATTR, None)
    if meta is None:
        raise ValueError(
            f"Module {type(module).__name__} has no _recir_qat_meta; "
            f"install_qat did not tag it."
        )
    return meta
