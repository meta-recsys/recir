# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""SparseCore-side QAT wrapper (kept out of the core ``recir.quant`` deps).

Split from ``recir/quant/eager_preproc.py`` so downstream targets that don't
touch SparseCore embeddings do NOT transitively pull in
``recir.embedding.sparsecore_ops`` (which drags in JAX, ~500 MiB of .par
bloat).

``install_qat`` in the core hook lazily imports :func:`wrap_sparsecore_forward`
via ``try/except ImportError`` -- users who need the SC integration must
explicitly depend on ``//recir/quant:sparsecore_hook`` in their BUCK target.
"""

from typing import Any

import torch
import torch.nn as nn
from recir.embedding.sparsecore_ops import SparsecoreBackedEC
from recir.quant.config import QATConfig
from recir.quant.fake_quant import _q_bounds, STEFakeQuant
from torchrec.sparse.jagged_tensor import JaggedTensor


def wrap_sparsecore_forward(
    ec: nn.Module,
    cfg: QATConfig,
    matched_keys: set[str],
) -> list[nn.Parameter]:
    """Monkey-patch ``SparsecoreBackedEC.forward`` (per-instance) to route
    each activation whose key is in ``matched_keys`` through STE before
    returning it.

    Creates per-key ``qat_scale`` / ``qat_clip`` (+ ``qat_zero_point`` if
    asymmetric) as ``nn.ParameterDict``s on the instance so they are jointly
    optimized. Returns the new parameters. No-op on non-``SparsecoreBackedEC``
    modules and on modules that have already been wrapped.
    """
    if not isinstance(ec, SparsecoreBackedEC):
        return []
    if getattr(ec, "_recir_qat_sc_wrapped", False):
        return []

    device = torch.device("cpu")
    dtype = torch.float32
    new_params: list[nn.Parameter] = []
    scale_dict = nn.ParameterDict()
    clip_dict = nn.ParameterDict()
    zp_dict: nn.ParameterDict | None = nn.ParameterDict() if not cfg.symmetric else None
    _q_min, q_max = _q_bounds(cfg.n_bits, cfg.symmetric)
    # For activation-side quantization we don't have a canonical weight to
    # derive init_clip from; fall back to the constant when auto is requested.
    init_clip = (
        float(cfg.init_clip)
        if not isinstance(cfg.init_clip, str)
        else 6.0  # standard well-behaved default for post-ReLU-ish activations
    )
    for key in matched_keys:
        s = nn.Parameter(torch.tensor(cfg.init_scale, device=device, dtype=dtype))
        c = nn.Parameter(torch.tensor(init_clip, device=device, dtype=dtype))
        scale_dict[key] = s
        clip_dict[key] = c
        new_params.extend([s, c])
        if zp_dict is not None:
            zp_scalar = float((q_max + 1) // 2)
            z = nn.Parameter(torch.tensor(zp_scalar, device=device, dtype=dtype))
            zp_dict[key] = z
            new_params.append(z)
    ec.add_module("_recir_qat_sc_scale", scale_dict)  # type: ignore[arg-type]
    ec.add_module("_recir_qat_sc_clip", clip_dict)  # type: ignore[arg-type]
    if zp_dict is not None:
        ec.add_module("_recir_qat_sc_zp", zp_dict)  # type: ignore[arg-type]

    original_forward = ec.forward

    def qat_forward(features: Any) -> Any:
        result = original_forward(features)
        wrapped: dict[str, JaggedTensor] = {}
        for key, jt in result.items():
            if key not in matched_keys:
                wrapped[key] = jt
                continue
            s = scale_dict[key]
            c = clip_dict[key]
            zp_tensor: torch.Tensor = (
                zp_dict[key] if zp_dict is not None else jt.values().new_zeros(())
            )
            new_values = STEFakeQuant.apply(
                jt.values(),
                s,
                c,
                zp_tensor,
                cfg.n_bits,
                cfg.symmetric,
                "per_table",
                cfg.group_size,
                cfg.round_mode,
                cfg.grad_factor,
            )
            wrapped[key] = JaggedTensor(values=new_values, lengths=jt.lengths())
        return wrapped

    ec.forward = qat_forward  # type: ignore[assignment]
    ec._recir_qat_sc_wrapped = True  # type: ignore[attr-defined]
    return new_params


def is_sparsecore_module(module: nn.Module) -> bool:
    """True iff ``module`` is a ``SparsecoreBackedEC``. Helper for the core
    QAT hook so it can skip walking the tree when SC isn't in use."""
    return isinstance(module, SparsecoreBackedEC)
