# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Make ``aten._scaled_mm``'s meta tolerant of unbacked (jagged) scale sizes."""

from __future__ import annotations

from functools import wraps
from typing import Callable, Optional

import torch


def install_scaled_mm_size_oblivious_meta() -> None:
    """Patch the ``aten._scaled_mm`` meta to tolerate unbacked scale sizes.

    ``torch._meta_registrations._check_scaled_mm_sizes`` selects the fp8 scaling
    recipe with plain ``scale.numel() == 1`` / size ``==`` checks that are NOT
    size-oblivious, so an unbacked size-like scale length (e.g. an fp8 row-wise
    scale over a jagged total-token dim) raises ``GuardOnDataDependentSymNode``
    during fake-tensor tracing.  Wrap it to fall back to shape-only output
    inference -- the same approach ``meta_scaled_mm_v2`` already documents -- but
    only when an unbacked guard would otherwise crash; static shapes keep full
    validation and the eager C++ kernel still validates at runtime.  Idempotent.
    """
    import torch._meta_registrations as _mr
    from torch.fx.experimental.symbolic_shapes import GuardOnDataDependentSymNode

    if getattr(_mr, "_recir_scaled_mm_size_oblivious", False):  # pyre-ignore[16]
        return
    _orig: Callable[..., torch.Tensor] = _mr._check_scaled_mm_sizes  # pyre-ignore[16]

    @wraps(_orig)
    def _check_scaled_mm_sizes(
        self: torch.Tensor,
        mat2: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        scale_result: Optional[torch.Tensor] = None,
        out_dtype: Optional[torch.dtype] = None,
        use_fast_accum: bool = False,
    ) -> torch.Tensor:
        try:
            return _orig(
                self,
                mat2,
                scale_a,
                scale_b,
                bias,
                scale_result,
                out_dtype,
                use_fast_accum,
            )
        except GuardOnDataDependentSymNode:
            _out_dtype = out_dtype if out_dtype is not None else self.dtype
            return torch.empty(
                self.size(0), mat2.size(1), dtype=_out_dtype, device=self.device
            )

    _mr._check_scaled_mm_sizes = _check_scaled_mm_sizes  # pyre-ignore[16]
    _mr._recir_scaled_mm_size_oblivious = True  # pyre-ignore[16]
