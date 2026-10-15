# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Configuration for the recir QAT pass.

Selects which embedding tables get fake-quantized and how (bit width,
granularity, symmetric/asymmetric grid, initial clip/scale, LSQ grad-factor
policy, rounding mode).
"""

from dataclasses import dataclass, field
from typing import Literal, Union

# Sentinel meaning "compute init_clip from weight statistics per LSQ" instead
# of using a caller-supplied constant.  Preferred default because a bad
# constant (too small -> heavy clipping; too large -> wasted grid) stalls
# range learning.
INIT_CLIP_AUTO: str = "auto"

# Sentinel meaning "compute the LSQ grad_factor per call as
# ``1 / sqrt(numel * q_max)``" instead of using a caller-supplied constant.
# Balances scale / clip / zp updates against per-element weight updates.
GRAD_FACTOR_AUTO: str = "auto"


@dataclass
class QATConfig:
    """Configuration for the recir QAT pass.

    Attributes:
        target_table_names: names of embedding tables to fake-quantize. Names
            must match the ``nn.ModuleDict`` keys under
            ``TracableEmbeddingCollection.embeddings`` (i.e. ``EmbeddingConfig.name``)
            or the ``SparsecoreBackedEC`` per-feature keys.
        n_bits: number of quantization bits. Default 8.
        granularity: one of:

            * ``"per_table"``: a single scalar scale/clip per table.
            * ``"per_row"``: scale/clip shape ``[V, 1]`` -- one per vocab row.
            * ``"per_group"``: scale/clip shape ``[V, G]`` where
              ``G = embedding_dim / group_size`` -- one per contiguous group of
              ``group_size`` columns per row. Standard for INT4 embeddings
              (torchao / on-device LLM QAT uses ``group_size=32``).

            Default per_table.
        group_size: size of each group along the embedding dim; only used
            when ``granularity="per_group"``. Must divide the embedding dim.
            Default 32 (INT4 standard).
        symmetric: symmetric quantization grid (signed, centered at 0). When
            ``False``, a learnable ``qat_zero_point`` is registered on each
            target embedding and updated jointly with ``qat_clip`` via LSQ.
            Default True.
        init_clip: initial clip range. When ``"auto"`` (default), computed
            per-embedding (per-row / per-group as appropriate) from LSQ:
            ``init_clip = 2 * mean(|weight|) * sqrt(q_max)``. Explicit float
            overrides.
        init_scale: initial multiplicative output gain. Rarely needs to move
            from ``1.0``.
        grad_factor: LSQ grad multiplier applied to the reduced grads for
            ``qat_scale`` / ``qat_clip`` / ``qat_zero_point``. ``"auto"``
            uses ``1 / sqrt(numel * q_max)`` per LSQ. Explicit float overrides.
        round_mode: ``"nearest"`` (round-to-nearest, default) or
            ``"stochastic"``. Stochastic rounding uses
            ``floor(x + U(0,1))`` -- keeps ``E[SR(x)] == x``, which helps
            LSQ convergence at very low bit widths (INT4, MXFP4).
        calibration_momentum: enables the MA-min-max warmup recipe
            (Meta ADI, 2020): when set to a float in ``(0, 1)``, ``install_qat``
            registers two extra non-trainable buffers per target embedding
            (``qat_running_max`` shape-matching ``qat_clip``, and
            ``qat_calibration_active`` scalar flag). During each forward the
            batch weight magnitudes update ``qat_running_max`` in-place via
            EMA (``momentum * old + (1 - momentum) * batch_stat``). The
            parametrize fn dispatches the effective clip via
            ``where(active > 0.5, running_max, qat_clip)``, so users can
            toggle from calibration to LSQ range-learning at the right step
            via :func:`set_qat_calibration`. Call
            :func:`finalize_qat_calibration` to copy the calibrated
            ``running_max`` into ``qat_clip.data`` and disable the flag in a
            single shot. Default ``None`` (disabled).
    """

    target_table_names: set[str] = field(default_factory=set)
    n_bits: int = 8
    granularity: Literal["per_table", "per_row", "per_group"] = "per_table"
    group_size: int = 32
    symmetric: bool = True
    init_clip: Union[float, str] = INIT_CLIP_AUTO
    init_scale: float = 1.0
    grad_factor: Union[float, str] = GRAD_FACTOR_AUTO
    round_mode: Literal["nearest", "stochastic"] = "nearest"
    calibration_momentum: Union[float, None] = None
