# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Stage-0 ``eager_preproc`` hook that installs QAT state on a module tree.

``install_qat(cfg, optimizer)`` returns a callable suitable for
``recir.compile(..., eager_preproc=[install_qat(...)])``. Given a live
``nn.Module``, it:

  1. Walks the tree to find every ``nn.Embedding`` whose dict key under an
     ``nn.ModuleDict`` matches ``cfg.target_table_names`` (matches the
     ``TracableEmbeddingCollection.embeddings`` layout).
  2. Registers ``qat_scale`` and ``qat_clip`` as ``nn.Parameter``s on that
     embedding (shape depends on ``cfg.granularity``). For asymmetric mode
     (``cfg.symmetric=False``), also registers ``qat_zero_point``.
  3. Initializes ``qat_clip`` via the LSQ formula
     ``init_clip = 2 * mean(|w|) * sqrt(q_max)`` when ``cfg.init_clip == "auto"``
     (default), otherwise uses the caller-supplied constant.
  4. Tags the module with ``_recir_qat_meta`` so the parametrize fn
     (:func:`recir.quant.parametrize.qat_fake_quant`) can dispatch.
  5. Adds the new parameters to ``optimizer.param_groups`` (or to
     ``qat_optimizer.param_groups`` when the ``qat_optimizer`` kwarg is
     provided -- see the split-optimizer Ads/Horizon recipe below) so they
     are jointly optimized -- must happen *before* ``@recir.compile`` fires
     so ``_local_compile`` snapshots them into ``named_states`` at Stage 1.
  6. Wraps any ``SparsecoreBackedEC`` instances (Python-side) so activations
     returned from the JAX kernel flow through STE at the PyTorch boundary.

The hook is idempotent per-module: running it twice on the same module tree
skips already-tagged embeddings.
"""

import logging
import math
from typing import Callable

import torch
import torch.nn as nn
from recir.quant.config import INIT_CLIP_AUTO, QATConfig
from recir.quant.fake_quant import _q_bounds
from recir.quant.parametrize import _QatModuleMeta, QAT_META_ATTR

logger: logging.Logger = logging.getLogger(__name__)


def _find_target_embeddings(
    mod: nn.Module, target_names: set[str]
) -> list[tuple[str, str, nn.Embedding]]:
    """Return ``[(parent_fqn, dict_key, embedding), ...]`` for every
    ``nn.Embedding`` whose dict-key parent is an ``nn.ModuleDict`` and whose
    key matches ``target_names``.

    This targets the ``TracableEmbeddingCollection.embeddings[<name>]``
    layout without requiring a dependency on the ``EmbeddingConfig`` type.
    """
    matches: list[tuple[str, str, nn.Embedding]] = []
    seen: set[int] = set()
    for parent_fqn, parent in mod.named_modules():
        if not isinstance(parent, nn.ModuleDict):
            continue
        for key, submod in parent.items():
            if not isinstance(submod, nn.Embedding):
                continue
            if key not in target_names:
                continue
            if id(submod) in seen:
                continue
            seen.add(id(submod))
            matches.append((parent_fqn, key, submod))
    return matches


def _compute_lsq_init_clip(
    weight: torch.Tensor, q_max: int, granularity: str, group_size: int
) -> torch.Tensor:
    """LSQ recommended clip init: ``init_clip = 2 * mean(|w|) * sqrt(q_max)``.

    Sets Δ = init_clip / q_max = 2 * mean(|w|) / sqrt(q_max), i.e. roughly
    half a standard deviation per grid step for a Gaussian-initialised weight
    — a range that covers most values with acceptable clipping frequency.

    Returns a tensor whose shape matches the target ``qat_clip`` shape
    (scalar for per_table, ``[V, 1]`` for per_row, ``[V, G]`` for per_group).
    """
    abs_w = weight.detach().abs()
    scale_factor = 2.0 * math.sqrt(q_max)
    if granularity == "per_row":
        mean_abs = abs_w.mean(dim=1, keepdim=True)  # [V, 1]
        return mean_abs * scale_factor
    if granularity == "per_group":
        v, d = abs_w.shape
        assert d % group_size == 0, (
            f"per_group group_size={group_size} must divide embedding_dim={d}"
        )
        g = d // group_size
        # Mean over each group -> [V, G, gsize] -> mean(dim=-1) -> [V, G].
        mean_abs = abs_w.reshape(v, g, group_size).mean(dim=-1)
        return mean_abs * scale_factor
    return abs_w.mean() * scale_factor  # scalar (per_table)


def _resolve_init_clip(
    weight: torch.Tensor,
    cfg: QATConfig,
    q_max: int,
) -> torch.Tensor:
    """Return an init_clip tensor matching the target ``qat_clip`` shape.

    ``cfg.init_clip == "auto"`` -> LSQ formula from weight stats.
    Otherwise: constant tensor of the caller-supplied value.
    """
    if isinstance(cfg.init_clip, str):
        assert cfg.init_clip == INIT_CLIP_AUTO, (
            f"unknown QATConfig.init_clip {cfg.init_clip!r}; "
            f"expected a float or {INIT_CLIP_AUTO!r}"
        )
        return _compute_lsq_init_clip(weight, q_max, cfg.granularity, cfg.group_size)
    device, dtype = weight.device, weight.dtype
    val = float(cfg.init_clip)
    if cfg.granularity == "per_row":
        vocab = weight.shape[0]
        return torch.full((vocab, 1), val, device=device, dtype=dtype)
    if cfg.granularity == "per_group":
        vocab, dim = weight.shape
        assert dim % cfg.group_size == 0
        num_groups = dim // cfg.group_size
        return torch.full((vocab, num_groups), val, device=device, dtype=dtype)
    return torch.tensor(val, device=device, dtype=dtype)


def _make_qat_params(
    weight: torch.Tensor,
    cfg: QATConfig,
) -> tuple[nn.Parameter, nn.Parameter, nn.Parameter | None]:
    """Materialize learnable ``qat_scale`` / ``qat_clip`` (and
    ``qat_zero_point`` when asymmetric) for ``weight``.

    Shapes by granularity:

    * ``per_table``: scalars.
    * ``per_row``: ``[V, 1]`` tensors so they broadcast against ``[V, D]``.
    * ``per_group``: ``[V, G]`` where ``G = D / group_size`` -- one qat param
      per contiguous group of ``group_size`` columns.
    """
    device = weight.device
    dtype = weight.dtype
    _q_min, q_max = _q_bounds(cfg.n_bits, cfg.symmetric)

    if cfg.granularity == "per_row":
        vocab = weight.shape[0]
        scale_shape: tuple[int, ...] = (vocab, 1)
    elif cfg.granularity == "per_group":
        vocab, dim = weight.shape
        assert dim % cfg.group_size == 0, (
            f"per_group group_size={cfg.group_size} must divide "
            f"embedding_dim={dim} (table shape {tuple(weight.shape)})"
        )
        scale_shape = (vocab, dim // cfg.group_size)
    else:
        scale_shape = ()

    if scale_shape == ():
        scale = nn.Parameter(torch.tensor(cfg.init_scale, device=device, dtype=dtype))
    else:
        scale = nn.Parameter(
            torch.full(scale_shape, cfg.init_scale, device=device, dtype=dtype)
        )

    # LSQ init for clip; falls back to caller-supplied constant.
    clip_init = _resolve_init_clip(weight, cfg, q_max)
    clip = nn.Parameter(clip_init.clone())

    zero_point: nn.Parameter | None = None
    if not cfg.symmetric:
        # Real-valued, rounded via STE inside the fwd. Initialize at the
        # unsigned midpoint (e.g. 128 for uint8) so the effective range is
        # centered on 0 pre-training.
        zp_init_scalar = float((q_max + 1) // 2)
        if scale_shape == ():
            zp = torch.tensor(zp_init_scalar, device=device, dtype=dtype)
        else:
            zp = torch.full(scale_shape, zp_init_scalar, device=device, dtype=dtype)
        zero_point = nn.Parameter(zp)

    return scale, clip, zero_point


def _install_on_embedding(
    emb: nn.Embedding,
    cfg: QATConfig,
) -> list[nn.Parameter]:
    """Attach ``qat_scale`` / ``qat_clip`` (+ ``qat_zero_point`` if asymmetric,
    + ``qat_running_max`` / ``qat_calibration_active`` buffers when
    ``cfg.calibration_momentum`` is set) to ``emb`` and tag it with meta.

    Returns the new parameters so the caller can add them to the optimizer.
    If ``emb`` is already tagged, returns an empty list.
    """
    if getattr(emb, QAT_META_ATTR, None) is not None:
        return []
    scale, clip, zero_point = _make_qat_params(emb.weight, cfg)
    emb.register_parameter("qat_scale", scale)
    emb.register_parameter("qat_clip", clip)
    new_params: list[nn.Parameter] = [scale, clip]
    if zero_point is not None:
        emb.register_parameter("qat_zero_point", zero_point)
        new_params.append(zero_point)

    # Optional MA-min-max calibration warmup buffers.  Seed
    # ``qat_running_max`` with the same LSQ-derived value as ``qat_clip``
    # so switching between the two doesn't cause a shock when calibration
    # turns off.  ``qat_calibration_active`` is a scalar 0/1 float flag
    # toggled by :func:`set_qat_calibration`; init to 1.0 (calibration on)
    # so the very first fwd already runs in calibration mode.
    if cfg.calibration_momentum is not None:
        device = emb.weight.device
        dtype = emb.weight.dtype
        emb.register_buffer("qat_running_max", clip.detach().clone())
        emb.register_buffer(
            "qat_calibration_active",
            torch.tensor(1.0, device=device, dtype=dtype),
        )

    setattr(
        emb,
        QAT_META_ATTR,
        _QatModuleMeta(
            n_bits=cfg.n_bits,
            symmetric=cfg.symmetric,
            granularity=cfg.granularity,
            group_size=cfg.group_size,
            round_mode=cfg.round_mode,
            grad_factor=cfg.grad_factor,
            calibration_momentum=cfg.calibration_momentum,
            weight_param_name="weight",
        ),
    )
    return new_params


def _wrap_sparsecore_forward_if_available(
    ec: nn.Module,
    cfg: QATConfig,
    matched_keys: set[str],
) -> list[nn.Parameter]:
    """Delegate to :func:`recir.quant.sparsecore_hook.wrap_sparsecore_forward`
    if the caller's target depends on ``//recir/quant:sparsecore_hook``;
    otherwise no-op.

    Kept as a lazy import + try/except so downstream targets that don't touch
    SparseCore do NOT transitively pull in ``recir.embedding.sparsecore_ops`` (which
    drags in JAX, ~500 MiB of .par bloat). Users who need SC-side QAT must
    explicitly add ``//recir/quant:sparsecore_hook`` to their BUCK deps.
    """
    try:
        # pyre-ignore[21]: optional dep -- see module docstring.
        from recir.quant.sparsecore_hook import (  # @manual  # noqa: WPS433
            wrap_sparsecore_forward,
        )
    except ImportError:
        return []
    return wrap_sparsecore_forward(ec, cfg, matched_keys)


def _iter_sparsecore_modules(mod: nn.Module) -> list[nn.Module]:
    """Return every ``SparsecoreBackedEC`` under ``mod``, or ``[]`` when
    the ``recir.quant.sparsecore_hook`` module is not available (caller did
    not opt into the SC-side dep chain).
    """
    try:
        # pyre-ignore[21]: optional dep -- see :func:`_wrap_sparsecore_forward_if_available`.
        from recir.quant.sparsecore_hook import (  # @manual  # noqa: WPS433
            is_sparsecore_module,
        )
    except ImportError:
        return []
    return [
        submod for _fqn, submod in mod.named_modules() if is_sparsecore_module(submod)
    ]


def _add_params_to_optimizer(
    optimizer: torch.optim.Optimizer,
    new_params: list[nn.Parameter],
) -> None:
    """Append ``new_params`` to a fresh param_group so they get stepped.

    Uses ``add_param_group`` (which snapshots defaults into a new group)
    rather than mutating an existing group -- keeping QAT params in their
    own group makes learning-rate overrides ergonomic later.
    """
    if not new_params:
        return
    known: set[int] = set()
    for group in optimizer.param_groups:
        for p in group["params"]:
            known.add(id(p))
    fresh = [p for p in new_params if id(p) not in known]
    if not fresh:
        return
    optimizer.add_param_group({"params": fresh})


def install_qat(
    cfg: QATConfig,
    optimizer: torch.optim.Optimizer,
    qat_optimizer: torch.optim.Optimizer | None = None,
) -> Callable[[nn.Module], None]:
    """Return an ``eager_preproc`` hook that installs QAT state on a module.

    Usage::

        cfg = QATConfig(target_table_names={"table_a", "table_b"}, n_bits=8)
        recir.compile(
            parallel_mode=...,
            backend=...,
            eager_preproc=[install_qat(cfg, sparse_opt, qat_optimizer=dense_opt)],
            _parametrize_fns=[qat_fake_quant(cfg)],
        )(train_step)

    The hook runs before ``recir.trace``, so all mutations happen on the
    eager module. ``qat_scale`` / ``qat_clip`` become tracked parameters
    that ``_local_compile`` snapshots into ``named_states`` at Stage 1.

    Args:
        cfg: quantization configuration.
        optimizer: the primary optimizer (typically the one that owns the
            embedding weights, e.g. row-wise AdaGrad for sparse embeddings).
        qat_optimizer: optional separate optimizer for the QAT parameters
            (``qat_scale`` / ``qat_clip`` / ``qat_zero_point``). When provided,
            new params are registered here instead of on ``optimizer``. This
            matches the Ads/Horizon recipe: sparse embeddings on row-wise
            AdaGrad, QAT params on dense Adam with a small LR -- because the
            LSQ grad geometry is unrelated to sparse-embedding grad geometry
            and coupling them in one optimizer collapses range learning.
            When ``None`` (default), new params go onto ``optimizer``.
    """

    def _hook(mod: nn.Module) -> None:
        new_params: list[nn.Parameter] = []
        # 1. In-graph embeddings.
        matches = _find_target_embeddings(mod, set(cfg.target_table_names))
        for _parent_fqn, key, emb in matches:
            added = _install_on_embedding(emb, cfg)
            new_params.extend(added)
            if added:
                logger.debug(f"[QAT] installed qat params on embedding {key!r}")

        # 2. SparseCore embeddings (optional). Delegated to
        #    ``recir.quant.sparsecore_hook`` so downstream targets that don't
        #    depend on ``//recir/quant:sparsecore_hook`` don't transitively
        #    pull in JAX / torchrec.sparse. See the helper's docstring.
        for sc_module in _iter_sparsecore_modules(mod):
            added = _wrap_sparsecore_forward_if_available(
                sc_module, cfg, set(cfg.target_table_names)
            )
            new_params.extend(added)

        # 3. Register with the optimizer BEFORE recir.compile fires.
        target_opt = qat_optimizer if qat_optimizer is not None else optimizer
        _add_params_to_optimizer(target_opt, new_params)

    return _hook


def set_qat_calibration(mod: nn.Module, active: bool) -> None:
    """Toggle MA-min-max calibration on / off across every ``install_qat``-
    tagged embedding in ``mod``.

    While ``active=True``, every fake-quant forward updates
    ``qat_running_max`` in-place (EMA-decayed by ``cfg.calibration_momentum``)
    and uses it as the effective clip -- ``qat_clip`` is untouched. Flip to
    ``False`` when you want LSQ range-learning to take over.

    No-op on modules that were installed without a calibration config.
    """
    val = 1.0 if active else 0.0
    for _fqn, submod in mod.named_modules():
        buf = submod._buffers.get("qat_calibration_active")
        if buf is None:
            continue
        buf.fill_(val)


def finalize_qat_calibration(mod: nn.Module) -> None:
    """Seed each ``qat_clip`` with the calibrated ``qat_running_max`` and
    turn calibration off in one shot.

    Standard handoff at the end of the warmup window: freeze the batch stat
    into the learnable ``qat_clip`` so LSQ starts from the calibrated value
    rather than the LSQ auto init. Idempotent; no-op on modules that were
    installed without a calibration config.
    """
    for _fqn, submod in mod.named_modules():
        active = submod._buffers.get("qat_calibration_active")
        running_max = submod._buffers.get("qat_running_max")
        clip = submod._parameters.get("qat_clip")
        if active is None or running_max is None or clip is None:
            continue
        with torch.no_grad():
            clip.data.copy_(running_max)
        active.fill_(0.0)
