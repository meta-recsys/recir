# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Shared dual-optimizer + warmup setup for the recir train step.

Both the internal GPU (HDP) and cloud TPU (GSPMD) cells need the IDENTICAL
pre-compile optimizer wiring: split params sparse vs dense by the embedding
plan, register the two 0-d LR buffers on the model (the names
`recir_train_step` reads), build the two tensor-LR optimizers, and build +
seed a WarmupController over them. This composes partition.py + optim.py +
the LR-buffer constants so neither cell re-implements it.

"""

import logging
from dataclasses import dataclass
from typing import List, Protocol, Set, Tuple

import torch
from recir.distributed.sharding_plan import EmbeddingShardingPlan
from recir.training.compile import (
    LR_DENSE_BUFFER,
    LR_ITER_BUFFER,
    LR_SPARSE_BUFFER,
    WARMUP_ATTR,
)
from recir.training.optim import build_torch_optimizer, OptimConfig, WarmupController

logger: logging.Logger = logging.getLogger(__name__)


class _OptimConfigGroup(Protocol):
    @property
    def default(self) -> OptimConfig: ...


class RecOptimizerConfig(Protocol):
    """Dense and sparse optimizer configuration contract."""

    @property
    def dense(self) -> _OptimConfigGroup: ...

    @property
    def sparse(self) -> _OptimConfigGroup: ...


@dataclass
class RecirOptimizerBundle:
    """Everything the compile step + train loop need from optimizer setup."""

    dense_opt: torch.optim.Optimizer
    sparse_opt: torch.optim.Optimizer
    warmup: WarmupController
    plan: EmbeddingShardingPlan
    sparse_params: List[torch.nn.Parameter]
    dense_params: List[torch.nn.Parameter]


def _maybe_schedule(schedule: object) -> List[object] | None:
    return list(schedule) if schedule else None  # pyre-ignore[6]


def _partition_params(
    model: torch.nn.Module,
    sparse_fqns: Set[str],
) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    """Split trainable float params into (sparse, dense) by the plan FQN set.

    Frozen (non-requires_grad) and non-float params are skipped so a future
    ``no_grad_param_filters`` plumb-through is honored. Asserts both buckets are
    non-empty and that every plan FQN resolves on the model.
    """
    sparse_params: List[torch.nn.Parameter] = []
    dense_params: List[torch.nn.Parameter] = []
    for fqn, p in model.named_parameters():
        if not (p.requires_grad and p.is_floating_point()):
            continue
        if fqn in sparse_fqns:
            sparse_params.append(p)
        else:
            dense_params.append(p)

    unmatched_in_plan: List[str] = []
    for fqn in sparse_fqns:
        try:
            model.get_parameter(fqn)
        except AttributeError:
            unmatched_in_plan.append(fqn)
    assert not unmatched_in_plan, f"Plan FQNs missing from model: {unmatched_in_plan}"

    assert sparse_params, (
        "No sparse params resolved from the embedding plan; the FQN match must "
        "be broken."
    )
    assert dense_params, (
        "No dense params after partitioning; check that non-embedding modules "
        "still own trainable Parameters."
    )
    logger.info(
        "[recir partition] sparse_params=%d, dense_params=%d",
        len(sparse_params),
        len(dense_params),
    )
    return sparse_params, dense_params


def _register_iter_buffer(model: torch.nn.Module, device: torch.device) -> torch.Tensor:
    """Register the 0-d int64 warmup counter under ``LR_ITER_BUFFER``."""
    model.register_buffer(
        LR_ITER_BUFFER,
        torch.zeros((), device=device, dtype=torch.int64),
        persistent=False,
    )
    return getattr(model, LR_ITER_BUFFER)


def _build_lr_buffered_optimizer(
    model: torch.nn.Module,
    params: List[torch.nn.Parameter],
    cfg: OptimConfig,
    lr_buffer_name: str,
    device: torch.device,
) -> tuple[torch.optim.Optimizer, torch.Tensor]:
    """Register a non-persistent 0-d LR buffer on ``model`` and build a tensor-LR
    optimizer over ``params`` from ``cfg``. Shared by the dense-only (TPU) and
    dual (GPU) setups so the per-optimizer wiring lives in one place.

    The 0-d buffer IS the optimizer's ``group['lr']`` from construction on --
    ``build_torch_optimizer`` binds this exact object, so the trace sees a
    lifted proxy rather than a baked constant and the WarmupController can move
    the LR by mutating it in place. ``recir_train_step`` asserts that binding
    still holds before tracing. As a buffer it lands in flat_state via
    ``named_buffers()`` (tagged is_weight=True / replicated, GSPMD-safe);
    persistent=False keeps it out of state_dict.
    """
    model.register_buffer(
        lr_buffer_name,
        torch.tensor(cfg.lr, device=device, dtype=torch.float32),
        persistent=False,
    )
    lr_buffer = getattr(model, lr_buffer_name)
    opt = build_torch_optimizer(params, cfg, lr_buffer)
    return opt, lr_buffer


def build_sparse_lr_buffered_optimizer(
    model: torch.nn.Module,
    params: List[torch.nn.Parameter],
    cfg: OptimConfig,
    device: torch.device,
) -> tuple[torch.optim.Optimizer, torch.Tensor]:
    """``_build_lr_buffered_optimizer`` under ``LR_SPARSE_BUFFER``: the
    optimizer of embedding tables that train inside the compiled step (e.g.
    the recir.embedding in_graph backend's ``sparse_optimizer_builder``)."""
    return _build_lr_buffered_optimizer(model, params, cfg, LR_SPARSE_BUFFER, device)


def setup_dense_optimizer_and_warmup(
    model: torch.nn.Module,
    dense_params: List[torch.nn.Parameter],
    dense_cfg: OptimConfig,
    device: torch.device,
    *,
    sparse_cfg: OptimConfig | None = None,
    sparse_lr_buffer: torch.Tensor | None = None,
) -> tuple[torch.optim.Optimizer, WarmupController]:
    """Build the dense optimizer + warmup the SAME way the GPU path does.

    For the TPU (GSPMD) cell only the DENSE optimizer is built here -- the sparse
    embedding update is owned by the embedding backend off-GM (SparseCore Adagrad
    / FBGEMM-fused DMP). It uses the SAME ``_build_lr_buffered_optimizer`` +
    ``WarmupController`` path as the dual (GPU) setup, so the dense optimizer is
    byte-identical across cells. Pass the FULL dense ``OptimConfig`` (e.g.
    ``module_config.dense_optimizer_config``) to control optim type / betas / eps
    / weight_decay / warmup schedule.

    Registers a non-persistent 0-d ``LR_DENSE_BUFFER`` on ``model`` and binds
    the dense optimizer's group['lr'] to it.
    """
    dense_opt, dense_lr_buffer = _build_lr_buffered_optimizer(
        model, dense_params, dense_cfg, LR_DENSE_BUFFER, device
    )
    if (sparse_cfg is None) != (sparse_lr_buffer is None):
        raise ValueError(
            "sparse_cfg and sparse_lr_buffer must either both be set or both be None"
        )
    warmup = WarmupController(
        iter_buffer=_register_iter_buffer(model, device),
        dense_base_lr=dense_cfg.lr,
        dense_lr_warmup_schedule=_maybe_schedule(dense_cfg.lr_warmup_schedule),
        dense_lr_buffer=dense_lr_buffer,
        sparse_base_lr=sparse_cfg.lr if sparse_cfg is not None else None,
        sparse_lr_warmup_schedule=(
            _maybe_schedule(sparse_cfg.lr_warmup_schedule)
            if sparse_cfg is not None
            else None
        ),
        sparse_lr_buffer=sparse_lr_buffer,
    )
    warmup.apply()  # seed iter-0 LR
    setattr(model, WARMUP_ATTR, warmup)  # recir_train_step advances it in-graph
    logger.info(
        "[recir setup] dense-only optimizer: %d params, optim=%s, lr=%g, stages=%d",
        len(dense_params),
        dense_cfg.optim_type.name,
        dense_cfg.lr,
        warmup.num_dense_stages,
    )
    return dense_opt, warmup


def setup_dual_optimizers_and_warmup(
    model: torch.nn.Module,
    optimizer_config: RecOptimizerConfig,
    embedding_plan: EmbeddingShardingPlan,
    device: torch.device,
    use_tbe_sparse: bool = False,
) -> RecirOptimizerBundle:
    """Use the resolved embedding plan, split params, register LR buffers, build two
    tensor-LR optimizers, and build + seed the WarmupController.

    Registers two non-persistent 0-d LR buffers on ``model`` under
    LR_DENSE_BUFFER / LR_SPARSE_BUFFER (the names `recir_train_step` rebinds each
    optimizer group LR to). The sparse buffer is created on CPU for TBE, whose
    fused optimizer requires a host learning-rate tensor. The WarmupController
    is seeded for iter 0 so the first compiled step uses the warmup LR.
    """
    sparse_params, dense_params = _partition_params(model, embedding_plan.param_fqns)

    dense_cfg = optimizer_config.dense.default
    sparse_cfg = optimizer_config.sparse.default

    dense_opt, dense_lr_buffer = _build_lr_buffered_optimizer(
        model, dense_params, dense_cfg, LR_DENSE_BUFFER, device
    )
    sparse_lr_device = torch.device("cpu") if use_tbe_sparse else device
    sparse_opt, sparse_lr_buffer = _build_lr_buffered_optimizer(
        model, sparse_params, sparse_cfg, LR_SPARSE_BUFFER, sparse_lr_device
    )

    warmup = WarmupController(
        iter_buffer=_register_iter_buffer(model, device),
        dense_base_lr=dense_cfg.lr,
        sparse_base_lr=sparse_cfg.lr,
        dense_lr_warmup_schedule=_maybe_schedule(dense_cfg.lr_warmup_schedule),
        sparse_lr_warmup_schedule=_maybe_schedule(sparse_cfg.lr_warmup_schedule),
        dense_lr_buffer=dense_lr_buffer,
        sparse_lr_buffer=sparse_lr_buffer,
    )
    warmup.apply()  # seed iter-0 LR
    setattr(model, WARMUP_ATTR, warmup)  # recir_train_step advances it in-graph

    logger.info(
        "[recir setup] dense_params=%d (lr=%g, %s, stages=%d), sparse_params=%d "
        "(lr=%g, %s, stages=%d); step0 LR dense=%g sparse=%g",
        len(dense_params),
        dense_cfg.lr,
        dense_cfg.optim_type.name,
        warmup.num_dense_stages,
        len(sparse_params),
        sparse_cfg.lr,
        sparse_cfg.optim_type.name,
        warmup.num_sparse_stages,
        float(dense_lr_buffer.item()),
        float(sparse_lr_buffer.item()),
    )

    return RecirOptimizerBundle(
        dense_opt=dense_opt,
        sparse_opt=sparse_opt,
        warmup=warmup,
        plan=embedding_plan,
        sparse_params=sparse_params,
        dense_params=dense_params,
    )
