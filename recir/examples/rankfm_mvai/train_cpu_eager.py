#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Pure CPU eager training entrypoint -- no compile, no XLA, no jax.

Sanity-check sibling to ``train.py``: builds the SAME model + data pipe via
the shared helpers (``_build_eager_core``, ``_build_data_pipe``, etc.)
and runs forward/backward/SGD natively on the host CPU. Only the per-step
body is different from the compiled loop -- everything around it
(preprocessor, data batch, model, loss) is the same code.

Use this to validate the eager ground-truth that
``parity_forward_loss.py`` compares the compiled path against: if loss
and NE behave sensibly for many steps here, then any future parity
divergence is a lowering bug, not a baseline bug.

Run:
    buck2 run //recir/examples/rankfm_mvai:train_cpu_eager -- --max_steps 50
"""

from __future__ import annotations

import logging
import math
import os
import tempfile
import time
from typing import Any, Dict, Iterator, List, NamedTuple, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from hammer.v3.models.udd_fm_v3.batch_loader import BlueReelsVDDFeatureConfig
from hammer.v3.models.udd_fm_v3.udd_fm_config import BlueReelsVDDModuleConfig
from recir.tests.xla.xsurface_test_utils import (
    _create_random_train_batch,
    _create_uniform_train_batch,
    build_model_from_config,
)

from .config import build_module_config, parse_cli, RankFMTrainerConfig
from .data_pipe import BatchReplayPipe, compute_kjt_key_to_size, replay_batch_iter
from .embedding import build_embedding_backend
from .embedding.cpu_backend import _build_single_node_optimizer
from .xsurface_helpers import (
    _build_data_pipe,
    _build_replay_iter,
    _extract_train_step_args,
    _log_batch_info,
    _resolve_nro_per_sample,
    _setup_logging,
)

logger: logging.Logger = logging.getLogger(__name__)


class _EagerCore(NamedTuple):
    """Model + first batch + KJT metadata + table_configs.

    Output of ``_build_eager_core``. Eager-only because the compiled path
    (in ``train.py``) interleaves padding + override-build between these
    steps, so it can't reuse a single helper without bending out of shape.
    """

    model: nn.Module
    batch: Any
    key_to_size: Dict[str, int]
    ro_kjt_keys: List[str]
    nro_kjt_keys: List[str]
    unary_kjt_keys: List[str]
    num_candidates_per_sample: int
    table_configs: List[Any]


def _build_eager_core(
    trainer_cfg: RankFMTrainerConfig,
    module_config: BlueReelsVDDModuleConfig,
    feature_config: BlueReelsVDDFeatureConfig,
    data_pipe: Optional[BatchReplayPipe],
) -> _EagerCore:
    """Build model + first batch + KJT metadata + table_configs.

    Mirrors the linear setup portion of ``train._build_compiled_train_step``
    (up to but not including ``backend.setup``). Caller must call
    ``backend.setup`` and ``model.set_recir_kjt_keys`` themselves -- single-
    process eager uses these too, but the order is fixed by the compile
    path's need to build overrides on the un-DMP'd model.
    """
    device = torch.device("cpu")
    model = build_model_from_config(module_config, device)
    for p in model.parameters():
        if p.is_floating_point():
            p.requires_grad_(True)

    num_candidates_per_sample: int = _resolve_nro_per_sample(
        module_config, trainer_cfg.enable_padding
    )
    key_to_size: Dict[str, int] = compute_kjt_key_to_size(model)

    torch.manual_seed(0)
    if data_pipe is not None:
        sample_iter = replay_batch_iter(
            data_pipe,
            key_to_size,
            max_steps=1,
            local_batch_size=trainer_cfg.local_batch_size,
            num_candidates=num_candidates_per_sample,
        )
        batch = next(sample_iter)
        logger.info("[EAGER-CPU] first batch from replay pipe")
    else:
        batch = _create_uniform_train_batch(
            feature_config,
            device,
            batch_size=trainer_cfg.local_batch_size,
        )

    ro_kjt_keys: List[str] = list(batch.ro_id_list_features.keys())
    nro_kjt_keys: List[str] = list(batch.nro_id_list_features.keys())
    _unary = batch.unary_features
    assert _unary is not None
    unary_kjt_keys: List[str] = list(_unary.keys())

    # pyre-ignore[16]: dynamic submodule traversal through hammer v3 nn.Module
    #   union; the runtime path exists on BlueReelsVDDTrainV3.
    ec = model.recblock.tokenizer_stack.mot_sparse_arch._sparse_arch._embedding_collection
    table_configs = ec.embedding_configs()

    return _EagerCore(
        model=model,
        batch=batch,
        key_to_size=key_to_size,
        ro_kjt_keys=ro_kjt_keys,
        nro_kjt_keys=nro_kjt_keys,
        unary_kjt_keys=unary_kjt_keys,
        num_candidates_per_sample=num_candidates_per_sample,
        table_configs=table_configs,
    )


def _num_classification_tasks(model: nn.Module) -> int:
    """Count BCE-loss heads so the NE report can slice ``all_losses``.

    Returns 0 if the expected attributes are missing -- NE is then skipped
    silently rather than failing the sanity loop.
    """
    if not hasattr(model, "num_main_tasks") or not hasattr(model, "num_main_reg_tasks"):
        return 0
    # pyre-ignore[16]: attributes exist on BlueReelsVDDTrainV3
    n = int(model.num_main_tasks) - int(model.num_main_reg_tasks)
    if getattr(model, "has_unpersonalized", False):
        n -= int(getattr(model, "num_unpersonalized_tasks", 0))
    return max(0, n)


def _compute_classification_ne(
    all_losses: torch.Tensor,
    named_tensors: Dict[str, torch.Tensor],
    num_cls: int,
) -> Optional[torch.Tensor]:
    """Per-task NE = weighted-mean BCE / H(empirical positive rate).

    Padded entries (weight=0) are excluded from the positive-rate estimate.
    The rate is clamped to (1e-6, 1-1e-6) so H(p) never hits log(0); a
    degenerate all-zero or all-one batch saturates NE at a finite ~14
    (still flags an anomaly, won't NaN).
    """
    if num_cls <= 0:
        return None
    label = named_tensors.get("label")
    weight = named_tensors.get("weight")
    if label is None or weight is None:
        return None
    label_c = label[:num_cls].to(torch.float32)
    weight_c = weight[:num_cls].to(torch.float32)
    p = (label_c * weight_c).sum(dim=1) / weight_c.sum(dim=1).clamp(min=1.0)
    p = p.clamp(min=1e-6, max=1.0 - 1e-6)
    h = -(p * p.log() + (1.0 - p) * (1.0 - p).log())
    return all_losses[:num_cls].detach().to(torch.float32) / h


def _resolve_eager_batch(
    step: int,
    init_batch: Any,
    replay_iter: Optional[Iterator[Any]],
    feature_config: Optional[BlueReelsVDDFeatureConfig],
    local_batch_size: int,
) -> Optional[Any]:
    """Pick the batch for this step. Returns None when replay is exhausted.

    Priority: real replay > fresh synthetic random > reuse the init batch.
    Fresh-random uses ``torch.manual_seed(step)`` so the sanity run is bit-
    exact reproducible across restarts.
    """
    if replay_iter is not None:
        try:
            return next(replay_iter)
        except StopIteration:
            return None
    if feature_config is not None and step > 1:
        torch.manual_seed(step)
        return _create_random_train_batch(
            feature_config, torch.device("cpu"), batch_size=local_batch_size
        )
    return init_batch


class _TbWriter:
    """Minimal SummaryWriter wrapper for the CPU eager loop.

    Local paths are written directly. gs:// paths buffer events to a local
    tempdir (gcsfs would truncate on every append) and upload the full dir to
    GCS on ``close``. Any init/log/upload failure degrades to a no-op so TB
    issues never break training.
    """

    def __init__(self, log_dir: str) -> None:
        self._writer: Any = None
        self._local_dir: str = log_dir
        self._gcs_dir: Optional[str] = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            if log_dir.startswith("gs://"):
                self._gcs_dir = log_dir.rstrip("/")
                self._local_dir = tempfile.mkdtemp(prefix="rankfm-eager-tb-")
            else:
                os.makedirs(self._local_dir, exist_ok=True)
            self._writer = SummaryWriter(log_dir=self._local_dir)
            target = self._gcs_dir or self._local_dir
            logger.info(f"[TB] writing scalars to {target} (local={self._local_dir})")
        except Exception:
            self._writer = None
            logger.exception("[TB] init failed - scalar logging disabled")

    def log_scalar(self, name: str, value: float, step: int) -> None:
        if self._writer is None or not math.isfinite(value):
            return
        try:
            self._writer.add_scalar(name, value, global_step=step)
        except Exception:
            self._writer = None
            logger.exception("[TB] add_scalar failed - disabling further TB logging")

    def close(self) -> None:
        if self._writer is None:
            return
        try:
            self._writer.flush()
            self._writer.close()
        except Exception:
            logger.exception("[TB] close failed - continuing")
        self._writer = None
        if self._gcs_dir is None:
            return
        try:
            from google.cloud import storage

            stripped = self._gcs_dir[len("gs://") :]
            bucket_name, _, prefix = stripped.partition("/")
            bucket = storage.Client().bucket(bucket_name)
            for root, _, files in os.walk(self._local_dir):
                for fname in files:
                    local_path = os.path.join(root, fname)
                    rel = os.path.relpath(local_path, self._local_dir)
                    blob_name = f"{prefix}/{rel}" if prefix else rel
                    bucket.blob(blob_name).upload_from_filename(local_path)
            logger.info(f"[TB] uploaded events to {self._gcs_dir}")
        except Exception:
            logger.exception(
                "[TB] GCS upload failed - events remain at %s", self._local_dir
            )


def _log_tb_step(
    tb: Optional[_TbWriter],
    step: int,
    log_every: int,
    step_time: float,
    loss_total: float,
    per_task: List[float],
    ne: Optional[List[float]],
) -> None:
    if tb is None or log_every <= 0 or step % log_every != 0:
        return
    tb.log_scalar("time/step_seconds", step_time, step)
    tb.log_scalar("loss/total", loss_total, step)
    for i, v in enumerate(per_task):
        tb.log_scalar(f"loss/task_{i}", v, step)
    if ne is not None:
        for i, v in enumerate(ne):
            tb.log_scalar(f"ne/task_{i}", v, step)


def _run_eager_training_loop(
    trainer_cfg: RankFMTrainerConfig,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    init_batch: Any,
    feature_config: Optional[BlueReelsVDDFeatureConfig],
    data_pipe: Optional[BatchReplayPipe] = None,
    key_to_size: Optional[Dict[str, int]] = None,
    tb: Optional[_TbWriter] = None,
) -> None:
    """Pure CPU eager forward/backward/SGD loop with loss + NE + NaN guard.

    Only piece that differs from ``train._run_training_loop``: the per-step
    body. Compile path = ``train_step(model, opt, *args); sync_device_state``;
    eager path here = ``opt.zero_grad(); model(*args); check finite;
    loss.backward(); opt.step()`` + per-step loss/NE log. Everything around
    that (replay iter, batch resolution, batch-info logging, kjt args
    extraction) comes from ``train`` unchanged.

    The MoT bundle dtype validator is bypassed for the duration (mirrors
    ``parity_forward_loss._eager_step``): the compiled path skips this check
    via ``is_fx_tracing()``, so the eager ground-truth must opt out too for
    a like-for-like signal -- otherwise the eager forward rejects fp32 in a
    bf16-declared slot and the compiled path silently doesn't.

    Per-step ``.item()`` / scalar reads are intentional: CPU host data, no
    device-sync cost, and NaN detection has to happen before backward.
    """
    # Lazy: only this loop needs the dtype-validator handle.
    import hammer.v3.modules.core.features.bundle as feature_bundle

    model.train(True)
    mode = (
        "replay"
        if data_pipe is not None
        else ("fresh-random" if feature_config is not None else "fixed")
    )
    logger.info(
        f"[EAGER-CPU] starting ({mode}): {trainer_cfg.max_steps} steps, "
        f"local_bs={trainer_cfg.local_batch_size}"
    )

    replay_iter = _build_replay_iter(
        trainer_cfg,
        _resolve_nro_per_sample(
            model.module_config,  # pyre-ignore[6]
            trainer_cfg.enable_padding,
        ),
        data_pipe,
        key_to_size,
    )

    num_cls = _num_classification_tasks(model)
    logger.info(
        f"[EAGER-CPU] reporting NE for {num_cls} classification tasks "
        f"(0 = unavailable, skipped)"
    )

    orig_validate = feature_bundle._validate_dtype
    feature_bundle._validate_dtype = lambda *a, **k: None  # pyre-ignore[8]
    start_time = time.time()
    batch = init_batch
    try:
        for step in range(1, trainer_cfg.max_steps + 1):
            next_batch = _resolve_eager_batch(
                step,
                init_batch,
                replay_iter,
                feature_config,
                trainer_cfg.local_batch_size,
            )
            if next_batch is None:
                logger.info(f"[EAGER-CPU] replay exhausted at step {step}; stopping.")
                break
            batch = next_batch
            _log_batch_info(step, batch)

            step_start = time.time()
            opt.zero_grad()
            all_losses, named_tensors = model(*_extract_train_step_args(batch))
            loss = all_losses.sum()

            loss_val = float(loss.detach())
            if not math.isfinite(loss_val):
                per_task = [float(x) for x in all_losses.detach()]
                raise RuntimeError(
                    f"[EAGER-CPU] non-finite loss at step {step}: "
                    f"total={loss_val}, per_task={per_task}"
                )

            loss.backward()
            # Cap the global grad norm before the optimizer step. Without this
            # the cold-start update for the regression head (label magnitude
            # ~30k -> initial gradient ~10^4-10^6) blows the model to NaN in
            # a couple of steps, even with Adagrad's adaptive LR -- on step 1
            # the per-param accumulator is 0 so the adaptive scaling adds no
            # protection. The compiled path survives the same cold-start via
            # bf16 saturation; eager fp32 needs an explicit bound.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            step_time = time.time() - step_start

            per_task = [float(x) for x in all_losses.detach()]
            log_parts: List[str] = [
                f"step={step}/{trainer_cfg.max_steps}",
                f"time={step_time:.2f}s",
                f"loss={loss_val:.4f}",
                f"per_task=[{', '.join(f'{v:.3f}' for v in per_task)}]",
            ]
            ne = _compute_classification_ne(all_losses, named_tensors, num_cls)
            ne_list: Optional[List[float]] = None
            if ne is not None:
                ne_list = [float(x) for x in ne]
                if not all(math.isfinite(v) for v in ne_list):
                    raise RuntimeError(
                        f"[EAGER-CPU] non-finite NE at step {step}: {ne_list}"
                    )
                log_parts.append(f"NE=[{', '.join(f'{v:.3f}' for v in ne_list)}]")
            logger.info("[EAGER-CPU] " + " | ".join(log_parts))
            _log_tb_step(
                tb,
                step,
                trainer_cfg.tb_log_every,
                step_time,
                loss_val,
                per_task,
                ne_list,
            )
    finally:
        feature_bundle._validate_dtype = orig_validate

    total = time.time() - start_time
    logger.info(
        f"[EAGER-CPU] complete: {trainer_cfg.max_steps} steps in "
        f"{total:.1f}s ({total / max(trainer_cfg.max_steps, 1):.2f}s/step)"
    )


def main() -> None:
    """CPU eager entrypoint.

    Forces ``embedding_backend='cpu'`` (DMP/CPU is the only eager-runnable
    backend) and ``enable_padding=False`` (PaddingRegionTransform is a
    compile-only concept). Works for both single-process (gloo_world=1,
    all params in one Adagrad optimizer) and multi-rank gloo launches
    (DMP ROW_WISE/TABLE_WISE sharding + FUSED in-backward embedding
    optimizer + dense-only Adagrad via gloo).
    """
    _setup_logging()
    trainer_cfg, model_args = parse_cli()

    if trainer_cfg.embedding_backend != "cpu":
        logger.warning(
            f"[EAGER-CPU] overriding embedding_backend "
            f"'{trainer_cfg.embedding_backend}' -> 'cpu' "
            f"(only cpu backend runs eagerly)"
        )
        trainer_cfg.embedding_backend = "cpu"
    if trainer_cfg.enable_padding:
        logger.info(
            "[EAGER-CPU] overriding enable_padding=True -> False "
            "(PaddingRegionTransform is compile-only)"
        )
        trainer_cfg.enable_padding = False

    logger.info(f"[EAGER-CPU] trainer config: {trainer_cfg}")
    logger.info(f"[EAGER-CPU] model args: {model_args}")

    backend = build_embedding_backend(trainer_cfg.embedding_backend)
    gloo_rank, gloo_world = backend.init_pg()
    logger.info(f"[EAGER-CPU] gloo rank={gloo_rank} world={gloo_world}")

    module_config, feature_config = build_module_config(
        model_args, enable_padding=trainer_cfg.enable_padding
    )
    data_pipe = _build_data_pipe(trainer_cfg, gloo_rank, gloo_world)

    core = _build_eager_core(trainer_cfg, module_config, feature_config, data_pipe)
    model = backend.setup(
        core.model,
        core.table_configs,
        trainer_cfg.embedding_lr,
        gloo_world,
    )
    if dist.is_initialized():
        dense_params = list(backend.dense_named_parameters(model).values())
        opt: torch.optim.Optimizer = torch.optim.Adagrad(
            dense_params,
            lr=trainer_cfg.dense_optimizer_config.lr,
            initial_accumulator_value=0.1,
            foreach=True,
        )
    else:
        opt = _build_single_node_optimizer(
            model,
            trainer_cfg.embedding_lr,
            trainer_cfg.dense_optimizer_config.lr,
        )
    # pyre-ignore[29]: nn.Module attribute access returns Union[Module, Tensor]
    #   per pyre's stubs; set_recir_kjt_keys is a runtime method on the model.
    model.set_recir_kjt_keys(core.ro_kjt_keys, core.nro_kjt_keys, core.unary_kjt_keys)

    tb: Optional[_TbWriter] = None
    if trainer_cfg.tb_dir and gloo_rank == 0:
        tb = _TbWriter(trainer_cfg.tb_dir)
    try:
        _run_eager_training_loop(
            trainer_cfg,
            model,
            opt,
            core.batch,
            feature_config=feature_config,
            data_pipe=data_pipe,
            key_to_size=core.key_to_size,
            tb=tb,
        )
    finally:
        if tb is not None:
            tb.close()

    if dist.is_initialized():
        dist.destroy_process_group()
    logger.info("[EAGER-CPU] cleanup complete; exiting.")
    os._exit(0)


if __name__ == "__main__":
    main()
