#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
RankFM (xsurface V3) E2E training on GKE TPU v7x.

Dense HSTU path always runs on TPU via recir @compile + XLA GSPMD (JAX PJRT),
using PreprocessorDispatcher + SparseArchDispatcher from xsurface_test_utils
(these wrap the heavy preprocessor and ECSparseArch with dummy modules during
tracing, and dispatch to the real modules at runtime).

Supports two embedding backends (--embedding_backend):
  cpu   torchrec DMP ROW_WISE/TABLE_WISE on CPU + gloo PG
  tpu   JAX/SparseCore: tpu_sparse_dense_matmul + on-chip Adagrad

Usage:
    python -u -m recir.examples.rankfm.train --max_steps 10
    python -u -m recir.examples.rankfm.train --max_steps 5 \\
        --embedding_backend tpu
"""

from __future__ import annotations

import copy
import logging
import math
import os
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

import fbgemm_gpu  # noqa: F401 -- force C++ op registration before fallback
import jax
import recir
import recir.lowering.fbgemm_fallback  # noqa: F401 -- registers SPMD_DECOMP_TABLE entries
import torch
import torch.distributed as dist
import torch.nn as nn
from hammer.v3.models.udd_fm_v3.batch_loader import BlueReelsVDDFeatureConfig
from hammer.v3.models.udd_fm_v3.udd_fm_config import BlueReelsVDDModuleConfig
from hammer.v3.modules.recblock import _inputs_to_feature_bundle
from recir.api import sync_device_state_to_torch
from recir.backends import XLAGSPMDBackend
from recir.decompositions import SPMD_DECOMP_TABLE
from recir.distributed.parallel_mode import GSPMDParallel
from recir.lowering.fbgemm_fallback import _batched_complete_cumsum_impl
from recir.region_transforms.padding import (
    compute_padded_shape,
    JaggedPairConfig,
    PaddingRegionTransform,
)
from recir.tests.xla.xsurface_test_utils import (
    _create_random_train_batch,
    _create_uniform_train_batch,
    build_model_from_config,
    resize_batch_inplace,
)

from .config import build_module_config, parse_cli, RankFMTrainerConfig
from .data_pipe import BatchReplayPipe, compute_kjt_key_to_size, replay_batch_iter
from .embedding import build_embedding_backend

# Bootstrap gr::batched_complete_cumsum if the C++ library failed to load
# (e.g. OSS Docker container where hammer/oss/generative_recommenders/ops/cpp
# is not compiled into a .so). The recir SPMD decomp table replaces this op
# with _batched_complete_cumsum_impl at XLA lowering time, but the registration
# below requires torch.ops.gr.batched_complete_cumsum.default to exist as a
# valid OpOverload first.
try:
    _cumsum_op: object = torch.ops.gr.batched_complete_cumsum.default  # noqa: B018
except AttributeError:
    torch.library.define("gr::batched_complete_cumsum", "(Tensor values) -> Tensor")
    torch.library.impl(
        "gr::batched_complete_cumsum",
        "CompositeImplicitAutograd",
        _batched_complete_cumsum_impl,
    )

# Register gr.batched_complete_cumsum decomposition for XLA lowering.
# Same as test_xsurface.py - required for the gr custom op used by the
# preprocessor to lower through SPMD.
SPMD_DECOMP_TABLE[torch.ops.gr.batched_complete_cumsum.default] = (
    _batched_complete_cumsum_impl
)

logger: logging.Logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    logging.basicConfig(
        format="%(levelname).1s%(asctime)s %(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
        datefmt="%m%d %H:%M:%S",
        force=True,
    )


def _build_compiled_train_step(
    trainer_cfg: RankFMTrainerConfig,
    module_config: BlueReelsVDDModuleConfig,
    feature_config: BlueReelsVDDFeatureConfig,
    backend: Any,
    num_ranks: int,
    num_processes: int,
    global_bs: int,
    data_pipe: Optional[BatchReplayPipe] = None,
) -> Tuple[Any, Any, Any, Any, Any, Dict[str, int], List[int]]:
    """Build model, batch, overrides, and the compiled train_step.

    Returns ``(train_step, model, opt, batch, feature_config, key_to_size,
    ro_per_key_budgets)``.  ``key_to_size`` is the model's KJT-feature ->
    num_embeddings map computed pre-backend.setup (post-DMP the EC is wrapped
    in a ``ShardedEmbeddingCollection`` that doesn't expose
    ``embedding_configs()``); use it to clamp loaded IDs into the model's
    hash space at every runtime step.  ``ro_per_key_budgets`` is the per-
    key RO eager-pad budget (returned so the caller's training-loop replay
    iter can reuse the same numbers we used at trace time).

    When ``data_pipe`` is set, the *first* batch read from it is used as
    the trace batch -- guarantees the trace's pytree schema (especially
    ``all_labels`` keys, KJT key order) matches every subsequent runtime
    batch from the same pipe.
    """
    device = torch.device("cpu")
    model = build_model_from_config(module_config, device)

    for p in model.parameters():
        if p.is_floating_point():
            p.requires_grad_(True)

    num_candidates_per_sample: int = int(module_config.hstu_max_output_len)

    # Effective sharding modulus = lcm(GSPMD device count, total SC count).
    # For JAX: lcm(jax.device_count()=256, sc_modulus(=16) × num_processes(=32)
    # = 512) = 512.  For CPU: lcm(num_ranks, 1 × num_processes) = num_ranks
    # (no change).  Padded shape thus divides both GSPMD shard width and SC
    # CSR width (per-host SC × num_hosts).
    padded_num_devices: int = math.lcm(num_ranks, backend.sc_modulus() * num_processes)

    # Per-rank batch (NOT global). Each pod feeds its own local mini-batch to
    # train_step at runtime; recir's _make_batch_transform expands the trace-
    # time inputs by num_processes (=gloo_world) to produce a global view for
    # GSPMD.  Use uniform (non-ROO) batches so B_NRO = local_bs * num_candidates
    # is deterministic across ranks - required by the GSPMD shard-divisibility
    # check on jagged tensor totals.
    torch.manual_seed(0)
    batch = _create_uniform_train_batch(
        feature_config,
        device,
        batch_size=trainer_cfg.local_batch_size,
    )

    if trainer_cfg.enable_padding:
        # Override-build sample uses GLOBAL sizes (matches PRT's pad_to_static
        # fake kernel which also returns global ``target_size``). Keeps FX
        # trace shapes consistent across override outputs and PRT pad outputs
        # (both at global). At runtime, fwd_fn returns PER-HOST data so
        # _to_sharded_jax can expand per-host to global, matching PRT runtime.
        padded_ro_bs = compute_padded_shape(global_bs, padded_num_devices)
        padded_nro_bs = compute_padded_shape(
            global_bs * num_candidates_per_sample, padded_num_devices
        )
        expanded_batch = copy.deepcopy(batch)
        resize_batch_inplace(
            expanded_batch,
            target_b_ro=padded_ro_bs,
            target_b_nro=padded_nro_bs,
        )
        override_bs = padded_ro_bs
    else:
        expanded_batch = copy.deepcopy(batch)
        resize_batch_inplace(
            expanded_batch,
            target_b_ro=global_bs,
            target_b_nro=global_bs * num_candidates_per_sample,
        )
        override_bs = global_bs

    all_labels = expanded_batch.all_labels
    assert all_labels is not None
    batch_sample = _inputs_to_feature_bundle(
        expanded_batch.ro_id_list_features,
        None,
        expanded_batch.nro_id_list_features,
        expanded_batch.num_candidates,
        all_labels,
    )

    # Compute KJT key -> num_embeddings map BEFORE backend.setup, while the EC
    # still exposes embedding_configs() (the sharded post-DMP wrapper does not).
    key_to_size: Dict[str, int] = compute_kjt_key_to_size(model)

    # Capture KJT keys against the uniform batch BEFORE the trace batch swap
    # below; needed by set_recir_kjt_keys and (for JAX) backend.seq_lens.
    ro_kjt_keys: List[str] = list(batch.ro_id_list_features.keys())
    nro_kjt_keys: List[str] = list(batch.nro_id_list_features.keys())
    _unary = batch.unary_features
    assert _unary is not None
    unary_kjt_keys: List[str] = list(_unary.keys())

    # Per-key RO budgets for eager pad in data_pipe (matches PRT's per-key
    # padding for ro_id_list_features). nro/unary use a single global formula
    # (B * num_candidates) inside data_pipe; ro is per-key so we compute it
    # here from the same module_config that PRT reads.
    contextual_budgets_for_pad = dict(module_config.contextual_feature_to_max_length)
    ro_per_key_budgets: List[int] = [
        trainer_cfg.local_batch_size
        * contextual_budgets_for_pad.get(k, int(module_config.hstu_max_seq_len))
        for k in ro_kjt_keys
    ]

    # Pick the trace batch: prefer the first real-data batch so trace and
    # runtime see the same pytree schema; fall back to a synthetic random
    # batch only when no data_pipe is configured.
    if trainer_cfg.enable_padding and feature_config is not None:
        if data_pipe is not None:
            trace_iter = replay_batch_iter(
                data_pipe,
                key_to_size,
                max_steps=1,
                local_batch_size=trainer_cfg.local_batch_size,
                num_candidates=num_candidates_per_sample,
                ro_target_keys=ro_kjt_keys,
                nro_target_keys=nro_kjt_keys,
                unary_target_keys=unary_kjt_keys,
                ro_per_key_budgets=ro_per_key_budgets,
            )
            batch = next(trace_iter)
            logger.info(
                "[TRACE] using first replay batch (real data) for trace; "
                "training loop will re-iterate data_pipe from the start"
            )
        else:
            torch.manual_seed(0)
            batch = _create_random_train_batch(
                feature_config,
                device,
                batch_size=trainer_cfg.local_batch_size,
            )
            logger.info("[TRACE] using synthetic random batch for trace")

    # Extract table configs from the EC (before backend.setup() mutates anything).
    # pyre-ignore[16]: dynamic submodule traversal through hammer v3 nn.Module
    #   union; the runtime path exists on BlueReelsVDDTrainV3.
    ec = model.recblock.tokenizer_stack.mot_sparse_arch._sparse_arch._embedding_collection
    table_configs = ec.embedding_configs()

    # Build overrides BEFORE backend.setup(). The override builder runs the
    # sparse arch eagerly and captures its output (and, for JAX, the input
    # KJT) via forward hooks; doing this against the original unmodified
    # model is cleanest.
    overrides = backend.get_override(
        model.recblock, batch_sample, device, override_bs, num_ranks, num_processes
    )

    # Set up embedding storage (DMP for CPU, SparseCore for TPU).
    model, dense_opt = backend.setup(
        model,
        table_configs,
        trainer_cfg.embedding_lr,
        trainer_cfg.dense_lr,
        num_processes,
    )

    opt = dense_opt

    model.set_recir_kjt_keys(  # pyre-ignore[29]
        ro_kjt_keys, nro_kjt_keys, unary_kjt_keys
    )

    region_transforms: List[Any] = []
    if trainer_cfg.enable_padding:
        region_transforms = [
            PaddingRegionTransform(
                jagged_pairs={
                    # RO/NRO id-list pairs are opted out below — they only
                    # flow into PreprocessorDispatcher (a CPU graph break)
                    # and stay at local stride throughout the preprocessor.
                    # The dispatcher handles the padding.
                    "unary_values": JaggedPairConfig(
                        "unary_lengths",
                        per_sample_budgets=1,
                        batch_size_arg="nro_float_features",
                        count_arg="num_candidates",
                        max_per_sample_count=num_candidates_per_sample,
                    ),
                },
                batch_size_arg="ro_float_features",
                # Align to lcm(GSPMD devices, total SC count) so PRT's runtime
                # delivery matches the override-build sample's padded shape
                # (which uses the same modulus above).
                num_devices=padded_num_devices,
                num_processes=num_processes,
                skip_args={
                    "ro_id_list_values",
                    "ro_id_list_lengths",
                    "nro_id_list_values",
                    "nro_id_list_lengths",
                },
            ),
        ]

    parallel_mode = GSPMDParallel(
        "replicate",
        loss_reduction="sum",
        num_ranks=num_ranks,
        num_processes=num_processes,
    )

    # pyre-ignore
    def train_step_fn(
        mod: nn.Module,
        opt: torch.optim.Optimizer,
        ro_float_features: torch.Tensor,
        nro_float_features: torch.Tensor,
        ro_id_list_values: torch.Tensor,
        ro_id_list_lengths: torch.Tensor,
        nro_id_list_values: torch.Tensor,
        nro_id_list_lengths: torch.Tensor,
        unary_values: torch.Tensor,
        unary_lengths: torch.Tensor,
        num_candidates: torch.Tensor,
        supervision_label_t: torch.Tensor,
        supervision_weight_t: torch.Tensor,
        all_labels: Dict[str, torch.Tensor],
    ) -> None:
        all_losses, _ = mod(
            ro_float_features,
            nro_float_features,
            ro_id_list_values,
            ro_id_list_lengths,
            nro_id_list_values,
            nro_id_list_lengths,
            unary_values,
            unary_lengths,
            num_candidates,
            supervision_label_t,
            supervision_weight_t,
            all_labels,
        )
        loss: torch.Tensor = all_losses.sum()
        loss.backward()
        opt.step()

    train_step = _compile_with_xsurface_dispatchers(
        model,
        opt,
        batch,
        overrides,
        train_step_fn,
        parallel_mode=parallel_mode,
        backend=XLAGSPMDBackend(),
        region_transforms=region_transforms,
    )

    return (
        train_step,
        model,
        opt,
        batch,
        feature_config,
        key_to_size,
        ro_per_key_budgets,
    )


def _compile_with_xsurface_dispatchers(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    batch: Any,
    dispatchers: List[Any],
    callable_: Any,
    *,
    parallel_mode: GSPMDParallel,
    backend: XLAGSPMDBackend,
    region_transforms: List[Any],
) -> Any:
    """Drive ``recir.trace`` / ``recir.compile_from_gm`` with the rankfm
    PreprocessorDispatcher + SparseArchDispatcher pair.
    """
    from recir.api import make_compiled_runner
    from torchrec.distributed.model_parallel import DistributedModelParallel

    # Stage 0
    model_to_train = (
        model.module if isinstance(model, DistributedModelParallel) else model
    )
    for dispatcher in dispatchers:
        dispatcher.install_eager(model_to_train)

    # Stage 2
    gm, flat_state = recir.trace(
        callable_,
        model_to_train,
        opt,
        *_extract_train_step_args(batch),
        parallel_mode=parallel_mode,
        region_transforms=region_transforms,
    )

    # Stage 3
    for dispatcher in dispatchers:
        gm, flat_state = dispatcher.install_graph(gm, flat_state)

    # Stages 4-10
    compiled_obj: Any = recir.compile_from_gm(
        gm,
        flat_state,
        parallel_mode=parallel_mode,
        backend=backend,
        region_transforms=region_transforms,
    )

    return make_compiled_runner(compiled_obj)


def _is_jax_profiling_enabled(trainer_cfg: RankFMTrainerConfig) -> bool:
    """Check if JAX profiling should be enabled (TPU present + start step > 0)."""
    try:
        is_tpu = len(jax.devices()) > 0 and jax.devices()[0].platform == "tpu"
        return is_tpu and trainer_cfg.profile_start_step > 0
    except (RuntimeError, IndexError):
        logger.warning("Could not detect JAX TPU platform, profiling disabled")
        return False


def _log_hbm_memory() -> None:
    """Log per-device HBM memory usage."""
    for i, dev in enumerate(jax.local_devices()):
        stats = dev.memory_stats()
        if stats:
            used = stats.get("bytes_in_use", 0)
            limit = stats.get("bytes_limit", 0)
            logger.info(
                f"  Device {i} HBM: {used / 1e9:.2f}/{limit / 1e9:.2f} GB "
                f"({100 * used / max(limit, 1):.1f}%)"
            )


def _extract_train_step_args(
    batch: Any,
) -> Tuple[Any, ...]:
    """Extract raw tensor args from a batch for the train step."""
    ro_kjt = batch.ro_id_list_features
    nro_kjt = batch.nro_id_list_features
    unary_kjt = batch.unary_features
    assert unary_kjt is not None
    return (
        batch.ro_float_features,
        batch.nro_float_features,
        ro_kjt.values(),
        ro_kjt.lengths(),
        nro_kjt.values(),
        nro_kjt.lengths(),
        unary_kjt.values(),
        unary_kjt.lengths(),
        batch.num_candidates,
        batch.supervision_label.t().contiguous(),
        batch.supervision_weight.t().contiguous(),
        batch.all_labels,
    )


def _log_per_key_stats(name: str, kjt: Any, B: int) -> str:
    """Format per-key length stats for a KJT."""
    keys = kjt.keys()
    lengths = kjt.lengths()
    num_keys = len(keys)
    per_key = lengths.reshape(num_keys, B)
    parts: List[str] = []
    for i, key in enumerate(keys):
        kl = per_key[i]
        total = int(kl.sum())
        mn = int(kl.min())
        mx = int(kl.max())
        if mn == mx:
            parts.append(f"{key}:{mn}x{B}={total}")
        else:
            parts.append(f"{key}:[{mn}-{mx}]sum={total}")
    return f"{name}({num_keys} keys): " + ", ".join(parts)


def _log_batch_info(step: int, batch: Any) -> None:
    """Log per-step batch shapes and per-key length stats."""
    nc = batch.num_candidates
    B = batch.ro_float_features.shape[0]
    logger.info(
        f"[BATCH step={step}] B={B} "
        f"ro_float={list(batch.ro_float_features.shape)} "
        f"nro_float={list(batch.nro_float_features.shape)} "
        f"num_candidates sum={int(nc.sum())} min={int(nc.min())} max={int(nc.max())}"
    )
    logger.info(
        f"[BATCH step={step}] %s",
        _log_per_key_stats("RO", batch.ro_id_list_features, B),
    )
    B_NRO = int(nc.sum())
    logger.info(
        f"[BATCH step={step}] %s",
        _log_per_key_stats("NRO", batch.nro_id_list_features, B_NRO),
    )


def _next_training_batch(
    step: int,
    trainer_cfg: RankFMTrainerConfig,
    batch: Any,
    replay_iter: Optional[Iterator[Any]],
    feature_config: Any,
) -> Optional[Any]:
    """Resolve the batch for ``step``. Returns None if the replay pipe is exhausted."""
    if replay_iter is not None:
        try:
            return next(replay_iter)
        except StopIteration:
            return None
    if trainer_cfg.enable_padding and feature_config is not None and step > 1:
        torch.manual_seed(step)
        return _create_random_train_batch(
            feature_config,
            torch.device("cpu"),
            batch_size=trainer_cfg.local_batch_size,
        )
    return batch


def _build_replay_iter(
    trainer_cfg: RankFMTrainerConfig,
    num_candidates: int,
    data_pipe: Optional[BatchReplayPipe],
    key_to_size: Optional[Dict[str, int]],
    ro_target_keys: Optional[List[str]] = None,
    nro_target_keys: Optional[List[str]] = None,
    unary_target_keys: Optional[List[str]] = None,
    ro_per_key_budgets: Optional[List[int]] = None,
) -> Optional[Iterator[Any]]:
    if data_pipe is None:
        return None
    if key_to_size is None:
        logger.warning(
            "Replay mode but key_to_size is None; loaded IDs will not be "
            "clamped against model table sizes (may IndexError)."
        )
    return replay_batch_iter(
        data_pipe,
        key_to_size or {},
        max_steps=trainer_cfg.max_steps,
        local_batch_size=trainer_cfg.local_batch_size,
        num_candidates=num_candidates,
        ro_target_keys=ro_target_keys,
        nro_target_keys=nro_target_keys,
        unary_target_keys=unary_target_keys,
        ro_per_key_budgets=ro_per_key_budgets,
    )


def _run_training_loop(
    trainer_cfg: RankFMTrainerConfig,
    train_step: Any,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    batch: Any,
    is_main: bool,
    global_bs: int,
    feature_config: Any = None,
    data_pipe: Optional[BatchReplayPipe] = None,
    key_to_size: Optional[Dict[str, int]] = None,
    ro_per_key_budgets: Optional[List[int]] = None,
) -> None:
    """Execute the compiled training loop with optional JAX profiling.

    Three batch-source modes:
      1. ``data_pipe`` is set => iterate over real batches loaded from GCS
         (or local mirror). KJT IDs are clamped via ``key_to_size``.
         PaddingRegionTransform pads each loaded batch to static budgets.
      2. ``trainer_cfg.enable_padding`` + ``feature_config`` => fresh random
         (jagged) batch per step; padded at trace time.
      3. otherwise => reuse the closure-captured uniform batch every step.
    """
    jax_profiling = _is_jax_profiling_enabled(trainer_cfg)
    profiling_active = False
    if jax_profiling:
        logger.info(
            f"JAX profiling enabled: steps {trainer_cfg.profile_start_step}-"
            f"{trainer_cfg.profile_end_step}, dir={trainer_cfg.profile_dir}"
        )

    model.train(True)
    mode = "replay" if data_pipe is not None else "synthetic"
    logger.info(
        f"Starting training ({mode}): {trainer_cfg.max_steps} steps, global_bs={global_bs}"
    )
    start_time = time.time()
    # The override was traced with the synthetic batch's key order. At
    # runtime, the override reconstructs the KJT with hardcoded keys (in
    # trace-time order) but applies them to whatever values our data_pipe
    # produces. If our KJT has keys in a different order than the trace-time
    # batch, values get silently misaligned per key.
    # Force our data_pipe output to match the synthetic batch's key order.
    ro_target_keys: List[str] = list(batch.ro_id_list_features.keys())
    nro_target_keys: List[str] = list(batch.nro_id_list_features.keys())
    unary_target_keys: Optional[List[str]] = (
        list(batch.unary_features.keys()) if batch.unary_features is not None else None
    )
    replay_iter = _build_replay_iter(
        trainer_cfg,
        int(model.module_config.hstu_max_output_len),  # pyre-ignore[16]
        data_pipe,
        key_to_size,
        ro_target_keys=ro_target_keys,
        nro_target_keys=nro_target_keys,
        unary_target_keys=unary_target_keys,
        ro_per_key_budgets=ro_per_key_budgets,
    )

    for step in range(1, trainer_cfg.max_steps + 1):
        if jax_profiling and step == trainer_cfg.profile_start_step:
            if not trainer_cfg.profile_dir.startswith("gs://"):
                os.makedirs(trainer_cfg.profile_dir, exist_ok=True)
            jax.profiler.start_trace(trainer_cfg.profile_dir)
            profiling_active = True
            logger.info(f"[PROFILE] Started trace at step {step}")

        next_batch = _next_training_batch(
            step, trainer_cfg, batch, replay_iter, feature_config
        )
        if next_batch is None:
            logger.info(f"Replay pipe exhausted at step {step}; ending training.")
            break
        batch = next_batch

        if is_main:
            _log_batch_info(step, batch)

        step_start = time.time()
        train_step(model, opt, *_extract_train_step_args(batch))
        sync_device_state_to_torch(train_step)
        step_time = time.time() - step_start

        if jax_profiling and step == trainer_cfg.profile_end_step and profiling_active:
            jax.profiler.stop_trace()
            profiling_active = False
            logger.info(
                f"[PROFILE] Stopped trace at step {step} — saved to "
                f"{trainer_cfg.profile_dir}"
            )

        if is_main:
            logger.info(f"Step {step}/{trainer_cfg.max_steps} | time={step_time:.2f}s")
        if jax_profiling and trainer_cfg.enable_memory_log and is_main:
            _log_hbm_memory()

    total_time = time.time() - start_time

    if profiling_active and jax_profiling:
        jax.profiler.stop_trace()
        logger.info(
            f"[PROFILE] Stopped trace (training ended) — saved to "
            f"{trainer_cfg.profile_dir}"
        )

    if is_main:
        logger.info(
            f"Training complete: {trainer_cfg.max_steps} steps in {total_time:.1f}s "
            f"({total_time / trainer_cfg.max_steps:.2f}s/step)"
        )


def main() -> None:
    # Must be called before any JAX operation. On GKE TPU the PJRT plugin reads
    # coordinator address and process index from TPU runtime metadata automatically.
    # Without this, each pod runs as jax.process_count()=1 and the SparseCore
    # hardware collective (which spans all 128 chips) deadlocks.
    jax.distributed.initialize()
    _setup_logging()
    logger.info(
        f"[DIST] jax.distributed.initialize() done: "
        f"process_count={jax.process_count()}, "
        f"device_count={jax.device_count()}, "
        f"local_device_count={jax.local_device_count()}, "
        f"process_index={jax.process_index()}"
    )
    trainer_cfg, model_args = parse_cli()
    logger.info(f"Trainer config: {trainer_cfg}")
    logger.info(f"Model args: {model_args}")

    # --- 1. Build embedding backend and init PG ---
    backend = build_embedding_backend(trainer_cfg.embedding_backend)
    gloo_rank, gloo_world = backend.init_pg()
    is_main = gloo_rank == 0

    # --- 2. Mesh shapes ---
    # global_bs = local_batch_size * gloo_world (NOT * num_ranks).
    # num_ranks (jax.device_count) is the GSPMD mesh size used for grad
    # averaging; gloo_world is the number of OS processes contributing
    # data.  global_bs must equal local_batch_size * gloo_world for the
    # override's cached_output (sized at global_bs) to match the
    # expanded trace input.
    num_tpu_devices = jax.device_count()
    num_ranks = num_tpu_devices if num_tpu_devices > 1 else gloo_world
    global_bs = trainer_cfg.local_batch_size * gloo_world

    # --- 3. Build module + feature config (single source of truth) ---
    module_config, feature_config = build_module_config(model_args)

    # --- 4. Build data_pipe (BEFORE compile, so the trace batch can be
    # pulled from it and the trace schema matches every runtime batch). ---
    data_pipe: Optional[BatchReplayPipe] = None
    if trainer_cfg.gcs_data_path:
        # When --data_file_template is set, generate filenames programmatically
        # and skip directory enumeration (avoids the 58-min ls /mnt/barnfs
        # tax). One file per epoch — the rotating window in
        # replay_batch_iter provides per-step diversity within that file.
        explicit_files: Optional[List[str]] = None
        if trainer_cfg.data_file_template:
            explicit_files = [trainer_cfg.data_file_template.format(i=1)]
            logger.info(
                f"Direct-read mode: file_list={explicit_files} (template "
                f"'{trainer_cfg.data_file_template}')"
            )
        data_pipe = BatchReplayPipe(
            path=trainer_cfg.gcs_data_path,
            rank=gloo_rank,
            world_size=gloo_world,
            replicate_across_ranks=trainer_cfg.data_replicate_across_ranks,
            file_list=explicit_files,
        )
        logger.info(
            f"Replay pipe: {len(data_pipe)} batches for rank {gloo_rank} "
            f"(replicate={trainer_cfg.data_replicate_across_ranks})"
        )
    else:
        logger.info("Using synthetic data (no --gcs_data_path)")

    # --- 5. Build model + overrides + compiled train_step.
    # _build_compiled_train_step pulls the FIRST batch from ``data_pipe``
    # (when set) and uses it as the trace batch, so the gm's input pytree
    # schema matches every subsequent runtime batch from the same pipe.
    # ``ro_per_key_budgets`` is returned here for the training loop's
    # replay iter to reuse the same per-key budgets we used at trace time.
    torch.manual_seed(0)
    (
        train_step,
        model,
        opt,
        batch,
        feature_config,
        key_to_size,
        ro_per_key_budgets,
    ) = _build_compiled_train_step(
        trainer_cfg,
        module_config,
        feature_config,
        backend=backend,
        num_ranks=num_ranks,
        num_processes=gloo_world,
        global_bs=global_bs,
        data_pipe=data_pipe,
    )

    # --- 6. Train ---
    _run_training_loop(
        trainer_cfg,
        train_step,
        model,
        opt,
        batch,
        is_main,
        global_bs,
        feature_config=feature_config if trainer_cfg.enable_padding else None,
        data_pipe=data_pipe,
        key_to_size=key_to_size,
        ro_per_key_budgets=ro_per_key_budgets,
    )

    # Cleanup. JAX's PJRT runtime + libtpu workers + the gloo collective
    # thread keep the process alive after main() returns, so the JobSet pod
    # never reaches Completed. Politely shut down both, then force-exit to
    # bypass any remaining background threads (atexit handlers / GC).
    if dist.is_initialized():
        dist.destroy_process_group()
    try:
        jax.distributed.shutdown()
    except Exception as e:  # noqa: BLE001 -- best-effort cleanup
        logger.warning(f"jax.distributed.shutdown() raised (non-fatal): {e}")
    logger.info("Cleanup complete; exiting.")
    os._exit(0)


if __name__ == "__main__":
    main()
