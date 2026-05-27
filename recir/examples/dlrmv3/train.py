#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
DLRMv3 E2E training on GKE TPU v7x.

Supports two embedding backends (--embedding_backend):
  cpu        torchrec DMP ROW_WISE on CPU + gloo PG
  tpu JAX-native tables + JAX gather / tpu_sparse_dense_matmul

Dense HSTU path always runs on TPU via recir @compile + XLA GSPMD.

Usage:
    python -u -m recir.examples.dlrmv3.train --preset small --max_steps 10
    python -u -m recir.examples.dlrmv3.train --max_steps 20 --profile_start_step 5 --profile_end_step 10
    python -u -m recir.examples.dlrmv3.train --preset small --max_steps 5 \\
        --embedding_backend tpu
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import fbgemm_gpu  # noqa: F401 -- force C++ op registration before fallback
import jax
import recir.lowering.fbgemm_fallback  # noqa: F401 -- registers SPMD_DECOMP_TABLE entries
import torch
import torch.distributed as dist
import torch.nn as nn
from recir.api import sync_device_state_to_torch
from recir.backends import XLAGSPMDBackend
from recir.distributed.parallel_mode import GSPMDParallel
from recir.examples.dlrmv3.jax_embedding import compile_with_jax_embedding
from recir.tests.xla.dlrmv3_test_utils import (
    create_dlrmv3_wrapper,
    create_static_batch,
    DummyPreprocess,
    get_hstu_config,
)
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.modules.embedding_configs import EmbeddingConfig

from .config import DLRMv3Config
from .embedding import build_embedding_backend, EmbeddingBackend, JAXTPUBackend

logger: logging.Logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    logging.basicConfig(
        format="%(levelname).1s%(asctime)s %(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
        datefmt="%m%d %H:%M:%S",
        force=True,
    )


def create_model_and_data(
    cfg: DLRMv3Config, global_bs: int
) -> tuple[nn.Module, list[EmbeddingConfig], list[str], tuple[torch.Tensor, ...], Any]:
    """Create DLRMv3 model with debug HSTU config and synthetic batch.

    Uses get_hstu_config() with dimensions from cfg, so medium/large presets
    produce correspondingly larger models.
    """
    table_configs = [
        EmbeddingConfig(
            num_embeddings=cfg.hash_size,
            embedding_dim=cfg.embedding_dim,
            name="post_id",
            feature_names=[
                "uih_post_id",
                "item_post_id",
                "uih_owner_id",
                "item_owner_id",
            ],
        ),
        EmbeddingConfig(
            num_embeddings=cfg.hash_size,
            embedding_dim=cfg.embedding_dim,
            name="viewer_id",
            feature_names=["viewer_id"],
        ),
        EmbeddingConfig(
            num_embeddings=cfg.hash_size,
            embedding_dim=cfg.embedding_dim,
            name="dummy_contexual",
            feature_names=["dummy_contexual"],
        ),
    ]

    hstu_cfg = get_hstu_config(
        embedding_dim=cfg.embedding_dim,
        hidden_dim=cfg.hidden_dim,
        transducer_dim=cfg.transducer_dim,
        attn_dim=cfg.attn_dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        uih_len=cfg.uih_len,
        num_candidates=cfg.num_candidates,
    )
    wrapper, hstu_config, _, emb_keys = create_dlrmv3_wrapper(
        global_bs, hstu_config=hstu_cfg, table_configs=table_configs
    )
    # Sync hstu_config with user-configurable values so the model,
    # DummyPreprocess, and batch all agree on sequence lengths.
    hstu_config.max_num_candidates = cfg.num_candidates
    hstu_config.max_seq_len = cfg.uih_len + cfg.num_candidates + 16

    gloo_rank = dist.get_rank() if dist.is_initialized() else 0
    batch = create_static_batch(
        hstu_config,
        cfg.local_batch_size,
        seed=42 + gloo_rank,
        uih_len_override=cfg.uih_len,
        num_candidates_override=cfg.num_candidates,
        hash_size_override=cfg.hash_size,
    )

    param_count = sum(p.numel() for p in wrapper.parameters())
    emb_count = sum(t.num_embeddings * t.embedding_dim for t in table_configs)
    logger.info(
        f"Model: {param_count:,} total params, {emb_count:,} embedding params, "
        f"dense: {param_count - emb_count:,}"
    )

    return wrapper, table_configs, emb_keys, batch, hstu_config


def _run_training_loop(
    cfg: DLRMv3Config,
    backend: EmbeddingBackend,
    model: nn.Module,
    dense_opt: torch.optim.Optimizer,
    local_batch: tuple[torch.Tensor, ...],
    hstu_config: Any,
    emb_keys: list[str],
    num_ranks: int,
    is_main: bool,
    global_bs: int,
) -> None:
    """Execute the compiled training loop with optional JAX profiling."""
    # Create the dispatcher (orchestrator-style override) for the embedding
    # backend.  Both CPU and JAX backends expose ``install_eager`` +
    # ``install_graph`` (the dispatcher pattern), so the same compile
    # orchestrator (`compile_with_jax_embedding`) works for both.
    dummy = DummyPreprocess(
        emb_keys=emb_keys,
        batch_size=cfg.local_batch_size,
        embedding_dim=cfg.embedding_dim,
        static_uih_len=cfg.uih_len,
        static_num_candidates=cfg.num_candidates,
        user_emb_feature_names=hstu_config.user_embedding_feature_names,
        item_emb_feature_names=hstu_config.item_embedding_feature_names,
        hstu_configs=hstu_config,
    )
    dispatcher = backend.get_dispatcher(dummy, num_ranks=num_ranks)

    # compile_ranks: use all JAX devices when on TPU (device_count > 1 on TPU,
    # = 1 on CPU-only machines where we fall back to the gloo world size).
    num_tpu_devices = jax.device_count()
    compile_ranks = num_tpu_devices if num_tpu_devices > 1 else num_ranks

    def train_step_fn(
        mod: nn.Module,
        opt: torch.optim.Optimizer,
        uih_values: torch.Tensor,
        uih_lengths: torch.Tensor,
        cand_values: torch.Tensor,
        cand_lengths: torch.Tensor,
    ) -> None:
        aux_losses: dict[str, torch.Tensor] = mod(
            uih_values, uih_lengths, cand_values, cand_lengths
        )
        loss: torch.Tensor = torch.stack(list(aux_losses.values())).sum()
        loss.backward()
        opt.step()

    model_to_train_for_compile = (
        model.module if isinstance(model, DistributedModelParallel) else model
    )
    train_step = compile_with_jax_embedding(
        model_to_train_for_compile,
        train_step_fn,
        model_to_train_for_compile,
        dense_opt,
        *local_batch,
        dispatcher=dispatcher,
        parallel_mode=GSPMDParallel(
            "replicate",
            loss_reduction="sum",
            num_ranks=compile_ranks,
            num_processes=num_ranks,
        ),
        backend=XLAGSPMDBackend(),
    )

    # JAX profiler setup — enabled when on TPU and profile_start_step > 0
    profiling_active = False
    jax_profiling = False
    try:
        is_tpu = len(jax.devices()) > 0 and jax.devices()[0].platform == "tpu"
        jax_profiling = is_tpu and cfg.profile_start_step > 0
    except (RuntimeError, IndexError):
        logger.warning("Could not detect JAX TPU platform, profiling disabled")
    if jax_profiling:
        logger.info(
            f"JAX profiling enabled: steps {cfg.profile_start_step}-{cfg.profile_end_step}, "
            f"dir={cfg.profile_dir}"
        )

    # Training loop
    model_to_train = (
        model.module if isinstance(model, DistributedModelParallel) else model
    )
    model_to_train.train(True)

    logger.info(f"Starting training: {cfg.max_steps} steps, global_bs={global_bs}")
    start_time = time.time()

    for step in range(1, cfg.max_steps + 1):
        step_start = time.time()

        train_step(model_to_train, dense_opt, *local_batch)
        sync_device_state_to_torch(train_step)

        step_time = time.time() - step_start

        # Profiling
        if jax_profiling and step == cfg.profile_start_step:
            if not cfg.profile_dir.startswith("gs://"):
                os.makedirs(cfg.profile_dir, exist_ok=True)
            jax.profiler.start_trace(cfg.profile_dir)
            profiling_active = True
            logger.info(f"[PROFILE] Started trace at step {step}")
        elif jax_profiling and step == cfg.profile_end_step and profiling_active:
            jax.profiler.stop_trace()
            profiling_active = False
            logger.info(f"[PROFILE] Stopped trace at step {step}")

        if is_main:
            logger.info(f"Step {step}/{cfg.max_steps} | time={step_time:.2f}s")

        # HBM memory logging
        if jax_profiling and cfg.enable_memory_log and is_main:
            for i, dev in enumerate(jax.local_devices()):
                stats = dev.memory_stats()
                if stats:
                    used = stats.get("bytes_in_use", 0)
                    limit = stats.get("bytes_limit", 0)
                    logger.info(
                        f"  Device {i} HBM: {used / 1e9:.2f}/{limit / 1e9:.2f} GB "
                        f"({100 * used / max(limit, 1):.1f}%)"
                    )

    total_time = time.time() - start_time

    if profiling_active and jax_profiling:
        jax.profiler.stop_trace()
        logger.info("[PROFILE] Stopped trace (training ended)")

    if is_main:
        logger.info(
            f"Training complete: {cfg.max_steps} steps in {total_time:.1f}s "
            f"({total_time / cfg.max_steps:.2f}s/step)"
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
    cfg = DLRMv3Config.from_args()
    logger.info(f"Config: {cfg}")

    # --- 1. Build embedding backend and init PG ---
    backend = build_embedding_backend(cfg.embedding_backend)
    gloo_rank, gloo_world = backend.init_pg()
    is_main = gloo_rank == 0
    global_bs = cfg.local_batch_size * gloo_world

    # --- 2. Create model and data ---
    torch.manual_seed(0)
    wrapper, table_configs, emb_keys, local_batch, hstu_config = create_model_and_data(
        cfg, global_bs
    )

    # --- 3. Set up embedding storage ---
    if isinstance(backend, JAXTPUBackend):
        # UIH seq features: uih_len tokens per sample (token-as-sample trick)
        # Candidate features: num_candidates IDs per sample (same trick)
        # Contextual features (viewer_id, dummy_contexual): pooled, stay at default 1
        backend.seq_lens = {
            "uih_post_id": cfg.uih_len,
            "uih_owner_id": cfg.uih_len,
            "item_post_id": cfg.num_candidates,
            "item_owner_id": cfg.num_candidates,
        }
        backend.global_batch_size = global_bs
    model, dense_opt = backend.setup(
        wrapper, table_configs, cfg.embedding_lr, cfg.dense_lr, gloo_world
    )

    # --- 4. Compile and train ---
    _run_training_loop(
        cfg,
        backend,
        model,
        dense_opt,
        local_batch,
        hstu_config,
        emb_keys,
        num_ranks=gloo_world,
        is_main=is_main,
        global_bs=global_bs,
    )

    # Cleanup

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
