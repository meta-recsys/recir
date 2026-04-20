#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
DLRMv3 E2E training on GKE TPU v7x with:
- torchrec DMP ROW_WISE embedding sharding on CPU (gloo PG)
- recir @compile XLA GSPMD dense path on TPU (JAX PJRT)

Usage:
    python -u -m recir.examples.dlrmv3.train --preset small --max_steps 10
    python -u -m recir.examples.dlrmv3.train --max_steps 20 --profile_start_step 5 --profile_end_step 10
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Tuple

import fbgemm_gpu  # noqa: F401 -- force C++ op registration before fallback
import jax
import recir.lowering.fbgemm_fallback  # noqa: F401 -- registers SPMD_DECOMP_TABLE entries
import torch
import torch.distributed as dist
import torch.nn as nn
from recir.api import compile, sync_device_state_to_torch
from recir.backend import BackendType
from recir.distributed.parallel_mode import GSPMDParallel
from recir.gm_transformation import GraphModuleTransformation
from recir.tests.xla.dlrmv3_test_utils import (
    create_dlrmv3_wrapper,
    create_static_batch,
    DlrmV3SparseOverride,
    DummyPreprocess,
    SMALL_EMBEDDING_DIM,
)
from torchrec.distributed.embedding import EmbeddingCollectionSharder
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.types import ShardingEnv, ShardingType
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.optim.keyed import KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter

from .config import DLRMv3Config

logger: logging.Logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    logging.basicConfig(
        format="%(levelname).1s%(asctime)s %(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
        datefmt="%m%d %H:%M:%S",
        force=True,
    )


def init_dual_pg() -> Tuple[int, int]:
    """Initialize gloo PG alongside JAX PJRT on GKE TPU.

    JAX is auto-initialized by GKE env vars.
    Gloo uses TPU_PROCESS_ADDRESSES for discovery and JOB_COMPLETION_INDEX for rank.

    Returns (gloo_rank, gloo_world_size).
    """
    jax_rank = jax.process_index()
    jax_world = jax.process_count()
    logger.info(
        f"JAX: process_index={jax_rank}, process_count={jax_world}, "
        f"devices={len(jax.devices())}, platform={jax.devices()[0].platform}"
    )

    tpu_addrs = os.environ.get("TPU_PROCESS_ADDRESSES", "")
    addr_list = [a.strip() for a in tpu_addrs.split(",") if a.strip()]

    if not addr_list or len(addr_list) <= 1:
        logger.info("Single-node mode: skipping gloo PG init")
        return 0, 1

    coordinator = addr_list[0].split(":")[0]
    gloo_rank = int(os.environ.get("JOB_COMPLETION_INDEX", "0"))
    gloo_world = len(addr_list)
    gloo_port = int(os.environ.get("MASTER_PORT", "29500"))

    logger.info(
        f"Gloo init: rank={gloo_rank}, world={gloo_world}, "
        f"coordinator={coordinator}:{gloo_port}"
    )
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://{coordinator}:{gloo_port}",
        world_size=gloo_world,
        rank=gloo_rank,
        timeout=torch.distributed.distributed_c10d.timedelta(seconds=300),
    )
    logger.info(
        f"Gloo PG initialized: rank={dist.get_rank()}, world={dist.get_world_size()}"
    )
    return gloo_rank, gloo_world


def create_model_and_data(
    cfg: DLRMv3Config, global_bs: int
) -> Tuple[nn.Module, List[EmbeddingConfig], List[str], Tuple[torch.Tensor, ...], Any]:
    """Create DLRMv3 model with debug HSTU config and synthetic batch.

    Uses the default get_hstu_config() (debug gin config) which includes all
    temporal features needed by DummyPreprocess and DlrmV3Dense. Custom
    embedding table sizes come from cfg.hash_size / cfg.embedding_dim.
    """
    table_configs = [
        EmbeddingConfig(
            num_embeddings=cfg.hash_size,
            embedding_dim=SMALL_EMBEDDING_DIM,
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
            embedding_dim=SMALL_EMBEDDING_DIM,
            name="viewer_id",
            feature_names=["viewer_id"],
        ),
        EmbeddingConfig(
            num_embeddings=cfg.hash_size,
            embedding_dim=SMALL_EMBEDDING_DIM,
            name="dummy_contexual",
            feature_names=["dummy_contexual"],
        ),
    ]

    wrapper, hstu_config, _, emb_keys = create_dlrmv3_wrapper(
        global_bs, table_configs=table_configs
    )

    gloo_rank = dist.get_rank() if dist.is_initialized() else 0
    batch = create_static_batch(hstu_config, cfg.local_batch_size, seed=42 + gloo_rank)

    param_count = sum(p.numel() for p in wrapper.parameters())
    emb_count = sum(t.num_embeddings * t.embedding_dim for t in table_configs)
    logger.info(
        f"Model: {param_count:,} total params, {emb_count:,} embedding params, "
        f"dense: {param_count - emb_count:,}"
    )

    return wrapper, table_configs, emb_keys, batch, hstu_config


def setup_dmp(
    wrapper: nn.Module,
    table_configs: List[EmbeddingConfig],
    cfg: DLRMv3Config,
    gloo_world: int,
) -> Tuple[nn.Module, torch.optim.Optimizer]:
    """Shard embeddings with DMP ROW_WISE on CPU, create dense optimizer."""

    if not dist.is_initialized():
        logger.info("Single-node: skipping DMP, using model directly")
        dense_opt = torch.optim.SGD(
            wrapper.parameters(), lr=cfg.dense_lr, foreach=False
        )
        return wrapper, dense_opt

    pg = dist.distributed_c10d._get_default_group()

    sharders = [
        _TestSharder(
            sharding_type=ShardingType.ROW_WISE.value,
            kernel_type=EmbeddingComputeKernel.FUSED.value,
            fused_params={"learning_rate": cfg.embedding_lr},
        )
    ]

    planner = EmbeddingShardingPlanner(
        topology=Topology(world_size=gloo_world, compute_device="cpu"),
    )
    plan = planner.collective_plan(wrapper, sharders, pg)

    logger.info("DMP sharding plan:")
    for module_path, module_plan in plan.plan.items():
        for param_name, param_sharding in module_plan.items():
            logger.info(f"  {module_path}.{param_name}: {param_sharding.sharding_type}")

    dmp = DistributedModelParallel(
        wrapper,
        env=ShardingEnv.from_process_group(pg),
        plan=plan,
        # pyre-ignore[6]: _TestSharder is a valid ModuleSharder subclass
        sharders=sharders,
        device=torch.device("cpu"),
        init_data_parallel=False,
    )

    dense_opt = KeyedOptimizerWrapper(
        dict(in_backward_optimizer_filter(dmp.named_parameters())),
        lambda params: torch.optim.SGD(params, lr=cfg.dense_lr, foreach=False),
    )

    logger.info("DMP initialized with ROW_WISE embedding sharding on CPU")
    return dmp, dense_opt


class _TestSharder(EmbeddingCollectionSharder):
    def __init__(
        self,
        sharding_type: str,
        kernel_type: str,
        fused_params: Dict[str, Any] | None = None,
    ) -> None:
        self._sharding_type = sharding_type
        self._kernel_type = kernel_type
        if fused_params is None:
            fused_params = {}
        if "learning_rate" not in fused_params:
            fused_params["learning_rate"] = 0.1
        super().__init__(fused_params=fused_params)

    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [self._sharding_type]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [self._kernel_type]


def _run_training_loop(
    cfg: DLRMv3Config,
    dmp: nn.Module,
    dense_opt: torch.optim.Optimizer,
    local_batch: Tuple[torch.Tensor, ...],
    hstu_config: Any,
    emb_keys: List[str],
    gloo_world: int,
    is_main: bool,
    global_bs: int,
) -> None:
    """Execute the compiled training loop with optional JAX profiling."""
    # Create sparse override for @compile
    dummy = DummyPreprocess(
        emb_keys=emb_keys,
        batch_size=global_bs,
        embedding_dim=SMALL_EMBEDDING_DIM,
        static_uih_len=cfg.uih_len,
        static_num_candidates=cfg.num_candidates,
        user_emb_feature_names=hstu_config.user_embedding_feature_names,
        item_emb_feature_names=hstu_config.item_embedding_feature_names,
        hstu_configs=hstu_config,
    )
    sparse_override = DlrmV3SparseOverride(dummy)
    sparse_override._num_ranks = gloo_world

    # Compile dense path
    num_tpu_devices = jax.device_count()
    num_ranks = num_tpu_devices if num_tpu_devices > 1 else gloo_world

    @compile(
        module_override=[sparse_override],
        parallel_mode=GSPMDParallel(
            "replicate",
            loss_reduction="sum",
            num_ranks=num_ranks,
            num_processes=gloo_world,
        ),
        # pyre-ignore[6]: GraphModuleTransformation is callable
        gm_transformation=GraphModuleTransformation(
            backend=BackendType.XLA_GSPMD,
            enable_device_state=True,
        ),
    )
    def train_step(
        mod: nn.Module,
        opt: torch.optim.Optimizer,
        uih_values: torch.Tensor,
        uih_lengths: torch.Tensor,
        cand_values: torch.Tensor,
        cand_lengths: torch.Tensor,
    ) -> None:
        aux_losses: Dict[str, torch.Tensor] = mod(
            uih_values, uih_lengths, cand_values, cand_lengths
        )
        loss: torch.Tensor = torch.stack(list(aux_losses.values())).sum()
        loss.backward()
        opt.step()

    # JAX profiler setup
    jax_profiling = False
    profiling_active = False
    try:
        if jax.devices()[0].platform == "tpu":
            jax_profiling = True
    except (RuntimeError, IndexError):
        logger.warning("Could not detect JAX TPU platform, profiling disabled")

    # Training loop
    model_to_train = dmp.module if isinstance(dmp, DistributedModelParallel) else dmp
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
    _setup_logging()
    cfg = DLRMv3Config.from_args()
    logger.info(f"Config: {cfg}")

    # --- 1. Dual PG init ---
    gloo_rank, gloo_world = init_dual_pg()
    is_main = gloo_rank == 0
    global_bs = cfg.local_batch_size * gloo_world

    # --- 2. Create model and data ---
    torch.manual_seed(0)
    wrapper, table_configs, emb_keys, local_batch, hstu_config = create_model_and_data(
        cfg, global_bs
    )

    # --- 3. DMP embedding sharding ---
    dmp, dense_opt = setup_dmp(wrapper, table_configs, cfg, gloo_world)

    # --- 4. Compile and train ---
    _run_training_loop(
        cfg,
        dmp,
        dense_opt,
        local_batch,
        hstu_config,
        emb_keys,
        gloo_world,
        is_main,
        global_bs,
    )

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
