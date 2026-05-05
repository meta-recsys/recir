#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
RankFM (xsurface V3) E2E training on GKE TPU v7x with:
- recir @compile XLA GSPMD dense path on TPU (JAX PJRT)
- PreprocessorOverride + SparseArchOverride from xsurface_test_utils
  (these wrap the heavy preprocessor and ECSparseArch with dummy modules
  during tracing, and dispatch to the real modules at runtime)

Usage:
    python -u -m recir.examples.rankfm.train --preset small --max_steps 10
    python -u -m recir.examples.rankfm.train --max_steps 20 --profile_start_step 5 --profile_end_step 10
"""

from __future__ import annotations

import dataclasses
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
from hammer.v3.modules.recblock import _inputs_to_feature_bundle, RecBlock
from recir.api import compile, sync_device_state_to_torch
from recir.backend import BackendType
from recir.decompositions import SPMD_DECOMP_TABLE
from recir.distributed.parallel_mode import GSPMDParallel
from recir.gm_transformation import GraphModuleTransformation
from recir.lowering.fbgemm_fallback import _batched_complete_cumsum_impl
from recir.tests.xla.xsurface_test_utils import (
    _build_full_model,
    _build_overrides,
    _create_random_train_batch,
    _make_sample_input,
    TestConfigProvider,
)
from torchrec.distributed.embedding import EmbeddingCollectionSharder
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.types import ParameterConstraints
from torchrec.distributed.types import ShardingEnv, ShardingType
from torchrec.optim.keyed import KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

from .config import RankFMConfig

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


class _ECSharder(EmbeddingCollectionSharder):
    """EmbeddingCollection sharder offering ROW_WISE + TABLE_WISE.

    The actual per-table sharding choice is left to the planner via
    `ParameterConstraints` in `setup_dmp` (small tables -> TABLE_WISE so
    they aren't sliced too thin for fbgemm's CPU FUSED path; large tables
    -> ROW_WISE so they scale across ranks). Kernel is always FUSED.
    """

    def __init__(
        self,
        kernel_type: str,
        fused_params: Dict[str, Any] | None = None,
    ) -> None:
        self._kernel_type = kernel_type
        if fused_params is None:
            fused_params = {}
        if "learning_rate" not in fused_params:
            fused_params["learning_rate"] = 0.1
        super().__init__(fused_params=fused_params)

    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [
            ShardingType.ROW_WISE.value,
            ShardingType.TABLE_WISE.value,
        ]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [self._kernel_type]


def setup_dmp(
    model: nn.Module,
    cfg: RankFMConfig,
    gloo_world: int,
) -> Tuple[nn.Module, torch.optim.Optimizer]:
    """Shard the EC ROW_WISE on CPU via gloo, mirroring dlrmv3."""
    if not dist.is_initialized():
        logger.info("Single-node: skipping DMP, using model directly")
        dense_opt = torch.optim.SGD(model.parameters(), lr=cfg.dense_lr, foreach=False)
        return model, dense_opt

    pg = dist.distributed_c10d._get_default_group()
    sharders = [
        _ECSharder(
            kernel_type=EmbeddingComputeKernel.FUSED.value,
            fused_params={"learning_rate": cfg.embedding_lr},
        )
    ]

    # Per-table sharding constraints. Pure ROW_WISE failed because some
    # categorical tables in rankfm are tiny (interface_type ~ 10 rows,
    # device_os ~ 100 rows, etc.); slicing them across 32 ranks gives
    # <=1 row per rank, which trips an fbgemm CUDA-only fast path even
    # though we requested CPU FUSED. Pure TABLE_WISE works but doesn't
    # scale large tables (post_id, viewer_rid). Mix: small tables ->
    # TABLE_WISE (each lives on one rank), large -> ROW_WISE (sliced
    # across ranks for memory + compute parallelism).
    ec = (
        model.recblock.tokenizer_stack.mot_sparse_arch._sparse_arch._embedding_collection  # pyre-ignore[16]
    )
    ROW_WISE_MIN_ROWS_PER_RANK = 100
    row_wise_min = gloo_world * ROW_WISE_MIN_ROWS_PER_RANK
    constraints: Dict[str, ParameterConstraints] = {}
    for table in ec.embedding_configs():
        if table.num_embeddings >= row_wise_min:
            sharding = ShardingType.ROW_WISE.value
        else:
            sharding = ShardingType.TABLE_WISE.value
        constraints[table.name] = ParameterConstraints(
            sharding_types=[sharding],
        )
        logger.info(
            "  constraint: %s (%d rows) -> %s",
            table.name,
            table.num_embeddings,
            sharding,
        )

    planner = EmbeddingShardingPlanner(
        topology=Topology(world_size=gloo_world, compute_device="cpu"),
        constraints=constraints,
    )
    plan = planner.collective_plan(model, sharders, pg)
    logger.info("DMP sharding plan:")
    for module_path, module_plan in plan.plan.items():
        for param_name, param_sharding in module_plan.items():
            logger.info(
                "  %s.%s: sharding_type=%s compute_kernel=%s",
                module_path,
                param_name,
                param_sharding.sharding_type,
                getattr(param_sharding, "compute_kernel", "?"),
            )

    dmp = DistributedModelParallel(
        model,
        env=ShardingEnv.from_process_group(pg),
        plan=plan,
        # pyre-ignore[6]: _ECSharder is a valid ModuleSharder subclass
        sharders=sharders,
        device=torch.device("cpu"),
        init_data_parallel=False,
    )
    dense_opt = KeyedOptimizerWrapper(
        dict(in_backward_optimizer_filter(dmp.named_parameters())),
        lambda params: torch.optim.SGD(params, lr=cfg.dense_lr, foreach=False),
    )
    logger.info("DMP initialized: ROW_WISE EC on CPU (gloo_world=%d)", gloo_world)
    # Return dmp.module so override matching (isinstance(orig_submodule,
    # ECSparseArch)) still works - DMP shards the EmbeddingCollection in
    # place inside ECSparseArch but doesn't replace the wrapper.
    return dmp.module, dense_opt


def _patch_test_config_provider(cfg: RankFMConfig) -> None:
    """Wire CLI knobs into TestConfigProvider so _build_full_model picks them up.

    _build_full_model() in xsurface_test_utils.py calls TestConfigProvider()
    directly and applies hardcoded overrides (TEST_UC_RID_HASH_SIZE = 1M,
    L1/L2 stacks trimmed to 1 layer). To honor the CLI knobs without forking
    _build_full_model, we monkey-patch the module constants and inject a
    custom create_config that respects cfg.
    """
    import recir.tests.xla.xsurface_test_utils as xtu

    xtu.TEST_UC_RID_HASH_SIZE = cfg.hash_size_cap
    xtu.TEST_BATCH_SIZE = cfg.local_batch_size
    xtu.TEST_MAX_UIH_SEQ_LEN = cfg.max_uih_seq_len
    xtu.TEST_NUM_CANDIDATES = cfg.num_candidates

    # Override stack trimming inside _build_full_model. _build_full_model
    # always slices stack.stu_configs[:1]; we want [:l1_num_layers] and
    # [:l2_num_layers]. Monkey-patch by wrapping create_config.
    _orig_create: Any = TestConfigProvider.create_config

    def _patched_create_config(self: TestConfigProvider) -> Any:
        config = _orig_create(self)
        module_config = config.module_config
        if len(module_config.stu_stack_configs) >= 1:
            module_config.stu_stack_configs[
                0
            ].stu_configs = module_config.stu_stack_configs[0].stu_configs[
                : cfg.l1_num_layers
            ]
        if len(module_config.stu_stack_configs) >= 2:
            module_config.stu_stack_configs[
                1
            ].stu_configs = module_config.stu_stack_configs[1].stu_configs[
                : cfg.l2_num_layers
            ]
        if cfg.disable_seq_len_sampling:
            module_config.hstu_seq_len_sampling_configs = None
        return config

    # pyre-ignore[8]
    TestConfigProvider.create_config = _patched_create_config


def _build_compiled_train_step(
    cfg: RankFMConfig,
    num_ranks: int,
    num_processes: int,
    gloo_world: int,
    global_bs: int,
) -> Tuple[Any, Any, Any, Any, Any]:
    """Build model, batch, overrides, and the compiled train_step.

    Returns (train_step, model, opt, batch, kjt_keys_tuple).
    """
    device = torch.device("cpu")
    model, feature_config = _build_full_model(device)

    for p in model.parameters():
        if p.is_floating_point():
            p.requires_grad_(True)

    # Sample input + batch must be built BEFORE DMP because _build_overrides
    # walks the recblock and registers forward hooks; doing this against the
    # post-DMP module is fine but cleanest before sharding.
    torch.manual_seed(0)
    _, ro_keys, nro_keys = _make_sample_input(
        model.recblock, device, batch_size=cfg.local_batch_size
    )
    batch = _create_random_train_batch(
        feature_config,
        device,
        batch_size=cfg.local_batch_size,
        ro_kjt_keys=ro_keys,
        nro_kjt_keys=nro_keys,
    )

    # Build overrides BEFORE DMP. _build_overrides runs the sparse arch
    # eagerly and captures its output via a forward hook; that captured
    # tensor becomes the SparseArchOverride dummy used during XLA tracing.
    expanded_batch = _create_random_train_batch(
        feature_config,
        device,
        batch_size=global_bs,
        ro_kjt_keys=ro_keys,
        nro_kjt_keys=nro_keys,
    )
    all_labels = expanded_batch.all_labels
    assert all_labels is not None
    batch_sample = _inputs_to_feature_bundle(
        expanded_batch.ro_id_list_features,
        None,
        expanded_batch.nro_id_list_features,
        expanded_batch.num_candidates,
        all_labels,
    )
    recblock = model.recblock
    assert isinstance(recblock, RecBlock)
    overrides = _build_overrides(
        recblock,
        batch_sample,
        device,
        batch_size=global_bs,
    )

    # Apply gloo + DMP for ROW_WISE embedding sharding. After this call,
    # model.recblock.tokenizer_stack.mot_sparse_arch._sparse_arch is still
    # an _ECSparseArch (so SparseArchOverride.replacement matches via
    # isinstance), but the EmbeddingCollection inside is sharded.
    model, dense_opt = setup_dmp(model, cfg, gloo_world)

    # Capture KJT keys for the runtime reconstruction inside train_step.
    ro_kjt = batch.ro_id_list_features
    ro_kjt_keys: List[str] = ro_kjt.keys()
    nro_kjt = batch.nro_id_list_features
    nro_kjt_keys: List[str] = nro_kjt.keys()
    unary_kjt = batch.unary_features
    assert unary_kjt is not None
    unary_kjt_keys: List[str] = unary_kjt.keys()

    opt = dense_opt

    @compile(
        module_override=overrides,
        parallel_mode=GSPMDParallel(
            "replicate",
            loss_reduction="sum",
            num_ranks=num_ranks,
            num_processes=num_processes,
        ),
        # pyre-ignore[6]: GraphModuleTransformation is callable
        gm_transformation=GraphModuleTransformation(
            backend=BackendType.XLA_GSPMD,
            enable_device_state=True,
        ),
    )
    # pyre-ignore
    def train_step(
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
        # supervision_{label,weight} are passed as (B_NRO, T) so dim-0
        # is the batch dim and divides evenly across num_ranks. The model
        # expects (T, B_NRO), so we transpose back inside.
        supervision_label_t: torch.Tensor,
        supervision_weight_t: torch.Tensor,
        all_labels: Dict[str, torch.Tensor],
    ) -> None:
        ro_id_list_features = KeyedJaggedTensor(
            keys=ro_kjt_keys,
            values=ro_id_list_values,
            lengths=ro_id_list_lengths,
        )
        nro_id_list_features = KeyedJaggedTensor(
            keys=nro_kjt_keys,
            values=nro_id_list_values,
            lengths=nro_id_list_lengths,
        )
        unary_features = KeyedJaggedTensor(
            keys=unary_kjt_keys,
            values=unary_values,
            lengths=unary_lengths,
        )
        new_batch = dataclasses.replace(
            batch,
            ro_float_features=ro_float_features,
            nro_float_features=nro_float_features,
            ro_id_list_features=ro_id_list_features,
            nro_id_list_features=nro_id_list_features,
            unary_features=unary_features,
            num_candidates=num_candidates,
            all_labels=all_labels,
            supervision_label=supervision_label_t.t().contiguous(),
            supervision_weight=supervision_weight_t.t().contiguous(),
        )
        all_losses, _ = mod.forward(new_batch)  # pyre-ignore[29]
        loss: torch.Tensor = all_losses.sum()
        loss.backward()
        opt.step()

    kjt_keys_tuple = (ro_kjt_keys, nro_kjt_keys, unary_kjt_keys)
    return train_step, model, opt, batch, kjt_keys_tuple


def _is_jax_profiling_enabled(cfg: RankFMConfig) -> bool:
    """Check if JAX profiling should be enabled (TPU present + start step > 0)."""
    try:
        is_tpu = len(jax.devices()) > 0 and jax.devices()[0].platform == "tpu"
        return is_tpu and cfg.profile_start_step > 0
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


def _run_training_loop(
    cfg: RankFMConfig,
    train_step: Any,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    batch: Any,
    is_main: bool,
    global_bs: int,
) -> None:
    """Execute the compiled training loop with optional JAX profiling."""
    ro_kjt = batch.ro_id_list_features
    nro_kjt = batch.nro_id_list_features
    unary_kjt = batch.unary_features
    assert unary_kjt is not None

    jax_profiling = _is_jax_profiling_enabled(cfg)
    profiling_active = False
    if jax_profiling:
        logger.info(
            f"JAX profiling enabled: steps {cfg.profile_start_step}-{cfg.profile_end_step}, "
            f"dir={cfg.profile_dir}"
        )

    model.train(True)
    logger.info(f"Starting training: {cfg.max_steps} steps, global_bs={global_bs}")
    start_time = time.time()

    for step in range(1, cfg.max_steps + 1):
        if jax_profiling and step == cfg.profile_start_step:
            if not cfg.profile_dir.startswith("gs://"):
                os.makedirs(cfg.profile_dir, exist_ok=True)
            jax.profiler.start_trace(cfg.profile_dir)
            profiling_active = True
            logger.info(f"[PROFILE] Started trace at step {step}")

        step_start = time.time()

        train_step(
            model,
            opt,
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
        sync_device_state_to_torch(train_step)

        step_time = time.time() - step_start

        if jax_profiling and step == cfg.profile_end_step and profiling_active:
            jax.profiler.stop_trace()
            profiling_active = False
            logger.info(f"[PROFILE] Stopped trace at step {step}")

        if is_main:
            logger.info(f"Step {step}/{cfg.max_steps} | time={step_time:.2f}s")

        if jax_profiling and cfg.enable_memory_log and is_main:
            _log_hbm_memory()

    total_time = time.time() - start_time

    if profiling_active:
        jax.profiler.stop_trace()
        logger.info("[PROFILE] Stopped trace (training ended)")

    if is_main:
        logger.info(
            f"Training complete: {cfg.max_steps} steps in {total_time:.1f}s "
            f"({total_time / cfg.max_steps:.2f}s/step)"
        )


def main() -> None:
    _setup_logging()
    cfg = RankFMConfig.from_args()
    logger.info(f"Config: {cfg}")

    # --- 1. Dual PG init ---
    gloo_rank, gloo_world = init_dual_pg()
    is_main = gloo_rank == 0

    # --- 2. Apply CLI knobs to xsurface_test_utils' TestConfigProvider ---
    _patch_test_config_provider(cfg)

    # --- 3. Build model + overrides + compiled train_step ---
    torch.manual_seed(0)
    num_tpu_devices = jax.device_count()
    num_ranks = num_tpu_devices if num_tpu_devices > 1 else gloo_world
    # global_bs = local_batch_size * gloo_world (NOT * num_ranks). num_ranks
    # (jax.device_count) is the GSPMD mesh size used for grad averaging;
    # gloo_world is the number of OS processes contributing data. global_bs
    # must equal local_batch_size * gloo_world for the override's
    # cached_output (sized at global_bs) to match the expanded trace input.
    global_bs = cfg.local_batch_size * gloo_world
    train_step, model, opt, batch, _ = _build_compiled_train_step(
        cfg,
        num_ranks=num_ranks,
        num_processes=gloo_world,
        gloo_world=gloo_world,
        global_bs=global_bs,
    )

    # --- 4. Train ---
    _run_training_loop(cfg, train_step, model, opt, batch, is_main, global_bs)

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
