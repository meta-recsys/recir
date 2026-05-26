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

import copy
import logging
import os
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

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
from recir.region_transforms.padding import (
    compute_padded_shape,
    JaggedPairConfig,
    PaddingRegionTransform,
)
from recir.tests.xla.xsurface_test_utils import (
    _build_full_model,
    _build_overrides,
    _create_random_train_batch,
    _create_uniform_train_batch,
    resize_batch_inplace,
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

from .config import RankFMConfig
from .data_pipe import BatchReplayPipe, compute_kjt_key_to_size, replay_batch_iter

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
) -> Tuple[Any, Any, Any, Any, Any, Dict[str, int]]:
    """Build model, batch, overrides, and the compiled train_step.

    Returns (train_step, model, opt, batch, feature_config, key_to_size).
    ``key_to_size`` is the model's KJT-feature -> num_embeddings map computed
    pre-DMP (post-DMP the EC is wrapped in a ``ShardedEmbeddingCollection``
    that doesn't expose ``embedding_configs()``); use it to clamp loaded IDs
    into the model's hash space at every runtime step.
    """
    device = torch.device("cpu")
    model, feature_config = _build_full_model(device)

    # Sync feature_config with the test constants so randgen produces
    # shapes that match the padding budgets and override expectations.
    feature_config.history_len = cfg.max_uih_seq_len
    feature_config.num_candidates_per_request = cfg.num_candidates

    for p in model.parameters():
        if p.is_floating_point():
            p.requires_grad_(True)

    module_config = model.module_config  # pyre-ignore[16]

    torch.manual_seed(0)
    batch = _create_uniform_train_batch(
        feature_config,
        device,
        batch_size=cfg.local_batch_size,
    )

    if cfg.enable_padding:
        # Pre-compute padded_bs and pair_padded_bs using the SAME formula as
        # PaddingRegionTransform's ``_discover``.  The override-built sample
        # batch must already be at the post-pad sizes, otherwise the cached
        # sparse-arch output won't dim-match the runtime-padded forward path
        # (cat fails downstream in ``_forward_post_hstu``).
        nd = num_ranks
        padded_bs = compute_padded_shape(global_bs, nd)
        # nro pair has its own batch dim (B_NRO != B_RO).  At trace time
        # B_NRO = global_bs * num_candidates_per_request (uniform)
        trace_pair_bs = global_bs * cfg.num_candidates
        pair_padded_bs = compute_padded_shape(trace_pair_bs, nd)

        # Override-build batch must be at the post-pad sizes so the cached
        # sparse-arch output dim-matches the runtime-padded forward path.
        # Build it by deep-copying the small runtime batch and resizing in
        # place; calling ``_create_uniform_train_batch(batch_size=padded_bs)``
        # directly can SIGABRT inside ``RandomDataPipe``'s C++ allocator at
        # large ``batch_size * num_candidates_per_request * history_len``.
        expanded_batch = copy.deepcopy(batch)
        resize_batch_inplace(
            expanded_batch,
            target_b_ro=padded_bs,
            target_b_nro=pair_padded_bs,
        )
        all_labels = expanded_batch.all_labels
        assert all_labels is not None
        override_batch_sample = _inputs_to_feature_bundle(
            expanded_batch.ro_id_list_features,
            None,
            expanded_batch.nro_id_list_features,
            expanded_batch.num_candidates,
            all_labels,
        )
        override_bs = padded_bs
    else:
        all_labels = batch.all_labels
        assert all_labels is not None
        override_batch_sample = _inputs_to_feature_bundle(
            batch.ro_id_list_features,
            None,
            batch.nro_id_list_features,
            batch.num_candidates,
            all_labels,
        )
        override_bs = global_bs

    recblock = model.recblock
    assert isinstance(recblock, RecBlock)
    overrides = _build_overrides(
        recblock,
        override_batch_sample,
        device,
        batch_size=override_bs,
        num_processes=num_processes,
    )

    # Compute KJT key -> num_embeddings map BEFORE DMP, while the EC still
    # exposes embedding_configs() (the sharded post-DMP wrapper does not).
    key_to_size: Dict[str, int] = compute_kjt_key_to_size(model)

    # Apply gloo + DMP for ROW_WISE embedding sharding.
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

    model.set_recir_kjt_keys(  # pyre-ignore[29]
        ro_kjt_keys, nro_kjt_keys, unary_kjt_keys
    )

    compile_kwargs: Dict[str, Any] = {
        "module_override": overrides,
        "parallel_mode": GSPMDParallel(
            "replicate",
            loss_reduction="sum",
            num_ranks=num_ranks,
            num_processes=num_processes,
        ),
        "gm_transformation": GraphModuleTransformation(
            backend=BackendType.XLA_GSPMD,
            enable_device_state=True,
        ),
    }

    if cfg.enable_padding:
        contextual_budgets = dict(module_config.contextual_feature_to_max_length)
        history_budget = cfg.max_uih_seq_len
        ro_per_sample = [
            contextual_budgets.get(key, history_budget) for key in ro_kjt_keys
        ]
        compile_kwargs["region_transforms"] = [
            PaddingRegionTransform(
                jagged_pairs={
                    "ro_id_list_values": JaggedPairConfig(
                        "ro_id_list_lengths",
                        per_sample_budgets=ro_per_sample,
                    ),
                    "nro_id_list_values": JaggedPairConfig(
                        "nro_id_list_lengths",
                        per_sample_budgets=1,
                        batch_size_arg="nro_float_features",
                        count_arg="num_candidates",
                        max_per_sample_count=cfg.num_candidates,
                    ),
                    "unary_values": JaggedPairConfig(
                        "unary_lengths",
                        per_sample_budgets=1,
                        batch_size_arg="nro_float_features",
                        count_arg="num_candidates",
                        max_per_sample_count=cfg.num_candidates,
                    ),
                },
                batch_size_arg="ro_float_features",
                num_devices=num_ranks,
                num_processes=num_processes,
            ),
        ]

    @compile(**compile_kwargs)  # pyre-ignore[6]
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

    return train_step, model, opt, batch, feature_config, key_to_size


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
    cfg: RankFMConfig,
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
    if cfg.enable_padding and feature_config is not None and step > 1:
        torch.manual_seed(step)
        return _create_random_train_batch(
            feature_config,
            torch.device("cpu"),
            batch_size=cfg.local_batch_size,
        )
    return batch


def _build_replay_iter(
    cfg: RankFMConfig,
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
        max_steps=cfg.max_steps,
        local_batch_size=cfg.local_batch_size,
        num_candidates=cfg.num_candidates,
        ro_target_keys=ro_target_keys,
        nro_target_keys=nro_target_keys,
        unary_target_keys=unary_target_keys,
        ro_per_key_budgets=ro_per_key_budgets,
    )


def _run_training_loop(
    cfg: RankFMConfig,
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
      2. ``cfg.enable_padding`` + ``feature_config`` => fresh random
         (jagged) batch per step; padded at trace time.
      3. otherwise => reuse the closure-captured uniform batch every step.
    """
    jax_profiling = _is_jax_profiling_enabled(cfg)
    profiling_active = False
    if jax_profiling:
        logger.info(
            f"JAX profiling enabled: steps {cfg.profile_start_step}-{cfg.profile_end_step}, "
            f"dir={cfg.profile_dir}"
        )

    model.train(True)
    mode = "replay" if data_pipe is not None else "synthetic"
    logger.info(
        f"Starting training ({mode}): {cfg.max_steps} steps, global_bs={global_bs}"
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
        cfg,
        data_pipe,
        key_to_size,
        ro_target_keys=ro_target_keys,
        nro_target_keys=nro_target_keys,
        unary_target_keys=unary_target_keys,
        ro_per_key_budgets=ro_per_key_budgets,
    )

    for step in range(1, cfg.max_steps + 1):
        if jax_profiling and step == cfg.profile_start_step:
            if not cfg.profile_dir.startswith("gs://"):
                os.makedirs(cfg.profile_dir, exist_ok=True)
            jax.profiler.start_trace(cfg.profile_dir)
            profiling_active = True
            logger.info(f"[PROFILE] Started trace at step {step}")

        next_batch = _next_training_batch(step, cfg, batch, replay_iter, feature_config)
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
    train_step, model, opt, batch, feature_config, key_to_size = (
        _build_compiled_train_step(
            cfg,
            num_ranks=num_ranks,
            num_processes=gloo_world,
            gloo_world=gloo_world,
            global_bs=global_bs,
        )
    )

    # --- 4. Optional GCS data pipe. When set, _run_training_loop iterates
    # real batches from disk (one batch per file per epoch) and clamps KJT
    # IDs into the model's hash space via key_to_size. PaddingRegionTransform
    # handles the dynamic-shape padding at runtime.
    data_pipe: Optional[BatchReplayPipe] = None
    if cfg.gcs_data_path:
        # When --data_file_template is set, generate filenames programmatically
        # and skip directory enumeration (avoids the 58-min ls /mnt/barnfs
        # tax). One file per epoch — the rotating window in
        # replay_batch_iter provides per-step diversity within that file.
        explicit_files: Optional[List[str]] = None
        if cfg.data_file_template:
            explicit_files = [cfg.data_file_template.format(i=1)]
            logger.info(
                f"Direct-read mode: file_list={explicit_files} (template "
                f"'{cfg.data_file_template}')"
            )
        data_pipe = BatchReplayPipe(
            path=cfg.gcs_data_path,
            rank=gloo_rank,
            world_size=gloo_world,
            replicate_across_ranks=cfg.data_replicate_across_ranks,
            file_list=explicit_files,
        )
        logger.info(
            f"Replay pipe: {len(data_pipe)} batches for rank {gloo_rank} "
            f"(replicate={cfg.data_replicate_across_ranks})"
        )
    else:
        logger.info("Using synthetic data (no --gcs_data_path)")

    # Per-key RO budgets for eager pad in data_pipe (matches PRT's per-key
    # padding for ro_id_list_features). nro/unary use a single global formula
    # (B * num_candidates) inside data_pipe; ro is per-key so we compute it
    # here from the same module_config that PRT reads.
    _module_config = model.module_config  # pyre-ignore[16]
    ro_kjt_keys_for_pad = list(batch.ro_id_list_features.keys())
    contextual_budgets_for_pad = dict(_module_config.contextual_feature_to_max_length)
    ro_per_key_budgets: List[int] = [
        cfg.local_batch_size * contextual_budgets_for_pad.get(k, cfg.max_uih_seq_len)
        for k in ro_kjt_keys_for_pad
    ]

    # --- 5. Train ---
    _run_training_loop(
        cfg,
        train_step,
        model,
        opt,
        batch,
        is_main,
        global_bs,
        feature_config=feature_config if cfg.enable_padding else None,
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
