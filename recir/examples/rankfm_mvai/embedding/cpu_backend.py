# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""CPU/torchrec DMP embedding backend for RankFM.

Selected via ``RankFMConfig.embedding_backend == "cpu"`` through the
factory in ``embedding.py``. Tables get TABLE_WISE sharding when small
and ROW_WISE when large; gloo coordinates ranks via JobSet env vars.

Kept in its own module so ``embedding.py`` (the factory) and
``jaxbackend.py`` don't pull in the heavy ``torchrec.distributed.*``
import surface when only the JAX backend is used.
"""

from __future__ import annotations

import logging
import os
from typing import Any, cast, Dict, List, Tuple

import jax
import torch
import torch.distributed as dist
import torch.nn as nn
from minimal_viable_ai.core.model_family_api.model_family import (
    optim_type_to_optim_class,
)
from minimal_viable_ai.core.model_family_api.optimizer_configs import OptimConfig
from recir.sparsecore.backend import (
    EmbeddingBackend,
    EmbeddingSetupResult,
    SparseOptimizerRuntime,
    SparseUpdatePlacement,
)
from recir.tests.xla.xsurface_test_utils import _build_overrides
from torch.distributed.optim import (
    _apply_optimizer_in_backward as apply_optimizer_in_backward,
)
from torchrec.distributed.embedding import EmbeddingCollectionSharder
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.types import ParameterConstraints
from torchrec.distributed.types import ShardingEnv, ShardingType
from torchrec.distributed.utils import optimizer_type_to_emb_opt_type
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.optim.optimizers import in_backward_optimizer_filter

logger: logging.Logger = logging.getLogger(__name__)

# Tables with fewer than this many rows per rank get TABLE_WISE sharding
# (avoids slicing tiny tables too thin for fbgemm's CPU FUSED path).
_ROW_WISE_MIN_ROWS_PER_RANK: int = 100


# Embedding parameters all live under this submodule path in the bare model.
_EMBEDDING_PARAM_PREFIX: str = (
    "recblock.tokenizer_stack.mot_sparse_arch._sparse_arch._embedding_collection"
)


def _attach_single_node_sparse_optimizers(
    model: nn.Module,
    sparse_config: OptimConfig,
) -> None:
    """Attach config-driven in-backward optimizers to bare EC parameters."""
    attached = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or not name.startswith(_EMBEDDING_PARAM_PREFIX):
            continue
        table_name = name.split(".")[-2]
        optimizer_class = optim_type_to_optim_class(sparse_config.optim_type)
        apply_optimizer_in_backward(
            optimizer_class,
            [parameter],
            {
                "_batch_key": table_name,
                "learning_rate": sparse_config.lr,
                "eps": sparse_config.eps,
            },
        )
        attached += 1
    if attached == 0:
        raise RuntimeError("CPU backend found no embedding parameters to optimize.")
    logger.info("Attached in-backward sparse optimizers to %d CPU tables", attached)


class _ECSharder(EmbeddingCollectionSharder):
    def __init__(
        self,
        kernel_type: str,
        fused_params: Dict[str, Any] | None = None,
    ) -> None:
        self._kernel_type = kernel_type
        if fused_params is None:
            fused_params = {}
        fused_params.setdefault("learning_rate", 0.1)
        super().__init__(fused_params=fused_params)

    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [ShardingType.ROW_WISE.value, ShardingType.TABLE_WISE.value]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [self._kernel_type]


class CPURankFMBackend(EmbeddingBackend):
    """Original torchrec DMP ROW_WISE/TABLE_WISE on CPU via gloo."""

    def init_pg(self) -> Tuple[int, int]:
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
        short_host = addr_list[0].split(":")[0]
        # MASTER_ADDR overrides TPU_PROCESS_ADDRESSES for gloo coordinator.
        # GKE's TPU runtime resolves short-form hostnames internally but gloo's
        # TCP rendezvous cannot — use the full FQDN set in the JobSet env.
        coordinator = os.environ.get("MASTER_ADDR", short_host)
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
            f"Gloo PG initialized: rank={dist.get_rank()}, "
            f"world={dist.get_world_size()}"
        )

        return gloo_rank, gloo_world

    def setup(
        self,
        model: nn.Module,
        table_configs: List[EmbeddingConfig],
        sparse_config: OptimConfig,
        device: torch.device,
        world_size: int,
    ) -> EmbeddingSetupResult:
        if device.type != "cpu":
            raise ValueError(f"CPU backend requires a CPU device, got {device}.")
        if dist.is_initialized():
            model = self._setup_dmp(model, table_configs, sparse_config, world_size)
        else:
            _attach_single_node_sparse_optimizers(model, sparse_config)
            logger.info("Single-node: using config-driven in-backward optimizers")

        return EmbeddingSetupResult(
            model=model,
            dense_named_parameters=self.dense_named_parameters(model),
            sparse_runtime=SparseOptimizerRuntime(
                placement=SparseUpdatePlacement.BACKEND_FUSED,
                config=sparse_config,
            ),
        )

    def _setup_dmp(
        self,
        model: nn.Module,
        table_configs: List[EmbeddingConfig],
        config: OptimConfig,
        world_size: int,
    ) -> nn.Module:
        emb_opt_type = optimizer_type_to_emb_opt_type(
            optim_type_to_optim_class(config.optim_type)
        )
        logger.info(
            "CPURankFMBackend: FUSED embedding optimizer=%s lr=%g eps=%g",
            emb_opt_type,
            config.lr,
            config.eps,
        )

        pg = dist.distributed_c10d._get_default_group()
        sharders = [
            _ECSharder(
                kernel_type=EmbeddingComputeKernel.FUSED.value,
                fused_params={
                    "optimizer": emb_opt_type,
                    "learning_rate": config.lr,
                    "eps": config.eps,
                },
            )
        ]

        row_wise_min = world_size * _ROW_WISE_MIN_ROWS_PER_RANK
        constraints: Dict[str, ParameterConstraints] = {}
        for table in table_configs:
            sharding = (
                ShardingType.ROW_WISE.value
                if table.num_embeddings >= row_wise_min
                else ShardingType.TABLE_WISE.value
            )
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
            topology=Topology(world_size=world_size, compute_device="cpu"),
            constraints=constraints,
        )
        plan = planner.collective_plan(model, cast(Any, sharders), pg)
        logger.info("DMP sharding plan:")
        for module_path, module_plan in plan.plan.items():
            for param_name, param_sharding in cast(Any, module_plan).items():
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
            sharders=cast(Any, sharders),
            device=torch.device("cpu"),
            init_data_parallel=False,
        )
        logger.info("CPURankFMBackend: DMP initialized (world=%d)", world_size)
        return dmp.module

    def dense_named_parameters(self, model: nn.Module) -> Dict[str, nn.Parameter]:
        return dict(in_backward_optimizer_filter(model.named_parameters()))

    def get_override(
        self,
        recblock: Any,
        batch_sample: Any,
        device: torch.device,
        global_bs: int,
        num_ranks: int,
        num_processes: int,
    ) -> List[Any]:
        return _build_overrides(
            recblock,
            batch_sample,
            device,
            batch_size=global_bs,
            num_processes=num_processes,
        )
