# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""CPU/torchrec DMP embedding backend.

Tables get TABLE_WISE sharding when small and ROW_WISE when large; gloo
coordinates ranks via JobSet env vars.

Kept in its own module so the factory in ``__init__.py`` and the other
backends don't pull in the heavy ``torchrec.distributed.*`` import surface
when only the SparseCore or in-graph backend is used.

Model-agnostic: the caller supplies ``embedding_param_prefix``, the FQN
prefix under which its embedding parameters live.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, cast, Dict, List, Tuple

import jax
import torch
import torch.distributed as dist
import torch.nn as nn
from minimal_viable_ai.core.model_family_api.model_family import (
    optim_type_to_optim_class,
)
from minimal_viable_ai.core.model_family_api.optimizer_configs import OptimConfig
from recir.eager_dispatch import install_graph_break_dispatch
from recir.embedding._sparse_arch_stub import (
    DummySparseArch,
    sparse_arch_marker_prefix,
    swap_sparse_arch,
)
from recir.embedding.backend import (
    EmbeddingBackend,
    EmbeddingSetupResult,
    SparseOptimizerRuntime,
    SparseUpdatePlacement,
)
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
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor

logger: logging.Logger = logging.getLogger(__name__)

# Tables with fewer than this many rows per rank get TABLE_WISE sharding
# (avoids slicing tiny tables too thin for fbgemm's CPU FUSED path).
_ROW_WISE_MIN_ROWS_PER_RANK: int = 100


def _attach_single_node_sparse_optimizers(
    model: nn.Module,
    sparse_config: OptimConfig,
    embedding_param_prefix: str,
) -> None:
    """Attach config-driven in-backward optimizers to bare EC parameters."""
    attached = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or not name.startswith(embedding_param_prefix):
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


class CPUEmbeddingBackend(EmbeddingBackend):
    """torchrec DMP ROW_WISE/TABLE_WISE on CPU via gloo.

    ``embedding_param_prefix`` is the FQN prefix under which the model keeps
    its embedding collection parameters, used to find the tables to attach
    in-backward optimizers to on the single-node path.
    """

    def __init__(self, embedding_param_prefix: str) -> None:
        self._embedding_param_prefix = embedding_param_prefix

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
        # Defer to an already-created PG (the entrypoint train.py::_init_gloo_pg
        # now owns creation); only self-init for standalone entrypoints that
        # don't. Avoids a double init_process_group.
        if dist.is_initialized():
            logger.info(
                f"Gloo PG already initialized; reusing rank={gloo_rank}, "
                f"world={gloo_world}"
            )
            return gloo_rank, gloo_world
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
            dmp_world_size = dist.get_world_size()
            if dmp_world_size != world_size:
                logger.info(
                    "CPU DMP uses process-group world size %d instead of %d JAX devices",
                    dmp_world_size,
                    world_size,
                )
            model = self._setup_dmp(model, table_configs, sparse_config, dmp_world_size)
        else:
            _attach_single_node_sparse_optimizers(
                model, sparse_config, self._embedding_param_prefix
            )
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
            "CPUEmbeddingBackend: FUSED embedding optimizer=%s lr=%g eps=%g",
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
        logger.info("CPUEmbeddingBackend: DMP initialized (world=%d)", world_size)
        return dmp.module

    def dense_named_parameters(self, model: nn.Module) -> Dict[str, nn.Parameter]:
        return dict(in_backward_optimizer_filter(model.named_parameters()))


class SparseArchDispatcher:
    """Graph-break dispatcher back to the real eager torchrec sparse arch.

    ``install_eager`` swaps the arch located by ``sparse_arch_getter`` for a
    :class:`DummySparseArch` so tracing sees stable shapes (the dummy plants
    input + forward/backward markers). ``install_graph`` finds those markers
    and installs a graph-break dispatch back to the captured real arch.
    """

    def __init__(
        self,
        cached_output: Dict[str, JaggedTensor],
        input_kjt_keys: List[str],
        sparse_arch_getter: Callable[[nn.Module], nn.Module],
        override_id: int = 1,
        num_ranks: int = 1,
    ) -> None:
        self._cached_output = cached_output
        self._input_kjt_keys = input_kjt_keys
        self._sparse_arch_getter = sparse_arch_getter
        self._override_id = override_id
        self._num_ranks = num_ranks
        self._orig_sparse_arch: nn.Module | None = None
        self._dummy: DummySparseArch | None = None

    def install_eager(self, model: nn.Module) -> None:
        target = self._sparse_arch_getter(model)
        self._orig_sparse_arch = target
        self._dummy = DummySparseArch(
            self._cached_output,
            input_kjt_keys=self._input_kjt_keys,
            override_id=self._override_id,
        )
        swap_sparse_arch(model, target, self._dummy)

    def install_graph(
        self,
        gm: torch.fx.GraphModule,
        flat_state: List[torch.Tensor],
    ) -> Tuple[torch.fx.GraphModule, List[torch.Tensor]]:
        if self._orig_sparse_arch is None or self._dummy is None:
            return gm, flat_state

        real_arch: nn.Module = self._orig_sparse_arch
        input_kjt_keys: List[str] = self._input_kjt_keys
        feature_names: List[str] = self._dummy._feature_names
        num_ranks: int = self._num_ranks

        def fwd_fn(*input_tensors: torch.Tensor) -> Tuple[torch.Tensor, ...]:
            # input_tensors == (values, lengths) thanks to mark_input markers
            # in DummySparseArch.forward (kept alive through trace by the
            # zero-valued anchor that depends on them).
            values = input_tensors[0]
            lengths = input_tensors[1]
            total = int(lengths.sum().item())
            if values.shape[0] < total:
                pad_size = total - values.shape[0]
                values = torch.nn.functional.pad(
                    values, [0] * (2 * values.ndim - 1) + [pad_size]
                )
            elif values.shape[0] > total:
                values = values.narrow(0, 0, total)
            kjt = KeyedJaggedTensor(
                keys=input_kjt_keys,
                values=values,
                lengths=lengths,
            )
            with torch.enable_grad():
                output = real_arch(kjt)

            flat: List[torch.Tensor] = []
            for name in feature_names:
                jt = output[name]
                flat.append(jt.values())
                flat.append(jt.lengths())
            # Return raw (grad-preserving) tensors; the install helper wraps
            # this callable with save-for-backward + detach.
            return tuple(flat)

        install_graph_break_dispatch(
            gm,
            sparse_arch_marker_prefix(self._override_id),
            fwd_fn,
            num_state_tensors=len(flat_state),
            num_ranks=num_ranks,
            disable_torchrec_gradient_division=True,
        )
        return gm, flat_state
