# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""CPU/torchrec DMP embedding backend.

Tables production places ROW_WISE or TABLE_WISE are sharded under DMP and
trained by a FUSED TBE inside backward. Tables it places DATA_PARALLEL are held
out of the sharded collection entirely (see ``split_collection``) so they keep a
real ``.grad`` and are trained by the DENSE optimizer, which is what
DATA_PARALLEL amounts to at world size 1 and the only form of it a CPU leg can
run. Among the shardable tables, row count picks ROW_WISE over TABLE_WISE.
Gloo coordinates ranks via JobSet env vars.

Kept in its own module so the factory in ``__init__.py`` and the other
backends don't pull in the heavy ``torchrec.distributed.*`` import surface
when only the SparseCore or in-graph backend is used.

Model-agnostic: the caller supplies ``embedding_collection_fqn``, the FQN of
the model's torchrec ``EmbeddingCollection``.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any, cast, Dict, List, Optional, Sequence, Tuple, Type

import jax
import torch
import torch.distributed as dist
import torch.nn as nn
import torchrec
from recir.embedding._sparse_arch_stub import DummySparseArch, sparse_arch_marker_prefix
from recir.embedding._torchrec_patches import apply_seq_a2a_int64_patch
from recir.embedding.backend import (
    EmbeddingBackend,
    EmbeddingSetupResult,
    SparseOptimizerRuntime,
    SparseUpdatePlacement,
    swap_submodule,
)
from recir.embedding.split_collection import (
    assert_replicated_half_is_unplanned,
    partition_by_sharding,
    split_embedding_collection,
    SplitEmbeddingCollection,
)
from recir.ops.eager_dispatch import install_graph_break_dispatch
from recir.training.optim import OptimConfig
from torchrec.distributed.embedding import EmbeddingCollectionSharder
from torchrec.distributed.embedding_types import EmbeddingComputeKernel, OptimType
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.types import ParameterConstraints
from torchrec.distributed.types import ShardingEnv, ShardingType
from torchrec.distributed.utils import optimizer_type_to_emb_opt_type
from torchrec.modules.embedding_configs import BaseEmbeddingConfig
from torchrec.optim.optimizers import in_backward_optimizer_filter
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor

logger: logging.Logger = logging.getLogger(__name__)


# Map TorchRec optimizer identifiers to their concrete implementations.
def optim_type_to_optim_class(optim_type: OptimType) -> Type[torch.optim.Optimizer]:
    # Unforunately not all fbgemm optimizers have native pytorch support, fallback if necessary
    # native pytorch optimizers may not accept kwargs like "max_gradient"
    if optim_type == OptimType.SGD:
        return torchrec.optim.SGD
    elif optim_type == OptimType.LARS_SGD:
        return torchrec.optim.LarsSGD
    elif optim_type == OptimType.LAMB:
        return torchrec.optim.LAMB
    elif optim_type == OptimType.PARTIAL_ROWWISE_LAMB:
        return torchrec.optim.PartialRowWiseLAMB
    elif optim_type == OptimType.ADAM:
        return torchrec.optim.Adam
    elif optim_type == OptimType.PARTIAL_ROWWISE_ADAM:
        return torchrec.optim.PartialRowWiseAdam
    elif optim_type == OptimType.ADAGRAD:
        return torchrec.optim.Adagrad
    elif optim_type == OptimType.ROWWISE_ADAGRAD:
        return torchrec.optim.RowWiseAdagrad
    raise ValueError(f"Invalid OptimType {optim_type}")


# Tables with fewer than this many rows per rank get TABLE_WISE sharding
# (avoids slicing tiny tables too thin for fbgemm's CPU FUSED path).
_ROW_WISE_MIN_ROWS_PER_RANK: int = 100


def _ensure_process_group() -> None:
    """Make sure DMP has a process group, creating a single-rank one if not.

    RecTrainer always initializes gloo on CPU, so a cluster run arrives with
    one. Local callers -- benchmarks, tests -- often do not, and the branch
    that used to serve them ran different optimizer code than the cluster.
    """
    if dist.is_initialized():
        return
    store = os.path.join(tempfile.mkdtemp(prefix="recir_cpu_pg_"), "store")
    dist.init_process_group(
        backend="gloo", init_method=f"file://{store}", world_size=1, rank=0
    )
    logger.info("CPU backend started a single-rank gloo process group")


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

    ``embedding_collection_fqn`` is the FQN of the model's torchrec
    ``EmbeddingCollection``.
    """

    def __init__(
        self,
        embedding_collection_fqn: str,
        embedding_tables: Optional[Dict[str, Any]] = None,
    ) -> None:
        # int32 sequence-embedding offsets overflow in torchrec's all-to-all at
        # large multi-host batch; see _torchrec_patches.
        apply_seq_a2a_int64_patch()
        self._embedding_collection_fqn = embedding_collection_fqn
        # The model's own table objects, read here ONLY for their ``sharding``
        # field -- torchrec's ``EmbeddingConfig`` has no such field, so the
        # per-table ROW_WISE/DATA_PARALLEL choice cannot ride along with
        # ``table_configs``. Taken at construction rather than through the
        # shared ``setup`` because only this backend can act on it: the
        # placement describes a torchrec DMP layout, which the in-graph and
        # SparseCore backends do not build.
        self._embedding_tables: Optional[Dict[str, Any]] = embedding_tables

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
        table_configs: Sequence[BaseEmbeddingConfig],
        sparse_config: OptimConfig,
        device: torch.device,
        world_size: int,
    ) -> EmbeddingSetupResult:
        if device.type != "cpu":
            raise ValueError(f"CPU backend requires a CPU device, got {device}.")
        # One path, always DMP. Without a process group this used to attach
        # in-backward optimizers instead -- a different optimizer, with no LR
        # handle to drive the schedule from, and one that swallowed the
        # configured rate through **unused and ran at torchrec's 1e-2 default.
        # A single-rank group costs nothing and keeps every caller on the code
        # the cluster runs.
        _ensure_process_group()
        dmp_world_size = dist.get_world_size()
        if dmp_world_size != world_size:
            logger.info(
                "CPU DMP uses process-group world size %d instead of %d JAX devices",
                dmp_world_size,
                world_size,
            )
        dmp = self._setup_dmp(model, table_configs, sparse_config, dmp_world_size)
        model = dmp.module

        return EmbeddingSetupResult(
            model=model,
            dense_named_parameters=self.dense_named_parameters(model),
            sparse_runtime=SparseOptimizerRuntime(
                placement=SparseUpdatePlacement.BACKEND_FUSED,
                config=sparse_config,
                fused_optimizer=dmp.fused_optimizer,
            ),
        )

    def _split_out_replicated_tables(
        self, model: nn.Module
    ) -> Tuple[Optional[SplitEmbeddingCollection], List[str]]:
        """Move prod's DATA_PARALLEL tables out of the collection the planner sees.

        Returns the new split module (``None`` when no split was needed) and the
        names of the tables that remain shardable.

        Without ``embedding_tables`` there is nothing to read a placement from --
        torchrec's ``EmbeddingConfig`` carries no ``sharding`` field -- so the
        collection is left whole and every table stays on the FUSED/sparse path.
        That is the pre-existing behaviour, and it is WRONG for any model whose
        config asks for DATA_PARALLEL: those tables get RowWiseAdagrad at the
        sparse LR instead of the dense optimizer's. Callers that have the tables
        should always pass them.
        """
        embedding_tables = self._embedding_tables
        if embedding_tables is None:
            logger.warning(
                "CPUEmbeddingBackend: no embedding_tables passed, so per-table "
                "placement is unknown and every table will be sharded onto the "
                "FUSED/sparse-optimizer path. Tables that production marks "
                "DATA_PARALLEL will be trained by the wrong optimizer."
            )
            return None, []

        sharded, replicated = partition_by_sharding(embedding_tables)
        if not replicated:
            return None, sharded

        parent_path, _, attr = self._embedding_collection_fqn.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        return split_embedding_collection(parent, attr, replicated), sharded

    def _setup_dmp(
        self,
        model: nn.Module,
        table_configs: Sequence[BaseEmbeddingConfig],
        config: OptimConfig,
        world_size: int,
    ) -> DistributedModelParallel:
        emb_opt_type = optimizer_type_to_emb_opt_type(
            optim_type_to_optim_class(config.optim_type)
        )
        logger.info(
            "CPUEmbeddingBackend: FUSED embedding optimizer=%s lr=%g eps=%g",
            emb_opt_type,
            config.lr,
            config.eps,
        )
        split, shardable = self._split_out_replicated_tables(model)

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
        keep = set(shardable) if split is not None else None
        constraints: Dict[str, ParameterConstraints] = {}
        for table in table_configs:
            if keep is not None and table.name not in keep:
                continue
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
        if split is not None:
            assert_replicated_half_is_unplanned(
                plan, split, self._embedding_collection_fqn
            )
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
        # The DMP itself, not dmp.module: setup() needs dmp.fused_optimizer to
        # drive the sparse LR schedule. The EmbeddingFusedOptimizer objects
        # survive inside the sharded EC either way, but the CombinedOptimizer
        # that gathers them is assembled in DMP.__init__ and is lost with the
        # wrapper. Rebuilding it would mean reimplementing torchrec's
        # _fused_optim_impl, including its module_id_cache dedupe.
        return dmp

    def dense_named_parameters(self, model: nn.Module) -> Dict[str, nn.Parameter]:
        return dict(in_backward_optimizer_filter(model.named_parameters()))


class SparseArchDispatcher:
    """Graph-break dispatcher back to the real eager torchrec lookup.

    ``install_eager`` swaps the ``EmbeddingCollection`` at
    ``embedding_collection_fqn`` for a :class:`DummySparseArch` so tracing sees
    stable shapes (the dummy plants input + forward/backward markers).
    ``install_graph`` finds those markers and installs a graph-break dispatch back
    to the captured real collection.
    """

    def __init__(
        self,
        cached_output: Dict[str, JaggedTensor],
        input_kjt_keys: List[str],
        embedding_collection_fqn: str,
        override_id: int = 1,
        num_ranks: int = 1,
    ) -> None:
        self._cached_output = cached_output
        self._input_kjt_keys = input_kjt_keys
        self._embedding_collection_fqn = embedding_collection_fqn
        self._override_id = override_id
        self._num_ranks = num_ranks
        self._orig_collection: nn.Module | None = None
        self._dummy: DummySparseArch | None = None

    def install_eager(self, model: nn.Module) -> None:
        self._dummy = DummySparseArch(
            self._cached_output,
            input_kjt_keys=self._input_kjt_keys,
            override_id=self._override_id,
        )
        self._orig_collection = swap_submodule(
            model, self._embedding_collection_fqn, self._dummy
        )

    def install_graph(
        self,
        gm: torch.fx.GraphModule,
        flat_state: List[torch.Tensor],
    ) -> Tuple[torch.fx.GraphModule, List[torch.Tensor]]:
        if self._orig_collection is None or self._dummy is None:
            return gm, flat_state

        real_arch: nn.Module = self._orig_collection
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
