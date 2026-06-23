# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Set

import torch
import torch.distributed as dist
import torch.nn as nn
from recir.backend import BackendType
from recir.distributed.data_parallel import (
    DataParallelStyle,
    OptOrOpts,
    partition_data_parallel,
    shard_params_and_states,
    shard_params_by_plan,
)
from recir.lowering.xla_lower import (
    _AOT_TOPOLOGY,
    _get_global_devices,
    _get_local_devices,
)
from recir.sharding_plan import EmbeddingShardingPlan
from torch.distributed.tensor import DeviceMesh
from torch.fx import GraphModule

logger: logging.Logger = logging.getLogger(__name__)


def _create_fake_mesh(
    device_type: str,
    size: int,
    group_name: str,
) -> DeviceMesh:
    """Create a DeviceMesh backed by a FakeProcessGroup.

    Used for XLA backends where process coordination happens at runtime.
    """
    from torch._C._distributed_c10d import FakeProcessGroup

    fake_pg = FakeProcessGroup._create_internal(0, size)
    mesh = DeviceMesh(
        device_type,
        torch.arange(size),
        _init_backend=False,
        _rank=0,
    )
    mesh._dim_group_names = [group_name]
    # DeviceMesh only creates _pg_registry on some init paths / torch versions
    # (notably not with _init_backend=False); create it if absent so the
    # FakeProcessGroup registration below works across versions.
    if getattr(mesh, "_pg_registry", None) is None:
        mesh._pg_registry = {}
    mesh._pg_registry[group_name] = fake_pg  # pyre-ignore[6]
    return mesh


class ParallelMode(ABC):
    """
    Basic Parallel Mode interface. Each parallelism pattern should implement
    this interface to describe how to partition and compile the graph in the
    spmd compiler.
    """

    input_batch_dim: int

    @property
    @abstractmethod
    def num_ranks(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def num_processes(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def partition(
        self,
        gm: GraphModule,
        *,
        params: Dict[str, nn.Parameter],
        buffers: Dict[str, torch.Tensor],
        named_states: Dict[str, Dict[str, Any]],
    ) -> GraphModule:
        """
        Partition a single device graph to a distributed graph.

        Inputs are FQN-keyed dicts so callers without a live ``nn.Module``
        (e.g. a bundle-runtime consumer that loads safetensors at MAST start)
        can drive partitioning directly.  ``named_states`` maps each parameter
        FQN to its optimizer state dict; ``step`` may be an int scalar, a
        tensor, or ``None`` depending on the optimizer flavor.
        """
        raise NotImplementedError

    def partition_from_module(
        self,
        gm: GraphModule,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        params_and_buffers: Dict[str, Any],
        named_states: Dict[str, Any],
    ) -> GraphModule:
        """Instance-method adapter for callers still holding a live ``nn.Module`` + ``Optimizer``.

        A legacy caller of ``mode.partition(gm, model, opt, params_and_buffers, named_states, args, kwargs)``
        migrates by renaming ``.partition`` to ``.partition_from_module`` and
        dropping the trailing positional ``args`` / ``kwargs`` (the example
        inputs were never read by any partition implementation).  Splits
        ``params_and_buffers`` into ``params`` / ``buffers`` via an
        ``isinstance(v, nn.Parameter)`` check and forwards to the kwargs-only
        ``partition`` method.  ``model`` and ``optimizer`` are accepted for
        API back-compat but are not used (they were unused even before the
        kwargs-only refactor).
        """
        params: Dict[str, nn.Parameter] = {
            k: v for k, v in params_and_buffers.items() if isinstance(v, nn.Parameter)
        }
        buffers: Dict[str, torch.Tensor] = {
            k: v
            for k, v in params_and_buffers.items()
            if not isinstance(v, nn.Parameter)
        }
        return self.partition(
            gm,
            params=params,
            buffers=buffers,
            named_states=named_states,
        )

    @abstractmethod
    def shard_params(
        self,
        model: torch.nn.Module,
        params_and_buffers: Dict[str, Any],
        opt: OptOrOpts,
    ) -> Set[str]:
        """
        Shard model parameters and optimizer states according to the
        parallel mode's strategy.  Updates ``opt.state`` and
        ``opt.param_groups`` in-place for every optimizer passed in so
        that parameter -> state mappings AND param_groups['params']
        lists remain valid after the Parameter identity swap that
        sharding performs.

        ``opt`` accepts either a single Optimizer (backward compat),
        a list of Optimizers (multi-opt sparse + dense partition),
        or None (no optimizer).

        Returns:
            Set of keys in ``params_and_buffers`` that were actually sharded.
        """
        raise NotImplementedError

    @abstractmethod
    def transform_and_compile(self, gm: GraphModule) -> GraphModule:
        """
        Transform and compile a distributed graph with a set of graph
        transformation and optimization passes for each parallel mode.

        The returned result should be a compiled executable graph in
        the distributed environment.
        """
        # TODO: add more necessary arguments to this interface.
        raise NotImplementedError


class GSPMDParallel(ParallelMode):
    """Parallel mode that delegates partitioning to XLA GSPMD.

    Instead of using DTensor to insert explicit collective ops, this mode
    skips partitioning entirely and attaches sharding metadata to the graph.
    The actual partitioning happens during XLA compilation via GSPMD
    auto-parallelism (jax.jit with in_shardings/out_shardings).

    Args:
        parallel_style: "replicate" (DDP) or "fully_shard" (FSDP).
        input_batch_dim: which dimension of input tensors is the batch dim.
        loss_reduction: reduction type used by the loss function.
             "sum" (default): the loss computes a sum over elements.
             "mean": the loss computes a mean over elements.
        num_ranks: Total number of devices for parallelism. If None,
            auto-detected from jax.device_count() or dist.get_world_size().
        num_processes: Number of OS processes. If None, auto-detected from
            dist.get_world_size() (or 1 if dist is not initialized).
        fsdp_table_substr: If set, row-wise (FSDP) shard every param whose name
            contains this substring, regardless of ``parallel_style`` -- e.g.
            table-only sharding of the in-graph seq-embedding ``_sharded_tables`` on
            an otherwise-"replicate" model. A no-op when no param matches, so
            plain "replicate" stays byte-identical.

            TODO: unify this substr-based table sharding with the
            EmbeddingShardingPlan used in hybrid parallel, and refactor
            parallel_style (this is a stopgap for table-only FSDP).
    """

    def __init__(
        self,
        parallel_style: str = "replicate",
        *,
        input_batch_dim: int = 0,
        loss_reduction: str = "sum",
        num_ranks: Optional[int] = None,
        num_processes: Optional[int] = None,
        fsdp_table_substr: Optional[str] = None,
    ) -> None:
        self.parallel_style = parallel_style
        self.input_batch_dim = input_batch_dim
        self.grad_reduction = "sum" if loss_reduction == "mean" else "avg"
        self._num_ranks = num_ranks
        self._num_processes = num_processes
        self._fsdp_table_substr = fsdp_table_substr
        self._process_mesh: Optional[DeviceMesh] = None

    @property
    def num_ranks(self) -> int:
        if self._num_ranks is not None:
            return self._num_ranks
        import jax  # @manual

        return jax.device_count()

    @property
    def num_processes(self) -> int:
        if self._num_processes is not None:
            return self._num_processes
        # Honor :func:`recir.lowering.xla_lower.aot_topology`: when a
        # virtual multi-host topology is installed (e.g., a CPU prebuild
        # targeting a TPU pod), `jax.process_count()` still reports the
        # host's process count (=1 on the prebuild box) while the
        # topology models the deployment with N processes. The trace-
        # time `_make_batch_transform` (which uses this property) must
        # match the runtime `_to_sharded_jax` local→global multiplier,
        # so use the topology-aware ratio when one is active.
        if _AOT_TOPOLOGY.get() is not None:
            return len(_get_global_devices()) // max(1, len(_get_local_devices()))
        import jax  # @manual

        return jax.process_count()

    @property
    def process_mesh(self) -> DeviceMesh:
        """Mesh sized to ``num_processes`` for physical model/state sharding.

        Uses a FakeProcessGroup since XLA handles process coordination
        at runtime.
        """
        if self._process_mesh is None:
            self._process_mesh = _create_fake_mesh(
                "cpu", self.num_processes, "fake_gspmd_process_mesh"
            )
        return self._process_mesh

    def shard_params(
        self,
        model: torch.nn.Module,
        params_and_buffers: Dict[str, Any],
        opt: OptOrOpts,
    ) -> Set[str]:
        sharded: Set[str] = set()

        # Table-only FSDP sharding: when ``fsdp_table_substr`` is set, row-wise
        # (FSDP) shard params whose name contains it, regardless of
        # ``parallel_style`` -- table-only sharding (e.g. the in-graph
        # seq-embedding ``_sharded_tables``) while every other weight follows the
        # base style. If no param matches, this is a no-op, so plain "replicate"
        # stays byte-identical to the no-substr path.
        if self._fsdp_table_substr is not None:
            substr: str = self._fsdp_table_substr

            def _is_table(key: str, t: torch.Tensor) -> bool:
                return (
                    substr in key
                    and t.ndim >= 1
                    and t.shape[0] >= self.num_ranks
                    and t.shape[0] % self.num_ranks == 0
                    and t.shape[0] % self.num_processes == 0
                )

            if any(_is_table(k, t) for k, t in params_and_buffers.items()):
                sharded = shard_params_and_states(
                    model,
                    params_and_buffers,
                    opt,
                    self.process_mesh,
                    DataParallelStyle.FULLY_SHARD,
                    shard_filter=_is_table,
                )

        if self.parallel_style != "fully_shard":
            return sharded

        # Only shard tensors where the local shard can be evenly split
        # across local devices. This ensures _to_sharded_jax and
        # _jit_gspmd agree on which tensors are FSDP-sharded. Skip params already
        # sharded by the fsdp_table_substr pass above to avoid double-sharding.
        def _can_fsdp(key: str, t: torch.Tensor) -> bool:
            return (
                key not in sharded
                and t.ndim >= 1
                and t.shape[0] >= self.num_ranks
                and t.shape[0] % self.num_ranks == 0
                and t.shape[0] % self.num_processes == 0
            )

        return sharded | shard_params_and_states(
            model,
            params_and_buffers,
            opt,
            self.process_mesh,
            DataParallelStyle.FULLY_SHARD,
            shard_filter=_can_fsdp,
        )

    def partition(
        self,
        gm: GraphModule,
        *,
        params: Dict[str, nn.Parameter],
        buffers: Dict[str, torch.Tensor],
        named_states: Dict[str, Dict[str, Any]],
    ) -> GraphModule:
        # No DTensor expansion. Return the graph as-is.
        return gm

    def transform_and_compile(self, gm: GraphModule) -> GraphModule:
        raise NotImplementedError(
            "GSPMDParallel uses gm_transformation, not transform_and_compile"
        )


class DataParallel(ParallelMode):
    """Data Parallelism mode.

    Args:
        parallel_style (str): parallel style to use. Currently supports
            "replicate", "fully_shard", and "default".

    Keyword args:
        input_batch_dim (int): the batch dimension of the input tensor.
             default: 0
        loss_reduction (str): reduction type used by the loss function.
             "sum" (default): the loss computes a sum over elements.
             "mean": the loss computes a mean over elements.
        num_ranks: Total number of devices for parallelism. If None,
            auto-detected from jax.device_count() or dist.get_world_size().
        num_processes: Number of OS processes. If None, auto-detected from
            dist.get_world_size() (or 1 if dist is not initialized).
        device_type: Device type for the DeviceMesh used in DTensor
            partitioning. Default: "cuda". Use "cpu" for single-process
            mode where CUDA process groups are not available.
        backend: Optional compilation backend. When set to an XLA backend
            (XLA_SHARD_MAP, XLA_GSPMD, XLA_DYNAMO_BRIDGE), a
            FakeProcessGroup-backed DeviceMesh is used because XLA handles
            collectives at runtime. Otherwise a real DeviceMesh backed by
            real process groups is created (requires dist to be initialized).
        custom_passes (Callable[[GraphModule], GraphModule], optional):
            A custom callable that overrides the default graph transformation
            and optimization passes.
    """

    parallel_style: DataParallelStyle

    def __init__(
        self,
        parallel_style: str = "replicate",
        *,
        input_batch_dim: int = 0,
        loss_reduction: str = "sum",
        num_ranks: Optional[int] = None,
        num_processes: Optional[int] = None,
        device_type: str = "cuda",
        backend: Optional[BackendType] = None,
        custom_passes: Optional[Callable[[GraphModule], GraphModule]] = None,
    ) -> None:
        if parallel_style == "replicate":
            self.parallel_style = DataParallelStyle.REPLICATE
        elif parallel_style == "fully_shard":
            self.parallel_style = DataParallelStyle.FULLY_SHARD
        elif parallel_style == "default":
            self.parallel_style = DataParallelStyle.DEFAULT
        else:
            raise RuntimeError(f"Unknown parallel style: {parallel_style}")

        # TODO: what if user passes in a incorrect `input_batch_dim`, how should we
        # detect that and do proper error handling?
        self.input_batch_dim = input_batch_dim
        # Infer the gradient collective reduction from the loss reduction:
        # sum loss -> avg gradient reduction (standard DDP semantics)
        # mean loss -> sum gradient reduction (the traced backward already
        #   normalises by global numel; summing avoids double-dividing)
        self.grad_reduction = "sum" if loss_reduction == "mean" else "avg"
        self._num_ranks = num_ranks
        self._num_processes = num_processes
        self._device_type = device_type
        self._backend = backend

        if custom_passes is not None:
            self._gm_passes: Callable[[GraphModule], GraphModule] = custom_passes
        else:
            # TODO: add a few default passes here.
            self._gm_passes = lambda gm: gm

        self._process_mesh: Optional[DeviceMesh] = None
        self._device_mesh: Optional[DeviceMesh] = None

    @property
    def num_ranks(self) -> int:
        if self._num_ranks is not None:
            return self._num_ranks
        if self._is_xla_backend:
            import jax  # @manual

            return jax.device_count()
        return dist.get_world_size() if dist.is_initialized() else 1

    @property
    def num_processes(self) -> int:
        if self._num_processes is not None:
            return self._num_processes
        if self._is_xla_backend:
            import jax  # @manual

            return jax.process_count()
        return dist.get_world_size() if dist.is_initialized() else 1

    @property
    def _is_xla_backend(self) -> bool:
        return self._backend is not None and self._backend in (
            BackendType.XLA_SHARD_MAP,
            BackendType.XLA_GSPMD,
            BackendType.XLA_DYNAMO_BRIDGE,
        )

    @property
    def process_mesh(self) -> DeviceMesh:
        """Mesh sized to ``num_processes`` for physical model/state sharding.

        Each position maps 1-to-1 to an OS process.  Used by
        ``shard_params`` to decide which slice of the model each process
        holds.  For XLA backends, uses a FakeProcessGroup since XLA
        handles process coordination at runtime.
        """
        if self._process_mesh is None:
            if self._is_xla_backend:
                self._process_mesh = _create_fake_mesh(
                    self._device_type,
                    self.num_processes,
                    "fake_recir_process_mesh",
                )
            else:
                self._process_mesh = DeviceMesh(
                    self._device_type, torch.arange(self.num_processes)
                )
        return self._process_mesh

    @property
    def device_mesh(self) -> DeviceMesh:
        """Mesh sized to ``num_ranks`` for distributed graph generation.

        Each position maps to a logical device/rank.  Used by
        ``partition`` so DTensor inserts the correct N-way collectives.
        For the XLA path this may be larger than ``process_mesh``
        (``nranks > nproc``); XLA handles the per-process → per-device
        data routing at runtime.
        """
        if self._device_mesh is None:
            if self._is_xla_backend:
                self._device_mesh = _create_fake_mesh(
                    self._device_type,
                    self.num_ranks,
                    "fake_recir_device_mesh",
                )
            else:
                self._device_mesh = DeviceMesh(
                    self._device_type, torch.arange(self.num_ranks)
                )
        return self._device_mesh

    def shard_params(
        self,
        model: torch.nn.Module,
        params_and_buffers: Dict[str, Any],
        opt: OptOrOpts,
    ) -> Set[str]:
        return shard_params_and_states(
            model,
            params_and_buffers,
            opt,
            self.process_mesh,
            self.parallel_style,
        )

    def partition(
        self,
        gm: GraphModule,
        *,
        params: Dict[str, nn.Parameter],
        buffers: Dict[str, torch.Tensor],
        named_states: Dict[str, Dict[str, Any]],
    ) -> GraphModule:
        gm = partition_data_parallel(
            gm,
            params=params,
            buffers=buffers,
            named_states=named_states,
            mesh=self.device_mesh,
            parallel_style=self.parallel_style,
            input_batch_dim=self.input_batch_dim,
            grad_reduction=self.grad_reduction,
        )
        return gm

    def transform_and_compile(self, gm: GraphModule) -> GraphModule:
        """optimize a distributed graph with a set of optimization passes"""
        # TODO: add more necessary arguments to this interface.
        return self._gm_passes(gm)


class HybridParallel(ParallelMode):
    """Hybrid parallel mode: data parallelism for dense params + model
    parallelism for embedding params.

    Dense parameters use FSDP or DDP (controlled by ``dense_style``).
    Embedding parameters are sharded according to ``embedding_plan``
    (e.g., row-wise sharding splits the vocabulary across ranks).

    Example::

        from recir.sharding_plan import EmbeddingShardingPlan, EmbeddingShardingType

        plan = EmbeddingShardingPlan(sharding_map={
            "emb.weight": EmbeddingShardingType.ROW_WISE,
        })
        mode = HybridParallel(
            dense_style="fully_shard",
            embedding_plan=plan,
        )

    The ``partition()`` method orchestrates two passes:
      1. ``partition_data_parallel`` for dense params (with embedding params
         marked as EMBEDDING_PARAM so the partitioner skips allgather for them).
      2. ``partition_embeddings`` for embedding params (TODO: Component 3).
    """

    def __init__(
        self,
        dense_style: str = "fully_shard",
        *,
        embedding_plan: Optional[EmbeddingShardingPlan] = None,
        input_batch_dim: int = 0,
        custom_passes: Optional[Callable[[GraphModule], GraphModule]] = None,
    ) -> None:
        self.dense_style: DataParallelStyle
        if dense_style == "replicate":
            self.dense_style = DataParallelStyle.REPLICATE
        elif dense_style == "fully_shard":
            self.dense_style = DataParallelStyle.FULLY_SHARD
        elif dense_style == "default":
            self.dense_style = DataParallelStyle.DEFAULT
        else:
            raise RuntimeError(f"Unknown dense parallel style: {dense_style}")

        self.embedding_plan: EmbeddingShardingPlan = (
            embedding_plan or EmbeddingShardingPlan()
        )
        self.input_batch_dim = input_batch_dim

        if custom_passes is not None:
            self._gm_passes: Callable[[GraphModule], GraphModule] = custom_passes
        else:
            self._gm_passes = lambda gm: gm

        self._mesh: Optional[DeviceMesh] = None

    @property
    def num_ranks(self) -> int:
        return dist.get_world_size() if dist.is_initialized() else 1

    @property
    def num_processes(self) -> int:
        return dist.get_world_size() if dist.is_initialized() else 1

    @property
    def mesh(self) -> DeviceMesh:
        if self._mesh is None:
            self._mesh = DeviceMesh("cuda", torch.arange(dist.get_world_size()))
        return self._mesh

    def _compute_embedding_param_indices(
        self, params_and_buffers: Dict[str, Any]
    ) -> Optional[Set[int]]:
        """Map embedding param FQNs to their placeholder indices in the graph."""
        if not self.embedding_plan.param_fqns:
            return None
        indices: Set[int] = set()
        for idx, key in enumerate(params_and_buffers):
            if self.embedding_plan.is_embedding_param(key):
                indices.add(idx)
        return indices if indices else None

    def shard_params(
        self,
        model: torch.nn.Module,
        params_and_buffers: Dict[str, Any],
        opt: OptOrOpts,
    ) -> Set[str]:
        return shard_params_by_plan(
            model,
            params_and_buffers,
            opt,
            self.mesh,
            self.dense_style,
            self.embedding_plan.param_fqns,
        )

    def partition(
        self,
        gm: GraphModule,
        *,
        params: Dict[str, nn.Parameter],
        buffers: Dict[str, torch.Tensor],
        named_states: Dict[str, Dict[str, Any]],
    ) -> GraphModule:
        # Embedding indices are computed against the same FQN ordering used
        # by partition_data_parallel: params first, then buffers.
        params_and_buffers: Dict[str, Any] = {**params, **buffers}
        embedding_param_indices = self._compute_embedding_param_indices(
            params_and_buffers
        )

        gm = partition_data_parallel(
            gm,
            params=params,
            buffers=buffers,
            named_states=named_states,
            mesh=self.mesh,
            parallel_style=self.dense_style,
            input_batch_dim=self.input_batch_dim,
            embedding_param_indices=embedding_param_indices,
        )

        return gm

    def transform_and_compile(self, gm: GraphModule) -> GraphModule:
        return self._gm_passes(gm)
