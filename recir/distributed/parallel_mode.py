# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.distributed as dist
from recir.backend import BackendType
from recir.distributed.data_parallel import (
    DataParallelStyle,
    partition_data_parallel,
    shard_params_and_states,
)
from torch.distributed.tensor import DeviceMesh
from torch.fx import GraphModule


class ParallelMode(ABC):
    """
    Basic Parallel Mode interface. Each parallelism pattern should implement
    this interface to describe how to partition and compile the graph in the
    spmd compiler.
    """

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
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        params_and_buffers: Dict[str, Any],
        named_states: Dict[str, Any],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> GraphModule:
        """
        Partition a single device graph to a distributed graph.

        TODO(@wanchaol): some of these arguments are not necessary for
        partitioning, remove the unnecessary ones later.
        """
        raise NotImplementedError

    @abstractmethod
    def shard_params(
        self,
        model: torch.nn.Module,
        params_and_buffers: Dict[str, Any],
        named_states: Dict[str, Any],
    ) -> None:
        """
        Shard model parameters and optimizer states according to the
        parallel mode's strategy.
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
    """

    def __init__(
        self,
        parallel_style: str = "replicate",
        *,
        input_batch_dim: int = 0,
        loss_reduction: str = "sum",
        num_ranks: Optional[int] = None,
        num_processes: Optional[int] = None,
    ) -> None:
        self.parallel_style = parallel_style
        self.input_batch_dim = input_batch_dim
        self.grad_reduction = "sum" if loss_reduction == "mean" else "avg"
        self._num_ranks = num_ranks
        self._num_processes = num_processes

    @property
    def num_ranks(self) -> int:
        if self._num_ranks is not None:
            return self._num_ranks
        import jax  # @manual  # pyre-ignore[21]

        return jax.device_count()

    @property
    def num_processes(self) -> int:
        if self._num_processes is not None:
            return self._num_processes
        import jax  # @manual  # pyre-ignore[21]

        return jax.process_count()

    def shard_params(
        self,
        model: torch.nn.Module,
        params_and_buffers: Dict[str, Any],
        named_states: Dict[str, Any],
    ) -> None:
        pass  # No-op: GSPMD handles sharding via annotations

    def partition(
        self,
        gm: GraphModule,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        params_and_buffers: Dict[str, Any],
        named_states: Dict[str, Any],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
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
            import jax  # @manual  # pyre-ignore[21]

            return jax.device_count()
        return dist.get_world_size() if dist.is_initialized() else 1

    @property
    def num_processes(self) -> int:
        if self._num_processes is not None:
            return self._num_processes
        if self._is_xla_backend:
            import jax  # @manual  # pyre-ignore[21]

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
                from torch._C._distributed_c10d import FakeProcessGroup

                fake_pg = FakeProcessGroup._create_internal(0, self.num_processes)
                mesh = DeviceMesh(
                    self._device_type,
                    torch.arange(self.num_processes),
                    _init_backend=False,
                    _rank=0,
                )
                group_name = "fake_recir_process_mesh"
                mesh._dim_group_names = [group_name]
                mesh._pg_registry[group_name] = fake_pg  # pyre-ignore[6]
                self._process_mesh = mesh
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
                from torch._C._distributed_c10d import FakeProcessGroup

                fake_pg = FakeProcessGroup._create_internal(0, self.num_ranks)
                mesh = DeviceMesh(
                    self._device_type,
                    torch.arange(self.num_ranks),
                    _init_backend=False,
                    _rank=0,
                )
                group_name = "fake_recir_device_mesh"
                mesh._dim_group_names = [group_name]
                mesh._pg_registry[group_name] = fake_pg  # pyre-ignore[6]
                self._device_mesh = mesh
            else:
                self._device_mesh = DeviceMesh(
                    self._device_type, torch.arange(self.num_ranks)
                )
        return self._device_mesh

    def shard_params(
        self,
        model: torch.nn.Module,
        params_and_buffers: Dict[str, Any],
        named_states: Dict[str, Any],
    ) -> None:
        shard_params_and_states(
            model,
            params_and_buffers,
            named_states,
            self.process_mesh,
            self.parallel_style,
        )

    def partition(
        self,
        gm: GraphModule,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        params_and_buffers: Dict[str, Any],
        named_states: Dict[str, Any],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> GraphModule:
        gm = partition_data_parallel(
            gm,
            model,
            optimizer,
            params_and_buffers,
            named_states,
            args,
            kwargs,
            self.device_mesh,
            self.parallel_style,
            self.input_batch_dim,
            self.grad_reduction,
        )
        return gm

    def transform_and_compile(self, gm: GraphModule) -> GraphModule:
        """optimize a distributed graph with a set of optimization passes"""
        # TODO: add more necessary arguments to this interface.
        return self._gm_passes(gm)
