# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Set, Tuple

# A module import, read at call time: xla_lower imports this package back, so a
# from-import here fails under eager (non-lazy) imports.
import recir.lowering.xla_lower as xla_lower
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils._pytree as pytree
from recir.backends.types import BackendType
from recir.distributed.data_parallel import (
    DataParallelStyle,
    OptOrOpts,
    partition_data_parallel,
    shard_optimizer_states_for_replicated_params,
    shard_params_and_states,
    shard_params_by_plan,
)
from recir.distributed.sharding_plan import EmbeddingShardingPlan
from recir.distributed.sharding_spec import FSDPWeightShardSpec, set_state_layout
from torch.distributed.tensor import DeviceMesh
from torch.fx import GraphModule

logger: logging.Logger = logging.getLogger(__name__)


# Keep wrapped fake ProcessGroups alive: the global c10d registry holds only a
# weak reference, so without this the wrapped PG is GC'd -> "already destroyed".
_FAKE_PG_KEEPALIVE: dict[str, Any] = {}


def _rank_for_matching_namespace(
    source: str,
    rank: int,
    source_size: int,
    expected_size: int,
) -> Optional[int]:
    if source_size != expected_size:
        return None
    if not 0 <= rank < source_size:
        raise RuntimeError(f"{source} reported rank {rank} outside [0, {source_size})")
    return rank


def _jax_process_rank(process_count: int) -> Tuple[Optional[int], Optional[str]]:
    try:
        import jax  # @manual
    except ImportError:
        logger.debug("JAX process topology is unavailable", exc_info=True)
        return None, None
    jax_size = int(jax.process_count())
    jax_rank = int(jax.process_index())
    resolved = _rank_for_matching_namespace(
        "JAX process topology",
        jax_rank,
        jax_size,
        process_count,
    )
    mismatch = f"jax={jax_size}" if resolved is None else None
    return resolved, mismatch


def _environment_process_rank(
    process_count: int,
) -> Tuple[Optional[int], list[str]]:
    mismatched_namespaces: list[str] = []
    for rank_var, size_var in (
        ("JAX_PROCESS_INDEX", "JAX_PROCESS_COUNT"),
        ("PROCESS_INDEX", "PROCESS_COUNT"),
        ("RANK", "WORLD_SIZE"),
    ):
        raw_rank = os.environ.get(rank_var)
        if raw_rank is None:
            continue
        raw_size = os.environ.get(size_var)
        if raw_size is None:
            logger.debug(
                "Ignoring incomplete process-rank namespace: %s is set without %s",
                rank_var,
                size_var,
            )
            continue
        try:
            rank = int(raw_rank)
            source_size = int(raw_size)
        except ValueError as error:
            raise RuntimeError(
                f"{rank_var} and {size_var} must be integers; "
                f"got {raw_rank!r} and {raw_size!r}"
            ) from error
        resolved = _rank_for_matching_namespace(
            f"environment {rank_var}/{size_var}",
            rank,
            source_size,
            process_count,
        )
        if resolved is not None:
            return resolved, mismatched_namespaces
        mismatched_namespaces.append(f"{rank_var}/{size_var}={source_size}")
    return None, mismatched_namespaces


def _current_process_rank(process_count: int) -> int:
    if process_count <= 1:
        return 0

    mismatched_namespaces: list[str] = []
    if dist.is_available() and dist.is_initialized():
        torch_size = dist.get_world_size()
        torch_rank = dist.get_rank()
        resolved = _rank_for_matching_namespace(
            "Torch process group",
            torch_rank,
            torch_size,
            process_count,
        )
        if resolved is not None:
            return resolved
        mismatched_namespaces.append(f"torch={torch_size}")

    jax_rank, jax_mismatch = _jax_process_rank(process_count)
    if jax_rank is not None:
        return jax_rank
    if jax_mismatch is not None:
        mismatched_namespaces.append(jax_mismatch)

    environment_rank, environment_mismatches = _environment_process_rank(process_count)
    if environment_rank is not None:
        return environment_rank
    mismatched_namespaces.extend(environment_mismatches)

    # AOT and single-process test hosts intentionally model a larger deployment
    # without joining its process namespace; rank zero is the only local shard.
    if xla_lower._AOT_TOPOLOGY.get() is not None:
        logger.debug(
            "AOT topology uses emulated process rank zero; ignored namespaces: %s",
            mismatched_namespaces,
        )
        return 0
    if not mismatched_namespaces:
        return 0
    raise RuntimeError(
        "No distributed rank source matches the process mesh size "
        f"{process_count}: {', '.join(mismatched_namespaces)}"
    )


def _current_device_rank(num_ranks: int, num_processes: int) -> int:
    if num_processes <= 0 or num_ranks % num_processes != 0:
        raise ValueError(
            "Device mesh requires num_ranks to be divisible by num_processes; "
            f"got num_ranks={num_ranks}, num_processes={num_processes}"
        )
    devices_per_process = num_ranks // num_processes
    return _current_process_rank(num_processes) * devices_per_process


def _current_jax_process_index() -> int:
    if xla_lower._AOT_TOPOLOGY.get() is not None:
        return 0
    import jax  # @manual

    return int(jax.process_index())


def _create_fake_mesh(
    device_type: str,
    size: int,
    group_name: str,
    *,
    rank: int,
) -> DeviceMesh:
    """Create a DeviceMesh backed by a FakeProcessGroup.

    Used for XLA backends where process coordination happens at runtime.
    """
    from torch._C._distributed_c10d import FakeProcessGroup

    if not 0 <= rank < size:
        raise ValueError(f"Fake mesh rank {rank} must be in [0, {size})")
    fake_pg = FakeProcessGroup._create_internal(rank, size)
    mesh = DeviceMesh(
        device_type,
        torch.arange(size),
        _init_backend=False,
        _rank=rank,
    )
    # pyrefly: ignore [bad-assignment]
    mesh._dim_group_names = [group_name]
    # DeviceMesh only creates _pg_registry on some init paths / torch versions
    # (notably not with _init_backend=False); create it if absent so the
    # FakeProcessGroup registration below works across versions.
    if getattr(mesh, "_pg_registry", None) is None:
        mesh._pg_registry = {}
    mesh._pg_registry[group_name] = fake_pg  # pyre-ignore[6]
    # DataParallel XLA (shard_map) explicit-collective path only. The traced
    # body inserts FSDP collectives via DTensor.redistribute (Shard(0)->Replicate
    # for weights, Partial->Shard(0) for grads); those _c10d_functional.* ops
    # resolve their group name through the GLOBAL c10d process_registry, NOT
    # mesh._pg_registry. Two things this comment answers:
    #
    #  * Why register a group at all, and why only now: earlier XLA sharding used
    #    GSPMD (fake_gspmd_*), which auto-partitions via XLA-SPMD and never emits
    #    DTensor functional collectives, so no group name ever had to resolve;
    #    GPU DataParallel had a real PG from init_process_group. This is the
    #    first path that runs DTensor.redistribute with NO real dist init (TPU
    #    trace time), so the group name would resolve to nothing -> we register a
    #    fake PG. (DTensor is also used elsewhere for pure sharding *specs*, e.g.
    #    partition_value/DTensorSpec, which is spec math and needs no live group;
    #    it's specifically redistribute-driven collective *insertion* that's new.)
    #
    #  * Why a NEW ProcessGroup instead of `fake_pg`: `fake_pg` above
    #    (FakeProcessGroup._create_internal) is a c10d *Backend*, not a
    #    ProcessGroup -- note it is passed as the backend arg to
    #    _register_backend below. _register_process_group() and the DTensor
    #    collective resolver both require a ProcessGroup (the object that exposes
    #    allgather/reduce_scatter and dispatches down to a backend), so a bare
    #    Backend can't be registered or called as one. The mesh was built with
    #    _init_backend=False (no PG of its own), so there's nothing to reuse:
    #    construct a minimal real ProcessGroup and attach `fake_pg` as its
    #    per-device CUSTOM backend. At runtime the collectives are executed by the
    #    shard_map lowering (jax all_gather / psum_scatter over "torch_dist");
    #    this PG is only the trace-time name-resolution target, hence FakeStore.
    #
    # GSPMD (fake_gspmd_*) -> skip, preserving its bare fake-mesh behavior.
    if group_name.startswith("fake_recir"):
        try:
            import torch.distributed.distributed_c10d as _c10d
            from torch._C._distributed_c10d import ProcessGroup
            from torch.testing._internal.distributed.fake_pg import FakeStore

            try:
                existing = _c10d._resolve_process_group(group_name)  # pyre-ignore[6]
            except Exception:
                existing = None
            if (
                existing is not None
                and existing.size() == size
                and existing.rank() == rank
            ):
                # Idempotent: a prior trace of the same mesh already registered
                # this group; re-registering the same name would collide.
                _FAKE_PG_KEEPALIVE[group_name] = existing
            else:
                if existing is not None:
                    # A group under this name exists for a different topology.
                    # Drop it first; otherwise the stale size or rank keeps
                    # resolving for DTensor functional collectives.
                    unregister = getattr(_c10d, "_unregister_process_group", None)
                    if unregister is not None:
                        unregister(group_name)  # pyre-ignore[6]
                wrapped = ProcessGroup(FakeStore(), rank, size)
                wrapped._register_backend(
                    torch.device(device_type),
                    ProcessGroup.BackendType.CUSTOM,
                    fake_pg,
                )
                _c10d._register_process_group(group_name, wrapped)  # pyre-ignore[6]
                _FAKE_PG_KEEPALIVE[group_name] = wrapped
        except (ImportError, AttributeError, RuntimeError):
            # Best-effort: older torch may lack these c10d internals, or the
            # group may already be registered. The DTensor collective path then
            # falls back to mesh._pg_registry; log so a genuine registration
            # failure is visible instead of silently mis-routing collectives.
            logger.exception(
                "recir: failed to register wrapped ProcessGroup %r for fake "
                "mesh; DTensor functional collectives may not resolve this group",
                group_name,
            )
    return mesh


@dataclass(frozen=True)
class _StateLayoutPlan:
    """Trace-scoped FSDP state-layout plan produced by ``DataParallel.shard_params``.

    Set fresh on the parallel mode each trace and consumed + cleared exactly
    once by :meth:`ParallelMode.apply_state_layout`. Carries the /P per-process
    weight store (``process_store``), the /P per-process optimizer-state store
    (``opt_process_store``) with the subset of its leaves actually /P-partitioned
    (``opt_partitioned``), the subset of keys that were actually sharded, and the
    world sizes needed to build the typed ``FSDPWeightShardSpec``.
    """

    process_store: Dict[str, Any]  # /P per-process weight store (all params/buffers)
    opt_process_store: Dict[
        str, Dict[str, Any]
    ]  # owning-param fqn -> /P per-process optimizer-state dict
    opt_partitioned: Dict[
        str, Set[str]
    ]  # owning-param fqn -> opt-state keys actually /P-partitioned (dim-0 divisible)
    sharded_keys: Set[str]  # subset of process_store that was FSDP-sharded
    device_world_size: int  # N = P*D  (baked all-gather factor)
    process_world_size: int  # P        (canonical host-store divisor)


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
        (e.g. a bundle-runtime consumer that loads safetensors at job start)
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
    def apply_state_layout(
        self,
        gm: GraphModule,
        params_and_buffers: Dict[str, Any],
        named_states: Dict[str, Any],
    ) -> None:
        """Stamp any state-layout meta this mode produced during ``shard_params``
        onto ``gm``'s placeholders and substitute its canonical host store into
        ``params_and_buffers`` / ``named_states`` (which ``compile_from_gm``
        rebuilds ``flat_state`` from).

        :func:`compile_from_gm` calls this generically for every parallel mode so
        ``api.py`` stays layout-agnostic. Modes that produce no state layout must
        override it with an explicit no-op (see :class:`GSPMDParallel` /
        :class:`HybridParallel`); :class:`DataParallel` on the XLA FSDP path
        provides the real implementation.
        """
        raise NotImplementedError

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
        fsdp_replicate_param_substrs: If set, leave matching params/buffers
            replicated even in ``parallel_style="fully_shard"``. This is for
            structured dimensions whose local shard shape is not a valid model
            shape during later traces.
        fsdp_axis_size: Number of devices in each FSDP group. The default uses
            every device. A smaller value creates multi-host or host-local FSDP
            groups and replicates their state across the remaining data axis.
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
        fsdp_replicate_param_substrs: Optional[Sequence[str]] = None,
        fsdp_shard_optimizer_state_for_replicated_param_substrs: Optional[
            Sequence[str]
        ] = None,
        fsdp_sharded_optimizer_state_keys: Optional[Sequence[str]] = None,
        fsdp_axis_size: Optional[int] = None,
    ) -> None:
        self.parallel_style = parallel_style
        self.input_batch_dim = input_batch_dim
        self.grad_reduction = "sum" if loss_reduction == "mean" else "avg"
        self._num_ranks = num_ranks
        self._num_processes = num_processes
        self._fsdp_table_substr = fsdp_table_substr
        self._fsdp_replicate_param_substrs = tuple(fsdp_replicate_param_substrs or ())
        self._fsdp_shard_optimizer_state_for_replicated_param_substrs = tuple(
            fsdp_shard_optimizer_state_for_replicated_param_substrs or ()
        )
        self._fsdp_sharded_optimizer_state_keys = frozenset(
            fsdp_sharded_optimizer_state_keys or ()
        )
        self._fsdp_axis_size = fsdp_axis_size
        if fsdp_axis_size is not None and fsdp_axis_size < 1:
            raise ValueError("fsdp_axis_size must be positive")
        if (
            fsdp_axis_size is not None
            and num_ranks is not None
            and num_ranks % fsdp_axis_size != 0
        ):
            raise ValueError(
                "num_ranks must be divisible by fsdp_axis_size; "
                f"got num_ranks={num_ranks}, fsdp_axis_size={fsdp_axis_size}"
            )
        if bool(self._fsdp_shard_optimizer_state_for_replicated_param_substrs) != bool(
            self._fsdp_sharded_optimizer_state_keys
        ):
            raise ValueError(
                "State-only optimizer sharding requires both parameter "
                "substrings and state keys"
            )
        if (
            self._fsdp_shard_optimizer_state_for_replicated_param_substrs
            and parallel_style != "fully_shard"
        ):
            raise ValueError("State-only optimizer sharding requires fully_shard mode")
        if (
            self._fsdp_shard_optimizer_state_for_replicated_param_substrs
            and self.uses_hybrid_fsdp
        ):
            raise ValueError(
                "State-only optimizer sharding is not supported with hybrid FSDP"
            )
        missing_replicate_substrs = set(
            self._fsdp_shard_optimizer_state_for_replicated_param_substrs
        ) - set(self._fsdp_replicate_param_substrs)
        if missing_replicate_substrs:
            raise ValueError(
                "State-only optimizer sharding is restricted to explicitly "
                "replicated parameters; missing replication substrings: "
                f"{sorted(missing_replicate_substrs)!r}"
            )
        self._optimizer_state_only_sharded_keys: frozenset[tuple[str, str]] = (
            frozenset()
        )
        self._process_mesh: Optional[DeviceMesh] = None

    @property
    def optimizer_state_only_sharded_keys(self) -> frozenset[tuple[str, str]]:
        return self._optimizer_state_only_sharded_keys

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
        # time `_make_trace_input_transform` (which uses this property) must
        # match the runtime `_to_sharded_jax` local→global multiplier,
        # so use the topology-aware ratio when one is active.
        if xla_lower._AOT_TOPOLOGY.get() is not None:
            return len(xla_lower._get_global_devices()) // max(
                1, len(xla_lower._get_local_devices())
            )
        import jax  # @manual

        return jax.process_count()

    @property
    def fsdp_axis_size(self) -> int:
        axis_size = self._fsdp_axis_size or self.num_ranks
        if self.num_ranks % axis_size != 0:
            raise ValueError(
                "num_ranks must be divisible by fsdp_axis_size; "
                f"got num_ranks={self.num_ranks}, fsdp_axis_size={axis_size}"
            )
        return axis_size

    @property
    def uses_hybrid_fsdp(self) -> bool:
        return self.fsdp_axis_size < self.num_ranks

    @property
    def process_mesh(self) -> DeviceMesh:
        """Mesh sized to ``num_processes`` for physical model/state sharding.

        Uses a FakeProcessGroup since XLA handles process coordination
        at runtime.
        """
        if self._process_mesh is None:
            if self.num_ranks % self.num_processes != 0:
                raise ValueError(
                    "num_ranks must be divisible by num_processes; "
                    f"got num_ranks={self.num_ranks}, "
                    f"num_processes={self.num_processes}"
                )
            local_device_count = self.num_ranks // self.num_processes
            if self.fsdp_axis_size % local_device_count != 0:
                raise ValueError(
                    "fsdp_axis_size must contain a whole number of processes; "
                    f"got fsdp_axis_size={self.fsdp_axis_size}, "
                    f"local_device_count={local_device_count}"
                )
            mesh_size = self.fsdp_axis_size // local_device_count
            mesh_rank = _current_process_rank(self.num_processes) % mesh_size
            self._process_mesh = _create_fake_mesh(
                "cpu",
                mesh_size,
                "fake_gspmd_process_mesh",
                rank=mesh_rank,
            )
        return self._process_mesh

    def shard_params(
        self,
        model: torch.nn.Module,
        params_and_buffers: Dict[str, Any],
        opt: OptOrOpts,
    ) -> Set[str]:
        sharded: Set[str] = set()

        def _should_replicate(key: str) -> bool:
            return any(substr in key for substr in self._fsdp_replicate_param_substrs)

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
                    and not _should_replicate(key)
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
                    own_storage=True,
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
                and not _should_replicate(key)
                and t.ndim >= 1
                and t.shape[0] >= self.fsdp_axis_size
                and t.shape[0] % self.fsdp_axis_size == 0
                and (self.uses_hybrid_fsdp or t.shape[0] % self.num_processes == 0)
            )

        sharded |= shard_params_and_states(
            model,
            params_and_buffers,
            opt,
            self.process_mesh,
            DataParallelStyle.FULLY_SHARD,
            shard_filter=_can_fsdp,
            own_storage=True,
        )
        state_only_substrs = (
            self._fsdp_shard_optimizer_state_for_replicated_param_substrs
        )
        if state_only_substrs:

            def _can_state_only_fsdp(key: str, tensor: torch.Tensor) -> bool:
                return (
                    _should_replicate(key)
                    and any(substr in key for substr in state_only_substrs)
                    and tensor.ndim >= 1
                    and tensor.shape[0] >= self.num_ranks
                    and tensor.shape[0] % self.num_ranks == 0
                    and tensor.shape[0] % self.num_processes == 0
                )

            self._optimizer_state_only_sharded_keys = frozenset(
                shard_optimizer_states_for_replicated_params(
                    model,
                    opt,
                    self.process_mesh,
                    parameter_filter=_can_state_only_fsdp,
                    state_keys=self._fsdp_sharded_optimizer_state_keys,
                )
            )
            logger.info(
                "GSPMD state-only optimizer sharding: %d leaves across %d FQNs",
                len(self._optimizer_state_only_sharded_keys),
                len(
                    {fqn for fqn, _state_key in self._optimizer_state_only_sharded_keys}
                ),
            )
        else:
            self._optimizer_state_only_sharded_keys = frozenset()
        return sharded

    def apply_state_layout(
        self,
        gm: GraphModule,
        params_and_buffers: Dict[str, Any],
        named_states: Dict[str, Any],
    ) -> None:
        """No-op: GSPMD auto-partitions, so it produces no recir_state_layout meta."""

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
        fsdp_replicate_param_substrs: Parameter or buffer name substrings to
            keep replicated in fully-sharded mode. The XLA shard_map path also
            replicates all buffers and parameters whose leading dimension
            cannot be split evenly over all logical devices.
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
        fsdp_replicate_param_substrs: Sequence[str] = (),
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
        self._fsdp_replicate_param_substrs = tuple(fsdp_replicate_param_substrs)

        if custom_passes is not None:
            self._gm_passes: Callable[[GraphModule], GraphModule] = custom_passes
        else:
            # TODO: add a few default passes here.
            self._gm_passes = lambda gm: gm

        self._process_mesh: Optional[DeviceMesh] = None
        self._device_mesh: Optional[DeviceMesh] = None
        self._local_device_mesh: Optional[DeviceMesh] = None
        # Trace-scoped FSDP state-layout plan: set fresh by ``shard_params`` and
        # consumed + cleared once by ``apply_state_layout``. None off the path.
        self._pending_state_layout: Optional[_StateLayoutPlan] = None
        self._active_fsdp_sharded_keys: Optional[Set[str]] = None

    def _should_fsdp_shard(self, key: str, value: torch.Tensor) -> bool:
        if any(substring in key for substring in self._fsdp_replicate_param_substrs):
            return False
        if not self._uses_explicit_collective_fsdp:
            return True
        return (
            value.ndim >= 1
            and value.shape[0] >= self.num_ranks
            and value.shape[0] % self.num_ranks == 0
        )

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
    def _uses_explicit_collective_fsdp(self) -> bool:
        """The XLA shard_map explicit-collective FSDP contract: weights traced /N
        over device_mesh with the host store kept /P. Specific to
        ``XLAShardMapBackend`` -- GSPMD auto-partitions and needs neither, so
        scope the /N-mesh + /P-capture to this backend, not all XLA backends."""
        return self._backend == BackendType.XLA_SHARD_MAP

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
                    rank=_current_process_rank(self.num_processes),
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
                    rank=_current_device_rank(self.num_ranks, self.num_processes),
                )
            else:
                self._device_mesh = DeviceMesh(
                    self._device_type, torch.arange(self.num_ranks)
                )
        return self._device_mesh

    @property
    def local_device_mesh(self) -> DeviceMesh:
        """Mesh for selecting rank zero's trace shard inside one process."""
        if self._local_device_mesh is None:
            devices_per_process = self.num_ranks // self.num_processes
            self._local_device_mesh = _create_fake_mesh(
                self._device_type,
                devices_per_process,
                "fake_local_fsdp_device_mesh",
                rank=0,
            )
        return self._local_device_mesh

    def shard_params(
        self,
        model: torch.nn.Module,
        params_and_buffers: Dict[str, Any],
        opt: OptOrOpts,
    ) -> Set[str]:
        # Non-XLA (real DTensor) shards over process_mesh (size P) but the traced
        # collectives all-gather over device_mesh (size N); the two meshes must
        # match, so require one device per process (D=1, i.e. N==P). D>1 here
        # would shard /P while the model spans N>P devices -> silent mis-shard.
        if not self._is_xla_backend and self.num_ranks != self.num_processes:
            raise ValueError(
                "Non-XLA DataParallel requires num_ranks == num_processes "
                f"(one device per process, D=1); got num_ranks={self.num_ranks} "
                f"!= num_processes={self.num_processes}. Multi-device-per-process "
                "(D>1) FSDP is only supported on the XLA shard_map path."
            )
        # XLA shard-map narrows full state to /P and then /P to /N, so N must
        # contain an integral number of logical devices per process.
        if self._is_xla_backend and self.num_ranks % self.num_processes != 0:
            raise ValueError(
                "XLA DataParallel requires num_ranks % num_processes == 0 "
                f"(D devices/process); got num_ranks={self.num_ranks}, "
                f"num_processes={self.num_processes}."
            )
        # GSPMD XLA and non-XLA physically shard over process_mesh. Explicit
        # shard-map FSDP takes the two-stage /P then /N path below while its
        # graph collectives continue to use device_mesh.
        shard_mesh = self.process_mesh

        # Explicit XLA FSDP first creates an owned /P process shard, snapshots it
        # for flat_state, and then exposes an aliased /N trace view. Other modes
        # retain their existing single-stage physical sharding.
        replicated_buffer_keys = (
            set(dict(model.named_buffers(remove_duplicate=False)))
            if self._uses_explicit_collective_fsdp
            else set()
        )

        def should_shard(key: str, value: torch.Tensor) -> bool:
            return key not in replicated_buffer_keys and self._should_fsdp_shard(
                key, value
            )

        candidate_sharded_keys = {
            key
            for key, value in params_and_buffers.items()
            if self.parallel_style == DataParallelStyle.FULLY_SHARD
            and should_shard(key, value)
        }
        if (
            self._uses_explicit_collective_fsdp
            and self.parallel_style == DataParallelStyle.FULLY_SHARD
        ):
            sharded_keys = self._shard_explicit_xla_fsdp_state(
                model,
                params_and_buffers,
                opt,
                should_shard,
                candidate_sharded_keys,
            )
        else:
            self._pending_state_layout = None
            sharded_keys = shard_params_and_states(
                model,
                params_and_buffers,
                opt,
                shard_mesh,
                self.parallel_style,
                shard_filter=should_shard,
            )
        self._active_fsdp_sharded_keys = set(sharded_keys)
        return sharded_keys

    def _shard_explicit_xla_fsdp_state(
        self,
        model: torch.nn.Module,
        params_and_buffers: Dict[str, Any],
        opt: OptOrOpts,
        shard_filter: Callable[[str, torch.Tensor], bool],
        candidate_sharded_keys: Set[str],
    ) -> Set[str]:
        self._validate_explicit_xla_optimizer_state(
            params_and_buffers,
            opt,
            candidate_sharded_keys,
        )
        sharded_keys = shard_params_and_states(
            model,
            params_and_buffers,
            opt,
            self.process_mesh,
            self.parallel_style,
            shard_filter=shard_filter,
            own_storage=True,
        )
        current_parameters = dict(model.named_parameters(remove_duplicate=False))
        missing_parameters = sharded_keys - set(current_parameters)
        if missing_parameters:
            raise RuntimeError(
                "XLA shard-map sharded state is not bound to model parameters: "
                f"{sorted(missing_parameters)!r}"
            )
        for key in sharded_keys:
            params_and_buffers[key] = current_parameters[key]

        stores = self.capture_fsdp_process_store(
            params_and_buffers,
            opt,
            sharded_keys=sharded_keys,
        )
        assert stores is not None
        process_store, opt_process_store, opt_partitioned = stores
        self._pending_state_layout = _StateLayoutPlan(
            process_store=process_store,
            opt_process_store=opt_process_store,
            opt_partitioned=opt_partitioned,
            sharded_keys=sharded_keys,
            device_world_size=self.num_ranks,
            process_world_size=self.num_processes,
        )

        if self.num_ranks != self.num_processes:
            trace_sharded_keys = shard_params_and_states(
                model,
                params_and_buffers,
                opt,
                self.local_device_mesh,
                self.parallel_style,
                shard_filter=lambda key, _value: key in sharded_keys,
            )
            if trace_sharded_keys != sharded_keys:
                raise RuntimeError(
                    "XLA shard-map trace sharding changed the selected parameter set"
                )
        return sharded_keys

    def _validate_explicit_xla_optimizer_state(
        self,
        params_and_buffers: Dict[str, Any],
        opt: OptOrOpts,
        sharded_keys: Set[str],
    ) -> None:
        opts = [opt] if isinstance(opt, torch.optim.Optimizer) else list(opt or [])
        for fqn in sharded_keys:
            parameter = params_and_buffers[fqn]
            if parameter.ndim < 1 or parameter.shape[0] % self.num_processes != 0:
                raise RuntimeError(
                    "XLA shard-map owned parameters require dim 0 to divide "
                    "evenly across processes: "
                    f"{fqn!r} has shape {tuple(parameter.shape)}, "
                    f"num_processes={self.num_processes}"
                )
            for optimizer in opts:
                if parameter not in optimizer.state:
                    continue
                for state_key, value in optimizer.state[parameter].items():
                    if (
                        isinstance(value, torch.Tensor)
                        and value.ndim >= 1
                        and value.shape != parameter.shape
                    ):
                        raise RuntimeError(
                            "XLA shard-map owned optimizer shards require ranked "
                            "state to match its parameter shape: "
                            f"{fqn!r}/{state_key!r} has {tuple(value.shape)}, "
                            f"parameter has {tuple(parameter.shape)}"
                        )

    def capture_fsdp_process_store(
        self,
        params_and_buffers: Dict[str, Any],
        opt: OptOrOpts,
        *,
        sharded_keys: Optional[Set[str]] = None,
    ) -> Optional[
        Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], Dict[str, Set[str]]]
    ]:
        """Snapshot owned /P state before it is narrowed to the /N trace view.

        The caller has already compacted full tensors over ``process_mesh``.
        Parameter references and shallow optimizer-state copies keep those /P
        allocations alive while the live module is rebound to /N views of the
        same storage. Runtime state sync can therefore update both views without
        retaining the original full tensors.
        """
        if not (
            self._uses_explicit_collective_fsdp
            and self.parallel_style == DataParallelStyle.FULLY_SHARD
        ):
            return None
        keys_to_capture = (
            set(params_and_buffers) if sharded_keys is None else sharded_keys
        )
        process_store = {
            key: value
            for key, value in params_and_buffers.items()
            if key in keys_to_capture
        }
        opt_process_store: Dict[str, Dict[str, Any]] = {}
        opt_partitioned: Dict[str, Set[str]] = {}
        opts = [opt] if isinstance(opt, torch.optim.Optimizer) else list(opt or [])
        for o in opts:
            for fqn, param in params_and_buffers.items():
                if fqn not in keys_to_capture:
                    continue
                if param not in o.state:
                    continue
                store = dict(o.state[param])
                partitioned = {
                    key
                    for key, value in store.items()
                    if isinstance(value, torch.Tensor) and value.ndim >= 1
                }
                opt_process_store[fqn] = store
                opt_partitioned[fqn] = partitioned
        return process_store, opt_process_store, opt_partitioned

    def apply_state_layout(
        self,
        gm: GraphModule,
        params_and_buffers: Dict[str, Any],
        named_states: Dict[str, Any],
    ) -> None:
        """Consume the trace-scoped plan produced by :meth:`shard_params`.

        Reproduces the original FSDP-on-XLA behavior via the typed
        ``recir_state_layout`` spec: (1) tag each sharded weight/buffer and
        optimizer-state placeholder with an :class:`FSDPWeightShardSpec`, and
        (2) substitute the /P per-process store into ``params_and_buffers`` /
        ``named_states`` (which ``compile_from_gm`` rebuilds ``flat_state``
        from). The plan is popped and cleared so a reused parallel mode never
        carries a stale /P store. No-op off the XLA FSDP path (plan is None).
        """
        plan = self._pending_state_layout
        self._pending_state_layout = None
        if plan is None:
            return

        spec = FSDPWeightShardSpec(
            shard_dim=0,
            device_world_size=plan.device_world_size,
            process_world_size=plan.process_world_size,
        )

        # TAG weights/buffers by ``node.meta["fqn"]`` (set in ``_local_compile``),
        # not positional index -- robust to leading-placeholder shifts from
        # region-transform passes.
        for node in gm.graph.nodes:
            if node.op == "placeholder" and node.meta.get("fqn") in plan.sharded_keys:
                set_state_layout(node, spec)

        # SUBSTITUTE: overwrite each sharded weight and its optimizer state with
        # the /P shard. Missing keys skip gracefully (matches the legacy inline
        # substitution).
        for key in plan.sharded_keys:
            if key in plan.process_store and key in params_and_buffers:
                params_and_buffers[key] = plan.process_store[key]
            if key in plan.opt_process_store and key in named_states:
                named_states[key] = plan.opt_process_store[key]

        # TAG optimizer-state placeholders positionally: they carry no
        # ``node.meta["fqn"]``, so mirror the GSPMD ``is_fsdp_sharded`` logic.
        # ``_local_compile`` lifts inputs as params+buffers, then the
        # ``named_states`` leaves, then forward inputs; a named_states leaf is
        # sharded iff its owning param fqn was sharded. Walk tensor placeholders
        # in that flat order and tag the sharded leaves past the params+buffers
        # prefix (the prefix is tagged by fqn above).
        is_sharded = [key in plan.sharded_keys for key in params_and_buffers]
        optimizer_metadata: list[tuple[str, Optional[str]]] = []
        for fqn, state in named_states.items():
            # Count TENSOR leaves only: a non-tensor state entry (e.g. a
            # Python-int step) is baked as a constant, not lifted as a
            # placeholder, so counting len(state) would drift the positional idx.
            partitioned = plan.opt_partitioned.get(fqn, frozenset())
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    optimizer_metadata.append((fqn, k if isinstance(k, str) else None))
                    # Sharded iff the owning param was sharded AND capture
                    # actually /P-partitioned this leaf (dim-0 divisible); a
                    # full-kept leaf stays untagged so its Shard(0) spec never
                    # over-claims vs. the un-partitioned runtime value.
                    is_sharded.append(fqn in plan.sharded_keys and k in partitioned)
        num_pb = len(params_and_buffers)
        tensor_idx = 0
        for node in gm.graph.nodes:
            if node.op != "placeholder":
                continue
            val = node.meta.get("val")
            if not isinstance(val, torch.Tensor):
                continue
            # Skip a 0-dim opt-state leaf (Adam ``step``): it has no dim-0 to
            # Shard(0), so tagging it would crash the /N shard in
            # _to_sharded_jax. Left untagged it stays replicated (correct for a
            # scalar). ndim>=1 leaves mirror their /N-divisible param (val here
            # is the /N trace shard, so no further shape check applies).
            if (
                num_pb <= tensor_idx < len(is_sharded)
                and is_sharded[tensor_idx]
                and val.ndim >= 1
            ):
                set_state_layout(node, spec)
            if num_pb <= tensor_idx < num_pb + len(optimizer_metadata):
                optimizer_idx = tensor_idx - num_pb
                fqn, state_name = optimizer_metadata[optimizer_idx]
                node.meta["recir_state_kind"] = "optimizer"
                node.meta["recir_state_fqn"] = fqn
                node.meta["recir_state_name"] = state_name
            tensor_idx += 1
        optimizer_placeholder_count = max(0, tensor_idx - num_pb)
        if optimizer_placeholder_count < len(optimizer_metadata):
            raise RuntimeError(
                "Optimizer placeholder count does not match canonical state metadata: "
                f"placeholders={optimizer_placeholder_count}, "
                f"entries={len(optimizer_metadata)}"
            )

    def partition(
        self,
        gm: GraphModule,
        *,
        params: Dict[str, nn.Parameter],
        buffers: Dict[str, torch.Tensor],
        named_states: Dict[str, Dict[str, Any]],
    ) -> GraphModule:
        replicated_param_state_indices: Optional[Set[int]] = None
        if (
            self.parallel_style == DataParallelStyle.FULLY_SHARD
            and self._active_fsdp_sharded_keys is not None
        ):
            param_buffer_keys = [*params, *buffers]
            replicated_param_state_indices = {
                index
                for index, key in enumerate(param_buffer_keys)
                if key not in self._active_fsdp_sharded_keys
            }
            state_index = len(param_buffer_keys)
            for fqn, state in named_states.items():
                state_leaves = pytree.tree_leaves(state)
                if fqn not in self._active_fsdp_sharded_keys:
                    replicated_param_state_indices.update(
                        range(state_index, state_index + len(state_leaves))
                    )
                state_index += len(state_leaves)
        gm = partition_data_parallel(
            gm,
            params=params,
            buffers=buffers,
            named_states=named_states,
            mesh=self.device_mesh,
            parallel_style=self.parallel_style,
            input_batch_dim=self.input_batch_dim,
            grad_reduction=self.grad_reduction,
            replicated_param_state_indices=replicated_param_state_indices,
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
    ``sparse_transform_factory`` configures the training transform, which may
    own embedding optimizer state and rewrite backward. Forward-only eval /
    inference graphs instead reuse the module's existing tables and apply the
    built-in forward-only sparse transform.

    Example::

        from recir.distributed.sharding_plan import EmbeddingShardingPlan, EmbeddingShardingType

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
        dense_fsdp_module_fqns: Optional[Set[str]] = None,
        input_batch_dim: int = 0,
        custom_passes: Optional[Callable[[GraphModule], GraphModule]] = None,
        sparse_transform_factory: Optional[
            Callable[[Set[str], DeviceMesh, nn.Module], Any]
        ] = None,
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
        if dense_fsdp_module_fqns is not None:
            if not dense_fsdp_module_fqns:
                raise ValueError("dense_fsdp_module_fqns must not be empty")
            if self.dense_style != DataParallelStyle.FULLY_SHARD:
                raise ValueError(
                    "dense_fsdp_module_fqns requires dense_style='fully_shard'"
                )
        self.dense_fsdp_module_fqns: Optional[Set[str]] = (
            None if dense_fsdp_module_fqns is None else set(dense_fsdp_module_fqns)
        )
        self.input_batch_dim = input_batch_dim
        self.sparse_transform_factory = sparse_transform_factory

        if custom_passes is not None:
            self._gm_passes: Callable[[GraphModule], GraphModule] = custom_passes
        else:
            self._gm_passes = lambda gm: gm

        self._mesh: Optional[DeviceMesh] = None
        self._active_fsdp_sharded_keys: Optional[Set[str]] = None

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

    def _resolve_dense_shard_param_fqns(
        self, model: torch.nn.Module
    ) -> Optional[Set[str]]:
        if self.dense_fsdp_module_fqns is None:
            return None

        result: Set[str] = set()
        for module_fqn in sorted(self.dense_fsdp_module_fqns):
            try:
                root_module = (
                    model if not module_fqn else model.get_submodule(module_fqn)
                )
            except AttributeError as error:
                raise ValueError(
                    f"Dense FSDP module does not exist: {module_fqn!r}"
                ) from error
            root_param_fqns = {
                f"{module_fqn}.{param_fqn}" if module_fqn else param_fqn
                for param_fqn, _ in root_module.named_parameters(remove_duplicate=False)
            }
            if not root_param_fqns:
                raise ValueError(f"Dense FSDP module has no parameters: {module_fqn!r}")
            result.update(root_param_fqns)
        return result

    def shard_params(
        self,
        model: torch.nn.Module,
        params_and_buffers: Dict[str, Any],
        opt: OptOrOpts,
    ) -> Set[str]:
        dense_shard_param_fqns = self._resolve_dense_shard_param_fqns(model)
        sharded_keys = shard_params_by_plan(
            model,
            params_and_buffers,
            opt,
            self.mesh,
            self.dense_style,
            embedding_plan=self.embedding_plan,
            dense_shard_param_fqns=dense_shard_param_fqns,
        )
        self._active_fsdp_sharded_keys = set(sharded_keys)
        return sharded_keys

    def apply_state_layout(
        self,
        gm: GraphModule,
        params_and_buffers: Dict[str, Any],
        named_states: Dict[str, Any],
    ) -> None:
        """No-op: HybridParallel produces no recir_state_layout meta."""

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

        replicated_param_state_indices: Optional[Set[int]] = None
        if (
            self.dense_style == DataParallelStyle.FULLY_SHARD
            and self._active_fsdp_sharded_keys is not None
        ):
            param_buffer_keys = [*params, *buffers]
            replicated_param_state_indices = {
                index
                for index, key in enumerate(param_buffer_keys)
                if key not in self._active_fsdp_sharded_keys
            }
            state_index = len(param_buffer_keys)
            for fqn, state in named_states.items():
                state_leaves = pytree.tree_leaves(state)
                if fqn not in self._active_fsdp_sharded_keys:
                    replicated_param_state_indices.update(
                        range(state_index, state_index + len(state_leaves))
                    )
                state_index += len(state_leaves)

        gm = partition_data_parallel(
            gm,
            params=params,
            buffers=buffers,
            named_states=named_states,
            mesh=self.mesh,
            parallel_style=self.dense_style,
            input_batch_dim=self.input_batch_dim,
            embedding_param_indices=embedding_param_indices,
            replicated_param_state_indices=replicated_param_state_indices,
        )

        return gm

    def transform_and_compile(self, gm: GraphModule) -> GraphModule:
        return self._gm_passes(gm)
