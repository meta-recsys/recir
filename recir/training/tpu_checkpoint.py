#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""TPU checkpoint support for RecIR GSPMD models with in-graph embeddings.

The in_graph embedding tables live as torch ``nn.Parameter``s, so the whole
checkpoint can stay in the ``torch.distributed.checkpoint`` (DCP) world. But the
``_sharded_tables`` are per-process ``Shard(0)`` row blocks stored as *plain*
Parameters with no sharding metadata: DCP would treat them as replicated and
dedup them to rank 0, silently dropping every other host's rows. We wrap those
entries as ``DTensor``s so DCP saves/loads each host's shard; all other
(replicated) entries pass through unchanged and DCP dedups them correctly.
"""

import copy
import itertools
import logging
import time
from typing import Any, cast, Dict, Iterator, List, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.utils._pytree as pytree
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard

logger: logging.Logger = logging.getLogger(__name__)

# Key under which the WarmupController iter is persisted inside the model
# checkpoint (it lives in no nn.Module state_dict, so we carry it explicitly).
WARMUP_ITER_KEY = "__recir_warmup_iter__"
# Keys under which optimizer state_dicts are nested inside the model checkpoint
# (so they too are sanitized to plain host tensors before DCP writes them).
OPTIMIZER_STATE_KEY = "__recir_optimizer__"
EMBEDDING_OPTIMIZER_STATE_KEY = "__recir_embedding_optimizer__"
# Save-time ``mesh slot -> torch rank`` permutation for the row-sharded tables.
SHARD_ORDER_KEY = "__recir_shard_order__"


def _to_plain_host_tensor(value: Any) -> Any:
    """Convert a checkpoint value to a plain host ``torch.Tensor``.

    recir/torchax ``Tensor`` is a ``torch.Tensor`` SUBCLASS that holds a
    ``jax.Array`` + the torchax ``Environment``; ``torch.save``/DCP can't pickle
    that env (it reaches a ``WeakValueDictionary`` finalizer inside JAX caches).
    So any tensor SUBCLASS that leaks into the state_dict must be converted to a
    plain tensor first. We detect subclasses structurally (by exact type), not by
    importing a specific torchax class -- that import can differ or fail, letting
    a subclass slip through. Non-tensor values pass through unchanged.
    """
    if not isinstance(value, torch.Tensor):
        return value
    # Fast path: a genuine plain tensor / Parameter -> just detach to host CPU.
    if type(value) in (torch.Tensor, torch.nn.Parameter):
        return value.detach().to("cpu")
    # Any OTHER torch.Tensor subclass (e.g. a torchax Tensor wrapping a jax.Array
    # + env). Convert to a plain host tensor and drop the unpicklable refs.
    torch_bridge = getattr(value, "torch", None)
    if callable(torch_bridge):  # torchax Tensor.torch(): direct jax->host copy
        return cast(torch.Tensor, torch_bridge()).detach()
    # Generic subclass with real data: copy into a fresh plain host tensor.
    # (Avoid .numpy()/np.asarray -- torch forbids .numpy() on tensor subclasses.)
    out = torch.empty(value.shape, dtype=value.dtype, device="cpu")
    with torch.no_grad():
        out.copy_(value.detach())
    return out


def _freeze_checkpoint_value(value: Any) -> Any:
    value = _to_plain_host_tensor(value)
    if isinstance(value, torch.Tensor):
        return value
    return copy.deepcopy(value)


class _RecirAsyncStager:
    """Use the snapshot already prepared by ``RecirCheckpointable``."""

    @property
    def should_synchronize_after_execute(self) -> bool:
        return True

    def stage(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        return state_dict

    def synchronize_staging(self) -> None:
        pass

    def close(self) -> None:
        pass


def reseed_warmup(warmup: Any, iteration: int) -> None:
    """Restore a freshly-constructed WarmupController to ``iteration`` after a
    checkpoint resume.

    The controller's iter counter and per-stage cursors live in no state_dict,
    so on resume they must be re-derived from the restored global step. We replay
    the schedule through the controller's own ``apply()``/``step()`` lifecycle
    (rather than duplicating its stage-advance logic), which is correct by
    construction: ``step()`` bumps the iter and rewrites the 0-d LR buffer(s)
    exactly as it does during training. ``warmup`` must be freshly constructed
    (iter 0); we ``apply()`` once to seed iter 0 then ``step()`` ``iteration``
    times.
    """
    if iteration < 0:
        raise ValueError(f"iteration must be non-negative, got {iteration}")
    warmup.set_iter(iteration)


def _is_sharded_table_key(key: str, sharded_prefixes: Sequence[str]) -> bool:
    return any(prefix in key for prefix in sharded_prefixes)


def host_ordinal_from_device_positions(
    positions: Sequence[int],
    num_global_devices: int,
    num_processes: int,
) -> int:
    """This process's slot on the GSPMD FSDP axis, from its devices' positions."""
    if num_processes <= 0 or num_global_devices % num_processes:
        raise RuntimeError(
            "GSPMD devices must divide evenly across processes; got "
            f"devices={num_global_devices}, processes={num_processes}"
        )
    local_count = num_global_devices // num_processes
    ordered = sorted(positions)
    if len(ordered) != local_count:
        raise RuntimeError(
            f"expected {local_count} local devices for {num_processes} "
            f"processes over {num_global_devices} devices, got {len(ordered)}"
        )
    first = ordered[0]
    if ordered != list(range(first, first + local_count)):
        raise RuntimeError(
            "this process's devices are not a contiguous block on the GSPMD "
            f"FSDP axis (positions {ordered}); a Shard(0) checkpoint cannot "
            "express a scattered host block"
        )
    if first % local_count:
        raise RuntimeError(
            f"the local device block starts at axis position {first}, which is "
            f"not a multiple of the {local_count} devices per process"
        )
    return first // local_count


def shard_mesh_order(
    rank_ordinals: Sequence[Tuple[int, int]], num_processes: int
) -> List[int]:
    order: List[int] = [-1] * num_processes
    for rank, ordinal in rank_ordinals:
        if not 0 <= rank < num_processes:
            raise RuntimeError(
                f"torch rank {rank} is out of range for {num_processes} processes"
            )
        if not 0 <= ordinal < num_processes:
            raise RuntimeError(
                f"FSDP ordinal {ordinal} is out of range for {num_processes} processes"
            )
        if order[ordinal] != -1:
            raise RuntimeError(
                f"torch ranks {order[ordinal]} and {rank} both claim FSDP "
                f"ordinal {ordinal}; the rank -> ordinal map is not a bijection"
            )
        order[ordinal] = rank
    missing = [j for j, rank in enumerate(order) if rank == -1]
    if missing:
        raise RuntimeError(f"no process reported FSDP ordinal(s) {missing}")
    return order


def fsdp_host_ordinal(num_processes: int) -> int:
    """Read this process's FSDP-axis slot off the live JAX device list."""
    from recir.lowering.xla_lower import _get_global_devices, _get_local_devices

    global_devices = _get_global_devices()
    local_devices = _get_local_devices()
    position = {id(device): i for i, device in enumerate(global_devices)}
    try:
        positions = [position[id(device)] for device in local_devices]
    except KeyError as error:
        raise RuntimeError(
            "a local JAX device is absent from the global device list, so the "
            "FSDP shard ordinal cannot be derived"
        ) from error
    return host_ordinal_from_device_positions(
        positions, len(global_devices), num_processes
    )


def logical_replay_rank(
    gloo_rank: int,
    num_processes: int,
    global_device_platforms: Sequence[str],
) -> int:
    """Keep TPU replay streams attached to stable GSPMD host ordinals."""
    if (
        num_processes <= 1
        or len(global_device_platforms) <= 1
        or any(platform != "tpu" for platform in global_device_platforms)
    ):
        return gloo_rank
    return fsdp_host_ordinal(num_processes)


def gather_shard_order(num_processes: int) -> List[int]:
    ordinal = fsdp_host_ordinal(num_processes)
    gathered: List[Any] = [None] * num_processes
    dist.all_gather_object(gathered, (dist.get_rank(), ordinal))
    order = shard_mesh_order(gathered, num_processes)
    logger.info(
        "[checkpoint] FSDP shard order (mesh slot -> torch rank): %s; "
        "this rank %d sits at ordinal %d",
        order,
        dist.get_rank(),
        ordinal,
    )
    return order


def wrap_sharded_for_save(
    state_dict: Dict[str, Any],
    sharded_prefixes: Sequence[str],
    mesh: DeviceMesh,
) -> Dict[str, Any]:
    """Return a copy of ``state_dict`` with sharded-table entries wrapped as
    ``DTensor`` (``Shard(0)`` over ``mesh``) and all other entries left as-is."""
    out: Dict[str, Any] = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor) and _is_sharded_table_key(
            key, sharded_prefixes
        ):
            out[key] = DTensor.from_local(value, mesh, [Shard(0)], run_check=False)
        else:
            out[key] = value
    return out


def unwrap_after_load(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of ``state_dict`` with top-level ``DTensor`` values local."""
    out: Dict[str, Any] = {}
    for key, value in state_dict.items():
        out[key] = value.to_local() if isinstance(value, DTensor) else value
    return out


def _unwrap_dtensors(value: Any) -> Any:
    """Recursively replace checkpoint ``DTensor`` leaves by their local shards."""
    return pytree.tree_map(
        lambda leaf: leaf.to_local() if isinstance(leaf, DTensor) else leaf,
        value,
    )


def _sharded_optimizer_state_ids(
    optimizer: Any,
    model: torch.nn.Module,
    sharded_prefixes: Sequence[str],
    state_dict: Dict[str, Any],
) -> set[Any]:
    """Optimizer state IDs whose parameters are row-sharded model tables."""
    parameter_names = {id(param): name for name, param in model.named_parameters()}
    state_ids: set[Any] = set()
    saved_groups = state_dict.get("param_groups", ())
    if len(saved_groups) != len(optimizer.param_groups):
        raise RuntimeError(
            "optimizer state_dict param-group count does not match the live optimizer"
        )
    for live_group, saved_group in zip(optimizer.param_groups, saved_groups):
        live_params = live_group["params"]
        saved_params = saved_group["params"]
        if len(live_params) != len(saved_params):
            raise RuntimeError(
                "optimizer state_dict parameter count does not match the live optimizer"
            )
        for param, state_id in zip(live_params, saved_params):
            name = parameter_names.get(id(param))
            if name is not None and _is_sharded_table_key(name, sharded_prefixes):
                state_ids.add(state_id)
    return state_ids


def _embedding_optimizer_state_for_save(
    optimizer: Any,
    model: torch.nn.Module,
    sharded_prefixes: Sequence[str],
    mesh: DeviceMesh | None,
) -> Dict[str, Any]:
    """Freeze embedding optimizer state and mark local row shards for DCP."""
    state_dict = optimizer.state_dict()
    sharded_state_ids = _sharded_optimizer_state_ids(
        optimizer, model, sharded_prefixes, state_dict
    )
    frozen = pytree.tree_map(_freeze_checkpoint_value, state_dict)
    if mesh is None:
        return frozen

    for state_id in sharded_state_ids:
        param_state = frozen["state"].get(state_id)
        if param_state is None:
            continue
        frozen["state"][state_id] = pytree.tree_map(
            lambda leaf: (
                DTensor.from_local(leaf, mesh, [Shard(0)], run_check=False)
                if isinstance(leaf, torch.Tensor) and leaf.ndim > 0
                else leaf
            ),
            param_state,
        )
    return frozen


def _load_optimizer_state_preserving_lr(
    optimizer: Any, state_dict: Dict[str, Any]
) -> None:
    """Load optimizer state without replacing its graph-captured LR tensors."""
    live_lrs = [group.get("lr") for group in optimizer.param_groups]
    optimizer.load_state_dict(_unwrap_dtensors(state_dict))
    if len(live_lrs) != len(optimizer.param_groups):
        raise RuntimeError("optimizer param-group count changed while loading state")
    for group, live_lr in zip(optimizer.param_groups, live_lrs):
        if isinstance(live_lr, torch.Tensor):
            group["lr"] = live_lr


def _replace_tensor_impl(tensor: torch.Tensor, replacement: torch.Tensor) -> None:
    """Replace storage while preserving the Python object and its weakrefs."""
    if tensor.grad_fn is not None or tensor.grad is not None:
        raise RuntimeError(
            "cannot replace storage while the tensor has live autograd state"
        )

    use_count = tensor._use_count()
    if use_count > 1:
        if use_count != 2 or not tensor.is_leaf:
            raise RuntimeError(
                "cannot replace storage while the tensor has live references"
            )
        accumulate_grad = torch.autograd.graph.get_gradient_edge(tensor).node
        if tensor._use_count() != 2:
            raise RuntimeError(
                "cannot replace storage while the tensor has live references"
            )

        def reject_backward(grad_outputs: Any) -> None:
            raise RuntimeError(
                "cannot run backward through checkpoint host-storage replacement"
            )

        accumulate_grad.register_prehook(reject_backward)

    replacement.requires_grad_(tensor.requires_grad)
    torch._C._swap_tensor_impl(tensor, replacement)


def _param_leaf_indices(state_fqns: Sequence[str]) -> Dict[str, int]:
    """Map a params/buffers FQN to its ``flat_state`` index.

    ``state_fqns`` entries are pytree keypaths over ``[params_buffers,
    named_states]`` (``recir.compiler.process_state._compute_state_fqns``), so a
    params/buffers leaf is exactly ``[0]['<fqn>']``. Optimizer state for the
    same parameter lives under ``[1]['<fqn>']['<state>']`` and must not match.
    """
    prefix, suffix = "[0]['", "']"
    indices: Dict[str, int] = {}
    for index, path in enumerate(state_fqns):
        if not path.startswith(prefix) or not path.endswith(suffix):
            continue
        fqn = path[len(prefix) : -len(suffix)]
        if "'" not in fqn:
            indices[fqn] = index
    return indices


def _state_leaf_indices(state_fqns: Sequence[str]) -> Dict[Tuple[str, str], int]:
    """Map an optimizer-state ``(param FQN, state key)`` to its index.

    The optimizer half of ``flat_state`` is keyed ``[1]['<fqn>']['<state>']``:
    ``recir.api`` builds ``named_states`` as ``{fqn: opt.state[param]}``, a dict
    per parameter. These leaves are row-sharded alongside the parameter they
    belong to (``shard_params_and_states`` -- note the *and states*), so they
    carry the same dual binding the parameters do.
    """
    prefix, suffix = "[1]['", "']"
    indices: Dict[Tuple[str, str], int] = {}
    for index, path in enumerate(state_fqns):
        if not path.startswith(prefix) or not path.endswith(suffix):
            continue
        parts = path[len(prefix) : -len(suffix)].split("']['")
        if len(parts) == 2 and not any("'" in part for part in parts):
            indices[(parts[0], parts[1])] = index
    return indices


def _optimizer_states(
    model: torch.nn.Module,
    optimizers: Sequence[Any],
    parameter_prefixes: Sequence[str] | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Return optimizer state keyed exactly like recir's ``named_states``."""
    named: Dict[str, Dict[str, Any]] = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        if parameter_prefixes is not None and not any(
            prefix in name for prefix in parameter_prefixes
        ):
            continue
        owners = [
            optimizer
            for optimizer in optimizers
            if optimizer is not None and param in optimizer.state
        ]
        if len(owners) > 1:
            raise RuntimeError(f"parameter {name!r} has state in multiple optimizers")
        if owners:
            named[name] = owners[0].state[param]
    return named


def _optimizer_payload_leaves(
    optimizer: Any, model: torch.nn.Module, payload: Dict[str, Any]
) -> Iterator[Tuple[str, str, Any]]:
    """``(param FQN, state key, tensor)`` for each tensor in an optimizer payload."""
    names = {id(p): n for n, p in model.named_parameters(remove_duplicate=False)}
    state = payload.get("state", {})
    for live_group, saved_group in zip(
        optimizer.param_groups, payload.get("param_groups", ())
    ):
        for param, state_id in zip(live_group["params"], saved_group["params"]):
            fqn = names.get(id(param))
            if fqn is None:
                continue
            for key, value in state.get(state_id, {}).items():
                yield fqn, key, value


class HostStateBuffers:
    """The run's one set of host tensors for checkpoint state."""

    def __init__(
        self,
        flat_state: List[torch.Tensor],
        state_fqns: Sequence[str],
        model: torch.nn.Module,
        optimizers: Sequence[Any],
        table_prefixes: Sequence[str],
    ) -> None:
        self._flat_state = flat_state
        self._model = model
        self._optimizers = optimizers
        self._table_prefixes = table_prefixes
        self._param_indices = _param_leaf_indices(state_fqns)
        self._state_indices = _state_leaf_indices(state_fqns)
        self._owned: Dict[int, int] = {}

    def _state_leaves(self) -> Iterator[Tuple[int, torch.Tensor, str, bool]]:
        """Every ``(flat_state index, owning tensor, name, zero_when_meta)``."""
        for name, owner in itertools.chain(
            self._model.named_parameters(remove_duplicate=False),
            self._model.named_buffers(remove_duplicate=False),
        ):
            index = self._param_indices.get(name)
            if index is None:
                if any(prefix in name for prefix in self._table_prefixes):
                    raise RuntimeError(
                        f"no flat_state leaf for table parameter {name!r}; the "
                        "compiled graph state and the module have diverged, and "
                        "a silent skip here restores a randomized table"
                    )
                continue
            yield index, owner, name, False

        for fqn, states in _optimizer_states(self._model, self._optimizers).items():
            for key, value in states.items():
                if not isinstance(value, torch.Tensor):
                    continue
                index = self._state_indices.get((fqn, key))
                if index is None:
                    raise RuntimeError(
                        f"no flat_state leaf for optimizer state {fqn!r}[{key!r}]"
                    )
                yield index, value, f"{fqn}[{key!r}]", True

    def build(self) -> None:
        """Give every state leaf real host storage and bind it. Idempotent."""
        allocated = 0
        for index, owner, _name, zero_when_meta in self._state_leaves():
            if owner.is_meta:
                # set_data cannot go meta->cpu; swapping the TensorImpl keeps the
                # Parameter object (and recir's flat_state binding) alive. zeros
                # for accumulators: a restore that fails before the optimizer
                # state loads must not expose uninitialized memory as a moment.
                make = torch.zeros if zero_when_meta else torch.empty
                _replace_tensor_impl(
                    owner, make(owner.shape, dtype=owner.dtype, device="cpu")
                )
                allocated += owner.numel() * owner.element_size()
            self._adopt(index, owner)

        logger.info(
            "[hostbuf] build: allocated %.2f GB, %d leaves owned",
            allocated / 1e9,
            len(self._owned),
        )

    def check_payload(
        self,
        module_entries: Dict[str, Any],
        optimizer_payloads: Sequence[Tuple[Any, Any]] = (),
    ) -> None:
        """Raise unless every tensor DCP touches IS the buffer ``build`` set up."""
        for fqn, value in module_entries.items():
            self._check_payload_entry(self._param_indices.get(fqn), value, fqn)

        for optimizer, payload in optimizer_payloads:
            if optimizer is None or not payload:
                continue
            for fqn, key, value in _optimizer_payload_leaves(
                optimizer, self._model, payload
            ):
                self._check_payload_entry(
                    self._state_indices.get((fqn, key)), value, f"{fqn}[{key!r}]"
                )

    def _check_payload_entry(self, index: int | None, value: Any, name: str) -> None:
        if index is None or not isinstance(value, (torch.Tensor, DTensor)):
            return
        expected = self._owned.get(index)
        if expected is None:
            return
        local = value.to_local() if isinstance(value, DTensor) else value
        if local.is_meta or local.data_ptr() != expected:
            raise RuntimeError(
                f"the checkpoint payload for {name} is not the host buffer "
                "build() established; a save would upload a tensor the graph "
                "does not compute on, and a load would write where it never reads"
            )

    def _adopt(self, index: int, owner: torch.Tensor) -> None:
        leaf = self._flat_state[index]
        if leaf.shape != owner.shape or leaf.dtype != owner.dtype:
            raise RuntimeError(
                f"flat_state[{index}] does not match its owner: "
                f"{tuple(leaf.shape)}/{leaf.dtype} vs "
                f"{tuple(owner.shape)}/{owner.dtype}"
            )
        # shard_params leaves the graph leaf and the module object as two
        # TensorImpls over one storage; point the graph at the object every
        # writer (dcp.load, the flush) reaches.
        self._flat_state[index] = owner.detach()
        self._owned[index] = owner.data_ptr()


def load_module_state_without_self_copy(
    module: torch.nn.Module, state_dict: Dict[str, Any]
) -> None:
    """``module.load_state_dict`` minus the entries that are already in place.

    DCP builds its read destination by calling ``state_dict()``, so ``dcp.load``
    has already written the bytes straight into the live Parameter's storage
    by the time we get here. Handing that value back to ``nn.Module.load_state_dict``
    makes it run ``param.copy_(input_param)`` with source and destination over one storage:
    ATen only short-circuits on ``self.is_same(src)``, TensorImpl identity
    (``aten/src/ATen/native/Copy.cpp``), and these are two impls sharing one
    buffer -- so it memcpys 256 GB onto itself on every restore.

    ``assign=True`` would also avoid the copy and is the wrong tool: it rebinds
    the Parameter to the incoming tensor, and the compiled graph's ``flat_state``
    holds the old object. Dropping the entry is what leaves identity alone.

    Strictness is preserved rather than traded away: the aliased names are the
    only ones allowed to come back as missing.
    """
    local = unwrap_after_load(state_dict)
    live: Dict[str, torch.Tensor] = {
        **dict(module.named_parameters()),
        **dict(module.named_buffers()),
    }

    def already_written(name: str, value: Any) -> bool:
        target = live.get(name)
        return (
            isinstance(value, torch.Tensor)
            and target is not None
            and not target.is_meta
            and not value.is_meta
            and target.shape == value.shape
            and target.data_ptr() == value.data_ptr()
        )

    aliased = {name for name, value in local.items() if already_written(name, value)}
    incompatible = module.load_state_dict(
        {k: v for k, v in local.items() if k not in aliased}, strict=False
    )
    missing = [k for k in incompatible.missing_keys if k not in aliased]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"checkpoint does not match the model: missing={sorted(missing)} "
            f"unexpected={sorted(incompatible.unexpected_keys)}"
        )


class RecirCheckpointable(Stateful):
    """DCP ``Stateful`` view of a ``RecirXLACompiledModule`` for the in_graph
    (TensorCore) backend.

    Wraps the compiled module so ``dcp.save``/``dcp.load`` capture the full model
    state as host torch tensors: the ``_sharded_tables`` are exposed as
    ``Shard(0)`` DTensors over a ``num_processes`` mesh (so each host saves/loads
    exactly its row block instead of DCP deduping them to rank 0); dense params,
    the replicated ``_dp_tables``, and buffers pass through as plain tensors
    (DCP dedups the replicated ones correctly). The WarmupController iter is
    carried under ``WARMUP_ITER_KEY`` and reseeded on load.

    Restore prerequisites (force-compile + host materialization) are handled by
    ``RecirXLACompiledModule.prepare_for_restore()``, which the CheckpointManager
    calls before ``dcp.load`` builds the target dict from ``state_dict()``.
    """

    def __init__(
        self,
        # Typed Any: RecirXLACompiledModule adds custom methods (flush_host_state,
        # etc.) that aren't on nn.Module, so pyre would resolve them via
        # nn.Module.__getattr__ to Tensor and reject the calls.
        wrapper: Any,
        *,
        num_processes: int,
        sharded_prefixes: Sequence[str],
        table_prefixes: Sequence[str],
        warmup: Any,
        host_buffers: Any = None,
        dense_optimizer: Any = None,
        embedding_optimizer: Any = None,
        shard_order: Sequence[int] | None = None,
    ) -> None:
        self._wrapper = wrapper
        self._num_processes = num_processes
        self._sharded_prefixes = tuple(sharded_prefixes)
        self._table_prefixes = tuple(table_prefixes)
        self._warmup = warmup
        self._host_buffers = host_buffers
        self._dense_optimizer = dense_optimizer
        self._embedding_optimizer = embedding_optimizer
        self._shard_order: List[int] | None = (
            None if shard_order is None else list(shard_order)
        )
        self._mesh_cache: DeviceMesh | None = None

    def shard_order(self) -> List[int]:
        order = self._shard_order
        if order is None:
            raise RuntimeError(
                "RecirCheckpointable needs a shard_order to number the "
                "checkpoint's row blocks (see gather_shard_order); numbering "
                "them by torch rank instead writes blocks that only the job "
                "that saved them can read back"
            )
        return order

    def _mesh(self) -> DeviceMesh:
        mesh = self._mesh_cache
        if mesh is None:
            # Cached: DeviceMesh materializes a process group, and state_dict()
            # runs on every checkpoint.
            mesh = DeviceMesh(
                "cpu", torch.tensor(self.shard_order(), dtype=torch.int64)
            )
            self._mesh_cache = mesh
        return mesh

    def state_dict(self) -> Dict[str, Any]:
        t_start = time.perf_counter()
        buffers = self._host_buffers() if self._host_buffers is not None else None
        # Full sync so the on-device embedding tables are mirrored back to host,
        # not just the dense state.
        self._wrapper.flush_host_state(full=True)
        t_flush = time.perf_counter()
        # Sanitize every value to a plain host tensor (drops any torchax
        # jax.Array/env refs that torch.save/DCP can't pickle).
        sd: Dict[str, Any] = {}
        for key, value in dict(self._wrapper.state_dict()).items():
            if _is_sharded_table_key(key, self._table_prefixes):
                # Per-step host sync skips table prefixes, so this large host
                # mirror remains immutable until the next checkpoint. The
                # manager permits only one in-flight save before another flush.
                sd[key] = _to_plain_host_tensor(value)
            else:
                sd[key] = pytree.tree_map(_freeze_checkpoint_value, value)
        t_sanitize = time.perf_counter()
        # Single host: tables are not row-sharded, so plain tensors are correct
        # (and a DTensor mesh would need a process group we don't init).
        mesh = self._mesh() if self._num_processes > 1 else None
        if mesh is not None:
            sd = wrap_sharded_for_save(sd, self._sharded_prefixes, mesh)
        t_wrap = time.perf_counter()
        if self._dense_optimizer is not None:
            sd[OPTIMIZER_STATE_KEY] = pytree.tree_map(
                _freeze_checkpoint_value, self._dense_optimizer.state_dict()
            )
        if self._embedding_optimizer is not None:
            sd[EMBEDDING_OPTIMIZER_STATE_KEY] = _embedding_optimizer_state_for_save(
                self._embedding_optimizer,
                self._wrapper,
                self._sharded_prefixes,
                mesh,
            )
        sd[WARMUP_ITER_KEY] = int(self._warmup.iter)
        if mesh is not None:
            sd[SHARD_ORDER_KEY] = list(self.shard_order())
        # Everything DCP is about to upload is now assembled. On a restore this
        # same dict is what it will write into, so this is the one place both
        # directions can be checked against the buffers.
        if buffers is not None:
            buffers.check_payload(
                sd,
                [
                    (self._dense_optimizer, sd.get(OPTIMIZER_STATE_KEY)),
                    (
                        self._embedding_optimizer,
                        sd.get(EMBEDDING_OPTIMIZER_STATE_KEY),
                    ),
                ],
            )
        logger.info(
            "[stage] state_dict %.1fs = flush %.1fs + sanitize %.1fs + "
            "wrap %.1fs + opt %.1fs",
            time.perf_counter() - t_start,
            t_flush - t_start,
            t_sanitize - t_flush,
            t_wrap - t_sanitize,
            time.perf_counter() - t_wrap,
        )
        return sd

    def make_async_stager(self) -> Any:
        """Avoid DCP's full CPU deepcopy after ``state_dict`` has frozen state."""
        return _RecirAsyncStager()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        warmup_iter = int(state_dict.get(WARMUP_ITER_KEY, 0))
        opt_sd = state_dict.get(OPTIMIZER_STATE_KEY, None)
        embedding_opt_sd = state_dict.get(EMBEDDING_OPTIMIZER_STATE_KEY, None)
        saved_order = state_dict.get(SHARD_ORDER_KEY, None)
        if saved_order is not None:
            logger.info(
                "[checkpoint] shard order at save %s, at restore %s",
                list(saved_order),
                self.shard_order(),
            )
        body = {
            k: v
            for k, v in state_dict.items()
            if k
            not in (
                WARMUP_ITER_KEY,
                OPTIMIZER_STATE_KEY,
                EMBEDDING_OPTIMIZER_STATE_KEY,
                SHARD_ORDER_KEY,
            )
        }
        load_module_state_without_self_copy(self._wrapper, body)
        if opt_sd is not None and self._dense_optimizer is not None:
            _load_optimizer_state_preserving_lr(self._dense_optimizer, opt_sd)
        if embedding_opt_sd is not None and self._embedding_optimizer is not None:
            _load_optimizer_state_preserving_lr(
                self._embedding_optimizer, embedding_opt_sd
            )
        # DCP has written, and `Optimizer.load_state_dict` has had its chance to
        # swap slot objects. Everything just written has to still be the buffer
        # the graph reads, or the restore landed somewhere nothing looks.
        if self._host_buffers is not None:
            self._host_buffers().check_payload(
                body,
                [
                    (self._dense_optimizer, opt_sd),
                    (self._embedding_optimizer, embedding_opt_sd),
                ],
            )
        reseed_warmup(self._warmup, warmup_iter)
