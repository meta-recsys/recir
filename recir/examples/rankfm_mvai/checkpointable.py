#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Checkpoint helpers for the recir in_graph (TensorCore) TPU model.

The in_graph embedding tables live as torch ``nn.Parameter``s, so the whole
checkpoint can stay in the ``torch.distributed.checkpoint`` (DCP) world. But the
``_sharded_tables`` are per-process ``Shard(0)`` row blocks stored as *plain*
Parameters with no sharding metadata: DCP would treat them as replicated and
dedup them to rank 0, silently dropping every other host's rows. We wrap those
entries as ``DTensor``s so DCP saves/loads each host's shard; all other
(replicated) entries pass through unchanged and DCP dedups them correctly.
"""

from typing import Any, cast, Dict, Sequence

import torch
import torch.utils._pytree as pytree
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard

# Key under which the WarmupController iter is persisted inside the model
# checkpoint (it lives in no nn.Module state_dict, so we carry it explicitly).
WARMUP_ITER_KEY = "__recir_warmup_iter__"
# Key under which the dense optimizer state_dict is nested inside the model
# checkpoint (so it too is sanitized to plain host tensors before DCP writes it).
OPTIMIZER_STATE_KEY = "__recir_optimizer__"


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
    warmup.apply()
    for _ in range(iteration):
        warmup.step()


def _is_sharded_table_key(key: str, sharded_prefixes: Sequence[str]) -> bool:
    return any(prefix in key for prefix in sharded_prefixes)


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
    """Return a copy of ``state_dict`` with any ``DTensor`` value replaced by its
    local shard tensor; non-DTensor entries pass through unchanged."""
    out: Dict[str, Any] = {}
    for key, value in state_dict.items():
        out[key] = value.to_local() if isinstance(value, DTensor) else value
    return out


def materialize_meta_params(
    module: torch.nn.Module, table_prefixes: Sequence[str]
) -> None:
    """Give real host storage to the in_graph embedding-table params.

    The in_graph tables (``_sharded_tables`` / ``_dp_tables``) are ``on_device_init``
    meta tensors; ``copy_`` into a meta tensor is a silent no-op, so both a full
    device->host flush (save) and ``dcp.load`` (restore) write nothing until the
    param has real storage. After the compile shape-shards a table to
    ``[V/num_processes, D]`` (still meta), we reassign ``.data`` IN PLACE (same
    Parameter object, so the compiled runner's ``flat_state`` binding is
    preserved) to an empty CPU tensor of the sharded shape. Idempotent: params
    that already have real storage are skipped.
    """
    for name, param in module.named_parameters():
        if not any(prefix in name for prefix in table_prefixes):
            continue
        if param.is_meta:
            # `param.data = torch.empty(..., device="cpu")` fails: a meta tensor
            # and a CPU tensor are shallow-copy-incompatible types (meta is not a
            # "dense backend"), so set_data raises. swap_tensors swaps the
            # underlying TensorImpl in place, preserving the Parameter object
            # identity (and thus recir's flat_state binding), and works meta->cpu.
            torch.utils.swap_tensors(
                param, torch.empty(param.shape, dtype=param.dtype, device="cpu")
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
        optimizer: Any = None,
    ) -> None:
        self._wrapper = wrapper
        self._num_processes = num_processes
        self._sharded_prefixes = tuple(sharded_prefixes)
        self._table_prefixes = tuple(table_prefixes)
        self._warmup = warmup
        # The dense optimizer is carried inside this checkpointable (nested +
        # sanitized) rather than as a separate DCP entry, so its tensors are also
        # converted to plain host tensors (a torchax optimizer-state tensor would
        # otherwise break torch.save; see _to_plain_host_tensor).
        self._optimizer = optimizer

    def _mesh(self) -> DeviceMesh:
        return DeviceMesh("cpu", torch.arange(self._num_processes))

    def state_dict(self) -> Dict[str, Any]:
        # Tables are meta until materialized; do it before the full flush so the
        # device->host copy actually lands (see materialize_meta_params).
        materialize_meta_params(self._wrapper, self._table_prefixes)
        # Full sync so the on-device embedding tables are mirrored back to host,
        # not just the dense state.
        self._wrapper.flush_host_state(full=True)
        # Sanitize every value to a plain host tensor (drops any torchax
        # jax.Array/env refs that torch.save/DCP can't pickle).
        sd: Dict[str, Any] = pytree.tree_map(
            _to_plain_host_tensor, dict(self._wrapper.state_dict())
        )
        # Single host: tables are not row-sharded, so plain tensors are correct
        # (and a DTensor mesh would need a process group we don't init).
        if self._num_processes > 1:
            sd = wrap_sharded_for_save(sd, self._sharded_prefixes, self._mesh())
        if self._optimizer is not None:
            sd[OPTIMIZER_STATE_KEY] = pytree.tree_map(
                _to_plain_host_tensor, self._optimizer.state_dict()
            )
        sd[WARMUP_ITER_KEY] = int(self._warmup.iter)
        return sd

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        warmup_iter = int(state_dict.get(WARMUP_ITER_KEY, 0))
        opt_sd = state_dict.get(OPTIMIZER_STATE_KEY, None)
        body = {
            k: v
            for k, v in state_dict.items()
            if k not in (WARMUP_ITER_KEY, OPTIMIZER_STATE_KEY)
        }
        self._wrapper.load_state_dict(unwrap_after_load(body))
        if opt_sd is not None and self._optimizer is not None:
            self._optimizer.load_state_dict(opt_sd)
        reseed_warmup(self._warmup, warmup_iter)
