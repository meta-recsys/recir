# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from recir.lowering.device_state import DeviceStateManager
from recir.lowering.xla_lower import (
    _copy_jax_addressable_shards_to_torch,
    _get_global_devices,
    _get_local_devices,
    _GSPMD_FSDP_AXIS,
    _gspmd_mesh,
    _jax_to_torch,
    _StateInputCache,
    _XLAModule,
)

logger: logging.Logger = logging.getLogger(__name__)


class XLADeviceStateManager(DeviceStateManager):
    """Keep XLA state resident and optionally share it across compiled graphs."""

    _shared: Optional[Dict[str, Any]] = None
    _shared_keys: List[str]
    _shared_host: List[torch.Tensor]

    def __init__(self, torch_device: Optional[torch.device] = None) -> None:
        super().__init__(
            torch_device if torch_device is not None else torch.device("cpu")
        )
        self._state_input_cache: _StateInputCache = {}

    def share_device_state(
        self,
        store: Dict[str, Any],
        compiled: torch.nn.Module,
        flat_state: List[torch.Tensor],
    ) -> None:
        state_fqns = self.state_fqns
        if state_fqns is None or len(state_fqns) != len(flat_state):
            return
        # A graph left entirely in torch (too small to lower) reads host tensors.
        if not any(isinstance(m, _XLAModule) for m in compiled.modules()):
            return
        local = self.device_state
        self._shared = store
        self._shared_keys = list(state_fqns)
        self._shared_host = flat_state
        if local is not None:
            self.device_state = local

    @property
    def device_state(self) -> Optional[List[Any]]:
        shared = self._shared
        if shared is None:
            return self._device_state
        if not any(key in shared for key in self._shared_keys):
            return None
        # A leaf no graph has made resident yet is its host tensor, which the
        # XLA call converts like on the slow path.
        return [
            shared.get(key, host)
            for key, host in zip(self._shared_keys, self._shared_host)
        ]

    @device_state.setter
    def device_state(self, value: Optional[List[Any]]) -> None:
        shared = self._shared
        if shared is None:
            self._device_state = value
            return
        if value is None:
            for key in self._shared_keys:
                shared.pop(key, None)
            return
        for key, array in zip(self._shared_keys, value):
            shared[key] = array

    def _configure_xla_modules(
        self, gm: torch.nn.Module, *, keep_jax_output: bool
    ) -> None:
        for m in gm.modules():
            if isinstance(m, _XLAModule):
                m._bind_state_input_cache(self._state_input_cache)
                m.keep_jax_output = keep_jax_output

    def _bind_state_input_cache(self, gm: torch.nn.Module) -> None:
        for m in gm.modules():
            if isinstance(m, _XLAModule):
                m._bind_state_input_cache(self._state_input_cache)

    def invalidate_device_state(self) -> None:
        """Discard manager-owned state and lowered-module input copies."""
        super().invalidate_device_state()
        self._state_input_cache.clear()

    @staticmethod
    def _convert_flat_state_to_jax(
        flat_state: List[torch.Tensor],
    ) -> List[Any]:
        """Convert flat_state tensors from PyTorch to JAX arrays on TPU.

        For inference, state tensors (model weights) are passed
        through the parent FX graph:

        They enter _XLAModule as inputs but are not returned as outputs.

        This means commit_step_state can never capture them as JAX arrays —
        they always remain whatever type was in flat_inps, i.e. torch.Tensor.

        Converting them here, once, and caching in
        device_state ensures _XLAModule._t2j_safe / _to_sharded_jax sees
        jax.Array on subsequent calls and skips the per-tensor dispatch.
        """
        import jax  # @manual
        from recir.torchax.ops.mappings import t2j  # @manual

        jax_device = jax.local_devices()[0]
        jax_state: List[Any] = []
        for t in flat_state:
            if isinstance(t, torch.Tensor):
                if t.is_meta:
                    # On-device-init params (FSDP / seq-embedding tables) are meta
                    # tensors with no host data, so t2j() -> t.cpu() would raise
                    # "Cannot copy out of meta tensor". Leave them unconverted:
                    # _XLAModule._to_sharded_jax has a meta branch that generates
                    # each device's shard directly in HBM via jax.random (no host
                    # alloc / H2D). After the first step, commit_step_state
                    # caches the
                    # materialized jax.Array, so this only matters on the first call.
                    jax_state.append(t)
                else:
                    jax_state.append(jax.device_put(t2j(t), jax_device))
            else:
                jax_state.append(t)
        return jax_state

    def prefill_device_state_from_host(
        self,
        gm: torch.fx.GraphModule,
        flat_state: List[torch.Tensor],
    ) -> None:
        """Convert every ``flat_state`` leaf to its device layout and cache it."""

        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        if len(placeholders) < len(flat_state):
            raise RuntimeError(
                "compiled graph has fewer placeholders than flat_state entries: "
                f"{len(placeholders)} < {len(flat_state)}"
            )
        meta_leaves = [i for i, t in enumerate(flat_state) if t.is_meta]
        if meta_leaves:
            named = ", ".join(
                f"{i} ({self.state_fqns[i]})"
                if self.state_fqns and i < len(self.state_fqns)
                else str(i)
                for i in meta_leaves
            )
            raise RuntimeError(
                f"flat_state entries [{named}] are still meta; seeding would "
                "regenerate them from jax.random instead of using the checkpoint"
            )

        devices = _get_global_devices()
        fsdp_axis_size = int(
            getattr(gm, "meta", {}).get("gspmd_fsdp_axis_size", len(devices))
        )
        mesh, batch_axis_resource, _ = _gspmd_mesh(devices, fsdp_axis_size)
        fsdp_host_shard_count = fsdp_axis_size // len(_get_local_devices())
        new_device_state = [
            _XLAModule._to_sharded_jax(
                leaf,
                mesh,
                # State entries are never batch inputs.
                False,
                0,
                is_fsdp_tensor=placeholders[i].meta.get("is_fsdp_sharded", False),
                batch_axis_resource=batch_axis_resource,
                fsdp_axis_name=_GSPMD_FSDP_AXIS,
                fsdp_host_shard_count=fsdp_host_shard_count,
            )
            for i, leaf in enumerate(flat_state)
        ]
        # Build the replacement completely before releasing the old state, so
        # a failed conversion leaves the prefill retryable. Once successful,
        # discard any frozen-input copies derived from the prior host state.
        self._bind_state_input_cache(gm)
        self.invalidate_device_state()
        self.device_state = new_device_state

    def prepare_inputs(
        self,
        gm: torch.fx.GraphModule,
        flat_state: List[torch.Tensor],
        user_inps: List[Any],
        last_train_step: bool,
    ) -> List[Any]:
        # First call (device_state is None) and not the last training step:
        # convert state tensors to JAX arrays once and cache. On subsequent
        # calls, _XLAModule._t2j_safe / _to_sharded_jax sees jax.Array and
        # returns immediately, skipping per-tensor dispatch.
        # For training, commit_step_state overwrites device_state with updated
        # weights from the graph output after each step, so this initial
        # conversion is overwritten and does not affect correctness.
        #
        # IMPORTANT: _convert_flat_state_to_jax uses single-device device_put,
        # which is only correct on a single device. On a multi-device GSPMD mesh
        # the compiled program requires per-arg mesh shardings (e.g. replicated
        # P() for DP/dense params), so single-device placement fails the input
        # sharding check. For multi-device we leave device_state unset on the
        # first call and fall through to the slow path; _XLAModule._to_sharded_jax
        # then applies the correct shardings (and the meta-init branch), and
        # commit_step_state caches the correctly-sharded JAX device_state so the
        # fast path is used on subsequent steps.
        import jax  # @manual

        if (
            self.device_state is None
            and not last_train_step
            and jax.device_count() == 1
        ):
            self.device_state = self._convert_flat_state_to_jax(flat_state)
        if last_train_step and self.device_state is not None:
            # The final slow path consumes every PyTorch flat_state leaf, so
            # partial/skip-FQN synchronization is not valid here. Keep the cache
            # until commit_step_state completes: if execution raises, the cached
            # pre-call state remains authoritative and a retry can refresh the
            # host mirror again.
            self.copy_state_to_host(flat_state)
        device_state = self.device_state
        self._used_fast_path = device_state is not None and not last_train_step
        if self._used_fast_path:
            # Only the fast path consumes the cached JAX device_state. The slow
            # path (last_train_step, or the first multi-device call) returns
            # flat_state and may legitimately have device_state is None.
            assert device_state is not None
            self._configure_xla_modules(gm, keep_jax_output=True)
            return device_state + user_inps
        else:
            self._configure_xla_modules(gm, keep_jax_output=False)
            return flat_state + user_inps

    def commit_step_state(
        self,
        flat_state: List[torch.Tensor],
        updated_flat: List[Any],
        last_train_step: bool,
    ) -> None:
        # On the fast path the updated state stays on device and is simply
        # re-cached; only the slow path (first multi-device call, last step)
        # mirrors it host-ward.
        if not self._used_fast_path:
            self._copy_state_into(flat_state, updated_flat)
        if last_train_step:
            # Release both cache layers only after the host mirror is current.
            # If execution or the host copy raises, the pre-call device state
            # and frozen-input copies remain available for a retry.
            self.invalidate_device_state()
        else:
            self.device_state = updated_flat[: len(flat_state)]

    def materialize_output(self, output: Any) -> Any:
        """Convert a deferred (JAX) step output to torch. Passes through values
        that are already torch (or tuples/lists thereof). Used by callers that set
        ``keep_output_jax=True`` and pull the output on a cadence."""
        if output is None:
            return None
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, list):
            return [self.materialize_output(value) for value in output]
        if isinstance(output, tuple):
            return tuple(self.materialize_output(value) for value in output)
        if isinstance(output, dict):
            return {
                key: self.materialize_output(value) for key, value in output.items()
            }
        return _jax_to_torch(output, self.torch_device)

    def _copy_state_into(
        self,
        flat_state: List[torch.Tensor],
        source: Sequence[Any],
        skip_fqns: Optional[Sequence[str]] = None,
        *,
        log_summary: bool = False,
    ) -> None:
        """Copy ``source[i]`` into ``flat_state[i]``, converting from JAX if needed.

        The one device->host copy loop. Its two callers differ only in where the
        values come from: ``commit_step_state`` passes the step's graph outputs,
        ``copy_state_to_host`` passes the cached ``device_state``.

        ``skip_fqns`` lets callers avoid the copy + transpose of named state
        entries (e.g. the on-device embedding-table shards). Those are updated
        in-graph and only need a host copy at checkpoint time; copying them every
        step is O(table size) and dominates the step for large tables. Dense
        params / metric buffers don't match and still sync. Matched as a
        substring of the entry's FQN so a table prefix skips the table and (if
        present) its optimizer state.
        """
        fqns = self.state_fqns
        started = time.perf_counter()
        copied = skipped = 0
        copied_bytes = 0
        from_jax_bytes = 0
        per_entry: List[Tuple[float, str, int, bool]] = []
        with torch.no_grad():
            for i, state_tensor in enumerate(flat_state):
                if i >= len(source):
                    break
                if (
                    skip_fqns
                    and fqns is not None
                    and i < len(fqns)
                    and any(pat in fqns[i] for pat in skip_fqns)
                ):
                    continue
                updated = source[i]
                entry_started = time.perf_counter() if log_summary else 0.0
                was_jax = False
                if isinstance(updated, torch.Tensor):
                    if updated.data_ptr() == state_tensor.data_ptr():
                        if log_summary:
                            skipped += 1
                        continue
                    assert updated.shape == state_tensor.shape
                    state_tensor.copy_(updated)
                else:
                    was_jax = True
                    if getattr(updated, "addressable_shards", None):
                        _copy_jax_addressable_shards_to_torch(updated, state_tensor)
                    else:
                        state_tensor.copy_(_jax_to_torch(updated, self.torch_device))
                if log_summary:
                    nbytes = state_tensor.numel() * state_tensor.element_size()
                    copied += 1
                    copied_bytes += nbytes
                    if was_jax:
                        from_jax_bytes += nbytes
                    name = fqns[i] if fqns is not None and i < len(fqns) else f"#{i}"
                    per_entry.append(
                        (time.perf_counter() - entry_started, name, nbytes, was_jax)
                    )
        if log_summary:
            elapsed = time.perf_counter() - started
            gb = copied_bytes / 1e9
            logger.info(
                f"[flush] full device->host sync: {copied} entries, {gb:.1f} GB "
                f"({from_jax_bytes / 1e9:.1f} GB via jax->numpy->torch), "
                f"{skipped} already-host, {elapsed:.1f}s "
                f"({gb / elapsed if elapsed else 0:.2f} GB/s)"
            )
            for dt, name, nbytes, was_jax in sorted(per_entry, reverse=True)[:8]:
                logger.info(
                    f"[flush]   {dt:7.1f}s {nbytes / 1e9:7.1f} GB "
                    f"{'jax' if was_jax else 'torch':>5} {name}"
                )

    def copy_state_to_host(
        self,
        flat_state: List[torch.Tensor],
        skip_fqns: Optional[Sequence[str]] = None,
    ) -> None:
        """Mirror the cached device state host-ward, on demand.

        Distinct from ``commit_step_state``, which folds THIS step's graph
        outputs in as part of the runner's epilogue. This one is driven by the
        training loop on its own cadence (``sync_device_state_to_torch``) and
        reads the cache.
        """
        device_state = self.device_state
        if device_state is None:
            return
        # Only the full sync is instrumented
        self._copy_state_into(
            flat_state, device_state, skip_fqns, log_summary=skip_fqns is None
        )

    def copy_named_state_to_host(
        self,
        flat_state: List[torch.Tensor],
        fqn_substrings: Sequence[str],
    ) -> None:
        """Copy back a named handful of entries, bypassing the bulk loop."""
        device_state = self.device_state
        fqns = self.state_fqns
        if device_state is None or fqns is None:
            return
        with torch.no_grad():
            for i, state_tensor in enumerate(flat_state):
                if i >= len(device_state) or i >= len(fqns):
                    break
                if not any(pat in fqns[i] for pat in fqn_substrings):
                    continue
                updated = device_state[i]
                if isinstance(updated, torch.Tensor):
                    if updated.data_ptr() != state_tensor.data_ptr():
                        state_tensor.copy_(updated)
                elif getattr(updated, "addressable_shards", None):
                    _copy_jax_addressable_shards_to_torch(updated, state_tensor)
                else:
                    state_tensor.copy_(_jax_to_torch(updated, self.torch_device))
