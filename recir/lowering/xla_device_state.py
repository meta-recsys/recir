# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Optional, Sequence

import torch
from recir.lowering.device_state import DeviceStateManager
from recir.lowering.xla_lower import _jax_to_torch, _XLAModule


class XLADeviceStateManager(DeviceStateManager):
    @staticmethod
    def _set_keep_jax_output(gm: torch.nn.Module, value: bool) -> None:
        for m in gm.modules():
            if isinstance(m, _XLAModule):
                m.keep_jax_output = value

    @staticmethod
    def _convert_flat_state_to_jax(
        flat_state: List[torch.Tensor],
    ) -> List[Any]:
        """Convert flat_state tensors from PyTorch to JAX arrays on TPU.

        For inference, state tensors (model weights) are passed
        through the parent FX graph:

        They enter _XLAModule as inputs but are not returned as outputs.

        This means process_outputs can never capture them as JAX arrays —
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
                    # alloc / H2D). After the first step, process_outputs caches the
                    # materialized jax.Array, so this only matters on the first call.
                    jax_state.append(t)
                else:
                    jax_state.append(jax.device_put(t2j(t), jax_device))
            else:
                jax_state.append(t)
        return jax_state

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
        # For training, process_outputs overwrites device_state with updated
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
        # process_outputs caches the correctly-sharded JAX device_state so the
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
            # until process_outputs completes: if execution raises, the cached
            # pre-call state remains authoritative and a retry can refresh the
            # host mirror again.
            self.sync_to_torch(flat_state)
        device_state = self.device_state
        self._used_fast_path = device_state is not None and not last_train_step
        if self._used_fast_path:
            # Only the fast path consumes the cached JAX device_state. The slow
            # path (last_train_step, or the first multi-device call) returns
            # flat_state and may legitimately have device_state is None.
            assert device_state is not None
            self._set_keep_jax_output(gm, True)
            return device_state + user_inps
        else:
            self._set_keep_jax_output(gm, False)
            return flat_state + user_inps

    def process_outputs(
        self,
        flat_state: List[torch.Tensor],
        output: Any,
        updated_flat: List[Any],
        last_train_step: bool,
    ) -> torch.Tensor:
        num_state = len(flat_state)
        if not self._used_fast_path:
            self._sync_flat_state(flat_state, updated_flat)
        if last_train_step:
            self.device_state = None
        else:
            self.device_state = updated_flat[:num_state]
        # Phase-3 pipelining: optionally skip the per-step output device->host
        # materialization so the caller can pull it on a cadence (lets JAX run the
        # host ahead, overlapping the next step's H2D/dispatch with this step's
        # compute). On the last step we always materialize so callers get a torch
        # result. device_state above is unaffected (state still stays on-device).
        if self.keep_output_jax and not last_train_step:
            return output
        if not isinstance(output, torch.Tensor) and output is not None:
            output = _jax_to_torch(output, self.torch_device)
        return output

    def materialize_output(self, output: Any) -> Any:
        """Convert a deferred (JAX) step output to torch. Passes through values
        that are already torch (or tuples/lists thereof). Used by callers that set
        ``keep_output_jax=True`` and pull the output on a cadence."""
        if output is None:
            return None
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, (list, tuple)) and all(
            isinstance(x, torch.Tensor) or x is None for x in output
        ):
            return output
        return _jax_to_torch(output, self.torch_device)

    def _sync_flat_state(
        self,
        flat_state: List[torch.Tensor],
        updated_flat: List[Any],
    ) -> None:
        """Copy graph outputs back to flat_state tensors."""
        for i, state_tensor in enumerate(flat_state):
            if i < len(updated_flat):
                updated = updated_flat[i]
                if isinstance(updated, torch.Tensor):
                    if updated.data_ptr() != state_tensor.data_ptr():
                        assert updated.shape == state_tensor.shape
                        state_tensor.copy_(updated)
                else:
                    state_tensor.copy_(_jax_to_torch(updated, self.torch_device))

    def sync_to_torch(
        self,
        flat_state: List[torch.Tensor],
        skip_fqns: Optional[Sequence[str]] = None,
    ) -> None:
        device_state = self.device_state
        if device_state is None:
            return
        fqns = self.state_fqns
        with torch.no_grad():
            for i, jax_val in enumerate(device_state):
                if i >= len(flat_state):
                    continue
                # skip_fqns lets callers avoid the per-step device->host copy +
                # transpose of named state entries (e.g. the on-device embedding-
                # table shards). Those are updated in-graph and only need a host
                # copy at checkpoint time; copying them every step is O(table
                # size) and dominates the step for large tables. Matched as a
                # substring of the entry's FQN so a table prefix skips the table
                # and (if present) its optimizer state.
                if (
                    skip_fqns
                    and fqns is not None
                    and i < len(fqns)
                    and any(pat in fqns[i] for pat in skip_fqns)
                ):
                    continue
                flat_state[i].copy_(_jax_to_torch(jax_val, self.torch_device))
