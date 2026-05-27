# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""XLA backend implementations (GSPMD, ShardMap, Dynamo bridge)."""

from __future__ import annotations

from typing import Any, ClassVar, TYPE_CHECKING

from recir.backend import BackendType
from recir.backends.base import Backend
from recir.lowering.device_state import DeviceStateManager
from recir.lowering.xla_device_state import XLADeviceStateManager
from recir.lowering.xla_lower import partial_lower_xla
from torch import fx, nn

if TYPE_CHECKING:
    from recir.distributed.parallel_mode import ParallelMode


class _XLABackendBase(Backend):
    """Shared logic for the three XLA-flavored backends.

    Each subclass picks the appropriate :class:`BackendType` tag and is
    expected to override :meth:`validate_with`.
    """

    # pyre-ignore[13]: Initialized in subclasses.
    _backend_type: ClassVar[BackendType]

    def __init__(self, *, dump_hlo: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._dump_hlo = dump_hlo

    @property
    def uses_iter_graph(self) -> bool:
        return True

    def dispatch(self, gm: fx.GraphModule) -> nn.Module:
        return partial_lower_xla(gm, backend_type=self._backend_type)

    def device_state_manager(self) -> DeviceStateManager:
        return XLADeviceStateManager()


class XLAGSPMDBackend(_XLABackendBase):
    """XLA backend using GSPMD auto-parallelism.

    Requires a :class:`GSPMDParallel` parallel mode.  XLA performs all of the
    graph-level optimization passes natively, so :meth:`default_optimize_passes`
    returns an empty tuple.
    """

    _backend_type: ClassVar[BackendType] = BackendType.XLA_GSPMD

    def validate_with(self, parallel_mode: "ParallelMode") -> None:
        from recir.distributed.parallel_mode import GSPMDParallel

        if not isinstance(parallel_mode, GSPMDParallel):
            raise ValueError(
                f"XLAGSPMDBackend requires GSPMDParallel, got "
                f"{type(parallel_mode).__name__}"
            )


class XLAShardMapBackend(_XLABackendBase):
    """XLA backend that maps ``_c10d_functional`` collectives to JAX via shard_map.

    Requires an explicit-collective parallel mode (DataParallel or
    HybridParallel — anything that retains the collectives in the FX graph
    instead of relying on GSPMD).
    """

    _backend_type: ClassVar[BackendType] = BackendType.XLA_SHARD_MAP

    def validate_with(self, parallel_mode: "ParallelMode") -> None:
        from recir.distributed.parallel_mode import (
            DataParallel,
            GSPMDParallel,
            HybridParallel,
        )

        if isinstance(parallel_mode, GSPMDParallel):
            raise ValueError(
                "XLAShardMapBackend is incompatible with GSPMDParallel; "
                "use XLAGSPMDBackend instead."
            )
        if not isinstance(parallel_mode, (DataParallel, HybridParallel)):
            raise ValueError(
                f"XLAShardMapBackend requires DataParallel or HybridParallel, "
                f"got {type(parallel_mode).__name__}"
            )


class XLADynamoBridgeBackend(_XLABackendBase):
    """XLA backend using ``torch_xla.core.dynamo_bridge`` for lazy-tensor lowering.

    Accepts any non-GSPMD parallel mode.  The dynamo bridge handles its own
    sharding via the underlying lazy-tensor stack.
    """

    _backend_type: ClassVar[BackendType] = BackendType.XLA_DYNAMO_BRIDGE

    def validate_with(self, parallel_mode: "ParallelMode") -> None:
        from recir.distributed.parallel_mode import GSPMDParallel

        if isinstance(parallel_mode, GSPMDParallel):
            raise ValueError(
                "XLADynamoBridgeBackend is incompatible with GSPMDParallel; "
                "use XLAGSPMDBackend instead."
            )
