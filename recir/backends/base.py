# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Backend ABC for the recir compile pipeline.

A ``Backend`` instance owns stages 6-10 of the compile pipeline:

- Stage 6: ``optimize_passes`` — graph optimization passes applied before
  dispatch.
- Stage 7: ``uses_iter_graph`` — whether the graph should be wrapped in an
  ``IterGraphModule`` for cross-iteration optimizations.
- Stage 8: ``dispatch`` — lower the graph to executable form (Inductor, XLA,
  etc.).
- Stage 9: build — lazy on first forward of the dispatch module.
- Stage 10: ``serialize_artifact`` — persist a compiled artifact for AOT
  reuse.

The recir compile entry point asks the backend questions (via these methods);
it does not switch on backend type.  Concrete backends live in
``recir/backends/xla.py`` and ``recir/backends/inductor.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Sequence, TYPE_CHECKING

from recir.lowering.device_state import DeviceStateManager
from torch import fx, nn

if TYPE_CHECKING:
    from recir.distributed.parallel_mode import ParallelMode


# Type alias mirroring the ``GraphPatch`` callable shape used by passes that
# rewrite a graph module in place (e.g. ``comm_fusion_with_concat``).
GraphPatch = Callable[..., Any]


class Backend(ABC):
    """Owns stages 6-10 of the recir compile pipeline.

    Subclasses declare their default optimization passes via
    :meth:`default_optimize_passes`, decide whether the graph needs
    ``IterGraphModule`` wrapping via :attr:`uses_iter_graph`, and implement
    :meth:`dispatch` to lower the graph to executable form.

    Args:
        optimize_passes: optional override for stage 6.  When ``None`` (the
            default), :meth:`default_optimize_passes` is used.  Passing an
            explicit sequence (including ``[]``) fully replaces the defaults.
    """

    def __init__(
        self,
        *,
        optimize_passes: Optional[Sequence[GraphPatch]] = None,
        **kwargs: Any,
    ) -> None:
        self._optimize_passes: Sequence[GraphPatch] = (
            optimize_passes
            if optimize_passes is not None
            else self.default_optimize_passes()
        )

    @classmethod
    def default_optimize_passes(cls) -> Sequence[GraphPatch]:
        """Subclasses override.  Default: no passes."""
        return ()

    @property
    def optimize_passes(self) -> Sequence[GraphPatch]:
        """Frozen at construction time.  Used in stage 6."""
        return self._optimize_passes

    @property
    def uses_iter_graph(self) -> bool:
        """Stage 7: whether to wrap the graph in an ``IterGraphModule``."""
        return False

    @abstractmethod
    def dispatch(self, gm: fx.GraphModule) -> nn.Module:
        """Stage 8: lower ``gm`` to an executable ``nn.Module``."""

    def device_state_manager(self) -> DeviceStateManager:
        """Return the device-state manager that mediates runtime I/O."""
        return DeviceStateManager()

    def validate_with(self, parallel_mode: "ParallelMode") -> None:
        """Raise if ``parallel_mode`` is incompatible with this backend."""
        return None

    def serialize_artifact(self, compiled: nn.Module, path: str) -> None:
        """Stage 10: persist a compiled artifact for AOT reuse."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement serialize_artifact"
        )
