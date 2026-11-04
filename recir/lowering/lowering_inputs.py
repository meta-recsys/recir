# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Lowering every graph a backend's dispatch holds, one input at a time."""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, cast, Protocol

from recir.composite import CompositeModule, Part
from recir.lowering.lowering_utils import CarvedRegion, LoweringHooks
from torch import fx, nn

logger: logging.Logger = logging.getLogger(__name__)

WHOLE = Part("whole")


class Lower(Protocol):
    """A backend's lowering of one graph, which reports its decisions to ``hooks`` when given any."""

    def __call__(
        self, gm: fx.GraphModule, /, *, hooks: LoweringHooks | None = None
    ) -> nn.Module: ...


class LoweringObserver:
    """
    Base class for all lowering observers: told what lowering decides, input by input.

    This class is responsible for containing faults: every event reaches a
    subclass hook through one guard, so an observer cannot fail lowering. An
    observer must not edit what lowering returns, except to set
    ``compile_scope`` on a carved ``LazilyBuiltModule``. This can be extended
    by subclassing and overriding the ``_on_*`` hooks.
    """

    def lowering(self, part: Part, gm: fx.GraphModule) -> _InputHooks:
        """
        Report that ``gm``, the input ``part``, is about to be lowered; returns
        the hooks to lower it with, whose ``lowered()`` reports what lowering
        returned.
        """
        self._guarded("input", self._on_input, part, gm)
        return _InputHooks(self, part)

    def _on_input(self, part: Part, gm: fx.GraphModule) -> None:
        """Lowering ``gm``, the input ``part``, is about to start."""

    def _on_carved(self, part: Part, region: CarvedRegion) -> None:
        """``region`` was carved from the input ``part``."""

    def _on_lowered(self, part: Part, lowered: nn.Module) -> None:
        """Lowering the input ``part`` returned ``lowered``."""

    def _guarded(self, event: str, hook: Callable[..., None], *args: Any) -> None:
        try:
            hook(*args)
        except Exception as e:
            logger.warning("lowering observer: %s failed: %r", event, e, exc_info=True)


class _InputHooks:
    def __init__(self, observer: LoweringObserver, part: Part) -> None:
        self.observer = observer
        self.part = part

    def on_carved(self, region: CarvedRegion) -> None:
        self.observer._guarded("carved", self.observer._on_carved, self.part, region)

    def lowered(self, lowered: nn.Module) -> None:
        self.observer._guarded("lowered", self.observer._on_lowered, self.part, lowered)


def _lower_input(
    lower: Lower, observer: LoweringObserver | None, part: Part, gm: fx.GraphModule
) -> nn.Module:
    if observer is None:
        return lower(gm, hooks=None)
    hooks = observer.lowering(part, gm)
    lowered = lower(gm, hooks=hooks)
    hooks.lowered(lowered)
    return lowered


def lower_inputs(
    gm: fx.GraphModule | CompositeModule,
    lower: Lower,
    observer: LoweringObserver | None = None,
    *,
    walk_composites: bool = True,
) -> nn.Module:
    """
    Lower each input of ``gm`` with ``lower`` and put the results back: each
    part of a ``CompositeModule``, or ``gm`` itself as the ``whole`` part.
    ``lower`` gets hooks reporting to ``observer`` for the input, or None
    without an observer. Without ``walk_composites``, a composite module is
    handed to ``lower`` whole and unobserved, for a backend that does not
    lower composites. Returns the lowered module.
    """
    if isinstance(gm, fx.GraphModule):
        return _lower_input(lower, observer, WHOLE, gm)
    if not walk_composites:
        return lower(cast(Any, gm), hooks=None)
    return gm.lower_parts(functools.partial(_lower_input, lower, observer))
