# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Lowering every graph a backend's dispatch holds, one input at a time."""

from __future__ import annotations

import functools
from typing import Callable, Protocol

from recir.composite import CompositeModule, Part
from recir.lowering.lowering_utils import ExtractedSubgraph
from recir.observability.lowering_observer import SnapshotLoweringObserver
from torch import fx, nn

WHOLE = Part("whole")


class Lower(Protocol):
    """A backend's lowering of one graph, which calls ``on_subgraph_extracted`` with each subgraph it extracts."""

    def __call__(
        self,
        gm: fx.GraphModule,
        /,
        *,
        on_subgraph_extracted: Callable[[ExtractedSubgraph], None] | None = None,
    ) -> nn.Module: ...


def _lower_input(
    lower: Lower,
    observer: SnapshotLoweringObserver | None,
    part: Part,
    gm: fx.GraphModule,
) -> nn.Module:
    if observer is None:
        return lower(gm)
    return lower(
        gm, on_subgraph_extracted=functools.partial(observer.subgraph_extracted, part)
    )


def lower_inputs(
    gm: fx.GraphModule | CompositeModule,
    lower: Lower,
    observer: SnapshotLoweringObserver | None = None,
) -> nn.Module:
    """
    Lower each input of ``gm`` with ``lower`` and put the results back: each
    part of a ``CompositeModule``, or ``gm`` itself as the ``whole`` part.
    ``observer``, when given, is told of each subgraph lowering extracts, with
    the input it came from. Returns the lowered module.
    """
    if isinstance(gm, fx.GraphModule):
        return _lower_input(lower, observer, WHOLE, gm)
    return gm.lower_parts(functools.partial(_lower_input, lower, observer))
