# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""RegionTransform ABC — base class for graph-level region transforms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Callable

import torch.fx as fx
import torch.nn as nn


class RegionTransform(ABC):
    """Base class for a graph-level region transform.

    Subclasses define:
    - detect_regions: which subgraphs to target
    - wrap_region: how to wrap each subgraph (the autograd.Function)
    """

    @abstractmethod
    def detect_regions(
        self,
        gm: fx.GraphModule,
        called_modules: OrderedDict[str, tuple[nn.Module, tuple[Any, ...]]],
    ) -> list[set[fx.Node]]:
        """Return a list of node sets, each representing a region to wrap."""
        ...

    @abstractmethod
    def wrap_region(
        self,
        region_gm: fx.GraphModule,
    ) -> Callable[..., Any]:
        """Given a sub-GraphModule for a region, return a callable that
        replaces it.  Typically wraps region_gm in an autograd.Function
        with custom save/backward behavior."""
        ...
