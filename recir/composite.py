# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""The protocol RecIR's composite graph modules share, which lowering walks."""

from __future__ import annotations

from typing import Callable, Protocol

from torch import fx, nn


class CompositeModule(Protocol):
    """
    A module that holds its graphs inside it: an ``IterGraphModule``'s phases,
    a pipeline's stages, or a steady state's flattened graph.

    This protocol is how a backend lowers every graph of such a module
    without checking its type. A module kind joins by implementing
    ``lower_parts``.
    """

    def lower_parts(self, lower: Callable[[fx.GraphModule], nn.Module]) -> nn.Module:
        """Replace each of the module's graphs with ``lower(graph)``, and return the module."""
        ...
