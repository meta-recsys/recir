# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""The protocol RecIR's composite graph modules share, which lowering walks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Protocol

from torch import fx, nn


@dataclass(frozen=True)
class Part:
    """One graph of a composite module: its ``name``, and its ``index`` when its kind numbers them."""

    name: str
    index: int | None = None

    @property
    def label(self) -> str:
        """An identifier for the part, unique within its module: its name, or ``stage{index}_{name}`` for a numbered part."""
        if self.index is None:
            return self.name
        return f"stage{self.index}_" + re.sub(r"[^0-9A-Za-z_]", "_", self.name)


class CompositeModule(Protocol):
    """
    A module that holds its graphs inside it: an ``IterGraphModule``'s phases,
    a pipeline's stages, or a steady state's flattened graph.

    This protocol is how a backend lowers every graph of such a module
    without checking its type. A module kind joins by implementing
    ``lower_parts``.
    """

    def lower_parts(
        self, lower: Callable[[Part, fx.GraphModule], nn.Module]
    ) -> nn.Module:
        """Replace each of the module's graphs with ``lower(part, graph)``, and return the module."""
        ...
