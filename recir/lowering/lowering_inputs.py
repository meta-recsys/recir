# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Lowering every graph a backend's dispatch holds, one input at a time."""

from __future__ import annotations

from typing import Callable

from recir.composite import CompositeModule
from torch import fx, nn


def lower_inputs(
    gm: fx.GraphModule | CompositeModule, lower: Callable[[fx.GraphModule], nn.Module]
) -> nn.Module:
    """
    Lower each input of ``gm`` with ``lower`` and put the results back: each
    part of a ``CompositeModule``, or ``gm`` itself. Returns the lowered module.
    """
    if isinstance(gm, fx.GraphModule):
        return lower(gm)
    return gm.lower_parts(lower)
