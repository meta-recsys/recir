# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Backend package: ``Backend`` ABC + concrete subclasses.

A ``Backend`` instance owns stages 6-10 of the recir compile pipeline (graph
optimization, IterGraph wrapping, dispatch, build, serialization).  Pass an
instance to :func:`recir.compile` via the ``backend=`` keyword argument once
the v7.1 surface lands.
"""

from recir.backends.base import Backend, GraphPatch
from recir.backends.inductor import InductorBackend
from recir.backends.torchtpu import TorchTPUBackend
from recir.backends.types import BackendType
from recir.backends.xla import (
    XLADynamoBridgeBackend,
    XLAGSPMDBackend,
    XLAShardMapBackend,
)

__all__ = [
    "Backend",
    "BackendType",
    "GraphPatch",
    "InductorBackend",
    "TorchTPUBackend",
    "XLADynamoBridgeBackend",
    "XLAGSPMDBackend",
    "XLAShardMapBackend",
]
