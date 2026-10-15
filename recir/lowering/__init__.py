# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from recir.lowering.lowering_utils import (  # noqa: F401
    create_subgraph_module,
    lower_subgraph_nodes,
    segment_and_lower,
)
from recir.lowering.op_swap import OpSwapRegistry, swap_ops  # noqa: F401
from recir.lowering.partial_lower import _InductorModule, partial_lower  # noqa: F401
from recir.lowering.xla_lower import (  # noqa: F401
    _is_xla_compatible,
    _XLAModule,
    partial_lower_xla,
)
