# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from recir.api import (  # noqa: F401
    add_state,
    compile,
    compile_from_gm,
    extract_subgraph,
    find_markers,
    GraphPatch,
    replace_region,
    trace,
)
from recir.backend import BackendType  # noqa: F401
from recir.backends import (  # noqa: F401
    Backend,
    InductorBackend,
    XLADynamoBridgeBackend,
    XLAGSPMDBackend,
    XLAShardMapBackend,
)
from recir.distributed.parallel_mode import (  # noqa: F401
    DataParallel,
    HybridParallel,
    ParallelMode,
)
from recir.ops.marker import attach_markers, mark_boundary  # noqa: F401
from recir.sharding_plan import (  # noqa: F401
    EmbeddingShardingPlan,
    EmbeddingShardingType,
)
