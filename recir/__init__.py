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
    materialize_compiled_output,
    replace_region,
    set_compiled_output_deferred,
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
from recir.pipeline import (  # noqa: F401
    PipelineAction,
    PipelineConfig,
    PipelinePlan,
    PipelineSchedule,
    PipelineStageSpec,
)
from recir.pipeline_strategies import (  # noqa: F401
    PipelineRegion,
    sdd_pipeline_config,
    sdd_pipeline_plan,
    semi_sync_pipeline_config,
    semi_sync_pipeline_plan,
)
from recir.sharding_plan import (  # noqa: F401
    EmbeddingShardingPlan,
    EmbeddingShardingType,
)
