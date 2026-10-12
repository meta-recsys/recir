# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from recir.api import (  # noqa: F401
    add_state,
    extract_subgraph,
    find_markers,
    GraphPatch,
    replace_region,
)
from recir.backends import (  # noqa: F401
    Backend,
    InductorBackend,
    XLADynamoBridgeBackend,
    XLAGSPMDBackend,
    XLAShardMapBackend,
)
from recir.backends.types import BackendType  # noqa: F401
from recir.compiler.compilation import compile, compile_from_gm  # noqa: F401
from recir.compiler.runtime import (  # noqa: F401
    materialize_compiled_output,
    set_compiled_output_deferred,
)
from recir.compiler.tracing import trace  # noqa: F401
from recir.distributed.parallel_mode import (  # noqa: F401
    DataParallel,
    HybridParallel,
    ParallelMode,
)
from recir.distributed.sharding_plan import (  # noqa: F401
    EmbeddingShardingPlan,
    EmbeddingShardingType,
)
from recir.ops.marker import attach_markers, mark_boundary  # noqa: F401
from recir.pipeline import (  # noqa: F401
    mark_execution_stream,
    PipelineAction,
    PipelineConfig,
    PipelinePlan,
    PipelineSchedule,
    PipelineStageSpec,
    PipelineStreamSpec,
)
from recir.pipeline_strategies import (  # noqa: F401
    embedding_steady_state_pipeline_config,
    embedding_steady_state_pipeline_plan,
    PipelineRegion,
    sdd_pipeline_config,
    sdd_pipeline_plan,
    semi_sync_pipeline_config,
    semi_sync_pipeline_plan,
)
