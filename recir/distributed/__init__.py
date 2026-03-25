# pyre-strict
from recir.distributed.data_parallel import (  # noqa: F401
    DataParallelStrategy,
    DataParallelStyle,
    gradients_tagging,
    NodeType,
    partition_data_parallel,
    shard_params_and_states,
)
from recir.distributed.parallel_mode import (  # noqa: F401
    DataParallel,
    GSPMDParallel,
    ParallelMode,
)
