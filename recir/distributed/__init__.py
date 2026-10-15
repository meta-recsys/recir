# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

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
