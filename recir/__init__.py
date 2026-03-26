# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from recir.api import compile, Override  # noqa: F401
from recir.backend import BackendType  # noqa: F401
from recir.distributed.parallel_mode import DataParallel, ParallelMode  # noqa: F401
from recir.sharding_plan import (  # noqa: F401
    EmbeddingShardingPlan,
    EmbeddingShardingType,
)
