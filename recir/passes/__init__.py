# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from recir.passes.comm_fusion import (  # noqa: F401
    comm_fusion_with_concat,
    CommBlock,
    get_all_comm_blocks,
    get_comm_block,
    schedule_comm_wait,
)
from recir.passes.optimizer_transforms import (  # noqa: F401
    AdamArgs,
    find_all_descendants,
    ForeachAddBlock,
    FusedAdamBlock,
    FusedOptimizerBlock,
    get_all_fused_optimizer_blocks,
    get_fused_optimizer_block,
    iter_move_grads_and_optimizers,
    remove_copy_from_optimizer,
    split_fused_optimizer,
)
from recir.passes.pass_registry import (  # noqa: F401
    default_registry,
    enable_graph_optimization_dump,
    graph_optimization_pass,
)
from recir.passes.weight_free import insert_weight_free_nodes  # noqa: F401
