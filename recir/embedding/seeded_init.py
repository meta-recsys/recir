# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Initialize sharded embedding tables independently of how they are sharded."""

from typing import Any, List, Mapping, Sequence

from recir.utils.meta_tensor_init import fill_table_shard
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torchrec.modules.embedding_configs import BaseEmbeddingConfig


def init_shards_from_seeded_tables(
    state: Mapping[str, Any], table_configs: Sequence[BaseEmbeddingConfig]
) -> List[str]:
    """Write each local shard of a sharded table from one seeded table.

    torchrec initializes every rank's shard independently from the global RNG,
    so ranks with the same RNG state hold identical rows and the table depends
    on the world size. Here every shard takes its rows of the same seeded table
    (``fill_table_shard``), so the embeddings match across ranks and world
    sizes. Returns the tables written.
    """
    configs = {config.name: config for config in table_configs}
    written = []
    for key, value in state.items():
        if not isinstance(value, ShardedTensor):
            continue
        config = configs.get(key.rsplit(".", 2)[-2])
        if config is None:
            continue
        for shard in value.local_shards():
            fill_table_shard(
                shard.tensor,
                config.name,
                tuple(value.size()),
                shard.metadata.shard_offsets,
                config.get_weight_init_min(),
                config.get_weight_init_max(),
            )
        written.append(config.name)
    return written
