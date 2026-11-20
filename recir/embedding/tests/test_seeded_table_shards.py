# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed as dist
from recir.embedding.seeded_init import init_shards_from_seeded_tables
from torch.distributed._shard.metadata import ShardMetadata
from torch.distributed._shard.sharded_tensor import (
    init_from_local_shards,
    Shard,
    ShardedTensor,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torchrec.modules.embedding_configs import EmbeddingConfig

_ROWS, _DIM = 6, 4


def _sharded(rows: range, rank: int, group: dist.ProcessGroup | None) -> ShardedTensor:
    metadata = ShardMetadata(
        shard_offsets=[rows.start, 0],
        shard_sizes=[len(rows), _DIM],
        placement=f"rank:{rank}/cpu",
    )
    shard = Shard(torch.zeros(len(rows), _DIM), metadata)
    return init_from_local_shards([shard], _ROWS, _DIM, process_group=group)


class SeededTableShardsTest(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @with_comms
    def test_row_shards_match_the_single_rank_table(self) -> None:
        config = EmbeddingConfig(
            name="t", embedding_dim=_DIM, num_embeddings=_ROWS, feature_names=["f"]
        )
        half = _ROWS // 2
        mine = range(self.rank * half, (self.rank + 1) * half)
        row_wise = _sharded(mine, self.rank, dist.group.WORLD)
        alone = [dist.new_group([rank]) for rank in range(self.world_size)]
        mine_alone = alone[self.rank]
        assert isinstance(mine_alone, dist.ProcessGroup)
        whole = _sharded(range(_ROWS), self.rank, mine_alone)

        key = "ec.embeddings.t.weight"
        self.assertEqual(
            init_shards_from_seeded_tables({key: row_wise}, [config]), ["t"]
        )
        init_shards_from_seeded_tables({key: whole}, [config])

        local = row_wise.local_shards()[0].tensor
        gathered = [torch.empty_like(local) for _ in range(self.world_size)]
        dist.all_gather(gathered, local)
        self.assertTrue(
            torch.equal(torch.cat(gathered), whole.local_shards()[0].tensor)
        )
        self.assertFalse(torch.equal(gathered[0], gathered[1]))
        bound = config.get_weight_init_max()
        self.assertLessEqual(float(local.abs().max()), bound)
