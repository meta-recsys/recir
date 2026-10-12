# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


#!/usr/bin/env python3


from copy import deepcopy
from typing import Callable, Dict, Tuple

import torch
from generative_recommenders.common import set_dev_mode
from recir.api import compile
from recir.distributed.parallel_mode import DataParallel
from recir.embedding.ec import TracableEmbeddingCollection
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torchrec.modules.embedding_configs import EmbeddingConfig

EMBEDDING_DIM = 16
NUM_EMBEDDINGS = 128
BATCH_SIZE = 8
MAX_UIH_LEN = 64
MAX_CANDIDATES = 20


def build_embedding_collection() -> TracableEmbeddingCollection:
    embedding_configs = [
        EmbeddingConfig(
            name="t_uih",
            embedding_dim=EMBEDDING_DIM,
            num_embeddings=NUM_EMBEDDINGS,
            feature_names=["f_uih_id"],
        ),
        EmbeddingConfig(
            name="t_candidate",
            embedding_dim=EMBEDDING_DIM,
            num_embeddings=NUM_EMBEDDINGS,
            feature_names=["f_candidate_id"],
        ),
    ]
    return TracableEmbeddingCollection(
        tables=embedding_configs,
        device=torch.device("cpu"),
    )


def build_inputs(
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    uih_values = torch.randint(
        0, NUM_EMBEDDINGS, (BATCH_SIZE, MAX_UIH_LEN), dtype=torch.int64, device=device
    )
    candidate_values = torch.randint(
        0,
        NUM_EMBEDDINGS,
        (BATCH_SIZE, MAX_CANDIDATES),
        dtype=torch.int64,
        device=device,
    )
    return uih_values, candidate_values


class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ec: TracableEmbeddingCollection = build_embedding_collection()

    def forward(
        self,
        uih_values: torch.Tensor,
        candidate_values: torch.Tensor,
    ) -> torch.Tensor:
        jt_dict: Dict[str, torch.Tensor] = {
            "f_uih_id": uih_values,
            "f_candidate_id": candidate_values,
        }
        seq_embeddings: Dict[str, torch.Tensor] = self.ec(jt_dict)
        return torch.sum(
            torch.stack([torch.sum(emb) for emb in seq_embeddings.values()])
        )


class EmbeddingCollectionTest(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    def _test_train_step(
        self,
        train_step: Callable[..., object],
        mod: torch.nn.Module,
        uih_values: torch.Tensor,
        candidate_values: torch.Tensor,
        data_parallel_mode: str = "replicate",
    ) -> None:
        ref_mod = (
            FSDP(deepcopy(mod), device_id=self.rank)
            if data_parallel_mode == "fully_shard"
            else DDP(deepcopy(mod), device_ids=[self.rank])
        )
        opt = torch.optim.SGD(mod.parameters(), lr=0.01, foreach=True)
        ref_opt = torch.optim.SGD(ref_mod.parameters(), lr=0.01, foreach=True)
        ref_uih_values = uih_values.clone()
        ref_candidate_values = candidate_values.clone()
        # warm up optimizers using the compiled train_step
        train_step(mod, opt, uih_values, candidate_values)
        opt.zero_grad()
        ref_mod(ref_uih_values, ref_candidate_values).sum().backward()
        ref_opt.step()
        ref_opt.zero_grad()
        # run one more compiled step and one reference step
        train_step(mod, opt, uih_values, candidate_values)
        ref_mod(ref_uih_values, ref_candidate_values).sum().backward()
        ref_opt.step()
        # test parameter parity
        if data_parallel_mode == "fully_shard":
            full_p1_list = []
            for p1 in mod.parameters():
                shards = [torch.empty_like(p1) for _ in range(self.world_size)]
                torch.distributed.all_gather(shards, p1)
                full_p1_list.append(torch.cat(shards, dim=0))

            with FSDP.summon_full_params(ref_mod):
                for full_p1, p2 in zip(full_p1_list, ref_mod.parameters()):
                    self.assertEqual(full_p1, p2)
        else:
            for p1, p2 in zip(mod.parameters(), ref_mod.parameters()):
                self.assertEqual(p1, p2)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_embedding_collection(self) -> None:
        torch.manual_seed(0)
        set_dev_mode(True)

        for data_parallel_mode in ["replicate", "fully_shard"]:

            @compile(parallel_mode=DataParallel(data_parallel_mode))
            def train_step(
                mod: torch.nn.Module,
                opt: torch.optim.Optimizer,
                uih_values: torch.Tensor,
                candidate_values: torch.Tensor,
            ) -> None:
                mod(uih_values, candidate_values).sum().backward()
                opt.step()

            mod = Model().cuda(self.rank)
            uih_values, candidate_values = build_inputs(
                torch.device(f"cuda:{self.rank}")
            )
            self._test_train_step(
                train_step,
                mod,
                uih_values,
                candidate_values,
                data_parallel_mode,
            )
