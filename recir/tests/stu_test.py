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

# pyre-strict


from copy import deepcopy
from typing import Callable, List, Tuple

import torch
from generative_recommenders.common import HammerKernel, set_dev_mode
from generative_recommenders.modules.stu import STU, STULayer, STULayerConfig, STUStack
from recir.api import compile
from recir.distributed.parallel_mode import DataParallel
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


def build_stu_model() -> STUStack:
    stu_layers: List[STU] = [
        STULayer(
            config=STULayerConfig(
                embedding_dim=16,
                num_heads=2,
                hidden_dim=16,
                attention_dim=16,
                output_dropout_ratio=0.0,
                causal=True,
                target_aware=True,
                max_attn_len=None,
                attn_alpha=None,
                use_group_norm=False,
                recompute_normed_x=True,
                recompute_uvqk=True,
                recompute_y=True,
                sort_by_length=False,
                contextual_seq_len=0,
            ),
            is_inference=False,
        )
        for _ in range(1)
    ]
    stu = STUStack(
        stu_list=stu_layers,
        is_inference=False,
    )
    stu.recursive_setattr("_hammer_kernel", HammerKernel.TRITON)
    return stu


def build_inputs() -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor
]:
    max_uih_len = 64
    batch_size = 8
    contextual_seq_len = 0
    embedding_dim = 16
    x_lengths = torch.randint(max_uih_len + 1, (batch_size,))
    x_lengths = x_lengths + contextual_seq_len
    max_seq_len = max_uih_len + contextual_seq_len
    max_targets = 20
    num_targets = torch.randint(1, max_targets, size=(batch_size,))
    x_lengths = x_lengths + num_targets
    max_seq_len = max_seq_len + max_targets
    x_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(x_lengths)
    total_seq_len = int(x_offsets[-1].cpu().item())
    x = torch.randn(int(total_seq_len), embedding_dim, dtype=torch.bfloat16)
    return x, x_lengths, x_offsets, max_seq_len, num_targets


class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stu: STUStack = build_stu_model()

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        x_offsets: torch.Tensor,
        max_seq_len: int,
        num_targets: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sum(
            self.stu(
                x=x,
                x_lengths=x_lengths,
                x_offsets=x_offsets,
                max_seq_len=max_seq_len,
                num_targets=num_targets,
            )
        )


class STUTest(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    def _test_train_step(
        self,
        train_step: Callable[..., object],
        mod: torch.nn.Module,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        x_offsets: torch.Tensor,
        max_seq_len: int,
        num_targets: torch.Tensor,
        data_parallel_mode: str = "replicate",
    ) -> None:
        ref_mod = (
            FSDP(deepcopy(mod), device_id=self.rank)
            if data_parallel_mode == "fully_shard"
            else DDP(deepcopy(mod), device_ids=[self.rank])
        )
        opt = torch.optim.SGD(mod.parameters(), lr=0.01, foreach=True)
        ref_opt = torch.optim.SGD(ref_mod.parameters(), lr=0.01, foreach=True)
        ref_x = deepcopy(x)
        ref_x_lengths = deepcopy(x_lengths)
        ref_x_offsets = deepcopy(x_offsets)
        ref_num_targets = deepcopy(num_targets)
        # warm up optimizers using the compiled train_step
        train_step(mod, opt, x, x_lengths, x_offsets, max_seq_len, num_targets)
        opt.zero_grad()
        ref_mod(
            ref_x, ref_x_lengths, ref_x_offsets, max_seq_len, ref_num_targets
        ).sum().backward()
        ref_opt.step()
        ref_opt.zero_grad()
        # run one more compiled step and one reference step
        train_step(mod, opt, x, x_lengths, x_offsets, max_seq_len, num_targets)
        ref_mod(
            ref_x, ref_x_lengths, ref_x_offsets, max_seq_len, ref_num_targets
        ).sum().backward()
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
    def test_stu(self) -> None:
        seed = torch.random.initial_seed()
        torch.manual_seed(0)
        set_dev_mode(True)

        for data_parallel_mode in ["replicate", "fully_shard"]:

            @compile(parallel_mode=DataParallel(data_parallel_mode))
            def train_step(
                mod: torch.nn.Module,
                opt: torch.optim.Optimizer,
                x: torch.Tensor,
                x_lengths: torch.Tensor,
                x_offsets: torch.Tensor,
                max_seq_len: int,
                num_targets: torch.Tensor,
            ) -> None:
                mod(x, x_lengths, x_offsets, max_seq_len, num_targets).sum().backward()
                opt.step()

            mod = Model().cuda(self.rank)
            torch.manual_seed(seed)
            x, x_lengths, x_offsets, max_seq_len, num_targets = build_inputs()
            x = x.cuda(self.rank)
            x_lengths = x_lengths.cuda(self.rank)
            x_offsets = x_offsets.cuda(self.rank)
            num_targets = num_targets.cuda(self.rank)
            self._test_train_step(
                train_step,
                mod,
                x,
                x_lengths,
                x_offsets,
                max_seq_len,
                num_targets,
                data_parallel_mode,
            )
