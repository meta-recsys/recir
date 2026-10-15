# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


from copy import deepcopy
from functools import wraps
from typing import Any, Callable, Type

import torch
import torch.nn as nn
from recir.api import compile
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests  # noqa: TCH001
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms as base_with_comms,
)


def with_comms(func: Callable[..., Any]) -> Callable[..., Any]:
    @base_with_comms
    @wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        # make sure we set different random seeds for each rank
        # otherwise we dont need DDP / SPMD
        # (we would have the same parameters and inputs everywhere)
        torch.manual_seed(self.rank)
        return func(self, *args, **kwargs)

    return wrapper


class CoverageTest(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    def _test_train_step(
        self, train_step: Callable[..., Any], mod: nn.Module, *args: torch.Tensor
    ) -> None:
        ddp_mod = DDP(deepcopy(mod), device_ids=[self.rank])

        opt = torch.optim.SGD(mod.parameters(), lr=0.01, foreach=True)
        ddp_opt = torch.optim.SGD(ddp_mod.parameters(), lr=0.01, foreach=True)

        ddp_args = deepcopy(args)

        # materialize optimizer states
        mod(*args).sum().backward()
        opt.step()
        opt.zero_grad()

        ddp_mod(*ddp_args).sum().backward()
        ddp_opt.step()
        ddp_opt.zero_grad()

        # test parameter parity
        train_step(mod, opt, *args)

        ddp_mod(*ddp_args).sum().backward()
        ddp_opt.step()

        for p1, p2 in zip(mod.parameters(), ddp_mod.parameters()):
            self.assertEqual(p1, p2)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_log_softmax(self) -> None:
        torch.manual_seed(0)

        @compile()
        def train_step(
            mod: nn.Module, opt: torch.optim.Optimizer, inp: torch.Tensor
        ) -> None:
            mod(inp).sum().backward()
            opt.step()

        mod = nn.Sequential(
            nn.Linear(10, 10),
            nn.LogSoftmax(dim=1),
        ).cuda(self.rank)
        inp = torch.randn(2, 10).cuda(self.rank)
        self._test_train_step(train_step, mod, inp)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_nll_loss(self) -> None:
        class ModuleWithLoss(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mod = nn.Sequential(
                    nn.Linear(10, 10),
                    nn.LogSoftmax(dim=1),
                )
                self.lss = nn.NLLLoss()

            def forward(self, x: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
                return self.lss(self.mod(x), tgt)

        torch.manual_seed(0)
        mod = ModuleWithLoss().cuda(self.rank)

        @compile()
        def train_step(
            mod: nn.Module,
            opt: torch.optim.Optimizer,
            inp: torch.Tensor,
            tgt: torch.Tensor,
        ) -> None:
            mod(inp, tgt).backward()
            opt.step()

        inp = torch.randn(2, 10).to(self.rank)
        tgt = torch.empty(2, dtype=torch.long).random_(0, 10).to(self.rank)

        self._test_train_step(train_step, mod, inp, tgt)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_replicated_embedding(self) -> None:
        N, D, B = 10, 8, 2

        class EmbeddingModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.emb = nn.Embedding(N, D)
                self.norm = nn.LayerNorm(D, elementwise_affine=False)
                self.fc = nn.Linear(D, D)
                self.softmax = nn.Softmax(dim=1)
                self.lss = nn.NLLLoss()

            def forward(self, ids: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
                return self.lss(self.softmax(self.fc(self.norm(self.emb(ids)))), tgt)

        torch.manual_seed(0)
        mod = EmbeddingModule().cuda(self.rank)

        @compile()
        def train_step(
            mod: nn.Module,
            opt: torch.optim.Optimizer,
            ids: torch.Tensor,
            tgt: torch.Tensor,
        ) -> None:
            mod(ids, tgt).sum().backward()
            opt.step()

        ids = torch.randint(0, N, (B,)).cuda(self.rank)
        tgt = torch.empty(B, dtype=torch.long).random_(0, D).to(self.rank)

        self._test_train_step(train_step, mod, ids, tgt)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_pos_embedding(self) -> None:
        N, D, B, Block = 10, 8, 2, 20

        class EmbeddingModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.wte = nn.Embedding(N, D)
                self.wpe = nn.Embedding(Block, D)
                self.norm = nn.LayerNorm(D, elementwise_affine=False)
                self.fc = nn.Linear(D, D)

            def forward(self, ids: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
                _, t = ids.size()
                wte = self.wte(ids)
                wpe = self.wpe(
                    torch.arange(0, t, dtype=torch.long, device=ids.device).unsqueeze(0)
                )
                emb = wpe + wte
                norm = self.norm(emb)
                fc = self.fc(norm)
                log = F.softmax(fc, dim=-1)
                return F.cross_entropy(log.view(-1, log.size(-1)), tgt.view(-1))

        torch.manual_seed(0)
        mod = EmbeddingModule().cuda(self.rank)

        @compile()
        def train_step(
            mod: nn.Module,
            opt: torch.optim.Optimizer,
            ids: torch.Tensor,
            tgt: torch.Tensor,
        ) -> None:
            mod(ids, tgt).sum().backward()
            opt.step()

        ids = torch.randint(0, N, (B, Block)).cuda(self.rank)
        tgt = torch.empty((B, Block), dtype=torch.long).random_(0, D).to(self.rank)

        self._test_train_step(train_step, mod, ids, tgt)

    def _test_op_with_train_step(self, Model: Type[nn.Module]) -> None:
        torch.manual_seed(0)
        mod = Model().cuda(self.rank)

        @compile()
        def train_step(
            mod: nn.Module, opt: torch.optim.Optimizer, inp: torch.Tensor
        ) -> None:
            mod(inp).sum().backward()
            opt.step()

        inp = torch.randn(2, 10).cuda(self.rank)
        self._test_train_step(train_step, mod, inp)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_factory_full(self) -> None:
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(10, 10)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                y = torch.full(x.shape, 7, device=x.device)
                return y + self.fc(x)

        self._test_op_with_train_step(Model)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_factory_arange(self) -> None:
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(10, 10)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                y = torch.arange(x.numel(), device=x.device).view(x.shape)
                z = torch.arange(0, x.numel(), device=x.device).view(x.shape)
                return self.fc(x) + y + z

        self._test_op_with_train_step(Model)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_sym_numel(self) -> None:
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(10, 10)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                y = self.fc.weight.numel()
                return self.fc(x) + y

        self._test_op_with_train_step(Model)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_sym_stride(self) -> None:
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(10, 10)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                y = self.fc.weight.stride(0)
                return self.fc(x) + y

        self._test_op_with_train_step(Model)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_scalar(self) -> None:
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(10, 10)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # FIXME: torch.tensor(x.numel()) is captured as a tensor constant
                y = torch.ops.aten.scalar_tensor.default(
                    7, dtype=x.dtype, device=x.device
                )
                return self.fc(x) + y

        self._test_op_with_train_step(Model)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_stack(self) -> None:
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(10, 10)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.stack([x, self.fc(x)], dim=1)

        self._test_op_with_train_step(Model)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_arithmetic_ops_on_symint(self) -> None:
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(10, 10)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fc(x) + x.shape[0] * x.numel() - x.shape[0] // 2

        self._test_op_with_train_step(Model)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_slice(self) -> None:
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(10, 10)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fc(x)[:, :1]

        self._test_op_with_train_step(Model)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_bulk_cat(self) -> None:
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(10, 10)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.cat([self.fc(x) for _ in range(100)], dim=1)

        self._test_op_with_train_step(Model)


if __name__ == "__main__":
    if False:
        run_tests()
