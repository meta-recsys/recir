#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import annotations

"""
Verify EmbeddingCollection row-wise sharding works on CPU with gloo backend.

Spawns 2 workers, each running forward + backward + optimizer step on a
minimal EC model, then compares sharded predictions to an unsharded baseline.
"""

import functools
import multiprocessing
import os
import traceback
import unittest
from typing import cast, List

import torch
import torch.distributed as dist
import torch.nn as nn
from torchrec.distributed.embedding import EmbeddingCollectionSharder
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.test_utils.multi_process import MultiProcessContext
from torchrec.distributed.test_utils.test_model import ModelInput
from torchrec.distributed.test_utils.test_sharding import copy_state_dict
from torchrec.distributed.types import (
    ModuleSharder,
    ShardingEnv,
    ShardingPlan,
    ShardingType,
)
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.test_utils import get_free_port

try:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")
except OSError:
    pass

_MAX_SEQ_LEN: int = 20
_BATCH_SIZE: int = 4


class SimpleECModel(nn.Module):
    def __init__(
        self,
        tables: List[EmbeddingConfig],
        embedding_names: List[str],
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        if device is None:
            device = torch.device("cpu")
        self.ec = EmbeddingCollection(tables=tables, device=device, need_indices=True)
        self.embedding_names = embedding_names
        self.embedding_dim: int = self.ec.embedding_dim()
        in_features = len(embedding_names) * _MAX_SEQ_LEN * self.embedding_dim
        self.linear = nn.Linear(in_features, 16, device=device)

    def forward(
        self,
        kjt: KeyedJaggedTensor,
        label: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        jt_dict = self.ec(kjt)
        padded = [
            torch.ops.fbgemm.jagged_2d_to_dense(
                values=jt_dict[e].values(),
                offsets=jt_dict[e].offsets(),
                max_sequence_length=_MAX_SEQ_LEN,
            ).view(-1, _MAX_SEQ_LEN * self.embedding_dim)
            for e in self.embedding_names
        ]
        sparse = torch.cat(padded, dim=1)
        pred = torch.mean(self.linear(sparse), dim=1)
        if self.training:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, label)
            return (loss, pred)
        return pred


class _TestECSharder(EmbeddingCollectionSharder):
    """ROW_WISE + FUSED kernel only."""

    def __init__(self) -> None:
        super().__init__(fused_params={"learning_rate": 0.1})

    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [ShardingType.ROW_WISE.value]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [EmbeddingComputeKernel.FUSED.value]


def _train_and_infer(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    kjt: KeyedJaggedTensor,
    label: torch.Tensor,
) -> torch.Tensor:
    """One train step (fwd → bwd → step) then return inference predictions."""
    opt.zero_grad()
    model.train(True)
    loss, _ = model(kjt, label)
    loss.backward()
    opt.step()
    with torch.no_grad():
        model.train(False)
        return model(kjt, label)


def _ec_rw_worker(
    rank: int,
    world_size: int,
    tables: List[EmbeddingConfig],
    embedding_names: List[str],
    backend: str,
) -> None:
    with MultiProcessContext(rank, world_size, backend) as ctx:
        assert ctx.pg is not None
        pg, device = ctx.pg, ctx.device
        torch.manual_seed(0)
        # Unsharded baseline
        global_model = SimpleECModel(tables, embedding_names, device=device)
        (global_input, local_inputs) = ModelInput.generate(
            world_size=world_size,
            tables=tables,
            weighted_tables=[],
            num_float_features=1,
            batch_size=_BATCH_SIZE,
        )
        global_input = global_input.to(device)
        local_input = local_inputs[rank].to(device)
        # Sharded model (meta → DMP materializes shards)
        sharders = [cast(ModuleSharder[nn.Module], _TestECSharder())]
        local_model = SimpleECModel(
            tables, embedding_names, device=torch.device("meta")
        )
        plan: ShardingPlan = EmbeddingShardingPlanner(
            topology=Topology(world_size=world_size, compute_device=device.type),
        ).collective_plan(local_model, sharders, pg)
        local_model = DistributedModelParallel(
            local_model,
            env=ShardingEnv.from_process_group(pg),
            plan=plan,
            sharders=sharders,
            device=device,
        )
        dense_optim = KeyedOptimizerWrapper(
            dict(in_backward_optimizer_filter(local_model.named_parameters())),
            lambda params: torch.optim.SGD(params, lr=0.1, foreach=False),
        )
        local_opt = CombinedOptimizer([local_model.fused_optimizer, dense_optim])
        copy_state_dict(local_model.state_dict(), global_model.state_dict())
        # Train + infer
        local_pred = _train_and_infer(
            local_model,
            local_opt,
            cast(KeyedJaggedTensor, local_input.idlist_features),
            local_input.label,
        )
        global_pred = _train_and_infer(
            global_model,
            torch.optim.SGD(global_model.parameters(), lr=0.1, foreach=False),
            cast(KeyedJaggedTensor, global_input.idlist_features),
            global_input.label,
        )
        # Gathered sharded predictions must match unsharded baseline
        all_local = [torch.empty_like(local_pred) for _ in range(world_size)]
        dist.all_gather(all_local, local_pred, group=pg)
        torch.testing.assert_close(global_pred, torch.cat(all_local))


# ── Test ─────────────────────────────────────────────────────────────────────


class ECRowWiseCPUTest(unittest.TestCase):
    """EmbeddingCollection ROW_WISE sharding on CPU with gloo."""

    def setUp(self) -> None:
        super().setUp()
        self._orig_cuda: str | None = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(get_free_port())
        os.environ["GLOO_DEVICE_TRANSPORT"] = "TCP"

        num_features = 4
        self.tables: List[EmbeddingConfig] = [
            EmbeddingConfig(
                num_embeddings=(i + 1) * 11,
                embedding_dim=16,
                name=f"table_{i}",
                feature_names=[f"feature_{i}"],
            )
            for i in range(num_features)
        ]
        self.embedding_names: List[str] = [
            f for t in self.tables for f in t.feature_names
        ]

    def tearDown(self) -> None:
        if self._orig_cuda is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = self._orig_cuda
        os.environ.pop("GLOO_DEVICE_TRANSPORT", None)
        super().tearDown()

    def test_ec_rw_fused_cpu(self) -> None:
        """Sharded EC predictions match unsharded baseline on CPU/gloo."""
        world_size = 2
        ctx = multiprocessing.get_context("forkserver")
        with ctx.Pool(processes=world_size) as pool:
            results = []
            errors: List[BaseException] = []
            for rank in range(world_size):
                r = pool.apply_async(
                    functools.partial(
                        _ec_rw_worker,
                        rank=rank,
                        world_size=world_size,
                        tables=self.tables,
                        embedding_names=self.embedding_names,
                        backend="gloo",
                    ),
                    error_callback=lambda e: errors.append(e),
                )
                results.append(r)

            fails = 0
            for r in results:
                r.wait()
                if not r.successful():
                    fails += 1

            msgs = [
                "\n".join(traceback.format_exception(type(e), e, e.__traceback__))
                for e in errors
            ]
            self.assertEqual(fails, 0, f"{fails} worker(s) failed:\n" + "\n".join(msgs))
