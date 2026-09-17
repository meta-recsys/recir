# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from collections.abc import Callable
from copy import deepcopy
from functools import partial
from typing import overload

import torch
import torch.distributed as dist
from recir.api import compile, sync_device_state_to_torch
from recir.backends.inductor import InductorBackend
from recir.distributed.parallel_mode import HybridParallel
from recir.ops.fake.kjt import patch_ebc_for_tracing
from recir.region_transforms.base import RegionTransform
from recir.region_transforms.tbe_sparse_embedding import TBESparseEmbeddingTransform
from recir.sharding_plan import EmbeddingShardingPlan, EmbeddingShardingType
from torch import nn
from torch.distributed.distributed_c10d import _get_default_group, _world
from torch.distributed.tensor import DeviceMesh
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torchrec.modules.embedding_configs import EmbeddingBagConfig, PoolingType
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.optim.rowwise_adagrad import RowWiseAdagrad
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


_FEATURE_NAMES: tuple[str, str] = ("sum_feature", "mean_feature")
_EMBEDDING_DIM: int = 4
_NUM_EMBEDDINGS: int = 16
_UNEVEN_NUM_EMBEDDINGS: int = 17


class _TracingRowWiseAdagrad(torch.optim.Optimizer):
    def __init__(self, parameters: list[nn.Parameter]) -> None:
        super().__init__(
            parameters,
            {
                "lr": 0.1,
                "eps": 1.0e-8,
                "weight_decay": 0.0,
                "stochastic_rounding": False,
            },
        )
        for parameter in parameters:
            self.state[parameter]["sum"] = torch.zeros(
                (parameter.shape[0], 1),
                device=parameter.device,
            )

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        return closure() if closure is not None else None


class _Model(nn.Module):
    def __init__(
        self,
        device: torch.device,
        num_embeddings: int = _NUM_EMBEDDINGS,
    ) -> None:
        super().__init__()
        self.ebc = EmbeddingBagCollection(
            tables=[
                EmbeddingBagConfig(
                    name="sum_table",
                    embedding_dim=_EMBEDDING_DIM,
                    num_embeddings=num_embeddings,
                    feature_names=[_FEATURE_NAMES[0]],
                    pooling=PoolingType.SUM,
                ),
                EmbeddingBagConfig(
                    name="mean_table",
                    embedding_dim=_EMBEDDING_DIM,
                    num_embeddings=num_embeddings,
                    feature_names=[_FEATURE_NAMES[1]],
                    pooling=PoolingType.MEAN,
                ),
            ],
            device=device,
        )

    def forward(
        self,
        values: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        features = KeyedJaggedTensor(
            keys=list(_FEATURE_NAMES),
            values=values,
            lengths=lengths,
        )
        return self.ebc(features).values()


def _fake_mesh() -> DeviceMesh:
    process_group = _get_default_group()
    group_name = _world.pg_names[process_group]
    mesh = DeviceMesh(
        "cuda",
        torch.arange(1),
        _init_backend=False,
        _rank=0,
    )
    mesh._dim_group_names = [group_name]
    mesh._pg_registry[group_name] = process_group
    return mesh


def _sharding_plan(model: nn.Module) -> EmbeddingShardingPlan:
    return EmbeddingShardingPlan(
        sharding_map={
            name: EmbeddingShardingType.ROW_WISE
            for name, _parameter in model.named_parameters()
        }
    )


def _sparse_transform_factory(
    fqns: set[str], mesh: DeviceMesh, module: nn.Module
) -> TBESparseEmbeddingTransform:
    return TBESparseEmbeddingTransform(
        fqns,
        mesh,
        module,
        stochastic_rounding=False,
        variable_pooled_batch_per_rank=True,
    )


def _uniform_sparse_transform_factory(
    fqns: set[str], mesh: DeviceMesh, module: nn.Module
) -> TBESparseEmbeddingTransform:
    return TBESparseEmbeddingTransform(
        fqns,
        mesh,
        module,
        stochastic_rounding=False,
    )


def _rank_batch(
    rank: int,
    device: torch.device,
    iteration: int,
    *,
    uniform_batch_per_rank: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    variable_batches = (
        (
            ([1, 9, 2, 10], [2, 0, 1, 1]),
            ([3, 12, 4, 13, 5], [1, 1, 0, 0, 2, 1]),
        ),
        (
            ([6, 7, 8, 14], [0, 2, 1, 1, 0, 0]),
            ([9, 15], [1, 1]),
        ),
    )
    uniform_batches = (
        (
            variable_batches[0][0],
            ([3, 12, 4, 13], [1, 1, 0, 2]),
        ),
        (
            variable_batches[1][0],
            ([9, 15, 5, 13, 7], [1, 1, 0, 1, 1, 1]),
        ),
    )
    batches = uniform_batches if uniform_batch_per_rank else variable_batches
    values, lengths = batches[iteration][rank]
    batch_size = len(lengths) // len(_FEATURE_NAMES)
    coefficient = torch.arange(
        rank * batch_size * _EMBEDDING_DIM * len(_FEATURE_NAMES) + 1,
        (rank + 1) * batch_size * _EMBEDDING_DIM * len(_FEATURE_NAMES) + 1,
        dtype=torch.float32,
        device=device,
    ).reshape(batch_size, -1)
    return (
        torch.tensor(values, dtype=torch.int64, device=device),
        torch.tensor(lengths, dtype=torch.int64, device=device),
        coefficient,
    )


@unittest.skipUnless(torch.cuda.device_count() >= 1, "requires one GPU")
class EmbeddingBagCollectionSingleRankTest(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        if dist.is_initialized():
            dist.destroy_process_group()
        dist.init_process_group(
            backend="fake",
            world_size=1,
            rank=0,
        )

    def tearDown(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()
        super().tearDown()

    def test_compiled_sum_and_mean_training_matches_eager(self) -> None:
        torch.manual_seed(0)
        device = torch.device("cuda:0")
        reference = _Model(device)
        compiled = deepcopy(reference)
        self.assertEqual(patch_ebc_for_tracing(compiled), 1)

        compiled_optimizer = _TracingRowWiseAdagrad(list(compiled.parameters()))
        reference_optimizer = RowWiseAdagrad(
            reference.parameters(),
            lr=0.1,
            eps=1.0e-8,
        )
        parallel_mode = HybridParallel(
            dense_style="replicate",
            embedding_plan=_sharding_plan(compiled),
            sparse_transform_factory=_sparse_transform_factory,
        )
        parallel_mode._mesh = _fake_mesh()

        @compile(parallel_mode=parallel_mode)
        def train_step(
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            values: torch.Tensor,
            lengths: torch.Tensor,
            coefficient: torch.Tensor,
        ) -> torch.Tensor:
            output = model(values, lengths)
            (output * coefficient).sum().backward()
            optimizer.step()
            return output.detach()

        batches = (
            (
                torch.tensor([1, 3, 2, 4, 5, 7], device=device),
                torch.tensor([2, 0, 1, 1, 2, 0], device=device),
            ),
            (
                torch.tensor([8, 9, 10, 11, 2, 3, 4], device=device),
                torch.tensor([0, 1, 1, 2, 1, 0, 2, 0], device=device),
            ),
        )
        for values, lengths in batches:
            batch_size = lengths.numel() // len(_FEATURE_NAMES)
            coefficient = torch.arange(
                1,
                batch_size * _EMBEDDING_DIM * len(_FEATURE_NAMES) + 1,
                dtype=torch.float32,
                device=device,
            ).reshape(batch_size, -1)

            compiled_optimizer.zero_grad()
            actual = train_step(
                compiled,
                compiled_optimizer,
                values,
                lengths,
                coefficient,
            )
            sync_device_state_to_torch(train_step)

            reference_optimizer.zero_grad()
            expected = reference(values, lengths)
            (expected * coefficient).sum().backward()
            reference_optimizer.step()

            torch.testing.assert_close(actual, expected.detach())
            compiled_parameters = dict(compiled.named_parameters())
            reference_parameters = dict(reference.named_parameters())
            for name in reference_parameters:
                compiled_parameter = compiled_parameters[name]
                reference_parameter = reference_parameters[name]
                torch.testing.assert_close(
                    compiled_parameter,
                    reference_parameter,
                    atol=1.0e-5,
                    rtol=1.0e-5,
                    msg=f"parameter mismatch after EBC update: {name}",
                )


class EmbeddingBagCollectionDistributedTest(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_native_two_rank_training_handles_runtime_bag_count_changes(self) -> None:
        self._run_two_rank_training(
            sparse_transform_factory=None,
            num_embeddings=_UNEVEN_NUM_EMBEDDINGS,
        )

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_tbe_two_rank_training_handles_runtime_bag_count_changes(self) -> None:
        self._run_two_rank_training(
            sparse_transform_factory=partial(
                TBESparseEmbeddingTransform,
                stochastic_rounding=False,
                variable_pooled_batch_per_rank=True,
            )
        )

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_tbe_uniform_rank_batches_match_global_eager_training(self) -> None:
        self._run_two_rank_training(
            sparse_transform_factory=_uniform_sparse_transform_factory,
            uniform_batch_per_rank=True,
        )

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_tbe_inductor_two_rank_training_handles_runtime_bag_count_changes(
        self,
    ) -> None:
        self._run_two_rank_training(
            sparse_transform_factory=_sparse_transform_factory,
            use_inductor=True,
        )

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_tbe_inductor_uniform_rank_batches_match_global_eager_training(
        self,
    ) -> None:
        self._run_two_rank_training(
            sparse_transform_factory=_uniform_sparse_transform_factory,
            use_inductor=True,
            uniform_batch_per_rank=True,
        )

    def _run_two_rank_training(
        self,
        sparse_transform_factory: (
            Callable[[set[str], DeviceMesh, nn.Module], RegionTransform] | None
        ),
        *,
        num_embeddings: int = _NUM_EMBEDDINGS,
        use_inductor: bool = False,
        uniform_batch_per_rank: bool = False,
    ) -> None:
        torch.manual_seed(0)
        device = torch.device(f"cuda:{self.rank}")
        reference = _Model(device, num_embeddings)
        compiled = deepcopy(reference)
        self.assertEqual(patch_ebc_for_tracing(compiled), 1)

        compiled_optimizer: torch.optim.Optimizer
        reference_optimizer: torch.optim.Optimizer
        if sparse_transform_factory is None:
            compiled_optimizer = torch.optim.SGD(
                compiled.parameters(), lr=0.1, foreach=False
            )
            reference_optimizer = torch.optim.SGD(
                reference.parameters(), lr=0.1, foreach=False
            )
        else:
            compiled_optimizer = _TracingRowWiseAdagrad(list(compiled.parameters()))
            reference_optimizer = RowWiseAdagrad(
                reference.parameters(),
                lr=0.1,
                eps=1.0e-8,
            )
        parallel_mode = HybridParallel(
            dense_style="replicate",
            embedding_plan=_sharding_plan(compiled),
            sparse_transform_factory=sparse_transform_factory,
        )

        backend = InductorBackend() if use_inductor else None

        @compile(parallel_mode=parallel_mode, backend=backend)
        def train_step(
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            values: torch.Tensor,
            lengths: torch.Tensor,
            coefficient: torch.Tensor,
        ) -> torch.Tensor:
            output = model(values, lengths)
            (output * coefficient).sum().backward()
            optimizer.step()
            return output.detach()

        for iteration in range(2):
            values, lengths, coefficient = _rank_batch(
                self.rank,
                device,
                iteration,
                uniform_batch_per_rank=uniform_batch_per_rank,
            )
            expected_output = reference(values, lengths).detach()
            compiled_optimizer.zero_grad()
            actual_output = train_step(
                compiled,
                compiled_optimizer,
                values,
                lengths,
                coefficient,
            )
            sync_device_state_to_torch(train_step)

            reference_optimizer.zero_grad()
            global_losses = []
            for rank in range(self.world_size):
                rank_values, rank_lengths, rank_coefficient = _rank_batch(
                    rank,
                    device,
                    iteration,
                    uniform_batch_per_rank=uniform_batch_per_rank,
                )
                global_losses.append(
                    (reference(rank_values, rank_lengths) * rank_coefficient).sum()
                )
            torch.stack(global_losses).sum().backward()
            reference_optimizer.step()

            torch.testing.assert_close(actual_output, expected_output)
            compiled_parameters = dict(compiled.named_parameters())
            reference_parameters = dict(reference.named_parameters())
            for name, reference_parameter in reference_parameters.items():
                expected_local_parameter = torch.chunk(
                    reference_parameter,
                    self.world_size,
                    dim=0,
                )[self.rank]
                self.assertEqual(
                    compiled_parameters[name].shape,
                    expected_local_parameter.shape,
                )
                torch.testing.assert_close(
                    compiled_parameters[name],
                    expected_local_parameter,
                    atol=1.0e-5,
                    rtol=1.0e-5,
                    msg=f"rank-local EBC update mismatch: {name}",
                )
