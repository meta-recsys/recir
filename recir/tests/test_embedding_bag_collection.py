# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from collections.abc import Callable
from copy import deepcopy
from typing import Literal, overload

import torch
import torch.distributed as dist
import torch.fx as fx
from recir.api import compile, sync_device_state_to_torch
from recir.backends.base import Backend
from recir.backends.inductor import InductorBackend
from recir.distributed.parallel_mode import HybridParallel
from recir.distributed.sharding_plan import EmbeddingShardingPlan, EmbeddingShardingType
from recir.ops.fake.kjt import patch_ebc_for_tracing
from recir.region_transforms.base import RegionTransform
from recir.region_transforms.tbe_sparse_embedding import TBESparseEmbeddingTransform
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
_ShardingCase = Literal[
    "row_wise",
    "mixed",
    "table_wise",
    "table_wise_same_owner",
]


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


class _EagerBackend(Backend):
    """Runs the compiled FX graph as is. The TBE graph does not lower through
    Inductor yet (see the tbe_inductor tests)."""

    def dispatch(self, gm: fx.GraphModule) -> nn.Module:
        return gm


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


def _sharding_plan(
    model: nn.Module,
    sharding_case: _ShardingCase = "row_wise",
    *,
    table_wise_owner_rank: int = 0,
    world_size: int = 2,
) -> EmbeddingShardingPlan:
    sharding_map = {}
    table_wise_rank_map = {}
    for table_index, (name, _parameter) in enumerate(model.named_parameters()):
        is_table_wise = sharding_case in ("table_wise", "table_wise_same_owner") or (
            sharding_case == "mixed" and ".sum_table.weight" in name
        )
        if is_table_wise:
            sharding_map[name] = EmbeddingShardingType.TABLE_WISE
            table_wise_rank_map[name] = (
                table_index % world_size
                if sharding_case == "table_wise"
                else table_wise_owner_rank
            )
        else:
            sharding_map[name] = EmbeddingShardingType.ROW_WISE
    return EmbeddingShardingPlan(
        sharding_map=sharding_map,
        table_wise_rank_map=table_wise_rank_map,
    )


def _tbe_sparse_transform_factory(
    embedding_plan: EmbeddingShardingPlan,
) -> Callable[[set[str], DeviceMesh, nn.Module], RegionTransform]:
    def create(
        fqns: set[str], mesh: DeviceMesh, module: nn.Module
    ) -> TBESparseEmbeddingTransform:
        return TBESparseEmbeddingTransform(
            fqns,
            mesh,
            module,
            embedding_plan=embedding_plan,
            stochastic_rounding=False,
        )

    return create


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
        embedding_plan = _sharding_plan(compiled)
        parallel_mode = HybridParallel(
            dense_style="replicate",
            embedding_plan=embedding_plan,
            sparse_transform_factory=_tbe_sparse_transform_factory(embedding_plan),
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
            use_tbe=False,
            num_embeddings=_UNEVEN_NUM_EMBEDDINGS,
        )

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_native_mixed_sharding_matches_global_eager_training(self) -> None:
        self._run_two_rank_training(
            use_tbe=False,
            sharding_case="mixed",
        )

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_native_inductor_mixed_sharding_matches_global_eager_training(
        self,
    ) -> None:
        self._run_two_rank_training(
            use_tbe=False,
            use_inductor=True,
            sharding_case="mixed",
        )

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_native_table_wise_tables_on_distinct_owners_match_eager(self) -> None:
        self._run_two_rank_training(
            use_tbe=False,
            sharding_case="table_wise",
        )

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_tbe_two_rank_training_handles_runtime_bag_count_changes(self) -> None:
        self._run_two_rank_training(
            use_tbe=True,
            num_embeddings=_UNEVEN_NUM_EMBEDDINGS,
        )

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_tbe_mixed_sharding_matches_global_eager_training(self) -> None:
        self._run_two_rank_training(
            use_tbe=True,
            sharding_case="mixed",
        )

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_tbe_mixed_sharding_supports_nonzero_table_owner(self) -> None:
        self._run_two_rank_training(
            use_tbe=True,
            sharding_case="mixed",
            table_wise_owner_rank=1,
        )

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_tbe_table_wise_tables_on_same_owner_match_eager(self) -> None:
        self._run_two_rank_training(
            use_tbe=True,
            sharding_case="table_wise_same_owner",
            table_wise_owner_rank=1,
        )

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_tbe_uniform_rank_batches_match_global_eager_training(self) -> None:
        self._run_two_rank_training(
            use_tbe=True,
            uniform_batch_per_rank=True,
        )

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_tbe_inductor_two_rank_training_handles_runtime_bag_count_changes(
        self,
    ) -> None:
        self._run_two_rank_training(
            use_tbe=True,
            use_inductor=True,
        )

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_tbe_inductor_uniform_rank_batches_match_global_eager_training(
        self,
    ) -> None:
        self._run_two_rank_training(
            use_tbe=True,
            use_inductor=True,
            uniform_batch_per_rank=True,
        )

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_tbe_inductor_mixed_sharding_matches_global_eager_training(
        self,
    ) -> None:
        self._run_two_rank_training(
            use_tbe=True,
            use_inductor=True,
            sharding_case="mixed",
        )

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_tbe_inductor_table_wise_tables_on_distinct_owners_match_eager(
        self,
    ) -> None:
        self._run_two_rank_training(
            use_tbe=True,
            use_inductor=True,
            sharding_case="table_wise",
        )

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_tbe_eval_graph_shares_train_parameters(self) -> None:
        self._run_two_rank_train_and_eval(eval_first=False)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_tbe_eval_graph_listed_first_shares_train_parameters(self) -> None:
        # Eval is traced before the train trace swaps in the TBE tables, so it
        # must be rebound to them at compile time.
        self._run_two_rank_train_and_eval(eval_first=True)

    def _run_two_rank_train_and_eval(self, eval_first: bool) -> None:
        """Train and forward-only eval graphs compiled together read the same
        rank-local (TBE-lowered) tables, and training continues after eval."""
        torch.manual_seed(0)
        device = torch.device(f"cuda:{self.rank}")
        reference = _Model(device, _UNEVEN_NUM_EMBEDDINGS)
        compiled = deepcopy(reference)
        self.assertEqual(patch_ebc_for_tracing(compiled), 1)
        compiled_optimizer = _TracingRowWiseAdagrad(list(compiled.parameters()))
        reference_optimizer = RowWiseAdagrad(reference.parameters(), lr=0.1, eps=1.0e-8)
        embedding_plan = _sharding_plan(
            compiled, "mixed", table_wise_owner_rank=1, world_size=self.world_size
        )

        def train_fn(
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

        def eval_fn(
            model: nn.Module, values: torch.Tensor, lengths: torch.Tensor
        ) -> torch.Tensor:
            with torch.no_grad():
                return model(values, lengths)

        values, lengths, coefficient = _rank_batch(self.rank, device, 0)
        functions = [train_fn, eval_fn]
        example_args = [
            (compiled, compiled_optimizer, values, lengths, coefficient),
            (compiled, values, lengths),
        ]
        order = [1, 0] if eval_first else [0, 1]
        runners = compile(
            parallel_mode=HybridParallel(
                dense_style="replicate",
                embedding_plan=embedding_plan,
                sparse_transform_factory=_tbe_sparse_transform_factory(embedding_plan),
            ),
            backend=_EagerBackend(),
        )(*(functions[i] for i in order), example_args=[example_args[i] for i in order])
        train_step, eval_step = (runners[order.index(i)] for i in (0, 1))

        def train(iteration: int) -> None:
            values, lengths, coefficient = _rank_batch(self.rank, device, iteration)
            compiled_optimizer.zero_grad()
            actual = train_step(
                compiled, compiled_optimizer, values, lengths, coefficient
            )
            reference_optimizer.zero_grad()
            torch.testing.assert_close(actual, reference(values, lengths).detach())
            losses = []
            for rank in range(self.world_size):
                rank_values, rank_lengths, rank_coefficient = _rank_batch(
                    rank, device, iteration
                )
                losses.append(
                    (reference(rank_values, rank_lengths) * rank_coefficient).sum()
                )
            torch.stack(losses).sum().backward()
            reference_optimizer.step()

        def evaluate(iteration: int) -> None:
            values, lengths, _ = _rank_batch(self.rank, device, iteration)
            torch.testing.assert_close(
                eval_step(compiled, values, lengths),
                reference(values, lengths).detach(),
                atol=1.0e-5,
                rtol=1.0e-5,
            )

        train(0)
        evaluate(1)
        train(1)
        evaluate(0)
        train(0)
        evaluate(1)

    def _run_two_rank_training(
        self,
        use_tbe: bool,
        *,
        num_embeddings: int = _NUM_EMBEDDINGS,
        use_inductor: bool = False,
        sharding_case: _ShardingCase = "row_wise",
        table_wise_owner_rank: int = 0,
        uniform_batch_per_rank: bool = False,
    ) -> None:
        torch.manual_seed(0)
        device = torch.device(f"cuda:{self.rank}")
        reference = _Model(device, num_embeddings)
        compiled = deepcopy(reference)
        self.assertEqual(patch_ebc_for_tracing(compiled), 1)

        compiled_optimizer: torch.optim.Optimizer
        reference_optimizer: torch.optim.Optimizer
        if not use_tbe:
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
        embedding_plan = _sharding_plan(
            compiled,
            sharding_case,
            table_wise_owner_rank=table_wise_owner_rank,
            world_size=self.world_size,
        )
        parallel_mode = HybridParallel(
            dense_style="replicate",
            embedding_plan=embedding_plan,
            sparse_transform_factory=(
                _tbe_sparse_transform_factory(embedding_plan) if use_tbe else None
            ),
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

        iteration_count = 3 if use_inductor else 2
        for iteration in range(iteration_count):
            batch_iteration = iteration % 2
            values, lengths, coefficient = _rank_batch(
                self.rank,
                device,
                batch_iteration,
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
                last_train_step=iteration == iteration_count - 1,
            )
            sync_device_state_to_torch(train_step)

            reference_optimizer.zero_grad()
            global_losses = []
            for rank in range(self.world_size):
                rank_values, rank_lengths, rank_coefficient = _rank_batch(
                    rank,
                    device,
                    batch_iteration,
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
                if embedding_plan.is_table_wise(name):
                    owner_rank = embedding_plan.table_wise_owner(name)
                    expected_local_parameter = (
                        reference_parameter
                        if self.rank == owner_rank
                        else reference_parameter.new_empty(
                            (0, reference_parameter.shape[1])
                        )
                    )
                else:
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

        if sharding_case == "table_wise_same_owner":
            tbe_groups = compiled.get_submodule("_recir_tbe_sparse_groups")
            self.assertEqual(len(dict(tbe_groups.named_children())), 2)
