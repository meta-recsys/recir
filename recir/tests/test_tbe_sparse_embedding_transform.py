# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import operator
import unittest
from collections import OrderedDict
from typing import Any
from unittest.mock import MagicMock, patch

import torch
import torch.fx as fx
import torch.nn as nn
from functorch import make_fx
from recir.distributed.sharding_plan import EmbeddingShardingPlan, EmbeddingShardingType
from recir.embedding.tbe.local import LocalTBEGroup
from recir.lowering.partial_lower import _InductorModule, partial_lower
from recir.ops.fake.kjt import patch_ebc_for_tracing
from recir.pipeline import build_pipeline, PipelinedGraphModule, SteadyStateGraphModule
from recir.pipeline_strategies import (
    embedding_steady_state_pipeline_config,
    sdd_pipeline_config,
    semi_sync_pipeline_config,
)
from recir.region_transforms.tbe_sparse_embedding import (
    _LookupSite,
    TBESparseEmbeddingTransform,
)
from torch._dynamo.source import LocalSource
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DeviceMesh
from torch.fx.experimental.symbolic_shapes import (
    DimDynamic,
    ShapeEnv,
    StatelessSymbolicContext,
)
from torchrec.modules.embedding_configs import EmbeddingBagConfig, PoolingType
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class _TwoTableModel(nn.Module):
    def __init__(self, rows_0: int = 8, rows_1: int = 6) -> None:
        super().__init__()
        self.table_0 = nn.Embedding(rows_0, 4)
        self.table_1 = nn.Embedding(rows_1, 4)


class _OneTableBagModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.table = nn.Embedding(8, 4)


class _TestRowWiseAdagrad(torch.optim.Optimizer):
    def __init__(
        self, parameters: list[nn.Parameter], learning_rate: torch.Tensor
    ) -> None:
        super().__init__(
            parameters,
            {
                "lr": learning_rate,
                "eps": 1.0e-8,
                "weight_decay": 0.0,
                "stochastic_rounding": False,
            },
        )

    def step(self, closure: Any = None) -> Any:
        del closure
        return None


def _mock_mesh(world_size: int, rank: int = 0) -> DeviceMesh:
    mesh = MagicMock(spec=DeviceMesh)
    mesh.ndim = 1
    mesh.shape = (world_size,)
    mesh.device_type = "cpu"
    mesh.mesh = torch.arange(world_size)
    mesh._flatten.return_value = mesh
    mesh.size.return_value = world_size
    mesh.get_coordinate.return_value = [rank]
    mesh._dim_group_names = ["fake_tbe_sparse_test_group"]
    return mesh


def _dynamic_fake_inputs(
    prefix: str,
    parameters: tuple[torch.Tensor, ...],
    inputs: tuple[torch.Tensor, ...],
) -> tuple[FakeTensorMode, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    fake_mode = FakeTensorMode(allow_non_fake_inputs=True, shape_env=ShapeEnv())
    fake_parameters = tuple(fake_mode.from_tensor(value) for value in parameters)
    fake_inputs = tuple(
        fake_mode.from_tensor(
            value,
            static_shapes=False,
            source=LocalSource(f"{prefix}_{index}"),
            symbolic_context=StatelessSymbolicContext(
                dynamic_sizes=[DimDynamic.DYNAMIC]
                + [DimDynamic.STATIC] * (value.ndim - 1)
            ),
        )
        for index, value in enumerate(inputs)
    )
    return fake_mode, fake_parameters, fake_inputs


def _embedding_graph(model: _TwoTableModel) -> fx.GraphModule:
    graph = fx.Graph()
    weight_0 = graph.placeholder("weight_0")
    weight_0.meta["val"] = model.table_0.weight
    weight_0.meta["fqn"] = "table_0.weight"
    weight_1 = graph.placeholder("weight_1")
    weight_1.meta["val"] = model.table_1.weight
    weight_1.meta["fqn"] = "table_1.weight"
    indices_0 = graph.placeholder("indices_0")
    indices_0.meta["val"] = torch.tensor([1, 3], dtype=torch.int64)
    indices_1 = graph.placeholder("indices_1")
    indices_1.meta["val"] = torch.tensor([0, 2, 5], dtype=torch.int64)
    output_0 = graph.call_function(
        torch.ops.aten.embedding.default, args=(weight_0, indices_0)
    )
    output_0.meta["val"] = torch.randn(2, 4)
    output_1 = graph.call_function(
        torch.ops.aten.embedding.default, args=(weight_1, indices_1)
    )
    output_1.meta["val"] = torch.randn(3, 4)
    graph.output((output_0, output_1))
    return fx.GraphModule(model, graph)


def _keyword_embedding_bag_graph(model: _OneTableBagModel) -> fx.GraphModule:
    graph = fx.Graph()
    weight = graph.placeholder("weight")
    weight.meta["val"] = model.table.weight
    weight.meta["fqn"] = "table.weight"
    indices = graph.placeholder("indices")
    indices.meta["val"] = torch.tensor([1, 3], dtype=torch.int64)
    offsets = graph.placeholder("offsets")
    offsets.meta["val"] = torch.tensor([0, 1, 2], dtype=torch.int64)
    embedding_bag = graph.call_function(
        torch.ops.aten._embedding_bag.default,
        kwargs={
            "weight": weight,
            "indices": indices,
            "offsets": offsets,
            "include_last_offset": True,
        },
    )
    output = graph.call_function(operator.getitem, args=(embedding_bag, 0))
    output.meta["val"] = torch.randn(2, 4)
    graph.output(output)
    return fx.GraphModule(model, graph)


class TBESparseEmbeddingTransformTest(unittest.TestCase):
    def test_recir_tbe_ops_have_meta_kernels(self) -> None:
        for op_name in (
            "bucketize_sparse_features",
            "dedup_sparse_features",
            "permute_sparse_data_1d",
            "permute_2d_sparse_data",
            "permute_sparse_data_2d",
            "pooled_source_to_feature_permute",
            "reshape_pooled_gradients",
            "sum_pooled_embeddings",
            "tbe_forward",
            "tbe_backward_update",
        ):
            table = torch._C._dispatch_dump_table(f"recir::{op_name}")
            self.assertIn("Meta:", table, table)

    def test_permute_sparse_data_2d_preserves_inner_segments(self) -> None:
        lengths = torch.tensor([[1, 2], [2, 0]], dtype=torch.int64)
        values = torch.tensor([10, 11, 12, 20, 21], dtype=torch.int64)
        permutation = torch.tensor([1, 0], dtype=torch.int32)

        actual_lengths, actual_values = torch.ops.recir.permute_sparse_data_2d(
            permutation, lengths, values
        )

        torch.testing.assert_close(
            actual_lengths,
            torch.tensor([[2, 0], [1, 2]], dtype=torch.int64),
        )
        torch.testing.assert_close(actual_values, torch.tensor([20, 21, 10, 11, 12]))

    def test_permute_sparse_data_1d_preserves_variable_segments(self) -> None:
        lengths = torch.tensor([1, 2, 0], dtype=torch.int64)
        values = torch.tensor([10, 20, 21], dtype=torch.int64)

        actual_lengths, actual_values = torch.ops.recir.permute_sparse_data_1d(
            torch.tensor([1, 0, 2], dtype=torch.int32),
            lengths,
            values,
        )

        torch.testing.assert_close(actual_lengths, torch.tensor([2, 1, 0]))
        torch.testing.assert_close(actual_values, torch.tensor([20, 21, 10]))

    def test_permute_2d_sparse_data_derives_stride_from_lengths(self) -> None:
        lengths = torch.tensor([1, 0, 2, 1, 0, 2], dtype=torch.int64)
        values = torch.tensor([10, 20, 21, 30, 40, 41], dtype=torch.int64)

        output_lengths, output_values, output_weights = (
            torch.ops.recir.permute_2d_sparse_data(
                torch.tensor([2, 0], dtype=torch.int32),
                lengths,
                values,
                3,
            )
        )

        torch.testing.assert_close(
            output_lengths,
            torch.tensor([0, 2, 1, 0], dtype=torch.int64),
        )
        torch.testing.assert_close(
            output_values,
            torch.tensor([40, 41, 10], dtype=torch.int64),
        )
        self.assertIsNone(output_weights)

    def test_permute_2d_sparse_data_rejects_uneven_feature_strides(self) -> None:
        args = (
            torch.tensor([0], dtype=torch.int32),
            torch.ones(5, dtype=torch.int64),
            torch.ones(5, dtype=torch.int64),
            3,
        )
        with self.assertRaisesRegex(ValueError, "same stride"):
            torch.ops.recir.permute_2d_sparse_data(*args)

        with FakeTensorMode(allow_non_fake_inputs=True, shape_env=ShapeEnv()):
            with self.assertRaisesRegex(ValueError, "same stride"):
                torch.ops.recir.permute_2d_sparse_data(*args)

    def test_permute_2d_sparse_data_requires_flat_lengths(self) -> None:
        args = (
            torch.tensor([0], dtype=torch.int32),
            torch.ones((1, 2), dtype=torch.int64),
            torch.ones(2, dtype=torch.int64),
            1,
        )
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            torch.ops.recir.permute_2d_sparse_data(*args)

        with FakeTensorMode(allow_non_fake_inputs=True):
            with self.assertRaisesRegex(ValueError, "one-dimensional"):
                torch.ops.recir.permute_2d_sparse_data(*args)

    def test_permute_2d_sparse_data_rejects_empty_feature_set(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            torch.ops.recir.permute_2d_sparse_data(
                torch.empty(0, dtype=torch.int32),
                torch.empty(0, dtype=torch.int64),
                torch.empty(0, dtype=torch.int64),
                0,
            )

    def test_pooled_permutation_handles_uneven_rank_batches(
        self,
    ) -> None:
        source_major_lengths = torch.empty(6, dtype=torch.int64)

        actual = torch.ops.recir.pooled_source_to_feature_permute(
            source_major_lengths,
            torch.tensor([2, 1], dtype=torch.int64),
            2,
        )

        torch.testing.assert_close(
            actual,
            torch.tensor([0, 1, 4, 2, 3, 5], dtype=torch.int64),
        )

    def test_pooled_permutation_handles_non_owner_empty_input(self) -> None:
        actual = torch.ops.recir.pooled_source_to_feature_permute(
            torch.empty(0, dtype=torch.int64),
            torch.zeros(2, dtype=torch.int64),
            2,
        )

        self.assertEqual(actual.shape, torch.Size((0,)))

    def _transform(
        self,
        world_size: int,
        *,
        rank: int = 0,
        model: _TwoTableModel | None = None,
        num_embeddings_map: dict[str, int] | None = None,
        use_index_dedup: bool = False,
        process_group: ProcessGroup | None = None,
        learning_rate_tensor: torch.Tensor | None = None,
        meta_optimizer_state: bool = False,
        optimizer_state_value: float | None = None,
        embedding_plan: EmbeddingShardingPlan | None = None,
    ) -> tuple[TBESparseEmbeddingTransform, _TwoTableModel, fx.GraphModule]:
        if model is None:
            model = _TwoTableModel()
        gm = _embedding_graph(model)
        optimizers: tuple[torch.optim.Optimizer, ...] = ()
        if learning_rate_tensor is not None:
            optimizers = (
                _TestRowWiseAdagrad(
                    list(model.parameters()),
                    learning_rate_tensor,
                ),
            )
        if meta_optimizer_state:
            if not optimizers:
                raise ValueError("Meta optimizer state requires an optimizer.")
            optimizer = optimizers[0]
            for parameter in model.parameters():
                optimizer.state[parameter]["sum"] = torch.empty(
                    (parameter.shape[0], 1),
                    dtype=torch.float32,
                    device="meta",
                )
        if optimizer_state_value is not None:
            if not optimizers:
                raise ValueError("Optimizer state requires an optimizer.")
            optimizer = optimizers[0]
            for parameter in model.parameters():
                optimizer.state[parameter]["sum"] = torch.full(
                    (parameter.shape[0], 1),
                    optimizer_state_value,
                    dtype=torch.float32,
                    device=parameter.device,
                )
        transform = TBESparseEmbeddingTransform(
            {"table_0.weight", "table_1.weight"},
            _mock_mesh(world_size, rank),
            model,
            process_group=process_group,
            embedding_plan=embedding_plan,
            num_embeddings_map=num_embeddings_map
            or {
                "table_0.weight": 8 * world_size,
                "table_1.weight": 6 * world_size,
            },
            stochastic_rounding=False,
            segments_per_feature=4,
            use_index_dedup=use_index_dedup,
        )
        transform.prepare_after_shard(model, optimizers)
        called: OrderedDict[str, tuple[nn.Module, tuple[Any, ...], dict[str, Any]]] = (
            OrderedDict({"": (model, (torch.tensor([1]), torch.tensor([2])), {})})
        )
        transform.select_forward_root_fqns(called)
        with patch(
            "fbgemm_gpu.config.feature_list.FeatureGate.is_enabled",
            return_value=False,
        ):
            transform.transform_graph(gm, called)
        return transform, model, gm

    def test_table_wise_rejects_unpooled_lookup(self) -> None:
        plan = EmbeddingShardingPlan(
            sharding_map={
                "table_0.weight": EmbeddingShardingType.TABLE_WISE,
                "table_1.weight": EmbeddingShardingType.TABLE_WISE,
            },
            table_wise_rank_map={
                "table_0.weight": 0,
                "table_1.weight": 0,
            },
        )

        with self.assertRaisesRegex(NotImplementedError, "pooled EBC"):
            self._transform(world_size=1, embedding_plan=plan)

    def test_plan_must_classify_each_embedding_weight(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not classify"):
            TBESparseEmbeddingTransform(
                {"table.weight"},
                _mock_mesh(1),
                embedding_plan=EmbeddingShardingPlan(),
            )

    def test_embedding_plan_supplies_storage_location(self) -> None:
        plan = EmbeddingShardingPlan(
            sharding_map={
                "table_0.weight": EmbeddingShardingType.ROW_WISE,
                "table_1.weight": EmbeddingShardingType.ROW_WISE,
            },
            location_map={"table_0.weight": "UVM"},
        )

        with self.assertRaisesRegex(ValueError, "CPU TBE only supports host storage"):
            self._transform(world_size=1, embedding_plan=plan)

    def test_row_wise_rejects_empty_local_table_shard(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "non-empty local"):
            self._transform(
                world_size=2,
                rank=1,
                model=_TwoTableModel(rows_0=0),
                num_embeddings_map={
                    "table_0.weight": 1,
                    "table_1.weight": 12,
                },
            )

    def _pooled_transform(
        self,
        world_size: int,
        *,
        table_wise: bool = False,
    ) -> tuple[
        TBESparseEmbeddingTransform,
        fx.GraphModule,
        tuple[torch.Tensor, ...],
        torch.Tensor,
        torch.Tensor,
    ]:
        ebc = EmbeddingBagCollection(
            tables=[
                EmbeddingBagConfig(
                    name="table_0",
                    embedding_dim=4,
                    num_embeddings=8 if table_wise else 8 // world_size,
                    feature_names=["feature_0"],
                    pooling=PoolingType.SUM,
                ),
                EmbeddingBagConfig(
                    name="table_1",
                    embedding_dim=4,
                    num_embeddings=6 if table_wise else 6 // world_size,
                    feature_names=["feature_1"],
                    pooling=PoolingType.MEAN,
                ),
            ],
            device=torch.device("cpu"),
        )
        patch_ebc_for_tracing(ebc)
        parameter_items = tuple(ebc.named_parameters())
        parameter_fqns = tuple(name for name, _ in parameter_items)
        parameters = tuple(parameter for _, parameter in parameter_items)
        embedding_plan = (
            EmbeddingShardingPlan(
                sharding_map={
                    fqn: EmbeddingShardingType.TABLE_WISE for fqn in parameter_fqns
                },
                table_wise_rank_map=dict.fromkeys(parameter_fqns, 0),
            )
            if table_wise
            else None
        )
        values = torch.tensor([1, 2], dtype=torch.int64)
        lengths = torch.tensor([1, 0, 0, 1], dtype=torch.int64)

        def forward(
            weight_0: torch.Tensor,
            weight_1: torch.Tensor,
            input_values: torch.Tensor,
            input_lengths: torch.Tensor,
        ) -> torch.Tensor:
            features = KeyedJaggedTensor(
                keys=["feature_0", "feature_1"],
                values=input_values,
                lengths=input_lengths,
            )
            return torch.func.functional_call(
                ebc,
                dict(zip(parameter_fqns, (weight_0, weight_1))),
                (features,),
            ).values()

        gm = make_fx(
            forward,
            tracing_mode="fake",
            _allow_non_fake_inputs=True,
        )(*parameters, values, lengths)
        transform = TBESparseEmbeddingTransform(
            set(parameter_fqns),
            _mock_mesh(world_size),
            ebc,
            embedding_plan=embedding_plan,
            num_embeddings_map={
                parameter_fqns[0]: 8,
                parameter_fqns[1]: 6,
            },
            stochastic_rounding=False,
        )
        transform.prepare_after_shard(ebc, ())
        called: OrderedDict[str, tuple[nn.Module, tuple[Any, ...], dict[str, Any]]] = (
            OrderedDict({"": (ebc, (), {})})
        )
        transform.select_forward_root_fqns(called)
        with patch(
            "fbgemm_gpu.config.feature_list.FeatureGate.is_enabled",
            return_value=False,
        ):
            transform.transform_graph(gm, called)
        return transform, gm, parameters, values, lengths

    def test_inner_transform_reuses_optimizer_learning_rate_tensor(self) -> None:
        learning_rate = torch.tensor(0.1, dtype=torch.float32, device="cpu")

        _transform, _model, gm = self._transform(
            world_size=1,
            learning_rate_tensor=learning_rate,
        )

        group = gm.get_submodule("_recir_tbe_group_0")
        self.assertIs(group.get_buffer("_recir_learning_rate"), learning_rate)

    def test_inner_transform_initializes_meta_optimizer_state(self) -> None:
        _transform, _model, gm = self._transform(
            world_size=1,
            learning_rate_tensor=torch.tensor(0.1, dtype=torch.float32),
            meta_optimizer_state=True,
        )

        group: Any = gm.get_submodule("_recir_tbe_group_0")
        optimizer_states = group.split_optimizer_states()
        self.assertEqual(len(optimizer_states), 2)
        for table_states in optimizer_states:
            self.assertEqual(len(table_states), 1)
            torch.testing.assert_close(
                table_states[0],
                torch.zeros_like(table_states[0]),
            )

    def test_inner_transform_preserves_materialized_optimizer_state(self) -> None:
        transform, model, gm = self._transform(
            world_size=1,
            learning_rate_tensor=torch.tensor(0.1, dtype=torch.float32),
            optimizer_state_value=2.5,
        )

        group: Any = gm.get_submodule("_recir_tbe_group_0")
        for table_states in group.split_optimizer_states():
            self.assertEqual(len(table_states), 1)
            torch.testing.assert_close(
                table_states[0],
                torch.full_like(table_states[0], 2.5),
            )

        optimizer = transform._optimizers[0]
        parameters = (model.table_0.weight, model.table_1.weight)
        for parameter, table_states in zip(parameters, group.split_optimizer_states()):
            self.assertTrue(
                any(
                    parameter is optimizer_parameter
                    for optimizer_parameter in optimizer.param_groups[0]["params"]
                )
            )
            checkpoint_state = optimizer.state[parameter]["sum"]
            checkpoint_state.fill_(4.0)
            torch.testing.assert_close(
                table_states[0],
                torch.full_like(table_states[0], 4.0),
            )
        serialized = optimizer.state_dict()
        self.assertEqual(len(serialized["state"]), 2)
        self.assertEqual(len(serialized["param_groups"][0]["params"]), 2)

    def test_transform_inputs_materializes_kjt_offsets(self) -> None:
        transform = TBESparseEmbeddingTransform(set(), _mock_mesh(1))
        features = KeyedJaggedTensor(
            keys=["feature"],
            values=torch.tensor([1, 2], dtype=torch.int64),
            lengths=torch.tensor([1, 1], dtype=torch.int64),
            stride=2,
        )

        args, kwargs = transform.transform_inputs((features,), {})

        self.assertIs(args[0], features)
        self.assertEqual(kwargs, {})
        torch.testing.assert_close(
            features.offsets(), torch.tensor([0, 1, 2], dtype=torch.int64)
        )

    def test_dedup_hash_metadata_covers_shared_table_features(self) -> None:
        transform = TBESparseEmbeddingTransform(
            {"table_0.weight", "table_1.weight"},
            _mock_mesh(2),
            num_embeddings_map={"table_0.weight": 16, "table_1.weight": 12},
        )
        node = fx.Graph().placeholder("feature")
        sites = (
            _LookupSite(node, node, "table_0.weight"),
            _LookupSite(node, node, "table_0.weight"),
            _LookupSite(node, node, "table_1.weight"),
        )

        self.assertEqual(
            transform._dedup_hash_size_cumsum(sites),
            (0, 0, 16, 28),
        )
        self.assertEqual(
            transform._dedup_hash_size_offset(sites),
            (0, 2, 2, 3),
        )

    def test_inner_transform_groups_local_lookups(self) -> None:
        _transform, model, gm = self._transform(world_size=1)

        embedding_nodes = [
            node
            for node in gm.graph.nodes
            if node.op == "call_function"
            and node.target == torch.ops.aten.embedding.default
        ]
        grouped_calls = [node for node in gm.graph.nodes if node.op == "call_module"]
        self.assertEqual(embedding_nodes, [])
        self.assertEqual(len(grouped_calls), 1)
        indices_0 = torch.tensor([1, 3], dtype=torch.int64)
        indices_1 = torch.tensor([0, 2, 5], dtype=torch.int64)
        actual_0, actual_1 = gm(
            model.table_0.weight,
            model.table_1.weight,
            indices_0,
            indices_1,
        )
        torch.testing.assert_close(actual_0, model.table_0.weight[indices_0])
        torch.testing.assert_close(actual_1, model.table_1.weight[indices_1])

        weight_0_before = model.table_0.weight.detach().clone()
        weight_1_before = model.table_1.weight.detach().clone()
        (actual_0.sum() + actual_1.sum()).backward()
        self.assertFalse(torch.equal(model.table_0.weight, weight_0_before))
        self.assertFalse(torch.equal(model.table_1.weight, weight_1_before))

    def test_pooled_transform_accepts_keyword_embedding_bag_arguments(self) -> None:
        model = _OneTableBagModel()
        gm = _keyword_embedding_bag_graph(model)
        transform = TBESparseEmbeddingTransform(
            {"table.weight"},
            _mock_mesh(1),
            model,
            stochastic_rounding=False,
            segments_per_feature=4,
        )
        transform.prepare_after_shard(model, ())
        called: OrderedDict[str, tuple[nn.Module, tuple[Any, ...], dict[str, Any]]] = (
            OrderedDict({"": (model, (), {})})
        )
        transform.select_forward_root_fqns(called)

        with patch(
            "fbgemm_gpu.config.feature_list.FeatureGate.is_enabled",
            return_value=False,
        ):
            transform.transform_graph(gm, called)

        indices = torch.tensor([1, 3], dtype=torch.int64)
        offsets = torch.tensor([0, 1, 2], dtype=torch.int64)
        actual = gm(model.table.weight, indices, offsets)
        expected = torch.nn.functional.embedding_bag(
            indices,
            model.table.weight,
            offsets,
            mode="sum",
            include_last_offset=True,
        )
        torch.testing.assert_close(actual, expected)

    def test_unpooled_transform_accepts_changed_feature_lengths(self) -> None:
        _transform, model, gm = self._transform(world_size=1)

        indices_0 = torch.tensor([1, 1, 3, 4, 2], dtype=torch.int64)
        indices_1 = torch.tensor([5], dtype=torch.int64)
        actual_0, actual_1 = gm(
            model.table_0.weight,
            model.table_1.weight,
            indices_0,
            indices_1,
        )

        torch.testing.assert_close(actual_0, model.table_0.weight[indices_0])
        torch.testing.assert_close(actual_1, model.table_1.weight[indices_1])

    def test_patched_ebc_preserves_mixed_dims_and_empty_feature_after_tbe_rewrite(
        self,
    ) -> None:
        ebc = EmbeddingBagCollection(
            tables=[
                EmbeddingBagConfig(
                    name="table_0",
                    embedding_dim=4,
                    num_embeddings=8,
                    feature_names=["feature_0"],
                    pooling=PoolingType.SUM,
                ),
                EmbeddingBagConfig(
                    name="table_1",
                    embedding_dim=8,
                    num_embeddings=6,
                    feature_names=["feature_1"],
                    pooling=PoolingType.MEAN,
                ),
            ],
            device=torch.device("cpu"),
        )
        patch_ebc_for_tracing(ebc)
        parameter_items = tuple(ebc.named_parameters())
        parameter_fqns = tuple(name for name, _ in parameter_items)
        parameters = tuple(parameter for _, parameter in parameter_items)
        values = torch.tensor([4, 5, 1, 3, 2, 0, 2], dtype=torch.int64)
        lengths = torch.tensor([1, 0, 1, 2, 0, 1, 0, 1, 1], dtype=torch.int64)

        def forward(
            weight_0: torch.Tensor,
            weight_1: torch.Tensor,
            input_values: torch.Tensor,
            input_lengths: torch.Tensor,
        ) -> torch.Tensor:
            features = KeyedJaggedTensor(
                keys=["unused", "feature_0", "feature_1"],
                values=input_values,
                lengths=input_lengths,
            )
            pooled = torch.func.functional_call(
                ebc,
                dict(zip(parameter_fqns, (weight_0, weight_1))),
                (features.permute([1, 2]),),
            ).values()
            dense = input_lengths.view(3, -1)[0].to(weight_0.dtype).unsqueeze(1)
            return torch.cat((pooled, dense), dim=1)

        weight_0, weight_1 = parameters
        expected = forward(weight_0, weight_1, values, lengths)
        fake_mode, fake_parameters, fake_inputs = _dynamic_fake_inputs(
            "pooled_tbe_input", parameters, (values, lengths)
        )
        with fake_mode:
            gm = make_fx(
                forward,
                tracing_mode="fake",
                _allow_non_fake_inputs=False,
            )(*fake_parameters, *fake_inputs)
        transform = TBESparseEmbeddingTransform(
            set(parameter_fqns),
            _mock_mesh(1),
            ebc,
            num_embeddings_map={
                fqn: 8 if index == 0 else 6 for index, fqn in enumerate(parameter_fqns)
            },
            stochastic_rounding=False,
        )
        transform.prepare_after_shard(ebc, ())
        called: OrderedDict[str, tuple[nn.Module, tuple[Any, ...], dict[str, Any]]] = (
            OrderedDict({"": (ebc, (), {})})
        )
        transform.select_forward_root_fqns(called)

        with patch(
            "fbgemm_gpu.config.feature_list.FeatureGate.is_enabled",
            return_value=False,
        ):
            transform.transform_graph(gm, called)
            with fake_mode:
                transformed = make_fx(
                    gm,
                    tracing_mode="fake",
                    _allow_non_fake_inputs=False,
                )(*fake_parameters, *fake_inputs)
        placeholders = [
            node for node in transformed.graph.nodes if node.op == "placeholder"
        ]
        for placeholder in placeholders[:2]:
            placeholder.meta["is_batch_input"] = False
        for placeholder in placeholders[2:]:
            placeholder.meta["is_batch_input"] = True
            placeholder.meta["batch_dim"] = 0
        lowered = partial_lower(
            transformed,
            node_predicate=lambda node: node.op == "call_function",
            subgraph_predicate=lambda _nodes: True,
        )
        lowered_targets = {
            node.target
            for module in lowered.modules()
            if isinstance(module, _InductorModule)
            for node in module.gm.graph.nodes
        }
        self.assertIn(
            torch.ops.recir_torchrec.kjt_split_per_feature_with_offsets.default,
            lowered_targets,
        )
        self.assertIn(
            torch.ops.recir.permute_2d_sparse_data.default,
            lowered_targets,
        )
        actual = lowered(weight_0, weight_1, values, lengths)

        torch.testing.assert_close(actual, expected)

        resized_values = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
        resized_lengths = torch.tensor([0, 0, 0, 0, 2, 2], dtype=torch.int64)
        torch.testing.assert_close(
            lowered(weight_0, weight_1, resized_values, resized_lengths),
            forward(weight_0, weight_1, resized_values, resized_lengths),
        )

        empty_values = torch.empty(0, dtype=torch.int64)
        empty_lengths = torch.zeros(6, dtype=torch.int64)
        torch.testing.assert_close(
            lowered(weight_0, weight_1, empty_values, empty_lengths),
            forward(weight_0, weight_1, empty_values, empty_lengths),
        )

    def test_pooled_outer_transform_builds_pipeline_stages(self) -> None:
        transform, gm, parameters, values, lengths = self._pooled_transform(
            world_size=2
        )

        def train_step(
            weight_0: torch.Tensor,
            weight_1: torch.Tensor,
            input_values: torch.Tensor,
            input_lengths: torch.Tensor,
        ) -> torch.Tensor:
            output = gm(weight_0, weight_1, input_values, input_lengths)
            output.sum().backward()
            return output.detach()

        with patch(
            "fbgemm_gpu.config.feature_list.FeatureGate.is_enabled",
            return_value=False,
        ):
            joint = make_fx(
                train_step,
                tracing_mode="fake",
                _allow_non_fake_inputs=True,
            )(*parameters, values, lengths)
        transform.finalize_compiled_graph(joint)

        cond_nodes = [
            node
            for node in joint.graph.nodes
            if node.op == "call_function" and node.target is torch.ops.higher_order.cond
        ]
        self.assertEqual(len(cond_nodes), 2)
        for cond_node in cond_nodes:
            for branch in cond_node.args[1:3]:
                if not isinstance(branch, fx.Node):
                    raise AssertionError(
                        f"Expected cond branch get_attr node, got {type(branch)}."
                    )
                self.assertEqual(branch.op, "get_attr")
                if not isinstance(branch.target, str):
                    raise AssertionError(
                        f"Expected string get_attr target, got {type(branch.target)}."
                    )
                self.assertIn(".", branch.target)
                self.assertIsInstance(
                    joint.get_submodule(branch.target),
                    fx.GraphModule,
                )

        sdd = build_pipeline(joint, sdd_pipeline_config())
        self.assertIsInstance(sdd, PipelinedGraphModule)
        self.assertEqual(len(sdd.stage_graph_modules()), 3)

        semi_sync = build_pipeline(
            joint,
            semi_sync_pipeline_config(start_batch=0),
        )
        self.assertIsInstance(semi_sync, PipelinedGraphModule)
        stages = semi_sync.stage_graph_modules()
        self.assertEqual(len(stages), 4)
        stage_targets = [
            [node.target for node in stage.graph.nodes] for stage in stages
        ]
        recursive_stage_targets = [
            [
                node.target
                for submodule in stage.modules()
                if isinstance(submodule, fx.GraphModule)
                for node in submodule.graph.nodes
            ]
            for stage in stages
        ]
        self.assertEqual(
            stage_targets[0].count(
                torch.ops._c10d_functional.all_to_all_single.default
            ),
            1,
        )
        self.assertEqual(
            stage_targets[1].count(
                torch.ops._c10d_functional.all_to_all_single.default
            ),
            1,
        )
        self.assertIn(torch.ops.higher_order.cond, stage_targets[2])
        self.assertIn(
            torch.ops._c10d_functional.reduce_scatter_tensor.default,
            recursive_stage_targets[2],
        )
        self.assertIn(
            torch.ops._c10d_functional.all_to_all_single.default,
            recursive_stage_targets[2],
        )
        self.assertIn(torch.ops.recir.tbe_forward.default, stage_targets[2])
        self.assertIn(torch.ops.higher_order.cond, stage_targets[3])
        self.assertIn(
            torch.ops._c10d_functional.all_gather_into_tensor.default,
            recursive_stage_targets[3],
        )
        self.assertIn(
            torch.ops._c10d_functional.all_to_all_single.default,
            recursive_stage_targets[3],
        )
        self.assertIn(torch.ops.recir.tbe_backward_update.default, stage_targets[3])

    def test_table_wise_pooled_transform_uses_native_pooling_and_pipeline(
        self,
    ) -> None:
        transform, gm, parameters, values, lengths = self._pooled_transform(
            world_size=2,
            table_wise=True,
        )
        local_groups = [
            module for module in gm.modules() if isinstance(module, LocalTBEGroup)
        ]
        self.assertCountEqual(
            [group.pooling_mode for group in local_groups],
            ["sum", "mean"],
        )

        def train_step(
            weight_0: torch.Tensor,
            weight_1: torch.Tensor,
            input_values: torch.Tensor,
            input_lengths: torch.Tensor,
        ) -> torch.Tensor:
            output = gm(weight_0, weight_1, input_values, input_lengths)
            output.sum().backward()
            return output.detach()

        with patch(
            "fbgemm_gpu.config.feature_list.FeatureGate.is_enabled",
            return_value=False,
        ):
            joint = make_fx(
                train_step,
                tracing_mode="fake",
                _allow_non_fake_inputs=True,
            )(*parameters, values, lengths)
        transform.finalize_compiled_graph(joint)

        sdd = build_pipeline(joint, sdd_pipeline_config())
        self.assertIsInstance(sdd, PipelinedGraphModule)
        self.assertEqual(len(sdd.stage_graph_modules()), 3)
        semi_sync = build_pipeline(
            joint,
            semi_sync_pipeline_config(start_batch=0),
        )
        self.assertIsInstance(semi_sync, PipelinedGraphModule)
        stages = semi_sync.stage_graph_modules()
        self.assertEqual(len(stages), 4)
        stage_targets = [
            [node.target for node in stage.graph.nodes] for stage in stages
        ]
        self.assertIn(
            torch.ops._c10d_functional.all_to_all_single.default,
            stage_targets[0],
        )
        self.assertIn(
            torch.ops._c10d_functional.all_to_all_single.default,
            stage_targets[1],
        )
        self.assertIn(torch.ops.recir.tbe_forward.default, stage_targets[2])
        self.assertIn(torch.ops.recir.tbe_backward_update.default, stage_targets[3])

    def test_pooled_outer_transform_preserves_source_bags_and_topology(
        self,
    ) -> None:
        transform, gm, parameters, values, lengths = self._pooled_transform(
            world_size=2
        )

        def train_step(
            weight_0: torch.Tensor,
            weight_1: torch.Tensor,
            input_values: torch.Tensor,
            input_lengths: torch.Tensor,
        ) -> torch.Tensor:
            output = gm(weight_0, weight_1, input_values, input_lengths)
            output.sum().backward()
            return output.detach()

        with patch(
            "fbgemm_gpu.config.feature_list.FeatureGate.is_enabled",
            return_value=False,
        ):
            joint = make_fx(
                train_step,
                tracing_mode="fake",
                _allow_non_fake_inputs=True,
            )(*parameters, values, lengths)
        forward = next(
            node
            for node in joint.graph.nodes
            if node.target == torch.ops.recir.tbe_forward.default
        )
        forward_output = next(
            node
            for node in forward.users
            if node.target == operator.getitem and node.args[1] == 0
        )
        source_offsets = forward.args[6]
        self.assertIsInstance(source_offsets, fx.Node)
        source_offsets.meta["val"] = torch.empty(7, dtype=torch.int64)
        graph_output = next(node for node in joint.graph.nodes if node.op == "output")
        with joint.graph.inserting_before(graph_output):
            late_alias = joint.graph.call_function(
                torch.ops.aten.alias.default,
                args=(forward_output,),
            )
            late_alias.meta = dict(forward_output.meta)
        positions = {node: index for index, node in enumerate(joint.graph.nodes)}
        earliest_user = min(forward_output.users, key=positions.__getitem__)
        with joint.graph.inserting_before(earliest_user):
            early_alias = joint.graph.call_function(
                torch.ops.aten.alias.default,
                args=(forward_output,),
            )
            early_alias.meta = dict(forward_output.meta)
        earliest_user.replace_input_with(forward_output, early_alias)

        transform.finalize_compiled_graph(joint)

        updated_positions = {
            node: index for index, node in enumerate(joint.graph.nodes)
        }
        routed_output = early_alias.args[0]
        self.assertIsInstance(routed_output, fx.Node)
        self.assertLess(
            updated_positions[routed_output], updated_positions[early_alias]
        )
        self.assertEqual(
            routed_output.meta["val"].shape,
            torch.Size((3, 8)),
        )
        targets = [
            str(node.target)
            for submodule in joint.modules()
            if isinstance(submodule, fx.GraphModule)
            for node in submodule.graph.nodes
        ]
        self.assertEqual(
            sum(
                "_c10d_functional.reduce_scatter_tensor" in target for target in targets
            ),
            1,
        )
        self.assertEqual(
            sum(
                "_c10d_functional.all_gather_into_tensor" in target
                for target in targets
            ),
            2,
        )
        self.assertEqual(
            sum("_c10d_functional.all_to_all_single" in target for target in targets),
            4,
        )

    def test_inner_transform_accepts_short_final_shard(self) -> None:
        local_model = _TwoTableModel(rows_0=7)
        _transform, model, gm = self._transform(
            world_size=2,
            rank=1,
            model=local_model,
            num_embeddings_map={
                "table_0.weight": 15,
                "table_1.weight": 12,
            },
        )
        indices_0 = torch.tensor([0, 6], dtype=torch.int64)
        indices_1 = torch.tensor([0, 5], dtype=torch.int64)

        actual_0, actual_1 = gm(
            model.table_0.weight,
            model.table_1.weight,
            indices_0,
            indices_1,
        )

        torch.testing.assert_close(actual_0, model.table_0.weight[indices_0])
        torch.testing.assert_close(actual_1, model.table_1.weight[indices_1])

    def test_inner_transform_rejects_non_chunk_shard(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "expected_local_rows=7"):
            self._transform(
                world_size=2,
                rank=1,
                model=_TwoTableModel(rows_0=6),
                num_embeddings_map={
                    "table_0.weight": 15,
                    "table_1.weight": 12,
                },
            )

    def test_outer_transform_keeps_collectives_and_fused_tbe(self) -> None:
        process_group = MagicMock(spec=ProcessGroup)
        process_group.size.return_value = 2
        process_group.group_name = "separate_tbe_test_group"
        transform, model, gm = self._transform(
            world_size=2,
            num_embeddings_map={
                "table_0.weight": 15,
                "table_1.weight": 11,
            },
            process_group=process_group,
        )

        def train_step(
            indices_0: torch.Tensor,
            indices_1: torch.Tensor,
        ) -> torch.Tensor:
            output_0, output_1 = gm(
                model.table_0.weight,
                model.table_1.weight,
                indices_0,
                indices_1,
            )
            output = output_0.sum() + output_1.sum()
            output.backward()
            return output.detach()

        with patch(
            "fbgemm_gpu.config.feature_list.FeatureGate.is_enabled",
            return_value=False,
        ):
            joint = make_fx(
                train_step,
                tracing_mode="fake",
                _allow_non_fake_inputs=True,
            )(
                torch.tensor([1, 3], dtype=torch.int64),
                torch.tensor([0, 2, 5], dtype=torch.int64),
            )
        transform.finalize_compiled_graph(joint)
        targets = [str(node.target) for node in joint.graph.nodes]
        forward_nodes = [
            node
            for node in joint.graph.nodes
            if node.target == torch.ops.recir.tbe_forward.default
        ]
        self.assertEqual(len(forward_nodes), 1)
        offsets = forward_nodes[0].args[6]
        if not isinstance(offsets, fx.Node):
            raise AssertionError(f"Expected FX offsets node, got {type(offsets)}.")
        self.assertEqual(offsets.meta["val"].shape, (17,))
        self.assertEqual(
            sum("recir.tbe_backward_update" in target for target in targets),
            1,
        )
        all_to_all_nodes = [
            node
            for node in joint.graph.nodes
            if node.target == torch.ops._c10d_functional.all_to_all_single.default
        ]
        self.assertEqual(len(all_to_all_nodes), 4)
        self.assertTrue(
            all(node.args[3] == process_group.group_name for node in all_to_all_nodes)
        )

        steady = build_pipeline(joint, embedding_steady_state_pipeline_config())
        self.assertIsInstance(steady, SteadyStateGraphModule)
        crossing_collectives = [
            (node, user)
            for node in all_to_all_nodes
            for user in node.users
            if user.target == torch.ops._c10d_functional.wait_tensor.default
            and steady._node_stages[node] == 0
            and steady._node_stages[user] == 1
        ]
        self.assertEqual(len(crossing_collectives), 1)
        pending_indices, _ = crossing_collectives[0]
        bucketized_indices = pending_indices.args[0]
        if not isinstance(bucketized_indices, fx.Node):
            raise AssertionError(
                f"Expected FX collective input, got {type(bucketized_indices)}."
            )
        self.assertIn(pending_indices, steady._live_after[0])
        self.assertIn(bucketized_indices, steady._live_after[0])
        self.assertEqual(steady._node_stages[forward_nodes[0]], 1)

        pipeline = build_pipeline(joint, sdd_pipeline_config())
        self.assertIsInstance(pipeline, PipelinedGraphModule)
        stages = pipeline.stage_graph_modules()
        self.assertEqual(len(stages), 3)
        stage_targets = [
            [str(node.target) for node in stage.graph.nodes] for stage in stages
        ]
        self.assertEqual(
            sum(
                "_c10d_functional.all_to_all_single" in target
                for target in stage_targets[0]
            ),
            1,
        )
        self.assertEqual(
            sum(
                "_c10d_functional.all_to_all_single" in target
                for target in stage_targets[1]
            ),
            1,
        )
        self.assertTrue(any("aten._to_copy" in target for target in stage_targets[1]))

        del stage_targets, stages, pipeline
        semi_sync = build_pipeline(
            joint,
            semi_sync_pipeline_config(start_batch=0),
        )
        self.assertIsInstance(semi_sync, PipelinedGraphModule)
        semi_sync_stages = semi_sync.stage_graph_modules()
        self.assertEqual(len(semi_sync_stages), 4)
        self.assertTrue(
            any(
                node.target == torch.ops.recir.tbe_forward.default
                for node in semi_sync_stages[2].graph.nodes
            )
        )
        self.assertTrue(
            any(
                node.target == torch.ops._c10d_functional.all_to_all_single.default
                for node in semi_sync_stages[2].graph.nodes
            )
        )
        self.assertTrue(
            any(
                node.target == torch.ops.recir.tbe_backward_update.default
                for node in semi_sync_stages[3].graph.nodes
            )
        )

        bucketize_nodes = [
            node
            for node in joint.graph.nodes
            if node.target == torch.ops.recir.bucketize_sparse_features.default
        ]
        self.assertEqual(len(bucketize_nodes), 1)
        block_sizes = bucketize_nodes[0].args[2]
        if not isinstance(block_sizes, fx.Node):
            raise AssertionError(
                f"Expected FX block-sizes node, got {type(block_sizes)}."
            )
        self.assertEqual(block_sizes.target, torch.ops.aten.stack.default)
        block_size_parts = block_sizes.args[0]
        if not isinstance(block_size_parts, (list, tuple)):
            raise AssertionError("Expected one block size per routed feature.")
        block_size_values: list[Any] = []
        for part in block_size_parts:
            if not isinstance(part, fx.Node):
                raise AssertionError(f"Expected FX block-size node, got {type(part)}.")
            block_size_values.append(part.args[2])
        self.assertEqual(tuple(block_size_values), (8, 6))
        packed_lengths = bucketize_nodes[0].args[0]
        if not isinstance(packed_lengths, fx.Node):
            raise AssertionError(
                f"Expected FX packed-lengths node, got {type(packed_lengths)}."
            )
        self.assertEqual(packed_lengths.meta["val"].shape, (8,))
        permute_nodes = [
            node
            for node in joint.graph.nodes
            if node.target == torch.ops.recir.permute_sparse_data_2d.default
        ]
        self.assertEqual(len(permute_nodes), 3)
        for node in permute_nodes:
            lengths = node.args[1]
            if not isinstance(lengths, fx.Node):
                raise AssertionError(f"Expected FX lengths node, got {type(lengths)}.")
            self.assertEqual(len(lengths.meta["val"].shape), 2)
            self.assertEqual(lengths.meta["val"].shape[1], 4)
        self.assertFalse(any("aten.argsort" in target for target in targets))
        self.assertFalse(
            any("embedding_dense_backward" in target for target in targets)
        )

    def test_outer_transform_deduplicates_before_routing(self) -> None:
        transform, model, gm = self._transform(world_size=2, use_index_dedup=True)

        def train_step(
            indices_0: torch.Tensor,
            indices_1: torch.Tensor,
        ) -> torch.Tensor:
            output_0, output_1 = gm(
                model.table_0.weight,
                model.table_1.weight,
                indices_0,
                indices_1,
            )
            output = output_0.sum() + output_1.sum()
            output.backward()
            return output.detach()

        with patch(
            "fbgemm_gpu.config.feature_list.FeatureGate.is_enabled",
            return_value=False,
        ):
            joint = make_fx(
                train_step,
                tracing_mode="fake",
                _allow_non_fake_inputs=True,
            )(
                torch.tensor([1, 1], dtype=torch.int64),
                torch.tensor([0, 2, 2], dtype=torch.int64),
            )
        transform.finalize_compiled_graph(joint)

        targets = [str(node.target) for node in joint.graph.nodes]
        self.assertEqual(
            sum("recir.dedup_sparse_features" in target for target in targets), 1
        )
        dedup_node = next(
            node
            for node in joint.graph.nodes
            if node.target == torch.ops.recir.dedup_sparse_features.default
        )
        for metadata_node in dedup_node.args[2:]:
            if not isinstance(metadata_node, fx.Node):
                raise AssertionError(
                    f"Expected FX hash metadata node, got {type(metadata_node)}."
                )
            self.assertEqual(metadata_node.op, "get_attr")
        index_select_nodes = [
            node
            for node in joint.graph.nodes
            if node.target == torch.ops.aten.index_select.default
        ]
        self.assertEqual(len(index_select_nodes), 1)
        reverse_indices = index_select_nodes[0].args[2]
        if not isinstance(reverse_indices, fx.Node):
            raise AssertionError(
                f"Expected FX reverse-indices node, got {type(reverse_indices)}."
            )
        self.assertEqual(reverse_indices.target, torch.ops.aten.view.default)
        self.assertEqual(sum("aten.index_add" in target for target in targets), 1)
        self.assertEqual(
            sum("_c10d_functional.all_to_all_single" in target for target in targets),
            4,
        )
