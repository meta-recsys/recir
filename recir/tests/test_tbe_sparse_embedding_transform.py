# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from collections import OrderedDict
from typing import Any
from unittest.mock import MagicMock, patch

import torch
import torch.fx as fx
import torch.nn as nn
from functorch import make_fx
from recir.pipeline import build_pipeline, PipelinedGraphModule
from recir.pipeline_strategies import sdd_pipeline_config, semi_sync_pipeline_config
from recir.region_transforms.tbe_sparse_embedding import TBESparseEmbeddingTransform
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DeviceMesh
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class _TwoTableModel(nn.Module):
    def __init__(self, rows_0: int = 8, rows_1: int = 6) -> None:
        super().__init__()
        self.table_0 = nn.Embedding(rows_0, 4)
        self.table_1 = nn.Embedding(rows_1, 4)


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


class TBESparseEmbeddingTransformTest(unittest.TestCase):
    def test_recir_tbe_ops_have_meta_kernels(self) -> None:
        for op_name in (
            "bucketize_sparse_features",
            "dedup_sparse_features",
            "permute_sparse_data_2d",
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
        transform = TBESparseEmbeddingTransform(
            {"table_0.weight", "table_1.weight"},
            _mock_mesh(world_size, rank),
            model,
            process_group=process_group,
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
            (node, node, "table_0.weight"),
            (node, node, "table_0.weight"),
            (node, node, "table_1.weight"),
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
            sum("recir.tbe_backward_update" in target for target in targets), 1
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
