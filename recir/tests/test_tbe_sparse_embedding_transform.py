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
from recir.region_transforms.tbe_sparse_embedding import TBESparseEmbeddingTransform
from torch.distributed.tensor import DeviceMesh
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class _TwoTableModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.table_0 = nn.Embedding(8, 4)
        self.table_1 = nn.Embedding(6, 4)


def _mock_mesh(world_size: int) -> DeviceMesh:
    mesh = MagicMock(spec=DeviceMesh)
    mesh.ndim = 1
    mesh.shape = (world_size,)
    mesh.device_type = "cpu"
    mesh.mesh = torch.arange(world_size)
    mesh._flatten.return_value = mesh
    mesh.size.return_value = world_size
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
        for op_name in ("tbe_forward", "tbe_backward_update"):
            table = torch._C._dispatch_dump_table(f"recir::{op_name}")
            self.assertIn("Meta:", table, table)

    def _transform(
        self, world_size: int
    ) -> tuple[TBESparseEmbeddingTransform, _TwoTableModel, fx.GraphModule]:
        model = _TwoTableModel()
        gm = _embedding_graph(model)
        transform = TBESparseEmbeddingTransform(
            {"table_0.weight", "table_1.weight"},
            _mock_mesh(world_size),
            model,
            num_embeddings_map={
                "table_0.weight": 8 * world_size,
                "table_1.weight": 6 * world_size,
            },
            stochastic_rounding=False,
        )
        transform.prepare_after_shard(model, ())
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

    def test_outer_transform_keeps_collectives_and_fused_tbe(self) -> None:
        transform, model, gm = self._transform(world_size=2)

        def train_step(
            indices_0: torch.Tensor, indices_1: torch.Tensor
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
        self.assertEqual(sum("recir.tbe_forward" in target for target in targets), 1)
        self.assertEqual(
            sum("recir.tbe_backward_update" in target for target in targets), 1
        )
        self.assertEqual(
            sum("_c10d_functional.all_to_all_single" in target for target in targets),
            4,
        )
        self.assertFalse(
            any("embedding_dense_backward" in target for target in targets)
        )
