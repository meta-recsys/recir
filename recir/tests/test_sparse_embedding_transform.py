# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors

"""Unit tests for SparseEmbeddingTransform.

CPU-only tests using a MagicMock DeviceMesh; the
``finalize_compiled_graph`` method inlines sharded sparse-lookup
decompositions via ``make_fx(tracing_mode="symbolic")`` whose Meta
implementations don't need real distributed setup.

End-to-end coverage with real comms is provided by
``TestHybridParallelV2E2E`` and ``TestHybridParallelV2NumericalCorrectness``
in ``test_data_parallel.py``.

Run: buck2 test //recir/tests:test_sparse_embedding_transform
"""

import operator
import unittest
from unittest.mock import MagicMock

import torch
import torch.fx as fx
import torch.nn as nn
from recir.distributed.sharding_plan import EmbeddingShardingPlan, EmbeddingShardingType
from recir.region_transforms.sparse_embedding import (
    decomposed_row_wise_embedding_bag,
    decomposed_row_wise_embedding_bag_backward,
    SparseEmbeddingTransform,
)
from torch.distributed.tensor import DeviceMesh


def _mock_mesh(world_size: int = 2) -> DeviceMesh:
    mesh = MagicMock(spec=DeviceMesh)
    mesh.ndim = 1
    mesh.shape = (world_size,)
    mesh.device_type = "cpu"
    mesh.mesh = torch.arange(world_size)
    mesh._flatten.return_value = mesh
    mesh.size.return_value = world_size
    mesh._dim_group_names = ["fake_test_group"]
    return mesh


def _build_joint_graph(
    *,
    param_fqns: list[str],
    embedding_index: int,
    V: int = 100,
    D: int = 16,
    batch: int = 8,
    include_backward: bool = True,
) -> fx.GraphModule:
    """Build a synthetic joint graph: parameter placeholders followed by
    an indices placeholder, an ``aten.embedding`` op, and (optionally)
    an ``aten.embedding_dense_backward`` op consuming the same
    ``indices`` to mimic AOTAutograd's joint forward+backward.

    Param placeholders are tagged with ``node.meta["fqn"]`` to mirror
    the FQN tagging done by ``_local_compile`` in production.
    """
    graph = fx.Graph()
    param_phs: list[fx.Node] = []
    for i, fqn in enumerate(param_fqns):
        ph = graph.placeholder(f"param_{i}")
        ph.meta["fqn"] = fqn
        if i == embedding_index:
            ph.meta["val"] = torch.randn(V, D)
        else:
            ph.meta["val"] = torch.randn(D, D)
        param_phs.append(ph)
    indices = graph.placeholder("indices")
    indices.meta["val"] = torch.randint(0, V, (batch,))

    emb_node = graph.call_function(
        torch.ops.aten.embedding.default,
        args=(param_phs[embedding_index], indices),
    )
    emb_node.meta["val"] = torch.randn(batch, D)

    outputs: list[fx.Node] = [emb_node]
    if include_backward:
        grad_ph = graph.placeholder("grad_output")
        grad_ph.meta["val"] = torch.randn(batch, D)
        bwd = graph.call_function(
            torch.ops.aten.embedding_dense_backward.default,
            args=(grad_ph, indices, V, -1, False),
        )
        bwd.meta["val"] = torch.randn(V, D)
        outputs.append(bwd)
    graph.output(tuple(outputs))
    return fx.GraphModule(nn.Module(), graph)


def _add_embedding_bag(
    graph: fx.Graph,
    weight: fx.Node,
    indices: fx.Node,
    offsets: fx.Node,
    grad_output: fx.Node,
    offset2bag_index: int = 1,
) -> tuple[fx.Node, fx.Node]:
    pooled = graph.call_function(
        torch.ops.aten._embedding_bag.default,
        args=(weight, indices, offsets, False, 0, False, None, True, -1),
    )
    output = graph.call_function(operator.getitem, args=(pooled, 0))
    output.meta["val"] = torch.randn(2, 4)
    offset2bag = graph.call_function(
        operator.getitem,
        args=(pooled, offset2bag_index),
    )
    backward = graph.call_function(
        torch.ops.aten._embedding_bag_backward.default,
        args=(grad_output, indices, offsets, offset2bag),
    )
    backward.meta["val"] = torch.randn(8, 4)
    return pooled, backward


def _crossed_embedding_bag_graph() -> fx.GraphModule:
    graph = fx.Graph()
    weight = graph.placeholder("weight")
    weight.meta["fqn"] = "emb.weight"
    weight.meta["val"] = torch.randn(8, 4)
    indices_0 = graph.placeholder("indices_0")
    indices_0.meta["val"] = torch.tensor([1, 2], dtype=torch.int64)
    indices_1 = graph.placeholder("indices_1")
    indices_1.meta["val"] = torch.tensor([3, 4], dtype=torch.int64)
    offsets_0 = graph.placeholder("offsets_0")
    offsets_0.meta["val"] = torch.tensor([0, 1, 2], dtype=torch.int64)
    offsets_1 = graph.placeholder("offsets_1")
    offsets_1.meta["val"] = torch.tensor([0, 1, 2], dtype=torch.int64)
    grad_0 = graph.placeholder("grad_0")
    grad_0.meta["val"] = torch.randn(2, 4)
    grad_1 = graph.placeholder("grad_1")
    grad_1.meta["val"] = torch.randn(2, 4)
    pooled_0, backward_0 = _add_embedding_bag(
        graph, weight, indices_0, offsets_0, grad_0
    )
    pooled_1, backward_1 = _add_embedding_bag(
        graph, weight, indices_1, offsets_1, grad_1
    )
    graph.erase_node(backward_0)
    graph.erase_node(backward_1)
    offset2bag_0 = graph.call_function(operator.getitem, args=(pooled_0, 1))
    offset2bag_1 = graph.call_function(operator.getitem, args=(pooled_1, 1))
    backward_0 = graph.call_function(
        torch.ops.aten._embedding_bag_backward.default,
        args=(grad_0, indices_0, offsets_0, offset2bag_1),
    )
    backward_0.meta["val"] = torch.randn(8, 4)
    backward_1 = graph.call_function(
        torch.ops.aten._embedding_bag_backward.default,
        args=(grad_1, indices_1, offsets_1, offset2bag_0),
    )
    backward_1.meta["val"] = torch.randn(8, 4)
    graph.output((backward_0, backward_1))
    return fx.GraphModule(nn.Module(), graph)


class DecomposedRowWiseEmbeddingBagTest(unittest.TestCase):
    def test_rejects_offsets_that_do_not_consume_all_indices(self) -> None:
        weight = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        indices = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
        offsets = torch.tensor([0, 1, 2], dtype=torch.int64)
        grad_output = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])

        for mode in (0, 1):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(RuntimeError, "consume every index"):
                    decomposed_row_wise_embedding_bag(
                        weight,
                        indices,
                        offsets,
                        num_embeddings=weight.shape[0],
                        group_name="unused",
                        world_size=1,
                        mode=mode,
                    )
                with self.assertRaisesRegex(RuntimeError, "consume every index"):
                    decomposed_row_wise_embedding_bag_backward(
                        grad_output,
                        indices,
                        offsets,
                        num_embeddings=weight.shape[0],
                        local_num_embeddings=weight.shape[0],
                        group_name="unused",
                        world_size=1,
                        mode=mode,
                    )

    def test_rejects_noncanonical_offsets(self) -> None:
        weight = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        indices = torch.tensor([1, 2, 3, 4], dtype=torch.int64)

        for offsets, message in (
            (torch.tensor([1, 2, 4], dtype=torch.int64), "start at zero"),
            (torch.tensor([0, 3, 2, 4], dtype=torch.int64), "nondecreasing"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(RuntimeError, message):
                    decomposed_row_wise_embedding_bag(
                        weight,
                        indices,
                        offsets,
                        num_embeddings=weight.shape[0],
                        group_name="unused",
                        world_size=1,
                        mode=0,
                    )
                with self.assertRaisesRegex(RuntimeError, message):
                    decomposed_row_wise_embedding_bag_backward(
                        torch.randn(offsets.shape[0] - 1, weight.shape[1]),
                        indices,
                        offsets,
                        num_embeddings=weight.shape[0],
                        local_num_embeddings=weight.shape[0],
                        group_name="unused",
                        world_size=1,
                        mode=0,
                    )

    def test_backward_requires_one_gradient_row_per_bag(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "one row per bag"):
            decomposed_row_wise_embedding_bag_backward(
                torch.randn(1, 4),
                torch.tensor([1, 2], dtype=torch.int64),
                torch.tensor([0, 1, 2], dtype=torch.int64),
                num_embeddings=8,
                local_num_embeddings=8,
                group_name="unused",
                world_size=1,
                mode=0,
            )


class SparseEmbeddingBagGraphContractTest(unittest.TestCase):
    def _graph_inputs(
        self,
    ) -> tuple[fx.Graph, fx.Node, fx.Node, fx.Node, fx.Node]:
        graph = fx.Graph()
        weight = graph.placeholder("weight")
        weight.meta["fqn"] = "emb.weight"
        weight.meta["val"] = torch.randn(8, 4)
        indices = graph.placeholder("indices")
        indices.meta["val"] = torch.tensor([1, 2], dtype=torch.int64)
        offsets = graph.placeholder("offsets")
        offsets.meta["val"] = torch.tensor([0, 1, 2], dtype=torch.int64)
        grad_output = graph.placeholder("grad_output")
        grad_output.meta["val"] = torch.randn(2, 4)
        return graph, weight, indices, offsets, grad_output

    def _transform(self, *, table_wise: bool = False) -> SparseEmbeddingTransform:
        embedding_plan = None
        module = None
        if table_wise:
            embedding_plan = EmbeddingShardingPlan(
                sharding_map={"emb.weight": EmbeddingShardingType.TABLE_WISE},
                table_wise_rank_map={"emb.weight": 0},
            )
            module = nn.Module()
            module.add_module("emb", nn.Embedding(8, 4))
        return SparseEmbeddingTransform(
            embedding_param_fqns={"emb.weight"},
            mesh=_mock_mesh(world_size=1),
            mod=module,
            embedding_plan=embedding_plan,
        )

    def test_table_wise_requires_pre_shard_module(self) -> None:
        plan = EmbeddingShardingPlan(
            sharding_map={"emb.weight": EmbeddingShardingType.TABLE_WISE},
            table_wise_rank_map={"emb.weight": 0},
        )

        with self.assertRaisesRegex(ValueError, "pre-shard module"):
            SparseEmbeddingTransform(
                embedding_param_fqns={"emb.weight"},
                mesh=_mock_mesh(world_size=1),
                embedding_plan=plan,
            )

    def test_table_wise_rejects_unpooled_embedding(self) -> None:
        gm = _build_joint_graph(
            param_fqns=["emb.weight"],
            embedding_index=0,
        )

        with self.assertRaisesRegex(NotImplementedError, "pooled EBC"):
            self._transform(table_wise=True).finalize_compiled_graph(gm)

    def test_plan_must_classify_each_embedding_weight(self) -> None:
        gm = _build_joint_graph(
            param_fqns=["emb.weight"],
            embedding_index=0,
        )

        with self.assertRaisesRegex(ValueError, "does not classify"):
            SparseEmbeddingTransform(
                embedding_param_fqns={"emb.weight"},
                mesh=_mock_mesh(world_size=1),
                embedding_plan=EmbeddingShardingPlan(),
            ).finalize_compiled_graph(gm)

    def test_row_wise_does_not_query_table_wise_mesh_coordinate(self) -> None:
        gm = _build_joint_graph(
            param_fqns=["emb.weight"],
            embedding_index=0,
        )
        mesh = _mock_mesh(world_size=1)
        mesh.get_coordinate.side_effect = AssertionError(
            "row-wise lowering queried table-wise placement"
        )
        plan = EmbeddingShardingPlan(
            sharding_map={
                "emb.weight": EmbeddingShardingType.ROW_WISE,
                "other.weight": EmbeddingShardingType.TABLE_WISE,
            },
            table_wise_rank_map={"other.weight": 0},
        )

        SparseEmbeddingTransform(
            embedding_param_fqns={"emb.weight"},
            mesh=mesh,
            embedding_plan=plan,
        ).finalize_compiled_graph(gm)

    def test_rejects_unsupported_embedding_bag_options(self) -> None:
        cases: tuple[tuple[int, object, str], ...] = (
            (3, True, "frequency scaling"),
            (4, 2, "mode=2"),
            (5, True, "sparse gradients"),
            (6, torch.tensor([1.0, 1.0]), "weights"),
            (7, False, "final offset"),
            (8, 0, "padding_idx"),
        )
        for position, value, message in cases:
            with self.subTest(message=message):
                graph, weight, indices, offsets, _grad_output = self._graph_inputs()
                args: list[object] = [
                    weight,
                    indices,
                    offsets,
                    False,
                    0,
                    False,
                    None,
                    True,
                    -1,
                ]
                args[position] = value
                pooled = graph.call_function(
                    torch.ops.aten._embedding_bag.default,
                    args=tuple(args),
                )
                output = graph.call_function(operator.getitem, args=(pooled, 0))
                output.meta["val"] = torch.randn(2, 4)
                graph.output(output)
                gm = fx.GraphModule(nn.Module(), graph)

                with self.assertRaisesRegex(NotImplementedError, message):
                    self._transform().finalize_compiled_graph(gm)

    def test_rejects_live_auxiliary_outputs(self) -> None:
        for table_wise in (False, True):
            with self.subTest(table_wise=table_wise):
                graph, weight, indices, offsets, grad_output = self._graph_inputs()
                pooled, backward = _add_embedding_bag(
                    graph, weight, indices, offsets, grad_output
                )
                live_auxiliary = graph.call_function(operator.getitem, args=(pooled, 2))
                graph.output((backward, live_auxiliary))
                gm = fx.GraphModule(nn.Module(), graph)

                with self.assertRaisesRegex(RuntimeError, "live auxiliary"):
                    self._transform(table_wise=table_wise).finalize_compiled_graph(gm)

    def test_rejects_non_offset2bag_backward_input(self) -> None:
        graph, weight, indices, offsets, grad_output = self._graph_inputs()
        _pooled, backward = _add_embedding_bag(
            graph,
            weight,
            indices,
            offsets,
            grad_output,
            offset2bag_index=2,
        )
        graph.output(backward)
        gm = fx.GraphModule(nn.Module(), graph)

        with self.assertRaisesRegex(RuntimeError, "offset2bag output"):
            self._transform().finalize_compiled_graph(gm)

    def test_requires_each_forward_to_match_its_own_backward(self) -> None:
        for table_wise in (False, True):
            with self.subTest(table_wise=table_wise):
                gm = _crossed_embedding_bag_graph()
                with self.assertRaisesRegex(RuntimeError, "same indices and offsets"):
                    self._transform(table_wise=table_wise).finalize_compiled_graph(gm)

    def test_rejects_multiple_backwards_for_one_forward(self) -> None:
        for table_wise in (False, True):
            with self.subTest(table_wise=table_wise):
                graph, weight, indices, offsets, grad_output = self._graph_inputs()
                pooled, backward = _add_embedding_bag(
                    graph, weight, indices, offsets, grad_output
                )
                offset2bag = graph.call_function(operator.getitem, args=(pooled, 1))
                duplicate_backward = graph.call_function(
                    torch.ops.aten._embedding_bag_backward.default,
                    args=(grad_output, indices, offsets, offset2bag),
                )
                duplicate_backward.meta["val"] = torch.randn(8, 4)
                graph.output((backward, duplicate_backward))
                gm = fx.GraphModule(nn.Module(), graph)

                with self.assertRaisesRegex(RuntimeError, "exactly one backward"):
                    self._transform(table_wise=table_wise).finalize_compiled_graph(gm)


class TestSparseEmbeddingTransformFqnLookup(unittest.TestCase):
    """Lookup of embedding placeholders is based on ``node.meta['fqn']``
    rather than positional indices, so it is robust to leading
    placeholders being added/removed by ``Override.transform`` passes
    inside ``_finalize_graph``."""

    def test_fqn_lookup_finds_embedding_placeholder(self) -> None:
        gm = _build_joint_graph(
            param_fqns=["linear.weight", "emb.weight"],
            embedding_index=1,
            include_backward=True,
        )
        transform = SparseEmbeddingTransform(
            embedding_param_fqns={"emb.weight"},
            mesh=_mock_mesh(world_size=2),
        )
        transform.finalize_compiled_graph(gm)

        # AllToAll nodes inserted (3 per rewritten op × 2 ops = 6).
        a2a_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target == torch.ops._c10d_functional.all_to_all_single.default
        ]
        self.assertEqual(len(a2a_nodes), 6)

    def test_lookup_unaffected_by_leading_placeholder_shift(self) -> None:
        """If an Override.transform prepends an extra placeholder before
        the embedding, FQN lookup still finds the embedding placeholder
        — unlike the previous positional-index design."""
        gm = _build_joint_graph(
            param_fqns=["linear.weight", "emb.weight"],
            embedding_index=1,
            include_backward=True,
        )
        # Simulate Override.transform prepending a new placeholder with
        # no FQN meta in front of the originals.
        with gm.graph.inserting_before(next(iter(gm.graph.nodes))):
            injected = gm.graph.placeholder("injected_state")
            injected.meta["val"] = torch.randn(4)

        transform = SparseEmbeddingTransform(
            embedding_param_fqns={"emb.weight"},
            mesh=_mock_mesh(world_size=2),
        )
        transform.finalize_compiled_graph(gm)

        # AllToAll inserted (3 per rewritten op × 2 = 6: 1 forward + 1 backward).
        a2a_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target == torch.ops._c10d_functional.all_to_all_single.default
        ]
        self.assertEqual(len(a2a_nodes), 6)


class TestSparseEmbeddingTransformExpandIntoFinegrained(unittest.TestCase):
    """Integration tests for the post-compile expansion that replaces
    ``aten.embedding`` and ``aten.embedding_dense_backward`` on sharded
    weights with the decomposed AllToAll sub-graphs."""

    def test_forward_and_backward_both_replaced(self) -> None:
        gm = _build_joint_graph(
            param_fqns=["dense.weight", "emb.weight"],
            embedding_index=1,
            include_backward=True,
        )
        transform = SparseEmbeddingTransform(
            embedding_param_fqns={"emb.weight"},
            mesh=_mock_mesh(world_size=2),
        )
        transform.finalize_compiled_graph(gm)

        # Any aten.embedding remaining in the graph must be the inlined
        # local lookup (step 3 of the decomposition).  Its weight arg is
        # the original sharded placeholder, but its indices arg must
        # come from a wait_tensor (the AllToAll-of-indices result), NOT
        # from the original indices placeholder directly.
        for node in gm.graph.nodes:
            if (
                node.op == "call_function"
                and node.target == torch.ops.aten.embedding.default
            ):
                indices_arg = node.args[1]
                self.assertIsInstance(indices_arg, fx.Node)
                self.assertEqual(
                    indices_arg.target,
                    torch.ops._c10d_functional.wait_tensor.default,
                )

        # Any aten.embedding_dense_backward remaining must be the inlined
        # local backward (step 5 of the backward decomposition); its
        # indices arg must come from a wait_tensor, NOT the original
        # indices placeholder.
        for node in gm.graph.nodes:
            if (
                node.op == "call_function"
                and node.target == torch.ops.aten.embedding_dense_backward.default
            ):
                indices_arg = node.args[1]
                self.assertIsInstance(indices_arg, fx.Node)
                self.assertEqual(
                    indices_arg.target,
                    torch.ops._c10d_functional.wait_tensor.default,
                )

        # AllToAll nodes are now present (3 per rewritten op × 2 ops = 6).
        a2a_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target == torch.ops._c10d_functional.all_to_all_single.default
        ]
        self.assertEqual(len(a2a_nodes), 6)

    def test_no_embedding_fqns_is_noop(self) -> None:
        gm = _build_joint_graph(
            param_fqns=["emb.weight"],
            embedding_index=0,
        )
        transform = SparseEmbeddingTransform(
            embedding_param_fqns=set(),
            mesh=_mock_mesh(world_size=2),
        )
        transform.finalize_compiled_graph(gm)

        # Original aten.embedding still present.
        embedding_targets = [
            n.target
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target == torch.ops.aten.embedding.default
        ]
        self.assertEqual(len(embedding_targets), 1)
        # No AllToAll inserted.
        a2a_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target == torch.ops._c10d_functional.all_to_all_single.default
        ]
        self.assertEqual(a2a_nodes, [])

    def test_unmatched_fqns_is_noop(self) -> None:
        gm = _build_joint_graph(
            param_fqns=["dense.weight"],
            embedding_index=0,
        )
        transform = SparseEmbeddingTransform(
            embedding_param_fqns={"nonexistent.weight"},
            mesh=_mock_mesh(world_size=2),
        )
        transform.finalize_compiled_graph(gm)

        a2a_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target == torch.ops._c10d_functional.all_to_all_single.default
        ]
        self.assertEqual(a2a_nodes, [])

    def test_dense_embedding_left_alone(self) -> None:
        """An ``aten.embedding`` whose weight is NOT in the sharding plan
        is left untouched (forward and backward both preserved)."""
        gm = _build_joint_graph(
            param_fqns=["dense.weight"],
            embedding_index=0,
            include_backward=True,
        )
        transform = SparseEmbeddingTransform(
            # The single param is "dense.weight" — not a sharded embedding.
            embedding_param_fqns={"emb.weight"},
            mesh=_mock_mesh(world_size=2),
        )
        transform.finalize_compiled_graph(gm)

        embedding_targets = [
            n.target
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target == torch.ops.aten.embedding.default
        ]
        self.assertEqual(len(embedding_targets), 1)
        bwd_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target == torch.ops.aten.embedding_dense_backward.default
        ]
        self.assertEqual(len(bwd_nodes), 1)


if __name__ == "__main__":
    unittest.main()
