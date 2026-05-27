# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors

"""Unit tests for SparseEmbeddingTransform.

CPU-only tests using a MagicMock DeviceMesh; the
``finalize_compiled_graph`` method inlines the row-wise AllToAll
decomposition via ``make_fx(tracing_mode="symbolic")`` whose Meta
implementations don't need real distributed setup.

End-to-end coverage with real comms is provided by
``TestHybridParallelV2E2E`` and ``TestHybridParallelV2NumericalCorrectness``
in ``test_data_parallel.py``.

Run: buck2 test //recir/tests:test_sparse_embedding_transform
"""

import unittest
from unittest.mock import MagicMock

import torch
import torch.fx as fx
import torch.nn as nn
from recir.region_transforms.sparse_embedding import SparseEmbeddingTransform
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
