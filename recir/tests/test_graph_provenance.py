# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import torch
from recir.observability.graph_provenance import (
    align_served_nodes,
    ARTIFACT_DIR_ENV,
    INSERTED_BY_KEY,
    mark_inserted_nodes,
    node_record,
    stamp_training_provenance,
    TRAIN_NODE_KEY,
    write_graph_artifact,
)


class GraphProvenanceTest(unittest.TestCase):
    def test_marks_only_nodes_added_after_stamping(self) -> None:
        gm = torch.fx.symbolic_trace(torch.nn.Linear(2, 2))
        stamp_training_provenance(gm)
        output = next(n for n in gm.graph.nodes if n.op == "output")
        with gm.graph.inserting_before(output):
            added = gm.graph.call_function(torch.relu, args=(output.args[0],))

        marked = mark_inserted_nodes(gm, "some_pass")

        self.assertEqual(marked, 1)
        self.assertEqual(added.meta[INSERTED_BY_KEY], "some_pass")
        traced = [n for n in gm.graph.nodes if n is not added]
        self.assertTrue(all(n.meta[TRAIN_NODE_KEY] == n.name for n in traced))

    def test_records_aten_targets_by_op_name(self) -> None:
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        relu = graph.call_function(torch.ops.aten.relu.default, args=(x,))
        graph.output(relu)

        self.assertEqual(node_record(relu)["target"], "aten.relu.default")

    def test_writes_artifact_only_when_dir_is_set(self) -> None:
        gm = torch.fx.symbolic_trace(torch.nn.Linear(2, 2))
        stamp_training_provenance(gm)
        self.assertIsNone(write_graph_artifact(gm, "unset"))

        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {ARTIFACT_DIR_ENV: directory}):
                path = write_graph_artifact(gm, "linear")
            assert path is not None
            with open(path) as f:
                graphs = json.load(f)

        nodes = graphs["<root>"]
        self.assertEqual([n["op"] for n in nodes][0], "placeholder")
        self.assertEqual({n[TRAIN_NODE_KEY] for n in nodes}, {n["name"] for n in nodes})


class AlignServedNodesTest(unittest.TestCase):
    def test_maps_renamed_served_nodes_and_skips_wrapper_nodes(self) -> None:
        publish = [
            {
                "name": "x",
                "op": "placeholder",
                "target": "x",
                "recir_train_node": "arg3_5",
            },
            {
                "name": "mm",
                "op": "call_function",
                "target": "aten.mm.default",
                "recir_train_node": "mm_7",
            },
            {
                "name": "to_1",
                "op": "call_method",
                "target": "to",
                "recir_inserted_by": "move_graph_to_device",
            },
            {
                "name": "relu",
                "op": "call_function",
                "target": "aten.relu.default",
                "recir_train_node": "relu_2",
            },
        ]
        served = [
            {"name": "adapter", "op": "call_module", "target": "local_adapter"},
            {"name": "x", "op": "placeholder", "target": "x"},
            {"name": "mm_3", "op": "call_function", "target": "torch.ops.aten.mm"},
            {"name": "to_9", "op": "call_method", "target": "to"},
            {"name": "relu_4", "op": "call_function", "target": "torch.ops.aten.relu"},
        ]

        mapping = align_served_nodes(publish, served)

        self.assertNotIn("adapter", mapping)
        self.assertEqual(mapping["mm_3"]["recir_train_node"], "mm_7")
        self.assertEqual(mapping["to_9"]["recir_inserted_by"], "move_graph_to_device")
        self.assertEqual(mapping["relu_4"]["recir_train_node"], "relu_2")

    def test_maps_reordered_weight_reads_by_parameter_path(self) -> None:
        publish = [
            {"name": "w", "op": "get_attr", "target": "model.mlp.weight"},
            {"name": "mm", "op": "call_function", "target": "aten.mm.default"},
        ]
        served = [
            {"name": "mm_1", "op": "call_function", "target": "torch.ops.aten.mm"},
            {
                "name": "model_model_mlp_weight",
                "op": "get_attr",
                "target": "model.model.mlp.weight",
            },
        ]

        mapping = align_served_nodes(publish, served)

        self.assertEqual(mapping["model_model_mlp_weight"]["name"], "w")
