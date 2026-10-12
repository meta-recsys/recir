# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from unittest.mock import patch

import torch
from functorch import make_fx
from recir.embedding.tbe.local import LocalTBEGroup, LocalTBETable


class LocalTBETraceTest(unittest.TestCase):
    def _group(self, pooling_mode: str = "none") -> LocalTBEGroup:
        with patch(
            "fbgemm_gpu.config.feature_list.FeatureGate.is_enabled",
            return_value=False,
        ):
            return LocalTBEGroup(
                tables=(LocalTBETable("table", 8, 4),),
                feature_table_map=(0,),
                device=torch.device("cpu"),
                weight_dtype=torch.float32,
                learning_rate=0.1,
                eps=1.0e-8,
                stochastic_rounding=False,
                pooling_mode=pooling_mode,
            )

    def test_fake_trace_contains_tbe_forward_boundary(self) -> None:
        group = self._group()

        def forward(indices: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
            return group.forward_flat(indices, offsets)

        traced = make_fx(
            forward,
            tracing_mode="fake",
            _allow_non_fake_inputs=True,
        )(
            torch.tensor([1, 3], dtype=torch.int64),
            torch.tensor([0, 2], dtype=torch.int64),
        )

        forwards = [
            node
            for node in traced.graph.nodes
            if node.target == torch.ops.recir.tbe_forward.default
        ]
        self.assertEqual(len(forwards), 1)
        learning_rate = forwards[0].args[4]
        self.assertIsInstance(learning_rate, torch.fx.Node)
        self.assertEqual(learning_rate.meta["val"].device.type, "cpu")
        self.assertEqual(learning_rate.meta["val"].shape, torch.Size([]))

    def test_fake_trace_contains_tbe_backward_update_boundary(self) -> None:
        group = self._group()

        def train_step(indices: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
            output = group.forward_flat(indices, offsets)
            output.sum().backward()
            return output.detach()

        traced = make_fx(
            train_step,
            tracing_mode="fake",
            _allow_non_fake_inputs=True,
        )(
            torch.tensor([1, 3], dtype=torch.int64),
            torch.tensor([0, 2], dtype=torch.int64),
        )

        updates = [
            node
            for node in traced.graph.nodes
            if node.target == torch.ops.recir.tbe_backward_update.default
        ]
        self.assertEqual(len(updates), 1)

    def test_pooled_offset_validation_is_fake_traceable(self) -> None:
        group = self._group(pooling_mode="sum")
        indices = (torch.tensor([1, 3], dtype=torch.int64),)
        offsets = (torch.tensor([0, 1, 2], dtype=torch.int64),)
        expected = group(indices, offsets)

        traced = make_fx(
            group,
            tracing_mode="fake",
            _allow_non_fake_inputs=True,
        )(indices, offsets)

        actual = traced(indices, offsets)
        torch.testing.assert_close(actual[0], expected[0])
        with self.assertRaisesRegex(RuntimeError, "consume every feature ID"):
            traced(
                indices,
                (torch.tensor([0, 1, 1], dtype=torch.int64),),
            )

    def test_grouped_features_trace_to_one_tbe_forward(self) -> None:
        with patch(
            "fbgemm_gpu.config.feature_list.FeatureGate.is_enabled",
            return_value=False,
        ):
            group = LocalTBEGroup(
                tables=(
                    LocalTBETable("table_0", local_rows=8, embedding_dim=4),
                    LocalTBETable("table_1", local_rows=6, embedding_dim=4),
                ),
                feature_table_map=(0, 1),
                device=torch.device("cpu"),
                weight_dtype=torch.float32,
                learning_rate=0.1,
                eps=1.0e-8,
                stochastic_rounding=False,
            )

        traced = make_fx(
            group,
            tracing_mode="fake",
            _allow_non_fake_inputs=True,
        )(
            (
                torch.tensor([1, 3], dtype=torch.int64),
                torch.tensor([0, 2, 5], dtype=torch.int64),
            )
        )

        tbe_forwards = [
            node
            for node in traced.graph.nodes
            if node.target == torch.ops.recir.tbe_forward.default
        ]
        self.assertEqual(len(tbe_forwards), 1)
