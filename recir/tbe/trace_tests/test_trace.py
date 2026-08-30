# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from unittest.mock import patch

import torch
from functorch import make_fx
from recir.tbe.local import LocalTBEGroup, LocalTBETable


class LocalTBETraceTest(unittest.TestCase):
    def _group(self) -> LocalTBEGroup:
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
            )

    def test_fake_trace_contains_native_tbe_forward(self) -> None:
        group = self._group()
        traced = make_fx(
            group.forward_flat,
            tracing_mode="fake",
            _allow_non_fake_inputs=True,
        )(
            torch.tensor([1, 3], dtype=torch.int64),
            torch.tensor([0, 2], dtype=torch.int64),
        )

        targets = {str(node.target) for node in traced.graph.nodes}
        self.assertTrue(
            any("split_embedding_nobag_codegen_forward" in target for target in targets)
        )

    def test_fake_trace_contains_fused_backward_update(self) -> None:
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

        targets = {str(node.target) for node in traced.graph.nodes}
        self.assertTrue(any("rowwise_adagrad" in target for target in targets))

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
            if "split_embedding_nobag_codegen_forward" in str(node.target)
        ]
        self.assertEqual(len(tbe_forwards), 1)
