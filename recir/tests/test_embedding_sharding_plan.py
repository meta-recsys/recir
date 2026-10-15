# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from collections import OrderedDict
from typing import cast
from unittest.mock import MagicMock

import torch
from recir.distributed.data_parallel import DataParallelStyle, shard_params_by_plan
from recir.distributed.sharding_plan import EmbeddingShardingPlan, EmbeddingShardingType
from torch import nn
from torch.distributed.tensor import DeviceMesh


class _SharedParameterModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.independent = nn.Parameter(torch.ones(4, 4))
        self.primary = nn.Linear(4, 4, bias=False)
        self.alias = nn.Linear(4, 4, bias=False)
        self.alias.weight = self.primary.weight


class _ParameterAndBufferModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(4, 4))
        self.register_buffer("buffer", torch.ones(4, 4))


class EmbeddingShardingPlanTest(unittest.TestCase):
    def test_empty_plan(self) -> None:
        plan = EmbeddingShardingPlan()

        self.assertEqual(plan.param_fqns, set())
        self.assertFalse(plan.is_embedding_param("anything"))
        self.assertFalse(plan.is_row_wise("anything"))

    def test_row_wise_parameter_queries(self) -> None:
        plan = EmbeddingShardingPlan(
            sharding_map={
                "emb1.weight": EmbeddingShardingType.ROW_WISE,
                "emb2.weight": EmbeddingShardingType.ROW_WISE,
            }
        )

        self.assertEqual(plan.param_fqns, {"emb1.weight", "emb2.weight"})
        self.assertTrue(plan.is_embedding_param("emb1.weight"))
        self.assertTrue(plan.is_row_wise("emb1.weight"))
        self.assertFalse(plan.is_embedding_param("linear.weight"))
        self.assertFalse(plan.is_row_wise("linear.weight"))

    def test_table_wise_parameter_queries(self) -> None:
        plan = EmbeddingShardingPlan(
            sharding_map={"emb.weight": EmbeddingShardingType.TABLE_WISE},
            table_wise_rank_map={"emb.weight": 1},
        )

        self.assertTrue(plan.is_table_wise("emb.weight"))
        self.assertEqual(plan.table_wise_owner("emb.weight"), 1)
        self.assertFalse(plan.is_row_wise("emb.weight"))

    def test_table_wise_requires_exactly_one_owner(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one owner"):
            EmbeddingShardingPlan(
                sharding_map={"emb.weight": EmbeddingShardingType.TABLE_WISE}
            )

    def test_rejects_table_wise_owner_for_row_wise_parameter(self) -> None:
        with self.assertRaisesRegex(ValueError, "unexpected"):
            EmbeddingShardingPlan(
                sharding_map={"emb.weight": EmbeddingShardingType.ROW_WISE},
                table_wise_rank_map={"emb.weight": 0},
            )

    def test_rejects_negative_table_wise_owner(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            EmbeddingShardingPlan(
                sharding_map={
                    "weight": EmbeddingShardingType.TABLE_WISE,
                },
                table_wise_rank_map={"weight": -1},
            )

    def test_rejects_location_for_unsharded_parameter(self) -> None:
        with self.assertRaisesRegex(ValueError, "sharded parameters"):
            EmbeddingShardingPlan(
                sharding_map={
                    "emb.weight": EmbeddingShardingType.ROW_WISE,
                },
                location_map={"dense.weight": "UVM"},
            )

    def test_shared_aliases_require_the_same_resolved_location(self) -> None:
        model = _SharedParameterModel()
        params = cast(
            dict[str, torch.Tensor],
            OrderedDict(model.named_parameters(remove_duplicate=False)),
        )
        plan = EmbeddingShardingPlan(
            sharding_map={
                "primary.weight": EmbeddingShardingType.ROW_WISE,
                "alias.weight": EmbeddingShardingType.ROW_WISE,
            },
            location_map={"primary.weight": "UVM"},
        )

        with self.assertRaisesRegex(RuntimeError, "one embedding storage location"):
            shard_params_by_plan(
                model,
                params,
                None,
                MagicMock(spec=DeviceMesh),
                DataParallelStyle.REPLICATE,
                embedding_plan=plan,
            )

    def test_rejects_buffer_in_embedding_plan(self) -> None:
        model = _ParameterAndBufferModel()
        original_weight = model.weight
        original_buffer = model.buffer
        params_and_buffers = {
            **dict(model.named_parameters()),
            **dict(model.named_buffers()),
        }
        plan = EmbeddingShardingPlan(
            sharding_map={
                "weight": EmbeddingShardingType.ROW_WISE,
                "buffer": EmbeddingShardingType.ROW_WISE,
            }
        )

        with self.assertRaisesRegex(ValueError, "not model parameters"):
            shard_params_by_plan(
                model,
                params_and_buffers,
                None,
                MagicMock(spec=DeviceMesh),
                DataParallelStyle.REPLICATE,
                embedding_plan=plan,
            )
        self.assertIs(model.weight, original_weight)
        self.assertIs(model.buffer, original_buffer)
        self.assertIs(params_and_buffers["weight"], original_weight)
        self.assertIs(params_and_buffers["buffer"], original_buffer)

    def test_validates_shared_aliases_before_mutating_parameters(self) -> None:
        model = _SharedParameterModel()
        original_independent = model.independent
        params = cast(
            dict[str, torch.Tensor],
            OrderedDict(model.named_parameters(remove_duplicate=False)),
        )
        plan = EmbeddingShardingPlan(
            sharding_map={
                "independent": EmbeddingShardingType.ROW_WISE,
                "primary.weight": EmbeddingShardingType.ROW_WISE,
                "alias.weight": EmbeddingShardingType.TABLE_WISE,
            },
            table_wise_rank_map={"alias.weight": 0},
        )

        with self.assertRaisesRegex(RuntimeError, "one embedding sharding type"):
            shard_params_by_plan(
                model,
                params,
                None,
                MagicMock(spec=DeviceMesh),
                DataParallelStyle.REPLICATE,
                embedding_plan=plan,
            )
        self.assertIs(model.independent, original_independent)
        self.assertIs(params["independent"], original_independent)

    def test_rejects_parameter_absent_from_state_snapshot(self) -> None:
        model = _ParameterAndBufferModel()
        plan = EmbeddingShardingPlan(
            sharding_map={"weight": EmbeddingShardingType.ROW_WISE}
        )

        with self.assertRaisesRegex(ValueError, "absent from the state snapshot"):
            shard_params_by_plan(
                model,
                {},
                None,
                MagicMock(spec=DeviceMesh),
                DataParallelStyle.REPLICATE,
                embedding_plan=plan,
            )

    def test_rejects_table_wise_owner_outside_mesh(self) -> None:
        model = _ParameterAndBufferModel()
        mesh = MagicMock(spec=DeviceMesh)
        mesh.ndim = 1
        mesh.size.return_value = 2
        plan = EmbeddingShardingPlan(
            sharding_map={"weight": EmbeddingShardingType.TABLE_WISE},
            table_wise_rank_map={"weight": 2},
        )

        with self.assertRaisesRegex(ValueError, "outside world size 2"):
            shard_params_by_plan(
                model,
                dict(model.named_parameters()),
                None,
                mesh,
                DataParallelStyle.REPLICATE,
                embedding_plan=plan,
            )

    def test_rejects_table_wise_plan_on_multidimensional_mesh(self) -> None:
        model = _ParameterAndBufferModel()
        mesh = MagicMock(spec=DeviceMesh)
        mesh.ndim = 2
        mesh.shape = (2, 2)
        plan = EmbeddingShardingPlan(
            sharding_map={"weight": EmbeddingShardingType.TABLE_WISE},
            table_wise_rank_map={"weight": 0},
        )

        with self.assertRaisesRegex(ValueError, "one-dimensional device mesh"):
            shard_params_by_plan(
                model,
                dict(model.named_parameters()),
                None,
                mesh,
                DataParallelStyle.REPLICATE,
                embedding_plan=plan,
            )
