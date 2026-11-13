# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for ``lower_inputs``."""

import dataclasses
import unittest
from typing import Any, cast
from unittest import mock

import torch
import torch.nn as nn
from functorch import make_fx
from recir.lowering.lowering_inputs import lower_inputs
from recir.passes.iter_graph_module import IterGraphModule
from recir.pipeline import build_pipeline, PipelinedGraphModule, SteadyStateGraphModule
from recir.pipeline_strategies import sdd_pipeline_config
from recir.tests.test_utils import _region_graph
from torch import fx


def _twelve_ops(x: torch.Tensor) -> torch.Tensor:
    for _ in range(6):
        x = torch.relu(x * 2)
    return x


def _graph() -> fx.GraphModule:
    return make_fx(_twelve_ops)(torch.randn(4))


def _stages(gm: PipelinedGraphModule) -> list[nn.Module]:
    return [gm.split_gm.get_submodule(cast(str, c.target)) for c in gm._stage_calls]


class TraversalTest(unittest.TestCase):
    def lowered(self, gm: Any) -> list[fx.GraphModule]:
        seen: list[fx.GraphModule] = []

        def lower(graph: fx.GraphModule) -> fx.GraphModule:
            seen.append(graph)
            return graph

        lower_inputs(gm, lower)
        return seen

    def test_whole_graph(self) -> None:
        gm = _graph()
        self.assertEqual(self.lowered(gm), [gm])

    def test_iter_graph_phases(self) -> None:
        gm = mock.MagicMock(spec=IterGraphModule)
        gm.setup_gm, gm.main_gm, gm.cleanup_gm = _graph(), _graph(), _graph()
        gm.lower_parts = lambda lower: IterGraphModule.lower_parts(gm, lower)
        phases = [gm.setup_gm, gm.main_gm, gm.cleanup_gm]
        self.assertEqual(self.lowered(gm), phases)

    def test_pipeline_stages(self) -> None:
        gm = build_pipeline(_region_graph(), sdd_pipeline_config())
        assert isinstance(gm, PipelinedGraphModule)
        self.assertEqual(self.lowered(gm), list(gm.stage_graph_modules()))

    def test_pipeline_stages_are_replaced_by_the_lowered_modules(self) -> None:
        gm = build_pipeline(_region_graph(), sdd_pipeline_config())
        assert isinstance(gm, PipelinedGraphModule)
        lowered: list[nn.Module] = []

        def lower(graph: fx.GraphModule) -> nn.Module:
            lowered.append(nn.Identity())
            return lowered[-1]

        lower_inputs(gm, lower)
        self.assertEqual(_stages(gm), lowered)

    def test_steady_state_graph(self) -> None:
        config = dataclasses.replace(
            sdd_pipeline_config(), materialize_steady_state=True
        )
        gm = build_pipeline(_region_graph(), config)
        assert isinstance(gm, SteadyStateGraphModule)
        steady = gm.steady_gm
        self.assertEqual(self.lowered(gm), [steady])

    def test_returns_what_lower_returns(self) -> None:
        lower = mock.Mock(side_effect=lambda gm: gm)
        gm = _graph()
        self.assertIs(lower_inputs(gm, lower), gm)
        lower.assert_called_once_with(gm)
