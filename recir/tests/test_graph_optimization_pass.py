# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Tests for code that survives the v7.1 module_override / gm_transformation
deletion.

These two test classes were relocated out of the deleted
``test_transformation.py`` because they cover still-live code:

* :class:`GraphPassWrapperTest` exercises the ``graph_optimization_pass``
  decorator's prerequisites / ``apply_after`` ordering logic
  (:mod:`recir.passes.pass_registry`).
* :class:`EnableIterGraphTest` exercises the
  :class:`GraphModuleTransformation.enable_iter_graph` flag.
  ``GraphModuleTransformation`` is intentionally retained (consumed by
  ``bundle_runtime/runtime.py`` and ``xla_test_mixin.py``).
"""

import unittest
from unittest.mock import MagicMock

import torch
import torch.nn as nn
from recir.passes import default_registry, graph_optimization_pass
from recir.passes.gm_transformation import GraphModuleTransformation
from recir.passes.iter_graph_module import IterGraphModule
from torch.testing._internal.distributed._tensor.common_dtensor import DTensorTestBase


class GraphPassWrapperTest(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 1

    def test_order(self) -> None:
        @graph_optimization_pass(
            prerequisites=[],
            apply_after=[],
        )
        def my_pass1(gm: IterGraphModule) -> None:
            return

        @graph_optimization_pass(
            prerequisites=[my_pass1],
            apply_after=[],
        )
        def my_pass2(gm: IterGraphModule) -> None:
            return

        @graph_optimization_pass(
            prerequisites=[],
            apply_after=[my_pass1],
        )
        def my_pass3(gm: IterGraphModule) -> None:
            return

        gm = MagicMock(spec=IterGraphModule)
        # No errors happen.
        my_pass1(gm)
        my_pass3(gm)
        my_pass2(gm)
        default_registry.reset()

        # Only my_pass3 is okay as it has no prerequisites.
        my_pass3(gm)
        default_registry.reset()

        # Prerequisite condition does not match.
        with self.assertRaisesRegex(AssertionError, "are the prerequisites of"):
            my_pass2(gm)
        default_registry.reset()

        # my_pass3 must be applied after my_pass1
        my_pass3(gm)
        with self.assertRaisesRegex(AssertionError, "must be applied after"):
            my_pass1(gm)
        default_registry.reset()


class EnableIterGraphTest(unittest.TestCase):
    """CPU-only tests for the `enable_iter_graph` flag in GraphModuleTransformation."""

    def _make_simple_gm(self) -> torch.fx.GraphModule:
        class SimpleModel(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + 1

        return torch.fx.symbolic_trace(SimpleModel())

    def test_enable_graph_optimization_requires_iter_graph(self) -> None:
        with self.assertRaisesRegex(ValueError, "enable_graph_optimization"):
            GraphModuleTransformation(
                enable_graph_optimization=True,
                enable_iter_graph=False,
            )

    def test_default_returns_iter_graph_module(self) -> None:
        gm = self._make_simple_gm()
        transform = GraphModuleTransformation()
        result = transform(gm)
        self.assertIsInstance(result, IterGraphModule)

    def test_disable_iter_graph_returns_plain_gm(self) -> None:
        gm = self._make_simple_gm()
        transform = GraphModuleTransformation(enable_iter_graph=False)
        result = transform(gm)
        self.assertNotIsInstance(result, IterGraphModule)
        self.assertIsInstance(result, torch.fx.GraphModule)

    def test_disable_iter_graph_no_backend_preserves_gm(self) -> None:
        gm = self._make_simple_gm()
        transform = GraphModuleTransformation(enable_iter_graph=False, backend=None)
        result = transform(gm)
        # Without a backend, lowering is a no-op, so the same gm is returned.
        self.assertIs(result, gm)
