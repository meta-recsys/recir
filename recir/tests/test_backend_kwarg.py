# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Tests for the ``backend=`` and ``eager_preproc=`` kwargs on recir.compile.

Verifies:
- ``compile(backend=...)`` drives the pipeline via ``trace`` + ``compile_from_gm``.
- ``eager_preproc`` hooks are invoked on the model argument before tracing.
"""

import unittest
from typing import List

import recir
import torch
import torch.nn as nn
from recir.tests.test_utils import _NoopBackend, _PassthroughParallel
from torch import fx


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).sum()


def _toy_train_step(
    model: nn.Module, opt: torch.optim.Optimizer, x: torch.Tensor
) -> None:
    opt.zero_grad()
    loss = model(x)
    loss.backward()
    opt.step()


class NewBackendPathTest(unittest.TestCase):
    """compile(backend=...) routes through trace + compile_from_gm."""

    def test_backend_path_invokes_dispatch(self) -> None:
        """backend.dispatch is invoked exactly once during compilation."""
        dispatch_calls: List[fx.GraphModule] = []

        class _RecordBackend(_NoopBackend):
            def dispatch(self, gm: fx.GraphModule) -> nn.Module:
                dispatch_calls.append(gm)
                return gm

        @recir.compile(
            parallel_mode=_PassthroughParallel(),
            backend=_RecordBackend(),
        )
        def train_step(m: nn.Module, o: torch.optim.Optimizer, x: torch.Tensor) -> None:
            _toy_train_step(m, o, x)

        model = _ToyModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=True)
        x = torch.randn(2, 4)
        train_step(model, opt, x)
        self.assertEqual(len(dispatch_calls), 1, "dispatch should be called once")

    def test_eager_preproc_runs_before_trace(self) -> None:
        """Each eager_preproc hook fires on the model before tracing."""
        hook_calls: List[nn.Module] = []

        def _record(m: nn.Module) -> None:
            hook_calls.append(m)

        @recir.compile(
            parallel_mode=_PassthroughParallel(),
            backend=_NoopBackend(),
            eager_preproc=[_record, _record],
        )
        def train_step(m: nn.Module, o: torch.optim.Optimizer, x: torch.Tensor) -> None:
            _toy_train_step(m, o, x)

        model = _ToyModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=True)
        x = torch.randn(2, 4)
        train_step(model, opt, x)
        self.assertEqual(len(hook_calls), 2)
        for called_model in hook_calls:
            self.assertIs(called_model, model)
