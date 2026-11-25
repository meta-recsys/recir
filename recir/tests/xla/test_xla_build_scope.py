#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""The build scope of an XLA dispatch's subgraphs, on CPU."""

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import unittest
from contextlib import contextmanager
from typing import Any, Callable, cast, Iterator
from unittest import mock

import torch
import torch.nn as nn
from recir.backends.xla import XLAShardMapBackend
from recir.lowering.xla_lower import _XLAModule
from recir.tests.test_utils import _twelve_ops
from torch.fx.experimental.proxy_tensor import make_fx


def _traced() -> torch.fx.GraphModule:
    return make_fx(_twelve_ops)(torch.randn(2, 8))


def _dispatch() -> torch.fx.GraphModule:
    return cast(torch.fx.GraphModule, XLAShardMapBackend().dispatch(_traced()))


def _xla_modules(gm: nn.Module) -> list[_XLAModule]:
    return [m for m in gm.modules() if isinstance(m, _XLAModule)]


def _example_args(mod: _XLAModule) -> list[torch.Tensor]:
    return [
        torch.zeros(n.meta["val"].shape, dtype=n.meta["val"].dtype)
        for n in mod.gm.graph.nodes
        if n.op == "placeholder"
    ]


class FirstCallScopeTest(unittest.TestCase):
    """The build scope of an XLA subgraph covers its compile and its first call, where GSPMD and shard_map compile."""

    def test_first_call_runs_inside_the_compile_scope(self) -> None:
        (mod, *_) = _xla_modules(_dispatch())
        inside: list[bool] = [False]
        seen: list[bool] = []

        @contextmanager
        def scope() -> Iterator[None]:
            inside[0] = True
            try:
                yield
            finally:
                inside[0] = False

        compile_xla = mod._compile_xla

        def watched_compile(*args: Any) -> Callable[..., Any]:
            run = compile_xla(*args)

            def watched_run(*run_args: Any) -> Any:
                seen.append(inside[0])
                return run(*run_args)

            return watched_run

        mod.compile_scope = scope
        with mock.patch.object(mod, "_compile_xla", watched_compile):
            args = _example_args(mod)
            mod(*args)
            mod(*args)
        self.assertEqual(seen, [True, False])
