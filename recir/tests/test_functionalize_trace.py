# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from unittest import TestCase

import recir.torchax
import torch
from parameterized import parameterized
from recir.compiler.tracing import _functionalize
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx


def _slice_assign(x: torch.Tensor) -> torch.Tensor:
    y = x * 1.0
    y[:2] = 0.0
    return y


def _fill_diagonal(x: torch.Tensor) -> torch.Tensor:
    m = (x.unsqueeze(1) == x.unsqueeze(0)).clone()
    m.fill_diagonal_(False)
    return m


def _detach_alias(x: torch.Tensor) -> torch.Tensor:
    loss = (x * 2.0).sum()
    logged = loss.detach()
    logged /= 4.0
    return loss


def _whole_tensor(x: torch.Tensor) -> torch.Tensor:
    y = x * 1.0
    y.mul_(2.0)
    return y


def _masked_index_put(x: torch.Tensor) -> torch.Tensor:
    y = x * 1.0
    y[x > 1.0] = 0.0
    return y


def _input_mutation(x: torch.Tensor) -> torch.Tensor:
    x.add_(1.0)
    return x * 2.0


_FNS: list[tuple[str, Callable[[torch.Tensor], torch.Tensor]]] = [
    ("slice_assign", _slice_assign),
    ("fill_diagonal", _fill_diagonal),
    ("detach_alias", _detach_alias),
    ("whole_tensor", _whole_tensor),
    ("masked_index_put", _masked_index_put),
    ("input_mutation", _input_mutation),
]


def _trace(
    fn: Callable[[torch.Tensor], torch.Tensor], functionalize: bool
) -> torch.fx.GraphModule:
    mode = FakeTensorMode()
    x = mode.from_tensor(torch.arange(4.0))
    with mode:
        gm = make_fx(fn, tracing_mode="fake")(x)
        return _functionalize(gm, x) if functionalize else gm


def _run_torchax(gm: torch.fx.GraphModule, x: torch.Tensor) -> torch.Tensor:
    env = recir.torchax.default_env()
    with env:
        return gm(env.to_xla(x)).torch()


class FunctionalizeTraceTest(TestCase):
    @parameterized.expand(_FNS)
    def test_torchax_matches_eager(
        self, _name: str, fn: Callable[[torch.Tensor], torch.Tensor]
    ) -> None:
        expected = fn(torch.arange(4.0))
        actual = _run_torchax(_trace(fn, functionalize=True), torch.arange(4.0))
        torch.testing.assert_close(actual, expected)

    def test_unfunctionalized_view_write_is_lost(self) -> None:
        expected = _slice_assign(torch.arange(4.0))
        actual = _run_torchax(
            _trace(_slice_assign, functionalize=False), torch.arange(4.0)
        )
        self.assertFalse(torch.equal(actual, expected))

    def test_node_meta_carries_over(self) -> None:
        mode = FakeTensorMode()
        x = mode.from_tensor(torch.arange(4.0))
        with mode:
            gm = make_fx(_slice_assign, tracing_mode="fake")(x)
            for node in gm.graph.nodes:
                node.meta["custom"] = {"from": node.name}
            gm = _functionalize(gm, x)
        custom = {
            str(n.target): n.meta.get("custom")
            for n in gm.graph.nodes
            if n.op == "call_function"
        }
        self.assertEqual(custom["aten.mul.Tensor"], {"from": "mul"})
        self.assertEqual(custom["aten.slice_scatter.default"], {"from": "fill_"})
