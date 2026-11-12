# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors

"""Random ops must not leave a tracer in torchax's environment key.

A random op rotates ``default_env()``'s key while its graph is traced. When the
rotated key (a tracer of that trace) stayed in the environment, the next
compiled graph closed over it, JAX lifted that graph's constants to inputs, and
the compiled call failed with "Computation compiled for N inputs but called
with M".
"""

import unittest

import jax  # @manual
import recir.torchax as torchax
import torch
import torch.nn as nn
from functorch import make_fx
from recir.backends.types import BackendType
from recir.lowering.xla_lower import partial_lower_xla


class _NoisyMLP(nn.Module):
    """Enough ops to clear ``MIN_ATEN_OPS_TO_LOWER``, plus one random op."""

    def __init__(self, dim: int = 8, layers: int = 6) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        for _ in range(layers):
            modules.extend([nn.Linear(dim, dim), nn.ReLU()])
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x + torch.rand_like(x))


def _lowered(
    backend_type: BackendType,
) -> tuple[torch.fx.GraphModule, dict, torch.Tensor]:
    model = _NoisyMLP()
    x = torch.randn(2, 8)
    params = dict(model.named_parameters())

    def fn(params, x):
        return torch.func.functional_call(model, params, (x,))

    gm = make_fx(fn)(params, x)
    return partial_lower_xla(gm, backend_type=backend_type), params, x


class TorchaxPrngKeyTest(unittest.TestCase):
    def test_second_compiled_graph_runs(self) -> None:
        for backend_type in (BackendType.XLA_GSPMD, BackendType.XLA_SHARD_MAP):
            with self.subTest(backend=backend_type.value):
                torch.manual_seed(0)
                for _ in range(2):
                    gm, params, x = _lowered(backend_type)
                    self.assertEqual(tuple(gm(params, x).shape), (2, 8))
                    self.assertNotIsInstance(
                        torchax.default_env().param.prng, jax.core.Tracer
                    )


if __name__ == "__main__":
    unittest.main()
