# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import recir.torchax
import torch
from recir.torchax import interop
from torch import nn

from . import base_test_util


class SharedWeightsModel(nn.Module):
    def __init__(self, vocab_size: int = 10, dim: int = 4) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.ReLU(), nn.Linear(dim * 2, dim)
        )
        self.head = nn.Linear(dim, vocab_size, bias=False)

        # Tie the weights: head.weight should be the same parameter as emb.weight
        self.head.weight = self.emb.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.emb(x)
        x = self.mlp(x)
        x = self.head(x)
        return x


class SharedWeightsTest(base_test_util.TestCase):  # pyre-ignore[11]
    def test_shared_weights(self) -> None:
        env = recir.torchax.default_env()
        model = SharedWeightsModel()
        self.assertTrue(model.head.weight is model.emb.weight)  # pyre-ignore[16]

        x = torch.randint(0, 10, (2, 5))
        res_torch = model(x)

        jax_weights, jax_func = recir.torchax.extract_jax(model, dedup_parameters=True)

        # Verify that the JAX weights are shared
        self.assertIn("emb.weight", jax_weights)  # pyre-ignore[16]
        self.assertNotIn("head.weight", jax_weights)  # pyre-ignore[16]

        arg = env.t2j_copy(x)
        res_jax = jax_func(jax_weights, (arg,))
        res_jax_torch = env.j2t_copy(res_jax)

        self.assertTrue(torch.allclose(res_torch, res_jax_torch, atol=1e-5))


if __name__ == "__main__":
    base_test_util.main()
