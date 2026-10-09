# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from recir.training.materialization import materialize_recir_parameters


class _SharedParameterModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.primary = torch.nn.Linear(2, 2, bias=False)
        self.alias = torch.nn.Linear(2, 2, bias=False)
        self.alias.weight = self.primary.weight


class MaterializeRecirParametersTest(unittest.TestCase):
    def test_preserves_shared_parameter_identity_and_values(self) -> None:
        model = _SharedParameterModel()
        expected = model.primary.weight.detach().clone()

        uninitialized = materialize_recir_parameters(
            model,
            torch.device("cpu"),
            set(),
            lambda _fqn, _parameter: None,
        )

        self.assertEqual(uninitialized, [])
        self.assertIs(model.primary.weight, model.alias.weight)
        torch.testing.assert_close(model.primary.weight, expected)
