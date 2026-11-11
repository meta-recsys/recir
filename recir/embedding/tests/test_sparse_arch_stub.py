# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from recir.embedding._sparse_arch_stub import DummySparseArch
from torchrec.sparse.jagged_tensor import JaggedTensor


class DummySparseArchTest(unittest.TestCase):
    def _cached(self) -> dict[str, JaggedTensor]:
        return {
            "f": JaggedTensor(values=torch.randn(3, 2), lengths=torch.tensor([1, 2]))
        }

    def test_stays_on_the_host_by_default(self) -> None:
        dummy = DummySparseArch(self._cached(), input_kjt_keys=["f"])
        devices = {t.device.type for t in [*dummy.parameters(), *dummy.buffers()]}
        self.assertEqual(devices, {"cpu"})

    def test_lives_on_the_trace_device(self) -> None:
        dummy = DummySparseArch(
            self._cached(), input_kjt_keys=["f"], trace_device=torch.device("meta")
        )
        devices = {t.device.type for t in [*dummy.parameters(), *dummy.buffers()]}
        self.assertEqual(devices, {"meta"})
