# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import torch
from recir.training.optim import (
    prime_optimizer_state,
    TensorLRRowWiseAdagrad,
    WarmupController,
)
from torchrec.optim.warmup import WarmupPolicy, WarmupStage


class WarmupControllerTest(unittest.TestCase):
    def test_sparse_warmup_updates_single_buffer(self) -> None:
        dense_lr = torch.tensor(0.0, dtype=torch.float32)
        sparse_lr = torch.tensor(0.0, dtype=torch.float32)
        controller = WarmupController(
            dense_base_lr=0.01,
            dense_lr_warmup_schedule=None,
            dense_lr_buffer=dense_lr,
            iter_buffer=torch.zeros((), dtype=torch.int64),
            sparse_base_lr=0.2,
            sparse_lr_warmup_schedule=[
                WarmupStage(
                    policy=WarmupPolicy.LINEAR,
                    max_iters=2,
                    value=0.1,
                )
            ],
            sparse_lr_buffer=sparse_lr,
        )

        for expected in (0.02, 0.11, 0.2):
            controller.apply()
            torch.testing.assert_close(sparse_lr, torch.tensor(expected))
            controller.step()

        restored_sparse_lr = torch.tensor(0.0, dtype=torch.float32)
        restored = WarmupController(
            dense_base_lr=0.01,
            dense_lr_warmup_schedule=None,
            dense_lr_buffer=torch.tensor(0.0, dtype=torch.float32),
            iter_buffer=torch.zeros((), dtype=torch.int64),
            sparse_base_lr=0.2,
            sparse_lr_warmup_schedule=[
                WarmupStage(
                    policy=WarmupPolicy.LINEAR,
                    max_iters=2,
                    value=0.1,
                )
            ],
            sparse_lr_buffer=restored_sparse_lr,
        )
        restored.load_state_dict(controller.state_dict())

        torch.testing.assert_close(restored_sparse_lr, sparse_lr)
        self.assertEqual(restored_sparse_lr.device.type, "cpu")

    def test_prime_rowwise_state_uses_parameter_device(self) -> None:
        parameter = torch.nn.Parameter(torch.ones((2, 4), device="meta"))
        optimizer = TensorLRRowWiseAdagrad(
            [parameter],
            lr_tensor=torch.tensor(0.1, device="cpu"),
            eps=1.0e-8,
            weight_decay=0.0,
        )

        prime_optimizer_state(optimizer)

        self.assertEqual(optimizer.state[parameter]["sum"].device, parameter.device)
