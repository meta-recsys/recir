# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


import types
import unittest
from typing import cast

import torch
from recir.training.optim import (
    build_torch_optimizer,
    OptimConfig,
    prime_optimizer_state,
    TensorLRRowWiseAdagrad,
    WarmupController,
)
from torchrec.distributed.embedding_types import OptimType
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


def _rowwise_adagrad(
    weight: torch.Tensor, grad_scale: float = 1.0
) -> tuple[TensorLRRowWiseAdagrad, torch.nn.Parameter]:
    param = torch.nn.Parameter(weight.clone())
    optimizer = TensorLRRowWiseAdagrad(
        [param],
        lr_tensor=torch.tensor(0.02),
        eps=1.0e-3,
        weight_decay=0.0,
        grad_scale=grad_scale,
    )
    return optimizer, param


class RowWiseAdagradGradScaleTest(unittest.TestCase):
    def test_scaled_step_equals_a_step_on_the_scaled_gradient(self) -> None:
        weight = torch.randn(3, 4)
        grad = torch.randn(3, 4)
        scaled, scaled_param = _rowwise_adagrad(weight, grad_scale=32.0)
        plain, plain_param = _rowwise_adagrad(weight)

        scaled_param.grad = grad.clone()
        plain_param.grad = grad * 32.0
        scaled.step()
        plain.step()

        torch.testing.assert_close(scaled_param, plain_param)
        torch.testing.assert_close(
            scaled.state[scaled_param]["sum"], plain.state[plain_param]["sum"]
        )

    def test_state_saved_without_grad_scale_keeps_the_constructor_value(self) -> None:
        saved, _ = _rowwise_adagrad(torch.ones(2, 4))
        state = saved.state_dict()
        del state["param_groups"][0]["grad_scale"]
        restored, _ = _rowwise_adagrad(torch.ones(2, 4), grad_scale=32.0)

        restored.load_state_dict(state)

        self.assertEqual(32.0, restored.param_groups[0]["grad_scale"])

    def test_only_rowwise_adagrad_takes_a_grad_scale(self) -> None:
        cfg = cast(OptimConfig, types.SimpleNamespace(optim_type=OptimType.ADAMW))

        with self.assertRaisesRegex(ValueError, "only implemented for ROWWISE_ADAGRAD"):
            build_torch_optimizer([], cfg, torch.tensor(0.1), grad_scale=32.0)
