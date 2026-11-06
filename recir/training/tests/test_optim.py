# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


import copy
import types
import unittest
from typing import cast

import torch
from recir.training.optim import (
    build_torch_optimizer,
    OptimConfig,
    prime_optimizer_state,
    TensorLRRowWiseAdagrad,
    warmup_lr,
    WarmupController,
)
from torch.fx.experimental.proxy_tensor import make_fx
from torchrec.distributed.embedding_types import OptimType
from torchrec.optim.warmup import WarmupPolicy, WarmupStage


class RowWiseAdagradRoundingTest(unittest.TestCase):
    def _optimizer(
        self, parameter: torch.nn.Parameter, stochastic_rounding: bool = True
    ) -> TensorLRRowWiseAdagrad:
        return TensorLRRowWiseAdagrad(
            [parameter],
            lr_tensor=torch.tensor(2**-12),
            eps=0.0,
            weight_decay=0.0,
            stochastic_rounding=stochastic_rounding,
            rounding_seed=37,
        )

    def test_fp16_writeback_preserves_small_updates_in_expectation(self) -> None:
        parameter = torch.nn.Parameter(torch.ones((16384, 1), dtype=torch.float16))
        optimizer = self._optimizer(parameter)
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()

        self.assertEqual(parameter.dtype, torch.float16)
        self.assertTrue(bool(((parameter == 1) | (parameter == 1 - 2**-11)).all()))
        self.assertAlmostEqual(float((parameter < 1).float().mean()), 0.5, delta=0.02)
        self.assertEqual(int(optimizer.state[parameter]["rounding_counter"]), 1)

        deterministic = torch.nn.Parameter(torch.ones_like(parameter))
        deterministic.grad = torch.ones_like(parameter)
        self._optimizer(deterministic, stochastic_rounding=False).step()
        torch.testing.assert_close(deterministic, torch.ones_like(deterministic))

    def test_fp16_gradient_squares_and_update_use_fp32(self) -> None:
        parameter = torch.nn.Parameter(torch.ones((2, 2), dtype=torch.float16))
        gradient = torch.tensor([[300, -300], [2**-12, -(2**-12)]], dtype=torch.float16)
        parameter.grad = gradient
        optimizer = self._optimizer(parameter, stochastic_rounding=False)
        optimizer.step()

        expected_sum = gradient.float().square().mean(dim=1, keepdim=True)
        torch.testing.assert_close(optimizer.state[parameter]["sum"], expected_sum)
        expected = (
            torch.ones((2, 2)) - gradient.float() / expected_sum.sqrt() * 2**-12
        ).half()
        torch.testing.assert_close(parameter, expected, rtol=0, atol=0)

    def test_checkpoint_replays_rng_and_preserves_state_buffers(self) -> None:
        parameter = torch.nn.Parameter(torch.ones((128, 4), dtype=torch.float16))
        optimizer = self._optimizer(parameter)
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        optimizer.state[parameter]["rounding_counter"].fill_(65537)
        optimizer.state[parameter]["rounding_seed"].fill_(2**31 + 123)
        checkpoint = copy.deepcopy(optimizer.state_dict())
        saved_weight = parameter.detach().clone()
        optimizer.step()
        expected = parameter.detach().clone()

        restored_parameter = torch.nn.Parameter(saved_weight)
        restored = self._optimizer(restored_parameter)
        prime_optimizer_state(restored)
        buffers = dict(restored.state[restored_parameter])
        restored.load_state_dict(checkpoint)
        for key, buffer in buffers.items():
            self.assertIs(restored.state[restored_parameter][key], buffer)
        self.assertEqual(buffers["sum"].dtype, torch.float32)
        self.assertEqual(buffers["rounding_counter"].dtype, torch.int64)
        self.assertEqual(int(buffers["rounding_counter"]), 65537)
        self.assertEqual(int(buffers["rounding_seed"]), 2**31 + 123)
        restored_parameter.grad = torch.ones_like(restored_parameter)
        restored.step()
        torch.testing.assert_close(restored_parameter, expected, rtol=0, atol=0)

    def test_legacy_checkpoint_and_meta_priming(self) -> None:
        parameter = torch.nn.Parameter(torch.ones((2, 4), dtype=torch.float16))
        optimizer = self._optimizer(parameter)
        prime_optimizer_state(optimizer)
        checkpoint = optimizer.state_dict()
        checkpoint["state"][0] = {"sum": torch.ones((2, 1))}
        del checkpoint["param_groups"][0]["stochastic_rounding"]
        del checkpoint["param_groups"][0]["rounding_seed"]
        optimizer.load_state_dict(checkpoint)
        self.assertEqual(int(optimizer.state[parameter]["rounding_counter"]), 0)
        torch.testing.assert_close(
            optimizer.state[parameter]["sum"], torch.ones((2, 1))
        )

        meta = torch.nn.Parameter(
            torch.empty((2, 4), dtype=torch.float16, device="meta")
        )
        meta_optimizer = self._optimizer(meta)
        prime_optimizer_state(meta_optimizer)
        self.assertEqual(meta_optimizer.state[meta]["sum"].device.type, "meta")
        self.assertEqual(
            meta_optimizer.state[meta]["rounding_counter"].device.type, "cpu"
        )

    def test_fp32_does_not_allocate_rounding_state(self) -> None:
        parameter = torch.nn.Parameter(torch.ones((2, 4)))
        optimizer = self._optimizer(parameter)
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        self.assertEqual(set(optimizer.state[parameter]), {"sum"})
        torch.testing.assert_close(parameter, torch.full_like(parameter, 1 - 2**-12))

    def test_meta_optimizer_traces_keyed_rounding(self) -> None:
        parameter = torch.nn.Parameter(
            torch.ones((2, 4), dtype=torch.float16, device="meta")
        )
        parameter.grad = torch.ones_like(parameter)
        optimizer = self._optimizer(parameter)
        prime_optimizer_state(optimizer)
        graph = make_fx(
            lambda: optimizer.step(), tracing_mode="fake", _allow_non_fake_inputs=True
        )()
        targets = [node.target for node in graph.graph.nodes]
        self.assertIn(torch.ops.recir.stochastic_round_fp16.default, targets)


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

    def test_warmup_lr_matches_the_controller_at_every_iteration(self) -> None:
        schedule = [
            WarmupStage(policy=WarmupPolicy.LINEAR, max_iters=5, value=0.1),
            WarmupStage(policy=WarmupPolicy.LINEAR, max_iters=9, value=0.5),
        ]
        sparse_lr = torch.tensor(0.0, dtype=torch.float32)
        controller = WarmupController(
            dense_base_lr=0.01,
            dense_lr_warmup_schedule=None,
            dense_lr_buffer=torch.tensor(0.0, dtype=torch.float32),
            iter_buffer=torch.zeros((), dtype=torch.int64),
            sparse_base_lr=0.2,
            sparse_lr_warmup_schedule=schedule,
            sparse_lr_buffer=sparse_lr,
        )
        cfg = types.SimpleNamespace(lr=0.2, lr_warmup_schedule=schedule)

        controller.apply()
        for iteration in range(12):
            # pyrefly: ignore[bad-argument-type]
            self.assertAlmostEqual(float(sparse_lr), warmup_lr(cfg, iteration))
            controller.step()

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
