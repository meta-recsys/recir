# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest

import jax
import jax.numpy as jnp
import numpy as np
import recir.ops.linear_cross_entropy_op  # noqa: F401
import recir.torchax
import torch
from recir.ops.linear_cross_entropy_op import _reference_loss
from recir.torchax.ops.jlinear_cross_entropy import (
    linear_cross_entropy_scan,
    linear_cross_entropy_scan_backward,
)
from torch.fx.experimental.proxy_tensor import make_fx


class LinearCrossEntropyOpTest(unittest.TestCase):
    def _inputs(self):
        torch.manual_seed(0)
        hidden = torch.randn(2, 5, 4)
        labels = torch.tensor([[0, 1, -100, 3, 2], [3, 2, 1, 0, -100]])
        weight = torch.randn(6, 4)
        return hidden, labels, weight

    def test_op_registered(self):
        self.assertTrue(hasattr(torch.ops.recir, "xla_linear_cross_entropy"))
        self.assertTrue(hasattr(torch.ops.recir, "xla_linear_cross_entropy_backward"))

    def test_forward_matches_reference_with_softcap_and_padded_vocab(self):
        hidden, labels, weight = self._inputs()

        actual = torch.ops.recir.xla_linear_cross_entropy(
            hidden,
            labels,
            weight,
            4,
            4,
            -100,
            30.0,
        )
        expected = _reference_loss(hidden, labels, weight, 4, -100, 30.0)

        torch.testing.assert_close(actual, expected)

    def test_autograd_matches_reference(self):
        hidden, labels, weight = self._inputs()
        op_hidden = hidden.clone().requires_grad_(True)
        op_weight = weight.clone().requires_grad_(True)
        reference_hidden = hidden.clone().requires_grad_(True)
        reference_weight = weight.clone().requires_grad_(True)

        actual = torch.ops.recir.xla_linear_cross_entropy(
            op_hidden,
            labels,
            op_weight,
            3,
            4,
            -100,
            30.0,
        )
        expected = _reference_loss(
            reference_hidden,
            labels,
            reference_weight,
            4,
            -100,
            30.0,
        )
        actual.backward()
        expected.backward()

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(op_hidden.grad, reference_hidden.grad)
        torch.testing.assert_close(op_weight.grad, reference_weight.grad)

    def test_forward_and_backward_remain_opaque_during_make_fx(self):
        hidden, labels, weight = self._inputs()
        hidden.requires_grad_(True)
        weight.requires_grad_(True)

        def forward_backward(
            input_hidden: torch.Tensor,
            input_labels: torch.Tensor,
            input_weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            loss = torch.ops.recir.xla_linear_cross_entropy(
                input_hidden,
                input_labels,
                input_weight,
                3,
                4,
                -100,
                30.0,
            )
            return torch.autograd.grad(loss, (input_hidden, input_weight))

        graph = make_fx(forward_backward, tracing_mode="fake")(
            hidden,
            labels,
            weight,
        )
        targets = [node.target for node in graph.graph.nodes]

        self.assertIn(torch.ops.recir.xla_linear_cross_entropy.default, targets)
        self.assertIn(
            torch.ops.recir.xla_linear_cross_entropy_backward.default,
            targets,
        )

    def test_fake_shape_and_dtype(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            hidden = torch.randn(2, 5, 4, dtype=torch.bfloat16)
            labels = torch.zeros(2, 5, dtype=torch.long)
            weight = torch.randn(6, 4, dtype=torch.bfloat16)
            loss = torch.ops.recir.xla_linear_cross_entropy(
                hidden,
                labels,
                weight,
                4,
                4,
                -100,
                30.0,
            )

            self.assertEqual((), tuple(loss.shape))
            self.assertEqual(torch.float32, loss.dtype)

    def test_torchax_dispatch_uses_registered_scan_lowering(self):
        hidden, labels, weight = self._inputs()
        environment = recir.torchax.default_env()
        xla_hidden, xla_labels, xla_weight = environment.to_xla(
            (hidden, labels, weight)
        )

        with environment:
            loss = torch.ops.recir.xla_linear_cross_entropy(
                xla_hidden,
                xla_labels,
                xla_weight,
                3,
                4,
                -100,
                30.0,
            )

        expected = _reference_loss(hidden, labels, weight, 4, -100, 30.0)
        torch.testing.assert_close(loss.torch(), expected)

    def test_jax_scan_matches_reference_loss_and_gradients(self):
        hidden, labels, weight = self._inputs()
        hidden_jax = jnp.asarray(hidden.numpy())
        labels_jax = jnp.asarray(labels.numpy())
        weight_jax = jnp.asarray(weight.numpy())

        def loss_fn(input_hidden, input_weight):
            return linear_cross_entropy_scan(
                input_hidden,
                labels_jax,
                input_weight,
                3,
                4,
                -100,
                30.0,
            )

        actual_loss = loss_fn(hidden_jax, weight_jax)
        actual_hidden_grad, actual_weight_grad = linear_cross_entropy_scan_backward(
            jnp.ones((), dtype=jnp.float32),
            hidden_jax,
            labels_jax,
            weight_jax,
            3,
            4,
            -100,
            30.0,
        )
        reference_hidden = hidden.clone().requires_grad_(True)
        reference_weight = weight.clone().requires_grad_(True)
        expected_loss = _reference_loss(
            reference_hidden,
            labels,
            reference_weight,
            4,
            -100,
            30.0,
        )
        expected_loss.backward()

        np.testing.assert_allclose(actual_loss, expected_loss.detach().numpy(), 1e-5)
        np.testing.assert_allclose(
            actual_hidden_grad,
            reference_hidden.grad.numpy(),
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            actual_weight_grad,
            reference_weight.grad.numpy(),
            rtol=1e-5,
            atol=1e-5,
        )
        forward_hlo = jax.jit(loss_fn).lower(hidden_jax, weight_jax).as_text()
        backward_lowered = jax.jit(
            linear_cross_entropy_scan_backward,
            static_argnums=(4, 5, 6, 7),
        ).lower(
            jnp.ones((), dtype=jnp.float32),
            hidden_jax,
            labels_jax,
            weight_jax,
            3,
            4,
            -100,
            30.0,
        )
        backward_hlo = backward_lowered.as_text()
        self.assertIn("while", forward_hlo)
        self.assertIn("while", backward_hlo)
        self.assertIn("dynamic_update_slice", backward_hlo)
        self.assertIn(
            "while", jax.jit(loss_fn).lower(hidden_jax, weight_jax).compile().as_text()
        )
        self.assertIn("while", backward_lowered.compile().as_text())

    def test_jax_bfloat16_scan_uses_fp32_tile_logits_with_nonunit_gradient(self):
        hidden, labels, weight = self._inputs()
        hidden_jax = jnp.asarray(hidden.numpy(), dtype=jnp.bfloat16)
        labels_jax = jnp.asarray(labels.numpy())
        weight_jax = jnp.asarray(weight.numpy(), dtype=jnp.bfloat16)
        grad_out = 0.125

        loss = linear_cross_entropy_scan(
            hidden_jax,
            labels_jax,
            weight_jax,
            3,
            4,
            -100,
            30.0,
        )
        grad_hidden, grad_weight = linear_cross_entropy_scan_backward(
            jnp.asarray(grad_out, dtype=jnp.float32),
            hidden_jax,
            labels_jax,
            weight_jax,
            3,
            4,
            -100,
            30.0,
        )
        reference_hidden = hidden.to(dtype=torch.bfloat16).requires_grad_(True)
        reference_weight = weight.to(dtype=torch.bfloat16).requires_grad_(True)
        reference_loss = _reference_loss(
            reference_hidden,
            labels,
            reference_weight,
            4,
            -100,
            30.0,
        )
        reference_loss.backward(torch.tensor(grad_out))

        self.assertEqual(jnp.float32, loss.dtype)
        self.assertEqual(jnp.bfloat16, grad_hidden.dtype)
        self.assertEqual(jnp.bfloat16, grad_weight.dtype)
        np.testing.assert_allclose(
            np.asarray(loss),
            reference_loss.detach().numpy(),
            rtol=2e-2,
            atol=1e-1,
        )
        np.testing.assert_allclose(
            np.asarray(grad_hidden, dtype=np.float32),
            reference_hidden.grad.float().numpy(),
            rtol=5e-2,
            atol=5e-2,
        )
        np.testing.assert_allclose(
            np.asarray(grad_weight, dtype=np.float32),
            reference_weight.grad.float().numpy(),
            rtol=5e-2,
            atol=5e-2,
        )
        np.testing.assert_array_equal(
            np.asarray(grad_weight[4:], dtype=np.float32),
            np.zeros((2, 4), dtype=np.float32),
        )

    def test_all_ignored_tokens_produce_zero_loss_and_gradients(self):
        hidden, _, weight = self._inputs()
        labels = jnp.full((2, 5), -100, dtype=jnp.int64)

        loss = linear_cross_entropy_scan(
            jnp.asarray(hidden.numpy()),
            labels,
            jnp.asarray(weight.numpy()),
            3,
            4,
            -100,
            30.0,
        )
        grad_hidden, grad_weight = linear_cross_entropy_scan_backward(
            jnp.ones((), dtype=jnp.float32),
            jnp.asarray(hidden.numpy()),
            labels,
            jnp.asarray(weight.numpy()),
            3,
            4,
            -100,
            30.0,
        )

        self.assertEqual(0.0, float(loss))
        np.testing.assert_array_equal(grad_hidden, np.zeros(hidden.shape))
        np.testing.assert_array_equal(grad_weight, np.zeros(weight.shape))

    def test_invalid_label_is_not_silently_ignored(self):
        hidden, labels, weight = self._inputs()
        invalid_labels = labels.clone()
        invalid_labels[0, 0] = 5

        with self.assertRaises(IndexError):
            _reference_loss(hidden, invalid_labels, weight, 4, -100, 30.0)
        jax_loss = linear_cross_entropy_scan(
            jnp.asarray(hidden.numpy()),
            jnp.asarray(invalid_labels.numpy()),
            jnp.asarray(weight.numpy()),
            3,
            4,
            -100,
            30.0,
        )

        self.assertTrue(bool(jnp.isnan(jax_loss)))
