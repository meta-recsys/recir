# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-ignore-all-errors

"""Tests for DLRM model compilation through RecIR's XLA pipeline.

Validates that a recommendation model (dense embeddings, bottom MLP,
DCNv2 cross-network, top MLP) can trace through make_fx, partition
with DTensor, lower to XLA, and execute correctly on multiple devices.

Uses the single-process multi-device pattern (TPU VM execution model):
num_processes=1, num_ranks=N (all local devices in one process).
"""

import unittest

import torch
from parameterized import parameterized
from recir.api import compile, sync_device_state_to_torch
from recir.backend import BackendType
from recir.distributed.parallel_mode import DataParallel, GSPMDParallel
from recir.gm_transformation import GraphModuleTransformation
from recir.lowering.xla_lower import _is_xla_compatible, _XLAModule
from recir.tests.xla.gspmd_test_utils import (
    _generate_dlrm_inputs,
    _run_dlrm_baseline,
    assert_hlo_has_collective,
    assert_no_graph_breaks,
    LOCAL_BS,
    NUM_ITERS,
    SimpleDLRM,
)


class TestDLRMSingleProcess(unittest.TestCase):
    def setUp(self):
        import jax  # @manual

        self.devices = jax.devices()
        if len(self.devices) < 2:
            self.skipTest("Need >= 2 JAX devices")
        self.num_ranks = len(self.devices)

    @parameterized.expand([("sum",), ("mean",)])
    def test_dlrm_gspmd(self, loss_reduction):
        self.assertGreaterEqual(self.num_ranks, 2)
        torch.manual_seed(42)
        global_bs = LOCAL_BS * self.num_ranks
        model = SimpleDLRM().cpu()
        opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
        dense, sparse_indices, labels = _generate_dlrm_inputs(global_bs)
        baseline_params = _run_dlrm_baseline(
            dense,
            sparse_indices,
            labels,
            self.num_ranks,
            loss_reduction=loss_reduction,
        )
        loss_fn = torch.nn.BCEWithLogitsLoss(
            reduction="sum" if loss_reduction == "sum" else "mean"
        )

        @compile(
            parallel_mode=GSPMDParallel(
                "replicate",
                loss_reduction=loss_reduction,
                num_ranks=self.num_ranks,
                num_processes=1,
            ),
            gm_transformation=GraphModuleTransformation(
                backend=BackendType.XLA_GSPMD,
                enable_device_state=True,
            ),
        )
        def train_step(mod, opt, dense, sparse_indices, labels):
            logits = mod(dense, sparse_indices)
            loss = loss_fn(logits, labels)
            loss.backward()
            opt.step()

        for step in range(NUM_ITERS):
            train_step(model, opt, dense, sparse_indices, labels)
            compiled_obj = train_step.__dict__["_compiled_obj"]
            sync_device_state_to_torch(train_step)
            self.assertIsNotNone(compiled_obj.state_manager.device_state)
            for (name, param), (_, baseline_param) in zip(
                model.named_parameters(),
                baseline_params[step].items(),
            ):
                torch.testing.assert_close(
                    param,
                    baseline_param,
                    msg=f"Step {step}: DLRM XLA_GSPMD vs baseline mismatch for {name}",
                )
        compiled_obj = train_step.__dict__["_compiled_obj"]
        assert_hlo_has_collective(self, compiled_obj, "all-reduce", "XLA_GSPMD")

    @parameterized.expand(
        [
            ("sum", True),
            ("sum", False),
            ("mean", True),
            ("mean", False),
        ]
    )
    def test_dlrm_shard_map(self, loss_reduction, use_parametrization):
        self.assertGreaterEqual(self.num_ranks, 2)
        torch.manual_seed(42)
        global_bs = LOCAL_BS * self.num_ranks
        model = SimpleDLRM().cpu()
        opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
        dense, sparse_indices, labels = _generate_dlrm_inputs(global_bs)
        baseline_params = _run_dlrm_baseline(
            dense,
            sparse_indices,
            labels,
            self.num_ranks,
            loss_reduction=loss_reduction,
        )
        loss_fn = torch.nn.BCEWithLogitsLoss(
            reduction="sum" if loss_reduction == "sum" else "mean"
        )

        @compile(
            parallel_mode=DataParallel(
                "replicate",
                loss_reduction=loss_reduction,
                num_ranks=self.num_ranks,
                num_processes=1,
                device_type="cpu",
                backend=BackendType.XLA_SHARD_MAP,
                use_parametrization=use_parametrization,
            ),
            gm_transformation=GraphModuleTransformation(
                backend=BackendType.XLA_SHARD_MAP,
                enable_device_state=True,
            ),
        )
        def train_step(mod, opt, dense, sparse_indices, labels):
            logits = mod(dense, sparse_indices)
            loss = loss_fn(logits, labels)
            loss.backward()
            opt.step()

        for step in range(NUM_ITERS):
            train_step(model, opt, dense, sparse_indices, labels)
            compiled_obj = train_step.__dict__["_compiled_obj"]
            sync_device_state_to_torch(train_step)
            self.assertIsNotNone(compiled_obj.state_manager.device_state)
            for (name, param), (_, baseline_param) in zip(
                model.named_parameters(),
                baseline_params[step].items(),
            ):
                torch.testing.assert_close(
                    param,
                    baseline_param,
                    msg=f"Step {step}: DLRM XLA_SHARD_MAP vs baseline mismatch for {name}",
                )
        compiled_obj = train_step.__dict__["_compiled_obj"]
        assert_no_graph_breaks(self, compiled_obj)
        assert_hlo_has_collective(self, compiled_obj, "all_reduce", "XLA_SHARD_MAP")

    def test_no_graph_breaks_gspmd(self):
        self.assertGreaterEqual(self.num_ranks, 2)
        torch.manual_seed(42)
        global_bs = LOCAL_BS * self.num_ranks
        model = SimpleDLRM().cpu()
        opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
        dense, sparse_indices, labels = _generate_dlrm_inputs(global_bs)
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")

        @compile(
            parallel_mode=GSPMDParallel(
                "replicate",
                loss_reduction="sum",
                num_ranks=self.num_ranks,
                num_processes=1,
            ),
            gm_transformation=GraphModuleTransformation(
                backend=BackendType.XLA_GSPMD,
                enable_device_state=True,
            ),
        )
        def train_step(mod, opt, dense, sparse_indices, labels):
            logits = mod(dense, sparse_indices)
            loss = loss_fn(logits, labels)
            loss.backward()
            opt.step()

        train_step(model, opt, dense, sparse_indices, labels)
        compiled_obj = train_step.__dict__["_compiled_obj"]
        for gm_name in ("setup_gm", "main_gm", "cleanup_gm"):
            gm = getattr(compiled_obj.gm, gm_name)
            xla_modules = [m for m in gm.modules() if isinstance(m, _XLAModule)]
            incompatible = []
            for node in gm.graph.nodes:
                if node.op in ("placeholder", "output", "call_module"):
                    continue
                ok, reason = _is_xla_compatible(node)
                if not ok:
                    incompatible.append(f"  {node.target} — {reason}")
            self.assertEqual(
                len(xla_modules),
                1,
                f"{gm_name}: expected exactly 1 XLA subgraph (no graph breaks), "
                f"got {len(xla_modules)}.\nIncompatible nodes:\n"
                + "\n".join(incompatible or ["  (none)"]),
            )

    def test_embedding_numerical_exactness(self):
        import jax  # @manual
        import jax.numpy as jnp  # @manual
        import recir.torchax as torchax
        from recir.torchax.interop import call_torch

        self.assertGreaterEqual(self.num_ranks, 2)
        torch.manual_seed(42)

        weight = torch.randn(32, 8)
        indices = torch.randint(0, 32, (4, 3))

        # CPU baseline: embedding forward + backward
        weight_cpu = weight.clone().requires_grad_(True)
        out_cpu = torch.nn.functional.embedding(indices, weight_cpu).sum()
        out_cpu.backward()
        grad_cpu = weight_cpu.grad.clone()

        # torchax path: run same ops through JAX
        torchax.default_env()

        w_jax = jnp.array(weight.numpy())
        idx_jax = jnp.array(indices.numpy().astype("int32"))

        def jax_embedding_loss(w_jax):
            def torch_fn(w, idx):
                return torch.nn.functional.embedding(idx, w).sum()

            return call_torch(torch_fn, w_jax, idx_jax)

        # Forward check
        out_jax = jax_embedding_loss(w_jax)
        torch.testing.assert_close(
            torch.tensor(out_jax.__array__()),
            out_cpu.detach(),
            msg="Embedding forward through torchax should match CPU exactly",
        )
        # Backward check: jax.grad computes d(loss)/d(weight)
        grad_jax = jax.grad(jax_embedding_loss)(w_jax)
        grad_from_jax = torch.tensor(grad_jax.__array__())
        torch.testing.assert_close(
            grad_from_jax,
            grad_cpu,
            msg="Embedding backward through torchax should match CPU exactly",
        )


if __name__ == "__main__":
    unittest.main()
