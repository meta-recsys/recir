# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Unit tests for FSDP support in the GSPMD path."""

import unittest
from typing import Any, Callable, Tuple

import jax  # @manual
import torch
import torch.nn as nn
from jax.sharding import Mesh  # @manual
from parameterized import parameterized
from recir.distributed.parallel_mode import GSPMDParallel
from recir.lowering.xla_lower import _XLAModule

# IMPORTANT: must run BEFORE any `import jax` so the cuda plugin path patch
# is in effect when jax.devices() / jax.* first triggers plugin discovery.
from recir.tests.xla import _jax_cuda_plugin_fix  # noqa: F401, I001


class TestGSPMDShardParams(unittest.TestCase):
    def _make_model_and_opt(self, dim: int = 256) -> Tuple[nn.Linear, torch.optim.SGD]:
        model = nn.Linear(dim, dim)
        opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=True)
        x = torch.randn(4, dim)
        model(x).sum().backward()
        opt.step()
        opt.zero_grad()
        return model, opt

    def test_shard_params_noop_for_replicate(self) -> None:
        mode = GSPMDParallel("replicate", num_ranks=4, num_processes=2)
        model, opt = self._make_model_and_opt()
        orig_weight_shape = model.weight.shape
        orig_bias_shape = model.bias.shape
        params_and_buffers = dict(model.named_parameters())
        mode.shard_params(model, params_and_buffers, opt)
        self.assertEqual(model.weight.shape, orig_weight_shape)
        self.assertEqual(model.bias.shape, orig_bias_shape)

    def test_shard_params_shards_weight(self) -> None:
        dim = 256
        num_procs = 2
        num_ranks = 4
        mode = GSPMDParallel(
            "fully_shard", num_ranks=num_ranks, num_processes=num_procs
        )
        model, opt = self._make_model_and_opt(dim=dim)
        original_weight = model.weight.clone()
        params_and_buffers = {
            **dict(model.named_parameters()),
            **dict(model.named_buffers()),
        }
        mode.shard_params(model, params_and_buffers, opt)
        expected_dim0 = dim // num_procs
        self.assertEqual(model.weight.shape[0], expected_dim0)
        self.assertEqual(model.weight.shape[1], dim)
        torch.testing.assert_close(
            model.weight, original_weight[:expected_dim0, :], atol=0, rtol=0
        )

    def test_shard_params_skips_small_tensors(self) -> None:
        dim = 256
        num_ranks = 4
        num_procs = 2
        mode = GSPMDParallel(
            "fully_shard", num_ranks=num_ranks, num_processes=num_procs
        )
        model, opt = self._make_model_and_opt(dim=dim)
        params_and_buffers = {
            **dict(model.named_parameters()),
            **dict(model.named_buffers()),
        }
        mode.shard_params(model, params_and_buffers, opt)
        self.assertEqual(model.bias.shape[0], dim // num_procs)

    def test_shard_params_skips_non_divisible(self) -> None:
        model = nn.Linear(7, 7)
        opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=True)
        x = torch.randn(2, 7)
        model(x).sum().backward()
        opt.step()
        opt.zero_grad()
        mode = GSPMDParallel("fully_shard", num_ranks=4, num_processes=2)
        orig_shape = model.weight.shape
        params_and_buffers = dict(model.named_parameters())
        mode.shard_params(model, params_and_buffers, opt)
        self.assertEqual(model.weight.shape, orig_shape)

    def test_shard_params_shards_optimizer_states(self) -> None:
        dim = 256
        num_procs = 2
        num_ranks = 4
        mode = GSPMDParallel(
            "fully_shard", num_ranks=num_ranks, num_processes=num_procs
        )
        model = nn.Linear(dim, dim)
        opt = torch.optim.Adam(model.parameters(), lr=0.01, foreach=True)
        x = torch.randn(4, dim)
        model(x).sum().backward()
        opt.step()
        opt.zero_grad()
        weight_state = opt.state[model.weight]
        self.assertIn("exp_avg", weight_state)
        orig_exp_avg_shape = weight_state["exp_avg"].shape
        self.assertEqual(orig_exp_avg_shape, (dim, dim))
        params_and_buffers = {
            **dict(model.named_parameters()),
            **dict(model.named_buffers()),
        }
        mode.shard_params(model, params_and_buffers, opt)
        new_state = opt.state[model.weight]  # pyre-ignore[6]
        expected_dim0 = dim // num_procs
        self.assertEqual(new_state["exp_avg"].shape[0], expected_dim0)
        self.assertEqual(new_state["exp_avg_sq"].shape[0], expected_dim0)

    def test_shard_params_updates_params_and_buffers_dict(self) -> None:
        dim = 256
        mode = GSPMDParallel("fully_shard", num_ranks=4, num_processes=2)
        model = nn.Linear(dim, dim)
        opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=True)
        x = torch.randn(4, dim)
        model(x).sum().backward()
        opt.step()
        opt.zero_grad()
        params_and_buffers = {
            **dict(model.named_parameters()),
            **dict(model.named_buffers()),
        }
        mode.shard_params(model, params_and_buffers, opt)
        self.assertEqual(params_and_buffers["weight"].shape[0], dim // 2)

    def test_shard_params_buffers(self) -> None:
        dim = 256
        num_procs = 2
        mode = GSPMDParallel("fully_shard", num_ranks=4, num_processes=num_procs)

        class ModelWithBuffer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(dim, dim)
                self.register_buffer("running_mean", torch.zeros(dim))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x)

        model = ModelWithBuffer()
        opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=True)
        x = torch.randn(4, dim)
        model(x).sum().backward()
        opt.step()
        opt.zero_grad()
        params_and_buffers = {
            **dict(model.named_parameters()),
            **dict(model.named_buffers()),
        }
        mode.shard_params(model, params_and_buffers, opt)
        self.assertEqual(model.running_mean.shape[0], dim // num_procs)


class TestToShardedJaxFSDP(unittest.TestCase):
    """Test _XLAModule._to_sharded_jax with is_fsdp_weight=True."""

    def setUp(self) -> None:
        self.devices = jax.devices()
        if len(self.devices) < 2:
            self.skipTest("Need >= 2 JAX devices")
        self.num_local: int = len(jax.local_devices())
        self.num_global: int = jax.device_count()

    def test_fsdp_weight_creates_sharded_array(self) -> None:
        """FSDP weight should produce a globally sharded JAX array."""
        import jax  # @manual  # pyre-ignore[21]
        from jax.sharding import Mesh  # @manual  # pyre-ignore[21]
        from recir.lowering.xla_lower import _XLAModule

        mesh = Mesh(jax.devices(), axis_names=("torch_dist",))
        local_shard = torch.randn(self.num_local, 4)
        jax_array = _XLAModule._to_sharded_jax(local_shard, mesh, is_fsdp_weight=True)
        expected_global_shape = (self.num_global, 4)
        self.assertEqual(jax_array.shape, expected_global_shape)
        sharding = jax_array.sharding
        self.assertIsNotNone(sharding)

    def test_fsdp_weight_preserves_values(self) -> None:
        """FSDP weight sharding should preserve tensor values in the global array."""
        import jax  # @manual  # pyre-ignore[21]
        import jax.numpy as jnp  # @manual  # pyre-ignore[21]
        from jax.sharding import Mesh  # @manual  # pyre-ignore[21]
        from recir.lowering.xla_lower import _XLAModule

        mesh = Mesh(jax.devices(), axis_names=("torch_dist",))
        local_shard = torch.arange(self.num_local * 2, dtype=torch.float32).reshape(
            self.num_local, 2
        )
        jax_array = _XLAModule._to_sharded_jax(local_shard, mesh, is_fsdp_weight=True)
        first_shard = jax_array[: self.num_local, :]
        expected = jnp.asarray(local_shard.numpy())
        torch.testing.assert_close(
            torch.tensor(first_shard.__array__()), torch.tensor(expected.__array__())
        )

    def test_fsdp_weight_1d_tensor(self) -> None:
        mesh = Mesh(jax.devices(), axis_names=("torch_dist",))
        local_shard = torch.randn(self.num_local * 4)
        jax_array = _XLAModule._to_sharded_jax(local_shard, mesh, is_fsdp_weight=True)
        expected_global_dim = self.num_local * 4 * self.num_global // self.num_local
        self.assertEqual(jax_array.shape, (expected_global_dim,))

    def test_non_fsdp_weight_replicated(self) -> None:
        """Without is_fsdp_weight, tensors should be replicated."""
        import jax  # @manual  # pyre-ignore[21]
        from jax.sharding import Mesh  # @manual  # pyre-ignore[21]
        from recir.lowering.xla_lower import _XLAModule

        mesh = Mesh(jax.devices(), axis_names=("torch_dist",))
        t = torch.randn(4, 8)
        jax_array = _XLAModule._to_sharded_jax(t, mesh, is_fsdp_weight=False)
        self.assertEqual(jax_array.shape, (4, 8))


class TestGSPMDFSDPSingleProcess(unittest.TestCase):
    """End-to-end test: GSPMDParallel('fully_shard') single-process compilation."""

    def setUp(self) -> None:
        import jax  # @manual  # pyre-ignore[21]

        self.devices = jax.devices()
        if len(self.devices) < 2:
            self.skipTest("Need >= 2 JAX devices")
        self.num_ranks: int = len(self.devices)

    def test_gspmd_fully_shard_compiles(self) -> None:
        """GSPMDParallel('fully_shard') should compile and set FSDP metadata."""
        from recir.api import compile
        from recir.backend import BackendType
        from recir.gm_transformation import GraphModuleTransformation
        from recir.tests.xla.gspmd_test_utils import DIM, SimpleMLP

        torch.manual_seed(42)
        dim: int = DIM
        if dim % self.num_ranks != 0:
            dim = self.num_ranks * (dim // self.num_ranks + 1)
        model = SimpleMLP(dim=dim).cpu()
        opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
        x = torch.randn(self.num_ranks, dim)

        @compile(
            parallel_mode=GSPMDParallel(
                "fully_shard",
                loss_reduction="sum",
                num_ranks=self.num_ranks,
                num_processes=1,
            ),
            gm_transformation=GraphModuleTransformation(  # pyre-ignore[6]
                backend=BackendType.XLA_GSPMD,
            ),
        )
        def train_step(mod: Any, opt: Any, x: Any) -> None:
            torch.sum(mod(x)).backward()
            opt.step()

        train_step(model, opt, x)

        compiled_obj = train_step.__dict__["_compiled_obj"]
        self.assertIsNotNone(compiled_obj)
        has_fsdp_meta = any(
            "is_fsdp_sharded" in n.meta
            for n in compiled_obj.gm.graph.nodes
            if n.op == "placeholder"
        )
        self.assertTrue(
            has_fsdp_meta, "Expected is_fsdp_sharded metadata on placeholders"
        )

    @parameterized.expand([("sum",), ("mean",)])
    def test_gspmd_fsdp_device_state_numerics(self, loss_reduction: str) -> None:
        """FSDP + GSPMD + enable_device_state=True matches CPU baseline."""
        from recir.api import compile, sync_device_state_to_torch
        from recir.backend import BackendType
        from recir.gm_transformation import GraphModuleTransformation
        from recir.tests.xla.gspmd_test_utils import (
            _run_single_rank_baseline,
            DIM,
            LOCAL_BS,
            NUM_ITERS,
            SimpleMLP,
        )

        torch.manual_seed(42)
        dim: int = DIM
        if dim % self.num_ranks != 0:
            dim = self.num_ranks * (dim // self.num_ranks + 1)
        model = SimpleMLP(dim=dim).cpu()
        opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
        x = torch.randn(LOCAL_BS * self.num_ranks, dim)
        baseline_params = _run_single_rank_baseline(
            x, self.num_ranks, loss_reduction=loss_reduction
        )
        reduce_fn: Callable[..., torch.Tensor] = (
            torch.sum if loss_reduction == "sum" else torch.mean
        )

        @compile(
            parallel_mode=GSPMDParallel(
                "fully_shard",
                loss_reduction=loss_reduction,
                num_ranks=self.num_ranks,
                num_processes=1,
            ),
            gm_transformation=GraphModuleTransformation(  # pyre-ignore[6]
                backend=BackendType.XLA_GSPMD,
                enable_device_state=True,
            ),
        )
        def train_step(mod: Any, opt: Any, x: Any) -> None:
            reduce_fn(mod(x)).backward()
            opt.step()

        for step in range(NUM_ITERS):
            train_step(model, opt, x)
            sync_device_state_to_torch(train_step)
            for (name, param), (_, baseline_param) in zip(
                model.named_parameters(),
                baseline_params[step].items(),
            ):
                torch.testing.assert_close(
                    param,
                    baseline_param,
                    msg=f"Step {step}: FSDP GSPMD device_state vs baseline "
                    f"mismatch for {name}",
                )


if __name__ == "__main__":
    unittest.main()
