# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-ignore-all-errors

"""Tests for single-process multi-device XLA compilation via @compile.

Unlike test_gspmd.py (which uses DTensorTestBase to spawn one subprocess per
GPU), this test runs a single process that sees all local devices. This is the
required execution model for TPU VMs where libtpu only allows one process to
access all chips.

The same tests also work on multi-GPU machines with JAX_PLATFORMS=cuda, making
them runnable in CI without TPU hardware.

Test matrix row: num_processes=1, num_ranks=N (all local devices).
"""

import unittest

import torch
from parameterized import parameterized
from recir.api import compile
from recir.backend import BackendType
from recir.backends import XLAGSPMDBackend, XLAShardMapBackend
from recir.distributed.parallel_mode import DataParallel, GSPMDParallel
from recir.tests.xla.gspmd_test_utils import (
    _run_single_rank_baseline,
    assert_hlo_has_collective,
    assert_no_graph_breaks,
    DIM,
    LOCAL_BS,
    NUM_ITERS,
    SimpleMLP,
)


class TestSingleProcessMultiDevice(unittest.TestCase):
    """Single-process multi-device tests for XLA backends.

    num_processes=1, num_ranks=N (all local devices in one process).
    """

    def setUp(self):
        import jax  # @manual

        self.devices = jax.devices()
        if len(self.devices) < 2:
            self.skipTest("Need >= 2 JAX devices")
        self.num_ranks = len(self.devices)

    @parameterized.expand([("sum",), ("mean",)])
    def test_gspmd_single_process(self, loss_reduction):
        """GSPMDParallel + XLA_GSPMD with 1 process, N devices."""
        torch.manual_seed(42)
        model = SimpleMLP().cpu()
        opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
        # Single process holds the full global batch
        x = torch.randn(LOCAL_BS * self.num_ranks, DIM)
        baseline_params = _run_single_rank_baseline(
            x, self.num_ranks, loss_reduction=loss_reduction
        )
        reduce_fn = torch.sum if loss_reduction == "sum" else torch.mean

        @compile(
            parallel_mode=GSPMDParallel(
                "replicate",
                loss_reduction=loss_reduction,
                num_ranks=self.num_ranks,
                num_processes=1,
            ),
            backend=XLAGSPMDBackend(),
        )
        def train_step(mod, opt, x):
            reduce_fn(mod(x)).backward()
            opt.step()

        for step in range(NUM_ITERS):
            train_step(model, opt, x)
            for (name, param), (_, baseline_param) in zip(
                model.named_parameters(),
                baseline_params[step].items(),
            ):
                torch.testing.assert_close(
                    param,
                    baseline_param,
                    msg=f"Step {step}: XLA_GSPMD vs single-rank mismatch for {name}",
                )
        compiled_obj = train_step.__dict__["_compiled_obj"]
        assert_hlo_has_collective(self, compiled_obj, "all-reduce", "XLA_GSPMD")

    @parameterized.expand([("sum",), ("mean",)])
    def test_shard_map_single_process(self, loss_reduction):
        """DataParallel + XLA_SHARD_MAP with 1 process, N devices."""
        torch.manual_seed(42)
        model = SimpleMLP().cpu()
        opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
        x = torch.randn(LOCAL_BS * self.num_ranks, DIM)
        baseline_params = _run_single_rank_baseline(
            x, self.num_ranks, loss_reduction=loss_reduction
        )
        reduce_fn = torch.sum if loss_reduction == "sum" else torch.mean

        @compile(
            parallel_mode=DataParallel(
                "replicate",
                loss_reduction=loss_reduction,
                num_ranks=self.num_ranks,
                num_processes=1,
                device_type="cpu",
                backend=BackendType.XLA_SHARD_MAP,
            ),
            backend=XLAShardMapBackend(),
        )
        def train_step(mod, opt, x):
            reduce_fn(mod(x)).backward()
            opt.step()

        for step in range(NUM_ITERS):
            train_step(model, opt, x)
            for (name, param), (_, baseline_param) in zip(
                model.named_parameters(),
                baseline_params[step].items(),
            ):
                torch.testing.assert_close(
                    param,
                    baseline_param,
                    msg=f"Step {step}: XLA_SHARD_MAP vs single-rank mismatch for {name}",
                )
        compiled_obj = train_step.__dict__["_compiled_obj"]
        assert_no_graph_breaks(self, compiled_obj)
        assert_hlo_has_collective(self, compiled_obj, "all_reduce", "XLA_SHARD_MAP")


if __name__ == "__main__":
    unittest.main()
