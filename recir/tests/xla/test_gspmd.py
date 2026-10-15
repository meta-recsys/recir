# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors

"""Tests for GSPMD and shard_map XLA compilation via @compile.

Uses DTensorTestBase (multi-process, 2 GPUs) so both XLA_SHARD_MAP
(DTensor + shard_map) and XLA_GSPMD (auto-parallel) paths exercise
real multi-GPU collectives.
"""

import torch
from parameterized import parameterized
from recir.api import compile
from recir.backends import XLAGSPMDBackend, XLAShardMapBackend
from recir.backends.types import BackendType
from recir.distributed.parallel_mode import DataParallel, GSPMDParallel
from recir.tests.xla.gspmd_test_utils import (
    _create_model_and_warmup,
    _generate_inputs,
    _run_ddp_baseline,
    assert_hlo_has_collective,
    assert_no_graph_breaks,
    DestroyPGMixin,
    NUM_ITERS,
)
from recir.tests.xla.xla_test_mixin import jax_distributed
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

# Port offsets per (test, reduction) to avoid JAX coordinator collisions
# between parameterized variants running in separate subprocesses.
_SHARD_MAP_PORT_OFFSETS = {"sum": 1000, "mean": 1100}
_GSPMD_PORT_OFFSETS = {"sum": 2000, "mean": 2100}


class TestGSPMDMultiDevice(DestroyPGMixin, DTensorTestBase):
    """Two-GPU tests for GSPMD and shard_map paths via @compile.

    Each test method runs in separate subprocesses (via DTensorTestBase)
    because torchax registers the "jax" device during XLA compilation,
    which is incompatible with make_fx tracing (backward) in the same process.
    """

    @property
    def world_size(self):
        return 2

    @parameterized.expand([("sum",), ("mean",)])
    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_2gpu_shard_map(self, loss_reduction):
        """Test @compile with DataParallel + XLA_SHARD_MAP vs DDP baseline."""
        model_xla, opt_xla = _create_model_and_warmup(self.rank)
        inputs = _generate_inputs(self.rank)
        ddp_params_per_step = _run_ddp_baseline(
            self.rank, inputs, loss_reduction=loss_reduction
        )
        reduce_fn = torch.sum if loss_reduction == "sum" else torch.mean
        port_offset = _SHARD_MAP_PORT_OFFSETS[loss_reduction]
        with jax_distributed(port_offset=port_offset):

            @compile(
                parallel_mode=DataParallel(
                    "replicate",
                    loss_reduction=loss_reduction,
                    backend=BackendType.XLA_SHARD_MAP,
                ),
                backend=XLAShardMapBackend(dump_hlo=True),
            )
            def train_step_shard_map(mod, opt, x):
                reduce_fn(mod(x)).backward()
                opt.step()

            for step in range(NUM_ITERS):
                train_step_shard_map(model_xla, opt_xla, inputs[step])

                for (name, param), (_, ddp_param) in zip(
                    model_xla.named_parameters(),
                    ddp_params_per_step[step].items(),
                ):
                    torch.testing.assert_close(
                        param,
                        ddp_param,
                        msg=f"Step {step}: XLA_SHARD_MAP vs DDP mismatch for {name}",
                    )

            compiled_obj_sm = train_step_shard_map.__dict__["_compiled_obj"]
            assert_no_graph_breaks(self, compiled_obj_sm)
            assert_hlo_has_collective(
                self, compiled_obj_sm, "all_reduce", "XLA_SHARD_MAP"
            )

    @parameterized.expand([("sum",), ("mean",)])
    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_2gpu_gspmd(self, loss_reduction):
        """Test @compile with GSPMDParallel + XLA_GSPMD vs DDP baseline."""
        model_xla, opt_xla = _create_model_and_warmup(self.rank)
        inputs = _generate_inputs(self.rank)
        ddp_params_per_step = _run_ddp_baseline(
            self.rank, inputs, loss_reduction=loss_reduction
        )
        reduce_fn = torch.sum if loss_reduction == "sum" else torch.mean
        port_offset = _GSPMD_PORT_OFFSETS[loss_reduction]
        with jax_distributed(port_offset=port_offset):

            @compile(
                parallel_mode=GSPMDParallel("replicate", loss_reduction=loss_reduction),
                backend=XLAGSPMDBackend(dump_hlo=True),
            )
            def train_step_gspmd(mod, opt, x):
                reduce_fn(mod(x)).backward()
                opt.step()

            for step in range(NUM_ITERS):
                train_step_gspmd(model_xla, opt_xla, inputs[step])

                for (name, param), (_, ddp_param) in zip(
                    model_xla.named_parameters(),
                    ddp_params_per_step[step].items(),
                ):
                    torch.testing.assert_close(
                        param,
                        ddp_param,
                        msg=f"Step {step}: XLA_GSPMD vs DDP mismatch for {name}",
                    )

            compiled_obj_gspmd = train_step_gspmd.__dict__["_compiled_obj"]
            assert_hlo_has_collective(
                self, compiled_obj_gspmd, "all-reduce", "XLA_GSPMD"
            )


if __name__ == "__main__":
    run_tests()
