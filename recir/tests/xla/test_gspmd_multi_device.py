# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors

"""Tests for multi-process, multi-device-per-process XLA compilation.

Uses DTensorTestBase (multi-process) but each process owns N/P devices
instead of the usual 1 device per process. This tests the general case
where num_ranks > num_processes.

E.g., 2 processes on a 4-GPU machine: each process owns 2 GPUs.
num_ranks=4, num_processes=2, devices_per_process=2.

Test matrix row: num_processes=P (>1), num_ranks=N (>P).
"""

import os

import torch
from parameterized import parameterized
from recir.api import compile
from recir.backends import XLAGSPMDBackend, XLAShardMapBackend
from recir.backends.types import BackendType
from recir.distributed.parallel_mode import DataParallel, GSPMDParallel
from recir.tests.xla.gspmd_test_utils import (
    _run_single_rank_baseline,
    assert_hlo_has_collective,
    assert_jax_multi_device,
    assert_no_graph_breaks,
    DestroyPGMixin,
    DIM,
    LOCAL_BS,
    NUM_ITERS,
    SimpleMLP,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

# Port offsets per (test, reduction) to avoid JAX coordinator collisions
# between parameterized variants running in separate subprocesses.
_GSPMD_PORT_OFFSETS = {"sum": 3000, "mean": 3100}
_SHARD_MAP_PORT_OFFSETS = {"sum": 4000, "mean": 4100}


class TestMultiProcessMultiDevice(DestroyPGMixin, DTensorTestBase):
    """Multi-process tests where each process handles multiple JAX devices.

    E.g., 2 processes on a 4-GPU machine: each process owns 2 GPUs.
    num_ranks=4, num_processes=2, devices_per_process=2.
    """

    @property
    def world_size(self):
        return 2  # 2 processes

    @parameterized.expand([("sum",), ("mean",)])
    @skip_if_lt_x_gpu(4)
    @with_comms
    def test_gspmd_multi_device_per_process(self, loss_reduction):
        """GSPMDParallel with num_ranks=4, num_processes=2."""
        import jax  # @manual

        total_gpus = torch.cuda.device_count()
        devices_per_process = total_gpus // self.world_size
        local_ids = list(
            range(
                self.rank * devices_per_process,
                (self.rank + 1) * devices_per_process,
            )
        )
        num_ranks = total_gpus

        # Deterministic global input: every process reconstructs the same
        # full batch, then slices out its local portion.
        torch.manual_seed(200)
        global_input = torch.randn(LOCAL_BS * num_ranks, DIM)
        local_bs = LOCAL_BS * devices_per_process
        local_input = global_input[
            self.rank * local_bs : (self.rank + 1) * local_bs
        ].cuda(self.rank)
        inputs = [local_input.clone() for _ in range(NUM_ITERS)]
        baseline_params = _run_single_rank_baseline(
            global_input, num_ranks, loss_reduction=loss_reduction
        )

        # Model WITHOUT warmup — matches _run_single_rank_baseline's seed-42 init.
        torch.manual_seed(42)
        model_xla = SimpleMLP().cuda(self.rank)
        opt_xla = torch.optim.SGD(model_xla.parameters(), lr=0.01, foreach=False)
        reduce_fn = torch.sum if loss_reduction == "sum" else torch.mean

        # Initialize JAX with multiple devices per process
        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        master_port = int(os.environ.get("MASTER_PORT", "29500"))
        port_offset = _GSPMD_PORT_OFFSETS[loss_reduction]
        jax.distributed.initialize(
            coordinator_address=f"{master_addr}:{master_port + port_offset}",
            num_processes=self.world_size,
            process_id=self.rank,
            local_device_ids=local_ids,
        )
        try:
            assert_jax_multi_device(
                self, total_gpus, devices_per_process, self.world_size
            )

            @compile(
                parallel_mode=GSPMDParallel(
                    "replicate",
                    loss_reduction=loss_reduction,
                    num_ranks=num_ranks,
                    num_processes=self.world_size,
                ),
                backend=XLAGSPMDBackend(dump_hlo=True),
            )
            def train_step(mod, opt, x):
                reduce_fn(mod(x)).backward()
                opt.step()

            for step in range(NUM_ITERS):
                train_step(model_xla, opt_xla, inputs[step])

                for (name, param), (_, baseline_param) in zip(
                    model_xla.named_parameters(),
                    baseline_params[step].items(),
                ):
                    torch.testing.assert_close(
                        param.cpu(),
                        baseline_param,
                        msg=f"Step {step}: XLA_GSPMD vs baseline mismatch for {name}",
                    )

            compiled_obj = train_step.__dict__["_compiled_obj"]
            assert_hlo_has_collective(self, compiled_obj, "all-reduce", "XLA_GSPMD")
        finally:
            jax.distributed.shutdown()

    @parameterized.expand([("sum",), ("mean",)])
    @skip_if_lt_x_gpu(4)
    @with_comms
    def test_shard_map_multi_device_per_process(self, loss_reduction):
        """DataParallel + XLA_SHARD_MAP with num_ranks=4, num_processes=2."""
        import jax  # @manual

        total_gpus = torch.cuda.device_count()
        devices_per_process = total_gpus // self.world_size
        local_ids = list(
            range(
                self.rank * devices_per_process,
                (self.rank + 1) * devices_per_process,
            )
        )
        num_ranks = total_gpus

        # Deterministic global input: every process reconstructs the same
        # full batch, then slices out its local portion.
        torch.manual_seed(200)
        global_input = torch.randn(LOCAL_BS * num_ranks, DIM)
        local_bs = LOCAL_BS * devices_per_process
        local_input = global_input[
            self.rank * local_bs : (self.rank + 1) * local_bs
        ].cuda(self.rank)
        inputs = [local_input.clone() for _ in range(NUM_ITERS)]
        baseline_params = _run_single_rank_baseline(
            global_input, num_ranks, loss_reduction=loss_reduction
        )

        # Model WITHOUT warmup — matches _run_single_rank_baseline's seed-42 init.
        torch.manual_seed(42)
        model_xla = SimpleMLP().cuda(self.rank)
        opt_xla = torch.optim.SGD(model_xla.parameters(), lr=0.01, foreach=False)
        reduce_fn = torch.sum if loss_reduction == "sum" else torch.mean

        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        master_port = int(os.environ.get("MASTER_PORT", "29500"))
        port_offset = _SHARD_MAP_PORT_OFFSETS[loss_reduction]
        jax.distributed.initialize(
            coordinator_address=f"{master_addr}:{master_port + port_offset}",
            num_processes=self.world_size,
            process_id=self.rank,
            local_device_ids=local_ids,
        )
        try:
            assert_jax_multi_device(
                self, total_gpus, devices_per_process, self.world_size
            )

            @compile(
                parallel_mode=DataParallel(
                    "replicate",
                    loss_reduction=loss_reduction,
                    num_ranks=num_ranks,
                    num_processes=self.world_size,
                    backend=BackendType.XLA_SHARD_MAP,
                ),
                backend=XLAShardMapBackend(dump_hlo=True),
            )
            def train_step(mod, opt, x):
                reduce_fn(mod(x)).backward()
                opt.step()

            for step in range(NUM_ITERS):
                train_step(model_xla, opt_xla, inputs[step])

                for (name, param), (_, baseline_param) in zip(
                    model_xla.named_parameters(),
                    baseline_params[step].items(),
                ):
                    torch.testing.assert_close(
                        param.cpu(),
                        baseline_param,
                        msg=f"Step {step}: XLA_SHARD_MAP vs baseline mismatch for {name}",
                    )

            compiled_obj = train_step.__dict__["_compiled_obj"]
            assert_no_graph_breaks(self, compiled_obj)
            assert_hlo_has_collective(self, compiled_obj, "all_reduce", "XLA_SHARD_MAP")
        finally:
            jax.distributed.shutdown()


if __name__ == "__main__":
    run_tests()
