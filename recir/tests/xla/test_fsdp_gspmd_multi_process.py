# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors

"""Multi-process FSDP tests for GSPMD.
Tests FSDP (fully_shard) with GSPMDParallel in a true multi-process
environment. Requires 4 GPUs (2 processes x 2 GPUs each).
"""

import socket

import torch
import torch.distributed as dist
from recir.api import compile
from recir.backends import XLAGSPMDBackend
from recir.distributed.parallel_mode import GSPMDParallel
from recir.lowering.xla_lower import _XLAModule
from recir.tests.xla.gspmd_test_utils import (
    assert_jax_multi_device,
    DestroyPGMixin,
    DIM,
    LOCAL_BS,
    SimpleMLP,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


def _shared_jax_coordinator_port(rank: int) -> int:
    coordinator_port: list[object] = [None]
    if rank == 0:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            coordinator_port[0] = listener.getsockname()[1]
    dist.broadcast_object_list(
        coordinator_port,
        src=0,
        device=torch.device(f"cuda:{rank}"),
    )
    port = coordinator_port[0]
    if not isinstance(port, int):
        raise RuntimeError(f"Invalid JAX coordinator port: {port!r}")
    return port


class TestMultiProcessFSDP(DestroyPGMixin, DTensorTestBase):
    @property
    def world_size(self):
        return 2

    @skip_if_lt_x_gpu(4)
    @with_comms
    def test_gspmd_fsdp(self):
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
        dim = DIM
        if dim % num_ranks != 0:
            dim = num_ranks * (dim // num_ranks + 1)

        local_bs = LOCAL_BS * devices_per_process
        torch.manual_seed(200)
        global_input = torch.randn(LOCAL_BS * num_ranks, dim)
        local_input = global_input[self.rank * local_bs : (self.rank + 1) * local_bs]

        coordinator_port = _shared_jax_coordinator_port(self.rank)
        jax.distributed.initialize(
            coordinator_address=f"127.0.0.1:{coordinator_port}",
            num_processes=self.world_size,
            process_id=self.rank,
            local_device_ids=local_ids,
        )
        try:
            assert_jax_multi_device(
                self, total_gpus, devices_per_process, self.world_size
            )
            # ---- Part 1: Memory reduction + HLO collectives ----
            torch.manual_seed(42)
            model = SimpleMLP(dim=dim).cpu()
            opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
            torch.manual_seed(42)
            reference = SimpleMLP(dim=dim).cpu()
            reference_opt = torch.optim.SGD(
                reference.parameters(), lr=0.01, foreach=False
            )
            (reference(global_input).sum() / num_ranks).backward()
            reference_opt.step()
            reference_params = dict(reference.named_parameters())

            @compile(
                parallel_mode=GSPMDParallel(
                    "fully_shard",
                    loss_reduction="sum",
                    num_ranks=num_ranks,
                    num_processes=self.world_size,
                ),
                backend=XLAGSPMDBackend(dump_hlo=True),
            )
            def train_step(mod, opt, x):
                torch.sum(mod(x)).backward()
                opt.step()

            train_step(model, opt, local_input)

            for name, parameter in model.named_parameters():
                expected_local_shard = reference_params[name].chunk(
                    self.world_size, dim=0
                )[self.rank]
                torch.testing.assert_close(
                    parameter,
                    expected_local_shard,
                    atol=1e-5,
                    rtol=1e-5,
                )

            compiled_obj = train_step.__dict__["_compiled_obj"]
            fsdp_size = sum(
                t.numel()
                for t in compiled_obj.flat_state
                if isinstance(t, torch.Tensor)
            )

            # Memory: FSDP flat_state should be smaller than unsharded params
            full_params = 2 * (dim * dim + dim)
            self.assertLess(
                fsdp_size,
                full_params,
                f"FSDP flat_state ({fsdp_size}) should be < "
                f"full params ({full_params})",
            )

            # HLO: should contain FSDP collectives
            hlo_texts = []
            for mod in compiled_obj.gm.modules():
                if isinstance(mod, _XLAModule) and mod.xla_hlo_text:
                    hlo_texts.append(mod.xla_hlo_text)
            combined_hlo = "\n".join(hlo_texts)

            self.assertTrue(
                "all-gather" in combined_hlo,
                "FSDP HLO should contain all-gather.\n" + combined_hlo[:2000],
            )
            self.assertTrue(
                "reduce-scatter" in combined_hlo,
                "FSDP HLO should contain reduce-scatter.\n" + combined_hlo[:2000],
            )

            # ---- Part 2: Multi-step with enable_device_state=True ----
            torch.manual_seed(42)
            model2 = SimpleMLP(dim=dim).cpu()
            opt2 = torch.optim.SGD(model2.parameters(), lr=0.01, foreach=False)

            @compile(
                parallel_mode=GSPMDParallel(
                    "fully_shard",
                    loss_reduction="sum",
                    num_ranks=num_ranks,
                    num_processes=self.world_size,
                ),
                backend=XLAGSPMDBackend(dump_hlo=True),
            )
            def train_step_ds(mod, opt, x):
                torch.sum(mod(x)).backward()
                opt.step()

            for _ in range(3):
                train_step_ds(model2, opt2, local_input)

            ds_obj = train_step_ds.__dict__["_compiled_obj"]
            self.assertIsNotNone(ds_obj)
            has_fsdp = any(
                n.meta.get("is_fsdp_sharded", False)
                for n in ds_obj.gm.graph.nodes
                if n.op == "placeholder"
            )
            self.assertTrue(has_fsdp, "Expected is_fsdp_sharded metadata")
            # HLO: should contain FSDP collectives
            hlo_texts = []
            for mod in compiled_obj.gm.modules():
                if isinstance(mod, _XLAModule) and mod.xla_hlo_text:
                    hlo_texts.append(mod.xla_hlo_text)
            combined_hlo = "\n".join(hlo_texts)

            self.assertTrue(
                "all-gather" in combined_hlo,
                "FSDP HLO should contain all-gather.\n" + combined_hlo[:2000],
            )
            self.assertTrue(
                "reduce-scatter" in combined_hlo,
                "FSDP HLO should contain reduce-scatter.\n" + combined_hlo[:2000],
            )
        finally:
            jax.distributed.shutdown()


if __name__ == "__main__":
    run_tests()
