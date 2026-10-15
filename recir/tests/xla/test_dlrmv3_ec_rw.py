#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for DLRMv3 with DMP EmbeddingCollection row-wise sharding on CPU
combined with XLA GSPMD compilation for the dense path.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import jax  # @manual
import recir.lowering.fbgemm_fallback  # noqa: F401 -- registers fbgemm decompositions
import torch
import torch.distributed as dist
import torch.nn as nn
from recir.api import sync_device_state_to_torch
from recir.backends import XLAGSPMDBackend
from recir.distributed.parallel_mode import GSPMDParallel
from recir.lowering.xla_lower import _XLAModule
from recir.tests.xla.dlrmv3_test_utils import (
    create_dlrmv3_wrapper,
    create_static_batch,
    DlrmV3SparseDispatcher,
    DummyPreprocess,
    LOCAL_BS,
    SMALL_EMBEDDING_DIM,
    STATIC_NUM_CANDIDATES,
    STATIC_UIH_LEN,
)
from recir.tests.xla.gspmd_test_utils import _iter_subgraphs, DestroyPGMixin
from recir.training.compile import compile_step
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torchrec.distributed.embedding import EmbeddingCollectionSharder
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.fbgemm_qcomm_codec import (
    get_qcomm_codecs_registry,
    QCommsConfig,
)
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.types import ShardingEnv, ShardingType
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter

try:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")
except OSError:
    pass

DENSE_LR = 0.01
EMBEDDING_LR = 0.1
_JAX_PORT_OFFSET = 2000


def _flat_batch(batch: tuple, *, global_loss_denom: bool) -> tuple:
    return tuple(batch)


class TestEmbeddingCollectionSharder(EmbeddingCollectionSharder):
    def __init__(
        self,
        sharding_type: str,
        kernel_type: str,
        qcomms_config: QCommsConfig | None = None,
        fused_params: Dict[str, Any] | None = None,
    ) -> None:
        self._sharding_type = sharding_type
        self._kernel_type = kernel_type

        qcomm_codecs_registry = {}
        if qcomms_config is not None:
            qcomm_codecs_registry = get_qcomm_codecs_registry(qcomms_config)

        if fused_params is None:
            fused_params = {}
        if "learning_rate" not in fused_params:
            fused_params["learning_rate"] = 0.1

        super().__init__(
            fused_params=fused_params,
            qcomm_codecs_registry=qcomm_codecs_registry,
        )

    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [self._sharding_type]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [self._kernel_type]


class TestDLRMv3ECRWTest(DestroyPGMixin, DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @property
    def backend(self) -> str:
        return "cpu:gloo,cuda:nccl"

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_dlrmv3_ec_rw_with_xla_dense(self) -> None:
        """DMP EC RW on CPU + RecIR XLA GSPMD dense on GPU, numerical parity
        with eager baseline."""
        rank: int = self.rank
        world_size = self.world_size
        pg = dist.distributed_c10d._get_default_group()
        # --- Initialize JAX distributed alongside gloo ---
        total_gpus = torch.cuda.device_count()
        local_ids = [rank % total_gpus]
        num_ranks = world_size
        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        master_port = int(os.environ.get("MASTER_PORT", "29500"))
        jax.distributed.initialize(
            coordinator_address=f"{master_addr}:{master_port + _JAX_PORT_OFFSET}",
            num_processes=world_size,
            process_id=rank,
            local_device_ids=local_ids,
        )
        try:
            global_bs = LOCAL_BS * num_ranks
            num_iters = 3
            # --- Assert JAX devices are GPUs ---
            jax_devs = jax.local_devices()
            self.assertGreater(len(jax_devs), 0, f"Rank {rank}: No JAX devices")
            self.assertEqual(
                jax_devs[0].platform,
                "gpu",
                f"Rank {rank}: Expected GPU, got '{jax_devs[0].platform}'",
            )
            sharders = [
                TestEmbeddingCollectionSharder(
                    sharding_type=ShardingType.ROW_WISE.value,
                    kernel_type=EmbeddingComputeKernel.FUSED.value,
                    fused_params={"learning_rate": EMBEDDING_LR},
                )
            ]
            planner = EmbeddingShardingPlanner(
                topology=Topology(world_size=world_size, compute_device="cpu"),
            )
            # Baseline: DMP EC (CPU) + DDP allreduce, eager
            torch.manual_seed(0)
            baseline_wrapper, hstu_config, _, emb_keys = create_dlrmv3_wrapper(
                global_bs
            )
            baseline_wrapper = baseline_wrapper.cpu()
            # pyrefly: ignore [bad-argument-type]
            plan = planner.collective_plan(baseline_wrapper, sharders, pg)
            # Assert sharding plan uses ROW_WISE for all EC tables
            for module_path, module_plan in plan.plan.items():
                # pyrefly: ignore [missing-attribute]
                for param_name, param_sharding in module_plan.items():
                    self.assertEqual(
                        param_sharding.sharding_type,
                        ShardingType.ROW_WISE.value,
                        f"Rank {rank}: Expected ROW_WISE for "
                        f"{module_path}.{param_name}, "
                        f"got {param_sharding.sharding_type}",
                    )
            # init_data_parallel=True: DMP shards EC AND wraps dense with DDP.
            baseline_dmp = DistributedModelParallel(
                baseline_wrapper,
                env=ShardingEnv.from_process_group(pg),
                plan=plan,
                # pyre-ignore[6]: Sharder generic variance.
                sharders=sharders,
                device=torch.device("cpu"),
                init_data_parallel=True,
            )
            # CombinedOptimizer: fused embedding optimizer (lr=EMBEDDING_LR)
            # + SGD for dense params (lr=DENSE_LR).
            baseline_dense_opt = KeyedOptimizerWrapper(
                dict(in_backward_optimizer_filter(baseline_dmp.named_parameters())),
                lambda params: torch.optim.SGD(params, lr=DENSE_LR, foreach=False),
            )
            baseline_opt = CombinedOptimizer(
                [baseline_dmp.fused_optimizer, baseline_dense_opt]
            )
            local_batch = create_static_batch(hstu_config, LOCAL_BS, seed=42 + rank)

            # Snapshot initial embedding weights to verify they change.
            def _is_embedding_param(name: str) -> bool:
                return "_embedding_collection." in name

            initial_emb_params = {
                n: p.detach().clone()
                for n, p in baseline_dmp.module.named_parameters()
                if _is_embedding_param(n)
            }
            baseline_dmp.train(True)
            baseline_losses: List[float] = []
            for _ in range(num_iters):
                baseline_opt.zero_grad()
                aux_losses: Dict[str, torch.Tensor] = baseline_dmp(*local_batch)
                loss: torch.Tensor = torch.stack(list(aux_losses.values())).sum()
                self.assertTrue(
                    torch.isfinite(loss),
                    f"Rank {rank}: Baseline loss not finite: {loss}",
                )
                self.assertNotEqual(
                    loss.item(), 0.0, f"Rank {rank}: Baseline loss is zero"
                )
                baseline_losses.append(loss.item())
                loss.backward()
                baseline_opt.step()
            self.assertNotEqual(
                baseline_losses[0],
                baseline_losses[-1],
                f"Rank {rank}: Baseline loss unchanged: {baseline_losses}",
            )
            # Snapshot baseline params (use dmp.module to avoid DMP prefix).
            baseline_params = {
                n: p.detach().clone() for n, p in baseline_dmp.module.named_parameters()
            }
            # Test: DMP EC (CPU) + RecIR XLA GSPMD (GPU)
            torch.manual_seed(0)
            test_wrapper, _, _, _ = create_dlrmv3_wrapper(global_bs)
            test_wrapper = test_wrapper.cpu()
            # pyrefly: ignore [bad-argument-type]
            test_plan = planner.collective_plan(test_wrapper, sharders, pg)
            test_dmp = DistributedModelParallel(
                test_wrapper,
                env=ShardingEnv.from_process_group(pg),
                plan=test_plan,
                # pyre-ignore[6]: Sharder generic variance.
                sharders=sharders,
                device=torch.device("cpu"),
                init_data_parallel=False,
            )
            # Dense-only SGD for @compile. The fused embedding optimizer
            # fires automatically during backward via the TBE kernel +
            # cpu_preprocess_backward.  No CombinedOptimizer needed.
            test_dense_opt = KeyedOptimizerWrapper(
                dict(in_backward_optimizer_filter(test_dmp.named_parameters())),
                lambda params: torch.optim.SGD(params, lr=DENSE_LR, foreach=False),
            )
            dummy = DummyPreprocess(
                emb_keys=emb_keys,
                batch_size=global_bs,
                embedding_dim=SMALL_EMBEDDING_DIM,
                static_uih_len=STATIC_UIH_LEN,
                static_num_candidates=STATIC_NUM_CANDIDATES,
                user_emb_feature_names=hstu_config.user_embedding_feature_names,
                item_emb_feature_names=hstu_config.item_embedding_feature_names,
                hstu_configs=hstu_config,
            )
            sparse_dispatcher = DlrmV3SparseDispatcher(dummy)
            sparse_dispatcher._num_ranks = num_ranks

            def train_step_fn(
                mod: nn.Module,
                opt: torch.optim.Optimizer,
                uih_values: torch.Tensor,
                uih_lengths: torch.Tensor,
                cand_values: torch.Tensor,
                cand_lengths: torch.Tensor,
            ) -> None:
                aux_losses: Dict[str, torch.Tensor] = mod(
                    uih_values, uih_lengths, cand_values, cand_lengths
                )
                loss: torch.Tensor = torch.stack(list(aux_losses.values())).sum()
                loss.backward()
                opt.step()

            train_step = compile_step(
                train_step_fn,
                test_dmp.module,
                (test_dense_opt,),
                local_batch,
                dispatchers=(sparse_dispatcher,),
                parallel_mode=GSPMDParallel(
                    "replicate",
                    loss_reduction="sum",
                    num_ranks=num_ranks,
                    num_processes=world_size,
                ),
                backend=XLAGSPMDBackend(dump_hlo=True),
                flatten_batch=_flat_batch,
            )

            for _ in range(num_iters):
                train_step(test_dmp.module, test_dense_opt, *local_batch)
                sync_device_state_to_torch(train_step)
            compiled_obj = train_step.__dict__["_compiled_obj"]
            self.assertIsNotNone(compiled_obj, f"Rank {rank}: XLA compilation failed")
            # --- Assert expected graph structure ---
            # Post-NC12 (uses_iter_graph=False default), the gm is a plain
            # fx.GraphModule rather than an IterGraphModule wrapper.  Use
            # the shared helper to iterate either shape.
            for gm_name, gm in _iter_subgraphs(compiled_obj.gm):
                xla_modules = [m for m in gm.modules() if isinstance(m, _XLAModule)]
                self.assertGreaterEqual(
                    len(xla_modules),
                    1,
                    f"Rank {rank}: {gm_name}: expected >= 1 XLA subgraph, "
                    f"got {len(xla_modules)}",
                )
            # --- Assert HLO contains all-reduce ---
            found_allreduce = False
            for mod in compiled_obj.gm.modules():
                if isinstance(mod, _XLAModule) and mod.xla_hlo_text:
                    if "all-reduce" in mod.xla_hlo_text:
                        found_allreduce = True
                        break
            self.assertTrue(found_allreduce, f"Rank {rank}: HLO missing all-reduce")

            # Numerical parity: compare all parameters
            def _normalize(name: str) -> str:
                for pfx in ("preprocess_mod.model.", "dense_mod.model."):
                    if name.startswith(pfx):
                        return name[len(pfx) :]
                return name

            normalized_baseline = {_normalize(n): p for n, p in baseline_params.items()}
            for name, param in test_dmp.module.named_parameters():
                norm = _normalize(name)
                bp = normalized_baseline.get(norm)
                if bp is not None:
                    torch.testing.assert_close(
                        param,
                        bp,
                        atol=1e-2,
                        rtol=1e-2,
                    )
            # Assert embedding weights actually changed.
            emb_changed = 0
            for name, init_val in initial_emb_params.items():
                current_val = dict(baseline_dmp.module.named_parameters())[name]
                if not torch.equal(init_val, current_val):
                    emb_changed += 1
            self.assertGreater(
                emb_changed,
                0,
                f"Rank {rank}: Embedding weights did not change after "
                f"{num_iters} training steps ({len(initial_emb_params)} "
                f"embedding params checked)",
            )

        finally:
            jax.distributed.shutdown()


if __name__ == "__main__":
    run_tests()
