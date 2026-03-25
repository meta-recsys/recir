# pyre-strict

import unittest
import unittest.mock
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Callable, cast
from unittest.mock import MagicMock

import torch
import torch.fx as fx
import torch.nn as nn
from parameterized import parameterized
from recir.api import compile
from recir.distributed.data_parallel import (
    _clone_param_chains_for_backward,
    _shard_optimizer_state,
    build_data_parallel_strategies,
    DataParallelStrategy,
    DataParallelStyle,
    NodeType,
    partition_value,
    shard_params_by_plan,
)
from recir.distributed.parallel_mode import DataParallel
from recir.gm_transformation import GraphModuleTransformation
from recir.sharding_plan import EmbeddingShardingPlan, EmbeddingShardingType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DeviceMesh
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests  # noqa: TCH001
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


class SimpleMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net1 = nn.Linear(50, 32)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(32, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net2(self.relu(self.net1(x))))

    def reset_parameters(self, *args: Any, **kwargs: Any) -> None:
        self.net1.reset_parameters()
        self.net2.reset_parameters()


class LargeLinearModel(nn.Module):
    """Model with large weights (4MB each) for memory measurement tests."""

    def __init__(self) -> None:
        super().__init__()
        self.net1 = nn.Linear(1024, 1024, bias=False)
        self.net2 = nn.Linear(1024, 1024, bias=False)
        self.net3 = nn.Linear(1024, 1024, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net3(torch.relu(self.net2(torch.relu(self.net1(x)))))

    def reset_parameters(self, *args: Any, **kwargs: Any) -> None:
        self.net1.reset_parameters()
        self.net2.reset_parameters()
        self.net3.reset_parameters()


def make_train_step(reduction: str = "sum") -> Callable[..., torch.Tensor]:
    def train_step(
        model: nn.Module,
        optim: torch.optim.Optimizer,
        train_batch: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        def loss_fn(out: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            return (out - labels).sum() if reduction == "sum" else (out - labels).mean()

        optim.zero_grad()
        inputs, labels = train_batch

        out = model(inputs)
        loss = loss_fn(out, labels)
        loss.backward()
        optim.step()
        return loss

    return train_step


train_step: Callable[..., torch.Tensor] = make_train_step("sum")


class TestDataParallel(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    def _test_ddp(
        self,
        ddp_mod: nn.Module,
        ref_mod: nn.Module,
        ddp_opt: torch.optim.Optimizer,
        ref_opt: torch.optim.Optimizer,
        inp: tuple[torch.Tensor, torch.Tensor],
        train_step: Callable[..., Any],
        data_parallel_options: dict[str, Any] | None = None,
        reduction: str = "sum",
    ) -> None:
        ref_inp = deepcopy(inp)

        # need one step to warm up optimizers
        train_step(ddp_mod, ddp_opt, inp)
        ddp_opt.zero_grad()
        # DDP run full train step once to align with the warmup
        train_step(ref_mod, ref_opt, ref_inp)
        ref_opt.zero_grad()

        # train a DDP model once manually as DDP grads are different
        reduce_fn = torch.sum if reduction == "sum" else torch.mean
        reduce_fn(ref_mod(ref_inp[0]) - ref_inp[1]).backward()
        ref_opt.step()

        # compile it with replicate and run step once
        data_parallel_options = data_parallel_options or {}
        compiled_fn = compile(
            parallel_mode=DataParallel(
                "replicate", loss_reduction=reduction, **data_parallel_options
            )
        )(train_step)
        compiled_fn(ddp_mod, ddp_opt, inp)

        for p1, p2 in zip(ddp_mod.parameters(), ref_mod.parameters()):
            self.assertEqual(p1, p2)

    def _test_fsdp(
        self,
        fsdp_mod: nn.Module,
        ref_mod: nn.Module,
        fsdp_opt: torch.optim.Optimizer,
        ref_opt: torch.optim.Optimizer,
        inp: tuple[torch.Tensor, torch.Tensor],
        train_step: Callable[..., Any],
        data_parallel_options: dict[str, Any] | None = None,
        reduction: str = "sum",
    ) -> None:
        ref_inp = deepcopy(inp)

        # need one step to warm up optimizers
        train_step(fsdp_mod, fsdp_opt, inp)
        fsdp_opt.zero_grad()
        # DDP run full train step once to align with the warmup
        train_step(ref_mod, ref_opt, ref_inp)
        ref_opt.zero_grad()

        # train a ref DDP model once manually as DDP grads are different
        reduce_fn = torch.sum if reduction == "sum" else torch.mean
        reduce_fn(ref_mod(ref_inp[0]) - ref_inp[1]).backward()
        ref_opt.step()

        # compile it with fully_shard and run step once
        data_parallel_options = data_parallel_options or {}
        compiled_fn = compile(
            parallel_mode=DataParallel(
                "fully_shard", loss_reduction=reduction, **data_parallel_options
            )
        )(train_step)
        compiled_fn(fsdp_mod, fsdp_opt, inp)

        # Reconstruct full parameters from fsdp_mod (RecIR per-parameter shards)
        full_p1_list = []
        for p1 in fsdp_mod.parameters():
            shards = [torch.empty_like(p1) for _ in range(self.world_size)]
            torch.distributed.all_gather(shards, p1)
            full_p1_list.append(torch.cat(shards, dim=0))

        with FSDP.summon_full_params(ref_mod):
            for full_p1, p2 in zip(full_p1_list, ref_mod.parameters(), strict=True):
                self.assertEqual(full_p1, p2)

    @parameterized.expand([("sum",), ("mean",)])
    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_replicate_sgd(self, reduction: str) -> None:
        sgd_configs: list[dict[str, Any]] = [
            {"lr": 0.1},
            {"lr": 0.1, "momentum": 0.9},
            {"lr": 0.1, "momentum": 0.9, "foreach": True},
        ]

        for config in sgd_configs:
            ddp_mod = SimpleMLP().cuda(self.rank)
            ddp_opt = torch.optim.SGD(ddp_mod.parameters(), **config)

            train_batch = (
                torch.randn((128, 50), device=torch.device(self.rank)),
                torch.randn((128, 8), device=torch.device(self.rank)),
            )

            ref_mod = DDP(deepcopy(ddp_mod), device_ids=[self.rank])
            ref_opt = torch.optim.SGD(ref_mod.parameters(), **config)

            self._test_ddp(
                ddp_mod,
                ref_mod,
                ddp_opt,
                ref_opt,
                train_batch,
                make_train_step(reduction),
                reduction=reduction,
            )

    @parameterized.expand([("sum",), ("mean",)])
    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_replicate_adam_fused(self, reduction: str) -> None:
        ddp_mod = SimpleMLP().cuda(self.rank)
        ddp_opt = torch.optim.Adam(ddp_mod.parameters(), lr=0.1, fused=True)

        train_batch = (
            torch.randn((128, 50), device=torch.device(self.rank)),
            torch.randn((128, 8), device=torch.device(self.rank)),
        )

        ref_mod = DDP(deepcopy(ddp_mod), device_ids=[self.rank])
        ref_opt = torch.optim.Adam(ref_mod.parameters(), lr=0.1, fused=True)

        self._test_ddp(
            ddp_mod,
            ref_mod,
            ddp_opt,
            ref_opt,
            train_batch,
            make_train_step(reduction),
            reduction=reduction,
        )

    @parameterized.expand([("sum",), ("mean",)])
    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_fully_shard_sgd(self, reduction: str) -> None:
        sgd_configs: list[dict[str, Any]] = [
            {"lr": 0.1},
            {"lr": 0.1, "momentum": 0.9},
            {"lr": 0.1, "momentum": 0.9, "foreach": True},
        ]

        for config in sgd_configs:
            fsdp_mod = SimpleMLP().cuda(self.rank)
            fsdp_opt = torch.optim.SGD(fsdp_mod.parameters(), **config)

            train_batch = (
                torch.randn((128, 50), device=torch.device(self.rank)),
                torch.randn((128, 8), device=torch.device(self.rank)),
            )

            ref_mod = FSDP(deepcopy(fsdp_mod), device_id=self.rank)
            ref_opt = torch.optim.SGD(ref_mod.parameters(), **config)
            self._test_fsdp(
                fsdp_mod,
                ref_mod,
                fsdp_opt,
                ref_opt,
                train_batch,
                make_train_step(reduction),
                reduction=reduction,
            )

    @parameterized.expand([("sum",), ("mean",)])
    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_fully_shard_adam_fused(self, reduction: str) -> None:
        fsdp_mod = SimpleMLP().cuda(self.rank)
        fsdp_opt = torch.optim.Adam(fsdp_mod.parameters(), lr=0.1, fused=True)

        train_batch = (
            torch.randn((128, 50), device=torch.device(self.rank)),
            torch.randn((128, 8), device=torch.device(self.rank)),
        )

        ref_mod = FSDP(deepcopy(fsdp_mod), device_id=self.rank)
        ref_opt = torch.optim.Adam(ref_mod.parameters(), lr=0.1, fused=True)

        self._test_fsdp(
            fsdp_mod,
            ref_mod,
            fsdp_opt,
            ref_opt,
            train_batch,
            make_train_step(reduction),
            reduction=reduction,
        )

    @parameterized.expand([("sum",), ("mean",)])
    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_data_parallel_batch_dim_analysis(self, reduction: str) -> None:
        # test batch dim analysis by adding a few ops that changes
        # the batch dim in non-trival ways

        class WrapperModule(nn.Module):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                self.mlp = SimpleMLP()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                output = self.mlp(x)
                new_output = output.clone().view(-1)
                squeezed_out = new_output.squeeze()
                unsqueezed_out = squeezed_out.unsqueeze(0)
                output = output + 0.1 * unsqueezed_out.view(output.shape[0], -1)

                # test factory ops with data parallel expansion
                arange = torch.arange(output.shape[1], device=output.device)
                ones = torch.ones(output.shape, device=output.device)
                added_arange = arange.unsqueeze(0) + ones

                # test repeat logic
                zeros = torch.zeros(output.shape[1], device=output.device)
                repeated_zeros = zeros.unsqueeze(0).repeat(output.shape[0], 1)

                output = output + added_arange + repeated_zeros

                return output

        step = make_train_step(reduction)
        for parallel_mode in ["replicate", "fully_shard"]:
            mod = WrapperModule().cuda(self.rank)
            opt = torch.optim.SGD(mod.parameters(), lr=0.1)

            train_batch = (
                torch.randn((128, 50), device=torch.device(self.rank)),
                torch.randn((128, 8), device=torch.device(self.rank)),
            )

            ref_mod = (
                FSDP(deepcopy(mod), device_id=self.rank)
                if parallel_mode == "fully_shard"
                else DDP(deepcopy(mod), device_ids=[self.rank])
            )
            ref_opt = torch.optim.SGD(ref_mod.parameters(), lr=0.1)
            test_fn = (
                self._test_fsdp if parallel_mode == "fully_shard" else self._test_ddp
            )
            test_fn(mod, ref_mod, opt, ref_opt, train_batch, step, reduction=reduction)

    @parameterized.expand([("sum",), ("mean",)])
    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_fully_shard_non_0_batch_dim(self, reduction: str) -> None:
        class WrapperModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mlp = SimpleMLP()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                reshaped_x = x.t().contiguous()
                return self.mlp(reshaped_x).t()

        fsdp_mod = WrapperModule().cuda(self.rank)
        fsdp_opt = torch.optim.Adam(fsdp_mod.parameters(), lr=0.1, fused=True)

        train_batch = (
            torch.randn((50, 128), device=torch.device(self.rank)),
            torch.randn((8, 128), device=torch.device(self.rank)),
        )

        ref_mod = FSDP(deepcopy(fsdp_mod), device_id=self.rank)
        ref_opt = torch.optim.Adam(ref_mod.parameters(), lr=0.1, fused=True)

        self._test_fsdp(
            fsdp_mod,
            ref_mod,
            fsdp_opt,
            ref_opt,
            train_batch,
            make_train_step(reduction),
            {"input_batch_dim": 1},
            reduction=reduction,
        )


class TestPerOpFSDP(DTensorTestBase):
    """Tests verifying per-op FSDP graph structure (separate allgathers per layer)."""

    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_fully_shard_separate_allgathers(self) -> None:
        """Verify per-op FSDP inserts separate allgathers for forward and backward."""
        mod = SimpleMLP().cuda(self.rank)
        opt = torch.optim.SGD(mod.parameters(), lr=0.1)

        train_batch = (
            torch.randn((128, 50), device=torch.device(self.rank)),
            torch.randn((128, 8), device=torch.device(self.rank)),
        )

        # Compile to get the partitioned graph
        compiled_fn = compile(parallel_mode=DataParallel("fully_shard"))(train_step)
        compiled_fn(mod, opt, train_batch)

        # Get the compiled graph
        from recir.api import COMPILED_OBJECT_KEY

        compiled_obj = compiled_fn.__dict__[COMPILED_OBJECT_KEY]
        gm = compiled_obj.gm

        # Count allgather and reduce_scatter nodes in the graph
        allgather_count = 0
        reduce_scatter_count = 0
        for node in gm.graph.nodes:
            if node.op == "call_function":
                target_name = str(node.target)
                if "all_gather_into_tensor" in target_name:
                    allgather_count += 1
                elif "reduce_scatter_tensor" in target_name:
                    reduce_scatter_count += 1

        # SimpleMLP has 4 PARAM placeholders (net1.weight, net1.bias,
        # net2.weight, net2.bias). Per-op FSDP allgathers each param in
        # forward (4), plus backward allgathers for params that are consumed
        # by backward compute ops. For SimpleMLP, only net2.weight is needed
        # in backward (to compute net1's input gradient). Biases and
        # net1.weight are not consumed by backward compute ops.
        # Expected: 4 forward + 1 backward = 5 allgathers.
        self.assertEqual(
            allgather_count,
            5,
            f"Expected 5 allgathers (4 forward + 1 backward), got {allgather_count}",
        )

        # There should be reduce_scatter nodes for gradients
        # (one per gradient-producing op, at least one per param).
        self.assertGreater(
            reduce_scatter_count,
            0,
            "Expected at least one reduce_scatter for per-layer gradient reduction",
        )

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_fully_shard_allgather_before_compute_not_before_non_compute(self) -> None:
        """Verify allgather is inserted right before compute ops, not before non-compute ops.

        With shard strategy propagation, non-compute ops like t() should operate
        on the local shard. The allgather should only appear right before the
        compute op (e.g., addmm, mm) that needs the full replicated weight.
        """
        mod = SimpleMLP().cuda(self.rank)
        opt = torch.optim.SGD(mod.parameters(), lr=0.1)

        train_batch = (
            torch.randn((128, 50), device=torch.device(self.rank)),
            torch.randn((128, 8), device=torch.device(self.rank)),
        )

        compiled_fn = compile(parallel_mode=DataParallel("fully_shard"))(train_step)
        compiled_fn(mod, opt, train_batch)

        from recir.api import COMPILED_OBJECT_KEY

        compiled_obj = compiled_fn.__dict__[COMPILED_OBJECT_KEY]
        gm = compiled_obj.gm

        # For each wait_tensor (allgather result), check that its immediate
        # consumers are NOT non-compute ops like t, view, reshape. They should
        # be compute ops (addmm, mm, etc.) or redistribution ops (split, cat).
        aten = torch.ops.aten
        non_compute_targets = {
            aten.t.default,
            aten.clone.default,
            aten.detach.default,
            aten._to_copy.default,
        }
        for node in gm.graph.nodes:
            if (
                node.op == "call_function"
                and node.target == torch.ops._c10d_functional.wait_tensor.default
                and node.meta.get("node_types", []) == [NodeType.PARAM]
            ):
                for user in node.users:
                    self.assertNotIn(
                        user.target,
                        non_compute_targets,
                        f"wait_tensor '{node.name}' feeds directly into "
                        f"non-compute op '{user.target}' — allgather should be "
                        f"deferred to right before the compute op instead.",
                    )

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_fully_shard_numerical_correctness_with_per_op(self) -> None:
        """Verify per-op FSDP still produces numerically correct results."""
        fsdp_mod = SimpleMLP().cuda(self.rank)
        fsdp_opt = torch.optim.SGD(fsdp_mod.parameters(), lr=0.1)

        train_batch = (
            torch.randn((128, 50), device=torch.device(self.rank)),
            torch.randn((128, 8), device=torch.device(self.rank)),
        )

        ref_mod = FSDP(deepcopy(fsdp_mod), device_id=self.rank)
        ref_opt = torch.optim.SGD(ref_mod.parameters(), lr=0.1)

        ref_inp = deepcopy(train_batch)

        # Warmup
        train_step(fsdp_mod, fsdp_opt, train_batch)
        fsdp_opt.zero_grad()
        train_step(ref_mod, ref_opt, ref_inp)
        ref_opt.zero_grad()

        # Train ref manually
        torch.sum(ref_mod(ref_inp[0]) - ref_inp[1]).backward()
        ref_opt.step()

        # Compile with fully_shard (per-op FSDP)
        compiled_fn = compile(parallel_mode=DataParallel("fully_shard"))(train_step)
        compiled_fn(fsdp_mod, fsdp_opt, train_batch)

        # Reconstruct full params and compare
        full_p1_list = []
        for p1 in fsdp_mod.parameters():
            shards = [torch.empty_like(p1) for _ in range(self.world_size)]
            torch.distributed.all_gather(shards, p1)
            full_p1_list.append(torch.cat(shards, dim=0))

        with FSDP.summon_full_params(ref_mod):
            for full_p1, p2 in zip(full_p1_list, ref_mod.parameters(), strict=True):
                self.assertEqual(full_p1, p2)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_fully_shard_weight_free_nodes(self) -> None:
        """Verify per-op FSDP inserts free_storage after each allgathered weight's last consumer."""
        mod = SimpleMLP().cuda(self.rank)
        opt = torch.optim.SGD(mod.parameters(), lr=0.1)

        train_batch = (
            torch.randn((128, 50), device=torch.device(self.rank)),
            torch.randn((128, 8), device=torch.device(self.rank)),
        )

        compiled_fn = compile(
            parallel_mode=DataParallel("fully_shard"),
            gm_transformation=GraphModuleTransformation(  # pyre-ignore[6]
                enable_weight_free=True
            ),
        )(train_step)
        compiled_fn(mod, opt, train_batch)

        from recir.api import COMPILED_OBJECT_KEY

        compiled_obj = compiled_fn.__dict__[COMPILED_OBJECT_KEY]
        gm = compiled_obj.gm

        # Count allgather wait_tensors and free_storage nodes.
        allgather_count = 0
        free_storage_count = 0
        for node in gm.graph.nodes:
            if node.op == "call_function":
                if "all_gather_into_tensor" in str(node.target):
                    allgather_count += 1
                elif node.target == torch.ops.recir.free_storage.default:
                    free_storage_count += 1

        # Each allgather should have a corresponding free_storage.
        self.assertEqual(
            free_storage_count,
            allgather_count,
            f"Expected {allgather_count} free_storage nodes (one per allgather), "
            f"got {free_storage_count}",
        )

        # Verify each free_storage comes after all consumers of its wait_tensor
        # in topological order.
        node_order = {}
        for idx, node in enumerate(gm.graph.nodes):
            node_order[node] = idx

        for node in gm.graph.nodes:
            if (
                node.op == "call_function"
                and node.target == torch.ops.recir.free_storage.default
            ):
                wait_tensor = node.args[0]
                free_order = node_order[node]
                for user in wait_tensor.users:
                    if user is not node:
                        self.assertLess(
                            node_order[user],
                            free_order,
                            f"free_storage at position {free_order} comes before "
                            f"consumer at position {node_order[user]}",
                        )

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_fully_shard_auto_free_without_free_storage(self) -> None:
        """Verify allgathered weights are freed automatically without free_storage.

        After chain cloning, each allgather's wait_tensor feeds a narrow
        chain to one compute consumer. The FX recompiler inserts '= None'
        after last use, which drops all references and frees GPU memory
        via Python refcounting — no explicit free_storage needed.

        This test compares peak memory WITH vs WITHOUT free_storage nodes.
        If peak memory is similar, automatic cleanup is sufficient.
        """
        device = torch.device(self.rank)

        # Use LargeLinearModel so memory differences are measurable.
        # Each weight is 1024x1024 float32 = 4MB.
        mod_with = LargeLinearModel().cuda(self.rank)
        opt_with = torch.optim.SGD(mod_with.parameters(), lr=0.1)
        mod_without = deepcopy(mod_with)
        opt_without = torch.optim.SGD(mod_without.parameters(), lr=0.1)

        batch_with = (
            torch.randn((4, 1024), device=device),
            torch.randn((4, 1024), device=device),
        )
        batch_without = deepcopy(batch_with)

        # --- Run WITH free_storage (compilation with weight_free optimization) ---
        compiled_with = compile(
            parallel_mode=DataParallel("fully_shard"),
            gm_transformation=GraphModuleTransformation(  # pyre-ignore[6]
                enable_weight_free=True
            ),
        )(train_step)
        compiled_with(mod_with, opt_with, batch_with)  # warmup

        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        compiled_with(mod_with, opt_with, batch_with)
        torch.cuda.synchronize(device)
        peak_with = torch.cuda.max_memory_allocated(device)

        # --- Run WITHOUT free_storage ---
        compiled_without = compile(
            parallel_mode=DataParallel("fully_shard"),
            gm_transformation=GraphModuleTransformation(  # pyre-ignore[6]
                enable_weight_free=True
            ),
        )(train_step)
        compiled_without(mod_without, opt_without, batch_without)  # warmup

        # Remove free_storage nodes from the compiled graph
        from recir.api import COMPILED_OBJECT_KEY

        compiled_obj = compiled_without.__dict__[COMPILED_OBJECT_KEY]
        gm = compiled_obj.gm
        free_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target == torch.ops.recir.free_storage.default
        ]
        for n in free_nodes:
            gm.graph.erase_node(n)
        gm.recompile()

        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        compiled_without(mod_without, opt_without, batch_without)
        torch.cuda.synchronize(device)
        peak_without = torch.cuda.max_memory_allocated(device)

        # Peak memory WITHOUT free_storage should be close to WITH.
        # Allow up to 1 full weight (4MB) difference as tolerance —
        # slight timing differences in when refcounting frees storage
        # vs explicit resize_(0) could cause minor variation.
        tolerance = 1024 * 1024 * 4  # 4MB
        self.assertLessEqual(
            peak_without - peak_with,
            tolerance,
            f"Peak memory without free_storage ({peak_without / 1e6:.1f}MB) is "
            f"significantly higher than with free_storage ({peak_with / 1e6:.1f}MB). "
            f"Difference: {(peak_without - peak_with) / 1e6:.1f}MB, "
            f"tolerance: {tolerance / 1e6:.1f}MB. "
            f"This suggests automatic cleanup is NOT freeing allgathered weights.",
        )


class TestBuildDataParallelStrategies(unittest.TestCase):
    """Unit tests for build_data_parallel_strategies edge cases."""

    def _create_mock_mesh(self) -> DeviceMesh:
        """Create a mock DeviceMesh for testing without requiring real distributed setup."""
        mesh = MagicMock(spec=DeviceMesh)
        mesh.ndim = 1
        mesh.shape = (2,)
        mesh.device_type = "cpu"
        mesh.mesh = torch.arange(2)
        mesh._flatten.return_value = mesh
        return mesh

    def test_grad_and_param_inputs(self) -> None:
        """Test that an op with GRAD and PARAM inputs produces a PARAM output."""
        # Graph structure:
        #   param_0 (PARAM placeholder)
        #   act_0   (ACT placeholder)
        #   mm_1 = mm(act_0, param_0)         -> marked GRAD via tag_grad_1
        #   tag_grad_1 = tag_grad(mm_1)       -> erased during backward pass
        #   detach_1 = detach(mm_1)           -> inherits GRAD from mm_1
        #   add = add(detach_1, param_0)      -> GRAD + PARAM inputs
        #   output
        graph = fx.Graph()
        p0 = graph.placeholder("param_0")
        p0.meta["val"] = torch.randn(4, 4)
        act = graph.placeholder("act_0")
        act.meta["val"] = torch.randn(2, 4)
        mm_1 = graph.call_function(torch.ops.aten.mm.default, args=(act, p0))
        mm_1.meta["val"] = torch.randn(2, 4)
        tag_grad_1 = graph.call_function(torch.ops._spmd.tag_grad.default, args=(mm_1,))
        tag_grad_1.meta["val"] = torch.randn(2, 4)
        detach_1 = graph.call_function(
            torch.ops.aten.detach.default, args=(tag_grad_1,)
        )
        detach_1.meta["val"] = torch.randn(2, 4)
        # Op with GRAD + PARAM inputs (e.g., optimizer update step)
        add_node = graph.call_function(torch.ops.aten.add.Tensor, args=(detach_1, p0))
        add_node.meta["val"] = torch.randn(4, 4)
        graph.output(add_node)
        gm = fx.GraphModule(torch.nn.Module(), graph)
        mesh = self._create_mock_mesh()
        strategy_map = build_data_parallel_strategies(
            gm,
            num_params=1,
            num_states=0,
            mesh=mesh,
            batch_dim=0,
            parallel_mode=DataParallelStyle.REPLICATE,
        )

        # GRAD + PARAM -> output should be PARAM
        add_strategy = cast(DataParallelStrategy, strategy_map[add_node])
        self.assertEqual(add_strategy.node_type, NodeType.PARAM)

    def test_state_inputs(self) -> None:
        """Test that an op with STATE inputs (no GRAD) produces the correct output type."""
        # Graph structure:
        #   param_0 (PARAM placeholder)
        #   state_0 (STATE placeholder)
        #   act_0   (ACT placeholder)
        #   add_1 = add(state_0, param_0)     -> STATE + PARAM inputs -> PARAM output
        #   add_2 = add(state_0, state_0)     -> STATE only inputs -> STATE output
        #   add_3 = add(add_1, add_2)         -> PARAM + STATE inputs -> PARAM output
        #   output

        graph = fx.Graph()
        p0 = graph.placeholder("param_0")
        p0.meta["val"] = torch.randn(4, 4)
        s0 = graph.placeholder("state_0")
        s0.meta["val"] = torch.randn(4, 4)
        act = graph.placeholder("act_0")
        act.meta["val"] = torch.randn(2, 4)
        # STATE + PARAM -> PARAM
        add_1 = graph.call_function(torch.ops.aten.add.Tensor, args=(s0, p0))
        add_1.meta["val"] = torch.randn(4, 4)
        # STATE + STATE -> STATE
        add_2 = graph.call_function(torch.ops.aten.add.Tensor, args=(s0, s0))
        add_2.meta["val"] = torch.randn(4, 4)
        # PARAM + STATE -> PARAM
        add_3 = graph.call_function(torch.ops.aten.add.Tensor, args=(add_1, add_2))
        add_3.meta["val"] = torch.randn(4, 4)
        graph.output(add_3)
        gm = fx.GraphModule(torch.nn.Module(), graph)
        mesh = self._create_mock_mesh()
        strategy_map = build_data_parallel_strategies(
            gm,
            num_params=1,
            num_states=1,
            mesh=mesh,
            batch_dim=0,
            parallel_mode=DataParallelStyle.REPLICATE,
        )
        add_1_strategy = cast(DataParallelStrategy, strategy_map[add_1])
        self.assertEqual(add_1_strategy.node_type, NodeType.PARAM)
        add_2_strategy = cast(DataParallelStrategy, strategy_map[add_2])
        self.assertEqual(add_2_strategy.node_type, NodeType.STATE)
        add_3_strategy = cast(DataParallelStrategy, strategy_map[add_3])
        self.assertEqual(add_3_strategy.node_type, NodeType.PARAM)

    def test_param_only_inputs(self) -> None:
        """Test that ops with only PARAM inputs (e.g. weight decay) produce PARAM output.

        AdamW's decoupled weight decay applies param.mul_(1 - lr * wd) which
        creates a node whose only tensor input is PARAM (the scalar is NON_TENSOR).
        This should produce a PARAM output, same as aten.copy_ (final step copy).
        """
        # Graph structure:
        #   param_0 (PARAM placeholder)
        #   act_0   (ACT placeholder)
        #   mul_1 = mul_(param_0)  -> PARAM-only input (weight decay)
        #   copy_1 = copy_(param_0, param_0)  -> PARAM-only inputs (final copy)
        #   output
        graph = fx.Graph()
        p0 = graph.placeholder("param_0")
        p0.meta["val"] = torch.randn(4, 4)
        act = graph.placeholder("act_0")
        act.meta["val"] = torch.randn(2, 4)
        # PARAM-only mul_ (weight decay: param *= scalar)
        mul_node = graph.call_function(torch.ops.aten.mul_.Scalar, args=(p0, 0.99))
        mul_node.meta["val"] = torch.randn(4, 4)
        # PARAM-only copy_ (final parameter write-back)
        copy_node = graph.call_function(
            torch.ops.aten.copy_.default, args=(p0, mul_node)
        )
        copy_node.meta["val"] = torch.randn(4, 4)
        graph.output(copy_node)
        gm = fx.GraphModule(torch.nn.Module(), graph)
        mesh = self._create_mock_mesh()
        strategy_map = build_data_parallel_strategies(
            gm,
            num_params=1,
            num_states=0,
            mesh=mesh,
            batch_dim=0,
            parallel_mode=DataParallelStyle.REPLICATE,
        )
        # Both should be classified as PARAM
        mul_strategy = cast(DataParallelStrategy, strategy_map[mul_node])
        self.assertEqual(mul_strategy.node_type, NodeType.PARAM)
        copy_strategy = cast(DataParallelStrategy, strategy_map[copy_node])
        self.assertEqual(copy_strategy.node_type, NodeType.PARAM)

    def _create_replicate_spec(self) -> DTensorSpec:
        """Create a DTensorSpec with Replicate placement for partition_value tests."""
        mesh = self._create_mock_mesh()
        return DTensorSpec(mesh=mesh, placements=(Replicate(),))

    def test_partition_value_tuple_with_none(self) -> None:
        """partition_value must handle tuples containing None (e.g., HuggingFace outputs).

        When make_fx traces a model returning (logits, None, None, None),
        the partitioner calls partition_value on the full tuple.  Without
        scalar passthrough, this crashes with:
          RuntimeError: val type <class 'NoneType'> not supported
        """
        spec = self._create_replicate_spec()
        tensor = torch.randn(4, 4)
        val = (tensor, None, None, None)
        result = partition_value(val, spec)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)
        self.assertTrue(torch.equal(result[0], tensor))
        self.assertIsNone(result[1])
        self.assertIsNone(result[2])
        self.assertIsNone(result[3])

    def test_partition_value_tuple_with_int(self) -> None:
        """partition_value must handle tuples containing int values.

        Some model outputs include integer metadata (e.g., sequence length,
        layer index) alongside tensors.  Without scalar passthrough, this
        crashes with:
          RuntimeError: val type <class 'int'> not supported
        """
        spec = self._create_replicate_spec()
        tensor = torch.randn(4, 4)
        val = (tensor, 128)
        result = partition_value(val, spec)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertTrue(torch.equal(result[0], tensor))
        self.assertEqual(result[1], 128)

    def test_partition_value_tuple_with_mixed_scalars(self) -> None:
        """partition_value must handle tuples with mixed scalar types."""
        spec = self._create_replicate_spec()
        tensor = torch.randn(4, 4)
        val = (tensor, None, 42, 3.14, True, "metadata")
        result = partition_value(val, spec)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 6)
        self.assertTrue(torch.equal(result[0], tensor))
        self.assertIsNone(result[1])
        self.assertEqual(result[2], 42)
        self.assertAlmostEqual(result[3], 3.14)
        self.assertTrue(result[4])
        self.assertEqual(result[5], "metadata")


class TestCloneParamChains(unittest.TestCase):
    """Unit tests for _clone_param_chains_for_backward."""

    def test_shared_chain_cloned_once(self) -> None:
        """One param with m non-compute ops shared by n compute ops → 1 clone.

        Graph:
            W (PARAM)
              └→ t(W)
                  └→ view(t(W))
                      ├→ mm_fwd (topo earliest) — keeps original chain
                      ├→ mm_bwd_1 — redirected to cloned chain
                      └→ mm_bwd_2 — redirected to cloned chain

        After cloning, mm_bwd_1 and mm_bwd_2 should share ONE cloned chain
        (t' → view'), not two separate clones.
        """
        graph = fx.Graph()

        # PARAM placeholder
        w = graph.placeholder("w")
        w.meta["val"] = torch.randn(8, 4)
        w.meta["sharding"] = "shard_spec_placeholder"

        # Non-compute chain: t → view (m=2 non-compute ops)
        t_node = graph.call_function(torch.ops.aten.t.default, args=(w,))
        t_node.meta["val"] = torch.randn(4, 8)
        t_node.meta["sharding"] = "replicate_spec"

        view_node = graph.call_function(
            torch.ops.aten.view.default, args=(t_node, [4, 8])
        )
        view_node.meta["val"] = torch.randn(4, 8)
        view_node.meta["sharding"] = "replicate_spec"

        # Activation placeholder
        act = graph.placeholder("act_0")
        act.meta["val"] = torch.randn(2, 4)

        # n=3 compute consumers of the same chain end (view_node)
        mm_fwd = graph.call_function(torch.ops.aten.mm.default, args=(act, view_node))
        mm_fwd.meta["val"] = torch.randn(2, 8)

        mm_bwd_1 = graph.call_function(torch.ops.aten.mm.default, args=(act, view_node))
        mm_bwd_1.meta["val"] = torch.randn(2, 8)

        mm_bwd_2 = graph.call_function(torch.ops.aten.mm.default, args=(act, view_node))
        mm_bwd_2.meta["val"] = torch.randn(2, 8)

        graph.output((mm_fwd, mm_bwd_1, mm_bwd_2))

        gm = fx.GraphModule(torch.nn.Module(), graph)

        strategy_map = {w: DataParallelStrategy([NodeType.PARAM], None)}

        # Count nodes before
        nodes_before = len(list(gm.graph.nodes))

        _clone_param_chains_for_backward(gm, strategy_map)  # pyre-ignore[6]

        # The chain has 2 non-compute ops (t, view). Cloning once adds 2 nodes.
        nodes_after = len(list(gm.graph.nodes))
        cloned_count = nodes_after - nodes_before
        self.assertEqual(
            cloned_count,
            2,
            f"Expected 1 clone of 2-node chain (2 new nodes), got {cloned_count}",
        )

        # mm_fwd (earliest) should still use the original view_node
        self.assertIs(mm_fwd.args[1], view_node)

        # mm_bwd_1 and mm_bwd_2 should use the SAME cloned view node
        cloned_view_1 = mm_bwd_1.args[1]
        cloned_view_2 = mm_bwd_2.args[1]
        self.assertIsNot(cloned_view_1, view_node, "bwd should not use original chain")
        self.assertIs(
            cloned_view_1,
            cloned_view_2,
            "All later consumers should share the same cloned chain",
        )

        # The cloned chain should preserve the chain node's own propagated sharding
        self.assertEqual(
            cloned_view_1.meta["sharding"],  # pyre-ignore[16]
            "replicate_spec",
        )

    def test_two_chain_branches_cloned_separately(self) -> None:
        """Two non-compute chains from same param, each with fwd and bwd consumers.

        Graph:
            W (PARAM)
              ├→ t(W) — chain A
              │   ├→ mm_C (fwd, keeps original)
              │   └→ mm_E (bwd, gets clone of chain A)
              └→ view(W) — chain B
                  ├→ mm_D (fwd, keeps original)
                  └→ mm_F (bwd, gets clone of chain B)

        Each branch independently finds its earliest consumer. mm_C keeps
        chain A's original, mm_D keeps chain B's original. mm_E and mm_F
        each get redirected to their branch's cloned chain.
        Expected: 2 new nodes (one t clone, one view clone).
        """
        graph = fx.Graph()

        # PARAM placeholder
        w = graph.placeholder("w")
        w.meta["val"] = torch.randn(8, 4)
        w.meta["sharding"] = "shard_spec_placeholder"

        # Chain A: t(W)
        t_node = graph.call_function(torch.ops.aten.t.default, args=(w,))
        t_node.meta["val"] = torch.randn(4, 8)
        t_node.meta["sharding"] = "replicate_spec"

        # Activation placeholder
        act = graph.placeholder("act_0")
        act.meta["val"] = torch.randn(2, 4)

        # Chain A fwd consumer (topologically earliest overall)
        mm_C = graph.call_function(torch.ops.aten.mm.default, args=(act, t_node))
        mm_C.meta["val"] = torch.randn(2, 8)

        # Chain B: view(W) — placed after mm_C so mm_D is not the global earliest
        view_node = graph.call_function(torch.ops.aten.view.default, args=(w, [4, 8]))
        view_node.meta["val"] = torch.randn(4, 8)
        view_node.meta["sharding"] = "replicate_spec"

        # Chain B fwd consumer
        mm_D = graph.call_function(torch.ops.aten.mm.default, args=(act, view_node))
        mm_D.meta["val"] = torch.randn(2, 8)

        # Chain A bwd consumer
        mm_E = graph.call_function(torch.ops.aten.mm.default, args=(act, t_node))
        mm_E.meta["val"] = torch.randn(2, 8)

        # Chain B bwd consumer
        mm_F = graph.call_function(torch.ops.aten.mm.default, args=(act, view_node))
        mm_F.meta["val"] = torch.randn(2, 8)

        graph.output((mm_C, mm_D, mm_E, mm_F))

        gm = fx.GraphModule(torch.nn.Module(), graph)

        strategy_map = {w: DataParallelStrategy([NodeType.PARAM], None)}

        nodes_before = len(list(gm.graph.nodes))

        _clone_param_chains_for_backward(gm, strategy_map)  # pyre-ignore[6]

        # Two chains cloned
        nodes_after = len(list(gm.graph.nodes))
        cloned_count = nodes_after - nodes_before
        self.assertEqual(
            cloned_count,
            2,
            f"Expected 2 clones (one per chain branch), got {cloned_count}",
        )

        # Fwd consumers keep originals (per-branch earliest)
        self.assertIs(mm_C.args[1], t_node)
        self.assertIs(mm_D.args[1], view_node)

        # Bwd consumers get cloned nodes
        cloned_t = mm_E.args[1]
        cloned_view = mm_F.args[1]
        self.assertIsNot(cloned_t, t_node)
        self.assertIsNot(cloned_view, view_node)

        # The two clones should be different nodes
        self.assertIsNot(cloned_t, cloned_view)

        # Both clones should preserve their chain node's own propagated sharding
        self.assertEqual(cloned_t.meta["sharding"], "replicate_spec")  # pyre-ignore[16]
        self.assertEqual(
            cloned_view.meta["sharding"],  # pyre-ignore[16]
            "replicate_spec",
        )

    def test_no_clone_when_single_consumer(self) -> None:
        """A chain with only one compute consumer should not be cloned."""
        graph = fx.Graph()

        w = graph.placeholder("w")
        w.meta["val"] = torch.randn(8, 4)
        w.meta["sharding"] = "shard_spec"

        t_node = graph.call_function(torch.ops.aten.t.default, args=(w,))
        t_node.meta["val"] = torch.randn(4, 8)
        t_node.meta["sharding"] = "replicate_spec"

        act = graph.placeholder("act_0")
        act.meta["val"] = torch.randn(2, 4)

        mm_only = graph.call_function(torch.ops.aten.mm.default, args=(act, t_node))
        mm_only.meta["val"] = torch.randn(2, 8)

        graph.output(mm_only)

        gm = fx.GraphModule(torch.nn.Module(), graph)

        strategy_map = {w: DataParallelStrategy([NodeType.PARAM], None)}

        nodes_before = len(list(gm.graph.nodes))
        _clone_param_chains_for_backward(gm, strategy_map)  # pyre-ignore[6]
        nodes_after = len(list(gm.graph.nodes))

        self.assertEqual(
            nodes_after,
            nodes_before,
            "No cloning should occur with a single compute consumer",
        )

    def test_linear_chain_cloned_for_deeper_consumer(self) -> None:
        """Linear chain W → t → view, consumers at different depths.

        Graph:
            W (PARAM)
              └→ t(W)
                  ├→ mm_fwd (fwd, earliest in branch)
                  └→ view(t(W))
                      └→ mm_bwd (bwd, later in same branch)

        Both chain nodes are in the same branch (rooted at t). The
        branch-level earliest is mm_fwd. mm_bwd is later, so the path
        W → t → view is cloned (2 new nodes: t', view').
        """
        graph = fx.Graph()

        w = graph.placeholder("w")
        w.meta["val"] = torch.randn(8, 4)
        w.meta["sharding"] = "shard_spec_placeholder"

        # Linear chain: t → view
        t_node = graph.call_function(torch.ops.aten.t.default, args=(w,))
        t_node.meta["val"] = torch.randn(4, 8)
        t_node.meta["sharding"] = "replicate_spec"

        act = graph.placeholder("act_0")
        act.meta["val"] = torch.randn(2, 4)

        # Fwd consumer of t (earliest in branch)
        mm_fwd = graph.call_function(torch.ops.aten.mm.default, args=(act, t_node))
        mm_fwd.meta["val"] = torch.randn(2, 8)

        # Deeper chain node
        view_node = graph.call_function(
            torch.ops.aten.view.default, args=(t_node, [4, 8])
        )
        view_node.meta["val"] = torch.randn(4, 8)
        view_node.meta["sharding"] = "replicate_spec"

        # Bwd consumer of view (later in same branch)
        mm_bwd = graph.call_function(torch.ops.aten.mm.default, args=(act, view_node))
        mm_bwd.meta["val"] = torch.randn(2, 8)

        graph.output((mm_fwd, mm_bwd))

        gm = fx.GraphModule(torch.nn.Module(), graph)

        strategy_map = {w: DataParallelStrategy([NodeType.PARAM], None)}

        nodes_before = len(list(gm.graph.nodes))

        _clone_param_chains_for_backward(gm, strategy_map)  # pyre-ignore[6]

        # Clone path W

        # Clone path W → t → view = 2 new nodes (t', view')
        nodes_after = len(list(gm.graph.nodes))
        cloned_count = nodes_after - nodes_before
        self.assertEqual(
            cloned_count,
            2,
            f"Expected 2 new nodes (t' and view'), got {cloned_count}",
        )

        # Fwd consumer keeps original t
        self.assertIs(mm_fwd.args[1], t_node)

        # Bwd consumer gets cloned view
        cloned_view = mm_bwd.args[1]
        self.assertIsNot(cloned_view, view_node)
        self.assertEqual(
            cloned_view.meta["sharding"],  # pyre-ignore[16]
            "replicate_spec",
        )

        # The cloned view should be fed by a cloned t (not the original)
        cloned_t = cloned_view.args[0]  # pyre-ignore[16]
        self.assertIsNot(cloned_t, t_node)
        self.assertEqual(cloned_t.meta["sharding"], "replicate_spec")

        # The cloned t should still take W as input (shared param)
        self.assertIs(cloned_t.args[0], w)


class EmbeddingMLP(nn.Module):
    """Model with an embedding table and dense layers for hybrid parallel tests."""

    def __init__(self, num_embeddings: int = 16, embedding_dim: int = 8) -> None:
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        self.linear = nn.Linear(embedding_dim, 4)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        emb_out = self.emb(indices)  # (batch, seq, embedding_dim)
        return self.linear(emb_out.mean(dim=1))  # (batch, 4)


class TestShardParamsByPlan(DTensorTestBase):
    """Tests for shard_params_by_plan with embedding and dense params."""

    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_fully_shard_all_params(self) -> None:
        """FULLY_SHARD with no embedding params: all params sharded on dim 0."""
        model = EmbeddingMLP().cuda(self.rank)
        params = OrderedDict(
            [
                ("emb.weight", model.emb.weight.data.clone()),
                ("linear.weight", model.linear.weight.data.clone()),
            ]
        )
        orig_emb_rows = params["emb.weight"].shape[0]  # 16
        orig_lin_rows = params["linear.weight"].shape[0]  # 4
        mesh = DeviceMesh("cuda", torch.arange(self.world_size))
        shard_params_by_plan(
            model, params, {}, mesh, DataParallelStyle.FULLY_SHARD, set()
        )
        self.assertEqual(
            params["emb.weight"].shape[0], orig_emb_rows // self.world_size
        )
        self.assertEqual(
            params["linear.weight"].shape[0], orig_lin_rows // self.world_size
        )

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_replicate_no_sharding(self) -> None:
        """REPLICATE with no embedding params: no params sharded."""
        model = EmbeddingMLP().cuda(self.rank)
        params = OrderedDict(
            [
                ("emb.weight", model.emb.weight.data.clone()),
                ("linear.weight", model.linear.weight.data.clone()),
            ]
        )
        orig_shapes = {k: v.shape for k, v in params.items()}
        mesh = DeviceMesh("cuda", torch.arange(self.world_size))
        shard_params_by_plan(
            model, params, {}, mesh, DataParallelStyle.REPLICATE, set()
        )
        for k in params:
            self.assertEqual(params[k].shape, orig_shapes[k])

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_embedding_sharded_in_replicate_mode(self) -> None:
        """Embedding params are Shard(0) even with REPLICATE dense style."""
        model = EmbeddingMLP().cuda(self.rank)
        params = OrderedDict(
            [
                ("emb.weight", model.emb.weight.data.clone()),
                ("linear.weight", model.linear.weight.data.clone()),
            ]
        )
        mesh = DeviceMesh("cuda", torch.arange(self.world_size))
        shard_params_by_plan(
            model,
            params,
            {},
            mesh,
            DataParallelStyle.REPLICATE,
            {"emb.weight"},
        )
        # Embedding: sharded (16, 8) -> (8, 8)
        self.assertEqual(params["emb.weight"].shape, (8, 8))
        # Linear: NOT sharded (REPLICATE, not embedding)
        self.assertEqual(params["linear.weight"].shape, (4, 8))

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_optimizer_states_sharded_for_embedding(self) -> None:
        """Optimizer states for embedding params are sharded; dense states are not in REPLICATE."""
        model = EmbeddingMLP().cuda(self.rank)
        device = f"cuda:{self.rank}"
        params = OrderedDict(
            [
                ("emb.weight", model.emb.weight.data.clone()),
                ("linear.weight", model.linear.weight.data.clone()),
            ]
        )
        named_states = {
            "emb.weight": {"momentum": torch.randn(16, 8, device=device)},
            "linear.weight": {"momentum": torch.randn(4, 8, device=device)},
        }
        mesh = DeviceMesh("cuda", torch.arange(self.world_size))
        shard_params_by_plan(
            model,
            params,
            named_states,
            mesh,
            DataParallelStyle.REPLICATE,
            {"emb.weight"},
        )
        # Embedding state sharded
        self.assertEqual(named_states["emb.weight"]["momentum"].shape, (8, 8))
        # Dense state NOT sharded in REPLICATE mode
        self.assertEqual(named_states["linear.weight"]["momentum"].shape, (4, 8))

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_model_attribute_updated(self) -> None:
        """After sharding, the model attribute reflects the local shard."""
        model = EmbeddingMLP().cuda(self.rank)
        params = OrderedDict(
            [
                ("emb.weight", model.emb.weight.data.clone()),
                ("linear.weight", model.linear.weight.data.clone()),
            ]
        )
        mesh = DeviceMesh("cuda", torch.arange(self.world_size))
        shard_params_by_plan(
            model,
            params,
            {},
            mesh,
            DataParallelStyle.REPLICATE,
            {"emb.weight"},
        )
        # model.emb.weight should now be the local shard
        self.assertEqual(model.emb.weight.shape, (8, 8))
        # model.linear.weight unchanged in REPLICATE
        self.assertEqual(model.linear.weight.shape, (4, 8))


class TestEmbeddingShardingPlan(unittest.TestCase):
    """Unit tests for EmbeddingShardingPlan."""

    def test_empty_plan(self):
        """Default-constructed plan has empty param_fqns, is_embedding_param returns False."""
        plan = EmbeddingShardingPlan()
        self.assertEqual(plan.param_fqns, set())
        self.assertFalse(plan.is_embedding_param("anything"))
        self.assertFalse(plan.is_row_wise("anything"))

    def test_param_fqns_returns_set_of_keys(self):
        """Plan with 2 entries returns correct set of FQNs."""
        plan = EmbeddingShardingPlan(
            sharding_map={
                "emb1.weight": EmbeddingShardingType.ROW_WISE,
                "emb2.weight": EmbeddingShardingType.ROW_WISE,
            }
        )
        self.assertEqual(plan.param_fqns, {"emb1.weight", "emb2.weight"})

    def test_is_row_wise_true(self):
        """is_row_wise returns True for ROW_WISE entries."""
        plan = EmbeddingShardingPlan(
            sharding_map={
                "emb.weight": EmbeddingShardingType.ROW_WISE,
            }
        )
        self.assertTrue(plan.is_row_wise("emb.weight"))
        self.assertTrue(plan.is_embedding_param("emb.weight"))

    def test_is_row_wise_false_for_missing(self):
        """is_row_wise returns False for FQNs not in the plan."""
        plan = EmbeddingShardingPlan(
            sharding_map={
                "emb.weight": EmbeddingShardingType.ROW_WISE,
            }
        )
        self.assertFalse(plan.is_row_wise("linear.weight"))
        self.assertFalse(plan.is_embedding_param("linear.weight"))


class TestBuildStrategiesWithEmbedding(unittest.TestCase):
    """Unit tests for build_data_parallel_strategies with embedding_param_indices."""

    def _create_mock_mesh(self) -> DeviceMesh:
        mesh = MagicMock(spec=DeviceMesh)
        mesh.ndim = 1
        mesh.shape = (2,)
        mesh.device_type = "cpu"
        mesh.mesh = torch.arange(2)
        mesh._flatten.return_value = mesh
        return mesh

    def test_embedding_param_placeholder_type(self):
        """Placeholder in embedding_param_indices gets EMBEDDING_PARAM type."""
        graph = fx.Graph()
        emb_w = graph.placeholder("emb_weight")
        emb_w.meta["val"] = torch.randn(100, 16)
        act = graph.placeholder("act_0")
        act.meta["val"] = torch.randn(4, 3)
        graph.output(emb_w)
        gm = fx.GraphModule(nn.Module(), graph)
        mesh = self._create_mock_mesh()
        strategy_map = build_data_parallel_strategies(
            gm,
            num_params=1,
            num_states=0,
            mesh=mesh,
            batch_dim=0,
            parallel_mode=DataParallelStyle.FULLY_SHARD,
            embedding_param_indices={0},
        )
        emb_strategy = cast(DataParallelStrategy, strategy_map[emb_w])
        self.assertEqual(emb_strategy.node_type, NodeType.EMBEDDING_PARAM)

    def test_non_embedding_placeholder_stays_param(self):
        """Placeholder NOT in embedding_param_indices stays PARAM."""
        graph = fx.Graph()
        dense_w = graph.placeholder("dense_weight")
        dense_w.meta["val"] = torch.randn(4, 4)
        act = graph.placeholder("act_0")
        act.meta["val"] = torch.randn(4, 4)
        graph.output(dense_w)
        gm = fx.GraphModule(nn.Module(), graph)
        mesh = self._create_mock_mesh()
        strategy_map = build_data_parallel_strategies(
            gm,
            num_params=1,
            num_states=0,
            mesh=mesh,
            batch_dim=0,
            parallel_mode=DataParallelStyle.FULLY_SHARD,
            embedding_param_indices=set(),
        )
        dense_strategy = cast(DataParallelStrategy, strategy_map[dense_w])
        self.assertEqual(dense_strategy.node_type, NodeType.PARAM)

    def test_non_compute_propagation_preserves_embedding_param(self):
        """Non-compute ops (view/reshape) on EMBEDDING_PARAM stay EMBEDDING_PARAM."""
        graph = fx.Graph()
        emb_w = graph.placeholder("emb_weight")
        emb_w.meta["val"] = torch.randn(100, 16)
        act = graph.placeholder("act_0")
        act.meta["val"] = torch.randn(4, 3)
        t_node = graph.call_function(torch.ops.aten.t.default, args=(emb_w,))
        t_node.meta["val"] = torch.randn(16, 100)
        graph.output(t_node)
        gm = fx.GraphModule(nn.Module(), graph)
        mesh = self._create_mock_mesh()
        strategy_map = build_data_parallel_strategies(
            gm,
            num_params=1,
            num_states=0,
            mesh=mesh,
            batch_dim=0,
            parallel_mode=DataParallelStyle.FULLY_SHARD,
            embedding_param_indices={0},
        )
        t_strategy = cast(DataParallelStrategy, strategy_map[t_node])
        self.assertEqual(t_strategy.node_type, NodeType.EMBEDDING_PARAM)

    def test_mixed_param_and_embedding_param(self):
        """PARAM and EMBEDDING_PARAM placeholders are typed correctly."""
        graph = fx.Graph()
        dense_w = graph.placeholder("dense_weight")
        dense_w.meta["val"] = torch.randn(4, 4)
        emb_w = graph.placeholder("emb_weight")
        emb_w.meta["val"] = torch.randn(100, 16)
        act = graph.placeholder("act_0")
        act.meta["val"] = torch.randn(4, 4)
        graph.output((dense_w, emb_w))
        gm = fx.GraphModule(nn.Module(), graph)
        mesh = self._create_mock_mesh()
        strategy_map = build_data_parallel_strategies(
            gm,
            num_params=2,
            num_states=0,
            mesh=mesh,
            batch_dim=0,
            parallel_mode=DataParallelStyle.FULLY_SHARD,
            embedding_param_indices={1},
        )
        dense_strategy = cast(DataParallelStrategy, strategy_map[dense_w])
        self.assertEqual(dense_strategy.node_type, NodeType.PARAM)
        emb_strategy = cast(DataParallelStrategy, strategy_map[emb_w])
        self.assertEqual(emb_strategy.node_type, NodeType.EMBEDDING_PARAM)

    def test_embedding_param_with_replicate_mode(self):
        """EMBEDDING_PARAM gets Shard(0) even when parallel_mode is REPLICATE."""
        graph = fx.Graph()
        emb_w = graph.placeholder("emb_weight")
        emb_w.meta["val"] = torch.randn(100, 16)
        act = graph.placeholder("act_0")
        act.meta["val"] = torch.randn(4, 3)
        graph.output(emb_w)
        gm = fx.GraphModule(nn.Module(), graph)
        mesh = self._create_mock_mesh()
        strategy_map = build_data_parallel_strategies(
            gm,
            num_params=1,
            num_states=0,
            mesh=mesh,
            batch_dim=0,
            parallel_mode=DataParallelStyle.REPLICATE,
            embedding_param_indices={0},
        )
        emb_strategy = cast(DataParallelStrategy, strategy_map[emb_w])
        self.assertEqual(emb_strategy.node_type, NodeType.EMBEDDING_PARAM)
        # Embedding params must be sharded on dim 0 regardless of parallel_mode
        output_specs = emb_strategy.op_spec.output_specs
        self.assertIsNotNone(output_specs)
        self.assertEqual(len(output_specs), 1)
        self.assertEqual(output_specs[0].placements, (Shard(0),))


class TestShardOptimizerState(unittest.TestCase):
    """Unit tests for _shard_optimizer_state edge cases."""

    def test_missing_key_is_noop(self) -> None:
        """Calling with a key not in named_states does nothing."""
        named_states = {"other_key": torch.randn(4, 4)}
        orig = named_states.copy()
        spec = MagicMock(spec=DTensorSpec)
        _shard_optimizer_state(named_states, "missing_key", spec)
        self.assertEqual(list(named_states.keys()), list(orig.keys()))

    @unittest.mock.patch("recir.distributed.data_parallel.partition_value")
    def test_tensor_state_sharded(self, mock_partition: Any) -> None:
        """A bare tensor state is partitioned via partition_value."""
        tensor = torch.randn(4, 4)
        sharded = torch.randn(2, 4)
        mock_partition.return_value = sharded
        named_states = {"key": tensor}
        spec = MagicMock(spec=DTensorSpec)
        _shard_optimizer_state(named_states, "key", spec)
        mock_partition.assert_called_once_with(tensor, spec)
        self.assertIs(named_states["key"], sharded)

    @unittest.mock.patch("recir.distributed.data_parallel.partition_value")
    def test_dict_state_non_tensor_untouched(self, mock_partition: Any) -> None:
        """Non-tensor entries in state dict (e.g., int step counter) are left unchanged."""
        momentum = torch.randn(4, 4)
        sharded_momentum = torch.randn(2, 4)
        mock_partition.return_value = sharded_momentum
        named_states = {
            "key": {
                "momentum": momentum,
                "step": 42,
            },
        }
        spec = MagicMock(spec=DTensorSpec)
        _shard_optimizer_state(named_states, "key", spec)
        # Tensor should have been passed to partition_value
        mock_partition.assert_called_once_with(momentum, spec)
        self.assertIs(named_states["key"]["momentum"], sharded_momentum)
        # Non-tensor should be untouched
        self.assertEqual(named_states["key"]["step"], 42)


if __name__ == "__main__":
    if False:
        run_tests()
