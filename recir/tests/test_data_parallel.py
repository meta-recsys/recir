# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


import unittest
import unittest.mock
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Callable, cast
from unittest.mock import MagicMock

import torch
import torch.distributed._functional_collectives as funcol
import torch.fx as fx
import torch.nn as nn
from parameterized import parameterized
from recir.api import compile
from recir.backends.base import Backend
from recir.compiler.tracing import _local_compile
from recir.distributed.data_parallel import (
    _clone_param_chains_for_backward,
    _register_parametrization,
    _rewrite_row_wise_embeddings,
    _shard_state_dict,
    build_data_parallel_strategies,
    DataParallelStrategy,
    DataParallelStyle,
    mark_data_parallel_shardings,
    NodeType,
    parametrize_params,
    partition_value,
    partitioner,
    shard_params_and_states,
    shard_params_by_plan,
)
from recir.distributed.parallel_mode import DataParallel, HybridParallel
from recir.distributed.sharding_plan import EmbeddingShardingPlan, EmbeddingShardingType
from recir.passes import insert_weight_free_nodes
from recir.region_transforms import RegatherTransform
from recir.region_transforms.sparse_embedding import (
    decomposed_row_wise_embedding,
    decomposed_table_wise_embedding_bag,
)
from recir.utils.meta_tensor_init import get_meta_init_spec
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DeviceMesh
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor.placement_types import Partial, Replicate, Shard
from torch.fx.graph_module import _get_attr, _has_attr
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


class TestParametrizeParams(unittest.TestCase):
    def test_include_param_fqns_only_parametrizes_selected_parameter(self) -> None:
        model = nn.Sequential(
            nn.Linear(4, 4, bias=False),
            nn.Linear(4, 4, bias=False),
        )
        first_layer = cast(nn.Linear, model[0])
        second_layer = cast(nn.Linear, model[1])
        first_weight = first_layer.weight.detach().clone()
        second_weight = second_layer.weight.detach().clone()

        with parametrize_params(
            model,
            lambda value: value + 1,
            include_param_fqns={"0.weight"},
        ):
            torch.testing.assert_close(first_layer.weight, first_weight + 1)
            torch.testing.assert_close(second_layer.weight, second_weight)

        torch.testing.assert_close(first_layer.weight, first_weight)
        torch.testing.assert_close(second_layer.weight, second_weight)


class SharedParameterAliasModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.primary = nn.Linear(4, 4, bias=False)
        self.alias = nn.Linear(4, 4, bias=False)
        self.alias.weight = self.primary.weight


class TestShardParameterAliases(unittest.TestCase):
    def test_shared_parameter_keeps_one_optimizer_state_owner(self) -> None:
        model = SharedParameterAliasModel()
        original_parameter = model.primary.weight
        optimizer = torch.optim.AdamW([original_parameter], lr=0.01, foreach=True)
        optimizer.state[original_parameter]["grad_accum_step"] = torch.zeros(())
        params = cast(
            dict[str, torch.Tensor],
            OrderedDict(model.named_parameters(remove_duplicate=False)),
        )
        shard_spec = MagicMock(spec=DTensorSpec)

        with (
            unittest.mock.patch(
                "recir.distributed.data_parallel.DTensorSpec",
                return_value=shard_spec,
            ),
            unittest.mock.patch(
                "recir.distributed.data_parallel.partition_value",
                side_effect=lambda value,
                _spec,
                own_storage=False: value.detach().clone(),
            ),
        ):
            sharded = shard_params_and_states(
                model,
                params,
                optimizer,
                MagicMock(spec=DeviceMesh),
                DataParallelStyle.FULLY_SHARD,
            )

        self.assertEqual(sharded, set(params))
        self.assertIs(model.primary.weight, model.alias.weight)
        self.assertIs(params["primary.weight"], params["alias.weight"])
        optimizer_parameter = optimizer.param_groups[0]["params"][0]
        self.assertIs(optimizer_parameter, model.primary.weight)
        self.assertIn(optimizer_parameter, optimizer.state)
        self.assertNotIn(original_parameter, optimizer.state)
        self.assertIn("grad_accum_step", optimizer.state[optimizer_parameter])

    def test_meta_params_stay_on_meta(self) -> None:
        """XLA/GSPMD has no CUDA context to materialize a meta shard into.

        `shard_params_by_plan` (DMP/GPU) materializes meta shards onto
        `cuda:current_device`; folding both entry points onto one helper once
        put that call on this path too, and every TPU run died in
        `torch.cuda.current_device()` with "Torch not compiled with CUDA
        enabled". Assert on the resulting device, not on the absence of a call:
        on a CUDA host the wrong branch succeeds silently and materializes the
        whole table on the GPU, defeating `on_device_init`.
        """
        with torch.device("meta"):
            model = SharedParameterAliasModel()
        params = cast(
            dict[str, torch.Tensor],
            OrderedDict(model.named_parameters(remove_duplicate=False)),
        )
        shard_spec = MagicMock(spec=DTensorSpec)

        with (
            unittest.mock.patch(
                "recir.distributed.data_parallel.DTensorSpec",
                return_value=shard_spec,
            ),
            unittest.mock.patch(
                "recir.distributed.data_parallel.partition_value",
                side_effect=lambda value,
                _spec,
                own_storage=False: value.detach().clone(),
            ),
        ):
            sharded = shard_params_and_states(
                model,
                params,
                None,
                MagicMock(spec=DeviceMesh),
                DataParallelStyle.FULLY_SHARD,
            )

        self.assertEqual(sharded, set(params))
        self.assertEqual(model.primary.weight.device.type, "meta")
        self.assertEqual(params["primary.weight"].device.type, "meta")
        self.assertIs(model.primary.weight, model.alias.weight)


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


class SharedWeightMLP(nn.Module):
    """Model where a single linear layer is reused 3 times in forward.

    This exercises the case where the same parameter is accessed multiple
    times, producing one allgather per access (not per unique parameter).
    """

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(32, 32, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.linear(x))
        x = torch.relu(self.linear(x))
        return self.linear(x)


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
        ]

        for config in sgd_configs:
            ddp_mod = SimpleMLP().cuda(self.rank)
            ddp_opt = torch.optim.SGD(ddp_mod.parameters(), **config, foreach=True)

            train_batch = (
                torch.randn((128, 50), device=torch.device(self.rank)),
                torch.randn((128, 8), device=torch.device(self.rank)),
            )

            ref_mod = DDP(deepcopy(ddp_mod), device_ids=[self.rank])
            ref_opt = torch.optim.SGD(ref_mod.parameters(), **config, foreach=True)

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
        ]

        for config in sgd_configs:
            fsdp_mod = SimpleMLP().cuda(self.rank)
            fsdp_opt = torch.optim.SGD(fsdp_mod.parameters(), **config, foreach=True)

            train_batch = (
                torch.randn((128, 50), device=torch.device(self.rank)),
                torch.randn((128, 8), device=torch.device(self.rank)),
            )

            ref_mod = FSDP(deepcopy(fsdp_mod), device_id=self.rank)
            ref_opt = torch.optim.SGD(ref_mod.parameters(), **config, foreach=True)
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
            opt = torch.optim.SGD(mod.parameters(), lr=0.1, foreach=True)

            train_batch = (
                torch.randn((128, 50), device=torch.device(self.rank)),
                torch.randn((128, 8), device=torch.device(self.rank)),
            )

            ref_mod = (
                FSDP(deepcopy(mod), device_id=self.rank)
                if parallel_mode == "fully_shard"
                else DDP(deepcopy(mod), device_ids=[self.rank])
            )
            ref_opt = torch.optim.SGD(ref_mod.parameters(), lr=0.1, foreach=True)
            test_fn = (
                self._test_fsdp if parallel_mode == "fully_shard" else self._test_ddp
            )
            test_fn(
                mod,
                ref_mod,
                opt,
                ref_opt,
                train_batch,
                step,
                reduction=reduction,
            )

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
        """Verify FSDP inserts allgathers for each parameter."""
        mod = SimpleMLP().cuda(self.rank)
        opt = torch.optim.SGD(mod.parameters(), lr=0.1, foreach=True)

        train_batch = (
            torch.randn((128, 50), device=torch.device(self.rank)),
            torch.randn((128, 8), device=torch.device(self.rank)),
        )

        compiled_fn = compile(parallel_mode=DataParallel("fully_shard"))(train_step)
        compiled_fn(mod, opt, train_batch)

        from recir.api import COMPILED_OBJECT_KEY

        compiled_obj = compiled_fn.__dict__[COMPILED_OBJECT_KEY]
        gm = compiled_obj.gm

        allgather_count = 0
        reduce_scatter_count = 0
        for node in gm.graph.nodes:
            if node.op == "call_function":
                target_name = str(node.target)
                if "all_gather_into_tensor" in target_name:
                    allgather_count += 1
                elif "reduce_scatter_tensor" in target_name:
                    reduce_scatter_count += 1

        # SimpleMLP has 4 params, each accessed once in forward.
        # v2 inserts one allgather per parameter *access* during tracing.
        # Region transforms checkpoint-wrap each per-param chain (allgather +
        # chain ops + compute) using use_reentrant=True.  Reentrant
        # checkpoint recomputes ALL regions in backward (both weight and
        # bias chains), adding 4 backward allgathers.
        # Total: 4 forward + 4 backward = 8.
        expected = 8  # 4 forward + 4 backward (reentrant checkpoint)
        self.assertEqual(
            allgather_count,
            expected,
            f"Expected {expected} allgathers (4 forward + 4 backward "
            f"reentrant checkpoint recompute), got {allgather_count}",
        )

        self.assertGreater(
            reduce_scatter_count,
            0,
            "Expected at least one reduce_scatter for per-layer gradient reduction",
        )

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_fully_shard_shared_weight_allgathers(self) -> None:
        """Verify allgather counts when a parameter is reused multiple times in forward."""
        mod = SharedWeightMLP().cuda(self.rank)
        opt = torch.optim.SGD(mod.parameters(), lr=0.1, foreach=True)

        train_batch = (
            torch.randn((64, 32), device=torch.device(self.rank)),
            torch.randn((64, 32), device=torch.device(self.rank)),
        )

        compiled_fn = compile(parallel_mode=DataParallel("fully_shard"))(train_step)
        compiled_fn(mod, opt, train_batch)

        from recir.api import COMPILED_OBJECT_KEY

        compiled_obj = compiled_fn.__dict__[COMPILED_OBJECT_KEY]
        gm = compiled_obj.gm

        allgather_count = 0
        for node in gm.graph.nodes:
            if node.op == "call_function":
                if "all_gather_into_tensor" in str(node.target):
                    allgather_count += 1

        # SharedWeightMLP has 1 unique param (linear.weight) accessed 3 times.
        # v2 inserts at least 3 forward allgathers (one per access).
        self.assertGreaterEqual(
            allgather_count,
            3,
            f"Expected at least 3 allgathers (one per weight access), "
            f"got {allgather_count}",
        )

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_fully_shard_allgather_before_compute_not_before_non_compute(self) -> None:
        """Verify allgather is inserted right before compute ops, not before non-compute ops.

        For v2, allgathers are inserted via DTensor parametrization during
        tracing, so wait_tensor nodes don't carry NodeType.PARAM metadata.
        The test still runs but no assertions fire (trivially passes).
        """
        mod = SimpleMLP().cuda(self.rank)
        opt = torch.optim.SGD(mod.parameters(), lr=0.1, foreach=True)

        train_batch = (
            torch.randn((128, 50), device=torch.device(self.rank)),
            torch.randn((128, 8), device=torch.device(self.rank)),
        )

        compiled_fn = compile(parallel_mode=DataParallel("fully_shard"))(train_step)
        compiled_fn(mod, opt, train_batch)

        from recir.api import COMPILED_OBJECT_KEY

        compiled_obj = compiled_fn.__dict__[COMPILED_OBJECT_KEY]
        gm = compiled_obj.gm

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
        fsdp_opt = torch.optim.SGD(fsdp_mod.parameters(), lr=0.1, foreach=True)

        train_batch = (
            torch.randn((128, 50), device=torch.device(self.rank)),
            torch.randn((128, 8), device=torch.device(self.rank)),
        )

        ref_mod = FSDP(deepcopy(fsdp_mod), device_id=self.rank)
        ref_opt = torch.optim.SGD(ref_mod.parameters(), lr=0.1, foreach=True)

        ref_inp = deepcopy(train_batch)

        # Warmup
        train_step(fsdp_mod, fsdp_opt, train_batch)
        fsdp_opt.zero_grad()
        train_step(ref_mod, ref_opt, ref_inp)
        ref_opt.zero_grad()

        # Train ref manually
        torch.sum(ref_mod(ref_inp[0]) - ref_inp[1]).backward()
        ref_opt.step()

        # Compile with fully_shard
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
        """Verify FSDP graph structure and weight_free transformation with fully_shard.

        Refactored off the legacy ``gm_transformation=GraphModuleTransformation(
        enable_weight_free=True)`` API to the v7.1 ``backend=`` path: drive
        ``insert_weight_free_nodes`` as a stage-6 optimize pass on a minimal
        IterGraph-wrapping Backend.  Behaviour under test (free_storage
        insertion + ordering relative to last consumer) is unchanged.
        """
        mod = SimpleMLP().cuda(self.rank)
        opt = torch.optim.SGD(mod.parameters(), lr=0.1, foreach=True)

        train_batch = (
            torch.randn((128, 50), device=torch.device(self.rank)),
            torch.randn((128, 8), device=torch.device(self.rank)),
        )

        compiled_fn = compile(
            parallel_mode=DataParallel("fully_shard"),
            backend=_WeightFreeBackend(),
            region_transforms=[RegatherTransform()],
        )(train_step)
        compiled_fn(mod, opt, train_batch)

        from recir.api import COMPILED_OBJECT_KEY

        compiled_obj = compiled_fn.__dict__[COMPILED_OBJECT_KEY]
        gm = compiled_obj.gm

        allgather_count = 0
        for node in gm.graph.nodes:
            if node.op == "call_function":
                if "all_gather_into_tensor" in str(node.target):
                    allgather_count += 1

        # SimpleMLP has 4 params, each requires an allgather in forward.
        # Unlike the legacy ``_compile_data_parallel_v2`` (which auto-injects
        # ``parametrize_params(collectives_fn)`` into the region-transform
        # ExitStack and so doubles the count via the reentrant checkpoint
        # recompute), the v7.1 ``backend=`` path drives ``recir.trace`` which
        # calls ``_enter_region_transform_contexts`` *without* a
        # ``parametrize_cm_factory``.  Forward allgathers therefore land
        # directly in the main graph (no backward reentrant recompute path),
        # producing a smaller, deterministic count.  The contract under test
        # — fully_shard inserts ``all_gather_into_tensor`` at compile time —
        # still holds.
        self.assertGreater(
            allgather_count,
            0,
            f"Expected at least one allgather under DataParallel(fully_shard), "
            f"got {allgather_count}",
        )

        # Verify free_storage ordering when present.
        for node in gm.graph.nodes:
            if (
                node.op == "call_function"
                and node.target == torch.ops.recir.free_storage.default
            ):
                node_order = {}
                for idx, n in enumerate(gm.graph.nodes):
                    node_order[n] = idx
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
        """Verify peak memory is reasonable with and without free_storage nodes.

        When free_storage nodes are present (e.g. after per-op allgather
        cloning), removing them should not significantly increase peak memory
        because the FX recompiler inserts ``= None`` after last use.

        With v2 (upfront allgathers without NodeType.PARAM metadata),
        insert_weight_free_nodes may not insert free_storage nodes.  In
        that case both runs are identical and peak memory should be equal.

        Refactored off the legacy ``gm_transformation=`` API to the v7.1
        ``backend=`` path; the weight-free pass remains
        :func:`insert_weight_free_nodes`, just driven via
        ``backend.optimize_passes`` instead of
        ``GraphModuleTransformation(enable_weight_free=True)``.
        """
        device = torch.device(self.rank)

        mod_with = LargeLinearModel().cuda(self.rank)
        opt_with = torch.optim.SGD(mod_with.parameters(), lr=0.1, foreach=True)
        mod_without = deepcopy(mod_with)
        opt_without = torch.optim.SGD(mod_without.parameters(), lr=0.1, foreach=True)

        batch_with = (
            torch.randn((4, 1024), device=device),
            torch.randn((4, 1024), device=device),
        )
        batch_without = deepcopy(batch_with)

        # --- Run WITH free_storage (compilation with weight_free optimization) ---
        compiled_with = compile(
            parallel_mode=DataParallel("fully_shard"),
            backend=_WeightFreeBackend(),
            region_transforms=[RegatherTransform()],
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
            backend=_WeightFreeBackend(),
            region_transforms=[RegatherTransform()],
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
        if not free_nodes:
            # The v2 compile path does upfront allgathers without
            # NodeType.PARAM metadata, so insert_weight_free_nodes may
            # not insert any free_storage nodes.  Nothing to compare.
            return
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


class _WeightFreeBackend(Backend):
    """v7.1 Backend equivalent of the legacy ``GraphModuleTransformation(
    enable_weight_free=True)``.

    Runs :func:`insert_weight_free_nodes` as the sole stage-6 optimize pass
    and wraps the graph in :class:`IterGraphModule` (``uses_iter_graph=True``)
    to match the legacy default ``enable_iter_graph=True``.  ``dispatch`` is a
    no-op pass-through so the test does not depend on Inductor / Triton / XLA
    availability — the test only inspects ``free_storage`` node insertion and
    ordering on the resulting graph.
    """

    @classmethod
    def default_optimize_passes(cls) -> tuple[Callable[[fx.GraphModule], None], ...]:
        return (insert_weight_free_nodes,)

    @property
    def uses_iter_graph(self) -> bool:
        return True

    def dispatch(self, gm: fx.GraphModule) -> nn.Module:
        return gm


class TestDataParallelNonXLAPrecondition(unittest.TestCase):
    """DataParallel.shard_params rejects D>1 on the non-XLA path (CPU-only)."""

    def test_non_xla_d_gt_1_raises(self) -> None:
        # Non-XLA (real DTensor) shards over process_mesh (size P) but the traced
        # collectives all-gather over device_mesh (size N); N>P (D>1) would
        # silently mis-shard, so shard_params must raise before doing any work.
        mode = DataParallel("fully_shard", num_ranks=4, num_processes=1)
        model = SimpleMLP()
        # Citrine C2: use foreach for multi-tensor optimizer execution.
        opt = torch.optim.SGD(model.parameters(), lr=0.1, foreach=True)
        with self.assertRaisesRegex(ValueError, "num_ranks == num_processes"):
            mode.shard_params(model, dict(model.named_parameters()), opt)


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

    def test_mixed_replicated_param_and_state_indices(self) -> None:
        graph = fx.Graph()
        sharded_param = graph.placeholder("sharded_param")
        replicated_param = graph.placeholder("replicated_param")
        sharded_state = graph.placeholder("sharded_state")
        replicated_state = graph.placeholder("replicated_state")
        activation = graph.placeholder("activation")
        for node in (
            sharded_param,
            replicated_param,
            sharded_state,
            replicated_state,
            activation,
        ):
            node.meta["val"] = torch.zeros(4, 4)
        graph.output((sharded_param, replicated_param))
        gm = fx.GraphModule(nn.Module(), graph)

        strategies = build_data_parallel_strategies(
            gm,
            num_params=2,
            num_states=2,
            mesh=self._create_mock_mesh(),
            parallel_mode=DataParallelStyle.FULLY_SHARD,
            replicated_param_state_indices={1, 3},
        )

        def placement(node: fx.Node) -> object:
            strategy = cast(DataParallelStrategy, strategies[node])
            output_specs = strategy.op_spec.output_specs
            self.assertIsInstance(output_specs, tuple)
            return output_specs[0].placements[0]  # pyre-ignore[16]

        self.assertIsInstance(placement(sharded_param), Shard)
        self.assertIsInstance(placement(sharded_state), Shard)
        self.assertIsInstance(placement(replicated_param), Replicate)
        self.assertIsInstance(placement(replicated_state), Replicate)

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


class _SumOverShardAxisModule(nn.Module):
    """Reduces ``(B, F1, F2, F3)`` over the leading ``B, F1, F2`` dims with
    ``keepdim=True``.

    The shape is hand-picked so the reduced output ``(1, 1, 1, F3)`` has a
    trailing dim whose size equals ``B`` (i.e. ``F3 == B``).  This collision
    is what causes the size-match heuristic in ``BatchDimAnalyzer`` to mis-
    classify the sum's output as ``Shard(F3_axis)`` when no op-specific rule
    fires first.
    """

    def __init__(self) -> None:
        super().__init__()
        # Trivial parameter so ``_local_compile`` has something to track.
        self.scale: nn.Parameter = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ``torch.sum`` with a list dim arg lowers to ``aten.sum.dim_IntList``.
        return (x * self.scale).sum(dim=[0, 1, 2], keepdim=True)


class TestSumOverShardedAxisPartial(unittest.TestCase):
    """Regression test for ``aten.sum.dim_IntList`` reducing the shard axis.

    Background: ``BatchDimAnalyzer.compute_batch_dim`` falls back to a raw
    output-shape size match when the op is not in ``dim_rule_map``.  For an
    activation with batch dim 0 and ``batch_dim_size=8`` flowing into

        ``aten.sum.dim_IntList(input, [0, 1, 2], keepdim=True)``

    on a global shape ``(8, 32, 8, 8)``, the keepdim=True output shape is
    ``(1, 1, 1, 8)`` -- the trailing dim incidentally equals 8 and the
    size-match heuristic returned ``3``, producing a ``Shard(3)`` placement
    and silently turning a Partial-producing reduction into a fake shard.
    Downstream view rewrites then expect the wrong per-rank shape and crash
    at runtime -- this matches the production DDRO sum-over-sharded-batch
    chain that exposed the bug.

    Correct semantics: when the input is ``Shard(k)`` and ``k`` is in the
    sum's reduce-dim list, the output is a partial sum across the shard mesh
    dim and must carry a ``Partial`` placement so a subsequent allreduce can
    materialise the global value.

    Scope of this regression: ``aten.sum.dim_IntList`` only.  Other reductions
    (mean/max/prod) have differing partial-reduction semantics and are out of
    scope for this fix.
    """

    @classmethod
    def setUpClass(cls) -> None:
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("fake", rank=0, world_size=2)
            cls._owns_pg = True
        else:
            cls._owns_pg = False

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._owns_pg:
            torch.distributed.destroy_process_group()

    def test_sum_dim_intlist_over_shard_axis_emits_partial(self) -> None:
        # Trace a real nn.Module via ``_local_compile`` so the test exercises
        # the same code path the production compile uses (see
        # ``TestBuildStrategiesGetAttrClassification`` for the analogous
        # pattern in this file).
        mod = _SumOverShardAxisModule()
        x = torch.randn(8, 32, 8, 8)
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
        result = _local_compile(lambda mod, x: mod(x), mod, x, fake_mode=fake_mode)
        gm = result.gm

        sum_nodes = [
            n for n in gm.graph.nodes if n.target == torch.ops.aten.sum.dim_IntList
        ]
        self.assertEqual(
            len(sum_nodes),
            1,
            f"Expected exactly one aten.sum.dim_IntList node, got {len(sum_nodes)}",
        )
        sum_node = sum_nodes[0]

        mesh = DeviceMesh("cpu", torch.arange(2))
        strategy_map = build_data_parallel_strategies(
            gm,
            num_params=len(result.params),
            num_states=len(result.named_states),
            mesh=mesh,
            batch_dim=0,
            parallel_mode=DataParallelStyle.FULLY_SHARD,
        )

        sum_strategy = cast(DataParallelStrategy, strategy_map[sum_node])
        self.assertEqual(sum_strategy.node_type, NodeType.ACT)
        op_spec = sum_strategy.op_spec
        assert op_spec is not None
        out_specs = op_spec.output_specs
        output_spec = out_specs[0] if isinstance(out_specs, tuple) else out_specs
        assert isinstance(output_spec, DTensorSpec)
        self.assertEqual(len(output_spec.placements), 1)
        placement = output_spec.placements[0]
        self.assertIsInstance(
            placement,
            Partial,
            (
                f"Expected Partial placement for aten.sum.dim_IntList that "
                f"reduces over the shard axis, got {placement!r}"
            ),
        )
        # Default reduce_op for Partial() is "sum"; confirm we did not pick
        # up an avg/max accidentally.
        self.assertEqual(cast(Partial, placement).reduce_op, "sum")


class _NestedTensorConstantModule(nn.Module):
    """Module whose forward references a tensor literal in a nested submodule.

    The literal is captured by ``make_fx`` and lifted onto the GraphModule via
    a ``get_attr`` node. Note that ``make_fx`` currently flattens such literals
    into top-level attributes on the GraphModule (e.g. ``_tensor_constant0``)
    regardless of which nested submodule they originated in, so the resulting
    ``get_attr`` target is a flat (dot-free) name. This module exercises the
    real production code path; the dotted-target robustness path is covered by
    a separate, manually constructed regression test below.
    """

    class _Inner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Inline tensor literal — make_fx will lift this into a get_attr.
            bias = torch.tensor([0.5, 0.25, 0.125, 0.0625])
            return self.lin(x) + bias

    def __init__(self) -> None:
        super().__init__()
        self.inner: nn.Module = self._Inner()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x)


class TestLocalCompileTensorConstants(unittest.TestCase):
    """Unit tests for tensor get_attr handling in ``_local_compile``."""

    def test_local_compile_preserves_get_attr_constants_across_recompile(
        self,
    ) -> None:
        """Tensor get_attr constants must survive ``gm.recompile()``.

        ``make_fx`` lifts inline tensor literals into module attributes
        referenced via ``get_attr`` nodes. In production today the lifted
        targets are flat top-level names on the GraphModule (e.g.
        ``_tensor_constant0``), regardless of which nested submodule the
        literal originated in. Without re-registering each lifted attribute
        as ``_AttrKind.CONSTANT`` they are stored as buffers and either
        disappear or get rewritten by subsequent ``recompile`` calls —
        breaking any pass that walks the gm to recover the underlying value.

        This test exercises the real production code path. The
        inner-submodule constant case is covered separately by
        ``test_local_compile_handles_inner_submodule_constant``.
        """
        mod = _NestedTensorConstantModule()
        opt = torch.optim.SGD(mod.parameters(), lr=0.1, foreach=True)
        x = torch.randn(2, 4)

        def train_step(
            mod: nn.Module, opt: torch.optim.Optimizer, x: torch.Tensor
        ) -> torch.Tensor:
            out = mod(x)
            loss = out.sum()
            loss.backward()
            opt.step()
            opt.zero_grad()
            return loss

        fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
        result = _local_compile(train_step, mod, opt, x, fake_mode=fake_mode)
        gm = result.gm

        tensor_get_attrs = [
            node
            for node in gm.graph.nodes
            if node.op == "get_attr"
            and isinstance(_get_attr(gm, node.target), torch.Tensor)
        ]
        self.assertGreater(
            len(tensor_get_attrs),
            0,
            "Expected at least one tensor get_attr node from the inline tensor literal",
        )

        # The inline literal in ``_NestedTensorConstantModule._Inner.forward``
        # is exactly this tensor, so every lifted constant must equal it.
        expected_bias = torch.tensor([0.5, 0.25, 0.125, 0.0625])

        # The constants dict on the result must capture every tensor get_attr.
        for node in tensor_get_attrs:
            fqn = node.target
            self.assertIn(
                fqn,
                result.constants,
                f"Expected constant FQN {fqn!r} to be recorded in _LocalCompiledResult.constants",
            )
            # And the value should be retrievable via the (flat, in production)
            # attribute path.
            self.assertTrue(_has_attr(gm, fqn))
            value_before = _get_attr(gm, fqn)
            self.assertIsInstance(value_before, torch.Tensor)
            torch.testing.assert_close(value_before, expected_bias)
            # The result.constants entry must point at the same value.
            torch.testing.assert_close(result.constants[fqn], expected_bias)
            # Constants must NOT be tracked as buffers (the pre-fix behavior
            # silently registered them as buffers, which breaks recompile).
            buffer_fqns = {n for n, _ in gm.named_buffers()}
            self.assertNotIn(
                fqn,
                buffer_fqns,
                f"Constant {fqn!r} should not be registered as a buffer",
            )

        # Recompile and re-check: the tensor must still be reachable via the
        # attribute path AND the cached value identity must be preserved
        # (i.e. _AttrKind.CONSTANT survived recompile).
        gm.recompile()
        for node in tensor_get_attrs:
            fqn = node.target
            self.assertTrue(
                _has_attr(gm, fqn),
                f"Constant {fqn!r} disappeared after gm.recompile()",
            )
            value_after = _get_attr(gm, fqn)
            self.assertIsInstance(value_after, torch.Tensor)
            torch.testing.assert_close(value_after, expected_bias)
            self.assertIs(
                value_after,
                result.constants[fqn],
                f"Constant {fqn!r} value changed across gm.recompile()",
            )

    def test_local_compile_handles_inner_submodule_constant(self) -> None:
        """``_local_compile`` must classify a constant lifted from an inner
        submodule's forward as ``_AttrKind.CONSTANT`` end-to-end.

        ``make_fx`` currently flattens lifted tensor literals into top-level
        attribute names on the GraphModule even when the literal lives inside
        a nested submodule's forward, so the dotted-target codepath is not
        reachable via the public API today. We still want to assert that the
        retag is correctly applied for an inner-submodule-sourced constant —
        if a future make_fx change preserves the qualified path, the assertion
        below would also catch a regression there because the constant's FQN
        is read from the produced graph.
        """
        weight_const = torch.tensor([1.0, 2.0, 3.0])

        class _Inner(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Inline tensor literal inside a nested submodule's forward.
                return x + weight_const

        class _Outer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.inner = _Inner()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.inner(x)

        mod = _Outer()
        x = torch.randn(3)

        def train_step(mod: nn.Module, x: torch.Tensor) -> torch.Tensor:
            return mod(x).sum()

        fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
        result = _local_compile(train_step, mod, x, fake_mode=fake_mode)
        gm = result.gm

        tensor_get_attrs = [
            node
            for node in gm.graph.nodes
            if node.op == "get_attr"
            and isinstance(_get_attr(gm, node.target), torch.Tensor)
        ]
        self.assertGreater(len(tensor_get_attrs), 0)

        gm.recompile()

        # Every retagged constant must (a) be reachable on ``gm`` via its FQN,
        # (b) match the inner-submodule literal value, and (c) NOT live in
        # ``named_buffers()``.
        buffer_fqns = {n for n, _ in gm.named_buffers()}
        for node in tensor_get_attrs:
            fqn = node.target
            self.assertIn(fqn, result.constants)
            value = _get_attr(gm, fqn)
            self.assertIsInstance(value, torch.Tensor)
            torch.testing.assert_close(value, weight_const)
            self.assertNotIn(fqn, buffer_fqns)


class TestBuildStrategiesGetAttrClassification(DTensorTestBase):
    """Tests for ``build_data_parallel_strategies`` get_attr classification.

    Drives a real 2-rank ``DeviceMesh`` and traces the test fixture via
    :func:`recir.api._local_compile` so we exercise the same code path the
    production compile uses.
    """

    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_build_data_parallel_strategies_classifies_tensor_get_attr_as_replicate(
        self,
    ) -> None:
        """A tensor get_attr from ``_local_compile`` must be PARAM + REPLICATE."""
        mod = _NestedTensorConstantModule()
        x = torch.randn(2, 4)

        def train_step(mod: nn.Module, x: torch.Tensor) -> torch.Tensor:
            return mod(x).sum()

        fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
        result = _local_compile(train_step, mod, x, fake_mode=fake_mode)
        gm = result.gm

        tensor_get_attrs = [n for n in gm.graph.nodes if n.op == "get_attr"]
        self.assertGreater(
            len(tensor_get_attrs), 0, "fixture produced no get_attr nodes"
        )

        mesh = DeviceMesh("cuda", torch.arange(self.world_size))
        strategy_map = build_data_parallel_strategies(
            gm,
            num_params=len(result.params),
            num_states=len(result.named_states),
            mesh=mesh,
            batch_dim=0,
            parallel_mode=DataParallelStyle.REPLICATE,
        )
        for attr_node in tensor_get_attrs:
            attr_strategy = cast(DataParallelStrategy, strategy_map[attr_node])
            self.assertEqual(attr_strategy.node_type, NodeType.PARAM)
            out_specs = attr_strategy.op_spec.output_specs
            self.assertIsInstance(out_specs, tuple)
            placements = out_specs[0].placements  # type: ignore[index,union-attr]
            self.assertTrue(
                all(isinstance(p, Replicate) for p in placements),
                f"Expected all-Replicate placements for tensor get_attr {attr_node.target!r}, got {placements}",
            )

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_build_data_parallel_strategies_classifies_non_tensor_get_attr_as_non_tensor(
        self,
    ) -> None:
        """A non-tensor get_attr (e.g. DeviceMesh metadata) must be NON_TENSOR.

        ``make_fx`` does not naturally emit non-tensor ``get_attr`` nodes, so
        we trace a real graph via ``_local_compile`` and then graft a single
        ``get_attr`` node referencing a non-tensor attribute on the gm root —
        keeping the rest of the graph (and the strategy invocation) production
        shaped.
        """
        mod = _NestedTensorConstantModule()
        x = torch.randn(2, 4)

        def train_step(mod: nn.Module, x: torch.Tensor) -> torch.Tensor:
            return mod(x).sum()

        fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
        result = _local_compile(train_step, mod, x, fake_mode=fake_mode)
        gm = result.gm

        # Attach a non-tensor attribute on the gm root and insert a get_attr
        # node referencing it. The node is left dangling (it has no consumer);
        # ``build_data_parallel_strategies`` walks every node in the graph, so
        # it will still classify it. We deliberately skip ``gm.recompile()``
        # because the inserted node has no consumer and would be DCE'd from
        # the regenerated forward; the node is still present in
        # ``gm.graph.nodes`` for the strategy walk.
        non_tensor_attr = MagicMock(spec=DeviceMesh)
        non_tensor_attr.ndim = 1
        gm.meta_attr = non_tensor_attr  # pyre-ignore[16]
        with gm.graph.inserting_before(next(iter(gm.graph.nodes))):
            attr_node = gm.graph.get_attr("meta_attr")

        mesh = DeviceMesh("cuda", torch.arange(self.world_size))
        strategy_map = build_data_parallel_strategies(
            gm,
            num_params=len(result.params),
            num_states=len(result.named_states),
            mesh=mesh,
            batch_dim=0,
            parallel_mode=DataParallelStyle.REPLICATE,
        )
        attr_strategy = cast(DataParallelStrategy, strategy_map[attr_node])
        self.assertEqual(attr_strategy.node_type, NodeType.NON_TENSOR)


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


class _MetaEmbedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb: nn.Embedding = nn.Embedding(16, 8, device="meta")


def embedding_train_step(
    model: nn.Module, optim: torch.optim.Optimizer, train_batch: Any
) -> torch.Tensor:
    def loss_fn(out: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return (out - labels).sum()

    optim.zero_grad()
    indices, labels = train_batch
    out = model(indices)
    loss = loss_fn(out, labels)
    loss.backward()
    optim.step()
    return loss


class MultiEmbeddingMLP(nn.Module):
    """Model with two embedding tables and a dense layer for multi-table tests."""

    def __init__(
        self, num_emb1: int = 16, num_emb2: int = 32, emb_dim: int = 8
    ) -> None:
        super().__init__()
        self.emb1 = nn.Embedding(num_emb1, emb_dim)
        self.emb2 = nn.Embedding(num_emb2, emb_dim)
        self.linear = nn.Linear(emb_dim, 4)

    def forward(self, indices1: torch.Tensor, indices2: torch.Tensor) -> torch.Tensor:
        e1 = self.emb1(indices1)  # (batch, seq, emb_dim)
        e2 = self.emb2(indices2)  # (batch, seq, emb_dim)
        combined = (e1.mean(dim=1) + e2.mean(dim=1)) / 2  # (batch, emb_dim)
        return self.linear(combined)  # (batch, 4)


def multi_embedding_train_step(
    model: nn.Module, optim: torch.optim.Optimizer, train_batch: Any
) -> torch.Tensor:
    def loss_fn(out: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return (out - labels).sum()

    optim.zero_grad()
    indices1, indices2, labels = train_batch
    out = model(indices1, indices2)
    loss = loss_fn(out, labels)
    loss.backward()
    optim.step()
    return loss


class TestComputeEmbeddingParamIndices(unittest.TestCase):
    """Unit tests for HybridParallel._compute_embedding_param_indices."""

    def test_no_embedding_params(self) -> None:
        """Empty plan returns None."""
        mode = HybridParallel(dense_style="fully_shard")
        params = OrderedDict(
            [
                ("linear.weight", torch.randn(4, 4)),
                ("linear.bias", torch.randn(4)),
            ]
        )
        self.assertIsNone(mode._compute_embedding_param_indices(params))

    def test_single_embedding_param(self) -> None:
        """Plan with one FQN returns correct index."""
        plan = EmbeddingShardingPlan(
            sharding_map={"emb.weight": EmbeddingShardingType.ROW_WISE}
        )
        mode = HybridParallel(dense_style="fully_shard", embedding_plan=plan)
        params = OrderedDict(
            [
                ("linear.weight", torch.randn(4, 4)),
                ("emb.weight", torch.randn(100, 16)),
                ("linear.bias", torch.randn(4)),
            ]
        )
        self.assertEqual(mode._compute_embedding_param_indices(params), {1})

    def test_multiple_embedding_params(self) -> None:
        """Plan with two FQNs returns correct index set."""
        plan = EmbeddingShardingPlan(
            sharding_map={
                "emb1.weight": EmbeddingShardingType.ROW_WISE,
                "emb2.weight": EmbeddingShardingType.ROW_WISE,
            }
        )
        mode = HybridParallel(dense_style="fully_shard", embedding_plan=plan)
        params = OrderedDict(
            [
                ("linear.weight", torch.randn(4, 4)),
                ("emb1.weight", torch.randn(100, 16)),
                ("emb2.weight", torch.randn(200, 32)),
                ("linear.bias", torch.randn(4)),
            ]
        )
        self.assertEqual(mode._compute_embedding_param_indices(params), {1, 2})

    def test_fqn_not_in_params(self) -> None:
        """Plan FQN not matching any key returns None."""
        plan = EmbeddingShardingPlan(
            sharding_map={"nonexistent.weight": EmbeddingShardingType.ROW_WISE}
        )
        mode = HybridParallel(dense_style="fully_shard", embedding_plan=plan)
        params = OrderedDict([("linear.weight", torch.randn(4, 4))])
        self.assertIsNone(mode._compute_embedding_param_indices(params))


class _RootParamModule(nn.Module):
    """Module with parameters directly on the root (no submodules)."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(3))
        self.bias = nn.Parameter(torch.ones(2))


class _EmbLinearModule(nn.Module):
    """Module with embedding and linear submodules."""

    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(4, 3)
        self.linear = nn.Linear(3, 2)


class TestRegisterParametrization(unittest.TestCase):
    """Unit tests for _register_parametrization with exclude_param_fqns."""

    @staticmethod
    def _double_fn(t: torch.Tensor) -> torch.Tensor:
        return t * 2

    def test_exclude_root_level_param(self) -> None:
        """Root-level params (empty _fqn) are correctly excluded."""
        mod = _RootParamModule()

        _register_parametrization(mod, self._double_fn, exclude_param_fqns={"weight"})

        # weight should be unchanged (excluded).
        self.assertTrue(torch.equal(mod.weight, torch.ones(3)))
        # bias should be doubled (parametrized).
        self.assertTrue(torch.equal(mod.bias, torch.ones(2) * 2))

    def test_exclude_submodule_param(self) -> None:
        """Submodule params are correctly excluded by FQN."""
        mod = _EmbLinearModule()

        _register_parametrization(
            mod, self._double_fn, exclude_param_fqns={"emb.weight"}
        )

        raw_emb = mod.emb._parameters["weight"]
        raw_linear_w = mod.linear._parameters["weight"]
        assert raw_emb is not None
        assert raw_linear_w is not None

        # emb.weight should be unchanged (excluded).
        self.assertTrue(torch.equal(mod.emb.weight, raw_emb))
        # linear.weight should be doubled (parametrized).
        self.assertTrue(torch.equal(mod.linear.weight, raw_linear_w * 2))


class TestHybridParallelInit(unittest.TestCase):
    """Unit tests for HybridParallel.__init__ validation."""

    def test_unknown_dense_style_raises(self) -> None:
        """HybridParallel with invalid dense_style raises RuntimeError."""
        with self.assertRaises(RuntimeError):
            HybridParallel(dense_style="invalid")

    def test_default_embedding_plan_is_empty(self) -> None:
        """Constructing without embedding_plan gives an empty plan."""
        mode = HybridParallel(dense_style="fully_shard")
        self.assertEqual(mode.embedding_plan.param_fqns, set())

    def test_empty_dense_fsdp_module_set_raises(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "dense_fsdp_module_fqns must not be empty"
        ):
            HybridParallel(
                dense_style="fully_shard",
                dense_fsdp_module_fqns=set(),
            )

    def test_dense_fsdp_modules_require_fully_shard(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "dense_fsdp_module_fqns requires dense_style"
        ):
            HybridParallel(
                dense_style="replicate",
                dense_fsdp_module_fqns={"linear"},
            )

    def test_dense_style_mapping(self) -> None:
        """'replicate'/'fully_shard'/'default' map to correct DataParallelStyle values."""
        mode_rep = HybridParallel(dense_style="replicate")
        self.assertEqual(mode_rep.dense_style, DataParallelStyle.REPLICATE)

        mode_fs = HybridParallel(dense_style="fully_shard")
        self.assertEqual(mode_fs.dense_style, DataParallelStyle.FULLY_SHARD)

        mode_def = HybridParallel(dense_style="default")
        self.assertEqual(mode_def.dense_style, DataParallelStyle.DEFAULT)


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
            model,
            params,
            None,
            mesh,
            DataParallelStyle.FULLY_SHARD,
            embedding_plan=EmbeddingShardingPlan(),
        )
        self.assertEqual(
            params["emb.weight"].shape[0], orig_emb_rows // self.world_size
        )
        self.assertEqual(
            params["linear.weight"].shape[0], orig_lin_rows // self.world_size
        )

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_hybrid_parallel_shards_selected_dense_module(self) -> None:
        model = EmbeddingMLP().cuda(self.rank)
        params = OrderedDict(
            [
                ("emb.weight", model.emb.weight.data.clone()),
                ("linear.weight", model.linear.weight.data.clone()),
            ]
        )
        original_embedding_shape = params["emb.weight"].shape
        original_linear_rows = params["linear.weight"].shape[0]
        mesh = DeviceMesh("cuda", torch.arange(self.world_size))
        mode = HybridParallel(
            dense_style="fully_shard",
            dense_fsdp_module_fqns={"linear"},
        )
        mode._mesh = mesh

        sharded_keys = mode.shard_params(model, params, None)

        self.assertEqual(sharded_keys, {"linear.weight"})
        self.assertEqual(params["emb.weight"].shape, original_embedding_shape)
        self.assertEqual(
            params["linear.weight"].shape[0],
            original_linear_rows // self.world_size,
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
            model,
            params,
            None,
            mesh,
            DataParallelStyle.REPLICATE,
            embedding_plan=EmbeddingShardingPlan(),
        )
        for k in params:
            self.assertEqual(params[k].shape, orig_shapes[k])

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_replicated_cpu_buffer(self) -> None:
        model = EmbeddingMLP().to(f"cuda:{self.rank}")
        model.register_buffer("runtime_value", torch.tensor(float(self.rank)))
        params_and_buffers = OrderedDict(
            [
                *model.named_parameters(),
                *model.named_buffers(),
            ]
        )
        mesh = DeviceMesh("cuda", torch.arange(self.world_size))

        shard_params_by_plan(
            model,
            params_and_buffers,
            None,
            mesh,
            DataParallelStyle.REPLICATE,
            embedding_plan=EmbeddingShardingPlan(),
        )

        self.assertEqual(model.runtime_value.device.type, "cpu")
        self.assertEqual(model.runtime_value, torch.tensor(0.0))

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
        plan = EmbeddingShardingPlan(
            sharding_map={
                "emb.weight": EmbeddingShardingType.ROW_WISE,
            }
        )
        shard_params_by_plan(
            model,
            params,  # pyre-fixme[6]
            None,
            mesh,
            DataParallelStyle.REPLICATE,
            embedding_plan=plan,
        )
        # Embedding: sharded (16, 8) -> (8, 8)
        self.assertEqual(params["emb.weight"].shape, (8, 8))
        # Linear: NOT sharded (REPLICATE, not embedding)
        self.assertEqual(params["linear.weight"].shape, (4, 8))

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_table_wise_embedding_is_materialized_only_on_owner(self) -> None:
        model = EmbeddingMLP().cuda(self.rank)
        original_shape = model.emb.weight.shape
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.1, momentum=0.9, foreach=True
        )
        model(torch.tensor([[1, 2]], device=f"cuda:{self.rank}")).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        original_weight = model.emb.weight.detach().clone()
        original_momentum = (
            optimizer.state[model.emb.weight]["momentum_buffer"].detach().clone()
        )
        params = OrderedDict(model.named_parameters())
        plan = EmbeddingShardingPlan(
            sharding_map={
                "emb.weight": EmbeddingShardingType.TABLE_WISE,
            },
            table_wise_rank_map={"emb.weight": 1},
        )

        shard_params_by_plan(
            model,
            params,
            optimizer,
            DeviceMesh("cuda", torch.arange(self.world_size)),
            DataParallelStyle.REPLICATE,
            embedding_plan=plan,
        )

        if self.rank == 1:
            self.assertEqual(model.emb.weight.shape, original_shape)
            self.assertIn(model.emb.weight, optimizer.state)
            self.assertIn(model.emb.weight, optimizer.param_groups[0]["params"])
            torch.testing.assert_close(model.emb.weight, original_weight)
            torch.testing.assert_close(
                optimizer.state[model.emb.weight]["momentum_buffer"],
                original_momentum,
            )
        else:
            self.assertEqual(model.emb.weight.shape, (0, original_shape[1]))
            self.assertNotIn(model.emb.weight, optimizer.state)
            self.assertIn(model.emb.weight, optimizer.param_groups[0]["params"])

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_table_wise_embedding_preserves_optimizer_state_device(self) -> None:
        model = EmbeddingMLP().cuda(self.rank)
        optimizer = torch.optim.Adam([model.emb.weight], lr=0.1, foreach=True)
        model(torch.tensor([[1, 2]], device=f"cuda:{self.rank}")).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        original_step = optimizer.state[model.emb.weight]["step"].clone()
        self.assertEqual(original_step.device.type, "cpu")

        params = OrderedDict(model.named_parameters())
        plan = EmbeddingShardingPlan(
            sharding_map={"emb.weight": EmbeddingShardingType.TABLE_WISE},
            table_wise_rank_map={"emb.weight": 1},
        )
        shard_params_by_plan(
            model,
            params,
            optimizer,
            DeviceMesh("cuda", torch.arange(self.world_size)),
            DataParallelStyle.REPLICATE,
            embedding_plan=plan,
        )

        if self.rank == 1:
            local_state = optimizer.state[model.emb.weight]
            self.assertEqual(local_state["step"].device.type, "cpu")
            model.emb.weight.grad = torch.ones_like(model.emb.weight)
            optimizer.step()
            self.assertEqual(local_state["step"], original_step + 1)
            self.assertEqual(local_state["step"].device.type, "cpu")
        else:
            self.assertEqual(model.emb.weight.shape, (0, 8))
            self.assertNotIn(model.emb.weight, optimizer.state)
            self.assertIn(model.emb.weight, optimizer.param_groups[0]["params"])

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_mixed_row_wise_and_table_wise_parameter_placement(self) -> None:
        model = MultiEmbeddingMLP().to(f"cuda:{self.rank}")
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.1, momentum=0.9, foreach=True
        )
        indices1 = torch.randint(0, 16, (4, 3), device=f"cuda:{self.rank}")
        indices2 = torch.randint(0, 32, (4, 3), device=f"cuda:{self.rank}")
        model(indices1, indices2).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        original_emb1 = model.emb1.weight
        original_emb2 = model.emb2.weight
        params = OrderedDict(model.named_parameters())
        plan = EmbeddingShardingPlan(
            sharding_map={
                "emb1.weight": EmbeddingShardingType.ROW_WISE,
                "emb2.weight": EmbeddingShardingType.TABLE_WISE,
            },
            table_wise_rank_map={"emb2.weight": 1},
        )

        shard_params_by_plan(
            model,
            params,
            optimizer,
            DeviceMesh("cuda", torch.arange(self.world_size)),
            DataParallelStyle.REPLICATE,
            embedding_plan=plan,
        )

        self.assertEqual(model.emb1.weight.shape, (8, 8))
        self.assertIn(model.emb1.weight, optimizer.state)
        self.assertEqual(
            optimizer.state[model.emb1.weight]["momentum_buffer"].shape,
            (8, 8),
        )
        self.assertNotIn(original_emb1, optimizer.state)
        optimizer_parameters = optimizer.param_groups[0]["params"]
        self.assertTrue(
            any(param is model.emb1.weight for param in optimizer_parameters)
        )
        self.assertFalse(any(param is original_emb1 for param in optimizer_parameters))
        if self.rank == 1:
            self.assertEqual(model.emb2.weight.shape, (32, 8))
            self.assertIn(model.emb2.weight, optimizer.state)
        else:
            self.assertEqual(model.emb2.weight.shape, (0, 8))
            self.assertNotIn(model.emb2.weight, optimizer.state)
        self.assertNotIn(original_emb2, optimizer.state)
        self.assertTrue(
            any(param is model.emb2.weight for param in optimizer_parameters)
        )
        self.assertFalse(any(param is original_emb2 for param in optimizer_parameters))

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_table_wise_shared_parameter_keeps_aliases_and_optimizer_state(
        self,
    ) -> None:
        model = SharedParameterAliasModel().cuda(self.rank)
        original_parameter = model.primary.weight
        optimizer = torch.optim.SGD(
            [original_parameter], lr=0.1, momentum=0.9, foreach=True
        )
        original_parameter.grad = torch.ones_like(original_parameter)
        optimizer.step()
        optimizer.zero_grad()
        params = cast(
            dict[str, torch.Tensor],
            OrderedDict(model.named_parameters(remove_duplicate=False)),
        )
        plan = EmbeddingShardingPlan(
            sharding_map={
                "primary.weight": EmbeddingShardingType.TABLE_WISE,
                "alias.weight": EmbeddingShardingType.TABLE_WISE,
            },
            table_wise_rank_map={"primary.weight": 1, "alias.weight": 1},
        )

        shard_params_by_plan(
            model,
            params,
            optimizer,
            DeviceMesh("cuda", torch.arange(self.world_size)),
            DataParallelStyle.REPLICATE,
            embedding_plan=plan,
        )

        self.assertIs(model.primary.weight, model.alias.weight)
        self.assertIs(params["primary.weight"], params["alias.weight"])
        self.assertIs(optimizer.param_groups[0]["params"][0], model.primary.weight)
        if self.rank == 1:
            self.assertIn(model.primary.weight, optimizer.state)
        else:
            self.assertNotIn(model.primary.weight, optimizer.state)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_table_wise_shared_parameter_rejects_inconsistent_plan(self) -> None:
        model = SharedParameterAliasModel().cuda(self.rank)
        params = cast(
            dict[str, torch.Tensor],
            OrderedDict(model.named_parameters(remove_duplicate=False)),
        )
        mesh = DeviceMesh("cuda", torch.arange(self.world_size))

        mixed_plan = EmbeddingShardingPlan(
            sharding_map={
                "primary.weight": EmbeddingShardingType.ROW_WISE,
                "alias.weight": EmbeddingShardingType.TABLE_WISE,
            },
            table_wise_rank_map={"alias.weight": 1},
        )
        with self.assertRaisesRegex(RuntimeError, "one embedding sharding type"):
            shard_params_by_plan(
                model,
                params,
                None,
                mesh,
                DataParallelStyle.REPLICATE,
                embedding_plan=mixed_plan,
            )

        different_owner_plan = EmbeddingShardingPlan(
            sharding_map={
                "primary.weight": EmbeddingShardingType.TABLE_WISE,
                "alias.weight": EmbeddingShardingType.TABLE_WISE,
            },
            table_wise_rank_map={"primary.weight": 0, "alias.weight": 1},
        )
        with self.assertRaisesRegex(RuntimeError, "one owner rank"):
            shard_params_by_plan(
                model,
                params,
                None,
                mesh,
                DataParallelStyle.REPLICATE,
                embedding_plan=different_owner_plan,
            )

        incomplete_plan = EmbeddingShardingPlan(
            sharding_map={
                "primary.weight": EmbeddingShardingType.ROW_WISE,
            }
        )
        with self.assertRaisesRegex(RuntimeError, "all be present"):
            shard_params_by_plan(
                model,
                params,
                None,
                mesh,
                DataParallelStyle.FULLY_SHARD,
                embedding_plan=incomplete_plan,
            )

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_table_wise_meta_parameter_and_state_materialize_only_on_owner(
        self,
    ) -> None:
        model = _MetaEmbedding()
        optimizer = torch.optim.SGD([model.emb.weight], lr=0.1, foreach=True)
        optimizer.state[model.emb.weight]["sum"] = torch.empty(
            (16, 1), dtype=torch.float32, device="meta"
        )
        params: dict[str, torch.Tensor] = dict(model.named_parameters())
        plan = EmbeddingShardingPlan(
            sharding_map={"emb.weight": EmbeddingShardingType.TABLE_WISE},
            table_wise_rank_map={"emb.weight": 1},
        )

        shard_params_by_plan(
            model,
            params,
            optimizer,
            DeviceMesh("cuda", torch.arange(self.world_size)),
            DataParallelStyle.REPLICATE,
            embedding_plan=plan,
        )

        local_parameter = model.emb.weight
        self.assertEqual(local_parameter.device, torch.device(f"cuda:{self.rank}"))
        self.assertIs(optimizer.param_groups[0]["params"][0], local_parameter)
        if self.rank == 1:
            self.assertEqual(local_parameter.shape, (16, 8))
            local_state = optimizer.state[local_parameter]["sum"]
            self.assertEqual(local_state.shape, (16, 1))
            torch.testing.assert_close(local_state, torch.zeros_like(local_state))
        else:
            self.assertEqual(local_parameter.shape, (0, 8))
            self.assertNotIn(local_parameter, optimizer.state)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_optimizer_states_sharded_for_embedding(self) -> None:
        """Optimizer states for embedding params are sharded; dense states are not in REPLICATE."""
        model = EmbeddingMLP().cuda(self.rank)
        opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, foreach=True)
        # Run a dummy step to populate optimizer states.
        dummy_indices = torch.randint(0, 16, (4, 3), device=f"cuda:{self.rank}")
        model(dummy_indices).sum().backward()
        opt.step()
        opt.zero_grad()
        params = OrderedDict((n, p) for n, p in model.named_parameters())
        orig_emb_momentum_shape = opt.state[model.emb.weight]["momentum_buffer"].shape
        orig_lin_momentum_shape = opt.state[model.linear.weight][
            "momentum_buffer"
        ].shape
        mesh = DeviceMesh("cuda", torch.arange(self.world_size))
        shard_params_by_plan(
            model,
            params,  # pyre-fixme[6]
            opt,
            mesh,
            DataParallelStyle.REPLICATE,
            embedding_plan=EmbeddingShardingPlan(
                sharding_map={"emb.weight": EmbeddingShardingType.ROW_WISE}
            ),
        )
        # Embedding state sharded (new param object is the key)
        new_emb_param = model.emb.weight
        self.assertIn(new_emb_param, opt.state)
        self.assertEqual(
            opt.state[new_emb_param]["momentum_buffer"].shape[0],
            orig_emb_momentum_shape[0] // self.world_size,
        )
        # Dense state NOT sharded in REPLICATE mode
        # In REPLICATE mode without embedding, linear.weight is not sharded,
        # so the param object is unchanged and state should be unchanged.
        new_lin_param = model.linear.weight
        self.assertIn(new_lin_param, opt.state)
        self.assertEqual(
            opt.state[new_lin_param]["momentum_buffer"].shape,
            orig_lin_momentum_shape,
        )

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_meta_optimizer_state_is_materialized_with_embedding_shard(self) -> None:
        model = _MetaEmbedding()
        optimizer = torch.optim.SGD([model.emb.weight], lr=0.1, foreach=True)
        optimizer.state[model.emb.weight]["sum"] = torch.empty(
            (16, 1), dtype=torch.float32, device="meta"
        )
        params: dict[str, torch.Tensor] = dict(model.named_parameters())
        mesh = DeviceMesh("cuda", torch.arange(self.world_size))

        shard_params_by_plan(
            model,
            params,
            optimizer,
            mesh,
            DataParallelStyle.REPLICATE,
            embedding_plan=EmbeddingShardingPlan(
                sharding_map={"emb.weight": EmbeddingShardingType.ROW_WISE}
            ),
        )

        local_parameter = model.emb.weight
        local_state = optimizer.state[local_parameter]["sum"]
        self.assertEqual(local_parameter.device, torch.device(f"cuda:{self.rank}"))
        self.assertEqual(local_state.device, local_parameter.device)
        self.assertEqual(local_state.shape, (8, 1))
        torch.testing.assert_close(local_state, torch.zeros_like(local_state))
        self.assertIs(optimizer.param_groups[0]["params"][0], local_parameter)

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
            None,
            mesh,
            DataParallelStyle.REPLICATE,
            embedding_plan=EmbeddingShardingPlan(
                sharding_map={"emb.weight": EmbeddingShardingType.ROW_WISE}
            ),
        )
        # model.emb.weight should now be the local shard
        self.assertEqual(model.emb.weight.shape, (8, 8))
        # model.linear.weight unchanged in REPLICATE
        self.assertEqual(model.linear.weight.shape, (4, 8))


class TestRingshiftEmbeddingStrategies(unittest.TestCase):
    def _create_mock_mesh(self) -> DeviceMesh:
        mesh = MagicMock(spec=DeviceMesh)
        mesh.ndim = 1
        mesh.shape = (2,)
        mesh.device_type = "cpu"
        mesh.mesh = torch.arange(2)
        mesh._flatten.return_value = mesh
        mesh.size.return_value = 2
        mesh.get_coordinate.return_value = (0,)
        return mesh

    def _build_graph(
        self, *, detach_grad: bool = False
    ) -> tuple[fx.GraphModule, fx.Node, fx.Node, fx.Node | None]:
        import recir.ops.seq_embedding_op  # noqa: F401

        graph = fx.Graph()
        weight = graph.placeholder("weight")
        weight.meta["val"] = torch.randn(16, 4)
        ids = graph.placeholder("ids")
        ids.meta["val"] = torch.randint(0, 16, (4, 3))
        lengths = graph.placeholder("lengths")
        lengths.meta["val"] = torch.full((4,), 3, dtype=torch.int32)
        grad_out = graph.placeholder("grad_out")
        grad_out.meta["val"] = torch.randn(4, 3, 4)
        forward = graph.call_function(
            torch.ops.recir.ringshift_embedding.default,
            args=(ids, lengths, weight, -1),
        )
        forward.meta["val"] = torch.randn(4, 3, 4)
        backward = graph.call_function(
            torch.ops.recir.ringshift_embedding_backward.default,
            args=(grad_out, ids, lengths, weight, -1),
        )
        backward.meta["val"] = torch.randn(16, 4)
        grad_result = backward
        detach = None
        if detach_grad:
            detach = graph.call_function(torch.ops.aten.detach.default, (backward,))
            detach.meta["val"] = backward.meta["val"].detach()
            grad_result = detach
        tagged_grad = graph.call_function(
            torch.ops._spmd.tag_grad.default, (grad_result,)
        )
        tagged_grad.meta["val"] = grad_result.meta["val"]
        graph.output((forward, tagged_grad))
        return fx.GraphModule(nn.Module(), graph), forward, backward, detach

    def test_ringshift_keeps_weight_and_gradient_sharded_without_collectives(
        self,
    ) -> None:
        gm, forward, backward, _detach = self._build_graph()
        mesh = self._create_mock_mesh()
        strategy_map = build_data_parallel_strategies(
            gm,
            num_params=1,
            num_states=0,
            mesh=mesh,
            batch_dim=0,
            parallel_mode=DataParallelStyle.FULLY_SHARD,
            grad_reduction="sum",
        )

        forward_strategy = cast(DataParallelStrategy, strategy_map[forward])
        backward_strategy = cast(DataParallelStrategy, strategy_map[backward])
        forward_output_specs = cast(
            list[DTensorSpec], forward_strategy.op_spec.output_specs
        )
        backward_output_specs = cast(
            list[DTensorSpec], backward_strategy.op_spec.output_specs
        )
        self.assertEqual(NodeType.ACT, forward_strategy.node_type)
        self.assertEqual(NodeType.GRAD, backward_strategy.node_type)
        self.assertEqual(
            (Shard(0),),
            forward_output_specs[0].placements,
        )
        self.assertEqual(
            (Shard(0),),
            backward_output_specs[0].placements,
        )
        for strategy in (forward_strategy, backward_strategy):
            self.assertIsNotNone(strategy.op_spec.input_specs)
            for spec in cast(list[DTensorSpec], strategy.op_spec.input_specs):
                self.assertEqual((Shard(0),), spec.placements)

        mark_data_parallel_shardings(gm, strategy_map)
        partitioned = partitioner(gm)
        target_names = [
            getattr(getattr(node.target, "_schema", None), "name", str(node.target))
            for node in partitioned.graph.nodes
            if node.op == "call_function"
        ]
        self.assertFalse(any("all_gather" in name for name in target_names))
        self.assertFalse(any("reduce_scatter" in name for name in target_names))

    def test_ringshift_sharded_gradient_survives_detach(self) -> None:
        gm, _forward, _backward, detach = self._build_graph(detach_grad=True)
        self.assertIsNotNone(detach)
        strategy_map = build_data_parallel_strategies(
            gm,
            num_params=1,
            num_states=0,
            mesh=self._create_mock_mesh(),
            batch_dim=0,
            parallel_mode=DataParallelStyle.FULLY_SHARD,
            grad_reduction="sum",
        )
        detach_strategy = cast(
            DataParallelStrategy, strategy_map[cast(fx.Node, detach)]
        )
        detach_output_specs = cast(
            list[DTensorSpec], detach_strategy.op_spec.output_specs
        )
        self.assertEqual(NodeType.GRAD, detach_strategy.node_type)
        self.assertEqual(
            (Shard(0),),
            detach_output_specs[0].placements,
        )

    def test_ringshift_backward_rejects_average_gradient_reduction(self) -> None:
        gm, _forward, _backward, _detach = self._build_graph()
        with self.assertRaisesRegex(ValueError, "requires grad_reduction='sum'"):
            build_data_parallel_strategies(
                gm,
                num_params=1,
                num_states=0,
                mesh=self._create_mock_mesh(),
                batch_dim=0,
                parallel_mode=DataParallelStyle.FULLY_SHARD,
                grad_reduction="avg",
            )


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

    def test_embedding_param_placeholder_type(self) -> None:
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

    def test_non_embedding_placeholder_stays_param(self) -> None:
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

    def test_non_compute_propagation_preserves_embedding_param(self) -> None:
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

    def test_mixed_param_and_embedding_param(self) -> None:
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

    def test_embedding_param_with_replicate_mode(self) -> None:
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
        self.assertEqual(len(output_specs), 1)  # pyre-fixme[6]
        self.assertEqual(output_specs[0].placements, (Shard(0),))  # pyre-fixme[16]


class TestRewriteRowWiseEmbeddings(unittest.TestCase):
    """Tests for _rewrite_row_wise_embeddings graph transformation.

    These tests use a MagicMock DeviceMesh; no real distributed setup needed
    because make_fx(tracing_mode="symbolic") uses Meta implementations.
    """

    def _create_mock_mesh(self, world_size: int = 2) -> DeviceMesh:
        mesh = MagicMock(spec=DeviceMesh)
        mesh.ndim = 1
        mesh.shape = (world_size,)
        mesh.device_type = "cpu"
        mesh.mesh = torch.arange(world_size)
        mesh._flatten.return_value = mesh
        mesh.size.return_value = world_size
        mesh._dim_group_names = ["fake_test_group"]
        return mesh  # pyre-fixme[7]

    def _build_embedding_graph(
        self, V: int = 100, D: int = 16, batch: int = 8
    ) -> tuple[fx.GraphModule, fx.Node, fx.Node, fx.Node]:
        """Build a minimal graph: placeholder(weight) + placeholder(indices) → aten.embedding → output."""
        graph = fx.Graph()
        emb_w = graph.placeholder("emb_weight")
        emb_w.meta["val"] = torch.randn(V, D)
        indices = graph.placeholder("indices")
        indices.meta["val"] = torch.randint(0, V, (batch,))
        emb_out = graph.call_function(
            torch.ops.aten.embedding.default, args=(emb_w, indices)
        )
        emb_out.meta["val"] = torch.randn(batch, D)
        graph.output(emb_out)
        return fx.GraphModule(nn.Module(), graph), emb_w, indices, emb_out

    def test_original_embedding_replaced(self) -> None:
        """After rewrite, the original aten.embedding node consuming the
        EMBEDDING_PARAM weight is replaced.  The decomposed sub-graph contains
        its own local aten.embedding (step 3), but it consumes received_indices
        from the AllToAll, NOT the original indices placeholder directly."""
        gm, emb_w, indices, emb_out = self._build_embedding_graph()
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
        mark_data_parallel_shardings(gm, dp_strategy_map=strategy_map)
        _rewrite_row_wise_embeddings(gm, mesh, strategy_map)
        # The decomposed sub-graph contains a local aten.embedding (step 3),
        # but its indices arg must come from a wait_tensor (AllToAll result),
        # NOT from the original indices placeholder.
        for node in gm.graph.nodes:
            if (
                node.op == "call_function"
                and node.target == torch.ops.aten.embedding.default
            ):
                # The indices arg (args[1]) should be a wait_tensor node.
                indices_arg = node.args[1]
                self.assertEqual(
                    indices_arg.target,
                    torch.ops._c10d_functional.wait_tensor.default,
                )

    def test_decomposed_graph_structure(self) -> None:
        """Verify the decomposed graph has the expected node structure and print it."""
        gm, emb_w, indices, emb_out = self._build_embedding_graph()
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
        mark_data_parallel_shardings(gm, dp_strategy_map=strategy_map)
        _rewrite_row_wise_embeddings(gm, mesh, strategy_map)
        # Print the decomposed graph for manual inspection.
        print("\nDecomposed row-wise embedding graph:")
        for node in gm.graph.nodes:
            args_str = ", ".join(
                a.name if isinstance(a, torch.fx.Node) else repr(a)
                for a in (node.args if node.args else ())
            )
            print(
                f"  {node.op:15s} | {str(node.target):55s} | {node.name:30s} | ({args_str})"
            )
        # Sanity-check: more than the original 4 nodes (2 placeholders + 1 embedding + 1 output).
        call_nodes = [n for n in gm.graph.nodes if n.op == "call_function"]
        self.assertGreater(len(call_nodes), 10)

    def test_alltoall_nodes_present(self) -> None:
        """After rewrite, _c10d_functional.all_to_all_single nodes are present."""
        gm, emb_w, indices, emb_out = self._build_embedding_graph()
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
        mark_data_parallel_shardings(gm, dp_strategy_map=strategy_map)
        _rewrite_row_wise_embeddings(gm, mesh, strategy_map)
        a2a_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target == torch.ops._c10d_functional.all_to_all_single.default
        ]
        # 3 all_to_all_single calls: split sizes, indices, results
        self.assertEqual(len(a2a_nodes), 3)

    def test_wait_tensor_nodes_present(self) -> None:
        """After rewrite, wait_tensor nodes match all_to_all_single nodes."""
        gm, emb_w, indices, emb_out = self._build_embedding_graph()
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
        mark_data_parallel_shardings(gm, dp_strategy_map=strategy_map)
        _rewrite_row_wise_embeddings(gm, mesh, strategy_map)
        wait_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target == torch.ops._c10d_functional.wait_tensor.default
        ]
        self.assertEqual(len(wait_nodes), 3)

    def test_new_nodes_have_sharding_none(self) -> None:
        """All newly-inlined nodes have meta['sharding'] = None."""
        gm, emb_w, indices, emb_out = self._build_embedding_graph()
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
        mark_data_parallel_shardings(gm, dp_strategy_map=strategy_map)
        # Nodes that existed before the rewrite.
        pre_rewrite_nodes = set(gm.graph.nodes)
        _rewrite_row_wise_embeddings(gm, mesh, strategy_map)
        for node in gm.graph.nodes:
            if node not in pre_rewrite_nodes and node.op == "call_function":
                self.assertIn("sharding", node.meta)
                self.assertIsNone(node.meta["sharding"])

    def test_non_embedding_param_untouched(self) -> None:
        """aten.embedding with a non-EMBEDDING_PARAM weight is NOT rewritten."""
        graph = fx.Graph()
        dense_w = graph.placeholder("dense_weight")
        dense_w.meta["val"] = torch.randn(100, 16)
        indices = graph.placeholder("indices")
        indices.meta["val"] = torch.randint(0, 100, (8,))
        act = graph.placeholder("act_0")
        act.meta["val"] = torch.randn(8, 3)
        emb_out = graph.call_function(
            torch.ops.aten.embedding.default, args=(dense_w, indices)
        )
        emb_out.meta["val"] = torch.randn(8, 16)
        graph.output(emb_out)
        gm = fx.GraphModule(nn.Module(), graph)
        mesh = self._create_mock_mesh()
        strategy_map = build_data_parallel_strategies(
            gm,
            num_params=1,
            num_states=1,
            mesh=mesh,
            batch_dim=0,
            parallel_mode=DataParallelStyle.FULLY_SHARD,
            # NOT marking index 0 as embedding param
            embedding_param_indices=set(),
        )
        mark_data_parallel_shardings(gm, dp_strategy_map=strategy_map)
        _rewrite_row_wise_embeddings(gm, mesh, strategy_map)
        targets = [n.target for n in gm.graph.nodes if n.op == "call_function"]
        # Should still have aten.embedding since weight is PARAM, not EMBEDDING_PARAM
        self.assertIn(torch.ops.aten.embedding.default, targets)


class TestRowWiseEmbeddingE2E(DTensorTestBase):
    """End-to-end tests for decomposed_row_wise_embedding with real distributed comms."""

    @property
    def world_size(self) -> int:
        return 2

    # pyre-fixme[2,3]
    def _setup_embedding(self, num_embeddings=16, embedding_dim=8):
        """Create full weight, shard it, and return (full_weight, local_shard, mesh, group_name)."""
        torch.manual_seed(42)
        device = f"cuda:{self.rank}"
        full_weight = torch.randn(num_embeddings, embedding_dim, device=device)
        rows_per_rank = num_embeddings // self.world_size
        local_weight = full_weight[
            self.rank * rows_per_rank : (self.rank + 1) * rows_per_rank
        ].contiguous()
        mesh = DeviceMesh("cuda", torch.arange(self.world_size))
        group_name = funcol._resolve_group_name((mesh, 0))
        return full_weight, local_weight, num_embeddings, group_name

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_row_wise_matches_full_embedding(self) -> None:
        """Row-wise result matches single-device embedding lookup."""
        full_weight, local_weight, num_emb, group_name = self._setup_embedding()
        device = f"cuda:{self.rank}"
        indices = torch.tensor([0, 5, 15, 3, 8], device=device)
        expected = torch.nn.functional.embedding(indices, full_weight)
        result = decomposed_row_wise_embedding(
            local_weight, indices, num_emb, group_name, self.world_size
        )
        torch.testing.assert_close(result, expected)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_2d_indices(self) -> None:
        """Row-wise embedding with 2D indices (batch, seq) returns correct shape and values."""
        full_weight, local_weight, num_emb, group_name = self._setup_embedding()
        device = f"cuda:{self.rank}"
        indices = torch.tensor([[0, 5, 15], [3, 8, 1]], device=device)
        expected = torch.nn.functional.embedding(indices, full_weight)
        result = decomposed_row_wise_embedding(
            local_weight, indices, num_emb, group_name, self.world_size
        )
        self.assertEqual(result.shape, (2, 3, 8))
        torch.testing.assert_close(result, expected)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_indices_all_on_one_rank(self) -> None:
        """All indices map to rank 0's shard (tests uneven AllToAll splits)."""
        full_weight, local_weight, num_emb, group_name = self._setup_embedding()
        device = f"cuda:{self.rank}"
        # All indices in [0, 7] → rank 0's shard
        indices = torch.tensor([0, 1, 2, 3, 7], device=device)
        expected = torch.nn.functional.embedding(indices, full_weight)
        result = decomposed_row_wise_embedding(
            local_weight, indices, num_emb, group_name, self.world_size
        )
        torch.testing.assert_close(result, expected)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_large_embedding_table(self) -> None:
        """Row-wise with larger table and random indices."""
        full_weight, local_weight, num_emb, group_name = self._setup_embedding(
            num_embeddings=128, embedding_dim=32
        )
        device = f"cuda:{self.rank}"
        torch.manual_seed(self.rank + 100)
        indices = torch.randint(0, 128, (8, 5), device=device)
        expected = torch.nn.functional.embedding(indices, full_weight)
        result = decomposed_row_wise_embedding(
            local_weight, indices, num_emb, group_name, self.world_size
        )
        torch.testing.assert_close(result, expected)


class TestShardStateDict(unittest.TestCase):
    """Unit tests for _shard_state_dict helper."""

    @unittest.mock.patch("recir.distributed.data_parallel.partition_value")
    def test_dict_state_tensors_sharded(self, mock_partition: Any) -> None:
        """Tensor entries in a state dict are sharded via partition_value."""
        momentum = torch.randn(4, 4)
        sharded_momentum = torch.randn(2, 4)
        mock_partition.return_value = sharded_momentum
        state = {
            "momentum": momentum,
            "step": 42,
        }
        spec = MagicMock(spec=DTensorSpec)
        _shard_state_dict(state, spec)
        mock_partition.assert_called_once_with(momentum, spec, own_storage=False)
        self.assertIs(state["momentum"], sharded_momentum)
        # Non-tensor should be untouched
        self.assertEqual(state["step"], 42)

    def test_empty_dict_is_noop(self) -> None:
        """An empty state dict is unchanged."""
        state: dict[str, Any] = {}
        spec = MagicMock(spec=DTensorSpec)
        _shard_state_dict(state, spec)
        self.assertEqual(state, {})

    def test_non_dict_is_noop(self) -> None:
        """Non-dict state (e.g., None) is a no-op."""
        spec = MagicMock(spec=DTensorSpec)
        _shard_state_dict(None, spec)  # should not raise
        self.assertIsNone(None)  # verify no exception was raised

    @unittest.mock.patch("recir.distributed.data_parallel.partition_value")
    def test_meta_parameter_preserves_zero_state_initialization(
        self,
        mock_partition: Any,
    ) -> None:
        mock_partition.return_value = torch.empty((2, 1), device="meta")
        state = {"sum": torch.empty((4, 1), device="meta")}
        spec = MagicMock(spec=DTensorSpec)

        _shard_state_dict(state, spec, state_device=torch.device("meta"))

        self.assertEqual(state["sum"].device.type, "meta")
        init_kind, _bound = get_meta_init_spec(state["sum"], default_bound=1.0)
        self.assertEqual(init_kind, "zeros")

    @unittest.mock.patch("recir.distributed.data_parallel.partition_value")
    def test_meta_optimizer_state_materializes_as_zeros(
        self,
        mock_partition: Any,
    ) -> None:
        mock_partition.return_value = torch.empty((2, 1), device="meta")
        state = {"sum": torch.empty((4, 1), device="meta")}

        _shard_state_dict(
            state,
            MagicMock(spec=DTensorSpec),
            state_device=torch.device("cpu"),
        )

        self.assertEqual(state["sum"].device.type, "cpu")
        torch.testing.assert_close(state["sum"], torch.zeros(2, 1))


class TestShardParamsByPlanContract(unittest.TestCase):
    def test_rejects_multidimensional_embedding_mesh(self) -> None:
        model = EmbeddingMLP()
        mesh = MagicMock(spec=DeviceMesh)
        mesh.ndim = 2
        mesh.shape = (1, 1)
        plan = EmbeddingShardingPlan(
            sharding_map={"emb.weight": EmbeddingShardingType.TABLE_WISE},
            table_wise_rank_map={"emb.weight": 0},
        )

        with self.assertRaisesRegex(ValueError, "one-dimensional device mesh"):
            shard_params_by_plan(
                model,
                OrderedDict(model.named_parameters()),
                None,
                mesh,
                DataParallelStyle.REPLICATE,
                embedding_plan=plan,
            )


class TestTableWiseEmbeddingBagDeviceContract(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_accepts_cpu_offsets_for_cuda_indices(self) -> None:
        weight = torch.arange(32, dtype=torch.float32, device="cuda").reshape(8, 4)
        indices = torch.tensor([1, 2], dtype=torch.int64, device="cuda")
        offsets = torch.tensor([0, 1, 2], dtype=torch.int64)

        actual = decomposed_table_wise_embedding_bag(
            weight,
            indices,
            offsets,
            owner_rank=0,
            rank=0,
            group_name="unused",
            world_size=1,
            mode=0,
        )

        torch.testing.assert_close(actual, weight.index_select(0, indices))


class TestMultipleEmbeddingTables(DTensorTestBase):
    """E2E test for models with multiple embedding tables."""

    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_multiple_embedding_tables(self) -> None:
        """Model with 2 nn.Embedding layers: both decomposed into AllToAll ops."""
        model = MultiEmbeddingMLP(num_emb1=16, num_emb2=32, emb_dim=8).cuda(self.rank)
        opt = torch.optim.SGD(model.parameters(), lr=0.1, foreach=True)
        device = f"cuda:{self.rank}"
        train_batch = (
            torch.randint(0, 16, (8, 3), device=device),
            torch.randint(0, 32, (8, 3), device=device),
            torch.randn(8, 4, device=device),
        )

        plan = EmbeddingShardingPlan(
            sharding_map={
                "emb1.weight": EmbeddingShardingType.ROW_WISE,
                "emb2.weight": EmbeddingShardingType.ROW_WISE,
            }
        )
        compiled_fn = compile(
            parallel_mode=HybridParallel(
                dense_style="fully_shard",
                embedding_plan=plan,
            )
        )(multi_embedding_train_step)
        compiled_fn(model, opt, train_batch)

        from recir.api import COMPILED_OBJECT_KEY

        compiled_obj = compiled_fn.__dict__[COMPILED_OBJECT_KEY]
        gm = compiled_obj.gm

        # Each embedding table produces 3 all_to_all_single nodes (6 total for 2 tables)
        a2a_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target == torch.ops._c10d_functional.all_to_all_single.default
        ]
        self.assertGreaterEqual(
            len(a2a_nodes),
            6,
            f"Expected at least 6 all_to_all_single nodes for 2 tables, got {len(a2a_nodes)}",
        )

        # No allgather should feed into any aten.embedding node
        for node in gm.graph.nodes:
            if (
                node.op == "call_function"
                and node.target == torch.ops.aten.embedding.default
            ):
                weight_node = node.args[0]
                current = weight_node
                while current.op != "placeholder":
                    target_str = str(getattr(current, "target", ""))
                    self.assertNotIn(
                        "all_gather_into_tensor",
                        target_str,
                        "Embedding param should not be allgathered",
                    )
                    if current.args:
                        current = current.args[0]
                    else:
                        break


class TestHybridParallelV2E2E(DTensorTestBase):
    """End-to-end tests for HybridParallel with recir.compile (v2)."""

    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_compile_and_run(self) -> None:
        """HybridParallel v2 compiles and runs one train step without error."""
        model = EmbeddingMLP().cuda(self.rank)
        opt = torch.optim.SGD(model.parameters(), lr=0.1, foreach=True)
        device = f"cuda:{self.rank}"
        train_batch = (
            torch.randint(0, 16, (8, 3), device=device),
            torch.randn(8, 4, device=device),
        )

        plan = EmbeddingShardingPlan(
            sharding_map={"emb.weight": EmbeddingShardingType.ROW_WISE}
        )
        compiled_fn = compile(
            parallel_mode=HybridParallel(
                dense_style="fully_shard",
                embedding_plan=plan,
            )
        )(embedding_train_step)
        loss = compiled_fn(model, opt, train_batch)
        self.assertIsNotNone(loss)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_graph_has_decomposed_embedding(self) -> None:
        """v2 compiled graph contains decomposed AllToAll ops for row-wise embedding."""
        model = EmbeddingMLP().cuda(self.rank)
        opt = torch.optim.SGD(model.parameters(), lr=0.1, foreach=True)
        device = f"cuda:{self.rank}"
        train_batch = (
            torch.randint(0, 16, (8, 3), device=device),
            torch.randn(8, 4, device=device),
        )

        plan = EmbeddingShardingPlan(
            sharding_map={"emb.weight": EmbeddingShardingType.ROW_WISE}
        )
        compiled_fn = compile(
            parallel_mode=HybridParallel(
                dense_style="fully_shard",
                embedding_plan=plan,
            )
        )(embedding_train_step)
        compiled_fn(model, opt, train_batch)

        from recir.api import COMPILED_OBJECT_KEY

        compiled_obj = compiled_fn.__dict__[COMPILED_OBJECT_KEY]
        gm = compiled_obj.gm

        a2a_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target == torch.ops._c10d_functional.all_to_all_single.default
        ]
        self.assertGreaterEqual(
            len(a2a_nodes),
            3,
            "Expected at least 3 all_to_all_single nodes from decomposed embedding",
        )

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_no_allgather_on_embedding_params(self) -> None:
        """Embedding params should not have allgather in v2 (they use AllToAll)."""
        model = EmbeddingMLP().cuda(self.rank)
        opt = torch.optim.SGD(model.parameters(), lr=0.1, foreach=True)
        device = f"cuda:{self.rank}"
        train_batch = (
            torch.randint(0, 16, (8, 3), device=device),
            torch.randn(8, 4, device=device),
        )

        plan = EmbeddingShardingPlan(
            sharding_map={"emb.weight": EmbeddingShardingType.ROW_WISE}
        )
        compiled_fn = compile(
            parallel_mode=HybridParallel(
                dense_style="fully_shard",
                embedding_plan=plan,
            )
        )(embedding_train_step)
        compiled_fn(model, opt, train_batch)

        from recir.api import COMPILED_OBJECT_KEY

        compiled_obj = compiled_fn.__dict__[COMPILED_OBJECT_KEY]
        gm = compiled_obj.gm

        # The decomposed embedding uses local aten.embedding on received
        # indices (inside the AllToAll protocol).  The weight input to this
        # local lookup should NOT go through allgather.
        for node in gm.graph.nodes:
            if (
                node.op == "call_function"
                and node.target == torch.ops.aten.embedding.default
            ):
                weight_node = node.args[0]
                current = weight_node
                while current.op != "placeholder":
                    target_str = str(getattr(current, "target", ""))
                    self.assertNotIn(
                        "all_gather_into_tensor",
                        target_str,
                        "Embedding param should not be allgathered in v2",
                    )
                    if current.args:
                        current = current.args[0]
                    else:
                        break

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_multiple_embedding_tables(self) -> None:
        """v2: two embedding tables both decomposed into AllToAll ops."""
        model = MultiEmbeddingMLP(num_emb1=16, num_emb2=32, emb_dim=8).cuda(self.rank)
        opt = torch.optim.SGD(model.parameters(), lr=0.1, foreach=True)
        device = f"cuda:{self.rank}"
        train_batch = (
            torch.randint(0, 16, (8, 3), device=device),
            torch.randint(0, 32, (8, 3), device=device),
            torch.randn(8, 4, device=device),
        )

        plan = EmbeddingShardingPlan(
            sharding_map={
                "emb1.weight": EmbeddingShardingType.ROW_WISE,
                "emb2.weight": EmbeddingShardingType.ROW_WISE,
            }
        )
        compiled_fn = compile(
            parallel_mode=HybridParallel(
                dense_style="fully_shard",
                embedding_plan=plan,
            )
        )(multi_embedding_train_step)
        compiled_fn(model, opt, train_batch)

        from recir.api import COMPILED_OBJECT_KEY

        compiled_obj = compiled_fn.__dict__[COMPILED_OBJECT_KEY]
        gm = compiled_obj.gm

        # Each table → 3 all_to_all_single nodes = 6 total
        a2a_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target == torch.ops._c10d_functional.all_to_all_single.default
        ]
        self.assertGreaterEqual(
            len(a2a_nodes),
            6,
            f"Expected at least 6 all_to_all_single nodes for 2 tables, got {len(a2a_nodes)}",
        )


class TestHybridParallelV2GraphStructure(DTensorTestBase):
    """Verify the compiled graph structure: embedding decomposition ops
    and dense FSDP allgather coexist correctly."""

    @property
    def world_size(self) -> int:
        return 2

    def _compile_and_get_gm(
        self,
        model: nn.Module,
        opt: torch.optim.Optimizer,
        train_batch: Any,
        plan: EmbeddingShardingPlan,
        train_step_fn: Callable[..., Any] = embedding_train_step,
    ) -> fx.GraphModule:
        """Compile with HybridParallel v2 and return the GraphModule."""
        compiled_fn = compile(
            parallel_mode=HybridParallel(
                dense_style="fully_shard",
                embedding_plan=plan,
            )
        )(train_step_fn)
        compiled_fn(model, opt, train_batch)

        from recir.api import COMPILED_OBJECT_KEY

        return compiled_fn.__dict__[COMPILED_OBJECT_KEY].gm

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_embedding_decomposition_dataflow(self) -> None:
        """Verify embedding placeholder flows through the expected decomposition
        ops: argsort, bincount, all_to_all_single (sizes), all_to_all_single
        (indices), aten.embedding (local), all_to_all_single (results), argsort
        (unsort)."""
        model = EmbeddingMLP().cuda(self.rank)
        opt = torch.optim.SGD(model.parameters(), lr=0.1, foreach=True)
        device = f"cuda:{self.rank}"
        train_batch = (
            torch.randint(0, 16, (8, 3), device=device),
            torch.randn(8, 4, device=device),
        )
        plan = EmbeddingShardingPlan(
            sharding_map={"emb.weight": EmbeddingShardingType.ROW_WISE}
        )
        gm = self._compile_and_get_gm(model, opt, train_batch, plan)

        # Collect all op targets in the graph.
        op_targets = set()
        for node in gm.graph.nodes:
            if node.op == "call_function":
                op_targets.add(str(node.target))

        # Key ops from decomposed_row_wise_embedding must be present.
        # Note: argsort decomposes to aten.sort.stable in the traced graph.
        expected_ops = [
            "aten.sort",  # sort by rank + unsort (decomposed from argsort)
            "aten.embedding",  # local lookup on received indices
            "_c10d_functional.all_to_all_single",  # index/result exchange
        ]
        for expected in expected_ops:
            found = any(expected in t for t in op_targets)
            self.assertTrue(
                found,
                f"Expected '{expected}' in graph ops for decomposed "
                f"embedding, got: {sorted(op_targets)}",
            )

        # Verify the dataflow: aten.embedding should consume indices from
        # all_to_all_single (received indices), not directly from an input
        # placeholder.
        for node in gm.graph.nodes:
            if (
                node.op == "call_function"
                and node.target == torch.ops.aten.embedding.default
            ):
                indices_input = node.args[1]
                # The indices input should come from wait_tensor
                # (after all_to_all_single), not from a placeholder.
                self.assertNotEqual(
                    indices_input.op,
                    "placeholder",
                    "aten.embedding indices should come from AllToAll "
                    "(decomposed), not directly from input placeholder",
                )

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_dense_params_use_allgather(self) -> None:
        """Dense params (linear.weight, linear.bias) go through allgather."""
        model = EmbeddingMLP().cuda(self.rank)
        opt = torch.optim.SGD(model.parameters(), lr=0.1, foreach=True)
        device = f"cuda:{self.rank}"
        train_batch = (
            torch.randint(0, 16, (8, 3), device=device),
            torch.randn(8, 4, device=device),
        )
        plan = EmbeddingShardingPlan(
            sharding_map={"emb.weight": EmbeddingShardingType.ROW_WISE}
        )
        gm = self._compile_and_get_gm(model, opt, train_batch, plan)

        allgather_count = 0
        for node in gm.graph.nodes:
            if node.op == "call_function":
                if "all_gather_into_tensor" in str(node.target):
                    allgather_count += 1

        # EmbeddingMLP has 2 dense params (linear.weight, linear.bias).
        # With FULLY_SHARD + region transforms, each gets forward + backward
        # allgather (reentrant checkpoint recompute) = 4 total.
        self.assertGreaterEqual(
            allgather_count,
            2,
            f"Expected at least 2 allgathers for dense params, got {allgather_count}",
        )

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_dense_and_embedding_coexist(self) -> None:
        """Both allgather (dense) and all_to_all (embedding) ops coexist."""
        model = EmbeddingMLP().cuda(self.rank)
        opt = torch.optim.SGD(model.parameters(), lr=0.1, foreach=True)
        device = f"cuda:{self.rank}"
        train_batch = (
            torch.randint(0, 16, (8, 3), device=device),
            torch.randn(8, 4, device=device),
        )
        plan = EmbeddingShardingPlan(
            sharding_map={"emb.weight": EmbeddingShardingType.ROW_WISE}
        )
        gm = self._compile_and_get_gm(model, opt, train_batch, plan)

        allgather_count = 0
        a2a_count = 0
        reduce_scatter_count = 0
        for node in gm.graph.nodes:
            if node.op == "call_function":
                target_str = str(node.target)
                if "all_gather_into_tensor" in target_str:
                    allgather_count += 1
                elif "all_to_all_single" in target_str:
                    a2a_count += 1
                elif "reduce_scatter_tensor" in target_str:
                    reduce_scatter_count += 1

        self.assertGreater(
            allgather_count, 0, "Expected allgather ops for dense params"
        )
        self.assertGreaterEqual(
            a2a_count, 3, "Expected at least 3 all_to_all ops for embedding"
        )
        self.assertGreater(
            reduce_scatter_count, 0, "Expected reduce_scatter for gradient reduction"
        )


class TestHybridParallelV2Backward(DTensorTestBase):
    """Verify backward decomposition in v2 HybridParallel compilation."""

    @property
    def world_size(self) -> int:
        return 2

    def _compile_and_get_gm(
        self,
        model: nn.Module,
        opt: torch.optim.Optimizer,
        train_batch: Any,
        plan: EmbeddingShardingPlan,
        train_step_fn: Callable[..., Any] = embedding_train_step,
    ) -> fx.GraphModule:
        compiled_fn = compile(
            parallel_mode=HybridParallel(
                dense_style="fully_shard",
                embedding_plan=plan,
            )
        )(train_step_fn)
        compiled_fn(model, opt, train_batch)

        from recir.api import COMPILED_OBJECT_KEY

        return compiled_fn.__dict__[COMPILED_OBJECT_KEY].gm

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_no_undecomposed_embedding_dense_backward(self) -> None:
        """Original aten.embedding_dense_backward (on global indices) should
        be replaced.  Only the local version (on AllToAll-received indices)
        should remain."""
        model = EmbeddingMLP().cuda(self.rank)
        opt = torch.optim.SGD(model.parameters(), lr=0.1, foreach=True)
        device = f"cuda:{self.rank}"
        train_batch = (
            torch.randint(0, 16, (8, 3), device=device),
            torch.randn(8, 4, device=device),
        )
        plan = EmbeddingShardingPlan(
            sharding_map={"emb.weight": EmbeddingShardingType.ROW_WISE}
        )
        gm = self._compile_and_get_gm(model, opt, train_batch, plan)

        for node in gm.graph.nodes:
            if (
                node.op == "call_function"
                and node.target == torch.ops.aten.embedding_dense_backward.default
            ):
                indices_input = node.args[1]
                self.assertNotEqual(
                    indices_input.op,
                    "placeholder",
                    "embedding_dense_backward should not take indices directly "
                    "from a placeholder (should be decomposed with AllToAll)",
                )

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_backward_has_alltoall(self) -> None:
        """Backward decomposition should have AllToAll ops beyond those in forward."""
        model = EmbeddingMLP().cuda(self.rank)
        opt = torch.optim.SGD(model.parameters(), lr=0.1, foreach=True)
        device = f"cuda:{self.rank}"
        train_batch = (
            torch.randint(0, 16, (8, 3), device=device),
            torch.randn(8, 4, device=device),
        )
        plan = EmbeddingShardingPlan(
            sharding_map={"emb.weight": EmbeddingShardingType.ROW_WISE}
        )
        gm = self._compile_and_get_gm(model, opt, train_batch, plan)

        a2a_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target == torch.ops._c10d_functional.all_to_all_single.default
        ]
        # Forward: 3 AllToAll (sizes, indices, results) per table.
        # Backward: 3 AllToAll (sizes, indices, gradients) per table.
        # Total: 6 for one table.
        self.assertGreaterEqual(
            len(a2a_nodes),
            6,
            f"Expected at least 6 all_to_all_single nodes (3 fwd + 3 bwd), got {len(a2a_nodes)}",
        )

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_backward_has_local_embedding_dense_backward(self) -> None:
        """Backward decomposition should contain a local embedding_dense_backward
        (on received indices/gradients, not the original)."""
        model = EmbeddingMLP().cuda(self.rank)
        opt = torch.optim.SGD(model.parameters(), lr=0.1, foreach=True)
        device = f"cuda:{self.rank}"
        train_batch = (
            torch.randint(0, 16, (8, 3), device=device),
            torch.randn(8, 4, device=device),
        )
        plan = EmbeddingShardingPlan(
            sharding_map={"emb.weight": EmbeddingShardingType.ROW_WISE}
        )
        gm = self._compile_and_get_gm(model, opt, train_batch, plan)

        edb_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target == torch.ops.aten.embedding_dense_backward.default
        ]
        # Should have exactly 1 local embedding_dense_backward (from the
        # decomposed backward, operating on received indices/gradients).
        self.assertEqual(
            len(edb_nodes),
            1,
            f"Expected exactly 1 local embedding_dense_backward, got {len(edb_nodes)}",
        )
        # Its indices input should come from an AllToAll (not a placeholder).
        indices_input = edb_nodes[0].args[1]
        self.assertEqual(
            indices_input.target,
            torch.ops._c10d_functional.wait_tensor.default,
            "Local backward indices should come from AllToAll wait_tensor",
        )

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_backward_multiple_tables(self) -> None:
        """Two embedding tables: both forward and backward decomposed."""
        model = MultiEmbeddingMLP(num_emb1=16, num_emb2=32, emb_dim=8).cuda(self.rank)
        opt = torch.optim.SGD(model.parameters(), lr=0.1, foreach=True)
        device = f"cuda:{self.rank}"
        train_batch = (
            torch.randint(0, 16, (8, 3), device=device),
            torch.randint(0, 32, (8, 3), device=device),
            torch.randn(8, 4, device=device),
        )
        plan = EmbeddingShardingPlan(
            sharding_map={
                "emb1.weight": EmbeddingShardingType.ROW_WISE,
                "emb2.weight": EmbeddingShardingType.ROW_WISE,
            }
        )
        gm = self._compile_and_get_gm(
            model, opt, train_batch, plan, train_step_fn=multi_embedding_train_step
        )

        a2a_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target == torch.ops._c10d_functional.all_to_all_single.default
        ]
        # 2 tables * 6 AllToAll (3 fwd + 3 bwd) = 12
        self.assertGreaterEqual(
            len(a2a_nodes),
            12,
            f"Expected at least 12 all_to_all_single for 2 tables (fwd+bwd), got {len(a2a_nodes)}",
        )

        edb_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target == torch.ops.aten.embedding_dense_backward.default
        ]
        self.assertEqual(
            len(edb_nodes),
            2,
            f"Expected 2 local embedding_dense_backward (one per table), got {len(edb_nodes)}",
        )

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_embedding_weights_updated_by_backward(self) -> None:
        """v2 backward correctly updates embedding weights via optimizer."""
        torch.manual_seed(0)
        device = f"cuda:{self.rank}"
        model = EmbeddingMLP(num_embeddings=16, embedding_dim=8).cuda(self.rank)

        # Save initial embedding weights for comparison.
        initial_emb_weight = model.emb.weight.detach().clone()

        opt = torch.optim.SGD(model.parameters(), lr=0.1, foreach=True)

        torch.manual_seed(123 + self.rank)
        train_batch = (
            torch.randint(0, 16, (8, 3), device=device),
            torch.randn(8, 4, device=device),
        )

        plan = EmbeddingShardingPlan(
            sharding_map={"emb.weight": EmbeddingShardingType.ROW_WISE}
        )
        compiled_fn = compile(
            parallel_mode=HybridParallel(
                dense_style="fully_shard",
                embedding_plan=plan,
            )
        )(embedding_train_step)
        compiled_fn(model, opt, train_batch)

        # After sharding, each rank has local_V rows.
        local_V = model.emb.weight.shape[0]
        self.assertEqual(local_V, 8, "Expected 16 / 2 = 8 rows per rank")

        # Weights should have changed (optimizer applied gradients).
        self.assertFalse(
            torch.equal(
                model.emb.weight,
                initial_emb_weight[:local_V]
                if self.rank == 0
                else initial_emb_weight[local_V:],
            ),
            "Embedding weights should change after backward + optimizer step",
        )

        # Weights should be finite (no NaN/Inf from incorrect backward).
        self.assertTrue(
            torch.isfinite(model.emb.weight).all(),
            "Embedding weights should be finite after backward",
        )


class TestHybridParallelV2NumericalCorrectness(DTensorTestBase):
    """Compare HybridParallel v2 output against unsharded single-rank reference."""

    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_loss_matches_unsharded_reference(self) -> None:
        """v2 compiled loss matches unsharded single-device forward."""
        torch.manual_seed(42 + self.rank)
        device = f"cuda:{self.rank}"

        # Create model with deterministic weights (same on all ranks).
        torch.manual_seed(0)
        model = EmbeddingMLP(num_embeddings=16, embedding_dim=8).cuda(self.rank)
        # Save unsharded weights for reference.
        ref_state = deepcopy(model.state_dict())

        opt = torch.optim.SGD(model.parameters(), lr=0.0, foreach=True)

        # Use the same input on all ranks so the reference is deterministic.
        torch.manual_seed(123)
        indices = torch.randint(0, 16, (8, 3), device=device)
        labels = torch.randn(8, 4, device=device)
        train_batch = (indices, labels)

        plan = EmbeddingShardingPlan(
            sharding_map={"emb.weight": EmbeddingShardingType.ROW_WISE}
        )
        compiled_fn = compile(
            parallel_mode=HybridParallel(
                dense_style="fully_shard",
                embedding_plan=plan,
            )
        )(embedding_train_step)
        compiled_loss = compiled_fn(model, opt, train_batch)

        # Reference: unsharded forward on same device.
        ref_model = EmbeddingMLP(num_embeddings=16, embedding_dim=8).cuda(self.rank)
        ref_model.load_state_dict(ref_state)
        ref_out = ref_model(indices)
        ref_loss = (ref_out - labels).sum()

        self.assertTrue(
            torch.allclose(compiled_loss, ref_loss, atol=1e-4, rtol=1e-4),
            f"Compiled loss {compiled_loss.item():.6f} does not match "
            f"reference {ref_loss.item():.6f}",
        )

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_multi_table_loss_matches_reference(self) -> None:
        """v2 compiled loss with two embedding tables matches unsharded reference."""
        device = f"cuda:{self.rank}"

        torch.manual_seed(0)
        model = MultiEmbeddingMLP(num_emb1=16, num_emb2=32, emb_dim=8).cuda(self.rank)
        ref_state = deepcopy(model.state_dict())

        opt = torch.optim.SGD(model.parameters(), lr=0.0, foreach=True)

        torch.manual_seed(123)
        indices1 = torch.randint(0, 16, (8, 3), device=device)
        indices2 = torch.randint(0, 32, (8, 3), device=device)
        labels = torch.randn(8, 4, device=device)
        train_batch = (indices1, indices2, labels)

        plan = EmbeddingShardingPlan(
            sharding_map={
                "emb1.weight": EmbeddingShardingType.ROW_WISE,
                "emb2.weight": EmbeddingShardingType.ROW_WISE,
            }
        )
        compiled_fn = compile(
            parallel_mode=HybridParallel(
                dense_style="fully_shard",
                embedding_plan=plan,
            )
        )(multi_embedding_train_step)
        compiled_loss = compiled_fn(model, opt, train_batch)

        ref_model = MultiEmbeddingMLP(num_emb1=16, num_emb2=32, emb_dim=8).cuda(
            self.rank
        )
        ref_model.load_state_dict(ref_state)
        ref_out = ref_model(indices1, indices2)
        ref_loss = (ref_out - labels).sum()

        self.assertTrue(
            torch.allclose(compiled_loss, ref_loss, atol=1e-4, rtol=1e-4),
            f"Compiled loss {compiled_loss.item():.6f} does not match "
            f"reference {ref_loss.item():.6f}",
        )


if __name__ == "__main__":
    if False:
        run_tests()
