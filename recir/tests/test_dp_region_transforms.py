# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors

"""Integration tests for region transforms on the DataParallel v2 path.

Exercises ``compile(..., region_transforms=[RecomputeTransform(...)])``
end to end through ``_compile_data_parallel_v2``, verifying that:
  1. Region transforms are not silently dropped.
  2. Numerics match a DDP baseline.
  3. The compile succeeds with both replicate and fully_shard styles.

Run: buck2 test //recir/tests:test_dp_region_transforms
"""

import unittest
from copy import deepcopy

import torch
import torch.nn as nn
from parameterized import parameterized
from recir.api import compile
from recir.distributed.parallel_mode import DataParallel
from recir.region_transforms import RecomputeTransform, RegatherTransform
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

DIM = 32
BS = 8


class SimpleMLP(nn.Module):
    def __init__(self, dim: int = DIM):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def _make_train_step(loss_reduction):
    reduce_fn = torch.sum if loss_reduction == "sum" else torch.mean

    def train_step(mod, opt, x):
        opt.zero_grad()
        reduce_fn(mod(x)).backward()
        opt.step()

    return train_step


def _count_chain_ops(gm):
    """Count transpose/reshape chain ops (t/transpose) in a compiled graph."""
    count = 0
    for node in gm.graph.nodes:
        if node.op == "call_function":
            target = str(node.target)
            if ".t." in target or "transpose" in target:
                count += 1
    return count


def _run_ddp_baseline(rank, model, opt, inputs, loss_reduction):
    ddp_model = DDP(model, device_ids=[rank])
    ddp_opt = torch.optim.SGD(ddp_model.parameters(), lr=0.01, foreach=False)
    ddp_opt.load_state_dict(opt.state_dict())
    reduce_fn = torch.sum if loss_reduction == "sum" else torch.mean
    params_per_step = []
    for inp in inputs:
        ddp_opt.zero_grad()
        reduce_fn(ddp_model(inp)).backward()
        ddp_opt.step()
        params_per_step.append(
            {n: p.clone() for n, p in ddp_model.module.named_parameters()}
        )
    return params_per_step


class TestDPRegionTransform(DTensorTestBase):
    @property
    def world_size(self):
        return 1

    @parameterized.expand([("sum",), ("mean",)])
    @skip_if_lt_x_gpu(1)
    @with_comms
    def test_replicate_recompute_numerics(self, loss_reduction):
        """RecomputeTransform(relu) through replicate DP v2 must match DDP."""
        torch.manual_seed(42)
        model = SimpleMLP().cuda(self.rank)
        opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)

        ref_model = deepcopy(model)
        ref_opt = torch.optim.SGD(ref_model.parameters(), lr=0.01, foreach=False)

        inputs = [torch.randn(BS, DIM, device=f"cuda:{self.rank}") for _ in range(4)]

        ref_params = _run_ddp_baseline(
            self.rank, ref_model, ref_opt, inputs, loss_reduction
        )

        train_step = _make_train_step(loss_reduction)
        compiled_step = compile(
            parallel_mode=DataParallel(
                "replicate",
                loss_reduction=loss_reduction,
            ),
            region_transforms=[RecomputeTransform(submodule_fqns={"relu"})],
        )(train_step)

        for step, inp in enumerate(inputs):
            compiled_step(model, opt, inp)
            for (name, param), (_, baseline) in zip(
                model.named_parameters(),
                ref_params[step].items(),
            ):
                torch.testing.assert_close(
                    param,
                    baseline,
                    msg=f"Step {step}: replicate recompute mismatch for {name}",
                )

    @parameterized.expand([("sum",), ("mean",)])
    @skip_if_lt_x_gpu(1)
    @with_comms
    def test_fully_shard_recompute_compiles(self, loss_reduction):
        """RecomputeTransform(relu) through fully_shard DP v2 must compile
        and produce non-zero gradients."""
        torch.manual_seed(42)
        model = SimpleMLP().cuda(self.rank)
        opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)

        train_step = _make_train_step(loss_reduction)
        compiled_step = compile(
            parallel_mode=DataParallel(
                "fully_shard",
                loss_reduction=loss_reduction,
            ),
            region_transforms=[RecomputeTransform(submodule_fqns={"relu"})],
        )(train_step)

        params_before = {n: p.clone() for n, p in model.named_parameters()}
        x = torch.randn(BS, DIM, device=f"cuda:{self.rank}")
        compiled_step(model, opt, x)

        for name, param in model.named_parameters():
            self.assertFalse(
                torch.equal(param, params_before[name]),
                f"Parameter {name} unchanged after training step with "
                f"fully_shard + recompute — transforms may have been dropped",
            )


class TestDPStructural(DTensorTestBase):
    """Structural graph checks that only need a single GPU."""

    @property
    def world_size(self):
        return 1

    @skip_if_lt_x_gpu(1)
    @with_comms
    def test_no_recompute_baseline_differs_structurally(self):
        """A compile with RecomputeTransform should produce a different
        compiled graph than one without, proving the transform was applied."""
        torch.manual_seed(42)

        def _compile_and_get_relu_count(region_transforms):
            model = SimpleMLP().cuda(self.rank)
            opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
            train_step = _make_train_step("sum")
            compiled_step = compile(
                parallel_mode=DataParallel(
                    "replicate",
                    loss_reduction="sum",
                ),
                region_transforms=region_transforms,
            )(train_step)
            x = torch.randn(BS, DIM, device=f"cuda:{self.rank}")
            compiled_step(model, opt, x)
            gm = compiled_step.__dict__["_compiled_obj"].gm
            fn_nodes = sum(1 for n in gm.graph.nodes if n.op == "call_function")
            relu_ops = sum(
                1
                for n in gm.graph.nodes
                if n.op == "call_function" and "relu" in str(n.target)
            )
            return fn_nodes, relu_ops

        ref_nodes, ref_relu_ops = _compile_and_get_relu_count(None)
        recompute_nodes, recompute_relu_ops = _compile_and_get_relu_count(
            [RecomputeTransform(submodule_fqns={"relu"})]
        )
        self.assertGreater(
            recompute_nodes,
            ref_nodes,
            f"Expected recompute to add nodes to the graph "
            f"(reference={ref_nodes}, recompute={recompute_nodes}). "
            f"region_transforms may have been silently dropped.",
        )

        self.assertGreater(
            recompute_relu_ops,
            ref_relu_ops,
            "Expected recompute to duplicate relu ops in the backward graph "
            f"(reference={ref_relu_ops}, recompute={recompute_relu_ops}). "
            "relu recomputation may not be present in the backward pass.",
        )

    @skip_if_lt_x_gpu(1)
    @with_comms
    def test_no_regather_baseline_differs_structurally(self):
        """A compile with RegatherTransform should produce a different
        compiled graph than one without, proving the transform was applied."""
        torch.manual_seed(42)

        def _compile_and_get_chain_count(region_transforms):
            model = SimpleMLP().cuda(self.rank)
            opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
            train_step = _make_train_step("sum")
            compiled_step = compile(
                parallel_mode=DataParallel(
                    "replicate",
                    loss_reduction="sum",
                ),
                region_transforms=region_transforms,
            )(train_step)
            x = torch.randn(BS, DIM, device=f"cuda:{self.rank}")
            compiled_step(model, opt, x)
            gm = compiled_step.__dict__["_compiled_obj"].gm
            fn_nodes = sum(1 for n in gm.graph.nodes if n.op == "call_function")
            chain_ops = _count_chain_ops(gm)
            return fn_nodes, chain_ops

        ref_nodes, ref_chain_ops = _compile_and_get_chain_count(None)
        regather_nodes, regather_chain_ops = _compile_and_get_chain_count(
            [RegatherTransform()]
        )

        self.assertGreater(
            regather_nodes,
            ref_nodes,
            f"Expected regather to add nodes to the graph "
            f"(reference={ref_nodes}, regather={regather_nodes}). "
            f"region_transforms may have been silently dropped.",
        )

        self.assertGreater(
            regather_chain_ops,
            ref_chain_ops,
            "Expected regather to duplicate weight-chain ops (t/transpose) "
            "in the backward graph "
            f"(reference={ref_chain_ops}, regather={regather_chain_ops}). "
            "Weight-chain recomputation may not be present in the backward pass.",
        )
