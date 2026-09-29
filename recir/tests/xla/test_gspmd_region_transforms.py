# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors

"""Tests for activation checkpointing on the GSPMD lowering path.

Exercises ``compile(..., region_transforms=[RecomputeTransform(...)])`` end
to end on the GSPMD parallel mode.

Run: buck2 test //recir/tests/xla:test_gspmd_region_transforms
"""

import unittest

import torch
import torch.nn as nn
from parameterized import parameterized
from recir.api import compile
from recir.backends import XLAGSPMDBackend
from recir.distributed.parallel_mode import GSPMDParallel
from recir.lowering.xla_lower import _XLAModule
from recir.region_transforms import RecomputeTransform
from recir.tests.xla.gspmd_test_utils import (
    _run_single_rank_baseline,
    DIM,
    LOCAL_BS,
    NUM_ITERS,
    SimpleMLP,
)


def _make_train_step(num_ranks, loss_reduction, region_transforms):
    reduce_fn = torch.sum if loss_reduction == "sum" else torch.mean

    @compile(
        parallel_mode=GSPMDParallel(
            "replicate",
            loss_reduction=loss_reduction,
            num_ranks=num_ranks,
            num_processes=1,
        ),
        backend=XLAGSPMDBackend(),
        region_transforms=region_transforms,
    )
    def train_step(mod, opt, x):
        reduce_fn(mod(x)).backward()
        opt.step()

    return train_step


def _count_inner_xla_nodes(gm):
    """Sum FX node counts across all inner ``_XLAModule.gm`` subgraphs.

    After XLA lowering the outer FX graph holds only ``call_module`` nodes
    pointing at ``_XLAModule`` wrappers; the recompute-induced node
    duplication shows up inside those wrappers' ``.gm`` graphs.
    """
    return sum(len(m.gm.graph.nodes) for m in gm.modules() if isinstance(m, _XLAModule))


def _count_relu_ops(gm):
    """Count relu ops in inner XLA graphs."""
    count = 0
    for m in gm.modules():
        if isinstance(m, _XLAModule):
            for node in m.gm.graph.nodes:
                if node.op == "call_function" and "relu" in str(node.target):
                    count += 1
    return count


class TestGSPMDRecompute(unittest.TestCase):
    """Activation checkpointing on the GSPMD lowering path."""

    def setUp(self):
        self.num_ranks = 2

    @parameterized.expand([("sum",), ("mean",)])
    def test_gspmd_recompute(self, loss_reduction):
        """RecomputeTransform on relu must preserve numerics, expand the
        joint graph, and not break GSPMD lowering.
        """
        torch.manual_seed(42)
        x = torch.randn(LOCAL_BS * self.num_ranks, DIM)
        baseline_params = _run_single_rank_baseline(
            x, self.num_ranks, loss_reduction=loss_reduction
        )

        # Reference compile: no recompute. One step is enough to populate
        # the inner _XLAModule graphs for the structural comparison below.
        torch.manual_seed(42)
        ref_model = SimpleMLP().cpu()
        ref_opt = torch.optim.SGD(ref_model.parameters(), lr=0.01, foreach=False)
        ref_step = _make_train_step(self.num_ranks, loss_reduction, None)
        ref_step(ref_model, ref_opt, x)
        ref_main_gm = ref_step.__dict__["_compiled_obj"].gm.main_gm
        ref_nodes = _count_inner_xla_nodes(ref_main_gm)
        ref_relu_ops = _count_relu_ops(ref_main_gm)

        # Recompute-enabled compile: wrap relu.
        torch.manual_seed(42)
        model = SimpleMLP().cpu()
        opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
        train_step = _make_train_step(
            self.num_ranks,
            loss_reduction,
            [RecomputeTransform(submodule_fqns={"relu"})],
        )
        for step in range(NUM_ITERS):
            train_step(model, opt, x)
            for (name, param), (_, baseline) in zip(
                model.named_parameters(),
                baseline_params[step].items(),
            ):
                torch.testing.assert_close(
                    param,
                    baseline,
                    msg=f"Step {step}: recompute mismatch for {name}",
                )
        compiled_obj = train_step.__dict__["_compiled_obj"]
        recompute_main_gm = compiled_obj.gm.main_gm
        recompute_nodes = _count_inner_xla_nodes(recompute_main_gm)
        recompute_relu_ops = _count_relu_ops(recompute_main_gm)

        self.assertGreater(
            recompute_nodes,
            ref_nodes,
            "Expected recompute to add nodes to the inner XLA graph "
            f"(reference={ref_nodes}, recompute={recompute_nodes}). "
            "region_transforms may have been silently dropped.",
        )

        self.assertGreater(
            recompute_relu_ops,
            ref_relu_ops,
            "Expected recompute to duplicate relu ops in the backward graph "
            f"(reference={ref_relu_ops}, recompute={recompute_relu_ops}). "
            "relu recomputation may not be present in the backward pass.",
        )

    @parameterized.expand([("sum",), ("mean",)])
    def test_input_transform_applied_before_region_discovery(self, loss_reduction):
        """Verify ``_compile_gspmd`` applies ``_input_transform`` BEFORE
        region discovery.

        Uses a model with a shape-baking ``view`` whose target shape is
        read from ``x.shape[0]`` (concrete int under
        ``tracing_mode="fake"``).  This bakes the per-process batch dim
        into the inner ``gm_fwd``'s ``aten.view`` args as a Python list
        ``[bs, DIM]``.

        Without the fix:
            - discovery sees raw user input ``x.shape == (1, DIM)``
            - inner ``gm_fwd`` bakes ``aten.view(out, [1, DIM])``
            - outer trace runs ``gm_fwd`` with global-batch ``(2, DIM)``
            - view: 2*DIM elements into shape ``[1, DIM]`` → RAISES at
              compile time, before ``_compiled_obj`` is set.

        With the fix:
            - discovery sees post-transform ``x.shape == (2, DIM)``
            - inner ``gm_fwd`` bakes ``aten.view(out, [2, DIM])``
            - outer trace at ``(2, DIM)`` matches → compile succeeds
              and ``_compiled_obj`` is set.

        We catch any later runtime errors (separate, pre-existing
        GSPMD-without-XLA limitation) and only assert compile completion.
        """

        class _ShapeBakedMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(DIM, DIM)

            def forward(self, x):
                # x.shape[0] is a concrete Python int under fake tracing,
                # so the view's target-shape list bakes the batch dim.
                return self.fc1(x).view(x.shape[0], DIM)

        torch.manual_seed(42)
        model = _ShapeBakedMLP().cpu()
        opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
        x = torch.randn(LOCAL_BS, DIM)  # per-process batch
        reduce_fn = torch.sum if loss_reduction == "sum" else torch.mean

        @compile(
            parallel_mode=GSPMDParallel(
                "replicate",
                loss_reduction=loss_reduction,
                num_ranks=2,
                num_processes=2,
            ),
            region_transforms=[RecomputeTransform(submodule_fqns={"fc1"})],
        )
        def train_step(mod, opt, x):
            reduce_fn(mod(x)).backward()
            opt.step()

        # Compile happens at the first call.  Pre-fix raises at compile
        # time and never sets _compiled_obj.  Post-fix sets _compiled_obj,
        # then may raise at runtime (gm traced at global, called at
        # per-process — separate issue).  Either way, we only check that
        # compile completed.
        try:
            train_step(model, opt, x)
        except RuntimeError:
            pass
        self.assertIn(
            "_compiled_obj",
            train_step.__dict__,
            "Compile failed: _input_transform was not applied before "
            "region discovery, so gm_fwd's baked view-target shape "
            "mismatched the outer trace's expanded inputs.",
        )


if __name__ == "__main__":
    unittest.main()
