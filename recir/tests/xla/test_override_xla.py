#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-ignore-all-errors

"""XLA GSPMD tests for Override forward/backward mechanics.

Tests the override machinery with the real GSPMD XLA backend, including
fused optimizer simulation (TBE-like weight updates during backward).
Requires GPU.
"""

import unittest
from copy import deepcopy

import recir.lowering.fbgemm_fallback  # noqa: F401
import torch
from recir.api import compile, sync_device_state_to_torch
from recir.backend import BackendType
from recir.distributed.parallel_mode import GSPMDParallel
from recir.gm_transformation import GraphModuleTransformation
from recir.tests.override_test_utils import (
    ToyDense as SimpleDense,
    ToyDummyPreprocess as DummyPreprocess,
    ToyFusedPreprocess as FusedPreprocess,
    ToyOverride as SimpleOverride,
    ToyWrapper as SimpleWrapper,
)

# IMPORTANT: must run BEFORE any `import jax` so the cuda plugin path patch
# is in effect when jax.devices() / jax.* first triggers plugin discovery.
from recir.tests.xla import _jax_cuda_plugin_fix  # noqa: F401, I001


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOverrideXLAGSPMD(unittest.TestCase):
    """Test override mechanics with real XLA GSPMD backend."""

    def setUp(self):
        import jax  # @manual

        self.devices = jax.devices()
        if len(self.devices) < 2:
            self.skipTest("Need >= 2 JAX devices")
        self.num_ranks = len(self.devices)

    def test_fused_optimizer_compiles_with_gspmd(self):
        """Fused optimizer model compiles through GSPMD without errors."""
        torch.manual_seed(42)
        global_bs = 3 * self.num_ranks
        preprocess = FusedPreprocess(num_embeddings=10, embedding_dim=4, lr=0.1)
        dense = SimpleDense(embedding_dim=4, output_dim=2)
        dummy = DummyPreprocess(embedding_dim=4)
        wrapper = SimpleWrapper(dummy, dense)
        override = SimpleOverride(dummy, num_ranks=self.num_ranks)
        override._orig_preprocess = preprocess

        opt = torch.optim.SGD(wrapper.parameters(), lr=0.01, foreach=False)
        indices = torch.randint(0, 10, (global_bs,))

        @compile(
            module_override=[override],
            parallel_mode=GSPMDParallel(
                "replicate",
                loss_reduction="sum",
                num_ranks=self.num_ranks,
                num_processes=1,
            ),
            gm_transformation=GraphModuleTransformation(
                backend=BackendType.XLA_GSPMD,
                enable_device_state=True,
            ),
        )
        def train_step(mod, opt, indices):
            loss = mod(indices)
            loss.backward()
            opt.step()

        train_step(wrapper, opt, indices)
        sync_device_state_to_torch(train_step)

        compiled_obj = train_step.__dict__["_compiled_obj"]
        self.assertIsNotNone(compiled_obj)

    def _run_eager_baseline(
        self,
        preprocess: FusedPreprocess,
        dense: SimpleDense,
        indices: torch.Tensor,
        num_iters: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        baseline_preprocess = deepcopy(preprocess)
        baseline_dense = deepcopy(dense)
        baseline_wrapper = SimpleWrapper(baseline_preprocess, baseline_dense)
        baseline_opt = torch.optim.SGD(
            baseline_wrapper.parameters(), lr=0.01, foreach=False
        )
        for _ in range(num_iters):
            loss = baseline_wrapper(indices)
            (loss / self.num_ranks).backward()
            baseline_opt.step()
            baseline_opt.zero_grad()
        baseline_emb_weights = baseline_preprocess.weight.detach().clone()
        baseline_dense_weights = {
            n: p.detach().clone() for n, p in baseline_dense.named_parameters()
        }
        return baseline_emb_weights, baseline_dense_weights

    def _compile_train_step(self, override: SimpleOverride):
        @compile(
            module_override=[override],
            parallel_mode=GSPMDParallel(
                "replicate",
                loss_reduction="sum",
                num_ranks=self.num_ranks,
                num_processes=1,
            ),
            gm_transformation=GraphModuleTransformation(
                backend=BackendType.XLA_GSPMD,
                enable_device_state=True,
            ),
        )
        def train_step(mod, opt, indices):
            loss = mod(indices)
            loss.backward()
            opt.step()

        return train_step

    def _count_graph_break_nodes(self, compiled_obj) -> int:
        bwd_fn_count = 0
        for gm_name in ("setup_gm", "main_gm", "cleanup_gm"):
            gm = getattr(compiled_obj.gm, gm_name)
            for n in gm.graph.nodes:
                if n.op == "call_function" and n.meta.get("xla_graph_break"):
                    bwd_fn_count += 1
            for _, mod in gm.named_modules():
                if hasattr(mod, "graph"):
                    for n in mod.graph.nodes:
                        if n.op == "call_function" and n.meta.get("xla_graph_break"):
                            bwd_fn_count += 1
        return bwd_fn_count

    def test_fused_optimizer_numerical_parity(self):
        """Fused optimizer: override backward should match eager baseline numerically."""
        num_iters = 3
        global_bs = 3 * self.num_ranks

        torch.manual_seed(42)
        preprocess = FusedPreprocess(num_embeddings=10, embedding_dim=4, lr=0.1)
        dense = SimpleDense(embedding_dim=4, output_dim=2)
        indices = torch.randint(0, 10, (global_bs,))

        baseline_emb_weights, baseline_dense_weights = self._run_eager_baseline(
            preprocess,
            dense,
            indices,
            num_iters,
        )

        override_preprocess = deepcopy(preprocess)
        override_dense = deepcopy(dense)
        dummy = DummyPreprocess(embedding_dim=4)
        wrapper = SimpleWrapper(dummy, override_dense)
        override = SimpleOverride(dummy, num_ranks=self.num_ranks)
        override._orig_preprocess = override_preprocess
        opt = torch.optim.SGD(wrapper.parameters(), lr=0.01, foreach=False)

        train_step = self._compile_train_step(override)

        for _ in range(num_iters):
            train_step(wrapper, opt, indices)
            sync_device_state_to_torch(train_step)

        compiled_obj = train_step.__dict__["_compiled_obj"]
        bwd_fn_count = self._count_graph_break_nodes(compiled_obj)

        override_emb_weights = override_preprocess.weight.detach().clone()
        initial_weights = deepcopy(preprocess).weight.detach()
        emb_changed = not torch.equal(override_emb_weights, initial_weights)
        baseline_changed = not torch.equal(baseline_emb_weights, initial_weights)
        max_diff = (override_emb_weights - baseline_emb_weights).abs().max().item()
        override_diff_from_init = (
            (override_emb_weights - initial_weights).abs().max().item()
        )
        baseline_diff_from_init = (
            (baseline_emb_weights - initial_weights).abs().max().item()
        )

        torch.testing.assert_close(
            override_emb_weights,
            baseline_emb_weights,
            atol=5e-3,
            rtol=1e-3,
            msg=(
                f"Fused optimizer: embedding weights should match baseline. "
                f"max_diff={max_diff:.6f}, "
                f"override_changed_from_init={emb_changed} (max_delta={override_diff_from_init:.6f}), "
                f"baseline_changed_from_init={baseline_changed} (max_delta={baseline_diff_from_init:.6f}), "
                f"bwd_fn_count_in_compiled_graph={bwd_fn_count}"
            ),
        )

        for name, param in override_dense.named_parameters():
            baseline_param = baseline_dense_weights[name]
            torch.testing.assert_close(
                param,
                baseline_param,
                atol=5e-3,
                rtol=1e-3,
                msg=f"Dense param {name} should match baseline",
            )
