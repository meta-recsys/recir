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
from typing import Any

import recir
import recir.lowering.fbgemm_fallback  # noqa: F401
import torch
from recir.api import make_compiled_runner, sync_device_state_to_torch
from recir.backends import XLAGSPMDBackend
from recir.distributed.parallel_mode import GSPMDParallel
from recir.tests.eager_test_utils import (
    ToyDense as SimpleDense,
    ToyDispatcher as SimpleDispatcher,
    ToyDummyPreprocess as DummyPreprocess,
    ToyFusedPreprocess as FusedPreprocess,
    ToyWrapper as SimpleWrapper,
)

# IMPORTANT: must run BEFORE any `import jax` so the cuda plugin path patch
# is in effect when jax.devices() / jax.* first triggers plugin discovery.
from recir.tests.xla import _jax_cuda_plugin_fix  # noqa: F401, I001
from recir.tests.xla.gspmd_test_utils import _iter_subgraphs


def _compile_with_toy_dispatcher(
    model: torch.nn.Module,
    callable_: Any,
    *args: Any,
    dispatcher: SimpleDispatcher,
    parallel_mode: GSPMDParallel,
    backend: XLAGSPMDBackend,
) -> Any:
    """Compile a toy override-based train step via the v7.1 trace + compile_from_gm pipeline."""
    dispatcher.install_eager(model)
    gm, flat_state = recir.trace(callable_, *args)
    gm, flat_state = dispatcher.install_graph(gm, flat_state)
    compiled_obj = recir.compile_from_gm(
        gm,
        flat_state,
        parallel_mode=parallel_mode,
        backend=backend,
    )
    return make_compiled_runner(compiled_obj)


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
        wrapper = SimpleWrapper(preprocess, dense)
        dispatcher = SimpleDispatcher(dummy, num_ranks=self.num_ranks)

        opt = torch.optim.SGD(wrapper.parameters(), lr=0.01, foreach=False)
        indices = torch.randint(0, 10, (global_bs,))

        def train_step_fn(mod, opt, indices):
            loss = mod(indices)
            loss.backward()
            opt.step()

        train_step = _compile_with_toy_dispatcher(
            wrapper,
            train_step_fn,
            wrapper,
            opt,
            indices,
            dispatcher=dispatcher,
            parallel_mode=GSPMDParallel(
                "replicate",
                loss_reduction="sum",
                num_ranks=self.num_ranks,
                num_processes=1,
            ),
            backend=XLAGSPMDBackend(),
        )

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

    def _compile_train_step(
        self,
        wrapper: SimpleWrapper,
        dispatcher: SimpleDispatcher,
        opt: torch.optim.Optimizer,
        indices: torch.Tensor,
    ):
        def train_step_fn(mod, opt, indices):
            loss = mod(indices)
            loss.backward()
            opt.step()

        return _compile_with_toy_dispatcher(
            wrapper,
            train_step_fn,
            wrapper,
            opt,
            indices,
            dispatcher=dispatcher,
            parallel_mode=GSPMDParallel(
                "replicate",
                loss_reduction="sum",
                num_ranks=self.num_ranks,
                num_processes=1,
            ),
            backend=XLAGSPMDBackend(),
        )

    def _count_graph_break_nodes(self, compiled_obj) -> int:
        # Post-NC12 (uses_iter_graph=False default), the gm is a plain
        # fx.GraphModule rather than an IterGraphModule wrapper.  Use the
        # shared helper to iterate either shape.
        bwd_fn_count = 0
        for _gm_name, gm in _iter_subgraphs(compiled_obj.gm):
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
        # Wrap the *real* preprocess; the dispatcher.install_eager call swaps
        # it for the dummy during stage 0 (and captures the original module
        # for the runtime dispatch).
        wrapper = SimpleWrapper(override_preprocess, override_dense)
        dispatcher = SimpleDispatcher(dummy, num_ranks=self.num_ranks)
        opt = torch.optim.SGD(wrapper.parameters(), lr=0.01, foreach=False)

        train_step = self._compile_train_step(wrapper, dispatcher, opt, indices)

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
