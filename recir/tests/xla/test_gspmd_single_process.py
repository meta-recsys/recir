# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors

"""Tests for single-process multi-device XLA compilation via @compile.

Unlike test_gspmd.py (which uses DTensorTestBase to spawn one subprocess per
GPU), this test runs a single process that sees all local devices. This is the
required execution model for TPU VMs where libtpu only allows one process to
access all chips.

The same tests also work on multi-GPU machines with JAX_PLATFORMS=cuda, making
them runnable in CI without TPU hardware.

Test matrix row: num_processes=1, num_ranks=N (all local devices).
"""

import copy
import unittest

import torch
from parameterized import parameterized
from recir.api import compile
from recir.backends import XLAGSPMDBackend, XLAShardMapBackend
from recir.backends.types import BackendType
from recir.distributed.parallel_mode import DataParallel, GSPMDParallel
from recir.tests.xla.gspmd_test_utils import (
    _run_single_rank_baseline,
    assert_hlo_has_collective,
    assert_no_graph_breaks,
    DIM,
    LOCAL_BS,
    NUM_ITERS,
    SimpleMLP,
)


class TestSingleProcessMultiDevice(unittest.TestCase):
    """Single-process multi-device tests for XLA backends.

    num_processes=1, num_ranks=N (all local devices in one process).
    """

    def test_replicated_embedding_backward_local_scatter_allreduce(self) -> None:
        """Replicated nn.Embedding backward must lower to a LOCAL scatter into
        [vocab, D] followed by a single [vocab, D] all-reduce (via the torchax
        _aten_embedding_dense_backward shard_map+psum op), NOT GSPMD's
        gather-the-[N_global, D]-activation-then-global-scatter pattern.

        Regression test for a DP-replicated embedding all-reduce bottleneck.
        Requires that
        aten.embedding_dense_backward is preserved through the GSPMD lowering
        (it is kept out of recir.torchax.decompositions.DECOMPOSITIONS).
        """
        import jax  # @manual
        from recir.lowering.xla_lower import _XLAModule

        if len(jax.devices()) < 2:
            self.skipTest("Need >= 2 JAX devices")
        num_ranks = len(jax.devices())

        torch.manual_seed(0)
        vocab, dim = 16, 4

        class EmbModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.emb = torch.nn.Embedding(vocab, dim)
                self.lin = torch.nn.Linear(dim, 1)

            def forward(self, idx: torch.Tensor) -> torch.Tensor:
                return self.lin(self.emb(idx)).sum()

        model = EmbModel().cpu()
        opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
        # Global token count (32*ranks*8) >> vocab so a gather-based partitioning
        # would produce a much larger collective than [vocab, D].
        idx = torch.randint(0, vocab, (32 * num_ranks, 8))

        @compile(
            parallel_mode=GSPMDParallel(
                "replicate",
                loss_reduction="sum",
                num_ranks=num_ranks,
                num_processes=1,
            ),
            backend=XLAGSPMDBackend(dump_hlo=True),
        )
        def train_step(
            mod: torch.nn.Module, opt: torch.optim.Optimizer, idx: torch.Tensor
        ) -> None:
            mod(idx).backward()
            opt.step()

        train_step(model, opt, idx)
        compiled_obj = train_step.__dict__["_compiled_obj"]

        # 1. embedding_dense_backward must survive to the lowered subgraph.
        lowered_ops: set[str] = set()
        for m in compiled_obj.gm.modules():
            if isinstance(m, _XLAModule):
                for n in m.gm.graph.nodes:
                    if n.op == "call_function":
                        lowered_ops.add(str(n.target))
        self.assertIn(
            "aten.embedding_dense_backward.default",
            lowered_ops,
            f"embedding_dense_backward was decomposed; lowered ops: {sorted(lowered_ops)}",
        )

        # 2. The HLO must show our shard_map+psum op fired (cheap order).
        hlo_texts = [
            m.xla_hlo_text
            for m in compiled_obj.gm.modules()
            if isinstance(m, _XLAModule) and m.xla_hlo_text
        ]
        joined = "\n".join(hlo_texts)
        self.assertIn(
            "aten::embedding_dense_backward/shard_map/psum",
            joined,
            "embedding_dense_backward did not lower via the DP shard_map+psum op",
        )

    def setUp(self):
        import jax  # @manual

        self.devices = jax.devices()
        if len(self.devices) < 2:
            self.skipTest("Need >= 2 JAX devices")
        self.num_ranks = len(self.devices)

    @parameterized.expand([("sum",), ("mean",)])
    def test_gspmd_single_process(self, loss_reduction):
        """GSPMDParallel + XLA_GSPMD with 1 process, N devices."""
        torch.manual_seed(42)
        model = SimpleMLP().cpu()
        opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
        # Single process holds the full global batch
        x = torch.randn(LOCAL_BS * self.num_ranks, DIM)
        baseline_params = _run_single_rank_baseline(
            x, self.num_ranks, loss_reduction=loss_reduction
        )
        reduce_fn = torch.sum if loss_reduction == "sum" else torch.mean

        @compile(
            parallel_mode=GSPMDParallel(
                "replicate",
                loss_reduction=loss_reduction,
                num_ranks=self.num_ranks,
                num_processes=1,
            ),
            backend=XLAGSPMDBackend(dump_hlo=True),
        )
        def train_step(mod, opt, x):
            reduce_fn(mod(x)).backward()
            opt.step()

        for step in range(NUM_ITERS):
            train_step(model, opt, x)
            for (name, param), (_, baseline_param) in zip(
                model.named_parameters(),
                baseline_params[step].items(),
            ):
                torch.testing.assert_close(
                    param,
                    baseline_param,
                    msg=f"Step {step}: XLA_GSPMD vs single-rank mismatch for {name}",
                )
        compiled_obj = train_step.__dict__["_compiled_obj"]
        assert_hlo_has_collective(self, compiled_obj, "all-reduce", "XLA_GSPMD")

    @parameterized.expand([("sum",), ("mean",)])
    def test_shard_map_single_process(self, loss_reduction):
        """DataParallel + XLA_SHARD_MAP with 1 process, N devices."""
        torch.manual_seed(42)
        model = SimpleMLP().cpu()
        opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
        x = torch.randn(LOCAL_BS * self.num_ranks, DIM)
        baseline_params = _run_single_rank_baseline(
            x, self.num_ranks, loss_reduction=loss_reduction
        )
        reduce_fn = torch.sum if loss_reduction == "sum" else torch.mean

        @compile(
            parallel_mode=DataParallel(
                "replicate",
                loss_reduction=loss_reduction,
                num_ranks=self.num_ranks,
                num_processes=1,
                device_type="cpu",
                backend=BackendType.XLA_SHARD_MAP,
            ),
            backend=XLAShardMapBackend(dump_hlo=True),
        )
        def train_step(mod, opt, x):
            reduce_fn(mod(x)).backward()
            opt.step()

        for step in range(NUM_ITERS):
            train_step(model, opt, x)
            for (name, param), (_, baseline_param) in zip(
                model.named_parameters(),
                baseline_params[step].items(),
            ):
                torch.testing.assert_close(
                    param,
                    baseline_param,
                    msg=f"Step {step}: XLA_SHARD_MAP vs single-rank mismatch for {name}",
                )
        compiled_obj = train_step.__dict__["_compiled_obj"]
        assert_no_graph_breaks(self, compiled_obj)
        assert_hlo_has_collective(self, compiled_obj, "all_reduce", "XLA_SHARD_MAP")

    def test_shard_map_replicated_embedding_backward(self) -> None:
        torch.manual_seed(0)
        vocab, dim = 16, 4

        class EmbModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.emb = torch.nn.Embedding(vocab, dim)

            def forward(self, idx: torch.Tensor) -> torch.Tensor:
                return self.emb(idx).sum()

        model = EmbModel().cpu()
        baseline = copy.deepcopy(model)
        opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
        baseline_opt = torch.optim.SGD(baseline.parameters(), lr=0.01, foreach=False)
        idx = torch.randint(0, vocab, (32 * self.num_ranks, 8))

        baseline(idx).backward()
        baseline_opt.step()

        @compile(
            parallel_mode=DataParallel(
                "replicate",
                loss_reduction="sum",
                num_ranks=self.num_ranks,
                num_processes=1,
                device_type="cpu",
                backend=BackendType.XLA_SHARD_MAP,
            ),
            backend=XLAShardMapBackend(dump_hlo=True),
        )
        def train_step(
            mod: torch.nn.Module, opt: torch.optim.Optimizer, idx: torch.Tensor
        ) -> None:
            mod(idx).backward()
            opt.step()

        train_step(model, opt, idx)
        for (name, param), (_, baseline_param) in zip(
            model.named_parameters(), baseline.named_parameters()
        ):
            torch.testing.assert_close(
                param,
                baseline_param,
                msg=f"XLA_SHARD_MAP embedding step mismatch for {name}",
            )

        compiled_obj = train_step.__dict__["_compiled_obj"]
        assert_no_graph_breaks(self, compiled_obj)
        assert_hlo_has_collective(self, compiled_obj, "all_reduce", "XLA_SHARD_MAP")


if __name__ == "__main__":
    unittest.main()
