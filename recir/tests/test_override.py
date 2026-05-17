#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-ignore-all-errors

"""Tests verifying Override forward/backward mechanics.

Tests the core override machinery (anchor ops, patch_graph, dispatch) using
a simple toy model without XLA, DMP, or multiprocess complexity.
"""

import unittest
from copy import deepcopy

import torch
from functorch import make_fx
from recir.tests.override_test_utils import (
    run_graph,
    ToyDense,
    ToyDummyPreprocess,
    ToyFusedPreprocess,
    ToyOverride,
    ToyPreprocess,
    ToyWrapper,
    trace_train_step,
)
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.nn.utils import stateless


class TestOverride(unittest.TestCase):
    def test_anchor_ops_appear_in_traced_graph(self):
        """make_fx should produce fwd_anchor and bwd_anchor nodes."""
        mod = ToyDummyPreprocess(batch_size=3, embedding_dim=4)
        fake_mode = FakeTensorMode()
        params = dict(mod.named_parameters())

        def trace_fn(params, x):
            with stateless._reparametrize_module(mod, params):
                out = mod(x)
                loss = out.sum()
                loss.backward()
                return out, params

        fake_x = fake_mode.from_tensor(torch.randint(0, 10, (3,)))
        with fake_mode, torch.autograd.detect_anomaly(check_nan=False):
            gm = make_fx(trace_fn, tracing_mode="fake", _allow_non_fake_inputs=False)(
                params, fake_x
            )

        fwd = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and "fwd_anchor" in str(n.target)
        ]
        bwd = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and "bwd_anchor" in str(n.target)
        ]
        self.assertGreater(len(fwd), 0, "Expected fwd_anchor nodes")
        self.assertGreater(len(bwd), 0, "Expected bwd_anchor nodes")

    def test_patch_graph_replaces_anchors(self):
        """override.patch_graph() should replace anchor ops with dispatch calls."""
        torch.manual_seed(42)
        preprocess = ToyPreprocess()
        dense = ToyDense()
        dummy = ToyDummyPreprocess(batch_size=3, embedding_dim=4)
        wrapper = ToyWrapper(dummy, dense)
        override = ToyOverride(dummy)
        override._orig_preprocess = preprocess
        opt = torch.optim.SGD(wrapper.parameters(), lr=0.01, foreach=False)
        x = torch.randint(0, 10, (3,))

        gm, flat_state = trace_train_step(wrapper, opt, x)
        gm = override.patch_graph(gm, flat_state)

        has_graph_break = any(
            n.op == "call_function" and n.meta.get("xla_graph_break")
            for n in gm.graph.nodes
        )
        self.assertTrue(has_graph_break, "Expected xla_graph_break dispatch node")
        has_fwd_anchor = any(
            "fwd_anchor" in str(n.target)
            for n in gm.graph.nodes
            if n.op == "call_function"
        )
        has_bwd_anchor = any(
            "bwd_anchor" in str(n.target)
            for n in gm.graph.nodes
            if n.op == "call_function"
        )
        self.assertFalse(has_fwd_anchor, "fwd_anchor should be removed")
        self.assertFalse(has_bwd_anchor, "bwd_anchor should be removed")

    def test_override_backward_produces_gradients(self):
        """override backward should produce embedding gradients."""
        torch.manual_seed(42)
        preprocess = ToyPreprocess()
        dense = ToyDense()
        dummy = ToyDummyPreprocess(batch_size=3, embedding_dim=4)
        wrapper = ToyWrapper(dummy, dense)
        override = ToyOverride(dummy)
        override._orig_preprocess = preprocess
        opt = torch.optim.SGD(wrapper.parameters(), lr=0.01, foreach=False)
        x = torch.randint(0, 10, (3,))

        gm, flat_state = trace_train_step(wrapper, opt, x)
        gm = override.patch_graph(gm, flat_state)
        run_graph(gm, wrapper, opt, x)

        self.assertIsNotNone(
            preprocess.embedding.weight.grad,
            "override backward should produce embedding gradients",
        )

    def test_embedding_gradients_after_override_backward(self):
        """override backward should trigger embedding backward."""
        torch.manual_seed(42)
        preprocess = ToyPreprocess(num_embeddings=10, embedding_dim=4)
        dense = ToyDense(embedding_dim=4, output_dim=2)
        dummy = ToyDummyPreprocess(batch_size=3, embedding_dim=4)
        wrapper = ToyWrapper(dummy, dense)
        override = ToyOverride(dummy)
        override._orig_preprocess = preprocess
        self.assertIsNone(preprocess.embedding.weight.grad)
        opt = torch.optim.SGD(wrapper.parameters(), lr=0.01, foreach=False)
        x = torch.randint(0, 10, (3,))

        gm, flat_state = trace_train_step(wrapper, opt, x)
        gm = override.patch_graph(gm, flat_state)
        run_graph(gm, wrapper, opt, x)

        self.assertIsNotNone(
            preprocess.embedding.weight.grad,
            "Embedding weight should have gradients after override backward",
        )

    def test_numerical_parity_with_eager_baseline(self):
        """Override path should match eager baseline numerically."""
        torch.manual_seed(42)
        preprocess = ToyPreprocess(num_embeddings=10, embedding_dim=4)
        dense = ToyDense(embedding_dim=4, output_dim=2)
        x = torch.randint(0, 10, (3,))

        # Eager baseline
        baseline_preprocess = deepcopy(preprocess)
        baseline_dense = deepcopy(dense)
        baseline_wrapper = ToyWrapper(baseline_preprocess, baseline_dense)
        baseline_opt = torch.optim.SGD(
            baseline_wrapper.parameters(), lr=0.01, foreach=False
        )
        loss = baseline_wrapper(x)
        loss.backward()
        baseline_opt.step()

        # Override path
        override_preprocess = deepcopy(preprocess)
        override_dense = deepcopy(dense)
        dummy = ToyDummyPreprocess(batch_size=3, embedding_dim=4)
        wrapper = ToyWrapper(dummy, override_dense)
        override = ToyOverride(dummy)
        override._orig_preprocess = override_preprocess
        opt = torch.optim.SGD(wrapper.parameters(), lr=0.01, foreach=False)

        gm, flat_state = trace_train_step(wrapper, opt, x)
        gm = override.patch_graph(gm, flat_state)
        run_graph(gm, wrapper, opt, x)

        self.assertIsNotNone(override_preprocess.embedding.weight.grad)
        torch.testing.assert_close(
            override_preprocess.embedding.weight.grad,
            baseline_preprocess.embedding.weight.grad,
            atol=1e-5,
            rtol=1e-5,
            msg="Override embedding gradients should match eager baseline",
        )

    def test_numerical_parity_under_no_grad(self):
        """Override backward should work under torch.no_grad()."""
        torch.manual_seed(42)
        preprocess = ToyPreprocess(num_embeddings=10, embedding_dim=4)
        dense = ToyDense(embedding_dim=4, output_dim=2)
        dummy = ToyDummyPreprocess(batch_size=3, embedding_dim=4)
        wrapper = ToyWrapper(dummy, dense)
        override = ToyOverride(dummy)
        override._orig_preprocess = preprocess
        opt = torch.optim.SGD(wrapper.parameters(), lr=0.01, foreach=False)
        x = torch.randint(0, 10, (3,))

        gm, flat_state = trace_train_step(wrapper, opt, x)
        gm = override.patch_graph(gm, flat_state)
        with torch.no_grad():
            run_graph(gm, wrapper, opt, x)

        self.assertIsNotNone(
            preprocess.embedding.weight.grad,
            "Override should produce gradients under no_grad",
        )

    def test_fused_optimizer_parity(self):
        """Fused optimizer (TBE-like) should match eager baseline."""
        torch.manual_seed(42)
        preprocess = ToyFusedPreprocess(num_embeddings=10, embedding_dim=4, lr=0.1)
        dense = ToyDense(embedding_dim=4, output_dim=2)
        x = torch.tensor([0, 1, 2])

        # Eager baseline (fused optimizer fires during backward)
        baseline_preprocess = deepcopy(preprocess)
        baseline_dense = deepcopy(dense)
        baseline_wrapper = ToyWrapper(baseline_preprocess, baseline_dense)
        loss = baseline_wrapper(x)
        loss.backward()

        # Override path
        override_preprocess = deepcopy(preprocess)
        override_dense = deepcopy(dense)
        dummy = ToyDummyPreprocess(batch_size=3, embedding_dim=4)
        wrapper = ToyWrapper(dummy, override_dense)
        override = ToyOverride(dummy)
        override._orig_preprocess = override_preprocess
        opt = torch.optim.SGD(wrapper.parameters(), lr=0.01, foreach=False)

        gm, flat_state = trace_train_step(wrapper, opt, x)
        gm = override.patch_graph(gm, flat_state)
        with torch.no_grad():
            run_graph(gm, wrapper, opt, x)

        torch.testing.assert_close(
            override_preprocess.weight,
            baseline_preprocess.weight,
            atol=1e-5,
            rtol=1e-5,
            msg="Fused optimizer: override should produce identical weight updates",
        )
