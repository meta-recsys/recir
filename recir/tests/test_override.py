#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-ignore-all-errors

"""Tests verifying orchestrator-style override forward/backward mechanics.

Tests the marker-based override machinery (mark_boundary + dispatcher) using
a simple toy model without XLA, DMP, or multiprocess complexity.
"""

import operator
import unittest
from copy import deepcopy

import torch
from functorch import make_fx
from recir.ops.eager_dispatch import _dispatch_bwd, install_graph_break_dispatch
from recir.tests.eager_test_utils import (
    run_graph,
    ToyDense,
    ToyDispatcher,
    ToyDummyPreprocess,
    ToyFusedPreprocess,
    ToyPreprocess,
    ToyWrapper,
    trace_train_step,
)
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.nn.utils import stateless


def _has_marker(gm, suffix: str) -> bool:
    for n in gm.graph.nodes:
        if (
            n.op == "call_function"
            and n.target is torch.ops.recir.mark_boundary.default
            and len(n.args) >= 2
            and isinstance(n.args[1], str)
            and suffix in n.args[1]
        ):
            return True
    return False


class TestOverride(unittest.TestCase):
    def test_scalar_dispatch_output_is_not_batch_sharded(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = torch.ones(1)
        output = graph.call_function(
            torch.ops.recir.mark_boundary.default,
            args=(x, "scalar::output::0"),
        )
        output.meta["val"] = torch.tensor(1.0)
        graph.output(output)
        gm = torch.fx.GraphModule({}, graph)

        install_graph_break_dispatch(
            gm,
            "scalar",
            lambda value: (value.sum(),),
            num_state_tensors=0,
            register_backward=False,
        )

        getitem = next(
            node
            for node in gm.graph.nodes
            if node.op == "call_function" and node.target is operator.getitem
        )
        self.assertFalse(getitem.meta["is_batch_input"])
        self.assertNotIn("batch_dim", getitem.meta)

    def test_install_without_backward_removes_backward_markers(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = torch.ones(1)
        graph.call_function(
            torch.ops.recir.mark_boundary.default,
            args=(x, "disabled::bwd::0"),
        )
        graph.output(x)
        gm = torch.fx.GraphModule({}, graph)

        dispatch_id = install_graph_break_dispatch(
            gm,
            "disabled",
            lambda value: (value,),
            num_state_tensors=0,
            register_backward=False,
        )

        self.assertIsNone(dispatch_id)
        self.assertFalse(_has_marker(gm, "disabled::bwd::0"))
        gm.graph.lint()

    def test_sequential_dispatch_install_preserves_backward_nodes(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = torch.ones(1)
        first_output = graph.call_function(
            torch.ops.recir.mark_boundary.default,
            args=(x, "first::output::0"),
        )
        graph.call_function(
            torch.ops.recir.mark_boundary.default,
            args=(x, "first::bwd::0"),
        )
        second_output = graph.call_function(
            torch.ops.recir.mark_boundary.default,
            args=(x, "second::output::0"),
        )
        graph.call_function(
            torch.ops.recir.mark_boundary.default,
            args=(x, "second::bwd::0"),
        )
        graph.output((first_output, second_output))
        gm = torch.fx.GraphModule({}, graph)

        def dispatch(value):
            return (value,)

        install_graph_break_dispatch(
            gm,
            "first",
            dispatch,
            num_state_tensors=0,
        )
        self.assertTrue(_has_marker(gm, "second::bwd::0"))

        install_graph_break_dispatch(
            gm,
            "second",
            dispatch,
            num_state_tensors=0,
        )
        gm.graph.eliminate_dead_code()
        backward_dispatches = [
            node
            for node in gm.graph.nodes
            if node.op == "call_function" and node.target is _dispatch_bwd
        ]
        self.assertEqual(len(backward_dispatches), 2)

    def test_marker_pairs_appear_in_traced_graph(self):
        """make_fx should produce forward + backward mark_boundary nodes."""
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
        self.assertTrue(_has_marker(gm, "::output::"), "Expected output marker node")
        self.assertTrue(_has_marker(gm, "::bwd::"), "Expected backward marker node")

    def test_install_graph_replaces_markers(self):
        """install_graph should replace markers with graph-break dispatch calls."""
        torch.manual_seed(42)
        preprocess = ToyPreprocess()
        dense = ToyDense()
        dummy = ToyDummyPreprocess(batch_size=3, embedding_dim=4)
        wrapper = ToyWrapper(dummy, dense)
        dispatcher = ToyDispatcher(dummy)
        # Caller wired install_eager separately when building the wrapper;
        # rebuild parity here so install_graph can fire on its own.
        dispatcher._orig_preprocess = preprocess
        opt = torch.optim.SGD(wrapper.parameters(), lr=0.01, foreach=False)
        x = torch.randint(0, 10, (3,))

        gm, flat_state = trace_train_step(wrapper, opt, x)
        gm, _ = dispatcher.install_graph(gm, flat_state)

        has_graph_break = any(
            n.op == "call_function" and n.meta.get("xla_graph_break")
            for n in gm.graph.nodes
        )
        self.assertTrue(has_graph_break, "Expected xla_graph_break dispatch node")
        self.assertFalse(
            _has_marker(gm, "::output::"), "Output marker should be removed"
        )
        self.assertFalse(
            _has_marker(gm, "::bwd::"), "Backward marker should be removed"
        )

    def test_override_backward_produces_gradients(self):
        """override backward should produce embedding gradients."""
        torch.manual_seed(42)
        preprocess = ToyPreprocess()
        dense = ToyDense()
        dummy = ToyDummyPreprocess(batch_size=3, embedding_dim=4)
        wrapper = ToyWrapper(dummy, dense)
        dispatcher = ToyDispatcher(dummy)
        dispatcher._orig_preprocess = preprocess
        opt = torch.optim.SGD(wrapper.parameters(), lr=0.01, foreach=False)
        x = torch.randint(0, 10, (3,))

        gm, flat_state = trace_train_step(wrapper, opt, x)
        gm, _ = dispatcher.install_graph(gm, flat_state)
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
        dispatcher = ToyDispatcher(dummy)
        dispatcher._orig_preprocess = preprocess
        self.assertIsNone(preprocess.embedding.weight.grad)
        opt = torch.optim.SGD(wrapper.parameters(), lr=0.01, foreach=False)
        x = torch.randint(0, 10, (3,))

        gm, flat_state = trace_train_step(wrapper, opt, x)
        gm, _ = dispatcher.install_graph(gm, flat_state)
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
        dispatcher = ToyDispatcher(dummy)
        dispatcher._orig_preprocess = override_preprocess
        opt = torch.optim.SGD(wrapper.parameters(), lr=0.01, foreach=False)

        gm, flat_state = trace_train_step(wrapper, opt, x)
        gm, _ = dispatcher.install_graph(gm, flat_state)
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
        dispatcher = ToyDispatcher(dummy)
        dispatcher._orig_preprocess = preprocess
        opt = torch.optim.SGD(wrapper.parameters(), lr=0.01, foreach=False)
        x = torch.randint(0, 10, (3,))

        gm, flat_state = trace_train_step(wrapper, opt, x)
        gm, _ = dispatcher.install_graph(gm, flat_state)
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
        dispatcher = ToyDispatcher(dummy)
        dispatcher._orig_preprocess = override_preprocess
        opt = torch.optim.SGD(wrapper.parameters(), lr=0.01, foreach=False)

        gm, flat_state = trace_train_step(wrapper, opt, x)
        gm, _ = dispatcher.install_graph(gm, flat_state)
        with torch.no_grad():
            run_graph(gm, wrapper, opt, x)

        torch.testing.assert_close(
            override_preprocess.weight,
            baseline_preprocess.weight,
            atol=1e-5,
            rtol=1e-5,
            msg="Fused optimizer: override should produce identical weight updates",
        )
