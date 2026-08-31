#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""The four digest insertion points in ``recir_train_step``."""

from __future__ import annotations

import io
import os
import unittest
import unittest.mock
from contextlib import redirect_stdout
from typing import Any, Dict, List, Tuple
from unittest import TestCase

import hammer.v2.modules.fused_hstu as fused_hstu
import torch
from hammer.v3.models.udd_fm_v3.parity.digest import (
    capture_activations,
    capture_grad_outputs,
    emit_param_digest,
    emit_update_digest,
    parse_digest,
)
from hammer.v3.models.udd_fm_v3.parity.step_digest import _trim_inputs, arm
from hammer.v3.models.udd_fm_v3.recir_core.compile import recir_train_step
from recir.api import trace
from recir.examples.rankfm_mvai.debug_digest_op import debug_digest


class _Toy(torch.nn.Module):
    """Minimal stand-in with the shape ``recir_train_step`` expects:
    ``forward -> (per-task losses, named tensors)``."""

    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(4))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return (self.w * x).sum().reshape(1), {}


class InGraphStepDigestTest(TestCase):
    def _run(self, **env: str) -> Tuple[str, Dict[str, Any]]:
        mod = _Toy()
        opt = torch.optim.SGD(mod.parameters(), lr=0.5, foreach=True)
        buf = io.StringIO()
        with unittest.mock.patch.dict(os.environ, env, clear=False):
            with redirect_stdout(buf):
                _, named = arm(recir_train_step)(mod, opt, torch.arange(4.0))
        return buf.getvalue(), named

    def test_the_step_emits_loss_and_grad(self) -> None:
        """Only these two. The parameter digests are read off the module either
        side of the compiled step (compiled_module.forward), not from in here:
        a host callback on a weight is a side-effecting HLO with replicated
        sharding, which XLA's SPMD partitioner rejects outright."""
        out, named = self._run(PARITY_DIGEST_KINDS="loss,grad,winit,weight")
        kinds = {kind for _rank, kind, _name in parse_digest(out)}
        self.assertEqual(kinds, {"loss", "grad"})
        # The gradient is a real graph output; there is no anchor any more,
        # because the only digests that needed one were the parameter ones.
        self.assertIn("grad/w", named)

    def test_grad_is_read_before_the_step_clips_it(self) -> None:
        """The gradient digest has to sit between backward and step: the step
        can mutate ``p.grad`` in place (TensorLRAdamW clips), and a digest taken
        after it would disagree with the eager leg exactly on the clipped
        elements."""
        out, _ = self._run(PARITY_DIGEST_KINDS="loss,grad,winit,weight")
        # grad of (w*x).sum() wrt w is x = [0,1,2,3] -> sum 6.
        self.assertEqual(parse_digest(out)[(0, "grad", "w")]["sum"], 6.0)

    def test_gradients_are_also_returned_as_graph_outputs(self) -> None:
        """The in-process diff in ``parity_forward_loss`` reads ``grad/<param>``
        off the returned dict rather than out of the log, so the same hook has
        to put them there. Keyed by the RAW fqn -- the consumer normalizes."""
        out, named = self._run(PARITY_DIGEST_KINDS="loss,grad,winit,weight")
        self.assertIn("grad/w", named)
        # Same value the digest reported: one hook, one tensor, two consumers.
        self.assertEqual(
            float(named["grad/w"].sum()),
            parse_digest(out)[(0, "grad", "w")]["sum"],
        )

    def test_gradient_outputs_follow_the_kinds_gate(self) -> None:
        """No gate, no extra graph outputs: these are full tensors, and an
        unarmed run must produce the graph it produced before any of this."""
        _out, named = self._run(PARITY_DIGEST_KINDS="winit,weight")
        self.assertFalse([k for k in named if str(k).startswith("grad/")])

    def test_disabled_by_default_emits_nothing(self) -> None:
        """Off unless armed, with no extra graph outputs either: an unarmed run
        must produce the same graph it produced before any of this existed."""
        out, named = self._run(PARITY_DIGEST_KINDS="")
        self.assertEqual(parse_digest(out), {})
        self.assertEqual(named, {})

    def test_the_step_emits_nothing_for_weight_only_kinds(self) -> None:
        """The parameter kinds are handled outside the step, so asking for only
        those must leave the traced graph uninstrumented rather than half so."""
        out, _ = self._run(PARITY_DIGEST_KINDS="winit,weight")
        self.assertEqual(parse_digest(out), {})

    def test_kinds_filter_selects_a_subset(self) -> None:
        """Each digest is a full device-to-host transfer, so a session that only
        wants the loss must be able to skip the 200-odd gradients."""
        out, _ = self._run(PARITY_DIGEST_KINDS="loss")
        kinds = {kind for _rank, kind, _name in parse_digest(out)}
        self.assertEqual(kinds, {"loss"})


class StuProbeIsClearedTest(TestCase):
    """The probe is process-global; an armed step must not leave one behind.

    Unlike the torch hooks, which the step removes in a ``finally``, the probe
    is a module-level symbol swap. A leftover one keeps firing with the closure
    it was built with -- this step's num_candidates and padded sizes -- so the
    NEXT step's transduction tensors get trimmed to the previous step's sample
    counts. That reads as divergence in the comparison, which is the failure
    this whole file exists to detect, arriving from the instrumentation itself.
    """

    def test_the_probe_does_not_outlive_the_step(self) -> None:
        before = (
            fused_hstu.hstu_preprocess_and_attention,
            fused_hstu._produce_stu_output_maybe_from_cache,
        )
        mod = _Toy()
        opt = torch.optim.SGD(mod.parameters(), lr=0.5, foreach=True)
        with unittest.mock.patch.dict(
            os.environ, {"PARITY_DIGEST_KINDS": "act,gradout"}, clear=False
        ):
            with redirect_stdout(io.StringIO()):
                arm(recir_train_step)(mod, opt, torch.arange(4.0))
        after = (
            fused_hstu.hstu_preprocess_and_attention,
            fused_hstu._produce_stu_output_maybe_from_cache,
        )
        self.assertEqual(
            before,
            after,
            "the step left hammer/v2's fused_hstu symbols patched, so the next "
            "step would digest against THIS step's trim inputs",
        )


class TracedStepDigestTest(TestCase):
    """The same step put through ``recir.api.trace`` -- the real entry point."""

    @staticmethod
    def _digest_kinds_in(gm: torch.fx.GraphModule) -> Dict[str, int]:
        """kind -> node count. ``kind`` is the op's second positional arg."""
        found: Dict[str, int] = {}
        for n in gm.graph.nodes:
            if n.op == "call_function" and "debug_digest" in str(n.target):
                found[n.args[1]] = found.get(n.args[1], 0) + 1
        return found

    def _trace(self, **env: str) -> torch.fx.GraphModule:
        mod = _Toy()
        opt = torch.optim.SGD(mod.parameters(), lr=0.5, foreach=True)
        with unittest.mock.patch.dict(os.environ, env, clear=False):
            with redirect_stdout(io.StringIO()):
                gm, _flat_state = trace(
                    arm(recir_train_step), mod, opt, torch.arange(4.0)
                )
        return gm

    def test_a_root_forward_hook_can_replace_the_loss_in_the_trace(self) -> None:
        """The property the digest's decoupling rests on."""
        mod = _Toy()
        opt = torch.optim.SGD(mod.parameters(), lr=0.5, foreach=True)

        def root_hook(
            _module: Any, _args: Any, out: Tuple[torch.Tensor, Dict[str, Any]]
        ) -> Tuple[torch.Tensor, Dict[str, Any]]:
            losses, named = out
            return debug_digest(losses, "loss", "per_task_vector", sharded=False), named

        handle = mod.register_forward_hook(root_hook)
        try:
            with redirect_stdout(io.StringIO()):
                gm, _flat_state = trace(recir_train_step, mod, opt, torch.arange(4.0))
        finally:
            handle.remove()

        found = self._digest_kinds_in(gm)
        self.assertEqual(
            found.get("loss", 0),
            1,
            "a root forward hook did not put the loss digest in the graph; "
            f"found {found}",
        )

    def test_grad_digest_survives_a_fake_trace(self) -> None:
        """The one that was assumed rather than checked: ``p.grad`` is populated
        by the traced backward, so the gradient digest is emitted into the graph
        instead of being skipped by the ``t is None`` guard."""
        gm = self._trace(PARITY_DIGEST_KINDS="loss,grad,winit,weight")
        found = self._digest_kinds_in(gm)
        self.assertGreaterEqual(
            found.get("grad", 0),
            1,
            "no grad digest in the traced graph: p.grad was None during the "
            f"fake trace, so the gradients are not being digested. Found: {found}",
        )

    def test_the_in_graph_kinds_reach_the_traced_graph(self) -> None:
        """loss and grad only -- the parameter kinds are read off the module
        outside the graph, so they must leave no node behind in it."""
        gm = self._trace(PARITY_DIGEST_KINDS="loss,grad,winit,weight")
        found = self._digest_kinds_in(gm)
        self.assertEqual(set(found), {"loss", "grad"}, f"found: {found}")

    def test_digests_survive_dce_on_the_traced_graph(self) -> None:
        """End of the chain the anchor exists for: nothing reads a weight
        digest, so without the anchor DCE would drop these before the lowering
        ever sees them."""
        gm = self._trace(PARITY_DIGEST_KINDS="loss,grad,winit,weight")
        before = self._digest_kinds_in(gm)
        gm.graph.eliminate_dead_code()
        self.assertEqual(self._digest_kinds_in(gm), before)

    def test_an_unarmed_trace_has_no_digest_nodes(self) -> None:
        """The gate is read at TRACE time, so an unarmed run must produce a
        graph with none of this in it at all."""
        gm = self._trace(PARITY_DIGEST_KINDS="")
        self.assertEqual(self._digest_kinds_in(gm), {})


class ParamDigestOutsideTheGraphTest(TestCase):
    """Parameter digests are read off the module, not captured in the graph."""

    def _emit(self, kind: str) -> Dict[Any, Any]:
        mod = _Toy()
        buf = io.StringIO()
        with redirect_stdout(buf):
            emit_param_digest(mod, kind)
        return parse_digest(buf.getvalue())

    def test_one_line_per_parameter(self) -> None:
        parsed = self._emit("winit")
        self.assertEqual(set(parsed), {(0, "winit", "w")})
        self.assertEqual(parsed[(0, "winit", "w")]["sum"], 4.0)

    def test_meta_parameters_are_skipped(self) -> None:
        """A meta parameter has no storage to reduce; digesting one raises."""
        mod = _Toy()
        with torch.device("meta"):
            mod.meta_w = torch.nn.Parameter(torch.ones(4))
        buf = io.StringIO()
        with redirect_stdout(buf):
            emit_param_digest(mod, "winit")
        self.assertEqual(set(parse_digest(buf.getvalue())), {(0, "winit", "w")})


class LossDigestNameTest(TestCase):
    """The loss digest has to carry the eager legs' name."""

    def test_the_name_matches_the_eager_legs(self) -> None:
        mod = _Toy()
        opt = torch.optim.SGD(mod.parameters(), lr=0.5, foreach=True)
        buf = io.StringIO()
        with unittest.mock.patch.dict(
            os.environ, {"PARITY_DIGEST_KINDS": "loss"}, clear=False
        ):
            with redirect_stdout(buf):
                arm(recir_train_step)(mod, opt, torch.arange(4.0))
        names = {
            name for _rank, kind, name in parse_digest(buf.getvalue()) if kind == "loss"
        }
        self.assertEqual(names, {"per_task_vector"})


class UpdateDigestTest(TestCase):
    """``weight - winit``: the step's actual update."""

    def test_it_reports_the_difference_not_the_new_value(self) -> None:
        mod = _Toy()  # w = ones(4)
        before = {n: p.detach().clone() for n, p in mod.named_parameters()}
        with torch.no_grad():
            mod.w.add_(0.5)
        buf = io.StringIO()
        with redirect_stdout(buf):
            emit_update_digest(mod, before)
        parsed = parse_digest(buf.getvalue())[(0, "update", "w")]
        self.assertEqual(parsed["sum"], 2.0)  # 4 x 0.5, not 4 x 1.5
        self.assertEqual(parsed["absmax"], 0.5)

    def test_an_orthogonal_update_is_visible_where_the_norm_is_not(self) -> None:
        """The whole reason this exists: a rotation leaves the norm alone."""
        mod = _Toy()
        with torch.no_grad():
            mod.w.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        before = {n: p.detach().clone() for n, p in mod.named_parameters()}
        with torch.no_grad():
            mod.w.copy_(torch.tensor([0.0, 1.0, 0.0, 0.0]))
        buf = io.StringIO()
        with redirect_stdout(buf):
            emit_update_digest(mod, before)
        # Same norm before and after, but the update is real.
        self.assertAlmostEqual(
            parse_digest(buf.getvalue())[(0, "update", "w")]["l2"], 2.0**0.5, places=5
        )

    def test_parameters_without_a_before_value_are_skipped(self) -> None:
        """A parameter that appeared after the snapshot has no update to
        report, and subtracting against nothing would be a fabricated one."""
        mod = _Toy()
        buf = io.StringIO()
        with redirect_stdout(buf):
            emit_update_digest(mod, {})
        self.assertEqual(parse_digest(buf.getvalue()), {})


class TrimInputsFollowKindsTest(TestCase):
    """The padding-trim inputs are prepared for gradout as well as act."""

    def test_gradout_alone_still_prepares_the_trim(self) -> None:
        """Behavioural, not a source grep: the gate has moved once already, and
        a grep for it reports a refactor as a regression."""
        batch_size = 4
        # (padded_b, cand_total, hist_slot, max_ctx) -- what the caller passes.
        padding_spec = (batch_size, batch_size * 2, 4, 1)
        batch_args = tuple(
            [torch.zeros(batch_size, 2)]
            + [torch.zeros(batch_size, dtype=torch.int32)] * 7
            + [torch.ones(batch_size, dtype=torch.int32)]
        )

        for kinds in ({"gradout"}, {"act"}, {"act", "gradout"}):
            num_candidates, ro_lengths, padded_sizes = _trim_inputs(
                batch_args, kinds, padding_spec
            )
            self.assertIsNotNone(num_candidates, f"{kinds} got no num_candidates")
            self.assertIsNotNone(ro_lengths, f"{kinds} got no ro_lengths")
            self.assertEqual(len(padded_sizes), 5, f"{kinds} got {padded_sizes}")

        # A kind that needs no trim must not pay for one.
        self.assertEqual(
            _trim_inputs(batch_args, {"loss"}, padding_spec), (None, None, [])
        )


class EagerCapturesTupleOutputsTest(TestCase):
    """The eager leg records tuple outputs, so the two legs pair."""

    class _Tuple(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = torch.nn.Parameter(torch.ones(2, 2))

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, str]:
            return x @ self.w, "cache"

    class _Wrap(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.inner(x)[0]

    def test_a_tuple_output_is_recorded(self) -> None:
        store: Dict[str, List[torch.Tensor]] = {}
        mod = self._Wrap(self._Tuple())
        handles = capture_activations(mod, store)
        try:
            mod(torch.randn(3, 2))
        finally:
            for h in handles:
                h.remove()
        self.assertIn("inner", store)
        self.assertEqual(store["inner"][0].shape, (3, 2))

    def test_a_non_tensor_output_records_nothing(self) -> None:
        """Better a gap than a fabricated entry that pairs against something
        unrelated."""

        class _NoTensor(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> Tuple[str, str]:
                return "a", "b"

        class _W(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.inner = _NoTensor()

            def forward(self, x: torch.Tensor) -> Tuple[str, str]:
                return self.inner(x)

        store: Dict[str, List[torch.Tensor]] = {}
        mod = _W()
        handles = capture_activations(mod, store)
        try:
            mod(torch.randn(2, 2))
        finally:
            for h in handles:
                h.remove()
        self.assertNotIn("inner", store)


class EagerCapturesScalarArgsTest(TestCase):
    """Scalar tensor arguments are recorded, as the compiled leg already does."""

    class _Scaled(torch.nn.Module):
        def forward(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            return x * scale

    class _Wrap(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inner = EagerCapturesScalarArgsTest._Scaled()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.inner(x, torch.tensor(1.0 / 266.0))

    def _store(self) -> Dict[str, List[torch.Tensor]]:
        store: Dict[str, List[torch.Tensor]] = {}
        mod = self._Wrap()
        handles = capture_activations(mod, store)
        try:
            mod(torch.randn(2, 2))
        finally:
            for h in handles:
                h.remove()
        return store

    def test_a_scalar_arg_is_recorded_by_position(self) -> None:
        store = self._store()
        self.assertIn("IN:inner.arg1", store)
        self.assertAlmostEqual(float(store["IN:inner.arg1"][0]), 1.0 / 266.0)

    def test_the_first_argument_is_still_the_plain_input(self) -> None:
        self.assertIn("IN:inner", self._store())

    def test_a_non_scalar_arg_is_not_recorded_as_one(self) -> None:
        """Only 0-D floats: a second full tensor argument is a different thing
        and would collide with the positional naming."""

        class _TwoT(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x + y

        class _W(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.inner = _TwoT()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.inner(x, torch.ones(2, 2))

        store: Dict[str, List[torch.Tensor]] = {}
        mod = _W()
        handles = capture_activations(mod, store)
        try:
            mod(torch.randn(2, 2))
        finally:
            for h in handles:
                h.remove()
        self.assertNotIn("IN:inner.arg1", store)


class EagerInputGradientIsAccumulatedTest(TestCase):
    """The eager leg reads the ACCUMULATED input gradient, as the compiled leg does."""

    class _TwoConsumers(torch.nn.Module):
        """x feeds both branches, so its gradient is a sum of two paths."""

        def __init__(self) -> None:
            super().__init__()
            self.a = torch.nn.Linear(4, 4)
            self.b = torch.nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.a(x) + self.b(x)

    class _Wrap(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inner = EagerInputGradientIsAccumulatedTest._TwoConsumers()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.inner(x)

    def _store(self) -> Dict[str, List[torch.Tensor]]:
        store: Dict[str, List[torch.Tensor]] = {}
        mod = self._Wrap()
        handles = capture_grad_outputs(mod, store)
        try:
            mod(torch.randn(3, 4, requires_grad=True)).sum().backward()
        finally:
            for h in handles:
                h.remove()
        return store

    def test_the_shared_input_gets_the_total_of_both_paths(self) -> None:
        store = self._store()
        self.assertIn("GRADIN:inner.a", store)
        self.assertIn("GRADIN:inner.b", store)
        # Both hooks sit on the same tensor, so both see the same total.
        torch.testing.assert_close(
            store["GRADIN:inner.a"][0], store["GRADIN:inner.b"][0]
        )

    def test_the_total_is_the_sum_not_one_branch(self) -> None:
        """The distinguishing property. A per-module grad_input would hand back
        one branch's contribution; the tensor hook has to hand back both."""
        store: Dict[str, List[torch.Tensor]] = {}
        mod = self._Wrap()
        x = torch.randn(3, 4, requires_grad=True)
        handles = capture_grad_outputs(mod, store)
        try:
            mod(x).sum().backward()
        finally:
            for h in handles:
                h.remove()
        # y = a(x) + b(x) and the loss is a plain sum, so dL/dx is
        # ones @ (Wa + Wb) -- each branch alone would give only its own term.
        expected = torch.ones(3, 4) @ (mod.inner.a.weight + mod.inner.b.weight)
        torch.testing.assert_close(store["GRADIN:inner.a"][0], expected)
