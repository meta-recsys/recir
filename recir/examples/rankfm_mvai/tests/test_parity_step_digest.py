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
from typing import Any, Dict, List, Optional, Sequence, Tuple
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
from recir.api import compile_from_gm, make_compiled_runner, trace
from recir.examples.rankfm_mvai.debug_digest_op import debug_digest
from recir.region_transforms.base import RegionTransform
from recir.tests.test_utils import _NoopBackend, _PassthroughParallel
from recir.training.compile import (
    _declared_tensor_attr_holders,
    LR_DENSE_BUFFER,
    LR_ITER_BUFFER,
    recir_train_step,
    WARMUP_ATTR,
)
from recir.training.optim import WarmupController
from torchrec.optim.warmup import WarmupPolicy, WarmupStage


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


class _ToyWarmup(torch.nn.Module):
    """``_Toy`` plus the two 0-d buffers the warmup schedule drives.

    Registered exactly as ``setup._build_lr_buffered_optimizer`` /
    ``_register_iter_buffer`` do it: non-persistent, on the top-level module, so
    they ride ``flat_state`` via ``named_buffers()`` as graph placeholders AND
    graph outputs.
    """

    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(4))
        self.register_buffer(
            LR_DENSE_BUFFER, torch.tensor(0.0, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            LR_ITER_BUFFER, torch.zeros((), dtype=torch.int64), persistent=False
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return (self.w * x).sum().reshape(1), {}


class WarmupAdvancesInGraphTest(TestCase):
    """The schedule has to move when the COMPILED step runs.

    Eager arithmetic is not enough evidence. A controller that holds its buffers
    as attributes captured at construction passes every eager test, because
    ``stateless._reparametrize_module`` cannot swap an attribute: the compiled
    step then writes pre-trace tensors while the graph reads placeholders
    nothing wrote, and the LR is frozen for the life of the job.

    So these compile the step and run it twice. That combination -- traced, then
    stepped more than once -- is what distinguishes graph state from a captured
    tensor, and no other test in this area does it.
    """

    BASE_LR: float = 4e-4
    MAX_ITERS: int = 37500
    START: float = 0.1

    def _stages(self) -> List[Any]:
        return [
            WarmupStage(
                policy=WarmupPolicy.LINEAR,
                max_iters=self.MAX_ITERS,
                value=self.START,
                lr_scale=1.0,
            )
        ]

    def _expected_lr(self, it: int) -> float:
        """torchrec's LINEAR multiplier; matches WarmupScheduleBaselineTest."""
        if it > self.MAX_ITERS:
            return self.BASE_LR
        return self.BASE_LR * (self.START + (1.0 - self.START) * it / self.MAX_ITERS)

    def _build(
        self, *, detached_tensors: bool = False
    ) -> Tuple[_ToyWarmup, torch.optim.Optimizer, WarmupController]:
        """A toy wired the way production is.

        ``detached_tensors=True`` is the NEGATIVE CONTROL: the controller holds
        0-d tensors that are NOT the module's buffers, so recir's identity match
        finds nothing to rewire and the writes cannot reach the graph. It pins
        the rule the mechanism runs on -- identity against the real params and
        buffers -- rather than merely observing that something got rewired.
        """
        mod = _ToyWarmup()
        opt = torch.optim.SGD(mod.parameters(), lr=0.5, foreach=True)
        if detached_tensors:
            iter_buffer = torch.zeros((), dtype=torch.int64)
            lr_buffer = torch.tensor(0.0, dtype=torch.float32)
        else:
            iter_buffer = getattr(mod, LR_ITER_BUFFER)
            lr_buffer = getattr(mod, LR_DENSE_BUFFER)
        warmup = WarmupController(
            iter_buffer=iter_buffer,
            dense_lr_buffer=lr_buffer,
            dense_base_lr=self.BASE_LR,
            dense_lr_warmup_schedule=self._stages(),
        )
        warmup.apply()  # seed iter 0, as setup.py does
        setattr(mod, WARMUP_ATTR, warmup)
        return mod, opt, warmup

    @staticmethod
    def _trace_only(
        mod: torch.nn.Module,
        opt: torch.optim.Optimizer,
        region_transforms: Sequence[Any] = (),
        holders: Optional[Sequence[Any]] = None,
    ) -> torch.fx.GraphModule:
        gm, _flat_state = trace(
            recir_train_step,
            mod,
            opt,
            torch.arange(4.0),
            region_transforms=region_transforms,
            _tensor_attr_holders=(
                _declared_tensor_attr_holders(mod) if holders is None else holders
            ),
        )
        return gm

    @staticmethod
    def _compile(
        mod: torch.nn.Module,
        opt: torch.optim.Optimizer,
        region_transforms: Sequence[Any] = (),
    ) -> Any:
        gm, flat_state = trace(
            recir_train_step,
            mod,
            opt,
            torch.arange(4.0),
            region_transforms=region_transforms,
            _tensor_attr_holders=_declared_tensor_attr_holders(mod),
        )
        compiled = compile_from_gm(
            gm,
            flat_state,
            parallel_mode=_PassthroughParallel(),
            backend=_NoopBackend(),
        )
        return make_compiled_runner(compiled)

    @staticmethod
    def _counter_is_graph_state(gm: torch.fx.GraphModule) -> bool:
        """Does the counter placeholder feed anything in the graph?

        Assert on the GRAPH, not on the counter's value, because on the CPU FX
        backend the value cannot tell the two cases apart. A controller holding
        a captured tensor still ends up mutating it: make_fx interns the real
        tensor as a graph CONSTANT, and eager FX execution really does run
        ``add_`` on it -- the same object the module holds -- so the counter
        climbs and everything looks fine.

        Under XLA that constant is baked into the HLO and the host never sees a
        write, which is why a real TPU run freezes where a CPU run of the same
        code does not. What distinguishes graph state from a baked constant is
        whether the PLACEHOLDER is what gets consumed -- the same distinction
        that makes a `_threefry_fold_in` call prove nothing about dropout unless
        a parameter reaches its operand.
        """
        counter = next(
            (
                n
                for n in gm.graph.nodes
                if n.op == "placeholder" and n.meta.get("fqn") == LR_ITER_BUFFER
            ),
            None,
        )
        if counter is None:
            raise AssertionError(
                f"no placeholder tagged {LR_ITER_BUFFER}; the buffer did not "
                "reach flat_state at all"
            )
        # Exclude the output node. Every buffer placeholder is echoed there --
        # ``stateless_func`` returns the whole ``buffers`` dict -- so counting
        # it would make this trivially true. Being ECHOED is precisely the
        # broken case: output == input is what froze the schedule. What matters
        # is whether some real op reads the placeholder.
        return any(
            counter in n.all_input_nodes for n in gm.graph.nodes if n.op != "output"
        )

    def test_the_counter_advances_once_per_compiled_step(self) -> None:
        """The property the whole design rests on: the counter is graph state,
        so running the compiled step is what moves it."""
        mod, opt, warmup = self._build()
        runner = self._compile(mod, opt)
        for expected in (1, 2, 3):
            runner(mod, opt, torch.arange(4.0))
            self.assertEqual(
                int(getattr(mod, LR_ITER_BUFFER)),
                expected,
                "the compiled step did not advance the warmup counter; the "
                "in-graph add_ is writing a tensor the graph cannot see",
            )
            self.assertEqual(warmup.iter, expected)

    def test_the_lr_ramps_across_compiled_steps(self) -> None:
        """Not just the counter: the LR derived from it has to land in the
        buffer the optimizer reads. A counter that advances while the LR stays
        put is the same bug one step further along."""
        mod, opt, _warmup = self._build()
        runner = self._compile(mod, opt)
        seeded = float(getattr(mod, LR_DENSE_BUFFER))
        self.assertAlmostEqual(seeded / self._expected_lr(0), 1.0, places=6)
        for expected_iter in (1, 2, 3):
            runner(mod, opt, torch.arange(4.0))
            got = float(getattr(mod, LR_DENSE_BUFFER))
            self.assertAlmostEqual(
                got / self._expected_lr(expected_iter),
                1.0,
                places=6,
                msg=f"LR wrong at iter {expected_iter}: {got:.9g}",
            )
        self.assertGreater(float(getattr(mod, LR_DENSE_BUFFER)), seeded)

    def test_captured_tensors_do_not_reach_the_graph(self) -> None:
        """NEGATIVE CONTROL -- tensors recir cannot match must not be rewired.

        The mechanism keys off identity against the module's real params and
        buffers. Hand the controller unrelated 0-d tensors and the counter
        placeholder must feed NOTHING: the graph carries a read-only input it
        echoes straight back out, which is exactly the shape the bug had.

        Note this control fails if you assert on the counter's VALUE instead.
        On the CPU FX backend a controller writing an unlifted tensor can still
        appear to work, because make_fx interns real tensors as graph constants
        and eager execution mutates them. It is XLA that turns the constant into
        a genuine constant and freezes the schedule. Asserting on the value
        would report the bug as fixed on CPU while 128 chips trained frozen.
        """
        mod, opt, _warmup = self._build(detached_tensors=True)
        gm = self._trace_only(mod, opt, holders=())
        self.assertFalse(
            self._counter_is_graph_state(gm),
            "the counter placeholder has a consumer even though the controller "
            "holds tensors that are not the module's buffers, so this control "
            "no longer distinguishes graph state from a baked constant",
        )

    def test_declaring_a_controller_recir_cannot_match_raises(self) -> None:
        mod, opt, _warmup = self._build(detached_tensors=True)
        with self.assertRaisesRegex(AssertionError, "nothing was rewired"):
            self._trace_only(mod, opt)

    def test_the_counter_placeholder_is_consumed_when_wired_normally(self) -> None:
        """The positive half of the control above, on the same measure.

        Together they are the whole fix in one property: a captured tensor that
        IS a module buffer gets rewired to the proxy and shows up as a real
        consumer of the placeholder; one that is not, does not.
        """
        mod, opt, _warmup = self._build()
        gm = self._trace_only(mod, opt)
        self.assertTrue(
            self._counter_is_graph_state(gm),
            "nothing in the graph consumes the counter placeholder, so the "
            "schedule is not graph state and will freeze once lowered",
        )

    def test_tracing_does_not_advance_the_schedule(self) -> None:
        """Compiling is not training.

        The rewiring is what makes this true: the trace writes proxies, so the
        real counter is untouched. Before it, make_fx's constant-propagation
        path ran ``add_`` on the real 0-d tensor for real (``add_`` / ``copy_``
        / ``_to_copy`` are all on torch's ``should_allow_numbers_as_tensors``
        list), leaving the schedule at iteration 1. That is why the frozen LR
        measured 240 ppm above ``base_lr * 0.1`` and was misread as a correct
        iter-0 seed.

        Armed with a RegionTransform so recir's eager discovery pass runs too
        (it is skipped when the transform list is empty). That pass invokes the
        real step against the real module, and reaches neither ``opt.step()``
        nor ``warmup.step()`` only because its ``torch.no_grad()`` makes
        ``loss.backward()`` throw first -- worth covering, since nothing states
        that ordering as a guarantee.
        """
        mod, opt, warmup = self._build()
        self.assertEqual(warmup.iter, 0)
        self._compile(mod, opt, region_transforms=[RegionTransform()])
        self.assertEqual(
            warmup.iter, 0, "compilation leaked an iteration into the schedule"
        )
        self.assertAlmostEqual(
            float(getattr(mod, LR_DENSE_BUFFER)) / self._expected_lr(0),
            1.0,
            places=6,
            msg="the LR buffer was left at a contaminated iteration",
        )
