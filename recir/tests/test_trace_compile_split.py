# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Tests for the v7.1 lower-level public API: recir.trace + recir.compile_from_gm.

Exercises the split where ``trace`` runs stages 0-2 (eager + make_fx) and
``compile_from_gm`` runs stages 4-10 (finalize -> partition -> optimize ->
dispatch).  The smoke pipeline uses a no-op parallel mode + no-op backend to
keep tests CPU-only and dependency-light.
"""

import unittest

import recir
import torch
import torch.nn as nn
from recir.api import _get_trace_ctx, _TraceContext
from recir.distributed.parallel_mode import ParallelMode
from recir.tests.test_utils import _NoopBackend, _PassthroughParallel
from torch import fx
from torch.fx.experimental.proxy_tensor import make_fx


class _ToyModel(nn.Module):
    """Tiny model with one Linear layer."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).sum()


class _FrozenStateModel(nn.Module):
    running: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.frozen = nn.Parameter(torch.ones(4), requires_grad=False)
        self.trainable = nn.Parameter(torch.ones(4))
        self.register_buffer("running", torch.ones(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((self.frozen + self.trainable + self.running) * x).sum()


def _toy_train_step(
    model: nn.Module, opt: torch.optim.Optimizer, x: torch.Tensor
) -> None:
    opt.zero_grad()
    loss = model(x)
    loss.backward()
    opt.step()


class TraceReturnsGmAndFlatStateTest(unittest.TestCase):
    """recir.trace returns (gm, flat_state) and stashes a trace context."""

    def test_trace_returns_tuple(self) -> None:
        """trace() returns a (gm, flat_state) tuple of expected shape."""
        model = _ToyModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=True)
        x = torch.randn(2, 4)

        result = recir.trace(_toy_train_step, model, opt, x)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        gm, flat_state = result
        self.assertIsInstance(gm, fx.GraphModule)
        self.assertIsInstance(flat_state, list)
        self.assertGreater(len(flat_state), 0)

    def test_trace_flat_state_matches_placeholders(self) -> None:
        """flat_state is 1:1 with the leading placeholders of gm."""
        model = _ToyModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=True)
        x = torch.randn(2, 4)

        gm, flat_state = recir.trace(_toy_train_step, model, opt, x)
        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        self.assertGreaterEqual(len(placeholders), len(flat_state))

    def test_trace_stashes_context(self) -> None:
        """The returned gm carries a _TraceContext for compile_from_gm."""
        model = _ToyModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=True)
        x = torch.randn(2, 4)

        gm, _ = recir.trace(_toy_train_step, model, opt, x)
        ctx = _get_trace_ctx(gm)
        self.assertIsInstance(ctx, _TraceContext)
        self.assertIs(ctx.mod, model)
        self.assertEqual(ctx.opts, [opt])

    def test_trace_and_compile_mark_only_frozen_parameters_as_cacheable(self) -> None:
        model = _FrozenStateModel()
        opt = torch.optim.SGD([model.trainable], lr=0.01, foreach=True)
        x = torch.randn(4)

        gm, flat_state = recir.trace(_toy_train_step, model, opt, x)
        compiled = recir.compile_from_gm(
            gm,
            flat_state,
            parallel_mode=_PassthroughParallel(),
            backend=_NoopBackend(),
        )
        state_meta = {
            node.meta.get("fqn"): node.meta.get("is_frozen_parameter", False)
            for node in compiled.gm.graph.nodes
            if node.op == "placeholder" and node.meta.get("fqn") is not None
        }

        self.assertTrue(state_meta["frozen"])
        self.assertFalse(state_meta["trainable"])
        self.assertFalse(state_meta["running"])


def _two_optimizer_train_step(
    model: nn.Module,
    opt0: torch.optim.Optimizer,
    opt1: torch.optim.Optimizer,
    x: torch.Tensor,
) -> None:
    opt0.zero_grad()
    opt1.zero_grad()
    model(x).sum().backward()
    opt0.step()
    opt1.step()


class CompileFromGmTest(unittest.TestCase):
    """recir.compile_from_gm runs stages 4-10 end-to-end."""

    def test_compile_from_gm_returns_compiled_result(self) -> None:
        """compile_from_gm returns a _CompiledResult with gm and flat_state."""
        model = _ToyModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=True)
        x = torch.randn(2, 4)

        gm, flat_state = recir.trace(_toy_train_step, model, opt, x)
        compiled = recir.compile_from_gm(
            gm,
            flat_state,
            parallel_mode=_PassthroughParallel(),
            backend=_NoopBackend(),
        )
        self.assertIsNotNone(compiled.gm)
        self.assertIsInstance(compiled.gm, fx.GraphModule)
        self.assertIsNotNone(compiled.flat_state)
        self.assertIs(compiled.mod, model)
        self.assertEqual(compiled.opts, [opt])

    def test_compile_from_gm_invokes_optimize_passes(self) -> None:
        """Each pass in backend.optimize_passes is called on (gm, flat_state)."""
        invocations: list[fx.GraphModule] = []

        def _track(gm: fx.GraphModule) -> fx.GraphModule:
            invocations.append(gm)
            return gm

        class _TrackBackend(_NoopBackend):
            @classmethod
            def default_optimize_passes(cls):
                return (_track,)

        model = _ToyModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=True)
        x = torch.randn(2, 4)

        gm, flat_state = recir.trace(_toy_train_step, model, opt, x)
        recir.compile_from_gm(
            gm,
            flat_state,
            parallel_mode=_PassthroughParallel(),
            backend=_TrackBackend(),
        )
        self.assertEqual(len(invocations), 1)

    def test_compile_from_gm_invokes_dispatch(self) -> None:
        """backend.dispatch is called and its return is in compiled.gm."""
        invocations: list[fx.GraphModule] = []

        class _RecordBackend(_NoopBackend):
            def dispatch(self, gm: fx.GraphModule) -> nn.Module:
                invocations.append(gm)
                return gm

        model = _ToyModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=True)
        x = torch.randn(2, 4)

        gm, flat_state = recir.trace(_toy_train_step, model, opt, x)
        recir.compile_from_gm(
            gm,
            flat_state,
            parallel_mode=_PassthroughParallel(),
            backend=_RecordBackend(),
        )
        self.assertEqual(len(invocations), 1)

    def test_compile_from_gm_validates_with_parallel_mode(self) -> None:
        """backend.validate_with is invoked early and may raise."""

        class _StrictBackend(_NoopBackend):
            def validate_with(self, parallel_mode: ParallelMode) -> None:
                raise ValueError(f"deliberate reject: {type(parallel_mode).__name__}")

        model = _ToyModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=True)
        x = torch.randn(2, 4)

        gm, flat_state = recir.trace(_toy_train_step, model, opt, x)
        with self.assertRaises(ValueError) as ctx:
            recir.compile_from_gm(
                gm,
                flat_state,
                parallel_mode=_PassthroughParallel(),
                backend=_StrictBackend(),
            )
        self.assertIn("deliberate reject", str(ctx.exception))

    def test_compile_from_gm_preserves_optimizer_state_order(self) -> None:
        model = nn.Sequential(
            nn.Linear(4, 4, bias=False),
            nn.Linear(4, 4, bias=False),
        )
        first, second = list(model.parameters())
        second_opt = torch.optim.SGD([second], lr=0.01, momentum=0.9, foreach=True)
        first_opt = torch.optim.SGD([first], lr=0.01, momentum=0.9, foreach=True)
        second_buffer = torch.full_like(second, 7.0)
        first_buffer = torch.full_like(first, 3.0)
        second_opt.state[second]["momentum_buffer"] = second_buffer
        first_opt.state[first]["momentum_buffer"] = first_buffer
        x = torch.randn(2, 4)

        gm, flat_state = recir.trace(
            _two_optimizer_train_step, model, second_opt, first_opt, x
        )
        compiled = recir.compile_from_gm(
            gm,
            flat_state,
            parallel_mode=_PassthroughParallel(),
            backend=_NoopBackend(),
        )

        second_idx = next(
            idx
            for idx, value in enumerate(compiled.flat_state)
            if value is second_buffer
        )
        first_idx = next(
            idx
            for idx, value in enumerate(compiled.flat_state)
            if value is first_buffer
        )
        self.assertLess(second_idx, first_idx)

    def test_added_tail_survives_the_post_shard_rebuild(self) -> None:
        """The happy path: nothing is dropped, so the tail lands where it began.

        ``shard_params`` substitutes state objects, so ``compile_from_gm``
        rebuilds ``flat_state`` from the post-shard snapshot. The substituted
        object must still be the one the graph's placeholder is named for, and
        an entry appended by ``add_state`` must still sit past the base state
        rather than inside it.
        """
        model = _ToyModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, foreach=True)
        weight = model.linear.weight
        opt.state[weight]["momentum_buffer"] = torch.zeros_like(weight)
        x = torch.randn(2, 4)

        gm, flat_state = recir.trace(_toy_train_step, model, opt, x)
        traced_leaf_count = len(_get_trace_ctx(gm).state_fqns)
        # Stand in for what shard_params does: same structure, new object.
        replacement = torch.zeros_like(weight)
        opt.state[weight] = {"momentum_buffer": replacement}
        tail = torch.tensor(11.0)
        recir.add_state(gm, flat_state, "extra_input", tail)

        compiled = recir.compile_from_gm(
            gm,
            flat_state,
            parallel_mode=_PassthroughParallel(),
            backend=_NoopBackend(),
        )

        state_fqns = compiled.state_manager.state_fqns
        if state_fqns is None:
            self.fail("compile_from_gm did not populate state FQNs")
        replacement_index = next(
            index
            for index, value in enumerate(compiled.flat_state)
            if value is replacement
        )
        self.assertEqual(
            state_fqns[replacement_index],
            "[1]['linear.weight']['momentum_buffer']",
        )
        # The tail is still the first entry past the traced base state.
        self.assertIs(compiled.flat_state[traced_leaf_count], tail)
        self.assertEqual(len(compiled.flat_state), traced_leaf_count + 1)

    def test_a_dropped_optimizer_leaf_is_refused_not_absorbed_as_tail(
        self,
    ) -> None:
        """Inferring the tail boundary from a length silently shifts indices.

        Momentum on both parameters makes the traced base state four leaves:

            0 [0]['linear.weight']                      W
            1 [0]['linear.bias']                        B
            2 [1]['linear.weight']['momentum_buffer']   Mw
            3 [1]['linear.bias']['momentum_buffer']     Mb

        so ``add_state`` leaves ``flat_state = [W, B, Mw, Mb, T]``, and the
        graph has one placeholder per entry, in that order.

        Now drop the FIRST optimizer leaf, ``Mw``. The rebuild is three entries
        long, and a boundary inferred from that length starts the tail one slot
        early::

            rebuilt = [W, B, Mb]
            tail    = flat_state[3:] = [Mb, T]    # should have been [T]
            result  = [W, B, Mb, Mb, T]

        Same length as before and no error raised, but ``Mw`` is gone, ``Mb``
        appears twice, and the placeholder the graph wired to ``Mw`` is now fed
        ``Mb``. Everything downstream that pairs ``flat_state`` with an FQN --
        a checkpoint save, a restore -- is then off by one, silently.

        Reading the boundary off the traced FQN list instead makes the
        divergence visible, and it is refused rather than re-aligned.
        """
        model = _ToyModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, foreach=True)
        for param in model.parameters():
            opt.state[param]["momentum_buffer"] = torch.zeros_like(param)
        x = torch.randn(2, 4)

        gm, flat_state = recir.trace(_toy_train_step, model, opt, x)
        # Pin the scenario the docstring describes: if pytree ever reorders the
        # leaves, this fails here rather than quietly testing something else.
        self.assertEqual(
            _get_trace_ctx(gm).state_fqns,
            [
                "[0]['linear.weight']",
                "[0]['linear.bias']",
                "[1]['linear.weight']['momentum_buffer']",
                "[1]['linear.bias']['momentum_buffer']",
            ],
        )
        recir.add_state(gm, flat_state, "extra_input", torch.tensor(11.0))
        self.assertEqual(len(flat_state), 5)

        opt.state[model.linear.weight].clear()

        with self.assertRaisesRegex(
            RuntimeError, "Post-shard state structure no longer matches"
        ):
            recir.compile_from_gm(
                gm,
                flat_state,
                parallel_mode=_PassthroughParallel(),
                backend=_NoopBackend(),
            )

    def test_compile_from_gm_without_trace_ctx_raises(self) -> None:
        """A naked gm (not produced by recir.trace) raises a clear error."""

        class _Naked(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x

        gm = make_fx(_Naked(), tracing_mode="fake", _allow_non_fake_inputs=True)(
            torch.randn(2, 2)
        )
        with self.assertRaises(RuntimeError) as ctx:
            recir.compile_from_gm(
                gm,
                [],
                parallel_mode=_PassthroughParallel(),
                backend=_NoopBackend(),
            )
        self.assertIn("recir.trace", str(ctx.exception))


class TopLevelExportsTest(unittest.TestCase):
    """trace and compile_from_gm are reachable from the top-level package."""

    def test_trace_exported(self) -> None:
        self.assertTrue(callable(recir.trace))

    def test_compile_from_gm_exported(self) -> None:
        self.assertTrue(callable(recir.compile_from_gm))


def _make_gspmd_parallel(num_processes: int) -> ParallelMode:
    """Build a real ``GSPMDParallel`` (the test-shadow can't satisfy the
    ``isinstance(parallel_mode, GSPMDParallel)`` check inside
    ``_resolve_trace_batch_ratio``)."""
    from recir.distributed.parallel_mode import GSPMDParallel

    return GSPMDParallel(
        parallel_style="replicate",
        num_ranks=num_processes,
        num_processes=num_processes,
    )


class _BakedExpandModel(nn.Module):
    """Toy model that bakes ``inputs.shape[0]`` as a Python-int into
    ``expand``-style ops (mirrors the HF Gemma3 RoPE / causal-mask
    pattern that motivates Path C)."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]  # captured as Python int at trace time
        y = self.linear(x)
        # Mask shaped [1, 1, F] expanded onto batch via baked B.
        mask = torch.ones(1, 1, y.shape[-1])
        mask = mask.expand(B, 1, -1)
        return (y.unsqueeze(1) * mask).sum()


def _baked_train_step(
    model: nn.Module, opt: torch.optim.Optimizer, x: torch.Tensor
) -> None:
    opt.zero_grad()
    loss = model(x)
    loss.backward()
    opt.step()


class TraceParallelModeBatchExpandTest(unittest.TestCase):
    """``recir.trace(parallel_mode=GSPMDParallel(num_processes=N))`` bakes
    the *global* batch (N x per-host) into ``expand`` literals so HF
    modeling code doesn't end up with stale per-host literals after
    GSPMD inflates placeholder shapes."""

    def _first_expand_literal(self, gm: fx.GraphModule) -> int:
        """Return the int literal at axis 0 of the first ``aten.expand`` node."""
        aten = torch.ops.aten
        for node in gm.graph.nodes:
            if node.op != "call_function":
                continue
            if node.target is not aten.expand.default:
                continue
            sizes = node.args[1]
            if not isinstance(sizes, (list, tuple)) or not sizes:
                continue
            if not isinstance(sizes[0], int):
                continue
            return int(sizes[0])
        raise AssertionError("no aten.expand with int leading literal found")

    def test_trace_without_parallel_mode_preserves_local_batch(self) -> None:
        """Default behavior: per-host batch is what the trace sees."""
        model = _BakedExpandModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=True)
        x = torch.randn(4, 4)

        gm, _ = recir.trace(_baked_train_step, model, opt, x)
        self.assertEqual(self._first_expand_literal(gm), 4)

    def test_trace_with_gspmd_parallel_mode_expands_to_global_batch(self) -> None:
        """With GSPMD num_processes=N, baked literals reflect global batch."""
        model = _BakedExpandModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=True)
        x = torch.randn(4, 4)
        pm = _make_gspmd_parallel(num_processes=8)

        gm, _ = recir.trace(_baked_train_step, model, opt, x, parallel_mode=pm)
        # Trace sees 4 x 8 = 32 instead of the per-host 4.
        self.assertEqual(self._first_expand_literal(gm), 32)

    def test_trace_with_single_process_gspmd_is_noop(self) -> None:
        """GSPMD num_processes=1 leaves baked literals at per-host value."""
        model = _BakedExpandModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=True)
        x = torch.randn(4, 4)
        pm = _make_gspmd_parallel(num_processes=1)

        gm, _ = recir.trace(_baked_train_step, model, opt, x, parallel_mode=pm)
        # ratio=1 -> no transform applied.
        self.assertEqual(self._first_expand_literal(gm), 4)

    def test_trace_with_passthrough_parallel_mode_is_noop(self) -> None:
        """Non-GSPMD parallel modes don't trigger the trace-side expand."""
        model = _BakedExpandModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=True)
        x = torch.randn(4, 4)

        gm, _ = recir.trace(
            _baked_train_step, model, opt, x, parallel_mode=_PassthroughParallel()
        )
        # PassthroughParallel is not GSPMDParallel; ratio resolves to None.
        self.assertEqual(self._first_expand_literal(gm), 4)
