#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""``recir::debug_digest``: identity, tracing survival, and emitted format.

Deliberately light. It imports the op and nothing else from this package, so it
does not drag in the prod xsurface chain -- the step-level instrumentation is
covered by ``test_parity_step_digest``, which does and pays for it in link time.

What a cluster run still has to prove: recir's op-splice (the stage that drops
top-level effects) and the real TPU mesh.
"""

from __future__ import annotations

import io
import os

# Force a MULTI-device CPU mesh before jax initializes (it latches the device
# list on first use, so this cannot move below the jax import). Single-device is
# what a devserver gives by default, and on one device the two properties these
# tests exist to check are both vacuous: a psum is the identity, and a sharded
# spec has nothing to split.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import unittest
from contextlib import redirect_stdout
from typing import Any, Dict, List, NamedTuple, Set, Tuple
from unittest import TestCase

import jax
import jax.numpy as jnp
import numpy as np
import torch
from hammer.v3.models.udd_fm_v3.parity.digest import digest_line, parse_digest
from jax.sharding import PartitionSpec as P
from pyre_extensions import none_throws

# Imported after the XLA_FLAGS above: this module pulls jax in, and jax latches
# its device list on first use.
from recir.examples.rankfm_mvai import debug_digest_op
from recir.examples.rankfm_mvai.debug_digest_op import (
    _drop_block_padding,
    _emit_on_host,
    _jdebug_digest,
    debug_digest,
    digest_activations_in_graph,
    digest_grads_via_hooks,
    param_is_sharded,
)
from torch.fx.experimental.proxy_tensor import make_fx


class DebugDigestOpTest(TestCase):
    def test_eager_is_a_true_identity(self) -> None:
        """Same tensor object out, values untouched.

        Not merely equal: an instrumented run has to be bit-identical to an
        uninstrumented one, or the digest is measuring the instrumentation.
        """
        x = torch.randn(4, 3)
        before = x.clone()
        with redirect_stdout(io.StringIO()):
            out = debug_digest(x, "weight", "t", sharded=False)
        self.assertIs(out, x)
        torch.testing.assert_close(x, before, rtol=0, atol=0)

    def test_eager_passes_gradients_through_unchanged(self) -> None:
        """The op sits on the backward path too, so a non-identity gradient
        would silently rescale everything downstream of every insertion point."""
        x = torch.randn(4, 3, requires_grad=True)
        with redirect_stdout(io.StringIO()):
            y = debug_digest(x, "act", "t", sharded=False)
        y.sum().backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x), rtol=0, atol=0)

    def test_emitted_line_matches_the_eager_emitter(self) -> None:
        """The op's line and ``digest_line``'s line are the same string.

        This is the property the whole comparison rests on: two legs whose
        emitters disagree differ by their instrumentation, not their models. It
        holds by construction today -- the op calls ``digest_line`` -- and this
        test is what keeps it that way if someone reimplements the format for
        the lowering.
        """
        x = torch.randn(5, 2, dtype=torch.float32)
        buf = io.StringIO()
        with redirect_stdout(buf):
            debug_digest(x, "grad", "some.param", sharded=False)
        self.assertEqual(buf.getvalue().strip(), digest_line("grad", "some.param", x))

    def test_emitted_line_is_parseable(self) -> None:
        """``parse_digest`` reads it out of a log unchanged, keyed the way the
        comparison keys it."""
        x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        buf = io.StringIO()
        with redirect_stdout(buf):
            debug_digest(x, "winit", "recblock.w", sharded=True)
        parsed = parse_digest(buf.getvalue())
        self.assertIn((0, "winit", "recblock.w"), parsed)
        self.assertEqual(parsed[(0, "winit", "recblock.w")]["count"], 6.0)
        self.assertEqual(parsed[(0, "winit", "recblock.w")]["sum"], 15.0)

    def test_rank_override_reaches_the_line(self) -> None:
        """The compiled leg emits under the mesh axis index, not the env rank;
        ``parse_digest`` keys on it, so a wrong rank pairs the wrong tensors."""
        line = digest_line("weight", "t", torch.ones(2), rank=3)
        self.assertIn(" rank=3 ", line)
        self.assertIn((3, "weight", "t"), parse_digest(line))

    def test_dtype_override_reaches_the_line(self) -> None:
        """bfloat16 is widened before it can become a torch tensor, so the dtype
        actually computed in has to be carried separately from the tensor."""
        line = digest_line("weight", "t", torch.ones(2), dtype_name="bfloat16")
        self.assertIn("dtype=bfloat16", line)

    def test_sharded_flag_follows_the_table_prefix(self) -> None:
        """Only the in_graph sharded tables are dim-0 sharded; everything else is
        replicated under GSPMDParallel("replicate"). Declaring a replicated
        tensor sharded would CUT it, and each device would digest 1/N of it
        while printing a complete-looking line."""
        self.assertTrue(
            param_is_sharded("recblock._embedding_collection._sharded_tables.big")
        )
        self.assertFalse(
            param_is_sharded("recblock._embedding_collection._dp_tables.small")
        )
        self.assertFalse(param_is_sharded("recblock.dense.weight"))

    def test_a_parameter_with_no_gradient_passes_through(self) -> None:
        """Autograd hands an UNDEFINED gradient -- None in Python -- to the hook
        for any parameter the step produced no gradient for. In a multi-task
        model that is routine, not exceptional: the first real backward hit it
        immediately. Digesting it crashes in the fake kernel
        (``empty_like(None)``), and the crash lands during backward, far from
        the instrumentation that caused it.
        """
        mod = torch.nn.Linear(3, 2)
        # 'grad' alone: the sink/anchor paths are not what is under test here.
        handles = digest_grads_via_hooks(mod, {"grad"})
        self.addCleanup(lambda: [h.remove() for h in handles])
        hooks = mod.weight._backward_hooks
        self.assertIsNotNone(hooks, "no hook was registered at all")
        hook = none_throws(hooks)[next(iter(none_throws(hooks)))]
        # A None gradient is exactly the case under test, so the type is wrong
        # on purpose: autograd hands one to the hook for a parameter the step
        # did not reach, and the hook has to return it untouched rather than
        # digest it.
        self.assertIsNone(
            hook(None),  # pyre-ignore[6]
            "a None gradient must pass straight back",
        )


class DebugDigestTracingTest(TestCase):
    """Survival through tracing and DCE -- the part that decides whether the
    digest reaches the lowering at all."""

    @staticmethod
    def _digest_nodes(gm: torch.fx.GraphModule) -> int:
        return sum(
            1
            for n in gm.graph.nodes
            if n.op == "call_function" and "debug_digest" in str(n.target)
        )

    def test_fake_tracing_does_not_emit_or_alter_the_result(self) -> None:
        """Under make_fx(tracing_mode="fake") -- how recir traces -- the op must
        stay a passthrough and must NOT print: the values there are fake, and a
        digest of a fake tensor is noise in the log at best."""

        def fn(t: torch.Tensor) -> torch.Tensor:
            return debug_digest(t * 2, "act", "traced", sharded=False) + 1

        buf = io.StringIO()
        with redirect_stdout(buf):
            gm = make_fx(fn, tracing_mode="fake")(torch.randn(3))
        self.assertEqual(buf.getvalue(), "", "traced a digest of fake values")

        x = torch.randn(3)
        with redirect_stdout(io.StringIO()):
            torch.testing.assert_close(gm(x), x * 2 + 1)

    def test_op_survives_tracing_as_an_opaque_node(self) -> None:
        """If it were inlined away there would be nothing left for the torchax
        lowering to replace, and the compiled leg would emit nothing."""

        def fn(t: torch.Tensor) -> torch.Tensor:
            return debug_digest(t, "act", "traced", sharded=True)

        with redirect_stdout(io.StringIO()):
            gm = make_fx(fn, tracing_mode="fake")(torch.randn(3))
        self.assertEqual(self._digest_nodes(gm), 1)

    def test_an_unread_digest_is_deleted_by_dce(self) -> None:
        """The failure mode the anchor exists for.

        A weight has no natural consumer for the identity, so digesting one and
        dropping the result leaves a node nothing reads -- and DCE removes it
        before the lowering ever sees it. The digest then does not appear, and
        nothing reports that it was dropped.
        """

        def fn(t: torch.Tensor) -> torch.Tensor:
            debug_digest(t, "weight", "dropped", sharded=False)
            return t * 2

        with redirect_stdout(io.StringIO()):
            gm = make_fx(fn, tracing_mode="fake")(torch.randn(3))
        self.assertEqual(self._digest_nodes(gm), 1)
        gm.graph.eliminate_dead_code()
        self.assertEqual(
            self._digest_nodes(gm), 0, "expected DCE to drop an unread digest"
        )

    def test_the_anchor_keeps_a_digest_alive_through_dce(self) -> None:
        """Same graph, but one element of the identity is threaded into an
        output -- which is what ``digest_params`` collects ``anchors`` for."""

        def fn(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            out = debug_digest(t, "weight", "anchored", sharded=False)
            return t * 2, out.detach().reshape(-1)[:1]

        with redirect_stdout(io.StringIO()):
            gm = make_fx(fn, tracing_mode="fake")(torch.randn(3))
        gm.graph.eliminate_dead_code()
        self.assertEqual(
            self._digest_nodes(gm), 1, "the anchor did not keep the digest alive"
        )


class DebugDigestLoweringTest(TestCase):
    """The torchax lowering, on a multi-device CPU jax mesh.

    The real target is TPU and only a cluster run proves the whole path, but the
    lowering body is plain jax and runs anywhere: this covers that the shard_map
    is well-formed for BOTH specs, that the host callback fires from inside it,
    and that what comes out is the line the eager legs emit. It does NOT cover
    recir's op-splice, which is the stage that drops top-level effects.

    Multi-device on purpose (see the XLA_FLAGS at the top of this file): on one
    device a stray psum and a wrong PartitionSpec are both invisible.
    """

    def _devices(self) -> int:
        n = len(jax.devices())
        if n < 2:
            self.skipTest(f"needs a multi-device mesh, got {jax.devices()}")
        return n

    def test_replicated_is_identity_and_emits_the_eager_line_per_device(self) -> None:
        """A replicated tensor is whole on every device, so every device's line
        describes the WHOLE tensor and equals what the eager emitter produces
        for it."""
        n = self._devices()
        x = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
        buf = io.StringIO()
        with redirect_stdout(buf):
            out = _jdebug_digest(x, "weight", "t", False)
            jax.block_until_ready(out)
        self.assertTrue(bool(jnp.all(out == x)), "lowering was not an identity")

        ref = digest_line(
            "weight", "t", torch.arange(6, dtype=torch.float32).reshape(2, 3)
        )
        lines = sorted(ln.strip() for ln in buf.getvalue().splitlines() if ln.strip())
        self.assertEqual(len(lines), n, f"expected one line per device: {lines}")
        for rank, line in enumerate(lines):
            # Identical to the eager line except for the rank it is tagged with.
            self.assertEqual(line, ref.replace("rank=0", f"rank={rank}", 1))

    def test_replicated_is_not_summed_across_the_mesh(self) -> None:
        """Under GSPMDParallel("replicate") the dense params, loss and
        activations are identical on every device. A psum would report N times
        the true value -- which is why nothing here reduces across the mesh."""
        n = self._devices()
        buf = io.StringIO()
        with redirect_stdout(buf):
            out = _jdebug_digest(jnp.ones((4,), dtype=jnp.float32), "loss", "l", False)
            jax.block_until_ready(out)
        parsed = parse_digest(buf.getvalue())
        self.assertEqual(len(parsed), n)
        for rank in range(n):
            fields = parsed[(rank, "loss", "l")]
            # 4 ones on EVERY device. 4*n would mean a psum crept in; 4/n would
            # mean the replicated tensor was split.
            self.assertEqual(fields["sum"], 4.0)
            self.assertEqual(fields["count"], 4.0)

    def test_sharded_digests_each_device_its_own_block(self) -> None:
        """The other spec, and the one with a real consequence: a dim-0 sharded
        table must be digested per shard, because that is what light's per-rank
        digests describe. Each device sees 1/N of the rows and reports them.
        """
        n = self._devices()
        rows = 4 * n
        x = jnp.ones((rows, 2), dtype=jnp.float32) * jnp.arange(
            rows, dtype=jnp.float32
        ).reshape(rows, 1)
        buf = io.StringIO()
        with redirect_stdout(buf):
            out = _jdebug_digest(x, "weight", "tbl", True)
            jax.block_until_ready(out)
        self.assertTrue(bool(jnp.all(out == x)), "lowering was not an identity")

        parsed = parse_digest(buf.getvalue())
        self.assertEqual(len(parsed), n)
        total = 0.0
        for rank in range(n):
            fields = parsed[(rank, "weight", "tbl")]
            # Each device holds rows/n rows of width 2.
            self.assertEqual(fields["count"], float(rows // n * 2))
            total += fields["sum"]
        # The shards partition the table: they sum to the whole, with no overlap
        # and nothing dropped.
        self.assertAlmostEqual(total, float(2 * sum(range(rows))), places=3)


class ActivationDigestTest(TestCase):
    """Module-boundary digests, the compiled leg's answer to the eager legs'
    ``capture_activations``.

    They have to pair with what light already emits, so the traversal (leaf
    modules), the naming (``IN:<fqn>`` and ``<fqn>``) and the per-invocation
    ``act[i]`` kind all have to match -- a digest the comparison cannot pair is
    the same as no digest.
    """

    class _Two(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = torch.nn.Linear(3, 3)
            self.b = torch.nn.Linear(3, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.b(self.a(x))

    def _emit(self, **env: str) -> str:
        mod = self._Two()
        handles = []
        buf = io.StringIO()
        with unittest.mock.patch.dict(os.environ, env, clear=False):
            handles = digest_activations_in_graph(mod, {"act"})
        try:
            with redirect_stdout(buf):
                mod(torch.randn(4, 3))
        finally:
            for h in handles:
                h.remove()
        return buf.getvalue()

    def test_both_input_and_output_are_digested_per_leaf(self) -> None:
        """Outputs alone show that an op disagrees but not whether its INPUT
        already did -- the one question that localises a bug to an op."""
        parsed = parse_digest(self._emit())
        names = {name for _rank, _kind, name in parsed}
        self.assertEqual(names, {"a", "b", "IN:a", "IN:b"})

    def test_the_kind_carries_the_invocation_index(self) -> None:
        """``act[i]``, matching digest.emit_digest. A module called twice must
        not overwrite itself: the two legs can call it a different number of
        times, and last-call-wins would compare different calls silently."""
        kinds = {kind for _rank, kind, _name in parse_digest(self._emit())}
        self.assertEqual(kinds, {"act[0]"})

    def test_the_substring_filter_selects_a_subtree(self) -> None:
        """Each activation digest is gathered to the global tensor and shipped
        to the host, so digesting every leaf is not free."""
        parsed = parse_digest(self._emit(PARITY_DIGEST_ACT_SUBSTR="b"))
        self.assertEqual({name for _r, _k, name in parsed}, {"b", "IN:b"})

    def test_the_hooks_do_not_change_the_result(self) -> None:
        """They rewrite the module's args and output with the op's identity, so
        an instrumented forward must still produce the uninstrumented value."""
        mod = self._Two()
        x = torch.randn(4, 3)
        expected = mod(x)
        handles = digest_activations_in_graph(mod, {"act"})
        try:
            with redirect_stdout(io.StringIO()):
                got = mod(x)
        finally:
            for h in handles:
                h.remove()
        torch.testing.assert_close(got, expected, rtol=0, atol=0)

    def test_off_unless_act_is_requested(self) -> None:
        self.assertEqual(digest_activations_in_graph(self._Two(), {"grad"}), [])


class ActivationPaddingTrimTest(TestCase):
    """Trimming the padded tail before reducing, on the host.

    The compiled leg's activations are STATIC-SHAPE: the HSTU layers get a
    [B*S_max, D] buffer where the eager leg has a jagged [sum(lengths), D] one,
    and the padded rows are not zeros (``pad_jagged_values`` can tile the real
    values into them). Reducing across those two compares different tensors --
    measured as ``sum`` 29% apart while ``l2`` looked fine to 0.6%.

    The first attempt at this shipped without a test and the wiring silently
    did not apply: the run came back with the padded count unchanged, which
    only showed up after a cluster round.
    """

    class _Leaf(torch.nn.Module):
        """Shaped like the HSTU layers: ``forward(x, x_offsets, ...)``."""

        def forward(
            self, x: torch.Tensor, offsets: torch.Tensor, max_len: int = 3
        ) -> torch.Tensor:
            return x

    class _Wrapped(torch.nn.Module):
        """A parent, because the traversal hooks LEAF modules and the root's
        own name is empty -- the same rule the eager leg's capture uses."""

        def __init__(self, child: torch.nn.Module) -> None:
            super().__init__()
            self.leaf = child

        def forward(self, x: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
            return self.leaf(x, offsets, 3)

    def _run(self, rows: int, real: int) -> Dict[Any, Any]:
        mod = self._Wrapped(self._Leaf())
        # Two real samples then a slack-absorbing dummy, the layout
        # pad_jagged_lengths produces per chip.
        offsets = torch.tensor([0, real // 2, real, rows], dtype=torch.int64)
        handles = digest_activations_in_graph(mod, {"act"})
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                mod(torch.ones(rows, 4), offsets)
        finally:
            for h in handles:
                h.remove()
        return parse_digest(buf.getvalue())

    def test_the_digest_covers_only_the_real_rows(self) -> None:
        """Lengths [3, 3, 4] against a per-sample bound of 3: the last entry is
        the dummy that absorbed the block's slack, so its rows go."""
        parsed = self._run(rows=10, real=6)
        # 6 real rows x 4 columns, not 10 x 4.
        self.assertEqual(parsed[(0, "act[0]", "IN:leaf")]["count"], 24.0)

    def test_the_offsets_are_digested_too(self) -> None:
        """So the trim can be checked against the eager leg rather than
        trusted: absmax is the last offset, which must equal the real rows."""
        parsed = self._run(rows=10, real=6)
        self.assertIn((0, "act[0]", "IN:leaf.offsets"), parsed)
        self.assertEqual(parsed[(0, "act[0]", "IN:leaf.offsets")]["absmax"], 10.0)

    def test_samples_within_the_bound_are_all_kept(self) -> None:
        """Nothing is dropped when no sample exceeds the bound -- an eager leg
        has no dummies, and trimming one would be a silent data loss."""
        mod = self._Wrapped(self._Leaf())
        handles = digest_activations_in_graph(mod, {"act"})
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                mod(torch.ones(8, 4), torch.tensor([0, 3, 6, 8], dtype=torch.int64))
        finally:
            for h in handles:
                h.remove()
        parsed = parse_digest(buf.getvalue())
        self.assertEqual(parsed[(0, "act[0]", "IN:leaf")]["count"], 32.0)

    def test_a_module_without_offsets_is_untrimmed(self) -> None:
        """Most leaf modules take no offsets, and treating an unrelated second
        argument as if it were them would trim the wrong rows."""
        mod = torch.nn.Sequential(torch.nn.Linear(4, 2))
        handles = digest_activations_in_graph(mod, {"act"})
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                mod(torch.ones(10, 4))
        finally:
            for h in handles:
                h.remove()
        parsed = parse_digest(buf.getvalue())
        self.assertEqual(parsed[(0, "act[0]", "IN:0")]["count"], 40.0)


class ActivationDigestSurvivesDCETest(TestCase):
    """The activation hooks' digests must survive dead-code elimination.

    Both of them. The module-input digest is load-bearing by construction (it
    replaces the argument), but the OFFSETS probe next to it was written as a
    bare call whose result was discarded -- dead code, deleted before lowering.
    The pod then logged no probe at all, which read as "the offsets were not
    found" when they were sitting in args[1] the whole time. One cluster round.
    """

    class _Leaf(torch.nn.Module):
        def forward(self, x: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
            return x * 2

    class _Wrapped(torch.nn.Module):
        def __init__(self, child: torch.nn.Module) -> None:
            super().__init__()
            self.leaf = child

        def forward(self, x: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
            return self.leaf(x, offsets)

    def test_both_the_input_and_the_offsets_digest_survive(self) -> None:
        mod = self._Wrapped(self._Leaf())
        handles = digest_activations_in_graph(mod, {"act"})
        self.addCleanup(lambda: [h.remove() for h in handles])
        x, offsets = torch.randn(10, 4), torch.tensor([0, 4, 8], dtype=torch.int32)

        with redirect_stdout(io.StringIO()):
            gm = make_fx(mod, tracing_mode="fake")(x, offsets)
        gm.graph.eliminate_dead_code()

        digests = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and "debug_digest" in str(n.target)
        ]
        # input + offsets + the leaf's output digest.
        self.assertGreaterEqual(
            len(digests), 2, f"a digest was DCE'd: {[str(n.target) for n in digests]}"
        )
        # The offsets ride in as an OPERAND of the input digest rather than as
        # a node of their own, so they cannot be eliminated separately.
        by_name = {str(n.args[2]): n for n in digests}
        self.assertIn("IN:leaf", by_name)
        # x, kind, name, sharded, offsets, max_len, num_candidates, ro_lengths, padded_sizes, num_devices
        self.assertEqual(
            len(by_name["IN:leaf"].args),
            10,
            "an operand was dropped, so the host cannot trim or report it",
        )
        self.assertIsNotNone(
            by_name["IN:leaf"].args[4], "the offsets operand was dropped"
        )


class DebugDigestOffsetsLoweringTest(TestCase):
    """The offsets operand, through the real lowering on a CPU mesh.

    Three cluster rounds were spent on a digest that came back untrimmed with
    no offsets line, each time chasing a different suspect -- a stale image, a
    wrong argument index, DCE -- when the whole path is plain jax and runs
    here in a second.
    """

    def _devices(self) -> int:
        n = len(jax.devices())
        if n < 2:
            self.skipTest(f"needs a multi-device mesh, got {jax.devices()}")
        return n

    def test_offsets_reach_the_host_and_trim_the_tail(self) -> None:
        self._devices()
        x = jnp.arange(20, dtype=jnp.float32).reshape(10, 2)
        # [3, 3, 4] with a bound of 3: the last is the dummy.
        offsets = jnp.array([0, 3, 6, 10], dtype=jnp.int32)
        buf = io.StringIO()
        with redirect_stdout(buf):
            out = _jdebug_digest(x, "act[0]", "t", False, offsets, 3)
            jax.block_until_ready(out)

        self.assertTrue(bool(jnp.all(out == x)), "lowering was not an identity")
        parsed = parse_digest(buf.getvalue())
        self.assertIn((0, "act[0]", "t.offsets"), parsed, f"no offsets line: {parsed}")
        self.assertEqual(parsed[(0, "act[0]", "t.offsets")]["absmax"], 10.0)
        # 6 real rows x 2 columns, not 10 x 2.
        self.assertEqual(parsed[(0, "act[0]", "t")]["count"], 12.0)

    def test_without_a_bound_nothing_is_dropped(self) -> None:
        """The bound is what separates a real sample from a dummy. Without it
        the safe answer is to keep everything: dropping rows on a guess would
        be a silent, unrecoverable difference in the digest."""
        self._devices()
        x = jnp.arange(20, dtype=jnp.float32).reshape(10, 2)
        offsets = jnp.array([0, 3, 6, 10], dtype=jnp.int32)
        buf = io.StringIO()
        with redirect_stdout(buf):
            jax.block_until_ready(_jdebug_digest(x, "act[0]", "t", False, offsets))
        parsed = parse_digest(buf.getvalue())
        self.assertEqual(parsed[(0, "act[0]", "t")]["count"], 20.0)


class DebugDigestUnderJitTest(TestCase):
    """Compiling the digest inside a jit'd, sharded program.

    The lowering tests above call the op directly, which exercises the
    shard_map body but not the SPMD partitioner. The partitioner is where the
    compiled leg actually failed:

        RET_CHECK (spmd_partitioner.cc:6464)
        side-effect HLO cannot have a replicated sharding:
        %debug_callback = send(...) is_host_transfer=true, sharding={replicated}

    and it took a cluster round to see it. This puts the same compile step
    within reach of a devserver.
    """

    def _mesh(self) -> Any:
        n = len(jax.devices())
        if n < 2:
            self.skipTest(f"needs a multi-device mesh, got {jax.devices()}")
        return jax.make_mesh((n,), ("torch_dist",))

    def test_a_replicated_digest_compiles_under_jit(self) -> None:
        mesh = self._mesh()
        rep = jax.sharding.NamedSharding(mesh, P())

        def fn(w: Any) -> Any:
            return _jdebug_digest(w, "weight", "w", False) * 2.0

        w = jax.device_put(jnp.arange(16, dtype=jnp.float32).reshape(4, 4), rep)
        with redirect_stdout(io.StringIO()):
            compiled = (
                jax.jit(fn, in_shardings=rep, out_shardings=rep).lower(w).compile()
            )
            out = compiled(w)
        # The partitioner is the subject here, but a compiled program that does
        # not also produce the right value would be a hollow pass.
        self.assertTrue(bool(jnp.all(out == w * 2.0)))

    def test_a_digest_whose_result_is_unused_compiles(self) -> None:
        """The weight digests are kept alive by an anchor rather than by a
        consumer, so their value reaches an output while the digested tensor
        itself feeds nothing."""
        mesh = self._mesh()
        rep = jax.sharding.NamedSharding(mesh, P())

        def fn(w: Any) -> Any:
            d = _jdebug_digest(w, "weight", "w", False)
            return d[0, :1]

        w = jax.device_put(jnp.arange(16, dtype=jnp.float32).reshape(4, 4), rep)
        with redirect_stdout(io.StringIO()):
            compiled = (
                jax.jit(fn, in_shardings=rep, out_shardings=rep).lower(w).compile()
            )
            out = compiled(w)
        # The slice is all that survives, which is the point: the digest itself
        # feeds nothing and still has to reach the lowering.
        self.assertTrue(bool(jnp.all(out == w[0, :1])))


class EmitOnHostArgsTest(TestCase):
    """``_emit_on_host`` with the arguments the lowering says it passes.

    Sits between the trim arithmetic (unit-tested) and the lowering
    (mesh-tested) so a failure names which of the two is wrong instead of
    leaving both open.
    """

    def test_candidate_total_reaches_the_trim(self) -> None:
        shard = np.arange(32 * 2, dtype=np.float32).reshape(32, 2)
        counts = np.ones(16, dtype=np.int32)
        hist = np.zeros(16, dtype=np.int32)
        buf = io.StringIO()
        with redirect_stdout(buf):
            _emit_on_host(
                "act[0]",
                "t",
                "float32",
                0,
                shard,
                None,
                0,
                counts,
                hist,
                [0, 32, 0, 0],
                8,
            )
        parsed = parse_digest(buf.getvalue())
        self.assertEqual(parsed[(0, "act[0]", "t")]["count"], 32.0, buf.getvalue())


class DropBlockPaddingUnitTest(TestCase):
    """The trim arithmetic on its own, with no jax in the way.

    Separates "the helper is wrong" from "the operand never arrived", which
    cost a round of guessing when only a lowering-level test existed and both
    were still possible.

    ``padded_sizes`` is positional by meaning: [sample axis, candidate axis, history
    axis, history+candidate axis], matching the four count tensors the helper
    builds from ``num_candidates`` and ``ro_lengths``.
    """

    NC: torch.Tensor = torch.ones(16, dtype=torch.int32)
    HIST: torch.Tensor = torch.full((16,), 2, dtype=torch.int32)

    def test_the_worked_example_in_the_docstring(self) -> None:
        """Pins the example `_drop_block_padding`'s docstring is written around,
        so the two cannot drift: 2 chips x 4 slots, 3 real rows in each."""
        x = torch.arange(8 * 2, dtype=torch.float32).reshape(8, 2)
        num_candidates = torch.tensor([1, 2, 1, 2], dtype=torch.int32)
        ro_lengths = torch.zeros(4, dtype=torch.int32)
        out = _drop_block_padding(x, num_candidates, ro_lengths, [4, 8, 0, 0], 2)
        self.assertEqual(list(out.shape), [6, 2])
        # Rows 3 and 7 are the padding at the back of each block.
        self.assertEqual(out[:, 0].tolist(), x[[0, 1, 2, 4, 5, 6], 0].tolist())

    def test_unreducible_ro_lengths_skip_the_trim_instead_of_raising(self) -> None:
        """The fallback has to survive its own caller.

        When the RO lengths cannot be reduced to per-sample counts,
        ``_history_tokens_per_sample`` hands them back raw rather than raising,
        because an exception inside the host callback takes the whole run down.
        The first version of that guard was defeated two lines later: the
        caller added the raw tensor to ``num_candidates`` and raised on the
        shape anyway. An untrimmed digest is the intended outcome -- it shows up
        as a size mismatch in the comparison, which is visible and not fatal.
        """
        x = torch.arange(8 * 2, dtype=torch.float32).reshape(8, 2)
        num_candidates = torch.tensor([1, 2, 1, 2], dtype=torch.int32)
        # 5 is neither 4 nor a multiple of it.
        ro_lengths = torch.ones(5, dtype=torch.int32)
        out = _drop_block_padding(x, num_candidates, ro_lengths, [4, 8, 0, 0], 2)
        self.assertTrue(
            torch.equal(out, x), "expected the tensor back untrimmed, not a raise"
        )

    def test_candidate_axis_on_dim0(self) -> None:
        x = torch.arange(32 * 2, dtype=torch.float32).reshape(32, 2)
        out = _drop_block_padding(x, self.NC, self.HIST, [0, 32, 0, 0], 8)
        self.assertEqual(out.shape[0], 16)

    def test_candidate_axis_on_dim1(self) -> None:
        """Most candidate-axis activations are [G, C, D], so the axis to cut is
        not dim 0."""
        x = torch.arange(3 * 32 * 2, dtype=torch.float32).reshape(3, 32, 2)
        out = _drop_block_padding(x, self.NC, self.HIST, [0, 32, 0, 0], 8)
        self.assertEqual(list(out.shape), [3, 16, 2])

    def test_sample_axis_uses_one_per_sample(self) -> None:
        """24 slots, 16 real samples: two of every three kept."""
        x = torch.arange(24 * 4, dtype=torch.float32).reshape(24, 4)
        out = _drop_block_padding(x, self.NC, self.HIST, [24, 0, 0, 0], 8)
        self.assertEqual(out.shape[0], 16)

    def test_history_axis_uses_the_history_lengths(self) -> None:
        """Two history tokens per sample, so 4 real rows in each 5-row block."""
        x = torch.arange(40 * 2, dtype=torch.float32).reshape(40, 2)
        out = _drop_block_padding(x, self.NC, self.HIST, [0, 0, 40, 0], 8)
        self.assertEqual(out.shape[0], 32)

    def test_history_plus_candidates_axis(self) -> None:
        """The two counts add: 2 history + 1 candidate = 3 real per sample."""
        x = torch.arange(48 * 2, dtype=torch.float32).reshape(48, 2)
        out = _drop_block_padding(x, self.NC, self.HIST, [0, 0, 0, 48], 8)
        self.assertEqual(out.shape[0], 48 // 8 // 6 * 0 + 48)

    def test_an_axis_matching_nothing_is_left_alone(self) -> None:
        """The guard a divisibility rule would fail: dim 1 here is bigger and
        divides evenly, but it is features, not samples."""
        x = torch.arange(24 * 64, dtype=torch.float32).reshape(24, 64)
        out = _drop_block_padding(x, self.NC, self.HIST, [0, 0, 0, 0], 8)
        self.assertEqual(list(out.shape), [24, 64])


class TrimThroughTheLoweringTest(TestCase):
    """The same trim, reached through the real shard_map + host callback.

    The unit test above proves the arithmetic; this proves the metadata
    survives the operand plumbing. Both exist because a failure in only one of
    them names which half is broken.
    """

    def _devices(self) -> int:
        n = len(jax.devices())
        if n < 2:
            self.skipTest(f"needs a multi-device mesh, got {jax.devices()}")
        return n

    def test_candidate_axis_metadata_reaches_the_host(self) -> None:
        n = self._devices()
        rows = 4 * n
        x = jnp.arange(rows * 2, dtype=jnp.float32).reshape(rows, 2)
        num_candidates = jnp.ones((2 * n,), dtype=jnp.int32)
        hist = jnp.zeros((2 * n,), dtype=jnp.int32)
        buf = io.StringIO()
        with redirect_stdout(buf):
            out = _jdebug_digest(
                x,
                "act[0]",
                "t",
                False,
                None,
                0,
                num_candidates,
                hist,
                [0, rows, 0, 0],
                n,
            )
            jax.block_until_ready(out)
        self.assertTrue(bool(jnp.all(out == x)), "lowering was not an identity")
        parsed = parse_digest(buf.getvalue())
        # 2 real rows per block x n blocks x 2 columns.
        self.assertEqual(
            parsed[(0, "act[0]", "t")]["count"], float(2 * n * 2), buf.getvalue()
        )

    def test_without_metadata_nothing_is_dropped(self) -> None:
        """Trimming on a guess would be a silent, unrecoverable difference."""
        n = self._devices()
        rows = 4 * n
        x = jnp.arange(rows * 2, dtype=jnp.float32).reshape(rows, 2)
        buf = io.StringIO()
        with redirect_stdout(buf):
            jax.block_until_ready(_jdebug_digest(x, "act[0]", "t", False))
        parsed = parse_digest(buf.getvalue())
        self.assertEqual(parsed[(0, "act[0]", "t")]["count"], float(rows * 2))


class ActivationsEmitFromRankZeroOnlyTest(TestCase):
    """Activations are digested once, other kinds once per device.

    Activations are always replicated -- the hook digests them whole so they
    pair with the eager leg's -- so the other ranks would repeat the same line,
    and the trim plus reduction is expensive: eightfold over 226 tensors, the
    largest [143520, 224], took a run from four minutes to nineteen and killed
    it.

    The per-device emission is kept for every other kind, because that is what
    makes a stray psum or a wrong PartitionSpec visible at all -- there is a
    test above that relies on exactly that.
    """

    def _lines(self, kind: str) -> int:
        n = len(jax.devices())
        if n < 2:
            self.skipTest(f"needs a multi-device mesh, got {jax.devices()}")
        x = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
        buf = io.StringIO()
        with redirect_stdout(buf):
            jax.block_until_ready(_jdebug_digest(x, kind, "t", False))
        return len([ln for ln in buf.getvalue().splitlines() if ln.strip()])

    def test_an_activation_is_emitted_once(self) -> None:
        self.assertEqual(self._lines("act[0]"), 1)

    def test_a_gradient_is_emitted_once(self) -> None:
        """Gradients are all-reduced before the hook sees them, so the other
        ranks only repeat the line -- and eight ranks writing 2224 of them to
        one stream dropped 177, which reads as missing gradients."""
        self.assertEqual(self._lines("grad"), 1)

    def test_a_weight_is_still_emitted_per_device(self) -> None:
        self.assertEqual(self._lines("weight"), len(jax.devices()))


class HistoryLengthsTrimTest(TestCase):
    """The history axis, whose per-sample counts need work before they fit.

    ``ro_lengths`` holds the history length BEFORE truncation. The measured run
    had ten per-event keys summing to 1,499,943 -- 2930 per sample -- against a
    real 129,214, because the model caps each sample at hstu_max_seq_len
    downstream. Feeding the raw numbers to the trim overran the per-chip block
    and index_select raised "index out of range" inside the host callback,
    which killed the run rather than producing a bad digest.
    """

    def test_lengths_are_clamped_to_the_slot(self) -> None:
        """Alternating raw lengths of 9 and 1 against a slot of 3: the 9s clamp
        to 3 and the 1s stay, so each 6-row block holds 4 real rows. Padding
        exists only because some samples fall SHORT of the slot -- when every
        sample is longer, the clamped total fills the extent exactly and there
        is nothing to drop, which is why the real run's 129,214 sits just under
        512 x 256."""
        x = torch.arange(48 * 2, dtype=torch.float32).reshape(48, 2)
        num_candidates = torch.zeros(16, dtype=torch.int32)
        raw = torch.tensor([9, 1] * 8, dtype=torch.int32)
        # padded_sizes: [padded_b, cand, history, history+cand]; slot = 48 // 16 = 3.
        out = _drop_block_padding(x, num_candidates, raw, [16, 0, 48, 0], 8)
        self.assertEqual(out.shape[0], 32)

    def test_counts_that_overrun_the_block_skip_the_trim(self) -> None:
        """Better an untrimmed digest, which the comparison flags as a size
        mismatch, than an exception in the callback."""
        x = torch.arange(32 * 2, dtype=torch.float32).reshape(32, 2)
        num_candidates = torch.full((16,), 99, dtype=torch.int32)
        hist = torch.zeros(16, dtype=torch.int32)
        out = _drop_block_padding(x, num_candidates, hist, [0, 32, 0, 0], 8)
        self.assertEqual(list(out.shape), [32, 2])


class GradOutputDigestTest(TestCase):
    """The gradient ARRIVING at each leaf module, in reverse topological order.

    Parameter gradients say WHICH parameters disagree; this says where the
    disagreement STARTS, and everything after the first divergence is
    downstream contamination. The eager leg has captured it all along
    (digest.capture_grad_outputs, 121 entries on a udd_fm_v3 run) and the
    compiled leg had no equivalent, so the whole backward was a single opaque
    step: the only reverse-side measurement was the endpoint.

    It exists because of a case the endpoint could not explain. Both the input
    AND the output of ``contextual_tokenizer._compressions.0`` matched exactly
    (1.000000), while the two parameters feeding the ``baddbmm`` just before it
    disagreed by 4% and 10%. A forward that agrees exactly cannot produce
    parameter gradients that disagree unless the gradient arriving from
    DOWNSTREAM already did.
    """

    class _Two(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = torch.nn.Linear(3, 3)
            self.b = torch.nn.Linear(3, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.b(self.a(x))

    def _emit(self, kinds: Set[str]) -> str:
        mod = self._Two()
        buf = io.StringIO()
        handles = digest_activations_in_graph(mod, kinds)
        try:
            with redirect_stdout(buf):
                mod(torch.randn(4, 3)).sum().backward()
        finally:
            for h in handles:
                h.remove()
        return buf.getvalue()

    def test_one_per_leaf_module(self) -> None:
        """The OUTPUT side: one per leaf. The input side is GRADIN:<fqn> and is
        covered by GradInDigestTest."""
        parsed = parse_digest(self._emit({"gradout"}))
        got = {
            (kind.split("[")[0], name)
            for _rank, kind, name in parsed
            if not name.startswith("GRADIN:")
        }
        self.assertEqual(got, {("gradout", "a"), ("gradout", "b")})

    def test_it_can_run_without_the_activation_digests(self) -> None:
        """They are separately expensive -- activations are gathered whole --
        so localising a backward must not require paying for the forward."""
        kinds = {k for _r, k, _n in parse_digest(self._emit({"gradout"}))}
        self.assertTrue(all(k.startswith("gradout") for k in kinds), kinds)

    def test_act_alone_emits_no_gradout(self) -> None:
        kinds = {k for _r, k, _n in parse_digest(self._emit({"act"}))}
        self.assertTrue(all(k.startswith("act") for k in kinds), kinds)

    def test_the_name_matches_the_eager_leg(self) -> None:
        """The eager leg keys on the bare module FQN, with the invocation index
        in the kind. A digest the comparison cannot pair is the same as none."""
        parsed = parse_digest(self._emit({"gradout"}))
        self.assertIn((0, "gradout[0]", "a"), parsed)


class GradOutputTupleOutputTest(TestCase):
    """Modules whose forward returns a tuple are covered too.

    The HSTU attention layers return an HSTUCacheState tuple, so a bare-tensor
    check skipped all 14 of them -- and they are precisely the region the
    backward evidence points at. The one divergence the first gradout run found
    (0.9004 arriving at contextual_tokenizer._compressions.0, matching that
    module's parameter gradients exactly) has to come from downstream, and
    downstream of the contextual tokens IS the HSTU stack.

    The eager leg reads ``grad_output[0]``, so taking the first tensor of the
    output pairs with it.
    """

    class _Tuple(torch.nn.Module):
        """A LEAF module returning a tuple -- no child modules, because only
        leaves are hooked (the rule the eager leg uses too), and the HSTU
        attention layers are leaves that return HSTUCacheState."""

        def __init__(self) -> None:
            super().__init__()
            self.w = torch.nn.Parameter(torch.ones(3, 3))

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, str]:
            # Second element is not a tensor, as in HSTUCacheState.
            return x @ self.w, "cache"

    class _Wrap(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.inner(x)[0]

    def test_a_tuple_returning_module_is_hooked(self) -> None:
        mod = self._Wrap(self._Tuple())
        buf = io.StringIO()
        handles = digest_activations_in_graph(mod, {"gradout"})
        try:
            with redirect_stdout(buf):
                mod(torch.randn(4, 3)).sum().backward()
        finally:
            for h in handles:
                h.remove()
        names = {name for _r, _k, name in parse_digest(buf.getvalue())}
        self.assertIn("inner", names)

    def test_a_tupleless_output_is_still_skipped(self) -> None:
        """A module returning no tensor at all has no gradient to report, and
        inventing one would be worse than the gap."""

        class _NoTensor(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> str:
                return "nothing"

        mod = _NoTensor()
        buf = io.StringIO()
        handles = digest_activations_in_graph(mod, {"gradout"})
        try:
            with redirect_stdout(buf):
                mod(torch.randn(2, 2))
        finally:
            for h in handles:
                h.remove()
        self.assertEqual(parse_digest(buf.getvalue()), {})


class GradOutIsTrimmedLikeActivationsTest(TestCase):
    """A gradient carries the padding of the tensor it belongs to.

    Activations get the per-chip dummy rows dropped before they are reduced;
    the gradient arriving at the same module has the same shape and needs the
    same treatment. Emitted untrimmed, a [520, 512, 10] gradient is reduced
    against the eager leg's [512, 512, 10] and the L2 ratio compares different
    element counts. That is how a 0.9048 got read at the contextual tokenizer
    with no way to tell how much of it was padding -- and a whole round of
    analysis was spent on it before the counts were looked at.
    """

    def test_the_gradient_is_trimmed_before_it_is_reduced(self) -> None:
        n = len(jax.devices())
        if n < 2:
            self.skipTest(f"needs a multi-device mesh, got {jax.devices()}")
        # One real sample per chip in a two-row-per-chip block: the sample axis
        # is 2n padded, n real.
        # Wrapped: only NAMED leaves are hooked, so a bare Linear as the root
        # module (name "") is skipped -- the same rule the eager leg uses.
        mod = torch.nn.Sequential(torch.nn.Linear(4, 4))
        num_candidates = torch.ones(n, dtype=torch.int32)
        hist = torch.zeros(n, dtype=torch.int32)
        buf = io.StringIO()
        handles = digest_activations_in_graph(
            mod,
            {"gradout"},
            num_candidates=num_candidates,
            ro_lengths=hist,
            padded_sizes=[2 * n, 0, 0, 0],
        )
        try:
            with redirect_stdout(buf):
                mod(torch.randn(2 * n, 4)).sum().backward()
        finally:
            for h in handles:
                h.remove()
        parsed = parse_digest(buf.getvalue())
        got = [v for (_r, k, _nm), v in parsed.items() if k.startswith("gradout")]
        self.assertTrue(got, "no gradout digest was emitted")
        # n real rows x 4 features, NOT 2n x 4.
        self.assertEqual(got[0]["count"], float(n * 4))


class TokenAxisTrimTest(TestCase):
    """The fifth padded axis: history + candidates + contextual.

    This is the sequence the HSTU stack operates on, and it was the one axis
    with no extent. All 14 attention layers therefore had their outputs and
    their gradients reduced over 148720 rows against the eager leg's 137491 --
    not one of them was comparable, so the entire HSTU backward was invisible
    while every other module paired.

    The contextual count is derived from the two widest padded_sizes rather than
    passed, which keeps the op's signature fixed.
    """

    def test_the_token_axis_is_trimmed(self) -> None:
        # 2 chips. Per sample: history 3 (slot 4), candidates 2, contextual 1.
        # Token slot = 4 + 2 + 1 = 7; per chip 2 samples -> extent 4*7 = 28.
        n_local, b = 2, 4
        num_candidates = torch.full((b,), 2, dtype=torch.int32)
        hist = torch.full((b,), 3, dtype=torch.int32)
        padded_sizes = [b, b * 2, b * 4, b * (4 + 2), b * (4 + 2 + 1)]
        x = torch.arange(padded_sizes[4] * 2, dtype=torch.float32).reshape(
            padded_sizes[4], 2
        )
        out = _drop_block_padding(x, num_candidates, hist, padded_sizes, n_local)
        # Real rows per sample: 3 + 2 + 1 = 6, over 4 samples.
        self.assertEqual(out.shape[0], 24)

    def test_the_narrower_axes_still_win_when_they_match(self) -> None:
        """The token extent must not shadow history+candidate: they are
        checked in order, and a tensor on the narrower axis has to keep
        getting the narrower count."""
        n_local, b = 2, 4
        num_candidates = torch.full((b,), 2, dtype=torch.int32)
        hist = torch.full((b,), 3, dtype=torch.int32)
        padded_sizes = [b, b * 2, b * 4, b * (4 + 2), b * (4 + 2 + 1)]
        x = torch.arange(padded_sizes[3] * 2, dtype=torch.float32).reshape(
            padded_sizes[3], 2
        )
        out = _drop_block_padding(x, num_candidates, hist, padded_sizes, n_local)
        self.assertEqual(out.shape[0], 4 * 5)  # (3 + 2) per sample

    def test_a_four_element_extents_list_still_works(self) -> None:
        """Older callers pass four; the contextual term is then absent, not
        garbage."""
        n_local, b = 2, 4
        num_candidates = torch.full((b,), 2, dtype=torch.int32)
        hist = torch.full((b,), 3, dtype=torch.int32)
        padded_sizes = [b, b * 2, b * 4, b * (4 + 2)]
        x = torch.arange(padded_sizes[3] * 2, dtype=torch.float32).reshape(
            padded_sizes[3], 2
        )
        out = _drop_block_padding(x, num_candidates, hist, padded_sizes, n_local)
        self.assertEqual(out.shape[0], 4 * 5)


class TupleOutputActivationTest(TestCase):
    """A tuple-returning module's OUTPUT activation is digested too.

    The HSTU attention layers return HSTUCacheState, so the bare-tensor check
    skipped their outputs for the whole exercise -- only their INPUTS ever
    paired, and every "the forward is exact" statement about those 14 layers
    rested on inputs alone.

    The hook must NOT return the digest for these: its return value replaces
    the module's entire output, so handing back a bare tensor where the caller
    unpacks a tuple breaks the model instead of measuring it.
    """

    class _Tuple(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = torch.nn.Parameter(torch.ones(3, 3))

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, str]:
            return x @ self.w, "cache"

    class _Wrap(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, tag = self.inner(x)
            assert tag == "cache", "the hook replaced a tuple output"
            return out

    def _run(self, kinds: Set[str]) -> str:
        mod = self._Wrap(self._Tuple())
        buf = io.StringIO()
        handles = digest_activations_in_graph(mod, kinds)
        try:
            with redirect_stdout(buf):
                mod(torch.randn(4, 3)).sum().backward()
        finally:
            for h in handles:
                h.remove()
        return buf.getvalue()

    def test_the_output_is_digested(self) -> None:
        got = {(k.split("[")[0], n) for _r, k, n in parse_digest(self._run({"act"}))}
        self.assertIn(("act", "inner"), got)

    def test_the_tuple_still_reaches_the_caller(self) -> None:
        """If the hook returned the bare digest, _Wrap's unpack would raise --
        so the model breaks instead of quietly measuring the wrong thing."""
        emitted = parse_digest(self._run({"act"}))
        self.assertTrue(emitted, "the module did not survive the hook")

    def test_a_named_tuple_keeps_its_type(self) -> None:
        """HSTUCacheState is a NamedTuple, so the rebuilt output has to be one
        too -- a plain tuple would break every caller that uses field access."""

        # lint-ignore: NoNamedTuple - HSTUCacheState is a NamedTuple and the
        # code under test branches on ``_fields`` to rebuild it. A dataclass is
        # neither a tuple nor has _fields, so it would not reach that path --
        # the suggested alternative cannot test the thing being tested.
        class _State(NamedTuple):
            values: torch.Tensor
            tag: str

        class _Named(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w = torch.nn.Parameter(torch.ones(3, 3))

            def forward(self, x: torch.Tensor) -> _State:
                return _State(values=x @ self.w, tag="cache")

        class _W(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.inner = _Named()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                st = self.inner(x)
                assert isinstance(st, _State), f"lost the NamedTuple: {type(st)}"
                return st.values

        mod = _W()
        handles = digest_activations_in_graph(mod, {"act"})
        try:
            with redirect_stdout(io.StringIO()):
                mod(torch.randn(4, 3)).sum().backward()
        finally:
            for h in handles:
                h.remove()

    def test_the_digest_is_load_bearing(self) -> None:
        """Discarded, the digest is consumed by nothing and DCE deletes it --
        which is exactly why these layers emitted a gradout and no act. The
        digested tensor must be the one the caller goes on to use."""
        mod = self._Wrap(self._Tuple())
        seen: List[torch.Tensor] = []
        real = debug_digest_op.debug_digest

        def spy(x: Any, *a: Any, **k: Any) -> Any:
            out = real(x, *a, **k)
            seen.append(out)
            return out

        with unittest.mock.patch.object(debug_digest_op, "debug_digest", spy):
            handles = digest_activations_in_graph(mod, {"act"})
            try:
                with redirect_stdout(io.StringIO()):
                    got = mod(torch.randn(4, 3))
            finally:
                for h in handles:
                    h.remove()
        self.assertTrue(seen, "no digest ran")
        self.assertIs(got, seen[-1])

    def test_act_and_gradout_together(self) -> None:
        got = {
            k.split("[")[0] for _r, k, _n in parse_digest(self._run({"act", "gradout"}))
        }
        self.assertEqual(got, {"act", "gradout"})


class GradInDigestTest(TestCase):
    """The gradient LEAVING a module, not just the one arriving.

    With outputs alone, a module that AMPLIFIES a disagreement is
    indistinguishable from one that passes it on. At the contextual tokenizer
    1.4% arrives at ``_compressions.0`` and 11% comes out at the baddbmm's
    bias, and the compression's own backward sits unmeasured between them --
    so the split cannot be attributed without this.

    Named ``GRADIN:<fqn>`` to pair with the eager leg, whose
    register_full_backward_hook was handed grad_input all along and stored only
    grad_output.
    """

    class _Two(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = torch.nn.Linear(3, 3)
            self.b = torch.nn.Linear(3, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.b(self.a(x))

    def _kinds(self, kinds: Set[str]) -> Dict[str, Set[str]]:
        mod = self._Two()
        buf = io.StringIO()
        handles = digest_activations_in_graph(mod, kinds)
        try:
            with redirect_stdout(buf):
                # requires_grad on the input, so the FIRST leaf has an incoming
                # gradient too. Without it that module has none -- correctly,
                # but it makes the chain look like it has a hole.
                mod(torch.randn(4, 3, requires_grad=True)).sum().backward()
        finally:
            for h in handles:
                h.remove()
        out: Dict[str, Set[str]] = {}
        for _r, k, n in parse_digest(buf.getvalue()):
            out.setdefault(k.split("[")[0], set()).add(n)
        return out

    def test_both_sides_of_each_module_are_digested(self) -> None:
        got = self._kinds({"gradout"})
        self.assertEqual(got["gradout"], {"a", "b", "GRADIN:a", "GRADIN:b"})

    def test_act_alone_emits_neither(self) -> None:
        self.assertNotIn("gradout", self._kinds({"act"}))

    def test_the_input_side_is_a_different_tensor(self) -> None:
        """GRADIN:b and a are the same point in the graph reached from two
        different modules; both must appear, or the chain has a hole exactly at
        a module boundary -- which is where the amplification question lives."""
        got = self._kinds({"gradout"})["gradout"]
        self.assertIn("GRADIN:b", got)
        self.assertIn("a", got)

    def test_the_name_is_not_double_prefixed(self) -> None:
        """The pre-hook's own key already carries "IN:"; reusing it produced
        GRADIN:IN:<fqn>, which pairs with nothing on the eager leg."""
        got = self._kinds({"gradout"})["gradout"]
        self.assertFalse([n for n in got if n.startswith("GRADIN:IN:")], got)


class _Desc:
    """Stand-in for FeatureDescriptor: the probe keys on ``.name``."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name


class BundleOutputGradientTest(TestCase):
    """Modules returning an EmbeddingBundle get their embeddings hooked.

    The composer tokenizer returns one. It is neither a Tensor nor a tuple, and
    it is not a leaf module either, so BOTH the leaf loop and the output-type
    check skipped it -- on both legs. That left the boundary between the
    tokenizers and the HSTU stack with no backward measurement at all, and the
    first divergence sits exactly there: 13 of 14 attention layers grow the
    gradient 1.006-1.042x, while the tensor entering layer 0 grows it 1.58x.
    """

    class _Bundle:
        def __init__(self, **kw: torch.Tensor) -> None:
            self._embeddings: Dict[Any, torch.Tensor] = {
                _Desc(k): v for k, v in kw.items()
            }

    class _Inner(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = torch.nn.Linear(3, 3)

        def forward(self, x: torch.Tensor) -> Any:
            return BundleOutputGradientTest._Bundle(seq=self.lin(x))

    class _Wrap(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inner = BundleOutputGradientTest._Inner()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # One call: each call builds fresh _Desc keys, so indexing one
            # bundle with another's key raises.
            bundle = self.inner(x)
            return next(iter(bundle._embeddings.values()))

    def _names(self, kinds: Set[str]) -> Set[str]:
        mod = self._Wrap()
        buf = io.StringIO()
        handles = digest_activations_in_graph(mod, kinds)
        try:
            with redirect_stdout(buf):
                mod(torch.randn(4, 3)).sum().backward()
        finally:
            for h in handles:
                h.remove()
        return {n for _r, _k, n in parse_digest(buf.getvalue())}

    def test_a_bundle_embedding_is_hooked(self) -> None:
        """The module is NOT a leaf, so this only works via the dedicated
        pass -- the leaf loop cannot reach it."""
        self.assertIn("inner#seq", self._names({"gradout"}))

    def test_the_key_carries_the_feature_name(self) -> None:
        """A bundle can hold several embeddings; without the feature name in
        the key they would collide and last-writer-wins would pick one."""
        got = [n for n in self._names({"gradout"}) if n.startswith("inner#")]
        self.assertEqual(got, ["inner#seq"])

    def test_act_alone_hooks_no_bundle(self) -> None:
        self.assertFalse(
            [n for n in self._names({"act"}) if "#" in n],
        )
