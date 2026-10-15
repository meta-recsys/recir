# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Unit coverage for DropoutPRNGTransform and recir::bernoulli_keyed.

Covers the two failure modes that let three previous attempts at this look
verified: a key that is actually a constant, and a rewrite that silently does
nothing. It does NOT cover
the compiled path -- the end-to-end check is still
``grep -ci fold_in`` plus ``grep '_threefry_split' | grep -c parameter`` on a
``RECIR_DUMP_GRAPHS=full`` run, and the >=3-step ``compile_gspmd_step`` test
PROD_CONFIG_OVERRIDES.md C4 asks for does not exist yet.
"""

import re
from unittest import TestCase

import jax
import jax.numpy as jnp
import torch
import torch.fx as fx
import torch.nn.functional as F
from recir.region_transforms.dropout_prng import DropoutPRNGTransform
from recir.torchax.ops.jrandom import _jbernoulli_keyed
from torch.fx.experimental.proxy_tensor import make_fx


class _M(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("_lr_iter_buffer", torch.zeros((), dtype=torch.int64))
        self.lin = torch.nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.dropout(self.lin(x), p=0.1, training=True)
        return F.dropout(y, p=0.1, training=True).sum()


def _trace() -> tuple[fx.GraphModule, list[str]]:
    m = _M()
    params = dict(m.named_parameters(remove_duplicate=False))
    buffers = dict(m.named_buffers(remove_duplicate=False))

    def f(params: object, buffers: object, x: torch.Tensor) -> object:
        from torch.nn.utils import stateless

        with stateless._reparametrize_module(m, {**params, **buffers}):  # pyre-ignore
            return m(x), params, buffers

    gm = make_fx(f, tracing_mode="fake")(params, buffers, torch.randn(2, 4))
    fqns = list(params.keys()) + list(buffers.keys())
    i = 0
    for n in gm.graph.nodes:
        if n.op != "placeholder":
            continue
        if i < len(fqns):
            n.meta["fqn"] = fqns[i]
        i += 1
    return gm, fqns


def _targets(gm: fx.GraphModule) -> list[str]:
    return [str(n.target) for n in gm.graph.nodes if n.op == "call_function"]


class TestDropoutPRNGSmoke(TestCase):
    def test_op_depends_on_counter_and_site(self) -> None:
        noise = torch.empty(4096)
        a = torch.ops.recir.bernoulli_keyed(noise, 0.9, torch.tensor(0), 0, 0)
        b = torch.ops.recir.bernoulli_keyed(noise, 0.9, torch.tensor(1), 0, 0)
        c = torch.ops.recir.bernoulli_keyed(noise, 0.9, torch.tensor(0), 1, 0)
        again = torch.ops.recir.bernoulli_keyed(noise, 0.9, torch.tensor(0), 0, 0)
        self.assertEqual(a.dtype, noise.dtype)
        self.assertTrue(torch.equal(a, again), "same inputs must be reproducible")
        self.assertFalse(torch.equal(a, b), "counter must change the mask")
        self.assertFalse(torch.equal(a, c), "site must change the mask")
        self.assertGreater(float(a.mean()), 0.8)
        self.assertLess(float(a.mean()), 1.0)

    def test_transform_rewrites_and_threads_counter(self) -> None:
        gm, _ = _trace()
        before = _targets(gm)
        self.assertEqual(sum("bernoulli" in t for t in before), 2, before)

        DropoutPRNGTransform(counter_fqn="_lr_iter_buffer").finalize_compiled_graph(gm)

        after = _targets(gm)
        self.assertEqual(sum("aten.bernoulli" in t for t in after), 0, after)
        keyed = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and "bernoulli_keyed" in str(n.target)
        ]
        self.assertEqual(len(keyed), 2)
        counter = next(
            n
            for n in gm.graph.nodes
            if n.op == "placeholder" and n.meta.get("fqn") == "_lr_iter_buffer"
        )
        # The whole point: the counter is a real argument of every draw, and the
        # per-site index differs.
        self.assertEqual([n.args[2] for n in keyed], [counter, counter])
        self.assertEqual([n.args[3] for n in keyed], [0, 1])
        for n in keyed:
            self.assertIn("val", n.meta)
            self.assertFalse(n.meta.get("xla_graph_break", False))

    def test_no_counter_raises_rather_than_degrading(self) -> None:
        gm, _ = _trace()
        with self.assertRaisesRegex(RuntimeError, "no placeholder is tagged"):
            DropoutPRNGTransform(counter_fqn="_nonexistent").finalize_compiled_graph(gm)

    def test_unrewritable_draw_raises_rather_than_being_skipped(self) -> None:
        """A draw this transform cannot rewrite must fail loudly.

        Silently stepping over it leaves that draw on torchax's Environment key
        -- a constant baked at construction time -- so every step gets the same
        mask and nothing raises. That is indistinguishable from success without
        reading the HLO, which is how three earlier attempts passed review.
        """
        for target, args in (
            (torch.ops.aten.native_dropout.default, (0.1, True)),
            (torch.ops.aten.bernoulli_.Tensor, None),
            (torch.ops.aten.bernoulli.default, ()),
        ):
            with self.subTest(target=str(target)):
                gm, _ = _trace()
                src = next(n for n in gm.graph.nodes if n.op == "placeholder")
                extra = (src,) if args is None else args
                with gm.graph.inserting_before(
                    next(n for n in gm.graph.nodes if n.op == "output")
                ):
                    gm.graph.call_function(target, args=(src, *extra))
                with self.assertRaisesRegex(RuntimeError, "cannot rewrite"):
                    DropoutPRNGTransform(
                        counter_fqn="_lr_iter_buffer"
                    ).finalize_compiled_graph(gm)

    def test_unrewritable_draw_raises_even_with_no_other_dropout(self) -> None:
        """The check runs before the no-dropout early return."""
        gm, _ = _trace()
        for n in list(gm.graph.nodes):
            if "bernoulli" in str(n.target):
                n.replace_all_uses_with(n.args[0])
                gm.graph.erase_node(n)
        src = next(n for n in gm.graph.nodes if n.op == "placeholder")
        with gm.graph.inserting_before(
            next(n for n in gm.graph.nodes if n.op == "output")
        ):
            gm.graph.call_function(
                torch.ops.aten.native_dropout.default, args=(src, 0.1, True)
            )
        gm.recompile()
        with self.assertRaisesRegex(RuntimeError, "cannot rewrite"):
            DropoutPRNGTransform(counter_fqn="_lr_iter_buffer").finalize_compiled_graph(
                gm
            )

    def test_no_dropout_is_a_noop(self) -> None:
        gm, _ = _trace()
        for n in list(gm.graph.nodes):
            if "bernoulli" in str(n.target):
                n.replace_all_uses_with(n.args[0])
                gm.graph.erase_node(n)
        gm.recompile()
        before = _targets(gm)
        DropoutPRNGTransform(counter_fqn="_nonexistent").finalize_compiled_graph(gm)
        self.assertEqual(_targets(gm), before)


def _param_feeds_threefry(hlo: str) -> bool:
    """True iff a func argument reaches a ``_threefry_fold_in`` operand in @main.

    Walks the SSA def-use chain of the ``@main`` block: taint the arguments, then
    propagate through every ``%v = op %operands`` line, and report whether any
    ``call @_threefry_fold_in`` consumes a tainted value.
    """
    lines: list[str] = []
    depth = 0
    for raw in hlo.splitlines():
        if raw.startswith("  func.func"):
            depth = 1 if "@main" in raw else 0
            continue
        if depth:
            lines.append(raw)

    tainted = set(re.findall(r"%arg\d+", "\n".join(lines)))
    for line in lines:
        lhs, sep, rhs = line.partition("=")
        if not sep:
            continue
        operands = set(re.findall(r"%[A-Za-z_0-9]+", rhs))
        if operands & tainted:
            tainted.update(re.findall(r"%[A-Za-z_0-9]+", lhs))
    for line in lines:
        if "_threefry_fold_in" in line or "_threefry_split" in line:
            _, _, rhs = line.partition("=")
            if set(re.findall(r"%[A-Za-z_0-9]+", rhs or line)) & tainted:
                return True
    return False


class TestJaxLowering(TestCase):
    """The lowering runs under jax.jit here, which is the stage that matters.

    Three of the failed attempts produced a key that was a compile-time constant.
    The check that distinguishes that from graph state is whether the counter is
    a traced ARGUMENT of the jitted function -- so jit over the counter, and
    assert the output actually varies with it.
    """

    def test_key_is_traced_from_the_counter(self) -> None:
        noise = jnp.zeros((4096,), dtype=jnp.float32)

        @jax.jit
        def run(counter: jax.Array) -> jax.Array:
            return _jbernoulli_keyed(noise, 0.9, counter, 0, 0)

        a = run(jnp.asarray(0, dtype=jnp.int32))
        b = run(jnp.asarray(1, dtype=jnp.int32))
        self.assertEqual(a.dtype, jnp.float32)
        self.assertFalse(bool(jnp.array_equal(a, b)), "counter must reach the key")
        self.assertTrue(bool(jnp.array_equal(a, run(jnp.asarray(0, jnp.int32)))))
        self.assertGreater(float(a.mean()), 0.85)
        self.assertLess(float(a.mean()), 0.95)

    def test_counter_reaches_threefry_in_the_lowered_hlo(self) -> None:
        """The end-to-end verification gate, run in-process.

        The presence of a threefry call proves nothing -- a constant key produces
        one too. What distinguishes graph state from a baked constant is whether
        a PARAMETER feeds the key derivation, so walk the SSA def-use chain from
        ``%arg0`` and require it to reach a ``_threefry_fold_in`` operand. The
        negative control below is the point of the exercise: it is the shape all
        three earlier attempts had, and it must NOT pass.
        """
        noise = jnp.zeros((256,), dtype=jnp.float32)
        keyed_from_arg = (
            jax.jit(lambda c: _jbernoulli_keyed(noise, 0.9, c, 0, 0))
            .lower(jnp.asarray(0, dtype=jnp.int32))
            .as_text()
        )
        # Negative control: counter closed over as a Python constant, which is
        # exactly what a key baked at Environment-construction time looks like.
        const_counter = jnp.asarray(0, dtype=jnp.int32)
        keyed_from_const = (
            jax.jit(lambda x: _jbernoulli_keyed(x, 0.9, const_counter, 0, 0))
            .lower(noise)
            .as_text()
        )
        self.assertTrue(_param_feeds_threefry(keyed_from_arg))
        self.assertFalse(
            _param_feeds_threefry(keyed_from_const),
            "the check cannot tell a traced key from a constant one",
        )

    def test_site_index_decorrelates_call_sites(self) -> None:
        noise = jnp.zeros((4096,), dtype=jnp.float32)
        c = jnp.asarray(7, dtype=jnp.int32)
        s0 = _jbernoulli_keyed(noise, 0.9, c, 0, 0)
        s1 = _jbernoulli_keyed(noise, 0.9, c, 1, 0)
        self.assertFalse(bool(jnp.array_equal(s0, s1)))

    def test_int64_counter_does_not_silently_break_fold_in(self) -> None:
        noise = jnp.zeros((64,), dtype=jnp.float32)
        out = _jbernoulli_keyed(noise, 0.9, jnp.asarray([[3]]), 0, 0)
        self.assertEqual(out.shape, noise.shape)
