# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Tests for ``IterGraphModule.erase_node`` cascade policy.

Background
==========

``IterGraphModule`` keeps three parallel ``fx.Graph`` instances (main,
setup, cleanup) and mirrors every mutation across all three so that
the cross-iteration optimizer can re-order nodes between iterations.

History
-------

* Original behavior: ``IterGraph.erase_node`` cascaded ``erase_node`` to
  the setup/cleanup mirror nodes WITHOUT first calling
  ``replace_all_uses_with(None)`` on them. This forced every caller to
  iterate erases in reverse topological order; any caller that erased
  a producer before its user-mirror in another graph would trip:

      RuntimeError: Tried to erase Node <producer> but it still had N
      users in the graph: {<user>: None}!

* Commit 16: Added ``replace_all_uses_with(None)`` on each mirror node
  before erasing, making the cascade self-healing regardless of
  caller order. This worked for the profiler-marker case where the
  user-mirror was itself about to be erased; it silently corrupted
  graphs where the mirror's output flowed into ``output`` (e.g.
  Adam's ``addcdiv_`` mirror) because the substituted ``None`` was
  baked into the recompiled setup_gm / cleanup_gm forward and crashed
  ``state_tensor.copy_(None)`` at runtime in ``_copy_state_into``.

* Commit 18 (this file): ``IterGraph.erase_node`` now RAISES when a
  mirror still has live users at erase time, unless the caller
  asserts the drop is safe via ``unsafe_drop_users=True``. The
  unsafe opt-in is reserved for cases (profiler markers, dead
  outputs) where every user-mirror is itself about to be erased in
  the same sweep, so ``None`` never survives into the recompiled
  forward.
"""

import unittest
from typing import Callable, cast

import torch
from recir.passes.iter_graph_module import IterGraph, IterGraphModule


def _make_iter_gm(graph: torch.fx.Graph) -> IterGraphModule:
    """Wrap an ``fx.Graph`` in an ``IterGraphModule``.

    The dummy ``{}`` root is fine because every node in our tests is
    either a placeholder or a ``call_function`` on a free-standing
    op (no attribute lookups).
    """
    gm = torch.fx.GraphModule(root={}, graph=graph)
    return IterGraphModule(gm)


def _find_main_node(
    iter_gm: IterGraphModule, predicate: Callable[[torch.fx.Node], bool]
) -> torch.fx.Node:
    """Return the unique main-graph node satisfying ``predicate``."""
    matches = [n for n in iter_gm.main_gm.graph.nodes if predicate(n)]
    assert len(matches) == 1, (
        f"expected exactly one matching node, got {len(matches)}: "
        f"{[n.name for n in matches]}"
    )
    return matches[0]


def _assert_no_call_function(iter_gm: IterGraphModule) -> None:
    """Assert all three sub-graphs have zero ``call_function`` nodes."""
    for sub_gm_name in ("main_gm", "setup_gm", "cleanup_gm"):
        sub_gm = getattr(iter_gm, sub_gm_name)
        call_nodes = [n for n in sub_gm.graph.nodes if n.op == "call_function"]
        if call_nodes:
            raise AssertionError(
                f"call_function nodes still present in {sub_gm_name}: "
                f"{[n.name for n in call_nodes]}"
            )


class IterGraphModuleEraseTest(unittest.TestCase):
    def test_erase_raises_when_mirror_has_live_users(self) -> None:
        """Default cascade refuses to silently drop mirror users.

        Generic shape of the bug the earlier always-cascade papered over: producer
        ``A`` is consumed by user ``B``. In the main graph we first
        detach ``A`` from ``B`` (so main-graph ``A.users == {}``),
        then erase ``A``. The setup/cleanup mirror of ``B`` still
        references the mirror of ``A``. With the current fail-fast policy, dropping
        those mirror users silently would risk substituting ``None``
        into a downstream value (e.g. the recompiled ``output``).
        The cascade raises instead.
        """
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        producer = graph.call_function(torch.ops.aten.relu.default, args=(x,))
        user = graph.call_function(torch.ops.aten.relu.default, args=(producer,))
        graph.output(user)
        iter_gm = _make_iter_gm(graph)

        main_producer = _find_main_node(
            iter_gm,
            lambda n: n.op == "call_function"
            and n.target == torch.ops.aten.relu.default
            and any(
                a.op == "placeholder" for a in n.args if isinstance(a, torch.fx.Node)
            ),
        )
        main_user = _find_main_node(
            iter_gm,
            lambda n: n.op == "call_function"
            and n.target == torch.ops.aten.relu.default
            and main_producer in n.args,
        )

        main_user.args = (next(iter(iter_gm.main_gm.graph.nodes)),)
        self.assertEqual(len(main_producer.users), 0)
        self.assertEqual(
            len(
                cast(IterGraph, iter_gm.main_gm.graph)
                ._setup_mapping[main_producer]
                .users
            ),
            1,
        )

        with self.assertRaises(RuntimeError) as ctx:
            cast(IterGraph, iter_gm.main_gm.graph).erase_node(main_producer)
        self.assertIn("setup-graph mirror", str(ctx.exception))
        self.assertIn("output/user rewrite cascade", str(ctx.exception))

    def test_unsafe_drop_users_flag_allows_cascade_drop(self) -> None:
        """``unsafe_drop_users=True`` re-enables the earlier always-cascade behavior.

        Intended for sweeps where every user-mirror is itself about to
        be erased (e.g. profiler ``enter``/``exit`` pairs). The ``None``
        substitution then never reaches the recompiled forward.
        """
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        producer = graph.call_function(torch.ops.aten.relu.default, args=(x,))
        graph.call_function(torch.ops.aten.relu.default, args=(producer,))
        graph.output(x)  # output ignores the chain
        iter_gm = _make_iter_gm(graph)

        main_placeholder = next(iter(iter_gm.main_gm.graph.nodes))
        main_producer = _find_main_node(
            iter_gm,
            lambda n: n.op == "call_function" and main_placeholder in n.args,
        )
        main_user = _find_main_node(
            iter_gm,
            lambda n: n.op == "call_function" and main_producer in n.args,
        )

        # Detach the main-graph chain so main use-counts are zero.
        main_user.args = (main_placeholder,)
        self.assertEqual(len(main_producer.users), 0)

        # FORWARD topo erase with the explicit opt-in succeeds; the
        # mirror-user keeps the now-None reference but is itself
        # erased on the next call.
        cast(IterGraph, iter_gm.main_gm.graph).erase_node(
            main_producer, unsafe_drop_users=True
        )
        self.assertEqual(len(main_user.users), 0)
        cast(IterGraph, iter_gm.main_gm.graph).erase_node(
            main_user, unsafe_drop_users=True
        )

        _assert_no_call_function(iter_gm)

    def test_addcdiv_pattern_now_raises_without_cascade(self) -> None:
        """Reproduces the addcdiv_ mirror -> output corruption.

        Mirrors the Adam-style functionalized-optimizer pattern: an
        in-place op (``addcdiv_``) is wired into the output tuple.
        After ``lower_subgraph_nodes`` does
        ``output.replace_all_uses_with(new_output)`` on the MAIN graph
        only, the setup/cleanup mirror ``addcdiv_`` is still wired
        into the mirror output. Commit 16 silently substituted
        ``None`` there; the current fail-fast policy raises so the caller is forced to
        perform the cascade in the mirror parents (see
        ``lower_subgraph_nodes``).
        """
        graph = torch.fx.Graph()
        param = graph.placeholder("param")
        grad = graph.placeholder("grad")
        denom = graph.placeholder("denom")
        addcdiv = graph.call_function(
            torch.ops.aten.addcdiv_.default,
            args=(param, grad, denom),
        )
        graph.output((addcdiv,))
        iter_gm = _make_iter_gm(graph)

        main_addcdiv = _find_main_node(
            iter_gm,
            lambda n: n.op == "call_function"
            and n.target == torch.ops.aten.addcdiv_.default,
        )

        # Emulate the lower_subgraph_nodes step that rewrites the
        # MAIN output's reference only (leaving mirrors hostile).
        main_output = next(n for n in iter_gm.main_gm.graph.nodes if n.op == "output")
        main_output.args = ((),)
        self.assertEqual(len(main_addcdiv.users), 0)
        self.assertEqual(
            len(
                cast(IterGraph, iter_gm.main_gm.graph)
                ._setup_mapping[main_addcdiv]
                .users
            ),
            1,
        )

        with self.assertRaises(RuntimeError) as ctx:
            cast(IterGraph, iter_gm.main_gm.graph).erase_node(main_addcdiv)
        self.assertIn("addcdiv", str(ctx.exception))
        self.assertIn("output/user rewrite cascade", str(ctx.exception))

    def test_replace_with_None_consumer_correctness_contract(self) -> None:
        """Erase-cascade safety invariant: never silently mis-route a Mirror.

        This test pins the user-visible contract that the
        ``replace_all_uses_with(None)`` on Mirror nodes during
        ``erase_node`` must preserve:

          * If a Mirror node still has live users when its main-graph
            producer is erased, the cascade MUST either fail-fast
            (``RuntimeError``) or short-circuit (caller opts in to
            ``unsafe_drop_users=True``, accepting responsibility).
          * It MUST NOT silently leave a ``None`` reference in a
            recompiled Mirror graph that downstream consumers (the
            recompiled forward, ``_copy_state_into``, etc.) would
            interpret as a real tensor and crash on.

        The contract is enforced by two complementary paths in
        ``IterGraph.erase_node``:

          * Default cascade -- raise the ``RuntimeError`` documented in
            ``IterGraph._erase_mirror_node`` (assert; tested below).
          * Opt-in ``unsafe_drop_users=True`` -- caller asserts every
            user-mirror is itself about to be erased in the same
            sweep; the ``None`` substitution never reaches a
            recompiled forward.

        This test exercises both legs against the same producer/user
        Mirror shape so a future refactor of either branch cannot
        regress correctness without flipping a clear, named assertion.
        """
        # ------------------------------------------------------------
        # Leg 1: default cascade fails fast when Mirror has live users.
        # ------------------------------------------------------------
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        producer = graph.call_function(torch.ops.aten.relu.default, args=(x,))
        user = graph.call_function(torch.ops.aten.relu.default, args=(producer,))
        graph.output(user)
        iter_gm = _make_iter_gm(graph)

        main_placeholder = next(iter(iter_gm.main_gm.graph.nodes))
        main_producer = _find_main_node(
            iter_gm,
            lambda n: n.op == "call_function" and main_placeholder in n.args,
        )
        main_user = _find_main_node(
            iter_gm,
            lambda n: n.op == "call_function" and main_producer in n.args,
        )

        # Detach producer from user in MAIN only; the setup mirror of
        # user still consumes the setup mirror of producer.
        main_user.args = (main_placeholder,)
        setup_producer = cast(IterGraph, iter_gm.main_gm.graph)._setup_mapping[
            main_producer
        ]
        self.assertEqual(len(main_producer.users), 0)
        self.assertGreater(
            len(setup_producer.users),
            0,
            "precondition: setup-mirror producer must have live users so "
            "the cascade has an opportunity to silently corrupt",
        )

        with self.assertRaises(
            RuntimeError,
            msg="default cascade must fail-fast rather than silently substitute "
            "None into the recompiled setup_gm forward",
        ) as ctx:
            cast(IterGraph, iter_gm.main_gm.graph).erase_node(main_producer)
        self.assertIn("setup-graph mirror", str(ctx.exception))

        # ------------------------------------------------------------
        # Leg 2: opt-in unsafe_drop_users + paired-erase short-circuits.
        # Every user-mirror is erased in the same sweep so the
        # transient ``None`` never reaches a recompiled forward.
        # ------------------------------------------------------------
        graph2 = torch.fx.Graph()
        x2 = graph2.placeholder("x")
        producer2 = graph2.call_function(torch.ops.aten.relu.default, args=(x2,))
        graph2.call_function(torch.ops.aten.relu.default, args=(producer2,))
        graph2.output(x2)  # output ignores the chain so both are dead in MAIN
        iter_gm2 = _make_iter_gm(graph2)

        main_ph2 = next(iter(iter_gm2.main_gm.graph.nodes))
        main_producer2 = _find_main_node(
            iter_gm2,
            lambda n: n.op == "call_function" and main_ph2 in n.args,
        )
        main_user2 = _find_main_node(
            iter_gm2,
            lambda n: n.op == "call_function" and main_producer2 in n.args,
        )
        # Detach the MAIN chain so both nodes have zero MAIN users.
        main_user2.args = (main_ph2,)

        # Caller asserts safety: both producer and user mirrors will
        # be erased in the same sweep.
        cast(IterGraph, iter_gm2.main_gm.graph).erase_node(
            main_producer2, unsafe_drop_users=True
        )
        cast(IterGraph, iter_gm2.main_gm.graph).erase_node(
            main_user2, unsafe_drop_users=True
        )
        # Post-sweep invariant: no stale ``None`` references remain in
        # any sub-graph (the corrupting state the earlier always-cascade papered over).
        _assert_no_call_function(iter_gm2)

    def test_unsafe_drop_users_paired_erase_forward_returns_expected_output(
        self,
    ) -> None:
        """Positive correctness: after paired ``unsafe_drop_users`` erase the
        recompiled forward returns the expected tensor.

        Companion to the structural assertions in
        ``test_unsafe_drop_users_flag_allows_cascade_drop``: the
        structural test only checks that the graph contains no
        ``call_function`` nodes; this test additionally recompiles
        the three sub-GMs and *runs* the forward of an
        ``IterGraphModule`` whose body had its op-chain erased via
        the ``replace_all_uses_with(None)`` codepath. The graph's
        output ignores the erased chain (the profiler-marker shape),
        so the recompiled forward must return the placeholder
        verbatim -- a positive proof that the cascade did not
        smuggle ``None`` into a slot consumed by the recompiled
        forward.
        """
        # Profiler-marker-style shape: build a chain whose output is
        # the raw input ``x``; the relu->relu chain is "dead" but
        # still present, mimicking enter/exit markers.
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        producer = graph.call_function(torch.ops.aten.relu.default, args=(x,))
        graph.call_function(torch.ops.aten.relu.default, args=(producer,))
        graph.output(x)
        iter_gm = _make_iter_gm(graph)

        main_placeholder = next(iter(iter_gm.main_gm.graph.nodes))
        main_producer = _find_main_node(
            iter_gm,
            lambda n: n.op == "call_function" and main_placeholder in n.args,
        )
        main_user = _find_main_node(
            iter_gm,
            lambda n: n.op == "call_function" and main_producer in n.args,
        )

        # Detach the MAIN chain so both nodes have zero MAIN users.
        main_user.args = (main_placeholder,)

        # Sweep erase: producer first triggers replace_all_uses_with(None)
        # on the setup/cleanup user-mirror; the user is erased on the
        # follow-up call, so the transient ``None`` never reaches a
        # recompiled forward.
        cast(IterGraph, iter_gm.main_gm.graph).erase_node(
            main_producer, unsafe_drop_users=True
        )
        cast(IterGraph, iter_gm.main_gm.graph).erase_node(
            main_user, unsafe_drop_users=True
        )

        # Positive contract: recompile + forward must succeed and
        # return the input tensor unchanged.
        iter_gm.recompile()
        iter_gm.finalize_setup()

        x_in = torch.arange(4, dtype=torch.float32)
        # Iter 1 dispatches through setup_gm (the mirror whose user
        # had its arg silently substituted by ``None``); the chain is
        # dead, so the recompiled setup forward must return ``x_in``.
        output_setup = iter_gm(x_in)
        torch.testing.assert_close(output_setup, x_in)

        # Iter 2 dispatches through main_gm (the graph we directly
        # mutated). Same expected output.
        output_main = iter_gm(x_in)
        torch.testing.assert_close(output_main, x_in)

        # Iter 3 marked last -> cleanup_gm (the other mirror, also
        # touched by replace_all_uses_with(None)). Same expectation.
        output_cleanup = iter_gm(x_in, last_iter=True)
        torch.testing.assert_close(output_cleanup, x_in)
