#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""The graph-break input replay reproduces what the graph hands the callback.

This is the check the first attempt at pipelining the preprocessor did not
have. Its producer built the callback's inputs by hand, that second path drifted
from the graph's, and the run trained ~0.02 NE worse for nine hours without
failing. Every test here asks the same question that run should have: are the
replayed inputs the tensors the graph actually passed?
"""

from typing import List, Tuple

import torch
from later.unittest import TestCase
from recir.eager_dispatch import install_graph_break_dispatch, mark_input
from recir.examples.rankfm_mvai.graph_input_replay import build_dispatch_input_replay
from recir.ops.marker import mark_boundary
from torch.fx.experimental.proxy_tensor import make_fx

_PREFIX = "test.replay::0"


def _step(state: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A step whose graph break takes one raw arg and two derived ones.

    ``x`` reaches the callback untouched, the way most of the real
    preprocessor's inputs do. ``y * 2`` and ``x + y`` stand in for the handful
    the PaddingRegionTransform derives inside the graph -- the ones a
    hand-written producer has to reimplement, and gets wrong.
    """
    a = mark_input(x, _PREFIX, 0)
    b = mark_input(y * 2, _PREFIX, 1)
    c = mark_input(x + y, _PREFIX, 2)
    out = torch.zeros_like(x)
    out = mark_boundary(out, f"{_PREFIX}::output::0")
    # Keep the marked inputs live so DCE cannot prune the derivations before
    # install_graph_break_dispatch rewires them.
    return out + a.sum() + b.sum() + c.sum() + state.sum()


class GraphInputReplayTest(TestCase):
    def _build(
        self, num_state_tensors: int = 1
    ) -> Tuple[torch.fx.GraphModule, List[List[torch.Tensor]], object]:
        state = torch.ones(3)
        x = torch.arange(4, dtype=torch.float32)
        y = torch.arange(4, dtype=torch.float32) + 10.0
        gm = make_fx(_step, tracing_mode="fake")(state, x, y)

        seen: List[List[torch.Tensor]] = []

        def fwd_fn(*inputs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
            seen.append([t.clone() for t in inputs])
            return (torch.zeros(4),)

        dispatch_id = install_graph_break_dispatch(
            gm,
            _PREFIX,
            fwd_fn,
            num_state_tensors=num_state_tensors,
            register_backward=False,
        )
        assert dispatch_id is not None
        replay = build_dispatch_input_replay(gm, dispatch_id, num_state_tensors)
        return gm, seen, replay

    def test_replay_reproduces_the_graphs_inputs(self) -> None:
        gm, seen, replay = self._build()
        state = torch.ones(3)
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        y = torch.tensor([5.0, 6.0, 7.0, 8.0])

        gm(state, x, y)

        self.assertEqual(len(seen), 1)
        # pyre-ignore[16]: _build returns the concrete DispatchInputReplay.
        replay.assert_matches(seen[0], [x, y])

    def test_replay_tracks_a_different_batch(self) -> None:
        """The point of the replay: a batch the graph has not run yet."""
        gm, seen, replay = self._build()
        state = torch.ones(3)
        first = (torch.tensor([1.0, 2.0, 3.0, 4.0]), torch.tensor([5.0, 6.0, 7.0, 8.0]))
        second = (
            torch.tensor([9.0, 8.0, 7.0, 6.0]),
            torch.tensor([-1.0, -2.0, -3.0, -4.0]),
        )

        # Replay the SECOND batch before the graph has seen it, then run it.
        # pyre-ignore[16]
        ahead = replay(list(second))
        gm(state, *first)
        gm(state, *second)

        for early, actual in zip(ahead, seen[1]):
            self.assertTrue(torch.equal(early, actual))

    def test_mismatched_args_fail_loudly(self) -> None:
        gm, seen, replay = self._build()
        state = torch.ones(3)
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        y = torch.tensor([5.0, 6.0, 7.0, 8.0])

        gm(state, x, y)

        with self.assertRaisesRegex(RuntimeError, "differs from the graph"):
            # pyre-ignore[16]
            replay.assert_matches(seen[0], [x, y + 1.0])

    def test_refuses_to_read_model_state(self) -> None:
        """A replay that reaches into state would silently serve stale weights."""

        def _state_step(state: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            a = mark_input(state, _PREFIX, 0)
            out = torch.zeros_like(x)
            out = mark_boundary(out, f"{_PREFIX}::output::0")
            return out + a.sum()

        gm = make_fx(_state_step, tracing_mode="fake")(torch.ones(3), torch.zeros(4))
        dispatch_id = install_graph_break_dispatch(
            gm,
            _PREFIX,
            lambda *inputs: (torch.zeros(4),),
            num_state_tensors=1,
            register_backward=False,
        )
        assert dispatch_id is not None
        with self.assertRaisesRegex(RuntimeError, "model-state placeholder"):
            build_dispatch_input_replay(gm, dispatch_id, 1)
