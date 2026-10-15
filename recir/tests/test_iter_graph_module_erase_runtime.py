# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Runtime test for the ``erase_node`` cascade + ``lower_subgraph_nodes`` fix.

The original bug fixed by commit 18 was invisible to the structural
``test_iter_graph_module_erase.py`` because none of those tests
recompiled the sub-GMs and invoked ``forward``. Commit 16's silent
``None`` substitution corrupted the recompiled setup_gm / cleanup_gm
forward; the corruption only surfaced at runtime when downstream code
(``recir/lowering/xla_device_state.py:_copy_state_into``) tried to
``state_tensor.copy_(None)``.

This module exercises the full flow:

1. Build an ``IterGraphModule`` with an in-place op (``copy_``) whose
   output flows into the graph's ``output`` node -- the Adam-style
   shape that triggered the runtime failure.
2. Lower the body via ``segment_and_lower`` (which calls
   ``lower_subgraph_nodes`` -> ``IterGraph.erase_node`` cascade).
3. Recompile the IterGraphModule and run ``forward`` once.
4. Assert the returned value(s) are real ``torch.Tensor`` objects,
   not ``None``.

Pre-commit-18 step (3) would return a tuple containing ``None`` in
the slot previously held by the in-place op, reproducing the runtime
``TypeError`` reported by downstream consumers.
"""

import unittest
from typing import List

import torch
from recir.lowering.lowering_utils import segment_and_lower
from recir.passes.iter_graph_module import IterGraphModule


class _PassthroughSubgraphModule(torch.nn.Module):
    """A minimal ``nn.Module`` wrapper that executes the wrapped subgraph.

    ``segment_and_lower`` splices a ``call_module`` node into the
    parent graph and expects the registered submodule to consume
    the lowered ``fx.GraphModule``'s inputs and return its outputs.
    For this runtime test we simply forward to the wrapped subgraph
    -- the goal is exercising the graph-surgery + recompile path,
    not the XLA backend.
    """

    def __init__(self, sub_gm: torch.fx.GraphModule) -> None:
        super().__init__()
        self.sub_gm = sub_gm

    def forward(self, *args: torch.Tensor, tag: str = "") -> List[torch.Tensor]:
        # The wrapped sub_gm returns a tuple of tensors (see
        # ``create_subgraph_module`` in ``lowering_utils.py``).
        del tag  # tag is just for human-readable labeling
        return self.sub_gm(*args)


def _always_compatible(node: torch.fx.Node) -> tuple[bool, str]:
    return True, ""


def _build_inplace_op_iter_gm() -> IterGraphModule:
    """Build an ``IterGraphModule`` with an Adam-style in-place op.

    Graph shape:

        param, grad, denom = placeholders
        relu_0 = relu(param)
        relu_1 = relu(relu_0)
        ... (12 relu chain to clear MIN_ATEN_OPS_TO_LOWER) ...
        copy = aten.copy_(param, relu_11)  # in-place mutation
        return (copy,)

    The trailing ``copy_`` mirrors the ``addcdiv_``/``copy_`` pattern
    in functionalized Adam: the in-place op's output flows directly
    into the graph's ``output`` node. The 12-relu prologue is purely
    to clear the ``MIN_ATEN_OPS_TO_LOWER=10`` threshold so
    ``segment_and_lower`` actually lowers the body.
    """
    graph = torch.fx.Graph()
    param = graph.placeholder("param")
    grad = graph.placeholder("grad")
    denom = graph.placeholder("denom")
    last = param
    for _ in range(12):
        last = graph.call_function(torch.ops.aten.relu.default, args=(last,))
    # In-place copy_ into param. Mirrors how functionalize_optim
    # often emits a trailing copy_ to commit the optimizer update.
    copy = graph.call_function(
        torch.ops.aten.copy_.default,
        args=(param, last),
    )
    graph.output((copy,))
    del grad, denom  # placeholders are still wired even if unused
    gm = torch.fx.GraphModule(root={}, graph=graph)
    return IterGraphModule(gm)


class IterGraphModuleEraseRuntimeTest(unittest.TestCase):
    def test_forward_returns_tensor_not_none_after_inplace_op_lowering(
        self,
    ) -> None:
        """End-to-end: lowered IterGraphModule.forward returns a Tensor.

        Pre-commit-18: the cascade silently dropped the mirror copy_
        node's reference from setup_gm.output (substituting ``None``),
        the recompiled setup_gm forward returned ``(None,)`` for that
        slot, and downstream code crashed with::

            TypeError: copy_(): argument 'other' (position 1) must be
            Tensor, not NoneType

        Post-commit-18: the lower_subgraph_nodes cascade rewrites the
        mirror output to forward the copy_'s self-arg (so the mirror
        parent returns a real Tensor), and the cascade no longer
        substitutes ``None``.
        """
        iter_gm = _build_inplace_op_iter_gm()

        # Before lowering, snapshot the setup_gm output structure so
        # we can sanity-check the mirror rewrite below.
        setup_output_before = next(
            n for n in iter_gm.setup_gm.graph.nodes if n.op == "output"
        )
        self.assertEqual(
            len(setup_output_before.all_input_nodes),
            1,
            "expected setup_gm output to reference one node before lowering",
        )

        # Run segment_and_lower. This invokes lower_subgraph_nodes,
        # whose erase loop cascades through IterGraph.erase_node and
        # must not raise post-commit-18.
        segment_and_lower(
            iter_gm,  # pyre-ignore[6] -- IterGraphModule shadows fx.GraphModule
            compatibility_fn=_always_compatible,
            module_factory=_PassthroughSubgraphModule,
            subgraph_prefix="xla_subgraph",
        )

        # Mirror sanity check: after the cascade the setup_gm output
        # references the copy_'s self-arg (a placeholder) directly,
        # not None.
        setup_output = next(n for n in iter_gm.setup_gm.graph.nodes if n.op == "output")
        setup_output_args, _ = torch.utils._pytree.tree_flatten(
            (setup_output.args, setup_output.kwargs)
        )
        for arg in setup_output_args:
            self.assertIsNotNone(
                arg,
                f"setup_gm output should not contain None placeholders; "
                f"got args {setup_output_args!r}",
            )

        # Recompile all three sub-GMs and run forward. Iter 1 routes
        # through setup_gm, so the recompiled setup_gm forward is
        # what previously baked ``return (None,)`` into the codegen.
        iter_gm.recompile()
        iter_gm.finalize_setup()

        param = torch.zeros(4, dtype=torch.float32)
        grad = torch.ones(4, dtype=torch.float32)
        denom = torch.ones(4, dtype=torch.float32)
        output = iter_gm(param, grad, denom)

        # Output may be unpacked from the trailing tuple or returned
        # directly depending on cross-iter optimization state; both
        # legal shapes must contain real tensors.
        if isinstance(output, (list, tuple)):
            self.assertGreater(len(output), 0)
            for slot_idx, slot in enumerate(output):
                self.assertIsNotNone(
                    slot,
                    f"setup_gm.forward output slot {slot_idx} is None; "
                    "commit 16's cascade silently corrupted the mirror.",
                )
                self.assertIsInstance(
                    slot,
                    torch.Tensor,
                    f"setup_gm.forward output slot {slot_idx} is not a "
                    f"Tensor: {type(slot).__name__}",
                )
        else:
            self.assertIsNotNone(
                output,
                "setup_gm.forward output is None; commit 16's cascade "
                "silently corrupted the mirror.",
            )
            self.assertIsInstance(output, torch.Tensor)

    def test_segment_and_lower_does_not_raise_on_iter_graph_with_inplace_op(
        self,
    ) -> None:
        """Structural guard: the cascade itself completes without raising.

        Before commit 18's per-caller fix, the cascade would either:
          * silently corrupt (commit 16 default), or
          * post-commit-18, raise the new ``IterGraph.erase_node``
            safety error.

        After the ``lower_subgraph_nodes`` fix that mirrors the output
        rewrite into setup/cleanup before erasing, the cascade
        completes cleanly without either failure mode.
        """
        iter_gm = _build_inplace_op_iter_gm()
        # The mere fact this does not raise is the guarantee.
        segment_and_lower(
            iter_gm,  # pyre-ignore[6]
            compatibility_fn=_always_compatible,
            module_factory=_PassthroughSubgraphModule,
            subgraph_prefix="xla_subgraph",
        )
        # And the main IterGraph passes lint post-cascade.
        iter_gm.main_gm.graph.lint()
