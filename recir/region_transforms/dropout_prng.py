# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""DropoutPRNGTransform -- give dropout a PRNG key that lives in the graph."""

from __future__ import annotations

import logging

import torch
import torch.fx as fx
from recir.utils.graph import OP, rebuild_graph

# Defines torch.ops.recir.bernoulli_keyed, which this transform inserts.
import recir.ops.random_op  # noqa: F401  # isort: skip
from recir.region_transforms.base import RegionTransform

logger: logging.Logger = logging.getLogger(__name__)


def _bernoulli_targets() -> tuple[object, ...]:
    return (
        torch.ops.aten.bernoulli_.float,
        torch.ops.aten.bernoulli.p,
    )


def _unrewritable_draw_targets() -> tuple[object, ...]:
    return (
        torch.ops.aten.native_dropout.default,
        torch.ops.aten._fused_dropout.default,
        torch.ops.aten.bernoulli_.Tensor,
        torch.ops.aten.bernoulli.Tensor,
        torch.ops.aten.bernoulli.default,
    )


class DropoutPRNGTransform(RegionTransform):
    """Rewrite dropout's Bernoulli draws to key off an in-graph counter.

    Args:
        counter_fqn: FQN of a 0-d integer buffer registered on the model and
            advanced in-graph once per step (a torch op, so ``make_fx`` records
        it). recir does not own this name; the caller supplies it. A learning
            rate warmup counter is a suitable choice when it already advances
            once per step.
        base_seed: fixed rather than host-random, so a rerun reproduces the same
            masks and step N depends only on (seed, iteration, call site).
    """

    def __init__(self, counter_fqn: str, base_seed: int = 0) -> None:
        self._counter_fqn = counter_fqn
        self._base_seed = base_seed

    def finalize_compiled_graph(self, gm: fx.GraphModule) -> None:
        # Checked BEFORE the no-dropout early return: a graph whose only draw is
        # an unrewritable one must fail, not look like a graph with no dropout.
        self._reject_unrewritable(gm)

        targets = _bernoulli_targets()
        draws = [
            n
            for n in gm.graph.nodes
            if n.op == OP.CALL_FUNCTION and n.target in targets
        ]
        if not draws:
            # Dropout is off (or this graph has none). Nothing to key.
            return

        counter = self._find_counter(gm)
        for site, node in enumerate(draws):
            self._rewrite(gm, node, counter, site)
        rebuild_graph(gm)
        logger.info(
            "DropoutPRNGTransform: keyed %d dropout draw(s) off %s (base_seed=%d)",
            len(draws),
            self._counter_fqn,
            self._base_seed,
        )

    def _reject_unrewritable(self, gm: fx.GraphModule) -> None:
        unsupported = _unrewritable_draw_targets()
        found = sorted(
            {
                str(n.target)
                for n in gm.graph.nodes
                if n.op == OP.CALL_FUNCTION and n.target in unsupported
            }
        )
        if found:
            raise RuntimeError(
                f"DropoutPRNGTransform: the graph contains random draws this "
                f"transform cannot rewrite: {found}. They would fall back to "
                f"torchax's Environment key -- a constant baked at construction "
                f"time, so every step would draw the identical mask and nothing "
                f"would raise. Extend recir::bernoulli_keyed to cover them, or "
                f"keep them out of the compiled graph."
            )

    def _find_counter(self, gm: fx.GraphModule) -> fx.Node:
        for node in gm.graph.nodes:
            if node.op == OP.PLACEHOLDER and node.meta.get("fqn") == self._counter_fqn:
                return node
        # Hard failure, deliberately. Falling back to torchax's environment key
        # would leave dropout drawing the SAME mask every step from a constant
        # baked at Environment construction -- which runs, reaches step N, and
        # raises nothing. Three previous attempts passed every check they were
        # given while doing exactly that.
        fqns = sorted(
            str(n.meta["fqn"])
            for n in gm.graph.nodes
            if n.op == OP.PLACEHOLDER and "fqn" in n.meta
        )
        raise RuntimeError(
            f"DropoutPRNGTransform: dropout is active but no placeholder is "
            f"tagged fqn={self._counter_fqn!r}. Without it the PRNG key cannot "
            f"be graph state and every step would draw an identical mask. "
            f"Register the counter buffer on the model, or disable dropout. "
            f"Tagged placeholders: {fqns}"
        )

    def _rewrite(
        self,
        gm: fx.GraphModule,
        node: fx.Node,
        counter: fx.Node,
        site: int,
    ) -> None:
        noise = node.args[0]
        # ``bernoulli.p`` defaults p to 0.5 and may leave it out of args.
        keep_p = node.args[1] if len(node.args) > 1 else node.kwargs.get("p", 0.5)
        with gm.graph.inserting_after(node):
            # ``.default``, not the OpOverloadPacket: node targets elsewhere in
            # recir are OpOverloads (e.g. fsdp_collectives.py's
            # ``reshard_fsdp_seq.default``), and _is_xla_compatible only runs its
            # meta-kernel check for OpOverload targets -- a packet would skip the
            # check that guarantees this node does not break the graph.
            keyed = gm.graph.call_function(
                torch.ops.recir.bernoulli_keyed.default,
                args=(noise, keep_p, counter, site, self._base_seed),
            )
        val = node.meta.get("val")
        if val is not None:
            with val.fake_mode:
                keyed.meta["val"] = torch.empty_like(val)
        for key, value in node.meta.items():
            # ``val`` is recomputed above. ``xla_graph_break`` must NOT be
            # inherited: it would make segment_and_lower cut here
            # (recir/lowering/xla_lower.py).
            if key not in ("val", "xla_graph_break"):
                keyed.meta.setdefault(key, value)
        # ``bernoulli_`` returns the tensor it mutated, so its users consume its
        # OUTPUT node and rewiring them is sound. ``noise`` (the ``empty_like``)
        # stays live as the keyed op's first argument.
        node.replace_all_uses_with(keyed)
        gm.graph.erase_node(node)
