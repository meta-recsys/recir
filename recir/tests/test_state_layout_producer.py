# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Unit tests for the ParallelMode-produced FSDP state-layout (R4).

These exercise ``DataParallel.apply_state_layout`` in isolation -- no full
trace: a tiny ``fx.Graph`` with placeholders carrying ``node.meta["fqn"]`` and a
hand-built ``_StateLayoutPlan`` stand in for what ``shard_params`` would stash
during a real trace. ``apply_state_layout`` must reproduce the original
FSDP-on-XLA behavior via the typed ``recir_state_layout`` spec: tag the
fqn-matched placeholders with an ``FSDPWeightShardSpec`` and substitute the /P
per-process store into ``params_and_buffers`` (rebuilt into ``flat_state``
downstream), then clear the trace-scoped plan.
"""

import unittest
from typing import Dict, List

import torch
import torch.fx as fx
from recir.distributed.parallel_mode import _StateLayoutPlan, DataParallel, ParallelMode
from recir.distributed.sharding_spec import (
    FSDPWeightShardSpec,
    get_state_layout,
    STATE_LAYOUT_META_KEY,
)


def _graph_with_placeholders(fqns: List[str]) -> fx.GraphModule:
    """Build a tiny GraphModule whose placeholders carry ``node.meta["fqn"]``."""
    graph = fx.Graph()
    for fqn in fqns:
        node = graph.placeholder(fqn.replace(".", "_"))
        node.meta["fqn"] = fqn
    graph.output(())
    return fx.GraphModule(torch.nn.Module(), graph)


def _graph_with_state_placeholders(
    param_fqns: List[str], num_opt_states: int, num_inputs: int
) -> fx.GraphModule:
    """Mirror ``_local_compile``'s placeholder order: params/buffers (fqn + val),
    then optimizer-state leaves (val only, no fqn), then forward inputs."""
    graph = fx.Graph()
    for fqn in param_fqns:
        node = graph.placeholder(fqn.replace(".", "_"))
        node.meta["fqn"] = fqn
        node.meta["val"] = torch.zeros(1)
    for i in range(num_opt_states):
        graph.placeholder(f"opt_state_{i}").meta["val"] = torch.zeros(1)
    for i in range(num_inputs):
        graph.placeholder(f"input_{i}").meta["val"] = torch.zeros(1)
    graph.output(())
    return fx.GraphModule(torch.nn.Module(), graph)


class StateLayoutProducerTest(unittest.TestCase):
    N: int = 8  # device world size (P * D)
    P: int = 2  # process world size

    def _make_plan(self) -> _StateLayoutPlan:
        # /P per-process store: distinct tensor objects so substitution can be
        # verified by identity. ``absent.weight`` is in the store + sharded_keys
        # but not in ``params_and_buffers`` -> must skip gracefully.
        process_store: Dict[str, torch.Tensor] = {
            "net1.weight": torch.zeros(4, 3),
            "net2.weight": torch.zeros(2, 5),
            "absent.weight": torch.zeros(1),
        }
        return _StateLayoutPlan(
            process_store=process_store,
            opt_process_store={},
            opt_partitioned={},
            sharded_keys={"net1.weight", "net2.weight", "absent.weight"},
            device_world_size=self.N,
            process_world_size=self.P,
        )

    def test_apply_state_layout_tags_and_substitutes(self) -> None:
        plan = self._make_plan()
        # net1.weight / net2.weight are sharded; net1.bias is not.
        gm = _graph_with_placeholders(["net1.weight", "net2.weight", "net1.bias"])
        mode = DataParallel("fully_shard")
        mode._pending_state_layout = plan

        # /N trace shards currently in params_and_buffers (pre-substitution).
        bias = torch.ones(4)
        params_and_buffers: Dict[str, torch.Tensor] = {
            "net1.weight": torch.ones(1, 3),
            "net2.weight": torch.ones(1, 5),
            "net1.bias": bias,
        }

        mode.apply_state_layout(gm, params_and_buffers, {})

        placeholders = {
            n.meta.get("fqn"): n for n in gm.graph.nodes if n.op == "placeholder"
        }
        expected = FSDPWeightShardSpec(
            shard_dim=0, device_world_size=self.N, process_world_size=self.P
        )

        # (a) exactly the sharded fqns got the typed spec.
        self.assertEqual(get_state_layout(placeholders["net1.weight"]), expected)
        self.assertEqual(get_state_layout(placeholders["net2.weight"]), expected)
        # (e) the non-sharded placeholder is left untouched (no meta key at all).
        self.assertIsNone(get_state_layout(placeholders["net1.bias"]))
        self.assertNotIn(STATE_LAYOUT_META_KEY, placeholders["net1.bias"].meta)

        # (b) /P substitution for the sharded-and-present keys only.
        self.assertIs(
            params_and_buffers["net1.weight"], plan.process_store["net1.weight"]
        )
        self.assertIs(
            params_and_buffers["net2.weight"], plan.process_store["net2.weight"]
        )
        # non-sharded key keeps its original tensor.
        self.assertIs(params_and_buffers["net1.bias"], bias)
        # key in the store but absent from params_and_buffers is not injected.
        self.assertNotIn("absent.weight", params_and_buffers)

        # (c) the trace-scoped plan is cleared after use.
        self.assertIsNone(mode._pending_state_layout)

    def test_apply_state_layout_tags_and_substitutes_optimizer_states(self) -> None:
        # One sharded param whose optimizer state has two leaves (exp_avg,
        # exp_avg_sq), plus a forward input that must stay untagged. Opt-state
        # placeholders carry no ``fqn`` -> they must be tagged positionally.
        opt_store: Dict[str, Dict[str, torch.Tensor]] = {
            "net1.weight": {
                "exp_avg": torch.zeros(4, 3),
                "exp_avg_sq": torch.zeros(4, 3),
            }
        }
        plan = _StateLayoutPlan(
            process_store={"net1.weight": torch.zeros(4, 3)},
            opt_process_store=opt_store,
            opt_partitioned={"net1.weight": {"exp_avg", "exp_avg_sq"}},
            sharded_keys={"net1.weight"},
            device_world_size=self.N,
            process_world_size=self.P,
        )
        gm = _graph_with_state_placeholders(
            ["net1.weight"], num_opt_states=2, num_inputs=1
        )
        mode = DataParallel("fully_shard")
        mode._pending_state_layout = plan

        params_and_buffers: Dict[str, torch.Tensor] = {"net1.weight": torch.ones(1, 3)}
        named_states: Dict[str, Dict[str, torch.Tensor]] = {
            "net1.weight": {"exp_avg": torch.ones(1, 3), "exp_avg_sq": torch.ones(1, 3)}
        }

        mode.apply_state_layout(gm, params_and_buffers, named_states)

        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        expected = FSDPWeightShardSpec(
            shard_dim=0, device_world_size=self.N, process_world_size=self.P
        )
        # placeholders = [param(net1.weight), exp_avg, exp_avg_sq, forward input].
        self.assertEqual(get_state_layout(placeholders[0]), expected)  # param via fqn
        self.assertEqual(get_state_layout(placeholders[1]), expected)  # exp_avg
        self.assertEqual(get_state_layout(placeholders[2]), expected)  # exp_avg_sq
        # the trailing forward input is past the state prefix -> left untouched.
        self.assertIsNone(get_state_layout(placeholders[3]))
        self.assertNotIn(STATE_LAYOUT_META_KEY, placeholders[3].meta)
        # named_states substituted in place with the /P optimizer-state store.
        self.assertIs(named_states["net1.weight"], opt_store["net1.weight"])
        self.assertIsNone(mode._pending_state_layout)

    def test_apply_state_layout_skips_zero_dim_optimizer_state(self) -> None:
        # Adam's ``step`` is a 0-dim tensor -- no dim-0 to Shard(0). It must be
        # left untagged (replicated) rather than crash the /N shard, while the
        # ndim>=1 ``exp_avg`` leaf is still tagged.
        plan = _StateLayoutPlan(
            process_store={"net1.weight": torch.zeros(4, 3)},
            opt_process_store={
                "net1.weight": {"exp_avg": torch.zeros(4, 3), "step": torch.zeros(())}
            },
            opt_partitioned={"net1.weight": {"exp_avg"}},
            sharded_keys={"net1.weight"},
            device_world_size=self.N,
            process_world_size=self.P,
        )
        graph = fx.Graph()
        w = graph.placeholder("net1_weight")
        w.meta["fqn"] = "net1.weight"
        w.meta["val"] = torch.zeros(1, 3)
        graph.placeholder("exp_avg").meta["val"] = torch.zeros(1, 3)  # ndim>=1 -> tag
        graph.placeholder("step").meta["val"] = torch.zeros(())  # 0-dim -> skip
        graph.output(())
        gm = fx.GraphModule(torch.nn.Module(), graph)
        mode = DataParallel("fully_shard")
        mode._pending_state_layout = plan

        mode.apply_state_layout(
            gm,
            {"net1.weight": torch.ones(1, 3)},
            {"net1.weight": {"exp_avg": torch.ones(1, 3), "step": torch.zeros(())}},
        )

        phs = [n for n in gm.graph.nodes if n.op == "placeholder"]
        expected = FSDPWeightShardSpec(
            shard_dim=0, device_world_size=self.N, process_world_size=self.P
        )
        self.assertEqual(get_state_layout(phs[0]), expected)  # weight (fqn)
        self.assertEqual(get_state_layout(phs[1]), expected)  # exp_avg (ndim>=1)
        self.assertIsNone(get_state_layout(phs[2]))  # step (0-dim) skipped

    def test_apply_state_layout_skips_non_divisible_optimizer_state(self) -> None:
        # An ndim>=1 opt-state leaf whose dim 0 is NOT divisible by P is kept
        # full at capture (mirroring ``capture_fsdp_process_store``), so apply
        # must leave its placeholder untagged -- its key is absent from
        # ``opt_partitioned`` -- even though ndim>=1. Tagging Shard(0) here would
        # over-claim vs. the un-partitioned full runtime value.
        plan = _StateLayoutPlan(
            process_store={"net1.weight": torch.zeros(4, 3)},
            opt_process_store={
                "net1.weight": {
                    "exp_avg": torch.zeros(4, 3),  # 4 % P == 0 -> partitioned
                    "odd": torch.zeros(3, 5),  # 3 % P != 0 -> kept full
                }
            },
            opt_partitioned={"net1.weight": {"exp_avg"}},
            sharded_keys={"net1.weight"},
            device_world_size=self.N,
            process_world_size=self.P,
        )
        graph = fx.Graph()
        w = graph.placeholder("net1_weight")
        w.meta["fqn"] = "net1.weight"
        w.meta["val"] = torch.zeros(1, 3)
        graph.placeholder("exp_avg").meta["val"] = torch.zeros(
            1, 3
        )  # partitioned -> tag
        graph.placeholder("odd").meta["val"] = torch.zeros(3, 5)  # full -> skip
        graph.output(())
        gm = fx.GraphModule(torch.nn.Module(), graph)
        mode = DataParallel("fully_shard")
        mode._pending_state_layout = plan

        mode.apply_state_layout(
            gm,
            {"net1.weight": torch.ones(1, 3)},
            {"net1.weight": {"exp_avg": torch.ones(1, 3), "odd": torch.ones(3, 5)}},
        )

        phs = [n for n in gm.graph.nodes if n.op == "placeholder"]
        expected = FSDPWeightShardSpec(
            shard_dim=0, device_world_size=self.N, process_world_size=self.P
        )
        self.assertEqual(get_state_layout(phs[0]), expected)  # weight (fqn)
        self.assertEqual(get_state_layout(phs[1]), expected)  # exp_avg (partitioned)
        self.assertIsNone(get_state_layout(phs[2]))  # odd (non-divisible) skipped
        self.assertNotIn(STATE_LAYOUT_META_KEY, phs[2].meta)

    def test_apply_state_layout_noop_when_no_plan(self) -> None:
        gm = _graph_with_placeholders(["net1.weight"])
        mode = DataParallel("fully_shard")
        self.assertIsNone(mode._pending_state_layout)
        original = torch.ones(2, 2)
        params_and_buffers: Dict[str, torch.Tensor] = {"net1.weight": original}

        mode.apply_state_layout(gm, params_and_buffers, {})

        placeholder = next(n for n in gm.graph.nodes if n.op == "placeholder")
        self.assertIsNone(get_state_layout(placeholder))
        self.assertIs(params_and_buffers["net1.weight"], original)
        self.assertIsNone(mode._pending_state_layout)

    def test_base_apply_state_layout_raises_not_implemented(self) -> None:
        # (d) the base ParallelMode.apply_state_layout is abstract: modes that
        # produce no state layout must override it with an explicit no-op, so
        # invoking the base directly (unbound) raises NotImplementedError rather
        # than silently doing nothing.
        self.assertIsNot(
            DataParallel.apply_state_layout, ParallelMode.apply_state_layout
        )
        gm = _graph_with_placeholders(["net1.weight"])
        mode = DataParallel("fully_shard")
        with self.assertRaises(NotImplementedError):
            ParallelMode.apply_state_layout(
                mode, gm, {"net1.weight": torch.ones(1, 3)}, {}
            )


if __name__ == "__main__":
    unittest.main()
