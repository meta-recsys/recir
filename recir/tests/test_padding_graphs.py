# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Capacity selection and continuous training across precompiled XLA graphs."""

import os
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import numpy as np
import torch
from recir.api import make_compiled_runner
from recir.backends.types import BackendType
from recir.fb.examples.fm_v3_common.capacity import CapacityBucket, CapacitySelector
from recir.fb.examples.fm_v3_common.padding_graphs import PaddingGraphs
from recir.lowering.xla_device_state import XLADeviceStateManager
from recir.lowering.xla_lower import _XLAModule
from recir.training.graph_state import SharedGraphState
from torch.fx.experimental.proxy_tensor import make_fx
from torchrec import KeyedJaggedTensor


class TestCapacitySelection(TestCase):
    def setUp(self) -> None:
        self.selector = CapacitySelector(
            [
                CapacityBucket({"nro": 24, "history": 100}),
                CapacityBucket({"nro": 60, "history": 200}),
            ]
        )

    def test_every_chip_and_resource_must_fit(self) -> None:
        self.assertEqual(self.selector.select({"nro": [24, 1], "history": [100, 0]}), 0)
        self.assertEqual(self.selector.select({"nro": [25, 1], "history": [99, 0]}), 1)
        self.assertEqual(self.selector.select({"nro": [1, 1], "history": [101, 0]}), 1)

    def test_maximum_overflow_and_invalid_demand_fail(self) -> None:
        for demand in ({"nro": [61], "history": [1]}, {"nro": [-1], "history": [1]}):
            with (
                self.subTest(demand=demand),
                self.assertRaisesRegex(ValueError, "maximum graph"),
            ):
                self.selector.select(demand)

    def test_remote_overflow_is_collective_before_error(self) -> None:
        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_world_size", return_value=2),
            patch("torch.distributed.get_backend", return_value="gloo"),
            patch(
                "torch.distributed.all_reduce",
                side_effect=lambda tensor, op: tensor.fill_(2),
            ) as reduce,
        ):
            with self.assertRaises(ValueError):
                self.selector.select({"nro": [1], "history": [1]})
            reduce.assert_called_once()

    def test_remote_host_selects_large_graph_for_everyone(self) -> None:
        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_world_size", return_value=2),
            patch("torch.distributed.get_backend", return_value="gloo"),
            patch(
                "torch.distributed.all_reduce",
                side_effect=lambda tensor, op: tensor.fill_(1),
            ),
        ):
            self.assertEqual(self.selector.select({"nro": [1], "history": [1]}), 1)


class _Preprocessor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._static_shape_mode = True
        self._static_nro_per_sample_budget = 20
        self._static_ro_per_sample_budgets_list = [{"history": 10}]


class TestPaddingDemand(TestCase):
    def test_preprocessor_budget_follows_bucket_and_restores_on_error(self) -> None:
        graphs = PaddingGraphs(
            _Preprocessor(), 20, {"history": 10}, '[{"nro":8},{"nro":20}]', 2, 3
        )
        for index, capacity in enumerate((8, 20)):
            with self.subTest(capacity=capacity), self.assertRaises(RuntimeError):
                with graphs.use_preprocessor_budget(index):
                    self.assertEqual(
                        graphs.preprocessor._static_nro_per_sample_budget, capacity
                    )
                    raise RuntimeError("compile failed")
            self.assertEqual(graphs.preprocessor._static_nro_per_sample_budget, 20)

    def test_ro_overflow_falls_back_without_changing_batch(self) -> None:
        model = _Preprocessor()
        graphs = PaddingGraphs(
            model,
            20,
            {"history": 10},
            '[{"nro":8,"ro":{"history":4}},{"nro":20,"ro":{"history":10}}]',
            2,
            3,
        )
        lengths = torch.tensor([7, 6, 1, 1])
        batch = SimpleNamespace(
            num_candidates=torch.tensor([20, 1, 1, 1]),
            ro_id_list_features=KeyedJaggedTensor.from_lengths_sync(
                keys=["history"], lengths=lengths, values=torch.arange(15)
            ),
        )
        original = batch.num_candidates.clone()
        self.assertEqual(graphs.select(batch), 1)
        torch.testing.assert_close(batch.num_candidates, original, rtol=0, atol=0)
        self.assertIs(batch.ro_id_list_features.lengths(), lengths)
        with graphs.use_preprocessor_budget(0):
            self.assertEqual(model._static_nro_per_sample_budget, 8)
            self.assertEqual(model._static_ro_per_sample_budgets_list, [{"history": 4}])
        self.assertEqual(model._static_nro_per_sample_budget, 20)
        self.assertEqual(model._static_ro_per_sample_budgets_list, [{"history": 10}])

    def test_budgeted_sampling_accepts_only_nro_buckets(self) -> None:
        with self.assertRaisesRegex(ValueError, "NRO-only"):
            PaddingGraphs(
                _Preprocessor(),
                20,
                {"history": 10},
                '[{"nro":8,"ro":{"history":4}},{"nro":20}]',
                2,
                3,
                budgeted_sampling=True,
            )
        graphs = PaddingGraphs(
            _Preprocessor(),
            20,
            {"history": 10},
            '[{"nro":8},{"nro":20}]',
            2,
            3,
            budgeted_sampling=True,
        )
        self.assertEqual([budget.nro for budget in graphs.budgets], [8, 20])

    def test_invalid_candidate_counts_are_rejected_at_selection(self) -> None:
        graphs = PaddingGraphs(
            _Preprocessor(), 20, {"history": 10}, '[{"nro":8},{"nro":20}]', 2, 3
        )
        for counts in (
            torch.tensor([21, 1, 1, 1]),
            torch.tensor([-1, 1, 1, 1]),
            torch.ones(4),
            torch.ones(3, dtype=torch.int64),
            torch.ones((2, 2), dtype=torch.int64),
            torch.empty(0, dtype=torch.int64),
        ):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                graphs.select(SimpleNamespace(num_candidates=counts))

    def test_one_request_can_have_twenty_in_small_graph(self) -> None:
        graphs = PaddingGraphs(
            _Preprocessor(), 20, {"history": 10}, '[{"nro":8},{"nro":20}]', 2, 3
        )
        batch = SimpleNamespace(num_candidates=torch.tensor([20, 1, 1, 1]))
        self.assertEqual(graphs.select(batch), 0)


def _step(w, momentum, counter, dummy, x):
    prediction = (x * w).sum(1)
    loss = prediction.square().mean()
    gradient = (x * prediction.unsqueeze(1)).mean(0) * 2
    next_momentum = momentum * 0.9 + gradient
    return loss, w - next_momentum * 0.01, next_momentum, counter + 1, dummy


def _compiled(rows: int):
    state = [
        torch.ones(4),
        torch.zeros(4),
        torch.zeros((), dtype=torch.int64),
        torch.zeros(rows),
    ]
    gm = make_fx(_step, tracing_mode="fake")(*state, torch.ones(rows, 4))
    inputs = [n for n in gm.graph.nodes if n.op == "placeholder"]
    inputs[-1].meta["is_batch_input"] = True
    inputs[-1].meta["batch_dim"] = 0
    module = _XLAModule(gm, backend_type=BackendType.XLA_GSPMD)
    module.safe_donate_indices = {0, 1, 2}
    parent_graph = torch.fx.Graph()
    parent_inputs = []
    for node in inputs:
        placeholder = parent_graph.placeholder(node.name)
        placeholder.meta = dict(node.meta)
        parent_inputs.append(placeholder)
    parent_graph.output(parent_graph.call_module("xla", tuple(parent_inputs)))
    parent = torch.fx.GraphModule({"xla": module}, parent_graph)
    manager = XLADeviceStateManager()
    manager.state_fqns = ["w", "momentum", "counter", "dummy.buffer"]
    return SimpleNamespace(
        gm=parent, xla=module, flat_state=state, state_manager=manager
    )


class TestGraphStateHandoff(TestCase):
    def test_precompile_and_small_large_small_preserve_training_state(self) -> None:
        objects = [_compiled(16), _compiled(40)]
        state = SharedGraphState(objects, ["dummy."])
        for obj in objects:
            before = [tensor.clone() for tensor in obj.flat_state]
            obj.xla.precompile()
            self.assertIsNotNone(obj.xla._jax_compiled)
            self.assertIsNone(obj.state_manager.device_state)
            for original, actual in zip(before, obj.flat_state):
                torch.testing.assert_close(actual, original, rtol=0, atol=0)
        runners = [make_compiled_runner(obj) for obj in objects]
        expected = [torch.ones(4), torch.zeros(4), torch.zeros((), dtype=torch.int64)]
        active = 0
        for index in [0, 1, 0, 1, 0]:
            state.handoff(active, index)
            active = index
            rows = 16 if index == 0 else 40
            x = torch.arange(rows * 4, dtype=torch.float32).reshape(rows, 4) / 100
            loss, *updated = _step(
                expected[0], expected[1], expected[2], torch.zeros(rows), x
            )
            expected = updated[:3]
            actual = runners[index](x)
            torch.testing.assert_close(actual, loss, rtol=1e-5, atol=1e-6)
            for value, expected_value in zip(
                objects[index].state_manager.device_state[:3], expected
            ):
                np.testing.assert_allclose(
                    np.asarray(jax.device_get(value)),
                    expected_value.numpy(),
                    rtol=1e-5,
                    atol=1e-6,
                )
            self.assertEqual(
                objects[index].state_manager.device_state[3].shape, (rows,)
            )
            if index:
                self.assertIsNone(objects[0].state_manager.device_state)
        state.invalidate()
        self.assertTrue(all(obj.state_manager.device_state is None for obj in objects))

    def test_seeded_device_state_keeps_host_leaves_on_host(self) -> None:
        obj = _compiled(16)
        obj.state_manager.initialize_device_state(
            obj.gm, obj.flat_state, host_indices={3}
        )
        state = obj.state_manager.device_state
        self.assertIs(state[3], obj.flat_state[3])
        self.assertTrue(all(isinstance(value, jax.Array) for value in state[:3]))
