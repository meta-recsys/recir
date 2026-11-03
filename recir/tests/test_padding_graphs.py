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
from recir.fb.examples.fm_v3_common.padding_graph_set import PaddingGraphSet
from recir.lowering.xla_device_state import XLADeviceStateManager
from recir.lowering.xla_lower import _XLAModule
from recir.training.graph_state import graph_local_state_indices, SharedGraphState
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
        graphs = PaddingGraphSet(
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
        graphs = PaddingGraphSet(
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
            PaddingGraphSet(
                _Preprocessor(),
                20,
                {"history": 10},
                '[{"nro":8,"ro":{"history":4}},{"nro":20}]',
                2,
                3,
                budgeted_sampling=True,
            )
        graphs = PaddingGraphSet(
            _Preprocessor(),
            20,
            {"history": 10},
            '[{"nro":8},{"nro":20}]',
            2,
            3,
            budgeted_sampling=True,
        )
        self.assertEqual(
            [budget.nro for budget in graphs.budgets if budget is not None], [8, 20]
        )

    def test_invalid_candidate_counts_are_rejected_at_selection(self) -> None:
        graphs = PaddingGraphSet(
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
        graphs = PaddingGraphSet(
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
    def test_default_sharing_preserves_existing_resident_state(self) -> None:
        objects = [_compiled(16), _compiled(16)]
        first, second = [obj.state_manager for obj in objects]
        first.device_state = list(objects[0].flat_state)
        store = {}
        for obj in objects:
            obj.state_manager.share_device_state(store, obj.gm, obj.flat_state)
        self.assertCountEqual(store, first.state_fqns)
        updated = [torch.full_like(value, 7) for value in objects[1].flat_state]
        second.commit_step_state(objects[1].flat_state, updated, False)
        for actual, expected in zip(first.device_state, updated):
            self.assertIs(actual, expected)
        second.invalidate_device_state()
        self.assertEqual(store, {})
        self.assertIsNone(first.device_state)

    def test_local_state_is_not_shared_or_reset_by_a_switch(self) -> None:
        objects = [_compiled(16), _compiled(40)]
        shared = SharedGraphState(objects, [{3}, {3}])
        for index, obj in enumerate(objects):
            obj.state_manager.device_state = [
                *obj.flat_state[:3],
                torch.full_like(obj.flat_state[3], index + 7),
            ]
        self.assertCountEqual(shared.store, ["w", "momentum", "counter"])
        objects[0].state_manager.keep_output_jax = True
        for source, target in [(0, 1), (1, 0)]:
            shared.handoff(source, target)
            state = objects[target].state_manager.device_state
            torch.testing.assert_close(
                state[3], torch.full_like(objects[target].flat_state[3], target + 7)
            )
            self.assertTrue(objects[target].state_manager.keep_output_jax)
            self.assertIs(state[0], objects[source].state_manager.device_state[0])
        shared.invalidate()
        self.assertEqual(shared.store, {})
        self.assertTrue(all(obj.state_manager.device_state is None for obj in objects))

    def test_local_declaration_matches_exact_model_and_optimizer_keys(self) -> None:
        names = [
            "[0]['dummy.buffer']",
            "[1]['dummy.buffer']['momentum']",
            "[0]['notdummy.buffer']",
            "[0]['nested.dummy.buffer']",
            "[0]['dummy.buffer_extra']",
            "[1]['notdummy.buffer']['momentum']",
        ]
        self.assertEqual(graph_local_state_indices(names, {"dummy.buffer"}), {0, 1})

    def test_similarly_named_training_state_is_validated_and_shared(self) -> None:
        objects = [_compiled(16), _compiled(40)]
        for obj in objects:
            obj.state_manager.state_fqns[0] = "[0]['notdummy.buffer']"
            obj.state_manager.state_fqns[3] = "[0]['dummy.buffer']"
        local = [
            graph_local_state_indices(obj.state_manager.state_fqns, {"dummy.buffer"})
            for obj in objects
        ]
        shared = SharedGraphState(objects, local)
        objects[0].state_manager.device_state = [
            torch.full((4,), 7),
            *objects[0].flat_state[1:],
        ]
        shared.handoff(0, 1)
        torch.testing.assert_close(
            objects[1].state_manager.device_state[0], torch.full((4,), 7)
        )

        objects[0].flat_state[0] = torch.zeros(5)
        with self.assertRaisesRegex(ValueError, "incompatible state: .*notdummy"):
            SharedGraphState(objects, local)

    def test_undeclared_local_state_and_invalid_layout_fail(self) -> None:
        objects = [_compiled(16), _compiled(40)]
        with self.assertRaisesRegex(ValueError, "incompatible state: dummy.buffer"):
            SharedGraphState(objects, [set(), set()])
        first_input = next(
            n for n in objects[0].gm.graph.nodes if n.op == "placeholder"
        )
        first_input.meta["is_fsdp_sharded"] = True
        with self.assertRaisesRegex(ValueError, "incompatible state layout: w"):
            SharedGraphState(objects, [{3}, {3}])

    def test_precompile_and_small_large_small_preserve_training_state(self) -> None:
        objects = [_compiled(16), _compiled(40)]
        graphs = PaddingGraphSet(
            _Preprocessor(), 20, {"history": 10}, '[{"nro":8},{"nro":20}]', 1, 3
        )
        graphs.steps = [make_compiled_runner(obj) for obj in objects]
        graphs.build_shared_state([{3}, {3}])
        for obj in objects:
            before = [tensor.clone() for tensor in obj.flat_state]
            obj.xla.precompile()
            self.assertIsNotNone(obj.xla._jax_compiled)
            self.assertIsNone(obj.state_manager.device_state)
            for original, actual in zip(before, obj.flat_state):
                torch.testing.assert_close(actual, original, rtol=0, atol=0)
        graphs.initialize_device_state()
        expected = [torch.ones(4), torch.zeros(4), torch.zeros((), dtype=torch.int64)]
        for index in [1, 0, 1, 0, 1, 0]:
            runner = graphs.activate(index)
            rows = 16 if index == 0 else 40
            x = torch.arange(rows * 4, dtype=torch.float32).reshape(rows, 4) / 100
            loss, *updated = _step(
                expected[0], expected[1], expected[2], torch.zeros(rows), x
            )
            expected = updated[:3]
            actual = runner(x)
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
            self.assertEqual(graphs.active_index, index)
            for leaf in range(3):
                self.assertIs(
                    objects[index].state_manager.device_state[leaf],
                    objects[1 - index].state_manager.device_state[leaf],
                )
            self.assertEqual(
                objects[1 - index].state_manager.device_state[3].shape,
                (40 if index == 0 else 16,),
            )
        assert graphs.shared_state is not None
        objects[0].state_manager.copy_state_to_host(objects[0].flat_state)
        checkpoint = [value.clone() for value in objects[0].flat_state[:3]]
        continuation = []
        for index in [1, 0]:
            rows = 16 if index == 0 else 40
            x = torch.arange(rows * 4, dtype=torch.float32).reshape(rows, 4) / 100
            actual = graphs.activate(index)(x)
            state = objects[index].state_manager.device_state
            continuation.append(
                (
                    actual.clone(),
                    [np.array(jax.device_get(value)) for value in state[:3]],
                )
            )
        graphs.shared_state.invalidate()
        objects[1].flat_state[:3] = [value.clone() for value in checkpoint]
        graphs.shared_state.refresh_host_state(1)
        objects[1].state_manager.prefill_device_state_from_host(
            objects[1].gm, objects[1].flat_state
        )
        for index, (loss, expected_state) in zip([1, 0], continuation):
            rows = 16 if index == 0 else 40
            x = torch.arange(rows * 4, dtype=torch.float32).reshape(rows, 4) / 100
            actual = graphs.activate(index)(x)
            torch.testing.assert_close(actual, loss, rtol=1e-5, atol=1e-6)
            for value, expected_value in zip(
                objects[index].state_manager.device_state[:3], expected_state
            ):
                np.testing.assert_allclose(
                    np.asarray(jax.device_get(value)),
                    expected_value,
                    rtol=1e-5,
                    atol=1e-6,
                )
        graphs.shared_state.invalidate()
        self.assertTrue(all(obj.state_manager.device_state is None for obj in objects))

    def test_seeded_device_state_keeps_host_leaves_on_host(self) -> None:
        obj = _compiled(16)
        obj.state_manager.initialize_device_state(
            obj.gm, obj.flat_state, host_indices={3}
        )
        state = obj.state_manager.device_state
        self.assertIs(state[3], obj.flat_state[3])
        self.assertTrue(all(isinstance(value, jax.Array) for value in state[:3]))
