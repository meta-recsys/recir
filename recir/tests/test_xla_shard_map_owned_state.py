# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from collections import OrderedDict
from typing import Any, cast
from unittest.mock import MagicMock, patch

import torch
import torch.fx as fx
import torch.nn as nn
from recir.backends.types import BackendType
from recir.distributed.data_parallel import partition_value
from recir.distributed.parallel_mode import DataParallel
from recir.lowering.device_state import DeviceStateManager
from torch.distributed.tensor import DeviceMesh, Shard
from torch.distributed.tensor._dtensor_spec import DTensorSpec


class SharedParameterAliasModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.primary = nn.Linear(4, 4, bias=False)
        self.alias = nn.Linear(4, 4, bias=False)
        self.alias.weight = self.primary.weight


class TestXLAShardMapOwnedState(unittest.TestCase):
    def _mesh(self, size: int, coordinate: int) -> DeviceMesh:
        mesh = MagicMock(spec=DeviceMesh)
        mesh.size.return_value = size
        mesh.get_coordinate.return_value = [coordinate]
        return mesh

    def _mode(self, process_rank: int) -> DataParallel:
        mode = DataParallel(
            "fully_shard",
            num_ranks=4,
            num_processes=2,
            device_type="cpu",
            backend=BackendType.XLA_SHARD_MAP,
        )
        mode._process_mesh = self._mesh(2, process_rank)
        mode._local_device_mesh = self._mesh(2, 0)
        return mode

    def test_owned_partition_is_a_compact_leaf(self) -> None:
        source = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        source.requires_grad_(True)
        spec = DTensorSpec(mesh=self._mesh(2, 0), placements=(Shard(0),))

        view = partition_value(source, spec)
        owned = partition_value(source, spec, own_storage=True)

        torch.testing.assert_close(view, source[:3], atol=0, rtol=0)
        torch.testing.assert_close(owned, source[:3], atol=0, rtol=0)
        self.assertEqual(
            source.untyped_storage().data_ptr(),
            view.untyped_storage().data_ptr(),
        )
        self.assertNotEqual(
            source.untyped_storage().data_ptr(),
            owned.untyped_storage().data_ptr(),
        )
        self.assertEqual(
            source.untyped_storage().nbytes() // 2,
            owned.untyped_storage().nbytes(),
        )
        self.assertTrue(owned.requires_grad)
        self.assertIsNone(owned.grad_fn)

    def test_own_storage_is_a_noop_when_nothing_is_split(self) -> None:
        """A size-1 mesh splits into one chunk, so the "shard" IS the source.

        Cloning there would double the footprint to save nothing, which is why
        callers used to gate `own_storage` on a topology predicate of their own
        (`process_mesh.size() > 1` in one class, `num_processes > 1` in
        another -- and those stopped agreeing once hybrid FSDP made the two
        numbers differ). The decision belongs here, where the mesh is known, so
        callers can state intent unconditionally.
        """
        source = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        spec = DTensorSpec(mesh=self._mesh(1, 0), placements=(Shard(0),))

        owned = partition_value(source, spec, own_storage=True)

        self.assertEqual(
            source.untyped_storage().data_ptr(),
            owned.untyped_storage().data_ptr(),
        )

    def test_process_store_owns_live_trace_storage(self) -> None:
        for process_rank in (0, 1):
            with self.subTest(process_rank=process_rank):
                self._assert_process_store_owns_live_trace_storage(process_rank)

    def test_owned_partition_rejects_uneven_process_rows(self) -> None:
        model = nn.Linear(4, 3, bias=False)
        parameter = model.weight
        optimizer = torch.optim.AdamW([parameter], foreach=True)
        optimizer.state[parameter]["exp_avg"] = torch.ones_like(parameter)
        mode = self._mode(0)

        with (
            patch.object(mode, "_should_fsdp_shard", return_value=True),
            self.assertRaisesRegex(RuntimeError, "divide evenly across processes"),
        ):
            mode.shard_params(
                model,
                cast(
                    dict[str, torch.Tensor],
                    OrderedDict(model.named_parameters()),
                ),
                optimizer,
            )

    def _assert_process_store_owns_live_trace_storage(
        self,
        process_rank: int,
    ) -> None:
        model = nn.Linear(4, 8, bias=False)
        original_parameter = model.weight
        original_state = torch.arange(32, dtype=torch.float32).reshape(8, 4)
        # Citrine C2: exercise the production multi-tensor optimizer path.
        optimizer = torch.optim.AdamW(
            [original_parameter],
            lr=0.01,
            foreach=True,
        )
        optimizer.state[original_parameter]["exp_avg"] = original_state
        mode = self._mode(process_rank)
        params = cast(
            dict[str, torch.Tensor],
            OrderedDict(model.named_parameters()),
        )

        sharded = mode.shard_params(model, params, optimizer)

        self.assertEqual({"weight"}, sharded)
        plan = cast(Any, mode._pending_state_layout)
        process_parameter = plan.process_store["weight"]
        process_state = plan.opt_process_store["weight"]["exp_avg"]
        live_parameter = model.weight
        live_state = optimizer.state[live_parameter]["exp_avg"]
        process_start = process_rank * 4
        torch.testing.assert_close(
            process_parameter,
            original_parameter[process_start : process_start + 4],
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            process_state,
            original_state[process_start : process_start + 4],
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            live_parameter,
            original_parameter[process_start : process_start + 2],
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            live_state,
            original_state[process_start : process_start + 2],
            atol=0,
            rtol=0,
        )
        self.assertIsNone(process_parameter.grad_fn)
        self.assertIsNone(process_state.grad_fn)
        self.assertNotEqual(
            original_parameter.untyped_storage().data_ptr(),
            process_parameter.untyped_storage().data_ptr(),
        )
        self.assertNotEqual(
            original_state.untyped_storage().data_ptr(),
            process_state.untyped_storage().data_ptr(),
        )
        self.assertEqual(
            process_parameter.untyped_storage().data_ptr(),
            live_parameter.untyped_storage().data_ptr(),
        )
        self.assertEqual(
            process_state.untyped_storage().data_ptr(),
            live_state.untyped_storage().data_ptr(),
        )
        self.assertEqual(
            original_parameter.untyped_storage().nbytes() // 2,
            process_parameter.untyped_storage().nbytes(),
        )
        self._assert_runtime_sync_updates_live_views(
            mode,
            process_parameter,
            process_state,
            live_parameter,
            live_state,
        )

    def _assert_runtime_sync_updates_live_views(
        self,
        mode: DataParallel,
        process_parameter: torch.Tensor,
        process_state: torch.Tensor,
        live_parameter: torch.Tensor,
        live_state: torch.Tensor,
    ) -> None:
        graph = fx.Graph()
        parameter_placeholder = graph.placeholder("weight")
        parameter_placeholder.meta["fqn"] = "weight"
        parameter_placeholder.meta["val"] = live_parameter
        state_placeholder = graph.placeholder("exp_avg")
        state_placeholder.meta["val"] = live_state
        graph.output(())
        gm = fx.GraphModule(torch.nn.Module(), graph)
        params_for_layout = {"weight": live_parameter}
        states_for_layout = {"weight": {"exp_avg": live_state}}

        mode.apply_state_layout(gm, params_for_layout, states_for_layout)

        self.assertIs(params_for_layout["weight"], process_parameter)
        self.assertIs(states_for_layout["weight"]["exp_avg"], process_state)
        updated_parameter = torch.arange(
            process_parameter.numel(),
            dtype=process_parameter.dtype,
        ).reshape_as(process_parameter)
        updated_state = updated_parameter + 100
        with torch.no_grad():
            DeviceStateManager().commit_step_state(
                [
                    params_for_layout["weight"],
                    states_for_layout["weight"]["exp_avg"],
                ],
                [updated_parameter, updated_state],
                last_train_step=False,
            )
        torch.testing.assert_close(
            live_parameter,
            updated_parameter[:2],
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            live_state,
            updated_state[:2],
            atol=0,
            rtol=0,
        )

    def test_tied_parameters_share_one_owned_backing(self) -> None:
        model = SharedParameterAliasModel()
        parameter = model.primary.weight
        # Citrine C2: exercise the production multi-tensor optimizer path.
        optimizer = torch.optim.AdamW([parameter], lr=0.01, foreach=True)
        optimizer.state[parameter]["exp_avg"] = torch.ones_like(parameter)
        mode = self._mode(0)

        mode.shard_params(
            model,
            cast(
                dict[str, torch.Tensor],
                OrderedDict(model.named_parameters(remove_duplicate=False)),
            ),
            optimizer,
        )

        plan = cast(Any, mode._pending_state_layout)
        self.assertIs(model.primary.weight, model.alias.weight)
        self.assertIs(
            plan.process_store["primary.weight"],
            plan.process_store["alias.weight"],
        )
        self.assertEqual(
            plan.process_store["primary.weight"].untyped_storage().data_ptr(),
            model.primary.weight.untyped_storage().data_ptr(),
        )
