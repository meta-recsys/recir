# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Unit tests for FSDP support in the GSPMD path."""

import hashlib
import os
import unittest
from typing import Any, Callable, cast, Sequence, Tuple
from unittest import mock

import jax  # @manual
import numpy as np
import torch
import torch.nn as nn
import torch.utils._pytree as pytree
from jax.sharding import Mesh, PartitionSpec as P  # @manual
from parameterized import parameterized
from recir.api import compile_from_gm, trace
from recir.backends.base import Backend
from recir.backends.types import BackendType
from recir.compiler.process_state import (
    _annotate_fsdp_state_output_sharding,
    _build_fsdp_sharded_flags,
)
from recir.distributed.parallel_mode import (
    _current_process_rank,
    _FAKE_PG_KEEPALIVE,
    DataParallel,
    GSPMDParallel,
)
from recir.lowering.xla_lower import _XLAModule

# IMPORTANT: must run BEFORE any `import jax` so the cuda plugin path patch
# is in effect when jax.devices() / jax.* first triggers plugin discovery.
from recir.tests.xla import _jax_cuda_plugin_fix  # noqa: F401, I001
from torch import fx


class _IdentityBackend(Backend):
    def dispatch(self, gm: fx.GraphModule) -> nn.Module:
        return gm


class TestGSPMDShardParams(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        process_count = mock.patch("jax.process_count", return_value=2)
        process_index = mock.patch("jax.process_index", return_value=0)
        process_count.start()
        process_index.start()
        self.addCleanup(process_count.stop)
        self.addCleanup(process_index.stop)

    def _make_model_and_opt(self, dim: int = 256) -> Tuple[nn.Linear, torch.optim.SGD]:
        model = nn.Linear(dim, dim)
        opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=True)
        x = torch.randn(4, dim)
        model(x).sum().backward()
        opt.step()
        opt.zero_grad()
        return model, opt

    def test_shard_params_noop_for_replicate(self) -> None:
        mode = GSPMDParallel("replicate", num_ranks=4, num_processes=2)
        model, opt = self._make_model_and_opt()
        orig_weight_shape = model.weight.shape
        orig_bias_shape = model.bias.shape
        params_and_buffers = dict(model.named_parameters())
        mode.shard_params(model, params_and_buffers, opt)
        self.assertEqual(model.weight.shape, orig_weight_shape)
        self.assertEqual(model.bias.shape, orig_bias_shape)

    def test_nested_parameter_flags_follow_pytree_tensor_leaves(self) -> None:
        flags = _build_fsdp_sharded_flags(
            {
                "nested": {
                    "vector": torch.ones(2),
                    "scalar": torch.tensor(1.0),
                    "matrix": torch.ones(2, 2),
                }
            },
            {},
            {"nested"},
        )

        self.assertEqual([True, False, True], flags)

    def test_state_output_metadata_rejects_reordered_state(self) -> None:
        first_state = torch.ones(2)
        second_state = torch.zeros(2)
        graph = fx.Graph()
        first_output = graph.placeholder("first_state")
        first_output.meta["val"] = first_state
        second_output = graph.placeholder("second_state")
        second_output.meta["val"] = second_state
        output = graph.output((None, first_output, second_output))
        graph_module = fx.GraphModule({}, graph)

        _annotate_fsdp_state_output_sharding(
            graph_module,
            [first_state, second_state],
            [True, False],
        )
        output.args = ((None, second_output, first_output),)

        with self.assertRaisesRegex(RuntimeError, "output order"):
            _annotate_fsdp_state_output_sharding(
                graph_module,
                [first_state, second_state],
                [True, False],
            )

    def test_shard_params_shards_weight(self) -> None:
        dim = 256
        num_procs = 2
        num_ranks = 4
        mode = GSPMDParallel(
            "fully_shard", num_ranks=num_ranks, num_processes=num_procs
        )
        model, opt = self._make_model_and_opt(dim=dim)
        original_weight = model.weight.clone()
        params_and_buffers = {
            **dict(model.named_parameters()),
            **dict(model.named_buffers()),
        }
        mode.shard_params(model, params_and_buffers, opt)
        expected_dim0 = dim // num_procs
        self.assertEqual(model.weight.shape[0], expected_dim0)
        self.assertEqual(model.weight.shape[1], dim)
        torch.testing.assert_close(
            model.weight, original_weight[:expected_dim0, :], atol=0, rtol=0
        )

    def test_multihost_shards_own_compact_parameter_and_optimizer_storage(
        self,
    ) -> None:
        dim = 32
        mode = GSPMDParallel("fully_shard", num_ranks=4, num_processes=2)
        model = nn.Linear(dim, dim, bias=False)
        # Citrine C2: use foreach AdamW for multi-tensor optimizer execution.
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, foreach=True)
        model(torch.randn(4, dim)).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        params_and_buffers = dict(model.named_parameters())

        mode.shard_params(model, params_and_buffers, optimizer)

        self.assertIn(model.weight, optimizer.state)
        self.assertEqual(
            model.weight.numel() * model.weight.element_size(),
            model.weight.untyped_storage().nbytes(),
        )
        for state_name in ("exp_avg", "exp_avg_sq"):
            state = optimizer.state[model.weight][state_name]
            self.assertEqual(
                state.numel() * state.element_size(),
                state.untyped_storage().nbytes(),
            )

    def test_hybrid_fsdp_keeps_full_state_on_each_host(self) -> None:
        dim = 16
        mode = GSPMDParallel(
            "fully_shard",
            num_ranks=8,
            num_processes=2,
            fsdp_axis_size=4,
        )
        model, optimizer = self._make_model_and_opt(dim=dim)
        original_weight = model.weight.detach().clone()
        params_and_buffers = dict(model.named_parameters())

        sharded_keys = mode.shard_params(model, params_and_buffers, optimizer)

        self.assertTrue(mode.uses_hybrid_fsdp)
        self.assertEqual(4, mode.fsdp_axis_size)
        self.assertIn("weight", sharded_keys)
        self.assertEqual(original_weight.shape, model.weight.shape)
        torch.testing.assert_close(model.weight, original_weight, atol=0, rtol=0)
        self.assertEqual(1, mode.process_mesh.size())

    def test_hybrid_fsdp_shards_state_across_hosts_in_each_group(self) -> None:
        dim = 16
        mode = GSPMDParallel(
            "fully_shard",
            num_ranks=8,
            num_processes=4,
            fsdp_axis_size=4,
        )
        model, optimizer = self._make_model_and_opt(dim=dim)
        original_weight = model.weight.detach().clone()
        params_and_buffers = dict(model.named_parameters())

        with mock.patch(
            "recir.distributed.parallel_mode._current_process_rank",
            return_value=3,
        ):
            sharded_keys = mode.shard_params(model, params_and_buffers, optimizer)

        self.assertIn("weight", sharded_keys)
        self.assertEqual((dim // 2, dim), model.weight.shape)
        torch.testing.assert_close(
            model.weight,
            original_weight[dim // 2 :, :],
            atol=0,
            rtol=0,
        )
        self.assertEqual(2, mode.process_mesh.size())

    def test_hybrid_fsdp_validates_axis_and_state_only_sharding(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible"):
            GSPMDParallel(
                "fully_shard",
                num_ranks=8,
                num_processes=2,
                fsdp_axis_size=3,
            )

        with self.assertRaisesRegex(ValueError, "not supported with hybrid"):
            GSPMDParallel(
                "fully_shard",
                num_ranks=8,
                num_processes=2,
                fsdp_replicate_param_substrs=("per_layer_",),
                fsdp_shard_optimizer_state_for_replicated_param_substrs=("per_layer_",),
                fsdp_sharded_optimizer_state_keys=("exp_avg",),
                fsdp_axis_size=4,
            )

    def test_shard_params_uses_current_process_rank(self) -> None:
        dim = 8
        num_procs = 2
        mode = GSPMDParallel("fully_shard", num_ranks=4, num_processes=num_procs)
        model = nn.Linear(dim, dim, bias=False)
        original_weight = torch.arange(dim * dim, dtype=model.weight.dtype).reshape(
            dim, dim
        )
        with torch.no_grad():
            model.weight.copy_(original_weight)
        opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=True)
        params_and_buffers = dict(model.named_parameters())

        with (
            mock.patch(
                "recir.distributed.parallel_mode.dist.is_available",
                return_value=True,
            ),
            mock.patch(
                "recir.distributed.parallel_mode.dist.is_initialized",
                return_value=True,
            ),
            mock.patch(
                "recir.distributed.parallel_mode.dist.get_world_size",
                return_value=num_procs,
            ),
            mock.patch("recir.distributed.parallel_mode.dist.get_rank", return_value=1),
        ):
            mode.shard_params(model, params_and_buffers, opt)

        expected_dim0 = dim // num_procs
        torch.testing.assert_close(
            model.weight,
            original_weight[expected_dim0:, :],
            atol=0,
            rtol=0,
        )

    def test_process_mesh_ignores_torch_rank_from_different_world(self) -> None:
        dim = 8
        num_processes = 2
        mode = GSPMDParallel(
            "fully_shard",
            num_ranks=4,
            num_processes=num_processes,
        )
        model = nn.Linear(dim, dim, bias=False)
        original_weight = torch.arange(dim * dim, dtype=model.weight.dtype).reshape(
            dim, dim
        )
        with torch.no_grad():
            model.weight.copy_(original_weight)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, foreach=True)
        params_and_buffers = dict(model.named_parameters())

        with (
            mock.patch(
                "recir.distributed.parallel_mode.dist.is_available",
                return_value=True,
            ),
            mock.patch(
                "recir.distributed.parallel_mode.dist.is_initialized",
                return_value=True,
            ),
            mock.patch(
                "recir.distributed.parallel_mode.dist.get_world_size",
                return_value=4,
            ),
            mock.patch(
                "recir.distributed.parallel_mode.dist.get_rank",
                return_value=3,
            ),
            mock.patch("jax.process_count", return_value=num_processes),
            mock.patch("jax.process_index", return_value=1),
        ):
            mode.shard_params(model, params_and_buffers, optimizer)

        expected_rows = dim // num_processes
        torch.testing.assert_close(
            model.weight,
            original_weight[expected_rows:, :],
            atol=0,
            rtol=0,
        )

    def test_process_mesh_rejects_out_of_range_matching_jax_rank(self) -> None:
        mode = GSPMDParallel("fully_shard", num_ranks=4, num_processes=2)

        with (
            mock.patch(
                "recir.distributed.parallel_mode.dist.is_initialized",
                return_value=False,
            ),
            mock.patch("jax.process_count", return_value=2),
            mock.patch("jax.process_index", return_value=2),
            self.assertRaisesRegex(RuntimeError, "outside"),
        ):
            _ = mode.process_mesh

    def test_process_mesh_propagates_jax_topology_runtime_failure(self) -> None:
        mode = GSPMDParallel("fully_shard", num_ranks=4, num_processes=2)

        with (
            mock.patch(
                "recir.distributed.parallel_mode.dist.is_initialized",
                return_value=False,
            ),
            mock.patch(
                "jax.process_count", side_effect=RuntimeError("topology failed")
            ),
            mock.patch.dict(os.environ, {"RANK": "1", "WORLD_SIZE": "2"}, clear=True),
            self.assertRaisesRegex(RuntimeError, "topology failed"),
        ):
            _ = mode.process_mesh

    def test_process_mesh_rejects_single_process_jax_topology(self) -> None:
        mode = GSPMDParallel("fully_shard", num_ranks=4, num_processes=2)

        with (
            mock.patch(
                "recir.distributed.parallel_mode.dist.is_initialized",
                return_value=False,
            ),
            mock.patch("jax.process_count", return_value=1),
            mock.patch("jax.process_index", return_value=0),
            mock.patch.dict(os.environ, {}, clear=True),
            self.assertRaisesRegex(RuntimeError, "jax=1"),
        ):
            _ = mode.process_mesh

    def test_process_mesh_ignores_incomplete_environment_namespace(self) -> None:
        mode = GSPMDParallel("fully_shard", num_ranks=4, num_processes=2)

        with (
            mock.patch(
                "recir.distributed.parallel_mode.dist.is_initialized",
                return_value=False,
            ),
            mock.patch(
                "recir.distributed.parallel_mode._jax_process_rank",
                return_value=(None, None),
            ),
            mock.patch.dict(os.environ, {"RANK": "unrelated"}, clear=True),
        ):
            process_rank = _current_process_rank(mode.num_processes)

        self.assertEqual(0, process_rank)

    def test_process_and_device_mesh_use_distinct_rank_namespaces(self) -> None:
        mode = DataParallel(
            "fully_shard",
            num_ranks=4,
            num_processes=2,
            device_type="cpu",
            backend=BackendType.XLA_SHARD_MAP,
        )

        with (
            mock.patch(
                "recir.distributed.parallel_mode.dist.is_available",
                return_value=True,
            ),
            mock.patch(
                "recir.distributed.parallel_mode.dist.is_initialized",
                return_value=True,
            ),
            mock.patch(
                "recir.distributed.parallel_mode.dist.get_world_size",
                return_value=2,
            ),
            mock.patch(
                "recir.distributed.parallel_mode.dist.get_rank",
                return_value=1,
            ),
        ):
            process_rank = mode.process_mesh.get_group().rank()
            device_rank = mode.device_mesh.get_group().rank()

        self.assertEqual(1, process_rank)
        self.assertEqual(2, device_rank)

    def test_fake_process_group_is_replaced_when_rank_changes(self) -> None:
        with (
            mock.patch(
                "recir.distributed.parallel_mode.dist.is_available", return_value=True
            ),
            mock.patch(
                "recir.distributed.parallel_mode.dist.is_initialized", return_value=True
            ),
            mock.patch(
                "recir.distributed.parallel_mode.dist.get_world_size", return_value=2
            ),
            mock.patch(
                "recir.distributed.parallel_mode.dist.get_rank", side_effect=(0, 1, 0)
            ),
        ):
            ranks = []
            for _ in range(3):
                mode = DataParallel(
                    "fully_shard",
                    num_ranks=4,
                    num_processes=2,
                    device_type="cpu",
                    backend=BackendType.XLA_SHARD_MAP,
                )
                _ = mode.process_mesh
                ranks.append(_FAKE_PG_KEEPALIVE["fake_recir_process_mesh"].rank())
        self.assertEqual([0, 1, 0], ranks)

    def test_shard_map_fsdp_replicates_unsafe_geometry_and_substrings(self) -> None:
        class MixedGeometryModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.dense = nn.Linear(8, 8, bias=False)
                self.per_layer_projection = nn.Linear(8, 8, bias=False)
                self.register_buffer("rotary", torch.ones(1, 8))
                self.register_buffer("position_table", torch.ones(8, 8))

        model = MixedGeometryModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, foreach=True)
        mode = DataParallel(
            "fully_shard",
            num_ranks=4,
            num_processes=2,
            device_type="cpu",
            backend=BackendType.XLA_SHARD_MAP,
            fsdp_replicate_param_substrs=("per_layer_",),
        )
        params_and_buffers = {
            **dict(model.named_parameters()),
            **dict(model.named_buffers()),
        }

        sharded_keys = mode.shard_params(model, params_and_buffers, optimizer)

        self.assertEqual({"dense.weight"}, sharded_keys)
        self.assertEqual((2, 8), tuple(model.dense.weight.shape))
        self.assertEqual((8, 8), tuple(model.per_layer_projection.weight.shape))
        self.assertEqual((1, 8), tuple(cast(torch.Tensor, model.rotary).shape))
        self.assertEqual((8, 8), tuple(cast(torch.Tensor, model.position_table).shape))
        plan = mode._pending_state_layout
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual({"dense.weight"}, set(plan.process_store))
        self.assertEqual((4, 8), tuple(plan.process_store["dense.weight"].shape))

    def test_shard_params_replicates_matching_substrings(self) -> None:
        dim = 8
        num_procs = 2

        class AttentionWithQKV(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.qkv = nn.Linear(dim, dim, bias=False)

        class ModelWithStructuredParams(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.dense = nn.Linear(dim, dim, bias=False)
                self.per_layer_projection = nn.Linear(dim, dim, bias=False)
                self.attn = AttentionWithQKV()

        mode = GSPMDParallel(
            "fully_shard",
            num_ranks=4,
            num_processes=num_procs,
            fsdp_replicate_param_substrs=("per_layer_", "qkv"),
        )
        model = ModelWithStructuredParams()
        opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=True)
        params_and_buffers = dict(model.named_parameters())

        sharded_keys = mode.shard_params(model, params_and_buffers, opt)

        self.assertIn("dense.weight", sharded_keys)
        self.assertNotIn("per_layer_projection.weight", sharded_keys)
        self.assertNotIn("attn.qkv.weight", sharded_keys)
        self.assertEqual(model.dense.weight.shape[0], dim // num_procs)
        self.assertEqual(model.per_layer_projection.weight.shape[0], dim)
        self.assertEqual(model.attn.qkv.weight.shape[0], dim)

    def test_shards_selected_optimizer_state_for_replicated_parameter(self) -> None:
        dim = 8
        num_processes = 2

        class StructuredModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.dense = nn.Linear(dim, dim, bias=False)
                self.per_layer_weight = nn.Parameter(torch.randn(dim, dim))

        model = StructuredModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, foreach=False)
        state = optimizer.state[model.per_layer_weight]
        state["step"] = torch.tensor(1.0)
        full_state = torch.arange(dim * dim, dtype=torch.float32).reshape(dim, dim)
        for state_key in ("master", "exp_avg", "exp_avg_sq", "grad_accum"):
            state[state_key] = full_state.clone()
        original_parameter = model.per_layer_weight
        original_value = original_parameter.detach().clone()
        mode = GSPMDParallel(
            "fully_shard",
            num_ranks=4,
            num_processes=num_processes,
            fsdp_replicate_param_substrs=("per_layer_",),
            fsdp_shard_optimizer_state_for_replicated_param_substrs=("per_layer_",),
            fsdp_sharded_optimizer_state_keys=(
                "master",
                "exp_avg",
                "exp_avg_sq",
                "grad_accum",
            ),
        )
        params_and_buffers = dict(model.named_parameters())

        sharded_parameter_keys = mode.shard_params(
            model,
            params_and_buffers,
            optimizer,
        )

        self.assertNotIn("per_layer_weight", sharded_parameter_keys)
        self.assertIs(original_parameter, model.per_layer_weight)
        torch.testing.assert_close(
            model.per_layer_weight,
            original_value,
            atol=0,
            rtol=0,
        )
        expected_state_keys = {
            ("per_layer_weight", state_key)
            for state_key in ("master", "exp_avg", "exp_avg_sq", "grad_accum")
        }
        self.assertEqual(
            expected_state_keys,
            set(mode.optimizer_state_only_sharded_keys),
        )
        expected_rows = dim // num_processes
        for state_key in ("master", "exp_avg", "exp_avg_sq", "grad_accum"):
            self.assertEqual((expected_rows, dim), state[state_key].shape)
            torch.testing.assert_close(
                state[state_key],
                full_state[:expected_rows],
                atol=0,
                rtol=0,
            )
        self.assertEqual((), state["step"].shape)
        self.assertEqual(1.0, float(state["step"]))

    def test_state_only_sharding_partitions_shared_parameter_once(self) -> None:
        dim = 8

        class SharedParameterModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                parameter = nn.Parameter(torch.randn(dim, dim))
                self.primary_weight = parameter
                self.alias_weight = parameter

        model = SharedParameterModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
        parameter = model.primary_weight
        full_state = torch.arange(dim * dim, dtype=torch.float32).reshape(dim, dim)
        optimizer.state[parameter]["moment"] = full_state.clone()
        mode = GSPMDParallel(
            "fully_shard",
            num_ranks=4,
            num_processes=2,
            fsdp_replicate_param_substrs=("primary_", "alias_"),
            fsdp_shard_optimizer_state_for_replicated_param_substrs=(
                "primary_",
                "alias_",
            ),
            fsdp_sharded_optimizer_state_keys=("moment",),
        )

        mode.shard_params(
            model,
            dict(model.named_parameters(remove_duplicate=False)),
            optimizer,
        )

        self.assertEqual((dim // 2, dim), optimizer.state[parameter]["moment"].shape)
        torch.testing.assert_close(
            optimizer.state[parameter]["moment"],
            full_state[: dim // 2],
            atol=0,
            rtol=0,
        )
        self.assertEqual(
            {
                ("primary_weight", "moment"),
                ("alias_weight", "moment"),
            },
            set(mode.optimizer_state_only_sharded_keys),
        )

    def test_state_only_sharding_rejects_inconsistent_alias_selection(self) -> None:
        dim = 8

        class SharedParameterModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                parameter = nn.Parameter(torch.randn(dim, dim))
                self.primary_weight = parameter
                self.alias_weight = parameter

        model = SharedParameterModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
        optimizer.state[model.primary_weight]["moment"] = torch.zeros(dim, dim)
        mode = GSPMDParallel(
            "fully_shard",
            num_ranks=4,
            num_processes=2,
            fsdp_replicate_param_substrs=("primary_", "alias_"),
            fsdp_shard_optimizer_state_for_replicated_param_substrs=("primary_",),
            fsdp_sharded_optimizer_state_keys=("moment",),
        )

        with self.assertRaisesRegex(RuntimeError, "every alias"):
            mode.shard_params(
                model,
                dict(model.named_parameters(remove_duplicate=False)),
                optimizer,
            )

    def test_state_only_sharding_marks_inputs_and_outputs_per_leaf(self) -> None:
        dim = 8

        class StructuredModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.per_layer_weight = nn.Parameter(torch.randn(dim, dim))

            def forward(self, value: torch.Tensor) -> torch.Tensor:
                return value @ self.per_layer_weight

        model = StructuredModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, foreach=False)
        state = optimizer.state[model.per_layer_weight]
        state["step"] = torch.tensor(0.0)
        state["exp_avg"] = torch.zeros_like(model.per_layer_weight, dtype=torch.float32)
        state["exp_avg_sq"] = torch.zeros_like(
            model.per_layer_weight,
            dtype=torch.float32,
        )
        value = torch.randn(4, dim)
        mode = GSPMDParallel(
            "fully_shard",
            num_ranks=4,
            num_processes=2,
            fsdp_replicate_param_substrs=("per_layer_",),
            fsdp_shard_optimizer_state_for_replicated_param_substrs=("per_layer_",),
            fsdp_sharded_optimizer_state_keys=("exp_avg", "exp_avg_sq"),
        )

        def train_step(
            mod: Any, opt: Any, batch: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            loss = mod(batch).sum()
            loss.backward()
            grad = mod.per_layer_weight.grad
            if grad is None:
                raise RuntimeError("Expected a parameter gradient")
            state = opt.state[mod.per_layer_weight]
            with torch.no_grad():
                state["step"].add_(1.0)
                state["exp_avg"].add_(grad)
                state["exp_avg_sq"].addcmul_(grad, grad)
            return loss, loss.detach()

        gm, flat_state = trace(
            train_step,
            model,
            optimizer,
            value,
            parallel_mode=mode,
        )

        def append_replicated_tail_state(
            graph_module: fx.GraphModule,
            state_values: list[torch.Tensor],
        ) -> tuple[fx.GraphModule, list[torch.Tensor]]:
            placeholders = [
                node for node in graph_module.graph.nodes if node.op == "placeholder"
            ]
            anchor = placeholders[len(state_values) - 1]
            tail_value = torch.zeros_like(state_values[-1])
            with graph_module.graph.inserting_after(anchor):
                tail_placeholder = graph_module.graph.placeholder("tail_state")
            tail_placeholder.meta["val"] = anchor.meta["val"]
            tail_placeholder.meta["is_state_input"] = True
            tail_placeholder.meta["is_batch_input"] = False
            tail_placeholder.meta["is_weight"] = False
            output_node = next(
                node for node in graph_module.graph.nodes if node.op == "output"
            )
            outputs = output_node.args[0]
            self.assertIsInstance(outputs, (list, tuple))
            output_node.args = ((*cast(Sequence[Any], outputs), tail_placeholder),)
            state_values.append(tail_value)
            graph_module.graph.lint()
            graph_module.recompile()
            return graph_module, state_values

        compiled = compile_from_gm(
            gm,
            flat_state,
            parallel_mode=mode,
            backend=_IdentityBackend(optimize_passes=(append_replicated_tail_state,)),
        )

        tensor_placeholders = [
            node
            for node in compiled.gm.graph.nodes
            if node.op == "placeholder"
            and isinstance(node.meta.get("val", None), torch.Tensor)
        ]
        parameter_placeholder = next(
            node
            for node in tensor_placeholders
            if node.meta.get("fqn") == "per_layer_weight"
        )
        self.assertFalse(parameter_placeholder.meta["is_fsdp_sharded"])
        state_placeholders = tensor_placeholders[1:4]
        self.assertEqual(
            [False, True, True],
            [
                bool(node.meta.get("is_fsdp_sharded", False))
                for node in state_placeholders
            ],
        )

        output_node = next(
            node for node in compiled.gm.graph.nodes if node.op == "output"
        )
        graph_outputs = output_node.args[0]
        self.assertIsInstance(graph_outputs, (list, tuple))
        graph_output_leaves = pytree.tree_leaves(cast(Sequence[Any], graph_outputs))
        state_output_values = graph_output_leaves[-len(compiled.flat_state) :]
        user_output_leaves = graph_output_leaves[: -len(compiled.flat_state)]
        self.assertEqual(2, len(user_output_leaves))
        self.assertFalse(
            any(
                isinstance(leaf, fx.Node) and "is_fsdp_sharded_output" in leaf.meta
                for leaf in user_output_leaves
            )
        )
        state_output_leaves = [
            leaf
            for leaf in state_output_values
            if isinstance(leaf, fx.Node)
            and isinstance(leaf.meta.get("val", None), torch.Tensor)
        ]
        self.assertEqual(
            [False, False, True, True, False],
            [bool(leaf.meta["is_fsdp_sharded_output"]) for leaf in state_output_leaves],
        )

    def test_state_input_annotation_rejects_missing_metadata(self) -> None:
        model = nn.Linear(8, 8, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, foreach=True)
        batch = torch.randn(4, 8)
        mode = GSPMDParallel(
            "fully_shard",
            num_ranks=4,
            num_processes=2,
        )

        def train_step(
            mod: nn.Module,
            _opt: torch.optim.Optimizer,
            value: torch.Tensor,
        ) -> torch.Tensor:
            loss = mod(value).sum()
            loss.backward()
            return loss

        gm, flat_state = trace(
            train_step,
            model,
            optimizer,
            batch,
            parallel_mode=mode,
        )

        def remove_state_input_metadata(
            graph_module: fx.GraphModule,
        ) -> fx.GraphModule:
            state_placeholder = next(
                node
                for node in graph_module.graph.nodes
                if node.op == "placeholder" and node.meta.get("is_state_input", False)
            )
            state_placeholder.meta.pop("is_state_input")
            return graph_module

        with self.assertRaisesRegex(RuntimeError, "requires is_state_input metadata"):
            compile_from_gm(
                gm,
                flat_state,
                parallel_mode=mode,
                backend=_IdentityBackend(
                    optimize_passes=(remove_state_input_metadata,)
                ),
            )

    def test_state_only_sharding_requires_matching_parameter(self) -> None:
        model = nn.Linear(8, 8, bias=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, foreach=False)
        mode = GSPMDParallel(
            "fully_shard",
            num_ranks=4,
            num_processes=2,
            fsdp_replicate_param_substrs=("per_layer_",),
            fsdp_shard_optimizer_state_for_replicated_param_substrs=("per_layer_",),
            fsdp_sharded_optimizer_state_keys=("exp_avg",),
        )

        with self.assertRaisesRegex(RuntimeError, "matched no parameters"):
            mode.shard_params(model, dict(model.named_parameters()), optimizer)

    def test_state_only_sharding_requires_initialized_optimizer_state(self) -> None:
        model = nn.Module()
        model.register_parameter(
            "per_layer_weight",
            nn.Parameter(torch.randn(8, 8)),
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, foreach=False)
        mode = GSPMDParallel(
            "fully_shard",
            num_ranks=4,
            num_processes=2,
            fsdp_replicate_param_substrs=("per_layer_",),
            fsdp_shard_optimizer_state_for_replicated_param_substrs=("per_layer_",),
            fsdp_sharded_optimizer_state_keys=("exp_avg",),
        )

        with self.assertRaisesRegex(RuntimeError, "uninitialized optimizer state"):
            mode.shard_params(model, dict(model.named_parameters()), optimizer)

    def test_state_only_sharding_requires_optimizer_owner(self) -> None:
        model = nn.Module()
        model.register_parameter(
            "per_layer_weight",
            nn.Parameter(torch.randn(8, 8)),
        )
        unrelated_parameter = nn.Parameter(torch.randn(8, 8))
        optimizer = torch.optim.AdamW(
            [unrelated_parameter],
            lr=0.01,
            foreach=False,
        )
        mode = GSPMDParallel(
            "fully_shard",
            num_ranks=4,
            num_processes=2,
            fsdp_replicate_param_substrs=("per_layer_",),
            fsdp_shard_optimizer_state_for_replicated_param_substrs=("per_layer_",),
            fsdp_sharded_optimizer_state_keys=("exp_avg",),
        )

        with self.assertRaisesRegex(RuntimeError, "no state owner"):
            mode.shard_params(model, dict(model.named_parameters()), optimizer)

    def test_state_only_sharding_requires_every_configured_state_key(self) -> None:
        model = nn.Module()
        model.register_parameter("per_layer_first", nn.Parameter(torch.randn(8, 8)))
        model.register_parameter("per_layer_second", nn.Parameter(torch.randn(8, 8)))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, foreach=False)
        first = cast(nn.Parameter, model.per_layer_first)
        second = cast(nn.Parameter, model.per_layer_second)
        first_exp_avg = torch.ones(8, 8)
        first_exp_avg_sq = torch.full((8, 8), 2.0)
        optimizer.state[first]["exp_avg"] = first_exp_avg
        optimizer.state[first]["exp_avg_sq"] = first_exp_avg_sq
        optimizer.state[second]["exp_avg"] = torch.zeros(8, 8)
        mode = GSPMDParallel(
            "fully_shard",
            num_ranks=4,
            num_processes=2,
            fsdp_replicate_param_substrs=("per_layer_",),
            fsdp_shard_optimizer_state_for_replicated_param_substrs=("per_layer_",),
            fsdp_sharded_optimizer_state_keys=("exp_avg", "exp_avg_sq"),
        )

        with self.assertRaisesRegex(RuntimeError, "missing required state"):
            mode.shard_params(model, dict(model.named_parameters()), optimizer)
        self.assertIs(first_exp_avg, optimizer.state[first]["exp_avg"])
        self.assertIs(first_exp_avg_sq, optimizer.state[first]["exp_avg_sq"])
        self.assertEqual((8, 8), optimizer.state[first]["exp_avg"].shape)
        self.assertEqual((8, 8), optimizer.state[first]["exp_avg_sq"].shape)
        torch.testing.assert_close(
            torch.ones(8, 8), optimizer.state[first]["exp_avg"], atol=0, rtol=0
        )
        torch.testing.assert_close(
            torch.full((8, 8), 2.0),
            optimizer.state[first]["exp_avg_sq"],
            atol=0,
            rtol=0,
        )

    def test_shard_params_skips_small_tensors(self) -> None:
        dim = 256
        num_ranks = 4
        num_procs = 2
        mode = GSPMDParallel(
            "fully_shard", num_ranks=num_ranks, num_processes=num_procs
        )
        model, opt = self._make_model_and_opt(dim=dim)
        params_and_buffers = {
            **dict(model.named_parameters()),
            **dict(model.named_buffers()),
        }
        mode.shard_params(model, params_and_buffers, opt)
        self.assertEqual(model.bias.shape[0], dim // num_procs)

    def test_shard_params_skips_non_divisible(self) -> None:
        model = nn.Linear(7, 7)
        opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=True)
        x = torch.randn(2, 7)
        model(x).sum().backward()
        opt.step()
        opt.zero_grad()
        mode = GSPMDParallel("fully_shard", num_ranks=4, num_processes=2)
        orig_shape = model.weight.shape
        params_and_buffers = dict(model.named_parameters())
        mode.shard_params(model, params_and_buffers, opt)
        self.assertEqual(model.weight.shape, orig_shape)

    def test_shard_params_shards_optimizer_states(self) -> None:
        dim = 256
        num_procs = 2
        num_ranks = 4
        mode = GSPMDParallel(
            "fully_shard", num_ranks=num_ranks, num_processes=num_procs
        )
        model = nn.Linear(dim, dim)
        opt = torch.optim.Adam(model.parameters(), lr=0.01, foreach=True)
        x = torch.randn(4, dim)
        model(x).sum().backward()
        opt.step()
        opt.zero_grad()
        weight_state = opt.state[model.weight]
        self.assertIn("exp_avg", weight_state)
        orig_exp_avg_shape = weight_state["exp_avg"].shape
        self.assertEqual(orig_exp_avg_shape, (dim, dim))
        params_and_buffers = {
            **dict(model.named_parameters()),
            **dict(model.named_buffers()),
        }
        mode.shard_params(model, params_and_buffers, opt)
        new_state = opt.state[model.weight]  # pyre-ignore[6]
        expected_dim0 = dim // num_procs
        self.assertEqual(new_state["exp_avg"].shape[0], expected_dim0)
        self.assertEqual(new_state["exp_avg_sq"].shape[0], expected_dim0)

    def test_gspmd_metadata_keeps_scalar_state_replicated(self) -> None:
        dim = 16
        mode = GSPMDParallel("fully_shard", num_ranks=4, num_processes=1)

        class ModelWithScalarBuffer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(dim, dim)
                self.register_buffer("embed_scale", torch.tensor(1.0))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x) * self.embed_scale

        model = ModelWithScalarBuffer()
        opt = torch.optim.AdamW(model.parameters(), lr=0.01, foreach=False)
        x = torch.randn(4, dim)
        model(x).sum().backward()
        opt.step()
        opt.zero_grad()

        def train_step(mod: Any, opt: Any, x: torch.Tensor) -> None:
            mod(x).sum().backward()
            with torch.no_grad():
                opt.state[mod.linear.weight]["step"].add_(1.0)

        gm, flat_state = trace(train_step, model, opt, x, parallel_mode=mode)
        compiled = compile_from_gm(
            gm, flat_state, parallel_mode=mode, backend=_IdentityBackend()
        )

        step_tensors = [state["step"] for state in opt.state.values()]
        self.assertTrue(step_tensors)
        state_placeholders = [
            node
            for node in compiled.gm.graph.nodes
            if node.op == "placeholder"
            and node.meta.get("is_state_input", False)
            and isinstance(node.meta.get("val", None), torch.Tensor)
        ]
        tensor_state = [
            value for value in compiled.flat_state if isinstance(value, torch.Tensor)
        ]
        for step_tensor in step_tensors:
            step_index = next(
                index
                for index, state_value in enumerate(tensor_state)
                if state_value is step_tensor
            )
            self.assertFalse(state_placeholders[step_index].meta["is_fsdp_sharded"])

        scalar_flags: list[bool] = []
        sharded_ranked_flags: list[bool] = []
        for node in compiled.gm.graph.nodes:
            if node.op != "placeholder":
                continue
            val = node.meta.get("val", None)
            if not isinstance(val, torch.Tensor):
                continue
            is_fsdp = node.meta.get("is_fsdp_sharded", False)
            if val.ndim == 0:
                scalar_flags.append(is_fsdp)
            elif is_fsdp:
                sharded_ranked_flags.append(is_fsdp)

        self.assertTrue(scalar_flags)
        self.assertFalse(any(scalar_flags))
        self.assertTrue(any(sharded_ranked_flags))

    def test_shard_params_updates_params_and_buffers_dict(self) -> None:
        dim = 256
        mode = GSPMDParallel("fully_shard", num_ranks=4, num_processes=2)
        model = nn.Linear(dim, dim)
        opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=True)
        x = torch.randn(4, dim)
        model(x).sum().backward()
        opt.step()
        opt.zero_grad()
        params_and_buffers = {
            **dict(model.named_parameters()),
            **dict(model.named_buffers()),
        }
        mode.shard_params(model, params_and_buffers, opt)
        self.assertEqual(params_and_buffers["weight"].shape[0], dim // 2)

    def test_shard_params_buffers(self) -> None:
        dim = 256
        num_procs = 2
        mode = GSPMDParallel("fully_shard", num_ranks=4, num_processes=num_procs)

        class ModelWithBuffer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(dim, dim)
                self.register_buffer("running_mean", torch.zeros(dim))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x)

        model = ModelWithBuffer()
        opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=True)
        x = torch.randn(4, dim)
        model(x).sum().backward()
        opt.step()
        opt.zero_grad()
        params_and_buffers = {
            **dict(model.named_parameters()),
            **dict(model.named_buffers()),
        }
        mode.shard_params(model, params_and_buffers, opt)
        # pyrefly: ignore [bad-index]
        self.assertEqual(model.running_mean.shape[0], dim // num_procs)


class TestToShardedJaxFSDP(unittest.TestCase):
    """Test _XLAModule._to_sharded_jax with is_fsdp_tensor=True."""

    def setUp(self) -> None:
        self.devices = jax.devices()
        if len(self.devices) < 2:
            self.skipTest("Need >= 2 JAX devices")
        self.num_local: int = len(jax.local_devices())
        self.num_global: int = jax.device_count()

    def test_fsdp_weight_creates_sharded_array(self) -> None:
        """FSDP weight should produce a globally sharded JAX array."""
        import jax  # @manual  # pyre-ignore[21]
        from jax.sharding import Mesh  # @manual  # pyre-ignore[21]
        from recir.lowering.xla_lower import _XLAModule

        mesh = Mesh(jax.devices(), axis_names=("torch_dist",))
        local_shard = torch.randn(self.num_local, 4)
        jax_array = _XLAModule._to_sharded_jax(local_shard, mesh, is_fsdp_tensor=True)
        expected_global_shape = (self.num_global, 4)
        self.assertEqual(jax_array.shape, expected_global_shape)
        sharding = jax_array.sharding
        self.assertIsNotNone(sharding)

    def test_fsdp_weight_preserves_values(self) -> None:
        """FSDP weight sharding should preserve tensor values in the global array."""
        import jax  # @manual  # pyre-ignore[21]
        import jax.numpy as jnp  # @manual  # pyre-ignore[21]
        from jax.sharding import Mesh  # @manual  # pyre-ignore[21]
        from recir.lowering.xla_lower import _XLAModule

        mesh = Mesh(jax.devices(), axis_names=("torch_dist",))
        local_shard = torch.arange(self.num_local * 2, dtype=torch.float32).reshape(
            self.num_local, 2
        )
        jax_array = _XLAModule._to_sharded_jax(local_shard, mesh, is_fsdp_tensor=True)
        first_shard = jax_array[: self.num_local, :]
        expected = jnp.asarray(local_shard.numpy())
        torch.testing.assert_close(
            torch.tensor(first_shard.__array__()), torch.tensor(expected.__array__())
        )

    def test_fsdp_weight_1d_tensor(self) -> None:
        mesh = Mesh(jax.devices(), axis_names=("torch_dist",))
        local_shard = torch.randn(self.num_local * 4)
        jax_array = _XLAModule._to_sharded_jax(local_shard, mesh, is_fsdp_tensor=True)
        expected_global_dim = self.num_local * 4 * self.num_global // self.num_local
        self.assertEqual(jax_array.shape, (expected_global_dim,))

    def test_fsdp_scalar_tensor_replicates(self) -> None:
        mesh = Mesh(jax.devices(), axis_names=("torch_dist",))
        scalar = torch.tensor(1.0)
        jax_array = _XLAModule._to_sharded_jax(scalar, mesh, is_fsdp_tensor=True)
        self.assertEqual(jax_array.shape, ())
        self.assertEqual(float(jax.device_get(jax_array)), 1.0)

    def test_non_fsdp_weight_replicated(self) -> None:
        """Without is_fsdp_tensor, tensors should be replicated."""
        import jax  # @manual  # pyre-ignore[21]
        from jax.sharding import Mesh  # @manual  # pyre-ignore[21]
        from recir.lowering.xla_lower import _XLAModule

        mesh = Mesh(jax.devices(), axis_names=("torch_dist",))
        t = torch.randn(4, 8)
        jax_array = _XLAModule._to_sharded_jax(t, mesh, is_fsdp_tensor=False)
        self.assertEqual(jax_array.shape, (4, 8))


class TestGSPMDFSDPSingleProcess(unittest.TestCase):
    """End-to-end test: GSPMDParallel('fully_shard') single-process compilation."""

    def setUp(self) -> None:
        import jax  # @manual  # pyre-ignore[21]

        self.devices = jax.devices()
        if len(self.devices) < 2:
            self.skipTest("Need >= 2 JAX devices")
        self.num_ranks: int = len(self.devices)

    def test_gspmd_fully_shard_compiles(self) -> None:
        """GSPMDParallel('fully_shard') should compile and set FSDP metadata."""
        from recir.api import compile
        from recir.backends import XLAGSPMDBackend
        from recir.tests.xla.gspmd_test_utils import DIM, SimpleMLP

        torch.manual_seed(42)
        dim: int = DIM
        if dim % self.num_ranks != 0:
            dim = self.num_ranks * (dim // self.num_ranks + 1)
        model = SimpleMLP(dim=dim).cpu()
        opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
        x = torch.randn(self.num_ranks, dim)

        @compile(
            parallel_mode=GSPMDParallel(
                "fully_shard",
                loss_reduction="sum",
                num_ranks=self.num_ranks,
                num_processes=1,
            ),
            backend=XLAGSPMDBackend(),
        )
        def train_step(mod: Any, opt: Any, x: Any) -> None:
            torch.sum(mod(x)).backward()
            opt.step()

        train_step(model, opt, x)

        compiled_obj = train_step.__dict__["_compiled_obj"]
        self.assertIsNotNone(compiled_obj)
        has_fsdp_meta = any(
            "is_fsdp_sharded" in n.meta
            for n in compiled_obj.gm.graph.nodes
            if n.op == "placeholder"
        )
        self.assertTrue(
            has_fsdp_meta, "Expected is_fsdp_sharded metadata on placeholders"
        )
        xla_modules = [
            module
            for module in compiled_obj.gm.modules()
            if isinstance(module, _XLAModule)
        ]
        self.assertTrue(xla_modules)
        self.assertTrue(
            all(
                module.gm.meta["gspmd_fsdp_axis_size"] == self.num_ranks
                for module in xla_modules
            )
        )

    def test_replicated_parameter_with_sharded_fp32_state_matches_two_updates(
        self,
    ) -> None:
        from recir.api import compile, sync_device_state_to_torch
        from recir.backends import XLAGSPMDBackend

        dim = self.num_ranks * 2

        class StructuredModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                values = torch.linspace(-0.5, 0.5, dim * 2).reshape(dim, 2)
                self.per_layer_weight = nn.Parameter(values.to(torch.bfloat16))

            def logits(self, scale: torch.Tensor) -> torch.Tensor:
                return self.per_layer_weight.float().sum(dim=0) * scale

            def forward(self, scale: torch.Tensor) -> torch.Tensor:
                return self.logits(scale).sum()

        def make_optimizer(model: StructuredModel) -> torch.optim.SGD:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0, foreach=False)
            state = optimizer.state[model.per_layer_weight]
            state["step"] = torch.zeros((), dtype=torch.float32)
            state["fp32_master_param"] = model.per_layer_weight.detach().float().clone()
            state["exp_avg"] = torch.zeros_like(state["fp32_master_param"])
            state["exp_avg_sq"] = torch.zeros_like(state["fp32_master_param"])
            state["grad_accum"] = torch.zeros_like(state["fp32_master_param"])
            return optimizer

        def train_step(
            model: StructuredModel,
            optimizer: torch.optim.SGD,
            scale: torch.Tensor,
        ) -> torch.Tensor:
            loss = model(scale)
            loss.backward()
            state = optimizer.state[model.per_layer_weight]
            with torch.no_grad():
                grad = model.per_layer_weight.grad
                if grad is None:
                    raise RuntimeError("Expected a parameter gradient")
                grad_accum = state["grad_accum"]
                grad_accum.copy_(grad.float())
                state["step"].add_(1.0)
                state["exp_avg"].mul_(0.9).add_(grad_accum, alpha=0.1)
                state["exp_avg_sq"].mul_(0.95).addcmul_(
                    grad_accum,
                    grad_accum,
                    value=0.05,
                )
                state["fp32_master_param"].addcdiv_(
                    state["exp_avg"],
                    state["exp_avg_sq"].sqrt().add_(1e-6),
                    value=-0.01,
                )
                model.per_layer_weight.copy_(
                    state["fp32_master_param"].to(torch.bfloat16)
                )
                grad.zero_()
                grad_accum.zero_()
            return loss.detach()

        torch.manual_seed(41)
        model = StructuredModel()
        reference = StructuredModel()
        reference.load_state_dict(model.state_dict())
        optimizer = make_optimizer(model)
        reference_optimizer = make_optimizer(reference)
        mode = GSPMDParallel(
            "fully_shard",
            loss_reduction="sum",
            num_ranks=self.num_ranks,
            num_processes=1,
            fsdp_replicate_param_substrs=("per_layer_",),
            fsdp_shard_optimizer_state_for_replicated_param_substrs=("per_layer_",),
            fsdp_sharded_optimizer_state_keys=(
                "fp32_master_param",
                "exp_avg",
                "exp_avg_sq",
                "grad_accum",
            ),
        )
        compiled_step = compile(
            parallel_mode=mode,
            backend=XLAGSPMDBackend(dump_hlo=True),
        )(train_step)

        for scale_value in (1.0, 2.0):
            scale = torch.tensor(scale_value)
            # loss_reduction="sum" selects averaged GSPMD gradients.
            train_step(reference, reference_optimizer, scale / self.num_ranks)
            compiled_step(model, optimizer, scale)
        compiled_obj = compiled_step.__dict__["_compiled_obj"]
        device_state = compiled_obj.state_manager.device_state
        self.assertIsNotNone(device_state)
        all_gather_starts = [
            line.strip()
            for module in compiled_obj.gm.modules()
            if isinstance(module, _XLAModule) and module.xla_hlo_text is not None
            for line in module.xla_hlo_text.splitlines()
            if " all-gather-start(" in line
        ]
        self.assertTrue(
            any(f"bf16[{dim},2]" in line for line in all_gather_starts),
            all_gather_starts,
        )
        self.assertFalse(
            any(f"f32[{dim},2]" in line for line in all_gather_starts),
            all_gather_starts,
        )
        sync_device_state_to_torch(compiled_step)

        torch.testing.assert_close(
            model.per_layer_weight,
            reference.per_layer_weight,
            atol=0,
            rtol=0,
        )
        evaluation_scale = torch.tensor(3.0)
        actual_logits = model.logits(evaluation_scale)
        expected_logits = reference.logits(evaluation_scale)
        torch.testing.assert_close(actual_logits, expected_logits, atol=0, rtol=0)
        self.assertTrue(torch.equal(actual_logits.argmax(), expected_logits.argmax()))
        actual_state = optimizer.state[model.per_layer_weight]
        expected_state = reference_optimizer.state[reference.per_layer_weight]
        torch.testing.assert_close(
            actual_state["fp32_master_param"],
            expected_state["fp32_master_param"],
            atol=1e-7,
            rtol=1e-6,
        )
        for state_key in ("exp_avg", "exp_avg_sq", "grad_accum"):
            torch.testing.assert_close(
                actual_state[state_key],
                expected_state[state_key],
                atol=0,
                rtol=0,
            )
        torch.testing.assert_close(
            actual_state["step"],
            expected_state["step"],
            atol=0,
            rtol=0,
        )

        flat_state = compiled_obj.flat_state
        parameter_index = next(
            index
            for index, value in enumerate(flat_state)
            if value is model.per_layer_weight
        )
        master_index = next(
            index
            for index, value in enumerate(flat_state)
            if value is actual_state["fp32_master_param"]
        )
        step_index = next(
            index
            for index, value in enumerate(flat_state)
            if value is actual_state["step"]
        )
        cached_state = cast(list[Any], device_state)
        self.assertEqual(P(), cached_state[parameter_index].sharding.spec)
        self.assertEqual(
            P("torch_dist", None), cached_state[master_index].sharding.spec
        )
        self.assertEqual(P(), cached_state[step_index].sharding.spec)

        master_shards = sorted(
            cached_state[master_index].addressable_shards,
            key=lambda shard: cast(slice, shard.index[0]).start,
        )
        expected_start = 0
        shard_arrays = []
        for shard in master_shards:
            row_slice = cast(slice, shard.index[0])
            self.assertEqual(expected_start, row_slice.start)
            self.assertEqual(dim // self.num_ranks, row_slice.stop - row_slice.start)
            expected_start = row_slice.stop
            shard_arrays.append(np.asarray(jax.device_get(shard.data)))
        self.assertEqual(dim, expected_start)
        reconstructed = np.concatenate(shard_arrays, axis=0)
        self.assertEqual(
            tuple(expected_state["fp32_master_param"].shape), reconstructed.shape
        )
        self.assertEqual(
            hashlib.sha256(
                actual_state["fp32_master_param"].numpy().tobytes()
            ).hexdigest(),
            hashlib.sha256(reconstructed.tobytes()).hexdigest(),
        )

    @parameterized.expand([("sum",), ("mean",)])
    def test_gspmd_fsdp_device_state_numerics(self, loss_reduction: str) -> None:
        """FSDP + GSPMD + XLA device-state manager matches CPU baseline."""
        from recir.api import compile, sync_device_state_to_torch
        from recir.backends import XLAGSPMDBackend
        from recir.tests.xla.gspmd_test_utils import (
            _run_single_rank_baseline,
            DIM,
            LOCAL_BS,
            NUM_ITERS,
            SimpleMLP,
        )

        torch.manual_seed(42)
        dim: int = DIM
        if dim % self.num_ranks != 0:
            dim = self.num_ranks * (dim // self.num_ranks + 1)
        model = SimpleMLP(dim=dim).cpu()
        opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
        x = torch.randn(LOCAL_BS * self.num_ranks, dim)
        baseline_params = _run_single_rank_baseline(
            x, self.num_ranks, loss_reduction=loss_reduction
        )
        reduce_fn: Callable[..., torch.Tensor] = (
            torch.sum if loss_reduction == "sum" else torch.mean
        )

        @compile(
            parallel_mode=GSPMDParallel(
                "fully_shard",
                loss_reduction=loss_reduction,
                num_ranks=self.num_ranks,
                num_processes=1,
            ),
            backend=XLAGSPMDBackend(),
        )
        def train_step(mod: Any, opt: Any, x: Any) -> None:
            reduce_fn(mod(x)).backward()
            opt.step()

        for step in range(NUM_ITERS):
            train_step(model, opt, x)
            sync_device_state_to_torch(train_step)
            for (name, param), (_, baseline_param) in zip(
                model.named_parameters(),
                baseline_params[step].items(),
            ):
                torch.testing.assert_close(
                    param,
                    baseline_param,
                    msg=f"Step {step}: FSDP GSPMD device_state vs baseline "
                    f"mismatch for {name}",
                )

    def test_gspmd_device_cached_state_matches_reference_after_explicit_sync(
        self,
    ) -> None:
        from recir.api import compile, sync_device_state_to_torch
        from recir.backends import XLAGSPMDBackend
        from recir.tests.xla.gspmd_test_utils import (
            _run_single_rank_baseline,
            DIM,
            LOCAL_BS,
            SimpleMLP,
        )

        torch.manual_seed(42)
        dim = DIM
        if dim % self.num_ranks != 0:
            dim = self.num_ranks * (dim // self.num_ranks + 1)
        model = SimpleMLP(dim=dim).cpu()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
        inputs = torch.randn(LOCAL_BS * self.num_ranks, dim)
        baseline_params = _run_single_rank_baseline(
            inputs,
            self.num_ranks,
            num_iters=3,
            loss_reduction="sum",
        )

        @compile(
            parallel_mode=GSPMDParallel(
                "fully_shard",
                loss_reduction="sum",
                num_ranks=self.num_ranks,
                num_processes=1,
            ),
            backend=XLAGSPMDBackend(),
        )
        def train_step(mod: Any, opt: Any, value: Any) -> None:
            torch.sum(mod(value)).backward()
            opt.step()

        train_step(model, optimizer, inputs)
        host_after_first = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, baseline_params[0][name])

        train_step(model, optimizer, inputs)
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(
                parameter, host_after_first[name], atol=0, rtol=0
            )
        sync_device_state_to_torch(train_step)
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, baseline_params[1][name])

        host_after_second = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        train_step(model, optimizer, inputs)
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(
                parameter,
                host_after_second[name],
                atol=0,
                rtol=0,
            )
        sync_device_state_to_torch(train_step)
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, baseline_params[2][name])


if __name__ == "__main__":
    unittest.main()
