#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import shutil
import tempfile
import unittest
from collections.abc import Callable, Sequence
from typing import Any, cast, Dict, TypeVar
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.utils._pytree as pytree
from recir.compiler.process_state import _compute_state_fqns
from recir.distributed.data_parallel import DataParallelStyle, shard_params_and_states
from recir.training.tpu_checkpoint import (
    EMBEDDING_OPTIMIZER_STATE_KEY,
    host_ordinal_from_device_positions,
    HostStateBuffers,
    logical_replay_rank,
    OPTIMIZER_STATE_KEY,
    RecirCheckpointable,
    shard_mesh_order,
    wrap_sharded_for_save,
)
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
from torch.testing._internal.distributed.fake_pg import FakeStore


class _Wrapper(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dense = torch.nn.Parameter(torch.tensor([1.0]))
        self._sharded_tables = torch.nn.ParameterDict(
            {"table": torch.nn.Parameter(torch.tensor([2.0]))}
        )
        self.flushed = False

    def flush_host_state(self, full: bool = False) -> None:
        self.flushed = full


class _ShardedTableWrapper(torch.nn.Module):
    """An in_graph-style module: one meta ``_sharded_tables`` entry, global shape."""

    def __init__(self, vocab: int, dim: int, *, meta: bool = True) -> None:
        super().__init__()
        device = "meta" if meta else "cpu"
        self._sharded_tables = torch.nn.ParameterDict(
            {"table": torch.nn.Parameter(torch.empty(vocab, dim, device=device))}
        )
        self.flushed = False

    def flush_host_state(self, full: bool = False) -> None:
        self.flushed = full


class _Warmup:
    def __init__(self, iteration: int = 3) -> None:
        self.iter = iteration

    def set_iter(self, iteration: int) -> None:
        self.iter = iteration

    def advance(self) -> int:
        self.iter += 1
        return self.iter


class _DenseModel(torch.nn.Module):
    running_scale: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.5, -1.0, 1.5, -2.0]))
        self.register_buffer("running_scale", torch.tensor(1.0))
        self.flushed = False

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return (self.weight * batch).sum().square()

    def flush_host_state(self, full: bool = False) -> None:
        self.flushed = full


def _build_buffers(
    flat_state: list[torch.Tensor],
    state_fqns: list[str],
    model: torch.nn.Module,
    optimizers: Sequence[object] = (),
    prefixes: tuple[str, ...] = ("_sharded_tables",),
) -> HostStateBuffers:
    buffers = HostStateBuffers(flat_state, state_fqns, model, optimizers, prefixes)
    buffers.build()
    return buffers


_TreeT = TypeVar("_TreeT")


def _clone_tree(value: _TreeT) -> _TreeT:
    return pytree.tree_map(
        lambda leaf: leaf.detach().clone() if isinstance(leaf, torch.Tensor) else leaf,
        value,
    )


def _new_dense_leg() -> tuple[_DenseModel, torch.optim.AdamW]:
    model = _DenseModel()
    # Citrine C2: exercise AdamW's multi-tensor optimizer path.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.03,
        betas=(0.8, 0.9),
        weight_decay=0.01,
        foreach=True,
    )
    return model, optimizer


def _run_dense_step(
    model: _DenseModel,
    optimizer: torch.optim.Optimizer,
    batch: torch.Tensor,
) -> torch.Tensor:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    optimizer.step()
    return loss.detach().clone()


def _optimizer_flat_state(
    model: _DenseModel,
    optimizer: torch.optim.AdamW,
) -> tuple[list[torch.Tensor], list[str]]:
    params_buffers = dict(model.named_parameters())
    named_states = {
        name: optimizer.state[param]
        for name, param in model.named_parameters()
        if param in optimizer.state
    }
    structure = [params_buffers, named_states]
    return pytree.tree_leaves(structure), _compute_state_fqns(structure)


def _run_dense_step_from_flat_state(
    model: _DenseModel,
    optimizer: torch.optim.AdamW,
    flat_state: list[torch.Tensor],
    state_fqns: list[str],
    batch: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    state = optimizer.state[model.weight]
    for key, value in list(state.items()):
        if isinstance(value, torch.Tensor):
            state[key] = flat_state[state_fqns.index(f"[1]['weight']['{key}']")]
    loss = _run_dense_step(model, optimizer, batch)
    return (
        loss,
        model.weight.detach().clone(),
        {key: value.detach().clone() for key, value in state.items()},
    )


class RecirCheckpointableTest(unittest.TestCase):
    def test_async_snapshot_aliases_live_storage_instead_of_copying(self) -> None:
        wrapper = _Wrapper()
        checkpointable = RecirCheckpointable(
            wrapper,
            num_processes=1,
            sharded_prefixes=("_sharded_tables",),
            table_prefixes=("_sharded_tables",),
            warmup=_Warmup(),
        )

        state = checkpointable.state_dict()
        dense = state["dense"]
        table = state["_sharded_tables.table"]

        self.assertTrue(wrapper.flushed)
        # Every entry is now a view of the live tensor, dense included. The
        # tables always took this branch; dense used to be cloned.
        self.assertEqual(wrapper.dense.data_ptr(), dense.data_ptr())
        self.assertEqual(wrapper._sharded_tables["table"].data_ptr(), table.data_ptr())
        # So a write to the module IS visible in a snapshot already handed to
        # DCP. That is safe only because nothing writes these entries between
        # two checkpoints -- the periodic host sync is narrowed to the LR
        # buffers (model_family.sync_lr_to_host). This asserts the aliasing
        # rather than the safety: if a writer comes back, it is that narrowing
        # that has to be revisited, not this snapshot.
        with torch.no_grad():
            wrapper.dense.fill_(4.0)
        torch.testing.assert_close(torch.tensor([4.0]), dense)
        self.assertIs(state, checkpointable.make_async_stager().stage(state))
        self.assertNotIn(EMBEDDING_OPTIMIZER_STATE_KEY, state)

    def test_dense_weight_roundtrip_preserves_the_next_update(self) -> None:
        reference_model = _DenseModel()
        # Citrine C2: use the multi-tensor optimizer path.
        reference_optimizer = torch.optim.SGD(
            reference_model.parameters(), lr=0.02, foreach=True
        )
        for offset in range(4):
            _run_dense_step(
                reference_model,
                reference_optimizer,
                torch.arange(1.0, 5.0) + offset,
            )
        checkpointable = RecirCheckpointable(
            reference_model,
            num_processes=1,
            sharded_prefixes=(),
            table_prefixes=(),
            warmup=_Warmup(),
        )
        # state_dict() hands out views now, so take the bytes here: this stands
        # in for DCP having uploaded them at save time. Stepping the reference
        # below would otherwise rewrite the "saved" values underneath us -- in a
        # real run nothing writes these entries between two checkpoints, which
        # is what the narrowed host sync guarantees.
        checkpoint = cast(Dict[str, Any], _clone_tree(checkpointable.state_dict()))
        expected_before = reference_model.weight.detach().clone()
        next_batch = torch.tensor([2.0, -1.0, 3.0, 0.5])
        expected_loss = _run_dense_step(
            reference_model, reference_optimizer, next_batch
        )
        expected_delta = reference_model.weight.detach() - expected_before

        restored_model = _DenseModel()
        # Citrine C2: use the multi-tensor optimizer path.
        restored_optimizer = torch.optim.SGD(
            restored_model.parameters(), lr=0.02, foreach=True
        )
        parameter = restored_model.weight
        with torch.no_grad():
            parameter.fill_(99.0)
        RecirCheckpointable(
            restored_model,
            num_processes=1,
            sharded_prefixes=(),
            table_prefixes=(),
            warmup=_Warmup(),
        ).load_state_dict(checkpoint)

        self.assertIs(restored_model.weight, parameter)
        torch.testing.assert_close(restored_model.weight, expected_before)
        restored_before = restored_model.weight.detach().clone()
        restored_loss = _run_dense_step(restored_model, restored_optimizer, next_batch)
        restored_delta = restored_model.weight.detach() - restored_before
        torch.testing.assert_close(restored_loss, expected_loss)
        torch.testing.assert_close(restored_delta, expected_delta)

    def test_buffer_and_warmup_roundtrip_preserves_the_next_step(self) -> None:
        source = _DenseModel()
        source_warmup = _Warmup(iteration=17)
        with torch.no_grad():
            source.running_scale.fill_(2.5)
        checkpoint = RecirCheckpointable(
            source,
            num_processes=1,
            sharded_prefixes=(),
            table_prefixes=(),
            warmup=source_warmup,
        ).state_dict()

        next_batch = torch.tensor([2.0, -1.0, 3.0, 0.5])
        expected_output = source(next_batch) * source.running_scale
        expected_iteration = source_warmup.advance()

        restored = _DenseModel()
        restored_warmup = _Warmup(iteration=99)
        buffer = restored.running_scale
        with torch.no_grad():
            restored.running_scale.fill_(-4.0)
        RecirCheckpointable(
            restored,
            num_processes=1,
            sharded_prefixes=(),
            table_prefixes=(),
            warmup=restored_warmup,
        ).load_state_dict(checkpoint)

        self.assertIs(restored.running_scale, buffer)
        torch.testing.assert_close(restored.running_scale, torch.tensor(2.5))
        torch.testing.assert_close(
            restored(next_batch) * restored.running_scale,
            expected_output,
        )
        self.assertEqual(restored_warmup.advance(), expected_iteration)

    def test_rebinding_dense_optimizer_state_preserves_the_next_update(self) -> None:
        reference_model, reference_optimizer = _new_dense_leg()
        for offset in range(4):
            _run_dense_step(
                reference_model,
                reference_optimizer,
                torch.arange(1.0, 5.0) + offset,
            )
        saved_model = _clone_tree(reference_model.state_dict())
        saved_optimizer = _clone_tree(reference_optimizer.state_dict())
        next_batch = torch.tensor([2.0, -1.0, 3.0, 0.5])
        expected = _run_dense_step_from_flat_state(
            reference_model,
            reference_optimizer,
            *_optimizer_flat_state(reference_model, reference_optimizer),
            next_batch,
        )

        results = {}
        for aliasing in (False, True):
            model, optimizer = _new_dense_leg()
            _run_dense_step(model, optimizer, torch.tensor([-4.0, 3.0, -2.0, 1.0]))
            flat_state, state_fqns = _optimizer_flat_state(model, optimizer)
            buffers = _build_buffers(flat_state, state_fqns, model, [optimizer])
            model.load_state_dict(saved_model)
            # An in-place load leaves the optimizer's slots aliasing the buffers
            # the graph reads, which is what dcp.load does once
            # `_freeze_checkpoint_value` stopped cloning. A load of fresh
            # tensors does not, and then flat_state points at storage nobody
            # writes any more.
            incoming = optimizer.state_dict()
            pytree.tree_map_(
                lambda live, saved: (
                    live.copy_(saved) if isinstance(live, torch.Tensor) else None
                ),
                incoming,
                saved_optimizer,
            )
            optimizer.load_state_dict(
                _clone_tree(incoming) if not aliasing else incoming
            )
            # The payload is what a real load would have written into. Aliasing
            # keeps it on the buffer; a cloned load moves it off, and then the
            # optimizer updates a tensor the graph never reads.
            payload = [(optimizer, optimizer.state_dict())]
            if aliasing:
                buffers.check_payload({}, payload)
            else:
                with self.assertRaisesRegex(RuntimeError, "is not the host buffer"):
                    buffers.check_payload({}, payload)
            results[aliasing] = _run_dense_step_from_flat_state(
                model, optimizer, flat_state, state_fqns, next_batch
            )

        # The aliasing leg is the one that has to reproduce the reference step:
        # its slots are still the buffers the graph reads, so the restored
        # moments are the ones the update consumes.
        torch.testing.assert_close(results[True][1], expected[1])
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(results[True][2][key], expected[2][key])
        # The non-aliasing leg is not asserted to diverge here.
        # `_run_dense_step_from_flat_state` re-points the optimizer at
        # `flat_state` before stepping, so both legs consume the same buffers
        # and land on the same answer. What that leg demonstrates is the
        # check_payload rejection above -- in a real run nothing re-points the
        # optimizer back, and the cloned slots would silently drift.
        torch.testing.assert_close(results[False][0], expected[0])

    def test_embedding_optimizer_state_round_trips_and_preserves_lr(self) -> None:
        wrapper = _ShardedTableWrapper(4, 2, meta=False)
        wrapper.register_parameter("dense", torch.nn.Parameter(torch.tensor([1.0])))
        dense = wrapper.get_parameter("dense")
        dense_lr = torch.tensor(0.01)
        embedding_lr = torch.tensor(0.02)
        # Citrine C2: use the multi-tensor optimizer path.
        dense_optimizer = torch.optim.SGD([dense], lr=dense_lr, foreach=True)
        # Citrine C2: use the multi-tensor optimizer path.
        embedding_optimizer = torch.optim.SGD(
            [wrapper._sharded_tables["table"]], lr=embedding_lr, foreach=True
        )
        param = wrapper._sharded_tables["table"]
        dense_optimizer.state[dense] = {"momentum": torch.tensor([3.0])}
        embedding_optimizer.state[param] = {"sum": torch.full((param.shape[0], 1), 7.0)}
        checkpointable = RecirCheckpointable(
            wrapper,
            num_processes=1,
            sharded_prefixes=("_sharded_tables",),
            table_prefixes=("_sharded_tables",),
            warmup=_Warmup(),
            dense_optimizer=dense_optimizer,
            embedding_optimizer=embedding_optimizer,
        )

        # state_dict() hands out views of the live optimizer state, so take
        # the bytes here: this stands in for DCP having uploaded them at save
        # time, which is what the assertions below model.
        state = cast(Dict[str, Any], _clone_tree(checkpointable.state_dict()))
        self.assertIn(OPTIMIZER_STATE_KEY, state)
        self.assertIn(EMBEDDING_OPTIMIZER_STATE_KEY, state)
        saved_embedding_state = next(
            iter(state[EMBEDDING_OPTIMIZER_STATE_KEY]["state"].values())
        )["sum"]
        torch.testing.assert_close(saved_embedding_state, torch.full((4, 1), 7.0))

        embedding_optimizer.state[param]["sum"].zero_()
        dense_optimizer.param_groups[0]["lr"] = torch.tensor(1.0)
        embedding_optimizer.param_groups[0]["lr"] = torch.tensor(2.0)
        live_dense_lr = dense_optimizer.param_groups[0]["lr"]
        live_embedding_lr = embedding_optimizer.param_groups[0]["lr"]
        # The asynchronous snapshot must not follow later optimizer mutations.
        torch.testing.assert_close(saved_embedding_state, torch.full((4, 1), 7.0))

        checkpointable.load_state_dict(state)

        torch.testing.assert_close(
            embedding_optimizer.state[param]["sum"], torch.full((4, 1), 7.0)
        )
        self.assertIs(dense_optimizer.param_groups[0]["lr"], live_dense_lr)
        self.assertIs(embedding_optimizer.param_groups[0]["lr"], live_embedding_lr)


class ShardedTableRestoreBindingTest(unittest.TestCase):
    """A restore must land in the tensor the compiled graph actually reads."""

    _WORLD_SIZE = 2
    _VOCAB = 8
    _DIM = 4
    _FQN = "_sharded_tables.table"

    def setUp(self) -> None:
        # A fake PG suffices: Shard(0) partitioning is local shape math
        # (partition_value -> _split_tensor); no collective is issued.
        dist.init_process_group(
            backend="fake",
            store=FakeStore(),
            rank=0,
            world_size=self._WORLD_SIZE,
        )
        self.addCleanup(dist.destroy_process_group)

    def _shard_and_compile_state(
        self, wrapper: torch.nn.Module
    ) -> tuple[list[torch.Tensor], list[str]]:
        """Row-shard the table as recir's compile does; return (flat_state, fqns)."""
        params_buffers: dict[str, torch.Tensor] = dict(wrapper.named_parameters())
        shard_params_and_states(
            wrapper,
            params_buffers,
            None,
            DeviceMesh("cpu", torch.arange(self._WORLD_SIZE)),
            DataParallelStyle.FULLY_SHARD,
        )
        named_states: dict[str, object] = {}
        structure = [params_buffers, named_states]
        return pytree.tree_leaves(structure), _compute_state_fqns(structure)

    def test_sparse_weight_rebind_preserves_the_next_update(self) -> None:
        source = _ShardedTableWrapper(self._VOCAB, self._DIM, meta=False)
        with torch.no_grad():
            source._sharded_tables["table"].copy_(
                torch.arange(self._VOCAB * self._DIM, dtype=torch.float32).reshape(
                    self._VOCAB, self._DIM
                )
                / 10.0
            )
        source_flat_state, source_fqns = self._shard_and_compile_state(source)
        _build_buffers(source_flat_state, source_fqns, source)
        checkpoint = RecirCheckpointable(
            source,
            num_processes=self._WORLD_SIZE,
            sharded_prefixes=("_sharded_tables",),
            table_prefixes=("_sharded_tables",),
            warmup=_Warmup(),
            shard_order=list(range(self._WORLD_SIZE)),
        ).state_dict()
        checkpoint[self._FQN] = checkpoint[self._FQN].detach().clone()
        table_index = source_fqns.index(f"[0]['{self._FQN}']")
        row_ids = torch.tensor([0, 2], dtype=torch.int64)
        expected = self._run_sparse_weight_step(source_flat_state[table_index], row_ids)

        results = {}
        for bound in (False, True):
            restored = _ShardedTableWrapper(self._VOCAB, self._DIM)
            flat_state, state_fqns = self._shard_and_compile_state(restored)
            restored_index = state_fqns.index(f"[0]['{self._FQN}']")
            flat_state[restored_index] = torch.full(
                (self._VOCAB // self._WORLD_SIZE, self._DIM), -9.0
            )
            if bound:
                _build_buffers(flat_state, state_fqns, restored)
            RecirCheckpointable(
                restored,
                num_processes=self._WORLD_SIZE,
                sharded_prefixes=("_sharded_tables",),
                table_prefixes=("_sharded_tables",),
                warmup=_Warmup(),
                shard_order=list(range(self._WORLD_SIZE)),
            ).load_state_dict(checkpoint)
            results[bound] = self._run_sparse_weight_step(
                flat_state[restored_index], row_ids
            )

        self.assertFalse(torch.equal(results[False][0], expected[0]))
        self.assertNotEqual(results[False][1], expected[1])
        torch.testing.assert_close(results[True][0], expected[0])
        torch.testing.assert_close(results[True][1], expected[1])
        torch.testing.assert_close(results[True][2], expected[2])

    def test_build_is_idempotent_and_keeps_the_same_storage(self) -> None:
        wrapper = _ShardedTableWrapper(self._VOCAB, self._DIM)
        flat_state, state_fqns = self._shard_and_compile_state(wrapper)
        buffers = _build_buffers(flat_state, state_fqns, wrapper)
        first = wrapper._sharded_tables["table"].data_ptr()

        buffers.build()

        self.assertEqual(wrapper._sharded_tables["table"].data_ptr(), first)
        self.assertEqual(flat_state[0].data_ptr(), first)
        buffers.check_payload({self._FQN: wrapper._sharded_tables["table"]})

    def test_a_payload_entry_that_is_not_the_buffer_is_rejected(self) -> None:
        # What a save uploads and what a load writes into is the payload, so
        # that is what has to be the buffer. A same-shape stand-in passes every
        # structural check and is exactly what would go unnoticed.
        wrapper = _ShardedTableWrapper(self._VOCAB, self._DIM)
        flat_state, state_fqns = self._shard_and_compile_state(wrapper)
        buffers = _build_buffers(flat_state, state_fqns, wrapper)
        live = wrapper._sharded_tables["table"]

        buffers.check_payload({self._FQN: live})

        with self.assertRaisesRegex(RuntimeError, "is not the host buffer"):
            buffers.check_payload({self._FQN: torch.zeros_like(live)})

    def _run_sparse_weight_step(
        self,
        table: torch.Tensor,
        row_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        working = table.detach().clone().requires_grad_(True)
        prediction = working.index_select(0, row_ids).sum(dim=1)
        loss = prediction.square().sum()
        (gradient,) = torch.autograd.grad(loss, working)
        with torch.no_grad():
            table.add_(gradient, alpha=-0.05)
        return prediction.detach(), loss.detach(), table.detach().clone()

    def _rowwise_optimizer(
        self,
        wrapper: _ShardedTableWrapper,
        *,
        accumulator: float = 0.25,
    ) -> torch.optim.SGD:
        param = wrapper._sharded_tables["table"]
        # Citrine C2: use the multi-tensor optimizer path. The explicit rowwise
        # sum mirrors TensorLRRowWiseAdagrad's checkpoint state.
        optimizer = torch.optim.SGD([param], lr=0.05, foreach=True)
        optimizer.state[param] = {"sum": torch.full((param.shape[0], 1), accumulator)}
        return optimizer

    def _shard_with_optimizer_state(
        self,
        wrapper: _ShardedTableWrapper,
        optimizer: torch.optim.Optimizer,
    ) -> tuple[list[torch.Tensor], list[str]]:
        params_buffers: dict[str, torch.Tensor] = dict(wrapper.named_parameters())
        shard_params_and_states(
            wrapper,
            params_buffers,
            optimizer,
            DeviceMesh("cpu", torch.arange(self._WORLD_SIZE)),
            DataParallelStyle.FULLY_SHARD,
        )
        param = wrapper._sharded_tables["table"]
        structure = [params_buffers, {self._FQN: optimizer.state[param]}]
        return pytree.tree_leaves(structure), _compute_state_fqns(structure)

    def _run_rowwise_step(
        self,
        table: torch.Tensor,
        accumulator: torch.Tensor,
        row_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        working = table.detach().clone().requires_grad_(True)
        prediction = working.index_select(0, row_ids).sum(dim=1)
        loss = prediction.square().sum()
        (gradient,) = torch.autograd.grad(loss, working)
        with torch.no_grad():
            accumulator.add_(gradient.square().mean(dim=1, keepdim=True))
            table.add_(gradient / (accumulator.sqrt() + 1e-8), alpha=-0.05)
        return (
            prediction.detach(),
            loss.detach(),
            table.detach().clone(),
            accumulator.detach().clone(),
        )

    def test_restore_reaches_the_flat_state_leaf(self) -> None:
        wrapper = _ShardedTableWrapper(self._VOCAB, self._DIM)
        flat_state, state_fqns = self._shard_and_compile_state(wrapper)
        self.assertEqual(state_fqns, [f"[0]['{self._FQN}']"])
        local_rows = self._VOCAB // self._WORLD_SIZE
        self.assertEqual(tuple(flat_state[0].shape), (local_rows, self._DIM))

        _build_buffers(flat_state, state_fqns, wrapper)

        # dcp.load writes into the module state_dict view, i.e. the Parameter.
        restored = torch.arange(local_rows * self._DIM, dtype=torch.float32).reshape(
            local_rows, self._DIM
        )
        param = wrapper._sharded_tables["table"]
        self.assertFalse(param.is_meta, "HostStateBuffers.build left the param meta")
        with torch.no_grad():
            param.copy_(restored)

        leaf = flat_state[0]
        self.assertFalse(
            leaf.is_meta,
            "flat_state leaf is still meta after restore: the compiled graph "
            "would re-randomize the table instead of reading the checkpoint",
        )
        torch.testing.assert_close(leaf, restored)
        # Same storage, so the Phase 1 release frees the mirror for both refs.
        self.assertEqual(leaf.data_ptr(), param.data_ptr())

    def test_restore_reaches_the_optimizer_state_leaf(self) -> None:
        """The tables' optimizer state has the same split binding."""
        wrapper = _ShardedTableWrapper(self._VOCAB, self._DIM)
        params_buffers: dict[str, torch.Tensor] = dict(wrapper.named_parameters())
        # Adagrad seeds state["sum"] in __init__, so off a meta table the slot
        # is born meta -- the production starting point.
        # Citrine C2: use the multi-tensor optimizer path.
        optimizer = torch.optim.Adagrad(
            list(wrapper.parameters()), lr=0.1, foreach=True
        )
        shard_params_and_states(
            wrapper,
            params_buffers,
            optimizer,
            DeviceMesh("cpu", torch.arange(self._WORLD_SIZE)),
            DataParallelStyle.FULLY_SHARD,
        )
        param = wrapper._sharded_tables["table"]
        local_rows = self._VOCAB // self._WORLD_SIZE

        named_states = {self._FQN: optimizer.state[param]}
        structure = [params_buffers, named_states]
        flat_state = pytree.tree_leaves(structure)
        state_fqns = _compute_state_fqns(structure)
        state_index = state_fqns.index(f"[1]['{self._FQN}']['sum']")
        self.assertTrue(flat_state[state_index].is_meta)

        # What the restore has to end up doing for the parameter AND its state:
        # give both real host storage before dcp.load can write into them.
        buffers = _build_buffers(flat_state, state_fqns, wrapper, [optimizer])

        restored = torch.full((local_rows, self._DIM), 7.0)
        slot = optimizer.state[param]["sum"]
        self.assertFalse(slot.is_meta, "optimizer slot was never materialized")
        restored_state = optimizer.state_dict()
        # In place, the way dcp.load writes: it fills the tensor state_dict()
        # handed it. Substituting a fresh tensor here would model a load that
        # never reaches the graph -- which check_payload now rejects outright.
        next(iter(restored_state["state"].values()))["sum"].copy_(restored)
        optimizer.load_state_dict(restored_state)

        buffers.check_payload({}, [(optimizer, optimizer.state_dict())])

        leaf = flat_state[state_index]
        self.assertFalse(
            leaf.is_meta,
            "optimizer-state leaf is still meta: seeding device_state would "
            "regenerate it instead of using the restored momentum",
        )
        torch.testing.assert_close(leaf, restored)

    def test_embedding_optimizer_state_is_row_sharded_for_dcp(self) -> None:
        wrapper = _ShardedTableWrapper(self._VOCAB, self._DIM, meta=False)
        params_buffers: dict[str, torch.Tensor] = dict(wrapper.named_parameters())
        shard_params_and_states(
            wrapper,
            params_buffers,
            None,
            DeviceMesh("cpu", torch.arange(self._WORLD_SIZE)),
            DataParallelStyle.FULLY_SHARD,
        )
        param = wrapper._sharded_tables["table"]
        local_rows = self._VOCAB // self._WORLD_SIZE
        embedding_lr = torch.tensor(0.02)
        # Citrine C2: use the multi-tensor optimizer path.
        embedding_optimizer = torch.optim.SGD([param], lr=embedding_lr, foreach=True)
        embedding_optimizer.state[param] = {
            "sum": torch.arange(local_rows, dtype=torch.float32).view(-1, 1)
        }
        checkpointable = RecirCheckpointable(
            wrapper,
            num_processes=self._WORLD_SIZE,
            sharded_prefixes=("_sharded_tables",),
            table_prefixes=("_sharded_tables",),
            warmup=_Warmup(),
            shard_order=list(range(self._WORLD_SIZE)),
            embedding_optimizer=embedding_optimizer,
        )

        state = checkpointable.state_dict()[EMBEDDING_OPTIMIZER_STATE_KEY]
        saved_sum = next(iter(state["state"].values()))["sum"]

        self.assertIsInstance(saved_sum, DTensor)
        self.assertEqual(tuple(saved_sum.shape), (self._VOCAB, 1))
        self.assertEqual(tuple(saved_sum.to_local().shape), (local_rows, 1))
        self.assertEqual(saved_sum.placements, (Shard(0),))
        self.assertNotIsInstance(state["param_groups"][0]["lr"], DTensor)

    def test_sparse_optimizer_rebind_preserves_the_next_update(self) -> None:
        source = _ShardedTableWrapper(self._VOCAB, self._DIM, meta=False)
        with torch.no_grad():
            source._sharded_tables["table"].copy_(
                torch.arange(self._VOCAB * self._DIM, dtype=torch.float32).reshape(
                    self._VOCAB, self._DIM
                )
                / 10.0
            )
        source_optimizer = self._rowwise_optimizer(source)
        source_flat_state, source_fqns = self._shard_with_optimizer_state(
            source, source_optimizer
        )
        _build_buffers(source_flat_state, source_fqns, source, [source_optimizer])
        checkpoint = RecirCheckpointable(
            source,
            num_processes=self._WORLD_SIZE,
            sharded_prefixes=("_sharded_tables",),
            table_prefixes=("_sharded_tables",),
            warmup=_Warmup(),
            shard_order=list(range(self._WORLD_SIZE)),
            embedding_optimizer=source_optimizer,
        ).state_dict()
        # state_dict() hands out views of the live buffers, so take the bytes:
        # this stands in for DCP having uploaded them at save time.
        checkpoint = cast(Dict[str, Any], _clone_tree(checkpoint))
        table_index = source_fqns.index(f"[0]['{self._FQN}']")
        sum_index = source_fqns.index(f"[1]['{self._FQN}']['sum']")
        row_ids = torch.tensor([0, 2], dtype=torch.int64)
        expected = self._run_rowwise_step(
            source_flat_state[table_index], source_flat_state[sum_index], row_ids
        )

        results = {}
        for bound in (False, True):
            restored = _ShardedTableWrapper(self._VOCAB, self._DIM, meta=False)
            with torch.no_grad():
                restored._sharded_tables["table"].fill_(-9.0)
            optimizer = self._rowwise_optimizer(restored, accumulator=7.0)
            flat_state, state_fqns = self._shard_with_optimizer_state(
                restored, optimizer
            )
            restored_table_index = state_fqns.index(f"[0]['{self._FQN}']")
            restored_sum_index = state_fqns.index(f"[1]['{self._FQN}']['sum']")
            # Fresh storage for the accumulator the graph reads, holding the
            # same pre-restore value. Only build() rebinds it to what the load
            # writes into; unbound, flat_state points at storage nobody writes,
            # which is what separates the table restore from the optimizer one.
            flat_state[restored_sum_index] = torch.full_like(
                flat_state[restored_sum_index], 7.0
            )
            if bound:
                _build_buffers(flat_state, state_fqns, restored, [optimizer])
            target = RecirCheckpointable(
                restored,
                num_processes=self._WORLD_SIZE,
                sharded_prefixes=("_sharded_tables",),
                table_prefixes=("_sharded_tables",),
                warmup=_Warmup(),
                shard_order=list(range(self._WORLD_SIZE)),
                embedding_optimizer=optimizer,
            )
            # dcp.load writes the checkpoint bytes into the tensors state_dict()
            # hands it, in place. Handing the source's payload straight to
            # load_state_dict instead would rebind the optimizer to the source's
            # tensors and leave the accumulator the graph reads untouched.
            incoming = target.state_dict()
            pytree.tree_map_(
                lambda live, saved: (
                    live.copy_(saved) if isinstance(live, torch.Tensor) else None
                ),
                incoming,
                checkpoint,
            )
            target.load_state_dict(incoming)

            results[bound] = self._run_rowwise_step(
                flat_state[restored_table_index],
                flat_state[restored_sum_index],
                row_ids,
            )

        torch.testing.assert_close(results[False][0], expected[0])
        torch.testing.assert_close(results[False][1], expected[1])
        self.assertFalse(torch.equal(results[False][2], expected[2]))
        self.assertFalse(torch.equal(results[False][3], expected[3]))
        torch.testing.assert_close(results[True][2], expected[2])
        torch.testing.assert_close(results[True][3], expected[3])


class HostOrdinalFromDevicePositionsTest(unittest.TestCase):
    """Which FSDP-axis slot a process owns, derived from its devices."""

    def test_second_host_of_four_owns_slot_one(self) -> None:
        self.assertEqual(
            host_ordinal_from_device_positions([4, 5, 6, 7], 16, 4),
            1,
        )

    def test_positions_may_arrive_unordered(self) -> None:
        self.assertEqual(
            host_ordinal_from_device_positions([10, 8, 11, 9], 16, 4),
            2,
        )

    def test_scattered_devices_are_rejected(self) -> None:
        # A Shard(0) checkpoint can only describe one contiguous row block per
        # host, so a host whose devices straddle the axis has no valid ordinal.
        with self.assertRaisesRegex(RuntimeError, "not a contiguous block"):
            host_ordinal_from_device_positions([0, 1, 2, 8], 16, 4)

    def test_unaligned_block_start_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "not a multiple"):
            host_ordinal_from_device_positions([2, 3, 4, 5], 16, 4)

    def test_wrong_local_device_count_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "expected 4 local devices"):
            host_ordinal_from_device_positions([0, 1], 16, 4)


class LogicalReplayRankTest(unittest.TestCase):
    @patch(
        "recir.training.tpu_checkpoint.fsdp_host_ordinal",
        return_value=7,
    )
    def test_multi_host_tpu_uses_stable_fsdp_ordinal(self, ordinal: MagicMock) -> None:
        self.assertEqual(logical_replay_rank(0, 16, ["tpu"] * 64), 7)
        ordinal.assert_called_once_with(16)

    @patch("recir.training.tpu_checkpoint.fsdp_host_ordinal")
    def test_multi_device_cpu_keeps_gloo_rank(self, ordinal: MagicMock) -> None:
        self.assertEqual(logical_replay_rank(3, 4, ["cpu"] * 4), 3)
        ordinal.assert_not_called()

    @patch("recir.training.tpu_checkpoint.fsdp_host_ordinal")
    def test_single_host_tpu_keeps_gloo_rank(self, ordinal: MagicMock) -> None:
        self.assertEqual(logical_replay_rank(0, 1, ["tpu"] * 4), 0)
        ordinal.assert_not_called()

    def test_rank_remapping_keeps_stream_with_logical_ordinal(self) -> None:
        platforms = ["tpu"] * 64
        with patch(
            "recir.training.tpu_checkpoint.fsdp_host_ordinal",
            return_value=2,
        ):
            source_stream = logical_replay_rank(7, 16, platforms)
            resumed_stream = logical_replay_rank(11, 16, platforms)
        self.assertEqual(source_stream, resumed_stream)
        self.assertEqual(resumed_stream, 2)


class ShardMeshOrderTest(unittest.TestCase):
    def test_identity_mapping_round_trips(self) -> None:
        pairs = [(r, r) for r in range(4)]
        self.assertEqual(shard_mesh_order(pairs, 4), [0, 1, 2, 3])

    def test_permutation_is_inverted_so_slot_j_holds_the_rank_at_ordinal_j(
        self,
    ) -> None:
        # pod0 -> JAX 2, pod1 -> JAX 0, pod2 -> JAX 3, pod3 -> JAX 1
        pairs = [(0, 2), (1, 0), (2, 3), (3, 1)]
        order = shard_mesh_order(pairs, 4)
        self.assertEqual(order, [1, 3, 0, 2])
        # Each rank's mesh coordinate is its FSDP ordinal, which is the whole point.
        for rank, ordinal in pairs:
            self.assertEqual(order.index(rank), ordinal)

    def test_two_ranks_claiming_one_ordinal_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "not a bijection"):
            shard_mesh_order([(0, 1), (1, 1), (2, 2), (3, 3)], 4)

    def test_unclaimed_ordinal_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, r"ordinal\(s\) \[3\]"):
            shard_mesh_order([(0, 0), (1, 1), (2, 2)], 4)


def _round_trip_blocks(
    save_pairs: list[tuple[int, int]],
    load_pairs: list[tuple[int, int]],
    num_processes: int,
    slot_of_rank: Callable[[Sequence[tuple[int, int]], int], list[int]],
) -> dict[int, int]:
    """``jax block after restore -> jax block before save``, for one table."""
    save_slots = slot_of_rank(save_pairs, num_processes)
    load_slots = slot_of_rank(load_pairs, num_processes)
    save_ordinal = dict(save_pairs)
    load_ordinal = dict(load_pairs)
    checkpoint = {
        block: save_ordinal[save_slots[block]] for block in range(num_processes)
    }
    return {
        load_ordinal[load_slots[block]]: checkpoint[block]
        for block in range(num_processes)
    }


class ShardNumberingRoundTripTest(unittest.TestCase):
    """A restore must return every row block to the ID range it came from."""

    _P = 4
    # Two jobs, two different pod -> JAX assignments.
    _SAVE = [(0, 2), (1, 0), (2, 3), (3, 1)]
    _LOAD = [(0, 0), (1, 1), (2, 2), (3, 3)]

    def test_ordinal_numbering_survives_a_reassignment(self) -> None:
        restored = _round_trip_blocks(self._SAVE, self._LOAD, self._P, shard_mesh_order)
        self.assertEqual(restored, {block: block for block in range(self._P)})

    def test_rank_numbering_permutes_the_table(self) -> None:
        # torch.arange(P): a rank's mesh coordinate is its rank.
        def by_rank(pairs: Sequence[tuple[int, int]], n: int) -> list[int]:
            return list(range(n))

        restored = _round_trip_blocks(self._SAVE, self._LOAD, self._P, by_rank)
        self.assertNotEqual(restored, {block: block for block in range(self._P)})
        # Every block lands on the wrong ID range: block 0 comes back holding
        # what used to be block 2, and so on.
        self.assertEqual(restored, {0: 2, 1: 0, 2: 3, 3: 1})

    def test_an_unchanged_assignment_hides_the_bug(self) -> None:
        # Why every single-host test, and every same-job restore, passes.
        def by_rank(pairs: Sequence[tuple[int, int]], n: int) -> list[int]:
            return list(range(n))

        restored = _round_trip_blocks(self._SAVE, self._SAVE, self._P, by_rank)
        self.assertEqual(restored, {block: block for block in range(self._P)})


class CheckpointMeshCoordinateTest(unittest.TestCase):
    """The DTensor the save path builds must be offset by ordinal, not by rank."""

    _WORLD = 4
    _RANK = 1
    _ORDINAL = 3

    def setUp(self) -> None:
        dist.init_process_group(
            backend="fake",
            store=FakeStore(),
            rank=self._RANK,
            world_size=self._WORLD,
        )

    def tearDown(self) -> None:
        dist.destroy_process_group()

    def test_local_rows_land_at_the_ordinal_offset(self) -> None:
        checkpointable = RecirCheckpointable(
            _Wrapper(),
            num_processes=self._WORLD,
            sharded_prefixes=("_sharded_tables",),
            table_prefixes=("_sharded_tables",),
            warmup=_Warmup(),
            # This job put rank 1 at FSDP ordinal 3.
            shard_order=shard_mesh_order([(0, 0), (1, 3), (2, 1), (3, 2)], self._WORLD),
        )
        local = torch.zeros(8, 2)
        wrapped = wrap_sharded_for_save(
            {"_sharded_tables.table": local},
            ("_sharded_tables",),
            checkpointable._mesh(),
        )
        dtensor = wrapped["_sharded_tables.table"]
        _, offset = compute_local_shape_and_global_offset(
            dtensor.shape, dtensor.device_mesh, dtensor.placements
        )
        # 24 = ordinal 3 * 8 rows. Numbered by rank it would be 8.
        self.assertEqual(offset, (self._ORDINAL * 8, 0))

    def test_a_missing_shard_order_is_refused(self) -> None:
        checkpointable = RecirCheckpointable(
            _Wrapper(),
            num_processes=self._WORLD,
            sharded_prefixes=("_sharded_tables",),
            table_prefixes=("_sharded_tables",),
            warmup=_Warmup(),
        )
        with self.assertRaisesRegex(RuntimeError, "needs a shard_order"):
            checkpointable._mesh()


class DcpRoundTripTest(unittest.TestCase):
    """The real dcp.save/dcp.load path, not a hand-rolled stand-in for it.

    Everything else about the host buffers rests on one unverified belief: that
    `dcp.load` writes into the tensors `state_dict()` hands it, in place. That
    belief is why `state_dict()` returns views and why
    `load_module_state_without_self_copy` skips what DCP already wrote. Nothing
    in this codebase owns that write -- it happens inside
    torch.distributed.checkpoint -- so the only way to know is to run it.
    """

    def setUp(self) -> None:
        self._dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self._dir, ignore_errors=True)

    def test_load_writes_in_place_into_the_established_buffers(self) -> None:
        wrapper = _Wrapper()
        flat_state = [wrapper.dense, wrapper._sharded_tables["table"]]
        state_fqns = ["[0]['dense']", "[0]['_sharded_tables.table']"]
        buffers = _build_buffers(flat_state, state_fqns, wrapper)

        def make_checkpointable() -> RecirCheckpointable:
            return RecirCheckpointable(
                wrapper,
                num_processes=1,
                sharded_prefixes=("_sharded_tables",),
                table_prefixes=("_sharded_tables",),
                warmup=_Warmup(),
                host_buffers=lambda: buffers,
            )

        with torch.no_grad():
            wrapper.dense.fill_(11.0)
            wrapper._sharded_tables["table"].fill_(22.0)
        dense_ptr = wrapper.dense.data_ptr()
        table_ptr = wrapper._sharded_tables["table"].data_ptr()

        dcp.save(
            {"model": make_checkpointable()},
            storage_writer=FileSystemWriter(self._dir),
        )

        with torch.no_grad():
            wrapper.dense.fill_(-1.0)
            wrapper._sharded_tables["table"].fill_(-2.0)

        dcp.load(
            {"model": make_checkpointable()},
            storage_reader=FileSystemReader(self._dir),
        )

        # The values came back...
        torch.testing.assert_close(wrapper.dense.detach(), torch.tensor([11.0]))
        torch.testing.assert_close(
            wrapper._sharded_tables["table"].detach(), torch.tensor([22.0])
        )
        # ...into the same storage, which is the part nothing else checks. A
        # load that allocated instead would restore these values too, and leave
        # the compiled graph reading the tensor it was given at build time.
        self.assertEqual(wrapper.dense.data_ptr(), dense_ptr)
        self.assertEqual(wrapper._sharded_tables["table"].data_ptr(), table_ptr)
