#!/usr/bin/env python3

import unittest
from collections.abc import Callable, Sequence

import torch
import torch.distributed as dist
from recir.examples.rankfm_mvai.checkpointable import (
    host_ordinal_from_device_positions,
    RecirCheckpointable,
    shard_mesh_order,
    wrap_sharded_for_save,
)
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


class _Warmup:
    iter = 3


class RecirCheckpointableTest(unittest.TestCase):
    def test_async_snapshot_copies_mutable_state_but_reuses_table_storage(self) -> None:
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
        self.assertNotEqual(wrapper.dense.data_ptr(), dense.data_ptr())
        self.assertEqual(wrapper._sharded_tables["table"].data_ptr(), table.data_ptr())
        with torch.no_grad():
            wrapper.dense.fill_(4.0)
        torch.testing.assert_close(torch.tensor([1.0]), dense)
        self.assertIs(state, checkpointable.make_async_stager().stage(state))


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
