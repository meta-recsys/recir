#!/usr/bin/env python3

import unittest

import torch
from recir.examples.rankfm_mvai.checkpointable import RecirCheckpointable


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
