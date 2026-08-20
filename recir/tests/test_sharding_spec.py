# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from dataclasses import FrozenInstanceError

from recir.distributed.sharding_spec import (
    copy_state_layout,
    FSDPWeightShardSpec,
    get_state_layout,
    set_state_layout,
    STATE_LAYOUT_META_KEY,
)
from torch import fx


def _make_placeholder() -> fx.Node:
    graph = fx.Graph()
    return graph.placeholder("x")


class ShardingSpecTest(unittest.TestCase):
    def test_spec_stores_fields(self) -> None:
        spec = FSDPWeightShardSpec(
            shard_dim=0, device_world_size=8, process_world_size=2
        )
        self.assertEqual(spec.shard_dim, 0)
        self.assertEqual(spec.device_world_size, 8)
        self.assertEqual(spec.process_world_size, 2)

    def test_spec_is_frozen(self) -> None:
        spec = FSDPWeightShardSpec(
            shard_dim=0, device_world_size=4, process_world_size=1
        )
        frozen_field = "shard_dim"
        with self.assertRaises(FrozenInstanceError):
            setattr(spec, frozen_field, 3)

    def test_set_get_round_trip(self) -> None:
        node = _make_placeholder()
        spec = FSDPWeightShardSpec(
            shard_dim=0, device_world_size=8, process_world_size=2
        )
        set_state_layout(node, spec)
        self.assertIn(STATE_LAYOUT_META_KEY, node.meta)
        self.assertIs(node.meta[STATE_LAYOUT_META_KEY], spec)
        self.assertIs(get_state_layout(node), spec)

    def test_get_returns_none_when_absent(self) -> None:
        node = _make_placeholder()
        self.assertNotIn(STATE_LAYOUT_META_KEY, node.meta)
        self.assertIsNone(get_state_layout(node))

    def test_copy_propagates_spec_when_present(self) -> None:
        src = _make_placeholder()
        dst = _make_placeholder()
        spec = FSDPWeightShardSpec(
            shard_dim=0, device_world_size=16, process_world_size=4
        )
        set_state_layout(src, spec)
        copy_state_layout(src, dst)
        self.assertIs(get_state_layout(dst), spec)

    def test_copy_is_noop_when_absent(self) -> None:
        src = _make_placeholder()
        dst = _make_placeholder()
        copy_state_layout(src, dst)
        self.assertNotIn(STATE_LAYOUT_META_KEY, dst.meta)
        self.assertIsNone(get_state_layout(dst))


class FSDPWeightShardSpecValidationTest(unittest.TestCase):
    def test_valid_spec_constructs(self) -> None:
        spec = FSDPWeightShardSpec(
            shard_dim=0, device_world_size=8, process_world_size=2
        )
        self.assertEqual(spec.shard_dim, 0)
        self.assertEqual(spec.device_world_size, 8)
        self.assertEqual(spec.process_world_size, 2)

    def test_n_not_divisible_by_p_raises(self) -> None:
        with self.assertRaises(ValueError):
            FSDPWeightShardSpec(shard_dim=0, device_world_size=6, process_world_size=4)

    def test_non_positive_device_world_size_raises(self) -> None:
        with self.assertRaises(ValueError):
            FSDPWeightShardSpec(shard_dim=0, device_world_size=0, process_world_size=2)
        with self.assertRaises(ValueError):
            FSDPWeightShardSpec(shard_dim=0, device_world_size=-8, process_world_size=2)

    def test_non_positive_process_world_size_raises(self) -> None:
        with self.assertRaises(ValueError):
            FSDPWeightShardSpec(shard_dim=0, device_world_size=8, process_world_size=0)
        with self.assertRaises(ValueError):
            FSDPWeightShardSpec(shard_dim=0, device_world_size=8, process_world_size=-2)

    def test_non_zero_shard_dim_raises(self) -> None:
        with self.assertRaises(ValueError):
            FSDPWeightShardSpec(shard_dim=1, device_world_size=8, process_world_size=2)


if __name__ == "__main__":
    unittest.main()
