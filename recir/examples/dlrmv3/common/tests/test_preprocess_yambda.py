# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from recir.examples.dlrmv3.common.preprocess_yambda import convert

# (uid, timestamp, item_id, is_organic, played_ratio_pct, track_length_seconds, event_type)
EVENTS = [
    (100, 10, 5, 1, 100, 200, "listen"),
    (100, 20, 7, 0, 30, 100, "listen"),
    (100, 25, 5, 0, None, None, "like"),
    (100, 30, 5, 0, 60, 100, "listen"),
    (100, 5000, 7, 0, None, None, "dislike"),
    (100, 5001, 9, 0, None, None, "like"),
    (300, 40, 9, 0, 0, 50, "listen"),
    (300, 40, 9, 0, None, None, "like"),
    (300, 26_000_000, 7, 0, 100, 10, "listen"),
    (300, 26_000_000, 7, 0, None, None, "dislike"),
]


class PreprocessYambdaTest(unittest.TestCase):
    def test_convert(self) -> None:
        cols = list(zip(*EVENTS))
        table = pa.table(
            {
                "uid": pa.array(cols[0], pa.uint32()),
                "timestamp": pa.array(cols[1], pa.uint32()),
                "item_id": pa.array(cols[2], pa.uint32()),
                "is_organic": pa.array(cols[3], pa.uint8()),
                "played_ratio_pct": pa.array(cols[4], pa.uint16()),
                "track_length_seconds": pa.array(cols[5], pa.uint32()),
                "event_type": pa.array(cols[6]).dictionary_encode(),
            }
        )
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "multi_event.parquet")
            pq.write_table(table, path)
            out = os.path.join(d, "out")
            meta = convert(path, out)

            def load(name: str) -> list[int]:
                return np.load(os.path.join(out, f"{name}.npy")).tolist()

            self.assertEqual(load("item_vocab"), [5, 7, 9])
            self.assertEqual(load("item_id"), [1, 2, 1, 3, 2])
            self.assertEqual(load("timestamp"), [10, 20, 30, 40, 26_000_000])
            self.assertEqual(load("action"), [11, 0, 1, 2, 5])
            self.assertEqual(load("watch_time"), [200, 30, 60, 0, 10])
            self.assertEqual(load("user_offsets"), [0, 3, 5])
            self.assertEqual(load("user_ids"), [100, 300])

        self.assertEqual(meta["num_users"], 2)
        self.assertEqual(meta["num_items"], 3)
        self.assertEqual(
            meta["stats"],
            {
                "listens": 5,
                "like_events": 3,
                "like_attached": 2,
                "dislike_events": 2,
                "dislike_attached": 1,
                "train_listens": 4,
                "test_listens": 1,
            },
        )
