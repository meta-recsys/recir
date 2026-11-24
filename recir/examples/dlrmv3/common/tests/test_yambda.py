# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import tempfile
import unittest

import numpy as np
from recir.examples.dlrmv3.common import yambda
from recir.examples.dlrmv3.common.config import DLRMv3TrainConfig
from recir.examples.dlrmv3.common.data import make_dataset, TokenBudgetCollate
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

DATASET = "yambda-test"
# User 0: listens 0-9 before the split, 10-12 on the test day. User 1: one
# listen before, two on the test day. User 2: test-day listens only.
TIMESTAMPS = list(range(10)) + [2000, 2001, 2002] + [5, 2005, 2006] + [2010, 2011, 2012]
OFFSETS = [0, 13, 16, 19]
NUM_ITEMS = 20


def _write(root: str) -> None:
    path = yambda.data_dir(root, DATASET)
    os.makedirs(path)
    n = len(TIMESTAMPS)
    arrays = {
        "item_id": (np.arange(n) % NUM_ITEMS + 1).astype(np.int32),
        "timestamp": np.array(TIMESTAMPS, np.int32),
        "action": (np.arange(n) % 16).astype(np.uint8),
        "watch_time": (np.arange(n) * 10).astype(np.uint16),
        "user_offsets": np.array(OFFSETS, np.int64),
    }
    for name, arr in arrays.items():
        np.save(os.path.join(path, f"{name}.npy"), arr)
    meta = {
        "num_users": len(OFFSETS) - 1,
        "num_items": NUM_ITEMS,
        "train_end_ts": 1000,
        "test_start_ts": 2000,
    }
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump(meta, f)


def _split(kjt: KeyedJaggedTensor) -> dict[str, list[int]]:
    return {k: v.values().tolist() for k, v in kjt.to_dict().items()}


class YambdaTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        _write(self.root)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _cfg(self, hash_size: int = 0) -> DLRMv3TrainConfig:
        return DLRMv3TrainConfig(
            dataset=DATASET,
            data_root=self.root,
            uih_len=3,
            num_candidates=2,
            hash_size=hash_size,
        )

    def _dataset(self, cfg: DLRMv3TrainConfig) -> yambda.YambdaDataset:
        hstu = cfg.hstu_config()
        gr = make_dataset(
            DATASET, self.root, hstu, cfg.embedding_tables(), cfg.layout(hstu)
        )
        return gr.dataset

    def test_tables_cover_the_dense_ids(self) -> None:
        tables = self._cfg().embedding_tables()
        self.assertEqual(tables[yambda.TRACK].num_embeddings, NUM_ITEMS + 1)
        self.assertEqual(tables[yambda.USER].num_embeddings, len(OFFSETS))
        tasks = [t.task_name for t in self._cfg().hstu_config().multitask_configs]
        self.assertEqual(tasks, ["listen_plus", "like", "dislike"])

    def test_windows_split_by_time(self) -> None:
        ds = self._dataset(self._cfg())
        # Training: whole windows tiling user 0's listens 1-9 back from the split.
        self.assertEqual(ds.num_train_samples, 4)
        self.assertEqual(ds.sample_pos[:4].tolist(), [8, 6, 4, 2])
        # Evaluation: every test-day listen with history, last window partial.
        self.assertEqual(ds.sample_pos[4:].tolist(), [10, 12, 14, 17])
        self.assertEqual(ds.sample_end[4:].tolist(), [12, 13, 16, 19])
        self.assertEqual(ds.sample_user.tolist(), [0, 0, 0, 0, 0, 0, 1, 2])

    def test_sample_reads_recent_history_and_candidates(self) -> None:
        uih, cand = (_split(k) for k in self._dataset(self._cfg()).get_sample(0))
        self.assertEqual(uih[yambda.USER], [1])
        self.assertEqual(uih[yambda.TRACK], [6, 7, 8])
        self.assertEqual(uih["action_timestamp"], [5, 6, 7])
        self.assertEqual(uih["action_weight"], [5, 6, 7])
        self.assertEqual(uih["watch_time"], [50, 60, 70])
        self.assertEqual(cand["item_track_id"], [9, 10])
        self.assertEqual(cand["item_action_weight"], [8, 9])
        self.assertEqual(cand["item_target_watchtime"], [80, 90])
        self.assertEqual(cand["item_query_time"], [8, 8])

    def test_resized_tables_hash_the_ids(self) -> None:
        uih, cand = (
            _split(k) for k in self._dataset(self._cfg(hash_size=4)).get_sample(0)
        )
        self.assertEqual(uih[yambda.TRACK], [2, 3, 0])
        self.assertEqual(cand["item_track_id"], [1, 2])

    def test_collates_into_the_model_layout(self) -> None:
        cfg = self._cfg()
        hstu = cfg.hstu_config()
        layout = cfg.layout(hstu)
        ds = self._dataset(cfg)
        batch = TokenBudgetCollate(layout, hstu, real_rows=2)(
            [ds.get_sample(5), ds.get_sample(6)]
        )
        self.assertEqual(batch.uih_seq.shape, (layout.tokens(2), len(layout.seq_keys)))
        # Sample 5 has one candidate: the second is padding with weight 0.
        self.assertEqual(batch.cand_weight[:2].tolist(), [[1.0, 0.0], [1.0, 1.0]])
