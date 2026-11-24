# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Yambda (Yandex Music, huggingface.co/datasets/yandex/yambda) for DLRMv3.

Reads the conversion ``preprocess_yambda.py`` writes to
``<data_root>/processed/yambda-<scale>/``: one token per listen, ordered by
(user, timestamp), with dense item ids (1-based) and action bits (listen_plus 1,
like 2, dislike 4, organic 8).

A sample is ``num_candidates`` consecutive listens of one user (the candidates)
after the most recent ``max_uih_len`` listens before them (the history). The
split follows the Yambda benchmark: training candidates end before
``train_end_ts`` (the test day minus a 30 min gap); evaluation candidates are
the listens of the last day, each window seeing every listen before it.
Training windows tile each user's training listens backwards from the split,
so the most recent listens are always used.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Tuple

import numpy as np
import torch
from generative_recommenders.dlrm_v3.configs import get_hstu_configs, HSTU_EMBEDDING_DIM
from generative_recommenders.modules.dlrm_hstu import DlrmHSTUConfig
from generative_recommenders.modules.multitask_module import (
    MultitaskTaskType,
    TaskConfig,
)
from torchrec.modules.embedding_configs import DataType, EmbeddingConfig
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

logger: logging.Logger = logging.getLogger(__name__)

PREFIX = "yambda-"
TRACK, USER = "track_id", "user_id"
UIH_KEYS: Tuple[str, ...] = (
    USER,
    TRACK,
    "action_timestamp",
    "action_weight",
    "watch_time",
)
CANDIDATE_KEYS: Tuple[str, ...] = (
    "item_track_id",
    "item_action_weight",
    "item_target_watchtime",
    "item_query_time",
)
# Action bits the preprocessing writes; "organic" is an input signal only.
TASKS: Dict[str, int] = {"listen_plus": 1, "like": 2, "dislike": 4}
ACTION_WEIGHTS = [1, 2, 4, 8]

Sample = Tuple[KeyedJaggedTensor, KeyedJaggedTensor]


def is_yambda(dataset: str) -> bool:
    return dataset.startswith(PREFIX)


def data_dir(data_root: str, dataset: str) -> str:
    return os.path.join(data_root, "processed", dataset)


def read_meta(data_root: str, dataset: str) -> Dict[str, Any]:
    with open(os.path.join(data_dir(data_root, dataset), "meta.json")) as f:
        return json.load(f)


def hstu_config() -> DlrmHSTUConfig:
    cfg = get_hstu_configs()
    cfg.user_embedding_feature_names = [TRACK, USER]
    cfg.item_embedding_feature_names = ["item_track_id"]
    cfg.uih_post_id_feature_name = TRACK
    cfg.uih_action_time_feature_name = "action_timestamp"
    cfg.uih_weight_feature_name = "action_weight"
    cfg.candidates_querytime_feature_name = "item_query_time"
    cfg.candidates_weight_feature_name = "item_action_weight"
    cfg.candidates_watchtime_feature_name = "item_target_watchtime"
    cfg.contextual_feature_to_max_length = {USER: 1}
    cfg.contextual_feature_to_min_uih_length = {}
    cfg.merge_uih_candidate_feature_mapping = [
        (TRACK, "item_track_id"),
        ("action_timestamp", "item_query_time"),
        ("action_weight", "item_action_weight"),
        ("watch_time", "item_target_watchtime"),
    ]
    cfg.hstu_uih_feature_names = list(UIH_KEYS)
    cfg.hstu_candidate_feature_names = list(CANDIDATE_KEYS)
    cfg.multitask_configs = [
        TaskConfig(
            task_name=name,
            task_weight=bit,
            task_type=MultitaskTaskType.BINARY_CLASSIFICATION,
        )
        for name, bit in TASKS.items()
    ]
    cfg.action_weights = ACTION_WEIGHTS
    return cfg


def embedding_tables(meta: Dict[str, Any]) -> Dict[str, EmbeddingConfig]:
    """Row 0 of each table is padding; ids are dense, so nothing is hashed."""
    rows = {TRACK: int(meta["num_items"]) + 1, USER: int(meta["num_users"]) + 1}
    features = {TRACK: [TRACK, "item_track_id"], USER: [USER]}
    return {
        name: EmbeddingConfig(
            num_embeddings=rows[name],
            embedding_dim=HSTU_EMBEDDING_DIM,
            name=name,
            data_type=DataType.FP16,
            feature_names=features[name],
        )
        for name in (TRACK, USER)
    }


def _windows(
    starts: np.ndarray, ends: np.ndarray, size: int, backwards: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """(owner, first position) of the ``size``-long windows tiling each
    [start, end); backwards keeps only whole windows, aligned to the end."""
    span = np.maximum(ends - starts, 0)
    count = span // size if backwards else -(-span // size)
    owner = np.repeat(np.arange(len(starts)), count)
    k = np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
    pos = ends[owner] - (k + 1) * size if backwards else starts[owner] + k * size
    return owner, pos


class YambdaDataset:
    """generative-recommenders ``Dataset`` protocol over the converted arrays.

    Samples ``[0, num_train_samples)`` are training windows, the rest evaluation.
    """

    def __init__(
        self,
        path: str,
        embedding_config: Dict[str, EmbeddingConfig],
        num_candidates: int,
        max_uih_len: int = 0,
    ) -> None:
        with open(os.path.join(path, "meta.json")) as f:
            meta = json.load(f)
        self.item_id: np.ndarray = np.load(os.path.join(path, "item_id.npy"))
        self.timestamp: np.ndarray = np.load(os.path.join(path, "timestamp.npy"))
        self.action: np.ndarray = np.load(os.path.join(path, "action.npy"))
        self.watch_time: np.ndarray = np.load(os.path.join(path, "watch_time.npy"))
        offsets = np.load(os.path.join(path, "user_offsets.npy"))
        self.user_start: np.ndarray = offsets[:-1]
        self.user_end: np.ndarray = offsets[1:]
        self.max_uih_len = max_uih_len
        self.track_rows: int = embedding_config[TRACK].num_embeddings
        self.user_rows: int = embedding_config[USER].num_embeddings

        def first_at(t: int) -> np.ndarray:
            return np.array(
                [
                    s + np.searchsorted(self.timestamp[s:e], t)
                    for s, e in zip(self.user_start, self.user_end)
                ],
                dtype=np.int64,
            )

        train_end = first_at(int(meta["train_end_ts"]))
        test_start = first_at(int(meta["test_start_ts"]))
        # Every window needs at least one listen of history.
        train = _windows(self.user_start + 1, train_end, num_candidates, True)
        test = _windows(
            np.maximum(test_start, self.user_start + 1),
            self.user_end,
            num_candidates,
            False,
        )
        self.num_train_samples: int = len(train[0])
        self.sample_user: np.ndarray = np.concatenate([train[0], test[0]])
        self.sample_pos: np.ndarray = np.concatenate([train[1], test[1]])
        self.sample_end: np.ndarray = np.minimum(
            self.sample_pos + num_candidates, self.user_end[self.sample_user]
        )
        logger.info(
            "%s: %d users, %d listens -> %d train / %d eval samples",
            path,
            len(self.user_start),
            len(self.item_id),
            self.num_train_samples,
            len(test[0]),
        )

    def get_item_count(self) -> int:
        return len(self.sample_user)

    def load_query_samples(self, sample_list: Any) -> None:
        pass

    def unload_query_samples(self, sample_list: Any) -> None:
        pass

    def get_sample(self, index: int) -> Sample:
        u = int(self.sample_user[index])
        p, q = int(self.sample_pos[index]), int(self.sample_end[index])
        h = int(self.user_start[u])
        if self.max_uih_len:
            h = max(h, p - self.max_uih_len)
        tracks = self.item_id[h:q].astype(np.int64) % self.track_rows
        uih = KeyedJaggedTensor(
            keys=list(UIH_KEYS),
            lengths=torch.tensor([1] + [p - h] * (len(UIH_KEYS) - 1)),
            values=torch.from_numpy(
                np.concatenate(
                    [
                        [(u + 1) % self.user_rows],
                        tracks[: p - h],
                        self.timestamp[h:p],
                        self.action[h:p],
                        self.watch_time[h:p],
                    ]
                ).astype(np.int64)
            ),
        )
        candidates = KeyedJaggedTensor(
            keys=list(CANDIDATE_KEYS),
            lengths=torch.full((len(CANDIDATE_KEYS),), q - p),
            values=torch.from_numpy(
                np.concatenate(
                    [
                        tracks[p - h :],
                        self.action[p:q],
                        self.watch_time[p:q],
                        # The request time: the first candidate's.
                        np.full(q - p, self.timestamp[p]),
                    ]
                ).astype(np.int64)
            ),
        )
        return uih, candidates
