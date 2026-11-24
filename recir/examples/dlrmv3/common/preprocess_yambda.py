# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Download Yambda (huggingface.co/datasets/yandex/yambda) and convert it for DLRMv3.

Layout written under --out_root:
  raw/README.md, raw/flat/<scale>/multi_event.parquet
  processed/yambda-<scale>/
    item_id.npy       int32  dense item id per listen (1-based, 0 = padding)
    timestamp.npy     int32  seconds
    action.npy        uint8  ACTION_BITS
    watch_time.npy    uint16 seconds (played_ratio_pct * track_length_seconds / 100)
    user_offsets.npy  int64  [num_users + 1], listens of user u are [off[u], off[u+1])
    user_ids.npy      uint32 original uid per user
    item_vocab.npy    uint32 original item_id of dense id i at position i - 1
    meta.json         written last; its presence marks a complete conversion

One token per listen, ordered by (uid, timestamp). Like/dislike events are folded
into the action bits of the latest listen of the same (uid, item) at or before them.
"""

import argparse
import datetime
import hashlib
import json
import logging
import os
import shutil
import urllib.request

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

logger: logging.Logger = logging.getLogger(__name__)

HF_REPO = "https://huggingface.co/datasets/yandex/yambda"
LAST_TIMESTAMP = 26_000_000
TEST_SECONDS = 86_400
TRAIN_TEST_GAP_SECONDS = 1_800
LISTEN_PLUS_PCT = 50
MAX_FEEDBACK_GAP_SECONDS = 3_600
ACTION_BITS: dict[str, int] = {"listen_plus": 1, "like": 2, "dislike": 4, "organic": 8}
ITEM_BITS = 24


def download(url: str, path: str) -> str:
    sha = hashlib.sha256()
    tmp = f"{path}.part"
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        while chunk := r.read(8 << 20):
            f.write(chunk)
            sha.update(chunk)
    os.replace(tmp, path)
    return sha.hexdigest()


def attach_feedback(
    l_key: np.ndarray, l_ts: np.ndarray, f_key: np.ndarray, f_ts: np.ndarray
) -> np.ndarray:
    """Listen index each feedback event attaches to, or -1 if there is none."""
    n_l = len(l_key)
    key = np.concatenate([l_key, f_key])
    ts = np.concatenate([l_ts, f_ts])
    is_fb = np.concatenate([np.zeros(n_l, np.int8), np.ones(len(f_key), np.int8)])
    order = np.lexsort((is_fb, ts, key))
    key, ts, is_fb = key[order], ts[order], is_fb[order]
    pos = np.arange(len(order))
    last_listen = np.maximum.accumulate(np.where(is_fb == 0, pos, -1))
    fb_pos = np.flatnonzero(is_fb)
    prev = last_listen[fb_pos]
    prev_c = np.maximum(prev, 0)
    ok = (prev >= 0) & (key[prev_c] == key[fb_pos])
    ok &= ts[fb_pos] - ts[prev_c] <= MAX_FEEDBACK_GAP_SECONDS
    out = np.full(len(f_key), -1, np.int64)
    out[order[fb_pos] - n_l] = np.where(ok, order[prev_c], -1)
    return out


def load_events(parquet_path: str) -> tuple[dict[str, np.ndarray], pa.ChunkedArray]:
    t = pq.read_table(parquet_path)
    logger.info(f"read {parquet_path}: {t.num_rows} rows")
    cols = {
        c: pc.fill_null(t[c], 0).to_numpy().astype(np.int64)
        for c in (
            "uid",
            "item_id",
            "timestamp",
            "is_organic",
            "played_ratio_pct",
            "track_length_seconds",
        )
    }
    assert cols["item_id"].max() < (1 << ITEM_BITS)
    cols["key"] = (cols["uid"] << ITEM_BITS) | cols["item_id"]
    return cols, pc.cast(t["event_type"], pa.string())


def build_listens(
    cols: dict[str, np.ndarray], event_type: pa.ChunkedArray
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    def where(name: str) -> np.ndarray:
        return np.flatnonzero(pc.equal(event_type, name).to_numpy())

    li = where("listen")
    uid, ts = cols["uid"][li], cols["timestamp"][li]
    d_uid, d_ts = np.diff(uid), np.diff(ts)
    assert np.all((d_uid > 0) | ((d_uid == 0) & (d_ts >= 0))), "not sorted by (uid, ts)"
    ratio = cols["played_ratio_pct"][li]
    action = np.where(ratio >= LISTEN_PLUS_PCT, ACTION_BITS["listen_plus"], 0)
    action |= np.where(cols["is_organic"][li] == 1, ACTION_BITS["organic"], 0)
    stats = {"listens": len(li)}
    for name in ("like", "dislike"):
        fi = where(name)
        target = attach_feedback(
            cols["key"][li], ts, cols["key"][fi], cols["timestamp"][fi]
        )
        hit = target[target >= 0]
        action[hit] |= ACTION_BITS[name]
        stats[f"{name}_events"] = len(fi)
        stats[f"{name}_attached"] = len(hit)
    watch = np.minimum(ratio * cols["track_length_seconds"][li] // 100, 65535)
    listens = {
        "uid": uid,
        "item": cols["item_id"][li],
        "timestamp": ts,
        "action": action.astype(np.uint8),
        "watch_time": watch.astype(np.uint16),
    }
    return listens, stats


def convert(parquet_path: str, out_dir: str) -> dict[str, object]:
    listens, stats = build_listens(*load_events(parquet_path))
    item_vocab, item_dense = np.unique(listens["item"], return_inverse=True)
    user_ids, user_start = np.unique(listens["uid"], return_index=True)
    ts = listens["timestamp"]
    arrays = {
        "item_id": (item_dense + 1).astype(np.int32),
        "timestamp": ts.astype(np.int32),
        "action": listens["action"],
        "watch_time": listens["watch_time"],
        "user_offsets": np.append(user_start, len(ts)).astype(np.int64),
        "user_ids": user_ids.astype(np.uint32),
        "item_vocab": item_vocab.astype(np.uint32),
    }
    os.makedirs(out_dir, exist_ok=True)
    for name, arr in arrays.items():
        np.save(os.path.join(out_dir, f"{name}.npy"), arr)
    test_start = LAST_TIMESTAMP - TEST_SECONDS
    train_end = test_start - TRAIN_TEST_GAP_SECONDS
    stats["train_listens"] = int((ts < train_end).sum())
    stats["test_listens"] = int((ts >= test_start).sum())
    return {
        "num_users": len(user_ids),
        "num_items": len(item_vocab),
        "num_listens": len(ts),
        "train_end_ts": train_end,
        "test_start_ts": test_start,
        "last_ts": LAST_TIMESTAMP,
        "action_bits": ACTION_BITS,
        "listen_plus_pct": LISTEN_PLUS_PCT,
        "max_feedback_gap_seconds": MAX_FEEDBACK_GAP_SECONDS,
        "stats": stats,
    }


def write_readme(raw_root: str, scale: str, sha: str) -> None:
    text = (
        "Yambda (Yandex Music) public dataset.\n"
        f"Source: {HF_REPO} (flat/{scale}/multi_event.parquet, sha256 {sha})\n"
        "License: Apache-2.0; the dataset card says it is published exclusively for\n"
        "scientific and research purposes.\n"
        f"Downloaded {datetime.date.today()} by pod {os.environ.get('HOSTNAME', '?')}\n"
        "with recir/examples/dlrmv3/common/preprocess_yambda.py.\n"
    )
    with open(os.path.join(raw_root, "README.md"), "a") as f:
        f.write(text + "\n")


def publish(local_dir: str, dest_dir: str, meta: dict[str, object]) -> None:
    os.makedirs(dest_dir, exist_ok=True)
    for name in sorted(os.listdir(local_dir)):
        shutil.copyfile(os.path.join(local_dir, name), os.path.join(dest_dir, name))
    with open(os.path.join(dest_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    for name in sorted(os.listdir(local_dir)):
        local, dest = os.path.join(local_dir, name), os.path.join(dest_dir, name)
        assert os.path.getsize(local) == os.path.getsize(dest), f"size mismatch {dest}"
    shape = np.load(os.path.join(dest_dir, "item_id.npy"), mmap_mode="r").shape
    logger.info(f"published {dest_dir}: {sorted(os.listdir(dest_dir))} item_id{shape}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="YAMBDA %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--scale", default="50m", choices=["50m", "500m", "5b"])
    p.add_argument("--out_root", required=True)
    p.add_argument("--work_dir", default="/tmp/yambda_work")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    dest_dir = os.path.join(args.out_root, "processed", f"yambda-{args.scale}")
    if os.path.exists(os.path.join(dest_dir, "meta.json")) and not args.force:
        logger.info(f"{dest_dir} already complete, skipping (pass --force to redo)")
        return
    work = os.path.join(args.work_dir, args.scale)
    os.makedirs(work, exist_ok=True)
    rel = f"flat/{args.scale}/multi_event.parquet"
    parquet = os.path.join(work, "multi_event.parquet")
    sha = download(f"{HF_REPO}/resolve/main/{rel}", parquet)
    logger.info(f"downloaded {rel}: {os.path.getsize(parquet)} bytes sha256 {sha}")

    meta = convert(parquet, os.path.join(work, "processed"))
    meta |= {
        "source": f"{HF_REPO}/{rel}",
        "sha256": sha,
        "created": str(datetime.date.today()),
    }
    logger.info(f"converted: {json.dumps(meta)}")

    raw_root = os.path.join(args.out_root, "raw")
    os.makedirs(os.path.join(raw_root, "flat", args.scale), exist_ok=True)
    shutil.copyfile(parquet, os.path.join(raw_root, rel))
    write_readme(raw_root, args.scale, sha)
    publish(os.path.join(work, "processed"), dest_dir, meta)
    logger.info("DONE")


if __name__ == "__main__":
    main()
