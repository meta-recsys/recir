#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Build a human- and machine-readable index of a run's DCP checkpoints in GCS.

The checkpoints themselves are TB-scale and stay in GCS; only this index is
published (see sync_run_artifacts.sh). Registering them in Model Store / MLHub
is not possible: UMM's StorageType enum has no GCS member and the trainer pods
have no Meta identity.

Layout produced by mvai_cloud's CheckpointManager:
    <prefix>/latest.txt            -> "step_0002500"
    <prefix>/step_%07d/meta.json   -> {"step": int, "timestamp": float}
    <prefix>/step_%07d/__*.distcp  -> the DCP shards

Stdlib only: this runs with plain python3 on a devserver.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

# `gsutil ls -l` lines look like:
#     123456  2026-09-04T18:02:10Z  gs://bucket/checkpoints/js/step_0002500/x
# and the listing ends with a "TOTAL: ..." line that has no URL.
_LS_LINE = re.compile(r"^\s*(\d+)\s+\S+\s+(gs://\S+)\s*$")
_STEP_DIR = re.compile(r"/(step_\d+)/")


class Checkpoint(TypedDict):
    step_dir: str
    step: int | None
    timestamp: float | None
    timestamp_utc: str | None
    bytes: int
    files: int
    path: str
    is_latest: bool


class CheckpointIndex(TypedDict):
    gcs_prefix: str
    latest: str | None
    generated_utc: str
    checkpoints: list[Checkpoint]


def _gsutil(args: list[str]) -> str:
    result = subprocess.run(
        ["gsutil", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"gsutil {' '.join(args)} failed ({result.returncode}): "
            f"{result.stderr.strip()[:500]}"
        )
    return result.stdout


def _read_latest(prefix: str) -> str | None:
    try:
        return _gsutil(["cat", f"{prefix}/latest.txt"]).strip() or None
    except RuntimeError:
        # latest.txt is written by rank 0 after the first successful save, so
        # its absence just means no checkpoint has completed yet.
        return None


def _collect_objects(prefix: str) -> dict[str, dict[str, int]]:
    """Map step dir name -> {"bytes": total, "files": count}."""
    try:
        listing = _gsutil(["ls", "-l", "-r", f"{prefix}/**"])
    except RuntimeError:
        # gsutil exits non-zero when nothing matches, which is the normal state
        # until the first save completes.
        return {}
    steps: dict[str, dict[str, int]] = {}
    for line in listing.splitlines():
        match = _LS_LINE.match(line)
        if match is None:
            continue
        size, url = int(match.group(1)), match.group(2)
        step_match = _STEP_DIR.search(url)
        if step_match is None:
            continue
        entry = steps.setdefault(step_match.group(1), {"bytes": 0, "files": 0})
        entry["bytes"] += size
        entry["files"] += 1
    return steps


def _read_meta(prefix: str, step_dir: str) -> dict[str, object]:
    try:
        return json.loads(_gsutil(["cat", f"{prefix}/{step_dir}/meta.json"]))
    except (RuntimeError, json.JSONDecodeError):
        # A save that died between the shard write and the rank-0 meta.json
        # write leaves a step dir with no meta. Report it rather than crash.
        return {}


def build_index(prefix: str) -> CheckpointIndex:
    prefix = prefix.rstrip("/")
    latest = _read_latest(prefix)
    steps = _collect_objects(prefix)

    checkpoints: list[Checkpoint] = []
    for step_dir in sorted(steps):
        meta = _read_meta(prefix, step_dir)
        step = meta.get("step")
        timestamp = meta.get("timestamp")
        timestamp = float(timestamp) if isinstance(timestamp, (int, float)) else None
        checkpoints.append(
            {
                "step_dir": step_dir,
                "step": step if isinstance(step, int) else None,
                "timestamp": timestamp,
                "timestamp_utc": (
                    datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(
                        timespec="seconds"
                    )
                    if timestamp is not None
                    else None
                ),
                "bytes": steps[step_dir]["bytes"],
                "files": steps[step_dir]["files"],
                "path": f"{prefix}/{step_dir}",
                "is_latest": step_dir == latest,
            }
        )

    return {
        "gcs_prefix": prefix,
        "latest": latest,
        "generated_utc": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "checkpoints": checkpoints,
    }


def _gib(num_bytes: int) -> str:
    return f"{num_bytes / (1024**3):.1f} GiB"


def render_markdown(index: CheckpointIndex) -> str:
    checkpoints = index["checkpoints"]
    lines = [
        "# Checkpoints",
        "",
        f"- GCS prefix: `{index['gcs_prefix']}`",
        f"- latest: `{index['latest'] or 'none'}`",
        f"- generated: {index['generated_utc']}",
        "",
        "| step | saved (UTC) | size | files | latest | path |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for entry in checkpoints:
        lines.append(
            "| {step} | {when} | {size} | {files} | {latest} | `{path}` |".format(
                step=entry["step"] if entry["step"] is not None else entry["step_dir"],
                when=entry["timestamp_utc"] or "?",
                size=_gib(entry["bytes"]),
                files=entry["files"],
                latest="yes" if entry["is_latest"] else "",
                path=entry["path"],
            )
        )
    if not checkpoints:
        lines.append("| _(none yet)_ | | | | | |")
    lines.append("")
    total = sum(entry["bytes"] for entry in checkpoints)
    lines.append(f"Total: {len(checkpoints)} checkpoint(s), {_gib(total)}.")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gcs-prefix",
        required=True,
        help="gs://<bucket>/checkpoints/<jobset-name>",
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    index = build_index(args.gcs_prefix)
    if not index["checkpoints"]:
        print("no checkpoints found", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    (args.out_dir / "index.md").write_text(render_markdown(index))
    print(f"{len(index['checkpoints'])} checkpoint(s) -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
