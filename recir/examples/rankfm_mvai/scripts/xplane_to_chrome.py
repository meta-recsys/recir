#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Convert an xprof ``.xplane.pb`` into a viewable multi-step Chrome trace.

**Why this exists.** A TPU profile directory holds two files per worker:
``<worker>.trace.json.gz`` (Chrome trace, what Perfetto reads) and
``<worker>.xplane.pb`` (the full capture). The ``.trace.json.gz`` is produced
under a hard event cap, and this model emits ~17k device ops *per step*, so the
cap is spent long before the capture is exhausted. What survives is roughly one
step -- and the eviction is indiscriminate, so the small structural lines
(``Steps``, ``XLA Modules``, ``XLA TraceMe``) get dropped alongside the tiny ops
that caused the overflow.

The symptom is a ``--profile_start_step 200 --profile_end_step 205`` capture
that opens in Perfetto showing a single step. The data is not missing; it is in
the ``.xplane.pb`` next to it. For one 6-step capture:

    .trace.json.gz    31 MB    1 step     13,180 ops in step 0
    .xplane.pb       675 MB    6 steps   ~17,457 ops in step 0

This script re-exports from the ``.xplane.pb`` with the priority inverted: every
structural line is kept whole, and only the high-cardinality op lines are
thresholded by duration. That same 6-step capture comes out at 87k events /
1.8 MB, which Perfetto opens without complaint.

**Reading the result.** On ``/device:TPU:N``: ``Steps`` is the whole step span,
``XLA Modules`` the compiled program executions, ``XLA TraceMe`` is where
``barrier-cores`` (cross-core sync at program entry) lives, ``Async XLA Ops``
holds the overlapping ``*-start``/``*-done`` pairs. On ``/host:CPU``: line
``python3`` is the main thread with python-frame events -- the input pipeline
and dispatch path are there. Line ``None`` is mostly parked threads
(``queue.get`` / ``threading.wait``); do not read its totals as work.

Stdlib only, so it runs under plain ``python3`` with no buck target -- which
matters because the profile is usually being triaged on a devserver next to a
675 MB download. It decodes protobuf wire format directly rather than depending
on generated xplane bindings, which are not readily available in fbcode. Lines
whose events are not wanted are skipped by advancing past the length prefix, so
the 675 MB traversal costs seconds rather than minutes.

XPlane schema (tensorflow/core/profiler/protobuf/xplane.proto), field numbers
this decoder relies on::

    XSpace         : planes=1
    XPlane         : id=1 name=2 lines=3 event_metadata=4(map) stat_metadata=5
    XLine          : id=1 name=2 timestamp_ns=3 events=4 duration_ps=9
    XEvent         : metadata_id=1 offset_ps=2 duration_ps=3 stats=4
    XEventMetadata : id=1 name=2 metadata=3 display_name=4

Durations and offsets are picoseconds.

Usage::

    gcloud storage cp \\
      "gs://cq-mrs-core-shared_mrs_tpu_y317h/tpu-profiles/<jobset>/plugins/profile/<ts>/<jobset>-worker-0-0.xplane.pb" .
    python3 xplane_to_chrome.py <worker>.xplane.pb out.trace.json.gz
    manifold put out.trace.json.gz \\
      perfdoctor_gpu_traces_test/tree/traces/test/$USER/rankfm_mvai/out.trace.json.gz
    # https://www.internalfb.com/intern/kernelhub/perfetto/?bucket=perfdoctor_gpu_traces_test&trace_path=tree%2Ftraces%2Ftest%2F$USER%2Frankfm_mvai%2Fout.trace.json.gz

To zoom into one region at full detail, drop the thresholds over a window::

    python3 xplane_to_chrome.py in.xplane.pb out.trace.json.gz --window 12.5 14.1
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import sys
from typing import Any, Dict, Iterator, List, Optional, Tuple

# Structural lines: always kept whole. These are what make the trace readable,
# and they are small -- 12 events each for a 6-step capture.
KEEP_ALL: Tuple[str, ...] = (
    "Steps",
    "XLA Modules",
    "XLA TraceMe",
    "Sparse Core Steps",
    "Sparse Core Modules",
)

# Per-line duration floor in seconds for everything else.
DEFAULT_THRESHOLDS: Dict[str, float] = {
    "XLA Ops": 200e-6,
    "Async XLA Ops": 1e-3,
    "python3": 200e-6,
    "None": 1e-3,
}
FALLBACK_THRESHOLD: float = 1e-3

# Hardware counter lines: tens of thousands of zero-duration samples, no use in
# a timeline view.
SKIP_LINES: frozenset[str] = frozenset({"counters_0", "_counters_"})


def _varint(buf: memoryview, i: int) -> Tuple[int, int]:
    result = 0
    shift = 0
    while True:
        byte = buf[i]
        i += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, i
        shift += 7


def _fields(
    buf: memoryview, start: int, end: int
) -> Iterator[Tuple[int, int, int, Optional[int]]]:
    """Yield (field_number, wire_type, a, b).

    For wire type 0, ``a`` is the value and ``b`` is None. For the
    length-delimited and fixed-width types, ``[a, b)`` is the payload.
    """
    i = start
    while i < end:
        key, i = _varint(buf, i)
        field_no, wire = key >> 3, key & 7
        if wire == 0:
            value, i = _varint(buf, i)
            yield field_no, wire, value, None
        elif wire == 2:
            length, i = _varint(buf, i)
            yield field_no, wire, i, i + length
            i += length
        elif wire == 1:
            yield field_no, wire, i, i + 8
            i += 8
        elif wire == 5:
            yield field_no, wire, i, i + 4
            i += 4
        else:
            raise ValueError(f"unsupported wire type {wire} at offset {i}")


def _text(buf: memoryview, a: int, b: int) -> str:
    return bytes(buf[a:b]).decode("utf-8", "replace")


def _event_metadata(buf: memoryview, a: int, b: int) -> Tuple[Optional[int], str]:
    """Decode one ``map<int64, XEventMetadata>`` entry into (id, name).

    Protobuf maps are repeated key=1/value=2 entries; the id is present on both
    the key and inside the value, and generated writers do not agree on which
    they populate, so take whichever appears.
    """
    entry_id: Optional[int] = None
    name = ""
    for field_no, wire, a2, b2 in _fields(buf, a, b):
        if field_no == 1 and wire == 0:
            entry_id = a2
        elif field_no == 2 and wire == 2 and b2 is not None:
            for f3, w3, a3, b3 in _fields(buf, a2, b2):
                if f3 == 1 and w3 == 0:
                    entry_id = a3
                elif f3 == 2 and w3 == 2 and b3 is not None:
                    name = _text(buf, a3, b3)
    return entry_id, name


def _parse_plane(
    buf: memoryview, a: int, b: int
) -> Tuple[str, List[Tuple[int, int]], Dict[int, str]]:
    """Decode one XPlane header: (name, line spans, event-metadata id -> name)."""
    name = ""
    line_spans: List[Tuple[int, int]] = []
    meta: Dict[int, str] = {}
    for field_no, wire, a2, b2 in _fields(buf, a, b):
        if wire != 2 or b2 is None:
            continue
        if field_no == 2:
            name = _text(buf, a2, b2)
        elif field_no == 3:
            line_spans.append((a2, b2))
        elif field_no == 4:
            meta_id, meta_name = _event_metadata(buf, a2, b2)
            if meta_id is not None:
                meta[meta_id] = meta_name
    return name, line_spans, meta


def _parse_line(buf: memoryview, a: int, b: int) -> Tuple[str, List[Tuple[int, int]]]:
    """Decode one XLine header: (name, event spans). Events stay undecoded."""
    name = ""
    event_spans: List[Tuple[int, int]] = []
    for field_no, wire, a2, b2 in _fields(buf, a, b):
        if wire != 2 or b2 is None:
            continue
        if field_no == 2:
            name = _text(buf, a2, b2)
        elif field_no == 4:
            event_spans.append((a2, b2))
    return name, event_spans


def _parse_event(buf: memoryview, a: int, b: int) -> Tuple[int, int, int]:
    """Decode one XEvent: (metadata_id, offset_ps, duration_ps)."""
    meta_id = offset_ps = duration_ps = 0
    for field_no, wire, a2, _b2 in _fields(buf, a, b):
        if wire != 0:
            continue
        if field_no == 1:
            meta_id = a2
        elif field_no == 2:
            offset_ps = a2
        elif field_no == 3:
            duration_ps = a2
    return meta_id, offset_ps, duration_ps


def _keep_event(
    offset_ps: int,
    duration_ps: int,
    floor: float,
    window: Optional[Tuple[float, float]],
) -> bool:
    """Duration floor outside an explicit window, everything inside it.

    That asymmetry is the point of ``--window``: a region can be re-exported at
    full detail without lifting the thresholds over the whole capture.
    """
    if window is None:
        return duration_ps / 1e12 >= floor
    start_s = offset_ps / 1e12
    end_s = start_s + duration_ps / 1e12
    return end_s >= window[0] and start_s <= window[1]


def _emit_line(
    buf: memoryview,
    event_spans: List[Tuple[int, int]],
    meta: Dict[int, str],
    floor: float,
    window: Optional[Tuple[float, float]],
    pid: int,
    tid: int,
    out: List[Dict[str, Any]],
) -> Tuple[int, int]:
    """Append this line's surviving events. Returns (kept, dropped)."""
    kept = dropped = 0
    for event_a, event_b in event_spans:
        meta_id, offset_ps, duration_ps = _parse_event(buf, event_a, event_b)
        if not _keep_event(offset_ps, duration_ps, floor, window):
            dropped += 1
            continue
        out.append(
            {
                "ph": "X",
                "pid": pid,
                "tid": tid,
                "ts": offset_ps / 1e6,  # ps -> us
                "dur": duration_ps / 1e6,
                "name": (meta.get(meta_id) or f"<meta {meta_id}>")[:160],
            }
        )
        kept += 1
    return kept, dropped


def convert(
    src: str,
    dst: str,
    thresholds: Dict[str, float],
    window: Optional[Tuple[float, float]],
) -> None:
    with open(src, "rb") as handle:
        buf = memoryview(handle.read())

    out: List[Dict[str, Any]] = []
    kept: collections.Counter[str] = collections.Counter()
    dropped: collections.Counter[str] = collections.Counter()
    pid = 0

    for field_no, wire, plane_a, plane_b in _fields(buf, 0, len(buf)):
        if field_no != 1 or wire != 2 or plane_b is None:
            continue
        plane_name, line_spans, meta = _parse_plane(buf, plane_a, plane_b)
        # /host:metadata carries no timeline.
        if not plane_name or "metadata" in plane_name:
            continue

        pid += 1
        out.append(
            {
                "ph": "M",
                "pid": pid,
                "name": "process_name",
                "args": {"name": plane_name},
            }
        )

        tid = 0
        for line_a, line_b in line_spans:
            line_name, event_spans = _parse_line(buf, line_a, line_b)
            if line_name in SKIP_LINES or not event_spans:
                continue
            tid += 1
            out.append(
                {
                    "ph": "M",
                    "pid": pid,
                    "tid": tid,
                    "name": "thread_name",
                    "args": {"name": line_name},
                }
            )
            floor = (
                0.0
                if line_name in KEEP_ALL
                else thresholds.get(line_name, FALLBACK_THRESHOLD)
            )
            line_kept, line_dropped = _emit_line(
                buf, event_spans, meta, floor, window, pid, tid, out
            )
            kept[line_name] += line_kept
            dropped[line_name] += line_dropped

    with gzip.open(dst, "wt") as handle:
        json.dump({"traceEvents": out}, handle)

    print(f"wrote {dst}: {sum(kept.values())} events ({len(out)} incl. metadata)")
    print(f"{'line':32} {'kept':>9} {'dropped':>9}")
    for line_name in sorted(set(kept) | set(dropped), key=lambda k: -kept[k]):
        print(f"{line_name[:32]:32} {kept[line_name]:9} {dropped[line_name]:9}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("xplane", help="path to <worker>.xplane.pb")
    parser.add_argument("out", help="output .trace.json.gz")
    parser.add_argument(
        "--window",
        nargs=2,
        type=float,
        metavar=("START_S", "END_S"),
        help=(
            "keep every event overlapping this range regardless of duration, "
            "and drop everything outside it"
        ),
    )
    parser.add_argument(
        "--threshold",
        action="append",
        default=[],
        metavar="LINE=SECONDS",
        help=(
            "override a line's duration floor, e.g. 'XLA Ops=5e-5'. "
            "Repeatable. Structural lines are always kept whole."
        ),
    )
    args = parser.parse_args()

    thresholds = dict(DEFAULT_THRESHOLDS)
    for override in args.threshold:
        if "=" not in override:
            parser.error(f"--threshold expects LINE=SECONDS, got {override!r}")
        line_name, _, raw = override.partition("=")
        thresholds[line_name] = float(raw)

    window = (args.window[0], args.window[1]) if args.window else None
    convert(args.xplane, args.out, thresholds, window)
    return 0


if __name__ == "__main__":
    sys.exit(main())
