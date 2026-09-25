#!/usr/bin/env python3
"""Decode a run's `rank*.mem.log` into per-phase allocation and fill rates.

The sampler a job runs (see the `MemAvailKB=... Committed_ASKB=...` loop in
`manifests/rankfm_mvai-64chip.yaml`) writes two /proc/meminfo fields every 15s.
Those two fields mean different things, and the difference is what makes them a
profiler:

  * `Committed_AS` jumps at **mmap** time -- a buffer that is allocated but never
    written still counts. It records allocation events.
  * `MemAvailable` falls only as pages are actually **written**. It records fill
    progress.

So a flat `Committed_AS` with a falling `MemAvailable` is a memcpy in flight, and
the slope of that fall is its achieved bandwidth. This script segments each
rank's timeline on that distinction and prints, per segment, how much was
allocated or filled and how fast.

    python3 analyze_mem_log.py /tmp/rvpf5
    python3 analyze_mem_log.py /tmp/rvpf5 --window 09:11:50-09:40:00

`rank*.tail.log` files alongside the mem logs are used, when present, to find the
checkpoint-flush window and to correlate the segments against the per-entry
`[flush]` timings the trainer itself reports.

Caveats worth keeping in mind before quoting a number:
  * 15s sampling. A phase shorter than that is invisible, and a brief allocation
    peak is seen only if a sample lands inside it -- absence of a spike on one
    rank is not evidence that the rank did not allocate.
  * `MemAvailable` counts reclaimable page cache, so a fill rate derived from it
    is a lower bound on pages actually dirtied.
"""

import argparse
import glob
import os
import re
import sys
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

KB_PER_GIB = 1048576.0

# A Committed_AS move smaller than this is noise from the rest of the process,
# not one of the full-size buffers this is meant to find.
DEFAULT_STEP_MIN_GIB = 5.0
DEFAULT_FLAT_TOL_GIB = 2.0
# Below this, a sample interval is making no real forward progress. Chosen well
# under the ~1 GiB/s a healthy fill sustains and well over sampler jitter.
DEFAULT_STALL_GIB_S = 0.2


class Sample(NamedTuple):
    t: int  # seconds since midnight, unwrapped across day boundaries
    fields: Dict[str, float]  # every k=v on the line, in KB as logged

    @property
    def avail(self) -> float:
        return self.fields.get("MemAvailKB", 0.0) / KB_PER_GIB

    @property
    def committed(self) -> float:
        return self.fields.get("Committed_ASKB", 0.0) / KB_PER_GIB


class Segment(NamedTuple):
    kind: str  # "alloc" | "free" | "fill"
    t0: int
    t1: int
    committed: float  # level during a fill, end level after an alloc/free
    d_committed: float
    d_avail: float
    stalled_s: int

    @property
    def secs(self) -> int:
        return self.t1 - self.t0

    @property
    def rate(self) -> float:
        return self.d_avail / self.secs if self.secs else 0.0


class FlushEntry(NamedTuple):
    secs: float
    gb: float
    name: str


class Flush(NamedTuple):
    end_t: int
    total_s: float
    total_gb: float
    entries: List[FlushEntry]

    @property
    def start_t(self) -> int:
        return self.end_t - int(self.total_s)


def parse_hms(text: str) -> int:
    h, m, s = (int(p) for p in text.split(":"))
    return h * 3600 + m * 60 + s


def fmt_hms(t: int) -> str:
    t %= 86400
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"


def read_mem_log(path: str) -> List[Sample]:
    samples: List[Sample] = []
    day = 0
    prev = -1
    for line in open(path):
        parts = line.split()
        if len(parts) < 2 or ":" not in parts[0]:
            continue
        try:
            t = parse_hms(parts[0])
        except ValueError:
            continue
        # The sampler logs a bare HH:MM:SS, so a run crossing midnight would
        # otherwise read as 24h of negative time deltas.
        if prev >= 0 and t < prev - 3600:
            day += 86400
        prev = t
        fields: Dict[str, float] = {}
        for tok in parts[1:]:
            key, _, val = tok.partition("=")
            try:
                fields[key] = float(val)
            except ValueError:
                continue
        samples.append(Sample(t + day, fields))
    return samples


FLUSH_SUMMARY = re.compile(
    r'"timestamp":"[\d-]+ (\d\d:\d\d:\d\d).*?'
    r"\[flush\] full device->host sync: .*?([\d.]+) GB .*?, ([\d.]+)s "
)
FLUSH_ENTRY = re.compile(r"\[flush\]\s+([\d.]+)s\s+([\d.]+) GB\s+\w+ (.*?)(?:\\n|\"|$)")


def read_flush(path: str) -> Optional[Flush]:
    """The last `[flush]` summary in a tail log, with the entry lines after it."""
    text = open(path, errors="replace").read()
    summaries = list(FLUSH_SUMMARY.finditer(text))
    if not summaries:
        return None
    last = summaries[-1]
    entries = [
        FlushEntry(float(m.group(1)), float(m.group(2)), m.group(3).strip())
        for m in FLUSH_ENTRY.finditer(text[last.end() :])
    ]
    return Flush(
        end_t=parse_hms(last.group(1)),
        total_s=float(last.group(3)),
        total_gb=float(last.group(2)),
        entries=entries,
    )


def segment(
    samples: Sequence[Sample],
    step_min: float,
    flat_tol: float,
    stall: float,
) -> List[Segment]:
    """Split a timeline into allocation moves and the fills between them."""
    out: List[Segment] = []
    pending: List[Tuple[Sample, Sample]] = []

    def flush_pending() -> None:
        if not pending:
            return
        a, b = pending[0][0], pending[-1][1]
        stalled = sum(
            (y.t - x.t)
            for x, y in pending
            if y.t > x.t and (x.avail - y.avail) / (y.t - x.t) < stall
        )
        out.append(
            Segment(
                "fill",
                a.t,
                b.t,
                a.committed,
                b.committed - a.committed,
                a.avail - b.avail,
                stalled,
            )
        )
        pending.clear()

    for x, y in zip(samples, samples[1:]):
        dc = y.committed - x.committed
        if abs(dc) <= flat_tol:
            pending.append((x, y))
            continue
        flush_pending()
        out.append(
            Segment(
                "alloc" if dc > 0 else "free",
                x.t,
                y.t,
                y.committed,
                dc,
                x.avail - y.avail,
                0,
            )
        )
    flush_pending()
    return [s for s in out if s.kind != "fill" or s.secs > 0]


def rank_of(path: str) -> int:
    m = re.search(r"rank(\d+)\.", os.path.basename(path))
    return int(m.group(1)) if m else -1


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
    return num / den if den else float("nan")


def print_rank(
    rank: int,
    segs: Sequence[Segment],
    flush: Optional[Flush],
    step_min: float,
    verbose: bool,
) -> None:
    head = f"===== rank{rank}"
    head += (
        f"  flush {flush.total_gb:.1f} GB in {flush.total_s:.1f}s"
        if flush
        else "  (no [flush] line -- never finished)"
    )
    print(head + " =====")
    if flush and flush.entries:
        for e in flush.entries[:3]:
            print(f"    [flush] {e.secs:8.1f}s {e.gb:7.1f} GB  {e.name[-70:]}")
    shown = [s for s in segs if s.kind == "fill" or abs(s.d_committed) >= step_min]
    for s in shown:
        if s.kind == "fill":
            # An idle process holds Committed_AS flat too; without a floor on
            # what was written every gap between allocations reads as a stall.
            if not verbose and (s.secs < 30 or s.d_avail < 1.0):
                continue
            flag = f"  STALLED {s.stalled_s}s" if s.stalled_s else ""
            print(
                f"  {fmt_hms(s.t0)}  fill  {s.secs:5d}s  @{s.committed:7.1f} GiB  "
                f"wrote {s.d_avail:6.1f} GiB  {s.rate:5.2f} GiB/s{flag}"
            )
        else:
            print(
                f"  {fmt_hms(s.t0)}  {s.kind:<5} {'':5}   "
                f"{s.d_committed:+8.1f} GiB -> {s.committed:7.1f} GiB"
            )
    print()


def bucket_rates(
    window: Sequence[Sample],
    segs: Sequence[Segment],
    step_min: float,
    finished: bool,
    into: Dict[int, List[float]],
) -> None:
    """Accumulate fill rates, keyed by the Committed_AS level they happened at.

    Restricted to the span the flush's own allocations bracket: idle and stalled
    are indistinguishable in these two fields, so bucketing the whole window
    would fold the collective wait after the flush into the slow tail.
    """
    moves = [s for s in segs if s.kind != "fill" and abs(s.d_committed) >= step_min]
    if not moves:
        return
    # A rank with no [flush] summary never finished, so there is no closing free
    # and everything after its last allocation is still the operation.
    lo = moves[0].t1
    hi = moves[-1].t1 if finished else window[-1].t
    for x, y in zip(window, window[1:]):
        dt = y.t - x.t
        # Frees release pages faster than any fill writes them; they are not a
        # rate this is trying to characterise.
        if dt <= 0 or dt > 60 or x.avail - y.avail < -0.5:
            continue
        if not lo <= x.t < hi:
            continue
        into.setdefault(int(x.committed // 50 * 50), []).append(
            (x.avail - y.avail) / dt
        )


def print_cross_rank(
    longest: Dict[int, Segment],
    flushes: Dict[int, Flush],
    buckets: Dict[int, List[float]],
    stall: float,
) -> None:
    if longest:
        print("Longest fill per rank (the phase that dominates a big flush)")
        print(
            f"{'rank':>5} {'secs':>6} {'@GiB':>8} {'wrote':>7} {'GiB/s':>7} "
            f"{'stalled':>8}  slowest [flush] entry"
        )
        for rank in sorted(longest, key=lambda r: -longest[r].secs):
            s = longest[rank]
            flush = flushes.get(rank)
            top = flush.entries[0] if flush and flush.entries else None
            note = f"{top.secs:.1f}s {top.gb:.0f} GB" if top else ""
            print(
                f"{rank:>5} {s.secs:>6} {s.committed:>8.1f} {s.d_avail:>7.1f} "
                f"{s.rate:>7.3f} {s.stalled_s:>8}  {note}"
            )
        print()

    if buckets:
        print(
            "Fill rate vs Committed_AS at the time (all ranks pooled, flush span only)"
        )
        print(f"{'Committed_AS':>16} {'n':>6} {'median GiB/s':>13} {'stalled':>9}")
        for b in sorted(buckets):
            v = sorted(buckets[b])
            if len(v) < 10:
                continue
            med = v[len(v) // 2]
            frac = sum(1 for x in v if x < stall) / len(v)
            print(
                f"{b:>6}-{b + 50:<9} {len(v):>6} {med:>13.3f} {frac:>8.0%}  "
                + "#" * int(med * 30)
            )
        print()

    common = [r for r in longest if r in flushes and flushes[r].entries]
    if len(common) >= 3:
        fill = [float(longest[r].secs) for r in common]
        entry = [flushes[r].entries[0].secs for r in common]
        total = [flushes[r].total_s for r in common]
        print(
            f"n={len(common)}  "
            f"pearson(longest fill, slowest entry) = {pearson(fill, entry):+.3f}   "
            f"pearson(longest fill, whole flush) = {pearson(fill, total):+.3f}"
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("logdir", help="directory holding rank*.mem.log")
    ap.add_argument(
        "--window",
        help="HH:MM:SS-HH:MM:SS; default is each rank's "
        "flush window from its tail.log, else all",
    )
    ap.add_argument("--step-min", type=float, default=DEFAULT_STEP_MIN_GIB)
    ap.add_argument("--flat-tol", type=float, default=DEFAULT_FLAT_TOL_GIB)
    ap.add_argument("--stall", type=float, default=DEFAULT_STALL_GIB_S)
    ap.add_argument(
        "-v", "--verbose", action="store_true", help="show fills shorter than 30s too"
    )
    args = ap.parse_args(argv)

    paths = sorted(glob.glob(os.path.join(args.logdir, "rank*.mem.log")), key=rank_of)
    if not paths:
        print(f"no rank*.mem.log under {args.logdir}", file=sys.stderr)
        return 1

    win = None
    if args.window:
        lo, _, hi = args.window.partition("-")
        win = (parse_hms(lo), parse_hms(hi))

    longest: Dict[int, Segment] = {}
    flushes: Dict[int, Flush] = {}
    buckets: Dict[int, List[float]] = {}

    everything: Dict[int, List[Sample]] = {}
    for path in paths:
        rank = rank_of(path)
        samples = read_mem_log(path)
        if not samples:
            continue
        everything[rank] = samples
        tails = glob.glob(os.path.join(args.logdir, f"rank{rank}.*.tail.log"))
        flush = read_flush(tails[0]) if tails else None
        if flush:
            flushes[rank] = flush

    # A rank that hung mid-flush prints no summary line, so it has no window of
    # its own -- and it is the one worth looking at. The flush is entered from a
    # collective, so borrow the window from the ranks that did report.
    fallback = None
    if flushes:
        fallback = (
            min(f.start_t for f in flushes.values()) - 60,
            max(f.end_t for f in flushes.values()) + 60,
        )

    for rank, samples in sorted(everything.items()):
        flush = flushes.get(rank)
        lo, hi = win or (
            (flush.start_t - 60, flush.end_t + 60)
            if flush
            else fallback or (samples[0].t, samples[-1].t)
        )
        w = [s for s in samples if lo <= s.t % 86400 <= hi]
        if len(w) < 3:
            continue

        segs = segment(w, args.step_min, args.flat_tol, args.stall)
        print_rank(rank, segs, flush, args.step_min, args.verbose)

        fills = [s for s in segs if s.kind == "fill"]
        if fills:
            longest[rank] = max(fills, key=lambda s: s.secs)
        bucket_rates(w, segs, args.step_min, flush is not None, buckets)

    print_cross_rank(longest, flushes, buckets, args.stall)
    return 0


if __name__ == "__main__":
    sys.exit(main())
