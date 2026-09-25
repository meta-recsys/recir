#!/usr/bin/env python3
"""Render a run's wall-clock timeline as a standalone HTML chart.

Reads the TensorBoard events files a rankfm_mvai run syncs to Manifold and plots
seconds-per-report-interval against wall time, so the things that stop training
-- XLA compile, checkpoint load, checkpoint save -- show up as what they cost
rather than as a gap in a metric curve.

One events file per attempt: a restart makes the pod write a new one, so each
file is one uninterrupted stretch of training and the boundary between two files
is a restart. The first big gap inside a file is its startup (compile, plus the
checkpoint read when the attempt resumed); later gaps are saves or stalls.

Only the event's wall_time is read, via a hand-rolled TFRecord walk, so this has
no tensorboard dependency and runs as a plain script:

    python3 tb_timeline.py rankfm-mvai-64chip-gvdd5 -o /tmp/timeline.html

Blocking costs are wall-clock deltas between report points, so they include
everything that stopped training, not just the phase a log line happens to name.
Cross-check anything load-bearing against the run's own `[stage]` /
`Checkpoint staged` lines before quoting it.
"""

import argparse
import glob
import html
import json
import os
import struct
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

MANIFOLD_TB = "mrs_recsys_gcp/tree/tensorboard"
PALETTE = ["#2563eb", "#16a34a", "#d97706", "#9333ea", "#dc2626", "#0891b2", "#ca8a04"]


def fetch(run: str, dest: str) -> None:
    subprocess.run(
        ["manifold", "getr", f"{MANIFOLD_TB}/{run}", dest],
        check=True,
        capture_output=True,
        env={
            **os.environ,
            "HTTPS_PROXY": os.environ.get("HTTPS_PROXY", "http://fwdproxy:8080"),
        },
    )


def wall_times(path: str) -> list[float]:
    """Walk the TFRecord stream and pull wall_time (Event field 1, a double).

    Every record is length + crc + payload + crc, and the payload starts with
    tag byte 0x09 for a 64-bit field 1. Truncated tail records are expected --
    the file is being appended to while we read it -- so stop rather than fail.
    """
    out = []
    with open(path, "rb") as handle:
        blob = handle.read()
    pos = 0
    while pos + 12 <= len(blob):
        length = struct.unpack_from("<Q", blob, pos)[0]
        pos += 12
        if length <= 0 or pos + length + 4 > len(blob):
            break
        record = blob[pos : pos + length]
        pos += length + 4
        if len(record) >= 9 and record[0] == 0x09:
            out.append(struct.unpack_from("<d", record, 1)[0])
    return sorted(out)


def report_points(path: str, cluster_secs: float) -> list[float]:
    """Collapse the hundreds of scalars written per report into one timestamp."""
    stamps = wall_times(path)
    if not stamps:
        return []
    points = [stamps[0]]
    for stamp in stamps[1:]:
        if stamp - points[-1] > cluster_secs:
            points.append(stamp)
    return points


def attempt_label(index: int, resumed: bool) -> str:
    return f"A{index + 1} ({'resume' if resumed else 'fresh'})"


def build(run: str, source_dir: str, cluster_secs: float, stall_factor: float) -> dict:
    files = sorted(
        glob.glob(os.path.join(source_dir, "**", "events*"), recursive=True),
        key=os.path.getmtime,
    )
    attempts = []
    for path in files:
        points = report_points(path, cluster_secs)
        if len(points) < 3:
            continue
        attempts.append(points)
    if not attempts:
        raise SystemExit(f"no usable events files under {source_dir}")
    attempts.sort(key=lambda p: p[0])

    origin = attempts[0][0]
    minutes = lambda t: round((t - origin) / 60, 2)  # noqa: E731

    # Pass one: locate each attempt's startup gap. It is the first outsized gap;
    # everything before it is the logger warming up, not training.
    starts = []
    for points in attempts:
        gaps = [points[i] - points[i - 1] for i in range(1, len(points))]
        head = min(8, len(gaps))
        start_i = max(range(head), key=lambda i: gaps[i])
        starts.append((start_i, gaps))

    # An attempt that reloaded a checkpoint pays for the read on top of the
    # compile every attempt pays, so the cheapest startup observed is a compile
    # with no read and anything well above it read a checkpoint. Self-calibrating
    # beats a hardcoded threshold: compile time moves with the model.
    cheapest = min(gaps[i] for i, gaps in starts)

    series, bands, stalls = [], [], []
    for index, (points, (start_i, gaps)) in enumerate(zip(attempts, starts)):
        resumed = gaps[start_i] > 1.5 * cheapest
        label = attempt_label(index, resumed)

        body = [
            (minutes(points[i + 1]), round(gaps[i]))
            for i in range(start_i + 1, len(gaps))
        ]
        body = [(x, y) for x, y in body if y > cluster_secs]
        if not body:
            continue
        ordered = sorted(y for _, y in body)
        baseline = ordered[len(ordered) // 2]

        bands.append(
            {
                "n": label,
                "x0": minutes(points[start_i]),
                "x1": minutes(points[start_i + 1]),
                "d": round(gaps[start_i] / 60, 1),
                "k": "startup+load" if resumed else "startup",
            }
        )
        series.append({"n": label, "b": baseline, "p": [[x, y] for x, y in body]})
        stalls.extend(
            {"x": x, "d": y - baseline, "a": label}
            for x, y in body
            if y > baseline * stall_factor and y - baseline > 120
        )

    fresh = [b["d"] for b in bands if b["k"] == "startup"]
    resumes = [b["d"] for b in bands if b["k"] == "startup+load"]
    return {
        "run": run,
        "origin": datetime.fromtimestamp(origin, timezone.utc).strftime(
            "%Y-%m-%d %H:%M UTC"
        ),
        "bands": bands,
        "series": series,
        "stalls": stalls,
        "fresh": fresh,
        "resumes": resumes,
        "load_only": (
            round(sum(resumes) / len(resumes) - sum(fresh) / len(fresh), 1)
            if fresh and resumes
            else None
        ),
    }


TEMPLATE = """<!DOCTYPE html>
<meta charset="utf-8">
<title>__RUN__ &mdash; wall-clock timeline</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
 body{background:#fff;color:#111827;font-family:system-ui,-apple-system,sans-serif;margin:0;padding:16px}
 table{width:100%;border-collapse:collapse;font-size:12px;margin-top:14px}
 th{text-align:left;border-bottom:1px solid #e5e7eb;padding:5px 6px}
 td{padding:5px 6px;border-bottom:1px solid #eee;vertical-align:top}
 .sw{display:inline-block;width:11px;height:11px;margin-right:4px}
</style>

<div style="font-size:15px;font-weight:600">__RUN__ &mdash; wall-clock timeline</div>
<div style="font-size:12px;color:#6b7280;margin:4px 0 10px">
 y = wall-clock seconds per report interval, log scale. Source: the <code>wall_time</code> of every report
 point in this run's TensorBoard events files, one file per attempt. x origin = __ORIGIN__.
 Shaded bands are attempt startups; dashed lines are stalls during training.
</div>

<div style="height:350px">
  <canvas id="c" role="img" aria-label="Wall-clock seconds per report interval over time, annotated with attempt startup windows and training stalls">__ALT__</canvas>
</div>

<div id="legend" style="display:flex;flex-wrap:wrap;gap:14px;font-size:12px;margin-top:10px"></div>
<table><thead><tr><th style="width:40%">Event</th><th style="width:22%">Minutes since origin</th><th>Cost</th></tr></thead><tbody id="tb"></tbody></table>
<div id="summary" style="font-size:12px;margin-top:16px"></div>
<div style="font-size:11px;color:#6b7280;margin-top:14px;line-height:1.5">
 Costs are wall-clock deltas between report points, so each includes everything that stopped training in that
 interval, not only the phase a log line names. Cross-check load-bearing numbers against the run's own
 <code>[stage]</code> and <code>Checkpoint staged</code> log lines.
</div>

<script>
const D = __DATA__;
const COL = __PALETTE__;

const bands = {
  id: 'bands',
  beforeDatasetsDraw(ch) {
    const ctx = ch.ctx, a = ch.chartArea, x = ch.scales.x;
    ctx.save();
    D.bands.forEach(function (b) {
      const x0 = x.getPixelForValue(b.x0), x1 = x.getPixelForValue(b.x1);
      ctx.fillStyle = b.k === 'startup+load' ? 'rgba(220,38,38,.16)' : 'rgba(107,114,128,.14)';
      ctx.fillRect(x0, a.top, Math.max(x1 - x0, 1), a.bottom - a.top);
      ctx.fillStyle = '#111827';
      ctx.font = '600 10px system-ui';
      ctx.textAlign = 'center';
      ctx.fillText(b.d + 'min', (x0 + x1) / 2, a.top + 12);
    });
    D.stalls.forEach(function (s) {
      const px = x.getPixelForValue(s.x);
      ctx.strokeStyle = '#dc2626';
      ctx.setLineDash([4, 3]);
      ctx.lineWidth = 1.3;
      ctx.beginPath(); ctx.moveTo(px, a.top); ctx.lineTo(px, a.bottom); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#dc2626';
      ctx.font = '600 10px system-ui';
      ctx.textAlign = 'left';
      ctx.fillText('+' + s.d + 's', px + 3, a.top + 30);
    });
    ctx.restore();
  }
};

new Chart(document.getElementById('c'), {
  type: 'line',
  data: {
    datasets: D.series.map(function (s, i) {
      return {
        label: s.n,
        data: s.p.map(function (p) { return { x: p[0], y: p[1] }; }),
        borderColor: COL[i % COL.length], backgroundColor: COL[i % COL.length],
        borderWidth: 1.6, pointRadius: 1.5, tension: 0
      };
    })
  },
  options: {
    responsive: true, maintainAspectRatio: false, animation: false,
    plugins: {
      legend: { display: false },
      tooltip: { callbacks: { label: function (c) { return c.dataset.label + ': ' + c.parsed.y + 's'; } } }
    },
    scales: {
      x: { type: 'linear', title: { display: true, text: 'minutes since ' + D.origin } },
      y: { type: 'logarithmic', title: { display: true, text: 'seconds per report interval (log)' } }
    }
  },
  plugins: [bands]
});

document.getElementById('legend').innerHTML =
  D.series.map(function (s, i) {
    return '<span><span class="sw" style="background:' + COL[i % COL.length] + '"></span>' + s.n +
           ' <span style="color:#6b7280">(' + s.b + 's)</span></span>';
  }).join('')
  + '<span><span class="sw" style="background:rgba(220,38,38,.3)"></span>startup + checkpoint load</span>'
  + '<span><span class="sw" style="background:rgba(107,114,128,.3)"></span>startup only</span>'
  + '<span><span class="sw" style="height:2px;vertical-align:middle;background:#dc2626"></span>stall</span>';

const rows = [];
D.bands.forEach(function (b) {
  rows.push([b.n + ' startup &mdash; ' + (b.k === 'startup+load' ? 'compile + checkpoint load' : 'compile only'),
             b.x0.toFixed(0) + ' &rarr; ' + b.x1.toFixed(0), b.d + ' min']);
});
D.stalls.forEach(function (s) {
  rows.push(['stall during ' + s.a, s.x.toFixed(0),
             s.d + ' s = ' + (s.d / 60).toFixed(1) + ' min']);
});
rows.sort(function (p, q) { return parseFloat(p[1]) - parseFloat(q[1]); });
document.getElementById('tb').innerHTML = rows.map(function (r) {
  return '<tr><td>' + r[0] + '</td><td>' + r[1] + '</td><td>' + r[2] + '</td></tr>';
}).join('');

if (D.load_only !== null) {
  document.getElementById('summary').innerHTML =
    '<b>Derived:</b> checkpoint load costs about <b>' + D.load_only + ' min</b> &mdash; the mean resumed startup (' +
    D.resumes.join(' / ') + ' min) minus the mean fresh startup (' + D.fresh.join(' / ') + ' min). ' +
    'This holds only while each group is internally consistent; check the spread before quoting it.';
}
</script>
"""


def render(data: dict) -> str:
    alt = (
        f"Timeline for {data['run']}: {len(data['series'])} attempts, "
        f"{len(data['stalls'])} stalls during training. "
        + (
            f"Checkpoint load costs about {data['load_only']} minutes."
            if data["load_only"]
            else ""
        )
    )
    return (
        TEMPLATE.replace("__RUN__", html.escape(data["run"]))
        .replace("__ORIGIN__", html.escape(data["origin"]))
        .replace("__ALT__", html.escape(alt))
        .replace("__DATA__", json.dumps(data, separators=(",", ":")))
        .replace("__PALETTE__", json.dumps(PALETTE))
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run", help="jobset name, e.g. rankfm-mvai-64chip-gvdd5")
    parser.add_argument(
        "-o", "--output", help="output path (default /tmp/<run>-timeline.html)"
    )
    parser.add_argument(
        "--events-dir", help="skip the fetch and read events files from here"
    )
    parser.add_argument(
        "--cluster-secs",
        type=float,
        default=5.0,
        help="collapse events written within this many seconds into one report point",
    )
    parser.add_argument(
        "--stall-factor",
        type=float,
        default=1.5,
        help="flag an interval as a stall at this multiple of the attempt's median",
    )
    args = parser.parse_args()

    output = args.output or f"/tmp/{args.run}-timeline.html"
    with tempfile.TemporaryDirectory() as tmp:
        source = args.events_dir
        if not source:
            source = os.path.join(tmp, "tb")
            fetch(args.run, source)
        data = build(args.run, source, args.cluster_secs, args.stall_factor)

    with open(output, "w") as handle:
        handle.write(render(data))

    print(f"wrote {output}")
    for band, s in zip(data["bands"], data["series"]):
        print(
            f"  {band['n']:<16} startup {band['d']:>5.1f} min   {len(s['p']):>3} points   median {s['b']}s"
        )
    for stall in data["stalls"]:
        print(f"  stall in {stall['a']:<16} at +{stall['x']:.0f} min   {stall['d']}s")
    if data["load_only"] is not None:
        print(f"  derived checkpoint load: {data['load_only']} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
