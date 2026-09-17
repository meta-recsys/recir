#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Value-level TensorBoard comparison between the MAST and TPU legs.

``tb_parity.py`` answers "does the TPU leg emit the same tag NAMES?". This one
answers "does it emit the same NUMBERS?" -- which is a different question and
the one that catches a metric that is present, correctly named, and off by a
factor of the host count.

Two modes:

``--dump``  read one run and write ``{tag: [[step, value], ...]}`` JSON.
``--compare`` diff two runs (dirs, ``manifold://`` paths, or dumps) and print
            the per-family median / max relative difference.

Splitting the read out matters in practice: only the read needs the tensorboard
dep (and therefore buck), a 96 MB event file takes ~60 s to parse, and one dump
then serves any number of analysis passes.

Both legs put CUMULATIVE EXAMPLES on the x-axis for the fixed-step namespaces
(``loss``/``llw_weights``/``loss_scale``/``learning_rate``/``metrics``/
``effective_sample_cnt``), so series line up directly even when the report
intervals differ -- MAST reporting every 500 steps and the TPU leg every 100
simply means the TPU has 5x the points. ``variable_step_*`` advances on its own
counter and ``qps`` on iterations; those are compared where the x values happen
to coincide, which is rare, so treat their rows as indicative only.

Comparison is done at ONE x per tag -- the median common x -- rather than
averaged over the series. Cumulative metrics are heavily autocorrelated, so
averaging a relative difference over 400 points reports a confidence the data
does not carry.

Usage::

    buck2 run //recir/examples/rankfm_mvai/scripts:tb_dump -- --compare \\
        --reference manifold://deep_retrieval/tree/jobs/<mast-job>/tensorboard \\
        --candidate manifold://mrs_recsys_gcp/tree/tensorboard/<tpu-jobset>

    # or, to iterate on the analysis without re-parsing:
    buck2 run //recir/examples/rankfm_mvai/scripts:tb_dump -- \\
        --dump manifold://... --output /tmp/mast.json
    buck2 run //recir/examples/rankfm_mvai/scripts:tb_dump -- --compare \\
        --reference /tmp/mast.json --candidate /tmp/tpu.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Sequence, Tuple

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.util import tensor_util

logger: logging.Logger = logging.getLogger(__name__)

MANIFOLD_PREFIX = "manifold://"

Series = Dict[str, List[Tuple[int, float]]]

# Same three tag schemes ``tb_parity.tag_family`` handles; see its docstring.
_SCHEME_A_NS = {"metrics", "variable_step_metrics"}
_SCOPES = {"global", "local"}
_WINDOWS = {"lifetime", "window", "interval"}

# NE's ``allow_missing_label_with_zero_weight`` sentinel (``ne.py:120-121``):
# emitted when a task has no weighted samples, on BOTH legs. Dividing by it
# manufactures relative differences of order 1e9, which is how an early pass
# produced a "3.5e9% gap".
_NE_ETA = 1e-12


def fetch(path: str, workdir: str) -> str:
    """Resolve ``path`` to a local directory, pulling from Manifold if needed."""
    if not path.startswith(MANIFOLD_PREFIX):
        return path
    remote = path[len(MANIFOLD_PREFIX) :]
    local = f"{workdir}/{remote.replace('/', '_')}"
    logger.info("fetching %s -> %s", path, local)
    subprocess.run(["manifold", "getr", remote, local], check=True)
    return local


def read_series(path: str) -> Series:
    """Return ``{tag: [(step, value), ...]}`` for every scalar-valued tag."""
    acc = EventAccumulator(path, size_guidance={"scalars": 0, "tensors": 0})
    acc.Reload()
    tags = acc.Tags()
    out: Series = {}
    for tag in tags.get("scalars", []):
        out[tag] = [(e.step, float(e.value)) for e in acc.Scalars(tag)]
    # Newer writers emit scalars as rank-0 tensors.
    for tag in tags.get("tensors", []):
        points: List[Tuple[int, float]] = []
        for event in acc.Tensors(tag):
            arr = tensor_util.make_ndarray(event.tensor_proto)
            if arr.size == 1:
                points.append((event.step, float(arr.reshape(-1)[0])))
        if points:
            out.setdefault(tag, []).extend(points)
    return out


def load(path: str, workdir: str) -> Series:
    """Read a JSON dump or a (possibly remote) TensorBoard dir."""
    if os.path.isfile(path):
        with open(path) as fh:
            raw = json.load(fh)
        return {k: [(int(s), float(v)) for s, v in pts] for k, pts in raw.items()}
    return read_series(fetch(path, workdir))


def tag_family(tag: str) -> str:
    """Collapse ``tag``'s per-task slot to ``<task>``."""
    parts = tag.split("/")
    if len(parts) < 5:
        return tag
    if parts[0] in _SCHEME_A_NS and parts[2] in _SCOPES and parts[3] in _WINDOWS:
        if len(parts) == 5:
            return tag
        return f"{parts[0]}/{parts[1]}/{parts[2]}/{parts[3]}/<task>/{parts[-1]}"
    if parts[1] in _SCOPES and parts[2] in _WINDOWS:
        return f"{parts[0]}/{parts[1]}/{parts[2]}/<task>/{parts[-1]}"
    return tag


def compare(ref: Series, cand: Series) -> Dict[str, List[float]]:
    """Signed relative differences ``(cand - ref) / |ref|``, grouped by family.

    One sample per tag, taken at the median common x. Tags with no common x, or
    where both legs sit on the NE sentinel, contribute nothing.
    """
    out: Dict[str, List[float]] = {}
    for tag in sorted(set(ref) & set(cand)):
        a, b = dict(ref[tag]), dict(cand[tag])
        common = sorted(set(a) & set(b))
        if not common:
            continue
        x = common[len(common) // 2]
        av, bv = a[x], b[x]
        if abs(av) <= _NE_ETA and abs(bv) <= _NE_ETA:
            continue
        if av == 0.0:
            continue
        out.setdefault(tag_family(tag), []).append((bv - av) / abs(av))
    return out


def report(ref: Series, cand: Series, ref_label: str, cand_label: str) -> None:
    rows = compare(ref, cand)
    ref_fams = {tag_family(t) for t in ref}
    cand_fams = {tag_family(t) for t in cand}

    print(f"\n{'=' * 100}")
    print(f"reference ({ref_label}): {len(ref)} tags, {len(ref_fams)} families")
    print(f"candidate ({cand_label}): {len(cand)} tags, {len(cand_fams)} families")
    print("relative difference = (candidate - reference) / |reference|, at the")
    print("median common x of each tag")
    print(f"{'=' * 100}")
    print(f"\n{'family':<62}{'n':>5}{'median':>14}{'max |.|':>14}")
    for family in sorted(rows):
        diffs = rows[family]
        print(
            f"{family:<62}{len(diffs):>5}"
            f"{statistics.median(diffs):>14.4g}"
            f"{max(diffs, key=abs):>14.4g}"
        )

    only_ref = sorted(ref_fams - cand_fams)
    only_cand = sorted(cand_fams - ref_fams)
    if only_ref:
        print(f"\nonly in {ref_label} ({len(only_ref)}):")
        for family in only_ref:
            print(f"  {family}")
    if only_cand:
        print(f"\nonly in {cand_label} ({len(only_cand)}):")
        for family in only_cand:
            print(f"  {family}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", help="TB dir or manifold:// path to dump as JSON")
    parser.add_argument("--output", help="JSON destination for --dump")
    parser.add_argument("--compare", action="store_true", help="diff two runs")
    parser.add_argument("--reference", help="TB dir, manifold:// path, or JSON dump")
    parser.add_argument("--candidate", help="TB dir, manifold:// path, or JSON dump")
    parser.add_argument("--reference-label", default="mast")
    parser.add_argument("--candidate-label", default="tpu")
    args = parser.parse_args(argv)

    workdir = tempfile.mkdtemp(prefix="tb_dump_")
    try:
        if args.dump:
            if not args.output:
                parser.error("--dump needs --output")
            series = load(args.dump, workdir)
            with open(args.output, "w") as fh:
                json.dump(series, fh)
            logger.info("wrote %d tags to %s", len(series), args.output)
            return 0
        if args.compare:
            if not (args.reference and args.candidate):
                parser.error("--compare needs --reference and --candidate")
            report(
                load(args.reference, workdir),
                load(args.candidate, workdir),
                args.reference_label,
                args.candidate_label,
            )
            return 0
        parser.error("pass --dump or --compare")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
