#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Family-level TensorBoard tag diff between two runs.

Answers "does the TPU leg emit the same metric set as the MAST/mvai leg?" by
collapsing the per-task slot out of every tag and diffing the resulting
families. Per-task tags are counted, not listed, so a 1587-tag run reduces to
~35 rows.

Both legs assemble tags from three different schemes, so ``tag_family`` has to
know all three:

  A. ``{ns}/{METRIC}/{scope}/{window}/{task}/{split}``  (shared, RecMetric)
  B. ``{ns}/{scope}/{window}/{task}/{split}``           (MAST, FixedStepLocalTensor)
  C. ``{ns}/{task}/{scope}/{window}/{split}``           (TPU, WindowLoss)

C disappears once the parity work lands; it is kept so the harness stays usable
while the migration is in flight.

Usage::

    buck2 run //recir/examples/rankfm_mvai/scripts:tb_parity -- \\
        --reference manifold://deep_retrieval/tree/jobs/<mast-job>/tensorboard \\
        --candidate /tmp/cpu_eager_run/tensorboard

Paths are either local directories, ``manifold://bucket/path`` (fetched with
``manifold getr`` into a temp dir), or a trainer **log file**. The log form
exists because the smoke manifests (``rankfm_mvai-verify-4chip.yaml``) mount
no GCS output and so write no event file, but their ``ConsoleMetricsLogger``
prints every tag it reports -- exactly what a tag-set diff needs. A 4-step
smoke run cannot produce meaningful metric *values* anyway, so only names and
counts are compared.

Exit code is 1 when the family sets differ, so this can gate a phase.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.util import tensor_util

logger: logging.Logger = logging.getLogger(__name__)

MANIFOLD_PREFIX = "manifold://"

# ConsoleMetricsLogger line format (core/trainer/metrics.py:110-120):
#   "{name} at step {step}: {value}"
# Searched rather than anchored: the logger emits one report as a single
# multi-line record, so under JSON log formatting the FIRST metric of every
# report arrives glued to the record's JSON prefix. Anchoring at ^ silently
# drops exactly one tag per report -- which reads as a missing family.
_LOG_LINE: "re.Pattern[str]" = re.compile(
    r"(?P<tag>[\w/@.|+-]+) at step (?P<step>-?\d+): (?P<value>\S+)"
)

# Namespaces whose tags carry a trailing task slot, keyed by which scheme they
# use. Scheme A is detected from the namespace alone; B and C need the
# namespace AND the position of the scope token, because `loss` is used by both
# MAST (scheme B) and the TPU WindowLoss (scheme C).
_SCHEME_A_NS: Set[str] = {"metrics", "variable_step_metrics"}
_SCHEME_B_NS: Set[str] = {
    "loss",
    "llw_weights",
    "loss_scale",
    "effective_sample_cnt",
    "mrn",
}
_SCOPES: Set[str] = {"global", "local"}
_WINDOWS: Set[str] = {"lifetime", "window", "interval"}

# `llw/w_0` .. `llw/w_104` -- a positional-index family the TPU leg emits when
# the task-name list length does not match the LLW vector.
_INDEXED = re.compile(r"^([ws])_(\d+)$")


@dataclass
class Family:
    """One metric family: a tag with its per-task slot removed."""

    key: str
    tasks: Set[str] = field(default_factory=set)
    n_tags: int = 0
    # Fewest data points across the family's tags. A family present in code but
    # not yet reported at read time shows up as a low count, not as absent.
    min_points: int = 0


def tag_family(tag: str) -> Tuple[str, Optional[str]]:
    """Collapse ``tag`` to ``(family_key, task_or_None)``.

    The task slot is replaced by the literal ``<task>`` so the same metric under
    different task names folds into one row.
    """
    parts = tag.split("/")
    if len(parts) < 3:
        return tag, None
    split = parts[-1]
    ns = parts[0]

    if ns in _SCHEME_A_NS and len(parts) >= 5:
        if parts[2] in _SCOPES and parts[3] in _WINDOWS:
            if len(parts) == 5:
                # {ns}/{METRIC}/{scope}/{window}/{split} -- no task slot, e.g.
                # variable_step_metrics/example_count/global/interval/train.
                # Matched before scheme C so `example_count` is not mistaken for
                # a task name, which would fold two such metrics into one family.
                return tag, None
            # {ns}/{METRIC}/{scope}/{window}/{task...}/{split}
            task = "/".join(parts[4:-1])
            return f"{ns}/{parts[1]}/{parts[2]}/{parts[3]}/<task>/{split}", task

    if ns in _SCHEME_B_NS and len(parts) >= 5:
        # {ns}/{scope}/{window}/{task...}/{split}
        if parts[1] in _SCOPES and parts[2] in _WINDOWS:
            task = "/".join(parts[3:-1])
            return f"{ns}/{parts[1]}/{parts[2]}/<task>/{split}", task

    if len(parts) >= 5 and parts[-3] in _SCOPES and parts[-2] in _WINDOWS:
        # {ns}/{task...}/{scope}/{window}/{split} -- TPU WindowLoss
        task = "/".join(parts[1:-3])
        indexed = _INDEXED.match(parts[1]) if len(parts) == 5 else None
        if indexed:
            return (
                f"{ns}/{indexed.group(1)}_<idx>/{parts[-3]}/{parts[-2]}/{split}",
                indexed.group(2),
            )
        return f"{ns}/<task>/{parts[-3]}/{parts[-2]}/{split}", task

    return tag, None


def read_log_tags(path: str) -> Dict[str, int]:
    """Return ``{tag: n_reports}`` parsed from a trainer log."""
    out: Dict[str, int] = {}
    with open(path, errors="replace") as fh:
        for line in fh:
            m = _LOG_LINE.search(line)
            if m:
                tag = m.group("tag")
                out[tag] = out.get(tag, 0) + 1
    if not out:
        raise ValueError(
            f"{path} contains no '<tag> at step N: <value>' lines; is it a "
            "trainer log with the console metrics logger enabled?"
        )
    return out


def read_tags(path: str) -> Dict[str, int]:
    """Return ``{tag: n_points}`` for every scalar-valued tag under ``path``."""
    if os.path.isfile(path):
        return read_log_tags(path)
    acc = EventAccumulator(path, size_guidance={"scalars": 0, "tensors": 0})
    acc.Reload()
    tags = acc.Tags()
    out: Dict[str, int] = {}
    for tag in tags.get("scalars", []):
        out[tag] = len(acc.Scalars(tag))
    # Newer TensorBoard writers emit scalars as rank-0 tensors.
    for tag in tags.get("tensors", []):
        n = 0
        for event in acc.Tensors(tag):
            arr = tensor_util.make_ndarray(event.tensor_proto)
            if arr.size == 1:
                n += 1
        if n:
            out[tag] = n
    return out


def build_families(tags: Dict[str, int]) -> Dict[str, Family]:
    fams: Dict[str, Family] = {}
    for tag, n_points in tags.items():
        key, task = tag_family(tag)
        fam = fams.get(key)
        if fam is None:
            fam = Family(key=key, min_points=n_points)
            fams[key] = fam
        fam.n_tags += 1
        fam.min_points = min(fam.min_points, n_points)
        if task is not None:
            fam.tasks.add(task)
    return fams


def fetch(path: str, workdir: str) -> str:
    """Resolve ``path`` to a local directory, pulling from Manifold if needed."""
    if not path.startswith(MANIFOLD_PREFIX):
        return path
    remote = path[len(MANIFOLD_PREFIX) :]
    local = f"{workdir}/{remote.replace('/', '_')}"
    logger.info("fetching %s -> %s", path, local)
    subprocess.run(["manifold", "getr", remote, local], check=True)
    return local


def _describe(fam: Family) -> str:
    detail = f"{fam.n_tags} tags"
    if fam.tasks:
        detail += f", {len(fam.tasks)} tasks"
    return f"{detail}, min {fam.min_points} points"


def report(
    ref: Dict[str, Family],
    cand: Dict[str, Family],
    ref_label: str,
    cand_label: str,
    min_points: int,
) -> bool:
    """Print the four lists. Returns True when the family sets are equal."""
    both = sorted(set(ref) & set(cand))
    only_ref = sorted(set(ref) - set(cand))
    only_cand = sorted(set(cand) - set(ref))

    print(f"\n{'=' * 78}")
    print(
        f"reference ({ref_label}): {sum(f.n_tags for f in ref.values())} tags, {len(ref)} families"
    )
    print(
        f"candidate ({cand_label}): {sum(f.n_tags for f in cand.values())} tags, {len(cand)} families"
    )
    print(f"{'=' * 78}")

    print(f"\n## BOTH ({len(both)})")
    count_mismatch: List[str] = []
    for key in both:
        r, c = ref[key], cand[key]
        flag = ""
        if r.n_tags != c.n_tags:
            flag = f"   <-- TAG COUNT DIFFERS ({r.n_tags} vs {c.n_tags})"
            count_mismatch.append(key)
        print(f"  {key}")
        print(f"      {ref_label}: {_describe(r)}{flag}")
        print(f"      {cand_label}: {_describe(c)}")

    print(
        f"\n## ONLY IN {ref_label.upper()} -- candidate is missing these ({len(only_ref)})"
    )
    for key in only_ref:
        print(f"  {key}   [{_describe(ref[key])}]")

    print(
        f"\n## ONLY IN {cand_label.upper()} -- delete or rename these ({len(only_cand)})"
    )
    for key in only_cand:
        print(f"  {key}   [{_describe(cand[key])}]")

    # A family whose tags have very few points may exist in code but not have
    # reported yet. Without this the caller reads a too-short run as a missing
    # family and chases a phantom bug.
    thin = sorted(k for k, f in cand.items() if f.min_points < min_points)
    if thin:
        print(
            f"\n## UNDER-REPORTED in {cand_label} (< {min_points} points) ({len(thin)})"
        )
        print("   Run longer before trusting an absence here.")
        for key in thin:
            print(f"  {key}   [{_describe(cand[key])}]")

    equal = not only_ref and not only_cand and not count_mismatch
    print(f"\n{'=' * 78}")
    if equal:
        print("PARITY: family sets and tag counts match.")
    else:
        print(
            f"PARITY: MISMATCH -- {len(only_ref)} missing, {len(only_cand)} extra, "
            f"{len(count_mismatch)} with differing tag counts."
        )
    print(f"{'=' * 78}")
    return equal


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        required=True,
        help="TB dir the candidate must match (local path or manifold://bucket/path)",
    )
    parser.add_argument(
        "--candidate",
        required=True,
        help="TB dir under test (local path or manifold://bucket/path)",
    )
    parser.add_argument("--reference-label", default="mast")
    parser.add_argument("--candidate-label", default="tpu")
    parser.add_argument(
        "--min-points",
        type=int,
        default=2,
        help="flag candidate families with fewer than this many points per tag",
    )
    args = parser.parse_args(argv)

    workdir = tempfile.mkdtemp(prefix="tb_parity_")
    try:
        ref_dir = fetch(args.reference, workdir)
        cand_dir = fetch(args.candidate, workdir)
        ref = build_families(read_tags(ref_dir))
        cand = build_families(read_tags(cand_dir))
        equal = report(
            ref, cand, args.reference_label, args.candidate_label, args.min_points
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return 0 if equal else 1


if __name__ == "__main__":
    sys.exit(main())
