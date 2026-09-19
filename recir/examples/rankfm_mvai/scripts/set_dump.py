#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Point the whole example at a new preproc dump, in one command.

Give it the two paths the dump is published under and it does the rest: reads
the dump's own ``metadata.json`` for the shape, rewrites ``dump.py``, rewrites
every manifest's ``sourcePath``, and rewrites the light leg's replay provider in
``hammer/``.

    python3 recir/examples/rankfm_mvai/scripts/set_dump.py \\
        --manifold manifold://mrs_recsys_gcp/tree/czheo_test/udd_fm_3d_20260824 \\
        --gcs 'gs://mrs-tpu-y317h/czheo_test/udd_fm_3d_20260824/dataporter_2:prod:dataporter:xdb.mwfs_orchestration_prod:4070:467780:v0/'

The manifests keep a literal ``sourcePath`` rather than reading a variable,
because they have to stay directly submittable with ``kubectl create -f``. The
light leg's provider keeps literals for a different reason: its XAR is built
from the overlays in ``blue_reels_vdd/standalone/common_deps.sh``, which have no
recir, so importing ``dump.py`` there is a ModuleNotFoundError on every trainer
rank. That is why this exists: the same string otherwise lives in every manifest
that mounts the dump, and a rotation was six hand-edits that were easy to do
incompletely. The shape is read rather than typed for the same reason -- a
rotation that changes the rank count silently invalidates every comment and doc
quoting the old one.

    --check   verify dump.py, the manifests and the light provider agree, write
              nothing, exit non-zero on drift. Takes no paths; suitable for CI.

Stdlib only, so it runs under plain python3 with no buck target.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

_PKG = Path(__file__).resolve().parent.parent
_DUMP_PY = _PKG / "dump.py"
_REPLAY_PY = (
    _PKG.parents[2] / "hammer/v3/models/udd_fm_v3/launch/replay_trainer_config.py"
)

# Matches any dataporter sourcePath, so a rotation is found regardless of which
# bucket, dump directory or transfer id the previous one used. Matching the old
# value instead would mean having to know it first.
_SOURCE_PATH = re.compile(r'(sourcePath:\s*")gs://[^"]*dataporter_2:[^"]*(")')
# The Manifold path appears in manifest comments as the layout's source of truth.
_MANIFOLD_IN_YAML = re.compile(r"manifold://mrs_recsys_gcp/tree/[A-Za-z0-9_/]+")

_GCS_RE = re.compile(
    r"^gs://(?P<bucket>[^/]+)/(?P<prefix>.+?)/(?P<dump>[^/]+)/(?P<xfer>dataporter_2:[^/]+)/?$"
)

# Manifests whose --max_steps means "one pass over the dump", so it has to be
# re-derived on a rotation. Every other manifest sets --max_steps as a
# deliberate short cap -- bench 30-60, parity 4-300, smoke 20 -- and rewriting
# those would destroy what they are for. The distinction is intent, which the
# file cannot express, so it is listed rather than detected.
_EPOCH_MANIFESTS: frozenset[str] = frozenset(
    {
        "rankfm_mvai-128chip.yaml",
        "rankfm_mvai-128chip-nodropout.yaml",
        "rankfm_mvai-64chip.yaml",
    }
)

_MAX_STEPS = re.compile(r"^(\s*--max_steps\s+)(\d+)\s*$", re.M)
_PARALLELISM = re.compile(r"^\s*parallelism:\s*(\d+)\s*$", re.M)
# Anchored to a whole line so a mention inside a comment is not mistaken for
# the flag; the command block is one flag per line.
_LOCAL_BATCH = re.compile(r"^\s*--local_batch_size\s+(\d+)\s*$", re.M)


def _epoch_steps(text: str, total_rows: int, name: str) -> int:
    """Steps for one pass over the dump at this manifest's global batch.

    ``--max_steps`` is consumed as a STEP count (``data_pipe.py``'s
    ``for step in range(max_steps)``), not as the per-rank file count it used
    to be set to; the two differ by ``local_batch_size / dump batch_size``.
    One pass is therefore ``total_rows / (parallelism * local_batch_size)``,
    which comes out the same for every chip count here because halving the
    chips doubles the local batch.

    Each field must appear exactly once. A jobset with two replicated jobs
    would make "the" parallelism ambiguous, and guessing which one owns the
    trainer would silently produce a wrong epoch.
    """
    pars, lbs = _PARALLELISM.findall(text), _LOCAL_BATCH.findall(text)
    if len(pars) != 1 or len(lbs) != 1:
        raise ValueError(
            f"{name}: need exactly one parallelism and one --local_batch_size "
            f"to derive --max_steps, found {len(pars)} and {len(lbs)}"
        )
    global_batch = int(pars[0]) * int(lbs[0])
    if total_rows % global_batch:
        raise ValueError(
            f"{name}: dump has {total_rows} rows, which is not a whole number "
            f"of {global_batch}-row global batches"
        )
    return total_rows // global_batch


def _load_dump_constants() -> dict[str, object]:
    """Read dump.py without importing the package (no buck target needed)."""
    ns: dict[str, object] = {}
    exec(compile(_DUMP_PY.read_text(), "dump.py", "exec"), ns)
    return ns


def _const_int(consts: dict[str, object], name: str) -> int:
    """One integer constant out of the exec'd dump.py.

    The namespace is genuinely untyped -- it is whatever that file happens to
    define -- so the check is a real boundary, not a formality: a constant
    turned into a string by a bad rotation would otherwise flow into the
    arithmetic and produce a silently wrong step count.
    """
    value = consts[name]
    if not isinstance(value, int):
        raise TypeError(f"dump.py: {name} is {type(value).__name__}, expected int")
    return value


def _fetch_metadata(manifold_root: str) -> dict[str, int]:
    """Read the dump's own metadata.json. This is what makes the shape trusted."""
    src = f"{manifold_root.removeprefix('manifold://')}/metadata.json"
    with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
        proc = subprocess.run(
            ["manifold", "get", src, tmp.name, "--overwrite"],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise SystemExit(f"could not read {src}:\n{proc.stderr.strip()}")
        return json.loads(Path(tmp.name).read_text())


def _sub_const(src: str, name: str, literal: str, where: str = "dump.py") -> str:
    """Replace one constant's value, keeping its comments.

    Handles the parenthesised continuation black produces for long strings::

        GCS_TRANSFER_ID: str = (
            "dataporter_2:..."
        )

    Matching only the first line would leave the closing paren behind as an
    orphan and the module would not import -- which is exactly what happened
    the first time this ran.

    Leading whitespace is allowed so the same substitution reaches the light
    provider's class attributes. Only the first occurrence is replaced: there
    the Hive control provider restates ``BATCH_SIZE`` by reading it off the
    replay provider, and rewriting that line would break the link.
    """
    pat_single = re.compile(rf"^([ \t]*{name}: \w+ = )(?!\().*$", re.M)
    pat_block = re.compile(rf"^([ \t]*{name}: \w+ = )\([^)]*\)", re.M)
    if pat_single.search(src):
        return pat_single.sub(rf"\g<1>{literal}", src, count=1)
    if pat_block.search(src):
        return pat_block.sub(rf"\g<1>{literal}", src, count=1)
    raise SystemExit(f"{name} not found in {where}; was it renamed?")


def _plan_manifests(
    gcs_root: str, manifold_root: str, total_rows: int
) -> list[tuple[Path, str]]:
    """Each manifest that needs rewriting, with its new text. Writes nothing.

    ``_epoch_steps`` refuses rather than guesses, and a manifest it cannot read
    a global batch out of is only discovered here. Planning every file before
    any is written is what keeps that refusal from landing halfway through a
    rotation -- a half-applied one is the state this whole script exists to
    prevent.
    """
    planned = []
    for path in sorted((_PKG / "manifests").glob("*.yaml")):
        original = path.read_text()
        updated = _SOURCE_PATH.sub(rf"\g<1>{gcs_root}\g<2>", original)
        updated = _MANIFOLD_IN_YAML.sub(manifold_root, updated)
        if path.name in _EPOCH_MANIFESTS:
            steps = _epoch_steps(updated, total_rows, path.name)
            updated = _MAX_STEPS.sub(rf"\g<1>{steps}", updated)
        if updated != original:
            planned.append((path, updated))
    return planned


def _plan_replay_provider(manifold_root: str, batch_size: int) -> str | None:
    """The light leg's config provider pointed at the same dump, or None."""
    original = _REPLAY_PY.read_text()
    updated = _sub_const(original, "BATCH_SIZE", str(batch_size), _REPLAY_PY.name)
    updated = _sub_const(updated, "REPLAY_PATH", f'"{manifold_root}"', _REPLAY_PY.name)
    return None if updated == original else updated


def _do_check() -> int:
    d = _load_dump_constants()
    problems: list[str] = []

    total_rows = (
        _const_int(d, "NUM_RANKS")
        * _const_int(d, "NUM_BATCHES")
        * _const_int(d, "BATCH_SIZE")
    )
    stale = _plan_manifests(str(d["GCS_ROOT"]), str(d["MANIFOLD_ROOT"]), total_rows)
    problems += [f"manifest out of date with dump.py: {p.name}" for p, _ in stale]

    if (
        _plan_replay_provider(str(d["MANIFOLD_ROOT"]), _const_int(d, "BATCH_SIZE"))
        is not None
    ):
        problems.append(f"{_REPLAY_PY.name} out of date with dump.py")

    meta = _fetch_metadata(str(d["MANIFOLD_ROOT"]))
    for key, const in (
        ("num_ranks", "NUM_RANKS"),
        ("num_batches", "NUM_BATCHES"),
        ("batch_size", "BATCH_SIZE"),
    ):
        if meta.get(key) != d[const]:
            problems.append(
                f"{const}: dump.py says {d[const]}, the dump says {meta.get(key)}"
            )

    if problems:
        print("drift detected:")
        for p in problems:
            print(f"  {p}")
        print("\nfix: rerun set_dump.py with --manifold/--gcs")
        return 1
    print(f"in sync: {d['DUMP_NAME']} ({meta})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--manifold", help="manifold:// root of the dump")
    ap.add_argument("--gcs", help="gs:// root of the BarnFS/Dataporter copy")
    ap.add_argument(
        "--check",
        action="store_true",
        help=(
            "verify dump.py, the manifests and the light provider against the "
            "dump; write nothing"
        ),
    )
    args = ap.parse_args()

    if args.check:
        if args.manifold or args.gcs:
            ap.error("--check takes no paths")
        return _do_check()
    if not (args.manifold and args.gcs):
        ap.error("give both --manifold and --gcs (or --check)")

    manifold_root = args.manifold.rstrip("/")
    m = _GCS_RE.match(args.gcs)
    if not m:
        ap.error(
            "--gcs must look like "
            "gs://<bucket>/<prefix>/<dump>/dataporter_2:...:v0/\n"
            f"got: {args.gcs}"
        )
    gcs_root = args.gcs if args.gcs.endswith("/") else args.gcs + "/"

    # The dump directory name has to agree across the two publications. A
    # mismatch means one of the paths is from a different rotation, which would
    # otherwise show up as the two legs quietly reading different data.
    dump_name = m.group("dump")
    if manifold_root.rsplit("/", 1)[-1] != dump_name:
        ap.error(
            f"the two paths name different dumps: manifold says "
            f"{manifold_root.rsplit('/', 1)[-1]!r}, gcs says {dump_name!r}"
        )

    meta = _fetch_metadata(manifold_root)
    missing = {"num_ranks", "num_batches", "batch_size"} - meta.keys()
    if missing:
        raise SystemExit(f"metadata.json is missing {sorted(missing)}: {meta}")

    src = _DUMP_PY.read_text()
    src = _sub_const(src, "DUMP_NAME", f'"{dump_name}"')
    src = _sub_const(src, "MANIFOLD_PREFIX", f'"{manifold_root.rsplit("/", 1)[0]}"')
    src = _sub_const(src, "GCS_BUCKET", f'"{m.group("bucket")}"')
    src = _sub_const(src, "GCS_PREFIX", f'"{m.group("prefix")}"')
    src = _sub_const(src, "GCS_TRANSFER_ID", f'"{m.group("xfer")}"')
    src = _sub_const(src, "NUM_RANKS", str(meta["num_ranks"]))
    src = _sub_const(src, "NUM_BATCHES", str(meta["num_batches"]))
    src = _sub_const(src, "BATCH_SIZE", str(meta["batch_size"]))

    total_rows = (
        int(meta["num_ranks"]) * int(meta["num_batches"]) * int(meta["batch_size"])
    )
    manifests = _plan_manifests(gcs_root, manifold_root, total_rows)
    replay = _plan_replay_provider(manifold_root, int(meta["batch_size"]))

    # Nothing above this line has written anything. Every rewrite is derived
    # first so that a refusal -- an epoch manifest with no single global batch,
    # a renamed constant -- aborts with the tree untouched instead of leaving
    # dump.py pointed at the new dump and some manifests still at the old one.
    _DUMP_PY.write_text(src)
    print(f"dump.py -> {dump_name} {dict(meta)}")

    if manifests:
        print(f"rewrote {len(manifests)} manifest(s):")
        for path, text in manifests:
            path.write_text(text)
            print(f"  {path.name}")
    else:
        print("manifests already pointed here")

    if replay is not None:
        _REPLAY_PY.write_text(replay)
        print(f"rewrote {_REPLAY_PY.name}")
    else:
        print(f"{_REPLAY_PY.name} already pointed here")

    print("\nremaining by hand: nothing. Commit dump.py, the manifests and")
    print(f"{_REPLAY_PY.relative_to(_PKG.parents[2])}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
