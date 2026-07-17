#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""One-shot A/B experiment harness for the rankfm_mvai recir example.

Launches TWO JobSets (e.g. the compiled TPU run and the matched CPU-eager
reference), waits for BOTH to finish, then:
  1. prints a side-by-side per-task NE diff (from each run's stdout), and
  2. copies each run's real TensorBoard events from GCS into ONE Manifold
     logdir (the two runs as overlaid sub-runs) and prints a shareable link.

Why copy from GCS rather than parse stdout for the TB: the in-pod TB export
writes real scalar event files to gs://<bucket>/tensorboard/<jobset>/, but they
are only flushed to GCS when the run COMPLETES (gcsfuse does not reliably flush
the open events file mid-run). So this harness waits for terminal state, then
mirrors the real events GCS -> Manifold (the proven rankfm_mvai_tb_to_manifold
flow). The stdout is used only for the text comparison table.

IMPORTANT: because the TB data must be flushed, the runs must COMPLETE -- this
harness waits for completion (bounded by --max-steps so that finishes in a
reasonable time) instead of time-boxing and killing mid-run.

Usage (defaults compare the 1-shard TPU run vs the matched CPU-eager run):
    python3 compare_experiment.py
    python3 compare_experiment.py \
        --yaml-a rankfm_mvai-1shard.yaml --label-a tpu \
        --yaml-b rankfm_mvai-eager-1shard.yaml --label-b eager \
        --max-steps 2000 --wait-minutes 90

Prereqs: image built/pushed (or pass --build to build+push it first via
build_image_podman.sh -- needs podman + `gcloud auth login`), kubectl on the TPU
cluster (or pass --context / --namespace), the run's data source reachable, and
the TB export writes to gs://<--tb-gcs-bucket>/tensorboard/<jobset>/.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_NE_RE = re.compile(
    r"NE/global/window/([A-Za-z0-9_]+)/train at step (\d+): ([0-9.eE+-]+)"
)
_CREATED_RE = re.compile(r"jobset[^/]*/([A-Za-z0-9-]+) created")


def sh(cmd: List[str], check: bool = True, quiet: bool = False) -> str:
    """Run a command, return stdout (also stderr merged)."""
    if not quiet:
        print(f"  $ {' '.join(cmd)}", file=sys.stderr)
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    if check and p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(cmd)}\n{out}")
    return out


class Run:
    def __init__(self, yaml: str, label: str, ctx: str, ns: str) -> None:
        self.yaml = yaml if os.path.isabs(yaml) else os.path.join(_HERE, yaml)
        self.label = label
        self.ctx = ctx
        self.ns = ns
        self.jobset: Optional[str] = None
        self.pod: Optional[str] = None
        self.container: Optional[str] = None

    def _kube(self, *args: str) -> List[str]:
        return ["kubectl", f"--context={self.ctx}", "-n", self.ns, *args]

    def launch(self) -> None:
        if not os.path.exists(self.yaml):
            raise FileNotFoundError(self.yaml)
        out = sh(self._kube("create", "-f", self.yaml))
        m = _CREATED_RE.search(out)
        if not m:
            raise RuntimeError(
                f"[{self.label}] could not parse jobset name from:\n{out}"
            )
        self.jobset = m.group(1)
        print(f"[{self.label}] launched jobset {self.jobset}")

    def resolve_pod(self) -> bool:
        """Find the worker-0-0 pod for this jobset (returns True once found)."""
        out = sh(self._kube("get", "pods", "-o", "name"), check=False, quiet=True)
        for line in out.splitlines():
            name = line.replace("pod/", "").strip()
            if self.jobset and self.jobset in name and "worker-0-0" in name:
                self.pod = name
                return True
        return False

    def resolve_container(self) -> None:
        if self.pod and not self.container:
            out = sh(
                self._kube(
                    "get",
                    "pod",
                    self.pod,
                    "-o",
                    "jsonpath={.spec.containers[0].name}",
                ),
                check=False,
                quiet=True,
            ).strip()
            # First container is the trainer (gcsfuse sidecar is injected after).
            self.container = out or "rankfm-mvai"

    def phase(self) -> str:
        if not self.pod:
            return "Pending"
        return (
            sh(
                self._kube("get", "pod", self.pod, "-o", "jsonpath={.status.phase}"),
                check=False,
                quiet=True,
            ).strip()
            or "Unknown"
        )

    def terminal(self) -> bool:
        if not self.jobset:
            return False
        st = sh(
            self._kube(
                "get",
                "jobset",
                self.jobset,
                "-o",
                "jsonpath={.status.terminalState}",
            ),
            check=False,
            quiet=True,
        ).strip()
        return bool(st)

    def logs(self) -> str:
        if not self.pod:
            return ""
        self.resolve_container()
        return sh(
            self._kube("logs", self.pod, "-c", self.container or "rankfm-mvai"),
            check=False,
            quiet=True,
        )

    def cleanup(self) -> None:
        if self.jobset:
            sh(self._kube("delete", "jobset", self.jobset), check=False)
            print(f"[{self.label}] deleted jobset {self.jobset}")


def parse_ne(text: str) -> Dict[str, Tuple[int, float]]:
    """task -> (latest_step, ne_value) from the raw stdout log (for the table)."""
    latest: Dict[str, Tuple[int, float]] = {}
    for task, step_s, val_s in _NE_RE.findall(text):
        step = int(step_s)
        try:
            val = float(val_s)
        except ValueError:
            continue
        if task not in latest or step > latest[task][0]:
            latest[task] = (step, val)
    return latest


def progress_line(run: Run, text: str) -> str:
    ne = parse_ne(text)
    like = ne.get("like")
    n_sub1 = sum(1 for _, (_, v) in ne.items() if v < 1.0)
    like_s = f"like={like[1]:.3f}@{like[0]}" if like else "like=--"
    term = "DONE" if run.terminal() else run.phase()
    return f"[{run.label}] {term} tasks={len(ne)} {like_s} (<1.0: {n_sub1})"


def compare(a: Run, b: Run, a_ne, b_ne) -> str:
    tasks = sorted(set(a_ne) & set(b_ne))
    lines: List[str] = []
    w = max((len(t) for t in tasks), default=12)
    header = f"{'task':<{w}}  {a.label:>12}  {b.label:>12}  {'delta(a-b)':>12}"
    lines.append(header)
    lines.append("-" * len(header))
    rows = []
    for t in tasks:
        rows.append((t, a_ne[t][1], b_ne[t][1], a_ne[t][1] - b_ne[t][1]))
    for t, av, bv, d in sorted(rows, key=lambda r: -abs(r[3])):
        lines.append(f"{t:<{w}}  {av:>12.4f}  {bv:>12.4f}  {d:>+12.4f}")

    def summary(label: str, ne) -> str:
        vals = [v for _, (_, v) in ne.items()]
        if not vals:
            return f"  {label}: no metrics"
        return (
            f"  {label}: n={len(vals)} "
            f"min={min(vals):.3f} <1.0={sum(1 for v in vals if v < 1.0)} "
            f"<0.5={sum(1 for v in vals if v < 0.5)} <0.2={sum(1 for v in vals if v < 0.2)}"
        )

    lines.append("")
    lines.append("SUMMARY (latest per-task NE; mean omitted -- rare tasks can blow up)")
    lines.append(summary(a.label, a_ne))
    lines.append(summary(b.label, b_ne))
    a_better = sum(1 for _, av, bv, d in rows if d < -0.05)
    b_better = sum(1 for _, av, bv, d in rows if d > 0.05)
    lines.append(
        f"  common tasks={len(tasks)}  {a.label} better (>0.05): {a_better}  "
        f"{b.label} better (>0.05): {b_better}"
    )
    return "\n".join(lines)


def copy_tb_from_gcs(jobset: str, dest_dir: str, bucket: str) -> int:
    """Mirror a finished run's TB events from GCS into dest_dir.

    Returns total bytes copied (0 if nothing real flushed -- a killed/short run
    leaves only 0/88-byte header files, which we treat as empty).
    """
    os.makedirs(dest_dir, exist_ok=True)
    src = f"gs://{bucket}/tensorboard/{jobset}"
    sh(
        ["gcloud", "storage", "cp", "-r", f"{src}/*", f"{dest_dir}/"],
        check=False,
        quiet=True,
    )
    total = 0
    for dp, _, fs in os.walk(dest_dir):
        for f in fs:
            total += os.path.getsize(os.path.join(dp, f))
    return total


def upload_dir_to_manifold(local_dir: str, dst: str, ttl: int) -> bool:
    """manifold putr a dir; success is decided by EXIT CODE (not substring).

    mrs_recsys_gcp is a user-data bucket with a TTL deletion plan, so the upload
    must pass --userData and --ttl (mirrors rankfm_mvai_tb_to_manifold).
    """
    p = subprocess.run(
        ["manifold", "putr", "--userData=true", f"--ttl={ttl}", local_dir, dst],
        capture_output=True,
        text=True,
    )
    if p.returncode != 0:
        sys.stderr.write((p.stdout or "") + (p.stderr or ""))
    return p.returncode == 0


def tb_link(dst: str) -> str:
    return f"https://www.internalfb.com/intern/tensorboard/?dir=manifold://{dst}"


def build_image(script: str, image_tag: Optional[str]) -> None:
    """Build + push the image (build_image_podman.sh) BEFORE launching.

    Streams output live (the build takes minutes) and raises on nonzero exit so
    the comparison aborts rather than launching against a stale/broken image.
    Pushes IMAGE_TAG (default 'latest') to the rankfm_mvai_tpu Artifact Registry
    repo; the launched yamls must reference that same tag (see --image-tag).
    Needs podman + an authenticated gcloud (`gcloud auth login`) -- the script
    itself fails fast with a clear message if creds are missing.
    """
    if not os.path.exists(script):
        raise FileNotFoundError(f"build script not found: {script}")
    env = dict(os.environ)
    if image_tag:
        env["IMAGE_TAG"] = image_tag
    print(f"=== building image (tag={image_tag or 'latest'}) via {script} ===")
    p = subprocess.run(["bash", script], env=env)  # inherit stdout/stderr (live)
    if p.returncode != 0:
        raise RuntimeError(
            f"image build failed (exit {p.returncode}); aborting before launch. "
            "Check podman and `gcloud auth login`, then see the build log above."
        )
    print("=== image build complete ===")


def patch_yaml(yaml_path: str, max_steps: int, image_tag: Optional[str]) -> str:
    """Apply --max_steps and/or image-tag overrides; return a /tmp copy.

    Returns the original path unchanged if neither override applies (orig file is
    never mutated). image_tag rewrites the rankfm_mvai_tpu image's :tag so the
    run uses the freshly-built/isolated tag instead of the yaml's default.
    """
    txt = open(yaml_path).read()
    changed = False
    if max_steps:
        txt, k = re.subn(r"--max_steps\s+\d+", f"--max_steps {max_steps}", txt)
        changed = changed or k > 0
    if image_tag:
        txt, k = re.subn(r"(rankfm_mvai_tpu):[\w.-]+", rf"\g<1>:{image_tag}", txt)
        changed = changed or k > 0
    if not changed:
        return yaml_path
    out = os.path.join("/tmp", f"_cmp_{os.path.basename(yaml_path)}")
    with open(out, "w") as f:
        f.write(txt)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--yaml-a", default="rankfm_mvai-1shard.yaml")
    ap.add_argument("--label-a", default="tpu")
    ap.add_argument("--yaml-b", default="rankfm_mvai-eager-1shard.yaml")
    ap.add_argument("--label-b", default="eager")
    ap.add_argument(
        "--max-steps",
        type=int,
        default=2000,
        help="patch both yamls' --max_steps so the runs COMPLETE (and flush TB to "
        "GCS) in a bounded time. Set 0 to use each yaml's own value.",
    )
    ap.add_argument(
        "--wait-minutes",
        type=float,
        default=90.0,
        help="max minutes to wait for BOTH runs to reach terminal state",
    )
    ap.add_argument(
        "--build",
        action="store_true",
        help="build + push the image (build_image_podman.sh) BEFORE launching, so "
        "the runs use freshly-built code. Needs podman + `gcloud auth login`; "
        "aborts the comparison if the build fails.",
    )
    ap.add_argument(
        "--build-script",
        default=os.path.join(_HERE, "build_image_podman.sh"),
        help="path to the image build+push script used by --build",
    )
    ap.add_argument(
        "--image-tag",
        default=None,
        help="image tag to build (with --build) and launch (default: latest, "
        "matching the yamls). Set a unique tag to isolate this experiment from a "
        "shared :latest; the launched yamls' rankfm_mvai_tpu image is patched to it.",
    )
    ap.add_argument("--poll-seconds", type=float, default=30.0)
    ap.add_argument(
        "--context",
        default=os.environ.get("CTX")
        or sh(
            ["kubectl", "config", "current-context"], check=False, quiet=True
        ).strip(),
    )
    ap.add_argument("--namespace", default=os.environ.get("NS", "ns-mrs-core-shared"))
    ap.add_argument("--out-dir", default="/tmp/recir_compare")
    ap.add_argument(
        "--no-cleanup",
        action="store_true",
        help="keep the JobSets after comparing (default: delete them)",
    )
    ap.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="skip the GCS->Manifold TensorBoard copy",
    )
    ap.add_argument(
        "--tb-name", default=None, help="Manifold subdir (default cmp-<A-jobset>)"
    )
    ap.add_argument(
        "--tb-gcs-bucket",
        default="cq-mrs-core-shared_mrs_tpu_y317h",
        help="GCS bucket the pods write tensorboard/<jobset>/ into",
    )
    ap.add_argument("--tb-manifold-path", default="mrs_recsys_gcp/tree/tensorboard")
    ap.add_argument("--tb-ttl", type=int, default=2592000)  # 30d
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Optionally (re)build + push the image first, so the runs exercise freshly
    # built code. Gated: a failed build aborts before any JobSet is launched.
    if args.build:
        build_image(args.build_script, args.image_tag)

    ya = args.yaml_a if os.path.isabs(args.yaml_a) else os.path.join(_HERE, args.yaml_a)
    yb = args.yaml_b if os.path.isabs(args.yaml_b) else os.path.join(_HERE, args.yaml_b)
    yaml_a = patch_yaml(ya, args.max_steps, args.image_tag)
    yaml_b = patch_yaml(yb, args.max_steps, args.image_tag)
    if (yaml_a, yaml_b) != (ya, yb):
        print(
            f"patched yamls (max_steps={args.max_steps or 'orig'}, "
            f"image_tag={args.image_tag or 'orig'}) -> {yaml_a}, {yaml_b}"
        )

    a = Run(yaml_a, args.label_a, args.context, args.namespace)
    b = Run(yaml_b, args.label_b, args.context, args.namespace)
    print(f"context={args.context} namespace={args.namespace}")
    a.launch()
    b.launch()

    deadline = time.time() + args.wait_minutes * 60
    a_text = b_text = ""
    both_done = False
    try:
        while time.time() < deadline:
            for r in (a, b):
                if not r.pod:
                    r.resolve_pod()
            a_text = a.logs() if a.pod else ""
            b_text = b.logs() if b.pod else ""
            print(
                f"[t-{int((deadline - time.time()) / 60)}m] "
                f"{progress_line(a, a_text)} | {progress_line(b, b_text)}"
            )
            if a.terminal() and b.terminal():
                both_done = True
                print("both JobSets reached terminal state.")
                break
            time.sleep(args.poll_seconds)

        if not both_done:
            print(
                "\nWARNING: wait window elapsed before both runs completed; their "
                "TB may not be flushed to GCS yet. Re-run with a larger "
                "--wait-minutes or a smaller --max-steps.",
                file=sys.stderr,
            )

        # --- text comparison table from stdout (latest per-task NE) ---
        a_text = a.logs() if (a.resolve_pod() or a.pod) else a_text
        b_text = b.logs() if (b.resolve_pod() or b.pod) else b_text
        for r, txt in ((a, a_text), (b, b_text)):
            with open(os.path.join(args.out_dir, f"{r.label}.log"), "w") as f:
                f.write(txt)
        a_ne, b_ne = parse_ne(a_text), parse_ne(b_text)
        report = compare(a, b, a_ne, b_ne)
        print("\n" + "=" * 72)
        print(f"COMPARISON  A=[{a.label}] {a.jobset}   B=[{b.label}] {b.jobset}")
        print("=" * 72 + "\n" + report)
        with open(os.path.join(args.out_dir, "comparison.txt"), "w") as f:
            f.write(f"A=[{a.label}] {a.jobset}\nB=[{b.label}] {b.jobset}\n\n{report}\n")

        # --- real TensorBoard: copy each run's events GCS -> one Manifold dir ---
        if not args.no_tensorboard:
            exp_name = args.tb_name or f"cmp-{a.jobset or a.label}"
            exp_dir = os.path.join(args.out_dir, exp_name)
            sizes = {}
            for r in (a, b):
                dest = os.path.join(exp_dir, r.label)
                n = copy_tb_from_gcs(r.jobset or "", dest, args.tb_gcs_bucket)
                sizes[r.label] = n
                print(
                    f"[{r.label}] TB from GCS: {n} bytes"
                    + ("" if n > 200 else "  <- empty (not flushed / no export)")
                )
            if any(v > 200 for v in sizes.values()):
                dst = f"{args.tb_manifold_path}/{exp_name}"
                if upload_dir_to_manifold(exp_dir, dst, args.tb_ttl):
                    print("\nTENSORBOARD (both runs overlaid, real events):")
                    print(f"  {tb_link(dst)}")
                else:
                    print(
                        f"manifold upload failed; view locally: tensorboard --logdir {exp_dir}",
                        file=sys.stderr,
                    )
            else:
                print(
                    "\nNo real TB events in GCS for either run -- they likely did "
                    "not complete + flush. Increase --wait-minutes / lower "
                    "--max-steps so the runs finish.",
                    file=sys.stderr,
                )
    finally:
        if not args.no_cleanup:
            a.cleanup()
            b.cleanup()
        else:
            print(f"\n--no-cleanup: kept {a.jobset}, {b.jobset}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
