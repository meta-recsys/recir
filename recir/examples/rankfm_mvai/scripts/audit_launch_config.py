#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Deep equivalence check: the 64-chip manifest vs the LaunchConfig render.

The gate for retiring ``manifests/rankfm_mvai-64chip.yaml``. Walks both
JobSets to every leaf and reports each path that differs, is only in the
manifest, or is only in the render, then inventories the capabilities the
manifest's shell body implements and checks the render achieves each one --
by whatever means, since several moved into a sidecar.

    # render the candidate first
    mvai-cloud run --hardware tpu-v7 -j 16x8 --launch-config rankfm-prod \
        --dry-run | sed -n '/^apiVersion:/,$p' > /tmp/new-16x8.yaml

    python3 recir/examples/rankfm_mvai/scripts/audit_launch_config.py \
        --manifest recir/examples/rankfm_mvai/manifests/rankfm_mvai-64chip.yaml \
        --render /tmp/new-16x8.yaml

Exits non-zero while any capability is still lost. The structural section is
informational: several differences there are the point of the refactor (the
prod config moved from a ConfigMap into the image, the image tag is per-job),
so it is read rather than asserted.

Needs PyYAML, which the launcher venv does not carry -- run it with a python
that has it, or through the buck test that wraps the capability half
(``//mvai_cloud/tests:test_tpu_template_capabilities``).
"""

import argparse
import re
import sys

import yaml

_ARGS = argparse.ArgumentParser(description=__doc__)
_ARGS.add_argument("--manifest", required=True)
_ARGS.add_argument("--render", required=True)
_OPTS = _ARGS.parse_args()
OLD, NEW = _OPTS.manifest, _OPTS.render


def flatten(node, path="", out=None):
    """Every leaf in the tree, keyed by a readable path.

    Lists of dicts that carry a `name` are keyed by that name rather than by
    index, so a reordered env block does not read as a wall of differences.
    """
    if out is None:
        out = {}
    if isinstance(node, dict):
        for key, value in node.items():
            flatten(value, f"{path}.{key}" if path else key, out)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            key = (
                value["name"] if isinstance(value, dict) and "name" in value else index
            )
            flatten(value, f"{path}[{key}]", out)
    else:
        out[path] = node
    return out


old_doc, new_doc = yaml.safe_load(open(OLD)), yaml.safe_load(open(NEW))


# The container is named differently on the two paths; normalise so the rest of
# the tree lines up and the rename shows once instead of on every nested field.
def normalise(doc, container_name):
    pod = doc["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]["spec"]
    for container in pod["containers"]:
        if container["name"] == container_name:
            container["name"] = "<trainer>"
    return doc


normalise(old_doc, "rankfm-mvai")
normalise(new_doc, "main")

old_flat, new_flat = flatten(old_doc), flatten(new_doc)

# The shell body and the job name are compared separately.
NOISE = re.compile(
    r"containers\[<trainer>\]\.(command|args)|"
    r"metadata\.(name|generateName)|"
    r"initContainers\[tensorboard-uploader\]\.(args|command)"
)

only_old = sorted(k for k in old_flat if k not in new_flat and not NOISE.search(k))
only_new = sorted(k for k in new_flat if k not in old_flat and not NOISE.search(k))
differ = sorted(
    k
    for k in set(old_flat) & set(new_flat)
    if old_flat[k] != new_flat[k] and not NOISE.search(k)
)

print("=" * 78)
print("  A. STRUCTURE — every leaf outside the shell body")
print("=" * 78)
print(f"\n  leaves: manifest={len(old_flat)}  render={len(new_flat)}\n")

print(f"  [{len(differ)}] SAME PATH, DIFFERENT VALUE")
for k in differ:
    print(f"      {k}")
    print(f"          manifest = {old_flat[k]!r}")
    print(f"          render   = {new_flat[k]!r}")

print(f"\n  [{len(only_old)}] ONLY IN THE MANIFEST  (dropped by the refactor)")
for k in only_old:
    print(f"      {k} = {old_flat[k]!r}")

print(f"\n  [{len(only_new)}] ONLY IN THE RENDER  (added by the refactor)")
for k in only_new:
    print(f"      {k} = {new_flat[k]!r}")


def shell_body(doc):
    """The trainer's shell, plus every sidecar's command line.

    A capability counts as present however it is implemented -- the render
    moves several of them out of the trainer's shell and into a sidecar or a
    pipeline, and an audit keyed on the manifest's variable names would report
    those as lost when they are not.
    """
    pod = doc["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]["spec"]
    parts = []
    for container in pod.get("containers", []) + pod.get("initContainers", []):
        parts += container.get("args") or []
        parts += container.get("command") or []
        for env in container.get("env") or []:
            parts.append(f"{env.get('name')}={env.get('value')}")
    return "\n".join(str(p) for p in parts)


old_sh, new_sh = shell_body(old_doc), shell_body(new_doc)

# Each capability lists every implementation that satisfies it. Where the two
# paths implement one differently, the comment says why the alternative counts.
CAPABILITIES = [
    ("mount-ready probe", r"mount-ready probe|metadata\.json"),
    ("resume: own checkpoints", r"OWN_CKPT"),
    # manifest: SEED_CHECKPOINT_JOBSET env. render: --resume-from, which the
    # launcher turns into MVAI_CHECKPOINT_LOAD_DIR pointing at another job.
    (
        "resume: another job",
        r"SEED_CKPT|SEED_CHECKPOINT_JOBSET|MVAI_CHECKPOINT_LOAD_DIR|resume source",
    ),
    # manifest: a `cp` loop backgrounded in the trainer's shell.
    # render: the tensorboard_uploader sidecar on --interval-seconds.
    ("TB sync, periodic", r"TB_SYNC_PID|TB_SRC|interval-seconds"),
    ("TB copy at exit", r"Final TensorBoard|--once"),
    # manifest: redirect into $LOCAL_LOG. render: `| tee /tmp/train.log`.
    ("trainer stdout -> file", r"LOCAL_LOG|tee \S*train\.log"),
    # manifest: `tail -f` the file so kubectl logs sees it. render: the trainer
    # writes to the container's own stdout through the tee, so it is already
    # there.
    ("trainer stdout -> kubectl logs", r"tail -f|tee \S*train\.log"),
    ("mem probe (.mem.log)", r"LOCAL_MEM|meminfo|--mem-log|pod_diagnostics"),
    ("gcsfuse probe (.fs.log)", r"LOCAL_FS|\.probe\.|--fs-log|pod_diagnostics"),
    ("log sync: every rank", r"LOGTAG|rank\$\{RANK\}|pod_diagnostics"),
    ("log sync: incremental", r"LOGSYNC_PID|LOG_SENT|pod_diagnostics"),
    ("SIGTERM/INT/HUP trap", r"flush_on_signal|trap "),
    ("trainer backgrounded + wait", r'wait "\$TRAIN_PID"'),
    ("exit-code record (.exit)", r"\.exit\b"),
    ("signal record (.signal)", r"\.signal\b"),
    ("error grep (.err.log)", r"\.err\.log|pod_diagnostics"),
]

print()
print("=" * 78)
print("  B. SHELL BODY — capability inventory")
print("=" * 78)
print(f"\n  {'capability':<32} {'manifest':<10} {'render':<10} verdict\n")
lost = []
for label, pattern in CAPABILITIES:
    in_old = bool(re.search(pattern, old_sh))
    in_new = bool(re.search(pattern, new_sh))
    if in_old and not in_new:
        verdict = "*** LOST ***"
        lost.append(label)
    elif in_new and not in_old:
        verdict = "added"
    elif in_old:
        verdict = "kept"
    else:
        verdict = "n/a both"
    print(f"  {label:<32} {str(in_old):<10} {str(in_new):<10} {verdict}")

print()
print("=" * 78)
print(
    f"  SUMMARY: {len(differ)} changed, {len(only_old)} dropped, "
    f"{len(only_new)} added, {len(lost)} shell capabilities LOST"
)
print("=" * 78)
sys.exit(1 if (lost or only_old) else 0)
