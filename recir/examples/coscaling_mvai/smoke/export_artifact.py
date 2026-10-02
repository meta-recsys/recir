# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Build the CoScaling config + batch on the devserver and write the artifact
the TPU image loads (see ``artifact.py``). The image build runs it as the
coscaling-smoke launch config's prod config export target.

    buck2 run @fbcode//mode/opt fbcode//recir/examples/coscaling_mvai/smoke:export_artifact \\
        -- --out /tmp/coscaling_artifact.pt

Knobs as for the CPU smoke: --batch-path, --max-rows, --max-seq (0: no
truncation, and the encoder's static bound is sized for prod). The image build
also passes --source-revision / --source-dirty; without them the revision is
read from the working copy.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
from typing import Any, Dict, Optional, Tuple

from recir.examples.coscaling_mvai.smoke.artifact import save_artifact
from recir.examples.coscaling_mvai.smoke.config import (
    build_batch,
    build_coscaling_config,
    DEFAULT_BATCH_PATH,
)

logger: logging.Logger = logging.getLogger(__name__)


def _source_revision() -> Tuple[str, bool]:
    try:
        rev = subprocess.run(
            ["sl", "log", "-r", ".", "-T", "{node|short}"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["sl", "status", "-mard", "path:fbcode/recir", "path:fbcode/hammer"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        return rev, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def build(
    batch_path: str,
    max_rows: int,
    max_seq: Optional[int],
    source: Optional[Tuple[str, bool]] = None,
) -> Tuple[Any, Any, Any, Any, Dict[str, Any]]:
    module_config, feature_config, prod = build_coscaling_config(
        max_rows, max_uih_len=max_seq
    )
    batch = build_batch(prod, batch_path, max_rows, max_seq)
    rev, dirty = source if source is not None else _source_revision()
    provenance: Dict[str, Any] = {
        "source_revision": rev,
        "source_dirty": dirty,
        "batch_path": batch_path,
        "max_rows": max_rows,
        "max_seq": max_seq,
        "static_max_ro_len": module_config.coscaling_static_max_ro_len,
        "static_max_num_targets": module_config.coscaling_static_max_num_targets,
    }
    return module_config, feature_config, prod.optimizer_config, batch, provenance


def main() -> None:
    # force: an import above may already have configured the root logger.
    logging.basicConfig(level=logging.INFO, force=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-path", default=DEFAULT_BATCH_PATH)
    parser.add_argument("--max-rows", type=int, default=4096)
    parser.add_argument("--max-seq", type=int, default=64)
    parser.add_argument("--source-revision", default="")
    parser.add_argument("--source-dirty", action="store_true")
    args = parser.parse_args()
    max_seq = args.max_seq if args.max_seq > 0 else None
    source = (args.source_revision, args.source_dirty) if args.source_revision else None
    save_artifact(args.out, *build(args.batch_path, args.max_rows, max_seq, source))


if __name__ == "__main__":
    main()
