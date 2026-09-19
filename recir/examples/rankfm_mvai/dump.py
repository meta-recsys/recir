# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""The preproc dump both legs of the comparison replay.

GENERATED VALUES -- do not hand-edit. Point the example at a new dump with::

    python3 recir/examples/rankfm_mvai/scripts/set_dump.py \\
        --manifold manifold://... --gcs 'gs://...'

That reads the dump's own ``metadata.json`` for the shape, rewrites the
constants below, and rewrites every manifest's ``sourcePath`` -- the one
command is the whole rotation. ``--check`` verifies all of it and is safe to
run from CI.

Before this existed the paths were copy-pasted across six manifests and two
python modules and the shape was repeated again in comments, tests and the
runbook, so a rotation touched a dozen files and the stale copies were found by
grep rather than by anything that fails.
"""

# The dump's directory name, shared by the Manifold original and the GCS copy.
DUMP_NAME: str = "udd_fm_3d_20260824"

# Manifold holds the original. The light (MAST) leg replays straight from here,
# and it is the source of truth for the layout -- the GCS side shows only
# content-addressed shard dirs, so it cannot be inspected directly.
MANIFOLD_PREFIX: str = "manifold://mrs_recsys_gcp/tree/czheo_test"
MANIFOLD_ROOT: str = f"{MANIFOLD_PREFIX}/{DUMP_NAME}"

# The BarnFS/Dataporter-packed copy the TPU pods mount. The transfer id is
# assigned per packing job, so it changes on every rotation even when the dump
# name does not. The trailing ":v0" is part of that id, not the dump format.
GCS_BUCKET: str = "mrs-tpu-y317h"
GCS_PREFIX: str = "czheo_test"
GCS_TRANSFER_ID: str = (
    "dataporter_2:prod:dataporter:xdb.mwfs_orchestration_prod:4070:467780:v0"
)
GCS_ROOT: str = f"gs://{GCS_BUCKET}/{GCS_PREFIX}/{DUMP_NAME}/{GCS_TRANSFER_ID}/"

# Layout: v0 bucketed, rank_R/batch_S/F.pt.zst.
#
# NUM_RANKS is how many shards exist, so it caps how many pods can read
# distinct data in one step. NUM_BATCHES is per rank.
NUM_RANKS: int = 400
NUM_BATCHES: int = 51316

# Rows per dumped file. Baked into every dumped KJT as its stride, so a
# mismatch breaks the sharded all-to-all rather than failing at load time.
# --local_batch_size is chosen as a multiple of this.
BATCH_SIZE: int = 512
