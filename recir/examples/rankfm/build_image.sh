#!/bin/bash
# Build the RankFM TPU image with plain docker (no fbcode credentials needed).
# Assumes you have local clones of recir, hammer/v3, and minimal_viable_ai
# already on disk; this script just stages them into a build context.
#
# Override paths with env vars:
#   RECIR_DIR=/path/to/recir HAMMER_DIR=/path/to/hammer MVAI_DIR=/path/to/mvai_cloud/minimal_viable_ai \
#       ./build_image.sh

set -euo pipefail

PROJECT_ID=mrs-tpu-y317h
REGION=us-central1
REPO=recml
IMAGE_NAME=rankfm_tpu
IMAGE_TAG="${IMAGE_TAG:-latest}"
IMAGE_PATH="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE_NAME}:${IMAGE_TAG}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIR_DIR="${RECIR_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
FBCODE_DIR="$(cd "${RECIR_DIR}/.." && pwd)"
HAMMER_DIR="${HAMMER_DIR:-${FBCODE_DIR}/hammer}"
MVAI_DIR="${MVAI_DIR:-${FBCODE_DIR}/mvai_cloud/minimal_viable_ai}"
STAGING_DIR="$(mktemp -d)"

cleanup_staging() {
    rm -rf "$STAGING_DIR"
}
trap cleanup_staging EXIT

echo "Staging build context..."
RSYNC_EXCLUDES=(
    --exclude=.git
    --exclude=__pycache__
    --exclude='*.pyc'
    --exclude=tests
    --exclude=test
)

# Keep recir/tests/xla/xsurface_test_utils.py (imported by train.py) but
# exclude every other test file. Includes MUST precede excludes — rsync's
# first-match-wins evaluation order means a bare `--exclude=tests` would
# prune the whole tree before the includes are seen.
rsync -a \
    --include='tests/' \
    --include='tests/__init__.py' \
    --include='tests/xla/' \
    --include='tests/xla/__init__.py' \
    --include='tests/xla/xsurface_test_utils.py' \
    --exclude='tests/**' \
    --exclude='test/' \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    "${RECIR_DIR}/" "${STAGING_DIR}/recir/"
cp "${RECIR_DIR}/github/setup.py" "${STAGING_DIR}/setup.py"

# Stage hammer/ from the AST import-trace manifest (much smaller than
# rsync -a hammer/v3/). The same trace_imports.py used by the OSS-import
# devserver harness produces the manifest -- if a file isn't reachable from
# the rankfm train entry point, it doesn't ship in the image.
TRACE_SCRIPT="${HAMMER_DIR}/v3/oss_test/trace_imports.py"
if [ ! -f "$TRACE_SCRIPT" ]; then
    echo "ERROR: ${TRACE_SCRIPT} not found. Falling back is unsafe; aborting." >&2
    exit 1
fi
TRACE_MANIFEST="$(mktemp)"
HAMMER_FILES="$(mktemp)"
cleanup_trace_files() {
    rm -f "$TRACE_MANIFEST" "$HAMMER_FILES"
}
trap 'cleanup_staging; cleanup_trace_files' EXIT

echo "--- Tracing hammer/v3 imports ---"
FBCODE="${FBCODE_DIR}" python3 "$TRACE_SCRIPT" \
    --fbcode "${FBCODE_DIR}" --out "$TRACE_MANIFEST"

# Filter manifest to hammer/* paths only (recir / minimal_viable_ai stay
# broadly rsynced below) and strip the leading 'fbcode/' so paths are
# relative to FBCODE_DIR for rsync --files-from.
grep '^fbcode/hammer/' "$TRACE_MANIFEST" | sed 's|^fbcode/||' > "$HAMMER_FILES"
echo "  hammer files in trace: $(wc -l < "$HAMMER_FILES")"

mkdir -p "${STAGING_DIR}/hammer"
rsync -a "${RSYNC_EXCLUDES[@]}" --files-from="$HAMMER_FILES" \
    "${FBCODE_DIR}/" "${STAGING_DIR}/"

# trace_imports.py only emits leaf .py files; create empty __init__.py
# markers for any package directories that don't already have one.
touch "${STAGING_DIR}/hammer/__init__.py"
init_created=0
while IFS= read -r dir; do
    [ -z "$dir" ] && continue
    if [ ! -f "${dir}/__init__.py" ]; then
        if find "$dir" -maxdepth 1 -name '*.py' -print -quit | grep -q .; then
            touch "${dir}/__init__.py"
            init_created=$((init_created + 1))
        fi
    fi
done < <(find "${STAGING_DIR}/hammer" -type d)
echo "  created ${init_created} __init__.py markers"

# minimal_viable_ai must be self-sufficient from the OSS clone. Do NOT
# overlay internal mvai — internal core/model_family_api/configs.py and
# core/ranking_common/configs.py drag in aiplatform/hpc/torchrec.fb/
# fbgemm.fb chains. If a missing module breaks the build, fix it by
# adding the file to mvai_cloud/ instead.
rsync -a "${RSYNC_EXCLUDES[@]}" "${MVAI_DIR}/" "${STAGING_DIR}/minimal_viable_ai/"

# generative_recommenders (was a github clone in the Dockerfile before;
# staged from fbsource so the build doesn't need outbound internet)
GENREC_DIR="${GENREC_DIR:-${FBCODE_DIR}/generative_recommenders}"
rsync -a "${RSYNC_EXCLUDES[@]}" "${GENREC_DIR}/" "${STAGING_DIR}/generative_recommenders/"

echo "============================================"
echo "  Building RankFM TPU image"
echo "  Context: ${STAGING_DIR}"
echo "  Image:   ${IMAGE_PATH}"
echo "============================================"

cd "$STAGING_DIR"
docker build -f recir/examples/rankfm/Dockerfile -t "${IMAGE_NAME}:${IMAGE_TAG}" .

echo ""
echo "--- Verifying image ---"
docker run --rm "${IMAGE_NAME}:${IMAGE_TAG}" python3 -c "
import torch; print(f'PyTorch: {torch.__version__}')
import jax; print(f'JAX: {jax.__version__}')
import recir; print('recir: OK')
import torchrec; print('torchrec: OK')
from torchrec.distributed.model_parallel import DistributedModelParallel  # noqa: F401
from torchrec.distributed.embedding import EmbeddingCollectionSharder  # noqa: F401
print('torchrec.distributed: OK')
import hammer.v3; print('hammer.v3: OK')
import minimal_viable_ai; print('minimal_viable_ai: OK')
from generative_recommenders.common import HammerKernel; print('generative_recommenders: OK')
from recir.examples.rankfm.config import RankFMConfig; print('rankfm example: OK')
from recir.tests.xla.xsurface_test_utils import PreprocessorDispatcher, SparseArchDispatcher; print('recir.tests.xla: OK')
print()
print('All imports OK!')
"

echo ""
echo "--- Tagging and pushing ---"
docker tag "${IMAGE_NAME}:${IMAGE_TAG}" "${IMAGE_PATH}"
docker push "${IMAGE_PATH}"

echo ""
echo "============================================"
echo "  BUILD COMPLETE: ${IMAGE_PATH}"
echo "============================================"
