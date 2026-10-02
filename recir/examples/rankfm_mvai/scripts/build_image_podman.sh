#!/bin/bash
# Build the RankFM-MVAI TPU image from fbsource using podman + with-proxy.
# Driven by the mvai_cloud RecTrainer framework. Stages recir + hammer/v3 +
# mvai_cloud/minimal_viable_ai into a temp
# build context, then builds and pushes to the GCP Artifact Registry.

set -euo pipefail

PROJECT_ID=mrs-tpu-y317h
REGION=us-central1
REPO=recml
IMAGE_NAME=rankfm_mvai_tpu
# Override with: IMAGE_TAG=v1 ./build_image_podman.sh
IMAGE_TAG="${IMAGE_TAG:-latest}"
IMAGE_PATH="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE_NAME}:${IMAGE_TAG}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIR_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
FBCODE_DIR="$(cd "${RECIR_DIR}/.." && pwd)"
STAGING_DIR="$(mktemp -d)"

# Installed before anything stages into it, so the barnfs and prod-config steps
# below -- both of which shell out to buck2 and can fail -- do not leak the
# temp tree.
cleanup_staging() {
    rm -rf "$STAGING_DIR"
}
trap cleanup_staging EXIT

# Validate prerequisites
for cmd in buck2 podman rsync gcloud; do
    if ! command -v "$cmd" > /dev/null 2>&1; then
        echo "Error: '$cmd' not found. Run 'feature install ${cmd/#podman/k8s_tools}' to install it." >&2
        exit 1
    fi
done

echo "--- Building BarnFS wheel ---"
# BARNFS_WHEEL=<path> skips the build and uses that wheel.
#
# The wheel encrypts DCP checkpoint writes, but only for a job that sets
# TORCH_ENABLE_BARNFS=ON -- no manifest in this example does, and no python
# module here imports barnfs. It is still a build prerequisite, and its
# dependency chain has broken twice under the opt config it uses (most recently
# common/system_features failing to link against boost_filesystem, which its
# BUCK does declare -- a version skew, not a missing dep).
#
# Both times the DEFAULT (dev) config linked fine, and the resulting .so is
# equally self-contained, so building it there and passing BARNFS_WHEEL=<path>
# works. SKIP_BARNFS=1 omits it entirely, which is correct for every job here
# and skips buck2 altogether. That choice is recorded in BUILD_INFO.json so a
# pod built that way says so in its log.
BARNFS_STAGE="${STAGING_DIR}/barnfs_wheel"
mkdir -p "$BARNFS_STAGE"
BARNFS_INCLUDED=true
if [ "${SKIP_BARNFS:-0}" = "1" ]; then
    BARNFS_INCLUDED=false
    echo "SKIP_BARNFS=1: building without the BarnFS wheel."
    echo "  DCP checkpoints will be UNENCRYPTED, and TORCH_ENABLE_BARNFS=ON"
    echo "  against this image will fail to import rather than encrypt."
elif [ -n "${BARNFS_WHEEL:-}" ]; then
    if [ ! -f "$BARNFS_WHEEL" ]; then
        echo "ERROR: BARNFS_WHEEL=$BARNFS_WHEEL does not exist" >&2
        exit 1
    fi
    echo "using the wheel from BARNFS_WHEEL: $BARNFS_WHEEL"
    cp "$BARNFS_WHEEL" "$BARNFS_STAGE/"
else
    if ! BARNFS_WHEEL="$(
        cd "$FBCODE_DIR"
        buck2 build @//mode/opt \
            //ai_infra/privacy_security/security/file_vault/python:barnfs.whl \
            --show-full-simple-output
    )" || [ ! -f "$BARNFS_WHEEL" ]; then
        echo "ERROR: could not build the BarnFS wheel under @//mode/opt." >&2
        echo "  Build it in the default config and pass it back in:" >&2
        echo "    buck2 build //ai_infra/privacy_security/security/file_vault/python:barnfs.whl --show-full-simple-output" >&2
        echo "    BARNFS_WHEEL=<that path> $0" >&2
        echo "  Or omit it, which is correct for every job here:" >&2
        echo "    SKIP_BARNFS=1 $0" >&2
        exit 1
    fi
    cp "$BARNFS_WHEEL" "$BARNFS_STAGE/"
fi

echo "--- Exporting prod config artifact ---"
# A model's prod config cannot be built inside the image: //minimal_viable_ai
# is not staged into it. So it is exported here, on the devserver under buck,
# and baked in; the entry point reads it from PROD_CONFIG_ARTIFACT (rankfm:
# config._load_prod_module_config).
#
# Which model's: PROD_CONFIG_EXPORT_TARGET is the exporter (a python_binary
# taking --out / --source-revision / --source-dirty) and
# PROD_CONFIG_CHECK_MODULE the module whose check(path) validates the result
# inside the built image. mvai-cloud sets both from the launch config's
# ProdConfigSpec; the defaults are rankfm's.
#
# Baked rather than mounted as a ConfigMap on purpose. The ConfigMap
# (rankfm-prod-cfg) is a single cluster-wide object shared with the parity
# manifests, which want a DIFFERENT artifact -- whoever submits last wins, and
# the loser gets a run built from someone else's config with no indication.
# An artifact inside the image cannot be overwritten by another submit.
#
# The env var is deliberately NOT set in the Dockerfile: artifact_path() raises
# if PROD_CONFIG_ARTIFACT is set and the file is missing, so a SKIP_PROD_CONFIG
# image would be unusable. The launcher sets it per run, which is also where
# the --prod_faithful decision belongs.
#
#   PROD_CONFIG_ARTIFACT_SRC=<path>  reuse an existing export, skip buck2
#   SKIP_PROD_CONFIG=1               omit it; correct for synthetic and bench
#                                    jobs, which never read it
#
# The "prod_config" directory and "prod_cfg.pt" filename below are duplicated
# from PROD_CONFIG_STAGE_DIR / PROD_CONFIG_ARTIFACT_FILENAME in
# recir/examples/rankfm_mvai/prod_config_spec.py -- bash cannot import it, and
# the Dockerfile's COPY repeats them again. tests/test_prod_config_spec.py
# asserts all three agree, so a rename fails a test rather than being found by
# grep.
PROD_CONFIG_EXPORT_TARGET="${PROD_CONFIG_EXPORT_TARGET:-//recir/examples/rankfm_mvai:export_prod_config}"
PROD_CONFIG_CHECK_MODULE="${PROD_CONFIG_CHECK_MODULE:-recir.examples.rankfm_mvai.prod_config_artifact}"
PROD_CFG_STAGE="${STAGING_DIR}/prod_config"
mkdir -p "$PROD_CFG_STAGE"
PROD_CONFIG_INCLUDED=true
PROD_CONFIG_REV=""
if [ "${SKIP_PROD_CONFIG:-0}" = "1" ]; then
    PROD_CONFIG_INCLUDED=false
    echo "SKIP_PROD_CONFIG=1: building without the prod config artifact."
    echo "  An entry point that reads PROD_CONFIG_ARTIFACT will fail to find it."
elif [ -n "${PROD_CONFIG_ARTIFACT_SRC:-}" ]; then
    if [ ! -f "$PROD_CONFIG_ARTIFACT_SRC" ]; then
        echo "ERROR: PROD_CONFIG_ARTIFACT_SRC=$PROD_CONFIG_ARTIFACT_SRC does not exist" >&2
        exit 1
    fi
    echo "using the artifact from PROD_CONFIG_ARTIFACT_SRC: $PROD_CONFIG_ARTIFACT_SRC"
    cp "$PROD_CONFIG_ARTIFACT_SRC" "$PROD_CFG_STAGE/prod_cfg.pt"
else
    PROD_CONFIG_REV="$(sl --cwd "$FBCODE_DIR" id -i 2>/dev/null)" || PROD_CONFIG_REV=""
    PROD_CONFIG_REV="${PROD_CONFIG_REV:-unknown}"
    PROD_CFG_DIRTY_ARG=""
    if [ -n "$(sl --cwd "$FBCODE_DIR" status -n minimal_viable_ai hammer 2>/dev/null)" ]; then
        PROD_CFG_DIRTY_ARG="--source-dirty"
    fi
    if ! (
        cd "$FBCODE_DIR"
        echo "  $PROD_CONFIG_EXPORT_TARGET"
        buck2 run "$PROD_CONFIG_EXPORT_TARGET" -- \
            --out "${PROD_CFG_STAGE}/prod_cfg.pt" \
            --source-revision "$PROD_CONFIG_REV" \
            ${PROD_CFG_DIRTY_ARG}
    ); then
        echo "ERROR: could not export the prod config ($PROD_CONFIG_EXPORT_TARGET)." >&2
        echo "  Reuse an earlier export:" >&2
        echo "    PROD_CONFIG_ARTIFACT_SRC=/tmp/prod_cfg.pt $0" >&2
        echo "  Or omit it, which is correct for synthetic and bench jobs:" >&2
        echo "    SKIP_PROD_CONFIG=1 $0" >&2
        exit 1
    fi
fi
echo ""
for dir in /var/facebook/x509_identities /var/facebook/credentials; do
    if [ ! -d "$dir" ]; then
        echo "Error: $dir not found." >&2
        exit 1
    fi
done

# Meta devservers route external traffic through fwdproxy. Export once so
# every downstream command (gcloud, podman build/push) inherits it. Override
# with PROXY_URL=... if your devserver uses a different forward proxy.
export PROXY_URL="${PROXY_URL:-http://fwdproxy:8080}"
export HTTPS_PROXY="${PROXY_URL}"
export HTTP_PROXY="${PROXY_URL}"
export NO_PROXY=localhost,127.0.0.1,.fbcdn.net,.facebook.com,.tfbnw.net
# apt only reads the lowercase spellings; pip/curl accept either.
export https_proxy="${HTTPS_PROXY}"
export http_proxy="${HTTP_PROXY}"
export no_proxy="${NO_PROXY}"

echo "Staging build context..."
RSYNC_EXCLUDES=(
    --exclude=.git
    --exclude=__pycache__
    --exclude='*.pyc'
    --exclude=tests
    --exclude=test
)

# recir + setup.py
# Keep recir/tests/stu_test.py (run by recir/tests/helpers.sh on TPU, which
# reuses this build script), but exclude every other test file. Includes MUST
# precede excludes -- rsync's first-match-wins evaluation order means a bare
# `--exclude=tests` would prune the whole tree before the includes are seen.
rsync -a \
    --include='tests/' \
    --include='tests/__init__.py' \
    --include='tests/stu_test.py' \
    --include='tests/xla/' \
    --include='tests/xla/__init__.py' \
    --include='tests/xla/dlrmv3_test_utils.py' \
    --include='tests/xla/gspmd_test_utils.py' \
    --exclude='tests/**' \
    --exclude='test/' \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    "${RECIR_DIR}/" "${STAGING_DIR}/recir/"
cp "${RECIR_DIR}/github/setup.py" "${STAGING_DIR}/setup.py"

# helion (Pallas backend): recir's HSTU op imports helion + exports it via
# Kernel.jax_fn. Stage the recir-pinned helion tree so the image installs it
# (-e) instead of PyPI helion. Exclude tests/benchmarks/wheels to keep the
# context small; the kernel + runtime + language packages are what we need.
rsync -a \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='test/' \
    --exclude='benchmarks/' \
    --exclude='dist/' \
    "${FBCODE_DIR}/helion/" "${STAGING_DIR}/helion/"

# hammer: stage from the AST import-trace manifest (much smaller than
# rsync -a hammer/v3/). The same trace_imports.py used by the OSS-import
# devserver harness produces the manifest -- if a file isn't reachable from
# the rankfm train entry point, it doesn't ship in the image.
TRACE_SCRIPT="${FBCODE_DIR}/hammer/v3/models/udd_fm_v3/oss_test/trace_imports.py"
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
# broadly rsynced) and strip 'fbcode/' so paths are relative to FBCODE_DIR
# for rsync --files-from.
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

# minimal_viable_ai (from mvai_cloud, the OSS clone). This must be
# self-sufficient for everything imported transitively from the rankfm_mvai
# train entry point — including RecTrainer, RecModelFamily, RecTrainPipeline
# and their config dataclasses. If a missing module breaks the build, fix it
# by adding the file to mvai_cloud/ rather than overlaying internal mvai
# (the internal versions of core/model_family_api/configs.py and
# core/ranking_common/configs.py drag in aiplatform/hpc/torchrec.fb/
# fbgemm.fb chains).
rsync -a "${RSYNC_EXCLUDES[@]}" "${FBCODE_DIR}/mvai_cloud/minimal_viable_ai/" "${STAGING_DIR}/minimal_viable_ai/"

# TPU job runtime helpers owned by mvai_cloud infrastructure.
mkdir -p "${STAGING_DIR}/mvai_cloud/gke"
rsync -a "${RSYNC_EXCLUDES[@]}" "${FBCODE_DIR}/mvai_cloud/gke/runtime/" "${STAGING_DIR}/mvai_cloud/gke/runtime/"

# generative_recommenders (was a github clone in the Dockerfile before;
# staged from fbsource so the build doesn't need outbound internet)
rsync -a "${RSYNC_EXCLUDES[@]}" "${FBCODE_DIR}/generative_recommenders/" "${STAGING_DIR}/generative_recommenders/"

# mrs + hpc: the udd_fm_v3 inter_arch DCN (use_inter_arch=True) imports
# create_module from mrs/.../dense_sparse/workplace.py. Its whole import tree is
# torchrec-only at runtime: module_config.py's hpc...blue_reels_model_config
# import is wrapped in try/except ImportError (the field annotation is a quoted
# forward ref, so it never evaluates when the name is absent), and dotdcn.py only
# pulls the torchrec-only general_arch_utils. Stage exactly the files reachable
# at runtime (NOT the rest of mrs/hpc, which pulls the torchrec.fb / hpc.new
# chains) plus __init__.py markers for every package dir along each path.
DCN_FILES=(
    "mrs/ranking/interaction/dense_sparse/module_config.py"
    "mrs/ranking/interaction/dense_sparse/workplace.py"
    "mrs/ranking/interaction/dense_sparse/dotdcn.py"
    "mrs/ranking/interaction/single_input/frozen/dcn_v0.py"
    "hpc/torchrec/models/feed/experiments/general_arch_utils.py"
)
for rel in "${DCN_FILES[@]}"; do
    mkdir -p "${STAGING_DIR}/$(dirname "$rel")"
    rsync -a "${FBCODE_DIR}/${rel}" "${STAGING_DIR}/${rel}"
    d="$(dirname "$rel")"
    while [ "$d" != "." ]; do
        touch "${STAGING_DIR}/${d}/__init__.py"
        d="$(dirname "$d")"
    done
done

# Record what actually went into the image. Everything above rsyncs the WORKING
# TREE, not a commit, so uncommitted edits ship silently -- and disappear from
# the checkout as soon as the tree moves on, leaving a running pod whose code
# exists in no revision. That has already happened once: an image shipped a
# `_warn_if_over_budget` and a `--prod_ro_per_sample_budget` flag that were
# replaced before landing, and the only way to read the running binary was to
# `podman run ... sed` it. BUILD_INFO.json makes the mapping recoverable, and
# train.py logs it at startup.
echo "--- Stamping build provenance ---"
BUILD_REV="$(sl --cwd "$FBCODE_DIR" id -i 2>/dev/null)" || BUILD_REV=""
BUILD_REV="${BUILD_REV:-unknown}"
BUILD_DIRTY="$(sl --cwd "$FBCODE_DIR" status -n recir hammer mvai_cloud generative_recommenders mrs hpc 2>/dev/null)" || BUILD_DIRTY=""

BUILD_REV="$BUILD_REV" \
BUILD_DIRTY="$BUILD_DIRTY" \
BUILD_TAG="$IMAGE_TAG" \
BUILD_BARNFS="$BARNFS_INCLUDED" \
BUILD_PROD_CONFIG="$PROD_CONFIG_INCLUDED" \
BUILD_PROD_CONFIG_REV="$PROD_CONFIG_REV" \
BUILD_PROD_CONFIG_TARGET="$PROD_CONFIG_EXPORT_TARGET" \
BUILD_OUT="${STAGING_DIR}/recir/BUILD_INFO.json" \
python3 -c '
import datetime, json, os

dirty = sorted(line for line in os.environ["BUILD_DIRTY"].splitlines() if line.strip())
with open(os.environ["BUILD_OUT"], "w") as f:
    json.dump(
        {
            "revision": os.environ["BUILD_REV"],
            "tag": os.environ["BUILD_TAG"],
            "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(
                timespec="seconds"
            ),
            "built_by": os.environ.get("USER", "unknown"),
            "dirty_files": dirty,
            "barnfs": os.environ["BUILD_BARNFS"] == "true",
            "prod_config": os.environ["BUILD_PROD_CONFIG"] == "true",
            "prod_config_revision": os.environ["BUILD_PROD_CONFIG_REV"],
            "prod_config_export_target": (
                os.environ["BUILD_PROD_CONFIG_TARGET"]
                if os.environ["BUILD_PROD_CONFIG"] == "true"
                else ""
            ),
        },
        f,
        indent=2,
    )
'
echo "  revision ${BUILD_REV}, tag ${IMAGE_TAG}"

if [ -n "$BUILD_DIRTY" ]; then
    echo "WARNING: building from a DIRTY tree. These uncommitted files go into" >&2
    echo "         ${IMAGE_TAG} and will exist in no commit once the tree moves on:" >&2
    echo "$BUILD_DIRTY" | sed 's/^/           /' >&2
fi

echo "============================================"
echo "  Building RankFM-MVAI TPU image"
echo "  Source:  ${RECIR_DIR}"
echo "  Context: ${STAGING_DIR}"
echo "  Image:   ${IMAGE_PATH}"
echo "============================================"

cd "$STAGING_DIR"
# --pull defaults to "missing", not to podman's bare default: the base image is
# evicted from the local store periodically, and with no pull policy the next
# build dies with `python:3.12-slim: image not known` before step 1. Override
# with PODMAN_PULL_POLICY=always|never.
#
# The comment lives ABOVE the command on purpose. Inside a `\`-continued
# command a comment line swallows the continuation, leaving `podman build`
# with no arguments at all -- and `bash -n` still reports the file as valid.
podman build \
    --pull="${PODMAN_PULL_POLICY:-missing}" \
    ${NO_CACHE:+--no-cache} \
    --build-arg HTTP_PROXY="$HTTP_PROXY" \
    --build-arg http_proxy="$HTTP_PROXY" \
    --build-arg HTTPS_PROXY="$HTTPS_PROXY" \
    --build-arg https_proxy="$HTTPS_PROXY" \
    --build-arg NO_PROXY="$NO_PROXY" \
    --build-arg http_proxy="$HTTP_PROXY" \
    --build-arg https_proxy="$HTTPS_PROXY" \
    --build-arg no_proxy="$NO_PROXY" \
    -v /var/facebook/x509_identities:/var/facebook/x509_identities \
    -v /var/facebook/credentials:/var/facebook/credentials \
    --build-context=credentials_dir=/var/facebook/credentials \
    --network=host \
    -f recir/examples/rankfm_mvai/scripts/Dockerfile \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" .

echo ""
echo "--- Verifying image ---"
# The barnfs assertions only apply when the wheel was staged.
if [ "$BARNFS_INCLUDED" = true ]; then
    BARNFS_CHECK="
import barnfs; print('BarnFS: OK')
import torch.distributed.checkpoint as dcp
assert dcp.save.__module__.endswith('dcp_filesystem_barnfs')
assert dcp.load.__module__.endswith('dcp_filesystem_barnfs')
print('BarnFS DCP hotpatch: OK')
"
else
    BARNFS_CHECK="
try:
    import barnfs
except ImportError:
    print('BarnFS: absent, as built (SKIP_BARNFS=1)')
else:
    raise SystemExit('barnfs is importable in an image built with SKIP_BARNFS=1')
"
fi

# The artifact describes the model the run builds, and a wrong one fails
# quietly: the run trains a different model. Each model's check module asserts
# what identifies it (rankfm: prod's sequence length), here at build time.
if [ "$PROD_CONFIG_INCLUDED" = true ]; then
    PROD_CONFIG_CHECK="
import importlib
from recir.examples.rankfm_mvai.prod_config_spec import (
    PROD_CONFIG_ARTIFACT_CONTAINER_PATH,
)
_summary = importlib.import_module('${PROD_CONFIG_CHECK_MODULE}').check(
    PROD_CONFIG_ARTIFACT_CONTAINER_PATH
)
print(f'prod config artifact: OK ({_summary})')
"
else
    PROD_CONFIG_CHECK="
import os
from recir.examples.rankfm_mvai.prod_config_spec import (
    PROD_CONFIG_ARTIFACT_CONTAINER_PATH,
)
if os.path.exists(PROD_CONFIG_ARTIFACT_CONTAINER_PATH):
    raise SystemExit(
        'a prod config artifact is present in an image built with '
        'SKIP_PROD_CONFIG=1'
    )
print('prod config artifact: absent, as built (SKIP_PROD_CONFIG=1)')
"
fi
podman run --rm \
    -e TORCH_ENABLE_BARNFS=ON \
    -e MODEL_TYPE_NAME=asset://ai.model.type/infra_mrs_research \
    "${IMAGE_NAME}:${IMAGE_TAG}" python3 -c "
import torch; print(f'PyTorch: {torch.__version__}')
${BARNFS_CHECK}
import jax; print(f'JAX: {jax.__version__}')
import recir; print('recir: OK')
import helion; print('helion: OK')
from recir.ops.helion.jagged_hstu_attention import (  # noqa: F401
    hstu_ragged_attention_bwd_jax_fn,
    hstu_ragged_attention_fwd_jax_fn,
)
import recir.ops.helion_hstu_attention_op  # noqa: F401
import recir.torchax.ops.jhstu_attention  # noqa: F401
# Exercise the rebind itself, not just the import: a wrong target module here
# means the swap never installs and attention silently decomposes into ~15 aten
# ops (a perf cliff, not an error).
from recir.ops.helion_hstu_attention_op import install_helion_hstu_attention_swap
install_helion_hstu_attention_swap()
import hammer.v2.ops.hstu_preprocess_and_attention as _hpa
from recir.ops.helion_hstu_attention_op import _recir_pytorch_hstu_mha
assert _hpa.pytorch_hstu_mha is _recir_pytorch_hstu_mha
print('recir helion fwd/bwd ops + lowering + attention swap: OK')
import torchrec; print('torchrec: OK')
from torchrec.distributed.model_parallel import DistributedModelParallel  # noqa: F401
from torchrec.distributed.embedding import EmbeddingCollectionSharder  # noqa: F401
print('torchrec.distributed: OK')
import hammer.v3; print('hammer.v3: OK')
import minimal_viable_ai; print('minimal_viable_ai: OK')
import mvai_cloud.gke.runtime.tensorboard_uploader; print('mvai_cloud runtime: OK')
import mvai_cloud.gke.runtime.pod_diagnostics; print('mvai_cloud pod_diagnostics: OK')
from minimal_viable_ai.core.trainer.trainer import RecTrainer  # noqa: F401
print('minimal_viable_ai.RecTrainer: OK')
from generative_recommenders.common import HammerKernel; print('generative_recommenders: OK')
from recir.examples.rankfm_mvai.rankfm_dispatchers import PreprocessorDispatcher; from recir.embedding.cpu import SparseArchDispatcher; print('rankfm_mvai dispatchers: OK')
# Import the ACTUAL training entry module (not just config). This exercises the
# full chain -- model_family -> hammer udd_fm_family, train_pipeline -> recir
# compile -- so a missing staged module fails the BUILD here instead of only at
# TPU pod startup. (main() is __name__-guarded, so no jax.distributed init runs.)
import recir.examples.rankfm_mvai.train  # noqa: F401
print('rankfm_mvai.train entry import: OK')
# Also import the parity harness (__name__-guarded, no jax init at import) so a
# missing staged dep fails the BUILD here, not at parity run time.
import recir.examples.rankfm_mvai.parity_forward_loss  # noqa: F401
print('rankfm_mvai.parity_forward_loss import: OK')
${PROD_CONFIG_CHECK}
print()
print('All imports OK!')
"

echo ""
echo "--- Authenticating to Artifact Registry ---"
# Use a fresh OAuth2 access token from gcloud (~1h TTL). If gcloud has no
# active account, prompt the user to log in first.
if ! gcloud auth print-access-token > /dev/null 2>&1; then
    echo "gcloud has no active credentials. Run 'gcloud auth login' first." >&2
    exit 1
fi
gcloud auth print-access-token | \
    podman login -u oauth2accesstoken --password-stdin \
        "https://${REGION}-docker.pkg.dev"

echo ""
echo "--- Tagging and pushing ---"
podman tag "${IMAGE_NAME}:${IMAGE_TAG}" "${IMAGE_PATH}"
podman push "${IMAGE_PATH}"

echo ""
echo "============================================"
echo "  BUILD COMPLETE: ${IMAGE_PATH}"
echo "============================================"
