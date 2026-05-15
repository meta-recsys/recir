#!/bin/bash
set -euo pipefail

# fbcode-compatible build of the ringshift TPU image. Mirrors the
# dlrmv3 build_image_podman.sh pattern: stage a tmp build context that
# matches what the Dockerfile expects (`<context>/setup.py` next to
# `<context>/recir/`), then podman-build with Meta corp credentials.
#
# OSS contributors should use ./build_image.sh instead -- it assumes the
# OSS recir checkout has setup.py at the top, which is not true in
# fbsource/fbcode (the only setup.py lives at recir/github/setup.py).

PROJECT_ID=mrs-tpu-y317h
REGION=us-central1
REPO=recml
IMAGE_NAME=ringshift_tpu
# Override with: IMAGE_TAG=v1 ./build_image_podman.sh
IMAGE_TAG="${IMAGE_TAG:-latest}"
IMAGE_PATH="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE_NAME}:${IMAGE_TAG}"
RECIR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAGING_DIR="$(mktemp -d)"

# Validate prerequisites
for cmd in podman with-proxy; do
    if ! command -v "$cmd" > /dev/null 2>&1; then
        echo "Error: '$cmd' not found. Run 'feature install ${cmd/#podman/k8s_tools}' to install it." >&2
        exit 1
    fi
done
for dir in /var/facebook/x509_identities /var/facebook/credentials; do
    if [ ! -d "$dir" ]; then
        echo "Error: $dir not found." >&2
        exit 1
    fi
done

# Stage build context to match Dockerfile's expected layout:
#   <context>/setup.py        (from recir/github/setup.py)
#   <context>/recir/           (the recir package)
cleanup_staging() {
    rm -rf "$STAGING_DIR"
}
trap cleanup_staging EXIT

echo "Staging build context..."
cp -r "$RECIR_DIR" "$STAGING_DIR/recir"
cp "$RECIR_DIR/github/setup.py" "$STAGING_DIR/setup.py"

echo "============================================"
echo "  Building ringshift TPU image"
echo "  Source:  ${RECIR_DIR}"
echo "  Context: ${STAGING_DIR}"
echo "  Image:   ${IMAGE_PATH}"
echo "============================================"

cd "$STAGING_DIR"
with-proxy podman build \
    -v /var/facebook/x509_identities:/var/facebook/x509_identities \
    -v /var/facebook/credentials:/var/facebook/credentials \
    --build-context=credentials_dir=/var/facebook/credentials \
    --network=host \
    -f recir/examples/embedding/Dockerfile \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" .

echo ""
echo "--- Verifying image ---"
podman run --rm "${IMAGE_NAME}:${IMAGE_TAG}" python3 -c "
import jax; print(f'JAX: {jax.__version__}')
import recir; print('recir: OK')
from recir.ops.pallas.embeddings.ringshift import (
    ringshift_embedding_lookup,
    ringshift_embedding_lookup_reference_jax,
    ringshift_embedding_lookup_reference,
); print('ringshift_embedding_lookup public API: OK')
print()
print('All imports OK!')
"

echo ""
echo "--- Tagging and pushing ---"
podman tag "${IMAGE_NAME}:${IMAGE_TAG}" "${IMAGE_PATH}"
podman push "${IMAGE_PATH}"

echo ""
echo "============================================"
echo "  BUILD COMPLETE: ${IMAGE_PATH}"
echo "============================================"
