#!/bin/bash
set -euo pipefail

# Build + push the ringshift TPU image to GCR.
# Run from the OSS recir checkout (where setup.py lives at the top).
#
# Image is pulled by the JobSet manifests under ./manifests/ at run time.

PROJECT_ID=mrs-tpu-y317h
REGION=us-central1
REPO=recml
IMAGE_NAME=ringshift_tpu
IMAGE_PATH="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE_NAME}:latest"
RECIR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

echo "============================================"
echo "  Building ringshift TPU image"
echo "  Context: ${RECIR_DIR}"
echo "  Image: ${IMAGE_PATH}"
echo "============================================"

cd "$RECIR_DIR"
docker build -f recir/examples/embedding/Dockerfile -t "${IMAGE_NAME}" .

echo ""
echo "--- Verifying image ---"
docker run --rm "${IMAGE_NAME}" python3 -c "
import jax; print(f\"JAX: {jax.__version__}\")
import recir; print(\"recir: OK\")
from recir.ops.pallas.embeddings.ringshift import (
    ringshift_embedding_lookup,
    ringshift_embedding_lookup_reference_jax,
    ringshift_embedding_lookup_reference,
); print(\"ringshift_embedding_lookup public API: OK\")
print()
print(\"All imports OK!\")
"

echo ""
echo "--- Tagging and pushing ---"
docker tag "${IMAGE_NAME}:latest" "${IMAGE_PATH}"
docker push "${IMAGE_PATH}"

echo ""
echo "============================================"
echo "  BUILD COMPLETE: ${IMAGE_PATH}"
echo "============================================"
