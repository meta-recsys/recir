#!/bin/bash
set -euo pipefail

PROJECT_ID=mrs-tpu-y317h
REGION=us-central1
REPO=recml
IMAGE_NAME=dlrmv3_tpu
IMAGE_PATH="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE_NAME}:latest"
RECIR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

echo "============================================"
echo "  Building DLRMv3 TPU image"
echo "  Context: ${RECIR_DIR}"
echo "  Image: ${IMAGE_PATH}"
echo "============================================"

cd "$RECIR_DIR"
docker build -f recir/examples/dlrmv3/Dockerfile -t "${IMAGE_NAME}" .

echo ""
echo "--- Verifying image ---"
docker run --rm "${IMAGE_NAME}" python3 -c "
import torch; print(f\"PyTorch: {torch.__version__}\")
import jax; print(f\"JAX: {jax.__version__}\")
import recir; print(\"recir: OK\")
import torchrec; print(\"torchrec: OK\")
from generative_recommenders.common import HammerKernel, set_dev_mode; from generative_recommenders.modules.dlrm_hstu import DlrmHSTU; print(\"generative_recommenders: OK\")
from recir.examples.dlrmv3.config import DLRMv3Config; print(\"dlrmv3 example: OK\")
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
