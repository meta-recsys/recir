#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

# Build + push the ringshift TPU image.
# Run from the OSS recir checkout (where setup.py lives at the top).

: "${IMAGE:?set IMAGE=<registry>/<name>:<tag>}"
IMAGE_NAME=ringshift_tpu
RECIR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

echo "============================================"
echo "  Building ringshift TPU image"
echo "  Context: ${RECIR_DIR}"
echo "  Image: ${IMAGE}"
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
docker tag "${IMAGE_NAME}:latest" "${IMAGE}"
docker push "${IMAGE}"

echo ""
echo "============================================"
echo "  BUILD COMPLETE: ${IMAGE}"
echo "============================================"
