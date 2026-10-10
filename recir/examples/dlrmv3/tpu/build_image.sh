#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# Build the DLRMv3 TPU image, smoke-test its imports, and push it to $IMAGE.
set -euo pipefail

: "${IMAGE:?set IMAGE=<registry>/<name>:<tag>}"
RECIR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

cd "$RECIR_DIR"
docker build -f recir/examples/dlrmv3/tpu/Dockerfile -t dlrmv3_tpu .

docker run --rm dlrmv3_tpu python3 -c "
import jax, torch, torchrec
from generative_recommenders.modules.dlrm_hstu import DlrmHSTU
from recir.examples.dlrmv3.common.model import DlrmV3TrainModule
from recir.examples.dlrmv3.tpu import embedding_backends
print('imports OK: torch', torch.__version__, 'jax', jax.__version__)
"

docker tag dlrmv3_tpu "$IMAGE"
docker push "$IMAGE"
echo "pushed $IMAGE"
