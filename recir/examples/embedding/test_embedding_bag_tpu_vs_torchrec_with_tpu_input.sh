#!/usr/bin/env bash
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# One public workflow for the cross-runtime parity test:
#   1. run the real lookup and gradient on TPU;
#   2. capture the self-describing artifact from the TPU log;
#   3. run TorchRec on CPU and compare forward results and table gradients.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FBCODE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ARTIFACT="${SCRIPT_DIR}/bench_logs/w4-ebc-parity-dump-latest.log"

"${SCRIPT_DIR}/run_microbench.sh" manifests/w4-ebc-parity-dump.yaml

if [ ! -s "${ARTIFACT}" ]; then
    echo "ERROR: TPU artifact log was not produced: ${ARTIFACT}" >&2
    exit 1
fi

cd "${FBCODE_ROOT}"
buck2 test \
    //recir/ops/pallas/embeddings/tests:test_embedding_bag_tpu_vs_torchrec_with_tpu_input \
    -- --env "RECIR_EBC_TPU_ARTIFACT=${ARTIFACT}"
