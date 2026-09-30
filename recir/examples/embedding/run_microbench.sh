#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.
#
# Microbench runner that mirrors recir/examples/rankfm/run_tpu_bench.sh's
# log-capture, for the ringshift_bench microbench manifests (manifests/*.yaml).
# Submits the JobSet and saves the FULL worker-0-0 stdout (where rank-0 prints
# [RS_LOAD] and the RESULT fwd=... line) to an in-repo log file that survives
# pod auto-cleanup. Does NOT build the image (rebuild via build_image_podman.sh).
#
# Usage:
#   cd <fbcode>
#   ./recir/examples/embedding/run_microbench.sh manifests/rankfm-550k.yaml
#
# Logs land in recir/examples/embedding/bench_logs/<name>-<ts>.log.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FBCODE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

YAML_ARG="${1:?usage: run_microbench.sh <manifest.yaml>}"
if [ -f "${YAML_ARG}" ]; then
    YAML="$(cd "$(dirname "${YAML_ARG}")" && pwd)/$(basename "${YAML_ARG}")"
elif [ -f "${SCRIPT_DIR}/${YAML_ARG}" ]; then
    YAML="${SCRIPT_DIR}/${YAML_ARG}"
else
    echo "ERROR: manifest not found: ${YAML_ARG}" >&2
    exit 1
fi

LOG_DIR="${SCRIPT_DIR}/bench_logs"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d-%H%M%S)"
NAME="$(basename "${YAML}" .yaml)"
LOG="${LOG_DIR}/${NAME}-${TS}.log"
LATEST="${LOG_DIR}/${NAME}-latest.log"

cd "${FBCODE_ROOT}" || exit 1
# helpers.sh exports CTX/NS + HTTPS_PROXY and the gateway kube context.
# shellcheck source=/dev/null
source "${FBCODE_ROOT}/recir/examples/rankfm_mvai/helpers.sh"

{
    echo "=== microbench run: ${NAME} @ ${TS} ==="
    echo "yaml: ${YAML}"
    JS="$(kubectl create -f "${YAML}" --context="${CTX}" -n "${NS}" \
        -o jsonpath='{.metadata.name}' 2>&1 | tail -1)"
    echo "jobset: ${JS}"
} | tee "${LOG}"

JS="$(grep '^jobset: ' "${LOG}" | tail -1 | awk '{print $2}')"
if [ -z "${JS}" ]; then
    echo "ERROR: submit failed" | tee -a "${LOG}"; exit 1
fi

# Poll: scan ALL workers for the process-0 output. On GKE TPU, JAX process 0
# (device 0, where [RS_LOAD] + RESULT print) lands on a RANDOM worker pod, NOT
# worker-0-0 -- so dumping only worker-0-0 misses them. The manifest's trailing
# `sleep` keeps pods alive so we can grab the full log before cleanup.
for i in $(seq 1 60); do
    sleep 20
    pods="$(kubectl get pods --context="${CTX}" -n "${NS}" \
        -l "jobset.sigs.k8s.io/jobset-name=${JS}" -o name 2>/dev/null | sed 's|pod/||')"
    [ -z "${pods}" ] && { echo "poll ${i}: no pods yet"; continue; }
    found=""
    for n in ${pods}; do
        full="$(kubectl logs "${n}" --context="${CTX}" -n "${NS}" 2>/dev/null)"
        if echo "${full}" | grep -qE "RESULT fwd"; then
            { echo "----- BEGIN process-0 worker (${n}) -----"; echo "${full}";
              echo "----- END ${n} -----"; } >> "${LOG}"
            found="${n}"
            echo "DONE -> ${LOG}"
            echo "  RS_LOAD: $(echo "${full}" | grep '\[RS_LOAD\]' | head -1)"
            echo "  RESULT : $(echo "${full}" | grep 'RESULT fwd' | head -1)"
            break
        fi
    done
    [ -n "${found}" ] && break
    echo "poll ${i}: running (${pods:+$(echo "${pods}" | grep -c .) pods})"
done

ln -sf "${LOG}" "${LATEST}"
echo "log: ${LOG}"
