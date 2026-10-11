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
# [RS_LOAD] and the headline RESULT line) to an in-repo log file that survives
# pod auto-cleanup. Does NOT build the image (rebuild via build_image_podman.sh).
#
# Works for every --mode the bench exposes (fwd / bwd / ebc) and for the
# correctness runners: the poll keys off the shared `=== RESULT` prefix rather
# than one mode's spelling. No trailing space -- the correctness tests print
# `=== RESULT:` with a colon, the bench modes `=== RESULT ebc:` with a space.
#
# Usage:
#   cd <fbcode>
#   ./recir/examples/embedding/run_microbench.sh manifests/rankfm-550k.yaml
#   MAX_POLLS=3 ./recir/examples/embedding/run_microbench.sh manifests/rankfm-550k.yaml
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
MAX_POLLS="${MAX_POLLS:-60}"

if ! [[ "${MAX_POLLS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: MAX_POLLS must be a positive integer, got: ${MAX_POLLS}" >&2
    exit 2
fi

cd "${FBCODE_ROOT}" || exit 1
# helpers.sh exports CTX/NS + HTTPS_PROXY and the gateway kube context.
HELPERS="${FBCODE_ROOT}/recir/fb/examples/rankfm_mvai/scripts/helpers.sh"
if [ ! -f "${HELPERS}" ]; then
    echo "ERROR: TPU JobSet helper not found: ${HELPERS}" >&2
    exit 1
fi
# shellcheck source=/dev/null
source "${HELPERS}"

{
    echo "=== microbench run: ${NAME} @ ${TS} ==="
    echo "yaml: ${YAML}"
} | tee "${LOG}"

submit_output="$(kubectl create -f "${YAML}" --context="${CTX}" -n "${NS}" \
    -o jsonpath='{.metadata.name}' 2>&1)"
submit_rc=$?
if [ "${submit_rc}" -ne 0 ]; then
    echo "${submit_output}" | tee -a "${LOG}" >&2
    echo "ERROR: submit failed (exit ${submit_rc})" | tee -a "${LOG}" >&2
    exit "${submit_rc}"
fi
JS="$(echo "${submit_output}" | tail -1)"
echo "jobset: ${JS}" | tee -a "${LOG}"

# Poll: scan ALL workers for the process-0 output. On GKE TPU, JAX process 0
# (device 0, where [RS_LOAD] + RESULT print) lands on a RANDOM worker pod, NOT
# worker-0-0 -- so dumping only worker-0-0 misses them. The manifest's trailing
# `sleep` keeps pods alive so we can grab the full log before cleanup.
found=""
result_failed=false
for i in $(seq 1 "${MAX_POLLS}"); do
    sleep 20
    pods="$(kubectl get pods --context="${CTX}" -n "${NS}" \
        -l "jobset.sigs.k8s.io/jobset-name=${JS}" -o name 2>/dev/null | sed 's|pod/||')"
    [ -z "${pods}" ] && { echo "poll ${i}: no pods yet"; continue; }
    for n in ${pods}; do
        full="$(kubectl logs "${n}" --context="${CTX}" -n "${NS}" 2>/dev/null)"
        # Keyed on the headline, which only the JAX process-0 pod prints --
        # a marker echoed by the manifest's shell would match every worker.
        # Wait for BENCH_EXIT too, so a parity artifact printed after RESULT
        # cannot be archived before the producer has finished writing it.
        if echo "${full}" | grep -q "=== RESULT" && \
                echo "${full}" | grep -q "BENCH_EXIT="; then
            { echo "----- BEGIN process-0 worker (${n}) -----"; echo "${full}";
              echo "----- END ${n} -----"; } >> "${LOG}"
            found="${n}"
            echo "DONE -> ${LOG}"
            echo "  RS_LOAD: $(echo "${full}" | grep '\[RS_LOAD\]' | head -1)"
            # --mode ebc gates its timings on the two reductions agreeing.
            # Surface that line too: a speedup is only meaningful once it holds.
            agree="$(echo "${full}" | grep 'agreement:' | head -1)"
            [ -n "${agree}" ] && echo "  AGREE  : ${agree}"
            echo "${full}" | grep '=== RESULT' | sed 's/^/  RESULT : /'
            bench_exit="$(echo "${full}" | grep 'BENCH_EXIT=' | tail -1 | cut -d= -f2)"
            if [ "${bench_exit}" != "0" ] || \
                    echo "${full}" | grep '=== RESULT' | grep -q 'FAIL'; then
                result_failed=true
            fi
            break
        fi
    done
    [ -n "${found}" ] && break
    echo "poll ${i}: running (${pods:+$(echo "${pods}" | grep -c .) pods})"
done

ln -sf "${LOG}" "${LATEST}"
echo "log: ${LOG}"

if [ -z "${found}" ]; then
    echo "ERROR: no RESULT found after ${MAX_POLLS} polls" | tee -a "${LOG}" >&2
    exit 1
fi
if [ "${result_failed}" = true ]; then
    echo "ERROR: test reported a failing RESULT" | tee -a "${LOG}" >&2
    exit 1
fi
