#!/usr/bin/env bash
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#
# One-shot CPU-eager vs TPU-compiled RankFM parity runner.
# Build and push the image, submit the TPU JobSet, and stream its logs.
#
# Usage: PARITY_GCS_DATA_PATH=gs://bucket/path ./run_parity.sh
# Required: PARITY_GCS_DATA_PATH
# Common overrides: IMAGE_TAG, PARITY_SKIP_BUILD, YAML, CTX, NS, DELETE_ON_EXIT,
# PARITY_RTOL, PARITY_ATOL, PARITY_GRAD_RTOL, PARITY_GRAD_ATOL,
# PARITY_PRECISION, PARITY_BF16, RANKFM_S, PARITY_HASH_SIZE_CAP,
# PARITY_DATA_FILE_TEMPLATE, PARITY_DATA_FILE_INDEX, PARITY_LOCAL_BATCH_SIZE.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FBCODE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

IMAGE_TAG="${IMAGE_TAG:-plund-parity}"
YAML_ARG="${YAML:-${SCRIPT_DIR}/rankfm_mvai-parity-4chip.yaml}"
if [ -f "${YAML_ARG}" ]; then
    YAML="$(cd "$(dirname "${YAML_ARG}")" && pwd)/$(basename "${YAML_ARG}")"
elif [ -f "${SCRIPT_DIR}/${YAML_ARG}" ]; then
    YAML="${SCRIPT_DIR}/${YAML_ARG}"
else
    echo "ERROR: yaml not found: ${YAML_ARG}" >&2
    exit 1
fi

# kubectl/gcloud need the Meta forward proxy on devservers.
export HTTPS_PROXY="${HTTPS_PROXY:-http://fwdproxy:8080}"
CTX="${CTX:-$(kubectl config current-context 2>/dev/null)}"
NS="${NS:-ns-mrs-core-shared}"
if [ -z "${CTX}" ]; then
    echo "ERROR: no kube context (set CTX or run 'kubectl config use-context ...')." >&2
    exit 1
fi

TS="$(date +%Y%m%d-%H%M%S)"
LOG="/tmp/parity-${IMAGE_TAG}-${TS}.log"

echo "================================================================" | tee "${LOG}"
echo "RankFM-MVAI parity run" | tee -a "${LOG}"
echo "  fbcode root : ${FBCODE_ROOT}" | tee -a "${LOG}"
echo "  yaml        : ${YAML}" | tee -a "${LOG}"
echo "  image tag   : ${IMAGE_TAG}" | tee -a "${LOG}"
echo "  context     : ${CTX}" | tee -a "${LOG}"
echo "  namespace   : ${NS}" | tee -a "${LOG}"
echo "  log file    : ${LOG}" | tee -a "${LOG}"
echo "================================================================" | tee -a "${LOG}"

cd "${FBCODE_ROOT}" || exit 1

# 1) Build + push the image under IMAGE_TAG (abort the run if the build fails so
#    we never submit against a stale/broken image).
if [ "${PARITY_SKIP_BUILD:-0}" = "1" ]; then
    echo "PARITY_SKIP_BUILD=1: reusing pushed image rankfm_mvai_tpu:${IMAGE_TAG}" | tee -a "${LOG}"
else
    echo "=== building + pushing rankfm_mvai_tpu:${IMAGE_TAG} ===" | tee -a "${LOG}"
    if ! IMAGE_TAG="${IMAGE_TAG}" bash "${SCRIPT_DIR}/build_image_podman.sh" > >(tee -a "${LOG}") 2>&1; then
        echo "ERROR: image build/push failed; aborting before submit. See ${LOG}." | tee -a "${LOG}"
        exit 1
    fi
fi

# 2) Patch a temp copy of the yaml: pin the image tag and forward supported
# parity settings into the pod environment.
PATCHED="$(mktemp "/tmp/_parity_${IMAGE_TAG}_XXXXXX.yaml")"
sed -E "s|(rankfm_mvai_tpu):[A-Za-z0-9._-]+|\1:${IMAGE_TAG}|g" "${YAML}" > "${PATCHED}"


# Inject extra env entries after PARITY_DEVICE (14-space env indentation).
# The literal "\n" is expanded to newlines by awk -v below.
EXTRA_ENV=""
_add_env() {  # $1=name $2=value; skips empty values
    if [ -n "$2" ]; then
        EXTRA_ENV+="              - name: $1\n                value: \"$2\"\n"
        echo "  inject env: $1=$2" | tee -a "${LOG}"
    fi
}
_add_env PARITY_RTOL "${PARITY_RTOL:-}"
_add_env PARITY_ATOL "${PARITY_ATOL:-}"
_add_env PARITY_GRAD_RTOL "${PARITY_GRAD_RTOL:-}"
_add_env PARITY_GRAD_ATOL "${PARITY_GRAD_ATOL:-}"
_add_env PARITY_PRECISION "${PARITY_PRECISION:-}"
_add_env PARITY_BF16 "${PARITY_BF16:-}"
_add_env RANKFM_S "${RANKFM_S:-}"
_add_env PARITY_HASH_SIZE_CAP "${PARITY_HASH_SIZE_CAP:-}"
# Replay-data settings. The parity pod has no gcsfuse mount, so use a gs:// URI.
_add_env PARITY_GCS_DATA_PATH "${PARITY_GCS_DATA_PATH:-}"
_add_env PARITY_DATA_FILE_TEMPLATE "${PARITY_DATA_FILE_TEMPLATE:-}"
_add_env PARITY_DATA_FILE_INDEX "${PARITY_DATA_FILE_INDEX:-}"
_add_env PARITY_LOCAL_BATCH_SIZE "${PARITY_LOCAL_BATCH_SIZE:-}"
if [ -n "${EXTRA_ENV}" ]; then
    awk -v extra="${EXTRA_ENV}" '
        /name: PARITY_DEVICE/ { print; getline; print; printf "%s", extra; next }
        { print }
    ' "${PATCHED}" > "${PATCHED}.tmp" && mv "${PATCHED}.tmp" "${PATCHED}"
fi

# 3) Submit (generateName -> capture the concrete JobSet name).
echo | tee -a "${LOG}"
echo "=== submitting ${PATCHED} ===" | tee -a "${LOG}"
JS="$(kubectl create -f "${PATCHED}" --context="${CTX}" -n "${NS}" \
    -o jsonpath='{.metadata.name}' 2>>"${LOG}")"
if [ -z "${JS}" ]; then
    echo "ERROR: submit failed (no JobSet name). See ${LOG}." | tee -a "${LOG}"
    exit 1
fi
echo "submitted JobSet: ${JS}" | tee -a "${LOG}"

cleanup() {
    rm -f "${PATCHED}"
    if [ "${DELETE_ON_EXIT:-0}" = "1" ]; then
        echo "deleting JobSet ${JS}" | tee -a "${LOG}"
        kubectl delete jobset "${JS}" --context="${CTX}" -n "${NS}" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

# 4) Wait for the pod to exist, then stream its logs until the run completes.
echo | tee -a "${LOG}"
echo "=== waiting for pod (jobset ${JS}) ===" | tee -a "${LOG}"
POD=""
for _ in $(seq 1 120); do  # up to ~10 min for image pull + libtpu init
    POD="$(kubectl get pods --context="${CTX}" -n "${NS}" \
        -l "jobset.sigs.k8s.io/jobset-name=${JS}" -o name 2>/dev/null \
        | grep 'worker-0-0-' | head -1 | sed 's:pod/::')"
    if [ -n "${POD}" ]; then
        phase="$(kubectl get pod "${POD}" --context="${CTX}" -n "${NS}" \
            -o jsonpath='{.status.phase}' 2>/dev/null)"
        echo "  pod ${POD}: ${phase}" | tee -a "${LOG}"
        if [ "${phase}" = "Running" ] || [ "${phase}" = "Succeeded" ] || [ "${phase}" = "Failed" ]; then
            break
        fi
    fi
    sleep 5
done
if [ -z "${POD}" ]; then
    echo "ERROR: no pod appeared for ${JS}. Check: kubectl describe jobset ${JS} -n ${NS}" | tee -a "${LOG}"
    exit 1
fi

echo | tee -a "${LOG}"
echo "=== streaming logs (${POD}) ===" | tee -a "${LOG}"
# -f follows to completion; container name is "parity" per the yaml.
kubectl logs -f "${POD}" -c parity --context="${CTX}" -n "${NS}" 2>&1 | tee -a "${LOG}"

# 5) Re-fetch the complete log (kubectl logs -f can drop lines under load) and
#    surface the PASS/FAIL verdict.
echo | tee -a "${LOG}"
echo "=== complete log (re-fetch) ===" | tee -a "${LOG}"
kubectl logs "${POD}" -c parity --context="${CTX}" -n "${NS}" >> "${LOG}" 2>&1 || true

echo | tee -a "${LOG}"
echo "================================================================" | tee -a "${LOG}"
echo "Done. JobSet: ${JS}" | tee -a "${LOG}"
echo "  verdict :" | tee -a "${LOG}"
grep -E "\[PARITY\] (PASS|FAIL)" "${LOG}" | tail -1 | sed 's/^/    /' | tee -a "${LOG}"
echo "  full log: ${LOG}" | tee -a "${LOG}"
echo "  delete  : kubectl delete jobset ${JS} --context=${CTX} -n ${NS}" | tee -a "${LOG}"
echo "================================================================" | tee -a "${LOG}"
