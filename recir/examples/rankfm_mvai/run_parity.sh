#!/usr/bin/env bash
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#
# One-shot driver for the RankFM-MVAI padding-inertness parity on TPU:
#   build + push the image  ->  submit rankfm_mvai-parity-4chip.yaml  ->  tail logs.
#
# The parity is a single short pod (parity_forward_loss.py) that runs once and
# exits, so -- unlike run_tpu_bench.sh -- there is no xprof/trace retrieval; this
# just streams the pod's stdout until it completes, then re-fetches the full log.
#
# Usage:
#   ./run_parity.sh
#
# Env knobs:
#   IMAGE_TAG           image tag to build/push AND launch (default: plund-parity,
#                       matching the yaml). The yaml's rankfm_mvai_tpu image is
#                       patched to this tag in a temp copy so build and submit can
#                       never drift.
#   PARITY_EMB_BACKEND  override the yaml's embedding backend: cpu | tpu | in_graph
#                       (default: leave the yaml's value).
#   PARITY_SKIP_BUILD=1 skip the podman rebuild and reuse the pushed IMAGE_TAG
#                       (only when the image already has your current code).
#   PARITY_RINGSHIFT_MIN_ROWS / PARITY_FSDP_MIN_ROWS / PARITY_ON_DEVICE_INIT /
#   PARITY_BATCH_RANKS / PARITY_NUM_CANDIDATES / PARITY_INPUT / PARITY_RTOL /
#   PARITY_ATOL         if set, forwarded into the pod env (injected into the yaml)
#                       so you can sweep parity knobs without editing the yaml.
#                       e.g. PARITY_RINGSHIFT_MIN_ROWS=2000000 -> all in_graph
#                       tables use DP F.embedding (skips the ring-shift kernel).
#   YAML                parity yaml (default: rankfm_mvai-parity-4chip.yaml).
#   CTX / NS            kube context / namespace (defaults: current context /
#                       ns-mrs-core-shared).
#   DELETE_ON_EXIT=1    delete the JobSet when the script exits (default: keep).
#
# Examples:
#   ./run_parity.sh                                  # build+push, yaml's backend
#   PARITY_EMB_BACKEND=in_graph ./run_parity.sh      # ring-shift backend
#   PARITY_EMB_BACKEND=in_graph PARITY_RINGSHIFT_MIN_ROWS=2000000 ./run_parity.sh
#   IMAGE_TAG=$USER-parity ./run_parity.sh           # isolate from the shared tag
#   PARITY_SKIP_BUILD=1 ./run_parity.sh              # reuse the already-pushed image

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
echo "  emb backend : ${PARITY_EMB_BACKEND:-<yaml default>}" | tee -a "${LOG}"
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

# 2) Patch a temp copy of the yaml: pin the image tag to IMAGE_TAG (so build and
#    submit can't drift), optionally override PARITY_EMB_BACKEND, and forward any
#    PARITY_* tuning knobs set in the environment into the pod env.
PATCHED="$(mktemp "/tmp/_parity_${IMAGE_TAG}_XXXXXX.yaml")"
sed -E "s|(rankfm_mvai_tpu):[A-Za-z0-9._-]+|\1:${IMAGE_TAG}|g" "${YAML}" > "${PATCHED}"

if [ -n "${PARITY_EMB_BACKEND:-}" ]; then
    awk -v val="${PARITY_EMB_BACKEND}" '
        /name: PARITY_EMB_BACKEND/ {
            print; getline; sub(/value:.*/, "value: \"" val "\""); print; next
        }
        { print }
    ' "${PATCHED}" > "${PATCHED}.tmp" && mv "${PATCHED}.tmp" "${PATCHED}"
fi

# Inject extra env entries after PARITY_EMB_BACKEND (14-space env indentation).
# The literal "\n" is expanded to newlines by awk -v below.
EXTRA_ENV=""
_add_env() {  # $1=name $2=value; skips empty values
    if [ -n "$2" ]; then
        EXTRA_ENV+="              - name: $1\n                value: \"$2\"\n"
        echo "  inject env: $1=$2" | tee -a "${LOG}"
    fi
}
_add_env PARITY_RINGSHIFT_MIN_ROWS "${PARITY_RINGSHIFT_MIN_ROWS:-}"
_add_env PARITY_FSDP_MIN_ROWS "${PARITY_FSDP_MIN_ROWS:-}"
_add_env PARITY_ON_DEVICE_INIT "${PARITY_ON_DEVICE_INIT:-}"
_add_env PARITY_BATCH_RANKS "${PARITY_BATCH_RANKS:-}"
_add_env PARITY_NUM_CANDIDATES "${PARITY_NUM_CANDIDATES:-}"
_add_env PARITY_INPUT "${PARITY_INPUT:-}"
_add_env PARITY_RTOL "${PARITY_RTOL:-}"
_add_env PARITY_ATOL "${PARITY_ATOL:-}"
_add_env PARITY_PRECISION "${PARITY_PRECISION:-}"
_add_env PARITY_CELLS "${PARITY_CELLS:-}"
_add_env PARITY_DUMP_HLO "${PARITY_DUMP_HLO:-}"
_add_env RANKFM_S "${RANKFM_S:-}"
_add_env PARITY_GLOBAL_BS "${PARITY_GLOBAL_BS:-}"
_add_env PARITY_HASH_SIZE_CAP "${PARITY_HASH_SIZE_CAP:-}"
if [ -n "${EXTRA_ENV}" ]; then
    awk -v extra="${EXTRA_ENV}" '
        /name: PARITY_EMB_BACKEND/ { print; getline; print; printf "%s", extra; next }
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
