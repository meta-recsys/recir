#!/usr/bin/env bash
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#
# Self-contained driver for the RankFM-MVAI FORWARD-ONLY benchmark on TPU:
#   build + push image  ->  submit rankfm_mvai-4chip-fwd-bench.yaml  ->  tail logs
#   ->  (if profiling) stage the device xplane.pb locally for xprof.
#
# Modeled on run_parity.sh: it does NOT use the shared run_tpu_bench.sh (which the
# train benches depend on), and it uses a PRIVATE image tag so it never clobbers
# the shared rankfm_mvai_tpu:latest. Build + submit are pinned to the same tag via
# a patched temp yaml so they can't drift, and FWD_* knobs can be overridden from
# the environment without editing the yaml.
#
# Usage:
#   ./run_fwd_profile.sh                              # build+push, run the yaml
#   FWD_SKIP_BUILD=1 ./run_fwd_profile.sh             # reuse the pushed image
#   FWD_SWEEP="seq=512,1024,2048,4096" ./run_fwd_profile.sh
#   FWD_EMB_BACKEND=cpu FWD_PROFILE=0 ./run_fwd_profile.sh
#
# Env knobs:
#   IMAGE_TAG        image tag to build/push AND launch (default: mlui-fwd; never
#                    touches :latest). The yaml image is patched to this tag.
#   FWD_SKIP_BUILD=1 skip the podman rebuild and reuse the pushed IMAGE_TAG (only
#                    when the image already has your current code).
#   FWD_XPLANE_DIR   where to stage the xplane locally (default /tmp/fwdtrace).
#   YAML             fwd yaml (default: rankfm_mvai-4chip-fwd-bench.yaml).
#   CTX / NS         kube context / namespace (defaults: current ctx / ns-mrs-core-shared).
#   DELETE_ON_EXIT=1 delete the JobSet when the script exits (default: keep).
#   Any of these, if set, override the yaml's value (or are appended):
#     FWD_SWEEP FWD_EMB_BACKEND FWD_PROFILE FWD_PROFILE_MODE FWD_DEVICE
#     FWD_GLOBAL_BS RANKFM_S FWD_NUM_CANDIDATES FWD_L1_LAYERS FWD_L2_LAYERS
#     FWD_L2_SEQ FWD_HASH_SIZE_CAP FWD_RINGSHIFT_MIN_ROWS FWD_FSDP_MIN_ROWS
#     FWD_WARMUP FWD_STEPS

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FBCODE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

IMAGE_TAG="${IMAGE_TAG:-mlui-fwd}"
YAML_ARG="${YAML:-${SCRIPT_DIR}/rankfm_mvai-4chip-fwd-bench.yaml}"
if [ -f "${YAML_ARG}" ]; then
    YAML="$(cd "$(dirname "${YAML_ARG}")" && pwd)/$(basename "${YAML_ARG}")"
elif [ -f "${SCRIPT_DIR}/${YAML_ARG}" ]; then
    YAML="${SCRIPT_DIR}/${YAML_ARG}"
else
    echo "ERROR: yaml not found: ${YAML_ARG}" >&2
    exit 1
fi

XPLANE_LOCAL_DIR="${FWD_XPLANE_DIR:-/tmp/fwdtrace}"
TRACE_BUCKET="gs://cq-mrs-core-shared_mrs_tpu_y317h/tpu-profiles"
# Manifold bucket kernelhub/perfetto reads from (browser view, no ssh tunnel).
KERNELHUB_BUCKET="perfdoctor_gpu_traces_test"
KERNELHUB_BASE="tree/traces/test/${USER}/rankfm_mvai"
KH_URL=""

# kubectl/gcloud/gsutil need the Meta forward proxy on devservers.
export HTTPS_PROXY="${HTTPS_PROXY:-http://fwdproxy:8080}"
CTX="${CTX:-$(kubectl config current-context 2>/dev/null)}"
NS="${NS:-ns-mrs-core-shared}"
if [ -z "${CTX}" ]; then
    echo "ERROR: no kube context (set CTX or run 'kubectl config use-context ...')." >&2
    exit 1
fi

TS="$(date +%Y%m%d-%H%M%S)"
LOG="/tmp/fwd-${IMAGE_TAG}-${TS}.log"

echo "================================================================" | tee "${LOG}"
echo "RankFM-MVAI forward-only benchmark" | tee -a "${LOG}"
echo "  fbcode root : ${FBCODE_ROOT}" | tee -a "${LOG}"
echo "  yaml        : ${YAML}" | tee -a "${LOG}"
echo "  image tag   : ${IMAGE_TAG}" | tee -a "${LOG}"
echo "  context     : ${CTX}" | tee -a "${LOG}"
echo "  namespace   : ${NS}" | tee -a "${LOG}"
echo "  log file    : ${LOG}" | tee -a "${LOG}"
echo "================================================================" | tee -a "${LOG}"

cd "${FBCODE_ROOT}" || exit 1

# 1) Build + push the image under IMAGE_TAG (abort if the build fails so we never
#    submit against a stale/broken image). Never touches :latest.
if [ "${FWD_SKIP_BUILD:-0}" = "1" ]; then
    echo "FWD_SKIP_BUILD=1: reusing pushed image rankfm_mvai_tpu:${IMAGE_TAG}" | tee -a "${LOG}"
else
    echo "=== building + pushing rankfm_mvai_tpu:${IMAGE_TAG} ===" | tee -a "${LOG}"
    if ! IMAGE_TAG="${IMAGE_TAG}" bash "${SCRIPT_DIR}/build_image_podman.sh" > >(tee -a "${LOG}") 2>&1; then
        echo "ERROR: image build/push failed; aborting before submit. See ${LOG}." | tee -a "${LOG}"
        exit 1
    fi
fi

# 2) Patch a temp copy of the yaml: pin the image tag (so build+submit can't
#    drift) and apply any FWD_* env overrides set in the environment.
PATCHED="$(mktemp "/tmp/_fwd_${IMAGE_TAG}_XXXXXX.yaml")"
sed -E "s|(rankfm_mvai_tpu):[A-Za-z0-9._-]+|\1:${IMAGE_TAG}|g" "${YAML}" > "${PATCHED}"

# Override an env var's value in place if the "- name: <n>" entry exists; else
# append a new entry right after FWD_EMB_BACKEND (14-space env indentation).
_set_env() {  # $1=name $2=value (skips empty)
    local n="$1" v="$2"
    [ -z "${v}" ] && return
    if grep -qE "^[[:space:]]*- name: ${n}$" "${PATCHED}"; then
        awk -v n="${n}" -v v="${v}" '
            $0 ~ "- name: "n"$" { print; getline; sub(/value:.*/, "value: \"" v "\""); print; next }
            { print }
        ' "${PATCHED}" > "${PATCHED}.tmp" && mv "${PATCHED}.tmp" "${PATCHED}"
    else
        awk -v n="${n}" -v v="${v}" '
            /- name: FWD_EMB_BACKEND$/ {
                print; getline; print
                printf "              - name: %s\n                value: \"%s\"\n", n, v
                next
            }
            { print }
        ' "${PATCHED}" > "${PATCHED}.tmp" && mv "${PATCHED}.tmp" "${PATCHED}"
    fi
    echo "  set env: ${n}=${v}" | tee -a "${LOG}"
}

_set_env FWD_DEVICE "${FWD_DEVICE:-}"
_set_env FWD_EMB_BACKEND "${FWD_EMB_BACKEND:-}"
_set_env FWD_SWEEP "${FWD_SWEEP:-}"
_set_env FWD_PROFILE "${FWD_PROFILE:-}"
_set_env FWD_PROFILE_MODE "${FWD_PROFILE_MODE:-}"
_set_env FWD_GLOBAL_BS "${FWD_GLOBAL_BS:-}"
_set_env RANKFM_S "${RANKFM_S:-}"
_set_env FWD_NUM_CANDIDATES "${FWD_NUM_CANDIDATES:-}"
_set_env FWD_L1_LAYERS "${FWD_L1_LAYERS:-}"
_set_env FWD_L2_LAYERS "${FWD_L2_LAYERS:-}"
_set_env FWD_L2_SEQ "${FWD_L2_SEQ:-}"
_set_env FWD_HASH_SIZE_CAP "${FWD_HASH_SIZE_CAP:-}"
_set_env FWD_RINGSHIFT_MIN_ROWS "${FWD_RINGSHIFT_MIN_ROWS:-}"
_set_env FWD_FSDP_MIN_ROWS "${FWD_FSDP_MIN_ROWS:-}"
_set_env FWD_WARMUP "${FWD_WARMUP:-}"
_set_env FWD_STEPS "${FWD_STEPS:-}"

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

# 4) Wait for the rank-0 pod, then stream its logs until the run completes.
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
# container name is "rankfm" per the fwd yaml.
kubectl logs -f "${POD}" -c rankfm --context="${CTX}" -n "${NS}" 2>&1 | tee -a "${LOG}"

# 5) Re-fetch the complete log (kubectl logs -f can drop lines under load).
echo | tee -a "${LOG}"
echo "=== complete log (re-fetch) ===" | tee -a "${LOG}"
kubectl logs "${POD}" -c rankfm --context="${CTX}" -n "${NS}" >> "${LOG}" 2>&1 || true

# 6) If a trace was written to GCS: push the rank-0 worker trace to Kernelhub
#    (perfetto browser view, no tunnel) AND stage the xplane.pb locally for
#    xprof. Kernelhub's perfetto json is event-capped and may drop the device
#    "XLA Ops" rows on this image; the local xplane (Op Profile tab, NOT Trace
#    Viewer) is the reliable device view.
echo | tee -a "${LOG}"
if gsutil ls "${TRACE_BUCKET}/${JS}/" >/dev/null 2>&1; then
    echo "=== retrieve trace from ${TRACE_BUCKET}/${JS} ===" | tee -a "${LOG}"
    # Kernelhub push (same as run_tpu_bench.sh).
    SRC_TRACE=$(gsutil ls -r "${TRACE_BUCKET}/${JS}/**" 2>/dev/null \
        | grep "worker-0-0.trace.json.gz$" | head -1)
    if [ -n "${SRC_TRACE}" ] && command -v manifold >/dev/null 2>&1; then
        TMP_TRACE=$(mktemp "/tmp/rankfm-fwd-trace-${JS}-XXXXXX.json.gz")
        if gsutil cp "${SRC_TRACE}" "${TMP_TRACE}" >/dev/null 2>&1; then
            KH_PATH="${KERNELHUB_BASE}/${JS}-worker-0-0.trace.json.gz"
            manifold mkdirs "${KERNELHUB_BUCKET}/${KERNELHUB_BASE}" >/dev/null 2>&1 || true
            if manifold put --overwrite "${TMP_TRACE}" "${KERNELHUB_BUCKET}/${KH_PATH}" >/dev/null 2>&1; then
                KH_ENC=$(printf '%s' "${KH_PATH}" | sed 's:/:%2F:g')
                KH_URL="https://www.internalfb.com/intern/kernelhub/perfetto/?bucket=${KERNELHUB_BUCKET}&trace_path=${KH_ENC}"
                echo "Kernelhub trace (perfetto):" | tee -a "${LOG}"
                echo "  ${KH_URL}" | tee -a "${LOG}"
            else
                echo "  (manifold upload failed; trace still in GCS: ${TRACE_BUCKET}/${JS})" | tee -a "${LOG}"
            fi
        fi
        rm -f "${TMP_TRACE}"
    fi
    # Local xplane staging for xprof Op Profile.
    DEST="${XPLANE_LOCAL_DIR}/${JS}"
    mkdir -p "${DEST}"
    if gsutil -m cp -r "${TRACE_BUCKET}/${JS}/*" "${DEST}/" >/dev/null 2>&1; then
        echo "  local xplane (Op Profile tab, NOT Trace Viewer):" | tee -a "${LOG}"
        echo "    xprof --logdir ${DEST}" | tee -a "${LOG}"
    fi
else
    echo "No trace at ${TRACE_BUCKET}/${JS}/ (profiling off, or none written)." | tee -a "${LOG}"
fi

echo | tee -a "${LOG}"
echo "================================================================" | tee -a "${LOG}"
echo "Done. JobSet: ${JS}" | tee -a "${LOG}"
echo "  full log: ${LOG}" | tee -a "${LOG}"
[ -n "${KH_URL}" ] && echo "  kernelhub: ${KH_URL}" | tee -a "${LOG}"
echo "  delete  : kubectl delete jobset ${JS} --context=${CTX} -n ${NS}" | tee -a "${LOG}"
echo "================================================================"
