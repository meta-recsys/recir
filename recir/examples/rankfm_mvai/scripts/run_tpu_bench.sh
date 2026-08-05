#!/usr/bin/env bash
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#
# One-shot driver for the RankFM-MVAI TPU benchmark run.
#
# Follows the README "Every session" + "Kick off training" flow:
#   cd <fbcode>
#   source recir/examples/rankfm_mvai/helpers.sh   # defines rankfm_mvai_*, exports CTX
#   rankfm_mvai_iterate <yaml>                      # build_image_podman.sh -> kubectl create -> tail rank 0
#
# ...but captures EVERYTHING (build + submit + full rank-0 training log) to a
# single log file, then retrieves the xprof TPU trace from GCS so you can view
# it in xprof / read the per-step stats.
#
# Trace is already enabled by the yaml:
#   --profile_start_step 3 --profile_end_step 13
#   --profile_dir gs://mrs-tpu-y317h/tpu-profiles/<JOBSET_NAME>
#   --enable_memory_log
# RecirXLATrainPipeline.progress calls jax.profiler.start_trace/stop_trace on
# those steps and writes the .xplane.pb to GCS.
#
# Usage:
#   ./run_tpu_bench.sh [yaml]
#
# Set RANKFM_SKIP_BUILD=1 to skip the podman image rebuild and reuse the current
# :latest image (use ONLY when recir/ source is unchanged -- e.g. iterating on a
# yaml env knob like RECIR_RINGSHIFT_RECV_CAPACITY or RANKFM_S):
#   RANKFM_SKIP_BUILD=1 ./run_tpu_bench.sh rankfm_mvai-64chip-bench.yaml
#
# yaml defaults to recir/examples/rankfm_mvai/manifests/rankfm_mvai-4chip-bench.yaml.
# Set RANKFM_S in the yaml (env var) to sweep sequence length {512,1024,2048,3072};
# the value is parsed into the log filename. Logs are saved under
# ~/rankfm_mvai_bench_logs/ (outside the repo, survives /tmp cleanup).

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
FBCODE_ROOT="$(cd "${PKG_DIR}/../../.." && pwd)"
# Resolve the yaml arg to an ABSOLUTE path: helpers (gke_iterate/submit) cd to
# the image-build dir before `kubectl create -f`, so a relative arg would not be
# found at submit time. Accept either a CWD-relative path or a bare name living
# in the package's manifests/ dir.
YAML_ARG="${1:-${PKG_DIR}/manifests/rankfm_mvai-4chip-bench.yaml}"
if [ -f "${YAML_ARG}" ]; then
    YAML="$(cd "$(dirname "${YAML_ARG}")" && pwd)/$(basename "${YAML_ARG}")"
elif [ -f "${PKG_DIR}/manifests/${YAML_ARG}" ]; then
    YAML="${PKG_DIR}/manifests/${YAML_ARG}"
else
    YAML="${YAML_ARG}"
fi

# Sequence length S for this run, parsed from the yaml's RANKFM_S env var so the
# log filename records exactly what was run.
S="$(awk '/name: RANKFM_S/{f=1; next} f && /value:/{v=$2; gsub(/"/,"",v); print v; exit}' "${YAML}" 2>/dev/null)"
[ -z "${S}" ] && S="unknown"

TS="$(date +%Y%m%d-%H%M%S)"
# Persist logs under $HOME so they survive /tmp cleanup without adding
# hundreds of MB of build output to the repo. Override with
# RANKFM_BENCH_LOG_DIR. Traces are NOT saved locally -- they stay in GCS and
# are pushed to Kernelhub (see below).
LOG_DIR="${RANKFM_BENCH_LOG_DIR:-${HOME}/rankfm_mvai_bench_logs}"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/tpu-s${S}-${TS}.log"
# Stable per-seqlen symlink for analysis to point at.
LATEST="${LOG_DIR}/tpu-s${S}-latest.log"

# GCS bucket where the xprof trace + tensorboard land (from the yaml).
TRACE_BUCKET="gs://cq-mrs-core-shared_mrs_tpu_y317h/tpu-profiles"
# Manifold bucket that kernelhub/perfetto reads from. The worker trace is
# uploaded here after the run so it can be opened in the browser directly.
KERNELHUB_BUCKET="perfdoctor_gpu_traces_test"
KERNELHUB_BASE="tree/traces/test/${USER}/rankfm_mvai"
KH_URL=""

echo "================================================================" | tee "${LOG}"
echo "RankFM-MVAI TPU benchmark run" | tee -a "${LOG}"
echo "  fbcode root : ${FBCODE_ROOT}" | tee -a "${LOG}"
echo "  yaml        : ${YAML}" | tee -a "${LOG}"
echo "  seq len S   : ${S}" | tee -a "${LOG}"
echo "  log file    : ${LOG}" | tee -a "${LOG}"
echo "  started     : $(date)" | tee -a "${LOG}"
echo "================================================================" | tee -a "${LOG}"

if [ ! -f "${YAML}" ]; then
    echo "ERROR: yaml not found: ${YAML}" | tee -a "${LOG}"
    exit 1
fi

cd "${FBCODE_ROOT}" || exit 1

# helpers.sh defines rankfm_* and exports CTX/NS + HTTPS_PROXY for kubectl/gcloud.
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/helpers.sh"

# build_image_podman.sh -> rankfm_submit (exports $JS) -> wait Running -> tail rank 0.
# Tee the whole thing so build + submit + streamed training log are all captured.
# NOTE: use process substitution (`> >(tee ...)`), NOT a pipe (`| tee`). A pipe
# runs rankfm_mvai_iterate in a subshell, so its `export JS=...` would be lost here.
# RANKFM_SKIP_BUILD=1 -> skip the podman rebuild (gke_iterate reads GKE_SKIP_BUILD).
export GKE_SKIP_BUILD="${RANKFM_SKIP_BUILD:-0}"
rankfm_mvai_iterate "${YAML}" > >(tee -a "${LOG}") 2>&1
# Let the process-substitution tee drain before we append more to ${LOG}.
sleep 1

# $JS was exported by gke_submit inside rankfm_mvai_iterate (same shell).
if [ -z "${JS:-}" ]; then
    echo "WARNING: JS not set; submit may have failed. See ${LOG}." | tee -a "${LOG}"
    ln -sf "${LOG}" "${LATEST}"
    exit 1
fi

echo | tee -a "${LOG}"
echo "==== Wait for run to finish / trace to flush (kubectl logs -f drops mid-compile) ====" | tee -a "${LOG}"
# gke_logs' kubectl logs -f frequently drops under load during the long (~3min)
# XLA compile and returns while the pod is still Running -- long before the
# profiler writes the trace at profile_end_step. Previously we then checked GCS
# once, found no trace, and exited. Instead, poll until the rank-0 pod reaches a
# terminal phase OR the worker-0-0 trace appears in GCS, printing a heartbeat.
_deadline=$(( $(date +%s) + 3600 ))
while :; do
    RANK0="$(kubectl get pods --context="${CTX}" -n "${NS}" \
        -l "jobset.sigs.k8s.io/jobset-name=${JS}" -o name 2>/dev/null \
        | grep 'worker-0-0-' | head -1 | sed 's|pod/||')"
    if [ -z "${RANK0}" ]; then
        echo "  rank-0 pod not found; stopping wait." | tee -a "${LOG}"
        break
    fi
    _phase="$(kubectl get pod "${RANK0}" --context="${CTX}" -n "${NS}" \
        -o jsonpath='{.status.phase}' 2>/dev/null)"
    if [ "${_phase}" = "Succeeded" ] || [ "${_phase}" = "Failed" ]; then
        echo "  rank-0 phase=${_phase}; done." | tee -a "${LOG}"
        break
    fi
    if gsutil ls -r "${TRACE_BUCKET}/${JS}/**" 2>/dev/null | grep -q "worker-0-0.trace.json.gz"; then
        echo "  trace flushed to GCS; proceeding to retrieval." | tee -a "${LOG}"
        break
    fi
    if [ "$(date +%s)" -ge "${_deadline}" ]; then
        echo "  timed out waiting for finish/trace (60min)." | tee -a "${LOG}"
        break
    fi
    _last="$(kubectl logs "${RANK0}" --context="${CTX}" -n "${NS}" --tail=1 2>/dev/null | tr '\n' ' ' | cut -c1-160)"
    echo "  [waiting] phase=${_phase:-?} :: ${_last}" | tee -a "${LOG}"
    sleep 20
done

echo | tee -a "${LOG}"
echo "==== Post-run: re-fetch COMPLETE rank-0 log (kubectl logs -f can drop lines) ====" | tee -a "${LOG}"
# Non-follow pull of the full rank-0 log; appended so the file is authoritative
# even if the live -f stream was rotated/truncated by kubelet under load.
RANK0="$(kubectl get pods --context="${CTX}" -n "${NS}" \
    -l "jobset.sigs.k8s.io/jobset-name=${JS}" -o name 2>/dev/null \
    | grep 'worker-0-0-' | head -1 | sed 's|pod/||')"
if [ -n "${RANK0}" ]; then
    {
        echo "----- BEGIN complete rank-0 log (${RANK0}) -----"
        kubectl logs "${RANK0}" --context="${CTX}" -n "${NS}" 2>&1
        echo "----- END complete rank-0 log -----"
    } >> "${LOG}"
    echo "Appended complete rank-0 log for ${RANK0}." | tee -a "${LOG}"
else
    echo "WARNING: could not find rank-0 pod to re-fetch log." | tee -a "${LOG}"
fi

echo | tee -a "${LOG}"
echo "==== Retrieve xprof TPU trace from ${TRACE_BUCKET}/${JS} ====" | tee -a "${LOG}"
if gsutil ls "${TRACE_BUCKET}/${JS}/" >/dev/null 2>&1; then
    # Traces are pushed to Kernelhub (perfetto) and remain in GCS; we do NOT keep
    # a local copy of the (multi-GB) full trace dir. Download only the rank-0
    # worker trace to a temp file for the Manifold/Kernelhub upload, then delete
    # it. To inspect the full multi-host trace, open the GCS path in xprof.
    echo "Trace in GCS (open in xprof):  ${TRACE_BUCKET}/${JS}" | tee -a "${LOG}"
    echo "  xprof --logdir ${TRACE_BUCKET}/${JS}" | tee -a "${LOG}"
    SRC_TRACE=$(gsutil ls -r "${TRACE_BUCKET}/${JS}/**" 2>/dev/null \
        | grep "worker-0-0.trace.json.gz$" | head -1)
    if [ -n "${SRC_TRACE}" ] && command -v manifold >/dev/null 2>&1; then
        TMP_TRACE=$(mktemp "/tmp/rankfm-mvai-trace-${JS}-XXXXXX.json.gz")
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
else
    echo "No trace found yet at ${TRACE_BUCKET}/${JS}/ (profiling steps may not" | tee -a "${LOG}"
    echo "have been reached, or the job is still flushing). Check later with:" | tee -a "${LOG}"
    echo "  gsutil ls ${TRACE_BUCKET}/${JS}/" | tee -a "${LOG}"
fi

ln -sf "${LOG}" "${LATEST}"

echo | tee -a "${LOG}"
echo "================================================================" | tee -a "${LOG}"
echo "Done. JobSet: ${JS}" | tee -a "${LOG}"
echo "  full log : ${LOG}" | tee -a "${LOG}"
echo "  latest   : ${LATEST}" | tee -a "${LOG}"
echo "  trace    : ${TRACE_BUCKET}/${JS} (GCS; not saved locally)" | tee -a "${LOG}"
[ -n "${KH_URL}" ] && echo "  kernelhub: ${KH_URL}" | tee -a "${LOG}"
echo "================================================================" | tee -a "${LOG}"
