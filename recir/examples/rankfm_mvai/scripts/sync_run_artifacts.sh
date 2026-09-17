#!/usr/bin/env bash
# Mirror a running job's TensorBoard and xprof trace into Manifold.
#
# The job already writes both to GCS, but GCS is awkward to read from a browser
# and the TensorBoard events file only becomes interesting while the job is
# still going. So this polls: TB is re-synced on an interval so the curves fill
# in live, and the trace is picked up once, when it appears (the profiler only
# flushes at profile_end_step, so it does not exist for the first few hundred
# steps).
#
# Runs until the job reaches a terminal phase and one final TB sync lands, or
# until --max-minutes. Safe to ^C: everything already uploaded stays.
#
#   bash recir/examples/rankfm_mvai/scripts/sync_run_artifacts.sh <jobset-name>
#   bash .../sync_run_artifacts.sh rankfm-mvai-128chip-92jzd --interval 120
#
# Upload targets are the same two buckets the bench script uses: kernelhub for
# the trace (perfetto reads it directly) and mrs_recsys_gcp for TB.

set -uo pipefail

JS="${1:-}"
if [ -z "${JS}" ]; then
    echo "usage: $0 <jobset-name> [--interval SECONDS] [--max-minutes N] [--ns NAMESPACE]" >&2
    exit 2
fi
shift

INTERVAL=180
MAX_MINUTES=720
NS="ns-mrs-core-shared"
while [ $# -gt 0 ]; do
    case "$1" in
        --interval)    INTERVAL="$2"; shift 2 ;;
        --max-minutes) MAX_MINUTES="$2"; shift 2 ;;
        --ns)          NS="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

export HTTPS_PROXY="${HTTPS_PROXY:-http://fwdproxy:8080}"

GCS_BUCKET="gs://cq-mrs-core-shared_mrs_tpu_y317h"
TRACE_SRC="${GCS_BUCKET}/tpu-profiles/${JS}"
TB_SRC="${GCS_BUCKET}/tensorboard/${JS}"

# Trace goes where perfetto can read it; TB where the intern viewer can.
KERNELHUB_BUCKET="perfdoctor_gpu_traces_test"
KERNELHUB_BASE="tree/traces/test/${USER}/rankfm_mvai"
TB_BUCKET="mrs_recsys_gcp"
TB_BASE="tree/tensorboard/${JS}"

TB_URL="https://www.internalfb.com/intern/tensorboard/?dir=manifold://${TB_BUCKET}/${TB_BASE}"

log() { echo "[$(date +%H:%M:%S)] $*"; }

sync_tb() {
    # gsutil rsync rather than cp: the events file grows in place and is
    # re-uploaded whole each time, so -d would be wrong (it would delete the
    # manifold copy mid-write) and plain cp re-transfers unchanged files.
    local tmp
    tmp=$(mktemp -d "/tmp/tbsync-${JS}-XXXXXX")
    if ! gsutil -q -m rsync -r "${TB_SRC}" "${tmp}" 2>/dev/null; then
        rm -rf "${tmp}"
        return 1
    fi
    if [ -z "$(ls -A "${tmp}" 2>/dev/null)" ]; then
        rm -rf "${tmp}"
        return 1
    fi
    manifold mkdirs "${TB_BUCKET}/${TB_BASE}" >/dev/null 2>&1 || true
    # --overwrite matters: without it manifold silently SKIPS files that already
    # exist, so every sync after the first would be a no-op and the curves would
    # stop updating while the command still reported success.
    # Errors are NOT swallowed here. This bucket rejects uploads that do not
    # set --userData, and hiding that made a hard failure look exactly like
    # "no data yet" -- the sync reported success semantics while nothing
    # was ever written.
    local err rc
    err=$(manifold putr --overwrite --userData false "${tmp}" "${TB_BUCKET}/${TB_BASE}" 2>&1)
    rc=$?
    if [ $rc -ne 0 ]; then
        log "  TB upload FAILED: $(echo "${err}" | grep -viE '^[IW][0-9]|WARNING' | head -2 | tr '\n' ' ')"
    fi
    local bytes
    bytes=$(du -sb "${tmp}" 2>/dev/null | cut -f1)
    rm -rf "${tmp}"
    [ $rc -eq 0 ] && log "  TB synced (${bytes} bytes from GCS)"
    return $rc
}

sync_trace() {
    local src
    src=$(gsutil ls -r "${TRACE_SRC}/**" 2>/dev/null \
        | grep "worker-0-0.trace.json.gz$" | head -1)
    [ -z "${src}" ] && return 1

    local tmp kh_path
    tmp=$(mktemp "/tmp/trace-${JS}-XXXXXX.json.gz")
    if ! gsutil -q cp "${src}" "${tmp}" 2>/dev/null; then
        rm -f "${tmp}"
        return 1
    fi
    kh_path="${KERNELHUB_BASE}/${JS}-worker-0-0.trace.json.gz"
    manifold mkdirs "${KERNELHUB_BUCKET}/${KERNELHUB_BASE}" >/dev/null 2>&1 || true
    if manifold put --overwrite --userData false "${tmp}" "${KERNELHUB_BUCKET}/${kh_path}" >/dev/null 2>&1; then
        local enc
        enc=$(printf '%s' "${kh_path}" | sed 's:/:%2F:g')
        log "  trace -> https://www.internalfb.com/intern/kernelhub/perfetto/?bucket=${KERNELHUB_BUCKET}&trace_path=${enc}"
        rm -f "${tmp}"
        return 0
    fi
    rm -f "${tmp}"
    return 1
}

job_is_live() {
    local phases
    phases=$(kubectl get pods -n "${NS}" \
        -l "jobset.sigs.k8s.io/jobset-name=${JS}" --no-headers 2>/dev/null \
        | awk '{print $3}')
    [ -z "${phases}" ] && return 1
    echo "${phases}" | grep -qE "Running|Pending|Init"
}

log "watching ${JS} (interval ${INTERVAL}s, max ${MAX_MINUTES}min)"
log "TB will appear at:"
log "  ${TB_URL}"

TRACE_DONE=0
DEADLINE=$(( $(date +%s) + MAX_MINUTES * 60 ))

while :; do
    sync_tb || log "  TB not ready yet"

    if [ "${TRACE_DONE}" -eq 0 ]; then
        if sync_trace; then
            TRACE_DONE=1
        fi
    fi

    if ! job_is_live; then
        # One last pass: the final metrics flush and the trace can both land
        # after the pods report terminal.
        log "job no longer live; final sync"
        sleep 30
        sync_tb || true
        [ "${TRACE_DONE}" -eq 0 ] && sync_trace
        break
    fi

    [ "$(date +%s)" -ge "${DEADLINE}" ] && { log "hit --max-minutes"; break; }
    sleep "${INTERVAL}"
done

echo
echo "================================================================"
echo "JobSet: ${JS}"
echo "  tensorboard: ${TB_URL}"
[ "${TRACE_DONE}" -eq 1 ] && echo "  trace      : uploaded to kernelhub (link above)"
[ "${TRACE_DONE}" -eq 0 ] && echo "  trace      : never appeared (profile window not reached?)"
echo "  gcs tb     : ${TB_SRC}"
echo "  gcs trace  : ${TRACE_SRC}"
echo "================================================================"
