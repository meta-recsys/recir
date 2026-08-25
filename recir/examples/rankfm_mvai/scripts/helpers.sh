#!/usr/bin/env bash
# Bash helpers for the rankfm_mvai GKE TPU example (the hammer udd_fm
# BlueReelsVDDTrainV3 driven by the mvai_cloud RecTrainer framework).
# Source from your shell:
#
#     source recir/examples/rankfm_mvai/helpers.sh
#
# Adds the following functions to your interactive shell:
#
#   rankfm_mvai_submit  [yaml]   submit a JobSet, exports JS=<jobset name>
#   rankfm_mvai_status            kubectl get pods -l jobset=$JS -o wide
#   rankfm_mvai_watch             kubectl get pods -w
#   rankfm_mvai_logs              tail logs from rank 0 (worker-0-0-*)
#   rankfm_mvai_logs_all          interleaved tail of all ranks
#   rankfm_mvai_dump_logs         dump every pod's logs to /tmp/ranklogs/, summarize
#   rankfm_mvai_traceback [rank]  show full Traceback from one rank's dumped log
#   rankfm_mvai_kill              delete the current $JS
#   rankfm_mvai_iterate [yaml]    build_image_podman.sh -> submit -> tail rank 0
#                                 (set GKE_SKIP_BUILD=1 to reuse the :latest image)
#   rankfm_mvai_tb_to_manifold [job]  copy a run's TensorBoard events GCS -> Manifold
#
# By default CTX is the currently-active kubeconfig context. Override it to
# point at a different cluster (use `kubectl config get-contexts` to see the
# exact name -- newer gcloud uses `<project>/<region>/<cluster>`, older uses
# `gke_<project>_<region>_<cluster>`), or NS to target a different namespace:
#     export CTX=mrs-tpu-y317h/us-central1/tpu-prod-training-cluster
#     export NS=ns-<your-team>
#
# Defaults assume the prod training cluster documented in
# fbcode/mrs/public_cloud/tpu/users/README.md and the MRS Core Shared namespace.

_RANKFM_MVAI_HELPERS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_RANKFM_MVAI_PKG_DIR="$(cd "${_RANKFM_MVAI_HELPERS_DIR}/.." && pwd)"
_RANKFM_MVAI_FBCODE_ROOT="$(cd "${_RANKFM_MVAI_PKG_DIR}/../../.." && pwd)"

# Shared GKE TPU JobSet helpers (gke_* functions, CTX/NS resolution, proxy).
# shellcheck source=/dev/null
source "${_RANKFM_MVAI_FBCODE_ROOT}/recir/utils/gke_helpers.sh"

_GKE_DEFAULT_YAML="${_RANKFM_MVAI_PKG_DIR}/manifests/rankfm_mvai-128chip.yaml"
_GKE_FBCODE_ROOT="$_RANKFM_MVAI_FBCODE_ROOT"
_GKE_BUILD_SCRIPT="${_RANKFM_MVAI_HELPERS_DIR}/build_image_podman.sh"
_GKE_LOG_DIR="/tmp/ranklogs"
_GKE_POD_NOUN="rank-0 pod"
_GKE_SUMMARY_TITLE="Per-rank exception summary"

gke_init "mrs-tpu-y317h/us-central1/tpu-prod-training-cluster" "ns-mrs-core-shared"

# TensorBoard archival (rankfm_mvai_tb_to_manifold): the TPU pods write TB event
# files to gs://$TB_GCS_BUCKET/tensorboard/$JOB (the ns-mrs-core-shared SA-owned
# bucket, matching the yaml). The pods can only reach GCS (GKE Workload
# Identity), so we mirror a run's events into a Manifold bucket that the
# internal `tensorboard` build / on-demand service can read directly.
: "${TB_GCS_BUCKET:=cq-mrs-core-shared_mrs_tpu_y317h}"
: "${TB_MANIFOLD_PATH:=mrs_recsys_gcp/tree/tensorboard}"
: "${TB_MANIFOLD_TTL:=2592000}"  # 30d; mrs_recsys_gcp is a user-data bucket w/ TTL deletion plan
export TB_GCS_BUCKET TB_MANIFOLD_PATH TB_MANIFOLD_TTL

# Torch prints "[rankN]: SomeError: ..." on the failing rank.
_gke_summarize_line() {
    local f="$1" rank err
    rank=$(basename "$f" .log)
    err=$(grep -E "^\[rank[0-9]+\]: [A-Za-z_]+(Error|Exception): " "$f" | tail -1)
    if [ -z "$err" ]; then
        local last
        last=$(tail -1 "$f" | tr '\n' ' ')
        echo "$rank: NO_ERROR -- last: $last"
    else
        echo "$rank: $err"
    fi
}

# Resolve a yaml arg: if it is a bare name (not an existing path), look for it
# next to this helpers.sh (the rankfm_mvai dir, where the *.yaml live). This lets
# callers pass just the filename, e.g. `rankfm_mvai_submit rankfm_mvai-helion-4chip.yaml`,
# regardless of CWD. An empty arg or an existing path is passed through unchanged
# (gke_submit/gke_iterate then apply the absolute _GKE_DEFAULT_YAML default).
_rankfm_mvai_yaml() {
    local y="$1"
    if [ -n "$y" ] && [ ! -f "$y" ] && [ -f "${_RANKFM_MVAI_HELPERS_DIR}/${y}" ]; then
        echo "${_RANKFM_MVAI_HELPERS_DIR}/${y}"
    else
        echo "$y"
    fi
}

rankfm_mvai_submit()    { gke_submit "$(_rankfm_mvai_yaml "$1")"; }
rankfm_mvai_status()    { gke_status; }
rankfm_mvai_watch()     { gke_watch; }
rankfm_mvai_logs()      { gke_logs; }
rankfm_mvai_logs_all()  { gke_logs_all; }
rankfm_mvai_dump_logs() { gke_dump_logs; }
rankfm_mvai_traceback() { gke_traceback "$@"; }
rankfm_mvai_kill()      { gke_kill; }
rankfm_mvai_iterate()   { gke_iterate "$(_rankfm_mvai_yaml "$1")"; }

# Copy a run's TensorBoard event files from GCS to Manifold so they can be
# viewed with the internal `tensorboard` build or the on-demand TB service
# (pip/OSS tensorboard cannot read manifold://). Usage:
#   rankfm_mvai_tb_to_manifold [jobset-name]   # defaults to current $JS
# View afterwards:
#   tensorboard --logdir manifold://$TB_MANIFOLD_PATH/<job>
rankfm_mvai_tb_to_manifold() {
    local job="${1:-$JS}"
    if [ -z "$job" ]; then
        echo "usage: rankfm_mvai_tb_to_manifold <jobset-name> (or submit first to set \$JS)" >&2
        return 1
    fi
    local src="gs://${TB_GCS_BUCKET}/tensorboard/${job}"
    local dst="${TB_MANIFOLD_PATH}/${job}"
    local tmp
    tmp=$(mktemp -d "/tmp/tb-${job}.XXXXXX") || return 1

    echo "[1/2] Downloading ${src}/ -> ${tmp} ..."
    if ! gcloud storage cp -r "${src}/*" "${tmp}/"; then
        echo "No TensorBoard data at ${src} (did the run log any scalars yet?)" >&2
        rm -rf "$tmp"
        return 1
    fi

    # mrs_recsys_gcp is a user-data bucket with a TTL deletion plan: writes are
    # rejected (HTTP 400) unless both --userData and --ttl are specified.
    echo "[2/2] Uploading -> manifold://${dst} (isUserData=true, ttl=${TB_MANIFOLD_TTL}s) ..."
    if ! manifold putr --userData=true --ttl="${TB_MANIFOLD_TTL}" "${tmp}" "${dst}"; then
        echo "manifold upload failed" >&2
        rm -rf "$tmp"
        return 1
    fi

    rm -rf "$tmp"
    echo "Done. View with:"
    echo "  tensorboard --logdir manifold://${dst}"
    echo "  https://www.internalfb.com/intern/tensorboard/?dir=manifold://${dst}"
}
