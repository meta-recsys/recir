#!/usr/bin/env bash
# Bash helpers for the rankfm GKE TPU example. Source from your shell:
#
#     source recir/examples/rankfm/helpers.sh
#
# Adds the following functions to your interactive shell:
#
#   rankfm_submit  [yaml]   submit a JobSet, exports JS=<jobset name>
#   rankfm_status            kubectl get pods -l jobset=$JS -o wide
#   rankfm_watch             kubectl get pods -w
#   rankfm_logs              tail logs from rank 0 (worker-0-0-*)
#   rankfm_logs_all          interleaved tail of all 32 ranks
#   rankfm_dump_logs         dump every pod's logs to /tmp/ranklogs/, summarize
#   rankfm_traceback [rank]  show full Traceback from one rank's dumped log
#   rankfm_kill              delete the current $JS
#   rankfm_iterate [yaml]    build_image_podman.sh -> submit -> tail rank 0
#
# By default CTX is the currently-active kubeconfig context. Override it to
# point at a different cluster (use `kubectl config get-contexts` to see the
# exact name — newer gcloud uses `<project>/<region>/<cluster>`, older uses
# `gke_<project>_<region>_<cluster>`), or NS to target a different namespace:
#     export CTX=mrs-tpu-y317h/us-central1/tpu-prod-training-cluster
#     export NS=ns-<your-team>
#
# Defaults assume the prod training cluster documented in
# fbcode/mrs/public_cloud/tpu/users/README.md and the Enablement MRS
# Foundation namespace.

_RANKFM_HELPERS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_RANKFM_FBCODE_ROOT="$(cd "${_RANKFM_HELPERS_DIR}/../../.." && pwd)"

# Shared GKE TPU JobSet helpers (gke_* functions, CTX/NS resolution, proxy).
# shellcheck source=/dev/null
source "${_RANKFM_FBCODE_ROOT}/recir/utils/gke_helpers.sh"

_GKE_DEFAULT_YAML="${_RANKFM_HELPERS_DIR}/rankfm-128chip.yaml"
_GKE_FBCODE_ROOT="$_RANKFM_FBCODE_ROOT"
_GKE_BUILD_SCRIPT="${_RANKFM_HELPERS_DIR}/build_image_podman.sh"
_GKE_LOG_DIR="/tmp/ranklogs"
_GKE_POD_NOUN="rank-0 pod"
_GKE_SUMMARY_TITLE="Per-rank exception summary"

gke_init "mrs-tpu-y317h/us-central1/tpu-prod-training-cluster" "ns-enablement-mrs-foundation"

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

rankfm_submit()    { gke_submit "$@"; }
rankfm_status()    { gke_status; }
rankfm_watch()     { gke_watch; }
rankfm_logs()      { gke_logs; }
rankfm_logs_all()  { gke_logs_all; }
rankfm_dump_logs() { gke_dump_logs; }
rankfm_traceback() { gke_traceback "$@"; }
rankfm_kill()      { gke_kill; }
rankfm_iterate()   { gke_iterate "$@"; }
