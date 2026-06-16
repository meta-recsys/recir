#!/usr/bin/env bash
# Bash helpers for the rankfm_mvai GKE TPU example. Same model as
# recir/examples/rankfm, driven by the mvai_cloud RecTrainer framework.
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
#   rankfm_mvai_logs_all          interleaved tail of all 32 ranks
#   rankfm_mvai_dump_logs         dump every pod's logs to /tmp/ranklogs/, summarize
#   rankfm_mvai_traceback [rank]  show full Traceback from one rank's dumped log
#   rankfm_mvai_kill              delete the current $JS
#   rankfm_mvai_iterate [yaml]    build_image_podman.sh -> submit -> tail rank 0
#
# By default CTX is the currently-active kubeconfig context. Override it to
# point at a different cluster (use `kubectl config get-contexts` to see the
# exact name — newer gcloud uses `<project>/<region>/<cluster>`, older uses
# `gke_<project>_<region>_<cluster>`), or NS to target a different namespace:
#     export CTX=mrs-tpu-y317h/us-central1/tpu-prod-training-cluster
#     export NS=ns-<your-team>
#
# Defaults assume the prod training cluster documented in
# fbcode/mrs/public_cloud/tpu/users/README.md and the MRS Core Shared
# namespace.

_RANKFM_MVAI_HELPERS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_RANKFM_MVAI_FBCODE_ROOT="$(cd "${_RANKFM_MVAI_HELPERS_DIR}/../../.." && pwd)"
_RANKFM_MVAI_DEFAULT_YAML="${_RANKFM_MVAI_HELPERS_DIR}/rankfm_mvai-128chip.yaml"
_RANKFM_MVAI_BUILD_SCRIPT="${_RANKFM_MVAI_HELPERS_DIR}/build_image_podman.sh"

# Context name format varies by gcloud version: older gcloud mints
# `gke_<project>_<region>_<cluster>` (underscores), newer gcloud mints
# `<project>/<region>/<cluster>` (slashes). Rather than hardcode one and
# break on the other, default to whatever context is currently active in
# the kubeconfig (set by `gcloud container clusters get-credentials`).
# Override CTX explicitly if you target a different cluster.
: "${CTX:=$(kubectl config current-context 2>/dev/null)}"
: "${CTX:=mrs-tpu-y317h/us-central1/tpu-prod-training-cluster}"
# Self-heal a stale CTX: if the configured context isn't actually present in
# the kubeconfig (e.g. inherited from an older `source helpers.sh` that used
# the legacy `gke_<project>_<region>_<cluster>` name), fall back to whatever
# context is currently active rather than failing every kubectl call.
if ! kubectl config get-contexts -o name 2>/dev/null | grep -qxF "$CTX"; then
    _ctx_active="$(kubectl config current-context 2>/dev/null)"
    if [ -n "$_ctx_active" ]; then
        echo "helpers.sh: CTX '$CTX' not found in kubeconfig; using active context '$_ctx_active'" >&2
        CTX="$_ctx_active"
    fi
    unset _ctx_active
fi
: "${NS:=ns-mrs-core-shared}"
export CTX NS

# Meta devservers need the corp proxy for kubectl / gcloud / podman push.
# These are no-ops if you've already exported them elsewhere.
: "${HTTPS_PROXY:=http://fwdproxy:8080}"
: "${HTTP_PROXY:=http://fwdproxy:8080}"
: "${NO_PROXY:=localhost,127.0.0.1,.googleapis.com}"
export HTTPS_PROXY HTTP_PROXY NO_PROXY

rankfm_mvai_submit() {
    local yaml="${1:-$_RANKFM_MVAI_DEFAULT_YAML}"
    if [ ! -f "$yaml" ]; then
        echo "Yaml not found: $yaml" >&2
        return 1
    fi
    local yaml_ns
    yaml_ns=$(awk '/^[[:space:]]*namespace:/ {print $2; exit}' "$yaml")
    if [ -n "$yaml_ns" ]; then
        NS="$yaml_ns"
        export NS
    fi
    local js_name
    js_name=$(kubectl create -f "$yaml" --context="$CTX" -n "$NS" \
        -o jsonpath='{.metadata.name}')
    export JS="$js_name"
    echo "Submitted: $JS (namespace: $NS)"
}

rankfm_mvai_status() {
    [ -z "$JS" ] && { echo "JS not set; run rankfm_mvai_submit first" >&2; return 1; }
    kubectl get pods --context="$CTX" -n "$NS" \
        -l "jobset.sigs.k8s.io/jobset-name=$JS" -o wide
}

rankfm_mvai_watch() {
    [ -z "$JS" ] && { echo "JS not set" >&2; return 1; }
    kubectl get pods --context="$CTX" -n "$NS" \
        -l "jobset.sigs.k8s.io/jobset-name=$JS" -w
}

rankfm_mvai_logs() {
    [ -z "$JS" ] && { echo "JS not set" >&2; return 1; }
    local rank0
    rank0=$(kubectl get pods --context="$CTX" -n "$NS" \
        -l "jobset.sigs.k8s.io/jobset-name=$JS" \
        -o name | grep 'worker-0-0-' | head -1 | sed 's|pod/||')
    if [ -z "$rank0" ]; then
        echo "No rank-0 pod found for $JS" >&2
        return 1
    fi
    echo "Tailing: $rank0"
    kubectl logs -f "$rank0" --context="$CTX" -n "$NS"
}

rankfm_mvai_logs_all() {
    [ -z "$JS" ] && { echo "JS not set" >&2; return 1; }
    kubectl logs -l "jobset.sigs.k8s.io/jobset-name=$JS" \
        --context="$CTX" -n "$NS" --max-log-requests=64 --prefix --tail=200 -f
}

rankfm_mvai_dump_logs() {
    [ -z "$JS" ] && { echo "JS not set" >&2; return 1; }
    mkdir -p /tmp/ranklogs
    rm -f /tmp/ranklogs/*.log
    for pod in $(kubectl get pods --context="$CTX" -n "$NS" \
        -l "jobset.sigs.k8s.io/jobset-name=$JS" -o name); do
        local name=${pod#pod/}
        kubectl logs "$name" --context="$CTX" -n "$NS" > "/tmp/ranklogs/${name}.log" 2>&1 &
    done
    wait
    echo "Dumped $(find /tmp/ranklogs/ -maxdepth 1 -name '*.log' | wc -l) logs to /tmp/ranklogs/"
    echo
    echo "=== Per-rank exception summary ==="
    for f in /tmp/ranklogs/*.log; do
        local rank
        rank=$(basename "$f" .log)
        local err
        err=$(grep -E "^\[rank[0-9]+\]: [A-Za-z_]+(Error|Exception): " "$f" | tail -1)
        if [ -z "$err" ]; then
            local last
            last=$(tail -1 "$f" | tr '\n' ' ')
            echo "$rank: NO_ERROR -- last: $last"
        else
            echo "$rank: $err"
        fi
    done | sort
}

rankfm_mvai_traceback() {
    local rank="${1:-0}"
    local f
    f=$(find /tmp/ranklogs/ -maxdepth 1 -name "*worker-0-${rank}-*.log" 2>/dev/null | head -1)
    if [ -z "$f" ]; then
        echo "No dumped log for rank $rank. Run rankfm_mvai_dump_logs first." >&2
        return 1
    fi
    echo "=== rank $rank: $f ==="
    grep --color=auto -B 1 -A 80 "Traceback" "$f" | tail -100
}

rankfm_mvai_kill() {
    [ -z "$JS" ] && { echo "JS not set" >&2; return 1; }
    kubectl delete "jobset/$JS" --context="$CTX" -n "$NS"
    unset JS
}

rankfm_mvai_iterate() {
    local yaml="${1:-$_RANKFM_MVAI_DEFAULT_YAML}"
    ( cd "$_RANKFM_MVAI_FBCODE_ROOT" && "$_RANKFM_MVAI_BUILD_SCRIPT" ) || return 1
    rankfm_mvai_submit "$yaml" || return 1
    echo "Waiting for rank 0 to be Running before tailing..."
    until kubectl get pods --context="$CTX" -n "$NS" \
        -l "jobset.sigs.k8s.io/jobset-name=$JS" \
        -o jsonpath='{.items[*].status.phase}' 2>/dev/null \
        | grep -q Running; do
        sleep 5
    done
    rankfm_mvai_logs
}
