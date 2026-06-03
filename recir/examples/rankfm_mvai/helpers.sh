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
# Override CTX in your environment to point at a different cluster, or NS to
# target a different team namespace:
#     export CTX=gke_<project>_<region>_<cluster>
#     export NS=ns-<your-team>
#
# Defaults assume the prod training cluster documented in
# fbcode/mrs/public_cloud/tpu/users/README.md and the Enablement MRS
# Foundation namespace.

_RANKFM_MVAI_HELPERS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_RANKFM_MVAI_FBCODE_ROOT="$(cd "${_RANKFM_MVAI_HELPERS_DIR}/../../.." && pwd)"
_RANKFM_MVAI_DEFAULT_YAML="${_RANKFM_MVAI_HELPERS_DIR}/rankfm_mvai-128chip.yaml"
_RANKFM_MVAI_BUILD_SCRIPT="${_RANKFM_MVAI_HELPERS_DIR}/build_image_podman.sh"

: "${CTX:=gke_mrs-tpu-y317h_us-central1_tpu-prod-training-cluster}"
: "${NS:=ns-enablement-mrs-foundation}"
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
    # Retry: when called manually the rankfm-mvai container may still be
    # ContainerCreating/PodInitializing (image pull, barnfs CSI / gcsfuse mount),
    # which makes `kubectl logs` fail with a BadRequest until it starts.
    local attempt
    for attempt in $(seq 1 60); do
        kubectl logs -f "$rank0" --container rankfm-mvai \
            --context="$CTX" -n "$NS" && return 0
        echo "  rank-0 container not ready yet (attempt $attempt); retrying..." >&2
        sleep 5
    done
    echo "rank-0 logs still unavailable after retries; check 'rankfm_mvai_status'." >&2
    return 1
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
    echo "Waiting for rank 0's container to start before tailing..."
    # Wait for the rank-0 pod's 'rankfm-mvai' container to actually be Running --
    # NOT just for any pod's phase to be Running. With 32 pods, another pod
    # reaches Running while rank-0 is still ContainerCreating/PodInitializing
    # (image pull, barnfs CSI mount, gcsfuse sidecar), so tailing rank-0 then
    # fails with "is waiting to start: PodInitializing". Poll rank-0's container
    # state.running instead.
    local rank0=""
    until [ -n "$rank0" ] && kubectl get pod "$rank0" --context="$CTX" -n "$NS" \
        -o jsonpath='{.status.containerStatuses[?(@.name=="rankfm-mvai")].state.running.startedAt}' \
        2>/dev/null | grep -q .; do
        rank0=$(kubectl get pods --context="$CTX" -n "$NS" \
            -l "jobset.sigs.k8s.io/jobset-name=$JS" -o name 2>/dev/null \
            | grep 'worker-0-0-' | head -1 | sed 's|pod/||')
        sleep 5
    done
    rankfm_mvai_logs
}
