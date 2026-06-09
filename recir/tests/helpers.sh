#!/usr/bin/env bash
# Bash helpers for running recir/tests/stu_test.py on a GKE TPU slice. Source
# from your shell:
#
#     source recir/tests/helpers.sh
#
# Adds the following functions to your interactive shell:
#
#   stu_submit  [yaml]   submit a JobSet, exports JS=<jobset name>
#   stu_status            kubectl get pods -l jobset=$JS -o wide
#   stu_watch             kubectl get pods -w
#   stu_logs              tail logs from the test pod (worker-0-0-*)
#   stu_logs_all          interleaved tail of all pods
#   stu_dump_logs         dump every pod's logs to /tmp/stulogs/, summarize
#   stu_traceback [rank]  show full Traceback from one pod's dumped log
#   stu_kill              delete the current $JS
#   stu_iterate [yaml]    build_image_podman.sh -> submit -> tail test pod
#
# Override CTX in your environment to point at a different cluster, or NS to
# target a different team namespace:
#     export CTX=gke_<project>_<region>_<cluster>
#     export NS=ns-<your-team>
#
# The image is the same rankfm_tpu image used by the rankfm example
# (recir + generative_recommenders are baked in). stu_test.py ships in that
# image because build_image_podman.sh includes recir/tests/stu_test.py in the
# staged context. The default 4-chip yaml runs the test as a single-node job.

_STU_HELPERS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_STU_FBCODE_ROOT="$(cd "${_STU_HELPERS_DIR}/../.." && pwd)"
_STU_DEFAULT_YAML="${_STU_HELPERS_DIR}/stu-test-4chip.yaml"
# The test image is the rankfm_tpu image; reuse its build script.
_STU_BUILD_SCRIPT="${_STU_FBCODE_ROOT}/recir/examples/rankfm/build_image_podman.sh"

: "${CTX:=gke_mrs-tpu-y317h_us-central1_tpu-prod-training-cluster}"
: "${NS:=ns-enablement-mrs-foundation}"
export CTX NS

# Meta devservers need the corp proxy for kubectl / gcloud / podman push.
# These are no-ops if you've already exported them elsewhere.
: "${HTTPS_PROXY:=http://fwdproxy:8080}"
: "${HTTP_PROXY:=http://fwdproxy:8080}"
: "${NO_PROXY:=localhost,127.0.0.1,.googleapis.com}"
export HTTPS_PROXY HTTP_PROXY NO_PROXY

stu_submit() {
    local yaml="${1:-$_STU_DEFAULT_YAML}"
    if [ ! -f "$yaml" ]; then
        echo "Yaml not found: $yaml" >&2
        return 1
    fi
    local js_name
    js_name=$(kubectl create -f "$yaml" --context="$CTX" -n "$NS" \
        -o jsonpath='{.metadata.name}')
    export JS="$js_name"
    echo "Submitted: $JS (namespace: $NS)"
}

stu_status() {
    [ -z "$JS" ] && { echo "JS not set; run stu_submit first" >&2; return 1; }
    kubectl get pods --context="$CTX" -n "$NS" \
        -l "jobset.sigs.k8s.io/jobset-name=$JS" -o wide
}

stu_watch() {
    [ -z "$JS" ] && { echo "JS not set" >&2; return 1; }
    kubectl get pods --context="$CTX" -n "$NS" \
        -l "jobset.sigs.k8s.io/jobset-name=$JS" -w
}

stu_logs() {
    [ -z "$JS" ] && { echo "JS not set" >&2; return 1; }
    local pod0
    pod0=$(kubectl get pods --context="$CTX" -n "$NS" \
        -l "jobset.sigs.k8s.io/jobset-name=$JS" \
        -o name | grep 'worker-0-0-' | head -1 | sed 's|pod/||')
    if [ -z "$pod0" ]; then
        echo "No test pod found for $JS" >&2
        return 1
    fi
    echo "Tailing: $pod0"
    kubectl logs -f "$pod0" --context="$CTX" -n "$NS"
}

stu_logs_all() {
    [ -z "$JS" ] && { echo "JS not set" >&2; return 1; }
    kubectl logs -l "jobset.sigs.k8s.io/jobset-name=$JS" \
        --context="$CTX" -n "$NS" --max-log-requests=64 --prefix --tail=200 -f
}

stu_dump_logs() {
    [ -z "$JS" ] && { echo "JS not set" >&2; return 1; }
    mkdir -p /tmp/stulogs
    rm -f /tmp/stulogs/*.log
    for pod in $(kubectl get pods --context="$CTX" -n "$NS" \
        -l "jobset.sigs.k8s.io/jobset-name=$JS" -o name); do
        local name=${pod#pod/}
        kubectl logs "$name" --context="$CTX" -n "$NS" > "/tmp/stulogs/${name}.log" 2>&1 &
    done
    wait
    echo "Dumped $(find /tmp/stulogs/ -maxdepth 1 -name '*.log' | wc -l) logs to /tmp/stulogs/"
    echo
    echo "=== Per-pod test-result summary ==="
    for f in /tmp/stulogs/*.log; do
        local pod
        pod=$(basename "$f" .log)
        # unittest prints "OK" / "FAILED (...)" on the final summary line.
        local result
        result=$(grep -E "^(OK|FAILED|ERROR)( |$|\()" "$f" | tail -1)
        if [ -z "$result" ]; then
            local last
            last=$(tail -1 "$f" | tr '\n' ' ')
            echo "$pod: NO_RESULT -- last: $last"
        else
            echo "$pod: $result"
        fi
    done | sort
}

stu_traceback() {
    local rank="${1:-0}"
    local f
    f=$(find /tmp/stulogs/ -maxdepth 1 -name "*worker-0-${rank}-*.log" 2>/dev/null | head -1)
    if [ -z "$f" ]; then
        echo "No dumped log for rank $rank. Run stu_dump_logs first." >&2
        return 1
    fi
    echo "=== rank $rank: $f ==="
    grep --color=auto -B 1 -A 80 "Traceback" "$f" | tail -100
}

stu_kill() {
    [ -z "$JS" ] && { echo "JS not set" >&2; return 1; }
    kubectl delete "jobset/$JS" --context="$CTX" -n "$NS"
    unset JS
}

stu_iterate() {
    local yaml="${1:-$_STU_DEFAULT_YAML}"
    ( cd "$_STU_FBCODE_ROOT" && "$_STU_BUILD_SCRIPT" ) || return 1
    stu_submit "$yaml" || return 1
    echo "Waiting for the test pod to be Running before tailing..."
    until kubectl get pods --context="$CTX" -n "$NS" \
        -l "jobset.sigs.k8s.io/jobset-name=$JS" \
        -o jsonpath='{.items[*].status.phase}' 2>/dev/null \
        | grep -q Running; do
        sleep 5
    done
    stu_logs
}
