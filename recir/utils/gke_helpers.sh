#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# Shared bash helpers for submitting and monitoring GKE TPU JobSets. Sourced by
# recir/tests/helpers.sh and recir/fb/examples/rankfm_mvai/helpers.sh -- not meant to be
# sourced directly.
#
# A wrapper helpers.sh should:
#   1. Set the per-example config vars (see below).
#   2. source this file.
#   3. Call gke_init "<default_ctx>" "<default_ns>".
#   4. Define thin command-name wrappers (e.g. rankfm_submit -> gke_submit) and,
#      optionally, override _gke_summarize_line to customize gke_dump_logs.
#
# Config vars (set before calling the functions that use them):
#   CTX                 kubectl context (resolved by gke_init / gke_resolve_ctx)
#   NS                  kubernetes namespace (defaulted by gke_init)
#   JS                  current JobSet name (exported by gke_submit)
#   _GKE_DEFAULT_YAML   default yaml for gke_submit / gke_iterate
#   _GKE_LOG_DIR        local dir for dumped pod logs (e.g. /tmp/ranklogs)
#   _GKE_POD_NOUN       human label for the rank-0 pod (e.g. "rank-0 pod")
#   _GKE_SUMMARY_TITLE  heading printed by gke_dump_logs
#   _GKE_BUILD_SCRIPT   path to build_image_podman.sh (for gke_iterate)
#   _GKE_FBCODE_ROOT    fbcode root to cd into for the build (for gke_iterate)

# Resolve $CTX: prefer an explicit env value, else the currently-active
# kubeconfig context, else the provided default. Context name format varies by
# gcloud version: older gcloud mints `gke_<project>_<region>_<cluster>`
# (underscores), newer gcloud mints `<project>/<region>/<cluster>` (slashes).
# Rather than hardcode one and break on the other, default to whatever context
# is currently active in the kubeconfig (set by
# `gcloud container clusters get-credentials`).
#
# Then self-heal a stale $CTX: if it isn't actually present in the kubeconfig
# (e.g. inherited from an older `source helpers.sh` that used the legacy
# gke_<project>_<region>_<cluster> name), fall back to whatever context is
# currently active rather than failing every kubectl call.
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
#
gke_resolve_ctx() {
    local default_ctx="$1"
    : "${CTX:=$(kubectl config current-context 2>/dev/null)}"
    : "${CTX:=$default_ctx}"
    if ! kubectl config get-contexts -o name 2>/dev/null | grep -qxF "$CTX"; then
        local ctx_active
        ctx_active="$(kubectl config current-context 2>/dev/null)"
        if [ -n "$ctx_active" ]; then
            echo "helpers.sh: CTX '$CTX' not found in kubeconfig; using active context '$ctx_active'" >&2
            CTX="$ctx_active"
        fi
    fi
    export CTX
}

# Meta devservers need the corp proxy for kubectl / gcloud / podman push.
# These are no-ops if you've already exported them elsewhere.
gke_setup_proxy() {
    : "${HTTPS_PROXY:=http://fwdproxy:8080}"
    : "${HTTP_PROXY:=http://fwdproxy:8080}"
    : "${NO_PROXY:=localhost,127.0.0.1,.googleapis.com}"
    export HTTPS_PROXY HTTP_PROXY NO_PROXY
}

# One-call setup: resolve CTX, default NS, and set up the corp proxy.
gke_init() {
    local default_ctx="$1" default_ns="$2"
    gke_resolve_ctx "$default_ctx"
    : "${NS:=$default_ns}"
    export NS
    gke_setup_proxy
}

gke_submit() {
    local yaml="${1:-$_GKE_DEFAULT_YAML}"
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

gke_status() {
    [ -z "$JS" ] && { echo "JS not set; run *_submit first" >&2; return 1; }
    kubectl get pods --context="$CTX" -n "$NS" \
        -l "jobset.sigs.k8s.io/jobset-name=$JS" -o wide
}

gke_watch() {
    [ -z "$JS" ] && { echo "JS not set" >&2; return 1; }
    kubectl get pods --context="$CTX" -n "$NS" \
        -l "jobset.sigs.k8s.io/jobset-name=$JS" -w
}

gke_logs() {
    [ -z "$JS" ] && { echo "JS not set" >&2; return 1; }
    local pod0
    pod0=$(kubectl get pods --context="$CTX" -n "$NS" \
        -l "jobset.sigs.k8s.io/jobset-name=$JS" \
        -o name | grep 'worker-0-0-' | head -1 | sed 's|pod/||')
    if [ -z "$pod0" ]; then
        echo "No ${_GKE_POD_NOUN:-rank-0 pod} found for $JS" >&2
        return 1
    fi
    echo "Tailing: $pod0"
    kubectl logs -f "$pod0" --context="$CTX" -n "$NS"
}

gke_logs_all() {
    [ -z "$JS" ] && { echo "JS not set" >&2; return 1; }
    kubectl logs -l "jobset.sigs.k8s.io/jobset-name=$JS" \
        --context="$CTX" -n "$NS" --max-log-requests=64 --prefix --tail=200 -f
}

# Per-pod one-line summary printed by gke_dump_logs. Wrappers override this to
# match their workload (unittest result line, per-rank exception, etc.). The
# default just reports the last line of each log.
_gke_summarize_line() {
    local f="$1" name last
    name=$(basename "$f" .log)
    last=$(tail -1 "$f" | tr '\n' ' ')
    echo "$name: $last"
}

gke_dump_logs() {
    [ -z "$JS" ] && { echo "JS not set" >&2; return 1; }
    local logdir="${_GKE_LOG_DIR:?_GKE_LOG_DIR not set}"
    mkdir -p "$logdir"
    rm -f "$logdir"/*.log
    for pod in $(kubectl get pods --context="$CTX" -n "$NS" \
        -l "jobset.sigs.k8s.io/jobset-name=$JS" -o name); do
        local name=${pod#pod/}
        kubectl logs "$name" --context="$CTX" -n "$NS" > "${logdir}/${name}.log" 2>&1 &
    done
    wait
    echo "Dumped $(find "$logdir" -maxdepth 1 -name '*.log' | wc -l) logs to ${logdir}/"
    echo
    echo "=== ${_GKE_SUMMARY_TITLE:-Per-pod summary} ==="
    for f in "$logdir"/*.log; do
        _gke_summarize_line "$f"
    done | sort
}

gke_traceback() {
    local rank="${1:-0}"
    local logdir="${_GKE_LOG_DIR:?_GKE_LOG_DIR not set}"
    local f
    f=$(find "$logdir" -maxdepth 1 -name "*worker-0-${rank}-*.log" 2>/dev/null | head -1)
    if [ -z "$f" ]; then
        echo "No dumped log for rank $rank. Run *_dump_logs first." >&2
        return 1
    fi
    echo "=== rank $rank: $f ==="
    grep --color=auto -B 1 -A 80 "Traceback" "$f" | tail -100
}

gke_kill() {
    [ -z "$JS" ] && { echo "JS not set" >&2; return 1; }
    kubectl delete "jobset/$JS" --context="$CTX" -n "$NS"
    unset JS
}

gke_iterate() {
    local yaml="${1:-$_GKE_DEFAULT_YAML}"
    # GKE_SKIP_BUILD=1 reuses the current :latest image (skip the podman rebuild
    # when the source is unchanged); unset/0 rebuilds as before.
    if [ "${GKE_SKIP_BUILD:-0}" = "1" ]; then
        echo "GKE_SKIP_BUILD=1 -> skipping image build (reusing current :latest)."
    else
        ( cd "$_GKE_FBCODE_ROOT" && "$_GKE_BUILD_SCRIPT" ) || return 1
    fi
    gke_submit "$yaml" || return 1
    echo "Waiting for the ${_GKE_POD_NOUN:-rank-0 pod} to be Running before tailing..."
    until kubectl get pods --context="$CTX" -n "$NS" \
        -l "jobset.sigs.k8s.io/jobset-name=$JS" \
        -o jsonpath='{.items[*].status.phase}' 2>/dev/null \
        | grep -q Running; do
        sleep 5
    done
    gke_logs
}
