#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

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
# By default CTX is the currently-active kubeconfig context. Override it to
# point at a different cluster (use `kubectl config get-contexts` to see the
# exact name — newer gcloud uses `<project>/<region>/<cluster>`, older uses
# `gke_<project>_<region>_<cluster>`), or NS to target a different namespace:
#     export CTX=mrs-tpu-y317h/us-central1/tpu-prod-training-cluster
#     export NS=ns-<your-team>
#
# The image is the rankfm_mvai_tpu image used by the rankfm_mvai example
# (recir + generative_recommenders are baked in). stu_test.py ships in that
# image because build_image_podman.sh includes recir/tests/stu_test.py in the
# staged context. The default 4-chip yaml runs the test as a single-node job.

_STU_HELPERS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_STU_FBCODE_ROOT="$(cd "${_STU_HELPERS_DIR}/../.." && pwd)"

# Shared GKE TPU JobSet helpers (gke_* functions, CTX/NS resolution, proxy).
# shellcheck source=/dev/null
source "${_STU_FBCODE_ROOT}/recir/utils/gke_helpers.sh"

_GKE_DEFAULT_YAML="${_STU_HELPERS_DIR}/stu-test-4chip.yaml"
_GKE_FBCODE_ROOT="$_STU_FBCODE_ROOT"
# The test image is the rankfm_mvai_tpu image; reuse its build script.
_GKE_BUILD_SCRIPT="${_STU_FBCODE_ROOT}/recir/examples/rankfm_mvai/build_image_podman.sh"
_GKE_LOG_DIR="/tmp/stulogs"
_GKE_POD_NOUN="test pod"
_GKE_SUMMARY_TITLE="Per-pod test-result summary"

gke_init "mrs-tpu-y317h/us-central1/tpu-prod-training-cluster" "ns-enablement-mrs-foundation"

# unittest prints "OK" / "FAILED (...)" / "ERROR" on its final summary line.
_gke_summarize_line() {
    local f="$1" pod result
    pod=$(basename "$f" .log)
    result=$(grep -E "^(OK|FAILED|ERROR)( |$|\()" "$f" | tail -1)
    if [ -z "$result" ]; then
        local last
        last=$(tail -1 "$f" | tr '\n' ' ')
        echo "$pod: NO_RESULT -- last: $last"
    else
        echo "$pod: $result"
    fi
}

stu_submit()    { gke_submit "$@"; }
stu_status()    { gke_status; }
stu_watch()     { gke_watch; }
stu_logs()      { gke_logs; }
stu_logs_all()  { gke_logs_all; }
stu_dump_logs() { gke_dump_logs; }
stu_traceback() { gke_traceback "$@"; }
stu_kill()      { gke_kill; }
stu_iterate()   { gke_iterate "$@"; }
