#!/usr/bin/env bash
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# Build the current TPU image, then run the pooled EBC performance suites.
# Each suite also times torchrec.EmbeddingBagCollection on the host and reports
# it as the baseline for the fused TPU kernel.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FBCODE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RUN_TPU="${SCRIPT_DIR}/run_microbench.sh"
BUILD_IMAGE="${SCRIPT_DIR}/build_image_podman.sh"

TESTS=(
    ebc-w4
    ebc-w16
    ebc-w32
    ebc-w64
    ebc-w128
)

usage() {
    echo "usage: $0 [--build-image|--skip-image-build] [test|all]" >&2
    echo "       image build/push is enabled by default" >&2
    echo "tests: ${TESTS[*]}" >&2
}

is_known_test() {
    local candidate="$1"
    local known
    for known in "${TESTS[@]}"; do
        [ "${candidate}" = "${known}" ] && return 0
    done
    return 1
}

run_one() {
    local test_name="$1"
    echo "=== performance suite: ${test_name} ==="
    case "${test_name}" in
        ebc-w4|ebc-w16|ebc-w32|ebc-w64|ebc-w128)
            local world_size="${test_name#ebc-w}"
            local tpu_log="${SCRIPT_DIR}/bench_logs/w${world_size}-ebc-perf-latest.log"
            "${RUN_TPU}" "manifests/w${world_size}-ebc-perf.yaml"
            echo "=== torchrec EBC baseline for the W=${world_size} TPU result ==="
            (cd "${FBCODE_ROOT}" && buck2 run \
                fbcode//recir/examples/embedding:ringshift_ebc_torchrec_bench -- \
                --tpu-log "${tpu_log}" 2>&1 | tee -a "${tpu_log}")
            ;;
        *)
            echo "ERROR: unknown performance test: ${test_name}" >&2
            usage
            return 2
            ;;
    esac
}

build_image=true
selected_test="all"
test_was_set=false

while [ "$#" -gt 0 ]; do
    case "$1" in
        --build-image)
            build_image=true
            ;;
        --skip-image-build)
            build_image=false
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -* )
            echo "ERROR: unknown option: $1" >&2
            usage
            exit 2
            ;;
        *)
            if [ "${test_was_set}" = true ]; then
                echo "ERROR: only one test may be specified" >&2
                usage
                exit 2
            fi
            selected_test="$1"
            test_was_set=true
            ;;
    esac
    shift
done

if [ "${selected_test}" != "all" ] && ! is_known_test "${selected_test}"; then
    echo "ERROR: unknown performance test: ${selected_test}" >&2
    usage
    exit 2
fi

if [ "${build_image}" = true ]; then
    echo "=== build and push current TPU image ==="
    "${BUILD_IMAGE}"
else
    echo "=== reusing published TPU image (--skip-image-build) ==="
fi

if [ "${selected_test}" != "all" ]; then
    run_one "${selected_test}"
else
    for test_name in "${TESTS[@]}"; do
        run_one "${test_name}"
    done
fi
