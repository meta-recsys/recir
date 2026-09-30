#!/usr/bin/env bash
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# Build the current TPU image, then run every ring-shift embedding correctness
# suite or one named suite. Pass --skip-image-build only when intentionally
# reusing the image already published in Artifact Registry.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_TPU="${SCRIPT_DIR}/run_microbench.sh"
BUILD_IMAGE="${SCRIPT_DIR}/build_image_podman.sh"

TESTS=(
    fwd-w4
    fwd-w4-chunked
    fwd-w4-dma-gather
    fwd-w32
    fwd-w64
    fwd-w128
    bwd-w4
    bwd-w4-chunked
    bwd-w4-dma-gather
    bwd-w32
    ebc-w4
    ebc-torchrec-w4
)

usage() {
    echo "usage: $0 [--build-image|--skip-image-build] [--max-polls N] [test|all]" >&2
    echo "       image build/push is enabled by default" >&2
    echo "       polling limit defaults to 60; override with --max-polls or MAX_POLLS" >&2
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
    echo "=== correctness suite: ${test_name} ==="
    case "${test_name}" in
        fwd-w4|fwd-w32|fwd-w64|fwd-w128)
            local world_size="${test_name#fwd-w}"
            "${RUN_TPU}" "manifests/w${world_size}-correctness.yaml"
            ;;
        fwd-w4-chunked)
            "${RUN_TPU}" manifests/w4-correctness-chunked.yaml
            ;;
        fwd-w4-dma-gather)
            "${RUN_TPU}" manifests/w4-dma-gather-fwd.yaml
            ;;
        bwd-w4)
            "${RUN_TPU}" manifests/w4-bwd-suite.yaml
            ;;
        bwd-w4-chunked)
            "${RUN_TPU}" manifests/w4-bwd-chunked.yaml
            ;;
        bwd-w4-dma-gather)
            "${RUN_TPU}" manifests/w4-dma-gather-bwd.yaml
            ;;
        bwd-w32)
            "${RUN_TPU}" manifests/w32-bwd-correctness.yaml
            ;;
        ebc-w4)
            "${RUN_TPU}" manifests/w4-ebc-correctness.yaml
            ;;
        ebc-torchrec-w4)
            "${SCRIPT_DIR}/test_embedding_bag_tpu_vs_torchrec_with_tpu_input.sh"
            ;;
        *)
            echo "ERROR: unknown correctness test: ${test_name}" >&2
            usage
            return 2
            ;;
    esac
}

PASSED_TESTS=()
FAILED_TESTS=()
FAILED_EXIT_CODES=()

run_and_record() {
    local test_name="$1"
    local exit_code

    if run_one "${test_name}"; then
        PASSED_TESTS+=("${test_name}")
    else
        exit_code=$?
        FAILED_TESTS+=("${test_name}")
        FAILED_EXIT_CODES+=("${exit_code}")
        echo "=== correctness suite FAILED: ${test_name} (exit ${exit_code}); continuing ===" >&2
    fi
}

print_summary() {
    local i

    echo
    echo "=== correctness test summary ==="
    echo "PASS (${#PASSED_TESTS[@]}):"
    if [ "${#PASSED_TESTS[@]}" -eq 0 ]; then
        echo "  (none)"
    else
        for i in "${!PASSED_TESTS[@]}"; do
            echo "  - ${PASSED_TESTS[$i]}"
        done
    fi

    echo "FAIL (${#FAILED_TESTS[@]}):"
    if [ "${#FAILED_TESTS[@]}" -eq 0 ]; then
        echo "  (none)"
    else
        for i in "${!FAILED_TESTS[@]}"; do
            echo "  - ${FAILED_TESTS[$i]} (exit ${FAILED_EXIT_CODES[$i]})"
        done
    fi
}

build_image=true
selected_test="all"
test_was_set=false
MAX_POLLS="${MAX_POLLS:-60}"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --build-image)
            build_image=true
            ;;
        --skip-image-build)
            build_image=false
            ;;
        --max-polls)
            if [ "$#" -lt 2 ]; then
                echo "ERROR: --max-polls requires a value" >&2
                usage
                exit 2
            fi
            MAX_POLLS="$2"
            shift
            ;;
        --max-polls=*)
            MAX_POLLS="${1#*=}"
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

if ! [[ "${MAX_POLLS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: max polls must be a positive integer, got: ${MAX_POLLS}" >&2
    exit 2
fi
export MAX_POLLS

if [ "${selected_test}" != "all" ] && ! is_known_test "${selected_test}"; then
    echo "ERROR: unknown correctness test: ${selected_test}" >&2
    usage
    exit 2
fi

if [ "${build_image}" = true ]; then
    echo "=== build and push current TPU image ==="
    bash "${BUILD_IMAGE}"
else
    echo "=== reusing published TPU image (--skip-image-build) ==="
fi

if [ "${selected_test}" != "all" ]; then
    run_and_record "${selected_test}"
else
    for test_name in "${TESTS[@]}"; do
        run_and_record "${test_name}"
    done
fi

print_summary

if [ "${#FAILED_TESTS[@]}" -gt 0 ]; then
    exit 1
fi
