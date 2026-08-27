#!/usr/bin/env bash
# Build the rankfm_mvai TPU image and push it to the cluster registry.
#
# The pod-side half of the edit/run loop: config reaches the pod through the
# artifact and the ConfigMap, but anything under recir/ or hammer/ that the
# trainer imports only gets there through the image. Forgetting this step is
# not an error -- the pod pulls the PREVIOUS image and runs it, and the log
# looks like a clean run of code that was never deployed.
#
# Must be run by a human from a shell with cluster credentials: the build pulls
# python:3.12-slim from Docker Hub and the push needs a gcloud token, neither of
# which resolves from an agent session.
#
#   bash recir/examples/rankfm_mvai/scripts/rebuild_push_image.sh
#   IMAGE_TAG=someone-else bash .../rebuild_push_image.sh
set -euo pipefail

: "${IMAGE_TAG:=${USER}-compiled}"
: "${REGISTRY:=us-central1-docker.pkg.dev/mrs-tpu-y317h/recml}"

FBCODE="$(hg root 2>/dev/null || sl root 2>/dev/null)/fbcode"
cd "$FBCODE"

LOCAL="localhost/rankfm_mvai_tpu:${IMAGE_TAG}"
REMOTE="${REGISTRY}/rankfm_mvai_tpu:${IMAGE_TAG}"

echo "=== build  ${LOCAL}"
# PODMAN_PULL_POLICY=never reuses the base image already in local storage;
# without it the build tries to reach Docker Hub and fails on a devserver.
IMAGE_TAG="${IMAGE_TAG}" PODMAN_PULL_POLICY=never \
    bash recir/examples/rankfm_mvai/scripts/build_image_podman.sh

# The build runs in a SUBSHELL, so the proxy it exports does not come back
# here -- and the registry does not resolve without one. The failure looks
# unrelated to the proxy: the build reports COMPLETE and then login dies with
# `lookup us-central1-docker.pkg.dev: no such host`, which reads like the
# registry is down. Same defaults as build_image_podman.sh; override with
# PROXY_URL=... .
export PROXY_URL="${PROXY_URL:-http://fwdproxy:8080}"
export HTTPS_PROXY="${PROXY_URL}"
export HTTP_PROXY="${PROXY_URL}"
export NO_PROXY=localhost,127.0.0.1,.fbcdn.net,.facebook.com,.tfbnw.net
export https_proxy="${HTTPS_PROXY}"
export http_proxy="${HTTP_PROXY}"
export no_proxy="${NO_PROXY}"

echo "=== login  ${REGISTRY}"
gcloud auth print-access-token | podman login \
    -u oauth2accesstoken --password-stdin "https://${REGISTRY%%/*}"

echo "=== push   ${REMOTE}"
podman tag "${LOCAL}" "${REMOTE}"
podman push "${REMOTE}"

echo
echo "pushed ${REMOTE}"
echo "run it with:  --image-tag ${IMAGE_TAG}"
