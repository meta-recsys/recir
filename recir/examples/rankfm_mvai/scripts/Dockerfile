# RankFM-MVAI (xsurface V3) on TPU v7x - the hammer/v3 udd_fm BlueReelsVDDTrainV3
# DRIVEN BY the mvai_cloud training framework (RecTrainer + RecModelFamily +
# RecTrainPipeline).
#
# Build context must contain:
#   <context>/setup.py                   (from recir/github/setup.py)
#   <context>/recir/                     (the recir package)
#   <context>/hammer/                    (rsynced from fbsource/fbcode/hammer)
#   <context>/minimal_viable_ai/         (rsynced from fbsource/fbcode/mvai_cloud/minimal_viable_ai)
#   <context>/mvai_cloud/                 (TPU job runtime helpers)
#   <context>/mrs/ + <context>/hpc/      (trimmed dense_sparse DCN subset)
#
# build_image_podman.sh stages this context for you.

# Fully qualified on purpose. The short name `python:3.12-slim` fails to
# resolve under this podman with `image not known` -- an unhelpful message that
# reads like a network problem but is short-name resolution, and it bites
# whenever the local store has been evicted. `--pull=missing` alone does not
# fix it, because the name never resolves to something to pull.
FROM docker.io/library/python:3.12-slim

WORKDIR /app

RUN pip install --upgrade pip

# PyTorch CPU (recir bridges to XLA via torchax).
# Pinned: torch 2.10 / fbgemm-gpu-cpu 1.5 — latest tuple in
# fbgemm_gpu/__init__.py's _fbgemm_torch_compat_table. torchrec stops at 1.4
# on PyPI (no 1.5+ release); it's tolerant of newer torch/fbgemm. fbgemm 1.4
# is too old: its jagged ops (dense_to_jagged etc.) lack the meta-kernel
# registrations that recir.trace's FakeTensor pass needs. Leaving versions
# unpinned lets pip pull fbgemm 1.7.0, which is built against a torch nightly
# ABI (torch::headeronly::Tag is only in torch >=2.12.dev) and breaks at import.
RUN pip install 'torch==2.10.*' --index-url https://download.pytorch.org/whl/cpu

# BarnFS encrypts DCP checkpoint payloads through its sitecustomize hook. TPU
# jobs enable it with TORCH_ENABLE_BARNFS=ON and set MODEL_TYPE_NAME. GKE jobs
# also set BARNFS_SHEPHERD_USE_INTERNET=1 to use their metadata identity.
#
# The directory is staged by build_image_podman.sh and is EMPTY when that ran
# with SKIP_BARNFS=1 -- COPY cannot be made conditional, so the conditional
# lives in the install. An image built that way has no barnfs at all, and
# TORCH_ENABLE_BARNFS=ON against it will fail to import rather than silently
# writing plaintext.
COPY barnfs_wheel/ /tmp/barnfs_wheel/
RUN if ls /tmp/barnfs_wheel/*.whl >/dev/null 2>&1; then \
        pip install /tmp/barnfs_wheel/*.whl; \
    else \
        echo "no BarnFS wheel staged; image is built without it"; \
    fi && \
    rm -rf /tmp/barnfs_wheel

# JAX with TPU support.
# Pinned <0.11: JAX 0.11.0 removed pltpu.DeviceIdType, which recir's ringshift
# Pallas kernel (recir/ops/pallas/embeddings/ringshift.py) still uses. Leaving
# jax unpinned floats past that removal and breaks TPU graph tracing at runtime.
# 0.10.x still provides the symbol (deprecated, warns).
RUN pip install "jax[tpu]<0.11"

# JAX TPU embedding for SparseCore embedding lookup. Reassert the jax<0.11
# constraint so this install can't transitively upgrade jax past the removal.
RUN pip install jax-tpu-embedding "jax[tpu]<0.11"

# torchrec + fbgemm (CPU variant for TPU machines).
# Drop the `|| true` swallowing — we want this layer to fail loudly if the
# pinned wheels become unavailable, not produce a silently broken image.
RUN pip install 'torchrec==1.4.*' && \
    pip install 'fbgemm-gpu-cpu==1.5.*' --force-reinstall

# triton (required by generative-recommenders common.py)
RUN pip install triton

# generative_recommenders: staged from fbsource (no git clone, no pip install).
# The package directory is mounted via PYTHONPATH below alongside hammer/ and
# minimal_viable_ai/.
COPY generative_recommenders /app/generative_recommenders

# Side deps that generative_recommenders code expects to import
RUN pip install gin-config matplotlib pybind11

# Extra deps for hammer/v3 and minimal_viable_ai
RUN pip install pyre-extensions iopath

# zstandard: decompress .pt.zst dataporter batches in BatchReplayPipe
RUN pip install zstandard

# google-cloud-storage: BatchReplayPipe reads gs:// dumps, and
# parity_forward_loss.py uploads parity_result.json to a tenant GCS bucket
# (`from google.cloud import storage`).
RUN pip install google-cloud-storage

# pyzstd: required by the mvai_cloud RecTrainer import chain
# (minimal_viable_ai.core.dataloader.preproc_output_data_pipe imports it at
# module load). rankfm doesn't need this; rankfm_mvai does because it drives
# the mvai_cloud trainer. Pin matches mvai_cloud/requirements.txt.
RUN pip install pyzstd==0.19.1

# tensorboard: torch.utils.tensorboard.SummaryWriter backend used by mvai's
# TensorboardLogger (enabled via MetricsConfig.tensorboard_logger=True). Not in
# the rankfm image; rankfm_mvai needs it because RecTrainer can log scalars to TB.
RUN pip install tensorboard

# Copy and install recir with XLA + rec extras
COPY recir /app/recir
COPY setup.py /app/setup.py
RUN pip install -e "/app[xla,rec]"

# helion (Pallas backend): recir's HSTU op (recir/ops/helion/*) imports helion
# and exports it via Kernel.jax_fn. Install -e from the staged recir-pinned tree
# (NOT PyPI helion) so the kernel/jax_fn API matches recir. The staged context
# has no .git, so pin the setuptools-scm version to avoid a git lookup.
COPY helion /app/helion
RUN SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 pip install -e /app/helion

# hammer/v3 and minimal_viable_ai are not pip-installable; mount via PYTHONPATH.
COPY hammer /app/hammer
COPY minimal_viable_ai /app/minimal_viable_ai
COPY mvai_cloud /app/mvai_cloud

# mrs + hpc: trimmed, torchrec-only subset staged by build_image_podman.sh --
# just the dense_sparse DCN files udd_fm_v3's inter_arch needs (use_inter_arch=
# True). NOT the full mrs/hpc trees (those pull torchrec.fb / hpc.new chains).
COPY mrs /app/mrs
COPY hpc /app/hpc

# Native BarnFS wheel dependencies absent from python:3.12-slim.
# apt ignores the uppercase HTTP_PROXY that pip honors, so on a devserver it
# would try to resolve deb.debian.org directly and fail. Write the proxy into
# apt.conf from the build arg instead of relying on env-var spelling.
ARG HTTP_PROXY=
RUN if [ -n "$HTTP_PROXY" ]; then \
        printf 'Acquire::http::Proxy "%s";\nAcquire::https::Proxy "%s";\n' \
            "$HTTP_PROXY" "$HTTP_PROXY" > /etc/apt/apt.conf.d/00proxy; \
    fi && \
    apt-get update && \
# would try to resolve deb.debian.org directly and fail. Write the proxy into
    rm -rf /var/lib/apt/lists/* /etc/apt/apt.conf.d/00proxy
ARG HTTP_PROXY=
RUN if [ -n "$HTTP_PROXY" ]; then \
        printf 'Acquire::http::Proxy "%s";\nAcquire::https::Proxy "%s";\n' \
            "$HTTP_PROXY" "$HTTP_PROXY" > /etc/apt/apt.conf.d/00proxy; \
    fi && \
    apt-get update && \
    apt-get install -y --no-install-recommends libatomic1 libtirpc3 && \
    rm -rf /var/lib/apt/lists/* /etc/apt/apt.conf.d/00proxy

ENV PYTHONPATH="/app:/app/recir:${PYTHONPATH}"
ENV CUDA_VISIBLE_DEVICES=-1
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# Helion uses the Pallas (JAX/TPU Mosaic) backend for the HSTU kernel. On a real
# TPU pod do NOT set HELION_PALLAS_INTERPRET or JAX_PLATFORMS=cpu (those force
# CPU interpret); the pod uses real Mosaic.
ENV HELION_BACKEND=pallas

# Devserver smoke run (JAX falls back to CPU): tiny model, a few steps. Uses
# explicit small flags rather than --preset (there is no --preset in parse_cli).
CMD ["python", "-u", "-m", "recir.examples.rankfm_mvai.train", \
     "--max_steps", "5", "--local_batch_size", "4", "--num_candidates", "4", \
     "--hash_size_cap", "1000", "--l1_num_layers", "1", "--l2_num_layers", "1", \
     "--embedding_backend", "cpu"]
