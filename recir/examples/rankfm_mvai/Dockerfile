# RankFM-MVAI (xsurface V3) on TPU v7x - the hammer/v3 udd_fm BlueReelsVDDTrainV3
# DRIVEN BY the mvai_cloud training framework (RecTrainer + RecModelFamily +
# RecTrainPipeline).
#
# Build context must contain:
#   <context>/setup.py                   (from recir/github/setup.py)
#   <context>/recir/                     (the recir package)
#   <context>/hammer/                    (rsynced from fbsource/fbcode/hammer)
#   <context>/minimal_viable_ai/         (rsynced from fbsource/fbcode/mvai_cloud/minimal_viable_ai)
#   <context>/mrs/ + <context>/hpc/      (trimmed dense_sparse DCN subset)
#
# build_image_podman.sh stages this context for you.

FROM python:3.12-slim

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

# JAX with TPU support
RUN pip install "jax[tpu]"

# JAX TPU embedding for SparseCore embedding lookup
RUN pip install jax-tpu-embedding

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

# hammer/v3 and minimal_viable_ai are not pip-installable; mount via PYTHONPATH.
COPY hammer /app/hammer
COPY minimal_viable_ai /app/minimal_viable_ai

# mrs + hpc: trimmed, torchrec-only subset staged by build_image_podman.sh --
# just the dense_sparse DCN files udd_fm_v3's inter_arch needs (use_inter_arch=
# True). NOT the full mrs/hpc trees (those pull torchrec.fb / hpc.new chains).
COPY mrs /app/mrs
COPY hpc /app/hpc

ENV PYTHONPATH="/app:/app/recir:${PYTHONPATH}"
ENV CUDA_VISIBLE_DEVICES=-1
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Devserver smoke run (JAX falls back to CPU): tiny model, a few steps. Uses
# explicit small flags rather than --preset (there is no --preset in parse_cli).
CMD ["python", "-u", "-m", "recir.examples.rankfm_mvai.train", \
     "--max_steps", "5", "--local_batch_size", "4", "--num_candidates", "4", \
     "--hash_size_cap", "1000", "--l1_num_layers", "1", "--l2_num_layers", "1", \
     "--embedding_backend", "cpu"]
