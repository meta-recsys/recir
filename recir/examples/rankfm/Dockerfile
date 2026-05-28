# RankFM (xsurface V3) on TPU v7x - E2E training with recir XLA GSPMD
# Build context must contain:
#   <context>/setup.py                   (from recir/github/setup.py)
#   <context>/recir/                     (the recir package)
#   <context>/hammer/                    (rsynced from fbsource/fbcode/hammer)
#   <context>/minimal_viable_ai/         (rsynced from fbsource/fbcode/mvai_cloud/minimal_viable_ai)
#
# build_image_podman.sh stages this context for you.

FROM python:3.12-slim

WORKDIR /app

RUN pip install --upgrade pip

# PyTorch CPU (recir bridges to XLA via torchax)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# JAX with TPU support
RUN pip install "jax[tpu]"

# JAX TPU embedding for SparseCore embedding lookup
RUN pip install jax-tpu-embedding

# torchrec + fbgemm (CPU variant for TPU machines)
RUN pip install torchrec && \
    pip install fbgemm-gpu-cpu --force-reinstall || true

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

# Copy and install recir with XLA + rec extras
COPY recir /app/recir
COPY setup.py /app/setup.py
RUN pip install -e "/app[xla,rec]"

# hammer/v3 and minimal_viable_ai are not pip-installable; mount via PYTHONPATH.
COPY hammer /app/hammer
COPY minimal_viable_ai /app/minimal_viable_ai

ENV PYTHONPATH="/app:/app/recir:${PYTHONPATH}"
ENV CUDA_VISIBLE_DEVICES=-1
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

CMD ["python", "-u", "-m", "recir.examples.rankfm.train", "--preset", "small", "--max_steps", "5"]
