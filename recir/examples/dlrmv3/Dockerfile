# DLRMv3 on TPU v7x — E2E training with torchrec DMP + recir XLA GSPMD
# Build: cd ~/recir && docker build -f recir/examples/dlrmv3/Dockerfile -t dlrmv3_tpu .

FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends git build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# PyTorch CPU (recir bridges to XLA via torchax)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# JAX with TPU support
RUN pip install "jax[tpu]"

# torchrec + fbgemm (CPU variant for TPU machines)
RUN pip install torchrec && \
    pip install fbgemm-gpu-cpu --force-reinstall || true

# triton (required by generative-recommenders common.py)
RUN pip install triton

# generative-recommenders: clone and install with --no-deps
# (pip package is incomplete; editable clone includes ops/ directory)
RUN git clone https://github.com/facebookresearch/generative-recommenders.git /app/generative-recommenders && \
    pip install -e /app/generative-recommenders --no-deps && \
    pip install gin-config matplotlib pybind11

# Copy and install recir with XLA + rec extras
COPY . /app/recir
RUN pip install -e "/app/recir[xla,rec]"

ENV PYTHONPATH="/app/recir:${PYTHONPATH}"
ENV CUDA_VISIBLE_DEVICES=-1
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

CMD ["python", "-u", "-m", "recir.examples.dlrmv3.train", "--preset", "small", "--max_steps", "5"]
