# DLRMv3 training on TPU with recir (GSPMD + XLA via torchax).
# Build from the recir repository root:
#   docker build -f recir/examples/dlrmv3/tpu/Dockerfile -t dlrmv3_tpu .

FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends git build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# Versions are pinned to a mutually compatible stack:
# - torch 2.10 / fbgemm-gpu-cpu 1.5: newest pair in fbgemm's torch compatibility
#   table. fbgemm 1.4's jagged ops lack the meta kernels recir's fake-tensor
#   trace needs; unpinned, pip pulls an fbgemm built against a torch nightly ABI.
# - jax <0.11: 0.11 removed pltpu.DeviceIdType, used by recir's ring-shift
#   embedding kernel (the default in_graph backend).
# PyTorch on CPU: recir lowers the traced step to XLA through torchax.
RUN pip install 'torch==2.10.*' --index-url https://download.pytorch.org/whl/cpu
RUN pip install "jax[tpu]<0.11"
RUN pip install 'torchrec==1.4.*' && \
    pip install 'fbgemm-gpu-cpu==1.5.*' --force-reinstall
# generative-recommenders imports triton at module import time.
RUN pip install triton

# The pip package omits ops/; install the source tree without its deps.
RUN git clone https://github.com/facebookresearch/generative-recommenders.git /app/generative-recommenders && \
    pip install -e /app/generative-recommenders --no-deps && \
    pip install gin-config pandas tensorboard pybind11

COPY . /app/recir
RUN pip install -e "/app/recir[xla,rec]"

ENV PYTHONPATH="/app/recir:${PYTHONPATH}"
ENV CUDA_VISIBLE_DEVICES=-1
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

CMD ["python", "-u", "-m", "recir.examples.dlrmv3.tpu.train", "--dataset", "debug", "--num_train_batches", "10"]
