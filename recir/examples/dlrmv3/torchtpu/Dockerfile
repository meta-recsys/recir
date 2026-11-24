# DLRMv3 training on TorchTPU (torch_tpu) with recir.
# Build from the recir repository root:
#   docker build -f recir/examples/dlrmv3/torchtpu/Dockerfile -t dlrmv3_torchtpu .

FROM docker.io/library/python:3.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends git build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

COPY recir/examples/dlrmv3/torchtpu/constraints.txt /app/constraints.txt
ENV PIP_CONSTRAINT=/app/constraints.txt

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install "jax[tpu]"
RUN pip install torchrec && \
    pip install fbgemm-gpu-cpu --force-reinstall
# generative-recommenders imports triton at module import time.
RUN pip install triton

# The pip package omits ops/; install the source tree without its deps.
# pyarrow: preprocess_yambda.py reads the Yambda parquet files.
RUN git clone https://github.com/facebookresearch/generative-recommenders.git /app/generative-recommenders && \
    pip install -e /app/generative-recommenders --no-deps && \
    pip install gin-config pandas pyarrow tensorboard pybind11

COPY . /app/recir
RUN pip install -e "/app/recir[xla,rec]"
RUN pip install helion torch-tpu

# torch_tpu and torchax both claim PyTorch's PrivateUse1 device; the trainer
# loads torch_tpu itself once its launch environment is set.
ENV TORCH_DEVICE_BACKEND_AUTOLOAD=0
ENV PYTHONPATH="/app/recir:${PYTHONPATH}"
ENV CUDA_VISIBLE_DEVICES=-1
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

CMD ["python", "-u", "-m", "recir.examples.dlrmv3.torchtpu.train", \
     "--dataset", "debug", "--num_workers", "0", "--num_train_batches", "20", \
     "--uih_len", "256", "--embedding_dim", "64", "--num_layers", "2", \
     "--hash_size", "1000", "--log_every", "1"]
