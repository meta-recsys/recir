# Ringshift sharded embedding lookup -- real-TPU correctness/perf bench
# Build (from OSS recir checkout root):
#   cd ~/recir && docker build -f recir/examples/embedding/Dockerfile -t ringshift_tpu .

FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends git build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# JAX with TPU support (>= 0.10 for tpu_sc + tpu split-DMA primitives).
RUN pip install "jax[tpu]"

# SparseCore embedding lookup (jte) for the ringshift_bench --backend sc/all arm
# (the SC-vs-TC comparison).
RUN pip install jax-tpu-embedding

# The package initializer imports torch even though these TPU runners use JAX.
# Install the CPU wheel explicitly so the subsequent editable install does not
# pull the much larger CUDA dependency set.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

# Copy and install recir. Runtime dependencies are installed explicitly above.
COPY . /app/recir
RUN pip install --no-deps -e "/app/recir"

ENV PYTHONPATH="/app/recir:${PYTHONPATH}"
ENV PYTHONUNBUFFERED=1
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Default command runs the W=4 single-host correctness bench (smallest scale,
# safe to run as a smoke test). The JobSet manifests in ./manifests/ override
# this to invoke specific W=32/64/128 methods.
CMD ["python", "-u", "-m", "unittest", "-v", \
     "recir.examples.embedding.ringshift_fwd.RingshiftFwdCorrectnessTest.test_w4_correctness"]
