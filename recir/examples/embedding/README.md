# Ring-shift embedding examples

Correctness and performance programs for the Pallas ring-shift embedding
lookup in `recir.ops.pallas.embeddings.ringshift`.

## Build the TPU image

From an open-source recir checkout:

```bash
IMAGE=<registry>/ringshift_tpu:latest \
  ./recir/examples/embedding/build_image.sh
```

The build script creates the image from
`recir/examples/embedding/Dockerfile`, verifies the public imports, and
pushes it to `IMAGE`.

## Run a JobSet

The manifests are provider-neutral templates. They use `${IMAGE}` and the
`default` namespace rather than a project-specific registry or namespace.
Render a manifest before applying it:

```bash
IMAGE=<registry>/ringshift_tpu:latest \
  envsubst < recir/examples/embedding/manifests/w4-correctness.yaml |
  kubectl create -f -
```

Choose a manifest whose TPU topology matches the target cluster. The
`ringshift_fwd.py`, `ringshift_bwd.py`, and `ringshift_bench.py` modules
contain the corresponding correctness and performance entry points.
Multi-host templates also use `${QUEUE_NAME}` for the cluster's Kueue local
queue; export it alongside `IMAGE` before running `envsubst`.

The cross-runtime EBC parity flow uses
`ringshift_ebc_parity_dump.py` to emit a self-describing TPU artifact and
`ringshift_ebc_torchrec_bench.py` to compare it with TorchRec on CPU.
