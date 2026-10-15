# RecIR

RecIR is a compiler framework for training recommendation models from a single
PyTorch definition on GPUs and TPUs. It captures a complete training step —
model forward, backward, and optimizer updates — as a mutable FX graph, applies
recommendation-specific graph transformations, and lowers the result to the
selected execution backend.

> **Project status:** alpha. The DLRMv3 GPU and TPU paths are the first
> end-to-end supported reference. APIs outside that path may still evolve.

## Why RecIR?

Recommendation models combine dense networks, large embedding tables, jagged
inputs, distributed collectives, and stateful optimizers. Porting the same
model between hardware stacks commonly requires backend-specific model code.
RecIR instead separates the model's computation from its execution strategy:

```text
one PyTorch train step
        │
        ▼
full-step FX graph (forward + backward + optimizer)
        │
        ├── GPU: distributed transforms → Inductor / FBGEMM
        └── TPU: GSPMD transforms → torchax → JAX / XLA
```

The initial release focuses on a concrete, reproducible result: the public
DLRMv3 model from
[generative-recommenders](https://github.com/meta-recsys/generative-recommenders)
runs through RecIR on both GPU and TPU using publicly available packages.

## Supported reference paths

| Path | Status | Execution stack | Sparse embeddings |
| --- | --- | --- | --- |
| GPU training | Supported reference | HybridParallel + Inductor | TorchRec / FBGEMM TBE |
| TPU training | Supported reference | GSPMD + torchax + XLA | In-graph Pallas ring shift |
| Training-to-inference transforms | Experimental | FX graph transforms | Backend dependent |

## Installation

RecIR requires Python 3.12 or newer. Install from a source checkout with the
extra matching the environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Host-only smoke test and local TPU-path debugging.
python -m pip install -e ".[dlrm-cpu]"

# CUDA training environment.
python -m pip install -e ".[dlrm-gpu]"

# TPU environment.
python -m pip install -e ".[dlrm-tpu]"
```

The DLRMv3 extras use the stack tested by the reference Dockerfiles: PyTorch
2.10, TorchRec 1.4, FBGEMM 1.5, and JAX 0.8–0.10.

## Five-minute CPU smoke test

The TPU trainer can execute locally through JAX's CPU backend while keeping
embedding lookups in TorchRec. This validates installation, tracing, graph
transforms, optimizer capture, and the training loop without an accelerator:

```bash
JAX_PLATFORMS=cpu python -m recir.examples.dlrmv3.tpu.train \
  --dataset debug \
  --batch_size 4 \
  --num_workers 0 \
  --num_train_batches 6 \
  --embedding_dim 64 \
  --num_layers 1 \
  --uih_len 16 \
  --num_candidates 4 \
  --hash_size 1000 \
  --embedding_backend cpu \
  --log_every 1
```

A successful run compiles the train step and reports lines of the form:

```text
train step compiled in ...s
epoch 0 step ... loss ... (... ms/step)
```

## DLRMv3 on GPU and TPU

The complete reference includes:

- one shared DLRMv3 model and data representation;
- GPU and TPU trainers;
- deterministic eager-versus-compiled parity tools;
- public debug and recommendation datasets;
- evaluation metrics, TensorBoard, and distributed checkpoints; and
- public container and GKE JobSet templates.

See the [DLRMv3 guide](recir/examples/dlrmv3/README.md) for copy-pasteable GPU,
TPU, parity, dataset, and troubleshooting commands.

## Core design

RecIR has four main extension mechanisms:

1. **Parallel modes** describe data, parameter, and embedding sharding.
2. **Region transforms** identify and rewrite coherent FX subgraphs, including
   padding, recomputation, sparse collectives, and training-to-inference swaps.
3. **Overrides** replace components that cannot be traced directly, then
   restore their runtime implementation at graph boundaries.
4. **Backends** lower the transformed graph to GPU or TPU execution.

Read [the design document](docs/design.md) for the capture pipeline, graph
invariants, state model, distributed execution, embedding architecture, and
current limitations.

Additional implementation notes cover the
[FX device pipeline](docs/pipeline_design.md) and
[cross-iteration steady-state scheduling](docs/cross_iteration_steady_state_scheduling.md).

## Repository layout

```text
recir/
  api.py                 graph capture and compile orchestration
  distributed/           parallel modes and sharding metadata
  region_transforms/     composable graph transformations
  embedding/             CPU, in-graph, and SparseCore backends
  lowering/              backend-specific graph lowering
  torchax/               PyTorch ATen to JAX bridge integration
  training/              train-step compilation and optimizer state
  examples/dlrmv3/       supported GPU and TPU reference
```

## Contributors

The overall project is made possible thanks to the joint work from many technical contributors (listed in alphabetical order):

Shakhzod Ali-Zade, Qingyun Bian, Zheyuan Chen, Cao Gao,
Daisy Shi He, Guxin Jin, Rafay Khurram, Han Li, Shen Li, Chloe Liu, Dunfan Lu,
Melody Lui, Pierson Lund, Linjian Ma, Yuheng Wang, Angela Yi, Chunxing Yin,
Qunshu Zhang, Shengpei Zhang, and Zhao Zhu.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and the
[Code of Conduct](CODE_OF_CONDUCT.md).

## License

RecIR is BSD-3-Clause licensed. See [LICENSE](LICENSE).
