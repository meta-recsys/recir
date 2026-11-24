# DLRMv3 on TorchTPU: playbook

How to train the generative-recommenders DLRMv3 model (`DlrmHSTU`) with recir on
TorchTPU (`torch_tpu`, PyTorch's native TPU device) on Google Cloud, with
synthetic data or with the public Yambda dataset. Everything here uses public
packages and a GKE TPU cluster; nothing needs Meta-internal infrastructure.

Contents: [how it runs](#how-it-runs) · [image](#1-build-the-image) ·
[synthetic data](#2-synthetic-data) · [Yambda](#3-yambda) · [knobs](#knobs) ·
[reference numbers](#reference-numbers) · [limitations](#limitations)

## How it runs

- One process per TPU device, as on GPU. `python -m recir.examples.dlrmv3.torchtpu.train`
  relaunches itself under `torchrun` with `--nproc_per_node` processes (default 8,
  every TorchTPU device of a v7x host: 4 chips x 2 TensorCores).
- Each process runs the dense model on its device. recir traces the whole step
  (forward, backward, optimizers) once and compiles it with torch_tpu; dense
  gradients are averaged with collectives inside the compiled program.
- The embedding tables sit behind a graph break, either in torchrec on the host
  CPUs (`--embedding_backend cpu`, the default) or on the TPUs' SparseCores
  (`--embedding_backend sparsecore`).
- Every batch has static shapes: see "Token-budget batch layout" in
  [../README.md](../README.md#token-budget-batch-layout).

## 1. Build the image

From the root of the recir repository:

```bash
export IMAGE=<region>-docker.pkg.dev/<project>/<repository>/dlrmv3_torchtpu:latest
docker build -f recir/examples/dlrmv3/torchtpu/Dockerfile -t "$IMAGE" .
docker push "$IMAGE"
```

The image pins a stack known to work together
([constraints.txt](constraints.txt)): torch 2.12.1 (CPU wheel), torch-tpu 0.2.0,
torchrec / fbgemm-gpu-cpu 1.7, JAX < 0.11 and libtpu < 0.0.53.1. Change one and
you will likely need to change the others; the comments in the file say why
each pin is there.

You need a GKE cluster with a TPU v7x node pool (`tpu7x`, topology `2x2x1` for
one host) and the [JobSet](https://jobset.sigs.k8s.io/) controller installed.
[manifests/dlrmv3-torchtpu-1host.yaml](manifests/dlrmv3-torchtpu-1host.yaml)
runs one host; the trainer command comes from `$COMMAND`.

On a single TPU v7x VM without Kubernetes, install the same packages in the
order the Dockerfile does and run the trainer commands below directly.

## 2. Synthetic data

`--dataset debug` generates random samples in the process (no download). Use it
to check the installation and to measure step time, not model quality: its ids
cover only 1-999, so every lookup hits the first rows of each table.

Smoke test, small model (64-dim tables of 1000 rows, 2 layers, history 256):

```bash
export COMMAND='python3 -u -m recir.examples.dlrmv3.torchtpu.train
  --dataset debug --num_workers 0 --batch_size 16 --num_train_batches 20 --log_every 1
  --uih_len 256 --embedding_dim 64 --num_layers 2 --hash_size 1000'
envsubst < recir/examples/dlrmv3/torchtpu/manifests/dlrmv3-torchtpu-1host.yaml | kubectl create -f -
```

Full-size model (three 10M x 512 tables, 15.4B embedding parameters, 5 layers,
history 1024); the tables need SparseCore:

```bash
export COMMAND='python3 -u -m recir.examples.dlrmv3.torchtpu.train
  --dataset debug --num_workers 0 --batch_size 16 --num_train_batches 100 --log_every 10
  --uih_len 1024 --embedding_backend sparsecore'
```

Follow the run with `kubectl logs -f <pod>`. A healthy run logs
`train step compiled in ...`, then `epoch 0 step N loss ... (... ms/step)`; the
first logged window includes compilation.

## 3. Yambda

[Yambda](https://huggingface.co/datasets/yandex/yambda) is Yandex Music's public
listening log (Apache-2.0; the dataset card restricts it to scientific and
research use). The 50m scale has 9,238 users, 877K tracks and 46.5M listens.

### Prepare the data

[`common/preprocess_yambda.py`](../common/preprocess_yambda.py) downloads
`flat/<scale>/multi_event.parquet` from Hugging Face and writes
`<out_root>/processed/yambda-<scale>/` (`.npy` arrays and a `meta.json`) plus
the raw file and a provenance README under `<out_root>/raw/`. It needs internet
access, numpy and pyarrow, and roughly 10 GB of RAM for 50m (the 500m scale
needs about ten times that; only 50m has been run). It skips a scale whose
`meta.json` already exists.

```bash
python3 -m recir.examples.dlrmv3.common.preprocess_yambda --scale 50m --out_root /data/yambda
```

What it builds: one token per listen, in time order per user, with an action
bit set for `listen_plus` (played at least 50%), `like` and `dislike` (folded
into the latest listen of the same track up to an hour before them) and
`organic`. Training samples are 10 consecutive listens (the candidates) after
up to `--uih_len` earlier listens (the history). The split follows the Yambda
benchmark: training candidates end 30 minutes before the last day; the last
day's listens are evaluation samples. 50m gives 4.6M training samples. The model
predicts `listen_plus`, `like` and `dislike`; tables are sized from `meta.json`
(877,169 track rows and 9,239 user rows for 50m).

There are two ways to get the data to the pods:

**A. Convert at the start of the job** (simplest; needs internet egress from
the pods, e.g. Cloud NAT; a few minutes for 50m):

```bash
export COMMAND='python3 -m recir.examples.dlrmv3.common.preprocess_yambda --scale 50m --out_root /tmp/yambda &&
  python3 -u -m recir.examples.dlrmv3.torchtpu.train
  --dataset yambda-50m --data_root /tmp/yambda --num_workers 0 --batch_size 16
  --num_train_batches 1000 --log_every 10 --uih_len 1024 --embedding_backend sparsecore'
envsubst < recir/examples/dlrmv3/torchtpu/manifests/dlrmv3-torchtpu-1host.yaml | kubectl create -f -
```

**B. Convert once into a bucket and mount it.** Copy `/data/yambda` to
`gs://<bucket>/yambda`, then mount the bucket in the pod with the
[Cloud Storage FUSE CSI driver](https://cloud.google.com/kubernetes-engine/docs/how-to/persistent-volumes/cloud-storage-fuse-csi-driver)
(pod annotation `gke-gcsfuse/volumes: "true"`, a `csi` volume with driver
`gcsfuse.csi.storage.gke.io` and `bucketName: <bucket>`, mounted read-only at,
say, `/data`, and a Kubernetes service account that can read the bucket). Then
pass `--dataset yambda-50m --data_root /data/yambda`. Each process reads the
arrays into memory once at startup.

## Knobs

### Environment

| Variable | Values | Effect |
| --- | --- | --- |
| `RECIR_TORCHTPU_COMPILER` | unset (default) | Each traced subgraph is compiled by torch_tpu's `TpuBackend`, with recir marking the collectives SPMD-safe so they stay inside one program. |
| | `static` | Each subgraph is functionalized and compiled as one program by torch_tpu's `StaticCompiler`, which skips TpuBackend's graph splitting; compiles a little faster (see the numbers below). |

Under both, every rank checks once per compile (a hash all-gather on the host)
that all ranks traced the same graph, and raises if not: a rank compiling a
different program would otherwise hang the job inside a collective.

Set it in the pod's `env:` or prefix the command (`RECIR_TORCHTPU_COMPILER=static python3 -u -m ...`).

### Trainer flags

| Flag | Default | Notes |
| --- | --- | --- |
| `--embedding_backend` | `cpu` | `cpu`: torchrec tables on the host CPUs, row-wise sharded over the processes (gloo all-to-all), fused row-wise Adagrad. Fine for small tables; host-bound at full size. `sparsecore`: tables row-sharded across the TPUs' SparseCores and trained there; needed for full-size tables. |
| `--nproc_per_node` | 8 | Processes (TorchTPU devices) per host. |
| `--dataset` / `--data_root` | `debug` / none | `debug`, `yambda-<scale>`, or a generative-recommenders dataset (`kuairand-1k`, `movielens-*`; see [../README.md](../README.md#data); not yet run on TorchTPU). |
| `--batch_size` | 16 | Real rows per device per step. |
| `--uih_len` | from the model config | History length per row (most recent events). Always set it: the model config's default (about 16K) runs out of HBM. |
| `--num_candidates` / `--uih_budget` | model config / `uih_len` | See "Token-budget batch layout" in [../README.md](../README.md#token-budget-batch-layout). |
| `--embedding_dim` / `--num_layers` / `--hash_size` | model config | Shrink the model for smoke tests (0 keeps the config). `--hash_size` caps every table's rows; ids are taken modulo it. |
| `--num_workers` | 2 | DataLoader worker processes. Use 0 unless the pod has a larger `/dev/shm` (GKE's default 64 MB is too small for workers). |
| `--num_train_batches` / `--num_epochs` / `--log_every` | all / 1 / 10 | Run length and logging interval. |
| `--dense_lr` / `--sparse_lr` / `--warmup_iters` / `--seed` | 1e-3 / 1e-3 / 0 / 0 | Optimization (Adam for dense, row-wise Adagrad for tables). |
| `--tb_dir` | none | Writes per-task training metrics (NE, accuracy, GAUC) and losses to TensorBoard. |

The logged loss is the sum of the per-task losses.

## Reference numbers

TPU v7x, one host, 8 processes, `--batch_size 16`, default compiler unless
noted. Steady step time excludes the first logged window.

| Run | Embeddings | First step | Steady step |
| --- | --- | --- | --- |
| `debug`, small model (above) | `cpu` | 15.5 s | 83 ms |
| `debug`, small model, `RECIR_TORCHTPU_COMPILER=static` | `cpu` | 13.0 s | 83 ms |
| `debug`, full size, history 1024 | `sparsecore` | 121 s | 99 ms |
| `yambda-50m`, full-size model, history 1024 | `sparsecore` | about 2 min | 62 ms |

On `yambda-50m` the summed loss falls from 0.19 to about 0.11 over the first
200 steps.

## Limitations

- Evaluation and checkpointing are not wired up for TorchTPU yet:
  `--eval_every`, `--ckpt_dir` and `--resume_from` raise.
- Multi-host slices run the same command: the trainer reads the GKE TPU
  environment (`TPU_WORKER_HOSTNAMES`, `TPU_WORKER_ID`, the slice topology) to
  set ranks, and two-host (16-device) runs have worked. The supplied manifest
  covers one host; for two hosts, change the topology to `2x2x2` and set
  `parallelism` and `completions` to 2.
- Keep `--batch_size`, `--uih_len`, `--num_candidates`, table sizes and the
  number of devices fixed within a run; a new shape means a new compile.
