# DLRMv3 on recir

End-to-end training of the [generative-recommenders](https://github.com/meta-recsys/generative-recommenders)
DLRMv3 model (`DlrmHSTU`: HSTU transducer + multitask head) with recir, on GPU
and on TPU. Everything here is open-source: torch, torchrec, fbgemm_gpu,
generative-recommenders, JAX (TPU), and recir.

```
dlrmv3/
  common/   model, token-budget batch layout, data pipeline, config, metrics (shared)
  gpu/      HybridParallel + Inductor + FBGEMM TBE, one process per GPU (torchrun)
  tpu/      GSPMD + XLA via torchax, embedding tables inside the program (in_graph)
```

Both trainers trace the full step (forward, backward, optimizers, LR warmup)
into one graph with `recir.training` (`compile_hdp_step` on GPU,
`compile_gspmd_step` on TPU), and support train, eval (NE / accuracy / GAUC), DCP
checkpoint + resume, and TensorBoard.

Requires a generative-recommenders version whose `DlrmHSTU` accepts static max
lengths and `candidate_weights` and whose jagged and Triton ops trace with fake
tensors (static-shape PyTorch concat/split, Triton kernels as custom ops).

## Token-budget batch layout

recir compiles static shapes. The batch keeps every sample's real history
length and makes only the totals static (`common/model.py:FeatureLayout`,
`common/data.py:TokenBudgetCollate`):

- The batch is one block per chip (TPU) or per rank (GPU): `--batch_size` real
  rows plus dummy rows.
- Each UIH sequence feature has exactly `batch_size * uih_budget` tokens per
  block. Real rows keep their real length (at most `--uih_len`, the most recent
  events). Dummy rows take the remaining tokens, at most `uih_len` each.
- Attention never sees padding. Dummy rows and padded candidates
  (`--num_candidates` per row) get weight 0, so they never affect the loss or
  metrics.
- `--uih_budget` defaults to `uih_len`: nothing is ever trimmed, at the cost of
  up to as many dummy rows as real rows. A smaller budget (for example the
  dataset's mean history length) needs fewer dummy rows. When a block's real
  tokens exceed it, the longest histories lose their oldest events.

`--uih_len` and `--num_candidates` default to the dataset's HSTU config.

## Data

Datasets come from `generative_recommenders.dlrm_v3.utils.get_dataset`:
`debug` (random), `kuairand-1k`, `movielens-1m`, `movielens-20m`, and so on.
For the real datasets, run generative-recommenders'
`dlrm_v3/preprocess_public_data.py` and pass the directory that contains
`data/` as `--data_root`.

## GPU

```bash
pip install torch torchrec fbgemm-gpu generative-recommenders gin-config pandas tensorboard
pip install -e /path/to/recir

# 8 GPUs, KuaiRand-1K
torchrun --nproc_per_node=8 -m recir.examples.dlrmv3.gpu.train \
  --dataset kuairand-1k --data_root /data --batch_size 64 \
  --eval_every 500 --ckpt_dir /ckpt/dlrmv3 --ckpt_every 1000 --tb_dir /tb/dlrmv3
```

- The model is upstream `DlrmHSTU` running generative-recommenders' Triton
  kernels; Inductor compiles the rest of the step.
- Tables with at least `--row_wise_min_rows` rows are row-wise sharded, with
  all-to-all routing in the graph, and lowered to FBGEMM TBE with fused
  RowWiseAdagrad. Smaller tables are replicated. `--no_tbe` keeps plain
  row-wise lookups.
- Dense parameters are replicated, and their gradients are all-reduced inside
  the graph.
- `--pipeline sdd|steady` overlaps the embedding all-to-all with the previous
  step.
- Eval uses a forward-only graph compiled together with the train graph; both
  read the same sharded parameters directly.

Smoke test on one GPU:

```bash
torchrun --nproc_per_node=1 -m recir.examples.dlrmv3.gpu.train --dataset debug \
  --batch_size 8 --uih_len 256 --num_candidates 16 --num_train_batches 20 \
  --embedding_dim 64 --num_layers 2 --hash_size 1000 --row_wise_min_rows 500
```

## TPU

Build the image and launch a GKE JobSet (see `tpu/manifests/`):

```bash
IMAGE=<registry>/dlrmv3_tpu:latest ./recir/examples/dlrmv3/tpu/build_image.sh
IMAGE=<registry>/dlrmv3_tpu:latest envsubst < recir/examples/dlrmv3/tpu/manifests/dlrmv3-4chip.yaml | kubectl create -f -
```

- `--batch_size` is per chip.
- torch runs on the host CPU, and the step is lowered to one GSPMD program
  over all chips.
- Embedding tables are parameters inside the same GSPMD program (the
  `recir.embedding` in_graph backend). They are row-wise sharded over all chips,
  looked up with the ring-shift kernel, and trained by a second optimizer
  compiled into the graph. The ring-shift kernel only runs on real TPUs.
- `--embedding_backend cpu` keeps the tables in torchrec DMP on the hosts
  instead (graph-break lookup, fused sparse optimizer), for debugging without
  TPUs.
- Dropout uses `DropoutPRNGTransform`, keyed by the in-graph warmup counter so
  each compiled step receives a deterministic, iteration-specific mask.
- Checkpointing supports the default `in_graph` backend on single- and
  multi-host jobs. The `cpu` embedding backend does not yet checkpoint its
  torchrec fused optimizer state.

The TPU path also runs on CPU for local debugging, with the `cpu` backend:

```bash
JAX_PLATFORMS=cpu python -m recir.examples.dlrmv3.tpu.train --dataset debug \
  --batch_size 4 --num_train_batches 6 --embedding_dim 64 --num_layers 2 \
  --hash_size 1000 --embedding_backend cpu
```
