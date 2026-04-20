# DLRMv3 on GKE TPU v7x

End-to-end DLRMv3 training with:
- **torchrec DMP** ROW_WISE embedding sharding on CPU (gloo PG)
- **recir @compile** XLA GSPMD dense path on TPU (JAX PJRT)
- **Dual process group**: gloo (TCP) for embeddings + PJRT (ICI) for dense

## Quick Start

### Build Docker image
```bash
cd ~/recir
./recir/examples/dlrmv3/build_image.sh
```

### Run on 128-chip TPU slice
```bash
kubectl apply -f recir/examples/dlrmv3/dlrmv3-128chip.yaml \
  --context=gke_mrs-tpu-y317h_us-central1_recir-v7-128
```

### Presets

| Preset | Hash size | Emb dim | Layers | Batch | Use case |
|--------|-----------|---------|--------|-------|----------|
| `--preset small` | 1K | 64 | 2 | 4 | Smoke test |
| `--preset medium` | 100K | 128 | 6 | 32 | Development |
| (default) | 10M | 256 | 12 | 128 | Production scale |

### Monitor
```bash
kubectl get pods -o wide -w --context=gke_mrs-tpu-y317h_us-central1_recir-v7-128
kubectl logs <pod-name> --context=gke_mrs-tpu-y317h_us-central1_recir-v7-128
```

## Architecture

```
Pod (1 of 32)
├── CPU: torchrec DMP EmbeddingCollection (ROW_WISE sharded via gloo)
│     └── Fused in-backward optimizer (TBE kernel)
└── TPU: recir @compile(XLA GSPMD)
      └── DlrmV3Dense → HSTU transducer (attention layers)
            └── GraphModuleTransformation(enable_device_state=True)
```

## Dual Process Group

- **JAX PJRT**: auto-initialized by GKE env vars (`TPU_HOST_BOUNDS`, `TPU_PROCESS_ADDRESSES`)
- **torch.distributed gloo**: manually initialized using `JOB_COMPLETION_INDEX` as rank, first `TPU_PROCESS_ADDRESSES` entry as coordinator on port 29500
- Ranks are different between the two PGs (JAX = topology order, gloo = pod index order)
