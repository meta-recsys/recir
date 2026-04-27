"""DLRMv3 HSTU TPU Inference Benchmark.

Measures latency and throughput of the dense (HSTU) forward pass on TPU
via RecIR's @compile with XLA GSPMD backend. Embeddings stay on CPU
(via DlrmV3SparseOverride), matching the production inference architecture.

Usage:
    python3 -u -m recir.examples.dlrmv3.infer_benchmark \
        --local_batch_size 16 --num_warmup 5 --num_steps 50

    # Sweep batch sizes:
    python3 -u -m recir.examples.dlrmv3.infer_benchmark --sweep
"""

import argparse
import logging
import statistics
import time
from typing import Any, Dict, List, Tuple

import jax
import torch
import torch.nn as nn
from recir.api import compile
from recir.backend import BackendType
from recir.distributed.parallel_mode import GSPMDParallel
from recir.examples.dlrmv3.config import DLRMv3Config
from recir.gm_transformation import GraphModuleTransformation
from recir.tests.xla.dlrmv3_test_utils import (
    create_dlrmv3_wrapper,
    create_static_batch,
    DlrmV3SparseOverride,
    DummyPreprocess,
    get_hstu_config,
)

logger: logging.Logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname).1s%(asctime)s %(filename)s:%(lineno)d] %(message)s",
        datefmt="%m%d %H:%M:%S",
    )


def create_model_and_batch(
    cfg: DLRMv3Config,
    batch_size: int,
    is_inference: bool = False,
) -> Tuple[nn.Module, List[str], Any, Tuple[torch.Tensor, ...]]:
    """Create model and static batch for inference."""
    from torchrec import EmbeddingConfig

    table_configs = [
        EmbeddingConfig(
            num_embeddings=cfg.hash_size,
            embedding_dim=cfg.embedding_dim,
            name="post_id",
            feature_names=[
                "uih_post_id",
                "item_post_id",
                "uih_owner_id",
                "item_owner_id",
            ],
        ),
        EmbeddingConfig(
            num_embeddings=cfg.hash_size,
            embedding_dim=cfg.embedding_dim,
            name="viewer_id",
            feature_names=["viewer_id"],
        ),
        EmbeddingConfig(
            num_embeddings=cfg.hash_size,
            embedding_dim=cfg.embedding_dim,
            name="dummy_contexual",
            feature_names=["dummy_contexual"],
        ),
    ]

    hstu_cfg = get_hstu_config(
        embedding_dim=cfg.embedding_dim,
        hidden_dim=cfg.hidden_dim,
        transducer_dim=cfg.transducer_dim,
        attn_dim=cfg.attn_dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        uih_len=cfg.uih_len,
        num_candidates=cfg.num_candidates,
    )
    wrapper, hstu_config, _, emb_keys = create_dlrmv3_wrapper(
        batch_size,
        hstu_config=hstu_cfg,
        table_configs=table_configs,
        is_inference=is_inference,
    )
    batch = create_static_batch(
        hstu_config,
        batch_size,
        seed=42,
        uih_len_override=cfg.uih_len,
        num_candidates_override=cfg.num_candidates,
        hash_size_override=cfg.hash_size,
    )

    return wrapper, emb_keys, hstu_config, batch


def run_benchmark(
    cfg: DLRMv3Config,
    batch_size: int,
    num_warmup: int,
    num_steps: int,
) -> Dict[str, float]:
    """Run inference benchmark for a single batch size.

    Returns dict with latency stats (ms) and throughput (samples/sec).
    """
    logger.info(f"--- Batch size {batch_size} ---")

    wrapper, emb_keys, hstu_config, batch = create_model_and_batch(
        cfg, batch_size, is_inference=True
    )
    # Inference mode: model.eval() + is_inference=True skips loss computation
    wrapper.eval()

    # Sparse override for @compile (embedding is not traceable)
    dummy = DummyPreprocess(
        emb_keys=emb_keys,
        batch_size=batch_size,
        embedding_dim=cfg.embedding_dim,
        static_uih_len=cfg.uih_len,
        static_num_candidates=cfg.num_candidates,
        user_emb_feature_names=hstu_config.user_embedding_feature_names,
        item_emb_feature_names=hstu_config.item_embedding_feature_names,
        hstu_configs=hstu_config,
    )
    sparse_override = DlrmV3SparseOverride(dummy)
    sparse_override._num_ranks = 1

    num_tpu_devices = jax.device_count()
    num_ranks = num_tpu_devices if num_tpu_devices > 1 else 1

    @compile(
        module_override=[sparse_override],
        parallel_mode=GSPMDParallel(
            "replicate",
            loss_reduction="sum",
            num_ranks=num_ranks,
            num_processes=1,
        ),
        # pyre-ignore[6]: GraphModuleTransformation is callable
        gm_transformation=GraphModuleTransformation(
            backend=BackendType.XLA_GSPMD,
            enable_device_state=True,
        ),
    )
    def inference_step(
        mod: nn.Module,
        uih_values: torch.Tensor,
        uih_lengths: torch.Tensor,
        cand_values: torch.Tensor,
        cand_lengths: torch.Tensor,
    ) -> None:
        # Pure forward — is_inference=True skips loss computation
        mod(uih_values, uih_lengths, cand_values, cand_lengths)

    uih_values, uih_lengths, cand_values, cand_lengths = batch

    # Warmup (includes XLA compilation)
    logger.info(f"  Warmup: {num_warmup} steps (includes XLA compile)...")
    warmup_start = time.perf_counter()
    for _ in range(num_warmup):
        inference_step(wrapper, uih_values, uih_lengths, cand_values, cand_lengths)
    warmup_time = time.perf_counter() - warmup_start
    logger.info(f"  Warmup done in {warmup_time:.2f}s")

    # Benchmark
    logger.info(f"  Benchmarking: {num_steps} steps...")
    latencies: List[float] = []
    for _ in range(num_steps):
        t0 = time.perf_counter()
        inference_step(wrapper, uih_values, uih_lengths, cand_values, cand_lengths)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]
    mean = statistics.mean(latencies)
    throughput = batch_size / (mean / 1000.0)

    result = {
        "batch_size": batch_size,
        "warmup_s": warmup_time,
        "mean_ms": mean,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "throughput_sps": throughput,
    }
    logger.info(
        f"  Results: p50={p50:.2f}ms p95={p95:.2f}ms p99={p99:.2f}ms "
        f"throughput={throughput:.0f} samples/s"
    )
    return result


def main() -> None:
    _setup_logging()

    parser = argparse.ArgumentParser(description="DLRMv3 TPU Inference Benchmark")
    parser.add_argument("--local_batch_size", type=int, default=16)
    parser.add_argument("--hash_size", type=int, default=1000)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--transducer_dim", type=int, default=64)
    parser.add_argument("--attn_dim", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--uih_len", type=int, default=50)
    parser.add_argument("--num_candidates", type=int, default=4)
    parser.add_argument("--num_warmup", type=int, default=5)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Sweep batch sizes [8, 16, 32, 64, 128, 256]",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=["small", "medium"],
    )
    args = parser.parse_args()

    if args.preset == "small":
        args.hash_size = 1000
        args.hidden_dim = 32
        args.transducer_dim = 64
        args.attn_dim = 32
        args.num_layers = 2
        args.num_heads = 2
        args.uih_len = 50
        args.num_candidates = 4
    elif args.preset == "medium":
        args.hash_size = 100_000
        args.embedding_dim = 128
        args.hidden_dim = 128
        args.transducer_dim = 128
        args.attn_dim = 64
        args.num_layers = 6
        args.num_heads = 4
        args.uih_len = 200
        args.num_candidates = 16

    cfg = DLRMv3Config(
        hash_size=args.hash_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        transducer_dim=args.transducer_dim,
        attn_dim=args.attn_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        local_batch_size=args.local_batch_size,
        uih_len=args.uih_len,
        num_candidates=args.num_candidates,
        max_steps=0,
    )

    num_tpu_devices = jax.device_count()
    platform = jax.devices()[0].platform
    logger.info(f"Platform: {platform}, TPU devices: {num_tpu_devices}")
    logger.info(f"Config: {cfg}")

    batch_sizes = [8, 16, 32, 64, 128, 256] if args.sweep else [args.local_batch_size]

    results: List[Dict[str, float]] = []
    for bs in batch_sizes:
        r = run_benchmark(cfg, bs, args.num_warmup, args.num_steps)
        results.append(r)

    # Print summary table
    print("\n" + "=" * 80)
    print("  DLRMv3 HSTU TPU Inference Benchmark Results")
    print("=" * 80)
    print(
        f"  Platform: {platform} | Devices: {num_tpu_devices} | "
        f"Layers: {cfg.num_layers} | Heads: {cfg.num_heads}"
    )
    print(f"  UIH len: {cfg.uih_len} | Candidates: {cfg.num_candidates}")
    print("-" * 80)
    print(
        f"{'Batch':>8} | {'Warmup(s)':>10} | {'Mean(ms)':>10} | "
        f"{'P50(ms)':>9} | {'P95(ms)':>9} | {'P99(ms)':>9} | {'Throughput':>12}"
    )
    print("-" * 80)
    for r in results:
        print(
            f"{r['batch_size']:>8} | {r['warmup_s']:>10.2f} | {r['mean_ms']:>10.2f} | "
            f"{r['p50_ms']:>9.2f} | {r['p95_ms']:>9.2f} | {r['p99_ms']:>9.2f} | "
            f"{r['throughput_sps']:>9.0f} s/s"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
