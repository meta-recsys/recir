# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""CPU half of the torchrec EBC perf baseline for deliverable 1.2.

Task 1.2 names ``torchrec.EmbeddingBagCollection`` as *the* perf baseline. It
cannot be timed inside the TPU job: that image does not install torchrec. So
this follows the same two-stage split as the correctness parity check
(``ringshift_ebc_parity_dump`` + ``test_embedding_bag_torchrec_tpu``) -- the TPU
side produces its own median, this side produces torchrec's, and this side
combines them into the headline.

Point ``--tpu-log`` at the log a ``--mode ebc`` run left in ``bench_logs/`` and
it extracts the fused median and prints::

    === RESULT ebc_vs_torchrec: baseline=torchrec.EmbeddingBagCollection \
        torchrec=..ms(CPU) fused=..ms(TPU) speedup=..x ... ===

Without ``--tpu-log`` it just reports the torchrec median, which is the useful
mode while iterating on shapes.

Two things this comparison is NOT, stated here because the headline number
invites both readings:

* It is not same-hardware. torchrec EBC runs eager on this host's CPU; the
  fused kernel runs on TPU. The speedup therefore folds in the hardware move,
  which is the point of the comparison but is not a kernel speedup.
* It is not the same tensors. Timing needs a matching *shape and distribution*,
  not matching values, so inputs are drawn here rather than shipped from the
  TPU run. Value parity is the parity check's job, not this one.
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
from torchrec.modules.embedding_configs import EmbeddingBagConfig, PoolingType
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

_TAG = "BENCH_EBC_TORCHREC"


@dataclass(frozen=True)
class TpuResult:
    world_size: int
    dim: int
    v_local: int
    max_local_ids: int
    num_bags: int
    pooling: str
    distribution: str
    xla_ms: float
    fused_ms: float


def _log(msg: str) -> None:
    print(f"[{_TAG}] {msg}", file=sys.stderr, flush=True)


def _bag_lengths(rng, num_bags: int, total: int, *, heavy_tailed: bool):
    """Jagged lengths summing exactly to ``total``.

    Mirrors the TPU bench's construction (normalised weights, floor, remainder
    into the last bag) so the two sides see the same *shape* of work, including
    the empty bags a heavy-tailed draw produces.
    """
    w = rng.exponential(size=num_bags) if heavy_tailed else rng.uniform(size=num_bags)
    w = w / w.sum()
    lengths = np.floor(w * total).astype(np.int64)
    lengths[-1] += total - int(lengths.sum())
    assert int(lengths.sum()) == total and lengths.min() >= 0
    return lengths


def _build(cfg):
    """One global batch: every rank's bags concatenated.

    torchrec eager is single-process, so the equivalent of the sharded TPU run
    is the whole global batch in one EBC call -- same total work, same total
    number of bags.
    """
    rng = np.random.default_rng(cfg.seed)
    vocab = cfg.world_size * cfg.v_local
    heavy = cfg.distribution == "skewed"
    # Match the TPU bench: skewed keeps 4/5 of the id budget per rank.
    per_rank = cfg.max_local_ids if not heavy else (cfg.max_local_ids * 4) // 5

    lengths = np.concatenate(
        [
            _bag_lengths(rng, cfg.num_bags, per_rank, heavy_tailed=heavy)
            for _ in range(cfg.world_size)
        ]
    )
    values = rng.integers(0, vocab, size=int(lengths.sum()), dtype=np.int64)

    table = EmbeddingBagConfig(
        name="t",
        embedding_dim=cfg.dim,
        num_embeddings=vocab,
        feature_names=["f"],
        pooling=PoolingType.SUM if cfg.pooling == "sum" else PoolingType.MEAN,
    )
    ebc = EmbeddingBagCollection(tables=[table], device=torch.device("cpu"))
    kjt = KeyedJaggedTensor.from_lengths_sync(
        keys=["f"],
        values=torch.from_numpy(values),
        lengths=torch.from_numpy(lengths),
    )
    _log(
        f"inputs ready: bags={len(lengths)} ids={int(lengths.sum())} "
        f"empty_bags={int((lengths == 0).sum())} max_bag={int(lengths.max())}"
    )
    return ebc, kjt


def _time(label: str, fn, n_warmup: int, n_measure: int) -> float:
    for _ in range(n_warmup):
        fn()
    times_ms = []
    for _ in range(n_measure):
        t0 = time.perf_counter()
        fn()
        times_ms.append((time.perf_counter() - t0) * 1e3)
    median = statistics.median(times_ms)
    _log(f"{label}: median={median:.3f}ms times={[round(t, 3) for t in times_ms]}")
    return median


def _tpu_result_from_log(path: str) -> TpuResult:
    """Read the measured TPU medians and their authoritative shape."""
    text = open(path, encoding="utf-8", errors="replace").read()
    shape_matches = re.findall(
        r"=== W=(\d+) D=(\d+) v_local=(\d+) max_local_ids=(\d+) "
        r"recv_capacity=\d+ distribution=(uniform|skewed) backend=\w+ ===",
        text,
    )
    pooling_matches = re.findall(
        r"ebc inputs ready \(num_bags=(\d+) pooling=(sum|mean)\)", text
    )
    timing_matches = re.findall(
        r"=== RESULT ebc:[^\n]*xla=([\d.]+)ms kernel=([\d.]+)ms", text
    )
    if not shape_matches or not pooling_matches or not timing_matches:
        raise SystemExit(f"incomplete `=== RESULT ebc:` record in {path}")

    world_size, dim, v_local, max_local_ids, distribution = shape_matches[-1]
    num_bags, pooling = pooling_matches[-1]
    xla_ms, fused_ms = timing_matches[-1]
    return TpuResult(
        world_size=int(world_size),
        dim=int(dim),
        v_local=int(v_local),
        max_local_ids=int(max_local_ids),
        num_bags=int(num_bags),
        pooling=pooling,
        distribution=distribution,
        xla_ms=float(xla_ms),
        fused_ms=float(fused_ms),
    )


def _use_tpu_shape(cfg, result: TpuResult) -> None:
    """Make the baseline consume exactly the shape recorded by the TPU run."""
    cfg.world_size = result.world_size
    cfg.dim = result.dim
    cfg.v_local = result.v_local
    cfg.max_local_ids = result.max_local_ids
    cfg.num_bags = result.num_bags
    cfg.pooling = result.pooling
    cfg.distribution = result.distribution


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--world-size", type=int, default=4)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--v-local", type=int, default=8192)
    p.add_argument("--max-local-ids", type=int, default=8192)
    p.add_argument("--num-bags", type=int, default=1024)
    p.add_argument("--pooling", choices=("sum", "mean"), default="sum")
    p.add_argument("--distribution", choices=("uniform", "skewed"), default="skewed")
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument(
        "--backward", action="store_true", help="time fwd+bwd instead of fwd"
    )
    p.add_argument(
        "--tpu-log",
        default=None,
        help="a bench_logs/*.log from a --mode ebc run, to print the combined headline",
    )
    cfg = p.parse_args(argv)

    tpu_result = _tpu_result_from_log(cfg.tpu_log) if cfg.tpu_log else None
    if tpu_result is not None:
        _use_tpu_shape(cfg, tpu_result)

    _log(
        f"=== W={cfg.world_size} D={cfg.dim} v_local={cfg.v_local} "
        f"max_local_ids={cfg.max_local_ids} num_bags={cfg.num_bags} "
        f"pooling={cfg.pooling} distribution={cfg.distribution} ==="
    )
    ebc, kjt = _build(cfg)

    if cfg.backward:

        def step():
            ebc.zero_grad(set_to_none=True)
            ebc(kjt)["f"].sum().backward()
    else:
        with_no_grad = torch.no_grad()

        def step():
            with with_no_grad:
                ebc(kjt)

    label = "torchrec_fwd_bwd" if cfg.backward else "torchrec_fwd"
    torchrec_median = _time(label, step, cfg.warmup, cfg.iters)

    if tpu_result is None:
        _log(f"=== RESULT ebc_torchrec: torchrec={torchrec_median:.3f}ms (CPU) ===")
        return 0

    valid_ids_per_rank = (
        cfg.max_local_ids
        if cfg.distribution == "uniform"
        else (cfg.max_local_ids * 4) // 5
    )
    average_bag_len = (
        valid_ids_per_rank / cfg.num_bags if cfg.num_bags else float("nan")
    )
    _log(
        "=== RESULT ebc_vs_torchrec: baseline=torchrec.EmbeddingBagCollection "
        f"torchrec={torchrec_median:.3f}ms(CPU) "
        f"fused={tpu_result.fused_ms:.3f}ms(TPU) "
        f"speedup={torchrec_median / tpu_result.fused_ms:.2f}x "
        f"xla={tpu_result.xla_ms:.3f}ms(TPU,reference) "
        f"shape=(world_size={cfg.world_size},dim={cfg.dim},"
        f"num_bags={cfg.num_bags},bag_len={average_bag_len:.3f},"
        f"distribution={cfg.distribution},pooling={cfg.pooling}) ==="
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
