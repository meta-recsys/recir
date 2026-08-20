#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Benchmark: fabricate a DLRMv3 inference graph from the training graph.

Demonstrates the RecIR "inference graph fabrication" thesis on the DLRMv3 dense
(HSTU) path: instead of *retracing* the model to get an inference graph, reuse
the FX graph already captured for the training step and surgically strip the
training-only ops (loss / backward / optimizer / collectives / dropout).

The headline comparison:

* **Surgery** — ``fabricate_inference_graph`` on a copy of the training graph.
  Pure graph rewriting; should take milliseconds.
* **Retrace** — ``recir.trace`` of a forward-only step from scratch.  This is
  the cost the fabrication path avoids (TGIF's retrace, minutes at prod scale).

It also reports node-count reduction (train -> inference) and confirms the
fabricated inference graph executes on CPU.

The dense path uses the PyTorch hammer kernel (set in ``create_dlrmv3_wrapper``)
so everything runs on CPU — no TPU / JAX required.  The sparse embedding step is
swapped for ``DummyPreprocess`` (zeros with stable shapes) so the model is
``make_fx``-traceable, exactly as the existing DLRMv3 tests/benchmarks do.

Usage::

    buck2 run //recir/examples/dlrmv3:fabricate_benchmark -- --preset small
"""

from __future__ import annotations

import argparse
import copy
import logging
import time

import fbgemm_gpu  # noqa: F401 -- force C++ op registration before fallback
import recir
import recir.lowering.fbgemm_fallback  # noqa: F401 -- registers SPMD decomp entries
import torch
import torch.fx as fx
import torch.nn as nn
from recir.examples.dlrmv3.config import DLRMv3Config
from recir.graph_utils import OP
from recir.region_transforms.training_to_inference import (
    fabricate_inference_graph,
    inputs_to_ops,
)
from recir.tests.xla.dlrmv3_test_utils import (
    create_dlrmv3_wrapper,
    create_static_batch,
    DlrmV3SparseDispatcher,
    DummyPreprocess,
    get_hstu_config,
)
from torchrec.modules.embedding_configs import EmbeddingConfig

logger: logging.Logger = logging.getLogger(__name__)

Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname).1s%(asctime)s %(filename)s:%(lineno)d] %(message)s",
        datefmt="%m%d %H:%M:%S",
        force=True,
    )


def build_model_and_batch(
    cfg: DLRMv3Config,
) -> tuple[nn.Module, DlrmV3SparseDispatcher, Batch]:
    """Build the DLRMv3 wrapper + a static batch, with the sparse step stubbed.

    Returns ``(wrapper, dispatcher, batch)``.  ``dispatcher.install_eager`` has
    already swapped the real preprocess for ``DummyPreprocess`` so the model is
    traceable; the embedding params remain on the module (dead during tracing).
    """
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
        cfg.local_batch_size, hstu_config=hstu_cfg, table_configs=table_configs
    )
    batch = create_static_batch(
        hstu_config,
        cfg.local_batch_size,
        seed=42,
        uih_len_override=cfg.uih_len,
        num_candidates_override=cfg.num_candidates,
        hash_size_override=cfg.hash_size,
    )
    dummy = DummyPreprocess(
        emb_keys=emb_keys,
        batch_size=cfg.local_batch_size,
        embedding_dim=cfg.embedding_dim,
        static_uih_len=cfg.uih_len,
        static_num_candidates=cfg.num_candidates,
        user_emb_feature_names=hstu_config.user_embedding_feature_names,
        item_emb_feature_names=hstu_config.item_embedding_feature_names,
        hstu_configs=hstu_config,
    )
    dispatcher = DlrmV3SparseDispatcher(dummy)
    dispatcher.install_eager(wrapper)
    return wrapper, dispatcher, batch


def train_step_fn(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    uih_values: torch.Tensor,
    uih_lengths: torch.Tensor,
    cand_values: torch.Tensor,
    cand_lengths: torch.Tensor,
) -> None:
    """Full training step: forward -> loss -> backward -> optimizer."""
    model.train(True)
    aux_losses: dict[str, torch.Tensor] = model(
        uih_values, uih_lengths, cand_values, cand_lengths
    )
    loss = torch.stack(list(aux_losses.values())).sum()
    loss.backward()
    opt.step()
    opt.zero_grad()


def forward_step_fn(
    model: nn.Module,
    uih_values: torch.Tensor,
    uih_lengths: torch.Tensor,
    cand_values: torch.Tensor,
    cand_lengths: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Forward-only step used to time a from-scratch inference retrace."""
    return model(uih_values, uih_lengths, cand_values, cand_lengths)


def _node_count(gm: fx.GraphModule) -> int:
    return len(list(gm.graph.nodes))


def _count_substr(gm: fx.GraphModule, needle: str) -> int:
    return sum(
        1
        for n in gm.graph.nodes
        if n.op == OP.CALL_FUNCTION and needle in str(n.target).lower()
    )


def run(cfg: DLRMv3Config, loss_op: str) -> None:
    """Trace the train step, fabricate inference, and benchmark vs a retrace."""
    torch.manual_seed(0)
    wrapper, _dispatcher, batch = build_model_and_batch(cfg)
    # Citrine C2: foreach=True for multi-tensor optimizer execution.
    opt = torch.optim.SGD(wrapper.parameters(), lr=0.01, foreach=True)

    # --- Shared training trace (amortized; this cost is paid for training) ---
    t0 = time.perf_counter()
    gm_train, _flat_state = recir.trace(train_step_fn, wrapper, opt, *batch)
    train_trace_s = time.perf_counter() - t0
    n_train = _node_count(gm_train)
    logger.info(
        "Training graph traced in %.2fs: %d nodes (dropout=%d, backward/sum=%d ops)",
        train_trace_s,
        n_train,
        _count_substr(gm_train, "dropout"),
        _count_substr(gm_train, "sum"),
    )

    # --- Path A: SURGERY (fabricate inference graph from the training graph) ---
    gm_surgery = copy.deepcopy(gm_train)
    t0 = time.perf_counter()
    fabricate_inference_graph(
        gm_surgery,
        inference_outputs=inputs_to_ops({loss_op}),
        strip_training_collectives=True,
    )
    surgery_ms = (time.perf_counter() - t0) * 1000.0
    n_inf = _node_count(gm_surgery)
    logger.info(
        "SURGERY: %.1f ms -> %d nodes (%.1f%% smaller, dropout=%d)",
        surgery_ms,
        n_inf,
        100.0 * (n_train - n_inf) / max(n_train, 1),
        _count_substr(gm_surgery, "dropout"),
    )

    # --- Path B: RETRACE a forward-only inference graph from scratch ---
    t0 = time.perf_counter()
    gm_retrace, _ = recir.trace(forward_step_fn, wrapper, *batch)
    retrace_s = time.perf_counter() - t0
    logger.info(
        "RETRACE: %.2fs (%.0fx slower than surgery) -> %d nodes",
        retrace_s,
        (retrace_s * 1000.0) / max(surgery_ms, 1e-6),
        _node_count(gm_retrace),
    )

    # --- Confirm the fabricated inference graph executes on CPU ---
    _try_execute(gm_surgery, wrapper, opt, batch)

    _print_summary(cfg, train_trace_s, surgery_ms, retrace_s, n_train, n_inf)


def _try_execute(
    gm: fx.GraphModule,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    batch: Batch,
) -> None:
    """Execute the fabricated graph and report determinism (dropout removed).

    A RecIR-traced gm is a functional graph whose ``forward`` signature is the
    five make_fx containers ``(params, buffers, named_states, args, kwargs)``;
    ``args`` mirrors the original train-step call ``(model, opt, *batch)`` (the
    non-tensor ``model`` / ``opt`` leaves are baked-in constants).
    """
    params = dict(model.named_parameters(remove_duplicate=False))
    buffers = dict(model.named_buffers(remove_duplicate=False))
    named_states: dict[str, object] = {}
    call_args = (model, opt, *batch)
    try:
        with torch.no_grad():
            out1 = gm(params, buffers, named_states, call_args, {})
            out2 = gm(params, buffers, named_states, call_args, {})
    except Exception as e:  # noqa: BLE001 -- benchmark: report, don't crash
        logger.warning("Fabricated graph did not execute: %s", e)
        return
    leaves1 = torch.utils._pytree.tree_leaves(out1)
    deterministic = all(
        torch.equal(a, b)
        for a, b in zip(leaves1, torch.utils._pytree.tree_leaves(out2))
        if isinstance(a, torch.Tensor)
    )
    shapes = [tuple(t.shape) for t in leaves1 if isinstance(t, torch.Tensor)]
    logger.info(
        "EXECUTE: ok, %d output tensor(s) shapes=%s, deterministic=%s",
        len(shapes),
        shapes,
        deterministic,
    )


def _print_summary(
    cfg: DLRMv3Config,
    train_trace_s: float,
    surgery_ms: float,
    retrace_s: float,
    n_train: int,
    n_inf: int,
) -> None:
    print("\n" + "=" * 72)
    print("  DLRMv3 Inference Graph Fabrication Benchmark")
    print("=" * 72)
    print(f"  layers={cfg.num_layers} heads={cfg.num_heads} uih_len={cfg.uih_len} ")
    print("-" * 72)
    print(f"  {'training trace (shared)':<34} {train_trace_s * 1000:>10.1f} ms")
    print(f"  {'fabricate inference (surgery)':<34} {surgery_ms:>10.1f} ms")
    print(f"  {'retrace inference (from scratch)':<34} {retrace_s * 1000:>10.1f} ms")
    speedup = (retrace_s * 1000.0) / max(surgery_ms, 1e-6)
    print("-" * 72)
    print(f"  surgery is {speedup:.0f}x faster than retrace")
    print(f"  nodes: train={n_train} -> inference={n_inf}")
    print("=" * 72)


def main() -> None:
    _setup_logging()
    parser = argparse.ArgumentParser(description="DLRMv3 inference fabrication bench")
    parser.add_argument("--hash_size", type=int, default=1000)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--transducer_dim", type=int, default=64)
    parser.add_argument("--attn_dim", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--local_batch_size", type=int, default=16)
    parser.add_argument("--uih_len", type=int, default=50)
    parser.add_argument("--num_candidates", type=int, default=4)
    parser.add_argument(
        "--loss_op",
        type=str,
        default="stack",
        help="op-name substring whose inputs become the inference outputs "
        "(the loss-stacking op; predictions feed into it)",
    )
    parser.add_argument(
        "--preset", type=str, default="small", choices=["small", "medium"]
    )
    args = parser.parse_args()

    if args.preset == "medium":
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
    logger.info("Config: %s", cfg)
    run(cfg, args.loss_op)


if __name__ == "__main__":
    main()
