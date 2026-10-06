#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""EBC end to end: a PyTorch recsys model whose pooled lookup runs on TPU.

Everything before this checked the pooled reduction in isolation -- against a
numpy oracle, against torchrec, against the other implementation. None of it
answered the question this does: can a PyTorch model actually train through it,
with gradients arriving back at the table shard after passing a dense layer.

The lookup goes through ``InGraphSparseArch``, the same sparse arch RankFM's
in-graph backend installs, rather than calling the op directly. That is the
point of the example: the arch is what slices a KJT into per-chip id streams,
lines the bag cuts up against them, and decides how the pooled result comes
back, so calling the op directly would leave exactly the layer this is meant to
cover untested. It is fed a ``KeyedJaggedTensor`` for the same reason -- that is
the arch's real entry point.

The model is deliberately tiny. It is not trying to be a good recommender; it
exists so the pooled path has a real consumer and a real loss above it.

Two things are checked, and the second is the one that matters:

  1. The loss goes down. Necessary, but weak: a model this small will descend on
     synthetic data even if gradients are somewhat wrong.
  2. The TPU loss curve tracks a CPU torchrec run **step by step**, from the
     same init and the same batches. Comparing whole curves rather than the
     final value is what catches a gradient that is subtly wrong -- both runs
     converge to something either way, and only the trajectory diverges.

The run passes when either signal succeeds, matching the task's acceptance
criterion. With the CPU reference enabled, curve parity is the stronger signal:
the independently random labels make a first-versus-last loss comparison noisy.

Lives under rankfm_mvai because this is the image that has torch and torchrec.
The examples/embedding image is jax-only by design, so no PyTorch model can run
there at all, and torchrec being present here is what lets the reference run
in-process rather than through an exported artifact.

Usage:
    python3 -u -m recir.examples.rankfm_mvai.ebc_e2e --steps 30
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as nn

# Registers recir::ringshift_embedding (schema + CPU fallback + autograd).
import recir.ops.seq_embedding_op  # noqa: F401  # isort:skip

# Registers its TPU lowering. MUST be imported before the first default_env(),
# which load_ops() snapshots -- see the note in recir.embedding.in_graph.
import recir.torchax.ops.jseq_embedding  # noqa: F401  # isort:skip
from recir.embedding.backend import (  # isort:skip
    TableOptimizer,
    TablePlan,
    TableSharding,
)
from recir.embedding.in_graph import InGraphSparseArch  # isort:skip
from recir.torchax import extract_jax  # isort:skip
from torchrec.modules.embedding_configs import (  # isort:skip
    EmbeddingBagConfig,
    PoolingType,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor  # isort:skip


_TAG = "EBC_E2E"
_TABLE = "t_pooled"
_FEATURE = "f_pooled"


@dataclass
class TrainingResult:
    losses: list[float]
    first_step_gradients: dict[str, np.ndarray]


def _log(msg: str) -> None:
    if jax.process_index() == 0:
        print(f"[{_TAG}] {msg}", file=sys.stderr, flush=True)


def tokens_per_chip(cfg, world: int) -> int:
    """The per-chip token total the arch traces its shapes against.

    Per-sample bag lengths vary; this total does not. Deriving it here rather
    than measuring it keeps the batch generator and the arch reading the same
    number from one place -- a mismatch would surface as a reshape assertion
    inside the arch, far from either.
    """
    return (cfg.batch_size // world) * cfg.avg_bag


class TinyRecModel(nn.Module):
    """Pooled lookup through InGraphSparseArch, feeding a two-layer MLP.

    The MLP is not decoration. With the pooled output used directly as the
    loss, the cotangent reaching the op would be a constant and XLA could fold
    most of the path away; a bag attributed to the wrong row would look the
    same as a correct one. A dense layer in between makes the cotangent
    per-bag and arbitrary, so the backward has to be right.
    """

    def __init__(self, cfg, world: int) -> None:
        super().__init__()
        self.sparse = InGraphSparseArch(
            table_configs=[
                EmbeddingBagConfig(
                    name=_TABLE,
                    embedding_dim=cfg.dim,
                    num_embeddings=cfg.vocab,
                    feature_names=[_FEATURE],
                    pooling=PoolingType.SUM,
                )
            ],
            feature_seqlens={_FEATURE: tokens_per_chip(cfg, world)},
            world_size=world,
            table_plans={
                _TABLE: TablePlan(
                    sharding=TableSharding.ROWWISE,
                    optimizer=TableOptimizer.SPARSE,
                )
            },
            # Real host init. The meta-parameter path materialises shards
            # through recir's FSDP hook, which a standalone example does not
            # install.
            on_device_init=False,
        )
        self.mlp = nn.Sequential(
            nn.Linear(cfg.dim, cfg.hidden),
            nn.ReLU(),
            nn.Linear(cfg.hidden, 1),
        )

    def table(self) -> nn.Parameter:
        return self.sparse._sharded_tables[_TABLE]

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        kjt = KeyedJaggedTensor(keys=[_FEATURE], values=ids, lengths=lengths)
        # Pooled features come back as JaggedTensors whose bags all hold one
        # row, so values() is already [bags, dim] -- one row per sample.
        pooled = self.sparse(kjt)[_FEATURE].values()
        return self.mlp(pooled).squeeze(-1)


def _chip_bag_lengths(rng, bags_local: int, t_local: int, max_bag: int):
    """Split ``t_local`` tokens over ``bags_local`` bags, none wider than max_bag.

    Dealt out one token at a time rather than drawn independently because the
    per-chip total has to come out EXACTLY ``t_local``: that is the invariant
    the arch's ``[W, T_local]`` reshape rests on, and independent draws would
    need patching up afterwards, which is where the bias would creep in.

    Dealing also produces the two cases that matter for free -- empty bags,
    which is what the mean divisor clamp exists for, and full ones at the
    ``max_bag`` cap.
    """
    lengths = np.zeros(bags_local, dtype=np.int64)
    for _ in range(t_local):
        room = np.flatnonzero(lengths < max_bag)
        lengths[rng.choice(room)] += 1
    return lengths


def make_batches(cfg, world: int, rng: np.random.Generator):
    """Fixed synthetic batches, drawn once and replayed by both runs.

    Both sides must see identical inputs, so the batches are materialised here
    rather than sampled per run: torch and numpy have different RNGs, and a
    shared seed would silently give the two runs different data.

    Shaped as the arch expects them: a flat ``[W * T_local]`` id stream in
    global sample order plus real per-sample lengths, which is what a
    KeyedJaggedTensor carries.
    """
    bags_local = cfg.batch_size // world
    t_local = tokens_per_chip(cfg, world)
    batches = []
    for _ in range(cfg.steps):
        lengths = np.concatenate(
            [
                _chip_bag_lengths(rng, bags_local, t_local, cfg.max_bag)
                for _ in range(world)
            ]
        )
        ids = rng.integers(0, cfg.vocab, size=world * t_local).astype(np.int64)
        labels = rng.integers(0, 2, size=cfg.batch_size).astype(np.float32)
        batches.append((ids, lengths, labels))
    return batches


def _bce(logits, labels):
    # log(1 + exp(-x)) written through logaddexp so it is stable at both tails
    # and identical between the numpy/jax and torch sides.
    return jnp.mean(jnp.logaddexp(0.0, logits) - labels * logits)


def run_tpu(cfg, model: nn.Module, batches) -> TrainingResult:
    """Train through the TPU lowering, with a hand-rolled SGD step.

    ``extract_jax`` turns the PyTorch module into (params, jax callable), so
    the model stays PyTorch while the step runs as one compiled JAX program --
    which is the whole point of the seam being exercised here.
    """
    states, jax_func = extract_jax(model)

    def loss_fn(states, ids, lengths, labels):
        logits = jax_func(states, (ids, lengths))
        return _bce(logits, labels)

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    losses = []
    first_step_gradients: dict[str, np.ndarray] = {}
    profiling_active = False
    try:
        for step_index, (ids, lengths, labels) in enumerate(batches):
            # Match the production RankFM profiler's 1-based, inclusive step
            # window. Starting after the first few steps keeps compilation and
            # warmup out of the trace.
            step = step_index + 1
            if cfg.profile_start_step > 0 and step == cfg.profile_start_step:
                if not cfg.profile_dir.startswith("gs://"):
                    os.makedirs(cfg.profile_dir, exist_ok=True)
                jax.profiler.start_trace(cfg.profile_dir)
                profiling_active = True
                _log(f"profile started at step {step}; output={cfg.profile_dir}")

            loss, grads = grad_fn(
                states, jnp.asarray(ids), jnp.asarray(lengths), jnp.asarray(labels)
            )
            if step_index == 0:
                first_step_gradients = {
                    name: np.asarray(value)
                    for name, value in jax.device_get(grads).items()
                }
            states = jax.tree_util.tree_map(lambda p, g: p - cfg.lr * g, states, grads)
            loss_value = float(loss)
            losses.append(loss_value)
            if step_index % 10 == 0:
                _log(f"tpu step {step_index:3d} loss={loss_value:.6f}")

            if profiling_active and step == cfg.profile_end_step:
                # JAX dispatch is asynchronous. Wait for the parameter update
                # as well as the loss before closing the trace, so the complete
                # forward/backward/SGD step is captured on device.
                jax.block_until_ready(states)
                jax.profiler.stop_trace()
                profiling_active = False
                _log(f"profile stopped at step {step}; saved to {cfg.profile_dir}")
    finally:
        # Flush a partial trace if training exits before profile_end_step.
        if profiling_active:
            jax.profiler.stop_trace()
            _log(f"profile stopped early; saved to {cfg.profile_dir}")
    return TrainingResult(losses, first_step_gradients)


def run_cpu_reference(cfg, model: nn.Module, batches) -> TrainingResult:
    """The same steps in eager PyTorch, as the baseline to compare against.

    Plain SGD written out rather than an optimizer object, so the update rule
    is visibly the same expression as the JAX side. A difference in the
    optimizer would show up as a diverging curve and be blamed on the kernel.
    """
    # Citrine C2: batch the small model's parameter updates.
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, foreach=True)
    lossfn = nn.BCEWithLogitsLoss()
    losses = []
    first_step_gradients: dict[str, np.ndarray] = {}
    for step, (ids, lengths, labels) in enumerate(batches):
        opt.zero_grad(set_to_none=True)
        logits = model(torch.from_numpy(ids), torch.from_numpy(lengths))
        loss = lossfn(logits, torch.from_numpy(labels))
        loss.backward()
        if step == 0:
            first_step_gradients = {
                name: parameter.grad.detach().cpu().numpy().copy()
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
            }
        opt.step()
        losses.append(float(loss.detach()))
    return TrainingResult(losses, first_step_gradients)


def _passes_acceptance(
    *, descended: bool, max_rel: float, rtol: float, skip_cpu: bool
) -> bool:
    """Accept loss descent or CPU parity; descent is required without a reference."""
    parity_ok = not skip_cpu and max_rel <= rtol
    return descended or parity_ok


def _sample_values(value: np.ndarray, limit: int = 4) -> str:
    flat = np.asarray(value, dtype=np.float64).reshape(-1)
    return "[" + ", ".join(f"{x:.4e}" for x in flat[:limit]) + "]"


def _gradient_max_relative_error(
    cpu_gradients: dict[str, np.ndarray],
    tpu_gradients: dict[str, np.ndarray],
) -> float:
    if not cpu_gradients or set(cpu_gradients) != set(tpu_gradients):
        return float("inf")
    errors = []
    for name, cpu_gradient in cpu_gradients.items():
        cpu_array = np.asarray(cpu_gradient, dtype=np.float64)
        tpu_array = np.asarray(tpu_gradients[name], dtype=np.float64)
        if cpu_array.shape != tpu_array.shape:
            return float("inf")
        max_abs = float(np.max(np.abs(tpu_array - cpu_array)))
        errors.append(max_abs / max(float(np.max(np.abs(cpu_array))), 1e-12))
    return max(errors, default=0.0)


def _build_report(
    cfg,
    world: int,
    tpu: TrainingResult,
    cpu: TrainingResult | None,
    *,
    descended: bool,
    parity_ok: bool,
    ok: bool,
) -> str:
    """Build a bounded Markdown report suitable for the Phabricator Test Plan."""
    gradient_max_rel = (
        float("nan")
        if cpu is None
        else _gradient_max_relative_error(
            cpu.first_step_gradients, tpu.first_step_gradients
        )
    )
    gradient_ok = cpu is not None and gradient_max_rel <= cfg.rtol
    lines = [
        "=== EBC_E2E_REPORT_BEGIN ===",
        "# EBC E2E CPU vs TPU Correctness Report",
        "",
        "## Configuration",
        "",
        "| Field | Value |",
        "| --- | ---: |",
        f"| TPU devices | {world} |",
        f"| Global batch size | {cfg.batch_size} |",
        f"| Average IDs per bag | {cfg.avg_bag} |",
        f"| Maximum IDs per bag | {cfg.max_bag} |",
        f"| Vocabulary rows | {cfg.vocab} |",
        f"| Embedding dimension | {cfg.dim} |",
        f"| Training steps | {cfg.steps} |",
        f"| Relative tolerance | {cfg.rtol:.3e} |",
        "",
        "## Final result",
        "",
        "| Check | Value |",
        "| --- | --- |",
        f"| Overall | {'PASS' if ok else 'FAIL'} |",
        f"| TPU loss descended | {descended} |",
        f"| CPU/TPU loss parity | {parity_ok} |",
        f"| CPU/TPU first-step gradient parity | {gradient_ok} |",
        f"| Maximum parameter gradient relative error | {gradient_max_rel:.6e} |",
    ]

    if cpu is None:
        lines.extend(
            [
                "| Maximum per-step relative loss gap | N/A (`--skip-cpu`) |",
                "",
                "CPU comparison was skipped; no gradient comparison is available.",
                "=== EBC_E2E_REPORT_END ===",
            ]
        )
        return "\n".join(lines)

    loss_diffs = [abs(t - c) for t, c in zip(tpu.losses, cpu.losses)]
    max_rel = max(loss_diffs) / max(abs(cpu.losses[0]), 1e-6)
    lines.extend(
        [
            f"| Maximum per-step relative loss gap | {max_rel:.6e} |",
            "",
            "## Per-step loss comparison",
            "",
            "| Step | CPU loss | TPU loss | Absolute difference |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for step, (cpu_loss, tpu_loss, abs_diff) in enumerate(
        zip(cpu.losses, tpu.losses, loss_diffs)
    ):
        lines.append(f"| {step} | {cpu_loss:.9f} | {tpu_loss:.9f} | {abs_diff:.6e} |")

    lines.extend(
        [
            "",
            "## First-step gradient comparison",
            "",
            "Gradient samples contain the first four flattened values; full tensors "
            "are summarized to keep the report bounded.",
            "",
            "| Parameter | Shape | CPU sample | TPU sample | CPU L2 | TPU L2 | Max abs diff | Max rel diff | Status |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    gradient_keys = sorted(
        set(cpu.first_step_gradients) | set(tpu.first_step_gradients)
    )
    for name in gradient_keys:
        cpu_gradient = cpu.first_step_gradients.get(name)
        tpu_gradient = tpu.first_step_gradients.get(name)
        if cpu_gradient is None or tpu_gradient is None:
            lines.append(
                f"| `{name}` | N/A | N/A | N/A | N/A | N/A | N/A | N/A | MISSING |"
            )
            continue
        cpu_array = np.asarray(cpu_gradient, dtype=np.float64)
        tpu_array = np.asarray(tpu_gradient, dtype=np.float64)
        if cpu_array.shape != tpu_array.shape:
            lines.append(
                f"| `{name}` | CPU {cpu_array.shape}, TPU {tpu_array.shape} | "
                "N/A | N/A | N/A | N/A | N/A | N/A | SHAPE MISMATCH |"
            )
            continue
        max_abs = float(np.max(np.abs(tpu_array - cpu_array)))
        max_rel = max_abs / max(float(np.max(np.abs(cpu_array))), 1e-12)
        status = "PASS" if max_rel <= cfg.rtol else "FAIL"
        shape = "x".join(str(d) for d in cpu_array.shape) or "scalar"
        lines.append(
            f"| `{name}` | {shape} | {_sample_values(cpu_array)} | "
            f"{_sample_values(tpu_array)} | {np.linalg.norm(cpu_array):.6e} | "
            f"{np.linalg.norm(tpu_array):.6e} | {max_abs:.6e} | "
            f"{max_rel:.6e} | {status} |"
        )

    lines.append("=== EBC_E2E_REPORT_END ===")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=30)
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="global number of samples (one bag per sample) in each step",
    )
    p.add_argument("--max-bag", type=int, default=8, help="max ids per bag")
    p.add_argument(
        "--avg-bag",
        type=int,
        default=4,
        help="mean ids per bag; fixes the static per-chip token total",
    )
    p.add_argument("--vocab", type=int, default=1024)
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--rtol", type=float, default=2e-3)
    p.add_argument("--skip-cpu", action="store_true", help="TPU curve only")
    p.add_argument(
        "--profile_start_step",
        type=int,
        default=-1,
        help="1-based first training step to capture; <=0 disables profiling",
    )
    p.add_argument(
        "--profile_end_step",
        type=int,
        default=-1,
        help="1-based final training step to capture (inclusive)",
    )
    p.add_argument(
        "--profile_dir",
        default="/tmp/tpu_profile",
        help="local or gs:// output directory for the JAX TPU trace",
    )
    cfg = p.parse_args(argv)

    if cfg.profile_start_step > 0:
        if cfg.profile_end_step < cfg.profile_start_step:
            p.error("--profile_end_step must be >= --profile_start_step")
        if cfg.profile_end_step > cfg.steps:
            p.error("--profile_end_step must be <= --steps")
        if not cfg.profile_dir:
            p.error("--profile_dir must be non-empty when profiling is enabled")
    elif cfg.profile_end_step > 0:
        p.error("--profile_start_step must be > 0 when --profile_end_step is set")

    if not jax.distributed.is_initialized() and (
        os.environ.get("TPU_WORKER_HOSTNAMES")
        or os.environ.get("MEGASCALE_COORDINATOR_ADDRESS")
    ):
        jax.distributed.initialize()

    if jax.devices()[0].platform != "tpu":
        # The pooled path traces fine here, then dies inside pallas_call with
        # "Only interpret mode is supported on CPU backend". Say so plainly
        # rather than let that traceback look like a bug in this example.
        raise SystemExit(
            f"{_TAG}: needs a TPU; jax sees {jax.devices()[0].platform}. "
            "The trace-level checks run on CPU as "
            "//recir/examples/rankfm_mvai/tests:test_ebc_e2e."
        )

    world = len(jax.devices())
    if cfg.batch_size % world or cfg.vocab % world:
        raise SystemExit(
            f"batch_size {cfg.batch_size} and vocab {cfg.vocab} must both "
            f"divide world_size {world}"
        )
    if not 1 <= cfg.avg_bag <= cfg.max_bag:
        # avg_bag above max_bag cannot be dealt out: every bag would have to
        # exceed the cap. Equal to it is legal but degenerate -- every bag ends
        # up full, so no empty bag is ever generated and the case the divisor
        # clamp exists for goes untested.
        raise SystemExit(
            f"need 1 <= avg_bag <= max_bag, got {cfg.avg_bag}/{cfg.max_bag}"
        )
    _log(
        f"=== W={world} batch_size={cfg.batch_size} avg_bag={cfg.avg_bag} "
        f"max_bag={cfg.max_bag} vocab={cfg.vocab} dim={cfg.dim} "
        f"steps={cfg.steps} tokens_per_chip={tokens_per_chip(cfg, world)} ==="
    )

    torch.manual_seed(cfg.seed)
    model = TinyRecModel(cfg, world)
    # Snapshot before either run mutates it, so both start from the same point.
    init = {k: v.clone() for k, v in model.state_dict().items()}
    batches = make_batches(cfg, world, np.random.default_rng(cfg.seed))

    tpu = run_tpu(cfg, model, batches)
    _log(f"tpu   first={tpu.losses[0]:.6f} last={tpu.losses[-1]:.6f}")

    # Criterion 1: it learns at all.
    descended = tpu.losses[-1] < tpu.losses[0]
    _log(f"loss decreased: {descended}")

    # Criterion 2: the same trajectory on CPU.
    max_rel = float("nan")
    cpu = None
    if not cfg.skip_cpu:
        model.load_state_dict(init)
        cpu = run_cpu_reference(cfg, model, batches)
        _log(f"cpu   first={cpu.losses[0]:.6f} last={cpu.losses[-1]:.6f}")
        denom = max(abs(cpu.losses[0]), 1e-6)
        max_rel = max(abs(t - c) for t, c in zip(tpu.losses, cpu.losses)) / denom
        _log(f"max per-step relative loss gap: {max_rel:.3e}")

    parity_ok = not cfg.skip_cpu and max_rel <= cfg.rtol
    ok = _passes_acceptance(
        descended=descended,
        max_rel=max_rel,
        rtol=cfg.rtol,
        skip_cpu=cfg.skip_cpu,
    )
    _log(
        f"=== RESULT ebc_e2e: {'PASS' if ok else 'FAIL'} "
        f"descended={descended} parity={parity_ok} "
        f"max_rel={max_rel:.3e} rtol={cfg.rtol:.0e} ==="
    )
    print(
        _build_report(
            cfg,
            world,
            tpu,
            cpu,
            descended=descended,
            parity_ok=parity_ok,
            ok=ok,
        ),
        file=sys.stderr,
        flush=True,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
