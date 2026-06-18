# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Standalone perf benchmark for ``ringshift_embedding_lookup`` (fwd / fwd+bwd).

A general, scenario-parameterized runner driven entirely by argparse flags --
configure the embedding dim, per-shard vocab, per-rank query fan-in, recv
buffer, world size, and id distribution, then measure the fused ring-shift
kernel against the ``lax.ragged_all_to_all`` reference. Numerical CORRECTNESS still
lives in ``ringshift_fwd.py`` / ``ringshift_bwd.py`` as ``unittest`` classes.

Example (rankfm-scale forward, 128 shards, multi-host JobSet)::

    python3 -m recir.examples.embedding.ringshift_bench \
        --mode fwd --world-size 128 --dim 256 --v-local 8192 \
        --max-local-ids 98304 --recv-capacity 8192 --distribution uniform

The mesh is built from --world-size devices, taken at --device-stride into
jax.devices() (1D ``"x"`` axis). On a TPU JobSet jax.distributed.initialize() is
auto-detected from the TPU runtime env and runs before jax.devices(); on CPU/dev
it is skipped.
"""

from __future__ import annotations

import argparse
import functools
import os
import statistics
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.multihost_utils import sync_global_devices
from recir.ops.pallas.embeddings.ringshift import (
    ringshift_embedding_lookup,
    ringshift_embedding_lookup_reference_jax,
)

_TILE = 128


def _log(tag: str, msg: str) -> None:
    if jax.process_index() == 0:
        print(f"[{tag}] {msg}", file=sys.stderr, flush=True)


def _round_up_tile(x: int) -> int:
    return ((x + _TILE - 1) // _TILE) * _TILE


# ---------------------------------------------------------------------------
# Per-rank input generators (built on-device via shard_map).
# ---------------------------------------------------------------------------


def _make_inputs(mesh, world_size, max_local_ids, v_local, seed, *, distribution):
    """Per-rank padded query ids + valid count.

    ``distribution="uniform"``: each id's owner ~ uniform over peers
    (rankfm-representative; per-pair traffic ~ max_local_ids/world_size).
    ``distribution="skewed"``: ~75% of ids target one hot peer (stress pattern;
    a single per-pair count approaches max_local_ids).
    """
    P = jax.sharding.PartitionSpec
    uniform = distribution == "uniform"

    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(),
        out_specs=(P("x"), P("x")),
        check_vma=False,
    )
    def make():
        rank = jax.lax.axis_index("x")
        key = jax.random.fold_in(jax.random.PRNGKey(seed), rank)
        valid = max_local_ids if uniform else (max_local_ids * 4) // 5
        u_hot, u_owner, u_row = jax.random.split(key, 3)
        random_owner = jax.random.randint(u_owner, (valid,), 0, world_size)
        if uniform:
            owner = random_owner
        else:
            hot_peer = (rank + 1) % world_size
            is_hot = jax.random.uniform(u_hot, (valid,)) < 0.75
            owner = jnp.where(is_hot, hot_peer, random_owner)
        row = jax.random.randint(u_row, (valid,), 0, v_local)
        ids = (owner + world_size * row).astype(jnp.int32)
        ids_padded = jnp.zeros(max_local_ids, dtype=jnp.int32).at[:valid].set(ids)
        return ids_padded[None, :], jnp.array(valid, dtype=jnp.int32)[None]

    return make()


def _make_table(mesh, v_local, embedding_dim, seed, *, random):
    """Per-rank table shard. ``random=False`` -> zeros (content irrelevant for a
    forward-only perf measurement); ``random=True`` -> per-rank normal (needed
    so backward gradients depend on real values)."""
    P = jax.sharding.PartitionSpec

    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(),
        out_specs=P("x"),
        check_vma=False,
    )
    def make():
        if not random:
            return jnp.zeros((1, v_local, embedding_dim), dtype=jnp.float32)
        rank = jax.lax.axis_index("x")
        key = jax.random.fold_in(jax.random.PRNGKey(seed), rank)
        return jax.random.normal(key, (1, v_local, embedding_dim), dtype=jnp.float32)

    return make()


def _make_target(mesh, max_local_ids, embedding_dim, seed):
    """Per-rank random fp32 target (the downstream consumer for the bwd loss)."""
    P = jax.sharding.PartitionSpec

    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(),
        out_specs=P("x"),
        check_vma=False,
    )
    def make():
        rank = jax.lax.axis_index("x")
        key = jax.random.fold_in(jax.random.PRNGKey(seed), rank)
        return jax.random.normal(
            key, (1, max_local_ids, embedding_dim), dtype=jnp.float32
        )

    return make()


# ---------------------------------------------------------------------------
# Benchmark drivers.
# ---------------------------------------------------------------------------


def _time_jit(tag, label, fn, args, n_warmup, n_measure):
    _log(tag, f"-- {label}: warmup {n_warmup}, measure {n_measure} --")
    for i in range(n_warmup):
        sync_global_devices(f"{label}_warmup_{i}")
        out = fn(*args)
        jax.block_until_ready(out)
    sync_global_devices(f"{label}_warmup_done")
    times_ms = []
    for i in range(n_measure):
        sync_global_devices(f"{label}_meas_{i}")
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        times_ms.append((time.perf_counter() - t0) * 1e3)
    sync_global_devices(f"{label}_meas_done")
    _log(tag, f"{label}: median={statistics.median(times_ms):.1f}ms times={times_ms}")
    return times_ms


def run_fwd(mesh, cfg, tag) -> None:
    """Forward-only perf: fused ring-shift vs ra2a reference."""
    local_ids, local_id_count = _make_inputs(
        mesh,
        cfg.world_size,
        cfg.max_local_ids,
        cfg.v_local,
        cfg.seed,
        distribution=cfg.distribution,
    )
    local_table = _make_table(mesh, cfg.v_local, cfg.dim, cfg.seed, random=False)
    jax.block_until_ready((local_ids, local_id_count, local_table))
    args = (local_ids, local_id_count, local_table)
    _log(tag, "inputs ready")

    fused = jax.jit(
        functools.partial(
            ringshift_embedding_lookup,
            mesh=mesh,
            axis_name="x",
            recv_capacity=cfg.recv_capacity,
            splitter="globalpad",
        )
    )
    ref = jax.jit(
        functools.partial(
            ringshift_embedding_lookup_reference_jax, mesh=mesh, axis_name="x"
        )
    )

    ref_times = _time_jit(tag, "ra2a_ref", ref, args, cfg.warmup, cfg.iters)
    fused_times = _time_jit(tag, "fused", fused, args, cfg.warmup, cfg.iters)
    _log(
        tag,
        f"=== RESULT fwd: ref_median={statistics.median(ref_times):.1f}ms "
        f"fused_median={statistics.median(fused_times):.1f}ms ===",
    )


def run_bwd(mesh, cfg, tag) -> None:
    """Forward + backward perf: time fwd-only and fwd+bwd (value_and_grad) for
    both the fused kernel and the ra2a reference; backward cost is the
    fwd+bwd - fwd difference."""
    local_ids, local_id_count = _make_inputs(
        mesh,
        cfg.world_size,
        cfg.max_local_ids,
        cfg.v_local,
        cfg.seed,
        distribution=cfg.distribution,
    )
    local_table = _make_table(mesh, cfg.v_local, cfg.dim, seed=7, random=True)
    target = _make_target(mesh, cfg.max_local_ids, cfg.dim, seed=13)
    jax.block_until_ready((local_ids, local_id_count, local_table, target))
    args = (local_ids, local_id_count, local_table, target)
    _log(tag, "inputs ready")

    def fused_loss(local_ids, local_id_count, local_table, target):
        out = ringshift_embedding_lookup(
            local_ids,
            local_id_count,
            local_table,
            mesh=mesh,
            axis_name="x",
            recv_capacity=cfg.recv_capacity,
            splitter="globalpad",
        )
        return jnp.sum(out * target)

    def ref_loss(local_ids, local_id_count, local_table, target):
        out = ringshift_embedding_lookup_reference_jax(
            local_ids, local_id_count, local_table, mesh=mesh, axis_name="x"
        )
        return jnp.sum(out * target)

    fwd_fused = jax.jit(fused_loss)
    fwd_ref = jax.jit(ref_loss)
    fb_fused = jax.jit(jax.value_and_grad(fused_loss, argnums=2))
    fb_ref = jax.jit(jax.value_and_grad(ref_loss, argnums=2))

    fwd_fused_t = _time_jit(tag, "fwd_fused", fwd_fused, args, cfg.warmup, cfg.iters)
    fwd_ref_t = _time_jit(tag, "fwd_ref", fwd_ref, args, cfg.warmup, cfg.iters)
    fb_fused_t = _time_jit(tag, "fb_fused", fb_fused, args, cfg.warmup, cfg.iters)
    fb_ref_t = _time_jit(tag, "fb_ref", fb_ref, args, cfg.warmup, cfg.iters)

    bwd_fused = max(0.0, statistics.median(fb_fused_t) - statistics.median(fwd_fused_t))
    bwd_ref = max(0.0, statistics.median(fb_ref_t) - statistics.median(fwd_ref_t))
    _log(
        tag,
        f"=== RESULT fwd+bwd: bwd_fused~={bwd_fused:.1f}ms bwd_ref~={bwd_ref:.1f}ms ===",
    )


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("fwd", "bwd"), default="fwd")
    p.add_argument(
        "--world-size",
        type=int,
        default=0,
        help="shard count / mesh size (0 = all visible devices)",
    )
    p.add_argument("--dim", type=int, default=256, help="embedding dim D")
    p.add_argument(
        "--v-local",
        type=int,
        default=8192,
        help="per-shard table rows (V_global=V_local*W)",
    )
    p.add_argument(
        "--max-local-ids", type=int, default=8192, help="per-rank padded query count"
    )
    p.add_argument(
        "--recv-capacity",
        type=int,
        default=0,
        help="per-wave VMEM recv buffer (0 = max_local_ids + TILE)",
    )
    p.add_argument("--distribution", choices=("uniform", "skewed"), default="uniform")
    p.add_argument(
        "--device-stride",
        type=int,
        default=1,
        help="stride into jax.devices() before taking --world-size of them "
        "(e.g. 8 = 1 device/host across 32 hosts, mirroring devices[::8])",
    )
    p.add_argument("--iters", type=int, default=10, help="measured iterations")
    p.add_argument("--warmup", type=int, default=5, help="warmup iterations")
    p.add_argument("--seed", type=int, default=2)
    return p.parse_args(argv)


def main(argv=None) -> None:
    cfg = _parse_args(argv)
    # Auto-init the global TPU mesh on a TPU JobSet (see ringshift_fwd.py);
    # no-op on CPU/dev. Must run before jax.devices().
    if not jax.distributed.is_initialized() and (
        os.environ.get("TPU_WORKER_HOSTNAMES")
        or os.environ.get("MEGASCALE_COORDINATOR_ADDRESS")
    ):
        jax.distributed.initialize()

    all_devices = jax.devices()
    strided = all_devices[:: cfg.device_stride]
    world_size = cfg.world_size or len(strided)
    if len(strided) < world_size:
        raise SystemExit(
            f"need {world_size} devices at stride {cfg.device_stride}, have "
            f"{len(strided)} (of {len(all_devices)} total)"
        )
    cfg.world_size = world_size
    cfg.recv_capacity = _round_up_tile(cfg.recv_capacity or (cfg.max_local_ids + _TILE))

    devices = strided[:world_size]
    mesh = jax.sharding.Mesh(np.array(devices), ("x",))
    tag = (
        f"BENCH_{cfg.mode.upper()}_W{world_size}_D{cfg.dim}"
        f"_Q{cfg.max_local_ids}_{cfg.distribution}"
    )
    _log(
        tag,
        f"=== W={world_size} D={cfg.dim} v_local={cfg.v_local} "
        f"max_local_ids={cfg.max_local_ids} recv_capacity={cfg.recv_capacity} "
        f"distribution={cfg.distribution} ===",
    )

    if cfg.mode == "fwd":
        run_fwd(mesh, cfg, tag)
    else:
        run_bwd(mesh, cfg, tag)


if __name__ == "__main__":
    main()
