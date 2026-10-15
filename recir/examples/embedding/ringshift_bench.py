# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Standalone perf benchmark: TensorCore ring-shift vs SparseCore lookup (fwd).

Compares, at a matched ``(world_size, dim, per-rank query count, distribution)``
shape, three non-pooling sharded embedding-lookup implementations:

  * ``ra2a``  -- TensorCore ``lax.ragged_all_to_all`` baseline.
  * ``fused`` -- TensorCore fused ring-shift Pallas kernel
                 (``ringshift_embedding_lookup``).
  * ``sc``    -- the existing SparseCore solution (DLRMv3): the raw
                 ``@SparseDenseMatmulOp`` custom call emitted by
                 ``recir.embedding.sparsecore_custom_call.sc_matmul_fwd`` (NOT jte's
                 ``tpu_sparse_dense_matmul`` primitive), in the non-pooling
                 "sequence token trick" mode (valency=1, one id per output row),
                 distributed across all cores via ``shard_map`` so it triggers the
                 SparseCore all-to-all -- the same non-pooling distributed lookup
                 the ring-shift kernel performs. ``jax_tpu_embedding`` (jte) is
                 used ONLY for host-side CSR preprocessing + feature/table specs.

Select arms with ``--backend {tc,sc,all}`` (default ``all`` for ``--mode fwd``).
A single ``=== RESULT fwd: ra2a=… fused=… sc=… ===`` line carries the headline
medians. ``--mode bwd`` covers the TensorCore arms only (SC is forward-only
here).

Example (large-scale forward SC-vs-TC, 128 shards, multi-host JobSet)::

    python3 -m recir.examples.embedding.ringshift_bench \
        --mode fwd --backend all --world-size 128 --dim 256 --v-local 8192 \
        --max-local-ids 98304 --recv-capacity 8192 --distribution uniform

The TC mesh is built from --world-size devices, taken at --device-stride into
jax.devices() (1D ``"x"`` axis). The SC arm builds its own 1D mesh over ALL
jax.devices() (table row/mod-sharded across every core), so run the SC-vs-TC
comparison with ``--device-stride 1`` so both use the same device set.

The SparseCore arm needs ``jax_tpu_embedding`` (jte), installed in the embedding
image (see Dockerfile). jte is imported lazily, so the ``tc`` arm still runs in
an image without it.
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
    _POOLING_IMPL_ENV,
    ringshift_embedding_lookup,
    ringshift_embedding_lookup_reference,
    ringshift_embedding_lookup_reference_jax,
)

_TILE = 128

# Agreement bound between the two pooled reductions. They differ only in
# summation ORDER, so the gap is pure f32 rounding over bags of up to a few
# thousand rows; anything above this is a real bug, not accumulation noise.
_EBC_REL_TOL = 1e-4

# SparseCores per JAX device. jte's num_sparsecores_per_device() raises
# ValueError on device kinds it doesn't recognise (e.g. "TPU7x"); fall back to
# 4 (the value jte uses internally for TPU v5/v6/v7), matching the DLRMv3
# SparseCore backend (recir/embedding/sparsecore_ops.py).
_NUM_SC_PER_DEVICE_FALLBACK = 4

# jte CSR padding sentinel (INT32_MAX); matches sparsecore_custom_call._PAD_VALUE.
_SC_PAD_VALUE = 2147483647


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
    (balanced traffic; per-pair traffic ~ max_local_ids/world_size).
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


def _trace_region(tag, cfg, label, fn, args) -> None:
    """Capture a jax.profiler trace of ``fn`` over ``cfg.trace_iters`` iters.

    Writes to ``<trace_dir>/<label>`` (use a gs:// path so the trace survives the
    ephemeral pod FS). All processes enter the trace so the multi-host xplane is
    complete; the dir is identical across hosts (required for xprof to group it).
    No-op when ``--trace-dir`` is unset.
    """
    if not cfg.trace_dir:
        return
    sub = cfg.trace_dir.rstrip("/") + "/" + label
    _log(tag, f"tracing {label} -> {sub} ({cfg.trace_iters} iters)")
    sync_global_devices(f"trace_{label}_start")
    with jax.profiler.trace(sub):
        for _ in range(cfg.trace_iters):
            out = fn(*args)
            jax.block_until_ready(out)
    sync_global_devices(f"trace_{label}_done")


def run_fwd_tc(mesh, cfg, tag) -> tuple[float, float]:
    """TensorCore forward perf: fused ring-shift vs ra2a reference.

    Returns ``(ra2a_median_ms, fused_median_ms)``.
    """
    local_ids, local_id_count = _make_inputs(
        mesh,
        cfg.world_size,
        cfg.max_local_ids,
        cfg.v_local,
        cfg.seed,
        distribution=cfg.distribution,
    )
    local_table = _make_table(
        mesh, cfg.v_local, cfg.dim, cfg.seed, random=cfg.random_table
    )
    jax.block_until_ready((local_ids, local_id_count, local_table))
    args = (local_ids, local_id_count, local_table)
    _log(tag, "tc inputs ready")

    fused = jax.jit(
        functools.partial(
            ringshift_embedding_lookup,
            mesh=mesh,
            axis_name="x",
            recv_capacity=cfg.recv_capacity,
            splitter="globalpad",
            exchange_bf16=cfg.exchange_bf16,
        )
    )
    ref = jax.jit(
        functools.partial(
            ringshift_embedding_lookup_reference_jax, mesh=mesh, axis_name="x"
        )
    )

    ref_times = (
        None
        if cfg.skip_ra2a
        else _time_jit(tag, "ra2a_ref", ref, args, cfg.warmup, cfg.iters)
    )
    fused_times = _time_jit(tag, "fused", fused, args, cfg.warmup, cfg.iters)
    _trace_region(tag, cfg, "fused", fused, args)
    ra2a_median = None if ref_times is None else statistics.median(ref_times)
    # pyrefly: ignore [bad-return]
    return ra2a_median, statistics.median(fused_times)


def _make_bag_lengths(mesh, num_bags, local_id_count, seed, *, distribution):
    """Per-rank bag lengths that sum EXACTLY to this rank's valid id count.

    Derived from ``local_id_count`` rather than recomputed, so the two can not
    drift: the pooled API rejects a separately-supplied count precisely because
    ``bag_lengths`` already determines it.

    Lengths are jagged (and, under ``skewed``, heavy-tailed) so the benchmark
    exercises the case the reduction actually has to handle. Some bags land on
    zero, which is the empty-bag path.
    """
    P = jax.sharding.PartitionSpec
    heavy_tailed = distribution == "skewed"

    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(P("x"),),
        out_specs=P("x"),
        check_vma=False,
    )
    def make(count):
        rank = jax.lax.axis_index("x")
        key = jax.random.fold_in(jax.random.PRNGKey(seed + 1), rank)
        w = (
            jax.random.exponential(key, (num_bags,))
            if heavy_tailed
            else jax.random.uniform(key, (num_bags,))
        )
        w = w / jnp.sum(w)
        total = count[0].astype(jnp.int32)
        lengths = jnp.floor(w * total.astype(jnp.float32)).astype(jnp.int32)
        # Floor can only undershoot, so the remainder is non-negative and the
        # sum is exact -- which the op asserts on.
        lengths = lengths.at[-1].add(total - jnp.sum(lengths))
        return lengths[None, :]

    return make(local_id_count)


def run_ebc_tc(mesh, cfg, tag) -> tuple[float, float]:
    """Pooled (EBC) forward perf: XLA reduction vs in-kernel reduction.

    Returns ``(xla_median_ms, kernel_median_ms)``.

    Both environment-selected arms are traced in ONE process against the SAME
    inputs, then alternate through the same ``_time_jit`` path. Two separate
    jobs would differ in slice, neighbours and thermal state, and a 10%
    difference between implementations is well inside that noise.
    """
    local_ids, local_id_count = _make_inputs(
        mesh,
        cfg.world_size,
        cfg.max_local_ids,
        cfg.v_local,
        cfg.seed,
        distribution=cfg.distribution,
    )
    local_table = _make_table(
        mesh, cfg.v_local, cfg.dim, cfg.seed, random=cfg.random_table
    )
    bag_lengths = _make_bag_lengths(
        mesh, cfg.num_bags, local_id_count, cfg.seed, distribution=cfg.distribution
    )
    jax.block_until_ready((local_ids, local_table, bag_lengths))
    args = (local_ids, local_table, bag_lengths)
    _log(tag, f"ebc inputs ready (num_bags={cfg.num_bags} pooling={cfg.pooling})")

    def _arm(impl):
        def call(ids, table, bags):
            # This Python scope runs while JAX traces this arm. Restoring the
            # process environment afterwards cannot change the compiled graph.
            previous = os.environ.get(_POOLING_IMPL_ENV)
            os.environ[_POOLING_IMPL_ENV] = impl
            try:
                return ringshift_embedding_lookup(
                    ids,
                    None,  # derived from bag_lengths when pooling is on
                    table,
                    mesh=mesh,
                    axis_name="x",
                    recv_capacity=cfg.recv_capacity,
                    splitter="globalpad",
                    exchange_bf16=cfg.exchange_bf16,
                    pooling=cfg.pooling,
                    bag_lengths=bags,
                )
            finally:
                if previous is None:
                    os.environ.pop(_POOLING_IMPL_ENV, None)
                else:
                    os.environ[_POOLING_IMPL_ENV] = previous

        return jax.jit(call)

    xla, kernel = _arm("xla"), _arm("fused_kernel")

    # Agreement first: a timing for a wrong kernel is worse than no timing.
    #
    # NOT bitwise-equal, unlike the fused-vs-ra2a check elsewhere in this file.
    # Both arms sum the same rows, but the kernel accumulates in wave/sorted
    # order while XLA accumulates in caller order, and f32 addition is not
    # associative. Requiring equality here would fail for a correct kernel.
    out_xla = jax.block_until_ready(xla(*args))
    out_kernel = jax.block_until_ready(kernel(*args))

    # Independent oracle, at the SAME scale the timings are taken at.
    #
    # The arms-agree check below is relative: it cannot catch a mistake both
    # make, and the likeliest such mistake is one inherited from the shared
    # host-side prep (segment ids, sort permutation, per-peer padding). The
    # numpy reference shares no code with either arm -- it is a double loop
    # over (rank, id), no segment_sum, no kernel, no permutation.
    #
    # Cheap enough to leave on: a Python loop over W x max_local_ids ids costs
    # seconds against a job that already waited in a queue. It is skipped on
    # multi-host, where the oracle would need the whole global table gathered
    # onto one process.
    if cfg.check_reference and jax.process_count() == 1:
        want = ringshift_embedding_lookup_reference(
            np.asarray(jax.device_get(local_ids)),
            None,
            np.asarray(jax.device_get(local_table)),
            cfg.world_size,
            cfg.pooling,
            np.asarray(jax.device_get(bag_lengths)),
        )
        want64 = np.asarray(want, dtype=np.float64)
        oracle_scale = max(float(np.max(np.abs(want64))), 1e-12)
        for label, got in (("xla", out_xla), ("kernel", out_kernel)):
            got64 = np.asarray(jax.device_get(got), dtype=np.float64)
            ref_rel = float(np.max(np.abs(got64 - want64)) / oracle_scale)
            _log(tag, f"ebc vs oracle [{label}]: max_rel_diff={ref_rel:.3e}")
            if ref_rel > _EBC_REL_TOL:
                raise SystemExit(
                    f"[{tag}] {label} disagrees with the numpy oracle: "
                    f"{ref_rel:.3e} > {_EBC_REL_TOL:.0e}"
                )
    elif cfg.check_reference:
        _log(tag, "ebc vs oracle: skipped (multi-host; oracle needs global table)")
    scale = jnp.maximum(jnp.max(jnp.abs(out_xla)), 1e-6)
    rel = float(jnp.max(jnp.abs(out_kernel - out_xla)) / scale)
    _log(tag, f"ebc agreement: max_rel_diff={rel:.3e}")
    if rel > _EBC_REL_TOL:
        raise SystemExit(
            f"[{tag}] pooled arms disagree: max_rel_diff={rel:.3e} > "
            f"{_EBC_REL_TOL:.0e} -- refusing to report timings"
        )

    xla_times = _time_jit(tag, "ebc_xla", xla, args, cfg.warmup, cfg.iters)
    kernel_times = _time_jit(tag, "ebc_kernel", kernel, args, cfg.warmup, cfg.iters)
    _trace_region(tag, cfg, "ebc_kernel", kernel, args)
    return statistics.median(xla_times), statistics.median(kernel_times)


# ---------------------------------------------------------------------------
# SparseCore arm (the existing DLRMv3 solution).
#
# Compute = recir.embedding.sparsecore_custom_call.sc_matmul_fwd (the raw
# @SparseDenseMatmulOp custom call), distributed via shard_map. jte is used ONLY
# for host CSR preprocessing + specs + SC-laid-out table init. Both sc_matmul_fwd
# and jte are jax-only (no torch/torchrec) and imported lazily, so the TC arms
# still run in an image without them.
# ---------------------------------------------------------------------------


def _sc_lazy_jte():
    """Lazily import jax_tpu_embedding (pip-only; embedding image installs it)."""
    # pyre-fixme[21]: jax_tpu_embedding is a runtime-only (pip) dependency.
    from jax_tpu_embedding.sparsecore.lib.nn import (
        embedding as jte_embedding,
        embedding_spec as jte_spec,
        table_stacking as jte_stacking,
    )

    # pyre-fixme[21]: jax_tpu_embedding is a runtime-only (pip) dependency.
    from jax_tpu_embedding.sparsecore.utils import utils as jte_utils

    return jte_embedding, jte_spec, jte_stacking, jte_utils


def _sc_num_sc_per_device(jte_utils) -> int:
    try:
        return jte_utils.num_sparsecores_per_device()
    except ValueError as e:
        _log(
            "SC",
            f"num_sparsecores_per_device() failed ({e}); "
            f"defaulting to {_NUM_SC_PER_DEVICE_FALLBACK}",
        )
        return _NUM_SC_PER_DEVICE_FALLBACK


def _sc_make_ids(cfg, local_bs: int) -> np.ndarray:
    """Host-side per-process query ids in jte's (local_bs, 1) non-pooling format.

    Reproduces ``_make_inputs`` semantics on the host (jte's CSR preprocessing is
    host-side): each id = ``owner + world_size * row`` in ``[0, v_local*W)``, with
    ``owner`` uniform (uniform) or 75%-hot-peer (skewed). Padding rows look up
    global id 0.
    """
    rng = np.random.RandomState(cfg.seed)
    uniform = cfg.distribution == "uniform"
    valid = local_bs if uniform else (local_bs * 4) // 5
    random_owner = rng.randint(0, cfg.world_size, size=valid)
    if uniform:
        owner = random_owner
    else:
        hot_peer = 1 % cfg.world_size
        is_hot = rng.random_sample(valid) < 0.75
        owner = np.where(is_hot, hot_peer, random_owner)
    row = rng.randint(0, cfg.v_local, size=valid)
    ids = (owner + cfg.world_size * row).astype(np.int32)
    ids_padded = np.zeros(local_bs, dtype=np.int32)
    ids_padded[:valid] = ids
    return ids_padded.reshape(local_bs, 1)


def _sc_init_emb_vars(jte_embedding, feature_specs, num_sc, global_sharding):
    """Initialize SparseCore-laid-out tables (mirrors DLRMv3
    init_jte_sparsecore_embedding_variables)."""
    import collections

    table_specs = jte_embedding.get_table_specs(feature_specs)
    stacks: dict = collections.defaultdict(list)
    for tspec in jax.tree.leaves(table_specs):
        stacks[tspec.setting_in_stack.stack_name].append(tspec)
    for ts in stacks.values():
        ts.sort(key=lambda t: t.setting_in_stack.row_offset_in_shard)

    sharding_axis = next(s for s in global_sharding.spec if s is not None)
    per_table_width = global_sharding.mesh.size * num_sc
    rngs = jax.random.split(jax.random.PRNGKey(0), per_table_width * len(stacks))
    emb_vars: dict = {}
    for i, (stack_name, ts) in enumerate(stacks.items()):
        rngs_for_table = rngs[i * per_table_width : (i + 1) * per_table_width]
        emb_vars[stack_name] = jte_embedding._init_stacked_embedding_table(
            rng=rngs_for_table,
            stack_name=stack_name,
            table_specs=ts,
            global_sharding=global_sharding,
            sharding_axis=sharding_axis,
            num_sparsecore_per_device=num_sc,
        )
    return emb_vars


def _sc_make_global_csr(csr_inputs, mesh):
    """Lift a per-process jte CSR pytree to globally-sharded arrays + in_specs.

    Mirrors scripts/yuzhenhuang/tpu/dlrm/embedding_sparsecore.py:make_global_csr:
    rank>=1 arrays shard on the data axis (P("data")); scalars replicate (P()).
    The returned PartitionSpec pytree feeds shard_map's in_specs so each device
    sees its own CSR slice.
    """
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
    repl_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    def _shard(x):
        arr = np.asarray(x)
        sharding = data_sharding if arr.ndim > 0 else repl_sharding
        return jax.make_array_from_process_local_data(sharding, arr)

    def _spec(x):
        return (
            jax.sharding.PartitionSpec("data")
            if np.asarray(x).ndim > 0
            else jax.sharding.PartitionSpec()
        )

    return jax.tree.map(_shard, csr_inputs), jax.tree.map(_spec, csr_inputs)


def run_sc_fwd(cfg, tag) -> float:
    """SparseCore forward perf via the DLRMv3 emission (``sc_matmul_fwd``).

    Compute uses ``recir.embedding.sparsecore_custom_call.sc_matmul_fwd`` (the raw
    ``@SparseDenseMatmulOp`` custom call) -- NOT jte's ``tpu_sparse_dense_matmul``
    primitive. jte is used only for host-side CSR preprocessing + feature/table
    specs + SC-laid-out table init.

    Distributed: the table is row/mod-sharded across all cores and the matmul
    runs inside a ``shard_map`` over the device mesh (mirroring the SC backward
    in embedding_sparsecore.py:build_sc_backward), so the in-kernel SparseCore
    all-to-all spans the mesh -- the distributed non-pooling lookup comparable to
    ring-shift.

    Returns the device-forward median (ms); host CSR preprocessing is logged
    separately so the SC-vs-TC device comparison is apples-to-apples.

    NOTE: distributed ``sc_matmul_fwd`` has no existing reference (the torchtpu
    path is single-device; the distributed reference uses jte's primitive), so
    this path must be validated on TPU. The CSR sharding + per-device batch
    contract is grounded in embedding_sparsecore.py (make_global_csr /
    build_sc_backward) and embedding_torchtpu.py (_stack_kernel_params).
    """
    jte_embedding, jte_spec, jte_stacking, jte_utils = _sc_lazy_jte()
    # sparsecore_custom_call is jax-only (no torch/torchrec); lazy-import keeps the tc
    # arm + buck build clean. Resolved at runtime via the editable recir install.
    from recir.embedding.sparsecore_custom_call import sc_matmul_fwd

    num_sc = _sc_num_sc_per_device(jte_utils)
    global_dev = jax.device_count()
    local_dev = jax.local_device_count()
    per_device_bs = cfg.max_local_ids
    per_process_bs = local_dev * per_device_bs
    global_bs = global_dev * per_device_bs

    if per_device_bs % num_sc != 0:
        raise SystemExit(
            f"SC requires per-device batch (max_local_ids={per_device_bs}) be a "
            f"multiple of num_sc_per_device ({num_sc})"
        )

    vocab = cfg.v_local * cfg.world_size
    table_spec = jte_spec.TableSpec(
        vocabulary_size=vocab,
        embedding_dim=cfg.dim,
        initializer=jax.nn.initializers.truncated_normal(stddev=1.0 / cfg.dim**0.5),
        optimizer=jte_spec.AdagradOptimizerSpec(
            learning_rate=cfg.sc_lr, initial_accumulator_value=0.0
        ),
        combiner="sum",  # no-op for 1-id samples (non-pooling sequence trick)
        name="t0",
    )
    feature_specs = {
        "f0": jte_spec.FeatureSpec(
            name="f0",
            table_spec=table_spec,
            input_shape=(global_bs, 1),
            output_shape=(global_bs, cfg.dim),
        )
    }
    jte_stacking.stack_tables(
        feature_specs,
        table_names=["t0"],
        global_device_count=global_dev,
        num_sc_per_device=num_sc,
        stack_name="t0",
    )

    # Table row/mod-sharded across ALL cores (1D "data" mesh over all devices).
    mesh = jax.sharding.Mesh(np.array(jax.devices()), ("data",))
    global_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec("data", None)
    )
    emb_vars = _sc_init_emb_vars(jte_embedding, feature_specs, num_sc, global_sharding)
    table = emb_vars["t0"].table

    # Static per-stack kernel params from jte (mirror embedding_torchtpu).
    stacked = jte_embedding.get_stacked_table_specs(feature_specs)["t0"]
    device_batch_size = stacked.total_sample_count // global_dev
    max_ids = stacked.max_ids_per_partition
    max_unique_ids = stacked.max_unique_ids_per_partition

    # Host CSR preprocessing (per-process features; timed separately).
    ids_np = _sc_make_ids(cfg, per_process_bs)
    t0 = time.perf_counter()
    csr_inputs, _ = jte_embedding.preprocess_sparse_dense_matmul_input(
        features={"f0": ids_np},
        features_weights=None,
        feature_specs=feature_specs,
        local_device_count=local_dev,
        global_device_count=global_dev,
        num_sc_per_device=num_sc,
        sharding_strategy="MOD",
        allow_id_dropping=cfg.sc_allow_id_dropping,
    )
    csr_global, csr_in_specs = _sc_make_global_csr(csr_inputs, mesh)
    _log(tag, f"sc CSR preprocess: {(time.perf_counter() - t0) * 1e3:.1f}ms (host)")

    P = jax.sharding.PartitionSpec

    @jax.jit
    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(csr_in_specs, P("data", None)),
        out_specs=P("data", None),
        check_vma=False,
    )
    def _sc_fwd(csr_local, table_local):
        # Per-device CSR slice (jte preprocessed-input pytree) -> raw
        # @SparseDenseMatmulOp; the in-kernel all-to-all spans the "data" axis.
        return sc_matmul_fwd(
            csr_local.lhs_row_pointers["t0"],
            csr_local.lhs_embedding_ids["t0"],
            csr_local.lhs_sample_ids["t0"],
            csr_local.lhs_gains["t0"],
            table_local,
            device_batch_size=device_batch_size,
            max_ids=max_ids,
            max_unique_ids=max_unique_ids,
            sharding_strategy=1,  # MOD
        )

    sc_times = _time_jit(tag, "sc", _sc_fwd, (csr_global, table), cfg.warmup, cfg.iters)
    _trace_region(tag, cfg, "sc", _sc_fwd, (csr_global, table))
    return statistics.median(sc_times)


def run_bwd(mesh, cfg, tag) -> None:
    """Forward + backward perf (TensorCore arms only): time fwd-only and fwd+bwd
    (value_and_grad) for both the fused kernel and the ra2a reference; backward
    cost is the fwd+bwd - fwd difference."""
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
    p.add_argument("--mode", choices=("fwd", "bwd", "ebc"), default="fwd")
    p.add_argument(
        "--num-bags",
        type=int,
        default=1024,
        help="pooled (--mode ebc) bags per rank = local batch x features",
    )
    p.add_argument(
        "--pooling", choices=("sum", "mean"), default="sum", help="--mode ebc only"
    )
    p.add_argument(
        "--no-check-reference",
        dest="check_reference",
        action="store_false",
        default=True,
        help="skip the --mode ebc numpy-oracle check (single-host only anyway)",
    )
    p.add_argument(
        "--backend",
        choices=("tc", "sc", "all"),
        default="all",
        help="which arms to run in fwd mode: tc (ra2a+ring-shift), sc "
        "(SparseCore), or all. bwd mode is TensorCore-only.",
    )
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
        "(e.g. 8 = 1 device/host across 32 hosts, mirroring devices[::8]). "
        "Use 1 for SC-vs-TC so both arms use the same device set.",
    )
    p.add_argument("--iters", type=int, default=10, help="measured iterations")
    p.add_argument("--warmup", type=int, default=5, help="warmup iterations")
    p.add_argument("--seed", type=int, default=2)
    p.add_argument(
        "--sc-lr",
        type=float,
        default=0.01,
        help="SparseCore Adagrad learning rate (unused in fwd-only timing)",
    )
    p.add_argument(
        "--sc-allow-id-dropping",
        action="store_true",
        default=True,
        help="let jte drop ids that overflow max_ids_per_partition (perf bench)",
    )
    p.add_argument(
        "--random-table",
        action="store_true",
        default=False,
        help="build the fwd table with real random values instead of zeros. A "
        "zeros table lets XLA/Mosaic make the gather/prefetch ~free, so fwd-only "
        "timing on a zeros table is unrealistic; use this to measure a real gather.",
    )
    p.add_argument(
        "--skip-ra2a",
        action="store_true",
        default=False,
        help="skip the ra2a reference arm (its recv buffer is W*max_local_ids "
        "and OOMs at large max_local_ids); times only fused (+sc with backend all).",
    )
    p.add_argument(
        "--exchange-bf16",
        action="store_true",
        default=False,
        help="ring-shift fused arm: exchange the embedding all-to-all in bf16 "
        "(halves cross-host ICI bytes).",
    )
    p.add_argument(
        "--trace-dir",
        type=str,
        default="",
        help="if set, capture jax.profiler traces of the fused + sc arms to "
        "<trace-dir>/{fused,sc}. Use a gs:// path so it survives the pod FS "
        "(view in xprof: xprof --logdir <trace-dir>/fused).",
    )
    p.add_argument(
        "--trace-iters",
        type=int,
        default=3,
        help="iterations captured per traced arm",
    )
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
        f"distribution={cfg.distribution} backend={cfg.backend} ===",
    )

    if cfg.mode == "ebc":
        xla_median, kernel_median = run_ebc_tc(mesh, cfg, tag)
        speedup = xla_median / kernel_median if kernel_median else float("nan")
        valid_ids_per_rank = (
            cfg.max_local_ids
            if cfg.distribution == "uniform"
            else (cfg.max_local_ids * 4) // 5
        )
        average_bag_len = (
            valid_ids_per_rank / cfg.num_bags if cfg.num_bags else float("nan")
        )
        _log(
            tag,
            # 3 decimals: at sub-millisecond medians `.1f` rounds 0.700 and
            # 0.634 to "0.7" and "0.6", which reads as a 17% gap when it is 10%.
            f"=== RESULT ebc: world_size={world_size} dim={cfg.dim} "
            f"num_bags={cfg.num_bags} bag_len={average_bag_len:.3f} "
            f"distribution={cfg.distribution} pooling={cfg.pooling} "
            f"v_local={cfg.v_local} max_local_ids={cfg.max_local_ids} "
            f"xla={xla_median:.3f}ms "
            f"kernel={kernel_median:.3f}ms speedup={speedup:.3f}x ===",
        )
        return

    if cfg.mode == "bwd":
        # SparseCore backward is out of scope here; bwd is TensorCore-only.
        if cfg.backend == "sc":
            _log(tag, "bwd mode is TensorCore-only; ignoring --backend sc")
        run_bwd(mesh, cfg, tag)
        return

    ra2a_median = fused_median = sc_median = None
    if cfg.backend in ("tc", "all"):
        ra2a_median, fused_median = run_fwd_tc(mesh, cfg, tag)
    if cfg.backend in ("sc", "all"):
        sc_median = run_sc_fwd(cfg, tag)

    parts = []
    if ra2a_median is not None:
        parts.append(f"ra2a={ra2a_median:.1f}ms")
    if fused_median is not None:
        parts.append(f"fused={fused_median:.1f}ms")
    if sc_median is not None:
        parts.append(f"sc={sc_median:.1f}ms")
    _log(tag, f"=== RESULT fwd: {' '.join(parts)} ===")


if __name__ == "__main__":
    main()
