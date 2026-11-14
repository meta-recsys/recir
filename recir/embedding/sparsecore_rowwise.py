# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""SparseCore lookup with exact row-wise Adagrad, for the in-graph backend.

Each table lives in JAX TPU Embedding (JTE)'s SparseCore layout. Per step:

  forward:  each device looks up its unique ids, spread evenly over its
            SparseCores, and expands the result to every position.
  backward: each device sums its gradients per row into a fixed-size send buffer
            per owner (the slots come from the host), sends them to the rows'
            owners, and a SparseCore kernel applies row-wise Adagrad there.

Ids reach JTE in its sparse (COO) input form: JTE counts every dense entry,
weight 0 included, against max_ids_per_partition, so repeated ids and padding
would let one hot id overflow a partition. A sample with no entry costs nothing.

JTE ids are int32, so a table with more rows is split into contiguous-range
sub-tables; an id is looked up only in its own sub-table.

The tables and the momentum stay in SparseCore layout across steps: every program
donates them and returns them in the layout they arrived in.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import repeat
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P
from recir.ops.pallas.embeddings.sparsecore_rowwise_adagrad import (
    fp16_rounding_key,
    rowwise_adagrad_update,
)

logger: logging.Logger = logging.getLogger(__name__)

# Leaves room for JTE's padding of every table to a multiple of 8 x #SparseCores.
# RECIR_SPARSECORE_SUB_TABLE_ROWS lowers it, to exercise sub-tables on small tables.
_SUB_TABLE_MAX_ROWS: int = 2**31 - 2**24
# SparseCore's per-tensor HBM limit; JTE stores every table in FP32.
_SPARSECORE_TENSOR_BYTES: int = 16 * 2**30


def _jte() -> Tuple[Any, Any, Any, Any]:
    # jax_tpu_embedding is pip-only (the TPU runtime image), not in Buck third-party.
    # pyrefly: ignore[missing-import]
    from jax_tpu_embedding.sparsecore.lib.nn import (
        embedding as jte_embedding,
        embedding_spec as jte_spec,
        table_stacking as jte_stacking,
    )

    # pyrefly: ignore[missing-import]
    from jax_tpu_embedding.sparsecore.utils import utils as jte_utils

    return jte_embedding, jte_spec, jte_stacking, jte_utils


@dataclass(frozen=True)
class SparseCoreTable:
    name: str
    num_embeddings: int
    embedding_dim: int
    feature_names: Tuple[str, ...]
    init_bound: float
    # Trained as FP16: weights hold FP16 values (stochastically rounded updates).
    fp16: bool = False


@dataclass(frozen=True)
class _SubTable:
    stack: str
    table: str
    start: int
    rows: int


def _round_up(x: int, m: int) -> int:
    return -(-x // m) * m


def _split_table(
    t: SparseCoreTable, max_rows: int, world: int, num_sc: int
) -> List[_SubTable]:
    """Contiguous id ranges of at most ``max_rows`` whose per-device piece, with
    JTE's padding to a multiple of 8 x #SparseCores, fits one SparseCore tensor."""
    device_rows = _SPARSECORE_TENSOR_BYTES // (4 * t.embedding_dim) - 8 * num_sc
    n_sub = -(-t.num_embeddings // min(max_rows, world * device_rows))
    rows = -(-t.num_embeddings // n_sub)
    return [
        _SubTable(
            stack=f"{t.name}@{k}",
            table=t.name,
            start=k * rows,
            rows=min(rows, t.num_embeddings - k * rows),
        )
        for k in range(n_sub)
    ]


def _zeros_f32(key: Any, shape: Any, dtype: Any = None) -> Any:
    return jnp.zeros(shape, jnp.float32)


def _int32(a: np.ndarray) -> np.ndarray:
    return a.astype(np.int32) if a.dtype == np.int64 else a


def _device_inputs(
    ids: np.ndarray,
    d: int,
    padded: int,
    num_sc: int,
    subs: Sequence[_SubTable],
    slots: np.ndarray,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
    """Device ``d``'s ids -> (per sub-table, its COO (samples, sub-table ids);
    the sorted unique ids; each position's index into them).

    Fills row ``d`` of ``slots`` (each position's sample). Unique id k goes to sample
    (k % num_sc) * per_sc + k // num_sc, so the unique ids spread evenly over the
    device's SparseCores (JTE gives each SparseCore a contiguous per_sc samples)
    and samples come out increasing.
    """
    per_sc = padded // num_sc
    sorted_uniq, inv = np.unique(ids, return_inverse=True)
    k = np.arange(_round_up(sorted_uniq.size, num_sc)).reshape(-1, num_sc).T.ravel()
    k = k[k < sorted_uniq.size]
    uniq, samples = sorted_uniq[k], (k % num_sc) * per_sc + k // num_sc
    slots[d] = (inv % num_sc) * per_sc + inv // num_sc
    out = []
    for sub in subs:
        inside = (uniq >= sub.start) & (uniq < sub.start + sub.rows)
        s, v = samples[inside], (uniq[inside] - sub.start).astype(np.int32)
        out.append((s + d * padded, v))
    return out, sorted_uniq, inv


def _exchange_layout(
    parts: Sequence[Tuple[np.ndarray, np.ndarray, int, _SubTable]],
    layout: Tuple[int, int, int],
    world: int,
    num_sc: int,
    cap: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """One device's exchange layout for one stack: (each position's send-buffer
    slot, world * cap meaning none; the owner-local rows to send, [world, cap] with
    -1 padding; the fullest owner's fill of its cap).

    ``parts``: per feature of the stack, in stack order, (sorted unique ids, each
    position's index into them, positions padded to JTE's per-device size, the
    stack's sub-table). Positions of one row, across features too, share a slot.
    Raises before building anything if a buffer would overflow.
    """
    shard_size, rotation, offset = layout
    shards, per_device = world * num_sc, shard_size * num_sc
    cands, ranks = [], []
    for sorted_uniq, _, _, sub in parts:
        inside = (sorted_uniq >= sub.start) & (sorted_uniq < sub.start + sub.rows)
        cands.append(sorted_uniq[inside] - sub.start)
        ranks.append(np.where(inside, np.cumsum(inside) - 1, -1))
    ids = np.concatenate(cands)
    phys = ((ids % shards + rotation) % shards) * shard_size + ids // shards + offset
    rows, row_of = np.unique(phys, return_inverse=True)
    owner = rows // per_device
    counts = np.bincount(owner, minlength=world)
    fill = float(counts.max(initial=0)) / cap
    if fill > 1.0:
        raise RuntimeError(
            f"[sparsecore] a device sends {int(counts.max())} unique rows to one owner, "
            f"more than the exchange buffer's {cap}; raise RECIR_SPARSECORE_EXCHANGE_CAP_SCALE"
        )
    j = np.arange(rows.size) - (np.cumsum(counts) - counts)[owner]
    send_rows = np.full((world, cap), -1, np.int32)
    send_rows[owner, j] = rows % per_device
    slot_of_row = owner * cap + j
    dest = np.full(sum(p[2] for p in parts), world * cap, np.int32)
    pos, cand = 0, 0
    for (_, inv, padded, _), c, rank in zip(parts, cands, ranks):
        r = rank[inv]
        hit = np.flatnonzero(r >= 0)
        dest[pos + hit] = slot_of_row[row_of[cand + r[hit]]]
        pos, cand = pos + padded, cand + c.size
    return dest, send_rows, fill


def _keep_formats(tree: Any) -> Any:
    return jax.tree.map(lambda a: getattr(a, "format", None) or a.layout, tree)


class SparseCoreRowwiseTables:
    """Owns the SparseCore tables of one sparse arch and their per-step programs.

    Args:
        tables: the tables to place on SparseCore.
        mesh: the compiled step's GSPMD mesh; activations leave in, and gradients
            arrive in, its batch sharding, so no resharding happens across the
            graph break.
        axis: the mesh's batch axis name.
        eps: row-wise Adagrad epsilon.
        lr: returns the current sparse learning rate (the warmup schedule's value).
        max_ids_per_partition / max_unique_ids_per_partition: JTE's per-partition
            input limits.
        grad_scale: multiplies the gradients before the update, as the in_graph
            sparse optimizer's ``grad_scale`` does.
    """

    def __init__(
        self,
        tables: Sequence[SparseCoreTable],
        mesh: Any,
        axis: str,
        eps: float,
        lr: Callable[[], float],
        max_ids_per_partition: int,
        max_unique_ids_per_partition: int,
        grad_scale: float = 1.0,
    ) -> None:
        jte_embedding, jte_spec, jte_stacking, jte_utils = _jte()
        self._jte_embedding = jte_embedding
        self._jte_spec = jte_spec
        self._jte_stacking = jte_stacking
        self._mesh = mesh
        self._axis = axis
        self._eps = eps
        self._lr = lr
        self._grad_scale = grad_scale
        self._limits = (max_ids_per_partition, max_unique_ids_per_partition)
        self._exchange_fill: float = 0.0
        self._num_sc: int = jte_utils.num_sparsecores_per_device()
        self._world: int = mesh.size
        self._hosts: int = jax.process_count()
        self._local_devices: int = self._world // self._hosts
        self._tables: Dict[str, SparseCoreTable] = {t.name: t for t in tables}
        self._feature_table: Dict[str, str] = {
            f: t.name for t in tables for f in t.feature_names
        }
        self._subs: Dict[str, List[_SubTable]] = {}
        max_rows = min(
            _SUB_TABLE_MAX_ROWS,
            int(os.environ.get("RECIR_SPARSECORE_SUB_TABLE_ROWS", _SUB_TABLE_MAX_ROWS)),
        )
        for t in tables:
            self._subs[t.name] = _split_table(t, max_rows, self._world, self._num_sc)
        self._programs: Dict[Tuple[Tuple[str, Tuple[int, int]], ...], Any] = {}
        self._pending: Dict[int, jax.Array] = {}
        self._step_inputs: Any = None
        self._warned_host_grad: bool = False
        self._expected: Optional[set] = None
        self._logged_grad_type: bool = False
        self._lookups: int = 0
        self._rounding_counter: int = 0
        self._pool = ThreadPoolExecutor(self._local_devices)
        self.ev_w: Dict[str, Any] = {}
        self.momentum: Dict[str, jax.Array] = {}

    @property
    def feature_names(self) -> List[str]:
        return list(self._feature_table)

    def _table_spec(self, sub: _SubTable, initializer: Any) -> Any:
        t = self._tables[sub.table]
        return self._jte_spec.TableSpec(
            vocabulary_size=sub.rows,
            embedding_dim=t.embedding_dim,
            initializer=initializer,
            optimizer=self._jte_spec.SGDOptimizerSpec(learning_rate=1.0),
            combiner="sum",
            name=sub.stack,
            max_ids_per_partition=self._limits[0],
            max_unique_ids_per_partition=self._limits[1],
        )

    def _feature_specs(
        self, global_sizes: Dict[str, int], initializers: Dict[str, Any]
    ) -> Dict[str, Any]:
        """One JTE feature per (feature, sub-table); one stack per sub-table."""
        # Every position is one id in each of its table's stacks; each per-SparseCore
        # partition segment is padded to 8. JTE's default buffer is sized for
        # max_ids_per_partition in every partition, ~10x this at 128 devices.
        num_scs = self._world * self._num_sc
        per_device: Dict[str, int] = {}
        for f, n in global_sizes.items():
            t = self._feature_table[f]
            per_device[t] = per_device.get(t, 0) + n // self._world
        table_specs = {
            sub.stack: self._table_spec(sub, initializers.get(sub.stack, _zeros_f32))
            for subs in self._subs.values()
            for sub in subs
        }
        specs: Dict[str, Any] = {}
        for f, n in global_sizes.items():
            t = self._tables[self._feature_table[f]]
            for sub in self._subs[t.name]:
                specs[f"{f}@{sub.stack}"] = self._jte_spec.FeatureSpec(
                    name=f"{f}@{sub.stack}",
                    table_spec=table_specs[sub.stack],
                    input_shape=(n, 1),
                    output_shape=(n, t.embedding_dim),
                )
        for stack in table_specs:
            self._jte_stacking.stack_tables(
                specs,
                table_names=[stack],
                global_device_count=self._world,
                num_sc_per_device=self._num_sc,
                stack_name=stack,
            )
        # stack_tables does not carry the buffer size over from the TableSpec. The
        # SparseCore kernel needs at least max_ids_per_partition per SparseCore.
        floor = _round_up(self._limits[0], 8) * self._num_sc
        ceiling = self._jte_embedding.pybind_input_preprocessing.compute_theoretical_max_coo_buffer_size(
            self._limits[0], self._world, self._num_sc
        )
        for subs in self._subs.values():
            for sub in subs:
                ts = table_specs[sub.stack]
                if ts.setting_in_stack is None:
                    continue
                needed = per_device.get(sub.table, 0) + 8 * num_scs * self._num_sc
                ts.stacked_table_spec = dataclasses.replace(
                    ts.stacked_table_spec,
                    suggested_coo_buffer_size_per_device=min(
                        ceiling, max(floor, _round_up(needed, 8 * self._num_sc))
                    ),
                )
        return specs

    def initialize(
        self, global_sizes: Dict[str, int], host_weights: Dict[str, np.ndarray]
    ) -> None:
        """Create the tables and the momentum on device.

        W is uniform(-init_bound, init_bound) per table, or ``host_weights[table]``
        ([num_embeddings, D]) where given, so a run can start from the same weights
        as another backend.
        """

        # float32 regardless of the requested dtype: JTE requires it, and under
        # JAX_ENABLE_X64 initializers default to float64.
        def uniform(bound: float) -> Any:
            def init(key: Any, shape: Any, dtype: Any = None) -> Any:
                return jax.random.uniform(key, shape, jnp.float32, -bound, bound)

            return init

        multiple = self._num_sc * self._world
        global_sizes = {
            f: -(-n // multiple) * multiple for f, n in global_sizes.items()
        }
        w_init = {
            sub.stack: uniform(self._tables[sub.table].init_bound)
            for subs in self._subs.values()
            for sub in subs
        }
        rows_sharding = NamedSharding(self._mesh, P(self._axis, None))
        width = self._world * self._num_sc
        specs = self._feature_specs(global_sizes, w_init)
        stacks: Dict[str, List[Any]] = {}
        for tspec in jax.tree.leaves(self._jte_embedding.get_table_specs(specs)):
            stacks.setdefault(tspec.setting_in_stack.stack_name, []).append(tspec)
        rngs = jax.random.split(jax.random.PRNGKey(0), width * len(stacks))
        for i, (stack, ts) in enumerate(sorted(stacks.items())):
            self.ev_w[stack] = self._jte_embedding._init_stacked_embedding_table(
                rng=rngs[i * width : (i + 1) * width],
                stack_name=stack,
                table_specs=ts,
                global_sharding=rows_sharding,
                sharding_axis=self._axis,
                num_sparsecore_per_device=self._num_sc,
            )
        for subs in self._subs.values():
            for sub in subs:
                if self._tables[sub.table].fp16:
                    ev = self.ev_w[sub.stack]
                    self.ev_w[sub.stack] = jax.jit(
                        lambda v: v._replace(
                            table=v.table.astype(jnp.float16).astype(jnp.float32)
                        ),
                        donate_argnums=0,
                        out_shardings=_keep_formats(ev),
                    )(ev)
        vec_sharding = NamedSharding(self._mesh, P(self._axis))
        self.momentum = {
            stack: jax.jit(
                lambda n=v.table.shape[0]: jnp.zeros((n,), jnp.float32),
                out_shardings=vec_sharding,
            )()
            for stack, v in self.ev_w.items()
        }
        for table, weights in host_weights.items():
            self._load_host_weights(table, weights, global_sizes)
        logger.info(
            "[sparsecore] tables=%s stacks=%s",
            {t: [(s.start, s.rows) for s in subs] for t, subs in self._subs.items()},
            {
                s: (list(v.table.shape), str(v.table.dtype))
                for s, v in self.ev_w.items()
            },
        )

    def _load_host_weights(
        self, table: str, weights: np.ndarray, global_sizes: Dict[str, int]
    ) -> None:
        specs = self._feature_specs(global_sizes, {})
        for sub in self._subs[table]:
            ts = next(
                s.table_spec for s in specs.values() if s.table_spec.name == sub.stack
            )
            ids = np.arange(sub.rows, dtype=np.int64)
            phys = np.asarray(
                self._jte_stacking.compute_physical_row_ids(
                    self._world * self._num_sc,
                    ts.stacked_table_spec.stack_vocab_size,
                    ts.setting_in_stack.shard_rotation,
                    ts.setting_in_stack.row_offset_in_shard,
                    ids,
                )
            )
            values = np.asarray(weights[sub.start : sub.start + sub.rows], np.float32)
            if self._tables[table].fp16:
                values = values.astype(np.float16).astype(np.float32)
            ev = self.ev_w[sub.stack]
            fmt = _keep_formats(ev)
            self.ev_w[sub.stack] = jax.jit(
                lambda v, r, x: v._replace(table=v.table.at[r].set(x)),
                donate_argnums=0,
                out_shardings=fmt,
            )(ev, jnp.asarray(phys, jnp.int32), jnp.asarray(values))

    def _program(self, sizes: Dict[str, Tuple[int, int]]) -> Any:
        """``sizes``: per feature, (positions, positions padded to JTE's per-device multiple)."""
        key = tuple(sorted(sizes.items()))
        if key in self._programs:
            return self._programs[key]
        specs = self._feature_specs(
            {f: padded * self._world for f, (_, padded) in sizes.items()}, {}
        )
        mesh, axis, world, num_sc = self._mesh, self._axis, self._world, self._num_sc
        jte = self._jte_embedding
        kw = {
            "global_device_count": world,
            "sharding_strategy": "MOD",
            "num_sc_per_device": num_sc,
        }
        stacks = sorted(self.ev_w)
        sub_of = {f: [s for s in specs if s.startswith(f + "@")] for f in sizes}
        layout = {}
        for s in specs.values():
            ts = s.table_spec
            layout[ts.name] = (
                ts.stacked_table_spec.stack_vocab_size // (world * num_sc),
                ts.setting_in_stack.shard_rotation,
                ts.setting_in_stack.row_offset_in_shard,
            )

        def fwd_local(
            csr: Any, ev_w: Any, slots: Dict[str, jax.Array]
        ) -> Dict[str, jax.Array]:
            acts = jte.tpu_sparse_dense_matmul(csr, ev_w, feature_specs=specs, **kw)
            out: Dict[str, jax.Array] = {}
            for f, subs in sub_of.items():
                total = acts[subs[0]]
                for s in subs[1:]:
                    total = total + acts[s]
                out[f] = total[slots[f]]
            return out

        batch = P(axis)
        fwd = jax.jit(
            jax.shard_map(
                fwd_local,
                mesh=mesh,
                in_specs=(batch, batch, batch),
                out_specs=batch,
                check_vma=False,
            ),
            out_shardings=NamedSharding(mesh, P(axis, None)),
        )
        stack_features = {
            stack: [
                f
                for f, subs in sub_of.items()
                if any(specs[s].table_spec.name == stack for s in subs)
            ]
            for stack in stacks
        }
        # Unique rows a device sends one owner; a device's rows spread evenly
        # over owners, so twice the mean (every position unique) leaves room for
        # skew. RECIR_SPARSECORE_EXCHANGE_CAP_SCALE shrinks it toward the logged
        # max exchange fill; the host refuses a step that would overflow.
        scale = float(os.environ.get("RECIR_SPARSECORE_EXCHANGE_CAP_SCALE", "1"))
        caps = {
            stack: _round_up(
                int(
                    scale
                    * (
                        2 * sum(sizes[f][1] for f in stack_features[stack]) // world
                        + 1024
                    )
                ),
                128,
            )
            for stack in stacks
        }

        # The transfer is bf16 when the ring-shift exchange is (the baseline's
        # setting); rows are summed before it and accumulated after it in f32.
        transfer = (
            jnp.bfloat16
            if os.environ.get("RECIR_RINGSHIFT_EXCHANGE_BF16", "0") == "1"
            else jnp.float32
        )

        def exchange_update(
            stack: str,
            g: jax.Array,
            dest: jax.Array,
            send_rows: jax.Array,
            lr: jax.Array,
            table: jax.Array,
            mom: jax.Array,
            rounding_key: Optional[jax.Array],
        ) -> Any:
            cap, dim = caps[stack], g.shape[1]
            send = jnp.zeros((world * cap + 1, dim), jnp.float32).at[dest].add(g)[:-1]
            recv = jax.lax.all_to_all(
                send.astype(transfer).reshape(world, cap, dim), axis, 0, 0
            )
            recv_rows = jax.lax.all_to_all(
                send_rows.reshape(world, cap), axis, 0, 0
            ).reshape(-1)
            return rowwise_adagrad_update(
                recv_rows,
                recv.reshape(-1, dim).astype(jnp.float32),
                table,
                mom,
                lr,
                self._eps,
                rounding_key,
            )

        fp16_stacks = {
            sub.stack
            for subs in self._subs.values()
            for sub in subs
            if self._tables[sub.table].fp16
        }

        def exchange_local(
            grads: Dict[str, jax.Array],
            dest: Dict[str, jax.Array],
            send_rows: Dict[str, jax.Array],
            lr: jax.Array,
            counter: jax.Array,
            ev_w: Any,
            mom: Any,
        ) -> Any:
            new_w, new_m = dict(ev_w), dict(mom)
            device = jax.lax.axis_index(axis).astype(jnp.int32)
            for stream, stack in enumerate(stacks):
                rounding_key = (
                    fp16_rounding_key(counter, stream, device)
                    if stack in fp16_stacks
                    else None
                )
                g = self._grad_scale * jnp.concatenate(
                    [
                        jnp.pad(
                            grads[f].astype(jnp.float32),
                            ((0, sizes[f][1] - sizes[f][0]), (0, 0)),
                        )
                        for f in stack_features[stack]
                    ]
                )
                w, m = exchange_update(
                    stack,
                    g,
                    dest[stack],
                    send_rows[stack],
                    lr,
                    ev_w[stack].table,
                    mom[stack],
                    rounding_key,
                )
                new_w[stack], new_m[stack] = ev_w[stack]._replace(table=w), m
            return new_w, new_m

        bwd = jax.jit(
            jax.shard_map(
                exchange_local,
                mesh=mesh,
                in_specs=(batch, batch, batch, P(), P(), batch, batch),
                out_specs=batch,
                check_vma=False,
            ),
            donate_argnums=(5, 6),
            out_shardings=_keep_formats((self.ev_w, self.momentum)),
        )
        self._programs[key] = (specs, fwd, bwd, (stack_features, caps, layout))
        return self._programs[key]

    def _global(self, local: np.ndarray) -> jax.Array:
        """This host's per-device blocks (local-device order) -> a batch-sharded global array."""
        sharding = NamedSharding(self._mesh, P(self._axis) if local.ndim else P())
        shape = (local.shape[0] * self._hosts,) + local.shape[1:] if local.ndim else ()
        return jax.make_array_from_process_local_data(sharding, local, shape)

    def lookup(self, host_ids: Dict[str, np.ndarray]) -> List[jax.Array]:
        """Forward. ``host_ids``: this host's ids per feature, flat in local-device order.

        Returns one batch-sharded [global positions, D] activation per feature, in
        ``host_ids`` order, and keeps what the backward needs.
        """
        if not self.ev_w:
            raise RuntimeError("SparseCoreRowwiseTables.initialize() must run first")
        # JTE splits each device's batch evenly over its SparseCores.
        sizes = {}
        for f, a in host_ids.items():
            n = a.size // self._local_devices
            sizes[f] = (n, -(-n // self._num_sc) * self._num_sc)
        specs, fwd, _, exchange = self._program(sizes)
        indices: Dict[str, np.ndarray] = {}
        values: Dict[str, np.ndarray] = {}
        shapes: Dict[str, np.ndarray] = {}
        slots: Dict[str, np.ndarray] = {}
        per_device: Dict[str, List[Any]] = {}
        # Per-device work runs in worker threads writing into preallocated arrays:
        # the main thread shares the GIL with the input pipeline's producer thread.
        local = self._local_devices
        for f, ids in host_ids.items():
            n, padded = sizes[f]
            subs = self._subs[self._feature_table[f]]
            ids = ids.astype(np.int64, copy=False).reshape(local, n)
            slots[f] = np.empty((local, n), np.int32)
            per_device[f] = list(
                self._pool.map(
                    _device_inputs,
                    ids,
                    range(local),
                    repeat(padded),
                    repeat(self._num_sc),
                    repeat(subs),
                    repeat(slots[f]),
                )
            )
            slots[f] = slots[f].reshape(-1)
            for i, sub in enumerate(subs):
                name = f"{f}@{sub.stack}"
                rows = [p[0][i][0] for p in per_device[f]]
                indices[name] = np.zeros((sum(r.size for r in rows), 2), np.int64)
                np.concatenate(rows, out=indices[name][:, 0])
                values[name] = np.concatenate([p[0][i][1] for p in per_device[f]])
                shapes[name] = np.array([local * padded, 1], np.int64)
        dest: Dict[str, np.ndarray] = {}
        send_rows: Dict[str, np.ndarray] = {}
        stack_features, caps, layout = exchange
        for stack, features in stack_features.items():
            sub = {
                f: next(
                    x for x in self._subs[self._feature_table[f]] if x.stack == stack
                )
                for f in features
            }
            parts = [
                [
                    (per_device[f][d][1], per_device[f][d][2], sizes[f][1], sub[f])
                    for f in features
                ]
                for d in range(local)
            ]
            out = list(
                self._pool.map(
                    _exchange_layout,
                    parts,
                    repeat(layout[stack]),
                    repeat(self._world),
                    repeat(self._num_sc),
                    repeat(caps[stack]),
                )
            )
            dest[stack] = np.concatenate([o[0] for o in out])
            send_rows[stack] = np.concatenate([o[1].reshape(-1) for o in out])
            self._exchange_fill = max(self._exchange_fill, max(o[2] for o in out))
        csr, stats = (
            self._jte_embedding.preprocess_sparse_dense_matmul_input_from_sparse_tensor(
                indices=indices,
                values=values,
                dense_shapes=shapes,
                feature_specs=specs,
                local_device_count=self._local_devices,
                global_device_count=self._world,
                num_sc_per_device=self._num_sc,
                sharding_strategy="MOD",
                allow_id_dropping=False,
            )
        )
        self._lookups += 1
        if self._lookups <= 3 or self._lookups % 100 == 0:
            logger.info(
                "[sparsecore] lookup %d: max ids / unique ids per partition %s (limit %d); max exchange fill %.3f",
                self._lookups,
                {
                    stack: (
                        int(np.max(stats.max_ids_per_partition[stack])),
                        int(np.max(stats.max_unique_ids_per_partition[stack])),
                    )
                    for stack in stats.max_ids_per_partition
                },
                self._limits[0],
                self._exchange_fill,
            )
        # JTE's inputs are int32; under JAX_ENABLE_X64 numpy integers default to int64.
        csr = jax.tree.map(lambda x: self._global(_int32(np.asarray(x))), csr)
        slots_g = {f: self._global(v) for f, v in slots.items()}
        self._step_inputs = (
            sizes,
            jax.tree.map(self._global, dest),
            jax.tree.map(self._global, send_rows),
        )
        self._pending = {}
        acts = fwd(csr, self.ev_w, slots_g)
        return [acts[f] for f in host_ids]

    def expect_gradients(self, indices: Sequence[int]) -> None:
        """The lookup outputs (``lookup`` order) the step delivers gradients for; an
        output the loss does not use gets none and counts as a zero gradient."""
        self._expected = set(indices)

    def backward(self, grad: Any, index: int) -> None:
        """Gradient of lookup output ``index``; updates once every expected gradient is in."""
        if self._step_inputs is None:
            raise RuntimeError("backward() without a preceding lookup()")
        self._pending[index] = grad
        expected = (
            self._expected
            if self._expected is not None
            else set(range(len(self._step_inputs[0])))
        )
        if not expected <= set(self._pending):
            return
        self._apply(self._pending)
        self._pending = {}
        self._step_inputs = None

    def _apply(self, pending: Dict[int, Any]) -> None:
        sizes, dest_g, send_rows_g = self._step_inputs
        grads = {}
        for i, (f, (n, _)) in enumerate(sizes.items()):
            grad = pending.get(i)
            if grad is None:
                dim = self._tables[self._feature_table[f]].embedding_dim
                grad = jax.jit(
                    lambda shape=(n * self._world, dim): jnp.zeros(shape, jnp.float32),
                    out_shardings=NamedSharding(self._mesh, P(self._axis, None)),
                )()
            elif not isinstance(grad, jax.Array):
                # The compiled step's last train step returns torch tensors (host copies).
                if not self._warned_host_grad:
                    logger.warning(
                        "[sparsecore] gradient arrived as %s; copying it to device",
                        type(grad).__name__,
                    )
                    self._warned_host_grad = True
                grad = self._global(grad.detach().float().cpu().numpy())
            grads[f] = grad
        if not self._logged_grad_type:
            logger.info(
                "[sparsecore] gradients arrive as %s, sharded %s",
                sorted({type(g).__name__ for g in pending.values()}),
                sorted({str(getattr(g, "sharding", None)) for g in pending.values()}),
            )
            self._logged_grad_type = True
        _, _, bwd, _ = self._program(sizes)
        lr = jnp.float32(self._lr())
        counter = jnp.int32(self._rounding_counter)
        self._rounding_counter = (self._rounding_counter + 1) % (2**31)
        self.ev_w, self.momentum = bwd(
            grads, dest_g, send_rows_g, lr, counter, self.ev_w, self.momentum
        )
