# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Embedding tables on TPU SparseCore through torch_tpu's native ops.

``SparseCoreEmbeddingCollection`` stands in for a torchrec
``EmbeddingCollection``. Each table is MOD-sharded over every SparseCore of
every process: with ``S = world * NUM_SC`` SparseCores, row ``r`` belongs to
SparseCore shard ``r % S`` at position ``r // S``, and the process on TPU device
``d`` holds shards ``d * NUM_SC`` to ``d * NUM_SC + NUM_SC - 1``, concatenated
(the layout TorchRec's torch_tpu modules use). Ownership follows the TPU's
global device id, which the SparseCore programs are laid out by, not the
process rank: torch_tpu orders devices by their chip coordinates.

Each step, a table's ids are deduplicated on the host and SparseCore looks up
each distinct id once (``torch.tpu.ops.sparse_dense_matmul``, one id per
sample); one gather on the device expands the rows to every token, and its
backward sums the tokens' gradients per row for exact row-wise Adagrad
(``sparse_dense_matmul_grad_with_adagrad``). A call stages each partition's rows
in the SparseCore's tile memory, which caps the ids a partition takes per call
(224 at 512 dims), so the distinct ids are split into as many chunks as that
needs, the same number on every process. An id's chunk follows from the id, so
every process puts it in the same chunk and its row gets one update with every
process's gradient summed; each chunk takes whole blocks of consecutive ids, so
it covers every table shard, and within a chunk the ids alternate between the
device's SparseCores. Each process passes its own shard and
its own ids; preprocessing told the world size routes every id to its owner,
and the ops exchange ids, embeddings and gradients across devices. Both ops are
compiled with ``torch.compile(backend="tpu")``; eagerly, the update is ~1000x
slower.

The collection is called at a graph break (``CPUSparseArchDispatcher``), like the
CPU backend's torchrec tables, but its outputs and gradients stay on the device.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from recir.embedding.backend import (
    check_binding,
    EmbeddingBackend,
    EmbeddingSetupResult,
    SparseOptimizerRuntime,
    SparseUpdatePlacement,
    swap_submodule,
)
from recir.training.optim import OptimConfig
from recir.utils.meta_tensor_init import fill_table_shard, table_init_bounds
from torchrec.distributed.embedding_types import OptimType
from torchrec.modules.embedding_configs import BaseEmbeddingConfig, DataType
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.optim.keyed import KeyedOptimizer
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor

logger: logging.Logger = logging.getLogger(__name__)

# SparseCores per device on TPU v7x and v6e.
NUM_SC: int = 2


def _round_up(x: int, m: int) -> int:
    return x + (-x) % m


# Rows of a table generated at a time when a process builds its shard.
_INIT_BLOCK_ROWS: int = 1 << 16


def _local_shard(
    config: BaseEmbeddingConfig, rows: int, owner: int, world: int
) -> torch.Tensor:
    """The shard of ``config``'s table on device ``owner``, from RecIR's seeded
    init of its ``init_fn`` (``table_init_bounds``), as the CPU backend draws it.

    ``rows`` is the padded row count (a multiple of ``8 * world * NUM_SC``);
    padding rows stay zero. The table is generated block by block, so no
    process holds all of it.
    """
    shards = world * NUM_SC
    per_shard = rows // shards
    local = torch.zeros(rows // world, config.embedding_dim)
    table_shape = (config.num_embeddings, config.embedding_dim)
    low, high = table_init_bounds(config)
    for start in range(0, config.num_embeddings, _INIT_BLOCK_ROWS):
        end = min(start + _INIT_BLOCK_ROWS, config.num_embeddings)
        block = torch.empty(end - start, config.embedding_dim)
        fill_table_shard(
            block,
            config.name,
            table_shape,
            (start, 0),
            low,
            high,
        )
        for c in range(NUM_SC):
            shard = owner * NUM_SC + c
            first = start + (shard - start) % shards  # first row of this shard
            picked = block[first - start :: shards]
            at = c * per_shard + first // shards
            local[at : at + picked.shape[0]] = picked
    return local


def _tpu_ops() -> Any:
    # torch.tpu.ops, by its defining module (torch_tpu is not a Buck dependency).
    return importlib.import_module("torch_tpu._internal.ops.ops")


def _tpu_distributed() -> Any:
    return importlib.import_module("torch_tpu._internal.distributed.tpu_distributed")


# Bytes of SparseCore tile memory a call stages one partition's looked-up rows
# in (XLA's "TileSpmem bytes" when they do not fit).
_TILE_SPMEM_BYTES: int = 524032
# Most chunks one table's lookup is split into in a step.
_MAX_CHUNKS: int = 64


def _max_ids_cap(embedding_dim: int) -> int:
    """Most ids one partition can take in a call: their fp32 rows must fit the
    SparseCore's tile memory (with 10% headroom); 224 at 512 dims."""
    return max(8, (_TILE_SPMEM_BYTES * 9 // 10) // (embedding_dim * 4) // 8 * 8)


@dataclass(frozen=True)
class _Limits:
    """Static limits of one SparseCore call over ``samples`` samples of at most
    one id each.

    No partition (one SparseCore's samples x one table shard) may take more than
    ``max_ids`` ids; the caller splits a lookup into chunks until each fits. The
    COO buffer is the theoretical maximum for that limit (max ids per partition
    x SparseCores per device x SparseCores in total): fewer slots than a batch
    needs drop ids (even with dropping off), and more abort.
    """

    samples: int
    max_ids: int
    buffer: int

    @classmethod
    def for_chunk(cls, samples: int, max_ids: int, world: int) -> "_Limits":
        return cls(samples, max_ids, max_ids * NUM_SC * world * NUM_SC)


def _chunk_samples(tokens: int, chunks: int) -> int:
    """Samples per chunk when ``tokens`` ids (at most) are split into ``chunks``."""
    return _round_up(-(-tokens // chunks), 8 * NUM_SC)


def _chunk_layout(
    unique: torch.Tensor, shards: int, num_chunks: int, num_samples: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Each distinct id's chunk, and its sample within the chunk.

    ``shards`` is the number of SparseCores in the world (``world * NUM_SC``),
    which is also the number of shards of every table: row ``r`` lives on shard
    ``r % shards`` at position ``r // shards``. So ``id // shards`` numbers the
    blocks of ``shards`` consecutive ids, each of which holds one id of every
    shard, and ``% num_chunks`` deals the blocks out to the chunks in turn.
    With 16 shards and 2 chunks, ids 0-15 go to chunk 0, ids 16-31 to chunk 1,
    ids 32-47 to chunk 0, and so on.

    The chunk ignores the id's shard, so each chunk's ids spread over the shards
    as all the ids do. ``id % num_chunks`` would not: with 16 shards and 2
    chunks, chunk 0 would hold only ids of even shards, which doubles the ids
    each of those shards takes per call, the quantity ``max_ids`` caps.

    The chunk depends on the id alone, so every process puts an id in the same
    chunk, and its row gets one update with every process's gradient summed. A
    device's SparseCores take contiguous halves of the samples, so the ids of a
    chunk alternate between the halves.
    """
    chunk = (unique.long() // shards) % num_chunks
    order = torch.argsort(chunk, stable=True)
    counts = torch.bincount(chunk, minlength=num_chunks)
    position = torch.empty_like(chunk)
    position[order] = (
        torch.arange(unique.numel()) - (counts.cumsum(0) - counts)[chunk[order]]
    )
    per_sc = num_samples // NUM_SC
    return chunk, (position % NUM_SC) * per_sc + position // NUM_SC


def _csr(
    ids: torch.Tensor, sample: torch.Tensor, num_samples: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """CSR input over ``num_samples`` samples where ``sample[i]`` holds
    ``ids[i]`` and every other sample holds none."""
    lengths = torch.zeros(num_samples, dtype=torch.int32)
    lengths[sample] = 1
    offsets = torch.zeros(num_samples + 1, dtype=torch.int32)
    offsets[1:] = lengths.cumsum(0)
    return ids[torch.argsort(sample)], offsets


class _Lookup(torch.autograd.Function):
    """One table's chunked lookup; backward applies row-wise Adagrad per chunk."""

    @staticmethod
    # pyre-ignore[14]: Function.forward takes these arguments.
    def forward(
        ctx: Any,
        anchor: torch.Tensor,
        collection: "SparseCoreEmbeddingCollection",
        table: str,
        csrs: Tuple[Tuple[torch.Tensor, ...], ...],
        limits: _Limits,
    ) -> torch.Tensor:
        ctx.collection, ctx.table, ctx.csrs, ctx.limits = (
            collection,
            table,
            csrs,
            limits,
        )
        return torch.cat([collection._lookup(table, csr, limits) for csr in csrs])

    @staticmethod
    # pyre-ignore[14]
    def backward(ctx: Any, grad: torch.Tensor) -> Tuple[Any, ...]:
        samples = ctx.limits.samples
        # Every process puts an id in the same chunk, so each row is updated in
        # one chunk, with every process's gradient for it summed.
        for j, csr in enumerate(ctx.csrs):
            chunk_grad = grad[j * samples : (j + 1) * samples].contiguous()
            ctx.collection._update(ctx.table, csr, ctx.limits, chunk_grad)
        return None, None, None, None, None


class SparseCoreEmbeddingCollection(nn.Module):
    """A torchrec ``EmbeddingCollection`` whose tables live on TPU SparseCore."""

    def __init__(
        self,
        collection: EmbeddingCollection,
        device: torch.device,
        lr: float,
        eps: float,
        owner: int = 0,
        world: int = 1,
    ) -> None:
        super().__init__()
        self._device = device
        self._eps = eps
        self._world = world
        self._shards = world * NUM_SC
        self._feature_names: List[List[str]] = [
            list(names) for names in collection._feature_names
        ]
        self._embedding_names: List[List[str]] = collection.embedding_names_by_table()
        self._tables: List[str] = []
        self._caps: Dict[str, int] = {}
        self._rows: Dict[str, int] = {}
        self.weights: Dict[str, torch.Tensor] = {}
        self.accumulators: Dict[str, torch.Tensor] = {}
        for config in collection.embedding_configs():
            if config.embedding_dim % 8:
                raise ValueError(f"{config.name}: embedding_dim must divide by 8")
            if config.num_embeddings > 2**31:
                raise ValueError(
                    f"{config.name}: {config.num_embeddings} rows, but SparseCore "
                    "ids are int32"
                )
            if config.data_type != DataType.FP32:
                raise ValueError(
                    f"{config.name}: SparseCore tables train in FP32, not "
                    f"{config.data_type}"
                )
            rows = _round_up(config.num_embeddings, 8 * world * NUM_SC)
            # RecIR's seeded init, so the tables match the CPU backend's.
            local = _local_shard(config, rows, owner, world)
            self._tables.append(config.name)
            self._caps[config.name] = _max_ids_cap(config.embedding_dim)
            self._rows[config.name] = config.num_embeddings
            self.weights[config.name] = local.to(device)
            self.accumulators[config.name] = torch.zeros(local.shape[0], device=device)
            logger.info(
                "SparseCore table %s: %d x %d (%d rows padded, %d on this process), "
                "features %s",
                config.name,
                config.num_embeddings,
                config.embedding_dim,
                rows,
                local.shape[0],
                config.feature_names,
            )
        # The autograd input that makes the lookups' outputs require grad.
        self._anchor: torch.Tensor = torch.zeros((), requires_grad=True)
        # The handle the trainer writes the sparse learning rate into
        # (SparseOptimizerRuntime.fused_optimizer); the update reads it.
        # It holds no table: the update donates the tables, and another reference
        # would keep a donated buffer alive.
        self.optimizer = KeyedOptimizer(
            params={"lr_anchor": self._anchor},
            state={},
            param_groups=[{"params": [self._anchor], "lr": lr}],
        )
        self._compiled: Dict[Tuple[str, str, _Limits], Callable[..., Any]] = {}
        # Every step has the first step's ids per table, so a table's calls
        # differ only in their chunk count, a power of two up to _MAX_CHUNKS.
        self._tokens: List[int] | None = None
        # Dynamo counts every table's lookups (and updates) against one cache, and
        # with fullgraph=True raises past its limit, 8 by default.
        self._recompile_limit: int = len(self._tables) * _MAX_CHUNKS.bit_length()

    def _compiled_fn(
        self, kind: str, table: str, limits: _Limits
    ) -> Callable[..., Any]:
        key = (kind, table, limits)
        if key not in self._compiled:
            ops = _tpu_ops()
            if kind == "lookup":

                def fn(rp: Any, ids: Any, samples: Any, gains: Any, w: Any) -> Any:
                    return ops.sparse_dense_matmul(
                        rp,
                        ids,
                        samples,
                        gains,
                        w,
                        device_batch_size=limits.samples,
                        max_ids_per_partition=limits.max_ids,
                        max_unique_ids_per_partition=limits.max_ids,
                    )

            else:

                def fn(
                    rp: Any,
                    ids: Any,
                    samples: Any,
                    gains: Any,
                    w: Any,
                    acc: Any,
                    grad: Any,
                    lr: Any,
                ) -> Any:
                    return ops.sparse_dense_matmul_grad_with_adagrad(
                        rp,
                        ids,
                        samples,
                        gains,
                        w,
                        acc,
                        grad,
                        lr,
                        self._eps,
                        device_batch_size=limits.samples,
                        max_ids_per_partition=limits.max_ids,
                        max_unique_ids_per_partition=limits.max_ids,
                        computation_name=f"recir_sparsecore_{table}",
                    )

            logger.info("SparseCore: compiling %s for %s with %s", kind, table, limits)
            self._compiled[key] = torch.compile(fn, backend="tpu", fullgraph=True)
        return self._compiled[key]

    def _max_ids_per_partition(self, ids: torch.Tensor, offsets: torch.Tensor) -> int:
        max_ids, _ = torch.ops.tpu.compute_sparse_dense_matmul_stats(
            ids, offsets, self._world, NUM_SC
        )
        return int(max_ids)

    def _fits(
        self, unique: torch.Tensor, num_chunks: int, tokens: int, cap: int
    ) -> bool:
        """Whether every chunk of ``unique`` fits its samples and every partition
        takes at most ``cap`` ids."""
        num_samples = _chunk_samples(tokens, num_chunks)
        chunk, sample = _chunk_layout(unique, self._shards, num_chunks, num_samples)
        if int(torch.bincount(chunk, minlength=num_chunks).max()) > num_samples:
            return False
        return all(
            self._max_ids_per_partition(
                *_csr(unique[chunk == j], sample[chunk == j], num_samples)
            )
            <= cap
            for j in range(num_chunks)
        )

    def _chunk_count(
        self, table: str, unique: torch.Tensor, tokens: int, start: int
    ) -> int:
        """The fewest chunks (a power of two, at least ``start``) that ``unique``
        fits in."""
        k = start
        while not self._fits(unique, k, tokens, self._caps[table]):
            if k >= _MAX_CHUNKS:
                raise RuntimeError(
                    f"SparseCore: the {unique.numel()} distinct ids of {table} do "
                    f"not fit {_MAX_CHUNKS} chunks of {self._caps[table]} ids per "
                    "partition"
                )
            k *= 2
        return k

    def _all_reduce_max(self, values: List[int]) -> List[int]:
        """The elementwise maximum of ``values`` over the processes."""
        if self._world == 1:
            return values
        agreed = torch.tensor(values)
        dist.all_reduce(agreed, op=dist.ReduceOp.MAX)
        return agreed.tolist()

    def _chunk_counts(
        self, uniques: Sequence[torch.Tensor], tokens: List[int]
    ) -> List[int]:
        """Each table's chunk count, the same on every process.

        The processes' SparseCore calls form one program across the devices, so
        every process makes the same calls with the same shapes. Doubling the
        count splits each chunk in two but halves its samples, so ids that fit
        one count can overflow a larger one when the split is uneven; the
        processes agree until none needs more than the agreed counts.
        """
        counts = [1] * len(self._tables)
        while True:
            local_counts = [
                self._chunk_count(table, unique, n, count)
                for table, unique, n, count in zip(
                    self._tables, uniques, tokens, counts
                )
            ]
            agreed_counts = self._all_reduce_max(local_counts)
            if agreed_counts == counts:
                return counts
            counts = agreed_counts

    def _preprocess(
        self, table: str, ids: torch.Tensor, offsets: torch.Tensor, limits: _Limits
    ) -> Tuple[torch.Tensor, ...]:
        out = torch.ops.tpu.preprocess_sparse_dense_matmul_input(
            ids,
            offsets,
            limits.max_ids,
            limits.max_ids,
            limits.buffer,
            limits.samples,
            "sum",
            1,  # local_device_count: one device per process
            self._world,  # global_device_count
            NUM_SC,
            False,  # allow_id_dropping
            table,
        )
        dropped = int(out["dropped_id_count"].sum()) if "dropped_id_count" in out else 0
        if dropped:
            raise RuntimeError(
                f"SparseCore preprocessing dropped {dropped} ids of {table}"
            )
        return tuple(
            out[k].to(self._device)
            for k in ("row_pointers", "embedding_ids", "sample_ids", "gains")
        )

    def _lookup(self, table: str, csr: Tuple, limits: _Limits) -> torch.Tensor:
        with torch._dynamo.config.patch(recompile_limit=self._recompile_limit):
            return self._compiled_fn("lookup", table, limits)(*csr, self.weights[table])

    def _update(
        self, table: str, csr: Tuple, limits: _Limits, grad: torch.Tensor
    ) -> None:
        lr = torch.tensor(
            float(next(iter(self.optimizer.param_groups))["lr"]), device=self._device
        )
        # The op donates the table and accumulator and returns their new values.
        with torch._dynamo.config.patch(recompile_limit=self._recompile_limit):
            self.weights[table], self.accumulators[table] = self._compiled_fn(
                "update", table, limits
            )(*csr, self.weights[table], self.accumulators[table], grad, lr)

    def forward(self, features: KeyedJaggedTensor) -> Dict[str, JaggedTensor]:
        jagged = features.to_dict()
        pieces = [[jagged[f] for f in names] for names in self._feature_names]
        # SparseCore looks up and updates each distinct id once; one gather
        # expands the rows to every token, and its backward sums the tokens'
        # gradients per row.
        uniques, inverses = zip(
            *(
                torch.unique(torch.cat([p.values() for p in ps]), return_inverse=True)
                for ps in pieces
            )
        )
        for table, unique in zip(self._tables, uniques):
            # The ids are sorted, and the ops take them as int32.
            if unique.numel() and (unique[0] < 0 or unique[-1] >= self._rows[table]):
                raise ValueError(
                    f"SparseCore: ids of {table} must be in [0, {self._rows[table]}), "
                    f"got {int(unique[0])} to {int(unique[-1])}"
                )
        uniques = [u.to(torch.int32) for u in uniques]
        tokens = self._all_reduce_max([int(inv.numel()) for inv in inverses])
        if self._tokens is None:
            self._tokens = tokens
        if tokens != self._tokens:
            raise RuntimeError(
                f"SparseCore: {dict(zip(self._tables, tokens))} ids per table this "
                f"step, but the first step had {dict(zip(self._tables, self._tokens))}; "
                "batches must have a fixed number of ids per table (as the "
                "token-budget collate gives), or every step compiles new calls"
            )
        counts = self._chunk_counts(uniques, tokens)
        out: Dict[str, JaggedTensor] = {}
        for table, ps, names, unique, inverse, n, k in zip(
            self._tables,
            pieces,
            self._embedding_names,
            uniques,
            inverses,
            tokens,
            counts,
        ):
            limits = _Limits.for_chunk(
                _chunk_samples(n, k), self._caps[table], self._world
            )
            chunk, sample = _chunk_layout(unique, self._shards, k, limits.samples)
            csrs = tuple(
                self._preprocess(
                    table,
                    *_csr(unique[chunk == j], sample[chunk == j], limits.samples),
                    limits,
                )
                for j in range(k)
            )
            gather = (chunk * limits.samples + sample)[inverse].to(self._device)
            chunks = _Lookup.apply(self._anchor, self, table, csrs, limits)
            rows = chunks.index_select(0, gather)
            sizes = [p.values().numel() for p in ps]
            for name, piece, values in zip(names, ps, rows.split(sizes)):
                out[name] = JaggedTensor(values=values, lengths=piece.lengths())
        return out


class TorchTPUSparseCoreBackend(EmbeddingBackend):
    """The model's embedding collection on TPU SparseCore, one device per process."""

    def init_pg(self) -> Tuple[int, int]:
        return 0, 1

    def setup(
        self,
        model: nn.Module,
        table_configs: Sequence[BaseEmbeddingConfig],
        sparse_config: OptimConfig,
        device: torch.device,
        world_size: int,
    ) -> EmbeddingSetupResult:
        check_binding(model, self.embedding_collection_fqn)
        owner = 0
        if world_size > 1:
            if dist.get_world_size() != world_size:
                raise ValueError(
                    f"one device per process: world_size {world_size} != "
                    f"{dist.get_world_size()} processes"
                )
            owner = int(_tpu_distributed().global_device_id())
            logger.info(
                "SparseCore: rank %d is on TPU device %d", dist.get_rank(), owner
            )
        if sparse_config.optim_type != OptimType.ROWWISE_ADAGRAD:
            raise NotImplementedError(
                f"the SparseCore backend trains with row-wise Adagrad, not "
                f"{sparse_config.optim_type}"
            )
        collection = model.get_submodule(self.embedding_collection_fqn)
        assert isinstance(collection, EmbeddingCollection)
        sparsecore = SparseCoreEmbeddingCollection(
            collection, device, sparse_config.lr, sparse_config.eps, owner, world_size
        )
        swap_submodule(model, self.embedding_collection_fqn, sparsecore)
        return EmbeddingSetupResult(
            model=model,
            dense_named_parameters=self.dense_named_parameters(model),
            sparse_runtime=SparseOptimizerRuntime(
                placement=SparseUpdatePlacement.BACKEND_FUSED,
                config=sparse_config,
                fused_optimizer=sparsecore.optimizer,
            ),
        )

    def dense_named_parameters(self, model: nn.Module) -> Dict[str, nn.Parameter]:
        prefix = self.embedding_collection_fqn + "."
        return {
            name: param
            for name, param in model.named_parameters()
            if param.requires_grad and not name.startswith(prefix)
        }
