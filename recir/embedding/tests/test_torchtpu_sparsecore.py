# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import functools
import unittest
from typing import Any, Callable, cast, Tuple
from unittest.mock import Mock, patch

import torch
import torch.nn as nn
from recir.embedding import torchtpu_sparsecore
from recir.embedding.torchtpu_sparsecore import (
    _Limits,
    NUM_SC,
    SparseCoreEmbeddingCollection,
    TorchTPUSparseCoreBackend,
)
from recir.utils.meta_tensor_init import fill_table_shard
from torchrec.distributed.embedding_types import OptimType
from torchrec.modules.embedding_configs import DataType, EmbeddingConfig
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

_LR, _EPS = 0.1, 1e-8


def _physical(ids: torch.Tensor, rows: int) -> torch.Tensor:
    """Where logical row ``r`` sits in SparseCore layout."""
    ids = ids.long()
    return (ids % NUM_SC) * (rows // NUM_SC) + ids // NUM_SC


def _samples_of(offsets: torch.Tensor) -> torch.Tensor:
    """The sample of each id of a CSR input."""
    return torch.repeat_interleave(torch.arange(offsets.numel() - 1), offsets.diff())


def _stand_in_ops(kind: str, _table: str, limits: _Limits) -> Callable[..., Any]:
    """The SparseCore ops in plain torch, taking a chunk's CSR ids and offsets
    as their only inputs besides the tables."""
    if kind == "lookup":

        def lookup(ids: Any, offsets: Any, w: Any) -> Any:
            rows = torch.zeros(limits.samples, w.shape[1])
            rows[_samples_of(offsets)] = w[_physical(ids, w.shape[0])]
            return rows

        return lookup

    def update(
        ids: Any, offsets: Any, w: Any, acc: Any, grad: Any, lr: Any
    ) -> Tuple[Any, Any]:
        rows = _physical(ids, w.shape[0])
        summed = torch.zeros_like(w).index_add_(0, rows, grad[_samples_of(offsets)])
        touched = torch.unique(rows)
        acc, w = acc.clone(), w.clone()
        acc[touched] += summed[touched].pow(2).mean(dim=1)
        w[touched] -= lr * summed[touched] / (acc[touched].sqrt() + _EPS)[:, None]
        return w, acc

    return update


def _max_ids_per_partition(ids: torch.Tensor, offsets: torch.Tensor) -> int:
    """Most ids any (SparseCore's samples, table shard) pair takes, on one device."""
    sparsecore = _samples_of(offsets) // ((offsets.numel() - 1) // NUM_SC)
    pairs = sparsecore * NUM_SC + ids.long() % NUM_SC
    return int(torch.bincount(pairs).max()) if ids.numel() else 0


def _patch_ops(collection: SparseCoreEmbeddingCollection) -> None:
    """Replace the torch_tpu calls with their plain-torch stand-ins (until stopall)."""
    patch.object(
        collection,
        "_preprocess",
        side_effect=lambda _table, ids, offsets, _limits: (ids, offsets),
    ).start()
    patch.object(
        collection, "_max_ids_per_partition", side_effect=_max_ids_per_partition
    ).start()
    patch.object(collection, "_compiled_fn", side_effect=_stand_in_ops).start()


def _collection() -> EmbeddingCollection:
    return EmbeddingCollection(
        tables=[
            EmbeddingConfig(
                name="shared",
                embedding_dim=8,
                num_embeddings=10,
                feature_names=["a", "b"],
            ),
            EmbeddingConfig(
                name="own", embedding_dim=8, num_embeddings=6, feature_names=["c"]
            ),
        ]
    )


def _features() -> KeyedJaggedTensor:
    return KeyedJaggedTensor(
        keys=["a", "b", "c"],
        values=torch.tensor([1, 3, 1, 7, 3, 2, 5, 0]),
        lengths=torch.tensor([2, 1, 1, 1, 2, 1]),
    )


def _mod_sharded(
    name: str, rows: int, dim: int, shards: int, bound: float = 0.0
) -> torch.Tensor:
    """The seeded table, padded, in TorchRec's MOD-sharded SparseCore layout."""
    padded = torchtpu_sparsecore._round_up(rows, 8 * shards)
    table = torch.cat(
        [_logical(name, rows, dim, bound), torch.zeros(padded - rows, dim)]
    )
    return table.view(padded // shards, shards, dim).permute(1, 0, 2).reshape(-1, dim)


def _logical(name: str, rows: int, dim: int, bound: float = 0.0) -> torch.Tensor:
    """The seeded table, uniform in [-bound, bound); 0 means torchrec's default
    bound, sqrt(1 / rows)."""
    table = torch.empty(rows, dim)
    bound = bound or (1.0 / rows) ** 0.5
    return fill_table_shard(table, name, (rows, dim), (0, 0), -bound, bound)


class SparseCoreEmbeddingCollectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sc = SparseCoreEmbeddingCollection(
            _collection(), torch.device("cpu"), _LR, _EPS
        )
        self.addCleanup(patch.stopall)
        _patch_ops(self.sc)

    def test_lookups_return_each_token_row_with_its_lengths(self) -> None:
        out = self.sc(_features())
        self.assertEqual(list(out), ["a", "b", "c"])
        shared, own = _logical("shared", 10, 8), _logical("own", 6, 8)
        torch.testing.assert_close(out["a"].values(), shared[[1, 3, 1]])
        torch.testing.assert_close(out["b"].values(), shared[[7, 3]])
        torch.testing.assert_close(out["c"].values(), own[[2, 5, 0]])
        torch.testing.assert_close(out["b"].lengths(), torch.tensor([1, 1]))

    def test_backward_applies_rowwise_adagrad_to_each_table(self) -> None:
        next(iter(self.sc.optimizer.param_groups))["lr"] = 0.5
        out = self.sc(_features())
        grads = {
            name: torch.full_like(jt.values(), i + 1.0)
            for i, (name, jt) in enumerate(out.items())
        }
        torch.autograd.backward(
            [out[n].values() for n in grads], [grads[n] for n in grads]
        )
        # Row 1 of "shared" is looked up twice by feature "a", row 3 by "a" and "b".
        shared = _logical("shared", 10, 8)
        for row, summed in ((1, 2.0), (3, 1.0 + 2.0), (7, 2.0)):
            acc = summed**2
            expected = shared[row] - 0.5 * summed / (acc**0.5 + _EPS)
            physical = _physical(
                torch.tensor([row]), self.sc.weights["shared"].shape[0]
            )
            torch.testing.assert_close(self.sc.weights["shared"][physical][0], expected)
        untouched = _physical(torch.tensor([4]), self.sc.weights["shared"].shape[0])
        torch.testing.assert_close(self.sc.weights["shared"][untouched][0], shared[4])

    def test_lookups_are_split_when_a_partition_exceeds_the_cap(self) -> None:
        reference = SparseCoreEmbeddingCollection(
            _collection(), torch.device("cpu"), _LR, _EPS
        )
        _patch_ops(reference)
        self.sc._caps["shared"] = 1
        names = ("a", "b", "c")
        split, unsplit = self.sc(_features()), reference(_features())
        # The distinct ids 1, 3 and 7 of "shared" all sit on shard 1. One chunk
        # puts 1 and 7 on SparseCore 0; with two, chunk 0 takes 1 and chunk 1
        # takes 3 and 7, one per SparseCore.
        lookups = [
            c.args
            for c in cast(Mock, self.sc._compiled_fn).call_args_list
            if c.args[0] == "lookup"
        ]
        self.assertEqual(sum(1 for c in lookups if c[1] == "shared"), 2)
        for name in names:
            torch.testing.assert_close(split[name].values(), unsplit[name].values())
        for out in (split, unsplit):
            values = [out[n].values() for n in names]
            torch.autograd.backward(values, [torch.ones_like(v) for v in values])
        torch.testing.assert_close(
            self.sc.weights["shared"], reference.weights["shared"]
        )

    def test_limits_fit_the_tile_memory(self) -> None:
        self.assertEqual(torchtpu_sparsecore._max_ids_cap(512), 224)
        self.assertEqual(torchtpu_sparsecore._max_ids_cap(64), 1840)
        limits = _Limits.for_chunk(samples=64, max_ids=224, world=8)
        self.assertEqual(limits.buffer, 224 * NUM_SC * 8 * NUM_SC)

    def test_chunk_count_raises_when_no_chunking_fits(self) -> None:
        self.sc._caps["shared"] = 0
        with self.assertRaisesRegex(RuntimeError, "do not fit"):
            self.sc(_features())

    def test_ids_outside_the_table_raise(self) -> None:
        for bad in (10, -1, 2**32 + 1):
            features = KeyedJaggedTensor(
                keys=["a", "b", "c"],
                values=torch.tensor([1, bad, 1, 7, 3, 2, 5, 0]),
                lengths=torch.tensor([2, 1, 1, 1, 2, 1]),
            )
            with self.assertRaisesRegex(ValueError, r"ids of shared must be in"):
                self.sc(features)

    def test_a_changed_id_count_raises(self) -> None:
        self.sc(_features())
        fewer = KeyedJaggedTensor(
            keys=["a", "b", "c"],
            values=torch.tensor([1, 3, 1, 7, 2, 5, 0]),
            lengths=torch.tensor([2, 1, 1, 1, 1, 1]),
        )
        with self.assertRaisesRegex(RuntimeError, "fixed number of ids"):
            self.sc(fewer)

    def test_calls_run_under_the_scoped_recompile_limit(self) -> None:
        seen = []

        def recording(kind: str, table: str, limits: _Limits) -> Callable[..., Any]:
            fn = _stand_in_ops(kind, table, limits)

            def call(*args: Any) -> Any:
                seen.append(torch._dynamo.config.recompile_limit)
                return fn(*args)

            return call

        cast(Mock, self.sc._compiled_fn).side_effect = recording
        before = torch._dynamo.config.recompile_limit
        out = self.sc(_features())
        values = [jt.values() for jt in out.values()]
        torch.autograd.backward(values, [torch.ones_like(v) for v in values])
        # Two tables, each with up to seven chunk counts (1, 2, ..., 64).
        self.assertEqual(seen, [14] * 4)
        self.assertEqual(torch._dynamo.config.recompile_limit, before)


class ChunkLayoutTest(unittest.TestCase):
    def test_chunks_take_whole_blocks_and_alternate_sparsecores(self) -> None:
        shards, samples = 4, 16
        unique = torch.arange(16, dtype=torch.int32)
        chunk, sample = torchtpu_sparsecore._chunk_layout(unique, shards, 2, samples)
        self.assertEqual(unique[chunk == 0].tolist(), [0, 1, 2, 3, 8, 9, 10, 11])
        for j in range(2):
            ids = unique[chunk == j]
            # Every shard gets two ids of each chunk, as many as with no split.
            self.assertEqual(torch.bincount(ids % shards).tolist(), [2, 2, 2, 2])
            # Each SparseCore (half of the samples) gets half of the chunk.
            half = sample[chunk == j] // (samples // NUM_SC)
            self.assertEqual(torch.bincount(half).tolist(), [4, 4])
        ids, offsets = torchtpu_sparsecore._csr(
            unique[chunk == 0], sample[chunk == 0], samples
        )
        self.assertEqual(ids.tolist(), [0, 2, 8, 10, 1, 3, 9, 11])
        self.assertEqual(
            offsets.tolist(), [0, 1, 2, 3, 4, 4, 4, 4, 4, 5, 6, 7, 8, 8, 8, 8, 8]
        )

    def test_an_id_takes_the_same_chunk_on_every_process(self) -> None:
        a = torch.tensor([3, 5, 7, 9], dtype=torch.int32)
        b = torch.tensor([2, 7, 8, 9], dtype=torch.int32)
        chunk_a, _ = torchtpu_sparsecore._chunk_layout(a, 2, 2, 16)
        chunk_b, _ = torchtpu_sparsecore._chunk_layout(b, 2, 2, 16)
        # Ids 7 and 9: positions 2 and 3 on one process, 1 and 3 on the other.
        self.assertEqual(chunk_a[[2, 3]].tolist(), chunk_b[[1, 3]].tolist())


class TorchTPUSparseCoreBackendTest(unittest.TestCase):
    def _setup(self, world_size: int, optim_type: OptimType) -> Any:
        model = nn.Module()
        model.ec = _collection()
        model.dense = nn.Linear(2, 2)
        config = type(
            "Config",
            (),
            {
                "optim_type": optim_type,
                "lr": _LR,
                "eps": _EPS,
                "lr_warmup_schedule": [],
            },
        )
        return TorchTPUSparseCoreBackend("ec").setup(
            model, [], config(), torch.device("cpu"), world_size
        )

    def test_setup_swaps_in_the_sparsecore_collection(self) -> None:
        result = self._setup(1, OptimType.ROWWISE_ADAGRAD)
        self.assertIsInstance(result.model.ec, SparseCoreEmbeddingCollection)
        self.assertEqual(
            set(result.dense_named_parameters), {"dense.weight", "dense.bias"}
        )
        self.assertIs(result.sparse_runtime.fused_optimizer, result.model.ec.optimizer)

    def test_setup_refuses_other_optimizers(self) -> None:
        with self.assertRaises(NotImplementedError):
            self._setup(1, OptimType.ADAM)

    def test_tables_use_the_seeded_init(self) -> None:
        sc = self._setup(1, OptimType.ROWWISE_ADAGRAD).model.ec
        torch.testing.assert_close(
            sc.weights["shared"], _mod_sharded("shared", 10, 8, shards=NUM_SC)
        )


class LocalShardTest(unittest.TestCase):
    def test_process_shards_assemble_the_mod_sharded_table(self) -> None:
        config = _collection().embedding_configs()[0]
        world = 2
        rows = torchtpu_sparsecore._round_up(10, 8 * world * NUM_SC)
        with patch.object(torchtpu_sparsecore, "_INIT_BLOCK_ROWS", 3):
            shards = [
                torchtpu_sparsecore._local_shard(config, rows, owner, world)
                for owner in range(world)
            ]
        torch.testing.assert_close(
            torch.cat(shards), _mod_sharded("shared", 10, 8, shards=world * NUM_SC)
        )

    def test_shards_draw_from_the_configs_uniform_init_fn(self) -> None:
        config = EmbeddingConfig(
            name="shared",
            embedding_dim=8,
            num_embeddings=10,
            feature_names=["a"],
            init_fn=functools.partial(torch.nn.init.uniform_, a=-0.5, b=0.5),
        )
        rows = torchtpu_sparsecore._round_up(10, 8 * NUM_SC)
        torch.testing.assert_close(
            torchtpu_sparsecore._local_shard(config, rows, 0, 1),
            _mod_sharded("shared", 10, 8, shards=NUM_SC, bound=0.5),
        )


class TableChecksTest(unittest.TestCase):
    def _tables(self, **kwargs: Any) -> EmbeddingCollection:
        config = EmbeddingConfig(
            name="t", embedding_dim=8, feature_names=["a"], **kwargs
        )
        return EmbeddingCollection(tables=[config], device=torch.device("meta"))

    def test_tables_must_have_int32_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "int32"):
            SparseCoreEmbeddingCollection(
                self._tables(num_embeddings=2**31 + 1), torch.device("cpu"), _LR, _EPS
            )

    def test_tables_must_be_fp32(self) -> None:
        with self.assertRaisesRegex(ValueError, "FP32"):
            SparseCoreEmbeddingCollection(
                self._tables(num_embeddings=10, data_type=DataType.FP16),
                torch.device("cpu"),
                _LR,
                _EPS,
            )

    def test_other_initializers_raise(self) -> None:
        with self.assertRaises(NotImplementedError):
            SparseCoreEmbeddingCollection(
                self._tables(num_embeddings=10, init_fn=torch.nn.init.normal_),
                torch.device("cpu"),
                _LR,
                _EPS,
            )
