# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import functools
import unittest
from typing import Any, Dict, List
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F
from recir.embedding import torchtpu_sparsecore
from recir.embedding.torchtpu_sparsecore import (
    NUM_SC,
    SparseCoreEmbeddingCollection,
    TorchTPUSparseCoreBackend,
    _compute_coo_buffer_size_per_device,
    _resolve_optimizer_type,
    _round_up,
    _seed_local_stack_shard,
)
from recir.utils.meta_tensor_init import fill_table_shard
from torchrec.distributed.embedding_types import OptimType
from torchrec.modules.embedding_configs import DataType, EmbeddingConfig
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.optim.rowwise_adagrad import RowWiseAdagrad

_LR, _EPS = 0.1, 1e-8


@dataclasses.dataclass
class _StubTable:
    config: EmbeddingConfig

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def num_embeddings(self) -> int:
        return self.config.num_embeddings

    @property
    def embedding_dim(self) -> int:
        return self.config.embedding_dim


@dataclasses.dataclass
class _StubStackConfig:
    stack_name: str
    tables: List[_StubTable]
    stack_num_embeddings: int
    stack_embedding_dim: int
    row_offsets_in_shard: Dict[str, int]
    shard_rotations: Dict[str, int]
    padded_vocab_sizes: Dict[str, int]


def _logical(
    name: str, rows: int, dim: int, bound: float = 0.0
) -> torch.Tensor:
    table = torch.empty(rows, dim)
    bound = bound or (1.0 / rows) ** 0.5
    return fill_table_shard(table, name, (rows, dim), (0, 0), -bound, bound)


def _expected_stacked_sharded(
    table_weights: Dict[str, torch.Tensor],
    stack_cfg: _StubStackConfig,
    num_shards: int,
) -> torch.Tensor:
    rows_per_shard = stack_cfg.stack_num_embeddings // num_shards
    out = torch.zeros(
        (num_shards, rows_per_shard, stack_cfg.stack_embedding_dim),
        dtype=torch.float32,
    )
    for table in stack_cfg.tables:
        name = table.name
        w = table_weights[name]
        padded_vocab = stack_cfg.padded_vocab_sizes[name]
        row_offset = stack_cfg.row_offsets_in_shard[name]
        rot = stack_cfg.shard_rotations[name] % num_shards
        pad_rows = padded_vocab - w.size(0)
        pad_cols = stack_cfg.stack_embedding_dim - w.size(1)
        w_padded = (
            F.pad(w, (0, pad_cols, 0, pad_rows), value=0.0)
            if (pad_rows > 0 or pad_cols > 0)
            else w
        )
        chunk = padded_vocab // num_shards
        sharded = w_padded.view(chunk, num_shards, -1).permute(1, 0, 2).contiguous()
        if rot != 0:
            sharded = torch.roll(sharded, shifts=rot, dims=0)
        out[:, row_offset : row_offset + chunk] = sharded
    return out.reshape(-1, stack_cfg.stack_embedding_dim)


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
                name="own",
                embedding_dim=8,
                num_embeddings=6,
                feature_names=["c"],
            ),
        ]
    )


class TorchTPUSparseCoreTorchRecTest(unittest.TestCase):
    def test_optimizer_mapping_uses_rowwise_adagrad(self) -> None:
        self.assertIs(
            _resolve_optimizer_type(OptimType.ROWWISE_ADAGRAD), RowWiseAdagrad
        )
        self.assertIs(
            _resolve_optimizer_type(OptimType.ADAGRAD), torch.optim.Adagrad
        )
        self.assertIs(_resolve_optimizer_type(OptimType.ADAM), torch.optim.Adam)
        self.assertIs(_resolve_optimizer_type(OptimType.SGD), torch.optim.SGD)

    def test_seed_local_stack_shard_matches_single_and_rotated_stacked_layout(
        self,
    ) -> None:
        configs = _collection().embedding_configs()
        world = 2
        num_shards = world * NUM_SC
        padded_shared = _round_up(10, 8 * num_shards)
        padded_own = _round_up(6, 8 * num_shards)
        stack_cfg = _StubStackConfig(
            stack_name="shared_own",
            tables=[_StubTable(configs[0]), _StubTable(configs[1])],
            stack_num_embeddings=padded_shared + padded_own,
            stack_embedding_dim=8,
            row_offsets_in_shard={
                "shared": 0,
                "own": padded_shared // num_shards,
            },
            shard_rotations={"shared": 0, "own": NUM_SC},
            padded_vocab_sizes={"shared": padded_shared, "own": padded_own},
        )
        with patch.object(torchtpu_sparsecore, "_INIT_BLOCK_ROWS", 3):
            shards = [
                _seed_local_stack_shard(stack_cfg, owner, world, NUM_SC)
                for owner in range(world)
            ]
        assembled = torch.cat(shards, dim=0)
        expected = _expected_stacked_sharded(
            {
                "shared": _logical("shared", 10, 8),
                "own": _logical("own", 6, 8),
            },
            stack_cfg,
            num_shards=num_shards,
        )
        torch.testing.assert_close(assembled, expected, atol=0.0, rtol=0.0)

    def test_seed_local_stack_shard_respects_custom_uniform_init_fn(self) -> None:
        cfg = EmbeddingConfig(
            name="custom",
            embedding_dim=8,
            num_embeddings=10,
            feature_names=["a"],
            init_fn=functools.partial(torch.nn.init.uniform_, a=-0.5, b=0.5),
        )
        padded = _round_up(10, 8 * NUM_SC)
        stack_cfg = _StubStackConfig(
            stack_name="custom",
            tables=[_StubTable(cfg)],
            stack_num_embeddings=padded,
            stack_embedding_dim=8,
            row_offsets_in_shard={"custom": 0},
            shard_rotations={"custom": 0},
            padded_vocab_sizes={"custom": padded},
        )
        shard = _seed_local_stack_shard(stack_cfg, 0, 1, NUM_SC)
        expected = _expected_stacked_sharded(
            {"custom": _logical("custom", 10, 8, bound=0.5)},
            stack_cfg,
            num_shards=NUM_SC,
        )
        torch.testing.assert_close(shard, expected, atol=0.0, rtol=0.0)

    def test_backend_setup_swaps_in_torchrec_sparsecore_collection(self) -> None:
        model = nn.Module()
        model.ec = _collection()
        model.dense = nn.Linear(2, 2)
        cfg = type(
            "Config",
            (),
            {
                "optim_type": OptimType.ROWWISE_ADAGRAD,
                "lr": _LR,
                "eps": _EPS,
                "beta1": 0.9,
                "beta2": 0.999,
                "lr_warmup_schedule": [],
            },
        )()
        result = TorchTPUSparseCoreBackend("ec").setup(
            model, [], cfg, torch.device("cpu"), 1
        )
        self.assertIsInstance(result.model.ec, SparseCoreEmbeddingCollection)
        self.assertEqual(
            set(result.dense_named_parameters), {"dense.weight", "dense.bias"}
        )
        self.assertIs(
            result.sparse_runtime.fused_optimizer, result.model.ec.optimizer
        )

    def test_table_validation_rejects_invalid_dtype_rows_dim_and_init(
        self,
    ) -> None:
        sparse_cfg: Any = type(
            "Config",
            (),
            {"optim_type": OptimType.ROWWISE_ADAGRAD, "lr": _LR, "eps": _EPS},
        )()
        with self.assertRaisesRegex(ValueError, "divide by 8"):
            SparseCoreEmbeddingCollection(
                [
                    EmbeddingConfig(
                        name="t",
                        embedding_dim=10,
                        num_embeddings=16,
                        feature_names=["a"],
                    )
                ],
                sparse_cfg,
                torch.device("cpu"),
                1,
            )
        with self.assertRaisesRegex(ValueError, "int32"):
            SparseCoreEmbeddingCollection(
                [
                    EmbeddingConfig(
                        name="t",
                        embedding_dim=8,
                        num_embeddings=2**31 + 1,
                        feature_names=["a"],
                    )
                ],
                sparse_cfg,
                torch.device("cpu"),
                1,
            )
        with self.assertRaisesRegex(ValueError, "FP32"):
            SparseCoreEmbeddingCollection(
                [
                    EmbeddingConfig(
                        name="t",
                        embedding_dim=8,
                        num_embeddings=16,
                        feature_names=["a"],
                        data_type=DataType.FP16,
                    )
                ],
                sparse_cfg,
                torch.device("cpu"),
                1,
            )
        with self.assertRaises(NotImplementedError):
            SparseCoreEmbeddingCollection(
                [
                    EmbeddingConfig(
                        name="t",
                        embedding_dim=8,
                        num_embeddings=16,
                        feature_names=["a"],
                        init_fn=torch.nn.init.normal_,
                    )
                ],
                sparse_cfg,
                torch.device("cpu"),
                1,
            )

    def test_coo_buffer_size_calculation(self) -> None:
        self.assertEqual(
            _compute_coo_buffer_size_per_device(224, 0, 8, 2, 8, False),
            224 * 2 * 16,
        )
        self.assertEqual(
            _compute_coo_buffer_size_per_device(224, 300, 8, 2, 8, False),
            304,
        )


if __name__ == "__main__":
    unittest.main()
