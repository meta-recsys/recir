# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""CPU tests for the pooled path through ``InGraphSparseArch``.

The backend's job for a pooled table is to line two things up: the group's ids,
concatenated along the token axis, and the bag lengths, cut the same way. Both
concatenations are feature-major within a chip and the op reads the cuts
positionally, so a mismatch between them silently attributes ids to the wrong
bag -- shapes still fit, values are just wrong.

That is what these cover. The reduction itself is covered at the op layer
(``test_seq_embedding_op_pooling``) and the kernel by the ringshift harness.

They run on CPU because the op falls back to a reference implementation there,
which is enough: a misalignment changes the values, not the layout.
"""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F
from recir.embedding.backend import TableOptimizer, TablePlan, TableSharding
from recir.embedding.in_graph import InGraphSparseArch
from torchrec.modules.embedding_configs import (
    EmbeddingBagConfig,
    EmbeddingConfig,
    PoolingType,
)


_WORLD = 2
_VOCAB = 16
_DIM = 4

# Bag 0 full-ish, bag 1 a singleton, bag 2 EMPTY, bag 3 the widest. Forced
# rather than drawn: the empty bag is what the mean divisor clamp exists for and
# a random split would usually miss it.
_LENGTHS = torch.tensor([2, 1, 0, 3])
# Chip 0 owns samples 0-1 (2+1 = 3 tokens), chip 1 owns samples 2-3 (0+3 = 3).
# The per-chip total being equal is the module's static-shape invariant.
_IDS_2D = torch.tensor([[5, 9, 2], [7, 1, 5]])
_T_LOCAL = 3


def _bag_config(name: str, pooling: PoolingType, rows: int = _VOCAB):
    return EmbeddingBagConfig(
        name=name,
        embedding_dim=_DIM,
        num_embeddings=rows,
        feature_names=[f"f_{name}"],
        pooling=pooling,
    )


def _arch(
    configs, sharding: TableSharding = TableSharding.ROWWISE
) -> InGraphSparseArch:
    torch.manual_seed(0)
    keys = [f for c in configs for f in c.feature_names]
    return InGraphSparseArch(
        table_configs=configs,
        feature_seqlens={k: _T_LOCAL for k in keys},
        world_size=_WORLD,
        table_plans={
            config.name: TablePlan(sharding, TableOptimizer.SPARSE)
            for config in configs
        },
        on_device_init=False,
    )


def _embedding_bag_reference(
    ids_2d: torch.Tensor, lengths: torch.Tensor, weight: torch.Tensor, mode: str
) -> torch.Tensor:
    """``F.embedding_bag`` over the same bags, straight from the flat stream.

    Built from the flat stream and offsets, so it shares no code with the arch:
    a mis-cut there cannot be mirrored here.
    """
    flat = ids_2d.reshape(-1).long()
    offsets = torch.cat([torch.zeros(1, dtype=torch.long), lengths.cumsum(0)[:-1]])
    return F.embedding_bag(flat, weight, offsets, mode=mode)


class PooledForwardTest(unittest.TestCase):
    def _run(self, pooling: PoolingType):
        cfg = _bag_config("t_bag", pooling)
        arch = _arch([cfg])
        out = arch._forward_presliced(["f_t_bag"], [_IDS_2D], [_LENGTHS])
        return out["f_t_bag"], arch._sharded_tables["t_bag"]

    def test_sum_matches_embedding_bag(self) -> None:
        jt, weight = self._run(PoolingType.SUM)
        torch.testing.assert_close(
            jt.values(), _embedding_bag_reference(_IDS_2D, _LENGTHS, weight, "sum")
        )

    def test_mean_matches_embedding_bag(self) -> None:
        jt, weight = self._run(PoolingType.MEAN)
        torch.testing.assert_close(
            jt.values(), _embedding_bag_reference(_IDS_2D, _LENGTHS, weight, "mean")
        )

    def test_output_is_one_row_per_sample(self) -> None:
        """Pooling collapsed the variable length, so every bag now holds one row."""
        jt, _weight = self._run(PoolingType.SUM)
        self.assertEqual(jt.values().shape, (_LENGTHS.shape[0], _DIM))
        torch.testing.assert_close(jt.lengths(), torch.ones_like(_LENGTHS))

    def test_empty_bag_is_zero_not_nan(self) -> None:
        for pooling in (PoolingType.SUM, PoolingType.MEAN):
            jt, _weight = self._run(pooling)
            self.assertTrue(torch.all(jt.values()[2] == 0), pooling)
            self.assertTrue(torch.isfinite(jt.values()).all(), pooling)

    def test_gradient_reaches_the_table(self) -> None:
        cfg = _bag_config("t_bag", PoolingType.SUM)
        arch = _arch([cfg])
        out = arch._forward_presliced(["f_t_bag"], [_IDS_2D], [_LENGTHS])
        out["f_t_bag"].values().sum().backward()
        weight = arch._sharded_tables["t_bag"]
        self.assertIsNotNone(weight.grad)
        # Rows 5, 9, 2, 7, 1 were touched; id 5 appears in two bags.
        self.assertEqual(float(weight.grad[5].sum()), 2.0 * _DIM)
        self.assertEqual(float(weight.grad[0].abs().sum()), 0.0)


class MixedTableTest(unittest.TestCase):
    """A pooled and a sequence table in the same arch.

    Worth its own test because the two take different id layouts, so they are
    grouped and called separately -- a regrouping bug would send one table's
    features down the other's path and only surface as a shape error.
    """

    def _arch(self) -> InGraphSparseArch:
        return _arch(
            [
                _bag_config("t_bag", PoolingType.SUM),
                EmbeddingConfig(
                    name="t_seq",
                    embedding_dim=_DIM,
                    num_embeddings=_VOCAB,
                    feature_names=["f_t_seq"],
                ),
            ]
        )

    def test_both_features_come_back_with_their_own_shape(self) -> None:
        arch = self._arch()
        out = arch._forward_presliced(
            ["f_t_bag", "f_t_seq"], [_IDS_2D, _IDS_2D], [_LENGTHS, _LENGTHS]
        )
        self.assertEqual(set(out), {"f_t_bag", "f_t_seq"})
        # Pooled: one row per sample. Sequence: one row per token.
        self.assertEqual(out["f_t_bag"].values().shape, (_LENGTHS.shape[0], _DIM))
        self.assertEqual(out["f_t_seq"].values().shape, (_WORLD * _T_LOCAL, _DIM))

    def test_the_sequence_feature_is_untouched_by_the_pooled_one(self) -> None:
        """No regression to the existing path: same values with and without a
        pooled table alongside."""
        mixed_arch = self._arch()
        seq_arch = _arch(
            [
                EmbeddingConfig(
                    name="t_seq",
                    embedding_dim=_DIM,
                    num_embeddings=_VOCAB,
                    feature_names=["f_t_seq"],
                )
            ]
        )
        # Tables init in config order off one seeded RNG, so the pooled table
        # sitting ahead of t_seq shifts the stream and the two t_seq tables come
        # out different. Copy one onto the other: the comparison is about the
        # grouping, not the init.
        with torch.no_grad():
            seq_arch._sharded_tables["t_seq"].copy_(mixed_arch._sharded_tables["t_seq"])
        mixed = mixed_arch._forward_presliced(
            ["f_t_bag", "f_t_seq"], [_IDS_2D, _IDS_2D], [_LENGTHS, _LENGTHS]
        )
        seq_only = seq_arch._forward_presliced(["f_t_seq"], [_IDS_2D], [_LENGTHS])
        torch.testing.assert_close(
            mixed["f_t_seq"].values(), seq_only["f_t_seq"].values()
        )


class PooledRejectionTest(unittest.TestCase):
    """The two ways a pooled table can be mis-wired, both caught by name."""

    def test_a_pooled_table_not_planned_rowwise_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "pooling needs ROWWISE"):
            _arch(
                [_bag_config("t_small", PoolingType.SUM)],
                sharding=TableSharding.FSDP,
            )

    def test_a_batch_the_chips_do_not_divide_is_rejected(self) -> None:
        """Bag rows are assigned to chips contiguously, so an indivisible batch
        would put part of a sample's bag on the wrong chip."""
        arch = _arch([_bag_config("t_bag", PoolingType.SUM)])
        odd_lengths = torch.tensor([2, 1, 0, 3, 0])  # 5 bags over 2 chips
        with self.assertRaisesRegex(ValueError, "not divisible by the 2 chips"):
            arch._forward_presliced(["f_t_bag"], [_IDS_2D], [odd_lengths])
