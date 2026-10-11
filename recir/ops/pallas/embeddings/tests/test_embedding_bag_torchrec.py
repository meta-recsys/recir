# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""CPU parity between the pooled (EBC) layer and ``torchrec.EmbeddingBagCollection``.

This is the reference the real-TPU correctness harness diffs against, so it is
worth pinning down *before* spending TPU time: if the torchrec reference and the
MOD-shard assembly disagree here, a TPU mismatch is unattributable.

What is and is not covered
--------------------------
The fused Pallas kernel cannot run on CPU (it uses runtime-extent slices that
``interpret=True`` cannot lower), so the id -> row lookup is stood in for by
:func:`_sharded_gather`, a plain-JAX gather with the *same* MOD-shard semantics
(``id i`` lives at ``shard[i % W][i // W]``). That covers everything new in the
EBC layer:

  * the shard-assembly mapping (the #1 source of phantom "kernel bugs"),
  * pooled forward for sum and mean,
  * pooled backward, including duplicate ids accumulating into one table row --
    ``jnp.take``'s VJP is a scatter-add, exactly what the fused backward does.

What it does not cover is the kernel's own forward/backward, which already have
real-TPU tests under ``examples/embedding/``.

Tolerance note: parity is asserted at ``atol=rtol=1e-5``, not bitwise. The
non-pooled EC test asserts bitwise equality because that path is a pure gather
with zero arithmetic; adding a reduction makes the float addition order differ
from torch's, so bitwise equality is no longer the right bar.
"""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np
import torch
from recir.ops.pallas.embeddings.ringshift import pool_per_rank
from torchrec.modules.embedding_configs import EmbeddingBagConfig, PoolingType
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

_WORLD_SIZE = 4
_V_LOCAL = 5
_VOCAB = _WORLD_SIZE * _V_LOCAL
_DIM = 3
_NUM_BAGS = 3
_MAX_LOCAL_IDS = 8

# Per-rank bags, hand-picked to cover the cases that break naive pooling:
#   rank 0 -- id 7 appears twice in bag 0 AND once in bag 2 (within-bag and
#             cross-bag duplicates must both accumulate into table row 7),
#   rank 2 -- two leading empty bags,
#   rank 3 -- a trailing empty bag, plus a repeated hot row (19).
_BAG_LENGTHS = np.array(
    [
        [3, 0, 2],
        [1, 2, 1],
        [0, 0, 4],
        [2, 2, 0],
    ],
    dtype=np.int32,
)
_VALID_IDS = [
    [7, 7, 2, 19, 7],
    [0, 5, 5, 13],
    [1, 2, 3, 4],
    [19, 19, 18, 17],
]


def _padded_ids() -> np.ndarray:
    ids = np.zeros((_WORLD_SIZE, _MAX_LOCAL_IDS), dtype=np.int32)
    for r, row in enumerate(_VALID_IDS):
        ids[r, : len(row)] = row
    return ids


def _mod_shard(full_table: np.ndarray) -> np.ndarray:
    """``[V, D]`` -> ``[W, V_local, D]`` with global row ``i`` at ``[i % W, i // W]``."""
    return np.stack(
        [full_table[np.arange(_V_LOCAL) * _WORLD_SIZE + r] for r in range(_WORLD_SIZE)]
    )


def _sharded_gather(
    ids: jax.Array, id_count: jax.Array, sharded_table: jax.Array
) -> jax.Array:
    """MOD-sharded per-id lookup, matching the fused kernel's contract.

    Stands in for ``ringshift_embedding_lookup`` on CPU. Padding slots are
    zeroed to match the kernel's output, though :func:`pool_per_rank` routes
    them to its trash segment and does not rely on that.
    """
    valid = jnp.arange(ids.shape[0]) < id_count
    world_size = sharded_table.shape[0]
    safe_ids = jnp.where(valid, ids, 0)
    emb = sharded_table[safe_ids % world_size, safe_ids // world_size]
    return jnp.where(valid[:, None], emb, 0.0)


def _jax_forward(
    sharded_table: jax.Array,
    ids: jax.Array,
    bag_lengths: jax.Array,
    pooling: str,
) -> jax.Array:
    """``[W, num_bags, D]`` pooled output for all ranks."""
    id_count = jnp.sum(bag_lengths, axis=1)

    def per_rank(rank_ids, rank_count, rank_lengths):
        per_id = _sharded_gather(rank_ids, rank_count, sharded_table)
        return pool_per_rank(per_id, rank_lengths, pooling=pooling)

    return jax.vmap(per_rank, in_axes=(0, 0, 0))(ids, id_count, bag_lengths)


def _torchrec_forward_and_grad(
    full_table: np.ndarray, seed: np.ndarray, pooling: str
) -> tuple[np.ndarray, np.ndarray]:
    """Run torchrec EBC per rank; return pooled outputs and the table gradient.

    All ranks' losses are summed before a single ``backward()`` so the table
    gradient accumulates across ranks -- the same thing the distributed backward
    produces once every rank's contribution has landed on the owning shard.
    """
    pooling_type = PoolingType.SUM if pooling == "sum" else PoolingType.MEAN
    config = EmbeddingBagConfig(
        name="t",
        embedding_dim=_DIM,
        num_embeddings=_VOCAB,
        feature_names=["f"],
        pooling=pooling_type,
    )
    ebc = EmbeddingBagCollection(tables=[config], device=torch.device("cpu"))
    # nn.Module.__getattr__ is typed as a Module|Tensor union, so narrow once
    # here and reuse the handle for both the weight write and the grad read.
    weight = ebc.embedding_bags["t"].weight
    assert isinstance(weight, torch.Tensor)
    with torch.no_grad():
        weight.copy_(torch.from_numpy(full_table))

    outputs = []
    loss = torch.zeros((), dtype=torch.float32)
    for rank in range(_WORLD_SIZE):
        kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=["f"],
            values=torch.tensor(_VALID_IDS[rank], dtype=torch.int64),
            lengths=torch.tensor(_BAG_LENGTHS[rank], dtype=torch.int64),
        )
        pooled = ebc(kjt)["f"]  # [num_bags, D]
        outputs.append(pooled)
        loss = loss + (pooled * torch.from_numpy(seed[rank])).sum()

    loss.backward()
    grad = weight.grad
    assert grad is not None
    return (
        torch.stack(outputs).detach().numpy(),
        grad.detach().numpy(),
    )


class EmbeddingBagTorchrecParityTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        self.full_table = rng.standard_normal((_VOCAB, _DIM)).astype(np.float32)
        self.sharded_table = _mod_shard(self.full_table)
        self.ids = _padded_ids()
        # A non-uniform upstream gradient, so a wrong fan-out cannot cancel out
        # the way an all-ones seed would let it.
        self.seed = rng.standard_normal((_WORLD_SIZE, _NUM_BAGS, _DIM)).astype(
            np.float32
        )

    def test_mod_shard_assembly_round_trips(self) -> None:
        """Guard the mapping itself before trusting anything built on it."""
        for global_id in range(_VOCAB):
            np.testing.assert_array_equal(
                self.sharded_table[global_id % _WORLD_SIZE, global_id // _WORLD_SIZE],
                self.full_table[global_id],
            )

    def _check(self, pooling: str) -> None:
        expected_fwd, expected_grad_full = _torchrec_forward_and_grad(
            self.full_table, self.seed, pooling
        )

        def loss_fn(table: jax.Array) -> jax.Array:
            pooled = _jax_forward(
                table, jnp.array(self.ids), jnp.array(_BAG_LENGTHS), pooling
            )
            return jnp.sum(pooled * jnp.array(self.seed))

        actual_fwd = _jax_forward(
            jnp.array(self.sharded_table),
            jnp.array(self.ids),
            jnp.array(_BAG_LENGTHS),
            pooling,
        )
        np.testing.assert_allclose(
            np.array(actual_fwd), expected_fwd, atol=1e-5, rtol=1e-5
        )

        actual_grad = np.array(jax.grad(loss_fn)(jnp.array(self.sharded_table)))
        np.testing.assert_allclose(
            actual_grad, _mod_shard(expected_grad_full), atol=1e-5, rtol=1e-5
        )

    def test_sum_forward_and_table_grad(self) -> None:
        self._check("sum")

    def test_mean_forward_and_table_grad(self) -> None:
        self._check("mean")

    def test_out_of_range_padding_ids_are_ignored(self) -> None:
        poisoned_ids = self.ids.copy()
        for rank, lengths in enumerate(_BAG_LENGTHS):
            poisoned_ids[rank, int(lengths.sum()) :] = _VOCAB + 100
        expected = _jax_forward(
            jnp.array(self.sharded_table),
            jnp.array(self.ids),
            jnp.array(_BAG_LENGTHS),
            "sum",
        )
        actual = _jax_forward(
            jnp.array(self.sharded_table),
            jnp.array(poisoned_ids),
            jnp.array(_BAG_LENGTHS),
            "sum",
        )
        np.testing.assert_array_equal(np.array(actual), np.array(expected))

    def test_hot_row_gradient_accumulates(self) -> None:
        """Row 7 is looked up 3x on rank 0; its grad must be the sum of all three.

        A pooled backward that forgets to accumulate duplicates still passes a
        forward-only test, so this is checked explicitly rather than left to the
        aggregate comparison.
        """
        _, grad_full = _torchrec_forward_and_grad(self.full_table, self.seed, "sum")
        # Bag 0 (rows 0,1) and bag 2 (row 3) on rank 0 all hit id 7.
        expected_row7 = 2.0 * self.seed[0, 0] + 1.0 * self.seed[0, 2]
        np.testing.assert_allclose(grad_full[7], expected_row7, atol=1e-5, rtol=1e-5)
