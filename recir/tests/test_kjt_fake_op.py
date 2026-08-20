#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.


"""Unit tests for recir.ops.fake.kjt.

Verifies that:
  * The kjt_split_per_feature custom op produces correct per-key index
    tensors when invoked with real tensors.
  * Its fake (Meta) kernel produces SymInt-sized outputs under
    FakeTensorMode so make_fx(tracing_mode="fake") can proceed.
  * patch_ec_for_tracing(model) replaces EmbeddingCollection.forward in
    a way that is numerically equivalent to the original on real tensors.
  * The patched EC is FakeTensor-traceable end-to-end (the original is
    NOT, due to KJT.to_dict's .tolist() call).
"""

import unittest
from typing import Dict, List

import torch
from functorch import make_fx
from recir.ops.fake.kjt import _register_kjt_split_op, patch_ec_for_tracing
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor


def _make_kjt(
    keys: List[str],
    per_key_lengths: List[List[int]],
    vocab_size: int,
    seed: int = 0,
) -> KeyedJaggedTensor:
    """Build a KJT with the given per-key lengths.

    ``per_key_lengths[k]`` is the list of per-sample lengths for feature k.
    All keys must share the same stride (= len(per_key_lengths[k])).
    """
    stride = len(per_key_lengths[0])
    for lk in per_key_lengths:
        assert len(lk) == stride, "all keys must share stride"
    flat_lengths = torch.tensor(
        [length for key_lengths in per_key_lengths for length in key_lengths],
        dtype=torch.int64,
    )
    total = int(flat_lengths.sum().item())
    g = torch.Generator().manual_seed(seed)
    values = torch.randint(0, vocab_size, (total,), generator=g, dtype=torch.int64)
    return KeyedJaggedTensor(keys=keys, values=values, lengths=flat_lengths)


def _make_ec(
    table_to_features: Dict[str, List[str]],
    num_embeddings: int = 16,
    embedding_dim: int = 4,
) -> EmbeddingCollection:
    """Build an EmbeddingCollection with the given table -> features mapping."""
    configs = [
        EmbeddingConfig(
            name=table_name,
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            feature_names=feats,
        )
        for table_name, feats in table_to_features.items()
    ]
    return EmbeddingCollection(tables=configs, device=torch.device("cpu"))


class KjtSplitPerFeatureTest(unittest.TestCase):
    """Tests for the kjt_split_per_feature custom op."""

    def setUp(self) -> None:
        _register_kjt_split_op()

    def test_real_impl_matches_manual_split(self) -> None:
        """Real impl produces the same per-key slices as a manual split."""
        kjt = _make_kjt(
            keys=["f0", "f1", "f2"],
            per_key_lengths=[[2, 3], [1, 0], [4, 2]],
            vocab_size=100,
            seed=42,
        )
        out = torch.ops.recir_torchrec.kjt_split_per_feature(
            kjt.values(), kjt.offsets(), kjt.stride(), len(kjt.keys())
        )
        # Manual: per-key slices of values along key-major lengths layout.
        # f0: lengths [2, 3] -> values[0:5]
        # f1: lengths [1, 0] -> values[5:6]
        # f2: lengths [4, 2] -> values[6:12]
        self.assertEqual(len(out), 3)
        self.assertTrue(torch.equal(out[0], kjt.values()[0:5]))
        self.assertTrue(torch.equal(out[1], kjt.values()[5:6]))
        self.assertTrue(torch.equal(out[2], kjt.values()[6:12]))

    def test_fake_kernel_emits_symint_sizes(self) -> None:
        """Fake kernel produces tensors with SymInt-sized first dim."""
        with FakeTensorMode(shape_env=ShapeEnv()):
            values = torch.empty(100, dtype=torch.int64)
            offsets = torch.empty(7, dtype=torch.int64)  # 3 keys * stride 2 + 1
            out = torch.ops.recir_torchrec.kjt_split_per_feature(values, offsets, 2, 3)
            self.assertEqual(len(out), 3)
            # Each output's size[0] should be a SymInt (i.e., not a plain int).
            for t in out:
                size = t.size(0)
                self.assertNotIsInstance(
                    size,
                    int,
                    f"Expected SymInt for fake-kernel output size, got int: {size}",
                )


class PatchEcForTracingTest(unittest.TestCase):
    """Tests for patch_ec_for_tracing."""

    def test_numerical_match_with_original(self) -> None:
        """Patched EC produces the same output as the original on real tensors."""
        ec_orig = _make_ec(
            table_to_features={
                "table_a": ["f0", "f1"],
                "table_b": ["f2"],
            },
            num_embeddings=32,
            embedding_dim=8,
        )
        # Deep-copy to a separate instance, share weights so outputs match.
        import copy

        ec_patched = copy.deepcopy(ec_orig)
        patched_count = patch_ec_for_tracing(ec_patched)
        self.assertEqual(patched_count, 1)

        kjt = _make_kjt(
            keys=["f0", "f1", "f2"],
            per_key_lengths=[[2, 3], [1, 4], [0, 2]],
            vocab_size=32,
            seed=7,
        )

        # Original may fail under FakeTensorMode but should run on real tensors.
        out_orig: Dict[str, JaggedTensor] = ec_orig(kjt)
        out_patched: Dict[str, JaggedTensor] = ec_patched(kjt)

        self.assertEqual(set(out_orig.keys()), set(out_patched.keys()))
        for key in out_orig:
            self.assertTrue(
                torch.equal(out_orig[key].values(), out_patched[key].values()),
                f"values mismatch for feature {key}",
            )
            self.assertTrue(
                torch.equal(out_orig[key].lengths(), out_patched[key].lengths()),
                f"lengths mismatch for feature {key}",
            )

    def test_patch_is_idempotent(self) -> None:
        """Calling patch_ec_for_tracing twice does not double-patch."""
        ec = _make_ec(table_to_features={"t": ["f"]})
        first = patch_ec_for_tracing(ec)
        second = patch_ec_for_tracing(ec)
        self.assertEqual(first, 1)
        self.assertEqual(second, 0)

    def test_make_fx_fake_traces_patched_ec(self) -> None:
        """Patched EC traces through make_fx(tracing_mode="fake") end-to-end."""
        ec: EmbeddingCollection = _make_ec(
            table_to_features={"t0": ["f0"], "t1": ["f1"]},
            num_embeddings=16,
            embedding_dim=4,
        )
        patch_ec_for_tracing(ec)

        kjt = _make_kjt(
            keys=["f0", "f1"],
            per_key_lengths=[[3, 2], [1, 4]],
            vocab_size=16,
            seed=11,
        )

        def fwd(values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
            inner_kjt = KeyedJaggedTensor(
                keys=["f0", "f1"], values=values, lengths=lengths
            )
            out = ec(inner_kjt)
            # Sum to a single tensor for a clean trace return.
            return out["f0"].values().sum() + out["f1"].values().sum()

        # The original EC fails this with DataDependentOutputException.
        # The patched EC should succeed.
        gm = make_fx(fwd, tracing_mode="fake", _allow_non_fake_inputs=True)(
            kjt.values(), kjt.lengths()
        )
        self.assertIsNotNone(gm)
        self.assertIsNotNone(gm.graph)
        # The graph should contain the custom-op split + at least 2
        # aten.embedding nodes (one per feature).
        node_targets = [n.target for n in gm.graph.nodes if n.op == "call_function"]
        self.assertIn(
            torch.ops.recir_torchrec.kjt_split_per_feature.default,
            node_targets,
            "kjt_split_per_feature node should be in the traced graph",
        )
        embedding_count = sum(
            1 for t in node_targets if t == torch.ops.aten.embedding.default
        )
        self.assertGreaterEqual(
            embedding_count,
            2,
            f"Expected >= 2 aten.embedding nodes, got {embedding_count}",
        )


if __name__ == "__main__":
    unittest.main()
