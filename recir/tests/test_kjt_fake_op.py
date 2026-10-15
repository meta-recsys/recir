#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


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
  * The EBC split variant confines data-dependent KJT reads to the opaque op
    and returns per-key indices, offsets, and optional weights.
  * patch_ebc_for_tracing(model) replaces EmbeddingBagCollection.forward
    while preserving unweighted SUM/MEAN outputs, metadata, and weight
    gradients, including empty bags.
  * Fixed-stride EBC forward and parameter backward are FakeTensor-traceable.
  * A traced EBC accepts a different fixed stride at execution time.
  * Weighted EBC forward/backward preserves native behavior.
  * Variable-stride and inverse-index inputs fail explicitly.
"""

import copy
import unittest
from typing import Dict, List
from unittest.mock import patch

import torch
from functorch import make_fx
from recir.ops.fake.kjt import (
    _register_kjt_split_op,
    patch_ebc_for_tracing,
    patch_ec_for_tracing,
)
from torch._dynamo.source import LocalSource
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import (
    DimDynamic,
    ShapeEnv,
    StatelessSymbolicContext,
)
from torchrec.modules.embedding_configs import (
    EmbeddingBagConfig,
    EmbeddingConfig,
    PoolingType,
)
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
)
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor


def _make_kjt(
    keys: List[str],
    per_key_lengths: List[List[int]],
    vocab_size: int,
    seed: int = 0,
    is_weighted: bool = False,
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
    weights = (
        torch.linspace(0.25, 1.25, total, dtype=torch.float32) if is_weighted else None
    )
    return KeyedJaggedTensor(
        keys=keys,
        values=values,
        weights=weights,
        lengths=flat_lengths,
    )


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


def _make_ebc(*, is_weighted: bool = False) -> EmbeddingBagCollection:
    """Build a mixed SUM/MEAN EBC with shared-table and mixed-dim outputs."""
    return EmbeddingBagCollection(
        tables=[
            EmbeddingBagConfig(
                name="table_a",
                embedding_dim=3,
                num_embeddings=32,
                feature_names=["f0", "f1"],
                pooling=PoolingType.SUM,
            ),
            EmbeddingBagConfig(
                name="table_b",
                embedding_dim=2,
                num_embeddings=32,
                feature_names=["f2"],
                pooling=PoolingType.SUM if is_weighted else PoolingType.MEAN,
            ),
        ],
        is_weighted=is_weighted,
        device=torch.device("cpu"),
    )


def _dynamic_fake_args(
    prefix: str,
    values: tuple[torch.Tensor, ...],
) -> tuple[FakeTensorMode, tuple[torch.Tensor, ...]]:
    fake_mode = FakeTensorMode(allow_non_fake_inputs=True, shape_env=ShapeEnv())
    return fake_mode, tuple(
        fake_mode.from_tensor(
            value,
            static_shapes=False,
            source=LocalSource(f"{prefix}_{index}"),
            symbolic_context=StatelessSymbolicContext(
                dynamic_sizes=[DimDynamic.DYNAMIC]
                + [DimDynamic.STATIC] * (value.ndim - 1)
            ),
        )
        for index, value in enumerate(values)
    )


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
            kjt.values(),
            kjt.offsets(),
            kjt.stride(),
            len(kjt.keys()),
        )
        # Manual: per-key slices of values along key-major lengths layout.
        # f0: lengths [2, 3] -> values[0:5]
        # f1: lengths [1, 0] -> values[5:6]
        # f2: lengths [4, 2] -> values[6:12]
        self.assertEqual(len(out), 3)
        self.assertTrue(torch.equal(out[0], kjt.values()[0:5]))
        self.assertTrue(torch.equal(out[1], kjt.values()[5:6]))
        self.assertTrue(torch.equal(out[2], kjt.values()[6:12]))

    def test_real_impl_does_not_read_individual_scalars(self) -> None:
        """Real impl materializes all split boundaries in one batch."""
        kjt = _make_kjt(
            keys=["f0", "f1", "f2"],
            per_key_lengths=[[2, 3], [1, 0], [4, 2]],
            vocab_size=100,
            seed=42,
        )
        with patch.object(
            torch.Tensor,
            "item",
            side_effect=AssertionError("split boundaries must not use Tensor.item"),
        ):
            out = torch.ops.recir_torchrec.kjt_split_per_feature(
                kjt.values(),
                kjt.offsets(),
                kjt.stride(),
                len(kjt.keys()),
            )
        self.assertEqual([tensor.numel() for tensor in out], [5, 1, 6])

    def test_real_impl_uses_runtime_offsets_when_traced_stride_is_stale(self) -> None:
        kjt = _make_kjt(
            keys=["f0", "f1"],
            per_key_lengths=[[2], [3]],
            vocab_size=100,
            seed=43,
        )

        out = torch.ops.recir_torchrec.kjt_split_per_feature(
            kjt.values(), kjt.offsets(), 16, len(kjt.keys())
        )

        self.assertTrue(torch.equal(out[0], kjt.values()[0:2]))
        self.assertTrue(torch.equal(out[1], kjt.values()[2:5]))

    def test_real_lengths_impl_uses_runtime_stride(self) -> None:
        kjt = _make_kjt(
            keys=["f0", "f1"],
            per_key_lengths=[[2], [3]],
            vocab_size=100,
            seed=44,
        )

        out = torch.ops.recir_torchrec.kjt_lengths_per_feature(
            kjt.offsets(), len(kjt.keys())
        )

        torch.testing.assert_close(out[0], torch.tensor([2]))
        torch.testing.assert_close(out[1], torch.tensor([3]))

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

    def test_fixed_stride_split_returns_indices_and_offsets(self) -> None:
        lengths = torch.tensor([2, 1, 0, 1, 2, 1], dtype=torch.int64)
        values = torch.tensor([10, 11, 12, 20, 21, 22, 23], dtype=torch.int64)

        outputs = torch.ops.recir_torchrec.kjt_split_per_feature_with_offsets(
            values,
            lengths,
            None,
            2,
        )

        self.assertEqual(len(outputs), 4)
        torch.testing.assert_close(outputs[0], torch.tensor([10, 11, 12]))
        torch.testing.assert_close(outputs[1], torch.tensor([20, 21, 22, 23]))
        torch.testing.assert_close(outputs[2], torch.tensor([0, 2, 3, 3]))
        torch.testing.assert_close(outputs[3], torch.tensor([0, 1, 3, 4]))

    def test_fixed_stride_split_does_not_read_individual_counts(self) -> None:
        lengths = torch.tensor([2, 1, 0, 1, 2, 1], dtype=torch.int64)
        values = torch.tensor([10, 11, 12, 20, 21, 22, 23], dtype=torch.int64)

        with patch.object(
            torch.Tensor,
            "item",
            side_effect=AssertionError("value counts must not use Tensor.item"),
        ):
            outputs = torch.ops.recir_torchrec.kjt_split_per_feature_with_offsets(
                values,
                lengths,
                None,
                2,
            )

        self.assertEqual([tensor.numel() for tensor in outputs[:2]], [3, 4])

    def test_fixed_stride_split_rejects_uneven_feature_strides(self) -> None:
        with self.assertRaisesRegex(ValueError, "same stride"):
            torch.ops.recir_torchrec.kjt_split_per_feature_with_offsets(
                torch.ones(5, dtype=torch.int64),
                torch.ones(5, dtype=torch.int64),
                None,
                3,
            )

    def test_fixed_stride_fake_offsets_preserve_input_shape_relation(self) -> None:
        fake_mode = FakeTensorMode(shape_env=ShapeEnv())
        with fake_mode:
            values = torch.empty(5, dtype=torch.int64)
            lengths = torch.empty(6, dtype=torch.int64)
            outputs = torch.ops.recir_torchrec.kjt_split_per_feature_with_offsets(
                values,
                lengths,
                None,
                2,
            )

        self.assertEqual(outputs[2].shape[0], lengths.shape[0] // 2 + 1)
        self.assertEqual(outputs[2].shape[0], outputs[3].shape[0])


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

    def test_make_fx_symbolic_trace_accepts_a_different_stride(self) -> None:
        """The traced KJT split derives stride from each runtime input."""
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

        def fwd(
            values: torch.Tensor, lengths: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            inner_kjt = KeyedJaggedTensor(
                keys=["f0", "f1"], values=values, lengths=lengths
            )
            out = ec(inner_kjt)
            value_sum = out["f0"].values().sum() + out["f1"].values().sum()
            stacked_lengths = torch.stack([out["f0"].lengths(), out["f1"].lengths()])
            return value_sum, stacked_lengths

        expected = fwd(kjt.values(), kjt.lengths())
        gm = make_fx(fwd, tracing_mode="symbolic", _allow_non_fake_inputs=True)(
            kjt.values(), kjt.lengths()
        )
        actual = gm(kjt.values(), kjt.lengths())
        torch.testing.assert_close(actual, expected)
        node_targets = [n.target for n in gm.graph.nodes if n.op == "call_function"]
        self.assertIn(
            torch.ops.recir_torchrec.kjt_lengths_per_feature.default,
            node_targets,
            "kjt_lengths_per_feature node should be in the traced graph",
        )
        single_stride_kjt = _make_kjt(
            keys=["f0", "f1"],
            per_key_lengths=[[2], [3]],
            vocab_size=16,
            seed=12,
        )
        torch.testing.assert_close(
            gm(single_stride_kjt.values(), single_stride_kjt.lengths()),
            fwd(single_stride_kjt.values(), single_stride_kjt.lengths()),
        )


class PatchEbcForTracingTest(unittest.TestCase):
    """The EBC patch preserves TorchRec's public forward/backward contract."""

    def setUp(self) -> None:
        self.kjt = _make_kjt(
            keys=["f2", "f0", "f1"],
            per_key_lengths=[[3, 1, 2], [2, 0, 1], [1, 2, 0]],
            vocab_size=32,
            seed=17,
        )

    def test_sum_mean_empty_bags_and_output_metadata_match(self) -> None:
        original = _make_ebc()
        patched = copy.deepcopy(original)

        self.assertEqual(patch_ebc_for_tracing(patched), 1)
        expected = original(self.kjt)
        actual = patched(self.kjt)

        self.assertEqual(actual.keys(), expected.keys())
        self.assertEqual(actual.offset_per_key(), expected.offset_per_key())
        torch.testing.assert_close(actual.values(), expected.values())

    def test_weight_gradients_match(self) -> None:
        original = _make_ebc()
        patched = copy.deepcopy(original)
        patch_ebc_for_tracing(patched)

        coefficient = torch.arange(1, 9, dtype=torch.float32).reshape(1, 8)
        (original(self.kjt).values() * coefficient).sum().backward()
        (patched(self.kjt).values() * coefficient).sum().backward()

        original_grads = dict(original.named_parameters())
        patched_grads = dict(patched.named_parameters())
        self.assertEqual(original_grads.keys(), patched_grads.keys())
        for name, original_param in original_grads.items():
            patched_param = patched_grads[name]
            self.assertIsNotNone(original_param.grad)
            self.assertIsNotNone(patched_param.grad)
            torch.testing.assert_close(original_param.grad, patched_param.grad)

    def test_dynamic_fake_trace_accepts_changed_stride(self) -> None:
        original = _make_ebc()
        ebc = copy.deepcopy(original)
        patch_ebc_for_tracing(ebc)
        keys = self.kjt.keys()

        def fwd(
            reference: torch.Tensor,
            values: torch.Tensor,
            lengths: torch.Tensor,
        ) -> torch.Tensor:
            torch._check(lengths.shape[0] == len(keys) * reference.shape[0])
            features = KeyedJaggedTensor(
                keys=keys,
                values=values,
                lengths=lengths,
            )
            return ebc(features).values() + reference

        fake_mode, fake_args = _dynamic_fake_args(
            "ebc_input",
            (
                torch.randn(3, 8),
                self.kjt.values(),
                self.kjt.lengths(),
            ),
        )
        with fake_mode:
            gm = make_fx(
                fwd,
                tracing_mode="fake",
                _allow_non_fake_inputs=False,
            )(*fake_args)
        output_node = next(node for node in gm.graph.nodes if node.op == "output")
        output_value = output_node.args[0]
        self.assertIsInstance(output_value, torch.fx.Node)
        self.assertEqual(output_value.meta["val"].shape[0], fake_args[0].shape[0])
        resized = _make_kjt(
            keys=keys,
            per_key_lengths=[[1, 2], [0, 1], [2, 0]],
            vocab_size=32,
            seed=23,
        )
        resized_reference = torch.randn(2, 8)
        torch.testing.assert_close(
            gm(resized_reference, resized.values(), resized.lengths()),
            original(resized.permute([1, 2, 0])).values() + resized_reference,
        )

    def test_fake_trace_executes_weight_gradient(self) -> None:
        ebc = _make_ebc()
        patch_ebc_for_tracing(ebc)
        params: Dict[str, torch.Tensor] = dict(ebc.named_parameters())
        keys = self.kjt.keys()

        def loss(
            parameters: Dict[str, torch.Tensor],
            values: torch.Tensor,
            lengths: torch.Tensor,
        ) -> torch.Tensor:
            features = KeyedJaggedTensor(keys=keys, values=values, lengths=lengths)
            output = torch.func.functional_call(ebc, parameters, (features,))
            return output.values().square().sum()

        grad = torch.func.grad(loss)
        expected = grad(params, self.kjt.values(), self.kjt.lengths())
        gm = make_fx(grad, tracing_mode="fake", _allow_non_fake_inputs=True)(
            params, self.kjt.values(), self.kjt.lengths()
        )
        actual = gm(params, self.kjt.values(), self.kjt.lengths())

        self.assertEqual(actual.keys(), expected.keys())
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name])

    def test_weighted_output_and_gradients_match(self) -> None:
        original = EmbeddingBagCollection(
            tables=[
                EmbeddingBagConfig(
                    name="table_a",
                    embedding_dim=3,
                    num_embeddings=32,
                    feature_names=["shared", "only"],
                    pooling=PoolingType.SUM,
                ),
                EmbeddingBagConfig(
                    name="table_b",
                    embedding_dim=2,
                    num_embeddings=32,
                    feature_names=["shared"],
                    pooling=PoolingType.SUM,
                ),
            ],
            is_weighted=True,
            device=torch.device("cpu"),
        )
        patched = copy.deepcopy(original)
        patch_ebc_for_tracing(patched)
        kjt = _make_kjt(
            keys=["unused", "shared", "only"],
            per_key_lengths=[[1, 0, 1], [2, 0, 1], [1, 2, 0]],
            vocab_size=32,
            seed=17,
            is_weighted=True,
        )
        original_weights = kjt.weights().detach().clone().requires_grad_()
        patched_weights = original_weights.detach().clone().requires_grad_()
        original_features = KeyedJaggedTensor(
            keys=kjt.keys(),
            values=kjt.values(),
            lengths=kjt.lengths(),
            weights=original_weights,
        )
        patched_features = KeyedJaggedTensor(
            keys=kjt.keys(),
            values=kjt.values(),
            lengths=kjt.lengths(),
            weights=patched_weights,
        )

        expected = original(original_features).values()
        actual = patched(patched_features).values()
        torch.testing.assert_close(actual, expected)
        expected.sum().backward()
        actual.sum().backward()
        original_weight_grad = original_weights.grad
        patched_weight_grad = patched_weights.grad
        assert original_weight_grad is not None
        assert patched_weight_grad is not None
        torch.testing.assert_close(patched_weight_grad, original_weight_grad)
        self.assertEqual(torch.count_nonzero(patched_weight_grad[:2]).item(), 0)
        for (expected_name, expected_param), (actual_name, actual_param) in zip(
            original.named_parameters(), patched.named_parameters()
        ):
            self.assertEqual(actual_name, expected_name)
            torch.testing.assert_close(actual_param.grad, expected_param.grad)

    def test_fake_trace_executes_weighted_parameter_gradients(self) -> None:
        ebc = _make_ebc(is_weighted=True)
        patch_ebc_for_tracing(ebc)
        params: Dict[str, torch.Tensor] = dict(ebc.named_parameters())
        kjt = _make_kjt(
            keys=self.kjt.keys(),
            per_key_lengths=[[3, 1, 2], [2, 0, 1], [1, 2, 0]],
            vocab_size=32,
            seed=17,
            is_weighted=True,
        )

        def loss(
            parameters: Dict[str, torch.Tensor],
            values: torch.Tensor,
            lengths: torch.Tensor,
            weights: torch.Tensor,
        ) -> torch.Tensor:
            features = KeyedJaggedTensor(
                keys=self.kjt.keys(),
                values=values,
                lengths=lengths,
                weights=weights,
            )
            output = torch.func.functional_call(ebc, parameters, (features,))
            return output.values().square().sum()

        grad = torch.func.grad(loss)
        expected_params = grad(
            params,
            kjt.values(),
            kjt.lengths(),
            kjt.weights(),
        )
        gm = make_fx(grad, tracing_mode="fake", _allow_non_fake_inputs=True)(
            params,
            kjt.values(),
            kjt.lengths(),
            kjt.weights(),
        )
        actual_params = gm(
            params,
            kjt.values(),
            kjt.lengths(),
            kjt.weights(),
        )

        self.assertEqual(actual_params.keys(), expected_params.keys())
        for name in expected_params:
            torch.testing.assert_close(actual_params[name], expected_params[name])

    def test_patch_functions_are_collection_specific(self) -> None:
        ec = _make_ec(table_to_features={"t": ["f"]})
        ebc = _make_ebc()

        self.assertEqual(patch_ec_for_tracing(ebc), 0)
        self.assertEqual(patch_ebc_for_tracing(ebc), 1)
        self.assertEqual(patch_ebc_for_tracing(ec), 0)
        self.assertEqual(patch_ec_for_tracing(ec), 1)

    def test_variable_stride_input_is_rejected(self) -> None:
        ebc = _make_ebc()
        patch_ebc_for_tracing(ebc)
        features = KeyedJaggedTensor(
            keys=self.kjt.keys(),
            values=self.kjt.values(),
            lengths=torch.tensor([3, 1, 2, 2, 0, 1, 1, 2, 0]),
            stride_per_key_per_rank=[[2], [3], [4]],
        )

        with self.assertRaisesRegex(NotImplementedError, "fixed-stride"):
            ebc(features)

    def test_inverse_indices_are_rejected(self) -> None:
        ebc = _make_ebc()
        patch_ebc_for_tracing(ebc)
        features = KeyedJaggedTensor(
            keys=self.kjt.keys(),
            values=self.kjt.values(),
            lengths=self.kjt.lengths(),
            inverse_indices=(
                self.kjt.keys(),
                torch.arange(self.kjt.stride()).repeat(len(self.kjt.keys()), 1),
            ),
        )

        with self.assertRaisesRegex(NotImplementedError, "inverse-index"):
            ebc(features)


if __name__ == "__main__":
    unittest.main()
