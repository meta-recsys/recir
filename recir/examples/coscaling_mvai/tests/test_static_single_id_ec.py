# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import unittest

import torch
from recir.embedding.split_collection import replicate_embedding_collection
from recir.examples.coscaling_mvai.model import StaticSingleIdEmbeddingCollection
from torch.fx.experimental.proxy_tensor import make_fx
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

_STRIDE = 5


def _collection() -> EmbeddingCollection:
    torch.manual_seed(0)
    return EmbeddingCollection(
        tables=[
            EmbeddingConfig(
                name=f"t_{n}",
                embedding_dim=4,
                num_embeddings=8,
                feature_names=[f"f_{n}"],
            )
            for n in ("a", "b")
        ],
        device=torch.device("cpu"),
    )


def _kjt(lengths: torch.Tensor) -> KeyedJaggedTensor:
    # An unrelated key first, so the lookup has to find its keys by name.
    keys = ["other", "f_a", "f_b"]
    return KeyedJaggedTensor(
        keys=keys,
        values=torch.randint(0, 8, (int(lengths.sum()),)),
        lengths=lengths,
    )


def _single_id_lengths() -> torch.Tensor:
    other = torch.tensor([2, 0, 1, 3, 1], dtype=torch.int32)
    return torch.cat([other, torch.ones(2 * _STRIDE, dtype=torch.int32)])


class StaticSingleIdEmbeddingCollectionTest(unittest.TestCase):
    def _pair(self) -> tuple[EmbeddingCollection, StaticSingleIdEmbeddingCollection]:
        parent = torch.nn.Module()
        ec = _collection()
        parent.ec = ec
        reference = _collection()
        reference.load_state_dict(ec.state_dict())
        static = replicate_embedding_collection(
            parent, "ec", StaticSingleIdEmbeddingCollection
        )
        return reference, static

    def test_matches_the_torchrec_lookup(self) -> None:
        reference, static = self._pair()
        features = _kjt(_single_id_lengths())
        expected = reference(features)
        actual = static(features)
        self.assertEqual(set(actual), {"f_a", "f_b"})
        for name in ("f_a", "f_b"):
            torch.testing.assert_close(actual[name].values(), expected[name].values())
            torch.testing.assert_close(actual[name].lengths(), expected[name].lengths())

    def test_traces_with_static_shapes(self) -> None:
        _, static = self._pair()
        features = _kjt(_single_id_lengths())
        params = dict(static.named_parameters())
        keys = features.keys()

        def fn(
            params: dict[str, torch.Tensor],
            values: torch.Tensor,
            lengths: torch.Tensor,
        ) -> list[torch.Tensor]:
            out = torch.func.functional_call(
                static,
                params,
                (KeyedJaggedTensor(keys=keys, values=values, lengths=lengths),),
            )
            return [out["f_a"].values(), out["f_b"].values()]

        gm = make_fx(fn, tracing_mode="fake")(
            params, features.values(), features.lengths()
        )
        output = next(n for n in gm.graph.nodes if n.op == "output")
        for node in output.args[0]:
            self.assertEqual(tuple(node.meta["val"].shape), (_STRIDE, 4))

    def test_rejects_a_row_with_more_than_one_id(self) -> None:
        _, static = self._pair()
        lengths = _single_id_lengths()
        lengths[-1] = 2
        with self.assertRaises(ValueError):
            static(_kjt(lengths))
