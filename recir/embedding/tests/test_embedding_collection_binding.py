#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Backends bind to the model's EmbeddingCollection by its FQN."""

from unittest import mock, TestCase

import torch
import torch.nn as nn
from recir.embedding.backend import check_binding, swap_submodule
from recir.embedding.cpu import CPUEmbeddingBackend
from recir.embedding.split_collection import ReplicatedEmbeddingCollection
from torchrec.modules.embedding_configs import EmbeddingBagConfig, EmbeddingConfig
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
)

_TABLES = [
    EmbeddingConfig(name="t", embedding_dim=4, num_embeddings=8, feature_names=["f"])
]


class _Owner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.collection: nn.Module = nn.Linear(2, 2)
        # Shares the bound collection's name as a prefix, but is a different module.
        self.collection_b: nn.Module = nn.Linear(2, 2)


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tower: nn.Module = nn.Linear(2, 2)
        self.owner: _Owner = _Owner()


def _model() -> _Model:
    return _Model()


class _CollectionOwner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.collection: nn.Module = EmbeddingCollection(
            tables=_TABLES, device=torch.device("meta")
        )


class _CollectionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.owner: _CollectionOwner = _CollectionOwner()
        self.replicated: nn.Module = ReplicatedEmbeddingCollection(
            tables=_TABLES, device=torch.device("meta")
        )


class SwapSubmoduleTest(TestCase):
    def test_binds_at_the_path_and_returns_the_original(self) -> None:
        model = _model()
        original = model.owner.collection
        replacement = nn.Identity()

        self.assertIs(swap_submodule(model, "owner.collection", replacement), original)
        self.assertIs(model.owner.collection, replacement)

    def test_a_top_level_path(self) -> None:
        model = _model()
        replacement = nn.Identity()

        swap_submodule(model, "tower", replacement)

        self.assertIs(model.tower, replacement)

    def test_the_root_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            swap_submodule(_model(), "", nn.Identity())

    def test_an_unknown_path_raises(self) -> None:
        with self.assertRaises(AttributeError):
            swap_submodule(_model(), "owner.missing", nn.Identity())


class CheckBindingTest(TestCase):
    def test_a_replicated_collection_may_sit_beside_the_bound_one(self) -> None:
        check_binding(_CollectionModel(), "owner.collection")

    def test_a_second_sharded_collection_is_named(self) -> None:
        model = _CollectionModel()
        model.owner.add_module(
            "other", EmbeddingCollection(tables=_TABLES, device=torch.device("meta"))
        )

        with self.assertRaisesRegex(ValueError, "owner.other"):
            check_binding(model, "owner.collection")

    def test_a_bag_collection_is_named(self) -> None:
        model = _CollectionModel()
        model.add_module(
            "ebc",
            EmbeddingBagCollection(
                tables=[
                    EmbeddingBagConfig(
                        name="b", embedding_dim=4, num_embeddings=8, feature_names=["g"]
                    )
                ],
                device=torch.device("meta"),
            ),
        )

        with self.assertRaisesRegex(ValueError, "ebc"):
            check_binding(model, "owner.collection")

    def test_the_fqn_must_name_an_embedding_collection(self) -> None:
        with self.assertRaises(TypeError):
            check_binding(_CollectionModel(), "owner")

    def test_setup_checks_the_binding_first(self) -> None:
        for backend in (CPUEmbeddingBackend("owner"),):
            with self.assertRaisesRegex(TypeError, "owner is a"):
                backend.setup(
                    _CollectionModel(),
                    table_configs=_TABLES,
                    sparse_config=mock.Mock(),
                    device=torch.device("cpu"),
                    world_size=1,
                )
