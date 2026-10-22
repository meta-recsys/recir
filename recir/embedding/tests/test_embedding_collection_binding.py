#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Backends bind to the model's EmbeddingCollection by its FQN."""

from unittest import TestCase

import torch.nn as nn
from recir.embedding.backend import swap_submodule
from recir.embedding.sparsecore import SparseCoreEmbeddingBackend


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


class SparseCoreDenseParametersTest(TestCase):
    def test_only_the_bound_collection_is_excluded(self) -> None:
        backend = SparseCoreEmbeddingBackend("owner.collection")

        dense = backend.dense_named_parameters(_model())

        self.assertEqual(
            set(dense),
            {
                "tower.weight",
                "tower.bias",
                "owner.collection_b.weight",
                "owner.collection_b.bias",
            },
        )
