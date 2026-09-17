# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from __future__ import annotations

import os
import unittest
from typing import cast
from unittest.mock import patch

import hammer.v2.ops.hstu_preprocess_and_attention as hstu_ops
from hammer.ops.pytorch.ragged_hstu_attention import pytorch_hstu_mha
from recir.examples.rankfm_mvai.rankfm_model_utils import (
    AttentionBackend,
    install_attention_backend,
    resolve_attention_backend,
)
from recir.ops.helion_hstu_attention_op import _recir_pytorch_hstu_mha as helion_mha
from recir.ops.hstu_attention_op import _recir_pytorch_hstu_mha as pallas_padded_mha
from recir.ops.pallas_jagged_hstu_attention_op import (
    _recir_pallas_jagged_hstu_mha as pallas_jagged_mha,
)


class AttentionBackendTest(unittest.TestCase):
    def tearDown(self) -> None:
        install_attention_backend("baseline")

    def test_unset_or_empty_backend_selects_baseline(self) -> None:
        for environment in ({}, {"RANKFM_MVAI_ATTENTION_BACKEND": ""}):
            with (
                self.subTest(environment=environment),
                patch.dict(os.environ, environment, clear=True),
            ):
                self.assertEqual("baseline", resolve_attention_backend())

    def test_supported_backends_resolve_unchanged(self) -> None:
        for backend in ("baseline", "pallas_padded", "pallas_jagged", "helion"):
            with (
                self.subTest(backend=backend),
                patch.dict(
                    os.environ,
                    {"RANKFM_MVAI_ATTENTION_BACKEND": backend},
                    clear=True,
                ),
            ):
                self.assertEqual(backend, resolve_attention_backend())

    def test_invalid_backend_fails_closed(self) -> None:
        with patch.dict(
            os.environ,
            {"RANKFM_MVAI_ATTENTION_BACKEND": "pallas_dense"},
            clear=True,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "baseline, helion, pallas_jagged, pallas_padded",
            ):
                resolve_attention_backend()

    def test_legacy_flags_no_longer_select_a_backend(self) -> None:
        with patch.dict(
            os.environ,
            {
                "RANKFM_MVAI_USE_HELION": "1",
                "RECIR_HSTU_PALLAS": "1",
                "RECIR_PALLAS_JAGGED": "1",
            },
            clear=True,
        ):
            self.assertEqual("baseline", resolve_attention_backend())

    def test_installs_each_backend_and_restores_baseline(self) -> None:
        expected = {
            "baseline": pytorch_hstu_mha,
            "pallas_padded": pallas_padded_mha,
            "pallas_jagged": pallas_jagged_mha,
            "helion": helion_mha,
        }

        for backend, expected_mha in expected.items():
            with self.subTest(backend=backend):
                install_attention_backend(cast(AttentionBackend, backend))
                self.assertIs(expected_mha, hstu_ops.pytorch_hstu_mha)

        install_attention_backend("baseline")
        self.assertIs(pytorch_hstu_mha, hstu_ops.pytorch_hstu_mha)

    def test_invalid_direct_install_does_not_change_hammer(self) -> None:
        install_attention_backend("pallas_jagged")
        installed = hstu_ops.pytorch_hstu_mha

        with self.assertRaisesRegex(ValueError, "attention backend must be one of"):
            install_attention_backend(cast(AttentionBackend, "pallas_dense"))

        self.assertIs(installed, hstu_ops.pytorch_hstu_mha)
