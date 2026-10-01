# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""The coscaling-smoke ProdConfigSpec check refuses what the smoke would reject."""

import unittest
from types import SimpleNamespace
from typing import Optional
from unittest import mock

import torch
from recir.examples.coscaling_mvai.smoke import artifact


def _artifact(
    static_shape_mode: bool = True, max_ro_len: Optional[int] = 128
) -> SimpleNamespace:
    return SimpleNamespace(
        module_config=SimpleNamespace(
            static_shape_mode=static_shape_mode,
            coscaling_static_max_ro_len=max_ro_len,
            coscaling_static_max_num_targets=16,
            embedding_tables={"t": 0},
        ),
        batch=SimpleNamespace(nro_float_features=torch.zeros(7, 2)),
    )


class ArtifactCheckTest(unittest.TestCase):
    def _check(self, loaded: SimpleNamespace) -> str:
        with mock.patch.object(artifact, "load_artifact", return_value=loaded):
            return artifact.check("x.pt")

    def test_a_smoke_ready_artifact_passes(self) -> None:
        summary = self._check(_artifact())
        self.assertIn("B_NRO=7", summary)
        self.assertIn("static_max_ro_len=128", summary)

    def test_static_shape_mode_off_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            self._check(_artifact(static_shape_mode=False))

    def test_missing_encoder_bounds_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            self._check(_artifact(max_ro_len=None))
