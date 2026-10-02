# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""rankfm's ProdConfigSpec check: the baked artifact must be prod's model."""

import unittest
from types import SimpleNamespace
from unittest import mock

from recir.examples.rankfm_mvai import prod_config_artifact
from recir.examples.rankfm_mvai.prod_config_spec import PROD_CONFIG_EXPECTED_SEQ_LEN


def _loaded(seq_len: int) -> tuple[SimpleNamespace, None]:
    return SimpleNamespace(hstu_max_seq_len=seq_len, embedding_tables={"t": 0}), None


class ProdConfigCheckTest(unittest.TestCase):
    def test_prod_sequence_length_passes(self) -> None:
        with mock.patch.object(
            prod_config_artifact,
            "load_prod_config",
            return_value=_loaded(PROD_CONFIG_EXPECTED_SEQ_LEN),
        ):
            summary = prod_config_artifact.check("x.pt")
        self.assertIn(f"hstu_max_seq_len={PROD_CONFIG_EXPECTED_SEQ_LEN}", summary)

    def test_other_sequence_length_is_refused(self) -> None:
        with mock.patch.object(
            prod_config_artifact, "load_prod_config", return_value=_loaded(1024)
        ):
            with self.assertRaises(ValueError) as caught:
                prod_config_artifact.check("x.pt")
        self.assertIn("hstu_max_seq_len=1024", str(caught.exception))
