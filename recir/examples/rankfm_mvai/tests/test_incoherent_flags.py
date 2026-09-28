# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Warnings for flag combinations that are silently not what they look like.

Each case here is one that used to pass without a word. ``--hash_size_cap``
under ``--prod_faithful`` clamps prod's ~3e9-row table and then rebuilds
feature_hash_sizes from the capped sizes, so the pipeline stays
self-consistent; ``--l1_num_layers 3 --l2_num_layers 11`` looks like it matches
prod (prod IS L1=3, L2=11) and reaches nothing.
"""

from __future__ import annotations

import logging
import unittest
from unittest import mock

from recir.examples.rankfm_mvai.trainer_args import (
    parse_cli,
    RankFMModelArgs,
    RankFMTrainerConfig,
)


class IncoherentFlagWarningTest(unittest.TestCase):
    def _warnings(self, argv: list[str]) -> list[str]:
        with mock.patch("sys.argv", ["train", *argv]):
            with self.assertLogs(
                "recir.examples.rankfm_mvai.trainer_args", level=logging.WARNING
            ) as captured:
                # assertLogs fails the test when nothing is logged, so emit a
                # marker and drop it -- "no warnings" is an outcome under test.
                logging.getLogger("recir.examples.rankfm_mvai.trainer_args").warning(
                    "__marker__"
                )
                parse_cli()
        return [m for m in captured.output if "__marker__" not in m]

    def test_clean_prod_run_warns_about_nothing(self) -> None:
        self.assertEqual(
            [],
            self._warnings(
                [
                    "--prod_faithful",
                    "--seq_len_cap",
                    "1024",
                    "--local_batch_size",
                    "2048",
                ]
            ),
        )

    def test_hash_size_cap_under_prod_faithful_warns(self) -> None:
        warnings = self._warnings(["--prod_faithful", "--hash_size_cap", "100000"])
        self.assertEqual(1, len(warnings))
        self.assertIn("no longer prod-faithful", warnings[0])

    def test_hash_size_cap_is_a_warning_not_an_error(self) -> None:
        # The 4-chip verify run and both parity legs cap the tables on
        # purpose; raising would break all three.
        with mock.patch(
            "sys.argv", ["train", "--prod_faithful", "--hash_size_cap", "100000"]
        ):
            _trainer_cfg, model_args = parse_cli()
        self.assertEqual(100000, model_args.hash_size_cap)

    def test_data_file_template_warns(self) -> None:
        warnings = self._warnings(["--data_file_template", "rank_0/{i}.pt.zst"])
        self.assertEqual(1, len(warnings))
        self.assertIn("throughput benchmark", warnings[0])

    def test_prod_layer_counts_warn_even_at_their_defaults(self) -> None:
        # The documented mistake. 3 and 11 ARE the dataclass defaults, so
        # comparing values instead of inspecting argv would see nothing.
        defaults = RankFMModelArgs()
        self.assertEqual(3, defaults.l1_num_layers)
        self.assertEqual(11, defaults.l2_num_layers)

        warnings = self._warnings(
            ["--prod_faithful", "--l1_num_layers", "3", "--l2_num_layers", "11"]
        )
        self.assertEqual(1, len(warnings))
        self.assertIn("--l1_num_layers=3", warnings[0])
        self.assertIn("--l2_num_layers=11", warnings[0])
        self.assertIn("IGNORED", warnings[0])

    def test_bench_run_without_prod_faithful_warns_about_nothing(self) -> None:
        self.assertEqual(
            [],
            self._warnings(
                [
                    "--num_candidates",
                    "32",
                    "--hash_size_cap",
                    "1000",
                    "--l1_num_layers",
                    "1",
                ]
            ),
        )

    def test_seq_len_cap_is_not_reported_as_inert(self) -> None:
        # It is the one shape flag that DOES apply under --prod_faithful.
        self.assertEqual(
            [], self._warnings(["--prod_faithful", "--seq_len_cap", "1024"])
        )

    def test_trainer_defaults_are_unchanged(self) -> None:
        # The probe parser runs a second pass over argv; it must not disturb
        # the values the real parser produced.
        with mock.patch("sys.argv", ["train"]):
            trainer_cfg, model_args = parse_cli()
        self.assertEqual(RankFMTrainerConfig(), trainer_cfg)
        self.assertEqual(RankFMModelArgs(), model_args)
