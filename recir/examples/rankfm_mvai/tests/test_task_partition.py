#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""The cls/reg task partition must cover every row the model emits."""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from later.unittest import TestCase
from recir.examples.rankfm_mvai.task_partition import build_cls_reg_partition


@dataclass
class _StubModuleConfig:
    """The five fields ``build_cls_reg_partition`` reads."""

    task_names: List[str] = field(
        default_factory=lambda: [f"main_{i}" for i in range(90)]
    )
    num_main_tasks: int = 90
    num_main_reg_tasks: int = 14
    has_unpersonalized: bool = True
    num_unpersonalized_tasks: int = 3
    mt_tasks_configs: Dict[str, Tuple[bool, int]] = field(
        default_factory=lambda: {
            "vvp100": (False, 1),
            "like": (False, 2),
            "share": (False, 4),
            "comment": (False, 8),
            "vvp200": (False, 16),
            "vvs10s": (False, 32),
            "follow": (False, 64),
            "save": (False, 128),
            "share_whatsapp": (False, 256),
            "share_messenger": (False, 512),
            "ifu_click": (False, 1024),
            "good_swipe": (False, 2097152),
            "vvp75": (False, 4194304),
            "ifu_nonskip": (False, 8388608),
            "log_vvd_long": (True, 16777216),
        }
    )


class TestTaskPartition(TestCase):
    def test_prod_shape_is_90_cls_15_reg(self) -> None:
        p = build_cls_reg_partition(_StubModuleConfig())
        self.assertEqual(len(p.all_task_names), 105)
        self.assertEqual(len(p.cls_task_names), 90)
        self.assertEqual(len(p.reg_task_names), 15)

    def test_indices_tile_every_row_exactly_once(self) -> None:
        # The guard against both silent metric gaps and double-counting: a row
        # missing from both lists gets no metrics, a row in both gets NE and MSE.
        p = build_cls_reg_partition(_StubModuleConfig())
        combined = sorted(p.cls_task_indices + p.reg_task_indices)
        self.assertEqual(combined, list(range(len(p.all_task_names))))

    def test_mt_names_are_prefixed_and_in_config_order(self) -> None:
        # mt_tasks_configs keys carry no "mt/" prefix, but every MAST tag does,
        # and dict order IS the model's row order (mot/loss/loss_module.py:84).
        cfg = _StubModuleConfig()
        p = build_cls_reg_partition(cfg)
        self.assertEqual(
            p.all_task_names[90:],
            [f"mt/{name}" for name in cfg.mt_tasks_configs],
        )

    def test_mt_regression_task_lands_in_reg_bucket(self) -> None:
        p = build_cls_reg_partition(_StubModuleConfig())
        self.assertIn("mt/log_vvd_long", p.reg_task_names)
        self.assertNotIn("mt/log_vvd_long", p.cls_task_names)
        self.assertIn("mt/vvp100", p.cls_task_names)

    def test_unpersonalized_tasks_are_classification(self) -> None:
        # They sit AFTER the regression block, so a naive
        # split-at-num_main_reg_tasks would score them as regression.
        p = build_cls_reg_partition(_StubModuleConfig())
        for i in (87, 88, 89):
            self.assertIn(i, p.cls_task_indices)

    def test_no_mt_tasks_leaves_the_main_split_untouched(self) -> None:
        cfg = _StubModuleConfig()
        cfg.mt_tasks_configs = {}
        p = build_cls_reg_partition(cfg)
        self.assertEqual(len(p.all_task_names), 90)
        self.assertEqual(len(p.cls_task_names), 76)
        self.assertEqual(len(p.reg_task_names), 14)

    def test_without_unpersonalized_reg_block_is_the_tail(self) -> None:
        cfg = _StubModuleConfig()
        cfg.has_unpersonalized = False
        cfg.mt_tasks_configs = {}
        p = build_cls_reg_partition(cfg)
        self.assertEqual(p.cls_task_indices, list(range(76)))
        self.assertEqual(p.reg_task_indices, list(range(76, 90)))
