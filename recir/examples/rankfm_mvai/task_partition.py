#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""The udd_fm_v3 task layout, as the metric layer needs to see it.

Deliberately a leaf module: it imports nothing at runtime, so both entry points
and the unit test can pull it in without dragging along ``train_common``'s
gr-op bootstrap (which registers torch library ops on import and cannot run
inside a plain test process).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskPartition:
    """``all_task_names`` is the authority for row ``i`` of every per-task tensor
    the model emits (prediction / label / weight / loss / llw_weights /
    loss_scale). The cls and reg lists are disjoint slices of it."""

    all_task_names: List[str]
    cls_task_names: List[str]
    cls_task_indices: List[int]
    reg_task_names: List[str]
    reg_task_indices: List[int]


def build_cls_reg_partition(module_config: Any) -> TaskPartition:
    """Partition the udd_fm_v3 tasks into cls/reg buckets for metric reporting.

    Two blocks, concatenated in the order the model emits them
    (``udd_fm_train_v3.py:1642-1645`` builds every per-task tensor as
    ``cat([main_block, mt_block])``):

    1. The ``num_main_tasks`` main rows, split arithmetically. The naive split
       using only num_main_reg_tasks is wrong when has_unpersonalized=True
       because the unpersonalized tasks live AFTER the regression block and are
       trained as classification (sigmoid+BCE); without this account, metric
       reporting would compute classification NE on regression heads and vice
       versa.
    2. The MoT aux rows, one per ``mt_tasks_configs`` entry, split by that
       entry's ``is_regression`` flag.

    Block 2 is why this cannot just read ``module_config.task_names``: that list
    covers only the main block (``train_module_wrapper.py:384`` sets it to
    ``main_task_names``), so anything derived from ``num_main_tasks`` alone
    leaves the MoT rows unreachable -- which is why the TPU leg reported metrics
    for 90 of 105 tasks while MAST reported all 105. The MoT names live in
    ``mt_tasks_configs``, and its iteration order IS the row order:
    ``hammer/v3/modules/mot/loss/loss_module.py:84`` walks the same dict to
    build the heads.

    The resulting split matches what the MAST leg derives by name from
    ``feature_config.regression_task_names``.

    ``module_config`` is typed ``Any`` rather than ``BlueReelsVDDModuleConfig``
    to keep this module import-free; only five attributes are read.
    """
    main_task_names: List[str] = list(module_config.task_names)
    n_total: int = module_config.num_main_tasks
    n_reg: int = module_config.num_main_reg_tasks
    if module_config.has_unpersonalized:
        n_unpers: int = module_config.num_unpersonalized_tasks
        cls_end = n_total - n_reg - n_unpers
        reg_end = n_total - n_unpers
        cls_indices = list(range(cls_end)) + list(range(reg_end, n_total))
        reg_indices = list(range(cls_end, reg_end))
    else:
        cls_end = n_total - n_reg
        cls_indices = list(range(cls_end))
        reg_indices = list(range(cls_end, n_total))

    # MoT aux tasks. Keys carry no "mt/" prefix in mt_tasks_configs but the
    # model's task_names (and every MAST tag) do, so prepend it here.
    mt_task_names: List[str] = []
    for offset, (name, (is_regression, _weight)) in enumerate(
        (module_config.mt_tasks_configs or {}).items()
    ):
        mt_task_names.append(f"mt/{name}")
        (reg_indices if is_regression else cls_indices).append(n_total + offset)

    all_task_names = main_task_names + mt_task_names
    cls_indices.sort()
    reg_indices.sort()
    logger.info(
        "task partition: %d total (%d main + %d mt) -> %d cls, %d reg",
        len(all_task_names),
        len(main_task_names),
        len(mt_task_names),
        len(cls_indices),
        len(reg_indices),
    )
    return TaskPartition(
        all_task_names=all_task_names,
        cls_task_names=[all_task_names[i] for i in cls_indices],
        cls_task_indices=cls_indices,
        reg_task_names=[all_task_names[i] for i in reg_indices],
        reg_task_indices=reg_indices,
    )
