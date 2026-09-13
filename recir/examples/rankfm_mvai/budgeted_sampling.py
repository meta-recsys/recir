#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""TPU-only: hold the sequence-length sampler to the static token budget.

Prod's load-balanced sampler MEASURES a target load over the first thousand
batches and then matches it, so the token total lands wherever the data puts
it. That is the right objective on a dynamic-shape backend, where the total
only has to be fair across ranks.

It is the wrong objective under static shapes, where the total is a hard
constraint. ``_pad_kjt_to_cached`` sizes the packed sequence tensor at
``rows * per_sample_budget`` per device and anything over it is dropped by a
cumsum-clamp that zeroes whole trailing samples rather than trimming evenly.
Feeding the budget in as the target turns the sampler into the thing that
enforces the bound it was already being silently held to.

Why the bound holds: the inherited selection rule keeps unsampled a prefix of
the batch whose cumulative ``delta`` stays under ``imbalance_load``, and

    load = threshold**gamma * B + sum_unsampled delta
        <= threshold**gamma * B + sum_below delta + imbalance_load
         = target_load

so passing the budget in as ``target_load`` makes the total a guarantee rather
than an average. Two things are needed for that guarantee to bound the right
quantity:

* ``gamma`` must be 1. Prod runs 1.5, which balances ``sum(L**1.5)`` -- a proxy
  for attention COST. The tensor being constrained is ``sum(L)``.
* the constraint must be applied per DEVICE block, because that is where the
  padder truncates: its budget is ``per_host_budget // num_local``, so a rank
  total that fits can still hide an over-budget chip. The block count is read
  from ``runtime_num_local`` -- the same helper the padder uses -- rather than
  configured, so the two splits cannot drift apart.
"""

import dataclasses
import logging
from typing import List

import torch
import torch.nn as nn
from hammer.v2.modules.uih_sampling_modules import (
    LoadBalancedSequenceLengthSamplingStrategy,
    SLSampleConstructor,
)
from hammer.v3.models.udd_fm_v3.udd_fm_config import BlueReelsVDDModuleConfig
from recir.region_transforms.padding import runtime_num_local

logger: logging.Logger = logging.getLogger(__name__)


class BudgetedSLSampleConstructor(SLSampleConstructor):
    """``SLSampleConstructor`` whose target load is a budget, not a measurement.

    Only ``users_to_sample`` is overridden; the selection rule itself
    (``_users_to_sample_w_target_load``) is prod's, so which rows get spared
    keeps the same short-biased distribution.
    """

    def __init__(
        self,
        strategy: LoadBalancedSequenceLengthSamplingStrategy,
        per_sample_budget: int,
    ) -> None:
        # gamma=1 so the inherited bound constrains sum(L) -- the packed tensor
        # -- rather than prod's compute-shaped sum(L**1.5). Copied rather than
        # mutated: the original instance is shared with whatever else holds the
        # module config.
        super().__init__(dataclasses.replace(strategy, gamma=1.0))
        self._per_sample_budget: int = int(per_sample_budget)
        self._infeasible_warned: bool = False

    def users_to_sample(
        self,
        ts_lengths: torch.Tensor,
        thresholds: float,
        no_sample_prob: torch.Tensor,
        sample_all_users: bool = False,
    ) -> torch.Tensor:
        if sample_all_users:
            # Every eligible row is truncated anyway, so there is no selection
            # left to make and no budget to spend.
            return super().users_to_sample(
                ts_lengths, thresholds, no_sample_prob, sample_all_users
            )

        if self._per_sample_budget < thresholds and not self._infeasible_warned:
            self._infeasible_warned = True
            logger.warning(
                "Budgeted SL: per_sample_budget=%d is below the sampled length "
                "%.0f. Sampling every eligible row still overshoots, so "
                "imbalance_load goes non-positive and the inherited rule falls "
                "back to sampling everyone -- the total is NOT bounded. Raise "
                "the budget above %d, or lower alpha/beta.",
                self._per_sample_budget,
                thresholds,
                int(thresholds),
            )

        rows = int(ts_lengths.shape[0])
        # Read at call time from the same helper the padder uses, rather than
        # carried as a setting: the two splits have to be identical, and a
        # config knob is a way for them to disagree.
        num_blocks = runtime_num_local()
        if rows % num_blocks:
            raise ValueError(
                f"Budgeted SL: {rows} rows do not divide into "
                f"{num_blocks} device blocks. The blocks bounded here "
                f"would not line up with the ones the padder truncates."
            )
        block = rows // num_blocks

        masks: List[torch.Tensor] = []
        for i in range(num_blocks):
            rows_slice = slice(i * block, (i + 1) * block)
            target_load = torch.tensor(
                float(block * self._per_sample_budget),
                dtype=torch.float32,
                device=ts_lengths.device,
            )
            masks.append(
                self._users_to_sample_w_target_load(
                    ts_lengths=ts_lengths[rows_slice],
                    thresholds=thresholds,
                    no_sample_prob=no_sample_prob[rows_slice],
                    target_load=target_load,
                )
            )
        return torch.cat(masks)


def install_budgeted_sampling(
    model: nn.Module, module_config: BlueReelsVDDModuleConfig
) -> None:
    budgets = module_config.static_ro_per_sample_budgets
    seq_keys = [ro for ro, _nro in module_config.merge_item_uih_feature_mapping]
    if not budgets or not seq_keys:
        raise ValueError(
            "budgeted_seq_len_sampling needs static_ro_per_sample_budgets, "
            "which only static-shape mode (enable_padding) fills in."
        )
    per_sample_budget = int(budgets[seq_keys[0]])

    # pyre-ignore[16]: dynamic submodule traversal through hammer v3 union
    preprocessor_stack = model.recblock.preprocessor_stack
    # pyre-ignore[16]: dynamic attr, set by PreprocessorStack.__setattr__
    preprocessors = list(preprocessor_stack.mot_preprocessor._hstu_preprocessor_list)
    installed = 0
    for hstu_preprocessor in preprocessors:
        feature_preprocessor = hstu_preprocessor._feature_preprocessor
        sampler = feature_preprocessor._uih_sampling_module
        if sampler is None:
            continue
        constructor = getattr(sampler, "_sample_constructor", None)
        if constructor is None:
            continue
        strategy = constructor._sampling_strategy
        if not isinstance(strategy, LoadBalancedSequenceLengthSamplingStrategy):
            raise TypeError(
                f"Budgeted SL needs the load-balanced strategy to inherit the "
                f"target-load rule from; this sampler carries "
                f"{type(strategy).__name__}."
            )
        sampler._sample_constructor = BudgetedSLSampleConstructor(
            strategy, per_sample_budget
        )
        installed += 1

    if not installed:
        raise ValueError(
            "budgeted_seq_len_sampling was asked for but no sampler was found "
            "to swap; the run would silently train with the budget "
            "unenforced. Is --disable_seq_len_sampling also set?"
        )
    logger.info(
        "[budgeted-sl] swapped %d sampler(s) to a static target: "
        "per_sample_budget=%d over %d device block(s), gamma=1.",
        installed,
        per_sample_budget,
        runtime_num_local(),
    )
