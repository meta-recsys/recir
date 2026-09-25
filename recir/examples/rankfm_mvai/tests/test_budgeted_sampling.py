#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.


import unittest
from typing import List
from unittest.mock import patch

import torch
from hammer.v2.modules.uih_sampling_modules import (
    LoadBalancedSequenceLengthSamplingStrategy,
)
from recir.examples.rankfm_mvai.budgeted_sampling import BudgetedSLSampleConstructor

# Prod's L1 sampler
# (model_v5_hstu_standalone_cint_datafm_fm_xsurface_config.py:142-150).
_ALPHA = 1.7
_BETA = 1.0


def _strategy(
    gamma: float = 1.5,
) -> LoadBalancedSequenceLengthSamplingStrategy:
    return LoadBalancedSequenceLengthSamplingStrategy(
        min_hour_dropout=None,
        weighted_sample=True,
        alpha=_ALPHA,
        beta=_BETA,
        gamma=gamma,
    )


def _sampler_params(max_seq_len: int) -> float:
    """The threshold ``UIHSampler.dropout_mask`` derives (:650)."""
    return _BETA * max_seq_len ** (_ALPHA / 2)


def _no_sample_prob(lengths: torch.Tensor, max_seq_len: int) -> torch.Tensor:
    """``UIHSampler.dropout_mask``:651-653, kept in the same form."""
    return (max_seq_len**_ALPHA) / torch.pow(lengths.to(torch.float32), 2)


def _realized_lengths(
    lengths: torch.Tensor, sampled: torch.Tensor, thresholds: float
) -> torch.Tensor:
    """What the preprocessor emits: sampled rows truncate, the rest keep theirs."""
    return torch.where(sampled, torch.full_like(lengths, thresholds), lengths)


def _blocks(total_rows: int, num_blocks: int) -> List[slice]:
    block = total_rows // num_blocks
    return [slice(i * block, (i + 1) * block) for i in range(num_blocks)]


class BudgetedSamplingTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def _run(
        self,
        lengths: torch.Tensor,
        max_seq_len: int,
        budget: int,
        num_blocks: int,
    ) -> torch.Tensor:
        thresholds = _sampler_params(max_seq_len)
        constructor = BudgetedSLSampleConstructor(_strategy(), budget)
        with patch(
            "recir.examples.rankfm_mvai.budgeted_sampling.runtime_num_local",
            return_value=num_blocks,
        ):
            return constructor.users_to_sample(
                ts_lengths=lengths,
                thresholds=thresholds,
                no_sample_prob=_no_sample_prob(lengths, max_seq_len),
            )

    def test_every_device_block_stays_within_its_budget(self) -> None:
        """The guarantee, checked per block rather than over the rank.

        Two length distributions. Random lengths leave slack everywhere, so
        they check the ordinary case; all-max is the worst one the budget is
        derived against -- no short row donates slack, ``imbalance_load`` is at
        its smallest and the rule has to truncate nearly everyone.
        """
        max_seq_len = 1400
        budget = 578  # config._sampler_derived_ro_budget at N=1400, alpha=1.7
        num_blocks = 4
        rows = 2048
        block = rows // num_blocks
        thresholds = _sampler_params(max_seq_len)

        for trial in range(20):
            torch.manual_seed(trial)
            lengths = torch.randint(1, max_seq_len + 1, (rows,), dtype=torch.float32)
            if trial == 0:
                lengths = torch.full((rows,), float(max_seq_len))
            sampled = self._run(lengths, max_seq_len, budget, num_blocks)
            realized = _realized_lengths(lengths, sampled, thresholds)
            for i, rows_slice in enumerate(_blocks(rows, num_blocks)):
                # Guard against a vacuous pass: if the block already fit
                # untouched, the selection rule never had to do anything and
                # the assertion below holds for a sampler that does nothing.
                self.assertGreater(
                    float(lengths[rows_slice].sum()),
                    block * budget,
                    f"trial {trial} block {i} fits before sampling; this case "
                    f"does not exercise the bound",
                )
                total = float(realized[rows_slice].sum())
                self.assertLessEqual(
                    total,
                    block * budget,
                    f"trial {trial} block {i}: {total:.0f} tokens over a "
                    f"{block * budget} budget -- the padder would zero "
                    f"trailing samples of this chip",
                )

    def test_a_rank_total_that_fits_can_still_hide_an_over_budget_chip(self) -> None:
        """Why the split is per block: one block of long rows, three of short.

        Bounding the RANK would pass this batch unchanged -- the short blocks
        more than pay for the long one. The padder does not average across
        chips, so block 0 has to be brought under on its own.
        """
        max_seq_len = 1400
        budget = 578
        rows, num_blocks = 1024, 4
        block = rows // num_blocks
        thresholds = _sampler_params(max_seq_len)

        lengths = torch.full((rows,), 8.0)
        lengths[:block] = float(max_seq_len)

        self.assertGreater(float(lengths[:block].sum()), block * budget)
        self.assertLess(float(lengths.sum()), rows * budget)

        sampled = self._run(lengths, max_seq_len, budget, num_blocks)
        realized = _realized_lengths(lengths, sampled, thresholds)
        self.assertLessEqual(float(realized[:block].sum()), block * budget)

    def test_rows_that_do_not_divide_into_blocks_raise(self) -> None:
        """Silently bounding the wrong blocks is the failure to avoid.

        An uneven split still produces a mask and a running job, whose blocks
        do not line up with the ones the padder truncates.
        """
        with self.assertRaisesRegex(ValueError, "device blocks"):
            self._run(torch.full((10,), 100.0), 1400, 578, 4)

    def test_gamma_is_forced_to_one_without_touching_the_caller_s_strategy(
        self,
    ) -> None:
        """Two separate requirements that are easy to satisfy only halfway.

        The inherited bound constrains ``sum(L**gamma)``; the packed tensor is
        ``sum(L)``, so gamma must be 1 or the wrong quantity is held down. And
        the strategy instance is shared with whatever else holds the module
        config, so prod's 1.5 must survive on it.
        """
        strategy = _strategy(gamma=1.5)
        constructor = BudgetedSLSampleConstructor(strategy, 578)

        # The constructor declares the base SequenceLengthSamplingStrategy,
        # which has no gamma; narrow to the subclass it is actually handed.
        forced = constructor._sampling_strategy
        assert isinstance(forced, LoadBalancedSequenceLengthSamplingStrategy)

        self.assertEqual(forced.gamma, 1.0)
        self.assertEqual(strategy.gamma, 1.5)
