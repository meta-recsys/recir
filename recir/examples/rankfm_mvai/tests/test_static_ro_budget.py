#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""The static RO budget must not be tied to the model's sequence bound.

Both were ``hstu_max_seq_len``, so the only way to shrink the padded sequence
tensor was ``--seq_len_cap``, which also truncates real history, re-scales the
sampler (the sampled length is ``N ** (alpha/2)``) and feeds the attention
scale. They measure different things: the budget sizes a PACKED tensor and is
a per-chip total, while ``hstu_max_seq_len`` is a per-row model bound.
"""

from typing import Dict, List, Tuple

from hammer.v2.modules.uih_sampling_modules import SequenceLengthSamplingStrategy
from hammer.v3.models.udd_fm_v3.udd_fm_config import BlueReelsVDDModuleConfig
from later.unittest import TestCase
from pyre_extensions import none_throws

# `prod_faithful=True` makes config.build_module_config reach
# prod_config_artifact.build_prod_config_live, which imports the shims inside
# the function body (they are `# @manual` there). Nothing in this file names
# them, so autodeps needs telling or it strips them and the test dies with
# ModuleNotFoundError: hammer.v3.models.udd_fm_v3.launch.
# @dep=//hammer/v3/models/udd_fm_v3/launch:light_replay_config_shim
# @dep=//hammer/v3/models/udd_fm_v3/launch:prod_config_shim
from recir.examples.rankfm_mvai.config import build_module_config, RankFMModelArgs


def _budgets(
    budgeted_seq_len_sampling: bool = False,
) -> Tuple[BlueReelsVDDModuleConfig, Dict[str, int]]:
    module_config, _ = build_module_config(
        RankFMModelArgs(prod_faithful=True),
        enable_padding=True,
        budgeted_seq_len_sampling=budgeted_seq_len_sampling,
    )
    return module_config, dict(module_config.static_ro_per_sample_budgets)


def _history_keys(module_config: BlueReelsVDDModuleConfig) -> List[str]:
    return [ro for ro, _nro in module_config.merge_item_uih_feature_mapping]


class StaticRoBudgetTest(TestCase):
    def test_default_path_still_pads_to_the_model_bound(self) -> None:
        """Without the budgeted sampler nothing holds a smaller total down."""
        module_config, budgets = _budgets()
        for key in _history_keys(module_config):
            self.assertEqual(budgets[key], module_config.hstu_max_seq_len)

    def test_budgeted_path_lands_between_the_sampled_length_and_the_bound(
        self,
    ) -> None:
        """The two properties the derived budget has to satisfy.

        Deliberately NOT a re-derivation of ``P * N + (1 - P) * T``: copying
        the formula into the test makes it agree with a wrong implementation.
        These are the bounds the design actually rests on.

        The floor is the one that matters. Below the sampled length even
        sampling every eligible row overshoots, so ``imbalance_load`` goes
        non-positive and the inherited rule falls back to sampling everyone --
        the guarantee is lost silently, with no error and no shape change.
        """
        module_config, budgets = _budgets(budgeted_seq_len_sampling=True)
        _distribution_type, strategy = none_throws(
            module_config.hstu_seq_len_sampling_configs
        )
        assert isinstance(strategy, SequenceLengthSamplingStrategy)
        max_seq_len = float(module_config.hstu_max_seq_len)
        sampled_len = float(strategy.beta) * max_seq_len ** (
            float(strategy.alpha) / 2.0
        )

        history_budgets = {budgets[key] for key in _history_keys(module_config)}
        self.assertEqual(
            len(history_budgets),
            1,
            "the sequential group shares one budget; a split would mean the "
            "padder and the sampler are enforcing different numbers",
        )
        budget = history_budgets.pop()
        self.assertGreaterEqual(
            budget,
            sampled_len,
            "a budget under the sampled length cannot be met, and the "
            "fallback is silent",
        )
        self.assertLess(
            budget,
            module_config.hstu_max_seq_len,
            "a budget at or above the model bound saves no padding at all",
        )

    def test_budgeted_path_moves_no_model_shape(self) -> None:
        """The budget is a padding size, not a sequence bound."""
        default_config, _ = _budgets()
        budgeted_config, _ = _budgets(budgeted_seq_len_sampling=True)
        for field in ("hstu_max_seq_len", "hstu_l2_max_seq_len", "hstu_max_output_len"):
            self.assertEqual(
                getattr(budgeted_config, field),
                getattr(default_config, field),
                f"{field} moved; the budget must not truncate history, "
                "re-scale the sampler, or change the attention scale",
            )

    def test_contextual_keys_keep_their_own_max_length(self) -> None:
        """The budget is scoped to the sequential group.

        Contextual budgets come from ``contextual_feature_to_max_length``, so
        sweeping them into the derived budget would change what those features
        carry.
        """
        module_config, budgets = _budgets(budgeted_seq_len_sampling=True)
        for key, max_len in module_config.contextual_feature_to_max_length:
            self.assertEqual(budgets[key], int(max_len))
