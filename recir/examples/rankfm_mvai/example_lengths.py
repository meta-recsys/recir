#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Example-length accounting for the snapshot channel."""

from __future__ import annotations

from typing import Any, Optional

import torch


def real_example_lengths(
    uih_lengths: Optional[torch.Tensor],
    num_candidates: Optional[torch.Tensor],
    num_groups: int,
) -> Optional[torch.Tensor]:
    """Mean real example length, one row per sequence group."""
    if num_groups != 1:
        return None
    if uih_lengths is None or num_candidates is None:
        return None
    if uih_lengths.shape != num_candidates.shape:
        return None
    total = (uih_lengths.float() + num_candidates.float()).mean().reshape(1)
    return total.reshape(1, 1)


def num_sequence_groups(model: Any) -> int:
    """How many HSTU sequence groups the model's MoT preprocessor runs."""
    inner = getattr(model, "_model", model)
    stack = getattr(getattr(inner, "recblock", None), "preprocessor_stack", None)
    # Dynamic attr set by PreprocessorStack -- the same access
    # rankfm_dispatchers uses to reach _needs_unsampled_inputs.
    count = getattr(getattr(stack, "mot_preprocessor", None), "num_sequence", None)
    return int(count) if isinstance(count, int) and count > 0 else 1
