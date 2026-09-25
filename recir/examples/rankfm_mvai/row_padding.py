# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Row-axis padding for the per-host dispatch boundary."""

from typing import List, Tuple

import torch
import torch.nn.functional as F


def fit_to_shape(
    t: torch.Tensor, target_shape: torch.Size | Tuple[int, ...]
) -> torch.Tensor:
    """End-pad with zeros (or narrow) each dim of *t* to ``target_shape``."""
    target = tuple(int(s) for s in target_shape)
    if tuple(t.shape) == target:
        return t
    pad_widths: List[int] = []
    for actual, tgt in zip(reversed(t.shape), reversed(target)):
        pad_widths += [0, max(0, int(tgt) - int(actual))]
    out = F.pad(t, pad_widths) if any(p > 0 for p in pad_widths) else t
    for d, (actual, tgt) in enumerate(zip(out.shape, target)):
        if int(actual) > int(tgt):
            out = out.narrow(d, 0, int(tgt))
    return out


def fit_rows_chip_major(
    t: torch.Tensor, target_rows: int, num_local: int
) -> torch.Tensor:
    """Pad a ROW-axis dense tensor chip-major: per chip, real rows then dummies."""
    current = int(t.shape[0])
    if current == target_rows:
        return t
    if (
        current > target_rows
        or num_local <= 1
        or current % num_local != 0
        or target_rows % num_local != 0
    ):
        return fit_to_shape(t, (target_rows, *t.shape[1:]))
    rows_per_chip = current // num_local
    padded_per_chip = target_rows // num_local
    out = t.new_zeros((target_rows, *t.shape[1:]))
    out.reshape(num_local, padded_per_chip, *t.shape[1:])[:, :rows_per_chip] = (
        t.reshape(num_local, rows_per_chip, *t.shape[1:])
    )
    return out
