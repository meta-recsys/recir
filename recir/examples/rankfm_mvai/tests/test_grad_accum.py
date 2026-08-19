#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""Pin gradient accumulation against a real DDP all-reduce.

The recir CPU-eager leg cannot hold a 2048-row batch -- the eager attention
kernel keeps every ``[B, H, N, N]`` score matrix alive for backward, and oomd
killed a 2048 batch at ~388GB while 4x512 peaks near 112GB. So it accumulates K
micro-batches to reach the same global batch the light leg gets from K GPUs.

The light leg computes a per-rank mean on each rank and lets DDP
ALL-REDUCE-MEAN the gradients. Reproducing that with accumulation means scaling
each micro-batch by ``1/K``.

The original implementation scaled by ``n_i / n_total`` instead, which telescopes
to the gradient of one global mean over the concatenated batch. That is a
different quantity whenever the per-micro-batch candidate counts differ, and on
the real dump adjacent batches differ by ~19%. The error compounds every step,
so it presented as slow monotonic NE divergence between the two legs and
survived a bit-exact single-forward parity test.

These tests therefore use DELIBERATELY UNEQUAL per-batch counts. With equal
counts both scalings agree and the bug is invisible -- which is precisely why it
went unnoticed.
"""

from typing import List, Optional

import torch
from later.unittest import TestCase

_K: int = 4
# Unequal on purpose, mirroring the spread seen on the dump (2254..2681).
_ROWS: List[int] = [3, 5, 2, 6]


def _make_model() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Linear(4, 1, bias=False)


def _batches() -> List[torch.Tensor]:
    torch.manual_seed(1)
    return [torch.randn(n, 4, dtype=torch.float64) for n in _ROWS]


def _grads(model: torch.nn.Module) -> torch.Tensor:
    parts: List[torch.Tensor] = []
    for p in model.parameters():
        grad = p.grad
        assert grad is not None, "backward did not populate a gradient"
        parts.append(grad.reshape(-1).clone())
    return torch.cat(parts)


def _ddp_reference(batches: List[torch.Tensor]) -> torch.Tensor:
    """(1/K) * sum_r grad(mean_r) -- what DDP's all-reduce-mean produces.

    This is DDP's documented semantics: it all-reduces gradients with SUM and
    divides by world_size, so each rank's contribution is weighted equally
    regardless of how many rows it drew. Encoded directly rather than spun up
    over a process group -- a single-rank all-reduce would only check that a
    mean equals a mean, which is not evidence of anything.
    """
    model = _make_model().double()
    acc: Optional[torch.Tensor] = None
    for x in batches:
        model.zero_grad(set_to_none=True)
        loss = model(x).mean()
        loss.backward()
        g = _grads(model)
        acc = g if acc is None else acc + g
    assert acc is not None, "no batches supplied"
    return acc / len(batches)


def _accumulate(batches: List[torch.Tensor], mode: str) -> torch.Tensor:
    """Run the pipeline's accumulation with the given per-micro-batch scaling."""
    model = _make_model().double()
    model.zero_grad(set_to_none=True)
    counts = [x.shape[0] for x in batches]
    n_total = sum(counts)
    for x, n_i in zip(batches, counts):
        loss = model(x).mean()
        if mode == "one_over_k":
            loss = loss / len(batches)
        elif mode == "n_i_over_total":
            loss = loss * (n_i / n_total)
        else:
            raise ValueError(mode)
        loss.backward()
    return _grads(model)


class GradAccumTest(TestCase):
    def test_one_over_k_matches_ddp(self) -> None:
        """The shipped scaling must equal DDP's all-reduce-mean."""
        batches = _batches()
        torch.testing.assert_close(
            _accumulate(batches, "one_over_k"),
            _ddp_reference(batches),
            rtol=0,
            atol=1e-12,
            msg="1/K accumulation diverged from the DDP all-reduce-mean",
        )

    def test_n_i_over_total_does_not_match_ddp(self) -> None:
        """The original scaling must NOT match, or the fix was a no-op.

        Guards against someone 'simplifying' the scale back to n_i/n_total, and
        proves the test above is actually discriminating rather than passing on
        a degenerate fixture.
        """
        batches = _batches()
        diff = (
            (_accumulate(batches, "n_i_over_total") - _ddp_reference(batches))
            .abs()
            .max()
        )
        self.assertGreater(
            float(diff),
            1e-6,
            "n_i/n_total agrees with DDP here, so the fixture has equal "
            "per-batch counts and these tests cannot detect the regression",
        )

    def test_equal_counts_hide_the_bug(self) -> None:
        """Documents why this went unnoticed: with equal n_i both scalings agree."""
        torch.manual_seed(1)
        equal = [torch.randn(4, 4, dtype=torch.float64) for _ in range(_K)]
        torch.testing.assert_close(
            _accumulate(equal, "n_i_over_total"),
            _accumulate(equal, "one_over_k"),
            rtol=0,
            atol=1e-12,
        )

    def test_single_micro_batch_is_unscaled(self) -> None:
        """K=1 must be a plain backward, matching pre-accumulation behaviour."""
        batches = _batches()[:1]
        model = _make_model().double()
        model.zero_grad(set_to_none=True)
        model(batches[0]).mean().backward()
        torch.testing.assert_close(
            _accumulate(batches, "one_over_k"), _grads(model), rtol=0, atol=1e-12
        )
