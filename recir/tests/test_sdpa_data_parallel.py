# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors

from __future__ import annotations

"""Multi-GPU DataParallel tests for models using scaled_dot_product_attention.

Tests that recir's @compile() with DataParallel correctly handles SDPA models
on multiple GPUs. recir uses collectives_insertion (parametrization-based) so
activations flow as plain local tensors and SDPA's multi-output placement
issue does not arise.
"""

from copy import deepcopy
from functools import wraps
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from recir.api import compile
from recir.distributed.parallel_mode import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms as base_with_comms,
)

# ── Tiny model constants ──
VOCAB_SIZE = 1000
HIDDEN_SIZE = 256
NUM_HEADS = 4
HEAD_DIM = 64
BATCH_SIZE = 4
SEQ_LEN = 32


def with_comms(func: Callable[..., Any]) -> Callable[..., Any]:
    @base_with_comms
    @wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        torch.manual_seed(self.rank)
        return func(self, *args, **kwargs)

    return wrapper


class SdpaModel(nn.Module):
    """Minimal model using F.scaled_dot_product_attention."""

    def __init__(
        self, hidden_size: int, num_heads: int, head_dim: int, vocab_size: int
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.num_heads = num_heads
        self.head_dim = head_dim

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, -1)
        logits = self.lm_head(self.o_proj(attn_out))
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
        )
        return loss


def _create_sdpa_model():
    return SdpaModel(HIDDEN_SIZE, NUM_HEADS, HEAD_DIM, VOCAB_SIZE)


def _create_data(rank: int = 0):
    torch.manual_seed(42 + rank)
    input_ids = torch.randint(1, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    labels = input_ids.clone()
    labels[:, :8] = -100
    return input_ids, labels


class TestSdpaDataParallelV2(DTensorTestBase):
    """Test recir @compile() + DataParallel with SDPA models on 2 GPUs.

    DataParallel inserts collectives via parametrization.  Activations flow
    as plain local tensors, so SDPA's multi-output ops (philox seed/offset)
    are never assigned DTensor placements.
    """

    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_sdpa_v2_numerics_parity_with_ddp(self) -> None:
        """Verify v2 @compile() produces the same params as DDP after 1 step."""
        torch.manual_seed(0)
        model = _create_sdpa_model().bfloat16().cuda(self.rank)
        opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
        input_ids, labels = _create_data()
        input_ids = input_ids.cuda(self.rank)
        labels = labels.cuda(self.rank)

        # DDP baseline
        ddp_model = DDP(deepcopy(model), device_ids=[self.rank])
        ddp_opt = torch.optim.SGD(ddp_model.parameters(), lr=0.01, foreach=False)
        ddp_input_ids = deepcopy(input_ids)
        ddp_labels = deepcopy(labels)

        # Materialize optimizer states
        model(input_ids, labels).backward()
        opt.step()
        opt.zero_grad()

        ddp_model(ddp_input_ids, ddp_labels).backward()
        ddp_opt.step()
        ddp_opt.zero_grad()

        @compile(parallel_mode=DataParallel("replicate"))
        def train_step(mod, opt, input_ids, labels):
            loss = mod(input_ids, labels)
            loss.backward()
            opt.step()

        train_step(model, opt, input_ids, labels)

        ddp_model(ddp_input_ids, ddp_labels).backward()
        ddp_opt.step()

        for p1, p2 in zip(model.parameters(), ddp_model.parameters()):
            torch.testing.assert_close(p1, p2)


if __name__ == "__main__":
    run_tests()
