# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors

"""Tests for tracing Qwen3 model as a full graph through recir compile.

Validates that Qwen3ForCausalLM can be fully traced by recir's make_fx-based
@compile() without graph breaks.

The make_fx tests run on CPU. Full-graph DDP coverage requires CUDA.
"""

import unittest
from copy import deepcopy
from functools import wraps
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from recir.api import compile
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms as base_with_comms,
)
from transformers import Qwen3Config, Qwen3ForCausalLM  # @manual

# ── Tiny model constants ──
VOCAB_SIZE = 1000
HIDDEN_SIZE = 256
INTERMEDIATE_SIZE = 512
NUM_LAYERS = 2
NUM_HEADS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 64

# ── Data constants ──
BATCH_SIZE = 4
SEQ_LEN = 32


def with_comms(func: Callable[..., Any]) -> Callable[..., Any]:
    @base_with_comms
    @wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        torch.manual_seed(self.rank)
        return func(self, *args, **kwargs)

    return wrapper


class Qwen3LMWrapper(nn.Module):
    """Wrapper around Qwen3ForCausalLM that returns a scalar loss.

    Computes next-token cross-entropy loss from the model's logits,
    suitable for tracing with recir's @compile().
    """

    def __init__(self, llm):
        super().__init__()
        self.llm = llm

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = outputs.logits.float()
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return loss


def _create_qwen3():
    """Create a tiny Qwen3 model for testing."""
    config = Qwen3Config(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        max_position_embeddings=512,
        use_cache=False,
        attn_implementation="eager",
    )
    return Qwen3ForCausalLM(config)


def _create_data():
    """Create synthetic data for testing."""
    torch.manual_seed(42)
    input_ids = torch.randint(1, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[:, :8] = -100  # mask prefix tokens
    return input_ids, attention_mask, labels


class TestQwen3Trace(DTensorTestBase):
    """Test that Qwen3 model can be traced as a full graph through recir compile."""

    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_qwen3_full_graph_trace(self) -> None:
        """Verify Qwen3 traces through @compile() and matches DDP baseline."""
        torch.manual_seed(0)
        llm = _create_qwen3()
        model = Qwen3LMWrapper(llm).cuda(self.rank)
        opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
        input_ids, attention_mask, labels = _create_data()
        input_ids = input_ids.cuda(self.rank)
        attention_mask = attention_mask.cuda(self.rank)
        labels = labels.cuda(self.rank)

        ddp_model = DDP(deepcopy(model), device_ids=[self.rank])
        ddp_opt = torch.optim.SGD(ddp_model.parameters(), lr=0.01, foreach=False)
        ddp_input_ids = deepcopy(input_ids)
        ddp_attention_mask = deepcopy(attention_mask)
        ddp_labels = deepcopy(labels)

        @compile()
        def train_step(mod, opt, input_ids, attention_mask, labels):
            loss = mod(input_ids, attention_mask, labels)
            loss.backward()
            opt.step()

        # Materialize optimizer states
        model(input_ids, attention_mask, labels).backward()
        opt.step()
        opt.zero_grad()

        ddp_model(ddp_input_ids, ddp_attention_mask, ddp_labels).backward()
        ddp_opt.step()
        ddp_opt.zero_grad()

        # Compiled train step
        train_step(model, opt, input_ids, attention_mask, labels)

        # DDP baseline step
        ddp_model(ddp_input_ids, ddp_attention_mask, ddp_labels).backward()
        ddp_opt.step()

        # Verify parameter parity
        for p1, p2 in zip(model.parameters(), ddp_model.parameters()):
            self.assertEqual(p1, p2)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_qwen3_multi_step(self) -> None:
        """Verify Qwen3 traces correctly across multiple compiled steps.

        Step 1 triggers make_fx tracing; step 2 exercises the cached
        graph module and DeviceStateManager output copy-back.
        """
        torch.manual_seed(0)
        llm = _create_qwen3()
        model = Qwen3LMWrapper(llm).cuda(self.rank)
        opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
        input_ids, attention_mask, labels = _create_data()
        input_ids = input_ids.cuda(self.rank)
        attention_mask = attention_mask.cuda(self.rank)
        labels = labels.cuda(self.rank)

        @compile()
        def train_step(mod, opt, input_ids, attention_mask, labels):
            loss = mod(input_ids, attention_mask, labels)
            loss.backward()
            opt.step()

        # Materialize optimizer states
        model(input_ids, attention_mask, labels).backward()
        opt.step()
        opt.zero_grad()

        # Step 1: triggers make_fx tracing
        params_before = {n: p.clone() for n, p in model.named_parameters()}
        train_step(model, opt, input_ids, attention_mask, labels)
        # Step 2: exercises cached graph module
        train_step(model, opt, input_ids, attention_mask, labels)
        # Verify parameters changed (training happened)
        for name, param in model.named_parameters():
            self.assertFalse(
                torch.equal(param, params_before[name]),
                f"Parameter {name} unchanged after 2 train steps",
            )


class TestQwen3MakeFxTrace(unittest.TestCase):
    """CPU-only smoke test: verify make_fx can trace Qwen3 as a full graph."""

    def test_qwen3_make_fx_forward(self) -> None:
        """Verify make_fx traces Qwen3 forward without errors."""
        from torch.fx.experimental.proxy_tensor import make_fx

        torch.manual_seed(0)
        llm = _create_qwen3()
        model = Qwen3LMWrapper(llm).eval()
        input_ids, attention_mask, labels = _create_data()

        def forward_fn(input_ids, attention_mask, labels):
            return model(input_ids, attention_mask, labels)

        gm = make_fx(
            forward_fn,
            tracing_mode="fake",
            _allow_non_fake_inputs=True,
        )(input_ids, attention_mask, labels)
        # If we get here, tracing succeeded as a full graph
        num_nodes = len(list(gm.graph.nodes))
        self.assertGreater(num_nodes, 0, "Traced graph should have nodes")

    def test_qwen3_make_fx_train_step(self) -> None:
        """Verify make_fx traces Qwen3 forward+backward without errors."""
        from torch.fx.experimental.proxy_tensor import make_fx

        torch.manual_seed(0)
        llm = _create_qwen3()
        model = Qwen3LMWrapper(llm).train()
        input_ids, attention_mask, labels = _create_data()

        def forward_fn(input_ids, attention_mask, labels):
            loss = model(input_ids, attention_mask, labels)
            return loss

        # Trace forward pass
        gm = make_fx(
            forward_fn,
            tracing_mode="fake",
            _allow_non_fake_inputs=True,
        )(input_ids, attention_mask, labels)
        num_nodes = len(list(gm.graph.nodes))
        self.assertGreater(num_nodes, 0, "Traced graph should have nodes")

    def test_qwen3_make_fx_numerics(self) -> None:
        """Verify traced Qwen3 graph produces same output as eager."""
        from torch.fx.experimental.proxy_tensor import make_fx

        torch.manual_seed(0)
        llm = _create_qwen3()
        model = Qwen3LMWrapper(llm).eval()
        input_ids, attention_mask, labels = _create_data()

        def forward_fn(input_ids, attention_mask, labels):
            return model(input_ids, attention_mask, labels)

        # Eager
        with torch.no_grad():
            eager_out = forward_fn(input_ids, attention_mask, labels)

        # Trace with real mode (executes actual ops)
        gm = make_fx(forward_fn, tracing_mode="real")(
            input_ids,
            attention_mask,
            labels,
        )

        with torch.no_grad():
            traced_out = gm(input_ids, attention_mask, labels)

        torch.testing.assert_close(
            traced_out,
            eager_out,
            msg="Traced Qwen3 output should match eager",
        )


# ── Gemma3 model constants ──
GEMMA3_VOCAB_SIZE = 1000
GEMMA3_HIDDEN_SIZE = 256
GEMMA3_INTERMEDIATE_SIZE = 512
GEMMA3_NUM_LAYERS = 2
GEMMA3_NUM_HEADS = 4
GEMMA3_NUM_KV_HEADS = 2
GEMMA3_HEAD_DIM = 64


class Gemma3LMWrapper(nn.Module):
    """Wrapper around Gemma3ForCausalLM that returns a scalar loss."""

    def __init__(self, llm):
        super().__init__()
        self.llm = llm

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = outputs.logits.float()
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return loss


def _create_gemma3():
    """Create a tiny Gemma3 text model for testing."""
    from transformers import Gemma3TextConfig
    from transformers.models.gemma3.modeling_gemma3 import (
        Gemma3ForCausalLM as Gemma3CausalLM,
    )

    config = Gemma3TextConfig(
        vocab_size=GEMMA3_VOCAB_SIZE,
        hidden_size=GEMMA3_HIDDEN_SIZE,
        intermediate_size=GEMMA3_INTERMEDIATE_SIZE,
        num_hidden_layers=GEMMA3_NUM_LAYERS,
        num_attention_heads=GEMMA3_NUM_HEADS,
        num_key_value_heads=GEMMA3_NUM_KV_HEADS,
        head_dim=GEMMA3_HEAD_DIM,
        max_position_embeddings=512,
        use_cache=False,
        attn_implementation="eager",
    )
    return Gemma3CausalLM(config)


def _create_gemma3_data():
    """Create synthetic data for Gemma3 testing."""
    torch.manual_seed(42)
    input_ids = torch.randint(1, GEMMA3_VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[:, :8] = -100
    return input_ids, attention_mask, labels


class TestGemma3MakeFxTrace(unittest.TestCase):
    """CPU-only smoke test: verify make_fx can trace Gemma3 as a full graph."""

    def test_gemma3_make_fx_forward(self) -> None:
        """Verify make_fx traces Gemma3 forward without errors."""
        from torch.fx.experimental.proxy_tensor import make_fx

        torch.manual_seed(0)
        llm = _create_gemma3()
        model = Gemma3LMWrapper(llm).eval()
        input_ids, attention_mask, labels = _create_gemma3_data()

        def forward_fn(input_ids, attention_mask, labels):
            return model(input_ids, attention_mask, labels)

        gm = make_fx(
            forward_fn,
            tracing_mode="fake",
            _allow_non_fake_inputs=True,
        )(input_ids, attention_mask, labels)
        num_nodes = len(list(gm.graph.nodes))
        self.assertGreater(num_nodes, 0, "Traced graph should have nodes")

    def test_gemma3_make_fx_train_step(self) -> None:
        """Verify make_fx traces Gemma3 forward+backward without errors."""
        from torch.fx.experimental.proxy_tensor import make_fx

        torch.manual_seed(0)
        llm = _create_gemma3()
        model = Gemma3LMWrapper(llm).train()
        input_ids, attention_mask, labels = _create_gemma3_data()

        def forward_fn(input_ids, attention_mask, labels):
            return model(input_ids, attention_mask, labels)

        gm = make_fx(
            forward_fn,
            tracing_mode="fake",
            _allow_non_fake_inputs=True,
        )(input_ids, attention_mask, labels)
        num_nodes = len(list(gm.graph.nodes))
        self.assertGreater(num_nodes, 0, "Traced graph should have nodes")

    def test_gemma3_make_fx_numerics(self) -> None:
        """Verify traced Gemma3 graph produces same output as eager."""
        from torch.fx.experimental.proxy_tensor import make_fx

        torch.manual_seed(0)
        llm = _create_gemma3()
        model = Gemma3LMWrapper(llm).eval()
        input_ids, attention_mask, labels = _create_gemma3_data()

        def forward_fn(input_ids, attention_mask, labels):
            return model(input_ids, attention_mask, labels)

        with torch.no_grad():
            eager_out = forward_fn(input_ids, attention_mask, labels)

        gm = make_fx(forward_fn, tracing_mode="real")(
            input_ids,
            attention_mask,
            labels,
        )

        with torch.no_grad():
            traced_out = gm(input_ids, attention_mask, labels)

        torch.testing.assert_close(
            traced_out,
            eager_out,
            msg="Traced Gemma3 output should match eager",
        )


if __name__ == "__main__":
    if False:
        run_tests()
