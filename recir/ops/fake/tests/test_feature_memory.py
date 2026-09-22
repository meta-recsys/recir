#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from __future__ import annotations

import unittest
from dataclasses import dataclass
from unittest import mock

import torch
from hammer.ops.jagged import split_2D_jagged_jagged
from hammer.utils import HammerKernel
from hammer.v3.modules.enrichment.feature_memory import FeatureMemory
from recir.lowering.partial_lower import partial_lower
from torch._dynamo.source import LocalSource
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import (
    DimDynamic,
    ShapeEnv,
    StatelessSymbolicContext,
)


@dataclass
class _FeatureMemoryConfig:
    feature_embedding_dim: int = 4
    projector_hidden_dim: int = 0
    cache_loss_weight: float = 1.0


def _make_feature_memory(*, custom_kernel: bool) -> FeatureMemory:
    torch.manual_seed(0)
    return FeatureMemory(
        embedding_dim=4,
        feature_memory_config=_FeatureMemoryConfig(),
        custom_kernel=custom_kernel,
    )


def _sample_inputs(device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        torch.randn(5, 4, device=device),
        torch.randn(2, 4, device=device),
        torch.tensor([3, 2], dtype=torch.int64, device=device),
        torch.tensor([1, 1], dtype=torch.int64, device=device),
        torch.tensor([0, 3, 5], dtype=torch.int64, device=device),
    )


def _resized_inputs(device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        torch.randn(6, 4, device=device),
        torch.randn(3, 4, device=device),
        torch.tensor([4, 2], dtype=torch.int64, device=device),
        torch.tensor([2, 1], dtype=torch.int64, device=device),
        torch.tensor([0, 4, 6], dtype=torch.int64, device=device),
    )


def _dynamic_fake_args(
    args: tuple[torch.Tensor, ...],
) -> tuple[FakeTensorMode, tuple[torch.Tensor, ...]]:
    fake_mode = FakeTensorMode(allow_non_fake_inputs=True, shape_env=ShapeEnv())
    fake_args = []
    for index, arg in enumerate(args):
        fake_args.append(
            fake_mode.from_tensor(
                arg,
                static_shapes=False,
                source=LocalSource(f"feature_memory_input_{index}"),
                symbolic_context=StatelessSymbolicContext(
                    dynamic_sizes=[DimDynamic.DYNAMIC]
                    + [DimDynamic.STATIC] * (arg.ndim - 1)
                ),
            )
        )
    return fake_mode, tuple(fake_args)


class FeatureMemoryTracingTest(unittest.TestCase):
    def _assert_explicit_split_size_parity(
        self,
        device: torch.device,
        kernel: HammerKernel,
    ) -> None:
        torch.manual_seed(0)
        expected_values = torch.randn(6, 4, device=device, requires_grad=True)
        actual_values = expected_values.detach().clone().requires_grad_()
        offsets_left = torch.tensor([0, 2, 3], device=device)
        num_nro_rows = 3
        offsets_right = torch.tensor([0, 2, num_nro_rows], device=device)

        with mock.patch(
            "hammer.ops.jagged.get_split_2d_jagged_kernel",
            return_value=None,
        ):
            expected = split_2D_jagged_jagged(
                values=expected_values,
                max_seq_len=4,
                offsets_left=offsets_left,
                offsets_right=offsets_right,
                kernel=kernel,
            )
            actual = split_2D_jagged_jagged(
                values=actual_values,
                max_seq_len=4,
                offsets_left=offsets_left,
                offsets_right=offsets_right,
                kernel=kernel,
                seq_len_a=actual_values.shape[0] - num_nro_rows,
                seq_len_b=num_nro_rows,
            )

        for actual_part, expected_part in zip(actual, expected):
            torch.testing.assert_close(actual_part, expected_part)
        expected_loss = expected[0].square().sum() + expected[1].square().sum()
        actual_loss = actual[0].square().sum() + actual[1].square().sum()
        expected_loss.backward()
        actual_loss.backward()
        torch.testing.assert_close(actual_values.grad, expected_values.grad)

    def test_explicit_split_sizes_preserve_pytorch_behavior(self) -> None:
        self._assert_explicit_split_size_parity(
            torch.device("cpu"),
            HammerKernel.PYTORCH,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_explicit_split_sizes_preserve_triton_behavior(self) -> None:
        self._assert_explicit_split_size_parity(
            torch.device("cuda"),
            HammerKernel.TRITON,
        )

    def test_lowered_forward_accepts_changed_candidate_count(self) -> None:
        device = torch.device("cpu")
        expected_model = _make_feature_memory(custom_kernel=False).eval()
        sample = _sample_inputs(device)

        def forward(
            content_embeddings: torch.Tensor,
            nro_feature_embeddings: torch.Tensor,
            past_lengths: torch.Tensor,
            num_targets: torch.Tensor,
            input_seq_offsets: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return expected_model(
                content_embeddings,
                nro_feature_embeddings,
                past_lengths,
                num_targets,
                4,
                input_seq_offsets,
            )

        fake_mode, fake_args = _dynamic_fake_args(sample)
        with fake_mode:
            traced = make_fx(
                forward,
                tracing_mode="fake",
                _allow_non_fake_inputs=False,
            )(*fake_args)
        for placeholder in (
            node for node in traced.graph.nodes if node.op == "placeholder"
        ):
            placeholder.meta["is_batch_input"] = True
            placeholder.meta["batch_dim"] = 0
        lowered = partial_lower(traced)

        resized = _resized_inputs(device)
        expected = expected_model(
            resized[0],
            resized[1],
            resized[2],
            resized[3],
            4,
            resized[4],
        )
        actual = lowered(*resized)
        torch.testing.assert_close(actual, expected)
