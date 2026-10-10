# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from __future__ import annotations

import copy
import logging

import torch
from generative_recommenders.modules.preprocessors import ContextualPreprocessor
from recir.lowering.partial_lower import partial_lower
from torch.fx.experimental.proxy_tensor import make_fx

logger: logging.Logger = logging.getLogger(__name__)


class _ThreeBranches(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.preprocessor = ContextualPreprocessor(
            input_embedding_dim=64,
            hidden_dim=256,
            output_embedding_dim=512,
            contextual_feature_to_max_length={"user_id": 1},
            contextual_feature_to_min_uih_length={"user_id": 20},
            action_embedding_dim=8,
            action_feature_name="item_weights",
            action_weights=[1, 2, 4, 8, 16],
            additional_embedding_features=["item_category_id"],
            is_inference=False,
        )

    def forward(
        self,
        content: torch.Tensor,
        additional: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return (
                self.preprocessor._content_embedding_mlp(content)
                + self.preprocessor._additional_embedding_mlp(additional)
                + self.preprocessor._action_embedding_mlp(action)
            )


def _difference(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    difference = (actual - expected).abs()
    relative = difference / expected.abs().clamp_min(torch.finfo(expected.dtype).tiny)
    return float(difference.max()), float(relative.max())


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    torch.manual_seed(1234)
    device = torch.device("cuda")
    module = _ThreeBranches().to(device).train()
    inputs = (
        torch.randn(192, 64, device=device),
        torch.randn(192, 64, device=device),
        torch.randn(
            192,
            module.preprocessor._action_encoder.output_embedding_dim,
            device=device,
        ),
    )
    expected = module(*inputs)

    compiled = torch.compile(copy.deepcopy(module), fullgraph=True)
    vanilla = compiled(*inputs)
    max_abs, max_rel = _difference(vanilla, expected)
    logger.info("torch.compile: max_abs=%g max_rel=%g", max_abs, max_rel)

    gm = make_fx(copy.deepcopy(module))(*inputs)
    traced = gm(*inputs)
    lowered = partial_lower(copy.deepcopy(gm))
    recir = lowered(*inputs)

    for name, output in (
        ("make_fx", traced),
        ("partial_lower", recir),
    ):
        max_abs, max_rel = _difference(output, expected)
        logger.info("%s: max_abs=%g max_rel=%g", name, max_abs, max_rel)


if __name__ == "__main__":
    main()
