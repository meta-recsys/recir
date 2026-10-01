# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""CoScaling bindings for the recir embedding backends."""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn
from recir.embedding.cpu import CPUEmbeddingBackend
from recir.examples.rankfm_mvai.rankfm_embedding_backends import build_eager_overrides

COSCALING_EMBEDDING_PARAM_PREFIX: str = (
    "pipeline.tokenizer_stack.mot_sparse_arch._sparse_arch._embedding_collection"
)


def coscaling_sparse_arch(model: nn.Module) -> nn.Module:
    """The ``ECSparseArch`` that owns the sequence ``_embedding_collection``."""
    # nn.Module.__getattr__ is typed Module | Tensor, so the chain needs Any.
    pipeline: Any = model.pipeline
    return pipeline.tokenizer_stack.mot_sparse_arch._sparse_arch


class CPUCoScalingBackend(CPUEmbeddingBackend):
    def get_override(
        self,
        pipeline: Any,
        batch_sample: Any,
        device: torch.device,
        global_bs: int,
        num_ranks: int,
        num_processes: int,
    ) -> List[Any]:
        return build_eager_overrides(
            pipeline,
            batch_sample,
            device,
            batch_size=global_bs,
            num_processes=num_processes,
            sparse_arch_getter=coscaling_sparse_arch,
        )


def build_cpu_backend(embedding_tables: Dict[str, Any]) -> CPUCoScalingBackend:
    # Same int64-offsets fix the rankfm cpu backend applies before DMP's
    # sequence all-to-all; see recir/embedding/_torchrec_patches.
    from recir.embedding._torchrec_patches import apply_seq_a2a_int64_patch

    apply_seq_a2a_int64_patch()
    return CPUCoScalingBackend(
        embedding_param_prefix=COSCALING_EMBEDDING_PARAM_PREFIX,
        embedding_tables=embedding_tables,
    )
