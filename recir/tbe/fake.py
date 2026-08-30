# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Meta implementations for FBGEMM TBE operators used by ReCIR tracing."""

from __future__ import annotations

from typing import Any

import fbgemm_gpu.split_table_batched_embeddings_ops_training  # noqa: F401
import torch


_NO_BAG_ROWWISE_ADAGRAD_BACKWARD_OPS = (
    "split_embedding_nobag_backward_codegen_rowwise_adagrad_unweighted_pt2_wrapper",
    "split_embedding_nobag_backward_codegen_rowwise_adagrad_unweighted_exact_cuda",
)


def _meta_nobag_rowwise_adagrad_backward_update(
    grad_output: torch.Tensor,
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor:
    """Model the zero-sized autograd placeholder returned by fused update."""
    del args, kwargs
    return grad_output.new_empty((0,))


for op_name in _NO_BAG_ROWWISE_ADAGRAD_BACKWARD_OPS:
    torch.library.register_fake(f"fbgemm::{op_name}")(
        _meta_nobag_rowwise_adagrad_backward_update
    )
