# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Pluggable embedding backends for recsys models on recir.

Two backends share the ``EmbeddingBackend`` ABC from ``backend.py``:

  CPUEmbeddingBackend (cpu.py)        - torchrec DMP on CPU via gloo PG.
                                        Small tables -> TABLE_WISE, large
                                        -> ROW_WISE.

  TPUEmbeddingBackend (tpu.py)        - each table's TablePlan picks its
                                        compute kernel: ring shift, traced
                                        inside the compiled GSPMD program, or
                                        SparseCore (sparsecore_rowwise.py).

Both are model-agnostic. Each binds the torchrec ``EmbeddingCollection``
the model names by FQN; the model pairs the backend with its own tracing
dispatchers.

``factory.build_embedding_backend`` picks a backend by name and so links
both. A binary that needs only one should construct it from its own submodule
instead, so it does not link torchrec.distributed.* (heavy) or
jax_tpu_embedding. This package re-exports only the interface.
"""

from __future__ import annotations

from recir.embedding.backend import (
    EmbeddingBackend,
    EmbeddingSetupResult,
    SparseOptimizerRuntime,
    SparseUpdatePlacement,
)

__all__ = [
    "EmbeddingBackend",
    "EmbeddingSetupResult",
    "SparseOptimizerRuntime",
    "SparseUpdatePlacement",
]
