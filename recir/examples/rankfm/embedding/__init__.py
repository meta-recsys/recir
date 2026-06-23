# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Pluggable embedding backends for RankFM: CPU/torchrec or JAX/SparseCore.

Two backends share the EmbeddingBackend ABC from recir.sparsecore.backend:

  CPURankFMBackend (cpu_backend.py)  - torchrec DMP on CPU via gloo PG.
                                        Small tables -> TABLE_WISE, large
                                        -> ROW_WISE.

  JAXRankFMBackend (jax_backend.py)  - jax_tpu_embedding SparseCore:
                                        tables as EmbeddingVariables sharded
                                        on SparseCore HW, lookup via
                                        tpu_sparse_dense_matmul, Adagrad
                                        fused on-chip. No gloo PG required.

Selected at runtime via RankFMConfig.embedding_backend ("cpu" | "tpu").

The two backend classes live in separate submodules so importing this
package's factory does not pull in torchrec.distributed.* (heavy) or
jax_tpu_embedding unless the corresponding backend is actually requested.
"""

from __future__ import annotations

from recir.sparsecore.backend import EmbeddingBackend


def build_embedding_backend(
    embedding_backend: str,
    *,
    ringshift_min_rows: int = 4096,
    fsdp_min_rows: int = -1,
    on_device_init: bool = True,
) -> EmbeddingBackend:
    if embedding_backend == "cpu":
        from .cpu_backend import CPURankFMBackend

        return CPURankFMBackend()
    if embedding_backend == "tpu":
        from .jax_backend import JAXRankFMBackend

        return JAXRankFMBackend()
    if embedding_backend == "in_graph":
        from .in_graph_embedding_backend import InGraphEmbeddingBackend

        return InGraphEmbeddingBackend(
            ringshift_min_rows=ringshift_min_rows,
            fsdp_min_rows=fsdp_min_rows,
            on_device_init=on_device_init,
        )
    raise ValueError(
        f"Unknown embedding_backend: {embedding_backend!r}. "
        f"Use 'cpu', 'tpu', or 'in_graph'."
    )
