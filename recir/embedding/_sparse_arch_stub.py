# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Shared eager-swap machinery for the per-backend sparse-arch dispatchers.

The dispatchers themselves live with their backends -- ``cpu.SparseArchDispatcher``
(eager torchrec) and ``sparsecore.JAXSparseArchDispatcher`` (SparseCore). Only
their *graph* half differs; the *eager* half is identical and lives here:

  * :class:`DummySparseArch` -- the tracing stub they both swap in. It plants
    marker ops so tracing sees stable shapes and the graph half can rediscover
    the dispatch inputs/outputs.
  * :func:`sparse_arch_marker_prefix` -- the marker namespace both halves agree on.

The in-graph backend shares none of this: it keeps its lookup in-graph and
breaks at the input instead (``in_graph.SparseArchInputBreakDispatcher``).

Model-agnostic: callers name the model's torchrec ``EmbeddingCollection`` by its
FQN rather than this module importing a model-specific class.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
from recir.ops.eager_dispatch import mark_input, mark_output_with_bwd
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor

SPARSE_ARCH_MARKER_PREFIX = "recir.embedding.sparse_arch"


def sparse_arch_marker_prefix(override_id: int) -> str:
    return f"{SPARSE_ARCH_MARKER_PREFIX}::{override_id}"


class DummySparseArch(nn.Module):
    """Replacement for the real sparse arch that returns cached embeddings.

    Wraps input KJT components in :func:`recir.ops.eager_dispatch.mark_input` and
    outputs in :func:`mark_output_with_bwd` so the post-trace orchestrator can
    find the markers and wire runtime dispatch to the real lookup.

    A _grad_anchor parameter maintains autograd graph connectivity through
    the dummy so that make_fx traces backward correctly. A _batch_dep term
    adds data dependency on inputs so GSPMD can infer sharding.
    """

    def __init__(
        self,
        cached_output: Dict[str, JaggedTensor],
        input_kjt_keys: List[str],
        override_id: int = 1,
    ) -> None:
        super().__init__()
        self._override_id: int = override_id
        self._grad_anchor = nn.Parameter(torch.zeros(1))
        self._feature_names: List[str] = list(cached_output.keys())
        self._input_kjt_keys: List[str] = input_kjt_keys

        for i, (_name, jt) in enumerate(cached_output.items()):
            self.register_buffer(f"_jt_values_{i}", jt.values())
            self.register_buffer(f"_jt_lengths_{i}", jt.lengths())

    def forward(
        self,
        id_list_features: KeyedJaggedTensor,
    ) -> Dict[str, JaggedTensor]:
        oid = self._override_id
        prefix = sparse_arch_marker_prefix(oid)

        # Mark the input KJT components so SparseArchDispatcher.install_graph
        # can rediscover them via the marker prefix and use them as dispatch
        # inputs (instead of falling back to all batch placeholders).
        m_v = mark_input(id_list_features.values(), prefix, 0)
        m_l = mark_input(id_list_features.lengths(), prefix, 1)

        # Zero-valued anchor wired into every output's computation so FX
        # trace DCE keeps the m_v / m_l input markers alive end-to-end.
        # The deps=[m_v, m_l] hint to mark_output_with_bwd alone is not
        # enough; without a real torch-op dependency the markers get pruned
        # and install_graph_break_dispatch falls back to passing all
        # train_step placeholders to fwd_fn.
        anchor = (m_v.to(torch.int64).sum() * 0) + (m_l.to(torch.int64).sum() * 0)

        deps: List[torch.Tensor] = [m_v, m_l, self._grad_anchor]

        out_idx = 0
        result: Dict[str, JaggedTensor] = {}
        for i, name in enumerate(self._feature_names):
            vbuf = getattr(self, f"_jt_values_{i}")
            lbuf = getattr(self, f"_jt_lengths_{i}")
            v_z = torch.zeros_like(vbuf) + anchor.to(vbuf.dtype)
            l_z = lbuf.clone() + anchor.to(lbuf.dtype)
            v = mark_output_with_bwd(v_z, prefix, out_idx, deps=deps)
            lengths = mark_output_with_bwd(l_z, prefix, out_idx + 1, deps=deps)
            result[name] = JaggedTensor(values=v, lengths=lengths)
            out_idx += 2
        return result
