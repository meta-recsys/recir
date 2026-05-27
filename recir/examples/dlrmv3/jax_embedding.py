#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""JAX SparseCore orchestrator for DLRMv3 embedding lookup.

Provides :class:`JAXEmbeddingDispatcher` and the
:func:`compile_with_jax_embedding` orchestrator that drive the v7.1
trace + compile_from_gm pipeline for the JAX/SparseCore embedding backend.

The dispatcher replaces :class:`recir.tests.xla.dlrmv3_test_utils.DlrmV3Preprocess`
with :class:`recir.tests.xla.dlrmv3_test_utils.DummyPreprocess` so tracing
sees stable shapes (with embedded marker pairs), then post-trace finds the
markers and installs a graph-break dispatch that runs
:class:`recir.examples.dlrmv3.embedding.JAXPreprocess` outside the XLA
compilation.  Backward dispatch is identity (the JAX SparseCore optimizer
is updated separately via the AdagradOptimizerSpec on the embedding spec).
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, Tuple

import recir
import torch
import torch.nn as nn
from recir.backends.base import Backend
from recir.distributed.parallel_mode import ParallelMode
from recir.eager_dispatch import install_graph_break_dispatch
from recir.region_transforms import RegionTransform
from recir.tests.xla.dlrmv3_test_utils import (
    _dlrmv3_marker_prefix,
    DlrmV3Preprocess,
    DummyPreprocess,
)


class JAXEmbeddingDispatcher:
    """Dispatcher that wires DLRMv3 preprocess to a JAX/SparseCore embedding.

    Builds the dummy preprocess up-front (via the caller-supplied
    ``dummy_factory``), swaps :class:`DlrmV3Preprocess` for the dummy in
    :meth:`install_eager`, and installs a graph-break dispatch into the
    :class:`recir.examples.dlrmv3.embedding.JAXPreprocess` callable in
    :meth:`install_graph`.
    """

    def __init__(
        self,
        dummy: DummyPreprocess,
        jax_ec: nn.Module,
        num_ranks: int,
    ) -> None:
        self._dummy: DummyPreprocess = dummy
        self._jax_ec: nn.Module = jax_ec
        self._num_ranks: int = num_ranks
        self._orig_preprocess: Optional[DlrmV3Preprocess] = None

    def install_eager(self, model: nn.Module) -> None:
        for fqn, mod in list(model.named_modules()):
            if isinstance(mod, DlrmV3Preprocess):
                self._orig_preprocess = mod
                parent_fqn, leaf = fqn.rsplit(".", 1) if "." in fqn else ("", fqn)
                parent = model.get_submodule(parent_fqn) if parent_fqn else model
                setattr(parent, leaf, self._dummy)

    def install_graph(
        self,
        gm: torch.fx.GraphModule,
        flat_state: List[torch.Tensor],
    ) -> Tuple[torch.fx.GraphModule, List[torch.Tensor]]:
        assert self._orig_preprocess is not None, (
            "install_eager() must run before install_graph()"
        )
        # Build the runtime JAXPreprocess callable on demand to avoid an
        # import cycle with recir.examples.dlrmv3.embedding (which imports
        # this module's siblings for orchestration).
        from recir.examples.dlrmv3.embedding import JAXPreprocess

        jax_preprocess = JAXPreprocess(self._orig_preprocess, self._jax_ec)
        install_graph_break_dispatch(
            gm,
            _dlrmv3_marker_prefix(self._dummy.override_id),
            jax_preprocess,
            num_state_tensors=len(flat_state),
            num_ranks=self._num_ranks,
        )
        return gm, flat_state


def compile_with_jax_embedding(
    model: nn.Module,
    callable_: Callable[..., Any],
    *args: Any,
    dispatcher: Any,
    parallel_mode: ParallelMode,
    backend: Backend,
    eager_preproc: Sequence[Callable[[nn.Module], None]] = (),
    region_transforms: Sequence[RegionTransform] = (),
    **kwargs: Any,
) -> Callable[..., Any]:
    """Compile ``callable_`` driven by a DLRMv3 graph-break dispatcher.

    Stage 0 — ``dispatcher.install_eager(model)`` + run user
    ``eager_preproc`` hooks.
    Stage 2 — :func:`recir.trace` produces ``(gm, flat_state)``.
    Stage 3 — ``dispatcher.install_graph(gm, flat_state)`` swaps marker pairs
    for a runtime graph-break dispatch.
    Stages 4-10 — :func:`recir.compile_from_gm` finishes the compile.

    Mirrors :func:`recir.compile` but inserts the embedding dispatch between
    trace and partition (per v7.1 Example 5).  Returns a callable with the
    same signature as ``callable_``.  Trailing ``*args`` / ``**kwargs`` are
    the fake-shape arguments forwarded to :func:`recir.trace` to drive the
    initial trace; they should match the runtime invocation signature.

    ``dispatcher`` may be any object exposing ``install_eager(model)`` and
    ``install_graph(gm, flat_state) -> (gm, flat_state)`` — typically the
    return value of
    :meth:`recir.examples.dlrmv3.embedding.EmbeddingBackend.get_dispatcher`.
    """
    from recir.api import make_compiled_runner

    # Stage 0
    dispatcher.install_eager(model)
    for hook in eager_preproc:
        hook(model)

    # Stage 2
    gm, flat_state = recir.trace(
        callable_,
        *args,
        parallel_mode=parallel_mode,
        region_transforms=region_transforms,
        **kwargs,
    )

    # Stage 3
    gm, flat_state = dispatcher.install_graph(gm, flat_state)

    # Stages 4-10
    compiled_obj: Any = recir.compile_from_gm(
        gm,
        flat_state,
        parallel_mode=parallel_mode,
        backend=backend,
        region_transforms=region_transforms,
    )

    return make_compiled_runner(compiled_obj)
