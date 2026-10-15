# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""RegionTransform — base class for region transforms."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Optional, Sequence

import torch
import torch.fx as fx
import torch.nn as nn


class RegionTransform:
    """Base class for a region transform.

    Subclasses override one or more optional hooks:

    * :meth:`prepare_after_shard` - mutate rank-local module or optimizer state
      after parameter sharding and before the inner/outer graph traces.
    * :meth:`transform_inputs` — Python pre-step that wraps the forward
      with input-side ops (e.g. padding).
    * :meth:`transform_graph` — In-place rewrite of the inner-traced
      ``gm_fwd`` (e.g. recompute, regather).  Implementing it triggers
      an inner ``make_fx`` trace with the already-transformed inputs.
    * :meth:`finalize_compiled_graph` — Post-compile pass on the final
      outer graph returned by the compile pipeline (after
      ``_finalize_graph``).  Override when the transform needs to inline
      fine-grained FX nodes that cannot be produced safely during the
      inner trace (e.g. decompositions involving ``Tensor.tolist()`` /
      ``_local_scalar_dense`` which crash under fake-mode dispatch).
    """

    def prepare_after_shard(
        self,
        module: nn.Module,
        optimizers: Sequence[torch.optim.Optimizer],
    ) -> None:
        """Prepare rank-local state after sharding and before graph tracing.

        Implementations may migrate optimizer ownership, but must not change
        the module call topology captured during region discovery.
        """
        return

    def transform_inputs(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Optional input pre-transform.  Default is a no-op."""
        return args, kwargs

    def select_forward_root_fqns(
        self,
        called_modules: OrderedDict[
            str, tuple[nn.Module, tuple[Any, ...], dict[str, Any]]
        ],
    ) -> Optional[set[str]]:
        """Optionally restrict which called module forwards are inner-traced.

        Returning ``None`` preserves the default shallowest-root selection.
        Returning FQNs selects those exact entries from ``called_modules``.
        """
        return None

    def transform_graph(
        self,
        gm: fx.GraphModule,
        called_modules: OrderedDict[
            str, tuple[nn.Module, tuple[Any, ...], dict[str, Any]]
        ],
    ) -> None:
        """Optional inner-graph rewrite.  Default is a no-op; the inner
        ``make_fx`` trace is skipped when no transform overrides it."""
        return

    def finalize_compiled_graph(self, gm: fx.GraphModule) -> None:
        """Optional post-compile hook on the final outer graph (e.g. tag
        XLA graph-break nodes, inline post-AOT decompositions).  Default
        is a no-op."""
        return
