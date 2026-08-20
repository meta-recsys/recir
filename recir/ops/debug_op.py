# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Debug custom ops for the in-graph recir path.

``recir::debug_ids`` -- identity passthrough that dumps the block-owner
histogram of an id tensor at RUNTIME. CPU/eager execution is a plain identity;
the torchax lowering (``recir/torchax/ops/jdebug.py``) emits a
``jax.debug.print`` of ``owner = x // v_local`` counts over the mesh, from
INSIDE the compiled program -- so it works regardless of whether the frontend
tensor is a torchax ``Tensor`` or a plain torch tensor on an xla device (a plain
``print`` would only see a tracer / have no ``.jax()``).

Insert inline for perf-bisect id-distribution debugging:

    x = torch.ops.recir.debug_ids(x, "ids_cat", v_local)

``x`` is returned unchanged (int ids; no autograd), so it is safe to wrap any
tensor that later feeds the ring-shift.
"""

import torch

_LIB = torch.library.Library("recir", "FRAGMENT")

_LIB.define("debug_ids(Tensor x, str tag, int v_local) -> Tensor")


def _debug_ids_fallback(x: torch.Tensor, tag: str, v_local: int) -> torch.Tensor:
    return x


_LIB.impl("debug_ids", _debug_ids_fallback, "CompositeExplicitAutograd")


@torch.library.register_fake("recir::debug_ids")
def _debug_ids_meta(x: torch.Tensor, tag: str, v_local: int) -> torch.Tensor:
    return torch.empty_like(x)
