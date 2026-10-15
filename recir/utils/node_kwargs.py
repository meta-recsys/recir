# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Helpers for editing :class:`torch.fx.Node` kwargs in a meta-aware way.

The plain ``node.kwargs = {...}`` pattern is fine for raw arg patching but
silently leaves ``node.meta['val']`` (the FakeTensor seen by downstream
passes and serialization checks) out of sync. In particular, mutating a
``device`` kwarg without also re-faking the ``val`` can make an otherwise legal
edit fail metadata consistency validation.

:func:`update_kwargs` performs the kwargs merge AND, when the new kwargs
include a ``device`` entry, refreshes ``node.meta['val']`` to point at a
FakeTensor on the new device so the meta and the kwargs stay in lockstep.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.fx as fx
from torch._subclasses.fake_tensor import FakeTensor


def _refake_to_device(val: FakeTensor, device: torch.device) -> FakeTensor:
    """Return a FakeTensor with the same shape/dtype/stride as ``val`` on ``device``.

    The FakeTensorMode is borrowed from ``val`` itself so the resulting
    tensor is interoperable with whatever tracing context produced the
    original.  An empty FakeTensor is built via ``torch.empty_strided``
    inside that mode -- this matches what ``make_fx`` would have generated
    if the device kwarg had been correct on the first trace.

    ``requires_grad`` is propagated from ``val`` so downstream autograd
    passes that key off it (e.g. partitioner saved-tensor selection) keep
    the same behaviour after the refake.
    """
    fake_mode = val.fake_mode
    with fake_mode:
        result = torch.empty_strided(
            val.shape,
            val.stride(),
            dtype=val.dtype,
            device=device,
        )
        result.requires_grad_(val.requires_grad)
        return result  # pyre-ignore[7]


def update_kwargs(node: fx.Node, **new_kwargs: Any) -> None:  # noqa: ANN401
    """Merge ``new_kwargs`` into ``node.kwargs`` and resync ``meta['val']``.

    Behaviour:

    * The merge is shallow and last-writer-wins (``{**old, **new}``).
    * If ``new_kwargs`` carries a ``device`` entry AND the node already has
      a ``FakeTensor`` (or a list/tuple of ``FakeTensor``) in ``meta['val']``,
      each FakeTensor is rebuilt on the new device so the meta agrees with
      the kwargs.  Other meta entries are left untouched.
    * If ``meta['val']`` is absent or holds a non-FakeTensor object, the
      kwargs merge still happens but no meta refresh is attempted -- the
      caller is responsible for ensuring the node remains consistent.

    Args:
        node: The :class:`fx.Node` to mutate in place.
        **new_kwargs: New kwargs to merge on top of ``node.kwargs``.
    """
    merged = {**node.kwargs, **new_kwargs}
    node.kwargs = merged

    if "device" not in new_kwargs:
        return

    raw_device = new_kwargs["device"]
    if raw_device is None:
        return
    target_device = (
        raw_device if isinstance(raw_device, torch.device) else torch.device(raw_device)
    )

    val = node.meta.get("val")
    if isinstance(val, FakeTensor):
        node.meta["val"] = _refake_to_device(val, target_device)
        return
    if (
        isinstance(val, (list, tuple))
        and val
        and all(isinstance(v, FakeTensor) for v in val)
    ):
        refreshed = [_refake_to_device(v, target_device) for v in val]
        node.meta["val"] = type(val)(refreshed)
