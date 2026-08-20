# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Re-flip ``get_attr``-backed tensor constants on each :class:`IterGraphModule` sub-GM.

:meth:`recir.iter_graph_module.IterGraphModule.__init__` rebuilds three
sub-GraphModules (``setup_gm``, ``main_gm``, ``cleanup_gm``) by calling
``fx.GraphModule(src, graph)`` on each.  The :class:`fx.GraphModule`
constructor walks every ``get_attr`` target and runs
``torch.fx.graph_module._copy_attr``, which silently re-registers any
tensor attribute that is not an :class:`nn.Parameter` as a *buffer* on
the new sub-GM -- erasing the
:data:`~torch.export.unflatten._AttrKind.CONSTANT` registration that the
upstream FX deserializer set on the source GM.

That re-registration is wrong for the recir bundle path: the producer
declares those tensors as :data:`~_AttrKind.CONSTANT` so they survive
:meth:`fx.GraphModule.recompile` and downstream graph passes intact, and
so the runtime call vector excludes them by design.  This module
restores the producer's intent on every sub-GM after the
:class:`IterGraphModule` is built, mirroring the canonical pattern from
:func:`torch.export._unlift._create_stateful_graph_module`.
"""

from __future__ import annotations

from typing import Dict, Iterable

import torch
import torch.fx as fx
from torch.export.unflatten import _assign_attr, _AttrKind
from torch.fx.graph_module import _del_attr, _get_attr


def _collect_tensor_get_attr_fqns(gm: fx.GraphModule) -> list[str]:
    """Return the FQNs of every ``get_attr`` node that resolves to a tensor.

    Walks ``gm.graph.nodes`` once and de-duplicates while preserving
    first-encounter order so the reflip pass touches each FQN exactly
    once and in a deterministic order.  Targets that resolve to
    non-tensor objects (e.g. :class:`DeviceMesh`, process-group
    metadata) are skipped -- only tensor constants need the
    CONSTANT-kind reflip.
    """
    seen: set[str] = set()
    fqns: list[str] = []
    for node in gm.graph.nodes:
        if node.op != "get_attr":
            continue
        target = node.target
        if not isinstance(target, str):
            continue
        if target in seen:
            continue
        try:
            value = _get_attr(gm, target)
        except (AssertionError, AttributeError):
            # ``_get_attr`` can raise either type for a missing
            # intermediate prefix or leaf; in neither case is there
            # anything to reflip.
            continue
        if not isinstance(value, torch.Tensor):
            continue
        seen.add(target)
        fqns.append(target)
    return fqns


def _reflip_one(gm: fx.GraphModule, fqn: str, value: torch.Tensor) -> None:
    """Replace ``gm.<fqn>`` with a ``_AttrKind.CONSTANT`` registration.

    ``_copy_attr`` registered ``value`` as a buffer on ``gm`` during
    sub-GM reconstruction; this helper deletes the buffer and
    re-installs the value via :func:`_assign_attr` with
    :data:`~_AttrKind.CONSTANT`.  Mirrors the post-construct reflip in
    :func:`recir.serde.deserializer._reflip_to_constant`.

    The qualname-aware :func:`_del_attr` walks the dotted path down to
    the owning submodule and ``delattr``s the leaf, which removes the
    entry from :attr:`nn.Module._buffers` (see
    :meth:`nn.Module.__delattr__`).
    """
    try:
        _del_attr(gm, fqn)
    except AttributeError:
        # ``_copy_attr`` may have left no attribute behind (e.g. the
        # owning submodule was already shared with another sub-GM);
        # nothing to undo before the CONSTANT reinstallation.
        pass
    _assign_attr(value, gm, fqn, attr_kind=_AttrKind.CONSTANT)


def reflip_constants_on_sub_gms(
    sub_gms: Iterable[fx.GraphModule],
    constants: Dict[str, torch.Tensor],
) -> None:
    """Restore ``_AttrKind.CONSTANT`` on every sub-GM for every FQN in ``constants``.

    For each sub-GM the routine:

      1. discovers the tensor-typed ``get_attr`` FQNs the sub-GM
         actually references (via :func:`_collect_tensor_get_attr_fqns`);
      2. for every FQN that is also present in ``constants``, reflips
         the silently-registered buffer back to a
         :data:`~_AttrKind.CONSTANT` attribute holding the value from
         ``constants``.

    Restricting the reflip to FQNs that appear BOTH in the sub-GM's
    ``get_attr`` nodes AND in ``constants`` keeps the helper safe under
    graph rewrites that prune unused get_attrs (no extra attrs are
    introduced) and under callers that pass a wider ``constants`` dict
    than any single sub-GM consumes (no spurious AttributeError on FQNs
    a particular sub-GM never referenced).

    Args:
        sub_gms: The :class:`IterGraphModule` sub-GMs to patch
            (typically ``[setup_gm, main_gm, cleanup_gm]``).
        constants: FQN -> tensor map sourced from
            :attr:`recir.bundle_runtime.LoadedBundle.constants`.  The
            producer wrote this off the source GM, so the values are
            authoritative.
    """
    for sub_gm in sub_gms:
        for fqn in _collect_tensor_get_attr_fqns(sub_gm):
            if fqn not in constants:
                continue
            _reflip_one(sub_gm, fqn, constants[fqn])
