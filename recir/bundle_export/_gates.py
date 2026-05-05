# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Export-time validation gates for the recir bundle producer.

These gates run BEFORE any artifact is written, so a failure leaves the
output directory clean rather than half-populated.  Each gate raises
:class:`recir.serde.SchemaViolationError` on mismatch -- subclassed from
:class:`RuntimeError` so callers can either catch it generically or import
the precise type.

Two gates ship in this module:

* :func:`_validate_kwargs_meta_consistency` -- for every node carrying both
  a ``device`` kwarg and a ``FakeTensor`` in ``meta['val']``, assert the
  two devices agree.  Catches accidental device-kwarg drift introduced by
  rewrite passes that mutated kwargs without resyncing meta.
* :func:`_validate_no_cpu_scalar_in_cuda_bundle` -- for every
  ``aten.scalar_tensor.default`` node, reject explicit ``device='cpu'``
  kwargs when the bundle target device is CUDA.  Bundles destined for a
  CUDA runner cannot ship with explicit CPU scalars baked into the graph;
  doing so causes silent device shuffles at runtime.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.fx as fx
from recir.serde import SchemaViolationError
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.node import _get_qualified_name


# ``aten.scalar_tensor.default`` is the canonical FX target for the
# scalar-tensor producer ops (``torch.scalar_tensor(x, ...)`` /
# ``torch.tensor(x)`` after decomposition).  The qualified-name string is
# computed via :func:`torch.fx.node._get_qualified_name` so the match is
# robust across torch builds and aligns with what
# :mod:`recir.serde.serializer` records on the wire.
_SCALAR_TENSOR_TARGETS: frozenset[str] = frozenset(
    {
        "torch.ops.aten.scalar_tensor.default",
    }
)


def _normalise_device(device: Any) -> torch.device | None:  # noqa: ANN401
    """Normalise a node-kwarg ``device`` value to a :class:`torch.device`.

    Accepts ``torch.device``, ``str``, or ``None``.  Returns ``None`` for
    ``None`` (meaning "let runtime decide"); otherwise a fully-qualified
    :class:`torch.device`.  Anything else is treated as ``None`` so the
    gate stays liberal in what it accepts.
    """
    if device is None:
        return None
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        try:
            return torch.device(device)
        except (RuntimeError, ValueError):
            return None
    return None


def _devices_compatible(a: torch.device, b: torch.device) -> bool:
    """Return True if devices ``a`` and ``b`` denote the same physical device.

    Two devices are compatible when their ``type`` agrees AND either both
    omit an index, both have the same index, or one omits the index (which
    is interpreted as "any index of this type" -- the standard torch
    convention).  This is the same rule
    :meth:`torch.device.__eq__` would apply if torch normalised
    ``cuda`` to ``cuda:0`` consistently, which it does not.
    """
    if a.type != b.type:
        return False
    if a.index is None or b.index is None:
        return True
    return a.index == b.index


def _node_target_str(node: fx.Node) -> str:
    """Return the qualified-name string of ``node.target`` for matching.

    For ``call_function`` nodes the target is a Python callable and
    :func:`torch.fx.node._get_qualified_name` produces a stable name like
    ``"torch.ops.aten.scalar_tensor.default"`` (matching what the recir
    serializer records on the wire).  For string targets (placeholder /
    output / get_attr) the value is returned verbatim.  Anything else
    falls back to ``str(target)`` so the gate can still match on best
    effort.
    """
    target = node.target
    if isinstance(target, str):
        return target
    try:
        return _get_qualified_name(target)
    except Exception:  # noqa: BLE001
        return str(target)


def _validate_kwargs_meta_consistency(gm: fx.GraphModule) -> None:
    """Reject FX graphs whose node ``device`` kwargs disagree with ``meta['val']``.

    For every node where BOTH:

    * ``node.kwargs['device']`` resolves to a concrete :class:`torch.device`, AND
    * ``node.meta['val']`` is a :class:`FakeTensor`,

    assert the two devices are compatible (per :func:`_devices_compatible`).
    If they disagree, raise :class:`SchemaViolationError`.

    Nodes that omit the ``device`` kwarg or whose ``meta['val']`` is not a
    FakeTensor are silently skipped -- the gate is concerned with active
    drift, not absent metadata.

    Args:
        gm: The :class:`fx.GraphModule` about to be exported.

    Raises:
        SchemaViolationError: On the first device-kwarg / meta drift.
    """
    for node in gm.graph.nodes:
        kw_device = _normalise_device(node.kwargs.get("device"))
        if kw_device is None:
            continue
        val = node.meta.get("val")
        if not isinstance(val, FakeTensor):
            continue
        meta_device = val.device
        if not _devices_compatible(kw_device, meta_device):
            raise SchemaViolationError(
                f"node {node.name!r} (target={_node_target_str(node)!r}) has "
                f"device kwarg {kw_device!s} but meta['val'].device is "
                f"{meta_device!s}; the two must agree before export. Use "
                f"recir.utils.update_kwargs to mutate device kwargs in a "
                f"meta-aware way."
            )


def _validate_no_cpu_scalar_in_cuda_bundle(
    gm: fx.GraphModule, target_device: str | torch.device
) -> None:
    """Reject CPU-pinned ``aten.scalar_tensor.default`` nodes in a CUDA bundle.

    Walks every node whose target matches one of
    :data:`_SCALAR_TENSOR_TARGETS`.  If the node carries an explicit
    ``device='cpu'`` kwarg AND the bundle's declared ``target_device`` is
    a CUDA device, raise :class:`SchemaViolationError`.

    A scalar_tensor without a ``device`` kwarg is allowed (the runtime
    fills the device in based on the consuming op).  A scalar_tensor on
    CUDA is allowed in a CUDA bundle.  Only the explicit-cpu-in-cuda
    combination is rejected.

    The check is a no-op when ``target_device`` is itself a CPU device --
    bundles bound for CPU runtime can carry CPU scalars without trouble.

    Args:
        gm: The :class:`fx.GraphModule` about to be exported.
        target_device: The bundle's declared target device class
            (e.g. ``"cuda"``, ``"cuda:0"``, ``"cpu"``).  String or
            :class:`torch.device`.

    Raises:
        SchemaViolationError: On the first CPU-in-CUDA scalar_tensor.
    """
    if isinstance(target_device, str):
        try:
            normalised_target = torch.device(target_device)
        except (RuntimeError, ValueError) as e:
            raise SchemaViolationError(
                f"Invalid target_device {target_device!r}: {e}"
            ) from e
    else:
        normalised_target = target_device
    if normalised_target.type != "cuda":
        return

    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        if _node_target_str(node) not in _SCALAR_TENSOR_TARGETS:
            continue
        kw_device = _normalise_device(node.kwargs.get("device"))
        if kw_device is None:
            continue
        if kw_device.type == "cpu":
            raise SchemaViolationError(
                f"node {node.name!r} (aten.scalar_tensor.default) declares "
                f"device='cpu' but the bundle target_device is "
                f"{normalised_target!s}; CPU scalars baked into a CUDA bundle "
                f"cause silent device shuffles at runtime. Either patch the "
                f"node's device kwarg before export (recir.utils.update_kwargs) "
                f"or strip the device kwarg entirely so the runtime fills it in."
            )
