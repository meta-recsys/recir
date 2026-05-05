# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Encode an FX :class:`torch.fx.GraphModule` to a JSON-friendly dict.

Public entry point: :func:`serialize_fx_graph`.

The output dict is structured according to :mod:`recir.serde.schema` and
suitable for ``json.dumps``.  The schema is *stateful-aware*:

* ``get_attr`` nodes are accepted; their FQNs are collected into
  ``required_constants`` so the deserializer (and downstream callers) can
  match them against the ``constants`` dict supplied at decode time.  Tensor
  values themselves are NOT inlined in the JSON payload -- they ship through
  a separate channel (e.g. safetensors).
* ``call_module`` and ``call_method`` nodes remain rejected: the
  deserializer constructs the GraphModule under a freshly-built root with
  no submodules and no live Python instances, so it cannot resolve those op
  kinds.
"""

from __future__ import annotations

import dataclasses
import operator
from typing import Any

import torch
import torch.fx as fx
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.node import _get_qualified_name

from ._codec import _StorageRegistry, encode_device, encode_dtype, encode_fake_tensor
from .schema import (
    Argument,
    BindingTable,
    FakeTensorSpec,
    SCHEMA_VERSION,
    SchemaViolationError,
    SerializedGraph,
    SerializedNode,
)


# Op kinds the deserializer cannot reconstruct.  ``get_attr`` is now
# accepted (handled via the ``constants`` argument at decode time);
# ``call_module`` and ``call_method`` would need a live Python instance to
# dispatch against and have no analogue in the empty-root construction
# strategy, so they remain rejected.
_UNSUPPORTED_NODE_OPS: frozenset[str] = frozenset({"call_module", "call_method"})


# ``operator.getitem`` is the only operator-module callable that survives FX
# tracing in the graphs we produce; encode it under the ``_operator``
# namespace to match torch._export's serialiser convention.
_OPERATOR_GETITEM_FQN: str = "_operator.getitem"


def _encode_target(op: str, target: Any) -> str:  # noqa: ANN401
    """Encode an FX node's target as a stable string."""
    if op in ("placeholder", "output", "get_attr"):
        if not isinstance(target, str):
            raise TypeError(f"Expected str target for op={op!r}, got {type(target)!r}")
        return target
    if target is operator.getitem:
        return _OPERATOR_GETITEM_FQN
    return _get_qualified_name(target)


def _encode_primitive(value: object) -> Argument | None:
    """Encode primitive scalars (``None``/``bool``/``int``/``float``/``str``)."""
    if value is None:
        return Argument(kind="none", value=None)
    # ``bool`` must be tested before ``int`` because ``bool`` is a subtype of
    # ``int`` in Python.
    if isinstance(value, bool):
        return Argument(kind="bool", value=value)
    if isinstance(value, int):
        return Argument(kind="int", value=value)
    if isinstance(value, float):
        return Argument(kind="float", value=value)
    if isinstance(value, str):
        return Argument(kind="str", value=value)
    return None


def _encode_torch_enum(value: object) -> Argument | None:
    """Encode ``torch.dtype``/``device``/``layout``/``memory_format``."""
    if isinstance(value, torch.dtype):
        return Argument(kind="dtype", value=encode_dtype(value))
    if isinstance(value, torch.device):
        return Argument(kind="device", value=encode_device(value))
    if isinstance(value, torch.layout):
        return Argument(kind="layout", value=str(value).removeprefix("torch."))
    if isinstance(value, torch.memory_format):
        return Argument(kind="memory_format", value=str(value).removeprefix("torch."))
    return None


def _encode_sequence(
    value: list[object] | tuple[object, ...], registry: _StorageRegistry
) -> Argument:
    kind = "list" if isinstance(value, list) else "tuple"
    return Argument(kind=kind, value=[_encode_argument(v, registry) for v in value])


def _encode_dict(value: dict[object, object], registry: _StorageRegistry) -> Argument:
    encoded: dict[str, Argument] = {}
    for k, v in value.items():
        if not isinstance(k, str):
            raise TypeError(f"Dict argument keys must be str, got {type(k)!r}")
        encoded[k] = _encode_argument(v, registry)
    return Argument(kind="dict", value=encoded)


def _encode_argument(value: object, registry: _StorageRegistry) -> Argument:
    """Recursively encode a Python value as an :class:`Argument`."""
    primitive = _encode_primitive(value)
    if primitive is not None:
        return primitive
    enum_arg = _encode_torch_enum(value)
    if enum_arg is not None:
        return enum_arg
    if isinstance(value, fx.Node):
        return Argument(kind="node", value=value.name)
    if isinstance(value, (list, tuple)):
        return _encode_sequence(value, registry)
    if isinstance(value, dict):
        return _encode_dict(value, registry)
    raise TypeError(f"Unsupported argument type for serialization: {type(value)!r}")


def _encode_val(
    val: object, registry: _StorageRegistry
) -> FakeTensorSpec | list[FakeTensorSpec] | None:
    """Encode the ``meta['val']`` payload attached to a node, if any."""
    if val is None:
        return None
    if isinstance(val, FakeTensor):
        return encode_fake_tensor(val, registry)
    if isinstance(val, (list, tuple)):
        encoded: list[FakeTensorSpec] = []
        for v in val:
            if not isinstance(v, FakeTensor):
                raise TypeError(
                    f"Unsupported val element type {type(v)!r} in tuple/list val"
                )
            encoded.append(encode_fake_tensor(v, registry))
        return encoded
    # Other val types (SymInt, plain ints from getitem on shapes, ...) are
    # discarded; they're recoverable from the args of the producing node.
    return None


def _argument_to_jsonable(arg: Argument) -> dict[str, object]:
    """Convert an :class:`Argument` into a plain JSON-serialisable dict."""
    if arg.kind == "list":
        return {
            "kind": "list",
            "value": [_argument_to_jsonable(a) for a in arg.value],
        }
    if arg.kind == "tuple":
        return {
            "kind": "tuple",
            "value": [_argument_to_jsonable(a) for a in arg.value],
        }
    if arg.kind == "dict":
        return {
            "kind": "dict",
            "value": {k: _argument_to_jsonable(v) for k, v in arg.value.items()},
        }
    return {"kind": arg.kind, "value": arg.value}


def _node_to_jsonable(n: SerializedNode) -> dict[str, object]:
    out: dict[str, object] = {
        "name": n.name,
        "op": n.op,
        "target": n.target,
        "args": [_argument_to_jsonable(a) for a in n.args],
        "kwargs": {k: _argument_to_jsonable(v) for k, v in n.kwargs.items()},
    }
    if n.val is not None:
        if isinstance(n.val, list):
            out["val"] = {
                "kind": "list",
                "value": [dataclasses.asdict(v) for v in n.val],
            }
        else:
            out["val"] = dataclasses.asdict(n.val)
    return out


def _bindings_to_jsonable(bt: BindingTable) -> dict[str, object]:
    return {
        "bindings": [
            {"kind": b.kind, "fqn": b.fqn, "state_key": b.state_key}
            for b in bt.bindings
        ],
    }


def _graph_to_jsonable(g: SerializedGraph) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": list(g.schema_version),
        "nodes": [_node_to_jsonable(n) for n in g.nodes],
        "storage_table": {
            sid: dataclasses.asdict(e) for sid, e in g.storage_table.items()
        },
        "in_spec": g.in_spec,
        "out_spec": g.out_spec,
        "orig_args": g.orig_args,
        "required_constants": list(g.required_constants),
    }
    if g.bindings is not None:
        payload["bindings"] = _bindings_to_jsonable(g.bindings)
    return payload


def serialize_fx_graph(
    gm: fx.GraphModule, bindings: BindingTable | None = None
) -> dict[str, object]:
    """Serialize an :class:`fx.GraphModule` to a JSON-friendly dict.

    The output dict can be passed directly to :func:`json.dumps`.  Only the
    *graph* of the module is serialised.  Tensor attributes referenced via
    ``get_attr`` nodes are NOT inlined; their FQNs are collected in
    ``required_constants`` and the deserializer expects the caller to supply
    the values out-of-band via its ``constants`` argument.

    Args:
        gm: The :class:`fx.GraphModule` to serialise.
        bindings: Optional :class:`BindingTable` describing the placeholder
            ordering and any constant FQNs.  Carried verbatim through the
            wire format; the deserializer surfaces it via the
            :class:`SerializedGraph` payload but does not interpret it.

    Returns:
        A JSON-serialisable ``dict`` containing the encoded graph.
    """
    if not isinstance(gm, fx.GraphModule):
        raise TypeError(f"serialize_fx_graph expects fx.GraphModule, got {type(gm)!r}")

    # Reject unsupported op kinds BEFORE any encoding work so the caller
    # never gets a payload that would fail at decode time.
    for node in gm.graph.nodes:
        if node.op in _UNSUPPORTED_NODE_OPS:
            raise SchemaViolationError(
                f"FX op kind {node.op!r} is not supported by serialize_fx_graph "
                f"(node target: {node.target!r}). The deserializer constructs "
                f"the GraphModule under a freshly-built root with no submodules "
                f"and no live Python instances, so it cannot resolve "
                f"call_module / call_method references."
            )

    registry = _StorageRegistry()
    nodes: list[SerializedNode] = []
    required_constants: list[str] = []
    seen_constants: set[str] = set()
    for node in gm.graph.nodes:
        snode = SerializedNode(
            name=node.name,
            op=node.op,
            target=_encode_target(node.op, node.target),
            args=[_encode_argument(a, registry) for a in node.args],
            kwargs={k: _encode_argument(v, registry) for k, v in node.kwargs.items()},
            val=_encode_val(node.meta.get("val"), registry),
        )
        nodes.append(snode)
        if node.op == "get_attr":
            fqn = snode.target
            if fqn not in seen_constants:
                required_constants.append(fqn)
                seen_constants.add(fqn)

    in_spec_str: str | None = None
    out_spec_str: str | None = None
    orig_args: list[str] | None = None
    codegen = gm.graph._codegen
    pytree_info = getattr(codegen, "pytree_info", None)
    if pytree_info is not None:
        from torch.utils._pytree import treespec_dumps

        if pytree_info.in_spec is not None:
            in_spec_str = treespec_dumps(pytree_info.in_spec)
        if pytree_info.out_spec is not None:
            out_spec_str = treespec_dumps(pytree_info.out_spec)
        # Guard ``orig_args`` -- recir's ``_to_caller_flattened_graph_module``
        # constructs the codegen with ``orig_args=None``; the deserializer
        # falls back to placeholder names in that case.
        if pytree_info.orig_args is not None:
            orig_args = list(pytree_info.orig_args)

    serialized = SerializedGraph(
        schema_version=SCHEMA_VERSION,
        nodes=nodes,
        storage_table=registry.entries(),
        in_spec=in_spec_str,
        out_spec=out_spec_str,
        orig_args=orig_args,
        required_constants=required_constants,
        bindings=bindings,
    )
    return _graph_to_jsonable(serialized)
