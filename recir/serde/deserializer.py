# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Decode an FX :class:`torch.fx.GraphModule` from JSON.

Public entry point: :func:`deserialize_fx_graph`.

The deserializer is the inverse of :func:`recir.serde.serializer.serialize_fx_graph`,
plus one stateful-aware capability: ``get_attr`` nodes are resolved against
a ``constants`` dict supplied by the caller.

Construction proceeds in two phases:

* **Phase A** -- parse the JSON payload into the in-memory schema dataclasses,
  build a fresh :class:`fx.Graph`, and extract the FQNs of every
  ``get_attr`` target referenced by the graph.
* **Phase B** -- build a freshly-constructed :class:`torch.nn.Module` root,
  seed every required constant on it via
  :func:`torch.export.unflatten._assign_attr` with
  :data:`~torch.export.unflatten._AttrKind.CONSTANT` BEFORE handing
  ``(root, graph)`` to :class:`fx.GraphModule`, then post-construct walk the
  same FQN list and re-flip each entry from buffer-back-to-CONSTANT.

The seed-before-construct dance is required because :class:`fx.GraphModule`
calls :func:`torch.fx.graph_module._copy_attr` for every ``get_attr`` target,
and ``_copy_attr`` silently registers any tensor attribute that isn't an
:class:`nn.Parameter` as a *buffer*.  Reflipping after construction matches
the canonical pattern used by
:func:`torch.export._unlift._create_stateful_graph_module` (see
``caffe2/torch/export/_unlift.py:528-554``).
"""

from __future__ import annotations

import importlib
import json
import operator
from pathlib import Path
from typing import Any

import torch
import torch.fx as fx
from torch._subclasses.fake_tensor import FakeTensor
from torch.export.unflatten import _assign_attr, _AttrKind
from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo
from torch.fx.graph_module import _del_attr

from ._codec import decode_device, decode_dtype, FakeTensorDecoder
from .schema import (
    Argument,
    Binding,
    BindingTable,
    FakeTensorSpec,
    SCHEMA_VERSION,
    SchemaViolationError,
    SerializedNode,
    StorageEntry,
)


# Targets that are encoded literally rather than as ``torch.<...>``.
_LITERAL_TARGETS: dict[str, Any] = {
    "_operator.getitem": operator.getitem,
}


# ---------------------------------------------------------------------------
# JSON -> dataclass parsing
# ---------------------------------------------------------------------------


def _parse_argument(payload: dict[str, object]) -> Argument:
    kind = payload.get("kind")
    if not isinstance(kind, str):
        raise SchemaViolationError(f"Argument missing string 'kind' field: {payload!r}")
    raw = payload.get("value")
    if kind == "list":
        if not isinstance(raw, list):
            raise SchemaViolationError("list argument must carry a list value")
        return Argument(kind="list", value=[_parse_argument(item) for item in raw])
    if kind == "tuple":
        if not isinstance(raw, list):
            raise SchemaViolationError("tuple argument must carry a list value")
        return Argument(kind="tuple", value=[_parse_argument(item) for item in raw])
    if kind == "dict":
        if not isinstance(raw, dict):
            raise SchemaViolationError("dict argument must carry a dict value")
        return Argument(
            kind="dict",
            value={
                k: _parse_argument(v)  # pyre-ignore[6]
                for k, v in raw.items()
                if isinstance(k, str)
            },
        )
    return Argument(kind=kind, value=raw)


def _parse_fake_tensor_spec(payload: dict[str, object]) -> FakeTensorSpec:
    try:
        return FakeTensorSpec(
            kind=str(payload["kind"]),
            storage_id=str(payload["storage_id"]),
            storage_offset=int(payload["storage_offset"]),  # pyre-ignore[6]
            shape=[int(s) for s in payload["shape"]],  # pyre-ignore[16]
            stride=[int(s) for s in payload["stride"]],  # pyre-ignore[16]
            dtype=str(payload["dtype"]),
            fake_device=str(payload["fake_device"]),
            requires_grad=bool(payload["requires_grad"]),
        )
    except (KeyError, TypeError, ValueError) as e:
        raise SchemaViolationError(f"Invalid FakeTensor spec: {e}") from e


def _parse_node(payload: dict[str, object]) -> SerializedNode:
    try:
        name = str(payload["name"])
        op = str(payload["op"])
        target = str(payload["target"])
    except KeyError as e:
        raise SchemaViolationError(f"Node missing required field: {e}") from e
    raw_args = payload.get("args", [])
    if not isinstance(raw_args, list):
        raise SchemaViolationError("Node 'args' must be a list")
    raw_kwargs = payload.get("kwargs", {})
    if not isinstance(raw_kwargs, dict):
        raise SchemaViolationError("Node 'kwargs' must be a dict")
    args = [_parse_argument(a) for a in raw_args]  # pyre-ignore[6]
    kwargs = {
        k: _parse_argument(v)  # pyre-ignore[6]
        for k, v in raw_kwargs.items()
        if isinstance(k, str)
    }
    val_payload = payload.get("val")
    val: FakeTensorSpec | list[FakeTensorSpec] | None = None
    if isinstance(val_payload, dict):
        if val_payload.get("kind") == "list":
            raw_list = val_payload.get("value", [])
            if not isinstance(raw_list, list):
                raise SchemaViolationError("list val must carry a list value")
            val = [_parse_fake_tensor_spec(item) for item in raw_list]  # pyre-ignore[6]
        else:
            val = _parse_fake_tensor_spec(val_payload)
    return SerializedNode(
        name=name, op=op, target=target, args=args, kwargs=kwargs, val=val
    )


def _parse_storage_table(payload: dict[str, object]) -> dict[str, StorageEntry]:
    table: dict[str, StorageEntry] = {}
    for sid, entry in payload.items():
        if not isinstance(entry, dict):
            raise SchemaViolationError(f"storage table entry {sid!r} is not a dict")
        try:
            table[sid] = StorageEntry(
                storage_id=str(entry["storage_id"]),
                nbytes=int(entry["nbytes"]),  # pyre-ignore[6]
                fake_device=str(entry["fake_device"]),
                base_dtype=str(entry["base_dtype"]),
            )
        except (KeyError, TypeError, ValueError) as e:
            raise SchemaViolationError(f"Invalid storage entry {sid!r}: {e}") from e
    return table


def parse_bindings(payload: object) -> BindingTable | None:
    """Parse a serialised :class:`BindingTable` payload back to a dataclass.

    Surfaced so consumers (e.g. the bundle puller) can inspect placeholder
    ordering metadata WITHOUT going through :func:`deserialize_fx_graph` --
    useful when the wire payload is split from the tensor sidecar.
    """
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise SchemaViolationError("'bindings' must be a dict or null")
    raw = payload.get("bindings", [])
    if not isinstance(raw, list):
        raise SchemaViolationError("bindings.bindings must be a list")
    bindings: list[Binding] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise SchemaViolationError("binding entry must be a dict")
        try:
            bindings.append(
                Binding(
                    kind=str(entry["kind"]),
                    fqn=str(entry["fqn"]),
                    state_key=(
                        str(entry["state_key"])
                        if entry.get("state_key") is not None
                        else None
                    ),
                )
            )
        except KeyError as e:
            raise SchemaViolationError(f"binding entry missing field: {e}") from e
    return BindingTable(bindings=bindings)


# ---------------------------------------------------------------------------
# Target resolution
# ---------------------------------------------------------------------------


def _resolve_target(target_fqn: str) -> Any:  # noqa: ANN401
    """Resolve a qualified-name string back to the Python callable."""
    if target_fqn in _LITERAL_TARGETS:
        return _LITERAL_TARGETS[target_fqn]
    parts = target_fqn.split(".")
    module = None
    module_idx = 0
    # Find the longest leading prefix that is an importable module, then
    # walk the remaining components via ``getattr``.
    for i in range(len(parts), 0, -1):
        candidate = ".".join(parts[:i])
        try:
            module = importlib.import_module(candidate)
            module_idx = i
            break
        except ImportError:
            continue
    if module is None:
        raise SchemaViolationError(
            f"Cannot resolve target {target_fqn!r}: no importable prefix"
        )
    obj: Any = module
    for component in parts[module_idx:]:
        try:
            obj = getattr(obj, component)
        except AttributeError as e:
            raise SchemaViolationError(
                f"Cannot resolve target {target_fqn!r}: missing attribute {component!r}"
            ) from e
    return obj


# ---------------------------------------------------------------------------
# Argument materialisation
# ---------------------------------------------------------------------------


_LAYOUT_LOOKUP: dict[str, torch.layout] = {
    "strided": torch.strided,
    "sparse_coo": torch.sparse_coo,
    "sparse_csr": torch.sparse_csr,
    "sparse_csc": torch.sparse_csc,
    "sparse_bsr": torch.sparse_bsr,
    "sparse_bsc": torch.sparse_bsc,
}

_MEMORY_FORMAT_LOOKUP: dict[str, torch.memory_format] = {
    "contiguous_format": torch.contiguous_format,
    "channels_last": torch.channels_last,
    "channels_last_3d": torch.channels_last_3d,
    "preserve_format": torch.preserve_format,
}


def _materialise_primitive(arg: Argument) -> object:
    if arg.kind == "none":
        return None
    return arg.value


def _materialise_torch_enum(arg: Argument) -> object:
    kind = arg.kind
    value = arg.value
    if kind == "dtype":
        return decode_dtype(str(value))
    if kind == "device":
        return decode_device(str(value))
    if kind == "layout":
        if value not in _LAYOUT_LOOKUP:
            raise SchemaViolationError(f"Unknown layout: {value!r}")
        return _LAYOUT_LOOKUP[str(value)]
    if kind == "memory_format":
        if value not in _MEMORY_FORMAT_LOOKUP:
            raise SchemaViolationError(f"Unknown memory_format: {value!r}")
        return _MEMORY_FORMAT_LOOKUP[str(value)]
    raise SchemaViolationError(f"Unknown torch enum kind: {kind!r}")


def _materialise_node_ref(arg: Argument, node_lookup: dict[str, fx.Node]) -> fx.Node:
    name = str(arg.value)
    if name not in node_lookup:
        raise SchemaViolationError(f"Argument references unknown node {name!r}")
    return node_lookup[name]


def _materialise_sequence(arg: Argument, node_lookup: dict[str, fx.Node]) -> object:
    if arg.kind == "list":
        return [_materialise_argument(a, node_lookup) for a in arg.value]
    return tuple(_materialise_argument(a, node_lookup) for a in arg.value)


def _materialise_dict(
    arg: Argument, node_lookup: dict[str, fx.Node]
) -> dict[str, object]:
    return {k: _materialise_argument(v, node_lookup) for k, v in arg.value.items()}


_PRIMITIVE_KINDS: frozenset[str] = frozenset({"none", "bool", "int", "float", "str"})
_TORCH_ENUM_KINDS: frozenset[str] = frozenset(
    {"dtype", "device", "layout", "memory_format"}
)
_SEQUENCE_KINDS: frozenset[str] = frozenset({"list", "tuple"})


def _materialise_argument(arg: Argument, node_lookup: dict[str, fx.Node]) -> object:
    kind = arg.kind
    if kind in _PRIMITIVE_KINDS:
        return _materialise_primitive(arg)
    if kind in _TORCH_ENUM_KINDS:
        return _materialise_torch_enum(arg)
    if kind == "node":
        return _materialise_node_ref(arg, node_lookup)
    if kind in _SEQUENCE_KINDS:
        return _materialise_sequence(arg, node_lookup)
    if kind == "dict":
        return _materialise_dict(arg, node_lookup)
    raise SchemaViolationError(f"Unknown argument kind: {kind!r}")


# ---------------------------------------------------------------------------
# Phase A -- graph construction (pure FX, no root attrs yet)
# ---------------------------------------------------------------------------


def _build_graph(
    nodes: list[SerializedNode],
    storage_table: dict[str, StorageEntry],
    in_spec_str: str | None,
    out_spec_str: str | None,
    orig_args: list[str] | None,
) -> tuple[fx.Graph, list[str]]:
    """Build the :class:`fx.Graph` and return the unique ``get_attr`` FQNs.

    The order of FQNs in the returned list matches first-encounter order in
    the node iteration.  Callers use this list to seed the matching constants
    on the root *before* :class:`fx.GraphModule` is constructed.
    """
    graph = fx.Graph()
    node_lookup: dict[str, fx.Node] = {}
    fake_decoder = FakeTensorDecoder()

    placeholder_names: list[str] = []
    get_attr_fqns: list[str] = []
    seen_get_attr: set[str] = set()
    for snode in nodes:
        if snode.op in ("placeholder", "output", "get_attr"):
            resolved_target: Any = snode.target
        else:
            resolved_target = _resolve_target(snode.target)

        args = tuple(_materialise_argument(a, node_lookup) for a in snode.args)
        kwargs = {
            k: _materialise_argument(v, node_lookup) for k, v in snode.kwargs.items()
        }

        new_node = graph.create_node(
            op=snode.op,
            target=resolved_target,  # pyre-ignore[6]
            args=args,  # pyre-ignore[6]
            kwargs=kwargs,  # pyre-ignore[6]
            name=snode.name,
        )
        if snode.val is not None:
            if isinstance(snode.val, list):
                val_obj: FakeTensor | list[FakeTensor] = [
                    fake_decoder.decode(spec, storage_table) for spec in snode.val
                ]
            else:
                val_obj = fake_decoder.decode(snode.val, storage_table)
            new_node.meta["val"] = val_obj
        node_lookup[snode.name] = new_node
        if snode.op == "placeholder":
            placeholder_names.append(snode.name)
        elif snode.op == "get_attr" and snode.target not in seen_get_attr:
            get_attr_fqns.append(snode.target)
            seen_get_attr.add(snode.target)

    if in_spec_str is not None:
        from torch.utils._pytree import treespec_loads

        in_spec = treespec_loads(in_spec_str)
        out_spec = treespec_loads(out_spec_str) if out_spec_str is not None else None
        effective_orig_args = (
            list(orig_args) if orig_args is not None else placeholder_names
        )
        graph._codegen = _PyTreeCodeGen(
            _PyTreeInfo(
                orig_args=effective_orig_args,
                in_spec=in_spec,
                out_spec=out_spec,
            )
        )
    return graph, get_attr_fqns


# ---------------------------------------------------------------------------
# Phase B -- seed constants + construct GraphModule + reflip
# ---------------------------------------------------------------------------


def _reflip_to_constant(gm: fx.GraphModule, fqn: str, value: torch.Tensor) -> None:
    """Replace whatever attr at ``fqn`` with a CONSTANT-kind registration.

    ``fx.GraphModule.__init__`` runs ``_copy_attr`` for every ``get_attr``
    target; ``_copy_attr`` silently calls ``register_buffer`` if the value
    is a tensor that isn't an ``nn.Parameter``.  This function deletes the
    silently-registered buffer and re-installs the value via
    :func:`_assign_attr` with :data:`~torch.export.unflatten._AttrKind.CONSTANT`.

    The qualname-aware ``_del_attr`` helper handles nested FQNs by walking
    the dotted path down to the owning submodule and ``delattr``ing the
    leaf field, which removes the entry from
    :attr:`~torch.nn.Module._buffers` (see :meth:`nn.Module.__delattr__`).

    .. note::
        We deliberately AVOID ``_has_attr`` from ``torch.fx.graph_module``:
        its current implementation conflates ``hasattr`` returns with the
        recursion variable (line 336 of ``graph_module.py``) and reports
        ``False`` for any nested attribute, which would silently skip the
        ``_del_attr`` call and leave the buffer in place.  Going through
        ``_del_attr`` unconditionally and catching ``AttributeError`` is
        both correct and resilient to that helper's bug.
    """
    try:
        _del_attr(gm, fqn)
    except AttributeError:
        # Unlikely path: ``_copy_attr`` left no attribute behind (e.g.
        # because a parent submodule was already shared).  Nothing to undo
        # before re-installing the CONSTANT.
        pass
    _assign_attr(value, gm, fqn, attr_kind=_AttrKind.CONSTANT)


def _seed_and_construct(
    graph: fx.Graph,
    get_attr_fqns: list[str],
    constants: dict[str, torch.Tensor] | None,
) -> fx.GraphModule:
    """Seed required constants on a fresh root, then construct the GraphModule.

    Implements the canonical seed-before-construct dance also used by
    :func:`torch.export._unlift._create_stateful_graph_module`:

    1. Build an empty ``torch.nn.Module`` root.
    2. For every ``get_attr`` FQN, install the value via
       :func:`_assign_attr` with ``attr_kind=_AttrKind.CONSTANT``.
    3. Construct ``fx.GraphModule(root, graph)``.  Internally this calls
       ``_copy_attr`` per target, which silently turns each tensor
       attribute into a *buffer* on the new GraphModule.
    4. For every FQN, ``_del_attr`` the silently-registered buffer and
       ``_assign_attr(..., CONSTANT)`` to flip it back.

    If ``constants`` is ``None`` and the graph references no ``get_attr``
    targets, the root remains empty -- this preserves back-compat for
    graphs that genuinely don't carry constants.
    """
    if get_attr_fqns and constants is None:
        raise SchemaViolationError(
            f"Graph references get_attr FQNs {sorted(get_attr_fqns)} but no "
            f"`constants` dict was supplied. Pass `constants={{fqn: tensor, ...}}` "
            f"covering every required FQN."
        )

    constants = constants or {}
    missing = [fqn for fqn in get_attr_fqns if fqn not in constants]
    if missing:
        raise SchemaViolationError(
            f"`constants` is missing required FQN(s): {missing}. "
            f"Required: {sorted(get_attr_fqns)}; supplied: {sorted(constants)}"
        )

    root = torch.nn.Module()
    # Phase 2: seed constants on the empty root BEFORE GraphModule
    # construction.  This ensures the root has the attribute available when
    # ``_copy_attr`` walks the get_attr nodes inside ``GraphModule.__init__``.
    for fqn in get_attr_fqns:
        _assign_attr(constants[fqn], root, fqn, attr_kind=_AttrKind.CONSTANT)

    # Phase 3: construct the GraphModule.  This invokes ``_copy_attr`` for
    # every ``get_attr`` target which silently registers each tensor as a
    # buffer on the GraphModule (NOT as a CONSTANT-kind attribute).
    gm = fx.GraphModule(root, graph)

    # Phase 4: reflip every silently-registered buffer back to CONSTANT.
    # Mirrors ``_create_stateful_graph_module`` in
    # ``caffe2/torch/export/_unlift.py:528-554``.
    for fqn in get_attr_fqns:
        _reflip_to_constant(gm, fqn, constants[fqn])

    return gm


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _load_payload(json_data: dict[str, object] | str | Path) -> dict[str, object]:
    if isinstance(json_data, dict):
        return json_data
    # ``Path`` check must precede ``str`` because ``Path`` is not a ``str``
    # subclass but is the only accepted way to pass a filesystem path.  A
    # bare ``str`` is ALWAYS parsed as JSON content -- never as a filesystem
    # path -- to avoid path-traversal / file-disclosure when the string
    # source is partly attacker-controlled.
    if isinstance(json_data, Path):
        result = json.loads(json_data.read_text())
        if not isinstance(result, dict):
            raise ValueError(f"Expected JSON object, got {type(result).__name__}")
        return result
    if isinstance(json_data, str):
        stripped = json_data.lstrip()
        if not stripped.startswith("{"):
            raise ValueError(
                f"Expected JSON object string starting with '{{', got: "
                f"{json_data[:50]}..."
            )
        return json.loads(json_data)
    raise TypeError(
        f"deserialize_fx_graph: unsupported json_data type {type(json_data)!r}"
    )


def deserialize_fx_graph(
    json_data: dict[str, object] | str | Path,
    constants: dict[str, torch.Tensor] | None = None,
) -> fx.GraphModule:
    """Decode a serialized graph back into an :class:`fx.GraphModule`.

    Args:
        json_data: One of:

            * a ``dict`` -- already-parsed payload;
            * a ``str`` -- raw JSON content; MUST start with ``{``.  ``str``
              is NEVER treated as a filesystem path, to avoid path-traversal
              when the caller's string is partly attacker-controlled.
            * a :class:`pathlib.Path` -- filesystem path to a JSON file.
        constants: Optional mapping from ``get_attr`` FQN to tensor value.
            Required when the encoded graph references ``get_attr`` nodes.
            Each value is registered on the GraphModule's root via
            :data:`torch.export.unflatten._AttrKind.CONSTANT` so it survives
            ``gm.recompile()`` and downstream graph passes.  Pass ``None``
            (or omit) when the graph has no ``get_attr`` nodes.

    Returns:
        A fresh :class:`fx.GraphModule` whose graph mirrors the serialised
        payload.  Tensor constants are accessible as attributes on the
        module via the encoded FQN; placeholder-bound state (parameters,
        buffers, optimiser state, user inputs) is the caller's
        responsibility -- it ships through a separate channel.

    Raises:
        SchemaViolationError: if the payload is malformed, schema major
            versions mismatch, or required constants are missing.
    """
    payload = _load_payload(json_data)

    # Schema-version compatibility: only accept majors equal to ours;
    # minors may exceed (forward-compatible additive changes).
    raw_version = payload.get("schema_version")
    if not isinstance(raw_version, list) or len(raw_version) != 2:
        raise SchemaViolationError(
            f"Missing or malformed schema_version: {raw_version!r}"
        )
    file_major = int(raw_version[0])
    expected_major, _ = SCHEMA_VERSION
    if file_major != expected_major:
        raise SchemaViolationError(
            f"Schema major version mismatch: file={raw_version}, "
            f"expected={list(SCHEMA_VERSION)}. Major version 2 is the first "
            f"stateful-aware schema; older payloads must be re-encoded by an "
            f"updated producer."
        )

    raw_nodes = payload.get("nodes", [])
    if not isinstance(raw_nodes, list):
        raise SchemaViolationError("'nodes' must be a list")
    nodes = [_parse_node(n) for n in raw_nodes]  # pyre-ignore[6]

    raw_storage = payload.get("storage_table", {})
    if not isinstance(raw_storage, dict):
        raise SchemaViolationError("'storage_table' must be a dict")
    storage_table = _parse_storage_table(raw_storage)  # pyre-ignore[6]

    in_spec = payload.get("in_spec")
    out_spec = payload.get("out_spec")
    if in_spec is not None and not isinstance(in_spec, str):
        raise SchemaViolationError("'in_spec' must be a string or null")
    if out_spec is not None and not isinstance(out_spec, str):
        raise SchemaViolationError("'out_spec' must be a string or null")
    raw_orig_args = payload.get("orig_args")
    orig_args: list[str] | None
    if raw_orig_args is None:
        orig_args = None
    elif isinstance(raw_orig_args, list) and all(
        isinstance(a, str) for a in raw_orig_args
    ):
        orig_args = [str(a) for a in raw_orig_args]
    else:
        raise SchemaViolationError("'orig_args' must be a list of strings or null")

    # Phase A: build the graph and discover required constant FQNs.
    graph, get_attr_fqns = _build_graph(
        nodes, storage_table, in_spec, out_spec, orig_args
    )

    # Cross-check: the producer-emitted ``required_constants`` (if present)
    # must match what we discovered while walking the nodes.  This catches
    # encoder/decoder drift early.
    raw_required = payload.get("required_constants")
    if raw_required is not None:
        if not isinstance(raw_required, list):
            raise SchemaViolationError("'required_constants' must be a list or null")
        if sorted(str(x) for x in raw_required) != sorted(get_attr_fqns):
            raise SchemaViolationError(
                f"required_constants={sorted(str(x) for x in raw_required)} "
                f"in payload disagrees with get_attr nodes "
                f"{sorted(get_attr_fqns)} in the graph; payload is corrupt."
            )

    # Phase B: seed constants on a fresh root, construct the GraphModule,
    # then reflip silently-registered buffers back to CONSTANT.
    return _seed_and_construct(graph, get_attr_fqns, constants)


def required_constants(payload: dict[str, object] | str | Path) -> list[str]:
    """Return the list of ``get_attr`` FQNs the payload requires.

    Convenience for callers that need to assemble the ``constants`` dict
    BEFORE calling :func:`deserialize_fx_graph` (e.g. to pull tensors from
    a sidecar safetensors file).
    """
    loaded = _load_payload(payload)
    raw = loaded.get("required_constants", [])
    if not isinstance(raw, list):
        raise SchemaViolationError("'required_constants' must be a list")
    return [str(x) for x in raw]
