# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""JSON schema dataclasses for the recir FX serde module.

Every field is JSON-friendly: a primitive, a list of primitives, a dict of
primitives, or a nested dataclass that obeys the same restriction.  The
serializer emits this schema as a single top-level JSON document; the
deserializer reverses it.

The schema is versioned via :data:`SCHEMA_VERSION`.  Forward-compatible
additive changes bump the minor version; breaking changes bump the major
version.  Major version ``2`` is the first stateful-aware schema: graphs can
carry ``get_attr`` nodes referencing tensor constants registered as
:data:`torch.export.unflatten._AttrKind.CONSTANT`.  Major version ``1``
(pre-stateful) graphs are NOT compatible with this deserializer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# Bump (major, minor) on additive changes; breaking changes bump major.
# Major 2 = stateful-aware: ``get_attr`` constants are first-class and the
# deserializer seeds them on the root via ``_AttrKind.CONSTANT`` BEFORE
# ``fx.GraphModule.__init__`` runs.
SCHEMA_VERSION: tuple[int, int] = (2, 0)


class SchemaViolationError(RuntimeError):
    """Raised when a JSON payload violates the schema or the serializer/
    deserializer policy.  Subclassed from :class:`RuntimeError` so it can be
    caught generically by callers that don't want to import this module.
    """


# ---------------------------------------------------------------------------
# FakeTensor encoding (node.meta['val'])
# ---------------------------------------------------------------------------


@dataclass
class StorageEntry:
    """One entry in the top-level storage table.

    Represents a single physical untyped storage shared by one or more
    FakeTensors that are aliases of one another.  The base tensor is
    materialised once during decode; aliases are reconstructed via
    :func:`torch.as_strided` on the base.
    """

    storage_id: str
    nbytes: int
    fake_device: str
    base_dtype: str


@dataclass
class FakeTensorSpec:
    """Encoding for a single ``FakeTensor`` instance attached to ``node.meta['val']``.

    The ``storage_id`` references an entry in the top-level
    :class:`SerializedGraph.storage_table`, modelling aliasing cleanly across
    nodes.
    """

    kind: str  # always "fake_tensor"
    storage_id: str
    storage_offset: int
    shape: list[int]
    stride: list[int]
    dtype: str
    fake_device: str
    requires_grad: bool


# ---------------------------------------------------------------------------
# Argument encoding (tagged union)
# ---------------------------------------------------------------------------


@dataclass
class Argument:
    """Tagged-union argument representation.

    ``kind`` enumerates:

    * ``"none"``        -- Python ``None``
    * ``"bool"``        -- Python ``bool``
    * ``"int"``         -- Python ``int``
    * ``"float"``       -- Python ``float``
    * ``"str"``         -- Python ``str``
    * ``"dtype"``       -- ``torch.dtype`` (encoded by canonical name)
    * ``"device"``      -- ``torch.device`` (encoded ``"<type>"`` or ``"<type>:<idx>"``)
    * ``"layout"``      -- ``torch.layout``
    * ``"memory_format"`` -- ``torch.memory_format``
    * ``"node"``        -- reference to another node by name
    * ``"list"``        -- list of ``Argument``
    * ``"tuple"``       -- tuple of ``Argument`` (decoded as a Python tuple)
    * ``"dict"``        -- dict of str -> ``Argument``
    """

    kind: str
    value: Any = None


# ---------------------------------------------------------------------------
# Node / Graph encoding
# ---------------------------------------------------------------------------


@dataclass
class SerializedNode:
    """One node in the serialized FX graph.

    For ``get_attr`` nodes ``target`` is the qualified attribute name (FQN)
    on the GraphModule's root.  The deserializer collects every distinct
    ``get_attr`` target into the ``required_constants`` list so callers can
    discover which tensors must be supplied via the ``constants`` argument.
    """

    name: str
    op: str
    target: str
    args: list[Argument] = field(default_factory=list)
    kwargs: dict[str, Argument] = field(default_factory=dict)
    val: FakeTensorSpec | list[FakeTensorSpec] | None = None


@dataclass
class SerializedGraph:
    """Top-level serialized GraphModule payload."""

    schema_version: tuple[int, int]
    nodes: list[SerializedNode]
    storage_table: dict[str, StorageEntry]
    # ``treespec_dumps`` strings (or ``None`` if the GM was not produced by
    # ``_PyTreeCodeGen``).  Decoding restores the codegen so the resulting
    # GM accepts and returns nested-pytree structures.
    in_spec: str | None = None
    out_spec: str | None = None
    # The user-facing positional argument names BEFORE pytree flattening.
    # ``None`` when the GM has no pytree codegen, in which case the
    # deserializer falls back to the placeholder names.
    orig_args: list[str] | None = None
    # FQNs of every distinct ``get_attr`` target referenced by the graph.
    # Callers consult this list to assemble the ``constants`` dict for the
    # deserializer.  Populated by the serializer; checked by the
    # deserializer.
    required_constants: list[str] = field(default_factory=list)
    # Optional placeholder ordering metadata; carried alongside the graph
    # so consumers can match positional inputs back to their semantic role.
    bindings: BindingTable | None = None


# ---------------------------------------------------------------------------
# BindingTable -- placeholder ordering metadata
# ---------------------------------------------------------------------------


# Allowed binding kinds.  ``constant`` is metadata-only -- constants live as
# attributes on the root, NOT as positional placeholder inputs.  Including
# ``constant`` here lets a producer record the FQNs of attribute-backed
# tensors alongside the placeholder ordering, but the deserializer's call
# vector is unaffected.
_BINDING_KINDS: frozenset[str] = frozenset(
    {"constant", "param", "buffer", "optim_state", "user_input"}
)


@dataclass
class Binding:
    """One entry in a :class:`BindingTable`.

    ``kind`` is one of ``"constant"``, ``"param"``, ``"buffer"``,
    ``"optim_state"`` or ``"user_input"``.

    ``constant`` bindings record the FQN of a tensor attribute on the root;
    they do NOT participate in the runtime call vector.  All other kinds are
    placeholders consumed positionally in the order they appear in the
    table.
    """

    kind: str
    fqn: str
    # Optional sub-key used by optimiser state (e.g. "exp_avg", "step").
    state_key: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in _BINDING_KINDS:
            raise SchemaViolationError(
                f"Unknown Binding kind {self.kind!r}; "
                f"expected one of {sorted(_BINDING_KINDS)}"
            )


@dataclass
class BindingTable:
    """Ordered collection of placeholder bindings.

    The ordering of :attr:`bindings` corresponds 1:1 with the order in which
    the graph expects placeholders to be supplied at call time, with one
    important exception: ``constant`` bindings are metadata only and do NOT
    occupy a positional slot.  Filter them out (``b.kind != "constant"``) to
    obtain the call-vector ordering.
    """

    bindings: list[Binding] = field(default_factory=list)

    def kinds(self) -> list[str]:
        """Convenience accessor returning a list of just the binding kinds."""
        return [b.kind for b in self.bindings]

    def constants(self) -> list[Binding]:
        """Return only the ``constant`` bindings (metadata-only entries)."""
        return [b for b in self.bindings if b.kind == "constant"]

    def call_vector(self) -> list[Binding]:
        """Return the bindings that occupy positional placeholder slots."""
        return [b for b in self.bindings if b.kind != "constant"]
