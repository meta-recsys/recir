# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""JSON FX :class:`torch.fx.GraphModule` serializer / deserializer.

The wire format is a self-contained JSON document: no pickle, no thrift, no
executable payload.  Tensor constants attached to the graph via ``get_attr``
nodes are NOT inlined in the payload; the deserializer expects the caller to
supply them out-of-band via the ``constants`` argument.  This separation keeps
the wire format small and human-debuggable while letting tensor weights ship
through a separate channel (e.g. safetensors).

Public API:

* :func:`serialize_fx_graph` -- encode an :class:`fx.GraphModule` to a dict.
* :func:`deserialize_fx_graph` -- decode a dict back into an
  :class:`fx.GraphModule`, seeding any ``get_attr`` constants on the root
  before :class:`fx.GraphModule` construction so they survive as
  :data:`_AttrKind.CONSTANT` rather than being silently re-registered as
  buffers by ``GraphModule.__init__``.
* :class:`Binding` / :class:`BindingTable` -- placeholder ordering metadata.
* :class:`SchemaViolationError` -- raised on malformed payload.
* :data:`SCHEMA_VERSION` -- current ``(major, minor)`` schema version.
"""

from __future__ import annotations

from .deserializer import deserialize_fx_graph, parse_bindings, required_constants
from .schema import Binding, BindingTable, SCHEMA_VERSION, SchemaViolationError
from .serializer import serialize_fx_graph


__all__ = [
    "Binding",
    "BindingTable",
    "SCHEMA_VERSION",
    "SchemaViolationError",
    "deserialize_fx_graph",
    "parse_bindings",
    "required_constants",
    "serialize_fx_graph",
]
