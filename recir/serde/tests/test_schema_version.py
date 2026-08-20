# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Schema-version compatibility tests for the recir FX-graph serde module.

These tests pin the schema-version contract:

* ``SCHEMA_VERSION`` is ``(2, 0)`` (the first stateful-aware schema).
* ``serialize_fx_graph`` emits the current major.minor in every payload.
* ``deserialize_fx_graph`` rejects payloads whose major version disagrees
  with the current major (e.g. ``(1, 1)`` -- the abandoned pre-stateful
  schema).
* ``deserialize_fx_graph`` accepts payloads with a higher minor (forward-
  compatible additive changes).
* Malformed / missing ``schema_version`` fields raise
  :class:`SchemaViolationError`.
"""

from __future__ import annotations

import unittest

import torch
from functorch import make_fx
from recir.serde import (
    deserialize_fx_graph,
    SCHEMA_VERSION,
    SchemaViolationError,
    serialize_fx_graph,
)
from torch._subclasses.fake_tensor import FakeTensorMode


def _trace_trivial() -> dict[str, object]:
    """Trace a constants-free graph and return its serialized payload."""

    def fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.ops.aten.add.Tensor(x, y)

    x = torch.randn(2, 3)
    y = torch.randn(2, 3)
    with FakeTensorMode(allow_non_fake_inputs=True) as fm:
        fx_args = (fm.from_tensor(x), fm.from_tensor(y))
        gm = make_fx(fn, tracing_mode="fake")(*fx_args)
    return serialize_fx_graph(gm)


class SchemaVersionTest(unittest.TestCase):
    def test_schema_version_is_2_0(self) -> None:
        """T4 establishes (2, 0) as the first stateful-aware schema."""
        self.assertEqual(SCHEMA_VERSION, (2, 0))

    def test_serialized_payload_carries_current_version(self) -> None:
        payload = _trace_trivial()
        self.assertEqual(payload["schema_version"], list(SCHEMA_VERSION))

    def test_deserialize_rejects_old_major_version(self) -> None:
        """A payload tagged with major 1 must be rejected by the v2 deserializer."""
        payload = _trace_trivial()
        # Mutate the schema_version to the abandoned (1, 1) value.
        payload["schema_version"] = [1, 1]
        with self.assertRaisesRegex(SchemaViolationError, r"major version mismatch"):
            deserialize_fx_graph(payload)

    def test_deserialize_rejects_future_major_version(self) -> None:
        """A payload tagged with major 3 (future) must also be rejected."""
        payload = _trace_trivial()
        payload["schema_version"] = [3, 0]
        with self.assertRaisesRegex(SchemaViolationError, r"major version mismatch"):
            deserialize_fx_graph(payload)

    def test_deserialize_accepts_higher_minor_version(self) -> None:
        """Forward-compatible: a payload at (2, N) for N > 0 must still load."""
        payload = _trace_trivial()
        # Bump only the minor version; loading must succeed.
        payload["schema_version"] = [SCHEMA_VERSION[0], SCHEMA_VERSION[1] + 7]
        gm = deserialize_fx_graph(payload)
        # Sanity: rebuilt module is a real GraphModule with at least one
        # placeholder node.
        self.assertTrue(
            any(n.op == "placeholder" for n in gm.graph.nodes),
            "deserialized graph has no placeholders",
        )

    def test_deserialize_rejects_missing_version(self) -> None:
        payload = _trace_trivial()
        del payload["schema_version"]
        with self.assertRaisesRegex(SchemaViolationError, r"schema_version"):
            deserialize_fx_graph(payload)

    def test_deserialize_rejects_malformed_version(self) -> None:
        payload = _trace_trivial()
        payload["schema_version"] = "not-a-list"
        with self.assertRaisesRegex(SchemaViolationError, r"schema_version"):
            deserialize_fx_graph(payload)

    def test_deserialize_rejects_wrong_length_version(self) -> None:
        payload = _trace_trivial()
        payload["schema_version"] = [2]
        with self.assertRaisesRegex(SchemaViolationError, r"schema_version"):
            deserialize_fx_graph(payload)
