# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Round-trip tests for the recir FX-graph serde module.

These tests cover the stateful-aware decode path:

* End-to-end ``get_attr`` constants survive serialize -> deserialize and are
  registered as :data:`~torch.export.unflatten._AttrKind.CONSTANT` on the
  reconstructed GraphModule (NOT silently re-registered as buffers by
  ``GraphModule.__init__``).
* Numerical parity between the original and reconstructed GraphModules.
* Constants-free graphs deserialize without supplying ``constants``.
* Nested submodule attributes (``inner.weight_const``) round-trip correctly.
* ``call_module`` and ``call_method`` nodes are rejected by the serializer.

Every test that needs a graph builds it via :func:`recir.api._local_compile`
so the wire format is exercised on the same gm shape downstream consumers
will see.
"""

from __future__ import annotations

import json
import unittest

import torch
import torch.fx as fx
import torch.nn as nn
from recir.api import _local_compile, _LocalCompiledResult
from recir.serde import (
    Binding,
    BindingTable,
    deserialize_fx_graph,
    parse_bindings,
    SchemaViolationError,
    serialize_fx_graph,
)
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.graph_module import _get_attr


def _trace(mod: nn.Module, *args: torch.Tensor) -> _LocalCompiledResult:
    """Drive ``recir.api._local_compile`` on ``mod`` with a forward-only
    train_step; return the full :class:`_LocalCompiledResult` so callers can
    use ``.gm``, ``.constants``, ``.params``, ``.buffers`` consistently.
    """
    fake_mode = FakeTensorMode(allow_non_fake_inputs=True)

    def train_step(mod: nn.Module, *args: torch.Tensor) -> torch.Tensor:
        return mod(*args).sum()

    return _local_compile(train_step, None, mod, *args, fake_mode=fake_mode)


class _ScalarConstModule(nn.Module):
    """A tiny module whose forward references an inline tensor literal.

    ``make_fx`` (called via ``_local_compile``) lifts ``self.scale`` into a
    ``get_attr`` node, so the traced GraphModule serves as the canonical
    fixture for the round-trip test.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc: nn.Linear = nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Inline tensor literal -> make_fx will hoist this into a `get_attr`.
        scale = torch.tensor([0.5, 0.25, 0.125, 0.0625])
        return self.fc(x) * scale


class _NestedConstModule(nn.Module):
    """Module with an inner submodule that emits its own get_attr constant."""

    class _Inner(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Inner-submodule inline literal: still becomes a get_attr after
            # ``_local_compile`` (which currently flattens dotted FQNs).
            weight_const = torch.tensor([1.0, 2.0, 3.0, 4.0])
            return x + weight_const

    def __init__(self) -> None:
        super().__init__()
        self.inner: nn.Module = self._Inner()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x)


class _ConstantsFreeModule(nn.Module):
    """A module whose forward has no inline tensor literals.

    Used by the constants-free round-trip test so the gm production path
    is the same ``_local_compile`` we use everywhere else.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1: nn.Linear = nn.Linear(4, 4, bias=False)
        self.fc2: nn.Linear = nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


class SerdeRoundTripTest(unittest.TestCase):
    def _invoke(
        self,
        gm: fx.GraphModule,
        result: _LocalCompiledResult,
        mod: nn.Module,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Call a ``_local_compile``-traced gm with concrete inputs.

        ``_local_compile`` traces ``stateless_func`` whose call signature is
        ``(params, buffers, named_states, args, kwargs)`` where ``args`` is
        the user's positional arg tuple (here ``(mod, x)``) -- ``mod`` is
        unused inside the traced graph but the pytree spec preserves it.
        """
        return gm(result.params, result.buffers, result.named_states, (mod, x), {})

    def test_serde_round_trip_preserves_get_attr_constants(self) -> None:
        """Build a get_attr-bearing GM via _local_compile, serialize, deserialize, assert parity."""
        torch.manual_seed(0)
        mod = _ScalarConstModule()
        x = torch.randn(2, 4)

        result = _trace(mod, x)
        gm = result.gm
        constants = result.constants

        # Sanity: the recir API tagged at least one tensor get_attr.
        self.assertTrue(
            constants,
            "fixture is degenerate: _local_compile produced no get_attr constants",
        )
        # Per-CONSTANT-kind contract: tensors are NOT in named_buffers().
        for fqn in constants:
            self.assertNotIn(fqn, dict(gm.named_buffers()))

        ref = self._invoke(gm, result, mod, x)[0].detach().clone()

        payload = serialize_fx_graph(gm)
        # Wire format must round-trip through json.
        payload_json = json.loads(json.dumps(payload))

        rebuilt = deserialize_fx_graph(payload_json, constants=constants)

        # Constants must be accessible as attributes on the rebuilt module
        # (matching the original) and NOT show up under named_buffers().
        for fqn, value in constants.items():
            got = _get_attr(rebuilt, fqn)
            self.assertIsNotNone(got)
            torch.testing.assert_close(got, value)
            self.assertNotIn(fqn, dict(rebuilt.named_buffers()))

        out = self._invoke(rebuilt, result, mod, x)[0]
        torch.testing.assert_close(out, ref)

    def test_round_trip_nested_submodule_get_attr(self) -> None:
        """A nested-submodule literal flows through _local_compile + serde cleanly."""
        torch.manual_seed(0)
        mod = _NestedConstModule()
        x = torch.randn(4)

        result = _trace(mod, x)
        gm = result.gm
        constants = result.constants

        self.assertTrue(
            constants,
            "fixture is degenerate: _local_compile produced no get_attr constants",
        )

        ref = self._invoke(gm, result, mod, x)[0].detach().clone()

        payload = serialize_fx_graph(gm)
        rebuilt = deserialize_fx_graph(payload, constants=constants)

        for fqn, value in constants.items():
            got = _get_attr(rebuilt, fqn)
            torch.testing.assert_close(got, value)
            self.assertNotIn(fqn, dict(rebuilt.named_buffers()))

        torch.testing.assert_close(self._invoke(rebuilt, result, mod, x)[0], ref)

    def test_round_trip_constants_free_graph(self) -> None:
        """A graph with no get_attr nodes deserializes without `constants`."""
        torch.manual_seed(0)
        mod = _ConstantsFreeModule()
        x = torch.randn(2, 4)

        result = _trace(mod, x)
        gm = result.gm

        self.assertFalse(
            any(n.op == "get_attr" for n in gm.graph.nodes),
            "fixture is degenerate: traced GM unexpectedly has get_attr nodes",
        )
        ref = self._invoke(gm, result, mod, x)[0].detach().clone()

        payload = serialize_fx_graph(gm)
        # No `constants` argument -- deserializer must accept absence.
        rebuilt = deserialize_fx_graph(payload)
        torch.testing.assert_close(self._invoke(rebuilt, result, mod, x)[0], ref)

    def test_serializer_rejects_call_module(self) -> None:
        """Serializer must reject FX graphs containing call_module nodes."""
        graph = fx.Graph()
        x = graph.placeholder("x")
        graph.call_module("submod", args=(x,))
        graph.output(x)
        root = nn.Module()
        root.submod = nn.Linear(4, 4)  # pyre-ignore[16]
        gm = fx.GraphModule(root, graph)

        with self.assertRaisesRegex(SchemaViolationError, r"call_module"):
            serialize_fx_graph(gm)

    def test_serializer_rejects_call_method(self) -> None:
        """Serializer must reject FX graphs containing call_method nodes."""
        graph = fx.Graph()
        x = graph.placeholder("x")
        graph.call_method("relu", args=(x,))
        graph.output(x)
        gm = fx.GraphModule(nn.Module(), graph)

        with self.assertRaisesRegex(SchemaViolationError, r"call_method"):
            serialize_fx_graph(gm)

    def test_serializer_accepts_get_attr(self) -> None:
        """Serializer must accept get_attr nodes (they are no longer blacklisted)."""
        torch.manual_seed(0)
        mod = _ScalarConstModule()
        result = _trace(mod, torch.randn(2, 4))
        # Should not raise.
        payload = serialize_fx_graph(result.gm)
        # And the FQN(s) must appear in required_constants.
        required = payload.get("required_constants")
        self.assertIsInstance(required, list)
        self.assertTrue(required, "expected at least one required_constants entry")

    def test_deserialize_missing_constants_raises(self) -> None:
        """If get_attr targets exist and `constants` is None, raise clearly."""
        torch.manual_seed(0)
        mod = _ScalarConstModule()
        result = _trace(mod, torch.randn(2, 4))
        payload = serialize_fx_graph(result.gm)

        with self.assertRaisesRegex(SchemaViolationError, r"constants"):
            deserialize_fx_graph(payload, constants=None)

    def test_deserialize_partial_constants_raises(self) -> None:
        """If some required FQN is missing from `constants`, raise clearly."""
        torch.manual_seed(0)
        mod = _ScalarConstModule()
        result = _trace(mod, torch.randn(2, 4))
        payload = serialize_fx_graph(result.gm)

        # Pass a dict that's missing every required FQN.
        with self.assertRaisesRegex(SchemaViolationError, r"missing required FQN"):
            deserialize_fx_graph(payload, constants={"_not_a_real_fqn": torch.zeros(1)})

    def test_binding_table_round_trip(self) -> None:
        """BindingTable carries through the wire format and deserializes structurally-equal."""
        torch.manual_seed(0)
        mod = _ScalarConstModule()
        result = _trace(mod, torch.randn(2, 4))

        # Build a sample BindingTable that includes a constant entry per
        # constant FQN the recir API tagged.
        get_attr_targets = sorted(result.constants.keys())
        original_bindings = BindingTable(
            bindings=[
                Binding(kind="user_input", fqn="<input_0>"),
                Binding(kind="param", fqn="fc.weight"),
                Binding(
                    kind="optim_state", fqn="fc.weight", state_key="momentum_buffer"
                ),
                *[Binding(kind="constant", fqn=fqn) for fqn in get_attr_targets],
            ],
        )
        payload = serialize_fx_graph(result.gm, bindings=original_bindings)
        # Bindings appear in payload as a dict.
        bindings_payload = payload.get("bindings")
        self.assertIsInstance(bindings_payload, dict)

        # Deserialize the bindings and assert structural equality with the
        # original entries (kind, fqn, state_key all match in order).
        rebuilt = parse_bindings(bindings_payload)
        self.assertIsNotNone(rebuilt)
        assert rebuilt is not None  # for pyre
        self.assertEqual(len(rebuilt.bindings), len(original_bindings.bindings))
        for orig, got in zip(original_bindings.bindings, rebuilt.bindings):
            self.assertEqual(got.kind, orig.kind)
            self.assertEqual(got.fqn, orig.fqn)
            self.assertEqual(got.state_key, orig.state_key)
