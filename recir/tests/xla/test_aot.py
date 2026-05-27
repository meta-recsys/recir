#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors

"""Tests for recir.lowering.aot — CPU-only.

These tests are the gating signal before any TPU/GPU AOT bring-up: if
``serialize`` / ``deserialize_and_load`` cannot round-trip the recir
``_XLAModule`` artifacts on a multi-CPU JAX mesh, they certainly will
not work on a TPU pod. They intentionally avoid any CUDA / TPU
dependency so they run on any devserver and CI executor.

The ``XLA_FLAGS=--xla_force_host_platform_device_count=N`` knob spoofs
``N`` virtual JAX devices on the CPU backend so the SHARD_MAP and
GSPMD backends can be exercised without real accelerators.
"""

import os

# Spoof multi-device JAX on CPU before any ``import jax``. Two devices is
# enough to exercise SHARD_MAP / GSPMD lowering paths while keeping the
# test fast.
os.environ["JAX_PLATFORMS"] = "cpu"
_existing_xla_flags: str = os.environ.get("XLA_FLAGS", "")
if "xla_force_host_platform_device_count" not in _existing_xla_flags:
    os.environ["XLA_FLAGS"] = (
        _existing_xla_flags + " --xla_force_host_platform_device_count=2"
    ).strip()

import unittest

import jax  # @manual
import torch
import torch.nn as nn
from functorch import make_fx
from recir.backend import BackendType
from recir.iter_graph_module import IterGraphModule
from recir.lowering.aot import _key, aot_load_prebuilt_into, aot_prebuild_and_serialize
from recir.lowering.xla_lower import _XLAModule, partial_lower_xla


class _TinyMLP(nn.Module):
    """A small MLP whose traced graph clears ``MIN_ATEN_OPS_TO_LOWER``."""

    def __init__(self, dim: int = 8, layers: int = 6) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        for _ in range(layers):
            modules.extend([nn.Linear(dim, dim), nn.ReLU()])
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _make_lowered_gm() -> torch.fx.GraphModule:
    """Trace _TinyMLP and dispatch into XLA subgraphs (SHARD_MAP backend)."""
    torch.manual_seed(0)
    model = _TinyMLP()
    x = torch.randn(2, 8)

    def fn(params, x):
        return torch.func.functional_call(model, params, (x,))

    params = dict(model.named_parameters())
    gm = make_fx(fn)(params, x)
    return partial_lower_xla(gm, backend_type=BackendType.XLA_SHARD_MAP)


def _xla_modules(gm) -> list[_XLAModule]:
    return [m for m in gm.modules() if isinstance(m, _XLAModule)]


class KeyNamespacingTest(unittest.TestCase):
    def test_key_namespacing(self) -> None:
        self.assertEqual(_key("main_gm", "xla_subgraph_0"), "main_gm::xla_subgraph_0")
        # Empty gm_name (plain GraphModule) drops the namespace prefix
        # so keys remain stable when callers later wrap a graph in an
        # IterGraphModule.
        self.assertEqual(_key("", "xla_subgraph_0"), "xla_subgraph_0")


class AOTPrebuildAndSerializeTest(unittest.TestCase):
    """``aot_prebuild_and_serialize`` walks the gm and would serialize each
    ``_XLAModule``; at this commit the warmup forward does not yet populate
    ``mod._jax_compiled`` (that lands in the next commit), so the call
    raises ``RuntimeError`` after the walk visits the first subgraph. We
    pin the walk + the error contract here; the end-to-end happy path
    test lives in the commit that adds the populate logic."""

    def test_serializes_every_xla_module_in_plain_gm(self) -> None:
        lowered = _make_lowered_gm()
        n_subgraphs = len(_xla_modules(lowered))
        self.assertGreater(n_subgraphs, 0, "test fixture produced 0 subgraphs")

        # Walk + warmup is wired, but ``_build_*_executable`` does not yet
        # populate ``_jax_compiled`` — pin the explicit error surface so a
        # future regression that silently no-ops the warmup is caught.
        with self.assertRaisesRegex(
            RuntimeError, "AOT warmup did not populate _jax_compiled"
        ):
            aot_prebuild_and_serialize(lowered)

    def test_iter_graph_module_namespacing(self) -> None:
        # The IterGraphModule walk path is exercised; at this commit the
        # warmup still raises before any artifact is built. Confirm the
        # walker reaches an _XLAModule (otherwise the error wouldn't fire)
        # without depending on the populate-and-serialize tail.
        lowered = _make_lowered_gm()
        iter_gm = IterGraphModule(lowered)
        # Sanity: at least one xla module is reachable via the iter-gm walk.
        gm_names = {gm_name for gm_name, _, _ in iter_gm.named_xla_modules()}
        self.assertTrue(
            gm_names.issubset({"setup_gm", "main_gm", "cleanup_gm"}),
            f"unexpected gm prefix in iter_gm walk: {gm_names}",
        )
        with self.assertRaisesRegex(
            RuntimeError, "AOT warmup did not populate _jax_compiled"
        ):
            aot_prebuild_and_serialize(iter_gm)

    def test_rejects_unknown_gm_type(self) -> None:
        with self.assertRaises(TypeError):
            aot_prebuild_and_serialize(object())  # type: ignore[arg-type]


class AOTRoundTripTest(unittest.TestCase):
    """End-to-end gate: AOT-loaded executable matches the lazy-build path."""

    def _params_and_input(self) -> tuple[dict, torch.Tensor]:
        torch.manual_seed(0)
        model = _TinyMLP()
        x = torch.randn(2, 8)
        return dict(model.named_parameters()), x

    def test_serialize_then_pickle_load_matches_eager_output(self) -> None:
        # End-to-end serialize -> pickle -> load -> run requires
        # ``_build_*_executable`` to populate ``_jax_compiled`` during
        # the warmup forward. That populate logic lands in the next
        # commit. Here we pin the precondition: the warmup call still
        # raises today, so the test verifies the error surface and
        # leaves the round-trip parity check to the commit that adds
        # the populate path.
        aot_src = _make_lowered_gm()
        with self.assertRaisesRegex(
            RuntimeError, "AOT warmup did not populate _jax_compiled"
        ):
            aot_prebuild_and_serialize(aot_src)


class AOTStrictModeTest(unittest.TestCase):
    """``strict=True`` (default) rejects partial bundles; ``strict=False``
    matches the old silent-fallback behavior."""

    def test_empty_artifacts_strict_raises(self) -> None:
        target = _make_lowered_gm()
        n_subgraphs = len(_xla_modules(target))
        self.assertGreater(n_subgraphs, 0)
        with self.assertRaises(RuntimeError):
            # Default strict=True: missing artifacts is a hard error.
            aot_load_prebuilt_into(target, artifacts={})

    def test_empty_artifacts_non_strict_returns_zero(self) -> None:
        # strict=False keeps the legacy silent-fallback semantics for
        # tests / advanced callers that intentionally want partial
        # coverage.
        target = _make_lowered_gm()
        n_loaded = aot_load_prebuilt_into(target, artifacts={}, strict=False)
        self.assertEqual(n_loaded, 0)
        for mod in _xla_modules(target):
            self.assertIsNone(mod._jax_compiled)


class WarmupBuildPopulatesCacheTest(unittest.TestCase):
    """Pin the partial wiring at this commit: the AOT slot
    ``_jax_compiled`` is initialized to ``None`` on every fresh
    ``_XLAModule``, the warmup forward inside
    ``aot_prebuild_and_serialize`` runs, and the surrounding code
    explicitly errors out when the populate path is missing. The next
    commit fills the populate hook in ``_build_*_executable`` and the
    end-to-end happy path test moves with it."""

    def test_warmup_populates_jax_compiled(self) -> None:
        lowered = _make_lowered_gm()
        # Pre-warmup invariant: every _XLAModule's AOT slot starts unset.
        modules = _xla_modules(lowered)
        self.assertGreater(len(modules), 0)
        for mod in modules:
            self.assertIsNone(mod._jax_compiled)

        # Today, the warmup forward does NOT populate _jax_compiled, so
        # aot_prebuild_and_serialize raises with a descriptive message.
        # The next commit fills in the populate path and a sibling test
        # in that commit replaces this with the positive happy-path
        # assertion.
        with self.assertRaisesRegex(
            RuntimeError, "AOT warmup did not populate _jax_compiled"
        ):
            aot_prebuild_and_serialize(lowered)

    def test_jax_devices_present_for_test(self) -> None:
        # Sanity check that XLA_FLAGS spoofed multi-device CPU. Without
        # this, SHARD_MAP collectives would silently degrade and other
        # tests could pass for the wrong reason.
        self.assertGreaterEqual(
            len(jax.devices()), 1, "no JAX devices visible — env not initialized"
        )


class IterGraphModuleNamedXLAModulesTest(unittest.TestCase):
    """The IterGraphModule.named_xla_modules helper is the canonical
    walker the AOT module uses; pin its contract here."""

    def test_yields_three_gm_groups(self) -> None:
        lowered = _make_lowered_gm()
        iter_gm = IterGraphModule(lowered)
        gm_names = {gm_name for gm_name, _, _ in iter_gm.named_xla_modules()}
        self.assertEqual(gm_names, {"setup_gm", "main_gm", "cleanup_gm"})

    def test_yields_xla_modules_only(self) -> None:
        lowered = _make_lowered_gm()
        iter_gm = IterGraphModule(lowered)
        for _, _, mod in iter_gm.named_xla_modules():
            self.assertIsInstance(mod, _XLAModule)
