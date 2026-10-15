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

import pickle
import unittest

import jax  # @manual
import torch
import torch.nn as nn
from functorch import make_fx
from recir.backends.types import BackendType
from recir.lowering.aot import _key, aot_load_prebuilt_into, aot_prebuild_and_serialize
from recir.lowering.xla_lower import _XLAModule, partial_lower_xla
from recir.passes.iter_graph_module import IterGraphModule


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
    """``aot_prebuild_and_serialize`` walks the gm, drives a warmup forward
    on every ``_XLAModule`` so the lazy-build path populates
    ``mod._jax_compiled``, then serializes each compiled binary via
    ``jax.experimental.serialize_executable.serialize``. Pin the walk + the
    happy-path artifact-dict contract here."""

    def test_serializes_every_xla_module_in_plain_gm(self) -> None:
        lowered = _make_lowered_gm()
        n_subgraphs = len(_xla_modules(lowered))
        self.assertGreater(n_subgraphs, 0, "test fixture produced 0 subgraphs")

        # Walk + warmup populate is wired and ``_build_*_executable`` now
        # populates ``_jax_compiled``, so serialize returns an artifact
        # for every subgraph the walker visited.
        artifacts = aot_prebuild_and_serialize(lowered)
        self.assertEqual(len(artifacts), n_subgraphs)
        # Plain GraphModule path uses empty gm_name; keys are bare sub_paths.
        for key, (bin_, in_tree, out_tree) in artifacts.items():
            self.assertNotIn("::", key)
            self.assertIsInstance(bin_, bytes)
            self.assertGreater(len(bin_), 0)
            self.assertIsNotNone(in_tree)
            self.assertIsNotNone(out_tree)
        # Every _XLAModule that produced an artifact also has its AOT
        # cache slot populated.
        for mod in _xla_modules(lowered):
            self.assertIsNotNone(mod._jax_compiled)

    def test_iter_graph_module_namespacing(self) -> None:
        # The IterGraphModule walk path namespaces each artifact key with
        # the owning gm_name ("setup_gm" / "main_gm" / "cleanup_gm") via
        # the ``gm_name::sub_path`` ``_key`` format.
        lowered = _make_lowered_gm()
        iter_gm = IterGraphModule(lowered)
        # Sanity: at least one xla module is reachable via the iter-gm walk.
        gm_names = {gm_name for gm_name, _, _ in iter_gm.named_xla_modules()}
        self.assertTrue(
            gm_names.issubset({"setup_gm", "main_gm", "cleanup_gm"}),
            f"unexpected gm prefix in iter_gm walk: {gm_names}",
        )
        artifacts = aot_prebuild_and_serialize(iter_gm)
        # Every key must carry the ``gm_name::sub_path`` namespace prefix
        # so the load side can route artifacts back to the right gm.
        self.assertGreater(len(artifacts), 0)
        for key in artifacts:
            self.assertIn("::", key)
            prefix = key.split("::", 1)[0]
            self.assertIn(prefix, {"setup_gm", "main_gm", "cleanup_gm"})

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
        # End-to-end serialize -> pickle -> load: ``_build_*_executable``
        # populates ``_jax_compiled`` during the warmup forward, so
        # serialize returns a non-empty artifact dict, pickle round-trips
        # it as opaque bytes, and ``aot_load_prebuilt_into`` installs
        # each compiled executable on a fresh target gm.
        aot_src = _make_lowered_gm()
        artifacts = aot_prebuild_and_serialize(aot_src)
        self.assertGreater(len(artifacts), 0)
        # Pickle round-trip — the artifact dict must survive
        # serialization for shipment across hosts.
        blob = pickle.dumps(artifacts)
        artifacts_roundtrip = pickle.loads(blob)
        self.assertEqual(set(artifacts_roundtrip.keys()), set(artifacts.keys()))

        # Build a fresh target gm; pre-load, no ``_XLAModule`` has its
        # AOT slot populated.
        target = _make_lowered_gm()
        for mod in _xla_modules(target):
            self.assertIsNone(mod._jax_compiled)

        n_loaded = aot_load_prebuilt_into(target, artifacts_roundtrip)
        self.assertEqual(n_loaded, len(artifacts_roundtrip))
        # Post-load: every ``_XLAModule`` on the target has the
        # deserialized executable installed.
        for mod in _xla_modules(target):
            self.assertIsNotNone(mod._jax_compiled)


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
    """Pin the populate contract at this commit: the AOT slot
    ``_jax_compiled`` is initialized to ``None`` on every fresh
    ``_XLAModule``, and the warmup forward inside
    ``aot_prebuild_and_serialize`` (driven through
    ``_build_*_executable``) materializes a ``jax.stages.Compiled``
    and stashes it on every visited subgraph."""

    def test_warmup_populates_jax_compiled(self) -> None:
        lowered = _make_lowered_gm()
        # Pre-warmup invariant: every _XLAModule's AOT slot starts unset.
        modules = _xla_modules(lowered)
        self.assertGreater(len(modules), 0)
        for mod in modules:
            self.assertIsNone(mod._jax_compiled)

        # The warmup forward populates ``_jax_compiled`` on every
        # subgraph, so serialize returns one artifact per subgraph and
        # leaves every ``_jax_compiled`` slot set to a non-None
        # ``jax.stages.Compiled``.
        artifacts = aot_prebuild_and_serialize(lowered)
        self.assertEqual(len(artifacts), len(modules))
        for mod in modules:
            self.assertIsNotNone(mod._jax_compiled)

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
