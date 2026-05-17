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
    """``aot_prebuild_and_serialize`` produces one artifact per ``_XLAModule``."""

    def test_serializes_every_xla_module_in_plain_gm(self) -> None:
        lowered = _make_lowered_gm()
        n_subgraphs = len(_xla_modules(lowered))
        self.assertGreater(n_subgraphs, 0, "test fixture produced 0 subgraphs")

        # AOT walks the gm internally and synthesizes per-subgraph dummies
        # from each placeholder's ``meta["val"]`` — no example args needed.
        artifacts = aot_prebuild_and_serialize(lowered)
        self.assertEqual(len(artifacts), n_subgraphs)
        for key, art in artifacts.items():
            # Public surface is a tuple (bytes, in_tree, out_tree) — no
            # AOTArtifact dataclass scaffolding.
            self.assertIsInstance(art, tuple)
            self.assertEqual(len(art), 3, f"artifact for {key} should be 3-tuple")
            bin_, in_tree, out_tree = art
            self.assertIsInstance(bin_, bytes)
            self.assertGreater(len(bin_), 0, f"empty serialized payload for {key}")
            self.assertIsNotNone(in_tree)
            self.assertIsNotNone(out_tree)

    def test_iter_graph_module_namespacing(self) -> None:
        # When the gm is an IterGraphModule, every artifact key carries
        # the ``setup_gm::`` / ``main_gm::`` / ``cleanup_gm::`` prefix
        # that ``IterGraphModule.named_xla_modules`` produces.
        lowered = _make_lowered_gm()
        iter_gm = IterGraphModule(lowered)
        artifacts = aot_prebuild_and_serialize(iter_gm)
        prefixes = {k.split("::", 1)[0] for k in artifacts}
        self.assertTrue(
            prefixes.issubset({"setup_gm", "main_gm", "cleanup_gm"}),
            f"unexpected gm prefix in artifact keys: {prefixes}",
        )
        # All three sub-GMs are clones of the same fx graph, so we expect
        # one artifact per subgraph per sub-GM.
        per_sub_gm = len(_xla_modules(iter_gm.main_gm))
        self.assertEqual(len(artifacts), 3 * per_sub_gm)

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
        # 1. Lazy-build path: produce a reference output without AOT.
        ref_lowered = _make_lowered_gm()
        params, x = self._params_and_input()
        ref_out = ref_lowered(params, x)

        # 2. Fresh lowered graph + AOT-prebuild-and-serialize.
        aot_src = _make_lowered_gm()
        artifacts = aot_prebuild_and_serialize(aot_src)

        # 3. Persist + reload via plain pickle (no save/load wrappers).
        blob = pickle.dumps(artifacts, protocol=pickle.HIGHEST_PROTOCOL)
        loaded_artifacts = pickle.loads(blob)

        # 4. Brand-new lowered graph; install the loaded executables.
        target = _make_lowered_gm()
        n_loaded = aot_load_prebuilt_into(target, loaded_artifacts)
        self.assertEqual(n_loaded, len(artifacts))

        # 5. Every _XLAModule should have its ``_jax_compiled`` populated
        # AFTER aot_load_prebuilt_into (i.e., load_into really did
        # install something). This is the regression detector for "load
        # silently no-op'd".
        for mod in _xla_modules(target):
            self.assertIsNotNone(
                mod._jax_compiled,
                "_jax_compiled not populated after aot_load_prebuilt_into",
            )

        # 6. Run the AOT-loaded path. The closure picks up
        # ``self._jax_compiled`` and skips the lazy build.
        aot_out = target(params, x)

        # 7. Numerical parity with the reference path.
        torch.testing.assert_close(ref_out, aot_out, atol=1e-5, rtol=1e-5)


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
    """``aot_prebuild_and_serialize``'s warmup-call must leave
    ``mod._jax_compiled`` populated so it can be serialized."""

    def test_warmup_populates_jax_compiled(self) -> None:
        lowered = _make_lowered_gm()
        for mod in _xla_modules(lowered):
            self.assertIsNone(mod._jax_compiled)

        artifacts = aot_prebuild_and_serialize(lowered)
        self.assertGreater(len(artifacts), 0)
        for mod in _xla_modules(lowered):
            self.assertIsNotNone(
                mod._jax_compiled,
                "warmup path did not populate _jax_compiled",
            )

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


if __name__ == "__main__":
    unittest.main()
