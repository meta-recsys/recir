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
GSPMD backends can be exercised without real accelerators (same trick
used by ``test_xsurface``).
"""

import os

# Spoof multi-device JAX on CPU before any ``import jax``. Four devices is
# enough to exercise SHARD_MAP / GSPMD lowering paths while keeping the
# test fast, and lets the AOT-topology test hold a *subset* of devices
# in a virtual topology while ``jax.local_devices()`` still reports the
# rest — i.e. the topology really does override device discovery.
os.environ["JAX_PLATFORMS"] = "cpu"
_existing_xla_flags: str = os.environ.get("XLA_FLAGS", "")
if "xla_force_host_platform_device_count" not in _existing_xla_flags:
    os.environ["XLA_FLAGS"] = (
        _existing_xla_flags + " --xla_force_host_platform_device_count=4"
    ).strip()

import dataclasses
import os.path
import tempfile
import unittest

import jax  # @manual
import torch
import torch.nn as nn
from functorch import make_fx
from recir.backend import BackendType
from recir.iter_graph_module import IterGraphModule
from recir.lowering.aot import (
    _ITER_GM_NAMES,
    _key,
    aot_compile_and_serialize,
    aot_load_into,
    aot_topology,
    AOTArtifact,
    load_artifacts,
    save_artifacts,
)
from recir.lowering.xla_lower import (
    _AOT_TOPOLOGY,
    _get_global_devices,
    _get_local_devices,
    _XLAModule,
    partial_lower_xla,
)


@dataclasses.dataclass
class _FakeTopology:
    """Minimal stand-in for ``jax.experimental.topologies.TopologyDescription``.

    The real :class:`TopologyDescription` only exposes a ``.devices``
    attribute, so any object implementing that contract is sufficient
    for the AOT-topology code path. Using a fake here keeps the test
    CPU-only (the real ``get_topology_desc(platform="tpu", ...)`` call
    would require a TPU runtime that is unavailable on devserver CI).
    """

    devices: list[object]


class _TinyMLP(nn.Module):
    """A small MLP whose traced graph clears ``MIN_ATEN_OPS_TO_LOWER``.

    The default segmentation predicate in ``recir.lowering.lowering_utils``
    drops any subgraph with fewer than 10 aten ops, so we stack enough
    Linear+ReLU pairs to comfortably exceed that floor while still
    staying tiny enough to compile in well under a second.
    """

    def __init__(self, dim: int = 8, layers: int = 6) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        for _ in range(layers):
            modules.extend([nn.Linear(dim, dim), nn.ReLU()])
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _make_lowered_gm() -> torch.fx.GraphModule:
    """Trace _TinyMLP and lower into XLA subgraphs (SHARD_MAP backend)."""
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


class AOTArtifactDataclassTest(unittest.TestCase):
    """The ``AOTArtifact`` dataclass is part of the on-disk contract."""

    def test_default_construction(self) -> None:
        art = AOTArtifact(serialized=b"abc", in_tree=None, out_tree=None)
        self.assertEqual(art.serialized, b"abc")
        self.assertIsNone(art.in_tree)
        self.assertIsNone(art.out_tree)

    def test_iter_gm_names_contract(self) -> None:
        # The aot module's IterGraphModule walk depends on these three
        # attribute names — keep them pinned so the contract is a test,
        # not a comment.
        self.assertEqual(_ITER_GM_NAMES, ("setup_gm", "main_gm", "cleanup_gm"))

    def test_key_namespacing(self) -> None:
        self.assertEqual(_key("main_gm", "xla_subgraph_0"), "main_gm::xla_subgraph_0")
        # Empty gm_name (plain GraphModule) drops the namespace prefix
        # so keys remain stable when callers later wrap a graph in an
        # IterGraphModule.
        self.assertEqual(_key("", "xla_subgraph_0"), "xla_subgraph_0")


class AOTCompileAndSerializeTest(unittest.TestCase):
    """``aot_compile_and_serialize`` produces one artifact per ``_XLAModule``."""

    def test_serializes_every_xla_module_in_plain_gm(self) -> None:
        lowered = _make_lowered_gm()
        n_subgraphs = len(_xla_modules(lowered))
        self.assertGreater(n_subgraphs, 0, "test fixture produced 0 subgraphs")

        # AOT walks the gm internally and synthesizes per-subgraph dummies
        # from each placeholder's ``meta["val"]`` — no example args needed.
        artifacts = aot_compile_and_serialize(lowered)
        self.assertEqual(len(artifacts), n_subgraphs)
        for key, art in artifacts.items():
            self.assertIsInstance(art, AOTArtifact)
            self.assertGreater(
                len(art.serialized), 0, f"empty serialized payload for {key}"
            )
            self.assertIsNotNone(art.in_tree)
            self.assertIsNotNone(art.out_tree)

    def test_iter_graph_module_namespacing(self) -> None:
        # When the gm is an IterGraphModule, every artifact key carries
        # the ``setup_gm::`` / ``main_gm::`` / ``cleanup_gm::`` prefix.
        lowered = _make_lowered_gm()
        iter_gm = IterGraphModule(lowered)
        artifacts = aot_compile_and_serialize(iter_gm)
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
            aot_compile_and_serialize(object())  # type: ignore[arg-type]


class AOTRoundTripTest(unittest.TestCase):
    """End-to-end gate: AOT-loaded executable matches the lazy-compile path."""

    def _params_and_input(self) -> tuple[dict, torch.Tensor]:
        torch.manual_seed(0)
        model = _TinyMLP()
        x = torch.randn(2, 8)
        return dict(model.named_parameters()), x

    def test_serialize_then_load_matches_eager_output(self) -> None:
        # 1. Lazy-compile path: produce a reference output without AOT.
        ref_lowered = _make_lowered_gm()
        params, x = self._params_and_input()
        ref_out = ref_lowered(params, x)

        # 2. Fresh lowered graph (so compile state isn't shared) +
        #    AOT-compile-and-serialize.
        aot_src = _make_lowered_gm()
        artifacts = aot_compile_and_serialize(aot_src)

        # 3. Persist + reload via the on-disk helpers (exercises pickle).
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "aot.bin")
            save_artifacts(artifacts, path)
            loaded_artifacts = load_artifacts(path)

        # 4. Brand-new lowered graph; install the loaded executables on it.
        target = _make_lowered_gm()
        n_loaded = aot_load_into(target, loaded_artifacts)
        self.assertEqual(n_loaded, len(artifacts))

        # 5. Every _XLAModule should be marked as AOT-loaded BEFORE we
        # invoke forward (i.e., load_into really did install something).
        for mod in _xla_modules(target):
            self.assertTrue(
                mod._aot_loaded,
                "AOT-loaded flag not set on _XLAModule after aot_load_into",
            )
            self.assertIsNotNone(
                mod._jax_compiled,
                "_jax_compiled not populated after aot_load_into",
            )

        # 6. Run the AOT-loaded path. This should NOT trigger a fresh
        # compile (the closure picks up self._jax_compiled).
        aot_out = target(params, x)

        # 7. Numerical parity with the reference path.
        torch.testing.assert_close(ref_out, aot_out, atol=1e-5, rtol=1e-5)

    def test_load_into_skips_unmatched_subgraphs(self) -> None:
        # If artifacts dict is empty, aot_load_into returns 0 and every
        # _XLAModule remains unloaded. The caller's next forward call
        # falls back to the regular lazy-compile path.
        target = _make_lowered_gm()
        n_loaded = aot_load_into(target, artifacts={})
        self.assertEqual(n_loaded, 0)
        for mod in _xla_modules(target):
            self.assertFalse(mod._aot_loaded)
            self.assertIsNone(mod._jax_compiled)


class WarmupCompilePopulatesCacheTest(unittest.TestCase):
    """``aot_compile_and_serialize``'s warmup-call must leave
    ``mod._jax_compiled`` populated so it can be serialized."""

    def test_warmup_populates_jax_compiled(self) -> None:
        lowered = _make_lowered_gm()
        for mod in _xla_modules(lowered):
            self.assertIsNone(mod._jax_compiled)

        artifacts = aot_compile_and_serialize(lowered)
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


class AOTTopologyContextManagerTest(unittest.TestCase):
    """``aot_topology`` overrides device discovery for AOT compile.

    These tests pin the contextvar contract: ``_get_global_devices`` and
    ``_get_local_devices`` must report ``topo.devices`` while the
    context is active, and revert to ``jax.{,_local}devices()`` once
    the context exits. The recir AOT compile path on a CPU host depends
    on this so the resulting binary targets the virtual TPU topology
    instead of the CPU backend that is actually running the compile.
    """

    def test_default_no_override(self) -> None:
        # Outside the context manager, the contextvar is None and the
        # device-discovery helpers passthrough to JAX directly.
        self.assertIsNone(_AOT_TOPOLOGY.get())
        self.assertEqual(_get_global_devices(), jax.devices())
        self.assertEqual(_get_local_devices(), jax.local_devices())

    def test_topology_overrides_device_lists(self) -> None:
        # Use a *subset* of the real CPU devices as the virtual topology
        # so the test asserts override behavior rather than identity.
        # Pick 2 devices out of the 4 spoofed by XLA_FLAGS.
        all_devs = jax.devices()
        self.assertGreaterEqual(
            len(all_devs),
            2,
            "test requires >= 2 spoofed CPU devices; check XLA_FLAGS",
        )
        topo = _FakeTopology(devices=all_devs[:2])
        with aot_topology(topo):
            self.assertIs(_AOT_TOPOLOGY.get(), topo)
            # Both helpers report the topology's devices — not just the
            # local subset — so GSPMD compilation sees the full mesh.
            self.assertEqual(_get_global_devices(), all_devs[:2])
            self.assertEqual(_get_local_devices(), all_devs[:2])
        # Context exit restores the JAX-backed view.
        self.assertIsNone(_AOT_TOPOLOGY.get())
        self.assertEqual(_get_global_devices(), all_devs)
        self.assertEqual(_get_local_devices(), jax.local_devices())

    def test_nested_contexts_stack_correctly(self) -> None:
        # ``aot_topology`` is implemented with ``contextvars`` so nested
        # invocations must restore the previous topology on exit, not
        # reset to None.
        all_devs = jax.devices()
        outer_topo = _FakeTopology(devices=all_devs[:2])
        inner_topo = _FakeTopology(devices=all_devs[:1])
        with aot_topology(outer_topo):
            self.assertIs(_AOT_TOPOLOGY.get(), outer_topo)
            with aot_topology(inner_topo):
                self.assertIs(_AOT_TOPOLOGY.get(), inner_topo)
                self.assertEqual(_get_global_devices(), all_devs[:1])
            # Inner exit restores OUTER, not None.
            self.assertIs(_AOT_TOPOLOGY.get(), outer_topo)
            self.assertEqual(_get_global_devices(), all_devs[:2])
        self.assertIsNone(_AOT_TOPOLOGY.get())

    def test_exception_inside_context_still_restores(self) -> None:
        # @contextmanager wraps the yield in try/finally so the contextvar
        # must reset even when the block raises.
        topo = _FakeTopology(devices=jax.devices()[:1])
        with self.assertRaises(RuntimeError):
            with aot_topology(topo):
                self.assertIs(_AOT_TOPOLOGY.get(), topo)
                raise RuntimeError("boom")
        self.assertIsNone(_AOT_TOPOLOGY.get())

    def test_aot_topology_re_exported_from_aot_module(self) -> None:
        # Public surface contract: callers import ``aot_topology`` from
        # ``recir.lowering.aot`` (the AOT module) — not from the
        # internal ``xla_lower`` module. Verify the re-export so we
        # don't accidentally drop it on a future refactor.
        from recir.lowering import aot as aot_module, xla_lower

        self.assertIs(aot_module.aot_topology, xla_lower.aot_topology)


class AOTTopologyDriveCompileTest(unittest.TestCase):
    """End-to-end: compile inside ``aot_topology`` produces a usable
    serialized executable.

    The earlier round-trip test validates serialize/deserialize on its
    own; this test additionally asserts the topology context manager
    does not break the AOT pipeline. We use a virtual CPU topology
    (i.e. devices the local backend already provides) so the compile
    actually executes — switching to a real TPU topology would require
    a TPU runtime.
    """

    def test_compile_inside_aot_topology_produces_artifacts(self) -> None:
        all_devs = jax.devices()
        topo = _FakeTopology(devices=all_devs[:2])
        with aot_topology(topo):
            lowered = _make_lowered_gm()
            artifacts = aot_compile_and_serialize(lowered)
        # Compile produced one artifact per _XLAModule, identical to the
        # non-topology path (the topology is what the binary targets,
        # not the count of subgraphs).
        n_subgraphs = len(_xla_modules(lowered))
        self.assertGreater(n_subgraphs, 0)
        self.assertEqual(len(artifacts), n_subgraphs)
        for key, art in artifacts.items():
            self.assertGreater(
                len(art.serialized), 0, f"empty serialized payload for {key}"
            )
        # Context-exit happened cleanly.
        self.assertIsNone(_AOT_TOPOLOGY.get())


if __name__ == "__main__":
    unittest.main()
