# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Tests for recir.backends — Backend ABC + concrete subclasses."""

import tempfile
import unittest

from recir.backend import BackendType
from recir.backends import (
    Backend,
    InductorBackend,
    XLADynamoBridgeBackend,
    XLAGSPMDBackend,
    XLAShardMapBackend,
)
from recir.distributed.parallel_mode import DataParallel, GSPMDParallel
from recir.lowering.device_state import DeviceStateManager
from recir.lowering.xla_device_state import XLADeviceStateManager
from recir.passes import (
    comm_fusion_with_concat,
    remove_copy_from_optimizer,
    schedule_comm_wait,
)


class InductorBackendTest(unittest.TestCase):
    """InductorBackend defaults + customization + behavior."""

    def test_inductor_default_optimize_passes(self) -> None:
        """default_optimize_passes returns the three baseline passes in order."""
        passes = InductorBackend.default_optimize_passes()
        self.assertEqual(
            tuple(passes),
            (comm_fusion_with_concat, schedule_comm_wait, remove_copy_from_optimizer),
        )

    def test_inductor_defaults_apply_when_no_kwarg(self) -> None:
        """Constructing without optimize_passes uses the class defaults."""
        backend = InductorBackend()
        self.assertEqual(
            tuple(backend.optimize_passes),
            (comm_fusion_with_concat, schedule_comm_wait, remove_copy_from_optimizer),
        )

    def test_inductor_custom_optimize_passes_replaces_defaults(self) -> None:
        """Passing optimize_passes= replaces the defaults wholesale (no append)."""

        def my_pass(_gm: object) -> None:
            pass

        backend = InductorBackend(optimize_passes=[my_pass])
        self.assertEqual(list(backend.optimize_passes), [my_pass])

    def test_inductor_empty_list_disables_optimization(self) -> None:
        """An explicit empty list means zero optimization passes."""
        backend = InductorBackend(optimize_passes=[])
        self.assertEqual(list(backend.optimize_passes), [])

    def test_inductor_uses_iter_graph(self) -> None:
        """InductorBackend requires IterGraph wrapping."""
        backend = InductorBackend()
        self.assertTrue(backend.uses_iter_graph)

    def test_inductor_device_state_manager_is_base(self) -> None:
        """InductorBackend uses the base DeviceStateManager (no XLA host-device transfer)."""
        backend = InductorBackend()
        manager = backend.device_state_manager()
        self.assertIsInstance(manager, DeviceStateManager)
        # Should NOT be the XLA variant.
        self.assertNotIsInstance(manager, XLADeviceStateManager)

    def test_inductor_serialize_artifact_not_implemented(self) -> None:
        """serialize_artifact raises by default — Inductor has no AOT path yet."""
        backend = InductorBackend()
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(NotImplementedError):
                backend.serialize_artifact(None, tmp_dir)  # pyre-ignore[6]


class XLAGSPMDBackendTest(unittest.TestCase):
    """XLAGSPMDBackend properties + validation."""

    def test_default_optimize_passes_is_empty(self) -> None:
        """XLA handles optimization natively, so defaults are empty."""
        self.assertEqual(tuple(XLAGSPMDBackend.default_optimize_passes()), ())

    def test_uses_iter_graph(self) -> None:
        backend = XLAGSPMDBackend()
        self.assertFalse(backend.uses_iter_graph)

    def test_device_state_manager_is_xla(self) -> None:
        backend = XLAGSPMDBackend()
        self.assertIsInstance(backend.device_state_manager(), XLADeviceStateManager)

    def test_validate_with_gspmd_parallel_passes(self) -> None:
        """validate_with(GSPMDParallel) does not raise."""
        backend = XLAGSPMDBackend()
        mode = GSPMDParallel("replicate", num_ranks=1, num_processes=1)
        # Should not raise.
        backend.validate_with(mode)

    def test_validate_with_data_parallel_raises(self) -> None:
        """validate_with(DataParallel) raises with a helpful message."""
        backend = XLAGSPMDBackend()
        mode = DataParallel("replicate")
        with self.assertRaises(ValueError) as ctx:
            backend.validate_with(mode)
        self.assertIn("GSPMDParallel", str(ctx.exception))

    def test_xla_gspmd_rejects_iter_graph_true(self) -> None:
        """validate_with raises when uses_iter_graph=True is paired with
        GSPMD — IterGraphModule's setup/cleanup mirrors cannot handle the
        global literals GSPMD bakes into the FX graph at trace time.
        Subclass and override the @property to force True (the base default
        is False)."""

        class _IterGraphXLAGSPMD(XLAGSPMDBackend):
            @property
            def uses_iter_graph(self) -> bool:
                return True

        backend = _IterGraphXLAGSPMD()
        mode = GSPMDParallel("replicate", num_ranks=1, num_processes=1)
        with self.assertRaises(ValueError) as ctx:
            backend.validate_with(mode)
        self.assertIn("uses_iter_graph", str(ctx.exception))


class XLAShardMapBackendTest(unittest.TestCase):
    """XLAShardMapBackend validates against DataParallel/HybridParallel."""

    def test_default_optimize_passes_is_empty(self) -> None:
        self.assertEqual(tuple(XLAShardMapBackend.default_optimize_passes()), ())

    def test_uses_iter_graph(self) -> None:
        self.assertFalse(XLAShardMapBackend().uses_iter_graph)

    def test_validate_with_data_parallel_passes(self) -> None:
        """DataParallel is accepted."""
        backend = XLAShardMapBackend()
        mode = DataParallel("replicate")
        backend.validate_with(mode)

    def test_validate_with_gspmd_raises(self) -> None:
        """GSPMDParallel is rejected."""
        backend = XLAShardMapBackend()
        mode = GSPMDParallel("replicate", num_ranks=1, num_processes=1)
        with self.assertRaises(ValueError) as ctx:
            backend.validate_with(mode)
        self.assertIn("GSPMDParallel", str(ctx.exception))


class XLADynamoBridgeBackendTest(unittest.TestCase):
    """XLADynamoBridgeBackend accepts any non-GSPMD mode."""

    def test_default_optimize_passes_is_empty(self) -> None:
        self.assertEqual(tuple(XLADynamoBridgeBackend.default_optimize_passes()), ())

    def test_validate_with_data_parallel_passes(self) -> None:
        backend = XLADynamoBridgeBackend()
        backend.validate_with(DataParallel("replicate"))

    def test_validate_with_gspmd_raises(self) -> None:
        backend = XLADynamoBridgeBackend()
        with self.assertRaises(ValueError):
            backend.validate_with(
                GSPMDParallel("replicate", num_ranks=1, num_processes=1)
            )


class BackendABCTest(unittest.TestCase):
    """Backend ABC mechanics."""

    def test_abstract_class_cannot_instantiate(self) -> None:
        """Backend is abstract — instantiating it directly should fail."""
        with self.assertRaises(TypeError):
            Backend()  # pyre-ignore[45]

    def test_default_validate_with_is_noop(self) -> None:
        """The default validate_with returns None for any parallel_mode."""

        class _NoopBackend(Backend):
            # pyrefly: ignore [bad-override]
            def dispatch(self, gm: object) -> object:
                return gm

        backend = _NoopBackend()
        result = backend.validate_with(DataParallel("replicate"))
        self.assertIsNone(result)
        result = backend.validate_with(
            GSPMDParallel("replicate", num_ranks=1, num_processes=1)
        )
        self.assertIsNone(result)

    def test_subclass_with_no_overrides_uses_empty_defaults(self) -> None:
        """A backend that overrides only dispatch inherits the empty defaults."""

        class _MinimalBackend(Backend):
            # pyrefly: ignore [bad-override]
            def dispatch(self, gm: object) -> object:
                return gm

        backend = _MinimalBackend()
        self.assertEqual(tuple(backend.optimize_passes), ())
        self.assertFalse(backend.uses_iter_graph)


class TopLevelExportsTest(unittest.TestCase):
    """The new public surface is importable from the top-level recir package."""

    def test_top_level_backend_imports(self) -> None:
        """Backend types are re-exported via ``recir``."""
        import recir

        self.assertIs(recir.Backend, Backend)
        self.assertIs(recir.InductorBackend, InductorBackend)
        self.assertIs(recir.XLAGSPMDBackend, XLAGSPMDBackend)
        self.assertIs(recir.XLAShardMapBackend, XLAShardMapBackend)
        self.assertIs(recir.XLADynamoBridgeBackend, XLADynamoBridgeBackend)

    def test_backend_type_enum_still_exists(self) -> None:
        """BackendType enum is kept internally until C6 — must still import."""
        # Sanity-check that the enum still has its members.
        self.assertEqual(BackendType.INDUCTOR.value, "inductor")
        self.assertEqual(BackendType.XLA_GSPMD.value, "xla_gspmd")
