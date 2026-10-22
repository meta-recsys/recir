# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Tests for recir.backends — Backend ABC + concrete subclasses."""

import tempfile
import types
import unittest
from typing import cast
from unittest import mock

import torch
from recir.backends import (
    Backend,
    InductorBackend,
    XLADynamoBridgeBackend,
    XLAGSPMDBackend,
    XLAShardMapBackend,
)
from recir.backends.types import BackendType
from recir.compiler.runtime import COMPILED_OBJECT_KEY
from recir.distributed.parallel_mode import DataParallel, GSPMDParallel
from recir.lowering.device_state import DeviceStateManager
from recir.lowering.xla_device_state import XLADeviceStateManager
from recir.passes import remove_copy_from_optimizer
from recir.passes.iter_graph_module import IterGraphModule
from recir.training.compiled_module import RecirGSPMDCompiledModuleBase


class InductorBackendTest(unittest.TestCase):
    """InductorBackend defaults + customization + behavior."""

    def test_inductor_default_optimize_passes(self) -> None:
        backend = InductorBackend()
        self.assertEqual((remove_copy_from_optimizer,), tuple(backend.optimize_passes))

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

    def test_inductor_dispatch_lowers_each_iter_subgraph(self) -> None:
        iter_gm = IterGraphModule(torch.fx.symbolic_trace(torch.nn.Identity()))
        original_subgraphs = [
            iter_gm.setup_gm,
            iter_gm.main_gm,
            iter_gm.cleanup_gm,
        ]
        lowered_subgraphs = [
            torch.fx.symbolic_trace(torch.nn.Identity()) for _ in range(3)
        ]

        with mock.patch(
            "recir.backends.inductor.partial_lower",
            side_effect=lowered_subgraphs,
        ) as lower:
            result = InductorBackend().dispatch(iter_gm)

        self.assertIs(iter_gm, result)
        self.assertEqual(
            [mock.call(subgraph) for subgraph in original_subgraphs],
            lower.call_args_list,
        )
        self.assertIs(lowered_subgraphs[0], iter_gm.setup_gm)
        self.assertIs(lowered_subgraphs[1], iter_gm.main_gm)
        self.assertIs(lowered_subgraphs[2], iter_gm.cleanup_gm)

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

    def test_inductor_validate_with_gspmd_raises(self) -> None:
        """GSPMDParallel is rejected — GSPMD has no CUDA/Inductor equivalent."""
        backend = InductorBackend()
        mode = GSPMDParallel("replicate", num_ranks=1, num_processes=1)
        with self.assertRaises(ValueError) as ctx:
            backend.validate_with(mode)
        self.assertIn("GSPMDParallel", str(ctx.exception))

    def test_inductor_validate_with_data_parallel_passes(self) -> None:
        """DataParallel is accepted — validate_with is a no-op return."""
        backend = InductorBackend()
        result = backend.validate_with(DataParallel("fully_shard"))
        self.assertIsNone(result)


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

    def test_dump_hlo_is_forwarded_to_lowering(self) -> None:
        gm = torch.fx.symbolic_trace(torch.nn.Identity())
        lowered = torch.fx.symbolic_trace(torch.nn.Identity())

        with mock.patch(
            "recir.backends.xla.partial_lower_xla",
            return_value=lowered,
        ) as partial_lower:
            result = XLAGSPMDBackend(dump_hlo=True).dispatch(gm)

        self.assertIs(result, lowered)
        partial_lower.assert_called_once_with(
            gm,
            backend_type=BackendType.XLA_GSPMD,
            capture_hlo_text=True,
        )

    def test_hlo_capture_is_disabled_by_default(self) -> None:
        gm = torch.fx.symbolic_trace(torch.nn.Identity())

        with mock.patch(
            "recir.backends.xla.partial_lower_xla",
            return_value=gm,
        ) as partial_lower:
            XLAGSPMDBackend().dispatch(gm)

        partial_lower.assert_called_once_with(
            gm,
            backend_type=BackendType.XLA_GSPMD,
            capture_hlo_text=False,
        )

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


class XLADeviceStateManagerTest(unittest.TestCase):
    def test_materialize_output_preserves_nested_structure(self) -> None:
        manager = XLADeviceStateManager()
        first = object()
        second = object()
        first_tensor = torch.tensor(1.0)
        second_tensor = torch.tensor(2.0)

        with mock.patch(
            "recir.lowering.xla_device_state._jax_to_torch",
            side_effect=[first_tensor, second_tensor],
        ) as jax_to_torch:
            output = manager.materialize_output(((first,), [None, second]))

        self.assertEqual(((first_tensor,), [None, second_tensor]), output)
        self.assertEqual(
            [
                mock.call(first, manager.torch_device),
                mock.call(second, manager.torch_device),
            ],
            jax_to_torch.call_args_list,
        )

    def test_fast_path_keeps_host_stale_until_explicit_sync(self) -> None:
        manager = XLADeviceStateManager()
        flat_state = [torch.zeros(2)]
        gm = torch.fx.symbolic_trace(torch.nn.Identity())
        manager.device_state = [object()]
        manager.prepare_inputs(gm, flat_state, [], last_train_step=False)
        updated = object()
        updated_torch = torch.tensor([3.0, 4.0])

        with mock.patch(
            "recir.lowering.xla_device_state._jax_to_torch",
            return_value=updated_torch,
        ) as mock_jax_to_torch:
            manager.commit_step_state(
                flat_state,
                [updated],
                last_train_step=False,
            )
            output = manager.materialize_output(torch.tensor(1.0))
            cached_state = manager.device_state

            self.assertTrue(torch.equal(flat_state[0], torch.zeros(2)))
            self.assertTrue(torch.equal(output, torch.tensor(1.0)))
            self.assertIsNotNone(cached_state)
            assert cached_state is not None
            self.assertIs(cached_state[0], updated)
            mock_jax_to_torch.assert_not_called()

            manager.copy_state_to_host(flat_state)

        self.assertTrue(torch.equal(flat_state[0], updated_torch))
        self.assertIs(manager.device_state, cached_state)
        mock_jax_to_torch.assert_called_once_with(updated, manager.torch_device)

    def test_slow_path_still_syncs_flat_state(self) -> None:
        manager = XLADeviceStateManager()
        flat_state = [torch.zeros(2)]
        gm = torch.fx.symbolic_trace(torch.nn.Identity())
        manager.prepare_inputs(gm, flat_state, [], last_train_step=True)
        updated = object()
        updated_torch = torch.tensor([5.0, 6.0])

        with mock.patch(
            "recir.lowering.xla_device_state._jax_to_torch",
            return_value=updated_torch,
        ) as mock_jax_to_torch:
            manager.commit_step_state(
                flat_state,
                [updated],
                last_train_step=True,
            )

        self.assertTrue(torch.equal(flat_state[0], updated_torch))
        mock_jax_to_torch.assert_called_once_with(updated, manager.torch_device)

    def test_first_multi_device_call_keeps_state_on_device(self) -> None:
        manager = XLADeviceStateManager()
        flat_state = [torch.zeros(2)]
        gm = torch.fx.symbolic_trace(torch.nn.Identity())
        with (
            mock.patch("jax.device_count", return_value=8),
            mock.patch.object(manager, "_configure_xla_modules") as configure,
        ):
            inputs = manager.prepare_inputs(gm, flat_state, [], last_train_step=False)
        updated = object()

        with mock.patch(
            "recir.lowering.xla_device_state._jax_to_torch"
        ) as mock_jax_to_torch:
            manager.commit_step_state(flat_state, [updated], last_train_step=False)

        self.assertIs(inputs[0], flat_state[0])
        configure.assert_called_once_with(gm, keep_jax_output=True)
        self.assertTrue(torch.equal(flat_state[0], torch.zeros(2)))
        cached_state = manager.device_state
        assert cached_state is not None
        self.assertIs(cached_state[0], updated)
        mock_jax_to_torch.assert_not_called()

    def test_meta_leaf_left_by_a_restore_raises(self) -> None:
        manager = XLADeviceStateManager()
        with self.assertRaisesRegex(RuntimeError, "still meta"):
            manager.check_no_meta_leaves(
                [torch.zeros(2), torch.empty(2, device="meta")]
            )

    def test_restore_on_a_mesh_checks_but_does_not_prefill(self) -> None:
        state_manager = mock.Mock()
        flat_state = [torch.zeros(2)]
        compiled_step = types.SimpleNamespace()
        compiled_step.__dict__[COMPILED_OBJECT_KEY] = types.SimpleNamespace(
            state_manager=state_manager, gm=object(), flat_state=flat_state
        )
        module = types.SimpleNamespace(_compiled_step=compiled_step)
        with mock.patch("jax.device_count", return_value=8):
            RecirGSPMDCompiledModuleBase.finalize_restore(
                cast(RecirGSPMDCompiledModuleBase, module)
            )
        state_manager.check_no_meta_leaves.assert_called_once_with(flat_state)
        state_manager.invalidate_device_state.assert_called_once_with()
        state_manager.prefill_device_state_from_host.assert_not_called()

    def test_final_call_flushes_device_state_before_host_inputs(self) -> None:
        manager = XLADeviceStateManager()
        cached_state = [object()]
        manager.device_state = cached_state
        gm = torch.fx.symbolic_trace(torch.nn.Identity())
        flat_state = [torch.zeros(2)]
        user_input = torch.ones(2)

        with mock.patch.object(manager, "copy_state_to_host") as copy_state_to_host:
            inputs = manager.prepare_inputs(
                gm,
                flat_state,
                [user_input],
                last_train_step=True,
            )
            retry_inputs = manager.prepare_inputs(
                gm,
                flat_state,
                [user_input],
                last_train_step=True,
            )

        self.assertEqual(
            [mock.call(flat_state), mock.call(flat_state)],
            copy_state_to_host.call_args_list,
        )
        self.assertIs(manager.device_state, cached_state)
        self.assertIs(inputs[0], flat_state[0])
        self.assertIs(inputs[1], user_input)
        self.assertIs(retry_inputs[0], flat_state[0])
        self.assertIs(retry_inputs[1], user_input)


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
