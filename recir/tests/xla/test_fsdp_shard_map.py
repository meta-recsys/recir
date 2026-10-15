# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Unit tests for FSDP support in the shard_map path (R5).

R5 makes ``_XLAModule._jit_shard_map`` consume the typed ``recir_state_layout``
meta (via ``get_state_layout``) produced by ``DataParallel`` / ``ParallelMode``
(R4), replacing the old ``is_fsdp_sharded`` bool. These tests assert that, given
placeholders tagged with an ``FSDPWeightShardSpec``, ``_jit_shard_map`` builds
the shard_map in/out ``PartitionSpec``s that the original FSDP-on-XLA path produced:

  * FSDP-weight and batch inputs -> ``P("torch_dist")`` (Shard(0)); everything
    else replicated ``P()``.
  * FSDP-derived outputs (including user diagnostics that carry /N shards) ->
    ``P("torch_dist", None...)`` (Shard(0)); other user and replicated outputs
    -> ``P()``.

These are STRUCTURAL: the specs are captured by patching ``shard_map`` to record
them and short-circuit before JAX lowering, so the test is hermetic and does not
execute the multi-device all_gather / reduce_scatter collectives. A full numeric
shard_map FSDP forward+step needs the multi-GPU / multi-process harness in
test_gspmd.py (``DataParallel`` + ``XLAShardMapBackend`` under DTensorTestBase);
the single-process end-to-end numeric analogue for the GSPMD path lives in
test_fsdp_gspmd.py::TestGSPMDFSDPSingleProcess.
"""

import unittest
from typing import Any, Dict, Optional, Tuple
from unittest import mock
from unittest.mock import patch

import jax
import torch
import torch.fx as fx
from jax.sharding import PartitionSpec as P
from recir.backends.types import BackendType
from recir.distributed.sharding_spec import FSDPWeightShardSpec, set_state_layout
from recir.lowering.xla_lower import _XLAModule

# IMPORTANT: must run BEFORE any `import jax` so the cuda plugin path patch
# is in effect when jax.devices() / jax.* first triggers plugin discovery.
from recir.tests.xla import _jax_cuda_plugin_fix  # noqa: F401, I001


class _StopBeforeLowering(Exception):
    """Sentinel raised from the patched ``shard_map`` once specs are captured,
    so ``_jit_shard_map`` stops before the JAX lower/compile (no collectives)."""


def _set_val(node: fx.Node, shape: Tuple[int, ...]) -> None:
    """Attach a fake ``node.meta['val']`` so ``_leaf_ndim`` sees an ndim."""
    node.meta["val"] = torch.empty(shape)


class ShardMapFsdpSpecsTest(unittest.TestCase):
    """``_jit_shard_map`` in/out specs are driven by ``recir_state_layout``."""

    N: int = 4  # device world size (P * D)
    PROC: int = 2  # process world size

    def _fsdp_spec(self) -> FSDPWeightShardSpec:
        return FSDPWeightShardSpec(
            shard_dim=0, device_world_size=self.N, process_world_size=self.PROC
        )

    def _capture_specs(
        self,
        gm: fx.GraphModule,
        torch_output_indices: Optional[set[int]] = None,
    ) -> Tuple[Any, Any]:
        """Run the real ``_jit_shard_map`` and capture the specs it hands to
        ``shard_map``, short-circuiting before the JAX lowering."""
        module = _XLAModule(gm, backend_type=BackendType.XLA_SHARD_MAP)
        module.torch_output_indices = torch_output_indices
        captured: Dict[str, Any] = {}

        def _fake_shard_map(
            fn: Any, mesh: Any, in_specs: Any, out_specs: Any, **kwargs: Any
        ) -> Any:
            captured["in_specs"] = in_specs
            captured["out_specs"] = out_specs
            raise _StopBeforeLowering()

        with patch("recir.lowering.xla_lower.shard_map", _fake_shard_map):
            with self.assertRaises(_StopBeforeLowering):
                module._jit_shard_map(lambda *a: a, (), torch.device("cpu"))
        return captured["in_specs"], captured["out_specs"]

    def test_state_layout_drives_in_and_out_specs(self) -> None:
        # placeholders: w (FSDP weight), x (batch input), b (replicated).
        graph = fx.Graph()
        w = graph.placeholder("w")
        set_state_layout(w, self._fsdp_spec())
        _set_val(w, (self.N, 3))
        x = graph.placeholder("x")
        x.meta["is_batch_input"] = True
        _set_val(x, (8, 3))
        b = graph.placeholder("b")
        _set_val(b, (3,))
        # outputs: updated FSDP weight (Shard(0)) + replicated bias (P()).
        graph.output((w, b))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        in_specs, out_specs = self._capture_specs(gm)

        # FSDP + batch inputs -> Shard(0); replicated input -> P().
        self.assertEqual(in_specs, (P("torch_dist"), P("torch_dist"), P()))
        # FSDP-derived 2D output -> Shard(0) = P("torch_dist", None);
        # replicated output -> P().
        self.assertEqual(out_specs, (P("torch_dist", None), P()))

    def test_legacy_is_fsdp_sharded_bool_is_ignored(self) -> None:
        # A placeholder carrying ONLY the legacy ``is_fsdp_sharded`` bool (no
        # typed ``recir_state_layout``) must be treated as replicated: the R5
        # shard_map path reads the spec via ``get_state_layout``, not the bool.
        graph = fx.Graph()
        w = graph.placeholder("w")
        w.meta["is_fsdp_sharded"] = True  # legacy bool, deliberately no spec
        _set_val(w, (self.N, 3))
        graph.output((w,))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        in_specs, out_specs = self._capture_specs(gm)

        self.assertEqual(in_specs, (P(),))
        self.assertEqual(out_specs, (P(),))

    def test_fsdp_derived_torch_output_keeps_sharded_layout(self) -> None:
        graph = fx.Graph()
        w = graph.placeholder("w")
        set_state_layout(w, self._fsdp_spec())
        _set_val(w, (self.N, 3))
        diagnostic = graph.call_function(torch.ops.aten.clone.default, args=(w,))
        _set_val(diagnostic, (self.N, 3))
        scalar = graph.call_function(torch.ops.aten.sum.default, args=(w,))
        _set_val(scalar, ())
        graph.output((diagnostic, scalar))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        in_specs, out_specs = self._capture_specs(
            gm,
            torch_output_indices={0, 1},
        )

        self.assertEqual(in_specs, (P("torch_dist"),))
        self.assertEqual(out_specs, (P("torch_dist", None), P()))

    def test_dead_state_inputs_are_donated_but_batch_inputs_are_not(self) -> None:
        graph = fx.Graph()
        w = graph.placeholder("w")
        set_state_layout(w, self._fsdp_spec())
        _set_val(w, (self.N, 3))
        x = graph.placeholder("x")
        x.meta["is_batch_input"] = True
        _set_val(x, (8, 3))
        state = graph.placeholder("state")
        _set_val(state, (3,))
        graph.output((w, state))
        gm = fx.GraphModule(torch.nn.Module(), graph)
        module = _XLAModule(gm, backend_type=BackendType.XLA_SHARD_MAP)
        module.safe_donate_indices = {0, 1, 2}
        captured: Dict[str, Any] = {}

        def _fake_shard_map(fn: Any, **kwargs: Any) -> Any:
            return fn

        def _fake_jit(fn: Any, **kwargs: Any) -> Any:
            captured.update(kwargs)
            raise _StopBeforeLowering()

        with (
            patch("recir.lowering.xla_lower.shard_map", _fake_shard_map),
            patch("recir.lowering.xla_lower.jax.jit", _fake_jit),
        ):
            with self.assertRaises(_StopBeforeLowering):
                module._jit_shard_map(lambda *a: a, (), torch.device("cpu"))

        self.assertEqual(captured["donate_argnums"], (0, 2))

    def test_donated_state_output_feeds_the_next_step(self) -> None:
        graph = fx.Graph()
        state = graph.placeholder("state")
        _set_val(state, (2,))
        batch = graph.placeholder("batch")
        batch.meta["is_batch_input"] = True
        _set_val(batch, (2,))
        updated = graph.call_function(torch.ops.aten.add.Tensor, args=(state, batch))
        _set_val(updated, (2,))
        graph.output((updated,))
        gm = fx.GraphModule(torch.nn.Module(), graph)
        module = _XLAModule(gm, backend_type=BackendType.XLA_SHARD_MAP)
        module.safe_donate_indices = {0}
        module.keep_jax_output = True
        _, run = module._jit_shard_map(
            lambda state_value, batch_value: (state_value + batch_value,),
            (torch.tensor([1.0, 2.0]), torch.tensor([2.0, 3.0])),
            torch.device("cpu"),
        )

        first = run(torch.tensor([1.0, 2.0]), torch.tensor([2.0, 3.0]))
        second = run(first[0], torch.tensor([4.0, 5.0]))

        self.assertEqual(jax.device_get(second[0]).tolist(), [7.0, 10.0])

    def test_fsdp_input_uses_process_count_for_global_shape(self) -> None:
        graph = fx.Graph()
        weight = graph.placeholder("weight")
        set_state_layout(weight, self._fsdp_spec())
        _set_val(weight, (8, 3))
        graph.output((weight,))
        gm = fx.GraphModule(torch.nn.Module(), graph)
        module = _XLAModule(gm, backend_type=BackendType.XLA_SHARD_MAP)
        fake_compiled = mock.Mock(return_value=())
        fake_compiled.memory_analysis = None
        fake_lowered = mock.Mock()
        fake_lowered.compile.return_value = fake_compiled
        fake_jitted = mock.Mock()
        fake_jitted.lower.return_value = fake_lowered

        with (
            patch(
                "recir.lowering.xla_lower._get_global_devices",
                return_value=list(jax.devices()),
            ),
            patch(
                "recir.lowering.xla_lower._get_local_devices",
                return_value=list(jax.devices())[:1],
            ),
            patch("recir.lowering.xla_lower.shard_map", side_effect=lambda fn, **_: fn),
            patch("recir.lowering.xla_lower.jax.jit", return_value=fake_jitted),
            mock.patch.object(
                module,
                "_to_sharded_jax",
                return_value=mock.sentinel.sharded_weight,
            ) as to_sharded_jax,
        ):
            _, run = module._jit_shard_map(
                lambda value: (value,),
                (torch.ones(8, 3),),
                torch.device("cpu"),
            )
            run(torch.ones(8, 3))

        self.assertEqual(2, to_sharded_jax.call_count)
        for call in to_sharded_jax.call_args_list:
            self.assertEqual(
                len(jax.devices()),
                call.kwargs["fsdp_host_shard_count"],
            )

    def test_fsdp_output_without_ndim_meta_raises(self) -> None:
        # An FSDP-derived (non-torch) output leaf with no usable ndim meta is a
        # meta-propagation gap: building a replicated P() spec would read only
        # device-0's shard and shrink the updated /N weight by mesh size every
        # step, so _jit_shard_map must raise instead of silently mis-sharding.
        graph = fx.Graph()
        w = graph.placeholder("w")
        set_state_layout(w, self._fsdp_spec())
        _set_val(w, (self.N, 3))
        # An op tracing back to the FSDP weight but carrying NO val meta.
        relu = graph.call_function(torch.ops.aten.relu.default, args=(w,))
        graph.output((relu,))
        gm = fx.GraphModule(torch.nn.Module(), graph)
        module = _XLAModule(gm, backend_type=BackendType.XLA_SHARD_MAP)
        module.torch_output_indices = None
        with self.assertRaises(RuntimeError):
            module._jit_shard_map(lambda *a: a, (), torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
