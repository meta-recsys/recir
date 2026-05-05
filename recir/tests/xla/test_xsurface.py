#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""Tests for xsurface V3 model tracing and XLA compilation.

- XSurfaceFakeTracingCompatTest: verifies make_fx tracing with FakeTensors.
- TestXSurfaceXLACompile: verifies full XLA GSPMD compile + execution (GPU).
"""

import os

# Spoof multi-device JAX on CPU so TestXSurfaceXLACompile (which requires
# >= 2 JAX devices) actually runs locally. Must be set BEFORE jax is imported.
# We unconditionally append the host_platform flag so it stays set even when
# the buck test env supplies its own XLA_FLAGS (which would shadow setdefault).
os.environ["JAX_PLATFORMS"] = "cpu"
_existing_xla_flags: str = os.environ.get("XLA_FLAGS", "")
if "xla_force_host_platform_device_count" not in _existing_xla_flags:
    os.environ["XLA_FLAGS"] = (
        _existing_xla_flags + " --xla_force_host_platform_device_count=2"
    ).strip()

import dataclasses
import unittest
from typing import Any, Dict, Tuple

import jax  # @manual
import torch
import torch.nn as nn
from hammer.v3.models.udd_fm_v3.udd_fm_train_v3 import BlueReelsVDDTrainV3
from hammer.v3.modules.recblock import _inputs_to_feature_bundle
from recir.api import _local_compile, compile, sync_device_state_to_torch
from recir.backend import BackendType
from recir.decompositions import SPMD_DECOMP_TABLE
from recir.distributed.parallel_mode import GSPMDParallel
from recir.gm_transformation import GraphModuleTransformation
from recir.lowering.fbgemm_fallback import _batched_complete_cumsum_impl
from recir.lowering.xla_lower import _XLAModule
from recir.tests.xla.gspmd_test_utils import assert_hlo_has_collective
from recir.tests.xla.xsurface_test_utils import (
    _build_full_model,
    _build_overrides,
    _create_random_train_batch,
    _make_sample_input,
    _verify_traced_graph,
    TEST_BATCH_SIZE,
)
from torch._subclasses.fake_tensor import FakeTensorMode
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

# Register gr.batched_complete_cumsum decomposition for XLA lowering.
SPMD_DECOMP_TABLE[torch.ops.gr.batched_complete_cumsum.default] = (
    _batched_complete_cumsum_impl
)

# buck2 test @//mode/opt //recir/tests/xla:test_xsurface


class XSurfaceFakeTracingCompatTest(unittest.TestCase):
    """Test fake tensor tracing compatibility for xsurface v3 modules.

    Uses recir._local_compile (which lifts all module params/buffers as
    function arguments and converts them to FakeTensors) to trace pipeline
    stages. The preprocessor and sparse arch are overridden with fixed-output
    dummies (following the recir Override pattern) to avoid dynamic shapes
    and data-dependent operations during tracing.
    """

    def test_make_fx_fake_trace_full_model(self) -> None:
        """Trace the full BlueReelsVDDTrainV3 via _local_compile.

        Builds the complete V3 model (RecBlock + over-arch) and traces
        forward end-to-end. The RecBlock's preprocessor and sparse arch
        are overridden; the over-arch components (DenseArchV2, MoE,
        task MLPs, debias, unary embeddings) are traced directly since they
        use only FakeTensor-compatible ops.

        KJT inputs are decomposed into plain tensors before passing to
        _local_compile and reconstructed inside forward_fn. This avoids
        the pytree unflatten path that calls KJT.__init__ with PH sentinels
        (which lack .numel()), causing an AttributeError during make_fx.
        """
        device = torch.device("cpu")
        model, feature_config = _build_full_model(device)

        # Build overrides for RecBlock's preprocessor + sparse arch.
        recblock_sample, _, _ = _make_sample_input(model.recblock, device)
        overrides = _build_overrides(model.recblock, recblock_sample, device)

        # Generate a random training batch for the full model inputs.
        batch = _create_random_train_batch(feature_config, device)

        # Decompose KJTs into plain tensors so make_fx does not try to
        # unflatten PH sentinels through KeyedJaggedTensor.__init__.
        ro_kjt = batch.ro_id_list_features
        ro_kjt_keys = ro_kjt.keys()
        ro_kjt_values = ro_kjt.values()
        ro_kjt_lengths = ro_kjt.lengths()

        nro_kjt = batch.nro_id_list_features
        nro_kjt_keys = nro_kjt.keys()
        nro_kjt_values = nro_kjt.values()
        nro_kjt_lengths = nro_kjt.lengths()

        unary_kjt = batch.unary_features
        unary_kjt_keys = unary_kjt.keys()  # pyre-ignore
        unary_kjt_values = unary_kjt.values()  # pyre-ignore
        unary_kjt_lengths = unary_kjt.lengths()  # pyre-ignore

        # pyre-ignore
        def forward_fn(
            mod: BlueReelsVDDTrainV3,
            ro_float_features: torch.Tensor,
            nro_float_features: torch.Tensor,
            ro_id_list_values: torch.Tensor,
            ro_id_list_lengths: torch.Tensor,
            nro_id_list_values: torch.Tensor,
            nro_id_list_lengths: torch.Tensor,
            unary_values: torch.Tensor,
            unary_lengths: torch.Tensor,
            num_candidates: torch.Tensor,
            all_labels: Dict[str, torch.Tensor],
            supervision_weight: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            ro_id_list_features = KeyedJaggedTensor(
                keys=ro_kjt_keys,
                values=ro_id_list_values,
                lengths=ro_id_list_lengths,
            )
            nro_id_list_features = KeyedJaggedTensor(
                keys=nro_kjt_keys,
                values=nro_id_list_values,
                lengths=nro_id_list_lengths,
            )
            unary_features = KeyedJaggedTensor(
                keys=unary_kjt_keys,
                values=unary_values,
                lengths=unary_lengths,
            )
            # The post-refactor model exposes only forward(batch); rebuild
            # the batch from dynamic tensors and reuse the captured ``batch``
            # for the remaining (static) fields like supervision_label.
            new_batch = dataclasses.replace(
                batch,
                ro_float_features=ro_float_features,
                nro_float_features=nro_float_features,
                ro_id_list_features=ro_id_list_features,
                nro_id_list_features=nro_id_list_features,
                unary_features=unary_features,
                num_candidates=num_candidates,
                all_labels=all_labels,
                supervision_weight=supervision_weight,
            )
            logits, named_tensors = mod.forward(new_batch)
            predictions = named_tensors["prediction"]
            return logits, predictions

        fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
        result = _local_compile(
            forward_fn,
            overrides,
            model,
            batch.ro_float_features,
            batch.nro_float_features,
            ro_kjt_values,
            ro_kjt_lengths,
            nro_kjt_values,
            nro_kjt_lengths,
            unary_kjt_values,
            unary_kjt_lengths,
            batch.num_candidates,
            batch.all_labels,
            batch.supervision_weight,
            fake_mode=fake_mode,
        )
        _verify_traced_graph(self, result.gm)


# ---------------------------------------------------------------------------
# XLA GSPMD compilation tests (require GPU)
# ---------------------------------------------------------------------------


class TestXSurfaceXLACompile(unittest.TestCase):
    """Test XLA GSPMD compilation for the xsurface V3 model via RecIR."""

    def setUp(self) -> None:
        self.devices = jax.devices()
        self.num_ranks: int = len(self.devices)

    def _build_compiled_train_step(
        self, num_steps: int = 1
    ) -> Tuple[Any, BlueReelsVDDTrainV3]:
        """Build the model, compile with XLA GSPMD, and run num_steps."""
        torch.manual_seed(42)
        device = torch.device("cpu")
        global_bs = TEST_BATCH_SIZE * self.num_ranks
        model, feature_config = _build_full_model(device)

        for p in model.parameters():
            if p.is_floating_point():
                p.requires_grad_(True)

        _, ro_keys, nro_keys = _make_sample_input(
            model.recblock, device, batch_size=global_bs
        )
        batch = _create_random_train_batch(
            feature_config,
            device,
            batch_size=global_bs,
            ro_kjt_keys=ro_keys,
            nro_kjt_keys=nro_keys,
        )
        all_labels = batch.all_labels
        assert all_labels is not None
        batch_sample = _inputs_to_feature_bundle(
            batch.ro_id_list_features,
            None,
            batch.nro_id_list_features,
            batch.num_candidates,
            all_labels,
        )
        overrides = _build_overrides(
            model.recblock, batch_sample, device, batch_size=global_bs
        )

        ro_kjt = batch.ro_id_list_features
        ro_kjt_keys: list[str] = ro_kjt.keys()
        ro_kjt_values = ro_kjt.values()
        ro_kjt_lengths = ro_kjt.lengths()

        nro_kjt = batch.nro_id_list_features
        nro_kjt_keys: list[str] = nro_kjt.keys()
        nro_kjt_values = nro_kjt.values()
        nro_kjt_lengths = nro_kjt.lengths()

        unary_kjt = batch.unary_features
        assert unary_kjt is not None
        unary_kjt_keys: list[str] = unary_kjt.keys()
        unary_kjt_values = unary_kjt.values()
        unary_kjt_lengths = unary_kjt.lengths()

        opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)

        @compile(
            module_override=overrides,
            parallel_mode=GSPMDParallel(
                "replicate",
                loss_reduction="sum",
                num_ranks=self.num_ranks,
                num_processes=1,
            ),
            # pyre-ignore[6]
            gm_transformation=GraphModuleTransformation(
                backend=BackendType.XLA_GSPMD,
                enable_device_state=True,
            ),
        )
        # pyre-ignore
        def train_step(
            mod: nn.Module,
            opt: torch.optim.Optimizer,
            ro_float_features: torch.Tensor,
            nro_float_features: torch.Tensor,
            ro_id_list_values: torch.Tensor,
            ro_id_list_lengths: torch.Tensor,
            nro_id_list_values: torch.Tensor,
            nro_id_list_lengths: torch.Tensor,
            unary_values: torch.Tensor,
            unary_lengths: torch.Tensor,
            num_candidates: torch.Tensor,
            # supervision_{label,weight} are passed as (B_NRO, T) so dim-0
            # is the batch dim and divides evenly across num_ranks. The model
            # expects (T, B_NRO), so we transpose back inside.
            supervision_label_t: torch.Tensor,
            supervision_weight_t: torch.Tensor,
            all_labels: Dict[str, torch.Tensor],
        ) -> None:
            ro_id_list_features = KeyedJaggedTensor(
                keys=ro_kjt_keys,
                values=ro_id_list_values,
                lengths=ro_id_list_lengths,
            )
            nro_id_list_features = KeyedJaggedTensor(
                keys=nro_kjt_keys,
                values=nro_id_list_values,
                lengths=nro_id_list_lengths,
            )
            unary_features = KeyedJaggedTensor(
                keys=unary_kjt_keys,
                values=unary_values,
                lengths=unary_lengths,
            )
            new_batch = dataclasses.replace(
                batch,
                ro_float_features=ro_float_features,
                nro_float_features=nro_float_features,
                ro_id_list_features=ro_id_list_features,
                nro_id_list_features=nro_id_list_features,
                unary_features=unary_features,
                num_candidates=num_candidates,
                all_labels=all_labels,
                supervision_label=supervision_label_t.t().contiguous(),
                supervision_weight=supervision_weight_t.t().contiguous(),
            )
            all_losses, _ = mod.forward(new_batch)  # pyre-ignore[29]
            loss: torch.Tensor = all_losses.sum()
            loss.backward()
            opt.step()

        for _ in range(num_steps):
            train_step(
                model,
                opt,
                batch.ro_float_features,
                batch.nro_float_features,
                ro_kjt_values,
                ro_kjt_lengths,
                nro_kjt_values,
                nro_kjt_lengths,
                unary_kjt_values,
                unary_kjt_lengths,
                batch.num_candidates,
                batch.supervision_label.t().contiguous(),
                batch.supervision_weight.t().contiguous(),
                batch.all_labels,
            )
            sync_device_state_to_torch(train_step)

        compiled_obj = train_step.__dict__["_compiled_obj"]
        return compiled_obj, model

    def test_xsurface_compiles(self) -> None:
        """Verify the full model compiles through the XLA GSPMD pipeline."""
        compiled_obj, _ = self._build_compiled_train_step()
        self.assertIsNotNone(compiled_obj)

    def test_xsurface_has_xla_subgraphs(self) -> None:
        """Verify XLA subgraphs exist in setup/main/cleanup graphs."""
        compiled_obj, _ = self._build_compiled_train_step()
        for gm_name in ("setup_gm", "main_gm", "cleanup_gm"):
            gm = getattr(compiled_obj.gm, gm_name)
            xla_modules = [m for m in gm.modules() if isinstance(m, _XLAModule)]
            self.assertGreaterEqual(
                len(xla_modules),
                1,
                f"{gm_name}: expected >= 1 XLA subgraph, got {len(xla_modules)}.",
            )

    def test_xsurface_hlo_has_collective(self) -> None:
        """Verify the compiled HLO contains all-reduce for GSPMD."""
        compiled_obj, _ = self._build_compiled_train_step()
        assert_hlo_has_collective(self, compiled_obj, "all-reduce", "XLA_GSPMD")

    def test_xsurface_params_change_after_training(self) -> None:
        """Verify parameters actually change after training (not frozen)."""
        torch.manual_seed(42)
        device = torch.device("cpu")
        model, _ = _build_full_model(device)
        init_params = {n: p.detach().clone() for n, p in model.named_parameters()}

        compiled_obj, trained_model = self._build_compiled_train_step(num_steps=2)

        changed = 0
        for name, param in trained_model.named_parameters():
            init_p = init_params.get(name)
            if init_p is not None and not torch.equal(init_p, param):
                changed += 1
        self.assertGreater(
            changed,
            0,
            "No parameters changed after training - model may be frozen.",
        )
