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
from recir.region_transforms.padding import (
    compute_padded_shape,
    JaggedPairConfig,
    PaddingRegionTransform,
)
from recir.tests.xla.gspmd_test_utils import assert_hlo_has_collective
from recir.tests.xla.xsurface_test_utils import (
    _build_full_model,
    _build_overrides,
    _create_random_train_batch,
    _make_sample_input,
    _verify_traced_graph,
    resize_batch_inplace,
    TEST_BATCH_SIZE,
    TEST_MAX_UIH_SEQ_LEN,
    TEST_NUM_CANDIDATES,
)
from torch._subclasses.fake_tensor import FakeTensorMode

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

        # Decompose KJTs into plain tensors for _local_compile.
        ro_kjt = batch.ro_id_list_features
        nro_kjt = batch.nro_id_list_features
        unary_kjt = batch.unary_features
        assert unary_kjt is not None

        model.set_recir_kjt_keys(ro_kjt.keys(), nro_kjt.keys(), unary_kjt.keys())

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
            supervision_label_t: torch.Tensor,
            supervision_weight_t: torch.Tensor,
            all_labels: Dict[str, torch.Tensor],
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            logits, named_tensors = mod(
                ro_float_features,
                nro_float_features,
                ro_id_list_values,
                ro_id_list_lengths,
                nro_id_list_values,
                nro_id_list_lengths,
                unary_values,
                unary_lengths,
                num_candidates,
                supervision_label_t,
                supervision_weight_t,
                all_labels,
            )
            predictions = named_tensors["prediction"]
            return logits, predictions

        from recir.api import _apply_module_overrides

        _apply_module_overrides(model, overrides)

        fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
        result = _local_compile(
            forward_fn,
            model,
            batch.ro_float_features,
            batch.nro_float_features,
            ro_kjt.values(),
            ro_kjt.lengths(),
            nro_kjt.values(),
            nro_kjt.lengths(),
            unary_kjt.values(),
            unary_kjt.lengths(),
            batch.num_candidates,
            batch.supervision_label.t().contiguous(),
            batch.supervision_weight.t().contiguous(),
            batch.all_labels,
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
        self, num_steps: int = 1, use_padding: bool = False
    ) -> Tuple[Any, BlueReelsVDDTrainV3]:
        """Build the model, compile with XLA GSPMD, and run num_steps."""
        torch.manual_seed(42)
        device = torch.device("cpu")
        global_bs = TEST_BATCH_SIZE * self.num_ranks
        model, feature_config = _build_full_model(device)

        for p in model.parameters():
            if p.is_floating_point():
                p.requires_grad_(True)

        # When padding is enabled, build the override at the post-pad
        # shapes PaddingRegionTransform feeds into the overridden sparse
        # arch.  Use the same formulas as ``_discover``: padded_bs is the
        # post-pad B_RO; pair_padded_bs is the post-pad B_NRO computed
        # from the trace-time batch's actual jagged B_NRO.  We synthesize
        # the override batch by resizing a fresh small batch up to those
        # post-pad sizes, avoiding a second large-batch ``RandomDataPipe``
        # call that can SIGABRT in its C++ allocator.  Mirrors
        # ``_build_compiled_train_step`` in
        # ``recir/examples/rankfm/train.py``.
        batch = _create_random_train_batch(
            feature_config,
            device,
            batch_size=global_bs,
        )
        if use_padding:
            nd = self.num_ranks
            padded_bs = compute_padded_shape(global_bs, nd)
            # Trace-time B_NRO comes from the actual (jagged) batch.
            trace_pair_bs = batch.nro_float_features.shape[0]
            pair_padded_bs = compute_padded_shape(trace_pair_bs, nd)
            override_bs = padded_bs
            override_batch = _create_random_train_batch(
                feature_config,
                device,
                batch_size=global_bs,
            )
            resize_batch_inplace(
                override_batch,
                target_b_ro=padded_bs,
                target_b_nro=pair_padded_bs,
            )
        else:
            override_bs = global_bs
            override_batch = batch
        override_all_labels = override_batch.all_labels
        assert override_all_labels is not None
        batch_sample = _inputs_to_feature_bundle(
            override_batch.ro_id_list_features,
            None,
            override_batch.nro_id_list_features,
            override_batch.num_candidates,
            override_all_labels,
        )
        overrides = _build_overrides(
            model.recblock, batch_sample, device, batch_size=override_bs
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

        base_gm_transform = GraphModuleTransformation(
            backend=BackendType.XLA_GSPMD,
            enable_device_state=True,
        )

        compile_kwargs: dict[str, Any] = {
            "module_override": overrides,
            "parallel_mode": GSPMDParallel(
                "replicate",
                loss_reduction="sum",
                num_ranks=self.num_ranks,
                num_processes=1,
            ),
            "gm_transformation": base_gm_transform,
        }

        # Store KJT keys so the model's forward can construct KJTs
        # from the raw tensor inputs.
        model.set_recir_kjt_keys(ro_kjt_keys, nro_kjt_keys, unary_kjt_keys)

        if use_padding:
            module_config = model.module_config
            contextual_budgets = dict(module_config.contextual_feature_to_max_length)
            history_budget = TEST_MAX_UIH_SEQ_LEN

            ro_per_sample = [
                contextual_budgets.get(key, history_budget) for key in ro_kjt_keys
            ]

            padding_transform = PaddingRegionTransform(
                jagged_pairs={
                    "ro_id_list_values": JaggedPairConfig(
                        "ro_id_list_lengths",
                        per_sample_budgets=ro_per_sample,
                    ),
                    "nro_id_list_values": JaggedPairConfig(
                        "nro_id_list_lengths",
                        per_sample_budgets=1,
                        batch_size_arg="nro_float_features",
                        count_arg="num_candidates",
                        max_per_sample_count=TEST_NUM_CANDIDATES,
                    ),
                    "unary_values": JaggedPairConfig(
                        "unary_lengths",
                        per_sample_budgets=1,
                        batch_size_arg="nro_float_features",
                        count_arg="num_candidates",
                        max_per_sample_count=TEST_NUM_CANDIDATES,
                    ),
                },
                batch_size_arg="ro_float_features",
                num_devices=self.num_ranks,
                num_processes=1,
            )
            compile_kwargs["region_transforms"] = [padding_transform]

        # pyre-ignore
        def train_step_fn(
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
            supervision_label_t: torch.Tensor,
            supervision_weight_t: torch.Tensor,
            all_labels: Dict[str, torch.Tensor],
        ) -> None:
            all_losses, _ = mod(
                ro_float_features,
                nro_float_features,
                ro_id_list_values,
                ro_id_list_lengths,
                nro_id_list_values,
                nro_id_list_lengths,
                unary_values,
                unary_lengths,
                num_candidates,
                supervision_label_t,
                supervision_weight_t,
                all_labels,
            )
            loss: torch.Tensor = all_losses.sum()
            loss.backward()
            opt.step()

        train_step = compile(**compile_kwargs)(train_step_fn)  # pyre-ignore[6]

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

    def test_xsurface_compiles_with_padding(self) -> None:
        """Verify XLA GSPMD compilation works with PaddingRegionTransform."""
        compiled_obj, _ = self._build_compiled_train_step(use_padding=True)
        self.assertIsNotNone(compiled_obj)

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
