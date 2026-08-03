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

import copy
import unittest
from typing import Any, Dict, List, Tuple
from unittest.mock import patch

import jax  # @manual
import torch
import torch.nn as nn
from hammer.v3.models.udd_fm_v3.udd_fm_train_v3 import BlueReelsVDDTrainV3
from hammer.v3.models.udd_fm_v3.utils import world_candidate_sum
from hammer.v3.modules.recblock import _inputs_to_feature_bundle
from recir.api import _local_compile, compile, sync_device_state_to_torch
from recir.backends import XLAGSPMDBackend
from recir.decompositions import SPMD_DECOMP_TABLE
from recir.distributed.parallel_mode import GSPMDParallel
from recir.examples.rankfm_mvai.rankfm_batch_utils import (
    create_uniform_train_batch,
    resize_batch_inplace,
)
from recir.examples.rankfm_mvai.rankfm_dispatchers import (
    _build_overrides,
    DummyPreprocessorStack,
    DummySparseArch,
    TEST_BATCH_SIZE,
    TEST_MAX_UIH_SEQ_LEN,
    TEST_NUM_CANDIDATES,
)
from recir.lowering.fbgemm_fallback import _batched_complete_cumsum_impl
from recir.lowering.xla_lower import _XLAModule
from recir.region_transforms.padding import (
    compute_padded_shape,
    JaggedPairConfig,
    PaddingRegionTransform,
)
from recir.tests.xla.gspmd_test_utils import _iter_subgraphs, assert_hlo_has_collective
from recir.tests.xla.xsurface_test_utils import (
    _build_full_model,
    _make_sample_input,
    _verify_traced_graph,
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

    def _run_full_model_trace(self, bf16: bool = False) -> None:
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
        model, feature_config = _build_full_model(device, disable_seq_len_sampling=True)
        if bf16:
            # Exercise the HSTU/loss explicit bf16 activation casts under make_fx.
            # (The train_step torchax AMP autocast only affects the JAX dispatch,
            # not make_fx tracing, so that path is validated on the TPU run.)
            model.recursive_setattr("_bf16_training", True)

        # Build overrides for RecBlock's preprocessor + sparse arch.
        recblock_sample, _, _ = _make_sample_input(model.recblock, device)
        overrides = _build_overrides(model.recblock, recblock_sample, device)

        feature_config.num_candidates_per_request = TEST_NUM_CANDIDATES
        feature_config.history_len = TEST_MAX_UIH_SEQ_LEN
        batch = create_uniform_train_batch(feature_config, device)

        # Decompose KJTs into plain tensors for _local_compile.
        ro_kjt = batch.ro_id_list_features
        nro_kjt = batch.nro_id_list_features
        unary_kjt = batch.unary_features
        assert unary_kjt is not None

        model.set_recir_kjt_keys(ro_kjt.keys(), nro_kjt.keys(), unary_kjt.keys())

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

        for dispatcher in overrides:
            dispatcher.install_eager(model)

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

    def test_make_fx_fake_trace_full_model(self) -> None:
        """fp32 (default) end-to-end make_fx trace."""
        self._run_full_model_trace(bf16=False)

    def test_make_fx_fake_trace_full_model_bf16(self) -> None:
        """end-to-end make_fx trace with _bf16_training=True (HSTU/loss bf16
        activation casts). Guards against the bf16 stream casts breaking
        FakeTensor tracing before a TPU run."""
        self._run_full_model_trace(bf16=True)


# ---------------------------------------------------------------------------
# XLA GSPMD compilation tests (require GPU)
# ---------------------------------------------------------------------------


class TestXSurfaceXLACompile(unittest.TestCase):
    """Test XLA GSPMD compilation for the xsurface V3 model via RecIR."""

    def setUp(self) -> None:
        self.devices = jax.devices()
        self.num_ranks: int = len(self.devices)

    def _build_compiled_train_step(
        self,
        num_steps: int = 1,
        use_padding: bool = False,
    ) -> Tuple[Any, BlueReelsVDDTrainV3, List[torch.Tensor]]:
        """Build the model, compile with XLA GSPMD, and run num_steps."""
        torch.manual_seed(42)
        device = torch.device("cpu")
        global_bs = TEST_BATCH_SIZE * self.num_ranks
        model, feature_config = _build_full_model(device, disable_seq_len_sampling=True)

        for p in model.parameters():
            if p.is_floating_point():
                p.requires_grad_(True)

        # Both branches need a deterministic uniform trace batch: in the
        # non-padding path, B_NRO must divide ``num_local`` up front; in
        # the padding path, the override-built ``r_inter`` is sized off
        # the preprocessor's structure (which is driven by feature_config),
        # so the trace input's B_NRO has to match ``global_bs *
        # num_candidates_per_request`` exactly.  Using a jagged ROO batch
        # for trace input desyncs the post-pad ``r_dense`` from
        # ``r_inter`` at ``_forward_post_hstu``'s cat.  Mirrors
        # ``recir/examples/rankfm_mvai/xsurface_helpers.py``.
        feature_config.num_candidates_per_request = TEST_NUM_CANDIDATES
        feature_config.history_len = TEST_MAX_UIH_SEQ_LEN
        batch = create_uniform_train_batch(
            feature_config,
            device,
            batch_size=global_bs,
        )
        if use_padding:
            nd = self.num_ranks
            padded_bs = compute_padded_shape(global_bs, nd)
            pair_padded_bs = padded_bs * TEST_NUM_CANDIDATES
            override_bs = padded_bs
            # Deep-copy + resize avoids a second large-batch
            # ``RandomDataPipe`` call that can SIGABRT in its C++
            # allocator.
            override_batch = copy.deepcopy(batch)
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

        # Adapt orchestrator-style dispatchers into ``eager_preproc=`` callables:
        # each dispatcher's ``install_eager(model)`` matches the
        # ``Callable[[nn.Module], None]`` shape expected by ``eager_preproc=``.
        compile_kwargs: dict[str, Any] = {
            "eager_preproc": [dispatcher.install_eager for dispatcher in overrides],
            "parallel_mode": GSPMDParallel(
                "replicate",
                loss_reduction="sum",
                num_ranks=self.num_ranks,
            ),
            "backend": XLAGSPMDBackend(),
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
                skip_args={
                    "ro_id_list_values",
                    "ro_id_list_lengths",
                    "nro_id_list_values",
                    "nro_id_list_lengths",
                    "world_candidate_sum",
                },
            )
            compile_kwargs["region_transforms"] = [padding_transform]

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
            world_candidate_sum: torch.Tensor,
        ) -> torch.Tensor:
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
                world_candidate_sum,
            )
            loss: torch.Tensor = all_losses.sum()
            loss.backward()
            opt.step()
            return all_losses

        train_step = compile(**compile_kwargs)(train_step_fn)

        wcs = world_candidate_sum(batch.num_candidates)
        all_losses_list = []
        for _ in range(num_steps):
            step_losses = train_step(
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
                wcs,
            )
            sync_device_state_to_torch(train_step)
            if step_losses is not None:
                all_losses_list.append(step_losses.detach().clone())

        compiled_obj = train_step.__dict__["_compiled_obj"]
        return compiled_obj, model, all_losses_list

    def test_xsurface_compiles(self) -> None:
        """Verify the full model compiles through the XLA GSPMD pipeline."""
        compiled_obj, _, _ = self._build_compiled_train_step()
        self.assertIsNotNone(compiled_obj)

    def test_xsurface_has_xla_subgraphs(self) -> None:
        """Verify XLA subgraphs exist in setup/main/cleanup graphs."""
        compiled_obj, _, _ = self._build_compiled_train_step()
        for gm_name, gm in _iter_subgraphs(compiled_obj.gm):
            xla_modules = [m for m in gm.modules() if isinstance(m, _XLAModule)]
            self.assertGreaterEqual(
                len(xla_modules),
                1,
                f"{gm_name}: expected >= 1 XLA subgraph, got {len(xla_modules)}.",
            )

    def test_xsurface_hlo_has_collective(self) -> None:
        """Verify the compiled HLO contains all-reduce for GSPMD."""
        compiled_obj, _, _ = self._build_compiled_train_step()
        assert_hlo_has_collective(self, compiled_obj, "all-reduce", "XLA_GSPMD")

    def test_xsurface_compiles_with_padding(self) -> None:
        """Verify XLA GSPMD compilation works with PaddingRegionTransform.

        Also asserts that recir_padding ops actually appear in the compiled
        graph — without this we'd be smoke-testing the compile but not
        verifying PRT inserted any pad nodes (a silently-skipped PRT would
        still ``assertIsNotNone``).
        """
        compiled_obj, _, _ = self._build_compiled_train_step(use_padding=True)
        self.assertIsNotNone(compiled_obj)
        pad_targets = {
            torch.ops.recir_padding.pad_to_static.default,
            torch.ops.recir_padding.pad_count_to_sum.default,
            torch.ops.recir_padding.pad_jagged_lengths.default,
            torch.ops.recir_padding.pad_jagged_values.default,
        }
        seen_graphs: set[int] = set()
        pad_op_count = 0
        for _, gm in _iter_subgraphs(compiled_obj.gm):
            for sub in [gm, *gm.modules()]:
                if not hasattr(sub, "graph") or id(sub.graph) in seen_graphs:
                    continue
                seen_graphs.add(id(sub.graph))
                for node in sub.graph.nodes:
                    if node.op == "call_function" and node.target in pad_targets:
                        pad_op_count += 1
        self.assertGreater(
            pad_op_count,
            0,
            "Expected recir_padding ops in compiled graph but found none — "
            "PaddingRegionTransform may not have inserted any pad nodes.",
        )

    def test_xsurface_padding_numerical_parity(self) -> None:
        """Padded vs unpadded compile must produce identical weight updates.

        Padding adds dummy entries to satisfy static-shape constraints, but
        ``PaddingRegionTransform`` pairs each forward pad op with an
        ``unpad_grad`` / ``unpad_jagged_grad`` on the backward, and
        ``supervision_label``/``supervision_weight`` are padded with zeros
        (so dummy candidates contribute zero loss). After one train step
        from identical seeded init, parameters and buffers should match
        bit-exactly. If this test fails, the padding op pairing is broken.

        """
        nd = self.num_ranks
        global_bs = TEST_BATCH_SIZE * nd
        self.assertGreater(
            compute_padded_shape(global_bs, nd),
            global_bs,
            "padding must add ≥1 dummy sample at the B_RO dim",
        )
        self.assertGreater(
            compute_padded_shape(global_bs * TEST_NUM_CANDIDATES, nd),
            global_bs * TEST_NUM_CANDIDATES,
            "padding must add ≥1 dummy pair at the B_NRO dim",
        )

        _, model_no_pad, losses_no_pad = self._build_compiled_train_step(
            use_padding=False,
        )
        _, model_padded, losses_padded = self._build_compiled_train_step(
            use_padding=True,
        )

        def state_tensors(m: nn.Module) -> Dict[str, torch.Tensor]:
            # Skip override scaffolding: `DummyPreprocessorStack` and
            # `DummySparseArch` cache the real upstream module's output as
            # buffers (`_kjt_values_*`, `_kjt_lengths_*`, `_tensor_*`,
            # `_jt_values_*`, `_jt_lengths_*`, `_grad_anchor`). They aren't
            # part of the trainable model state we want to compare.
            skip_prefixes = tuple(
                mod_name + "."
                for mod_name, mod in m.named_modules()
                if isinstance(mod, (DummyPreprocessorStack, DummySparseArch))
            )
            state: Dict[str, torch.Tensor] = {}
            for n, p in m.named_parameters():
                if not n.startswith(skip_prefixes):
                    state[n] = p
            for n, b in m.named_buffers():
                if not n.startswith(skip_prefixes):
                    state[n] = b
            return state

        for step_idx, (ref_losses, pad_losses) in enumerate(
            zip(losses_no_pad, losses_padded)
        ):
            torch.testing.assert_close(
                pad_losses,
                ref_losses,
                atol=1e-6,
                rtol=1e-5,
                msg=lambda msg, s=step_idx: f"step {s} per-task losses diverged: {msg}",
            )

        pad_state = state_tensors(model_padded)
        ref_state = state_tensors(model_no_pad)
        self.assertEqual(set(pad_state.keys()), set(ref_state.keys()))
        for name, ref in ref_state.items():
            torch.testing.assert_close(
                pad_state[name],
                ref,
                atol=1e-4,
                rtol=1e-3,
                msg=lambda msg, n=name: f"state {n} diverged after train step: {msg}",
            )


class WorldCandidateSumTest(unittest.TestCase):
    """Regression coverage for the multi-host loss-normalization bug: the
    per-task loss denominator must be the GLOBAL real candidate count, not the
    per-host ``num_candidates.sum()``."""

    def test_local_sum_when_not_distributed(self) -> None:
        nc = torch.tensor([2, 3, 4], dtype=torch.int64)
        out = world_candidate_sum(nc)
        self.assertEqual(out.item(), 9)
        self.assertEqual(out.ndim, 0)

    def test_all_reduce_global_sum_when_distributed(self) -> None:
        nc = torch.tensor([2, 3, 4], dtype=torch.int64)

        def fake_all_reduce(t: torch.Tensor, op: object = None) -> None:
            t.add_(9)

        with (
            patch(
                "hammer.v3.models.udd_fm_v3.utils.dist.is_available",
                return_value=True,
            ),
            patch(
                "hammer.v3.models.udd_fm_v3.utils.dist.is_initialized",
                return_value=True,
            ),
            patch(
                "hammer.v3.models.udd_fm_v3.utils.dist.all_reduce",
                side_effect=fake_all_reduce,
            ) as mock_ar,
        ):
            out = world_candidate_sum(nc)

        mock_ar.assert_called_once()
        self.assertEqual(mock_ar.call_args.kwargs["op"], torch.distributed.ReduceOp.SUM)
        self.assertEqual(out.item(), 18)
        self.assertEqual(nc.sum().item(), 9)
