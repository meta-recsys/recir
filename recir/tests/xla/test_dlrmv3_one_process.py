# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for DLRMv3 model compilation through RecIR's XLA pipeline."""

from __future__ import annotations

import unittest
from copy import deepcopy
from typing import Any, Dict, Tuple

import jax  # @manual
import recir.lowering.fbgemm_fallback  # noqa: F401 -- registers fbgemm decompositions
import torch
import torch.nn as nn
from recir.api import sync_device_state_to_torch
from recir.backends import XLAGSPMDBackend
from recir.distributed.parallel_mode import GSPMDParallel
from recir.lowering.xla_lower import _XLAModule
from recir.tests.xla.dlrmv3_test_utils import (
    create_dlrmv3_wrapper,
    create_static_batch,
    create_tiny_dlrmv3,
    DlrmV3SparseDispatcher,
    DummyPreprocess,
    LOCAL_BS,
    SMALL_EMBEDDING_DIM,
    STATIC_NUM_CANDIDATES,
    STATIC_UIH_LEN,
)
from recir.tests.xla.gspmd_test_utils import _iter_subgraphs, assert_hlo_has_collective
from recir.training.compile import compile_step


def _flat_batch(batch: tuple, *, global_loss_denom: bool) -> tuple:
    return tuple(batch)


class TestDLRMv3XLA(unittest.TestCase):
    def setUp(self) -> None:
        self.devices = jax.devices()
        if len(self.devices) < 2:
            self.skipTest("Need >= 2 JAX devices")
        self.num_ranks: int = len(self.devices)

    def _build_compiled_train_step(
        self,
        num_steps: int = 1,
        *,
        dump_hlo: bool = False,
    ) -> Tuple[Any, nn.Module]:
        """Build wrapper model, compile, and run num_steps. Return compiled_obj."""
        torch.manual_seed(42)
        global_bs = LOCAL_BS * self.num_ranks
        wrapper, dummy, hstu_config = create_tiny_dlrmv3(global_bs)
        wrapper = wrapper.cpu()
        opt = torch.optim.SGD(wrapper.parameters(), lr=0.01, foreach=False)
        uih_values, uih_lengths, cand_values, cand_lengths = create_static_batch(
            hstu_config, global_bs
        )

        def train_step_fn(
            mod: nn.Module,
            opt: torch.optim.Optimizer,
            uih_values: torch.Tensor,
            uih_lengths: torch.Tensor,
            cand_values: torch.Tensor,
            cand_lengths: torch.Tensor,
        ) -> None:
            aux_losses: Dict[str, torch.Tensor] = mod(
                uih_values, uih_lengths, cand_values, cand_lengths
            )
            loss: torch.Tensor = torch.stack(list(aux_losses.values())).sum()
            loss.backward()
            opt.step()

        train_step = compile_step(
            train_step_fn,
            wrapper,
            (opt,),
            (uih_values, uih_lengths, cand_values, cand_lengths),
            dispatchers=(DlrmV3SparseDispatcher(dummy),),
            parallel_mode=GSPMDParallel(
                "replicate",
                loss_reduction="sum",
                num_ranks=self.num_ranks,
                num_processes=1,
            ),
            backend=XLAGSPMDBackend(dump_hlo=dump_hlo),
            flatten_batch=_flat_batch,
        )

        for _ in range(num_steps):
            train_step(wrapper, opt, uih_values, uih_lengths, cand_values, cand_lengths)
            # Sync JAX device state back to PyTorch tensors so model.parameters()
            # are up-to-date for the next step and for numerical comparison.
            sync_device_state_to_torch(train_step)

        return train_step.__dict__["_compiled_obj"], wrapper

    def test_dlrmv3_compiles(self) -> None:
        compiled_obj, _ = self._build_compiled_train_step()
        self.assertIsNotNone(compiled_obj)

    def test_dlrmv3_has_xla_subgraphs(self) -> None:
        # Post-NC12 (uses_iter_graph=False default), the gm is a plain
        # fx.GraphModule rather than an IterGraphModule wrapper. Use the
        # shared helper to iterate either shape.
        compiled_obj, _ = self._build_compiled_train_step()
        for gm_name, gm in _iter_subgraphs(compiled_obj.gm):
            xla_modules = [m for m in gm.modules() if isinstance(m, _XLAModule)]
            self.assertGreaterEqual(
                len(xla_modules),
                1,
                f"{gm_name}: expected at least 1 XLA subgraph, got {len(xla_modules)}.",
            )

    def test_dlrmv3_hlo_has_collective(self) -> None:
        compiled_obj, _ = self._build_compiled_train_step(dump_hlo=True)
        assert_hlo_has_collective(self, compiled_obj, "all-reduce", "XLA_GSPMD")

    def test_dlrmv3_numerical_parity(self) -> None:
        num_iters = 3
        # Create model and data ONCE
        torch.manual_seed(42)
        global_bs = LOCAL_BS * self.num_ranks
        wrapper, dummy, hstu_config = create_tiny_dlrmv3(global_bs)
        wrapper = wrapper.cpu()
        uih_values, uih_lengths, cand_values, cand_lengths = create_static_batch(
            hstu_config, global_bs
        )
        # Save initial params before any training
        init_params = {n: p.detach().clone() for n, p in wrapper.named_parameters()}
        # ── CPU baseline using same DlrmV3Wrapper path (eager mode) ──
        baseline_wrapper = deepcopy(wrapper)
        baseline_opt = torch.optim.SGD(
            baseline_wrapper.parameters(), lr=0.01, foreach=False
        )
        for _ in range(num_iters):
            aux_losses: Dict[str, torch.Tensor] = baseline_wrapper(
                uih_values, uih_lengths, cand_values, cand_lengths
            )
            loss: torch.Tensor = torch.stack(list(aux_losses.values())).sum()
            (loss / self.num_ranks).backward()
            baseline_opt.step()
            baseline_opt.zero_grad()
        baseline_params = {
            n: p.detach().clone() for n, p in baseline_wrapper.named_parameters()
        }
        # ── XLA GSPMD path (uses the original wrapper) ──
        opt = torch.optim.SGD(wrapper.parameters(), lr=0.01, foreach=False)

        def train_step_fn(
            mod: nn.Module,
            opt: torch.optim.Optimizer,
            uih_values: torch.Tensor,
            uih_lengths: torch.Tensor,
            cand_values: torch.Tensor,
            cand_lengths: torch.Tensor,
        ) -> None:
            aux_losses: Dict[str, torch.Tensor] = mod(
                uih_values, uih_lengths, cand_values, cand_lengths
            )
            loss: torch.Tensor = torch.stack(list(aux_losses.values())).sum()
            loss.backward()
            opt.step()

        train_step = compile_step(
            train_step_fn,
            wrapper,
            (opt,),
            (uih_values, uih_lengths, cand_values, cand_lengths),
            dispatchers=(DlrmV3SparseDispatcher(dummy),),
            parallel_mode=GSPMDParallel(
                "replicate",
                loss_reduction="sum",
                num_ranks=self.num_ranks,
                num_processes=1,
            ),
            backend=XLAGSPMDBackend(),
            flatten_batch=_flat_batch,
        )

        for _ in range(num_iters):
            train_step(wrapper, opt, uih_values, uih_lengths, cand_values, cand_lengths)
            sync_device_state_to_torch(train_step)

        # Verify params actually changed after training (not frozen)
        for name, init_p in init_params.items():
            baseline_p = baseline_params.get(name)
            if baseline_p is not None:
                self.assertFalse(
                    torch.equal(init_p, baseline_p),
                    f"CPU baseline param {name} did not change after training",
                )
        for name, param in wrapper.named_parameters():
            init_p = init_params.get(name)
            if init_p is not None:
                self.assertFalse(
                    torch.equal(init_p, param),
                    f"XLA param {name} did not change after training",
                )

        # Compare XLA vs CPU baseline parameters
        for name, param in wrapper.named_parameters():
            baseline_param = baseline_params.get(name)
            if baseline_param is not None:
                torch.testing.assert_close(
                    param,
                    baseline_param,
                    msg=f"XLA vs baseline mismatch for {name}",
                )


class TestDummyPreprocessShapes(unittest.TestCase):
    def test_shapes_match(self) -> None:
        batch_size = LOCAL_BS * 2  # global_bs for 2 ranks
        wrapper, hstu_config, _, emb_keys = create_dlrmv3_wrapper(batch_size)
        wrapper = wrapper.cpu()
        batch = create_static_batch(hstu_config, batch_size)
        uih_values, uih_lengths, cand_values, cand_lengths = batch
        # Run real DlrmV3Preprocess
        real_preprocess = wrapper.preprocess_mod
        real_out = real_preprocess(uih_values, uih_lengths, cand_values, cand_lengths)
        (
            real_emb_lengths,
            real_emb_values,
            real_payload_features,
            _real_max_uih_len,
            real_uih_seq_lengths,
            _real_max_num_candidates,
            real_num_candidates,
            _real_total_targets,
            _real_total_uih_len,
        ) = real_out
        # Run DummyPreprocess
        dummy = DummyPreprocess(
            emb_keys=emb_keys,
            batch_size=batch_size,
            embedding_dim=SMALL_EMBEDDING_DIM,
            static_uih_len=STATIC_UIH_LEN,
            static_num_candidates=STATIC_NUM_CANDIDATES,
            user_emb_feature_names=hstu_config.user_embedding_feature_names,
            item_emb_feature_names=hstu_config.item_embedding_feature_names,
            hstu_configs=hstu_config,
        )
        dummy_out = dummy(uih_values, uih_lengths, cand_values, cand_lengths)
        (
            dummy_emb_lengths,
            dummy_emb_values,
            dummy_payload_features,
            _dummy_max_uih_len,
            dummy_uih_seq_lengths,
            _dummy_max_num_candidates,
            dummy_num_candidates,
            _dummy_total_targets,
            _dummy_total_uih_len,
        ) = dummy_out
        # Compare emb_lengths shapes
        self.assertEqual(
            len(real_emb_lengths),
            len(dummy_emb_lengths),
            f"Number of emb_lengths should match: real={len(real_emb_lengths)} dummy={len(dummy_emb_lengths)}",
        )
        for i, (real_l, dummy_l) in enumerate(zip(real_emb_lengths, dummy_emb_lengths)):
            self.assertEqual(
                real_l.shape,
                dummy_l.shape,
                f"emb_lengths[{i}] ({emb_keys[i]}): real={real_l.shape} dummy={dummy_l.shape}",
            )
        # Compare emb_values shapes — THIS IS THE CRITICAL CHECK
        self.assertEqual(
            len(real_emb_values),
            len(dummy_emb_values),
            f"Number of emb_values should match: real={len(real_emb_values)} dummy={len(dummy_emb_values)}",
        )
        for i, (real_v, dummy_v) in enumerate(zip(real_emb_values, dummy_emb_values)):
            self.assertEqual(
                real_v.shape,
                dummy_v.shape,
                f"emb_values[{i}] ({emb_keys[i]}): real={real_v.shape} dummy={dummy_v.shape}",
            )
        # Compare uih_seq_lengths shape
        self.assertEqual(
            real_uih_seq_lengths.shape,
            dummy_uih_seq_lengths.shape,
            f"uih_seq_lengths: real={real_uih_seq_lengths.shape} dummy={dummy_uih_seq_lengths.shape}",
        )
        # Compare num_candidates shape
        self.assertEqual(
            real_num_candidates.shape,
            dummy_num_candidates.shape,
            f"num_candidates: real={real_num_candidates.shape} dummy={dummy_num_candidates.shape}",
        )
        # Compare payload_features shapes
        for key in real_payload_features:
            self.assertIn(
                key, dummy_payload_features, f"Missing payload feature: {key}"
            )
            self.assertEqual(
                real_payload_features[key].shape,
                dummy_payload_features[key].shape,
                f"payload_features[{key}]: real={real_payload_features[key].shape} dummy={dummy_payload_features[key].shape}",
            )
