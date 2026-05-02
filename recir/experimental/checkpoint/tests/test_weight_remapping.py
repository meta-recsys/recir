# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import unittest

import torch
from recir.experimental.checkpoint import (
    remap_backbone_state_dict,
    remap_click_head_state_dict,
)


def _t() -> torch.Tensor:
    return torch.zeros(1)


class RemapBackboneStateDictTest(unittest.TestCase):
    def test_strips_default_prefix(self) -> None:
        sd = {"model.layers.0.weight": _t(), "model.embed.weight": _t()}
        out = remap_backbone_state_dict(sd)
        self.assertEqual(set(out.keys()), {"layers.0.weight", "embed.weight"})

    def test_passthrough_when_prefix_absent(self) -> None:
        sd = {"layers.0.weight": _t()}
        out = remap_backbone_state_dict(sd)
        self.assertEqual(set(out.keys()), {"layers.0.weight"})

    def test_custom_prefix(self) -> None:
        sd = {"engram.foo": _t(), "bar": _t()}
        out = remap_backbone_state_dict(sd, prefix="engram.")
        self.assertEqual(set(out.keys()), {"foo", "bar"})


class RemapClickHeadStateDictTest(unittest.TestCase):
    def test_projector_net_renames_to_proj(self) -> None:
        sd = {"projector.net.0.weight": _t(), "projector.net.0.bias": _t()}
        out = remap_click_head_state_dict(sd)
        self.assertEqual(set(out.keys()), {"proj.0.weight", "proj.0.bias"})

    def test_mlp_net_renames_to_mlp(self) -> None:
        sd = {"mlp.net.0.weight": _t(), "mlp.net.3.bias": _t()}
        out = remap_click_head_state_dict(sd)
        self.assertEqual(set(out.keys()), {"mlp.0.weight", "mlp.3.bias"})

    def test_skip_a_b_renames(self) -> None:
        sd = {"skip.a": _t(), "skip.b": _t()}
        out = remap_click_head_state_dict(sd)
        self.assertEqual(set(out.keys()), {"skip_a", "skip_b"})

    def test_dense_fusion_inserts_mlp(self) -> None:
        sd = {
            "dense_fusion.0.weight": _t(),
            "dense_fusion.0.bias": _t(),
            "dense_fusion.3.weight": _t(),
        }
        out = remap_click_head_state_dict(sd)
        self.assertEqual(
            set(out.keys()),
            {
                "dense_fusion.mlp.0.weight",
                "dense_fusion.mlp.0.bias",
                "dense_fusion.mlp.3.weight",
            },
        )

    def test_dense_fusion_idempotent(self) -> None:
        sd = {"dense_fusion.mlp.0.weight": _t(), "dense_fusion.mlp.0.bias": _t()}
        out = remap_click_head_state_dict(sd)
        self.assertEqual(
            set(out.keys()),
            {"dense_fusion.mlp.0.weight", "dense_fusion.mlp.0.bias"},
        )

    def test_passthrough_for_already_target_keys(self) -> None:
        sd = {"proj.0.weight": _t(), "mlp.0.weight": _t(), "skip_a": _t()}
        out = remap_click_head_state_dict(sd)
        self.assertEqual(set(out.keys()), {"proj.0.weight", "mlp.0.weight", "skip_a"})

    def test_unrelated_keys_untouched(self) -> None:
        sd = {"some_other.weight": _t(), "config.value": _t()}
        out = remap_click_head_state_dict(sd)
        self.assertEqual(set(out.keys()), {"some_other.weight", "config.value"})

    def test_mixed_input_all_rules(self) -> None:
        sd = {
            "projector.net.0.weight": _t(),
            "mlp.net.3.bias": _t(),
            "skip.a": _t(),
            "skip.b": _t(),
            "dense_fusion.0.weight": _t(),
            "dense_fusion.mlp.3.bias": _t(),
            "passthrough.foo": _t(),
        }
        out = remap_click_head_state_dict(sd)
        self.assertEqual(
            set(out.keys()),
            {
                "proj.0.weight",
                "mlp.3.bias",
                "skip_a",
                "skip_b",
                "dense_fusion.mlp.0.weight",
                "dense_fusion.mlp.3.bias",
                "passthrough.foo",
            },
        )

    def test_values_preserved(self) -> None:
        v_proj = torch.tensor([1.0, 2.0])
        v_dense = torch.tensor([3.0])
        sd = {"projector.net.0.weight": v_proj, "dense_fusion.0.weight": v_dense}
        out = remap_click_head_state_dict(sd)
        self.assertTrue(torch.equal(out["proj.0.weight"], v_proj))
        self.assertTrue(torch.equal(out["dense_fusion.mlp.0.weight"], v_dense))
