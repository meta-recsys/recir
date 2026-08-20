# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Unit tests for the recir QAT pass.

CPU-only; no distributed setup needed.  ``QatTraceTests`` uses
``functorch.make_fx`` + ``parametrize_params`` directly (instead of the
higher-level ``recir.trace``) to keep this test target free of the
``recir.torchax`` import chain.

Run: ``buck2 test //recir/tests:test_qat``
"""

import math
import unittest
from typing import Any

import torch
import torch.nn as nn
from functorch import make_fx
from recir.distributed.data_parallel import parametrize_params
from recir.quant import (
    fake_quantize_ste,
    finalize_qat_calibration,
    install_qat,
    qat_fake_quant,
    QATConfig,
    set_qat_calibration,
)
from recir.quant.config import INIT_CLIP_AUTO
from torch.nn.utils import stateless


def _make_toy_ec(table_names: list[str], vocab: int = 10, dim: int = 4) -> "_ToyEC":
    """Return a tiny ``TracableEmbeddingCollection``-shaped module: a
    ``ModuleDict`` of ``nn.Embedding``s keyed by table name."""
    return _ToyEC(table_names, vocab, dim)


class _ToyEC(nn.Module):
    """Test-only ``TracableEmbeddingCollection``-shaped module. Typed at
    module scope so pyre can resolve ``mod.embeddings["..."]`` and the
    dynamic ``qat_scale``/``qat_clip``/``qat_zero_point`` attributes that
    ``install_qat`` adds are still accessible via the un-suppressed getter.
    """

    embeddings: nn.ModuleDict

    def __init__(self, table_names: list[str], vocab: int, dim: int) -> None:
        super().__init__()
        self.embeddings = nn.ModuleDict()
        for name in table_names:
            self.embeddings[name] = nn.Embedding(vocab, dim)

    def forward(self, ids: dict[str, torch.Tensor]) -> torch.Tensor:
        out: torch.Tensor | None = None
        for name, emb in self.embeddings.items():
            x = emb(ids[name]).float()
            out = x if out is None else out + x
        assert out is not None
        return out


def _get_emb(mod: "_ToyEC", name: str) -> Any:
    """Typed accessor: ``mod.embeddings[name]`` returns ``nn.Module`` per
    pyre, and the ``qat_*`` attributes attached by ``install_qat`` are
    dynamic (pyre types them as the ``Module | Tensor`` union produced by
    ``nn.Module.__getattr__``). Return ``Any`` so test call-sites like
    ``emb.qat_scale.fill_(0)`` don't drown in pyre noise.
    """
    emb = mod.embeddings[name]
    assert isinstance(emb, nn.Embedding)
    return emb


def _sgd(params: Any, lr: float = 0.1) -> torch.optim.Optimizer:
    """Wrap ``torch.optim.SGD`` with ``foreach=True`` (Citrine C2:
    multi-tensor SGD is ~3% faster; our toy embeddings are dense so
    ``foreach`` is safe)."""
    # Citrine C2: foreach=True for multi-tensor multi-optimizer execution.
    return torch.optim.SGD(params, lr=lr, foreach=True)


class STETest(unittest.TestCase):
    """Numeric checks on the STE autograd Function in isolation."""

    def test_ste_forward_identity_within_range(self) -> None:
        # Values inside the clip range should quantize close to themselves.
        x = torch.tensor([0.0, 1.0, -2.5, 3.7])
        scale = torch.tensor(1.0)
        clip = torch.tensor(6.0)
        y = fake_quantize_ste(x, scale, clip, n_bits=8, symmetric=True)
        # 8-bit symmetric with clip=6 gives step=6/127 ≈ 0.047; quantization
        # error should be <= step/2 ≈ 0.024.
        self.assertTrue(torch.all((y - x).abs() < 0.05))

    def test_ste_forward_clamps_outside_range(self) -> None:
        x = torch.tensor([10.0, -100.0])
        scale = torch.tensor(1.0)
        clip = torch.tensor(6.0)
        y = fake_quantize_ste(x, scale, clip, n_bits=8, symmetric=True)
        # Values outside clip range are pinned to ±clip * scale.
        self.assertAlmostEqual(y[0].item(), 6.0, places=1)
        self.assertAlmostEqual(y[1].item(), -6.0, places=1)

    def test_ste_backward_passes_grad_in_range(self) -> None:
        # Use non-zero-sum x so scale.grad has signal (the STE grad_scale
        # formula sums x*grad_out; symmetric x cancels).
        x = torch.tensor([1.0, 2.0, 4.0], requires_grad=True)
        scale = torch.tensor(1.0, requires_grad=True)
        clip = torch.tensor(6.0, requires_grad=True)
        y = fake_quantize_ste(x, scale, clip)
        y.sum().backward()
        # In-range grads should all be 1 (identity STE).
        assert x.grad is not None
        self.assertTrue(torch.allclose(x.grad, torch.ones_like(x)))
        # scale/clip should get non-zero grads.
        assert scale.grad is not None and clip.grad is not None
        self.assertFalse(torch.isclose(scale.grad, torch.zeros(())))

    def test_ste_backward_zeros_grad_outside_range(self) -> None:
        x = torch.tensor([10.0, -100.0, 0.5], requires_grad=True)
        scale = torch.tensor(1.0, requires_grad=True)
        clip = torch.tensor(6.0, requires_grad=True)
        y = fake_quantize_ste(x, scale, clip)
        y.sum().backward()
        assert x.grad is not None
        # Outside-range positions should have grad 0; inside-range grad 1.
        expected = torch.tensor([0.0, 0.0, 1.0])
        self.assertTrue(torch.allclose(x.grad, expected))

    def test_ste_per_row_shapes(self) -> None:
        vocab, dim = 5, 3
        weight = torch.randn(vocab, dim, requires_grad=True)
        scale = torch.full((vocab, 1), 1.0, requires_grad=True)
        clip = torch.full((vocab, 1), 6.0, requires_grad=True)
        y = fake_quantize_ste(weight, scale, clip, granularity="per_row")
        self.assertEqual(y.shape, weight.shape)
        y.sum().backward()
        assert weight.grad is not None
        assert scale.grad is not None
        assert clip.grad is not None
        self.assertEqual(scale.grad.shape, (vocab, 1))
        self.assertEqual(clip.grad.shape, (vocab, 1))

    def test_ste_asymmetric_forward_range(self) -> None:
        # Asymmetric uint8 with zp=64 shifts the representable range so it
        # covers [-64*Δ, 191*Δ] (unequal negative/positive extents).
        x = torch.linspace(-10.0, 10.0, 21)
        scale = torch.tensor(1.0)
        clip = torch.tensor(8.0)  # -> Δ = 8/255 ≈ 0.031
        zp = torch.tensor(64.0)
        y = fake_quantize_ste(x, scale, clip, n_bits=8, symmetric=False, zero_point=zp)
        delta = 8.0 / 255.0
        min_repr = -64.0 * delta  # ≈ -2.01
        max_repr = (255.0 - 64.0) * delta  # ≈ 5.99
        # Values below min_repr should be pinned; above max_repr also pinned.
        self.assertGreaterEqual(y.min().item(), min_repr - 1e-4)
        self.assertLessEqual(y.max().item(), max_repr + 1e-4)

    def test_ste_asymmetric_backward_zp_grad_nonzero(self) -> None:
        # zp should receive non-zero grad only when some elements saturate.
        # LSQ zp grad = -Δ at saturation, 0 inside.
        x = torch.tensor([10.0, -50.0, 0.1], requires_grad=True)  # saturates both ends
        scale = torch.tensor(1.0, requires_grad=True)
        clip = torch.tensor(4.0, requires_grad=True)
        zp = torch.tensor(64.0, requires_grad=True)
        y = fake_quantize_ste(x, scale, clip, n_bits=8, symmetric=False, zero_point=zp)
        y.sum().backward()
        assert zp.grad is not None
        self.assertFalse(
            torch.isclose(zp.grad, torch.zeros(())),
            "zp should get non-zero grad when saturation occurs",
        )

    def test_ste_grad_factor_scales_scale_and_clip_grads(self) -> None:
        # Explicit grad_factor should linearly scale the qat-param grads
        # (but not grad_x, which is pass-through STE).
        x = torch.tensor([1.0, 2.0, 4.0], requires_grad=True)
        scale = torch.tensor(1.0, requires_grad=True)
        clip = torch.tensor(6.0, requires_grad=True)
        y = fake_quantize_ste(x, scale, clip, grad_factor=1.0)
        y.sum().backward()
        assert scale.grad is not None and clip.grad is not None and x.grad is not None
        grad_scale_1 = scale.grad.detach().clone()
        grad_clip_1 = clip.grad.detach().clone()
        grad_x_1 = x.grad.detach().clone()

        # Second pass with grad_factor=0.25.
        x.grad = None
        scale.grad = None
        clip.grad = None
        y = fake_quantize_ste(x, scale, clip, grad_factor=0.25)
        y.sum().backward()
        assert scale.grad is not None and clip.grad is not None and x.grad is not None
        self.assertTrue(torch.allclose(scale.grad, grad_scale_1 * 0.25))
        self.assertTrue(torch.allclose(clip.grad, grad_clip_1 * 0.25))
        # grad_x is not scaled by grad_factor -- LSQ only scales scale/clip/zp.
        self.assertTrue(torch.allclose(x.grad, grad_x_1))

    def test_ste_per_group_forward_and_backward_shapes(self) -> None:
        # per_group: weight [V, D], scale/clip shape [V, G] where G=D/group_size.
        vocab, dim, group_size = 4, 8, 4
        num_groups = dim // group_size
        weight = torch.randn(vocab, dim, requires_grad=True)
        scale = torch.full((vocab, num_groups), 1.0, requires_grad=True)
        clip = torch.full((vocab, num_groups), 6.0, requires_grad=True)
        y = fake_quantize_ste(
            weight,
            scale,
            clip,
            granularity="per_group",
            group_size=group_size,
        )
        # Output preserves [V, D] shape.
        self.assertEqual(y.shape, (vocab, dim))
        y.sum().backward()
        assert weight.grad is not None
        assert scale.grad is not None
        assert clip.grad is not None
        self.assertEqual(weight.grad.shape, (vocab, dim))
        self.assertEqual(scale.grad.shape, (vocab, num_groups))
        self.assertEqual(clip.grad.shape, (vocab, num_groups))

    def test_ste_per_group_different_group_clips_produce_different_quants(
        self,
    ) -> None:
        # Two groups with very different clip ranges should quantize their
        # slice of the weight differently. Shape: 1 row, 4 cols, 2 groups.
        group_size = 2
        weight = torch.tensor([[2.0, 4.0, 2.0, 4.0]], requires_grad=False)
        # Group 0: tight clip -> heavy saturation. Group 1: clip=127 gives
        # Δ=1.0 on 8-bit so integer weights land exactly on the grid.
        scale = torch.tensor([[1.0, 1.0]])
        clip = torch.tensor([[0.5, 127.0]])
        y = fake_quantize_ste(
            weight, scale, clip, granularity="per_group", group_size=group_size
        )
        # Group 0 saturates (|2| and |4| both >> 0.5) -> both pinned to ±0.5.
        self.assertAlmostEqual(y[0, 0].item(), 0.5, places=2)
        self.assertAlmostEqual(y[0, 1].item(), 0.5, places=2)
        # Group 1 with clip=127, Δ=1.0 -> exact quantization of 2.0 and 4.0.
        self.assertAlmostEqual(y[0, 2].item(), 2.0, places=5)
        self.assertAlmostEqual(y[0, 3].item(), 4.0, places=5)

    def test_ste_stochastic_rounding_is_unbiased(self) -> None:
        # E[SR(x)] == x. Choose clip=127 so Δ=1.0 on 8-bit -> x=3.3 stochastically
        # rounds to 3 (Pr=0.7) or 4 (Pr=0.3); the mean over many samples must
        # land close to 3.3.
        torch.manual_seed(0)
        x = torch.full((5000,), 3.3)
        scale = torch.tensor(1.0)
        clip = torch.tensor(127.0)
        y = fake_quantize_ste(x, scale, clip, round_mode="stochastic")
        # With 5000 samples the std of the mean is sqrt(0.21/5000) ≈ 0.0065,
        # so ±0.03 is safe.
        self.assertAlmostEqual(y.mean().item(), 3.3, places=1)

    def test_ste_stochastic_rounding_differs_across_calls(self) -> None:
        # Two identical calls should NOT produce identical outputs (with
        # overwhelming probability) because torch.rand_like is called anew.
        torch.manual_seed(42)
        x = torch.linspace(-3.0, 3.0, 100)
        scale = torch.tensor(1.0)
        clip = torch.tensor(4.0)
        y1 = fake_quantize_ste(x, scale, clip, round_mode="stochastic")
        y2 = fake_quantize_ste(x, scale, clip, round_mode="stochastic")
        self.assertFalse(torch.equal(y1, y2))

    def test_ste_nearest_rounding_is_deterministic(self) -> None:
        # Nearest mode: same inputs -> exact same outputs.
        x = torch.linspace(-3.0, 3.0, 100)
        scale = torch.tensor(1.0)
        clip = torch.tensor(4.0)
        y1 = fake_quantize_ste(x, scale, clip, round_mode="nearest")
        y2 = fake_quantize_ste(x, scale, clip, round_mode="nearest")
        self.assertTrue(torch.equal(y1, y2))


class InstallQatTest(unittest.TestCase):
    """Tests for the eager_preproc hook."""

    def test_install_qat_adds_params(self) -> None:
        mod = _make_toy_ec(["tbl_a", "tbl_b"])
        opt = _sgd(list(mod.parameters()), lr=0.1)
        cfg = QATConfig(target_table_names={"tbl_a"}, n_bits=8)
        install_qat(cfg, opt)(mod)
        # tbl_a gets qat_scale/qat_clip; tbl_b does not.
        emb_a = _get_emb(mod, "tbl_a")
        emb_b = _get_emb(mod, "tbl_b")
        self.assertTrue(hasattr(emb_a, "qat_scale"))
        self.assertTrue(hasattr(emb_a, "qat_clip"))
        self.assertFalse(hasattr(emb_b, "qat_scale"))
        self.assertFalse(hasattr(emb_b, "qat_clip"))
        # Params landed in the optimizer.
        opt_param_ids = {id(p) for g in opt.param_groups for p in g["params"]}
        self.assertIn(id(emb_a.qat_scale), opt_param_ids)
        self.assertIn(id(emb_a.qat_clip), opt_param_ids)

    def test_install_qat_is_idempotent(self) -> None:
        mod = _make_toy_ec(["tbl_a"])
        opt = _sgd(list(mod.parameters()), lr=0.1)
        cfg = QATConfig(target_table_names={"tbl_a"})
        hook = install_qat(cfg, opt)
        hook(mod)
        n_groups_after_first = len(opt.param_groups)
        # Second call should be a no-op (idempotency).
        hook(mod)
        self.assertEqual(len(opt.param_groups), n_groups_after_first)

    def test_install_qat_per_row_shapes(self) -> None:
        vocab, dim = 7, 4
        mod = _make_toy_ec(["tbl_a"], vocab=vocab, dim=dim)
        opt = _sgd(list(mod.parameters()), lr=0.1)
        cfg = QATConfig(target_table_names={"tbl_a"}, granularity="per_row")
        install_qat(cfg, opt)(mod)
        emb = _get_emb(mod, "tbl_a")
        self.assertEqual(emb.qat_scale.shape, (vocab, 1))
        self.assertEqual(emb.qat_clip.shape, (vocab, 1))

    def test_install_qat_auto_init_clip_matches_lsq_formula(self) -> None:
        # init_clip = 2 * mean(|w|) * sqrt(q_max) per LSQ paper.
        # int8 symmetric: q_max = 127.
        vocab, dim = 8, 4
        mod = _make_toy_ec(["tbl_a"], vocab=vocab, dim=dim)
        emb = _get_emb(mod, "tbl_a")
        # Deterministic weights so init_clip is predictable.
        with torch.no_grad():
            emb.weight.copy_(torch.linspace(-1.0, 1.0, vocab * dim).reshape(vocab, dim))
        mean_abs = emb.weight.abs().mean().item()
        expected = 2.0 * mean_abs * math.sqrt(127)

        opt = _sgd(list(mod.parameters()), lr=0.1)
        cfg = QATConfig(target_table_names={"tbl_a"})  # init_clip default = "auto"
        self.assertEqual(cfg.init_clip, INIT_CLIP_AUTO)
        install_qat(cfg, opt)(mod)
        self.assertAlmostEqual(emb.qat_clip.item(), expected, places=4)

    def test_install_qat_auto_init_clip_per_row(self) -> None:
        vocab, dim = 5, 4
        mod = _make_toy_ec(["tbl_a"], vocab=vocab, dim=dim)
        emb = _get_emb(mod, "tbl_a")
        with torch.no_grad():
            # Row 0 has small values, row 4 has large values -> different clip.
            for i in range(vocab):
                emb.weight[i].fill_(float(i + 1) * 0.1)
        opt = _sgd(list(mod.parameters()), lr=0.1)
        cfg = QATConfig(target_table_names={"tbl_a"}, granularity="per_row")
        install_qat(cfg, opt)(mod)
        # Row 4's clip should be strictly larger than row 0's.
        self.assertGreater(emb.qat_clip[4, 0].item(), emb.qat_clip[0, 0].item())

    def test_install_qat_explicit_init_clip_overrides_auto(self) -> None:
        vocab, dim = 4, 3
        mod = _make_toy_ec(["tbl_a"], vocab=vocab, dim=dim)
        opt = _sgd(list(mod.parameters()), lr=0.1)
        cfg = QATConfig(target_table_names={"tbl_a"}, init_clip=6.0)
        install_qat(cfg, opt)(mod)
        emb = _get_emb(mod, "tbl_a")
        self.assertAlmostEqual(emb.qat_clip.item(), 6.0, places=5)

    def test_install_qat_asymmetric_registers_zero_point(self) -> None:
        mod = _make_toy_ec(["tbl_a"])
        opt = _sgd(list(mod.parameters()), lr=0.1)
        cfg = QATConfig(target_table_names={"tbl_a"}, symmetric=False, n_bits=8)
        install_qat(cfg, opt)(mod)
        emb = _get_emb(mod, "tbl_a")
        self.assertTrue(hasattr(emb, "qat_zero_point"))
        # uint8 default zp is q_max // 2 + 1 = 128.
        self.assertAlmostEqual(emb.qat_zero_point.item(), 128.0, places=5)
        # zp lands in the optimizer.
        opt_ids = {id(p) for g in opt.param_groups for p in g["params"]}
        self.assertIn(id(emb.qat_zero_point), opt_ids)

    def test_install_qat_symmetric_omits_zero_point(self) -> None:
        mod = _make_toy_ec(["tbl_a"])
        opt = _sgd(list(mod.parameters()), lr=0.1)
        cfg = QATConfig(target_table_names={"tbl_a"}, symmetric=True)
        install_qat(cfg, opt)(mod)
        emb = _get_emb(mod, "tbl_a")
        self.assertFalse(hasattr(emb, "qat_zero_point"))

    def test_install_qat_per_group_shapes(self) -> None:
        # per_group with group_size=4 on a dim=8 table -> 2 groups per row.
        vocab, dim, group_size = 5, 8, 4
        num_groups = dim // group_size
        mod = _make_toy_ec(["tbl_a"], vocab=vocab, dim=dim)
        opt = _sgd(list(mod.parameters()), lr=0.1)
        cfg = QATConfig(
            target_table_names={"tbl_a"},
            granularity="per_group",
            group_size=group_size,
        )
        install_qat(cfg, opt)(mod)
        emb = _get_emb(mod, "tbl_a")
        self.assertEqual(emb.qat_scale.shape, (vocab, num_groups))
        self.assertEqual(emb.qat_clip.shape, (vocab, num_groups))

    def test_install_qat_per_group_auto_init_clip_uses_per_group_mean(
        self,
    ) -> None:
        vocab, dim, group_size = 3, 4, 2  # 2 groups per row
        mod = _make_toy_ec(["tbl_a"], vocab=vocab, dim=dim)
        emb = _get_emb(mod, "tbl_a")
        with torch.no_grad():
            # Row 0: group 0 all 0.1, group 1 all 1.0 -> group 1 clip 10x larger.
            emb.weight.copy_(
                torch.tensor(
                    [
                        [0.1, 0.1, 1.0, 1.0],
                        [0.1, 0.1, 1.0, 1.0],
                        [0.1, 0.1, 1.0, 1.0],
                    ]
                )
            )
        opt = _sgd(list(mod.parameters()), lr=0.1)
        cfg = QATConfig(
            target_table_names={"tbl_a"},
            granularity="per_group",
            group_size=group_size,
        )
        install_qat(cfg, opt)(mod)
        # int8 symmetric: q_max = 127.
        expected_g0 = 2.0 * 0.1 * math.sqrt(127)
        expected_g1 = 2.0 * 1.0 * math.sqrt(127)
        self.assertAlmostEqual(emb.qat_clip[0, 0].item(), expected_g0, places=4)
        self.assertAlmostEqual(emb.qat_clip[0, 1].item(), expected_g1, places=4)

    def test_install_qat_per_group_rejects_bad_group_size(self) -> None:
        # group_size that doesn't divide the embedding dim should raise.
        vocab, dim = 4, 6
        mod = _make_toy_ec(["tbl_a"], vocab=vocab, dim=dim)
        opt = _sgd(list(mod.parameters()), lr=0.1)
        cfg = QATConfig(
            target_table_names={"tbl_a"},
            granularity="per_group",
            group_size=4,  # 4 does not divide 6
        )
        with self.assertRaises(AssertionError):
            install_qat(cfg, opt)(mod)

    def test_install_qat_separate_qat_optimizer(self) -> None:
        # qat_optimizer=... routes new params to a separate optimizer,
        # leaving the primary optimizer untouched. Matches the Horizon/Ads
        # split-optimizer pattern.
        mod = _make_toy_ec(["tbl_a"])
        primary_opt = _sgd(list(mod.parameters()), lr=0.1)
        qat_opt = torch.optim.Adam(
            [torch.zeros(1)], lr=1e-3, foreach=True
        )  # seed placeholder; Citrine C2: foreach=True for multi-tensor.
        primary_ids_before = {
            id(p) for g in primary_opt.param_groups for p in g["params"]
        }
        qat_ids_before = {id(p) for g in qat_opt.param_groups for p in g["params"]}

        cfg = QATConfig(target_table_names={"tbl_a"}, n_bits=8)
        install_qat(cfg, primary_opt, qat_optimizer=qat_opt)(mod)

        emb = _get_emb(mod, "tbl_a")
        primary_ids_after = {
            id(p) for g in primary_opt.param_groups for p in g["params"]
        }
        qat_ids_after = {id(p) for g in qat_opt.param_groups for p in g["params"]}

        # Primary optimizer got no new params.
        self.assertEqual(primary_ids_before, primary_ids_after)
        # qat_scale/qat_clip landed in qat_optimizer.
        self.assertIn(id(emb.qat_scale), qat_ids_after)
        self.assertIn(id(emb.qat_clip), qat_ids_after)
        self.assertNotIn(id(emb.qat_scale), primary_ids_after)
        self.assertNotIn(id(emb.qat_clip), primary_ids_after)
        # qat_optimizer gained exactly the new params.
        self.assertEqual(
            qat_ids_after - qat_ids_before,
            {id(emb.qat_scale), id(emb.qat_clip)},
        )

    def test_install_qat_calibration_registers_buffers(self) -> None:
        # calibration_momentum=<float> should install running_max + active buffers.
        mod = _make_toy_ec(["tbl_a"])
        opt = _sgd(list(mod.parameters()), lr=0.1)
        cfg = QATConfig(
            target_table_names={"tbl_a"},
            n_bits=8,
            calibration_momentum=0.9,
        )
        install_qat(cfg, opt)(mod)
        emb = _get_emb(mod, "tbl_a")
        self.assertIn("qat_running_max", dict(emb.named_buffers()))
        self.assertIn("qat_calibration_active", dict(emb.named_buffers()))
        # Init: running_max seeded to LSQ clip; active=1.0 (calibration on).
        self.assertTrue(torch.allclose(emb.qat_running_max, emb.qat_clip.detach()))
        self.assertAlmostEqual(emb.qat_calibration_active.item(), 1.0, places=5)

    def test_install_qat_no_calibration_omits_buffers(self) -> None:
        # Without calibration_momentum, buffers are not registered (backwards
        # compat: no extra state to serialize).
        mod = _make_toy_ec(["tbl_a"])
        opt = _sgd(list(mod.parameters()), lr=0.1)
        cfg = QATConfig(target_table_names={"tbl_a"})
        install_qat(cfg, opt)(mod)
        emb = _get_emb(mod, "tbl_a")
        self.assertNotIn("qat_running_max", dict(emb.named_buffers()))
        self.assertNotIn("qat_calibration_active", dict(emb.named_buffers()))


class QatTraceTest(unittest.TestCase):
    """End-to-end trace with QAT parametrize fn active.

    Uses ``make_fx`` + ``parametrize_params`` directly instead of
    ``recir.trace`` so this test target does not need the full
    ``recir.api`` import chain (which currently pulls in the
    ``recir.torchax`` module, whose ``aten.qr`` reference is broken in
    some torch builds).
    """

    def _trace_qat(
        self,
        cfg: QATConfig,
        vocab: int = 10,
        dim: int = 4,
    ) -> tuple[Any, "_ToyEC", int]:
        mod = _make_toy_ec(
            list(cfg.target_table_names) or ["tbl_a"], vocab=vocab, dim=dim
        )
        opt = _sgd(list(mod.parameters()), lr=0.1)
        install_qat(cfg, opt)(mod)
        ids = {name: torch.randint(0, vocab, (3,)) for name in mod.embeddings.keys()}

        # Mirror recir._local_compile's stateless-func pattern: reparametrize
        # the module with proxy params, then run the fwd+bwd inside the
        # ``parametrize_params`` context so make_fx captures the STE.
        params = dict(mod.named_parameters(remove_duplicate=False))
        buffers = dict(mod.named_buffers(remove_duplicate=False))

        n_trainable = sum(1 for p in params.values() if p.requires_grad)

        def stateless_func(
            params: dict[str, torch.Tensor],
            buffers: dict[str, torch.Tensor],
            ids: dict[str, torch.Tensor],
        ) -> Any:
            with stateless._reparametrize_module(mod, {**params, **buffers}):
                with parametrize_params(mod, qat_fake_quant(cfg)):
                    out = mod(ids)
                    loss = out.sum()
                    trainable = [p for p in params.values() if p.requires_grad]
                    grads = torch.autograd.grad(loss, trainable, allow_unused=True)
                return loss, grads

        gm = make_fx(stateless_func, tracing_mode="fake")(params, buffers, ids)
        return gm, mod, n_trainable

    def _count_placeholders(self, gm: Any) -> int:
        return sum(1 for n in gm.graph.nodes if n.op == "placeholder")

    def _has_op_target(self, gm: Any, target_needle: str) -> bool:
        for n in gm.graph.nodes:
            tgt = getattr(n, "target", None)
            tgt_name = tgt.__name__ if hasattr(tgt, "__name__") else str(tgt)
            if target_needle in tgt_name:
                return True
        return False

    def test_qat_captures_fwd_and_bwd(self) -> None:
        cfg = QATConfig(target_table_names={"tbl_a"}, n_bits=8)
        gm, _mod, n_trainable = self._trace_qat(cfg)
        # weight + qat_scale + qat_clip = 3 trainable params for symmetric.
        self.assertEqual(n_trainable, 3)
        # STE.forward decomposes into aten.round + aten.clamp; both must
        # appear in the traced joint graph (confirms fwd fake-quant captured).
        self.assertTrue(self._has_op_target(gm, "round"))
        self.assertTrue(self._has_op_target(gm, "clamp"))

    def test_qat_asymmetric_captures_zero_point_grad(self) -> None:
        cfg = QATConfig(target_table_names={"tbl_a"}, n_bits=8, symmetric=False)
        gm, _mod, n_trainable = self._trace_qat(cfg)
        # weight + qat_scale + qat_clip + qat_zero_point = 4 trainable params.
        self.assertEqual(n_trainable, 4)
        self.assertTrue(self._has_op_target(gm, "round"))
        self.assertTrue(self._has_op_target(gm, "clamp"))

    def test_qat_per_group_traces_cleanly(self) -> None:
        # Weight [V=4, D=8], group_size=4 -> 2 groups per row.
        cfg = QATConfig(
            target_table_names={"tbl_a"},
            n_bits=8,
            granularity="per_group",
            group_size=4,
        )
        gm, _mod, n_trainable = self._trace_qat(cfg, vocab=4, dim=8)
        # weight + qat_scale + qat_clip = 3 trainable params.
        self.assertEqual(n_trainable, 3)
        # STE fwd nodes.
        self.assertTrue(self._has_op_target(gm, "round"))
        self.assertTrue(self._has_op_target(gm, "clamp"))
        # per_group inserts an aten.view/reshape to fold columns into groups.
        self.assertTrue(
            self._has_op_target(gm, "view") or self._has_op_target(gm, "reshape"),
            "per_group should emit an aten.view/reshape op",
        )

    def test_qat_stochastic_rounding_captures_rand_like(self) -> None:
        # Stochastic mode should trace to an aten.rand op, so replay draws
        # fresh randomness each step.
        cfg = QATConfig(target_table_names={"tbl_a"}, n_bits=8, round_mode="stochastic")
        gm, _mod, _ = self._trace_qat(cfg)
        self.assertTrue(
            self._has_op_target(gm, "rand"),
            "stochastic round should emit an aten.rand* op in the traced gm",
        )

    def test_qat_per_row_captures(self) -> None:
        cfg = QATConfig(target_table_names={"tbl_a"}, n_bits=8, granularity="per_row")
        gm, mod, n_trainable = self._trace_qat(cfg, vocab=6, dim=3)
        self.assertEqual(n_trainable, 3)
        self.assertTrue(self._has_op_target(gm, "round"))
        emb = _get_emb(mod, "tbl_a")
        self.assertEqual(emb.qat_scale.shape, (6, 1))

    def test_qat_optimizer_updates_scale_and_clip(self) -> None:
        # Eager check: STE flowing through SGD.step should actually move
        # qat_scale and qat_clip. Uses grad_factor=1.0 (disables LSQ auto-
        # scaling) + explicit small init_clip so most weights saturate and
        # produce a clear signal, so 3 steps give a visible movement.
        mod = _make_toy_ec(["tbl_a"], vocab=8, dim=4)
        opt = _sgd(list(mod.parameters()), lr=0.1)
        cfg = QATConfig(
            target_table_names={"tbl_a"},
            n_bits=8,
            init_clip=0.05,
            grad_factor=1.0,
        )
        install_qat(cfg, opt)(mod)
        emb = _get_emb(mod, "tbl_a")
        scale_before = emb.qat_scale.detach().clone()
        clip_before = emb.qat_clip.detach().clone()

        for _ in range(3):
            ids = torch.randint(0, 8, (5,))
            # Amplify the weight so backward hits both in- and out-of-range.
            with torch.no_grad():
                emb.weight.mul_(2.0)
            fq = fake_quantize_ste(
                emb.weight,
                emb.qat_scale,
                emb.qat_clip,
                cfg.n_bits,
                cfg.symmetric,
                grad_factor=cfg.grad_factor,
            )
            out = torch.nn.functional.embedding(ids, fq)
            out.sum().backward()
            opt.step()
            opt.zero_grad()

        self.assertFalse(
            torch.allclose(emb.qat_scale.detach(), scale_before),
            "qat_scale should have moved after training steps",
        )
        self.assertFalse(
            torch.allclose(emb.qat_clip.detach(), clip_before),
            "qat_clip should have moved after training steps",
        )


class CalibrationTest(unittest.TestCase):
    """MA-min-max calibration warmup (Meta ADI 2020 recipe)."""

    def _tbl(self) -> tuple["_ToyEC", torch.optim.Optimizer, QATConfig]:
        mod = _make_toy_ec(["tbl_a"], vocab=6, dim=4)
        opt = _sgd(list(mod.parameters()), lr=0.1)
        cfg = QATConfig(
            target_table_names={"tbl_a"},
            n_bits=8,
            calibration_momentum=0.5,
        )
        install_qat(cfg, opt)(mod)
        return mod, opt, cfg

    def test_running_max_moves_toward_batch_stat_during_calibration(
        self,
    ) -> None:
        mod, _opt, cfg = self._tbl()
        emb = _get_emb(mod, "tbl_a")
        # Amplify weights so batch stat is much larger than initial running_max.
        with torch.no_grad():
            emb.weight.mul_(10.0)
        rm_before = emb.qat_running_max.detach().clone()
        # Run the parametrize fn once via a fwd through the embedding.
        fn = qat_fake_quant(cfg)
        # Access self.weight to trigger the property getter; without
        # parametrize_params installed, we invoke fn directly with the same
        # signature the getter would use.
        _ = fn(emb.weight, emb, "weight")
        rm_after = emb.qat_running_max.detach().clone()
        # EMA with momentum=0.5 and a much larger batch stat should push
        # running_max UP.
        self.assertGreater(rm_after.item(), rm_before.item())

    def test_effective_clip_switches_when_calibration_off(self) -> None:
        mod, _opt, cfg = self._tbl()
        emb = _get_emb(mod, "tbl_a")
        # Set very different clip vs running_max so we can distinguish.
        with torch.no_grad():
            emb.qat_clip.fill_(1.0)
            emb.qat_running_max.fill_(50.0)
        fn = qat_fake_quant(cfg)
        x = torch.tensor([25.0])  # in-range under clip=50, saturates under clip=1
        # Calibration ON: effective clip = 50 -> Δ = 50/127 ≈ 0.394,
        # x=25 -> round(63.5)=64, x_fq = 64 * 0.394 ≈ 25.20. Near 25.
        set_qat_calibration(mod, True)
        y_on = fn(x, emb, "weight")
        # Calibration OFF: effective clip = 1 -> 25 saturates to +1.
        set_qat_calibration(mod, False)
        y_off = fn(x, emb, "weight")
        self.assertAlmostEqual(y_on.item(), 25.0, delta=0.5)
        self.assertAlmostEqual(y_off.item(), 1.0, places=2)

    def test_finalize_copies_running_max_into_qat_clip(self) -> None:
        mod, _opt, _cfg = self._tbl()
        emb = _get_emb(mod, "tbl_a")
        with torch.no_grad():
            emb.qat_clip.fill_(1.0)
            emb.qat_running_max.fill_(3.14)
        finalize_qat_calibration(mod)
        self.assertAlmostEqual(emb.qat_clip.item(), 3.14, places=5)
        self.assertAlmostEqual(emb.qat_calibration_active.item(), 0.0, places=5)

    def test_calibration_off_does_not_move_running_max(self) -> None:
        mod, _opt, cfg = self._tbl()
        emb = _get_emb(mod, "tbl_a")
        set_qat_calibration(mod, False)
        with torch.no_grad():
            emb.weight.mul_(10.0)
        rm_before = emb.qat_running_max.detach().clone()
        fn = qat_fake_quant(cfg)
        _ = fn(emb.weight, emb, "weight")
        rm_after = emb.qat_running_max.detach().clone()
        # With active=0, the where-gate keeps running_max frozen.
        self.assertTrue(torch.allclose(rm_before, rm_after))

    def test_calibration_helpers_are_noops_without_calibration_cfg(
        self,
    ) -> None:
        # Without calibration_momentum, no buffers exist. Helpers must not raise
        # AND must not mutate the module (no buffers to touch, clip untouched).
        mod = _make_toy_ec(["tbl_a"])
        opt = _sgd(list(mod.parameters()), lr=0.1)
        cfg = QATConfig(target_table_names={"tbl_a"})
        install_qat(cfg, opt)(mod)
        emb = _get_emb(mod, "tbl_a")
        clip_before = emb.qat_clip.detach().clone()

        set_qat_calibration(mod, True)
        set_qat_calibration(mod, False)
        finalize_qat_calibration(mod)

        # Buffers should still be absent.
        buffer_names = dict(emb.named_buffers()).keys()
        self.assertNotIn("qat_running_max", buffer_names)
        self.assertNotIn("qat_calibration_active", buffer_names)
        # qat_clip untouched by the no-op finalize.
        self.assertTrue(torch.equal(emb.qat_clip.detach(), clip_before))
