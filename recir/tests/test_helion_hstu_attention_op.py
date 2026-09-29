# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""CPU tests for the recir Helion HSTU attention ops and their hammer adapter.

Covers the four things that can break silently:

1. The ``register_fake`` kernels still record the ``(L, H, D, B, N, dtype)`` warm-up
   signatures. This is the ONLY automated guard against the multi-chip failure
   mode: an empty warm set means Helion compiles lazily inside the jax lower tracer
   and dies with "SymIntArrayRef expected to contain only concrete integers" -- on
   TPU only, while every CPU test still passes.
2. A recir-style ``make_fx`` trace through the adapter collapses attention to
   exactly one fwd node plus one autograd-produced bwd node (no aten decomposition).
3. ``install_helion_hstu_attention_swap()`` rebinds the module attribute the v2 STU
   actually resolves. The wrong module is a silent perf cliff, not an error.
4. Eager numerics still go through the captured original ``pytorch_hstu_mha``
   byte-for-byte, so ``parity_forward_loss``'s eager cell stays a valid reference.
"""

from unittest import TestCase

import hammer.v2.ops.hstu_preprocess_and_attention as hpa
import recir.ops.helion_hstu_attention_op as hop
import torch
from functorch import make_fx
from recir.ops.helion.jagged_hstu_attention import (
    clear_traced_sigs,
    get_traced_bwd_sigs,
    get_traced_fwd_sigs,
)

# Small ragged batch: 3 sequences of 4/3/5 tokens packed on L, so N (the
# per-sequence pad width) is 5 and is distinct from L = 12.
_SEQ_LENGTHS: list[int] = [4, 3, 5]
_L: int = sum(_SEQ_LENGTHS)
_B: int = len(_SEQ_LENGTHS)
_N: int = max(_SEQ_LENGTHS)
_H: int = 2
_D: int = 4
_ALPHA: float = 0.125


def _seq_offsets() -> torch.Tensor:
    return torch.tensor([0, 4, 7, 12], dtype=torch.int64)


def _qkv(requires_grad: bool = False) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    return tuple(
        torch.randn(_L, _H, _D, dtype=torch.float32, requires_grad=requires_grad)
        for _ in range(3)
    )


def _adapter(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    **overrides: object,
) -> torch.Tensor:
    """Call the adapter the way the v2 STU does: all kwargs, hammer's names."""
    kwargs: dict[str, object] = {
        "N": _N,
        "alpha": _ALPHA,
        "q": q,
        "k": k,
        "v": v,
        "dropout_pr": 0.0,
        "training": False,
        "seq_offsets": seq_offsets,
        "invalid_attn_mask_type": "lower_triangular",
        "num_targets": None,
        "attn_scale": None,
        "attn_bias": None,
        "max_attn_len": 0,
        "full_attn_size": 0,
        "contextual_seq_len": 0,
    }
    kwargs.update(overrides)
    # pyre-ignore[6]: kwargs are typed by the adapter signature.
    return hop._recir_pytorch_hstu_mha(**kwargs)


def _original(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: torch.Tensor | None,
    attn_scale: torch.Tensor | None,
) -> torch.Tensor:
    """The captured original at the trivial-arg values the adapter enforces."""
    return hop._ORIG_PYTORCH_HSTU_MHA(
        q=q,
        k=k,
        v=v,
        alpha=_ALPHA,
        N=_N,
        seq_offsets=seq_offsets,
        invalid_attn_mask_type="lower_triangular",
        dropout_pr=0.0,
        training=False,
        num_targets=num_targets,
        attn_bias=None,
        attn_scale=attn_scale,
    )


class HelionHstuAttentionOpTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        clear_traced_sigs()
        self.addCleanup(clear_traced_sigs)

    # --- 1. traced warm-up signatures ------------------------------------
    def test_register_fake_records_the_fwd_and_bwd_warmup_sigs(self) -> None:
        """The fake kernels must record (L, H, D, B, N, dtype) for both directions.

        ``N`` is in the schema precisely so it can be recorded here: it is the 5th
        element of the warm-up key, and dropping it as "vestigial" changes the key,
        misses the ``BoundKernel`` cache, and recompiles inside the lower tracer.
        """
        q, k, v = _qkv(requires_grad=True)
        seq_offsets = _seq_offsets()

        def fn(
            q_: torch.Tensor, k_: torch.Tensor, v_: torch.Tensor, so_: torch.Tensor
        ) -> tuple[torch.Tensor, ...]:
            out = _adapter(q_, k_, v_, so_)
            return torch.autograd.grad(out.sum(), (q_, k_, v_))

        # Mirrors recir's own trace (recir/api.py): fake tensors, static shapes.
        make_fx(fn, tracing_mode="fake")(q, k, v, seq_offsets)

        expected = {(_L, _H, _D, _B, _N, torch.float32)}
        self.assertEqual(get_traced_fwd_sigs(), expected)
        self.assertEqual(get_traced_bwd_sigs(), expected)

    def test_clear_traced_sigs_empties_both_sets(self) -> None:
        q, k, v = _qkv(requires_grad=True)
        seq_offsets = _seq_offsets()

        def fn(
            q_: torch.Tensor, k_: torch.Tensor, v_: torch.Tensor, so_: torch.Tensor
        ) -> tuple[torch.Tensor, ...]:
            out = _adapter(q_, k_, v_, so_)
            return torch.autograd.grad(out.sum(), (q_, k_, v_))

        make_fx(fn, tracing_mode="fake")(q, k, v, seq_offsets)
        self.assertTrue(get_traced_fwd_sigs())
        self.assertTrue(get_traced_bwd_sigs())

        clear_traced_sigs()
        self.assertEqual(get_traced_fwd_sigs(), set())
        self.assertEqual(get_traced_bwd_sigs(), set())

    # --- 2. graph shape ---------------------------------------------------
    def test_trace_emits_exactly_one_fwd_and_one_bwd_node(self) -> None:
        """Attention must stay ONE opaque node, not ~15 decomposed aten ops."""
        q, k, v = _qkv(requires_grad=True)
        seq_offsets = _seq_offsets()

        def fn(
            q_: torch.Tensor, k_: torch.Tensor, v_: torch.Tensor, so_: torch.Tensor
        ) -> tuple[torch.Tensor, ...]:
            out = _adapter(q_, k_, v_, so_)
            return torch.autograd.grad(out.sum(), (q_, k_, v_))

        gm = make_fx(fn, tracing_mode="fake")(q, k, v, seq_offsets)
        targets = [str(n.target) for n in gm.graph.nodes if n.op == "call_function"]

        fwd = [t for t in targets if t.endswith("helion_ragged_attention_fwd.default")]
        bwd = [t for t in targets if t.endswith("helion_ragged_attention_bwd.default")]
        self.assertEqual(len(fwd), 1, f"expected 1 fwd node, got {targets}")
        self.assertEqual(len(bwd), 1, f"expected 1 bwd node, got {targets}")
        # aten.silu is the giveaway that the op decomposed instead of staying opaque.
        self.assertFalse(
            [t for t in targets if "silu" in t],
            f"attention decomposed into aten ops: {targets}",
        )

    # --- 3. the rebind ----------------------------------------------------
    def test_install_rebinds_the_module_the_v2_stu_resolves(self) -> None:
        """The STU reads ``pytorch_hstu_mha`` off hstu_preprocess_and_attention.

        Patching any other module (e.g. the origin
        ``hammer.ops.pytorch.ragged_hstu_attention``) leaves the STU on the eager
        path: no error, just a silent decomposition and a perf cliff.
        """
        original = hpa.pytorch_hstu_mha
        self.addCleanup(setattr, hpa, "pytorch_hstu_mha", original)

        hop.install_helion_hstu_attention_swap()
        self.assertIs(hpa.pytorch_hstu_mha, hop._recir_pytorch_hstu_mha)

        # Idempotent, and never the raw OpOverloadPacket (which would not accept
        # hammer's 16 kwargs).
        hop.install_helion_hstu_attention_swap()
        self.assertIs(hpa.pytorch_hstu_mha, hop._recir_pytorch_hstu_mha)

    # --- 4. adapter validation -------------------------------------------
    def test_adapter_rejects_every_arg_the_kernels_do_not_implement(self) -> None:
        q, k, v = _qkv()
        so = _seq_offsets()
        for override, pattern in (
            ({"invalid_attn_mask_type": "upper_triangular"}, "invalid_attn_mask_type"),
            ({"invalid_attn_mask_type": "none"}, "invalid_attn_mask_type"),
            ({"attn_bias": torch.zeros(_B, 1, _N, _N)}, "attn_bias"),
            ({"dropout_pr": 0.1}, "dropout_pr"),
            ({"max_attn_len": 3}, "max_attn_len"),
            ({"contextual_seq_len": 2}, "contextual_seq_len"),
            ({"full_attn_size": 8}, "full_attn_size"),
            ({"num_softmax_heads": 1}, "num_softmax_heads"),
        ):
            with self.subTest(override=override):
                with self.assertRaisesRegex(ValueError, pattern):
                    _adapter(q, k, v, so, **override)

    def test_adapter_accepts_the_udd_fm_configuration(self) -> None:
        q, k, v = _qkv()
        so = _seq_offsets()
        out = _adapter(
            q,
            k,
            v,
            so,
            num_targets=torch.tensor([1, 0, 2], dtype=torch.int64),
            attn_scale=torch.tensor(0.25),
        )
        self.assertEqual(out.shape, (_L, _H, _D))

    # --- 5. eager numerics ------------------------------------------------
    def test_eager_forward_is_byte_identical_to_the_captured_original(self) -> None:
        q, k, v = _qkv()
        so = _seq_offsets()
        num_targets = torch.tensor([1, 0, 2], dtype=torch.int64)
        attn_scale = torch.tensor(0.25)

        got = _adapter(q, k, v, so, num_targets=num_targets, attn_scale=attn_scale)
        want = _original(q, k, v, so, num_targets, attn_scale)
        self.assertTrue(torch.equal(got, want), f"{got}\n!=\n{want}")

    def test_eager_backward_is_byte_identical_to_the_captured_original(self) -> None:
        q, k, v = _qkv(requires_grad=True)
        so = _seq_offsets()
        num_targets = torch.tensor([1, 0, 2], dtype=torch.int64)
        attn_scale = torch.tensor(0.25)

        got = torch.autograd.grad(
            _adapter(q, k, v, so, num_targets=num_targets, attn_scale=attn_scale).sum(),
            (q, k, v),
        )
        want = torch.autograd.grad(
            _original(q, k, v, so, num_targets, attn_scale).sum(),
            (q, k, v),
        )
        for name, g, w in zip("qkv", got, want):
            self.assertTrue(torch.equal(g, w), f"d{name}: {g}\n!=\n{w}")

    def test_none_optionals_normalize_to_zeros_and_ones(self) -> None:
        """``attn_scale=None`` means 1.0 here, NOT hammer's 1/N.

        Both Helion sites (the kernel wrappers' ``v_scaled`` fold and the padded
        reference) read a missing scale as 1.0, so deferring to hammer's 1/N would
        be a silent factor-N error. Pin the normalization at the adapter.
        """
        q, k, v = _qkv()
        so = _seq_offsets()

        got = _adapter(q, k, v, so, num_targets=None, attn_scale=None)
        want = _original(q, k, v, so, so.new_zeros((_B,)), v.new_ones(()))
        self.assertTrue(torch.equal(got, want))

        # And it is NOT hammer's 1/N reading of the same call.
        hammer_none = _original(q, k, v, so, so.new_zeros((_B,)), None)
        self.assertFalse(torch.equal(got, hammer_none))
