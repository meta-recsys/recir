# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""CPU checks for the EBC end-to-end example.

The example itself needs a TPU: its last step is a Pallas kernel, which has no
CPU backend. Everything up to that step does run here, and that turns out to be
most of what can go wrong.

Two things are covered:

  * The eager path trains. This exercises the model, the synthetic batches,
    ``InGraphSparseArch`` and the op's CPU fallback, so a bad reference cannot
    quietly become the baseline the TPU run is compared against.
  * The TPU path *traces*, all the way into ``pallas_call``, and produces a
    table gradient of the right shape. ``jax.eval_shape`` stops short of
    executing, so this runs on CPU while still proving the op was registered,
    the lowering was found, and the pooled branch was taken.

That second one is worth more than it looks. A missing lowering does not raise:
the op falls back to CPU and the run merely gets slower, so "it ran" is not
evidence the kernel was used.
"""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np
import torch
from recir.examples.rankfm_mvai.ebc_e2e import (
    _bce,
    _build_report,
    _passes_acceptance,
    make_batches,
    run_cpu_reference,
    TinyRecModel,
    TrainingResult,
)
from recir.torchax import extract_jax


# One device on CPU, so the arch sees world=1 and the whole batch is one chip.
_WORLD = 1


class _Cfg:
    steps = 40
    batch_size = 8
    # 8 bags x avg 2 = 16 tokens per chip, capped at 4 per bag. Well under the
    # cap on average, so the dealing leaves some bags empty -- which is the case
    # the mean divisor clamp exists for and the one a uniform batch never makes.
    avg_bag = 2
    max_bag = 4
    vocab = 32
    dim = 8
    hidden = 4
    lr = 0.5
    seed = 3
    rtol = 2e-3


def _model(cfg=_Cfg) -> TinyRecModel:
    torch.manual_seed(cfg.seed)
    return TinyRecModel(cfg, _WORLD)


class EagerTrainingTest(unittest.TestCase):
    def test_loss_decreases(self) -> None:
        cfg = _Cfg()
        batches = make_batches(cfg, _WORLD, np.random.default_rng(cfg.seed))
        losses = run_cpu_reference(cfg, _model(cfg), batches).losses
        self.assertEqual(len(losses), cfg.steps)
        head = sum(losses[:5]) / 5
        tail = sum(losses[-5:]) / 5
        self.assertLess(tail, head, f"no descent: {head:.4f} -> {tail:.4f}")

    def test_batches_include_an_empty_bag(self) -> None:
        """The descent above is only meaningful if the hard case is in the data.

        Empty bags are what the mean divisor clamp and the padding mask exist
        for; a batch generator that never produced one would make the training
        test pass while covering nothing interesting.
        """
        cfg = _Cfg()
        batches = make_batches(cfg, _WORLD, np.random.default_rng(cfg.seed))
        self.assertTrue(any((lengths == 0).any() for _, lengths, _ in batches))

    def test_gradient_reaches_the_table(self) -> None:
        """A dense layer sits between the loss and the table; the grad must cross it."""
        cfg = _Cfg()
        model = _model(cfg)
        ids, lengths, labels = make_batches(
            cfg, _WORLD, np.random.default_rng(cfg.seed)
        )[0]
        logits = model(torch.from_numpy(ids), torch.from_numpy(lengths))
        torch.nn.BCEWithLogitsLoss()(logits, torch.from_numpy(labels)).backward()
        table = model.table()
        self.assertIsNotNone(table.grad)
        self.assertGreater(float(table.grad.abs().sum()), 0.0)


class AcceptanceTest(unittest.TestCase):
    def test_cpu_parity_passes_when_random_batch_loss_does_not_descend(self) -> None:
        self.assertTrue(
            _passes_acceptance(
                descended=False,
                max_rel=6.473e-5,
                rtol=2e-3,
                skip_cpu=False,
            )
        )

    def test_failure_when_neither_signal_passes(self) -> None:
        self.assertFalse(
            _passes_acceptance(
                descended=False,
                max_rel=3e-3,
                rtol=2e-3,
                skip_cpu=False,
            )
        )

    def test_skip_cpu_requires_loss_descent(self) -> None:
        self.assertFalse(
            _passes_acceptance(
                descended=False,
                max_rel=float("nan"),
                rtol=2e-3,
                skip_cpu=True,
            )
        )


class ReportTest(unittest.TestCase):
    def test_report_contains_loss_and_gradient_comparison(self) -> None:
        cfg = _Cfg()
        cfg.rtol = 2e-3
        tpu = TrainingResult(
            losses=[0.7, 0.6],
            first_step_gradients={"weight": np.array([1.0, 2.0])},
        )
        cpu = TrainingResult(
            losses=[0.70001, 0.60001],
            first_step_gradients={"weight": np.array([1.00001, 2.00001])},
        )
        report = _build_report(
            cfg,
            _WORLD,
            tpu,
            cpu,
            descended=True,
            parity_ok=True,
            ok=True,
        )
        self.assertIn("# EBC E2E CPU vs TPU Correctness Report", report)
        self.assertIn("## Per-step loss comparison", report)
        self.assertIn("## First-step gradient comparison", report)
        self.assertIn("`weight`", report)
        self.assertIn("=== EBC_E2E_REPORT_END ===", report)


class LoweringTraceTest(unittest.TestCase):
    """Trace the TPU path without executing it."""

    def _shapes(self):
        cfg = _Cfg()
        model = _model(cfg)
        states, jax_func = extract_jax(model)
        ids, lengths, labels = make_batches(
            cfg, _WORLD, np.random.default_rng(cfg.seed)
        )[0]

        def loss_fn(states, ids, lengths, labels):
            return _bce(jax_func(states, (ids, lengths)), labels)

        return (
            cfg,
            states,
            jax.eval_shape(
                jax.grad(loss_fn),
                states,
                jnp.asarray(ids),
                jnp.asarray(lengths),
                jnp.asarray(labels),
            ),
        )

    def test_table_gradient_keeps_the_table_shape(self) -> None:
        cfg, _states, grads = self._shapes()
        (table_key,) = [k for k in grads if "_sharded_tables" in k]
        self.assertEqual(grads[table_key].shape, (cfg.vocab, cfg.dim))
        self.assertEqual(grads[table_key].dtype, jnp.float32)

    def test_every_parameter_gets_a_gradient(self) -> None:
        """Including the MLP: a break anywhere in the chain shows up as a miss."""
        _cfg, states, grads = self._shapes()
        self.assertEqual(set(grads), set(states))
        for name, g in grads.items():
            self.assertEqual(g.shape, states[name].shape, name)

    def test_jax_and_torch_parameter_names_match_for_report(self) -> None:
        cfg, states, _grads = self._shapes()
        self.assertEqual(set(states), set(dict(_model(cfg).named_parameters())))
