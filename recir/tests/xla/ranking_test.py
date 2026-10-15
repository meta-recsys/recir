# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors

"""Tests for causal-LM ranking model through recir GSPMD compilation.

Validates that a causal LM ranking model (Qwen2 + scoring + DDRO loss)
compiles through the XLA_GSPMD backend without graph breaks, and that
training produces valid results.

Uses the single-process multi-device pattern (TPU VM execution model):
num_processes=1, num_ranks=N (all local devices in one process).
Works on both GPU and TPU.

"""

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from recir.api import compile
from recir.backends import XLAGSPMDBackend
from recir.lowering.lowering_utils import has_meta_kernel
from recir.lowering.xla_lower import _XLAModule
from recir.tests.xla.gspmd_test_utils import assert_hlo_has_collective
from transformers import Qwen2Config, Qwen2ForCausalLM  # @manual

# ── Tiny model constants ──
VOCAB_SIZE = 1000
HIDDEN_SIZE = 256
INTERMEDIATE_SIZE = 512
NUM_LAYERS = 2
NUM_HEADS = 4
NUM_KV_HEADS = 2

# ── Data constants ──
BATCH_SIZE = 8
NUM_ITEMS = 4
SEQ_LEN = 32
PREFIX_LEN = 16


class DDROWrapper(nn.Module):
    """Minimal ranking model: Qwen2 CausalLM -> per-item scores -> DDRO loss.

    Replicates the forward logic of DDROModelWrapper + compute_scores_from_logits
    + RankingLossVectorized without external model-framework dependencies.

    The forward path is fully traceable by torch.fx / make_fx:
      1. Flatten [batch, items, seq] -> [batch*items, seq] for LLM forward
      2. Shift logits/labels for next-token prediction
      3. Cross-entropy per token -> average log-prob per item -> scores
      4. Vectorized pairwise DDRO loss on scores vs weights
    """

    def __init__(self, llm, ddro_beta=1.0):
        super().__init__()
        self.llm = llm
        self.ddro_beta = ddro_beta

    def forward(self, input_ids, attention_mask, labels, weights):
        batch_size, num_items, seq_len = input_ids.shape

        # Flatten for LLM forward
        flat_ids = input_ids.view(batch_size * num_items, seq_len).contiguous()
        flat_labels = labels.view(batch_size * num_items, seq_len).contiguous()

        # Pre-compute a simple 4D causal mask so that transformers >=4.46
        # skips its create_causal_mask() utility (which introduces ops
        # without Meta dispatch kernels, causing XLA graph breaks).
        bsz = flat_ids.shape[0]
        min_dtype = torch.finfo(torch.float32).min
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), min_dtype, device=flat_ids.device),
            diagonal=1,
        )
        causal_mask = causal_mask[None, None, :, :].expand(bsz, 1, -1, -1)

        outputs = self.llm(
            input_ids=flat_ids,
            attention_mask=causal_mask,
            use_cache=False,
        )
        logits = outputs.logits.float()

        # ── compute_scores_from_logits (inlined) ──
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = flat_labels[:, 1:].contiguous()
        per_token_ce = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        )
        per_token_lp = -per_token_ce.view(batch_size * num_items, -1)
        mask = (shift_labels != -100).float()
        valid_tokens = mask.sum(dim=-1).clamp(min=1.0)
        scores = (per_token_lp.sum(dim=-1) / valid_tokens).view(batch_size, num_items)

        # ── Vectorized DDRO loss (inlined, fully traceable) ──
        w_diff = weights.float().unsqueeze(2) - weights.float().unsqueeze(1)
        pair_mask = (w_diff > 1e-4).float()
        s_diff = scores.unsqueeze(2) - scores.unsqueeze(1)
        pair_losses = F.softplus(-self.ddro_beta * s_diff)
        masked = (pair_losses * pair_mask).float()
        loss = masked.sum() / pair_mask.sum().clamp(min=1)
        return loss


def _create_qwen2():
    """Create a tiny Qwen2 model matching ToyDDROWorkflowConfig dimensions."""
    config = Qwen2Config(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        max_position_embeddings=512,
        use_cache=False,
        attn_implementation="eager",
    )
    return Qwen2ForCausalLM(config)


def _create_ranking_data():
    """Create synthetic ranking batch matching DDRO collator output format."""
    torch.manual_seed(42)
    input_ids = torch.randint(1, VOCAB_SIZE, (BATCH_SIZE, NUM_ITEMS, SEQ_LEN))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[:, :, :PREFIX_LEN] = -100  # mask prefix tokens
    weights = torch.randint(0, 5, (BATCH_SIZE, NUM_ITEMS)).float()
    return input_ids, attention_mask, labels, weights


def _run_ranking_baseline(num_ranks, num_iters=3):
    """Run ranking model on CPU without XLA, return per-step param snapshots.

    For sum loss reduction, divides loss by num_ranks to match the
    all-reduce averaging in replicated data parallelism.
    """
    torch.manual_seed(42)
    llm = _create_qwen2()
    model = DDROWrapper(llm).cpu()
    opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
    input_ids, attention_mask, labels, weights = _create_ranking_data()
    params_per_step = []
    for _ in range(num_iters):
        loss = model(input_ids, attention_mask, labels, weights)
        (loss / num_ranks).backward()
        opt.step()
        opt.zero_grad()
        params_per_step.append({n: p.clone() for n, p in model.named_parameters()})
    return params_per_step


def _collect_ops_missing_meta_kernel(compiled_obj):
    """Return ops without a Meta dispatch kernel across all compiled subgraphs.

    Checks both outer graph nodes and _XLAModule inner subgraphs.
    Ops without a Meta kernel cause segment_and_lower to emit
    'partial_lower: graph break' warnings and fragment the XLA computation.

    Profiler ops (``profiler::*``) are excluded because they are stripped
    by ``_strip_profiler_nodes()`` before XLA lowering and never cause
    real graph breaks.
    """
    missing = []
    for gm_name in ("setup_gm", "main_gm", "cleanup_gm"):
        gm = getattr(compiled_obj.gm, gm_name)
        for scope, graph in _iter_graphs(gm_name, gm):
            for node in graph.nodes:
                if node.op == "call_function" and isinstance(
                    node.target, torch._ops.OpOverload
                ):
                    # Profiler ops are stripped before XLA lowering by
                    # _strip_profiler_nodes() and don't cause real graph breaks.
                    if node.target.name().startswith("profiler::"):
                        continue
                    if not has_meta_kernel(node.target):
                        missing.append(f"  [{scope}] {node} -- {node.target}")
    return missing


def _iter_graphs(gm_name, gm):
    """Yield (scope_label, graph) for the outer graph and each _XLAModule subgraph."""
    yield gm_name, gm.graph
    for mod in gm.modules():
        if isinstance(mod, _XLAModule):
            yield f"{gm_name}/xla", mod.gm.graph


class TestRankingCompile(unittest.TestCase):
    """Causal-LM ranking model through recir compile: no graph breaks.

    Uses single-process multi-device pattern (all JAX devices in one process).
    Validates that the causal-LM ranking model compiles through XLA_GSPMD
    without graph breaks and produces valid training results.

    Runs on both GPU and TPU.
    """

    def setUp(self):
        import jax  # @manual

        self.devices = jax.devices()
        if len(self.devices) < 2:
            self.skipTest("Need >= 2 JAX devices")
        self.num_ranks = len(self.devices)

    def _compile_ranking_model(self, *, dump_hlo: bool = False):
        """Compile the ranking model through GSPMD and return compiled_obj."""
        from recir.distributed.parallel_mode import GSPMDParallel

        torch.manual_seed(42)
        llm = _create_qwen2()
        model = DDROWrapper(llm).cpu()
        opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
        input_ids, attention_mask, labels, weights = _create_ranking_data()

        @compile(
            parallel_mode=GSPMDParallel(
                "replicate",
                loss_reduction="sum",
                num_ranks=self.num_ranks,
                num_processes=1,
            ),
            backend=XLAGSPMDBackend(dump_hlo=dump_hlo),
        )
        def train_step(mod, opt, input_ids, attention_mask, labels, weights):
            loss = mod(input_ids, attention_mask, labels, weights)
            loss.backward()
            opt.step()

        train_step(model, opt, input_ids, attention_mask, labels, weights)
        return train_step.__dict__["_compiled_obj"]

    def test_no_graph_break(self):
        """Ranking model compiles with exactly 1 XLA subgraph per graph."""
        compiled_obj = self._compile_ranking_model()
        for gm_name in ("setup_gm", "main_gm", "cleanup_gm"):
            gm = getattr(compiled_obj.gm, gm_name)
            xla_modules = [m for m in gm.modules() if isinstance(m, _XLAModule)]
            self.assertEqual(
                len(xla_modules),
                1,
                f"{gm_name}: expected exactly 1 XLA subgraph (no graph "
                f"breaks), got {len(xla_modules)}.",
            )

    def test_no_partial_lower_fragmentation(self):
        """No ops without a Meta kernel should appear in the compiled graph.

        The absence of a Meta kernel is the exact condition that makes
        segment_and_lower emit 'partial_lower: graph break' warnings and
        fragment the XLA computation. This checks every call_function node
        in both the outer graph and inside _XLAModule subgraphs, bypassing
        _is_xla_compatible's allowlist to catch ops that pass on GPU but
        break on other PyTorch versions or platforms.
        """
        compiled_obj = self._compile_ranking_model()
        missing_meta = _collect_ops_missing_meta_kernel(compiled_obj)
        self.assertEqual(
            len(missing_meta),
            0,
            f"Found {len(missing_meta)} ops without Meta kernel "
            f"(will cause partial_lower graph breaks):\n" + "\n".join(missing_meta),
        )

    def test_hlo_has_collective(self):
        """GSPMD-compiled HLO must contain all-reduce for gradient averaging."""
        compiled_obj = self._compile_ranking_model(dump_hlo=True)
        assert_hlo_has_collective(self, compiled_obj, "all-reduce", "XLA_GSPMD")

    def test_training_runs_multiple_steps(self):
        """Multi-step training matches CPU baseline numerically."""
        from recir.distributed.parallel_mode import GSPMDParallel

        num_iters = 3
        baseline_params = _run_ranking_baseline(self.num_ranks, num_iters)

        torch.manual_seed(42)
        llm = _create_qwen2()
        model = DDROWrapper(llm).cpu()
        opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
        input_ids, attention_mask, labels, weights = _create_ranking_data()

        @compile(
            parallel_mode=GSPMDParallel(
                "replicate",
                loss_reduction="sum",
                num_ranks=self.num_ranks,
                num_processes=1,
            ),
            backend=XLAGSPMDBackend(),
        )
        def train_step(mod, opt, input_ids, attention_mask, labels, weights):
            loss = mod(input_ids, attention_mask, labels, weights)
            loss.backward()
            opt.step()

        for step in range(num_iters):
            train_step(model, opt, input_ids, attention_mask, labels, weights)
            for (name, param), (_, baseline_param) in zip(
                model.named_parameters(),
                baseline_params[step].items(),
            ):
                torch.testing.assert_close(
                    param,
                    baseline_param,
                    msg=f"Step {step}: ranking XLA_GSPMD vs baseline mismatch for {name}",
                )
