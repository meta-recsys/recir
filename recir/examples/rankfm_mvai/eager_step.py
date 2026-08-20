#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""The recir CPU-eager optimizer step, with no mvai_cloud / jax dependency.

Split out of ``train_pipeline.py`` so this code can run OUTSIDE the TPU image.
``EagerCPUTrainPipeline`` is unimportable anywhere else: it subclasses
``RankingSGDTrainPipeline``, which exists only in the ``mvai_cloud`` fork --
rsynced into the Docker image, no BUCK target, and shadowed on a devserver by
the canonical ``minimal_viable_ai`` package where those names do not exist.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from hammer.v3.models.udd_fm_v3.recir_core.trace_sample import flatten_batch
from torchrec.optim.keyed import KeyedOptimizer

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class EagerCPUStep:
    """One optimizer step of the CPU-eager leg, in the order it runs.

    Over K micro-batches: ``zero_grad()`` once, then per micro-batch
    ``forward()`` -> ``scale_loss()`` -> ``backward()``, then
    ``optimizer_step()`` once.

    ``zero_grad`` runs at the START, not after the optimizer: the caller's
    digest reads ``p.grad`` and post-step ``p.data`` in one pass.

    Embedding tables are NOT stepped here -- an in-backward optimizer owns them
    and fires inside ``backward()``, once per micro-batch, while the dense side
    steps once per group. That asymmetry is real.
    """

    raw_model: nn.Module
    dense_opt: torch.optim.Optimizer
    dense_params: List[nn.Parameter]
    # 0-d buffer the WarmupController mutates in place; the current LR is read
    # off it every step rather than baked into the optimizer.
    lr_buffer: torch.Tensor
    warmup: Any
    max_grad: Optional[float]
    # The fused embedding TBEs' optimizer, and the 0-d buffer the shared
    # WarmupController writes the sparse LR into. Both or neither.
    fused_opt: Optional[KeyedOptimizer] = None
    sparse_lr_buffer: Optional[torch.Tensor] = None
    light_loss_norm: bool = False
    grad_accum_steps: int = 1

    @classmethod
    def from_parts(
        cls,
        raw_model: nn.Module,
        tensor_lr_opt: Any,
        warmup: Any,
        *,
        fused_opt: Optional[KeyedOptimizer] = None,
        sparse_lr_buffer: Optional[torch.Tensor] = None,
        light_loss_norm: bool = False,
        grad_accum_steps: int = 1,
    ) -> EagerCPUStep:
        """Rebuild the dense optimizer as a plain CPU AdamW over the same params.

        ``TensorLRAdamW`` takes its LR from a 0-d tensor through the
        ``capturable=True`` path, which cannot run on CPU. Same params, same
        hyperparameters, scalar LR -- an identical update. The pre-allocated
        Adam state is transferred rather than left for a second lazy copy.
        """
        assert grad_accum_steps >= 1, "grad_accum_steps must be >= 1"
        dense_params: List[nn.Parameter] = [
            p for group in tensor_lr_opt.param_groups for p in group["params"]
        ]
        defaults = tensor_lr_opt.defaults
        dense_opt = torch.optim.AdamW(
            dense_params,
            lr=defaults["lr"],
            betas=defaults["betas"],
            eps=defaults["eps"],
            weight_decay=defaults["weight_decay"],
            amsgrad=defaults.get("amsgrad", False),
            capturable=False,
            # Citrine C2: multi-tensor AdamW. Same update math (parity
            # preserved); the dense params are non-sparse (embeddings run
            # in-backward Adagrad, handled separately).
            foreach=True,
        )
        dense_opt.state.update(tensor_lr_opt.state)
        tensor_lr_opt.state.clear()
        return cls(
            raw_model=raw_model,
            dense_opt=dense_opt,
            dense_params=dense_params,
            lr_buffer=tensor_lr_opt.param_groups[0]["lr"],
            warmup=warmup,
            max_grad=getattr(tensor_lr_opt, "_max_grad", None),
            fused_opt=fused_opt,
            sparse_lr_buffer=sparse_lr_buffer,
            light_loss_norm=light_loss_norm,
            grad_accum_steps=grad_accum_steps,
        )

    @classmethod
    def from_wrapper(
        cls,
        wrapper: Any,
        *,
        light_loss_norm: bool = False,
        grad_accum_steps: int = 1,
    ) -> EagerCPUStep:
        """Reach through ``RecirXLACompiledModule`` for the already-wired parts.

        The wrapper compiles lazily on its first ``__call__`` and this leg never
        calls it; it is used only as the thing that already ran
        ``backend.setup`` (DMP) and ``setup_dense_optimizer_and_warmup``.
        """
        return cls.from_parts(
            wrapper._model,
            wrapper._dense_opt,
            wrapper._warmup,
            fused_opt=wrapper._sparse_runtime.fused_optimizer,
            sparse_lr_buffer=wrapper._sparse_lr_buffer,
            light_loss_norm=light_loss_norm,
            grad_accum_steps=grad_accum_steps,
        )

    def zero_grad(self) -> None:
        self.dense_opt.zero_grad(set_to_none=True)
        self._push_sparse_lr()

    def _push_sparse_lr(self) -> None:
        """Send the warmup's current sparse LR into the fused TBEs.

        Before BACKWARD, not at step time: the fused embedding update runs
        inside backward, so a rate pushed afterwards would apply one step late.
        ``EmbeddingFusedOptimizer.zero_grad`` is the push -- its whole body is
        ``set_learning_rate(param_groups[0]["lr"])``, same as its ``step()``.
        Without it the tables keep the rate baked in at construction
        (``fused_params["learning_rate"]``) and the schedule never reaches them.
        """
        if self.fused_opt is None or self.sparse_lr_buffer is None:
            return
        lr_val = float(self.sparse_lr_buffer.item())
        for group in self.fused_opt.param_groups:
            # KeyedOptimizer types param_groups read-only, but the dicts are the
            # live ones the fused optimizer reads; torchrec's own WarmupOptimizer
            # writes them the same way (warmup.py:146-148).
            # pyre-ignore[16]
            group["lr"] = lr_val
        self.fused_opt.zero_grad()

    def forward(self, batch: Any, n_total: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """One micro-batch through the raw-tensor entrypoint.

        ``flatten_batch``'s last element is an all-reduced
        ``world_candidate_sum``, replaced because this leg picks the
        denominator itself: ``None`` falls back to ``torch.mean`` (what light
        gets, since it passes none), ``n_total`` normalizes by the group.
        """
        flat_args = list(flatten_batch(batch))
        flat_args[-1] = None if self.light_loss_norm else n_total
        return self.raw_model(*flat_args)

    def scale_loss(self, all_losses: torch.Tensor) -> torch.Tensor:
        """Weight one micro-batch's contribution to the accumulated gradient.

        1/K, NOT n_i/n_total. Under ``light_loss_norm`` each micro-batch arrives
        as a per-batch mean, and light runs those same means on K ranks with DDP
        all-reduce-MEAN -- every rank weighted equally.

        Weighting by candidate count instead gives the gradient of one global mean
        over the concatenated batch, a different quantity whenever the counts
        differ (~19% apart on this dump). It compounds every step, so it surfaces
        as slow NE divergence rather than anything one forward would catch. This
        code shipped with that bug once.

        Without ``light_loss_norm`` the ``n_total`` denominator already did it.
        """
        if self.light_loss_norm and self.grad_accum_steps > 1:
            return all_losses / self.grad_accum_steps
        return all_losses

    def backward(self, all_losses: torch.Tensor) -> None:
        """Also fires the in-backward embedding optimizer, per micro-batch."""
        all_losses.sum().backward()

    def optimizer_step(self) -> None:
        """Clip, step at the CURRENT warmup LR, then advance the warmup.

        Mirrors ``RecirXLACompiledModule._post_step``. Once per optimizer step,
        not per micro-batch.

        The sparse side is not touched here -- its rate reaches the tables
        before BACKWARD, from :meth:`_push_sparse_lr`, because the fused update
        runs inside backward rather than at step time.
        """
        if self.max_grad is not None:
            torch.nn.utils.clip_grad_value_(self.dense_params, self.max_grad)
        lr_val = float(self.lr_buffer.item())
        for group in self.dense_opt.param_groups:
            group["lr"] = lr_val
        self.dense_opt.step()
        self.warmup.step()

    def run_step(
        self,
        micro_batches: List[Any],
        *,
        before_forward: Optional[Callable[[Any], None]] = None,
        after_forward: Optional[Callable[[torch.Tensor], None]] = None,
        after_backward: Optional[Callable[[Dict[str, Any], int], None]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """The whole step: the ORDER, not just the pieces.

        Accumulation exists because a materialized global batch does not fit: the
        eager attention kernel holds every [B, H, N, N] score matrix for backward,
        and 2048 rows was oomd-killed at ~388GB against ~112GB for 4x512. K
        micro-batches give the optimizer the same global batch as a K-rank GPU leg.

        ``n_total`` must be known before the FIRST backward, so the caller
        materializes the whole group and passes it in -- batch handles, not
        activations, so holding K is cheap.

        The callbacks let the pipeline interleave its metrics and digest without
        owning the sequence; none of them touch a gradient.

        Returns the accumulated per-task loss and the LAST micro-batch's tensors.
        """
        self.zero_grad()
        per_batch_candidates: List[int] = [
            int(b.num_candidates.sum()) for b in micro_batches
        ]
        n_total = sum(per_batch_candidates)

        total_losses: Optional[torch.Tensor] = None
        named: Dict[str, Any] = {}
        for batch, n_i in zip(micro_batches, per_batch_candidates):
            if before_forward is not None:
                before_forward(batch)
            all_losses, named = self.forward(batch, n_total)
            if after_forward is not None:
                after_forward(all_losses)
            all_losses = self.scale_loss(all_losses)
            self.backward(all_losses)
            total_losses = (
                all_losses.detach()
                if total_losses is None
                else total_losses + all_losses.detach()
            )
            if after_backward is not None:
                after_backward(named, n_i)

        self.optimizer_step()
        assert total_losses is not None, "run_step needs at least one micro-batch"
        return total_losses, named
