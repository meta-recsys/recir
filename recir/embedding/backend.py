# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Backend-agnostic embedding interface shared by every recir embedding backend.

Deliberately free of JAX and of any model-specific import so that CPU-only
callers (and the quantization hooks) can depend on the interface without
pulling in the ~500 MiB JAX/SparseCore stack. The concrete backends live in
``cpu.py``, ``sparsecore.py``, and ``in_graph.py``.

Models bind a backend to their own architecture by supplying the seam
callables the concrete backends accept (``sparse_arch_getter``,
``sparse_optimizer_builder``) and, where tracing overrides are needed, by
subclassing to add a model-specific ``get_override``.
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn
from minimal_viable_ai.core.model_family_api.optimizer_configs import OptimConfig
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.optim.keyed import KeyedOptimizer


class SparseUpdatePlacement(enum.Enum):
    """Where the sparse optimizer update executes."""

    BACKEND_FUSED = "backend_fused"
    COMPILED_OPTIMIZER = "compiled_optimizer"


@dataclass
class SparseOptimizerRuntime:
    """Backend-independent description of sparse optimizer execution."""

    placement: SparseUpdatePlacement
    config: OptimConfig
    optimizer: torch.optim.Optimizer | None = None
    lr_buffer: torch.Tensor | None = None
    # BACKEND_FUSED only: the CombinedOptimizer gathering the fused TBEs. It does
    # NOT perform the update -- the backend does that inside backward. Kept
    # separate from ``optimizer`` on purpose: a BACKEND_FUSED runtime must not
    # own an update optimizer, but it can own the handle that drives the LR.
    # This is the only handle to the TBEs' param_groups, and zero_grad()/step()
    # are what push param_groups[0]["lr"] into FBGEMM, so a leg that wants a
    # sparse LR schedule writes the rate here and then calls one of them BEFORE
    # backward (see EagerCPUStep._push_sparse_lr); a rate pushed afterwards
    # applies one step late. Carried because setup() returns dmp.module, not the
    # DMP, and the CombinedOptimizer is assembled in DMP.__init__ -- nothing
    # downstream could rebuild it.
    fused_optimizer: KeyedOptimizer | None = None

    def __post_init__(self) -> None:
        if (
            self.placement == SparseUpdatePlacement.COMPILED_OPTIMIZER
            and self.optimizer is None
        ):
            raise ValueError("A compiled sparse runtime requires an optimizer.")
        if (
            self.placement == SparseUpdatePlacement.BACKEND_FUSED
            and self.optimizer is not None
        ):
            raise ValueError("A backend-fused sparse runtime cannot own an optimizer.")
        if (
            self.config.lr_warmup_schedule
            and self.lr_buffer is None
            and self.fused_optimizer is None
        ):
            raise NotImplementedError(
                "This backend cannot apply the configured sparse LR schedule: it "
                "supplied neither an LR buffer (compiled-optimizer path) nor a "
                "fused optimizer (backend-fused path)."
            )


@dataclass
class EmbeddingSetupResult:
    """Artifacts returned by an optimizer-aware embedding backend setup."""

    model: nn.Module
    dense_named_parameters: dict[str, nn.Parameter]
    sparse_runtime: SparseOptimizerRuntime


class EmbeddingBackend(ABC):
    """Backend-agnostic interface for embedding storage, lookup, and PG init.

    Models add get_override() with model-specific signature since the
    override type and arguments differ between model architectures.
    """

    @abstractmethod
    def init_pg(self) -> tuple[int, int]:
        """Initialize process group. Returns (rank, world_size)."""
        ...

    def setup(
        self,
        model: nn.Module,
        table_configs: list[EmbeddingConfig],
        sparse_config: OptimConfig,
        device: torch.device,
        world_size: int,
    ) -> EmbeddingSetupResult:
        """Set up storage and sparse optimizer ownership in one operation.

        The result tells the caller whether sparse updates execute in a
        backend-fused path or through a compiled torch optimizer.
        """
        raise NotImplementedError

    @abstractmethod
    def dense_named_parameters(self, model: nn.Module) -> dict[str, nn.Parameter]:
        """Trainable DENSE params (those NOT owned/trained by the embedding
        backend), as an FQN -> Parameter dict.

        Backend-specific: the DMP backend uses ``in_backward_optimizer_filter``;
        the SparseCore backend excludes the embedding collection.
        """
        ...

    def sc_modulus(self) -> int:
        """Per-host SparseCore CSR shard width this backend requires.

        Default 1 means no extra alignment (PRT's existing num_devices
        rounding is sufficient). JAX/SparseCore backends override to return
        ``local_dev × num_sc_per_device`` so per-host per-key budgets
        become multiples of SC's CSR shard width — required for
        ``tpu_sparse_dense_matmul``. Callers that need a global modulus
        multiply by ``num_processes``.
        """
        return 1
