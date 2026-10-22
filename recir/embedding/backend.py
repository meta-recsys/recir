# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Backend-agnostic embedding interface shared by every recir embedding backend.

Deliberately free of JAX and of any model-specific import so that CPU-only
callers (and the quantization hooks) can depend on the interface without
pulling in the ~500 MiB JAX/SparseCore stack. The concrete backends live in
``cpu.py``, ``sparsecore.py``, and ``in_graph.py``.

Models bind a backend to their own architecture by supplying what the concrete
backends accept (``embedding_collection_fqn``, ``sparse_optimizer_builder``)
and, where tracing overrides are needed, by
subclassing to add a model-specific ``get_override``.
"""

from __future__ import annotations

import enum
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import torch.nn as nn
from recir.training.optim import OptimConfig
from torchrec.distributed.types import ShardingType
from torchrec.modules.embedding_configs import BaseEmbeddingConfig
from torchrec.optim.keyed import KeyedOptimizer

logger: logging.Logger = logging.getLogger(__name__)


class TableSharding(enum.Enum):
    """How one embedding table is laid out across devices, and how it is read.

    Storage and lookup only. It says nothing about which optimizer trains the
    table -- see :class:`TableOptimizer`.
    """

    # Dim-0 sharded; looked up via the ring-shift kernel.
    ROWWISE = "rowwise"
    # Dim-0 sharded; looked up via ``recir.fsdp_embedding`` (all-gather).
    FSDP = "fsdp"
    # Full copy on every device; looked up via ``F.embedding``.
    REPLICATED = "replicated"


class TableOptimizer(enum.Enum):
    """Which optimizer trains one embedding table.

    Orthogonal to :class:`TableSharding`: a sharded table can be stepped by the
    dense optimizer (that is what FSDP does) and a replicated one by the sparse
    optimizer. They are separate axes because they answer separate questions --
    sharding is a memory decision, this is a training one, and the two rates are
    ~50x apart.
    """

    SPARSE = "sparse"
    DENSE = "dense"


@dataclass(frozen=True)
class TablePlan:
    """The complete per-table decision: where it lives, and who trains it."""

    sharding: TableSharding
    optimizer: TableOptimizer


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


def swap_submodule(model: nn.Module, fqn: str, replacement: nn.Module) -> nn.Module:
    """Bind ``replacement`` at ``fqn`` in ``model``; return the module it replaces."""
    if not fqn:
        raise ValueError("fqn names the root model; it must name a submodule")
    original = model.get_submodule(fqn)
    parent_fqn, _, leaf = fqn.rpartition(".")
    parent = model.get_submodule(parent_fqn) if parent_fqn else model
    setattr(parent, leaf, replacement)
    return original


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
        table_configs: Sequence[BaseEmbeddingConfig],
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


def _sharding_value(name: str, table: object) -> str:
    """``table.sharding``'s enum VALUE, rejecting the two silent failures.

    A bare string means something deserialized the enum by NAME: it compares
    unequal to every ShardingType member while still being non-None, so every
    reader sees "no placement" and says nothing. ``None`` means the config never
    expressed one, which used to fall through to a row-count heuristic and now
    has nowhere to fall through to.
    """
    sharding = getattr(table, "sharding", None)
    if isinstance(sharding, str):
        raise TypeError(
            f"table {name!r} has sharding={sharding!r} as a plain string "
            "rather than a ShardingType member. Something deserialized the "
            "enum by name; comparisons against ShardingType will all be "
            "False and the placement would be silently lost."
        )
    if sharding is None:
        raise ValueError(
            f"table {name!r} has no sharding. Every table needs an explicit "
            "placement: it decides both the layout and which optimizer trains "
            "the table, and the two learning rates are ~50x apart."
        )
    value = getattr(sharding, "value", sharding)
    return str(value)


def table_plans_from_embedding_tables(
    embedding_tables: Dict[str, object],
) -> Dict[str, TablePlan]:
    """Per-table layout + optimizer, from production's ``EmbeddingTable.sharding``.

    ``sharding`` is one field that decides two things, because that is how
    torchrec couples them: the sharding type picks the compute kernel, and the
    kernel picks the optimizer (see this module's header table). The in-graph
    backend has no such coupling -- it reads plain ``nn.Parameter``s -- so it
    has to reproduce the pairing deliberately:

    ==============  ==================  ====================
                    ROW_WISE            DATA_PARALLEL
    ==============  ==================  ====================
    layout          dim-0 sharded       full replica
    lookup          ring-shift          ``F.embedding``
    optimizer       sparse              dense
    ==============  ==================  ====================

    ``TableSharding.FSDP`` is deliberately unreachable from here: no production
    config asks for a sharded non-ring-shift table. It stays available to
    callers that construct plans directly.

    Raises on anything it cannot map, rather than defaulting. A table in the
    wrong bucket keeps its shape, its key and its state_dict entry and only
    trains at the wrong rate, so a silent fallback is not detectable downstream.
    """
    plans: Dict[str, TablePlan] = {}
    for name in embedding_tables:
        value = _sharding_value(name, embedding_tables[name])
        if value == ShardingType.ROW_WISE.value:
            plans[name] = TablePlan(TableSharding.ROWWISE, TableOptimizer.SPARSE)
        elif value == ShardingType.DATA_PARALLEL.value:
            plans[name] = TablePlan(TableSharding.REPLICATED, TableOptimizer.DENSE)
        else:
            raise ValueError(
                f"table {name!r} asks for sharding={value!r}, which the in-graph "
                "backend has no lookup for. It implements ROW_WISE and "
                "DATA_PARALLEL only."
            )
    logger.info(
        "table plans: %s",
        {n: (p.sharding.value, p.optimizer.value) for n, p in plans.items()},
    )
    return plans
