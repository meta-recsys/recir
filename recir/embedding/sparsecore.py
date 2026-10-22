# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""JAX/SparseCore embedding backend.

Tables stored as SparseCore EmbeddingVariables; lookup via
``tpu_sparse_dense_matmul``; Adagrad fused on-chip via
``tpu_sparse_dense_matmul_grad``. No gloo PG.

Gradient hooks in ``SparsecoreBackedEC`` fire ``SparseCoreLookup.
update_embeddings`` (on-chip Adagrad), so no PyTorch optimizer manages
embedding weights.

Model-agnostic. The model binding is responsible for computing the
per-feature per-host PRT-padded input sizes and handing them to
:meth:`SparseCoreEmbeddingBackend.set_feature_padded_input_sizes` before
:meth:`setup`, and for wiring a graph-break dispatcher against
:attr:`SparseCoreEmbeddingBackend.jax_ec_ref` (a mutable one-slot list that
``setup()`` fills, since ``install_graph`` fires after ``setup()``).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Sequence, Tuple

import jax
import torch
import torch.nn as nn
from recir.embedding._sparse_arch_stub import DummySparseArch, sparse_arch_marker_prefix
from recir.embedding.backend import (
    EmbeddingBackend,
    EmbeddingSetupResult,
    SparseOptimizerRuntime,
    SparseUpdatePlacement,
    swap_submodule,
)
from recir.embedding.sparsecore_ops import (
    _num_sc_per_device,
    init_sc_embedding_variables,
    SparsecoreBackedEC,
)
from recir.ops.eager_dispatch import install_graph_break_dispatch
from recir.training.optim import OptimConfig
from torchrec.modules.embedding_configs import BaseEmbeddingConfig, EmbeddingConfig
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor

logger: logging.Logger = logging.getLogger(__name__)


def _sequence_configs_only(
    table_configs: Sequence[BaseEmbeddingConfig],
) -> List[EmbeddingConfig]:
    """Narrow to sequence configs, refusing pooled ones with a named error."""
    pooled = [tc.name for tc in table_configs if not isinstance(tc, EmbeddingConfig)]
    if pooled:
        raise ValueError(
            f"SparseCore has no pooled-lookup path, but these tables are pooled: "
            f"{pooled}. Pooling is only implemented on the ring-shift path "
            f"(embedding_backend='in_graph')."
        )
    return [tc for tc in table_configs if isinstance(tc, EmbeddingConfig)]


class SparseCoreEmbeddingBackend(EmbeddingBackend):
    """SparseCore embedding backend using jax_tpu_embedding.

    Tables stored as EmbeddingVariables sharded on SparseCore HW.
    Lookup: tpu_sparse_dense_matmul (SparseCore).
    Optimizer: Adagrad fused on-chip via tpu_sparse_dense_matmul_grad.
    No gloo PG required.

    ``embedding_collection_fqn`` is the FQN of the model's torchrec
    ``EmbeddingCollection``. ``set_feature_padded_input_sizes()`` must run
    before ``setup()``.
    """

    def __init__(self, embedding_collection_fqn: str) -> None:
        self._embedding_collection_fqn = embedding_collection_fqn
        self._jax_ec: SparsecoreBackedEC | None = None
        # Shared mutable ref the model's sparse-arch dispatcher reads lazily
        # (install_graph fires after setup() has run).
        self._jax_ec_ref: List[SparsecoreBackedEC | None] = [None]
        self._feature_padded_input_sizes: Dict[str, int] = {}

    @property
    def jax_ec_ref(self) -> List[SparsecoreBackedEC | None]:
        return self._jax_ec_ref

    def set_feature_padded_input_sizes(self, sizes: Dict[str, int]) -> None:
        """Per-feature per-host PRT-padded total values, consumed by setup()."""
        self._feature_padded_input_sizes = sizes
        logger.info(
            f"SparseCoreEmbeddingBackend: per-feature SC input sizes "
            f"(per host) = {sizes}"
        )

    def init_pg(self) -> Tuple[int, int]:
        rank = jax.process_index()
        world = jax.process_count()
        logger.info(
            f"SparseCoreEmbeddingBackend: JAX PJRT PG (rank={rank}, world={world}), "
            f"no gloo init needed"
        )
        return rank, world

    def sc_modulus(self) -> int:
        return jax.local_device_count() * _num_sc_per_device()

    def setup(
        self,
        model: nn.Module,
        table_configs: Sequence[BaseEmbeddingConfig],
        sparse_config: OptimConfig,
        device: torch.device,
        world_size: int,
    ) -> EmbeddingSetupResult:
        if not self._feature_padded_input_sizes:
            raise RuntimeError(
                "SparseCoreEmbeddingBackend.set_feature_padded_input_sizes() "
                "must be called before setup()."
            )
        if device.type != "cpu":
            raise ValueError(f"JAX backend expects CPU host state, got {device}.")

        _, _, _, jax_ec = init_sc_embedding_variables(
            table_configs=_sequence_configs_only(table_configs),
            feature_padded_input_sizes=self._feature_padded_input_sizes,
            world_size=world_size,
            embedding_lr=sparse_config.lr,
        )
        self._jax_ec = jax_ec
        self._jax_ec_ref[0] = jax_ec
        logger.info("SparseCoreEmbeddingBackend: SparsecoreBackedEC initialized")
        return EmbeddingSetupResult(
            model=model,
            dense_named_parameters=self.dense_named_parameters(model),
            sparse_runtime=SparseOptimizerRuntime(
                placement=SparseUpdatePlacement.BACKEND_FUSED,
                config=sparse_config,
            ),
        )

    def dense_named_parameters(self, model: nn.Module) -> Dict[str, nn.Parameter]:
        # Embedding state lives in emb_vars on SparseCore; everything else is dense.
        return {
            name: p
            for name, p in model.named_parameters()
            if not name.startswith(self._embedding_collection_fqn + ".")
        }


class JAXSparseArchDispatcher:
    """SparseCore variant of :class:`recir.embedding.cpu.SparseArchDispatcher`.

    ``install_eager`` swaps the ``EmbeddingCollection`` at
    ``embedding_collection_fqn`` for a :class:`DummySparseArch` exactly like the
    eager path. ``install_graph`` instead wires a graph-break ``fwd_fn`` that builds
    a KJT from the runtime (values, lengths) and forwards into a
    :class:`SparsecoreBackedEC`, looked up through ``jax_ec_ref`` -- the mutable
    one-slot list that :meth:`SparseCoreEmbeddingBackend.setup` fills, since
    ``install_graph`` fires after ``setup()``.
    """

    def __init__(
        self,
        cached_output: Dict[str, JaggedTensor],
        input_kjt_keys: List[str],
        jax_ec_ref: List[SparsecoreBackedEC | None],
        embedding_collection_fqn: str,
        override_id: int = 1,
        num_ranks: int = 1,
    ) -> None:
        self._cached_output = cached_output
        self._input_kjt_keys = input_kjt_keys
        self._jax_ec_ref = jax_ec_ref
        self._embedding_collection_fqn = embedding_collection_fqn
        self._override_id = override_id
        self._num_ranks = num_ranks
        self._dummy: DummySparseArch | None = None

    def install_eager(self, model: nn.Module) -> None:
        self._dummy = DummySparseArch(
            self._cached_output,
            input_kjt_keys=self._input_kjt_keys,
            override_id=self._override_id,
        )
        swap_submodule(model, self._embedding_collection_fqn, self._dummy)

    def install_graph(
        self,
        gm: torch.fx.GraphModule,
        flat_state: List[torch.Tensor],
    ) -> Tuple[torch.fx.GraphModule, List[torch.Tensor]]:
        if self._dummy is None:
            return gm, flat_state

        input_kjt_keys: List[str] = self._input_kjt_keys
        feature_names: List[str] = self._dummy._feature_names
        num_ranks: int = self._num_ranks
        jax_ec_ref: List[SparsecoreBackedEC | None] = self._jax_ec_ref

        def fwd_fn(*input_tensors: torch.Tensor) -> Tuple[torch.Tensor, ...]:
            # PRT delivers values/lengths already aligned to FeatureSpec.input_shape,
            # so values.shape[0] == sum(lengths) by construction. Build the KJT
            # directly — no pad/narrow bridge, no D2H sync per step.
            kjt = KeyedJaggedTensor(
                keys=input_kjt_keys,
                values=input_tensors[0],
                lengths=input_tensors[1],
            )
            jax_ec = jax_ec_ref[0]
            assert jax_ec is not None, (
                "SparseCoreEmbeddingBackend.setup() must run before train_step()"
            )
            with torch.enable_grad():
                output = jax_ec(kjt)

            flat: List[torch.Tensor] = []
            for name in feature_names:
                jt = output[name]
                flat.append(jt.values())
                flat.append(jt.lengths())
            return tuple(flat)

        install_graph_break_dispatch(
            gm,
            sparse_arch_marker_prefix(self._override_id),
            fwd_fn,
            num_state_tensors=len(flat_state),
            num_ranks=num_ranks,
        )
        return gm, flat_state
