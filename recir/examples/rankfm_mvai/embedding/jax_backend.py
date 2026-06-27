# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""JAX/SparseCore embedding backend for RankFM.

Selected via ``RankFMConfig.embedding_backend == "tpu"`` through the
factory in ``embedding.py``.  Tables stored as SparseCore
EmbeddingVariables; lookup via ``tpu_sparse_dense_matmul``; Adagrad
fused on-chip via ``tpu_sparse_dense_matmul_grad``.  No gloo PG.

Architecture:
  - JAXSparseArchDispatcher is a standalone dispatcher (parallel to
    SparseArchDispatcher in xsurface_test_utils). install_eager swaps
    ECSparseArch for DummySparseArch like the base. install_graph wires
    a graph-break fwd_fn that calls SparsecoreBackedEC directly with
    the runtime KJT.
  - PreprocessorDispatcher (from xsurface_test_utils) is reused unchanged -
    preprocessing always runs on CPU.
  - Gradient hooks in SparsecoreBackedEC fire SparseCoreLookup.update_
    embeddings (on-chip Adagrad), so no PyTorch optimizer manages
    embedding weights.

Set seq_lens and global_batch_size on JAXRankFMBackend before calling
setup():

    backend.seq_lens = {k: cfg.max_uih_seq_len for k in ro_kjt_keys}
    backend.seq_lens.update({k: cfg.num_candidates for k in nro_kjt_keys})
    backend.global_batch_size = global_bs
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import jax
import torch
import torch.nn as nn
from hammer.v3.modules.mot.tokenizers.mot_sparse_arch import ECSparseArch
from recir.eager_dispatch import install_graph_break_dispatch
from recir.sparsecore.backend import (
    _num_sc_per_device,
    EmbeddingBackend,
    init_sc_embedding_variables,
    SparsecoreBackedEC,
)
from recir.tests.xla.xsurface_test_utils import (
    _disable_sampling,
    _xsurface_sparse_prefix,
    DummySparseArch,
    PreprocessorDispatcher,
)
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor

logger: logging.Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JAXSparseArchDispatcher: SparseCore runtime dispatch replacing ECSparseArch
# ---------------------------------------------------------------------------


class JAXSparseArchDispatcher:
    """Standalone dispatcher that routes the embedding lookup to SparseCore.

    Mirrors the structure of xsurface_test_utils.SparseArchDispatcher but
    calls into JAX/SparseCore (SparsecoreBackedEC) instead of the original
    torchrec ECSparseArch at runtime. install_eager swaps ECSparseArch for
    DummySparseArch the same way the base does; install_graph wires a
    graph-break fwd_fn that constructs a KJT from the runtime
    (values, lengths) tensors and forwards into the JAX EC (looked up via
    a mutable reference filled by JAXRankFMBackend.setup).
    """

    def __init__(
        self,
        cached_output: Dict[str, JaggedTensor],
        input_kjt_keys: List[str],
        jax_ec_ref: List[SparsecoreBackedEC | None],
        override_id: int = 1,
        num_ranks: int = 1,
    ) -> None:
        self._cached_output: Dict[str, JaggedTensor] = cached_output
        self._input_kjt_keys: List[str] = input_kjt_keys
        # Mutable list so setup() can fill [0] after get_override() returns.
        self._jax_ec_ref: List[SparsecoreBackedEC | None] = jax_ec_ref
        self._override_id: int = override_id
        self._num_ranks: int = num_ranks
        self._dummy: DummySparseArch | None = None

    def install_eager(self, model: nn.Module) -> None:
        for fqn, mod in list(model.named_modules()):
            if isinstance(mod, ECSparseArch):
                self._dummy = DummySparseArch(
                    self._cached_output,
                    input_kjt_keys=self._input_kjt_keys,
                    override_id=self._override_id,
                )
                parent_fqn, leaf = fqn.rsplit(".", 1) if "." in fqn else ("", fqn)
                parent = model.get_submodule(parent_fqn) if parent_fqn else model
                setattr(parent, leaf, self._dummy)

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
                "JAXRankFMBackend.setup() must run before train_step()"
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
            _xsurface_sparse_prefix(self._override_id),
            fwd_fn,
            num_state_tensors=len(flat_state),
            num_ranks=num_ranks,
        )
        return gm, flat_state


# ---------------------------------------------------------------------------
# JAXRankFMBackend: SparseCore embedding backend for RankFM
# ---------------------------------------------------------------------------


class JAXRankFMBackend(EmbeddingBackend):
    """SparseCore embedding backend for RankFM using jax_tpu_embedding.

    Tables stored as EmbeddingVariables sharded on SparseCore HW.
    Lookup: tpu_sparse_dense_matmul (SparseCore).
    Optimizer: Adagrad fused on-chip via tpu_sparse_dense_matmul_grad.
    No gloo PG required.

    get_override() must run before setup() — it captures the per-feature
    PRT-padded input sizes from cached_input_kjt that setup() consumes.
    """

    def __init__(self) -> None:
        self._jax_ec: SparsecoreBackedEC | None = None
        # Shared mutable ref passed to JAXSparseArchDispatcher so it can access
        # _jax_ec lazily (install_graph fires after setup() has run).
        self._jax_ec_ref: List[SparsecoreBackedEC | None] = [None]
        # Per-feature per-host PRT-padded total values, derived from the
        # cached_input_kjt captured in get_override(). Consumed by setup().
        self._feature_padded_input_sizes: Dict[str, int] = {}

    def init_pg(self) -> Tuple[int, int]:
        rank = jax.process_index()
        world = jax.process_count()
        logger.info(
            f"JAXRankFMBackend: JAX PJRT PG (rank={rank}, world={world}), "
            f"no gloo init needed"
        )
        return rank, world

    def sc_modulus(self) -> int:
        return jax.local_device_count() * _num_sc_per_device()

    def setup(
        self,
        model: nn.Module,
        table_configs: List[EmbeddingConfig],
        embedding_lr: float,
        world_size: int,
    ) -> nn.Module:
        if not self._feature_padded_input_sizes:
            raise RuntimeError(
                "JAXRankFMBackend.get_override() must be called before "
                "setup() so the per-feature PRT-padded input sizes are "
                "captured from the cached_input_kjt."
            )

        _, _, lookup, jax_ec = init_sc_embedding_variables(
            table_configs=table_configs,
            feature_padded_input_sizes=self._feature_padded_input_sizes,
            world_size=world_size,
            embedding_lr=embedding_lr,
        )
        self._jax_ec = jax_ec
        self._jax_ec_ref[0] = jax_ec
        logger.info("JAXRankFMBackend: SparsecoreBackedEC initialized")
        return model

    def dense_named_parameters(self, model: nn.Module) -> Dict[str, nn.Parameter]:
        # Embedding state lives in emb_vars on SparseCore; everything else is dense.
        return {
            name: p
            for name, p in model.named_parameters()
            if "_embedding_collection" not in name
        }

    def get_override(
        self,
        recblock: Any,
        batch_sample: Any,
        device: torch.device,
        global_bs: int,
        num_ranks: int,
        num_processes: int,
    ) -> List[Any]:
        # The new xsurface_test_utils.SparseArchDispatcher does not expose
        # cached_input_kjt (only cached_output + input_kjt_keys). We still
        # need cached_input_kjt to derive per-feature SC FeatureSpec sizes,
        # so run the same eager preprocessor + tokenizer pass _build_overrides
        # uses, capture both I/O via hooks, then assemble:
        #   - master's PreprocessorDispatcher for the preprocessor leg
        #   - our JAXSparseArchDispatcher for the sparse-arch leg
        # pyre-ignore[16]: dynamic attr from setattr in TokenizerStack
        sparse_arch = recblock.tokenizer_stack.mot_sparse_arch._sparse_arch
        captured: Dict[str, Any] = {}
        input_captured: Dict[str, Any] = {}

        def pre_hook(_module: nn.Module, input_args: Tuple[Any, ...]) -> None:
            input_captured["kjt"] = input_args[0]

        def hook(
            _module: nn.Module,
            _input: Any,
            output: Dict[str, JaggedTensor],
        ) -> None:
            captured["output"] = output

        pre_handle = sparse_arch.register_forward_pre_hook(pre_hook)
        handle = sparse_arch.register_forward_hook(hook)
        with torch.no_grad(), _disable_sampling(recblock.preprocessor_stack):
            preproc_output = recblock.preprocessor_stack(batch_sample)
            try:
                recblock.tokenizer_stack(preproc_output)
            except (RuntimeError, AttributeError):
                # Sparse-arch I/O already captured via hooks; downstream
                # errors (e.g. 0-dim MLP reshape from disabled features)
                # are expected.
                pass
        pre_handle.remove()
        handle.remove()
        assert "output" in captured, "Sparse arch forward hook did not fire"
        assert "kjt" in input_captured, "Sparse arch pre-hook did not fire"

        cached_input_kjt: KeyedJaggedTensor = input_captured["kjt"]
        cached_output: Dict[str, JaggedTensor] = captured["output"]
        input_kjt_keys: List[str] = cached_input_kjt.keys()

        # Per-feature per-host PRT-padded total values from cached_input_kjt.
        # The eager run was at global_bs, so per-host = global / num_processes.
        cached_lengths = cached_input_kjt.lengths()
        global_stride = len(cached_lengths) // len(input_kjt_keys)
        self._feature_padded_input_sizes = {
            k: int(
                cached_lengths[i * global_stride : (i + 1) * global_stride].sum().item()
            )
            // num_processes
            for i, k in enumerate(input_kjt_keys)
        }
        logger.info(
            f"JAXRankFMBackend: per-feature SC input sizes (per host) = "
            f"{self._feature_padded_input_sizes}"
        )

        preprocessor_dispatcher = PreprocessorDispatcher(
            device,
            override_id=0,
            batch_size=global_bs,
            sample_feature_bundle=batch_sample,
            precomputed_output=preproc_output,
            num_processes=num_processes,
        )
        jax_sparse = JAXSparseArchDispatcher(
            cached_output=cached_output,
            input_kjt_keys=input_kjt_keys,
            jax_ec_ref=self._jax_ec_ref,
            override_id=1,
            num_ranks=num_ranks,
        )
        return [preprocessor_dispatcher, jax_sparse]
