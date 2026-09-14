# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""RankFM bindings for the general recir embedding backends.

The backends live in ``recir.embedding`` and know nothing about hammer or
xsurface. This module supplies everything model-specific:

  * ``RANKFM_EMBEDDING_PARAM_PREFIX`` / ``rankfm_sparse_arch`` -- where the
    ``ECSparseArch`` and its embedding params sit inside ``BlueReelsVDDTrainV3``.
  * ``rankfm_sparse_optimizer`` -- hammer's LR-buffered optimizer factory.
  * a ``get_override`` per backend -- derives the traced shapes from one eager
    preprocessor + tokenizer pass and returns the tracing dispatchers. The
    dispatchers themselves live in ``rankfm_dispatchers``.

Selected at runtime via ``RankFMConfig.embedding_backend``
("cpu" | "tpu" | "in_graph") through :func:`build_embedding_backend`.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import jax
import torch
import torch.nn as nn
from minimal_viable_ai.core.model_family_api.optimizer_configs import OptimConfig
from recir.embedding.backend import EmbeddingBackend, table_plans_from_embedding_tables
from recir.embedding.cpu import CPUEmbeddingBackend, SparseArchDispatcher
from recir.embedding.in_graph import (
    InGraphEmbeddingBackend,
    SparseArchInputBreakDispatcher,
)
from recir.embedding.sparsecore import (
    JAXSparseArchDispatcher,
    SparseCoreEmbeddingBackend,
)
from recir.examples.rankfm_mvai.rankfm_dispatchers import (
    capture_sparse_arch_io,
    PreprocessorDispatcher,
    TEST_BATCH_SIZE,
)

logger: logging.Logger = logging.getLogger(__name__)

# Embedding parameters all live under this submodule path in the bare model.
RANKFM_EMBEDDING_PARAM_PREFIX: str = (
    "recblock.tokenizer_stack.mot_sparse_arch._sparse_arch._embedding_collection"
)


def rankfm_sparse_arch(model: nn.Module) -> nn.Module:
    """The ``ECSparseArch`` that owns ``_embedding_collection`` in RankFM."""
    # nn.Module.__getattr__ is typed Module | Tensor, so the chain needs Any.
    recblock: Any = model.recblock
    return recblock.tokenizer_stack.mot_sparse_arch._sparse_arch


def rankfm_sparse_optimizer(
    model: nn.Module,
    params: List[nn.Parameter],
    sparse_config: OptimConfig,
    device: torch.device,
) -> Tuple[torch.optim.Optimizer, torch.Tensor]:
    """hammer's LR-buffered optimizer over the in-graph table params.

    Imported lazily: ``recir_core`` pulls the full UDD config chain at import
    time, and the in_graph backend must stay network-free to import.
    """
    from hammer.v3.models.udd_fm_v3.recir_core.compile import LR_SPARSE_BUFFER
    from hammer.v3.models.udd_fm_v3.recir_core.setup import _build_lr_buffered_optimizer

    return _build_lr_buffered_optimizer(
        model, params, sparse_config, LR_SPARSE_BUFFER, device
    )


class CPURankFMBackend(CPUEmbeddingBackend):
    def get_override(
        self,
        recblock: Any,
        batch_sample: Any,
        device: torch.device,
        global_bs: int,
        num_ranks: int,
        num_processes: int,
    ) -> List[Any]:
        return build_eager_overrides(
            recblock,
            batch_sample,
            device,
            batch_size=global_bs,
            num_processes=num_processes,
        )


def build_eager_overrides(
    recblock: Any,
    sample_input: Any,
    device: torch.device,
    batch_size: int = TEST_BATCH_SIZE,
    num_processes: int = 1,
) -> List[Any]:
    """Dispatchers that graph-break back to the real eager ``ECSparseArch``.

    This is the eager/torchrec lookup path. The SparseCore and in-graph
    backends pair the same :class:`PreprocessorDispatcher` with their own
    sparse-arch dispatcher instead.

    Each returned dispatcher exposes ``install_eager(model)`` and
    ``install_graph(gm, flat_state)``.
    """
    input_kjt, cached_output, preproc_output = capture_sparse_arch_io(
        recblock, sample_input, capture_output=True
    )
    assert cached_output is not None

    # Pass the precomputed preprocessor output to the DummyPreprocessorStack
    # so it uses the SAME output that was used to capture sparse arch I/O.
    # Running the preprocessor twice can produce different outputs due to
    # module state mutation (e.g., running stats).
    return [
        PreprocessorDispatcher(
            device,
            override_id=0,
            batch_size=batch_size,
            sample_feature_bundle=sample_input,
            precomputed_output=preproc_output,
            num_processes=num_processes,
        ),
        SparseArchDispatcher(
            cached_output=cached_output,
            input_kjt_keys=input_kjt.keys(),
            sparse_arch_getter=rankfm_sparse_arch,
            override_id=1,
        ),
    ]


class JAXRankFMBackend(SparseCoreEmbeddingBackend):
    """SparseCore backend bound to RankFM.

    get_override() must run before setup() — it captures the per-feature
    PRT-padded input sizes from cached_input_kjt that setup() consumes.
    """

    def get_override(
        self,
        recblock: Any,
        batch_sample: Any,
        device: torch.device,
        global_bs: int,
        num_ranks: int,
        num_processes: int,
    ) -> List[Any]:
        # SparseArchDispatcher does not expose cached_input_kjt (only
        # cached_output + input_kjt_keys), and we need it to derive per-feature
        # SC FeatureSpec sizes. So run the same eager pass
        # build_eager_overrides uses, capture both I/O, then assemble master's
        # PreprocessorDispatcher plus the SparseCore sparse-arch dispatcher.
        cached_input_kjt, cached_output, preproc_output = capture_sparse_arch_io(
            recblock, batch_sample, capture_output=True
        )
        assert cached_output is not None
        input_kjt_keys: List[str] = cached_input_kjt.keys()

        # Per-feature per-host PRT-padded total values from cached_input_kjt.
        # The eager run was at global_bs, so per-host = global / num_processes.
        cached_lengths = cached_input_kjt.lengths()
        global_stride = len(cached_lengths) // len(input_kjt_keys)
        self.set_feature_padded_input_sizes(
            {
                k: int(
                    cached_lengths[i * global_stride : (i + 1) * global_stride]
                    .sum()
                    .item()
                )
                // num_processes
                for i, k in enumerate(input_kjt_keys)
            }
        )

        return [
            PreprocessorDispatcher(
                device,
                override_id=0,
                batch_size=global_bs,
                sample_feature_bundle=batch_sample,
                precomputed_output=preproc_output,
                num_processes=num_processes,
            ),
            JAXSparseArchDispatcher(
                cached_output=cached_output,
                input_kjt_keys=input_kjt_keys,
                jax_ec_ref=self.jax_ec_ref,
                sparse_arch_getter=rankfm_sparse_arch,
                override_id=1,
                num_ranks=num_ranks,
            ),
        ]


class InGraphRankFMBackend(InGraphEmbeddingBackend):
    """In-graph ring-shift backend bound to RankFM."""

    def get_override(
        self,
        recblock: Any,
        batch_sample: Any,
        device: torch.device,
        global_bs: int,
        num_ranks: int,
        num_processes: int,
    ) -> List[Any]:
        # The sparse arch is NOT dispatched here -- it is swapped in setup()
        # and traced in-graph. We only need the merged id-list KJT (for the
        # per-feature per-chip token total) and the precomputed preprocessor
        # output for the PreprocessorDispatcher.
        kjt, _, preproc_output = capture_sparse_arch_io(
            recblock, batch_sample, capture_output=False
        )
        keys: List[str] = kjt.keys()
        lengths = kjt.lengths()
        stride = len(lengths) // len(keys)  # = global batch B (eager run @ global_bs)
        world_size = jax.device_count()  # = ring-shift W (per-chip shard count)
        # Per-feature per-chip token total = (sum of that feature's lengths) / W.
        # Per-sample lengths may vary; only this per-chip total is static (the
        # data-pipeline invariant), so it is the shape we trace. Fail fast if a
        # feature's total is not divisible by W (invariant violated -> downstream
        # reshapes would silently mis-slice).
        feature_seqlens: Dict[str, int] = {}
        for i, k in enumerate(keys):
            total = int(lengths[i * stride : (i + 1) * stride].sum().item())
            assert total % world_size == 0, (
                f"feature {k!r} total tokens {total} not divisible by world_size "
                f"{world_size}; per-chip token total is not static"
            )
            feature_seqlens[k] = total // world_size
        self.set_feature_seqlens(feature_seqlens)

        return [
            PreprocessorDispatcher(
                device,
                override_id=0,
                batch_size=global_bs,
                sample_feature_bundle=batch_sample,
                precomputed_output=preproc_output,
                num_processes=num_processes,
                skip_backward_break=True,
            ),
            SparseArchInputBreakDispatcher(
                input_kjt_keys=keys,
                override_id=1,
                num_ranks=num_ranks,
                feature_seqlens=feature_seqlens,
            ),
        ]


def build_embedding_backend(
    embedding_backend: str,
    *,
    on_device_init: bool = True,
    embedding_tables: Optional[Dict[str, Any]] = None,
) -> EmbeddingBackend:
    if embedding_backend == "cpu":
        # The cpu (torchrec DMP) backend's sequence-embedding all-to-all needs
        # int64 offsets: int32 overflows / SEGVs in fbgemm.permute_2D_sparse_data
        # at large global batch (multi-host). The in_graph / tpu backends don't
        # use torchrec's a2a, so only patch here. See _torchrec_patches.
        from recir.embedding._torchrec_patches import apply_seq_a2a_int64_patch

        apply_seq_a2a_int64_patch()
        return CPURankFMBackend(
            embedding_param_prefix=RANKFM_EMBEDDING_PARAM_PREFIX,
            embedding_tables=embedding_tables,
        )
    if embedding_backend == "tpu":
        return JAXRankFMBackend()
    if embedding_backend == "in_graph":
        if embedding_tables is None:
            raise ValueError(
                "the in_graph backend needs embedding_tables: each table's "
                "sharding decides both its layout and which optimizer trains "
                "it, and there is no default to fall back on."
            )
        return InGraphRankFMBackend(
            sparse_arch_getter=rankfm_sparse_arch,
            sparse_optimizer_builder=rankfm_sparse_optimizer,
            table_plans=table_plans_from_embedding_tables(embedding_tables),
            on_device_init=on_device_init,
        )
    raise ValueError(
        f"Unknown embedding_backend: {embedding_backend!r}. "
        f"Use 'cpu', 'tpu', or 'in_graph'."
    )
