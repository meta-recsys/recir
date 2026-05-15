# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-ignore-all-errors

"""FakeTensor (Meta dispatch) registration for torchrec KJT operations.

torchrec ``EmbeddingCollection.forward`` calls ``KJT.to_dict()`` which routes
through ``length_per_key()`` -> ``_safe_tolist()`` -> ``tensor.tolist()``.
That call is data-dependent and breaks ``make_fx(tracing_mode="fake")`` with
``DataDependentOutputException: aten._local_scalar_dense.default``.

Strategy:
  1. Register a custom op ``recir_torchrec::kjt_split_per_feature`` whose
     real impl computes per-key index tensors (using ``.item()``, safe at
     execution time on real tensors) and whose fake (Meta) kernel emits
     fresh ``SymInt``-sized output tensors via ``new_dynamic_size()``.
  2. Replace ``EmbeddingCollection.forward`` with a version that calls the
     custom op for the KJT split, then calls ``nn.Embedding`` per feature.
     The per-feature ``aten.embedding(weight, indices)`` nodes remain
     visible in the traced graph so HybridParallel's
     ``rewrite_sparse_embeddings`` can shard them row-wise.

Usage:
    from recir.ops.fake.kjt import patch_ec_for_tracing
    patch_ec_for_tracing(model)
    # Now make_fx(tracing_mode="fake") can trace ECSparseArch end-to-end.

Limitations:
  * Assumes single stride per key (``KJT.variable_stride_per_key`` is
    False).  Asserted at patch-call time.
  * Assumes pooling=NONE (the ``EmbeddingCollection`` path; not
    ``EmbeddingBagCollection``).
"""

import logging
import types
from typing import Dict, List

import torch
import torch.nn as nn
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor

logger: logging.Logger = logging.getLogger(__name__)

_op_registered: bool = False


def _register_kjt_split_op() -> None:
    """Register ``recir_torchrec::kjt_split_per_feature`` (idempotent)."""
    global _op_registered
    if _op_registered:
        return
    _op_registered = True

    @torch.library.custom_op("recir_torchrec::kjt_split_per_feature", mutates_args=())
    def kjt_split_per_feature(
        values: torch.Tensor,
        lengths: torch.Tensor,
        stride: int,
        num_keys: int,
    ) -> List[torch.Tensor]:
        """Real impl: split a flat KJT values tensor into per-key index tensors.

        ``lengths`` is laid out key-major: ``lengths[k*stride + b]`` is the
        length of feature ``k`` in sample ``b``.  Per-key total is the sum of
        the corresponding stride-sized slice; per-key value range is
        ``values[per_key_offsets[k]:per_key_offsets[k+1]]``.

        Uses ``.item()`` to materialize the slice bounds.  Safe here: this
        path only runs at execution time on real tensors, never inside
        ``make_fx``; the fake kernel below replaces it during tracing.
        """
        per_key_total = lengths.view(num_keys, stride).sum(dim=1)
        per_key_offsets = torch.cat(
            [
                torch.zeros(1, dtype=per_key_total.dtype, device=per_key_total.device),
                per_key_total.cumsum(0),
            ]
        )
        out: List[torch.Tensor] = []
        for k in range(num_keys):
            start = int(per_key_offsets[k].item())
            end = int(per_key_offsets[k + 1].item())
            # Clone instead of slice-view: ``custom_op`` forbids any
            # output from aliasing an input.
            out.append(values[start:end].clone())
        return out

    @kjt_split_per_feature.register_fake
    def _(
        values: torch.Tensor,
        lengths: torch.Tensor,
        stride: int,
        num_keys: int,
    ) -> List[torch.Tensor]:
        """Fake kernel: emit one SymInt-sized output per key.

        Each per-feature length is data-dependent on ``lengths``, so we
        allocate a fresh dynamic-size symbol per output via
        ``torch.library.get_ctx().new_dynamic_size()``.  We then call
        ``torch._check_is_size`` on each symbol so PyTorch knows it
        represents a non-negative size, which lets downstream
        ``guard_size_oblivious`` predicates (e.g. ``s > 0`` inside
        ``switch_to_contiguous_if_needed``) evaluate optimistically
        instead of failing the strict ``torch._check`` default.
        Downstream ``aten.embedding`` propagates the symbol through
        the lookup.
        """
        ctx = torch.library.get_ctx()
        out: List[torch.Tensor] = []
        for _ in range(num_keys):
            sz = ctx.new_dynamic_size()
            torch._check_is_size(sz)
            out.append(values.new_empty(sz))
        return out


def _traceable_ec_forward(
    self: EmbeddingCollection,
    features: KeyedJaggedTensor,
) -> Dict[str, JaggedTensor]:
    """Replacement for ``EmbeddingCollection.forward`` that avoids ``to_dict``.

    Mirrors the original's iteration order and output contract:
    table-major over ``self.embeddings``, per-table iterating over
    ``_feature_names[i]`` and ``_embedding_names_by_table[i]`` in lockstep.
    Output dict is keyed by ``embedding_name`` (same as the original).

    KJT split goes through ``recir_torchrec::kjt_split_per_feature`` (with
    a SymInt fake kernel); per-feature ``nn.Embedding`` calls produce
    ``aten.embedding`` nodes visible to the tracer.
    """
    assert not features.variable_stride_per_key(), (
        "variable_stride_per_key KJTs are not supported by the traceable EC forward"
    )
    keys: List[str] = features.keys()
    stride: int = features.stride()
    num_keys: int = len(keys)
    flat_lengths: torch.Tensor = features.lengths()

    per_key_indices: List[torch.Tensor] = (
        torch.ops.recir_torchrec.kjt_split_per_feature(
            features.values(),
            flat_lengths,
            stride,
            num_keys,
        )
    )

    key_to_idx: Dict[str, int] = {k: i for i, k in enumerate(keys)}

    out: Dict[str, JaggedTensor] = {}
    for table_idx, emb_module in enumerate(self.embeddings.values()):
        feature_names: List[str] = self._feature_names[table_idx]
        embedding_names: List[str] = self._embedding_names_by_table[table_idx]
        for j, emb_name in enumerate(embedding_names):
            feat_name = feature_names[j]
            kjt_idx = key_to_idx[feat_name]
            indices = per_key_indices[kjt_idx]
            lookup = emb_module(indices)
            feat_lengths = flat_lengths[kjt_idx * stride : (kjt_idx + 1) * stride]
            out[emb_name] = JaggedTensor(values=lookup, lengths=feat_lengths)
    return out


def patch_ec_for_tracing(model: nn.Module) -> int:
    """Replace ``EmbeddingCollection.forward`` on every EC instance in ``model``.

    Registers the KJT-split custom op (idempotent), then walks the module
    tree and monkey-patches each ``EmbeddingCollection``'s ``forward`` to
    the FakeTensor-traceable variant.  Caches per-instance metadata
    (``_recir_patched`` flag) so re-invocation is a no-op.

    Returns the number of EC instances patched.
    """
    _register_kjt_split_op()

    patched = 0
    for fqn, mod in model.named_modules():
        if not isinstance(mod, EmbeddingCollection):
            continue
        if getattr(mod, "_recir_patched", False):
            continue
        mod.forward = types.MethodType(_traceable_ec_forward, mod)
        mod._recir_patched = True
        patched += 1
        logger.info("Patched EmbeddingCollection at %s for tracing", fqn)
    return patched
