# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors

"""FakeTensor (Meta dispatch) registration for TorchRec KJT operations.

TorchRec ``EmbeddingCollection.forward`` and
``EmbeddingBagCollection.forward`` call ``KJT.to_dict()``, which routes through
``length_per_key()`` -> ``_safe_tolist()`` -> ``tensor.tolist()``. That call is
data-dependent and breaks ``make_fx(tracing_mode="fake")`` with
``DataDependentOutputException: aten._local_scalar_dense.default``.

Strategy:
  1. Register a custom op ``recir_torchrec::kjt_split_per_feature`` whose
     real impl computes per-key index tensors after one batched D2H of the split
     boundaries, and whose fake (Meta) kernel emits
     fresh ``SymInt``-sized output tensors via ``new_dynamic_size()``.
     EBC uses a companion variant that also returns per-feature offsets and
     optional weights. Fixed-stride offset sizes stay shape-derived.
  2. Replace collection forwards with versions that call the custom op for the
     KJT split, then call ``nn.Embedding`` or ``embedding_bag`` per feature.
     These lookup nodes remain visible in the traced graph so HybridParallel
     can shard them row-wise.

Usage:
    from recir.ops.fake.kjt import patch_ebc_for_tracing, patch_ec_for_tracing
    patch_ec_for_tracing(model)
    patch_ebc_for_tracing(model)
    # Now make_fx(tracing_mode="fake") can trace supported collection paths.

EBC contract:
  * Every feature has the same stride within one invocation, while that shared
    stride may change between invocations.
  * Forward and parameter backward are traceable for unweighted and weighted
    inputs. Eager gradients to per-sample weights are preserved.
  * Variable-stride inputs and inverse-index reconstruction are not supported.
"""

import logging
import types
from collections.abc import Callable
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
    process_pooled_embeddings,
    reorder_inverse_indices,
)
from torchrec.sparse.jagged_tensor import (
    _cuda_to_cpu_safe,
    JaggedTensor,
    KeyedJaggedTensor,
    KeyedTensor,
)
from torchrec.sparse.tensor_dict import maybe_td_to_kjt

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
        offsets: torch.Tensor,
        stride: int,
        num_keys: int,
    ) -> List[torch.Tensor]:
        """Real impl: split a flat KJT values tensor into per-key index tensors.

        ``offsets`` is the cumsum of lengths; slicing with stride yields per-key
        offsets: ``per_key_offsets[k] = offsets[k * stride]``.  Per-key value
        range is ``values[per_key_offsets[k]:per_key_offsets[k+1]]``.
        """
        # The graph may carry the stride captured during training. Derive the
        # runtime value from the actual offsets so serving requests with a
        # different batch size split correctly.
        runtime_stride = (offsets.numel() - 1) // num_keys
        per_key_offsets = offsets[::runtime_stride]
        # Citrine C6: batch the data-dependent boundaries into one D2H sync.
        host_offsets: List[int] = (
            _cuda_to_cpu_safe(per_key_offsets).tolist()
            if per_key_offsets.is_cuda
            else per_key_offsets.tolist()
        )
        out: List[torch.Tensor] = []
        for k in range(num_keys):
            start = host_offsets[k]
            end = host_offsets[k + 1]
            # Clone instead of slice-view: ``custom_op`` forbids any
            # output from aliasing an input.
            out.append(values[start:end].clone())
        return out

    @kjt_split_per_feature.register_fake
    def _(
        values: torch.Tensor,
        offsets: torch.Tensor,
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

    @torch.library.custom_op("recir_torchrec::kjt_lengths_per_feature", mutates_args=())
    def kjt_lengths_per_feature(
        offsets: torch.Tensor,
        num_keys: int,
    ) -> List[torch.Tensor]:
        """Split KJT lengths per feature using the runtime request stride."""
        runtime_stride = (offsets.numel() - 1) // num_keys
        flat_lengths = offsets[1:] - offsets[:-1]
        return [
            flat_lengths[k * runtime_stride : (k + 1) * runtime_stride].clone()
            for k in range(num_keys)
        ]

    @kjt_lengths_per_feature.register_fake
    def _(
        offsets: torch.Tensor,
        num_keys: int,
    ) -> List[torch.Tensor]:
        # Mirror the real impl's stride instead of minting an unbacked size: a
        # fresh symbol gets equated to the batch symbol downstream, and that
        # assert can land in a lowered subgraph that does not bind the batch.
        runtime_stride = (offsets.shape[0] - 1) // num_keys
        return [offsets.new_empty(runtime_stride) for _ in range(num_keys)]

    split_bags = torch.library.custom_op(
        "recir_torchrec::kjt_split_per_feature_with_offsets", mutates_args=()
    )(_kjt_split_per_feature_with_offsets_impl)
    split_bags.register_fake(_kjt_split_per_feature_with_offsets_fake)
    split_bags.register_autograd(
        _split_backward,
        setup_context=_setup_split_context,
    )


def _stride_per_key(lengths: torch.Tensor, num_keys: int) -> List[int]:
    total_bags = lengths.numel()
    if total_bags % num_keys != 0:
        raise ValueError("KJT lengths must contain the same stride for every key")
    return [total_bags // num_keys] * num_keys


def _kjt_split_per_feature_with_offsets_impl(
    values: torch.Tensor,
    lengths: torch.Tensor,
    weights: Optional[torch.Tensor],
    num_keys: int,
) -> List[torch.Tensor]:
    if num_keys <= 0:
        raise ValueError("KJT key count must be positive")
    if weights is not None and weights.numel() != values.numel():
        raise ValueError("KJT weights must contain one value per ID")

    stride_per_key = _stride_per_key(lengths, num_keys)
    per_key_lengths = torch.split(lengths, stride_per_key)
    value_count_tensor = lengths.reshape(num_keys, stride_per_key[0]).sum(dim=1)
    # Citrine C6: batch the data-dependent value counts into one D2H sync.
    value_counts: List[int] = (
        _cuda_to_cpu_safe(value_count_tensor).tolist()
        if value_count_tensor.is_cuda
        else value_count_tensor.tolist()
    )
    per_key_values = [part.clone() for part in torch.split(values, value_counts)]
    per_key_offsets = [
        torch.cat((key_lengths.new_zeros((1,)), key_lengths)).cumsum(dim=0)
        for key_lengths in per_key_lengths
    ]
    per_key_weights = (
        [part.clone() for part in torch.split(weights, value_counts)]
        if weights is not None
        else []
    )
    return [*per_key_values, *per_key_offsets, *per_key_weights]


def _kjt_split_per_feature_with_offsets_fake(
    values: torch.Tensor,
    lengths: torch.Tensor,
    weights: Optional[torch.Tensor],
    num_keys: int,
) -> List[torch.Tensor]:
    if num_keys <= 0:
        raise ValueError("KJT key count must be positive")
    ctx = torch.library.get_ctx()
    value_sizes = [ctx.new_dynamic_size() for _ in range(num_keys)]
    for size in value_sizes:
        torch._check_is_size(size)
    offset_sizes = [lengths.numel() // num_keys + 1] * num_keys
    outputs = [values.new_empty(size) for size in value_sizes]
    outputs.extend(lengths.new_empty(size) for size in offset_sizes)
    if weights is not None:
        outputs.extend(weights.new_empty(size) for size in value_sizes)
    return outputs


def _setup_split_context(
    ctx: Any,
    inputs: tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        int,
    ],
    output: List[torch.Tensor],
) -> None:
    del output
    _, _, weights, num_keys = inputs
    ctx.num_keys = num_keys
    ctx.has_weights = weights is not None


def _split_backward(
    ctx: Any,
    grad_outputs: List[torch.Tensor],
) -> tuple[None, None, Optional[torch.Tensor], None]:
    grad_weights = torch.cat(grad_outputs[-ctx.num_keys :]) if ctx.has_weights else None
    return None, None, grad_weights, None


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
    num_keys: int = len(keys)
    flat_lengths: torch.Tensor = features.lengths()
    # Derive stride from a traced tensor dimension. KJT.stride() is a Python
    # integer and make_fx would otherwise bake the training batch size into the
    # inference graph, making a graph captured at batch N fail at serving batch 1.
    stride = flat_lengths.size(0) // num_keys

    per_key_indices: List[torch.Tensor] = (
        torch.ops.recir_torchrec.kjt_split_per_feature(
            features.values(),
            features.offsets(),
            stride,
            num_keys,
        )
    )
    per_key_lengths: List[torch.Tensor] = (
        torch.ops.recir_torchrec.kjt_lengths_per_feature(
            features.offsets(),
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
            feat_lengths = per_key_lengths[kjt_idx]
            out[emb_name] = JaggedTensor(values=lookup, lengths=feat_lengths)
    return out


def _traceable_ebc_forward(
    self: EmbeddingBagCollection,
    features: KeyedJaggedTensor,
) -> KeyedTensor:
    """Mirror native EBC forward with only KJT decomposition replaced."""
    features = maybe_td_to_kjt(features, None)
    if features.variable_stride_per_key():
        raise NotImplementedError(
            "RecIR EBC tracing requires a fixed-stride KeyedJaggedTensor."
        )
    input_inverse_indices = features.inverse_indices_or_none()
    if input_inverse_indices is not None:
        raise NotImplementedError(
            "RecIR EBC tracing does not support inverse-index reconstruction."
        )
    flat_feature_names = [name for names in self._feature_names for name in names]
    inverse_indices = reorder_inverse_indices(
        inverse_indices=input_inverse_indices,
        feature_names=flat_feature_names,
    )
    keys: List[str] = features.keys()
    weights = features.weights() if self._is_weighted else None
    split_outputs: List[torch.Tensor] = (
        torch.ops.recir_torchrec.kjt_split_per_feature_with_offsets(
            features.values(),
            features.lengths(),
            weights,
            len(keys),
        )
    )
    per_key_indices = split_outputs[: len(keys)]
    per_key_offsets = split_outputs[len(keys) : 2 * len(keys)]
    per_key_weights = split_outputs[2 * len(keys) :]
    key_to_idx: Dict[str, int] = {key: index for index, key in enumerate(keys)}

    pooled_embeddings: List[torch.Tensor] = []
    for table_index, embedding_bag in enumerate(self.embedding_bags.values()):
        for feature_name in self._feature_names[table_index]:
            feature_index = key_to_idx[feature_name]
            pooled_embeddings.append(
                embedding_bag(
                    input=per_key_indices[feature_index],
                    offsets=per_key_offsets[feature_index],
                    per_sample_weights=(
                        per_key_weights[feature_index] if self._is_weighted else None
                    ),
                ).float()
            )

    return KeyedTensor(
        keys=self._embedding_names,
        values=process_pooled_embeddings(
            pooled_embeddings=pooled_embeddings,
            inverse_indices=inverse_indices,
        ),
        length_per_key=self._lengths_per_embedding,
    )


def _patch_collection_for_tracing(
    model: nn.Module,
    collection_type: type[nn.Module],
    forward: Callable[..., object],
) -> int:
    patched = 0
    for fqn, mod in model.named_modules():
        if not isinstance(mod, collection_type):
            continue
        if getattr(mod, "_recir_patched", False):
            continue
        mod.forward = types.MethodType(forward, mod)
        mod._recir_patched = True
        patched += 1
        logger.info("Patched %s at %s for tracing", type(mod).__name__, fqn)
    return patched


def patch_ec_for_tracing(model: nn.Module) -> int:
    """Replace TorchRec EC forwards with the FakeTensor-traceable variant.

    Registers the KJT-split custom op (idempotent), then walks the module
    tree and monkey-patches each ``EmbeddingCollection`` forward. Caches
    per-instance metadata so re-invocation is a no-op.

    Returns the number of EC instances patched.
    """
    _register_kjt_split_op()
    return _patch_collection_for_tracing(
        model,
        EmbeddingCollection,
        _traceable_ec_forward,
    )


def patch_ebc_for_tracing(model: nn.Module) -> int:
    """Replace TorchRec EBC forwards with the KJT-split traceable variant."""
    _register_kjt_split_op()
    return _patch_collection_for_tracing(
        model,
        EmbeddingBagCollection,
        _traceable_ebc_forward,
    )
