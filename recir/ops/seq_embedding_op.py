# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Custom embedding ops for the in-graph embedding path.

Defines the ``recir::*`` ops used by ``InGraphSparseArch``
(``recir/embedding/in_graph.py``):

  * ``ringshift_embedding`` -- ROWWISE sequence lookup, lowered on the
    recir/torchax TPU path to the on-TensorCore ring-shift kernel (see
    ``recir/torchax/ops/jseq_embedding.py``) INSIDE the single compiled GSPMD
    program, so the cross-host all-to-all is scheduled with the outer
    data-parallel collectives and does NOT deadlock (unlike the graph-break
    path).
  * ``fsdp_embedding`` -- FSDP (Shard(0)) dense lookup; the TPU lowering does an
    explicit all-gather (fwd) + reduce-scatter (bwd) of the table.
  * ``shard_batch0`` -- identity hint that pins an activation batch-sharded.

The DP (replicated) table path does NOT use a custom op -- it calls plain
``F.embedding``, which recir already lowers to ``jnp.take`` + the DP-aware
``aten.embedding_dense_backward``. Replicated tables are always host-initialized
(never meta), so plain ``aten.embedding`` traces cleanly for them and no
meta-tolerant custom op is needed.

``ringshift_embedding`` contract:
  ringshift_embedding(
      ids, lengths, weight, padding_idx=-1, pooling="none"
  ) -> activations
    ids:     int32 [R, C]      ids reshaped to [W, max_local_ids] (leading axis =
                               mesh axis). The in-graph backend passes [W, T_local]
                               (one row per chip; T_local = per-chip token total).
    weight:  [vocab, D]        embedding table; row-wise (FSDP/block) sharded by
                               recir's GSPMD lowering. BF16 tables use FP32 inside
                               the TPU kernel and cast the result/gradient back.
    padding_idx: int           nonnegative row whose gradient is suppressed, or
                               -1 for no padding row (matching ``F.embedding``).
    pooling: str               "none" (sequence, the default and the unchanged
                               behaviour), or "sum"/"mean" (EBC).

  lengths and the return shape depend on pooling:
    pooling="none"   lengths int32 [R]     valid count per row; entries past it
                                           are padding and are dropped from the
                                           ring-shift traffic
                     returns  f32 [R, C, D]
    pooling="sum"    lengths int32 [R, K]  row r's C ids cut into K consecutive
      or "mean"                            bags; bag k holds lengths[r, k] of
                                           them, so the row sums to C
                     returns  f32 [R, K, D]

``ids`` has ONE layout for both modes: a flat per-row id stream. Pooling only
adds structure to ``lengths`` -- where the cuts fall -- and correspondingly
gains an axis on the output.

This mirrors ``KeyedJaggedTensor`` (flat values plus lengths), which is also
what torchrec feeds ``nn.EmbeddingBag``, and what the kernel consumes. A padded
[bags, max_bag] rectangle would need a static bag width nothing here can derive,
would be compacted straight back to this layout before reaching the kernel, and
would move max_bag/avg_bag times more ids to do it.

Bags do not cross rows. Row r is one chip's tokens, so its bags are that chip's
samples, and the reduction never needs data from another chip.

Only ``weight`` is differentiable; ``ids``/``lengths``/``padding_idx``/
``pooling`` get ``None`` gradients.
The CPU/GPU fallback is a plain ``F.embedding`` so the op is correct off-TPU and
opaque to ``make_fx`` (CompositeExplicitAutograd), matching the HSTU op.

Both ``ringshift_embedding`` and ``fsdp_embedding`` are META-TOLERANT: their fake
kernels place the forward output on the ids (compute) device and the table grad
on the weight's device, so a table with on-device (meta) init does not break
recir's fake-tensor trace (plain aten.embedding raises "Unhandled FakeTensor
Device Propagation ... meta, cpu").
"""

import torch
import torch.nn.functional as F

_LIB = torch.library.Library("recir", "FRAGMENT")

_SCHEMA = (
    "ringshift_embedding(Tensor ids, Tensor lengths, Tensor weight, "
    'int padding_idx=-1, str pooling="none") -> Tensor'
)
_BWD_SCHEMA = (
    "ringshift_embedding_backward(Tensor grad_out, Tensor ids, Tensor lengths, "
    'Tensor weight, int padding_idx=-1, str pooling="none") -> Tensor'
)

_LIB.define(_SCHEMA)
_LIB.define(_BWD_SCHEMA)

# Identity op that hints the activation must be batch-sharded on dim 0 (mesh axis
# "torch_dist"). Eager/CPU = identity; the torchax lowering applies
# with_sharding_constraint. Needed because constant activations (e.g. a zeros
# embedding for a skipped table) otherwise constant-fold to a REPLICATED scalar
# broadcast -> materialized full per device -> XLA SPMD partitioner segfault.
_SHARD_SCHEMA = "shard_batch0(Tensor x) -> Tensor"
_LIB.define(_SHARD_SCHEMA)


def _shard_batch0_fallback(x: torch.Tensor) -> torch.Tensor:
    return x


_LIB.impl("shard_batch0", _shard_batch0_fallback, "CompositeExplicitAutograd")


@torch.library.register_fake("recir::shard_batch0")
def _shard_batch0_meta(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


# pyre-ignore
def _shard_batch0_backward(ctx, grad_out: torch.Tensor) -> torch.Tensor:  # noqa
    return grad_out


torch.library.register_autograd("recir::shard_batch0", _shard_batch0_backward)


def _fwd_ref(
    ids: torch.Tensor,
    lengths: torch.Tensor,
    weight: torch.Tensor,
    padding_idx: int,
    pooling: str,
) -> torch.Tensor:
    """Reference forward, shared by the CPU fallback and its backward.

    ``pooling="none"`` keeps one row per id (sequence). Otherwise each row's
    ids are cut into consecutive bags by ``lengths`` and each bag is reduced --
    ``nn.EmbeddingBag`` semantics for flat values plus offsets.

    Shapes (matching the contract at the top of this module)::

        ids      [R, C]        R rows, C ids each
        lengths  [R]           pooling="none": valid slots, rest is padding
                 [R, K]        pooled: the cut points, summing to C per row
        weight   [vocab, D]
        returns  [R, C, D]     pooling="none"
                 [R, K, D]     pooled

        emb      [R, C, D]     F.embedding
        seg      [R, C]        which bag each id falls in, via searchsorted
        out      [R, K, D]     scatter-add of emb into seg
    """
    if pooling not in ("none", "sum", "mean"):
        raise ValueError(f"unknown pooling mode: {pooling!r}")

    emb = F.embedding(
        ids.long(),
        weight,
        padding_idx=None if padding_idx < 0 else padding_idx,
    )
    if pooling == "none":
        return emb
    mean_lengths = lengths
    if padding_idx >= 0:
        valid = (ids != padding_idx).to(emb.dtype).unsqueeze(-1)
        emb = emb * valid
        if pooling == "mean":
            mean_lengths = _reduce_into_bags(valid, lengths, "sum").squeeze(-1)
    return _reduce_into_bags(emb, lengths, pooling, mean_lengths=mean_lengths)


def _reduce_into_bags(
    emb: torch.Tensor,
    lengths: torch.Tensor,
    pooling: str,
    *,
    mean_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sum ``emb`` [R, C, D] into [R, K, D] along the cuts given by lengths.

    ``searchsorted`` over the inclusive scan gives each id its bag in one
    batched op: the number of bag ends at or before position c IS c's bag
    index. Ids past the row's total land on index K, one past the last bag --
    hence the extra slot, which is allocated, scattered into, and dropped. That
    slot only receives anything if a row's lengths sum to less than C; keeping
    it means such a row degrades to ignoring its tail rather than corrupting
    the last bag.
    """
    rows, cols, dim = emb.shape
    n_bags = lengths.shape[-1]
    ends = lengths.cumsum(-1)
    positions = torch.arange(cols, device=emb.device).expand(rows, cols)
    seg = torch.searchsorted(ends.contiguous(), positions.contiguous(), right=True)

    out = torch.zeros(rows, n_bags + 1, dim, dtype=emb.dtype, device=emb.device)
    out = out.scatter_add(1, seg.unsqueeze(-1).expand(rows, cols, dim), emb)
    out = out[:, :n_bags, :]
    if pooling == "mean":
        # Clamp so an empty bag yields a zero row rather than a NaN.
        divisor = lengths if mean_lengths is None else mean_lengths
        out = out / divisor.clamp(min=1).unsqueeze(-1).to(out.dtype)
    return out


def _fwd_fallback(
    ids: torch.Tensor,
    lengths: torch.Tensor,
    weight: torch.Tensor,
    padding_idx: int = -1,
    pooling: str = "none",
) -> torch.Tensor:
    return _fwd_ref(ids, lengths, weight, padding_idx, pooling)


def _bwd_fallback(
    grad_out: torch.Tensor,
    ids: torch.Tensor,
    lengths: torch.Tensor,
    weight: torch.Tensor,
    padding_idx: int = -1,
    pooling: str = "none",
) -> torch.Tensor:
    # Differentiating the reference forward, so the pooled backward (fan the bag
    # gradient back out to its members, scaled for mean) is derived rather than
    # written twice.
    wd = weight.detach().requires_grad_(True)
    with torch.enable_grad():
        out = _fwd_ref(ids, lengths, wd, padding_idx, pooling)
        (grad_weight,) = torch.autograd.grad(out, (wd,), grad_out)
    return grad_weight


# CompositeExplicitAutograd: opaque to autograd (so make_fx keeps it as one op);
# gradients come from the registered autograd below.
_LIB.impl("ringshift_embedding", _fwd_fallback, "CompositeExplicitAutograd")
_LIB.impl("ringshift_embedding_backward", _bwd_fallback, "CompositeExplicitAutograd")


@torch.library.register_fake("recir::ringshift_embedding")
def _fwd_meta(
    ids: torch.Tensor,
    lengths: torch.Tensor,
    weight: torch.Tensor,
    padding_idx: int = -1,
    pooling: str = "none",
) -> torch.Tensor:
    if pooling not in ("none", "sum", "mean"):
        raise ValueError(f"unknown pooling mode: {pooling!r}")
    d = weight.shape[1]
    # Sequence keeps one row per id; pooling replaces the id axis with the bag
    # axis, whose length is only knowable from lengths.
    shape = (*ids.shape, d) if pooling == "none" else (*lengths.shape, d)
    return ids.new_empty(shape, dtype=weight.dtype)


@torch.library.register_fake("recir::ringshift_embedding_backward")
def _bwd_meta(
    grad_out: torch.Tensor,
    ids: torch.Tensor,
    lengths: torch.Tensor,
    weight: torch.Tensor,
    padding_idx: int = -1,
    pooling: str = "none",
) -> torch.Tensor:
    return torch.empty_like(weight)


# pyre-ignore[2]: variadic ctx/inputs for torch autograd registration.
def _setup_context(ctx, inputs, output) -> None:  # noqa
    ids, lengths, weight, padding_idx, pooling = inputs
    ctx.save_for_backward(ids, lengths, weight)
    ctx.padding_idx = padding_idx
    # A str is not a tensor, so it rides on ctx rather than save_for_backward.
    ctx.pooling = pooling


# pyre-ignore[2,3]
def _backward(ctx, grad_out):  # noqa
    ids, lengths, weight = ctx.saved_tensors
    grad_weight = torch.ops.recir.ringshift_embedding_backward(
        grad_out,
        ids,
        lengths,
        weight,
        ctx.padding_idx,
        ctx.pooling,
    )
    # One entry per forward input. The non-tensor options get no gradients.
    return None, None, grad_weight, None, None


torch.library.register_autograd(
    "recir::ringshift_embedding", _backward, setup_context=_setup_context
)


# ---------------------------------------------------------------------------
# recir::fsdp_embedding -- FSDP (Shard(0)) dense embedding lookup.
#
# CPU/GPU behavior is a plain dense gather (F.embedding), but its TPU lowering
# (jseq_embedding.py) does an EXPLICIT FSDP all-gather (fwd) of the Shard(0)
# weight + reduce-scatter (bwd) of the table grad -- guaranteeing true FSDP
# collectives (full table materialized only transiently inside fwd/bwd,
# persistent storage stays sharded) instead of GSPMD's all-reduce reshard
# emulation. Used for TableMode.FSDP. The fake kernels are META-TOLERANT (output
# on the ids device; grad on the weight device) so an FSDP table can keep
# on-device (meta) init -- plain aten.embedding would crash the fake-tensor trace
# with a meta weight ("FakeTensor Device Propagation ... meta, cpu").
# ---------------------------------------------------------------------------
def _dense_embedding_fallback(
    ids: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return F.embedding(ids.long(), weight)


def _dense_embedding_backward_fallback(
    grad_out: torch.Tensor,
    ids: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.aten.embedding_dense_backward(
        grad_out, ids, weight.shape[0], -1, False
    )


_LIB.define("fsdp_embedding(Tensor ids, Tensor weight) -> Tensor")
_LIB.define(
    "fsdp_embedding_backward(Tensor grad_out, Tensor ids, Tensor weight) -> Tensor"
)

_LIB.impl("fsdp_embedding", _dense_embedding_fallback, "CompositeExplicitAutograd")
_LIB.impl(
    "fsdp_embedding_backward",
    _dense_embedding_backward_fallback,
    "CompositeExplicitAutograd",
)


@torch.library.register_fake("recir::fsdp_embedding")
def _dense_embedding_meta(
    ids: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    d = weight.shape[-1]
    return ids.new_empty((*ids.shape, d), dtype=weight.dtype)


@torch.library.register_fake("recir::fsdp_embedding_backward")
def _dense_embedding_backward_meta(
    grad_out: torch.Tensor,
    ids: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(weight)


# pyre-ignore[2]
def _fsdp_setup_context(ctx, inputs, output) -> None:  # noqa
    ids, weight = inputs
    ctx.save_for_backward(ids, weight)


# pyre-ignore[2,3]
def _fsdp_emb_backward(ctx, grad_out):  # noqa
    ids, weight = ctx.saved_tensors
    grad_weight = torch.ops.recir.fsdp_embedding_backward(grad_out, ids, weight)
    return None, grad_weight


torch.library.register_autograd(
    "recir::fsdp_embedding", _fsdp_emb_backward, setup_context=_fsdp_setup_context
)
