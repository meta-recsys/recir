# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Custom embedding ops for the in-graph RankFM embedding path.

Defines the ``recir::*`` ops used by ``InGraphSparseArch``
(``recir/examples/rankfm/embedding/in_graph_embedding_backend.py``):

  * ``ringshift_embedding`` -- ROWWISE sequence lookup, lowered on the
    recir/torchax TPU path to the on-TensorCore ring-shift kernel (see
    ``recir/torchax/ops/jseq_embedding.py``) INSIDE the single compiled GSPMD
    program, so the cross-host all-to-all is scheduled with the outer
    data-parallel collectives and does NOT deadlock (unlike the graph-break
    path). Mirrors ``hammer.ops.pytorch.hstu_attn_op``.
  * ``fsdp_embedding`` -- FSDP (Shard(0)) dense lookup; the TPU lowering does an
    explicit all-gather (fwd) + reduce-scatter (bwd) of the table.
  * ``shard_batch0`` -- identity hint that pins an activation batch-sharded.

The DP (replicated) table path does NOT use a custom op -- it calls plain
``F.embedding``, which recir already lowers to ``jnp.take`` + the DP-aware
``aten.embedding_dense_backward``. Replicated tables are always host-initialized
(never meta), so plain ``aten.embedding`` traces cleanly for them and no
meta-tolerant custom op is needed.

``ringshift_embedding`` contract:
  ringshift_embedding(ids, lengths, weight) -> activations
    ids:     int32 [R, C]      ids reshaped to [W, max_local_ids] (leading axis =
                               mesh axis). The in-graph backend passes [W, T_local]
                               (one row per chip; T_local = per-chip token total).
    lengths: int32 [R]         valid count per row (entries past it are padding,
                               dropped from the ring-shift traffic). The in-graph
                               backend passes full counts (no padding).
    weight:  f32   [vocab, D]  embedding table; row-wise (FSDP/block) sharded by
                               recir's GSPMD lowering.
    returns: f32   [R, C, D]   per-position embeddings.

Only ``weight`` is differentiable; ``ids``/``lengths`` get ``None`` gradients.
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

_SCHEMA = "ringshift_embedding(Tensor ids, Tensor lengths, Tensor weight) -> Tensor"
_BWD_SCHEMA = (
    "ringshift_embedding_backward(Tensor grad_out, Tensor ids, Tensor lengths, "
    "Tensor weight) -> Tensor"
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


def _fwd_fallback(
    ids: torch.Tensor,
    lengths: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return F.embedding(ids.long(), weight)


def _bwd_fallback(
    grad_out: torch.Tensor,
    ids: torch.Tensor,
    lengths: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    wd = weight.detach().requires_grad_(True)
    with torch.enable_grad():
        out = F.embedding(ids.long(), wd)
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
) -> torch.Tensor:
    d = weight.shape[1]
    return ids.new_empty((*ids.shape, d), dtype=weight.dtype)


@torch.library.register_fake("recir::ringshift_embedding_backward")
def _bwd_meta(
    grad_out: torch.Tensor,
    ids: torch.Tensor,
    lengths: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(weight)


# pyre-ignore[2]: variadic ctx/inputs for torch autograd registration.
def _setup_context(ctx, inputs, output) -> None:  # noqa
    ids, lengths, weight = inputs
    ctx.save_for_backward(ids, lengths, weight)


# pyre-ignore[2,3]
def _backward(ctx, grad_out):  # noqa
    ids, lengths, weight = ctx.saved_tensors
    grad_weight = torch.ops.recir.ringshift_embedding_backward(
        grad_out, ids, lengths, weight
    )
    return None, None, grad_weight


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
