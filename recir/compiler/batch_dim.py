# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Forward-propagate is_batch_input/batch_dim across an FX graph.

`_tag_batch_inputs` (in `recir/compiler/tracing.py`) only marks top-level placeholders with
`is_batch_input` / `batch_dim`. After `segment_and_lower` cuts the gm at every
graph break, intermediate FX nodes that cross subgraph boundaries become NEXT
subgraph's placeholders. Without `is_batch_input` tags on those intermediate
nodes, `_jit_gspmd` defaults to `replicated` sharding for them and JAX sees
per-process shape (instead of global) - producing broadcast-shape mismatches
at downstream elementwise ops that mix per-process and global tensors.

`propagate_batch_input_meta` walks the FX gm in topological order and tags each
call_function node it can. Two rule tables drive the decision:

  * `_output_batch_dim_for(target, input_batch_dim, node)` for ops where any
    batched input determines the output batch_dim.
  * `_output_batch_dim_for_positional(node)` for matmul-family ops where the
    batch axis depends on which operand carries it.
"""

import operator
from typing import Any, cast, List, Optional

import torch
from torch import fx


def _output_batch_dim_for(  # noqa: C901 -- centralized operator semantics table
    target: Any, input_batch_dim: int, node: fx.Node
) -> Optional[int]:
    """Compute the output batch_dim of `node` given an input batch_dim.

    Conservative policy: only ops in the explicit safe list propagate.
    Unknown ops return None, leaving the output untagged (so the next
    subgraph boundary will use replicated sharding for that tensor).
    Unknown-but-batch-preserving ops show up as broadcast errors at JAX
    lowering and we add them here.
    """
    aten = torch.ops.aten

    # 2D transpose: aten.t / aten.t_copy. Only valid for 2D tensors;
    # batch_dim must be 0 or 1 and gets swapped.
    if target in (aten.t.default, getattr(aten, "t_copy", None)):
        if input_batch_dim == 0:
            return 1
        if input_batch_dim == 1:
            return 0
        return None

    # transpose(input, dim0, dim1) / transpose_copy
    if target in (
        aten.transpose.int,
        getattr(aten.transpose_copy, "int", None),
    ):
        dim0 = cast(int, node.args[1]) if len(node.args) > 1 else 0
        dim1 = cast(int, node.args[2]) if len(node.args) > 2 else 1
        # Normalize negative dims so comparison with input_batch_dim
        # (always non-negative) works correctly.
        if dim0 < 0 or dim1 < 0:
            val = node.meta.get("val")
            if hasattr(val, "ndim"):
                ndim = val.ndim
                if dim0 < 0:
                    dim0 = dim0 + ndim
                if dim1 < 0:
                    dim1 = dim1 + ndim
            else:
                return None
        if input_batch_dim == dim0:
            return dim1
        if input_batch_dim == dim1:
            return dim0
        return input_batch_dim

    # permute(input, perm)
    if target in (aten.permute.default, aten.permute_copy.default):
        perm = node.args[1]
        if isinstance(perm, (list, tuple)) and input_batch_dim in perm:
            return cast(List[int], perm).index(input_batch_dim)
        return None

    # unsqueeze(input, dim): adds an axis at position dim.
    if target in (aten.unsqueeze.default, aten.unsqueeze_copy.default):
        dim = cast(int, node.args[1])
        if dim < 0:
            val = node.meta.get("val")
            if hasattr(val, "ndim"):
                dim = val.ndim + dim
        if dim <= input_batch_dim:
            return input_batch_dim + 1
        return input_batch_dim

    # squeeze(dim): removes a unit-size axis at dim.
    if target in (aten.squeeze.dim, aten.squeeze_copy.dim):
        dim = cast(int, node.args[1])
        if dim < 0:
            return None
        if dim == input_batch_dim:
            return None
        if dim < input_batch_dim:
            return input_batch_dim - 1
        return input_batch_dim

    # Axis-preserving ops: batch_dim unchanged.
    axis_preserving = {
        # arithmetic
        aten.add.Tensor,
        aten.add.Scalar,
        aten.sub.Tensor,
        aten.sub.Scalar,
        aten.rsub.Tensor,
        aten.rsub.Scalar,
        aten.mul.Tensor,
        aten.mul.Scalar,
        aten.div.Tensor,
        aten.div.Scalar,
        aten.div.Tensor_mode,
        aten.remainder.Tensor,
        aten.remainder.Scalar,
        aten.neg.default,
        aten.abs.default,
        aten.exp.default,
        aten.log.default,
        aten.log1p.default,
        aten.sqrt.default,
        aten.rsqrt.default,
        aten.reciprocal.default,
        aten.pow.Tensor_Scalar,
        aten.pow.Tensor_Tensor,
        # activations / normalization / softmax
        aten.relu.default,
        aten.sigmoid.default,
        aten.tanh.default,
        aten.gelu.default,
        aten.silu.default,
        aten.softmax.int,
        aten._softmax.default,
        aten.log_softmax.int,
        aten._log_softmax.default,
        aten.layer_norm.default,
        aten.native_layer_norm.default,
        aten.dropout.default,
        aten.native_dropout.default,
        # comparisons
        aten.eq.Tensor,
        aten.eq.Scalar,
        aten.ne.Tensor,
        aten.ne.Scalar,
        aten.lt.Tensor,
        aten.lt.Scalar,
        aten.gt.Tensor,
        aten.gt.Scalar,
        aten.le.Tensor,
        aten.le.Scalar,
        aten.ge.Tensor,
        aten.ge.Scalar,
        # min/max elementwise + clamp + where
        aten.maximum.default,
        aten.minimum.default,
        aten.clamp.default,
        aten.clamp_min.default,
        aten.clamp_max.default,
        aten.where.self,
        # bitwise
        aten.bitwise_and.Tensor,
        aten.bitwise_or.Tensor,
        aten.bitwise_xor.Tensor,
        aten.bitwise_not.default,
        # slice / narrow - dim-preserving
        aten.slice.Tensor,
        aten.narrow.default,
        aten.narrow_copy.default,
        # casting / no-op
        aten.contiguous.default,
        aten.clone.default,
        aten.detach.default,
        aten._to_copy.default,
        aten.to.dtype,
        aten.to.dtype_layout,
        # bucketize / sort / topk - shape-preserving along batch axis
        aten.bucketize.Tensor,
        aten.zeros_like.default,
        aten.ones_like.default,
        aten.empty_like.default,
        aten.full_like.default,
        # masking / fill (preserve shape)
        aten.masked_fill.Scalar,
        aten.masked_fill.Tensor,
        # activation / softmax backwards: same shape as the corresponding
        # forward, batch dim preserved.
        aten.gelu_backward.default,
        aten._softmax_backward_data.default,
        aten._log_softmax_backward_data.default,
        # nll_loss_backward(grad_output, self, target, weight, reduction,
        # ignore_index, total_weight) -> Tensor with shape == self.shape.
        # The batched ``self`` arg (the predictions tensor, [N, C]) and
        # batched ``target`` arg ([N,]) both carry batch_dim=0; the
        # consolidator picks 0 and the output preserves it.
        aten.nll_loss_backward.default,
        # slice_backward(grad, input_sizes, dim, start, end, step):
        # scatters grad into a zero-filled tensor of shape ``input_sizes``.
        # As long as the slice was NOT along the batch axis (typical engram
        # / Gemma3 pattern: slicing seq or head dims), batch_dim is
        # preserved. Reductions over the batch axis are extremely rare in
        # the canary so this passthrough is safe in practice.
        aten.slice_backward.default,
        # Pure-view alias (``aten::alias(Tensor(a) self) -> Tensor(a)``):
        # returns a view of the same storage with the same shape. Inserted
        # by autograd on detach-style paths; batch_dim is unchanged.
        aten.alias.default,
        # Pure elementwise unary ops emitted by engram's RoPE / sign /
        # ngram-hash paths. None of these change shape, so the batch_dim
        # carries through unchanged. Without these handlers each call
        # drops is_batch_input across a graph-break boundary and the
        # downstream subgraph sees REPLICATED -> JAX broadcast mismatch.
        aten.sign.default,
        aten.cos.default,
        aten.sin.default,
        # In-place ops (``Tensor(a!) self ... -> Tensor(a!)``): self is
        # arg 0, the output IS self, batch_dim is preserved. Functional
        # behavior is identical to the corresponding out-of-place op for
        # propagation purposes (the consolidator walks all_input_nodes
        # and picks self's batch_dim; any other tensor args -- like
        # lerp_'s `end` or addcmul_'s `tensor1`/`tensor2` -- are usually
        # unbatched accumulators, but if batched they must agree with
        # self on batch_dim or the consolidator bails). These are the
        # 12 add_.Scalar + 1 lerp_.Scalar + 1 addcmul_ (+ addcdiv_ from
        # the optimizer step) nodes that broke propagation on engram.
        aten.add_.Scalar,
        aten.add_.Tensor,
        aten.sub_.Scalar,
        aten.sub_.Tensor,
        aten.mul_.Scalar,
        aten.mul_.Tensor,
        aten.div_.Scalar,
        aten.div_.Tensor,
        aten.lerp_.Scalar,
        aten.addcmul_.default,
        aten.addcdiv_.default,
    }
    if target in axis_preserving:
        return input_batch_dim

    # matmul-family ops are handled via _output_batch_dim_for_positional
    # in propagate_batch_input_meta because the batch axis depends on
    # WHICH operand carries the batch tag (mat1 vs mat2) and which axis.

    # cat / stack on a non-batch axis preserve batch_dim.
    # cat(tensors, dim): rank is preserved, so the batch axis comes through.
    #
    # NOTE: this does not check the concat dim. Concatenating ALONG the batch
    # axis is common on the jagged paths (RO and NRO token blocks), and leaving
    # those untagged would replicate a [global_batch*seq, D] activation on every
    # device -- the failure the ringshift note below describes. Keeping the tag
    # is the deliberate choice; revisit only with a measurement.
    if target is aten.cat.default:
        return input_batch_dim

    # stack(tensors, dim) INSERTS a new axis, so it is NOT cat: every axis at or
    # after ``dim`` shifts right by one, and the batch axis moves with them.
    #
    # Sharing cat's rule here (returning input_batch_dim unchanged) mislabels
    # the INSERTED axis as the batch one. That is how a per-task stack of
    # [batch] vectors can end up claiming batch_dim=0 on their task axis. It
    # then reaches a graph break, _jit_gspmd builds P('torch_dist', None) from
    # the tag, and jax rejects it: "array axis 0 is partitioned 8 times, but the
    # dimension size is 105". Worse when the task count HAPPENS to divide the
    # mesh -- then it silently shards the task axis instead of the batch, and
    # every device computes a slice of the tasks over the whole batch.
    if target is aten.stack.default:
        dim = _normalize_sequence_dim(node, rank_delta=1)
        if dim is None:
            return None
        return input_batch_dim + 1 if dim <= input_batch_dim else input_batch_dim

    # embedding lookup: indices (batch_dim) -> (..., D), batch_dim preserved.
    if target == aten.embedding.default:
        return input_batch_dim

    # recir::ringshift_embedding(ids[B,S], lengths[B], weight[V,D]) -> [B,S,D]: the custom
    # ring-shift embedding op. Like aten.embedding, the batched ids' batch_dim is
    # preserved at the output (dim 0). WITHOUT this rule the op is not in the safe
    # list, so its [B,S,D] activation is left untagged -> _jit_gspmd defaults it to
    # REPLICATED -> a [global_batch*seq, D] tensor is materialized per device
    # (~tens of GB) -> XLA SPMD partitioner segfault. (String match avoids importing
    # the op def into recir core / registration-order issues.)
    # NOTE: only the FORWARD is tagged. The backward returns grad_weight[V,D] whose
    # dim 0 is the VOCAB (not batch) and is FSDP-sharded via _find_fsdp_source;
    # tagging it batch_dim=0 would wrongly batch-shard the vocab dim.
    if str(target) == "recir.ringshift_embedding.default":
        return input_batch_dim

    # recir::fsdp_embedding(ids[B*S], weight[V,D]) -> [B*S, D]: the FSDP dense
    # lookup custom op. Like aten.embedding, the batched ids' batch_dim (dim 0) is
    # preserved at the output. It is a custom op (not covered by the aten.embedding
    # rule above), so tag it here too; otherwise its activation is left untagged ->
    # REPLICATED -> per-device full materialization -> SPMD segfault. As with
    # ring-shift, only the FORWARD is tagged (the backward grad_weight[V,D] dim 0 is
    # the VOCAB, FSDP-sharded via _find_fsdp_source).
    if str(target) == "recir.fsdp_embedding.default":
        return input_batch_dim

    # The Helion jagged HSTU attention ops: fwd(q[L,H,D], k, v, ...) -> out[L,H,D],
    # and the functional bwd(...) -> (dq, dk, dv), each [L,H,D]. L (total tokens) is
    # the batch axis (dim 0), preserved from q/k/v exactly like the eager attention
    # subgraph they replace. WITHOUT these rules the opaque ops are left untagged ->
    # _jit_gspmd defaults them to REPLICATED (per-process L) while the batch-sharded
    # consumers use the global L -> aten.mul broadcast mismatch.
    if str(target) in (
        "recir.helion_ragged_attention_fwd.default",
        "recir.helion_ragged_attention_bwd.default",
    ):
        return input_batch_dim

    # getitem(recir.helion_ragged_attention_bwd(...), i) -> dq/dk/dv[L,H,D]:
    # propagate the bwd op's batch_dim onto each extracted grad (getitem is not
    # otherwise in the safe list).
    if target is operator.getitem:
        src = node.args[0]
        if (
            isinstance(src, fx.Node)
            and str(src.target) == "recir.helion_ragged_attention_bwd.default"
        ):
            return input_batch_dim

    # convolution(input, weight, ...): input is [N, C, *spatial]; the conv
    # walks the spatial dims and emits [N, C_out, *spatial_out]. As long as
    # input_batch_dim points at the N axis (the typical case), the output
    # batch axis is unchanged. Used by ShortConv inside engram (depthwise
    # conv1d along the seq dim with batch on axis 0).
    if target in (aten.convolution.default, aten._convolution.default):
        return input_batch_dim

    # embedding_dense_backward(grad_output, indices, num_weights, padding_idx,
    # scale_grad_by_freq) -> grad_weight[V, D]. Dim 0 is the VOCAB axis, NOT the
    # batch axis (the op reduces over grad_output's batch axis), so it must stay
    # untagged -> replicated. Tagging batch_dim=0 makes _to_sharded_jax shape-
    # shard the vocab along the batch mesh axis: a REPLICATED dense embedding
    # then gets its grad inflated to [V*host_count, D] while its replicated AdamW
    # state stays [V, D] -> broadcast crash at the optimizer lerp under multi-host.
    if target == aten.embedding_dense_backward.default:
        return None

    # cumsum(input, dim): preserves shape. Always propagate batch_dim.
    if target == aten.cumsum.default:
        return input_batch_dim

    # sum.dim_IntList(input, dim, keepdim=False) / mean.dim(input, dim,
    # keepdim=False): reductions. The batch axis survives iff none of the
    # reduce dims target it. With keepdim=True the surviving axes keep
    # their positions; with keepdim=False each reduced dim left of the
    # batch axis shifts batch_dim down by 1.
    if target in (aten.sum.dim_IntList, aten.mean.dim):
        return _reduce_passthrough(node, input_batch_dim)

    # constant_pad_nd(input, pad): pads the LAST few dims (length of pad / 2);
    # earlier dims (including batch) are untouched.
    if target == aten.constant_pad_nd.default:
        return input_batch_dim

    # ---- Indexing / scatter on the jagged (embedding) activation path ----
    # These ops drive the HSTU jagged<->dense marshalling. Without batch_dim
    # propagation here, the [B*S, D] activation tensors lose their batch tag,
    # so _jit_gspmd marks them REPLICATED and GSPMD inserts an all-reduce
    # over the FULL GLOBAL buffer for each one (a large-model benchmark had ~154 all-reduces
    # of f32[276640,512] = ~473ms/step, ~45% of the step). Keeping the tag
    # lets the activations stay sharded [B*S/world, D] with no collective.

    # select(input, dim, index): removes axis ``dim`` (output ndim =
    # input ndim - 1). The batch axis survives unless it IS the selected
    # axis; selecting an axis left of batch shifts batch down by one.
    if target in (aten.select.int, getattr(aten.select_copy, "int", None)):
        dim = _normalize_dim(node, cast(int, node.args[1]))
        if dim is None or dim == input_batch_dim:
            return None
        return input_batch_dim - 1 if dim < input_batch_dim else input_batch_dim

    # index_select(input, dim, index) / scatter_add(input, dim, index, src):
    # output ndim == input ndim, shape preserved except along ``dim``. The
    # batch axis is preserved as long as we don't index/scatter along it
    # (operating on the sharded batch axis would need cross-device data).
    if target in (aten.index_select.default, aten.scatter_add.default):
        dim = _normalize_dim(node, cast(int, node.args[1]))
        if dim is None or dim == input_batch_dim:
            return None
        return input_batch_dim

    # index / _unsafe_index / gather (shape-preserving on non-indexed dims)
    # and index_put / _unsafe_index_put (output shape == self shape).
    # Conservatively propagate batch_dim, assuming the advanced indexing is
    # NOT along the batch axis (true for the preprocessor's bucketize-based
    # indexing and the HSTU jagged feature-dim scatter). ``index_put`` is the
    # op whose replicated fallback produced the 566MB scatter-add all-reduces.
    if target in (
        aten.index.Tensor,
        aten.gather.default,
        getattr(getattr(aten, "_unsafe_index", None), "Tensor", None),
        aten.index_put.default,
        getattr(getattr(aten, "_unsafe_index_put", None), "default", None),
    ):
        return input_batch_dim

    # view / reshape / _unsafe_view(input, sizes): default-passthrough.
    # The leading-axis batch dim is preserved by ALL common view patterns
    # used in engram / Gemma3 / FlatMM:
    #   * [B, S, D] -> [B*S, D]              (flatten batch+seq)
    #   * [B, S, H, D] -> [B, S, H*D]        (flatten heads)
    #   * [B, S, D] -> [-1, D]               (negative-one collapse)
    #   * [B, 1, D] -> [B, D]                (squeeze-like)
    # In each, axis 0 still carries the batch sharding even if its size
    # changes. Position-changing views (e.g. [B, S, D] -> [D, S, B]) are
    # rare in practice; we accept the small risk of over-propagating
    # because the alternative (returning None) caused 182 view nodes per
    # engram step to drop is_batch_input, which surgery-downstream
    # consumes as REPLICATED -> JAX broadcast mismatch at lower time.
    # The only case we still bail on is "batch_dim is out of bounds for
    # the new shape" (e.g. a view that drops dims below batch_dim).
    if target in (
        aten.view.default,
        aten.reshape.default,
        aten._unsafe_view.default,
        getattr(aten.view_copy, "default", None),
    ):
        return _passthrough_if_rank_ok(node, input_batch_dim)

    # expand(input, sizes): only changes size-1 dims; non-1 dims must match
    # input. So a batch_dim with size > 1 will pass through unchanged; a
    # batch_dim with size 1 being broadcast to size N is treated as a new
    # axis-0 batch and we propagate (the broadcast happens per-process so
    # the sharding is still axis 0 = batch).
    if target in (aten.expand.default, aten.expand_copy.default):
        return _passthrough_if_rank_ok(node, input_batch_dim)

    if isinstance(target, torch._ops.OpOverload) and target.name() in (
        "recir_padding::pad_to_static",
        "recir_padding::pad_count_to_sum",
        "recir_padding::pad_jagged_lengths",
        "recir_padding::pad_jagged_values",
        "recir_padding::unpad_grad",
        "recir_padding::unpad_jagged_grad",
    ):
        return input_batch_dim

    # recir::mark_boundary(x, name): pure identity (clone). Preserve
    # batch_dim. Without this handler, every marker insertion would drop
    # is_batch_input across a graph-break boundary.
    if isinstance(target, torch._ops.OpOverload) and target.name() == (
        "recir::mark_boundary"
    ):
        return input_batch_dim
    # Unknown op - don't propagate.
    return None


def _passthrough_if_rank_ok(node: fx.Node, input_batch_dim: int) -> Optional[int]:
    """Return input_batch_dim iff the new shape has a valid batch_dim slot.

    For view / reshape / _unsafe_view / expand, default-passthrough rule:
      * The new shape must have rank > input_batch_dim (so the batch slot
        still exists in the output).

    This intentionally does NOT require sizes[batch_dim] == input size —
    flatten-batch-into-leading-axis views (e.g. [B, S, D] -> [B*S, D]) are
    common and still preserve "axis 0 = batch", just with a different
    per-process size. Over-propagating here is far less harmful than
    under-propagating (which silently produces JAX broadcast mismatches).
    """
    if len(node.args) < 2:
        return None
    sizes = node.args[1]
    if not isinstance(sizes, (list, tuple)):
        return None
    if input_batch_dim >= len(sizes):
        return None
    return input_batch_dim


def _normalize_sequence_dim(node: fx.Node, rank_delta: int = 0) -> Optional[int]:
    """Normalize ``dim`` for ops whose first arg is a LIST of tensors.

    ``_normalize_dim`` reads the rank off ``node.args[0]``, which for cat/stack
    is the tensor list rather than a tensor, so it cannot serve here.
    ``rank_delta`` is 1 for ``stack``, whose output has one more axis than its
    inputs -- a negative dim counts back from the OUTPUT rank.
    """
    dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
    if not isinstance(dim, int):
        return None
    if dim >= 0:
        return dim
    seq = node.args[0] if node.args else None
    if not isinstance(seq, (list, tuple)) or not seq:
        return None
    first = seq[0]
    if not isinstance(first, fx.Node):
        return None
    ndim = getattr(first.meta.get("val"), "ndim", None)
    if ndim is None:
        return None
    return dim + ndim + rank_delta


def _normalize_dim(node: fx.Node, dim: int) -> Optional[int]:
    """Normalize a possibly-negative ``dim`` against the input tensor's ndim.

    The input tensor is the first positional arg (``self``) for the
    select / index_select / scatter_add family. Returns None when the input
    ndim can't be determined (so the caller leaves the node untagged).
    """
    if dim >= 0:
        return dim
    if not node.args:
        return None
    inp = node.args[0]
    if not isinstance(inp, fx.Node):
        return None
    val = inp.meta.get("val")
    if not hasattr(val, "ndim"):
        return None
    return val.ndim + dim


def _reduce_passthrough(node: fx.Node, input_batch_dim: int) -> Optional[int]:
    """Compute the output batch_dim for ``sum.dim_IntList`` / ``mean.dim``.

    Returns None when:
      * the reduce dims include batch_dim (batch axis is collapsed), or
      * we cannot determine the input's ndim or normalise the dims.

    Otherwise returns the new batch axis position, accounting for the
    keepdim flag: with keepdim=False any reduced dims to the left of the
    batch axis shift it down by 1 each.
    """
    if len(node.args) < 2:
        return None
    dims_arg = node.args[1]
    if not isinstance(dims_arg, (list, tuple)):
        return None
    # Locate the input tensor (first positional arg) and use its ndim to
    # normalise negative reduce dims.
    inputs = [a for a in node.args if isinstance(a, fx.Node)]
    if not inputs:
        return None
    val = inputs[0].meta.get("val")
    if not hasattr(val, "ndim"):
        return None
    ndim = val.ndim
    norm_dims = {(d + ndim) if d < 0 else d for d in dims_arg if isinstance(d, int)}
    if input_batch_dim in norm_dims:
        return None
    # keepdim is positional arg 2 in both signatures; default False.
    keepdim = False
    if len(node.args) > 2 and isinstance(node.args[2], bool):
        keepdim = node.args[2]
    elif "keepdim" in node.kwargs:
        keepdim = bool(node.kwargs["keepdim"])
    if keepdim:
        return input_batch_dim
    # No keepdim: each reduced dim left of batch_dim shrinks the batch axis.
    shift = sum(1 for d in norm_dims if d < input_batch_dim)
    return input_batch_dim - shift


def _output_batch_dim_for_positional(  # noqa: C901 -- positional operator semantics
    node: fx.Node,
) -> Optional[int]:
    """Per-arg batch_dim handling for matmul-family ops.

    For mm/matmul(A, B) with A (M, K) x B (K, N) -> (M, N):
      * A.batch_dim == 0 (axis 0 = batch): output keeps batch on axis 0.
      * A.batch_dim == 1 (axis K = batch): K gets contracted, no batch out.
      * B.batch_dim == 0 (axis K = batch): K gets contracted, no batch out.
      * B.batch_dim == 1 (axis N = batch): output keeps batch on axis 1.

    For bmm(A, B) with A (B, N, K) x B (B, K, M) -> (B, N, M):
      * A or B.batch_dim == 0: output keeps batch on axis 0.

    For addmm(C, A, B) with C bias + A x B: same as mm on A,B. C must
    be broadcast-compatible with output shape.
    """
    aten = torch.ops.aten
    target = node.target

    def _bd(arg: Any) -> Optional[int]:
        if hasattr(arg, "meta") and arg.meta.get("is_batch_input"):
            return arg.meta.get("batch_dim", 0)
        return None

    if target in (aten.mm.default, aten.matmul.default):
        if len(node.args) < 2:
            return None
        a, b = node.args[0], node.args[1]
        a_bd, b_bd = _bd(a), _bd(b)
        # A's batch axis 0 propagates to output axis 0 (free dim survives).
        if a_bd == 0:
            return 0
        # B's batch axis 1 propagates to output axis 1 (free dim survives).
        if b_bd == 1:
            return 1
        # All other cases either contract the batch axis or don't have one.
        return None

    if target == aten.addmm.default:
        # addmm(input, mat1, mat2): output = input + mat1 @ mat2.
        # Treat mat1/mat2 like mm.
        if len(node.args) < 3:
            return None
        a, b = node.args[1], node.args[2]
        a_bd, b_bd = _bd(a), _bd(b)
        if a_bd == 0:
            return 0
        if b_bd == 1:
            return 1
        return None

    if target == aten.bmm.default:
        if len(node.args) < 2:
            return None
        a, b = node.args[0], node.args[1]
        a_bd, b_bd = _bd(a), _bd(b)
        # bmm: leading axis 0 is batch dimension (preserved).
        # axes 1 and 2 are matmul dims (axis 2 of A = axis 1 of B is contracted).
        if a_bd == 0 or b_bd == 0:
            return 0
        return None

    return None


def propagate_batch_input_meta(  # noqa: C901 -- graph metadata propagation
    gm: fx.GraphModule,
) -> None:
    import logging as _lg

    _logger = _lg.getLogger(__name__)
    aten = torch.ops.aten
    # Ops where the operand position matters for batch tracking. For these
    # we pass per-arg batch_dim to _output_batch_dim_for via node.args
    # rather than collapsing into a single input_batch_dim.
    POSITIONAL_OPS = {
        aten.mm.default,
        aten.matmul.default,
        aten.bmm.default,
        aten.addmm.default,
    }
    _missing: dict[str, int] = {}
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        # Skip ops that already have explicit tagging (e.g., dispatch_fwd
        # outputs stamped by install_graph_break_dispatch).
        if node.meta.get("is_batch_input"):
            continue
        # For positional ops (matmul-family), look up batch_dim per-arg
        # from node.args directly so the rule can distinguish mat1 vs mat2.
        if node.target in POSITIONAL_OPS:
            new_batch_dim = _output_batch_dim_for_positional(node)
            if new_batch_dim is None:
                continue
            node.meta["is_batch_input"] = True
            node.meta["batch_dim"] = new_batch_dim
            continue
        # Find a tensor input carrying batch sharding.
        input_batch_dim: Optional[int] = None
        for arg in node.all_input_nodes:
            if arg.meta.get("is_batch_input"):
                bd = arg.meta.get("batch_dim", 0)
                if input_batch_dim is None:
                    input_batch_dim = bd
                elif input_batch_dim != bd:
                    # Multiple batched inputs disagree on batch_dim - can't
                    # reliably pick one; bail.
                    input_batch_dim = None
                    break
        if input_batch_dim is None:
            continue
        new_batch_dim = _output_batch_dim_for(node.target, input_batch_dim, node)
        if new_batch_dim is None:
            tname = (
                node.target.name()
                if isinstance(node.target, torch._ops.OpOverload)
                else str(node.target)
            )
            _missing[tname] = _missing.get(tname, 0) + 1
            continue
        node.meta["is_batch_input"] = True
        node.meta["batch_dim"] = new_batch_dim
    if _missing:
        _logger.info(
            "[propagate_batch_input_meta] %d distinct ops broke propagation "
            "(input was batched but op not in allowlist):",
            len(_missing),
        )
        for _name, _count in sorted(_missing.items(), key=lambda x: -x[1]):
            _logger.info("  %s -> %d nodes", _name, _count)


def _output_fsdp_shard_dim_for(
    target: Any, input_shard_dim: int, node: fx.Node
) -> Optional[int]:
    """Return the output FSDP shard axis for explicitly supported ops."""
    aten = torch.ops.aten
    if target in (aten.transpose.int, getattr(aten.transpose_copy, "int", None)):
        dim0 = _normalize_dim(node, cast(int, node.args[1]))
        dim1 = _normalize_dim(node, cast(int, node.args[2]))
        if dim0 is None or dim1 is None:
            return None
        if input_shard_dim == dim0:
            return dim1
        if input_shard_dim == dim1:
            return dim0
        return input_shard_dim
    if target in (aten.permute.default, aten.permute_copy.default):
        perm = node.args[1]
        if isinstance(perm, (list, tuple)) and input_shard_dim in perm:
            return cast(List[int], perm).index(input_shard_dim)
        return None
    if target in (aten.sum.dim_IntList, aten.mean.dim):
        return _reduce_passthrough(node, input_shard_dim)
    axis_preserving = {
        aten.add.Tensor,
        aten.add.Scalar,
        aten.add_.Tensor,
        aten.add_.Scalar,
        aten.sub.Tensor,
        aten.sub.Scalar,
        aten.sub_.Tensor,
        aten.sub_.Scalar,
        aten.mul.Tensor,
        aten.mul.Scalar,
        aten.mul_.Tensor,
        aten.mul_.Scalar,
        aten.div.Tensor,
        aten.div.Scalar,
        aten.div_.Tensor,
        aten.div_.Scalar,
        aten.neg.default,
        aten.sqrt.default,
        aten.rsqrt.default,
        aten.lerp_.Scalar,
        aten.addcmul_.default,
        aten.addcdiv_.default,
        aten.clone.default,
        aten.detach.default,
        aten._to_copy.default,
    }
    return input_shard_dim if target in axis_preserving else None


def propagate_fsdp_sharded_meta(gm: fx.GraphModule) -> None:
    """Propagate GSPMD FSDP metadata using explicit shard-axis rules."""
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        if node.meta.get("is_fsdp_sharded") or node.meta.get("is_batch_input"):
            continue
        output_shard_dim: Optional[int] = None
        for arg in node.all_input_nodes:
            if not arg.meta.get("is_fsdp_sharded") or arg.meta.get("is_batch_input"):
                continue
            input_shard_dim = arg.meta.get("fsdp_shard_dim", 0)
            candidate = _output_fsdp_shard_dim_for(node.target, input_shard_dim, node)
            if candidate is None:
                output_shard_dim = None
                break
            if output_shard_dim is not None and output_shard_dim != candidate:
                output_shard_dim = None
                break
            output_shard_dim = candidate
        if output_shard_dim is not None:
            node.meta["is_fsdp_sharded"] = True
            node.meta["fsdp_shard_dim"] = output_shard_dim
