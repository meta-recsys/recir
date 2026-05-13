# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Forward-propagate is_batch_input/batch_dim across an FX graph.

`_tag_batch_inputs` (in `recir/api.py`) only marks top-level placeholders with
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

from typing import Any, cast, List, Optional

import torch
from torch import fx


def _output_batch_dim_for(
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
    }
    if target in axis_preserving:
        return input_batch_dim

    # matmul-family ops are handled via _output_batch_dim_for_positional
    # in propagate_batch_input_meta because the batch axis depends on
    # WHICH operand carries the batch tag (mat1 vs mat2) and which axis.

    # cat / stack on a non-batch axis preserve batch_dim.
    if target in (aten.cat.default, aten.stack.default):
        return input_batch_dim

    # embedding lookup: indices (batch_dim) -> (..., D), batch_dim preserved.
    if target == aten.embedding.default:
        return input_batch_dim

    # cumsum(input, dim): preserves shape. Always propagate batch_dim.
    if target == aten.cumsum.default:
        return input_batch_dim

    # constant_pad_nd(input, pad): pads the LAST few dims (length of pad / 2);
    # earlier dims (including batch) are untouched.
    if target == aten.constant_pad_nd.default:
        return input_batch_dim

    # index / gather: shape-preserving on non-indexed dims. Conservatively
    # propagate batch_dim (assumes indexing isn't along the batch axis,
    # which is true for our preprocessor's bucketize-based indexing).
    if target in (aten.index.Tensor, aten.gather.default):
        return input_batch_dim

    if isinstance(target, torch._ops.OpOverload) and target.name() in (
        "recir_padding::pad_to_static",
        "recir_padding::pad_count_to_sum",
        "recir_padding::pad_jagged_lengths",
        "recir_padding::pad_jagged_values",
        "recir_padding::unpad_grad",
        "recir_padding::unpad_jagged_grad",
    ):
        return input_batch_dim
    # Unknown op - don't propagate.
    return None


def _output_batch_dim_for_positional(node: fx.Node) -> Optional[int]:
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


def propagate_batch_input_meta(gm: fx.GraphModule) -> None:
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
        # outputs stamped by replace_override_anchors).
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
