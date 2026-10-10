# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""jte-free SparseCore forward/backward kernel emission.

Reconstructs the SparseCore custom_calls that jax_tpu_embedding emits, WITHOUT
its tpu_sparse_dense_matmul / tpu_sparse_dense_matmul_grad primitives. The
kernels themselves live in libtpu; this module only reproduces the thin emission
(custom_call target + JSON backend_config + operands, and for the backward the
fused-Adagrad optimizer subfunction referenced via called_computations).

  sc_matmul_fwd  -> stablehlo.custom_call @SparseDenseMatmulOp
  sc_matmul_grad -> stablehlo.custom_call @SparseDenseMatmulGradOpWithOptimizerUpdate
                    (+ a private FuncOp doing the Adagrad update)

Both mirror the JTE lowering for the single-device TorchTPU embedding path. JTE
is still needed for host-side CSR preprocessing + the feature/table specs
(max_ids / device_batch_size come from the StackedTableSpec).
"""

import collections
import functools
import json

import jax
import jax.extend as jex
import jax.numpy as jnp
import numpy as np
from jax import core
from jax.extend.mlir import ir
from jax.extend.mlir.dialects import func as func_dialect, stablehlo as hlo
from jax.interpreters import mlir, xla

# jte constants.PADDING_VALUE; INT32_MAX (confirmed from the dumped backend_config).
_PAD_VALUE = 2147483647


def _annotate_sparse(op):
    op.attributes["mhlo.frontend_attributes"] = ir.DictAttr.get(
        {"_xla_compute_type": ir.StringAttr.get("sparse")}
    )
    return op


def _result_type(ctx, aval):
    return (
        mlir.aval_to_ir_type(ctx.module_context, aval)
        if jax.__version_info__ >= (0, 10, 1)
        else mlir.aval_to_ir_type(aval)
    )


# --------------------------------------------------------------------------
# forward: @SparseDenseMatmulOp
# --------------------------------------------------------------------------
_sc_fwd_p = jex.core.Primitive("sc_sparse_dense_matmul_fwd")


@_sc_fwd_p.def_abstract_eval
def _fwd_abstract(
    rp,
    ei,
    si,
    gn,
    table,
    *,
    device_batch_size,
    max_ids,
    max_unique_ids,
    sharding_strategy,
):
    del rp, ei, si, gn, max_ids, max_unique_ids, sharding_strategy
    return core.ShapedArray((device_batch_size, table.shape[1]), jnp.float32)


_sc_fwd_p.def_impl(functools.partial(xla.apply_primitive, _sc_fwd_p))


def _fwd_lowering(
    ctx,
    rp,
    ei,
    si,
    gn,
    table,
    *,
    device_batch_size,
    max_ids,
    max_unique_ids,
    sharding_strategy,
):
    (out_aval,) = ctx.avals_out
    dim = ir.RankedTensorType(table.type).get_dim_size(1)
    activation_init = hlo.broadcast(
        hlo.constant(ir.DenseElementsAttr.get(np.float32(0.0))),
        mlir.dense_int_array([device_batch_size, dim]),
    )
    backend_config = json.dumps(
        {
            "sparse_dense_matmul_config": {
                "max_ids_per_partition": max_ids,
                "max_unique_ids_per_partition": max_unique_ids,
                "sharding_strategy": sharding_strategy,
                "pad_value": _PAD_VALUE,
            },
            "device_type": "DEVICE_TYPE_SPARSECORE",
        }
    )
    return jax.ffi.ffi_lowering(
        "SparseDenseMatmulOp",
        result_types=[_result_type(ctx, out_aval)],
        api_version=1,
        backend_config=backend_config,
        skip_ffi_layout_processing=True,
    )(ctx, rp, ei, si, gn, table, activation_init)


mlir.register_lowering(_sc_fwd_p, _fwd_lowering)


def sc_matmul_fwd(
    rp,
    ei,
    si,
    gn,
    table,
    *,
    device_batch_size,
    max_ids,
    max_unique_ids,
    sharding_strategy=1,
):
    """SparseCore forward (stacked activation) without jte's primitive."""
    return _sc_fwd_p.bind(
        rp,
        ei,
        si,
        gn,
        table,
        device_batch_size=device_batch_size,
        max_ids=max_ids,
        max_unique_ids=max_unique_ids,
        sharding_strategy=sharding_strategy,
    )


# --------------------------------------------------------------------------
# backward: @SparseDenseMatmulGradOpWithOptimizerUpdate (fused Adagrad)
# --------------------------------------------------------------------------
_sc_grad_p = jex.core.Primitive("sc_sparse_dense_matmul_grad_adagrad")
_sc_grad_p.multiple_results = True


@_sc_grad_p.def_abstract_eval
def _grad_abstract(
    rp,
    ei,
    si,
    gn,
    table,
    accumulator,
    grad,
    lr,
    *,
    max_ids,
    max_unique_ids,
    sharding_strategy,
    computation_name,
):
    del (
        rp,
        ei,
        si,
        gn,
        grad,
        lr,
        max_ids,
        max_unique_ids,
        sharding_strategy,
        computation_name,
    )
    return table, accumulator


_sc_grad_p.def_impl(functools.partial(xla.apply_primitive, _sc_grad_p))


def _grad_lowering(
    ctx,
    rp,
    ei,
    si,
    gn,
    table,
    accumulator,
    grad,
    lr,
    *,
    max_ids,
    max_unique_ids,
    sharding_strategy,
    computation_name,
):
    dim = ir.RankedTensorType(table.type).get_dim_size(1)
    row_t = ir.RankedTensorType.get([1, dim], ir.F32Type.get())
    opt = func_dialect.FuncOp(
        computation_name,
        ([row_t, row_t, row_t, row_t], [ir.TupleType.get_tuple([row_t, row_t])]),
        ip=ctx.module_context.ip,
        visibility="private",
    )
    eb = opt.add_entry_block()
    with ir.InsertionPoint(eb):
        g, tbl, acc, lr_a = eb.arguments  # grad, table, accumulator, learning_rate
        new_acc = hlo.add(acc, hlo.multiply(g, g))
        new_tbl = hlo.subtract(
            tbl, hlo.divide(hlo.multiply(lr_a, g), hlo.sqrt(new_acc))
        )
        func_dialect.ReturnOp([hlo.tuple([new_tbl, new_acc])])

    backend_config = json.dumps(
        {
            "sparse_dense_matmul_config": {
                "max_ids_per_partition": max_ids,
                "max_unique_ids_per_partition": max_unique_ids,
                "pad_value": _PAD_VALUE,
                "sharding_strategy": sharding_strategy,
                "num_slot_variables": 1,
                "num_hyperparameters": 1,
            },
            "device_type": "DEVICE_TYPE_SPARSECORE",
        }
    )
    op = jax.ffi.ffi_lowering(
        "SparseDenseMatmulGradOpWithOptimizerUpdate",
        result_types=[ir.TupleType.get_tuple([table.type, accumulator.type])],
        backend_config=backend_config,
        called_computations=[computation_name],
        skip_ffi_layout_processing=True,
        api_version=1,
    )(ctx, rp, ei, si, gn, grad, table, accumulator, lr)
    new_table = _annotate_sparse(hlo.GetTupleElementOp(op, 0))
    new_acc = _annotate_sparse(hlo.GetTupleElementOp(op, 1))
    return [*new_table.results, *new_acc.results]


mlir.register_lowering(_sc_grad_p, _grad_lowering)


def sc_matmul_grad(
    rp,
    ei,
    si,
    gn,
    table,
    accumulator,
    grad,
    lr,
    *,
    max_ids,
    max_unique_ids,
    sharding_strategy=1,
    computation_name="adagrad_optimizer_update",
):
    """SparseCore backward + fused Adagrad (new table, new accumulator) without jte."""
    return _sc_grad_p.bind(
        rp,
        ei,
        si,
        gn,
        table,
        accumulator,
        grad,
        lr,
        max_ids=max_ids,
        max_unique_ids=max_unique_ids,
        sharding_strategy=sharding_strategy,
        computation_name=computation_name,
    )


# --------------------------------------------------------------------------
# stacked-activation <-> per-feature reshaping (jte's *default*, non-primitive
# path -- pure array ops over the feature-spec metadata). A stacked table can
# back several features; the matmul produces one stacked (sum_of_samples, dim)
# activation that must be split per feature (fwd), and per-feature grads must be
# concatenated back (bwd). einops "(f b) d -> f b d" with f=num_sc is just a
# reshape (f is the outer index), so jnp.reshape/concatenate reproduces it.
# --------------------------------------------------------------------------
def unstack_activation(stacked, feature, global_device_count, num_sc):
    """Slice one feature's activation out of its stacked-table activation."""
    f = num_sc
    b = stacked.shape[0] // f
    out_dim = feature.output_shape[-1]
    row_off = feature.id_transformation.row_offset // (global_device_count * num_sc)
    per_slice = feature.output_shape[0] // (global_device_count * num_sc)
    act = stacked.reshape(f, b, stacked.shape[1])
    sl = act[:, row_off : row_off + per_slice, 0:out_dim]
    return sl.reshape(f * per_slice, out_dim)


def stack_gradients(grads, feature_specs, num_sc):
    """Concatenate per-feature grads into one stacked-table grad (per stack)."""
    by_stack = collections.defaultdict(list)
    for fname, spec in feature_specs.items():
        by_stack[spec.table_spec.stacked_table_spec.stack_name].append(
            (spec, grads[fname])
        )
    result = {}
    for stack, items in by_stack.items():
        items.sort(key=lambda x: x[0].id_transformation.row_offset)
        prepared = []
        for spec, g in items:
            g = g.reshape(-1, spec.table_spec.embedding_dim)
            extra = (
                spec.table_spec.setting_in_stack.padded_embedding_dim
                - spec.table_spec.embedding_dim
            )
            if extra:
                g = jnp.pad(g, ((0, 0), (0, extra)))
            f = num_sc
            prepared.append(g.reshape(f, g.shape[0] // f, g.shape[1]))
        cat = jnp.concatenate(prepared, axis=1)
        result[stack] = cat.reshape(cat.shape[0] * cat.shape[1], cat.shape[2])
    return result
