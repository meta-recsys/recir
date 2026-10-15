# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Torch ops implemented using jax."""

import functools
import math
import sys
from collections.abc import Sequence
from typing import Any, Callable

import jax
import numpy as np
import torch
import torch.distributed._functional_collectives
from jax import numpy as jnp
from recir.torchax import interop
from recir.torchax.ops import jax_reimplement, mappings, op_base, ops_registry
from recir.torchax.ops.decorators import op
from recir.torchax.view import View

# Keys are OpOverload, value is a callable that takes
# Tensor
all_ops: dict[Any, Any] = {}


def op_if_present(
    *names: str, **kwargs: Any
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Like @op but resolves aten ops by name and skips any absent from the
    linked torch build.

    Some deprecated ATen ops (e.g. aten.cholesky, removed in favor of
    aten.linalg_cholesky) no longer exist in newer PyTorch builds; referencing
    them via torch.ops.aten.<name> raises AttributeError at import time. This
    guard registers only the ops that exist and is a no-op when none do.
    """
    targets: list[Any] = []
    for name in names:
        target = getattr(torch.ops.aten, name, None)
        if target is not None:
            targets.append(target)
    if not targets:

        def _identity(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return _identity
    return op(*targets, **kwargs)


@op(  # pyre-ignore[56]
    torch.ops.aten.view_copy,
    torch.ops.aten.view,
    torch.ops.aten._unsafe_view,
    torch.ops.aten.reshape,
)
def _aten_unsafe_view(x: Any, shape: Any) -> Any:
    return jnp.reshape(x, shape)


def _run_binary_pointwise_operator(op: Any, x: Any, y: Any) -> Any:
    if isinstance(x, float):
        dtype = _torch_binary_scalar_type(x, y)
        x = jnp.array(x, dtype=dtype)
    if isinstance(y, float):
        dtype = _torch_binary_scalar_type(y, x)
        y = jnp.array(y, dtype=dtype)
    return op(x, y)


@op(torch.ops.aten.add.Tensor)
@op(torch.ops.aten.add.Scalar)  # pyre-ignore[56]
def _aten_add(x: Any, y: Any, *, alpha: Any = 1) -> Any:
    """if isinstance(x, jnp.ndarray) and isinstance(y, jnp.ndarray):

    assert x.dtype == y.dtype, (x.dtype, y.dtype)
    """
    return _run_binary_pointwise_operator(lambda x, y: x + y, x, y * alpha)


@op(  # pyre-ignore[56]
    torch.ops.aten.copy_, is_jax_function=False, is_view_op=True, needs_env=True
)
def _aten_copy(x: Any, y: Any, memory_format: Any = None, env: Any = None) -> Any:
    if y.device.type == "cpu":
        y = env.to_xla(y)

    if isinstance(x, View):
        x.update(y)
        return x

    if x.ndim == 1 and y.ndim == 0:
        # case of torch.empty((1,)).copy_(tensor(N))
        # we need to return 0D tensor([N]) and not scalar tensor(N)
        # ref: https://github.com/pytorch/xla/issues/7505#issuecomment-2395319131
        x._elem = jnp.array([y._elem.astype(x._elem.dtype)])
    else:
        x._elem = y._elem.astype(x._elem.dtype)
    return x


@op(torch.ops.aten.clone)  # pyre-ignore[56]
def _aten_clone(x: Any, memory_format: Any = None) -> Any:
    return x


# aten.trunc
@op(torch.ops.aten.trunc)  # pyre-ignore[56]
def _aten_trunc(x: Any) -> Any:
    res = jnp.trunc(x)
    return res.astype(x.dtype)


@op(torch.ops.aten.index_copy)  # pyre-ignore[56]
def _aten_index_copy(x: Any, dim: Any, indexes: Any, source: Any) -> Any:
    if x.ndim == 0:
        return source
    if x.ndim == 1:
        source = jnp.squeeze(source)
    # return jax.lax.scatter(x, index, dim)
    if dim < 0:
        dim = dim + x.ndim
    dims = []
    for i in range(len(x.shape)):
        if i == dim:
            dims.append(indexes)
        else:
            dims.append(slice(None, None, None))
    return x.at[tuple(dims)].set(source)


# aten.cauchy_
@op(torch.ops.aten.cauchy_)  # pyre-ignore[56]
def _aten_cauchy_(x: Any, median: Any = 0, sigma: Any = 1) -> Any:
    """
    Fills the input array with values drawn from a Cauchy distribution.

    Args:
      x: An array to be filled with Cauchy samples.
      median: The median of the Cauchy distribution.
      sigma: The scale parameter of the Cauchy distribution.

    Returns:
      The input array filled with Cauchy samples.
    """
    key = jax.random.PRNGKey(0)  # You should use a different key for each call
    samples = jax.random.cauchy(key, x.shape) * sigma + median
    return x.at[:].set(samples)


@op(torch.ops.aten.atleast_2d)  # pyre-ignore[56]
def _aten_atleast_2d(inputs: Any) -> Any:
    return jnp.atleast_2d(inputs)


@op(torch.ops.aten.atleast_1d)  # pyre-ignore[56]
def _aten_atleast_1d(inputs: Any) -> Any:
    return jnp.atleast_1d(inputs)


# aten.complex
@op(torch.ops.aten.complex)  # pyre-ignore[56]
def _aten_complex(real: Any, imag: Any) -> Any:
    """
    Constructs a complex array from real and imaginary parts.

    Args:
      real: An array of real values.
      imag: An array of imaginary values.

    Returns:
      A complex array with the specified real and imaginary parts.
    """
    return jnp.array(real, dtype=jnp.float32) + 1j * jnp.array(imag, dtype=jnp.float32)


# aten.exponential_
@op(torch.ops.aten.exponential_)  # pyre-ignore[56]
def _aten_exponential_(x: Any, lambd: Any = 1.0) -> Any:
    """
    Fills the input array with values drawn from an exponential distribution.

    Args:
      x: An array to be filled with exponential samples.
      lambd: The rate parameter of the exponential distribution.

    Returns:
      The input array filled with exponential samples.
    """
    key = jax.random.PRNGKey(0)  # Use a different key for each call
    samples = jax.random.exponential(key, x.shape) / lambd
    return x.at[:].set(samples)


# aten.linalg_householder_product
@op(torch.ops.aten.linalg_householder_product)  # pyre-ignore[56]
def _aten_linalg_householder_product(input: Any, tau: Any) -> Any:
    return jax.lax.linalg.householder_product(a=input, taus=tau)


@op(torch.ops.aten.select)  # pyre-ignore[56]
def _aten_select(x: Any, dim: Any, indexes: Any) -> Any:
    return jax.lax.index_in_dim(x, index=indexes, axis=dim, keepdims=False)


@op(torch.ops.aten.index_select)
@op(torch.ops.aten.select_copy)  # pyre-ignore[56]
def _aten_index_select(x: Any, dim: Any, index: Any) -> Any:
    if x.shape == ():
        return x
    return jnp.take(x, index, dim)


@op(torch.ops.aten.linalg_cholesky)  # pyre-ignore[56]
def _aten_cholesky(input: Any, upper: Any = False) -> Any:
    return jax.scipy.linalg.cholesky(input, lower=(not upper))


@op(torch.ops.aten.linalg_cholesky_ex)  # pyre-ignore[56]
def _aten_linalg_cholesky_ex(
    input: Any, upper: Any = False, check_errors: Any = False
) -> Any:
    if check_errors:
        raise NotImplementedError(
            "check_errors=True is not supported in this JAX implementation. "
            "Check for positive definiteness using jnp.linalg.eigvalsh before "
            "calling this function."
        )

    L = jax.scipy.linalg.cholesky(input, lower=not upper)
    if len(L.shape) > 2:
        info = jnp.zeros(shape=L.shape[:-2], dtype=jnp.int32)
    else:
        info = jnp.array(0, dtype=jnp.int32)
    return L, info


@op(torch.ops.aten.cholesky_solve)  # pyre-ignore[56]
def _aten_cholesky_solve(input: Any, input2: Any, upper: Any = False) -> Any:
    # Ensure input2 is lower triangular for cho_solve
    L = input2 if not upper else input2.T
    # Use cho_solve to solve the linear system
    solution = jax.scipy.linalg.cho_solve((L, True), input)
    return solution


@op(torch.ops.aten.special_zeta)  # pyre-ignore[56]
def _aten_special_zeta(x: Any, q: Any) -> Any:
    new_dtype = mappings.t2j_dtype(torch.get_default_dtype())
    res = jax.scipy.special.zeta(x, q)
    if isinstance(x, int) or isinstance(q, int):
        res = res.astype(new_dtype)
    return res  # jax.scipy.special.zeta(x, q)


# aten.igammac
@op(torch.ops.aten.igammac)  # pyre-ignore[56]
def _aten_igammac(input: Any, other: Any) -> Any:
    if isinstance(input, jnp.ndarray):
        input = jnp.where(input < 0, jnp.nan, input)
    if isinstance(other, jnp.ndarray):
        other = jnp.where(other < 0, jnp.nan, other)
    else:
        if (input == 0 and other == 0) or (input < 0) or (other < 0):
            other = jnp.nan
    return jnp.array(jax.scipy.special.gammaincc(input, other))


@op(torch.ops.aten.mean)  # pyre-ignore[56]
def _aten_mean(x: Any, dim: Any = None, keepdim: Any = False) -> Any:
    if x.shape == () and dim is not None:
        dim = None  # disable dim for jax array without dim
    return jnp.mean(x, dim, keepdims=keepdim)


def _torch_binary_scalar_type(scalar: Any, tensor: Any) -> Any:
    if "float" in str(tensor.dtype) or "complex" in str(tensor.dtype):
        return tensor.dtype

    if isinstance(scalar, int):
        if "int" in str(tensor.dtype):
            return tensor.dtype

    return jnp.float32


# Above this many bins/needles we keep ``jnp.searchsorted`` (a binary-search
# ``while`` loop). At or below it we replace searchsorted/bucketize with a flat,
# fully-vectorized broadcasted compare-and-sum, which avoids the serialized loop
# (a big TPU win when the sorted array is small but the value array is large).
# Above the cap the O(len(values) * n_bins) broadcast would waste memory/compute.
_SEARCHSORTED_VECTORIZE_MAX_BINS: int = 16384


def _searchsorted(
    sorted_sequence: Any,
    values: Any,
    side: str,
    sorter: Any = None,
) -> Any:
    """searchsorted that avoids the binary-search ``while`` loop for small arrays.

    ``jnp.searchsorted`` lowers to a serialized binary-search ``while`` loop on
    TPU. When the sorted array is short along its last axis, the same result is
    ``sum(a < v)`` (``side="left"``) or ``sum(a <= v)`` (``side="right"``) over
    that axis -- a single fused reduction, no loop. Works for a 1-D sorted array
    as well as the batched (n-D) case that recir's batch-dim propagation can
    produce (it broadcasts along the last, per-row bin axis). The returned
    insertion index equals that count regardless of ``sorter`` (counting is
    order-independent). Falls back to ``jnp.searchsorted`` for large sorted
    arrays where the ``[..., N, L]`` broadcast would be too big.
    """
    last = sorted_sequence.shape[-1] if sorted_sequence.ndim >= 1 else 0
    if 0 < last <= _SEARCHSORTED_VECTORIZE_MAX_BINS:
        # a: [..., 1, L]  vs  v: [..., N, 1]  -> compare over the last (bin) axis.
        a = sorted_sequence[..., None, :]
        v = values[..., None]
        cmp = a < v if side == "left" else a <= v
        return cmp.sum(axis=-1).astype(jnp.int32)
    return jnp.searchsorted(sorted_sequence, values, side=side, sorter=sorter)


@op(torch.ops.aten.searchsorted.Tensor)  # pyre-ignore[56]
def _aten_searchsorted(
    sorted_sequence: Any,
    values: Any,
    *,
    out_int32: bool = False,
    right: bool = False,
    side: str | None = None,
    sorter: Any = None,
) -> Any:
    mappings.t2j_dtype(torch.get_default_dtype())
    jax_side = "right" if right else "left"
    if side is not None:
        jax_side = side
    res = _searchsorted(sorted_sequence, values, jax_side, sorter)
    if out_int32:
        res = res.astype(np.dtype(np.int32))
    elif sorted_sequence.dtype == np.dtype(np.int32):
        res = res.astype(np.dtype(np.int64))
    return res


@op(torch.ops.aten.sub.Tensor)
@op(torch.ops.aten.sub.Scalar)  # pyre-ignore[56]
def _aten_sub(x: Any, y: Any, alpha: Any = 1) -> Any:
    return _run_binary_pointwise_operator(lambda x, y: x - y, x, y * alpha)


@op(torch.ops.aten.numpy_T)  # pyre-ignore[56]
def _aten_numpy_T(input: Any) -> Any:
    """
    Jax implementation of torch.numpy_T.

    Args:
      input: JAX array.

    Returns:
      Transposed JAX array.
    """
    return jnp.transpose(input)


@op(torch.ops.aten.mm)  # pyre-ignore[56]
def _aten_mm(x: Any, y: Any) -> Any:
    res = x @ y
    return res


@op(torch.ops.aten.mul.Tensor, torch.ops.aten.mul.Scalar)  # pyre-ignore[56]
def _aten_mul(x: Any, y: Any) -> Any:
    return _run_binary_pointwise_operator(lambda x, y: x * y, x, y)


@op(torch.ops.aten.silu)
@op(torch.ops.aten.silu.default)  # pyre-ignore[56]
def _aten_silu(x: Any) -> Any:
    return jax.nn.silu(x)


@op(torch.ops.aten.t)  # pyre-ignore[56]
def _aten_t(x: Any) -> Any:
    return jnp.transpose(x)


@op(torch.ops.aten.transpose)
@op(torch.ops.aten.transpose_copy)  # pyre-ignore[56]
def _aten_transpose(x: Any, dim0: Any, dim1: Any) -> Any:
    if x.ndim == 0:
        return x
    dim0 = dim0 if dim0 >= 0 else dim0 + x.ndim
    dim1 = dim1 if dim1 >= 0 else dim1 + x.ndim
    return jnp.swapaxes(x, dim0, dim1)


@op(torch.ops.aten.triu)  # pyre-ignore[56]
def _aten_triu(m: Any, k: Any = 0) -> Any:
    return jnp.triu(m, k)


@op(torch.ops.aten.slice)
@op(torch.ops.aten.slice_copy)  # pyre-ignore[56]
def _aten_slice(
    self: Any, dim: Any = 0, start: Any = None, end: Any = None, step: Any = 1
) -> Any:
    if dim < 0:
        dim += self.ndim
    if end == sys.maxsize:
        end = self.shape[dim]
    sl = slice(start, end, step)
    dims = []
    for i in range(len(self.shape)):
        if i == dim:
            dims.append(sl)
        else:
            dims.append(slice(None, None, None))
    return self[tuple(dims)]


@op(torch.ops.aten.positive)
@op(torch.ops.aten.detach)  # pyre-ignore[56]
def _aten_detach(self: Any) -> Any:
    return self


@op(torch.ops.aten.imag)  # pyre-ignore[56]
def _aten_imag(x: Any) -> Any:
    return jnp.imag(x)


@op(torch.ops.aten.isfinite)  # pyre-ignore[56]
def _aten_isfinite(x: Any) -> Any:
    return jnp.isfinite(x)


@op(torch.ops.aten.real)  # pyre-ignore[56]
def _aten_real(x: Any) -> Any:
    return jnp.real(x)


@op(torch.Tensor.resize_)
def _aten_resize_(x: Any, size: Any, interpolation: Any = "linear") -> Any:
    new_size = tuple(size)
    return jax.numpy.resize(x, new_size)


@op(torch.ops.aten.resize_as_)  # pyre-ignore[56]
def _aten_resize_as_(x: Any, y: Any) -> Any:
    return jax.numpy.resize(x, y.shape)


@op(torch.ops.aten.repeat_interleave.Tensor)
def repeat_interleave(repeats: Any, dim: Any = 0, *, output_size: Any = None) -> Any:
    return jnp.repeat(
        np.arange(repeats.shape[dim]),
        repeats,
        total_repeat_length=output_size,
    )


@op(torch.ops.aten.repeat_interleave.self_int)
@op(torch.ops.aten.repeat_interleave.self_Tensor)  # pyre-ignore[56]
def repeat_interleave(self: Any, repeats: Any, dim: Any = 0) -> Any:  # noqa: F811
    total_repeat_length = None
    if isinstance(repeats, int):
        total_repeat_length = self.shape[dim] * repeats
        repeats = np.array([repeats] * self.shape[dim])
    return jnp.repeat(self, repeats, dim, total_repeat_length=total_repeat_length)


@op(torch.ops.aten.view_as_real)  # pyre-ignore[56]
def _aten_view_as_real(x: Any) -> Any:
    real = jnp.real(x)
    im = jnp.imag(x)
    res = jnp.stack([real, im], -1)
    return res


@op(torch.ops.aten.stack)  # pyre-ignore[56]
def _aten_stack(tensors: Any, dim: Any = 0) -> Any:
    return jnp.stack(tensors, dim)


@op(torch.ops.aten._softmax)
@op(torch.ops.aten.softmax)
@op(torch.ops.aten.softmax.int)  # pyre-ignore[56]
def _aten_softmax(x: Any, dim: Any, halftofloat: Any = False) -> Any:
    # Use dtype check instead of halftofloat flag
    if x.dtype in (jnp.float16, jnp.bfloat16):
        x = x.astype(jnp.float32)
    if x.shape == ():
        return jax.nn.softmax(x.reshape([1]), axis=0).reshape([])
    return jax.nn.softmax(x, dim)


def _is_int(x: Any) -> Any:
    if isinstance(x, int):
        return True
    if isinstance(x, jax.Array) and (
        x.dtype.name.startswith("int") or x.dtype.name.startswith("uint")
    ):
        return True
    return False


def highest_precision_int_dtype(tensor1: Any, tensor2: Any) -> Any:
    if isinstance(tensor1, int):
        return tensor2.dtype
    if isinstance(tensor2, int):
        return tensor1.dtype

    dtype_hierarchy = {
        "uint8": 8,
        "int8": 8,
        "uint16": 16,
        "int16": 16,
        "uint32": 32,
        "int32": 32,
        "uint64": 64,
        "int64": 64,
    }
    return max(
        tensor1.dtype,
        tensor2.dtype,
        key=lambda dtype: dtype_hierarchy[str(dtype)],
    )


@op(torch.ops.aten.pow)  # pyre-ignore[56]
def _aten_pow(x: Any, y: Any) -> Any:
    y_orig = y
    if isinstance(y, int):
        y = float(y)
    if _is_int(x) and _is_int(y_orig):
        # Do the math in float then cast
        res = jnp.power(jnp.astype(x, jnp.dtype("float")), y)
        return res.astype(highest_precision_int_dtype(x, y_orig))
    res = jnp.power(x, y)
    if isinstance(x, float):
        return res.astype(_torch_binary_scalar_type(x, y_orig))
    if isinstance(y_orig, float):
        return res.astype(_torch_binary_scalar_type(y_orig, x))
    return res


@op(torch.ops.aten.view_as_complex)  # pyre-ignore[56]
def _aten_view_as_complex(input: Any) -> Any:
    if input.dtype == jnp.bfloat16:
        input = input.astype(jnp.float32)
    x, y = input[..., 0], input[..., 1]
    return jax.lax.complex(x, y)


@op(torch.ops.aten.div)  # pyre-ignore[56]
def _aten_div(x: Any, y: Any, rounding_mode: Any = "") -> Any:
    res_dtype = None
    if _is_int(x) and _is_int(y):
        res_dtype = jnp.dtype("float32")

    if isinstance(x, float):
        res_dtype = _torch_binary_scalar_type(x, y)
    elif isinstance(y, float):
        res_dtype = _torch_binary_scalar_type(y, x)

    if rounding_mode == "floor":
        res = jnp.floor_divide(x, y)
        if _is_int(x) and _is_int(y):
            res_dtype = jnp.dtype("int64")
    else:
        res = x / y
    if rounding_mode == "trunc":
        res = jnp.trunc(res)
        if _is_int(x) and _is_int(y):
            res_dtype = jnp.dtype("int64")
    if res_dtype:
        res = res.astype(res_dtype)
    return res


@op(torch.ops.aten.true_divide)  # pyre-ignore[56]
def _aten_true_divide(x: Any, y: Any) -> Any:
    return x / y


@op(torch.ops.aten.dist)  # pyre-ignore[56]
def _aten_dist(input: Any, other: Any, p: Any = 2) -> Any:
    diff = jnp.abs(jnp.subtract(input, other))
    return _aten_linalg_vector_norm(diff, ord=p)


@op(torch.ops.aten.bmm)  # pyre-ignore[56]
def _aten_bmm(x: Any, y: Any) -> Any:
    res = x @ y
    return res
    # return jnp.einsum('bnm,bmk->bnk', x, y)


##################### Custom Embedding with padding_idx autograd hooks ###############################################
@functools.partial(jax.custom_jvp, nondiff_argnums=(2, 3, 4))
def _disable_pad_idx_grad_embedding_impl(
    a: Any, w: Any, padding_idx: Any, scale_grad_by_freq: Any, sparse: Any
) -> Any:
    """
    Forward pass: same as the original one but with the custom jvp decorator.
    It assumes pytorch already sets the embedding tensor padding row to zeros by this moment.
    """
    output_embeds = jnp.take(a, w, axis=0)
    return output_embeds


@_disable_pad_idx_grad_embedding_impl.defjvp  # pyre-ignore[56]
def _embedding_jvp(
    padding_idx: Any, scale_grad_by_freq: Any, sparse: Any, primals: Any, tangents: Any
) -> Any:
    """Custom JVP rule for embedding to handle padding_idx gradients."""
    a, w = primals
    a_dot, _w_dot = tangents

    y = _disable_pad_idx_grad_embedding_impl(  # pyre-ignore[29]
        a, w, padding_idx, scale_grad_by_freq, sparse
    )
    a_dot = a_dot.at[padding_idx].set(0.0)
    y_dot = jnp.take(a_dot, w, axis=0)
    return y, y_dot


#################################################################################


@op(torch.ops.aten.embedding)  # pyre-ignore[56]
# embedding(Tensor weight, Tensor indices, SymInt padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False)
def _aten_embedding(
    a: Any,
    w: Any,
    padding_idx: Any = -1,
    scale_grad_by_freq: Any = False,
    sparse: Any = False,
) -> Any:
    if padding_idx == -1:
        return jnp.take(a, w, axis=0)
    return _disable_pad_idx_grad_embedding_impl(  # pyre-ignore[29]
        a, w, padding_idx, scale_grad_by_freq, sparse
    )


@op(torch.ops.aten.embedding_renorm_)  # pyre-ignore[56]
def _aten_embedding_renorm_(
    weight: Any, indices: Any, max_norm: Any, norm_type: Any
) -> Any:
    # Adapted from https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Embedding.cpp
    unique_indices = jnp.unique(indices)

    norm = jnp.linalg.norm(
        _aten_embedding(weight, unique_indices),
        ord=norm_type,
        axis=1,
    )

    indice_idx = jnp.where(norm > max_norm)

    scale = max_norm / (norm[indice_idx] + 1e-7)

    indices_to_update = unique_indices[indice_idx]

    weight = weight.at[indices_to_update].set(
        weight[indices_to_update] * scale[:, None]
    )
    return weight


# - func: _embedding_bag_forward_only(
# Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False,
# int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False, int padding_idx=-1) -> (Tensor, Tensor, Tensor, Tensor)
@op(torch.ops.aten._embedding_bag)
@op(torch.ops.aten._embedding_bag_forward_only)  # pyre-ignore[56]
def _aten__embedding_bag(
    weight: Any,
    indices: Any,
    offsets: Any = None,
    scale_grad_by_freq: Any = False,
    mode: Any = 0,
    sparse: Any = False,
    per_sample_weights: Any = None,
    include_last_offset: Any = False,
    padding_idx: Any = -1,
) -> Any:
    """Jax implementation of the PyTorch _embedding_bag function.

    Args:
        weight: The learnable weights of the module of shape (num_embeddings, embedding_dim).
        indices: A LongTensor containing the indices to extract.
        offsets: A LongTensor containing the starting offset of each bag.
        scale_grad_by_freq: Whether to scale gradients by the inverse of frequency of the words in the mini-batch.
        mode: 0 = "sum", 1 = "mean" or 2 = "max"
        sparse: Whether the gradients with respect to weight should be a sparse tensor.
        per_sample_weights: If given, each embedding vector is weighted by per_sample_weights
        include_last_offset: Whether to include the last offset as a valid bag.
        padding_idx: If specified, the entries at padding_idx do not contribute to the gradient.

    Returns:
        A tuple of (output, offset2bag, bag_size, max_indices).
    """
    embedded = _aten_embedding(weight, indices, padding_idx)

    if offsets is None:
        # offsets is None only when indices.ndim > 1
        if mode == 0:  # sum
            output = jnp.sum(embedded, axis=1)
        elif mode == 1:  # mean
            output = jnp.mean(embedded, axis=1)
        elif mode == 2:  # max
            output = jnp.max(embedded, axis=1)
        return output, None, None, None  # pyre-ignore[61]

    # When include_last_offset is True, the last offset is the total number
    # of indices (a sentinel), so num_bags = len(offsets) - 1.
    if include_last_offset:
        bag_offsets = offsets[:-1]
        next_offsets = offsets[1:]
        num_bags = offsets.shape[0] - 1
    else:
        bag_offsets = offsets
        num_bags = offsets.shape[0]
        next_offsets = jnp.concatenate(
            [offsets[1:], jnp.array([indices.shape[0]], dtype=offsets.dtype)]
        )

    bag_size = next_offsets - bag_offsets

    # Build segment_ids: map each index position to its bag.
    # searchsorted(bag_offsets, positions, side='right') - 1 gives the bag id.
    positions = jnp.arange(indices.shape[0], dtype=offsets.dtype)
    segment_ids = jnp.searchsorted(bag_offsets, positions, side="right") - 1
    segment_ids = jnp.clip(segment_ids, 0, num_bags - 1)

    if per_sample_weights is not None:
        embedded = embedded * per_sample_weights[:, None]

    if mode == 0:  # sum
        output = jax.ops.segment_sum(embedded, segment_ids, num_segments=num_bags)
    elif mode == 1:  # mean
        output = jax.ops.segment_sum(embedded, segment_ids, num_segments=num_bags)
        safe_sizes = jnp.maximum(bag_size, 1).astype(embedded.dtype)
        output = output / safe_sizes[:, None]
    elif mode == 2:  # max
        output = jax.ops.segment_max(embedded, segment_ids, num_segments=num_bags)
    else:
        raise ValueError(
            f"mode can only be 0 (sum), 1 (mean) or 2 (max) but got {mode=}"
        )

    offset2bag = segment_ids
    max_indices = jnp.zeros_like(indices)

    return output, offset2bag, bag_size, max_indices


@op(torch.ops.aten._embedding_bag_backward)  # pyre-ignore[56]
def _aten__embedding_bag_backward(
    grad: Any,
    indices: Any,
    offsets: Any,
    offset2bag: Any,
    bag_size: Any,
    maximum_indices: Any,
    num_weights: int,
    scale_grad_by_freq: bool = False,
    mode: int = 0,
    sparse: bool = False,
    per_sample_weights: Any = None,
    padding_idx: int = -1,
) -> Any:
    """Backward for _embedding_bag: scatter gradients back to weight matrix."""
    embedding_dim = grad.shape[-1]

    # Expand grad from (num_bags, D) to per-index (num_indices, D)
    grad_per_index = grad[offset2bag]

    if mode == 1:  # mean: scale by 1/bag_size
        safe_sizes = jnp.maximum(bag_size[offset2bag], 1).astype(grad.dtype)
        grad_per_index = grad_per_index / safe_sizes[:, None]

    if per_sample_weights is not None:
        grad_per_index = grad_per_index * per_sample_weights[:, None]

    # Scatter-add into grad_weight of shape (num_weights, embedding_dim).
    # indices is 1-D (num_indices,); each entry selects a row in grad_weight.
    grad_weight = jnp.zeros((num_weights, embedding_dim), dtype=grad.dtype)
    grad_weight = grad_weight.at[indices].add(grad_per_index)

    return grad_weight


@op(torch.ops.aten.rsqrt)  # pyre-ignore[56]
@op_base.promote_int_input
def _aten_rsqrt(x: Any) -> Any:
    return jax.lax.rsqrt(x)


@op(torch.ops.aten.expand)
@op(torch.ops.aten.expand_copy)  # pyre-ignore[56]
def _aten_expand(x: Any, dims: Any) -> Any:
    def fix_dims(d: Any, xs: Any) -> Any:
        if d == -1:
            return xs
        return d

    shape = list(x.shape)
    if len(shape) < len(dims):
        shape = [
            1,
        ] * (len(dims) - len(shape)) + shape
        # make sure that dims and shape is the same by
        # left pad with 1s. Otherwise the zip below will
        # truncate
    dims = [fix_dims(p, s) for p, s in zip(dims, shape, strict=False)]
    return jnp.broadcast_to(x, dims)


@op(torch.ops.aten.dot)  # pyre-ignore[56]
def _aten_dot(x: Any, y: Any) -> Any:
    return jnp.dot(x, y)


@op(torch.ops.aten._to_copy)  # pyre-ignore[56]
def _aten__to_copy(self: Any, **kwargs: Any) -> Any:
    dtype = mappings.t2j_dtype(kwargs["dtype"])
    if dtype != self.dtype:
        return self.astype(dtype)
    return jnp.copy(self)


@op(torch.ops.aten.empty)  # pyre-ignore[56]
@op_base.convert_dtype(use_default_dtype=False)
def _aten_empty(size: Sequence[int], *, dtype: Any = None, **kwargs: Any) -> Any:
    return jnp.empty(size, dtype=dtype)


@op(torch.ops.aten.empty_like)  # pyre-ignore[56]
@op_base.convert_dtype(use_default_dtype=False)
def _aten_empty_like(input: Any, *, dtype: Any = None, **kwargs: Any) -> Any:
    return jnp.empty_like(input, dtype)


@op(torch.ops.aten.ones)  # pyre-ignore[56]
@op_base.convert_dtype()
def _ones(size: Sequence[int], dtype: Any = None, **kwargs: Any) -> Any:
    return jnp.ones(size, dtype)


@op(torch.ops.aten.zeros)  # pyre-ignore[56]
@op_base.convert_dtype()
def _zeros(size: Sequence[int], dtype: Any = None, **kwargs: Any) -> Any:
    return jnp.zeros(size, dtype)


@op(torch.ops.aten.full)  # pyre-ignore[56]
@op_base.convert_dtype()
def _full(
    size: Sequence[int], fill_value: Any, *, dtype: Any = None, **kwargs: Any
) -> Any:
    # TODO: handle torch.Size
    return jnp.full(size, fill_value, dtype=dtype)


@op(torch.ops.aten.empty_permuted)  # pyre-ignore[56]
@op_base.convert_dtype()
def _aten_empty_permuted(
    sizes: Any, physical_layout: Any, dtype: Any = None, **kwargs: Any
) -> Any:
    # Ignore the physical layout,
    # since JAX and torch tensor doesn't share the same memory.
    return jnp.empty(sizes, dtype=dtype)


@op(torch.ops.aten.empty_strided)  # pyre-ignore[56]
@op_base.convert_dtype()
def _aten_empty_strided(
    sizes: Any, stride: Any, dtype: Any = None, **kwargs: Any
) -> Any:
    # Ignore stride, since JAX and torch tensor doesn't share the same memory.
    return jnp.empty(sizes, dtype=dtype)


@op(torch.ops.aten.index_put)  # pyre-ignore[56]
def _aten_index_put(
    self: Any, indexes: Any, values: Any, accumulate: Any = False
) -> Any:
    indexes = [slice(None, None, None) if i is None else i for i in indexes]
    indexes = tuple(indexes)
    # Boolean indices are not supported by JAX's .at[].set() under JIT because
    # boolean indexing produces data-dependent output shapes.  When the index
    # is a single boolean mask with the same shape as self, fall back to
    # jnp.where which is JIT-compatible.
    non_slice = [i for i in indexes if not isinstance(i, slice)]
    if (
        len(non_slice) == 1
        and hasattr(non_slice[0], "dtype")
        and non_slice[0].dtype == jnp.bool_
    ):
        mask = non_slice[0]
        if accumulate:
            return jnp.where(mask, self + values, self)
        else:
            return jnp.where(mask, values, self)
    if accumulate:
        return self.at[indexes].add(values)
    else:
        return self.at[indexes].set(values)


@op(torch.ops.aten.index)
@op(torch.ops.aten._unsafe_index)
@op(torch.ops.aten.index.Tensor)  # pyre-ignore[56]
def _aten_index(self: Any, indexes: Any) -> Any:
    indexes = [slice(None, None, None) if i is None else i for i in indexes]
    indexes = tuple(indexes)
    return self[indexes]


@op(torch.ops.aten.split)
@op(torch.ops.aten.split_copy)
@op(torch.ops.aten.split_with_sizes)  # pyre-ignore[56]
def split_with_sizes(x: Any, sizes: Any, dim: Any = 0) -> Any:
    """Splits an array `x` into sub-arrays based on static sizes `sizes`.

    Args:
      x: The input array to split.
      sizes: A 1D array of integer sizes for each sub-array.

    Returns:
      A list of sub-arrays.
    """
    if isinstance(sizes, int):
        # split equal size, round up
        new_sizes = [sizes] * (-(-x.shape[dim] // sizes))
        sizes = new_sizes
    rank = x.ndim
    splits = np.cumsum(sizes)  # Cumulative sum for split points

    def make_range(rank: Any, dim: Any, start: Any, end: Any) -> Any:
        res = [slice(None, None, None)] * rank
        res[dim] = slice(start, end)
        return tuple(res)

    return [
        x[make_range(rank, dim, start, end)]
        for start, end in zip([0] + list(splits[:-1]), splits, strict=False)
    ]


@op(torch.ops.aten.permute)
@op(torch.ops.aten.permute_copy)  # pyre-ignore[56]
def permute(t: Any, dims: Any) -> Any:
    # TODO: return a View instead
    return jnp.transpose(t, dims)


@op(torch.ops.aten.unsqueeze)
@op(torch.ops.aten.unsqueeze_copy)  # pyre-ignore[56]
def _aten_unsqueeze(self: Any, dim: Any) -> Any:
    if dim < 0:
        dim += self.ndim + 1
    return jnp.expand_dims(self, dim)


@op(torch.ops.aten.ne)  # pyre-ignore[56]
def _aten_ne(x: Any, y: Any) -> Any:
    return jnp.not_equal(x, y)


# Create indices along a specific axis
#
# For example
# x = jnp.zeros((3,4))
#
# _indices_along_axis(x, axis=0)
# >> [[0], [1], [2]] shape (3, 1)
#
# _indices_along_axis(x, axis=1)
# >> [[0, 1, 2, 3]] shape (1, 4)
def _indices_along_axis(x: Any, axis: Any) -> Any:
    return jnp.expand_dims(
        jnp.arange(x.shape[axis]),
        axis=[d for d in range(len(x.shape)) if d != axis],
    )


def _broadcast_indices(indices: Any, shape: Any) -> Any:
    return jnp.broadcast_to(indices, shape)


@op(torch.ops.aten.cummax)  # pyre-ignore[56]
def _aten_cummax(x: Any, dim: Any) -> Any:
    if not x.shape:
        return x, jnp.zeros_like(x, dtype=jnp.int64)

    axis = dim

    indice_along_axis = _indices_along_axis(x, axis)
    indices = _broadcast_indices(indice_along_axis, x.shape)

    def cummax_reduce_func(carry: Any, elem: Any) -> Any:
        v1, v2 = carry["val"], elem["val"]
        i1, i2 = carry["idx"], elem["idx"]

        v = jnp.maximum(v1, v2)
        i = jnp.where(v1 > v2, i1, i2)
        return {"val": v, "idx": i}

    res = jax.lax.associative_scan(
        cummax_reduce_func, {"val": x, "idx": indices}, axis=axis
    )
    return res["val"], res["idx"]


@op(torch.ops.aten.cummin)  # pyre-ignore[56]
def _aten_cummin(x: Any, dim: Any) -> Any:
    if not x.shape:
        return x, jnp.zeros_like(x, dtype=jnp.int64)

    axis = dim

    indice_along_axis = _indices_along_axis(x, axis)
    indices = _broadcast_indices(indice_along_axis, x.shape)

    def cummin_reduce_func(carry: Any, elem: Any) -> Any:
        v1, v2 = carry["val"], elem["val"]
        i1, i2 = carry["idx"], elem["idx"]

        v = jnp.minimum(v1, v2)
        i = jnp.where(v1 < v2, i1, i2)
        return {"val": v, "idx": i}

    res = jax.lax.associative_scan(
        cummin_reduce_func, {"val": x, "idx": indices}, axis=axis
    )
    return res["val"], res["idx"]


@op(torch.ops.aten.cumsum)  # pyre-ignore[56]
def _aten_cumsum(x: Any, y: Any, dtype: Any = None) -> Any:
    if dtype:
        dtype = mappings.t2j_dtype(dtype)
    if not x.shape:
        return x
    res = jnp.cumsum(x, y, dtype)
    return res


@op(torch.ops.aten.cumprod)  # pyre-ignore[56]
def _aten_cumprod(input: Any, dim: Any, dtype: Any = None, out: Any = None) -> Any:
    if dtype:
        dtype = mappings.t2j_dtype(dtype)
    if len(input.shape) > 0:
        res = jnp.cumprod(input, axis=dim, dtype=dtype)
    elif dtype:
        res = input.astype(dtype)
    else:
        res = input
    return res


@op(torch.ops.aten.native_layer_norm)  # pyre-ignore[56]
def _aten_native_layer_norm(
    input: Any,
    normalized_shape: Any,
    weight: Any = None,
    bias: Any = None,
    eps: Any = 1e-5,
) -> Any:
    """Implements layer normalization in Jax as defined by `aten::native_layer_norm`.

    Args:
      input: The input tensor.
      normalized_shape: A list of integer dimensions to be normalized over.
      weight: Optional weight tensor for the affine transformation.
      bias: Optional bias tensor for the affine transformation.
      eps: A small epsilon value for numerical stability.

    Returns:
      output: The normalized tensor.
      mean: The calculated mean tensor.
      std: The calculated standard deviation tensor.
    """
    if isinstance(normalized_shape, int):
        normalized_shape = [normalized_shape]
    axis = [len(input.shape) - i - 1 for i in range(len(normalized_shape))]

    # Calculate mean and standard deviation
    mean = jnp.mean(input, axis=axis, keepdims=True)
    var = jnp.var(input, axis=axis, keepdims=True)
    rstd = jax.lax.rsqrt(var + eps)

    # Normalize the input
    norm_x = (input - mean) * rstd

    # Apply affine transformation (if provided)
    if weight is not None:
        norm_x *= weight
    if bias is not None:
        norm_x += bias
    return norm_x, mean, rstd


@op(torch.ops.aten.matmul)  # pyre-ignore[56]
def _aten_matmul(x: Any, y: Any) -> Any:
    return x @ y


# - func: addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
@op(torch.ops.aten.addmm)
@op(torch.ops.aten.addmv)  # pyre-ignore[56]
def _aten_addmm(
    self: Any, mat1: Any, mat2: Any, *, beta: Any = 1.0, alpha: Any = 1.0
) -> Any:
    alpha = jnp.array(alpha).astype(mat1.dtype)
    beta = jnp.array(beta).astype(mat1.dtype)
    self *= beta
    self += alpha * jnp.matmul(mat1, mat2)
    return self


@op(torch.ops.aten.sparse_sampled_addmm)  # pyre-ignore[56]
def _aten_sparse_addmm(
    self: Any, mat1: Any, mat2: Any, *, beta: Any = 1.0, alpha: Any = 1.0
) -> Any:
    alpha = jnp.array(alpha).astype(mat1.dtype)
    beta = jnp.array(beta).astype(mat1.dtype)
    self *= beta
    self += alpha * jnp.matmul(mat1, mat2) * (self != 0)
    return self


@op(torch.ops.aten.addbmm.default)  # pyre-ignore[56]
def _aten_addbmm(
    input: Any, batch1: Any, batch2: Any, *, beta: Any = 1, alpha: Any = 1
) -> Any:
    alpha = jnp.array(alpha).astype(batch1.dtype)
    beta = jnp.array(beta).astype(batch1.dtype)
    mm = jnp.einsum("bxy, byz -> xz", batch1, batch2)
    return jax.lax.cond(
        beta == 0, lambda: alpha * mm, lambda: beta * input + alpha * mm
    )


@op(torch.ops.aten.gelu)  # pyre-ignore[56]
def _aten_gelu(self: Any, *, approximate: Any = "none") -> Any:
    approx = approximate == "tanh"
    return jax.nn.gelu(self, approx)


@op(torch.ops.aten.squeeze)
@op(torch.ops.aten.squeeze_copy)  # pyre-ignore[56]
def _aten_squeeze_dim(self: Any, dim: Any = None) -> Any:
    if self.ndim == 0:
        return self
    if dim is not None:
        if isinstance(dim, int):
            if self.shape[dim] != 1:
                return self
            if dim < 0:
                dim += self.ndim
        else:
            # NOTE: torch leaves the dims that is not 1 unchanged,
            # but jax raises error.
            dim = [i if i >= 0 else (i + self.ndim) for i in dim if self.shape[i] == 1]

    return jnp.squeeze(self, dim)


@op(torch.ops.aten.bucketize)  # pyre-ignore[56]
def _aten_bucketize(
    input: Any,
    boundaries: Any,
    *,
    out_int32: Any = False,
    right: Any = False,
    out: Any = None,
) -> Any:
    return_type = jnp.int32 if out_int32 else jnp.int64
    # torch.bucketize(right=False) -> #{boundaries < x}  (side="left");
    # torch.bucketize(right=True)  -> #{boundaries <= x} (side="right").
    # ``_searchsorted`` avoids the binary-search while loop for small boundaries.
    side = "right" if right else "left"
    return _searchsorted(boundaries, input, side).astype(return_type)


@op(torch.ops.aten.conv2d)  # pyre-ignore[56]
def _aten_conv2d(
    input: Any,
    weight: Any,
    bias: Any,
    stride: Any,
    padding: Any,
    dilation: Any,
    groups: Any,
) -> Any:
    return _aten_convolution(
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        transposed=False,
        output_padding=1,
        groups=groups,
    )


@op(torch.ops.aten.convolution)  # pyre-ignore[56]
def _aten_convolution(
    input: Any,
    weight: Any,
    bias: Any,
    stride: Any,
    padding: Any,
    dilation: Any,
    transposed: Any,
    output_padding: Any,
    groups: Any,
) -> Any:
    num_shape_dim = weight.ndim - 1
    batch_dims = input.shape[:-num_shape_dim]

    input = input.reshape((-1, *input.shape[-num_shape_dim:]))

    def make_padding(padding: Any, num_spatial_dims: Any) -> Any:
        # Expand single padding to pairs expected by jax
        if len(padding) == 1 and len(padding) < num_spatial_dims:
            padding *= num_spatial_dims
        if transposed:
            # See https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html
            pad_out = []
            for i in range(num_spatial_dims):
                front = dilation[i] * (weight.shape[i + 2] - 1) - padding[i]
                back = front + output_padding[i]
                pad_out.append((front, back))
            return pad_out
        else:
            return ((p, p) for p in padding)

    def create_default_conv_dimension_numbers(num_spatial_dims: Any) -> Any:
        # Ref: https://github.com/openxla/xla/blob/main/xla/client/xla_builder.cc#L4211
        # (batch dimension, feature dimension, spatial dimensions...)
        lhs_spec = [0, 1]
        # (out feature dimension, in feature dimension, spatial dimensions...)
        # swapped for transposed convolution
        rhs_spec = [1, 0] if transposed else [0, 1]
        # (batch dimension, feature dimension, spatial dimensions...)
        out_spec = [0, 1]
        for i in range(0, num_spatial_dims):
            lhs_spec.append(i + 2)
            rhs_spec.append(i + 2)
            out_spec.append(i + 2)
        return jax.lax.ConvDimensionNumbers(
            *map(tuple, (lhs_spec, rhs_spec, out_spec))  # pyre-ignore[6]
        )

    if transposed:
        rhs = jnp.flip(weight, range(2, 1 + num_shape_dim))
        if groups != 1:
            # reshape filters for tranposed depthwise convolution
            assert rhs.shape[0] % groups == 0
            rhs_shape = [rhs.shape[0] // groups, rhs.shape[1] * groups]
            rhs_shape.extend(rhs.shape[2:])
            rhs = jnp.reshape(rhs, rhs_shape)
        res = jax.lax.conv_general_dilated(
            input,
            rhs,
            (1,) * len(stride),
            make_padding(padding, len(stride)),
            lhs_dilation=stride,
            rhs_dilation=dilation,
            dimension_numbers=create_default_conv_dimension_numbers(len(stride)),
            feature_group_count=groups,
            batch_group_count=1,
        )
    else:
        res = jax.lax.conv_general_dilated(
            input,
            weight,
            stride,
            make_padding(padding, len(stride)),
            lhs_dilation=(1,) * len(stride),
            rhs_dilation=dilation,
            dimension_numbers=create_default_conv_dimension_numbers(len(stride)),
            feature_group_count=groups,
            batch_group_count=1,
        )

    if bias is not None:
        # TODO(qihqi): bias always on channel?
        if len(bias.shape) == 1:
            shape = [1] * len(res.shape)
            shape[1] = bias.shape[0]
            bias = bias.reshape(tuple(shape))
        res = res + bias

    res = res.reshape((*batch_dims, *res.shape[-num_shape_dim:]))  # pyre-ignore[60]
    return res


# _native_batch_norm_legit(Tensor input, Tensor? weight, Tensor? bias, Tensor(a!) running_mean, Tensor(b!) running_var, bool training, float momentum, float eps)
@op(torch.ops.aten._native_batch_norm_legit.default)  # pyre-ignore[56]
def _aten__native_batch_norm_legit(
    input: Any,
    weight: Any,
    bias: Any,
    running_mean: Any,
    running_var: Any,
    training: Any,
    momentum: Any,
    eps: Any,
) -> Any:
    """JAX implementation of batch normalization with optional parameters.
    Refers to https://github.com/pytorch/pytorch/blob/cd3a71f754a2248bcfe500de7c9860bd7d2002bf/torch/_decomp/decompositions.py#L1713.

    Args:
      input (DeviceArray): Input data (N, C, H, W).
      running_mean ([DeviceArray]): Running mean of input (C,).
      running_var ([DeviceArray]): Running variance of input (C,).
      weight (Optional[DeviceArray]): Scaling factor (gamma) (C,). Can be None.
      bias (Optional[DeviceArray]): Shift factor (beta) (C,). Can be None.
      training (bool): If True, use batch statistics for normalization.
                       If False, use running statistics.
      momentum (float): Momentum factor for updating running statistics.
      eps (float): Small constant for numerical stability.

    Returns:
      DeviceArray: Normalized output
      DeviceArray: Batch mean (C,) or empty if training is False
      DeviceArray: Reversed batch variance (C,) or empty if training is False
    """
    reduction_dims = [0] + list(range(2, input.ndim))
    reshape_dims = [1, -1] + [1] * (input.ndim - 2)
    if training:
        # Calculate batch mean and variance
        mean = jnp.mean(input, axis=reduction_dims, keepdims=True)
        saved_mean = jnp.squeeze(mean, reduction_dims)
        var = jnp.var(input, axis=reduction_dims)
        rstd = jax.lax.rsqrt(var.reshape(reshape_dims) + eps)
        # Update running statistics using momentum
        running_mean = (1 - momentum) * running_mean + momentum * saved_mean
        running_var = (1 - momentum) * running_var + momentum * var
        saved_rstd = jnp.squeeze(rstd, reduction_dims)
    else:
        rstd = jax.lax.rsqrt(running_var.reshape(reshape_dims) + eps)
        saved_mean = jnp.array(
            [], dtype=input.dtype
        )  # No need to calculate batch statistics in inference mode
        saved_rstd = jnp.array([], dtype=input.dtype)

    # Normalize
    if training:
        # use batch statistics if training
        x_hat = (input - mean) * rstd  # pyre-ignore[61]
    else:
        # Use running statistics in inference mode
        x_hat = (input - running_mean.reshape(reshape_dims)) * rstd

    # Scale and shift
    if weight is not None:
        x_hat *= weight.reshape(reshape_dims)  # Reshape weight for broadcasting
    if bias is not None:
        x_hat += bias.reshape(reshape_dims)  # Reshape bias for broadcasting

    return x_hat, saved_mean, saved_rstd


@op(torch.ops.aten._native_batch_norm_legit_no_training)  # pyre-ignore[56]
def _aten__native_batch_norm_legit_no_training(
    input: Any,
    weight: Any,
    bias: Any,
    running_mean: Any,
    running_var: Any,
    momentum: Any,
    eps: Any,
) -> Any:
    return _aten__native_batch_norm_legit(
        input, weight, bias, running_mean, running_var, False, momentum, eps
    )


@op(torch.ops.aten.relu)  # pyre-ignore[56]
def _aten_relu(self: Any) -> Any:
    return jax.nn.relu(self)


@op(torch.ops.aten.cat)  # pyre-ignore[56]
def _aten_cat(tensors: Any, dims: Any = 0) -> Any:
    # handle empty tensors as a special case.
    # torch.cat will ignore the empty tensor, while jnp.concatenate
    # will error if the dims > 0.
    filtered_tensors = [t for t in tensors if not (t.ndim == 1 and t.shape[0] == 0)]
    if filtered_tensors:
        return jnp.concatenate(filtered_tensors, dims)
    return tensors[0]


def _ceil_mode_padding(
    padding: list[int],
    input_shape: list[int],
    kernel_size: list[int],
    stride: list[int],
    dilation: list[int],
    ceil_mode: bool,
) -> Any:
    """Creates low and high padding specification for the given padding (which is symmetric) and ceil mode.

    Additional high padding could be required when ceil mode is set.
    """
    ceil_mode_padding = []
    for i in range(len(padding)):
        left_padding = padding[i]
        right_padding = left_padding

        input_size = input_shape[2 + i]
        output_size_rem = (
            input_size + 2 * left_padding - (kernel_size[i] - 1) * dilation[i] - 1
        ) % stride[i]
        if ceil_mode and output_size_rem != 0:
            extra_padding = stride[i] - output_size_rem
            new_output_size = (
                input_size
                + left_padding
                + right_padding
                + extra_padding
                - (kernel_size[i] - 1) * dilation[i]
                - 1
                + stride[i]
                - 1
            ) // stride[i] + 1
            # Ensure that the last pooling starts inside the image.
            size_to_compare = input_size + left_padding

            if (new_output_size - 1) * stride[i] < size_to_compare:
                right_padding += extra_padding

        ceil_mode_padding.append((left_padding, right_padding))
    return ceil_mode_padding


def max_pool(
    inputs: Any,
    kernel_size: Any,
    strides: Any = None,
    padding: Any = 0,
    dilation: Any = 1,
    ceil_mode: Any = False,
    with_index: Any = False,
) -> Any:
    num_batch_dims = len(inputs.shape) - len(kernel_size) - 1
    kernel_size = tuple(kernel_size)
    # Default stride is kernel_size
    strides = tuple(strides) if strides else kernel_size
    if isinstance(padding, int):
        padding = [padding for _ in range(len(kernel_size))]
    if isinstance(dilation, int):
        dilation = tuple(dilation for _ in range(len(kernel_size)))
    elif isinstance(dilation, list):
        dilation = tuple(dilation)

    input_shape = inputs.shape
    if num_batch_dims == 0:
        input_shape = [1, *input_shape]
    padding = _ceil_mode_padding(
        # pyrefly: ignore [bad-argument-type]
        padding,
        input_shape,
        # pyrefly: ignore [bad-argument-type]
        kernel_size,
        # pyrefly: ignore [bad-argument-type]
        strides,
        dilation,
        ceil_mode,
    )

    assert len(kernel_size) == len(strides), (
        f"len({kernel_size=}) must equal len({strides=})"
    )
    assert len(kernel_size) == len(dilation), (
        f"len({kernel_size=}) must equal len({dilation=})"
    )
    strides = (1,) * (1 + num_batch_dims) + strides
    dims = (1,) * (1 + num_batch_dims) + kernel_size
    dilation = (1,) * (1 + num_batch_dims) + dilation

    is_single_input = False
    if num_batch_dims == 0:
        # add singleton batch dimension because lax.reduce_window always
        # needs a batch dimension.
        inputs = inputs[None]
        strides = (1,) + strides
        dims = (1,) + dims
        dilation = (1,) + dilation
        is_single_input = True

    assert inputs.ndim == len(dims), f"len({inputs.shape}) != len({dims})"
    if not isinstance(padding, str):
        padding = tuple(map(tuple, padding))  # pyre-ignore[6]
        assert len(padding) == len(kernel_size), (
            f"padding {padding} must specify pads for same number of dims as "
            f"kernel_size {kernel_size}"
        )
        assert all(len(x) == 2 for x in padding), (
            f"each entry in padding {padding} must be length 2"
        )
        padding = ((0, 0), (0, 0)) + padding

    indices = jnp.arange(np.prod(inputs.shape[-len(kernel_size) :]))
    indices = indices.reshape(inputs.shape[-len(kernel_size) :])
    indices = jnp.broadcast_to(indices, inputs.shape)

    return_dtype = inputs.dtype
    if jnp.issubdtype(inputs.dtype, jnp.integer):
        init_val = -np.array(1 << 31).astype(np.int32)
        inputs = inputs.astype(jnp.array(init_val).dtype)
    else:
        init_val = -jnp.inf
        inputs = inputs.astype(jnp.array(init_val).dtype)

    y = jax.lax.reduce_window(
        inputs,
        init_val,
        jax.lax.max,
        dims,
        strides,
        padding,
        window_dilation=dilation,
    )

    if is_single_input:
        y = jnp.squeeze(y, axis=0)

    y = y.astype(return_dtype)

    if with_index:

        def reduce_fn(a: Any, b: Any) -> Any:
            ai, av = a
            bi, bv = b
            which = av >= bv  # torch breaks ties in favor of later indices
            return jnp.where(which, ai, bi), jnp.where(which, av, bv)

        indices, _ = jax.lax.reduce_window(
            (indices, jax.lax.stop_gradient(inputs)),
            (0, init_val),
            reduce_fn,
            dims,
            strides,
            padding,
            window_dilation=dilation,
        )
        if is_single_input:
            indices = jnp.squeeze(indices, axis=0)
        return y, indices
    else:
        return y


@op(torch.ops.aten.max_pool2d_with_indices)
@op(torch.ops.aten.max_pool3d_with_indices)  # pyre-ignore[56]
def _aten_max_pool2d_with_indices(
    inputs: Any,
    kernel_size: Any,
    strides: Any = None,
    padding: Any = 0,
    dilation: Any = 1,
    ceil_mode: Any = False,
) -> Any:
    return max_pool(
        inputs,
        kernel_size,
        strides,
        padding,
        dilation,
        ceil_mode,
        with_index=True,
    )


# Aten ops registered under the `xla` library.
try:

    @op(torch.ops.xla.max_pool2d_forward)  # pyre-ignore[56]
    def _xla_max_pool2d_forward(*args: Any, **kwargs: Any) -> Any:
        return _aten_max_pool2d_with_indices(*args, **kwargs)[0]

    @op(torch.ops.xla.aot_mark_sharding)  # pyre-ignore[56]
    def _xla_aot_mark_sharding(t: Any, mesh: str, partition_spec: str) -> Any:
        import ast

        import torch_xla.distributed.spmd as xs  # pyre-fixme[21]
        from jax.sharding import NamedSharding, PartitionSpec as P

        pmesh = xs.Mesh.from_str(mesh)
        assert pmesh is not None
        partition_spec_eval = ast.literal_eval(partition_spec)
        jmesh = pmesh.get_jax_mesh()
        return jax.lax.with_sharding_constraint(
            t, NamedSharding(jmesh, P(*partition_spec_eval))
        )

    @op(torch.ops.xla.einsum_linear_forward)  # pyre-ignore[56]
    def _xla_einsum_linear_forward(input: Any, weight: Any, bias: Any) -> Any:
        with jax.named_scope("einsum_linear_forward"):
            product = jax.numpy.einsum("...n,mn->...m", input, weight)
            if bias is not None:
                return product + bias
            return product

except AttributeError:
    pass

# TODO add more ops


@op(torch.ops.aten.min)  # pyre-ignore[56]
def _aten_min(x: Any, dim: Any = None, keepdim: Any = False) -> Any:
    if dim is not None:
        return _with_reduction_scalar(jnp.min, x, dim, keepdim), _with_reduction_scalar(
            jnp.argmin, x, dim, keepdim
        ).astype(jnp.int64)
    else:
        return _with_reduction_scalar(jnp.min, x, dim, keepdim)


@op(torch.ops.aten.mode)  # pyre-ignore[56]
def _aten_mode(
    input: Any, dim: Any = -1, keepdim: Any = False, *, out: Any = None
) -> Any:
    if input.ndim == 0:  # single number
        return input, jnp.array(0)
    dim = (
        input.ndim + dim
    ) % input.ndim  # jnp.scipy.stats.mode does not accept -1 as dim
    # keepdims must be True for accurate broadcasting
    mode, _ = jax.scipy.stats.mode(input, axis=dim, keepdims=True)
    mode_broadcast = jnp.broadcast_to(mode, input.shape)
    if not keepdim:
        mode = mode.squeeze(axis=dim)
    indices = jnp.argmax(jnp.equal(mode_broadcast, input), axis=dim, keepdims=keepdim)
    return mode, indices


@op(torch.ops.aten.amin)  # pyre-ignore[56]
def _aten_amin(x: Any, dim: Any = None, keepdim: Any = False) -> Any:
    return _with_reduction_scalar(jnp.amin, x, dim, keepdim)


@op(torch.ops.aten.argmin)  # pyre-ignore[56]
def _aten_argmin(self: Any, dim: Any = None, keepdim: Any = False) -> Any:
    return _with_reduction_scalar(jnp.argmin, self, dim, keepdim)


@op(torch.ops.aten.sin)  # pyre-ignore[56]
@op_base.promote_int_input
def _aten_sin(x: Any) -> Any:
    return jnp.sin(x)


@op(torch.ops.aten.sym_size)  # pyre-ignore[56]
def _aten_sym_size(x: Any, dim: Any) -> Any:
    return x.shape[dim]


@op(torch.ops.aten.var.correction)
@op(torch.ops.prims.var)  # pyre-ignore[56]
def _aten_var(
    x: Any,
    dim: Any = None,
    *,
    correction: Any = 1,
    keepdim: Any = False,
    out: Any = None,
) -> Any:
    return jnp.var(x, axis=dim, ddof=correction, keepdims=keepdim)


@op(torch.ops.prims.broadcast_in_dim)  # pyre-ignore[56]
def _prims_broadcast_in_dim(t: Any, shape: Any, broadcast_dimensions: Any) -> Any:
    return jax.lax.broadcast_in_dim(t, shape, broadcast_dimensions=broadcast_dimensions)


# aten.native_group_norm -- should use decomp table
# func: native_group_norm(Tensor input, Tensor? weight, Tensor? bias, SymInt N, SymInt C, SymInt HxW, int group, float eps) -> (Tensor, Tensor, Tensor)


@op(torch.ops.aten.native_group_norm)  # pyre-ignore[56]
def _aten_native_group_norm(
    input: Any,
    weight: Any,
    bias: Any,
    N: Any,
    C: Any,
    HxW: Any,
    group: Any,
    eps: Any = 1e-5,
) -> Any:
    """Group Normalization implementation in JAX.

    Args:
      input: Input tensor. Expected shape (batch_size, channels, ... spatial dims
        ...)
      weight: Optional scaling (gamma) parameter. Shape (channels,)
      bias: Optional shifting (beta) parameter. Shape (channels,)
      N: Batch size.
      C: Number of channels.
      HxW: Product of spatial dimensions (number of elements per channel after
        flattening).
      group: Number of groups for Group Normalization.
      eps: Small value added for numerical stability.

    Returns:
      A tuple of (normalized_output, mean, rstd)
    """

    input_shape = input.shape

    if 0 in input_shape:
        return input, input, input

    # Reshape for group-wise normalization
    reshaped_input = jnp.reshape(input, (1, N * group, -1))

    # **Core Group Normalization**
    def group_norm_body(x: Any) -> Any:  # Function to apply within each group
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        rstd = jax.lax.rsqrt(var + eps)  # Reciprocal of std with epsilon
        normalized = (x - mean) * rstd
        return normalized, mean, rstd

    normalized, group_mean, group_rstd = jax.lax.map(group_norm_body, reshaped_input)

    # Reshape back to original input shape
    output = jnp.reshape(normalized, input_shape)

    # **Affine transformation**
    affine_shape = [
        -1 if i == 1 else 1 for i in range(input.ndim)
    ]  # Shape for broadcasting
    if weight is not None and bias is not None:
        output = bias.reshape(affine_shape) + output * weight.reshape(affine_shape)
    elif weight is not None:
        output = output * weight.reshape(affine_shape)
    elif bias is not None:
        output = output + bias.reshape(affine_shape)

    # Reshape mean and rstd
    mean = jnp.reshape(group_mean, (N, group))
    rstd = jnp.reshape(group_rstd, (N, group))

    return output, mean, rstd


@op(torch.ops.aten.linalg_vector_norm)  # pyre-ignore[56]
def _aten_linalg_vector_norm(
    self: Any, ord: Any = 2, dim: Any = None, keepdim: Any = False, dtype: Any = None
) -> Any:
    """Calculates the vector norm along specified dimensions.

    Args:
        self: The input tensor.
        ord: The order of the norm. Can be a float or 'inf', '-inf', 'fro'.
          Default is 2 (Euclidean norm).
        dim: Dimensions along which to calculate the norm. If None, the norm is
          calculated over all dimensions.
        keepdim: Whether to keep the reduced dimensions.
        dtype: Optional data type for the output.

    Returns:
        The tensor containing the calculated vector norms.
    """

    if ord not in {2, float("inf"), float("-inf"), "fro"} and not isinstance(
        ord, (int, float)
    ):
        raise ValueError(
            f"Unsupported ord value: {ord}. Supported values are 2, inf, -inf, and 'fro'."
        )

    # Special cases (for efficiency and clarity)
    if ord == 0:
        if self.shape == ():
            # float sets it to float64. set it back to input type
            result = jnp.astype(jnp.array(float(self != 0)), self.dtype)
        else:
            result = _with_reduction_scalar(
                jnp.sum, jnp.where(self != 0, 1, 0), dim, keepdim
            )

    elif ord == 2:  # Euclidean norm
        result = jnp.sqrt(
            _with_reduction_scalar(jnp.sum, jnp.abs(self) ** 2, dim, keepdim)
        )

    elif ord == float("inf"):
        result = _with_reduction_scalar(jnp.max, jnp.abs(self), dim, keepdim)

    elif ord == float("-inf"):
        result = _with_reduction_scalar(jnp.min, jnp.abs(self), dim, keepdim)

    elif ord == "fro":  # Frobenius norm
        result = jnp.sqrt(
            _with_reduction_scalar(jnp.sum, jnp.abs(self) ** 2, dim, keepdim)
        )

    else:  # General case (e.g., ord = 1, ord = 3)
        result = _with_reduction_scalar(
            jnp.sum,
            jnp.abs(self) ** ord,  # pyre-ignore[58]
            dim,
            keepdim,
        ) ** (1.0 / ord)  # pyre-ignore[58]

    # (Optional) dtype conversion
    if dtype is not None:
        result = jnp.astype(result, self.dtype)

    new_dtype = mappings.t2j_dtype(torch.get_default_dtype())
    if result.dtype == jax.numpy.int64:
        result = result.astype(new_dtype)
    return result


# aten.reflection_pad1d
@op(torch.ops.aten.reflection_pad1d)  # pyre-ignore[56]
def _aten_reflection_pad1d(input: Any, padding: Any) -> Any:
    rank = len(input.shape)
    pad_size = [(0, 0)] * rank
    pad_size[-1] = padding
    return jnp.pad(input, pad_size, mode="reflect")


# aten.alias
@op(torch.ops.aten.alias)  # pyre-ignore[56]
def _aten_alias(self: Any, *args: Any) -> Any:
    return self


# aten.sinh
@op(torch.ops.aten.sinh)  # pyre-ignore[56]
@op_base.promote_int_input
def _aten_sinh(self: Any) -> Any:
    return jnp.sinh(self)


# aten.native_layer_norm_backward
@op(torch.ops.aten.native_layer_norm_backward)  # pyre-ignore[56]
def _aten_native_layer_norm_backward(
    grad_out: Any,
    input: Any,
    normalized_shape: Any,
    mean: Any,
    rstd: Any,
    weight: Any = None,
    bias: Any = None,
    output_mask: Any = None,
) -> Any:
    """aten::native_layer_norm_backward.

    The previous signature took (grad_out, input, normalized_shape, weight,
    bias, eps) -- six parameters against aten's eight -- so every call raised
    TypeError before reaching the body, and had it not, ``mean`` and ``rstd``
    would have been read as ``weight`` and ``bias``. It also called
    ``jax.lax.native_layer_norm_backward``, which does not exist. The compiled
    path decomposes this op and so never reached it; eager torchax does reach
    it, and the forward had a test while the backward did not.

    Returns (grad_input, grad_weight, grad_bias), each None where
    ``output_mask`` is False, matching aten.
    """
    ndim = len(normalized_shape)
    axes = tuple(range(input.ndim - ndim, input.ndim))
    # mean and rstd arrive with the normalized axes already reduced away but
    # still present as size-1 dims in aten's layout; reshape defensively so the
    # broadcast is right for any leading-shape.
    stat_shape = list(input.shape[: input.ndim - ndim]) + [1] * ndim
    mean_b = jnp.reshape(mean, stat_shape)
    rstd_b = jnp.reshape(rstd, stat_shape)

    x_hat = (input - mean_b) * rstd_b
    grad_x_hat = grad_out if weight is None else grad_out * weight

    n = 1
    for a in axes:
        n *= input.shape[a]
    mean_grad = jnp.mean(grad_x_hat, axis=axes, keepdims=True)
    mean_grad_xhat = jnp.mean(grad_x_hat * x_hat, axis=axes, keepdims=True)
    grad_input = rstd_b * (grad_x_hat - mean_grad - x_hat * mean_grad_xhat)

    batch_axes = tuple(range(0, input.ndim - ndim))
    grad_weight = None if weight is None else jnp.sum(grad_out * x_hat, axis=batch_axes)
    grad_bias = None if bias is None else jnp.sum(grad_out, axis=batch_axes)

    mask = (True, True, True) if output_mask is None else tuple(output_mask)
    return (
        grad_input if mask[0] else None,
        grad_weight if len(mask) > 1 and mask[1] else None,
        grad_bias if len(mask) > 2 and mask[2] else None,
    )


# aten.reflection_pad3d_backward
# aten.reflection_pad2d


# aten.atanh
@op(torch.ops.aten.atanh)  # pyre-ignore[56]
@op_base.promote_int_input
def _aten_atanh(self: Any) -> Any:
    res = jnp.arctanh(self)
    return res


# aten.bincount
@op(torch.ops.aten.bincount)  # pyre-ignore[56]
def _aten_bincount(input: Any, weights: Any = None, minlength: Any = 0) -> Any:
    return jnp.bincount(input, weights, minlength)


# aten.bitwise_not
@op(torch.ops.aten.bitwise_not)  # pyre-ignore[56]
def _aten_bitwise_not(self: Any) -> Any:
    return ~self


# aten.bitwise_left_shift
@op(torch.ops.aten.__lshift__)
@op(torch.ops.aten.bitwise_left_shift)  # pyre-ignore[56]
def _aten_bitwise_left_shift(input: Any, other: Any) -> Any:
    return jnp.left_shift(input, other)


# aten.bitwise_right_shift
@op(torch.ops.aten.__rshift__)
@op(torch.ops.aten.bitwise_right_shift)  # pyre-ignore[56]
def _aten_bitwise_right_shift(input: Any, other: Any) -> Any:
    return jnp.right_shift(input, other)


# aten.embedding_dense_backward
#
# Backward of F.embedding for a DATA-PARALLEL (DP-replicated) weight.
#
# This op is deliberately kept OUT of recir.torchax.decompositions.DECOMPOSITIONS.
# The default torch decomposition lowers it to a generic scatter-add
# (aten.index_put). Under GSPMD, a scatter-add into a {replicated} table whose
# indices/updates are batch-sharded on the scatter (token) dim is partitioned by
# ALL-GATHERING the indices and the per-token grad up to the global token set and
# then doing one global scatter -- i.e. it all-reduces the huge [N_global, D]
# activation grad (17.8GB / ~87ms per table in a 128-way DP run) instead
# of the small [num_weights, D] table grad. To force the cheap order (per-device
# LOCAL scatter-add, then ONE all-reduce of [num_weights, D]) we make the
# per-device locality explicit with shard_map + psum.
#
# ASSUMPTIONS (hold for recir's single "torch_dist" DP mesh):
#   * weight is DP-replicated. Row-/FSDP-sharded embedding tables go through
#     recir.seq_embedding (ring-shift) or aten._embedding_bag, NOT aten.embedding,
#     so the remaining aten.embedding calls are the replicated dense/EC tables.
#   * indices / grad_output are batch-sharded on axis 0 (per-sample inputs), the
#     same convention as recir.shard_batch0 and the ring-shift path.
# Single-device runs short-circuit to a plain local scatter (no collective).
_DP_AXIS = "torch_dist"


@op(torch.ops.aten.embedding_dense_backward)  # pyre-ignore[56]
def _aten_embedding_dense_backward(
    grad_output: Any,
    indices: Any,
    num_weights: Any,
    padding_idx: Any = -1,
    scale_grad_by_freq: Any = False,
) -> Any:
    dim = grad_output.shape[-1]
    num_weights = int(num_weights)

    def _table_grad(grad_flat: Any, idx_flat: Any) -> Any:
        gw = jnp.zeros((num_weights, dim), dtype=grad_flat.dtype)
        return gw.at[idx_flat].add(grad_flat)

    def _counts(idx_flat: Any) -> Any:
        c = jnp.zeros((num_weights,), dtype=jnp.float32)
        return c.at[idx_flat].add(1.0)

    def _scale(gw: Any, counts: Any) -> Any:
        # pyrefly: ignore [missing-attribute]
        inv = jnp.where(counts > 0, 1.0 / counts, 0.0).astype(gw.dtype)
        return gw * inv[:, None]

    world_size = len(jax.devices())
    if world_size <= 1:
        # No DP axis to reduce over -- plain local scatter-add.
        grad_flat = grad_output.reshape(-1, dim)
        idx_flat = indices.reshape(-1)
        grad_weight = _table_grad(grad_flat, idx_flat)
        if scale_grad_by_freq:
            grad_weight = _scale(grad_weight, _counts(idx_flat))
    else:
        mesh = jax.sharding.Mesh(jax.devices(), (_DP_AXIS,))
        spec = jax.sharding.PartitionSpec

        def _per_device(grad_out_s: Any, idx_s: Any) -> Any:
            # Runs on each device with only its local batch shard. The scatter is
            # therefore local; psum sums the small [num_weights, D] table grad.
            grad_flat = grad_out_s.reshape(-1, dim)
            idx_flat = idx_s.reshape(-1)
            gw = jax.lax.psum(_table_grad(grad_flat, idx_flat), axis_name=_DP_AXIS)
            if scale_grad_by_freq:
                counts = jax.lax.psum(_counts(idx_flat), axis_name=_DP_AXIS)
                gw = _scale(gw, counts)
            return gw

        current_mesh = jax.sharding.get_abstract_mesh()
        if _DP_AXIS in current_mesh.manual_axes:
            grad_weight = _per_device(grad_output, indices)
        else:
            grad_weight = jax.shard_map(
                _per_device,
                mesh=mesh,
                in_specs=(
                    spec(_DP_AXIS, *([None] * (grad_output.ndim - 1))),
                    spec(_DP_AXIS, *([None] * (indices.ndim - 1))),
                ),
                out_specs=spec(),  # replicated after psum
                check_vma=False,
                # pyrefly: ignore [bad-argument-count]
            )(grad_output, indices)

    if padding_idx is not None and int(padding_idx) >= 0:
        # pyrefly: ignore [missing-attribute]
        grad_weight = grad_weight.at[int(padding_idx)].set(0.0)
    return grad_weight


# aten.sum
@op(torch.ops.aten.sum)  # pyre-ignore[56]
def _aten_sum(
    self: Any, dim: Any = None, keepdim: Any = False, dtype: Any = None
) -> Any:
    if not dim:
        dim = None
    return _with_reduction_scalar(jnp.sum, self, dim, keepdim)


# aten.sqrt
@op(torch.ops.aten.sqrt)  # pyre-ignore[56]
@op_base.promote_int_input
def _aten_sqrt(self: Any) -> Any:
    return jnp.sqrt(self)


@op(torch.ops.aten.tan)
@op_base.promote_int_input
def _aten_tan(self: Any) -> Any:
    res = jnp.tan(self)
    return res


# aten.tanh
@op(torch.ops.aten.tanh)  # pyre-ignore[56]
@op_base.promote_int_input
def _aten_tanh(self: Any) -> Any:
    res = jnp.tanh(self)
    return res


# aten.ceil
@op(torch.ops.aten.ceil)  # pyre-ignore[56]
def _aten_ceil(self: Any) -> Any:
    return jnp.ceil(self).astype(self.dtype)


# aten.asin
@op(torch.ops.aten.asin)  # pyre-ignore[56]
@op_base.promote_int_input
def _aten_asin(self: Any) -> Any:
    res = jnp.arcsin(self)
    return res


# aten.minimum
@op(torch.ops.aten.minimum)  # pyre-ignore[56]
def _aten_minimum(self: Any, other: Any) -> Any:
    return jnp.minimum(self, other)


# aten.max_pool2d_backward


def _scatter_index(dim: Any, index: Any) -> Any:
    """Returns a tuple of indexes;

    The first is to select in input (to modify),
    the second is to select from the values.
    """
    index_shape = list(index.shape)
    input_indexes = []
    source_indexes = []
    if dim < 0:
        dim += len(index_shape)
    for i in range(len(index_shape)):
        source_indexes.append(slice(0, index_shape[i]))
        if i == dim:
            input_indexes.append(index)
        else:
            target_shape = [1] * len(index_shape)
            target_shape[i] = index_shape[i]
            input_indexes.append(
                jnp.broadcast_to(
                    jnp.arange(index_shape[i]).reshape(target_shape),
                    index_shape,
                )
            )
    return tuple(input_indexes), tuple(source_indexes)


# aten.scatter_add
@op(torch.ops.aten.scatter_add)  # pyre-ignore[56]
def _aten_scatter_add(input: Any, dim: Any, index: Any, src: Any) -> Any:
    """JAX implementation of scatter, mimicking torch.scatter behavior"""

    input_indexes, source_indexes = _scatter_index(dim, index)
    return input.at[input_indexes].add(src[source_indexes])


# aten.masked_scatter
@op(torch.ops.aten.masked_scatter)  # pyre-ignore[56]
def _aten_masked_scatter(self: Any, mask: Any, source: Any) -> Any:
    broadcast_shape = jnp.broadcast_shapes(self.shape, mask.shape)

    if self.shape != broadcast_shape:
        self = jnp.broadcast_to(self, broadcast_shape)
    elif mask.shape != broadcast_shape:
        mask = jnp.broadcast_to(mask, broadcast_shape)

    self_flat = self.flatten()
    mask_flat = mask.flatten()
    source_flat = source.flatten()

    true_indices = jnp.where(mask_flat)[0]
    self_flat = self_flat.at[true_indices].set(source_flat[: len(true_indices)])
    final_arr = self_flat.reshape(self.shape)

    return final_arr


@op(torch.ops.aten.masked_select)  # pyre-ignore[56]
def _aten_masked_select(self: Any, mask: Any, *args: Any, **kwargs: Any) -> Any:
    broadcast_shape = jnp.broadcast_shapes(self.shape, mask.shape)

    if self.shape != broadcast_shape:
        self = jnp.broadcast_to(self, broadcast_shape)
    if mask.shape != broadcast_shape:
        mask = jnp.broadcast_to(mask, broadcast_shape)

    self_flat = self.flatten()
    mask_flat = mask.flatten()
    true_indices = jnp.where(mask_flat)[0]

    return self_flat[true_indices]


# aten.logical_not


# aten.sign
@op(torch.ops.aten.sign)  # pyre-ignore[56]
def _aten_sign(x: Any) -> Any:
    return jnp.sign(x)


# aten.signbit
@op(torch.ops.aten.signbit)  # pyre-ignore[56]
def _aten_signbit(x: Any) -> Any:
    return jnp.signbit(x)


# aten.sigmoid
@op(torch.ops.aten.sigmoid)  # pyre-ignore[56]
@op_base.promote_int_input
def _aten_sigmoid(x: Any) -> Any:
    return jax.nn.sigmoid(x)


# implement aten.asinh in jax
@op(torch.ops.aten.asinh)  # pyre-ignore[56]
@op_base.promote_int_input
def _aten_asinh(self: Any) -> Any:
    res = jnp.arcsinh(self)
    return res


# aten.atan
@op(torch.ops.aten.atan)  # pyre-ignore[56]
@op_base.promote_int_input
def _aten_atan(self: Any) -> Any:
    res = jnp.arctan(self)
    return res


@op(torch.ops.aten.scatter_reduce)
@op(torch.ops.aten.scatter)  # pyre-ignore[56]
def _aten_scatter_reduce(
    input: Any,
    dim: Any,
    index: Any,
    src: Any,
    reduce: Any = None,
    *,
    include_self: Any = True,
) -> Any:
    if not isinstance(src, jnp.ndarray):
        src = jnp.array(src, dtype=input.dtype)
    input_indexes, source_indexes = _scatter_index(dim, index)
    # "Zero out" target elements when not included
    if not include_self:
        if reduce in ["sum", "mean"]:
            base_input = jnp.zeros_like(src)
        elif reduce == "prod":
            base_input = jnp.ones_like(src)
        elif reduce == "amax":
            base_input = jnp.full_like(src, -jnp.inf)
        else:  # amin
            base_input = jnp.full_like(src, jnp.inf)
        input = input.at[input_indexes].set(base_input[source_indexes])

    if reduce == "sum" or reduce == "add":
        return input.at[input_indexes].add(src[source_indexes])
    elif reduce == "prod" or reduce == "multiply":
        return input.at[input_indexes].multiply(src[source_indexes])
    elif reduce == "mean":
        if include_self:
            count = jnp.ones_like(input)
        else:
            count = jnp.zeros_like(input)
        count = count.at[input_indexes].add(jnp.ones_like(src)[source_indexes])
        count = jnp.clip(count, min=1)
        mean = input.at[input_indexes].add(src[source_indexes])
        if _is_int(input):
            return mean // count
        return mean / count
    elif reduce == "amax":
        return input.at[input_indexes].max(src[source_indexes])
    elif reduce == "amin":
        return input.at[input_indexes].min(src[source_indexes])
    else:
        return input.at[input_indexes].set(src[source_indexes])


# aten.acos
@op(torch.ops.aten.acos)  # pyre-ignore[56]
@op_base.promote_int_input
def _aten_acos(self: Any) -> Any:
    return jnp.arccos(self)


# aten.sym_storage_offset
# aten.native_layer_norm_backward
# aten.max_pool3d_with_indices


# aten.gt
@op(torch.ops.aten.gt)  # pyre-ignore[56]
def _aten_gt(self: Any, other: Any) -> Any:
    return self > other


# aten.sym_stride
# aten.lt
@op(torch.ops.aten.lt)  # pyre-ignore[56]
def _aten_lt(self: Any, other: Any) -> Any:
    return self < other


def pool(
    inputs: Any,
    init: Any,
    reduce_fn: Any,
    window_shape: Any,
    strides: Any,
    padding: Any,
) -> Any:
    """Helper function to define pooling functions.

    Pooling functions are implemented using the ReduceWindow XLA op.
    NOTE: Be aware that pooling is not generally differentiable.
    That means providing a reduce_fn that is differentiable does not imply that
    pool is differentiable.

    Args:
      inputs: input data with dimensions (batch, window dims..., features).
      init: the initial value for the reduction
      reduce_fn: a reduce function of the form ``(T, T) -> T``.
      window_shape: a shape tuple defining the window to reduce over.
      strides: a sequence of ``n`` integers, representing the inter-window
        strides (default: ``(1, ..., 1)``).
      padding: either the string ``'SAME'``, the string ``'VALID'``, or a sequence
        of ``n`` ``(low, high)`` integer pairs that give the padding to apply before
        and after each spatial dimension.
    Returns:
      The output of the reduction for each window slice.
    """
    num_batch_dims = inputs.ndim - (len(window_shape) + 1)
    strides = strides or (1,) * len(window_shape)
    assert len(window_shape) == len(strides), (
        f"len({window_shape}) must equal len({strides})"
    )
    strides = (1,) * (1 + num_batch_dims) + strides
    dims = (1,) * (1 + num_batch_dims) + window_shape

    is_single_input = False
    if num_batch_dims == 0:
        # add singleton batch dimension because lax.reduce_window always
        # needs a batch dimension.
        inputs = inputs[None]
        strides = (1,) + strides
        dims = (1,) + dims
        is_single_input = True

    assert inputs.ndim == len(dims), f"len({inputs.shape}) != len({dims})"
    if not isinstance(padding, str):
        padding = tuple(map(tuple, padding))  # pyre-ignore[6]
        assert len(padding) == len(window_shape), (
            f"padding {padding} must specify pads for same number of dims as "
            f"window_shape {window_shape}"
        )
        assert all(len(x) == 2 for x in padding), (
            f"each entry in padding {padding} must be length 2"
        )
        padding = ((0, 0), (0, 0)) + padding
    y = jax.lax.reduce_window(inputs, init, reduce_fn, dims, strides, padding)
    if is_single_input:
        y = jnp.squeeze(y, axis=0)
    return y


@op(torch.ops.aten._adaptive_avg_pool2d)
@op(torch.ops.aten._adaptive_avg_pool3d)  # pyre-ignore[56]
def adaptive_avg_pool2or3d(
    input: jnp.ndarray, output_size: tuple[int, int]
) -> jnp.ndarray:
    """
    Applies a 2/3D adaptive average pooling over an input signal composed of several input planes.

    See :class:`~torch.nn.AdaptiveAvgPool2d` for details and output shape.

    Args:
        input: input tensor
        output_size: the target output size (single integer or double-integer tuple)

    Context:
      https://github.com/pytorch/pytorch/blob/main/torch/_decomp/decompositions.py#L2401
    """
    shape = input.shape
    ndim = len(shape)
    out_dim = len(output_size)
    num_spatial_dim = ndim - out_dim

    # Preconditions

    assert ndim in (out_dim + 1, out_dim + 2), (
        f"adaptive_avg_pool{num_spatial_dim}d(): Expected {num_spatial_dim + 1}D or {num_spatial_dim + 2}D tensor, but got {ndim}"
    )
    for d in input.shape[-2:]:
        assert d != 0, (
            "adaptive_avg_pool{num_spactial_dim}d(): Expected input to have non-zero size for "
            f"non-batch dimensions, but input has shape {tuple(shape)}."
        )

    # Optimisation (we should also do this in the kernel implementation)
    if all(s % o == 0 for o, s in zip(output_size, shape[-out_dim:], strict=False)):
        stride = tuple(
            i // o for i, o in zip(shape[-out_dim:], output_size, strict=False)
        )
        kernel = tuple(
            i - (o - 1) * s
            for i, o, s in zip(shape[-out_dim:], output_size, stride, strict=False)
        )
        return _aten_avg_pool(
            input,
            kernel,
            strides=stride,
        )

    def start_index(a: Any, b: Any, c: Any) -> Any:
        return (a * c) // b

    def end_index(a: Any, b: Any, c: Any) -> Any:
        return ((a + 1) * c + b - 1) // b

    def compute_idx(in_size: Any, out_size: Any) -> Any:
        orange = jnp.arange(out_size, dtype=jnp.int64)
        i0 = start_index(orange, out_size, in_size)
        # Let length = end_index - start_index, i.e. the length of the pooling kernels
        # length.max() can be computed analytically as follows:
        maxlength = in_size // out_size + 1
        in_size_mod = in_size % out_size
        # adaptive = True iff there are kernels with different lengths
        adaptive = not (in_size_mod == 0 or out_size % in_size_mod == 0)
        if adaptive:
            maxlength += 1
        elif in_size_mod == 0:
            maxlength -= 1

        range_max = jnp.arange(maxlength, dtype=jnp.int64)
        idx = i0[:, None] + range_max
        if adaptive:
            # Need to clamp to avoid accessing out-of-bounds memory
            idx = jnp.minimum(idx, in_size - 1)

            # Compute the length
            i1 = end_index(orange, out_size, in_size)
            length = i1 - i0
        else:
            length = maxlength
        return idx, length, range_max, adaptive

    idx, length, range_max, adaptive = [[None] * out_dim for _ in range(4)]
    # length is not None if it's constant, otherwise we'll need to compute it
    for i, (s, o) in enumerate(zip(shape[-out_dim:], output_size, strict=False)):
        idx[i], length[i], range_max[i], adaptive[i] = compute_idx(s, o)

    def _unsqueeze_to_dim(x: Any, dim: Any) -> Any:
        ndim = len(x.shape)
        return jax.lax.expand_dims(x, tuple(range(ndim, dim)))

    if out_dim == 2:
        # NOTE: unsqueeze to insert extra 1 in ranks; so they
        # would broadcast
        vals = input[..., _unsqueeze_to_dim(idx[0], 4), idx[1]]
        reduce_axis = (-3, -1)
    else:
        assert out_dim == 3
        vals = input[
            ...,
            _unsqueeze_to_dim(idx[0], 6),
            _unsqueeze_to_dim(idx[1], 4),
            idx[2],
        ]
        reduce_axis = (-5, -3, -1)

    # Shortcut for the simpler case
    if not any(adaptive):
        return jnp.mean(vals, axis=reduce_axis)

    def maybe_mask(
        vals: Any, length: Any, range_max: Any, adaptive: Any, dim: Any
    ) -> Any:
        if isinstance(length, int):
            return vals, length
        else:
            # zero-out the things we didn't really want to select
            assert dim < 0
            # hack
            mask = range_max >= length[:, None]
            if dim == -2:
                mask = _unsqueeze_to_dim(mask, 4)
            elif dim == -3:
                mask = _unsqueeze_to_dim(mask, 6)
            vals = jnp.where(mask, 0.0, vals)
            # Compute the length of each window
            length = _unsqueeze_to_dim(length, -dim)
            return vals, length

    for i in range(len(length)):
        vals, length[i] = maybe_mask(
            vals,
            length[i],
            range_max[i],
            adaptive=adaptive[i],
            dim=(i - out_dim),
        )

    # We unroll the sum as we assume that the kernels are going to be small
    ret = jnp.sum(vals, axis=reduce_axis)
    # NOTE: math.prod because we want to expand it to length[0] * length[1] * ...
    # this is multiplication with broadcasting, not regular pointwise product
    return ret / math.prod(length)  # pyre-fixme[6]


@op(torch.ops.aten.avg_pool1d)
@op(torch.ops.aten.avg_pool2d)
@op(torch.ops.aten.avg_pool3d)  # pyre-ignore[56]
def _aten_avg_pool(
    inputs: Any,
    kernel_size: Any,
    strides: Any = None,
    padding: Any = 0,
    ceil_mode: Any = False,
    count_include_pad: Any = True,
    divisor_override: Any = None,
) -> Any:
    num_batch_dims = len(inputs.shape) - len(kernel_size) - 1
    kernel_size = tuple(kernel_size)
    strides = tuple(strides) if strides else kernel_size
    if isinstance(padding, list) and len(padding) == 1:
        padding = padding[0]
    if isinstance(padding, int):
        padding = [padding for _ in range(len(kernel_size))]

    input_shape = inputs.shape
    if num_batch_dims == 0:
        input_shape = [1, *input_shape]
    padding = _ceil_mode_padding(
        padding,
        input_shape,
        # pyrefly: ignore [bad-argument-type]
        kernel_size,
        # pyrefly: ignore [bad-argument-type]
        strides,
        [1] * len(kernel_size),
        ceil_mode,
    )

    y = pool(inputs, 0.0, jax.lax.add, kernel_size, strides, padding)
    if divisor_override is not None:
        y = y / jnp.array(divisor_override, y.dtype)
    elif count_include_pad:
        div_shape = list(y.shape)
        div_by = jnp.ones(div_shape, y.dtype) * np.prod(kernel_size)
        unequal_paddings = (pad[0] != pad[1] for pad in padding)
        unequal_padding_indices = np.where(list(unequal_paddings))[0]
        if len(unequal_padding_indices) > 0:
            # indices to update kernel size
            offset = len(div_shape) - len(padding)
            skip_indices = [x + offset for x in unequal_padding_indices]
            indices = _generate_indices(div_shape, skip_dim_indices=skip_indices)
            # updated kernel size accounting for maximum padding
            new_kernel_size = list(kernel_size)
            for j in unequal_padding_indices:
                new_kernel_size[j] = kernel_size[j] - padding[j][1] + padding[j][0]

            for idx in indices:
                for j in unequal_padding_indices:
                    idx[j + offset] = -1
                div_by = div_by.at[tuple(idx)].set(np.prod(new_kernel_size))

        y = y / div_by
    else:
        div_shape = list(inputs.shape)
        div_shape[num_batch_dims] = 1
        div_shape = tuple(div_shape)
        if len(div_shape) - 2 == len(kernel_size):
            div_shape = (1,) + div_shape[1:]
        y = y / pool(
            jnp.ones(div_shape, y.dtype),
            jnp.array(0.0, y.dtype),
            jax.lax.add,
            kernel_size,
            strides,
            padding,
        )
    return y.astype(inputs.dtype)


# helper function to generate all indices to iterate through ndarray
def _generate_indices(dims: Any, skip_dim_indices: Any = None) -> Any:
    if skip_dim_indices is None:
        skip_dim_indices = []
    res = []

    def _helper(curr_dim_idx: Any, sofar: Any) -> Any:  # pyre-ignore[53]
        if curr_dim_idx in skip_dim_indices:
            _helper(curr_dim_idx + 1, sofar[:])
            return
        if curr_dim_idx >= len(dims):
            res.append(sofar)
            return
        for i in range(dims[curr_dim_idx]):
            sofar[curr_dim_idx] = i
            _helper(curr_dim_idx + 1, sofar[:])

    _helper(0, [0 for _ in dims])
    return res


# aten.sym_numel
# aten.reciprocal
@op(torch.ops.aten.reciprocal)  # pyre-ignore[56]
def _aten_reciprocal(a: Any) -> Any:
    if _is_int(a):
        return (1 / a).astype(jnp.dtype("float32"))  # pyre-ignore[16]
    return 1 / a


# aten.select_scatter
@op(torch.ops.aten.select_scatter)  # pyre-ignore[56]
def _aten_select_scatter(input: Any, src: Any, dim: Any, index: Any) -> Any:
    input_indexes = []
    if dim < 0:
        dim += len(input.shape)
    for x in range(len(input.shape)):
        if x == dim:
            input_indexes.append(index)
        else:
            input_indexes.append(slice(None, None, None))
    return input.at[tuple(input_indexes)].set(src)


@op(torch.ops.aten.scatter.src)  # pyre-ignore[56]
def _aten_scatter_src(
    input: Any, dim: Any, index: Any, src: Any, reduce: Any = None
) -> Any:
    input_index, source_indexes = _scatter_index(dim, index)
    return input.at[input_index].set(src[source_indexes])


@op(torch.ops.aten.scatter.value)  # pyre-ignore[56]
def _aten_scatter(
    input: Any, dim: Any, index: Any, src: Any, reduce: Any = None
) -> Any:
    input_index, source_indexes = _scatter_index(dim, index)
    return input.at[input_index].set(src)


# aten.acosh
@op(torch.ops.aten.acosh)  # pyre-ignore[56]
@op_base.promote_int_input
def _aten_acosh(self: Any) -> Any:
    return jnp.arccosh(self)


# aten.avg_pool2d_backward
# aten.col2im
# aten.avg_pool3d
# aten.round
@op(torch.ops.aten.round)  # pyre-ignore[56]
def _aten_round(input: Any, decimals: Any = 0) -> Any:
    return jnp.round(input, decimals)


# aten.max
@op(torch.ops.aten.max)  # pyre-ignore[56]
def _aten_max(self: Any, dim: Any = None, keepdim: Any = False) -> Any:
    if dim is not None:
        return _with_reduction_scalar(
            jnp.max, self, dim, keepdim
        ), _with_reduction_scalar(jnp.argmax, self, dim, keepdim).astype(jnp.int64)
    else:
        return _with_reduction_scalar(jnp.max, self, dim, keepdim)


# aten.maximum
@op(torch.ops.aten.maximum)  # pyre-ignore[56]
def _aten_maximum(self: Any, other: Any) -> Any:
    return jnp.maximum(self, other)


# aten.abs
@op(torch.ops.aten.abs)  # pyre-ignore[56]
def _aten_abs(self: Any) -> Any:
    return jnp.abs(self)


# generate aten.amax only
@op(torch.ops.aten.amax)  # pyre-ignore[56]
def _aten_amax(self: Any, dim: Any = None, keepdim: Any = False) -> Any:
    return _with_reduction_scalar(jnp.amax, self, dim, keepdim)


def _with_reduction_scalar(jax_func: Any, self: Any, dim: Any, keepdim: Any) -> Any:
    expanded = False
    if self.ndim == 0:
        # for self of rank 0:
        # torch.any(x, 0), torch.any(x, -1) works;
        # torch.any(x, 1) throws out of bounds, so it's
        # behavior is the same as a jnp array of rank 1
        expanded = True
        self = jnp.expand_dims(self, 0)
    res = jax_func(self, axis=dim, keepdims=keepdim)
    if expanded:
        res = res.squeeze()
    return res


# aten.any
@op(torch.ops.aten.any)  # pyre-ignore[56]
def _aten_any(self: Any, dim: Any = None, keepdim: Any = False) -> Any:
    return _with_reduction_scalar(jnp.any, self, dim, keepdim)


# aten.arange
@op(torch.ops.aten.arange.start_step)
@op(torch.ops.aten.arange.start)
@op(torch.ops.aten.arange.default)  # pyre-ignore[56]
@op_base.convert_dtype(use_default_dtype=False)
def _aten_arange(
    start: Any,
    end: Any = None,
    step: Any = None,
    *,
    dtype: Any = None,
    layout: Any = None,
    requires_grad: Any = False,
    device: Any = None,
    pin_memory: Any = False,
) -> Any:
    return jnp.arange(
        op_base.maybe_convert_constant_dtype(start, dtype),
        op_base.maybe_convert_constant_dtype(end, dtype),
        op_base.maybe_convert_constant_dtype(step, dtype),
        dtype=dtype,
    )


# aten.argmax
@op(torch.ops.aten.argmax)  # pyre-ignore[56]
def _aten_argmax(self: Any, dim: Any = None, keepdim: Any = False) -> Any:
    return _with_reduction_scalar(jnp.argmax, self, dim, keepdim)


def _strided_index(sizes: Any, strides: Any, storage_offset: Any = None) -> Any:
    ind = jnp.zeros(sizes, dtype=jnp.int32)

    for i, (size, stride) in enumerate(zip(sizes, strides, strict=False)):
        result_shape = (1,) * i + (size,) + (1,) * (len(sizes) - i - 1)
        indexes = (jnp.arange(size) * stride).reshape(result_shape)
        ind += indexes

    if storage_offset is not None:
        ind += storage_offset
    return ind


# aten.as_strided
@op(torch.ops.aten.as_strided)
@op(torch.ops.aten.as_strided_copy)  # pyre-ignore[56]
def _aten_as_strided(
    x: Any, sizes: Any, strides: Any, storage_offset: Any = None
) -> Any:
    ind = _strided_index(sizes, strides, storage_offset)
    flattened = jnp.ravel(x)
    return flattened[ind]


@op(torch.ops.aten.as_strided_scatter)  # pyre-ignore[56]
def _aten_as_strided_scatter(
    x: Any, src: Any, sizes: Any, strides: Any, storage_offset: Any
) -> Any:
    ind = _strided_index(sizes, strides, storage_offset)
    flattened = jnp.ravel(x)
    modified = flattened.at[ind].set(src)
    return modified.reshape(x.shape)


# aten.atan2
@op(torch.ops.aten.atan2)  # pyre-ignore[56]
@op_base.promote_int_input
def _aten_atan2(input: Any, other: Any) -> Any:
    return jnp.arctan2(input, other)


# aten.bitwise_and
@op(torch.ops.aten.bitwise_and)  # pyre-ignore[56]
def _aten_bitwise_and(self: Any, other: Any) -> Any:
    return self & other


# aten.bitwise_or
@op(torch.ops.aten.bitwise_or)  # pyre-ignore[56]
def _aten_bitwise_or(self: Any, other: Any) -> Any:
    return self | other


# aten.bitwise_xor
@op(torch.ops.aten.bitwise_xor)  # pyre-ignore[56]
def _aten_bitwise_xor(self: Any, other: Any) -> Any:
    return self ^ other


# aten.broadcast_tensors
@op(torch.ops.aten.broadcast_tensors)  # pyre-ignore[56]
def _aten_broadcast_tensors(*tensors: Any) -> Any:
    def _get_broadcast_shape(shapes: Any) -> Any:
        """
        Determines the output shape by broadcasting all input shapes.

        Args:
          shapes: A list of tuples representing the shapes of the input tensors.

        Returns:
          A tuple representing the broadcasted output shape.
        """

        # Find the maximum number of dimensions among all input tensors
        max_dims = max(len(shape) for shape in shapes)
        # Pad shorter shapes with 1s on the left to match the maximum number of dimensions
        padded_shapes = [(1,) * (max_dims - len(shape)) + shape for shape in shapes]

        # Initialize the output shape with 1s
        output_shape = [1] * max_dims
        # Iterate through each dimension and apply broadcasting rules
        for dim in range(max_dims):
            dim_sizes = [shape[dim] for shape in padded_shapes]
            max_size = max(dim_sizes)
            if all(size == 1 or size == max_size for size in dim_sizes):
                output_shape[dim] = max_size
            else:
                raise ValueError("Incompatible shapes for broadcasting")
        return tuple(output_shape)

    def _broadcast_dimensions(input_shape: Any, output_shape: Any) -> Any:
        """
        Determines the broadcast_dimensions argument for jax.lax.broadcast_in_dim.

        Args:
          input_shape: The shape of the input tensor.
          output_shape: The desired output shape after broadcasting.

        Returns:
          A tuple specifying which dimensions of the input tensor should be broadcasted.
        """

        res = tuple(
            i
            for i, (in_dim, out_dim) in enumerate(
                zip(input_shape, output_shape, strict=False)
            )
        )
        return res

    # clean some function's previous wrap
    if (
        len(tensors) == 1
        and len(tensors[0]) >= 1
        and isinstance(tensors[0][0], jax.Array)
    ):
        tensors = tensors[0]

    # Get the shapes of all input tensors
    shapes = [t.shape for t in tensors]
    # Find the output shape by broadcasting all input shapes
    output_shape = _get_broadcast_shape(shapes)
    # Broadcast each tensor to the output shape
    broadcasted_tensors = [
        jax.lax.broadcast_in_dim(
            t, output_shape, _broadcast_dimensions(t.shape, output_shape)
        )
        for t in tensors
    ]

    return broadcasted_tensors


# aten.broadcast_to
@op(torch.ops.aten.broadcast_to)  # pyre-ignore[56]
def _aten_broadcast_to(input: Any, shape: Any) -> Any:
    return jnp.broadcast_to(input, shape)


# aten.clamp
@op(torch.ops.aten.clamp.default)
@op(torch.ops.aten.clamp.Tensor)  # pyre-ignore[56]
def _aten_clamp(self: Any, min: Any = None, max: Any = None) -> Any:
    return jnp.clip(self, min, max)


@op(torch.ops.aten.clamp_min)  # pyre-ignore[56]
def _aten_clamp_min(input: Any, min: Any) -> Any:
    return jnp.clip(input, min=min)


# aten.constant_pad_nd
@op(torch.ops.aten.constant_pad_nd)  # pyre-ignore[56]
def _aten_constant_pad_nd(input: Any, padding: Any, value: Any = 0) -> Any:
    # NOTE: Torch padding is flat and reversed: (1, 1, 2, 2)
    #  means last dim get padded 1 in front and 1 in back;
    #  and second last dim get padded 2 in front and 2 in back.
    # Jax padding tuple of 3-tuple: the same padding is
    # [(0, 0, 0), ..., (2,2,0), (1,1,0)], where the last dimension
    # is the amount of padding added between any two elements in each dimension
    m = len(padding)
    rev_padding = [(padding[i - 1], padding[i], 0) for i in range(m - 1, 0, -2)]
    pad_dim = tuple(([(0, 0, 0)] * (len(input.shape) - m // 2)) + rev_padding)
    value_casted = jax.numpy.array(value, dtype=input.dtype)
    return jax.lax.pad(input, padding_value=value_casted, padding_config=pad_dim)


@op(torch.ops.aten.convolution_backward.default)  # pyre-ignore[56]
def _aten_convolution_backward(
    grad_output: Any,
    input: Any,
    weight: Any,
    bias_sizes: Any,
    stride: Any,
    padding: Any,
    dilation: Any,
    transposed: Any,
    output_padding: Any,
    groups: Any,
    output_mask: Any,
) -> Any:
    def fwd(inp: Any, w: Any) -> Any:
        return _aten_convolution(
            inp,
            w,
            None,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
        )

    _, vjp_fn = jax.vjp(fwd, input, weight)
    grad_input, grad_weight = vjp_fn(grad_output)

    grad_bias = None
    if output_mask[2] and bias_sizes is not None and len(bias_sizes) > 0:
        sum_dims = (0,) + tuple(range(2, grad_output.ndim))
        grad_bias = jnp.sum(grad_output, axis=sum_dims)

    return (
        grad_input if output_mask[0] else None,
        grad_weight if output_mask[1] else None,
        grad_bias,
    )


@op(torch.ops.aten.lift_fresh_copy)  # pyre-ignore[56]
def _aten_lift_fresh_copy(x: Any) -> Any:
    return jnp.copy(x)


@op(torch.ops.aten.copy)  # pyre-ignore[56]
def _aten_copy_no_inplace(self: Any, src: Any) -> Any:
    return jnp.broadcast_to(src, self.shape).astype(self.dtype)


@op(torch.ops.aten._cdist_forward)  # pyre-ignore[56]
def _aten_cdist_forward(x1: Any, x2: Any, p: Any, compute_mode: Any = "") -> Any:
    # x1 is B x P x M
    # x2 is B x Q x M
    # res is B x P x Q
    x1 = jnp.expand_dims(x1, len(x1.shape) - 1)
    x2 = jnp.expand_dims(x2, len(x2.shape) - 2)
    return jnp.linalg.norm(x1 - x2, ord=p, axis=-1)


@op(torch.ops.aten._pdist_forward)  # pyre-ignore[56]
def _aten__pdist_forward(x: Any, p: Any = 2) -> Any:
    pairwise_dists = _aten_cdist_forward(x, x, p)
    condensed_dists = pairwise_dists[jnp.triu_indices(pairwise_dists.shape[0], k=1)]
    return condensed_dists


@op(torch.ops.aten.cholesky_inverse)  # pyre-ignore[56]
def _aten_cholesky_inverse(input: Any, upper: Any = False) -> Any:
    t = jnp.matrix_transpose(input)
    if "complex" in str(input.dtype):
        t = t.conjugate()
    return jnp.linalg.inv(input @ t)


# aten.cos
@op(torch.ops.aten.cos)  # pyre-ignore[56]
@op_base.promote_int_input
def _aten_cos(input: Any) -> Any:
    return jnp.cos(input)


# aten.cosh
@op(torch.ops.aten.cosh)  # pyre-ignore[56]
@op_base.promote_int_input
def _aten_cosh(input: Any) -> Any:
    return jnp.cosh(input)


@op(torch.ops.aten.diag)  # pyre-ignore[56]
def _aten_diag(input: Any, diagonal: Any = 0) -> Any:
    return jnp.diag(input, diagonal)


# aten.diagonal
@op(torch.ops.aten.diagonal)
@op(torch.ops.aten.diagonal_copy)  # pyre-ignore[56]
def _aten_diagonal(input: Any, offset: Any = 0, dim1: Any = 0, dim2: Any = 1) -> Any:
    return jnp.diagonal(input, offset, dim1, dim2)


def diag_indices_with_offset(
    input_shape: Any, offset: Any, dim1: Any = 0, dim2: Any = 1
) -> Any:
    input_len = len(input_shape)
    if dim1 == dim2 or not (0 <= dim1 < input_len and 0 <= dim2 < input_len):
        raise ValueError(
            "dim1 and dim2 must be different and in range [0, "
            + str(input_len - 1)
            + "]"
        )

    size1, size2 = input_shape[dim1], input_shape[dim2]
    if offset >= 0:
        indices1 = jnp.arange(min(size1, size2 - offset))
        indices2 = jnp.arange(offset, offset + len(indices1))
    else:
        indices2 = jnp.arange(min(size1 + offset, size2))
        indices1 = jnp.arange(-offset, -offset + len(indices2))
    return [indices1, indices2]


@op(torch.ops.aten.diagonal_scatter)  # pyre-ignore[56]
def _aten_diagonal_scatter(
    input: Any, src: Any, offset: Any = 0, dim1: Any = 0, dim2: Any = 1
) -> Any:
    indexes = diag_indices_with_offset(input.shape, offset, dim1, dim2)

    if input.ndim == 2:
        return input.at[tuple(indexes)].set(src)
    else:
        # src has the same shape as the output of
        # jnp.diagonal(input, offset, dim1, dim2).
        # Last dimension always contains the diagonal elements,
        # while the preceding dimensions represent the "slices"
        # from which these diagonals are extracted. Thus,
        # we alter input axes to match this assumption, write src
        # and then move the axes back to the original state.
        input = jnp.moveaxis(input, (dim1, dim2), (-2, -1))
        multi_indexes = [slice(None)] * (input.ndim - 2) + indexes
        input = input.at[tuple(multi_indexes)].set(src)
        return jnp.moveaxis(input, (-2, -1), (dim1, dim2))


# aten.diagflat
@op(torch.ops.aten.diagflat)  # pyre-ignore[56]
def _aten_diagflat(input: Any, offset: Any = 0) -> Any:
    return jnp.diagflat(jnp.array(input), offset)


@op(torch.ops.aten.movedim)  # pyre-ignore[56]
def _aten_movedim(input: Any, source: Any, destination: Any) -> Any:
    return jnp.moveaxis(input, source, destination)


# aten.eq
@op(torch.ops.aten.eq)  # pyre-ignore[56]
def _aten_eq(input1: Any, input2: Any) -> Any:
    return input1 == input2


# aten.equal
@op(torch.ops.aten.equal)  # pyre-ignore[56]
def _aten_equal(input: Any, other: Any) -> Any:
    res = jnp.array_equal(input, other)
    return bool(res)


# aten.erf
@op(torch.ops.aten.erf)  # pyre-ignore[56]
@op_base.promote_int_input
def _aten_erf(x: Any) -> Any:
    return jax.lax.erf(x)


@op(torch.ops.aten.erfinv)  # pyre-ignore[56]
@op_base.promote_int_input
def _aten_erfinv(input: Any) -> Any:
    return jax.lax.erf_inv(input)


# aten.exp
@op(torch.ops.aten.exp)  # pyre-ignore[56]
def _aten_exp(input: Any) -> Any:
    res = jnp.exp(input)
    new_dtype = mappings.t2j_dtype(torch.get_default_dtype())
    if input.dtype == jax.numpy.int64:
        res = res.astype(new_dtype)
    return res


# aten.expm1
@op(torch.ops.aten.expm1)  # pyre-ignore[56]
def _aten_expm1(input: Any) -> Any:
    res = jnp.expm1(input)
    new_dtype = mappings.t2j_dtype(torch.get_default_dtype())
    if input.dtype == jax.numpy.int64:
        res = res.astype(new_dtype)
    return res


# aten.exp2
@op(torch.ops.aten.exp2)  # pyre-ignore[56]
def _aten_exp2(input: Any) -> Any:
    res = jnp.exp2(input)
    new_dtype = mappings.t2j_dtype(torch.get_default_dtype())
    if input.dtype == jax.numpy.int64:
        res = res.astype(new_dtype)
    return res


# aten.fill
@op(torch.ops.aten.fill)
@op(torch.ops.aten.full_like)  # pyre-ignore[56]
def _aten_fill(
    x: Any,
    value: Any,
    dtype: Any = None,
    pin_memory: Any = None,
    memory_format: Any = None,
    device: Any = None,
) -> Any:
    if dtype is None:
        dtype = x.dtype
    else:
        dtype = mappings.t2j_dtype(dtype)
    return jnp.full(x.shape, value, dtype)


# aten.flip
@op(torch.ops.aten.flip)  # pyre-ignore[56]
def _aten_flip(input: Any, dims: Any) -> Any:
    if dims is not None:
        return jnp.flip(input, tuple(dims))
    else:
        return jnp.flip(input)


# aten.floor
@op(torch.ops.aten.floor)  # pyre-ignore[56]
def _aten_floor(input: Any) -> Any:
    return jnp.floor(input).astype(input.dtype)


# aten.fmax
@op(torch.ops.aten.fmax)  # pyre-ignore[56]
def _aten_fmax(input: Any, other: Any) -> Any:
    return jnp.fmax(input, other)


# aten.fmin
@op(torch.ops.aten.fmin)  # pyre-ignore[56]
def _aten_fmin(input: Any, other: Any) -> Any:
    return jnp.fmin(input, other)


# aten.fmod
@op(torch.ops.aten.fmod)  # pyre-ignore[56]
def _aten_fmod(input: Any, other: Any) -> Any:
    return input - other * _aten_div(input, other, "trunc")


# aten.frexp
@op(torch.ops.aten.frexp)  # pyre-ignore[56]
def _aten_frexp(input: Any) -> Any:
    return jnp.frexp(input)


# aten.gather
@op(torch.ops.aten.gather)  # pyre-ignore[56]
def _aten_gather(input: Any, dim: Any, index: Any) -> Any:
    if input.ndim == 0:
        return jnp.broadcast_to(input, index.shape)
    # short circuit for empty outputs
    if not all(index.shape):
        return jnp.zeros(index.shape, dtype=input.dtype)
    if dim < 0:
        dim += input.ndim
    input_indexes, source_indexes = _scatter_index(dim, index)
    return input[input_indexes]


# aten.ge
@op(torch.ops.aten.ge)  # pyre-ignore[56]
def _aten_ge(self: Any, other: Any) -> Any:
    return self >= other


@op(torch.ops.aten.glu)  # pyre-ignore[56]
def _aten_glu(x: Any, dim: Any = -1) -> Any:
    return jax.nn.glu(x, dim)


# aten.hardtanh
@op(torch.ops.aten.hardtanh)  # pyre-ignore[56]
def _aten_hardtanh(
    input: Any, min_val: Any = -1, max_val: Any = 1, inplace: Any = False
) -> Any:
    if (
        input.dtype == np.int64
        and isinstance(max_val, float)
        and isinstance(min_val, float)
    ):
        min_val = int(min_val)
        max_val = int(max_val)
    return jnp.clip(input, min_val, max_val)


# aten.histc
@op(torch.ops.aten.histc)  # pyre-ignore[56]
def _aten_histc(input: Any, bins: Any = 100, min: Any = 0, max: Any = 0) -> Any:
    # TODO(@manfei): this function might cause some uncertainty
    if min == 0 and max == 0:
        if isinstance(input, jnp.ndarray) and input.size == 0:
            min = 0
            max = 0
        else:
            min = jnp.min(input)
            max = jnp.max(input)
    range_value = (min, max)
    hist, bin_edges = jnp.histogram(
        input, bins=bins, range=range_value, weights=None, density=None
    )
    return hist


@op(torch.ops.aten.hypot)  # pyre-ignore[56]
def _aten_hypot(input: Any, other: Any) -> Any:
    return jnp.hypot(input, other)


@op(torch.ops.aten.digamma)  # pyre-ignore[56]
def _aten_digamma(input: Any, *, out: Any = None) -> Any:
    res = jax.scipy.special.digamma(input).astype(jnp.float32)
    # replace indices where input == 0 with -inf in res
    return jnp.where(jnp.equal(input, jnp.zeros(input.shape)), -jnp.inf, res)


@op(torch.ops.aten.igamma)  # pyre-ignore[56]
def _aten_igamma(input: Any, other: Any) -> Any:
    return jax.scipy.special.gammainc(input, other)


@op(torch.ops.aten.lgamma)  # pyre-ignore[56]
def _aten_lgamma(input: Any, *, out: Any = None) -> Any:
    return jax.scipy.special.gammaln(input).astype(jnp.float32)


@op(torch.ops.aten.mvlgamma)  # pyre-ignore[56]
def _aten_mvlgamma(input: Any, p: Any, *, out: Any = None) -> Any:
    input = input.astype(mappings.t2j_dtype(torch.get_default_dtype()))
    return jax.scipy.special.multigammaln(input, p)


@op(torch.ops.aten.linalg_eig)  # pyre-ignore[56]
def _aten_linalg_eig(A: Any) -> Any:
    return jnp.linalg.eig(A)


@op(torch.ops.aten._linalg_eigh)  # pyre-ignore[56]
def _aten_linalg_eigh(A: Any, UPLO: Any = "L") -> Any:
    return jnp.linalg.eigh(A, UPLO)


@op(torch.ops.aten.linalg_lstsq)  # pyre-ignore[56]
def _aten_linalg_lstsq(A: Any, B: Any, rcond: Any = None, driver: Any = "gelsy") -> Any:
    input_dtype = A.dtype

    m = A.shape[-2]
    n = A.shape[-1]

    is_batched = A.ndim > 2

    if is_batched:
        batch_shape = jnp.broadcast_shapes(A.shape[:-2], B.shape[:-2])
        batch_size = int(np.prod(batch_shape))
        A_reshaped = A.reshape((batch_size,) + A.shape[-2:])
        B_reshaped = B.reshape((batch_size,) + B.shape[-2:])

        X, residuals, rank, singular_values = jax.vmap(
            jnp.linalg.lstsq, in_axes=(0, 0)
        )(A_reshaped, B_reshaped, rcond=rcond)

        X = X.reshape(batch_shape + X.shape[-2:])

        if driver in ["gelsd", "gelsy", "gelss"]:
            rank = rank.reshape(batch_shape)
        else:
            rank = jnp.array([], dtype=jnp.int64)

        full_rank = jnp.all(rank == n)
        if driver == "gelsy" or m <= n or (not full_rank):
            residuals = jnp.array([], dtype=input_dtype)
        else:
            residuals = residuals.reshape(batch_shape + residuals.shape[-1:])

        if driver in ["gelsd", "gelss"]:
            singular_values = singular_values.reshape(
                batch_shape + singular_values.shape[-1:]
            )
        else:
            singular_values = jnp.array([], dtype=input_dtype)

    else:
        X, residuals, rank, singular_values = jnp.linalg.lstsq(A, B, rcond=rcond)

        if driver not in ["gelsd", "gelsy", "gelss"]:
            rank = jnp.array([], dtype=jnp.int64)

        rank_value = None
        if rank.size > 0:
            rank_value = int(rank.item())
            rank = jnp.array(rank_value, dtype=jnp.int64)

        # When driver is ‘gels’, assume that A is full-rank.
        full_rank = driver == "gels" or rank_value == n
        if driver == "gelsy" or m <= n or (not full_rank):
            residuals = jnp.array([], dtype=input_dtype)

        if driver not in ["gelsd", "gelss"]:
            singular_values = jnp.array([], dtype=input_dtype)

    return X, residuals, rank, singular_values


@op(torch.ops.aten.linalg_ldl_factor_ex)  # pyre-ignore[56]
def _aten_linalg_ldl_factor_ex(
    A: Any, hermitian: Any = False, check_errors: Any = False
) -> Any:
    # TODO: Replace with native LDL when available:
    # https://github.com/jax-ml/jax/issues/12779
    # TODO: Not tested for complex inputs. Does not support hermitian=True
    pivots = jnp.broadcast_to(
        jnp.arange(1, A.shape[-1] + 1, dtype=jnp.int32), A.shape[:-1]
    )
    info = jnp.zeros(A.shape[:-2], jnp.int32)
    C = jnp.linalg.cholesky(A)
    if C.size == 0:
        return C, pivots, info

    # Fill diagonals of stacked matrices
    @functools.partial(jnp.vectorize, signature="(k,k),(k,k)->(k,k)")
    def fill_diagonal_batch(x: Any, y: Any) -> Any:
        return jnp.fill_diagonal(x, jnp.diag(y), inplace=False)

    D = C * jnp.eye(C.shape[-1], dtype=A.dtype)
    LD = C @ jnp.linalg.inv(D)
    LD = fill_diagonal_batch(LD, D * D)
    return LD, pivots, info


@op(torch.ops.aten.linalg_lu)  # pyre-ignore[56]
def _aten_linalg_lu(A: Any, pivot: Any = True, out: Any = None) -> Any:
    dtype = A.dtype

    *_, m, n = A.shape
    k = jnp.minimum(m, n)

    lu, _, permutation = jax.lax.linalg.lu(A)

    L = jnp.tril(lu[..., :, :k], k=-1)
    eye_L = jnp.eye(m, k, dtype=dtype)
    L = L + eye_L

    U = jnp.triu(lu[..., :k, :])

    def perm_to_P(perm: Any) -> Any:  # pyre-ignore[53]
        m = perm.shape[-1]
        P = jnp.eye(m, dtype=dtype)[perm].T
        return P

    if permutation.ndim > 1:
        num_batch_dims = permutation.ndim - 1
        for _ in range(num_batch_dims):
            perm_to_P = jax.vmap(perm_to_P, in_axes=0)

    P = perm_to_P(permutation)

    return P, L, U


@op(torch.ops.aten.linalg_lu_factor_ex)  # pyre-ignore[56]
def _aten_linalg_lu_factor_ex(
    A: Any, pivot: Any = True, check_errors: Any = False
) -> Any:
    lu, pivots, _ = jax.lax.linalg.lu(A)
    # PT pivots vector is 1-indexed
    pivots = pivots + 1
    info = jnp.zeros(A.shape[:-2], jnp.int32)
    return lu, pivots, info


@op(torch.ops.aten.linalg_lu_solve)  # pyre-ignore[56]
def _aten_linalg_lu_solve(
    LU: Any, pivots: Any, B: Any, left: Any = True, adjoint: Any = False
) -> Any:
    # JAX pivots are offset by 1 compared to torch
    pivots = pivots - 1
    if not left:
        # XA = B is same as A'X = B'
        trans = 0 if adjoint else 2
        x = jax.scipy.linalg.lu_solve((LU, pivots), jnp.matrix_transpose(B), trans)
        x = jnp.matrix_transpose(x)
    else:
        trans = 2 if adjoint else 0
        x = jax.scipy.linalg.lu_solve((LU, pivots), B, trans)
    return x


@op(torch.ops.aten.gcd)  # pyre-ignore[56]
def _aten_gcd(input: Any, other: Any) -> Any:
    return jnp.gcd(input, other)


# aten.lcm
@op(torch.ops.aten.lcm)  # pyre-ignore[56]
def _aten_lcm(input: Any, other: Any) -> Any:
    return jnp.lcm(input, other)


# aten.isinf
@op(torch.ops.aten.isinf)  # pyre-ignore[56]
def _aten_isinf(input: Any) -> Any:
    return jnp.isinf(input)


# aten.isnan
@op(torch.ops.aten.isnan)  # pyre-ignore[56]
def _aten_isnan(input: Any) -> Any:
    return jnp.isnan(input)


@op(torch.ops.aten.le)  # pyre-ignore[56]
def _aten_le(self: Any, other: Any) -> Any:
    return self <= other


# aten.leaky_relu
@op(torch.ops.aten.leaky_relu)  # pyre-ignore[56]
def _aten_leaky_relu(x: Any, negative_slope: Any = 0.01) -> Any:
    return jax.nn.leaky_relu(x, negative_slope)


# aten.log
@op(torch.ops.aten.log)  # pyre-ignore[56]
@op_base.promote_int_input
def _aten_log(x: Any) -> Any:
    return jnp.log(x)


# aten.log10
@op(torch.ops.aten.log10)  # pyre-ignore[56]
@op_base.promote_int_input
def _aten_log10(x: Any) -> Any:
    return jnp.log10(x)


# aten.log1p
@op(torch.ops.aten.log1p)  # pyre-ignore[56]
@op_base.promote_int_input
def _aten_log1p(x: Any) -> Any:
    return jnp.log1p(x)


# aten.log2
@op(torch.ops.aten.log2)  # pyre-ignore[56]
@op_base.promote_int_input
def _aten_log2(x: Any) -> Any:
    return jnp.log2(x)


# aten.logical_and (incl. in-place variant emitted by isposinf/isneginf
# decompositions under proxy_tensor: `(x == ±inf) & ~isnan(x)` lowers to
# `aten::logical_and_`).
@op(torch.ops.aten.logical_and)
@op(torch.ops.aten.logical_and_)  # pyre-ignore[56]
@op(torch.ops.aten.__and__)  # pyre-ignore[56]
def _aten_logical_and(self: Any, other: Any) -> Any:
    return jnp.logical_and(self, other)


# aten.logical_or (incl. in-place variant for symmetry with logical_and_).
@op(torch.ops.aten.logical_or)
@op(torch.ops.aten.logical_or_)  # pyre-ignore[56]
@op(torch.ops.aten.__or__)  # pyre-ignore[56]
def _aten_logical_or(self: Any, other: Any) -> Any:
    return jnp.logical_or(self, other)


# aten.logical_not (incl. in-place variant for symmetry).
@op(torch.ops.aten.logical_not)  # pyre-ignore[56]
@op(torch.ops.aten.logical_not_)  # pyre-ignore[56]
def _aten_logical_not(self: Any) -> Any:
    return jnp.logical_not(self)


# aten.log_softmax
@op(torch.ops.aten._log_softmax)  # pyre-ignore[56]
def _aten_log_softmax(self: Any, axis: Any = -1, half_to_float: Any = False) -> Any:
    if half_to_float and self.dtype in (jnp.float16, jnp.bfloat16):
        self = self.astype(jnp.float32)
    if self.shape == ():
        return jnp.astype(0.0, self.dtype)
    return jax.nn.log_softmax(self, axis)


# aten.logaddexp
@op(torch.ops.aten.logaddexp)  # pyre-ignore[56]
def _aten_logaddexp(self: Any, other: Any) -> Any:
    return jnp.logaddexp(self, other)


# aten.logaddexp2
@op(torch.ops.aten.logaddexp2)  # pyre-ignore[56]
def _aten_logaddexp2(self: Any, other: Any) -> Any:
    return jnp.logaddexp2(self, other)


# aten.logcumsumexp
@op(torch.ops.aten.logcumsumexp)  # pyre-ignore[56]
def _aten_logcumsumexp(self: Any, dim: Any = None) -> Any:
    if self.shape == ():
        return self
    return jax.lax.cumlogsumexp(self, axis=dim)


# aten.max_pool3d_backward
# aten.logical_xor
@op(torch.ops.aten.logical_xor)
@op(torch.ops.aten.__xor__)  # pyre-ignore[56]
def _aten_logical_xor(self: Any, other: Any) -> Any:
    return jnp.logical_xor(self, other)


# aten.max_pool2d_with_indices_backward
# aten.native_dropout
# aten.native_group_norm_backward
# aten.neg
@op(torch.ops.aten.neg)  # pyre-ignore[56]
def _aten_neg(x: Any) -> Any:
    return -1 * x


@op(torch.ops.aten.nextafter)  # pyre-ignore[56]
def _aten_nextafter(input: Any, other: Any, *, out: Any = None) -> Any:
    return jnp.nextafter(input, other)


@op(torch.ops.aten.nonzero_static)  # pyre-ignore[56]
def _aten_nonzero_static(input: Any, size: Any, fill_value: Any = -1) -> Any:
    indices = jnp.argwhere(input)

    if size < indices.shape[0]:
        indices = indices[:size]
    elif size > indices.shape[0]:
        padding = jnp.full(
            (size - indices.shape[0], indices.shape[1]),
            fill_value,
            dtype=indices.dtype,
        )
        indices = jnp.concatenate((indices, padding))

    return indices


# aten.nonzero
@op(torch.ops.aten.nonzero)  # pyre-ignore[56]
def _aten_nonzero(x: Any, as_tuple: Any = False) -> Any:
    if jnp.ndim(x) == 0 and (as_tuple or x.item() == 0):
        return torch.empty(0, 0, dtype=torch.int64)
    if (
        jnp.ndim(x) == 0
    ):  # when x is scalar, return torch.tensor([], size=(1, 0), dtype=torch.int64)
        res = torch.empty(1, 0, dtype=torch.int64)
        return jnp.array(res.numpy())
    index_tuple = jnp.nonzero(x)
    index_tuple = [jnp.expand_dims(p, -1) for p in index_tuple]
    return jnp.concatenate(index_tuple, axis=-1)


# aten.prod
@op(torch.ops.aten.prod)  # pyre-ignore[56]
def _aten_prod(
    input: Any, dim: Any = None, keepdim: Any = False, *, dtype: Any = None
) -> Any:
    if dtype:
        input = input.astype(mappings.t2j_dtype(dtype))
    return _with_reduction_scalar(jnp.prod, input, dim, keepdim)


@op(torch.ops.aten.put)  # pyre-ignore[56]
def _aten_put(self: Any, index: Any, source: Any, accumulate: Any = False) -> Any:
    expanded = False
    res = None

    if self.ndim == 0:
        expanded = True
        self = jnp.expand_dims(self, 0)

    if accumulate:
        tmp = jnp.zeros(self.shape)
        tmp = jnp.put(tmp, index, source, inplace=False)
        res = jnp.add(self, tmp).astype(self.dtype)
    else:
        res = jnp.put(self, index, source, inplace=False)

    if expanded:
        res = res.squeeze()

    return res


# aten.randperm
# randperm.generator(SymInt n, *, Generator? generator, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None)
@op(torch.ops.aten.randperm, needs_env=True)  # pyre-ignore[56]
def _aten_randperm(
    n: Any,
    *,
    generator: Any = None,
    dtype: Any = None,
    layout: Any = None,
    device: Any = None,
    pin_memory: Any = None,
    env: Any = None,
) -> Any:
    """
    Generates a random permutation of integers from 0 to n-1.

    Args:
        n: The upper bound (exclusive) of the permutation range.
        generator: A PRNGKey used as the random key. If None, a new key is created.
        dtype: The desired data type of the output array. Default is jnp.int64.
        layout: The desired layout of the output array (e.g., 'row-major', 'column-major').
        device: The desired device on which to place the output array (e.g., jax.devices()[0]).
        pin_memory: Whether to pin the output array's memory to the host.

    Returns:
        A DeviceArray containing a random permutation of integers from 0 to n-1.
    """
    if dtype:
        dtype = mappings.t2j_dtype(dtype)
    else:
        dtype = jnp.int64.dtype
    key = env.get_and_rotate_prng_key(generator)
    indices = jnp.arange(n, dtype=dtype)
    permutation = jax.random.permutation(key, indices)
    return permutation


# aten.reflection_pad3d


# aten.remainder
@op(torch.ops.aten.remainder)  # pyre-ignore[56]
def _aten_remainder(inputs: Any, other: Any) -> Any:
    return inputs % other


# aten.repeat
@op(torch.ops.aten.repeat)  # pyre-ignore[56]
def _aten_repeat(x: Any, reps: Any) -> Any:
    return jnp.tile(x, reps)


# aten.replication_pad2d
# aten.replication_pad3d
# aten.roll
@op(torch.ops.aten.roll)  # pyre-ignore[56]
def _aten_roll(input: Any, shifts: Any, dims: Any = None) -> Any:
    return jnp.roll(input, shifts, dims)


# aten.slice_scatter
@op(torch.ops.aten.slice_scatter)  # pyre-ignore[56]
def _aten_slice_scatter(
    input: Any,
    src: Any,
    dim: Any = 0,
    start: Any = None,
    end: Any = None,
    step: Any = 1,
) -> Any:
    input_index = []
    for x in range(len(input.shape)):
        if x == dim:
            input_index.append(slice(start, end, step))
        else:
            input_index.append(slice(None, None, None))
    return input.at[tuple(input_index)].set(src)


# aten.sort
# torch.sort(input, dim=-1, descending=False, stable=False, *, out=None)
@op(torch.ops.aten.sort)  # pyre-ignore[56]
def _aten_sort(
    a: Any, dim: Any = -1, descending: Any = False, stable: Any = False
) -> Any:
    if a.shape == ():
        return (a, jnp.astype(0, "int64"))
    return (
        jnp.sort(a, axis=dim, stable=stable, descending=descending),
        jnp.argsort(a, axis=dim, stable=stable, descending=descending),
    )


# aten.sym_size


# aten.topk
@op(torch.ops.aten.topk)  # pyre-ignore[56]
def _aten_topk(
    input: Any,
    k: Any,
    dim: Any = None,
    largest: Any = True,
    sorted: Any = True,
    *,
    out: Any = None,
) -> Any:
    """JAX top-k implementation using jax.lax.top_k for improved efficiency.

    Args:
        input: The input JAX array.
        k: The number of top elements to return.
        dim: The dimension along which to find the top-k. If None, operates on the
          flattened array.
        largest: If True, returns the largest k elements. Otherwise, smallest k.
        sorted: If True, returns the elements in sorted order.

    Returns:
        A tuple (values, indices) containing:
            - values: The top k values.
            - indices: The indices of the top k values in the original array.
    """
    if dim is None:
        # last dim is chosen
        dim = input.ndim - 1

    if dim < 0:
        dim = dim + input.ndim

    if not largest:
        input = -input  # Find top-k of negated input if we want the smallest

    if input.ndim == 0:
        return input, jnp.array(0, dtype=jnp.int64.dtype)

    transpose_shape = None
    if dim != -1 and dim != len(input.shape) - 1:
        transpose_shape = list(range(len(input.shape)))
        transpose_shape[dim], transpose_shape[-1] = (
            transpose_shape[-1],
            transpose_shape[dim],
        )
        input = jnp.transpose(input, transpose_shape)

    values, indices = jax.lax.top_k(input, k)

    if sorted:
        values = jnp.sort(values, descending=True)
        indices = jnp.take_along_axis(
            indices, jnp.argsort(values, axis=-1, descending=True), axis=-1
        )

    if not largest:
        values = -values  # Negate values back if we found smallest

    if transpose_shape is not None:
        values = jnp.transpose(values, transpose_shape)
        indices = jnp.transpose(indices, transpose_shape)

    return values, indices


# aten.tril_indices
# tril_indices(int row, int col, int offset=0, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None)
@op(torch.ops.aten.tril_indices)  # pyre-ignore[56]
def _aten_tril_indices(
    row: Any,
    col: Any,
    offset: Any = 0,
    *,
    dtype: Any = jnp.int64.dtype,
    layout: Any = None,
    device: Any = None,
    pin_memory: Any = None,
) -> Any:
    a, b = jnp.tril_indices(row, offset, col)
    return jnp.stack((a, b))


# aten.tril_indices
# tril_indices(int row, int col, int offset=0, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None)
@op(torch.ops.aten.triu_indices)  # pyre-ignore[56]
def _aten_triu_indices(
    row: Any,
    col: Any,
    offset: Any = 0,
    *,
    dtype: Any = jnp.int64.dtype,
    layout: Any = None,
    device: Any = None,
    pin_memory: Any = None,
) -> Any:
    a, b = jnp.triu_indices(row, offset, col)
    return jnp.stack((a, b))


@op(torch.ops.aten.unbind_copy)  # pyre-ignore[56]
def _aten_unbind(a: Any, dim: Any = 0) -> Any:
    return [
        jax.lax.index_in_dim(a, i, dim, keepdims=False) for i in range(a.shape[dim])
    ]


# aten.unique_dim
#
# NOTE: Like the CUDA and CPU implementations, this implementation always sorts
# the tensor regardless of the `sorted` argument passed to `torch.unique`.
@op(torch.ops.aten.unique_dim)  # pyre-ignore[56]
def _aten_unique_dim(
    input_tensor: Any,
    dim: Any,
    sort: Any = True,
    return_inverse: Any = False,
    return_counts: Any = False,
) -> Any:
    result_tensor_or_tuple = jnp.unique(
        input_tensor,
        return_index=False,
        return_inverse=return_inverse,
        return_counts=return_counts,
        axis=dim,
        equal_nan=False,
    )
    result_list = (
        list(result_tensor_or_tuple)
        if isinstance(result_tensor_or_tuple, tuple)
        else [result_tensor_or_tuple]
    )

    if not return_inverse:
        result_list.insert(1, None)
    elif _jax_version < (0, 4, 31) and dim is not None:
        result_list[1] = result_list[1].flatten()

    if not return_counts:
        result_list.insert(2, None)

    # [result, None,    None]    if return_inverse=False and return_counts=False
    # [result, inverse, None]    if return_inverse=True  and return_counts=False
    # [result, None,    counts]  if return_inverse=False and return_counts=True
    # [result, inverse, counts]  if return_inverse=True  and return_counts=True
    return result_list


# aten._unique
#
# NOTE: Like the CUDA and CPU implementations, this implementation always sorts
# the tensor regardless of the `sorted` argument passed to `torch.unique`.
@op(torch.ops.aten._unique)  # pyre-ignore[56]
def _aten_unique(
    input_tensor: Any, sort: Any = True, return_inverse: Any = False
) -> Any:
    result_tensor_or_tuple = jnp.unique(
        input_tensor,
        return_index=False,
        return_inverse=return_inverse,
        return_counts=False,
        axis=None,
        equal_nan=False,
    )
    if return_inverse:
        return result_tensor_or_tuple
    else:
        return (result_tensor_or_tuple, None)


# aten._unique2
#
# NOTE: Like the CUDA and CPU implementations, this implementation always sorts
# the tensor regardless of the `sorted` argument passed to `torch.unique`.
@op(torch.ops.aten._unique2)  # pyre-ignore[56]
def _aten_unique2(
    input_tensor: Any,
    sort: Any = True,
    return_inverse: Any = False,
    return_counts: Any = False,
) -> Any:
    return _aten_unique_dim(
        input_tensor=input_tensor,
        dim=None,
        sort=sort,
        return_inverse=return_inverse,
        return_counts=return_counts,
    )


# aten.unique_consecutive
@op(torch.ops.aten.unique_consecutive)  # pyre-ignore[56]
def _aten_unique_consecutive(
    input_tensor: Any,
    return_inverse: Any = False,
    return_counts: Any = None,
    dim: Any = None,
) -> Any:
    # Explanation of computations (shown in 1D for simplicity):
    #
    #   Input                                      [a b b c c c d d d d e e e e e]
    #   Slice dropping final element (input[:-1])    [a b b c c c d d d d e e e e]
    #   Slice dropping first element (input[1:])     [b b c c c d d d d e e e e e]
    #   Boolean != operation on shifted slices       [1 0 1 0 0 1 0 0 0 1 0 0 0 0]
    #   Prepend 1 to represent the first element   [1 1 0 1 0 0 1 0 0 0 1 0 0 0 0]
    #   Filter input by the resulting bool array   [a b   c     d       e        ]
    #   Output                                     [a b c d e]

    if dim is None:
        inverse_shape = input_tensor.shape
        input_tensor = input_tensor.flatten()
        ndim = 1
        dim = 0
    else:
        inverse_shape = input_tensor.shape[dim]
        ndim = input_tensor.ndim
        if dim < 0:
            dim += ndim

    nd_slice_0 = tuple(
        slice(None, -1) if d == dim else slice(None) for d in range(ndim)
    )
    nd_slice_1 = tuple(slice(1, None) if d == dim else slice(None) for d in range(ndim))

    axes_to_reduce = tuple(d for d in range(ndim) if d != dim)

    does_not_equal_prior = jnp.any(
        input_tensor[nd_slice_0] != input_tensor[nd_slice_1],
        axis=axes_to_reduce,
        keepdims=False,
    )

    if input_tensor.shape[dim] != 0:
        # Prepend `True` to represent the first element of the input.
        does_not_equal_prior = jnp.insert(does_not_equal_prior, 0, True)

    include_indices = jnp.argwhere(does_not_equal_prior)[:, 0]

    output_tensor = input_tensor[
        tuple(include_indices if d == dim else slice(None) for d in range(ndim))
    ]

    if return_inverse or return_counts:
        counts = (
            jnp.append(include_indices[1:], input_tensor.shape[dim])
            - include_indices[:]
        )

        inverse = (
            jnp.reshape(jnp.repeat(jnp.arange(len(counts)), counts), inverse_shape)
            if return_inverse
            else None
        )

        return output_tensor, inverse, counts

    return output_tensor, None, None


# NOTE: skip aten.upsample_nearest2d and aten.upsample_bilinear2d
# despite those being core aten ops, they also have decompositions.
# here we are using torch decompositions.


# aten.where
@op(torch.ops.aten.where)
@op(torch.ops.aten.where.self)
@op(torch.ops.aten.where.ScalarSelf)
@op(torch.ops.aten.where.ScalarOther)
@op(torch.ops.aten.where.Scalar)  # pyre-ignore[56]
def _aten_where(condition: Any, x: Any = None, y: Any = None) -> Any:
    return jnp.where(condition, x, y)


# aten.to.dtype
# Tensor(a) self, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None
@op(torch.ops.aten.to.dtype)  # pyre-ignore[56]
def _aten_to_dtype(
    a: Any,
    dtype: Any,
    non_blocking: Any = False,
    copy: Any = False,
    memory_format: Any = None,
) -> Any:
    if dtype:
        jaxdtype = mappings.t2j_dtype(dtype)
    return a.astype(jaxdtype)  # pyre-ignore[61]


@op(torch.ops.aten.to.dtype_layout)  # pyre-ignore[56]
def _aten_to_dtype_layout(
    a: Any,
    *,
    dtype: Any = None,
    layout: Any = None,
    device: Any = None,
    pin_memory: Any = None,
    non_blocking: Any = False,
    copy: Any = False,
    memory_format: Any = None,
) -> Any:
    return _aten_to_dtype(
        a,
        dtype,
        non_blocking=non_blocking,
        copy=copy,
        memory_format=memory_format,
    )


# aten.to.device


# Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False
@op(torch.ops.aten.var_mean.correction)  # pyre-ignore[56]
def _aten_var_mean_correction(
    tensor: Any, dim: Any = None, correction: Any = 1, keepdim: Any = False
) -> Any:
    # The internal API technically has a default `correction` argument of `None`,
    # but the public API has a default argument of 1. Therefore, we simply set our
    # default argument to 1. However, since the argument is officially supposed to
    # be nullable, we still need to check for `None` per the API contract.
    if correction is None:
        correction = 1
    mean = jnp.mean(tensor, axis=dim, keepdims=keepdim)
    # TODO: Pass in the `mean=mean` argument once `jax.numpy.var` supports it.
    var = jnp.var(tensor, axis=dim, ddof=correction, keepdims=keepdim)
    return var, mean


@op(torch.ops.aten.scalar_tensor)  # pyre-ignore[56]
@op_base.convert_dtype()
def _aten_scalar_tensor(
    s: Any,
    dtype: Any = None,
    layout: Any = None,
    device: Any = None,
    pin_memory: Any = None,
) -> Any:
    return jnp.array(s, dtype=dtype)


@op(torch.ops.aten.to.device)  # pyre-ignore[56]
def _aten_to_device(x: Any, device: Any, dtype: Any) -> Any:
    return x


@op(torch.ops.aten.max_pool2d_with_indices_backward)  # pyre-ignore[56]
def max_pool2d_with_indices_backward_custom(
    grad_output: Any,
    self: Any,
    kernel_size: Any,
    stride: Any,
    padding: Any,
    dilation: Any,
    ceil_mode: Any,
    indices: Any,
) -> Any:
    """
    Approximates the gradient calculation of PyTorch's max_pool2d_with_indices_backward.

    Args:
        grad_output: The gradient tensor from the preceding layer.
        self: The input tensor on which the original max pooling was performed.
        kernel_size: The size of the pooling window.
        stride: The stride of the pooling window.
        padding: The padding applied during max pooling.
        dilation: The dilation factor for the pooling operation.
        ceil_mode: Whether to use ceil or floor when calculating output shapes.
        indices: The indices of the maximum values, as produced by max_pool2d_with_indices.

    Returns:
        The calculated gradient with respect to the input (grad_input).
    """

    kH, kW = kernel_size
    dH, dW = stride
    padH, padW = padding
    dilH, dilW = dilation

    # Calculate output shape (may need adjustment based on ceil_mode)
    jnp.array(self.shape)
    grad_input = jnp.zeros_like(self)

    # Iterate over the flattened input and output tensors
    for i, _idx in enumerate(indices.flatten()):
        # Calculate input coordinates corresponding to the maximum value
        out_y, out_x = i // grad_output.shape[3], i % grad_output.shape[3]
        in_y = out_y * dH - padH + out_y * (dilH - 1)
        in_x = out_x * dW - padW + out_x * (dilW - 1)

        # Scatter the gradient to the appropriate input locations (handling potential overlaps)
        for y in range(in_y, in_y + kH):
            for x in range(in_x, in_x + kW):
                if 0 <= y < grad_input.shape[2] and 0 <= x < grad_input.shape[3]:
                    grad_input = grad_input.at[y, x].add(grad_output.flatten()[i])

    return grad_input


@op(torch.ops.aten._local_scalar_dense)  # pyre-ignore[56]
def _aten_local_scalar_dense(x: Any) -> Any:
    return x.item()


@op(torch.ops.aten.tensor_split.sections)  # pyre-ignore[56]
def _aten_tensor_split(ary: Any, indices_or_sections: Any, axis: Any = 0) -> Any:
    return jnp.array_split(ary, indices_or_sections, axis)


@op(torch.ops.aten.randn, needs_env=True)  # pyre-ignore[56]
@op_base.convert_dtype()
def _aten_randn(
    *size: Any,
    generator: Any = None,
    out: Any = None,
    dtype: Any = None,
    layout: Any = torch.strided,
    device: Any = None,
    requires_grad: Any = False,
    pin_memory: Any = False,
    env: Any = None,
) -> Any:
    shape = size
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = shape[0]
    key = env.get_and_rotate_prng_key(generator)
    res = jax.random.normal(key, shape)
    if dtype is not None:
        res = res.astype(dtype)
    return res


@op(torch.ops.aten.bernoulli.p, needs_env=True)  # pyre-ignore[56]
def _aten_bernoulli(
    self: Any,
    p: Any = 0.5,
    *,
    generator: Any = None,
    env: Any = None,
) -> Any:
    key = env.get_and_rotate_prng_key(generator)
    res = jax.random.uniform(key, self.shape) < p
    return res


@op(torch.ops.aten.geometric, needs_env=True)  # pyre-ignore[56]
def geometric(self: Any, p: Any, *, generator: Any = None, env: Any = None) -> Any:
    key = env.get_and_rotate_prng_key(generator)
    res = jax.random.geometric(key, p, self.shape)
    return res


@op(torch.ops.aten.randn_like, needs_env=True)  # pyre-ignore[56]
@op_base.convert_dtype()
def _aten_randn_like(
    x: Any,
    *,
    dtype: Any = None,
    layout: Any = None,
    device: Any = None,
    pin_memory: Any = False,
    memory_format: Any = torch.preserve_format,
    env: Any = None,
) -> Any:
    key = env.get_and_rotate_prng_key()
    return jax.random.normal(key, dtype=dtype or x.dtype, shape=x.shape)


@op(torch.ops.aten.rand, needs_env=True)  # pyre-ignore[56]
@op_base.convert_dtype()
def _rand(
    *size: Any,
    generator: Any = None,
    out: Any = None,
    dtype: Any = None,
    layout: Any = torch.strided,
    device: Any = None,
    requires_grad: Any = False,
    pin_memory: Any = False,
    env: Any = None,
) -> Any:
    shape = size
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = shape[0]
    key = env.get_and_rotate_prng_key(generator)
    res = jax.random.uniform(key, shape)
    if dtype is not None:
        res = res.astype(dtype)
    return res


@op(torch.ops.aten.outer)  # pyre-ignore[56]
def _aten_outer(a: Any, b: Any) -> Any:
    return jnp.outer(a, b)


@op(torch.ops.aten.allclose)  # pyre-ignore[56]
def _aten_allclose(
    input: Any, other: Any, rtol: Any = 1e-05, atol: Any = 1e-08, equal_nan: Any = False
) -> Any:
    return jnp.allclose(input, other, rtol, atol, equal_nan)


@op(torch.ops.aten.native_batch_norm)  # pyre-ignore[56]
def _aten_native_batch_norm(
    input: Any,
    weight: Any,
    bias: Any,
    running_mean: Any,
    running_var: Any,
    training: Any = False,
    momentum: Any = 0.1,
    eps: Any = 1e-5,
) -> Any:
    if running_mean is None:
        running_mean = jnp.zeros(
            input.shape[1], dtype=input.dtype
        )  # Initialize running mean if None
    if running_var is None:
        running_var = jnp.ones(
            input.shape[1], dtype=input.dtype
        )  # Initialize running variance if None

    if training:
        return _aten__native_batch_norm_legit(
            input,
            weight,
            bias,
            running_mean,
            running_var,
            training,
            momentum,
            eps,
        )
    else:
        return _aten__native_batch_norm_legit_no_training(
            input, weight, bias, running_mean, running_var, momentum, eps
        )


@op(torch.ops.aten.normal, needs_env=True)  # pyre-ignore[56]
def _aten_normal(
    self: Any, mean: Any = 0, std: Any = 1, generator: Any = None, env: Any = None
) -> Any:
    shape = self.shape
    res = _aten_randn(*shape, generator=generator, env=env)
    return res * std + mean


# TODO: not clear what this function should actually do
# https://github.com/pytorch/pytorch/blob/d96c80649f301129219469d8b4353e52edab3b78/aten/src/ATen/native/native_functions.yaml#L7933-L7940
@op(torch.ops.aten.lift_fresh)  # pyre-ignore[56]
def _aten_lift_fresh(self: Any) -> Any:
    return self


@op(torch.ops.aten.uniform, needs_env=True)  # pyre-ignore[56]
def _aten_uniform(
    self: Any, from_: Any = 0, to: Any = 1, *, generator: Any = None, env: Any = None
) -> Any:
    assert from_ <= to, (
        f"Uniform from(passed in {from_}) must be less than to(passed in {to})"
    )
    shape = self.shape
    res = _rand(*shape, generator=generator, env=env)
    return res * (to - from_) + from_


# func: randint.low_generator(SymInt low, SymInt high, SymInt[] size, *, Generator? generator, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor


_INT32_MAX = 2**31 - 1
_INT32_MIN = -(2**31)


def _randint_via_bits(key: Any, size: Any, low: int, high: int) -> Any:
    """jax.random.randint replacement that supports `high` slightly > int32_max.

    `jax.random.randint(key, shape, minval, maxval)` restricts minval and
    maxval to fit in int32. Engram's `NgramHashMapping` uses `high=2**31`,
    which is one over the limit. Use `jax.random.bits` to generate raw
    uint32 and reduce into `[0, range_size)` via uint32 modulo (stays in
    uint32 throughout to avoid the int32 overflow that arithmetic on
    `int(2**31)` triggers). Then cast back to int32 (values fit since
    range_size <= 2**31 means max output is 2**31 - 1) and add `low`.

    Bias is bounded by `range_size / 2**32`; for engram's case
    (range=2**31, bits=32) bias is 0.5 -- acceptable for hash init.

    Supports range up to 2**31 (one slot beyond stock `jax.random.randint`).
    For range > 2**31, uint64 + x64 would be needed; raise instead.
    """
    range_size = int(high) - int(low)
    if range_size <= 0:
        raise ValueError(f"_aten_randint: empty range [{low}, {high})")
    # Algorithm correctness: we cast the modulo result back to int32, so
    # ``res_uint`` must fit in [0, 2**31 - 1]. That bounds ``range_size``
    # by 2**31 (one slot beyond stock ``jax.random.randint``). Larger
    # ranges would silently wrap to negative int32 values; raise instead
    # so the caller routes through a uint64 / x64 path.
    if range_size > 2**31:
        raise NotImplementedError(
            f"_aten_randint range {range_size} > 2**31 needs JAX_ENABLE_X64; "
            "not supported by this fallback"
        )
    # Matching bound on the endpoints: ``low`` must be >= int32_min (so
    # the final ``+ low`` cast stays in int32) and ``high`` must be
    # <= 2**31 (so ``range_size`` stays in the supported regime above).
    if int(low) < -(2**31) or int(high) > 2**31:
        raise NotImplementedError(
            f"_aten_randint low={low} high={high} outside [int32_min, 2**31]"
        )
    uints = jax.random.bits(key, size, dtype=jnp.uint32)
    mod_arr = jnp.uint32(range_size)
    res_uint = uints % mod_arr
    # Cast back to int32 -- safe because range_size <= 2**31 means
    # values fit in [0, 2**31 - 1].
    return res_uint.astype(jnp.int32) + jnp.int32(int(low))


@op(torch.ops.aten.randint, needs_env=True)  # pyre-ignore[56]
@op_base.convert_dtype(use_default_dtype=False)
def _aten_randint(
    *args: Any,
    generator: Any = None,
    dtype: Any = None,
    env: Any = None,
    **kwargs: Any,
) -> Any:
    if len(args) == 3:
        # low, high, size
        low, high, size = args
    elif len(args) == 2:
        high, size = args
        low = 0
    else:
        raise AssertionError(
            f"Expected at 2 or 3 args for Aten::randint, got {len(args)}"
        )

    key = env.get_and_rotate_prng_key(generator)
    if int(low) < _INT32_MIN or int(high) > _INT32_MAX:
        res = _randint_via_bits(key, size, low, high)
    else:
        res = jax.random.randint(key, size, low, high)
    if dtype is not None:
        res = res.astype(dtype)
    return res


@op(  # pyre-ignore[56]
    torch.ops.aten.randint_like,
    torch.ops.aten.randint.generator,
    needs_env=True,
)
@op_base.convert_dtype(use_default_dtype=False)
def _aten_randint_like(
    input: Any,
    *args: Any,
    generator: Any = None,
    dtype: Any = None,
    env: Any = None,
    **kwargs: Any,
) -> Any:
    if len(args) == 2:
        low, high = args
    elif len(args) == 1:
        high = args[0]
        low = 0
    else:
        raise AssertionError(
            f"Expected at 1 or 2 args for Aten::randint_like, got {len(args)}"
        )

    shape = input.shape
    dtype = dtype or input.dtype
    key = env.get_and_rotate_prng_key(generator)
    if int(low) < _INT32_MIN or int(high) > _INT32_MAX:
        res = _randint_via_bits(key, shape, low, high)
    else:
        res = jax.random.randint(key, shape, low, high)
    if dtype is not None:
        res = res.astype(dtype)
    return res


@op(torch.ops.aten.dim, is_jax_function=False)  # pyre-ignore[56]
def _aten_dim(self: Any) -> Any:
    return len(self.shape)


@op(torch.ops.aten.copysign)  # pyre-ignore[56]
def _aten_copysign(input: Any, other: Any, *, out: Any = None) -> Any:
    result = jnp.copysign(input, other)
    # torch.copysign(x, y) returns float32 for integer x and y,
    # regardless of their exact integer dtype, whereas jax.copysign returns
    # float64 when one or both of them is int64.
    if jnp.issubdtype(input.dtype, jnp.integer) and jnp.issubdtype(
        other.dtype, jnp.integer
    ):
        result = result.astype(jnp.float32)
    return result


@op(torch.ops.aten.i0)  # pyre-ignore[56]
@op_base.promote_int_input
def _aten_i0(self: Any) -> Any:
    return jax.scipy.special.i0(self)


@op(torch.ops.aten.special_i0e)  # pyre-ignore[56]
@op_base.promote_int_input
def _aten_i0e(self: Any) -> Any:
    return jax.scipy.special.i0e(self)


@op(torch.ops.aten.special_i1)  # pyre-ignore[56]
@op_base.promote_int_input
def _aten_special_i1(self: Any) -> Any:
    return jax.scipy.special.i1(self)


@op(torch.ops.aten.special_i1e)  # pyre-ignore[56]
@op_base.promote_int_input
def _aten_special_i1e(self: Any) -> Any:
    return jax.scipy.special.i1e(self)


@op(torch.ops.aten.special_laguerre_polynomial_l)  # pyre-ignore[56]
@op_base.promote_int_input
def _aten_special_laguerre_polynomial_l(self: Any, n: Any) -> Any:
    # Adapted from https://github.com/pytorch/pytorch/blob/f8f41dcb24cb4f4e87a51bb04847942dd835e496/aten/src/ATen/native/Math.h#L3106-L3134

    @jnp.vectorize
    def vectorized(x: Any, n_i: Any) -> Any:
        def negative_n(x: Any) -> Any:
            return jnp.zeros_like(x)

        def zero_n(x: Any) -> Any:
            return jnp.ones_like(x)

        def one_n(x: Any) -> Any:
            return jnp.ones_like(x) - x

        def zero_abs(x: Any) -> Any:
            return jnp.ones_like(x)

        def default(x: Any) -> Any:
            def f(k: Any, carry: Any) -> Any:
                p, q = carry
                return (
                    q,
                    ((k * 2 + (jnp.ones_like(x) - x)) * q - k * p) / (k + 1),
                )

            _, q = jax.lax.fori_loop(1, n_i, f, init_val=(1.0, jnp.ones_like(x) - x))
            return q

        return jnp.piecewise(
            x,
            [n_i == 1, n_i == 0, jnp.abs(n_i) == jnp.zeros_like(x), n_i < 0],
            [one_n, zero_n, zero_abs, negative_n, default],
        )

    return vectorized(self, n.astype(jnp.int64))


@op(torch.ops.aten.special_modified_bessel_i0)  # pyre-ignore[56]
@op_base.promote_int_input
def _aten_special_modified_bessel_i0(self: Any) -> Any:
    # Adapted from https://github.com/pytorch/pytorch/blob/f8f41dcb24cb4f4e87a51bb04847942dd835e496/aten/src/ATen/native/Math.h#L3182-L3268

    def small(x: Any) -> Any:
        A = jnp.array(
            [
                -4.41534164647933937950e-18,
                3.33079451882223809783e-17,
                -2.43127984654795469359e-16,
                1.71539128555513303061e-15,
                -1.16853328779934516808e-14,
                7.67618549860493561688e-14,
                -4.85644678311192946090e-13,
                2.95505266312963983461e-12,
                -1.72682629144155570723e-11,
                9.67580903537323691224e-11,
                -5.18979560163526290666e-10,
                2.65982372468238665035e-09,
                -1.30002500998624804212e-08,
                6.04699502254191894932e-08,
                -2.67079385394061173391e-07,
                1.11738753912010371815e-06,
                -4.41673835845875056359e-06,
                1.64484480707288970893e-05,
                -5.75419501008210370398e-05,
                1.88502885095841655729e-04,
                -5.76375574538582365885e-04,
                1.63947561694133579842e-03,
                -4.32430999505057594430e-03,
                1.05464603945949983183e-02,
                -2.37374148058994688156e-02,
                4.93052842396707084878e-02,
                -9.49010970480476444210e-02,
                1.71620901522208775349e-01,
                -3.04682672343198398683e-01,
                6.76795274409476084995e-01,
            ],
            dtype=self.dtype,
        )

        def f(carry: Any, val: Any) -> Any:
            p, q, a = carry
            p, q = q, a
            return (p, q, ((x / 2.0) - 2.0) * q - p + val), None

        (p, _, a), _ = jax.lax.scan(
            f, init=(jnp.zeros_like(x), jnp.zeros_like(x), 0), xs=A
        )

        return jnp.exp(x) * (0.5 * (a - p))

    def default(x: Any) -> Any:
        B = jnp.array(
            [
                -7.23318048787475395456e-18,
                -4.83050448594418207126e-18,
                4.46562142029675999901e-17,
                3.46122286769746109310e-17,
                -2.82762398051658348494e-16,
                -3.42548561967721913462e-16,
                1.77256013305652638360e-15,
                3.81168066935262242075e-15,
                -9.55484669882830764870e-15,
                -4.15056934728722208663e-14,
                1.54008621752140982691e-14,
                3.85277838274214270114e-13,
                7.18012445138366623367e-13,
                -1.79417853150680611778e-12,
                -1.32158118404477131188e-11,
                -3.14991652796324136454e-11,
                1.18891471078464383424e-11,
                4.94060238822496958910e-10,
                3.39623202570838634515e-09,
                2.26666899049817806459e-08,
                2.04891858946906374183e-07,
                2.89137052083475648297e-06,
                6.88975834691682398426e-05,
                3.36911647825569408990e-03,
                8.04490411014108831608e-01,
            ],
            dtype=self.dtype,
        )

        def f(carry: Any, val: Any) -> Any:
            p, q, b = carry
            p, q = q, b
            return (p, q, (32.0 / x - 2.0) * q - p + val), None

        (p, _, b), _ = jax.lax.scan(
            f, init=(jnp.zeros_like(x), jnp.zeros_like(x), 0), xs=B
        )

        return jnp.exp(x) * (0.5 * (b - p)) / jnp.sqrt(x)

    self = jnp.abs(self)
    return jnp.piecewise(self, [self <= 8], [small, default])


@op(torch.ops.aten.special_modified_bessel_i1)  # pyre-ignore[56]
@op_base.promote_int_input
def _aten_special_modified_bessel_i1(self: Any) -> Any:
    # Adapted from https://github.com/pytorch/pytorch/blob/f8f41dcb24cb4f4e87a51bb04847942dd835e496/aten/src/ATen/native/Math.h#L3271-L3364

    def small(x: Any) -> Any:
        A = jnp.array(
            [
                2.77791411276104639959e-18,
                -2.11142121435816608115e-17,
                1.55363195773620046921e-16,
                -1.10559694773538630805e-15,
                7.60068429473540693410e-15,
                -5.04218550472791168711e-14,
                3.22379336594557470981e-13,
                -1.98397439776494371520e-12,
                1.17361862988909016308e-11,
                -6.66348972350202774223e-11,
                3.62559028155211703701e-10,
                -1.88724975172282928790e-09,
                9.38153738649577178388e-09,
                -4.44505912879632808065e-08,
                2.00329475355213526229e-07,
                -8.56872026469545474066e-07,
                3.47025130813767847674e-06,
                -1.32731636560394358279e-05,
                4.78156510755005422638e-05,
                -1.61760815825896745588e-04,
                5.12285956168575772895e-04,
                -1.51357245063125314899e-03,
                4.15642294431288815669e-03,
                -1.05640848946261981558e-02,
                2.47264490306265168283e-02,
                -5.29459812080949914269e-02,
                1.02643658689847095384e-01,
                -1.76416518357834055153e-01,
                2.52587186443633654823e-01,
            ],
            dtype=self.dtype,
        )

        def f(carry: Any, val: Any) -> Any:
            p, q, a = carry
            p, q = q, a
            return (p, q, ((jnp.abs(x) / 2.0) - 2.0) * q - p + val), None

        (p, _, a), _ = jax.lax.scan(
            f, init=(jnp.zeros_like(x), jnp.zeros_like(x), 0), xs=A
        )

        return jax.lax.cond(
            x < 0,
            lambda: -(0.5 * (a - p) * jnp.abs(x) * jnp.exp(jnp.abs(x))),
            lambda: 0.5 * (a - p) * jnp.abs(x) * jnp.exp(jnp.abs(x)),
        )

    def default(x: Any) -> Any:
        B = jnp.array(
            [
                7.51729631084210481353e-18,
                4.41434832307170791151e-18,
                -4.65030536848935832153e-17,
                -3.20952592199342395980e-17,
                2.96262899764595013876e-16,
                3.30820231092092828324e-16,
                -1.88035477551078244854e-15,
                -3.81440307243700780478e-15,
                1.04202769841288027642e-14,
                4.27244001671195135429e-14,
                -2.10154184277266431302e-14,
                -4.08355111109219731823e-13,
                -7.19855177624590851209e-13,
                2.03562854414708950722e-12,
                1.41258074366137813316e-11,
                3.25260358301548823856e-11,
                -1.89749581235054123450e-11,
                -5.58974346219658380687e-10,
                -3.83538038596423702205e-09,
                -2.63146884688951950684e-08,
                -2.51223623787020892529e-07,
                -3.88256480887769039346e-06,
                -1.10588938762623716291e-04,
                -9.76109749136146840777e-03,
                7.78576235018280120474e-01,
            ],
            dtype=self.dtype,
        )

        def f(carry: Any, val: Any) -> Any:
            p, q, b = carry
            p, q = q, b
            return (p, q, (32.0 / jnp.abs(x) - 2.0) * q - p + val), None

        (p, _, b), _ = jax.lax.scan(
            f, init=(jnp.zeros_like(x), jnp.zeros_like(x), 0), xs=B
        )

        return jax.lax.cond(
            x < 0,
            lambda: -(jnp.exp(jnp.abs(x)) * (0.5 * (b - p)) / jnp.sqrt(jnp.abs(x))),
            lambda: jnp.exp(jnp.abs(x)) * (0.5 * (b - p)) / jnp.sqrt(jnp.abs(x)),
        )

    return jnp.piecewise(self, [self <= 8], [small, default])


@op(torch.ops.aten.special_modified_bessel_k0)  # pyre-ignore[56]
@op_base.promote_int_input
def _aten_special_modified_bessel_k0(self: Any) -> Any:
    # Adapted from https://github.com/pytorch/pytorch/blob/f8f41dcb24cb4f4e87a51bb04847942dd835e496/aten/src/ATen/native/Math.h#L3367-L3441

    def zero(x: Any) -> Any:
        return jnp.array(jnp.inf, x.dtype)

    def negative(x: Any) -> Any:
        return jnp.array(jnp.nan, x.dtype)

    def small(x: Any) -> Any:
        A = jnp.array(
            [
                1.37446543561352307156e-16,
                4.25981614279661018399e-14,
                1.03496952576338420167e-11,
                1.90451637722020886025e-09,
                2.53479107902614945675e-07,
                2.28621210311945178607e-05,
                1.26461541144692592338e-03,
                3.59799365153615016266e-02,
                3.44289899924628486886e-01,
                -5.35327393233902768720e-01,
            ],
            dtype=self.dtype,
        )

        def f(carry: Any, val: Any) -> Any:
            p, q, a = carry
            p, q = q, a
            return (p, q, (x * x - 2.0) * q - p + val), None

        (p, _, a), _ = jax.lax.scan(
            f, init=(jnp.zeros_like(x), jnp.zeros_like(x), 0), xs=A
        )

        return 0.5 * (a - p) - jnp.log(0.5 * x) * _aten_special_modified_bessel_i0(x)

    def default(x: Any) -> Any:
        B = jnp.array(
            [
                5.30043377268626276149e-18,
                -1.64758043015242134646e-17,
                5.21039150503902756861e-17,
                -1.67823109680541210385e-16,
                5.51205597852431940784e-16,
                -1.84859337734377901440e-15,
                6.34007647740507060557e-15,
                -2.22751332699166985548e-14,
                8.03289077536357521100e-14,
                -2.98009692317273043925e-13,
                1.14034058820847496303e-12,
                -4.51459788337394416547e-12,
                1.85594911495471785253e-11,
                -7.95748924447710747776e-11,
                3.57739728140030116597e-10,
                -1.69753450938905987466e-09,
                8.57403401741422608519e-09,
                -4.66048989768794782956e-08,
                2.76681363944501510342e-07,
                -1.83175552271911948767e-06,
                1.39498137188764993662e-05,
                -1.28495495816278026384e-04,
                1.56988388573005337491e-03,
                -3.14481013119645005427e-02,
                2.44030308206595545468e00,
            ],
            dtype=self.dtype,
        )

        def f(carry: Any, val: Any) -> Any:
            p, q, b = carry
            p, q = q, b
            return (p, q, (8.0 / x - 2.0) * q - p + val), None

        (p, _, b), _ = jax.lax.scan(
            f, init=(jnp.zeros_like(x), jnp.zeros_like(x), 0), xs=B
        )

        return jnp.exp(-x) * (0.5 * (b - p)) / jnp.sqrt(x)

    return jnp.piecewise(
        self, [self <= 2, self < 0, self == 0], [small, negative, zero, default]
    )


@op(torch.ops.aten.special_modified_bessel_k1)  # pyre-ignore[56]
@op_base.promote_int_input
def _aten_special_modified_bessel_k1(self: Any) -> Any:
    # Adapted from https://github.com/pytorch/pytorch/blob/f8f41dcb24cb4f4e87a51bb04847942dd835e496/aten/src/ATen/native/Math.h#L3444-L3519

    def zero(x: Any) -> Any:
        return jnp.array(jnp.inf, x.dtype)

    def negative(x: Any) -> Any:
        return jnp.array(jnp.nan, x.dtype)

    def small(x: Any) -> Any:
        A = jnp.array(
            [
                -7.02386347938628759343e-18,
                -2.42744985051936593393e-15,
                -6.66690169419932900609e-13,
                -1.41148839263352776110e-10,
                -2.21338763073472585583e-08,
                -2.43340614156596823496e-06,
                -1.73028895751305206302e-04,
                -6.97572385963986435018e-03,
                -1.22611180822657148235e-01,
                -3.53155960776544875667e-01,
                1.52530022733894777053e00,
            ],
            dtype=self.dtype,
        )

        def f(carry: Any, val: Any) -> Any:
            p, q, a = carry
            p, q = q, a
            a = (x * x - 2.0) * q - p + val
            return (p, q, a), None

        (p, _, a), _ = jax.lax.scan(
            f, init=(jnp.zeros_like(x), jnp.zeros_like(x), 0), xs=A
        )

        return (
            jnp.log(0.5 * x) * _aten_special_modified_bessel_i1(x) + 0.5 * (a - p) / x
        )

    def default(x: Any) -> Any:
        B = jnp.array(
            [
                -5.75674448366501715755e-18,
                1.79405087314755922667e-17,
                -5.68946255844285935196e-17,
                1.83809354436663880070e-16,
                -6.05704724837331885336e-16,
                2.03870316562433424052e-15,
                -7.01983709041831346144e-15,
                2.47715442448130437068e-14,
                -8.97670518232499435011e-14,
                +3.34841966607842919884e-13,
                -1.28917396095102890680e-12,
                5.13963967348173025100e-12,
                -2.12996783842756842877e-11,
                9.21831518760500529508e-11,
                -4.19035475934189648750e-10,
                2.01504975519703286596e-09,
                -1.03457624656780970260e-08,
                5.74108412545004946722e-08,
                -3.50196060308781257119e-07,
                2.40648494783721712015e-06,
                -1.93619797416608296024e-05,
                1.95215518471351631108e-04,
                -2.85781685962277938680e-03,
                1.03923736576817238437e-01,
                2.72062619048444266945e00,
            ],
            dtype=self.dtype,
        )

        def f(carry: Any, val: Any) -> Any:
            p, q, b = carry
            p, q = q, b
            b = (8.0 / x - 2.0) * q - p + val
            return (p, q, b), None

        (p, _, b), _ = jax.lax.scan(
            f, init=(jnp.zeros_like(x), jnp.zeros_like(x), 0), xs=B
        )

        return jnp.exp(-x) * (0.5 * (b - p)) / jnp.sqrt(x)

    return jnp.piecewise(
        self, [self <= 2, self < 0, self == 0], [small, negative, zero, default]
    )


@op(torch.ops.aten.polygamma)  # pyre-ignore[56]
def _aten_polygamma(x: Any, n: Any) -> Any:
    if n.dtype in [jnp.int8, jnp.int16, jnp.int32, jnp.int64]:
        n = n.astype(mappings.t2j_dtype(torch.get_default_dtype()))
    return jax.lax.polygamma(jnp.float32(x), n)


@op(torch.ops.aten.special_ndtri)  # pyre-ignore[56]
@op_base.promote_int_input
def _aten_special_ndtri(self: Any) -> Any:
    return jax.scipy.special.ndtri(self)


@op(torch.ops.aten.special_bessel_j0)  # pyre-ignore[56]
@op_base.promote_int_input
def _aten_special_bessel_j0(self: Any) -> Any:
    # Adapted from https://github.com/pytorch/pytorch/blob/f8f41dcb24cb4f4e87a51bb04847942dd835e496/aten/src/ATen/native/Math.h#L2379-L2489

    def very_small(x: Any) -> Any:
        return 1.0 - x * x / 4.0

    def small(x: Any) -> Any:
        RP = jnp.array(
            [
                -4.79443220978201773821e09,
                1.95617491946556577543e12,
                -2.49248344360967716204e14,
                9.70862251047306323952e15,
            ],
            dtype=self.dtype,
        )
        RQ = jnp.array(
            [
                4.99563147152651017219e02,
                1.73785401676374683123e05,
                4.84409658339962045305e07,
                1.11855537045356834862e10,
                2.11277520115489217587e12,
                3.10518229857422583814e14,
                3.18121955943204943306e16,
                1.71086294081043136091e18,
            ],
            dtype=self.dtype,
        )

        rp = op_base.foreach_loop(RP, lambda carry, rp_i: carry * (x * x) + rp_i)
        rq = op_base.foreach_loop(RQ, lambda carry, rq_i: carry * (x * x) + rq_i)

        return (
            (x * x - 5.78318596294678452118e00)
            * (x * x - 3.04712623436620863991e01)
            * rp
            / rq
        )

    def default(x: Any) -> Any:
        PP = jnp.array(
            [
                7.96936729297347051624e-04,
                8.28352392107440799803e-02,
                1.23953371646414299388e00,
                5.44725003058768775090e00,
                8.74716500199817011941e00,
                5.30324038235394892183e00,
                9.99999999999999997821e-01,
            ],
            dtype=self.dtype,
        )
        PQ = jnp.array(
            [
                9.24408810558863637013e-04,
                8.56288474354474431428e-02,
                1.25352743901058953537e00,
                5.47097740330417105182e00,
                8.76190883237069594232e00,
                5.30605288235394617618e00,
                1.00000000000000000218e00,
            ],
            dtype=self.dtype,
        )
        QP = jnp.array(
            [
                -1.13663838898469149931e-02,
                -1.28252718670509318512e00,
                -1.95539544257735972385e01,
                -9.32060152123768231369e01,
                -1.77681167980488050595e02,
                -1.47077505154951170175e02,
                -5.14105326766599330220e01,
                -6.05014350600728481186e00,
            ],
            dtype=self.dtype,
        )
        QQ = jnp.array(
            [
                6.43178256118178023184e01,
                8.56430025976980587198e02,
                3.88240183605401609683e03,
                7.24046774195652478189e03,
                5.93072701187316984827e03,
                2.06209331660327847417e03,
                2.42005740240291393179e02,
            ],
            dtype=self.dtype,
        )

        pp = op_base.foreach_loop(
            PP, lambda carry, pp_i: carry * (25.0 / (x * x)) + pp_i
        )
        pq = op_base.foreach_loop(
            PQ, lambda carry, pq_i: carry * (25.0 / (x * x)) + pq_i
        )
        qp = op_base.foreach_loop(
            QP, lambda carry, qp_i: carry * (25.0 / (x * x)) + qp_i
        )
        qq = op_base.foreach_loop(
            QQ, lambda carry, qq_i: carry * (25.0 / (x * x)) + qq_i
        )

        return (
            (
                pp / pq * jnp.cos(x - 0.785398163397448309615660845819875721)
                - 5.0
                / x
                * (qp / qq)
                * jnp.sin(x - 0.785398163397448309615660845819875721)
            )
            * 0.797884560802865355879892119868763737
            / jnp.sqrt(x)
        )

    self = jnp.abs(self)
    # Last True condition in  `piecewise` takes priority, but last function is
    # default. See https://github.com/numpy/numpy/issues/16475
    return jnp.piecewise(
        self, [self <= 5.0, self < 0.00001], [small, very_small, default]
    )


@op(torch.ops.aten.special_bessel_j1)  # pyre-ignore[56]
@op_base.promote_int_input
def _aten_special_bessel_j1(self: Any) -> Any:
    # Adapted from https://github.com/pytorch/pytorch/blob/f8f41dcb24cb4f4e87a51bb04847942dd835e496/aten/src/ATen/native/Math.h#L2491-L2597

    def small(x: Any) -> Any:
        RP = jnp.array(
            [
                -8.99971225705559398224e08,
                4.52228297998194034323e11,
                -7.27494245221818276015e13,
                3.68295732863852883286e15,
            ],
            dtype=self.dtype,
        )
        RQ = jnp.array(
            [
                6.20836478118054335476e02,
                2.56987256757748830383e05,
                8.35146791431949253037e07,
                2.21511595479792499675e10,
                4.74914122079991414898e12,
                7.84369607876235854894e14,
                8.95222336184627338078e16,
                5.32278620332680085395e18,
            ],
            dtype=self.dtype,
        )

        rp = op_base.foreach_loop(RP, lambda carry, rp_i: carry * (x * x) + rp_i)
        rq = op_base.foreach_loop(RQ, lambda carry, rq_i: carry * (x * x) + rq_i)

        return (
            rp
            / rq
            * x
            * (x * x - 1.46819706421238932572e01)
            * (x * x - 4.92184563216946036703e01)
        )

    def default(x: Any) -> Any:
        PP = jnp.array(
            [
                7.62125616208173112003e-04,
                7.31397056940917570436e-02,
                1.12719608129684925192e00,
                5.11207951146807644818e00,
                8.42404590141772420927e00,
                5.21451598682361504063e00,
                1.00000000000000000254e00,
            ],
            dtype=self.dtype,
        )
        PQ = jnp.array(
            [
                5.71323128072548699714e-04,
                6.88455908754495404082e-02,
                1.10514232634061696926e00,
                5.07386386128601488557e00,
                8.39985554327604159757e00,
                5.20982848682361821619e00,
                9.99999999999999997461e-01,
            ],
            dtype=self.dtype,
        )
        QP = jnp.array(
            [
                5.10862594750176621635e-02,
                4.98213872951233449420e00,
                7.58238284132545283818e01,
                3.66779609360150777800e02,
                7.10856304998926107277e02,
                5.97489612400613639965e02,
                2.11688757100572135698e02,
                2.52070205858023719784e01,
            ],
            dtype=self.dtype,
        )
        QQ = jnp.array(
            [
                7.42373277035675149943e01,
                1.05644886038262816351e03,
                4.98641058337653607651e03,
                9.56231892404756170795e03,
                7.99704160447350683650e03,
                2.82619278517639096600e03,
                3.36093607810698293419e02,
            ],
            dtype=self.dtype,
        )

        pp = op_base.foreach_loop(
            PP, lambda carry, pp_i: carry * (25.0 / (x * x)) + pp_i
        )
        pq = op_base.foreach_loop(
            PQ, lambda carry, pq_i: carry * (25.0 / (x * x)) + pq_i
        )
        qp = op_base.foreach_loop(
            QP, lambda carry, qp_i: carry * (25.0 / (x * x)) + qp_i
        )
        qq = op_base.foreach_loop(
            QQ, lambda carry, qq_i: carry * (25.0 / (x * x)) + qq_i
        )

        return (
            (
                pp / pq * jnp.cos(x - 2.356194490192344928846982537459627163)
                - 5.0
                / x
                * (qp / qq)
                * jnp.sin(x - 2.356194490192344928846982537459627163)
            )
            * 0.797884560802865355879892119868763737
            / jnp.sqrt(x)
        )

    # If x < 0, bessel_j1(x) = -bessel_j1(-x)
    sign = jnp.sign(self)
    self = jnp.abs(self)
    return sign * jnp.piecewise(
        self,
        [self <= 5.0],
        [small, default],
    )


@op(torch.ops.aten.special_bessel_y0)  # pyre-ignore[56]
@op_base.promote_int_input
def _aten_special_bessel_y0(self: Any) -> Any:
    # Adapted from https://github.com/pytorch/pytorch/blob/f8f41dcb24cb4f4e87a51bb04847942dd835e496/aten/src/ATen/native/Math.h#L2599-L2712

    def zero(x: Any) -> Any:
        return jnp.array(-jnp.inf, x.dtype)

    def negative(x: Any) -> Any:
        return jnp.array(jnp.nan, x.dtype)

    def small(x: Any) -> Any:
        YP = jnp.array(
            [
                1.55924367855235737965e04,
                -1.46639295903971606143e07,
                5.43526477051876500413e09,
                -9.82136065717911466409e11,
                8.75906394395366999549e13,
                -3.46628303384729719441e15,
                4.42733268572569800351e16,
                -1.84950800436986690637e16,
            ],
            dtype=self.dtype,
        )
        YQ = jnp.array(
            [
                1.04128353664259848412e03,
                6.26107330137134956842e05,
                2.68919633393814121987e08,
                8.64002487103935000337e10,
                2.02979612750105546709e13,
                3.17157752842975028269e15,
                2.50596256172653059228e17,
            ],
            dtype=self.dtype,
        )

        yp = op_base.foreach_loop(YP, lambda carry, yp_i: carry * (x * x) + yp_i)
        yq = op_base.foreach_loop(YQ, lambda carry, yq_i: carry * (x * x) + yq_i)

        return yp / yq + (
            0.636619772367581343075535053490057448
            * jnp.log(x)
            * _aten_special_bessel_j0(x)
        )

    def default(x: Any) -> Any:
        PP = jnp.array(
            [
                7.96936729297347051624e-04,
                8.28352392107440799803e-02,
                1.23953371646414299388e00,
                5.44725003058768775090e00,
                8.74716500199817011941e00,
                5.30324038235394892183e00,
                9.99999999999999997821e-01,
            ],
            dtype=self.dtype,
        )
        PQ = jnp.array(
            [
                9.24408810558863637013e-04,
                8.56288474354474431428e-02,
                1.25352743901058953537e00,
                5.47097740330417105182e00,
                8.76190883237069594232e00,
                5.30605288235394617618e00,
                1.00000000000000000218e00,
            ],
            dtype=self.dtype,
        )
        QP = jnp.array(
            [
                -1.13663838898469149931e-02,
                -1.28252718670509318512e00,
                -1.95539544257735972385e01,
                -9.32060152123768231369e01,
                -1.77681167980488050595e02,
                -1.47077505154951170175e02,
                -5.14105326766599330220e01,
                -6.05014350600728481186e00,
            ],
            dtype=self.dtype,
        )
        QQ = jnp.array(
            [
                6.43178256118178023184e01,
                8.56430025976980587198e02,
                3.88240183605401609683e03,
                7.24046774195652478189e03,
                5.93072701187316984827e03,
                2.06209331660327847417e03,
                2.42005740240291393179e02,
            ],
            dtype=self.dtype,
        )

        factor = 25.0 / (x * x)
        pp = op_base.foreach_loop(PP, lambda carry, pp_i: carry * factor + pp_i)
        pq = op_base.foreach_loop(PQ, lambda carry, pq_i: carry * factor + pq_i)
        qp = op_base.foreach_loop(QP, lambda carry, qp_i: carry * factor + qp_i)
        qq = op_base.foreach_loop(QQ, lambda carry, qq_i: carry * factor + qq_i)

        return (
            (
                pp / pq * jnp.sin(x - 0.785398163397448309615660845819875721)
                + 5.0
                / x
                * (qp / qq)
                * jnp.cos(x - 0.785398163397448309615660845819875721)
            )
            * 0.797884560802865355879892119868763737
            / jnp.sqrt(x)
        )

    return jnp.piecewise(
        self,
        [self <= 5.0, self < 0.0, self == 0.0],
        [small, negative, zero, default],
    )


@op(torch.ops.aten.special_bessel_y1)  # pyre-ignore[56]
@op_base.promote_int_input
def _aten_special_bessel_y1(self: Any) -> Any:
    # Adapted from https://github.com/pytorch/pytorch/blob/f8f41dcb24cb4f4e87a51bb04847942dd835e496/aten/src/ATen/native/Math.h#L2714-L2826

    def zero(x: Any) -> Any:
        return jnp.array(-jnp.inf, x.dtype)

    def negative(x: Any) -> Any:
        return jnp.array(jnp.nan, x.dtype)

    def small(x: Any) -> Any:
        YP = jnp.array(
            [
                1.26320474790178026440e09,
                -6.47355876379160291031e11,
                1.14509511541823727583e14,
                -8.12770255501325109621e15,
                2.02439475713594898196e17,
                -7.78877196265950026825e17,
            ],
            dtype=self.dtype,
        )
        YQ = jnp.array(
            [
                5.94301592346128195359e02,
                2.35564092943068577943e05,
                7.34811944459721705660e07,
                1.87601316108706159478e10,
                3.88231277496238566008e12,
                6.20557727146953693363e14,
                6.87141087355300489866e16,
                3.97270608116560655612e18,
            ],
            dtype=self.dtype,
        )

        yp = op_base.foreach_loop(YP, lambda carry, yp_i: carry * (x * x) + yp_i)
        yq = op_base.foreach_loop(YQ, lambda carry, yq_i: carry * (x * x) + yq_i)

        return x * (yp / yq) + (
            0.636619772367581343075535053490057448
            * (_aten_special_bessel_j1(x) * jnp.log(x) - 1.0 / x)
        )

    def default(x: Any) -> Any:
        PP = jnp.array(
            [
                7.62125616208173112003e-04,
                7.31397056940917570436e-02,
                1.12719608129684925192e00,
                5.11207951146807644818e00,
                8.42404590141772420927e00,
                5.21451598682361504063e00,
                1.00000000000000000254e00,
            ],
            dtype=self.dtype,
        )
        PQ = jnp.array(
            [
                5.71323128072548699714e-04,
                6.88455908754495404082e-02,
                1.10514232634061696926e00,
                5.07386386128601488557e00,
                8.39985554327604159757e00,
                5.20982848682361821619e00,
                9.99999999999999997461e-01,
            ],
            dtype=self.dtype,
        )
        QP = jnp.array(
            [
                5.10862594750176621635e-02,
                4.98213872951233449420e00,
                7.58238284132545283818e01,
                3.66779609360150777800e02,
                7.10856304998926107277e02,
                5.97489612400613639965e02,
                2.11688757100572135698e02,
                2.52070205858023719784e01,
            ],
            dtype=self.dtype,
        )
        QQ = jnp.array(
            [
                7.42373277035675149943e01,
                1.05644886038262816351e03,
                4.98641058337653607651e03,
                9.56231892404756170795e03,
                7.99704160447350683650e03,
                2.82619278517639096600e03,
                3.36093607810698293419e02,
            ],
            dtype=self.dtype,
        )

        factor = 25.0 / (x * x)
        pp = op_base.foreach_loop(PP, lambda carry, pp_i: carry * factor + pp_i)
        pq = op_base.foreach_loop(PQ, lambda carry, pq_i: carry * factor + pq_i)
        qp = op_base.foreach_loop(QP, lambda carry, qp_i: carry * factor + qp_i)
        qq = op_base.foreach_loop(QQ, lambda carry, qq_i: carry * factor + qq_i)

        return (
            (
                pp / pq * jnp.sin(x - 2.356194490192344928846982537459627163)
                + 5.0
                / x
                * (qp / qq)
                * jnp.cos(x - 2.356194490192344928846982537459627163)
            )
            * 0.797884560802865355879892119868763737
            / jnp.sqrt(x)
        )

    return jnp.piecewise(
        self,
        [self <= 5.0, self < 0.0, self == 0.0],
        [small, negative, zero, default],
    )


@op(torch.ops.aten.special_chebyshev_polynomial_t)  # pyre-ignore[56]
@op_base.promote_int_input
def _aten_special_chebyshev_polynomial_t(self: Any, n: Any) -> Any:
    # Adapted from https://github.com/pytorch/pytorch/blob/f8f41dcb24cb4f4e87a51bb04847942dd835e496/aten/src/ATen/native/Math.h#L2828-L2865

    @jnp.vectorize
    def vectorized(x: Any, n_i: Any) -> Any:
        def negative_n(x: Any) -> Any:
            return jnp.zeros_like(x)

        def one_x(x: Any) -> Any:
            return jnp.where(
                (x > 0) | (n_i % 2 == 0), jnp.ones_like(x), -jnp.ones_like(x)
            )

        def large_n_small_x(x: Any) -> Any:
            return jnp.cos(n_i * jnp.acos(x))

        def zero_n(x: Any) -> Any:
            return jnp.ones_like(x)

        def one_n(x: Any) -> Any:
            return x

        def default(x: Any) -> Any:
            def f(_: Any, carry: Any) -> Any:
                p, q = carry
                return (q, 2 * x * q - p)

            _, r = jax.lax.fori_loop(0, n_i - 1, f, init_val=(1.0, x))
            return r

        return jnp.piecewise(
            x,
            [
                n_i == 1,
                n_i == 0,
                (n_i == 6) & (jnp.abs(x) < 1),
                jnp.abs(x) == 1.0,
                n_i < 0,
            ],
            [one_n, zero_n, large_n_small_x, one_x, negative_n, default],
        )

    # Explcicitly vectorize since we must vectorizes over both self and n
    return vectorized(self, n.astype(jnp.int64))


@op(torch.ops.aten.special_chebyshev_polynomial_u)  # pyre-ignore[56]
@op_base.promote_int_input
def _aten_special_chebyshev_polynomial_u(self: Any, n: Any) -> Any:
    # Adapted from https://github.com/pytorch/pytorch/blob/f8f41dcb24cb4f4e87a51bb04847942dd835e496/aten/src/ATen/native/Math.h#L2872-L2913

    @jnp.vectorize
    def vectorized(x: Any, n_i: Any) -> Any:
        def negative_n(x: Any) -> Any:
            return jnp.zeros_like(x)

        def one_x(x: Any) -> Any:
            return jnp.where((x > 0) | (n_i % 2 == 0), n_i + 1, -(n_i + 1))

        def large_n_small_x(x: Any) -> Any:
            sin_acos_x = jnp.sin(jnp.acos(x))
            return jnp.where(
                sin_acos_x != 0,
                jnp.sin((n_i + 1) * jnp.acos(x)) / sin_acos_x,
                (n_i + 1) * jnp.cos((n_i + 1) * jnp.acos(x)) / x,
            )

        def zero_n(x: Any) -> Any:
            return jnp.ones_like(x)

        def one_n(x: Any) -> Any:
            return 2 * x

        def default(x: Any) -> Any:
            def f(_: Any, carry: Any) -> Any:
                p, q = carry
                return (q, 2 * x * q - p)

            _, r = jax.lax.fori_loop(0, n_i - 1, f, init_val=(1.0, 2 * x))
            return r

        return jnp.piecewise(
            x,
            [
                n_i == 1,
                n_i == 0,
                (n_i > 8) & (jnp.abs(x) < 1),
                jnp.abs(x) == 1.0,
                n_i < 0,
            ],
            [one_n, zero_n, large_n_small_x, one_x, negative_n, default],
        )

    return vectorized(self, n.astype(jnp.int64))


@op(torch.ops.aten.special_erfcx)  # pyre-ignore[56]
@op_base.promote_int_input
def _aten_special_erfcx(x: Any) -> Any:
    return jnp.exp(x * x) * jax.lax.erfc(x)


@op(torch.ops.aten.erfc)  # pyre-ignore[56]
@op_base.promote_int_input
def _aten_erfcx(x: Any) -> Any:
    return jax.lax.erfc(x)


@op(torch.ops.aten.special_hermite_polynomial_h)  # pyre-ignore[56]
@op_base.promote_int_input
def _aten_special_hermite_polynomial_h(self: Any, n: Any) -> Any:
    # Adapted from https://github.com/pytorch/pytorch/blob/f8f41dcb24cb4f4e87a51bb04847942dd835e496/aten/src/ATen/native/Math.h#L3036-L3061

    @jnp.vectorize
    def vectorized(x: Any, n_i: Any) -> Any:
        def negative_n(x: Any) -> Any:
            return jnp.zeros_like(x)

        def zero_n(x: Any) -> Any:
            return jnp.ones_like(x)

        def one_n(x: Any) -> Any:
            return 2 * x

        def default(x: Any) -> Any:
            def f(k: Any, carry: Any) -> Any:
                p, q = carry
                return (q, 2 * x * q - 2 * k * p)

            _, r = jax.lax.fori_loop(1, n_i, f, init_val=(1.0, 2 * x))
            return r

        return jnp.piecewise(
            x,
            [n_i == 1, n_i == 0, n_i < 0],
            [one_n, zero_n, negative_n, default],
        )

    return vectorized(self, n.astype(jnp.int64))


@op(torch.ops.aten.special_hermite_polynomial_he)  # pyre-ignore[56]
@op_base.promote_int_input
def _aten_special_hermite_polynomial_he(self: Any, n: Any) -> Any:
    # Adapted from https://github.com/pytorch/pytorch/blob/f8f41dcb24cb4f4e87a51bb04847942dd835e496/aten/src/ATen/native/Math.h#L3073-L3098

    @jnp.vectorize
    def vectorized(x: Any, n_i: Any) -> Any:
        def negative_n(x: Any) -> Any:
            return jnp.zeros_like(x)

        def zero_n(x: Any) -> Any:
            return jnp.ones_like(x)

        def one_n(x: Any) -> Any:
            return x

        def default(x: Any) -> Any:
            def f(k: Any, carry: Any) -> Any:
                p, q = carry
                return (q, x * q - k * p)

            _, r = jax.lax.fori_loop(1, n_i, f, init_val=(1.0, x))
            return r

        return jnp.piecewise(
            x,
            [n_i == 1.0, n_i == 0.0, n_i < 0],
            [one_n, zero_n, negative_n, default],
        )

    return vectorized(self, n.astype(jnp.int64))


@op(torch.ops.aten.multinomial, needs_env=True)  # pyre-ignore[56]
def _aten_multinomial(
    input: Any,
    num_samples: Any,
    replacement: Any = False,
    *,
    generator: Any = None,
    out: Any = None,
    env: Any = None,
) -> Any:
    assert num_samples <= input.shape[-1] or replacement, (
        "cannot take a larger sample than population when replacement=False"
    )
    key = env.get_and_rotate_prng_key(generator)
    if input.ndim == 1:
        return jax.random.choice(
            key, input.shape[-1], (num_samples,), replace=replacement, p=input
        )
    else:
        return jnp.array(
            [
                jax.random.choice(
                    key,
                    input.shape[-1],
                    (num_samples,),
                    replace=replacement,
                    p=input[i, :],
                )
                for i in range(input.shape[0])
            ]
        )


@op(torch.ops.aten.narrow)
@op(torch.ops.aten.narrow_copy)  # pyre-ignore[56]
def _aten_narrow(input: Any, dim: Any, start: Any, length: Any) -> Any:
    return jax.lax.dynamic_slice_in_dim(input, start, length, axis=dim)


@op(torch.ops.aten.flatten)  # pyre-ignore[56]
def _aten_flatten(x: Any, start_dim: Any = 0, end_dim: Any = -1) -> Any:
    """
    Flattens a JAX array (similar to torch.flatten).

    Args:
        x: The JAX array to be flattened.
        start_dim: The first dimension to include in the flattening.
        end_dim: The last dimension to include in the flattening.

    Returns:
        A flattened JAX array.
    """
    shape = x.shape

    if end_dim < 0:
        end_dim += len(shape)  # Handle negative indexing

    new_shape = (*shape[:start_dim], -1, *shape[end_dim + 1 :])  # pyre-ignore[60]
    return jnp.reshape(x, new_shape)


@op(torch.ops.aten.new_empty)  # pyre-ignore[56]
def _new_empty(self: Any, size: Any, **kwargs: Any) -> Any:
    dtype = kwargs.get("dtype")
    if dtype is not None:
        dtype = mappings.t2j_dtype(dtype)
    else:
        dtype = self.dtype
    return jnp.empty(size, dtype=dtype)


@op(torch.ops.aten.new_empty_strided)  # pyre-ignore[56]
def _new_empty_strided(
    self: Any, size: Any, stride: Any, dtype: Any = None, **kwargs: Any
) -> Any:
    # Ignore stride, since JAX and torch tensor doesn't share the same memory.
    if not dtype:
        return jnp.empty(size, dtype=self.dtype)
    else:
        jax_dtype = mappings.t2j_dtype(dtype)
        return jnp.empty(size, dtype=jax_dtype)


@op(torch.ops.aten._unsafe_index_put)  # pyre-ignore[56]
def _aten_unsafe_index_put(
    self: Any, indices: Any, values: Any, accumulate: Any = False
) -> Any:
    return _aten_index_put(self, indices, values, accumulate)


@op(  # pyre-ignore[56]
    torch.ops.aten.conj_physical,
    torch.ops.aten.conj,
    torch.ops.aten._conj_physical,
    torch.ops.aten._conj,
)
def _aten_conj_physical(self: Any) -> Any:
    return jnp.conjugate(self)


@op(torch.ops.aten.log_sigmoid)  # pyre-ignore[56]
def _aten_log_sigmoid(x: Any) -> Any:
    return jax.nn.log_sigmoid(x)


# torch.qr
@op_if_present("qr")  # pyre-ignore[56]
def _aten_qr(input: Any, *args: Any, **kwargs: Any) -> Any:
    jax_mode = "reduced"
    # torch bool param 'simple=True' corresponds to jax 'reduced' mode,
    # and simple=False corresponds to jax 'complete' mode.
    if kwargs.get("simple") is False:
        jax_mode = "complete"
    return jax.numpy.linalg.qr(input, mode=jax_mode)


# torch.linalg.qr
@op(torch.ops.aten.linalg_qr)  # pyre-ignore[56]
def _aten_linalg_qr(input: Any, *args: Any, **kwargs: Any) -> Any:
    mode = kwargs.get("mode", "reduced")
    return jax.numpy.linalg.qr(input, mode=mode)


# torch.linalg.matrix_exp
@op(torch.ops.aten.linalg_matrix_exp)  # pyre-ignore[56]
def _aten_linalg_matrix_exp(input: Any) -> Any:
    return jax.scipy.linalg.expm(input)


# torch._linalg.slogdet
@op(torch.ops.aten._linalg_slogdet)  # pyre-ignore[56]
def _aten__linalg_slogdet(input: Any) -> Any:
    res = jnp.linalg.slogdet(input)
    return res.sign, res.logabsdet


# torch.linalg.svd
@op(torch.ops.aten._linalg_svd)  # pyre-ignore[56]
def _aten__linalg_svd(a: Any, full_matrices: Any = False, **kwargs: Any) -> Any:
    return jnp.linalg.svd(a, full_matrices=full_matrices, **kwargs)


# torch.linalg.pinv
@op(torch.ops.aten.linalg_pinv.atol_rtol_tensor)  # pyre-ignore[56]
def _aten_linalg_pinv_atol_rtol_tensor(a: Any, rtol: Any = None, **kwargs: Any) -> Any:
    return jnp.linalg.pinv(a, rtol, hermitian=False)


# torch.linalg.solve
@op(torch.ops.aten._linalg_solve_ex)  # pyre-ignore[56]
def _aten__linalg_solve_ex(a: Any, b: Any) -> Any:
    batched = False
    if b.ndim > 1 and b.shape[-1] == a.shape[-1]:
        batched = True
        b = b[..., None]
    res = jnp.linalg.solve(a, b)
    if batched:
        res = res.squeeze(-1)
    info_shape = a.shape[:-2]
    info = jnp.zeros(info_shape, dtype=mappings.t2j_dtype(torch.int32))
    return res, info


# torch.linalg.solve_triangular
@op(torch.ops.aten.linalg_solve_triangular)  # pyre-ignore[56]
def _aten_linalg_solve_triangular(
    a: Any, b: Any, *, upper: Any = True, left: Any = True, unitriangular: Any = False
) -> Any:
    if left is False:
        a = jnp.matrix_transpose(a)
        b = jnp.matrix_transpose(b)
        upper = not upper
    res = jax.scipy.linalg.solve_triangular(
        a, b, lower=not upper, unit_diagonal=unitriangular
    )
    if left is False:
        res = jnp.matrix_transpose(res)
    return res


@op(torch.ops.aten.linalg_inv_ex)  # pyre-ignore[56]
def _aten_linalg_inv_ex(a: Any) -> Any:
    ainv = jnp.linalg.inv(a)
    info = jnp.zeros(a.shape[:-2], jnp.int32)
    return ainv, info


@op(torch.ops.aten._linalg_check_errors)  # pyre-ignore[56]
def _aten__linalg_check_errors(*args: Any, **kwargs: Any) -> Any:
    pass


@op(torch.ops.aten.median)  # pyre-ignore[56]
def _aten_median(self: Any, dim: Any = None, keepdim: Any = False) -> Any:
    output = _with_reduction_scalar(
        functools.partial(jnp.quantile, q=0.5, method="lower"),
        self,
        dim=dim,
        keepdim=keepdim,
    ).astype(self.dtype)
    if dim is None:
        return output
    else:
        index = _with_reduction_scalar(_get_median_index, self, dim, keepdim).astype(
            jnp.int64
        )
        return output, index


@op(torch.ops.aten.nanmedian)  # pyre-ignore[56]
def _aten_nanmedian(
    input: Any, dim: Any = None, keepdim: Any = False, *, out: Any = None
) -> Any:
    output = _with_reduction_scalar(
        functools.partial(jnp.nanquantile, q=0.5, method="lower"),
        input,
        dim=dim,
        keepdim=keepdim,
    ).astype(input.dtype)
    if dim is None:
        return output
    else:
        index = _with_reduction_scalar(_get_median_index, input, dim, keepdim).astype(
            jnp.int64
        )
        return output, index


def _get_median_index(x: Any, axis: Any = None, keepdims: Any = False) -> Any:
    sorted_arg = jnp.argsort(x, axis=axis)
    n = x.shape[axis] if axis is not None else x.size
    if n % 2 == 1:
        index = n // 2
    else:
        index = (n // 2) - 1
    if axis is None:
        median_index = sorted_arg[index]
    else:
        median_index = jnp.take(sorted_arg, index, axis=axis)
    if keepdims and axis is not None:
        median_index = jnp.expand_dims(median_index, axis)
    return median_index


@op_if_present("triangular_solve")  # pyre-ignore[56]
def _aten_triangular_solve(
    b: Any,
    a: Any,
    upper: Any = True,
    transpose: Any = False,
    unittriangular: Any = False,
) -> Any:
    return (
        jax.lax.linalg.triangular_solve(
            a,
            b,
            left_side=True,
            lower=not upper,
            transpose_a=transpose,
            unit_diagonal=unittriangular,
        ),
        a,
    )


# func: _fft_c2c(Tensor self, SymInt[] dim, int normalization, bool forward) -> Tensor
@op(torch.ops.aten._fft_c2c)  # pyre-ignore[56]
def _aten__fft_c2c(self: Any, dim: Any, normalization: Any, forward: Any) -> Any:
    if forward:
        norm = [
            "backward",
            "ortho",
            "forward",
        ][normalization]
        return jnp.fft.fftn(self, axes=dim, norm=norm)
    else:
        norm = [
            "forward",
            "ortho",
            "backward",
        ][normalization]
        return jnp.fft.ifftn(self, axes=dim, norm=norm)


@op(torch.ops.aten._fft_r2c)  # pyre-ignore[56]
def _aten__fft_r2c(self: Any, dim: Any, normalization: Any, onesided: Any) -> Any:
    norm = [
        "backward",
        "ortho",
        "forward",
    ][normalization]
    if onesided:
        return jnp.fft.rfftn(self, axes=dim, norm=norm)
    else:
        return jnp.fft.fftn(self, axes=dim, norm=norm)


@op(torch.ops.aten._fft_c2r)  # pyre-ignore[56]
def _aten__fft_c2r(self: Any, dim: Any, normalization: Any, last_dim_size: Any) -> Any:
    norm = [
        "forward",
        "ortho",
        "backward",
    ][normalization]
    if len(dim) == 1:
        s = [last_dim_size]
    else:
        s = None
    return jnp.fft.irfftn(self, norm=norm, axes=dim, s=s)


@op(torch.ops.aten._trilinear)  # pyre-ignore[56]
def _aten_trilinear(
    i1: Any,
    i2: Any,
    i3: Any,
    expand1: Any,
    expand2: Any,
    expand3: Any,
    sumdim: Any,
    unroll_dim: Any = 1,
) -> Any:
    return _aten_sum(
        jnp.expand_dims(i1, expand1)
        * jnp.expand_dims(i2, expand2)
        * jnp.expand_dims(i3, expand3),
        sumdim,
    )


@op(torch.ops.aten.max_unpool2d)
@op(torch.ops.aten.max_unpool3d)  # pyre-ignore[56]
def _aten_max_unpoolxd(
    input: Any, indices: Any, output_size: Any, stride: Any = None, padding: Any = 0
) -> Any:
    if output_size is None:
        raise ValueError(
            "output_size value is not set correctly. It cannot be None or empty."
        )

    output_size = [input.shape[0], input.shape[1]] + output_size
    output = jnp.zeros(output_size, dtype=input.dtype)

    for idx in np.ndindex(input.shape):
        max_index = indices[idx]
        spatial_dims = output_size[2:]  # (D, H, W)
        unpooled_spatial_idx = np.unravel_index(max_index, spatial_dims)
        full_idx = idx[:2] + unpooled_spatial_idx
        output = output.at[full_idx].set(input[idx])

    return output


def _aten_upsample(
    input: Any,
    output_size: Any,
    align_corners: Any,
    antialias: Any,
    method: Any,
    scale_factors: Any = None,
    scales_h: Any = None,
    scales_w: Any = None,
) -> Any:
    # input: is of type jaxlib.xla_extension.ArrayImpl
    image = input

    # https://jax.readthedocs.io/en/latest/_autosummary/jax.image.resize.html
    # Resize does not distinguish batch, channel size.
    # We need to leave them as is
    # https://pytorch.org/vision/stable/transforms.html#supported-input-types-and-conventions
    # pytorch image shape is (C,H,W) or (N,C,H,W)
    # N - batch size
    # C - no of channels
    # H,W - height, width

    shape = list(image.shape)
    # overriding output_size
    if scale_factors:
        shape[-1] = int(math.floor(shape[-1] * scale_factors[-1]))
        shape[-2] = int(math.floor(shape[-2] * scale_factors[-2]))
    if scales_h:
        shape[-2] = int(math.floor(shape[-2] * scales_h))
    if scales_w:
        shape[-1] = int(math.floor(shape[-1] * scales_w))
    # output_size overrides scale_factors, scales_*
    if output_size:
        shape[-1] = output_size[-1]
        shape[-2] = output_size[-2]

    # pytorch upsample_bilinear returns the input as is when the shape is the same as input
    if shape == list(image.shape):
        return image

    spatial_dims = (2, 3)
    if len(shape) == 3:
        spatial_dims = (1, 2)

    scale = [shape[i] / image.shape[i] for i in spatial_dims]
    if scale_factors:
        scale = scale_factors
    if scales_h:
        scale[0] = scales_h
    if scales_w:
        scale[1] = scales_w
    scale = jnp.array(scale)

    # align_corners is not supported in resize()
    # https://github.com/jax-ml/jax/issues/11206
    if align_corners:
        scale = jnp.array(
            [(shape[i] - 1.0) / (image.shape[i] - 1.0) for i in spatial_dims]
        )

    translation = jnp.array([0 for i in spatial_dims])

    return jax_reimplement.scale_and_translate(
        image,
        shape,
        method=method,
        scale=scale,
        spatial_dims=spatial_dims,
        translation=translation,
        antialias=antialias,
    )


@op(torch.ops.aten._upsample_bilinear2d_aa)  # pyre-ignore[56]
def _aten_upsample_billinear_aa(
    input: Any,
    output_size: Any,
    align_corners: Any,
    scale_factors: Any = None,
    scales_h: Any = None,
    scales_w: Any = None,
) -> Any:
    return _aten_upsample(
        input,
        output_size,
        align_corners,
        True,  # antialias
        "bilinear",  # method
        scale_factors,
        scales_h,
        scales_w,
    )


@op(torch.ops.aten._upsample_bicubic2d_aa)  # pyre-ignore[56]
def _aten_upsample_bicubic2d_aa(
    input: Any,
    output_size: Any,
    align_corners: Any,
    scale_factors: Any = None,
    scales_h: Any = None,
    scales_w: Any = None,
) -> Any:
    return _aten_upsample(
        input,
        output_size,
        align_corners,
        True,  # antialias
        "bicubic",  # method
        scale_factors,
        scales_h,
        scales_w,
    )


@op(torch.ops.aten.polar)  # pyre-ignore[56]
def _aten_polar(abs: Any, angle: Any, *, out: Any = None) -> Any:
    return jax.lax.complex(abs * jnp.cos(angle), abs * jnp.sin(angle))


@op(torch.ops.aten.cdist)  # pyre-ignore[56]
def _aten_cdist(
    x1: Any,
    x2: Any,
    p: Any = 2.0,
    compute_mode: Any = "use_mm_for_euclid_dist_if_necessary",
) -> Any:
    x1 = x1.astype(jnp.float32)
    x2 = x2.astype(jnp.float32)

    if p == 0.0:
        # For p = 0, use Hamming-like distance multiplied by the number of elements
        return _hamming_distance(x1, x2).astype(jnp.float32)
    elif p == 2.0:
        # Use optimized Euclidean distance calculation
        if compute_mode == "use_mm_for_euclid_dist_if_necessary" and (
            x1.shape[-2] > 25 or x2.shape[-2] > 25
        ):
            return _euclidean_mm(x1, x2)
        elif compute_mode == "use_mm_for_euclid_dist":
            return _euclidean_mm(x1, x2)
        else:
            return _euclidean_direct(x1, x2)
    else:
        # General p-norm distance calculation
        diff = jnp.abs(jnp.expand_dims(x1, -2) - jnp.expand_dims(x2, -3))
        return jnp.sum(jnp.power(diff, p), axis=-1).astype(jnp.float32) ** (1 / p)


def _hamming_distance(x1: Any, x2: Any) -> Any:
    """
    Computes the Hamming-like distance for p=0.

    Args:
        x1: JAX array of shape (..., P, M)
        x2: JAX array of shape (..., R, M)

    Returns:
        JAX array of shape (..., P, R) representing pairwise Hamming distances.
    """
    diff = jnp.not_equal(jnp.expand_dims(x1, -2), jnp.expand_dims(x2, -3))

    hamming_dist = jnp.sum(diff, axis=-1).astype(jnp.float32)

    return hamming_dist


def _euclidean_mm(x1: Any, x2: Any) -> Any:
    """
    Computes the Euclidean distance using matrix multiplication.

    Args:
        x1: JAX array of shape (..., P, M)
        x2: JAX array of shape (..., R, M)

    Returns:
        JAX array of shape (..., P, R) representing pairwise Euclidean distances.
    """
    x1_sq = jnp.sum(x1**2, axis=-1, keepdims=True).astype(jnp.float32)
    x2_sq = jnp.sum(x2**2, axis=-1, keepdims=True).astype(jnp.float32)

    x2_sq = jnp.swapaxes(x2_sq, -2, -1)

    dot_product = jnp.matmul(x1, jnp.swapaxes(x2, -1, -2))

    dist_sq = x1_sq + x2_sq - 2 * dot_product
    dist_sq = jnp.maximum(dist_sq, 0.0)
    dist = jnp.sqrt(dist_sq).astype(jnp.float32)

    return dist


def _euclidean_direct(x1: Any, x2: Any) -> Any:
    """
    Computes the Euclidean distance directly without matrix multiplication.

    Args:
        x1: JAX array of shape (..., P, M)
        x2: JAX array of shape (..., R, M)

    Returns:
        JAX array of shape (..., P, R) representing pairwise Euclidean distances.
    """
    diff = jnp.expand_dims(x1, -2) - jnp.expand_dims(x2, -3)

    dist_sq = jnp.sum(diff**2, axis=-1).astype(jnp.float32)

    dist_sq = jnp.maximum(dist_sq, 0.0)

    dist = jnp.sqrt(dist_sq).astype(jnp.float32)

    return dist


@op(torch.ops.aten.lu_unpack)  # pyre-ignore[56]
def _aten_lu_unpack(
    LU_data: Any, LU_pivots: Any, unpack_data: Any = True, unpack_pivots: Any = True
) -> Any:
    # lu_unpack doesnt exist in jax.
    # Get commonly used data shape variables
    n = LU_data.shape[-2]
    m = LU_data.shape[-1]
    dim = min(n, m)

    ### Compute the Lower and Upper triangle
    if unpack_data:
        # Extract lower triangle
        L = jnp.tril(LU_data, k=-1)

        # emulate pytorch behavior: Add ones to the diagonal of L
        eye = jnp.eye(n, m, dtype=LU_data.dtype)
        L = L + eye

        # emulate pytorch behavior: Reshape lower triangle to match pivot
        start_indices = jnp.zeros(len(LU_data.shape), dtype=int)
        limit_indices = list(LU_data.shape)
        limit_indices[-1] = dim
        L = jax.lax.slice(L, start_indices, limit_indices)  # pyre-ignore[6]

        # Extract upper triangle
        U = jnp.triu(LU_data)

        # emulate pytorch behavior: Reshape upper triangle to match pivot
        start_indices = jnp.zeros(len(LU_data.shape), dtype=int)
        limit_indices = list(LU_data.shape)
        limit_indices[-2] = dim
        U = jax.lax.slice(U, start_indices, limit_indices)  # pyre-ignore[6]
    else:
        # emulate pytroch behavior: return empty tensors
        L = torch.empty(torch.Size([0]))
        U = torch.empty(torch.Size([0]))

    ### Compute the Permutation matrix
    if unpack_pivots:
        # We should return a permutation matrix (2D) for each pivot array (1D)
        # The shape of the final Permutation matrix depends on the shape of the input
        # data and the pivots

        # start with a 2D identity matrix and tile it to the other dims of input data
        identity2d = jnp.identity(n, dtype=jnp.float32)
        tile_shape = list(LU_data.shape)
        tile_shape[-1] = 1
        tile_shape[-2] = 1
        P = jnp.tile(identity2d, tile_shape)

        # closure to be called for each input 2D matrix.
        def _lu_unpack_2d(p: Any, pivot: Any) -> Any:  # pyre-ignore[53]
            _pivot = pivot - 1  # pivots are offset by 1 in jax
            indices = jnp.array([*range(n)], dtype=jnp.int32)

            def update_indices(i: Any, _indices: Any) -> Any:  # pyre-ignore[53]
                tmp = _indices[i]
                _indices = _indices.at[i].set(_indices[_pivot[i]])
                _indices = _indices.at[_pivot[i]].set(tmp)
                return _indices

            indices = jax.lax.fori_loop(0, _pivot.size, update_indices, indices)
            p = p[jnp.array(indices)]
            p = jnp.transpose(p)
            return p

        if len(LU_pivots.shape) == 1:
            # if we are dealing with a simple 2D input and 1D pivot, call the closure directly
            P = _lu_unpack_2d(P, LU_pivots)
        else:
            # We are dealing with >=3D inputs. Flatten inputs to 3D and use vmap to call the
            # closure for each 2D matrix. Finally unflatten the result to match the input data
            # shape.

            # reshape permutation matrix to 3d
            dim_size = jnp.prod(jnp.array(P.shape[:-2]))
            newPshape = (dim_size, P.shape[-2], P.shape[-1])
            reshapedP = P.reshape(newPshape)

            # reshape pivots to 3d
            dim_size = jnp.prod(jnp.array(LU_pivots.shape[:-1]))
            newPivotshape = (dim_size, LU_pivots.shape[-1])
            reshapedPivot = LU_pivots.reshape(newPivotshape)

            # vmap the reshaped 3d tensors
            v_lu_unpack_2d = jax.vmap(_lu_unpack_2d, in_axes=(0, 0))
            unpackedP = v_lu_unpack_2d(reshapedP, reshapedPivot)

            # reshape result back to P's shape
            newRetshape = (
                *P.shape[:-2],
                unpackedP.shape[-2],
                unpackedP.shape[-1],
            )
            P = unpackedP.reshape(newRetshape)
    else:
        # emulate pytroch behavior: return empty tensors
        P = torch.empty(torch.Size([0]))

    return P, L, U


@op(torch.ops.aten.linear)  # pyre-ignore[56]
def linear(input: Any, weight: Any, bias: Any = None) -> Any:
    res = input @ jnp.transpose(weight)
    if bias is not None:
        res += bias
    return res


@op(torch.ops.aten.kthvalue)  # pyre-ignore[56]
def kthvalue(
    input: Any, k: Any, dim: Any = None, keepdim: Any = False, *, out: Any = None
) -> Any:
    if input.ndim == 0:
        return input, jnp.array(0)
    dimension = -1
    if dim is not None:
        dimension = dim
    while dimension < 0:
        dimension = dimension + input.ndim
    values = jax.lax.index_in_dim(
        jnp.partition(input, k - 1, dimension), k - 1, dimension, keepdim
    )
    indices = jax.lax.index_in_dim(
        jnp.argpartition(input, k - 1, dimension).astype("int64"),
        k - 1,
        dimension,
        keepdim,
    )
    return values, indices


@op(torch.ops.aten.take)  # pyre-ignore[56]
def _aten_take(self: Any, index: Any) -> Any:
    return self.flatten()[index]


# func: pad(Tensor self, SymInt[] pad, str mode="constant", float? value=None) -> Tensor
@op(torch.ops.aten.pad)  # pyre-ignore[56]
def _aten_pad(self: Any, pad: Any, mode: Any = "constant", value: Any = None) -> Any:
    if not isinstance(pad, (tuple, list)) or len(pad) % 2 != 0:
        raise ValueError("Padding must be a sequence of even length.")

    num_dims = self.ndim
    if len(pad) > 2 * num_dims:
        raise ValueError(
            f"Padding sequence length ({len(pad)}) exceeds 2 * number of dimensions ({2 * num_dims})."
        )

    # JAX's pad function expects padding for each dimension as a tuple of (low, high)
    # We need to reverse the pad sequence and group them for JAX.
    # pad = [p_l0, p_r0, p_l1, p_r1, ...]
    # becomes ((..., ..., (p_l1, p_r1), (p_l0, p_r0)))
    jax_pad_width = []
    # Iterate in reverse pairs
    for i in range(len(pad) // 2):
        jax_pad_width.append((pad[(2 * i)], pad[(2 * i + 1)]))

    # Pad any leading dimensions with (0, 0) if the pad sequence is shorter
    # than the number of dimensions.
    for _ in range(num_dims - len(pad) // 2):
        jax_pad_width.append((0, 0))

    # Reverse the jax_pad_width list to match the dimension order
    jax_pad_width.reverse()

    if mode == "constant":
        if value is None:
            value = 0.0
        return jnp.pad(
            self,
            pad_width=jax_pad_width,
            mode="constant",
            constant_values=value,
        )
    elif mode == "reflect":
        return jnp.pad(self, pad_width=jax_pad_width, mode="reflect")
    elif mode == "edge":
        return jnp.pad(self, pad_width=jax_pad_width, mode="edge")
    else:
        raise ValueError(
            f"Unsupported padding mode: {mode}. Expected 'constant', 'reflect', or 'edge'."
        )


@op(torch.ops.aten.is_nonzero)  # pyre-ignore[56]
def _aten_is_nonzero(a: Any) -> Any:
    a = jnp.squeeze(a)
    if a.shape == (0,):
        raise RuntimeError("bool value of Tensor with no values is ambiguous")
    if a.ndim != 0:
        raise RuntimeError("bool value of Tensor with more than one value is ambiguous")
    return a.item() != 0


@op(torch.ops.aten.logit)  # pyre-ignore[56]
def _aten_logit(self: jax.Array, eps: float | None = None) -> jax.Array:
    """
    Computes the logit function of the input tensor.

    logit(p) = log(p / (1 - p))

    Args:
      self: Input tensor.
      eps: A small value to clip the input tensor to avoid log(0) or division by zero.
           If None, no clipping is performed.

    Returns:
      A tensor with the logit of each element of the input.
    """
    if eps is not None:
        self = jnp.clip(self, eps, 1.0 - eps)
    res = jnp.log(self / (1.0 - self))
    res = res.astype(mappings.t2j_dtype(torch.get_default_dtype()))
    return res


@op(torch.ops.aten.floor_divide)  # pyre-ignore[56]
def _aten_floor_divide(x: Any, y: Any) -> Any:
    res = jnp.floor_divide(x, y)
    return res


@op(torch.ops.aten._assert_tensor_metadata)
@op(torch.ops.aten._assert_scalar)  # pyre-ignore[56]
def _aten__assert_tensor_metadata(*args: Any, **kwargs: Any) -> Any:
    pass


@op(torch.ops.aten.unfold)  # pyre-ignore[56]
def _aten_unfold(x: Any, dimension: Any, size: Any, step: Any) -> Any:
    ndim = x.ndim if x.ndim > 0 else 1
    shape = x.shape if len(x.shape) > 0 else (1,)
    if dimension < -ndim or dimension >= ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of [{-ndim}, {ndim}], but got {dimension})"
        )
    if size < 0:
        raise RuntimeError(f"size is {size} but must be >= 0")

    if dimension < 0:
        dimension += ndim

    if shape[dimension] < size:
        raise RuntimeError(
            f"RuntimeError: maximum size for tensor at dimension {dimension} is {shape[dimension]} but size is {size}"
        )

    if x.ndim == 0:
        return _aten_slice(_aten_unsqueeze(x, 0), end=size)

    num_windows = (shape[dimension] - size) // step + 1

    # Start indices for each window in the specified dimension
    start_indices = jnp.arange(num_windows) * step

    # Function to extract a single slice
    def slice_fn(start_index: Any) -> Any:
        return jax.lax.dynamic_slice_in_dim(x, start_index, size, axis=dimension)

    # Vectorize the slice_fn over all start_indices
    unfolded_slices = jax.vmap(slice_fn)(start_indices)
    # The shape of unfolded_slices is (num_windows, d0, ..., size, ..., dN-1)
    # where 'size' is at the 'dimension'-th position of the original shape part.

    # We need to move the axes to match torch.unfold's output shape:
    # (d0, ..., d(dim-1), num_windows, d(dim+1), ..., dN-1, size)

    # Move the num_windows axis (axis 0) to the 'dimension' position
    temp = jnp.moveaxis(unfolded_slices, 0, dimension)
    # The shape is now (d0, ..., num_windows, size, ..., dN-1)

    # Move the 'size' axis (now at dimension + 1) to the very end
    return jnp.moveaxis(temp, dimension + 1, -1)


# ============================================================
# Foreach ops (list-of-tensor → list-of-tensor)
# These are used by PyTorch optimizers with foreach=True.
# ============================================================


@op(torch.ops.aten._foreach_add.Scalar)  # pyre-ignore[56]
def _aten_foreach_add_scalar(tensors: Any, scalar: Any) -> Any:
    return [t + scalar for t in tensors]


@op(torch.ops.aten._foreach_add.List)  # pyre-ignore[56]
def _aten_foreach_add_list(tensors1: Any, tensors2: Any, alpha: Any = 1) -> Any:
    return [x + y * alpha for x, y in zip(tensors1, tensors2)]


@op(torch.ops.aten._foreach_sub.Scalar)  # pyre-ignore[56]
def _aten_foreach_sub_scalar(tensors: Any, scalar: Any) -> Any:
    return [t - scalar for t in tensors]


@op(torch.ops.aten._foreach_mul.Scalar)  # pyre-ignore[56]
def _aten_foreach_mul_scalar(tensors: Any, scalar: Any) -> Any:
    return [t * scalar for t in tensors]


@op(torch.ops.aten._foreach_mul.List)  # pyre-ignore[56]
def _aten_foreach_mul_list(tensors1: Any, tensors2: Any) -> Any:
    return [x * y for x, y in zip(tensors1, tensors2)]


@op(torch.ops.aten._foreach_div.Scalar)  # pyre-ignore[56]
def _aten_foreach_div_scalar(tensors: Any, scalar: Any) -> Any:
    return [t / scalar for t in tensors]


@op(torch.ops.aten._foreach_div.List)  # pyre-ignore[56]
def _aten_foreach_div_list(tensors1: Any, tensors2: Any) -> Any:
    return [x / y for x, y in zip(tensors1, tensors2)]


@op(torch.ops.aten._foreach_neg.default)  # pyre-ignore[56]
def _aten_foreach_neg(tensors: Any) -> Any:
    return [-t for t in tensors]


@op(torch.ops.aten._foreach_sqrt.default)  # pyre-ignore[56]
def _aten_foreach_sqrt(tensors: Any) -> Any:
    return [jnp.sqrt(t) for t in tensors]


@op(torch.ops.aten._foreach_reciprocal.default)  # pyre-ignore[56]
def _aten_foreach_reciprocal(tensors: Any) -> Any:
    return [1.0 / t for t in tensors]


@op(torch.ops.aten._foreach_addcdiv.Scalar)  # pyre-ignore[56]
def _aten_foreach_addcdiv_scalar(
    tensors1: Any, tensors2: Any, tensors3: Any, scalar: Any = 1
) -> Any:
    return [t1 + scalar * t2 / t3 for t1, t2, t3 in zip(tensors1, tensors2, tensors3)]


@op(torch.ops.aten._foreach_addcmul.Scalar)  # pyre-ignore[56]
def _aten_foreach_addcmul_scalar(
    tensors1: Any, tensors2: Any, tensors3: Any, scalar: Any = 1
) -> Any:
    return [t1 + scalar * t2 * t3 for t1, t2, t3 in zip(tensors1, tensors2, tensors3)]


@op(torch.ops.aten._foreach_lerp.Scalar)  # pyre-ignore[56]
def _aten_foreach_lerp_scalar(tensors1: Any, tensors2: Any, scalar: Any) -> Any:
    return [t1 + scalar * (t2 - t1) for t1, t2 in zip(tensors1, tensors2)]


@op(torch.ops.aten._foreach_pow.ScalarAndTensor)  # pyre-ignore[56]
def _aten_foreach_pow_scalar_and_tensor(scalar: Any, tensors: Any) -> Any:
    return [scalar**t for t in tensors]


@op(torch.ops.aten._foreach_maximum.List)  # pyre-ignore[56]
def _aten_foreach_maximum_list(tensors1: Any, tensors2: Any) -> Any:
    return [jnp.maximum(t1, t2) for t1, t2 in zip(tensors1, tensors2)]


@op(torch.ops.aten._foreach_clamp_min.Scalar)  # pyre-ignore[56]
def _aten_foreach_clamp_min_scalar(tensors: Any, scalar: Any) -> Any:
    return [jnp.maximum(t, scalar) for t in tensors]


mutation_ops_to_functional: dict[Any, Any] = {
    torch.ops.aten.add_: op_base.InplaceOp(torch.ops.aten.add),
    torch.ops.aten.sub_: op_base.InplaceOp(torch.ops.aten.sub),
    torch.ops.aten.mul_: op_base.InplaceOp(torch.ops.aten.mul),
    torch.ops.aten.div_: op_base.InplaceOp(torch.ops.aten.div),
    torch.ops.aten.pow_: op_base.InplaceOp(torch.ops.aten.pow),
    torch.ops.aten.lt_: op_base.InplaceOp(torch.ops.aten.lt),
    torch.ops.aten.le_: op_base.InplaceOp(torch.ops.aten.le),
    torch.ops.aten.gt_: op_base.InplaceOp(torch.ops.aten.gt),
    torch.ops.aten.ge_: op_base.InplaceOp(torch.ops.aten.ge),
    torch.ops.aten.eq_: op_base.InplaceOp(torch.ops.aten.eq),
    torch.ops.aten.ne_: op_base.InplaceOp(torch.ops.aten.ne),
    torch.ops.aten.bernoulli_: op_base.InplaceOp(torch.ops.aten.bernoulli.p),
    torch.ops.aten.bernoulli_.float: op_base.InplaceOp(
        _aten_bernoulli, is_jax_func=True
    ),
    torch.ops.aten.geometric_: op_base.InplaceOp(torch.ops.aten.geometric),
    torch.ops.aten.normal_: op_base.InplaceOp(torch.ops.aten.normal),
    torch.ops.aten.random_: op_base.InplaceOp(torch.ops.aten.uniform),
    torch.ops.aten.uniform_: op_base.InplaceOp(torch.ops.aten.uniform),
    torch.ops.aten.relu_: op_base.InplaceOp(torch.ops.aten.relu),
    # squeeze_ is expected to change tensor's shape. So replace with new value
    torch.ops.aten.squeeze_: op_base.InplaceOp(torch.ops.aten.squeeze, True),
    torch.ops.aten.sqrt_: op_base.InplaceOp(torch.ops.aten.sqrt),
    torch.ops.aten.clamp_: op_base.InplaceOp(torch.ops.aten.clamp),
    torch.ops.aten.clamp_min_: op_base.InplaceOp(torch.ops.aten.clamp_min),
    torch.ops.aten.sigmoid_: op_base.InplaceOp(torch.ops.aten.sigmoid),
    torch.ops.aten.tanh_: op_base.InplaceOp(torch.ops.aten.tanh),
    torch.ops.aten.ceil_: op_base.InplaceOp(torch.ops.aten.ceil),
    torch.ops.aten.logical_not_: op_base.InplaceOp(torch.ops.aten.logical_not),
    torch.ops.aten.unsqueeze_: op_base.InplaceOp(torch.ops.aten.unsqueeze),
    torch.ops.aten.transpose_: op_base.InplaceOp(torch.ops.aten.transpose),
    torch.ops.aten.log_normal_: op_base.InplaceOp(torch.ops.aten.log_normal),
    torch.ops.aten.scatter_add_: op_base.InplaceOp(torch.ops.aten.scatter_add),
    torch.ops.aten.scatter_reduce_.two: op_base.InplaceOp(
        torch.ops.aten.scatter_reduce
    ),
    torch.ops.aten.scatter_: op_base.InplaceOp(torch.ops.aten.scatter),
    torch.ops.aten.bitwise_not_: op_base.InplaceOp(torch.ops.aten.bitwise_not),
    torch.ops.aten.bitwise_and_: op_base.InplaceOp(torch.ops.aten.bitwise_and),
    torch.ops.aten.bitwise_or_: op_base.InplaceOp(torch.ops.aten.bitwise_or),
    torch.ops.aten.bitwise_xor_: op_base.InplaceOp(torch.ops.aten.bitwise_xor),
    torch.ops.aten.bitwise_left_shift_: op_base.InplaceOp(
        torch.ops.aten.bitwise_left_shift
    ),
    torch.ops.aten.bitwise_right_shift_: op_base.InplaceOp(
        torch.ops.aten.bitwise_right_shift
    ),
    torch.ops.aten.floor_divide_: op_base.InplaceOp(torch.ops.aten.floor_divide),
    torch.ops.aten.remainder_: op_base.InplaceOp(torch.ops.aten.remainder),
    torch.ops.aten.index_put_: op_base.InplaceOp(torch.ops.aten.index_put),
    torch.ops.aten.masked_scatter_: op_base.InplaceOp(torch.ops.aten.masked_scatter),
}

_out_variant_to_functional: dict[Any, Any] = {
    torch.ops.aten.add.out: op_base.OutVariant(torch.ops.aten.add),
    torch.ops.aten.sub.out: op_base.OutVariant(torch.ops.aten.sub),
    torch.ops.aten.mul.out: op_base.OutVariant(torch.ops.aten.mul),
    torch.ops.aten.div.out: op_base.OutVariant(torch.ops.aten.div),
    torch.ops.aten.bernoulli.out: op_base.OutVariant(torch.ops.aten.bernoulli.p),
    torch.ops.aten.geometric.out: op_base.OutVariant(torch.ops.aten.geometric),
    torch.ops.aten.normal.out: op_base.OutVariant(torch.ops.aten.normal),
    torch.ops.aten.random.out: op_base.OutVariant(torch.ops.aten.uniform),
    torch.ops.aten.uniform.out: op_base.OutVariant(torch.ops.aten.uniform),
    torch.ops.aten.relu.out: op_base.OutVariant(torch.ops.aten.relu),
    torch.ops.aten.sqrt.out: op_base.OutVariant(torch.ops.aten.sqrt),
    torch.ops.aten.clamp.out: op_base.OutVariant(torch.ops.aten.clamp),
    torch.ops.aten.clamp_min.out: op_base.OutVariant(torch.ops.aten.clamp_min),
    torch.ops.aten.sigmoid.out: op_base.OutVariant(torch.ops.aten.sigmoid),
    torch.ops.aten.tanh.out: op_base.OutVariant(torch.ops.aten.tanh),
    torch.ops.aten.ceil.out: op_base.OutVariant(torch.ops.aten.ceil),
    torch.ops.aten.logical_not.out: op_base.OutVariant(torch.ops.aten.logical_not),
    torch.ops.aten.log_normal.out: op_base.OutVariant(torch.ops.aten.log_normal),
    torch.ops.aten.scatter_add.out: op_base.OutVariant(torch.ops.aten.scatter_add),
    torch.ops.aten.scatter_reduce.two_out: op_base.OutVariant(
        torch.ops.aten.scatter_reduce.two
    ),
    torch.ops.aten.bitwise_not.out: op_base.OutVariant(torch.ops.aten.bitwise_not),
    torch.ops.aten.floor_divide.out: op_base.OutVariant(torch.ops.aten.floor_divide),
    torch.ops.aten.index_put.out: op_base.OutVariant(torch.ops.aten.index_put),
    torch.ops.aten.masked_scatter.out: op_base.OutVariant(
        torch.ops.aten.masked_scatter
    ),
}

# Note: tuple comparisons work intuitively, e.g. `_jax_version >= (0, 4, 32)`.
_jax_version: tuple[int, ...] = tuple(int(v) for v in jax.version._version.split("."))

mutation_needs_env: set[Any] = {
    torch.ops.aten.bernoulli_,
    torch.ops.aten.bernoulli_.float,
}

for operator, mutation in mutation_ops_to_functional.items():
    ops_registry.register_torch_dispatch_op(
        operator,
        mutation,
        is_jax_function=False,
        is_view_op=True,
        needs_env=(operator in mutation_needs_env),
    )

for operator, definition in _out_variant_to_functional.items():
    ops_registry.register_torch_dispatch_op(
        operator,
        definition,
        is_jax_function=False,
        is_view_op=True,
        needs_env=False,
    )
