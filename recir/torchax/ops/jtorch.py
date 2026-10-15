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


"""Tensor constructor overrides"""

import collections.abc
import functools
import math
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import recir.torchax.tensor
import torch
import torch.utils._pytree as pytree
from recir.torchax.ops import jaten, jimage, mappings, op_base
from recir.torchax.ops.ops_registry import register_torch_function_op
from recir.torchax.view import NarrowInfo, View


def register_function(torch_func: Any, **kwargs: Any) -> Any:
    return functools.partial(register_torch_function_op, torch_func, **kwargs)


@register_function(torch.as_tensor, is_jax_function=False, needs_env=True)
@op_base.convert_dtype(use_default_dtype=False)  # Attempt to infer type from elements
def _as_tensor(
    data: Any, dtype: Any = None, device: Any = None, env: Any = None
) -> Any:
    if isinstance(data, torch.Tensor):
        return env._to_copy(data, dtype, device)
    if isinstance(data, np.ndarray):
        jax_res = jnp.asarray(data)
    else:
        jax_res = _tensor(data, dtype=dtype)
    return recir.torchax.tensor.Tensor(jax_res, env)


@register_function(torch.tensor)
@op_base.convert_dtype(use_default_dtype=False)  # Attempt to infer type from elements
def _tensor(data: Any, *, dtype: Any = None, **kwargs: Any) -> Any:
    python_types_to_torch_types = {
        bool: jnp.bool,
        int: jnp.int64,
        float: jnp.float32,
        complex: jnp.complex64,
    }
    if not dtype:
        leaves = jax.tree_util.tree_leaves(data)
        if len(leaves) > 0:
            dtype = python_types_to_torch_types.get(type(leaves[0]))

    return jnp.array(data, dtype=dtype or mappings.t2j_dtype(torch.get_default_dtype()))


@register_function(torch.allclose)
def _aten_allclose(
    input: Any,
    other: Any,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> Any:
    return jnp.allclose(input, other, rtol, atol, equal_nan)


@register_function(torch.angle)
def _torch_angle(input: Any) -> Any:
    if input.dtype.name == "int64":
        input = input.astype(jnp.dtype("float32"))
    return jnp.angle(input)


@register_function(torch.argsort)
def _torch_argsort(
    input: Any, dim: int = -1, descending: bool = False, stable: bool = False
) -> Any:
    expanded = False
    if input.ndim == 0:
        # for self of rank 0:
        # torch.any(x, 0), torch.any(x, -1) works;
        # torch.any(x, 1) throws out of bounds, so it's
        # behavior is the same as a jnp array of rank 1
        expanded = True
        input = jnp.expand_dims(input, 0)
    res = jnp.argsort(input, axis=dim, descending=descending, stable=stable)
    if expanded:
        res = res.squeeze()
    return res


@register_function(torch.diag)
def _diag(input: Any, diagonal: int = 0) -> Any:
    return jnp.diag(input, k=diagonal)


@register_function(torch.einsum)
@register_function(torch.ops.aten.einsum)  # pyre-ignore[56]
def _einsum(equation: Any, *operands: Any) -> Any:
    def get_params(*a: Any) -> Any:
        inner_list = a[0]
        if not isinstance(inner_list, jax.Array):
            if len(inner_list) == 1:
                A = inner_list
                return A
            elif len(inner_list) == 2:
                A, B = inner_list
                return A, B
        return operands

    assert isinstance(equation, str), "Only accept str equation"
    filtered_operands = get_params(*operands)
    return jnp.einsum(equation, *filtered_operands)


def _sdpa_reference(
    query: Any,
    key: Any,
    value: Any,
    attn_mask: Any = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Any = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(
            diagonal=0
        )
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    if enable_gqa:
        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    if dropout_p > 0:
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


@register_function(torch.nn.functional.one_hot)
def one_hot(tensor: Any, num_classes: int = -1) -> Any:
    if num_classes == -1:
        # pyrefly: ignore [bad-assignment]
        num_classes = jnp.max(tensor) + 1
    return jax.nn.one_hot(tensor, num_classes).astype(jnp.int64)  # pyre-ignore[6]


@register_function(torch.nn.functional.pad)
def pad(tensor: Any, pad: Any, mode: str = "constant", value: Any = None) -> Any:
    # For padding modes that have different names between Torch and NumPy, this
    # dict provides a Torch-to-NumPy translation. Any string not in this dict will
    # be passed through as-is.
    MODE_NAME_TRANSLATION = {
        "circular": "wrap",
        "replicate": "edge",
    }

    numpy_mode = MODE_NAME_TRANSLATION.get(mode, mode)

    num_prefix_dims = tensor.ndim - len(pad) // 2

    numpy_pad_width = [(0, 0)] * num_prefix_dims
    nd_slice = [slice(None)] * num_prefix_dims

    for i in range(len(pad) - 2, -1, -2):
        pad_start, pad_end = pad[i : i + 2]
        slice_start, slice_end = None, None

        if pad_start < 0:
            slice_start = -pad_start
            pad_start = 0

        if pad_end < 0:
            slice_end = pad_end
            pad_end = 0

        numpy_pad_width.append((pad_start, pad_end))
        nd_slice.append(slice(slice_start, slice_end))

    nd_slice = tuple(nd_slice)

    # `jax.numpy.pad` complains if we provide an irrelevant `constant_values` arg,
    # even if the value we pass in is `None`. (It treats `None` as `nan`.)
    kwargs: dict[str, Any] = {}
    if mode == "constant" and value is not None:
        kwargs["constant_values"] = value

    # The "replicate" mode pads first and then slices, whereas the "circular" mode
    # slices first and then pads. The latter approach deals with smaller tensors,
    # so we default to that option in modes where the order of operations doesn't
    # affect the result.
    if mode == "replicate":
        return jnp.pad(tensor, numpy_pad_width, mode=numpy_mode, **kwargs)[nd_slice]
    else:
        return jnp.pad(tensor[nd_slice], numpy_pad_width, mode=numpy_mode, **kwargs)


@register_function(
    torch.nn.functional.scaled_dot_product_attention,
    is_jax_function=False,
    needs_env=True,
)
# pyre-ignore[56]
@register_function(
    torch.ops.aten.scaled_dot_product_attention,
    is_jax_function=False,
    needs_env=True,
)
def scaled_dot_product_attention(
    query: Any,
    key: Any,
    value: Any,
    attn_mask: Any = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Any = None,
    enable_gqa: bool = False,
    env: Any = None,
) -> torch.Tensor:
    return _sdpa_reference(
        query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa
    )


@register_function(torch.Tensor.__getitem__, is_jax_function=False, is_view_op=True)
def getitem(self: Any, indexes: Any) -> Any:
    if isinstance(indexes, list) and isinstance(indexes[0], int):
        # list of int, i.e. x[[1, 2]] NOT x[1, 2] (the second would be tuple of int)
        indexes = (indexes,)
    elif isinstance(indexes, list):
        indexes = tuple(indexes)

    def is_narrow_slicing() -> bool:
        tensor_free = not pytree.tree_any(
            lambda x: isinstance(x, torch.Tensor) or isinstance(x, jax.Array),
            indexes,
        )
        list_free = not isinstance(indexes, tuple) or all(
            False if isinstance(x, list) else True for x in indexes
        )
        return tensor_free and list_free

    if is_narrow_slicing():
        return View(self, view_info=NarrowInfo(indexes), env=self._env)

    indexes = self._env.t2j_iso(indexes)
    return recir.torchax.tensor.Tensor(self._elem[indexes], self._env)


@register_function(torch.corrcoef)
def _corrcoef(x: Any) -> Any:
    if x.dtype.name == "int64":
        return jnp.corrcoef(x).astype(jnp.float32)
    return jnp.corrcoef(x)


@register_function(torch.sparse.mm, is_jax_function=False)
def _sparse_mm(mat1: Any, mat2: Any, reduce: str = "sum") -> Any:
    return torch.mm(mat1, mat2)


@register_function(torch.isclose)
def _aten_isclose(
    input: Any,
    other: Any,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> Any:
    return jnp.isclose(input, other, rtol, atol, equal_nan)


@register_function(torch.linalg.det)
def linalg_det(input: Any) -> Any:
    return jnp.linalg.det(input)


@register_function(torch.ones)
def _ones(*size: int, dtype: Any = None, **kwargs: Any) -> Any:
    if len(size) == 1 and isinstance(size[0], collections.abc.Iterable):
        size = size[0]  # pyre-fixme[9]
    if "size" in kwargs:
        size = kwargs["size"]
    return jaten._ones(size, dtype=dtype)


@register_function(torch.zeros, is_jax_function=True)
def _zeros(*size: int, dtype: Any = None, **kwargs: Any) -> Any:
    if len(size) == 1 and isinstance(size[0], collections.abc.Iterable):
        size = size[0]  # pyre-fixme[9]
    if "size" in kwargs:
        size = kwargs["size"]
    return jaten._zeros(size, dtype=dtype)


@register_function(torch.eye)
@op_base.convert_dtype()
def _eye(n: int, m: int | None = None, *, dtype: Any = None, **kwargs: Any) -> Any:
    return jnp.eye(n, m, dtype=dtype)


@register_function(torch.full)
@op_base.convert_dtype(use_default_dtype=False)
def _full(
    size: Sequence[int], fill_value: Any, *, dtype: Any = None, **kwargs: Any
) -> Any:
    # TODO: handle torch.Size
    return jnp.full(size, fill_value, dtype=dtype)


@register_function(torch.empty)
@op_base.convert_dtype()
def empty(*size: Sequence[int], dtype: Any = None, **kwargs: Any) -> Any:
    if len(size) == 1 and isinstance(size[0], collections.abc.Iterable):
        size = size[0]  # pyre-fixme[9]
    if "size" in kwargs:
        size = kwargs["size"]
    return jnp.empty(size, dtype=dtype)


@register_function(torch.arange, is_jax_function=True)
def arange(
    start: Any,
    end: Any = None,
    step: Any = None,
    out: Any = None,
    dtype: Any = None,
    layout: Any = torch.strided,
    device: Any = None,
    requires_grad: bool = False,
    pin_memory: Any = None,
) -> Any:
    if end is None:
        end = start
        start = 0
    if step is None:
        step = 1
    return jaten._aten_arange(start, end, step, dtype=dtype)


@register_function(torch.empty_strided, is_jax_function=True)
def empty_strided(
    size: Any,
    stride: Any,
    *,
    dtype: Any = None,
    layout: Any = None,
    device: Any = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> Any:
    return empty(size, dtype=dtype, requires_grad=requires_grad)


@register_function(torch.unravel_index)
def unravel_index(indices: Any, shape: Any) -> Any:
    return jnp.unravel_index(indices, shape)


@register_function(torch.rand, is_jax_function=True, needs_env=True)
def rand(*size: Any, **kwargs: Any) -> Any:
    if len(size) == 1 and isinstance(size[0], collections.abc.Iterable):
        # pyrefly: ignore [bad-assignment]
        size = size[0]
    if "size" in kwargs:
        size = kwargs["size"]
    return jaten._rand(size, **kwargs)


@register_function(torch.randn, is_jax_function=True, needs_env=True)
def randn(
    *size: Any,
    generator: Any = None,
    out: Any = None,
    dtype: Any = None,
    layout: Any = torch.strided,
    device: Any = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
    env: Any = None,
) -> Any:
    if len(size) == 1 and isinstance(size[0], collections.abc.Iterable):
        # pyrefly: ignore [bad-assignment]
        size = size[0]
    return jaten._aten_randn(size, generator=generator, dtype=dtype, env=env)


@register_function(torch.randint, is_jax_function=False, needs_env=True)
def randint(*args: Any, **kwargs: Any) -> Any:
    return jaten._aten_randint(*args, **kwargs)


@register_function(torch.logdet)
def logdet(input: Any) -> Any:
    _, logabsdet = jaten._aten__linalg_slogdet(input)
    return logabsdet


@register_function(torch.linalg.slogdet)
def linalg_slogdet(input: Any) -> Any:
    sign, logabsdet = jaten._aten__linalg_slogdet(input)
    return torch.return_types.slogdet((sign, logabsdet))  # pyre-ignore[16]


@register_function(torch.tensor_split)
def tensor_split(input: Any, indices_or_sections: Any, dim: int = 0) -> Any:
    return jnp.array_split(input, indices_or_sections, axis=dim)


@register_function(torch.linalg.solve)
def linalg_solve(a: Any, b: Any) -> Any:
    res, _ = jaten._aten__linalg_solve_ex(a, b)
    return res


@register_function(torch.linalg.solve_ex)
def linalg_solve_ex(a: Any, b: Any) -> Any:
    res, info = jaten._aten__linalg_solve_ex(a, b)
    return res, info


@register_function(torch.linalg.svd)
def linalg_svd(a: Any, full_matrices: bool = True) -> Any:
    return jaten._aten__linalg_svd(a, full_matrices=full_matrices)


@register_function(torch.linalg.matrix_power)
def matrix_power(A: Any, n: Any, *, out: Any = None) -> Any:
    return jnp.linalg.matrix_power(A, n)


@register_function(torch.svd)
def svd(a: Any, some: bool = True, compute_uv: bool = True) -> Any:
    if not compute_uv:
        S = jaten._aten__linalg_svd(a, full_matrices=False)[1]
        U = jnp.zeros((a.shape[-2], a.shape[-2]), dtype=a.dtype)
        V = jnp.zeros((a.shape[-1], a.shape[-1]), dtype=a.dtype)
        return U, S, V
    U, S, V = jaten._aten__linalg_svd(a, full_matrices=not some)
    return U, S, jnp.matrix_transpose(V)


@register_function(torch.cdist)
def _cdist(
    x1: Any,
    x2: Any,
    p: float = 2.0,
    compute_mode: str = "use_mm_for_euclid_dist_if_necessary",
) -> Any:
    return jaten._aten_cdist(x1, x2, p, compute_mode)


@register_function(torch.lu)
def lu(A: Any, **kwargs: Any) -> Any:
    lu, pivots, _ = jax.lax.linalg.lu(A)
    # JAX pivots are offset by 1 compared to torch
    _pivots = pivots + 1
    info_shape = pivots.shape[:-1]
    info = jnp.zeros(info_shape, dtype=mappings.t2j_dtype(torch.int32))
    if kwargs["get_infos"]:
        return lu, _pivots, info
    return lu, _pivots


@register_function(torch.lu_solve)
def lu_solve(b: Any, LU_data: Any, LU_pivots: Any, **kwargs: Any) -> Any:
    # JAX pivots are offset by 1 compared to torch
    _pivots = LU_pivots - 1
    x = jax.scipy.linalg.lu_solve((LU_data, _pivots), b)
    return x


@register_function(torch.linalg.tensorsolve)
def linalg_tensorsolve(A: Any, b: Any, dims: Any = None) -> Any:
    # examples:
    # A = torch.randn(2, 3, 6), b = torch.randn(3, 2)
    # A = torch.randn(2, 3, 6), b = torch.randn(2, 3) -> torch.Size([3, 6])
    # A = torch.randn(9, 2, 6, 3) b = torch.randn(6, 3) -> torch.Size([6, 3])
    # A = torch.randn(9, 2, 3, 6) b = torch.randn(6, 3) -> torch.Size([3, 6])
    # A = torch.randn(18, 6, 3) b = torch.randn(18) -> torch.Size([6, 3])
    # A = torch.randn(3, 8, 4, 6) b = torch.randn(4, 6) -> torch.Size([4,6])
    # A = torch.randn(3, 8, 1, 2, 2, 6) b = torch.randn(3, 4, 2) -> torch.Size([2, 2, 6])

    # torch allows b to be shaped differently.
    # especially when axes are moved using dims.
    # ValueError: After moving axes to end, leading shape of a must match shape of b. got a.shape=(3, 2, 6), b.shape=(2, 3)
    # So we are handling the moveaxis and forcing b's shape to match what jax expects
    if dims is not None:
        A = jnp.moveaxis(A, dims, len(dims) * (A.ndim - 1,))
        dims = None
    if A.shape[: b.ndim] != b.shape:
        b = jnp.reshape(b, A.shape[: b.ndim])
    return jnp.linalg.tensorsolve(A, b, axes=dims)


@register_function(torch.nn.functional.linear)
def functional_linear(self: Any, weights: Any, bias: Any = None) -> Any:
    res = jnp.einsum("...a,ba->...b", self, weights)
    if bias is not None:
        res += bias
    return res


@register_function(torch.nn.functional.interpolate)
def functional_interpolate(
    input: Any,
    size: tuple[int, int],
    scale_factor: float | None,
    mode: str,
    align_corners: bool,
    recompute_scale_factor: bool,
    antialias: bool,
) -> Any:
    supported_methods = (
        "nearest",
        "linear",
        "bilinear",
        "trilinear",
        "cubic",
        "bicubic",
        "tricubic",
        "lanczos3",
        "lanczos5",
    )
    is_jax_supported = mode in supported_methods
    if not is_jax_supported:
        raise recir.torchax.tensor.OperatorNotFound(
            f"JAX does not support interpolation mode: {mode}. Supported modes are: {supported_methods}"
        )
    # None check
    antialias = antialias or False
    align_corners = align_corners or False

    if mode in ("cubic", "bicubic", "tricubic") and not antialias and size is not None:
        return jimage.interpolate_bicubic_no_aa(
            input,
            size[0],
            size[1],
            align_corners,
        )
    else:
        # fallback
        raise recir.torchax.tensor.OperatorNotFound(
            f"JAX does not support interpolation mode: {mode}. Supported modes are: {supported_methods}"
        )


@register_function(torch.Tensor.repeat_interleave)
def torch_Tensor_repeat_interleave(
    self: Any, repeats: Any, dim: Any = None, *, output_size: Any = None
) -> Any:
    return jnp.repeat(self, repeats, axis=dim, total_repeat_length=output_size)


@register_function(torch.nn.functional.max_pool2d)
def _functional_max_pool2d(
    input: Any,
    kernel_size: Any,
    stride: Any = None,
    padding: Any = 0,
    dilation: Any = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
) -> Any:
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if stride is None:
        stride = kernel_size
    if isinstance(stride, int):
        stride = (stride, stride)

    return jaten.max_pool(
        input,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        with_index=return_indices,
    )
