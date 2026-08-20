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


import collections
import copy
import functools
from functools import wraps
from inspect import signature
from typing import Any

import jax
import jax.numpy as jnp
import recir.torchax
import torch
from jax import tree_util as pytree
from recir.torchax import tensor, util
from recir.torchax.ops import mappings
from recir.torchax.types import JaxCallable, JaxValue, TorchCallable, TorchValue
from torch.nn.utils import stateless as torch_stateless

try:
    from jax import shard_map as shard_map  # for jax since v0.8.0
except ImportError:
    from jax.experimental.shard_map import shard_map


def extract_all_buffers(
    m: torch.nn.Module,
) -> tuple[dict[str, torch.nn.parameter.Parameter], dict[str, torch.Tensor]]:
    buffers: dict[str, torch.Tensor] = {}
    params: dict[str, torch.nn.parameter.Parameter] = {}

    def extract_one(module: torch.nn.Module, prefix: str) -> None:
        for k in dir(module):
            try:
                v = getattr(module, k)
            except Exception:
                continue
            qual_name = prefix + k
            if isinstance(v, torch.nn.parameter.Parameter) and v.requires_grad:
                params[qual_name] = v
            elif isinstance(v, torch.Tensor):
                buffers[qual_name] = v
        for name, child in module.named_children():
            extract_one(child, prefix + name + ".")

    extract_one(m, "")
    return params, buffers


def set_all_buffers(
    m: torch.nn.Module,
    params: dict[str, torch.nn.parameter.Parameter],
    buffers: dict[str, torch.Tensor],
) -> None:
    def set_one(module: torch.nn.Module, prefix: str) -> None:
        for k in dir(module):
            qual_name = prefix + k
            if (potential_v := buffers.get(qual_name)) is not None:
                setattr(module, k, potential_v)
            elif (potential_v := params.get(qual_name)) is not None:
                setattr(module, k, torch.nn.Parameter(potential_v))
        for name, child in module.named_children():
            set_one(child, prefix + name + ".")

    set_one(m, "")


class JittableModule(torch.nn.Module):
    params: dict[str, Any]
    buffers: dict[str, Any]  # pyre-ignore[15]
    _model: torch.nn.Module
    _jitted: dict[str, Any]
    _extra_jit_args: dict[str, Any]
    _extra_dumped_weights: dict[str, list[str]]

    def __init__(
        self,
        m: torch.nn.Module,
        extra_jit_args: dict[str, Any] | None = None,
        dedup_parameters: bool = True,
    ) -> None:
        if extra_jit_args is None:
            extra_jit_args = {}
        super().__init__()
        self.params, self.buffers = extract_all_buffers(m)
        self._model = m
        self._jitted = {}

        self._extra_jit_args = extra_jit_args

        self._extra_dumped_weights = {}

        if dedup_parameters:
            temp = collections.defaultdict(list)
            for k, v in self.params.items():
                temp[id(v)].append(k)

            for v in temp.values():
                if len(v) > 1:
                    # duplicated weights with different name
                    self._extra_dumped_weights[v[0]] = v[1:]
                    for extra_keys in v[1:]:
                        del self.params[extra_keys]

    @property
    def __class__(self) -> type:  # pyre-ignore[24]
        # Lie about the class type so that
        # isinstance(jittable_module, self._model.__class__) works
        return self._model.__class__

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def functional_call(
        self,
        method_or_name: Any,
        params: dict[str, Any],
        buffers: dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        kwargs = kwargs or {}
        params_copy = copy.copy(params)
        params_copy.update(buffers)
        # reinflate the state dict so there are not any missing keys
        for k, v in self._extra_dumped_weights.items():
            for new_key in v:
                params_copy[new_key] = params_copy[k]

        if isinstance(method_or_name, str):
            method = getattr(self._model, method_or_name)
        else:
            if not callable(method_or_name):
                raise TypeError(
                    f"method_or_name should be a callable or a string, got {type(method_or_name)}"
                )
            method = method_or_name
            args = (self._model,) + args
        with torch_stateless._reparametrize_module(self._model, params_copy):
            res = method(*args, **kwargs)
        return res

    def jittable_call(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        if method_name not in self._jitted:
            jitted = jax_jit(
                functools.partial(self.functional_call, method_name),
                kwargs_for_jax_jit=self._extra_jit_args,
            )

            def jitted_forward(*args: Any, **kwargs: Any) -> Any:  # pyre-ignore[53]
                return jitted(self.params, self.buffers, *args, **kwargs)

            self._jitted[method_name] = jitted_forward
        return self._jitted[method_name](*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.jittable_call("forward", *args, **kwargs)

    # pyre-ignore[14]
    def __getattr__(self, key: str) -> Any:
        if key == "_model":
            return super().__getattr__(key)
        if key in self._jitted:
            return self._jitted[key]
        return getattr(self._model, key)

    def make_jitted(self, key: str) -> None:
        jitted = jax_jit(
            functools.partial(self.functional_call, key),
            kwargs_for_jax_jit=self._extra_jit_args,
        )

        def call(*args: Any, **kwargs: Any) -> Any:  # pyre-ignore[53]
            return jitted(self.params, self.buffers, *args, **kwargs)

        self._jitted[key] = call


def _torch_view(t: JaxValue) -> TorchValue:
    # t is an object from jax land
    # view it as-if it's a torch land object
    if isinstance(t, jax.Array):
        return tensor.Tensor(t, recir.torchax.default_env())
    if isinstance(t, jnp.dtype):
        return mappings.j2t_dtype(t)
    if callable(t):  # t is a JaxCallable
        return functools.partial(call_jax, t)
    # regular types are not changed
    return t


torch_view: Any = functools.partial(pytree.tree_map, _torch_view)


def _jax_view(t: TorchValue) -> JaxValue:
    # t is an object from torch land
    # view it as-if it's a jax land object
    if isinstance(t, torch.Tensor):
        assert isinstance(t, tensor.Tensor) or isinstance(t, tensor.View), type(t)
        return t.jax()  # pyre-ignore[16]
    if isinstance(t, type(torch.int32)):
        return mappings.t2j_dtype(t)

    # torch.nn.Module needs special handling
    if not isinstance(t, torch.nn.Module) and callable(t):  # t is a TorchCallable
        return functools.partial(call_torch, t)
    # regular types are not changed
    return t


jax_view: Any = functools.partial(pytree.tree_map, _jax_view)


def call_jax(
    jax_func: JaxCallable, *args: TorchValue, **kwargs: TorchValue
) -> TorchValue:
    args, kwargs = jax_view((args, kwargs))
    res: JaxValue = jax_func(*args, **kwargs)
    return torch_view(res)


def call_torch(
    torch_func: TorchCallable, *args: JaxValue, **kwargs: JaxValue
) -> JaxValue:
    args, kwargs = torch_view((args, kwargs))
    with recir.torchax.default_env():
        res: TorchValue = torch_func(*args, **kwargs)
    return jax_view(res)


def j2t_autograd(fn: Any, call_jax: Any = call_jax) -> Any:
    """Given a JAX function, returns a PyTorch autograd function implemented with `jax.vjp(fn)`.

    It wraps `fn` with `jax.vjp` to compute both the output and residuals (intermediate
    activations). The wrapped function is then run via `call_jax` and integrated into
    the PyTorch autograd framework by saving the residuals into the context object.
    """

    # NOTE(qihqi): This function cannot be inlined from the callsite
    #  Because if it does, then it won't hit the compilation cache for
    #  call_jax. Call jax uses functions' id as key.
    # It is nested inside j2t_autograd to ensure it gets a unique ID for each
    # wrapped pure function, preventing cache collisions between different pure modules.
    def _jax_forward(fn: Any, other: Any, tree_def: Any, tensors: Any) -> Any:
        """JAX function to compute output and vjp function.

        primals should be a tuple (args, kwargs).
        """
        import jax
        from jax.tree_util import tree_unflatten

        def fn_wrapper(*tensors: Any) -> Any:
            # Reconstruct the original args and kwargs
            flat_inputs = util.merge(tensors, other)  # pyre-ignore[6]
            args, kwargs = tree_unflatten(tree_def, flat_inputs)
            return fn(*args, **kwargs)

        return jax.vjp(fn_wrapper, *tensors)

    def _jax_backward(vjp_spec: Any, saved_tensors: Any, grad_out: Any) -> Any:
        """JAX function to compute input gradients.

        Unflattening `saved_tensors` with `vjp_spec` should restore the original vjp function.
        """
        from jax.tree_util import tree_unflatten

        fun_vjp = tree_unflatten(vjp_spec, saved_tensors)
        return fun_vjp(grad_out)

    @wraps(fn)
    def inner(*args: Any, **kwargs: Any) -> Any:
        from jax.tree_util import tree_flatten

        class JaxFun(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, tree_def: Any, *flat_args_kwargs: Any) -> Any:
                tensors, other = util.partition(
                    # pyrefly: ignore [bad-argument-type]
                    flat_args_kwargs,
                    lambda x: isinstance(x, torch.Tensor),
                )
                # We want the arguments that don't require grads to be closured?

                y, fun_vjp = call_jax(_jax_forward, fn, other, tree_def, tensors)

                # Save necessary information for backward
                # Flatten the vjp function. `vjp_spec` contains a jaxpr for the backward pass.
                # `residuals` contains the tensors needed for the backward pass.`
                residuals, vjp_spec = tree_flatten(fun_vjp)
                ctx.vjp_spec = vjp_spec
                ctx.save_for_backward(*residuals)
                return y

            @staticmethod
            def backward(ctx: Any, *grad_out: Any) -> tuple[Any, ...]:
                assert len(grad_out) > 0
                grad_out = grad_out if len(grad_out) > 1 else grad_out[0]

                input_grads_structured = call_jax(
                    _jax_backward, ctx.vjp_spec, ctx.saved_tensors, grad_out
                )

                # Construct the gradient tuple to be returned.
                # It needs to match the inputs to forward: (tree_def, *flat_inputs)
                # The first gradient (for tree_def) is None.
                # The subsequent gradients correspond to flat_inputs.
                # We need to put a None for inputs that did not require gradients.
                final_grads: list[Any] = [None]
                for needs_grad, grad in zip(
                    ctx.needs_input_grad[1:], input_grads_structured, strict=True
                ):
                    final_grads.append(grad if needs_grad else None)

                return tuple(final_grads)

        sig = signature(fn)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        flat_args_kwargs, tree_def = tree_flatten((bound.args, bound.kwargs))
        y = JaxFun.apply(tree_def, *flat_args_kwargs)  # pyre-ignore[16]
        return y

    return inner


fori_loop: Any = torch_view(jax.lax.fori_loop)


def wrap_jax_jit(
    torch_function: Any,
    jax_jit_func: Any = jax.jit,
    kwargs_for_jax: dict[str, Any] | None = None,
) -> Any:
    kwargs_for_jax = kwargs_for_jax or {}
    jax_func = jax_view(torch_function)
    jitted = jax_jit_func(jax_func, **kwargs_for_jax)
    return torch_view(jitted)


def jax_jit(
    torch_function: Any,
    kwargs_for_jax_jit: dict[str, Any] | None = None,
    fix_for_buffer_donation: bool = False,
) -> Any:
    return wrap_jax_jit(
        torch_function, jax_jit_func=jax.jit, kwargs_for_jax=kwargs_for_jax_jit
    )


def jax_shard_map(
    torch_function: Any, kwargs_for_jax_shard_map: dict[str, Any] | None = None
) -> Any:
    return wrap_jax_jit(
        torch_function, jax_jit_func=shard_map, kwargs_for_jax=kwargs_for_jax_shard_map
    )


def jax_value_and_grad(
    torch_function: Any, kwargs_for_value_and_grad: dict[str, Any] | None = None
) -> Any:
    return wrap_jax_jit(
        torch_function,
        jax_jit_func=jax.value_and_grad,
        kwargs_for_jax=kwargs_for_value_and_grad,
    )


def gradient_checkpoint(
    torch_function: Any, kwargs: dict[str, Any] | None = None
) -> Any:
    return wrap_jax_jit(
        torch_function, jax_jit_func=jax.checkpoint, kwargs_for_jax=kwargs
    )
