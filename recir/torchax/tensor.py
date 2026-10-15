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


from __future__ import annotations

import contextlib
import itertools
import logging
import sys
import threading
from collections.abc import Iterator
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import numpy
import torch
import torch.distributed._functional_collectives
import torch.func
import torch.utils._mode_utils as mode_utils
import torch.utils._python_dispatch as torch_dispatch
import torch.utils._pytree as torch_pytree
from recir.torchax import amp, config
from recir.torchax.ops import mappings, ops_registry
from recir.torchax.view import View
from torch.utils._python_dispatch import _get_current_dispatch_mode_stack


def _resolve_trace_mode_types() -> Tuple[type, ...]:
    """Lazily collect the TorchDispatchMode subclasses that count as
    "trace context".  Some PyTorch versions don't ship every mode, so each
    import is wrapped in try/except.

    Module-level memoization: we call this once at import time below and
    stash the tuple in ``_TRACE_MODE_TYPES`` so the per-dispatch
    ``_torchax_in_trace_context()`` hot path doesn't re-run the imports.
    """
    trace_mode_types: list[type] = []
    try:
        from torch._subclasses.fake_tensor import FakeTensorMode

        trace_mode_types.append(FakeTensorMode)
    except ImportError:
        pass
    try:
        from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode

        trace_mode_types.append(ProxyTorchDispatchMode)
    except ImportError:
        pass
    try:
        from torch._subclasses.functional_tensor import FunctionalTensorMode

        trace_mode_types.append(FunctionalTensorMode)
    except ImportError:
        pass
    return tuple(trace_mode_types)


# Cache the trace-mode types tuple at import time.  This function is invoked
# from every dispatch escape hatch (3 call-sites in this file), so the
# per-call cost must stay minimal — see _torchax_in_trace_context.
_TRACE_MODE_TYPES: Tuple[type, ...] = _resolve_trace_mode_types()


def _torchax_in_trace_context() -> bool:
    """True if any TRACE/COMPILE-related TorchDispatchMode is currently active.

    Specifically checks for FakeTensorMode, ProxyTorchDispatchMode,
    and FunctionalTensorMode.
    Excludes torchax's own Environment (which is pushed as a
    TorchDispatchMode by `TORCHAX_ENABLE_GLOBALLY=1` and would
    otherwise make every eager call look like a trace call).

    Used by the dispatch escape hatches to decide whether
    `mode_utils.no_dispatch()` is safe:
      * Trace context  -> drop no_dispatch (let trace mode record the op)
      * Eager context  -> keep no_dispatch (pass through to raw torch)

    Performance: this is on the per-op dispatch escape hatch path. The
    trace-mode type tuple is resolved once at module import time (see
    ``_TRACE_MODE_TYPES``); per-call cost is now a stack walk + tuple-isinstance.

    See P2341454293 for the bug history that motivated this check.
    """
    if not _TRACE_MODE_TYPES:
        return False
    stack = _get_current_dispatch_mode_stack()
    if not stack:
        return False
    return any(isinstance(m, _TRACE_MODE_TYPES) for m in stack)


logger: logging.Logger = logging.getLogger(__name__)


class OperatorNotFound(Exception):
    pass


@contextlib.contextmanager
def log_nested(env: "Environment", message: str) -> Iterator[None]:
    if env.config.debug_print_each_op:
        print((" " * env._log_nesting_level) + message, file=sys.stderr)
    env._log_nesting_level += 1
    yield
    env._log_nesting_level -= 1


class Tensor(torch.Tensor):
    @staticmethod
    def __new__(
        cls, elem: jax.Array, env: "Environment", requires_grad: bool = False
    ) -> "Tensor":
        dtype = mappings.j2t_dtype(elem.dtype)
        shape = list(elem.shape)
        for i, s in enumerate(shape):
            if not isinstance(s, int):
                shape[i] = 1
        if dtype is None:
            dtype = torch.float32
        # dispatch_keys = torch.DispatchKeySet(torch._C.DispatchKey.PrivateUse1).add(torch._C.DispatchKey.AutogradPrivateUse1)
        if not (dtype.is_floating_point or dtype.is_complex):
            requires_grad = False

        return torch.Tensor._make_wrapper_subclass(
            cls,
            shape,
            dtype=dtype,
            device="meta",  # pyre-ignore[6]
            requires_grad=requires_grad,
        )

    def __init__(
        self, elem: jax.Array, env: "Environment", requires_grad: bool = False
    ) -> None:
        super().__init__()
        self._elem: jax.Array = elem
        self._env: "Environment" = env

    def __str__(self) -> str:
        return f"Tensor({str(type(self._elem))} {str(self._elem)})"

    __repr__ = __str__  # pyre-ignore[4]

    @property
    # pyrefly: ignore [bad-override]
    def shape(self) -> torch.Size:
        return torch.Size(self._elem.shape)

    @property
    # pyrefly: ignore [bad-override]
    def ndim(self) -> int:
        return len(self._elem.shape)

    # pyre-ignore[14]
    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> "Tensor":
        if end_dim == -1:
            end_dim = self.ndim
        new_shape = (
            self._elem.shape[:start_dim] + (-1,) + self._elem.shape[end_dim + 1 :]
        )
        new_elem = jnp.reshape(self._elem, new_shape)
        return Tensor(new_elem, self._env)
        # return torch.reshape(self, new_shape)

    def __setitem__(self, key: Any, val: Any) -> None:
        key, val = self._env.t2j_iso((key, val))
        self._elem = self._elem.at[key].set(val)

    # pyre-ignore[14]
    def type_as(self, other: "Tensor") -> "Tensor":
        self._elem = self._elem.astype(other._elem.dtype)
        return self

    __torch_function__ = torch._C._disabled_torch_function_impl  # pyre-ignore[4]

    @classmethod
    # pyrefly: ignore [bad-override]
    def __torch_dispatch__(
        cls,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        # TODO(hanq): figure out why is dispatch mode not sufficient
        if func == torch.ops._c10d_functional.wait_tensor.default:
            return args[0]._env.dispatch(func, types, args, kwargs)
        if func == torch.ops.prim.device.default:
            return torch.device("privateuseone", 0)
        raise AssertionError(
            "torchax Tensors can only do math within the torchax environment."
            "Please wrap your code with `with torchax.default_env()` or "
            "call torchax.enable_globally() before."
        )

    def detach(self) -> "Tensor":
        return Tensor(jax.lax.stop_gradient(self.jax()), self._env)

    def numpy(self, *, force: bool = False) -> "numpy.ndarray":  # pyre-fixme[11]
        import numpy as np

        return np.array(self._elem)

    def jax(self) -> jax.Array:
        return self._elem

    def torch(self) -> torch.Tensor:
        return self._env.j2t_copy(self.jax())

    @property
    def dtype(self) -> torch.dtype:  # pyre-ignore[11]
        return mappings.j2t_dtype(self._elem.dtype)

    def dim(self) -> int:
        return self.ndim

    @property
    def device(self) -> torch.device:  # pyre-ignore[11]
        return torch.device("jax:0")

    @property
    def jax_device(self) -> Any:
        return self._elem.device

    @property
    # pyrefly: ignore [bad-override]
    def data(self) -> "Tensor":
        logger.warning("In-place to .data modifications still results a copy on TPU")
        return self

    @data.setter
    def data(self, other: "Tensor") -> None:
        if isinstance(other, Tensor):
            self._elem = other._elem

    def apply_jax(self, jax_function: Any, *args: Any, **kwargs: Any) -> Any:
        # Call a jax function on _elem
        res = jax_function(self._elem, *args, **kwargs)
        return self._env.j2t_iso(res)

    def apply_jax_(self, jax_function: Any, *args: Any, **kwargs: Any) -> "Tensor":
        self._elem = jax_function(self._elem, *args, **kwargs)
        return self

    def tolist(self) -> Any:
        return self._elem.tolist()

    def shard_(self, sharding: Any) -> None:
        self.apply_jax_(jax.lax.with_sharding_constraint, sharding)


def debug_accuracy(func: Any, args: Any, kwargs: Any, current_output: Any) -> bool:
    args_torch, kwargs_torch, out_torch = torch_pytree.tree_map_only(
        # pyrefly: ignore [missing-attribute]
        torch.Tensor,
        # pyrefly: ignore [missing-attribute]
        lambda x: x.torch(),
        (args, kwargs, current_output),
    )

    with mode_utils.no_dispatch(), torch._C.DisableTorchFunction():
        if "device" in kwargs_torch:
            kwargs_torch["device"] = "cpu"  # do the torch native for comparison
        expected_out = func(*args_torch, **kwargs_torch)

    flattened_current_out, _ = torch_pytree.tree_flatten(out_torch)
    flattened_expected_out, _ = torch_pytree.tree_flatten(expected_out)

    for ex, real in zip(flattened_expected_out, flattened_current_out, strict=False):
        if isinstance(ex, torch.Tensor) and ex.dtype != real.dtype:
            ex = ex.to(real.dtype)
        try:
            if isinstance(ex, torch.Tensor) and not torch.allclose(
                ex, real, atol=1e-3, equal_nan=True
            ):
                import pdb

                pdb.set_trace()
        except Exception:
            import pdb

            pdb.set_trace()

    return True


def _make_debug_msg(
    is_dispatch: bool, log_args: bool, func: Any, args: Any, kwargs: Any
) -> str:
    def _display(a: Any) -> str:
        if isinstance(a, torch.Tensor):
            return f"Tensor of {type(a)}: {a.dtype}{a.shape}"
        elif isinstance(a, jax.Array):
            return f"Jax Array of {type(a)}: {a.dtype}{a.shape}"
        else:
            return str(a)

    kwargs = kwargs or {}
    title = "DISPATCH" if is_dispatch else "FUNCTION"
    args_msg = "args: " + ",".join(_display(a) for a in args) if log_args else ""
    kwargs_msg = (
        "kwargs: " + ",".join(f"{key}: {_display(a)}" for key, a in kwargs.items())
        if log_args
        else ""
    )
    return f"{title}: {_name_of_func(func)} {args_msg} ~ {kwargs_msg}"


class XLAFunctionMode(torch.overrides.TorchFunctionMode):
    """Context manager that dispatches torch function calls to JAX."""

    env: "Environment"

    def __init__(self, env: "Environment") -> None:
        self.env = env

    def __torch_function__(
        self,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        message = f"FUNCTION: {_name_of_func(func)}"
        if self.env.config.debug_print_each_op_operands:
            message = message + "f"
        message = _make_debug_msg(
            False, self.env.config.debug_print_each_op_operands, func, args, kwargs
        )
        with log_nested(self.env, message):
            try:
                return self.env.dispatch(func, types, args, kwargs)
            except OperatorNotFound:
                pass
            if _name_of_func(func) in (
                "rot90"
            ):  # skip rot90 with k%4==0 due to no change
                if len(args) >= 2 and isinstance(args[1], int):
                    if (args[1]) % 4 == 0:
                        return args[0]
            return func(*args, **(kwargs or {}))


class XLADispatchMode(torch_dispatch.TorchDispatchMode):
    env: "Environment"

    def __init__(self, env: "Environment") -> None:
        self.env = env

    def __torch_dispatch__(
        self,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        message = _make_debug_msg(
            True, self.env.config.debug_print_each_op_operands, func, args, kwargs
        )
        with log_nested(self.env, message):
            if isinstance(func, torch._ops.OpOverloadPacket):
                with self:
                    return func(*args, **kwargs)  # pyre-ignore[32]
            # Only functions under these namespaces will be intercepted
            if func.namespace not in (
                "aten",
                "_c10d_functional",
                "torchvision",
                "xla",
            ):
                return func(*args, **kwargs)  # pyre-ignore[32]
            return self.env.dispatch(func, types, args, kwargs)


def _name_of_func(func: Any) -> str:
    if hasattr(func, "name"):
        return func.name()
    return func.__name__


# Constructors that don't take other tensor as input
TENSOR_CONSTRUCTORS: set[Any] = {
    torch.ones,
    torch.zeros,
    torch.empty,
    torch.empty_strided,
    torch.tensor,
    torch.arange,
    torch.eye,
    torch.randn,
    torch.rand,
    torch.randint,
    torch.full,
    torch.as_tensor,
}

# TODO(wen): use existing types, either from torch or jax
SUPPORTED_JAX_PLATFROM: list[str] = ["cpu", "tpu"]


class RuntimeProperty:
    mesh: Any
    prng: Any
    autocast_dtype: Any

    def __init__(self, mesh: Any, prng: Any, autocast_dtype: Any) -> None:
        self.mesh = mesh
        self.prng = prng
        self.autocast_dtype = autocast_dtype

    def override(self, **kwargs: Any) -> "OverrideProperty":
        return OverrideProperty(self, kwargs)

    def get_and_rotate_prng_key(self) -> Any:
        old_key = self.prng
        new_prng_key, next_key = jax.random.split(old_key)
        self.prng = new_prng_key
        return next_key


class OverrideProperty(RuntimeProperty):
    parent: RuntimeProperty
    _override: dict[str, Any]

    def __init__(self, parent: RuntimeProperty, override: dict[str, Any]) -> None:
        self.parent = parent
        self._override = dict(override)

    def __getattr__(self, name: str) -> Any:
        if name in self._override:
            return self._override[name]
        return getattr(self.parent, name)


class Environment(contextlib.ContextDecorator):
    """This class holds a set of configurations and "globals" needed

    for executing torch program using jax.
    Things included so far:

    op registry
    PRNGKey
    Configs

    Also helper functions to manipulate those.
    """

    _function_mode: XLAFunctionMode
    _dispatch_mode: XLADispatchMode
    _ops: dict[Any, Any]
    _decomps: dict[Any, Any]
    config: config.Configuration
    _log_nesting_level: int
    enabled: bool
    _property: threading.local
    _initial_content: RuntimeProperty

    def __init__(
        self,
        configuration: config.Configuration | None = None,  # pyre-ignore[11]
    ) -> None:
        # Ensure the "jax" device is registered with PyTorch before any
        # Environment usage (deferred from import time).
        import recir.torchax as _torchax_mod

        _torchax_mod._ensure_device_registered()

        self._function_mode = XLAFunctionMode(self)
        self._dispatch_mode = XLADispatchMode(self)

        # name is torch callable
        self._ops = {}
        self._decomps = {}

        self.load_ops()

        _mesh = None
        self.config = configuration or config.Configuration()

        self._log_nesting_level = 0
        self.enabled = False

        autocast_dtype = None

        _prng_key = jax.random.key(torch.initial_seed() % (1 << 63))
        self._property = threading.local()
        self._initial_content = RuntimeProperty(
            mesh=_mesh, prng=_prng_key, autocast_dtype=autocast_dtype
        )

    @property
    def param(self) -> RuntimeProperty:
        if not hasattr(self._property, "content"):
            self._property.content = [self._initial_content]  # pyre-ignore[16]
        return self._property.content[-1]  # pyre-ignore[16]

    def manual_seed(self, key: Any) -> None:
        if isinstance(key, torch.Tensor):
            assert key.ndim == 0, "manual seed can only take scalars"
            assert not key.dtype.is_floating_point, "manual seed can only be integers"

            if isinstance(key, Tensor):
                key = key._elem
            else:
                key = key.item()
        jax_key = jax.random.PRNGKey(key)
        new_prop = self.param.override(prng=jax_key)
        self._property.content.append(new_prop)  # pyre-ignore[16]

    @property
    def prng_key(self) -> Any:
        return self.param.prng

    def _should_use_torchax_tensor(self, device: Any) -> bool:
        if device is None:
            device = torch.get_default_device()

        if isinstance(device, torch.device):
            device = device.type

        if ":" in device:
            device = device.split(":")[0]

        match device:
            case "cpu":
                return False
            case "cuda":
                return self.config.treat_cuda_as_jax_device
            case "jax":
                return True
            case "privateuseone":
                return True
            case "meta":
                return self.enabled
        return False

    def load_ops(self) -> None:
        from recir.torchax.ops import (  # noqa: F401
            jagged_ops,
            jaten,
            jc10d,
            jrandom,
            jtorch,
            jtorchvision_nms,
        )

        for k, v in itertools.chain(*(r.items() for r in ops_registry.ALL_REGISTRIES)):
            self._ops[k] = v

        from recir.torchax.decompositions import DECOMPOSITIONS, MUTABLE_DECOMPOSITION

        for k, v in DECOMPOSITIONS.items():
            if k not in self._decomps:
                self._decomps[k] = ops_registry.Operator(
                    k,
                    v,
                    is_jax_function=False,
                    is_user_defined=False,
                    needs_env=False,
                    is_view_op=k in MUTABLE_DECOMPOSITION,
                )

    def _get_op_or_decomp(self, func: Any) -> Any:
        def _get_from_dict(op_dict: dict[Any, Any], op: Any) -> Any:
            op = op_dict.get(func)
            if op is None and isinstance(func, torch._ops.OpOverloadPacket):
                op = op_dict.get(func.default)
            if op is None and isinstance(func, torch._ops.OpOverload):
                op = op_dict.get(func.overloadpacket)
            return op

        op = _get_from_dict(self._ops, func)

        if op is None:
            # Registered AFTER this Environment was built. ``load_ops()``
            # snapshots the registry into ``self._ops`` at construction, so an
            # op registered later is invisible to it -- and lazily imported
            # modules are the normal way to register one, precisely so that
            # importing them is opt-in
            for registry in ops_registry.ALL_REGISTRIES:
                op = _get_from_dict(registry, func)
                if op is not None:
                    self._ops[func] = op
                    break

        if op is None:
            # fallback to decompose
            op = _get_from_dict(self._decomps, func)

        if op is None:
            raise OperatorNotFound(
                f"Operator with name {_name_of_func(func)} has no lowering"
            )

        return op

    def _is_same_device(self, the_tensor: Any, new_device: Any) -> bool:
        if new_device is None:
            return True
        if the_tensor.device.type == "jax":
            if new_device == "meta":
                return True
            # During jax.jit tracing, FX graphs traced on CPU carry
            # device='cpu' annotations. The underlying JAX tracers cannot be
            # materialised (numpy.asarray would raise
            # TracerArrayConversionError), so treat the move as a no-op.
            if new_device == "cpu" and isinstance(
                getattr(the_tensor, "_elem", None), jax.core.Tracer
            ):
                return True
            # When treat_cuda_as_jax_device is set, jax→cuda is a no-op
            # (both live on the same accelerator). Without this, _to_copy
            # would call j2t_copy which fails on JAX tracers.
            if new_device == "cuda" and self.config.treat_cuda_as_jax_device:
                return True
        if the_tensor.device.type != new_device:
            if the_tensor.device.type == "cuda":
                return self.config.treat_cuda_as_jax_device
            return False
        return True

    def _to_copy(self, the_tensor: Any, new_dtype: Any, new_device: Any) -> Any:
        if isinstance(the_tensor, View):
            the_tensor = the_tensor.torch()
        if isinstance(new_device, torch.device):
            new_device = new_device.type
        res = the_tensor
        if not self._is_same_device(the_tensor, new_device):
            if isinstance(the_tensor, Tensor):
                torch_tensor = self.j2t_copy(the_tensor._elem)
                with torch._C.DisableTorchFunction():
                    if _torchax_in_trace_context():
                        return torch_tensor.to(device=new_device, dtype=new_dtype)
                    with mode_utils.no_dispatch():
                        return torch_tensor.to(device=new_device, dtype=new_dtype)
            else:
                arr = self.t2j_copy(the_tensor)
                res = Tensor(arr, self, the_tensor.requires_grad)

        if new_dtype is not None and new_dtype != res.dtype:
            if isinstance(res, Tensor):
                res = res.apply_jax(jnp.astype, mappings.t2j_dtype(new_dtype))
            else:
                with torch._C.DisableTorchFunction():
                    if _torchax_in_trace_context():
                        return res.to(device=new_device, dtype=new_dtype)
                    with mode_utils.no_dispatch():
                        return res.to(device=new_device, dtype=new_dtype)
        return res

    def get_and_rotate_prng_key(self, generator: torch.Generator | None = None) -> Any:
        if generator is not None:
            return jax.random.PRNGKey(generator.initial_seed() % (2**63))
        return self.param.get_and_rotate_prng_key()

    def _handle_tensor_constructor(self, func: Any, args: Any, kwargs: Any) -> Any:
        device = kwargs.get("device")
        if self._should_use_torchax_tensor(device):
            # don't set default device, let caller set it
            requires_grad = kwargs.get("requires_grad", False)
            op = self._get_op_or_decomp(func)
            if op.needs_env:
                kwargs["env"] = self
            if op.is_jax_function:
                (args, kwargs) = self.t2j_iso((args, kwargs))
            res = op.func(*args, **kwargs)
            if isinstance(res, jax.Array):
                res = Tensor(res, self, requires_grad)
            return res
        else:
            with torch._C.DisableTorchFunction():
                if _torchax_in_trace_context():
                    return func(*args, **kwargs)
                with mode_utils.no_dispatch():
                    return func(*args, **kwargs)

    def _torch_Tensor_to(self, args: Any, kwargs: Any) -> Any:
        the_tensor = args[0]
        args = args[1:]
        if len(args) >= 1 and isinstance(args[0], torch.Tensor):
            dtype = args[0].dtype
            device = args[0].device
            return self._to_copy(the_tensor, dtype, device)
        device = kwargs.get("device")
        dtype = kwargs.get("dtype")
        # args like pin_memory etc that we will ignore
        args = list(filter(lambda x: not isinstance(x, bool), args))
        if len(args) >= 2:
            device, dtype, *_ = args
        elif len(args) == 1 and isinstance(args[0], torch.dtype):
            dtype = args[0]
        elif len(args) == 1:
            device = args[0]
        return self._to_copy(the_tensor, dtype, device)

    def dispatch(self, func: Any, types: Any, args: Any, kwargs: Any) -> Any:
        kwargs = kwargs or {}
        if func in TENSOR_CONSTRUCTORS:
            return self._handle_tensor_constructor(func, args, kwargs)
        if func in (
            torch.Tensor.to,
            torch.ops.aten.lift_fresh.default,
            torch.ops.aten._to_copy,
            torch.ops.aten._to_copy.default,
        ):
            return self._torch_Tensor_to(args, kwargs)

        # If the func doesn't act on Tensor, and is not a tensor constructor,
        # We should skip and let torch handle it.

        tensor_args = [
            t for t in torch_pytree.tree_flatten(args)[0] if isinstance(t, torch.Tensor)
        ]

        def is_not_torchax_tensor(x: Any) -> bool:
            return not isinstance(x, Tensor) and not isinstance(x, View)

        if tensor_args and all(is_not_torchax_tensor(t) for t in tensor_args):
            res = func(*args, **kwargs)
            return res

        with jax.named_scope(_name_of_func(func)):
            op = self._get_op_or_decomp(func)

            old_args, old_kwargs = args, kwargs
            with self._dispatch_mode:
                args, kwargs = torch_pytree.tree_map_only(
                    torch.distributed._functional_collectives.AsyncCollectiveTensor,
                    torch.distributed._functional_collectives.wait_tensor,
                    (args, kwargs),
                )

            try:
                if not op.is_view_op:
                    args, kwargs = self.v2t_iso((args, kwargs))

                with self:
                    if self.param.autocast_dtype is not None:
                        autocast_policy = amp.autocast_policy.get(func)
                        if autocast_policy is not None:
                            args, kwargs = amp.execute_policy(
                                autocast_policy, args, kwargs, self.param.autocast_dtype
                            )

                if op.is_jax_function:
                    args, kwargs = self.t2j_iso((args, kwargs))
            except AssertionError:
                if self.config.debug_mixed_tensor:
                    breakpoint()
                else:
                    raise

            if op.needs_env:
                kwargs["env"] = self

            if op.is_jax_function:
                res = op.func(*args, **kwargs)
            else:
                # enable dispatch mode because this op could be a composite autograd op
                # meaning, it will decompose in C++
                with self._dispatch_mode:
                    res = op.func(*args, **kwargs)

            if op.is_jax_function:
                res = self.j2t_iso(res)

            if self.config.force_materialize_views and isinstance(res, View):
                res = res.torch()

            if self.config.debug_accuracy_for_each_op:
                debug_accuracy(func, old_args, old_kwargs, res)
            return res

    def enable_torch_modes(self) -> None:
        self._dispatch_mode.__enter__()
        self._function_mode.__enter__()
        self.enabled = True

    def disable_torch_modes(self, *exc: Any) -> None:
        if not exc:
            exc = (None, None, None)
        self._function_mode.__exit__(*exc)
        self._dispatch_mode.__exit__(*exc)
        self.enabled = False

    def __enter__(self) -> "Environment":
        self.enable_torch_modes()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.disable_torch_modes(*exc)

    def _move_one_value(self, val: Any) -> Any:
        if isinstance(val, torch.nn.Module):
            with self:
                return val.to("jax")
        if isinstance(val, Tensor):
            return val
        if isinstance(val, torch.Tensor):
            return Tensor(self.t2j_copy(val), self)
        return val

    def to_xla(self, torchvalues: Any) -> Any:
        # tensors are torch.Tensors (not XLATensor)
        res = torch_pytree.tree_map(self._move_one_value, torchvalues)
        return res

    def t2j_iso(self, torchtensors: Any) -> Any:
        """Convert torchax Tensor to jax array.

        This function will not copy, will just unwrap the inner jax array out.
        Note: iso is short for "isomorphic"
        """

        def to_jax(x: Any) -> Any:
            if (
                self.config.allow_mixed_math_with_scalar_tensor
                and not isinstance(x, Tensor)
                and not isinstance(x, View)
            ):
                if x.squeeze().ndim == 0:
                    return x.item()
            if isinstance(
                x, torch.distributed._functional_collectives.AsyncCollectiveTensor
            ):
                x = x.wait()
            assert isinstance(x, Tensor) or isinstance(x, View), (
                f"Expect a Tensor or a View but got {type(x)}; usually this means there is a mixed math between XLATensor and torch.Tensor"
            )
            return x.jax()

        res = torch_pytree.tree_map_only(torch.Tensor, to_jax, torchtensors)
        return res

    def v2t_iso(self, views: Any) -> Any:
        def to_tensor(x: Any) -> Any:
            if isinstance(x, View):
                return x.torch()
            return x

        res = torch_pytree.tree_map_only(View, to_tensor, views)
        return res

    def j2t_iso(self, jaxarray: Any) -> Any:
        """Convert jax array to torchax Tensor.

        This function will not copy, will just wrap the jax array with a torchax Tensor
        Note: iso is short for "isomorphic"
        """
        return torch_pytree.tree_map_only(
            jax.Array, lambda x: Tensor(x, self), jaxarray
        )

    def j2t_copy(self, args: Any) -> Any:
        """Convert torch.Tensor in cpu to a jax array

        This might involves copying the data (depending if dlpack is enabled)
        """
        return torch_pytree.tree_map_only(
            jax.Array,
            lambda x: mappings.j2t(x, self.config.use_dlpack_for_data_conversion),
            args,
        )

    def t2j_copy(self, args: Any) -> Any:
        """Convert jax array to torch.Tensor in cpu.

        This might involves copying the data (depending if dlpack is enabled)
        """
        return torch_pytree.tree_map_only(
            torch.Tensor,
            lambda x: mappings.t2j(x, self.config.use_dlpack_for_data_conversion),
            args,
        )

    def override_op_definition(
        self, op_to_override: Any, op_impl: Any, is_view_op: bool = False
    ) -> None:
        self._ops[op_to_override] = ops_registry.Operator(
            op_to_override,
            op_impl,
            is_jax_function=False,
            is_user_defined=True,
            needs_env=False,
            is_view_op=is_view_op,
        )

    @contextlib.contextmanager
    def override_property(self, **kwargs: Any) -> Iterator[None]:
        # try/finally, not a bare yield: an exception raised while the override
        # is live must still pop it. Under a jax trace the override holds
        # tracers, and a stranded one turns every LATER step into an
        # UnexpectedTracerError raised from an unrelated call site.
        new_prop = self.param.override(**kwargs)
        self._property.content.append(new_prop)  # pyre-ignore[16]
        try:
            yield
        finally:
            self._property.content.pop()  # pyre-ignore[16]
