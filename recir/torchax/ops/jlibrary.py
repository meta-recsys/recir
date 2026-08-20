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


"""The `jlibrary` module has functions which help to preserve torch.library ops
during export. This includes aten ops, and custom operations.
"""

from typing import Any

import jax
import recir.torchax
import torch
from recir.torchax.ops import jaten


def _jit_composite_impl(composite_name: str, jaxpr_impl: Any, **jit_args: Any) -> Any:
    """Wrap a jaxpr in a jitted function with the proper composite name
    TODO: Wrap JIT in a `stablehlo.composite` op, instead of generating a call op.
    """

    def composite_impl(*args: Any) -> Any:
        return jaxpr_impl(*args)

    composite_impl.__name__ = composite_name  # pyre-ignore[16]
    composite_impl.__qualname__ = composite_name
    return jax.jit(composite_impl, **jit_args)


def register_jax_composite(
    composite_name: str, impl: Any, *ops: Any, **jit_args: Any
) -> None:
    """Register a composite using a JAX implementation.
      composite_name - The name of the library op to use in the exported composite
      impl           - A JAX lowering for the library operation
      *ops           - Variadic torch.ops to lower using `impl`.
      **jit_args     - Additional parameters to forward to JAX jit.

    This is used to register custom lowerings with an explicit jaxpr
    implementation, such as preserving a specific aten op using a jaten impl.

    For custom torch op registration with a decomposition written in torch,
    use `register_torch_composite`.

    For jit params and troubleshooting see:
    https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html
    """

    @jaten.op(*ops)
    def _composite_impl(*args: Any) -> Any:
        return _jit_composite_impl(composite_name, impl, **jit_args)(*args)


def register_torch_composite(
    composite_name: str, impl: Any, *ops: Any, **jit_args: Any
) -> None:
    """Register a torch decomposition as a composite.
    This is useful for registerring custom torch op libraries as composite ops.

    The `impl` can be the `@impl` used to define the torch custom library op.
    This must be a function or module impl that provides the decompositions, and
    not an instance of the custom op.

    TODO: Better error handling, or can we make this an instance of the op as a param?

    For jit params and troubleshooting see:
    https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html
    """

    @jaten.op(*ops)
    def _composite_impl(*args: Any) -> Any:
        class ImplWrapper(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, *args: Any) -> Any:
                return impl(*args)

        # Note: avoid refactoring to share code with register_jaxpr_composite.
        # The `extract_jax` call must live in the `@jaten.op` handler. If called
        # outside of the handler, we would build the jaxpr representation of the
        # module once during registration, potentially missing op registrations that
        # come after. I.e. may miss nested abstractions if we build jaxpr AoT.
        state, jfn = recir.torchax.extract_jax(ImplWrapper())  # pyre-ignore[16]

        def jaxpr_impl(*args: Any) -> Any:  # pyre-ignore[53]
            return jfn(state, (*args,))

        return _jit_composite_impl(composite_name, jaxpr_impl, **jit_args)(*args)
