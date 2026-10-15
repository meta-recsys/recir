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


"""Shared decorator for registering torch operator implementations."""

from collections.abc import Callable
from typing import Any

from recir.torchax.ops import ops_registry


def op(*aten: Any, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a function as the JAX implementation for one or more ATen operators."""

    def inner(func: Callable[..., Any]) -> Callable[..., Any]:
        for a in aten:
            ops_registry.register_torch_dispatch_op(a, func, **kwargs)
        return func

    return inner
