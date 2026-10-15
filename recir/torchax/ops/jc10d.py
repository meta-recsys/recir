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


from typing import Any

import jax
import jax.numpy as jnp
import torch
from recir.torchax.ops.decorators import op


@op(torch.ops._c10d_functional.all_gather_into_tensor)  # pyre-ignore[56]
def _c10d_all_gather(input: Any, group_size: int, group_name: str) -> Any:
    # tiled=True concatenates the shards along dim 0 -> [group_size*k, ...],
    # matching torch all_gather_into_tensor semantics (FSDP Shard(0)->Replicate
    # uses gather_dim=0 with no follow-up reshape). tiled=False would prepend a
    # stacked [group_size, k, ...] axis and break the consuming matmul.
    return jax.lax.all_gather(input, "torch_dist", tiled=True)


@op(torch.ops._c10d_functional.reduce_scatter_tensor)  # pyre-ignore[56]
def _c10d_reduce_scatter(
    input: Any, reduceOp: str, group_size: int, group_name: str
) -> Any:
    # FSDP grad Partial(reduce)->Shard(0): reduce across "torch_dist" AND scatter
    # along dim 0. tiled=True divides dim 0 by group_size and keeps rank, matching
    # torch reduce_scatter_tensor ([N,...]->[N/group_size,...]). psum_scatter only
    # sums, so divide by group_size for the "avg" (DDP-style) reduction.
    res = jax.lax.psum_scatter(input, "torch_dist", scatter_dimension=0, tiled=True)
    if reduceOp == "avg":
        res = res / group_size
    elif reduceOp != "sum":
        raise RuntimeError(
            f"reduce_scatter reduceOp {reduceOp!r} not supported "
            "(only 'sum'/'avg'; psum_scatter reduces by sum)"
        )
    return res


@op(torch.ops._c10d_functional.all_reduce)  # pyre-ignore[56]
def _c10d_all_reduce(self: Any, reduceOp: str, group_name: str) -> Any:
    if reduceOp == "sum":
        res = jax.lax.psum(self, axis_name="torch_dist")
    elif reduceOp == "avg":
        res = jax.lax.pmean(self, axis_name="torch_dist")
    elif reduceOp == "min":
        res = jax.lax.pmin(self, axis_name="torch_dist")
    elif reduceOp == "max":
        res = jax.lax.pmax(self, axis_name="torch_dist")
    else:
        raise RuntimeError(f"Reduce op {reduceOp} not implemented")
    return res


@op(torch.ops._c10d_functional.broadcast)  # pyre-ignore[56]
def _c10d_broadcast(self: Any, src: int, group_name: str) -> Any:
    masked = jnp.where(
        jax.lax.axis_index("torch_dist") == src,
        self,
        jnp.zeros_like(self),
    )
    return jax.lax.psum(masked, "torch_dist")


@op(torch.ops._c10d_functional.wait_tensor)  # pyre-ignore[56]
def _c10d_wait_tensor(tensor: Any) -> Any:
    # Async tensor is aleady `wait`ed by dispatcher
    return tensor
