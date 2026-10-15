# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Torchax (torch->jax) lowering for recir debug ops (recir/ops/debug_op.py).

``recir::debug_ids`` prints the block-owner histogram of the id tensor.

Why a shard_map (not a top-level jax.debug.print): recir's op-splice drops
effects that sit at the top level of an ``@op`` lowering, so a plain
``jax.debug.print`` (even ordered) or an ``io_callback`` there gets removed. The
ring-shift kernel's own ``[RS_LOAD]`` print survives precisely because it lives
*inside* a ``jax.shard_map`` body. We mirror that: the histogram + print run
inside a ``shard_map`` over the ``"torch_dist"`` mesh, and we return the
shard_map output (identity) so the op is also load-bearing. ``x`` is unchanged.
"""

from typing import Any

import jax
import jax.numpy as jnp
import recir.ops.debug_op  # noqa: F401  (defines torch.ops.recir.debug_ids)
import torch
from jax.sharding import Mesh, PartitionSpec as P
from recir.torchax.ops.decorators import op

_AXIS = "torch_dist"


def _device_mesh() -> Mesh:
    # Matches recir's Mesh(_get_global_devices(), ("torch_dist",)).
    return Mesh(jax.devices(), (_AXIS,))


@op(torch.ops.recir.debug_ids)  # pyre-ignore[56]
def _jdebug_ids(x: Any, tag: str, v_local: int) -> Any:
    """Identity passthrough that prints the block-owner histogram of x.

    owner = x // v_local; histogram length = world_size (mesh size). The print
    runs inside a shard_map (like the ring-shift [RS_LOAD] print) so recir's
    op-splice does not drop it, and is emitted once from rank 0.
    """
    world_size = len(jax.devices())
    print(
        f"[JDEBUG_TRACE] _jdebug_ids lowering invoked: tag={tag} "
        f"shape={x.shape} v_local={v_local} world_size={world_size}",
        flush=True,
    )
    n_global = int(x.size)
    fmt = (
        "[IDS_DBG:"
        + tag
        + "] rank={r} n={n} min={mn} max={mx} block={blk} zero_frac={zf} "
        "blk_off8={bo} mod_off8={mo} head16={hd}"
    )

    def _per_device(xd: Any) -> Any:
        # v_local passed in is the GLOBAL vocab; per-owner block = vocab // W.
        block = max(v_local // world_size, 1)
        flat = xd.reshape(-1)
        blk_owner = (flat // block).astype(jnp.int32)
        mod_owner = (flat % world_size).astype(jnp.int32)
        blk_hist = jax.lax.psum(jnp.bincount(blk_owner, length=world_size), _AXIS)
        mod_hist = jax.lax.psum(jnp.bincount(mod_owner, length=world_size), _AXIS)
        n_zero = jax.lax.psum((flat == 0).sum(), _AXIS)
        g_min = jax.lax.pmin(flat.min(), _AXIS)
        g_max = jax.lax.pmax(flat.max(), _AXIS)
        rank = jax.lax.axis_index(_AXIS)
        # offset-mod-8 buckets: bucket j = mass on owners == j (mod 8). Ties the
        # skew to the period-8 (8 cores/host) hot-wave structure.
        blk_off8 = blk_hist.reshape(world_size // 8, 8).sum(axis=0)
        mod_off8 = mod_hist.reshape(world_size // 8, 8).sum(axis=0)
        jax.debug.print(
            fmt,
            r=rank,
            n=n_global,
            mn=g_min,
            mx=g_max,
            blk=block,
            zf=n_zero,
            bo=blk_off8,
            mo=mod_off8,
            hd=blk_hist[:16],
        )
        return xd

    return jax.shard_map(
        _per_device,
        mesh=_device_mesh(),
        in_specs=P(_AXIS, *([None] * (x.ndim - 1))),
        out_specs=P(_AXIS, *([None] * (x.ndim - 1))),
        check_vma=False,
    )(x)
