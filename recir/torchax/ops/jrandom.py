# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""torchax JAX lowering of ``recir::bernoulli_keyed``."""

import logging
import math
import os
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

# Defines torch.ops.recir.bernoulli_keyed (schema + fallbacks) before registering.
import recir.ops.random_op  # noqa: F401
import torch
from jax.sharding import Mesh, PartitionSpec as P
from recir.torchax.gspmd_context import current_gspmd_mesh_context
from recir.torchax.ops.decorators import op

logger: logging.Logger = logging.getLogger(__name__)


def _hardware_bernoulli_mask(key: Any, shape: tuple[int, ...], p: Any) -> Any:
    """Generate per-chip RBG masks on the GSPMD leading batch axis."""

    context = current_gspmd_mesh_context()
    if context is not None:
        mesh = context.mesh
        resource = context.batch_axis_resource
        axes = (resource,) if isinstance(resource, str) else tuple(resource)
        chips = math.prod(int(mesh.shape[name]) for name in axes)
        axis = 0 if shape and shape[0] >= chips and shape[0] % chips == 0 else None
    else:
        axes = ("dropout_chips",)
        resource = axes[0]
        chips = jax.device_count()
        mesh = Mesh(np.asarray(jax.devices()), axes)
        axis = next(
            (i for i, n in enumerate(shape) if n >= chips and n % chips == 0), None
        )
    probability = jnp.asarray(p, dtype=jnp.float32)
    if chips <= 1:
        return jax.random.bernoulli(jax.random.fold_in(key, 0), probability, shape)
    if axis is None:
        logger.warning(
            "[Dropout hardware RNG] no shardable axis for shape=%s chips=%d; "
            "falling back to unsharded RBG draw",
            shape,
            chips,
        )
        return jax.random.bernoulli(key, probability, shape)
    local_shape = (*shape[:axis], shape[axis] // chips, *shape[axis + 1 :])
    spec = P(*(resource if i == axis else None for i in range(len(shape))))

    def per_chip(local_key: Any) -> Any:
        chip_index = jnp.uint32(0)
        for name in axes:
            chip_index = chip_index * mesh.shape[name] + jax.lax.axis_index(name)
        local_key = jax.random.fold_in(local_key, chip_index)
        return jax.random.bernoulli(local_key, probability, local_shape)

    return jax.shard_map(
        per_chip, mesh=mesh, in_specs=P(), out_specs=spec, check_vma=False
    )(key)


def _fold_counter(counter: Any) -> Any:
    """Reduce the step counter to the uint32 scalar ``fold_in`` requires."""
    folded = jnp.asarray(counter).reshape(()).astype(jnp.uint32)
    if folded.dtype != jnp.uint32:
        raise RuntimeError(
            f"dropout PRNG counter came out as {folded.dtype}, not uint32; "
            "jax.random.fold_in needs a uint32-convertible scalar"
        )
    return folded


@op(torch.ops.recir.bernoulli_keyed)  # pyre-ignore[56]
def _jbernoulli_keyed(
    self: Any,
    p: Any,
    counter: Any,
    site: Any,
    base_seed: Any,
) -> Any:
    impl = os.environ.get("RECIR_DROPOUT_PRNG_IMPL", "threefry2x32")
    if impl not in ("threefry2x32", "rbg"):
        raise ValueError(f"Unsupported dropout PRNG implementation: {impl}")
    if impl == "rbg":
        key = jax.random.key(base_seed, impl="rbg")
        key = jax.random.fold_in(key, _fold_counter(counter))
        key = jax.random.fold_in(key, site)
        logger.info(
            "[Dropout hardware RNG] impl=rbg site=%s shape=%s", site, self.shape
        )
        return _hardware_bernoulli_mask(key, self.shape, p).astype(self.dtype)
    key = jax.random.fold_in(jax.random.PRNGKey(base_seed), _fold_counter(counter))
    key = jax.random.fold_in(key, site)
    # Cast back to the caller's dtype: the consumer is ``noise.div_(p)`` then
    # ``input * noise``, so a bool mask would be wrong. torchax's InplaceOp gets
    # away with returning bool only because it lands in ``to_mutate.copy_(...)``
    # (recir/torchax/ops/op_base.py).
    return (jax.random.uniform(key, self.shape) < p).astype(self.dtype)
