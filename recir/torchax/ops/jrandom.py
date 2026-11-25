# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Torchax JAX lowerings for explicitly keyed random ops."""

import logging
import math
import os
from typing import Any, cast

import jax
import jax.numpy as jnp

# Defines torch.ops.recir.bernoulli_keyed (schema + fallbacks) before registering.
import recir.ops.random_op  # noqa: F401
import torch
from jax.sharding import NamedSharding, PartitionSpec as P
from recir.torchax.gspmd_context import (
    batch_mesh_layout,
    batch_shard_index,
    current_gspmd_mesh_context,
)
from recir.torchax.ops.decorators import op

logger: logging.Logger = logging.getLogger(__name__)


def _bernoulli(key: Any, p: Any, shape: tuple[int, ...]) -> Any:
    """``jax.random.bernoulli``, or with ``RECIR_DROPOUT_MASK_BITS=16`` two masks per
    32-bit RBG word.

    Halving the generated words halves rng-bit-generator time. The keep
    probability becomes ``round(p * 2**16) / 2**16`` (0.89999 for 0.9) and the
    masks differ from the 32-bit draw, so it is opt-in.
    """
    bits = os.environ.get("RECIR_DROPOUT_MASK_BITS", "32")
    if bits not in ("16", "32"):
        raise ValueError(f"Unsupported RECIR_DROPOUT_MASK_BITS: {bits}")
    if bits == "32":
        return jax.random.bernoulli(key, p, shape)
    if not shape or shape[0] % 2:
        logger.warning(
            "[Dropout hardware RNG] RECIR_DROPOUT_MASK_BITS=16 needs an even "
            "leading dimension; shape=%s draws 32-bit masks",
            shape,
        )
        return jax.random.bernoulli(key, p, shape)
    words = jax.random.bits(key, (shape[0] // 2, *shape[1:]), jnp.uint32)
    threshold = jnp.round(jnp.asarray(p, jnp.float32) * 65536).astype(jnp.uint32)
    return jnp.concatenate(
        ((words & 0xFFFF) < threshold, (words >> 16) < threshold), axis=0
    )


def _hardware_bernoulli_mask(key: Any, shape: tuple[int, ...], p: Any) -> Any:
    """Generate per-chip RBG masks on the GSPMD leading batch axis."""

    context = current_gspmd_mesh_context()
    mesh, resource, chips = batch_mesh_layout("dropout_chips")
    if context is not None:
        axis = 0 if shape and shape[0] >= chips and shape[0] % chips == 0 else None
    else:
        axis = next(
            (i for i, n in enumerate(shape) if n >= chips and n % chips == 0), None
        )
    probability = jnp.asarray(p, dtype=jnp.float32)
    if chips <= 1:
        return _bernoulli(jax.random.fold_in(key, 0), probability, shape)
    if axis is None:
        logger.warning(
            "[Dropout hardware RNG] no shardable axis for shape=%s chips=%d; "
            "falling back to unsharded RBG draw",
            shape,
            chips,
        )
        return _bernoulli(key, probability, shape)
    local_shape = (*shape[:axis], shape[axis] // chips, *shape[axis + 1 :])
    spec = P(*(resource if i == axis else None for i in range(len(shape))))

    def per_chip(local_key: Any) -> Any:
        chip_index = batch_shard_index(mesh, resource)
        local_key = jax.random.fold_in(local_key, chip_index)
        return _bernoulli(local_key, probability, local_shape)

    return jax.shard_map(
        per_chip, mesh=mesh, in_specs=P(), out_specs=spec, check_vma=False
    )(key)


def _hash32(x: Any) -> Any:
    """A bijective mix of uint32 values (the lowbias32 integer finalizer)."""
    x = x ^ (x >> 16)
    x = x * jnp.uint32(0x7FEB352D)
    x = x ^ (x >> 15)
    x = x * jnp.uint32(0x846CA68B)
    return x ^ (x >> 16)


def _hash_bernoulli_mask(key: Any, shape: tuple[int, ...], p: Any) -> Any:
    """Keep mask from a hash of each element's (row, column) position, the last
    dimension being the column, with 16-bit precision like
    ``RECIR_DROPOUT_MASK_BITS=16``.

    The mask is elementwise in its position, so XLA computes it inside the
    fusions that consume it instead of materializing random bits, and each chip
    computes only its own rows of the batch-sharded mask. Rows are hashed once
    and broadcast, so each element costs one hash.
    """
    shape = shape or (1,)
    if math.prod(shape[:-1]) >= 1 << 32:
        raise ValueError(f"Dropout mask rows overflow uint32: shape={shape}")
    row_shape = (*shape[:-1], 1)
    row = jnp.zeros(row_shape, jnp.uint32)
    stride = 1
    for axis in reversed(range(len(shape) - 1)):
        row = row + jax.lax.broadcasted_iota(jnp.uint32, row_shape, axis) * jnp.uint32(
            stride
        )
        stride *= shape[axis]
    column = jax.lax.broadcasted_iota(jnp.uint32, shape, len(shape) - 1)
    mesh, resource, chips = batch_mesh_layout("dropout_chips")
    if chips > 1 and shape[0] % chips == 0:
        spec = P(resource, *(None,) * (len(shape) - 1))
        row = jax.lax.with_sharding_constraint(row, NamedSharding(mesh, spec))
        column = jax.lax.with_sharding_constraint(column, NamedSharding(mesh, spec))
    bits = _hash32(_hash32(row ^ key) ^ column)
    threshold = jnp.round(jnp.asarray(p, jnp.float32) * 65536).astype(jnp.uint32)
    return (bits >> 16) < threshold


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
    if impl not in ("threefry2x32", "rbg", "hash"):
        raise ValueError(f"Unsupported dropout PRNG implementation: {impl}")
    if impl == "hash":
        key = _hash32(
            _hash32(jnp.uint32(base_seed % (1 << 32))) ^ _fold_counter(counter)
        )
        key = _hash32(key ^ jnp.uint32(site))
        mask = _hash_bernoulli_mask(key, tuple(self.shape), p)
        return mask.reshape(self.shape).astype(self.dtype)
    if os.environ.get("RECIR_DROPOUT_MASK_BITS", "32") == "16" and impl != "rbg":
        # Only the RBG path draws the paired masks, so letting this through
        # would measure a 32-bit draw and report it as the 16-bit one.
        raise ValueError(
            "RECIR_DROPOUT_MASK_BITS=16 needs RECIR_DROPOUT_PRNG_IMPL=rbg; "
            f"impl={impl} draws 32-bit masks"
        )
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


@op(torch.ops.recir.stochastic_round_fp16)
def _jstochastic_round_fp16(value: Any, counter: Any, seed: Any) -> Any:
    key = jax.random.fold_in(
        jax.random.PRNGKey(_fold_counter(seed)), _fold_counter(counter)
    )
    if "torch_dist" in jax.sharding.get_abstract_mesh().manual_axes:
        key = jax.random.fold_in(key, jax.lax.axis_index("torch_dist"))
    _, value_exponent = jnp.frexp(value)
    quantum = jnp.maximum(jnp.ldexp(jnp.ones_like(value), value_exponent - 11), 2**-24)
    nearest_f32 = jnp.rint(value / quantum) * quantum
    nearest_f32 = jnp.where(
        jnp.abs(nearest_f32) > 65504,
        jnp.copysign(jnp.inf, value),
        nearest_f32,
    )
    mantissa, exponent = jnp.frexp(jnp.abs(nearest_f32))
    toward_zero = jnp.abs(value) < jnp.abs(nearest_f32)
    exponent = exponent - ((mantissa == 0.5) & toward_zero).astype(jnp.int32)
    gap = jnp.where(
        nearest_f32 == 0,
        2**-24,
        jnp.maximum(jnp.ldexp(jnp.ones_like(value), exponent - 11), 2**-24),
    )
    neighbor = nearest_f32 + jnp.where(value > nearest_f32, gap, -gap)
    probability = jnp.abs(value - nearest_f32) / gap
    uniform = jax.random.uniform(key, value.shape, dtype=jnp.float32)
    use_neighbor = (
        (uniform < probability) & jnp.isfinite(value) & (jnp.abs(value) <= 65504)
    )
    return cast(jax.Array, jnp.where(use_neighbor, neighbor, nearest_f32)).astype(
        jnp.float16
    )
