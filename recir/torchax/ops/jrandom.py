# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""torchax JAX lowering of ``recir::bernoulli_keyed``."""

from typing import Any

import jax
import jax.numpy as jnp

# Defines torch.ops.recir.bernoulli_keyed (schema + fallbacks) before registering.
import recir.ops.random_op  # noqa: F401
import torch
from recir.torchax.ops.decorators import op


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
    key = jax.random.fold_in(jax.random.PRNGKey(base_seed), _fold_counter(counter))
    key = jax.random.fold_in(key, site)
    # Cast back to the caller's dtype: the consumer is ``noise.div_(p)`` then
    # ``input * noise``, so a bool mask would be wrong. torchax's InplaceOp gets
    # away with returning bool only because it lands in ``to_mutate.copy_(...)``
    # (recir/torchax/ops/op_base.py).
    return (jax.random.uniform(key, self.shape) < p).astype(self.dtype)
