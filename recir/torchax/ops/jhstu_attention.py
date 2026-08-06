# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""torchax JAX lowering of ``recir::hstu_attention`` to the Pallas HSTU kernel.

Importing this module registers, on the recir/torchax path:
  - torch.ops.recir.hstu_attention          -> pallas_hstu_attention (Pallas fwd)
  - torch.ops.recir.hstu_attention_backward -> hstu_attention_reference VJP (q/k/v grads)

The op (schema + CPU fallback + autograd) is defined in
``recir.ops.hstu_attention_op``. On a CPU JAX backend the Pallas kernel runs in
``interpret`` mode (Mosaic targets TPU); on TPU it runs natively. This is the
seam the hammer ``HammerKernel`` dispatch calls (see Diff 3).
"""

from typing import Any, Tuple

import jax

# Define torch.ops.recir.hstu_attention (schema + fallbacks) before registering.
import recir.ops.hstu_attention_op  # noqa: F401
import torch
from jax.sharding import Mesh, PartitionSpec as P
from recir.ops.pallas.hstu_attention import (
    hstu_attention_reference,
    pallas_hstu_attention,
)
from recir.torchax.ops.decorators import op

# Data-parallel axis of recir's GSPMD mesh (Mesh(_get_global_devices(),
# ("torch_dist",))). q/k/v and their per-example lengths/num_targets are sharded
# along the batch axis on this axis.
_DP_AXIS = "torch_dist"


def _device_mesh() -> Any:
    return Mesh(jax.devices(), (_DP_AXIS,))


def _interpret() -> bool:
    # Pallas/Mosaic targets TPU; fall back to the interpreter on a CPU backend.
    try:
        return jax.default_backend() == "cpu"
    except Exception:  # noqa: BLE001 -- default to native TPU path on any probe error
        return False


@op(torch.ops.recir.hstu_attention)  # pyre-ignore[56]
def _jhstu_attention(
    q: Any,
    k: Any,
    v: Any,
    lengths: Any,
    num_targets: Any,
    alpha: float,
    inv_n: float,
) -> Any:
    interpret = _interpret()

    def _fwd(q_, k_, v_, lengths_, num_targets_):
        return pallas_hstu_attention(
            q_,
            k_,
            v_,
            lengths_,
            num_targets=num_targets_,
            scale=alpha,
            inv_n=inv_n,
            interpret=interpret,
        )

    # Single device (CPU interpret / tests): no partitioning needed, and a
    # degenerate 1-shard shard_map is avoided.
    world_size = len(jax.devices())
    if world_size <= 1:
        return _fwd(q, k, v, lengths, num_targets)

    # GSPMD cannot auto-partition a Mosaic/Pallas custom call, so run the kernel
    # inside a shard_map: the batch axis (axis 0) of q/k/v and of the per-example
    # lengths/num_targets is sharded on the data-parallel axis, and each device
    # runs the (B_local, H) Pallas grid entirely shard-local. lengths/num_targets
    # MUST be co-sharded with q/k/v -- the kernel indexes lengths[program_id(0)].
    b = q.shape[0]
    if b % world_size != 0:
        raise ValueError(
            f"hstu_attention batch {b} not divisible by world_size {world_size}"
        )
    qkv_spec = P(_DP_AXIS, None, None, None)
    len_spec = P(_DP_AXIS)
    mapped_fwd: Any = jax.shard_map(
        _fwd,
        mesh=_device_mesh(),
        in_specs=(qkv_spec, qkv_spec, qkv_spec, len_spec, len_spec),
        out_specs=qkv_spec,
        check_vma=False,
    )
    return mapped_fwd(q, k, v, lengths, num_targets)


@op(torch.ops.recir.hstu_attention_backward)  # pyre-ignore[56]
def _jhstu_attention_backward(
    grad_out: Any,
    q: Any,
    k: Any,
    v: Any,
    lengths: Any,
    num_targets: Any,
    alpha: float,
    inv_n: float,
) -> Tuple[Any, Any, Any]:
    _, vjp_fn = jax.vjp(
        lambda q_, k_, v_: hstu_attention_reference(
            q_, k_, v_, lengths, num_targets, alpha, inv_n
        ),
        q,
        k,
        v,
    )
    dq, dk, dv = vjp_fn(grad_out)
    return dq, dk, dv
