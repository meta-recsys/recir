# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""torchax JAX lowerings for the recir Pallas and Helion HSTU attention ops.

Importing this module registers, on the recir/torchax path:
  - torch.ops.recir.hstu_attention -> Pallas forward kernel
  - torch.ops.recir.hstu_attention_backward -> tiled Pallas backward kernel
  - torch.ops.recir.helion_ragged_attention_fwd -> forward Helion kernel (returns
    a single ``out`` array)
  - torch.ops.recir.helion_ragged_attention_bwd -> the three backward Helion
    kernels; a **functional** lowering returning ``(dq, dk, dv)``.

The ops themselves (schemas + CPU fallbacks + Meta kernels + autograd) are defined
in ``recir.ops.hstu_attention_op`` and ``recir.ops.helion_hstu_attention_op``;
this module force-imports both so the torch ops exist before the ``@op``
registration runs, and must itself be imported before any torchax Environment is
built (``load_ops`` snapshots ``all_aten_ops`` once). Mirrors
``recir/torchax/ops/jseq_embedding.py``.

The args the target-aware-causal SiLU kernels do not implement (``attn_bias``,
``dropout_pr``, ``max_attn_len``, ``contextual_seq_len``, ``full_attn_size``,
``num_softmax_heads``, and any non-causal ``invalid_attn_mask_type``) are not in
the op schema at all: ``_recir_pytorch_hstu_mha`` rejects them at the call site,
so a bad config fails at the first forward instead of deep inside the XLA lower.
That adapter also guarantees ``num_targets``/``attn_scale`` arrive as real
tensors, so there is nothing to normalize here.

Both lowerings run the kernel inside a **clean tensor env** (``no_dispatch()`` +
``unset_fake_temporarily()``) so Helion's ``Kernel.jax_fn`` -- which drives its
Pallas host wrapper with internal torch META tensors -- is not intercepted by the
active torchax modes ("SymIntArrayRef expected to contain only concrete integers").

On a multi-device mesh they delegate to the shard-local helpers
``sample_sharded_hstu_{fwd,bwd}_jax`` in ``recir.ops.helion.jagged_hstu_attention``,
which own the ``shard_map`` and the full sharding derivation (see the section
comment there). q/k/v arrive jagged and L-sharded and stay
``Shard(0) -> Shard(0)`` on L, so no all-gather / all-reduce is inserted. The clean-env
kernel runner is passed into the helper so the sharding math stays pure-JAX and
CPU-unit-testable. Requires BATCH-ALIGNED packing, which
``_assert_batch_aligned`` guards. The single-device branch runs the jagged kernel
directly on the full ``L`` with the global ``seq_offsets``.

Both lowerings are ``is_jax_function=True`` (default): they receive JAX arrays and
return JAX arrays. Neither mutates its inputs -- the forward op's
``register_autograd`` produced the separate backward node these two lower.
"""

from typing import Any

# Define the torch ops (schema + CPU fallback + Meta kernels) before registering.
import jax
import recir.ops.helion_hstu_attention_op  # noqa: F401
import recir.ops.hstu_attention_op  # noqa: F401
import recir.ops.pallas_jagged_hstu_attention_op  # noqa: F401
import torch
import torch.utils._mode_utils as mode_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from recir.ops.helion.jagged_hstu_attention import (
    hstu_ragged_attention_bwd_jax_fn,
    hstu_ragged_attention_fwd_jax_fn,
    sample_sharded_hstu_bwd_jax,
    sample_sharded_hstu_fwd_jax,
)
from recir.ops.pallas.hstu_attention import (
    pallas_hstu_attention,
    pallas_hstu_attention_backward,
)
from recir.ops.pallas.jagged_hstu_attention import (
    pallas_jagged_hstu_attention,
    pallas_jagged_hstu_attention_backward,
    sample_sharded_jagged_hstu_attention,
    sample_sharded_jagged_hstu_attention_backward,
)
from recir.torchax.ops.decorators import op
from torch._subclasses.fake_tensor import unset_fake_temporarily

# recir's GSPMD data-parallel mesh axis (see recir/lowering/xla_lower.py).
_AXIS = "torch_dist"


def _device_mesh() -> Any:
    # Matches recir's Mesh(jax.devices(), ("torch_dist",)); built from jax
    # directly to avoid a recir.lowering import cycle. Mirrors jseq_embedding.
    return Mesh(jax.devices(), (_AXIS,))


def _interpret_pallas() -> bool:
    # Pallas/Mosaic targets TPU; fall back to the interpreter on a CPU backend.
    try:
        return jax.default_backend() == "cpu"
    except Exception:  # noqa: BLE001 -- default to native TPU path on probe error
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
    interpret = _interpret_pallas()

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
    # inside a shard_map: q/k/v and per-example lengths/num_targets are sharded
    # along their batch axis on the data-parallel mesh axis.
    batch = q.shape[0]
    if batch % world_size != 0:
        raise ValueError(
            f"hstu_attention batch {batch} not divisible by world_size {world_size}"
        )
    qkv_spec = P(_AXIS, None, None, None)
    len_spec = P(_AXIS)
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
) -> tuple[Any, Any, Any]:
    interpret = _interpret_pallas()

    def _bwd(grad_out_, q_, k_, v_, lengths_, num_targets_):
        return pallas_hstu_attention_backward(
            grad_out_,
            q_,
            k_,
            v_,
            lengths_,
            num_targets_,
            alpha,
            inv_n,
            interpret=interpret,
        )

    world_size = len(jax.devices())
    if world_size <= 1:
        return _bwd(grad_out, q, k, v, lengths, num_targets)

    b = q.shape[0]
    if b % world_size != 0:
        raise ValueError(
            f"hstu_attention batch {b} not divisible by world_size {world_size}"
        )
    qkv_spec = P(_AXIS, None, None, None)
    len_spec = P(_AXIS)
    mapped_bwd: Any = jax.shard_map(
        _bwd,
        mesh=_device_mesh(),
        in_specs=(qkv_spec, qkv_spec, qkv_spec, qkv_spec, len_spec, len_spec),
        out_specs=(qkv_spec, qkv_spec, qkv_spec),
        check_vma=False,
    )
    return mapped_bwd(grad_out, q, k, v, lengths, num_targets)


@op(torch.ops.recir.pallas_jagged_hstu_attention_fwd)  # pyre-ignore[56]
def _jpallas_jagged_hstu_attention_fwd(
    q: Any,
    k: Any,
    v: Any,
    seq_offsets: Any,
    num_targets: Any,
    attn_scale: Any,
    alpha: float,
    N: int,
) -> Any:
    interpret = _interpret_pallas()
    world_size = len(jax.devices())
    if world_size <= 1:
        return pallas_jagged_hstu_attention(
            q,
            k,
            v,
            seq_offsets,
            num_targets,
            max_seq_len=N,
            scale=alpha,
            inv_n=attn_scale,
            interpret=interpret,
        )

    mesh = _device_mesh()
    output = sample_sharded_jagged_hstu_attention(
        q,
        k,
        v,
        seq_offsets,
        num_targets,
        attn_scale,
        max_seq_len=N,
        alpha=alpha,
        world_size=world_size,
        mesh=mesh,
        interpret=interpret,
    )
    return jax.lax.with_sharding_constraint(
        output,
        NamedSharding(mesh, P(_AXIS, None, None, None)),
    )


@op(torch.ops.recir.pallas_jagged_hstu_attention_bwd)  # pyre-ignore[56]
def _jpallas_jagged_hstu_attention_bwd(
    grad_out: Any,
    q: Any,
    k: Any,
    v: Any,
    seq_offsets: Any,
    num_targets: Any,
    attn_scale: Any,
    alpha: float,
    N: int,
) -> tuple[Any, Any, Any]:
    interpret = _interpret_pallas()
    world_size = len(jax.devices())
    if world_size <= 1:
        return pallas_jagged_hstu_attention_backward(
            grad_out,
            q,
            k,
            v,
            seq_offsets,
            num_targets,
            attn_scale,
            max_seq_len=N,
            alpha=alpha,
            interpret=interpret,
        )

    mesh = _device_mesh()
    dq, dk, dv = sample_sharded_jagged_hstu_attention_backward(
        grad_out,
        q,
        k,
        v,
        seq_offsets,
        num_targets,
        attn_scale,
        max_seq_len=N,
        alpha=alpha,
        world_size=world_size,
        mesh=mesh,
        interpret=interpret,
    )
    packed_sharding = NamedSharding(mesh, P(_AXIS, None, None))
    return (
        jax.lax.with_sharding_constraint(dq, packed_sharding),
        jax.lax.with_sharding_constraint(dk, packed_sharding),
        jax.lax.with_sharding_constraint(dv, packed_sharding),
    )


@op(torch.ops.recir.helion_ragged_attention_fwd)  # pyre-ignore[56]
def _jhelion_ragged_attention_fwd(
    q: Any,
    k: Any,
    v: Any,
    seq_offsets: Any,
    num_targets: Any,
    attn_scale: Any,
    alpha: float,
    N: int,
) -> Any:
    batch = int(seq_offsets.shape[0]) - 1

    def _run_fwd(q_: Any, k_: Any, v_: Any, so_: Any, nt_: Any, sc_: Any) -> Any:
        # Run the Helion forward kernel in a clean tensor env (see module docstring).
        # Inputs are jagged ``[L, H, D]`` + ``[B+1]`` seq_offsets.
        with unset_fake_temporarily(), mode_utils.no_dispatch():
            out, _m = hstu_ragged_attention_fwd_jax_fn(
                N, alpha, q_, k_, v_, so_, nt_, sc_
            )
            return out

    world_size = len(jax.devices())
    if world_size > 1:
        # Shard-local sample-sharding: q/k/v stay jagged + L-sharded; the helper runs
        # the jagged kernel per-device INSIDE the shard_map (rebuilds local
        # seq_offsets), so the op stays Shard(0)->Shard(0) on L (no all-gather/
        # all-reduce; see the helper + module docstring). Requires batch-aligned
        # packing; B % W == 0 follows.
        assert batch % world_size == 0, (
            f"jagged HSTU sample-shard: num_sequences {batch} not divisible by "
            f"world_size {world_size}"
        )

        mesh = _device_mesh()
        out = sample_sharded_hstu_fwd_jax(
            N,
            alpha,
            q,
            k,
            v,
            seq_offsets,
            num_targets,
            attn_scale,
            world_size,
            mesh,
            run_kernel=_run_fwd,
        )
        # `out` is already L-sharded by the shard_map's out_specs; re-pin the same
        # spec so batch_dim propagation sees it explicitly (mirrors jseq_embedding).
        out = jax.lax.with_sharding_constraint(
            out, NamedSharding(mesh, P(_AXIS, None, None))
        )
    else:
        # Single device: run the jagged kernel directly on the global ``[L, H, D]``
        # with the global ``seq_offsets`` (no densify).
        #
        # No PRT pad-row drop here: ``_drop_pad_rows`` lives inside the sharded
        # helper's ``shard_map``, and this branch is the CPU / unit-test path where
        # the caller passes real unpadded offsets. A single-device run against a
        # PRT-padded batch is not a supported configuration.
        out = _run_fwd(q, k, v, seq_offsets, num_targets, attn_scale)

    return out


@op(torch.ops.recir.helion_ragged_attention_bwd)  # pyre-ignore[56]
def _jhelion_ragged_attention_bwd(
    grad_out: Any,
    q: Any,
    k: Any,
    v: Any,
    seq_offsets: Any,
    num_targets: Any,
    attn_scale: Any,
    alpha: float,
    N: int,
) -> tuple[Any, Any, Any]:
    batch = int(seq_offsets.shape[0]) - 1

    def _run_bwd(
        do_: Any, q_: Any, k_: Any, v_: Any, so_: Any, nt_: Any, sc_: Any
    ) -> tuple[Any, Any, Any]:
        # Run the three Helion backward kernels in a clean tensor env.
        # Inputs are jagged ``[L, H, D]`` + ``[B+1]`` seq_offsets.
        with unset_fake_temporarily(), mode_utils.no_dispatch():
            return hstu_ragged_attention_bwd_jax_fn(
                do_, q_, k_, v_, so_, nt_, sc_, N, alpha
            )

    world_size = len(jax.devices())
    if world_size > 1:
        # Shard-local sample-sharding (see the forward + module docstring):
        # grad_out/q/k/v stay jagged + L-sharded; the helper runs the jagged backward
        # per-device INSIDE the shard_map (rebuilds local seq_offsets;
        # Shard(0)->Shard(0) on L, no all-reduce/scatter-add). Requires batch-aligned
        # packing.
        assert batch % world_size == 0, (
            f"jagged HSTU sample-shard: num_sequences {batch} not divisible by "
            f"world_size {world_size}"
        )

        mesh = _device_mesh()
        gq, gk, gv = sample_sharded_hstu_bwd_jax(
            grad_out,
            q,
            k,
            v,
            seq_offsets,
            num_targets,
            attn_scale,
            N,
            alpha,
            world_size,
            mesh,
            run_kernel=_run_bwd,
        )
        # Same re-pin as the forward: the grads are already L-sharded by out_specs.
        spec = NamedSharding(mesh, P(_AXIS, None, None))
        gq = jax.lax.with_sharding_constraint(gq, spec)
        gk = jax.lax.with_sharding_constraint(gk, spec)
        gv = jax.lax.with_sharding_constraint(gv, spec)
    else:
        # Single device: run the jagged backward directly on the global ``[L, H, D]``
        # with the global ``seq_offsets`` (no densify). No PRT pad-row drop -- same
        # CPU / unit-test-path caveat as the forward; see the note there.
        gq, gk, gv = _run_bwd(grad_out, q, k, v, seq_offsets, num_targets, attn_scale)

    return gq, gk, gv
